import os
import numpy as np
import torch
from multiprocessing import Pool
from typing import Union, List
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler


# ############################################################################################################
# Data format:
# raw event: np.ndarray, shape=(n_events, 4), dtype=np.int64, columns=(timestamp, x, y, polarity)
# event frame: np.ndarray, shape=(2, height, width), dtype=np.float32, channels=(positive, negative)
# time surface: np.ndarray, shape=(2, height, width), dtype=np.float32, channels=(positive, negative)
# sensor size / frame size / time surface size: tuple, (height, width)
# ############################################################################################################
# directory structure of the dataset:
# root
# ├── events_np
# ├── patch_size_count_stride(_inverse)
# │   ├── data_path (e.g. train/test/..)
# │   │   ├── subdir0
# │   │   │   ├── data_file_path: {raw_file_name}_{patch_idx}_{slice_idx}.npy
# │   │   │   ├── ...
# │   │   ├── subdir1
# │   │   ├── ...
# │   ├── ...
# ├── ...
# ############################################################################################################


def patchify_raw_events(events: np.ndarray, patch_size: int, sensor_size: tuple)-> List[np.ndarray]:
    """
    Divides raw event data into patches based on the patch size and sensor size.

    Args:
        events (np.ndarray): A numpy array of events with shape (n_events, 4) and columns (timestamp, x, y, polarity).
        patch_size (int): The size of patches (square).
        sensor_size (tuple): The size of the event camera (height, width).
    
    Returns:
        List[np.ndarray]: A list of numpy arrays, where each array contains the events in one patch.
    """
    sensor_height, sensor_width = sensor_size
    n_patches_x = sensor_width // patch_size
    n_patches_y = sensor_height // patch_size

    valid_events = events[
        (events[:, 1] >= 0) & (events[:, 1] < sensor_width) &
        (events[:, 2] >= 0) & (events[:, 2] < sensor_height)
    ]

    patch_indices_x = valid_events[:, 1] // patch_size
    patch_indices_y = valid_events[:, 2] // patch_size
    patch_indices = patch_indices_y * n_patches_x + patch_indices_x

    valid_events[:, 1] = valid_events[:, 1] % patch_size
    valid_events[:, 2] = valid_events[:, 2] % patch_size

    patches = [[] for _ in range(n_patches_x * n_patches_y)]

    for idx in np.unique(patch_indices):
        patches[idx] = valid_events[patch_indices == idx]
    
    return patches


def slice_raw_events_by_count(events: np.ndarray, count: int, stride: int, inverse: bool = False) -> List[np.ndarray]:
    """
    Slice events into size `count`, with optional `stride`.
    
    Args:
        events (np.ndarray): Raw events to be split.
        count (int): The number of events in each slice.
        stride (int): The stride between slices. If 'stride' is 0, only return the first slice.
        inverse (bool): If True, the slices are taken from the end of the events.

    Returns:
        List[np.ndarray]: List of event slices.
    """
    nevents = len(events)

    n_slices = 1
    if (stride == 0) and (nevents < count):
            return []
        
    if stride > 0:
        n_slices = (len(events) - count) // stride + 1

    if not inverse:
        slices = [events[i * stride : i * stride + count] for i in range(n_slices)]
    else:
        slices = [events[nevents - i * stride - count: nevents - i * stride] for i in range(n_slices)]
    return slices


def _raw_events_to_frame_numpy(events: np.ndarray, frame_size: tuple) -> np.ndarray:
    """
    Converts raw events to an event frame (NumPy implementation).

    Args:
        events (np.ndarray): Raw events to be converted.
        frame_size (tuple): The size of the frame (height, width).
    
    Returns:
        np.ndarray: Event frame of size (2, height, width).
    """
    if len(events.shape) != 2:
        raise NotImplementedError("Batch dimension is not supported in the NumPy version.")
    
    height, width = frame_size
    frame = np.zeros((2, height, width), dtype=np.float32)

    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    polarity = events[:, 3].astype(np.int32)

    np.add.at(frame, (polarity, y, x), 1)
    return frame


def _raw_events_to_frame_torch(events: torch.Tensor, frame_size: tuple) -> torch.Tensor:
    """
    Converts raw events to an event frame (PyTorch implementation).

    Args:
        events (torch.Tensor): Raw events to be converted, shape (n_events, 4) or (batch_size, n_events, 4),
                               where each event is [timestamp, x, y, polarity].
        frame_size (tuple): The size of the frame (height, width).
    
    Returns:
        torch.Tensor: Event frame of size (2, height, width) or (batch_size, 2, height, width).
        The frame is stored on the same device as the input events.
    """
    use_batched_operate = True
    if events.dim() == 2:
        events = events.unsqueeze(0)    # [n_events, 4] -> [1, n_events, 4]
        use_batched_operate = False

    device = events.device
    batch_size, n_events = events.shape[:2]
    height, width = frame_size

    frame = torch.zeros(batch_size, 2 * height * width, dtype=torch.float32, device=device)
    x = events[:, :, 1].long()
    y = events[:, :, 2].long()
    polarity = events[:, :, 3].long()

    indices = polarity * width * height + y * width + x    # [batch_size, n_events] 
    values = torch.ones_like(indices, dtype=torch.float32, device=device)
    frame.scatter_add_(dim=1, index=indices, src=values)
    frame = frame.reshape(batch_size, 2, height, width)

    if not use_batched_operate:
        frame = frame.squeeze(0)

    return frame


def raw_events_to_frame(events: Union[np.ndarray, torch.Tensor], frame_size: tuple) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts raw events to an event frame.

    Args:
        events (Union[np.ndarray, torch.Tensor]): Raw events to be converted.
        frame_size (tuple): The size of the frame (height, width).
    
    Returns:
        Union[np.ndarray, torch.Tensor]: Event frame of size (2, height, width).
    """
    if isinstance(events, np.ndarray):
        return _raw_events_to_frame_numpy(events, frame_size)
    elif isinstance(events, torch.Tensor):
        return _raw_events_to_frame_torch(events, frame_size)
    else:
        raise ValueError("Input type must be either numpy.ndarray or torch.Tensor.")


def _raw_events_to_time_surface_numpy(events: np.ndarray, time_surface_size: tuple, tau: float) -> np.ndarray:
    """
    Converts raw events to a time surface (NumPy implementation).

    Args:
        events (np.ndarray): Raw events to be converted, shape (n_events, 4), 
                             where each event is [timestamp, x, y, polarity].
        time_surface_size (tuple): The size of the time surface (height, width).
        tau (float): Time constant for the exponential decay.

    Returns:
        np.ndarray: Time surface of size (2, height, width).
    """
    if len(events.shape) != 2:
        raise NotImplementedError("Batch dimension is not supported in the NumPy version.")

    height, width = time_surface_size
    time_surface = np.zeros(2 * height * width, dtype=np.float32)
    x = events[:, 1].astype(np.int32)
    y = events[:, 2].astype(np.int32)
    polarity = events[:, 3].astype(np.int32)
    t = events[:, 0].astype(np.float32)

    indices = polarity * height * width + y * width + x
    time_surface[indices] = t

    # time_surface = np.exp((time_surface - time_surface.max()) / tau)
    t_end = t[-1]
    diff = - (t_end - time_surface)
    time_surface = np.exp(diff / tau)
    time_surface = time_surface.reshape(2, height, width)
    return time_surface


def _raw_events_to_time_surface_torch(events: torch.Tensor, time_surface_size: tuple, tau: float) -> torch.Tensor:
    """
    Converts raw events to a time surface (PyTorch implementation).

    Args:
        events (torch.Tensor): Raw events to be converted, shape (n_events, 4) or (batch_size, n_events, 4),
                               where each event is [timestamp, x, y, polarity].
        time_surface_size (tuple): The size of the time surface (height, width).
        tau (float): Time constant for the exponential decay.

    Returns:
        torch.Tensor: Time surface of size (2, height, width) or (batch_size, 2, height, width).
    """
    use_batched_operate = True
    if events.dim() == 2:
        events = events.unsqueeze(0)    # [n_events, 4] -> [1, n_events, 4]
        use_batched_operate = False

    device = events.device
    batch_size, n_events = events.shape[:2]
    height, width = time_surface_size

    time_surface = torch.zeros(batch_size, 2 * height * width, dtype=torch.float32, device=device)
    x = events[:, :, 1].long()
    y = events[:, :, 2].long()
    polarity = events[:, :, 3].long()
    t = events[:, :, 0].float()

    indices = polarity * height * width + y * width + x    # [batch_size, n_events]

    # Update time_surface with the latest timestamps
    time_surface.scatter_(dim=1, index=indices, src=t)

    # Compute the exponential decay
    t_end = t[:, -1].unsqueeze(1)  # [batch_size, 1]
    diff = -(t_end - time_surface)  # [batch_size, 2 * height * width]
    time_surface = torch.exp(diff / tau)
    time_surface = time_surface.reshape(batch_size, 2, height, width)

    if not use_batched_operate:
        time_surface = time_surface.squeeze(0)
    
    return time_surface


def raw_events_to_time_surface(events: Union[np.ndarray, torch.Tensor], time_surface_size: tuple, tau: float) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts raw events to a time surface.

    Args:
        events (Union[np.ndarray, torch.Tensor]): Raw events to be converted.
        time_surface_size (tuple): The size of the time surface (height, width).
        tau (float): Time constant for the exponential decay.
    
    Returns:
        Union[np.ndarray, torch.Tensor]: Time surface of size (2, height, width).
    """
    if isinstance(events, np.ndarray):
        return _raw_events_to_time_surface_numpy(events, time_surface_size, tau)
    elif isinstance(events, torch.Tensor):
        return _raw_events_to_time_surface_torch(events, time_surface_size, tau)
    else:
        raise ValueError("Input type must be either numpy.ndarray or torch.Tensor.")


def process_raw_events(args):
    """
    Processes a single event data file: loads events, applies patchification and slicing, and saves the result.

    Args:
        args (tuple): A tuple containing all necessary arguments.
    """
    load_fn, raw_file_path, data_subdir, patch_size, sensor_size, count, stride, inverse = args

    # Load events
    events = load_fn(raw_file_path)

    # Process events
    patches = patchify_raw_events(events, patch_size, sensor_size)
    for idx_patch, patch in enumerate(patches):
        slices = slice_raw_events_by_count(patch, count, stride, inverse)
        for idx_slice, event_slice in enumerate(slices):
            save_path = os.path.join(
                data_subdir,
                "{}_{}_{}.npy".format(
                    os.path.splitext(os.path.basename(raw_file_path))[0],
                    idx_patch, 
                    idx_slice
                )
            )
            np.save(save_path, event_slice)
    print("Processed: [{}]".format(raw_file_path))



class DVS128Gesture(Dataset):
    """
    A PyTorch Dataset class for the DVS128 Gesture dataset.
        root (str): Root directory of the dataset.
        train (bool): If True, creates dataset from training set, otherwise from test set.
        count (int): Number of events in each slice.
        stride (int): Stride for slicing events.
        patch_size (int): Size of the patches to be extracted from the raw event data.
        inverse (bool): If True, the slices are taken from the end of the events. Default: False.
    Attributes:
        root (str): Root directory of the dataset.
        train (bool): If True, creates dataset from training set, otherwise from test set.
        count (int): Number of events in each slice.
        stride (int): Stride for slicing events.
        inverse (bool): If True, the slices are taken from the end of the events.
        patch_size (int): Size of the patches to be extracted from the raw event data.
        sensor_size (tuple): Size of the sensor (128, 128).
        data_path (str): Path to the processed data.
        data_subdirs (list): List of subdirectories in the data path.
        data_file_pathes (list): List of paths to the data files.
        labels (list): List of labels.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the data and its index.
        get_sensor_size(): Returns the size of the sensor.
        generate_data(): Generates the data by processing raw event data.
        load_file(file_path): Loads event data from a file.
    """
    def __init__(self, root: str, train: bool, count: int, stride: int, patch_size: int, inverse: bool = False):
        self.root = root
        self.train = train
        self.count = count
        self.stride = stride
        self.inverse = inverse
        self.patch_size = patch_size
        self.sensor_size = self.get_sensor_size()

        self.data_path = os.path.join(
            root,
            "patch{}_count{}_stride{}".format(patch_size, count, stride) + "_inverse" if inverse else "",
            "train" if train else "test"
        )

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"Make dir: [{self.data_path}]")
            self.generate_data()
        
        self.data_subdirs = [os.path.join(self.data_path, dir) for dir in os.listdir(self.data_path)]
        self.data_file_pathes = []
        self.labels = []
        
        for idx, data_subdir in enumerate(self.data_subdirs):
            for filename in os.listdir(data_subdir):
                self.data_file_pathes.append(os.path.join(data_subdir, filename))
                self.labels.append(idx)

    def __len__(self):
        return len(self.data_file_pathes)
    
    def __getitem__(self, idx):
        data = np.load(self.data_file_pathes[idx], allow_pickle=False)
        return data, idx
    
    @staticmethod
    def get_sensor_size():
        """
        Get the size of the sensor (width, height).
        """
        return (128, 128)
    
    def generate_data(self):
        
        raw_data_path = os.path.join(
            self.root,
            "events_np",
            "train" if self.train else "test"
        )
        data_path = self.data_path
        
        raw_data_subdirs = []
        data_subdirs = []
        for dirname in os.listdir(raw_data_path):
            raw_data_subdir = os.path.join(raw_data_path, dirname)
            data_subdir = os.path.join(data_path, dirname)
            if not os.path.exists(data_subdir):
                os.makedirs(data_subdir)
                print(f"Make dir: [{data_subdir}]")
            raw_data_subdirs.append(raw_data_subdir)
            data_subdirs.append(data_subdir)
                
        # Collect list of raw event files to process
        raw_event_file_list = []
        for raw_data_subdir, data_subdir in zip(raw_data_subdirs, data_subdirs):
            for file in os.listdir(raw_data_subdir):
                raw_file_path = os.path.join(raw_data_subdir, file)
                args = (
                    self.load_file,
                    raw_file_path,
                    data_subdir,
                    self.patch_size,
                    self.sensor_size,
                    self.count,
                    self.stride,
                    self.inverse,
                )
                raw_event_file_list.append(args)
                # process_raw_events(args)
        
        num_workers = os.cpu_count() or 1
        with Pool(processes=num_workers) as pool:
            pool.map(process_raw_events, raw_event_file_list)

        
    @ staticmethod
    def load_file(file_path: str) -> np.ndarray:
        """
        Loads event data from a file.

        Args:
            file_path (str): Path to the event data file.

        Returns:
            np.ndarray: A numpy array of events with shape (n_events, 4) and columns (timestamp, x, y, polarity).
        """
        data = np.load(file_path, allow_pickle=False)  # data = {'t': timestamp, 'x': x, 'y': y, 'p': polarity}
        t = data['t'].astype(int)
        x = data['x'].astype(int)
        y = data['y'].astype(int)
        p = data['p'].astype(int) 
        events = np.column_stack((t, x, y, p))  # events.shape = (n_events, 4)
        return events


def split_dataset(dataset, train_split):
    nsamples_total = len(dataset)
    nsamples_train = int(nsamples_total * train_split)
    train_dataset, val_dataset = random_split(dataset, [nsamples_train, nsamples_total - nsamples_train])
    return train_dataset, val_dataset


def get_data_loader_for_pretraining(
    dataset_name: str, 
    root: str,
    patch_size: int, 
    count: int, 
    stride: int,
    inverse: bool, 
    batch_size: int, 
    shuffle: bool, 
    num_workers: int, 
    train_split: float,
    val_split: float,
    dist: bool,
):  
    """
    Get data loaders for pretraining.
    Args:
        dataset_name (str): Name of the dataset.
        root (str): Root directory of the dataset.
        patch_size (int): Size of the patches.
        count (int): Number of events in each slice.
        stride (int): Stride for slicing events.
        inverse (bool): If True, the slices are taken from the end of the event sequence.
        batch_size (int): Number of samples in each batch.
        shuffle (bool): If True, shuffle the data.
        num_workers (int): Number of workers for data loading.
        train_split (float): Ratio of training set.
        val_split (float): Ratio of validation set, only used to split training set, validation set and test set.
        dist (bool): If True, use DDP.
    Returns:
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
    """
    if dataset_name == 'dvs128gesture':
        train_dataset = DVS128Gesture(root=root, train=True, count=count, stride=stride, patch_size=patch_size, inverse=inverse)
        train_dataset, val_dataset = split_dataset(train_dataset, train_split=train_split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=shuffle)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
        return train_loader, val_loader
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader


class HiddenStateDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading unpatched hidden state data.
    Args:
        root (str): The root directory of data files.
    Attributes:
        root (str): The root directory of data files.
        data_files (list): A list of paths to unpatched hidden state data files.
        labels (list): A list of labels.
    Methods:
        __len__(): Returns the total number of data files.
        __getitem__(idx): Returns the data and label at the specified index.
    """
    def __init__(self, root):
        self.root = root
        self.data_file_pathes = []
        self.labels = []

        # Collect list of data files and labels
        data_subdirs = [os.path.join(self.root, dir) for dir in os.listdir(self.root)]
        for idx, data_subdir in enumerate(data_subdirs):
            for file in os.listdir(data_subdir):
                self.data_file_pathes.append(os.path.join(data_subdir, file))
                self.labels.append(idx)
    
    def __len__(self):
        return len(self.data_file_pathes)
    
    def __getitem__(self, idx):
        data = np.load(self.data_file_pathes[idx], allow_pickle=False)
        label = self.labels[idx]
        return data, label


def get_data_loader_list_for_caching_hidden_states(
    dataset_name: str,
    root: str,
    patch_size: int,
    count: int,
    batch_size: int,
    num_workers: int,
) -> list:
    datasets = []
    if dataset_name == 'dvs128gesture':
        datasets.append(
            DVS128Gesture(root, train=True, count=count, stride=0, patch_size=patch_size, inverse=True)
        )
        datasets.append(
            DVS128Gesture(root, train=False, count=count, stride=0, patch_size=patch_size, inverse=True)
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    data_loaders = []
    for dataset in datasets:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        data_loaders.append(data_loader)
    return data_loaders
    

def get_data_loader_for_training(
    dataset_name: str,
    root: str,
    train_split: float,
    val_split: float,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):

    if dataset_name == 'dvs128gesture':
        train_path = os.path.join(root, "train")
        test_path = os.path.join(root, "test")
        train_dataset = HiddenStateDataset(train_path)
        test_dataset = HiddenStateDataset(test_path)
        train_dataset, val_dataset = split_dataset(train_dataset, train_split=train_split)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader