import os
import multiprocessing
import numpy as np
import torch
from typing import Union
from torch.utils.data import Dataset, DataLoader



# Data format
# raw event: np.ndarray, shape=(n_events, 4), dtype=np.int64, columns=(timestamp, x, y, polarity)
# event frame: np.ndarray, shape=(2, height, width), dtype=np.float32, channels=(positive, negative), each pixel is the sum of positive/negative events


def patchify_raw_events(events: np.ndarray, patch_size: int, sensor_size: tuple):
    """
    Divides raw event data into patches based on the specified patch size and sensor size.

    Args:
        events (np.ndarray): A numpy array of events with shape (n_events, 4) and columns (timestamp, x, y, polarity).
        patch_size (int): The size of each patch (assumed to be square).
        sensor_size (tuple): A tuple representing the size of the event camera (width, height).
    
    Returns:
        List[np.ndarray]: A list of numpy arrays, where each array contains the events in one patch.
    """
    sensor_width, sensor_height = sensor_size
    n_patches_x = sensor_width // patch_size
    n_patches_y = sensor_height // patch_size

    valid_events = events[
        (events[:, 1] >= 0) & (events[:, 1] < sensor_width) &
        (events[:, 2] >= 0) & (events[:, 2] < sensor_height)
    ]

    patch_x = valid_events[:, 1] // patch_size
    patch_y = valid_events[:, 2] // patch_size
    patch_idx = patch_y * n_patches_x + patch_x

    patches = [[] for _ in range(n_patches_x * n_patches_y)]

    for idx in np.unique(patch_idx):
        patches[idx] = valid_events[patch_idx == idx]
    
    return patches


def slice_raw_events_by_count(events: np.ndarray, count: int, stride: int = None):
    """
    Slice events of size (count, 4).
    
    Args:
        events (np.ndarray): Raw events to be split.
        count (int): The number of events in each slice.
        stride (int): The stride between slices. If None, stride is set to count.
    
    Returns:
        List[np.ndarray]: List of event slices.
    """

    if stride is None:
        stride = count

    n_chunks = (len(events) - count) // stride + 1
    chunks = [events[i * stride : i * stride + count] for i in range(n_chunks)]
    return chunks


def _raw_events_to_frame_numpy(events: np.ndarray, frame_size: tuple):
    """
    Converts raw events to a frame. NumPy version.

    Args:
        events (np.ndarray): Raw events to be converted.
        frame_size (tuple): The size of the frame (width, height).
    
    Returns:
        np.ndarray: Event frame of size (2, width, height).
    """
    width, height = frame_size
    frame = np.zeros((2, width, height), dtype=np.float32)
    x, y, polarity = events[:, 1], events[:, 2], events[:, 3]
    np.add.at(frame, (polarity, x, y), 1)
    return frame


def _raw_events_to_frame_torch(events: torch.Tensor, frame_size: tuple):
    """
    Converts raw events to a frame. Pytorch version.

    Args:
        events (torch.Tensor): Raw events to be converted, shape (n_events, 4), 
                               where each event is [timestamp, x, y, polarity].
        frame_size (tuple): The size of the frame (width, height).
    
    Returns:
        torch.Tensor: Event frame of size (2, width, height). The frame is stored on the same device as the input events.
    """

    width, height = frame_size
    device = events.device
    frame = torch.zeros(2 * width * height, dtype=torch.float32, device=device)
    x, y, polarity = events[:, 1], events[:, 2], events[:, 3]
    indices = polarity * width * height + x * height + y 
    frame.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32, device=device))
    frame = frame.reshape(2, width, height)
    return frame


def raw_events_to_frame(events: Union[np.ndarray, torch.Tensor], frame_size: tuple):
    """
    Converts raw events to a frame.

    Args:
        events (Union[np.ndarray, torch.Tensor]): Raw events to be converted.
        frame_size (tuple): The size of the frame (width, height).
    
    Returns:
        Union[np.ndarray, torch.Tensor]: Event frame of size (2, width, height).
    """
    if isinstance(events, np.ndarray):
        return _raw_events_to_frame_numpy(events, frame_size)
    elif isinstance(events, torch.Tensor):
        return _raw_events_to_frame_torch(events, frame_size)
    else:
        raise ValueError("Input type must be either numpy.ndarray or torch.Tensor.")


def raw_events_to_time_surface(events: np.ndarray, time_surface_size: tuple, tau: float):
    """
    Converts raw events to a time surface.

    Args:
        events (np.ndarray): Raw events to be converted, shape (n_events, 4), 
                             where each event is [timestamp, x, y, polarity].
        time_surface_size (tuple): The size of the time surface (width, height).
        tau (float): Time constant for the exponential decay.

    Returns:
        np.ndarray: Time surface of size (2, width, height).
    """
    width, height = time_surface_size
    time_surface = np.zeros(2 * width * height, dtype=np.float32)
    x, y, polarity, t = events[:, 1], events[:, 2], events[:, 3], events[:, 0]
    indices = polarity * width * height + x * height + y
    time_surface[indices] = t
    # time_surface = np.exp((time_surface - time_surface.max()) / tau)
    t_end = t[-1]
    diff = - (t_end - time_surface)
    time_surface = np.exp(diff / tau)
    time_surface = time_surface.reshape(2, width, height)
    return time_surface


class DVS128Gesture(Dataset):
    def __init__(self, root: str, train: bool, count: int, stride: int, patch_size: int):
        self.root = root
        self.train = train
        self.count = count
        self.stride = stride
        self.patch_size = patch_size
        self.sensor_size = self.get_sensor_size()

        self.data_path = os.path.join(root, "patch{}_count{}_stride{}".format(patch_size, count, stride), "train" if train else "test")
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print("Mkdir:", self.data_path)
            self.generate_data()
        
        self.data_subdirs = [os.path.join(self.data_path, dir) for dir in os.listdir(self.data_path)]
        self.data_files = []
        self.labels = []
        for idx, data_subdir in enumerate(self.data_subdirs):
            for file in os.listdir(data_subdir):
                self.data_files.append(os.path.join(data_subdir, file))
                self.labels.append(idx)
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx], allow_pickle=False)
        label = self.labels[idx]
        return data, label
    
    @staticmethod
    def get_sensor_size():
        return (128, 128)
    
    def generate_data(self):
        
        raw_data_path = os.path.join(self.root, "events_np", "train" if self.train else "test")
        data_path = self.data_path
        
        raw_data_subdirs = []
        data_subdirs = []
        for dir in os.listdir(raw_data_path):
            raw_data_subdirs.append(os.path.join(raw_data_path, dir))
            data_subdir = os.path.join(data_path, dir)
            if not os.path.exists(data_subdir):
                os.makedirs(data_subdir)
                print("Mkdir:", data_subdir)
            data_subdirs.append(data_subdir)
                
        # save the patchified data in a new dir with same structure as raw data
        for raw_data_subdir, data_subdir in zip(raw_data_subdirs, data_subdirs):
            for file in os.listdir(raw_data_subdir):
                events = self.load_file(os.path.join(raw_data_subdir, file))
                patches = patchify_raw_events(events, self.patch_size, self.sensor_size)
                for idx_patch, patch in enumerate(patches):
                    slices = slice_raw_events_by_count(patch, self.count, self.stride)
                    for idx_slice, slice in enumerate(slices):
                        np.save(os.path.join(data_subdir, "{}_{}_{}.npy".format(file, idx_patch, idx_slice)), slice)
                print("Processed:", os.path.join(raw_data_subdir, file))
        

    def load_file(self, file_path):
        """
        Loads event data from a file.

        Args:
            file_path (str): Path to the file containing event data.

        Returns:
            np.ndarray: A numpy array of events with shape (n_events, 4) and columns (timestamp, x, y, polarity).
        """
        data = np.load(file_path, allow_pickle=False)  # data = {'t': t, 'x': x, 'y': y, 'p': p}
        t = data['t'].astype(int)
        x = data['x'].astype(int)
        y = data['y'].astype(int)
        p = data['p'].astype(int) 
        events = np.column_stack((t, x, y, p))  # events.shape = (n_events, 4)
        return events

if __name__ == "__main__":
    root = "/home/haohq/datasets/DVS128Gesture"
    dataset = DVS128Gesture(root=root, train=True, count=1000, stride=1000, patch_size=32)
    print("Number of samples:", len(dataset))
    data, label = dataset[0]
    print("Data shape:", data.shape)
    print("Label:", label)
    print("Finished.")

