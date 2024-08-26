import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# [batch_size, ctx_len, num_events]
# time = 500 ms
# temporal resolution = 1 ms
# ctx_len = time / temporal resolution = 1000
# patch_size = 16
# num_patches = height / patch_size * width / patch_size
# num_events = max_num_events

def convert_npz_to_patches_by_time(npz_file_path, npz_file_loader, patch_size, height, width, duration, resolution, max_num_events):
    """
    Convert an npz file containing event data into patches by time.

    Args:
        npz_file_path (str): The path to the npz file.
        npz_file_loader (function): A function to load the npz file and return the event data.
        patch_size (int): The size of each patch.
        height (int): The height of the image.
        width (int): The width of the image.
        duration (int): The duration of each chunk.
        resolution (int): The resolution of each group within a chunk.
        max_num_events (int): The maximum number of events in each group.

    Returns:
        numpy.ndarray: An array of patches, where each patch contains groups of events.

    """

    # load the npz file
    t, x, y, p = npz_file_loader(npz_file_path)
    print(f'Loaded [{npz_file_path}].')
    t = t - t[0]

    # prepare the patches
    num_chunks = t[-1] // duration
    num_patches = (height // patch_size) * (width // patch_size)
    ctx_len = duration // resolution

    # split the events into chunks
    chunks = []
    for idx_chunk in range(num_chunks):
        index_left = np.searchsorted(t, idx_chunk * duration, side='left')
        index_right = np.searchsorted(t, (idx_chunk + 1) * duration, side='left')
        t_ = t[index_left: index_right]
        p_ = p[index_left: index_right]
        x_ = x[index_left: index_right]
        y_ = y[index_left: index_right]

        t_ = t_ - t_[0]

        # split the chunked events into patches
        patches = []
        for idx_patch in range(num_patches):
            x_patch_start = idx_patch % (width // patch_size) * patch_size        # start x of the patch
            y_patch_start = idx_patch // (width // patch_size) * patch_size       # start y of the patch
            indices = (x_ >= x_patch_start) & (x_ < x_patch_start + patch_size) & (y_ >= y_patch_start) & (y_ < y_patch_start + patch_size)     # indices of the events in the patch
            t_patch = t_[indices]
            p_patch = p_[indices]
            x_patch = x_[indices] - x_patch_start
            y_patch = y_[indices] - y_patch_start
            ids = p_patch * patch_size * patch_size + y_patch * patch_size + x_patch + 1    # ids of the events in the patch, 0 is reserved for padding

            # split the patch into aggregated groups by resolution
            groups = []
            for idx_group in range(ctx_len):
                _index_left = np.searchsorted(t_patch, idx_group * resolution, side='left')
                _index_right = np.searchsorted(t_patch, (idx_group + 1) * resolution, side='left')
                ids_group = ids[_index_left: _index_right]
                if len(ids_group) > max_num_events:
                    print(f'Warning: The number of events [{len(ids_group)}] in the group is greater than the maximum number of events [{max_num_events}].')
                    ids_group = ids_group[:max_num_events]
                else:
                    ids_group = np.pad(ids_group, (0, max_num_events - len(ids_group)), mode='constant', constant_values=0)  # padding, ids_group.shape = (max_num_events,)
                groups.append(ids_group)
            
            groups = np.stack(groups, axis=0)   # groups.shape = (ctx_len, max_num_events)
            patches.append(groups)
        
        patches = np.stack(patches, axis=0)     # patches.shape = (num_patches, ctx_len, max_num_events)
        chunks.append(patches)
    
    chunks = np.concatenate(chunks, axis=0)    # chunks.shape = (num_chunks * num_patches, ctx_len, max_num_events)
    return chunks   


def convert_events_save_to_tensor_patches_by_time(
        root: str,
        npz_file_loader: callable,
        duration: int,
        resolution: int,
        patch_size: int,
        max_num_events: int,
        height: int,
        width: int,
        sub_dirs = None,):
    # root
    events_np_root = os.path.join(root, 'events_np')
    tensor_patches_root = os.path.join(root, 'tensor_patches_T_{}_R_{}_P_{}'.format(duration, resolution, patch_size))
    if not os.path.exists(tensor_patches_root):
        os.makedirs(tensor_patches_root)
        print(f'Mkdir [{tensor_patches_root}].')
    
    # sub dirs, e.g., train, test
    events_np_dirs = []
    tensor_patches_dirs = []
    if sub_dirs is None:
        events_np_dirs.append(events_np_root)
        tensor_patches_dirs.append(tensor_patches_root)
    else:
        for sub_dir in sub_dirs:
            events_np_dirs.append(os.path.join(events_np_root, sub_dir))
            tensor_patches_dirs.append(os.path.join(tensor_patches_root, sub_dir))
            if not os.path.exists(tensor_patches_dirs[-1]):
                os.makedirs(tensor_patches_dirs[-1])
                print(f'Mkdir [{tensor_patches_dirs[-1]}].')

    # convert events to tensor patches
    start_time = time.time()
    for events_np_dir, tensor_patch_dir in zip(events_np_dirs, tensor_patches_dirs):
        data = []
        labels = []
        count = 0
        for sub_dir in os.listdir(events_np_dir):
            events_np_sub_dir = os.path.join(events_np_dir, sub_dir)
            for file in os.listdir(events_np_sub_dir):
                if file.endswith('.npz'):
                    file_path = os.path.join(events_np_sub_dir, file)
                    chunks = convert_npz_to_patches_by_time(file_path, npz_file_loader, patch_size, height, width, duration, resolution, max_num_events)
                    # chunks.shape = (num_samples, ctx_len, max_num_events)
                    data.append(chunks)
                    label = np.array([count] * chunks.shape[0])  # label.shape = (num_samples,)
                    labels.append(label)
                    print(f'Processed [{file_path}].')
            count += 1
        data = np.concatenate(data, axis=0)    # data.shape = (num_samples, ctx_len, max_num_events)
        labels = np.concatenate(labels, axis=0)    # labels.shape = (num_samples,)
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels).int()
        torch.save(data, os.path.join(tensor_patch_dir, 'data.pt'))
        torch.save(labels, os.path.join(tensor_patch_dir, 'labels.pt'))
        print(f'Saved [{os.path.join(tensor_patch_dir, "data.pt")}].')
        print(f'Saved [{os.path.join(tensor_patch_dir, "labels.pt")}].')

    print(f'Used time = [{round(time.time() - start_time, 2)}s].')


class TensorDVS128GesturePatched(TensorDataset):
    def __init__(self, root, train, duration, resolution, patch_size, max_num_events):
        patches_root = os.path.join(root, 'tensor_patches_T_{}_R_{}_P_{}'.format(duration, resolution, patch_size))
        
        if not os.path.exists(patches_root):
            print('No such directory [{}]'.format(patches_root))

            def npz_loader(file_path):
                data = np.load(file_path)       # data = {'t': t, 'x': x, 'y': y, 'p': p}
                t = data['t'].astype(int)       # t.shape = (num_events,)
                x = data['x'].astype(int)       # x.shape = (num_events,)
                y = data['y'].astype(int)       # y.shape = (num_events,)
                p = data['p'].astype(int)       # p.shape = (num_events,)
                return t, x, y, p
            height = 128
            width = 128
            sub_dirs = ['train', 'test']
            convert_events_save_to_tensor_patches_by_time(root, npz_loader, duration, resolution, patch_size, max_num_events, height, width, sub_dirs)
        self.root = os.path.join(patches_root, 'train' if train else 'test')
        data = torch.load(os.path.join(self.root, 'data.pt'))
        labels = torch.load(os.path.join(self.root, 'labels.pt'))
        super().__init__(data, labels)


def get_tensor_data_loader(args):
    if args.dataset == 'dvs128_gesture':
        dataset_type = TensorDVS128GesturePatched
        duration = args.time
        resolution = args.resolution
        patch_size = args.patch_size
        max_num_events = args.max_num_events
        train_dataset = dataset_type(root=args.data_dir, train=True, duration=duration, resolution=resolution, patch_size=patch_size, max_num_events=max_num_events)
        valid_dataset = dataset_type(root=args.data_dir, train=False, duration=duration, resolution=resolution, patch_size=patch_size, max_num_events=max_num_events)
    else:
        raise NotImplementedError(args.dataset)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False)

    return train_loader, valid_loader


if __name__ == '__main__':

    root = '/home/haohq/datasets/DVS128Gesture'
    duration = 1000000
    resolution = 5000
    patch_size = 32
    max_num_events = 768
    dataset = TensorDVS128GesturePatched(root, train=True, duration=duration, resolution=resolution, patch_size=patch_size, max_num_events=max_num_events)
    print(len(dataset))
    print('Done.')