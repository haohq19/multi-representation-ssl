import os
import multiprocessing
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



def preprocess_events(raw_data_dir, raw_data_loader, preprocessed_data_dir, patch_size, num_patches_x, num_patches_y, num_time_steps, resolution):
    """
    Preprocess the raw event data by splitting it into patches and time steps.

    Args:
        raw_data_dir (str): The directory containing the raw event data.
        raw_data_loader (callable): A function to load the raw event data.
        preprocessed_data_dir (str): The directory to save the preprocessed data.
        patch_size (int): The size of each patch.
        num_patches_x (int): The number of patches in the x direction.
        num_patches_y (int): The number of patches in the y direction.
        num_time_steps (int): The number of time steps in each sample.
        resolution (int): The duration of each time step.

    """
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)
        print(f'Created directory [{preprocessed_data_dir}].')

    raw_files = [f for f in os.listdir(raw_data_dir)]
    global_sample_idx  = 0

    for raw_file in raw_files:
        # load the raw event data
        raw_data_path = os.path.join(raw_data_dir, raw_file)
        events = raw_data_loader(raw_data_path)  # events.shape = (num_events, 4) with rows (timestamp, x, y, polarity)
        print(f'Loaded [{raw_data_path}].')
        
        # calculate the number of full samples
        t_start = events[0, 0]
        t_end = events[-1, 0]
        total_time = t_end - t_start 
        sample_duration = num_time_steps * resolution
        num_samples = int(total_time // sample_duration)
        
        # skip the file if the number of samples is 0
        if num_samples == 0:
            print(f'Warning: No full samples in file [{raw_file}]. Skipping.')
            continue
        
        # filter out events that are outside the time range of the samples
        t_end = t_start + num_samples * sample_duration
        end_idx = np.searchsorted(events[:, 0], t_end, side='right')
        events = events[:end_idx]

        # generate the sample time starts
        sample_time_starts = np.arange(t_start, t_end, sample_duration)

        # for each sample time start
        for sample_time_start in sample_time_starts:
            sample_time_end = sample_time_start + sample_duration

            # filter the events for the current sample
            start_idx = np.searchsorted(events[:, 0], sample_time_start, side='left')
            end_idx = np.searchsorted(events[:, 0], sample_time_end, side='right')
            sample_events = events[start_idx:end_idx].copy()

            # normalize the timestamps to start from 0
            sample_events[:, 0] -= sample_events[0, 0]

            # calculate the time indices
            time_indices = (sample_events[:, 0] // resolution).astype(np.int32)

            # calculate the patch indices
            patch_x_indices = (sample_events[:, 1] // patch_size).astype(np.int32)
            patch_y_indices = (sample_events[:, 2] // patch_size).astype(np.int32)

            # filter out events that are outside the patch range
            valid_indices = (
                (patch_x_indices >= 0) & (patch_x_indices < num_patches_x) &
                (patch_y_indices >= 0) & (patch_y_indices < num_patches_y) &
                (time_indices >= 0) & (time_indices < num_time_steps)
            )

            sample_events = sample_events[valid_indices]
            time_indices = time_indices[valid_indices]
            patch_x_indices = patch_x_indices[valid_indices]
            patch_y_indices = patch_y_indices[valid_indices]

            # Initialize the dictionary to store the events for each patch
            patch_dict = {}

            # calculate the patch ids
            patch_ids = patch_y_indices * num_patches_x + patch_x_indices
            
            # split the events into patches
            unique_patch_ids = np.unique(patch_ids)
            for p_id in unique_patch_ids:
                # Mask for the events in the current patch
                patch_mask = (patch_ids == p_id)
                patch_events = sample_events[patch_mask]
                patch_time_indices = time_indices[patch_mask]

                # Adjust x and y coordinates relative to the patch
                patch_events[:, 1] %= patch_size
                patch_events[:, 2] %= patch_size

                # initialize the list to store the events for each time step
                time_step_events = [np.empty((0, 4), dtype=np.float32) for _ in range(num_time_steps)]

                # split the events into time steps
                unique_time_indices = np.unique(patch_time_indices)
                for t_idx in unique_time_indices:
                    time_mask = (patch_time_indices == t_idx)
                    events_in_time = patch_events[time_mask]
                    time_step_events[t_idx] = events_in_time
                
                patch_dict[p_id] = time_step_events
            
            # save the events for each patch
            for p_id, time_step_events in patch_dict.items():
                sample_file = os.path.join(
                    preprocessed_data_dir,
                    f'sample_{global_sample_idx}.npz'
                )

                # Convert time_step_events to an object array to handle variable-length arrays
                events_array = np.array(time_step_events, dtype=object)

                np.savez_compressed(sample_file,
                                    events=events_array)
                print(f'Saved [{sample_file}].')
                global_sample_idx += 1
                

def preprocess_wrapper(args):
    preprocess_events(*args)


def convert_events_to_patches(root_dir, preprocessed_root_dir, patch_size, num_patches_x, num_patches_y, num_time_steps, resolution, raw_data_loader, max_concurrent_processes=None):
    subdirectories = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]
    preprocessed_dirs = [os.path.join(preprocessed_root_dir, os.path.basename(d)) for d in subdirectories]

    args_list = []
    for raw_dir, preprocessed_dir in zip(subdirectories, preprocessed_dirs):
        args = (
            raw_dir,           # raw_data_dir
            raw_data_loader,   # raw_data_loader function
            preprocessed_dir,  # preprocessed_data_dir
            patch_size,
            num_patches_x,
            num_patches_y,
            num_time_steps,
            resolution
        )
        args_list.append(args)

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(preprocess_wrapper, args_list)

    # for args in args_list:
    #     preprocess_wrapper(args)

# class DVS128GesturePatched(TensorDataset):
#     def __init__(self, root, train, num_time_steps, resolution, patch_size):
#         self.height = 128
#         self.width = 128
#         patches_root = os.path.join(root, 'patched_T_{}_R_{}_P_{}'.format(num_time_steps, resolution, patch_size))
#         self.root = os.path.join(patches_root, 'train' if train else 'test')
#         def loader(file_path):
#                 data = np.load(file_path)       # data = {'t': t, 'x': x, 'y': y, 'p': p}
#                 t = data['t'].astype(np.float32)       # t.shape = (num_events,)
#                 x = data['x'].astype(np.float32)       # x.shape = (num_events,)
#                 y = data['y'].astype(np.float32)       # y.shape = (num_events,)
#                 p = data['p'].astype(np.float32)       # p.shape = (num_events,)
#                 events = np.stack([t, x, y, p], axis=-1)       # events.shape = (num_events, 4)
#                 return events
#         data = torch.load(os.path.join(self.root, 'data.pt'))
#         labels = torch.load(os.path.join(self.root, 'labels.pt'))
#         super().__init__(data, labels)


# def get_tensor_data_loader(args):
#     if args.dataset == 'dvs128_gesture':
#         dataset_type = TensorDVS128GesturePatched
#         duration = args.time
#         resolution = args.resolution
#         patch_size = args.patch_size
#         max_num_events = args.max_num_events
#         train_dataset = dataset_type(root=args.data_dir, train=True, duration=duration, resolution=resolution, patch_size=patch_size, max_num_events=max_num_events)
#         valid_dataset = dataset_type(root=args.data_dir, train=False, duration=duration, resolution=resolution, patch_size=patch_size, max_num_events=max_num_events)
#     else:
#         raise NotImplementedError(args.dataset)
    
#     if args.distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#         valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
#     else:
#         train_sampler = torch.utils.data.RandomSampler(train_dataset)
#         valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
    
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False)
#     valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False)

#     return train_loader, valid_loader

def loader(file_path):
                data = np.load(file_path)       # data = {'t': t, 'x': x, 'y': y, 'p': p}
                t = data['t'].astype(np.float32)       # t.shape = (num_events,)
                x = data['x'].astype(np.float32)       # x.shape = (num_events,)
                y = data['y'].astype(np.float32)       # y.shape = (num_events,)
                p = data['p'].astype(np.float32)       # p.shape = (num_events,)
                events = np.stack([t, x, y, p], axis=-1)       # events.shape = (num_events, 4)
                return events

if __name__ == '__main__':

    root = '/home/haohq/datasets/DVS128Gesture/events_np/train'
    preprocessed_root = '/home/haohq/datasets/DVS128Gesture/patched_test/train'
    patch_size = 32
    num_patches_x = 8
    num_patches_y = 8
    num_time_steps = 1000
    resolution = 1000
    raw_data_loader = loader

    convert_events_to_patches(root, preprocessed_root, patch_size, num_patches_x, num_patches_y, num_time_steps, resolution, raw_data_loader)
