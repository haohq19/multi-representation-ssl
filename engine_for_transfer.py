import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, random_split


class HiddenStateDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_subdirs = [os.path.join(self.root, dir) for dir in os.listdir(self.root)]
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



def cache_hidden_states(
        model: nn.Module,
        data_loader: DataLoader,
        input_len: int,
        hidden_dir: str,
):
    model.eval()
    nsteps_per_epoch = len(data_loader)

    # progress bar
    process_bar = tqdm.tqdm(total=nsteps_per_epoch)

    # train one step
    with torch.no_grad():
        for _, (data, data_ids) in enumerate(data_loader):
            data = data.cuda(non_blocking=True)
            # preprocess data
            input = data[:, :input_len].float()
            # forward
            _, hidden = model(input, return_hidden=True)

            # get data file name and save hidden states
            data_ids = data_ids.cpu().numpy()
            hidden = hidden.cpu().detach().numpy() 
            data_files = [data_loader.dataset.data_files[data_id] for data_id in data_ids]  # absolute path
            relative_data_files = [os.path.relpath(data_file, data_loader.dataset.data_path) for data_file in data_files]   # absolute path -> relative path
            for i, relative_data_file in enumerate(relative_data_files):
                hidden_file = os.path.join(hidden_dir, relative_data_file)
                if not os.path.exists(os.path.dirname(hidden_file)):
                    os.makedirs(os.path.dirname(hidden_file))
                    print('Make dir [{}]'.format(os.path.dirname(hidden_file)))
                np.save(hidden_file, hidden[i])
            process_bar.update(1)

    process_bar.close()


def unpatchify_hidden_states(
        patch_size: int,
        hidden_dir: str,
        unpatched_hidden_dir: str,
        sensor_size: tuple,
        d_hidden: int,
):     
    height, width = sensor_size
    n_patches = height * width // patch_size ** 2
    
    hidden_subdirs = []
    unpatched_hidden_subdirs = []

    for dirname in os.listdir(hidden_dir):
        hidden_subdirs.append(os.path.join(hidden_dir, dirname))
        unpatched_hidden_subdirs.append(os.path.join(unpatched_hidden_dir, dirname))
        if not os.path.exists(unpatched_hidden_subdirs[-1]):
            os.makedirs(unpatched_hidden_subdirs[-1])
            print('Make dir [{}]'.format(unpatched_hidden_subdirs[-1]))
    
    for hidden_subdir, unpatched_hidden_subdir in zip(hidden_subdirs, unpatched_hidden_subdirs):
        hidden_filenames = os.listdir(hidden_subdir)
        unpatched_hidden_filenames = defaultdict(list)

        for hidden_filename in hidden_filenames:
            prefix = "_".join(hidden_filename.split("_")[:-2])
            unpatched_hidden_filenames[prefix].append(hidden_filename)
        
        for prefix, filenames in unpatched_hidden_filenames.items():
            unpatched_hidden = np.zeros((n_patches, d_hidden))
            for i in range(n_patches):
                filename = prefix + f"_{i}_0.npy"
                if filename in filenames:
                    file_path = os.path.join(hidden_subdir, filename)
                    hidden = np.load(file_path)
                    unpatched_hidden[i] = hidden
            
            save_path = os.path.join(unpatched_hidden_subdir, prefix).replace('.npz', '.npy') 
            np.save(save_path, unpatched_hidden)
            print(f"Save unpatched hidden states to [{save_path}]")



def train_one_epoch(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
):
    model.train()
    nsamples_per_epoch = len(data_loader.dataset)
    nsteps_per_epoch = len(data_loader)
    epoch_total_loss = 0

    # progress bar
    process_bar = tqdm.tqdm(total=nsteps_per_epoch)

    # train one step
    correct = 0
    for data, label in data_loader:
        input = data.float().cuda(non_blocking=True)
        target = label.cuda(non_blocking=True)
        # forward
        output = model(input)
        # loss
        loss = loss_fn(output, target)
        # acc
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        process_bar.set_description('loss: {:.5e}'.format(loss.item()))
        process_bar.update(1)   
        epoch_total_loss += loss.item() * data.size(0)

    process_bar.close()
    acc = correct / nsamples_per_epoch
    print('Train average loss: {:.5e}, accuracy: {:.5f}'.format(epoch_total_loss / nsamples_per_epoch, acc))


def validate(
    model: nn.Module,
    data_loader: DataLoader,
):
    # validate
    model.eval()
    nsamples_per_epoch = len(data_loader.dataset)
    correct = 0

    with torch.no_grad():
        for data, label in data_loader:
            input = data.float().cuda(non_blocking=True)
            targets = label.cuda(non_blocking=True)
            # forward
            output = model(input)
            # acc
            pred = output.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
        
    acc = correct / nsamples_per_epoch
    print('Validation accuracy: {:.5f}'.format(acc))


def transfer(
    pretrain_model: nn.Module,
    data_loader: DataLoader,
    input_len: int,
    patch_size: int,
    d_hidden: int,
    output_dir: str,
    transfer_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    nepochs: int,
    batch_size: int,
    shuffle: bool,
):  
    dataset = data_loader.dataset
    
    # hidden dir
    hidden_dir = os.path.join(output_dir, 'hidden', data_loader.dataset.relative_data_path)
    if not os.path.exists(hidden_dir):
        os.makedirs(hidden_dir)
        print('Make dir [{}]'.format(hidden_dir))
        cache_hidden_states(
            model=pretrain_model,
            data_loader=data_loader,
            input_len=input_len,
            hidden_dir=hidden_dir,
        )

    # unpatchify hidden states
    sensor_size = dataset.get_sensor_size()
    unpatched_hidden_dir = os.path.join(output_dir, 'unpatched_hidden', data_loader.dataset.relative_data_path)
    if not os.path.exists(unpatched_hidden_dir):
        os.makedirs(unpatched_hidden_dir)
        print('Make dir [{}]'.format(unpatched_hidden_dir))
        unpatchify_hidden_states(
            patch_size=patch_size,
            hidden_dir=hidden_dir,
            unpatched_hidden_dir=unpatched_hidden_dir,
            sensor_size=sensor_size,
            d_hidden=d_hidden,
        )

    hidden_state_dataset = HiddenStateDataset(root=unpatched_hidden_dir)
    train_dataset, val_dataset = random_split(hidden_state_dataset, [int(0.8 * len(hidden_state_dataset)), len(hidden_state_dataset) - int(0.8 * len(hidden_state_dataset))])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


    # transfer
    epoch = 0
    while(epoch < nepochs):
        print('Epoch [{}/{}]'.format(epoch + 1, nepochs))

        train_one_epoch(
            model=transfer_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_loader=train_loader,
        )
        validate(
            model=transfer_model,
            data_loader=val_loader,
        )
        epoch += 1

    

    # save model
    torch.save(transfer_model.state_dict(), os.path.join(output_dir, "checkpoints", 'transfer_model.pth'))




    
