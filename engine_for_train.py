import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
from collections import defaultdict
from utils.data import HiddenStateDataset
from torch.utils.data import DataLoader, random_split

def cache_hidden_states(
        model: nn.Module,
        data_loader: DataLoader,
        input_len: int,
        hidden_dir: str,
):        
    """
    Caches the hidden states of a model for a given dataset.
    Args:
        model (nn.Module): The neural network model to use for generating hidden states.
        data_loader (DataLoader): DataLoader providing the dataset to process.
        input_len (int): The length of the input sequence.
        hidden_dir (str): The directory to save hidden states.
    Returns:
        None
    """
    model.eval()
    nsteps = len(data_loader)

    # progress bar
    process_bar = tqdm.tqdm(total=nsteps)

    with torch.no_grad():
        for data, indices in data_loader:
            data = data.cuda(non_blocking=True)
            # preprocess data
            input = data[:, :input_len].float()
            # forward
            _, hidden = model(input, return_hidden=True)
            # get the path to the data files in the batch 
            indices = indices.cpu().numpy().long()
            batch_file_pathes = [data_loader.dataset.data_file_pathes[idx] for idx in indices]  # absolute path of data files in the batch
            # absolute path -> relative path 
            # data_path is the data directory containing all subdirs of data files
            # one subdir contains data files of one label
            # relpath havs the format of 'subdirname/filename'
            batch_file_relpathes = [os.path.relpath(file_path, data_loader.dataset.data_path) for file_path in batch_file_pathes]   
            # save hidden states
            hidden = hidden.cpu().detach().numpy()
            for i, relpath in enumerate(batch_file_relpathes):
                hidden_file_path = os.path.join(hidden_dir, relpath)    # copy the dir structure: data_path -> hidden_dir
                if not os.path.exists(os.path.dirname(hidden_file_path)):
                    os.makedirs(os.path.dirname(hidden_file_path))
                    print('Make dir [{}]'.format(os.path.dirname(hidden_file_path)))
                np.save(hidden_file_path, hidden[i])
            process_bar.update(1)

    process_bar.close()


def unpatchify_hidden_states(
        patch_size: int,
        hidden_dir: str,
        unpatched_hidden_dir: str,
        sensor_size: tuple,
        d_hidden: int,
):  
    """
    Reconstructs unpatched hidden states from patched hidden states and saves them to the specified directory.
    Args:
        patch_size (int): The size of each patch.
        hidden_dir (str): The directory containing the patched hidden states.
        unpatched_hidden_dir (str): The directory to save the unpatched hidden states.
        sensor_size (tuple): The size of the sensor (height, width).
        d_hidden (int): The dimension of the hidden states.
    Returns:
        None
    """

    height, width = sensor_size
    npatches = height * width // patch_size ** 2
    
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
            # get the raw file name (prefix) of the hidden state
            # hidden_filename = data_filename
            # data_filename = "{}_{}_{}.npy".format(os.path.splitext(os.path.basename(raw_file_path))[0], idx_patch, idx_slice)
            prefix = "_".join(hidden_filename.split("_")[:-2])
            # group hidden states by their raw file name (prefix)
            unpatched_hidden_filenames[prefix].append(hidden_filename)  
        
        for prefix, filenames in unpatched_hidden_filenames.items():
            unpatched_hidden = np.zeros((npatches, d_hidden))   # unpatched hidden states, shape: (npatches, d_hidden)
            for idx_patch in range(npatches):
                filename = prefix + f"_{idx_patch}_0.npy"
                if filename in filenames:
                    hidden_file_path = os.path.join(hidden_subdir, filename)
                    hidden = np.load(hidden_file_path)
                    unpatched_hidden[idx_patch] = hidden
            
            # save unpatched hidden states
            unpatched_hidden_path = os.path.join(unpatched_hidden_subdir, prefix)
            np.save(unpatched_hidden_path, unpatched_hidden)
            print(f"Save unpatched hidden states to [{unpatched_hidden_path}]")



def train_one_epoch(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
):
    model.train()
    nsamples = len(data_loader.dataset)
    nsteps = len(data_loader)
    total_loss = 0
    correct = 0

    # process bar
    process_bar = tqdm.tqdm(total=nsteps)

    # train
    for data, label in data_loader:
        # preprocess data
        input = data.cuda(non_blocking=True).float()
        target = label.cuda(non_blocking=True).long()
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

        # update process bar
        process_bar.set_description('loss: {:.5e}'.format(loss.item()))
        process_bar.update(1) 
        # update total loss  
        total_loss += loss.item() * data.size(0)

    process_bar.close()
    print('Train average loss: {:.5e}, accuracy: {:.5f}'.format(total_loss / nsamples, correct / nsamples))


def validate(
    model: nn.Module,
    data_loader: DataLoader,
):
    model.eval()
    nsamples = len(data_loader.dataset)
    correct = 0
    
    # validate
    with torch.no_grad():
        for data, label in data_loader:
            # preprocess data
            input = data.cuda(non_blocking=True).float()
            targets = label.cuda(non_blocking=True).long()
            # forward
            output = model(input)
            # acc
            pred = output.argmax(dim=1)
            correct += pred.eq(targets).sum().item()

    print('Validation accuracy: {:.5f}'.format(correct / nsamples))

    acc = correct / nsamples
    return acc



def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    nepochs: int,
    output_dir: str,
):  

    # train
    epoch = 0
    while(epoch < nepochs):
        print('Epoch [{}/{}]'.format(epoch + 1, nepochs))
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_loader=train_loader,
        )
        validate(
            model=model,
            data_loader=val_loader,
        )
        
        epoch += 1

    # save model
    torch.save(model.state_dict(), os.path.join(output_dir, "checkpoints", 'ckpt_{}.pth'.format(epoch)))
    print('Save model to [{}]'.format(os.path.join(output_dir, "checkpoints", 'ckpt_{}.pth'.format(epoch))))
    




    
