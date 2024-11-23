import os
import argparse
import random
import glob
import numpy as np
import torch
import yaml
import time
from pprint import pprint
from utils.data import get_data_loader
from modeling_pretrain import PretrainModel
from modeling_train import TransferModel
from engine_for_transfer import transfer
from utils.data import DVS128Gesture

def init_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_output_dir(output_dir, dataset):
    # time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    # output_dir = os.path.join(
    #     output_dir,
    #     f'{time_str}_{dataset}')
    
    output_dir = os.path.join(
        output_dir,
        dataset)
    return output_dir


def main():
    
    # init seed
    seed = 42
    init_seed(seed)
    

    device_id = 0
    torch.cuda.set_device(device_id)

    # data
    root = '/home/haohq/datasets/DVS128Gesture'
    count = 2048
    stride = 0
    patch_size = 32
    inverse = True
    dataset = DVS128Gesture(root, train=True, count=count, stride=stride, patch_size=patch_size, inverse=inverse)

    batch_size = 32

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # output_dir
    output_dir = "outputs/transfer_rwkv4"
    output_dir = get_output_dir(output_dir, "dvs128gesture")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Make dir [{}]'.format(output_dir))
    if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
        os.makedirs(os.path.join(output_dir, 'checkpoints'))
        print('Make dir [{}]'.format(os.path.join(output_dir, 'checkpoints')))

    # model
    config_pretrain_model = {
        'name': 'rwkv4',
        'd_model': 256,
        'depth': 4,
        'd_ffn': 1024,
        'num_classes': 2 * patch_size * patch_size * 2,
    }
    config_tokenizer = {
        'name': 'event_tokenizer',
        'patch_size': patch_size,
        'd_embed': config_pretrain_model['d_model'],
    }
    config_pretrain_models = {
        'model': config_pretrain_model,
        'tokenizer': config_tokenizer,
    }
    pretrain_model = PretrainModel(config_pretrain_models)
    print('Number of parameters of PTM: {} M'.format(pretrain_model.num_params/1e6))
    
    # load checkpoint
    checkpoint_path = "/home/haohq/multi-representation-ssl/outputs/pretrain_rwkv4/20241119-005924_dvs128gesture/checkpoints/checkpoint_100.pth"
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model']
    pretrain_model.load_state_dict(state_dict)
    pretrain_model.cuda()

    in_chans = config_pretrain_model['d_model'] * config_pretrain_model['depth']
    config_transfer_model = {
        'name': 'vit',
        'in_chans': in_chans,
        'n_patches': 16,
        'd_model': 64,
        'depth': 4,
        'nheads': 4,
        'num_classes': 11,
        'drop_rate': 0.2,
        'attn_drop_rate': 0.2,
    }
    config_transfer_models = {
        'model': config_transfer_model,
    }
    transfer_model = TransferModel(config_transfer_models)
    print('Number of parameters of TM: {} M'.format(transfer_model.num_params/1e6))
    transfer_model.cuda()

    # run
    lr = 1e-4
    params = filter(lambda p: p.requires_grad, transfer_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    input_len = 2048
    patch_size = 32
    d_hidden = config_transfer_model['in_chans']
    nepochs = 100
    batch_size = 32
    
    transfer(
        pretrain_model=pretrain_model,
        data_loader=data_loader,
        input_len=input_len,
        patch_size=patch_size,
        d_hidden=d_hidden,
        output_dir=output_dir,
        transfer_model=transfer_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        nepochs=nepochs,
        batch_size=batch_size,
        shuffle=True,
    )

if __name__ == '__main__':
    main()
    


''''
python run_pretrain_rwkv4.py    
torchrun --nproc_per_node=4 run_pretrain_rwkv4.py
'''