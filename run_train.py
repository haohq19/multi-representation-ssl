import os
import argparse
import random
import glob
import numpy as np
import torch
import torch.utils
import yaml
import time
from pprint import pprint
from utils.data import get_data_loader_list_for_caching_hidden_states
from modeling_pretrain import PretrainModel
from modeling_train import TrainModel
from engine_for_train import cache_hidden_states, unpatchify_hidden_states, train
from utils.data import DVS128Gesture



def load_config(cfg_path):
    # load base config
    cfg_base_path = os.path.join(cfg_path, 'train.yaml')
    with open(cfg_base_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model_config(cfg_path, cfg):
    # dynamically load cfg.encoder and cfg.tokenizer
    decoder = cfg['decoder']
    cfg_dec_path = os.path.join(cfg_path, f'models/{decoder}.yaml')
    with open(cfg_dec_path, 'r') as f:
        cfg_dec = yaml.safe_load(f)
        cfg['decoder'] = cfg_dec
        print(f'Load decoder config [{decoder}]')

def load_pretrain_config(cfg):
    checkpoint_path = cfg['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    pretrain_cfg_path = os.path.dirname(checkpoint_path).replace('checkpoints', 'config.yaml')
    with open(pretrain_cfg_path, 'r') as f:
        cfg_pretrain = yaml.safe_load(f)
        cfg['pretrain'] = cfg_pretrain
        print(f'Load pretrain config [{pretrain_cfg_path}]')


def update_config_from_args(cfg):
    parser = argparse.ArgumentParser()
    for param, value in cfg.items():
        param_type = type(value)
        parser.add_argument(f"--{param}", type=param_type, help=f"{param}")
    args, _ = parser.parse_known_args()
    
    # update config
    for key, value in vars(args).items():
        if value is not None and key in cfg:
            cfg[key] = value


def init_random(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_output_dir(output_dir, model_name, dataset):
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(output_dir, f'{model_name}_{dataset}_{timestamp}')
    return output_dir


def main(cfg):
    
    # init random
    __seed__ = cfg['seed']
    init_random(__seed__)
    
    # device
    device_id = cfg['device_id']
    torch.cuda.set_device(device_id)
    
    # cache hidden states
    cfg_pretrain = cfg['pretrain']
    
    dataset_name = cfg_pretrain['dataset']
    root = cfg_pretrain['root']
    patch_size = cfg_pretrain['patch_size']
    count = cfg_pretrain['count']
    batch_size = cfg_pretrain['batch_size']
    num_workers = cfg_pretrain['num_workers']
    
    data_loaders = get_data_loader_list_for_caching_hidden_states(
        dataset_name=dataset_name,
        root=root,
        patch_size=patch_size,
        count=count,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # output_dir
    output_dir = cfg['output_dir']
    dataset_name = cfg_pretrain['dataset']
    decoder_name = cfg['decoder']
    output_dir = get_output_dir(output_dir=output_dir, model_name=decoder_name, dataset=dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Make dir [{}]'.format(output_dir))
    if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
        os.makedirs(os.path.join(output_dir, 'checkpoints'))
        print('Make dir [{}]'.format(os.path.join(output_dir, 'checkpoints')))
    
    # model
    pretrain_model = PretrainModel(cfg=cfg_pretrain)
    print('Number of parameters of PTM: {} M'.format(pretrain_model.num_params/1e6))
    
    # load checkpoint
    checkpoint_path = cfg['checkpoint_path']
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    pretrain_model.load_state_dict(state_dict)
    pretrain_model.cuda()
    
    # cache hidden states
    input_len = cfg_pretrain['input_len']
    
    for data_loader in data_loaders:
        # get dataset
        dataset = data_loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            raise ValueError(f"Unsupported dataset: Subset")
        
        # relpath to root
        relpath = os.path.relpath(data_loader.dataset.data_path, data_loader.dataset.root)

        # hidden dir
        hidden_dir = os.path.join(
            output_dir,
            'hidden',
            relpath
        )
        # cache hidden states
        if not os.path.exists(hidden_dir):
            os.makedirs(hidden_dir)
            print('Make dir [{}]'.format(hidden_dir))
            cache_hidden_states(
                model=pretrain_model,
                data_loader=data_loader,
                input_len=input_len,
                hidden_dir=hidden_dir,
            )

        # unpatchify hidden dir
        unpatched_hidden_dir = os.path.join(
            output_dir, 
            'unpatched_hidden', 
            relpath
        )
        # unpatchify hidden states
        sensor_size = dataset.get_sensor_size()
        d_hidden = cfg_pretrain['d_hidden']
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
            
    # data
    
    dataset_name = cfg['dataset']
    root = cfg['root']
    

    # model, optimizer
    model = TrainModel(cfg)
    print('Number of parameters of TM: {} M'.format(model.num_params/1e6))
    params = filter(lambda p: p.requires_grad, model.parameters())
    lr = cfg['lr']
    optimizer = torch.optim.Adam(params=params, lr=lr)
    

    # training  
    loss_fn = torch.nn.CrossEntropyLoss()
    nepochs = cfg['nepochs']
    
    # model to device
    model.cuda()
  
    # print and save args
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    pprint(cfg)
    
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_loaders=data_loaders,
        nepochs=nepochs,
        output_dir=output_dir,
    )

if __name__ == '__main__':
    cfg_path = "configs"
    cfg = load_config(cfg_path)
    update_config_from_args(cfg)
    load_model_config(cfg_path, cfg)
    update_config_from_args(cfg['decoder'])
    load_pretrain_config(cfg)
    main(cfg)
    


''''
python run_pretrain_rwkv4.py    
torchrun --nproc_per_node=4 run_pretrain_rwkv4.py
'''