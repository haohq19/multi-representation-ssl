import os
import argparse
import random
import numpy as np
import torch
import yaml
import time
from pprint import pprint
from utils.data import get_data_loader_for_pretraining
from utils.dist import init_ddp, is_master
from modeling_pretrain import PretrainModel
from models.loss.multi_representation_loss import MultiRepresentationLoss
from engine_for_pretrain import train


def load_config(cfg_path):
    # load base config
    cfg_base_path = os.path.join(cfg_path, 'pretrain.yaml')
    with open(cfg_base_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model_config(cfg_path, cfg):
    # dynamically load cfg.encoder and cfg.tokenizer
    enc_name = cfg['encoder']
    cfg_enc_path = os.path.join(cfg_path, f'models/{enc_name}.yaml')
    with open(cfg_enc_path, 'r') as f:
        cfg_enc = yaml.safe_load(f)
        cfg['encoder'] = cfg_enc
        print(f'Load encoder config [{enc_name}]')

    tok_name = cfg['tokenizer']
    cfg_tok_path = os.path.join(cfg_path, f'models/{tok_name}.yaml')
    with open(cfg_tok_path, 'r') as f:
        cfg_tok = yaml.safe_load(f)
        cfg['tokenizer'] = cfg_tok
        print(f'Load tokenizer config [{tok_name}]')


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
    
    # init DDP
    init_ddp(cfg)

    # device
    if cfg['dist']:
        torch.cuda.set_device(cfg['local_rank'])
    else:
        torch.cuda.set_device(cfg['device_id'])

    # data
    dataset_name = cfg['dataset']
    root = cfg['root']
    patch_size = cfg['patch_size']
    count = cfg['count']
    stride = cfg['stride']
    inverse = cfg['inverse']
    batch_size = cfg['batch_size']
    shuffle = cfg['shuffle']
    num_workers = cfg['num_workers']
    train_split = cfg['train_split']
    val_split = cfg['val_split']
    dist = cfg['dist']
    
    train_loader, val_loader = get_data_loader_for_pretraining(
        dataset_name=dataset_name,
        root=root,
        patch_size=patch_size,
        count=count,
        stride=stride,
        inverse=inverse,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        train_split=train_split,
        val_split=val_split,
        dist=dist,
    )

    # output directory
    output_dir = cfg['output_dir']
    model_name = cfg['encoder']['name']
    # dataset_name = cfg['dataset']
    output_dir = get_output_dir(output_dir, model_name, dataset_name)
    if is_master():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print('Make dir [{}]'.format(output_dir))
        if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
            os.makedirs(os.path.join(output_dir, 'checkpoints'))
            print('Make dir [{}]'.format(os.path.join(output_dir, 'checkpoints')))

    # loss
    loss_fn_name = cfg['loss_fn']
    loss_fn = MultiRepresentationLoss(loss_fn_name=loss_fn_name)
    
    # model, optimizer and epoch
    model = PretrainModel(cfg)
    print('Number of parameters: {} M'.format(model.num_params/1e6))
    params = filter(lambda p: p.requires_grad, model.parameters())
    lr = cfg['lr']
    optimizer = torch.optim.Adam(params=params, lr=lr)
    epoch = 0

    # model to device
    model.cuda()
    if cfg['dist']:
        local_rank = cfg['local_rank']
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
   
    # print and save args
    if is_master():
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f)
        pprint(cfg)

    # train
    nepochs = cfg['nepochs']
    input_len = cfg['input_len']
    repr_len = cfg['repr_len']
    # patch_size = cfg['patch_size']
    use_frame_target = cfg['use_frame_target']
    use_ts_target = cfg['use_ts_target']
    use_next_frame_target = cfg['use_next_frame_target']
    ts_tau = cfg['ts_tau']
    save_freq = cfg['save_freq']
    dist = cfg['dist']
    
    train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=nepochs,
        epoch=epoch,
        input_len=input_len,
        repr_len=repr_len,
        patch_size=patch_size,
        use_frame_target=use_frame_target,
        use_ts_target=use_ts_target,
        use_next_frame_target=use_next_frame_target,
        ts_tau=ts_tau,
        save_freq=save_freq,
        output_dir=output_dir,
        dist=dist
    )


if __name__ == '__main__':
    cfg_path = "configs"
    cfg = load_config(cfg_path)
    update_config_from_args(cfg)
    load_model_config(cfg_path, cfg)
    update_config_from_args(cfg['encoder'])
    update_config_from_args(cfg['tokenizer'])
    main(cfg)


''''
python run_pretrain.py --device_id 0
torchrun --nproc_per_node=4 run_pretrain.py
'''