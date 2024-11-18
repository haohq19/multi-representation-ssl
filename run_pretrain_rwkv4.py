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
from utils.dist import init_ddp, is_master
from modeling_pretrain import PretrainModel
from models.loss.multi_representation_loss import MultiRepresentationLoss
from engine_for_pretraining import train


def load_config(config_path):
    config_base_path = os.path.join(config_path, 'base.yaml')
    with open(config_base_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # dynamically load config.models.model and config.models.tokenizer
    config_models = config['models']
    model_name = config_models['model']
    tokenizer_name = config_models['tokenizer']
    config_model_path = os.path.join(config_path, f'models/{model_name}.yaml')
    config_tokenizer_path = os.path.join(config_path, f'models/{tokenizer_name}.yaml')
    with open(config_model_path, 'r') as f:
        config_model = yaml.safe_load(f)
        config_models['model'] = config_model
    with open(config_tokenizer_path, 'r') as f:
        config_tokenizer = yaml.safe_load(f)
        config_models['tokenizer'] = config_tokenizer
    
    return config


def update_config_from_args(config):
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--root', type=str, help='path to dataset')
    parser.add_argument('--count', type=int, help='number of events per sample')
    parser.add_argument('--stride', type=int, help='stride of slicing')
    parser.add_argument('--input_len', type=int, help='length of input sequence')
    parser.add_argument('--rep_len', type=int, help='length of output sequence')
    parser.add_argument('--patch_size', type=int, help='patch size')
    parser.add_argument('--train_split', type=float, help='train split ratio')
    parser.add_argument('--shuffle', type=bool, help='shuffle dataset')
    parser.add_argument('--num_workers', type=int, help='number of workers')
    
    # models
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--tokenizer', type=str, help='tokenizer name')
    parser.add_argument('--d_model', type=int, help='dimension of embedding')
    parser.add_argument('--depth', type=int, help='number of encoder layers')
    parser.add_argument('--d_ffn', type=int, help='dimension of feedforward')
    
    # training
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--device_id', type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', type=int, help='number of epochs')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--output_dir', type=str, help='dir to save')
    parser.add_argument('--save_freq', type=int, help='save frequency')
    parser.add_argument('--optimizer', type=str, help='optimizer')
    parser.add_argument('--step_size', type=int, help='step size of scheduler')
    parser.add_argument('--gamma', type=float, help='gamma of scheduler')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    
    # loss
    parser.add_argument('--loss_fn', type=str, help='loss function')
    parser.add_argument('--pred_frame', type=bool, help='predict frame')
    parser.add_argument('--pred_ts', type=bool, help='predict timestamp')
    parser.add_argument('--pred_next_frame', type=bool, help='predict next frame')
    parser.add_argument('--tau', type=float, help='tau')

    # dist
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--backend', default='gloo', help='distributed backend')

    # misc
    parser.add_argument('--seed', type=int, help='random seed')

    args = parser.parse_args()
    
    # data  
    for key in ['dataset', 'root', 'count', 'stride', 'input_len', 'rep_len', 'patch_size', 'train_split', 'shuffle', 'num_workers']:
        if getattr(args, key) is not None:
            config['data'][key] = getattr(args, key)
            
    # models
    for key in ['model', 'tokenizer', 'd_model', 'depth', 'd_ffn']:
        if getattr(args, key) is not None:
            config['models'][key] = getattr(args, key)
            
    # training
    for key in ['batch_size', 'device_id', 'nepochs', 'lr', 'output_dir', 'save_freq', 'optimizer', 'step_size', 'gamma', 'resume']:
        if getattr(args, key) is not None:
            config['training'][key] = getattr(args, key)
            
    # loss
    for key in ['loss_fn', 'pred_frame', 'pred_ts', 'pred_next_frame', 'tau']:
        if getattr(args, key) is not None:
            config['loss'][key] = getattr(args, key)
            
    # dist
    for key in ['dist_url', 'backend']:
        if getattr(args, key) is not None:
            config['dist'][key] = getattr(args, key)
            
    # misc
    if args.seed is not None:
        config['misc']['seed'] = args.seed
        
    # update config
    # data
    config['data']['batch_size'] = config['training']['batch_size']
    
    # models
    patch_size = config['data']['patch_size']
    pred_frame = config['loss']['pred_frame']
    pred_ts = config['loss']['pred_ts']
    pred_next_frame = config['loss']['pred_next_frame']
    ntargets = pred_frame + pred_ts + pred_next_frame
    config['models']['model']['num_classes'] = 2 * patch_size * patch_size * ntargets
    config['models']['tokenizer']['patch_size'] = patch_size
    config['models']['tokenizer']['d_embed'] = config['models']['model']['d_model']
    
    return config



def init_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_output_dir(output_dir, dataset):
    time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

    output_dir = os.path.join(
        output_dir,
        f'{time_str}_{dataset}')
    
    return output_dir


def main(config):
    # configs
    config_dist = config['dist']
    config_data = config['data']
    config_training = config['training']
    config_models = config['models']
    config_loss = config['loss']
    config_misc = config['misc']
    
    # init seed
    init_seed(config_misc['seed'])
    
    # init distributed data parallel
    init_ddp(config_dist)

    # device
    if config_dist['dist']:
        torch.cuda.set_device(config_dist['local_rank'])
    else:
        torch.cuda.set_device(config_training['device_id'])

    # data
    train_loader, val_loader, _ = get_data_loader(config_data)

    # output_dir
    output_dir = get_output_dir(config_training['output_dir'], config_data['dataset'])
    if is_master():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print('Mkdir [{}]'.format(output_dir))
        if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
            os.makedirs(os.path.join(output_dir, 'checkpoints'))
            print('Mkdir [{}]'.format(os.path.join(output_dir, 'checkpoints')))

    # model
    model = PretrainModel(config_models)
    print('Number of parameters: {} M'.format(model.num_params/1e6))
    
    # loss
    loss_fn = MultiRepresentationLoss(config_loss)

    # resume
    checkpoint = None
    if config_training['resume']:
        checkpoint_files = glob.glob(os.path.join(output_dir, 'checkpoints/*.pth'))
        if checkpoint_files:
            latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)  # get the latest checkpoint
            checkpoint = torch.load(latest_checkpoint_file)
            print('Resume from checkpoint [{}]'.format(latest_checkpoint_file))
            model.load_state_dict(checkpoint['model'])
            print('Resume model from epoch {}'.format(checkpoint['epoch']))

    model.cuda()
    if config_dist['dist']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=config_dist['local_rank'])
   
    # run
    epoch = 0
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=config_training['lr'])

    if checkpoint:
        epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Resume optimizer from epoch {}'.format(epoch))
   
    # print and save args
    if is_master():
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
        pprint(config)

    # train
    train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=config_training['nepochs'],
        epoch=epoch,
        input_len=config_data['input_len'],
        rep_len=config_data['rep_len'],
        patch_size=config_data['patch_size'],
        pred_frame=config_loss['pred_frame'],
        pred_ts=config_loss['pred_ts'],
        pred_next_frame=config_loss['pred_next_frame'],
        tau=config_loss['tau'],
        output_dir=output_dir,
        save_freq=config_training['save_freq'],
        dist=config_dist['dist'],
    )


if __name__ == '__main__':
    config_path = "configs"
    config = load_config(config_path)
    config = update_config_from_args(config)
    main(config)
    

''''
python pretrain_dae.py
torchrun --nproc_per_node=4 pretrain_dae.py
'''