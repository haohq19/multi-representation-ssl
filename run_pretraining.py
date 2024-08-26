import os
import argparse
import random
import glob
import numpy as np
import torch
import yaml
from models.downsampled_autoencoder.causal_event_model_infctx import CausalEventModelInfctx
from utils.data2 import get_tensor_data_loader
from utils.distributed2 import init_ddp, is_master
from engines.pretrain2 import pretrain

_seed_ = 2024
random.seed(2024)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='Causal Event Model')
    # data
    parser.add_argument('--dataset', default='dvs128_gesture', type=str, help='dataset')
    parser.add_argument('--time', default=10000, type=int, help='time duration')
    parser.add_argument('--seq_len', default=128, type=int, help='sequence length')
    parser.add_argument('--root', default='/home/haohq/datasets/DVS128Gesture', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    # model
    parser.add_argument('--ctx_len', default=128, type=int, help='context length of a chunk')
    parser.add_argument('--patch_size', default=32, type=int, help='patch size')
    parser.add_argument('--d_model', default=512, type=int, help='dimension of embedding')
    parser.add_argument('--num_layers', default=4, type=int, help='number of encoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dimension of feedforward')
    parser.add_argument('--dropout', default=0, type=float, help='dropout rate')
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=4, type=int, help='number of workers')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--output_dir', default='outputs/pretrain_cem', help='dir to save')
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency')
    parser.add_argument('--resume', help='resume from latest checkpoint', action='store_true')
    parser.add_argument('--test', type=str, default=None, help='add test info')
    # distributed
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--backend', default='gloo', help='distributed backend')
    return parser.parse_args()


def get_output_dir(args):

    output_dir = os.path.join(args.output_dir, f'{args.dataset}_t{args.time}_T{args.seq_len}')
    output_dir += f'_ctx{args.ctx_len}_psz{args.patch_size}_dmd{args.d_model}_nly{args.num_layers}_dff{args.dim_feedforward}_dp{args.dropout}'
    
    output_dir += f'_nep{args.nepochs}_lr{args.lr}'
   
    if args.test:
        output_dir += f'_test_{args.test}'
    
    if args.resume:
        output_dir += '_resume'
    
    if args.distributed:
        output_dir += f'_dist{args.world_size}'
    
    return output_dir


def main(args):
    # init distributed data parallel
    init_ddp(args)

    # device
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(args.device_id)

    # data
    train_loader, val_loader = get_tensor_data_loader(args)

    # output_dir
    output_dir = get_output_dir(args)
    if is_master():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
            os.makedirs(os.path.join(output_dir, 'checkpoint'))



    # model
    model = CausalEventModelInfctx(
        ctx_len=args.ctx_len,
        patch_size=args.patch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )

    print('Model size: {:.2f} MB'.format(model.num_params / 1024 / 1024))

    # resume
    checkpoint = None
    if args.resume:
        checkpoint_files = glob.glob(os.path.join(output_dir, 'checkpoint/*.pth'))
        if checkpoint_files:
            latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)  # get the latest checkpoint
            checkpoint = torch.load(latest_checkpoint_file)
            print('Resume from checkpoint [{}]'.format(latest_checkpoint_file))
            model.load_state_dict(checkpoint['model'])
            print('Resume model from epoch {}'.format(checkpoint['epoch']))

    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
   
    # run
    epoch = 0
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if checkpoint:
        epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Resume optimizer from epoch {}'.format(epoch))
   
    # print and save args
    if is_master():
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        print('Args:' + str(vars(args)))

    # pretrain
    pretrain(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
        save_freq=args.save_freq,
        distributed=args.distributed,
    )

if __name__ == '__main__':
    args = parser_args()
    main(args)
    

''''
python pretrain_dae.py
torchrun --nproc_per_node=4 pretrain_dae.py
'''