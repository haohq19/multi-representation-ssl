import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dist import is_master, global_meters_sum, save_on_master


def _visualize_reconstruction(output: torch.Tensor, target: torch.Tensor, epoch: int, tb_writer: SummaryWriter, max_outputs=64):
    # output.shape = (batch_size, 2, H, W)
    # target.shape = (batch_size, 2, H, W)
    max_outputs = min(max_outputs, output.size(1))
    output = output[0, :max_outputs]
    target = target[0, :max_outputs]
    output_0 = output[:, 0].unsqueeze(1)
    output_1 = output[:, 1].unsqueeze(1)
    target_0 = target[:, 0].unsqueeze(1)
    target_1 = target[:, 1].unsqueeze(1)
    tb_writer.add_images(tag='output_0', img_tensor=output_0, global_step=epoch + 1, dataformats='NCHW')
    tb_writer.add_images(tag='output_1', img_tensor=output_1, global_step=epoch + 1, dataformats='NCHW')
    tb_writer.add_images(tag='target_0', img_tensor=target_0, global_step=epoch + 1, dataformats='NCHW')
    tb_writer.add_images(tag='target_1', img_tensor=target_1, global_step=epoch + 1, dataformats='NCHW')


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    save_freq: int,
    distributed: bool = False,
):  
    if is_master():
        tb_writer = SummaryWriter(output_dir + '/log')
    print('Save log to {}'.format(output_dir + '/log'))
    
    epoch = epoch
    while(epoch < nepochs):
        print('Epoch [{}/{}]'.format(epoch+1, nepochs))

        # train
        model.train()
        nsamples_per_epoch = len(train_loader.dataset)
        nsteps_per_epoch = len(train_loader)
        epoch_total_loss = 0

        # progress bar
        if is_master():
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)

        # train one step
        for step, (data, _) in enumerate(train_loader):
            input = data.float().cuda(non_blocking=True)
            loss, output = model(input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # process bar & tensorboard info
            if is_master():
                process_bar.set_description('loss: {:.5e}'.format(loss.item()))
                process_bar.update(1)
                tb_writer.add_scalar(tag='step/loss', scalar_value=loss.item(), global_step=epoch * nsteps_per_epoch + step)
                if step == nsteps_per_epoch - 1:
                    pred = output[:, :-1]
                    target = input[:, 1:]
                    _visualize_reconstruction(pred, target, epoch, tb_writer)
            
            epoch_total_loss += loss.item() * data.size(0)

        # process bar & tensorboard info
        if distributed:
            epoch_total_loss = global_meters_sum(epoch_total_loss)
        if is_master():
            process_bar.close()
            tb_writer.add_scalar('train/loss', epoch_total_loss / nsamples_per_epoch, epoch + 1)
        print('Train average loss: {:.5e}'.format(epoch_total_loss / nsamples_per_epoch))
        
        # valid
        model.eval()
        nsamples_per_epoch = len(val_loader.dataset)
        epoch_total_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                input = data.float().cuda(non_blocking=True)
                loss, _ = model(input)
                epoch_total_loss += loss.item() * data.size(0)
        if distributed:
            epoch_total_loss = global_meters_sum(epoch_total_loss)
        if is_master():
            tb_writer.add_scalar('valid/loss', epoch_total_loss / nsamples_per_epoch, epoch + 1)
        print('Valid average loss: {:.5e}'.format(epoch_total_loss / nsamples_per_epoch))

        epoch += 1

        # save
        if epoch % save_freq == 0:
            checkpoint = {
                'model': model.module.state_dict() if distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_name = 'checkpoint/checkpoint_{}.pth'.format(epoch)
            save_on_master(checkpoint, os.path.join(output_dir, save_name))
            print('Save checkpoint to {}'.format(output_dir))
    
