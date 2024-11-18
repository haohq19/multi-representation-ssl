import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dist import is_master, global_meters_sum, save_on_master
from utils.log import visualize_frame_in_tb
from utils.data import raw_events_to_frame, raw_events_to_time_surface


def preprocess_data(data: torch.Tensor, input_len: int, rep_len: int, patch_size: int, pred_frame: bool, pred_ts: bool, pred_next_frame: bool, tau: float):
    """
    Preprocess the data for training.
    Args:
        data: torch.Tensor, shape [batch_size, n_events, 4]
        input_len: int, the length of input events
        rep_len: int, the length of representation events
        patch_size: int, the size of patch
        pred_frame: bool, whether to predict the frame
        pred_ts: bool, whether to predict the time surface
        pred_next_frame: bool, whether to predict the next frame
        tau: float, the time constant for time surface
    Returns:
        input: torch.Tensor, shape [batch_size, input_len, 4], dtype float
        targets: torch.Tensor, shape [batch_size, n_targets, 2, frame_size, frame_size], dtype float
    """

    _, n_events, _ = data.size()
    input = data[:, :input_len].float()   # [batch_size, input_len, 4]
    targets = []
    if pred_frame:
        frame = raw_events_to_frame(data[:, input_len-rep_len:input_len], frame_size=(patch_size, patch_size))
        targets.append(frame)   # [batch_size, 2, frame_size, frame_size]
    if pred_ts:
        ts = raw_events_to_time_surface(data[:, input_len-rep_len:input_len], time_surface_size=(patch_size, patch_size), tau=tau)
        targets.append(ts)      # [batch_size, 2, frame_size, frame_size]
    if pred_next_frame:
        if rep_len + input_len > n_events:
            raise ValueError('Rep_len + input_len should be less than n_events')
        next_frame = raw_events_to_frame(data[:, input_len:input_len+rep_len], frame_size=(patch_size, patch_size))
        targets.append(next_frame)  # [batch_size, 2, frame_size, frame_size]
    targets = torch.stack(targets, dim=1).float()  # [batch_size, n_targets, 2, frame_size, frame_size]
    return input, targets
    

def postprocess_data(data: torch.Tensor, patch_size: int, pred_frame: bool, pred_ts: bool, pred_next_frame: bool):
    """
    Postprocess the data for visualization.
    Args:
        data: torch.Tensor, shape [batch_size, num_classes]
        patch_size: int, the size of patch
        pred_frame: bool, whether to predict the frame
        pred_ts: bool, whether to predict the time surface
        pred_next_frame: bool, whether to predict the next frame
    Returns:
        preds: torch.Tensor, shape [batch_size, n_targets, 2, frame_size, frame_size]
    """
    n_targets = pred_frame + pred_ts + pred_next_frame
    preds = data.view(-1, n_targets, 2, patch_size, patch_size)
    return preds
    

def train_one_epoch(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        epoch: int,
        tb_writer: SummaryWriter,
        input_len: int,
        rep_len: int,
        patch_size: int,
        pred_frame: bool,
        pred_ts: bool,
        pred_next_frame: bool,
        tau: float,
        dist: bool,
):
    model.train()
    nsamples_per_epoch = len(data_loader.dataset)
    nsteps_per_epoch = len(data_loader)
    epoch_total_loss = 0

    # progress bar
    if is_master():
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)

    # train one step
    for step, (data, _) in enumerate(data_loader):
        data = data.cuda(non_blocking=True)
        # preprocess data
        input, targets = preprocess_data(data, input_len, rep_len, patch_size, pred_frame, pred_ts, pred_next_frame, tau)
        # forward
        output = model(input)
        # postprocess data
        preds = postprocess_data(output, patch_size, pred_frame, pred_ts, pred_next_frame)
        # loss
        loss = loss_fn(preds, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if is_master():
            # process bar
            process_bar.set_description('loss: {:.5e}'.format(loss.item()))
            process_bar.update(1)
            # log in tb
            tb_writer.add_scalar(tag='step/loss', scalar_value=loss.item(), global_step=epoch * nsteps_per_epoch + step)
            if step == nsteps_per_epoch - 1:
                # output.shape = [batch_size, n_targets, 2, frame_size, frame_size]
                if pred_frame:
                    frame_predicted = preds[:, 0]    # [batch_size, 2, frame_size, frame_size]
                    frame_target = targets[:, 0]     # [batch_size, 2, frame_size, frame_size]
                    visualize_frame_in_tb(frame_predicted, epoch, tb_writer, tag='train/predicted_frame', max_out=4)
                    visualize_frame_in_tb(frame_target, epoch, tb_writer, tag='train/target_frame', max_out=4)

            
        epoch_total_loss += loss.item() * data.size(0)

    if dist:
        epoch_total_loss = global_meters_sum(epoch_total_loss)
    
    if is_master():
        # process bar
        process_bar.close()
        # log in tb
        tb_writer.add_scalar('train/loss', epoch_total_loss / nsamples_per_epoch, epoch + 1)

    print('Train average loss: {:.5e}'.format(epoch_total_loss / nsamples_per_epoch)) 

def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    epoch: int,
    tb_writer: SummaryWriter,
    input_len: int,
    rep_len: int,
    patch_size: int,
    pred_frame: bool,
    pred_ts: bool,
    pred_next_frame: bool,
    tau: float,
    dist: bool,
):
    # validate
    model.eval()
    nsamples_per_epoch = len(data_loader.dataset)
    epoch_total_loss = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.cuda(non_blocking=True)
            input, targets = preprocess_data(data, input_len, rep_len, patch_size, pred_frame, pred_ts, pred_next_frame, tau)
            input = data.float().cuda(non_blocking=True)
            output = model(input)
            loss = loss_fn(output, targets)
            epoch_total_loss += loss.item() * data.size(0)
    if dist:
        epoch_total_loss = global_meters_sum(epoch_total_loss)
    if is_master():
        tb_writer.add_scalar('valid/loss', epoch_total_loss / nsamples_per_epoch, epoch + 1)
    print('Validation average loss: {:.5e}'.format(epoch_total_loss / nsamples_per_epoch))
    



def train(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    input_len: int,
    rep_len: int,
    patch_size: int,
    pred_frame: bool,
    pred_ts: bool,
    pred_next_frame: bool,
    tau: float,
    output_dir: str,
    save_freq: int,
    dist: bool,
):  
    # log
    if is_master():
        tb_writer = SummaryWriter(output_dir + '/log')
    print('Save log to {}'.format(output_dir + '/log'))
    
    epoch = epoch
    while(epoch < nepochs):
        print('Epoch [{}/{}]'.format(epoch + 1, nepochs))

        # train
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            tb_writer=tb_writer,
            input_len=input_len,
            rep_len=rep_len,
            patch_size=patch_size,
            pred_frame=pred_frame,
            pred_ts=pred_ts,
            pred_next_frame=pred_next_frame,
            tau=tau,
            dist=dist,
        )

        # validate
        validate(
            model=model,
            loss_fn=loss_fn,
            data_loader=val_loader,
            epoch=epoch,
            tb_writer=tb_writer,
            input_len=input_len,
            rep_len=rep_len,
            patch_size=patch_size,
            pred_frame=pred_frame,
            pred_ts=pred_ts,
            pred_next_frame=pred_next_frame,
            tau=tau,
            dist=dist,
        )

        epoch += 1

        # save
        if epoch % save_freq == 0:
            checkpoint = {
                'model': model.module.state_dict() if dist else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_name = 'checkpoints/checkpoint_{}.pth'.format(epoch)
            save_on_master(checkpoint, os.path.join(output_dir, save_name))
            print('Save checkpoint to {}'.format(output_dir))
    
