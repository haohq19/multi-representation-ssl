import os
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dist import is_master, global_meters_sum, save_on_master
from utils.log import visualize_frame_in_tb
from utils.data import raw_events_to_frame, raw_events_to_time_surface


def preprocess_data(
        data: torch.Tensor,
        input_len: int, 
        repr_len: int, 
        patch_size: int, 
        use_frame_target: bool, 
        use_ts_target: bool, 
        use_next_frame_target: bool, 
        ts_tau: float
    ):
    """
    Preprocess the data for training. 
    Args:
        data: torch.Tensor, shape [batch_size, n_events, 4]
        input_len: int, the length of events to put into the model
        repr_len: int, the length of events to generate representations
        patch_size: int, the size of patch
        use_frame_target: bool, whether to use current frame as target
        use_ts_target: bool, whether to use time surface as target
        use_next_frame_target: bool, whether to use next frame as target
        ts_tau: float, the time constant for time surface
    Returns:
        input: torch.Tensor, shape [batch_size, input_len, 4], dtype float
        targets: torch.Tensor, shape [batch_size, n_targets, 2, frame_size, frame_size], dtype float
    """

    _, nevents, _ = data.size()
    if input_len > nevents:
        raise ValueError('No enough events for input')
    input = data[:, :input_len].float()   # [batch_size, input_len, 4]
    targets = []
    if use_frame_target:
        if input_len < repr_len:
            raise ValueError('No enough events for frame target')
        frame = raw_events_to_frame(data[:, input_len-repr_len:input_len], frame_size=(patch_size, patch_size))
        targets.append(frame)   # [batch_size, 2, frame_size, frame_size]
    if use_ts_target:
        if input_len < repr_len:
            raise ValueError('No enough events for time surface target')
        ts = raw_events_to_time_surface(data[:, input_len-repr_len:input_len], time_surface_size=(patch_size, patch_size), tau=ts_tau)
        targets.append(ts)      # [batch_size, 2, frame_size, frame_size]
    if use_next_frame_target:
        if repr_len + input_len > nevents:
            raise ValueError('No enough events for next frame target')
        next_frame = raw_events_to_frame(data[:, input_len:input_len+repr_len], frame_size=(patch_size, patch_size))
        targets.append(next_frame)  # [batch_size, 2, frame_size, frame_size]

    targets = torch.stack(targets, dim=1).float()  # [batch_size, n_targets, 2, frame_size, frame_size]

    return input, targets
    

def postprocess_data(
        data: torch.Tensor, 
        patch_size: int, 
        use_frame_target: bool, 
        use_ts_target: bool, 
        use_next_frame_target: bool):
    """
    Postprocess the output.
    Args:
        data: torch.Tensor, shape [batch_size, num_classes]
        patch_size: int, the size of patch
        use_frame_target: bool, whether to use current frame as target
        use_ts_target: bool, whether to use time surface as target
        use_next_frame_target: bool, whether to use next frame as target
    Returns:
        preds: torch.Tensor, shape [batch_size, n_targets, 2, frame_size, frame_size]
    """
    ntargets = use_frame_target + use_ts_target + use_next_frame_target
    if ntargets == 0:
        raise ValueError('No target to predict')
    preds = data.view(-1, ntargets, 2, patch_size, patch_size)
    return preds
    

def train_one_epoch(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        epoch: int,
        tb_writer: SummaryWriter,
        input_len: int,
        repr_len: int,
        patch_size: int,
        use_frame_target: bool,
        use_ts_target: bool,
        use_next_frame_target: bool,
        ts_tau: float,
        dist: bool,
):
    model.train()
    nsamples = len(data_loader.dataset)
    nsteps = len(data_loader)
    total_loss = 0

    # progress bar
    if is_master():
        process_bar = tqdm.tqdm(total=nsteps)

    # train one step
    for step, (data, _) in enumerate(data_loader):
        data = data.cuda(non_blocking=True)
        # preprocess data
        input, target = preprocess_data(data, input_len, repr_len, patch_size, use_frame_target, use_ts_target, use_next_frame_target, ts_tau)
        # forward
        output = model(input)
        # postprocess data
        pred = postprocess_data(output, patch_size, use_frame_target, use_ts_target, use_next_frame_target)
        # loss
        loss = loss_fn(pred, target)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        
        if is_master():
            # process bar
            process_bar.set_description('loss: {:.5e}'.format(loss.item()))
            process_bar.update(1)
            # log in tb
            tb_writer.add_scalar(tag='step/loss', scalar_value=loss.item(), global_step=epoch * nsteps + step)
            # visualize in tb
            if step == nsteps - 1:
                # output.shape = [batch_size, ntargets, 2, frame_size, frame_size]
                use_targets = [use_frame_target, use_ts_target, use_next_frame_target]
                target_indices = [-1 for _ in range(3)]
                target_idx = 0
                for i, use_target in enumerate(use_targets):
                    if use_target:
                        target_indices[i] = target_idx
                        target_idx += 1

                if use_frame_target:
                    target_idx = target_indices[0]
                    frame_predicted = pred[:, target_idx]    # [batch_size, 2, frame_size, frame_size]
                    frame_target = target[:, target_idx]     # [batch_size, 2, frame_size, frame_size]
                    visualize_frame_in_tb(frame_predicted, epoch, tb_writer, tag='train/pred_frame')
                    visualize_frame_in_tb(frame_target, epoch, tb_writer, tag='train/target_frame')
                if use_ts_target:
                    target_idx = target_indices[1]
                    ts_predicted = pred[:, target_idx]
                    ts_target = target[:, target_idx]
                    raise NotImplementedError('Visualization of time surface is not implemented yet')
                if use_next_frame_target:
                    target_idx = target_indices[2]
                    next_frame_predicted = pred[:, target_idx]
                    next_frame_target = target[:, target_idx]
                    visualize_frame_in_tb(next_frame_predicted, epoch, tb_writer, tag='train/pred_next_frame')
                    visualize_frame_in_tb(next_frame_target, epoch, tb_writer, tag='train/target_next_frame')

    if dist:
        total_loss = global_meters_sum(total_loss)
    
    if is_master():
        # process bar
        process_bar.close()
        # log in tb
        tb_writer.add_scalar('train/loss', total_loss / nsamples, epoch + 1)

    print('Train average loss: {:.5e}'.format(total_loss / nsamples))


def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    epoch: int,
    tb_writer: SummaryWriter,
    input_len: int,
    repr_len: int,
    patch_size: int,
    use_frame_target: bool,
    use_ts_target: bool,
    use_next_frame_target: bool,
    ts_tau: float,
    dist: bool,
):
    # validate
    model.eval()
    nsamples = len(data_loader.dataset)
    total_loss = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.cuda(non_blocking=True)
            # preprocess data
            input, target = preprocess_data(data, input_len, repr_len, patch_size, use_frame_target, use_ts_target, use_next_frame_target, ts_tau)
            # forward
            output = model(input)
            # postprocess data
            pred = postprocess_data(output, patch_size, use_frame_target, use_ts_target, use_next_frame_target)
            # loss
            loss = loss_fn(pred, target)
            total_loss += loss.item() * data.size(0)
    if dist:
        total_loss = global_meters_sum(total_loss)
    if is_master():
        tb_writer.add_scalar('val/loss', total_loss / nsamples, epoch + 1)
        
    print('Validate average loss: {:.5e}'.format(total_loss / nsamples))


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    input_len: int,
    repr_len: int,
    patch_size: int,
    use_frame_target: bool,
    use_ts_target: bool,
    use_next_frame_target: bool,
    ts_tau: float,
    output_dir: str,
    save_freq: int,
    dist: bool,
):  
    # log
    tb_writer = None
    if is_master():
        tb_writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        print('Save tensorboard logs to [{}]'.format(output_dir + '/tensorboard'))
    
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
            repr_len=repr_len,
            patch_size=patch_size,
            use_frame_target=use_frame_target,
            use_ts_target=use_ts_target,
            use_next_frame_target=use_next_frame_target,
            ts_tau=ts_tau,
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
            repr_len=repr_len,
            patch_size=patch_size,
            use_frame_target=use_frame_target,
            use_ts_target=use_ts_target,
            use_next_frame_target=use_next_frame_target,
            ts_tau=ts_tau,
            dist=dist,
        )

        epoch += 1

        # save checkpoint
        if epoch % save_freq == 0:
            checkpoint = {
                'model': model.module.state_dict() if dist else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            checkpoint_name = 'ckpt_{}.pth'.format(epoch)
            save_on_master(checkpoint, os.path.join(output_dir, 'checkpoints', checkpoint_name))
            print('Save checkpoint to [{}]'.format(output_dir + '/checkpoints/' + checkpoint_name))
    
