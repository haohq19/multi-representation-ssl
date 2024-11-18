import torch
from torch.utils.tensorboard import SummaryWriter


def visualize_frame_in_tb(frame: torch.Tensor, epoch: int, tb_writer: SummaryWriter, tag: str, max_out=64):
    """
    Visualize the event frame in tensorboard
    Args:
        frame: torch.Tensor, shape [batch_size, 2, height, width]
        epoch: int, the current epoch
        tb_writer: SummaryWriter
        tag: str, tensorboard tag
        max_out: int, the maximum number of frames to visualize
    """
    batch_size, n_channels, height, width = frame.shape
    max_outs = min(max_out, batch_size)
    frame = frame[0, :max_outs]
    for chan in range(n_channels):
        tb_writer.add_images(tag=tag + f'_{chan}', img_tensor=frame[:, chan].unsqueeze(1), global_step=epoch + 1, dataformats='NCHW')
