import torch
from torch.utils.tensorboard import SummaryWriter


def visualize_frame_in_tb(frame: torch.Tensor, epoch: int, tb_writer: SummaryWriter, tag: str, max_visualizations: int = None):
    """
    Visualize the event frame in tensorboard
    Args:
        frame: torch.Tensor, shape [batch_size, 2, height, width]
        epoch: int, the current epoch
        tb_writer: SummaryWriter
        tag: str, tensorboard tag
        max_visualizations: int, the maximum number of samples to visualize
    """
    batch_size, nchans, _, _ = frame.shape
    if max_visualizations is None:
        max_visualizations = batch_size
    max_visualizations = min(max_visualizations, batch_size)
    frame = frame[:max_visualizations]
    for chan in range(nchans):
        tb_writer.add_images(tag=tag + f'_{chan}', img_tensor=frame[:, chan].unsqueeze(1), global_step=epoch + 1, dataformats='NCHW')
