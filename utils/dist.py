import os
import torch
import torch.distributed as dist

def is_master():
    if not dist.is_available():  # if not distributed mode, return True
        return True
    elif not dist.is_initialized():  # if distributed mode but not initialized, return True
        return True
    else:  # if distributed mode, return True only when rank is 0
        return dist.get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_master():
        torch.save(*args, **kwargs)
        
def init_ddp(config):
    """
    Initialize Distributed Data Parallel (DDP) mode
    """
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:     # distributed mode
        config['rank'] = int(os.environ["RANK"])
        config['world_size'] = int(os.environ['WORLD_SIZE'])
        config['local_rank'] = int(os.environ['LOCAL_RANK'])
        config['dist'] = True
    else:                                                       # non-distributed mode                             
        config['dist'] = False  
        return

    print('init distributed data parallel (rank {}): {}'.format(config['rank'], config['dist_url']), flush=True)
    dist.init_process_group(
        backend=config['backend'], 
        init_method=config['dist_url'], 
        world_size=config['world_size'], 
        rank=config['rank'])
    enable_print(is_master())
    
    
def enable_print(is_master):
    '''
    This function enables printing only in the master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def global_meters_sum(*meters):
    '''
    meters: scalar values calculated in each rank
    '''
    local_rank = int(os.environ['LOCAL_RANK'])
    tensors = [torch.tensor(meter, device=local_rank, dtype=torch.float32) for meter in meters]
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if len(tensors) == 1:
        return tensors[0].item()
    else:
        return tuple(tensor.item() for tensor in tensors)