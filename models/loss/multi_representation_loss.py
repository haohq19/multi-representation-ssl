import torch.nn as nn

class MultiRepresentationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        loss_fn_name = config['loss_fn']
        if loss_fn_name == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_fn_name == 'l2':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function name: {loss_fn_name}")
        
    def forward(self, preds, targets):
        """
        Args:
            output: torch.Tensor, shape [batch_size, n_targets, 2, frame_size, frame_size]
            targets: torch.Tensor, shape [batch_size, n_targets, 2, frame_size, frame_size]
        Returns:
            loss: torch.Tensor, shape []
        """
        loss = self.loss_fn(preds, targets)
        return loss
        
