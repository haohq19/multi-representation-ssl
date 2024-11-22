import torch.nn as nn
from models.vit.vit_decoder import ViTDecoder

class TransferModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        config_model = config['model']
        self.model_name = config_model['name']
        
        if self.model_name == 'vit':
            self.model = ViTDecoder(config_model)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input):
        # input.shape = [batch_size, n_patches, d_model]
        # output.shape = [batch_size, num_classes]

        output = self.model(input)      # [batch_size, num_classes]

        return output