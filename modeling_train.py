import torch.nn as nn
from models.vit.vit_decoder import ViTDecoder

class TrainModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        cfg_dec = cfg['decoder']
        cfg_pretrain = cfg['pretrain']
        self.dec_name = cfg_dec['name']
        
        if self.dec_name == 'vit':
            
            in_chans = cfg_pretrain['d_hidden']
            npatches = cfg_dec['npatches']
            d_model = cfg_dec['d_model']
            depth = cfg_dec['depth']
            nheads = cfg_dec['nheads']
            num_classes = cfg_dec['num_classes']
            drop_rate = cfg_dec['drop_rate']
            attn_drop_rate = cfg_dec['attn_drop_rate']
            self.decoder = ViTDecoder(
                in_chans=in_chans,
                npatches=npatches,
                d_model=d_model,
                depth=depth,
                nheads=nheads,
                num_classes=num_classes,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
            )
        else:
            raise ValueError(f"Unsupported decoder name: {self.dec_name}")
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input):
        # input.shape = [batch_size, n_patches, d_model]
        # output.shape = [batch_size, num_classes]

        output = self.decoder(input)    # [batch_size, num_classes]

        return output