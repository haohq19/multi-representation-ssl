import torch.nn as nn
from models.rwkv4.rwkv4 import RWKV4
from models.event_tokenizer.event_tokenizer import EventTokenizer

class PretrainModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        cfg_tok = cfg['tokenizer']
        cfg_enc = cfg['encoder']
        self.tokenizer_name = cfg_tok['name']
        self.encoder_name = cfg_enc['name']

        if self.tokenizer_name == 'event_tokenizer':
            patch_size = cfg['patch_size']
            d_embed = cfg_tok['d_embed']
            self.tokenizer = EventTokenizer(
                patch_size=patch_size,
                d_embed=d_embed,
            )
        else:
            raise ValueError(f"Unsupported tokenizer name: {self.tokenizer_name}")
        
        if self.encoder_name == 'rwkv4':
            d_model = cfg_enc['d_model']
            depth = cfg_enc['depth']
            d_ffn = cfg_enc['d_ffn']
            patch_size = cfg['patch_size']
            use_frame_target = cfg['use_frame_target']
            use_next_frame_target = cfg['use_next_frame_target']
            use_ts_target = cfg['use_ts_target']
            ntargets = use_frame_target + use_next_frame_target + use_ts_target
            num_classes = ntargets * 2 * patch_size * patch_size
            cfg_enc['num_classes'] = num_classes
            self.encoder = RWKV4(
                d_model=d_model,
                depth=depth,
                d_ffn=d_ffn,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unsupported encoder name: {self.encoder_name}")
        
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input, return_hidden=False):
        # input.shape = [batch_size, nevents, 4]
        # output.shape = [batch_size, num_classes]

        x = self.tokenizer(input)   # [batch_size, nevents, d_model]

        if return_hidden:
            output, hidden = self.encoder(x, return_hidden=True)
            # output.shape = [batch_size, num_classes]
            # hidden.shape = [batch_size, d_hidden = depth * d_model]
            return output, hidden   
        
        output = self.encoder(x)      
        # output.shape = [batch_size, num_classes]
        return output