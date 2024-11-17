import torch.nn as nn
from models.rwkv4.rwkv4 import RWKV4
from models.event_tokenizer.event_tokenizer import EventTokenizer

class PretrainModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        config_tokenizer = config['tokenizer']
        config_model = config['model']
        self.tokenizer_name = config_tokenizer['name']
        self.model_name = config_model['name']  

        if self.tokenizer_name == 'event_tokenizer':
            self.tokenizer = EventTokenizer(config_tokenizer)
        else:
            raise ValueError(f"Unsupported tokenizer name: {self.tokenizer_name}")
        
        if self.model_name == 'rwkv4':
            self.model = RWKV4(config_model)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        

    def forward(self, input):
        # input.shape = [batch_size, n_events, 4]
        # output.shape = [batch_size, num_classes]

        x = self.tokenizer(input)   # [batch_size, n_events, d_model]
        output = self.model(x)      # [batch_size, num_classes]

        return output