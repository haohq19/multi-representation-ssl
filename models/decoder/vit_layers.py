import numpy as np
import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc0 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc1 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc0(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc1(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, input):
        # input.shape = [batch_size, num_patches, d_model]
        # output.shape = [batch_size, num_patches, d_model]
        
        B, N, C = input.shape
        qkv = self.qkv(input)                                                   # [batch_size, num_patches, d_model * 3]
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # [3, batch_size, num_heads, num_patches, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]                                        # [batch_size, num_heads, num_patches, head_dim]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))                                        # [batch_size, num_heads, num_patches, num_patches]

        
        attn = attn.softmax(dim=-1)                                             # [batch_size, num_heads, num_patches, num_patches]
        attn = self.attn_drop(attn)                                             # [batch_size, num_heads, num_patches, num_patches]

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)                        # [batch_size, num_patches, d_model]
        x = self.proj(x)                                                        # [batch_size, num_patches, d_model]
        output = self.proj_drop(x)                                              # [batch_size, num_patches, d_model]

        return output                                                         


class Block(nn.Module):

    def __init__(self, d_model, num_heads, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.norm0 = norm_layer(d_model)
        self.attention = Attention(d_model, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(d_model)
        self.ffn = FFN(in_features=d_model, hidden_features=d_model * 4, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x.shape = [batch_size, num_patches, d_model]

        x = x + self.attention(self.norm0(x))
        x = x + self.ffn(self.norm1(x))
        
        return x

    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(num_patches, d_model):

    def get_position_angle_vec(position):
        # position.shape = [1]
        return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_patches)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1
    
    # sinusoid_table.shape = [num_patches, d_model]
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)   # [1, num_patches, d_model]