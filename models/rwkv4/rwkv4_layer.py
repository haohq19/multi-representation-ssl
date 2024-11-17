import os
import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

T_MAX = 4096
path = os.path.dirname(__file__)
sources = [os.path.join(path, "./cuda/wkv_op.cpp"), os.path.join(path, "./cuda/wkv_cuda.cu")]
wkv_cuda = load(
    name="wkv",
    sources=sources,
    verbose=True, 
    extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}']
    )


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w = -torch.exp(w.float().contiguous())
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        wkv = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, wkv)
        return wkv

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)

        return (None, None, None, gw, gu, gk, gv)


class TokenMixing(nn.Module):
    def __init__(self, d_model, layer_id, num_layers):
        super().__init__()
        self.layer_id = layer_id
        self.d_model = d_model
        self.num_layers = num_layers

        with torch.no_grad(): # fancy init
            ratio_0_to_1 = (layer_id / (num_layers - 1))                                # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / num_layers))                        # 1 to ~0
            
            # init w
            h = torch.arange(d_model)                                                   # h = [0, 1, 2, ..., d_model-1]
            decay_speed = -5 + 8 * (h / (d_model-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.w = nn.Parameter(decay_speed)

            # init u
            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(d_model)]) * 0.5)    # zigzag = [0, 0.5, -0.5, 0, 0.5, -0.5, ...]
            self.u = nn.Parameter(torch.ones(d_model) * math.log(0.3) + zigzag)         # u = [log(0.3), log(0.3) + 0.5, log(0.3) - 0.5, log(0.3), ...]
            
            # init time_mix
            x = torch.ones(1, 1, d_model)
            for i in range(d_model):
                x[0, 0, i] = i / d_model
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
            
        self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_r = nn.Linear(d_model, d_model, bias=False)

        self.wkv = WKV.apply

        self.to_output = nn.Linear(d_model, d_model, bias=False)

        self.to_k.scale_init = 0
        self.to_r.scale_init = 0
        self.to_output.scale_init = 0


    def forward(self, x):
        # x.shape = [batch_size, num_tokens, num_channels]

        B, T, C = x.size()

        # token shift
        x_token_shifted = self.token_shift(x)
        xk = x * self.time_mix_k + x_token_shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_token_shifted * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_token_shifted * (1 - self.time_mix_r)

        # K, V, R
        k = self.to_k(xk)
        v = self.to_v(xv)
        r = self.to_r(xr)
        
        wkv = self.wkv(B, T, C, self.w, self.u, k, v)

        rwkv = torch.sigmoid(r) * wkv
        output = self.to_output(rwkv)   
        
        # output.shape = [batch_size, num_tokens, num_channels]
        return output  


class ChannelMixing(nn.Module):
    def __init__(self, d_model, layer_id, num_layers, dim_feedforward):
        super().__init__()
        self.layer_id = layer_id

        self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad(): # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / num_layers)) # 1 to ~0

            x = torch.ones(1, 1, d_model)
            for i in range(d_model):
                x[0, 0, i] = i / d_model

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        self.to_k = nn.Linear(d_model, dim_feedforward, bias=False)
        self.to_v = nn.Linear(dim_feedforward, d_model, bias=False)
        self.to_r = nn.Linear(d_model, d_model, bias=False)
        
        self.to_v.scale_init = 0
        self.to_r.scale_init = 0

    def forward(self, x):
        # x.shape = [batch_size, num_tokens, num_channels]
        
        x_token_shifted = self.token_shift(x)
        xk = x * self.time_mix_k + x_token_shifted * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_token_shifted * (1 - self.time_mix_r)

        x = self.to_k(xk)
        x = torch.square(torch.relu(x))
        x = self.to_v(x)

        output = torch.sigmoid(self.to_r(xr)) * x   
        
        # output.shape = [batch_size, num_tokens, num_channels]
        return output
    

class RWKV4Layer(nn.Module):

    def __init__(self, d_model, layer_id, num_layers, dim_feedforward):
        super().__init__()
        self.d_model = d_model
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

        self.ln0 = nn.LayerNorm(d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.token_mixing = TokenMixing(d_model=d_model, layer_id=layer_id, num_layers=num_layers)
        self.channel_mixing = ChannelMixing(d_model=d_model, layer_id=layer_id, num_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, x):
        # x.shape = [batch_size, num_tokens, num_channels]

        output = self.token_mixing(self.ln0(x))
        x = x + output
        output = self.channel_mixing(self.ln1(x))
        x = x + output

        # x.shape = [batch_size, num_tokens, num_channels]
        return x