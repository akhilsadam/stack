import torch
import torch.nn as nn
import torch.nn.functional as F

# class _Affine(nn.Module):
#     def __init__(self, scale=1.0, shift=0.0):
#         super().__init__()

#         init = torch.atanh(torch.clamp(torch.tensor(scale) / 10.0, -0.999, 0.999)) * 10
#         self.raw_scale = nn.Parameter(init)
#         init = torch.atanh(torch.clamp(torch.tensor(shift) / 10.0, -0.999, 0.999)) * 10
#         self.raw_shift = nn.Parameter(init)
    
#     def forward(self, _in):
#         return torch.tanh(self.raw_scale / 10.0) * 10.0 * _in + torch.tanh(self.raw_shift / 10.0) * 10.0

class _Affine(nn.Module):
    def __init__(self, scale=1.0, shift=0.0, rate=0.1):
        super().__init__()

        self.rate = rate
        self.raw_scale = nn.Parameter(torch.tensor(scale) / self.rate)
        self.raw_shift = nn.Parameter(torch.tensor(shift) / self.rate)
    
    def forward(self, _in, *args):
        return self.raw_scale * self.rate * _in + self.raw_shift * self.rate


class AffineReLU(nn.Module):
    def __init__(self, scale=1.0, shift=0.0, rate=0.1):
        super().__init__()

        self.rate = rate
        raw_scale = torch.log(torch.tensor(scale) + 1e-8) / self.rate
        # raw_scale = torch.tensor(scale) / self.rate
        self.raw_scale = nn.Parameter(raw_scale)
        self.raw_shift = nn.Parameter(torch.tensor(shift) / self.rate)
    
    def forward(self, _in, *args):
        _scale = torch.exp(self.raw_scale * self.rate)
        # _scale = F.relu(self.raw_scale * self.rate)
        return _scale * _in + self.raw_shift * self.rate


class _AffineWNorm(nn.Module):
    def __init__(self, scale=1.0, shift=0.0, rate=0.01):
        super().__init__()

        self.rate = rate
        self.raw_scale = nn.Parameter(torch.tensor(scale) / self.rate)
        self.raw_shift = nn.Parameter(torch.tensor(shift) / self.rate)
    
    def forward(self, _in, hardness):
        # _norm = F.layer_norm(_in, (*_in.shape[1:],)) if _in.dim() > 1 else _in
        _norm = _in / (_in.pow(2).mean(dim=tuple(range(1, _in.dim())), keepdim=True) + 1e-8).sqrt() if _in.dim() > 1 else _in
        #### Better, but somehow not the same between hard and soft norm, even if I copy scale. EMA layer norm??
        return self.raw_scale * self.rate * _norm + self.raw_shift * self.rate


class PostAffine(nn.Module):
    def __init__(self, aff=_Affine):
        # self.norm = lambda x: F.layer_norm(x, (x.shape[-1],)) if len(x.shape) > 1 else x

        # self.preprocess = lambda x, *args: F.layer_norm(x, (x.shape[-1],)) if len(x.shape) > 1 else x
        # self.postprocess = _Affine() #WNorm()

        self.preprocess = lambda x, *args: x
        self.postprocess = aff() 

        # self.preprocess = _Affine()
        # self.postprocess = lambda x, *args: x