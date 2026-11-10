import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from space.util import eps

class SoftmaxMux(nn.Module):
    def __init__(self):

        # self.term_logits = nn.Parameter(0.1 * torch.ones(self.n) / self.n)
        self.term_logits = nn.Parameter(0.1 * torch.rand(self.n))
        if self.token_id >= 0: 
            self.term_logits.data.fill_(0.0)
            self.term_logits.data[self.token_id] = 1.0

    def term_entropy(self, p):
        return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

    def p(self, temp):
        logits = self.term_logits
        logits = logits / temp 
        # logits = logits * 0.001 / temp # doesn't work even with lower temp, softmax pushes
        p = F.softmax(logits, dim=0)
        hp = F.softmax(logits / 1e-5, dim=0)

        # p = F.gumbel_softmax(logits, dim=0, tau=temp, hard=True)

        entropy = self.term_entropy(p)
        return p, hp, entropy

    def channel_ids(self, *args, **kwargs):
        temp = kwargs.get('temp', 1e-4)
        logits_valid = self.term_logits
        p = F.softmax(logits_valid / temp, dim=0)
        ids = torch.where(p > 0.1)[0]
        return ids.tolist(), p[ids]


class SigmoidMux(nn.Module):
    def __init__(self):

        # self.term_logits = nn.Parameter(0.1 * torch.ones(self.n) / self.n)
        # self.term_logits = nn.Parameter(0.0001 * torch.rand(self.n)) # init temp not important, need temp scaling
        self.term_logits = nn.Parameter(0.1 * (torch.rand(self.n) - 0.5))
        # self.term_logits = nn.Parameter(0.1 * torch.rand(self.n))

        if self.token_id >= 0: 
            self.term_logits.data.fill_(0.0)
            self.term_logits.data[self.token_id] = 1.0

        # self.term_logits.scaling_factor = (1/6) * self.m * self.L

    def term_entropy(self, p):
        return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

    def p(self, temp):
        logits = self.term_logits * 0.001
        logits = logits / temp
        p = torch.sigmoid(logits)
        p = p / p.sum()

        hp = torch.sigmoid(logits / 1e-5)
        hp = hp / hp.sum()
        # print(p)

        entropy = self.term_entropy(p)
        return p, hp, entropy

    def channel_ids(self, *args, **kwargs):
        temp = kwargs.get('temp', 1.0)
        logits_valid = self.term_logits * 0.001
        p = torch.sigmoid(logits_valid / temp)
        ids = torch.where(p > 0.1)[0]
        return ids.tolist(), p[ids], 1.0 + 0.0 * p[ids]

class TernaryMuxV4(nn.Module): # 
    def __init__(self):

        # self.term_logits = nn.Parameter(torch.zeros(self.n))
        self.weights = nn.Parameter(torch.ones(self.n) * 0.01) 

    def term_entropy(self, p):
        return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

    def p(self, temp):

        q = self.weights

        a = q
        ha = q
        # q = self.quantize(self.weights)

        entropy = 0.0 #self.term_entropy(q.abs())
        return a, ha, entropy

    def channel_ids(self, *args, **kwargs):
        cache_mag = kwargs.get('cache_mag', 1.0)
        temp = kwargs.get('temp', 1.0)
        p = self.weights.abs() * (cache_mag > 1e-10)
        # print("token_p", p, self.term_logits)

        ids = torch.where(p > 1e-6)[0]
        return ids.tolist(), p[ids], self.weights[ids]

class TernaryMuxV3(nn.Module): # 
    def __init__(self):

        self.term_logits = nn.Parameter(torch.zeros(self.n))
        self.weights = nn.Parameter(torch.ones(self.n) * 0.1) 

    def term_entropy(self, p):
        return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

    def quantize(self, w, p):
        # https://arxiv.org/pdf/2402.17764
        # (mod) BitNet b1.58
        y = min(w.abs().mean(), 1e-8)
        w_q = w / y
        w_q = w_q + p * (torch.round(w_q) - w_q).detach()
        w_q = w_q * y
        w_q = w_q + p * (torch.clamp(w_q, -1.0, 1.0) - w_q).detach()

        return w_q

    def p(self, temp):
        p = torch.sigmoid(self.term_logits * 0.001 / temp - 10)
        q = self.quantize(self.weights, p)

        a = q
        ha = q

        entropy = 0.0 # (1-p).sum() # 0.0 #self.term_entropy(a.abs())
        return a, ha, entropy

    def channel_ids(self, *args, **kwargs):
        cache_mag = kwargs.get('cache_mag', 1.0)
        temp = kwargs.get('temp', 1.0)
        w = self.quantize(self.weights, 0.0)
        p = w.abs() * (cache_mag > 0.01)
        # print("token_p", p, self.term_logits)
        ids = torch.where(p > 1e-6)[0]
        return ids.tolist(), 1.0 + 0.0 * p[ids], w[ids]


      # logits = self.term_logits * 0.001
        # logits = logits / temp

        # p = torch.sigmoid(1/temp - 10) # not bad, but quant can get noisy
        # q = self.weights * p + self.quantize(self.weights) * (1 - p)


        # p = torch.sigmoid(1 / temp - 10)
        # q = self.quantize(self.weights, p)

# class TernaryMux(nn.Module): # WORKS! but messy parametrization [UPDATE: depends on weight init, so not great; weights can get stuck]
#     def __init__(self):

#         self.term_logits = nn.Parameter(0.1 * (torch.rand(self.n) - 0.5))
#         self.weights = nn.Parameter(0.1 * (torch.rand(self.n) - 0.5)) # works might be special case, somehow this learns???
#         # self.weights = nn.Parameter(1e-5 * torch.randn(self.n)) # ones, zeros fails, not great...

#         if self.token_id >= 0: 
#             self.term_logits.data.fill_(0.0)
#             self.term_logits.data[self.token_id] = 1.0

#         # self.term_logits.scaling_factor = (1/6) * self.m * self.L

#     def term_entropy(self, p):
#         return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

#     def quantize(self, w):
#         # https://arxiv.org/pdf/2402.17764
#         # (exactly as in) BitNet b1.58
#         y = w.abs().mean()
#         w_q = torch.round(w / (y + 1e-10))
#         w_q = torch.clamp(w_q, -1.0, 1.0)
#         return w_q

#     def p(self, temp):
#         logits = self.term_logits * 0.001
#         logits = logits / temp
#         p = torch.sigmoid(logits)
#         q = self.quantize(self.weights)
#         a = p * q

#         hp = torch.sigmoid(logits / 1e-5)
#         ha = hp * q

#         entropy = self.term_entropy(a.abs())
#         return a, ha, entropy

#     def channel_ids(self, *args, **kwargs):
#         cache_mag = kwargs.get('cache_mag', 1.0)
#         temp = kwargs.get('temp', 1.0)
#         p = torch.sigmoid(self.term_logits * 0.001 / temp) * self.quantize(self.weights).abs() * (cache_mag > 0.01)
#         # print("token_p", p, self.term_logits)
#         ids = torch.where(p > 0.1)[0]
#         return ids.tolist(), p[ids]


# class TernaryMux(nn.Module):
#     def __init__(self):

#         self.term_logits = nn.Parameter(0.1 * (torch.rand(self.n) - 0.5))
#         self.sign_logits = nn.Parameter(0.1 * (torch.rand(self.n) - 0.5))

#         if self.token_id >= 0: 
#             self.term_logits.data.fill_(0.0)
#             self.term_logits.data[self.token_id] = 1.0

#         # self.term_logits.scaling_factor = (1/6) * self.m * self.L

#     def term_entropy(self, p):
#         return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

#     def p(self, temp):
#         logits = self.term_logits * 0.001
#         logits = logits / temp
#         p = torch.sigmoid(logits)
#         # sign_p = self.sign_logits# ## currently redundant w.r.t affine

#         # print(p)
#         entropy = self.term_entropy(p) #+ self.term_entropy(sign_p)

#         # sign = self.sign_logits # 2 * sign_p - 1

#         sign = self.sign_logits #+ 0.2 * (torch.round(self.sign_logits) - self.sign_logits).detach()

#         a = p * sign

#         return a, entropy

#     def channel_id(self):
#         return torch.argmax(self.term_logits).item()

# class TernaryMux(nn.Module):
#     def __init__(self):

#         self.term_logits = nn.Parameter(0.1 * (torch.rand(self.n) - 0.5))
        
#         if self.token_id >= 0: 
#             self.term_logits.data.fill_(0.0)
#             self.term_logits.data[self.token_id] = 1.0

#         # self.term_logits.scaling_factor = (1/6) * self.m * self.L

#     def term_entropy(self, p):
#         return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

#     def f(self, x, t, a=1e-4):
#         # # smooth interpolation between sigmoid and double step
#         u = 0.5 * (F.tanh((x-a)/t) + F.tanh((x+a)/t))
#         tm = a * torch.clamp(t, max=1.0)
#         s = F.tanh(x / tm)
#         p = F.tanh(t)
#         f = u * (1-p) + s * p
#         # return f

#         return F.tanh(x/t) # sigmoid

#     def p(self, temp):
#         logits = self.term_logits * 0.001 / temp
#         # p = logits 
#         # p = self.f(logits, torch.tensor(temp))
#         p = F.tanh(logits)

#         _p = p.abs()
#         _p = _p / _p.sum()
        
#         # logits = self.term_logits * 0.001
#         # logits = logits / temp
#         # _p = torch.sigmoid(logits)
#         # _p = _p / _p.sum()

#         entropy = self.term_entropy(_p) # prob of nonzero
#         # notice it cannot be + and - simultaneously so this is equivalent to ternary prob.
#         # because + and - in mixture will collapse to 0.

#         return p, entropy

#     def channel_id(self):
#         p = self.f(self.term_logits, torch.tensor(1e-4))
#         p_nonz = p.abs()
#         p_nonz = p_nonz / p_nonz.sum()

#         return torch.argmax(p_nonz).item()

# class TernaryMux(nn.Module):
#     def __init__(self):

#         self.term_logits = nn.Parameter(0.1 * torch.rand((self.n,3)))
#         self.states = torch.tensor([0.0, -1.0, 1.0])[None,:]
#         # problem with this is p(-1.0) = p(1.0) equiv. to p(0.0) = 1...

#         if self.token_id >= 0: 
#             self.term_logits.data.fill_(0.0)
#             self.term_logits.data[self.token_id] = torch.tensor([0.0, 0.0, 1.0])

#         # self.term_logits.scaling_factor = (1/6) * self.m * self.L

#     def term_entropy(self, p):
#         return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

#     def p(self, temp):
#         logits = self.term_logits * 0.001
#         logits = logits / temp
#         _p = F.softmax(logits, dim=-1)

#         entropy = self.term_entropy(_p)
#         p = (_p * self.states).sum(dim=-1)
#         return p, entropy

#     def channel_id(self):
#         logits = self.term_logits
#         _p = F.softmax(logits, dim=-1)
#         p_nonz = _p[:,1:].sum(dim=-1)
#         p_nonz = p_nonz / p_nonz.sum()

#         return torch.argmax(p_nonz).item()

# class TernaryMux(nn.Module):
#     def __init__(self):

#         self.term_logits = nn.Parameter(0.1 * torch.rand(self.n))
#         self.sign_logits = nn.Parameter(0.1 * torch.rand(self.n))

#         if self.token_id >= 0: 
#             self.term_logits.data.fill_(0.0)
#             self.term_logits.data[self.token_id] = 1.0

#         # self.term_logits.scaling_factor = (1/6) * self.m * self.L

#     def term_entropy(self, p):
#         return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

#     def p(self, temp):
#         logits = self.term_logits * 0.001
#         logits = logits / temp
#         p = torch.sigmoid(logits)

#         # print(p)
#         entropy = self.term_entropy(p)

#         sign_p = torch.sigmoid(self.sign_logits / temp) ## currently redundant w.r.t affine when temp is high (can be float then, do STE??)
#         sign = 2 * sign_p - 1

#         p = p * sign

#         return p, entropy

#     def channel_id(self):
#         return torch.argmax(self.term_logits).item()