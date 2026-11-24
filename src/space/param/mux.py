import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from space.util import eps
from space.report import MuxReport, select_channels

from .mag import alive_mask

def _simplex_report(p, hp, label):
    """Report of a probabilistic mux: the channels that carry real probability."""
    ids = torch.where(p > 0.1)[0]
    return MuxReport(
        ids=ids.tolist(),
        weights=p[ids],
        probs=p[ids],
        hard_weights=hp[ids],
        label=label,
    )

def _signed_report(weights, alive, label):
    """Report of a signed weight mux: it has no probability, the weight *is* the answer."""
    ids, w, mask = select_channels(weights, alive)
    return MuxReport(ids=ids, weights=w, hard_weights=w, mask=mask, label=label)


class SoftmaxMux(nn.Module):
    def __init__(self):
        self.term_logits = nn.Parameter(0.1 * torch.rand(self.n))
        if self.token_id >= 0:
            self.term_logits.data.fill_(0.0)
            self.term_logits.data[self.token_id] = 1.0

    def term_entropy(self, p):
        return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

    def p(self, working_set, temp=1.0):
        logits = self.term_logits / temp
        p = F.softmax(logits, dim=0)
        hp = F.softmax(logits / 1e-5, dim=0)

        entropy = self.term_entropy(p)
        self._cached_p = p.detach()
        self._cached_hp = hp.detach()
        return p, hp, entropy

    def channel_report(self, *args, **kwargs):
        return _simplex_report(self._cached_p, self._cached_hp, __class__.__name__)


class SigmoidMux(nn.Module):
    def __init__(self):
        self.term_logits = nn.Parameter(0.1 * (torch.rand(self.n) - 0.5))
        if self.token_id >= 0:
            self.term_logits.data.fill_(0.0)
            self.term_logits.data[self.token_id] = 1.0

    def term_entropy(self, p):
        return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

    def p(self, working_set, temp=1.0):
        logits = self.term_logits * 0.001 / temp
        p = torch.sigmoid(logits)
        p = p / p.sum()

        hp = torch.sigmoid(logits / 1e-5)
        hp = hp / hp.sum()

        entropy = self.term_entropy(p)
        self._cached_p = p.detach()
        self._cached_hp = hp.detach()
        return p, hp, entropy

    def channel_report(self, *args, **kwargs):
        return _simplex_report(self._cached_p, self._cached_hp, __class__.__name__)


class TernaryMuxV4(nn.Module):
    def __init__(self):
        self.weights = nn.Parameter(torch.ones(self.n) * 0.0001)

    def term_entropy(self, p):
        return -(p * torch.log(p + eps)).sum() / math.log(p.shape[0] + eps)

    def p(self, working_set, temp=1.0):
        self._cached_alive = alive_mask(working_set, 1e-10, spatial=getattr(self, 'SPATIAL', False)).detach()

        q = self.weights
        a = q
        ha = q
        entropy = 0.0
        return a, ha, entropy

    def channel_report(self, *args, **kwargs):
        return _signed_report(self.weights, self._cached_alive, __class__.__name__)


def _spatial_shape(grid):
    """The spatial part of a grid: everything after the batch and channel.

    (B, C, H, W) -> (H, W);  (B, W) -> (W,)
    """
    g = tuple(grid)
    if len(g) >= 3:
        return g[2:]
    return g[1:]


class TernaryMuxV4Spatial(nn.Module):
    """Like :class:`TernaryMuxV4`, but the weights vary over space.

    The weights are ``(n, *spatial)``: one field per channel, so the mux can
    pick a different channel in different places.  The operator mixes over
    the channel axis, so :meth:`p` hands the field back with the channel
    axis last -- everything else about the mux is unchanged.
    """

    SPATIAL = True

    def __init__(self):
        self.spatial_shape = _spatial_shape(self.shape)
        self.weights = nn.Parameter(torch.ones(self.n, *self.spatial_shape) * 0.01)

    def p(self, working_set, temp=1.0):
        self._cached_alive = alive_mask(working_set, 1e-10, spatial=True).detach()

        q = self.weights
        a = q
        ha = q
        entropy = 0.0
        return self._op_axis(a), self._op_axis(ha), entropy

    def _op_axis(self, x):
        """(n, *spatial) -> (*spatial, n): the operator mixes over the last axis."""
        return x.movedim(0, -1)

    def channel_report(self, *args, **kwargs):
        return _signed_report(self.weights, self._cached_alive, __class__.__name__)


class TernaryMuxV3(nn.Module):
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

    def p(self, working_set, temp=1.0):
        self._cached_alive = alive_mask(working_set, 0.01, spatial=getattr(self, 'SPATIAL', False)).detach()

        prob = torch.sigmoid(self.term_logits * 0.001 / temp - 10)
        self._cached_prob = prob.detach()
        q = self.quantize(self.weights, prob)

        a = q
        ha = q
        entropy = 0.0
        return a, ha, entropy

    def channel_report(self, *args, **kwargs):
        ids, weights, mask = select_channels(self.weights, self._cached_alive)
        return MuxReport(
            ids=ids,
            weights=weights,
            probs=self._cached_prob[ids],
            hard_weights=self.quantize(self.weights, 0.0)[ids],
            mask=mask,
            label=__class__.__name__,
        )

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