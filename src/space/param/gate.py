import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from space.util import eps

# v0, A/B gate during training
class SigmoidGate(nn.Module):
    def __init__(self):
        init = torch.tensor(self.gate_init)
        init = (init + 0.1 * torch.rand_like(init)) / self.L
        self.gate_logit = nn.Parameter(torch.logit(init))

    def gate(self, temp):
        a = torch.sigmoid(self.gate_logit / temp)
        # a = 0.5 + 0 * a
        return a, -(a * torch.log(a + eps) + (1-a) * torch.log(1-a + eps))

    def apply_gate(self, a, value, skip):
        return a * value + (1-a) * skip

    def gate_info(self, gate_temp=1.0):
        return [torch.sigmoid(self.gate_logit / gate_temp),]

# v0.1 # wip, A & B during training
class SoftmaxGate(nn.Module):
    def __init__(self):        
        init = torch.clamp((self.gate_init + 0.2 * torch.rand((2,))), min=0.01, max=0.99)
        init = init / init.sum()
        print(init)
        self.gate_logits = nn.Parameter(torch.logit(init))        

    def gate(self, temp):
        a = torch.softmax(self.gate_logits / temp, dim=0)
        print('a', a)
        return a, -(a * torch.log(a + eps)).sum() / math.log(a.shape[0] + eps)

    def apply_gate(self, a, value, skip):
        return a[0] * value + a[1] * skip

    def gate_info(self, gate_temp=1.0):
        return F.softmax(self.gate_logits / gate_temp, dim=0)

class SimpleGate(nn.Module):
    def __init__(self):
        init = torch.tensor(self.gate_init)
        init = (init + 0.1 * torch.rand_like(init)) / self.L
        self.gate_logit = nn.Parameter(init)

    def gate(self, temp):
        a = self.gate_logit
        qa = torch.clamp(F.relu(self.gate_logit), 0, 1)
        a = a + (qa - a).detach()
        # torch.sigmoid(self.gate_logit / temp)
        # a = 0.5 + 0 * a
        return a, -(a * torch.log(a + eps) + (1-a) * torch.log(1-a + eps))

    def apply_gate(self, a, value, skip):
        return a * value + (1-a) * skip

    def gate_info(self, gate_temp=1.0):
        return [torch.sigmoid(self.gate_logit / gate_temp),]
