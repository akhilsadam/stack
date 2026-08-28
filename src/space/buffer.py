import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os

# from arch.flow import NF
# from arch.layer import MLP
# # from arch.dist import SNE as metric
# # from arch.dist import log_odds_SNE as metric

# from arch.dist import D as metric
# from space.tokens import TokenEmbedding as TE

import wandb

def dim(z):
    return np.prod(z.shape[1:])

class Buffer(nn.Module):
    def __init__(self, batch):
        super().__init__()
        self.batch = batch
        self.Q = 0
        self.queue = None

    def __call__(self, new):
        if self.queue is None:
            self.batch = max(self.batch, new.shape[0])
            self.queue = torch.empty((self.batch, *new.shape[1:]), device=new.device)

        with torch.no_grad():
            N = new.shape[0]
            self.queue[:-N] = self.queue.clone()[N:]
            self.queue[-N:] = new.detach()
            self.Q = min(self.Q + N, self.batch)

        return torch.cat([self.queue[-self.Q:-N], new], dim=0)


