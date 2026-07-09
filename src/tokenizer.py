import torch
import torch.nn as nn
import torch.nn.functional as F

# import qg

import arch.flow.NF as NF
import arch.layer.MLP as MLP
import arch.dist.SNE as metric

class Tokenizer(nn.Module):
    def __init__(self, batch, seq_len, dim, steps=5):
        self.steps = steps
        self.seq_len = seq_len
        self.dim = dim
        self.batch = batch
        
        self.f_z = MLP(seq_len, dim)
        self.f_n = MLP(seq_len, dim)
        # assumed within standard normal distribution
        
        self.crit = nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    def noise(self, z):
        return torch.randn_like(z)
    
    def _noise(self):
        return torch.randn((self.batch, self.seq_len, self.dim), self.f_z.device)
        
    def forward(self, tok):
        return NF._fwd(tok, self.f_z, self.f_n)
    
    def reverse(self, z, n):
        return NF._rev(z, n, self.f_z, self.f_n)
    
    ## loss
    
    # train from random vector or token?
    def loss(self, value_function, metric):
        with torch.no_grad():
            sample_vector = self._noise()
            token = self.sample_token_from_vector(sample_vector)
        
        z, n = self.forward(token)
        
        d = metric(value_function(token))
        d_hat = metric(z)
        
        return self.crit(d_hat, d) # z-metric
    
    ## eval
        
    def token_to_vector(self, tok):
        return self.forward(tok)[0]
    
    def sample_token_from_vector(self, z):
        n = self.noise(z)
        return self.reverse(z, n)
        
        