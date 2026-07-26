import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def SNE(x, k: int = 5):
	# assume input is batch tensor B ...
	# compute MSE distance matrices (with inf on diagonal), then KNN to compute sigma_i
	# then softmax to get PDFs

#     assert torch.all(torch.isfinite(x)), x

	# remove overflow
	x = torch.clamp(x, min=-1e15, max=1e15) 
	B = x.shape[0]
	x_flat = x.view(B, -1)
	# x_flat = x_flat / (torch.std(x_flat, dim=0, keepdim=True) + 1e-8)

	# ||a - b||^2 = ||a||^2 - 2<a, b> + ||b||^2
	x2 = torch.sum(x_flat**2, dim=1, keepdim=True) # B_

	assert torch.all(torch.isfinite(x2)), x_flat

	d2 = x2 - 2 * torch.matmul(x_flat, x_flat.T) + x2.T # BB
	d2 = torch.clamp(d2, min=1e-8) # 0 causes issues with logits.


	k_dist, _ = torch.topk(d2, k, dim=1, largest=False) # Bk
	# print(x_flat.mean(dim=-1), k_dist)
	# sigma2 = torch.mean(k_dist, dim=1, keepdim=True) / 2.0 # B_
	# sigma2 = torch.clamp(sigma2, min=1e-8)
	# sigma2 = sigma2.view(B, 1).detach()

	sigma2 = 0.1

	### unstable
	# d2.fill_diagonal_(1e9) # ignore self-dist
	# p = F.log_softmax(-d2 / (2 * sigma2), dim=1)
	
	logits = -d2 / (2 * sigma2)
	# assert torch.all(torch.isfinite(d2)), d2
	# assert torch.all(torch.isfinite(sigma2)), sigma2
	assert torch.all(torch.isfinite(logits)), x

	logits_masked = logits.clone().fill_diagonal_(float('-inf'))
	log_denom = torch.logsumexp(logits_masked, dim=1, keepdim=True)
	p = logits_masked - log_denom

	return p
	

def log_odds_SNE(x, *args, **kwargs):
	# assume input is batch tensor B ...
	# compute MSE distance matrices (with inf on diagonal), then KNN to compute sigma_i
	# then softmax to get PDFs

	x = torch.clamp(x, min=-1e15, max=1e15) 
	B = x.shape[0]
	x_flat = x.view(B, -1)

	x_flat = x_flat / (torch.std(x_flat, dim=1, keepdim=True) + 1e-8)
	
	

	# ||a - b||^2 = ||a||^2 - 2<a, b> + ||b||^2
	x2 = torch.sum(x_flat**2, dim=1, keepdim=True) # B_
	assert torch.all(torch.isfinite(x2)), x_flat

	d2 = x2 - 2 * torch.matmul(x_flat, x_flat.T) + x2.T # BB
	d2 = torch.clamp(d2, min=1e-8) # 0 causes issues with logits.
	
	sim = torch.sigmoid(-d2)
	log_odds = torch.arctanh(torch.clamp(sim, min=-0.999, max=0.999))

	return log_odds



def D(x, k=5, **kwargs):
	precision = 1e-4 # anything below is ignored
	# remove overflow
	# x = torch.clamp(x, min=-1e15, max=1e15)
	B = x.shape[0]
	x_flat = x.view(1, B, -1) #/ (x.shape[-1])

	x_flat = F.normalize(x_flat, p=2, dim=-1)

	x_flat = torch.where(x_flat < precision, 0, x_flat)

	d = torch.cdist(x_flat, x_flat, p=2.0)[0] # BxB

	# assert torch.all(d2 > 0), 'd2 < 0'
	
	# k_dist, _ = torch.topk(d2, k, dim=1, largest=False)
	# scale = (k_dist.mean(dim=1, keepdim=True) / 2).detach() + 1e-3

	nd = d / (torch.max(torch.abs(d)).detach() + 1e-8)
	# return nd #(d2 + 1e-8).sqrt()
	return torch.log(nd + 1e-8)
	# return torch.log(1 + nd)
	# return torch.log(torch.clamp(nd, min=1e-6))

	# return d / (d + 1e-6)
