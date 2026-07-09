import torch
import torch.nn as nn
import torch.nn.functional as F

def SNE(x, k: int = 5):
     # assume input is batch tensor B ...
    # compute MSE distance matrices (with inf on diagonal), then KNN to compute sigma_i
    # then softmax to get PDFs

    B = x.shape[0]
    x_flat = x.view(batch_size, -1)
    
    # ||a - b||^2 = ||a||^2 - 2<a, b> + ||b||^2
    sum_x_sq = torch.sum(x_flat**2, dim=1, keepdim=True) # B_
    distances_sq = sum_x_sq - 2 * torch.matmul(x_flat, x_flat.T) + sum_x_sq.T # BB
    distances_sq.fill_diagonal_(float('inf')) # ignore self-dist

    k_nearest_distances, _ = torch.topk(distances_sq, k, dim=1, largest=False) # Bk
    sigma2 = torch.mean(k_nearest_distances, dim=1, keepdim=True) / 2.0 # B_
    sigma2 = torch.clamp(sigma, 1e-8, 1e9)
    sigma2 = sigma2.view(batch_size, 1)

    p = F.log_softmax(-distances_sq / (2 * sigma_2), dim=1)
    return p
    
    