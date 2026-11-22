
eps = 1e-12

def create_optimizer(model, optimizer, base_lr=1e-3):
    param_groups = {}
    
    for param in model.parameters():
        if not param.requires_grad:
            continue

        lr = getattr(param, 'scaling_factor', 1.0) * base_lr
        
        if lr not in param_groups:
            param_groups[lr] = []
        param_groups[lr].append(param)
    
    optimizer_list = [{"params": params, "lr": lr} for lr, params in param_groups.items()]
    
    return optimizer(optimizer_list)

import torch
def logdet_si_loss(W, eps=1e-5):
    W = torch.movedim(W, (0, 1), (-2, -1)) # first two dims only

    # W has shape (..., M, N)
    if W.shape[-2] < W.shape[-1]:
        W = W.mT  # Orient to (..., max(M, N), min(M, N))

    # Normalize each matrix in the batch independently
    fro_norm = torch.linalg.matrix_norm(W, ord='fro', keepdim=True)
    W_norm = W / (fro_norm + 1e-8)

    # Batched Gram matrix: (..., min(M, N), min(M, N))
    gram = W_norm.mT @ W_norm

    I = torch.eye(gram.shape[-1], device=W.device, dtype=W.dtype)
    _, logdet = torch.linalg.slogdet(gram + eps * I)

    # Average across batch dimensions
    return -logdet.mean()