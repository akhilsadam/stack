
eps = 1e-6

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