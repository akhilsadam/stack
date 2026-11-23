import torch
from torch.optim import Adam


class GradNormAdam(Adam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        target_grad_norm=1.0,
        grad_eps=1e-8,
    ):
        if target_grad_norm <= 0:
            raise ValueError(f"Invalid target_grad_norm: {target_grad_norm}")
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.target_grad_norm = target_grad_norm
        self.grad_eps = grad_eps

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norm = p.grad.norm()
                    if norm > self.grad_eps:
                        p.grad.mul_(self.target_grad_norm / norm)

        return super().step(closure=closure)