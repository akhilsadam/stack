import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _flatten(x):
    return x.reshape(x.shape[0], -1)

# based on SGD_RWR
# global version v1
# depth version v1

class Search(nn.Module):
    """Method 1: Generative Modeling via Drifting in Token Space.

    Evolves continuous token representations using an attraction-repulsion
    kernel drifting field V(x) projected through manifold round-trips.
    """

    def __init__(
        self,
        tokenizer,
        evaluator=None,
        steps=100,
        pop_size=64,
        noise_std=0.75,
        lr=1e-1,
        tau=0.05,
        sigma=1.0,
        log_every=1,
    ):
        super().__init__()
        self.tok = tokenizer
        self.eval = evaluator or tokenizer._eval
        self.steps = steps
        self.pop_size = pop_size
        self.noise_std = noise_std
        self.lr = lr
        self.tau = tau
        self.sigma = sigma
        self.log_every = log_every
        self._fixed_q = None

    # ── helpers ─────────────────────────────────────────────────────

    def _fix_state(self):
        with torch.no_grad():
            self.eval.random_state()
            self._fixed_q = self.eval.q_phys.clone()

    def _eval_fields(self, strings):
        self.eval.q_phys = self._fixed_q
        with torch.no_grad():
            return torch.stack([self.eval.eval_one(s) for s in strings], dim=0)

    def _target_field(self, rpn: str | None = None):
        if rpn is not None and rpn != self.eval.target_pde:
            self.eval.target_pde = rpn
        self.eval.q_phys = self._fixed_q
        return self.eval.target().detach()

    def _round_trip(self, x: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """Snap off-manifold continuous vectors back to nearest codebook embeddings."""
        with torch.no_grad():
            strings = self.tok.token_embed.reverse(x.detach())
            x_rounded = self.tok.token_embed(strings).to(x.device)
        return x_rounded, strings

    def _compute_drift_field(
        self,
        center: torch.Tensor,
        pop_rounded: torch.Tensor,
        losses: torch.Tensor,
    ) -> torch.Tensor:
        """Computes kernel drifting field V(center) pulling toward low-loss candidates."""
        center_flat = center.reshape(1, -1)
        pop_flat = pop_rounded.reshape(self.pop_size, -1)

        # 1. Pairwise distance & RBF Kernel relative to search center
        dists_sq = torch.sum((pop_flat - center_flat) ** 2, dim=-1)
        kernel = torch.exp(-dists_sq / (2 * (self.sigma**2)))

        # 2. Fitness-weighted attraction (softmax over negative PDE losses)
        weights = torch.softmax(-losses / self.tau, dim=0)

        # 3. Drifting field: V(center) = \sum_j w_j * K(center, x_j) * (x_j - center)
        diffs = pop_rounded - center
        drift_vec = (weights.view(-1, 1, 1) * kernel.view(-1, 1, 1) * diffs).sum(
            dim=0, keepdim=True
        )

        return drift_vec

    # ── main entry point ────────────────────────────────────────────

    def search(self, init_pde: str, target_pde: str | None = None):
        device = next(self.tok.parameters()).device

        self._fix_state()
        target = self._target_field(target_pde)
        target_flat = target.reshape(1, -1)

        best_loss = float("inf")
        best_string = init_pde

        # Continuous search state initialized at baseline string embedding
        tok_p = self.tok.token_embed([init_pde]).to(device).detach()

        for step in tqdm(range(self.steps), desc="Drifting Search"):
            # 1. Perturb continuous search state
            eps = torch.randn(self.pop_size, *tok_p.shape[1:], device=device)
            x_cand = tok_p + self.noise_std * eps

            # 2. Snap continuous perturbed vectors to discrete syntax manifold
            x_rounded, strings = self._round_trip(x_cand)

            # 3. Evaluate PDE fitness on snapped RPN expressions
            fields = self._eval_fields(strings)
            pde_losses = F.mse_loss(
                _flatten(fields),
                target_flat.expand(self.pop_size, -1),
                reduction="none",
            ).mean(dim=1)

            # Track global best string and loss
            min_idx = torch.argmin(pde_losses).item()
            if pde_losses[min_idx].item() < best_loss:
                best_loss = pde_losses[min_idx].item()
                best_string = strings[min_idx]

            # 4. Compute attraction vector V(tok_p) via kernel drifting field
            drift_vector = self._compute_drift_field(
                tok_p, x_rounded, pde_losses
            )

            # 5. Step continuous state along continuous drifting field
            tok_p = tok_p + self.lr * drift_vector

            # 6. Logging & center trajectory evaluation
            if self.log_every and step % self.log_every == 0:
                _, s_center = self._round_trip(tok_p)
                f_center = self._eval_fields(s_center)
                cur_loss = F.mse_loss(_flatten(f_center), target_flat).item()

                tqdm.write(
                    f"step {step:4d}  |  center_loss {cur_loss:.4e}  "
                    f"|  best {best_loss:.4e}"
                )
                tqdm.write(f"  best: {best_string[:100]}")

                if cur_loss < 1e-5:
                    break

        return best_string, best_loss