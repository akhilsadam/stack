from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from arch.dist import D as metric


def _flatten(x):
    return x.reshape(x.shape[0], -1)

# based on flow matching with velocity
# global version v0
# depth version v0


class Search(nn.Module):
    """Online local-linear search with fine-tuned flow manifold.

    Core idea: the tokenizer's alignment loss  MSE(D(z), D(v))  is what
    makes the flow's latent space locally linear w.r.t. PDE fields.
    During search we **fine-tune the flow** on collected (string, field)
    pairs, so that D(z) ≈ D(v) holds in the region of interest.  This
    gives the ridge-regression linear model a meaningful signal.

    Each outer step:
    1. Perturb ``z`` in the flow's latent space, decode → evaluate → collect.
    2. **Fine-tune** the tokenizer on the replay buffer (alignment + commit loss).
    3. Re-encode the current best through the updated flow.
    4. Fit a local linear model  δlog(L) ≈ W · ε  via dual ridge regression.
    5. Step ``z`` along ``-W``.
    """

    def __init__(
        self,
        tokenizer,
        evaluator=None,
        steps=200,
        pop_size=64,
        noise_std=0.5,
        lr=1e-2,
        log_every=10,
        buffer_size=128,
        train_every=1,
        train_steps=20,
        train_batch=8,
        train_lr=3e-4,
        tau = 0.1       # Temperature for softmax weighting
    ):
        super().__init__()
        self.tok = tokenizer
        self.eval = evaluator or tokenizer._eval
        self.steps = steps
        self.pop_size = pop_size
        self.noise_std = noise_std
        self.lr = lr
        self.log_every = log_every
        self.buffer_size = buffer_size
        self.train_every = train_every
        self.train_steps = train_steps
        self.train_batch = train_batch
        self.tau = tau

        # Optimiser for online fine-tuning (separate from search)
        self.fine_opt = torch.optim.Adam(self.tok.parameters(), lr=train_lr)

        self._fixed_q = None

        self.kcrit = nn.KLDivLoss(reduction='batchmean', log_target=True)

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

    def _encode(self, rpn: str):
        tok = self.tok.token_embed([rpn])
        z, n = self.tok.forward(tok)
        return z[0:1].detach(), n[0:1].detach()

    def _decode(self, z, n):
        tok_cont = self.tok.reverse(z, n)
        with torch.no_grad():
            strings = self.tok.token_embed.reverse(tok_cont.detach())
        return strings, tok_cont

    def _round_trip(self, x: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """Projects continuous embeddings back onto the discrete codebook manifold."""
        with torch.no_grad():
            strings = self.tok.token_embed.reverse(x.detach())
            x_rounded = self.tok.token_embed(strings).to(x.device)
        return x_rounded, strings

    def _fine_tune(self, strings, fields):
        """Fine-tune the tokenizer so D(z) ≈ D(v) for collected pairs."""
        self.tok.train()
        B = len(strings)
        for _ in range(self.train_steps):
            idx = torch.randperm(B)[: min(self.train_batch, B)]
            batch_str = [strings[i] for i in idx]
            batch_fld = fields[idx]

            tok = self.tok.token_embed(batch_str)              # (B, S, dim)
            z, n = self.tok.forward(tok)                       # (B, S, dim)

            # Alignment loss: D(z) ≈ D(v)
            d_z = metric(_flatten(z))
            d_v = metric(_flatten(batch_fld))
            align = F.mse_loss(d_z, d_v)

            # Commit loss: one-step reversibility
            mse = F.mse_loss
            n2 = torch.randn_like(z) / z.shape[1:].numel()
            tok2 = self.tok.reverse(z, n2)
            tok3 = self.tok.token_embed(self.tok.token_embed.reverse(tok2))
            z2, _ = self.tok.forward(tok3)
            commit = mse(tok2, tok3) + mse(z, z2)

            n_logits = F.log_softmax(self.tok.noise(n).view(B, -1), dim=1)
            loss_dist = self.kcrit(F.log_softmax(n.view(B, -1), dim=1), n_logits) \
                        + self.kcrit(F.log_softmax(z.view(B, -1), dim=1), n_logits)   

            loss = align + 0.1 * commit + 0.1 * loss_dist
            self.fine_opt.zero_grad()
            loss.backward()
            self.fine_opt.step()

        self.tok.eval()

    # ── main entry point ────────────────────────────────────────────

    def search(self, init_pde: str, target_pde: str | None = None):
        """Predict-x1 Flow Matching search with round-trip manifold projection."""
        device = next(self.tok.parameters()).device

        self._fix_state()
        target = self._target_field(target_pde)
        target_flat = target.reshape(1, -1)

        best_loss = float("inf")
        best_string = init_pde

        # Initialize current state on the valid embedding manifold
        tok_p = self.tok.token_embed([init_pde]).to(device).detach()

        for step in tqdm(range(self.steps), desc="Flow Matching Search"):
            # 1. Sample continuous perturbations around current state
            eps = torch.randn(self.pop_size, *tok_p.shape[1:], device=device)
            x_cand = tok_p + self.noise_std * eps

            # 2. Predict-x1 Round Trip: Snap continuous samples back to syntax manifold
            x1_rounded, strings = self._round_trip(x_cand.reshape(-1, *tok_p.shape[1:]))
            print('\n'.join(strings))

            # 3. Evaluate PDE fields on snap-projected strings
            fields = self._eval_fields(strings)
            pde_losses = F.mse_loss(
                _flatten(fields),
                target_flat.expand(self.pop_size, -1),
                reduction="none",
            ).mean(dim=1)

            # Track global best
            min_idx = torch.argmin(pde_losses).item()
            if pde_losses[min_idx].item() < best_loss:
                best_loss = pde_losses[min_idx].item()
                best_string = strings[min_idx]

            # 4. Form velocity target (x1_hat) via soft-min weighting of valid manifold states
            weights = torch.softmax(-pde_losses / self.tau, dim=0)
            x1_target = (weights.view(-1, 1, 1) * x1_rounded).sum(dim=0, keepdim=True)

            # 5. Euler Step along predicted velocity field: v = (x1_target - tok_p)
            velocity = x1_target - tok_p
            tok_p = tok_p + self.lr * velocity

            # 6. Logging & Centre Check
            if self.log_every and step % self.log_every == 0:
                tok_p_snap, s_centre = self._round_trip(tok_p)
                f_centre = self._eval_fields(s_centre)
                cur_loss = F.mse_loss(_flatten(f_centre), target_flat).item()

                tqdm.write(
                    f"step {step:4d}  |  center_loss {cur_loss:.4e}  "
                    f"|  best {best_loss:.4e}"
                )
                tqdm.write(f"  best: {best_string[:100]}")

                if cur_loss < 1e-5:
                    break

        return best_string, best_loss