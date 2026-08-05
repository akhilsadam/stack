from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from arch.dist import D as metric


def _flatten(x):
    return x.reshape(x.shape[0], -1)


class SGD_guided(nn.Module):
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
        steps=10,
        pop_size=64,
        noise_std=0.5,
        lr=1e-1,
        log_every=1,
        buffer_size=128,
        train_every=1,
        train_steps=20,
        train_batch=8,
        train_lr=3e-4,
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

    def search(self, init_pde, target_pde=None):
        """Run online fine-tuned search in the flow's latent space."""
        device = next(self.tok.parameters()).device

        self._fix_state()
        target = self._target_field(target_pde)
        target_flat = target.reshape(1, -1)

        # Replay buffer for fine-tuning
        buf_str: list[str] = []
        buf_fld: list[torch.Tensor] = []

        # Replay buffer for the linear model (perturbations)
        buf_pop: list[torch.Tensor] = []
        buf_dL: list[torch.Tensor] = []

        best_loss = float("inf")
        best_string = init_pde

        tok_p = nn.Parameter(self.tok.token_embed([init_pde])).to(device)
        opt = torch.optim.Adam([tok_p], lr=self.lr)

        for step in tqdm(range(self.steps), desc="SGD_guided search"):
            # ── sample & evaluate ──────────────────────────────────
            z_p, n_p = self.tok.forward(tok_p)
            eps = torch.randn(self.pop_size, *z_p.shape[1:], device=device)

            z_p_sample = z_p + self.noise_std * eps
            n_p_sample = self.tok.noise(z_p_sample)
            print(z_p_sample.shape, n_p_sample.shape)
            tok_sample = self.tok.reverse(z_p_sample, n_p_sample)

            strings = self.tok.token_embed.reverse(tok_sample.detach())

            print('\n'.join(strings))

            fields = self._eval_fields(strings)
            pde_losses = F.mse_loss(
                _flatten(fields),
                target_flat.expand(self.pop_size, -1),
                reduction="none",
            ).mean(dim=1)
            log_losses = torch.log(pde_losses.clamp(min=1e-10))
            dL = log_losses #- log_losses.mean()

            # ── accumulate replay buffer ───────────────────────────
            buf_str.extend(strings)
            buf_fld.append(fields.detach().cpu())
            if len(buf_str) > self.buffer_size:
                n_remove = len(buf_str) - self.buffer_size
                del buf_str[:n_remove]
                # buf_fld stores one tensor per step; remove whole steps
                n_step = len(buf_fld[0]) if buf_fld else 0
                n_step_remove = (n_remove + n_step - 1) // max(n_step, 1)
                del buf_fld[:n_step_remove]

            buf_pop.append(tok_sample.detach().cpu())
            buf_dL.append(dL.detach().cpu())
            if len(buf_pop) > 3:          # keep only 3 most recent steps
                buf_pop.pop(0)
                buf_dL.pop(0)

            # ── online fine-tune the flow ──────────────────────────
            do_ft = step > 0 and step % self.train_every == 0 and len(buf_str) >= self.train_batch
            if do_ft:
                all_fld = torch.cat(buf_fld, dim=0)[:len(buf_str)]
                self._fine_tune(buf_str, all_fld)
                # Re-encode current best through the updated flow
                z, n = self._encode(best_string)
                z, n = z.to(device), n.to(device)

            # ── fit local linear model ─────────────────────────────
            X = torch.cat(buf_pop, dim=0)
            X = X.reshape(X.shape[0], -1)
            y = torch.cat(buf_dL, dim=0)
            tau = 10
            weights = torch.softmax(-y.squeeze() / tau, dim=0)
            # z_hat = torch.linalg.lstsq(E, y).solution.reshape_as(z)

            print('Y', weights)

            tok_hat = (weights @ X).reshape_as(tok_p)

            # ── step ──────────────────────────────────────────────
            opt.zero_grad()
            loss = F.mse_loss(tok_p, tok_hat)
            loss.backward()
            opt.step()

            # tok_p = tok_p + self.lr * (tok_hat - tok_p)

            # ── track centre ──────────────────────────────────────
            with torch.no_grad():
                s_centre = self.tok.token_embed.reverse(tok_p.detach())
                f_centre = self._eval_fields(s_centre)
                cur_loss = F.mse_loss(_flatten(f_centre), target_flat).item()

            if cur_loss < best_loss:
                best_loss = float(cur_loss)
                best_string = s_centre[0]

            if self.log_every and step % self.log_every == 0:
                tqdm.write(
                    f"step {step:4d}  |  loss {cur_loss:.4e}  "
                    f"|  best {best_loss:.4e}  "
                    f"|  buf {len(buf_str)}  |  ft {'Y' if do_ft else 'n'}"
                )
                tqdm.write(f"  best: {best_string[:100]}")

        return best_string, best_loss