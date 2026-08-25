from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from arch.dist import D as metric


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


class Search(nn.Module):
    """Method 1: Generative Modeling via Flow-Based Sampling & Drifting Adaptation.

    Replaces isotropic token noise with generative sampling from the tokenizer's flow
    model. The tokenizer is continuously fine-tuned online using a drifting objective
    (reward-weighted regression + metric alignment) so that each iteration's flow 
    naturally samples higher-fitness token regions.
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
        log_every=1,
        buffer_size=128,
        train_every=1,
        train_steps=10,
        train_batch=16,
        train_lr=3e-4,
    ):
        super().__init__()
        self.tok = tokenizer
        self.eval = evaluator or tokenizer._eval
        self.steps = steps
        self.pop_size = pop_size
        self.noise_std = noise_std
        self.lr = lr
        self.tau = tau
        self.log_every = log_every
        self.buffer_size = buffer_size
        self.train_every = train_every
        self.train_steps = train_steps
        self.train_batch = train_batch

        # Online optimizer for flow fine-tuning
        self.fine_opt = torch.optim.Adam(self.tok.parameters(), lr=train_lr)
        self.kcrit = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self._fixed_q = None

    # ── helpers ─────────────────────────────────────────────────────

    def _fix_state(self):
        with torch.no_grad():
            self.eval.random_state()
            self._fixed_q = self.eval.q_phys.clone()

    def _eval_fields(self, strings: list[str]) -> torch.Tensor:
        self.eval.q_phys = self._fixed_q
        with torch.no_grad():
            return torch.stack([self.eval.eval_one(s) for s in strings], dim=0)

    def _target_field(self, rpn: str | None = None) -> torch.Tensor:
        if rpn is not None and rpn != self.eval.target_pde:
            self.eval.target_pde = rpn
        self.eval.q_phys = self._fixed_q
        return self.eval.target().detach()

    def _round_trip(self, x: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """Snap continuous flow vectors back to nearest codebook embeddings."""
        with torch.no_grad():
            strings = self.tok.token_embed.reverse(x.detach())
            x_rounded = self.tok.token_embed(strings).to(x.device)
        return x_rounded, strings

    def _sample_from_flow(self, tok_center: torch.Tensor) -> torch.Tensor:
        """Samples candidates by perturbing in flow latent space (z) and decoding."""
        with torch.no_grad():
            # Encode current center to latent z space
            z_center, _ = self.tok.forward(tok_center)

            # Perturb in latent space
            eps_z = torch.randn(
                self.pop_size, *z_center.shape[1:], device=tok_center.device
            )
            z_cand = z_center + self.noise_std * eps_z

            # Sample auxiliary noise n and decode via flow inverse mapping
            n_cand = self.tok.noise(z_cand)
            x_cand = self.tok.reverse(z_cand, n_cand)

        return x_cand

    def _fine_tune(
        self,
        buf_str: list[str],
        buf_fld: torch.Tensor,
        buf_tok: torch.Tensor,
        buf_loss: torch.Tensor,
    ):
        """Fine-tunes the tokenizer flow via drifting (reward-weighted flow alignment)."""
        self.tok.train()
        B = len(buf_str)

        for _ in range(self.train_steps):
            idx = torch.randperm(B)[: min(self.train_batch, B)]
            batch_str = [buf_str[i] for i in idx]
            batch_fld = buf_fld[idx]
            batch_tok = buf_tok[idx]
            batch_loss = buf_loss[idx]

            # 1. Forward flow pass
            tok = self.tok.token_embed(batch_str)
            z, n = self.tok.forward(tok)

            # 2. Distance metric alignment: D(z) ≈ D(v)
            d_z = metric(_flatten(z))
            d_v = metric(_flatten(batch_fld))
            align_loss = F.mse_loss(d_z, d_v)

            # 3. Flow Reversibility / Commitment Loss
            n2 = self.tok.noise(z)
            tok2 = self.tok.reverse(z, n2)
            tok3 = self.tok.token_embed(self.tok.token_embed.reverse(tok2))
            z2, _ = self.tok.forward(tok3)
            commit_loss = F.mse_loss(tok2, tok3) + F.mse_loss(z, z2)

            # 4. Latent distribution regularizer
            n_logits = F.log_softmax(self.tok.noise(n).view(len(idx), -1), dim=1)
            dist_loss = self.kcrit(
                F.log_softmax(n.view(len(idx), -1), dim=1), n_logits
            ) + self.kcrit(F.log_softmax(z.view(len(idx), -1), dim=1), n_logits)

            # 5. Drifting Loss: Reward-weighted likelihood over token space
            # Train flow to generate candidates proportional to fitness exp(-loss / tau)
            weights = torch.softmax(-batch_loss / self.tau, dim=0).detach()
            tok_rec = self.tok.reverse(z, n)
            drift_loss = (weights.view(-1, 1, 1) * (tok_rec - batch_tok) ** 2).sum()

            # Total online fine-tuning loss
            total_loss = align_loss + 0.1 * commit_loss + 0.1 * dist_loss + 1.0 * drift_loss

            self.fine_opt.zero_grad()
            total_loss.backward()
            self.fine_opt.step()

        self.tok.eval()

    # ── main entry point ────────────────────────────────────────────

    def search(self, init_pde: str, target_pde: str | None = None):
        device = next(self.tok.parameters()).device

        self._fix_state()
        target = self._target_field(target_pde)
        target_flat = target.reshape(1, -1)

        best_loss = float("inf")
        best_string = init_pde

        # Replay buffers for flow training
        buf_str: list[str] = []
        buf_fld: list[torch.Tensor] = []
        buf_tok: list[torch.Tensor] = []
        buf_loss: list[torch.Tensor] = []

        # Continuous state initialized on valid embedding manifold
        tok_p = self.tok.token_embed([init_pde]).to(device).detach()

        for step in tqdm(range(self.steps), desc="Flow-Drifting Search"):
            # 1. Sample continuous candidate vectors using Flow Model (not isotropic noise)
            x_cand = self._sample_from_flow(tok_p)

            # 2. Snap continuous candidate vectors to syntax manifold
            x_rounded, strings = self._round_trip(x_cand)

            # 3. Evaluate PDE fitness
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

            # 4. Update Replay Buffer
            buf_str.extend(strings)
            buf_fld.append(fields.detach())
            buf_tok.append(x_rounded.detach())
            buf_loss.append(pde_losses.detach())

            # Maintain buffer capacity
            if len(buf_str) > self.buffer_size:
                overflow = len(buf_str) - self.buffer_size
                del buf_str[:overflow]
                step_chunks = (overflow + self.pop_size - 1) // self.pop_size
                del buf_fld[:step_chunks]
                del buf_tok[:step_chunks]
                del buf_loss[:step_chunks]

            # 5. Train tokenizer online via drifting
            if (
                step > 0
                and step % self.train_every == 0
                and len(buf_str) >= self.train_batch
            ):
                all_fld = torch.cat(buf_fld, dim=0)[: len(buf_str)]
                all_tok = torch.cat(buf_tok, dim=0)[: len(buf_str)]
                all_loss = torch.cat(buf_loss, dim=0)[: len(buf_str)]
                self._fine_tune(buf_str, all_fld, all_tok, all_loss)

            # 6. Form velocity target via soft-min weighting of valid manifold states
            weights = torch.softmax(-pde_losses / self.tau, dim=0)
            x1_target = (weights.view(-1, 1, 1) * x_rounded).sum(dim=0, keepdim=True)

            # 7. Step continuous state along velocity field
            velocity = x1_target - tok_p
            tok_p = tok_p + self.lr * velocity

            # 8. Logging & center evaluation
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