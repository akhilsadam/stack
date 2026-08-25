from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class Search(nn.Module):
    """Stochastic gradient descent in latent space via straight-through commit.

    A latent vector ``z`` (a continuous token embedding) is optimised so that
    its argmax-decoded RPN string minimises PDE-field MSE to a target.

    Because the argmax decoder and the QG solver are both non-differentiable,
    we use a **straight-through estimator (STE)** on the decoder logits:

        forward:  hard_embed = Embedding(argmax(z @ Vᵀ))  — non-diff
        backward: ∇z  ≈  ∇ MSE(z, softmax(z @ Vᵀ/τ) @ V)  — smooth

    Candidates are weighted by PDE quality so that the gradient pulls ``z``
    toward embeddings that decode to better-matching RPN strings.
    """

    def __init__(
        self,
        tokenizer,
        evaluator=None,
        steps=200,
        pop_size=64,
        noise_std=4.0,
        lr=0.3,
        log_every=20,
        ste_temp=0.5,
    ):
        super().__init__()
        self.tok = tokenizer
        self.eval = evaluator or tokenizer._eval
        self.steps = steps
        self.pop_size = pop_size
        self.noise_std = noise_std
        self.lr = lr
        self.log_every = log_every
        self.ste_temp = ste_temp

        self._fixed_q = None

    # ── helpers ─────────────────────────────────────────────────────

    def _embed(self, strings):
        """Token embeddings of RPN strings (detached from the argmax graph)."""
        return self.tok.token_embed(strings).detach()

    @property
    def _token_dim(self):
        """Dimensionality of the token part of the embedding."""
        return self.tok.token_embed.embed_dim - self.tok.token_embed.phys_dim

    @property
    def _phys_dim(self):
        return self.tok.token_embed.phys_dim

    @property
    def _vocab_weight(self):
        """Unit-normalised vocabulary embedding matrix."""
        w = self.tok.token_embed.embedding.weight
        return F.normalize(w, p=2, dim=-1)

    def _soft_embed(self, z):
        """Differentiable soft-decoded embedding via temperature-softmax."""
        z_tok = F.normalize(z[..., : self._token_dim], p=2, dim=-1)
        logits = z_tok @ self._vocab_weight.T          # (…, V)
        weights = F.softmax(logits / self.ste_temp, dim=-1)
        return weights @ self._vocab_weight             # (…, D_tok)

    def _ste_embed(self, z, strings):
        """Straight-through token embedding.

        Forward:  uses the argmax-derived embedding (``strings``).
        Backward: gradient flows through the differentiable softmax.
        """
        hard = self._embed(strings)                     # (N, S, dim)
        soft = self._soft_embed(z)                      # (N, S, D_tok)
        ste = (hard[..., : self._token_dim] - soft).detach() + soft
        return torch.cat([ste, hard[..., -self._phys_dim :]], dim=-1)

    def _fix_state(self):
        with torch.no_grad():
            self.eval.random_state()
            self._fixed_q = self.eval.q_phys.clone()

    def _eval_fields(self, strings):
        self.eval.q_phys = self._fixed_q
        with torch.no_grad():
            fields = torch.stack([self.eval.eval_one(s) for s in strings], dim=0)
        return fields

    def _target_field(self, rpn: str | None = None):
        if rpn is not None and rpn != self.eval.target_pde:
            self.eval.target_pde = rpn
        self.eval.q_phys = self._fixed_q
        return self.eval.target().detach()

    # ── main entry point ────────────────────────────────────────────

    def search(self, init_pde, target_pde=None):
        """Run straight-through gradient search in embedding space."""
        device = next(self.tok.parameters()).device

        self._fix_state()
        target = self._target_field(target_pde)
        target_flat = target.reshape(1, -1)

        # Initialise z from the init PDE's token embedding
        z_init = self.tok.token_embed([init_pde]).to(device)  # (1, S, dim)
        z_param = nn.Parameter(z_init.clone())

        opt = torch.optim.Adam([z_param], lr=self.lr)

        sigma = self.noise_std
        best_loss = float("inf")
        best_string = init_pde

        for step in tqdm(range(self.steps), desc="SGD search"):
            # --- sample population ---
            eps = torch.randn(self.pop_size, *z_param.shape[1:], device=device)
            z_pop = z_param + sigma * eps                        # (N, S, dim)

            # --- hard argmax decode (forward only) ---
            strings = self.tok.token_embed.reverse(z_pop.detach())

            # --- differentiable STE commit embedding ---
            ste_embed = self._ste_embed(z_pop, strings)          # (N, S, dim)
            commit_per = (z_pop - ste_embed).pow(2).sum(dim=[1, 2])  # (N,)

            print('\n'.join(strings))

            # --- PDE fields & losses ---
            fields = self._eval_fields(strings)
            pde_losses = F.mse_loss(
                fields.reshape(self.pop_size, -1),
                target_flat.expand(self.pop_size, -1),
                reduction="none",
            ).mean(dim=1)                                        # (N,)
            pde_losses = torch.log(torch.nan_to_num(pde_losses, nan=1e10, posinf=1e10) + 1e-8)

            # --- weighted commit loss ---
            # Candidates with lower PDE loss get higher weight
            with torch.no_grad():
                weights = F.softmin(pde_losses / self.ste_temp, dim=0)

            loss = (weights * commit_per).sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            # --- track centre ---
            with torch.no_grad():
                z_centre = z_param.detach()
                s_centre = self.tok.token_embed.reverse(z_centre)[0]
                f_centre = self._eval_fields([s_centre])
                cur_loss = F.mse_loss(f_centre.reshape(1, -1), target_flat).item()

            if cur_loss < best_loss:
                best_loss = float(cur_loss)
                best_string = s_centre

            if self.log_every and step % self.log_every == 0:
                tqdm.write(
                    f"step {step:4d}  |  loss {cur_loss:.4e}  "
                    f"|  best {best_loss:.4e}  "
                    f"|  pde_worst {pde_losses.max().item():.4e}"
                )
                tqdm.write(f"  best: {best_string[:100]}")

        return best_string, best_loss