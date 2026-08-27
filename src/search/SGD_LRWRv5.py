from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from arch.dist import D as metric

import random


def _flatten(x):
    return x.reshape(x.shape[0], -1)

# based on flow matching with velocity
# global version v0
# depth version v0

from arch.layer import MLP

# class Net(nn.Module):
#     def __init__(self, tok):
#         super().__init__()
        
#         # self.enc = MLP(tok.seq_len, tok.dim, tok.tok_dim, depth=4, num_heads=1)
#         # self.dec = MLP(tok.seq_len, tok.dim, tok.tok_dim, depth=4, num_heads=1)

#         # random projection to large dim
#         self.enc = nn.Sequential(
#             nn.Linear(tok.dim, 256),
            
#         )
        
#     def encoder(self, t):
        


class SphericalProjector(nn.Module):
    """Simple linear autoencoder that projects latents onto a high-dimensional hypersphere.
    
    Uses L2 normalization in the latent space to enforce continuous spherical topology.
    """
    def __init__(self, tok, proj_dim: int = 512):
        super().__init__()
        # self.enc = nn.Sequential(
        #     nn.Linear(in_dim, proj_dim),
        #     nn.ReLU(),
        #     nn.Linear(proj_dim, proj_dim),
        # )
        # self.dec = nn.Sequential(
        #     nn.Linear(proj_dim, proj_dim),
        #     nn.ReLU(),
        #     nn.Linear(proj_dim, in_dim),
        # )
        self.enc = nn.Sequential(
            nn.Linear(tok.dim, proj_dim),
            MLP(tok.seq_len, proj_dim, 4, depth=2, num_heads=8),
        )
        self.dec = nn.Sequential(
            MLP(tok.seq_len, proj_dim, 4, depth=2, num_heads=8),
            nn.Linear(proj_dim, tok.dim),
        )
        self.proj_dim = proj_dim
        self.opt = torch.optim.Adam(self.parameters(), lr=3e-4)

        self.tok = tok

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Project to high-dim and map onto unit hypersphere surface
        z = self.enc(x)
        z = torch.mean(z, dim=1) # pool tokens
        # z = F.normalize(z, p=2, dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(1).repeat(1, self.tok.seq_len, 1) # expand to sequence
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec

    def train_on_fly(self, x: torch.Tensor, steps: int = 30):
        """Ultra-fast online reconstruction training on candidates."""
        self.train()
        for _ in range(steps):
            self.opt.zero_grad()
            x_tilde = torch.randn_like(x)
            x = self.tok.token_embed(self.tok.token_embed.reverse(x_tilde))
            x_tilde = x + torch.randn_like(x) * 0.05
            _, x_rec = self(x_tilde)
            loss = F.mse_loss(x_rec, x)
            loss.backward()
            self.opt.step()
        self.eval()

    def pretrain(self, x, steps: int = 1000):
        """Pretrain the projector on the initial state."""
        self.train()
        for i in range(steps):
            self.opt.zero_grad()

            x_1 = torch.randn_like(x)
            _str_1 = self.tok.token_embed.reverse(x_1, max_seq_len=self.tok.seq_len // 2)
            x_1 = self.tok.token_embed(_str_1)

            x_2 = torch.randn_like(x)
            _str_2 = self.tok.token_embed.reverse(x_2, max_seq_len=self.tok.seq_len // 2)
            _str_2 = [a[1:] for a in _str_2]



            loss_reco = F.mse_loss(self.decode(e_1), x_1)

            loss = loss_reco + loss_enc

            if i % 100 == 0:
                print(loss_reco.item(), loss_enc.item(), loss_enc_2.item())
            loss.backward()
            self.opt.step()
        self.eval()

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
        pop_size=128,
        noise_std=0.75,
        lr=1e-2,
        log_every=10,
        buffer_size=128,
        train_every=1,
        train_steps=20,
        train_batch=8,
        train_lr=3e-4,
        tau = 0.05       # Temperature for softmax weighting
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
        # self.fine_opt = torch.optim.Adam(self.tok.parameters(), lr=train_lr)

        self._fixed_q = None

        self.projector = SphericalProjector(tokenizer, proj_dim=64)

    # Replace old placeholder encoder/decoder with projector calls
    # def encoder(self, t: torch.Tensor) -> torch.Tensor:
    #     return self.projector.encode(t)

    # def decoder(self, z: torch.Tensor) -> torch.Tensor:
    #     return self.projector.decode(z)

    def encoder(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        return t

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

    # ── main entry point ────────────────────────────────────────────


    def _eval(self, strings, target_flat):
        fields = self._eval_fields(strings)
        pde_losses = F.mse_loss(
            _flatten(fields),
            target_flat.expand(len(strings), -1),
            reduction="none",
        ).mean(dim=1)
        return pde_losses
        
    def search(self, init_pde: str, target_pde: str | None = None):
        """Predict-x1 Flow Matching search with round-trip manifold projection."""
        device = next(self.tok.parameters()).device

        self._fix_state()
        target = self._target_field(target_pde)
        target_flat = target.reshape(1, -1)

        best_loss = float("inf")
        best_string = init_pde
        search_count = 0

        # Initialize current state on the valid embedding manifold
        tok_p = self.tok.token_embed([init_pde]).to(device).detach()

        self.buffer = {} # string -> loss

        for step in tqdm(range(self.steps), desc="Flow Matching Search"):
            # 1. Sample continuous perturbations around current state
            eps = torch.randn(self.pop_size, *tok_p.shape[1:], device=device)
            x_cand = tok_p + self.noise_std * eps

            # 2. Predict-x1 Round Trip: Snap continuous samples back to syntax manifold
            x1_rounded, strings = self._round_trip(x_cand.reshape(-1, *tok_p.shape[1:]))
            print('\n'.join(strings))

            # 3. Evaluate PDE fields on snap-projected strings if not in buffer & add to buffer
            new_strings = []
            for s in strings:
                if s.strip() not in self.buffer:
                    new_strings.append(s.strip())

            if (n := len(new_strings)) > 0:
                search_count += n
                losses = self._eval(new_strings, target_flat)
                for s, loss in zip(new_strings, losses):
                    self.buffer[s] = loss
            
            pde_losses = torch.tensor([self.buffer[s] for s in strings], device=device)

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
                
                print("search_count", search_count)
                print(len(self.buffer))
        return best_string, best_loss
