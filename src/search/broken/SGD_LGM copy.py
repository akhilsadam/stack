import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from arch.dist import D as metric


def _flatten(x):
    return x.reshape(x.shape[0], -1)

# SGD_RWR <- SGD_GMvD <- SGD_LGM
# GMvD in latent space
# global version v2
# depth version v2

class Search(nn.Module):
    """Method 1 + Online Fine-Tuning: Latent Drifting over an Aligned Manifold.

    Fine-tunes the tokenizer online so that latent metric distances D(z) align
    with field distances D(v). The drifting field V(z) is then computed directly
    within this fine-tuned, PDE-aligned latent space.
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
        buffer_size=128,
        train_every=1,
        train_steps=10,
        train_batch=16,
        train_lr=3e-4,
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
        self.buffer_size = buffer_size
        self.train_every = train_every
        self.train_steps = train_steps
        self.train_batch = train_batch
        self.log_every = log_every
        self._fixed_q = None

        # Optimizer for online manifold alignment
        self.fine_opt = torch.optim.Adam(self.tok.parameters(), lr=train_lr)

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

    def _decode_and_snap(
        self, z: torch.Tensor, n: torch.Tensor
    ) -> tuple[torch.Tensor, list[str]]:
        """Decodes latent vectors to tokens and snaps back to discrete syntax."""
        with torch.no_grad():
            tok_cont = self.tok.reverse(z, n)
            strings = self.tok.token_embed.reverse(tok_cont.detach())
            tok_rounded = self.tok.token_embed(strings)
        return tok_rounded, strings

    def _fine_tune(self, strings: list[str], fields: torch.Tensor):
        """Aligns tokenizer latent metric space with physical PDE field distance."""
        self.tok.train()
        B = len(strings)

        for _ in range(self.train_steps):
            idx = torch.randperm(B)[: min(self.train_batch, B)]
            batch_str = [strings[i] for i in idx]
            batch_fld = fields[idx]

            tok = self.tok.token_embed(batch_str)
            z, n = self.tok.forward(tok)

            # 1. Manifold alignment loss: D(z) ≈ D(v)
            d_z = metric(_flatten(z))
            d_v = metric(_flatten(batch_fld))
            align_loss = F.mse_loss(d_z, d_v)

            # 2. One-step reconstruction / commitment loss
            tok_rec = self.tok.reverse(z, n)
            commit_loss = F.mse_loss(tok, tok_rec)

            loss = align_loss + 0.5 * commit_loss

            self.fine_opt.zero_grad()
            loss.backward()
            self.fine_opt.step()

        self.tok.eval()

    def _compute_latent_drift(
        self, z_center: torch.Tensor, z_pop: torch.Tensor, losses: torch.Tensor
    ) -> torch.Tensor:
        """Computes drifting field V(z_center) in the fine-tuned latent space."""
        z_center_flat = z_center.reshape(1, -1)
        z_pop_flat = z_pop.reshape(self.pop_size, -1)

        # Pairwise distance & RBF similarity in fine-tuned latent z-space
        dists_sq = torch.sum((z_pop_flat - z_center_flat) ** 2, dim=-1)
        kernel = torch.exp(-dists_sq / (2 * (self.sigma**2)))

        # Fitness weighting
        weights = torch.softmax(-losses / self.tau, dim=0)

        # Latent Drifting Vector: V(z) = \sum_j w_j * K(z, z_j) * (z_j - z)
        diffs = z_pop - z_center
        drift_vec = (weights.view(-1, 1, 1) * kernel.view(-1, 1, 1) * diffs).sum(
            dim=0, keepdim=True
        )

        return drift_vec

    # ── main entry point ────────────────────────────────────────────

    def search(self, init_pde: str, target_pde: str | None = None):
        device = next(self.tok.parameters()).device

        self._fix_state()
        target_flat = self._target_field(target_pde).reshape(1, -1)

        # Replay buffer for online tokenizer alignment
        buf_str: list[str] = []
        buf_fld: list[torch.Tensor] = []

        best_loss = float("inf")
        best_string = init_pde

        # Encode seed RPN string to get initial latent state z_center
        with torch.no_grad():
            tok_init = self.tok.token_embed([init_pde]).to(device)
            z_center, n_center = self.tok.forward(tok_init)
            z_center = z_center.detach()
            n_center = n_center.detach()

        for step in tqdm(range(self.steps), desc="Online Drifting Search"):
            # 1. Sample perturbations in fine-tuned latent space
            eps = torch.randn(self.pop_size, *z_center.shape[1:], device=device)
            z_samples = z_center + self.noise_std * eps
            n_samples = self.tok.noise(z_samples)

            # 2. Decode latents to syntax-valid token strings
            _, strings = self._decode_and_snap(z_samples, n_samples)

            # 3. Evaluate physical PDE fitness
            fields = self._eval_fields(strings)
            pde_losses = F.mse_loss(
                _flatten(fields),
                target_flat.expand(self.pop_size, -1),
                reduction="none",
            ).mean(dim=1)

            # Track best string
            min_idx = torch.argmin(pde_losses).item()
            if pde_losses[min_idx].item() < best_loss:
                best_loss = pde_losses[min_idx].item()
                best_string = strings[min_idx]

            # 4. Update Replay Buffer
            buf_str.extend(strings)
            buf_fld.append(fields.detach().cpu())
            if len(buf_str) > self.buffer_size:
                overflow = len(buf_str) - self.buffer_size
                del buf_str[:overflow]
                step_size = fields.shape[0]
                num_steps_del = (overflow + step_size - 1) // step_size
                del buf_fld[:num_steps_del]

            # 5. Online Fine-Tune Tokenizer
            do_ft = (
                step > 0
                and step % self.train_every == 0
                and len(buf_str) >= self.train_batch
            )
            if do_ft:
                all_fld = torch.cat(buf_fld, dim=0)[: len(buf_str)].to(device)
                self._fine_tune(buf_str, all_fld)

                # Re-encode current best string through freshly aligned manifold
                with torch.no_grad():
                    tok_best = self.tok.token_embed([best_string]).to(device)
                    z_center, n_center = self.tok.forward(tok_best)

            # 6. Re-encode population through the updated tokenizer manifold
            with torch.no_grad():
                tok_pop = self.tok.token_embed(strings).to(device)
                z_pop, _ = self.tok.forward(tok_pop)

            # 7. Compute Drifting Field V(z) and update latent center
            drift_v = self._compute_latent_drift(z_center, z_pop, pde_losses)
            z_center = z_center + self.lr * drift_v

            # 8. Logging
            if self.log_every and step % self.log_every == 0:
                _, s_center = self._decode_and_snap(z_center, n_center)
                f_center = self._eval_fields(s_center)
                cur_loss = F.mse_loss(_flatten(f_center), target_flat).item()

                tqdm.write(
                    f"step {step:4d}  |  center_loss {cur_loss:.4e}  "
                    f"|  best {best_loss:.4e}  |  ft {'Y' if do_ft else 'n'}"
                )
                tqdm.write(f"  best: {best_string[:100]}")

                if cur_loss < 1e-5:
                    break
                
        return best_string, best_loss