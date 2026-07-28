import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os

from arch.flow import NF
from arch.layer import MLP
# from arch.dist import SNE as metric
# from arch.dist import log_odds_SNE as metric

from arch.dist import D as metric
from space.tokens import TokenEmbedding as TE

import wandb

def dim(z):
    return np.prod(z.shape[1:])

class Buffer(nn.Module):
    def __init__(self, batch):
        super().__init__()
        self.batch = batch
        self.Q = 0
        self.queue = None

    def __call__(self, new):
        if self.queue is None:
            self.queue = torch.empty((self.batch, *new.shape[1:]), device=new.device)

        with torch.no_grad():
            N = new.shape[0]
            self.queue[:-N] = self.queue.clone()[N:]
            self.queue[-N:] = new.detach()
            self.Q = min(self.Q + N, self.batch)

        return torch.cat([self.queue[-self.Q:-N], new], dim=0)

class Tokenizer(nn.Module):
    def __init__(self, vocab, _eval, batch, seq_len, dim, depth, steps=5, lr=1e-3, _iter=4000,
                 vis=None):
        super().__init__()
        self.vocab = vocab
        self.token_embed = TE(vocab, seq_len, dim)
        self._eval = _eval

        self.steps = steps
        self.seq_len = seq_len
        self.dim = dim
        self.batch = batch
        self._iter = _iter

        self.f_z = MLP(seq_len, dim, depth)
        self.f_n = MLP(seq_len, dim, depth)

        self.kcrit = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.crit = nn.MSELoss()
        # self.crit = nn.KLDivLoss(reduction='batchmean', log_target=True)

        # Train BOTH the embedding and the flow
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

        self.max_condition_num = 1000
        self.k = 5
        self.debug = True

        self.vis = vis

        buf_len = 4
        self.d_buffer = Buffer(buf_len*batch)
        self.dh_buffer = Buffer(buf_len*batch)


        self._train()

    def noise(self, z):
        z = torch.randn_like(z)
        return z / dim(z)

    def v_noise(self, size=None):
        device = next(self.parameters()).device
        if size is None:
            size = self.batch
        z = torch.randn((size, self.seq_len, self.dim), device=device)
        return z / dim(z)
        
    def forward(self, tok):
        return NF._fwd(tok, tok, self.f_z, self.f_n, self.steps)

    def reverse(self, z, n):
        return NF._rev(z, n, self.f_z, self.f_n, self.steps)

    def embed(self, strings):
        tok = self.token_embed(strings)
        return NF._fwd(tok, tok, self.f_z, self.f_n, self.steps)[0]

    def value(self, tok):
        strings = self.token_embed.reverse(tok)
        result = self._eval(strings)
        # Replace NaN/inf with 0
        return torch.nan_to_num(result, nan=1000000.0, posinf=1000000.0, neginf=-1000000.0)

    def loss(self, strings=None):

        # sample_vector = self.v_noise()
        # tok = self.sample_token_from_vector(sample_vector) # extremely BAD idea, can cause distribution to collapse (flow only learns "safe" distribution)
        tok = self.token_embed.generate(self.v_noise()) # not great either, doesn't learn "good/stable" PDEs
    
        # TODO: sampling
        # tok = 0.9 * self.sample_token_from_vector(self.v_noise()) + 0.1 * self.v_noise()

        # toka = self.token_embed.generate(self.v_noise())
        # tokb = self.sample_token_from_vector(self.v_noise())
        # r2 = torch.rand(self.batch, device=toka.device)

        # tok = torch.where(r2[:,None,None] < 0.5, toka, tokb)


        # noise = self.v_noise()
        # base = torch.repeat_interleave(noise[0:1], repeats=self.batch, dim=0)
        # head_len = 3
        # base[:, :head_len, :] = noise[:, :head_len, :]
        # tok = self.token_embed.generate(base)


        # unquantized "samples" VQ loss, on complete random tokens -- prevents collapse of dist
        token = self.token_embed(self.token_embed.reverse(tok))
        loss_commit = F.mse_loss(tok, token.detach())


        _str = self.token_embed.generate_rpns(self.batch)

        # max_seq_len = tok.shape[1]
        # r = torch.rand(self.batch, device=tok.device)
        # # b = 6.0
        # # y = (r - 0.02)**b + 1 - (0.98)**b  # weighted toward small seqs
        # random_seq_len = ((r) * max_seq_len).long()

        # Sample from the token space directly (no flow yet)
        if strings is None:
            # _str = self.token_embed.reverse(tok, random_seq_len)
            token = self.token_embed(_str)
            # print("\\\\ ".join(self._eval.to_latex(s) for s in _str))
        else:
            token = self.token_embed(strings)
            # print("\\\\ ".join(self._eval.to_latex(s) for s in strings))

        # Get semantic evaluations
        value = self.value(token)
        value_flat = value.reshape(value.shape[0], -1)

        # Filter valid items - reject ones with huge values (numerical instability)
        stable_mask = (torch.linalg.norm(value_flat, dim=-1) < self.max_condition_num * self._eval.norm)
        valid_mask = (
            stable_mask &
            torch.all(torch.isfinite(value_flat), dim=-1)
        )

        # Debug: log unstable RPN expressions
        if valid_mask.sum() < self.batch:
            _strings = self.token_embed.reverse(token)
            for i, (mask, s) in enumerate(zip(valid_mask, _strings)):
                if not mask:
                    condition_num  = torch.linalg.norm(value_flat[i]) / self._eval.norm
                    print(f"UNSTABLE (|cond|={condition_num:.2e}): {s}")

        n_valid = valid_mask.sum()
        if n_valid < self.k:
            print("Warning: not enough valid samples")
            return False

        # Compute semantic distances
        v = value_flat[valid_mask]

        # Now apply flow and get latent distances
        z, n = self.forward(token[valid_mask])

        # v = self.d_buffer(v)
        # v_hat = self.dh_buffer(z)
        v_hat = z

        # tok_buf = self.dh_buffer(token[valid_mask])
        # v_hat,n_hat = self.forward(tok_buf)
        # z = v_hat[-n_valid:]
        # n = n_hat[-n_valid:]


        d = metric(v)
        d_hat = metric(v_hat)

        # # Mask out diagonal where targets are -inf (log(0))        
        # # Compute KLDivLoss only on the off-diagonal elements
        # mask = ~torch.eye(n_valid, dtype=torch.bool, device=d.device)
        # loss_align = F.kl_div(d_hat[mask], d[mask], reduction='sum', log_target=True) / n_valid

        # if strings:
        #     from matplotlib import pyplot as plt
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(d.cpu().numpy() , cmap='inferno')
        #     plt.colorbar()
        #     # label by string
        #     xticklabels = strings
        #     plt.xticks(np.arange(len(xticklabels)), xticklabels, rotation=90)
        #     plt.yticks(np.arange(len(xticklabels)), xticklabels)

        #     plt.savefig(f'dist_.png')
        #     plt.close()
            
        # mask = mask & (d < 1e-5)

        # loss_align = 10 * torch.sqrt(d + 1e-8).mean() # self.crit(d_hat, d) +
        # mask = ~torch.eye(d.shape[0], dtype=torch.bool, device=d.device)
        loss_align = self.crit(d_hat, d)

        # print("D_hat sample:", d_hat[mask][:5])
        # print("D target sample:", d[mask][:5])
        # print("Are they identical?:", torch.allclose(d_hat[mask], d[mask]))

        n_logits = F.log_softmax(self.noise(n).view(n_valid,-1), dim=1)
        loss_dist = self.kcrit(F.log_softmax(n.view(n_valid,-1), dim=1), n_logits) + self.kcrit(F.log_softmax(z.view(n_valid,-1), dim=1), n_logits)

        return loss_align, loss_commit, loss_dist

    def _train(self):
        run = wandb.init(project='vectorspace', name='qg_tokenizer', config={
            'vocab_size': len(self.vocab),
            'seq_len': self.seq_len,
            'dim': self.dim,
            'steps': self.steps,
            'lr': self.opt.param_groups[0]['lr'],
            'iter': self._iter,
        }, mode='offline')

        for i in tqdm(range(self._iter)):
            self.opt.zero_grad()
            try:
                result = self.loss()
                if result is False:
                    continue
                align_loss, commit_loss, dist_loss = result
                
                wandb.log({'align_loss': align_loss.item(), 'commit_loss': commit_loss.item(), 'dist_loss': dist_loss.item()})
                total_loss = align_loss + 0.1 * commit_loss + 0.1 * dist_loss

                # strings = self.vis.snapshot(i, self.embed)
                # val_align_loss, val_commit_loss, _ = self.loss(strings)
                # wandb.log({'val_align_loss': val_align_loss.item(), 'val_commit_loss': val_commit_loss.item()})
                # # cheating to debug
                # total_loss = total_loss + val_align_loss

                total_loss.backward()
                self.opt.step()

                with torch.no_grad():
                    strings = self.vis.snapshot(i, self.embed)
                    val_align_loss, val_commit_loss, _ = self.loss(strings)
                    wandb.log({'val_align_loss': val_align_loss.item(), 'val_commit_loss': val_commit_loss.item()})

                # Debug gradient norm
                if self.debug and i % 100 == 0:
                    total_grad_norm = 0
                    for param in self.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.data.norm(2).item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    print(f"Iter {i}: loss={total_loss.item():.6f}, grad_norm={total_grad_norm:.6e}")
                    self.vis.plot(i)

                # # Visualization snapshot
                    # if self.vis_every > 0 and i % self.vis_every == 0:
                #     with torch.no_grad():
                #         emb = self.token_embed(self.vis_prompts)
                #         z, n = self.forward(emb)
                #         self.snapshots.append({
                #             'iter': i,
                #             'rpns': self.vis_prompts,
                #             'latents': z.cpu().clone()
                #         })
                #     if self.vis_plot:
                #         from vis.atlas import AtlasPlotter
                #         output_path = os.path.join(self.vis_out, f'atlas_iter_{i:06d}.png')
                #         AtlasPlotter.plot(self.snapshots, clusters=self.vis_clusters, output=output_path)

            except Exception as e:
                print(f"Error at iteration {i}: {e}")
                continue

        # Save final snapshots
        # if self.vis_every > 0:
        #     torch.save({'snapshots': self.snapshots}, os.path.join(self.vis_out, 'snapshots.pt'))
        #     print(f"Saved visualization snapshots to {self.vis_out}/snapshots.pt")

    ## eval

    def token_to_vector(self, tok):
        return self.forward(tok)[0]

    def sample_token_from_vector(self, z):
        n = self.noise(z)
        return self.reverse(z, n)
