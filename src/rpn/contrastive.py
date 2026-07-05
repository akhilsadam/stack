"""
Contrastive learning for RPN token embeddings.

**Anchor** = tokenized expression. **Positive** = one sound algebraic rewrite
(from :mod:`algebra`). **Negatives** = other sequences in the batch (InfoNCE).

This module keeps a single encoder path: :class:`RPNTokenEmbedder` → pool →
projection. Pooling uses a mask so padded ``__scalar__`` slots do not contribute.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebra import AlgebraicRuleSet, create_composite_ruleset
from .embeddings import (
    TOKEN_TO_ID,
    TOKEN_TO_CAT,
    ID_TO_TOKEN,
    ID_TO_ARITY,
    RPNTokenEmbedder,
    batch_tokenize_rpn,
    TokenCategory,
)


from .ae.naive_mlp import RPN_AE as RPN_AE_NMLP
from .ae.mlp_ae import RPN_AE as RPN_AE_MLP_deterministic
from .ae.mlp import RPN_AE as RPN_AE_MLP_Mixer
from .ae.att import RPN_AE as RPN_AE_ATT
from .ae.att2 import RPN_AE as RPN_AE_ATT2
from .ae.att3 import RPN_AE as RPN_AE_ATT3
from .gen.mlp import RPN_GEN


RPN_AE_ = {
    "nmlp": RPN_AE_NMLP,
    "mlp_deterministic": RPN_AE_MLP_deterministic,
    "mlp_mix": RPN_AE_MLP_Mixer,
    "att": RPN_AE_ATT,
    "att2": RPN_AE_ATT2,
    "att3": RPN_AE_ATT3,
}

def masked_mean_pool(
    seq: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Parameters
    ----------
    seq : (B, L, E)
    mask : (B, L) bool — True = real token (not padding)
    """
    w = mask.unsqueeze(-1).float()
    num = (seq * w).sum(dim=1)
    den = w.sum(dim=1).clamp_min(eps)
    return num / den


def infonce_logits(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Symmetric InfoNCE logits (B, B): diagonal = positive pairs.
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    return (z_a @ z_b.T) / temperature

def infonce_single_loss(
    z_anchor: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy on both directions; positives are batch-aligned."""
    logits_ab = infonce_logits(z_anchor, z_anchor, temperature)
    B = z_anchor.size(0)
    targets = torch.arange(B, device=z_anchor.device)
    return F.cross_entropy(logits_ab, targets)

def infonce_symmetric_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy on both directions; positives are batch-aligned."""
    logits_ab = infonce_logits(z_anchor, z_positive, temperature)
    logits_ba = infonce_logits(z_positive, z_anchor, temperature)
    B = z_anchor.size(0)
    targets = torch.arange(B, device=z_anchor.device)
    loss_a = F.cross_entropy(logits_ab, targets)
    loss_b = F.cross_entropy(logits_ba, targets)
    return 0.5 * (loss_a + loss_b)


def masked_supcon(embd, ids, reco_ids, temp=0.1):
    """
    embd  : (..., d) embeddings
    ids      : (...) original token ids
    reco_ids : (...) reconstructed token ids
    """

    z = embd.flatten(0, -2)
    ids = ids.flatten()
    reco_ids = reco_ids.flatten()

    # only include positions where reconstruction changed token
    mask = reco_ids != ids

    z = z[mask]
    ids = ids[mask]
    reco_ids = reco_ids[mask]

    if len(z) <= 1:
        return torch.tensor(0.0, device=embd.device)

    # normalize embeddings
    z = F.normalize(z, dim=-1)

    # pairwise similarities
    logits = z @ z.T
    logits = logits / temp

    # remove self-comparisons
    self_mask = torch.eye(len(z), device=z.device, dtype=torch.bool)

    # positives = same ORIGINAL token id
    pos_mask = (ids[:, None] == ids[None, :]) & (~self_mask)

    # log prob
    log_probs = F.log_softmax(logits.masked_fill(self_mask, -1e9), dim=1)

    # supervised contrastive loss
    pos_counts = pos_mask.sum(dim=1)

    valid = pos_counts > 0
    if not valid.any():
        return 0.0
    
    loss = -(log_probs * pos_mask.float()).sum(dim=1)

    loss = loss[valid] / pos_counts[valid]
    loss = loss.mean()

    return loss

def validate_rpn_syntax(token_ids: torch.Tensor) -> torch.Tensor:
    """
    Validate RPN syntax per sequence via stack-based evaluation.
    Uses token categories from embeddings.py to determine operator arity.
    Valid RPN ends with stack_depth = 1 (single value).
    
    Stack semantics by category:
      - SCALAR_CONST, VARIABLE: push value (stack += 1)
      - NONLINEAR_UNARY, LINEAR_DIFF, VECTOR_OP, JACOBIAN, MISC_OP: unary (pop 1, push 1)
      - BINARY_OP: binary (pop 2, push 1)
    
    Returns
    -------
    validity : (B,) binary tensor — 1.0 if valid, 0.0 if invalid
    """
    batch_size = token_ids.shape[0]
    validity = torch.zeros(batch_size, device=token_ids.device, dtype=torch.float32)
    
    for b in range(batch_size):
        stack_depth = 0
        valid = True
        
        for token_id in token_ids[b]:
            token_id_val = token_id.item()
            token = ID_TO_TOKEN.get(token_id_val, "__pad__")
            
            # Get token category
            category = TOKEN_TO_CAT.get(token, None)
            
            # Scalar constants and variables: push to stack
            if category in (
                TokenCategory.SCALAR_CONST,
                TokenCategory.VARIABLE
            ):
                stack_depth += 1
            # Unary operators: pop 1, push 1 (no net change, but need >= 1)
            elif category in (
                TokenCategory.NONLINEAR_UNARY,
                TokenCategory.LINEAR_DIFF,
                TokenCategory.VECTOR_OP,
                TokenCategory.MISC_OP,
            ):
                if stack_depth < 1:
                    valid = False
                    break
                # Stack depth unchanged (pop 1, push 1)
            # Binary operators: pop 2, push 1
            elif category in (
                TokenCategory.BINARY_OP,
                TokenCategory.JACOBIAN,
            ):
                if stack_depth < 2:
                    valid = False
                    break
                stack_depth -= 1  # pop 2, push 1
            # Padding: SKIP
            elif token == "__pad__":
                continue
            else:
                # Unknown token
                valid = False
                break
        
        # Valid if syntax check passed and final stack has exactly 1 element
        if valid and stack_depth == 1:
            validity[b] = 1.0
    
    return validity
    
class ContrastiveRPN(nn.Module):
    """
    End-to-end: RPN strings → pooled embedding → projection → InfoNCE vs
    algebra-augmented positives.
    """
    # TODO fix naming so pooled actually represents the pooled embedding not
    # the sequence embedding

    def __init__(
        self,
        seq_len: int = 100,
        embed_dim: int = 32,
        proj_dim: int = 64,
        sem_dim: int = 64,
        struct_dim: int = 64,
        rules: Optional[AlgebraicRuleSet] = None,
        temperature: float = 0.1,
        ae_type: str = "att",
    ):
        super().__init__()

        self.temperature = temperature
        self.embedder = RPNTokenEmbedder(embed_dim=embed_dim)
        
        RPN_AE = RPN_AE_.get(ae_type, RPN_AE_NMLP)
        self.head = RPN_AE(self.embedder, TOKEN_TO_ID, ID_TO_ARITY, seq_len, embed_dim, proj_dim)
        
        self.gen = RPN_GEN(proj_dim, sem_dim, struct_dim)
        
        self.use_rules = rules
        self.rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__pad__"])
            
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.criterion = lambda x_hat, x: ((x_hat - x).pow(2).mean() / ((x - x.mean(dim=(-1),keepdim=True)).pow(2).mean() + 1e-8))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def masked_criterion(self, pred: torch.Tensor, target: torch.Tensor,
                         key_padding_mask: torch.Tensor, scalar_mask: torch.Tensor) -> torch.Tensor:
        # Apply the padding mask to the loss
        # key_padding_mask: True = padding, False = real token
        w = 0.97  # Weight for real tokens
        mask = (~key_padding_mask).float() * w + key_padding_mask.float() * (1 - w)
        mask = mask[:, :, None].to(pred.device)
        
        scalar_mask = scalar_mask.to(pred.device)
        scalar_mask = scalar_mask[:, :, None].to(pred.device)
        
        pn = F.normalize(pred, p=2, dim=-1)
        tn = F.normalize(target, p=2, dim=-1)
        
        token_cos_dist = torch.mean((1 - torch.sum(pn * tn, dim=-1)) * mask[:,:,0]) / (pn.shape[-1] ** 0.5)
        
        scalar_mse = self.criterion(pred * scalar_mask, target * scalar_mask)
        
        return token_cos_dist, scalar_mse

    def encode_token_batch(
        self,
        token_ids: torch.Tensor,
        amp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pad_mask : (B, L) bool — True for non-padding positions (inverse of padding column).
        """
        pooled = self.embedder(token_ids, amp)
        return self.head(pooled, token_ids)

    def loss(
        self,
        rpns: Sequence[str],
    ) -> torch.Tensor:
        
        ### basic tokenization
        token_ids, amp = self.tokenize(rpns)
        device = self.embedder.token_embed.device
        token_ids = token_ids.to(device)
        amp = amp.to(device)
        
        ### compute padding mask for attention (True = mask out padding)
        key_padding_mask = (token_ids == TOKEN_TO_ID["__pad__"])
        scalar_mask = (token_ids == TOKEN_TO_ID["__scalar__"])
        
        ### encode original batch
        x = self.embedder(token_ids, amp)
        z_a = self.head(x, token_ids)
        
        loss = 0.0 
                
        ### pure reconstruction        
        decoded = self.head.reverse(z_a)
        d_token_ids = self._decode_tokens(decoded)[0].to(device)
        scalar_mask = scalar_mask.float() * (d_token_ids == TOKEN_TO_ID["__scalar__"]).float()
                
        loss_relmse = self.criterion(decoded, x)         
        denoise_distortion_loss_token, denoise_distortion_loss_scalar = \
            self.masked_criterion(decoded, x, 
                                  key_padding_mask, scalar_mask)
        denoise_distortion_loss_scalar = denoise_distortion_loss_scalar + loss_relmse
        loss = loss + denoise_distortion_loss_token + denoise_distortion_loss_scalar

        ### contrastive loss (simple) to avoid representation collapse among ids
        # want LLM to be able to distinguish ids...and not collapse them further
        # i.e. use supcon (Isola, supervised contrastive learning)
        # with positive pairs the same token ids, negative pairs different token ids
        # on the output of the decoder.
        # importantly both pairs are only when different from initial token
        masked_supcon_loss = 0.0
        if self.training:
            masked_supcon_loss = 0.1 * masked_supcon(x, token_ids, d_token_ids, self.temperature)
        loss = loss + masked_supcon_loss

        recoded = self.head(decoded, d_token_ids)
        denoise_perception_loss = self.criterion(recoded, z_a)
        loss = loss + denoise_perception_loss
        
        ### symmetry-based (algebra) generator
        denoise_loss = 0.0
        rule_loss = 0.0
        if self.use_rules or not self.training:
            r_token_ids, r_amp = self.rules.random_positive_view(token_ids, amp)
            # apply random rewrite to each expression in the batch, encode with same head, compute contrastive loss
            # truncate to what's available TODO check that this is properly padded and only padding is truncated
            # r_token_ids = r_token_ids[:,:self.seq_len,:]
            if r_token_ids.shape[1] == self.seq_len:
                r_token_ids = r_token_ids.to(device)
                r_amp = r_amp.to(device)                
                z_positive = self.encode_token_batch(r_token_ids, r_amp)
                # rule_loss = infonce_symmetric_loss(z_a, z_p, self.temperature)
                
                denoise_loss, rule_loss = self.gen.loss(z_a, z_positive)
                loss = loss + denoise_loss + rule_loss
        
        ### GRPO-style syntax reward: sample multiple rollouts and encourage valid ones
        syntax_loss = self._grpo_syntax_loss(z_a, x, device)
        loss = loss + syntax_loss
        
        w = ((key_padding_mask).float() * 0.03 + (~key_padding_mask).float() * 0.97)
        n = torch.sum(w)
        token_acc = torch.sum((token_ids == d_token_ids).float() * w) / n
        
        return loss, token_acc, masked_supcon_loss, \
            denoise_distortion_loss_token, denoise_distortion_loss_scalar, denoise_perception_loss, \
            syntax_loss, denoise_loss, rule_loss
    
    def _grpo_syntax_loss(
        self,
        z_a: torch.Tensor,
        pooled: torch.Tensor,
        device: torch.device,
        num_samples: int = 8,
    ) -> torch.Tensor:
        """
        GRPO-style loss: sample multiple decoded rollouts, compute syntax validity,
        then use relative advantages to encourage valid RPN generation.
        
        Parameters
        ----------
        z_a : (B, proj_dim) — encoded embeddings
        pooled : (B, seq_len, embed_dim) — pooled sequence representation
        num_samples : int — number of rollout samples per batch element
        
        Returns
        -------
        syntax_loss : scalar tensor
        """
        batch_size = pooled.shape[0]
        syntax_loss = torch.tensor(0.0, device=device)
        
        # Sample multiple decoded sequences
        validity_scores = []
        # reconstruction_errors = []
        
        for _ in range(num_samples): # TODO remove loop, deterministic
            decoded = self.head.reverse(self.gen.fm_gen(z_a.detach()))
            
            # Get token predictions
            token_ids_sample, _ = self._decode_tokens(decoded)
            
            # Validate syntax
            validity = validate_rpn_syntax(token_ids_sample)
            validity_scores.append(validity)
            
            # # Reconstruction error (lower is better)
            # recon_error = self.criterion(decoded, pooled)
            # reconstruction_errors.append(recon_error)
        
        # Stack validity scores: (num_samples, B)
        validity_scores = torch.stack(validity_scores, dim=0)  # (num_samples, B)
        # reconstruction_errors = torch.stack(reconstruction_errors, dim=0)  # (num_samples,)
        
        # Compute relative advantages (GRPO style)
        # Higher validity → lower loss
        mean_validity = validity_scores.mean(dim=0, keepdim=True)  # (1, B)
        validity_advantage = validity_scores - mean_validity  # (num_samples, B)
        
        # Weight samples by validity advantage: encourage high-validity samples
        # and penalize low-validity ones (relative to batch mean)
        # weighted_errors = reconstruction_errors.unsqueeze(-1) * (1.0 - validity_advantage)
        weighted_errors = (1.0 - validity_advantage)
        
        # Average over samples and batch
        syntax_loss = weighted_errors.mean()
        
        return syntax_loss
    
    def _decode_tokens(self, decoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to decode embeddings to token IDs. Returns token_ids, amplitudes."""
        amp = decoded.norm(dim=-1, keepdim=True)
        decoded_normalized = decoded / (amp + 1e-8)
        token_ids = self.embedder.token_embed.decode(decoded_normalized)

        return token_ids, amp.squeeze(-1)

    def tokenize(self, rpns: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize with :func:`batch_tokenize_rpn`."""
        token_ids, amp = batch_tokenize_rpn(
            rpns, max_len = self.seq_len
        )
        return token_ids, amp
    
    def detokenize(self, token_ids: torch.Tensor, amp: torch.Tensor) -> List[str]:
        """Convert token IDs back to RPN strings."""
        npy_ids = token_ids.detach().cpu().numpy()
        amps = amp.detach().cpu().numpy()
        
        rpns = []
        for seq_ids, seq_amp in zip(npy_ids, amps):
            rpn = []
            for token_id, a in zip(seq_ids, seq_amp):
                # Convert numpy int to Python int for safe dictionary lookup
                token_id_int = int(token_id)
                
                # Bounds check: skip if token_id is out of range
                if token_id_int < 0 or token_id_int >= len(ID_TO_TOKEN):
                    print(f"Warning: token_id {token_id_int} out of range [0, {len(ID_TO_TOKEN)})")
                    continue
                
                token = ID_TO_TOKEN.get(token_id_int, "__pad__")
                
                # Stop at padding
                if token == "__pad__":
                    continue
                
                # If scalar, use the amplitude value
                if token == "__scalar__" and abs(float(a)) > 1e-8:
                    rpn.append(f"{float(a):.6f}")
                else:
                    rpn.append(token)
            
            rpns.append(" ".join(rpn))
        
        return rpns
    
    def forward(
        self,
        rpns: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids, amp = self.tokenize(rpns)
        device = self.embedder.token_embed.device
        token_ids = token_ids.to(device)
        amp = amp.to(device)
        return self.gen.semantic(self.encode_token_batch(token_ids, amp)) # semantics for diffusion

    def decode(self, encoded):
        decoded = self.head.reverse(encoded)
        return self._decode_tokens(decoded)
    
    def sample(self, encoded):
        z = self.gen.encode(encoded)
        d = self.gen.gen(z)
        return d
    
    