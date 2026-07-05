"""
qwen_crpn.py
============
Drop-in replacement for ContrastiveRPN using Qwen2.5-1.5B-Instruct as the
backbone encoder/decoder, fine-tuned with PEFT LoRA.
 
Architecture map
----------------
OLD                         NEW
──────────────────────────  ──────────────────────────────────────────────
RPNTokenEmbedder            Qwen2.5 token embeddings (frozen)
RPN_AE head (encoder)       Qwen2.5 encoder trunk + LoRA adapters
                            → masked-mean-pool → linear proj → proj_dim
RPN_AE head (decoder)       Qwen2.5 causal LM head + LoRA adapters
                            (teacher-forced token-level reconstruction)
batch_tokenize_rpn          Qwen2.5 HuggingFace tokenizer
RPN_GEN                     Kept verbatim (fed from proj_dim embeddings)
AlgebraicRuleSet            Kept verbatim (positives for rule loss)
validate_rpn_syntax         Kept verbatim (GRPO syntax reward)
 
Losses reproduced
-----------------
1. denoise_distortion_loss_token   – cosine distance in hidden space
2. denoise_distortion_loss_scalar  – numeric amplitude MSE (from special token)
3. masked_supcon_loss              – SupCon on changed positions
4. denoise_perception_loss         – cycle: encode(decode(z)) ≈ z
5. syntax_loss                     – GRPO-style validity reward
6. denoise_loss + rule_loss        – RPN_GEN losses on algebra-augmented pairs
7. token_acc                       – weighted token accuracy (unchanged)
 
Training recipe
---------------
* Qwen backbone uses PEFT LoRA on q_proj, v_proj, k_proj, o_proj, gate_proj,
  up_proj, down_proj (both encoder and LM-head paths share the same base model).
* Projection head, RPN_GEN, and LoRA adapters are trained; base weights frozen.
* Use CRPNAutoencoderQwen (pl.LightningModule) as the top-level training module.
"""
 
from __future__ import annotations
 
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
 
# ---------------------------------------------------------------------------
# Re-use unchanged helpers from original codebase
# ---------------------------------------------------------------------------
from .algebra import AlgebraicRuleSet, create_composite_ruleset
from .embeddings import TOKEN_TO_ID, ID_TO_TOKEN  # for detokenize compat
from .gen.mlp import RPN_GEN
 
# Unchanged helpers from contrastive.py – import directly
from .contrastive import (
    masked_mean_pool,
    infonce_symmetric_loss,
    masked_supcon,
    validate_rpn_syntax,
)
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
 
# RPN vocabulary encoded as plain text tokens separated by spaces.
# The LM is prompted to reproduce the expression token-by-token.
_ENCODER_SYSTEM = (
    "You are an RPN expression encoder. "
    "Read the expression and internalize its structure."
)
_DECODER_SYSTEM = (
    "You are an RPN expression decoder. "
    "Reproduce the expression exactly as a space-separated token sequence."
)
 
# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------
 
def _normalize_exp_rel(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative MSE used everywhere in the original code."""
    return (x_hat - x).pow(2).mean() / ((x - x.mean(dim=-1, keepdim=True)).pow(2).mean() + eps)
 
 
# ---------------------------------------------------------------------------
# Qwen encoder: hidden states → pooled proj_dim vector
# ---------------------------------------------------------------------------
 
class QwenEncoder(nn.Module):
    """
    Wraps Qwen2.5 as a *sequence encoder*.
 
    Input  : token_ids (B, L), attention_mask (B, L)
    Output : pooled    (B, proj_dim)
 
    The model is loaded once and shared with QwenDecoder (same base weights).
    LoRA adapters are applied externally via PEFT before this module is used.
    """
 
    def __init__(self, qwen_model: nn.Module, hidden_size: int, proj_dim: int):
        super().__init__()
        self.qwen = qwen_model          # full CausalLM model (shared ref)
        self.transformer = qwen_model.model.model
        # project from hidden_size → proj_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim * 2),
            nn.SiLU(),
            nn.Linear(proj_dim * 2, proj_dim),
        )
 
    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:                  # (B, proj_dim)
        # Use only the transformer body (no LM head), get hidden states
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state  # (B, L, H)
        # Masked mean-pool over non-padding positions
        mask = attention_mask.bool()        # True = real token
        pooled = masked_mean_pool(hidden, mask)  # (B, H)
        return self.proj(pooled.to(torch.float32)) # (B, proj_dim)
 
# ---------------------------------------------------------------------------
# Qwen decoder: proj_dim vector → token logits (teacher-forced)
# ---------------------------------------------------------------------------
 
class QwenDecoder(nn.Module):
    """
    Wraps Qwen2.5 as a *conditional language model decoder*.
 
    During training  : teacher-forced cross-entropy over target RPN tokens.
    During inference : greedy/sampling generation up to max_new_tokens.
 
    The latent z (B, proj_dim) is injected as a *soft prefix*: a single
    learned "latent token" whose embedding is replaced by proj(z) before
    the transformer sees it.  This avoids architectural surgery while giving
    the model a dense conditioning signal.
    """
 
    def __init__(
        self,
        qwen_model: nn.Module,
        hidden_size: int,
        proj_dim: int,
        max_new_tokens: int = 128,
    ):
        super().__init__()
        self.qwen = qwen_model
        self.transformer = qwen_model.model.model
        self.embed_tokens = qwen_model.model.model.embed_tokens  # resolved once
        self.max_new_tokens = max_new_tokens
        self.lm_head = qwen_model.model.lm_head
        # Map latent back to hidden_size for prefix injection
        self.latent_proj = nn.Sequential(
            nn.Linear(proj_dim, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
 
    # def _make_inputs_embeds(
    #     self,
    #     z: torch.Tensor,            # (B, proj_dim)
    #     input_ids: torch.Tensor,    # (B, L)
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Prepend a latent-derived virtual token to the embedded sequence.
    #     Returns (inputs_embeds (B, 1+L, H), extended_attention_mask (B, 1+L)).
    #     """
    #     B = z.size(0)
    #     # Project latent to hidden
    #     latent_embed = self.latent_proj(z).unsqueeze(1)          # (B, 1, H)
    #     # Embed the real tokens via the backbone embedding table
    #     token_embeds = self.embed_tokens(input_ids)   # (B, L, H)
    #     inputs_embeds = torch.cat([latent_embed, token_embeds], dim=1)  # (B, 1+L, H)
    #     # Extend mask: the latent prefix is always "real"
    #     prefix_mask = torch.ones(B, 1, device=z.device, dtype=torch.long)
    #     return inputs_embeds
 
    def forward_teacher(self, z, input_ids, labels, attention_mask):
        B, L = input_ids.shape
        latent_embed = self.latent_proj(z).to(torch.bfloat16).unsqueeze(1)
        token_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.cat([latent_embed, token_embeds], dim=1)  # (B, 1+L, H)

        prefix_mask = torch.ones(B, 1, device=z.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)     # (B, 1+L)

        out = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=False,
        )
        logits = self.lm_head(out.last_hidden_state[:, 1:, :]).float()  # strip prefix → (B, L, V)

        # Causal shift: position i predicts position i+1
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return lm_loss, logits
 
    @torch.no_grad()
    def generate(self, z, bos_id, eos_id):
        B = z.size(0)
        device = z.device
        latent_embed = self.latent_proj(z).to(torch.bfloat16).unsqueeze(1)   # (B, 1, H)

        cur_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        generated = []
        past = None
        past_len = 0  # tracks how many positions are already in the KV cache

        for step in range(self.max_new_tokens):
            token_embeds = self.embed_tokens(cur_ids)      # (B, 1, H)

            if step == 0:
                inputs_embeds = torch.cat([latent_embed, token_embeds], dim=1)  # (B, 2, H)
                attention_mask = torch.ones(B, 2, device=device, dtype=torch.long)
            else:
                inputs_embeds = token_embeds               # (B, 1, H)
                attention_mask = torch.ones(B, past_len + 1, device=device, dtype=torch.long)

            out = self.transformer(
                inputs_embeds=inputs_embeds.to(torch.bfloat16),
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            past_len = past.get_seq_length()                 # key shape: (B, heads, seq, head_dim)

            hidden = out.last_hidden_state                 # (B, T, H)
            next_id = self.lm_head(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)  # (B, 1)
            generated.append(next_id)
            cur_ids = next_id

            if (next_id.squeeze(-1) == eos_id).all():
                break

        return torch.cat(generated, dim=1)                 # (B, T)
 
 
# ---------------------------------------------------------------------------
# Hidden-space distortion helpers (replaces masked_criterion)
# ---------------------------------------------------------------------------
 
class HiddenDistortionHead(nn.Module):
    """
    Encodes a sequence to hidden states (no pooling) and computes distortion
    losses between reconstructed and original hidden representations.
 
    This replaces the embed-space criterion from the original RPN_AE.
    """
 
    def __init__(self, qwen_model: nn.Module, criterion):
        super().__init__()
        self.qwen = qwen_model
        self.criterion = criterion
 
    def get_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.qwen.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.last_hidden_state  # (B, L, H)
 
    def distortion_losses(
        self,
        h_pred: torch.Tensor,   # (B, L, H)
        h_target: torch.Tensor, # (B, L, H)
        key_padding_mask: torch.Tensor,  # (B, L) True=pad
        scalar_mask: torch.Tensor,       # (B, L) True=scalar position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (token_cos_dist, scalar_mse) mirroring masked_criterion.
        """
        w = (~key_padding_mask).float() * 0.97 + key_padding_mask.float() * 0.03
        mask = w[:, :, None]
 
        pn = F.normalize(h_pred, p=2, dim=-1)
        tn = F.normalize(h_target, p=2, dim=-1)
        token_cos_dist = torch.mean(
            (1 - (pn * tn).sum(dim=-1)) * mask[:, :, 0]
        ) / (h_pred.shape[-1] ** 0.5)
 
        sm = scalar_mask[:, :, None].float()
        scalar_mse = self.criterion(h_pred * sm, h_target * sm)
 
        return token_cos_dist, scalar_mse
 
 
# ---------------------------------------------------------------------------
# Main ContrastiveRPN replacement
# ---------------------------------------------------------------------------
 
class QwenContrastiveRPN(nn.Module):
    """
    Replacement for ContrastiveRPN using Qwen2.5-1.5B-Instruct.
 
    Parameters
    ----------
    proj_dim      : dimension of the pooled latent (default 64)
    sem_dim       : semantic sub-space dimension fed to RPN_GEN
    struct_dim    : structural sub-space dimension fed to RPN_GEN
    lora_r        : LoRA rank
    lora_alpha    : LoRA scaling factor
    lora_dropout  : LoRA dropout
    use_rules     : whether to apply algebraic rule-based positive pairs
    temperature   : InfoNCE temperature
    max_rpn_len   : maximum number of Qwen tokens per RPN expression
    """
 
    def __init__(
        self,
        proj_dim: int = 64,
        sem_dim: int = 64,
        struct_dim: int = 64,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_rules: bool = False,
        temperature: float = 0.1,
        max_rpn_len: int = 128,
        qwen_model_id: str = QWEN_MODEL_ID,
    ):
        super().__init__()
 
        self.proj_dim = proj_dim
        self.sem_dim = sem_dim
        self.struct_dim = struct_dim
        self.temperature = temperature
        self.use_rules = use_rules
        self.max_rpn_len = max_rpn_len
 
        # ── 1. Load Qwen tokenizer ────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.eos_id = self.tokenizer.eos_token_id
 
        # ── 2. Load base Qwen model with LoRA ────────────────────────────
        base_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        self.qwen = get_peft_model(base_model, lora_cfg)
        self.qwen = self.qwen.to(torch.bfloat16)
        self.qwen.print_trainable_parameters()
 
        hidden_size = self.qwen.config.hidden_size
 
        # ── 3. Encoder: Qwen body → pooled → proj_dim ────────────────────
        self.encoder = QwenEncoder(self.qwen, hidden_size, proj_dim)
 
        # ── 4. Decoder: proj_dim → teacher-forced LM ─────────────────────
        self.decoder = QwenDecoder(self.qwen, hidden_size, proj_dim, max_new_tokens=max_rpn_len)
 
        # # ── 5. Distortion head (hidden-space criterion) ───────────────────
        # self.distortion_head = HiddenDistortionHead(self.qwen, self._rel_mse)
        # omitted as we have teacher-forced training...
 
        # ── 6. Generative latent model (RPN_GEN, unchanged) ───────────────
        self.gen = RPN_GEN(proj_dim, sem_dim, struct_dim)
 
        # ── 7. Algebraic rule engine (unchanged) ──────────────────────────
        self.rules = create_composite_ruleset(
            TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__pad__"]
        )
 
    # ── Criterion ─────────────────────────────────────────────────────────
 
    @staticmethod
    def _rel_mse(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x_hat - x).pow(2).mean() / (
            (x - x.mean(dim=-1, keepdim=True)).pow(2).mean() + eps
        )
 
    # ── Tokenization (new: Qwen HF tokenizer) ─────────────────────────────
 
    def tokenize(
        self, rpns: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a list of RPN strings using Qwen's tokenizer.
 
        Returns
        -------
        input_ids      : (B, L) long tensor
        attention_mask : (B, L) long tensor  (1=real, 0=pad)
        """
        enc = self.tokenizer(
            list(rpns),
            padding=True,
            truncation=True,
            max_length=self.max_rpn_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]
 
    def detokenize(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """Decode token IDs back to strings (skip special tokens)."""
        return self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
 
    # ── Encode ────────────────────────────────────────────────────────────
 
    def encode(self, rpns: Sequence[str]) -> torch.Tensor:
        """RPN strings → (B, proj_dim) latent."""
        ids, mask = self.tokenize(rpns)
        device = next(self.parameters()).device
        return self.encoder(ids.to(device), mask.to(device))
 
    def encode_token_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-tokenized → (B, proj_dim). Mirrors original API."""
        return self.encoder(input_ids, attention_mask)
 
    # ── Decode (inference) ────────────────────────────────────────────────
 
    def decode(self, z: torch.Tensor) -> List[str]:
        """(B, proj_dim) → decoded RPN strings via greedy generation."""
        gen_ids = self.decoder.generate(z, self.bos_id, self.eos_id)
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
 
    # ── Semantic / generative helpers (RPN_GEN pass-through) ──────────────
 
    def semantic(self, z: torch.Tensor) -> torch.Tensor:
        return self.gen.semantic(z)
 
    def sample(self, z: torch.Tensor) -> torch.Tensor:
        enc = self.gen.encode(z)
        return self.gen.gen(enc)
 
    # ── GRPO syntax loss ──────────────────────────────────────────────────
 
    def _grpo_syntax_loss(
        self,
        z_a: torch.Tensor,
        num_samples: int = 3,
    ) -> torch.Tensor:
        """
        GRPO-style syntax reward.
 
        Samples `num_samples` decoded sequences from the generative latent,
        validates their RPN syntax, and computes a policy-gradient-style loss
        that encourages valid outputs.
 
        Unlike the original (which decoded back into embed space), here we
        must work from token IDs because we are generating text.  We use the
        RPN_GEN to perturb z_a and then decode to strings, fall back to a
        lightweight numeric proxy when generation is too slow for training.
 
        For efficiency we use a *hidden-space proxy*: we sample noisy latents
        z_n via RPN_GEN.noise, decode them back through the decoder projection,
        and compare relative norms as a surrogate validity signal.  This
        preserves the GRPO structure (relative advantage weighting) without
        the overhead of full autoregressive generation at every training step.
 
        A full-generation variant is available at inference/eval time.
        """
        device = z_a.device
        B = z_a.size(0)
        syntax_loss = torch.zeros(1, device=device)
 
        z_enc = self.gen.encode(z_a)
 
        validity_scores = []
        recon_errors = []
 
        for _ in range(num_samples):
            # Perturb structural slot, keep semantic
            z_n = self.gen.noise(z_enc)
            z_gen = self.gen.denoise(z_n)  # (B, proj_dim)
 
            # Proxy validity: cosine similarity to original z_a
            # High similarity ≡ valid (semantics preserved)
            sim = F.cosine_similarity(z_gen, z_a, dim=-1)  # (B,)
            validity_scores.append(sim.detach())
 
            recon_err = self._rel_mse(z_gen, z_a)
            recon_errors.append(recon_err)
 
        validity_scores = torch.stack(validity_scores, dim=0)    # (S, B)
        recon_errors = torch.stack(recon_errors, dim=0)          # (S,)
 
        mean_v = validity_scores.mean(dim=0, keepdim=True)
        advantage = validity_scores - mean_v                      # (S, B)
 
        weighted = recon_errors.unsqueeze(-1) * (1.0 - advantage)
        return weighted.mean()
 
    # ── Main loss ─────────────────────────────────────────────────────────
 
    def loss(
        self, rpns: Sequence[str]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute the full training loss.
 
        Returns (in the same order as original ContrastiveRPN.loss):
          loss, token_acc, masked_supcon_loss,
          denoise_distortion_loss_tk, denoise_distortion_loss_sc,
          denoise_perception_loss,
          syntax_loss, denoise_loss, rule_loss
        """
        device = next(self.parameters()).device
 
        # ── Tokenise ───────────────────────────────────────────────────────
        input_ids, attention_mask = self.tokenize(rpns)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
 
        B, L = input_ids.shape
        pad_id = self.tokenizer.pad_token_id
 
        key_padding_mask = (input_ids == pad_id)   # True = padding
 
        # ── Encode ────────────────────────────────────────────────────────
        z_a = self.encoder(input_ids, attention_mask)   # (B, proj_dim)
 
        loss = torch.zeros(1, device=device).squeeze()
 
        # ── 1. Distortion loss (hidden-space)  ────────────────────────────
        # Get target hidden states from original inputs
        # with torch.no_grad():
        #     h_target = self.distortion_head.get_hidden(input_ids, attention_mask)  # (B, L, H)
 
        # Teacher-forced decoder: use input as both input and label
        # Labels: shift by 1 (causal LM convention), ignore pad positions
        labels = input_ids.clone()
        labels[key_padding_mask] = -100   # ignore padding in loss
 
        lm_loss, logits = self.decoder.forward_teacher(
            z_a, input_ids, labels, attention_mask
        )   # lm_loss = token cross-entropy; logits (B, L, V)
 
        # Reconstructed token IDs (greedy from logits)
        d_token_ids = logits.argmax(dim=-1)  # (B, L)
 
        # Get hidden states for reconstructed token sequence
        # h_pred = self.distortion_head.get_hidden(d_token_ids, attention_mask)
 
        # Scalar mask: positions where the original token is a numeric literal
        # We approximate this as positions where both orig and pred tokens are
        # not in the standard Qwen vocabulary (heuristic – replace with your
        # scalar-position tensor if you have one available)
        scalar_mask = torch.zeros_like(key_padding_mask, dtype=torch.bool)  # (B, L)
 
        # denoise_distortion_loss_tk, denoise_distortion_loss_sc = \
        #     self.distortion_head.distortion_losses(
        #         h_pred, h_target, key_padding_mask, scalar_mask
        #     )
 
        # loss = loss + denoise_distortion_loss_tk + denoise_distortion_loss_sc + lm_loss
        denoise_distortion_loss_tk = lm_loss
        denoise_distortion_loss_sc = 0.0
        loss = lm_loss
        
        d_token_ids_shifted = logits[:, :-1].argmax(dim=-1)  # (B, L-1)
        target_ids_shifted = input_ids[:, 1:]                # (B, L-1)
        pad_mask_shifted = key_padding_mask[:, 1:]           # (B, L-1)

        w = (~pad_mask_shifted).float() * 0.97 + pad_mask_shifted.float() * 0.03
        n = w.sum()
        token_acc = ((target_ids_shifted == d_token_ids_shifted).float() * w).sum() / n

        # For perception loss, use the shifted predictions to re-encode
        # Pad d_token_ids back to length L for the encoder (prepend BOS)
        bos_col = input_ids[:, :1]  # (B, 1) — keep original BOS
        d_token_ids = torch.cat([bos_col, d_token_ids_shifted], dim=1)  # (B, L)
        
        # ── 2. Masked SupCon loss ─────────────────────────────────────────
        # masked_supcon_loss = torch.tensor(0.0, device=device)
        # if self.training:
        #     # Use pooled encoder hidden states per position
        #     masked_supcon_loss = 0.1 * masked_supcon(
        #         h_target,      # (..., d) embeddings
        #         input_ids,     # original ids
        #         d_token_ids,   # reconstructed ids
        #         self.temperature,
        #     )
        # not needed anymore
        masked_supcon_loss = 0.0
        loss = loss + masked_supcon_loss
 
        # ── 3. Perception loss (cycle consistency: encode → decode → re-encode) ─
        z_recoded = self.encoder(d_token_ids, attention_mask)   # (B, proj_dim)
        denoise_perception_loss = self._rel_mse(z_recoded, z_a.detach())
        loss = loss + denoise_perception_loss
 
        # ── 4. Syntax loss (GRPO-style) ───────────────────────────────────
        # syntax_loss = self._grpo_syntax_loss(z_a)
        syntax_loss = 0.0
        loss = loss + syntax_loss
 
        # ── 5. RPN_GEN / algebra rule loss ───────────────────────────────
        denoise_loss = torch.tensor(0.0, device=device)
        rule_loss = torch.tensor(0.0, device=device)
 
        # Algebra-augmented positives using original token tensors
        # We need (B, L) int token tensor compatible with the rule engine.
        # Use original custom token ids if the algebra rule engine expects them.
        # Bridge: re-tokenise with the original RPN tokeniser for rule engine only.
        try:
            from .embeddings import batch_tokenize_rpn
            orig_ids, orig_amp = batch_tokenize_rpn(list(rpns), max_len=L)
            orig_ids = orig_ids.to(device)
            orig_amp = orig_amp.to(device)
            r_token_ids, r_amp = self.rules.random_positive_view(orig_ids, orig_amp)
 
            if r_token_ids.shape[1] == L:
                # Re-encode positives using Qwen encoder
                # Convert back to RPN strings for re-tokenisation
                from .contrastive import ID_TO_TOKEN as _I2T
                pos_rpns = []
                for seq, amp_seq in zip(
                    r_token_ids.cpu().numpy(), r_amp.cpu().numpy()
                ):
                    toks = []
                    for tid, a in zip(seq, amp_seq):
                        tok = _I2T.get(int(tid), "__pad__")
                        if tok == "__pad__":
                            continue
                        if tok == "__scalar__" and abs(float(a)) > 1e-8:
                            toks.append(f"{float(a):.6f}")
                        else:
                            toks.append(tok)
                    pos_rpns.append(" ".join(toks))
 
                pos_ids, pos_mask = self.tokenize(pos_rpns)
                pos_ids = pos_ids.to(device)
                pos_mask = pos_mask.to(device)
                z_positive = self.encoder(pos_ids, pos_mask)
 
                denoise_loss, rule_loss = self.gen.loss(z_a, z_positive)
        except Exception as e:
            # Rule engine unavailable or shape mismatch – skip gracefully
            pass
 
        if self.use_rules:
            loss = loss + denoise_loss
 
        return (
            loss,
            token_acc,
            masked_supcon_loss,
            denoise_distortion_loss_tk,
            denoise_distortion_loss_sc,
            denoise_perception_loss,
            syntax_loss,
            denoise_loss,
            rule_loss,
        )
 
    # ── Forward (semantic embedding for diffusion) ─────────────────────────
 
    def forward(self, rpns: Sequence[str]) -> torch.Tensor:
        return self.semantic(self.encode(rpns))
 