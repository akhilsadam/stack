import math
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO extend to multidimensionality / matrix / vector / tensor equations
## needs to add physical dimension to token
## needs a PhysicalTensor class to extract tensor constants
## needs to handle tensor contractions and maybe Einstein summation notation...
## quite difficult for now as solver doesn't support either yet

# TokenSchema
@dataclass
class Token():
    """
    Token definition (by user)
    Note variables and constants have arity 0, unary ops have arity 1, binary ops have arity 2
    """
    def __init__(self, token: str, arity: int):
        self.token = token.strip().lower()
        self.arity = arity
        
@dataclass
class TokenRep():
    """
    Token definition (by user)
    Note variables and constants have arity 0, unary ops have arity 1, binary ops have arity 2
    """
    def __init__(self, _id: int, _mag: float):
        self._id = _id
        self._mag = _mag

@dataclass
class Vocab():
    def __init__(self, tokens: List[Token], seq_length: int = 64):
        self.tokens = \
            [Token("<unk>", 1), Token("<scalar>", 0), *tokens]
        self.token_to_id = {token.token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token.token for i, token in enumerate(self.tokens)}
        self.id_to_arity = {i: token.arity for i, token in enumerate(self.tokens)}

        self.pad_token_id = self.token_to_id["<unk>"]
        self.scalar_token_id = self.token_to_id["<scalar>"]
        
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.tokens)

    def rep_from_str(self, token_str: str):
        token_str = token_str.strip()
        try:
            f = float(token_str)
            return TokenRep(self.scalar_token_id, f)
        except ValueError:
            return TokenRep(self.token_to_id.get(token_str.lower(), self.pad_token_id), 0.0)

    def tokenize_str(self, string: str) -> List[TokenRep]:
        tokens = string.strip().split()
        out = [self.rep_from_str(token) for token in tokens]
        # Pad or truncate to seq_length
        if len(out) < self.seq_length:
            out.extend([TokenRep(self.pad_token_id, 0.0)] * (self.seq_length - len(out)))
        else:
            out = out[:self.seq_length]
        return out
    
    def detokenize_reps(self, reps: List[int], mags: List[float]) -> str:
        tokens = []
        for _id, _mag in zip(reps, mags):
            if _id == self.scalar_token_id:
                tokens.append(f'{_mag:.4g}')
            else:
                tokens.append(self.id_to_token.get(_id, "<UNK>"))
        
        return " ".join(tokens).replace('<unk>','').strip()
    
# TokenEmbedding
class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab: Vocab, embedding_dim: int = 8, physical_dim: int = 1):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.physical_dim = physical_dim
        self.embedding = nn.Embedding(len(vocab), embedding_dim - physical_dim) # concat with _mag
        self._init_embedding_weights()
        
        self.arity = torch.tensor([token.arity for token in vocab.tokens], dtype=torch.long)
        self.pad_id = vocab.pad_token_id
        self.device = 'cpu'
        
    def _init_embedding_weights(self):
        emb_dim = self.embedding.weight.shape[1]
        if emb_dim >= len(self.vocab):
            nn.init.orthogonal_(self.embedding.weight)
        else:
            nn.init.normal_(self.embedding.weight)

    def to(self, device):
        self.device = device
        self.embedding.to(device)
        return super().to(device)
        
    def forward(self, strings: List[str]) -> torch.Tensor:
        """
        strings: List of input strings
        returns: Tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        batch_size = len(strings)
        token_reps = [self.vocab.tokenize_str(s) for s in strings]

        _ids = torch.tensor([[rep._id for rep in reps] for reps in token_reps], dtype=torch.long, device=self.device)
        _mags = torch.tensor([[rep._mag for rep in reps] for reps in token_reps], dtype=torch.float, device=self.device).unsqueeze(-1)
        
        token_embeddings = F.normalize(self.embedding(_ids), p=2, dim=-1) # (batch_size, seq_length, embedding_dim - physical_dim)
        embeddings = torch.cat([token_embeddings, _mags], dim=-1) # (batch_size, seq_length, embedding_dim)
        
        return embeddings


    def reverse(self, embeddings: torch.Tensor) -> List[str]:
        """
        embeddings: Tensor of shape (batch_size, max_seq_len, embedding_dim)
        returns: List of output strings

        RPN/postfix stack decode. `count` = size of the stack of finished
        values: starts at 0, each token of arity a does count -= a; count += 1.
        A token is legal only if a <= count, and only if the result is still
        reachable to count == 1 in the steps that remain. PAD is legal once
        count == 1 and, once chosen, all later tokens are forced to PAD.
        """
        batch_size, max_seq_len, _ = embeddings.shape
        device = embeddings.device

        token_embeddings = embeddings[..., :-self.physical_dim]   # (B, T, D-1)
        mags = embeddings[..., -self.physical_dim:].squeeze(-1)   # (B, T)

        arity = self.arity.to(device)
        pad_id = self.pad_id
        max_arity = int(arity.max().item())

        out_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long, device=device)
        count = torch.zeros(batch_size, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_seq_len):
            chosen, count, finished = _decode_step(
                step_embedding=token_embeddings[:, t, :],
                vocab_weight=self.embedding.weight,
                count=count,
                finished=finished,
                arity=arity,
                pad_id=pad_id,
                remaining_steps_after=max_seq_len - t - 1,
                max_arity=max_arity,
            )
            out_ids[:, t] = chosen

        return [
            self.vocab.detokenize_reps(out_ids[b].tolist(), mags[b].tolist())
            for b in range(batch_size)
        ]
    
    # def reverse(self, embeddings: torch.Tensor) -> List[str]:
    #     """
    #     embeddings: Tensor of shape (batch_size, max_seq_len, embedding_dim)
    #     returns: List of output strings
    #     """
        
    #     # non autoregressive, doesn't consider _ids arity...
    #     _ids = torch.argmax(self.embedding.weight @ embeddings[..., :-self.physical_dim].transpose(-1, -2), dim=-1) # (batch_size, seq_length, embed_dim)
    #     _mags = embeddings[..., -self.physical_dim:].squeeze(-1) # (batch_size, seq_length)
        
    
def _reachable_count(remaining_steps_after: int, max_arity: int) -> int:
    """
    Best-case value `count` can be driven down to, given `remaining_steps_after`
    further tokens (not counting the one being chosen right now). Each step
    changes count by (1 - arity), so the fastest possible per-step decrease
    is (max_arity - 1). Note: when max_arity == 1, this correctly reduces to
    `reachable == 1` for any remaining_steps_after, since count can never
    move once it's above 1 -- no special-casing needed.
    """
    return 1 + remaining_steps_after * (max_arity - 1)


def _feasibility_mask(
    count: torch.Tensor,       # (B,)
    arity: torch.Tensor,       # (vocab_size,)
    remaining_steps_after: int,
    max_arity: int,
) -> torch.Tensor:
    """
    (B, vocab_size) bool mask, True where a token is ILLEGAL right now:
      (a) it would pop more operands than exist on the stack, or
      (b) taking it leaves the stack unreachable to count == 1 in the
          steps that remain after this one.
    Note: this does not special-case PAD -- its column gets overwritten
    by `_gate_pad` afterward, since PAD doesn't follow arity semantics
    (it freezes count rather than popping/pushing).
    """
    count_ = count.unsqueeze(-1)      # (B, 1)
    arity_ = arity.unsqueeze(0)       # (1, vocab_size)

    pops_too_much = arity_ > count_
    resulting_count = count_ - arity_ + 1
    too_slow = resulting_count > _reachable_count(remaining_steps_after, max_arity)

    return pops_too_much | too_slow


def _gate_pad(logits: torch.Tensor, pad_id: int, count: torch.Tensor) -> torch.Tensor:
    """PAD is legal exactly when the stack holds a single finished expression."""
    pad_legal = count == 1
    logits[:, pad_id] = torch.where(
        pad_legal, logits[:, pad_id], torch.full_like(logits[:, pad_id], float('-inf'))
    )
    return logits


def _decode_step(
    step_embedding: torch.Tensor,   # (B, D-1)
    vocab_weight: torch.Tensor,     # (vocab_size, D-1)
    count: torch.Tensor,            # (B,)
    finished: torch.Tensor,         # (B,) bool
    arity: torch.Tensor,            # (vocab_size,)
    pad_id: int,
    remaining_steps_after: int,
    max_arity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One autoregressive decode step. Returns (chosen_ids, new_count, new_finished)."""
    vocab_dirs = F.normalize(vocab_weight, p=2, dim=-1)   # unit-norm rows for cosine similarity
    logits = step_embedding @ vocab_dirs.T

    illegal = _feasibility_mask(count, arity, remaining_steps_after, max_arity)
    logits = logits.masked_fill(illegal, float('-inf'))
    logits = _gate_pad(logits, pad_id, count)

    chosen = torch.argmax(logits, dim=-1)
    chosen = torch.where(finished, torch.full_like(chosen, pad_id), chosen)

    is_pad = (chosen == pad_id) & (~finished)
    new_count = torch.where(finished | is_pad, count, count - arity[chosen] + 1)
    new_finished = finished | is_pad

    return chosen, new_count, new_finished        