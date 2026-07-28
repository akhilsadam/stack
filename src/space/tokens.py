import math
import random
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# from .arity import get_tree_coords

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

class Node:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity
        self.children = []

    def to_rpn(self):
        rpn = []
        for child in self.children:
            rpn.extend(child.to_rpn())
        rpn.append(self.name)
        return rpn

@dataclass
class Vocab():
    def __init__(self, tokens: List[Token]):
        self.tokens = \
            [Token("<unk>", 1), Token("<scalar>", 0), *tokens]
        self.token_to_id = {token.token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token.token for i, token in enumerate(self.tokens)}
        self.id_to_arity = {i: token.arity for i, token in enumerate(self.tokens)}

        self.pad_token_id = self.token_to_id["<unk>"]
        self.scalar_token_id = self.token_to_id["<scalar>"]
        
    def __len__(self):
        return len(self.tokens)

    def rep_from_str(self, token_str: str):
        token_str = token_str.strip()
        try:
            f = float(token_str)
            return TokenRep(self.scalar_token_id, f)
        except ValueError:
            return TokenRep(self.token_to_id.get(token_str.lower(), self.pad_token_id), 0.0)

    def tokenize_str(self, string: str, seq_length: int = 64) -> List[TokenRep]:
        tokens = string.strip().split()
        out = [self.rep_from_str(token) for token in tokens]
        # Pad or truncate to seq_length
        if len(out) < seq_length:
            out.extend([TokenRep(self.pad_token_id, 0.0)] * (seq_length - len(out)))
        else:
            out = out[:seq_length]
        return out
    
    def detokenize_reps(self, reps: List[int], mags: List[float]) -> str:
        tokens = []
        for _id, _mag in zip(reps, mags):
            if _id == self.scalar_token_id:
                tokens.append(f'{_mag:.4g}')
                # tokens.append(f'1.0') # debug
            else:
                tokens.append(self.id_to_token.get(_id, "<UNK>"))
        
        return " ".join(tokens).replace('<unk>','').strip()
    
# TokenEmbedding
class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab: Vocab, seq_len: int = 64, embed_dim: int = 8, phys_dim: int = 1):
        super().__init__()
        self.vocab = vocab
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.phys_dim = phys_dim
        self.embedding = nn.Embedding(len(vocab), embed_dim - phys_dim) # concat with _mag
        self._init_embedding_weights()
        
        self.arity = torch.tensor([token.arity for token in vocab.tokens], dtype=torch.long)
        self.pad_id = vocab.pad_token_id
        self.device = 'cpu'
        
    def _init_embedding_weights(self):
        if self.embed_dim >= len(self.vocab):
            nn.init.orthogonal_(self.embedding.weight)
        else:
            nn.init.normal_(self.embedding.weight)

    def dequantize(self, _ids):
        emb = F.normalize(self.embedding(_ids), p=2, dim=-1)
        if self.training:
            weights = F.normalize(self.embedding.weight, p=2, dim=-1).detach()
            
            cos_sim = torch.mm(weights, weights.t())
            
            # Clamp for numerical stability before arccos
            cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # 3. Convert to angular distances (radians)
            angles = torch.acos(cos_sim)
            
            # Ignore self-angle (0.0) by filling the diagonal with a large number
            angles.fill_diagonal_(float('inf'))
            
            # 4. Find the minimum angular distance to the closest neighbor
            min_angles = angles.min(dim=1).values
            
            # 5. Extract the specific angular limits for this batch
            batch_min_angle = min_angles[_ids]
            
            # 6. Maximum safe rotation angle is exactly half the distance to the neighbor
            # Apply a 0.95 safety factor so it never perfectly touches the boundary
            max_rotation = (batch_min_angle / 2.0) * 0.95
            
            # 7. Sample a random noise angle uniformly distributed in [-max_rotation, max_rotation]
            # Reshape to [batch, seq_len, 1] for broadcasting
            noise_angle = (torch.rand_like(batch_min_angle) * 2.0 - 1.0) * max_rotation
            noise_angle = noise_angle.unsqueeze(-1)
            
            # 8. Generate a random direction vector orthogonal to the embedding
            raw_noise = torch.randn_like(emb)
            # Project out the component parallel to 'emb' to make it strictly orthogonal
            orthogonal_dir = raw_noise - torch.sum(raw_noise * emb, dim=-1, keepdim=True) * emb
            orthogonal_dir = F.normalize(orthogonal_dir, p=2, dim=-1)
            
            return emb * torch.cos(noise_angle) + orthogonal_dir * torch.sin(noise_angle)
        else:
            return emb

    def to(self, device):
        self.device = device
        self.embedding.to(device)
        return super().to(device)
        
    def fwd(self, strings: List[str], noisy=False) -> torch.Tensor:
        """
        strings: List of input strings
        returns: Tensor of shape (batch_size, max_seq_len, embed_dim)
        """
        batch_size = len(strings)
        token_reps = [self.vocab.tokenize_str(s, self.seq_len) for s in strings]

        _ids = torch.tensor([[rep._id for rep in reps] for reps in token_reps], dtype=torch.long, device=self.device)
        _mags = torch.tensor([[rep._mag for rep in reps] for reps in token_reps], dtype=torch.float, device=self.device).unsqueeze(-1)
        
        # if noisy:
        #     token_embeddings = F.normalize(self.dequantize(_ids), p=2, dim=-1)  # (batch_size, seq_length, embed_dim - phys_dim)
        # else:
        token_embeddings = F.normalize(self.embedding(_ids), p=2, dim=-1)
        embeddings = torch.cat([token_embeddings, torch.tanh(_mags)], dim=-1) # (batch_size, seq_length, embed_dim)
        
        return embeddings, _ids

    def forward(self, strings):
        return self.fwd(strings)[0]

    # def forward_with_depth(self, strings):
    #     emb, ids = self.fwd(strings)
    #     arity = self.arity[ids]
    #     return emb, *get_tree_coords(arity)

    def generate(self, noise):
        tok_noise = noise[:, :, :-1]
        mag_noise = noise[:, :, -1:]
        tok_noise = F.normalize(tok_noise, p=2, dim=-1)
        return torch.cat([tok_noise, mag_noise], dim=-1)
    
    def reverse(self, embeddings: torch.Tensor, max_seq_len: Optional[Union[int, torch.Tensor]] = None) -> List[str]:
        """
        embeddings: Tensor of shape (batch_size, current_seq_len, embed_dim)
        max_seq_len: Can be an int, or a 1D tensor of shape (batch_size,) containing 
                     the randomized target generation length for each batch item.
                     Defaults to current_seq_len if None.
        returns: List of output strings
        """
        batch_size, current_seq_len, _ = embeddings.shape
        device = embeddings.device

        # 1. Resolve random or variable target lengths per batch item
        if max_seq_len is None:
            target_lens = torch.full((batch_size,), current_seq_len, dtype=torch.long, device=device)
        elif isinstance(max_seq_len, torch.Tensor):
            target_lens = max_seq_len.to(device=device, dtype=torch.long)
        else: # plain integer
            target_lens = torch.full((batch_size,), max_seq_len, dtype=torch.long, device=device)

        # The absolute maximum loop boundary required across this batch execution
        absolute_max_len = int(target_lens.max().item())

        token_embeddings = embeddings[..., :-self.phys_dim]   # (B, current_seq_len, D-1)
        mags = embeddings[..., -self.phys_dim:].squeeze(-1)   # (B, current_seq_len)
        mags = torch.atanh(torch.clamp(mags, -1 + 1e-4, 1 - 1e-4))

        arity = self.arity.to(device)
        pad_id = self.pad_id
        max_arity = int(arity.max().item())

        out_ids = torch.full((batch_size, absolute_max_len), pad_id, dtype=torch.long, device=device)
        count = torch.zeros(batch_size, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(absolute_max_len):
            if finished.all():
                break

            # Handle indexing safety if we're evaluating beyond the input tensor frames
            embedding_idx = min(t, current_seq_len - 1)
            step_emb = token_embeddings[:, embedding_idx, :]

            # Force individual rows to finish if they've hit their specific randomized target length limit
            hit_target_limit = (t >= target_lens)
            finished = finished | hit_target_limit

            # Dynamically tracking remaining steps per batch item:
            # remaining_steps_after = target_len - current_index - 1
            remaining_steps = torch.clamp(target_lens - t - 1, min=0)

            # --- Batched Evaluation Step (Inlined & Tensorized to support variable remaining_steps) ---
            vocab_dirs = F.normalize(self.embedding.weight, p=2, dim=-1)   
            logits = step_emb @ vocab_dirs.T

            # Tensorized feasibility mask evaluation
            count_ = count.unsqueeze(-1)          # (B, 1)
            arity_ = arity.unsqueeze(0)           # (1, vocab_size)
            rem_steps_ = remaining_steps.unsqueeze(-1) # (B, 1)

            pops_too_much = arity_ > count_
            resulting_count = count_ - arity_ + 1
            
            # 1 + remaining_steps * (max_arity - 1) calculated independently per batch row
            reachable_count = 1 + rem_steps_ * (max_arity - 1)
            too_slow = resulting_count > reachable_count
            illegal = pops_too_much | too_slow

            logits = logits.masked_fill(illegal, float('-inf'))
            logits = _gate_pad(logits, pad_id, count)

            chosen = torch.argmax(logits, dim=-1)
            chosen = torch.where(finished, torch.full_like(chosen, pad_id), chosen)

            is_pad = (chosen == pad_id) & (~finished)
            count = torch.where(finished | is_pad, count, count - arity[chosen] + 1)
            finished = finished | is_pad
            # -----------------------------------------------------------------------------------------

            out_ids[:, t] = chosen

        # Dynamic string detokenization based on each row's unique evaluated length slice
        out_strings = []
        for b in range(batch_size):
            row_target_len = int(target_lens[b].item())
            row_ids = out_ids[b, :row_target_len].tolist()
            
            # Reconstruct trailing magnitude arrays based on individual row lengths safely
            if row_target_len <= current_seq_len:
                row_mags = mags[b, :row_target_len].tolist()
            else:
                row_mags = mags[b].tolist() + [0.0] * (row_target_len - current_seq_len)
                
            out_strings.append(self.vocab.detokenize_reps(row_ids, row_mags))

        return out_strings

    def generate_random_rpn(self, max_high_arity_nodes: int = 4, num_unary: int = 2) -> str:
        """
        Generates a valid RPN sequence string dynamically using the initialized vocabulary.
        Extracts operators and leaves based strictly on their defined arity.
        """

        # 1. Group vocabulary by arity dynamically
        operators_by_arity = {}
        for token_id, token_str in self.vocab.id_to_token.items():
            if token_id == self.vocab.pad_token_id:
                continue
            ar = self.vocab.id_to_arity[token_id]
            if ar not in operators_by_arity:
                operators_by_arity[ar] = []
            operators_by_arity[ar].append(token_str)

        high_arities = sorted([k for k in operators_by_arity.keys() if k >= 2], reverse=True)
        
        # 2. Sample operator counts for k >= 2
        counts = {}
        total_high_nodes = 0
        for k in high_arities:
            c = random.randint(0, max(0, max_high_arity_nodes - total_high_nodes))
            counts[k] = c
            total_high_nodes += c

        # Enforce at least one high-arity operator to build a tree if we only drew 0s
        if total_high_nodes == 0 and 2 in operators_by_arity:
            counts[2] = 1

        # 3. Compute exact required leaves based on sampled higher-arity nodes
        num_leaves = 1 + sum((k - 1) * count for k, count in counts.items())
        pool = [Node("LEAF", arity=0) for _ in range(num_leaves)]

        # 4. Build skeleton tree (bottom-up leaf pairing)
        for k in high_arities:
            for _ in range(counts.get(k, 0)):
                if len(pool) < k:
                    break
                op_name = random.choice(operators_by_arity[k])
                parent = Node(op_name, arity=k)
                parent.children = [pool.pop(random.randint(0, len(pool) - 1)) for _ in range(k)]
                pool.append(parent)

        # Collapse any remaining unconnected subtrees
        while len(pool) > 1 and 2 in operators_by_arity:
            op_name = random.choice(operators_by_arity[2])
            parent = Node(op_name, arity=2)
            parent.children = [pool.pop(0), pool.pop(0)]
            pool.append(parent)

        root = pool[0] if pool else Node("<scalar>", 0)

        def collect_nodes(node):
            nodes = [node]
            for c in node.children:
                nodes.extend(collect_nodes(c))
            return nodes

        # 5. Insert Unary Operators (k = 1) at random tree depths
        if 1 in operators_by_arity:
            for _ in range(num_unary):
                all_nodes = collect_nodes(root)
                target = random.choice(all_nodes)
                
                # Insert unary node above a randomly chosen branch/leaf
                if target.children:
                    idx = random.randint(0, len(target.children) - 1)
                    child = target.children[idx]
                    unary_node = Node(random.choice(operators_by_arity[1]), arity=1)
                    unary_node.children = [child]
                    target.children[idx] = unary_node

        # 6. Fill LEAF placeholders with physical variables or constants
        leaves = operators_by_arity.get(0, ["<scalar>"])
        for node in collect_nodes(root):
            if node.name == "LEAF":
                node.name = random.choice(leaves)
                
                # Resolve continuous constant scalar values
                if node.name == "<scalar>":
                    node.name = f"{random.uniform(-3.0, 3.0):.4f}"

        return " ".join(root.to_rpn())

    def generate_rpns(self, batch):
        return [self.generate_random_rpn() for i in range(batch)]

    
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