from typing import List

import pytest
import torch

from space.tokens import Token, TokenEmbedding, Vocab


def build_vocab() -> Vocab:
    token_specs = [
        ("q", 0),
        ("psi", 0),
        ("u", 0),
        ("v", 0),
        ("x", 0),
        ("y", 0),
        ("dx", 1),
        ("dy", 1),
        ("lap", 1),
        ("invlap", 1),
        ("sqrt", 1),
        ("cos", 1),
        ("sin", 1),
        ("exp", 1),
        ("square", 1),
        ("cube", 1),
        ("abs", 1),
        ("+", 2),
        ("-", 2),
        ("*", 2),
        ("jacobian", 2),
        ("neg", 1),
        ("dealias", 1),
    ]
    tokens = [Token(name, arity) for name, arity in token_specs]
    return Vocab(tokens=tokens)


def make_batch_strings() -> List[str]:
    return [
        "q",
        "psi",
        "u v +",
        "q 1.25 *",
        "q psi jacobian",
        "q lap",
        "q psi + sin",
        "x y -",
    ]

@pytest.mark.parametrize("seq_len", [12])
@pytest.mark.parametrize("embed_dim", [3, 8])
def test_token_embedding_round_trip_is_invertible(seq_len: int, embed_dim: int) -> None:
    torch.manual_seed(7)

    vocab = build_vocab()
    embedder = TokenEmbedding(vocab=vocab, seq_len=seq_len, embed_dim=embed_dim, phys_dim=1)

    batch_strings = make_batch_strings()
    embeddings = embedder(batch_strings)
    decoded = embedder.reverse(embeddings)

    roundtrip_embeddings = embedder(decoded)
    roundtrip_decoded = embedder.reverse(roundtrip_embeddings)

    assert decoded == batch_strings
    assert roundtrip_decoded == batch_strings
