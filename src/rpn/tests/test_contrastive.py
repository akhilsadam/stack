"""
Exhaustive tests for contrastive learning module.
"""

import torch
import pytest
from typing import Dict, List, Optional

from qg.solver.opt.operator.rpn.contrastive import (
    ContrastiveRPN,
    masked_mean_pool,
    infonce_logits,
    infonce_single_loss,
    infonce_symmetric_loss,
    SelfAttention,
    RPN_AE,
)
from qg.solver.opt.operator.rpn.embeddings import (
    TOKEN_TO_ID,
    SCALAR_TOKEN_ID,
    batch_tokenize_rpn,
)


def test_masked_mean_pool():
    """Test masked mean pooling function."""
    batch_size = 2
    seq_len = 5
    embed_dim = 8

    # Create test data
    seq = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.tensor([
        [True, True, True, False, False],  # First 3 tokens real
        [True, False, False, False, False],  # Only first token real
    ], dtype=torch.bool)

    pooled = masked_mean_pool(seq, mask)

    assert pooled.shape == (batch_size, embed_dim)
    assert torch.isfinite(pooled).all()

    # Check pooling logic
    # For batch 0: average of first 3 vectors
    expected0 = seq[0, :3].mean(dim=0)
    assert torch.allclose(pooled[0], expected0, rtol=1e-5)

    # For batch 1: average of first vector only (so equals first vector)
    expected1 = seq[0, 0]
    # Actually seq[1, 0] for batch 1
    expected1 = seq[1, 0]
    assert torch.allclose(pooled[1], expected1, rtol=1e-5)

    # Test with all False mask (avoid division by zero)
    mask_all_false = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    pooled_all_false = masked_mean_pool(seq, mask_all_false)
    assert pooled_all_false.shape == (batch_size, embed_dim)
    # Should return zeros or very small values (clamped)

    # Test with eps parameter
    pooled_eps = masked_mean_pool(seq, mask, eps=1e-10)
    assert pooled_eps.shape == (batch_size, embed_dim)


def test_infonce_logits():
    """Test InfoNCE logits computation."""
    batch_size = 3
    embed_dim = 4

    z_a = torch.randn(batch_size, embed_dim)
    z_b = torch.randn(batch_size, embed_dim)
    temperature = 0.1

    logits = infonce_logits(z_a, z_b, temperature)

    assert logits.shape == (batch_size, batch_size)
    assert torch.isfinite(logits).all()

    # Check symmetry property (if we swap, should get transpose)
    logits_ba = infonce_logits(z_b, z_a, temperature)
    assert torch.allclose(logits.t(), logits_ba, rtol=1e-5)

    # Check diagonal should be relatively high (cosine similarity of normalized vectors)
    z_a_norm = torch.nn.functional.normalize(z_a, dim=-1)
    z_b_norm = torch.nn.functional.normalize(z_b, dim=-1)
    diag_expected = (z_a_norm * z_b_norm).sum(dim=-1) / temperature
    logits_diag = logits.diag()

    # Should be close (might differ due to numerical precision)
    assert torch.allclose(logits_diag, diag_expected, rtol=1e-5)


def test_infonce_symmetric_loss():
    """Test symmetric InfoNCE loss."""
    batch_size = 4
    embed_dim = 8

    z_anchor = torch.randn(batch_size, embed_dim)
    z_positive = torch.randn(batch_size, embed_dim)

    loss = infonce_symmetric_loss(z_anchor, z_positive, temperature=0.1)

    assert loss.shape == ()  # Scalar
    assert torch.isfinite(loss)
    assert loss > 0  # Should be positive

    # Loss should be symmetric
    loss_reverse = infonce_symmetric_loss(z_positive, z_anchor, temperature=0.1)
    assert torch.allclose(loss, loss_reverse, rtol=1e-5)

    # Test with different temperature
    loss_temp2 = infonce_symmetric_loss(z_anchor, z_positive, temperature=0.5)
    assert torch.isfinite(loss_temp2)
    assert not torch.allclose(loss, loss_temp2)  # Different temperature -> different loss


def test_rpn_encoder_head():
    """Test RPNEncoderHead module."""
    batch_size = 2
    embed_dim = 16
    proj_dim = 32

    head = RPNEncoderHead(embed_dim, proj_dim)

    # Test forward
    pooled = torch.randn(batch_size, embed_dim)
    projected = head(pooled)

    assert projected.shape == (batch_size, proj_dim)
    assert torch.isfinite(projected).all()

    # Test that proj attribute exists
    assert hasattr(head, 'proj')
    assert isinstance(head.proj, torch.nn.Sequential)
    assert len(head.proj) == 3  # Linear -> SiLU -> Linear


# Note: padding utilities _pad_2d_right, _pad_2d_bool, _pad_1d_right were removed from contrastive.py
# They may be available elsewhere or no longer needed




def test_contrastive_trainer_initialization():
    """Test ContrastiveRPN initialization."""
    # Test with default parameters
    trainer1 = ContrastiveRPN()
    assert trainer1.temperature == 0.1
    assert hasattr(trainer1, 'embedder')
    assert hasattr(trainer1, 'head')
    assert hasattr(trainer1, 'rules')

    # Test with custom parameters
    trainer2 = ContrastiveRPN(
        seq_len=100,
        embed_dim=64,
        proj_dim=128,
        temperature=0.2,
    )
    assert trainer2.temperature == 0.2
    assert trainer2.embedder.embed_dim == 64
    assert trainer2.head.embed_dim == 64
    assert trainer2.head.proj_dim == 128

    # Test embedder and head have correct dimensions
    assert trainer2.embedder.embed_dim == 64
    # Head should accept 64-dim input, output 128-dim
    assert trainer2.head.proj[-1].in_features == 64  # Last linear layer in proj
    assert trainer2.head.proj[-1].out_features == 128  # Should be proj_dim


def test_encode_token_batch():
    """Test encode_token_batch method."""
    trainer = ContrastiveRPN()
    trainer.eval()  # Disable any dropout

    batch_size = 2
    seq_len = 6

    # Create mock inputs - note: new signature is (token_ids, amp)
    token_ids = torch.randint(0, len(TOKEN_TO_ID), (batch_size, seq_len))
    amp = torch.randn(batch_size, seq_len)  # amplitude values

    encoded = trainer.encode_token_batch(token_ids, amp)

    assert encoded.shape == (batch_size, 64)  # Default proj_dim=64
    assert torch.isfinite(encoded).all()


# Note: forward_from_token_tensors method was removed from ContrastiveRPN
# Use forward method instead which takes RPN strings


def test_forward_from_rpn_strings():
    """Test forward method (replaces forward_from_rpn_strings)."""
    trainer = ContrastiveRPN()
    trainer.eval()

    # Test with simple expressions
    rpns = ["q psi +", "psi q +", "q dx"]
    # Note: forward method returns embedding, not (z_a, z_p, loss)
    embeddings = trainer.forward(rpns)

    assert embeddings.shape == (3, 64)  # Batch size 3, default proj_dim=64
    assert torch.isfinite(embeddings).all()


def test_device_consistency():
    """Test that trainer works on appropriate device."""
    trainer = ContrastiveRPN()
    trainer.eval()

    # Get the device of the trainer
    trainer_device = next(trainer.parameters()).device

    # Create inputs on same device
    rpns = ["q psi +", "psi q +"]
    embeddings = trainer.forward(rpns)

    # Outputs should be on same device as trainer
    assert embeddings.device == trainer_device

    # Test moving to CPU explicitly
    trainer_cpu = ContrastiveRPN()
    trainer_cpu.eval()
    trainer_cpu.to('cpu')

    embeddings_cpu = trainer_cpu.forward(rpns)
    assert embeddings_cpu.device == torch.device('cpu')


def test_trainer_state_management():
    """Test trainer state (train/eval) affects computation."""
    trainer = ContrastiveRPN()

    # Set to eval mode (disables dropout if any)
    trainer.eval()
    assert not trainer.training

    # Set to train mode
    trainer.train()
    assert trainer.training

    # Both modes should produce valid outputs
    rpns = ["q psi +"]
    embeddings_eval = trainer.forward(rpns)

    trainer.eval()
    embeddings_train = trainer.forward(rpns)

    # Outputs might differ due to dropout/batchnorm, but shapes should match
    assert embeddings_eval.shape == embeddings_train.shape


def test_gradient_flow():
    """Test that gradients flow through trainer using loss method."""
    trainer = ContrastiveRPN()
    trainer.train()

    rpns = ["q psi +", "psi q +"]
    loss = trainer.loss(rpns)

    # Backward pass
    loss.backward()

    # Check gradients exist for parameters
    for name, param in trainer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_batch_size_handling():
    """Test trainer handles different batch sizes."""
    trainer = ContrastiveRPN()
    trainer.eval()

    # Test single example
    rpns1 = ["q psi +"]
    embeddings1 = trainer.forward(rpns1)
    assert embeddings1.shape[0] == 1

    # Test many examples
    rpns10 = ["q psi +"] * 10
    embeddings10 = trainer.forward(rpns10)
    assert embeddings10.shape[0] == 10

    # Test empty batch (should fail gracefully or with error)
    # Implementation might raise error or handle empty batch


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])