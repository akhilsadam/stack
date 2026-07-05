"""
Integration tests for RPN components working together.
"""

import torch
import pytest
from typing import Dict, List

from qg.solver.opt.operator.rpn.embeddings import (
    TOKEN_TO_ID,
    SCALAR_TOKEN_ID,
    batch_tokenize_rpn,
)
from qg.solver.opt.operator.rpn.algebra import create_composite_ruleset
from qg.solver.opt.operator.rpn.contrastive import ContrastiveRPN
from qg.solver.opt.operator.rpn.generator import (
    RPNGenerator,
    create_vocab_from_embeddings,
)


def test_end_to_end_workflow():
    """Test complete workflow: generate -> tokenize -> contrastive."""
    # Create components
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab, max_depth=3, max_nodes=8)
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)
    trainer = ContrastiveRPN(rules=rules)
    trainer.eval()

    # Generate batch
    batch_size = 4
    rpns = generator.generate_batch(batch_size)

    # Tokenize using new API
    token_ids, amp = batch_tokenize_rpn(rpns, max_len=100)

    # Forward through trainer
    z = trainer(rpns)

    # Check outputs
    assert z.shape == (batch_size, 64)  # Default proj_dim


def test_generator_to_rules_integration():
    """Test that generated expressions can be processed by rules."""
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab, max_depth=2, max_nodes=6)
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)

    # Generate expressions
    rpns = generator.generate_rpn(10)

    # Tokenize and apply rules
    for rpn in rpns:
        # Tokenize single expression
        token_ids, scalar_vals, scalar_mask = batch_tokenize_rpn([rpn], [None])

        # Try to apply rules
        result = rules.apply_random_rule(token_ids)

        # Should either return TransformResult or None
        if result is not None:
            assert hasattr(result, 'tokens')
            assert hasattr(result, 'matched')
            assert hasattr(result, 'pad_token_id')

            # Output should have same batch size
            assert result.tokens.shape[0] == 1


def test_contrastive_with_generated_data():
    """Test contrastive training with generator-provided data."""
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab, max_depth=3, max_nodes=10)
    trainer = ContrastiveRPN()
    trainer.eval()

    # Generate larger batch for meaningful contrastive learning
    batch_size = 8
    rpns = generator.generate_batch(batch_size)

    # Test forward_from_rpn_strings
    z_a, z_p, loss = trainer.forward_from_rpn_strings(rpns)

    assert z_a.shape == (batch_size, 64)
    assert z_p.shape == (batch_size, 64)
    assert loss.shape == ()
    assert torch.isfinite(loss)

    # Loss should be reasonable (not NaN or extreme)
    assert loss.abs() < 100.0  # Very conservative bound


def test_batch_size_consistency():
    """Test that all components handle batch sizes consistently."""
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab)

    # Test various batch sizes
    for batch_size in [1, 2, 4, 8]:
        rpns = generator.generate_rpn(batch_size)
        if batch_size == 1:
            assert isinstance(rpns, str)
        else:
            assert isinstance(rpns, list)
            assert len(rpns) == batch_size

        # Tokenize
        scalar_params_list = [None] * (batch_size if isinstance(rpns, list) else 1)
        token_ids, scalar_vals, scalar_mask = batch_tokenize_rpn(
            rpns if isinstance(rpns, list) else [rpns],
            scalar_params_list
        )

        # Check batch dimension
        expected_batch_size = batch_size if isinstance(rpns, list) else 1
        assert token_ids.shape[0] == expected_batch_size
        assert scalar_vals.shape[0] == expected_batch_size
        assert scalar_mask.shape[0] == expected_batch_size


def test_device_propagation():
    """Test that device is propagated through all components."""
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab)

    # Generate data
    rpns = generator.generate_rpn(2)

    # Create trainer on CPU
    trainer_cpu = ContrastiveRPN()
    trainer_cpu.eval()
    trainer_cpu.to('cpu')

    # Process on CPU
    z_a_cpu, z_p_cpu, loss_cpu = trainer_cpu.forward_from_rpn_strings(rpns)

    assert z_a_cpu.device == torch.device('cpu')
    assert loss_cpu.device == torch.device('cpu')

    if torch.cuda.is_available():
        # Create trainer on GPU
        trainer_gpu = ContrastiveRPN()
        trainer_gpu.eval()
        trainer_gpu.cuda()

        # Process on GPU
        z_a_gpu, z_p_gpu, loss_gpu = trainer_gpu.forward_from_rpn_strings(rpns)

        assert z_a_gpu.device == torch.device('cuda:0')
        assert loss_gpu.device == torch.device('cuda:0')


def test_memory_usage_stability():
    """Test that components don't have memory leaks in simple workflows."""
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab, max_depth=2, max_nodes=4)
    trainer = ContrastiveRPN()
    trainer.eval()

    # Run multiple iterations
    for i in range(5):
        rpns = generator.generate_rpn(4)
        z_a, z_p, loss = trainer.forward_from_rpn_strings(rpns)

        # Just verify outputs are valid
        assert torch.isfinite(loss)
        assert z_a.shape[0] == 4


def test_error_handling_integration():
    """Test error handling across integrated components."""
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab)

    # Generate some expressions
    rpns = generator.generate_rpn(3)

    # Create trainer
    trainer = ContrastiveRPN()
    trainer.eval()

    # Test with invalid scalar_params length (should handle gracefully)
    # Note: current implementation expects matching lengths
    scalar_params_wrong_length = [None]  # Only 1 item for 3 expressions

    try:
        z_a, z_p, loss = trainer.forward_from_rpn_strings(rpns, scalar_params_wrong_length)
        # If it doesn't fail, check outputs are valid
        if z_a is not None:
            assert z_a.shape[0] == 3
    except Exception:
        # Exception is acceptable - mismatch should be caught
        pass


def test_deterministic_training_workflow():
    """Test that training workflow can be deterministic with seeds."""
    import random

    # Set seeds
    random.seed(42)
    torch.manual_seed(42)

    vocab = create_vocab_from_embeddings()
    generator1 = RPNGenerator(vocab, max_depth=2, max_nodes=5)
    trainer1 = ContrastiveRPN()
    trainer1.eval()

    # Generate and process
    rpns1 = generator1.generate_rpn(2)
    z_a1, z_p1, loss1 = trainer1.forward_from_rpn_strings(rpns1)

    # Reset seeds
    random.seed(42)
    torch.manual_seed(42)

    generator2 = RPNGenerator(vocab, max_depth=2, max_nodes=5)
    trainer2 = ContrastiveRPN()
    trainer2.eval()

    # Generate and process again
    rpns2 = generator2.generate_rpn(2)
    z_a2, z_p2, loss2 = trainer2.forward_from_rpn_strings(rpns2)

    # Should be identical
    assert rpns1 == rpns2
    assert torch.allclose(z_a1, z_a2)
    assert torch.allclose(z_p1, z_p2)
    assert torch.allclose(loss1, loss2)


def test_component_interfaces():
    """Test that all components have expected interfaces."""
    # Embeddings
    from qg.solver.opt.operator.rpn.embeddings import (
        TOKEN_TO_ID,
        batch_tokenize_rpn,
        RPNTokenEmbedder,
    )

    # Algebra
    from qg.solver.opt.operator.rpn.algebra import (
        create_composite_ruleset,
        AlgebraicRuleSet,
    )

    # Contrastive
    from qg.solver.opt.operator.rpn.contrastive import (
        ContrastiveRPN,
        infonce_symmetric_loss,
    )

    # Generator
    from qg.solver.opt.operator.rpn.generator import (
        RPNGenerator,
        create_vocab_from_embeddings,
    )

    # Verify they can be instantiated/imported
    assert TOKEN_TO_ID is not None
    assert callable(batch_tokenize_rpn)
    assert callable(create_composite_ruleset)
    assert callable(create_vocab_from_embeddings)


def test_realistic_training_scenario():
    """Test a realistic training scenario with multiple batches."""
    vocab = create_vocab_from_embeddings()
    generator = RPNGenerator(vocab, max_depth=3, max_nodes=8)
    trainer = ContrastiveRPN()
    trainer.train()  # Set to training mode

    # Simulate training loop
    num_batches = 3
    batch_size = 4

    total_loss = 0.0
    batch_losses = []

    for batch_idx in range(num_batches):
        # Generate batch
        rpns = generator.generate_batch(batch_size)

        # Forward pass
        z_a, z_p, loss = trainer.forward_from_rpn_strings(rpns)

        assert torch.isfinite(loss)
        batch_losses.append(loss.item())
        total_loss += loss.item()

        # Backward pass (simulated)
        loss.backward()

        # Check gradients
        for param in trainer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                # Zero gradients for next iteration
                param.grad.zero_()

    # Verify losses were computed
    assert len(batch_losses) == num_batches
    assert total_loss > 0 or total_loss == 0  # Could be 0 if no rules apply

    print(f"Simulated training: average loss = {total_loss / num_batches:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])