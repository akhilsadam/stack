"""
Run all RPN tests as a script (bypassing pytest import issues).
"""

import sys
import traceback
from typing import List, Dict


def run_test_module(module_name: str, test_functions: List[str]) -> Dict[str, bool]:
    """Run test functions from a module."""
    results = {}
    module_path = f"qg.solver.opt.operator.rpn.tests.{module_name}"

    try:
        module = __import__(module_path, fromlist=[""])
        print(f"\n{'='*60}")
        print(f"Running tests from {module_name}")
        print(f"{'='*60}")
    except ImportError as e:
        print(f"ERROR: Could not import {module_name}: {e}")
        return {f"{module_name}.import": False}

    for test_name in test_functions:
        if hasattr(module, test_name):
            test_func = getattr(module, test_name)
            try:
                print(f"\n  Running {test_name}...")
                test_func()
                results[f"{module_name}.{test_name}"] = True
                print(f"  ✓ {test_name} passed")
            except Exception as e:
                results[f"{module_name}.{test_name}"] = False
                print(f"  ✗ {test_name} failed: {e}")
                traceback.print_exc()
        else:
            print(f"  WARNING: Test function {test_name} not found in {module_name}")
            results[f"{module_name}.{test_name}"] = False

    return results


def main():
    """Run all test modules."""
    # Define test modules and their test functions
    test_modules = {
        "test_embeddings": [
            "test_vocab_structure",
            # "test_normalize_token",
            "test_tokenize_single_rpn",
            "test_batch_tokenize_rpn",
            "test_embedder_initialization",
            "test_embedder_forward",
            "test_scalar_embedding",
            "test_token_embedding",
            "test_token_category_enum",
            "test_consistency_with_compiler",
        ],
        "test_compiler": [ ### TODO needs lot of work, most of these tests are wrong!
            "test_ir_utilities",
            "test_compiled_pde_dataclass",
            # "test_compiler_initialization",
            "test_expr_builder_smoke",
            # "test_compile_simple_expressions",
            # "test_compiler_state_handling",
            # "test_compiler_error_handling",
            # "test_compiler_with_parameters",
            "test_to_physical_to_spectral_mocks",
            # "test_compiler_registry_consistency",
        ],
        "test_algebra": [
            "test_rule_category_enum",
            "test_simple_algebraic_rule_structure",
            "test_algebraic_rule_set_basic",
            "test_transform_result",
            "test_composite_algebraic_rule_set",
            "test_create_composite_ruleset",
            "test_rule_application_mechanics",
            "test_random_positive_view",
            "test_padding_behavior",
            "test_batch_with_variable_lengths",
            "test_rule_matching_edge_cases",
            "test_rule_metadata",
            "test_scalar_token_handling",
            "test_device_handling",
            "test_deterministic_with_seed",
        ],
        "test_contrastive": [
            "test_masked_mean_pool",
            "test_infonce_logits",
            "test_infonce_symmetric_loss",
            "test_rpn_encoder_head",
            "test_padding_utilities",
            "test_contrastive_trainer_initialization",
            "test_encode_token_batch",
            "test_forward_from_token_tensors",
            "test_forward_from_rpn_strings",
            "test_device_consistency",
            "test_trainer_state_management",
            "test_gradient_flow",
            "test_batch_size_handling",
        ],
        "test_generator": [
            "test_vocab_creation",
            "test_generator_initialization",
            "test_node_type_enum",
            "test_arity_enum",
            "test_operator_info_building",
            "test_random_operator_selection",
            "test_expression_tree_generation",
            "test_tree_to_rpn_conversion",
            "test_generate_rpn_single",
            "test_generate_rpn_multiple",
            "test_generate_batch",
            "test_expression_complexity_limits",
            "test_random_variable_selection",
            "test_random_constant_generation",
            "test_jacobian_special_handling",
            "test_deterministic_with_seed",
            "test_vocab_completeness",
        ],
        "test_integration": [
            "test_end_to_end_workflow",
            "test_generator_to_rules_integration",
            "test_contrastive_with_generated_data",
            "test_batch_size_consistency",
            "test_device_propagation",
            "test_memory_usage_stability",
            "test_error_handling_integration",
            "test_deterministic_training_workflow",
            "test_component_interfaces",
            "test_realistic_training_scenario",
        ],
    }

    print("RPN Comprehensive Test Suite")
    print("="*60)

    all_results = {}
    total_passed = 0
    total_failed = 0

    # Run all tests
    for module_name, test_functions in test_modules.items():
        module_results = run_test_module(module_name, test_functions)
        all_results.update(module_results)

        # Count results for this module
        module_passed = sum(1 for v in module_results.values() if v)
        module_failed = sum(1 for v in module_results.values() if not v)
        total_passed += module_passed
        total_failed += module_failed

        print(f"\n  {module_name}: {module_passed} passed, {module_failed} failed")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")

    if total_failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed:")
        for test_name, passed in all_results.items():
            if not passed:
                print(f"  - {test_name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())