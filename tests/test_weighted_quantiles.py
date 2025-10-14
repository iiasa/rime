#!/usr/bin/env python
"""Test script for weighted_quantiles function.

This script validates the fix for the weighted quantiles implementation,
ensuring consistency with numpy's quantile function for equal weights
and proper behavior for weighted cases.
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import the stats module directly
sys.path.insert(0, str(Path(__file__).parent.parent / "rimeX"))

# Import the function directly to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location("stats", Path(__file__).parent.parent / "rimeX" / "stats.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
weighted_quantiles = stats.weighted_quantiles


def test_equal_weights_consistency():
    """Test that weighted_quantiles matches np.quantile when weights are equal."""
    print("=" * 70)
    print("Test 1: Equal weights consistency with np.quantile")
    print("=" * 70)
    
    # Test case from the original note
    values = np.array([0, 1, 2])
    weights = np.array([1, 1, 1])
    q = 0.5
    
    result = weighted_quantiles(values, weights, q)
    expected = np.quantile(values, q)
    
    print(f"Values: {values}")
    print(f"Weights: {weights} (equal)")
    print(f"Quantile: {q}")
    print(f"weighted_quantiles result: {result:.4f}")
    print(f"np.quantile result: {expected:.4f}")
    print(f"Match: {np.isclose(result, expected)}")
    
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ Test passed\n")


def test_weighted_case_original():
    """Test the originally problematic case from the note."""
    print("=" * 70)
    print("Test 2: Original problematic weighted case")
    print("=" * 70)
    
    values = np.array([0, 1, 2])
    weights = np.array([0.5, 0.25, 0.25])
    q = 0.5
    
    result = weighted_quantiles(values, weights, q)
    
    # With the fix, the median should be around 0.5
    # because 50% of the cumulative weight is at value 0.5
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Cumulative weights: {np.cumsum(weights)}")
    print(f"Quantile: {q}")
    print(f"Result: {result:.4f}")
    print(f"Expected: ~0.5 (50% of cumulative weight)")

    # The 50th percentile should be 0.75, since 0*0.5 + 1*0.25 + 2*0.25 = 0.75
    assert result == 0.75, f"Median should be 0.75, got {result}"
    print("✓ Test passed\n")


def test_uniform_weights():
    """Test case with uniform weights from the note."""
    print("=" * 70)
    print("Test 3: Uniform weights comparison")
    print("=" * 70)
    
    values = np.array([0, 0, 1, 2])
    weights = np.ones(4)
    q = 0.5
    
    result = weighted_quantiles(values, weights, q)
    expected = np.quantile(values, q)
    
    print(f"Values: {values}")
    print(f"Weights: {weights} (all ones)")
    print(f"Quantile: {q}")
    print(f"weighted_quantiles result: {result:.4f}")
    print(f"np.quantile result: {expected:.4f}")
    print(f"Match: {np.isclose(result, expected)}")
    
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ Test passed\n")


def test_mean_preservation():
    """Test that the mean is preserved from the original requirement."""
    print("=" * 70)
    print("Test 4: Mean preservation (original requirement)")
    print("=" * 70)
    
    values = np.array([1, 2, 3, 4])
    weights = np.ones(4)
    
    result = weighted_quantiles(values, weights, 0.5, interpolate=True)
    expected = 2.5
    
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Quantile: 0.5")
    print(f"Result: {result:.4f}")
    print(f"Expected: {expected}")
    print(f"Match: {np.isclose(result, expected)}")
    
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ Test passed\n")


def test_multiple_quantiles():
    """Test with multiple quantiles at once."""
    print("=" * 70)
    print("Test 5: Multiple quantiles")
    print("=" * 70)
    
    values = np.array([1, 2, 3, 4, 5])
    weights = np.ones(5)
    quantiles = np.array([0.25, 0.5, 0.75])
    
    result = weighted_quantiles(values, weights, quantiles)
    expected = np.quantile(values, quantiles)
    
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Quantiles: {quantiles}")
    print(f"weighted_quantiles result: {result}")
    print(f"np.quantile result: {expected}")
    print(f"All match: {np.allclose(result, expected)}")
    
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ Test passed\n")


def test_edge_cases():
    """Test edge cases like 0th and 100th percentiles."""
    print("=" * 70)
    print("Test 6: Edge cases (0th and 100th percentiles)")
    print("=" * 70)
    
    values = np.array([1, 2, 3, 4, 5])
    weights = np.array([1, 1, 1, 1, 1])
    
    result_0 = weighted_quantiles(values, weights, 0.0)
    result_100 = weighted_quantiles(values, weights, 1.0)
    
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"0th percentile: {result_0:.4f} (expected: 1.0)")
    print(f"100th percentile: {result_100:.4f} (expected: 5.0)")
    
    assert np.isclose(result_0, 1.0), f"0th percentile should be 1.0, got {result_0}"
    assert np.isclose(result_100, 5.0), f"100th percentile should be 5.0, got {result_100}"
    print("✓ Test passed\n")


def test_non_interpolate():
    """Test non-interpolating mode."""
    print("=" * 70)
    print("Test 7: Non-interpolating mode")
    print("=" * 70)
    
    values = np.array([1, 2, 3, 4])
    weights = np.ones(4)
    q = 0.5
    
    result = weighted_quantiles(values, weights, q, interpolate=False)
    
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Quantile: {q}")
    print(f"Result (no interpolation): {result}")
    print(f"Result should be one of the actual values: {result in values}")
    
    assert result in values, f"Result should be one of the actual values, got {result}"
    print("✓ Test passed\n")


def test_skipna():
    """Test skipna functionality."""
    print("=" * 70)
    print("Test 8: Skip NaN values")
    print("=" * 70)
    
    values = np.array([1, 2, np.nan, 4, 5])
    weights = np.array([1, 1, 1, 1, 1])
    q = 0.5
    
    result = weighted_quantiles(values, weights, q, skipna=True)
    expected = np.nanquantile([1, 2, 4, 5], q)
    
    print(f"Values: {values}")
    print(f"Weights: {weights}")
    print(f"Quantile: {q}")
    print(f"Result (skipna=True): {result:.4f}")
    print(f"Expected (nanquantile): {expected:.4f}")
    print(f"Match: {np.isclose(result, expected)}")
    
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"
    print("✓ Test passed\n")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 70)
    print("WEIGHTED QUANTILES TEST SUITE")
    print("=" * 70 + "\n")
    
    tests = [
        test_equal_weights_consistency,
        test_weighted_case_original,
        test_uniform_weights,
        test_mean_preservation,
        test_multiple_quantiles,
        test_edge_cases,
        test_non_interpolate,
        test_skipna,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ Test failed: {e}\n")
            failed.append((test.__name__, str(e)))
    
    print("=" * 70)
    if not failed:
        print("ALL TESTS PASSED ✓")
        print("=" * 70 + "\n")
        return 0
    else:
        print(f"FAILED: {len(failed)} test(s)")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
