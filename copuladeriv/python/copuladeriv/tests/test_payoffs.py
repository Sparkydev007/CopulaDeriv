"""Payoff tests for Day 1 CopulaDeriv project."""

import numpy as np
from copuladeriv import payoffs


def test_best_of_call_both_below_strike():
    """Both assets below strike: payoff should be zero."""
    S1_T = np.array([90.0, 95.0])
    S2_T = np.array([85.0, 97.0])
    K = 100.0
    result = payoffs.best_of_call(S1_T, S2_T, K)
    assert np.all(result == 0.0), f"Expected all zeros, got {result}"


def test_best_of_call_both_above_strike():
    """Both assets above strike: payoff should be positive."""
    S1_T = np.array([110.0, 115.0])
    S2_T = np.array([112.0, 118.0])
    K = 100.0
    result = payoffs.best_of_call(S1_T, S2_T, K)
    # max(max(110,112)-100,0) = 12, max(max(115,118)-100,0) = 18
    expected = np.array([12.0, 18.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_best_of_call_one_above_one_below():
    """One asset above, one below strike: payoff depends on max."""
    S1_T = np.array([110.0, 90.0])
    S2_T = np.array([95.0, 115.0])
    K = 100.0
    result = payoffs.best_of_call(S1_T, S2_T, K)
    # Path 0: max(110, 95) - 100 = 10
    # Path 1: max(90, 115) - 100 = 15
    expected = np.array([10.0, 15.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_best_of_call_terminal_equal_to_strike():
    """Terminal price exactly at strike."""
    S1_T = np.array([100.0, 100.0])
    S2_T = np.array([100.0, 100.0])
    K = 100.0
    result = payoffs.best_of_call(S1_T, S2_T, K)
    # max(max(100, 100) - 100, 0) = 0
    expected = np.array([0.0, 0.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_best_of_call_identical_terminals():
    """Identical terminal prices for both assets."""
    S1_T = np.array([105.0, 105.0])
    S2_T = np.array([105.0, 105.0])
    K = 100.0
    result = payoffs.best_of_call(S1_T, S2_T, K)
    # max(max(105, 105) - 100, 0) = 5
    expected = np.array([5.0, 5.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_worst_of_put_both_above_strike():
    """Both assets above strike: put should be zero."""
    S1_T = np.array([110.0, 115.0])
    S2_T = np.array([112.0, 118.0])
    K = 100.0
    result = payoffs.worst_of_put(S1_T, S2_T, K)
    assert np.all(result == 0.0), f"Expected all zeros, got {result}"


def test_worst_of_put_both_below_strike():
    """Both assets below strike: put should be positive."""
    S1_T = np.array([80.0, 85.0])
    S2_T = np.array([82.0, 88.0])
    K = 100.0
    result = payoffs.worst_of_put(S1_T, S2_T, K)
    expected = np.array([18.0, 15.0])
    # Path 0: min(80, 82) = 80, K - 80 = 20. Wait, that's 20 not 18.
    # Actually: min(80, 82) = 80, 100 - 80 = 20. Hmm let me recheck.
    # Path 1: min(85, 88) = 85, 100 - 85 = 15
    expected = np.array([20.0, 15.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_worst_of_put_one_above_one_below():
    """One above, one below strike."""
    S1_T = np.array([110.0, 90.0])
    S2_T = np.array([95.0, 85.0])
    K = 100.0
    result = payoffs.worst_of_put(S1_T, S2_T, K)
    # Path 0: min(110, 95) = 95, 100 - 95 = 5
    # Path 1: min(90, 85) = 85, 100 - 85 = 15
    expected = np.array([5.0, 15.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_worst_of_put_terminal_equal_to_strike():
    """Terminal price exactly at strike."""
    S1_T = np.array([100.0, 100.0])
    S2_T = np.array([100.0, 100.0])
    K = 100.0
    result = payoffs.worst_of_put(S1_T, S2_T, K)
    # min(min(100, 100), ...) wait: min(100, 100) = 100, 100 - 100 = 0
    expected = np.array([0.0, 0.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_worst_of_put_identical_terminals():
    """Identical terminal prices for both assets."""
    S1_T = np.array([95.0, 95.0])
    S2_T = np.array([95.0, 95.0])
    K = 100.0
    result = payoffs.worst_of_put(S1_T, S2_T, K)
    # min(95, 95) = 95, 100 - 95 = 5
    expected = np.array([5.0, 5.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_european_call_basic():
    """Basic European call payoff."""
    S_T = np.array([105.0, 95.0, 100.0])
    K = 100.0
    result = payoffs.european_call(S_T, K)
    expected = np.array([5.0, 0.0, 0.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_european_call_in_the_money():
    """European call deep in the money."""
    S_T = np.array([150.0, 200.0])
    K = 100.0
    result = payoffs.european_call(S_T, K)
    expected = np.array([50.0, 100.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_european_call_out_of_the_money():
    """European call deep out of the money."""
    S_T = np.array([50.0, 80.0])
    K = 100.0
    result = payoffs.european_call(S_T, K)
    assert np.all(result == 0.0), f"Expected all zeros, got {result}"


def test_payoff_scalar_input():
    """Payoff functions should accept scalar inputs."""
    result = payoffs.best_of_call(110.0, 112.0, 100.0)
    assert result == 12.0, f"Expected 12.0, got {result}"

    result = payoffs.worst_of_put(95.0, 98.0, 100.0)
    assert result == 5.0, f"Expected 5.0, got {result}"

    result = payoffs.european_call(105.0, 100.0)
    assert result == 5.0, f"Expected 5.0, got {result}"


def test_payoff_with_numpy_float():
    """Payoff functions should handle numpy scalars."""
    import numpy as np
    result = payoffs.best_of_call(np.float64(110.0), np.float64(112.0), 100.0)
    assert result == 12.0, f"Expected 12.0, got {result}"