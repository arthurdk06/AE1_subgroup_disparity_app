import numpy as np
import pandas as pd

from scr.stats_utils import (
    compute_standard_deviation,
    compute_correlation,
    compute_confidence_interval,
    calculate_effect_size,
)


def test_compute_standard_deviation():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert np.isclose(compute_standard_deviation(df, "x"), np.std([1, 2, 3], ddof=0))


def test_compute_correlation():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    assert np.isclose(compute_correlation(df, "x", "y"), 1.0)


def test_compute_confidence_interval():
    series = pd.Series([10, 10, 10, 10])
    low, high = compute_confidence_interval(series)
    assert np.isclose(low, 10.0)
    assert np.isclose(high, 10.0)


def test_calculate_effect_size():
    group1 = [1, 2, 3]
    group2 = [4, 5, 6]
    effect = calculate_effect_size(group1, group2)
    assert effect < 0
