import numpy as np
import pandas as pd

from scr.metrics import (
    compute_gender_gap,
    compute_gender_gap_sd_units,
    compute_gender_ratio,
    compute_sen_proportion,
    compute_sen_gap,
    compute_sen_gap_sd_units,
    compute_total_pupils,
    compute_school_z_scores,
    compute_gap_variability,
)


def test_compute_gender_gap():
    df = pd.DataFrame({"boys": [10, 12], "girls": [11, 14]})
    result = compute_gender_gap(df, "boys", "girls")
    assert result["gender_gap"].tolist() == [1, 2]


def test_compute_gender_gap_sd_units():
    df = pd.DataFrame({"gap": [1, 3, 5]})
    result = compute_gender_gap_sd_units(df, "gap", "z")
    assert np.isclose(result["z"].mean(), 0.0, atol=1e-8)


def test_compute_gender_ratio():
    df = pd.DataFrame({"boys": [2, 0], "girls": [4, 1]})
    result = compute_gender_ratio(df, "boys", "girls", "ratio")
    assert result["ratio"].iloc[0] == 2
    assert np.isnan(result["ratio"].iloc[1])


def test_compute_sen_proportion():
    df = pd.DataFrame({"sen": [5, 2], "total": [10, 4]})
    result = compute_sen_proportion(df, "sen", "total")
    assert result["sen_proportion"].tolist() == [0.5, 0.5]


def test_compute_sen_gap():
    df = pd.DataFrame({"non_sen": [20], "sen": [15]})
    result = compute_sen_gap(df, "non_sen", "sen")
    assert result["sen_gap"].iloc[0] == 5


def test_compute_sen_gap_sd_units():
    df = pd.DataFrame({"gap": [2, 4, 6]})
    result = compute_sen_gap_sd_units(df, "gap", "z")
    assert np.isclose(result["z"].mean(), 0.0, atol=1e-8)


def test_compute_total_pupils():
    df = pd.DataFrame({"boys": [10, 12], "girls": [11, 14]})
    result = compute_total_pupils(df, "boys", "girls")
    assert result["total_pupils"].tolist() == [21, 26]


def test_compute_school_z_scores():
    df = pd.DataFrame({"outcome": [40, 50, 60]})
    result = compute_school_z_scores(df, "outcome", "z")
    assert np.isclose(result["z"].mean(), 0.0, atol=1e-8)


def test_compute_gap_variability():
    df = pd.DataFrame({"gap": [1, 2, 3, 4]})
    stats = compute_gap_variability(df, "gap")
    assert stats["mean"] == 2.5
    assert stats["min"] == 1
    assert stats["max"] == 4




