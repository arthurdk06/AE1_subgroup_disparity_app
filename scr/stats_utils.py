"""Statistical utilities for gap analysis. Uses stdlib statistics for NormalDist."""
from statistics import NormalDist

import numpy as np
import pandas as pd


def compute_standard_deviation(df, col):
    values = pd.to_numeric(df[col], errors="coerce")
    return values.std(ddof=0)


def compute_correlation(df, col1, col2):
    series = df[[col1, col2]].apply(pd.to_numeric, errors="coerce")
    return series[col1].corr(series[col2])


def compute_confidence_interval(series, confidence=0.95):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return (np.nan, np.nan)
    mean = values.mean()
    std = values.std(ddof=0)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    margin = z * (std / np.sqrt(len(values)))
    return (mean - margin, mean + margin)


def calculate_effect_size(group1, group2):
    g1 = pd.to_numeric(pd.Series(group1), errors="coerce").dropna()
    g2 = pd.to_numeric(pd.Series(group2), errors="coerce").dropna()
    if g1.empty or g2.empty:
        return np.nan
    pooled_std = np.sqrt(((g1.var(ddof=0) + g2.var(ddof=0)) / 2))
    if pooled_std == 0:
        return np.nan
    return (g1.mean() - g2.mean()) / pooled_std
