"""
Borough- and school-level aggregation utilities for gap analysis.

Reusable analytics functions for computing means, ranks, and comparisons
across London boroughs. Used by tests; can be integrated into the app for
additional metrics (e.g. identify_extreme_schools, confidence intervals).
"""
import numpy as np
import pandas as pd


def calculate_borough_mean(df, value_col):
    values = pd.to_numeric(df[value_col], errors="coerce")
    return values.mean()


def calculate_borough_gap(df, borough_name, gap_col):
    row = df.loc[df["region_name"] == borough_name, gap_col]
    if row.empty:
        return np.nan
    return pd.to_numeric(row.iloc[0], errors="coerce")


def compare_borough_to_london_average(df, borough_name, value_col):
    borough_value = df.loc[df["region_name"] == borough_name, value_col]
    if borough_value.empty:
        return np.nan
    return pd.to_numeric(borough_value.iloc[0], errors="coerce") - calculate_borough_mean(
        df, value_col
    )


def rank_boroughs_by_gap(df, gap_col):
    ranked = df.copy()
    ranked["gap_rank"] = ranked[gap_col].rank(ascending=False, method="min")
    return ranked.sort_values("gap_rank")


def get_tower_hamlets_data(df):
    return df[df["region_name"] == "Tower Hamlets"].copy()


def compare_th_gap_to_london(df, gap_col):
    th_gap = calculate_borough_gap(df, "Tower Hamlets", gap_col)
    return th_gap - calculate_borough_mean(df, gap_col)


def calculate_school_gap_distribution(df, gap_col):
    gap = pd.to_numeric(df[gap_col], errors="coerce")
    return gap.describe()


def identify_extreme_schools(df, gap_col, threshold):
    gap = pd.to_numeric(df[gap_col], errors="coerce")
    return df.loc[gap.abs() >= threshold].copy()


def summarise_school_variation(df, outcome_col):
    values = pd.to_numeric(df[outcome_col], errors="coerce")
    return {
        "mean": values.mean(),
        "std": values.std(ddof=0),
        "min": values.min(),
        "max": values.max(),
        "iqr": values.quantile(0.75) - values.quantile(0.25),
    }


