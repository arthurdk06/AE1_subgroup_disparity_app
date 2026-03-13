"""
Metric computation utilities for gender and SEN gap analysis.

Reusable functions for computing gaps, z-scores, proportions, and ratios.
Used by tests; complements scr.data_processing for more flexible column naming.
"""
import pandas as pd
import numpy as np


def compute_gender_gap(df, boys_col, girls_col, new_col="gender_gap"):
    """
    Adds a new column with gender gap (girls - boys).
    """
    df[new_col] = pd.to_numeric(df[girls_col], errors="coerce") - pd.to_numeric(
        df[boys_col], errors="coerce"
    )
    return df


def compute_gender_gap_sd_units(df, gap_col, new_col="gender_gap_zscore"):
    gap = pd.to_numeric(df[gap_col], errors="coerce")
    std = gap.std(ddof=0)
    df[new_col] = (gap - gap.mean()) / std if std else np.nan
    return df


def compute_gender_ratio(df, boys_col, girls_col, new_col="gender_ratio"):
    boys = pd.to_numeric(df[boys_col], errors="coerce")
    girls = pd.to_numeric(df[girls_col], errors="coerce")
    df[new_col] = girls / boys.replace({0: np.nan})
    return df


def compute_sen_proportion(df, sen_col, total_col, new_col="sen_proportion"):
    sen = pd.to_numeric(df[sen_col], errors="coerce")
    total = pd.to_numeric(df[total_col], errors="coerce")
    df[new_col] = sen / total.replace({0: np.nan})
    return df


def compute_sen_gap(df, non_sen_col, sen_col, new_col="sen_gap"):
    non_sen = pd.to_numeric(df[non_sen_col], errors="coerce")
    sen = pd.to_numeric(df[sen_col], errors="coerce")
    df[new_col] = non_sen - sen
    return df


def compute_sen_gap_sd_units(df, gap_col, new_col="sen_gap_zscore"):
    gap = pd.to_numeric(df[gap_col], errors="coerce")
    std = gap.std(ddof=0)
    df[new_col] = (gap - gap.mean()) / std if std else np.nan
    return df


def compute_total_pupils(df, boys_col, girls_col, new_col="total_pupils"):
    boys = pd.to_numeric(df[boys_col], errors="coerce")
    girls = pd.to_numeric(df[girls_col], errors="coerce")
    df[new_col] = boys + girls
    return df


def compute_school_z_scores(df, outcome_col, new_col="outcome_zscore"):
    outcome = pd.to_numeric(df[outcome_col], errors="coerce")
    std = outcome.std(ddof=0)
    df[new_col] = (outcome - outcome.mean()) / std if std else np.nan
    return df


def compute_gap_variability(df, gap_col):
    gap = pd.to_numeric(df[gap_col], errors="coerce")
    return {
        "mean": gap.mean(),
        "std": gap.std(ddof=0),
        "min": gap.min(),
        "max": gap.max(),
    }




