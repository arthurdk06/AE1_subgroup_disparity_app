import numpy as np
import pandas as pd

from scr.data_processing import (
    compute_gender_metrics,
    compute_sen_metrics,
    compute_school_metrics,
    find_data_files,
)


def test_compute_gender_metrics():
    df = pd.DataFrame(
        {
            "region_name": ["A", "B"],
            "girls_attainment_8_score": [52, 56],
            "boys_attainment_8_score": [50, 50],
        }
    )
    result = compute_gender_metrics(df)
    assert "gender_gap" in result.columns
    assert result["gender_gap"].tolist() == [2, 6]
    assert "gender_gap_zscore_london" in result.columns


def test_compute_sen_metrics():
    df = pd.DataFrame(
        {
            "total_num_pupils_ks4_total": [100],
            "total_num_pupils_ks4_no_sen": [80],
            "total_num_pupils_ks4_sen_state_ehc": [10],
            "total_num_pupils_ks4_sen_supp": [10],
            "total_attainment_8_no_sen": [50],
            "total_attainment_8_sen_state_ehc": [35],
            "total_attainment_8_sen_supp": [40],
        }
    )
    result = compute_sen_metrics(df)
    assert np.isclose(result["sen_proportion"].iloc[0], 0.2)
    assert np.isclose(result["sen_gap"].iloc[0], 50 - 37.5)
    assert "sen_gap_zscore_london" in result.columns


def test_compute_school_metrics():
    df = pd.DataFrame(
        {
            "attainment_8_score_girls": [55, 50],
            "attainment_8_score_boys": [50, 45],
            "sen_total_pupils": [20, 10],
            "total_pupils": [100, 50],
            "attainment_8_score_all": [52, 47],
        }
    )
    result = compute_school_metrics(df)
    assert "gender_gap" in result.columns
    assert "sen_proportion" in result.columns
    assert np.isclose(result["sen_proportion"].iloc[0], 0.2)


def test_find_data_files(tmp_path):
    (tmp_path / "GCSE results by sex - 2022-23.csv").write_text("a,b\n1,2\n")
    (tmp_path / "GCSE results by SEN - 2022-23.csv").write_text("a,b\n1,2\n")
    (tmp_path / "TH results 2022-23.csv").write_text("a,b\n1,2\n")

    found = find_data_files(str(tmp_path))
    assert len(found["gender"]) == 1
    assert len(found["sen"]) == 1
    assert len(found["th"]) == 1