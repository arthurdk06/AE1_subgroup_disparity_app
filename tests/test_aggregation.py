import pandas as pd

from scr.aggregation import (
    calculate_borough_mean,
    calculate_borough_gap,
    compare_borough_to_london_average,
    rank_boroughs_by_gap,
    get_tower_hamlets_data,
    compare_th_gap_to_london,
    calculate_school_gap_distribution,
    identify_extreme_schools,
    summarise_school_variation,
)


def _borough_df():
    return pd.DataFrame(
        {
            "region_name": ["Tower Hamlets", "Camden", "Hackney"],
            "gap": [4, 2, 6],
            "value": [50, 48, 52],
        }
    )


def test_calculate_borough_mean():
    df = _borough_df()
    assert calculate_borough_mean(df, "value") == 50


def test_calculate_borough_gap():
    df = _borough_df()
    assert calculate_borough_gap(df, "Camden", "gap") == 2


def test_compare_borough_to_london_average():
    df = _borough_df()
    diff = compare_borough_to_london_average(df, "Camden", "value")
    assert diff == 48 - 50


def test_rank_boroughs_by_gap():
    df = _borough_df()
    ranked = rank_boroughs_by_gap(df, "gap")
    assert ranked.iloc[0]["region_name"] == "Hackney"


def test_get_tower_hamlets_data():
    df = _borough_df()
    th = get_tower_hamlets_data(df)
    assert th.iloc[0]["region_name"] == "Tower Hamlets"


def test_compare_th_gap_to_london():
    df = _borough_df()
    diff = compare_th_gap_to_london(df, "gap")
    assert diff == 4 - df["gap"].mean()


def test_calculate_school_gap_distribution():
    df = pd.DataFrame({"gap": [1, 2, 3, 4]})
    dist = calculate_school_gap_distribution(df, "gap")
    assert dist["mean"] == 2.5


def test_identify_extreme_schools():
    df = pd.DataFrame({"school": ["A", "B"], "gap": [1, 5]})
    extreme = identify_extreme_schools(df, "gap", threshold=4)
    assert extreme["school"].tolist() == ["B"]


def test_summarise_school_variation():
    df = pd.DataFrame({"outcome": [10, 20, 30, 40]})
    summary = summarise_school_variation(df, "outcome")
    assert summary["mean"] == 25
