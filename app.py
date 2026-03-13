import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scr.data_processing import (
    compute_gender_metrics,
    compute_sen_metrics,
    compute_school_metrics,
    find_data_files,
    load_gender_borough,
    load_sen_borough,
    load_th_schools,
)


DATA_DIR = Path(__file__).parent / "data"


def _year_from_name(path: Path) -> str:
    match = re.search(r"(\d{4}-\d{2})", path.stem)
    if match:
        return match.group(1)
    return path.stem


data_files = find_data_files(str(DATA_DIR))
_year_to_path = {
    key: {_year_from_name(path): path for path in paths}
    for key, paths in data_files.items()
}


@st.cache_data
def _load_gender(year: str) -> pd.DataFrame:
    file_path = _year_to_path["gender"][year]
    df = load_gender_borough(str(file_path))
    return compute_gender_metrics(df)


@st.cache_data
def _load_sen(year: str) -> pd.DataFrame:
    file_path = _year_to_path["sen"][year]
    df = load_sen_borough(str(file_path))
    return compute_sen_metrics(df)


@st.cache_data
def _load_th(year: str) -> pd.DataFrame:
    file_path = _year_to_path["th"][year]
    df = load_th_schools(str(file_path))
    return compute_school_metrics(df)


st.set_page_config(
    page_title="Tower Hamlets GCSE Attainment Gaps",
    layout="wide",
)

st.title("Tower Hamlets GCSE Attainment Gaps")
st.caption("Explore gender and SEN attainment gaps across London and within Tower Hamlets.")

view = st.radio("View", ["Borough level", "School level"], horizontal=True)
analysis_type = st.radio("Analysis", ["Gender", "SEN"], horizontal=True)

if view == "Borough level":
    years = sorted(_year_to_path["gender"] if analysis_type == "Gender" else _year_to_path["sen"])
    if not years:
        st.error("No borough-level datasets found in the data folder.")
        st.stop()

    year = st.selectbox("Year", years, index=len(years) - 1)

    if analysis_type == "Gender":
        df = _load_gender(year)
        gap_col = "gender_gap"
        z_col = "gender_gap_zscore_london"
        gap_label = "Gender gap (girls - boys)"
    else:
        df = _load_sen(year)
        gap_col = "sen_gap"
        z_col = "sen_gap_zscore_london"
        gap_label = "SEN gap (non-SEN - SEN)"

    boroughs = sorted(df["region_name"].dropna().unique())
    selected_borough = st.selectbox("Borough", boroughs, index=boroughs.index("Tower Hamlets") if "Tower Hamlets" in boroughs else 0)

    df_plot = df.copy()
    df_plot["highlight"] = np.where(df_plot["region_name"] == selected_borough, "Selected borough", "Other boroughs")

    fig = px.bar(
        df_plot.sort_values(gap_col),
        x="region_name",
        y=gap_col,
        color="highlight",
        color_discrete_map={"Selected borough": "#1f77b4", "Other boroughs": "#cccccc"},
        labels={"region_name": "Borough", gap_col: gap_label},
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    selected_row = df_plot[df_plot["region_name"] == selected_borough].iloc[0]
    gap_value = selected_row[gap_col]
    z_value = selected_row[z_col]
    rank = df_plot[gap_col].rank(ascending=False, method="min")[df_plot["region_name"] == selected_borough].iloc[0]

    st.subheader("Summary statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gap value", f"{gap_value:.2f}")
    col2.metric("London mean", f"{df_plot[gap_col].mean():.2f}")
    col3.metric("London SD", f"{df_plot[gap_col].std(ddof=0):.2f}")
    col4.metric("Rank (largest gap = 1)", f"{int(rank)} / {len(df_plot)}")
    st.write(f"**Z-score vs London mean:** {z_value:.2f}" if pd.notna(z_value) else "**Z-score vs London mean:** N/A")

else:
    years = sorted(_year_to_path["th"])
    if not years:
        st.error("No Tower Hamlets school datasets found in the data folder.")
        st.stop()
    year = st.selectbox("Year", years, index=len(years) - 1)

    df_schools = _load_th(year)

    left, right = st.columns(2)
    with left:
        fig_att8 = px.histogram(
            df_schools,
            x="attainment_8_score_all",
            nbins=20,
            labels={"attainment_8_score_all": "Attainment 8 score"},
            title="Distribution of Attainment 8 (Tower Hamlets)",
        )
        st.plotly_chart(fig_att8, use_container_width=True)

    with right:
        fig_gap = px.histogram(
            df_schools.dropna(subset=["gender_gap"]),
            x="gender_gap",
            nbins=20,
            labels={"gender_gap": "Gender gap (girls - boys)"},
            title="Distribution of Gender Gaps (Tower Hamlets)",
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    fig_scatter = px.scatter(
        df_schools,
        x="sen_proportion",
        y="attainment_8_score_all",
        hover_name="school_name",
        labels={"sen_proportion": "SEN proportion", "attainment_8_score_all": "Attainment 8 score"},
        title="SEN proportion vs Attainment 8",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Summary statistics")
    att8_mean = df_schools["attainment_8_score_all"].mean()
    att8_std = df_schools["attainment_8_score_all"].std(ddof=0)
    corr = df_schools[["sen_proportion", "attainment_8_score_all"]].corr().iloc[0, 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Attainment 8", f"{att8_mean:.2f}")
    col2.metric("SD Attainment 8", f"{att8_std:.2f}")
    col3.metric("Correlation (SEN proportion vs Attainment 8)", f"{corr:.2f}")


st.subheader("Interpretation notes")
st.markdown(
    """
- **Gender gap** is the difference between girls' and boys' average Attainment 8 scores.
- **Z-score** shows how far a value is from the London mean, measured in standard deviations.
- **Correlation does not imply causation**; a relationship in the data does not prove cause.
- **Context matters**: school size, intake, and local factors can influence outcomes.
"""
)
