import logging
import os
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_DATA_DIR_OVERRIDE = os.environ.get("DATA_DIR")
DATA_DIR = Path(_DATA_DIR_OVERRIDE) if _DATA_DIR_OVERRIDE else Path(__file__).parent / "data"

# Muted color palettes to distinguish Gender vs SEN charts
COLOR_GENDER_SELECTED = "#5a7d8a"  # muted teal
COLOR_GENDER_OTHER = "#b8cdd4"  # light teal
COLOR_SEN_SELECTED = "#8b7355"  # muted amber/brown
COLOR_SEN_OTHER = "#c4b5a0"  # light amber
COLOR_GIRLS = "#5a7d8a"  # muted teal
COLOR_BOYS = "#8ba3b0"  # lighter teal-blue


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


@st.cache_data(ttl=3600)
def _load_gender(year: str) -> pd.DataFrame:
    file_path = _year_to_path["gender"][year]
    logger.info("Loading gender borough data for year %s from %s", year, file_path)
    df = load_gender_borough(str(file_path))
    return compute_gender_metrics(df)


@st.cache_data(ttl=3600)
def _load_sen(year: str) -> pd.DataFrame:
    file_path = _year_to_path["sen"][year]
    logger.info("Loading SEN borough data for year %s from %s", year, file_path)
    df = load_sen_borough(str(file_path))
    return compute_sen_metrics(df)


@st.cache_data(ttl=3600)
def _load_th(year: str) -> pd.DataFrame:
    file_path = _year_to_path["th"][year]
    logger.info("Loading Tower Hamlets schools data for year %s from %s", year, file_path)
    df = load_th_schools(str(file_path))
    return compute_school_metrics(df)


st.set_page_config(
    page_title="Tower Hamlets GCSE Attainment Gaps",
    layout="wide",
)

if not DATA_DIR.exists():
    st.error(f"Data directory not found: {DATA_DIR}. Set DATA_DIR environment variable or ensure the data folder exists.")
    st.stop()

st.title("Tower Hamlets GCSE Attainment Gaps")
st.caption("Explore gender and SEN attainment gaps across London and within Tower Hamlets.")

view = st.radio("View", ["Borough level", "School level"], horizontal=True)

if view == "Borough level":
    analysis_type = st.radio("Analysis", ["Gender", "SEN"], horizontal=True)
    years = sorted(_year_to_path["gender"] if analysis_type == "Gender" else _year_to_path["sen"])
    if not years:
        st.error("No borough-level datasets found in the data folder.")
        st.stop()

    year = years[-1]  # Use most recent

    try:
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
    except (FileNotFoundError, KeyError, pd.errors.ParserError, ValueError) as e:
        logger.exception("Failed to load borough data")
        st.error(f"Failed to load data. Please check that data files exist and are valid CSV format. Error: {e}")
        st.stop()

    if df.empty:
        st.error("No borough data available after processing.")
        st.stop()

    boroughs = sorted(df["region_name"].dropna().unique())
    selected_borough = st.selectbox("Borough", boroughs, index=boroughs.index("Tower Hamlets") if "Tower Hamlets" in boroughs else 0)

    if analysis_type == "Gender":
        # Split into two side-by-side charts: girls and boys attainment by borough
        df_plot = df.copy()
        df_plot_sorted = df_plot.sort_values("girls_attainment_8_score")
        col_selected = COLOR_GENDER_SELECTED
        col_other = COLOR_GENDER_OTHER

        col_left, col_right = st.columns(2)
        with col_left:
            colors_girls = [col_selected if r == selected_borough else col_other for r in df_plot_sorted["region_name"]]
            fig_girls = px.bar(
                df_plot_sorted,
                x="region_name",
                y="girls_attainment_8_score",
                labels={"region_name": "Borough", "girls_attainment_8_score": "Girls Attainment 8"},
            )
            fig_girls.update_traces(marker_color=colors_girls)
            fig_girls.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_girls, use_container_width=True)
            st.caption("Girls' average Attainment 8 score by borough. Z-score measures how far from the London mean.")
        with col_right:
            df_plot_sorted_boys = df_plot.sort_values("boys_attainment_8_score")
            colors_boys = [col_selected if r == selected_borough else col_other for r in df_plot_sorted_boys["region_name"]]
            fig_boys = px.bar(
                df_plot_sorted_boys,
                x="region_name",
                y="boys_attainment_8_score",
                labels={"region_name": "Borough", "boys_attainment_8_score": "Boys Attainment 8"},
            )
            fig_boys.update_traces(marker_color=colors_boys)
            fig_boys.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_boys, use_container_width=True)
            st.caption("Boys' average Attainment 8 score by borough. Compare side-by-side with girls for gap context.")

        gap_col = "gender_gap"
        selected_row = df_plot[df_plot["region_name"] == selected_borough].iloc[0]
        gap_value = selected_row["gender_gap"]
        z_value = selected_row["gender_gap_zscore_london"]
    else:
        # SEN: single bar chart of gap, with correct sort order and colors
        df_plot = df.copy().sort_values(gap_col)
        col_selected = COLOR_SEN_SELECTED
        col_other = COLOR_SEN_OTHER
        colors = [col_selected if r == selected_borough else col_other for r in df_plot["region_name"]]
        fig = px.bar(
            df_plot,
            x="region_name",
            y=gap_col,
            labels={"region_name": "Borough", gap_col: gap_label},
        )
        fig.update_traces(marker_color=colors)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("SEN gap = non-SEN minus SEN Attainment 8. Z-score shows deviation from London mean.")

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
    st.caption("Gap = girls − boys (Gender) or non-SEN − SEN (SEN). Z-score measures standard deviations from London mean.")

else:
    years = sorted(_year_to_path["th"])
    if not years:
        st.error("No Tower Hamlets school datasets found in the data folder.")
        st.stop()
    year = years[-1]  # Use most recent

    try:
        df_schools = _load_th(year)
    except (FileNotFoundError, KeyError, pd.errors.ParserError, ValueError) as e:
        logger.exception("Failed to load Tower Hamlets schools data")
        st.error(f"Failed to load school data. Please check that data files exist and are valid CSV format. Error: {e}")
        st.stop()

    if df_schools.empty:
        st.error("No school data available for Tower Hamlets.")
        st.stop()

    col_girls, col_boys = st.columns(2)
    with col_girls:
        fig_girls_dist = px.histogram(
            df_schools.dropna(subset=["attainment_8_score_girls"]),
            x="attainment_8_score_girls",
            nbins=20,
            labels={"attainment_8_score_girls": "Attainment 8 score"},
            title="Girls' Attainment 8 Distribution (Tower Hamlets)",
        )
        fig_girls_dist.update_traces(marker_color=COLOR_GIRLS)
        st.plotly_chart(fig_girls_dist, use_container_width=True)
        st.caption("Distribution of girls' Attainment 8 scores across schools.")

    with col_boys:
        fig_boys_dist = px.histogram(
            df_schools.dropna(subset=["attainment_8_score_boys"]),
            x="attainment_8_score_boys",
            nbins=20,
            labels={"attainment_8_score_boys": "Attainment 8 score"},
            title="Boys' Attainment 8 Distribution (Tower Hamlets)",
        )
        fig_boys_dist.update_traces(marker_color=COLOR_BOYS)
        st.plotly_chart(fig_boys_dist, use_container_width=True)
        st.caption("Distribution of boys' Attainment 8 scores. Compare with girls for gap context.")

    fig_att8 = px.histogram(
        df_schools,
        x="attainment_8_score_all",
        nbins=20,
        labels={"attainment_8_score_all": "Attainment 8 score"},
        title="Distribution of Attainment 8 (Tower Hamlets)",
    )
    fig_att8.update_traces(marker_color=COLOR_SEN_SELECTED)
    st.plotly_chart(fig_att8, use_container_width=True)
    st.caption("Overall Attainment 8 across Tower Hamlets schools.")

    scatter_hover = {
        "attainment_8_score_all": ":.1f",
        "sen_proportion": ":.2f",
        "total_pupils": True,
    }
    for col in ["attainment_8_score_girls", "attainment_8_score_boys", "gender_gap"]:
        if col in df_schools.columns:
            scatter_hover[col] = ":.1f"
    fig_scatter = px.scatter(
        df_schools,
        x="sen_proportion",
        y="attainment_8_score_all",
        hover_name="school_name",
        hover_data=scatter_hover,
        labels={"sen_proportion": "SEN proportion", "attainment_8_score_all": "Attainment 8 score"},
        title="SEN proportion vs Attainment 8",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Correlation does not imply causation. School size, intake, and local factors can influence outcomes.")

    df_schools_sorted = df_schools.sort_values("attainment_8_score_all").reset_index(drop=True)
    bar_hover = {
        "attainment_8_score_all": ":.1f",
        "sen_proportion": ":.2f",
        "total_pupils": True,
    }
    for col in ["attainment_8_score_girls", "attainment_8_score_boys", "gender_gap"]:
        if col in df_schools_sorted.columns:
            bar_hover[col] = ":.1f"
    fig_schools_bar = px.bar(
        df_schools_sorted,
        x="school_name",
        y="attainment_8_score_all",
        hover_name="school_name",
        hover_data=bar_hover,
        labels={"school_name": "School", "attainment_8_score_all": "Attainment 8 score"},
        title="Schools by Attainment 8 (lowest to highest)",
    )
    fig_schools_bar.update_traces(marker_color=COLOR_SEN_SELECTED)
    fig_schools_bar.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_schools_bar, use_container_width=True)
    st.caption("Each bar is one school. Hover for context: SEN proportion, pupil count, girls/boys scores.")

    st.subheader("Summary statistics")
    att8_mean = df_schools["attainment_8_score_all"].mean()
    att8_std = df_schools["attainment_8_score_all"].std(ddof=0)
    corr = df_schools[["sen_proportion", "attainment_8_score_all"]].corr().iloc[0, 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Attainment 8", f"{att8_mean:.2f}")
    col2.metric("SD Attainment 8", f"{att8_std:.2f}")
    col3.metric("Correlation (SEN proportion vs Attainment 8)", f"{corr:.2f}")
    st.caption("Correlation describes association; it does not prove causation.")

with st.expander("Glossary and notes"):
    st.markdown(
        """
- **Gender gap** = girls' − boys' average Attainment 8. **SEN gap** = non-SEN − SEN Attainment 8.
- **Z-score** = standard deviations from the London mean.
- **Correlation does not imply causation**; context (school size, intake, local factors) matters.
"""
    )
