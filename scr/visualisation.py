"""
Plotly-based visualisation utilities for gap analysis.

Optional module for scripts and notebooks. Functions return Plotly figures
that can be displayed with fig.show() or passed to Streamlit via st.plotly_chart().
"""
import plotly.express as px
import plotly.graph_objects as go


class Visualiser:
    """
    Converts analytical outputs into Plotly visualisations (line, bar, scatter).
    Methods return Plotly figures for display or integration.
    """

    def __init__(self):
        pass

    def plot_line_chart(self, x_data, y_data, title, xlabel, ylabel):
        fig = px.line(
            x=x_data,
            y=y_data,
            markers=True,
            title=title,
            labels={"x": xlabel, "y": ylabel},
        )
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
        return fig

    def plot_bar_chart(self, categories, values, title, xlabel, ylabel):
        fig = px.bar(
            x=categories,
            y=values,
            title=title,
            labels={"x": xlabel, "y": ylabel},
            color_discrete_sequence=["skyblue"],
        )
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, showlegend=False)
        return fig

    def plot_scatter_chart(self, x_data, y_data, title, xlabel, ylabel):
        fig = px.scatter(
            x=x_data,
            y=y_data,
            title=title,
            labels={"x": xlabel, "y": ylabel},
            color_discrete_sequence=["red"],
        )
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, showlegend=False)
        return fig


def plot_borough_gap_comparison(df, gap_col, borough_col="region_name"):
    """Bar chart of borough gaps with Tower Hamlets highlighted."""
    sorted_df = df.sort_values(gap_col).copy()
    sorted_df["highlight"] = sorted_df[borough_col].apply(
        lambda x: "Tower Hamlets" if x == "Tower Hamlets" else "Other boroughs"
    )
    fig = px.bar(
        sorted_df,
        x=borough_col,
        y=gap_col,
        color="highlight",
        color_discrete_map={"Tower Hamlets": "orange", "Other boroughs": "skyblue"},
        title="Borough gap comparison",
        labels={borough_col: "Borough", gap_col: gap_col},
    )
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=500,
    )
    return fig


def plot_school_gap_distribution(df, gap_col):
    """Histogram of school-level gap distribution."""
    fig = px.histogram(
        df[gap_col].dropna(),
        nbins=20,
        title="School gap distribution",
        labels={"value": gap_col, "count": "Number of schools"},
        color_discrete_sequence=["steelblue"],
    )
    fig.update_layout(height=400)
    return fig


def plot_sen_vs_attainment(df, sen_prop_col, outcome_col):
    """Scatter plot of SEN proportion vs attainment."""
    fig = px.scatter(
        df,
        x=sen_prop_col,
        y=outcome_col,
        title="SEN proportion vs attainment",
        labels={sen_prop_col: sen_prop_col, outcome_col: outcome_col},
        color_discrete_sequence=["tomato"],
    )
    fig.update_layout(showlegend=False, height=450)
    return fig
