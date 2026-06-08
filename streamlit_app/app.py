from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import confusion_matrix


st.set_page_config(
    page_title="BabiPoly AI Dashboard",
    page_icon="🎮",
    layout="wide",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "cleaned_gaming.csv"
RESULTS_PATH = ROOT_DIR / "results" / "evaluation_results.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest_tuned.joblib"
X_TEST_PATH = ROOT_DIR / "data" / "X_test.csv"
Y_TEST_PATH = ROOT_DIR / "data" / "y_test.csv"
API_URL = "http://localhost:8000/predict"

ENGAGEMENT_ORDER = ["Low", "Medium", "High"]
ENGAGEMENT_COLORS = {
    "Low": "#E74C3C",
    "Medium": "#F2B84B",
    "High": "#1D9E75",
}
CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}


st.markdown(
    """
    <style>
        .stApp {
            background: #0f1720;
            color: #f4f7f6;
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #f7fffc;
        }

        .stApp p,
        .stApp span,
        .stApp label,
        .stApp div,
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stForm"] label,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #e6f2ee;
        }

        [data-testid="stCaptionContainer"],
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {
            color: #a9c5bd;
        }

        .stSelectbox div[data-baseweb="select"],
        .stSelectbox div[data-baseweb="select"] *,
        .stMultiSelect div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] *,
        .stTextInput input,
        .stNumberInput input {
            color: #111827;
        }

        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background-color: #f8fafc;
            border-color: #cbd5e1;
        }

        .stSelectbox div[data-baseweb="select"] svg,
        .stMultiSelect div[data-baseweb="select"] svg {
            fill: #111827;
            color: #111827;
        }

        div[data-baseweb="popover"] *,
        ul[data-testid="stVirtualDropdown"] *,
        li[role="option"] {
            color: #111827;
        }

        .stButton button,
        .stDownloadButton button,
        [data-testid="stFormSubmitButton"] button {
            background: #1D9E75;
            border: 1px solid #34c79a;
            color: #ffffff;
            font-weight: 700;
        }

        .stButton button:hover,
        .stDownloadButton button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background: #25b989;
            border-color: #5ce0b4;
            color: #ffffff;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07130f 0%, #10241d 100%);
            border-right: 1px solid rgba(29, 158, 117, 0.35);
        }

        [data-testid="stSidebar"] * {
            color: #edf7f4;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: #dcefe9;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff;
        }

        [data-testid="stSidebar"] .stCaptionContainer {
            color: #a9c5bd;
        }

        .brand-header {
            padding: 1.1rem 1.2rem;
            margin-bottom: 1.1rem;
            border-left: 6px solid #1D9E75;
            background: linear-gradient(90deg, rgba(29, 158, 117, 0.22), rgba(29, 158, 117, 0.04));
            border-radius: 8px;
        }

        .brand-header h1 {
            margin: 0;
            color: #f7fffc;
            font-size: 2.1rem;
        }

        .brand-header p {
            margin: 0.35rem 0 0;
            color: #b7d8cd;
            font-size: 1.05rem;
        }

        .metric-card {
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(255, 255, 255, 0.045);
            border-radius: 8px;
        }

        .result-card {
            padding: 1.25rem;
            margin-top: 1rem;
            border: 1px solid rgba(29, 158, 117, 0.35);
            background: rgba(29, 158, 117, 0.08);
            border-radius: 8px;
        }

        .footer {
            margin-top: 2.5rem;
            padding-top: 1rem;
            color: #9fb8b0;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            text-align: center;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="brand-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<div class="footer">Built by Mediane Ozeir | BabiPoly AI Internship 2026</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    numeric_columns = [
        "Age",
        "PlayTimeHours",
        "InGamePurchases",
        "SessionsPerWeek",
        "AvgSessionDurationMinutes",
        "PlayerLevel",
        "AchievementsUnlocked",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["WeeklyPlayLoad"] = (
        data["SessionsPerWeek"] * data["AvgSessionDurationMinutes"]
    )
    return data


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    results = pd.read_csv(RESULTS_PATH)
    return results.rename(
        columns={
            "F1-Score (Weighted)": "F1",
            "ROC-AUC (OvR Weighted)": "ROC-AUC",
        }
    )


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    x_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")
    return x_test, y_test


def probability_chart(probabilities: dict[str, float]) -> go.Figure:
    rows = pd.DataFrame(
        {
            "EngagementLevel": ENGAGEMENT_ORDER,
            "Probability": [float(probabilities.get(label, 0.0)) for label in ENGAGEMENT_ORDER],
        }
    )
    fig = px.bar(
        rows,
        x="Probability",
        y="EngagementLevel",
        orientation="h",
        text=rows["Probability"].map(lambda value: f"{value:.1%}"),
        color="EngagementLevel",
        color_discrete_map=ENGAGEMENT_COLORS,
        category_orders={"EngagementLevel": ENGAGEMENT_ORDER},
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=20, t=20, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis_title=None,
        xaxis_title="Probability",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7f6",
    )
    return fig


def engagement_label(raw_label: str) -> str:
    labels = {
        "High": "🟢 High Engagement",
        "Medium": "🟡 Medium Engagement",
        "Low": "🔴 Low Engagement",
    }
    return labels.get(raw_label, raw_label)


def business_interpretation(raw_label: str) -> str:
    interpretations = {
        "High": "This player is highly engaged — consider targeting with premium content offers, tournaments, loyalty rewards, or referral prompts.",
        "Medium": "This player is moderately engaged — consider nudges such as personalized challenges, reminders, or limited-time offers.",
        "Low": "This player is at risk of low engagement — consider retention campaigns, onboarding help, or reactivation incentives.",
    }
    return interpretations.get(
        raw_label,
        "Use this prediction to guide the next best engagement action.",
    )


def page_predict() -> None:
    st.sidebar.markdown("### Page Tools")
    st.sidebar.caption("Submit a profile to call the FastAPI prediction service.")

    render_header(
        "Player Engagement Predictor",
        "Enter a player's profile to predict their engagement level",
    )

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            sessions_per_week = st.slider("SessionsPerWeek", 1, 20, 5)
            avg_session_duration = st.slider("AvgSessionDurationMinutes", 5, 180, 45)
            player_level = st.slider("PlayerLevel", 1, 50, 12)

        with col2:
            achievements_unlocked = st.slider("AchievementsUnlocked", 0, 100, 8)
            in_game_purchase_label = st.selectbox(
                "InGamePurchases",
                options=["0 = No", "1 = Yes"],
                index=0,
            )
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

        location = st.selectbox(
            "Location",
            options=["Nigeria", "Ghana", "Kenya", "South Africa", "Egypt"],
        )
        game_genre = st.selectbox(
            "GameGenre",
            options=["Strategy", "Action", "RPG", "Sports", "Puzzle"],
        )

        submitted = st.form_submit_button("Predict Engagement", use_container_width=True)

    if not submitted:
        st.info("Submit a player profile to generate a prediction.")
        return

    payload = {
        "SessionsPerWeek": sessions_per_week,
        "AvgSessionDurationMinutes": float(avg_session_duration),
        "PlayerLevel": player_level,
        "AchievementsUnlocked": achievements_unlocked,
        "InGamePurchases": 1 if in_game_purchase_label.startswith("1") else 0,
        "Gender": gender,
        "Location": location,
        "GameGenre": game_genre,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=8)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.warning(
            f"The prediction API is not available at {API_URL}. Start FastAPI and try again. Details: {exc}"
        )
        return

    predicted_class = result.get("predicted_class", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    probabilities = result.get("all_probabilities", {})

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    metric_col, progress_col = st.columns([1, 2])
    with metric_col:
        st.metric("Prediction", engagement_label(predicted_class))
    with progress_col:
        st.write(f"Confidence: **{confidence:.1%}**")
        st.progress(min(max(confidence, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.plotly_chart(probability_chart(probabilities), use_container_width=True)
    st.success(business_interpretation(predicted_class))


def sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Dataset Filters")
    genres = sorted(data["GameGenre"].dropna().unique().tolist())
    locations = sorted(data["Location"].dropna().unique().tolist())
    engagement_levels = [
        label for label in ENGAGEMENT_ORDER if label in set(data["EngagementLevel"])
    ]

    selected_genres = st.sidebar.multiselect("GameGenre", genres, default=genres)
    selected_locations = st.sidebar.multiselect("Location", locations, default=locations)
    selected_levels = st.sidebar.multiselect(
        "EngagementLevel",
        engagement_levels,
        default=engagement_levels,
    )

    return data[
        data["GameGenre"].isin(selected_genres)
        & data["Location"].isin(selected_locations)
        & data["EngagementLevel"].isin(selected_levels)
    ].copy()


def page_dataset_explorer() -> None:
    render_header(
        "Dataset Explorer",
        "Explore player behavior patterns across engagement classes",
    )

    try:
        data = load_dataset()
    except Exception as exc:
        st.warning(f"Unable to load dataset from {DATA_PATH}: {exc}")
        return

    filtered = sidebar_filters(data)
    total_players = len(filtered)
    high_rate = (
        (filtered["EngagementLevel"].eq("High").mean() * 100)
        if total_players
        else 0.0
    )
    avg_sessions = filtered["SessionsPerWeek"].mean() if total_players else 0.0
    avg_duration = (
        filtered["AvgSessionDurationMinutes"].mean() if total_players else 0.0
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Players", f"{total_players:,}")
    metric_cols[1].metric("High Engaged %", f"{high_rate:.1f}%")
    metric_cols[2].metric("Avg Sessions/Week", f"{avg_sessions:.1f}")
    metric_cols[3].metric("Avg Session Duration", f"{avg_duration:.1f} min")

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    distribution = (
        filtered["EngagementLevel"]
        .value_counts()
        .reindex(ENGAGEMENT_ORDER, fill_value=0)
        .reset_index()
    )
    distribution.columns = ["EngagementLevel", "Players"]

    fig_distribution = px.bar(
        distribution,
        x="EngagementLevel",
        y="Players",
        color="EngagementLevel",
        color_discrete_map=ENGAGEMENT_COLORS,
        category_orders={"EngagementLevel": ENGAGEMENT_ORDER},
        text="Players",
        title="Engagement Level Distribution",
    )
    fig_distribution.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7f6",
    )

    fig_scatter = px.scatter(
        filtered,
        x="SessionsPerWeek",
        y="AvgSessionDurationMinutes",
        color="EngagementLevel",
        color_discrete_map=ENGAGEMENT_COLORS,
        category_orders={"EngagementLevel": ENGAGEMENT_ORDER},
        hover_data=["GameGenre", "Location", "PlayerLevel"],
        title="Sessions per Week vs Average Session Duration",
        opacity=0.72,
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7f6",
    )

    fig_box = px.box(
        filtered,
        x="EngagementLevel",
        y="WeeklyPlayLoad",
        color="EngagementLevel",
        color_discrete_map=ENGAGEMENT_COLORS,
        category_orders={"EngagementLevel": ENGAGEMENT_ORDER},
        title="WeeklyPlayLoad Distribution by Engagement Level",
    )
    fig_box.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7f6",
    )

    st.plotly_chart(fig_distribution, use_container_width=True)
    scatter_col, box_col = st.columns(2)
    scatter_col.plotly_chart(fig_scatter, use_container_width=True)
    box_col.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Filtered Player Data")
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PlayerID": st.column_config.NumberColumn("Player ID", format="%d"),
            "Age": st.column_config.NumberColumn("Age", format="%d"),
            "PlayTimeHours": st.column_config.NumberColumn("Play Time Hours", format="%.1f"),
            "SessionsPerWeek": st.column_config.NumberColumn("Sessions / Week", format="%.1f"),
            "AvgSessionDurationMinutes": st.column_config.NumberColumn("Avg Duration", format="%.1f min"),
            "WeeklyPlayLoad": st.column_config.NumberColumn("Weekly Play Load", format="%.1f min"),
            "EngagementLevel": st.column_config.TextColumn("Engagement"),
        },
    )


def model_results_table(results: pd.DataFrame) -> None:
    display_columns = ["Model", "Accuracy", "F1", "ROC-AUC"]
    table = results[display_columns].copy()
    styler = (
        table.style.format(
            {
                "Accuracy": "{:.4f}",
                "F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }
        )
        .highlight_max(subset=["Accuracy", "F1", "ROC-AUC"], color="#1D9E75")
        .set_properties(**{"color": "#111827"})
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)


def performance_bar_chart(results: pd.DataFrame) -> go.Figure:
    long_results = results.melt(
        id_vars="Model",
        value_vars=["F1", "Accuracy", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        long_results,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text=long_results["Score"].map(lambda value: f"{value:.3f}"),
        color_discrete_sequence=["#1D9E75", "#F2B84B", "#4DA3FF"],
        title="Model Performance by Metric",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7f6",
    )
    return fig


def confusion_matrix_chart() -> go.Figure:
    model = load_model()
    x_test, y_test = load_test_data()
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    labels = [CLASS_LABELS[index] for index in [0, 1, 2]]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=[
                [0, "#15231f"],
                [0.5, "#1D9E75"],
                [1, "#B8F2DF"],
            ],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Players: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Tuned Random Forest Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7f6",
    )
    return fig


def feature_importance_chart() -> go.Figure:
    model = load_model()
    x_test, _ = load_test_data()

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("The tuned model does not expose feature_importances_.")

    importances = pd.DataFrame(
        {
            "Feature": x_test.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(10)

    fig = px.bar(
        importances.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        text=importances.sort_values("Importance")["Importance"].map(
            lambda value: f"{value:.3f}"
        ),
        color_discrete_sequence=["#1D9E75"],
        title="Top 10 Feature Importances",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7f6",
        yaxis_title=None,
    )
    return fig


def page_model_performance() -> None:
    st.sidebar.markdown("### Page Tools")
    st.sidebar.caption("Review saved model metrics and tuned Random Forest diagnostics.")

    render_header(
        "Model Performance",
        "Compare trained models and inspect the tuned Random Forest",
    )

    try:
        results = load_results()
    except Exception as exc:
        st.warning(f"Unable to load evaluation results from {RESULTS_PATH}: {exc}")
        return

    st.subheader("Model Comparison")
    model_results_table(results)
    st.plotly_chart(performance_bar_chart(results), use_container_width=True)

    heatmap_col, importance_col = st.columns(2)
    with heatmap_col:
        try:
            st.plotly_chart(confusion_matrix_chart(), use_container_width=True)
        except Exception as exc:
            st.warning(f"Unable to render confusion matrix: {exc}")

    with importance_col:
        try:
            st.plotly_chart(feature_importance_chart(), use_container_width=True)
        except Exception as exc:
            st.warning(f"Unable to render feature importances: {exc}")

    csv_data = pd.read_csv(RESULTS_PATH).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv_data,
        file_name="evaluation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.sidebar.title("🎮 BabiPoly AI")
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🔮 Predict Engagement",
            "📊 Dataset Explorer",
            "🏆 Model Performance",
        ],
    )

    if page == "🔮 Predict Engagement":
        page_predict()
    elif page == "📊 Dataset Explorer":
        page_dataset_explorer()
    else:
        page_model_performance()

    render_footer()


if __name__ == "__main__":
    main()
