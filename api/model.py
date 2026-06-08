from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from api.schemas import PlayerInput, PredictionOutput


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "random_forest_tuned.joblib"
SELECTED_FEATURES_PATH = BASE_DIR / "data" / "selected_features.txt"
TRAINING_DATA_PATH = BASE_DIR / "data" / "cleaned_gaming.csv"

CATEGORICAL_COLUMNS = ("Gender", "Location", "GameGenre")
CATEGORICAL_PREFIXES = tuple(f"{column}_" for column in CATEGORICAL_COLUMNS)
CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}
PROBABILITY_LABELS = ("Low", "Medium", "High")

MODEL = None
SELECTED_FEATURES: list[str] = []
MODEL_LOAD_ERROR: str | None = None
SCALER_COLUMNS: list[str] = []
SCALER_MEANS: dict[str, float] = {}
SCALER_SCALES: dict[str, float] = {}
SCALER_LOAD_ERROR: str | None = None


def _read_selected_features() -> list[str]:
    """Read the selected feature names saved by the Task 3 pipeline."""
    features = [
        line.strip()
        for line in SELECTED_FEATURES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not features:
        raise ValueError("selected_features.txt is empty.")
    return features


def _add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the same engineered features used before model training."""
    frame = frame.copy()
    frame["WeeklyPlayLoad"] = (
        frame["SessionsPerWeek"] * frame["AvgSessionDurationMinutes"]
    )
    frame["AchievementRate"] = frame["AchievementsUnlocked"] / (
        frame["PlayerLevel"] + 1
    )
    frame["SessionDepth"] = frame["AvgSessionDurationMinutes"] / (
        frame["SessionsPerWeek"] + 1
    )
    return frame


def _fit_task3_scaler() -> None:
    """Rebuild Task 3 numeric scaling parameters from the cleaned training data."""
    global SCALER_COLUMNS, SCALER_MEANS, SCALER_SCALES, SCALER_LOAD_ERROR

    try:
        training_frame = pd.read_csv(TRAINING_DATA_PATH)
        training_frame = _add_engineered_features(training_frame)

        target_mapping = {"Low": 0, "Medium": 1, "High": 2}
        training_frame["target"] = training_frame["EngagementLevel"].map(
            target_mapping
        )

        # Match Task 3: one-hot columns are not scaled, but regular numeric
        # columns such as SessionsPerWeek and InGamePurchases are standardized.
        training_frame = pd.get_dummies(
            training_frame,
            columns=list(CATEGORICAL_COLUMNS),
            drop_first=True,
        )
        encoded_categorical_columns = [
            column
            for column in training_frame.columns
            if column.startswith(CATEGORICAL_PREFIXES)
        ]
        numeric_columns = [
            column
            for column in training_frame.select_dtypes(include=[np.number]).columns
            if column != "target" and column not in encoded_categorical_columns
        ]

        scaler = StandardScaler()
        scaler.fit(training_frame[numeric_columns])

        SCALER_COLUMNS = numeric_columns
        SCALER_MEANS = {
            column: float(mean)
            for column, mean in zip(numeric_columns, scaler.mean_, strict=True)
        }
        SCALER_SCALES = {
            column: float(scale)
            for column, scale in zip(numeric_columns, scaler.scale_, strict=True)
        }
        SCALER_LOAD_ERROR = None
    except Exception as exc:  # pragma: no cover - surfaced by /health.
        SCALER_COLUMNS = []
        SCALER_MEANS = {}
        SCALER_SCALES = {}
        SCALER_LOAD_ERROR = str(exc)


def _load_artifacts() -> None:
    """Load model artifacts once when this module is imported."""
    global MODEL, SELECTED_FEATURES, MODEL_LOAD_ERROR

    try:
        SELECTED_FEATURES = _read_selected_features()
        MODEL = joblib.load(MODEL_PATH)
        MODEL_LOAD_ERROR = None
    except Exception as exc:  # pragma: no cover - surfaced by /health.
        MODEL = None
        SELECTED_FEATURES = []
        MODEL_LOAD_ERROR = str(exc)

    _fit_task3_scaler()


def _model_dump(player: PlayerInput) -> dict[str, Any]:
    """Return a dict from Pydantic v1 or v2 model instances."""
    if hasattr(player, "model_dump"):
        return player.model_dump()
    return player.dict()


def _scale_numeric_features(features: dict[str, Any]) -> dict[str, Any]:
    """Standardize available numeric values with the Task 3 scaler parameters."""
    scaled_features = features.copy()
    for column in SCALER_COLUMNS:
        if column not in scaled_features:
            continue

        scale = SCALER_SCALES.get(column, 1.0)
        if scale == 0:
            scaled_features[column] = 0.0
            continue

        scaled_features[column] = (
            float(scaled_features[column]) - SCALER_MEANS[column]
        ) / scale
    return scaled_features


def _set_categorical_dummies(
    features: dict[str, Any],
    raw_values: dict[str, Any],
) -> dict[str, Any]:
    """Create one-hot categorical columns that match the selected feature list."""
    encoded_features = features.copy()
    for feature_name in SELECTED_FEATURES:
        for column in CATEGORICAL_COLUMNS:
            prefix = f"{column}_"
            if feature_name.startswith(prefix):
                expected_value = feature_name.removeprefix(prefix)
                encoded_features[feature_name] = int(
                    str(raw_values[column]) == expected_value
                )
    return encoded_features


def _build_feature_frame(player: PlayerInput) -> pd.DataFrame:
    """Build a single-row feature frame aligned to selected_features.txt."""
    raw_values = _model_dump(player)
    feature_frame = pd.DataFrame([raw_values])
    feature_frame = _add_engineered_features(feature_frame)

    features = feature_frame.iloc[0].to_dict()
    features = _scale_numeric_features(features)
    features = _set_categorical_dummies(features, raw_values)

    # Align request-time columns with training-time selected features. Any
    # selected feature not present in the API input is filled with 0.
    aligned_features = {
        feature_name: features.get(feature_name, 0)
        for feature_name in SELECTED_FEATURES
    }
    return pd.DataFrame([aligned_features], columns=SELECTED_FEATURES)


def _class_to_label(class_value: Any) -> str:
    """Convert model class values into API-facing labels."""
    if hasattr(class_value, "item"):
        class_value = class_value.item()
    return CLASS_LABELS.get(class_value, str(class_value))


def get_model_status() -> dict[str, Any]:
    """Return model and preprocessing load status for the health endpoint."""
    return {
        "model_loaded": MODEL is not None,
        "selected_features_loaded": bool(SELECTED_FEATURES),
        "selected_feature_count": len(SELECTED_FEATURES),
        "scaler_loaded": not SCALER_LOAD_ERROR,
        "model_path": str(MODEL_PATH),
        "selected_features_path": str(SELECTED_FEATURES_PATH),
        "training_data_path": str(TRAINING_DATA_PATH),
        "model_error": MODEL_LOAD_ERROR,
        "scaler_error": SCALER_LOAD_ERROR,
    }


def predict_engagement(player: PlayerInput) -> PredictionOutput:
    """Predict engagement class and probabilities for one player."""
    if MODEL is None or not SELECTED_FEATURES:
        raise RuntimeError(f"Model artifacts are not loaded: {MODEL_LOAD_ERROR}")

    feature_frame = _build_feature_frame(player)
    predicted_class = MODEL.predict(feature_frame)[0]
    probabilities = MODEL.predict_proba(feature_frame)[0]

    all_probabilities = {label: 0.0 for label in PROBABILITY_LABELS}
    for class_value, probability in zip(MODEL.classes_, probabilities, strict=True):
        class_label = _class_to_label(class_value)
        if class_label in all_probabilities:
            all_probabilities[class_label] = round(float(probability), 4)

    predicted_label = _class_to_label(predicted_class)
    confidence = all_probabilities[predicted_label]

    return PredictionOutput(
        predicted_class=predicted_label,
        confidence=confidence,
        all_probabilities=all_probabilities,
    )


_load_artifacts()
