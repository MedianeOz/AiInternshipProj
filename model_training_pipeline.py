# %% [markdown]
# # STEP 1 - Imports

# %%
from pathlib import Path
import re
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# %% [markdown]
# # Load Train/Test Splits

# %%
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")


# %% [markdown]
# # STEP 2 - Define Models

# %%
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300, random_state=42
    ),
}


# %% [markdown]
# # STEP 3 - Cross-Validation on Training Set

# %%
cv = StratifiedKFold(n_splits=5)
cv_results = {}

for model_name, model in MODELS.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
    )
    cv_results[model_name] = {
        "mean_f1_weighted": scores.mean(),
        "std_f1_weighted": scores.std(),
        "scores": scores,
    }
    print(
        f"{model_name}: "
        f"mean CV F1 = {scores.mean():.4f}, "
        f"std CV F1 = {scores.std():.4f}"
    )


# %% [markdown]
# # STEP 4 - Fit All Models on Full Training Set

# %%
training_times = {}

for model_name, model in MODELS.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    training_times[model_name] = elapsed_time
    print(f"{model_name}: training time = {elapsed_time:.2f} seconds")


# %% [markdown]
# # STEP 5 - Plot CV Results Comparison

# %%
PLOTS_DIR.mkdir(exist_ok=True)

model_names = list(cv_results.keys())
mean_scores = [cv_results[name]["mean_f1_weighted"] for name in model_names]
std_scores = [cv_results[name]["std_f1_weighted"] for name in model_names]

colorblind_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(
    model_names,
    mean_scores,
    yerr=std_scores,
    capsize=6,
    color=colorblind_palette,
    edgecolor="black",
    linewidth=0.8,
)

ax.set_title("5-Fold Cross-Validation F1 (Weighted) — All Models")
ax.set_ylabel("Mean CV F1 (Weighted)")
ax.set_ylim(0, max(mean_scores) + max(std_scores) + 0.08)
ax.tick_params(axis="x", rotation=20)

for bar, score in zip(bars, mean_scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

fig.tight_layout()
cv_plot_path = PLOTS_DIR / "cv_f1_weighted_all_models.png"
fig.savefig(cv_plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"CV results plot saved to {cv_plot_path}")


# %% [markdown]
# # STEP 6 - Feature Importance (Random Forest)

# %%
feature_names = [
    line.strip()
    for line in (DATA_DIR / "selected_features.txt").read_text().splitlines()
    if line.strip()
]

random_forest = MODELS["Random Forest"]
feature_importances = random_forest.feature_importances_

importance_df = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importances,
        }
    )
    .sort_values("importance", ascending=False)
    .head(10)
    .sort_values("importance", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    importance_df["feature"],
    importance_df["importance"],
    color="#009E73",
    edgecolor="black",
    linewidth=0.8,
)
ax.set_title("Random Forest — Top 10 Feature Importances")
ax.set_xlabel("Importance")

for index, value in enumerate(importance_df["importance"]):
    ax.text(value + 0.002, index, f"{value:.3f}", va="center", fontsize=9)

fig.tight_layout()
feature_importance_plot_path = PLOTS_DIR / "random_forest_top_10_feature_importances.png"
fig.savefig(feature_importance_plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Feature importance plot saved to {feature_importance_plot_path}")


# %% [markdown]
# # STEP 7 - Save All Models

# %%
MODELS_DIR.mkdir(exist_ok=True)


def to_snake_case(model_name):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", model_name.lower())).strip("_")


for model_name, model in MODELS.items():
    model_path = MODELS_DIR / f"{to_snake_case(model_name)}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved {model_name} to {model_path}")

print("All 4 models saved to models/")
