# %% [markdown]
# # Random Forest Two-Stage Hyperparameter Tuning

# %%
from pathlib import Path
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve


# %% [markdown]
# # STEP 1 - Baseline Score

# %%
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

print("Loaded train/test files")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

baseline_rf = joblib.load(MODELS_DIR / "random_forest.joblib")
baseline_pred = baseline_rf.predict(X_test)
baseline_proba = baseline_rf.predict_proba(X_test)

baseline_f1 = f1_score(y_test, baseline_pred, average="weighted", zero_division=0)
baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_roc_auc = roc_auc_score(
    y_test,
    baseline_proba,
    multi_class="ovr",
    average="weighted",
)

print("\nSTEP 1 - Baseline Random Forest test results")
print(f"Baseline weighted F1: {baseline_f1:.4f}")
print(f"Baseline accuracy:    {baseline_accuracy:.4f}")
print(f"Baseline ROC-AUC:     {baseline_roc_auc:.4f}")


# %% [markdown]
# # STEP 2 - Stage 1: RandomizedSearchCV Broad Search

# %%
param_distributions = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

stage1_model = RandomForestClassifier(random_state=42, n_jobs=1)

random_search = RandomizedSearchCV(
    estimator=stage1_model,
    param_distributions=param_distributions,
    cv=5,
    n_iter=30,
    scoring="f1_weighted",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

print("\nSTEP 2 - Starting RandomizedSearchCV broad search")
stage1_start = time.time()
random_search.fit(X_train, y_train)
stage1_elapsed = time.time() - stage1_start

stage1_best_params = random_search.best_params_
stage1_best_cv_f1 = random_search.best_score_

print("\nSTEP 2 - RandomizedSearchCV results")
print(f"Stage 1 runtime: {stage1_elapsed / 60:.2f} minutes")
print("Best params:")
print(stage1_best_params)
print(f"Best CV F1: {stage1_best_cv_f1:.4f}")


# %% [markdown]
# # STEP 3 - Stage 2: GridSearchCV Narrow Search

# %%
def values_around_best(best_value, ordered_values):
    """Return the best value plus its immediate neighbors in an ordered grid."""
    best_index = ordered_values.index(best_value)
    start_index = max(best_index - 1, 0)
    end_index = min(best_index + 2, len(ordered_values))
    return ordered_values[start_index:end_index]


narrow_param_grid = {
    "n_estimators": values_around_best(
        stage1_best_params["n_estimators"],
        param_distributions["n_estimators"],
    ),
    "max_depth": values_around_best(
        stage1_best_params["max_depth"],
        param_distributions["max_depth"],
    ),
    "min_samples_split": values_around_best(
        stage1_best_params["min_samples_split"],
        param_distributions["min_samples_split"],
    ),
    "min_samples_leaf": values_around_best(
        stage1_best_params["min_samples_leaf"],
        param_distributions["min_samples_leaf"],
    ),
    "max_features": values_around_best(
        stage1_best_params["max_features"],
        param_distributions["max_features"],
    ),
    # Bootstrap is boolean, so the narrow grid keeps the Stage 1 winner.
    "bootstrap": [stage1_best_params["bootstrap"]],
}

print("\nSTEP 3 - Narrow parameter grid")
print(narrow_param_grid)
grid_candidate_count = int(np.prod([len(values) for values in narrow_param_grid.values()]))
print(f"Narrow grid candidates: {grid_candidate_count}")
print(f"Narrow grid total CV fits: {grid_candidate_count * 5}")

stage2_model = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    estimator=stage2_model,
    param_grid=narrow_param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
)

print("\nSTEP 3 - Starting GridSearchCV narrow search")
stage2_start = time.time()
grid_search.fit(X_train, y_train)
stage2_elapsed = time.time() - stage2_start

final_best_params = grid_search.best_params_
final_best_cv_f1 = grid_search.best_score_

print("\nSTEP 3 - GridSearchCV results")
print(f"Stage 2 runtime: {stage2_elapsed / 60:.2f} minutes")
print("Final best params:")
print(final_best_params)
print(f"Final best CV F1: {final_best_cv_f1:.4f}")


# %% [markdown]
# # STEP 4 - Retrain with Best Params

# %%
tuned_rf = RandomForestClassifier(
    **final_best_params,
    random_state=42,
    n_jobs=-1,
)

print("\nSTEP 4 - Retraining tuned Random Forest on full X_train")
retrain_start = time.time()
tuned_rf.fit(X_train, y_train)
retrain_elapsed = time.time() - retrain_start

tuned_pred = tuned_rf.predict(X_test)
tuned_proba = tuned_rf.predict_proba(X_test)

tuned_f1 = f1_score(y_test, tuned_pred, average="weighted", zero_division=0)
tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_roc_auc = roc_auc_score(
    y_test,
    tuned_proba,
    multi_class="ovr",
    average="weighted",
)

print("\nSTEP 4 - Tuned Random Forest test results")
print(f"Retraining time: {retrain_elapsed:.2f} seconds")
print(f"Tuned weighted F1: {tuned_f1:.4f}")
print(f"Tuned accuracy:    {tuned_accuracy:.4f}")
print(f"Tuned ROC-AUC:     {tuned_roc_auc:.4f}")


# %% [markdown]
# # STEP 5 - Improvement Summary

# %%
comparison_df = pd.DataFrame(
    [
        {
            "Model": "Baseline RF",
            "Test F1": baseline_f1,
            "Accuracy": baseline_accuracy,
            "ROC-AUC": baseline_roc_auc,
        },
        {
            "Model": "Tuned RF",
            "Test F1": tuned_f1,
            "Accuracy": tuned_accuracy,
            "ROC-AUC": tuned_roc_auc,
        },
        {
            "Model": "Improvement (%)",
            "Test F1": ((tuned_f1 - baseline_f1) / baseline_f1) * 100,
            "Accuracy": ((tuned_accuracy - baseline_accuracy) / baseline_accuracy)
            * 100,
            "ROC-AUC": ((tuned_roc_auc - baseline_roc_auc) / baseline_roc_auc)
            * 100,
        },
    ]
)

print("\nSTEP 5 - Baseline vs Tuned Random Forest")
print(comparison_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


# %% [markdown]
# # STEP 6 - Learning Curve

# %%
print("\nSTEP 6 - Calculating learning curve for tuned Random Forest")

learning_curve_rf = RandomForestClassifier(
    **final_best_params,
    random_state=42,
    n_jobs=1,
)

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=learning_curve_rf,
    X=X_train,
    y=y_train,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    shuffle=True,
    random_state=42,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
validation_mean = validation_scores.mean(axis=1)
validation_std = validation_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    train_sizes,
    train_mean,
    marker="o",
    linewidth=2,
    label="Training F1-weighted",
)
ax.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2,
)

ax.plot(
    train_sizes,
    validation_mean,
    marker="s",
    linewidth=2,
    label="Validation F1-weighted",
)
ax.fill_between(
    train_sizes,
    validation_mean - validation_std,
    validation_mean + validation_std,
    alpha=0.2,
)

ax.set_title("Learning Curve — Tuned Random Forest")
ax.set_xlabel("Training Set Size")
ax.set_title("Learning Curve — Tuned Random Forest")
ax.set_ylabel("F1-weighted")
ax.legend(loc="best")
ax.grid(alpha=0.3)

fig.tight_layout()
learning_curve_path = PLOTS_DIR / "learning_curve_tuned_random_forest.png"
fig.savefig(learning_curve_path, dpi=300)
plt.close(fig)

print(f"Learning curve saved to {learning_curve_path}")


# %% [markdown]
# # STEP 7 - Save Tuned Model and Best Params

# %%
tuned_model_path = MODELS_DIR / "random_forest_tuned.joblib"
best_params_path = RESULTS_DIR / "best_params.json"

joblib.dump(tuned_rf, tuned_model_path)

with best_params_path.open("w", encoding="utf-8") as file:
    json.dump(final_best_params, file, indent=4)

print("\nSTEP 7 - Saved tuned artifacts")
print(f"Tuned model path: {tuned_model_path}")
print(f"Best params path: {best_params_path}")
print(f"Tuned model saved. Final F1: {tuned_f1:.4f}")
