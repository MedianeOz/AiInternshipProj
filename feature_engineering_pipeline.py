"""
Task 3 - Feature Engineering Pipeline

This script continues from Task 2 by loading the cleaned gaming dataset,
engineering new features, encoding categorical columns, scaling numeric
features, selecting the strongest features, splitting the data, and saving the
train/test artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# STEP 1 - Load data
# Load the cleaned dataset created in Task 2 and inspect its structure.
data_dir = Path("data")
cleaned_data_path = data_dir / "cleaned_gaming.csv"

df = pd.read_csv(cleaned_data_path)

print("STEP 1 - df.columns:")
print(df.columns)
print("\nSTEP 1 - df.shape:")
print(df.shape)


# STEP 2 - Encode the target
# Convert EngagementLevel into numeric class labels for machine learning.
target_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["target"] = df["EngagementLevel"].map(target_mapping)

if df["target"].isnull().any():
    missing_targets = df.loc[df["target"].isnull(), "EngagementLevel"].unique()
    raise ValueError(f"Unmapped EngagementLevel values found: {missing_targets}")

print("\nSTEP 2 - Encoded target value counts:")
print(df["target"].value_counts().sort_index())


# STEP 3 - Create 3 new engineered features
# WeeklyPlayLoad: total estimated gaming minutes per week.
# AchievementRate: achievements unlocked per player level.
# SessionDepth: average session depth adjusted by weekly session frequency.
df["WeeklyPlayLoad"] = (
    df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"]
)
df["AchievementRate"] = df["AchievementsUnlocked"] / (df["PlayerLevel"] + 1)
df["SessionDepth"] = (
    df["AvgSessionDurationMinutes"] / (df["SessionsPerWeek"] + 1)
)

print("\nSTEP 3 - Engineered feature summary:")
print(df[["WeeklyPlayLoad", "AchievementRate", "SessionDepth"]].describe())


# STEP 4 - Encode categorical features
# One-hot encode the requested categorical columns while dropping the first
# category from each group to reduce multicollinearity.
categorical_columns = ["Gender", "Location", "GameGenre"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

print("\nSTEP 4 - Column list after one-hot encoding:")
print(df.columns.tolist())


# STEP 5 - Scale numeric features
# Scale only continuous numeric columns. The one-hot encoded boolean columns and
# the target column are intentionally excluded from scaling.
excluded_from_scaling = {"target"}
encoded_categorical_columns = [
    column
    for column in df.columns
    if column.startswith(("Gender_", "Location_", "GameGenre_"))
]

numeric_columns = [
    column
    for column in df.select_dtypes(include=[np.number]).columns
    if column not in excluded_from_scaling
    and column not in encoded_categorical_columns
]

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nSTEP 5 - Numeric columns scaled:")
print(numeric_columns)


# STEP 6 - Feature selection
# Build X from model-ready numeric and one-hot encoded columns, then score every
# candidate feature with ANOVA F-values.
columns_to_drop_from_x = ["target", "EngagementLevel"]
X = df.drop(columns=columns_to_drop_from_x)
X = X.select_dtypes(include=[np.number, "bool"])
y = df["target"].astype(int)

selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X, y)

feature_scores = pd.DataFrame(
    {
        "feature": X.columns,
        "score": selector.scores_,
    }
).sort_values(by="score", ascending=False)

print("\nSTEP 6 - Feature scores ranked descending:")
print(feature_scores.to_string(index=False))

top_k = 14
selected_feature_names = feature_scores.head(top_k)["feature"].tolist()
X_selected = X[selected_feature_names]

print("\nSTEP 6 - Selected top 14 features:")
print(selected_feature_names)


# STEP 7 - Train/test split
# Stratify by target so the train and test sets preserve the class balance.
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nSTEP 7 - X_train shape:")
print(X_train.shape)
print("\nSTEP 7 - X_test shape:")
print(X_test.shape)
print("\nSTEP 7 - Class distribution in y_train:")
print(y_train.value_counts().sort_index())


# STEP 8 - Save artifacts
# Save the split datasets and selected feature names for the next modeling task.
X_train.to_csv(data_dir / "X_train.csv", index=False)
X_test.to_csv(data_dir / "X_test.csv", index=False)
y_train.to_csv(data_dir / "y_train.csv", index=False, header=["target"])
y_test.to_csv(data_dir / "y_test.csv", index=False, header=["target"])

with open(data_dir / "selected_features.txt", "w", encoding="utf-8") as file:
    for feature_name in selected_feature_names:
        file.write(f"{feature_name}\n")

print(
    f"\nFeature engineering complete. {len(X_selected.columns)} features selected."
)
