# 🎮 BabiPoly Player Engagement Prediction

Predict Low, Medium, or High player engagement from gameplay behavior and demographics for BabiPoly's game analytics workflow.

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

BabiPoly needs a reliable way to understand player engagement so product and retention teams can act before users lose interest. This project trains and evaluates machine learning models that classify players into Low, Medium, or High engagement groups using demographic, gameplay, session, achievement, and purchase behavior. The final model is exposed through a FastAPI prediction service so engagement predictions can be reused by dashboards, experiments, or future production applications.

## Demo

Screenshot / GIF here

## Project Structure

```text
github/
|-- api/
|   |-- main.py
|   |-- model.py
|   |-- requirements.txt
|   `-- schemas.py
|-- data/
|   |-- raw/
|   |   `-- online_gaming_behavior_dataset.csv
|   |-- cleaned_gaming.csv
|   |-- selected_features.txt
|   |-- X_train.csv
|   |-- X_test.csv
|   |-- y_train.csv
|   `-- y_test.csv
|-- models/
|   |-- decision_tree.joblib
|   |-- logistic_regression.joblib
|   |-- neural_network_mlp.joblib
|   |-- random_forest.joblib
|   `-- random_forest_tuned.joblib
|-- notebooks/
|   |-- engagement_prediction.ipynb
|   `-- engagement_prediction_executed.ipynb
|-- plots/
|   |-- cv_f1_weighted_all_models.png
|   |-- learning_curve_tuned_random_forest.png
|   `-- random_forest_top_10_feature_importances.png
|-- results/
|   |-- best_params.json
|   `-- evaluation_results.csv
|-- feature_engineering_pipeline.py
|-- model_evaluation_pipeline.ipynb
|-- model_training_pipeline.py
|-- random_forest_tuning_pipeline.py
|-- requirements.txt
|-- .gitattributes
|-- .gitignore
`-- README.md
```

## Dataset

- **Name:** Predict Online Gaming Behavior Dataset
- **Source:** [Kaggle - Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset/data)
- **Raw size in this repo:** 40,037 rows and 13 columns
- **Cleaned size:** 40,036 rows and 13 columns
- **Target:** `EngagementLevel` with `Low`, `Medium`, and `High` classes
- **Original features:** `PlayerID`, `Age`, `Gender`, `Location`, `GameGenre`, `PlayTimeHours`, `InGamePurchases`, `GameDifficulty`, `SessionsPerWeek`, `AvgSessionDurationMinutes`, `PlayerLevel`, `AchievementsUnlocked`, `EngagementLevel`
- **Selected model features:** `WeeklyPlayLoad`, `SessionsPerWeek`, `AvgSessionDurationMinutes`, `SessionDepth`, `AchievementsUnlocked`, `PlayerLevel`, `AchievementRate`, `GameGenre_RPG`, `PlayTimeHours`, `GameGenre_Simulation`, `Age`, `GameGenre_Strategy`, `InGamePurchases`, `Location_Europe`

## ML Pipeline

1. **Data Prep** - Load the raw Kaggle CSV, inspect structure, repair invalid IDs, handle missing values, clean unexpected categorical labels, and save `data/cleaned_gaming.csv`.
2. **Feature Eng** - Encode the target, create `WeeklyPlayLoad`, `AchievementRate`, and `SessionDepth`, one-hot encode categorical variables, scale numeric features, select the top 14 features, and save train/test files.
3. **Training** - Train Logistic Regression, Decision Tree, Random Forest, and Neural Network MLP classifiers, then save the fitted models into `models/`.
4. **Evaluation** - Evaluate all four models on the held-out test set with accuracy, weighted precision, weighted recall, weighted F1, and weighted one-vs-rest ROC-AUC.
5. **Tuning** - Tune the Random Forest with randomized search followed by grid search, save `models/random_forest_tuned.joblib`, and store best parameters in `results/best_params.json`.
6. **Deploy** - Serve the tuned Random Forest through FastAPI with `/predict` and `/health` endpoints.

## Model Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Decision Tree | 0.9170 | 0.9167 | 0.9299 |
| Logistic Regression | 0.8764 | 0.8753 | 0.9339 |
| Neural Network MLP | 0.9175 | 0.9172 | 0.9433 |
| Random Forest | 0.9245 | 0.9242 | 0.9467 |

The baseline Random Forest is the strongest four-model evaluator result. The tuned Random Forest is used by the API and reaches a held-out weighted F1 of `0.9276`.

## Installation & Running

### 1. Clone & install

```bash
git clone https://github.com/MedianeOz/AiInternshipProj.git
cd AiInternshipProj

git lfs install
git lfs pull

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r api/requirements.txt
```

The `.joblib` model artifacts are tracked with Git LFS because `models/random_forest.joblib` is larger than GitHub's regular 100 MiB file limit.

### 2. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/engagement_prediction.ipynb
```

To rerun the evaluation notebook from the terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace model_evaluation_pipeline.ipynb
```

### 3. Start the FastAPI server

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 4. Run the Streamlit dashboard

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## API Usage

### curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"SessionsPerWeek\":5,\"AvgSessionDurationMinutes\":45.0,\"PlayerLevel\":12,\"AchievementsUnlocked\":8,\"InGamePurchases\":1,\"Gender\":\"Male\",\"Location\":\"Nigeria\",\"GameGenre\":\"Strategy\"}"
```

Expected response shape:

```json
{
  "predicted_class": "Medium",
  "confidence": 0.8734,
  "all_probabilities": {
    "Low": 0.0312,
    "Medium": 0.8734,
    "High": 0.0954
  }
}
```

### Python requests

```python
import requests

payload = {
    "SessionsPerWeek": 5,
    "AvgSessionDurationMinutes": 45.0,
    "PlayerLevel": 12,
    "AchievementsUnlocked": 8,
    "InGamePurchases": 1,
    "Gender": "Male",
    "Location": "Nigeria",
    "GameGenre": "Strategy",
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
response.raise_for_status()
print(response.json())
```

## Key Insights

1. **Low-engagement players need retention focus.** The model can help identify players who may need reactivation campaigns, onboarding improvements, or personalized incentives before they churn.
2. **Medium-engagement players are strong candidates for nudges.** These users may respond well to targeted offers, reminders, new challenges, or content recommendations that move them toward high engagement.
3. **High-engagement players can support growth loops.** Highly engaged users are good candidates for loyalty programs, tournaments, referrals, premium features, and community-building efforts.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras for future deep learning extensions
- FastAPI
- Streamlit dashboard placeholder
- Jupyter Notebook
- Matplotlib and Seaborn
- Joblib
- Git LFS

## Author & Internship

- **Author:** Mediane Ozeir (`MedianeOz`)
- **Company:** BabiPoly
- **Program:** XpertBot Academy AI Internship
