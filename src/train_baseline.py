from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_churn_data
from preprocess import preprocess_telco
from pathlib import Path
import joblib
import numpy as np



def build_pipeline(cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=300)),
        ]
    )


def main():
    df = preprocess_telco(load_churn_data())

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(cat_cols, num_cols)
    pipeline.fit(X_train, y_train)
        # Save trained model
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "baseline_logreg_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to: {model_path}")


    # Predict probabilities (churn probability)
y_probs = pipeline.predict_proba(X_test)[:, 1]

# Tune threshold for higher recall (business prefers catching churners)
threshold = 0.35
y_pred_tuned = (y_probs >= threshold).astype(int)

print(f"Using threshold: {threshold}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))



if __name__ == "__main__":
    main()
