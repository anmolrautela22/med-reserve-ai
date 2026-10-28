"""Synthetic data generation, model selection, and evaluation helpers."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["Age", "SpO2", "Heart_Rate", "Comorbidities"]
LABELS = {0: "Low risk", 1: "Moderate risk", 2: "High risk"}


def generate_synthetic_data(samples=2000, seed=42):
    """Generate rule-labeled data for software demonstration only."""
    if samples < 30:
        raise ValueError("At least 30 samples are needed for the stratified demo split.")
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, size=samples)
    spo2 = rng.integers(80, 100, size=samples)
    heart_rate = rng.integers(50, 140, size=samples)
    comorbidities = rng.choice([0, 1, 2, 3], size=samples, p=[0.4, 0.3, 0.2, 0.1])
    score = (100 - spo2) * 1.5 + age * 0.2 + comorbidities * 5
    levels = np.where(score > 35, 2, np.where(score > 20, 1, 0))
    return pd.DataFrame({
        "Age": age, "SpO2": spo2, "Heart_Rate": heart_rate,
        "Comorbidities": comorbidities, "Triage_Level": levels
    })


def _models():
    # Scaling helps Logistic Regression; tree models do not require it.
    return {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


def train_and_evaluate(output_dir, samples=2000, seed=42):
    """Select by training-fold CV macro-F1, then report held-out test results."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_data(samples=samples, seed=seed)
    X, y = data[FEATURES], data["Triage_Level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    candidates = _models()
    cv_rows = []
    for name, estimator in candidates.items():
        scores = cross_validate(estimator, X_train, y_train, cv=5,
                                scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"})
        cv_rows.append({
            "model": name, "cv_macro_f1": float(scores["test_macro_f1"].mean()),
            "cv_accuracy": float(scores["test_accuracy"].mean())
        })
    selected_name = max(cv_rows, key=lambda row: row["cv_macro_f1"])["model"]
    comparison = []
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        predicted = estimator.predict(X_test)
        comparison.append({
            "model": name,
            "cv_macro_f1": next(row["cv_macro_f1"] for row in cv_rows if row["model"] == name),
            "test_accuracy": float(accuracy_score(y_test, predicted)),
            "test_macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0))
        })
    selected_model = candidates[selected_name].fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)
    report = classification_report(
        y_test, y_pred, labels=[0, 1, 2],
        target_names=[LABELS[i] for i in range(3)], output_dict=True, zero_division=0
    )
    metrics = {
        "synthetic_demo_only": True, "samples": int(samples), "seed": int(seed),
        "selected_model": selected_name,
        "cv_comparison": sorted(cv_rows, key=lambda row: row["cv_macro_f1"], reverse=True),
        "test_comparison": sorted(comparison, key=lambda row: row["cv_macro_f1"], reverse=True),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "class_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist(),
        "class_labels": [LABELS[i] for i in range(3)], "features": FEATURES,
    }
    with (output / "med_reserve_model.pkl").open("wb") as file:
        pickle.dump(selected_model, file)
    (output / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path):
    """Load a locally generated pickle model artifact."""
    with Path(model_path).open("rb") as file:
        return pickle.load(file)


