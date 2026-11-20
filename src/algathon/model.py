"""Entraînement et évaluation avec découpage temporel."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def temporal_split(df: pd.DataFrame, test_start: str):
    cut = pd.Timestamp(test_start)
    return df[df["date"] < cut], df[df["date"] >= cut]


def train(train_df: pd.DataFrame, feature_cols: list[str], params: dict | None = None):
    model = HistGradientBoostingClassifier(**(params or {}))
    model.fit(train_df[feature_cols], train_df["target"].astype(int))
    return model


def evaluate(model, df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.5) -> dict:
    y = df["target"].astype(int)
    p = model.predict_proba(df[feature_cols])[:, 1]
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "f1": f1_score(y, p >= threshold),
        "prevalence": y.mean(),
    }
