"""Analysis routines for Knicks loss decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_wins_losses(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Knicks wins and losses."""
    return df[df["knicks_win"] == 1].copy(), df[df["knicks_win"] == 0].copy()


def compute_win_loss_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Summarize feature means in wins and losses."""
    wins, losses = split_wins_losses(df)
    rows = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        rows.append(
            {
                "feature": feature,
                "win_mean": wins[feature].mean(),
                "loss_mean": losses[feature].mean(),
                "difference": losses[feature].mean() - wins[feature].mean(),
            }
        )
    return pd.DataFrame(rows)


def compute_cohens_d(wins: pd.DataFrame, losses: pd.DataFrame, feature: str) -> float:
    """Compute Cohen's d as loss mean minus win mean over pooled standard deviation."""
    win_values = wins[feature].dropna().astype(float)
    loss_values = losses[feature].dropna().astype(float)
    n_win = len(win_values)
    n_loss = len(loss_values)
    if n_win < 2 or n_loss < 2:
        return np.nan

    pooled_variance = (
        ((n_win - 1) * win_values.var(ddof=1))
        + ((n_loss - 1) * loss_values.var(ddof=1))
    ) / (n_win + n_loss - 2)
    if pooled_variance <= 0:
        return 0.0
    return float((loss_values.mean() - win_values.mean()) / np.sqrt(pooled_variance))


def run_loss_decomposition(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Rank loss-associated features by absolute effect size."""
    wins, losses = split_wins_losses(df)
    rows = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        win_mean = wins[feature].mean()
        loss_mean = losses[feature].mean()
        difference = loss_mean - win_mean
        percent_difference = (
            difference / abs(win_mean) if pd.notna(win_mean) and abs(win_mean) > 1e-9 else np.nan
        )
        cohens_d = compute_cohens_d(wins, losses, feature)
        rows.append(
            {
                "feature": feature,
                "win_mean": win_mean,
                "loss_mean": loss_mean,
                "difference": difference,
                "percent_difference": percent_difference,
                "cohens_d": cohens_d,
                "abs_cohens_d": abs(cohens_d) if pd.notna(cohens_d) else np.nan,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_cohens_d", ascending=False)
        .reset_index(drop=True)
    )


def _model_frame(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    available_features = [feature for feature in feature_cols if feature in df.columns]
    model_df = df[available_features + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return model_df[available_features], model_df[target_col].astype(int)


def _safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def run_logistic_regression(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "knicks_win"
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit scaled logistic regression and return sorted coefficients plus metrics."""
    x, y = _model_frame(df, feature_cols, target_col)
    if y.nunique() < 2:
        raise ValueError("Logistic regression requires both wins and losses.")

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=stratify
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    coefficients = pipeline.named_steps["model"].coef_[0]

    coefficients_df = (
        pd.DataFrame({"feature": x.columns, "coefficient": coefficients})
        .assign(abs_coefficient=lambda data: data["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": _safe_roc_auc(y_test, y_prob),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    return coefficients_df, metrics


def run_random_forest(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "knicks_win"
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit random forest and return sorted feature importances plus metrics."""
    x, y = _model_frame(df, feature_cols, target_col)
    if y.nunique() < 2:
        raise ValueError("Random forest requires both wins and losses.")

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    importance_df = (
        pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": _safe_roc_auc(y_test, y_prob),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    return importance_df, metrics
