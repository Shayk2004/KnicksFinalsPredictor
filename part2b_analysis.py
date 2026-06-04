"""Analysis functions for Part 2B opponent vulnerabilities."""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def compute_cohens_d(wins: pd.DataFrame, losses: pd.DataFrame, feature: str) -> float:
    """Compute Cohen's d as loss mean minus win mean over pooled standard deviation."""
    win_values = wins[feature].dropna().astype(float)
    loss_values = losses[feature].dropna().astype(float)
    if len(win_values) < 2 or len(loss_values) < 2:
        return np.nan
    pooled_var = (
        ((len(win_values) - 1) * win_values.var(ddof=1))
        + ((len(loss_values) - 1) * loss_values.var(ddof=1))
    ) / (len(win_values) + len(loss_values) - 2)
    if pooled_var <= 0:
        return 0.0
    return float((loss_values.mean() - win_values.mean()) / np.sqrt(pooled_var))


def _effect_size_table(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    wins = df[df[target_col] == 1]
    losses = df[df[target_col] == 0]
    rows = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        win_mean = wins[feature].mean()
        loss_mean = losses[feature].mean()
        difference = loss_mean - win_mean
        cohens_d = compute_cohens_d(wins, losses, feature)
        if pd.isna(cohens_d) or cohens_d == 0:
            direction = "no_difference"
        elif cohens_d > 0:
            direction = "higher_in_losses"
        else:
            direction = "lower_in_losses"
        rows.append(
            {
                "feature": feature,
                "win_mean": win_mean,
                "loss_mean": loss_mean,
                "difference": difference,
                "cohens_d": cohens_d,
                "abs_cohens_d": abs(cohens_d) if pd.notna(cohens_d) else np.nan,
                "direction": direction,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)


def run_team_vulnerability_analysis(team_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Rank team-level features associated with losses."""
    return _effect_size_table(team_df, feature_cols, "team_win")


def run_player_driver_analysis(team_player_features_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Rank player and bench features associated with losses."""
    return _effect_size_table(team_player_features_df, feature_cols, "team_win")


OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def run_threshold_analysis(
    df: pd.DataFrame, thresholds: list[dict[str, object]], target_col: str = "team_win"
) -> pd.DataFrame:
    """Evaluate win/loss rates when specific vulnerability conditions are met."""
    rows = []
    overall_win_rate = df[target_col].mean()
    overall_loss_rate = 1 - overall_win_rate if pd.notna(overall_win_rate) else np.nan
    for threshold in thresholds:
        feature = str(threshold["feature"])
        op = str(threshold["op"])
        value = threshold["value"]
        if feature not in df.columns or op not in OPS:
            continue
        if value == "median":
            value = df[feature].median()
        mask = OPS[op](df[feature], value).fillna(False)
        true_df = df[mask]
        false_df = df[~mask]
        wins_when_true = int(true_df[target_col].sum())
        wins_when_false = int(false_df[target_col].sum())
        losses_when_true = int(len(true_df) - wins_when_true)
        losses_when_false = int(len(false_df) - wins_when_false)
        true_win_rate = true_df[target_col].mean() if len(true_df) else np.nan
        false_win_rate = false_df[target_col].mean() if len(false_df) else np.nan
        true_loss_rate = 1 - true_win_rate if pd.notna(true_win_rate) else np.nan
        false_loss_rate = 1 - false_win_rate if pd.notna(false_win_rate) else np.nan
        condition_name = threshold.get("name", f"{feature} {op} {value}")
        rows.append(
            {
                "condition_name": condition_name,
                "threshold": condition_name,
                "feature": feature,
                "operator": op,
                "value": value,
                "overall_loss_rate": overall_loss_rate,
                "games_condition_true": int(mask.sum()),
                "games_condition_false": int((~mask).sum()),
                "wins_when_true": wins_when_true,
                "losses_when_true": losses_when_true,
                "wins_when_false": wins_when_false,
                "losses_when_false": losses_when_false,
                "games_meeting_condition": int(mask.sum()),
                "team_win_rate_when_true": true_win_rate,
                "team_loss_rate_when_true": true_loss_rate,
                "team_win_rate_when_false": false_win_rate,
                "team_loss_rate_when_false": false_loss_rate,
                "loss_rate_lift": true_loss_rate - overall_loss_rate,
                "difference_in_loss_rate": true_loss_rate - false_loss_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("loss_rate_lift", ascending=False).reset_index(drop=True)


def _model_frame(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    available = [feature for feature in feature_cols if feature in df.columns]
    model_df = df[available + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return model_df[available], model_df[target_col].astype(int)


def _safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def run_logistic_regression(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "team_win"
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit scaled logistic regression for secondary validation."""
    x, y = _model_frame(df, feature_cols, target_col)
    if y.nunique() < 2:
        raise ValueError("Logistic regression requires both wins and losses.")
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=stratify
    )
    model = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    coeffs = model.named_steps["model"].coef_[0]
    coeff_df = (
        pd.DataFrame({"feature": x.columns, "coefficient": coeffs})
        .assign(abs_coefficient=lambda data: data["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": _safe_auc(y_test, y_prob),
    }
    return coeff_df, metrics


def run_random_forest(
    df: pd.DataFrame, feature_cols: list[str], target_col: str = "team_win"
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit random forest for secondary validation."""
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
        "roc_auc": _safe_auc(y_test, y_prob),
    }
    return importance_df, metrics
