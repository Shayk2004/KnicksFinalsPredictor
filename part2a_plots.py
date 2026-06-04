"""Matplotlib plots for Part 2A Knicks loss decomposition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")


def _save_barh(data: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_data[y_col], plot_data[x_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_top_loss_factors(decomposition_df: pd.DataFrame, top_n: int = 12) -> None:
    """Plot top features by absolute Cohen's d."""
    data = decomposition_df.head(top_n)
    _save_barh(
        data,
        "abs_cohens_d",
        "feature",
        "Top Knicks Loss Indicators by Absolute Cohen's d",
        "part2a_top_loss_factors.png",
    )


def plot_win_loss_comparison(decomposition_df: pd.DataFrame, top_n: int = 8) -> None:
    """Plot win and loss means for the top decomposed features."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = decomposition_df.head(top_n).iloc[::-1]
    y_positions = range(len(data))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data["win_mean"], y_positions, label="Wins")
    ax.scatter(data["loss_mean"], y_positions, label="Losses")
    for idx, (_, row) in enumerate(data.iterrows()):
        ax.plot([row["win_mean"], row["loss_mean"]], [idx, idx], alpha=0.45)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(data["feature"])
    ax.set_title("Knicks Win vs Loss Feature Means")
    ax.set_xlabel("Feature mean")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "part2a_win_loss_comparison.png", dpi=150)
    plt.close(fig)


def plot_logistic_coefficients(coefficients_df: pd.DataFrame, top_n: int = 12) -> None:
    """Plot largest logistic coefficients by absolute value."""
    data = coefficients_df.head(top_n).copy()
    _save_barh(
        data,
        "coefficient",
        "feature",
        "Top Logistic Regression Coefficients",
        "part2a_logistic_coefficients.png",
    )


def plot_random_forest_importance(importance_df: pd.DataFrame, top_n: int = 12) -> None:
    """Plot top random forest feature importances."""
    data = importance_df.head(top_n)
    _save_barh(
        data,
        "importance",
        "feature",
        "Top Random Forest Feature Importances",
        "part2a_random_forest_importance.png",
    )
