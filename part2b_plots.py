"""Matplotlib plots for Part 2B."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")


def _barh(df: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data[y_col], data[x_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def _signed_barh(df: pd.DataFrame, x_col: str, y_col: str, title: str, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2f6f9f" if value >= 0 else "#b65f5f" for value in data[x_col]]
    ax.barh(data[y_col], data[x_col], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_team_vulnerability(team_vulnerability_df: pd.DataFrame, team: str, top_n: int = 12) -> None:
    """Plot top team vulnerabilities by absolute Cohen's d."""
    _barh(
        team_vulnerability_df.head(top_n),
        "abs_cohens_d",
        "feature",
        f"{team} Team-Level Loss Vulnerabilities",
        f"part2b_{team.lower()}_team_vulnerabilities.png",
    )


def plot_team_vulnerability_signed(team_vulnerability_df: pd.DataFrame, team: str, top_n: int = 12) -> None:
    """Plot top team vulnerabilities using signed Cohen's d."""
    _signed_barh(
        team_vulnerability_df.head(top_n),
        "cohens_d",
        "feature",
        f"{team} Team-Level Signed Loss Vulnerabilities",
        f"part2b_{team.lower()}_team_vulnerabilities_signed.png",
    )


def plot_player_driver_factors(player_driver_df: pd.DataFrame, team: str, top_n: int = 12) -> None:
    """Plot top player/bench loss drivers by absolute Cohen's d."""
    _barh(
        player_driver_df.head(top_n),
        "abs_cohens_d",
        "feature",
        f"{team} Player and Bench Loss Drivers",
        f"part2b_{team.lower()}_player_loss_drivers.png",
    )


def plot_player_driver_factors_signed(player_driver_df: pd.DataFrame, team: str, top_n: int = 12) -> None:
    """Plot top player/bench loss drivers using signed Cohen's d."""
    _signed_barh(
        player_driver_df.head(top_n),
        "cohens_d",
        "feature",
        f"{team} Signed Player and Bench Loss Drivers",
        f"part2b_{team.lower()}_player_loss_drivers_signed.png",
    )


def plot_threshold_loss_rates(threshold_df: pd.DataFrame, team: str) -> None:
    """Plot loss rate when each threshold is true."""
    data = threshold_df.sort_values("team_loss_rate_when_true", ascending=False).head(12).iloc[::-1]
    _barh(
        data,
        "team_loss_rate_when_true",
        "threshold",
        f"{team} Loss Rate Under Threshold Conditions",
        f"part2b_{team.lower()}_threshold_loss_rates.png",
    )


def plot_threshold_loss_rate_lift(threshold_df: pd.DataFrame, team: str) -> None:
    """Plot threshold loss-rate lift relative to the team's overall loss rate."""
    data = threshold_df.sort_values("loss_rate_lift", ascending=False).head(12)
    _signed_barh(
        data,
        "loss_rate_lift",
        "condition_name",
        f"{team} Threshold Loss-Rate Lift",
        f"part2b_{team.lower()}_threshold_loss_rate_lift.png",
    )


def plot_star_feature_win_loss_comparison(
    df: pd.DataFrame, team: str, selected_features: list[str]
) -> None:
    """Plot win/loss means for selected star features."""
    available = [feature for feature in selected_features if feature in df.columns]
    if not available:
        return
    wins = df[df["team_win"] == 1]
    losses = df[df["team_win"] == 0]
    data = pd.DataFrame(
        {
            "feature": available,
            "win_mean": [wins[feature].mean() for feature in available],
            "loss_mean": [losses[feature].mean() for feature in available],
        }
    ).iloc[::-1]
    y_positions = range(len(data))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data["win_mean"], y_positions, label="Wins")
    ax.scatter(data["loss_mean"], y_positions, label="Losses")
    for idx, row in enumerate(data.itertuples(index=False)):
        ax.plot([row.win_mean, row.loss_mean], [idx, idx], alpha=0.45)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(data["feature"])
    ax.set_title(f"{team} Star Feature Win/Loss Comparison")
    ax.set_xlabel("Feature mean")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"part2b_{team.lower()}_star_win_loss_comparison.png", dpi=150)
    plt.close(fig)
