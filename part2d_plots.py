"""Matplotlib plots for Part 2D personnel actionability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")


def _barh(df: pd.DataFrame, label_col: str, score_col: str, title: str, filename: str, top_n: int = 10) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data[label_col], data[score_col])
    ax.set_title(title)
    ax.set_xlabel(score_col)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_sga_pressure_candidates(df: pd.DataFrame) -> None:
    _barh(df, "player_name", "sga_pressure_score", "SGA Pressure Candidates", "part2d_sga_pressure_candidates.png")


def plot_sas_three_point_pressure_candidates(df: pd.DataFrame) -> None:
    _barh(df, "player_name", "three_point_pressure_score", "SAS 3PT Pressure Candidates", "part2d_sas_three_point_pressure_candidates.png")


def plot_okc_bench_survival_combos(df: pd.DataFrame) -> None:
    _barh(df, "combo", "okc_bench_survival_score", "OKC Bench Survival Combos", "part2d_okc_bench_survival_combos.png")


def plot_sas_spacing_rebounding_combos(df: pd.DataFrame) -> None:
    _barh(df, "combo", "sas_spacing_rebounding_score", "SAS Spacing/Rebounding Combos", "part2d_sas_spacing_rebounding_combos.png")


def plot_actionability_summary(df: pd.DataFrame) -> None:
    plot_df = df.head(12).copy()
    plot_df["label"] = plot_df["opponent"] + ": " + plot_df["player_or_combo"]
    _barh(plot_df, "label", "score", "Part 2D Actionability Summary", "part2d_actionability_summary.png", top_n=12)
