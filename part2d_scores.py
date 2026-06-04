"""Scoring formulas for Part 2D Knicks personnel actionability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def min_max_scale(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.5, index=series.index)
    return (values - min_value) / (max_value - min_value)


def z_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def reliability_weight(minutes: pd.Series, min_minutes: float = 300, full_weight_minutes: float = 1000) -> pd.Series:
    """Scale minutes reliability between 0 and 1."""
    values = pd.to_numeric(minutes, errors="coerce").fillna(0)
    return ((values - min_minutes) / (full_weight_minutes - min_minutes)).clip(0, 1)


def _eligible_players(player_summary: pd.DataFrame) -> pd.DataFrame:
    return player_summary[player_summary["total_minutes"] >= 300].copy()


def calculate_sga_pressure_score(player_summary: pd.DataFrame) -> pd.DataFrame:
    """Score players as SGA pressure/containment options."""
    out = _eligible_players(player_summary)
    reliability = reliability_weight(out["total_minutes"])
    defensive_events = out.get("blocks_per_36", pd.Series(0, index=out.index)) + out["steals_per_36"]
    out["sga_pressure_score"] = (
        0.30 * min_max_scale(out["steals_per_36"])
        + 0.25 * min_max_scale(out["plus_minus_per_36"])
        + 0.20 * min_max_scale(-out["fouls_per_36"])
        + 0.15 * min_max_scale(defensive_events)
        + 0.10 * reliability
    )
    return out.sort_values("sga_pressure_score", ascending=False).reset_index(drop=True)


def calculate_three_point_pressure_score(player_summary: pd.DataFrame) -> pd.DataFrame:
    """Score players as high-volume, credible 3PT pressure options."""
    out = _eligible_players(player_summary)
    reliability = reliability_weight(out["total_minutes"])
    volume_reliability = min_max_scale(out["fg3a"])
    accuracy_component = min_max_scale(out["fg3_pct"].fillna(0)) * volume_reliability
    out["three_point_pressure_score"] = (
        0.35 * min_max_scale(out["fg3a_per_36"])
        + 0.30 * accuracy_component
        + 0.20 * min_max_scale(out["three_point_attempt_rate"])
        + 0.10 * min_max_scale(out["efg_pct"].fillna(0))
        + 0.05 * reliability
    )
    return out.sort_values("three_point_pressure_score", ascending=False).reset_index(drop=True)


def calculate_rebounding_support_score(player_summary: pd.DataFrame) -> pd.DataFrame:
    """Score players as rebounding/rim support options."""
    out = _eligible_players(player_summary)
    reliability = reliability_weight(out["total_minutes"])
    out["rebounding_support_score"] = (
        0.45 * min_max_scale(out["rebounds_per_36"])
        + 0.25 * min_max_scale(out["blocks_per_36"])
        + 0.20 * min_max_scale(out["plus_minus_per_36"])
        + 0.10 * reliability
    )
    return out.sort_values("rebounding_support_score", ascending=False).reset_index(drop=True)


def calculate_bench_stability_score(player_summary: pd.DataFrame) -> pd.DataFrame:
    """Score players as bench stabilizers."""
    out = _eligible_players(player_summary)
    reliability = reliability_weight(out["total_minutes"])
    out["bench_stability_score"] = (
        0.30 * min_max_scale(out["plus_minus_per_36"])
        + 0.25 * min_max_scale(out["ast_tov_ratio"])
        + 0.20 * min_max_scale(out["efg_pct"].fillna(0))
        + 0.15 * min_max_scale(-out["turnovers_per_36"])
        + 0.10 * reliability
    )
    return out.sort_values("bench_stability_score", ascending=False).reset_index(drop=True)


def _eligible_combos(combo_summary: pd.DataFrame) -> pd.DataFrame:
    return combo_summary[
        (combo_summary["games_together"] >= 10)
        & (combo_summary["total_combined_minutes"] >= 200)
    ].copy()


def calculate_okc_bench_survival_score(combo_summary: pd.DataFrame) -> pd.DataFrame:
    """Score combos for stabilizing non-starter/bench-style minutes vs OKC."""
    out = _eligible_combos(combo_summary)
    reliability = reliability_weight(out["total_combined_minutes"], min_minutes=200, full_weight_minutes=1200)
    out["okc_bench_survival_score"] = (
        0.30 * min_max_scale(out["average_plus_minus"])
        + 0.25 * min_max_scale(out["combined_assists_per_turnover"])
        + 0.20 * min_max_scale(out["combined_efg_pct"].fillna(0))
        + 0.15 * min_max_scale(-out["combined_turnovers_per_36"])
        + 0.10 * reliability
    )
    return out.sort_values("okc_bench_survival_score", ascending=False).reset_index(drop=True)


def calculate_spacing_rebounding_score(combo_summary: pd.DataFrame) -> pd.DataFrame:
    """Score combos for SAS spacing plus rebounding pressure."""
    out = _eligible_combos(combo_summary)
    reliability = reliability_weight(out["total_combined_minutes"], min_minutes=200, full_weight_minutes=1200)
    out["sas_spacing_rebounding_score"] = (
        0.35 * min_max_scale(out["combined_fg3a_per_36"])
        + 0.25 * min_max_scale(out["combined_fg3_pct"].fillna(0))
        + 0.20 * min_max_scale(out["combined_rebounds_per_36"])
        + 0.10 * min_max_scale(out["combined_efg_pct"].fillna(0))
        + 0.10 * reliability
    )
    return out.sort_values("sas_spacing_rebounding_score", ascending=False).reset_index(drop=True)
