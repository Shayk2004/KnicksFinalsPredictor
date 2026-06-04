"""Analysis wrappers for Part 2D personnel actionability."""

from __future__ import annotations

import pandas as pd


SCENARIO_GAINS = {
    "OKC-1 SGA pressure and efficiency reduction": "+4.02%",
    "OKC-2 Bench survival": "+3.11%",
    "SAS-1 Wembanyama spacing neutralization": "+0.92%",
    "SAS-3 Bench and rebounding control": "+0.66%",
    "Combined best actionable plan": "+6.54%",
}


def rank_sga_pressure_candidates(player_summary: pd.DataFrame) -> pd.DataFrame:
    return player_summary.sort_values("sga_pressure_score", ascending=False).reset_index(drop=True)


def rank_okc_bench_survival_combos(combo_summary: pd.DataFrame) -> pd.DataFrame:
    return combo_summary.sort_values("okc_bench_survival_score", ascending=False).reset_index(drop=True)


def rank_sas_three_point_pressure_candidates(player_summary: pd.DataFrame) -> pd.DataFrame:
    return player_summary.sort_values("three_point_pressure_score", ascending=False).reset_index(drop=True)


def rank_sas_spacing_rebounding_combos(combo_summary: pd.DataFrame) -> pd.DataFrame:
    return combo_summary.sort_values("sas_spacing_rebounding_score", ascending=False).reset_index(drop=True)


def _reason_from_row(row: pd.Series, score_col: str) -> str:
    if score_col == "sga_pressure_score":
        return (
            f"steals/36 {row['steals_per_36']:.2f}, fouls/36 {row['fouls_per_36']:.2f}, "
            f"plus-minus/36 {row['plus_minus_per_36']:.2f}"
        )
    if score_col == "three_point_pressure_score":
        return (
            f"3PA/36 {row['fg3a_per_36']:.2f}, 3P% {row['fg3_pct']:.3f}, "
            f"3PA rate {row['three_point_attempt_rate']:.3f}"
        )
    if score_col == "okc_bench_survival_score":
        return (
            f"avg plus-minus {row['average_plus_minus']:.2f}, AST/TOV "
            f"{row['combined_assists_per_turnover']:.2f}, eFG% {row['combined_efg_pct']:.3f}"
        )
    return (
        f"3PA/36 {row['combined_fg3a_per_36']:.2f}, 3P% {row['combined_fg3_pct']:.3f}, "
        f"REB/36 {row['combined_rebounds_per_36']:.2f}"
    )


def create_actionability_table(
    sga_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    three_pt_df: pd.DataFrame,
    spacing_df: pd.DataFrame,
) -> pd.DataFrame:
    """Link top players/combos to tactical scenarios."""
    rows = []

    for idx, row in enumerate(sga_df.head(5).itertuples(index=False), start=1):
        role = "primary SGA pressure defender" if idx == 1 else "secondary help defender"
        rows.append(
            {
                "opponent": "OKC",
                "scenario": "OKC-1 SGA pressure and efficiency reduction",
                "condition_to_create": "SGA_TOV >= 4 and SGA_TS_PCT < 0.58",
                "player_or_combo": row.player_name,
                "score": row.sga_pressure_score,
                "reason": _reason_from_row(pd.Series(row._asdict()), "sga_pressure_score"),
                "tactical_role": role,
                "related_title_probability_gain": SCENARIO_GAINS["OKC-1 SGA pressure and efficiency reduction"],
            }
        )

    for idx, row in enumerate(bench_df.head(5).itertuples(index=False), start=1):
        rows.append(
            {
                "opponent": "OKC",
                "scenario": "OKC-2 Bench survival",
                "condition_to_create": "OKC_BENCH_POINTS < team median and OKC_BENCH_MARGIN < 0",
                "player_or_combo": row.combo,
                "score": row.okc_bench_survival_score,
                "reason": _reason_from_row(pd.Series(row._asdict()), "okc_bench_survival_score"),
                "tactical_role": "bench stabilizer" if idx <= 2 else "non-starter minutes support",
                "related_title_probability_gain": SCENARIO_GAINS["OKC-2 Bench survival"],
            }
        )

    for idx, row in enumerate(three_pt_df.head(5).itertuples(index=False), start=1):
        rows.append(
            {
                "opponent": "SAS",
                "scenario": "SAS-1 Wembanyama spacing neutralization",
                "condition_to_create": "increase opponent 3P pressure and pull Wembanyama from the rim",
                "player_or_combo": row.player_name,
                "score": row.three_point_pressure_score,
                "reason": _reason_from_row(pd.Series(row._asdict()), "three_point_pressure_score"),
                "tactical_role": "high-volume spacer" if idx <= 2 else "catch-and-shoot pressure",
                "related_title_probability_gain": SCENARIO_GAINS["SAS-1 Wembanyama spacing neutralization"],
            }
        )

    for idx, row in enumerate(spacing_df.head(5).itertuples(index=False), start=1):
        rows.append(
            {
                "opponent": "SAS",
                "scenario": "SAS-3 Bench and rebounding control",
                "condition_to_create": "win bench/rebounding minutes while keeping spacing",
                "player_or_combo": row.combo,
                "score": row.sas_spacing_rebounding_score,
                "reason": _reason_from_row(pd.Series(row._asdict()), "sas_spacing_rebounding_score"),
                "tactical_role": "spacing/rebounding combo" if idx <= 2 else "glass-control lineup",
                "related_title_probability_gain": SCENARIO_GAINS["SAS-3 Bench and rebounding control"],
            }
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
