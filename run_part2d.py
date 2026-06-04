"""Run Part 2D Knicks personnel and lineup actionability analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from part2d_analysis import (
    create_actionability_table,
    rank_okc_bench_survival_combos,
    rank_sas_spacing_rebounding_combos,
    rank_sas_three_point_pressure_candidates,
    rank_sga_pressure_candidates,
)
from part2d_data import (
    KNICKS_PLAYER_LOGS_PATH,
    RAW_PLAYER_LOGS_PATH,
    add_player_features,
    build_2man_combinations,
    build_3man_combinations,
    build_game_level_player_matrix,
    build_player_season_summary,
    clean_player_logs,
    fetch_knicks_player_game_logs,
    load_knicks_player_game_logs,
)
from part2d_plots import (
    plot_actionability_summary,
    plot_okc_bench_survival_combos,
    plot_sas_spacing_rebounding_combos,
    plot_sas_three_point_pressure_candidates,
    plot_sga_pressure_candidates,
)
from part2d_scores import (
    calculate_bench_stability_score,
    calculate_okc_bench_survival_score,
    calculate_rebounding_support_score,
    calculate_sga_pressure_score,
    calculate_spacing_rebounding_score,
    calculate_three_point_pressure_score,
)


OUTPUT_DIR = Path("outputs")
SEASONS = ["2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]
EXCLUDED_PLAYERS = {
    "Guerschon Yabusele",
    "Pacôme Dadiet",
    "Tyler Kolek",
}

PLAYER_SUMMARY_PATH = OUTPUT_DIR / "part2d_knicks_player_summary.csv"
SGA_CANDIDATES_PATH = OUTPUT_DIR / "part2d_sga_pressure_candidates.csv"
SAS_3PT_CANDIDATES_PATH = OUTPUT_DIR / "part2d_sas_three_point_pressure_candidates.csv"
OKC_BENCH_COMBOS_PATH = OUTPUT_DIR / "part2d_okc_bench_survival_combos.csv"
SAS_SPACING_COMBOS_PATH = OUTPUT_DIR / "part2d_sas_spacing_rebounding_combos.csv"
ACTIONABILITY_PATH = OUTPUT_DIR / "part2d_actionability_table.csv"
COMBO_SUMMARY_PATH = OUTPUT_DIR / "part2d_knicks_combo_summary.csv"


def _load_or_fetch_knicks_logs() -> pd.DataFrame:
    if RAW_PLAYER_LOGS_PATH.exists():
        return load_knicks_player_game_logs(RAW_PLAYER_LOGS_PATH)
    if KNICKS_PLAYER_LOGS_PATH.exists():
        return load_knicks_player_game_logs(KNICKS_PLAYER_LOGS_PATH)
    logs = fetch_knicks_player_game_logs(SEASONS, SEASON_TYPES)
    KNICKS_PLAYER_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    logs.to_csv(KNICKS_PLAYER_LOGS_PATH, index=False)
    return logs


def _merge_player_scores(player_summary: pd.DataFrame) -> pd.DataFrame:
    score_frames = [
        calculate_sga_pressure_score(player_summary)[["player_name", "sga_pressure_score"]],
        calculate_three_point_pressure_score(player_summary)[["player_name", "three_point_pressure_score"]],
        calculate_rebounding_support_score(player_summary)[["player_name", "rebounding_support_score"]],
        calculate_bench_stability_score(player_summary)[["player_name", "bench_stability_score"]],
    ]
    out = player_summary.copy()
    for frame in score_frames:
        out = out.merge(frame, on="player_name", how="left")
    return out


def _build_combo_scores(player_matrix: pd.DataFrame) -> pd.DataFrame:
    two_man = build_2man_combinations(player_matrix)
    three_man = build_3man_combinations(player_matrix)
    combo_summary = pd.concat([two_man, three_man], ignore_index=True)
    if combo_summary.empty:
        return combo_summary

    okc_scores = calculate_okc_bench_survival_score(combo_summary)[["combo", "okc_bench_survival_score"]]
    sas_scores = calculate_spacing_rebounding_score(combo_summary)[["combo", "sas_spacing_rebounding_score"]]
    combo_summary = combo_summary.merge(okc_scores, on="combo", how="left")
    combo_summary = combo_summary.merge(sas_scores, on="combo", how="left")
    return combo_summary


def _exclude_players(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["player_name"].isin(EXCLUDED_PLAYERS)].copy()


def _exclude_combos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pattern = "|".join(EXCLUDED_PLAYERS)
    return df[~df["combo"].str.contains(pattern, regex=True, na=False)].copy()


def _print_ranking(title: str, df: pd.DataFrame, label_col: str, score_col: str, n: int = 5) -> None:
    print(title)
    if df.empty:
        print("  No eligible rows.")
        print()
        return
    for idx, row in enumerate(df.head(n).itertuples(index=False), start=1):
        print(f"  {idx}. {getattr(row, label_col)}: {getattr(row, score_col):.3f}")
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_logs = _load_or_fetch_knicks_logs()
    player_logs = add_player_features(clean_player_logs(raw_logs))
    player_logs = _exclude_players(player_logs)
    player_matrix = build_game_level_player_matrix(player_logs)

    player_summary = _merge_player_scores(build_player_season_summary(player_logs))
    player_summary.to_csv(PLAYER_SUMMARY_PATH, index=False)

    combo_summary = _exclude_combos(_build_combo_scores(player_matrix))
    combo_summary.to_csv(COMBO_SUMMARY_PATH, index=False)

    sga_candidates = rank_sga_pressure_candidates(
        player_summary.dropna(subset=["sga_pressure_score"]).copy()
    )
    sas_3pt_candidates = rank_sas_three_point_pressure_candidates(
        player_summary.dropna(subset=["three_point_pressure_score"]).copy()
    )
    okc_bench_combos = rank_okc_bench_survival_combos(
        combo_summary.dropna(subset=["okc_bench_survival_score"]).copy()
    )
    sas_spacing_combos = rank_sas_spacing_rebounding_combos(
        combo_summary.dropna(subset=["sas_spacing_rebounding_score"]).copy()
    )

    actionability = create_actionability_table(
        sga_candidates,
        okc_bench_combos,
        sas_3pt_candidates,
        sas_spacing_combos,
    )

    sga_candidates.to_csv(SGA_CANDIDATES_PATH, index=False)
    sas_3pt_candidates.to_csv(SAS_3PT_CANDIDATES_PATH, index=False)
    okc_bench_combos.to_csv(OKC_BENCH_COMBOS_PATH, index=False)
    sas_spacing_combos.to_csv(SAS_SPACING_COMBOS_PATH, index=False)
    actionability.to_csv(ACTIONABILITY_PATH, index=False)

    plot_sga_pressure_candidates(sga_candidates)
    plot_sas_three_point_pressure_candidates(sas_3pt_candidates)
    plot_okc_bench_survival_combos(okc_bench_combos)
    plot_sas_spacing_rebounding_combos(sas_spacing_combos)
    plot_actionability_summary(actionability)

    low_sample_players = player_summary[player_summary["total_minutes"] < 300]["player_name"].tolist()

    print("Part 2D Knicks Personnel and Lineup Actionability")
    print("=================================================")
    print("These rankings are actionability proxies, not proof of causation.")
    print("Combination results use game-log co-appearance, not true possession-level lineup data.")
    print("Minutes reliability is included to reduce low-sample overrating.")
    print()
    print("Removed from Part 2D pools: " + ", ".join(sorted(EXCLUDED_PLAYERS)))
    print()
    if low_sample_players:
        print(
            "Excluded low-minute players from score rankings: "
            + ", ".join(low_sample_players[:12])
            + ("..." if len(low_sample_players) > 12 else "")
        )
        print()

    _print_ranking("Top SGA pressure candidates", sga_candidates, "player_name", "sga_pressure_score")
    _print_ranking("Top SAS 3PT pressure candidates", sas_3pt_candidates, "player_name", "three_point_pressure_score")
    _print_ranking("Top OKC bench survival combos", okc_bench_combos, "combo", "okc_bench_survival_score")
    _print_ranking("Top SAS spacing/rebounding combos", sas_spacing_combos, "combo", "sas_spacing_rebounding_score")

    print("Most actionable Knicks tactical groups")
    for idx, row in enumerate(actionability.head(10).itertuples(index=False), start=1):
        print(
            f"  {idx}. vs {row.opponent} ({row.scenario}): {row.player_or_combo} "
            f"as {row.tactical_role}, score={row.score:.3f}"
        )
    print()
    print(f"Saved player summary to {PLAYER_SUMMARY_PATH}")
    print(f"Saved actionability table to {ACTIONABILITY_PATH}")


if __name__ == "__main__":
    main()
