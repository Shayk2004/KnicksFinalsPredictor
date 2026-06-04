"""Run Part 2A: Knicks loss decomposition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from part2a_analysis import (
    run_logistic_regression,
    run_loss_decomposition,
    run_random_forest,
)
from part2a_data import (
    RAW_LOGS_PATH,
    add_engineered_features,
    build_knicks_game_dataset,
    fetch_team_game_logs,
    save_part2a_dataset,
)
from part2a_plots import (
    plot_logistic_coefficients,
    plot_random_forest_importance,
    plot_top_loss_factors,
    plot_win_loss_comparison,
)


OUTPUT_DIR = Path("outputs")
LOSS_DECOMPOSITION_PATH = OUTPUT_DIR / "part2a_loss_decomposition.csv"
LOGISTIC_COEFFICIENTS_PATH = OUTPUT_DIR / "part2a_logistic_coefficients.csv"
RANDOM_FOREST_IMPORTANCE_PATH = OUTPUT_DIR / "part2a_random_forest_importance.csv"
MATCHUP_FILTER_PATH = OUTPUT_DIR / "part2a_knicks_vs_okc_sas.csv"

SEASONS = ["2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]
FEATURE_COLS = [
    "knicks_efg_pct",
    "opp_efg_pct",
    "knicks_fg3_pct",
    "opp_fg3_pct",
    "knicks_fg3a",
    "opp_fg3a",
    "knicks_tov",
    "opp_tov",
    "knicks_tov_margin",
    "knicks_oreb",
    "opp_oreb",
    "oreb_margin",
    "rebound_margin",
    "knicks_fta",
    "opp_fta",
    "free_throw_attempt_margin",
    "knicks_ast",
    "assist_turnover_ratio",
    "opp_assist_turnover_ratio",
    "knicks_pf",
    "opp_pf",
    "is_home",
    "game_type_binary",
]


def _load_or_fetch_team_logs() -> pd.DataFrame:
    if RAW_LOGS_PATH.exists():
        return pd.read_csv(RAW_LOGS_PATH)

    team_logs = fetch_team_game_logs(SEASONS, SEASON_TYPES)
    team_logs.to_csv(RAW_LOGS_PATH, index=False)
    return team_logs


def _suggest_what_if_scenarios(decomposition_df: pd.DataFrame) -> list[str]:
    scenario_map = {
        "knicks_tov": "Reduce Knicks turnovers",
        "knicks_tov_margin": "Reduce Knicks turnovers relative to opponents",
        "knicks_efg_pct": "Improve Knicks eFG%",
        "knicks_fg3_pct": "Improve Knicks 3P%",
        "opp_efg_pct": "Reduce opponent eFG%",
        "opp_fg3_pct": "Reduce opponent 3P%",
        "opp_oreb": "Reduce opponent offensive rebounds",
        "oreb_margin": "Improve offensive rebounding margin",
        "opp_fta": "Reduce opponent free throw attempts",
        "free_throw_attempt_margin": "Improve free throw attempt margin",
        "assist_turnover_ratio": "Improve Knicks assist/turnover ratio",
        "opp_assist_turnover_ratio": "Reduce opponent assist/turnover ratio",
    }
    suggestions = []
    for feature in decomposition_df["feature"]:
        if feature in scenario_map and scenario_map[feature] not in suggestions:
            suggestions.append(scenario_map[feature])
        if len(suggestions) == 6:
            break
    return suggestions


def _print_top_rows(title: str, df: pd.DataFrame, value_col: str, n: int = 10) -> None:
    print(title)
    for idx, row in enumerate(df.head(n).itertuples(index=False), start=1):
        print(f"{idx}. {row.feature}: {value_col}={getattr(row, value_col):.4f}")
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    team_logs = _load_or_fetch_team_logs()
    knicks_games = add_engineered_features(build_knicks_game_dataset(team_logs))
    save_part2a_dataset(knicks_games)

    matchup_df = knicks_games[knicks_games["opponent"].isin(["OKC", "SAS"])].copy()
    matchup_df.to_csv(MATCHUP_FILTER_PATH, index=False)

    available_features = [feature for feature in FEATURE_COLS if feature in knicks_games.columns]
    decomposition_df = run_loss_decomposition(knicks_games, available_features)
    decomposition_df.to_csv(LOSS_DECOMPOSITION_PATH, index=False)

    logistic_df, logistic_metrics = run_logistic_regression(knicks_games, available_features)
    logistic_df.to_csv(LOGISTIC_COEFFICIENTS_PATH, index=False)

    forest_df, forest_metrics = run_random_forest(knicks_games, available_features)
    forest_df.to_csv(RANDOM_FOREST_IMPORTANCE_PATH, index=False)

    plot_top_loss_factors(decomposition_df)
    plot_win_loss_comparison(decomposition_df)
    plot_logistic_coefficients(logistic_df)
    plot_random_forest_importance(forest_df)

    wins = int(knicks_games["knicks_win"].sum())
    losses = int(len(knicks_games) - wins)
    suggestions = _suggest_what_if_scenarios(decomposition_df)

    print("Part 2A Knicks Loss Decomposition")
    print("=================================")
    print(f"Knicks games: {len(knicks_games)}")
    print(f"Knicks wins: {wins}")
    print(f"Knicks losses: {losses}")
    print()

    print("Top Knicks loss indicators:")
    for idx, row in enumerate(decomposition_df.head(10).itertuples(index=False), start=1):
        direction = "higher" if row.difference > 0 else "lower"
        print(
            f"{idx}. {row.feature}: losses are {direction} by "
            f"{abs(row.difference):.4f}, Cohen's d = {row.cohens_d:.3f}"
        )
    print()

    print(
        "Logistic regression: "
        f"accuracy={logistic_metrics['accuracy']:.3f}, "
        f"ROC AUC={logistic_metrics['roc_auc']:.3f}"
    )
    _print_top_rows("Top logistic regression coefficients:", logistic_df, "coefficient", n=8)

    print(
        "Random forest: "
        f"accuracy={forest_metrics['accuracy']:.3f}, "
        f"ROC AUC={forest_metrics['roc_auc']:.3f}"
    )
    _print_top_rows("Top random forest importances:", forest_df, "importance", n=8)

    print("Suggested what-if scenario candidates:")
    for suggestion in suggestions:
        print(f"- {suggestion}")
    print()

    if len(matchup_df) < 10:
        print(
            "Direct Knicks vs OKC/SAS sample is small; use as descriptive context, "
            "not model training."
        )
    print(f"Knicks vs OKC/SAS rows saved: {len(matchup_df)}")


if __name__ == "__main__":
    main()
