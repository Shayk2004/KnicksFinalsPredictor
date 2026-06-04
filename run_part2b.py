"""Run Part 2B: OKC and SAS vulnerabilities plus player-driver analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from part2b_analysis import (
    run_logistic_regression,
    run_player_driver_analysis,
    run_random_forest,
    run_team_vulnerability_analysis,
    run_threshold_analysis,
)
from part2b_data import (
    RAW_PLAYER_LOGS_PATH,
    RAW_TEAM_LOGS_PATH,
    add_player_engineered_features,
    add_team_engineered_features,
    build_player_game_dataset,
    build_team_game_dataset,
    create_bench_features,
    create_star_player_features,
    fetch_player_game_logs,
    fetch_team_game_logs,
)
from part2b_plots import (
    plot_player_driver_factors,
    plot_player_driver_factors_signed,
    plot_star_feature_win_loss_comparison,
    plot_team_vulnerability,
    plot_team_vulnerability_signed,
    plot_threshold_loss_rate_lift,
    plot_threshold_loss_rates,
)


OUTPUT_DIR = Path("outputs")
SEASONS = ["2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]

TEAM_FEATURES = [
    "team_efg_pct",
    "opp_efg_pct",
    "team_fg3_pct",
    "opp_fg3_pct",
    "team_fg3a",
    "opp_fg3a",
    "team_tov",
    "opp_tov",
    "tov_margin",
    "team_oreb",
    "opp_oreb",
    "oreb_margin",
    "rebound_margin",
    "team_fta",
    "opp_fta",
    "free_throw_attempt_margin",
    "team_ast",
    "assist_turnover_ratio",
    "opp_assist_turnover_ratio",
    "team_pf",
    "opp_pf",
    "is_home",
    "game_type_binary",
]

OKC_PLAYER_FEATURES = [
    "sga_points",
    "sga_player_ts_pct",
    "sga_player_efg_pct",
    "sga_fta",
    "sga_assists",
    "sga_turnovers",
    "sga_rebounds",
    "sga_plus_minus",
    "jalen_williams_points",
    "jalen_williams_player_ts_pct",
    "jalen_williams_player_efg_pct",
    "jalen_williams_assists",
    "jalen_williams_turnovers",
    "jalen_williams_plus_minus",
    "chet_points",
    "chet_rebounds",
    "chet_blocks",
    "chet_player_ts_pct",
    "chet_plus_minus",
    "team_bench_points",
    "team_bench_rebounds",
    "team_bench_assists",
    "team_bench_plus_minus",
    "opponent_bench_points",
    "bench_margin",
]

SAS_PLAYER_FEATURES = [
    "wembanyama_points",
    "wembanyama_player_ts_pct",
    "wembanyama_player_efg_pct",
    "wembanyama_fta",
    "wembanyama_rebounds",
    "wembanyama_blocks",
    "wembanyama_assists",
    "wembanyama_turnovers",
    "wembanyama_pf",
    "wembanyama_plus_minus",
    "team_bench_points",
    "team_bench_rebounds",
    "team_bench_assists",
    "team_bench_plus_minus",
    "opponent_bench_points",
    "bench_margin",
]


def _load_or_fetch_team_logs() -> pd.DataFrame:
    if RAW_TEAM_LOGS_PATH.exists():
        return pd.read_csv(RAW_TEAM_LOGS_PATH)
    logs = fetch_team_game_logs(SEASONS, SEASON_TYPES)
    logs.to_csv(RAW_TEAM_LOGS_PATH, index=False)
    return logs


def _load_or_fetch_player_logs() -> pd.DataFrame:
    if RAW_PLAYER_LOGS_PATH.exists():
        return pd.read_csv(RAW_PLAYER_LOGS_PATH)
    logs = fetch_player_game_logs(SEASONS, SEASON_TYPES)
    logs.to_csv(RAW_PLAYER_LOGS_PATH, index=False)
    return logs


def _combine_player_drivers(team_games: pd.DataFrame, player_games: pd.DataFrame, team: str) -> pd.DataFrame:
    star_df = create_star_player_features(team_games, player_games, team)
    bench_df = create_bench_features(team_games, player_games, team)
    bench_cols = [col for col in bench_df.columns if col != "team_win"]
    return star_df.merge(bench_df[bench_cols], on="game_id", how="left")


def _okc_thresholds(df: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {"name": "SGA_FTA < 8", "feature": "sga_fta", "op": "<", "value": 8},
        {"name": "SGA_TS_PCT < 0.58", "feature": "sga_player_ts_pct", "op": "<", "value": 0.58},
        {"name": "SGA_TOV >= 4", "feature": "sga_turnovers", "op": ">=", "value": 4},
        {"name": "SGA_PTS < 28", "feature": "sga_points", "op": "<", "value": 28},
        {"name": "OKC_BENCH_MARGIN < 0", "feature": "bench_margin", "op": "<", "value": 0},
        {"name": "OKC_BENCH_POINTS < team median", "feature": "team_bench_points", "op": "<", "value": "median"},
        {"name": "CHET_BLK <= 1", "feature": "chet_blocks", "op": "<=", "value": 1},
        {"name": "JALEN_WILLIAMS_PTS < team median", "feature": "jalen_williams_points", "op": "<", "value": "median"},
    ]


def _sas_thresholds(team_df: pd.DataFrame, player_df: pd.DataFrame) -> list[dict[str, object]]:
    merged = player_df.merge(
        team_df[["game_id", "opp_fg3a", "opp_fg3_pct"]],
        on="game_id",
        how="left",
    )
    player_df["opp_fg3a"] = merged["opp_fg3a"]
    player_df["opp_fg3_pct"] = merged["opp_fg3_pct"]
    return [
        {"name": "WEMBANYAMA_BLK <= 2", "feature": "wembanyama_blocks", "op": "<=", "value": 2},
        {"name": "WEMBANYAMA_REB < 10", "feature": "wembanyama_rebounds", "op": "<", "value": 10},
        {"name": "WEMBANYAMA_TS_PCT < 0.55", "feature": "wembanyama_player_ts_pct", "op": "<", "value": 0.55},
        {"name": "WEMBANYAMA_TOV >= 4", "feature": "wembanyama_turnovers", "op": ">=", "value": 4},
        {"name": "WEMBANYAMA_PF >= 4", "feature": "wembanyama_pf", "op": ">=", "value": 4},
        {"name": "SAS_BENCH_MARGIN < 0", "feature": "bench_margin", "op": "<", "value": 0},
        {"name": "SAS_BENCH_POINTS < team median", "feature": "team_bench_points", "op": "<", "value": "median"},
        {"name": "OPP_FG3A above opponent median", "feature": "opp_fg3a", "op": ">", "value": "median"},
        {"name": "OPP_FG3_PCT above opponent median", "feature": "opp_fg3_pct", "op": ">", "value": "median"},
    ]


def _actionability_table(
    okc_team: pd.DataFrame,
    sas_team: pd.DataFrame,
    okc_player: pd.DataFrame,
    sas_player: pd.DataFrame,
    okc_thresholds: pd.DataFrame,
    sas_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    minimum_loss_rate_gap = 0.05

    def add(finding: str, metric: str, opponent: str, score: int, tactic: str) -> None:
        rows.append(
            {
                "finding": finding,
                "evidence_metric": metric,
                "opponent": opponent,
                "knicks_actionability_score": score,
                "possible_knicks_tactic": tactic,
            }
        )

    okc_threshold_tactics = {
        "SGA_TS_PCT < 0.58": (
            "OKC loss rate rises when SGA efficiency is pushed down",
            5,
            "Show early help, crowd driving gaps, force contested pull-up jumpers",
        ),
        "SGA_TOV >= 4": (
            "OKC is vulnerable when SGA has a high-turnover game",
            4,
            "Send selective digs, load up at the elbows, rotate out of help with discipline",
        ),
        "OKC_BENCH_POINTS < team median": (
            "OKC is vulnerable when bench scoring is muted",
            4,
            "Stagger Brunson/Towns, avoid all-bench units, shorten rotation",
        ),
        "OKC_BENCH_MARGIN < 0": (
            "OKC is vulnerable when its bench loses the minutes battle",
            4,
            "Stagger Brunson/Towns, attack second-unit matchups, keep one creator on court",
        ),
        "SGA_FTA < 8": (
            "OKC loss rate rises when SGA free throws are held down",
            5,
            "Defend without fouling, show early help, force pull-up jumpers",
        ),
    }
    for threshold, (finding, score, tactic) in okc_threshold_tactics.items():
        if threshold in set(okc_thresholds["threshold"]):
            row = okc_thresholds[okc_thresholds["threshold"] == threshold].iloc[0]
            if row["loss_rate_lift"] >= minimum_loss_rate_gap:
                add(finding, f"loss-rate lift {row['loss_rate_lift']:.3f}", "OKC", score, tactic)

    if "tov_margin" in set(okc_team["feature"]):
        row = okc_team[okc_team["feature"] == "tov_margin"].iloc[0]
        add(
            "OKC losses are associated with a worse turnover margin",
            f"Cohen's d {row['cohens_d']:.3f}",
            "OKC",
            4,
            "Reduce live-ball passes and use Brunson/Towns as primary decision hubs",
        )

    sas_threshold_tactics = {
        "WEMBANYAMA_TS_PCT < 0.55": (
            "SAS loss rate rises when Wembanyama scoring efficiency is reduced",
            5,
            "Use Towns pick-and-pop, early post help, and mixed coverages to pull him from comfort zones",
        ),
        "OPP_FG3_PCT above opponent median": (
            "SAS losses are tied to opponents shooting well from three",
            4,
            "Create corner threes through Brunson paint touches and Towns spacing",
        ),
        "WEMBANYAMA_TOV >= 4": (
            "SAS is vulnerable when Wembanyama has a high-turnover game",
            4,
            "Dig at the ball, crowd catches, and make him pass through traffic",
        ),
        "SAS_BENCH_POINTS < team median": (
            "SAS is vulnerable when bench scoring is muted",
            3,
            "Win the second-unit minutes with steady creator staggering",
        ),
        "SAS_BENCH_MARGIN < 0": (
            "SAS is vulnerable when its bench loses the minutes battle",
            3,
            "Attack bench matchups and keep spacing-heavy units on the floor",
        ),
        "WEMBANYAMA_BLK <= 2": (
            "SAS loss rate rises when Wembanyama block impact is muted",
            4,
            "Use Towns pick-and-pop to pull Wembanyama away from the rim",
        ),
        "WEMBANYAMA_PF >= 4": (
            "SAS becomes more fragile when Wembanyama is in foul stress",
            4,
            "Attack his body selectively, use pump fakes, force defensive decisions",
        ),
    }
    for threshold, (finding, score, tactic) in sas_threshold_tactics.items():
        if threshold in set(sas_thresholds["threshold"]):
            row = sas_thresholds[sas_thresholds["threshold"] == threshold].iloc[0]
            if row["loss_rate_lift"] >= minimum_loss_rate_gap:
                add(finding, f"loss-rate lift {row['loss_rate_lift']:.3f}", "SAS", score, tactic)

    return pd.DataFrame(rows)


def _save_outputs(prefix: str, team_games: pd.DataFrame, player_drivers: pd.DataFrame, team_vuln: pd.DataFrame, player_vuln: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    team_games.to_csv(OUTPUT_DIR / f"part2b_{prefix}_team_games.csv", index=False)
    player_drivers.to_csv(OUTPUT_DIR / f"part2b_{prefix}_player_drivers.csv", index=False)
    team_vuln.to_csv(OUTPUT_DIR / f"part2b_{prefix}_team_vulnerabilities.csv", index=False)
    player_vuln.to_csv(OUTPUT_DIR / f"part2b_{prefix}_player_loss_drivers.csv", index=False)
    thresholds.to_csv(OUTPUT_DIR / f"part2b_{prefix}_thresholds.csv", index=False)


def _print_findings(team: str, team_vuln: pd.DataFrame, player_vuln: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    print(f"{team} signed team-level vulnerabilities")
    for idx, row in enumerate(team_vuln.head(6).itertuples(index=False), start=1):
        direction = "higher" if row.cohens_d > 0 else "lower" if row.cohens_d < 0 else "unchanged"
        print(
            f"{idx}. {row.feature}: {direction} in losses, "
            f"d={row.cohens_d:.3f}, diff={row.difference:.4f}"
        )
    print()

    print(f"{team} signed player-level loss drivers")
    for idx, row in enumerate(player_vuln.head(6).itertuples(index=False), start=1):
        direction = "higher" if row.cohens_d > 0 else "lower" if row.cohens_d < 0 else "unchanged"
        print(
            f"{idx}. {row.feature}: {direction} in losses, "
            f"d={row.cohens_d:.3f}, diff={row.difference:.4f}"
        )
    print()

    print(f"{team} threshold conditions ranked by loss-rate lift")
    for idx, row in enumerate(thresholds.head(5).itertuples(index=False), start=1):
        print(
            f"{idx}. {row.condition_name}: true games={row.games_condition_true}, "
            f"loss-rate lift={row.loss_rate_lift:.3f}, "
            f"true loss rate={row.team_loss_rate_when_true:.3f}"
        )
    print()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    team_logs = _load_or_fetch_team_logs()
    player_logs = _load_or_fetch_player_logs()

    okc_team = add_team_engineered_features(build_team_game_dataset(team_logs, "OKC"))
    sas_team = add_team_engineered_features(build_team_game_dataset(team_logs, "SAS"))

    okc_players = add_player_engineered_features(build_player_game_dataset(player_logs, "OKC"))
    sas_players = add_player_engineered_features(build_player_game_dataset(player_logs, "SAS"))
    okc_drivers = _combine_player_drivers(okc_team, okc_players, "OKC")
    sas_drivers = _combine_player_drivers(sas_team, sas_players, "SAS")

    okc_team_vuln = run_team_vulnerability_analysis(okc_team, TEAM_FEATURES)
    sas_team_vuln = run_team_vulnerability_analysis(sas_team, TEAM_FEATURES)
    okc_player_vuln = run_player_driver_analysis(okc_drivers, OKC_PLAYER_FEATURES)
    sas_player_vuln = run_player_driver_analysis(sas_drivers, SAS_PLAYER_FEATURES)

    okc_threshold_df = run_threshold_analysis(okc_drivers, _okc_thresholds(okc_drivers))
    sas_threshold_df = run_threshold_analysis(sas_drivers, _sas_thresholds(sas_team, sas_drivers))

    _save_outputs("okc", okc_team, okc_drivers, okc_team_vuln, okc_player_vuln, okc_threshold_df)
    _save_outputs("sas", sas_team, sas_drivers, sas_team_vuln, sas_player_vuln, sas_threshold_df)

    actionability = _actionability_table(
        okc_team_vuln,
        sas_team_vuln,
        okc_player_vuln,
        sas_player_vuln,
        okc_threshold_df,
        sas_threshold_df,
    )
    actionability.to_csv(OUTPUT_DIR / "part2b_actionability_table.csv", index=False)

    plot_team_vulnerability(okc_team_vuln, "OKC")
    plot_team_vulnerability(sas_team_vuln, "SAS")
    plot_team_vulnerability_signed(okc_team_vuln, "OKC")
    plot_team_vulnerability_signed(sas_team_vuln, "SAS")
    plot_player_driver_factors(okc_player_vuln, "OKC")
    plot_player_driver_factors(sas_player_vuln, "SAS")
    plot_player_driver_factors_signed(okc_player_vuln, "OKC")
    plot_player_driver_factors_signed(sas_player_vuln, "SAS")
    plot_threshold_loss_rates(okc_threshold_df, "OKC")
    plot_threshold_loss_rates(sas_threshold_df, "SAS")
    plot_threshold_loss_rate_lift(okc_threshold_df, "OKC")
    plot_threshold_loss_rate_lift(sas_threshold_df, "SAS")
    plot_star_feature_win_loss_comparison(
        okc_drivers,
        "OKC",
        ["sga_points", "sga_player_ts_pct", "sga_fta", "sga_turnovers", "team_bench_points", "bench_margin"],
    )
    plot_star_feature_win_loss_comparison(
        sas_drivers,
        "SAS",
        ["wembanyama_points", "wembanyama_player_ts_pct", "wembanyama_blocks", "wembanyama_pf", "team_bench_points", "bench_margin"],
    )

    okc_logit, okc_logit_metrics = run_logistic_regression(okc_team, TEAM_FEATURES)
    sas_logit, sas_logit_metrics = run_logistic_regression(sas_team, TEAM_FEATURES)
    okc_rf, okc_rf_metrics = run_random_forest(okc_team, TEAM_FEATURES)
    sas_rf, sas_rf_metrics = run_random_forest(sas_team, TEAM_FEATURES)
    okc_logit.to_csv(OUTPUT_DIR / "part2b_okc_team_logistic_coefficients.csv", index=False)
    sas_logit.to_csv(OUTPUT_DIR / "part2b_sas_team_logistic_coefficients.csv", index=False)
    okc_rf.to_csv(OUTPUT_DIR / "part2b_okc_team_random_forest_importance.csv", index=False)
    sas_rf.to_csv(OUTPUT_DIR / "part2b_sas_team_random_forest_importance.csv", index=False)

    print("Part 2B Opponent Vulnerability + Player Driver Analysis")
    print("======================================================")
    print("These are win/loss associations, not causal proof.")
    print()
    _print_findings("OKC", okc_team_vuln, okc_player_vuln, okc_threshold_df)
    _print_findings("SAS", sas_team_vuln, sas_player_vuln, sas_threshold_df)

    print("Secondary validation model metrics")
    print(f"OKC logistic: accuracy={okc_logit_metrics['accuracy']:.3f}, ROC AUC={okc_logit_metrics['roc_auc']:.3f}")
    print(f"SAS logistic: accuracy={sas_logit_metrics['accuracy']:.3f}, ROC AUC={sas_logit_metrics['roc_auc']:.3f}")
    print(f"OKC random forest: accuracy={okc_rf_metrics['accuracy']:.3f}, ROC AUC={okc_rf_metrics['roc_auc']:.3f}")
    print(f"SAS random forest: accuracy={sas_rf_metrics['accuracy']:.3f}, ROC AUC={sas_rf_metrics['roc_auc']:.3f}")
    print()

    print("Most actionable Knicks tactics")
    for idx, row in enumerate(actionability.itertuples(index=False), start=1):
        print(
            f"{idx}. vs {row.opponent}: {row.finding} "
            f"({row.evidence_metric}) -> {row.possible_knicks_tactic}"
        )


if __name__ == "__main__":
    main()
