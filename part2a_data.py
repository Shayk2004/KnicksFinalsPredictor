"""Data preparation for Part 2A Knicks loss decomposition."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


RAW_LOGS_PATH = Path("raw_nba_team_game_logs.csv")
PART2A_DATASET_PATH = Path("outputs/part2a_knicks_games.csv")
BOXSCORE_COLUMNS = [
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "PF",
    "PTS",
]


def fetch_team_game_logs(seasons: list[str], season_types: list[str]) -> pd.DataFrame:
    """Fetch team-level LeagueGameLog rows with nba_api."""
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError as exc:
        raise ImportError(
            "nba_api is required to fetch fresh team logs. Install it or provide "
            "raw_nba_team_game_logs.csv."
        ) from exc

    frames = []
    for season in seasons:
        for season_type in season_types:
            response = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
                sorter="DATE",
                direction="ASC",
                timeout=60,
            )
            df = response.get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            frames.append(df)
            time.sleep(1)

    if not frames:
        raise RuntimeError("No team game logs were fetched.")
    return pd.concat(frames, ignore_index=True)


def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError("Missing required team log columns: " + ", ".join(missing))


def _prefixed_stats(row: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{column.lower()}": row[column]
        for column in BOXSCORE_COLUMNS
        if column in row.index
    }


def build_knicks_game_dataset(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Convert team-level logs into one row per Knicks game."""
    required_cols = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "SEASON_TYPE",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "WL",
        *BOXSCORE_COLUMNS,
    ]
    _require_columns(team_logs, required_cols)

    df = team_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)

    rows = []
    for _, group in df.groupby("GAME_ID"):
        if "NYK" not in set(group["TEAM_ABBREVIATION"]):
            continue
        if len(group) < 2:
            continue

        knicks_row = group[group["TEAM_ABBREVIATION"] == "NYK"].iloc[0]
        opponent_rows = group[group["TEAM_ABBREVIATION"] != "NYK"]
        if opponent_rows.empty:
            continue
        opponent_row = opponent_rows.iloc[0]

        game_type = (
            "playoffs"
            if str(knicks_row["SEASON_TYPE"]).strip().lower() == "playoffs"
            else "regular_season"
        )
        knicks_score = float(knicks_row["PTS"])
        opponent_score = float(opponent_row["PTS"])

        row = {
            "date": knicks_row["GAME_DATE"],
            "season": knicks_row["SEASON"],
            "game_type": game_type,
            "opponent": opponent_row["TEAM_ABBREVIATION"],
            "is_home": bool(knicks_row["is_home"]),
            "knicks_score": knicks_score,
            "opponent_score": opponent_score,
            "margin": knicks_score - opponent_score,
            "knicks_win": int(knicks_score > opponent_score),
        }
        row.update(_prefixed_stats(knicks_row, "knicks"))
        row.update(_prefixed_stats(opponent_row, "opp"))
        rows.append(row)

    games = pd.DataFrame(rows)
    if games.empty:
        raise ValueError("No Knicks games could be built from team logs.")

    games = games.sort_values(["date", "season"]).reset_index(drop=True)
    return games


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the loss decomposition and models."""
    out = df.copy()
    out["knicks_efg_pct"] = (out["knicks_fgm"] + 0.5 * out["knicks_fg3m"]) / out["knicks_fga"]
    out["opp_efg_pct"] = (out["opp_fgm"] + 0.5 * out["opp_fg3m"]) / out["opp_fga"]
    out["knicks_tov_margin"] = out["knicks_tov"] - out["opp_tov"]
    out["rebound_margin"] = out["knicks_reb"] - out["opp_reb"]
    out["oreb_margin"] = out["knicks_oreb"] - out["opp_oreb"]
    out["three_point_attempt_margin"] = out["knicks_fg3a"] - out["opp_fg3a"]
    out["free_throw_attempt_margin"] = out["knicks_fta"] - out["opp_fta"]
    out["assist_turnover_ratio"] = out["knicks_ast"] / np.maximum(out["knicks_tov"], 1)
    out["opp_assist_turnover_ratio"] = out["opp_ast"] / np.maximum(out["opp_tov"], 1)
    out["point_margin"] = out["knicks_score"] - out["opponent_score"]
    out["game_type_binary"] = (out["game_type"] == "playoffs").astype(int)
    out["is_home"] = out["is_home"].astype(int)
    return out


def save_part2a_dataset(df: pd.DataFrame) -> None:
    """Save the Part 2A Knicks game dataset."""
    PART2A_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PART2A_DATASET_PATH, index=False)
