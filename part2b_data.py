"""Data preparation for Part 2B opponent vulnerability and player drivers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


RAW_TEAM_LOGS_PATH = Path("raw_nba_team_game_logs.csv")
RAW_PLAYER_LOGS_PATH = Path("raw_nba_player_game_logs.csv")

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

PLAYER_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "SEASON_TYPE",
    "TEAM_ABBREVIATION",
    "MATCHUP",
    "PLAYER_NAME",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "PF",
    "PLUS_MINUS",
]


def fetch_team_game_logs(seasons: list[str], season_types: list[str]) -> pd.DataFrame:
    """Fetch team-level LeagueGameLog rows from nba_api."""
    return _fetch_league_game_logs(seasons, season_types, player_or_team="T")


def fetch_player_game_logs(seasons: list[str], season_types: list[str]) -> pd.DataFrame:
    """Fetch player-level LeagueGameLog rows from nba_api."""
    return _fetch_league_game_logs(seasons, season_types, player_or_team="P")


def _fetch_league_game_logs(
    seasons: list[str], season_types: list[str], player_or_team: str
) -> pd.DataFrame:
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError as exc:
        raise ImportError("nba_api is required to fetch fresh LeagueGameLog data.") from exc

    frames = []
    for season in seasons:
        for season_type in season_types:
            response = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation=player_or_team,
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
        raise RuntimeError("No LeagueGameLog data was fetched.")
    return pd.concat(frames, ignore_index=True)


def _require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def _prefixed_stats(row: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{column.lower()}": row[column]
        for column in BOXSCORE_COLUMNS
        if column in row.index
    }


def _game_type(season_type: object) -> str:
    return "playoffs" if str(season_type).strip().lower() == "playoffs" else "regular_season"


def _minutes_to_float(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value)
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60
    return float(value)


def _normalize_game_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        normalized = text.lstrip("0")
        return normalized if normalized else "0"
    return text


def build_team_game_dataset(team_logs: pd.DataFrame, target_team: str) -> pd.DataFrame:
    """Build one row per game from target_team's perspective."""
    required_cols = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "SEASON_TYPE",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        *BOXSCORE_COLUMNS,
    ]
    _require_columns(team_logs, required_cols)

    df = team_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)

    rows = []
    for game_id, group in df.groupby("GAME_ID"):
        if target_team not in set(group["TEAM_ABBREVIATION"]):
            continue
        team_row = group[group["TEAM_ABBREVIATION"] == target_team].iloc[0]
        opponent_rows = group[group["TEAM_ABBREVIATION"] != target_team]
        if opponent_rows.empty:
            continue
        opp_row = opponent_rows.iloc[0]

        team_score = float(team_row["PTS"])
        opponent_score = float(opp_row["PTS"])
        row = {
            "game_id": _normalize_game_id(game_id),
            "date": team_row["GAME_DATE"],
            "season": team_row["SEASON"],
            "game_type": _game_type(team_row["SEASON_TYPE"]),
            "team": target_team,
            "opponent": opp_row["TEAM_ABBREVIATION"],
            "is_home": bool(team_row["is_home"]),
            "team_score": team_score,
            "opponent_score": opponent_score,
            "margin": team_score - opponent_score,
            "team_win": int(team_score > opponent_score),
        }
        row.update(_prefixed_stats(team_row, "team"))
        row.update(_prefixed_stats(opp_row, "opp"))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No games found for {target_team}.")
    return out.sort_values(["date", "season"]).reset_index(drop=True)


def build_player_game_dataset(player_logs: pd.DataFrame, target_team: str) -> pd.DataFrame:
    """Build player-game rows for every game involving target_team."""
    available_required = [col for col in PLAYER_COLUMNS if col in player_logs.columns]
    _require_columns(player_logs, ["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "TEAM_ABBREVIATION", "MATCHUP", "PLAYER_NAME"])

    df = player_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    target_game_ids = set(df.loc[df["TEAM_ABBREVIATION"] == target_team, "GAME_ID"])
    df = df[df["GAME_ID"].isin(target_game_ids)].copy()
    df["opponent"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+([A-Z]{2,3})", expand=False)

    rename_map = {
        "GAME_ID": "game_id",
        "GAME_DATE": "date",
        "SEASON": "season",
        "SEASON_TYPE": "game_type",
        "TEAM_ABBREVIATION": "team",
        "PLAYER_NAME": "player_name",
        "MIN": "minutes",
        "PTS": "points",
        "REB": "rebounds",
        "AST": "assists",
        "TOV": "turnovers",
        "STL": "steals",
        "BLK": "blocks",
        "FGM": "fgm",
        "FGA": "fga",
        "FG3M": "fg3m",
        "FG3A": "fg3a",
        "FTM": "ftm",
        "FTA": "fta",
        "PF": "pf",
        "PLUS_MINUS": "plus_minus",
        "START_POSITION": "starter",
    }
    keep_cols = [col for col in available_required + ["START_POSITION", "opponent"] if col in df.columns]
    out = df[keep_cols].rename(columns=rename_map)
    out["game_id"] = out["game_id"].map(_normalize_game_id)
    out["game_type"] = out["game_type"].map(_game_type)
    if "minutes" in out.columns:
        out["minutes"] = out["minutes"].map(_minutes_to_float)
    if "starter" in out.columns:
        out["starter"] = out["starter"].fillna("").astype(str).str.strip().ne("").astype(int)
    return out.sort_values(["date", "game_id", "team", "minutes"]).reset_index(drop=True)


def add_team_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team-level vulnerability features."""
    out = df.copy()
    out["team_efg_pct"] = (out["team_fgm"] + 0.5 * out["team_fg3m"]) / out["team_fga"]
    out["opp_efg_pct"] = (out["opp_fgm"] + 0.5 * out["opp_fg3m"]) / out["opp_fga"]
    out["tov_margin"] = out["team_tov"] - out["opp_tov"]
    out["rebound_margin"] = out["team_reb"] - out["opp_reb"]
    out["oreb_margin"] = out["team_oreb"] - out["opp_oreb"]
    out["three_point_attempt_margin"] = out["team_fg3a"] - out["opp_fg3a"]
    out["free_throw_attempt_margin"] = out["team_fta"] - out["opp_fta"]
    out["assist_turnover_ratio"] = out["team_ast"] / np.maximum(out["team_tov"], 1)
    out["opp_assist_turnover_ratio"] = out["opp_ast"] / np.maximum(out["opp_tov"], 1)
    out["is_home"] = out["is_home"].astype(int)
    out["game_type_binary"] = (out["game_type"] == "playoffs").astype(int)
    return out


def add_player_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add player shooting and creation efficiency features."""
    out = df.copy()
    out["player_efg_pct"] = (out["fgm"] + 0.5 * out["fg3m"]) / out["fga"].replace(0, np.nan)
    out["player_ts_pct"] = out["points"] / (2 * (out["fga"] + 0.44 * out["fta"]).replace(0, np.nan))
    out["ast_tov_ratio"] = out["assists"] / np.maximum(out["turnovers"], 1)
    return out


def _find_player_rows(player_games_df: pd.DataFrame, team: str, name_parts: list[str]) -> pd.DataFrame:
    mask = player_games_df["team"].eq(team)
    for part in name_parts:
        mask &= player_games_df["player_name"].str.contains(part, case=False, na=False)
    return player_games_df[mask].copy()


def _player_feature_frame(player_rows: pd.DataFrame, prefix: str, features: list[str]) -> pd.DataFrame:
    if player_rows.empty:
        return pd.DataFrame(columns=["game_id"])
    cols = ["game_id", *[feature for feature in features if feature in player_rows.columns]]
    frame = player_rows[cols].copy()
    return frame.rename(columns={feature: f"{prefix}_{feature}" for feature in cols if feature != "game_id"})


def create_star_player_features(
    team_games_df: pd.DataFrame, player_games_df: pd.DataFrame, target_team: str
) -> pd.DataFrame:
    """Create game-level star player features for OKC or SAS."""
    base = team_games_df[["game_id", "date", "season", "game_type", "team", "opponent", "team_win"]].copy()
    star_specs = {}
    if target_team == "OKC":
        star_specs = {
            "sga": (["Shai"], ["points", "player_ts_pct", "player_efg_pct", "fta", "assists", "turnovers", "rebounds", "plus_minus"]),
            "jalen_williams": (["Jalen", "Williams"], ["points", "player_ts_pct", "player_efg_pct", "assists", "turnovers", "plus_minus"]),
            "chet": (["Chet"], ["points", "rebounds", "blocks", "player_ts_pct", "plus_minus"]),
        }
    elif target_team == "SAS":
        star_specs = {
            "wembanyama": (["Wembanyama"], ["points", "player_ts_pct", "player_efg_pct", "fta", "rebounds", "blocks", "assists", "turnovers", "pf", "plus_minus"]),
        }

    out = base
    for prefix, (name_parts, features) in star_specs.items():
        rows = _find_player_rows(player_games_df, target_team, name_parts)
        frame = _player_feature_frame(rows, prefix, features)
        out = out.merge(frame, on="game_id", how="left")
    return out


def _bench_by_game(player_games_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    has_starter = "starter" in player_games_df.columns and player_games_df["starter"].notna().any()
    for (game_id, team), group in player_games_df.groupby(["game_id", "team"]):
        if has_starter and "starter" in group.columns:
            bench = group[group["starter"] == 0]
        else:
            bench = group.sort_values("minutes", ascending=False).iloc[5:]
        rows.append(
            {
                "game_id": game_id,
                "team": team,
                "bench_points": bench["points"].sum(),
                "bench_rebounds": bench["rebounds"].sum(),
                "bench_assists": bench["assists"].sum(),
                "bench_plus_minus": bench["plus_minus"].sum(),
            }
        )
    return pd.DataFrame(rows)


def create_bench_features(
    team_games_df: pd.DataFrame, player_games_df: pd.DataFrame, target_team: str
) -> pd.DataFrame:
    """Create target and opponent bench production by game."""
    bench = _bench_by_game(player_games_df)
    team_bench = bench[bench["team"] == target_team].drop(columns=["team"]).copy()
    team_bench = team_bench.rename(columns={col: f"team_{col}" for col in team_bench.columns if col != "game_id"})

    opponent_rows = []
    for row in team_games_df[["game_id", "opponent"]].itertuples(index=False):
        opp_bench = bench[(bench["game_id"] == row.game_id) & (bench["team"] == row.opponent)]
        if opp_bench.empty:
            continue
        record = opp_bench.iloc[0].to_dict()
        opponent_rows.append(
            {
                "game_id": row.game_id,
                "opponent_bench_points": record["bench_points"],
                "opponent_bench_rebounds": record["bench_rebounds"],
                "opponent_bench_assists": record["bench_assists"],
                "opponent_bench_plus_minus": record["bench_plus_minus"],
            }
        )
    opp_bench = pd.DataFrame(opponent_rows)

    out = team_games_df[["game_id", "team_win"]].merge(team_bench, on="game_id", how="left")
    out = out.merge(opp_bench, on="game_id", how="left")
    out["bench_margin"] = out["team_bench_points"] - out["opponent_bench_points"]
    return out
