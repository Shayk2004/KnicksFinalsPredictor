"""Data preparation for Part 2D Knicks personnel and combo actionability."""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PLAYER_LOGS_PATH = Path("raw_nba_player_game_logs.csv")
KNICKS_PLAYER_LOGS_PATH = Path("outputs/part2d_knicks_player_logs.csv")
MIN_PLAYER_MINUTES = 300


def fetch_knicks_player_game_logs(seasons: list[str], season_types: list[str]) -> pd.DataFrame:
    """Fetch Knicks player game logs from nba_api LeagueGameLog."""
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError as exc:
        raise ImportError("nba_api is required to fetch Knicks player game logs.") from exc

    frames = []
    for season in seasons:
        for season_type in season_types:
            response = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                sorter="DATE",
                direction="ASC",
                timeout=60,
            )
            df = response.get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            frames.append(df[df["TEAM_ABBREVIATION"] == "NYK"].copy())
            time.sleep(1)

    if not frames:
        raise RuntimeError("No Knicks player logs were fetched.")
    return pd.concat(frames, ignore_index=True)


def load_knicks_player_game_logs(filepath: str | Path) -> pd.DataFrame:
    """Load player game logs and keep Knicks rows."""
    df = pd.read_csv(filepath)
    if "TEAM_ABBREVIATION" in df.columns:
        df = df[df["TEAM_ABBREVIATION"] == "NYK"].copy()
    elif "team" in df.columns:
        df = df[df["team"] == "NYK"].copy()
    if df.empty:
        raise ValueError(f"No Knicks player rows found in {filepath}.")
    return df


def _normalize_game_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        normalized = text.lstrip("0")
        return normalized if normalized else "0"
    return text


def _minutes_to_float(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) + float(seconds) / 60
    return float(text)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def clean_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Knicks player logs to the Part 2D schema."""
    rename_map = {
        "GAME_ID": "game_id",
        "GAME_DATE": "date",
        "SEASON": "season",
        "SEASON_TYPE": "game_type",
        "TEAM_ABBREVIATION": "team",
        "MATCHUP": "matchup",
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
        "PF": "personal_fouls",
        "PLUS_MINUS": "plus_minus",
    }
    out = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()
    required = [
        "game_id",
        "date",
        "season",
        "game_type",
        "team",
        "matchup",
        "player_name",
        "minutes",
        "points",
        "rebounds",
        "assists",
        "turnovers",
        "steals",
        "blocks",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "personal_fouls",
        "plus_minus",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError("Missing required Knicks player log columns: " + ", ".join(missing))

    out = out[required].copy()
    out["game_id"] = out["game_id"].map(_normalize_game_id)
    out["date"] = pd.to_datetime(out["date"])
    out["game_type"] = out["game_type"].map(
        lambda value: "playoffs" if str(value).lower() == "playoffs" else "regular_season"
    )
    out["opponent"] = out["matchup"].str.extract(r"(?:vs\.|@)\s+([A-Z]{2,3})", expand=False)
    out["minutes_float"] = out["minutes"].map(_minutes_to_float)

    numeric_cols = [
        "points",
        "rebounds",
        "assists",
        "turnovers",
        "steals",
        "blocks",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "personal_fouls",
        "plus_minus",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out.sort_values(["date", "game_id", "player_name"]).reset_index(drop=True)


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-36 and efficiency features."""
    out = df.copy()
    minutes = out["minutes_float"].replace(0, np.nan)
    for raw_col, feature_col in [
        ("points", "points_per_36"),
        ("rebounds", "rebounds_per_36"),
        ("assists", "assists_per_36"),
        ("turnovers", "turnovers_per_36"),
        ("steals", "steals_per_36"),
        ("blocks", "blocks_per_36"),
        ("personal_fouls", "fouls_per_36"),
        ("fg3a", "fg3a_per_36"),
        ("fta", "fta_per_36"),
        ("plus_minus", "plus_minus_per_36"),
    ]:
        out[feature_col] = out[raw_col] * 36 / minutes

    out["efg_pct"] = _safe_divide(out["fgm"] + 0.5 * out["fg3m"], out["fga"])
    out["ts_pct"] = _safe_divide(out["points"], 2 * (out["fga"] + 0.44 * out["fta"]))
    out["ast_tov_ratio"] = out["assists"] / np.maximum(out["turnovers"], 1)
    out["three_point_attempt_rate"] = out["fg3a"] / np.maximum(out["fga"], 1)
    return out


def build_player_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Knicks player logs into one row per player."""
    rows = []
    for player, group in df.groupby("player_name"):
        total_minutes = group["minutes_float"].sum()
        games = group["game_id"].nunique()
        row = {
            "player_name": player,
            "games": games,
            "total_minutes": total_minutes,
            "rotation_included": bool(total_minutes >= MIN_PLAYER_MINUTES),
        }
        for col in [
            "points",
            "rebounds",
            "assists",
            "turnovers",
            "steals",
            "blocks",
            "fgm",
            "fga",
            "fg3m",
            "fg3a",
            "ftm",
            "fta",
            "personal_fouls",
            "plus_minus",
        ]:
            row[col] = group[col].sum()

        minute_denom = total_minutes if total_minutes > 0 else np.nan
        row["points_per_36"] = row["points"] * 36 / minute_denom
        row["rebounds_per_36"] = row["rebounds"] * 36 / minute_denom
        row["assists_per_36"] = row["assists"] * 36 / minute_denom
        row["turnovers_per_36"] = row["turnovers"] * 36 / minute_denom
        row["steals_per_36"] = row["steals"] * 36 / minute_denom
        row["blocks_per_36"] = row["blocks"] * 36 / minute_denom
        row["fouls_per_36"] = row["personal_fouls"] * 36 / minute_denom
        row["fg3a_per_36"] = row["fg3a"] * 36 / minute_denom
        row["fta_per_36"] = row["fta"] * 36 / minute_denom
        row["plus_minus_per_36"] = row["plus_minus"] * 36 / minute_denom
        row["efg_pct"] = (row["fgm"] + 0.5 * row["fg3m"]) / row["fga"] if row["fga"] else np.nan
        row["fg3_pct"] = row["fg3m"] / row["fg3a"] if row["fg3a"] else np.nan
        row["ts_pct"] = row["points"] / (2 * (row["fga"] + 0.44 * row["fta"])) if (row["fga"] + 0.44 * row["fta"]) else np.nan
        row["ast_tov_ratio"] = row["assists"] / max(row["turnovers"], 1)
        row["three_point_attempt_rate"] = row["fg3a"] / max(row["fga"], 1)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("total_minutes", ascending=False).reset_index(drop=True)


def build_game_level_player_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a player-game matrix with one row per player appearance."""
    return df[df["minutes_float"] > 0].copy()


def _combo_summary(df: pd.DataFrame, combo_size: int) -> pd.DataFrame:
    rows = []
    for game_id, group in df[df["minutes_float"] >= 8].groupby("game_id"):
        players = sorted(group["player_name"].unique())
        for combo in itertools.combinations(players, combo_size):
            combo_group = group[group["player_name"].isin(combo)]
            total_minutes = combo_group["minutes_float"].sum()
            fga = combo_group["fga"].sum()
            fg3a = combo_group["fg3a"].sum()
            turnovers = combo_group["turnovers"].sum()
            rows.append(
                {
                    "combo": " / ".join(combo),
                    "combo_size": combo_size,
                    "game_id": game_id,
                    "combined_minutes": total_minutes,
                    "points": combo_group["points"].sum(),
                    "fgm": combo_group["fgm"].sum(),
                    "fga": fga,
                    "fg3m": combo_group["fg3m"].sum(),
                    "fg3a": fg3a,
                    "rebounds": combo_group["rebounds"].sum(),
                    "assists": combo_group["assists"].sum(),
                    "turnovers": turnovers,
                    "plus_minus": combo_group["plus_minus"].sum(),
                }
            )

    if not rows:
        return pd.DataFrame()

    game_combo_df = pd.DataFrame(rows)
    summaries = []
    for combo, group in game_combo_df.groupby("combo"):
        total_minutes = group["combined_minutes"].sum()
        if total_minutes <= 0:
            continue
        points = group["points"].sum()
        fgm = group["fgm"].sum()
        fga = group["fga"].sum()
        fg3m = group["fg3m"].sum()
        fg3a = group["fg3a"].sum()
        rebounds = group["rebounds"].sum()
        assists = group["assists"].sum()
        turnovers = group["turnovers"].sum()
        summaries.append(
            {
                "combo": combo,
                "combo_size": combo_size,
                "games_together": group["game_id"].nunique(),
                "total_combined_minutes": total_minutes,
                "combined_points_per_36": points * 36 / total_minutes,
                "combined_fg3a_per_36": fg3a * 36 / total_minutes,
                "combined_fg3_pct": fg3m / fg3a if fg3a else np.nan,
                "combined_rebounds_per_36": rebounds * 36 / total_minutes,
                "combined_assists_per_turnover": assists / max(turnovers, 1),
                "average_plus_minus": group["plus_minus"].mean(),
                "combined_turnovers_per_36": turnovers * 36 / total_minutes,
                "combined_efg_pct": (fgm + 0.5 * fg3m) / fga if fga else np.nan,
            }
        )

    return pd.DataFrame(summaries).sort_values(
        ["games_together", "total_combined_minutes"], ascending=False
    ).reset_index(drop=True)


def build_2man_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """Build approximate 2-man combo summaries from co-appearances."""
    return _combo_summary(df, combo_size=2)


def build_3man_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """Build approximate 3-man combo summaries from co-appearances."""
    return _combo_summary(df, combo_size=3)
