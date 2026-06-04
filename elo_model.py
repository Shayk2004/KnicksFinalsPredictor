"""Bayesian-style Elo model for NBA game-level data."""

from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


REQUIRED_COLUMNS = {
    "date",
    "season",
    "game_type",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


def validate_games_df(df: pd.DataFrame) -> None:
    """Validate that the games dataframe has the columns needed by the Elo model."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(
            "nba_games.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError("nba_games.csv contains no games.")

    required_columns = list(REQUIRED_COLUMNS)
    if df[required_columns].isnull().any().any():
        null_columns = sorted(df[required_columns].columns[df[required_columns].isnull().any()])
        raise ValueError(
            "Required columns contain missing values: " + ", ".join(null_columns)
        )

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isnull().any():
        raise ValueError("Column 'date' contains values that cannot be parsed as dates.")

    for score_col in ("home_score", "away_score"):
        if pd.to_numeric(df[score_col], errors="coerce").isnull().any():
            raise ValueError(f"Column '{score_col}' must contain numeric scores.")


def estimate_home_court_elo(df: pd.DataFrame) -> float:
    """Estimate home-court advantage in Elo points using win ~ is_home."""
    home_rows = pd.DataFrame(
        {
            "win": (df["home_score"] > df["away_score"]).astype(int),
            "is_home": 1,
        }
    )
    away_rows = pd.DataFrame(
        {
            "win": (df["away_score"] > df["home_score"]).astype(int),
            "is_home": 0,
        }
    )
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)

    if team_games["win"].nunique() < 2:
        raise ValueError("Cannot estimate home-court advantage without both wins and losses.")

    model = LogisticRegression()
    model.fit(team_games[["is_home"]], team_games["win"])
    beta_home = model.coef_[0][0]

    return float((400 / log(10)) * beta_home)


def expected_score(rating_diff: float) -> float:
    """Convert an Elo rating difference to a win probability."""
    return float(1 / (1 + 10 ** (-rating_diff / 400)))


def margin_multiplier(margin: float, rating_diff: float) -> float:
    """Scale updates by margin of victory and pregame raw rating difference."""
    return float(
        (np.log(abs(margin) + 1) * 2.2)
        / ((0.001 * abs(rating_diff)) + 2.2)
    )


def regress_to_new_season(
    ratings: dict[str, dict[str, float]], regression_factor: float = 0.75
) -> dict[str, dict[str, float]]:
    """Regress team ratings toward league average and increase uncertainty."""
    for team_rating in ratings.values():
        team_rating["mu"] = 1500 + regression_factor * (team_rating["mu"] - 1500)
        team_rating["sigma"] = min(100, team_rating["sigma"] + 25)
    return ratings


def initialize_team_if_needed(
    team: str, ratings: dict[str, dict[str, float]]
) -> None:
    """Create a default rating record for a team that has not appeared yet."""
    if team not in ratings:
        ratings[team] = {"mu": 1500.0, "sigma": 100.0}


def update_uncertainty(
    sigma: float, prediction_error: float, sigma_min: float = 35, decay: float = 0.985, c: float = 1.5
) -> float:
    """Decay uncertainty, then add uncertainty proportional to prediction error."""
    return float(max(sigma_min, sigma * decay + c * prediction_error))


def _is_neutral_site(row: pd.Series) -> bool:
    if "neutral_site" not in row.index or pd.isna(row["neutral_site"]):
        return False
    value = row["neutral_site"]
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return bool(value)


def _is_playoff_game(game_type: object) -> bool:
    return str(game_type).strip().lower() in {"playoff", "playoffs", "postseason"}


def run_bayesian_elo(
    df: pd.DataFrame,
    initial_mu: float = 1500,
    initial_sigma: float = 100,
    k_regular: float = 20,
    k_playoff: float = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Bayesian-style Elo model and return final ratings plus game history."""
    validate_games_df(df)
    home_court_elo = estimate_home_court_elo(df)

    games = df.copy()
    games["date"] = pd.to_datetime(games["date"])
    games["home_score"] = pd.to_numeric(games["home_score"])
    games["away_score"] = pd.to_numeric(games["away_score"])
    games = games.sort_values(["date", "season"]).reset_index(drop=True)

    ratings: dict[str, dict[str, float]] = {}
    history = []
    current_season = None

    for _, row in games.iterrows():
        season = row["season"]
        if current_season is None:
            current_season = season
        elif season != current_season:
            regress_to_new_season(ratings)
            current_season = season

        home_team = row["home_team"]
        away_team = row["away_team"]
        home_is_new = home_team not in ratings
        away_is_new = away_team not in ratings
        initialize_team_if_needed(home_team, ratings)
        initialize_team_if_needed(away_team, ratings)

        if home_is_new:
            ratings[home_team] = {"mu": float(initial_mu), "sigma": float(initial_sigma)}
        if away_is_new:
            ratings[away_team] = {"mu": float(initial_mu), "sigma": float(initial_sigma)}

        home_mu_pre = ratings[home_team]["mu"]
        away_mu_pre = ratings[away_team]["mu"]
        home_sigma_pre = ratings[home_team]["sigma"]
        away_sigma_pre = ratings[away_team]["sigma"]

        neutral_site = _is_neutral_site(row)
        raw_rating_diff = home_mu_pre - away_mu_pre
        rating_diff = raw_rating_diff if neutral_site else raw_rating_diff + home_court_elo

        expected_home = expected_score(rating_diff)
        expected_away = 1 - expected_home

        margin = abs(row["home_score"] - row["away_score"])
        multiplier = margin_multiplier(margin, raw_rating_diff)
        actual_home = 1 if row["home_score"] > row["away_score"] else 0
        k_value = k_playoff if _is_playoff_game(row["game_type"]) else k_regular

        # Elo update: observed result minus expected result, scaled by K and margin.
        delta = k_value * multiplier * (actual_home - expected_home)
        home_mu_post = home_mu_pre + delta
        away_mu_post = away_mu_pre - delta

        prediction_error_home = abs(actual_home - expected_home)
        prediction_error_away = abs((1 - actual_home) - expected_away)
        home_sigma_post = update_uncertainty(home_sigma_pre, prediction_error_home)
        away_sigma_post = update_uncertainty(away_sigma_pre, prediction_error_away)

        ratings[home_team] = {"mu": home_mu_post, "sigma": home_sigma_post}
        ratings[away_team] = {"mu": away_mu_post, "sigma": away_sigma_post}

        history.append(
            {
                "date": row["date"],
                "season": season,
                "game_type": row["game_type"],
                "home_team": home_team,
                "away_team": away_team,
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                "home_mu_pre": home_mu_pre,
                "away_mu_pre": away_mu_pre,
                "home_sigma_pre": home_sigma_pre,
                "away_sigma_pre": away_sigma_pre,
                "expected_home": expected_home,
                "expected_away": expected_away,
                "margin": margin,
                "multiplier": multiplier,
                "k_value": k_value,
                "delta": delta,
                "home_mu_post": home_mu_post,
                "away_mu_post": away_mu_post,
                "home_sigma_post": home_sigma_post,
                "away_sigma_post": away_sigma_post,
            }
        )

    final_rows = [
        {
            "team": team,
            "mu": team_rating["mu"],
            "sigma": team_rating["sigma"],
            "lower_95": team_rating["mu"] - 1.96 * team_rating["sigma"],
            "upper_95": team_rating["mu"] + 1.96 * team_rating["sigma"],
        }
        for team, team_rating in ratings.items()
    ]
    final_elo = pd.DataFrame(final_rows).sort_values("mu", ascending=False).reset_index(drop=True)
    game_history = pd.DataFrame(history)

    return final_elo, game_history
