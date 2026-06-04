"""Tournament-style NBA series simulation using Part 1A Elo distributions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ELO_COLUMNS = {"team", "mu", "sigma"}


def load_elo_ratings(filepath: str | Path) -> dict[str, dict[str, float]]:
    """Load final Elo ratings from Part 1A into a team-keyed dictionary."""
    df = pd.read_csv(filepath)
    missing_columns = sorted(REQUIRED_ELO_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Elo ratings file is missing required columns: "
            + ", ".join(missing_columns)
        )

    ratings = {}
    for _, row in df.iterrows():
        ratings[str(row["team"])] = {
            "mu": float(row["mu"]),
            "sigma": float(row["sigma"]),
        }
    return ratings


def elo_win_probability(
    rating_a: float, rating_b: float, is_home_a: bool, home_court_elo: float
) -> float:
    """Return Team A's single-game win probability against Team B."""
    home_adjustment = home_court_elo if is_home_a else -home_court_elo
    rating_diff = rating_a - rating_b + home_adjustment
    return float(1 / (1 + 10 ** (-(rating_diff / 400))))


def sample_team_rating(team: str, ratings: dict[str, dict[str, float]]) -> float:
    """Sample one team rating from its Normal(mu, sigma) Elo distribution."""
    if team not in ratings:
        raise KeyError(f"Team '{team}' is not present in the Elo ratings.")
    return float(np.random.normal(ratings[team]["mu"], ratings[team]["sigma"]))


def _validate_home_schedule(home_schedule_a: list[bool]) -> None:
    if len(home_schedule_a) != 7:
        raise ValueError("home_schedule_a must contain exactly 7 boolean values.")


def _simulate_one_series(
    team_a: str,
    team_b: str,
    ratings: dict[str, dict[str, float]],
    home_schedule_a: list[bool],
    rng: np.random.Generator,
    home_court_elo: float,
) -> tuple[str, int, str]:
    """Simulate one best-of-seven series, sampling ratings once for the series."""
    rating_a = float(rng.normal(ratings[team_a]["mu"], ratings[team_a]["sigma"]))
    rating_b = float(rng.normal(ratings[team_b]["mu"], ratings[team_b]["sigma"]))

    wins_a = 0
    wins_b = 0

    for game_number, is_home_a in enumerate(home_schedule_a, start=1):
        p_a_wins = elo_win_probability(rating_a, rating_b, is_home_a, home_court_elo)
        if rng.random() < p_a_wins:
            wins_a += 1
        else:
            wins_b += 1

        if wins_a == 4:
            return team_a, game_number, f"{team_a} in {game_number}"
        if wins_b == 4:
            return team_b, game_number, f"{team_b} in {game_number}"

    raise RuntimeError("Best-of-seven series ended without a winner.")


def simulate_series(
    team_a: str,
    team_b: str,
    ratings: dict[str, dict[str, float]],
    home_schedule_a: list[bool],
    n_sims: int = 100000,
    home_court_elo: float = 50,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Simulate a best-of-seven series between Team A and Team B."""
    _validate_home_schedule(home_schedule_a)
    for team in (team_a, team_b):
        if team not in ratings:
            raise KeyError(f"Team '{team}' is not present in the Elo ratings.")

    rng = np.random.default_rng(random_seed)
    winner_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    result_counts: Counter[str] = Counter()

    for _ in range(n_sims):
        winner, series_length, result = _simulate_one_series(
            team_a,
            team_b,
            ratings,
            home_schedule_a,
            rng,
            home_court_elo,
        )
        winner_counts[winner] += 1
        length_counts[series_length] += 1
        result_counts[result] += 1

    average_series_length = sum(
        length * count for length, count in length_counts.items()
    ) / n_sims

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_series_win_prob": winner_counts[team_a] / n_sims,
        "team_b_series_win_prob": winner_counts[team_b] / n_sims,
        "average_series_length": average_series_length,
        "series_length_distribution": dict(sorted(length_counts.items())),
        "result_distribution": dict(result_counts.most_common()),
        "n_sims": n_sims,
    }


def simulate_championship_path(
    ratings: dict[str, dict[str, float]],
    n_sims: int = 100000,
    home_court_elo: float = 50,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Simulate the current simplified Knicks championship path."""
    required_teams = {"NYK", "OKC", "SAS"}
    missing_teams = sorted(required_teams - set(ratings))
    if missing_teams:
        raise KeyError("Missing expected teams: " + ", ".join(missing_teams))

    okc_home_schedule = [True, True, False, False, True, False, True]
    nyk_vs_okc_home_schedule = [False, False, True, True, False, True, False]
    nyk_vs_sas_home_schedule = [True, True, False, False, True, False, True]

    rng = np.random.default_rng(random_seed)
    opponent_counts: Counter[str] = Counter()
    finals_result_counts: Counter[str] = Counter()
    nyk_title_count = 0
    nyk_over_okc_count = 0
    nyk_over_sas_count = 0

    for _ in range(n_sims):
        west_winner, _, _ = _simulate_one_series(
            "OKC",
            "SAS",
            ratings,
            okc_home_schedule,
            rng,
            home_court_elo,
        )
        opponent_counts[west_winner] += 1

        if west_winner == "OKC":
            finals_winner, finals_length, _ = _simulate_one_series(
                "NYK",
                "OKC",
                ratings,
                nyk_vs_okc_home_schedule,
                rng,
                home_court_elo,
            )
            finals_result_counts[f"{finals_winner} over {'OKC' if finals_winner == 'NYK' else 'NYK'} in {finals_length}"] += 1
            if finals_winner == "NYK":
                nyk_title_count += 1
                nyk_over_okc_count += 1
        else:
            finals_winner, finals_length, _ = _simulate_one_series(
                "NYK",
                "SAS",
                ratings,
                nyk_vs_sas_home_schedule,
                rng,
                home_court_elo,
            )
            finals_result_counts[f"{finals_winner} over {'SAS' if finals_winner == 'NYK' else 'NYK'} in {finals_length}"] += 1
            if finals_winner == "NYK":
                nyk_title_count += 1
                nyk_over_sas_count += 1

    okc_finals_count = opponent_counts["OKC"]
    sas_finals_count = opponent_counts["SAS"]

    return {
        "p_okc_reaches_finals": okc_finals_count / n_sims,
        "p_sas_reaches_finals": sas_finals_count / n_sims,
        "p_nyk_beats_okc": nyk_over_okc_count / okc_finals_count if okc_finals_count else 0.0,
        "p_nyk_beats_sas": nyk_over_sas_count / sas_finals_count if sas_finals_count else 0.0,
        "p_nyk_title": nyk_title_count / n_sims,
        "title_simulation_count": n_sims,
        "opponent_counts": dict(opponent_counts),
        "finals_result_counts": dict(finals_result_counts.most_common()),
    }
