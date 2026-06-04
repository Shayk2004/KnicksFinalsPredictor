"""Command-line entrypoint for creating NBA Bayesian-style Elo ratings."""

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from elo_model import estimate_home_court_elo, run_bayesian_elo, validate_games_df


INPUT_PATH = Path("nba_games.csv")
OUTPUT_DIR = Path("outputs")
FINAL_ELO_PATH = OUTPUT_DIR / "final_elo.csv"
HISTORY_PATH = OUTPUT_DIR / "elo_game_history.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Place nba_games.csv in this folder, "
            "or run this script from the project root."
        )

    try:
        games = pd.read_csv(INPUT_PATH)
    except EmptyDataError as exc:
        raise ValueError(
            f"{INPUT_PATH} is empty. Add a header row and game data before running "
            "the Elo model."
        ) from exc
    validate_games_df(games)

    home_court_elo = estimate_home_court_elo(games)
    final_elo, game_history = run_bayesian_elo(games)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_elo.to_csv(FINAL_ELO_PATH, index=False)
    game_history.to_csv(HISTORY_PATH, index=False)

    print(f"Estimated home-court advantage: {home_court_elo:.2f} Elo points")
    print()
    print(final_elo.sort_values("mu", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
