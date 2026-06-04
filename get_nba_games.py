import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog


SEASONS = ["2024-25", "2025-26"]
SEASON_TYPES = ["Regular Season", "Playoffs"]


def fetch_league_game_log(season, season_type):
    """
    Pulls team-level NBA game logs from NBA.com through nba_api.
    PlayerOrTeam='T' means team-level game logs.
    """
    print(f"Fetching {season} {season_type}...")

    response = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
        sorter="DATE",
        direction="ASC",
        timeout=60
    )

    df = response.get_data_frames()[0]
    df["SEASON"] = season
    df["SEASON_TYPE"] = season_type

    # Be polite to the NBA stats API
    time.sleep(1)

    return df


def fetch_all_games():
    frames = []

    for season in SEASONS:
        for season_type in SEASON_TYPES:
            try:
                df = fetch_league_game_log(season, season_type)
                frames.append(df)
            except Exception as e:
                print(f"Failed to fetch {season} {season_type}: {e}")

    if not frames:
        raise RuntimeError("No data was downloaded.")

    return pd.concat(frames, ignore_index=True)


def convert_team_logs_to_game_rows(team_logs):
    """
    LeagueGameLog returns one row per team per game.
    That means each NBA game appears twice:
    - one row for the home team
    - one row for the away team

    This function converts that into one row per actual game:
    date, season, game_type, home_team, away_team, home_score, away_score
    """

    df = team_logs.copy()

    # Keep useful columns
    required_cols = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "SEASON_TYPE",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "PTS"
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from nba_api: {missing}")

    df = df[required_cols]

    # Determine home/away from MATCHUP.
    # NBA format examples:
    # "NYK vs. BOS" means NYK home
    # "NYK @ BOS" means NYK away
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)

    game_rows = []

    for game_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            # Skip weird cases if any appear
            continue

        home = group[group["is_home"] == True]
        away = group[group["is_home"] == False]

        if len(home) != 1 or len(away) != 1:
            continue

        home = home.iloc[0]
        away = away.iloc[0]

        game_rows.append({
            "date": home["GAME_DATE"],
            "season": home["SEASON"],
            "game_type": (
                "regular_season"
                if home["SEASON_TYPE"] == "Regular Season"
                else "playoffs"
            ),
            "home_team": home["TEAM_ABBREVIATION"],
            "away_team": away["TEAM_ABBREVIATION"],
            "home_score": int(home["PTS"]),
            "away_score": int(away["PTS"]),
            "neutral_site": False
        })

    games = pd.DataFrame(game_rows)
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["date", "season"]).reset_index(drop=True)

    return games


def main():
    raw_logs = fetch_all_games()

    raw_logs.to_csv("raw_nba_team_game_logs.csv", index=False)
    print("Saved raw_nba_team_game_logs.csv")

    games = convert_team_logs_to_game_rows(raw_logs)

    games.to_csv("nba_games.csv", index=False)
    print("Saved nba_games.csv")

    print("\nPreview:")
    print(games.head())

    print("\nRows:", len(games))
    print("Seasons:", games["season"].unique())
    print("Game types:")
    print(games["game_type"].value_counts())


if __name__ == "__main__":
    main()