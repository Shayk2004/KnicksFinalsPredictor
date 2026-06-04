"""Run Part 1B tournament simulations from Part 1A final Elo ratings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tournament_simulator import load_elo_ratings, simulate_championship_path, simulate_series


ELO_PATH = Path("outputs/final_elo.csv")
OUTPUT_DIR = Path("outputs")
SERIES_RESULTS_PATH = OUTPUT_DIR / "part1b_series_results.csv"
CHAMPIONSHIP_SUMMARY_PATH = OUTPUT_DIR / "part1b_championship_summary.csv"

N_SIMS = 100000
HOME_COURT_ELO = 50

OKC_HOME_SCHEDULE = [True, True, False, False, True, False, True]
NYK_VS_OKC_HOME_SCHEDULE = [False, False, True, True, False, True, False]
NYK_VS_SAS_HOME_SCHEDULE = [True, True, False, False, True, False, True]


def _format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _most_common_result(result_distribution: dict[str, int]) -> str:
    if not result_distribution:
        return "N/A"
    result, count = next(iter(result_distribution.items()))
    return f"{result} ({count:,})"


def _series_summary_row(name: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "series": name,
        "team_a": result["team_a"],
        "team_b": result["team_b"],
        "team_a_series_win_prob": result["team_a_series_win_prob"],
        "team_b_series_win_prob": result["team_b_series_win_prob"],
        "average_series_length": result["average_series_length"],
        "most_common_result": _most_common_result(result["result_distribution"]),
        "series_length_distribution": result["series_length_distribution"],
        "result_distribution": result["result_distribution"],
        "n_sims": result["n_sims"],
    }


def main() -> None:
    ratings = load_elo_ratings(ELO_PATH)

    okc_sas = simulate_series(
        "OKC",
        "SAS",
        ratings,
        OKC_HOME_SCHEDULE,
        n_sims=N_SIMS,
        home_court_elo=HOME_COURT_ELO,
        random_seed=101,
    )
    nyk_okc = simulate_series(
        "NYK",
        "OKC",
        ratings,
        NYK_VS_OKC_HOME_SCHEDULE,
        n_sims=N_SIMS,
        home_court_elo=HOME_COURT_ELO,
        random_seed=102,
    )
    nyk_sas = simulate_series(
        "NYK",
        "SAS",
        ratings,
        NYK_VS_SAS_HOME_SCHEDULE,
        n_sims=N_SIMS,
        home_court_elo=HOME_COURT_ELO,
        random_seed=103,
    )
    championship = simulate_championship_path(
        ratings,
        n_sims=N_SIMS,
        home_court_elo=HOME_COURT_ELO,
        random_seed=104,
    )

    series_rows = [
        _series_summary_row("OKC vs SAS Western Finals", okc_sas),
        _series_summary_row("NYK vs OKC Finals", nyk_okc),
        _series_summary_row("NYK vs SAS Finals", nyk_sas),
    ]
    series_df = pd.DataFrame(series_rows)

    championship_df = pd.DataFrame(
        [
            {
                "p_okc_reaches_finals": championship["p_okc_reaches_finals"],
                "p_sas_reaches_finals": championship["p_sas_reaches_finals"],
                "p_nyk_beats_okc": championship["p_nyk_beats_okc"],
                "p_nyk_beats_sas": championship["p_nyk_beats_sas"],
                "p_nyk_title": championship["p_nyk_title"],
                "title_simulation_count": championship["title_simulation_count"],
                "opponent_counts": championship["opponent_counts"],
                "finals_result_counts": championship["finals_result_counts"],
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    series_df.to_csv(SERIES_RESULTS_PATH, index=False)
    championship_df.to_csv(CHAMPIONSHIP_SUMMARY_PATH, index=False)

    print("Part 1B Championship Forecast")
    print("=============================")
    print(f"OKC reaches Finals: {_format_pct(championship['p_okc_reaches_finals'])}")
    print(f"SAS reaches Finals: {_format_pct(championship['p_sas_reaches_finals'])}")
    print(f"Knicks beat OKC: {_format_pct(championship['p_nyk_beats_okc'])}")
    print(f"Knicks beat SAS: {_format_pct(championship['p_nyk_beats_sas'])}")
    print(f"Overall Knicks championship probability: {_format_pct(championship['p_nyk_title'])}")
    print()
    print("Series Details")
    print("--------------")
    for row in series_rows:
        print(
            f"{row['series']}: avg length {row['average_series_length']:.2f}, "
            f"most common {row['most_common_result']}"
        )
    print()
    print(f"Saved series results to {SERIES_RESULTS_PATH}")
    print(f"Saved championship summary to {CHAMPIONSHIP_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
