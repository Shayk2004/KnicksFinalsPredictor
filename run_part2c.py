"""Run Part 2C what-if scenario analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from part2c_what_if import (
    load_thresholds,
    plot_title_probability,
    run_scenarios,
    save_results,
)


OKC_THRESHOLDS_PATH = Path("outputs/part2b_okc_thresholds.csv")
SAS_THRESHOLDS_PATH = Path("outputs/part2b_sas_thresholds.csv")
RESULTS_PATH = Path("outputs/part2c_what_if_results.csv")
PLOT_PATH = Path("outputs/part2c_what_if_title_probability.png")
PART1B_SUMMARY_PATH = Path("outputs/part1b_championship_summary.csv")

DEFAULT_BASELINE_PROBS = {
    "p_okc_finals": 0.76049,
    "p_sas_finals": 0.23951,
    "p_nyk_beats_okc": 0.3976909624058173,
    "p_nyk_beats_sas": 0.6899085633167718,
    "p_title": 0.46768,
}


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _load_baseline_probs() -> dict[str, float]:
    if not PART1B_SUMMARY_PATH.exists():
        return DEFAULT_BASELINE_PROBS

    summary = pd.read_csv(PART1B_SUMMARY_PATH)
    if summary.empty:
        return DEFAULT_BASELINE_PROBS

    row = summary.iloc[0]
    return {
        "p_okc_finals": float(row["p_okc_reaches_finals"]),
        "p_sas_finals": float(row["p_sas_reaches_finals"]),
        "p_nyk_beats_okc": float(row["p_nyk_beats_okc"]),
        "p_nyk_beats_sas": float(row["p_nyk_beats_sas"]),
        "p_title": float(row["p_nyk_title"]),
    }


def main() -> None:
    okc_thresholds = load_thresholds(OKC_THRESHOLDS_PATH)
    sas_thresholds = load_thresholds(SAS_THRESHOLDS_PATH)
    baseline_probs = _load_baseline_probs()
    results = run_scenarios(okc_thresholds, sas_thresholds, baseline_probs)

    save_results(results, RESULTS_PATH)
    plot_title_probability(results, baseline_probs["p_title"], PLOT_PATH)

    print("Part 2C What-if Scenario Analysis")
    print("=================================")
    print(
        "These what-if scenarios are not causal guarantees. They translate "
        "historically associated opponent-loss conditions into estimated "
        "probability gains using conservative actionability discounts."
    )
    print()
    print(f"Baseline title probability: {_fmt_pct(baseline_probs['p_title'])}")
    print()

    for row in results.itertuples(index=False):
        print(row.scenario_name)
        if row.adjusted_p_nyk_beats_okc != row.baseline_p_nyk_beats_okc:
            print(f"  Adjusted NYK vs OKC: {_fmt_pct(row.adjusted_p_nyk_beats_okc)}")
        if row.adjusted_p_nyk_beats_sas != row.baseline_p_nyk_beats_sas:
            print(f"  Adjusted NYK vs SAS: {_fmt_pct(row.adjusted_p_nyk_beats_sas)}")
        print(f"  Adjusted title probability: {_fmt_pct(row.adjusted_p_title)}")
        print(f"  Title probability gain: {_fmt_pct(row.title_probability_gain)}")
        print()

    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
