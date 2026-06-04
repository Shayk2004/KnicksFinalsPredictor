"""Part 2C what-if scenario engine using Part 2B threshold lifts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_THRESHOLD_COLUMNS = {
    "condition_name",
    "overall_loss_rate",
    "team_loss_rate_when_true",
    "team_loss_rate_when_false",
    "games_condition_true",
    "games_condition_false",
}

OUTPUT_DIR = Path("outputs")
RESULTS_PATH = OUTPUT_DIR / "part2c_what_if_results.csv"
PLOT_PATH = OUTPUT_DIR / "part2c_what_if_title_probability.png"


def load_thresholds(filepath: str | Path) -> pd.DataFrame:
    """Load a Part 2B threshold CSV and ensure required fields are present."""
    df = pd.read_csv(filepath)
    missing = sorted(REQUIRED_THRESHOLD_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{filepath} is missing required columns: " + ", ".join(missing)
        )

    if "loss_rate_lift" not in df.columns:
        df["loss_rate_lift"] = df["team_loss_rate_when_true"] - df["overall_loss_rate"]

    return df


def get_lift(threshold_df: pd.DataFrame, condition_name: str) -> float:
    """Return loss-rate lift for a named threshold condition."""
    matches = threshold_df[threshold_df["condition_name"] == condition_name]
    if matches.empty:
        available = "\n- ".join(threshold_df["condition_name"].astype(str).tolist())
        raise ValueError(
            f"Condition '{condition_name}' was not found. Available conditions:\n- {available}"
        )
    return float(matches.iloc[0]["loss_rate_lift"])


def combine_lifts_two(lift1: float, lift2: float) -> float:
    """Combine two lifts with conservative overlap adjustment."""
    return float(0.6 * max(lift1, lift2) + 0.4 * min(lift1, lift2))


def combine_lifts_three(lifts: list[float]) -> float:
    """Combine three lifts with conservative max/median/min weighting."""
    if len(lifts) != 3:
        raise ValueError("combine_lifts_three requires exactly three lifts.")
    return float(0.5 * max(lifts) + 0.3 * np.median(lifts) + 0.2 * min(lifts))


def apply_actionability_discount(combined_lift: float, alpha: float) -> float:
    """Apply a scenario-specific conversion discount to a combined lift."""
    return float(alpha * combined_lift)


def clamp_probability(p: float, lower: float = 0.05, upper: float = 0.95) -> float:
    """Clamp a probability to a conservative range."""
    return float(min(upper, max(lower, p)))


def recompute_title_probability(
    p_okc_finals: float,
    p_sas_finals: float,
    p_nyk_beats_okc: float,
    p_nyk_beats_sas: float,
) -> float:
    """Recompute Knicks title probability from opponent mix and matchup probabilities."""
    return float(p_okc_finals * p_nyk_beats_okc + p_sas_finals * p_nyk_beats_sas)


def _scenario_specs() -> list[dict[str, object]]:
    return [
        {
            "scenario_name": "OKC-1 SGA pressure and efficiency reduction",
            "opponent": "OKC",
            "conditions": ["SGA_TOV >= 4", "SGA_TS_PCT < 0.58"],
            "target": "p_nyk_beats_okc",
            "alpha": 0.50,
            "description": "Knicks force SGA into a higher-turnover and lower-efficiency creation game.",
        },
        {
            "scenario_name": "OKC-2 Bench survival",
            "opponent": "OKC",
            "conditions": ["OKC_BENCH_POINTS < team median", "OKC_BENCH_MARGIN < 0"],
            "target": "p_nyk_beats_okc",
            "alpha": 0.60,
            "description": "Knicks prevent OKC's bench from creating its usual scoring and plus-minus advantage.",
        },
        {
            "scenario_name": "SAS-1 Wembanyama spacing neutralization",
            "opponent": "SAS",
            "conditions": [
                "WEMBANYAMA_BLK <= 2",
                "OPP_FG3A above opponent median",
                "OPP_FG3_PCT above opponent median",
            ],
            "target": "p_nyk_beats_sas",
            "alpha": 0.65,
            "description": "Knicks use Towns spacing and drive-and-kick offense to reduce Wembanyama's rim-protection impact and increase three-point pressure.",
        },
        {
            "scenario_name": "SAS-3 Bench and rebounding control",
            "opponent": "SAS",
            "conditions": [
                "SAS_BENCH_MARGIN < 0",
                "SAS_BENCH_POINTS < team median",
                "WEMBANYAMA_REB < 10",
            ],
            "target": "p_nyk_beats_sas",
            "alpha": 0.55,
            "description": "Knicks win non-Wembanyama minutes and prevent San Antonio from controlling the glass.",
        },
    ]


def _combine_lifts(lifts: list[float]) -> float:
    if len(lifts) == 2:
        return combine_lifts_two(lifts[0], lifts[1])
    if len(lifts) == 3:
        return combine_lifts_three(lifts)
    raise ValueError("Only two- and three-condition scenarios are supported.")


def _scenario_row(
    spec: dict[str, object],
    condition_lifts: list[float],
    combined_lift: float,
    adjusted_probability_gain: float,
    baseline_probs: dict[str, float],
    adjusted_okc: float,
    adjusted_sas: float,
) -> dict[str, object]:
    adjusted_title = recompute_title_probability(
        baseline_probs["p_okc_finals"],
        baseline_probs["p_sas_finals"],
        adjusted_okc,
        adjusted_sas,
    )
    return {
        "scenario_name": spec["scenario_name"],
        "opponent": spec["opponent"],
        "description": spec["description"],
        "conditions_used": "; ".join(spec["conditions"]),
        "condition_lifts": "; ".join(f"{lift:.6f}" for lift in condition_lifts),
        "combined_lift": combined_lift,
        "alpha": spec["alpha"],
        "adjusted_probability_gain": adjusted_probability_gain,
        "baseline_p_nyk_beats_okc": baseline_probs["p_nyk_beats_okc"],
        "baseline_p_nyk_beats_sas": baseline_probs["p_nyk_beats_sas"],
        "adjusted_p_nyk_beats_okc": adjusted_okc,
        "adjusted_p_nyk_beats_sas": adjusted_sas,
        "baseline_p_title": baseline_probs["p_title"],
        "adjusted_p_title": adjusted_title,
        "title_probability_gain": adjusted_title - baseline_probs["p_title"],
    }


def run_scenarios(
    okc_thresholds: pd.DataFrame,
    sas_thresholds: pd.DataFrame,
    baseline_probs: dict[str, float],
) -> pd.DataFrame:
    """Run selected Part 2C what-if scenarios."""
    rows = []
    scenario_gains: dict[str, float] = {}

    for spec in _scenario_specs():
        threshold_df = okc_thresholds if spec["opponent"] == "OKC" else sas_thresholds
        condition_lifts = [get_lift(threshold_df, condition) for condition in spec["conditions"]]
        combined_lift = _combine_lifts(condition_lifts)
        gain = apply_actionability_discount(combined_lift, float(spec["alpha"]))
        scenario_gains[str(spec["scenario_name"])] = gain

        adjusted_okc = baseline_probs["p_nyk_beats_okc"]
        adjusted_sas = baseline_probs["p_nyk_beats_sas"]
        if spec["target"] == "p_nyk_beats_okc":
            adjusted_okc = clamp_probability(adjusted_okc + gain)
        else:
            adjusted_sas = clamp_probability(adjusted_sas + gain)

        rows.append(
            _scenario_row(
                spec,
                condition_lifts,
                combined_lift,
                gain,
                baseline_probs,
                adjusted_okc,
                adjusted_sas,
            )
        )

    okc_total_gain = 0.75 * (
        scenario_gains["OKC-1 SGA pressure and efficiency reduction"]
        + scenario_gains["OKC-2 Bench survival"]
    )
    sas_total_gain = 0.75 * (
        scenario_gains["SAS-1 Wembanyama spacing neutralization"]
        + scenario_gains["SAS-3 Bench and rebounding control"]
    )
    adjusted_okc = clamp_probability(baseline_probs["p_nyk_beats_okc"] + okc_total_gain)
    adjusted_sas = clamp_probability(baseline_probs["p_nyk_beats_sas"] + sas_total_gain)
    combined_title = recompute_title_probability(
        baseline_probs["p_okc_finals"],
        baseline_probs["p_sas_finals"],
        adjusted_okc,
        adjusted_sas,
    )
    rows.append(
        {
            "scenario_name": "Combined best actionable plan",
            "opponent": "OKC/SAS",
            "description": "Applies OKC-1, OKC-2, SAS-1, and SAS-3 with a 0.75 overlap factor within each opponent.",
            "conditions_used": "OKC-1; OKC-2; SAS-1; SAS-3",
            "condition_lifts": "",
            "combined_lift": np.nan,
            "alpha": np.nan,
            "adjusted_probability_gain": np.nan,
            "baseline_p_nyk_beats_okc": baseline_probs["p_nyk_beats_okc"],
            "baseline_p_nyk_beats_sas": baseline_probs["p_nyk_beats_sas"],
            "adjusted_p_nyk_beats_okc": adjusted_okc,
            "adjusted_p_nyk_beats_sas": adjusted_sas,
            "baseline_p_title": baseline_probs["p_title"],
            "adjusted_p_title": combined_title,
            "title_probability_gain": combined_title - baseline_probs["p_title"],
        }
    )

    return pd.DataFrame(rows)


def save_results(results_df: pd.DataFrame, filepath: str | Path = RESULTS_PATH) -> None:
    """Save Part 2C scenario results."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(filepath, index=False)


def plot_title_probability(
    results_df: pd.DataFrame,
    baseline_title_probability: float,
    filepath: str | Path = PLOT_PATH,
) -> None:
    """Plot baseline and adjusted title probabilities for each scenario."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    plot_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "scenario_name": ["Baseline"],
                    "title_probability": [baseline_title_probability],
                }
            ),
            results_df[["scenario_name", "adjusted_p_title"]].rename(
                columns={"adjusted_p_title": "title_probability"}
            ),
        ],
        ignore_index=True,
    )
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot_df["scenario_name"], plot_df["title_probability"])
    ax.set_xlabel("Knicks title probability")
    ax.set_xlim(0, max(0.95, plot_df["title_probability"].max() + 0.05))
    ax.set_title("Part 2C What-if Title Probability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
