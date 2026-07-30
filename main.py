"""Run the U.S. Middle East Strategic Withdrawal Response Model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.backtesting import apply_calibration_status, build_historical_calibration
from src.data_refresh import refresh_context_data
from src.live_context import build_live_context
from src.model import StrategicWithdrawalModel
from src.reporting import build_all_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate actor responses and economic transmission under U.S. withdrawal scenarios."
    )
    parser.add_argument("--runs", type=int, default=4000, help="Monte Carlo runs per scenario.")
    parser.add_argument("--quarters", type=int, default=20, help="Simulation horizon in quarters.")
    parser.add_argument("--seed", type=int, default=20260727, help="Reproducibility seed.")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Attempt optional World Bank and FRED context refreshes.",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip executive PDF generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    print("=" * 76)
    print("U.S. Middle East Strategic Withdrawal Response Model")
    print("=" * 76)

    if args.refresh_data:
        print("[1/9] Refreshing optional public economic context...")
        refresh_context_data(project_root)
    else:
        print("[1/9] Using versioned baseline and assumption files...")

    print("[2/9] Loading scenarios, actors, actions, and exposure data...")
    model = StrategicWithdrawalModel(project_root, seed=args.seed)

    print("[3/9] Calculating transparent actor-action probabilities...")
    action_probabilities = model.calculate_action_probabilities()

    print(f"[4/9] Running {args.runs:,} Monte Carlo paths per scenario...")
    results = model.run_all_scenarios(runs=args.runs, quarters=args.quarters)

    print("[5/9] Translating strategic states into economic stress ranges...")
    outputs = model.build_output_tables(action_probabilities, results)

    print("[6/9] Auditing live context and data-quality gates...")
    live_context = build_live_context(
        project_root, economic_impacts=outputs["economic_impacts"]
    )
    print("[7/9] Backtesting historical event windows and calibration gates...")
    calibration = build_historical_calibration(project_root, seed=args.seed)
    live_context["model_status"] = apply_calibration_status(
        live_context["model_status"], calibration["calibration_summary"]
    )
    status = live_context["model_status"].iloc[0]
    calibration_status = calibration["calibration_summary"].iloc[0]
    print(
        f"      Status: {status['effective_model_status']} | "
        f"Data quality: {status['data_quality_score']:.1f} | "
        f"Market stress: {status['market_stress_score']:.1f} "
        f"({status['market_stress_level']})"
    )
    print(
        f"      Calibration gates: {calibration_status['gates_passed']}/"
        f"{calibration_status['gates_total']} passed | "
        "Coefficients: diagnostic only"
    )

    print("[8/9] Writing tables, charts, methodology, and executive brief...")
    build_all_outputs(
        project_root=project_root,
        model=model,
        outputs={**outputs, **live_context, **calibration},
        create_pdf=not args.no_pdf,
    )

    print("[9/9] Complete.")
    print(f"Outputs: {project_root / 'output'}")
    print()
    print("Important: results are conditional counterfactuals, not forecasts.")


if __name__ == "__main__":
    main()
