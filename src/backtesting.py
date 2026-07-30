"""Historical event-window backtesting for economic transmission coefficients.

This module calibrates only the Brent stress translation. It does not infer
actor intent, estimate the probability of withdrawal, or alter the simulation's
strategic state transitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import BRENT_STRESS_COEFFICIENTS


FEATURE_COLUMNS = ["security_shock", "energy_shock", "maritime_shock"]
FEATURE_LABELS = {
    "security_shock": "Security tension",
    "energy_shock": "Energy disruption",
    "maritime_shock": "Maritime risk",
}
PRIOR_COEFFICIENTS = np.array(
    [
        BRENT_STRESS_COEFFICIENTS["security_tension"],
        BRENT_STRESS_COEFFICIENTS["energy_disruption"],
        BRENT_STRESS_COEFFICIENTS["maritime_risk"],
    ]
)
DEFAULT_RIDGE_STRENGTH = 5.0
BOOTSTRAP_DRAWS = 1000


def _validate_registry(events: pd.DataFrame) -> None:
    required = {
        "event_id",
        "event",
        "event_date",
        *FEATURE_COLUMNS,
        "calibration_eligible",
        "source_name",
        "source_url",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Historical event registry is missing: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("Historical event IDs must be unique")
    for column in FEATURE_COLUMNS:
        if not pd.to_numeric(events[column], errors="coerce").between(0, 1).all():
            raise ValueError(f"{column} must remain within [0, 1]")
    if events["source_url"].fillna("").str.strip().eq("").any():
        raise ValueError("Every historical event must include a source URL")


def _load_fred_pivot(path: Path) -> pd.DataFrame:
    fred = pd.read_csv(path)
    required = {"date", "value", "series"}
    missing = required - set(fred.columns)
    if missing:
        raise ValueError(f"FRED context is missing columns: {sorted(missing)}")
    fred["date"] = pd.to_datetime(fred["date"], errors="coerce")
    fred["value"] = pd.to_numeric(fred["value"], errors="coerce")
    fred = fred.dropna(subset=["date", "series"])
    if fred.duplicated(["series", "date"]).any():
        raise ValueError("FRED context contains duplicate series/date keys")
    return fred.pivot(index="date", columns="series", values="value").sort_index()


def _series_window(
    series: pd.Series, event_date: pd.Timestamp, horizons: tuple[int, ...]
) -> dict[str, float | pd.Timestamp]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    pre = clean[clean.index < event_date]
    post = clean[clean.index >= event_date]
    result: dict[str, float | pd.Timestamp] = {
        "pre_date": pd.NaT,
        "pre_value": np.nan,
    }
    if pre.empty or post.empty:
        for horizon in horizons:
            result[f"end_{horizon}d"] = np.nan
            result[f"change_{horizon}d"] = np.nan
        result["peak_21d"] = np.nan
        result["peak_21d_change"] = np.nan
        return result

    baseline = float(pre.iloc[-1])
    result["pre_date"] = pre.index[-1]
    result["pre_value"] = baseline
    for horizon in horizons:
        if len(post) >= horizon:
            end_value = float(post.iloc[horizon - 1])
            result[f"end_{horizon}d"] = end_value
            result[f"change_{horizon}d"] = 100.0 * (end_value / baseline - 1.0)
        else:
            result[f"end_{horizon}d"] = np.nan
            result[f"change_{horizon}d"] = np.nan
    first_21 = post.iloc[:21]
    if len(first_21):
        peak = float(first_21.max())
        result["peak_21d"] = peak
        result["peak_21d_change"] = 100.0 * (peak / baseline - 1.0)
    else:
        result["peak_21d"] = np.nan
        result["peak_21d_change"] = np.nan
    return result


def extract_event_windows(
    events: pd.DataFrame, fred: pd.DataFrame
) -> pd.DataFrame:
    """Extract pre-event levels and 5/21/63-observation market changes."""
    _validate_registry(events)
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="raise")
    series_specs = {
        "DCOILBRENTEU": ("brent", "pct"),
        "VIXCLS": ("vix", "level"),
        "T10YIE": ("breakeven_10y", "level"),
        "DTWEXBGS": ("broad_dollar", "pct"),
    }
    records: list[dict] = []
    for event in events.itertuples(index=False):
        record = event._asdict()
        for series_id, (prefix, change_kind) in series_specs.items():
            if series_id not in fred:
                window = _series_window(
                    pd.Series(dtype=float), event.event_date, (5, 21, 63)
                )
            else:
                window = _series_window(
                    fred[series_id], event.event_date, (5, 21, 63)
                )
            record[f"{prefix}_pre_date"] = window["pre_date"]
            record[f"{prefix}_pre_value"] = window["pre_value"]
            for horizon in (5, 21, 63):
                record[f"{prefix}_{horizon}d_value"] = window[f"end_{horizon}d"]
                change = window[f"change_{horizon}d"]
                if change_kind == "level" and pd.notna(change):
                    end_value = float(window[f"end_{horizon}d"])
                    pre_value = float(window["pre_value"])
                    change = end_value - pre_value
                record[f"{prefix}_{horizon}d_change"] = change
            if prefix == "brent":
                record["brent_peak_21d_value"] = window["peak_21d"]
                record["brent_peak_21d_change_pct"] = max(
                    0.0, float(window["peak_21d_change"])
                ) if pd.notna(window["peak_21d_change"]) else np.nan
        records.append(record)
    return pd.DataFrame(records)


def fit_ridge_to_prior(
    features: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray = PRIOR_COEFFICIENTS,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
) -> np.ndarray:
    """Fit nonnegative coefficients with quadratic shrinkage to a prior."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    prior = np.asarray(prior, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(prior):
        raise ValueError("Feature matrix and prior dimensions do not match")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("At least two aligned observations are required")
    penalty = ridge_strength * np.eye(x.shape[1])
    coefficients = np.linalg.solve(
        x.T @ x + penalty,
        x.T @ y + penalty @ prior,
    )
    return np.clip(coefficients, 0.0, None)


def _leave_one_out_predictions(
    x: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    ridge_strength: float,
) -> np.ndarray:
    predictions = np.empty(len(y), dtype=float)
    for position in range(len(y)):
        keep = np.arange(len(y)) != position
        coefficients = fit_ridge_to_prior(
            x[keep], y[keep], prior=prior, ridge_strength=ridge_strength
        )
        predictions[position] = float(x[position] @ coefficients)
    return predictions


def _bootstrap_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    ridge_strength: float,
    draws: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    estimates = np.empty((draws, x.shape[1]), dtype=float)
    for draw in range(draws):
        selection = rng.integers(0, len(y), len(y))
        estimates[draw] = fit_ridge_to_prior(
            x[selection],
            y[selection],
            prior=prior,
            ridge_strength=ridge_strength,
        )
    return estimates


def calibrate_brent_transmission(
    event_windows: pd.DataFrame,
    seed: int = 20260727,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, pd.DataFrame]:
    """Backtest and calibrate Brent stress coefficients from event windows."""
    eligible = event_windows[
        event_windows["calibration_eligible"].astype(bool)
        & event_windows["brent_peak_21d_change_pct"].notna()
    ].copy()
    if len(eligible) < 3:
        raise ValueError("At least three eligible Brent event windows are required")

    x = eligible[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = eligible["brent_peak_21d_change_pct"].to_numpy(dtype=float)
    coefficients = fit_ridge_to_prior(
        x, y, prior=PRIOR_COEFFICIENTS, ridge_strength=ridge_strength
    )
    loo = _leave_one_out_predictions(
        x, y, prior=PRIOR_COEFFICIENTS, ridge_strength=ridge_strength
    )
    bootstrap = _bootstrap_coefficients(
        x,
        y,
        prior=PRIOR_COEFFICIENTS,
        ridge_strength=ridge_strength,
        draws=bootstrap_draws,
        seed=seed,
    )

    predictions = eligible[
        [
            "event_id",
            "event",
            "event_date",
            *FEATURE_COLUMNS,
            "brent_peak_21d_change_pct",
            "brent_21d_change",
        ]
    ].copy()
    predictions["fitted_peak_21d_change_pct"] = x @ coefficients
    predictions["loo_predicted_peak_21d_change_pct"] = loo
    predictions["loo_error_pct_points"] = (
        predictions["loo_predicted_peak_21d_change_pct"]
        - predictions["brent_peak_21d_change_pct"]
    )
    predictions["observed_21d_direction"] = np.sign(
        predictions["brent_21d_change"].fillna(0.0)
    )
    predictions["predicted_direction"] = np.sign(
        predictions["loo_predicted_peak_21d_change_pct"]
    )
    predictions["direction_match"] = (
        predictions["observed_21d_direction"]
        == predictions["predicted_direction"]
    )

    coefficient_rows = []
    for position, feature in enumerate(FEATURE_COLUMNS):
        mean = float(bootstrap[:, position].mean())
        standard_deviation = float(bootstrap[:, position].std(ddof=1))
        coefficient_rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "prior_coefficient": PRIOR_COEFFICIENTS[position],
                "calibrated_coefficient": coefficients[position],
                "bootstrap_p05": np.quantile(bootstrap[:, position], 0.05),
                "bootstrap_p50": np.quantile(bootstrap[:, position], 0.50),
                "bootstrap_p95": np.quantile(bootstrap[:, position], 0.95),
                "bootstrap_std": standard_deviation,
                "coefficient_cv": (
                    standard_deviation / abs(mean) if abs(mean) > 1e-9 else np.inf
                ),
                "ridge_strength": ridge_strength,
                "bootstrap_draws": bootstrap_draws,
            }
        )
    coefficient_table = pd.DataFrame(coefficient_rows)

    errors = predictions["loo_error_pct_points"].to_numpy(dtype=float)
    mean_target = float(np.mean(y))
    metrics = pd.DataFrame(
        [
            {
                "eligible_events": len(eligible),
                "source_coverage_pct": (
                    100.0
                    * event_windows["source_url"].fillna("").str.strip().ne("").mean()
                ),
                "brent_target_coverage_pct": (
                    100.0
                    * event_windows.loc[
                        event_windows["calibration_eligible"].astype(bool),
                        "brent_peak_21d_change_pct",
                    ].notna().mean()
                ),
                "loo_mae_pct_points": np.mean(np.abs(errors)),
                "loo_rmse_pct_points": np.sqrt(np.mean(errors**2)),
                "normalized_loo_mae": (
                    np.mean(np.abs(errors)) / mean_target
                    if mean_target > 1e-9
                    else np.inf
                ),
                "directional_agreement_pct": 100.0
                * predictions["direction_match"].mean(),
                "max_coefficient_cv": coefficient_table["coefficient_cv"].max(),
                "ridge_strength": ridge_strength,
                "bootstrap_draws": bootstrap_draws,
            }
        ]
    )
    return {
        "calibrated_coefficients": coefficient_table,
        "backtest_predictions": predictions,
        "calibration_metrics": metrics,
    }


def _calibration_gates(metrics: pd.Series) -> pd.DataFrame:
    definitions = [
        ("event_count", float(metrics["eligible_events"]), 8.0, ">="),
        ("source_coverage", float(metrics["source_coverage_pct"]), 100.0, ">="),
        (
            "brent_target_coverage",
            float(metrics["brent_target_coverage_pct"]),
            80.0,
            ">=",
        ),
        (
            "normalized_loo_mae",
            float(metrics["normalized_loo_mae"]),
            0.75,
            "<=",
        ),
        (
            "directional_agreement",
            float(metrics["directional_agreement_pct"]),
            60.0,
            ">=",
        ),
        (
            "coefficient_stability",
            float(metrics["max_coefficient_cv"]),
            0.75,
            "<=",
        ),
    ]
    rows = []
    for gate, value, threshold, operator in definitions:
        passed = value >= threshold if operator == ">=" else value <= threshold
        rows.append(
            {
                "gate": gate,
                "value": value,
                "threshold": threshold,
                "operator": operator,
                "passed": bool(passed),
            }
        )
    return pd.DataFrame(rows)


def build_historical_calibration(
    project_root: Path,
    seed: int = 20260727,
) -> dict[str, pd.DataFrame]:
    """Build event windows, calibration estimates, metrics, and promotion gates."""
    project_root = Path(project_root)
    events = pd.read_csv(project_root / "data" / "historical_event_registry.csv")
    fred = _load_fred_pivot(project_root / "data" / "live" / "fred_market_context.csv")
    windows = extract_event_windows(events, fred)
    calibration = calibrate_brent_transmission(windows, seed=seed)
    gates = _calibration_gates(calibration["calibration_metrics"].iloc[0])
    passed = bool(gates["passed"].all())
    summary = pd.DataFrame(
        [
            {
                "calibration_status": (
                    "Calibrated" if passed else "Backtested / preliminary"
                ),
                "all_promotion_gates_passed": passed,
                "gates_passed": int(gates["passed"].sum()),
                "gates_total": len(gates),
                "coefficients_activated": False,
                "activation_policy": (
                    "Historical estimates are diagnostic in version 1.2. "
                    "Structural scenario coefficients remain versioned priors."
                ),
                "target_definition": (
                    "Maximum nonnegative Brent percentage excursion during the "
                    "first 21 available observations after each event."
                ),
            }
        ]
    )
    return {
        "historical_event_windows": windows,
        **calibration,
        "calibration_gate_detail": gates,
        "calibration_summary": summary,
    }


def apply_calibration_status(
    model_status: pd.DataFrame,
    calibration_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Merge historical backtesting status into the live-context status row."""
    status = model_status.copy()
    calibration = calibration_summary.iloc[0]
    label = str(calibration["calibration_status"])
    status["calibration_status"] = label
    status["effective_model_status"] = (
        status["data_status"].astype(str) + " context / " + label.lower()
    )
    status["status_reason"] = (
        "Live context was audited and historical Brent transmission was "
        "backtested. Calibrated coefficients remain diagnostic and are not "
        "activated in version 1.2."
    )
    return status
