from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting import (
    PRIOR_COEFFICIENTS,
    _calibration_gates,
    calibrate_brent_transmission,
    extract_event_windows,
    fit_ridge_to_prior,
)
from src.model import BRENT_STRESS_COEFFICIENTS


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": f"event_{position}",
                "event": f"Event {position}",
                "event_date": pd.Timestamp("2020-01-06")
                + pd.offsets.BDay(position * 30),
                "security_shock": 0.2 + 0.07 * position,
                "energy_shock": 0.8 - 0.04 * position,
                "maritime_shock": 0.1 + 0.06 * position,
                "calibration_eligible": 1,
                "source_name": "Test source",
                "source_url": "https://example.test/event",
            }
            for position in range(8)
        ]
    )


def _fred() -> pd.DataFrame:
    dates = pd.bdate_range("2019-12-02", "2021-06-30")
    base = np.linspace(60.0, 95.0, len(dates))
    return pd.DataFrame(
        {
            "DCOILBRENTEU": base,
            "VIXCLS": np.linspace(15.0, 23.0, len(dates)),
            "T10YIE": np.linspace(1.8, 2.3, len(dates)),
            "DTWEXBGS": np.linspace(110.0, 115.0, len(dates)),
        },
        index=dates,
    )


def test_event_windows_use_last_pre_event_observation():
    events = _events().iloc[:1]
    fred = _fred()
    windows = extract_event_windows(events, fred)
    event_date = events["event_date"].iloc[0]
    expected_date = fred.index[fred.index < event_date][-1]
    assert windows["brent_pre_date"].iloc[0] == expected_date
    assert windows["brent_peak_21d_change_pct"].notna().all()


def test_ridge_fit_is_nonnegative_and_shrunk_to_prior():
    x = np.eye(3)
    y = np.array([-100.0, 5.0, 10.0])
    coefficients = fit_ridge_to_prior(x, y, ridge_strength=5.0)
    assert np.all(coefficients >= 0)
    assert np.linalg.norm(coefficients - PRIOR_COEFFICIENTS) < np.linalg.norm(
        y - PRIOR_COEFFICIENTS
    )
    assert np.allclose(
        PRIOR_COEFFICIENTS,
        [
            BRENT_STRESS_COEFFICIENTS["security_tension"],
            BRENT_STRESS_COEFFICIENTS["energy_disruption"],
            BRENT_STRESS_COEFFICIENTS["maritime_risk"],
        ],
    )


def test_calibration_is_deterministic():
    windows = extract_event_windows(_events(), _fred())
    first = calibrate_brent_transmission(windows, seed=44, bootstrap_draws=100)
    second = calibrate_brent_transmission(windows, seed=44, bootstrap_draws=100)
    a = first["calibrated_coefficients"]
    b = second["calibrated_coefficients"]
    assert np.allclose(a["bootstrap_p05"], b["bootstrap_p05"])
    assert len(first["backtest_predictions"]) == 8


def test_promotion_gate_reports_each_threshold():
    metrics = pd.Series(
        {
            "eligible_events": 8,
            "source_coverage_pct": 100.0,
            "brent_target_coverage_pct": 100.0,
            "normalized_loo_mae": 0.50,
            "directional_agreement_pct": 75.0,
            "max_coefficient_cv": 0.40,
        }
    )
    gates = _calibration_gates(metrics)
    assert len(gates) == 6
    assert gates["passed"].all()
