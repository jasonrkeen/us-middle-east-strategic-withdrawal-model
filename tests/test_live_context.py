from pathlib import Path

import numpy as np
import pandas as pd

from src.live_context import build_live_context


def _write_live_fixtures(project_root: Path) -> None:
    live_dir = project_root / "data" / "live"
    live_dir.mkdir(parents=True)
    dates = pd.bdate_range("2026-01-02", "2026-07-29")
    series_values = {
        "DCOILBRENTEU": np.linspace(72.0, 92.0, len(dates)),
        "VIXCLS": np.linspace(15.0, 19.0, len(dates)),
        "T10YIE": np.linspace(2.0, 2.3, len(dates)),
        "DTWEXBGS": np.linspace(118.0, 121.0, len(dates)),
    }
    fred = pd.concat(
        [
            pd.DataFrame(
                {"date": dates, "value": values, "series": series_id}
            )
            for series_id, values in series_values.items()
        ],
        ignore_index=True,
    )
    fred.to_csv(live_dir / "fred_market_context.csv", index=False)

    indicators = [
        "real_gdp_growth_pct",
        "inflation_pct",
        "current_account_pct_gdp",
        "military_spending_pct_gdp",
    ]
    rows = []
    for country, country_code in [("Alpha", "ALP"), ("Beta", "BET")]:
        for year in [2024, 2025]:
            for position, indicator in enumerate(indicators):
                rows.append(
                    {
                        "country": country,
                        "country_code": country_code,
                        "year": year,
                        "indicator": indicator,
                        "value": float(year - 2020 + position),
                        "source": "World Bank API",
                    }
                )
    pd.DataFrame(rows).to_csv(
        live_dir / "world_bank_context.csv", index=False
    )


def _economic_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": "baseline",
                "scenario": "Baseline",
                "metric": "brent_price_change_pct",
                "p05": -5.0,
                "median": 0.0,
                "p95": 5.0,
            },
            {
                "scenario_id": "stress",
                "scenario": "Stress",
                "metric": "brent_price_change_pct",
                "p05": 10.0,
                "median": 20.0,
                "p95": 30.0,
            },
        ]
    )


def test_live_context_builds_quality_and_market_outputs(tmp_path):
    _write_live_fixtures(tmp_path)
    outputs = build_live_context(
        tmp_path,
        economic_impacts=_economic_fixture(),
        as_of=pd.Timestamp("2026-07-29"),
    )

    status = outputs["model_status"].iloc[0]
    components = outputs["market_stress_components"]
    assert status["data_status"] == "Live"
    assert 0 <= status["data_quality_score"] <= 100
    assert 0 <= status["market_stress_score"] <= 100
    assert np.isclose(components["effective_weight"].sum(), 1.0)
    assert len(outputs["live_brent_scenario_overlay"]) == 2


def test_live_brent_overlay_uses_latest_observation(tmp_path):
    _write_live_fixtures(tmp_path)
    outputs = build_live_context(
        tmp_path,
        economic_impacts=_economic_fixture(),
        as_of=pd.Timestamp("2026-07-29"),
    )
    overlay = outputs["live_brent_scenario_overlay"].set_index("scenario_id")
    latest = outputs["market_context_summary"]["brent_latest"].iloc[0]
    assert np.isclose(overlay.loc["baseline", "implied_brent_median"], latest)
    assert np.isclose(
        overlay.loc["stress", "implied_brent_median"], latest * 1.20
    )


def test_missing_live_files_use_versioned_fallback(tmp_path):
    outputs = build_live_context(tmp_path)
    status = outputs["model_status"].iloc[0]
    assert status["data_status"] == "Versioned fallback"
    assert status["calibration_status"] == "Preliminary"

