"""Live economic context, source-freshness, and model-status controls.

Live data provide a current observational baseline. They do not silently
recalibrate the structural counterfactual coefficients in ``src/model.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_VERSION = "1.2.0"
EXPECTED_FRED_SERIES = {
    "DCOILBRENTEU": "Brent crude spot price",
    "VIXCLS": "CBOE VIX",
    "T10YIE": "10-year breakeven inflation",
    "DTWEXBGS": "Nominal broad U.S. dollar index",
}
EXPECTED_WB_INDICATORS = {
    "real_gdp_growth_pct": "Real GDP growth",
    "inflation_pct": "Consumer-price inflation",
    "current_account_pct_gdp": "Current account balance",
    "military_spending_pct_gdp": "Military spending",
}


def _percentile_rank(history: pd.Series, value: float) -> float:
    clean = pd.to_numeric(history, errors="coerce").dropna()
    if clean.empty or pd.isna(value):
        return np.nan
    return float(100.0 * (clean <= value).mean())


def _stress_level(score: float) -> str:
    if pd.isna(score):
        return "Unavailable"
    if score < 25:
        return "Low"
    if score < 45:
        return "Guarded"
    if score < 65:
        return "Elevated"
    if score < 80:
        return "High"
    return "Severe"


def _empty_live_context(reason: str) -> dict[str, pd.DataFrame]:
    status = pd.DataFrame(
        [
            {
                "model_version": MODEL_VERSION,
                "data_status": "Versioned fallback",
                "data_quality_score": np.nan,
                "market_stress_score": np.nan,
                "market_stress_level": "Unavailable",
                "calibration_status": "Preliminary",
                "effective_model_status": "Versioned fallback / preliminary calibration",
                "status_reason": reason,
            }
        ]
    )
    market_summary = pd.DataFrame(
        [
            {
                "as_of_date": pd.NaT,
                "market_stress_score": np.nan,
                "market_stress_level": "Unavailable",
                "brent_latest": np.nan,
                "brent_21d_change_pct": np.nan,
                "brent_21d_annualized_volatility_pct": np.nan,
                "vix_latest": np.nan,
                "breakeven_10y_latest_pct": np.nan,
                "broad_dollar_latest": np.nan,
                "broad_dollar_21d_change_pct": np.nan,
            }
        ]
    )
    return {
        "data_freshness": pd.DataFrame(),
        "market_stress_components": pd.DataFrame(),
        "market_context_summary": market_summary,
        "macro_data_coverage": pd.DataFrame(),
        "macro_latest_snapshot": pd.DataFrame(),
        "model_status": status,
    }


def _audit_fred(
    fred_path: Path, as_of: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    fred = pd.read_csv(fred_path)
    required = {"date", "value", "series"}
    missing = required - set(fred.columns)
    if missing:
        raise ValueError(f"FRED context is missing columns: {sorted(missing)}")

    fred["date"] = pd.to_datetime(fred["date"], errors="coerce")
    fred["value"] = pd.to_numeric(fred["value"], errors="coerce")
    fred = fred.dropna(subset=["date", "series"]).sort_values(["series", "date"])
    if fred.duplicated(["series", "date"]).any():
        raise ValueError("FRED context contains duplicate series/date keys")

    latest_observation = fred.loc[fred["value"].notna(), "date"].max()
    effective_as_of = pd.Timestamp(as_of or latest_observation).normalize()

    freshness_records: list[dict] = []
    for series_id, label in EXPECTED_FRED_SERIES.items():
        rows = fred[fred["series"].eq(series_id)]
        valid = rows.dropna(subset=["value"])
        latest_date = valid["date"].max() if not valid.empty else pd.NaT
        lag_days = (
            int((effective_as_of - latest_date).days)
            if pd.notna(latest_date)
            else np.nan
        )
        freshness_records.append(
            {
                "source": "FRED",
                "series_id": series_id,
                "series": label,
                "rows": len(rows),
                "non_null_rows": int(rows["value"].notna().sum()),
                "completeness_pct": (
                    100.0 * rows["value"].notna().mean() if len(rows) else 0.0
                ),
                "first_observation": valid["date"].min() if not valid.empty else pd.NaT,
                "latest_observation": latest_date,
                "freshness_lag_days": lag_days,
                "freshness_status": (
                    "Current"
                    if pd.notna(lag_days) and lag_days <= 7
                    else "Watch"
                    if pd.notna(lag_days) and lag_days <= 14
                    else "Stale or missing"
                ),
            }
        )
    freshness = pd.DataFrame(freshness_records)

    pivot = fred.pivot(index="date", columns="series", values="value").sort_index()
    trailing = pivot[pivot.index >= effective_as_of - pd.DateOffset(years=5)]
    brent = trailing.get("DCOILBRENTEU", pd.Series(dtype=float)).dropna()
    vix = trailing.get("VIXCLS", pd.Series(dtype=float)).dropna()
    breakeven = trailing.get("T10YIE", pd.Series(dtype=float)).dropna()
    dollar = trailing.get("DTWEXBGS", pd.Series(dtype=float)).dropna()

    brent_return_21d = brent.pct_change(21).dropna()
    brent_volatility = (
        brent.pct_change().rolling(21).std().mul(np.sqrt(252)).dropna()
    )
    dollar_return_21d = dollar.pct_change(21).dropna()
    breakeven_deviation = (breakeven - 2.0).abs()

    component_specs = [
        (
            "brent_level",
            "Brent price level",
            "DCOILBRENTEU",
            brent,
            brent.iloc[-1] if not brent.empty else np.nan,
            0.15,
        ),
        (
            "brent_21d_move",
            "Absolute 21-day Brent move",
            "DCOILBRENTEU",
            brent_return_21d.abs(),
            abs(brent_return_21d.iloc[-1]) if not brent_return_21d.empty else np.nan,
            0.15,
        ),
        (
            "brent_volatility",
            "21-day Brent volatility",
            "DCOILBRENTEU",
            brent_volatility,
            brent_volatility.iloc[-1] if not brent_volatility.empty else np.nan,
            0.20,
        ),
        (
            "vix_level",
            "VIX level",
            "VIXCLS",
            vix,
            vix.iloc[-1] if not vix.empty else np.nan,
            0.20,
        ),
        (
            "inflation_deviation",
            "Breakeven deviation from 2%",
            "T10YIE",
            breakeven_deviation,
            breakeven_deviation.iloc[-1] if not breakeven_deviation.empty else np.nan,
            0.15,
        ),
        (
            "dollar_21d_move",
            "Absolute 21-day dollar move",
            "DTWEXBGS",
            dollar_return_21d.abs(),
            abs(dollar_return_21d.iloc[-1]) if not dollar_return_21d.empty else np.nan,
            0.15,
        ),
    ]
    components = pd.DataFrame(
        [
            {
                "component_id": component_id,
                "component": label,
                "source_series": source_series,
                "latest_value": latest_value,
                "stress_percentile": _percentile_rank(history, latest_value),
                "weight": weight,
            }
            for component_id, label, source_series, history, latest_value, weight in component_specs
        ]
    )
    available = components["stress_percentile"].notna()
    available_weight = components.loc[available, "weight"].sum()
    components["effective_weight"] = np.where(
        available,
        components["weight"] / available_weight if available_weight else 0.0,
        0.0,
    )
    components["weighted_contribution"] = (
        components["stress_percentile"] * components["effective_weight"]
    )
    market_stress_score = float(components["weighted_contribution"].sum())

    market_summary = pd.DataFrame(
        [
            {
                "as_of_date": effective_as_of,
                "market_stress_score": market_stress_score,
                "market_stress_level": _stress_level(market_stress_score),
                "brent_latest": brent.iloc[-1] if not brent.empty else np.nan,
                "brent_21d_change_pct": (
                    100.0 * brent_return_21d.iloc[-1]
                    if not brent_return_21d.empty
                    else np.nan
                ),
                "brent_21d_annualized_volatility_pct": (
                    100.0 * brent_volatility.iloc[-1]
                    if not brent_volatility.empty
                    else np.nan
                ),
                "vix_latest": vix.iloc[-1] if not vix.empty else np.nan,
                "breakeven_10y_latest_pct": (
                    breakeven.iloc[-1] if not breakeven.empty else np.nan
                ),
                "broad_dollar_latest": dollar.iloc[-1] if not dollar.empty else np.nan,
                "broad_dollar_21d_change_pct": (
                    100.0 * dollar_return_21d.iloc[-1]
                    if not dollar_return_21d.empty
                    else np.nan
                ),
            }
        ]
    )

    coverage_score = (
        100.0
        * freshness["series_id"].isin(
            fred.loc[fred["value"].notna(), "series"].unique()
        ).mean()
    )
    completeness_score = float(freshness["completeness_pct"].mean())
    freshness_score = float(
        np.mean(
            [
                max(0.0, 100.0 - 5.0 * lag)
                if pd.notna(lag)
                else 0.0
                for lag in freshness["freshness_lag_days"]
            ]
        )
    )
    history_score = float(
        np.mean(
            [
                min(
                    100.0,
                    max(
                        0.0,
                        100.0
                        * (
                            (row.latest_observation - row.first_observation).days
                            / (365.25 * 5)
                        ),
                    ),
                )
                if pd.notna(row.latest_observation)
                and pd.notna(row.first_observation)
                else 0.0
                for row in freshness.itertuples(index=False)
            ]
        )
    )
    fred_quality = (
        0.35 * coverage_score
        + 0.30 * completeness_score
        + 0.25 * freshness_score
        + 0.10 * history_score
    )
    return freshness, components, market_summary, float(fred_quality)


def _audit_world_bank(
    world_bank_path: Path, as_of: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    world_bank = pd.read_csv(world_bank_path)
    required = {"country", "country_code", "year", "indicator", "value", "source"}
    missing = required - set(world_bank.columns)
    if missing:
        raise ValueError(
            f"World Bank context is missing columns: {sorted(missing)}"
        )

    world_bank["year"] = pd.to_numeric(world_bank["year"], errors="coerce")
    world_bank["value"] = pd.to_numeric(world_bank["value"], errors="coerce")
    if world_bank.duplicated(["country_code", "year", "indicator"]).any():
        raise ValueError("World Bank context contains duplicate country/year/indicator keys")

    countries = sorted(world_bank["country_code"].dropna().unique())
    expected_country_count = len(countries)
    coverage_records: list[dict] = []
    for indicator_id, label in EXPECTED_WB_INDICATORS.items():
        rows = world_bank[world_bank["indicator"].eq(indicator_id)]
        valid = rows.dropna(subset=["value", "year"])
        latest_year = int(valid["year"].max()) if not valid.empty else np.nan
        allowed_lag = 2 if indicator_id == "military_spending_pct_gdp" else 1
        publication_lag = (
            int(as_of.year - latest_year) if pd.notna(latest_year) else np.nan
        )
        currency_score = (
            max(0.0, 100.0 - 25.0 * max(0, publication_lag - allowed_lag))
            if pd.notna(publication_lag)
            else 0.0
        )
        coverage_records.append(
            {
                "source": "World Bank",
                "indicator_id": indicator_id,
                "indicator": label,
                "rows": len(rows),
                "non_null_rows": int(rows["value"].notna().sum()),
                "completeness_pct": (
                    100.0 * rows["value"].notna().mean() if len(rows) else 0.0
                ),
                "countries_present": int(rows["country_code"].nunique()),
                "expected_countries": expected_country_count,
                "latest_year": latest_year,
                "publication_lag_years": publication_lag,
                "publication_currency_score": currency_score,
            }
        )
    coverage = pd.DataFrame(coverage_records)

    latest = (
        world_bank.dropna(subset=["value", "year"])
        .sort_values("year")
        .groupby(["country_code", "indicator"], as_index=False)
        .tail(1)
    )
    value_snapshot = latest.pivot(
        index=["country_code", "country"],
        columns="indicator",
        values="value",
    )
    year_snapshot = latest.pivot(
        index=["country_code", "country"],
        columns="indicator",
        values="year",
    ).add_suffix("_year")
    snapshot = value_snapshot.join(year_snapshot).reset_index()

    expected_indicator_coverage = (
        100.0
        * coverage["indicator_id"].isin(world_bank["indicator"].unique()).mean()
    )
    country_coverage = (
        100.0
        * coverage["countries_present"].sum()
        / max(1, coverage["expected_countries"].sum())
    )
    structural_coverage_score = 0.5 * expected_indicator_coverage + 0.5 * country_coverage
    completeness_score = float(coverage["completeness_pct"].mean())
    publication_currency_score = float(
        coverage["publication_currency_score"].mean()
    )
    world_bank_quality = (
        0.35 * structural_coverage_score
        + 0.35 * completeness_score
        + 0.30 * publication_currency_score
    )
    return coverage, snapshot, float(world_bank_quality)


def _live_brent_overlay(
    market_summary: pd.DataFrame, economic_impacts: pd.DataFrame
) -> pd.DataFrame:
    brent_latest = float(market_summary["brent_latest"].iloc[0])
    frame = economic_impacts[
        economic_impacts["metric"].eq("brent_price_change_pct")
    ].copy()
    frame["live_brent_baseline"] = brent_latest
    frame["implied_brent_p05"] = brent_latest * (1.0 + frame["p05"] / 100.0)
    frame["implied_brent_median"] = brent_latest * (
        1.0 + frame["median"] / 100.0
    )
    frame["implied_brent_p95"] = brent_latest * (1.0 + frame["p95"] / 100.0)
    return frame[
        [
            "scenario_id",
            "scenario",
            "live_brent_baseline",
            "p05",
            "median",
            "p95",
            "implied_brent_p05",
            "implied_brent_median",
            "implied_brent_p95",
        ]
    ].rename(
        columns={
            "p05": "change_p05_pct",
            "median": "change_median_pct",
            "p95": "change_p95_pct",
        }
    )


def build_live_context(
    project_root: Path,
    economic_impacts: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    live_dir = Path(project_root) / "data" / "live"
    fred_path = live_dir / "fred_market_context.csv"
    world_bank_path = live_dir / "world_bank_context.csv"
    if not fred_path.exists() or not world_bank_path.exists():
        return _empty_live_context(
            "One or both live context files are unavailable; versioned model inputs remain active."
        )

    freshness, components, market_summary, fred_quality = _audit_fred(
        fred_path, as_of=as_of
    )
    effective_as_of = pd.Timestamp(market_summary["as_of_date"].iloc[0])
    macro_coverage, macro_snapshot, world_bank_quality = _audit_world_bank(
        world_bank_path, effective_as_of
    )
    overall_quality = 0.60 * fred_quality + 0.40 * world_bank_quality
    critical_fred_current = (
        freshness["series_id"].isin(EXPECTED_FRED_SERIES)
        & freshness["freshness_status"].isin(["Current", "Watch"])
    ).all()
    data_status = (
        "Live"
        if overall_quality >= 85 and critical_fred_current
        else "Degraded"
        if overall_quality >= 60
        else "Versioned fallback"
    )
    market_score = float(market_summary["market_stress_score"].iloc[0])
    status = pd.DataFrame(
        [
            {
                "model_version": MODEL_VERSION,
                "data_status": data_status,
                "data_quality_score": overall_quality,
                "fred_quality_score": fred_quality,
                "world_bank_quality_score": world_bank_quality,
                "market_stress_score": market_score,
                "market_stress_level": _stress_level(market_score),
                "calibration_status": "Preliminary",
                "effective_model_status": f"{data_status} context / preliminary calibration",
                "status_reason": (
                    "Live market and macroeconomic context verified; structural "
                    "counterfactual coefficients have not yet completed historical backtesting."
                ),
            }
        ]
    )
    result = {
        "data_freshness": freshness,
        "market_stress_components": components,
        "market_context_summary": market_summary,
        "macro_data_coverage": macro_coverage,
        "macro_latest_snapshot": macro_snapshot,
        "model_status": status,
    }
    if economic_impacts is not None and pd.notna(
        market_summary["brent_latest"].iloc[0]
    ):
        result["live_brent_scenario_overlay"] = _live_brent_overlay(
            market_summary, economic_impacts
        )
    return result
