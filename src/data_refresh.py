"""Optional public-data refresh utilities.

The core model remains reproducible without network access. Refreshed data are
contextual inputs and do not silently recalibrate the model's expert-assumption
coefficients.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


WORLD_BANK_COUNTRIES = [
    "SAU",
    "ARE",
    "QAT",
    "ISR",
    "IRN",
    "IRQ",
    "JOR",
    "EGY",
    "TUR",
    "CHN",
    "IND",
    "JPN",
    "KOR",
]

WORLD_BANK_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "real_gdp_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "BN.CAB.XOKA.GD.ZS": "current_account_pct_gdp",
    "MS.MIL.XPND.GD.ZS": "military_spending_pct_gdp",
}

FRED_SERIES = ["DCOILBRENTEU", "VIXCLS", "T10YIE", "DTWEXBGS"]


def _fetch_world_bank(timeout: int = 30) -> pd.DataFrame:
    rows: list[dict] = []
    countries = ";".join(WORLD_BANK_COUNTRIES)
    for indicator, label in WORLD_BANK_INDICATORS.items():
        url = (
            f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
            "?format=json&per_page=2000&date=2018:2026"
        )
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        for record in payload[1] or []:
            rows.append(
                {
                    "country": record.get("country", {}).get("value"),
                    "country_code": record.get("countryiso3code"),
                    "year": record.get("date"),
                    "indicator": label,
                    "value": record.get("value"),
                    "source": "World Bank API",
                }
            )
    return pd.DataFrame(rows)


def _fetch_fred(timeout: int = 30) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        frame.columns = ["date", "value"]
        frame["series"] = series
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def refresh_context_data(project_root: Path) -> None:
    live_dir = project_root / "data" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    try:
        world_bank = _fetch_world_bank()
        world_bank.to_csv(live_dir / "world_bank_context.csv", index=False)
        print(f"      World Bank: {len(world_bank):,} observations")
    except Exception as exc:  # network and upstream formats are outside model control
        print(f"      World Bank refresh unavailable: {exc}")

    try:
        fred = _fetch_fred()
        fred.to_csv(live_dir / "fred_market_context.csv", index=False)
        print(f"      FRED: {len(fred):,} observations")
    except Exception as exc:
        print(f"      FRED refresh unavailable: {exc}")

