# Copilot instructions: U.S. Middle East Strategic Withdrawal Response Model

## Purpose
Transparent counterfactual model of actor responses and economic stress under alternative U.S. military withdrawal scenarios in the Middle East.

## Tech stack
- Python; dependencies in `requirements.txt`
- Key libraries: `numpy`, `pandas`, `matplotlib`, `reportlab`, `requests`, `pytest`

## Folder structure
- `data/` (8 files)
- `output/` (8 files)
- `src/` (6 files)
- `tests/` (3 files)
- `CHANGELOG.md`
- `README.md`
- `main.py`
- `requirements.txt`

## Setup, run, test
Commands documented in README.md (referenced files exist):
```bash
python -m venv .venv
python -m pip install -r requirements.txt
python main.py
python3 -m venv .venv
python main.py --runs 10000
python main.py --refresh-data
pytest -q
```
- Tests: 3 Python test file(s) under `tests/`; framework appears to be pytest. Add/update tests with any logic change.

## Conventions observed
- Functions use type hints; keep annotating new code.
- Uses `dataclasses` for structured config/results.
- Random seeds are set for reproducibility (Monte Carlo/simulation); keep seeded, deterministic runs.
- Public data sources referenced: ACLED, FRED, EIA, World Bank, UN Comtrade, SIPRI.
- Write paths relative to the repo root (no absolute local paths).

## Output artifacts
- A PDF report is produced with `reportlab`; keep it in sync with model changes.
- Charts are generated with matplotlib.

## Guardrails
- Never commit secrets, API keys, tokens, or `.env` files.
- Do not commit large raw datasets or binaries; keep data fetched or documented with its source.
- Keep analysis source-backed: cite data sources and assumptions; label scenario/forecast outputs as model estimates, not facts.
- Preserve reproducibility: no unseeded randomness, no hidden manual steps.
