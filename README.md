# U.S. Middle East Strategic Withdrawal Response Model

A transparent counterfactual model of possible actor responses and economic
transmission if the United States materially reduces or ends its military and
security posture in the Middle East.

The model combines:

- five defined U.S. posture scenarios;
- ten state, external-power, and non-state actor groups;
- twelve strategic action classes;
- a probabilistic actor-choice model;
- a quarterly five-year Monte Carlo state-transition simulation;
- energy, shipping, inflation, growth, defense-spending, and sovereign-risk
  stress translation;
- country and regional vulnerability scoring;
- charts, data tables, methodology documentation, and an executive PDF.

> **Important:** This is a conditional scenario model, not a forecast,
> operational plan, targeting tool, or statement of U.S. policy. Version 1.2
> backtests selected economic-transmission assumptions but does not claim
> causal identification.

## Version 1.2 historical backtesting

Version 1.2 adds:

- a sourced registry of ten regional-security, energy, and maritime events;
- reproducible 5-, 21-, and 63-observation market-event windows;
- ridge-to-prior calibration of the Brent stress coefficients;
- leave-one-event-out prediction errors and directional diagnostics;
- deterministic 1,000-draw coefficient bootstrap intervals;
- six explicit promotion gates;
- a dedicated historical-calibration chart and executive-brief page.

The historical estimates are diagnostic in version 1.2. Because four of six
promotion gates passed, the model retains the versioned structural coefficients.
No backtest output silently changes actor utilities, strategic transitions, or
scenario results.

## Research question

If the United States changes its military and security posture in the Middle
East, which configured actor responses become more competitive, how might those
actions interact over time, and how could the resulting strategic states affect
energy, shipping, macroeconomic, and sovereign-risk conditions?

## Scenarios

| Scenario | Interpretation |
|---|---|
| Current posture baseline | July 2026 reference posture |
| Limited drawdown | Selected force and facility reductions |
| Combat-force withdrawal | Combat forces leave; some naval, intelligence, arms, and guarantee functions remain |
| Full military withdrawal | Most forces, bases, and operational support leave |
| Strategic disengagement | Military withdrawal plus sharply reduced guarantees, assistance, and diplomacy |

The model separates force reduction from withdrawal speed, residual capability,
and security-guarantee credibility. This prevents the word "withdrawal" from
acting as an analytically ambiguous binary switch.

## How the model works

### 1. Withdrawal Shock Index

The primary scenario index is:

```text
WSI = 0.35 * force_reduction
    + 0.20 * withdrawal_speed
    + 0.25 * (1 - residual_capability)
    + 0.20 * (1 - security_guarantee)
```

Separate deterrence-gap and alliance-ambiguity indices capture related but
different mechanisms.

### 2. Actor choice

Each actor has versioned scores for:

- security exposure;
- economic capacity;
- opportunism;
- dependence on U.S. security;
- restraint;
- diplomatic flexibility.

Each action has security and opportunity value, capacity requirements,
escalation and economic costs, and documented effects on the strategic state.
The model calculates conditional utilities and converts them to probabilities
with a softmax transformation. A maintain-current-posture option is always
included.

### 3. Strategic state transitions

In each simulated quarter, every actor selects an action or maintains posture.
Actions update five bounded, persistent state variables:

- security tension;
- maritime risk;
- arms-race pressure;
- diplomatic hedging;
- energy disruption.

Monte Carlo variation represents unobserved political, leadership, market, and
conflict shocks. Final-year states are assigned to one of six pathway classes:
baseline continuity, managed hedging, incremental adaptation, competitive
rearmament, maritime disruption, or regional escalation.

### 4. Economic transmission

Differences from the simulated baseline are translated into distributions for:

- Brent crude price changes;
- shipping-cost changes;
- global inflation changes;
- global GDP changes;
- MENA non-oil GDP changes;
- regional military-spending changes;
- regional sovereign-spread changes.

These are structural stress translations. The Brent coefficients are compared
with historical event windows, but remain intentionally labeled as scenario
estimates rather than econometric forecasts.

### 5. Historical backtesting and calibration governance

`data/historical_event_registry.csv` records event dates, source URLs, and three
bounded shock-severity assumptions. Event dates are sourced facts; severity
scores are expert-coded model inputs.

For each event, the pipeline extracts the last available pre-event observation
and 5-, 21-, and 63-observation changes in Brent, VIX, 10-year breakeven
inflation, and the broad dollar index. The calibration target is the maximum
nonnegative Brent excursion during the first 21 available observations.

The estimator minimizes squared error plus a penalty toward the versioned prior.
Promotion requires passing all six gates covering event count, source coverage,
target coverage, leave-one-out error, directional agreement, and coefficient
stability. The version 1.2 sample passes four gates; fitted coefficients are
reported but not activated.

## Installation

Python 3.10 or newer is recommended.

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## Commands

Run the default model:

```bash
python main.py
```

Run 10,000 paths per scenario:

```bash
python main.py --runs 10000
```

Attempt optional World Bank and FRED context refreshes:

```bash
python main.py --refresh-data
```

The refresh operation writes separate files under `data/live/`. The pipeline
audits those files and reports their status, but it does not silently change
the versioned scenario or exposure assumptions.

Run validation tests:

```bash
pytest -q
```

## Output

The pipeline writes:

```text
output/
├── charts/
│   ├── actor_action_probability_heatmap.png
│   ├── economic_stress_ranges.png
│   ├── historical_calibration_diagnostics.png
│   ├── live_market_stress_components.png
│   ├── scenario_strategic_stress.png
│   └── strategic_disengagement_vulnerability.png
├── data/
│   ├── action_frequency.csv
│   ├── actor_action_probabilities.csv
│   ├── backtest_predictions.csv
│   ├── calibrated_coefficients.csv
│   ├── calibration_gate_detail.csv
│   ├── calibration_metrics.csv
│   ├── calibration_summary.csv
│   ├── country_vulnerability.csv
│   ├── data_freshness.csv
│   ├── economic_impacts.csv
│   ├── historical_event_windows.csv
│   ├── live_brent_scenario_overlay.csv
│   ├── macro_data_coverage.csv
│   ├── macro_latest_snapshot.csv
│   ├── market_context_summary.csv
│   ├── market_stress_components.csv
│   ├── model_status.csv
│   ├── pathway_probabilities.csv
│   ├── scenario_definitions.csv
│   ├── scenario_summary.csv
│   └── sensitivity_analysis.csv
├── methodology_and_governance.md
└── pdf/
    └── us_middle_east_withdrawal_executive_brief.pdf
```

## Baseline sources

The source register is in `data/source_register.csv`. Principal sources include:

- [U.S. EIA world oil transit chokepoints](https://www.eia.gov/international/analysis/special-topics/world_oil_transit_Chokepoints)
- [U.S. EIA Short-Term Energy Outlook](https://www.eia.gov/outlooks/steo/report/global_oil.php)
- [IMF PortWatch data and methodology](https://portwatch.imf.org/pages/data-and-methodology)
- [IMF World Economic Outlook data](https://data.imf.org/en/datasets/IMF.RES%3AWEO)
- [World Bank Open Data](https://data.worldbank.org/)
- [SIPRI Military Expenditure Database](https://www.sipri.org/databases/milex)
- [FRED](https://fred.stlouisfed.org/)
- ACLED and UN Comtrade as optional validation and extension sources

## Repository structure

```text
.
├── data/
│   ├── actions.csv
│   ├── actor_action_affinity.csv
│   ├── actors.csv
│   ├── country_exposures.csv
│   ├── historical_analogues.csv
│   ├── scenarios.csv
│   └── source_register.csv
├── src/
│   ├── data_refresh.py
│   ├── live_context.py
│   ├── model.py
│   └── reporting.py
├── tests/
│   └── test_model.py
├── main.py
├── requirements.txt
└── README.md
```

## Interpretation rules

1. Compare scenarios; do not treat a score as a standalone fact.
2. Report the median and uncertainty interval together.
3. Keep scenario likelihood separate from consequence severity.
4. Treat action probabilities as conditional rankings within a configured
   choice set.
5. Do not infer intent from modeled capability or opportunity.
6. Record every assumption revision and calibration vintage.

## Model-status interpretation

- **Live context:** required live files pass coverage and freshness gates.
- **Degraded context:** data are usable but one or more quality gates need
  attention.
- **Versioned fallback:** live files are missing or materially incomplete.
- **Preliminary calibration:** counterfactual coefficients have not yet passed
  historical backtesting and stability thresholds.

## Validation roadmap

- Structured expert elicitation with documented disagreement
- Historical analogue review and qualitative backtesting
- Event-study or local-projection calibration for energy and shipping shocks
- Bayesian quarterly updates from public conflict and economic indicators
- Out-of-sample tracking of actor-action rankings
- Versioned calibration reports and model cards

## License

MIT License. See `LICENSE`.
