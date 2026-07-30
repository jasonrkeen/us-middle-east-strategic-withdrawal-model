# Methodology and Model Governance

## Purpose

The U.S. Middle East Strategic Withdrawal Response Model is a conditional
counterfactual decision-support tool. It estimates how selected actors might
respond to alternative U.S. posture changes and how those choices could
transmit into energy, shipping, macroeconomic, and sovereign-risk stress.

It is not a forecast, operational plan, targeting tool, or statement of U.S.
policy.

## Scenario construction

The Withdrawal Shock Index (WSI) is:

`WSI = 0.35(F) + 0.20(S) + 0.25(1-R) + 0.20(1-G)`

where:

- `F` is the share of regional force posture removed;
- `S` is withdrawal speed;
- `R` is residual military capability;
- `G` is the credibility of remaining security guarantees.

The related Deterrence Gap and Alliance Ambiguity indices use separately
documented weights in `src/model.py`. All scenario parameters are stored in
`data/scenarios.csv`.

## Actor action probabilities

Each actor-action utility combines:

- the actor's security exposure and the action's security value;
- opportunism and the opening created by the deterrence gap;
- diplomatic flexibility and alliance ambiguity;
- demand for substitute security partnerships;
- economic capacity constraints;
- escalation costs, restraint, and residual U.S. deterrence.

Utilities are converted to probabilities with a softmax transformation,
including a separate maintain-current-posture option. Probabilities are
conditional on the configured assumptions; they are not empirical frequencies.

## Dynamic simulation

Each actor selects one action or maintains posture in each quarter. The selected
actions alter five bounded strategic states:

1. security tension;
2. maritime risk;
3. arms-race pressure;
4. diplomatic hedging;
5. energy disruption.

States are persistent but partially mean-reverting. Random innovations represent
unobserved political, leadership, market, and conflict variation. Pathways are
classified from the mean state during the final simulation year.

The most severe configured scenario has a WSI of
0.985; its output should be interpreted as a
stress test, not as the most likely U.S. policy.

## Economic transmission

Strategic-state differences relative to the simulated baseline are translated
into distributions for Brent prices, shipping costs, global inflation, global
GDP, MENA non-oil GDP, military expenditure, and sovereign spreads.

Version 1.2 backtests the Brent translation on 10
sourced regional-security, energy-infrastructure, and maritime events. For each
event it compares the last pre-event observation with 5-, 21-, and 63-observation
market windows. The calibration target is the maximum nonnegative Brent price
excursion during the first 21 available observations.

The coefficient estimate minimizes squared event-window error plus a quadratic
penalty toward the versioned structural prior. Leave-one-event-out predictions
produce a mean absolute error of 19.9 percentage
points and normalized MAE of 1.07. Bootstrap
resampling measures coefficient stability. These diagnostics do not establish
causality because events are heterogeneous and confounded by wars, sanctions,
production decisions, demand conditions, and policy responses.

## Live economic context

The observational market baseline is dated July 29, 2026. Version 1.2 audits
four FRED series and four World Bank indicators before producing any live
context output.

The Live Market Stress Index is a weighted historical-percentile score:

- Brent price level: 15%
- absolute 21-day Brent move: 15%
- 21-day annualized Brent volatility: 20%
- VIX level: 20%
- absolute 10-year breakeven deviation from 2%: 15%
- absolute 21-day broad-dollar move: 15%

The current score is 63.6/100
(Elevated). The data-quality score is
96.4/100. Live market levels create an overlay,
including implied Brent price ranges, but do not silently alter the strategic
state-transition coefficients.

## Model-status gating

The effective status is **Live context / backtested / preliminary**. Live data are
classified as Live, Degraded, or Versioned fallback based on series coverage,
missingness, history, and publication freshness. The historical calibration
passed 4 of
6 promotion gates. Version 1.2 therefore keeps
the historical coefficients diagnostic and leaves the versioned scenario
coefficients active.

## Baseline and data provenance

The live baseline is dated July 29, 2026. Public sources are recorded in
`data/source_register.csv`. Versioned exposure scores are expert assumptions
informed by those sources. World Bank and FRED refreshes are stored separately
and never silently overwrite model assumptions.

## Validation controls

- Unit-interval validation for bounded input parameters
- Referential-integrity checks for actor-action pairs
- Reproducible random seeds
- Probability-sum checks
- Scenario monotonicity tests
- Bounded strategic states
- Minimum Monte Carlo and time-horizon requirements
- Separate baseline and counterfactual distributions
- Last-pre-event and fixed-observation-window event extraction
- Leave-one-event-out backtest predictions
- Deterministic coefficient bootstrap
- Six explicit calibration-promotion gates
- FRED duplicate-key, required-series, freshness, and completeness checks
- World Bank duplicate-key, country, indicator, and publication-lag checks
- Explicit Live, Degraded, and Versioned fallback gates

## Limitations

- Public reporting does not fully reveal military capabilities, intentions, or
  private diplomatic commitments.
- State and non-state actors are internally heterogeneous.
- Actor choices are not statistically independent in reality.
- Historical event windows are small, heterogeneous, and not causal estimates.
- Extreme escalation, leadership change, and nuclear use are not explicitly
  simulated.
- Scores should be reviewed by regional, military, energy, and macroeconomic
  specialists before decision use.

## Recommended validation roadmap

1. Conduct structured expert elicitation and record inter-rater disagreement.
2. Add production-loss and shipping-cost controls to reduce event heterogeneity.
3. Estimate shipping elasticities with PortWatch and freight-rate observations.
4. Backtest qualitative actor-action rankings against posture-change episodes.
5. Add quarterly Bayesian updating from ACLED, PortWatch, EIA, and macro data.
6. Publish calibration vintages and retain all prior assumption files.
