"""Write model tables, charts, methodology notes, and executive PDF."""

from __future__ import annotations

from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = "#102A43"
BLUE = "#2D6A9F"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
RED = "#C44536"
LIGHT = "#EAF0F5"
TEXT = "#243B53"
FONT_REGULAR = "VeraModel"
FONT_BOLD = "VeraModel-Bold"
REPORTLAB_FONT_DIR = Path(reportlab.__file__).resolve().parent / "fonts"

pdfmetrics.registerFont(
    TTFont(FONT_REGULAR, str(REPORTLAB_FONT_DIR / "Vera.ttf"))
)
pdfmetrics.registerFont(
    TTFont(FONT_BOLD, str(REPORTLAB_FONT_DIR / "VeraBd.ttf"))
)
pdfmetrics.registerFontFamily(
    "VeraModel",
    normal=FONT_REGULAR,
    bold=FONT_BOLD,
    italic=FONT_REGULAR,
    boldItalic=FONT_BOLD,
)


METRIC_LABELS = {
    "brent_price_change_pct": "Brent price change (%)",
    "shipping_cost_change_pct": "Shipping cost change (%)",
    "global_inflation_change_pp": "Global inflation change (pp)",
    "global_gdp_change_pp": "Global GDP change (pp)",
    "mena_non_oil_gdp_change_pp": "MENA non-oil GDP change (pp)",
    "regional_military_spending_change_pct": "Regional military spending change (%)",
    "regional_sovereign_spread_change_bps": "Regional sovereign spread change (bps)",
}


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#9FB3C8",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _write_tables(data_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False)


def _scenario_stress_chart(
    chart_dir: Path, summary: pd.DataFrame
) -> Path:
    labels = summary["scenario"].str.replace(" withdrawal", "\nwithdrawal").tolist()
    x = np.arange(len(summary))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.1))
    ax.bar(
        x - width,
        100 * summary["median_security_tension"],
        width,
        label="Security tension",
        color=RED,
    )
    ax.bar(
        x,
        100 * summary["median_maritime_risk"],
        width,
        label="Maritime risk",
        color=AMBER,
    )
    ax.bar(
        x + width,
        100 * summary["median_arms_race"],
        width,
        label="Arms-race pressure",
        color=BLUE,
    )
    ax.set_title("Strategic stress rises nonlinearly with U.S. disengagement")
    ax.set_ylabel("Median simulated state (0-100)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = chart_dir / "scenario_strategic_stress.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _economic_range_chart(
    chart_dir: Path, economic: pd.DataFrame
) -> Path:
    subset = economic[
        economic["metric"].isin(
            [
                "brent_price_change_pct",
                "shipping_cost_change_pct",
                "regional_military_spending_change_pct",
            ]
        )
        & ~economic["scenario_id"].eq("baseline")
    ].copy()
    metric_order = [
        "brent_price_change_pct",
        "shipping_cost_change_pct",
        "regional_military_spending_change_pct",
    ]
    scenario_order = [
        "limited_drawdown",
        "combat_exit",
        "full_military_withdrawal",
        "strategic_disengagement",
    ]
    colors_list = [TEAL, BLUE, AMBER, RED]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    for axis, metric in zip(axes, metric_order):
        frame = (
            subset[subset["metric"].eq(metric)]
            .set_index("scenario_id")
            .reindex(scenario_order)
        )
        y = np.arange(len(frame))
        med = frame["median"].to_numpy()
        low = med - frame["p05"].to_numpy()
        high = frame["p95"].to_numpy() - med
        axis.errorbar(
            med,
            y,
            xerr=[low, high],
            fmt="none",
            ecolor="#829AB1",
            elinewidth=2,
            capsize=4,
            zorder=1,
        )
        axis.scatter(med, y, c=colors_list, s=54, zorder=2)
        axis.axvline(0, color="#486581", linewidth=0.8)
        axis.set_title(METRIC_LABELS[metric].replace(" change", ""))
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(
        np.arange(4),
        ["Limited drawdown", "Combat exit", "Full withdrawal", "Disengagement"],
    )
    fig.suptitle("Five-year economic stress ranges relative to the baseline", y=1.02)
    fig.tight_layout()
    path = chart_dir / "economic_stress_ranges.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _action_heatmap(
    chart_dir: Path, action_probabilities: pd.DataFrame
) -> Path:
    frame = action_probabilities[
        action_probabilities["scenario_id"].eq("strategic_disengagement")
    ].copy()
    top_actions = (
        frame.groupby("action")["probability"]
        .mean()
        .nlargest(8)
        .index.tolist()
    )
    matrix = (
        frame[frame["action"].isin(top_actions)]
        .pivot(index="actor", columns="action", values="probability")
        .fillna(0)
    )
    matrix = matrix.loc[
        matrix.max(axis=1).sort_values(ascending=False).index,
        matrix.max(axis=0).sort_values(ascending=False).index,
    ]
    short_action_labels = {
        "Expand proxy or partner pressure": "Proxy pressure",
        "Seek alternative security partnerships": "Alternative security",
        "Expand diplomatic and economic influence": "Diplomatic expansion",
        "Increase defense spending": "Defense buildup",
        "Expand arms-market position": "Arms-market expansion",
        "Regional accommodation": "Accommodation",
        "Maritime coercion or disruption": "Maritime coercion",
        "Increase strategic stockpiles": "Strategic stockpiling",
    }

    fig, ax = plt.subplots(figsize=(11, 6.2))
    image = ax.imshow(matrix.to_numpy(), cmap="YlOrRd", vmin=0, vmax=0.45, aspect="auto")
    ax.set_xticks(
        np.arange(len(matrix.columns)),
        [
            short_action_labels.get(label, shorten(label, width=23, placeholder="..."))
            for label in matrix.columns
        ],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_title("Conditional action probabilities under strategic disengagement")
    ax.grid(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            if value >= 0.025:
                ax.text(
                    column,
                    row,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value > 0.25 else TEXT,
                )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78)
    colorbar.set_label("Probability")
    fig.tight_layout()
    path = chart_dir / "actor_action_probability_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _vulnerability_chart(
    chart_dir: Path, vulnerability: pd.DataFrame
) -> Path:
    frame = (
        vulnerability[
            vulnerability["scenario_id"].eq("strategic_disengagement")
        ]
        .nlargest(10, "vulnerability_score")
        .sort_values("vulnerability_score")
    )
    colors_by_value = [
        RED if value >= 70 else AMBER if value >= 55 else BLUE
        for value in frame["vulnerability_score"]
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.barh(
        frame["economy"],
        frame["vulnerability_score"],
        color=colors_by_value,
    )
    for y, value in enumerate(frame["vulnerability_score"]):
        ax.text(value + 1, y, f"{value:.1f}", va="center", fontsize=8)
    ax.set_title("Economic-security vulnerability under strategic disengagement")
    ax.set_xlabel("Composite vulnerability score (0-100)")
    ax.set_xlim(0, 100)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path = chart_dir / "strategic_disengagement_vulnerability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _market_stress_chart(
    chart_dir: Path, components: pd.DataFrame
) -> Path:
    frame = components.sort_values("stress_percentile").copy()
    colors_by_value = [
        RED if value >= 80 else AMBER if value >= 65 else BLUE
        for value in frame["stress_percentile"]
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.barh(
        frame["component"],
        frame["stress_percentile"],
        color=colors_by_value,
    )
    for y, value in enumerate(frame["stress_percentile"]):
        ax.text(value + 1.2, y, f"{value:.1f}", va="center", fontsize=8)
    ax.axvline(65, color=AMBER, linewidth=1.0, linestyle="--", alpha=0.85)
    ax.axvline(80, color=RED, linewidth=1.0, linestyle="--", alpha=0.85)
    ax.set_title("Live market-stress components relative to trailing five-year history")
    ax.set_xlabel("Historical percentile (0-100)")
    ax.set_xlim(0, 104)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    path = chart_dir / "live_market_stress_components.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _calibration_diagnostics_chart(
    chart_dir: Path,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    observed = predictions["brent_peak_21d_change_pct"].to_numpy(dtype=float)
    predicted = predictions["loo_predicted_peak_21d_change_pct"].to_numpy(dtype=float)
    upper = max(5.0, float(np.nanmax([observed.max(), predicted.max()]))) * 1.08
    axes[0].scatter(predicted, observed, color=BLUE, s=46, alpha=0.9)
    axes[0].plot([0, upper], [0, upper], color="#829AB1", linestyle="--", linewidth=1)
    axes[0].set_xlim(0, upper)
    axes[0].set_ylim(0, upper)
    axes[0].set_xlabel("Leave-one-out predicted peak change (%)")
    axes[0].set_ylabel("Observed 21-day peak change (%)")
    axes[0].set_title("Out-of-sample event predictions")
    axes[0].spines[["top", "right"]].set_visible(False)

    y = np.arange(len(coefficients))
    height = 0.34
    axes[1].barh(
        y + height / 2,
        coefficients["prior_coefficient"],
        height,
        label="Versioned prior",
        color="#9FB3C8",
    )
    axes[1].barh(
        y - height / 2,
        coefficients["calibrated_coefficient"],
        height,
        label="Historical estimate",
        color=TEAL,
    )
    axes[1].errorbar(
        coefficients["calibrated_coefficient"],
        y - height / 2,
        xerr=[
            coefficients["calibrated_coefficient"] - coefficients["bootstrap_p05"],
            coefficients["bootstrap_p95"] - coefficients["calibrated_coefficient"],
        ],
        fmt="none",
        ecolor=NAVY,
        capsize=3,
        linewidth=1,
    )
    axes[1].set_yticks(y, coefficients["feature_label"])
    axes[1].set_xlabel("Brent stress coefficient")
    axes[1].set_title("Prior and diagnostic coefficient estimates")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)

    fig.suptitle("Historical Brent-transmission calibration diagnostics", y=1.02)
    fig.tight_layout()
    path = chart_dir / "historical_calibration_diagnostics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def create_charts(
    chart_dir: Path, outputs: dict[str, pd.DataFrame]
) -> dict[str, Path]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    _set_plot_style()
    charts = {
        "strategic_stress": _scenario_stress_chart(
            chart_dir, outputs["scenario_summary"]
        ),
        "economic_ranges": _economic_range_chart(
            chart_dir, outputs["economic_impacts"]
        ),
        "action_heatmap": _action_heatmap(
            chart_dir, outputs["actor_action_probabilities"]
        ),
        "vulnerability": _vulnerability_chart(
            chart_dir, outputs["country_vulnerability"]
        ),
    }
    components = outputs.get("market_stress_components", pd.DataFrame())
    if not components.empty:
        charts["market_stress"] = _market_stress_chart(chart_dir, components)
    predictions = outputs.get("backtest_predictions", pd.DataFrame())
    coefficients = outputs.get("calibrated_coefficients", pd.DataFrame())
    if not predictions.empty and not coefficients.empty:
        charts["calibration_diagnostics"] = _calibration_diagnostics_chart(
            chart_dir, predictions, coefficients
        )
    return charts


def _methodology_markdown(
    project_root: Path,
    model,
    outputs: dict[str, pd.DataFrame],
) -> Path:
    path = project_root / "output" / "methodology_and_governance.md"
    scenario = outputs["scenario_summary"]
    strongest = scenario.iloc[-1]
    status = outputs["model_status"].iloc[0]
    market = outputs["market_context_summary"].iloc[0]
    calibration = outputs["calibration_summary"].iloc[0]
    metrics = outputs["calibration_metrics"].iloc[0]
    baseline_date = (
        pd.Timestamp(market["as_of_date"]).strftime("%B %d, %Y")
        if pd.notna(market["as_of_date"])
        else "versioned fallback"
    )
    content = f"""# Methodology and Model Governance

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
{strongest['withdrawal_shock_index']:.3f}; its output should be interpreted as a
stress test, not as the most likely U.S. policy.

## Economic transmission

Strategic-state differences relative to the simulated baseline are translated
into distributions for Brent prices, shipping costs, global inflation, global
GDP, MENA non-oil GDP, military expenditure, and sovereign spreads.

Version 1.2 backtests the Brent translation on {int(metrics['eligible_events'])}
sourced regional-security, energy-infrastructure, and maritime events. For each
event it compares the last pre-event observation with 5-, 21-, and 63-observation
market windows. The calibration target is the maximum nonnegative Brent price
excursion during the first 21 available observations.

The coefficient estimate minimizes squared event-window error plus a quadratic
penalty toward the versioned structural prior. Leave-one-event-out predictions
produce a mean absolute error of {metrics['loo_mae_pct_points']:.1f} percentage
points and normalized MAE of {metrics['normalized_loo_mae']:.2f}. Bootstrap
resampling measures coefficient stability. These diagnostics do not establish
causality because events are heterogeneous and confounded by wars, sanctions,
production decisions, demand conditions, and policy responses.

## Live economic context

The observational market baseline is dated {baseline_date}. Version 1.2 audits
four FRED series and four World Bank indicators before producing any live
context output.

The Live Market Stress Index is a weighted historical-percentile score:

- Brent price level: 15%
- absolute 21-day Brent move: 15%
- 21-day annualized Brent volatility: 20%
- VIX level: 20%
- absolute 10-year breakeven deviation from 2%: 15%
- absolute 21-day broad-dollar move: 15%

The current score is {status['market_stress_score']:.1f}/100
({status['market_stress_level']}). The data-quality score is
{status['data_quality_score']:.1f}/100. Live market levels create an overlay,
including implied Brent price ranges, but do not silently alter the strategic
state-transition coefficients.

## Model-status gating

The effective status is **{status['effective_model_status']}**. Live data are
classified as Live, Degraded, or Versioned fallback based on series coverage,
missingness, history, and publication freshness. The historical calibration
passed {int(calibration['gates_passed'])} of
{int(calibration['gates_total'])} promotion gates. Version 1.2 therefore keeps
the historical coefficients diagnostic and leaves the versioned scenario
coefficients active.

## Baseline and data provenance

The live baseline is dated {baseline_date}. Public sources are recorded in
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
"""
    path.write_text(content, encoding="utf-8")
    return path


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            fontName=FONT_BOLD,
            fontSize=24,
            leading=29,
            textColor=colors.HexColor(NAVY),
            alignment=TA_LEFT,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubtitleCustom",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#486581"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingCustom",
            parent=styles["Heading2"],
            fontName=FONT_BOLD,
            fontSize=15,
            leading=18,
            textColor=colors.HexColor(NAVY),
            spaceBefore=8,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=9.2,
            leading=13.2,
            textColor=colors.HexColor(TEXT),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCustom",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#486581"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="CalloutCustom",
            parent=styles["BodyText"],
            fontName=FONT_BOLD,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor(NAVY),
            backColor=colors.HexColor(LIGHT),
            borderPadding=10,
            spaceAfter=10,
        )
    )
    return styles


def _table(data, widths, font_size=7.2, header=True):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCCCDC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if not header:
        commands = [command for command in commands if command[0] not in {"BACKGROUND", "TEXTCOLOR"}]
    table.setStyle(TableStyle(commands))
    return table


def _header_footer(canvas, doc):
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.line(0.65 * inch, 0.55 * inch, width - 0.65 * inch, 0.55 * inch)
    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.setFillColor(colors.HexColor("#627D98"))
    canvas.drawString(0.65 * inch, 0.35 * inch, "Conditional counterfactual - not a forecast")
    canvas.drawRightString(width - 0.65 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def create_executive_pdf(
    pdf_path: Path,
    outputs: dict[str, pd.DataFrame],
    charts: dict[str, Path],
) -> None:
    styles = _pdf_styles()
    status = outputs["model_status"].iloc[0]
    market = outputs["market_context_summary"].iloc[0]
    baseline_date = (
        pd.Timestamp(market["as_of_date"]).strftime("%B %d, %Y")
        if pd.notna(market["as_of_date"])
        else "versioned fallback"
    )
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.70 * inch,
        title="U.S. Middle East Strategic Withdrawal Response Model",
        author="Jason Keen",
    )
    story = []

    story.extend(
        [
            Spacer(1, 0.25 * inch),
            Paragraph(
                "U.S. Middle East Strategic<br/>Withdrawal Response Model",
                styles["TitleCustom"],
            ),
            Paragraph(
                "Actor decisions, strategic pathways, and economic transmission | "
                f"Version 1.2 | Live baseline: {baseline_date}",
                styles["SubtitleCustom"],
            ),
            Paragraph(
                "Decision question: If the United States materially reduces or ends "
                "its military and security posture in the Middle East, which actor "
                "responses become more plausible, and how might those responses "
                "transmit into energy, shipping, macroeconomic, and sovereign-risk stress?",
                styles["CalloutCustom"],
            ),
            Paragraph("Executive assessment", styles["HeadingCustom"]),
            Paragraph(
                "The simulation does not identify a single inevitable outcome. It finds "
                "a widening distribution of pathways as withdrawal removes residual "
                "capability and weakens security guarantees. Limited reductions are "
                "absorbed mainly through hedging and procurement. Full withdrawal and "
                "strategic disengagement increase the modeled frequency of competitive "
                "rearmament, maritime disruption, and regional escalation.",
                styles["BodyCustom"],
            ),
            Paragraph(
                "The most decision-relevant variables are not troop numbers alone. "
                "Withdrawal speed, residual intelligence and naval capability, and the "
                "credibility of remaining guarantees materially change actor incentives.",
                styles["BodyCustom"],
            ),
            Paragraph(
                f"Model status: {status['effective_model_status']} | "
                f"Data quality: {status['data_quality_score']:.1f}/100 | "
                f"Live market stress: {status['market_stress_score']:.1f}/100 "
                f"({status['market_stress_level']})",
                styles["CalloutCustom"],
            ),
            Image(str(charts["strategic_stress"]), width=7.15 * inch, height=3.65 * inch),
            Paragraph(
                "Figure 1. Median state values across Monte Carlo paths. The scale is "
                "an internally consistent stress index, not an observed probability.",
                styles["SmallCustom"],
            ),
        ]
    )

    story.append(PageBreak())
    story.append(Paragraph("Live economic baseline", styles["HeadingCustom"]))
    story.extend(
        [
            Paragraph(
                "The live context layer audits market freshness and macroeconomic "
                "coverage before presenting an observational baseline. It does not "
                "silently recalibrate the strategic simulation.",
                styles["BodyCustom"],
            ),
            Paragraph(
                f"Current assessment: {status['market_stress_level']} market stress "
                f"({status['market_stress_score']:.1f}/100) with "
                f"{status['data_status']} data status and "
                f"{status['calibration_status']} calibration.",
                styles["CalloutCustom"],
            ),
        ]
    )
    if "market_stress" in charts:
        story.extend(
            [
                Image(
                    str(charts["market_stress"]),
                    width=7.0 * inch,
                    height=3.25 * inch,
                ),
                Paragraph(
                    "Figure 2. Percentiles compare current observations with their "
                    "trailing five-year histories. They are stress indicators, not "
                    "probabilities of military or political outcomes.",
                    styles["SmallCustom"],
                ),
            ]
        )
    market_table = [
        ["Live indicator", "Latest value", "Context"],
        ["Brent crude", f"${market['brent_latest']:.2f}/bbl", f"21-day change {market['brent_21d_change_pct']:+.1f}%"],
        ["Brent volatility", f"{market['brent_21d_annualized_volatility_pct']:.1f}%", "21-day annualized"],
        ["VIX", f"{market['vix_latest']:.2f}", "Equity-market volatility"],
        ["10-year breakeven", f"{market['breakeven_10y_latest_pct']:.2f}%", "Market inflation expectation"],
        ["Broad dollar index", f"{market['broad_dollar_latest']:.2f}", f"21-day change {market['broad_dollar_21d_change_pct']:+.1f}%"],
    ]
    story.extend(
        [
            Spacer(1, 0.10 * inch),
            _table(
                market_table,
                [1.70 * inch, 1.30 * inch, 3.55 * inch],
                font_size=7.2,
            ),
            Spacer(1, 0.12 * inch),
            Paragraph("Live Brent scenario overlay", styles["HeadingCustom"]),
        ]
    )
    overlay = outputs.get("live_brent_scenario_overlay", pd.DataFrame())
    if not overlay.empty:
        overlay_table = [["Scenario", "5th percentile", "Median", "95th percentile"]]
        for row in overlay.itertuples(index=False):
            overlay_table.append(
                [
                    row.scenario,
                    f"${row.implied_brent_p05:.2f}",
                    f"${row.implied_brent_median:.2f}",
                    f"${row.implied_brent_p95:.2f}",
                ]
            )
        story.extend(
            [
                _table(
                    overlay_table,
                    [2.60 * inch, 1.20 * inch, 1.20 * inch, 1.25 * inch],
                    font_size=7.0,
                ),
                Spacer(1, 0.08 * inch),
                Paragraph(
                    "The overlay applies simulated percentage changes to the latest "
                    "observed Brent price. It is a conditional arithmetic translation, "
                    "not a market-price forecast.",
                    styles["SmallCustom"],
                ),
            ]
        )

    story.append(PageBreak())
    story.append(Paragraph("Scenario comparison", styles["HeadingCustom"]))
    summary = outputs["scenario_summary"]
    table_data = [["Scenario", "WSI", "Tension", "Maritime", "Arms race", "Escalation", "Brent median"]]
    for row in summary.itertuples(index=False):
        table_data.append(
            [
                row.scenario,
                f"{row.withdrawal_shock_index:.2f}",
                f"{row.median_security_tension:.2f}",
                f"{row.median_maritime_risk:.2f}",
                f"{row.median_arms_race:.2f}",
                f"{row.regional_escalation_probability:.1%}",
                f"{row.median_brent_change_pct:+.1f}%",
            ]
        )
    story.extend(
        [
            _table(
                table_data,
                [1.70 * inch, 0.50 * inch, 0.66 * inch, 0.66 * inch, 0.66 * inch, 0.72 * inch, 0.80 * inch],
                font_size=6.9,
            ),
            Spacer(1, 0.18 * inch),
            Paragraph(
                "WSI is the Withdrawal Shock Index. Escalation is the share of paths "
                "classified as regional escalation in the final simulation year. "
                "Brent changes are conditional stress ranges relative to the simulated baseline.",
                styles["SmallCustom"],
            ),
            Spacer(1, 0.15 * inch),
            Image(str(charts["economic_ranges"]), width=7.1 * inch, height=2.95 * inch),
            Paragraph(
                "Figure 3. Dots show medians; whiskers show 5th-95th percentile ranges. "
                "Economic results use transparent structural stress coefficients and "
                "should not be read as causally identified forecasts.",
                styles["SmallCustom"],
            ),
            Paragraph("Economic interpretation", styles["HeadingCustom"]),
            Paragraph(
                "The principal transmission chain is deterrence ambiguity to actor "
                "hedging or coercion, then to maritime risk and energy disruption, and "
                "finally to shipping costs, oil prices, inflation, growth, and sovereign "
                "risk. Oil exporters can gain revenue from higher prices while still "
                "incurring infrastructure, shipping, investment, and fiscal-volatility costs.",
                styles["BodyCustom"],
            ),
        ]
    )

    story.append(PageBreak())
    story.append(Paragraph("Historical calibration", styles["HeadingCustom"]))
    calibration = outputs["calibration_summary"].iloc[0]
    metrics = outputs["calibration_metrics"].iloc[0]
    gate_detail = outputs["calibration_gate_detail"]
    story.extend(
        [
            Paragraph(
                "Version 1.2 tests the Brent stress translation against sourced "
                "regional-security, energy-infrastructure, and maritime events. "
                "Event dates are sourced facts; the three shock-severity scores are "
                "versioned expert inputs.",
                styles["BodyCustom"],
            ),
            Paragraph(
                f"Calibration status: {calibration['calibration_status']} | "
                f"{int(calibration['gates_passed'])}/{int(calibration['gates_total'])} "
                "promotion gates passed | Historical coefficients remain diagnostic.",
                styles["CalloutCustom"],
            ),
            Image(
                str(charts["calibration_diagnostics"]),
                width=7.1 * inch,
                height=2.95 * inch,
            ),
            Paragraph(
                "Figure 4. Left: observed peak Brent excursions versus leave-one-event-out "
                "predictions. Right: versioned priors versus historical estimates with "
                "5th-95th percentile bootstrap intervals.",
                styles["SmallCustom"],
            ),
        ]
    )
    gate_table = [["Promotion gate", "Observed", "Threshold", "Result"]]
    gate_labels = {
        "event_count": "Eligible events",
        "source_coverage": "Source coverage (%)",
        "brent_target_coverage": "Brent coverage (%)",
        "normalized_loo_mae": "Normalized LOO MAE",
        "directional_agreement": "Directional agreement (%)",
        "coefficient_stability": "Maximum coefficient CV",
    }
    for row in gate_detail.itertuples(index=False):
        gate_table.append(
            [
                gate_labels.get(row.gate, row.gate),
                f"{row.value:.2f}",
                f"{row.operator} {row.threshold:.2f}",
                "PASS" if row.passed else "HOLD",
            ]
        )
    story.extend(
        [
            Spacer(1, 0.10 * inch),
            _table(
                gate_table,
                [2.45 * inch, 1.15 * inch, 1.25 * inch, 0.85 * inch],
                font_size=7.0,
            ),
            Spacer(1, 0.10 * inch),
            Paragraph(
                f"Leave-one-event-out MAE is {metrics['loo_mae_pct_points']:.1f} "
                f"percentage points; directional agreement is "
                f"{metrics['directional_agreement_pct']:.0f}%. The normalized-error "
                "and directional gates did not pass, so the model does not activate "
                "the fitted coefficients. This is a governed hold, not a failed run.",
                styles["BodyCustom"],
            ),
        ]
    )

    story.append(PageBreak())
    story.append(Paragraph("Conditional actor responses", styles["HeadingCustom"]))
    story.extend(
        [
            Paragraph(
                "Probabilities below are generated from actor characteristics, action "
                "requirements, and scenario conditions. They represent structured "
                "comparisons among configured choices, not measured intent.",
                styles["BodyCustom"],
            ),
            Image(str(charts["action_heatmap"]), width=7.1 * inch, height=4.0 * inch),
            Paragraph(
                "Figure 5. Higher values identify actions that become more competitive "
                "within an actor's configured choice set under strategic disengagement.",
                styles["SmallCustom"],
            ),
        ]
    )
    probs = outputs["actor_action_probabilities"]
    severe = probs[probs["scenario_id"].eq("strategic_disengagement")]
    top = severe.loc[severe.groupby("actor_id")["probability"].idxmax()].sort_values(
        "probability", ascending=False
    )
    response_table = [["Actor", "Highest-ranked configured action", "Probability", "Maintain"]]
    for row in top.itertuples(index=False):
        response_table.append(
            [
                Paragraph(row.actor, styles["SmallCustom"]),
                Paragraph(row.action, styles["SmallCustom"]),
                f"{row.probability:.1%}",
                f"{row.maintain_posture_probability:.1%}",
            ]
        )
    story.extend(
        [
            Spacer(1, 0.12 * inch),
            _table(
                response_table,
                [1.75 * inch, 3.05 * inch, 0.75 * inch, 0.85 * inch],
                font_size=6.8,
            ),
        ]
    )

    story.append(PageBreak())
    story.append(Paragraph("Economic-security exposure", styles["HeadingCustom"]))
    story.extend(
        [
            Paragraph(
                "Vulnerability combines conflict proximity, U.S. security dependence, "
                "Hormuz exposure, energy dependence, fiscal capacity, reserve buffers, "
                "trade diversification, and the simulated strategic state. It measures "
                "sensitivity to the scenario, not the likelihood that the scenario occurs.",
                styles["BodyCustom"],
            ),
            Image(str(charts["vulnerability"]), width=6.85 * inch, height=4.2 * inch),
            Paragraph(
                "Figure 6. Highest composite exposure scores under the most severe "
                "configured scenario.",
                styles["SmallCustom"],
            ),
            Paragraph("Key analytical cautions", styles["HeadingCustom"]),
            Paragraph(
                "<b>Public-information constraint.</b> Force posture, commitments, and "
                "private diplomacy are incompletely observed. <b>Behavioral uncertainty.</b> "
                "States and non-state groups are internally heterogeneous. "
                "<b>Calibration constraint.</b> Version 1.2 historical event windows "
                "are heterogeneous and did not pass every promotion gate. "
                "<b>Tail-risk constraint.</b> Nuclear use and other extreme outcomes are "
                "outside the modeled state space.",
                styles["BodyCustom"],
            ),
        ]
    )

    story.append(PageBreak())
    story.append(Paragraph("Method and sources", styles["HeadingCustom"]))
    story.extend(
        [
            Paragraph(
                "<b>Withdrawal Shock Index:</b> 35% force reduction, 20% withdrawal "
                "speed, 25% loss of residual capability, and 20% weakening of guarantees. "
                "Actor utilities are transformed with a softmax choice model. Each actor "
                "selects one action or maintains posture in every simulated quarter. "
                "Action effects update five persistent strategic states over a five-year horizon.",
                styles["BodyCustom"],
            ),
            Paragraph(
                "<b>Reproducibility:</b> All assumptions are stored in CSV files, random "
                "seeds are fixed by default, and the pipeline writes full probability, "
                "pathway, economic, vulnerability, and sensitivity tables.",
                styles["BodyCustom"],
            ),
            Paragraph("Selected public baselines", styles["HeadingCustom"]),
        ]
    )
    source_rows = [
        ["Source", "Use", "Public URL"],
        [
            "U.S. EIA",
            "Hormuz flows, bypass capacity, and global oil outlook",
            "eia.gov/world-oil-chokepoints",
        ],
        [
            "IMF PortWatch",
            "Maritime trade and chokepoint disruption",
            "portwatch.imf.org/pages/data-and-methodology",
        ],
        [
            "IMF WEO",
            "Growth, inflation, fiscal, and external-sector context",
            "data.imf.org/en/datasets/IMF.RES:WEO",
        ],
        [
            "World Bank",
            "Country macroeconomic and governance context",
            "data.worldbank.org",
        ],
        [
            "SIPRI",
            "Military expenditure through 2025",
            "sipri.org/databases/milex",
        ],
        [
            "FRED",
            "Brent, volatility, inflation expectations, and dollar context",
            "fred.stlouisfed.org",
        ],
    ]
    source_rows = [
        source_rows[0],
        *[
            [
                Paragraph(str(row[0]), styles["SmallCustom"]),
                Paragraph(str(row[1]), styles["SmallCustom"]),
                Paragraph(str(row[2]), styles["SmallCustom"]),
            ]
            for row in source_rows[1:]
        ],
    ]
    story.extend(
        [
            _table(
                source_rows,
                [1.0 * inch, 2.55 * inch, 3.1 * inch],
                font_size=6.8,
            ),
            Spacer(1, 0.18 * inch),
            Paragraph(
                "Full source register, assumption files, formulas, limitations, and "
                "validation roadmap are included in the project repository. Baseline "
                "verified July 29, 2026.",
                styles["CalloutCustom"],
            ),
            Spacer(1, 0.08 * inch),
            Paragraph(
                "Prepared as an open-source strategic-economic research model. Findings "
                "should be reviewed by regional, military, energy, and macroeconomic "
                "specialists before any policy or investment use.",
                styles["BodyCustom"],
            ),
        ]
    )

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def build_all_outputs(
    project_root: Path,
    model,
    outputs: dict[str, pd.DataFrame],
    create_pdf: bool = True,
) -> None:
    output_root = project_root / "output"
    data_dir = output_root / "data"
    chart_dir = output_root / "charts"
    pdf_dir = output_root / "pdf"
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    _write_tables(data_dir, outputs)
    charts = create_charts(chart_dir, outputs)
    _methodology_markdown(project_root, model, outputs)
    if create_pdf:
        create_executive_pdf(
            pdf_dir / "us_middle_east_withdrawal_executive_brief.pdf",
            outputs,
            charts,
        )
