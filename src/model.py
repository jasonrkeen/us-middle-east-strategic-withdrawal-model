"""Core actor-choice and strategic-economic Monte Carlo model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STATE_COLUMNS = [
    "security_tension",
    "maritime_risk",
    "arms_race",
    "diplomatic_hedging",
    "energy_disruption",
]

BRENT_STRESS_COEFFICIENTS = {
    "energy_disruption": 34.0,
    "maritime_risk": 16.0,
    "security_tension": 12.0,
}


@dataclass
class ScenarioSimulation:
    scenario_id: str
    states: pd.DataFrame
    pathway_probabilities: pd.DataFrame
    action_frequency: pd.DataFrame


class StrategicWithdrawalModel:
    """Conditional counterfactual model with auditable assumption files."""

    def __init__(self, project_root: Path, seed: int = 20260727) -> None:
        self.project_root = Path(project_root)
        self.seed = seed
        data_dir = self.project_root / "data"
        self.scenarios = pd.read_csv(data_dir / "scenarios.csv")
        self.actors = pd.read_csv(data_dir / "actors.csv")
        self.actions = pd.read_csv(data_dir / "actions.csv")
        self.affinities = pd.read_csv(data_dir / "actor_action_affinity.csv")
        self.exposures = pd.read_csv(data_dir / "country_exposures.csv")
        self.sources = pd.read_csv(data_dir / "source_register.csv")
        self._validate_inputs()
        self.scenarios = self._add_scenario_indices(self.scenarios)

    @staticmethod
    def _add_scenario_indices(scenarios: pd.DataFrame) -> pd.DataFrame:
        result = scenarios.copy()
        result["withdrawal_shock_index"] = (
            0.35 * result["force_reduction"]
            + 0.20 * result["withdrawal_speed"]
            + 0.25 * (1.0 - result["residual_capability"])
            + 0.20 * (1.0 - result["security_guarantee"])
        )
        result["deterrence_gap"] = (
            0.50 * result["force_reduction"]
            + 0.30 * (1.0 - result["residual_capability"])
            + 0.20 * result["withdrawal_speed"]
        )
        result["alliance_ambiguity"] = (
            0.60 * (1.0 - result["security_guarantee"])
            + 0.40 * result["withdrawal_speed"]
        )
        return result

    def _validate_inputs(self) -> None:
        unit_interval_columns = {
            "scenarios": (
                self.scenarios,
                [
                    "force_reduction",
                    "withdrawal_speed",
                    "residual_capability",
                    "security_guarantee",
                ],
            ),
            "actors": (
                self.actors,
                [
                    "security_exposure",
                    "economic_capacity",
                    "opportunism",
                    "us_security_dependency",
                    "restraint",
                    "diplomatic_flexibility",
                ],
            ),
            "actions": (
                self.actions,
                [
                    "security_weight",
                    "opportunity_weight",
                    "capacity_requirement",
                    "escalation_cost",
                    "economic_cost",
                ],
            ),
        }
        for table_name, (frame, columns) in unit_interval_columns.items():
            for column in columns:
                if not frame[column].between(0, 1).all():
                    raise ValueError(f"{table_name}.{column} must remain within [0, 1]")

        known_actors = set(self.actors["actor_id"])
        known_actions = set(self.actions["action_id"])
        if not set(self.affinities["actor_id"]).issubset(known_actors):
            raise ValueError("Affinity file contains an unknown actor_id")
        if not set(self.affinities["action_id"]).issubset(known_actions):
            raise ValueError("Affinity file contains an unknown action_id")
        if self.affinities.duplicated(["actor_id", "action_id"]).any():
            raise ValueError("Actor-action affinity pairs must be unique")

    def calculate_action_probabilities(self) -> pd.DataFrame:
        merged = (
            self.affinities.merge(self.actors, on="actor_id", validate="many_to_one")
            .merge(self.actions, on="action_id", validate="many_to_one")
        )
        records: list[dict] = []

        for scenario in self.scenarios.itertuples(index=False):
            scenario_rows = merged.copy()
            gap = scenario.deterrence_gap
            ambiguity = scenario.alliance_ambiguity

            scenario_rows["utility"] = (
                scenario_rows["base_utility"]
                + 1.35
                * scenario_rows["security_weight"]
                * scenario_rows["security_exposure"]
                * gap
                + 1.10
                * scenario_rows["opportunity_weight"]
                * scenario_rows["opportunism"]
                * gap
                + 0.85
                * scenario_rows["diplomatic_flexibility"]
                * ambiguity
                * scenario_rows["diplomatic_action"]
                + 0.70
                * scenario_rows["us_security_dependency"]
                * ambiguity
                * scenario_rows["security_substitution"]
                - 1.20
                * np.maximum(
                    scenario_rows["capacity_requirement"]
                    - scenario_rows["economic_capacity"],
                    0,
                )
                - 0.75
                * scenario_rows["escalation_cost"]
                * scenario_rows["restraint"]
                * scenario.residual_capability
                - 0.45
                * scenario_rows["economic_cost"]
                * (1 - scenario_rows["economic_capacity"])
            )

            for actor_id, actor_rows in scenario_rows.groupby("actor_id", sort=False):
                maintain_utility = (
                    0.85
                    - 1.35 * gap
                    - 0.55 * scenario.withdrawal_speed
                    + 0.35
                    * float(
                        self.actors.loc[
                            self.actors["actor_id"].eq(actor_id), "restraint"
                        ].iloc[0]
                    )
                )
                utilities = actor_rows["utility"].to_numpy(dtype=float)
                all_utilities = np.append(utilities, maintain_utility)
                exp_values = np.exp(all_utilities - all_utilities.max())
                probabilities = exp_values / exp_values.sum()

                for row, probability in zip(
                    actor_rows.itertuples(index=False), probabilities[:-1]
                ):
                    records.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "scenario": scenario.scenario,
                            "actor_id": actor_id,
                            "actor": row.actor,
                            "action_id": row.action_id,
                            "action": row.action,
                            "probability": probability,
                            "utility": row.utility,
                            "maintain_posture_probability": probabilities[-1],
                        }
                    )

        probabilities = pd.DataFrame(records)
        return probabilities.sort_values(
            ["scenario_id", "actor_id", "probability"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    def _scenario_rng(self, scenario_position: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + 1009 * scenario_position)

    def _simulate_scenario(
        self,
        scenario: pd.Series,
        action_probabilities: pd.DataFrame,
        runs: int,
        quarters: int,
        scenario_position: int,
    ) -> ScenarioSimulation:
        rng = self._scenario_rng(scenario_position)
        actor_ids = self.actors["actor_id"].tolist()
        action_ids = self.actions["action_id"].tolist()
        action_index = {action: index for index, action in enumerate(action_ids)}
        maintain_index = len(action_ids)

        effect_matrix = self.actions[
            [
                "effect_tension",
                "effect_maritime",
                "effect_arms",
                "effect_hedging",
                "effect_energy_buffer",
            ]
        ].to_numpy(dtype=float)
        effect_matrix = np.vstack([effect_matrix, np.zeros((1, 5))])

        initial = np.array([0.58, 0.47, 0.50, 0.42, 0.34], dtype=float)
        state = np.tile(initial, (runs, 1))
        state_history = np.zeros((runs, quarters, len(STATE_COLUMNS)), dtype=float)
        action_counts = np.zeros((runs, len(action_ids)), dtype=np.int32)

        probability_vectors: dict[str, np.ndarray] = {}
        for actor_id in actor_ids:
            actor_rows = action_probabilities[
                action_probabilities["actor_id"].eq(actor_id)
            ]
            action_map = actor_rows.set_index("action_id")["probability"].to_dict()
            vector = np.array(
                [action_map.get(action, 0.0) for action in action_ids]
                + [float(actor_rows["maintain_posture_probability"].iloc[0])],
                dtype=float,
            )
            vector /= vector.sum()
            probability_vectors[actor_id] = vector

        target = np.array(
            [
                0.31 + 0.46 * scenario["deterrence_gap"],
                0.24 + 0.39 * scenario["withdrawal_shock_index"],
                0.28 + 0.47 * scenario["alliance_ambiguity"],
                0.29 + 0.50 * scenario["alliance_ambiguity"],
                0.20 + 0.42 * scenario["withdrawal_shock_index"],
            ]
        )
        # Persistent run-level variation prevents the model from implying that
        # leaders, alliances, and crises respond uniformly to the same posture
        # shock. It is deliberately larger than quarterly process noise.
        run_regime_variation = rng.normal(0.0, 0.10, size=(runs, 5))
        run_targets = np.clip(target + run_regime_variation, 0.0, 1.0)

        for quarter in range(quarters):
            selected_effects = np.zeros((runs, 5), dtype=float)
            for actor_id in actor_ids:
                selected = rng.choice(
                    len(action_ids) + 1,
                    size=runs,
                    p=probability_vectors[actor_id],
                )
                selected_effects += effect_matrix[selected]
                for index in range(len(action_ids)):
                    action_counts[:, index] += selected == index

            mean_effects = selected_effects / len(actor_ids)
            noise = rng.normal(0.0, 0.018, size=(runs, 5))
            innovation = np.column_stack(
                [
                    0.78 * mean_effects[:, 0],
                    0.74 * mean_effects[:, 1],
                    0.70 * mean_effects[:, 2],
                    0.72 * mean_effects[:, 3],
                    0.66 * mean_effects[:, 1]
                    - 0.60 * mean_effects[:, 4],
                ]
            )
            persistence = np.array([0.84, 0.86, 0.89, 0.87, 0.86])
            state = (
                persistence * state
                + (1.0 - persistence) * run_targets
                + innovation
                + noise
            )
            state = np.clip(state, 0.0, 1.0)
            state_history[:, quarter, :] = state

        final_state = state_history[:, max(0, quarters - 4) :, :].mean(axis=1)
        states = pd.DataFrame(final_state, columns=STATE_COLUMNS)
        states.insert(0, "run", np.arange(runs))
        states.insert(0, "scenario_id", scenario["scenario_id"])

        pathways = np.select(
            [
                (states["security_tension"] >= 0.78),
                (states["maritime_risk"] >= 0.64),
                (states["arms_race"] >= 0.67),
                (states["diplomatic_hedging"] >= 0.58)
                & (states["security_tension"] < 0.62),
                (states["security_tension"] < 0.55)
                & (states["arms_race"] < 0.55)
                & (scenario["scenario_id"] == "baseline"),
            ],
            [
                "Regional escalation",
                "Maritime disruption",
                "Competitive rearmament",
                "Managed hedging",
                "Baseline continuity",
            ],
            default="Incremental adaptation",
        )
        pathway_probabilities = (
            pd.Series(pathways, name="pathway")
            .value_counts(normalize=True)
            .rename("probability")
            .reset_index()
        )
        pathway_probabilities.insert(0, "scenario_id", scenario["scenario_id"])

        frequencies = action_counts.sum(axis=0) / action_counts.sum()
        action_frequency = pd.DataFrame(
            {
                "scenario_id": scenario["scenario_id"],
                "action_id": action_ids,
                "selection_share": frequencies,
            }
        ).merge(self.actions[["action_id", "action"]], on="action_id", how="left")

        return ScenarioSimulation(
            scenario_id=scenario["scenario_id"],
            states=states,
            pathway_probabilities=pathway_probabilities,
            action_frequency=action_frequency,
        )

    def run_all_scenarios(
        self, runs: int = 4000, quarters: int = 20
    ) -> dict[str, ScenarioSimulation]:
        if runs < 200:
            raise ValueError("Use at least 200 Monte Carlo runs")
        if quarters < 4:
            raise ValueError("Use a horizon of at least four quarters")

        probabilities = self.calculate_action_probabilities()
        results: dict[str, ScenarioSimulation] = {}
        for position, scenario in self.scenarios.iterrows():
            scenario_probs = probabilities[
                probabilities["scenario_id"].eq(scenario["scenario_id"])
            ]
            results[scenario["scenario_id"]] = self._simulate_scenario(
                scenario=scenario,
                action_probabilities=scenario_probs,
                runs=runs,
                quarters=quarters,
                scenario_position=position,
            )
        return results

    def _economic_impacts(
        self, results: dict[str, ScenarioSimulation]
    ) -> pd.DataFrame:
        baseline = results["baseline"].states
        baseline_medians = baseline[STATE_COLUMNS].median()
        records: list[dict] = []

        for position, scenario in self.scenarios.iterrows():
            state = results[scenario["scenario_id"]].states
            rng = np.random.default_rng(self.seed + 7919 + position)
            delta = state[STATE_COLUMNS] - baseline_medians

            brent = (
                BRENT_STRESS_COEFFICIENTS["energy_disruption"]
                * delta["energy_disruption"]
                + BRENT_STRESS_COEFFICIENTS["maritime_risk"]
                * delta["maritime_risk"]
                + BRENT_STRESS_COEFFICIENTS["security_tension"]
                * delta["security_tension"]
                + rng.normal(0, 3.0, len(state))
            )
            shipping = (
                55 * delta["maritime_risk"]
                + 18 * delta["security_tension"]
                + rng.normal(0, 4.0, len(state))
            )
            global_inflation = 0.028 * brent + 0.010 * shipping
            global_gdp = -0.018 * brent - 0.007 * shipping
            mena_non_oil_gdp = (
                -0.013 * brent
                - 0.020 * shipping
                - 0.45 * delta["security_tension"]
            )
            military_spending = (
                25 * delta["arms_race"]
                + 8 * scenario["deterrence_gap"]
                + rng.normal(0, 1.7, len(state))
            )
            sovereign_spread = (
                180 * delta["security_tension"]
                + 90 * delta["maritime_risk"]
                - 40 * delta["diplomatic_hedging"]
                + rng.normal(0, 15, len(state))
            )

            metrics = {
                "brent_price_change_pct": brent,
                "shipping_cost_change_pct": shipping,
                "global_inflation_change_pp": global_inflation,
                "global_gdp_change_pp": global_gdp,
                "mena_non_oil_gdp_change_pp": mena_non_oil_gdp,
                "regional_military_spending_change_pct": military_spending,
                "regional_sovereign_spread_change_bps": sovereign_spread,
            }
            for metric, values in metrics.items():
                records.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "scenario": scenario["scenario"],
                        "metric": metric,
                        "p05": float(np.quantile(values, 0.05)),
                        "median": float(np.median(values)),
                        "p95": float(np.quantile(values, 0.95)),
                    }
                )
        return pd.DataFrame(records)

    def _scenario_summary(
        self,
        results: dict[str, ScenarioSimulation],
        economic_impacts: pd.DataFrame,
    ) -> pd.DataFrame:
        records: list[dict] = []
        for _, scenario in self.scenarios.iterrows():
            states = results[scenario["scenario_id"]].states
            pathways = results[scenario["scenario_id"]].pathway_probabilities
            escalation = pathways.loc[
                pathways["pathway"].eq("Regional escalation"), "probability"
            ]
            brent = economic_impacts[
                economic_impacts["scenario_id"].eq(scenario["scenario_id"])
                & economic_impacts["metric"].eq("brent_price_change_pct")
            ]
            records.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario": scenario["scenario"],
                    "withdrawal_shock_index": scenario["withdrawal_shock_index"],
                    "deterrence_gap": scenario["deterrence_gap"],
                    "alliance_ambiguity": scenario["alliance_ambiguity"],
                    "median_security_tension": states["security_tension"].median(),
                    "median_maritime_risk": states["maritime_risk"].median(),
                    "median_arms_race": states["arms_race"].median(),
                    "median_diplomatic_hedging": states[
                        "diplomatic_hedging"
                    ].median(),
                    "regional_escalation_probability": (
                        float(escalation.iloc[0]) if not escalation.empty else 0.0
                    ),
                    "median_brent_change_pct": float(brent["median"].iloc[0]),
                    "brent_p05": float(brent["p05"].iloc[0]),
                    "brent_p95": float(brent["p95"].iloc[0]),
                }
            )
        return pd.DataFrame(records)

    def _vulnerability_table(
        self, results: dict[str, ScenarioSimulation]
    ) -> pd.DataFrame:
        records: list[dict] = []
        for _, scenario in self.scenarios.iterrows():
            states = results[scenario["scenario_id"]].states
            med = states[STATE_COLUMNS].median()
            for exposure in self.exposures.itertuples(index=False):
                score = 100 * (
                    0.18 * exposure.conflict_proximity * scenario["deterrence_gap"]
                    + 0.18
                    * exposure.us_security_dependency
                    * (1 - scenario["security_guarantee"])
                    + 0.14 * exposure.hormuz_trade_exposure * med["maritime_risk"]
                    + 0.12 * exposure.oil_import_dependence * med["energy_disruption"]
                    + 0.09 * exposure.oil_export_dependence * med["maritime_risk"]
                    + 0.10 * (1 - exposure.fiscal_capacity) * med["security_tension"]
                    + 0.08 * (1 - exposure.reserve_buffer) * med["energy_disruption"]
                    + 0.06 * (1 - exposure.trade_diversification)
                    + 0.05 * med["arms_race"]
                )
                score = float(np.clip(score, 0, 100))
                level = pd.cut(
                    [score],
                    bins=[-np.inf, 25, 40, 55, 70, np.inf],
                    labels=["Low", "Guarded", "Elevated", "High", "Severe"],
                )[0]
                records.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "scenario": scenario["scenario"],
                        "economy": exposure.economy,
                        "vulnerability_score": score,
                        "vulnerability_level": str(level),
                    }
                )
        return pd.DataFrame(records)

    def _sensitivity_table(self) -> pd.DataFrame:
        rows: list[dict] = []
        base = self.scenarios[
            self.scenarios["scenario_id"].eq("full_military_withdrawal")
        ].iloc[0]
        parameters = [
            "force_reduction",
            "withdrawal_speed",
            "residual_capability",
            "security_guarantee",
        ]
        base_index = float(base["withdrawal_shock_index"])
        for parameter in parameters:
            for delta in (-0.10, 0.10):
                modified = base.copy()
                modified[parameter] = float(np.clip(modified[parameter] + delta, 0, 1))
                frame = self._add_scenario_indices(pd.DataFrame([modified]))
                rows.append(
                    {
                        "parameter": parameter,
                        "change": delta,
                        "base_withdrawal_shock_index": base_index,
                        "new_withdrawal_shock_index": float(
                            frame["withdrawal_shock_index"].iloc[0]
                        ),
                        "index_change": float(
                            frame["withdrawal_shock_index"].iloc[0] - base_index
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def build_output_tables(
        self,
        action_probabilities: pd.DataFrame,
        results: dict[str, ScenarioSimulation],
    ) -> dict[str, pd.DataFrame]:
        economic_impacts = self._economic_impacts(results)
        return {
            "scenario_definitions": self.scenarios.copy(),
            "actor_action_probabilities": action_probabilities,
            "scenario_summary": self._scenario_summary(results, economic_impacts),
            "economic_impacts": economic_impacts,
            "pathway_probabilities": pd.concat(
                [result.pathway_probabilities for result in results.values()],
                ignore_index=True,
            ),
            "action_frequency": pd.concat(
                [result.action_frequency for result in results.values()],
                ignore_index=True,
            ),
            "country_vulnerability": self._vulnerability_table(results),
            "sensitivity_analysis": self._sensitivity_table(),
        }
