from pathlib import Path

import numpy as np

from src.model import StrategicWithdrawalModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scenario_indices_are_monotonic():
    model = StrategicWithdrawalModel(PROJECT_ROOT)
    values = model.scenarios["withdrawal_shock_index"].to_numpy()
    assert np.all(np.diff(values) > 0)


def test_action_probabilities_are_bounded():
    model = StrategicWithdrawalModel(PROJECT_ROOT)
    probabilities = model.calculate_action_probabilities()
    assert probabilities["probability"].between(0, 1).all()
    assert probabilities["maintain_posture_probability"].between(0, 1).all()


def test_action_probabilities_sum_to_one_with_maintain_option():
    model = StrategicWithdrawalModel(PROJECT_ROOT)
    probabilities = model.calculate_action_probabilities()
    grouped = probabilities.groupby(["scenario_id", "actor_id"])
    for _, group in grouped:
        total = group["probability"].sum() + group[
            "maintain_posture_probability"
        ].iloc[0]
        assert np.isclose(total, 1.0)


def test_simulated_states_are_bounded():
    model = StrategicWithdrawalModel(PROJECT_ROOT)
    results = model.run_all_scenarios(runs=250, quarters=8)
    for result in results.values():
        state_values = result.states.drop(columns=["scenario_id", "run"])
        assert state_values.min().min() >= 0
        assert state_values.max().max() <= 1


def test_severe_scenario_exceeds_limited_median_tension():
    model = StrategicWithdrawalModel(PROJECT_ROOT)
    results = model.run_all_scenarios(runs=400, quarters=12)
    limited = results["limited_drawdown"].states["security_tension"].median()
    severe = results["strategic_disengagement"].states[
        "security_tension"
    ].median()
    assert severe > limited


def test_seed_is_reproducible():
    first = StrategicWithdrawalModel(PROJECT_ROOT, seed=42)
    second = StrategicWithdrawalModel(PROJECT_ROOT, seed=42)
    result_a = first.run_all_scenarios(runs=200, quarters=4)["combat_exit"].states
    result_b = second.run_all_scenarios(runs=200, quarters=4)["combat_exit"].states
    assert np.allclose(
        result_a.drop(columns=["scenario_id"]).to_numpy(),
        result_b.drop(columns=["scenario_id"]).to_numpy(),
    )

