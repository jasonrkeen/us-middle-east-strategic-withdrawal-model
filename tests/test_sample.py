import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import sample_run as sample


def test_hand_calculated_scenario_indices():
    frame = pd.DataFrame([dict(force_reduction=.4,withdrawal_speed=.6,residual_capability=.2,security_guarantee=.5)])
    before = frame.copy(deep=True)
    result = sample.StrategicWithdrawalModel._add_scenario_indices(frame).iloc[0]
    assert result.withdrawal_shock_index == pytest.approx(.56)
    assert result.deterrence_gap == pytest.approx(.56)
    assert result.alliance_ambiguity == pytest.approx(.54)
    pd.testing.assert_frame_equal(frame,before)


def test_all_tables_repeat_and_have_valid_ranges():
    settings = sample.SampleSettings(runs=200,quarters=4)
    a=sample.calculate_sample(settings); b=sample.calculate_sample(settings)
    assert len(a) == 9
    for name in a:
        pd.testing.assert_frame_equal(a[name],b[name],check_exact=True)
        assert np.isfinite(a[name].select_dtypes("number")).all().all(), name
    economic=a["economic_impacts"]
    assert (economic.p05 <= economic["median"]).all()
    assert (economic["median"] <= economic.p95).all()
    assert a["country_vulnerability"].vulnerability_score.between(0,100).all()
    assert len(a["terminal_states"]) == 1000
    assert a["pathway_probabilities"].probability.between(0,1).all()


def test_changed_seed_changes_simulations_not_assumptions():
    a=sample.calculate_sample(sample.SampleSettings(seed=42,runs=200,quarters=4))
    b=sample.calculate_sample(sample.SampleSettings(seed=43,runs=200,quarters=4))
    assert not a["terminal_states"].equals(b["terminal_states"])
    pd.testing.assert_frame_equal(a["scenario_definitions"],b["scenario_definitions"])
    pd.testing.assert_frame_equal(a["actor_action_probabilities"],b["actor_action_probabilities"])


def test_repeat_artifacts_and_manifest(tmp_path):
    settings=sample.SampleSettings(runs=200,quarters=4)
    a=sample.run_sample(tmp_path/"a",settings); b=sample.run_sample(tmp_path/"b",settings)
    assert a == b
    for name,expected in a["artifact_sha256"].items():
        assert hashlib.sha256((tmp_path/"a"/name).read_bytes()).hexdigest() == expected
        assert (tmp_path/"a"/name).read_bytes() == (tmp_path/"b"/name).read_bytes()


def test_independent_offline_processes(tmp_path):
    (tmp_path/"sitecustomize.py").write_text(
        "import socket\ndef blocked(*a, **k):\n raise RuntimeError('Network forbidden')\n"
        "socket.socket.connect=blocked\nsocket.socket.connect_ex=blocked\nsocket.create_connection=blocked\n")
    for name in ["a","b"]:
        result=subprocess.run([sys.executable,str(sample.ROOT/"sample_run.py"),"--runs","200","--quarters","4",
            "--output-dir",str(tmp_path/name)],cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(tmp_path)),
            text=True,capture_output=True,check=True)
        assert "sample complete" in result.stdout
    assert (tmp_path/"a/run_manifest.json").read_bytes() == (tmp_path/"b/run_manifest.json").read_bytes()


@pytest.mark.parametrize("kwargs",[{"seed":-1},{"runs":199},{"quarters":3}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError): sample.SampleSettings(**kwargs)


@pytest.mark.parametrize("path",[sample.ROOT,sample.ROOT.parent,sample.ROOT/"output/pdf/new",sample.ROOT/"data/new"])
def test_protected_destinations(path):
    with pytest.raises(ValueError): sample.destination_path(path)


def test_existing_output_preserved(tmp_path):
    (tmp_path/"sentinel").write_text("keep")
    with pytest.raises(FileExistsError): sample.run_sample(tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path/"sentinel"]
    assert (tmp_path/"sentinel").read_text() == "keep"


def test_preserved_model_inputs_calibration_and_published_artifacts():
    record=json.loads((sample.ROOT/"tests/preserved_files.json").read_text())
    for name,expected in record["files"].items():
        content=(sample.ROOT/name).read_bytes()
        if expected["text_lf"]: content=content.replace(b"\r\n",b"\n")
        assert hashlib.sha256(content).hexdigest() == expected["sha256"],name
