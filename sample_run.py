"""Offline conditional-scenario demonstration using unchanged versioned assumptions."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

import pandas as pd
from src.model import StrategicWithdrawalModel

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SampleSettings:
    seed: int = 20260727
    runs: int = 4000
    quarters: int = 20

    def __post_init__(self) -> None:
        if self.seed < 0 or self.runs < 200 or self.quarters < 4:
            raise ValueError("Seed must be nonnegative; use at least 200 runs and four quarters")


def text_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def destination_path(path: Path) -> Path:
    destination = (ROOT/path).resolve()
    if destination == ROOT or destination in ROOT.parents:
        raise ValueError("Output cannot be the repository or its parent")
    if ROOT in destination.parents and not destination.is_relative_to((ROOT/"output/sample").resolve()):
        raise ValueError("Repository output must be under output/sample")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    return destination


def calculate_sample(settings: SampleSettings) -> dict[str,pd.DataFrame]:
    model = StrategicWithdrawalModel(ROOT,seed=settings.seed)
    probabilities = model.calculate_action_probabilities()
    results = model.run_all_scenarios(runs=settings.runs,quarters=settings.quarters)
    tables = model.build_output_tables(probabilities,results)
    tables["terminal_states"] = pd.concat([r.states for r in results.values()],ignore_index=True)
    return tables


def run_sample(output: Path = Path("output/sample"), settings: SampleSettings = SampleSettings()) -> dict[str,Any]:
    destination = destination_path(output)
    tables = calculate_sample(settings)
    destination.mkdir(parents=True,exist_ok=False)
    for name,frame in tables.items():
        frame.to_csv(destination/f"sample_{name}.csv",index=False,lineterminator="\n")
    (destination/"README.md").write_text(
        "# Offline conditional-scenario sample\n\n"
        "Versioned actor/scenario assumptions, not current observations or forecasts.\n"
        "Original structural coefficients retained; no calibration promotion or live overlay performed.\n"
        "These sample artifacts do not replace published reports or establish empirical validation.\n",
        encoding="utf-8",newline="\n")
    inputs = [ROOT/"data"/name for name in ("scenarios.csv","actors.csv","actions.csv",
              "actor_action_affinity.csv","country_exposures.csv","source_register.csv")]
    manifest = {
        "status":"OFFLINE_CONDITIONAL_SAMPLE_NOT_EMPIRICAL_VALIDATION", "settings":asdict(settings),
        "input_sha256_lf":{p.relative_to(ROOT).as_posix():text_hash(p) for p in inputs},
        "source_sha256_lf":{p.relative_to(ROOT).as_posix():text_hash(p) for p in [ROOT/"sample_run.py",ROOT/"src/model.py"]},
        "requirements_sha256_lf":text_hash(ROOT/"requirements.txt"),
        "environment":{"python":platform.python_version(),"system":platform.system(),
                       "packages":{name:version(name) for name in ("numpy","pandas")}},
        "artifact_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir())},
        "limits":"Repeatability within a recorded environment; no live-context or historical-calibration run, published-result parity, or predictive-validity claim.",
    }
    (destination/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",
                                               encoding="utf-8",newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/sample"))
    parser.add_argument("--seed",type=int,default=SampleSettings.seed)
    parser.add_argument("--runs",type=int,default=SampleSettings.runs)
    parser.add_argument("--quarters",type=int,default=SampleSettings.quarters)
    args = parser.parse_args()
    try:
        run_sample(args.output_dir,SampleSettings(args.seed,args.runs,args.quarters))
    except (ValueError,FileExistsError) as exc:
        parser.error(str(exc))
    print("Offline conditional sample complete; no live refresh or calibration promotion.")


if __name__ == "__main__":
    main()
