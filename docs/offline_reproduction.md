# Offline scenario reproduction and validation scope

Use Python 3.13.5 with exact runtime and transitive pins in `requirements.txt`. Test tools are pinned separately in `requirements-dev.txt`. These are tested software versions, not a reconstruction of the undocumented environment behind published illustrations.

```bash
python -m pip install -r requirements.txt
python sample_run.py
```

The separate runner loads six existing versioned CSVs: scenarios, actors, actions, actor/action affinities, country exposures, and the source register. It calls the unchanged actor-choice, scenario simulation, and output-table methods. Default settings match the original entry point: seed 20260727, 4,000 runs per scenario, and 20 quarters. The original scenario-specific seed offsets and economic-noise seeds are retained.

These are analyst-defined conditional assumptions involving real actor/economy labels, not synthetic observations or current source data. No live refresh, market overlay, historical calibration run, or coefficient promotion occurs. The original structural coefficients are unchanged. This sample does not represent a new empirical validation or updated market assessment.

## Output and repeatability

The sample creates nine `sample_*.csv` tables: scenario definitions, actor/action probabilities, scenario summary, economic impacts, pathway probabilities, action frequency, country vulnerability, sensitivity analysis, and terminal states. It also writes an interpretation note and `run_manifest.json` with settings, input/model/runner/dependency-file hashes, Python/platform/numerical package versions, and artifact hashes. Timestamps and absolute paths are omitted.

The default destination is a new `output/sample/` directory. Existing destinations are refused. Repeat with `python sample_run.py --output-dir output/sample/repeat`. Relative paths resolve from the repository root; in-repository destinations must be under `output/sample/`. New external directories are supported. Optional `--runs`, `--quarters`, and `--seed` change only sample settings; the existing minimum of 200 runs and four quarters is enforced, and seeds must be nonnegative.

Repeated artifacts are tested for byte equality within the same recorded environment. Cross-platform byte identity is not promised. Exact version pins are not an archive of packages or operating-system libraries. The sample does not regenerate charts/PDFs or claim parity with the published report.

## Verification and preservation

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

All 27 tests pass locally on Windows with Python 3.13.5. The original model, live-context, and calibration tests are retained. New checks cover hand-calculated scenario indices, exact repeated tables, seed sensitivity, finite/bounded outputs, economic quantile ordering, repeated artifact hashes, independent network-blocked processes, invalid settings, protected destinations, and preservation hashes. Tests block socket connections and require no external data.

[GitHub Actions](../.github/workflows/tests.yml) installs the pinned environment, checks dependencies, runs all tests and the default sample on Windows and Ubuntu, verifies unchanged tracked files, and uploads sample artifacts. See [actual workflow runs](https://github.com/jasonrkeen/us-middle-east-strategic-withdrawal-model/actions/workflows/tests.yml) for execution evidence.

[Preservation hashes](../tests/preserved_files.json) record the original entry point, all model/calibration/reporting code, versioned inputs, published outputs, original tests, and changelog at the baseline commit specified in that file. Text hashes normalize checkout line endings; binary hashes preserve exact bytes. Git attributes prevent PDF/PNG line-ending conversion on Windows. Updating preservation references requires authorization for the protected changes, not merely making tests pass.

Model logic, published results, historical artifacts, calibration-promotion rules, and releases are unchanged. Historical calibration remains diagnostic under the existing rules. Passing software checks does not establish causal identification, forecast validity, independent replication, or current data quality.
