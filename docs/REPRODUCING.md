# Reproduction levels

## 1. Inspect existing evidence (no model or dataset download)

Install `requirements.txt`, run `python tools/verify_results.py`, and inspect
`results/closed_loop/test/{summary,paired_statistics,episode_trials}.csv`.
The verifier checks raw records against reported aggregate means. This is a
packaging consistency check, not a new scientific experiment.

## 2. Run the procedural simulator

```bash
python src/run_closed_loop_simulation.py --split validation --episodes 2 --seed 20260731 --bootstrap-samples 100 --contract CLOSED_LOOP_SIMULATION_CONTRACT_20260728.md --output-dir scratch/closed_loop_smoke
```

To reproduce the frozen test protocol, change episodes to 1000, split to test,
seed to 20260801 and bootstrap samples to 10000. Use a NEW output directory so
the exported evidence is not overwritten. The historical contract mentions an
RCMR candidate; the frozen implementation and metadata explicitly exclude it
because its validation gate failed. The seven methods in the exported code are authoritative.

## 3. Re-run the static-image benchmark

`python src/run_mr_cwrf_benchmark.py --help` lists inputs. Original imagery,
YOLO labels, the three model prediction files, and frozen partition/selection
artifacts must be obtained separately. This snapshot does not provide a one-command
reproduction of detector training or all static-image conditions.
Raw clean static result tables are included in `results/mr_cwrf/frozen_clean_test`.

## 4. Original analysis code

The analysis entry points are preserved for provenance. Some require historical
validation candidate tables not included here; inspect their required arguments.
`make_closed_loop_mechanism_figure.py` expects `manuscript/figures` to exist;
create that directory first if rerunning it. Existing display figures live in `assets`.

`requirements.txt` pins the local export-check environment, not the historical
detector-training environment. See `FROZEN_SOFTWARE_ENVIRONMENT_20260729.md` for
the latter. No fresh pip installation or GPU inference is implied by a packaging check.
