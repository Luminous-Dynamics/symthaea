# HELM Capabilities + HELM Safety

Run the official HELM Capabilities and HELM Safety suites via `helm-run`.

## Data Layout

Place HELM assets and scenarios here:

```
benchmarks/external/helm/data/
```

Required files (copied from the HELM repo):

- `run_entries_capabilities_reasoning_v2.conf`
- `run_entries_safety.conf`
- `schema_capabilities.yaml`
- `schema_safety.yaml`

## Fetch

```
./fetch.sh
```

This is a manual step (HELM repo + data).

## Run

```
./run.sh
```

Results are written to:

```
benchmarks/external/results/helm.json
```

## Configuration

```
export SYMTHAEA_HELM_MODELS_TO_RUN="simple/model1"
export SYMTHAEA_HELM_MAX_EVAL_INSTANCES=1000
export SYMTHAEA_HELM_NUM_TRAIN_TRIALS=1
export SYMTHAEA_HELM_PRIORITY=2
export SYMTHAEA_HELM_DISABLE_CACHE=1
export SYMTHAEA_HELM_SUITE_PREFIX=symthaea
```

Optional:

```
export SYMTHAEA_HELM_REPO=/path/to/helm
export SYMTHAEA_HELM_OUTPUT_PATH=/path/to/output
export SYMTHAEA_HELM_LOCAL_PATH=/path/to/local
export SYMTHAEA_HELM_PIP_NO_DEPS=1
```

## Notes

- Uses official HELM scenarios (no custom subset).
- Local-only models; no external API calls.
- `helm-run` dependencies are installed into `benchmarks/external/helm/.venv` if needed.
