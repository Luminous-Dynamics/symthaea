# Broca Measurement Loop

Broca measurement artifacts are the promotion surface for checkpoint and
decoder changes. The goal is to compare evidence, not impressions.

## Fast PR Measurement

GitHub Actions runs `scripts/broca_measurement_artifacts.sh` with
`BROCA_SKIP_EXERCISM=1`. This produces:

- `measurement-manifest.env`
- `decoder-ab.json`
- `exercism-bench.json` with `evidence_level=skipped`

This keeps PR jobs bounded while still tracking structured decoder metrics and
semantic drift gates. The structured lane records both confidence and a
validity score, where validity rewards complete semantic frames with the core
`AGENT`, `ACTION`, `PATIENT`, `PREDICATE`, and `EVALUATOR` roles present.
Routine artifacts omit the full HDC molecule vectors to keep CI output small;
set `BROCA_INCLUDE_STRUCTURED_MOLECULE=1` for deep decoder forensics.

## Capture a Local Smoke Baseline

```bash
BROCA_EVAL_LIMIT=8 \
BROCA_SKIP_EXERCISM=1 \
scripts/broca_capture_baseline.sh local-smoke
```

The baseline directory is written under `target/broca-baselines/` by default and
includes `baseline-manifest.env` plus the measurement artifacts.

## Run a Candidate Against a Baseline

```bash
BROCA_BASELINE_ARTIFACT_DIR=target/broca-baselines/local-smoke \
BROCA_SKIP_EXERCISM=1 \
BROCA_FAIL_ON_REGRESSION=1 \
scripts/broca_measurement_artifacts.sh target/broca-measurements/candidate
```

This writes `checkpoint-compare.json` into the candidate artifact directory.

## Full Exercism Measurement

The Exercism path uses CPU Mamba synthesis unless a GPU-backed feature path is
selected elsewhere, so run it deliberately. Locally,
`BROCA_MAMBA_BACKEND=auto` prefers the CUDA-backed `mamba` feature when both
`nvidia-smi` and `nvcc` are available, and falls back to `mamba-cpu` otherwise:

```bash
BROCA_MAMBA_BACKEND=auto \
BROCA_EVAL_LIMIT=8 \
BROCA_EXERCISM_MAX_EXERCISES=1 \
BROCA_EXERCISM_ATTEMPTS=1 \
BROCA_EXERCISM_TIMEOUT=15m \
scripts/broca_capture_baseline.sh full-exercism-smoke
```

Backend selection is centralized in `scripts/broca_runtime_crust.sh`. Scripts
that source it write both the requested backend and the resolved backend into
their manifests:

- `broca_mamba_backend`: requested value, usually `auto`.
- `broca_selected_backend`: resolved value, `gpu` or `cpu`.
- `broca_backend_reason`: why the backend was selected.
- `broca_mamba_feature`: Cargo feature used for Mamba-backed measurement.

Use `BROCA_REQUIRE_ALL_COMPARISON_METRICS=1` only when the baseline and
candidate both include the same decoder/benchmark lanes. Otherwise missing
metrics are recorded in `checkpoint-compare.json` but do not fail promotion.

## Important Knobs

- `BROCA_EVAL_LIMIT`: canonical cases evaluated by decoder A/B.
- `BROCA_MAX_DIRECT_DRIFT`: direct decoder semantic drift gate.
- `BROCA_MAX_MAMBA_DRIFT`: Mamba semantic drift gate.
- `BROCA_MAX_HALLUCINATION_RATE`: hallucination-rate gate.
- `BROCA_MIN_STRUCTURED_VALIDITY`: minimum structured-frame validity gate.
- `BROCA_MIN_STRUCTURED_REQUIRED_ROLE_RATE`: minimum complete-role-frame rate.
- `BROCA_INCLUDE_STRUCTURED_MOLECULE`: include full HDC molecule vectors.
- `BROCA_MAMBA_BACKEND`: `auto`, `gpu`, or `cpu`; `auto` prefers CUDA locally.
- `BROCA_DECODER_FEATURES`: override decoder Cargo feature set.
- `BROCA_EXERCISM_FEATURES`: override Exercism Cargo feature set.
- `BROCA_SKIP_EXERCISM`: write an explicit skipped Exercism artifact.
- `BROCA_EXERCISM_MAX_EXERCISES`: cap Exercism fixtures.
- `BROCA_EXERCISM_TIMEOUT`: shell timeout for full Exercism runs.
- `BROCA_BASELINE_ARTIFACT_DIR`: compare candidate artifacts against a baseline.
- `BROCA_FAIL_ON_REGRESSION`: make comparison failures exit nonzero.
- `BROCA_REQUIRE_ALL_COMPARISON_METRICS`: fail if expected metrics are missing.
