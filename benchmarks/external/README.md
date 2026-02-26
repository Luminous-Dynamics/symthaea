# External Benchmarks

This directory contains external benchmark suites that evaluate Symthaea's
capabilities with the Cyborg Mind configuration:

- HDC primitives (reasoning + control)
- Local-only LLM as a translation organ (no remote APIs)
- ActionIR tool bindings with strict budgets

## Suites

- `arc-agi-2/`           ARC-AGI-2 generalization + efficiency
- `gaia/`                GAIA public dev (tool-use assistant)
- `osworld-verified/`    OSWorld-Verified on NixOS
- `swe-bench-verified/`  SWE-bench Verified (Mini subset)
- `helm/`                HELM Capabilities + HELM Safety (official)

## How It Works

Each suite provides:

- `fetch.sh`  dataset/tool acquisition
- `run.sh`    execution that writes a normalized result JSON
- `data/READY` marker file once data is in place

Use `benchmarks/manifest.json` as the registry and
`benchmarks/REPRO.md` for the reviewer path.

If you want to plug in a custom agent, see `benchmarks/external/agent_protocol.md`.

## Reviewer Run (All Suites)

From the repo root:

```
benchmarks/external/run_all.sh
```

Override the environment file if needed:

```
export SYMTHAEA_BENCH_ENV=benchmarks/external/config/cyborg_mind.env
benchmarks/external/run_all.sh
```

## Configuration

Local LLM model is selected via:

```
export SYMTHAEA_LLM_MODEL=phi4:mini-q4_K_M
```

For ActionIR execution (non-simulated):

```
export SYMTHAEA_ALLOW_REAL_EXEC=1
```

Agent brain settings (HDC+CfC/LTC):

```
export SYMTHAEA_AGENT_PROFILE=standard
export SYMTHAEA_AGENT_TEMPORAL_BACKEND=cfc
export SYMTHAEA_AGENT_CYCLES=3
export SYMTHAEA_AGENT_CONTEXT_LIMIT=4000
# export SYMTHAEA_AGENT_GENESIS=cyborg-mind-v1
```

A sample environment file is provided at `benchmarks/external/config/cyborg_mind.env`.

## Result Schema

See `benchmarks/external/results/README.md`.
