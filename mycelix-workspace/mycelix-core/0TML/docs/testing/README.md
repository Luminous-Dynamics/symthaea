# Testing Overview

`poetry run python -m pytest` now runs the entire ZeroTrustML suite with
meaningful skips when optional dependencies are missing.

## Required Prerequisites per Suite

| Test Module | Requirement to Run | Default Action |
|-------------|--------------------|----------------|
| `test_official_api.py` | Rust bridge (`holochain_credits_bridge` with admin API) and running conductor | Skipped when bridge missing |
| `test_install_dna.py` | Rust bridge + `zerotrustml-dna/zerotrustml.happ` accessible via admin API | Skipped when bridge missing |
| `test_real_conductor.py` / `test_reconnection*.py` | Real conductor reachable at `ws://localhost:8888` | Skipped when conductor/bridge absent |
| `test_raw_admin_api.py` | Real conductor accepting admin WebSocket connections | Skipped when not running |
| `tests/test_phase4_integration.py` | Optional deps (`cryptography`, `zstandard`, `redis`, `prometheus-client`) and monitoring stack | Skipped when deps missing |
| `tests/test_40_50_bft_breakthrough.py` | Full CIFAR-10 datasets + conductor infrastructure | Skipped by default |
| `test_refactored_bridge.py` | Legacy manual harness for Phase 7 refactor | Marked skipped (manual) |

Run a full battery (including these suites) with:

```bash
# Ensure conductor + bridge + monitoring stack are running
just check-rust
just py-test  # now runs all suites; skipped ones will execute when deps provided
```

Document any new optional suite here so the skip conditions remain obvious.

## Manual 30% BFT Harness

The coordinate-median + committee validation harness lives in
`tests/test_30_bft_validation.py`. It is guarded by a `pytest.skip` call so
regular `pytest` runs stay fast. To run the full CIFAR-10 experiment:

```bash
cd 0TML
RUN_30_BFT=1 poetry run python tests/test_30_bft_validation.py
```

The latest run (2025-10-21) achieved **100% detection** with **0% false
positives**; see `30_BFT_VALIDATION_RESULTS.md` for details.

Environment knobs:

- `BFT_DATASET=cifar10|emnist_balanced|breast_cancer`
- `BFT_DISTRIBUTION=iid|label_skew` (label-skew applies a Dirichlet 0.1 split
  and is expected to surface FAIL results until committee heuristics mature)
- `BFT_RATIO=0.30|0.40|0.50` to control the Byzantine fraction (defaults to 0.30)
- `ROBUST_AGGREGATOR=coordinate_median|trimmed_mean|krum` to toggle aggregation
  strategy when experimenting with hybrid defences
- `RUN_30_BFT=1` still guards the heavy CIFAR/EMNIST downloads

To target a specific dataset, either export `BFT_DATASET` (`cifar10`,
`emnist_balanced`, `breast_cancer`) or pass the name directly via the harness
API. Each profile ships with a compatible model/loss combo so the PoGQ +
reputation stack can stress both vision and tabular gradients.

To sweep multiple attack types without hitting missing-lib errors, invoke the
matrix harness via the Nix shell (it uses the system Python with packaged
libraries):

```bash
nix develop --command poetry run python scripts/generate_bft_matrix.py
# optional trend plot
nix develop --command poetry run python 0TML/scripts/plot_bft_matrix.py

# sweep individual attack types (33/40/50%) with ML overrides
USE_ML_DETECTOR=1 nix develop --command poetry run python scripts/run_attack_matrix.py
```

Each run stores `results/bft-matrix/matrix_<timestamp>.json` so we can track
success rates over time; link notable outputs in documentation when tuning
thresholds. Tabular profiles (e.g., `breast_cancer`) reuse the synthetic PoGQ
probe to avoid bias from the small private validation set while we build
deterministic edge-proof datasets. Label-skew sweeps intentionally stress the
system—track those JSON artefacts to quantify committee tuning progress.

Use `scripts/sweep_label_skew.py` to iterate over PoGQ/reputation thresholds,
robust aggregators, and BFT ratios when trying to close the remaining label-
skew gaps. The script writes `label_skew_sweep_<timestamp>.json` under
`results/bft-matrix/`; feed the output into `scripts/analyze_sweep.py` to
identify parameter sets that satisfy the current target (≥ 90 % detection,
≤ 5 % false positives). The matrix runner loops across ratios (0.30/0.40 by
default) and refreshes `results/bft-matrix/latest_summary.md` so we can spot
regressions quickly.

## Local EVM Harness (Anvil)

Edge-proof committee attestations can be validated against an Anvil
instance. If you are in `nix develop` (from the project flake) Anvil is already
available; otherwise install Foundry/Anvil manually, e.g.:

```bash
nix profile install 'github:NixOS/nixpkgs/nixpkgs-unstable#foundry'
# or one-off:
nix-shell -I nixpkgs=channel:nixpkgs-unstable -p foundry --run anvil
```

Optional Python deps for the integration script/tests:

```bash
poetry add web3 asyncio py-solc-x
```

Then run the integration test:

```bash
poetry run python -m pytest 0TML/tests/test_ethereum_backend_local.py
```

## Polygon Simulation (ProofLog.sol)

- Start Anvil in another terminal and deploy `contracts/ProofLog.sol`:
  `poetry run python -m pytest 0TML/tests/test_polygon_attestation.py`.
- This verifies the `EthereumBackend.record_committee_attestation` path
  against a Solidity contract before moving to Polygon Amoy.

## Manual End-to-End Flow

```bash
poetry run python scripts/run_edge_proof_integration.py
```

## Holochain Sandbox (WIP)

`tests/test_holochain_sandbox.py` is intentionally skipped until the Holonix
stack (conductor, lair, bridge) is wired into automation. Run the test inside
the Holonix Docker shell once `hc sandbox` launches multiple conductors and
export `HC_SANDBOX_PATH` / `HC_ADMIN_URL`; replace the skip with assertions that
store and fetch edge proofs through the Rust bridge.
