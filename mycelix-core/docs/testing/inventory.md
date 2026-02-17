# Testing & Experiment Inventory

_Last updated: 2025-10-21_

This document catalogs the current testing and experimentation assets across the Mycelix/Zero-TrustML codebase. Use it as a map before adding new work or archiving legacy flows.

## 1. Automated Test Suites (Pytest)
Location: `0TML/tests/`

| File | Scope |
|------|-------|
| `test_zerotrustml_credits_integration.py` | Core credits/DHT integration (Phase 10 baseline) |
| `test_holochain_credits.py` | Holochain credit issuance helper validation |
| `test_integration_complete.py` | Networked Zero-TrustML node end-to-end harness |
| `test_multi_node_p2p.py` | Multi-node P2P messaging/resilience checks |
| `test_phase4_integration.py` | Legacy Stage 4 integration regression |
| `test_grant_demo_5nodes.py` / `test_grant_demo_5nodes_production.py` | Grant/demo orchestration smoke tests |
| `test_adaptive_byzantine_resistance.py` | Reputation-based adaptive defenses |
| `test_holochain_integration.py` | Admin API + DNA installation workflow |
| `tests/benchmarks/` | (Currently empty placeholder for benchmark-specific assertions) |

CI Shell command: `nix develop .#ci --command bash -c "cd 0TML && pytest"`

## 2. Experiment Pipelines

### 2.1 `0TML/experiments/`
- `run_grand_slam.py`, `run_stage1_set_*.py` – phased experiment orchestration (Phase 0/Stage 1).
- `run_mini_validation.py`, `validate_pogq_integration.py` – PoGQ validation scripts.
- `visualize_results.py` – utilities for plotting experiment outputs.
- `configs/`, `models/`, `utils/` – configuration and helper modules.

### 2.2 `0TML/benchmarks/`
- Dataset loaders (`datasets/*.py`), baseline algorithms (`baselines/*.py`).
- Experiment entry points (`experiments/mnist_accuracy_comparison.py`).
- Visualization scripts (`visualization/plot_mnist_results.py`).
- `monitor_experiment.sh` – helper for long-running benchmark monitoring.

### 2.3 `scripts/experiments/`
- `run_bft_scaling.py` (scaffold) – new runner for Week 1 BFT scaling trials; fill in training loop before execution.

## 3. Results & Data Artifacts
- `0TML/results/` – Mixed outputs from prior stages (needs consolidation; tag new runs with date + experiment name).
- `0TML/experiments/results/` – Stage 1 “Grand Slam” outputs.
- `0TML/test_results/` – JSON summaries for multi-node P2P scenarios.
- `results/` (root) – Legacy experiment outputs from `production-fl-system/` and earlier mocks.

## 4. Legacy / Phase-Specific Systems
- `production-fl-system/` – Older production orchestration (Docker, k8s). Keep for historical reference until migrated.
- `hybrid-trustml/` (archived in git) – Original TrustML project tree; do not modify without migration plan.
- Phase documentation: `0TML/PHASE_*` markdown files summarize completed milestones and should be cross-referenced before removing code.

## 5. Documentation Touchpoints
- Active sprint log: `docs/testing/week-2025-10-20.md`.
- Master roadmap: `docs/testing/master-testing-roadmap.md`.
- Detailed Zero-TrustML plan: `0TML/docs/06-architecture/0TML Testing Status & Completion Roadmap.md`.

## 6. Housekeeping Guidelines
1. **New experiments** → add scripts under `scripts/experiments/` (or augment `0TML/experiments/`) and log them in the sprint doc.
2. **Outputs** → store under `results/<category>/<YYYY-MM-DD>/` with metadata (`config.json`, seed list, hashes).
3. **Retiring legacy code** → update this inventory and the master roadmap so future readers know where to look.
4. **Tests** → extend `0TML/tests/` and run `just ci-tests` before committing.

Keeping this inventory current ensures each testing phase remains discoverable as we move from the submission sprint into Phase 1 validation.
