# Symthaea HDC-LTC Runtime Hardening Verification Ledger

This document serves as the formal verification ledger for the Hyperdimensional Liquid Time-Constant (HDC-LTC) neural network runtime contracts and boundary safety features.

## Bounded Verification Target

The verification bounds explicitly test:
- **Floating Point Stability**: Safety against underflows, NaN, and Inf propagation.
- **Timing & Timestep Irregularity**: Strict boundary contracts on arbitrary, backward, sub-minimum, and super-maximum time deltas ($\Delta t$).
- **Analytical Invariants**: Proving that the simplified bounded LTC model preserves state norm bounds under numeric updates ($\|h(t)\| \le 5.0$).

## Hardening Evidence Results

| Test Target | Validation Harness | Status | Context |
|---|---|---|---|
| `symthaea-hdc-ltc` | Unit Tests & Prop Tests | Verified | Bounded parameters, clamping validation |
| `symthaea-probe-stream` | Crate Integration Tests | Verified | Finite-state streams & timing contracts |
| `irregular_timestep_replay` | Example Replay | Verified | Deterministic replay under backward/irregular timing |
| `fol_ext_stability_verification` | SMT / Z3 Formal Check | Verified | Simplified bounded model stability proof |

> [!NOTE]
> *Symbolic Stability Disclaimer:* The SMT / Z3 verification validates a simplified, bounded algebraic model of the LTC update step under explicit assumptions ($0 \le \sigma_i \le 1$, $-1 \le x_{inf, i} \le 1$). It does not model floating point architecture or custom clamping boundaries, which are instead protected by Rust runtime sanity checks and unit/property tests.

## Phase 2.7 — Epistemic Modulation (Active Inference)

### What Was Added

A conservative surprise-driven curiosity modulator was wired into
`NixActiveInference::learn_from_outcome` in `symthaea-nix`.

```
prediction_error > 0.4  →  curiosity_weight += 0.05  (ceiling 0.8)
prediction_error ≤ 0.4  →  curiosity_weight -= 0.02  (floor 0.1)
```

Default `curiosity_weight` is `0.3`. The modulation is applied before
world-model learning updates, so it influences the *next* action scoring
cycle, not the current one.

### Test Inventory

All five tests reside in `src/mind/active_inference.rs` (inline test module):

| Test | What It Proves |
|---|---|
| `test_epistemic_high_surprise_increases_curiosity_bounded` | Single high-surprise step raises weight; ceiling (0.8) respected |
| `test_epistemic_repeated_high_surprise_saturates_safely` | 40 consecutive high-surprise steps never breach 0.8 |
| `test_epistemic_low_surprise_decreases_curiosity_bounded` | Identical-vector transitions lower weight; floor (0.1) respected |
| `test_epistemic_repeated_low_surprise_stabilises_safely` | 40 consecutive low-surprise steps never breach 0.1 |
| `test_epistemic_action_selection_deterministic` | Identical seed + input → identical top-ranked action |

All 11 active-inference tests (6 pre-existing + 5 new) pass under
`CARGO_TARGET_DIR=/tmp/symthaea-ltc-hardening-target`.

### Precise Verification Boundary

> **Verified:** a simplified bounded linear LTC update model preserves
> bounded state under stated assumptions.
>
> **Not verified:** the full floating-point Rust implementation under
> all possible runtime configurations.

The Z3 result is a proof that the *model you encoded is stable under
the constraints you gave it*. It is not a proof of the full
implementation. That distinction is intentional and is preserved here
for reviewer credibility.

## Phase 2.8 — Causal Graph & Drift Detection Hardening

### What Was Hardened

The causal failure path:
```
sensor shift → false drift detection → wrong causal edge → wrong exploration boost → unstable behaviour
```
was exercised systematically for the first time.

### Causal Graph Test Inventory (`causal_graph.rs`)

| Test | Failure Mode Covered |
|---|---|
| `test_self_edge_no_panic_no_infinite_loop` | Self-loop terminates via `visited` set |
| `test_duplicate_edge_no_double_count` | HashMap key uniqueness: no count inflation |
| `test_cycle_traversal_terminates` | A→B→C→A cycle bounded by visited set |
| `test_disconnected_subgraph_isolation` | Components don't bleed across walk queries |
| `test_confidence_never_nan_after_repeated_learning` | 100 Hebbian iters stay in [0,1] |
| `test_side_effects_unknown_node_returns_empty` | Missing node → empty, not panic |
| `test_root_cause_leaf_node_returns_empty` | Source node has no root causes |
| `test_stable_drift_replay_example` | Full replay: normal → sensor miss → missing sensor |

### Drift Detection Test Inventory (`hdc_world_model.rs`)

| Test | Failure Mode Covered |
|---|---|
| `test_drift_tiny_change_no_false_alarm` | Zero-delta obs doesn't trigger alarm |
| `test_drift_massive_change_detected` | Orthogonal input at α=1.0 correctly triggers |
| `test_drift_noisy_observations_stay_stable` | 50 random obs keep similarity finite |
| `test_drift_similarity_always_in_unit_interval` | 7 diverse seeds, invariant [0,1] holds |
| `test_drift_no_expected_state_never_reports_drift_with_arbitrary_obs` | Unknown mode = safe uncertainty, not false certainty |
| `test_drift_facet_scores_finite_and_bounded` | 4 facets all return scores in [0,1] |

### Result

All 137 `mind::` tests pass (0 failures) under
`CARGO_TARGET_DIR=/tmp/symthaea-ltc-hardening-target`.

Commit: `a99a947255` — `symthaea-nix: Phase 2.8 — causal graph and drift detection hardening`

### Precise Verification Boundary

> **Verified:** topology traversal terminates on cycles, self-edges, and
> disconnected graphs under the current visited-set implementation. Drift
> similarity stays in [0, 1] and is never NaN under EMA blending with
> clamped alpha. Causal edge confidence stays in [0, 1] after Hebbian
> strengthening and anti-Hebbian weakening.
>
> **Not verified:** causal discovery correctness (the PC-algorithm
> approximation in `CausalDiscoveryEngine`). Not verified: drift
> thresholds are tuned correctly for production sensor data.
