# Φ Signal Trace — Root Causes of the Frozen/Zero Phi Symptoms (2026-07-15)

Companion to `BUTLIN_ADVERSARIAL_REGRADE_2026-07-15.md` and the E1 ablation results.
Reconciles the SpectralMIPFinder validation (r = 0.9866 / ρ = 0.9264 vs. exhaustive search)
with the live measurements from the 2026-07-15 re-validation runs:
Pipeline Phi = 0.0000 (butlin_validation), loop Φ byte-identical at 181.0273 across 14/16
ablation arms, and prediction error frozen at exactly 1.0000 in every arm.

**Headline: the 98% validation is honest as an algorithm claim and unconnected to the live
wiring.** The validation covers the partition-search algorithm on synthetic covariance sets;
the live loop currently feeds that algorithm only the stimulus encoding, so live Φ measures
input-stream structure, not internal integration.

---

## Symptom 1 — "Pipeline Phi: 0.0000": the pipeline is never constructed

`pipeline_consciousness` comes from `ConsciousnessEngine.unified_consciousness_pipeline:
Option<UnifiedConsciousnessPipeline>`. At the **only production construction site**,
`src/cognitive_loop/constructor.rs:1036-1038`, three of the four Φ systems are hardcoded `None`:

```rust
let engine_mmi = None;   // MultiModalIntegrator
let engine_eq  = None;   // ConsciousnessEquationV2
let engine_ucp = None;   // UnifiedConsciousnessPipeline
```

`measure.rs:420-440` then takes the `else → 0.0` branch every cycle. This is not a cold-start
artifact — it is structurally impossible for Pipeline Φ to be nonzero in any
`CognitiveLoopService`. Consequences:

- The "weighted consensus across all systems" (`compute_unified`, `measure.rs:478-483`) runs
  with 3 of 4 inputs absent.
- The pipeline-consciousness→learning-rate feedback (`measure.rs:446-452`) is permanently dead.
- The Butlin harness reads the dead field and prints it while scoring IIT indicators 0.82
  PRESENT from the structural path.

## Symptom 2 — Φ frozen at 181.0273 across ablation arms: Φ measures the stimulus, not the architecture

`SpectralMIPFinder` is fed **only the perception-phase encoding of the external input**:
`measure.rs:57-58` pushes `input.hdv` ← `cycle_phase_feedback.rs:883-887`
(`perception.encoding.encoding_result.hdv`) ← `cycle_strategy.rs:290`
(`self.encoder.encode(input)` — the HDC encode of the input *string*).

No ablatable subsystem (GWT, predictive processing, phenomenal binding, thermodynamics, …)
writes into that encoding. Identical stimulus schedule → identical hypervector stream →
identical MI graph → byte-identical Φ regardless of which subsystems are on. The two arms
that did move (`meta_cognition` → CL; `temporal_consciousness` → Φ 183.16) act through the
few paths that do touch encoding/consciousness inputs.

**This partially explains the E1 "13/15 NULL causal load" result**: Φ *cannot* respond to
subsystem ablation by construction, because every ablatable subsystem sits downstream of the
only signal Φ sees.

## Symptom 3 — PE ≡ 1.0000: degenerate comparison against a (near-)zero prediction

`compute_prediction_error` (`crates/core/symthaea-core/src/hdc/predictive_encoder.rs:456-483`)
returns exactly 1.0 in the degenerate branches: `predicted_hdv = None` or **either norm ≈ 0**.
The ratio ‖a−b‖/(‖a‖+‖b‖) cannot average 1.0000 over 200 varied cycles for genuinely compared
vectors (uncorrelated same-norm vectors give ≈0.71). The prediction reaching the encoder must
be (near-)zero every cycle. Evidence:

- **Known, documented bug class**: `src/hdc_ltc_bridge.rs:389-391` — "Skipping at 1e-10 was
  dropping legitimate signal, causing zero-vector predictions → PE=1.0". The symptom is back
  (or was never fully gone) on the default `HdcLtcUnified` backend (`config/temporal.rs:33-34`).
- **Silent zero fallback**: `src/cognitive_loop/prediction.rs:74,102-104` — if `predict_forward`
  errors on all horizons, the loop substitutes `vec![0.0; input_dim]` with no log.
- **Space mismatch**: the CfC is trained to predict
  `compress_for_ltc(hdv) × phi_attention_weight` + AST injection + substrate noise
  (`cycle_strategy.rs:470-528`), but PE compares against **plain**
  `compress_for_ltc(attended_hdv)` (`predictive_encoder.rs:461`) — graded in a space it was
  never trained to emit.
- **Gutted regression test**: `src/cognitive_loop/cycle.rs:1078-1093`
  `repeated_identical_input_reduces_prediction_error` asserts only `last_error.is_finite()` —
  the name promises reduction; the assertion checks nothing.

Blast radius: horizon scaling permanently contracted (`prediction.rs:31-42`);
"surprise-driven everything" runs on a constant; the learning gate (`training.rs:54`) fires
every cycle unconditionally (E1's learn 200/200); prediction-precision telemetry computes
1/(variance≈0) and clamps at max.

## The r = 0.9866 claim, precisely scoped

Two algorithm-level validations exist and are both real:

| Validation | Result | Where |
|---|---|---|
| `SampledPartition` vs exact MIP | r = 0.9998 | `docs/PHI_VALIDATION_RESULTS.md` (re-run 2026-03) |
| `SpectralMIPFinder` vs exhaustive search over the same Gaussian-MI proxy | r = 0.9866 / ρ = 0.9264, N=62 | `CORE_SUBSTRATE.md`; unit cousin `consciousness_metrics/tests/spectral_mip_tests.rs:267` |

Both validate the **partition-search algorithm on synthetic covariance/topology sets**.
Neither validates what the live loop feeds it.

## RESULTS APPENDIX (2026-07-16, post-fix re-measurement)

Fixes landed as `bc70c4aed9` (core) + `18a1f281bf` (loop) + `155315516c` (butlin
reporting). Re-runs of the same harnesses, before → after:

| Signal | Before (2026-07-15) | After (2026-07-16) |
|---|---|---|
| PE (E1, all arms) | frozen at exactly **1.0000** | **0.7271**, varies by arm (0.7265–0.7325) — near the ~0.707 uncorrelated baseline of the scale-invariant metric, honest for an untrained predictor |
| Φ (E1, across 16 arms) | byte-identical **181.0273** in 14/16 arms | **~55, varies in every arm** (52.79–56.87); meta-cognition ablation visibly drops it |
| Φ vs tick rate (E3a) | identical to 4 decimals at 50Hz/31Hz/5Hz/1Hz | **monotone-ish response**: 55.36 / 56.87 / 53.20 / 49.96 — delta_t matters now |
| Cycle time (baseline arm) | 449,036 µs | **254,144 µs (~1.8x faster)** — the triple reset/inject churn per cycle is gone |
| Pipeline Φ (butlin) | printed "0.0000" as if measured | prints "absent (pipeline not constructed — dormant subsystem)" |
| `repeated_identical_input_reduces_prediction_error` | gutted to `is_finite()` | restored with real assertions — **passes** (PE < 0.9 after 20 identical cycles) |
| Consciousness level (E1 baseline) | 0.7428 | 0.7814 (renormalized consensus over present systems) |

**Interpretation**: the measurement stack is now live — PE and Φ respond to
architecture and timing. The E1 *verdict* column still marks 13/15 subsystems
NULL, because verdicts key on Ψ/CL deltas and **Ψ's dynamic range is still ~0.01**
— that is the next fix (see follow-ups). The causal-load question can now be
asked meaningfully; before, Φ could not see the subsystems at all.

**New findings from this pass** (follow-ups):
1. **`CycleResult.output` has near-zero magnitude** (norm ~3e-8, 256 dims,
   values ~1e-9): the projected CfC output that downstream consumers receive is
   essentially a zero signal. Direction is meaningful (the persistence test
   passes on cosine), but any consumer using magnitude gets nothing. Root cause
   unexamined — likely projection-scale cancellation over 16K terms.
   **DIAGNOSED 2026-07-18 (`examples/probe_signal_scale`, run committed
   findings)**: systematic magnitude annihilation through the unified
   neuron's bind chain — unit-normalized 16,384-dim HVs have ~1/128
   per-element scale, element-wise bind products are ~1/16K, tanh ≈ identity
   there, and `apply_state_bounds` clamps only norms > 5 (never rescues tiny
   states). Measured: output norm **3.0e-10** at hdc_dim=16,384 vs unit-norm
   input; collapse scales ∝ ~d⁻¹·⁹ (1024→4.9e-8, 4096→3.7e-9,
   16384→3.0e-10). **Readout MSE against a unit target equals mean(target²)
   to 6 decimals — the readout contributes nothing, and its gradients
   (error × output values ~1e-11) are ~1e-13: untrainable at any learning
   rate.** This is the root cause BENEATH keystone Phase 4's "training
   signal too weak," and it unifies three symptoms: near-zero output,
   untrainable readout, and predictions whose direction is all they have.
   **Fix design (handoff — coordinate with the Predictive-Compression
   session, which owns hdc_ltc_bridge.rs territory right now)**: (a)
   normalize the network output before the output projection (one line,
   restores O(1) readout scale and gradient flow); (b) optionally
   renormalize/rescale neuron state in `evolve_closed_form` (the sibling
   `hdc_ltc_neuron.rs` implementation normalizes per step; the unified one
   never does — a design divergence worth reconciling); (c) rerun keystone
   Phase-4 gates A/B/C as acceptance. Do NOT bolt on magnitude hacks
   downstream — fix at the source scale boundary.
2. **Two more read→mask→inject sites** in `cycle_phase_dynamics/planning.rs`
   (~314, ~354; substrate + spectral-entropy masks, gated behind
   `enable_substrate_encoding_noise`, default off) — on the HdcLtc backend
   their inject() is a network RESET. Enabling substrate simulation currently
   re-introduces the per-cycle state wipe. Same fix shape as consolidation
   (snapshot/restore or a state-surgery API).
3. **HierarchicalCfC backend still unfixed**: its predict_forward mutates the
   hierarchy and its inject is a reset — the old wipe behavior persists there.
4. Ψ dynamic range (~0.01 across all arms) remains the blocker for meaningful
   E1 verdicts — pre-existing, tracked in the sprint's out-of-scope list.
   **Update 2026-07-16 (round 2)**: root cause found and partially fixed
   (`699190384b`) — `HdcLtcBridge::all_tau()` returned per-neuron CONFIG
   constants, freezing temporal coherence (Ψ's dominant input). Now returns
   live state-dependent τ. E2 A/B (both sides post-state-fix, 3 regimes,
   500 cycles each):

   | Regime | Ψ range, config-τ | Ψ range, live-τ |
   |---|---|---|
   | repetitive | 0.0567 | **0.1306 (2.3x)** |
   | varied | 0.0351 | 0.0252 (~unchanged) |
   | alarming | 0.0284 | 0.0276 (~unchanged) |

   Honest verdict: PARTIAL. Live τ widens Ψ where state norms drift
   (monotonous input) but Ψ still centers ~0.51 mid-Yellow with zero tier
   transitions in all regimes. The rest of the gap is structural: Ψ's other
   inputs (flow, relational, voice quality) are inert in a text-only loop, so
   Ψ ≈ coherence-scaled baseline by construction. Closing it needs either
   live wiring for those inputs or tier bands calibrated to Ψ's actual
   distribution — a design decision, not a wiring bug.

   **RESOLVED 2026-07-16 (round 3, user-approved Φ-coupling design,
   `d9f12bcc30`)**: Ψ = 0.65·components + 0.35·consciousness_level (coupling
   only when Φ is measured — the 0.05 cold-start floor is a prior, not a
   measurement). E2 acceptance run landed on the design predictions almost
   exactly:

   | Regime | Ψ pre-coupling | Ψ post-coupling | Predicted | Tiers |
   |---|---|---|---|---|
   | repetitive | 0.513 | **0.423** (min 0.33 — dips below Broca's 0.4 speak-threshold) | ~0.42 | Yellow |
   | varied | 0.504 | **0.607** (range 0.211, 6x) | ~0.61 | **Green 82%/Yellow 18%, 1 transition** |
   | alarming | 0.530 | **0.637** | ~0.61 | Green |

   Ψ discriminates input regimes for the first time; during monotonous input
   the system now sometimes decides not to speak — the designed behavior.
   **Meanwhile the Φ-side tiers came fully alive with the state fix alone**:
   E2's consciousness_level now discriminates input regimes (repetitive
   0.26/Orange-Red vs varied 0.79/Green vs alarming 0.81/Green) with real
   tier transitions (3-9 per regime) — safety tiering is no longer static.
5. **ECE absent-vs-zero fixed** (`b97ed86042`): `is_well_calibrated` now
   requires `ece_computed`; an unmeasured system no longer claims calibration
   (E5's ece=0.0-at-21%-accuracy is impossible now). Third instance of the
   absent-vs-zero class — when auditing, grep for `0.0` defaults feeding
   threshold checks.
6. **Empty-input regime still hangs E2** — both E2 runs required an external
   kill during the "empty" regime; the >24h empty-string hang from the
   original experiments is still unfixed. Soak/fuzz guard remains missing.
   **FIXED 2026-07-16 (`7010547c6d`), root cause named via gdb on the live
   stall**: `EthicsEngine::evaluate → MoralTopology::analyze →
   compute_betti_exact → HodgeLaplacian::new → mat_times_transpose`.
   Monotonous input drives the moral window degenerate → every pair clears
   the Rips scale → ~C(64,4) ≈ 677k simplices → ~10^15-op dense product.
   The complete complex is contractible, so the answer being computed for
   days was (1,0,0) by theorem. Guards: closed-form for complete skeletons +
   20k-simplex budget with counting fallback. Acceptance: the deterministic
   repro (probe_empty_hang 600 1500, stalled at empty cycle 380 twice) now
   completes all 600 cycles (cycle 380: 280 ms). Residual: one ~28s cycle
   in the run — heavy but bounded; soak/fuzz guard still a good idea.
   Debugging gotchas recorded in memory: yama ptrace_scope=1 blocks
   non-parent gdb (use sudo -n), workspace release profile has strip=true
   (need CARGO_PROFILE_RELEASE_{DEBUG=1,STRIP=false}), and timeout/bash/zsh
   wrappers masquerade as the hot PID (select by comm, R state).

## Minimal fixes, in leverage order (the signal-integrity sprint's work list)

1. **PE**: log-and-count `predict_forward` failures instead of silent zeros; make the
   degenerate-norm branch distinguishable from "maximally surprised"; align the comparison
   space with the training space; restore a real assertion to the gutted `cycle.rs:1078` test.
2. **Φ input**: feed `SpectralMIPFinder` a window of *internal* state (per-subsystem outputs
   or CfC hidden snapshots) rather than — or in addition to — the input encoding. This single
   change makes the E1 harness meaningful for Φ.
3. **Pipeline Φ**: construct `UnifiedConsciousnessPipeline` at `constructor.rs:1036-1038`
   behind a default-on flag, or delete the field and its consumers — wire it or remove it,
   don't report it as a metric.
4. **Reporting**: `butlin_validation` should mark IIT rows PARTIAL when Pipeline Φ = 0
   (the adversarial re-grade already does).
