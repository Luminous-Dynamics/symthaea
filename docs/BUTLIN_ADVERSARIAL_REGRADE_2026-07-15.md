# Butlin Indicator Validation — Adversarial Re-Grade (2026-07-15)

**Verdict up front: 1 PRESENT / 6 PARTIAL / 7 NOT DEMONSTRATED** — against the harness's
reported 14/14 PRESENT.

This document re-grades the 14 Butlin et al. (2023) consciousness indicators against
*measured* July 2026 evidence, adversarially: an indicator is only PRESENT if the mechanism
exists **and** there is measured evidence it functions. It supersedes the interpretation
(not the raw data) of `docs/BUTLIN_VALIDATION_RESULTS.md` (2026-03-21) and the 2026-07-15
re-run of `examples/butlin_validation.rs`.

## Why the harness's 14/14 is self-graded

Verified against `examples/butlin_validation.rs` and the shared
`symthaea-psych-bench/src/benchmarks/butlin/indicators.rs` (2026-07-15):

1. **Every indicator starts from a hand-assigned constant** (`static_score`: 0.70–1.00).
   The shared module even documents raising constants with literature citations — citations
   justify the *constant*, they are not measurements of this system.
2. **Half the indicators never see runtime data at all.** In the example harness, GWT-1,
   GWT-2, HOT-1, PP-1, PP-2, AST-1, and AST-2 are pure constants — the 150 cycles of live
   loop data cannot move them.
3. **The runtime blend saturates.** The 7 blended indicators use
   `normalize_phi(x) = 2/(1+e^(−x))−1`, a sigmoid that is ≈1.0 for any structural Φ above
   ~5. Measured structural Φ is 20–150, so the "runtime" component is effectively the
   binary fact "structural Φ is nonzero," providing no discrimination.
4. **The PRESENT threshold cannot fail given the constants.** With every static ≥ 0.70,
   blend = 0.6·static + 0.4·runtime, and saturated runtime ≈ 1.0, every score lands ≥ 0.82.
   The "CI gate: expect 14/14" is a ratchet on a computation that cannot produce another
   answer under realistic inputs.
5. **The harness collects contradicting measurements and ignores them.** `pipeline_phi`
   (measured **0.0000**), `prediction_coherence`, and `attention_prediction_accuracy`
   (measured **0.23–0.26**) are collected, printed, and used in **no** indicator score.

## Measured evidence used for re-grading

| Source | Finding |
|--------|---------|
| E1 subsystem ablation (2026-07-08, re-verified 2026-07-15) | 13/15 loop subsystems NULL causal load; only `meta_cognition` (moves consciousness level 0.74→0.61) and `embodied_cognition` (moves Ψ) load-bearing. GWT, prefrontal, predictive-processing, phi-attention, phenomenal-binding, dream-replay all Δ≈0. Φ frozen at 181.0273 (4 d.p.) across nearly all arms. |
| E1/E2 signal audit | Prediction error frozen at exactly **1.0000** in every arm — the prediction path is disconnected or never learns. Loop Ψ dynamic range ~0.03. |
| E3 tick-rate ablation | delta_t 50 Hz → 1 Hz changes outputs by nothing to 4 d.p. |
| E6 facade↔loop divergence | The two cognitions share no state; their consciousness signals **anticorrelate** (r = −0.76). |
| 2026-07-15 butlin_validation run | Pipeline Φ = 0.0000; AST prediction accuracy 0.24 (CfC) / 0.24 (HCfC); structural Φ micro/meso/macro 33/20/134 (CfC), 39/29/150 (HCfC). |

## Per-indicator re-grade

| ID | Description | Harness | Re-grade | Evidence |
|----|-------------|---------|----------|----------|
| RPT-1 | Algorithmic recurrence | 1.000 PRESENT | **PARTIAL** | CfC state feedback genuinely runs every cycle and the backend swap (CfC vs HCfC) measurably changes structural values — recurrence exists. But E3 shows outputs invariant to tick rate and PE is frozen, so no measurement shows the recurrent *temporal* dynamics doing cognitive work. |
| RPT-2 | Integrated perceptual representations | 0.800 PRESENT | **PARTIAL** | HDC bundling integrates features by construction. No functional test distinguishes integrated from separable percepts, and E1 found `phenomenal_binding` causally NULL. |
| GWT-1 | Multiple specialized processors | 0.900 PRESENT (pure constant) | **PARTIAL** | The 12-region / multi-subsystem architecture exists as code, but E1 measured 13/15 of those "specialized processors" carrying zero causal load — modules that influence nothing are not functioning processors. |
| GWT-2 | Global broadcast mechanism | 0.850 PRESENT (pure constant) | **NOT DEMONSTRATED** | Directly contradicted: `enable_gwt = false` produced Δ≈0 on every output metric (E1). Additionally the system's two cognitions share no state at all (E6, r = −0.76) — there is no *global* broadcast even between its own halves. |
| GWT-3 | Information integration across modules | 0.880 PRESENT | **NOT DEMONSTRATED** | Scored from saturated meso-Φ. The integration measure is invariant (4 d.p.) to ablating the integrative machinery it supposedly reflects (GWT, binding) — the signal does not measure cross-module integration. |
| HOT-1 | Higher-order representations | 0.850 PRESENT (pure constant) | **PRESENT** | The one indicator the ablations *support*: `meta_cognition` is one of only two load-bearing subsystems — disabling it measurably moves consciousness level (0.743→0.607). Mechanism exists and demonstrably functions. |
| HOT-2 | Misrepresentation possibility (prefrontal veto) | 0.850 PRESENT | **NOT DEMONSTRATED** | Directly contradicted: `enable_prefrontal = false` is causally NULL (E1). A veto that changes nothing when removed is not a demonstrated veto. |
| PP-1 | Hierarchical predictive model | 0.850 PRESENT (pure constant) | **NOT DEMONSTRATED** | The strongest contradiction in the set: prediction error sits at exactly 1.0000 (maximum) in every measured arm — the model demonstrably is not predicting — and `enable_predictive_processing = false` is causally NULL. |
| PP-2 | Hierarchical prediction at multiple scales | 0.850 PRESENT (pure constant) | **NOT DEMONSTRATED** | HierarchicalCfC (4 tau levels) exists, but with PE frozen at maximum there is no evidence of prediction at *any* scale, let alone multiple. E3's tick-rate invariance further undercuts multi-timescale processing. |
| AST-1 | Self-model of attention | 0.850 PRESENT (pure constant) | **PARTIAL** | The attention schema genuinely runs and produces a measurable self-prediction — which is wrong ~76% of the time (accuracy 0.23–0.26). A functioning-but-poor self-model; the harness collects this number and ignores it. |
| AST-2 | Attention influences processing | 0.900 PRESENT (pure constant) | **NOT DEMONSTRATED** | Directly contradicted: `enable_phi_attention = false` is causally NULL (E1). No measurement shows attention modulating downstream processing. |
| IIT-1 | Non-zero integrated information | 0.820 PRESENT | **PARTIAL** | Structural Φ is genuinely computed (SpectralMIPFinder, validated r=0.9866 against exhaustive search on the same Gaussian-MI proxy) and is nonzero — the literal bar is met. Caveats: pipeline Φ measured 0.0000, and structural Φ is invariant to subsystem ablation, so what it integrates over is unclear. |
| IIT-2 | Exclusion (single maximum) | 0.820 PRESENT | **NOT DEMONSTRATED** | Scored by an unrelated quantity (macro-Φ magnitude). The MIP search returns a single partition *by construction*; no test measures uniqueness/stability of a Φ maximum, which is what the exclusion postulate claims. |
| IIT-3 | Intrinsic causal structure | 0.820 PRESENT | **PARTIAL** | Some intrinsic causal structure is genuinely measured — E1 itself demonstrates two causally load-bearing subsystems and a real MI graph. But the same experiment shows most claimed causal structure absent, and the indicator's score derives from Φ magnitude, not any causal measurement. |

## Summary

| Grade | Count | Indicators |
|-------|-------|------------|
| PRESENT | **1** | HOT-1 |
| PARTIAL | **6** | RPT-1, RPT-2, GWT-1, AST-1, IIT-1, IIT-3 |
| NOT DEMONSTRATED | **7** | GWT-2, GWT-3, HOT-2, PP-1, PP-2, AST-2, IIT-2 |

Two points of honesty in both directions:

- **This is not a claim that the mechanisms don't exist.** All 14 mechanisms exist as real,
  compiled, running code. NOT DEMONSTRATED means the *functional* claim is either purely
  self-asserted or contradicted by the ablation/signal measurements. Several downgrades trace
  to two shared root causes (PE≡1.0 and the NULL-causal-load subsystems) — fixing those two
  input problems could legitimately restore multiple indicators at once.
- **The March report's own "12 present / 2 partial" was closer to honest than the current
  14/14**, and the March doc printed the contradicting values (Pipeline Φ 0.0000, AST accuracy
  0.35) without using them — the raw data has been honest all along; the scoring layer has not.

## Path back to PRESENT (specific measurement per indicator)

| ID | Required measurement |
|----|----------------------|
| GWT-2 | An ablation where `enable_gwt = false` produces a nonzero Δ on an output metric — i.e., first make broadcast load-bearing, then re-run E1. Post-Seam-C, add a facade↔loop shared-state test (the bridge landing is the natural fix). |
| GWT-3 | An integration metric that *responds* to removing integrative machinery: re-run E1 asserting meso-Φ drops when GWT/binding are disabled. If it doesn't, the metric — not the grade — is wrong. |
| HOT-2 | A red-team case where the prefrontal veto demonstrably overrides a lower-level judgment (input crafted to trigger veto; assert output differs from veto-disabled arm). |
| PP-1/PP-2 | Fix the frozen-PE bug first (prediction path disconnected — see the signal-integrity sprint). PRESENT requires PE that varies with input predictability and decreases on repeated stimuli; PP-2 additionally needs per-tau-level error traces that differ across timescales. |
| AST-1 | Already measurable: raise attention self-prediction accuracy meaningfully above chance and *use* the collected `attention_prediction_accuracy` in the score instead of a constant. |
| AST-2 | Re-run E1 asserting `enable_phi_attention = false` changes attention-weighted downstream outputs (e.g., which stimuli reach episodic memory). |
| IIT-1 | Show structural Φ responds to a manipulation that should change integration (e.g., severing the state graph) — distinguishing it from a constant of the architecture. |
| IIT-2 | A direct exclusion test: perturb the state and verify the MIP maximum is unique and stable vs. degenerate (multiple near-equal partitions). |
| IIT-3 | Score from an actual causal measurement (the E1 harness is exactly this) rather than Φ magnitude: intrinsic causal structure = number/strength of load-bearing components. |
| RPT-1/RPT-2, GWT-1 | Largely restored as a side effect of the above: once PE varies and more subsystems carry measured load, recurrence/integration/specialization claims inherit real evidence. |

**Recommended harness change**: replace the `static_score` constants with per-indicator
measured signals (the table above names each), keep the 14/14 CI gate only after the
measured version passes once, and have the gate *fail* — not warn — when a collected
runtime signal contradicts an indicator it bears on.
