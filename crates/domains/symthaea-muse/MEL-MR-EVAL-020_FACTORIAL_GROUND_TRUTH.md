# MEL-MR-EVAL-020 — Factorial Ground-Truth Benchmark

**Status:** first deterministic reference-engine slice implemented on a branch stacked on MEL-MR-EVAL-019.  
**Claim ceiling:** exhaustive synthetic coalition renders and samplewise Möbius/Harsanyi dividends for explicitly defined mathematical effects. No FluidSynth calibration, real-music benchmark, stochastic-renderer validation, human-listening study or SOTA improvement is claimed.

## Research question

MEL-MR-EVAL-018 measures a second-order event interaction from four verified render coalitions. MEL-MR-EVAL-019 allocates a limited render budget to candidate pairs. Both need a known-answer reference to separate measurement error from candidate-selection error.

This increment builds that oracle for small synthetic scenes. For N events it enumerates every coalition:

\[
\mathcal{P}(N)=\{S:S\subseteq\{0,\ldots,N-1\}\},\quad |\mathcal{P}(N)|=2^N
\]

For a representation \(\Phi\), it computes the Möbius/Harsanyi dividend of each non-empty subset:

\[
H_\Phi(S)=\Phi(Y(S))-\sum_{T\subsetneq S}H_\Phi(T)
\]

This gives explicit pairwise and higher-order ground truth instead of assuming all interaction can be represented by edges in a graph.

## Implemented engine

Module: `src/factorial_ground_truth.rs`

Public API:
- `build_factorial_ground_truth(events, effect, config)`
- `verify_factorial_ground_truth(report, events, effect, config)`

The input fixture contains event-indexed, equal-length, finite source-isolated signals. Event indices must be canonical and complete. Small-scene exhaustive enumeration is bounded to 2–8 events; sample frames are bounded to 8,192, and the spectral window must be a power of two.

### Effects implemented

1. **LinearMix** — plain event summation.
2. **LinearFir** — fixed causal FIR convolution applied after summation. The output includes its tail frames. This is a linear, time-invariant control and should have no second-or-higher-order PCM dividends.
3. **Polynomial** — shared pointwise processor \(y=x+qx^2+cx^3\), with explicit finite/bounded coefficients. Produces analytically known pair and higher-order dividends.
4. **HardClip** — shared static hard limiter; a simple nonlinear positive control.
5. **ThreeWayProduct** — explicit \(g x_a x_b x_c\) term, creating a pure third-order interaction while the pairwise dividends remain zero.

The reference engine stores all coalition PCM outputs and a fixed first-window Hann/rFFT magnitude feature for each coalition. It then computes two independent dividend families:
- **PCM Harsanyi dividends:** samplewise waveform effects.
- **Magnitude-spectrum Harsanyi dividends:** effects after a nonlinear feature transform.

The spectral feature is a single leading window, not a full multi-resolution/perceptual representation. It is designed to provide a tightly specified phase-interference control, not a quality score.

The report includes exact coalition masks/event indices, source/effect/config digests, per-coalition artifacts, all non-empty dividends, norms/peaks and a replayable SHA-256 receipt.

The Möbius transform uses the fast Boolean-lattice algorithm, \(O(N2^N F)\) rather than explicit proper-subset enumeration \(O(3^N F)\), where F is the feature-vector length.

## Source-level tests authored

- Linear summation produces zero pairwise/higher PCM dividends within numerical tolerance.
- Fixed FIR convolution preserves PCM additivity and exposes its tail frames.
- Quadratic shared processing yields the known pair dividend \(2qab\).
- A pure three-way product has zero pairwise dividends and the exact third-order value \(gabc\).
- Phase-quadrature signals produce a spectral-magnitude pair dividend while their PCM pair dividend remains zero.
- Hard clipping produces a known shared-processing interaction.
- Reports replay from source inputs and reject digest tampering.
- Non-canonical event indices, non-finite input and invalid effect parameters fail closed.

Independent arithmetic sanity checks for the fixture constants give: quadratic pair dividend 0.06 per frame, pure three-way dividend 0.012 per frame, hard-clip pair dividend -0.6 per frame, and a phase-quadrature magnitude residual peak of approximately 9.37 in the chosen unnormalized FFT representation. These are formula checks for the synthetic fixtures, not Rust test results or acoustic measurements.

## Still required

1. Expand the effect family: soft saturation, attack/release compressor, limiter with release, side-chain ducking, chorus/modulation, voice-stealing/resource contention and seeded stochastic effects.
2. Add randomized fixtures with known effect equations and a preregistered train/development/held-out split by score, renderer/effect family and parameter settings.
3. Add exhaustive candidate-edge truth for larger N while using all-coalition enumeration only for small-N fixtures.
4. Add common-random-number records and repeated blocks before measuring stochastic interactions.
5. Add a real FluidSynth adapter that exports exact event-isolated fixture inputs and receipts; synthetic renderer truth must not be represented as proof of FluidSynth behavior.
6. Connect the exhaustive reference to EVAL-018 report measurements and EVAL-019 plan recall/cost evaluations.
7. Add blinded human listening and target/preservation checks as separate product-quality evidence, not as part of the mathematical oracle.

## Benchmark matrix

Compare:
- exhaustive factorial reference (truth ceiling for the synthetic fixture);
- EVAL-018 pairwise measurement against exact second-order dividends;
- EVAL-019 support/topology planner;
- temporal-overlap-only selector;
- shared-routing-only selector;
- random budget-matched selector;
- adaptive uncertainty/information-per-render selector (future extension).

Report:
- PCM pairwise sensitivity/specificity and magnitude error against exact truth;
- magnitude-feature false-positive rate on PCM-additive phase controls;
- higher-order dividend norms and pairwise reconstruction error;
- candidate recall/precision, recall-vs-render-budget curve and area under that curve;
- false-negative audit estimates and confidence intervals;
- deterministic replay, malformed-receipt rejection and abstention;
- per-effect and worst-family performance, not only micro-averaged results.

## External evaluation and industry comparison

- **SMEBench** (October 27, 2026) provides 3,600 paired synthetic test examples across twelve music-edit operations, plus a real-audio complement. It exposes target agreement, source preservation and text compatibility separately. Its analysis warns that common metrics can rank systems differently. It is a music-edit outcome benchmark, not a source-event factorial interaction oracle.
- **MMAE / ICASSP 2027 Audio Editing Challenge** supplies broad audio-editing evaluation with IFR, consistency rate and exact-match rate. It evaluates execution and preservation, not event-level Harsanyi truth.
- **HEARTBEAT Challenge 2027** is an adjacent future evaluation for real-time agent behavior: whether to wait, respond, continue or yield, and when to proactively alert. It is not a music-edit benchmark.
- **XACLE Challenge 2027** highlights the need to evaluate audio-text alignment on generation systems held out from training, due to generalization limits in automated semantic alignment. Use it as a generalization test for perception/quality models if task and data licenses permit.

Keep signal-level correctness, music-edit target/preservation quality, human perception and online agent behavior as separate score families. Do not collapse them into one scalar.

## Validation status

- Rust unit tests are authored but not executed.
- `cargo`, `rustc`, `rustfmt`, `clippy` and `nix` are unavailable in the current execution environment.
- Exact-head CI has not been treated as a pass; Actions marked `skipped` are not evidence of validation.
- No real renderer execution, official dataset run, listener study or performance comparison has been performed.

## Related work

- [Issue #9155 — EVAL-020 benchmark plan](https://github.com/Luminous-Dynamics/symthaea/issues/9155)
- [PR #9151 — EVAL-019 sparse probe planner](https://github.com/Luminous-Dynamics/symthaea/pull/9151)
- [PR #9133 — EVAL-018 pairwise factorial measurement](https://github.com/Luminous-Dynamics/symthaea/pull/9133)
- [PR #9128 — EVAL-017 counterfactual event influence](https://github.com/Luminous-Dynamics/symthaea/pull/9128)
