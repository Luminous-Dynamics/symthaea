# Research Notes

## Purpose

`symthaea-quantum-comp` is a research scaffold for testing quantum-inspired and future quantum-backend hypotheses around hyperdimensional cognition. It is designed to be useful even when every result is negative.

A negative result is valuable if it is reproducible, well-scoped, and honest about assumptions.

## Alpha.3 hypothesis additions

Alpha.3 adds two guardrail-oriented probes.

### Negative control probe

The negative control asks a simple question:

Does recovery behave like HDC binding should behave?

Expected behavior:

- matched key recovery remains high;
- wrong-key recovery trends toward chance;
- random-item similarity trends toward chance;
- the control gap remains visible.

This protects the crate from mistaking deterministic implementation artifacts for meaningful substrate behavior.

### Entanglement proxy probe

The entanglement proxy is not physical entanglement.

It is a classical parity/coherence sketch that gives the project a stable place to test questions like:

- Does explicit pair-coherence metadata change noise behavior?
- Does decoherence-like degradation alter recovery gaps differently from bit-flip noise?
- Do topology proxies shift as coherence collapses?
- What should later QASM or backend adapters measure?

The proxy is intentionally conservative. It creates experiment language without claiming hardware results.

## Claim boundaries

Every serious report should carry a claim boundary:

- implementation check;
- local simulation;
- circuit export only;
- external backend observation.

A local simulation result must not be promoted into a hardware claim.

A circuit export must not be promoted into an executed quantum result.

An external backend result should include backend, transpilation, shots, noise model, date, calibration metadata, and reproducibility artifacts.

## Near-term roadmap

Alpha.4 should focus on one of:

1. a proper experiment manifest file format;
2. CSV/JSON exports behind optional features;
3. CPU/GPU parity hooks for phase-HDC;
4. beta-1 proxy comparison across noise/decoherence schedules;
5. Python notebook interop without making Python mandatory.

## Non-claims

This crate does not claim:

- quantum consciousness;
- quantum advantage;
- physical quantum state preparation;
- validated quantum backend execution;
- production cryptography;
- medical, legal, or safety relevance.

## Alpha.4 reporting discipline

Alpha.4 adds replicated comparison and robustness summaries. These features are intentionally modest. They help researchers see whether a result is stable across deterministic seed replicates and how quickly a method degrades across a noise curve.

The approximate confidence intervals in this crate are convenience summaries, not publication-grade statistical guarantees. They use a simple normal approximation around the sample mean. For papers or hardware claims, export the raw reports and analyze them with a full statistics environment.

Alpha.4 still does not execute physical quantum hardware. The entanglement proxy remains a classical parity/coherence sketch. Any hardware observation must be marked as an external backend observation and accompanied by backend metadata, calibration context, and execution receipts.

## Alpha.5 notes: provenance and reporting

Alpha.5 adds local reproducibility metadata, CSV/Markdown export helpers, and conservative audit helpers. These additions are meant to make the crate easier to use in research notes without increasing the strength of the scientific claims.

Important distinction:

- `RunEnvironment` and `ReproducibilityRecord` are lab-note conveniences.
- They are not cryptographic receipts.
- Mycelix or a real digest/signature layer should be used for artifact commitments.

Alpha.5 also adds `audit_binding_probe`, `audit_negative_control`, and `audit_robustness`. These functions are local guardrails only. They are not peer review and do not make a result publishable.

## First independent run and a real finding (2026-07-24)

This crate reached alpha.10 (10 releases, ~8,300 lines) without any recorded run
of its own core probes — every changelog entry documents added scaffolding
(docs, CLI commands, release checklists), never an actual measurement. It also
has zero consumers anywhere else in this workspace. This section is the first
record of what the probes actually report.

Commands run (`cargo run -p symthaea-quantum-comp --example <name>`):

```
comparative_report (replicates=8, dimension=1024, trials=16, noise=0.05):
  classical_noisy_mean = 0.950363   phase_noisy_mean = 0.998450   correlation_noisy_mean = 0.949257
  classical_minus_phase_noisy_dz = -33.364006

negative_control (dimension=512, trials=12):
  matched_key = 0.959473   wrong_key = 0.501465   random_item = 0.499512   control_gap = 0.458008

entanglement_proxy (dimension=512, trials=12, decoherence=0.05):
  mean_recovery = 0.963053   wrong_key = 0.502767   recovery_gap = 0.460286

noise_sweep (classical_noisy vs phase_noisy across noise 0.00-0.25):
  0.05: 0.948568 vs 0.998473   0.10: 0.895671 vs 0.993917   0.25: 0.753581 vs 0.963015
```

**The negative controls are sound.** Matched-key recovery sits near 0.96 and
wrong-key/random-item similarity sits at ~0.50 (chance) in both probes, with a
real, stable gap. The binding/unbinding implementation is not falsely
reporting recovery.

**The comparative-binding headline number is a noise-model calibration
artifact, not a substrate finding.** `classical_minus_phase_noisy_dz = -33.36`
is an enormous effect size — the kind of number that should trigger suspicion,
not excitement (cf. the project's general "suspiciously tight/extreme
statistic is a red flag" heuristic). Root cause, traced directly in
`classical_hdc.rs` and `phase_hdc.rs`:

- `BinaryHypervector::with_bitflip_noise(p)` flips each bit independently with
  probability `p`. `similarity() = 1 - hamming_distance/dim`, so expected
  similarity degrades *linearly*: `≈ 1 - p`. Matches the observed
  classical numbers exactly (p=0.05 → 0.9486).
- `PhaseHypervector::with_phase_noise(sigma)` perturbs each dimension's angle
  by noise scaled by `sigma = noise * π` (≈9° at `noise=0.05`).
  `circular_similarity() = (mean(cos(Δphase)) + 1) / 2`, and `cos(small θ) ≈
  1 - θ²/2` — a *quadratic*, much gentler falloff for small angles. That's why
  phase similarity barely moves (0.9985) at the identical nominal `noise`
  value.

The same raw `noise` parameter is fed into two structurally different
perturbation models — a discrete per-bit flip probability vs. a continuous
per-dimension angular jitter — with no calibration establishing that they
represent comparable perturbation magnitudes. Nothing in the code (not even
`audit_binding_probe`) checks for this. Anyone reading `comparative_report`'s
output at face value would conclude phase/"quantum-inspired" encoding is
dramatically more noise-robust than classical HDC; that conclusion does not
survive tracing the actual noise-injection code.

**Implication for future work on this crate**: before any claim of the form
"representation X degrades more gracefully than representation Y under
noise," the two noise models need a principled equivalence (e.g. matched
expected bit-error-rate, or matched mutual-information loss per dimension) —
not a shared literal parameter value. Until the section below, this had not
been done in any alpha release. The `classical_minus_phase_noisy_dz` and
`noise_sweep` comparative columns should still be read as "these two toy
noise processes behave differently at the same literal parameter value," not
as evidence about phase encoding's intrinsic robustness — use
`calibrated_comparison` (below) for any cross-representation noise claim.

## Calibrated cross-representation comparison, and a second bug it found (2026-07-24)

`calibrated_comparison.rs` fixes the artifact above by calibrating both
channels to a shared, representation-neutral unit — bit-error-rate (BER)
under each representation's own natural hard-decision rule — and scoring
both arms with the *same* final metric (`BinaryHypervector::similarity`
between the original item bits and the noisy-recovered result, hard-decoding
phase-HDC's result first). See the module's own doc comment for the full
design and `examples/calibrated_comparison.rs` to reproduce.

**Building the calibration exposed a second, independent, pre-existing bug**:
`PhaseHypervector::to_binary_halfplane` decoded `bit = phase >= π`, but
`from_binary` encodes bit `0` at phase `0` and bit `1` at phase `π` — the
decode boundary sat exactly *on top of* both encoded symbols instead of at
their angular midpoints (`π/2`, `3π/2`). Any infinitesimal noise in the
"wrong" direction flipped the bit regardless of magnitude, so measured
phase-channel BER jumped to ~0.5 for almost any nonzero `sigma`. This
function had zero other consumers anywhere in the crate and zero test
coverage of its own before this — nothing had ever exercised it. Fixed to the
correct nearest-symbol rule (`cos(phase) < 0`, the sign of the projection
onto the symbol axis) in `phase_hdc.rs`, with a regression test
(`to_binary_halfplane_tolerates_small_noise_near_either_symbol`) confirming
small noise near either symbol no longer flips the bit.

**Real calibrated result** (`cargo run -p symthaea-quantum-comp --example
calibrated_comparison`, dimension=1024, trials=16 per point, target BER
0.01–0.40):

```
target_ber  classical_recovery  phase_recovery  mean_delta  sign_test_p
0.01        0.989868            0.989624        +0.000244   1.000
0.05        0.949890            0.952087        -0.002197   0.454
0.10        0.900146            0.898499        +0.001648   0.454
0.20        0.801758            0.797424         0.004333   0.804
0.30        0.703064            0.693359         0.009705   0.454
0.40        0.605103            0.593994         0.011108   0.454
```

Negative controls stay honest across every point: wrong-key recovery is
~0.502 for both channels regardless of target BER. Calibration is accurate
(`measured_classical_ber`/`measured_phase_ber` both within ~0.002 of
`target_ber` at every point).

**Honest reading**: at matched noise strength, classical XOR-HDC and
phase-HDC recovery accuracy are statistically indistinguishable — the
opposite conclusion from the uncalibrated `dz = -33.36`. There is a small,
consistent trend of classical edging ahead as BER grows (mean delta rises
from ~0 to +0.011 by BER=0.40), but it never reaches significance at n=16
(sign-test p ≥ 0.45 throughout) and should be read as inconclusive, not as a
finding — a larger `trials` value (the config exposes it directly) would be
needed to resolve whether that trend is real.

## Calibrated capacity comparison (2026-07-24)

Noise-robustness (above) is not the theoretically strongest claim for a
phase/holographic representation — **capacity**, how many items can be
superposed into one bundle before reliable retrieval fails, is the more
distinctive claim in the Fourier holographic reduced representation
literature. `capacity_comparison.rs` tests that instead, with the same
fairness discipline as `calibrated_comparison.rs`: both representations
bundle the *same* underlying random bit patterns (phase items are
`PhaseHypervector::from_binary` of the classical items, not independently
random), and both are scored by the same metric — two-alternative
forced-choice accuracy (does the representation's own native similarity
correctly rank a true bundled member above a never-bundled foil?).

This needed a new primitive: `PhaseHypervector` had no bundling/superposition
operation. Added `circular_bundle` (per-dimension circular mean — the
continuous analog of `BinaryHypervector::majority_bundle`'s per-dimension
majority vote), with its own tests (identical-copies round-trip, dimension
mismatch rejection, member-beats-foil sanity).

**Shipped default** (`cargo run -p symthaea-quantum-comp --example
capacity_comparison`, dimension=1024, trials=32/point):

```
bundle_size  classical_accuracy  phase_accuracy
1..32        1.000000            1.000000
64           0.968750            1.000000
128          0.906250            0.937500
256          0.875000            0.843750
```

Both representations are still comfortably above chance at 256 members in a
1024-D space — 2-AFC recall is a much easier task than exact recovery, so
capacity in this sense is high relative to dimension, consistent with known
binary-HDC bundling theory. This default range doesn't reach a real
degradation crossing, so it was extended for verification:

**Extended range** (ad hoc `--release` run, not the shipped default — used to
find where a real crossing exists before writing any test around one; see
`feedback_physics_plan_review_fpu` — "verify via real run, don't guess a
threshold"):

```
dimension=1024, trials=24:
bundle_size  classical  phase
256          0.917      0.875
512          0.750      0.667
1024         0.625      0.667
2048         0.542      0.625
4096         0.583      0.583
8192         0.708      0.708
16384        0.417      0.417

dimension=512, trials=32:
bundle_size  classical  phase
1            1.000      1.000
512          0.656      0.656
2048         0.562      0.562
```

**Honest reading**: capacity degrades toward chance (0.5) as bundle size
grows well past the dimension, as expected, and — like the noise-robustness
result above — classical and phase-HDC track each other closely throughout,
with no representation showing a consistent, meaningful edge. Values near
chance level are noisy at these trial counts (e.g. the dimension=1024,
trials=24 table's non-monotonic bump at bundle_size=8192) and should not be
over-read point-by-point; the shape of the curve (steady decline from ~1.0
toward ~0.5 as bundle size crosses the dimension) is the real signal, not any
single value. **Second real finding, on the same property as the first**:
neither the noise-robustness test nor the capacity test found a
representational advantage for phase/quantum-inspired encoding over
classical binary HDC in this crate's simulation. Treat this as reasonably
strong local evidence against the hypothesis that phase encoding is
intrinsically more capable, at least for the two properties tested here —
not as a general verdict on quantum-inspired HDC, which this crate's
`CLAIM_BOUNDARIES.md` already scopes tightly. See below for a third property
where the picture changes.

## Calibrated continuous-value comparison — a real, positive result (2026-07-24)

Both properties above forced phase-HDC into a purely binary role: encoding
only two symbol points (0, π) and hard-decoding back to bits before scoring.
That discards the one real structural difference phase-HDC has — a
*continuous* degree of freedom per dimension, which binary HDC does not have
at all. `continuous_value_comparison.rs` tests that directly: storing and
recovering a scalar `x ∈ [0, 1]`, not a bit.

**Fair encodings** (see the module doc comment for full reasoning): classical
uses `BinaryHypervector::thermometer_encode` (unary/thermometer coding — the
graceful-degradation-by-construction baseline for continuous values in
binary vectors, not a strawman). Phase maps `x` onto a *half*-circle,
`θ = π/2 + x·π`, so `x=0` and `x=1` land maximally far apart (antipodal, π
radians) — the phase-space analog of thermometer coding's `x=0`/`x=1` being
maximally different bit patterns. Both representations spend their *entire*
dimension as redundancy for one scalar and are scored by the same metric
(mean absolute error in `x`-units, `[0, 1]`, via a wraparound-safe circular
distance for the phase arm).

**Real result** (`cargo run -p symthaea-quantum-comp --example
continuous_value_comparison`, dimension=1024, trials=32/point):

```
target_ber  classical_mae  phase_mae  paired_p
0.00        0.000217       0.000001   4.7e-10
0.01        0.005768       0.005956   0.860 (not significant)
0.05        0.024270       0.007390   0.0021
0.10        0.047998       0.008736   0.0001
0.20        0.091706       0.012012   0.0021
0.30        0.135993       0.017724   0.0005
0.40        0.182563       0.032699   1.9e-5
```

Phase-HDC is dramatically more precise — 3-200x lower error — at every
target BER except 0.01, where the difference is within noise. This is a
real, large, statistically significant effect, not another calibration
artifact: verified the calibration accuracy the same way as the other two
modules, and the underlying mechanism was independently confirmed by direct
probe, not just inferred from the headline numbers.

**The mechanism (verified, not assumed)**: thermometer decode
(`popcount / dimension`) has a genuine *systematic bias toward 0.5* under
bit-flip noise, proportional to `p·(1 - 2x)`. Probed directly at `x=0.9`,
`p=0.1`, 2000 trials: mean *signed* error was `-0.0798`, essentially the
entire magnitude of the mean *absolute* error (`0.0798`) — confirming the
error is almost pure bias, not random noise, and matching the closed-form
prediction `0.1·(1-1.8) = -0.08` almost exactly. This happens because more
`1`-bits than `0`-bits are available to flip when `x` is far from 0.5 (or
vice versa), so bit-flip noise pulls the popcount estimate toward the
middle. `circular_mean` has no equivalent bias — phase noise is symmetric
around the true angle, so it averages toward the true value, not toward any
fixed point.

**Honest caveats, not just a headline claim**: (1) At `target_ber=0`, the
~200x gap is a *different*, smaller-in-kind effect — classical's inherent
quantization floor (`~1/(2·dimension)`, since popcount only has
`dimension + 1` distinct levels) versus phase's `f32`-precision-bounded
continuous decode. (2) The bias mechanism above is a property of *thermometer
decode specifically* (plain popcount), not a proof that no classical
encoding could do better. Since the bias has a known closed form
(`p·(1-2x)`), a decoder that knows the channel's noise rate could in
principle apply a linear debiasing correction (`x_corrected = (x_hat - p) /
(1 - 2p)`) and close some — untested how much — of the gap. That's real,
open, not-yet-tested follow-up work, not a rebuttal of the result as
measured. As measured, with the fairest *naive* encoding this crate has
available on each side, phase encoding wins clearly and reproducibly on this
property — the first positive result in this crate's history, after two
negative results on other properties. See below, though, for what happened
once the classical decoder was allowed to know its own channel noise rate.

## Debiasing correction — a real crossover, not a clean win for either side (2026-07-24)

The result above compared phase-HDC against classical's *naive*
`thermometer_decode`. Since that decode's bias has a known closed form
(`E[x_hat] = x·(1-2p) + p`), a decoder that knows the channel's
bit-error-rate can invert it exactly:
`x_corrected = (x_hat - p) / (1 - 2p)` — implemented as
`debias_thermometer_estimate`, clamped to `[0, 1]` since the linear
correction can otherwise land outside the valid range under an unlucky noise
draw. This is a legitimate assumption, not cheating: this module's own setup
already calibrates to a known target BER, so a decoder "knowing `p`" is the
same information the phase arm's calibration already uses.

**Real result** (dimension=1024, seed=default; low-BER points at
trials=32/point, high-BER points re-run at trials=200/point once the
trend needed more power to resolve — see below):

```
target_ber  classical_mae  debiased_mae  phase_mae   debiased_vs_phase_p   winner
0.00        0.000217       0.000217      0.000001    4.7e-10                phase (unaffected by debiasing)
0.01        0.005768       0.002452      0.005956    0.020                  debiased classical
0.05        0.024270       0.004113      0.007390    0.020                  debiased classical
0.10        0.050857       0.009868      0.008836    0.104 (n=200)          not resolved
0.20        0.100391       0.016042      0.012423    0.0023 (n=200)         phase
0.30        0.150746       0.028634      0.018901    2.7e-5 (n=200)         phase
0.40        0.199789       0.057200      0.035531    5.0e-5 (n=200)         phase
```

Debiasing does not just shrink the gap — it produces a genuine **crossover**
around `target_ber ≈ 0.10`. Below that, debiased classical significantly
*beats* phase (bias-removal dominates: the correction fixes a real
systematic error at low cost). At and above that point, phase wins by a
growing, statistically solid margin (the correction's variance amplification
— dividing by `1 - 2p`, shrinking toward zero as `p → 0.5` — starts to
dominate: removing bias makes any residual random error in the raw estimate
`1/(1-2p)` times larger). The `target_ber=0.10` point itself stayed
unresolved even at n=200 (p=0.10) — a genuine near-tie at the crossover, not
an artifact of insufficient power elsewhere in the table.

**Honest bottom line, now with three data points instead of one**: this
crate's overall finding on continuous-value storage is not "phase wins" —
it's "which representation wins depends on the noise regime and on whether
the classical decoder is allowed to know its own channel." The zero-noise
quantization-floor gap (phase's real, structural, undebiasable advantage)
and the high-noise regime both favor phase; the low-noise regime favors
classical once it's allowed the same kind of calibration information the
phase arm already uses. Do not cite "phase-HDC has 3-200x lower error" from
the section above without this qualification — it was true against the
naive decoder only, and this crate's own comparative discipline requires
comparing each representation at its best, not its naive, attempt.

Not done: whether a different (non-linear, or bias-variance-optimal rather
than fully-bias-corrected) classical estimator could close the high-noise
gap too — the debiasing correction tested here removes bias exactly but
makes no attempt to trade off the resulting variance amplification, so it is
not necessarily classical's *best* possible decoder, only a legitimate and
straightforward one.

## The shrinkage probe: tested, and it does not close the gap (2026-07-24)

Tested the open question above directly: `ShrinkageProbeRunner` blends the
raw and fully-debiased estimates by a factor `lambda ∈ [0, 1]`
(`lambda=0` = raw, `lambda=1` = full debiasing) and sweeps `lambda` at a
fixed target BER, comparing every point against phase-HDC with the same
paired-trial significance testing used throughout this crate.

**Real result** (dimension=1024, seed=default):

```
target_ber=0.20 (n=200): MAE strictly decreases from lambda=0 to lambda=1.
                          best_lambda=1.0 (full debiasing is already optimal)
target_ber=0.30 (n=200): same pattern. best_lambda=1.0
target_ber=0.40 (n=400): best_lambda=0.9 (mae=0.058045) vs lambda=1.0's
                          0.059068 -- a tiny, real improvement. Phase still
                          wins decisively at every lambda (p down to 1e-62).
target_ber=0.45 (n=400): best_lambda=0.8 (mae=0.107611) vs lambda=1.0's
                          0.115060 -- a real, meaningful improvement this
                          time. Phase still wins at the optimum (p=7.4e-5).
target_ber=0.48 (n=400): best_lambda=0.5 (mae=0.191540). Every single lambda
                          from 0.0 to 1.0 has a *negative* mean delta
                          (classical numerically lower than phase) -- but
                          the paired sign test does not reach significance
                          at any lambda (p ranges 0.12-0.96). A consistent
                          but statistically unresolved signal, not a
                          confirmed finding.
```

**Honest reading**: in the noise range that actually mattered for the
crossover found earlier (`target_ber` 0.10-0.40), full debiasing
(`lambda=1`, already implemented and tested) turns out to already be at or
extremely close to MAE-optimal within this linear-blend family — shrinkage
does not meaningfully close phase's advantage there. Only much closer to the
noise ceiling (`target_ber` ≳ 0.45, where both representations' errors are
already large — phase MAE 0.078-0.24 in this range, more than a fifth of the
whole `[0,1]` output range) does interior shrinkage produce a real,
significant improvement over full debiasing (confirmed at `target_ber=0.45`,
`p=7.4e-5` for the delta between `lambda=0.8` and `lambda=1.0`'s errors
against phase) — but even at its best, classical still loses to phase there.
The one place classical's numbers looked genuinely competitive
(`target_ber=0.48`, deep in the near-chance regime) did not survive proper
significance testing at `n=400` and is disclosed as an open, low-priority
question rather than a finding — that operating point is also of limited
practical interest, since neither representation preserves much signal
there.

**Bottom line across all four probes now**: the "smarter classical decoder
might close the gap" hypothesis was worth testing (it did work, dramatically,
against the *naive* decoder in the debiasing probe) but does not extend
further via this shrinkage family — the continuous-value crossover found
earlier (phase wins at zero noise and `target_ber` ≳ 0.10, debiased
classical wins below that) is robust to this refinement, not narrowed by it.
This line of inquiry is a reasonable place to stop without chasing further
diminishing-returns refinements to the classical decoder.
