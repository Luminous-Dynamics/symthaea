# SPECTRA-CARD-001 synthetic design/evaluator

This crate builds task-specific synthetic campaigns for the modernized
face-down red/black playing-card protocol in issue #10070.

## Separate estimands

The design distinguishes:
- enclosure-context effect: InsideUnenergized minus OutsideReference;
- electrical-state increment: InsideActive minus InsideSham.

It does not pool these contrasts. The first is a context/enclosure contrast;
the second is an active-versus-sham contrast within the inside context.

## Four abstract conditions

- OutsideReference: outside the enclosure.
- InsideUnenergized: inside with no added apparatus state.
- InsideActive: inside with an abstract active apparatus state.
- InsideSham: inside with a matched abstract sham state.

Active and sham are evaluator-only labels. The public trial contains an opaque
randomized condition code and the visible inside/outside context only. This is
a synthetic software condition, not an instruction for a real field source.

## Synthetic worlds

- Null: all response conditions have 500-permille success probability.
- EnclosureContextOnly: inside conditions have higher response probability;
  active and sham do not differ.
- ElectricalIncrementOnly: active differs from sham; unenergized inside and
  outside do not differ.
- BothEffects: both preregistered contrasts are positive.
- PracticeOnly: outcome probability increases by period, but condition order is
  randomized within every participant/cycle.
- OrderConfounded: a deliberately fixed condition order makes a practice trend
  mimic a condition effect. The evaluator flags the schedule as confounded.

Each participant/cycle receives all four conditions exactly once, in randomized
order except for the deliberate order-confound world. The evaluator rejects
missing/duplicate conditions, repeated period positions, invalid session/cycle
identity and inside/outside context labels inconsistent with the blinded code.
The four card targets are balanced two red/two black within each cycle and then
shuffled.

## Candidate/evaluator separation

The public campaign contains participant response, opaque condition code,
inside/outside context, period and session identity. It contains no target
labels, hidden active/sham mapping, seed or synthetic-world label. Evaluator
truth is a separate sidecar.

## Scoring and inference boundary

The evaluator reports:
- participant-weighted condition accuracy;
- participant-level B−A enclosure-context contrasts;
- participant-level active−sham electrical-increment contrasts;
- mean period by condition and a schedule-order confounding diagnostic.

This crate emits descriptive contrasts only. It does not calculate p-values,
confidence intervals, randomization inference, equivalence tests or confirmatory
claims. Bind the participant-level effects to the exact blocked-randomization
inference owner (#9670 / PR #9673) only after the allocation and analysis
manifest are frozen.


The raw four-outcome counts are preserved per condition so the shared
`symthaea-statistics` signal-detection implementation in PR #10086 can compute
d-prime/criterion and ROC/PR summaries. This crate does not compute d-prime,
AUC, p-values or confidence intervals itself. The signal-detection equal-
variance Gaussian assumption and loglinear correction are owned by that shared
statistics API; raw counts remain the primary auditable data.

## Run

    cargo test --manifest-path tools/spectra-card-001-simulator/Cargo.toml

## Claim ceiling

This is a synthetic design/evaluator only. It does not recover Puharich's
historical trial-level data, directly replicate the 1969 experiment, authorize
a human electrical intervention, specify an EM source, or establish ESP or a
new physical mechanism. Any physical active/sham study requires separate
instrumentation, electrical-safety and ethics review.

## Frozen known-answer campaigns

The six deterministic synthetic worlds and descriptive outputs are frozen in
`research/spectra-card-001/fixtures/known-answer-v0.1.json`. The Rust test
replays the exact participant-level contrasts, condition means and schedule
diagnostic for all six worlds. These are fixture outputs for deterministic
code replay, not human effect-size expectations or statistical power results.
