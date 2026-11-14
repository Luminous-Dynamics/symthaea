# MEL-MR-EVAL-037 — Musical edit validity envelopes and multi-reference qualification

**Date:** 14 November 2026  
**Status:** Initial contract implementation; exact-head qualification pending  
**Tracking:** [Issue #12044](https://github.com/Luminous-Dynamics/symthaea/issues/12044)  
**Parent research:** [SMEBench / EVAL-030](https://github.com/Luminous-Dynamics/symthaea/pull/9284), [EVAL-021 / #9184](https://github.com/Luminous-Dynamics/symthaea/issues/9184)

## Why the field needs this

A paired target is valuable evidence, but it is one realization—not automatically the unique correct answer. Musical edits frequently admit multiple valid realizations because timbre, articulation, voicing, dynamic nuance, renderer behavior and performance can vary while satisfying the same musical obligation.

SMEBench itself demonstrates why this distinction matters. It contains 11,160 paired examples, with a balanced 3,600-example test set (300 per operation) and a separate 416-pair real-audio complement. Its human target validation collected 236 ratings from 17 participants: mean target quality was 4.23/5 and source alignment 4.65/5, but instruction alignment was 3.99/5. Legato, Remove Instrument and Swap Drums did not reliably exceed the midpoint on instruction alignment. The paper also warns about predominantly Western-centric music/instrument coverage and correlated caption-generation/filtering model families. Source: https://arxiv.org/abs/2610.36715 (Sections 3–4 and Appendix E).

The implication is not that paired targets are useless. They are important diagnostic references. The implication is that exact waveform similarity to one target must not be confused with complete musical correctness.

## Paradigm shift

The envelope separates four objects:

1. **Requested obligation:** which operation, target, scope and musical property must change.
2. **Protected invariants:** which properties and events must remain intact outside authorized edit support.
3. **Permitted variation:** which dimensions are deliberately allowed to vary, with explicit limits and evidence.
4. **Reference adequacy:** whether each paired target or alternative is valid, ambiguous, inaudible, invalid or unreviewed.

A candidate edit should ultimately pass hard obligations and protected invariants; permitted variation defines a bounded set, not a license for arbitrary changes. Reference similarity is a diagnostic measure rather than a substitute for those conditions.

## Implemented v1 contract

- Typed envelope identity, immutable envelope/dataset versions, benchmark track, sample, source lineage, split, operation, target scope and selector commitment.
- Separate typed arrays for requested changes, protected invariants, permitted variations and forbidden changes.
- Constraint relations for equality, difference, ranges, member/not-member sets, presence/absence, no-change and monotonic contours.
- Canonical ordering of constraint identifiers, allowed values and references.
- Multiple references with roles: valid alternative, demonstration and negative control.
- Explicit reference adequacy: Validated, Ambiguous, Inaudible, Invalid or Unreviewed.
- Valid alternatives require a review record commitment, reviewer-set commitment, at least two reviewers and an independence attestation. These commitments/attestations are provenance fields; they do not cryptographically authenticate reviewers or prove legal/ethical approval.
- Reference lineage must equal the sample's source lineage.
- Constraint-only policy allows evaluation without any paired target; the alternative-required policy fails closed without a reviewed valid alternative.
- Domain-separated SHA-256 envelope commitment.
- Twelve authored Rust tests cover multiple alternatives, constraint-only operation, missing alternatives, unreviewed alternatives, lineage substitution, negative-control promotion, duplicate obligations, invalid ranges, digest sensitivity, mutable versions, canonical ordering and no-op obligations.

## What this is not

The first slice validates the contract, not a rendered candidate. It does not decide whether an edit sounds good, test whether a change is perceptually audible, verify an operation-specific symbolic transformation, adjudicate rights, or replace EVAL-021's blinded human metric calibration. The schema fixture contains metadata and synthetic digests only; it contains no audio and is not a benchmark result.

## Evaluation design enabled by the contract

Report the following independently:
- operation-obligation satisfaction;
- protected-invariant violations and collateral-change magnitude;
- permitted-variation compliance;
- reference adequacy and ambiguity rates;
- edit audibility, plausibility and artifact audibility;
- false acceptance on plausible-but-wrong and no-op controls;
- abstention coverage/risk;
- worst-group behavior across renderer, instrument family, density, target salience, source type and musical tradition.

Use source-lineage-clustered intervals and preserve distinct synthetic and real-audio strata. Human judgments should be blinded, randomized and analyzed with listener/item clustering. Do not average an unsupported operation into a favorable aggregate.

## Prospective targets (not results)

- Zero false acceptance on deterministic hard-constraint/protected-content mutation controls.
- At least 95% requested-obligation success on supported held-out controlled edits, reported per operation.
- At least 90% acceptance of adjudicated valid alternatives and no more than 1% acceptance of adjudicated invalid controls on a frozen qualification set.
- Held-out renderer and instrument-family performance reported separately.
- Keep all thresholds prospective until pilot variance, sample size and confidence intervals are established. These thresholds are stretch targets, not current measurements.

## Relationship to the existing programme

- #12007 / EVAL-035: actual benchmark byte verification and lineage-bound receipts.
- #12029 / RIGHTS-001 and #12035 / INTEGRATION-001: purpose-specific rights and joint admission.
- #9128 / EVAL-017, #9133 / EVAL-018, #9151 / EVAL-019, #9155 / EVAL-020: event influence, interactions, probe selection and factorial truth.
- #9185 / EVAL-022: minimal locality-constrained repair.
- #9184 / EVAL-021: blinded human calibration of evaluator metrics on unseen renderers.
- #10088 / RMI-Bench: reciprocal musician–Melothaea interaction and authorship.

## Validation boundary

The production module, standalone harness mirror, schema, synthetic fixture and fixture-check example are committed in this tranche. Formatting, tests, fixture execution and Clippy remain pending exact-head Rust 1.96 execution. No external dataset was fetched, no human panel was run, and no musical quality or SOTA claim is made.
