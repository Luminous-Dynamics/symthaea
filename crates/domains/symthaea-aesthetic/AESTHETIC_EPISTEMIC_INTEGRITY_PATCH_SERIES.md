# Symthaea Aesthetic Epistemic Integrity Patch Series

This series continues from the operational-resilience snapshot and deliberately
returns to the epistemic core of aesthetic evaluation. It does not add another
universal beauty formula. It adds evidence for when preferences disagree, when
an assessment falls outside its calibration domain, when optimization begins to
game its proxy, and when multiple candidates contain irreducible trade-offs.

## Bundle A — Pluralistic preference

1. Preserve independently grounded individual, cohort, community, expert, and
   public-sample perspectives.
2. Apply population, consent, confidence, and grounding weights explicitly.
3. Measure effective perspective count, concentration, weighted disagreement,
   grounded evidence, median, range, and optional aggregate preference.
4. Preserve polarized or productive disagreement instead of silently emitting a
   population mean.
5. Run leave-one-perspective-out representation audits to reveal aggregate
   sensitivity and minority-erasure risk.

Primary modules:

- `pluralism.rs`
- `representation.rs`

## Bundle B — Epistemic authority and abstention

1. Fit versioned reference domains from replay-valid assessments.
2. Preserve modality and creative-mode scope.
3. Track online moments and observed ranges for ten assessment features.
4. Measure standardized multichannel distance and feature-range extrapolation.
5. Decompose measurement, context, novelty, preference, distribution-shift, and
   pluralistic uncertainty.
6. Grant `Automatic`, require `HumanReview`, or `Abstain` under an explicit
   policy.

Primary module:

- `epistemic.rs`

## Bundle C — Anti-Goodhart and Pareto decisions

1. Retain replayable optimization checkpoints and independent holdout signals.
2. Detect proxy/target divergence, confidence erosion, preference-grounding
   erosion, saturation pressure, boundary exploitation, and novelty substitution.
3. Keep intrinsic, contextual, novelty, preference, confidence, pluralistic,
   epistemic, and Goodhart-safety objectives separate.
4. Compute the nondominated candidate frontier.
5. Treat missing required safety evidence as infeasible.
6. Preserve material trade-offs or select by an explicit maximin or weighted
   policy.

Primary modules:

- `optimization.rs`
- `goodhart.rs`
- `pareto.rs`

## Bundle D — Epistemic release closure

1. Bind every Pareto candidate to the exact epistemic assessment and final
   optimization checkpoint that produced its objectives.
2. Reject orphan, duplicated, substituted, or forged evidence.
3. Require configurable pluralistic coverage, representation audits, epistemic
   authority, Goodhart evidence, and Pareto closure.
4. Produce `Ready`, `HumanReview`, or `Blocked` outcomes from a complete,
   replayable finding set.
5. Advance the public API contract to `1.4.0` with explicit epistemic-integrity
   capabilities.
6. Expand the schema catalog from 57 to 65 persisted families.

Primary module:

- `epistemic_release.rs`

## Recommended integration sequence

```text
artifact evidence + context
        |
        v
AestheticAssessment
        |
        +--> pluralistic preference + omission audit
        |
        +--> reference-domain distance + uncertainty decomposition
        |         |
        |         +--> automatic / human review / abstain
        |
optimization checkpoints + independent outcomes
        |
        +--> Goodhart risk
        |
candidate objective vectors
        |
        +--> Pareto frontier + explicit selection policy
        |
        v
EpistemicReleaseBundle -> Ready / HumanReview / Blocked
```

## Compatibility and migration

- Existing `AestheticAssessment`, pipeline, archive, governance, and operational
  evidence remain unchanged.
- New integrations can require `ApiRequirement::epistemic_integrity()`.
- The API minor version advances from `1.3.0` to `1.4.0`; the major generation
  remains compatible.
- All new persisted structures begin at schema version 1.
- Stable FNV identifiers remain replay keys, not cryptographic signatures.

## Important limitations

- Reference-domain distance uses transparent global moments and observed ranges;
  it is not a learned density model.
- Weighted population evidence does not prove fairness or cultural legitimacy.
- Omission sensitivity is a diagnostic, not a complete theory of representation.
- Goodhart findings depend on genuinely independent holdout signals. Reusing the
  optimized proxy as a holdout defeats the purpose.
- Pareto selection prevents hidden scalarization only when downstream systems
  honor the preserved frontier and explicit policy.
- The crate must still be compiled, formatted, linted, and tested in the parent
  Rust workspace before merge.
