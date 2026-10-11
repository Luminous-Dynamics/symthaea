# PIE-009V — Metrology Intercomparison, Drift Detection, and Conservative Quarantine

## Purpose

This independent reference freezes decision semantics for comparing multiple metrology references without pretending the system knows absolute truth.

A pairwise comparison can establish that two qualified references disagree. It cannot, by itself, establish which reference is wrong.

## Core rule

A single reference may be isolated only as a **relative outlier** when:

1. two qualified peer references agree with each other;
2. the suspect independently disagrees with both peers; and
3. the three comparison channels used for that witness do not share any declared comparison failure group.

Otherwise the disagreement remains `AMBIGUOUS_CONFLICT`.

`RELATIVE_OUTLIER_ISOLATED` is not proof that the isolated reference is physically wrong. It only proves inconsistency relative to an independently agreeing cohort under the declared comparison model.

## Statuses

- `NO_DETECTED_DISAGREEMENT` — at least one qualified agreement exists and no qualified disagreement is present.
- `INSUFFICIENT_EVIDENCE` — no qualified agreement/disagreement is available.
- `RELATIVE_OUTLIER_ISOLATED` — exactly one reference has a complete independent isolation witness.
- `AMBIGUOUS_CONFLICT` — qualified disagreement exists but no unique independent isolation witness exists.

## Conservative quarantine

For an isolated relative outlier, only that reference is a quarantine candidate.

For an ambiguous conflict, every reference directly implicated by a qualified disagreement becomes a quarantine candidate until additional independent evidence resolves the conflict.

## Independent execution

The final candidate oracle was executed locally with Python 3 on 2026-09-13 and returned:

`ok`

The fixture covers:

- mutual agreement;
- one-pair ambiguity;
- two-peer relative-outlier isolation;
- shared-comparison common mode preventing isolation;
- unqualified evidence preventing isolation;
- all-pairs disagreement remaining ambiguous;
- exact comparison-bound equality;
- deterministic metadata-sensitive evidence digests;
- duplicate/reversed duplicate pair rejection.

## Boundaries

This reference does not estimate failure probabilities, prove which reference is physically correct, model calibration physics, detect cyber compromise, replace PIE-009T traceability precision, or authorize hardware use.

It consumes already-qualified comparison evidence and declared comparison failure groups. Evidence qualification itself belongs to the existing PIE sensing/metrology layers.
