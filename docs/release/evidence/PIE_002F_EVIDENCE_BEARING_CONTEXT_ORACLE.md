# PIE-002F Evidence-Bearing Utility Context Oracle

Status: independent reference semantics; synthetic fixtures only.

## Purpose

Freeze the provenance boundary above PIE-002D's explicit numerical recovery/storage/supply context.

The theorem is deliberately narrow:

```text
all nine required external numerical facts
+ valid evidence references on every fact
-> evidence-bearing supply/recovery context
```

Evidence-bearing does not mean true, fresh, independent, applicable to the current process/site/configuration, calibrated, feasible, dispatchable, economical, or authorized.

## Why this tranche exists

PIE-002D correctly removed numerical defaults, but its context still accepts bare ranges. A caller can therefore provide a well-formed capacity, power, recovery, or efficiency range without recording where that value came from.

PIE-002F makes that provenance omission impossible at the stronger boundary while preserving PIE-002D as the lower numerical binder.

## Required facts

Every evidence-bearing context contains all nine PIE-002D external facts:

1. recoverable electrical energy;
2. recovery-window duration;
3. storage energy acceptance;
4. storage charge-power capability;
5. storage discharge-power capability;
6. recovery delivery / round-trip fraction;
7. available electrical-energy capacity;
8. available sustained-power capacity;
9. available peak-power capacity.

There are no implicit zero, infinity, nominal-capacity, or guessed values.

## Evidence semantics

The oracle reuses the existing PIE evidence classes exactly:

- Hypothesis;
- LiteratureModel;
- VendorProjection;
- LabMeasured;
- RelevantEnvironmentMeasured;
- IntegratedDemonstration;
- Qualified.

It does not assign a scalar score or ranking to those classes.

Each external fact must carry at least one `EvidenceRef`. Non-hypothesis evidence requires a nonblank source, matching the existing PIE evidence rule. Hypothesis evidence may retain an empty source and remains visibly hypothetical.

Duplicate evidence IDs within one fact fail closed. The same evidence record may legitimately appear on multiple different facts; that reuse is preserved exactly and is **not** interpreted as independent corroboration.

Evidence order is preserved by this theorem. PIE-002F does not define evidence-list order as an identity commitment; a later canonical receipt may normalize it under a separately frozen profile.

## Stronger lexical admission

Following PIE-ID-001 (#2870), the F-layer admission boundary is stricter than legacy `require_text` for durable `evidence_id` spelling:

- nonempty;
- exact equality with `trim()`;
- no ASCII control characters;
- maximum 4,096 UTF-8 bytes.

The value is rejected rather than silently normalized.

This does not globally change the lower ontology's string semantics.

## Numerical semantics

Underlying numerical validation remains conservative and independent of evidence metadata:

- ranges must be finite, nonnegative, and ordered;
- recovery duration must have a strictly positive lower bound;
- delivery fraction must satisfy `0 <= min <= max <= 1`.

Adding evidence cannot rescue an invalid number. A `Qualified` evidence class attached to an invalid range still fails.

## Lossless lower-context projection

A validated evidence-bearing context can be stripped to the exact bare PIE-002D numerical context. The stripping operation removes only evidence metadata; every numerical range is preserved exactly.

The oracle additionally emits an evidence-bearing receipt structure that retains the evidence tuple for each named fact alongside the bare context. This demonstrates the intended production boundary:

```text
EvidenceBearingSupplyRecoveryContext
    -> validate
    -> exact PIE-002D numerical context
       + exact per-field evidence lineage
```

No field may borrow evidence from another field merely because units or values match.

## Boundedness

V1 freezes:

- maximum 64 evidence references per external fact;
- maximum 4,096 UTF-8 bytes for evidence ID, source, or note strings.

Budget exhaustion fails before an evidence-bearing context is admitted.

These are F-layer admission bounds, not claims about every lower PIE API.

## Adversarial fixtures

The self-test proves at minimum:

1. all nine facts with valid evidence admit successfully;
2. the bare PIE-002D numerical context is preserved exactly;
3. evidence records/classes/notes are preserved exactly;
4. missing evidence on one required fact fails;
5. measured/non-hypothesis evidence without a source fails;
6. duplicate evidence IDs within one fact fail even if the records differ;
7. the same evidence reused across multiple fields remains legal and visibly shared;
8. reused evidence is not transformed into an independence claim;
9. hypothesis evidence remains Hypothesis;
10. zero-inclusive recovery duration fails;
11. invalid fraction fails;
12. non-canonical durable evidence IDs fail;
13. per-fact evidence-budget exhaustion fails.

## Sequencing with PIE-002G

PIE-002F should precede PIE-002G in the preferred authoritative path.

```text
ProcessDefinition
    -> PIE-002E fresh in-call projection/composition
    -> PIE-002F evidence-bearing external context
    -> PIE-002D numerical binding
    -> PIE-002G opaque subject-bound witness
```

An opaque witness over provenance-free external values would prevent type forgery but would not solve the larger epistemic gap. G should therefore make the evidence-bearing composition result proof-carrying rather than treating bare numeric context as the preferred endpoint.

## Non-claims

PIE-002F does not establish source truth, source independence, freshness, scenario applicability, evidence conflict resolution, calibration validity, process-content currentness, utility feasibility, storage dispatch, thermodynamic closure, equipment qualification, economics, or execution authority.

Tracks #2785, #2764, #2782, #2826, #2870, #2867, #1610, #1647, and master #1604.
