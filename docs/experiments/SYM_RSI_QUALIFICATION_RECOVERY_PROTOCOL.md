# SYM-RSI Qualification Recovery Protocol

**Status:** pre-measurement recovery contract. This document defines how qualification capabilities are reconstructed after process loss without silently rerunning one-shot fresh evidence.

## Purpose

SYM-RSI qualification tokens deliberately do not implement `Deserialize`. A serialized boolean or token-shaped object must never recreate authority by itself.

At the same time, process restarts must not force fresh scientific measurements to be rerun merely to regain an in-memory capability. The correct recovery mechanism depends on the epistemic class of the stage.

The recovery rule is therefore **heterogeneous by evidence type**.

## Recovery classes

| Stage | Evidence class | Recovery mechanism | Executes fixture environment during recovery? | External trusted anchor required? |
| --- | --- | --- | --- | --- |
| Parent C qualification | reproducible training + held-out replay evidence | recompute canonical training selection and 101-104 holdout qualification | no new fresh measurement | no separate fresh anchor |
| D verification qualification | frozen 301-304 replay corpus | rerun verification over the already-frozen replay corpus | **no**; replay traversal only | frozen verification corpus identity |
| C-vs-A fresh qualification | one-shot 201-204 environment measurement | semantic receipt requalification | **no** | **yes** |
| D-vs-C fresh qualification | one-shot 401-404 environment measurement | semantic receipt requalification | **no** | **yes** |
| Final two-stage claim | pure synthesis | rebuild from recovered v2 C and D tokens | **no** | inherited from both fresh anchors |
| D OOD follow-up | secondary 1201-1204 environment measurement | not part of primary claim recovery | n/a | fresh D token required before execution |

## Parent C recovery

`ParentCQualification` is recreated by `qualify_parent_c_for_dream`.

Recovery must rederive:

1. canonical TrainingReplay selection from the frozen 1-8 corpus;
2. the independent 101-104 holdout gate from the frozen held-out corpus;
3. exact equality with the frozen selection and holdout receipts;
4. `FreshExecutionEligible` status and full replay support.

This stage is intentionally recomputed rather than restored from a serialized capability.

## Dream verification recovery

The 301-304 verification **corpus acquisition** is the protected operation. Once that corpus has been frozen, `validate_grounded_dream_after_parent_c` evaluates C and D only against recorded replay edges.

A restart therefore recovers `QualifiedDreamVerification` by rerunning the verification calculation over the frozen corpus. It must not reacquire or regenerate the 301-304 corpus merely to recover the token.

The frozen corpus receipt/evidence identity must remain unchanged.

## Fresh qualification recovery

Fresh measurements differ because replaying 201-204 or 401-404 against the environment would constitute another measurement execution.

The v2 fresh qualification wrappers therefore commit to the complete semantic measurement receipt, while recovery uses a separately trusted `QualificationAnchor`.

### C-vs-A

Preferred API:

`requalify_replay_fresh_from_anchor`

The recovery path revalidates:

- parent C qualification;
- parent holdout gate;
- training-selection identity;
- complete expected fresh domain/seed pair set;
- every nested run receipt and metric;
- baseline/treatment arms and policy identities;
- every primary contrast;
- domain summaries and aggregate deltas;
- safety and authority predicates;
- disposition;
- the original scientific v1 C-fresh evidence digest;
- the complete semantic v2 qualified-wrapper digest;
- the independently frozen anchor.

No fixture transition is executed.

### D-vs-C

Preferred API:

`requalify_dream_fresh_from_anchor`

Recovery additionally requires the reconstituted `QualifiedDreamVerification` token and validates:

- verification decision is exactly `FreshDreamExecutionEligible`;
- verification/model lineage;
- complete 401-404 pair set;
- C and D nested run receipts;
- D override, prediction, and model-simulation accounting;
- policy churn consistency;
- safety, authority, and generated-evidence boundaries;
- generic compute-cost claim boundary and the explicit absence of an efficiency claim;
- disposition;
- original scientific v1 D-fresh evidence digest;
- complete semantic v2 qualified-wrapper digest;
- independently frozen anchor.

No fixture transition is executed.

## QualificationAnchor trust model

`QualificationAnchor` has its own deterministic integrity digest, but **self-integrity is not authenticity**.

A valid anchor object proves only that its fields have not changed relative to its own digest. The anchor is authoritative only when the object or its digest is loaded from an independently immutable source.

Acceptable sources include:

- a frozen Git evidence commit whose commit identity is independently pinned;
- a signed release/evidence manifest;
- an append-only evidence ledger with independently verified identity;
- another equivalent trust domain that cannot be rewritten together with the serialized measurement artifact.

The following is **not** sufficient:

- storing `qualified_receipt.json` and `expected_digest.txt` beside each other in the same mutable directory and trusting both after restart.

If an attacker or accidental process can rewrite both the measurement receipt and its supposed expected digest, the anchor provides no external authority.

## Anchor contents

Every v1 `QualificationAnchor` binds:

- anchor schema and stage;
- qualified wrapper schema;
- experiment ID;
- preregistration digest;
- subject digest;
- environment digest;
- parent-C qualification evidence digest;
- parent holdout-gate evidence digest;
- stage-specific upstream qualification digest;
- source scientific measurement evidence digest;
- complete semantic qualified-wrapper evidence digest.

For C-vs-A, the upstream qualification is the canonical training-selection evidence digest.

For D-vs-C, the upstream qualification is the qualified dream-verification wrapper evidence digest.

Cross-stage use is rejected.

## Crash/restart state machine

A recovery implementation should follow this order:

1. Load the frozen manifests, training corpus, parent held-out corpus, and their independently pinned identities.
2. Recompute `ParentCQualification`.
3. If D evidence is needed, load the already-frozen 301-304 replay corpus and recreate `QualifiedDreamVerification` by replay-only validation.
4. Load the C fresh serialized qualified receipt and C `QualificationAnchor` from **different trust domains**; recover `QualifiedReplayFresh`.
5. Load the D fresh serialized qualified receipt and D `QualificationAnchor` from **different trust domains**; recover `QualifiedDreamFresh`.
6. Build the v2 qualified two-stage chain from the two recovered tokens.
7. Compare the rebuilt chain identity with any separately frozen final-chain evidence identity before publishing or promoting a claim.

Any mismatch is fail-closed. Recovery must not respond to a mismatch by rerunning fresh seeds automatically.

## Measurement-boundary rules

Recovery code must never:

- reacquire 301-304 when the frozen verification corpus already exists;
- execute 201-204 to recreate `QualifiedReplayFresh`;
- execute 401-404 to recreate `QualifiedDreamFresh`;
- execute 1201-1204 as part of primary-chain recovery;
- convert a blocked/no-run receipt into a fresh-measurement token;
- treat a self-generated anchor as externally trusted until it has been frozen in an independent evidence lineage.

Manual or off-protocol evaluation of a protected partition remains lineage contamination, not a valid recovery mechanism.

## Claim boundary

Recovery restores evidence **authority**, not evidence **strength**.

A correctly recovered token means the current process has re-established the same qualified lineage that previously existed. It does not improve the measured result, change a disposition, create new generalization evidence, or strengthen a scientific claim.
