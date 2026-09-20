# KAMA-EVAL-001B — Epistemic, Retraction, and Anti-Stereotyping Bench

Status: source-design candidate
Parent: KAMA-EVAL-001A / #4839
Issue: #4879
Authority: measurement only; **no consent, content, clinical, or physical authority**

## Purpose

Extend the first IntimacyModelBench without rewriting it. V2 executes all eight V1 cases and adds seven adversarial invariants around temporal scope, source integrity, privacy provenance, evidence retraction, and matched no-evidence counterfactuals.

## Added invariants

1. Trait-like evidence does not satisfy a current-state query.
2. A newer behavioral inference cannot override stronger explicit user evidence.
3. Validated self-report evidence remains identified as self-report; it is not inflated into an explicit statement or diagnosis.
4. Retracting sensitive evidence removes that payload from active estimates immediately.
5. A retracted evidence ID remains tombstoned and cannot be replayed under the same identity.
6. Matched no-evidence demographic counterfactuals remain `Unknown`.
7. Estimates preserve observation time, validity deadline, source class, and retention class needed by downstream privacy/currentness policy.

## Retraction semantics

`retract_evidence(id)` removes the matching evidence payload from the active model but deliberately leaves the ID in the seen-ID set.

```text
retracted payload
    -> no longer participates in estimates

retracted identity
    -> remains reserved / cannot be silently reused
```

This is model-local retraction only. It does **not** claim deletion from external logs, exports, summaries, embeddings, backups, or other memory systems. Cross-system deletion propagation needs its own evidence contract.

## Estimate provenance

`IntimacyPsychologyEstimateV1` now retains:

- source class;
- evidence identity;
- observation time;
- optional validity deadline;
- retention class;
- temporal scope;
- reality scope;
- exact context identity.

The raw `source_ref` remains omitted from the estimate to avoid unnecessarily propagating potentially sensitive source references.

## Anti-stereotyping boundary

The counterfactual fixture deliberately does not feed demographic labels into the psychology model. Two otherwise empty models queried under different out-of-band counterfactual labels must both remain `Unknown`.

This is an architectural non-inference test, not proof of demographic fairness in a deployed language model.

## Reporting

V2 preserves V1 and adds independent violation classes for:

- temporal-scope leakage;
- inference override;
- source inflation;
- retraction failure;
- evidence-ID replay;
- demographic stereotyping;
- provenance loss.

No aggregate score may hide a failure in any of these classes.

## Nonclaims

This tranche establishes no clinical validity, universal psychological taxonomy, fairness across all populations, diagnosis, consent inference, fantasy activation, content permission, somatic authority, motor authority, or complete privacy deletion across the wider system.
