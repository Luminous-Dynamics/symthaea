# PIE-009W — Metrology Quarantine, Remediation Generations, and Conservative Re-entry

## Purpose

This independent reference freezes lifecycle semantics for a quarantined metrology reference.

A reference cannot regain trusted status because time passed, because one later comparison happened to agree, or because old evidence is replayed after remediation.

## Core rule

Re-entry requires all of the following:

1. the reference is already `QUARANTINED`;
2. remediation occurs after the quarantine event;
3. remediation advances the reference generation;
4. all supporting evidence is post-remediation and bound to the exact new generation;
5. the candidate agrees with two currently active peer references;
6. those two peers agree with each other; and
7. the three comparison channels forming the witness have no shared declared failure group.

The result is only `REENTRY_ELIGIBLE`. The oracle never mutates the registry back to `ACTIVE` and never grants hardware authority.

## Causal evidence

Evidence collected at or before the remediation step cannot support re-entry.

Evidence bound to the pre-remediation generation also cannot support re-entry.

This prevents a repaired or recalibrated reference from inheriting evidence that described a different physical/logical generation.

## Independent execution

The final candidate oracle was executed locally with Python 3 on 2026-09-13 and returned:

`ok`

The fixture covers:

- successful independent post-remediation re-entry witness;
- pre-remediation evidence rejection;
- insufficient one-peer evidence;
- shared comparison common-mode rejection;
- post-remediation disagreement;
- stale generation evidence;
- quarantined peer exclusion;
- non-advancing remediation generations;
- remediation at/before quarantine;
- deterministic metadata-sensitive eligibility receipts.

## Boundaries

This reference does not perform repair, calibration, certification, digital signatures, authorization, registry mutation, or plant control.

`REENTRY_ELIGIBLE` means only that the declared structural evidence satisfies this oracle's re-entry contract. A separate authority layer must decide whether and when to restore active use.
