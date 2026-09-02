# VART-WORLD-CREATIVE-001 — Dual Source Identity v1

Status: experimental-integrity contract. It does not authorize confirmatory execution or scientific claims.

## Problem

VART has two materially different kinds of code:

1. the **subject** — Symthaea / World Forge / `symtropy-world-forge-author` behavior being scientifically evaluated;
2. the **instrument** — pilot/confirmatory orchestrators, schemas, auditors, evidence verifiers, attack suites, and analysis tooling used to measure the subject.

When both identities are represented by one monorepo HEAD, changing an auditor or orchestration script can make the apparent "system under test" HEAD change even when the runtime mechanism being evaluated did not. That complicates replication, pilot reruns, and interpretation of post-pilot instrumentation fixes.

## Rule

Every future VART transition/freeze records **two independent immutable source identities**.

### Subject source

The subject source identifies the code whose scientific behavior is under test.

Required identity:

- repository;
- durable ref;
- `subject_head`;
- `subject_tree`;
- qualified WORLD-FORGE v0.5-A ancestor HEAD/TREE;
- subject environment digest;
- subject lock/reproduction manifest digest;
- subject qualification receipt digest.

The subject checkout is clean and is never modified by the experiment instrument during a run.

### Instrument source

The instrument source identifies the code that schedules, records, audits, verifies, and analyzes the experiment.

Required identity:

- repository;
- durable ref;
- `instrument_head`;
- `instrument_tree`;
- instrument manifest digest covering the exact orchestrator/verifier/schema/test files;
- instrument qualification receipt digest;
- clean fresh-checkout verification.

The instrument executes *against* the subject checkout; it is not copied into or committed onto the subject branch merely to run an experiment.

## Same repository is allowed

Subject and instrument may live in the same GitHub repository, but they should be bound by separate refs/commits and executed from separate clean checkouts. A dedicated instrument repository is optional, not required.

## Runtime evidence

Every campaign-level receipt records both:

- `subject_source_head/tree`;
- `instrument_source_head/tree`.

Trial evidence does not need to duplicate every instrument file hash, but it must bind the campaign/instrument digest that the external freeze already anchors.

## Post-pilot changes

Instrumentation-only repairs may change **instrument** HEAD without changing the scientific subject. They still require a fresh pilot when they affect execution/evidence behavior.

A subject mechanism change after pilot inspection is not an instrumentation repair. It follows the preregistered disposition rule and, when scientifically material, requires a new preregistration lineage.

## Confirmatory freeze

The prospective confirmatory freeze retains its existing subject `source` block and additionally binds an `instrument_source` block. The raw source-closure receipts for both are externally hashed before confirmatory outcomes exist.

## Claim boundary

Dual-source closure proves which subject was measured by which instrument. It does not establish efficacy, calibration improvement, generalization, creativity, or consciousness.
