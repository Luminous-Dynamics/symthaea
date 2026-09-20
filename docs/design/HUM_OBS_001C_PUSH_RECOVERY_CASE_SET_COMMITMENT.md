# HUM-OBS-001C — Deterministic Push-Recovery Case-Set Commitment

Status: source-design candidate
Issue: #4843
Parent: HUM-OBS-001B / #4841
Authority: benchmark identity only; **no qualification or deployment authority**

## Purpose

Replace caller-invented push-recovery case-set labels with a deterministic commitment over the semantic inputs that actually define `run_push_recovery_matrix`.

## Commitment

The v1 commitment uses BLAKE3 with an explicit domain separator and fixed little-endian numeric encoding. It binds:

- physics rate;
- settle duration;
- evaluation duration;
- recovered capture-margin criterion;
- recovered uprightness criterion;
- recovered hold duration;
- the fixed v1 cardinal/diagonal direction vocabulary;
- the exact ordered force-level sequence.

The template `push_force_n` value is intentionally excluded because `run_push_recovery_matrix` overwrites it for every generated case.

The resulting ID is namespaced as:

```text
push-recovery-v1:<blake3-hex>
```

## Fail-closed inputs

Case-set identity rejects:

- non-finite or non-positive physics rates;
- negative/non-finite durations;
- non-finite recovery criteria;
- uprightness criteria outside `[0, 1]`;
- empty force sets;
- negative or non-finite force levels.

## Run/identity binding

The public `run_and_observe_push_recovery_matrix_v1(...)` function derives the commitment and executes the matrix from the same protocol + force slice before constructing the observatory receipt.

This removes the earlier public path where an already-computed matrix could be paired with an unrelated free-form case-set label.

## Invariants

- same semantic inputs -> same case-set commitment;
- relevant protocol or force changes -> different commitment;
- force ordering is preserved;
- changing the unused template `push_force_n` does not change matrix identity;
- negative capability outcomes remain distinct from benchmark-execution failure.

## Tests

Source tests cover:

- deterministic identity for identical case sets;
- sensitivity to a changed force level;
- exclusion of the overwritten template force;
- empty/negative force rejection;
- commitment retention in observatory output;
- capability failure remaining `Completed` execution with explicit negative evidence.

## Nonclaims

The commitment authenticates no external actor and proves no benchmark result by itself. It establishes deterministic case-set identity only. It does not establish recovery performance, qualification PASS, hardware performance, safety certification, or deployment authority.
