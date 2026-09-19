# SYM-RSI-SEM-003S — Preregistration Amendment 1

Status: **FROZEN BEFORE SEM-003 / SEM-003S MEASUREMENT**

Parent SEM-003S preregistration:

`a8640c13c5cb3e1c56d7006750eb21ea85c200ca`

This amendment changes exactly one protocol binding: the primary SEM-003 implementation subject consumed by SEM-003S.

## Why the subject changed

The original SEM-003S preregistration named primary implementation candidate:

`df3eedea80cb21ffc7f4896b64ff6480ea8d2680`

A later static source audit, still before any accepted SEM-003 or SEM-003S measurement, found duplicated Rust helper definitions in that exact subject. The failure and replacement lineage are recorded in:

`docs/experiments/SYM_RSI_SEM_003_IMPLEMENTATION_REPAIR_1.md`

The compile-invalid candidate remains immutable historical evidence and is not reinterpreted as a valid implementation.

## Replacement primary implementation subject

SEM-003S v1 now requires the canonical primary `Sem3Receipt` to bind:

`dd533164f8aff636f06e49a161fa75b4e70359d1`

This replacement subject remains pending exact-head executable qualification. If it does not qualify, SEM-003S must not execute against it merely because this amendment names it.

## Unchanged stability protocol

Everything else in `a8640c13…` remains unchanged, including:

- eight folds;
- query-identity sort order;
- fold assignment `i mod 8`;
- exactly four omitted calibration queries per dimension per fold;
- 28 retained calibration queries per dimension;
- unchanged alpha `0.10`;
- unchanged SEM-003 quantile rule;
- unchanged candidate-set rule;
- unchanged eight SEM-003 primary criteria;
- `StablePass` requiring canonical primary PASS plus PASS in all eight recalibrated folds;
- no majority-vote fallback;
- no threshold-range cutoff;
- candidate-set drift diagnostics remaining descriptive;
- OOD queries remaining operationally rejected before set construction;
- no protected SYM-RSI measurement partition consumption.

No new outcome threshold is introduced by this amendment.

## Receipt binding update

The eventual SEM-003S top-level receipt must bind both:

- original SEM-003S preregistration SHA `a8640c13c5cb3e1c56d7006750eb21ea85c200ca`;
- this amendment SHA;
- repaired primary implementation subject `dd533164f8aff636f06e49a161fa75b4e70359d1`;
- its own exact stability implementation subject.

A stability receipt bound to `df3eedea…` is invalid under the amended v1 lineage.

## Claim boundary

This amendment authorizes no benchmark execution and establishes no semantic-retrieval result.

It only prevents a known compile-invalid implementation candidate from remaining the declared subject of the frozen stability audit.