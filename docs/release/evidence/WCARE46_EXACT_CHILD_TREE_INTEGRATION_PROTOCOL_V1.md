# WCARE-46 — Exact child-tree integration protocol v1

Status: `PREREGISTERED_INTEGRATION`
Authority: `MeasurementOnly`
Protocol version: `wcare46-exact-child-tree-integration-v1`
Tracks: #2911

## Purpose

WCARE-45 proved that exact WCARE-42/WCARE-43 ancestry could converge without importing either child implementation tree. WCARE-46 performs the next bounded step: integrate the **complete frozen additive file census** of both child review units by exact Git blob identity, while preserving every execution blocker.

The governing distinction is:

`lineage present != implementation present != dependency lock present != child execution proven`

No later stage may collapse those four facts into one readiness bit.

## Exact subjects

- WCARE-45 parent: `a53595ad9ef7ebd16eac8e047f70666323083f86`
- WCARE-42 frozen head: `4bb84790bab3dc6a6d2e30d0ef081950a3954717`
- WCARE-43 frozen head: `9e4bfd621e3c48ba6335f95b3dd1fd7b36dccf25`
- exact selective-integration commit: `7e7aaf0314ea2c15c663351f463afe589c35cf4f`

The integration commit has WCARE-45 as its direct parent and differs from it by exactly 24 added files: 9 from WCARE-42 and 15 from WCARE-43. No WCARE-45 path is modified or deleted by that commit.

## Full-census theorem

For every imported child path, the integration commit and evaluated HEAD must contain the exact frozen blob SHA from the corresponding child head. Counts or sentinel files are insufficient.

WCARE-46 therefore verifies:

1. exact integration-commit parentage;
2. exact 24-path added-file census at the integration commit;
3. exact blob SHA for every imported path at the integration commit;
4. exact blob SHA for every imported path at evaluated HEAD;
5. WCARE-42/WCARE-43 lineage ancestry inherited through WCARE-45;
6. standalone WCARE-42 Cargo.lock presence as a separate fact;
7. child-execution establishment as a separate fact.

## Initial expected state

The initial WCARE-46 subject intentionally has:

- exact lineage convergence: `true`;
- full WCARE-42 implementation census present: `true`;
- full WCARE-43 implementation census present: `true`;
- WCARE-42 standalone `Cargo.lock` present: `false`;
- WCARE-42 executable qualification established: `false`;
- WCARE-43 external execution lineage established: `false`;
- child-verifier execution established: `false`.

Therefore its truthful classification remains:

`CHILD_EXECUTION_INDETERMINATE`

Tree integration is progress in provenance, not executable qualification.

## Fail-closed behavior

A missing path, changed imported blob, unexpected modification/deletion in the selective-integration commit, missing ancestry, or wrong direct parent is `INVALID_PROTOCOL`.

A correct integrated tree with unresolved lock/execution prerequisites is `CHILD_EXECUTION_INDETERMINATE` and returns a non-success preflight exit.

## Promotion boundary

A later Stage-B tranche may strengthen a claim only by explicitly satisfying missing prerequisites. It may not reinterpret:

- exact Git ancestry;
- exact child file presence;
- source-review success;
- queued CI;
- synthetic WCARE-43 fixtures;
- or the absence of a standalone lock

as child execution evidence.

## Non-claims

WCARE-46 does not establish builder authentication, temporal preregistration, authenticated-preregistered replication, subject correctness, reviewer independence, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority, runtime authority, or solved alignment.