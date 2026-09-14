# WCARE-46 — Selective child-tree integration protocol v1

Status: `SOURCE_REVIEW_CANDIDATE`
Authority: `MeasurementOnly`
Protocol version: `wcare46-selective-child-tree-integration-v1`

## Purpose

WCARE-45 proves that exact Git ancestry convergence does not imply child-tree integration or child-verifier execution. WCARE-46 advances only the tree-integration dimension.

It imports the complete additive file census from the exact frozen WCARE-42 and WCARE-43 heads into the WCARE-45 tree by reusing their Git blob objects and modes directly.

The governing distinction is:

`exact child tree integrated != child verifier executed != child claim established`

## Exact subjects

Parent WCARE-45 head:

`a53595ad9ef7ebd16eac8e047f70666323083f86`

Frozen child heads:

- WCARE-42: `4bb84790bab3dc6a6d2e30d0ef081950a3954717`
- WCARE-43: `9e4bfd621e3c48ba6335f95b3dd1fd7b36dccf25`

Selective integration commit:

`bc8701b207ac6ddc024f76d4bd84767a120e2804`

Integration tree:

`842881c38aa4d148aad9b5f13fe84a8b36556ef4`

## Complete child census

WCARE-42 contributes exactly 9 additive files. WCARE-43 contributes exactly 15 additive files. WCARE-46 therefore imports exactly 24 files.

Every imported entry must match the manifest's exact:

- source lineage;
- repository path;
- Git mode;
- Git blob SHA-1.

The integration commit was constructed from existing Git blobs rather than rewriting file content through the contents API.

A WCARE-46 verifier must reject any missing entry, extra manifest entry, duplicate path, mode drift, type drift, or blob drift.

## Diff theorem

Relative to exact WCARE-45 head `a53595ad...`, the integration commit must contain:

- exactly 24 changed paths;
- every changed path status `added`;
- zero modifications;
- zero deletions.

No WCARE-45 parent file may be changed by the child-tree integration commit.

## WCARE-42 lock boundary

The standalone path:

`tools/wcare42_builder_attestation_verifier/Cargo.lock`

must remain absent in this tranche.

WCARE-46 must not synthesize a lockfile from an uncontrolled dependency resolution merely to unblock execution. Therefore WCARE-42 executable cryptographic qualification remains unresolved.

## WCARE-45 preflight transition

After exact tree integration, WCARE-45 preflight should report:

- `wcare42_child_tree_present = true`;
- `wcare43_child_tree_present = true`;
- `wcare42_standalone_lock_present = false`;
- classification remains `CHILD_EXECUTION_INDETERMINATE`.

The previous child-tree-absence blockers must disappear. Remaining blockers must include at least:

- `wcare42_standalone_lock_missing`;
- `child_verifiers_not_reexecuted_by_wcare45_preflight`.

## Non-promotion theorem

Tree integration changes no Stage-B establishment claim.

All of these remain false:

- `child_verifier_lineage_established`;
- `wcare42_executable_qualification_established`;
- `wcare43_external_execution_lineage_established`;
- `builder_authentication_established`;
- `preregistration_temporal_precedence_established`;
- `authenticated_preregistered_replication_established`;
- `runtime_authority_granted`.

## Claim boundary

WCARE-46 establishes only exact byte/mode integration of frozen child review units into the WCARE-45 tree. It does not establish that those child verifiers executed, that their dependencies are fully locked, that external builder/TSA evidence exists, that the WCARE subject is correct, or any consciousness, phenomenal-experience, suffering, moral-patienthood, consent, veto/self-preservation, or solved-alignment claim.
