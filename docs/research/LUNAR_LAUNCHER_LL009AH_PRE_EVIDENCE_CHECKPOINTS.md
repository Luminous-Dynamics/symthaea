# LL-009AH — Pre-evidence lineage lock and monotonic checkpoints

LL-009AH closes a remaining provenance window in LL-009AG. AG proves exact Git subject, protected campaign files, environment capsules, evidence-manifest semantics, native receipt self-hashes, exact dependency bindings, completeness state and claim ceilings. AH moves the lineage boundary to **before the first scientific output exists**, then admits evidence only through ordered append-only stage checkpoints.

AH is campaign-control evidence. It adds no terrain, visibility, calibration, probability, site, safety, power, RF, operations or mission theorem.

## Protocol

The campaign-control sequence is:

`AG PREPARED -> AH PRE_EVIDENCE_LOCK -> AH GUARD(stage 0) -> scientific stage -> AH CHECKPOINT(stage 0) -> ... -> AG FROZEN/FINALIZED`

AH checkpoint receipts are internal substates of the frozen campaign lineage. They do not replace AG's final evidence-root and semantic receipts.

## Pre-evidence lock

The lock requires the evidence directory to be exactly empty. It binds:

- exact AG PREPARED receipt self-hash;
- exact AG v2 evidence-manifest bytes and canonical payload;
- exact reviewed AH artifact-to-stage map bytes and canonical payload;
- exact AG stage order, environment profile and network-authority bits;
- zero initial evidence files.

Every manifest artifact must be assigned to exactly one AG stage. Stage order/profile/network authority must reproduce the checked AG policy exactly. Duplicate, missing or unknown artifact assignments fail closed.

Because the lock is created before evidence exists, later scientific files cannot retroactively choose a different campaign graph without creating a new lock/root.

## Guard boundary

Immediately before a stage, `guard` replays:

- AG PREPARED, including clean protected Git subject and unchanged environment-capsule files;
- AH lock identity;
- exact manifest and stage map;
- exact predecessor checkpoint;
- exact next-stage order;
- current partial evidence tree.

No uncheckpointed evidence growth, removal or mutation may already be present.

The guard then recaptures the **actual currently executing runtime profile** using AG's environment-capture implementation and requires canonical equality to the capsule bound by PREPARED. Therefore a correct environment JSON file is insufficient if the process is actually running under a different interpreter/package/library/environment-variable state.

The guard records the stage ID, stage index, environment profile, network-authority bit, predecessor checkpoint identity, runtime witness and exact pre-stage file population.

## Checkpoint boundary

Immediately after the external scientific stage, `checkpoint` repeats the lineage/runtime verification and binds the complete partial evidence tree.

Rules are fail-closed:

- every previously checkpointed artifact must remain exactly identical;
- undeclared files fail;
- newly present files may belong only to the currently guarded stage;
- every required artifact assigned to the current stage must exist before the checkpoint succeeds;
- optional diagnostics assigned to the stage may remain absent and are recorded explicitly;
- `not_yet_available` nodes cannot materialize;
- a present child cannot depend on an unavailable parent;
- native receipt self-hashes and reviewed AG dependency-hash paths are replayed unchanged;
- stage skipping, replay or reassignment fails.

Each checkpoint contains the full cumulative artifact-binding state plus only the newly admitted IDs, so the latest checkpoint is sufficient to prove the current append-only population while predecessor hashes preserve the chain.

## Runtime theorem and limitation

AH proves **boundary-time** runtime identity before and after each stage. It does not claim continuous mediation of an arbitrary external process.

Specifically, AH does not prove that a stage process could not temporarily mutate and restore a protected file, nor does an offline stage's `may_access_network=false` bit itself create kernel-enforced network isolation. A stronger theorem would require a mediated/sandboxed stage runner (for example, read-only mounted campaign inputs plus an isolated network namespace/capability boundary).

AH records `continuous_runtime_mediation_performed = false` explicitly on lock, guard and checkpoint receipts so the stronger claim cannot be inferred accidentally.

## Real Site01 stage map

No real checked-in stage map is created in this PR because the final real AG evidence manifest does not yet exist. That is intentional.

For the real Site01 run, the reviewed runtime stage map must assign every exact AG manifest artifact ID once across the checked stages:

1. `source_acquisition`
2. `exact_extraction_and_role_binding`
3. `terrain_and_calibration_audit`
4. `horizon_and_visibility_analysis`
5. `semantic_reconciliation`
6. `campaign_finalization`

The stage-map file itself is hash-bound by the pre-evidence lock before acquisition begins.

## Qualification

The dedicated AH workflow compiles the exact AG/AH control surface, requires `scripts/ll009ah_checkpoint.py` to be inside AG's protected campaign subject, and executes a synthetic campaign with a real temporary Git repository and real runtime capsules.

The synthetic theorem covers:

- deterministic pre-evidence lock replay;
- rejection of a non-empty evidence root before lock;
- exact stage-order enforcement;
- actual runtime drift rejection even when stored capsule files remain unchanged;
- missing required stage output;
- append-only checkpoint success;
- prior-artifact mutation rejection;
- valid child receipt with wrong parent digest rejection;
- undeclared evidence-file rejection; and
- completed final stage checkpoint.

No NASA data are downloaded and no scientific Site01 result is produced by this qualification.

## Real-data boundary

There is still no real AH checkpoint chain. A real chain exists only after:

- exact AG PREPARED is produced on the intended Site01 subject;
- the real AG manifest and reviewed stage map are frozen while the evidence root is empty;
- every acquisition/GIS/analysis stage receives a valid guard and post-stage checkpoint under that same lineage;
- the resulting scientific evidence subsequently satisfies AG's own FROZEN/FINALIZED rules.

Until then AH is control-logic evidence only.
