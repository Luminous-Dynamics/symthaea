# LL-009AI — Mediated stage runner

LL-009AI strengthens the AG/AH Site01 campaign-control chain from boundary receipts into controller-enforced execution sequencing.

It does **not** change the scientific meaning of any terrain, uncertainty, horizon, visibility, calibration or mission result. It improves provenance: the campaign controller persists the stage guard, launches the exact reviewed command directly without a shell, waits for completion, replays the locked campaign inputs, and only then emits the AH checkpoint and AI execution receipt.

## Sequence theorem

For each mediated stage:

1. replay AG PREPARED, AH pre-evidence lock, AI execution-plan lock and predecessor checkpoint;
2. create the AH stage guard;
3. persist and fsync the guard receipt before launching the child;
4. resolve the current prepared Python executable and compute its SHA-256;
5. launch the reviewed argv with `shell=False` in the reviewed repository-relative working directory;
6. expose only AI control pointers for the child to discover the guard/stage/evidence root;
7. require zero process exit status;
8. replay the AI execution-plan lock after process exit, catching persistent Git/policy/manifest/stage-map/command-plan/runtime drift;
9. re-read and verify that the persisted guard receipt itself did not change;
10. create and persist the AH post-stage checkpoint;
11. emit one AI execution receipt binding guard, predecessor, checkpoint, exact executable identity, argv, working directory, network-policy semantics and exit code.

The controller therefore establishes a concrete ordering:

`persisted guard -> direct child process -> successful exit -> replay locked lineage -> checkpoint`

This is stronger than a manually produced guard/checkpoint pair whose chronology cannot be inferred from self-hashes alone.

## Command plan

Schema `ll009ai.command-plan.v1` is a reviewed runtime input and is itself frozen by `ll009ai.execution-plan-lock.v1` while the evidence root is still empty.

The plan must reproduce every AH stage in exact order. For each stage it binds:

- exact stage ID;
- exact expected artifact-ID population from the AH stage map;
- exact network semantics;
- one or more ordered commands;
- `runner = prepared_python`;
- exact argv token sequence;
- exact repository-relative working directory.

V1 intentionally supports only the already-prepared Python interpreter. It does not resolve arbitrary tools through `PATH`. The real executable path and SHA-256 are recorded again at execution time.

Shell interpolation is forbidden. Shell metacharacters are ordinary argv bytes. The qualification campaign passes a literal `$(touch SHOULD_NOT_EXIST)` token to the child and requires that no such shell side effect occurs.

## Control environment

The runner adds three explicit control variables to the child environment:

- `LL009AI_GUARD_PATH`
- `LL009AI_STAGE_ID`
- `LL009AI_EVIDENCE_ROOT`

These let a scientific adapter confirm which guarded campaign context launched it. Their exact values are recorded in the AI execution receipt.

The broader scientific runtime identity remains governed by AG/AH environment capsules and boundary re-capture.

## Network semantics

AI v1 deliberately distinguishes policy from enforcement.

For a network-authorized stage the command plan requires:

`network_mode = authorized_network`

For an AG stage declared offline it requires:

`network_mode = declared_offline_only`

The execution receipt always records:

- `network_enforcement_level = declared_only`;
- `network_namespace_enforced = false` on v1 commands.

Therefore V1 does **not** claim kernel-enforced offline execution. A later qualification may add Linux network namespaces or an equivalent capability sandbox, but the semantic class must change only when an actual in-child network probe proves the isolation mechanism.

## Filesystem semantics

AI replays the protected AG Git subject and command plan after child exit, so persistent mutation of protected campaign inputs is fatal before checkpointing.

V1 does not claim continuous read-only filesystem mediation while the child is executing. It records:

- `filesystem_write_boundary_enforced = false`;
- `continuous_filesystem_immutability_proven = false`.

A child that temporarily changes and perfectly restores an input is outside this theorem. A stronger lane would execute inside a read-only subject/input mount with writes restricted to declared stage outputs.

## Non-zero exits

A command with non-zero exit status raises a campaign-control failure immediately. No AH checkpoint and no successful AI execution receipt may be emitted from that run.

The pre-stage guard intentionally remains on disk. This makes partial stage execution visible rather than silently erasing the fact that an attempt began. Before retrying, the operator/controller must reconcile any partial outputs; AH's guard rules reject uncheckpointed evidence growth.

## Real Site01 command plan

This PR does not check in a real Site01 command plan. That is intentional: the exact real AG evidence manifest and AH artifact-stage map have not yet been frozen.

For the real campaign, the reviewed command plan must be created and AI-locked while the evidence root is still empty. It should map the final acquisition/GIS/analysis adapters to the exact six AG stages without shell wrappers or ad-hoc operator commands.

## Qualification

The dedicated AI workflow compiles the AG/AH/AI control stack, requires AH and AI to be inside AG's protected execution surface, and runs a synthetic two-stage campaign.

The synthetic run proves:

- the command plan is locked before evidence exists;
- the child can observe that its guard file already exists;
- the prepared Python executable is launched directly;
- exact argv tokens are preserved;
- a shell metacharacter remains literal and creates no shell side effect;
- the runner replays the command/campaign lock after process exit;
- successful stage output is admitted only through AH checkpointing;
- a second offline-declared stage executes with the weaker `declared_only` network semantic rather than pretending isolation;
- the final AH checkpoint covers all declared stages.

No NASA data are downloaded and no scientific Site01 result is produced by this qualification.

## Evidence boundary

AI changes execution/provenance assurance only.

It does not establish:

- Product90 RMS calibration;
- a joint Q × far-field probability model;
- continuous-hard terrain completeness;
- risk-qualified or deterministic visibility;
- site safety;
- delivered power;
- RF link budget;
- operational readiness; or
- mission authority.

Those remain separate scientific/engineering theorems.
