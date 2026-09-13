# WCARE-40 — Independent execution replication protocol v1

Status: `PREREGISTERED_PROTOCOL`
Authority: `MeasurementOnly`
Protocol version: `wcare40-execution-replication-v1`

## Purpose

WCARE-39 binds one execution to an exact source, command plan, environment lineage, and observed process result. WCARE-40 asks a different question: did multiple planned executions reproduce the same result, and how much independent evidentiary weight can those replicas honestly carry?

The governing theorem is:

`matching execution replicas != independent execution evidence`

WCARE-40 therefore separates:

1. **replica agreement** — whether qualified WCARE-39 executions agree on the subject/result commitments;
2. **builder provenance** — which builder identity and fault domains produced each replica;
3. **effective independence** — the conservative connected-component count after provenance and pairwise relationships are applied.

A reproducible FAIL is still a FAIL. Replication strengthens confidence that an outcome is reproducible; it never changes the subject outcome.

## Replication means more than one run

WCARE-40 v1 cannot produce `REPLICATION_SUPPORTED` from one execution.

The plan must contain at least two replica slots, `minimum_qualified_replicas >= 2`, and `minimum_effective_independent_components >= 2`.

A repeated run on one builder can provide useful repeatability evidence, but it cannot satisfy the minimum independent-component requirement by itself.

## Planned replica slots

The WCARE-40 plan contains a complete `replica_slots` array. Every slot binds:

- a stable `replica_id`;
- a privacy-preserving builder identity commitment.

Replica IDs must be unique. Multiple slots may intentionally carry the same builder identity commitment. Such retries remain useful replication attempts but do not become independent builder components merely because timestamps, capsule hashes, or nonces differ.

Every expected slot must be accounted for. Missing expected executions produce `INFRASTRUCTURE_INDETERMINATE`; they are never silently removed from the denominator after results are known.

Unexpected/supplemental execution artifacts are invalid input to v1 rather than extra weight.

### Temporal preregistration boundary

`plan_created_utc` is recorded and the exact plan bytes are subsequently bound into provenance and relationship receipts. That is useful lineage evidence, but v1 does **not** independently establish that the plan truly existed before replica outcomes were inspected.

Accordingly, `preregistration_temporal_precedence_established` is always false in WCARE-40 v1. A stronger claim requires an external authenticated timestamp, transparency-log commitment, trusted source-control attestation, or equivalent pre-result commitment mechanism.

The plan should still be created before execution in normal use; WCARE-40 simply refuses to claim that this ordering has been independently proven by a self-declared timestamp.

## Exact WCARE-39 subject and integrity gate

The plan binds SHA-256 identities for the exact WCARE-39 protocol, runner, and integrity gate used by the campaign. The WCARE-40 verifier recomputes those repository file hashes before evaluating replicas.

Hash identity alone is insufficient. Before any replica analysis, the core verifier executes the exact bound `wcare39-integrity.sh` and requires a zero exit status plus `classification = PASS_PROTOCOL_INTEGRITY`. The SHA-256 of that integrity receipt is recorded in the WCARE-40 result.

If the WCARE-39 integrity gate cannot execute, emits malformed output, or does not pass, WCARE-40 returns `INFRASTRUCTURE_INDETERMINATE` rather than evaluating replicas under an unknown WCARE-39 verifier lineage.

Every replica slot has one builder provenance receipt whether or not execution succeeds. The receipt carries `execution_observed` plus nullable exact SHA-256 commitments for both WCARE-39 PREPARED and FINAL capsule bytes.

When `execution_observed = true`, both capsule digests must be present and both artifacts must be supplied. The core verifier then re-executes the exact plan-bound WCARE-39 runner in `compare PREPARED FINAL` mode and records the SHA-256 of that comparison receipt. A copied FINAL JSON cannot qualify merely because its own fields say `QUALIFIED_EXECUTION`.

When `execution_observed = false`, both capsule digests must be null. That slot remains in the campaign and makes the campaign infrastructure-indeterminate rather than disappearing after outcomes are known.

## Official two-layer verifier

The official WCARE-40 entry point is `wcare40-qualify.py`. It is intentionally separate from `wcare40_verify_replication.py`.

The front door validates internal WCARE-39 capsule semantics before any replication mathematics:

- PREPARED must be a pristine `CAPSULE_PREPARED` / `QUALIFIED` / `NOT_RUN` state;
- PREPARED must have a clean worktree, no drift, no self-binding hash, and no already-executed command records;
- FINAL subject outcome is recomputed from the exact command-record outcomes using WCARE-39 precedence (`INVALID`, then `INDETERMINATE`, then `FAIL`, otherwise all-PASS or `NOT_RUN`);
- a favorable top-level FINAL outcome that contradicts its own command census is rejected;
- FINAL classification, environment-integrity state, and drift representation must be mutually consistent.

Only after these checks does the front door invoke the core verifier.

The core verifier is responsible for exact WCARE-39 integrity execution, PREPARED→FINAL comparison, same-subject checking, required-receipt agreement, builder provenance, mechanically derived shared fault domains, conflict policy, and conservative independence-component recomputation.

The front door also verifies that the core process exit code agrees with the core disposition. A mismatch emits `REPLICATION_INVALID`; the favorable core payload is not forwarded.

Full WCARE-40 results carry SHA-256 identities for both `wcare40-qualify.py` and `wcare40_verify_replication.py` so later readers can identify the exact two-layer algorithm that produced the receipt.

## Qualified WCARE-39 replica

A supplied execution lineage is a qualified WCARE-39 replica only when:

- the re-executed WCARE-39 comparison returns `QUALIFIED_EXECUTION`;
- PREPARED and FINAL use `protocol_version = wcare39-execution-capsule-v1`;
- both artifacts retain `authority = MeasurementOnly`;
- PREPARED is accepted by the front-door pristine-state checks;
- FINAL has `capsule_phase = FINAL`;
- FINAL has `classification = QUALIFIED_EXECUTION`;
- FINAL has `environment_integrity = QUALIFIED`;
- FINAL has an empty drift set;
- FINAL binds the exact supplied PREPARED bytes;
- FINAL's top-level subject outcome agrees with the recomputed command census.

Qualified-environment FINAL capsules whose subject outcome is `PASS` or `FAIL` are subject-eligible replicas. `INVALID`, `INDETERMINATE`, or `NOT_RUN` remain visible but cannot support replication.

## Same-subject requirement

Subject-eligible replicas must agree exactly on:

- Git HEAD;
- command-plan SHA-256;
- subject-digest map;
- deterministic-seed commitments.

If these differ, the artifacts are not replicas of the same subject and the evidence is invalid rather than averaged together.

## Subject outcome agreement

v1 fixes `require_subject_outcome_agreement = true`.

- all subject-eligible replicas PASS → agreed outcome `PASS`;
- all subject-eligible replicas FAIL → agreed outcome `FAIL`;
- at least one PASS and at least one FAIL → `REPLICATION_CONTRADICTED`.

No majority vote can turn a qualified dissenting outcome into agreement.

## Required evidence-receipt agreement

The plan declares zero or more `required_receipt_stage_ids`.

For each such stage, every subject-eligible replica must contain exactly one matching command record and a non-null `output_receipt_sha256`. All non-null commitments for that stage must be identical.

A differing required evidence receipt yields `REPLICATION_CONTRADICTED`. Missing required receipt evidence blocks support and yields `REPLICATION_LIMITED` unless the underlying slot is infrastructure-indeterminate.

Agreement here is agreement of exact receipt commitments. WCARE-40 does not infer the truth of the receipt contents merely because their hashes match.

## Builder provenance

Every replica slot has exactly one builder provenance receipt binding:

- exact WCARE-40 plan SHA-256;
- replica ID;
- whether an execution was observed;
- exact WCARE-39 PREPARED and FINAL capsule SHA-256 values, or null/null when no execution was observed;
- builder identity commitment;
- organization commitment;
- infrastructure/provider commitment;
- toolchain-lineage commitment;
- operator-process commitment;
- evidence-source/storage commitment;
- provenance strength;
- conflict-of-interest flag.

The builder identity commitment must equal the commitment declared for that slot in the plan.

Provenance-strength labels are evidence metadata, not cryptographic authentication. WCARE-40 does **not** establish that an `ExternalVerified` or `InstitutionalAttestation` label is authentic. Authentication remains a separate future gate.

## Complete pairwise relationship census

The verifier requires exactly one relationship receipt for every unordered pair of planned replica slots, even if a slot later fails to produce a qualified execution.

Each relationship receipt binds the exact plan, both replica IDs, and both planned builder identity commitments.

For each pair, shared fault domains are mechanically recomputed from the two builder provenance receipts:

- `BuilderIdentity`;
- `Organization`;
- `Infrastructure`;
- `ToolchainLineage`;
- `OperatorProcess`;
- `EvidenceSource`.

The receipt's `declared_shared_fault_domains` must equal the mechanically derived set exactly. Omitting or inventing a shared domain makes the evidence invalid.

## Conservative independence graph

Only subject-eligible replicas with builder provenance strength accepted by the plan enter the qualified builder graph.

A pair separates into different effective components only when all conditions hold:

1. the pairwise relation is `Independent`;
2. the pairwise relation-evidence strength is accepted by the plan;
3. both builder provenance strengths are accepted by the plan;
4. neither provenance receipt declares a conflict of interest;
5. the relationship receipt is not `ConflictOfInterest`;
6. the mechanically derived shared-fault-domain set is empty;
7. the two builder identity commitments differ.

Every other pair is connected.

Thus `Unknown`, `Related`, conflict, same builder, shared organization, shared infrastructure, shared toolchain lineage, shared operator process, shared evidence source, or weak provenance can only preserve or reduce effective independence. Missing knowledge cannot inflate it.

## Environment replication modes

For each subject-eligible replica, WCARE-40 derives an environment fingerprint from WCARE-39 immutable environment fields including platform, tools, materials, safe/ambient environment commitments, and declared network/sandbox policy.

The result separately reports exact-environment replica groups and the number of distinct environment fingerprints.

Identical environments can demonstrate repeatability. Different environments can demonstrate broader reproducibility. Neither automatically proves independent builders.

## Conflict policy

v1 fixes `require_no_conflict_of_interest = true`.

Any provenance or pairwise conflict-of-interest evidence prevents `REPLICATION_SUPPORTED`, even if numerical minima would otherwise pass.

## Dispositions

`REPLICATION_SUPPORTED`
: every expected slot has an observed, re-verified, subject-eligible qualified WCARE-39 execution; at least two qualified replicas and at least two effective independent components satisfy the plan; subject outcome and required receipt commitments agree; and provenance/conflict policy is satisfied.

`REPLICATION_LIMITED`
: the campaign is complete enough to evaluate, but one or more support requirements are unmet without a direct PASS/FAIL or required-receipt contradiction.

`REPLICATION_CONTRADICTED`
: qualified replicas disagree on PASS versus FAIL, or disagree on a declared required evidence-receipt commitment.

`INFRASTRUCTURE_INDETERMINATE`
: the WCARE-39 integrity gate cannot be established, one or more planned replica slots have no observed execution, or a slot has an infrastructure-indeterminate WCARE-39 lineage that prevents completion of the campaign.

`REPLICATION_INVALID`
: malformed, duplicate, internally contradictory, subject-mismatched, plan-mismatched, PREPARED/FINAL-mismatched, incomplete pair-census, forged shared-domain, unexpected-replica, core-exit/disposition mismatch, or otherwise structurally invalid evidence.

Known contradiction takes precedence over slot-level infrastructure indeterminacy because a missing third replica cannot erase an already observed contradiction. Failure of the global WCARE-39 integrity prerequisite stops evaluation before replica semantics are interpreted.

## Counts that must remain separate

WCARE-40 separately reports:

- expected replica slots;
- observed slots;
- qualified-environment replicas;
- subject-eligible replicas;
- PASS and FAIL replicas;
- raw distinct builder identities;
- qualified builder replicas;
- qualified distinct builder identities;
- accepted independent pairs;
- downgraded independent pairs;
- effective independent components;
- distinct environment fingerprints.

These quantities are not interchangeable.

## Synthetic qualification campaigns

The core adversarial campaign covers at least:

- multiple distinct builders reproducing one PASS result;
- repeated execution by the same builder failing to multiply independence;
- PASS/FAIL contradiction preservation;
- required evidence-receipt contradiction preservation;
- missing planned replica slot as infrastructure-indeterminate;
- forged shared-fault-domain declaration rejection;
- rejection of a one-execution plan as replication;
- execution of the exact WCARE-39 integrity gate before replica evaluation.

A separate front-door campaign verifies a normal supported path through the official entry point and rejects a forged favorable FINAL whose command census recomputes to FAIL.

Passing these synthetic campaigns validates WCARE-40 verifier invariants only. It does not establish any real Symthaea replication claim.

## Claim boundary

WCARE-40 may support the bounded claim that an exact tested outcome was reproducible across a planned set of execution slots and that some number of effective builder/fault-domain components survived a conservative independence analysis.

It does not establish temporal preregistration precedence from `plan_created_utc`, builder identity beyond the supplied commitments, cryptographic authenticity of provenance-strength labels, authenticity of execution merely from self-authored capsule files, reviewer independence, subject correctness, network isolation, sandbox enforcement, consciousness, phenomenal experience, suffering, moral patienthood, objective moral truth, cultural universality, binding consent, veto/self-preservation authority, or solved alignment.

No WCARE-40 artifact grants live cognitive or action authority.
