# WCARE-40 — Independent execution replication protocol v1

Status: `PREREGISTERED_PROTOCOL`
Authority: `MeasurementOnly`
Protocol version: `wcare40-execution-replication-v1`

## Purpose

WCARE-39 binds one execution to an exact source, command plan, environment lineage, and observed process result. WCARE-40 asks a different question: did multiple preregistered executions reproduce the same result, and how much independent evidentiary weight can those replicas honestly carry?

The governing theorem is:

`matching execution replicas != independent execution evidence`

WCARE-40 therefore separates:

1. **replica agreement** — whether qualified WCARE-39 executions agree on the preregistered subject/result commitments;
2. **builder provenance** — which builder identity and fault domains produced each replica;
3. **effective independence** — the conservative connected-component count after provenance and pairwise relationships are applied.

A reproducible FAIL is still a FAIL. Replication strengthens confidence that an outcome is reproducible; it never changes the subject outcome.

## Preregistered replica slots

The WCARE-40 plan is created before replica outcomes are inspected. It contains a complete `replica_slots` array. Every slot binds:

- a stable `replica_id`;
- a privacy-preserving builder identity commitment.

Replica IDs and builder commitments must be unique as records, but multiple slots may intentionally carry the same builder identity commitment. Such retries remain useful replication attempts but do not become independent builder components merely because timestamps, capsule hashes, or nonces differ.

Every expected slot must be accounted for. Missing expected slots produce `INFRASTRUCTURE_INDETERMINATE`; they are never silently removed from the denominator after results are known.

Unexpected/supplemental capsules are invalid input to v1 rather than extra weight.

## Exact WCARE-39 subject

The plan binds SHA-256 identities for the exact WCARE-39 protocol, runner, and integrity gate used by the campaign. The WCARE-40 verifier recomputes those repository file hashes before evaluating replicas.

A supplied execution artifact is a qualified WCARE-39 replica only when it is a FINAL capsule with:

- `protocol_version = wcare39-execution-capsule-v1`;
- `capsule_phase = FINAL`;
- `classification = QUALIFIED_EXECUTION`;
- `environment_integrity = QUALIFIED`;
- no drift fields.

Qualified-environment capsules whose subject outcome is `PASS` or `FAIL` are subject-eligible replicas. `INVALID`, `INDETERMINATE`, or `NOT_RUN` remain visible but cannot support replication.

## Same-subject requirement

Subject-eligible replicas must agree exactly on:

- Git HEAD;
- command-plan SHA-256;
- subject-digest map;
- deterministic-seed commitments.

If these differ, the artifacts are not replicas of the same preregistered subject and the evidence is invalid rather than averaged together.

## Subject outcome agreement

v1 fixes `require_subject_outcome_agreement = true`.

- all subject-eligible replicas PASS → agreed outcome `PASS`;
- all subject-eligible replicas FAIL → agreed outcome `FAIL`;
- at least one PASS and at least one FAIL → `REPLICATION_CONTRADICTED`.

No majority vote can turn a qualified dissenting outcome into agreement.

## Required evidence-receipt agreement

The plan preregisters zero or more `required_receipt_stage_ids`.

For each such stage, every subject-eligible replica must contain exactly one matching command record and a non-null `output_receipt_sha256`. All non-null commitments for that stage must be identical.

A differing required evidence receipt yields `REPLICATION_CONTRADICTED`. Missing required receipt evidence blocks support and yields `REPLICATION_LIMITED` unless the underlying slot is infrastructure-indeterminate.

## Builder provenance

Each observed replica has one builder provenance receipt binding:

- exact WCARE-40 plan SHA-256;
- replica ID;
- exact WCARE-39 capsule SHA-256;
- builder identity commitment;
- organization commitment;
- infrastructure/provider commitment;
- toolchain-lineage commitment;
- operator-process commitment;
- evidence-source/storage commitment;
- provenance strength;
- conflict-of-interest flag.

The replica's builder identity commitment must equal the commitment preregistered for that slot.

Provenance-strength labels are evidence metadata, not cryptographic authentication. WCARE-40 does **not** establish that an `ExternalVerified` or `InstitutionalAttestation` label is authentic. Authentication remains a separate future gate.

## Complete pairwise relationship census

The verifier requires exactly one relationship receipt for every unordered pair of preregistered replica slots, even if a slot later fails to produce a qualified execution.

Each relationship receipt binds the exact plan, both replica IDs, and both preregistered builder identity commitments.

For each pair, shared fault domains are mechanically recomputed from the two builder provenance receipts:

- `BuilderIdentity`;
- `Organization`;
- `Infrastructure`;
- `ToolchainLineage`;
- `OperatorProcess`;
- `EvidenceSource`.

The receipt's `declared_shared_fault_domains` must equal the mechanically derived set exactly. Omitting or inventing a shared domain makes the evidence invalid.

## Conservative independence graph

Only subject-eligible replicas with builder provenance strength accepted by the preregistered plan enter the qualified builder graph.

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
: every expected slot is observed; all slots produce subject-eligible qualified WCARE-39 executions; subject outcome and required receipt commitments agree; provenance/conflict policy is satisfied; and preregistered minimum qualified-replica and effective-component thresholds are met.

`REPLICATION_LIMITED`
: the campaign is complete enough to evaluate, but one or more support requirements are unmet without a direct PASS/FAIL or required-receipt contradiction.

`REPLICATION_CONTRADICTED`
: qualified replicas disagree on PASS versus FAIL, or disagree on a preregistered required evidence-receipt commitment.

`INFRASTRUCTURE_INDETERMINATE`
: one or more preregistered replica slots are missing or infrastructure-indeterminate, preventing completion of the preregistered campaign.

`REPLICATION_INVALID`
: malformed, duplicate, subject-mismatched, plan-mismatched, incomplete pair-census, forged shared-domain, unexpected-replica, or otherwise structurally invalid evidence.

Known contradiction takes precedence over infrastructure indeterminacy because a missing third replica cannot erase an already observed contradiction.

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

## Claim boundary

WCARE-40 may support the bounded claim that an exact tested outcome was reproducible across a preregistered set of execution slots and that some number of effective builder/fault-domain components survived a conservative independence analysis.

It does not establish builder identity beyond the supplied commitments, cryptographic authenticity of provenance-strength labels, reviewer independence, subject correctness, network isolation, sandbox enforcement, consciousness, phenomenal experience, suffering, moral patienthood, objective moral truth, cultural universality, binding consent, veto/self-preservation authority, or solved alignment.

No WCARE-40 artifact grants live cognitive or action authority.
