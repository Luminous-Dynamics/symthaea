# WCARE-44 — Authenticated replication aggregation protocol v1

Status: `PREREGISTERED_CONTRACT`
Authority: `MeasurementOnly`
Protocol version: `wcare44-authenticated-replication-aggregation-v1`

## Purpose

WCARE-40 establishes a conservative execution-replication graph using builder provenance and pairwise relationship claims. WCARE-42 is intended to authenticate exact builder provenance/relation receipts. WCARE-43 is intended to establish external temporal preregistration of the exact WCARE-40 plan.

WCARE-44 composes those dimensions without allowing either to manufacture strength in the other.

The governing distinctions are:

`authenticated receipt count != authenticated independent components`

`builder authentication != temporal preregistration`

`temporal preregistration != builder independence`

`candidate I44 <= I40`

No WCARE-44 artifact grants runtime authority.

## Three-layer architecture

WCARE-44 deliberately separates graph mathematics, candidate-contract validation, and final child-verifier promotion.

### Stage A1 — pure candidate kernel

The candidate kernel:

- re-hashes the exact WCARE-40 plan/result/provenance/relation bytes;
- re-derives the WCARE-40 conservative builder graph from exact receipts and exact plan thresholds;
- rejects a WCARE-40 result whose reported component count/partition does not match recomputation;
- applies typed child-verifier observations to the exact WCARE-40 receipt digests;
- adds conservative graph edges wherever authentication is incomplete;
- computes candidate authenticated components;
- keeps builder-authentication and temporal-preregistration candidate dimensions separate.

The pure candidate kernel does **not** establish that its child-verifier observations are genuine. Therefore it cannot set final WCARE-44 establishment booleans.

### Stage A2 — strict candidate-contract front door

The strict Stage-A front door exists in v1 and runs before the candidate kernel. It validates the exact candidate input contract without claiming that the child verifier actually executed.

It requires:

- exact WCARE-41 authentication-plan field census with no unknown/missing fields;
- exact WCARE-41 backend field census for builder and temporal verifier entries;
- valid SHA-256 syntax for bound subject/backend/policy identities;
- valid token syntax for backend identifiers;
- exact `evaluation_utc` shape;
- optional `notes` fields, when present, to remain strings;
- both WCARE-41 complete-builder-coverage and temporal-preregistration requirements fixed true;
- the WCARE-41 `wcare40_frontdoor_sha256` to equal the exact WCARE-40 result's `wcare40_frontdoor_sha256`;
- the WCARE-41 `wcare40_core_verifier_sha256` to equal the exact WCARE-40 result's `wcare40_core_verifier_sha256`;
- exact builder-observation field census and typed status/subject-kind values;
- exact temporal-observation field census and typed temporal status;
- real JSON booleans for `verifier_execution_qualified` and `synthetic` rather than truthy strings or numbers;
- every `wcare42_verifier_sha256` to equal the exact WCARE-41 preregistered builder-verifier executable SHA-256;
- every `wcare43_verifier_sha256` to equal the exact WCARE-41 preregistered temporal-verifier executable SHA-256;
- candidate-kernel exit code to agree with the candidate disposition;
- every final-promotion field to remain false.

Thus Stage A2 establishes a closed attribution chain at the contract level:

`exact WCARE-40 result verifier identities <- exact WCARE-41 authentication plan -> exact claimed WCARE-42/WCARE-43 verifier identities`

A candidate cannot aggregate a WCARE-40 result from one verifier lineage while claiming a different WCARE-40 lineage in WCARE-41, and child observations attributed to a different verifier than the one frozen in WCARE-41 are structurally invalid.

Stage A2 still does **not** establish that the supplied WCARE-42/WCARE-43 result or qualification receipt is authentic, nor that the child verifier actually executed. It only proves the candidate evidence is well-formed and attributed consistently to the preregistered verifier identities.

### Stage B — future child-verifier re-execution and promotion

A qualifying WCARE-44 Stage-B front door must independently re-execute or otherwise establish the exact WCARE-42 and WCARE-43 verifier lineages before it may promote candidate dimensions into final establishment claims.

It must not trust precomputed `ATTESTATION_ACCEPTED`, `ESTABLISHED`, `verifier_execution_qualified = true`, verifier hashes, result hashes, or qualification-receipt hashes merely because they appear in supplied JSON.

Until Stage B exists and executes successfully:

- `child_verifier_lineage_established = false`;
- `wcare42_executable_qualification_established = false`;
- `wcare43_external_execution_lineage_established = false`;
- `builder_authentication_established = false`;
- `preregistration_temporal_precedence_established = false`;
- `authenticated_preregistered_replication_established = false`.

## Exact WCARE-40 binding

The kernel binds exact bytes of:

- WCARE-40 replication plan;
- WCARE-40 replication result;
- every WCARE-40 builder-provenance receipt named by the result;
- every WCARE-40 builder-relation receipt named by the result.

The exact SHA-256 sets of supplied provenance/relation receipts must equal the corresponding digest sets in the WCARE-40 result. Extra, missing, or duplicate receipt subjects are invalid.

Every planned replica must have exactly one provenance receipt. Every unordered pair of planned replicas must have exactly one relation receipt.

Stage A2 additionally verifies that WCARE-41's frozen WCARE-40 front-door/core-verifier identities match the exact WCARE-40 result before A1 interprets graph semantics.

## Recomputing the WCARE-40 baseline graph

The kernel does not trust `effective_independent_components` as a free-standing count.

It replays WCARE-40's v1 graph theorem.

Only subject-eligible replicas whose builder-provenance strength is accepted by the WCARE-40 plan enter the baseline qualified-builder graph.

For a qualified pair to remain separated, all seven WCARE-40 conditions must hold:

1. relation is `Independent`;
2. relation-evidence strength is accepted by the plan;
3. both provenance strengths are accepted by the plan;
4. neither provenance receipt declares conflict of interest;
5. relation is not `ConflictOfInterest`;
6. mechanically derived shared fault domains are empty;
7. builder identity commitments differ.

Every other pair receives an edge.

The kernel mechanically recomputes the shared fault-domain set from exact provenance commitments and requires it to equal the relation receipt's declared set.

The recomputed qualified node set, connected-component count, and component partition must equal the WCARE-40 result exactly. A favorable but inconsistent WCARE-40 count is invalid rather than inherited.

## Candidate builder-authentication observations

A candidate observation binds:

- exact WCARE-40 subject receipt SHA-256;
- subject kind (`BuilderProvenance` or `BuilderRelation`);
- candidate verification status;
- purported WCARE-42 result SHA-256;
- purported WCARE-42 verifier SHA-256;
- purported WCARE-42 qualification-receipt SHA-256;
- whether executable qualification is claimed;
- whether the observation is synthetic.

The strict Stage-A front door requires the purported WCARE-42 verifier SHA-256 to equal the exact builder-verifier executable SHA-256 frozen in the WCARE-41 authentication plan. This is attribution consistency, not proof of execution.

An observation may preserve candidate separation only when all of the following are supplied:

- status `ACCEPTED`;
- subject kind matches the exact WCARE-40 receipt type;
- `verifier_execution_qualified = true`;
- `synthetic = false`.

Even then the result remains a candidate until Stage B re-executes/verifies the child lineage.

Duplicate observations for one receipt are invalid. An observation for an unknown WCARE-40 builder receipt is invalid.

## Monotone authenticated graph overlay

Let `G40` be the recomputed WCARE-40 graph and `G44c` the candidate authenticated graph.

WCARE-44 starts `G44c` with **every edge in G40**.

For a pair that is separated in G40, WCARE-44 preserves that separation only when:

- the exact left provenance receipt has a candidate accepted authentication observation;
- the exact right provenance receipt has a candidate accepted authentication observation;
- the exact relation receipt for that pair has a candidate accepted authentication observation.

Otherwise WCARE-44 adds an edge between the pair.

Thus WCARE-44 may add edges but never delete a WCARE-40 edge.

Consequences:

`0 <= candidate_authenticated_effective_independent_components <= wcare40_effective_independent_components`

and every WCARE-40 connected component must be wholly contained within exactly one WCARE-44 candidate component. WCARE-44 may merge WCARE-40 components but may never split one.

## Complete-coverage candidate

The frozen WCARE-41 authentication plan requires complete builder-attestation coverage.

For the Stage-A candidate this means every exact WCARE-40 builder-provenance and builder-relation receipt named by the WCARE-40 result must have one candidate accepted, execution-qualified, non-synthetic observation attributed to the exact WCARE-41 preregistered builder verifier.

`candidate_builder_authentication_requirements_met` additionally requires:

- complete candidate coverage; and
- candidate authenticated effective components meeting the WCARE-40 plan's minimum effective independent-component threshold.

This is still not `builder_authentication_established`; only Stage B can make that promotion.

## Candidate temporal observation

The temporal observation binds:

- exact WCARE-40 result SHA-256;
- exact WCARE-41 authentication-plan SHA-256;
- purported WCARE-43 result SHA-256;
- purported WCARE-43 verifier SHA-256;
- temporal status;
- whether executable qualification is claimed;
- whether the source policy/result is synthetic.

The strict Stage-A front door requires the purported WCARE-43 verifier SHA-256 to equal the exact temporal-verifier executable SHA-256 frozen in the WCARE-41 authentication plan. This is attribution consistency, not proof of execution.

The Stage-A candidate temporal requirement is met only when:

- status is `ESTABLISHED`;
- `verifier_execution_qualified = true`;
- `synthetic = false`;
- the exact WCARE-40 result and WCARE-41 authentication-plan digests match the aggregate subject.

That candidate state cannot change the builder graph or component count.

Stage B must independently establish the WCARE-43 execution lineage before setting final temporal precedence true.

## Orthogonality

The candidate builder graph is computed without consulting temporal status.

The candidate temporal status is computed without consulting builder component count except at the final conjunction.

Therefore:

- a valid temporal observation cannot increase candidate authenticated builder components;
- complete builder authentication cannot set temporal precedence;
- the conjunction candidate is true only if both candidate dimensions meet their independent requirements.

## Invalid and indeterminate child states

A structurally invalid WCARE-40 subject makes WCARE-44 invalid.

A structurally invalid Stage-A authentication plan or observation makes the Stage-A front door invalid before candidate semantics are interpreted.

Candidate builder observations classify each exact receipt independently as:

- `ACCEPTED`;
- `UNTRUSTED`;
- `REJECTED`;
- `INDETERMINATE`.

`UNTRUSTED`, `REJECTED`, missing, synthetic, or unqualified observations cannot preserve a separated pair. They conservatively add edges where relevant.

If any required builder receipt is `INDETERMINATE`, the candidate builder-authentication status is `INDETERMINATE` even though the graph is still conservatively collapsed for measurement.

Temporal status remains separate:

- `ESTABLISHED`;
- `NOT_ESTABLISHED`;
- `INDETERMINATE`;
- `INVALID`.

A temporal `INVALID` candidate makes the aggregate candidate invalid. A required temporal `INDETERMINATE` remains indeterminate rather than guessed.

## WCARE-42 blocker inheritance

At the time this contract is frozen, WCARE-42's standalone verifier does not have a committed standalone `Cargo.lock`, and therefore executable cryptographic qualification remains blocked.

WCARE-44 must not route around that blocker. Until exact WCARE-42 execution qualification exists, final builder authentication cannot be established.

## Synthetic temporal evidence

WCARE-43 synthetic fixtures may exercise WCARE-44 candidate composition. They can never establish production temporal preregistration and therefore cannot make the final conjunction true.

## Stage-A result boundary

Stage A may expose:

- recomputed WCARE-40 components;
- candidate authenticated components;
- per-receipt candidate authentication census;
- added conservative edges and reasons;
- candidate builder-authentication status;
- candidate temporal status;
- candidate conjunction.

Stage A also proves its supplied child observations are structurally complete and attributed to the exact child verifier identities frozen in WCARE-41, and that WCARE-41's WCARE-40 verifier identities match the exact WCARE-40 result. It does not prove those child verifiers actually executed or that the supplied child result/qualification receipts are genuine.

Stage A must keep these final claims false:

- child-verifier lineage established;
- WCARE-42 executable qualification established;
- WCARE-43 external execution lineage established;
- builder authentication established;
- preregistration temporal precedence established;
- authenticated preregistered replication established;
- subject correctness;
- reviewer independence;
- runtime authority.

## Claim boundary

WCARE-44 does not establish subject correctness, reviewer independence, consciousness, phenomenal experience, suffering, moral patienthood, objective moral truth, binding consent, veto/self-preservation authority, universal TSA trust, network/sandbox isolation, or solved alignment.
