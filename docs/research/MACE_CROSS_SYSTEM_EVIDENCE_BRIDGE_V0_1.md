# MACE cross-system evidence and authority bridge v0.1

**Snapshot:** 2026-11-22  
**Status:** adapter contract and research mapping; no live production connector or end-to-end cryptographic verification is claimed.

## 1. Purpose

MACE must not create a second source-of-truth system for evidence, provenance, authority, execution or heritage rights. Its job is to transform already-qualified source records into candidate experiences and memory representations, preserve the source chain, and measure whether those memories improve later decisions.

This document maps existing Luminous interfaces into one cross-system contract:

- **Mycelix:** immutable signed source events, append receipts, authority receipts, source/claim lineage, rights and revocation history.
- **Symtropy:** environment and tool execution, action/outcome separation, simulation or physical observations, resource/workflow evidence.
- **Sol-Atlas:** federated evidence references, native-schema and payload identity, transformations, rights, semantic mappings, spatial/temporal context.
- **Symthaea / Broca:** evidence qualification, causal transition hypotheses, consolidation, retrieval, re-observation and portfolio selection.

The adapter is a projection boundary, not a new authority service.

## 2. Non-interchangeable identities

The adapter MUST preserve distinct identities. They answer different questions.

| Identity | What it establishes | What it does not establish |
|---|---|---|
| Native source record ID | Which record the originating system identifies | Byte integrity, authority or semantic correctness |
| Native payload digest | Which exact source bytes/payload were referenced | Who was authorized to publish them |
| Mycelix scientific event ID + event hash | Which immutable signed event and stream position are referenced | Upstream dataset independence or scientific truth |
| Mycelix append receipt | Server receipt time, event hash and committed stream position | The truth of the source proposition |
| Mycelix authority receipt | Actor/key/role/policy/registry snapshot accepted at receipt time | That the proposition is factually correct |
| Upstream source-lineage root digest | Common upstream origin for independence/common-cause analysis | A cryptographic signature or authorization by itself |
| Transformation digest | Exact parser, normalization, aggregation, translation or derived transformation | Validity of the transformation unless qualified |
| MACE experience ID | Candidate-side derived representation | Original evidence identity or proof of causal identification |

**Critical rule:** a Mycelix event hash is not an upstream source-root ID. One dataset release may appear in multiple Mycelix events and publications; those remain one lineage family for independence accounting. Conversely, one event may attach multiple artifacts that have distinct native payload hashes and transformation ancestry.

The current MACE numeric `source_root_id` and `source_family` fixtures are synthetic identifiers only. They MUST NOT be populated from untrusted agent labels in production.

## 3. Mycelix source-event and authority mapping

The current Mycelix-DeSci source provides:

- `ScientificEventEnvelope`: protocol/version/codec/schema, event ID, claim stream ID, sequence, previous event hash, actor, acting organization, occurred-at time, idempotency key and typed payload;
- `SignedScientificEvent`: canonical envelope bytes, signer public key and signature, with a domain-separated event hash;
- `EvidenceArtifact`: content hash, media type, logical locator, license and availability;
- `AppendReceipt`: stream ID, sequence, event ID, event hash and server-generated `received_at`;
- a separate signed scientific authority receipt, binding the accepted event, actor/key binding, registry revision, authorized-key validity/revocation, roles, normalized action, policy ID/version/reason, authority snapshot hash and previous receipt hash.

Primary implementation anchors:

- [Scientific event kernel](https://github.com/Luminous-Dynamics/mycelix/blob/main/mycelix-desci/src/core/src/scientific_events.rs)
- [Scientific authority receipt contract](https://github.com/Luminous-Dynamics/mycelix/blob/main/mycelix-desci/docs/AUTHORITY_RECEIPTS.md)
- [Signed authority delivery envelope](https://github.com/Luminous-Dynamics/mycelix/blob/main/mycelix-desci/src/core/src/authority_delivery.rs)

### Required mapping

| Mycelix field/record | MACE use |
|---|---|
| `event_id`, `stream_id`, `sequence`, `previous_hash`, `event_hash` | Immutable source-event reference and ordered history |
| `occurred_at` | Source-asserted event/observation time; retain clock domain and uncertainty |
| AppendReceipt `received_at` | Authoritative transaction/knowledge time for `known_at` |
| Actor, organization, signer key | Attribution and audit context |
| Signed authority receipt hash and receipt chain | Independent authorization qualification and revocation state |
| EvidenceArtifact content hash | Exact attached evidence payload identity |
| Attestation / correction / withdrawal / retraction / supersession events | Evidence maturity, dispute, retraction and descendant invalidation |
| Source adapter profile + artifact ancestry | Upstream source-family/root identity and transformation lineage |

### Receipt and authority rules

1. A valid event signature proves integrity/control of the event signing key. It does **not** alone prove that the actor was authorized.
2. MACE MUST require the corresponding accepted Mycelix authority receipt for authority-qualified events. A missing receipt is not silently upgraded by verifying the event signature.
3. `known_at` MUST come from the governed server append receipt `received_at`, not from client-supplied `occurred_at`.
4. `occurred_at` remains the event's claimed source time. The source's valid interval, clock mapping and temporal uncertainty remain separate.
5. Later corrections/retractions do not rewrite an earlier `known_at` snapshot. They alter eligibility only at the transaction/knowledge time at which the correction becomes known and effective under the source policy.
6. Historical authority is reconstructed from the signed receipt snapshot. Ordinary later role changes do not silently backdate a loss of authority; compromise/revocation annotations require their own governed semantics.
7. The adapter MUST preserve Mycelix's `legacy_unattested` / `unsafe_unattested` distinction where present. It cannot invent an attestation for legacy history.

## 4. Sol-Atlas federated evidence-reference mapping

The existing [ITIP federated evidence reference example](https://github.com/Luminous-Dynamics/sol-atlas-leptos/blob/main/schemas/examples/ieac-federated-evidence-reference-synthetic-v0.1.json) carries source system, namespace and native object/record IDs; native schema URI/version/digest; native payload URI/digest; source sequence, observed time and ingestion time; rights state, permitted uses and export permission; observations with native property path, semantic concept, value, unit, uncertainty bounds, observed time, evidence state, semantic mapping status and calibration state; transformation lineage and explicit unsupported semantics. Its claim ceiling is `source_reference_and_transformation_lineage_only`.

MACE SHOULD consume this as a federated *reference*, not copy the source record into an authoritative local event. The native payload remains owned by the source system subject to its rights/access contract.

The following must remain explicit:

- `unsupported_semantics` is not an invitation to let a model guess a mapping;
- `export_permitted=false` or unverified/unknown rights block export and external benchmark fixture use;
- `semantic_status` and `calibration_state` constrain which derived claims can be made;
- native schema version/digest changes require a new interpretation/adapter qualification;
- transformations preserve exact input references and method versions.

## 5. Symtropy execution and outcome boundary

Symtropy's Mycelix bridge provides asynchronous request/response transport into Bevy; its content provenance standard requires stable IDs, rights, import/derivation ancestry and validation states. Its historical-evidence runtime separately models evidence objects, provenance graphs, custody, integrity, access/privacy and historical claims.

Useful anchors:

- [Symtropy–Mycelix bridge](https://github.com/Luminous-Dynamics/symtropy/blob/main/crates/bridges/symtropy-mycelix-bridge/src/resource.rs)
- [Content authoring validation and provenance standard](https://github.com/Luminous-Dynamics/symtropy/blob/main/docs/ops/CONTENT_AUTHORING_VALIDATION_AND_PROVENANCE_STANDARD_V0_1.md)
- [Knowledge, archive and historical evidence runtime](https://github.com/Luminous-Dynamics/symtropy/blob/main/docs/tech/KNOWLEDGE_ARCHIVE_AND_HISTORICAL_EVIDENCE_RUNTIME_V0_1.md)
- [Weld bitemporal evidence snapshot oracle](https://github.com/Luminous-Dynamics/symtropy/blob/main/docs/tech/WELD_EVIDENCE_TEMPORAL_V0_1.md)

MACE MUST distinguish the following state transitions:

`proposed action → authorized action → dispatched command → accepted command → observed execution → reconciled outcome`

A tool's `success=true`, a completed work order, a simulation prediction or an accepted command is not itself proof of the intended external effect. Symtropy/environment adapters should provide independent outcome observations bound to exact source/artifact/time/scope receipts.

A later outcome observation may correct future decisions or invalidate dependent memories, but it does not retroactively upgrade the original action record or rewrite what the agent knew at its earlier decision cutoff. An outcome is new evidence with its own provenance, not a mutation of the action event.

Important distinction: Symtropy's current `SpatialAuthority` helper assigns physics computation ownership of bodies to peers. It is not a general Mycelix actor/capability authorization verifier and MUST NOT be treated as such.

## 6. MACE projection contract

The proposed adapter projection is a read-only envelope. It should not duplicate source content or accept authority claims from the model.

```text
VerifiedEvidenceProjectionV1
  native_source_reference
  native_payload_digest
  native_schema_identity
  mycelix_event_reference
  append_receipt_reference
  authority_receipt_reference
  source_lineage_root_digest
  source_family_profile_digest
  observed_time_interval
  known_at_receipt_time
  validity_interval
  rights_and_permitted_uses
  calibration_and_semantic_status
  transformation_chain
  correction_supersession_state
  claim_ceiling
```

Each field is either copied from a verified source object, derived by a named/versioned transform over verified inputs, or explicitly `Unknown`.

The source lineage root is obtained from a verified source profile, dataset/series ID, original artifact digest and transformation ancestry. It is not created by hashing arbitrary display labels. If root independence cannot be established, MACE records `Unknown` and abstains from treating the sources as independent.

A measurement-error bound must be bound to the measurement method, units, instrument, calibration revision, operating interval and source evidence. It is not a free confidence value that Symthaea can lower to make transport pass.

The MACE memory projection is then:

`VerifiedEvidenceProjection → qualified claim/transition candidate → salience/re-observation decision → portfolio option → decision-time eligibility check`

The reverse flow is prohibited: a model-generated memory summary cannot become a Mycelix-signed source event merely by being stored in the projection.

## 7. Cross-system portfolio and rights rules

Portfolio options must be homogeneous in principal, audience, purpose, resource/tenant boundary, policy revision, task scope and bitemporal snapshot. A mixed-scope portfolio is rejected rather than silently merged.

The Sol-Atlas rights envelope informs whether an item may be retained, cached, re-observed, exported to a benchmark, or shared across nodes. Mycelix authority receipts inform whether the principal may perform the requested read/write/derivation. These are separate checks:

- **rights** answer whether the underlying material may be used/exported;
- **authority** answers whether the principal may perform the operation;
- **availability** answers whether the referenced source can be reached in time;
- **integrity** answers whether source bytes and receipts still verify;
- **semantic qualification** answers what claims the evidence can support.

Passing one must not imply the others.

A privacy erasure request is not ordinary cache eviction or active-index forgetting. The adapter must propagate governed erasure through derived MACE memories while preserving only permitted non-content audit evidence. Any retention conflict is an explicit policy decision, not an optimizer trade-off.

## 8. Required adversarial conformance tests

1. Signed event with missing authority receipt: preserve as unattested; reject authority-qualified use.
2. Valid event signature but signer not authorized at server receipt: reject.
3. Client backdates `occurred_at` while server `received_at` is later: do not backdate `known_at`.
4. Late-arriving correction/retraction: preserve prior as-known replay and invalidate future projections only from its applicable transaction/effective time.
5. Multiple Mycelix events referencing one upstream source series: count one source root.
6. One event attaches multiple artifacts with distinct payload digests: preserve distinct artifacts without inferring independent source families.
7. Sol-Atlas unknown rights or export prohibition: deny cross-system export/benchmark inclusion.
8. Unsupported semantic mapping: preserve raw value and mark mapping unknown; no guessed concept.
9. Calibration expired or not effective at decision time: reject the measurement for that decision.
10. Transformation output whose input digest/method revision is absent: reject qualified use.
11. Symtropy tool success without observed external state change: do not promote a successful outcome.
12. Reconciled outcome later superseded/retracted: invalidate dependent transitions and portfolio plans.
13. Model-generated memory summary forged as source evidence: reject without source/authority receipts.
14. Cross-principal, cross-audience or wrong-purpose portfolio composition: reject.
15. Export/caching disallowed by rights profile: reject that representation even if its predicted value is highest.
16. Source lineage root unavailable: treat independence as unknown; do not count claimed families as independent.

## 9. Qualification and benchmark plan

Keep these benchmark claims separate:

| Evaluation | What it qualifies |
|---|---|
| MACE raw-session ingest/consolidation | Candidate formation from raw sessions and receipt lineage |
| Mycelix receipt conformance | Event signatures, authority receipts, receipt time, stream order and revocation state |
| Sol-Atlas federated-reference conformance | Native identity, rights, schema, transformation and unsupported semantics |
| Symtropy execution/outcome conformance | Intent/dispatch/execution/outcome separation and point-in-time outcome receipts |
| MemoryArena / AMA-Bench | Official multi-session agent memory/transfer performance under each benchmark's protocol |
| TimlMemBench | Raw-native governance, activation, revocation and restraint |
| WorldMemArena | Multimodal memory lifecycle write/maintain/retrieve/use, where licensed |
| MACE portfolio frontier | Resource/utility trade-offs of memory representations; not realized task utility unless independently scored |

Minimum release evidence for a production connector:

1. Live Mycelix event and authority receipt verification against trusted key/registry state.
2. Exact source payload and native schema digests.
3. Server receipt-time / bitemporal replay.
4. Rights and purpose decision from the source owner.
5. Transformation and correction lineage.
6. Independent action/outcome receipt for Symtropy execution.
7. Deterministic negative controls listed above.
8. Matched no-memory, full-history, persist-only, locator-only, snapshot-only, greedy, exact scalar and Pareto-frontier baselines.
9. Held-out utility with cluster bootstrap, cost, token, storage, re-observation calls and p50/p95/p99 wall-clock latency.

## 10. Claim boundary and next implementation

This is an adapter contract and cross-system mapping, not a live connector. The Mycelix signed event/authority APIs, Symtropy bridge and Sol-Atlas federated schema were inspected as source contracts; no production cryptographic verifier was added to MACE in this tranche.

Next implementation issue: [MACE cross-system evidence adapter and receipt conformance](https://github.com/Luminous-Dynamics/symthaea/issues/13187). The first code task should implement a read-only verifier boundary over actual Mycelix receipt/event bundles, with a deterministic mock verifier for tests and no caller-constructed `Verified` flag. Rights and source lineage must be resolved before an item can become a qualified MACE experience.
