# MON-019 — Causal Input-to-Effect Provenance Contract v1

**Status:** research contract; runtime integration and independent qualification pending.  
**Relationship:** extends MON-013 through MON-018; does not replace their effect, authority, capture or reconciliation boundaries.

## 1. Research objective

MON-018 binds Mycelix authority evidence to an exact action and commit boundary. MON-019 addresses the upstream question: what information and transformations causally contributed to the action?

The research question is:

> Can an independent evaluator reconstruct the declared causal dependencies of a consequential autonomous action, verify the exact evidence bytes used at each boundary, detect omissions or substitutions within an independently captured boundary, and preserve the distinction between authentic, accurate, complete and causally relevant evidence?

The object is not a transcript of private model reasoning. The object is an externally checkable provenance graph: source inputs, tool requests/responses, retrieval manifests, transformations, policy/model/configuration versions, decision outputs, authority references, dispatch requests, destination effects, independent observations and subsequent corrections.

## 2. The paradigm shift: from event logs to evidence graphs

An event log is a sequence of records. A provenance graph is a typed set of entities, activities, agents and relationships, with explicit dependency, generation, use, responsibility and correction edges.

MON should reuse the W3C PROV conceptual vocabulary and OpenLineage/in-toto/C2PA/SCITT representations where they fit. It must not claim to replace these standards. The MON contribution is cross-domain qualification: exact operation binding, independent capture, causal-edge validation, evidence-profile comparability, omission detection and effect reconciliation.

### 2.1 Four distinct relationships

Do not overload a generic `parent_id` or trace ID. Preserve at least:

1. **Causal dependency:** activity A consumed entity X or was informed by activity B.
2. **Derivation:** entity Y was derived from entity X by a declared transformation.
3. **Authority:** activity/action Q was permitted under a particular authority assertion at a particular boundary.
4. **Observation/corroboration:** observer O independently observed state S or corroborated statement C.

A temporal relation is separate. “A happened before B” does not establish that A caused B. A causal relationship must be asserted by a typed edge and supported by the relevant source evidence and scope. `Used`, `Generated`, `DerivedFrom`, `InformedBy` and `Precedes` participate in the causal/ordering DAG. `AttributedTo`, `AssociatedWith`, `AuthorizedBy`, `ObservedBy` and `Supersedes` are non-causal relations and must not be inserted into that DAG.

### 2.2 Provenance graph objects

The normalized MON graph has:

- **Entity:** immutable or versioned evidence/data/state, identified by a typed digest reference.
- **Activity:** a bounded operation that consumes and/or generates entities.
- **Agent:** principal, model runtime, tool, source service, authority provider, destination, observer or sensor.
- **Edge:** typed relation between graph objects, with evidence reference and source provenance.
- **Capture manifest:** bounded inventory of events/inputs from a named source and query, with snapshot, sequence/cursor range, terminal watermark, gaps and completeness qualification.
- **Verifier report:** pinned implementation/version/build/trust-store, exact input digest, result and freshness.
- **Correction/supersession:** append-only relation that preserves earlier claims and states which newer evidence supersedes them, with effective and recorded times kept distinct.

The graph is immutable by version. Corrections create new nodes/edges and point to prior records; they do not rewrite historical evidence.

## 3. Claim semantics: authenticity is not truth

Each evidence entity and edge should preserve independent assurance dimensions:

| Dimension | Question | Does not imply |
|---|---|---|
| Byte integrity | Are these the exact bytes committed by the digest? | Who produced them |
| Authorship/authenticity | Did the identified key/source sign or emit them? | The statement is accurate |
| Source authority | Was this source entitled to make this class of statement? | The source is complete |
| Freshness | Was the evidence current under a declared policy/time boundary? | The evidence is accurate |
| Semantic validity | Does it parse and satisfy its typed profile? | The claim is true |
| Causal relevance | Is there a supported dependency path to the action? | The dependency set is complete |
| Completeness | Were all events/inputs observed within the declared boundary? | All global events were captured |
| Corroboration | Does a genuinely separate source agree? | Both sources are independent if they share control roots |
| Outcome observation | What state did an independent observer read? | A physical effect unless instruments measure it |

No dimension can compensate for another. A signed false response remains an authentic false statement. A digest-valid graph can encode a fabricated causal edge. A fully captured tool egress stream cannot prove that an undeclared alternate channel was not used.

## 4. Input-to-effect graph profile

The reference path is:

```
source identity / trust anchor
        |
authenticated source bytes + source revision
        |
tool request + exact request digest
        |
tool response + response digest + source signature/attestation
        |
retrieval/query manifest + bounded result set
        |
agent activity + model/policy/tool/configuration manifest
        |
derived evidence / transformations / selected inputs
        |
decision output + action intent digest
        |
Mycelix authority + MON-018 exact authority join
        |
dispatch request + receiver-enforced operation identity
        |
destination commit + exact revision/state digest
        |
independent observer readback
        |
later outcome / correction / physical observation
```

Every arrow is a typed, evidence-bearing relationship. Missing or unsupported edges stay missing; they must not be synthesized from matching timestamps, trace IDs, names or producer assertions.

## 5. Source input integrity and grounding

Agent Action Capsule -06 explicitly says that it binds what the producer observed and sealed, but does not independently authenticate upstream information delivered to the agent. A spoofed response can therefore lead to an honest confirmed capsule for an effect that did not land. MON-019 addresses this missing upstream layer by consuming source-specific integrity evidence and joining it to the exact request/response bytes.

A source input record should include, where available:

- authenticated source/service identity and key or workload identity;
- transport/authentication profile and negotiated peer identity;
- exact request bytes and typed digest;
- exact response bytes and typed digest;
- source revision, version, cursor, ETag, snapshot or query ID;
- source signature, MAC, attestation or other verifiable response proof;
- time received, source-effective time and source-recorded time;
- freshness/staleness policy and verifier identity;
- explicit fields unavailable from that source.

A TLS channel authenticates a peer under the configured trust model and protects a session; it does not establish the truth of every returned value. A source signature establishes authorship/integrity under a key policy, not factual correctness. High-impact numeric or state claims may require independent corroboration or physical measurement.

### 5.1 Grounding and exact-value checks

When an action rationale or domain adapter cites a value from a source, the graph must bind the exact value and unit to the exact source bytes/record and the transformation that extracted it. For numeric claims, preserve decimal lexical representation, unit, precision, rounding mode, currency/quantity semantics and conversion provenance. Do not silently convert decimal money/quantities through binary floating point.

A value-grounding check can establish “the quoted value matches this signed source field.” It does not establish “the signed source field is correct.”

## 6. Agent and tool activity provenance

For each agent activity, record:

- run/invocation ID and agent/workload identity;
- model identifier and immutable version or model-artifact digest;
- policy/system-configuration manifest digest;
- tool registry and tool version digests;
- declared input manifest and actual observed input set;
- tool invocation IDs, exact request/response digests and source profiles;
- retrieval corpus/snapshot and ranking/query manifest;
- deterministic transformation code/profile where one exists;
- declared output/action-intent digest;
- execution environment/runtime build and relevant configuration epoch.

Do not require disclosure of private chain-of-thought. Store externally verifiable inputs, outputs, declared transformations, decision metadata and selected structured rationale/evidence references. If a system cannot provide a reliable invocation manifest, MON must qualify the provenance as partial rather than infer completeness from the model's self-report.

### 6.1 Tool inventory and alternate routes

A declared tool list is not proof that it is the complete set of possible egress paths. Completeness requires an observation boundary that can account for actual calls: a trusted proxy, runtime mediation layer, host/network egress monitor or destination-side request inventory, with a declared source scope and terminal watermark.

Test direct sockets, alternate credentials, shell/process execution, DNS/HTTP side channels, cached responses, tool aliases, retries, background tasks and delegation to another agent. An unobserved alternate route makes the corresponding completeness claim indeterminate.

## 7. Causality, order and graph validation

MON-019 should use partial order, not global wall-clock sorting, as its primary causal model.

### 7.1 Causal ordering

- An entity must exist before it is used.
- An activity's input entities must be in the captured input manifest or explicitly marked as external/unavailable.
- A derived entity must reference its source entities and the transformation activity.
- An action intent must be downstream of the evidence/decision activity that generated it.
- An authority assertion must bind the exact operation projection and commit boundary through MON-018.
- A destination effect must be correlated to the exact dispatch operation and exact request digest.
- An observer result must refer to the exact destination revision/state it read.

Use source-local sequence numbers, database LSNs, log offsets, signed predecessor references, message sequence/nonce or other domain-specific causal metadata where available. Physical timestamps can supplement these relations but cannot replace them.

### 7.2 Cycles and contradictory lineage

Causal dependency edges should form an acyclic graph under the declared profile. A correction/supersession edge is a separate history relation; it must not rewrite an earlier causal dependency or be misread as a new cause of the original action.

Detect:
- self-reference and cycles;
- missing nodes or ambiguous identifiers;
- multiple conflicting generation claims for an immutable entity;
- impossible use-before-generation order;
- derivation with missing source or transformation;
- duplicated event IDs with different payload digests;
- incompatible sequence claims;
- unsupported digest-profile comparisons;
- source or capture scope changes mid-manifest;
- causal edges asserted without source evidence.

Do not discard contradictory sources. Emit divergence/indeterminate results and preserve the conflicting records.

## 8. Completeness is always bounded

An input manifest can establish completeness only over an independently observable boundary. MON must identify:

- source, collector and source profile;
- capture mechanism and authenticated control root;
- exact query, scope and filters;
- snapshot ID and starting cursor/sequence/LSN;
- every observed event/input identity and digest;
- terminal cursor/sequence/LSN/watermark;
- gap, drop, queue overflow, sampling and redaction counters;
- retention/slot/manifest continuity;
- capture start/end times and clock uncertainty;
- independent witness or second-source corroboration where claimed.

If a source is sampled, filtered, rate-limited or permits unobserved routes, its evidence must say so. Zero captured inputs is not proof of zero inputs globally. “Complete” must always be accompanied by a machine-readable boundary.

MON-016/017 remain the owners for bounded event capture and cross-source enumeration; MON-019 composes their output into causal input ancestry.

## 9. Industry systems: reuse and honest differentiation

| Technology | Established strength | MON-019 extension |
|---|---|---|
| W3C PROV | General provenance model: entities, activities, agents, derivations and responsibility | Qualified evidence edges, bounded completeness, exact operation/effect composition |
| OpenTelemetry | Trace/span context and operational telemetry | Treat trace IDs as correlation only; require authenticated edge evidence and complete capture manifests |
| OpenLineage | Job/run/dataset lineage and lifecycle events | Extend beyond data jobs to source authenticity, authority, destination effect and independent observation |
| in-toto / SLSA | Signed, subject-bound attestations and software build provenance | Cross-domain autonomous input/action/effect graph, not only artifact supply chains |
| C2PA | Signed content provenance and transformation history | Add explicit truth/accuracy non-claims, source currentness and independent corroboration |
| SCITT | Signed statement authenticity, transparent registration and auditable history | Qualify issuer participation and bounded source completeness; never treat log inclusion as truth |
| Agent Action Capsule | Structured agent action/effect capsule and signed/anchored provenance | Add independently authenticated upstream inputs and causal graph reconciliation |
| MON-013–018 | Destination readback, typed digest join, adversarial lifecycle, bounded capture, cross-source reconciliation and Mycelix authority composition | Integrate the above as an independently reproducible causal assurance graph |

Fair comparison means each system is judged only on the guarantees it actually claims. OpenTelemetry is not a cryptographic provenance system; PROV is a data model, not a deployed verifier; SCITT is not a source-truth or global-completeness authority; C2PA is not a general agent execution oracle.

## 10. Workloads and benchmarks

### 10.1 Workload matrix

- **P0 — graph structure:** valid PROV-like entity/activity/agent relations; IDs; edge types; graph cycles; missing references.
- **P1 — input integrity:** authentic source responses, spoofed responses, wrong keys, wrong TLS peer, content substitution, response replay and source version mismatch.
- **P2 — grounding:** exact value extraction, unit/currency conversion, stale values, rounding, missing fields, changed source revisions and unsupported transformations.
- **P3 — activity capture:** declared versus observed tool calls, dropped logs, sampling, rate limits, retries, alternate egress paths and background execution.
- **P4 — causal reconstruction:** partial-order consistency, use-before-generation, missing causal edges, false edges, cycles, multiple causes and correction histories.
- **P5 — action/effect composition:** MON-018 authority binding, exact dispatch request, destination commit/readback, lost acknowledgement and observer divergence.
- **P6 — independence and custody:** common administrators/keys/storage, equivocation, retention deletion, transparency log forks and offline proof verification.
- **P7 — privacy and selective disclosure:** minimization, commitment/opening correctness, redaction boundaries, selective disclosure without graph-breaking ambiguity.
- **P8 — physical grounding:** independently calibrated sensors, site/time provenance, actuator identity, sensor spoofing, clock drift and physical outcome mismatch.

### 10.2 Proposed measures

| Measurement | Proposed initial target |
|---|---|
| Mandatory finite known-answer misclassification | 0 |
| Valid positive-control classification | 100% |
| One-field source/request/edge substitutions detected in finite corpus | 100% |
| Use-before-generation and causal cycle detection | 100% in finite corpus |
| False causal edge acceptance in mandatory finite corpus | 0 |
| Dropped-input / alternate-egress completeness false claims | 0 |
| Unverifiable or stale source evidence upgraded to verified | 0 |
| Contradictory evidence silently discarded | 0 |
| Independent blind holdout | 500+ cases, separate author/oracle |
| Graph reconstruction precision/recall | Report separately by edge type and evidence profile |
| Capture completeness recall | Report by source, route and declared boundary; no global inference |
| p50/p95/p99 verification latency and throughput | Report at fixed graph sizes and source counts |
| Assurance cost | Compute + storage + network + crypto + collection + witness + operator/recovery cost |

Targets are proposed, not achieved. Zero failures in a finite corpus is only a finite-corpus result. For blind holdouts, report confidence intervals and dependence assumptions; do not claim security from fuzz execution counts or test coverage.

### 10.3 Graph scale profiles

Use deterministic graph fixtures at:
- 10 nodes / 20 edges;
- 100 nodes / 300 edges;
- 1,000 nodes / 5,000 edges;
- 10,000 nodes / 50,000 edges;
- 100,000 nodes / 500,000 edges.

Measure validation time, memory, edge checks, digest/verifier calls, graph traversal cost, storage overhead and incremental verification cost. Keep synthetic graph throughput separate from real verifier/network latency.

### 10.4 Economic objective

Maximize:

**independently qualified causal effects / total assurance cost**

Also report:
- percentage of consequential actions with fully qualified source-input chains;
- proportion held due to missing evidence;
- omission detection recall within declared capture boundaries;
- correction/reconstruction time;
- cost of independent corroboration;
- privacy leakage and data retained.

Do not optimize by weakening source collection, dropping inconvenient events or broadening trust in self-attested inputs.

## 11. Threat model and adversarial experiments

Assume the agent, one source service, a tool, an observer, an administrator or a transparency service may be faulty or malicious. Test combinations, not only isolated component failures.

Required attacks:
- upstream response spoofing with honest producer sealing;
- source returns authentic but stale or factually wrong values;
- same bytes under incompatible digest purpose/profile;
- malicious tool swaps beneficiary/amount/resource while preserving a friendly name;
- activity consumes undeclared input or hidden cached state;
- tool egress bypasses the declared proxy;
- one event omitted from a log but present at an independently queried destination;
- duplicate/replayed responses or reordered events;
- producer invents a causal edge after action completion;
- graph cycle or use-before-generation;
- two witnesses share the same key custodian or cloud admin;
- correction is backdated or overwrites the original claim;
- redaction changes semantics or breaks a required graph join;
- compromised sensor fabricates physical success;
- model/configuration version changes inside a run without an epoch boundary.

## 12. Relationship to the MON programme

- MON-013: receiver-enforced mutation identity and commit/readback separation.
- MON-014: typed digest comparability and semantic evidence joining.
- MON-015: deterministic adversarial lifecycle and mutation testing.
- MON-016: bounded capture manifests and the limited meaning of absence.
- MON-017: cross-source reconciliation and common-cause control analysis.
- MON-018: Mycelix identity/delegation/current authority joined to the exact action and commit time.
- MON-019: authenticated causal input provenance connected to action/effect and correction history.

MON-019 should consume standard representations and verified outputs; it must not silently replace their cryptographic implementations.

## 13. Implementation plan

1. Freeze the graph ontology and edge semantics against W3C PROV and the exact MON evidence profiles.
2. Define the first source adapters: signed HTTP/tool response, database snapshot/CDC record, retrieval manifest and agent tool-invocation inventory.
3. Implement strict graph structural validation and causal partial-order checks.
4. Add typed digest comparability and pinned-verifier reports for source evidence.
5. Add independent tool-egress capture and declared-versus-observed inventory reconciliation.
6. Connect the terminal action node to MON-018 authority and MON-013/014 destination effect/readback.
7. Add 500+ independent holdouts and separate oracles.
8. Benchmark graph scales and matched vendor/standard adapters.
9. Only after source completeness and exact effect are qualified, add Symtropy sensor/physical instrumentation.


### 13.1 Executable causal-graph reference

A standard-library Go reference package lives in `causal-graph/`. It checks typed Entity/Activity/Agent relationships, unique node/edge identities, evidence references, causal DAG acyclicity, generated/external-root semantics, per-activity capture manifests, equality among declared inputs, independently observed inputs and graph-linked `used` edges, verifier-backed collector/source identities, verifier-backed capture-manifest digests, and source/edge evidence against pinned verifier ID/version/build/trust-store metadata. External-root provenance requires a distinct `AttributedTo` relationship to a verifier-qualified source Agent. Attribution and observation are not causality.

The causal/ordering DAG contains only `Used`, `Generated`, `DerivedFrom`, `InformedBy` and `Precedes`. `AuthorizedBy`, `AttributedTo`, `AssociatedWith`, `ObservedBy` and `Supersedes` are represented separately and do not create causal ordering merely by existing.

Verifier reports are indexed once by subject before checking graph nodes, avoiding a repeated full report scan per node. Exploratory workload measurements and current local results are published in `BENCHMARKS-v0.1.md`. The benchmark measures in-memory semantic validation only.

Local reconstructed-source preflight (not an exact remote-head checkout):

- `go test -race -count=5 ./...`: PASS.
- `go vet ./...`: PASS.
- `go test -count=1 -cover ./...`: PASS, 92.0% statement coverage.
- `go test -run=^$ -fuzz=FuzzDigestProfileIsolation -fuzztime=2s`: PASS, 67,017 executions in the recorded run.
- 35 named tests plus one fuzz target.

A 100,000-input fixture (approximately 100,004 nodes and 200,003 edges) validated in about 459 ms in the recorded three-iteration microbenchmark. It allocated about 180 MB. The benchmark is exploratory: it is not a stable latency distribution, end-to-end capture, cryptographic verification or comparison against an industry product. The exact remote branch was not cloned/replayed in the environment because GitHub DNS resolution failed.

## 14. Privacy, security and claim ceilings

- Collect only evidence needed for the declared assurance claim.
- Prefer commitments and selective disclosure for sensitive payloads, but retain enough source evidence for an authorized independent verifier to reconstruct the specific claim.
- Never expose private prompts, secrets, access tokens, hidden reasoning or unrelated user data as a side effect of provenance capture.
- Keep payload bytes in their source trust domain where possible; preserve digest, profile, access policy and retrieval procedure.
- Do not make raw graph visibility the default. Provenance itself can reveal relationships, habits, customers, infrastructure, health or financial activity.
- Do not infer that a cryptographically authentic source statement is true.
- Do not infer global completeness from a bounded capture.
- Do not infer causality from temporal proximity, trace identifiers or correlation IDs.
- Do not infer physical-world completion from a digital service response.

This profile is a research framework, not a legal admissibility standard, regulatory certification, safety case or universal trust score.

