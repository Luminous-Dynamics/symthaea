# MON-018 — Authority Evidence Composition Contract v1

**Status:** research contract; adapter and runtime qualification pending.  
**Repository:** Luminous Dynamics / Symthaea.  
**Companion:** Mycelix identity, delegation and governance authority providers.

## 1. Research objective

MON is evolving from an action-evidence verifier into an independent assurance and experimental-science framework. MON-018 addresses a missing composition boundary: evidence that a principal was permitted to perform a specific operation must be bound to the exact operation that the destination actually committed.

The research question is:

> Can an independent verifier establish that the exact principal, delegation, authority epoch, action, resource, destination and request were authorized at the destination's actual commit boundary, using cryptographically verified and sufficiently current evidence?

This is not equivalent to asking whether a DID exists, whether an agent possesses a capability, whether a policy engine returned Allow, or whether an institution previously approved a similar action.

The output is a typed authority-assurance result. It is not an instruction to execute.

## 2. Architectural decision: Mycelix is the authority provider; MON is the independent evaluator

Mycelix should participate directly, but its responsibility must be bounded.

### Mycelix owns

- Principal and institutional identity records.
- Credential issuance, issuer admission and key-rotation history.
- Delegation and attenuation semantics.
- Proposal authority versus execution authority.
- Governance/constitutional authority epochs.
- Grant issuance, revocation and supersession histories.
- The authoritative source records and source-specific history required to reconstruct authority at a given time.

### MON owns

- Independent verification of the evidence bytes and verifier reports it consumes.
- Exact binding between authority and action intent.
- Reconciliation of authority time with the destination commit boundary.
- Control-root and source-independence analysis.
- Explicit missing, stale, contradictory and incomplete outcomes.
- The cross-source evidence graph and adversarial qualification corpus.
- Independent destination readback and eventual physical-effect reconciliation.

### The destination owns

- Enforcement of its own operation identity and idempotency semantics.
- Atomic mutation and durable deduplication, where the provider supports them.
- Fencing or another receiver-enforced mechanism against stale authority epochs.
- An authoritative revision/state record that an independent observer can read.

### Independent witnesses own

- Separate custody and anchoring of evidence commitments.
- Detection of deletion, equivocation and later mutation within their declared coverage.
- A declared control-root graph demonstrating which administrative, cryptographic, storage and operational domains are actually separate.

A Mycelix authority event is an input to MON, not a MON verdict. A MON verdict is not a Mycelix governance decision. A destination's acknowledgement is not an independent observation. A transparency receipt is not proof that a statement is true or complete.

## 3. Existing Mycelix work and reuse boundaries

### 3.1 ROS-005 delegated authority

Mycelix PR #3660 adds Rust primitives for bounded delegation. Its semantic dimensions include principal, delegate, action, resource, proposal/execute mode, grant time, expiry, authority epoch and parent delegation activity. It explicitly separates action authority from consent and from proof that an external action occurred.

MON-018 should reuse these semantics rather than inventing a parallel delegation vocabulary.

The adapter must nevertheless require a reconstructed, authenticated delegation lineage. A caller-provided boolean such as parent_active is not evidence of parent currentness. Each ancestor must be resolved to an authoritative source record, checked for scope attenuation and evaluated at the relevant effective time.

### 3.2 Mycelix provider-authority timeline

Mycelix PR #1836 adds a stateful provider-authority timeline with append-only events, stable event identity, contiguous logical sequencing, idempotent exact replay, sticky integrity halts for contradictory history, issuance/revocation commitments, canonical successor uniqueness and time-relative historical status.

This is a useful authority-history owner. Its declared limits must be carried forward unchanged: the inspected increment binds issuance/revocation proof commitments but does not cryptographically verify those proofs; threshold attestation and historical committee-state reconstruction are not thereby established; a pure timeline theorem is not live runtime activation.

MON must not turn a structurally well-bound proof commitment into a cryptographically verified signature.

### 3.3 Legacy TypeScript capability manager

The inspected Mycelix TypeScript CapabilityManager is in-memory and its generateProof routine is a SHA-256 digest over capability ID, issuer DID and timestamp. That is not an issuer signature or a credential proof. Local maps, locally updated use counts, Date.now and pseudo-random IDs do not establish distributed currentness or replay-safe consumption.

Do not treat that implementation as an authoritative cryptographic evidence source for consequential MON actions. This is a specific limitation of the inspected implementation, not a claim about every Mycelix identity component.

### 3.4 Mycelix maturity

The Mycelix repository describes itself as a pre-alpha research project and distinguishes unit-test coverage from multi-agent integration maturity and deployed functionality. MON-018 therefore begins with read-only source fixtures and a versioned adapter contract. Runtime authority activation is a separate qualification gate.

## 4. Authority is a set of independently checked claims

MON must not collapse these claims into a single “authorized” boolean:

1. **Identity:** the evidence binds the acting principal to an authenticated subject and the relevant key/controller state.
2. **Issuer authority:** the issuer was admitted under the exact institutional/root-authority profile that applies to the grant.
3. **Delegation:** every delegation hop is valid, non-amplifying, correctly ordered and not revoked/superseded at the relevant time.
4. **Action scope:** the grant permits the exact action class and operation mode. Proposal authority is not execution authority.
5. **Resource scope:** the grant covers the exact tenant, institution, jurisdiction, asset, resource and destination.
6. **Intent binding:** authority covers the exact canonical request digest and its declared semantic projection, not a human-readable operation label.
7. **Temporal currentness:** grant and all authority-bearing dependencies were valid at the actual effectuation time.
8. **Evidence authenticity:** the signatures, credential proofs, status/revocation evidence and trust anchors have been verified by pinned implementations.
9. **Source coverage:** the history query has a declared snapshot/cursor/terminal boundary and no unaccounted gap relevant to the claim.
10. **Independence:** the purported corroborating authority, observer and witness do not share a control root that defeats the independence claim.

A result can be positive on some dimensions and indeterminate on others. Missing revocation evidence is not silently equivalent to “not revoked.” Missing issuer history is not a permissive default.

## 5. Required evidence envelope

The adapter should normalize, but not erase, source-native evidence. Each field must retain an exact source reference or a typed digest reference to the bytes from which it was derived.

### 5.1 Source identity and provenance

Required fields:

- Mycelix hApp/DNA and role identity.
- Source profile name and version.
- Source event ID, source sequence and source record/action hash.
- Source database/chain identity where applicable.
- Issuer DID, credential/key identifier and credential profile.
- Authority root/constitution identifier, digest and epoch.
- Snapshot identity, query identity, cursor or sequence range, terminal watermark and capture completion state.
- Source event effective time and recorded/observed time, kept distinct.

A friendly name such as “governance” is not a trust anchor. The adapter must bind to concrete keys, profile digests, epochs and source histories.

### 5.2 Principal and delegation

Required fields:

- Principal subject and authenticated controller/key binding.
- Delegate subject and authenticated controller/key binding.
- Grant/delegation identifier and typed commitment.
- Full parent lineage, with one record reference per hop.
- Each hop's issuer, grantee, action, resource, mode, jurisdiction and epoch.
- Parent/child scope relation and attenuation result.
- Grant effective interval and terminal event history.
- Explicit distinction between proposal permission, execution permission, observation permission and administrative permission.

No ambient privilege, broad wildcard or role label may be inferred where the evidence does not encode it.

### 5.3 Exact operation binding

The authority envelope must bind the same immutable operation projection as the action/effect evidence:

- destination/provider identity and provider profile;
- destination tenant and resource;
- logical operation ID;
- principal and delegate;
- action class and exact operation mode;
- request digest algorithm, profile/version, canonicalization and payload type;
- exact request digest;
- amount, quantity, recipient, beneficiary, asset, audience or other domain-specific high-risk fields where applicable;
- relevant resource precondition/revision;
- authority epoch and policy/constitution digest;
- provider route and idempotency key where supported.

A digest over a lossy projection does not establish semantic equality. Each adapter must enumerate the fields it does not support; unsupported fields cannot be assumed equal.

The MON typed-digest rule remains mandatory: matching hexadecimal text under different algorithms, canonicalization rules, payload types or purpose profiles is incomparable, not equal.

### 5.4 Bitemporal currentness

Every authority transition needs at least:

- **effective time:** when the transition is claimed to govern authority;
- **recorded time:** when the source committed or recorded the transition;
- **observed time:** when the independent verifier obtained the record;
- **source ordering:** sequence, event identity and any source-specific predecessor/cutover reference.

Historical validity is evaluated at the destination's actual commit boundary, not at the auditor's current wall clock and not merely at the time an agent asked for approval.

A revocation recorded after a transaction may still have an earlier effective time. MON must preserve both facts. If the source cannot establish effective-time authenticity or ordering, the result is indeterminate.

### 5.5 Cryptographic and verifier metadata

Every cryptographic assertion must identify:

- verifier implementation, version and binary/source digest;
- credential/signature/receipt profile and version;
- trusted issuer/root set and trust-store digest;
- key status and revocation source;
- verification time and evidence freshness bounds;
- exact input byte digests;
- result code and structured diagnostics.

A boolean supplied by the same producer that supplied the claim is not independent verification. MON should consume normalized reports only from pinned maintained verifiers and must preserve their exact identity and inputs.

The initial implementation should reuse maintained COSE, Verifiable Credential, AAC, OAR and SCITT verification components wherever applicable. MON should not create replacement cryptographic formats or handwritten signature implementations.

## 6. Decision model

The authority adapter returns one of four non-interchangeable outcomes:

| Outcome | Meaning | Required behavior |
|---|---|---|
| qualified | All declared authority predicates for the exact operation and commit boundary are supported | Continue to the independent destination/effect gate; do not dispatch solely from this result |
| denied | Evidence establishes that a mandatory authority predicate failed | Do not dispatch |
| indeterminate | Required evidence is absent, stale, unverifiable, unsupported or outside the declared capture boundary | Hold; obtain evidence or reconcile; never upgrade to qualified |
| divergent | Two admissible histories or sources make incompatible claims | Preserve both; quarantine the affected authority claim; do not choose a winner without an independently defined resolution rule |

The result also carries an assurance vector for identity, issuer, delegation, action scope, resource scope, intent binding, temporal currentness, cryptographic verification, source coverage and independence. These dimensions are non-compensatory. A strong identity proof cannot compensate for a missing revocation status; a transparency receipt cannot compensate for a mismatched request digest.

### 6.1 Authority versus effect

The following implications are forbidden:

- authorized → dispatched;
- dispatched → committed;
- committed → independently observed;
- independently observed → complete source coverage;
- complete digital evidence → physical-world completion.

Authority qualification must remain orthogonal to producer verdict, sink effect state, observer state, custody, completeness and physical outcome.

## 7. Commit-time revocation and fencing

The hard problem is not merely checking whether a grant was valid when a request was formed. It is preventing a stale authorization from being used after revocation, supersession, expiry or constitutional rotation but before the destination commits.

Preferred pattern:

1. Mycelix issues a bounded, signed authority lease over the exact principal, delegate, action, resource, request digest, authority epoch and policy digest.
2. The lease includes a generation or freshness token, not just a wall-clock expiry.
3. The destination enforces that generation/fence atomically with the mutation, or verifies a source-signed currentness assertion whose semantics the destination contract explicitly supports.
4. The destination records the exact accepted authority reference and commit revision.
5. An independent observer reads that destination revision and binds it to the operation and authority reference.
6. MON reconstructs historical authority state at the destination commit boundary and verifies no stale generation or conflicting transition was accepted.

If the destination cannot enforce a fence or expose authoritative commit-time information, MON must say that commit-time currentness was not established. A pre-dispatch policy check is not a substitute.

Do not synchronously query a global DHT for every operation as an assumed availability model. A local policy-enforcement point may use a bounded, signed snapshot/lease, but the destination must enforce its expiry/generation or the assurance level remains lower. Revocation propagation delay must be measured, not hidden.

## 8. Control-root independence

Different service names, DIDs or cloud accounts do not automatically indicate independent control.

Represent a dependency graph across at least:

- human/organizational administrator;
- signing key custodian and key-recovery authority;
- Holochain conductor and DNA authorities;
- cloud account, tenant and control-plane administrator;
- database superuser, replication/backup administrator and WAL custody;
- deployment and build pipeline;
- evidence storage, timestamp authority and transparency service;
- observer identity and operator;
- physical sensor, calibration authority and site operator.

For each independence claim, MON should report common ancestors and unclosed dependencies. If two services share the same administrator, key material, deployment pipeline or mutable storage, they may still provide useful corroboration, but MON must not call them independent control domains.

Mycelix can be the authority source and still be a useful source of evidence. It cannot also be counted as an independent witness to its own issuer history merely because that evidence is replicated across peers in the same administrative trust domain.

## 9. Mycelix adapter phases

### Phase A — read-only source adapter

Consume exact Mycelix source records and emit a normalized evidence bundle. No new authority issuance, payment, or execution pathway is introduced.

Inputs:
- ROS-005 delegation records;
- provider-authority grant/rotation/revocation timeline;
- identity credential and issuer-currentness evidence;
- governance/constitution epoch records.

Outputs:
- source-native record references;
- typed commitments and verifier reports;
- temporal authority reconstruction;
- explicit unsupported/unknown fields;
- no mutable authorization cache presented as source truth.

### Phase B — cryptographic qualification

Only after the relevant Mycelix record format and issuer semantics are frozen:
- verify issuance and revocation signatures;
- verify credential subject/controller binding;
- verify issuer/root authority and key currentness;
- verify threshold signatures and historical committee state where claimed;
- pin trust anchors and verifier versions;
- run upstream conformance vectors and independent mutations.

### Phase C — receiver-enforced currentness

Add a destination adapter that accepts bounded authority generations/leases and enforces fencing atomically with the effect. Test races between grant revocation, authority rotation, failover, dispatch and commit.

### Phase D — independent source reconciliation

Use a separately controlled authority-history witness or independently administered source. Compare event enumeration, terminal watermarks, transition histories and omissions without assuming one source is correct. Report common-cause dependencies.

No later phase is implied by completing an earlier phase.

## 10. Adversarial qualification corpus

The companion corpus begins with 48 synthetic cases across identity, scope, delegation, temporal validity, revocation, history integrity, cryptographic verification, source availability, independence, and operation replay.

It includes positive controls as well as attacks. Expected dispositions are finite known-answer classifications, not statistical security estimates.

Mandatory cases include:
- proposal authority used as execution authority;
- principal/delegate/action/resource substitution;
- parent revocation and delegation amplification;
- a grant revoked before, exactly at, and after commit time;
- historical authority remaining valid for an action committed before later revocation;
- missing/stale key or revocation state;
- commitment present without signature verification;
- timeline gaps, same-sequence conflicts, event-ID reuse and competing successors;
- cross-DNA grant references without a verified source path;
- shared control roots falsely presented as independence;
- reputation/consent evidence incorrectly elevated into execution authority;
- same operation ID with same request digest versus a changed request digest;
- unknown destination outcome after dispatch;
- stale coordinator/failover using an obsolete authority epoch.

Each case specifies a required control: continue to the independent effect gate, deny, hold for evidence/reconciliation, or surface divergence. A qualified authority result does not prove that the operation should be executed or that it already executed.

## 11. Benchmark programme

### 11.1 Matched comparison arms

| System | Fair comparison | Not established by that system alone |
|---|---|---|
| Mycelix authority primitives | Identity, grant/delegation semantics and history reconstruction | Destination effect or independent physical result |
| OPA / Cedar | Policy decision correctness, input completeness, error/fail-closed integration | Provider commit and effect observation |
| Zanzibar-style authorization | Consistent relationship/authorization state and revocation propagation | Destination mutation or physical effect |
| Temporal | Durable workflow orchestration, retry and idempotency integration | Institutional authority truth or independent effect |
| DBOS | PostgreSQL-backed workflow throughput/durability | Full MON effect-assurance semantics |
| PostgreSQL logical decoding | Source event capture, snapshot continuity, WAL/slot recovery | Completeness outside the configured source/query boundary |
| AAC / COSE / SCITT | Signed statement format, signature verification and transparency receipt | Truth of issuer claims or complete issuer participation |
| MON composed profile | Exact authority-action binding, independent source reconciliation, commit-time currentness and explicit unknowns | Physical outcome unless W6 instrumentation is added |

Do not compare MON W2/W3 against no-op workflow throughput. DBOS's published no-op/workflow figures are useful orchestration baselines, not direct effect-assurance comparisons. Temporal documents that an activity may execute more than once if the worker succeeds externally and crashes before reporting completion; its guidance to use idempotency keys at the called service is consistent with receiver-enforced operation identity.

### 11.2 Workload matrix

- **W0 — orchestration:** workflow start/completion, durability and retry baseline with no consequential effect.
- **W1 — effect idempotency:** duplicates, replay, lost acknowledgements, crash-after-commit and receiver-side deduplication.
- **W2 — evidence join:** typed digests, exact request matching, sink commit and independent readback.
- **W3 — authority:** identity, issuer currentness, delegation lineage, scope, expiry, revocation and epoch changes.
- **W4 — completeness/independence:** bounded source enumerations, omissions, equivocation and common-cause control roots.
- **W5 — adversarial concurrency:** revoke-vs-commit, rotation-vs-commit, failover, stale-primary fencing, rollback, restore and mutation/fuzz testing.
- **W6 — physical effect:** independent sensor evidence, calibration, time synchronization and site-level provenance.

For each arm record exact code/version, configuration, input corpus digest, hardware, topology, workload semantics, warm-up, repetitions, confidence intervals, p50/p95/p99 latency, throughput, CPU, memory, network, storage, cryptographic cost, external lookups, operator burden and qualified effects per total assurance cost.

### 11.3 Proposed qualification targets

These are prospective targets, not achieved results.

| Measure | Initial target |
|---|---|
| False qualified outcomes on mandatory finite adversarial corpus | 0 |
| Unauthorized acceptance on mandatory finite corpus | 0 |
| Proposal-to-execution privilege escalation | 0 |
| Stale/revoked authority accepted at a tested commit boundary | 0 |
| False independence acceptance on declared common-root cases | 0 |
| Conflicting authority history silently resolved | 0 |
| Required expected classifications | 100% on frozen corpus |
| Positive-control acceptance | 100% on the frozen valid subset |
| Blinded independent holdout | At least 500 cases, with source and oracle separated |
| W0 assurance-layer overhead | Measure first; propose <=5% only after matched semantics and noise floor are established |
| W2/W3 overhead | Report absolute cost and cost per qualified effect; do not hide lookup or witness costs |
| Unknown outcomes | 100% preserved as unknown/hold, never silently retried |

A result of zero false qualifications on 500 independent holdouts does not prove a zero population error rate. Under an idealized independent Bernoulli model, the rough 95% upper bound is about 3/500 (0.6%); adversarial holdouts are not necessarily IID, so even that bound must be labeled carefully. Publish finite-corpus results and statistical estimates separately.

### 11.4 Primary economic metric

Maximize:

**Qualified effects / (compute + storage + network + cryptography + capture + independent verification + recovery/operator cost)**

Report the assurance-cost frontier rather than selecting a single impressive throughput point. Faster execution cannot compensate for an unauthorized effect, duplicate mutation, false-complete claim, false independence claim or discarded unknown outcome.

## 12. Relationship to existing MON increments

- **MON-013:** receiver-side identity, mutation, commit-pending/readback separation and PostgreSQL enforcement.
- **MON-014:** typed digest comparability and normalized producer/sink/observer joins.
- **MON-015:** adversarial lifecycle and effect-state mutation harness.
- **MON-016:** bounded capture and the limited meaning of an empty enumeration.
- **MON-017:** cross-source event reconciliation and common-cause analysis.
- **MON-018:** authority evidence composed against exact action and commit time.

MON-018 depends semantically on MON-013/014/015, and should consume MON-016/017 evidence where available. It does not assume the earlier PostgreSQL or cryptographic qualification gates have passed.

## 13. Immediate implementation sequence

1. Freeze this adapter contract and the 44-case synthetic corpus.
2. Review each mapping against the exact Mycelix source profile and source-native event history.
3. Implement a read-only parser/normalizer that retains source records and rejects unsupported profile versions.
4. Integrate maintained cryptographic verifiers; do not promote proof commitments to verified proofs.
5. Add action-to-authority typed joins and bitemporal commit-time tests to the MON-015 semantic harness.
6. Add tests for stale leases and receiver-enforced epoch fencing.
7. Build independent holdouts with a separate oracle author; include positive controls and one-field mutations.
8. Run local format, unit, race, static and corpus checks; report each separately from hosted CI, upstream vectors, PostgreSQL runtime and multi-agent Holochain results.
9. Only then consider a runtime integration with a live Mycelix cell.

## 14. Claim ceiling and open questions

MON-018 does not establish:
- cryptographic verification of Mycelix issuance/revocation/threshold proofs;
- current production Mycelix authority state;
- Holochain/DHT multi-agent correctness or global total ordering;
- atomic coupling of Mycelix revocation with arbitrary external provider commit;
- receiver-enforced fencing on any real provider;
- PostgreSQL logical-decoding completeness;
- SCITT issuer completeness or truth of issuer claims;
- independently controlled witness deployment;
- CI qualification, vendor superiority, regulatory compliance or production readiness;
- a physical-world outcome.

Open questions for implementation review:
1. Which Mycelix authority profile is canonical for each domain: ROS-005, provider-authority timeline, or a composed institutional grant?
2. How are identity keys, issuer state and revocation status cryptographically verified and made historically reconstructible?
3. What is the authoritative time/ordering relation between Mycelix effective time and provider commit time?
4. Can the first destination enforce a Mycelix authority generation atomically, or must the result remain indeterminate at the commit boundary?
5. Which independent operator can witness Mycelix history without sharing its administrative control roots?

The intended result is an honest, modular bridge: Mycelix supplies institutional authority facts; MON determines what those facts support; the destination controls the effect; independent observers establish what became visible.