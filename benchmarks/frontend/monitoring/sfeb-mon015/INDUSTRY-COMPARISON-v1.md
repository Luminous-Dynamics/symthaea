# MON-015 industry comparison and research position v1

**Snapshot:** 2026-11-26. This is a capability comparison, not a measured vendor ranking.

## The paradigm shift

The industry often optimizes one subsystem at a time: workflow throughput, agent traces, tool-call audit, signed receipts, or transparency-log integrity. MON's research proposition is a cross-layer assurance vector whose dimensions remain independently testable:

1. Intent binding — what exact action/request was evaluated?
2. Authority — who or what authorized it, and was authority still valid at the effect's commit point?
3. Dispatch — did execution begin, and was the outcome unknown, failed, confirmed, or reverted?
4. Destination effect — what did the authoritative system of record commit?
5. Independent readback — can a separately authenticated principal observe the exact operation, revision and state digest?
6. Evidence custody — who signed, sealed, registered, timestamped or witnessed the evidence?
7. Capture completeness — what observation boundary is covered, what gaps are known, and what remains unknown?
8. Physical effect — did the real-world system actually do the claimed thing, measured independently?

These dimensions should not be compressed into a single scalar score. A strong transparency receipt cannot compensate for absent sink readback; a gate-level effect attestation cannot compensate for revoked authority; a bilateral reference cannot compensate for shared administration; and a successful database read cannot compensate for missing physical instrumentation.

## Capability comparison

| System / standard | What it already does well | What it does not establish by itself | MON relationship |
|---|---|---|---|
| Temporal | Durable workflow history and retry semantics. Activities may execute more than once, including after a worker completes work but fails before reporting completion. | Receiver-side external mutation is not automatically exactly-once; Temporal recommends idempotency at the called service. | Integrate with Temporal. Use destination-enforced operation IDs, immutable request bindings and independent post-commit readback. Do not replace its workflow engine. |
| DBOS | PostgreSQL-backed durable workflow execution and horizontal scaling. Its 28 Oct 2026 article reports 20K no-op workflows/s on one PostgreSQL server and 1M/s over 50 shards. | The published workload is explicitly no-op workflows with no steps; it is not a measured external-effect assurance workload. | Use as a W0 orchestration throughput reference, then build same-hardware W1-W5 matched tests. |
| OpenTelemetry | Distributed trace context, logs, metrics and cross-service diagnostic correlation. | Trace continuity and telemetry presence do not themselves provide signed action intent, receiver-side commit proof, independent authority recheck, or complete tamper-evident capture. | Use trace/span IDs as diagnostic correlation only; bind them to typed operation IDs and separately verified evidence. |
| OAR 0.3 | Signed action lifecycle, approval binding, authority checks, before/after commitments, confirmation readback, seals, checkpoints and optional RFC 3161 timestamps. | An OAR issuer's signed/confirmed lifecycle still needs source/sink cross-layer correlation, observer independence and externally assessed control-root independence. | Reuse the maintained OAR verifier and official vectors; MON adds causal reconciliation, trust-boundary and adversarial qualification. |
| Agent Action Capsule -06 | Explicit effect states, effect attestation, verdict/effect orthogonality, four independent assurance axes, cross-party evidence and SCITT integration. | An individual Internet-Draft is not an RFC; Capsule effect attestations remain producer-side evidence and do not alone prove independent destination state or physical effect. | Consume the profile; do not fork its wire format. Test the relationship between Capsule evidence and destination/observer evidence. |
| COSE Receipts (RFC 9942) | Generic cryptographic receipts for verifiable data structures. | A receipt proves a VDS-defined statement/proof, not the truth of an external business or physical effect. | Pin VDS profile, log identity, key and trust-root policies. |
| SCITT (RFC 9943) | Signed statement registration and transparency receipts; verifiable append-only history. | Transparency does not prevent dishonest or compromised issuers. Registration is not source truth, semantic correctness, destination state or completeness outside the registered set. | Use SCITT for custody/transparency, while keeping source truth, sink effect, independence and capture coverage separate. |

## Fair industry baselines

### Temporal baseline

Temporal documents Activity retries, including the case where an Activity performs its external work and the worker crashes before reporting completion. Its documentation recommends that the called service be idempotent. MON's comparable workload must therefore use Temporal as the orchestration baseline and test the same receiver idempotency key and operation ledger. Comparing MON with an unprotected non-idempotent receiver would be a strawman.

### DBOS baseline

The DBOS article reports:
- 20,000 workflows/s on one PostgreSQL shard;
- 1,000,000 workflows/s on 50 PostgreSQL shards;
- each shard is an RDS db.m7i.4xlarge with 16 vCPU and 64 GB RAM;
- 48 worker processes per shard at 50 shards;
- the benchmark uses simple no-op workflows with no steps.

Those results are valuable for W0. They must not be reported as a direct MON W2 effect-assurance comparator. MON has more serialized evidence and readback work. Publish a matched workload with the same topology and durability, and separately report the extra cost per qualified effect.

### OAR / AAC / SCITT baseline

OAR is a published open format/toolkit with a public 0.3 specification and implementations. Its public repo snapshot has TypeScript and Python verifiers, and main/compatibility test vectors. The current public repo does not yet implement every OAR -01 proposal. Agent Action Capsule -06 is an individual Internet-Draft. SCITT is an RFC architecture for registering signed statements and receipts. These maturity levels must remain explicit in reports; draft mapping is not standards conformance.

## Where MON should be stronger

- Make the destination—not the agent runtime—the authority for operation uniqueness and conditional commit.
- Resolve lost acknowledgements with the same immutable operation ID, never a fresh ID that could duplicate the action.
- Distinguish commit-pending, observed-committed, qualified-digital-effect, not-confirmed, refused, failed, reverted, indeterminate and contradicted.
- Require a distinct database principal for readback; then separately assess whether the observer is controlled by a different operator, cloud, credential and storage root.
- Re-evaluate authority at the actual commit timestamp or preserve uncheckable where the timestamp/witness is unavailable.
- Keep digest profile, canonicalization, payload type, purpose and subject explicit. Equal hex is not necessarily equal meaning.
- Model completeness as a scoped claim with a declared observation boundary, independent capture source, gap detector and reconciliation protocol.
- Separate digital readback from physical instrumentation, calibration and site witness evidence.

## Research hypotheses

1. Receiver-enforced global operation identity plus independent readback can reduce false-success and duplicate-effect rates under retries, partitions and lost acknowledgements versus application-only idempotency.
2. Commit-time authority checks plus retained commit-time evidence can reduce authorization TOCTOU failures relative to checks performed only at planning or before dispatch.
3. Typed digest references can reduce cross-profile substitution and canonicalization confusion without material throughput cost.
4. Independent holdout mutation testing can reveal semantic false-accept defects missed by the format's own positive-vector suite.
5. Evidence-carrying transactions can reach a useful assurance/cost frontier when expensive cryptographic and external witness checks run off the critical path but cannot upgrade a missing effect result.

These are hypotheses to test, not achieved results.
