# Domain-General Delay-Tolerant Telemetry: Symthaea, Mycelix and Sol-Atlas

**Status:** Research architecture plus deterministic in-memory store-and-forward policy prototype.  
**Threat-model review:** 2026-10-12.  
**Not claimed:** production queue durability, deployed satellite connectivity, source authentication, cryptographic signature verification, a formal safety case, real meter/sensor evaluation, or live Sol-Atlas/Mycelix integration.

## Decision

Use one transport-neutral telemetry contract across energy, water, agriculture, infrastructure and spacecraft-health domains. Keep each domain's physical meaning and admissible action policy separate. The shared layer handles delivery semantics, time integrity, quality/freshness and retry behavior; it must not equate transport success with truth.

This follows the core idea of delay-tolerant networking rather than inventing a satellite protocol. IETF RFC 9171 specifies Bundle Protocol Version 7 for store-carry-forward environments with intermittent connectivity, delay and error-prone links. The prototype here is intentionally smaller than BPv7 and must not be described as a BPv7 implementation: https://www.rfc-editor.org/rfc/rfc9171.html

For actual spacecraft packet interoperability, select a mission-appropriate CCSDS profile instead of assuming an ordinary Internet IoT envelope is a space-link packet: https://ccsds.org/publications/bluebooks/

## Proposed division of responsibility

### Symthaea: inference, prediction and epistemic state

- Consume observations with separately represented observation time, receipt time, measurement quality, freshness and provenance confidence.
- Distinguish an observation from an inference; include the evidence and model revision behind a prediction.
- Treat stale, partial, estimated, unknown-quality and future-dated observations as non-actionable by default.
- Use multi-sensor evidence to update beliefs, and preserve disagreement instead of silently collapsing conflicting reports into one value.
- Decide when a compact normal summary is sufficient and when a bounded diagnostic batch is worth its airtime/energy cost. Such scheduling remains a research objective until benchmarked.

### Mycelix: permission, provenance and coordination evidence

- Verify producer credentials, authorization, signatures and policy receipts at a separate trust boundary; queue validation alone is not authentication.
- Record purpose-bound access, release, correction, revocation and coordination receipts without copying individual sensor payloads into broadly distributed or immutable records.
- Bind receipts to a canonical envelope digest and policy/version identifiers. Do not expose stable participant identities, exact household locations or low-entropy hashes of private measurements as substitutes for data protection.
- Let resource owners authorize which aggregated facts another organization can receive. Do not grant physical control authority merely because an observation or governance receipt is valid.

### Sol-Atlas: resilience and long-horizon system modelling

- Combine authorized summaries across resource dependencies, such as energy → water pumping → food production → health-service continuity.
- Keep the source observations, derived estimates, assumptions and scenario projections distinguishable in the model.
- Compare near-term response, operational maintenance, seasonal planning and multiyear resilience; report uncertainty, affected groups and unobserved dependencies.
- Return requests for more evidence as explicit, bounded policy actions rather than silently collecting finer-grained telemetry.

These are proposed responsibilities and interface boundaries, not claims that these integrations have already been built.

## Shared event contract implemented by the current prototype

The current tools/energy_telemetry/store_forward.py module is a deterministic in-memory policy model. Each event carries:

- Random per-event ID (format checked; randomness is not established by the module).
- Domain label: energy, water, agriculture, infrastructure or spacecraft.
- Source sequence number with an explicitly local source scope.
- Observation time and expiry in UTC; these are distinct from receiver time.
- Maximum actionable age, priority and one of validated / partial / estimated / unknown quality.
- Payload digest and estimated wire size. This prototype carries **no payload**; a digest is not encryption, authentication, or anonymization.

The queue's now_utc argument is injected by the caller for reproducibility. It does not consult wall-clock time.

## Delivery rules and safety invariants

1. **At-least-once until acknowledged.** Selection does not remove an event. The same event can be resent after lost acknowledgements. Receivers must be idempotent. ACKs must refer to an event that was selected and match its digest; production ACKs still need authenticated transport.
2. **Duplicate is not new evidence.** Exact event replay is idempotent. Reusing an ID for different metadata or payload is an identity conflict and fails closed.
3. **Expiry is explicit.** Expiration returns event IDs so the caller can write audit receipts. Queue expiration in this prototype is volatile; a production system must durably record decisions before deleting stored bytes.
4. **Priority is deterministic.** Candidate batches sort by priority, then observation time, then event ID; events that do not fit the byte budget remain pending. Queue capacity errors do not silently evict earlier events.
5. **Freshness does not erase history.** A stale but unexpired observation can be transmitted for historical analysis, clearly marked non-actionable. Expired observations are not selected. Mild future clock skew is marked non-actionable; timestamps beyond configured future skew are rejected.
6. **Quality gates current-state action.** Only a fresh, validated event is actionable in this baseline. Partial, estimated and unknown-quality events may inform later analysis but cannot independently trigger immediate action.
7. **Transport success is not truth.** A well-formed event can still be faulty, compromised, miscalibrated or misleading. Source provenance, sensor-health evidence and authorized domain validation are separate checks.
8. **No silent data loss.** Oversize-for-this-batch events stay pending; full-queue insertions fail explicitly; stale and expired paths have distinct outcomes.
9. **No physical control plane in v1.** The prototype cannot dispatch equipment or send commands. Later actuation requires local interlocks, authenticated commands, bounded authority, replay protection, rollback/fail-safe behavior and a separate safety case.

## Privacy boundary

- Minimize at the source; aggregate before any uplink where the use case permits.
- Keep household/participant membership lists and local source keys outside the wire envelope.
- Do not publish participant lists, exact coordinates, raw fine-grained series or stable per-person identifiers into a shared ledger or satellite envelope.
- Do not publish hashes of low-entropy private values as if they anonymize them. In this prototype the payload digest is only an idempotency binding; a production envelope should authenticate canonical metadata and payload together (for example with a signed canonical-envelope digest). No digest scheme is anonymization, and privacy-sensitive values must not be inferable from published material.
- Keep differential-privacy accounting separate from delivery idempotency. At-least-once transmission must not spend a privacy budget twice for the same exact release, while a semantically new release must be accounted for even if it has similar values.

The policy model does not implement a durable DP accountant, secure aggregation, encryption, key rotation, signatures or storage encryption. Those are mandatory production design tasks, not implied by these tests.

## Reference standards and research anchors

- **IETF RFC 9171 — Bundle Protocol Version 7:** store-carry-forward networking for stressed and intermittently connected paths. Use the existing standard where appropriate instead of treating this policy prototype as a new transport standard: https://www.rfc-editor.org/rfc/rfc9171.html
- **NIST IR 8259A — IoT Device Cybersecurity Capability Core Baseline:** useful device and data security capabilities, including the separation between device identity, data protection, access control and security-state awareness: https://csrc.nist.gov/pubs/ir/8259/a/final
- **CCSDS Blue Books — recommended space communication standards:** select space-packet, encapsulation and link procedures based on mission requirements: https://ccsds.org/publications/bluebooks/

## Verification evidence

The accompanying unittest suite uses synthetic timestamps, domain labels and digest strings only. It covers five domain profiles; duplicate/id conflicts; UTC and future-clock validation; freshness/quality gating; deterministic priority ordering; byte limits; queue capacity; explicit expiry; and digest-checked, repeatable acknowledgements.

Local verification recorded during implementation: **17 store-forward tests passed** under Python's unittest runner. This is a local result for the new prototype tests; it is not the GitHub Actions result, not a run of the full repository test suite, and not field evidence. The exact-head GitHub workflow remains the authority for CI status.

## Required next steps before deployment

- Replace in-memory state with an encrypted durable queue and a crash-consistent transactional receipt/ACK ledger.
- Specify canonical serialization and authenticate message, ACK, source identity, policy version and freshness metadata end to end.
- Define retention ceilings and bounds for terminal receipts, replay windows and participant-level privacy accounting.
- Add deterministic chaos simulation for link loss, reordering, duplication, clock drift, key expiry, reboot/recovery, storage-full and compromised-sensor cases.
- Benchmark bytes and joules per useful observation against cellular, fibre, radio mesh and local buffering; satellite should be selected per route rather than made mandatory.
- Add domain-specific policy tests: grid operations, water-quality alerts, food-security indicators, infrastructure fault reports and spacecraft fault handling must not share unexamined alert thresholds.
- Connect only through narrow versioned interfaces. Mycelix owns trust/policy receipts, Symthaea owns inference state, and Sol-Atlas owns scenario/dependency modelling; none receives blanket authority over the others.
