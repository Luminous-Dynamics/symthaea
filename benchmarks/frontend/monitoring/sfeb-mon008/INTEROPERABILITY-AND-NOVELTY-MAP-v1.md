# MON-008 competitive and interoperability map — 2026-11-25

**Purpose:** avoid duplicate standards work, set the right baselines and isolate the actual research gap. This is a landscape snapshot, not an endorsement or a claim of conformance.

## Stable building blocks

| Building block | Status | Use in SFEB | Do not claim |
|---|---|---|---|
| RFC 9942 COSE Receipts | Published IETF Proposed Standard, June 2026 | Signed VDS receipts and proof transport for inclusion, consistency and other proof types | That a receipt makes the statement factually true |
| RFC 9943 SCITT architecture | Published IETF RFC, March 2026 | Signed Statements, registration, policy, trust anchors, Transparent Statements and auditing | That registration validates the real-world truth of issuer content |
| RFC 9162 | IETF Experimental RFC | MON-007 Merkle proof primitive; preserve exact RFC semantics | That a custom envelope is RFC 9942 interoperable |
| Sigstore Rekor / Witness ecosystem | Existing transparency infrastructure and witness direction | Prefer an adapter and verified witness evidence over a new log or bespoke gossip network | That a local threshold demo is a production witness network |

RFC 9943 requires a TS to produce RFC 9942 COSE receipts, maintain registration policies, and make historical policy/trust-anchor material available so an auditor can reproduce checks applied at registration. The VDS must support append-only, non-equivocation and replayability properties.

## Adjacent AI-agent evidence drafts

All are active Internet-Drafts, not published RFCs. Treat them as work-in-progress comparators and follow their current revisions.

### A. Agent Execution Evidence (AEE), draft-thaha-scitt-agent-execution-evidence-02

- Datatracker snapshot: November 2026; intended Standards Track; active individual Internet-Draft.
- Profiles RFC 9942/9943 with registered agent identity, capabilities/assurance, runtime identity, policy/configuration digests, per-call hash-chained execution receipts, serving-party admission receipts, cross-party call linking, and offline-verifiable bundles.
- Its useful insight is two-sided evidence: the caller records why a call was made; the serving party independently records what it admitted and on what basis. Linked verification compares the same `call_digest` and exposes a mismatch when one side's record is absent or inconsistent.
- Its explicit limits: PASS does not prove the harness recorded every call, that a decision was correct, or that every call-chain hop is present. Harness completeness must be composed with remote attestation and external telemetry. A channel not mediated by the harness may be outside the evidence boundary.

**SFEB response:** use the receipt vocabulary and cross-party binding as a baseline; extend qualification at the sink boundary. The actual destination should produce a receipt bound to atomic operation ID, tenant/resource, authority revision, fencing token, request digest and committed resource revision. A separate principal reads back the exact state. Compare those facts against caller, receiver and transparency records.

### B. AI Agent Execution Profile of SCITT, draft-emirdag-scitt-ai-agent-execution-01

- Datatracker snapshot: October 2026; intended Informational; active Internet-Draft.
- Defines AgentInteractionRecord as COSE_Sign1; Agent Operator as Issuer; independent Evidence Custodian as Transparency Service; sequence completeness and redaction receipts.
- The draft says tamper-evidence does not establish record accuracy. That revision says its evidence receipt is not yet the RFC 9942 form and non-equivocation proofs are not yet defined.
- A sequence gap can expose a record produced but not registered; an action never recorded leaves no sequence gap.

**SFEB response:** retain it as a comparison arm and use its caveats as negative controls. Do not fork its agent-record semantics.

### C. Vaara Receipt, draft-sirkkavaara-vaara-receipt-15

- Datatracker snapshot: October 31, 2026; intended Informational; active individual Internet-Draft.
- Defines JCS-canonical signed decision/execution receipts; the execution receipt links to the decision/predecessor, with standalone offline recomputation.
- Its completeness mechanism includes a signed sequence/running count and a sealing record for a declared held set. That set boundary is not evidence about actions outside it.
- It is a receipt and decision/effect-pair comparator, not a sink-side transaction guarantee by itself.

**SFEB response:** consume or map to this pair where practical; evaluate whether the outcome is independently backed by a sink-issued operation receipt and separately authorized readback.

## The differentiating question

Do not position SFEB as another signed agent receipt, SCITT profile or Merkle log. These areas already have active standards and implementations.

The sharper contribution to test is:

> Can a fresh auditor prove from independent evidence that an external state transition was authorized, atomically accepted by the destination, observed independently, and consistently represented across caller, receiver, sink and transparency records—even after the original operator and database are unavailable?

Keep **record integrity**, **effect truth**, and **observation completeness** distinct:

1. A signature says a key signed bytes.
2. A transparency receipt says a VDS committed a statement under a policy.
3. Caller/receiver linkage says two independently held records describe the same request.
4. A sink-issued receipt says the destination asserts a specific operation outcome.
5. Independent readback says a different principal observed the resulting state.
6. A coverage argument says which paths/events were in scope and how omissions could be detected.
7. Offline verification reproduces each claim; unsupported claims stay missing, contradicted, stale or indeterminate.

No one layer implies the next.

## Interoperability policy

- Keep MON-008's custom Ed25519 binary profile strictly as a state-machine test fixture, not an SFEB wire format.
- Implement RFC 9942/9943 using independently maintained libraries and official vectors; do not hand-roll COSE/CBOR.
- Add AEE/Vaara import/export adapters only behind explicit, version-pinned profiles.
- Use Rekor/Witness or another established TS/witness implementation. SFEB should qualify action/sink/effect evidence above it.
- Treat the remote sink's idempotency, fencing, receipt, readback and observation coverage as the actual external trust boundary.
