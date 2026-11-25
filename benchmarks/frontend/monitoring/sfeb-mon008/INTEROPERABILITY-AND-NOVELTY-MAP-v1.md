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

- Datatracker snapshot: November 25, 2026; active individual Internet-Draft. The current Datatracker record has no assigned RFC stream or intended status and explicitly says the draft has no formal standing. The source repository README describes it as intended Standards Track; that author-maintained label is not IETF endorsement.
- Profiles RFC 9942/9943 with registered agent identity, capabilities/assurance, runtime identity, policy/configuration digests, per-call hash-chained execution receipts, serving-party admission receipts, cross-party call linking, and offline-verifiable bundles.
- Its useful insight is two-sided evidence: the caller records why a call was made; the serving party independently records what it admitted and on what basis. Linked verification compares the same `call_digest` and exposes a mismatch when one side's record is absent or inconsistent.
- Its explicit limits: PASS does not prove the harness recorded every call, that a decision was correct, or that every call-chain hop is present. Harness completeness must be composed with remote attestation and external telemetry. A channel not mediated by the harness may be outside the evidence boundary.
- The draft's own §15 implementation-status section calls the implementation a prototype. It says the prototype does not yet implement COSE wrapping, epochs, bundles, SCITT registration, serving-party admission receipts, linked verification, or the `call_digest` / host-binding checks added in -02. Its published Appendix A vectors are still useful as an independent profile-conformance corpus; they must not be mistaken for a complete -02 implementation.

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


### D. WIMSE Evidence Binding -03, draft-thaha-wimse-agent-evidence-binding-03

- Datatracker snapshot: November 2026; published November 1, 2026; active individual Internet-Draft with no formal IETF standing, RFC stream or intended status assigned in Datatracker metadata (the I-D body labels it Informational). The I-D announcement says it is a companion to WIMSE AIMS and makes no change to AIMS.
- Maps AIMS minimum audit fields into AEE Execution Receipt fields and defines the `aee_evidence` workload-credential claim. The claim binds authenticated runtime identity to a persistent `logical_agent_id`, an expected Evidence Service, governing policy and optional assurance context.
- Defines `AEE-Caller-Ref` as a per-request HTTP Structured Field carrying the caller's evidence-service identifier, receipt sequence/hash, transaction ID and call digest. It MUST NOT be carried in a token intended for more than one call.
- Admission and post-hoc audit are deliberately separate: a serving party validates the token, identity, proof of possession, registration, freshness and assurance before acting; the later linked-bundle audit checks caller/serving-party evidence. The serving party computes its own call digest; it does not trust the caller-supplied digest. It records allow or refusal, and MUST refuse if any required admission step cannot complete. The actor chain is attribution, not authority; only the current actor and token claims are used in the admission decision.
- Scope limitation: this is a binding profile. It does not itself specify the evidence retrieval protocol, registry/status service or audit schema. It is not a sink receipt or proof of the resulting external state.

**SFEB response:** add WIMSE -03 as an identity/admission integration arm, and explicitly test credential-to-logical-agent mapping, status freshness/revocation, single-call token binding, caller-reference replay, serving-party host binding and separation between admission and post-hoc execution verification. Keep the actual external destination receipt and independent readback separate.
