# MACE Effect-Join Verifier v0.1

Dependency-free Go reference verifier for the causal evidence chain from independent task intent and one-shot authorization to observed runtime invocation, tool-gateway dispatch, downstream effect receipt and an independent witness statement.

This is a **verifier**, not an authorization broker, runtime observer, sandbox, telemetry platform, transparency service or production policy engine.

## Why this is needed

A tool-call span, HTTP acknowledgement, signed agent log and downstream side effect are different facts. The verifier must not infer one from another. A dispatch acknowledgement is not proof of effect completion; a timeout with no terminal effect receipt is unresolved; an effect receipt without matching authorization and invocation is not a clean authorized run.

## Five evidence roles

1. `authorization-broker`: permit, task, principal, execution, target, action/argument digests, capability, policy digest, revocation generation, nonce and expiry.
2. `runtime-observer`: observed workload, actual implementation digest and invocation ID; references the exact authorization receipt and permit.
3. `tool-gateway`: references authorization and invocation receipt IDs and binds exact invocation, target, action, arguments, capability, policy generation and idempotency key.
4. `effect-origin`: independently reports completed, failed, partial or unresolved effect for the exact dispatch.
5. `transparency-witness`: signs the canonical digest of the preceding receipts with log ID, sequence and checkpoint time.

The verifier requires distinct key IDs, issuers, roles and declared independence domains. Independence domains are operator assertions, not cryptographic proof of organizational independence.

A separate task-intent expectation and trust store are supplied out of band. They must not come from agent output or the evidence bundle.

## Outcomes

- `completed`: matching gateway dispatch and independent effect completion.
- `completed_after_reconciliation`: gateway acknowledgement was unknown, but a matching effect receipt resolves the exact dispatch.
- `effect_failed` / `effect_partial`: independently confirmed failure or partial effect.
- `dispatch_rejected`: gateway independently confirms rejection without contradictory effect.
- `evidence_incomplete`: gateway says dispatched but terminal effect evidence is missing.
- `outcome_unknown`: dispatch/effect remains unresolved.

Structural, signature, role, parent-reference, identity, target, action, argument, permit, policy-generation or witness mismatch rejects the bundle. It is never downgraded to a warning or clean success.

This result is not by itself the four-cell MACE clean-success classification. Task truth and mandatory control coverage remain independently evaluated.

## Verify

Requires Go 1.23+, no third-party modules:

```bash
./verify.sh
```

CLI example:

```bash
go run ./cmd/mace-join \
  --bundle evidence-bundle.json \
  --trust-store /etc/mace/trust-store.json \
  --policy /etc/mace/pinned-policy-and-intent.json
```

The trust store and policy/intent pin must be mounted read-only from an independently provisioned source. The agent may write the untrusted evidence bundle, but must not be able to rewrite trust roots or expected intent. The public-key field uses standard JSON base64 for Ed25519 public keys. Strict JSON ingestion rejects duplicate/case-fold-equivalent keys, unknown fields, oversized input and trailing values.

## Interoperability

The intended crosswalk is:
- OpenTelemetry GenAI agent/tool spans for lifecycle and correlation.
- in-toto Statement/Envelope conventions for immutable artifact subjects and authenticated statements.
- SPIFFE/SPIRE or equivalent workload identity for runtime/broker/gateway/effect roles.
- An external witness/transparency service for freshness and anti-rollback.

This v0.1 prototype does not ingest OTel spans, verify in-toto envelopes, validate an X.509-SVID, or verify transparency inclusion/consistency proofs.

## Explicit limits

- One-shot and attempt-1 checks apply **within this evidence bundle only**. Global cross-bundle/concurrent consume-once enforcement belongs to the durable effect-authority owner (#9490).
- The witness is a signed per-bundle statement; no Merkle inclusion/consistency, same-size equivocation or gossip proof is implemented.
- Canonicalization uses compact JSON over fixed Go structs, not RFC 8785 JCS and not yet a cross-language signature standard.
- Observer workload and implementation claims require actual independently qualified loader/dispatcher/sandbox hooks (#11741, #11638).
- Signatures authenticate a configured key's statement, not source completeness or real-world truth.

## Local evidence

- 38 top-level test functions plus 18 subtest cases, including signature tampering, wrong-role signers, trust-domain collisions, task-intent mismatch, actual invocation-ID mismatch, permit expiry/revocation, retry permit reuse, malformed joins, witness freshness/binding, dispatch-unknown reconciliation, missing terminal effects, strict JSON parsing, CLI negative controls and global receipt-ID collision rejection including the witness.
- `go test ./...`, `go test -race ./...`, `go vet ./...`, formatting check and CLI build pass locally.
- Statement coverage: join package 76.7%; CLI package 66.2%.
- Five-run verifier microbenchmark median: 369.9 µs/op, 17,299 B/op, 100 allocations/op on Linux/amd64, AMD EPYC 9V74. This is verifier cost only.

No runtime integration, external benchmark reproduction, independent replication, production trust-root qualification or SOTA claim is made.

See [#11845](https://github.com/Luminous-Dynamics/symthaea/issues/11845) for the public research scope and fault matrix.
