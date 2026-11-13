# MACE Effect-Join v1 — executable reference contract

## Verification inputs

The verifier receives four independent inputs:

1. Signed evidence bundle: authorization, runtime invocation, gateway dispatch, optional effect-origin receipt and signed witness.
2. Operator-pinned trust store: expected issuer, role, key ID, public key and declared independence domain for each signing role.
3. Policy state: verifier time, maximum permit lifetime, allowed clock skew, witness maximum age/log/sequence and minimum revocation generation.
4. Intent expectation: task, execution, principal, target, action digest, argument digest, capability and policy digest from the task harness or separately verified mandate.

Trust and intent are never derived from the untrusted evidence bundle.

## Roles and causal edges

Authorization → Invocation → Dispatch → Effect → Witness.

Each receipt has a unique ID, schema version, issuer/key identity, task/execution/principal context and exact parent receipt references. The invocation binds the authorization receipt and permit. Dispatch binds both receipt IDs, the actual runtime invocation ID, the same permit and action bindings plus one idempotency key. Effect binds the exact dispatch receipt, idempotency key, task/principal/execution and terminal outcome. Witness binds the canonical digest of all signed receipts, log ID, sequence and checkpoint time.

All exact payload fields are re-compared at every edge; timestamps are not used to infer parentage.

## Fail-closed rules

Reject:
- duplicate IDs, wrong role/key or invalid signature;
- broken parent edges or cross-task/principal/execution joins;
- changed target, action, arguments, capability or policy digest;
- permit expiry, invalid lifetime or stale revocation generation;
- dispatch attempt other than one;
- wrong witness log, digest mismatch, stale checkpoint or sequence rollback relative to an external pin.

Missing terminal evidence yields incomplete/unknown, never completed. A contradictory gateway rejection plus effect receipt rejects. No effect is inferred from agent prose, tool span, HTTP status or absence of error.

## Canonicalization and trust caveats

V0.1 signs compact JSON serialization of fixed Go structs. This is deterministic for this implementation but not RFC 8785 JCS and not cross-language canonicalization. Production needs a frozen canonical encoding plus independent implementations and test vectors.

Declared independence domains are operator-supplied. Different strings do not establish different administrative control. The witness is a signed digest/time/sequence statement, not a Merkle inclusion/consistency proof or full transparency log.

A signed runtime workload ID and implementation digest remain observer claims. This verifier does not establish that every kernel/process path was observed or that the claimed implementation executed. The effect-origin signature authenticates the configured source, not the truth of the physical world.

Global one-shot replay prevention across multiple bundles, concurrent workers or restarts remains owned by the durable authority/consume-once ledger (#9490). The verifier checks only the one dispatch represented inside this bundle.

## MACE integration

A successful join is one mandatory evidence input to the MACE task × control classifier; it does not itself establish task success or full harness compliance. Missing observer/effect/witness evidence must produce a non-clean control result. Task-oracle uncertainty remains separate.
