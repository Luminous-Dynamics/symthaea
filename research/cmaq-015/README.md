# CMAQ-015: Signed benchmark receipts and claim-relative evidence lineage

**Status:** protocol artifacts v0.1; no runtime verifier or qualification claim  
**Tracking issue:** [#9374](https://github.com/Luminous-Dynamics/symthaea/issues/9374)  
**Base snapshot:** `c82ea1be710eb9a4c3d3de133ea0326257a6e524`  
**Claim ceiling:** schema and test vectors only

## Why this exists

CMAQ-010 deliberately treats `provenance_group` as authenticated simulator metadata. That made its common-mode/Sybil/replay experiments small and deterministic. It is not a real security boundary.

CMAQ-015 defines a signed receipt that binds the evidence subject to the benchmark, environment, model, policy, action candidates, selected action, evaluator, scorer, source lineage and freshness context. It also explicitly refuses to turn signatures into independence or truth.

The five claims stay separate:

| Claim | What establishes it | What it does not establish |
|---|---|---|
| Byte integrity | Recomputed digest matches exact subject bytes | Who produced those bytes |
| Cryptographic authenticity | Signature verifies over domain-separated canonical payload | Signer authorization or truth |
| Purpose authorization | Trusted key lifecycle policy permits this key for this receipt purpose | Source independence |
| Freshness / replay status | Monotonic state plus consumed challenge or durable replay ledger | Historical truth |
| Claim-relative independence | Authenticated lineage registry + reviewed failure-domain profile | Scientific validity |
| Empirical validity | Independent environment-side outcome evidence and qualification procedure | Generalization outside tested scope |

A signed, validly authorized Byzantine source is still Byzantine. A valid signature over a false observation is still a false observation.

## Wire format

The envelope is closed by `receipt.schema.json`:

```json
{
  "schema_version": "cmaq-015-signed-receipt-v0.1",
  "payload": { "...": "..." },
  "signature": {
    "encoding": "base64",
    "value": "..."
  }
}
```

The payload carries a closed schema version and includes:

- benchmark/task/run/split identity;
- exact source revision, model, policy, candidate-set, observation, action, environment, manifest, evaluator and scorer digests;
- evidence class and exact subject digest;
- opaque source commitment, registry entry/snapshot digests, independence-profile digest and parent receipt digests;
- source epoch/sequence, nonce and historical/live mode;
- dedicated key purpose `CmaqBenchmarkEvidenceReceipt`;
- budget-contract, resource-counter, raw-episode, result-row and scoring-artifact digests.

The signature is detached from the payload. The signing input is:

```text
UTF8("luminousdynamics.symthaea.cmaq-015.receipt.v0.1") || 0x00 || JCS(payload)
```

The algorithm and key ID are inside the signed payload. The signature object itself is not included in the signed bytes.

### Canonicalization requirements

The protocol selects RFC 8785 JSON Canonicalization Scheme (JCS) for the v0.1 wire profile. This schema cannot enforce canonical bytes, duplicate-member rejection, Unicode/parser constraints, array ordering, signature correctness, or trusted lineage resolution. Those are verifier obligations.

The checked-in positive-control vector uses ASCII strings, integer numeric values, sorted object keys, no floating-point values, and a deterministic Ed25519 test key. Its `canonical_payload_utf8_base64` is the exact byte string a verifier must reconstruct. The vector is a narrow signature/canonicalization regression, not a complete JCS conformance suite.

The source revision field is a 40-character Git SHA; all other content identities are lower-case 64-character SHA-256 hex strings. Do not insert unhashed mutable URLs in place of these identities.

## Source identity and independence

The producer does not get to declare its own independence. Resolve `source_commitment_sha256`, signer/key identity and lineage edges against a separately governed, authenticated registry snapshot.

The graph is claim-relative. Two sources may be independent with respect to one fault class and share a different common cause (for example, distinct sensors with one calibration standard or clock). The independence profile therefore identifies the fault domains relevant to a particular claim.

Parent arrays must be unique and sorted by bytewise lower-case digest order. The verifier must resolve every parent, detect cycles and missing parents, and reject omitted required ancestors according to the registry/profile contract. A valid JSON Schema result is not proof of any of those properties.

Six signing keys mapped to one source commitment are one source. Distinct keys, organizations, processes or toolchains do not automatically establish independent failure domains.

## Freshness and replay

`observed_at_unix_ms` and `nonce_hex` are signed claims, not proof of trusted time or single use.

- Historical mode is suitable for archived, immutable evidence. It cannot assert current signer authority or fresh execution.
- Live mode requires a non-null digest of a separately verifiable challenge-consumption receipt. The verifier must check that the challenge was issued for the exact expected context and consumed once.
- The verifier must persist an anti-rollback checkpoint for trust and lineage-registry sequences. A process-local sequence tracker does not survive restart or prevent verifier forks.
- Exact historical signature validity and current signer authorization after revocation are reported separately.

## Required verification report

The companion `verification-report.schema.json` separates cryptographic/authentication, signer authorization, trust freshness, replay, lineage completeness, claim-relative independence, evaluator integrity, empirical validity and decision qualification. It binds the report to the exact envelope and policy/checkpoint digests.

Do not serialize a positive-looking report and deserialize it into a trusted type. A positive verifier type must be minted only by re-running the verifier over the receipt and its authenticated dependencies.

## Integration plan

1. Freeze the payload, canonicalization, schema and cross-language vectors.
2. Review the dependency direction. Reuse the trust lifecycle and signer/verifier provider patterns in `symthaea-fabrication-kernel` where public API boundaries permit; do not reuse `KeyUsage::FabricationManifest`.
3. Introduce a dedicated CMAQ receipt key purpose in a separately reviewed common trust change. Avoid a parallel key store or crypto implementation.
4. Implement strict parsing, signature verification, trust-policy evaluation, challenge consumption and persisted checkpoint verification as independently testable layers.
5. Add the hostile corpus in `fixtures/hostile-cases.json` to the exact verifier entry point.
6. Adapt CMAQ-010 evidence reports and CMAQ-011 run manifests into receipts. Keep source-lineage and evaluator-integrity evidence separate.
7. Only then move from synthetic fixture receipts to real Gymnasium/MuJoCo and DMC run artifacts.

## Hostile controls and claim ceilings

The hostile case catalog lists expected dispositions. A test passes only when the exact verifier reports the expected independent axes. A signature-valid case may still be unauthorized, replayed, dependent, evaluator-contaminated or scientifically unsupported.

The Ed25519 fixture is signed by a publicly disclosed **test-only** key. Its public key is included so independent implementations can verify the signature. Never use this key or its key ID outside the fixture. The fixture's signature-positive result does not mean the key is trusted by a real trust snapshot.

No runtime verifier, challenge service, durable replay ledger, authenticated lineage registry, benchmark adapter, physical safety claim or SOTA result is implemented by these protocol artifacts.
