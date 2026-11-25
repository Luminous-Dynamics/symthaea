# MACE MON-005 outer-record adapter v0.1

**Status:** source-matched Rust codec and consumer verifier authored. Eight Rust tests are authored but have not been compiled or executed. The record/leaf reference vector has been independently recalculated in Python against the published MON-005 v1 algorithm. No append API, live witness call, checkpoint authentication, or external anchoring is implemented here.

## Exact MON-005 record contract

The adapter was written against the repository's actual MON-005 v2/v3 reference source, not a guessed DTO. Its record builder uses:

- `schema = "sfeb-evidence-record-v1"`
- `tenantId`, `sequence`, `previousRecordHash`, and `payload`
- canonical body bytes over exactly those five fields (the `recordHash` output field is excluded)
- `recordHash = SHA-256("SFEB-RECORD-V1" || 0x00 || JCS(body))`
- tenant genesis predecessor `SHA-256("SFEB-GENESIS-V1" || 0x00 || UTF8(tenantId))`
- RFC 9162 leaf `SHA-256(0x00 || decode_hex(recordHash))`

The payload uses the repository's existing `serde_json_canonicalizer` dependency and preserves the MON-005 source's camelCase envelope. The embedded MACE payload retains its own snake_case schema and independent domain-separated digest.

## Payload shape

The outer MON-005 `payload` is:

- `kind = "mace-evaluation-anchor"`
- `schema = "mace-mon005-evaluation-anchor-v1"`
- `macePayload`: the complete `MaceEvaluationAnchorPayloadV1` object
- `macePayloadSha256`: the independently pinned MACE pre-anchor digest

Retaining the full object is necessary: an outer record containing only the MACE digest would not let an independent consumer retrieve and re-verify the exact MACE payload. The outer record's `recordHash` and Merkle leaf remain distinct from the MACE payload digest.

## Rust API

- `mon005_genesis_record_hash_v1(tenant_id)` computes the exact MON-005 genesis predecessor.
- `prepare_mon005_mace_anchor_record_v1(...)` accepts a verified MACE payload capability and an append cursor. It emits the canonical record/body bytes, record digest and RFC 9162 leaf digest. It does not prove the supplied cursor is trusted.
- `parse_and_verify_mon005_mace_anchor_record_v1(...)` checks the retrieved record against consumer-pinned tenant, sequence, predecessor and MACE digest; rejects unknown/duplicate fields; recomputes the outer record digest; re-verifies the inner MACE payload; and returns a private verified capability plus the RFC 9162 leaf.
- The returned leaf can be passed to the existing `verify_rfc9162_inclusion_v1`.

A consumer must still authenticate the MON-005 checkpoint and witness roster/quorum, verify the inclusion proof under that authenticated root, and verify consistency from a separately pinned prior checkpoint. A valid record hash is not proof of publication; a valid inclusion path is not proof that the root is trusted or current.

## Independent reference vector

For the existing deterministic T9/T10 MACE two-episode fixture, tenant `mace-test`, sequence 1:

- MACE pre-anchor digest: `117579a8803ecedd6b5eee4efbc73c4fe66ce9088c7687a5b8785ac4815d1154`
- MON-005 genesis predecessor: `646ed2e98580c76d575fc3b8fdd20bcc754a7828a6588fc2f64d1ebd62739228`
- MON-005 outer record digest: `075a9638686c559c547e1c1cbeb77671c833eeef5e52cf9daa36ad967d72d26c`
- RFC 9162 leaf digest: `46a32360e04e669cadb1b917781405c751d5cbbfedd70c2cdb1e37b6e5e1ca8c`

The 1,499-byte inner MACE JCS payload and its digest match the vector already pinned in #13664. The outer record was independently reconstructed with Python's sorted compact JSON serializer (ASCII-only fields, so key ordering matches the JavaScript source), SHA-256 and exact MON-005 domain prefixes. The Rust test pins both outer digests.

## Tests authored (not executed)

1. Exact MON-005 record and RFC 9162 leaf vectors.
2. Consumer re-verification of outer record and embedded MACE payload.
3. Compatibility with the existing RFC 9162 verifier on a one-leaf tree.
4. Wrong tenant, sequence and predecessor rejection.
5. Tenant-specific genesis requirement.
6. Outer digest rejection after nested payload mutation.
7. Wrong external MACE digest rejection.
8. Duplicate and unknown outer JSON field rejection.

## Remaining boundary

This is the outer-record codec seam only. No client/network transport, log append, returned detached witness receipt, checkpoint signature/quorum authentication, proof retrieval, pinned checkpoint store, split-view gossip, or distributed witness deployment is present. MON-005 remains authoritative for append sequencing, record persistence, checkpoints and witnesses. The MACE caller must use a cursor from an independently verified MON-005 head and must not label the prepared record externally anchored until the entire verification chain succeeds.
