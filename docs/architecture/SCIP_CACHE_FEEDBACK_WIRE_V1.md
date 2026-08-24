# SCIP Cache Feedback Wire v1

Status: **experimental / draft wire contract**

Crate: `crates/bridges/symthaea-interlingua`

This note freezes the canonical binary representation used to bind SCIP semantic-cache feedback into authenticated transport transcripts.

JSON serialization of `SemanticCacheFeedback` remains useful for diagnostics, logging, and human inspection. It is not the canonical transport/transcript representation because independent JSON implementations need not share object-key ordering or serializer details.

## Frame

Every v1 cache-feedback frame is exactly **38 bytes**:

| Offset | Length | Field |
|---:|---:|---|
| 0 | 3 | ASCII magic `SCF` |
| 3 | 1 | wire version: `0x01` |
| 4 | 1 | feedback kind |
| 5 | 1 | requirement |
| 6 | 32 | raw semantic content digest |

There is no variable-length field and no padding.

## Feedback kind

| Value | Meaning |
|---:|---|
| `0x01` | ACK |
| `0x02` | MISS |
| `0x03` | REVOKE |

All other kind values are invalid in v1.

## Requirement

| Value | Meaning |
|---:|---|
| `0x00` | none |
| `0x01` | semantic-reference target missing |
| `0x02` | exact graph-delta base missing |

Legal kind/requirement pairs are:

- ACK / none
- MISS / semantic-reference target
- MISS / graph-delta base
- REVOKE / none

ACK or REVOKE with a non-zero requirement is invalid. MISS with `none` or any unknown requirement is invalid.

## Semantic digest

SCIP semantic hashes are exposed by the API as 64 hexadecimal characters representing a 32-byte content digest. The binary frame carries the **raw 32 digest bytes**, not the ASCII hexadecimal form.

Consequences:

- the wire frame is 32 bytes smaller than carrying the textual hash;
- uppercase and lowercase textual spellings of the same legacy-accepted hash map to the same frame;
- decoding always reconstructs the canonical lowercase hexadecimal string;
- arbitrary changes to the 32 digest bytes remain syntactically valid content addresses and must be detected by the authenticated transport transcript and/or subsequent semantic resolution, not by the cache-frame parser itself.

The frame therefore canonicalizes representation; it does not authenticate content.

## Decoder rules

A conforming decoder MUST reject:

- any length other than 38 bytes;
- magic other than `SCF`;
- a version other than `0x01`;
- unknown feedback kinds;
- illegal kind/requirement combinations.

A decoder MUST NOT infer a missing, future, or malformed field. Unknown versions fail closed so a future frame revision cannot be interpreted accidentally under v1 semantics.

## Golden vectors

The repository integration test `crates/bridges/symthaea-interlingua/tests/cache_feedback_wire_vectors.rs` freezes complete frame bytes for ACK, both MISS forms, and REVOKE.

For example, an ACK for semantic hash `aa...aa` (64 lowercase `a` characters, representing 32 bytes of `0xaa`) is:

`534346010100aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`

A semantic-reference-target MISS for hash `bb...bb` is:

`534346010201bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`

These vectors are intended to be copied into independent Rust, Python, C++, or Xenia implementations as interoperability ratchets.

## Trust boundary

`wire_bytes` and `from_wire_bytes` provide canonical encoding and structural validation only.

They do **not** establish:

- peer identity;
- session membership;
- freshness or ordering;
- replay protection;
- authorization;
- transport confidentiality or integrity.

For Symthaea/Mycelix, Xenia remains the intended owner of those properties. A Xenia binding should authenticate the exact 38-byte frame together with peer/session identity and transcript position before SCIP applies the decoded feedback to `PeerSemanticInventory`.

## Failure and recovery

A valid authenticated MISS or REVOKE changes only the local possession claim for the named semantic digest. It does not mutate semantic graph content.

After applying the feedback, the sender rebuilds its ordinary `TransferPlanningInput` and reruns `plan_transfer`. The existing exact semantic path then chooses among reference, graph delta, or complete grounded graph according to the remaining acknowledged prerequisites and actual byte economics.

No special recovery planner is introduced by this wire format.

## Versioning

The frame carries an explicit one-byte version. Any incompatible future layout must use a new version and new golden vectors.

v1 reserves no extension bytes. This is intentional: silent extension parsing would weaken canonical transcript identity. A larger future frame is a new version rather than an optional suffix.
