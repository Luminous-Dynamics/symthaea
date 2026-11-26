# Bootstrap Codec Profile — v0.1

**Programme:** PHYSICAL-STORAGE-BOOTSTRAP-CODEC-001  
**Parent:** [Symthaea #13774](https://github.com/Luminous-Dynamics/symthaea/issues/13774)  
**Implementation branch:** `research/physical-storage-bootstrap-survivability-001`

## 1. Purpose and evidence class

This profile closes the first executable gap identified by Bootstrap Survivability v0.1: the four in-media control streams previously had a generator matrix but no canonical byte serialization, stream length, padding, or reference erasure decoder.

This is a synthetic reference implementation and finite KAT corpus. It does not assert the matrix/layout is already wired into the optical layout engine, does not qualify physical media, and does not establish a standard or production format.

## 2. Code profile

- Field: GF(2^8), primitive polynomial 0x11d.
- Data/parity: systematic [6,4,3], rate 4/6.
- Generator rows, in exact shard order:

```text
D0  [1,0,0,0]
D1  [0,1,0,0]
D2  [0,0,1,0]
D3  [0,0,0,1]
P0  [1,1,1,1]
P1  [1,2,3,4]
```

Any four rows are linearly independent over the declared field; the independent oracle inverted all 15 four-row subsets. This means any two **known shard erasures** are recoverable in the finite algebraic model.

The reference decoder selects four surviving rows, in ascending shard index, inverts their 4×4 matrix, reconstructs the four systematic data shards, regenerates all six shards, and cross-checks every additional survivor.

It is not an unknown-error decoder. If exactly four shards survive, arbitrary corruption among those four may be indistinguishable from a different valid codeword. With extra survivors, inconsistent redundant shards are rejected. The caller must still check external object digests.

## 3. Canonical manifest encoding

All integers are big-endian. No implicit separators, native-endian integers, or map iteration order are permitted.

```text
Manifest :=
    "BSM1"              // 4-byte magic
    u16 format_version  // currently 1
    u32 record_count
    Record[record_count]

Record :=
    u16 object_id_length
    u8  semantic_type
    u16 version_length
    u64 content_length
    byte[32] expected_content_sha256
    byte[object_id_length] object_id_utf8
    byte[version_length] version_utf8
    byte[content_length] content
```

Records are sorted by unsigned bytewise UTF-8 object ID. IDs and versions are non-empty; duplicate IDs are invalid. The record decoder rejects truncation, invalid UTF-8, noncanonical order, unsupported versions, and trailing bytes. Count-based allocation is bounded by the actual input length before allocation.

The SHA-256 field is supplied by an independent manifest. The reference codec does not calculate or verify the digest. A caller must recompute SHA-256 over each reconstructed content and compare it to an independently trusted expected digest; authenticity and provenance are separate checks.

## 4. Four-way bundle framing and padding

The canonical manifest is framed as:

```text
BundleFrame :=
    "BSP1"             // 4-byte magic
    u64 manifest_length
    byte[manifest_length] canonical_manifest
    zero_bytes[0..3]   // minimum zeros to reach a multiple of four
```

The framed byte stream is split into four equal contiguous data shards D0..D3. P0/P1 are generated from those data shards. The exact manifest length is inside the recoverable frame, so an absent outer index does not make zero padding ambiguous. Decoding rejects nonzero padding.

This profile's logical shard index maps to the synthetic placement convention in the parent fixture only. It does not prove physical independence or spatially safe placement.

## 5. Known-answer and negative-control corpus

`fixtures/bootstrap_codec_kat_v0_1.json` contains:
- fixed parity bytes for the short four-stream vector;
- a canonical manifest and complete framed bundle with six exact shard byte strings;
- all 15 two-shard erasure combinations;
- a three-erasure rejection;
- an inconsistent-survivor rejection;
- a nonzero-padding rejection.

The independent finite-field oracle reproduced the parity and bundle bytes, inverted all 15 possible four-row submatrices, and recovered the original data for all 15 two-erasure patterns. These calculations are independent of the Rust implementation.

Rust unit tests encode the same fixed vectors and cases. They are authored but not executed in this environment because Cargo, rustc and rustfmt are unavailable. Therefore compile/test status is **pending**, not pass.

## 6. Acceptance conditions

Before calling this implementation qualified:
1. Run `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test` at the exact PR head.
2. Validate the fixture against its JSON Schema 2020-12 schema.
3. Compare the Rust output byte-for-byte with the frozen KAT fixture.
4. Independently verify every decoded object's SHA-256, signature/provenance and semantic relations.
5. Exercise every two-erasure pair and negative controls in a second implementation.
6. Verify the matrix, shard order, and format ID are embedded in the Recovery Capsule and in-media bootstrap, without either being allowed to self-authorize.
7. Keep reader/calibration qualification and physical placement trials separate from this software codec result.

## 7. Evidence ceiling

Current evidence is: authored reference source, authored Rust tests, and independent arithmetic/vector calculation. It is not evidence of Rust compilation, CI execution, hardware/reader behavior, physical control placement, retention, cryptographic authenticity, or production readiness.
