# Connectome target inventory fate verifier

Rust CLI verifier for the bidirectional target-fate contract.

## Usage

```sh
cargo test --locked
cargo run --locked -- fate-inventory.json target-world-manifest.json correspondence-receipt.json
```

The third input must be the exact correspondence receipt bytes, after its own correspondence verifier has checked the receipt structure. This tool independently reads its target-world identity and mapping table. The current synthetic fixture uses a minimal correspondence envelope containing the target-world identity and mapping rows.

## Structural checks

- exact SHA-256 pins over target-manifest and correspondence-receipt raw bytes;
- target world and release identity across all inputs;
- unique target manifest IDs and unique fate IDs;
- exact target inventory coverage, with no missing or unknown targets;
- per-object type agreement;
- MAPPED links to a non-ambiguous mapping row that contains the target and evidence IDs;
- ALTERNATIVE_CANDIDATE links to the exact ambiguity group and candidate row;
- alternative groups contain at least two distinct target sets with shared relation type/source IDs;
- TARGET_ONLY_UNRESOLVED and OUT_OF_SCOPE objects do not occur in mapping rows;
- OUT_OF_SCOPE requires a policy digest and reason;
- REJECTED references a mapping row whose support status is REJECTED;
- status and per-object-type counters are recomputed and compared.

## Trust boundary

The report is `COUNTERS_RECOMPUTED` with `trusted_promotion=false`. It does not authenticate referenced evidence bytes, edit transactions, biological identity, biological novelty, adjudicator identity, or signer trust roots. A target-only unresolved object is not evidence of a new biological object.

The synthetic correspondence envelope is an input fixture, not a replacement for the full upstream receipt schema or its verifier.

## Tests

Nineteen Rust unit tests are authored for known-answer, exact-byte digest, inventory closure, uniqueness, type agreement, mapping membership, ambiguous alternatives, target-only protection, policy requirements, rejection linkage, counter mismatch and world identity.

They have not yet been compiled or executed in the current authoring environment because Rust toolchain binaries are unavailable. No compile, test, Clippy, schema-validation or CI pass is claimed.

## Fixture digests

The target manifest fixture raw-byte SHA-256 is `d2fdcdd223956d2d7882b7e89e12e3c566cf97a0d4e2bdea046d1e4360079299`. The correspondence envelope fixture raw-byte SHA-256 is `7a0ac82ae2c6fa062d1d9680c3d408e8b80e72fb4983e42008af62a240f0fe2f`. Both are pinned in the known-answer fate receipt.
