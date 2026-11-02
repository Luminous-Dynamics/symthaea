# BASALT witness verifier

**Status:** isolated cryptographic verifier prototype. This is not a deployed witness service and does not establish physical anti-rollback.

## Purpose

The crate verifies a DSSE envelope under a separately pinned trust store. It is deliberately independent of SQLite/PostgreSQL/Turso/etcd and has no database credentials or signer secrets in its runtime.

It currently:
- parses DSSE envelopes with duplicate-field and unknown-field rejection;
- validates the local trust store against an exact SHA-256 pin supplied by the operator/provisioning channel;
- validates Ed25519 public-key byte lengths and canonical Base64;
- derives each key ID as SHA-256 of the RFC 8785 JCS bytes for `{"keytype":"ed25519","public":"<base64>" }`;
- rejects duplicate member IDs, key IDs, public keys and active administrative failure domains;
- checks BFT quorum arithmetic (`n = 3f + 1`, `read_quorum = write_quorum = 2f + 1`);
- excludes revoked members;
- verifies DSSE signatures over PAE with threshold counting over distinct trusted key identities;
- checks payload-type binding;
- returns the exact authenticated payload bytes from the DSSE verification result;
- requires those bytes to be byte-for-byte RFC 8785 canonical JSON;
- reports exact envelope, payload and trust-store digests;
- authenticates nonce-bound head responses from individual members;
- verifies each referenced checkpoint DSSE envelope against the write quorum and recomputes its domain-separated checkpoint digest;
- accepts lagging replicas, rejects same-sequence certified forks, and selects the highest certified sequence among the distinct read-quorum members.

The verifier uses the `dsse` crate for DSSE PAE, Ed25519 signing/verification and distinct-key threshold verification; `serde_json_canonicalizer` for RFC 8785 JCS; and `ed25519-dalek` for public-key identity fixtures. See [DSSE crate documentation](https://docs.rs/dsse/latest/dsse/), [serde_json_canonicalizer](https://docs.rs/serde_json_canonicalizer/latest/serde_json_canonicalizer/), and [ed25519-dalek](https://docs.rs/ed25519-dalek/latest/ed25519_dalek/).

## Run

From the repository root:

```sh
cargo fmt --manifest-path tools/basalt-witness-verifier/Cargo.toml -- --check
cargo test --locked --manifest-path tools/basalt-witness-verifier/Cargo.toml
cargo clippy --locked --manifest-path tools/basalt-witness-verifier/Cargo.toml --all-targets -- -D warnings
```

The standalone Cargo lockfile is committed. All validation commands use `--locked`.

CLI:

```sh
cargo run --manifest-path tools/basalt-witness-verifier/Cargo.toml -- \
  --envelope checkpoint.dsse.json \
  --trust-store provisioned-witness-set.json \
  --trust-store-sha256 sha256:<64-lowercase-hex> \
  --payload-type application/vnd.luminous.basalt.witness-checkpoint.v1+json \
  --quorum write \
  --out witness-verification-report.json
```

The `--trust-store-sha256` digest must come from an independent provisioning/release trust channel. Do not calculate the expected digest from the same untrusted trust-store file and pass it straight back to the verifier.

The trust-store shape is described in `benchmarks/basalt/schema/witness-verifier-trust-store.v1.schema.json`. The library independently performs key-identity, quorum and failure-domain checks; JSON Schema validation alone is not sufficient.

## Challenge-bound read API

The public entry point is `verify_head_read_quorum`. It accepts a batch of `WitnessHeadEvidence` values, each containing one member-signed head-response envelope and the corresponding write-quorum-certified checkpoint envelope. It:

- checks the caller-generated 32-byte unpadded Base64URL nonce and expected authority incarnation/epoch;
- authenticates each head response with `QuorumKind::Member` (one active trusted Ed25519 key);
- authenticates each checkpoint with `QuorumKind::Write` (the configured write threshold);
- validates response/checkpoint identity, sequence, revision, state digest and policy digest binding;
- recomputes the checkpoint digest over the exact canonical payload bytes;
- counts distinct member keys, ignores exact duplicate reports from the same member, and rejects a member reporting conflicting heads;
- permits lagging replicas, then selects the highest write-certified sequence;
- rejects different write-certified digests at the same sequence.

The caller must generate the challenge nonce using a cryptographically secure RNG; this crate validates nonce shape and binding but does not generate it. A single member response never establishes a commit: the corresponding checkpoint envelope must meet the write quorum.

Example:

```rust
let selected = verify_head_read_quorum(
    &evidence,                 // Vec<WitnessHeadEvidence>
    &trust_store_bytes,
    &out_of_band_trust_pin,
    &fresh_nonce_base64url,
    &expected_authority_incarnation_id,
    &expected_authority_epoch,
);
match selected.status.as_str() {
    "fresh_head_selected" => { /* compare full state manifest with restored store */ }
    "anti_rollback_unknown" => { /* fail closed: no freshness verdict */ }
    "rejected_witness_equivocation" => { /* halt writes and investigate */ }
    _ => { /* preserve typed rejection; never fall back to local revision */ }
}
```

The function currently verifies the supplied checkpoint certificates and detects same-sequence certified forks, but it does not walk predecessor links to pinned genesis. Network transport, nonce generation and persist-before-sign service behavior are outside this crate.

## Tests authored

- DSSE PAE matches a separately checked-in 144-byte vector.
- Valid 3-of-4 signatures pass.
- 2-of-4 fails.
- One signature copied three times fails a 3-of-4 threshold.
- Wrong payload type fails.
- Payload modification fails.
- Signed but noncanonical JSON fails.
- Duplicate envelope properties fail.
- Duplicate JSON object keys in the signed payload fail canonical-byte equality.
- A revoked signer cannot contribute to quorum.
- Duplicate public-key identity under a second member ID fails.
- Duplicate administrative failure domain fails.
- Incorrect out-of-band trust-store pin fails.
- Three-of-four challenge-bound read with one lagging replica selects the highest certified sequence.
- Two-of-four read is anti_rollback_unknown.
- Repeating one member response does not increase read quorum.
- One member reporting two different heads for the same challenge is rejected.
- A checkpoint envelope below write quorum is rejected.
- A response paired with a different checkpoint is rejected.
- Two different write-quorum-certified checkpoint digests at the same sequence are rejected as equivocation.
- A response with a replayed/wrong nonce is rejected.
- A head response without an adequate write-quorum certificate is rejected.
- Malformed or padded Base64URL challenge nonce is rejected.

These are authored tests. They are not described as passing until the exact-head Rust workflow reports success.

## Claim ceiling / what is not verified

A `pass` means only that a canonical payload was authenticated by the configured threshold of Ed25519 keys in the exact locally pinned trust-store bytes; the library verifier loads that trust store from the bytes itself, so callers cannot substitute a separately constructed key set. It does **not** establish:
- that the trust-store pin was provisioned securely (the caller must supply that assurance);
- that trust-root signatures, dual-threshold root rotation, revocation history or witness-set metadata chains are valid;
- that the signers are independent in actual operations (the verifier checks only unique declared administrative-domain labels);
- that the witness service durably persists before signing;
- that the checkpoint predecessor chain is continuous back to pinned genesis (only supplied checkpoint certificates and their sequence/digest identities are checked);
- that the supplied challenge nonce was generated unpredictably or that members durably read their current heads before responding;
- that a witness quorum has not been compromised beyond the declared threshold;
- that any database snapshot is fresh or reconstructible;
- that power-loss durability, replication or backend qualification passed.

The remaining protocol work includes verifying the offline trust-root and witness-set metadata chain, verifying predecessor-chain continuity back to pinned genesis, validating durable witness persist-before-sign behavior under crashes, and comparing the selected complete authority-state manifest with a restored database projection. The nonce generator/network transport are caller responsibilities. The protocol is documented in `docs/research/BASALT_INDEPENDENT_WITNESS_PROTOCOL_2026-11-01.md`.
