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

The library exposes three staged calls:

1. `verify_member_head_response(...)` authenticates one member's response under the exact pinned trust-store bytes and checks the expected incarnation and caller-supplied 32-byte Base64URL challenge nonce.
2. `verify_write_checkpoint(...)` requires the configured write quorum, validates checkpoint identity and canonical counters/digests, and computes the domain-separated checkpoint digest.
3. `aggregate_witness_read(...)` requires the read quorum of distinct active member keys, matches every response to a supplied write-certified checkpoint, rejects same-sequence certified forks, permits lagging heads and selects the highest certified sequence.

The caller must generate and retain the challenge nonce using a cryptographically secure RNG; this crate validates nonce shape and binding but does not generate challenges. A head response from one member does not establish a commit. Every selected head must be backed by a write-quorum certificate.

Example protocol pseudocode:

```rust
let nonce = secure_random_32_bytes_base64url();
let responses = collect_member_responses(&nonce).await?;
let verified_responses = responses.iter()
    .map(|raw| verify_member_head_response(
        raw, &trust_bytes, &trust_pin, &incarnation_id, &nonce
    ))
    .collect::<Result<Vec<_>, _>>()?;
let certificates = collect_referenced_checkpoint_envelopes(&verified_responses).await?;
let verified_checkpoints = certificates.iter()
    .map(|raw| verify_write_checkpoint(
        raw, &trust_bytes, &trust_pin, &incarnation_id
    ))
    .collect::<Result<Vec<_>, _>>()?;
let selected = aggregate_witness_read(
    &verified_responses, &verified_checkpoints, &trust_bytes,
    &trust_pin, &incarnation_id, &nonce
)?;
```

The pseudocode intentionally leaves network transport and nonce generation outside this verification crate.

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
- Three-of-four challenge-bound read with one lagging replica selects sequence five.
- Two-of-four read is anti_rollback_unknown.
- Repeating one member response does not increase read quorum.
- Two different write-quorum-certified checkpoint digests at the same sequence are rejected as equivocation.
- A response with a replayed/wrong nonce is rejected.
- A head response without its write-quorum certificate is anti_rollback_unknown.
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
