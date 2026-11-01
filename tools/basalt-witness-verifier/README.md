# BASALT witness verifier

**Status:** isolated cryptographic verifier prototype. This is not a deployed witness service and does not establish physical anti-rollback.

## Purpose

The crate verifies DSSE envelopes under independently provisioned trust material. It is deliberately independent of SQLite/PostgreSQL/Turso/etcd and has no database credentials or signer secrets in its runtime.

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
- verifies a pinned canonical genesis trust root and sequential successor roots;
- requires each successor root payload to meet both the previous-root threshold and the successor-root threshold;
- enforces exact version increments, predecessor digest continuity, stable authority identity and immutable initial witness-set commitment;
- validates witness-set metadata with the root version named by the signed payload, including BFT quorum arithmetic, member key IDs, and administrative-domain uniqueness;
- keeps the verified root chain and previous verified witness-set state opaque so callers cannot inject substitute predecessor keys or metadata.

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

The trust-store shape is described in `benchmarks/basalt/schema/witness-verifier-trust-store.v1.schema.json`. The root payload and witness-set metadata shapes are described in `benchmarks/basalt/schema/witness-trust-root-payload.v1.schema.json` and `witness-set-metadata-payload.v1.schema.json`. The library independently performs key-identity, quorum and failure-domain checks; JSON Schema validation alone is not sufficient.

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

These are authored tests. They are not described as passing until the exact-head Rust workflow reports success.

## Claim ceiling / what is not verified

A `pass` means only that a canonical payload was authenticated by the configured threshold of Ed25519 keys in the exact locally pinned trust-store bytes; the library verifier loads that trust store from the bytes itself, so callers cannot substitute a separately constructed key set. It does **not** establish:
- that the trust-store pin was provisioned securely (the caller must supply that assurance);
- that trust-root signatures, dual-threshold root rotation, revocation history or witness-set metadata chains are valid;
- that the signers are independent in actual operations (the verifier checks only unique declared administrative-domain labels);
- that the witness service durably persists before signing;
- that a committed checkpoint is fresh, latest or immune to replay/freeze;
- that a challenge nonce was unpredictable or answered by a live quorum;
- that a witness quorum has not been compromised beyond the declared threshold;
- that any database snapshot is fresh or reconstructible;
- that power-loss durability, replication or backend qualification passed.

The next implementation slice must verify the offline trust-root and witness-set metadata chain, then implement challenge-bound read quorum/head selection and checkpoint predecessor/sequence validation. The protocol is documented in `docs/research/BASALT_INDEPENDENT_WITNESS_PROTOCOL_2026-11-01.md`.
