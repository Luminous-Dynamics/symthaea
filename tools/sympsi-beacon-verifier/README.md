# SYMPsi offline Quicknet beacon verifier

**Status:** isolated prototype / offline known-answer verifier. The verifier has no network or clock access. It checks a supplied Quicknet response against a pinned chain profile and verifies that the returned randomness equals the randomness derived from the verified signature.

It does **not** establish freshness, that the beacon was published at the scheduled time, that a response commitment existed before the beacon, that the relay view was independent, or that the target was hidden from the model.

## Pinned profile

| Field | Pinned value |
|---|---|
| Beacon ID | `quicknet` |
| Chain hash | `52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971` |
| Scheme | `bls-unchained-g1-rfc9380` |
| Group public key | 96-byte BLS12-381 G2 public key, embedded in `src/lib.rs` |
| Genesis seed | `f477d5c89f21a17c863a7f937c6a6d15859414d2be09cd448d4279af331c5d3e` |
| Genesis timestamp | 1692803367 Unix seconds |
| Period | 3 seconds |
| Test vector | Quicknet round 1000 |

The profile pins scheme ID separately because drand's chain hash does not include the scheme ID. The crate is pinned to `drandite-core = 0.1.0`; this is a prototype dependency, not a statement that the dependency has received an independent cryptographic audit. Dependency license, source provenance, maintenance and cryptographic implementation review remain P0 items in SCI-PSI-003.

## Reproduce

From repository root:

    cargo fmt --manifest-path tools/sympsi-beacon-verifier/Cargo.toml --all -- --check
    cargo test --manifest-path tools/sympsi-beacon-verifier/Cargo.toml
    cargo clippy --manifest-path tools/sympsi-beacon-verifier/Cargo.toml --all-targets -- -D warnings

    cargo run --manifest-path tools/sympsi-beacon-verifier/Cargo.toml --       --requested-round 1000       --input tools/sympsi-beacon-verifier/fixtures/quicknet-round-1000.json       --out /tmp/sympsi-quicknet-round-1000-verification.json

The output directory must not already contain the receipt. Input JSON is strict: only `round`, `randomness`, and `signature` fields are accepted for this unchained profile.

## Receipt predicates

On success:
- `signature_valid_for_pinned_profile_and_round = true`;
- `published_randomness_matches_signature = true`;
- `freshness_verified = false`;
- `response_chronology_verified = false`.

The receipt claim ceiling is `PINNED_QUICKNET_SIGNATURE_AND_ROUND_ONLY_NOT_FRESHNESS_OR_CHRONOLOGY`.

A successful BLS signature is not proof that the relay returned the right *time* or that a model response was locked before the target became available. Those need independent witness/clock evidence and a public audit trail.

## Known-answer source

The round-1000 signature, randomness, public key, genesis seed, chain hash, scheme and verifier construction are taken from the published `drandite-core 0.1.0` Quicknet verification example:
https://docs.rs/drandite-core/latest/drandite_core/

The drand developer documentation lists Quicknet's chain hash and scheme, and its public endpoints:
https://drand.love/developer

The verifier is offline. It does not fetch these sources dynamically and does not treat a relay's HTTPS response as authenticated.

## Scope exclusions

- No HTTP client or relay polling.
- No timestamp authority verification.
- No target derivation or mapping.
- No response commitment service or model isolation.
- No trial ledger or real experiment.
- No claim about psi / anomalous information acquisition.

The next integration gate is an independently reviewed dependency and trust-root audit, followed by timestamp/witness chronology verification, then a byte-reproducible end-to-end trial verifier.
