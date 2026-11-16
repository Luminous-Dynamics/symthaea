# Claim Plan IR preflight prototype

Standalone research-only Rust workspace for **structural and semantic preflight** of Claim Plan IR v1.

## Run

From the repository root:

```sh
cargo test --manifest-path research-tools/claim-plan-ir/validator/Cargo.toml
cargo run --manifest-path research-tools/claim-plan-ir/validator/Cargo.toml -- research-tools/claim-plan-ir/fixtures/positive_zar_balance_plan_v1.json
```

The crate is intentionally outside Symthaea's default Cargo workspace. Direct dependencies are exact-pinned. A Cargo.lock is not included yet because the authoring environment has no Rust/Cargo toolchain; the transitive dependency graph is therefore **not yet frozen**.

## Admission pipeline

1. Bound raw input size.
2. Parse into a duplicate-aware JSON representation. Duplicate decoded member names reject before ordinary map parsing.
3. Reject floating-point and non-safe-integer numeric tokens before canonicalization.
4. Validate against the checked-in JSON Schema 2020-12 with format assertions enabled.
5. Run semantic checks the schema cannot express: unique field IDs and pointers, RFC 6901 `~0`/`~1` escape validity, consistent parser/schema profile across same-response fields, typed predicate references, freshness field type, status specification/maturity pairing, canonical lowercase ASCII LDH HTTPS DNS origins (no IDN/IP literals), numeric port range/default-port rules, DNS server identity/origin continuity, origin-relative path shape, positive request lifetime, and source-mode/trust-witness/verification-scope compatibility.
6. Canonicalize with RFC 8785 JCS and compute the versioned SHA-256 plan digest.

## Claim ceiling

A preflight pass means only that the plan is structurally admissible under this prototype and its semantic checks. Output deliberately keeps source authenticity, proof validity, issuer qualification, currentness and authority unevaluated. No TLS verifier, notary signature, credential, proof backend, status-list network retrieval, LISE adapter, nullifier ledger or Xenia action boundary is implemented.

The 18 authored tests additionally pin RFC 8785 UTF-16 property-key ordering with a BMP/astral pair, trust-scope anti-overclaim, port-range/default-port constraints, and DNS server/origin identity matching. The implementation is research code. It is **not compiled or tested in the authoring environment** because rustc/Cargo are unavailable. Authored tests are not passed tests.

## Trust and transferability

Every source profile now carries a separate `trust` block with a digest-addressed witness profile, witness kind/identity, evidence artifact kinds, post-session verification scope, plaintext-visibility profile and collusion-assumption profile. Semantic preflight requires the source mode's minimum witness/scope pairing. For example, a delegated TLSNotary attestation cannot be relabeled as a public cryptographic proof, and a Scribe-style STARK still carries the attestation-node provenance dependency.

This is deliberately conservative. A preflight pass does not authenticate witness identities or profile digests; their owners must resolve and qualify those dependencies. The trust block prevents internal mode/scope contradictions, not malicious profile owners or compromised trust roots.
