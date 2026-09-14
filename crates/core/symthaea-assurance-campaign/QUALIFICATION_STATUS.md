# ASSURE-002 qualification and claim boundary

This note records the executable state and the narrow theorem currently established by the draft ASSURE-002 campaign kernel. It is intentionally stricter than legacy identifier names in the source.

## Current exact-subject history

Historical head `0f4396592574e0e72d261bf956c88d197707f5f2` reached the permanent Rust 1.96 qualifier and failed first at `cargo fmt --check`. No compile/test claim was made from that run.

The read-only qualifier exposed the required root `Cargo.lock` campaign-package stanza. A temporary narrowly constrained repair job committed only that reviewed lock transformation. A separate read-only formatter-capture job reproduced the exact Rust 1.96 formatter output for:

- `src/lib.rs`;
- `tests/assure002.rs`.

Its write-authority repair job then verified the original Git blob identities, reproduced the exact reviewed output SHA-256 values, refused unrelated tracked changes, and committed only those two formatted files.

Those repair jobs are process evidence, not qualification. A fresh ordinary repository commit starts a new exact-head read-only qualification. Historical failures remain failures.

## Temporal provenance boundary

The current implementation contains names such as:

```text
evidence_production_statement_digest
ProducedBeforeOrAtRegistration
registration < production[i] < admission[i]
```

Until ASSURE-002C (#2877) performs the source/API rename and optional production-witness integration, these names must be interpreted under the **narrower commitment-ordering theorem**:

```text
exact evidence-artifact commitment
was durably ordered after the current registration
```

This does **not** establish:

```text
the underlying evidence bytes were first created after registration
```

A pre-existing artifact can be committed to an ordering service after registration. Therefore a normal ordering receipt over an evidence digest proves post-registration commitment ordering, not first production time.

The intended base relation is:

```text
registration < commitment[i] < admission[i]
```

with the authoritative ledger still requiring:

```text
registration < admission[1] < admission[2] < ...
```

This preserves the useful preregistration/admission theorem without laundering a later digest commitment into a stronger production claim.

## Stronger production theorem

`ProductionWitnessedAfterRegistration` is a separate, optional stronger theorem tracked by ASSURE-002C. It requires causal execution evidence binding the exact current registration + execution request + real execution + exact output/evidence commitment, with authority outside the candidate worker. Ordinary digest ordering cannot satisfy that stronger class.

Existing Symthaea secure-worker parent recomputation and privilege-separated trusted-verifier patterns should be reused rather than creating a parallel provenance universe.

## Semantic-definition boundary

The campaign-local semantic type remains provisional and binds semantic ID + definition digest only. ASSURE-002A is being requalified independently as the reusable `symthaea-assurance-semantics` shared-core primitive. ASSURE-002B (#2874) will make that shared schema-bearing commitment authoritative inside campaign semantics before final public campaign golden vectors are frozen.

## Current status

ASSURE-002 remains **draft and unqualified** until a fresh exact head passes, at minimum:

```text
exact checkout
Rust 1.96 format
cargo check
focused campaign tests
strict Clippy -D warnings
committed root-lock parity
tracked-checkout immutability
```

No historical repair or queued/action-required workflow is substituted for those gates.
