# Symthaea Music Theory Patch Series 18 Plan

Date: 2026-07-21

## Objective

Series 18 makes publication-catalog continuity portable and independently
auditable. Series 17 could prove that one catalog file was internally valid and
could issue a status proof at that exact head. It could not efficiently answer:

- Does this later catalog preserve the complete earlier history?
- Has a mirror observed rollback or incompatible same-height heads?
- Which externally governed witnesses authenticated this exact head?
- Can a third party carry the current head, one exact predecessor proof,
  witness evidence, mirror observations, and status proofs in one artifact?

The series adds those contracts without modifying the Series-17 catalog,
publication-record, event, delegation, or status-proof schemas.

## Patch groups

### 1. Exact catalog checkpoints

Add `CalibrationPublicationCatalogCheckpoint` with:

- catalog and authority identities;
- catalog version and SHA-256;
- record and event counts;
- head-event SHA-256;
- optional predecessor checkpoint;
- logical issuance epoch;
- canonical checkpoint SHA-256.

Add checkpoint-bound publication status proofs so a status classification is
explicitly tied to one checkpoint rather than merely copied beside it.

### 2. Prefix-consistency proofs

Add a transparent proof verified against both complete catalog snapshots. It
requires byte-for-byte preservation of the earlier record and event prefixes,
exact suffix identities, monotonic checkpoint epochs, and direct checkpoint
lineage.

This is intentionally not a compact Merkle proof. Simplicity and reviewability
are preferred at this layer.

### 3. Mirror rollback and equivocation evidence

Add an append-only mirror-observation ledger that detects:

- per-mirror height rollback;
- same-height conflicting catalog states;
- multiple distinct children of one predecessor checkpoint;
- observation-epoch regression;
- checkpoint and ledger tampering.

Conflict findings remain persistable evidence. The ledger distinguishes
structural integrity from an incident-free state.

### 4. External witness quorum

Add canonical checkpoint-witness payloads, externally signed envelopes,
explicit accepted-key policies, threshold evaluation, and caller-supplied
verification.

The crate does not select algorithms, manage keys, enroll witnesses, or prove
witness independence.

### 5. Portable catalog-head bundle

Add a self-auditing bundle containing:

- current catalog and checkpoint;
- optional predecessor catalog, checkpoint, and consistency proof;
- witness policy and signed statements;
- optional mirror ledger that must have observed the packaged head;
- checkpoint-bound publication status proofs;
- mandatory machine-readable trust limitations.

The authenticated builder fails closed unless the external witness threshold is
met.

### 6. Persistence governance

Advance the schema registry to v9 and append all Series-18 roles after the
Series-17 roles. Existing `#[repr(u16)]` role ordinals are frozen by regression
test so older numeric identities are not renumbered.

### 7. Operator tools

Add:

- `evidence_publication_checkpoint`
- `evidence_publication_mirror`
- `evidence_publication_witness`
- `evidence_publication_head_bundle`

The witness verifier is executed directly without a shell and receives the
canonical payload and signature request through standard input.

## Trust model

Series 18 distinguishes these claims:

1. **Catalog validity** — one complete catalog passes its internal audit.
2. **Checkpoint identity** — one checkpoint binds that exact catalog state.
3. **Prefix continuity** — one later snapshot exactly extends one earlier
   snapshot.
4. **Observed mirror behavior** — configured mirrors have or have not reported
   rollback, equivocation, or a fork.
5. **Witness authentication** — a caller-selected threshold of accepted key IDs
   was authenticated by an external verifier.

No single layer is treated as a substitute for the others.

## Deliberate limitations

The series does not establish:

- wall-clock freshness;
- universal catalog availability;
- global consensus or global non-equivocation;
- legal publisher authority;
- operational independence of witness keys;
- authentication of mirror identities;
- absence of a fork withheld from every configured observer.

## Landing order

The mail series is ordered so each conceptual boundary can be reviewed and
landed separately:

1. checkpoint and status anchoring;
2. consistency proof;
3. mirror ledger;
4. witness quorum;
5. portable head bundle;
6. explicit crate exports and schema registry;
7. operator tooling;
8. semantic hardening and adversarial tests;
9. release documentation.

## Required canonical verification

Run in the project development shell:

```text
cargo fmt --all -- --check
cargo check --all-targets
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

Then exercise at least one complete external-verifier workflow:

```text
cargo run --example evidence_publication_checkpoint -- --help
cargo run --example evidence_publication_mirror -- --help
cargo run --example evidence_publication_witness -- --help
cargo run --example evidence_publication_head_bundle -- --help
```

The patch-building environment did not contain Cargo, rustc, rustfmt, Clippy, or
Nix. Static verification therefore cannot replace the canonical build and test
run before merge.
