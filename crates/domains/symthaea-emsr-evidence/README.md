# symthaea-emsr-evidence

Fail-closed evidence contracts for the defensive Electromagnetic-Spectrum and PNT Resilience (EMSR) programme.

## Scope

This crate validates:
- versioned dataset catalogues and source-level rights state;
- exact-byte SHA-256 digests;
- leakage-resistant dataset split manifests;
- pairwise common-mode/failure-domain relationships between observation sources;
- separate, provenance-bearing run receipts with independent-verifier evidence.

It does **not** implement an RF detector, infer hostile intent, prove physical truth, authorize an operational action, or certify safety.

## Run locally

From the Symthaea workspace root:

```bash
cargo test -p symthaea-emsr-evidence
cargo clippy -p symthaea-emsr-evidence --all-targets -- -D warnings
```

The crate is an explicit workspace package under `crates/domains/*`; it is not currently in the workspace's default-members list. Use the package-scoped commands above or include it explicitly in the relevant CI matrix.

## Important semantics

- Hash exact original bytes. Do not parse and reserialize a payload before hashing it.
- An imported payload requires a reviewed rights record permitting the exact intended use and per-file SHA-256 digests.
- A pending rights record may remain in the discovery catalogue, but cannot authorize binary ingestion.
- Event and recording groups may not cross train/validation/test partitions. Receiver, campaign and location dimensions are additionally held out when declared by an experiment.
- Pairwise source relations are `shared_failure_domain`, `unknown`, or `no_declared_common_domain`. The last means only that both declared inventories are complete and no dependency identifier overlaps. It is not proof of physical independence.
- A run receipt binds exact code, model, configuration, dataset, split, toolchain and environment identities. Each case binds its exact output-artifact SHA-256. The independent verifier identity must differ from the declared executor identity, and verifier/report artifacts are digest-bound. This identity separation is a structural check, not proof of organizational independence.
- Design fixtures contain no measurements. Only separate run receipts carry observed outcomes.
- A structurally valid receipt is not a detector-performance pass, operational safety case, certification, or data-rights opinion.

## Machine-readable contracts

`docs/research/emsr-v0.1/emsr-evidence-contracts-v0.1.schema.json` contains Draft 2020-12 definitions for the dataset catalogue, split manifest, run receipt, source failure-domain profile and pairwise-independence graph. Cross-record group leakage, unique identifiers, exact intended-use matching, exact-byte digest checks and executor/verifier identity separation are additionally enforced by the Rust validator.

## Frozen examples

- `docs/research/emsr-v0.1/emsr-dataset-catalog-v0.1.json` is a discovery-only inventory. All four payloads remain not imported; rights review remains pending.
- `docs/research/emsr-v0.1/emsr-split-manifest-example-v0.1.json` is synthetic-only and illustrates separate event, recording, receiver, campaign and location partitions.

Tracking: Symthaea EMSR-002 (#10814), Mycelix common-mode evidence (#5904).
