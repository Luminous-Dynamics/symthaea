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
- `verify_run_receipt_input_bindings` recomputes catalogue, split and ground-truth manifest hashes over exact bytes; checks that the split refers to the exact imported dataset/version; verifies every payload file; and compares the payload digest set with the receipt. The ground-truth manifest's semantics and source independence are a separate qualification task (#10826).
- A pending rights record may remain in the discovery catalogue, but cannot authorize binary ingestion.
- Event and recording groups may not cross train/validation/test partitions. Receiver, campaign and location dimensions are additionally held out when declared by an experiment.
- Pairwise source relations are `shared_failure_domain`, `unknown`, or `no_declared_common_domain`. All profiles must share one immutable snapshot ID and one valid RFC 3339 `as_of` timestamp. The last relation means only that both declared inventories are complete and no dependency identifier overlaps. It is not proof of physical independence.
- A run receipt binds exact code, model, configuration, dataset, split, toolchain and environment identities. Each case binds its exact output-artifact SHA-256. The independent verifier identity must differ from the declared executor identity, and verifier/report artifacts are digest-bound. This identity separation is a structural check, not proof of organizational independence.
- Design fixtures contain no measurements. Only separate run receipts carry observed outcomes.
- A structurally valid receipt is not a detector-performance pass, operational safety case, certification, or data-rights opinion.

## Run receipt and ground-truth binding

Every run receipt requires a SHA-256 digest for the exact ground-truth/adjudication manifest bytes. `verify_run_receipt_input_bindings` recomputes the catalogue, split and ground-truth manifest hashes; binds the split to the exact imported dataset ID/version; verifies every supplied payload file against its catalogue digest; and checks exact payload-digest-set equality with the receipt. A matching digest pins the manifest bytes, but does not prove that labels are correct or independent. Semantic ground-truth/oracle independence is tracked separately in Symthaea #10826.

The common-mode dependency graph also binds every source profile to one immutable snapshot ID and RFC 3339 `as_of` time. Unknown dependency inventories remain unknown; `no_declared_common_domain` is not a proof of physical independence.

## Open-set RF evaluation profile

`docs/research/emsr-v0.1/emsr-open-set-rf-profile-v0.1.json` freezes the passive open-set RF task boundary, RF-RPL comparator metadata, rights-pending dataset references, event/recording split policy, known/unknown/indeterminate decisions, metrics and ESP32/RF-front-end claim separation. It is a design profile; the comparator has not been reproduced and no dataset payload is imported.

## Machine-readable contracts

`docs/research/emsr-v0.1/emsr-evidence-contracts-v0.1.schema.json` contains Draft 2020-12 definitions for the dataset catalogue, split manifest, run receipt, source failure-domain profile and pairwise-independence graph. Cross-record group leakage, unique identifiers, exact intended-use matching, exact-byte digest checks and executor/verifier identity separation are additionally enforced by the Rust validator.

## Frozen examples

- `docs/research/emsr-v0.1/emsr-dataset-catalog-v0.1.json` is a discovery-only inventory. All four payloads remain not imported; rights review remains pending.
- `docs/research/emsr-v0.1/emsr-split-manifest-example-v0.1.json` is synthetic-only and illustrates separate event, recording, receiver, campaign and location partitions.

Tracking: Symthaea EMSR-002 (#10814), Mycelix common-mode evidence (#5904).
