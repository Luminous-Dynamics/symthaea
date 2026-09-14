# GEOM-003D0A5 — Independent run-seal commitment oracle

Status: independent qualification candidate for D0A3 commitment semantics. No target-system lesion result was used to define or derive these vectors.

Issue: #3194  
Exact predecessor: `47c73abe00e93b87a69ddfcba7e4c66c2c0b2a47`

## Why this tranche exists

D0A3's Rust unit tests can establish that its producer and verifier agree with each other. They cannot, by themselves, rule out a shared encoding mistake.

D0A5 therefore freezes expected values from a separate implementation lineage:

- Rust side: repository `blake3` crate + D0A3 public APIs;
- independent side: dependency-free pure Python implementation of unkeyed BLAKE3 and the GEOM commitment encoding.

The Python oracle does not import, shell out to, or copy output from the Rust implementation under qualification.

## BLAKE3 self-check

Before computing any GEOM value, the Python implementation must reproduce these public BLAKE3 digests:

- empty input: `af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262`
- `abc`: `6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85`

Failure stops the oracle before any GEOM vector is considered.

## Frozen vectors

The fixture `docs/research/vectors/GEOM_RUN_SEAL_V1.json` was frozen before hosted Rust qualification with these independent outputs:

| Vector | Expected commitment |
|---|---|
| Process environment | `5b3117d45ec98f55103b1cb1a7cd6c629b0d73212521218ee03a62c58f26e767` |
| Canonical JSON | `6bae876adb98a10119d7787637ac0666a9d160d919e381bf4debd9da0f819190` |
| Complete run environment | `b4bae3ffee9cfdacde6d6a9ae3b18e2f7ac918643b58e239950dee06d57a4f94` |
| Evidence inventory | `2b6f7f110b7b877e448659c40ba555325b416758925a8498d8b457c9746859ef` |

The evidence fixture additionally freezes:

- `a.txt` → `ac678d92b3d739773d18cd952cfcea443fa4a5a98ffc9554b66795bb22d5532d`
- `nested/b.bin` → `10f847936eb4f56573613478660da66b5871069884957535f8ed979cecb88ea4`

The process-environment input is intentionally stored in reverse lexical order. The canonical-JSON fixture is intentionally written in non-canonical key order. The run snapshot intentionally contains duplicate/unsorted Cargo features. These choices make the vectors exercise normalization rather than merely hash already-normalized input.

## Qualification surfaces

The Rust integration test may use only these public A3 APIs:

- `process_environment_commitment_from_entries`
- `canonical_json_commitment`
- `seal_run_environment`
- `inventory_evidence_directory`

It must not call D0A3 private helpers or duplicate the private commitment writer in Rust.

The evidence test materializes the fixture files in an isolated temporary directory and requires both individual BLAKE3 file digests and the final inventory commitment to match the independently frozen values.

## Claim boundary

A PASS supports this statement only:

> The D0A3 Rust implementation and an independently implemented Python oracle agree exactly on the frozen GEOM run-seal v1 vectors.

It does not prove collision resistance, sign evidence, establish environment capture correctness, or validate any GWT/consciousness/gravity claim.
