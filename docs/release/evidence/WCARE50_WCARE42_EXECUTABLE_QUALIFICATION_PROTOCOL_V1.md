# WCARE-50 — frozen WCARE-42 executable qualification protocol v1

Status: `PREREGISTERED_WHILE_EXACT_WCARE49_RUN_QUEUED`
Authority: `MeasurementOnly`
Tracks: #3052

## Purpose

Qualify the exact frozen WCARE-42 builder-attestation verifier and its unchanged CLI qualifier after the exact standalone dependency lock has reached WCARE-49 bounded execution-input closure.

WCARE-50 qualifies verifier execution. It does not authenticate any production builder.

## Exact upstream subject

- WCARE-42 source head: `4bb84790bab3dc6a6d2e30d0ef081950a3954717`
- exact lock-bearing FINAL: `5bc23735f545b1b82044820f0e02ece59be04b4a`
- FINAL parent: `52d1d9fb741250ab8bcab205113689a8cc9431bb`
- manifest blob: `5410040e5616241dd4ba581af8f297675d083830`
- verifier source blob: `1c300a455f054d118e55556aac81b623824629bc`
- golden-test blob: `666242f74fb302f9be2b62fa4b3050f3ee9ffefd`
- verifier protocol blob: `1cd6e3f5f2f4edcf4fccf45b35c4c68bdb9c12a5`
- result-schema blob: `b2d6b4b61af46c8924cdb95b2f958df3d3d7ab96`
- independent golden-vector blob: `dc42f404537795f37a9bf178d3f30fd52c74a355`
- unchanged qualifier blob: `8a80d1a74e409503a72b4607425cc61d8072ca2e`
- lock blob: `1a9b126e3d5062bd3be5290bfa5930d189760ceb`
- lock SHA-256: `7288b7dd64b533a4e08ae9a66ff120fb69d5885effd80b0e3b7f51455028d976`
- Rust/Cargo: `1.96.0`

## Exact WCARE-49 precondition

V1 consumes only:

- WCARE-49 head `720c7ebf6c03b5698c48a4d2315ea9c4d25107e7`
- exact hosted run `34877774593`
- run number `7`
- run attempt `1`
- exact WCARE-49 input-closure guard blob `8170c65551eba6fa8d1356229a78667e6c634f4a`

The theorem and fixture matrix were frozen while that exact WCARE-49 run was still queued. A retry, replacement run, changed WCARE-49 head or changed guard blob is outside v1.

A green run is insufficient. WCARE-50 must retrieve exact run/job/log evidence and bind the canonical WCARE-49 JSON whose bytes match `WCARE49_RESULT_SHA256`.

## Frozen independent fixture package

The synthetic Ed25519 package was produced independently of the Rust verifier using Python `cryptography` and frozen before any WCARE-49 result was consumed.

- compressed archive SHA-256: `81ea12f18312bcd8542cdc8e3066fe85d1f32f329c06e06e6870ac2f304820d9`
- decompressed package SHA-256: `c6b66a25b5a04649306d81a4275019885edddd84d90bd4510a522441c3a8a9d9`
- synthetic public key: `d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a`
- case count: `11`

Exact case order:

1. `accepted_provenance`
2. `accepted_relation`
3. `accepted_pre_result`
4. `signature_valid_issuer_untrusted`
5. `invalid_signature`
6. `altered_subject_replay`
7. `altered_policy_binding`
8. `subject_scope_unauthorized`
9. `unauthorized_strength`
10. `expired_attestation`
11. `revoked_issuer`

Every materialized envelope/policy/plan/result/subject file has its own frozen SHA-256 inside the package. Materialization must reproduce those bytes exactly.

The key material and every fixture are synthetic qualification data. An accepted fixture cannot be reinterpreted as evidence about a real builder, organization or person.

## Input closure inheritance

WCARE-50 imports the exact WCARE-49 guard by frozen Git blob and applies its pre/post rules to the exact FINAL checkout:

- clean tracked, ordinary-untracked and ignored-untracked state;
- Cargo hierarchical config census through filesystem root;
- fresh external Cargo home with no Cargo config;
- fresh external Cargo target directory;
- semantic compiler/Cargo environment override rejection;
- exact Rust/Cargo 1.96.0;
- exact source/toolchain/lock identities before and after execution.

WCARE-50 does not weaken or locally fork those rules.

## Execution theorem

Before fixture execution, the campaign must run:

- `cargo test --locked`
- `cargo test --locked --test golden`

Then every frozen case is executed through unchanged `scripts/wcare42-qualify.sh` with its exact six arguments.

For every case the campaign requires:

- exact qualifier exit code;
- exactly one WCARE-42 verifier JSON result;
- exact expected disposition;
- exact `result_bound_by_signature` state;
- exact ten-check trust/signature vector;
- exact hashes of all five supplied input artifacts;
- `builder_authentication_established = false`;
- `preregistration_temporal_precedence_established = false`;
- `subject_correctness_established = false`;
- `runtime_authority_granted = false`.

The only fixtures expected to set `attestation_accepted = true` are:

- `accepted_provenance`
- `accepted_relation`
- `accepted_pre_result`

The accepted pre-result case must keep `result_bound_by_signature = false`.

## Success theorem

Only the complete exact matrix may emit:

`WCARE42_EXECUTABLE_QUALIFIED`

with:

- `lock_admitted = true`
- `execution_inputs_closed = true`
- `cryptographic_execution_qualified = true`
- `wcare42_executable_qualification_established = true`
- `builder_authentication_established = false`
- `preregistration_temporal_precedence_established = false`
- `subject_correctness_established = false`
- `full_machine_hermeticity_established = false`
- `trusted_hardware_established = false`
- `runtime_authority_granted = false`

## Boundary

`WCARE42_EXECUTABLE_QUALIFIED != production builder authenticated != WCARE-41 complete authentication coverage != external temporal preregistration != Stage-B aggregate promotion`.

No consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority, subject correctness, objective moral truth or solved alignment is established.