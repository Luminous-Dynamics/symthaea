# LQCD provider-independent execution receipt oracle

Issue: #3491  
Authority: independent standard-library synthetic oracle  
Scientific authority: none  
Rust execution authority: none

## Purpose

Freeze v1 provider-independent execution-receipt validation, semantic-equivalence
classification, and authority separation while preserving the existing hosted
`RustQualified` boundary.

The oracle does **not** execute Cargo, rustc, Clippy, lattice kernels, or any real
beta6 subject.

## Frozen subject

- schema: `symthaea.lqcd.portable-execution-receipt.v1`
- executed subject SHA-256: `9c46ca8489036b659798437fbf8f738141daed7626bd6da41fbfcb0723466fb6`
- canonical stdout SHA-256: `ed993d05e15d4bc67fc813502d5943f50a29e7517158a586fd157d35dd785f57`

## Executed semantics

The synthetic oracle establishes that:

- a `PortableExecutionCandidate` cannot satisfy the Rust-qualified predicate;
- a synthetic hosted receipt satisfies that engineering predicate only when the
  provider is hosted and all declared gates pass;
- non-authoritative provider-instance changes alter the full receipt identity
  but preserve execution-semantic identity;
- the same-semantics fixture classifies as
  `ExecutionSemanticsEquivalent`;
- authoritative mutations classify as `ToolchainShift`, `RecipeShift`,
  `DependencyShift`, `TargetShift`, `NumericalProfileShift`, `SubjectShift`,
  or `EnvironmentEquivalenceUnknown` as appropriate;
- eight negative controls fail closed: dirty pre-state, post-execution subject
  mutation, unlocked dependency resolution, missing rustc identity, portable
  self-promotion, pre-subject mismatch, invalid gate status, and duplicate gate.

## Frozen identities

- portable receipt SHA-256: `3ce32bcdfb31e80a1396ffafa83c20a4837648c70a68d1072a7222fa332d2562`
- portable semantic SHA-256: `80e24c3684832ec7fbe86b2c000f762d45851ffc04d0a3906bd1f9b39cccc477`
- same-semantics second receipt SHA-256: `1956c8f043c9db060f935f1cbcfe89133e6d27a0efe5063168fd21d49bbee747`

The two full receipt identities differ while the semantic identity remains
stable by construction.

## Negative claim theorem

This evidence MUST NOT be used to claim:

- real Rust execution;
- `RustQualified`;
- a completed #3434 base comparison;
- authorization to repair the current Clippy findings;
- beta6 pilot/final campaign admission;
- EHK agreement;
- any lattice-QCD numerical or physics result.

The result explicitly records:

- `real_rust_execution_performed=false`
- `real_beta6_campaign_authorized=false`
- `portable_can_satisfy_rust_qualified=false`

## Next step

Wrap an exact immutable Rust subject with a portable receipt producer, preferably
under a Nix-backed environment, and validate the resulting receipt independently.
Keep that execution evidence below hosted authority until a policy-authorized
hosted receipt exists for comparison.
