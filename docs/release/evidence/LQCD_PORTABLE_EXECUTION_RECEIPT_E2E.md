# LQCD portable execution receipt producer/verifier E2E

Issue: #3491  
Authority: synthetic process/mechanics evidence only  
Rust execution authority: none  
Scientific authority: none

## Purpose

Split portable execution receipt construction from independent receipt
verification and exercise both programs in a temporary clean Git repository.

No Symthaea Rust code, Cargo build, rustc compilation, Clippy analysis, lattice
kernel, or beta6 subject was executed. Deterministic fake `rustc`/`cargo`
identity commands were placed earlier on PATH solely to test identity capture.

## Frozen subjects

- producer SHA-256:
  `5d481a5b29ab7096d175fada60291b9af7be4edd2a985be2b7a099fd3473d1ca`
- independent verifier SHA-256:
  `90dde0bb068cb8384710751f177638163411908ac09530a2eb17595cd359a6a0`
- canonical E2E stdout SHA-256:
  `8f37353e0a0a5e9ca4492f22f9ab3ced37d5e74c7579d763bf60e4965838c81e`

## Positive fixture

The synthetic repository was initialized, committed, and queried for exact HEAD
and tree identities. The producer required those exact identities, required a
clean worktree, hashed `Cargo.lock`, `rust-toolchain.toml`, and `Cargo.toml`,
captured synthetic tool identities, executed a harmless `/bin/sh` command, then
rechecked HEAD/tree/worktree immutability before writing the receipt.

The independent verifier accepted the produced receipt.

Frozen positive identities:

- receipt SHA-256:
  `5cb271690f4fbcd8d398688d29fcee44e488c89922271fd53169e037572e7852`
- execution-semantic SHA-256:
  `ea931b95cc85439fcd5078931be5acdf5e7e6a65ca036ff7d4c30e73e98059b6`
- producer exit: `0`
- verifier exit: `0`
- `portable_can_satisfy_rust_qualified=false`

## Negative fixtures

Three process-level negative controls were executed:

1. mutating the receipt authority to `HostedExactHeadQualified` makes the
   independent verifier reject with `INVALID:authority`;
2. an untracked file before execution makes the producer fail before producing
   a receipt;
3. a command that mutates a tracked file makes the producer fail after
   execution and before producing a receipt.

Thus a portable producer cannot silently bless a dirty or mutated subject, and
the verifier does not accept portable self-promotion.

## Negative claim theorem

This tranche establishes only producer/verifier mechanics in a synthetic Git
repository. It explicitly records:

- `real_rust_execution_performed=false`
- `real_beta6_campaign_authorized=false`

It does not establish a real #3434 base execution, a real #3436 candidate
execution, `RustQualified`, whole-workspace CI, EHK reproduction, or any
lattice-QCD numerical/physics result.

## Next step

Run the producer around one exact immutable LQCD Rust subject under a real
Nix/toolchain environment. Keep that receipt at `PortableExecutionCandidate`
authority until an independently verified hosted exact-head receipt exists.
