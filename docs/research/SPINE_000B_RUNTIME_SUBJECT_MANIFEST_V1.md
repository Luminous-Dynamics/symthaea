# SPINE-000B-M1 Runtime Subject Manifest v1

Status: **FORMAT FROZEN / RUNTIME USE PENDING**

Authority: `measurement-only`

This contract defines the canonical subject identity that must be committed before a qualified SPINE runtime observation lineage begins. It does not establish observer non-interference, application correctness, causal load, benefit, or action/epistemic authority.

## Domain

Canonical bytes are prefixed by:

`symthaea.spine.000b.subject-manifest.v1\0`

All integers are unsigned little-endian. Strings are strict UTF-8/ASCII where specified and encoded as `u16 byte_len || bytes`. SHA-256 values are exactly 32 raw bytes decoded from lowercase 64-hex transport strings.

## Canonical fields

In order:

1. repository identity string (`Luminous-Dynamics/symthaea` for this repository);
2. Git HEAD as lowercase 40-hex ASCII;
3. Git tree as lowercase 40-hex ASCII;
4. `clean_worktree_required` as `u8` (`1` for qualified campaigns);
5. runtime profile ID string;
6. target triple string;
7. observed `rustc --version` string;
8. observed `cargo --version` string;
9. `default_features` as `u8`;
10. sorted unique Cargo feature strings;
11. sorted unique bound-file entries `(repo-relative canonical path, sha256)`;
12. sorted unique protocol-version IDs;
13. workload ID string;
14. workload/materialized-input digest;
15. sorted unique named seeds `(name, u64)`;
16. cycle start `u64`;
17. cycle count `u64`;
18. stopping-rule ID string;
19. manager observer capacity `u16`;
20. application observer capacity `u16`;
21. guard observer capacity `u16`;
22. qualification/campaign-policy digest.

Lists are encoded as `u16 count` followed by entries in canonical sort order.

## Canonical path law

Bound file paths must be repository-relative POSIX paths. Reject:

- absolute paths;
- empty paths;
- `.` or `..` segments;
- backslashes;
- duplicate paths;
- non-canonical repeated separators.

The path itself is committed so the same bytes at a different semantic location are a different subject.

## Required bound surfaces

A qualified runtime manifest must include, when present in the selected lineage:

- `Cargo.toml`;
- `Cargo.lock`;
- `rust-toolchain.toml`;
- `flake.lock`;
- `src/cognitive_loop/subsystem_trait.rs`;
- `src/cognitive_loop/cycle_phase_dynamics/mod.rs`;
- `src/cognitive_loop/cycle_phase_output/mod.rs`;
- `src/cognitive_loop/helpers/feedback_helpers.rs`;
- every live SPINE runtime-observer implementation module;
- every SPINE interpretation contract/registry/oracle used by the campaign.

The manifest generator may bind additional files but may not omit required applicable files.

## Feature law

The exact Cargo feature set and whether default features are enabled are canonical. Feature names are sorted and unique. Adding, removing, renaming, or changing default-feature policy starts a new subject identity.

## Environment law

The repository pins Rust through `rust-toolchain.toml`; the qualified manifest binds both that file hash and the observed `rustc`/`cargo` version strings. The committed Nix environment is bound through `flake.lock` when the campaign claims Nix-qualified execution.

Machine-local hostnames, PIDs, absolute worktree paths, wall-clock time, CPU serials, and timing telemetry are excluded from canonical subject identity.

## Workload and stopping law

The manifest must bind either a materialized workload digest or the digest of a frozen generator/policy artifact represented by the selected `workload_digest`. Random/genesis seeds are named, sorted, unique, and committed.

`cycle_start`, `cycle_count`, and `stopping_rule_id` are canonical. A result outside those bounds cannot be attached to the same qualified observation lineage.

## Observer capacity law

All fixed observer capacities are part of subject identity. A capacity change starts a new lineage even if no overflow occurred.

## Clean subject law

Qualified runtime evidence requires a clean worktree after manifest materialization and before genesis. Any source/environment drift after evidence begins starts a new observation lineage. Never mix observation roots across subject-manifest digests.

## Genesis binding

C2 Observation Commitment v2 may create an observation-chain genesis root only from the exact qualified M1 manifest digest. The manifest digest is not inferred from Git HEAD alone.

## Required controls

- JSON formatting/key order changes -> identical canonical bytes;
- source-file input order changes -> identical canonical bytes after sorting;
- feature/protocol/seed input order changes -> identical canonical bytes after sorting;
- bound source hash mutation -> digest changes;
- source path mutation -> digest changes;
- feature addition/removal -> digest changes;
- seed/workload/cycle-bound/stopping-rule mutation -> digest changes;
- observer-capacity mutation -> digest changes;
- toolchain/lockfile hash or observed tool version mutation -> digest changes;
- duplicate path/feature/protocol/seed -> reject;
- absolute/noncanonical path -> reject;
- dirty subject -> campaign qualification reject;
- Python and independent Rust implementations reproduce frozen byte vectors exactly before runtime use.

## Claim boundary

M1 establishes only deterministic runtime-subject identity. It does not establish that an observer is non-perturbing, that receipts are complete, that guard predicates are true, that an application executed, that a state change was caused by a subsystem, or that any subsystem is beneficial/load-bearing.