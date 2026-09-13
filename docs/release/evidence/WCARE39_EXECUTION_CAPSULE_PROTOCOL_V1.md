# WCARE-39 — Execution capsule and environment-lineage protocol v1

Status: `PREREGISTERED_PROTOCOL`
Authority: `MeasurementOnly`
Protocol version: `wcare39-execution-capsule-v1`

## Purpose

WCARE-33 through WCARE-38 increasingly bind exact source, evidence, review, and authentication subjects. WCARE-39 binds those claims to the environment that actually executes them.

The governing theorem is:

`execution evidence = exact subject + exact command plan + exact environment capsule + observed process result`

A source-qualified algorithm executed under an unknown or drifting environment is not the same evidence lineage.

## Separate environment integrity from subject outcome

WCARE-39 never equates a qualified execution environment with a successful subject.

A subject may fail under a perfectly qualified environment. Such a failure is valid scientific evidence.

WCARE-39 therefore records two orthogonal dimensions:

- `environment_integrity`: `QUALIFIED`, `DRIFTED`, `INDETERMINATE`, or `INVALID`;
- `subject_outcome`: `PASS`, `FAIL`, `INVALID`, `INDETERMINATE`, or `NOT_RUN`.

`QUALIFIED + FAIL` means the tested subject failed under a stable qualified environment. It must not be relabeled infrastructure failure.

## Two-seal lineage

Every qualifying execution lineage has two environment seals:

1. `PREPARED` — captured before evidence-producing commands begin;
2. `FINAL` — captured after the command plan has finished.

The FINAL seal is compared against PREPARED over all immutable lineage fields.

Any difference in an immutable field after evidence begins yields `ENVIRONMENT_DRIFT`. Pre-drift and post-drift outputs must not be combined into one evidence lineage.

## Source identity

A capsule binds:

- exact Git `HEAD`;
- clean/dirty worktree state;
- repository root identity;
- exact declared protocol/algorithm subject digests;
- exact command plan digest.

A qualifying PREPARED seal requires the worktree to be clean unless the protocol version explicitly defines another source-state policy. v1 has no such exception.

## Dependency and toolchain materials

The capsule records declared materials as `(path, required, present, sha256-or-null)` entries.

Standard v1 materials are:

- `Cargo.lock`;
- `flake.lock`;
- `rust-toolchain.toml`;
- `tools/wcare37_attestation_verifier/Cargo.lock` when the command plan includes WCARE-37 or WCARE-38 executable qualification.

Additional lockfiles/materials may be declared by the command plan.

A required material that is absent makes PREPARED qualification impossible. Missing WCARE-37 standalone `Cargo.lock` remains a blocker; WCARE-39 must not normalize or ignore it.

## Tool identity

For every declared tool used by the command plan, capture separately:

- resolved executable path;
- SHA-256 of executable bytes when readable;
- version output;
- tool role.

Version text is not binary identity. Binary hash is not semantic-version identity. Both are retained where available.

Core tools include Python, Git, and—when used—Rust/Cargo.

## Platform identity

Capture:

- operating system;
- kernel/system release;
- machine architecture;
- Python implementation and version;
- locale;
- timezone representation.

Platform identity is evidence about the execution environment, not proof of isolation or builder independence.

## Environment-variable privacy

WCARE-39 must never dump arbitrary process environment variables.

The command plan preregisters an allowlist. Each allowed variable is recorded using one of:

- `Literal` — only for explicitly non-sensitive deterministic values;
- `Sha256` — the variable value is hashed and the plaintext is not stored;
- `Absent` — the variable is not present.

Any non-allowlisted environment variable is ignored by the capsule rather than serialized.

Secrets, tokens, credentials, API keys, cookies, and authentication headers must never be intentionally recorded as literals.

## Command plan

Commands are represented as exact argv arrays, never reconstructed shell strings.

Each stage binds:

- stable stage ID;
- exact argv array;
- working directory relative to repository root;
- required materials;
- declared tool roles;
- declared network policy;
- declared sandbox policy;
- optional deterministic seed commitment.

A policy declaration is not evidence that the operating system enforced it.

## Observed command result

For each executed stage record:

- started UTC;
- finished UTC;
- exit code or signal/termination class;
- SHA-256 of stdout bytes;
- SHA-256 of stderr bytes;
- SHA-256 of the produced evidence receipt when one exists;
- observed subject outcome.

Raw stdout/stderr may be archived separately, but the capsule itself needs only their commitments.

## Network and sandbox boundary

Record declared network and sandbox policy separately from observed enforcement evidence.

`network_policy_declared = Deny` does not prove no network traffic occurred.

`sandbox_policy_declared = Restricted` does not prove filesystem/process/device isolation.

A stronger isolation claim requires separate observation/enforcement evidence.

## Drift fields

PREPARED and FINAL must agree on all immutable fields, including:

- Git HEAD;
- clean source state;
- declared subject digests;
- command-plan digest;
- required material presence and SHA-256;
- executable path/hash/version for used tools;
- platform identity;
- allowlisted environment commitments;
- locale/timezone;
- declared network/sandbox policy;
- deterministic seed commitments.

Command start/end timestamps and observed outputs are expected FINAL-only evidence and are not drift fields.

## Classifications

The execution capsule yields one of:

- `CAPSULE_PREPARED` — PREPARED seal is complete and qualifying commands may begin;
- `QUALIFIED_EXECUTION` — FINAL immutable environment matches PREPARED and the observed process result is bound;
- `ENVIRONMENT_DRIFT` — immutable environment/source/toolchain state changed after PREPARED;
- `INFRASTRUCTURE_INDETERMINATE` — environment/tool capture could not establish the required facts;
- `INVALID_CAPSULE` — malformed, contradictory, unsafe, or subject-mismatched capsule evidence.

Subject PASS/FAIL is separately reported and never inferred from the capsule classification alone.

## Replica boundary

Multiple matching capsules may demonstrate reproducibility across executions.

Replica count is not independent-builder count. Builder/organization/toolchain/fault-domain independence requires a separate provenance analysis.

## Claim boundary

WCARE-39 may support claims that an exact command plan executed under a recorded, stable environment lineage and produced an exact observed result.

It does not establish:

- consciousness or phenomenal experience;
- suffering or moral patienthood;
- objective moral truth or cultural universality;
- binding consent;
- veto or self-preservation authority;
- reviewer correctness or independence merely from replica count;
- operating-system isolation from policy declarations alone;
- solved alignment.

No WCARE-39 artifact grants live cognitive or action authority.
