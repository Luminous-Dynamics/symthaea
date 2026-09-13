# WCARE-39 — Execution capsule and environment-lineage protocol v1

Status: `PREREGISTERED_PROTOCOL`
Authority: `MeasurementOnly`
Protocol version: `wcare39-execution-capsule-v1`

## Purpose

WCARE-33 through WCARE-38 increasingly bind exact source, evidence, review, and authentication subjects. WCARE-39 binds those claims to the environment that actually executes them.

The governing theorem is:

`execution evidence = exact subject + exact command plan + exact environment capsule + observed process result`

A source-qualified algorithm executed under an unknown or drifting environment is not the same evidence lineage.

## Environment integrity is not subject outcome

WCARE-39 records two orthogonal dimensions:

- `environment_integrity`: `QUALIFIED`, `DRIFTED`, `INDETERMINATE`, or `INVALID`;
- `subject_outcome`: `PASS`, `FAIL`, `INVALID`, `INDETERMINATE`, or `NOT_RUN`.

A subject may fail under a fully qualified environment. `QUALIFIED + FAIL` is valid scientific evidence and must not be relabeled infrastructure failure.

## Two-seal execution lineage

Every qualifying run has:

1. `PREPARED` — captured before evidence-producing commands begin;
2. `FINAL` — captured after the ordered command plan has stopped.

For the official `run` path, the exact canonical PREPARED bytes are durably written to a Git-ignored evidence directory **before the first stage launches**. FINAL carries `prepared_capsule_sha256`, and `compare` refuses a FINAL capsule that does not bind the exact PREPARED bytes supplied to it.

The evidence directory must be Git-ignored so the capsule writer cannot dirty its own subject merely by persisting evidence.

Any immutable difference between PREPARED and FINAL yields `ENVIRONMENT_DRIFT`. Pre-drift and post-drift outputs must not be combined into one lineage.

## Source identity

A capsule binds:

- exact Git `HEAD`;
- clean/dirty worktree state;
- privacy-preserving repository identity commitment;
- exact declared protocol/algorithm subject file digests;
- exact command-plan byte digest.

A qualifying PREPARED seal requires a clean worktree. v1 has no dirty-tree exception.

Declared subject digests are path→SHA-256 commitments and are re-read from disk. PREPARED requires every declared subject file to exist and match. FINAL records the observed digest; a removed subject is represented as `null`, allowing deletion to be classified as drift rather than infrastructure failure.

Tracked or repository-local source/config files named directly in stage argv must be bound through `subject_digests` or declared materials. This prevents an uncommitted command input from escaping the evidence subject.

## Dependency and toolchain materials

The capsule records each material as `(path, required, present, sha256-or-null)`.

Standard v1 materials are always part of the environment lineage:

- `Cargo.lock`;
- `flake.lock`;
- `rust-toolchain.toml`.

When any planned argv references WCARE-37 or WCARE-38 qualification, WCARE-39 additionally requires:

- `tools/wcare37_attestation_verifier/Cargo.lock`.

That requirement is automatic, not caller-optional. The currently missing standalone WCARE-37 lock therefore remains an explicit blocker for WCARE-37/WCARE-38 executable qualification.

Additional materials may be preregistered by the command plan.

## Tool identity

For Git, the running Python implementation, each stage executable, and detected shebang interpreters, record:

- stable tool role;
- privacy-safe executable locator;
- SHA-256 of executable bytes when present;
- version output and its SHA-256.

A repository executable is represented relative to the repository. `/nix/store/...` locators may remain explicit. Other host paths are reduced to basename plus a path commitment so personal filesystem layouts are not unnecessarily disclosed.

Unknown stage programs are **not executed with `--version` during PREPARED**. Their bytes are hashed, but arbitrary code is never run merely to collect version metadata. Known toolchain executables may be queried with `--version`.

Version text is not binary identity; binary hash is not semantic-version identity. Both are retained where available.

## Platform identity

Capture:

- operating system;
- kernel/system release;
- machine architecture;
- Python implementation and version;
- locale;
- timezone representation.

Platform identity is environment evidence, not proof of isolation or builder independence.

## Environment privacy and hidden drift

WCARE-39 never dumps arbitrary process environment variables.

The command plan may preregister a safe allowlist. Each listed variable is recorded as:

- `Literal` — only for explicitly non-sensitive bounded values;
- `Sha256` — only the value commitment is stored;
- `Absent` — variable not present.

Sensitive-looking variable names such as tokens, passwords, cookies, authentication material, API keys, or private keys cannot be stored as literals.

Unlisted variables are not serialized individually. However, WCARE-39 also records one aggregate `ambient_environment_sha256` over the complete inherited environment. This commitment detects hidden environment drift without publishing arbitrary names or values.

The aggregate commitment is not a substitute for publishing reproducible configuration; it is only a drift detector.

## Command plan

Commands are represented as exact argv arrays and are never reconstructed shell strings. The runner uses `shell = false` semantics.

Each stage binds:

- stable stage ID;
- exact argv array;
- working directory relative to repository root;
- timeout in seconds;
- optional fresh output-receipt path.

Timeout is part of experiment identity. A 30-second evaluation and a 30-minute evaluation are different command plans.

Output-receipt paths must be unique. A declared output receipt that already exists before its stage begins invalidates that stage rather than allowing stale evidence to be reused.

Stages are ordered. After a FAIL, INVALID, timeout, or infrastructure termination, later stages remain `NOT_RUN`; the runner does not silently continue a dependent campaign.

## Observed command result

For every attempted stage record:

- started UTC;
- finished UTC;
- exit code or termination class;
- SHA-256 of stdout bytes;
- SHA-256 of stderr bytes;
- SHA-256 of a fresh output receipt when declared and produced;
- observed subject outcome.

Raw stdout/stderr are not embedded into the capsule, reducing accidental disclosure. Their byte commitments remain available for archive binding.

## CI-facing qualifier contract

`wcare39-qualify.sh` is the official automation-facing entry point. It resolves the exact WCARE-39 runner relative to the qualifier itself while evaluating the Git worktree from which the qualifier is invoked. This allows the frozen WCARE-39 implementation to evaluate an isolated checkout without requiring that checkout to contain another copy of the runner.

The JSON capsule remains the authoritative scientific record. The shell exit status is an operational projection designed to prevent CI from treating a qualified environment as a passing subject:

- `0` — `QUALIFIED_EXECUTION + PASS`;
- `1` — `QUALIFIED_EXECUTION + FAIL` or `QUALIFIED_EXECUTION + INVALID`;
- `2` — `ENVIRONMENT_DRIFT`;
- `3` — `INFRASTRUCTURE_INDETERMINATE` or an unparseable/indeterminate wrapper result;
- `4` — `INVALID_CAPSULE` or invalid qualifier invocation.

The underlying runner may represent `QUALIFIED_EXECUTION + FAIL` as a scientifically valid capsule; the qualifier nevertheless returns 1 so automated promotion cannot go green on subject failure.

## Network and sandbox boundary

Declared network and sandbox policies are recorded separately from observed enforcement.

`network_policy_declared = Deny` does not prove no network traffic occurred.

`sandbox_policy_declared = Restricted` does not prove filesystem/process/device isolation.

Accordingly, `network_isolation_established` and `sandbox_enforcement_established` remain false in v1 unless a separate evidence program establishes them.

## Drift fields

PREPARED and FINAL compare all immutable fields:

- Git HEAD and clean source state;
- repository identity commitment;
- declared subject digests, including deletion;
- exact command-plan digest;
- required material presence and hashes;
- tool locators/hashes/version identities;
- platform identity;
- safe environment observations;
- aggregate ambient environment commitment;
- declared network/sandbox policy;
- deterministic seed commitments;
- stage IDs, argv, cwd, and timeout.

Start/end timestamps, exit codes, stdout/stderr digests, output-receipt digests, and subject outcomes are expected observations and are not drift fields.

## Classifications

The capsule classification is one of:

- `CAPSULE_PREPARED` — PREPARED seal is valid and commands may begin;
- `QUALIFIED_EXECUTION` — FINAL immutable state matches PREPARED and the process result is bound;
- `ENVIRONMENT_DRIFT` — immutable environment/source/toolchain state changed after PREPARED;
- `INFRASTRUCTURE_INDETERMINATE` — required facts or execution infrastructure could not be established;
- `INVALID_CAPSULE` — malformed, contradictory, unsafe, stale-output, or subject-mismatched evidence.

Subject PASS/FAIL is separate and never inferred from capsule classification alone.

## Synthetic qualification campaign

The v1 implementation includes a dependency-free synthetic Git campaign that must exercise at least:

- `QUALIFIED_EXECUTION + PASS`;
- `QUALIFIED_EXECUTION + FAIL`;
- dependency/worktree drift;
- deletion of a bound subject file as drift;
- automatic WCARE-37 standalone-lock blocking for a WCARE-38-like stage;
- exact FINAL→PREPARED byte binding;
- stale output-receipt rejection;
- sensitive environment-literal rejection;
- the qualifier's exact `0/1/2/3/4` automation exit contract.

Passing the synthetic campaign validates runner and qualifier invariants only; it does not qualify any real Symthaea evidence subject.

## Replica boundary

Multiple matching capsules may demonstrate execution reproducibility.

Replica count is not independent-builder count. Builder, organization, toolchain, review-process, and fault-domain independence remain separate evidence questions.

## Claim boundary

WCARE-39 may support the bounded claim that an exact command plan executed under a recorded stable environment lineage and produced an exact observed process result.

It does not establish consciousness, phenomenal experience, suffering, moral patienthood, objective moral truth, cultural universality, binding consent, veto/self-preservation authority, reviewer correctness, builder independence merely from replica count, operating-system isolation from policy declarations alone, or solved alignment.

No WCARE-39 artifact grants live cognitive or action authority.
