# Qualification Execution Identity Census v1

Date: 2026-09-09
Status: architecture / non-authorizing
Series: SCI-Q4

## Purpose

SCI-Q3 isolates the identity of the candidate artifact being qualified. SCI-Q4 asks the next separate question:

> Under what exact executable environment and input state was that candidate actually evaluated?

This document is a census and boundary contract only. It does not introduce an execution capability, does not qualify any current PR, and does not authorize materialization.

## Core non-equivalences

```text
artifact identity
    != execution profile
    != realized execution environment
    != execution occurrence
    != execution result
    != verified execution identity
    != reproducibility
    != scientific qualification
    != authority
```

Additional rules:

```text
GitHub run id != execution environment identity
ubuntu-latest != immutable runner identity
actions/checkout@v4 != immutable action implementation
rustc --version != rustc binary identity
Cargo.lock != complete dependency/runtime closure
flake.lock != complete host/runtime closure
Docker tag != container image digest
seed != complete stochastic execution identity
successful exit != scientific success
replayable != scientifically valid
```

SCI-Q4 must preserve these distinctions.

## Existing independent implementations

### 1. Muse independent reproduction environment

`crates/domains/symthaea-muse/src/reproducibility_attestation.rs` records a `ReproductionEnvironment` containing:

- operating system;
- architecture;
- Nix version;
- `flake.lock` SHA-256;
- toolchain-evidence SHA-256;
- execution-environment SHA-256;
- commands SHA-256.

The same attestation separately records verifier identity, output matches, command success, analysis cross-check state, exact release-root reproduction, limitations, timestamps, external receipt URI, and signature commitment.

Useful recurring mechanic:

```text
execution environment evidence != output equivalence != verifier independence != scientific conclusion
```

Keep Muse's study-specific release semantics local.

### 2. Fabrication deterministic replay environment

`crates/domains/symthaea-fabrication-kernel/src/replay.rs` records a `ReplayEnvironment` containing:

- kernel version;
- source revision;
- target triple;
- Rust compiler version;
- optional Cargo.lock digest;
- canonicalized feature flags.

Its replay contract additionally binds:

- fabrication manifest digest;
- deterministic seed;
- algorithm/version inventory.

Useful recurring mechanic:

```text
source + toolchain + dependency policy + features + seed + algorithms
    -> replay profile
```

Keep fabrication manifest/trust/audit semantics local.

### 3. Muse execution/environment commitments across the study pipeline

Multiple Muse structures independently retain source, input, verifier, renderer, execution-environment, and toolchain digests. Examples include structural evidence, analysis cross-checks, pilot reports, artifact production, final release, and replication execution.

This recurrence supports a shared execution-identity substrate, but not a universal scientific-study schema.

## Proposed SCI-Q4 object model

The shared contract should distinguish at least four objects.

### A. DeclaredExecutionProfileV1

The prospective execution contract.

Possible fields:

- profile schema/domain;
- expected artifact identity reference;
- command/argv commitment;
- working-directory contract;
- environment-variable allowlist/commitment;
- toolchain requirements;
- dependency/lock requirements;
- target architecture/ABI requirements;
- feature/configuration requirements;
- external-tool requirements;
- input artifact identities;
- RNG/stochasticity policy;
- network/external-state policy;
- resource-limit policy.

A profile is a declaration. It does not prove realization.

### B. RealizedExecutionEnvironmentV1

What the runner actually measured before/during execution.

Possible fields:

- operating-system image/content identity;
- kernel/runtime identity where relevant;
- architecture and target triple;
- CPU/GPU/device capabilities where scientifically relevant;
- exact compiler/tool binaries or trusted closure identities;
- Cargo.lock / flake.lock / dependency closure commitments;
- feature/configuration realization;
- exact external-tool identities;
- relevant environment-variable realization;
- container/Nix/store closure identities;
- uncontrolled-state inventory.

A realized environment can differ from the declared profile. That mismatch is evidence, not something to normalize away.

### C. ExecutionOccurrenceV1

One actual attempt.

Possible fields:

- artifact identity;
- declared profile identity;
- realized-environment identity;
- input identities;
- command identity;
- RNG seed/state identity where applicable;
- start/end occurrence metadata;
- exit/termination disposition;
- stdout/stderr/log artifact identities;
- output artifact identities;
- failure/timeout/cancellation state;
- CI/provider locator such as GitHub run/job id.

Run/job IDs are useful locators, not environment identities.

### D. VerifiedExecutionIdentity

Opaque witness returned only after a verifier checks the declared profile, realized environment, occurrence, artifact identity, and required inputs are mutually coherent.

It must not be serializable as an independently trusted authority token.

It means only:

> this execution occurrence is bound to these exact execution-identity claims under this verifier contract.

It does not mean the scientific theorem passed.

## Required execution identity coordinates

The first qualification profile should treat the following as independent coordinates rather than one generic `environment_hash`.

### Candidate / source

Provided by SCI-Q3:

- repository;
- base commit;
- candidate patch bytes;
- resulting tree;
- changed-path scope;
- workflow bytes;
- qualification-profile semantics.

### Toolchain

At minimum for Rust qualification:

- rustc identity;
- Cargo identity;
- rustfmt identity;
- Clippy identity;
- target triple;
- enabled feature set.

Version strings are useful metadata but do not necessarily identify binaries.

### Dependency closure

At minimum:

- Cargo.lock content identity;
- Nix/flake lock identity when Nix defines the environment;
- relevant vendored/source dependency closure;
- dynamically loaded native libraries where they can affect results.

A lockfile digest is not automatically the same as a realized store/runtime closure.

### Runner/runtime

Record what materially affects the result:

- OS/image identity;
- kernel/runtime identity if relevant;
- architecture;
- CPU instruction capabilities when numerics/code paths depend on them;
- GPU/accelerator identity when used;
- container/Nix/store closure identity;
- locale/timezone only when semantically relevant.

Do not add irrelevant host trivia merely to maximize hash size.

### External tools

Examples:

- Lean;
- Z3;
- Connectome Workbench;
- FluidSynth;
- Python/interpreters;
- compilers/linkers;
- domain solvers.

For each tool, distinguish:

```text
name != version string != binary/content identity != invocation identity
```

### Inputs

Every evidence-relevant external input should be identity-bearing:

- datasets;
- fixtures;
- models/checkpoints;
- soundfonts;
- configuration;
- generated intermediate artifacts;
- previous receipts used as inputs.

### Commands and environment

Bind exact command semantics, including where relevant:

- executable identity;
- argv;
- working directory;
- selected environment variables;
- feature flags;
- config files;
- resource limits.

Avoid hashing the entire ambient environment without a semantic allowlist: secrets, ephemeral variables, and irrelevant scheduler metadata should not become scientific identity merely because they existed.

## Stochasticity and nondeterminism

SCI-Q4 must not reduce stochastic identity to `seed: u64`.

Distinguish:

- RNG algorithm/version;
- seed/state;
- deterministic scheduling assumptions;
- parallel reduction/order effects;
- GPU nondeterminism;
- external entropy;
- nondeterministic services;
- wall-clock/time-dependent behavior.

Suggested state:

```text
DeterministicByConstruction
SeededUnderDeclaredRng
KnownNondeterminism { sources }
UncontrolledExternalState { sources }
Unknown
```

`Unknown` and uncontrolled state remain valid evidence states. They reduce reproducibility claims rather than disappearing.

## Network and external state

A live network response cannot be made reproducible by recording only the URL.

Prefer:

```text
request contract
+ fetched artifact snapshot/content identity
+ relevant response metadata
```

If exact snapshotting is impossible, record the external dependency as uncontrolled.

```text
live API consulted != stable input artifact
```

## Failure receipts remain evidence

Execution identity must survive:

- process failure;
- theorem/test failure;
- timeout;
- cancellation;
- resource exhaustion;
- infrastructure interruption.

A failed occurrence can still have a fully valid execution identity.

Therefore:

```text
VerifiedExecutionIdentity != SuccessfulExecution
```

This mirrors SCI-Q1's separation between execution observations and qualification disposition.

## Mutable CI references

Current qualification workflows commonly use conveniences such as:

- `actions/checkout@v4`;
- `ubuntu-latest`;
- version-oriented toolchain selectors.

These are reasonable operational selectors but are not sufficient immutable execution identities.

SCI-Q4 should eventually record the realized immutable action commit/image/tool closure used by a run where available.

Do not weaken existing workflows merely because this information is not yet captured. The correct current status is `execution identity incomplete`, not fabricated precision.

## Profile versus realization

A central invariant:

```text
what the workflow requested != what the runner actually realized
```

For example:

```text
requested: Rust 1.96
realized: exact rustc binary/toolchain closure X
```

Both belong in the evidence chain.

The verifier should reject or explicitly classify disallowed drift rather than overwriting the declaration with the observed value.

## Output identity is separate

Execution identity should bind output artifact identities but should not treat output equality as part of the environment identity itself.

This preserves:

```text
same execution identity + different output
```

as a meaningful nondeterminism observation.

Likewise:

```text
different execution identity + equivalent output
```

can be meaningful replication evidence.

## Relationship to SCI-Q1/Q2/Q3

```text
SCI-Q1
execution observations -> qualification disposition -> bounded reporting witness

SCI-Q2
identity census and non-equivalences

SCI-Q3
exact candidate artifact identity

SCI-Q4
exact execution profile/realization/occurrence identity
```

Neither Q3 nor Q4 alone authorizes materialization.

## Future SCI-Q4 qualification candidate

A later qualification-only PR should test an execution identity envelope against a real hosted run.

The hosted negative controls should mutate independently:

1. candidate artifact identity;
2. command identity;
3. Cargo.lock/flake-lock identity;
4. compiler/tool identity;
5. feature flags;
6. target architecture;
7. external-tool identity;
8. input identity;
9. RNG policy;
10. realized environment identity.

Every required mismatch must fail closed.

It should additionally prove:

- missing required realization is not treated as equality;
- duplicate/conflicting environment observations are malformed;
- mutable labels are not accepted as exact identities unless resolved to immutable realized identities;
- an execution failure can still produce a valid execution-identity witness;
- the witness cannot mint qualification or materialization authority.

## Non-goals

SCI-Q4 does not attempt to:

- create one universal environment hash;
- force every domain into Nix;
- require deterministic reproduction for intrinsically stochastic experiments;
- authenticate the runner;
- establish remote attestation;
- establish scientific correctness;
- authorize deployment/materialization.

Those are separate contracts.

## Recommended implementation order

1. Let SCI-Q1 and SCI-Q3 exact hosted candidates qualify or fail on their own evidence.
2. Preserve the existing Muse/Fabrication local implementations.
3. Define a small shared execution profile/realization vocabulary in evidence-plane only after Q4's recurring fields are reviewed against at least three domain implementations.
4. Qualify identity verification before adding signatures/attestation.
5. Add anti-replay binding between Q1 receipt, Q3 artifact identity, and Q4 execution identity.
6. Only then design materialization authority.

## Governing principle

> An execution is not identified by where it was scheduled or what version label was requested. It is identified by the artifact, inputs, command semantics, and realized computational environment that could materially affect the result.
