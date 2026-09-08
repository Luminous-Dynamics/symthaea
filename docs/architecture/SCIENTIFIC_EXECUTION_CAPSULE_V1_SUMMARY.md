# SCI-003 — Scientific Execution Capsule v1 — Summary

**Status:** architecture-only; non-authorizing; non-qualifying.

SCI-003 defines the execution-identity layer that sits between SCI-002 artifact identity and later experiment/verification receipts.

## Core separation

```text
profile
    != realized capsule
    != execution attempt
    != completed execution
    != valid output
    != independently verified receipt
    != reproducible replay
    != scientific qualification
    != action authority
```

## Capsule coordinates

A domain-controlled execution profile may require identities for:

```text
program/generator
source
runtime/dependency closure
toolchain
process environment
platform/hardware class
scientific input snapshot
configuration
RNG/stochasticity
external tools
external state/network inputs
execution policy
```

All immutable artifact references should eventually use SCI-002 identities.

## Ambient state is part of science when it changes results

Profiles may need to freeze or declare:

```text
locale / encoding / time zone
HOME/XDG/cache/temp roots
thread/OpenMP/BLAS settings
GPU visibility/determinism
floating-point/runtime flags
PATH/tool resolution
network/proxy state
```

The shared kernel owns the mechanism; each scientific domain decides which fields are material.

## Stochasticity is typed

Avoid `deterministic: bool` or `reproducible: bool` as universal authority.

Illustrative states:

```text
DeterministicDeclared
SeededPseudoRandom
RecordedEntropy
ExternalNondeterminismBound
NondeterminismUncontrolled
Unknown
```

A seed without exact RNG identity is insufficient.

## Network/external state

If a computation consumes mutable remote state, the profile must either:

```text
forbid network access
snapshot exact remote responses
retain external interaction receipts
or explicitly declare uncontrolled external state
```

Service endpoint identity is not response identity.

## Failure is evidence

Execution-attempt receipts must preserve failure, timeout, cancellation, resource-limit, unavailable, and incomplete states.

```text
verified failed execution != successful scientific result
```

Process exit zero likewise does not prove a scientific outcome.

## Producer/verifier separation

SCI-003 adopts the strongest NeuroBridge lesson:

```text
producer receipt
    -> hostile-input independent verifier
    -> verified execution receipt
```

The verifier itself has exact implementation/execution identity. A producer field such as `verified=true` cannot mint verification.

## Replay semantics

Exact byte replay and scientific/numerical equivalence are separate claims.

```text
byte-identical != scientifically equivalent
scientifically equivalent != byte-identical
```

Any toleranced numerical-equivalence profile must be separately versioned and qualified.

## First implementation slice

Start non-executing:

```text
ScientificExecutionProfileV1
ScientificExecutionCapsuleV1
ExecutionStochasticityProfileV1
```

Then pilot one small raw attempt receipt, then a hostile-input verifier, then domain-specific qualification.

## Dependency order

```text
SCI-001
    -> SCI-002 artifact identity
    -> SCI-003 execution capsule
    -> raw attempt receipt
    -> independent verifier
    -> qualified scientific execution
    -> SCI-004 ExperimentContractV1
```

#786 verification receipts should eventually bind both SCI-002 target/artifact identities and SCI-003 verifier-execution identities while remaining separately gated by verifier soundness and target-binding qualification.
