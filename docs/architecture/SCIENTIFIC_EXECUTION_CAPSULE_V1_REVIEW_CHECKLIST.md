# SCI-003 Review Checklist — Scientific Execution Capsule v1

**Status:** review aid only; non-authorizing; non-qualifying.

## A. Layer separation

- [ ] Execution profile is distinct from realized capsule.
- [ ] Realized capsule is distinct from execution attempt.
- [ ] Attempt is distinct from completion.
- [ ] Completion is distinct from output validity.
- [ ] Output validity is distinct from independent receipt verification.
- [ ] Verified execution receipt is distinct from scientific qualification.
- [ ] Scientific qualification is distinct from experiment success, replication, safety, and action authority.

## B. Program/runtime identity

- [ ] Exact program/generator identity is bound.
- [ ] Source commit alone is not assumed to identify the built/runtime computation.
- [ ] Runtime/dependency closure semantics are explicit.
- [ ] Toolchain/interpreter/solver versions are bound where material.
- [ ] Model weights/tokenizers/plugins/external tool artifacts are bound where material.
- [ ] SCI-002 artifact identities are used rather than mutable paths as sole identity.

## C. Inputs/configuration

- [ ] Exact scientific input snapshot identity is bound.
- [ ] Input membership/order semantics are explicit.
- [ ] Scientific configuration identity is bound.
- [ ] `DefaultHasher` / Debug-based config diagnostics are not used as authority-bearing scientific identity.
- [ ] Preprocessing/analysis options that change output are part of execution lineage.

## D. Environment/platform

- [ ] Relevant process environment is frozen/declared by profile.
- [ ] Locale/time-zone/threading/cache/temp semantics are explicit where material.
- [ ] Platform/architecture/runtime identities are explicit where material.
- [ ] Generic identity does not include irrelevant machine serials by default.
- [ ] Cross-platform numerical equivalence is not assumed.

## E. Stochasticity/nondeterminism

- [ ] Stochasticity is a typed state, not a boolean.
- [ ] Seeded execution binds RNG algorithm/profile as well as seed.
- [ ] Parallel RNG/stream policy is explicit where material.
- [ ] Recorded entropy becomes an explicit input/evidence artifact when replay requires it.
- [ ] Uncontrolled nondeterminism remains explicitly uncontrolled.
- [ ] Thread/GPU reduction nondeterminism cannot be hidden behind a fixed seed.

## F. External state/network

- [ ] Network access policy is explicit.
- [ ] Mutable remote responses are snapshotted/content-addressed or explicitly uncontrolled.
- [ ] Service endpoint identity is distinct from exact response identity.
- [ ] Secrets are not serialized merely to make a capsule complete.
- [ ] Security/access authority remains distinct from scientific input identity.

## G. Invocation/time

- [ ] Structured argv/request semantics are retained where possible.
- [ ] Shell text is not assumed equivalent to structured invocation.
- [ ] Working directory/stdin/resource-limit semantics are bound where material.
- [ ] Attempt timestamps are provenance, not automatically capsule identity.
- [ ] Program-observed current time is treated as input/nondeterminism if it affects output.
- [ ] Claimed timestamps are not treated as authenticated chronology.

## H. Failure/output evidence

- [ ] Failure/timeout/cancellation/incomplete states are first-class.
- [ ] Failed runs can be independently verified as failed.
- [ ] Exit zero does not imply scientific success.
- [ ] Outputs are SCI-002 artifact identities bound to the exact attempt.
- [ ] stdout/stderr, primary outputs, intermediate artifacts, proof witnesses, and diagnostics have explicit roles.
- [ ] Raw execution observation remains distinct from normalized interpretation.

## I. Independent verification

- [ ] Producer cannot self-certify by setting a boolean/field.
- [ ] Hostile-input verifier can reconstruct the declared capsule/attempt relation from retained evidence.
- [ ] Verifier implementation identity is bound.
- [ ] Hash implementation A / execute implementation B substitution fails.
- [ ] Extra/missing receipt artifacts are handled according to explicit closed/open receipt semantics.

## J. Reproducibility semantics

- [ ] No universal `reproducible: bool` is proposed.
- [ ] Exact byte replay is distinct from numerical/scientific equivalence.
- [ ] Stochastic process specification is distinct from exact replay.
- [ ] Any numerical-equivalence tolerance belongs to a versioned/preregistered comparison profile.

## K. Migration safety

- [ ] Neuro Workbench semantics are not weakened.
- [ ] Matter solver receipts remain domain-native.
- [ ] Physical Agency strict simulation receipts remain domain-native.
- [ ] Existing execution receipts are not retroactively qualified by SCI-003.
- [ ] Adapters must prove the generic capsule contains all domain-required execution coordinates.

## L. First implementation gate

The first shared Rust slice should remain non-executing:

```text
ScientificExecutionProfileV1
ScientificExecutionCapsuleV1
ExecutionStochasticityProfileV1
```

Reject first-tranche expansion into:

- actually running tools/solvers;
- experiment contracts;
- outcome adjudication;
- scientific qualification;
- replication;
- safety;
- action authority.

## Review question

> Does SCI-003 define a sufficiently exact executable-computation boundary for later reproducibility and verification without turning environment identity into execution evidence or execution evidence into scientific/action authority?
