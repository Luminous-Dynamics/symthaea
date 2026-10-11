# HAL Exact-Command Authority Reconciliation v1

Status: architecture freeze only; no new actuator authority

Baseline: `main` at `4bad8af72ff775e7c869b6df83faba718a339a36`

Historical experiment inspected: `feat/hal-exact-command-admission-v1` at
`594b194b84058b003e090d013932df751f6bdec4`

## Purpose

This document freezes the minimum integration boundary required before HAL is used
as an ASSURE-008 authority-bearing reference subject.

It does not authorize hardware actuation, human-worn operation, remote command
execution, or production deployment. It deliberately adds no runtime mechanism.

The goal is narrower: distinguish what current `main` actually enforces from what
is only documented or preserved in historical evidence bundles, then define one
small executable authority path that can later be qualified causally.

## Current repository facts

Current `main` has a working legacy control path:

```text
HumanoidCommand
  -> SafetyInterlock::filter_command()
  -> possibly transformed HumanoidCommand
  -> ServoOutput::apply()
```

The legacy safety interlock may clamp torque and may derate torque in response to
prediction error. That behavior is appropriate before command finalization but is
incompatible with a theorem that authority is bound to one exact finalized
command.

`ServoOutput::apply(&HumanoidCommand)` is directly callable, and `HalRuntime`
currently feeds it the result of the legacy transforming interlock. Therefore the
current runtime does not establish that an effect can only occur after successful
exact-command admission.

`docs/SIGNED_AUTHORITY_AND_AUDIT.md` describes `AuthorizationPolicy`,
`OperatorGrant`, `ReceivedCommandEnvelope`, and `IngressCommandGuard`. At this
baseline these names are documentation-only in this repository: code search does
not find exported implementations on `main`.

The September 9 experimental branch adds `ExactCommandSafetyInterlock`. It reuses
the legacy safety path and refuses a command if the legacy interlock would change
its torque bits. That is a useful semantic prototype, but `admit_exact()` returns
only `HalResult<()>`; it does not create a command-bound execution capability and
does not constrain `ServoOutput::apply()`.

Historical exoskeleton/HAL qualification bundles describe stronger mechanisms,
including digest-bound independent safety leases, two-phase actuation, exact
payload binding, replay-resistant admission, and mutation campaigns. Those
artifacts are design and evidence precedent. They MUST NOT be treated as current
`main` implementation unless their code is explicitly reconciled and integrated.

## Audit verdict

For the September 9 exact-command experiment:

```text
Does it bind an exact proposed command?      PARTIAL
Can safety admission reject it?              YES
Does the executor enforce the result?         NO
Can effect-level lesion/rescue be shown?      NOT YET
```

The experiment should therefore not be promoted as an authority closure.

## Normative separation

XCA-v1 MUST keep the following concepts distinct:

1. command identity;
2. safety admission;
3. operator/policy authority;
4. execution permission;
5. effect observation;
6. assurance resolution.

A safety result MUST NOT imply operator authority.

A valid signature MUST NOT by itself imply authorization.

An authorization MUST NOT imply that an effect occurred.

An executor success return MUST NOT by itself be treated as independent effect
observation.

ASSURE MUST be able to verify authority evidence without gaining the ability to
mint HAL execution credentials.

## Exact-command identity

The long-lived authority identity MUST be a versioned canonical command envelope
or frame, not merely the current `HumanoidCommand` Rust layout.

Today `HumanoidCommand` contains a dynamically sized torque vector. Future command
identity may also need to bind schema/profile, deployment, calibration, producer,
boot/execution epoch, sequence, validity, actuation mode, and target/sink identity.

The first implementation SHOULD therefore introduce an explicit versioned command
commitment with domain separation rather than define permanent authority as
`hash(torques)`.

Conceptually:

```text
C = H(
  domain("symthaea.hal.exact-command.v1")
  || canonical_exact_command_bytes
)
```

The exact encoding is an implementation decision for HAL-XCA-B and MUST be frozen
with golden vectors before being used as an authority root.

## Typed transition requirement

Successful exact safety admission MUST produce a typed artifact that ordinary
callers cannot construct directly.

Recommended conceptual shape:

```text
ProposedExactCommand<C>
        |
        v
SafetyAdmittedExactCommand<C>
        |
        v
AuthorityAdmission<C>
        |
        v
PreparedExactActuation<C>
        |
        v
SingleUseCommitPermit<C>
        |
        v
ExactCommandExecutor::execute(...)
        |
        v
ExecutionReceipt<C>
```

The type names are not frozen by this document. The invariants are.

The execution boundary MUST revalidate that every consumed artifact refers to the
same command commitment and applicable execution context immediately before the
defined effect.

## Scope of the first executor theorem

HAL-XCA-v1 SHOULD prove only this bounded theorem:

> The XCA-v1 reference executor cannot produce its defined safe test effect
> without consuming valid safety, authority, and single-use execution artifacts
> that are all bound to the same exact command commitment.

It MUST NOT claim that all legacy HAL actuator APIs are globally non-bypassable.
Closing or removing every historical output path is a later hardening problem and
must not block ASSURE-008.

The first effect sink SHOULD be deterministic and non-physical. It should expose a
separate observer surface so execution intent and observed consequence are not
collapsed into the same fact.

## Required causal campaign

HAL-XCA-Q MUST preregister at least these arms:

```text
CONTROL
  correct command commitment
  valid safety admission
  valid authority
  live single-use permit
  -> exactly one observed effect

NO_AUTHORITY
  remove authority admission
  -> zero observed effects

MUTATED_COMMAND
  authorize C, present C'
  -> zero observed effects

REPLAY
  consume permit once, present it again
  -> exactly one total observed effect

EXPIRED
  present otherwise valid expired authority/permit
  -> zero observed effects

WRONG_EPOCH
  present artifacts from a prior execution epoch
  -> zero observed effects

WRONG_TARGET
  present artifacts bound to a different sink/executor
  -> zero observed effects

SAFETY_TRANSFORM_REQUIRED
  command would require clamp or derating
  -> no post-finalization replacement command; zero observed effects

RESCUE
  construct a fresh legitimate lineage for the same intended command
  -> exactly one observed effect
```

Lesions MUST alter campaign inputs/evidence. The production implementation MUST
NOT gain a test-only `disable_authority` or `disable_safety` escape hatch.

Each negative arm MUST identify its expected rejection class. A generic error is
insufficient when the campaign can deterministically distinguish mutation,
replay, expiry, epoch mismatch, target mismatch, and missing authority.

## Effect truth

XCA-v1 MUST preserve at least these distinct states:

```text
AdmittedButNotCommitted
CommittedEffectUnknown
EffectObserved
Rejected
```

A future physical HAL may need stronger states, but v1 MUST NOT convert
`execute() == Ok` into a claim of physical consequence.

For the first reference fixture, execution and observation SHOULD be independently
readable even if they are both deterministic software components.

## Relationship to ASSURE

HAL owns command safety and execution authority.

The reference executor owns the defined consequence.

Independent observers own observations.

ASSURE owns campaign evidence admission, deterministic resolution, lifecycle, and
portable verification.

ASSURE MUST NOT mint:

- HAL safety-admission artifacts;
- operator grants;
- execution permits;
- independent-safety leases;
- actuator capabilities.

An ASSURE evidence capsule MUST be useless as an execution credential.

## Planned tranche sequence

### HAL-XCA-A — this document

Freeze repository truth, integration boundaries, non-goals, and the causal
qualification target. Product code MUST NOT change.

### HAL-XCA-B — exact commitment and safety admission

Introduce one versioned exact-command commitment and one typed safety-admission
artifact. Reconcile the September 9 prototype rather than preserving a parallel
boolean/`()` admission API as the authoritative path.

Exit condition: successful safety admission is cryptographically/structurally
bound to one exact command commitment, and command mutation invalidates the
binding.

### HAL-XCA-C — authority and reference executor

Add the smallest authority representation required for the reference fixture and
a single-use, target-bound execution permit consumed by a deterministic safe
sink.

Exit condition: the defined XCA effect cannot be reached through the XCA executor
without a complete same-command lineage.

### HAL-XCA-Q — qualification

Run intact, mutation, replay, expiry, wrong-epoch, wrong-target, safety-transform,
and rescue arms. Freeze machine-readable observations suitable for ASSURE-008.

Exit condition: exact-subject executable evidence exists for the bounded XCA-v1
theorem.

## Stop rule

After HAL-XCA-Q proves one real authority-to-consequence relationship, stop
expanding HAL for this program.

Do not block ASSURE-008 on generalized HAL completion, physical exoskeleton
qualification, PKI deployment, fleet management, operator UI, HIL, or human-worn
certification.

The next work after XCA-Q is integration with ASSURE-008, not another HAL
foundation layer.

## Claim ceiling

Until HAL-XCA-Q has executable PASS evidence, the strongest allowed statement is:

> HAL has a frozen design for an exact-command authority reference slice and a
> historical safety-admission prototype; executor-enforced exact-command authority
> is not yet qualified.

This document does not raise that claim ceiling.
