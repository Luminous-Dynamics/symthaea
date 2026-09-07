# Scientific Execution Capsule v1

**Status:** architecture contract only; non-authorizing; non-qualifying.

**Series:** SCI-003, stacked on SCI-002.

**Parent:** `architecture/scientific-artifact-identity-v1@aaadaee6d474f427ea5269587de6acc84e337c90`

## 1. Purpose

SCI-001 identifies scientific execution identity as a reusable cross-domain primitive. SCI-002 defines the artifact-identity substrate needed to bind exact source, inputs, configuration, runtime closure, tools, and outputs without confusing identity with scientific validity.

Symthaea already has a strong concrete execution lineage in NeuroBridge:

```text
Workbench execution profile
    -> canonical Nix closure identity
    -> raw realization/closure observation receipt
    -> hostile-input independent receipt verification
    -> future execution-capsule qualification
```

Other domains reinforce complementary parts of the same theorem:

- Matter binds exact solver/input/output/Hamiltonian/basis/reference identities without treating a solver receipt as experimental truth;
- Physical Agency binds exact simulation request/output/claim lineage while preserving `solver result != successful outcome != safety`;
- Futures separates artifact provenance, transform closure, availability, and custody;
- scientific evidence work repeatedly preserves failed/incomplete execution as evidence instead of dropping it.

SCI-003 freezes the common execution semantics before any universal `ScientificExecutionCapsuleV1` implementation exists.

---

## 2. Core theorem

The execution model must preserve:

```text
execution profile
    != realized runtime environment
    != execution attempt
    != execution completed
    != output produced
    != independently verified execution receipt
    != reproducible replay
    != scientifically qualified result
    != action authority
```

and:

```text
same source code
    != same executable computation

same executable binary
    != same runtime closure

same runtime closure
    != same process environment

same process environment
    != same scientific inputs

same capsule
    != same stochastic outcome

process exit success
    != scientific success
```

The central rule is:

> A scientific execution capsule identifies the declared computation boundary. It does not assert that the computation ran, succeeded, reproduced, or established a scientific claim.

---

## 3. Execution layers

SCI-003 defines five distinct layers.

### 3.1 Execution profile

A versioned declaration of what the domain considers execution-relevant.

Conceptually:

```text
ScientificExecutionProfileV1 {
    profile_identity
    domain_namespace
    required_identity_slots
    environment_policy
    stochasticity_policy
    external_state_policy
    platform_policy
    output_capture_policy
}
```

The profile says **which facts must be bound**, not that they currently exist.

### 3.2 Realized execution capsule

The exact immutable identities selected for one executable computation boundary.

Conceptually:

```text
ScientificExecutionCapsuleV1 {
    profile_identity
    generator_or_program_identity
    source_identity
    dependency_runtime_closure_identity
    toolchain_identity
    process_environment_identity
    platform_identity
    input_snapshot_identity
    configuration_identity
    stochasticity_identity
    external_tool_identities
    external_state_identities
    execution_policy_identity
}
```

Not every domain needs every slot. Optionality must be controlled by the profile, not by ad-hoc callers.

### 3.3 Execution attempt receipt

Records that a process/job/solver attempt actually occurred.

Conceptually:

```text
ScientificExecutionAttemptReceiptV1 {
    capsule_identity
    attempt_identity
    start_observation
    completion_observation
    argv_or_invocation_identity
    exit_state
    stdout_identity
    stderr_identity
    output_artifact_identities
    resource_limit_state
    timeout_state
    external_interaction_receipts
}
```

An attempt receipt is allowed to represent failure, timeout, interruption, missing output, or partial output.

### 3.4 Verified execution receipt

A separate verifier reconstructs enough of the attempt/capsule relation from retained evidence to establish that the receipt is internally consistent under its declared profile.

```text
raw attempt receipt
    + retained artifacts
    + exact profile
    + verifier implementation
        -> VerifiedScientificExecutionReceiptV1
```

This is the general form of the NeuroBridge producer/verifier split.

### 3.5 Scientific execution qualification

A later domain/scientific policy may decide that a verified execution receipt is eligible for a specific scientific use.

```text
VerifiedExecutionReceipt
    + domain qualification policy
    + required prerequisites
        -> QualifiedScientificExecution
```

Qualification remains separate from result interpretation, experimental validity, replication, safety, or action authority.

---

## 4. Required identity coordinates

### 4.1 Generator / program identity

The capsule must bind what implementation actually executes.

Depending on the domain this may include:

```text
source tree / commit identity
built executable or script identity
compiler/interpreter identity
build configuration
feature flags
code-generation artifacts
model weights
prompt/template artifacts
solver implementation
```

A source commit alone is insufficient when the build/runtime product can vary.

### 4.2 Dependency and runtime closure

The execution must bind the dependencies that can materially affect computation.

Examples:

```text
Nix runtime closure
container image/root filesystem
shared libraries
Python/R package lock closure
JVM/.NET runtime
GPU runtime libraries
solver plugins
model/tokenizer/embedding artifacts
```

The Workbench Nix closure lineage is the leading concrete pattern.

A dependency list is not necessarily a closed runtime closure. The profile must state its closure semantics.

### 4.3 Toolchain identity

Where output can depend on compilation/interpreter/tool semantics, bind the relevant toolchain.

Examples:

```text
Rust compiler + target
C/C++ compiler/linker
Python interpreter
Lean version + Mathlib environment
Z3 version
Nix version/protocol
CUDA compiler/runtime
Workbench version
```

Tool labels such as `z3` or `lean` are insufficient without version/profile identity when results depend on them.

### 4.4 Scientific input snapshot

Inputs must refer to SCI-002 artifact identities, not mutable paths alone.

Examples:

```text
datasets
fixtures
model weights
initial conditions
Hamiltonian/basis files
source documents
experiment configuration
world snapshot
proof target
```

Input snapshots should make membership and ordering semantics explicit.

### 4.5 Configuration identity

Configuration that changes scientific behavior belongs in the capsule.

Examples:

```text
solver tolerances
model hyperparameters
analysis options
feature flags
preprocessing policy
integration timestep
iteration limits
termination criteria
measurement configuration
```

Do not use `symthaea-evidence-plane::config_hash()` / `DefaultHasher` diagnostics as the authority-bearing configuration identity. SCI-003 configuration references must eventually use SCI-002 canonical artifact identities.

---

## 5. Process environment boundary

Ambient process state can change scientific results.

A profile must explicitly declare which environment state is frozen, cleared, allowed, or ignored.

Potentially relevant fields include:

```text
locale
language
character encoding
time zone
HOME / config roots
cache roots
temporary directory
PATH/tool resolution
threading variables
BLAS/OpenMP variables
GPU visibility
determinism flags
floating-point environment
library search paths
proxy/network variables
application-specific environment variables
```

The Neuro Workbench profile provides a strong concrete pattern by freezing `LANG`, `LC_ALL`, `TZ`, OpenMP settings, and private HOME/XDG/TMP roots.

The generic kernel should preserve the principle, not copy Workbench-specific values into every domain.

---

## 6. Platform and hardware boundary

### 6.1 Platform identity

The capsule must bind platform facts that can materially affect the computation under the selected profile.

Possible coordinates:

```text
OS/platform family
architecture / ISA
kernel/runtime ABI
GPU model/architecture
GPU driver
accelerator runtime
CPU feature requirements
endianness
libc/runtime family
```

### 6.2 Do not overfit identity to irrelevant hardware serials

The goal is scientific execution semantics, not host inventory for its own sake.

Machine serial number, hostname, physical rack, or every hardware identifier should not be part of scientific identity unless the scientific profile declares them relevant.

### 6.3 Cross-platform equivalence is not assumed

```text
same source + same inputs
+ different CPU/GPU/runtime
    != automatically same scientific execution semantics
```

A separate qualification may later establish bounded cross-platform numerical equivalence for a specific computation/profile.

Until then, platform differences remain explicit lineage.

---

## 7. Stochasticity and nondeterminism

A reproducibility architecture must distinguish deterministic and stochastic computations.

### 7.1 Stochasticity profile

Suggested classes:

```text
DeterministicDeclared
SeededPseudoRandom
RecordedEntropy
ExternalNondeterminismBound
NondeterminismUncontrolled
Unknown
```

These are descriptive execution semantics, not quality rankings.

### 7.2 Seeded pseudo-random execution

Bind at minimum:

```text
RNG algorithm/profile
RNG implementation/version when material
seed bytes/content identity
stream/substream policy
parallel RNG policy
```

A numeric seed without RNG identity is not enough.

### 7.3 Recorded entropy

If real entropy or external random values affect the computation and exact replay is intended, those realized values must become explicit input artifacts or execution-side evidence.

### 7.4 Uncontrolled nondeterminism

If scheduling, hardware atomics, network races, wall-clock sampling, external services, or other uncontrolled state can materially alter output, the capsule must not claim exact deterministic replay.

The correct state may remain:

```text
NondeterminismUncontrolled
```

rather than pretending a seed closes the execution boundary.

---

## 8. Parallelism and numerical nondeterminism

Thread count and accelerator execution can affect floating-point reduction order and therefore output bytes/numerics.

Relevant profile fields may include:

```text
thread count
threading runtime
OpenMP/BLAS policy
GPU deterministic-kernel policy
parallel reduction mode
compiler fast-math flags
FMA behavior
precision mode
```

The kernel must not assume:

```text
same algorithm == bitwise-identical parallel result
```

A domain may define toleranced semantic reproducibility separately from byte-identical replay.

---

## 9. External state and network access

### 9.1 Network access is scientific input when it affects computation

A computation that reads mutable remote state is not fully identified by local source/runtime inputs.

Examples:

```text
web pages
APIs
databases
remote model endpoints
remote object stores
live market/weather/sensor feeds
package registries
```

The profile must choose one of these strategies:

```text
network forbidden / hermetic
exact remote responses snapshotted and content-addressed
external interaction receipts retained
mutable external state explicitly uncontrolled
```

### 9.2 Service identity is not response identity

```text
API endpoint identity
    != exact response bytes
```

For replayable science, exact consumed response artifacts should normally be retained/content-addressed.

### 9.3 Authentication/secrets

Secrets may be required to access scientific inputs, but raw credentials should not be exposed in scientific receipts.

If a secret value itself affects computation, bind a safe opaque commitment or secret-input identity under an appropriate security boundary rather than serializing the secret.

Access authorization and scientific input identity remain separate.

---

## 10. Invocation identity

An execution receipt must bind how the program was invoked.

Depending on the execution model:

```text
argv
working directory semantics
stdin identity
entrypoint/function identity
request payload
selected subcommand
process environment profile
resource limits
sandbox policy
```

Shell command text is not equivalent to a structured argv vector.

Where possible, retain structured invocation semantics so quoting/shell expansion cannot alter meaning invisibly.

---

## 11. Time semantics

Time appears in several distinct roles:

```text
scientific/reference time
input data event time
execution attempt start/end time
wall-clock value read by the program
availability/custody time
preregistration time
```

These must not be collapsed.

### 11.1 Attempt timestamps

Execution start/end observations belong to occurrence provenance and audit.

They normally should not redefine the capsule identity unless the computation is explicitly time-dependent.

### 11.2 Program-observed time

If the program reads current time and that value affects scientific output, it is an input/nondeterminism source and must be frozen/recorded or declared uncontrolled.

### 11.3 Claimed timestamps are not verified chronology

SCI-003 does not authenticate clock sources or prove chronology merely because a receipt stores timestamps.

---

## 12. Resource limits, interruption, and failures

Failed executions are first-class scientific evidence about execution, not missing data to be silently discarded.

An attempt receipt should distinguish at least:

```text
CompletedExitSuccess
CompletedExitFailure
TimedOut
ResourceLimitExceeded
KilledOrCancelled
ExecutionUnavailable
OutputIncomplete
ReceiptIncomplete
```

A process may exit zero yet still fail the domain's scientific outcome contract.

A process may fail yet produce a valid, independently verifiable failure receipt.

This follows the NeuroBridge principle:

```text
verified incomplete observation
    != successful execution
```

---

## 13. Output binding

Outputs must be SCI-002 artifact identities bound to the exact execution attempt.

The receipt should distinguish:

```text
stdout/stderr
primary scientific outputs
intermediate outputs
logs/diagnostics
proof/witness artifacts
checkpoints
sidecars
```

Not every log is a scientific output, but output roles must be explicit.

Output identity alone does not establish that the producing execution was valid; execution ancestry must be retained separately.

---

## 14. Raw observation vs normalized interpretation

SCI-003 preserves:

```text
RawExecutionObservationRoot
    != NormalizedExecutionInterpretationRoot
```

A parser may normalize raw runtime/tool metadata into a scientific projection, but raw evidence must remain recoverable/auditable where the profile requires it.

The Neuro Nix lineage demonstrates why: unconsumed metadata may change the raw observation while leaving a deliberately normalized closure identity unchanged.

Normalization is a transformation and should eventually be SCI-002 receipt-bound.

---

## 15. Independent receipt verification

### 15.1 Producer cannot certify itself by assertion

A producer may perform internal consistency checks, but those do not replace a hostile-input verifier when the scientific authority model requires independent reconstruction.

### 15.2 Verifier responsibilities

Depending on the profile, an independent verifier may reconstruct:

```text
capsule identity
input membership
invocation
sidecar digests
output identities
exit state
runtime closure
profile conformance
forbidden extra files
canonical serialization
producer implementation identity
normalizer/transform identity
```

### 15.3 Verifier implementation identity

The verifier itself is an artifact/tool dependency and must have exact identity in any authority-bearing qualification receipt.

Hashing verifier A while executing verifier B must fail closed.

---

## 16. Reproducibility classes

SCI-003 should avoid one boolean `reproducible`.

Illustrative scoped classes:

```text
ExactByteReplayExpected
DeterministicWithinDeclaredProfile
NumericallyEquivalentWithinDeclaredTolerance
StochasticProcessSpecified
ExternalStateSnapshotted
ExecutionRecordedButReplayGuaranteeNotEstablished
NondeterminismUncontrolled
```

These classes require domain-specific evidence to issue. The capsule itself does not automatically grant them.

### 16.1 Exact replay vs scientific equivalence

```text
byte-identical output
    != scientifically equivalent output

scientifically equivalent output
    != byte-identical replay
```

For floating-point/scientific computations, a domain may care about toleranced semantic equivalence rather than exact bytes—but the tolerance and comparison method must be preregistered/versioned and separately qualified.

---

## 17. Execution identity vs experiment identity

One experiment may contain multiple execution capsules/attempts.

Examples:

```text
training runs
calibration run
structural holdout evaluation
multiple seeds
replicates
negative controls
placebo runs
```

Therefore:

```text
execution capsule
    != experiment contract
    != experimental observation
```

SCI-004 owns the higher-level experiment contract and must bind its required SCI-003 execution profiles/capsules explicitly.

---

## 18. Execution identity vs verification identity

A verifier execution is itself a scientific/tool execution and may use a SCI-003 capsule.

However:

```text
verifier execution occurred
    != verifier outcome positive
```

#786 should therefore retain:

```text
verification target identity
verification method identity
verifier execution capsule/receipt
verification output/witness identity
admission/completeness state
verification outcome
```

rather than collapsing these into one `verified: bool`.

---

## 19. Execution identity vs action/effect authority

SCI-003 is scientific provenance infrastructure.

It must not issue:

```text
actuation permit
hardware capability
network execution authority
self-modification permission
deployment authorization
safety approval
```

Even a fully verified scientific execution receipt remains evidence, not an effect capability.

---

## 20. Proposed shared vocabulary

Names are illustrative; semantics are normative.

```text
ScientificExecutionProfileV1
ScientificExecutionCapsuleV1
ExecutionStochasticityProfileV1
ExecutionPlatformProfileV1
ExecutionEnvironmentProfileV1
ExecutionInvocationV1
ScientificExecutionAttemptReceiptV1
VerifiedScientificExecutionReceiptV1
ScientificReplayAssessmentV1
```

Do not require every domain to use one concrete runtime technology.

Nix, containers, bare-metal solvers, WASM, Lean, Z3, GPU jobs, and remote services may all need adapters.

---

## 21. First implementation tranche

The first shared implementation should remain non-authorizing and small.

Suggested slice:

```text
ScientificExecutionProfileV1
ScientificExecutionCapsuleV1
ExecutionStochasticityProfileV1
```

with SCI-002 artifact identities for all bound components.

Required properties:

1. profile-controlled required slots;
2. explicit domain/profile identity;
3. exact source/program identity;
4. exact input snapshot identity;
5. exact configuration identity;
6. exact runtime/toolchain identities when required;
7. exact environment/platform profile identities when required;
8. explicit stochasticity state;
9. no execution-occurrence claim;
10. no result/qualification/action authority.

The first tranche should not execute anything.

---

## 22. Second implementation tranche

Pilot a raw execution-attempt receipt with one already-well-understood deterministic executable path.

Preferred candidate:

```text
Ramanujan / formal-verification helper
```

or another small deterministic scientific executable where inputs, argv, outputs, toolchain, and runtime are easier to close than Workbench.

The Workbench lineage should remain the stronger reference implementation for full runtime-closure semantics; generic SCI-003 must not claim equivalent closure strength unless it actually reproduces that theorem.

---

## 23. Third implementation tranche

Add a hostile-input verifier that reconstructs the attempt receipt independently.

Only afterward may a domain-specific policy issue a `QualifiedScientificExecution` wrapper for a declared scientific use.

The qualification wrapper should remain private-fielded and non-deserializable into live authority.

---

## 24. Migration policy

Existing domain execution receipts remain domain-native.

SCI-003 does not retroactively qualify or replace:

- Neuro Workbench execution-profile/closure/capture/verifier artifacts;
- Matter relativistic solver receipts;
- Physical Agency strict simulation receipts;
- historical benchmark run manifests;
- domain-specific experiment/result receipts.

Migration requires a typed adapter that proves the shared capsule captures every execution-relevant semantic coordinate required by the domain-native contract.

If the generic profile is weaker, the domain-native identity remains authoritative for that domain.

---

## 25. Adversarial requirements

Future implementation should include at least:

### Identity substitution

- source/program substitution changes capsule identity;
- input substitution changes capsule identity;
- configuration substitution changes capsule identity;
- runtime-closure substitution changes capsule identity;
- toolchain substitution changes capsule identity;
- platform/profile substitution changes capsule identity;
- RNG algorithm/seed substitution changes stochastic execution identity.

### Ambient-state leakage

- forbidden undeclared environment variable access fails or marks capsule non-hermetic;
- mutable network input without snapshot/receipt cannot claim hermetic replay;
- wall-clock-dependent execution cannot claim deterministic replay unless time is bound;
- unbound external tool/service use cannot remain invisible.

### Receipt integrity

- valid capsule + forged success outcome cannot create verified execution;
- output artifact substitution fails;
- stdout/stderr sidecar tamper fails where bound;
- extra unbound files fail under closed-receipt profiles;
- missing sidecars fail or produce explicit incomplete state;
- producer-supplied `verified=true` cannot mint verified receipt;
- verifier implementation substitution fails.

### Failure preservation

- timeout is retained distinctly from exit failure;
- failed run can be independently verified as failed;
- incomplete receipt cannot be promoted to complete execution;
- process exit zero does not imply domain outcome success.

### Authority separation

- execution receipt cannot deserialize/convert into scientific qualification without the qualifier;
- qualified scientific execution cannot become experiment success, replication, safety, or action authority by convenience conversion.

---

## 26. Relationship to current Neuro lineage

SCI-003 should preserve, not weaken, these NeuroBridge theorems:

```text
profile != realized closure
raw observation != normalized closure
producer != independent verifier
verified receipt != qualified execution
execution capsule != scientific transform result
```

The Neuro lineage is therefore a reference implementation of stronger concrete semantics, not a qualification source for SCI-003.

---

## 27. Relationship to SCI-002

SCI-003 consumes SCI-002 identities for all immutable artifacts it binds.

It must never use mutable path/name strings as the sole scientific identity of source, input, config, executable, model, or output.

The capsule identity itself should be built from a versioned canonical SCI-002 composite identity profile.

---

## 28. Relationship to SCI-004

SCI-004 `ExperimentContractV1` will specify:

```text
which execution profiles/capsules are admissible
which inputs are frozen
which interventions/treatments occur
which outputs/measurements are evaluated
which stopping/reveal rules apply
```

SCI-003 establishes execution lineage only. SCI-004 establishes prospective experiment semantics.

---

## 29. Relationship to #786

A future verification receipt should bind both:

```text
verification target / proof artifact identities      [SCI-002]
verifier computation identity / attempt evidence      [SCI-003]
```

and then separately retain method/outcome/admission semantics.

This prevents:

```text
"Lean accepted"
```

from standing in for the full chain:

```text
exact theorem
+ exact generated proof artifact
+ exact Lean/Mathlib/runtime capsule
+ exact verifier execution
+ admission-free outcome
+ target binding
```

---

## 30. Dependency order

```text
SCI-001 audit
    -> SCI-002 artifact identity architecture
    -> future SCI-002 strict verified-content implementation
    -> SCI-003 execution capsule architecture
    -> SCI-003 non-executing capsule implementation
    -> one raw attempt receipt pilot
    -> hostile-input receipt verifier
    -> domain-specific scientific execution qualification
    -> SCI-004 experiment contract
```

#786 verification authority may consume SCI-003 once its own verifier-soundness prerequisites qualify.

---

## 31. Review boundary

Review SCI-003 on this question:

> Does this contract identify the executable scientific computation precisely enough to support reproducibility and later verification without confusing declared environment, realized runtime, execution occurrence, successful output, reproducibility, scientific qualification, or action authority?

A positive architecture review does not qualify a runtime, solver, experiment, or scientific result.
