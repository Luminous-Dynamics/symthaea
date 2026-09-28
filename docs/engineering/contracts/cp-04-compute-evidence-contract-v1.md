# CP-04 Compute Evidence Contract v1

## Purpose
CP-04 defines a bounded evidence contract for compute configurations used by Symthaea engineering and CIV service projections.

Canonical thread:
`engineering demand -> representation -> model -> runtime -> accelerator/configuration -> deployment artifact -> execution context -> observation -> uncertainty/statistics -> engineering disposition`

This composes existing BinaryHV/HDC, CfC/LTC, software verification, artifact, and operational-provenance owners. It creates no replacement model registry, benchmark database, hardware registry, deployment authority, or operational ledger.

## Claim ceiling
This contract establishes only deterministic software semantics for binding a compute claim to declared representation, model, runtime, accelerator, deployment, execution, observation, uncertainty/statistics, and provenance references.

It establishes **no** hardware performance, accelerator correctness, compiler correctness, model validity, benchmark generalization, production availability, safety certification, security guarantee, capacity guarantee, deployment authorization, or physical execution authority.

Synthetic qualification establishes only deterministic compute identity, generation, dependency, evidence-separation, replay, negative-result, and authority semantics over synthetic/reference workflows. A synthetic PASS never authorizes or demonstrates physical execution.

## Canonical ownership and seams
- BinaryHV representation semantics: existing HDC/BinaryHV owners; SYM-FV-001A/002/003/005 remain authoritative for their proof subjects.
- CfC/LTC model semantics and temporal state: existing temporal/CfC owners; #3587 is an explicit exact-state boundary.
- Requirement/verification: ENG-DESIGN / SE-VV.
- Model identity/version/training artifact: SE-MODEL / model owner.
- Runtime/compiler/toolchain: software/build/deployment owners.
- Accelerator/device identity/configuration: compute/HDL/hardware owners.
- Deployment artifact: build/release/deployment owners.
- Physical observations: SE-OBS / FIELD or domain observation owner.
- Benchmark/statistical interpretation: relevant evaluation owner.
- Custody/provenance/coordination: Mycelix where applicable.
- Service availability/continuity: CIV-Service where applicable.

CP-04 must not invent replacement BinaryHV, CfC, model, hardware, observation, or operational identities.

## Core distinctions
`representation != model != model parameters != runtime != compiler/toolchain != accelerator != deployment artifact != execution != observation != engineering disposition != operational authority`

`capability != availability != capacity != execution authorization`

`declared benchmark result != physical execution evidence`

Model output is not physical observation. A benchmark score is not a population claim unless sampling and statistical assumptions are independently satisfied.

## Compute evidence subject
A claim-bearing subject should bind exact immutable identities for:
- compute subject and generation
- representation
- model and model parameters/checkpoint
- runtime and runtime configuration
- compiler/toolchain
- accelerator and accelerator configuration
- deployment artifact
- execution context
- observations
- uncertainty/statistics
- dependencies
- currentness and applicability
- authority disposition
- claim ceiling

Never resolve a claim through mutable “latest runtime”, “current model”, or “available accelerator” aliases.

## Generation/currentness law
A new semantic generation is required when an evidence-relevant dependency changes, including representation encoding/width; model architecture/checkpoint/parameters/preprocessing; CfC/LTC state semantics or snapshot profile; compiler/toolchain or relevant flags; runtime/kernel/driver/numerical library/scheduler; accelerator identity/firmware/memory/execution mode; deployment bytes; precision/quantization; I/O transformation; benchmark harness; workload/profile/applicability; or execution isolation/environment.

Historical executions and observations remain attached to their original generation. Requalification derives a new disposition; it never rewrites history.

## Dependency graph and invalidation
The compute evidence thread is represented by the versioned directed graph used by the independent qualifier. Each edge means that a change to the upstream identity can invalidate downstream derived evidence; it does not rewrite or invalidate upstream historical records.

Canonical graph order:
`requirement -> representation -> model -> runtime -> accelerator -> deployment -> execution -> observation -> statistics -> engineering disposition`

The graph identity is part of replay identity and the qualification receipt. The qualifier rejects missing required edges, dangling nodes/edges, duplicate nodes, cycles, or graph identities that differ from the registered CP-04 graph. A deployment change therefore invalidates deployment and downstream execution/observation/statistical/disposition evidence, but does not retroactively change the historical model or representation identity. Graph changes themselves require a new graph generation and requalification.

## Closed evidence thread
`requirement -> representation -> model -> runtime/toolchain -> accelerator/config -> deployment -> execution context -> observation -> uncertainty/statistics -> engineering disposition`

Missing links create explicit unresolved/narrowed states. Model+runtime without deployment identity is deployment-unresolved; deployment without execution observation is observation-missing; observation without required uncertainty is insufficient; one benchmark run is bounded run evidence, not population inference; synthetic execution is semantic/replay evidence only.

## Independence/common mode
Potential common-mode roots include benchmark harness, timing source, host clock, compiler-generated kernel, runtime, accelerator, firmware/driver, input generator, reference implementation, telemetry pipeline, and preprocessing/postprocessing.

`numeric agreement != independent evidence`

A differently labeled channel sharing a causal measurement root cannot be promoted to independent confirmation.

## Model versus observation
Keep separate:
1. declared model;
2. model prediction/output;
3. execution observation;
4. discrepancy/residual;
5. engineering interpretation.

Prospective temporal prediction must preserve the exact model/input snapshot and outcome-blind commitment boundary where applicable. #3587 demonstrates why a state read/write API cannot be assumed to provide exact multilayer restoration.

## Performance/statistical boundaries
A benchmark result binds exact workload/input corpus and generation, model/deployment generation, runtime/toolchain, accelerator/configuration, harness, warm-up and measurement protocol, timing source, sample count, aggregation/statistical method, exclusions/failures, and applicability envelope.

Changing workload, harness, runtime, accelerator, or statistical profile requires requalification or explicit applicability rebinding. A mean can conceal hard tail failure; envelope requirements must check the envelope directly.

## Negative evidence
Negative evidence is immutable and first-class: timeout, out-of-envelope latency, numerical divergence, missing deployment, runtime/compiler mismatch, accelerator mismatch, stale driver/firmware, unsupported instruction path, insufficient memory, non-exact state snapshot, common-mode telemetry, missing workload coverage, unavailable physical execution, or attestation without performance evidence.

`negative evidence != no evidence`

Recomputation cannot delete, overwrite, or silently reinterpret historical failure.

## Provenance and authority
Mycelix provenance/attestation can bind identity, custody, coordination, and operational facts; it cannot manufacture compute performance evidence.

CIV-Service can project separately qualified compute capability into availability/continuity; a service event cannot create or upgrade engineering evidence.

Engineering qualification cannot grant production execution authority. Physical execution remains behind hardware, OS/runtime, HAL, operator, commissioning, and safety boundaries.

## Replay semantics
Replay identity uses immutable case order and exact representation/model/runtime/toolchain/accelerator/deployment/execution/observation/statistical/dependency/currentness/applicability identities. Derived dispositions are excluded from replay identity. Equivalent immutable inputs must yield equivalent dispositions.

## Adversarial requirements
The qualifier must reject/narrow:
- model output numerically equal to measurement;
- changed model parameters under the same human-readable name;
- changed runtime/compiler/driver under the same deployment label;
- changed accelerator configuration;
- changed deployment bytes;
- one benchmark run presented as population evidence;
- mean-positive result with hard tail failure;
- correlated channels presented as independent;
- missing execution observation;
- missing uncertainty/statistics;
- stale currentness;
- lossy CfC snapshot presented as exact;
- operational event presented as qualification;
- attestation presented as performance evidence;
- synthetic PASS presented as physical qualification;
- disappearance of negative evidence after recomputation.

## Design review gates
1. What does CP-04 own versus reference?
2. Is every representation/model/runtime/accelerator/deployment identity exact?
3. What changes force a new generation?
4. Is prediction separated from observation?
5. Are workload, harness, uncertainty, and tail requirements explicit?
6. Which measurements share common-mode roots?
7. Does negative evidence survive recomputation?
8. Can provenance or service events accidentally promote engineering claims?
9. Can the result replay from immutable references?
10. What does PASS prove, and what does it explicitly not prove?

## Implementation boundary
Initial qualification is a pure stdlib/reference-data oracle. It must not import production compute code and must not claim physical hardware execution. Later production adapters may bind exact BinaryHV/CfC/model/runtime/accelerator/deployment artifacts only after independent qualification.

## Related existing work
- BinaryHV formal refinement: #5713, #5716, #5718 and related SYM-FV work.
- CfC exact inference-state concern: #3587.
- Prospective LTC/CfC engineering prediction: #6201.
- Existing software/formal VV owners remain authoritative.
