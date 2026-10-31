# Exact deployment-artifact identity and measurement binding v0.1

Tracking: NAS-ARTIFACT-001 / #9594  
Joint co-design: #9515  
Fleet envelope: #9511

## Thesis

The object being optimized and compared must be the deployable artifact, not an architecture description or a proxy score. A candidate is a complete chain from graph and quantization through compiler/runtime and firmware to the exact model and firmware binaries that are subsequently measured.

This module creates a fail-closed, typed evidence boundary for that chain.

## Pipeline manifest

The v0.1 pipeline manifest requires exactly one digest for each sorted component:

- architecture graph;
- quantization specification and calibration dataset;
- compiler toolchain and compiler flags;
- operator library and memory planner;
- firmware and runtime;
- board support package and linker script;
- build environment.

It also binds candidate ID/configuration SHA-256, benchmark ID, target ID, exact HardwareProfileManifest digest and exact ObjectiveRegistry digest.

Cross-validation requires compiler toolchain, quantization, firmware and runtime component digests to equal the exact hardware profile. The target profile must also exist inside the objective registry. No silent fallback to a nearby profile or toolchain is allowed.

## Build receipt and artifact identity

The build receipt is a separate object to avoid a hash cycle. It binds the pipeline digest and one disposition:

- BUILT requires digests for the model binary, firmware binary, build log, linker map, flash report and RAM report.
- FAILED requires a reason and build log, and may not claim successful outputs.
- NOT_ATTEMPTED requires a reason and carries no build/output digests.

Only BUILT can produce a non-serializable in-process VerifiedDeployableArtifact capability. Its domain-separated identity commits to candidate/configuration, target profile, objective registry, pipeline, build receipt and both produced binary digests.

The capability cannot be deserialized or caller-constructed outside the module.

## Measurement binding

Each ObjectiveObservation must match the verified candidate ID, configuration SHA-256, target hardware profile and objective registry. The objective scorer is rerun. The resulting measurement receipt binds the objective observation and its source artifact digest to the exact deployable-artifact identity.

Rehashing a forged receipt does not make a candidate/config/profile substitution valid: the verifier checks the entire scope against the verified artifact and rescoring registry.

## Falsification tests

Thirteen Rust unit tests are authored for stable artifact identity, missing components, duplicate/unsorted components, compiler/profile mismatch, target/registry substitution, failed/unattempted build rejection, altered binary digest, missing resource-report evidence, exact measurement binding, rehashed candidate substitution, wrong-target observations, unknown JSON fields, and build/pipeline mismatch.

The tests are authored, not compiled or executed in the current authoring environment because cargo, rustc and rustfmt are unavailable.

## Claim ceiling

These are content commitments and deterministic structural checks. A SHA-256 digest does not authenticate who built an artifact, prove the measurement operator's identity, prove that a physical board was flashed with the stated binary, or prove instrument calibration and physical measurement truth.

The v0.1 receipt is not a signed attestation and does not perform secure-boot readback. Physical deployment evidence needs a hardware-in-the-loop runner that records device identity, flash/readback evidence, reset/session boundaries, raw acquisition files, instrument calibration state and independent custody. That work must remain a separate evidence tier.

## Benchmark programme

First, reproduce NATS-Bench-MCU v2 on its exact nRF5340 firmware/quantization/protocol and corrected 14,281-measured / 1,344-flash-infeasible partition. Preserve the 746 KiB INT8 flatbuffer deployment limit and keep measured failure categories distinct.

Then compare against:

- EdgeVolution's physical hardware-in-the-loop, multi-objective search and end-to-end binary deployment across nRF5340, nRF52840 and nRF52833, with STM32H753ZI extension.
- ConNAS on NATS-Bench under identical feasible-search constraints.
- ENAS's CPU-only random → top-K → mutation baseline, with its 2026 preprint results independently reproduced.

Report both the architecture-only and complete-artifact baselines. Core metrics are:

1. artifact build success and physical deployment success, by target;
2. false-feasible rate, defined as predicted feasible but build/deploy/measurement proves infeasible;
3. exact artifact identity coverage: fraction of scored physical objectives linked to one verified artifact digest;
4. flash, static RAM, tensor arena, stack and heap with separate measured/estimated labels;
5. measured task accuracy, latency distributions, energy per inference and tail latency;
6. worst-target fleet utility, fleet Pareto regret and per-target deployment fraction;
7. vector measurement/build/deployment cost to a preregistered fleet-regret threshold;
8. rebuild reproducibility across clean environments and explicit compiler nondeterminism.

Aspirational gates, not achieved results:

- 100% of scored physical observations carry an exact artifact binding.
- Zero false-feasible outcomes in the adversarial verifier suite.
- Zero promotion from FAILED or NOT_ATTEMPTED build dispositions.
- 100% required component coverage before an artifact is labelled built.
- At matched task, target, objective and total resource budget, at least 10% lower worst-target Pareto regret than the strongest replicated fleet-matched baseline.
- No more than 2 percentage points of worst-target accuracy loss against the per-device oracle unless preregistered otherwise.

Do not collapse all metrics into one leaderboard scalar. Show accuracy/latency/energy/memory and build reliability side by side. A result is only comparable when dataset split, training budget, quantization, compiler, firmware, runtime, target board, acquisition protocol and resource budget are all matched.
