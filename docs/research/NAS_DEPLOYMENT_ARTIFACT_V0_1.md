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


## 2026 comparator specifics and the actual gap

### NATS-Bench-MCU v2

The authoritative corrected release contains 15,625 topology candidates: 14,281 physically profiled nRF5340 deployments and 1,344 flash-infeasible candidates. The INT8 flatbuffer limit is 746 KiB after the fixed 277.6 KiB firmware overhead within 1,024 KiB application-core flash. The tabular export includes hardware metrics, accuracies, failures, coverage and an INT8/FP32 ranking study. The correction overlay is part of the benchmark contract; do not mix v1 taxonomy or counts with v2.

Source: https://zenodo.org/records/21644470

### ConNAS (ICLR 2026)

ConNAS reports feasible NATS-Bench architectures within 0.14% of the optimal feasible architecture, and up to 1.55 percentage points higher accuracy than handcrafted architectures in one practical constrained deployment. This is the appropriate optimization-quality baseline on a matched single-target problem.

Source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7c9f851081499ed5c60c51e312d3235f-Abstract-Conference.html

### EdgeVolution (Communications Engineering, 2026)

EdgeVolution is already an end-to-end hardware-in-the-loop pipeline with functional binary deployment. The paper reports 4,600 evaluated architectures per run, with three repeated runs; the physical hardware-in-the-loop step accounts for less than 1% of total runtime, while training accounts for approximately 99%. It reports measurements on nRF5340, nRF52840 and nRF52833, with an STM32H753ZI extension. Its proxy ablation shows operation count can miss the high-accuracy/high-energy region of the Pareto front and fail to distinguish deployability.

This tells us not to position physical measurement itself as novel. The bigger optimization opportunity is reducing training and conversion overhead without letting estimated feasibility become a false deployment guarantee.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC13291272/

### ENAS (SuRE 2026 / arXiv 2026)

ENAS reports 636 trained and deployed models across two tasks, eight MCU platforms and nine image resolutions, with zero deployment-pipeline failures in its reported evaluation. It replaces repeated measured feasibility checks with static analytical RAM/flash/MACC estimation, reducing per-candidate evaluation time from approximately 5–10 minutes to approximately 30 seconds. It reports 2.41× and 1.70× search-time speedups over the greedy CPU-only NanoNAS baseline, and about 0.38× peak activation RAM at matched accuracy, with higher flash and MACC. Mean accuracy reductions versus NanoNAS were 0.79 and 1.31 percentage points on the two datasets; selected configurations also showed task- and device-specific gains.

These are paper-reported results, not independently reproduced by this repository. Static estimation is useful for triage, but it is not a substitute for build and physical evidence on a candidate promoted as deployable.

Source: https://arxiv.org/abs/2609.30272

## Proposed synthesis: confidence-bounded staged qualification

The next search policy should use a sequence of progressively more expensive fidelities:

1. Static structural and memory lower/upper bounds.
2. Cheap analytical resource estimates with calibrated uncertainty.
3. Partial training or a validated learning-curve forecast.
4. Exact quantization and compiler build.
5. Physical flash/deploy plus readback.
6. Repeated physical latency/energy measurement on the target fleet.

A cheap stage may reject only when a conservative bound proves a hard constraint impossible. An uncertain result escalates; it is not imputed as feasible or infeasible. Build and deployment receipts gate promotion. A success-driven quota should attempt candidates in objective-aware order until it has a preregistered count of fully verified deployable artifacts, rather than assuming the top-k architecture predictions will all build.

Use existing vector-resource budgets, cost-aware value of information, anytime confidence ledger, run-wide pruning risk ledger and CAS reservation foundation as the scheduling substrate. Every stage's acquisition cost must be charged to the same experiment ledger. This should be compared against random search, ENAS-style random/top-K/mutation, ConNAS, and direct EdgeVolution-style hardware-in-the-loop evaluation on matched tasks and targets.

Primary hypotheses to test:

- reduce end-to-end search cost per verified deployable artifact by at least 2× against the strongest matched baseline;
- reduce worst-target Pareto regret by at least 10% at matched total resource budget;
- preserve zero accepted false-feasible candidates in adversarial negative controls;
- keep every physical score bound to the exact artifact, target, firmware/runtime and measurement protocol;
- show confidence calibration, failure recovery and deployment quota completion rates separately from model accuracy.

These are preregistered targets for future measurement, not results from this PR.
