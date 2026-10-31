# Fleet-robust multi-device deployment envelopes v0.1

Tracking: [NAS-FLEET-001 / #9511](https://github.com/Luminous-Dynamics/symthaea/issues/9511)

Implementation branch: research/nas-fleet-envelope-v0.1

## 1. Motivation

Most hardware-aware NAS results are target-specific. A model can be excellent on one MCU and infeasible or slow on another. Mean accuracy, mean latency or a single weighted fitness score can conceal the exact device that breaks deployment.

This module changes the decision unit from "best architecture for one board" to "architecture that satisfies a frozen set of required deployment targets."

It is a conservative tabular evidence evaluator. It does not claim generalization to unseen boards, different firmware/compiler builds, environmental shifts or new production lots.

## 2. Frozen fleet manifest

The manifest binds:
- fleet ID and benchmark ID;
- the complete required target set, strictly sorted by target ID;
- exact hardware-profile SHA-256 per target;
- exact objective-registry SHA-256 per target;
- common objective IDs and coordinate order;
- common physical unit, direction and raw normalization range per objective;
- a hard minimum normalized utility for each objective;
- domain-separated canonical fleet digest.

Each target ID must equal its exact HardwareProfileManifest.profile_id. Every objective must be attached to that target's exact hardware profile. All target registries must contain exactly the same objective set, with identical units, directions and normalization ranges. Per-device protocols, cost models, firmware and instrument profiles may differ but are pinned by each target registry/profile digest.

## 3. Three distinct evidence states

For each required target × objective pair, the existing ObjectiveObservation / score_objective_observation contract is reused.

- MEASURED: one finite raw value, normalized under the common pre-registered range, with replicate, sample count, source artifact and acquisition-cost provenance.
- EXPLICITLY_INFEASIBLE: a known hard deployment failure, e.g. flash budget exceeded. This makes the fleet candidate infeasible.
- NOT_MEASURED / FAILED / missing: incomplete evidence. This never becomes zero, never counts as a successful deployment, and cannot produce a fleet-feasible label.

Candidate ID and exact configuration SHA-256 must be identical across every target. Duplicate target/objective observations, unknown profiles, wrong registries, missing objective definitions and configuration substitution are rejected.

## 4. Worst-case aggregation, not averaging

For fully measured candidates, each objective envelope reports:
- lowest normalized utility and its worst target;
- highest normalized utility and its best target;
- cross-target spread;
- required minimum utility;
- whether every target clears the hard floor.

The fleet objective is the minimum normalized utility across all required targets. A candidate is fleet-feasible only if every required target is measured and every per-objective worst-case utility meets its hard threshold.

If an objective's mean passes while its worst target fails, the fleet result is INFEASIBLE. There is no weighted-average override. A known infeasible target remains infeasible even when another target's evidence is incomplete.

## 5. Evidence receipts and verification

- Target/objective evidence is sorted canonically before its domain-separated SHA-256 digest.
- The evaluation binds fleet manifest digest, candidate/configuration, complete evidence vector, status, hard constraint violations, incomplete reasons, and objective envelopes.
- verify_fleet_candidate_evaluation recomputes the result from original observations and rejects a caller who changes the status and recomputes the outer digest.
- Digest integrity is not proof of authorship, wall-clock chronology, instrument truth or independent board sampling. Preserve original evidence artifacts and use external custody/attestation where required.

## 6. Tests

Twelve Rust unit tests are authored for:
- worst-target aggregation overriding a passing fleet mean;
- hard-floor violation on one target;
- incomplete required-target coverage;
- explicit hardware infeasibility without scalar imputation;
- Failed/NotMeasured evidence remaining incomplete;
- duplicate target/objective pair rejection;
- candidate-configuration replay rejection;
- profile/registry mismatch rejection;
- inconsistent normalization rejection;
- evaluation digest and independent recomputation;
- sorted manifest and floor validation.

The current authoring environment lacks cargo, rustc and rustfmt; these tests have not been compiled or executed.

## 7. Industry and research baselines

- EdgeVolution, Communications Engineering (23 June 2026): end-to-end hardware-in-the-loop multi-objective NAS, directly measuring latency and power across nRF5340, nRF52840 and nRF52833, with an STM32H753ZI extension. The paper reports 4,600 sampled/evaluated architectures per optimization run, repeated three times; training was approximately 99% of total optimization time and hardware measurement less than 1% for its keyword-spotting use case. Its ablation found operation-count proxy search missed the upper accuracy/energy Pareto region. https://pmc.ncbi.nlm.nih.gov/articles/PMC13291272/
- ENAS, arXiv (10 Aug 2026; SuRE Workshop @ IJCAI 2026): CPU-only random → top-K → mutation NAS evaluated across eight MCU targets. The authors report mean search-time speedups of 2.41× and 1.70× on two TinyML tasks and lower peak activation RAM at matched accuracy. Treat these as paper-reported results pending independent replication. https://arxiv.org/abs/2609.30272 ; https://github.com/EdgeIntelligenceLab/ENAS
- ConNAS, ICLR 2026: gradient modification for hard hardware constraints; reports results within 0.14% of optimal feasible NATS-Bench accuracy under its tested conditions. It is a single-search/task comparison, not a fleet-envelope metric. https://proceedings.iclr.cc/paper_files/paper/2026/hash/7c9f851081499ed5c60c51e312d3235f-Abstract-Conference.html
- NATS-Bench-MCU v2, Zenodo version 2.0 (28 July 2026): 15,625 topologies; 14,281 physical nRF5340 profiles and 1,344 flash-infeasible candidates. The publication export contains 14,281 rows in hw_metrics.csv, 14,281 accuracy rows, 1,344 failure rows, and an explicit v2 corrections overlay. https://zenodo.org/records/21644470

The fleet envelope should not claim to beat these papers from source code alone. First reproduce their protocols, then compare on a shared fleet/task/constraint set.

## 8. Benchmark design

Initial sequence:
1. Reproduce the full NATS-Bench-MCU v2 nRF5340 tabular envelope and its hard 746 KiB INT8 flatbuffer limit.
2. Add other targets where actual measurements and protocol metadata can be acquired: EdgeVolution's nRF52840/nRF52833/STM32H753ZI and ENAS's eight-target set.
3. Evaluate one fleet-robust architecture against per-device oracle and per-device optimized solutions.
4. Run prospective physical replications on independently sampled boards and a second firmware/compiler/quantization configuration before claiming device generalization.

Report:
- worst-target accuracy and normalized utility per objective;
- number/fraction of required targets deployable;
- false-feasible and false-infeasible rates;
- fleet Pareto hypervolume and worst-target Pareto regret;
- per-target flash, ROM, SRAM/tensor arena, latency and energy;
- physical measurement count and complete vector cost to a fixed fleet-regret threshold;
- generalization under compiler, runtime, quantization, temperature, voltage, workload and sensor shifts;
- mean and worst-case results side by side, never only one aggregate.

## 9. Aspirational acceptance targets (not achieved)

- 100% required target × objective evidence coverage before a fleet-feasible label.
- Zero false-feasible labels in adversarial missing/infeasible/failure controls.
- No required target below its hard threshold.
- At a matched fleet budget, at least 10% lower worst-target regret or 10% lower vector-resource cost than the strongest fleet-matched baseline.
- No more than 2 percentage points worst-target task-accuracy loss versus the best per-device oracle, unless a different fleet policy is preregistered.

These are go/no-go targets, not observed performance.

## 10. Next research extensions

- Pre-registered lower-tail/CVaR risk objectives over independent board/device samples, without weakening hard feasibility.
- Environmental scenario profiles (temperature, voltage, clock/thermal state) and workload profiles.
- Joint architecture + quantization + compiler/runtime/operator-mapping co-search.
- Transfer learning between hardware profiles, with explicit out-of-distribution abstention.
- Active fleet profiling: measure the target device/scenario with greatest expected reduction in worst-case fleet regret.
