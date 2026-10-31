# HARDSEC-017 — Matched Multi-Platform Hardware Attack Detection Benchmark

**Status:** benchmark protocol, machine-readable plan and executable Rust plan auditor authored; no benchmark campaign or detector result is claimed  
**Issue:** [#9468](https://github.com/Luminous-Dynamics/symthaea/issues/9468)  
**Related:** HARDSEC-004 (#9060 / #9063), HARDSEC-012 (#9346 / #9410), HARDSEC-014 (#9413 / #9416), HARDSEC-016 (#9463)

## 1. The research shift: from accuracy to security operating risk

A high random-window accuracy is not the same thing as an operationally dependable detector. A defensible result must answer:

1. Did the malicious microarchitectural effect actually occur?
2. Was the tested binary, attack implementation, execution and device genuinely held out?
3. Did counter collection change the phenomenon being measured?
4. At the same benign false-alert budget, how much successful attack effect is detected?
5. What is the worst attack family, adaptation strategy and platform—not only the pooled average?
6. How quickly is the effect detected, and at what acquisition, compute, memory, bandwidth and energy cost?
7. Does the monitor fail closed when its sensors are missing, stale, misaligned, multiplexed or untrusted?

The benchmark therefore treats the evidence chain and operating point as first-class outputs. Accuracy remains a descriptive metric, not the primary security claim.

## 2. What the current literature actually establishes

### Physical XuanTie C910 / HARPY-V

Pou et al. (2026) report 99% detection precision for benign and malicious samples and about 95% malicious-sample classification on a real XuanTie C910. Their collection uses 10 ms windows. Crucially, they found that collecting more counters concurrently could perturb some measured counter values by more than 300%, so they collected each counter in a separate run. Their two multi-process tests (one attack with one benign application, or one attack with three benign applications) report 98.56% after normalizing counters by retired instructions; raw-count accuracy falls to 60% in that setup. The same paper lists a single CPU model, no cross-architecture evaluation and no adaptive attack variants as limitations. These results are a strong physical reference, not a universal transfer guarantee.

Source: [Pou et al., Cybersecurity 9, 222 (2026)](https://doi.org/10.1186/s42400-026-00662-8).

### HARPY-V-GEM5 and HARPY-V-GEM5-100c

Pujol et al. (2026) report known-workload detection above 99.5% with HPC features and 99.0% with instruction features on a 32-program GEM5 corpus. Their leave-one-program-out analysis is much more informative for generalization: HPC-based random forest is reported at 90.08% aggregate LOPO accuracy, with 25–36% standard deviations across programs and near-zero results for several held-out variants. The authors also report a 21.6 percentage-point gap between HPC-RF training accuracy (99.55%) and a separate LOPO mean figure (77.95% in their overfitting analysis); those metrics arise from different tables/analyses and must not be conflated. Some instruction-only and HPC-only failures are complementary, suggesting a hybrid detection cascade. Each LOPO test fold contains one held-out program and therefore only one class; ordinary per-fold precision/F1/MCC are not valid mixed-class measures. GhostWrite's natural trace was extended by repeated iterations; those repeated windows remain the same lineage, not additional independent programs.

Sources: [Pujol et al., Neural Computing and Applications (published 27 Oct 2026)](https://doi.org/10.1007/s00521-026-12467-4), especially the [LOPO protocol and results](https://link.springer.com/article/10.1007/s00521-026-12467-4).


A separate 2026 catalogue record identifies **HARPY-V-GEM5-100c** (DOI [10.34810/DATA3542](https://doi.org/10.34810/DATA3542)): 24 side-channel attack implementations and 23 benign workloads, simulated on a single-core RISC-V gem5 O3 CPU with two cache configurations and 4-/8-wide pipeline variants. It is a separate corpus/configuration family, not an extra fold of DATA2711. The B2FIND metadata lists CC0/open-access rights, but archive bytes, exact file inventory, and checksums have not been materialized or independently verified here. Treat this as a newly catalogued simulator input candidate until those artifacts are obtained and hashed. [B2FIND record](https://b2find.eudat.eu/dataset/c210e2d9-edd3-5062-8ad9-7cecf9a0b820).

### Evasive Spectre / interleaving baselines

T-SMADE (2024) is a required workload-aware two-stage baseline. It reports 95.42% average attack-detection success for Evasive Spectre-NOP and 100% for Evasive Spectre-memory, with non-attack success rates of 87.49% and 88.89% in its comparison table. It is explicitly designed around two evasion variants and a limited workload taxonomy; it does not establish robustness against arbitrary new adaptation strategies.

Source: [Jiao et al., Electronics 13(20), 4090 (2024)](https://doi.org/10.3390/electronics13204090).

Additional comparison arms include Tong et al.'s sample-augmentation adversarial training for evasive Spectre (2026), and Mandal et al.'s statistical detection of probabilistically interleaved attacks (2024). Their published figures are reference points only; platform, attack, sampling and operating-point differences forbid pooling.

Sources: [Tong et al. (2026)](https://doi.org/10.1016/j.engappai.2025.113634); [Mandal et al. (2024)](https://eprint.iacr.org/2024/1856).

## 3. Reproduction before claimed improvement

Each public baseline gets two lanes:

- **Paper-faithful reproduction:** preserve the original split, feature pipeline, counter set, threshold and metrics as closely as the artifacts permit. Report deviations and unavailable fields.
- **Strict nested reproduction:** repeat with program/execution/family-aware outer holdouts. Fit every learned operation only on outer-training groups: feature ranking, variance filters, normalization, imputation, dimensionality reduction, model selection, calibration, thresholds and alert suppression/re-arm policy.

Do not silently replace a paper's result with a stricter result or vice versa. Publish both, with protocol IDs and source digests.

HARPY-V-GEM5 and physical HARPY-V V2 are distinct evidence domains. Synthetic fixtures, simulator results and physical measurements must never be combined into one headline score. HARPY-V acquisition, exact archive inventory, provenance and permissions remain gated by issue #9045; the current benchmark work does not claim those bytes have been obtained.

## 4. Mandatory comparison matrix

| Arm | Modality | Purpose |
|---|---|---|
| Decision Tree | HPC | lightweight interpretable baseline |
| Random Forest | HPC | strong conventional ensemble baseline |
| SVM / Naive Bayes | HPC | margin and simple probabilistic comparators |
| DT | committed-instruction features | low-cost structural signal |
| Normalized HPC | HPC / retired instructions | reproduce physical contention result |
| T-SMADE | workload classifier + four HPC events | evasive Spectre comparator |
| Tong sample augmentation | HPC + adversarial examples | adversarial-training comparator |
| Interleaving statistical detector | selected HPC distributions | low-frequency interleaving comparator |
| Simple robust statistics | rates, quantiles, change detection | non-ML control and transparent floor |
| Symthaea HDC / liquid temporal model | multimodal temporal telemetry | candidate, not presumed winner |
| Symthaea cascade | cheap first-stage + selective second-stage | test complementary modalities and abstention |

Every arm receives the same permitted training groups, outer test groups, acquisition budget and predeclared alert constraint. Models may use different modalities, but must disclose them and their measurement costs.

## 5. Split ladder

Report each split independently; do not collapse them into one “generalization” label.

1. Random-window split — interpolation diagnostic only; never a zero-day claim.
2. Contiguous temporal-block holdout — temporal drift within represented execution.
3. Execution holdout — a new independent run of a represented binary.
4. Program/binary holdout — unseen application/program identity.
5. Attack-family holdout — unseen mechanism family.
6. Adaptation-family holdout — unseen timing, interleaving, padding, workload-mimicry or counter-aware strategy.
7. Device/configuration holdout — unseen board, firmware, kernel, PMU configuration or event mapping.
8. Cross-platform holdout — new physical microarchitecture and ISA family.

Nested splits must prevent trace/window parentage, replays, repeated iterations or duplicated content from crossing a claimed independent holdout. For class-homogeneous LOPO folds, report per-program accuracy and class-conditional recall/specificity; mark mixed-class precision, F1 and MCC unavailable for that fold.

## 6. Primary metrics and operating point

**Primary security endpoint:** attack-effect-conditioned recall. Only malicious cases with independently observed attack effect enter the successful-effect denominator. Attempted attacks with no observed effect and unknown-effect cases remain separate.

**Primary benign endpoint:** false alerts per device-hour, only where actual monitored exposure time is measured. Also report per-program and per-execution benign FAR. Do not derive device-hour exposure by multiplying synthetic windows or nominal sampling periods.

**Operating point:** choose the alert threshold and alert deduplication/re-arm policy on validation groups only. Freeze before test. Candidate and baseline must satisfy the same predeclared benign alert budget; equal threshold hashes are not proof of equal realized FAR.

**Worst-case reporting:** macro and worst-group recall across program, execution, attack family, adaptation family/level and device. Include each group denominator and all zero/failed groups.

**Latency:** p50/p95 time-to-detection only when onset and capture clocks are independently aligned. Otherwise report window-level detection delay or N/A with a reason.

**Calibration and abstention:** Brier/log loss/ECE, coverage-risk curves, abstain fraction and sensor-invalid fraction. Calibration must be fit on validation groups; test labels must not influence the calibrator.

**Resource cost:** acquisition overhead, inference latency, CPU time, resident memory, telemetry bytes/bandwidth, dropped samples, missed sampling deadlines and measured energy. Energy must remain unavailable when no power sensor or defensible energy estimate exists. Separate counter acquisition from model inference and end-to-end overhead.

## 7. Preregistered aspirational targets

These are research goals—not standards, published guarantees or achieved Symthaea results.

| Dimension | Aspirational target | Qualification |
|---|---:|---|
| Pooled attack-effect recall | ≥95% | at a frozen, matched false-alert budget |
| Worst-family recall | ≥90% | report every family; do not hide a failing family in the macro |
| Physical benign false alerts | ≤0.1/device-hour | measured on actual benign exposure, not inferred from windows |
| p95 detection delay | ≤2 sampling windows | onset/capture clocks must be aligned |
| End-to-end overhead | ≤1% median, ≤2% p95 | on declared reference workloads; acquisition included |
| Invalid-sensor handling | zero silent clears | missing/stale/misaligned/multiplexed/untrusted must abstain or fail closed |
| Cross-platform scope | ≥3 distinct physical microarchitectures, ≥2 ISA families | required before any cross-platform transfer claim |

The resource goals are intentionally ambitious. They may need revision after the first pinned physical reproduction, but revisions must be preregistered and must not be used to excuse a failed test post hoc.

## 8. Independent support is the limiting resource

The benchmark counts independent programs, executions, attack implementations/families and physical devices—not rows or windows. Under a one-sided exact Clopper–Pearson bound, with zero failures, a single endpoint needs 59 independent Bernoulli groups to put its upper risk bound below 5%, or 299 to put it below 1%. If six group-risk endpoints share family-wise alpha 0.05 using Bonferroni, the corresponding counts become 94 and 477 independent groups.

These are conditional design calculations: the groups must actually be independent/exchangeable for the selected endpoint. A lineage receipt helps audit identity; it does not prove independence. When support is inadequate, publish wide bounds or N/A—not a narrow window-level bootstrap interval.

## 9. Required adversarial and measurement controls

- Evasive Spectre with NOP/padding, memory delay, timing drift and workload mimicking.
- Probabilistic interleaving at predeclared rates, including 10^-2 through 10^-6 where safely measurable.
- Benign CPU, memory, cache, branch and I/O contention; multi-process workloads; workload phase changes.
- Attacker-aware counter selection, sparse sampling, counter multiplexing and threshold adaptation.
- Counter acquisition interference: isolated vs simultaneous event collection, order randomization, perf overhead, counter scaling/time-enabled/time-running, event aliasing and event semantics.
- Repeated/replayed traces, duplicate windows, overlapping windows, label permutation within correct groups and split leakage.
- Sensor faults: missing, stale, misaligned, multiplexed, reset, overflow, untrusted or implausible readings.
- Ablations: raw HPC, instruction-only, normalized HPC, multiple denominator choices, temporal features, HDC, CfC/LTC, calibration, abstention and cascaded stages.

Attack generation and physical testing must remain within controlled, authorized research platforms; the benchmark protocol does not require or authorize deployment against third-party systems.

## 10. Paradigm-shift architecture hypothesis

Test this as a falsifiable system design, not a foregone conclusion:

1. **Low-cost front door:** committed-instruction statistics and a minimal, verified subset of normalized counters.
2. **Temporal invariant layer:** HDC representations plus liquid temporal dynamics to model changes and relations across windows, not only static per-window class boundaries.
3. **Workload/context conditioning:** infer workload regime and contention state, then compare against the correct conditional baseline rather than a single global threshold.
4. **Evidence-aware escalation:** a more expensive detector or forensic attribution stage runs only when uncertainty/risk requires it.
5. **Fail-closed sensor contract:** invalid telemetry never becomes a silent benign verdict; uncertainty and abstention are explicit outputs.
6. **Adversarial co-evolution:** train/evaluate against adaptive strategies selected after seeing the defender's policy, with hidden attack families and held-out adaptation rules.
7. **Traceable explanation:** explanations must localize the relevant evidence and survive source-bound mutation tests; attractive saliency pictures alone do not qualify.

The key hypothesis is that a *selective, evidence-aware multimodal cascade* can improve worst-family detection at a fixed alert and acquisition budget—not merely increase accuracy on a balanced random split.

## 11. Industry positioning

This work is complementary to, not a replacement for, silicon roots of trust. OpenTitan's security model covers device/software attestation, provisioning, secure boot, lifecycle, firmware update, chip identity and ownership transfer; its testing framework also targets side-channel and fault-injection testing on FPGA and silicon. NIST IR 8615 emphasizes lifecycle-wide assurance, interoperable trust models, provenance/traceability and scalable verification/validation.

Symthaea's proposed niche is continuous runtime behavior sensing plus traceable, independently auditable evidence for monitoring and incident response. It should not claim that an anomaly detector supplies secure boot, attestation, silicon authenticity, side-channel resistance or certification.

Sources: [OpenTitan Security Model](https://opentitan.org/book/doc/security/); [NIST IR 8615](https://doi.org/10.6028/NIST.IR.8615).

## 12. Release gates and stop conditions

No benchmark result is confirmatory until:
- dataset rights, exact archive bytes, source hashes and provenance are resolved;
- all relevant traces are linked and duplicates/replays retained;
- acquisition semantics and counter interference are measured;
- outer split and nested model-selection receipts are valid;
- validation-frozen thresholds and matched alert budgets are recorded;
- primary endpoint, family-wise alpha and minimum support are preregistered;
- all planned baselines and all failed runs are preserved;
- code, configuration, environment, model, threshold, input and receipt digests are bound.

A missing artifact, unavailable physical platform, unsupported onset clock or insufficient group count produces a precise N/A / insufficient-support status. It must not be silently dropped from the report.

## 13. Claim ceiling

A successful campaign establishes measured performance for the named software, platform, event set, workloads, attack implementations, adaptations, sampling protocol and operating point. It does not prove universal evasion resistance, all-architecture transfer, causal prevention, hardware security, or certification.

## 14. Executable plan contract

The plan is represented in `tests/fixtures/hardsec/hardware-benchmark-plan-v1.json` and constrained by `schemas/hardsec-hardware-benchmark-plan-v1.schema.json`. The Rust plan auditor is `examples/hardsec_benchmark_plan_audit.rs`, built by the isolated `hardsec-adaptive-evasion-audit-harness` with feature `hardsec-benchmark-plan-audit`.

Run it with:

```bash
cargo test --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-benchmark-plan-audit \
  --example hardsec_benchmark_plan_audit

cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-benchmark-plan-audit \
  --example hardsec_benchmark_plan_audit -- \
  --plan tests/fixtures/hardsec/hardware-benchmark-plan-v1.json \
  --receipt /tmp/hardsec-017-plan-receipt.json
```

The auditor rejects missing required baseline arms, duplicate arm identities, an incomplete split ladder, domain pooling, insufficient platform targets, missing adversarial scenarios/negative controls, inconsistent Clopper–Pearson sample-size arithmetic, non-zero invalid-sensor silent-clear targets, and report rules that permit mixed-domain pooling or random-window confirmatory claims. It emits a SHA-256-bound, unsigned, no-overwrite receipt.

Twelve Rust unit tests exercise the valid fixture and negative controls. The focused workflow pins the Rust toolchain, checks formatting, runs the exact example test suite and produces a receipt from the fixture. A valid receipt means only that the **plan** satisfies these structural commitments. It does not run or validate any detector experiment, nor prove that the planned targets are achieved.
