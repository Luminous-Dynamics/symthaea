# HARDSEC-001 — Leakage-resistant hardware assurance benchmark

**Status:** preregistered protocol; no benchmark results are claimed yet  
**Tracking issue:** [Luminous-Dynamics/symthaea#8990](https://github.com/Luminous-Dynamics/symthaea/issues/8990)  
**Protocol revision:** 0.1  
**Protocol date:** 2026-10-29  
**Primary implementation language:** Rust for the harness, split validation, result aggregation, and evidence verification

## 1. Research question

Can a Symthaea-assisted investigation loop improve **held-out attack-family detection, time-to-validated-finding, or test-selection efficiency** over strong lightweight baselines without unacceptable false alarms, excess monitoring cost, or unsupported security claims?

This is a falsifiable research program, not a claim that Symthaea or Symtropy currently outperforms established tools.

The central design change is to evaluate *transfer across independent groups and evidence layers*, rather than optimize a classifier on randomly split trace windows. A detector that recognizes a known program's signature is useful for a bounded deployment, but it has not thereby demonstrated generalization to a new program, exploit implementation, processor, or design family.

## 2. Evidence motivating the protocol

Recent RISC-V research supplies a baseline and a warning:

- Pou et al., *RISC-V hardware attack detection using on-chip hardware performance counters*, reports results for the HARPY-V data and describes leave-one-program-out evaluation. The authors report strong aggregate results alongside important differences between raw and normalized counters and acknowledge the single-processor limitation. [Paper](https://link.springer.com/article/10.1186/s42400-026-00662-8) · [HARPY-V dataset, DOI 10.34810/DATA2538](https://doi.org/10.34810/DATA2538)
- Pujol et al., *Lightweight detection and classification of RISC-V hardware exploits using machine learning techniques*, evaluates the separate HARPY-V-GEM5 **simulator** dataset and reports above-99% known-sample accuracy, but attack-family-dependent transfer to unseen programs. The authors explicitly note that all experiments use gem5 and may not transfer to physical hardware. The dataset is identified as *RISC-V hardware attack traces on gem5 o3 CPU performance and instruction counters*, DOI [10.34810/data2711](https://doi.org/10.34810/data2711). [Paper](https://link.springer.com/article/10.1007/s00521-026-12467-4)
- Barredo et al.'s GJALLARHORN physical EM study reports 95.94% recall on an STM NUCLEO-144 and 73.33% on a Raspberry Pi 3B. The platform gap supports treating physical leakage as device-specific measured evidence, not as a property inferred from simulation. [Paper](https://www.sciencedirect.com/science/article/pii/S0167404825003815)
- A 2026 Trust-Hub evaluation reports that random/pooled evaluation can substantially overstate gate-level hardware-Trojan generalization when variants of the same base circuit appear across splits. Its random-forest F1 falls from 0.914 in pooled evaluation to 0.636 for leave-one-circuit-out and 0.460 for leave-one-base-family-out. This motivates a *separate* RTL/netlist lane with family-level holdouts, not pooling netlists with CPU telemetry. [Study](https://research.google/pubs/protocol-effects-on-feature-based-hardware-trojan-detection-across-trust-hub-families/)
- OpenTitan's design-verification approach combines simulation, formal-property verification, and security-countermeasure checks; its physical penetration-test framework also includes side-channel and fault-injection methods. The benchmark should complement these methods, not replace them. [Verification methodology](https://opentitan.org/book/doc/contributing/dv/methodology/index.html) · [Security overview](https://opentitan.org/book/doc/security/)
- NIST SP 800-193 frames platform firmware resilience around protection, detection, and secure recovery after unauthorized change. Firmware update/recovery testing will be a future, separately labeled lane. [NIST SP 800-193](https://csrc.nist.gov/pubs/sp/800/193/final)

Reported numbers above are results from their respective experimental settings. They are not directly comparable across papers and are not our baseline results.

## 3. Initial scope and threat model

### Phase A: telemetry transfer benchmark

**In scope:** authorized, reproducible experiments using published RISC-V telemetry datasets and clearly labeled benign/malicious workloads. Compare detector quality under prespecified splits and feature-access constraints.

**Out of scope:** autonomous exploitation, persistence, stealth, unauthorized targets, claims of exploitability based only on an anomaly, and any assertion that silence from a detector proves a device secure.

At first, the research question is whether telemetry supports robust behavioral discrimination. A later stage must connect a detection to a defined security property and reproduce the violation independently before the result can be labeled a confirmed finding.

### Keep evidence domains separate

1. **Physical-device telemetry:** HARPY-V on-chip counter traces. Record CPU model/stepping, board and software environment whenever the dataset provides them.
2. **Simulator telemetry:** HARPY-V-GEM5 or other explicitly named simulator-produced traces. Record simulator version, configuration, workload binary digest, instrumentation settings and sampling interval.
3. **RTL/netlist hardware-Trojan data:** a future track using circuit-family-aware splits and its own labels, metrics and tools.
4. **Physical side-channel/fault data:** a future laboratory track requiring calibrated instruments, exact capture setup, controls, and repeatability evidence.

Do not merge these datasets into one headline score. Agreement across independent domains is evidence of transfer; it is not permission to obscure domain differences.

### Measurement provenance is part of the threat model

The physical HARPY-V study reports that recording several counters together changed some measured values by over 300% in an example, attributing this to `perf` instrumentation effects. The researchers therefore collected each event in isolation to avoid obscuring measurements. This has direct consequences for our benchmark: we must not join counters from separate invocations into a fictional synchronized time series, and we must not train/test split counter-specific variants of the same logical program as if they were independent programs. Preserve capture/run identity, counter-set semantics, sampling interval and instrumentation version; compare raw versus normalized features as separate preregistered arms. Instrumentation cost and fidelity are part of the deployment trade-off, not merely preprocessing details. [HARPY-V study, instrumentation discussion](https://link.springer.com/article/10.1186/s42400-026-00662-8)

This suggests a second, distinctive benchmark dimension: **security gain per monitoring burden**. For each monitor, report detection delay and held-out recall alongside capture overhead, counter coverage, data loss, and any measured perturbation caused by observation. Never claim an energy or overhead result when that sensor was unavailable or not measured.

## 4. Preregistered hypotheses

- **H1 — Leakage-resistant generalization:** at least one evaluated arm transfers better to held-out programs or attack families than a random-window evaluation would suggest.
- **H2 — Model value is empirical:** Symthaea's temporal HDC/CfC features improve a prespecified transfer metric or resource frontier over the strongest baseline. A null result is valid, and a simpler baseline should remain the default if it wins.
- **H3 — Active investigation value:** after Phase A, ranking the next authorized test by expected decision value per unit cost improves validated-findings-per-budget over random selection and a simple coverage-guided policy. Information gain alone is not the endpoint.
- **H4 — Evidence chain value:** typed artifact lineage and independent replay prevent a detector score from being reported as a verified vulnerability without the required supporting evidence.

Phase A can test H1 and establish the baseline needed for H2. H3 and H4 require later instrumentation and cannot be claimed from classification metrics alone.

## 5. Baseline arms

All arms must receive the same permitted input features, training groups and validation budget within each comparison.

| Arm | Purpose | Important constraint |
| --- | --- | --- |
| Fixed threshold / rules | Transparent floor and interpretable control | Thresholds selected on validation groups only |
| Decision Tree | Lightweight interpretable supervised baseline | Fixed tuning budget |
| Random Forest | Strong tabular baseline consistent with recent work | Fixed tuning budget; no hidden feature leakage |
| Statistical/temporal detector | Detect shifts without supervised attack-family labels | Report its false-alarm rate on benign-only sequences |
| Symthaea HDC/CfC temporal arm | Test the proposed distinctive mechanism | Ablate HDC and temporal dynamics separately; no privileged feature access |

Do not add a complex model without a stated reason. Record model and dependency versions, hyperparameters, training time, inference time, memory use, and host information. Prefer the same executable data path and feature schema across arms.

## 6. Splits that answer the real question

Split at the **highest independent grouping unit supported by the data**. Never randomly distribute adjacent windows from a single execution across train and test.

Required results, when sample counts support them:

1. **Reproduction split:** a faithful reconstruction of the source study's split, clearly identified as a reproduction rather than zero-day evidence.
2. **Grouped program holdout:** all windows from each held-out program stay outside training and threshold selection.
3. **Attack-implementation/family holdout:** hold out complete attack implementations or families, not merely a different trace of the same attack.
4. **Execution holdout:** no shared execution ID or contiguous source trace spans multiple splits.
5. **Device/configuration holdout:** where a second independently measured device or configuration exists, preserve it as an external transfer set.
6. **Simulator-to-device transfer:** only when features and semantics are comparable. Report this as cross-domain transfer, not ordinary IID testing.

If the dataset has too few independent groups for a meaningful family-holdout confidence interval, report the result as descriptive, include each group result, and do not imply that many correlated windows substitute for more independent attacks.

### Leakage and integrity controls

- Audit for label-derived features, path/name tokens, timestamps that encode collection order, run length, sampling artifacts, and simulator-only fields unavailable on devices.
- Canonicalize and hash raw inputs; detect exact duplicates and near-duplicate executions before splitting.
- Run a label-permutation control, an explicitly leaky random-window split, and a benign-only negative control. The deliberately leaky result is a diagnostic demonstration, never the preferred score.
- Freeze split manifests and feature transforms before final test evaluation. Test labels must not influence threshold choice, feature selection, early stopping, or model selection.
- Keep seeds and split IDs fixed for paired comparisons. Any exploratory change after seeing test results requires a new versioned holdout or must be labeled exploratory.

## 7. Metrics and decision gates

### Report every run

- TP, FP, TN, FN; recall/sensitivity; precision; specificity; FPR; F1; PR-AUC; and ROC-AUC as a secondary metric.
- Per-program and per-family outcomes, macro-averages, worst-group outcomes, and uncertainty intervals.
- Calibration (for example, Brier score and reliability plots) if confidence scores are emitted.
- Detection delay in samples/instructions and wall time, throughput, memory, and monitoring overhead.
- Missing traces, parse failures, abstentions, unsupported input, and verifier rejections as explicit counts; never silently drop them.
- Precision at a range of deployment base rates, because a small false-positive rate can still overwhelm an operator when true incidents are rare.
- Report the 1% FPR operating point as a challenge target only when the number of independent benign groups supports a meaningful estimate. If it is not estimable, say so explicitly; a handful of benign programs cannot substantiate a precise 1% false-positive guarantee.

Windows from one program/execution are correlated. Uncertainty estimation should resample at the program/family/device level where possible; never treat every window as an independent experimental unit.

### Proposed challenge gate for a Symthaea-specific win

A result merits a stronger claim only if all of the following are met:

1. On a preregistered held-out-group evaluation, Symthaea improves **macro family-level recall by at least 5 absolute percentage points at a fixed, validation-selected operating FPR** (target 1% FPR when the independent benign sample supports it), *or* reaches baseline recall with at least 25% lower measured inference/monitoring cost.
2. The paired, group-level uncertainty interval supports a positive improvement. If too few independent groups exist to estimate that interval credibly, the result remains exploratory rather than passing by default.
3. The result survives at least three prespecified seeds, exact-data/hash reproduction, the leakage audit, and negative controls.
4. Gains are not purchased by unacceptable false-positive burden, input exclusions, or materially worse performance on a held-out family.
5. The evaluation records the best non-Symthaea baseline too; it is not sufficient to compare against only a weak baseline.

These are proposed project gates—not industry standards, expected outcomes, or a substitute for a sufficiently powered study. If they are infeasible on the available dataset, first establish and publish the honest baseline rather than alter the rules after inspecting results.

### Claim hierarchy

- **Anomaly signal:** model score crosses a declared threshold.
- **Reproduced anomaly:** an independent replay reproduces a defined observation.
- **Property violation:** a named invariant fails under a controlled test, formal counterexample, or independently validated observation.
- **Confirmed vulnerability:** the violation's security significance and scope have been reviewed with adequate evidence.
- **Remediated:** a fix is retested against the same witness and relevant regression/negative controls.

Only evidence appropriate to the level supports the corresponding label. A high probability, a simulation screenshot, a model explanation, or a passed benchmark is not by itself a vulnerability verdict.

## 8. Cross-layer evidence graph

The longer-term architectural differentiator is not a larger classifier. It is a typed, hash-linked evidence graph that allows an investigator to move from a weak symptom to a narrowly testable property without losing provenance.

Every record should link, when applicable:

**design/RTL digest → synthesized netlist digest → firmware/image digest → board/device/config identity → test and threat-model version → raw trace digest → feature-transform digest → model/config digest → candidate hypothesis → replay witness → independent checker receipt → reviewed finding/remediation**

Each edge must declare whether it is measured, derived, asserted by an operator, or modeled. An absent link is unknown, not implicitly trusted. Keep secrets and sensitive device credentials out of public receipts.

Proposed minimum result fields:

- schema/version and run ID;
- source revision, tool versions, dataset/license reference and cryptographic hashes;
- threat-model/property ID, target scope and device or simulator identity;
- split manifest hash, seed(s), feature schema, model hash and operating threshold;
- raw evidence hashes, coverage/missingness, result counts and uncertainty method;
- reproduction command, exit status, verifier identity/version and receipt hash;
- claim level, limitations, unresolved alternatives, and reviewer state.

Use a canonical serialization and explicit hash domain/version before any record is called signed or attested. Hashes establish byte identity, not measurement truth or hardware identity. Signatures establish control of a key, not scientific correctness.

## 9. Bounded division of responsibility

**Symthaea — investigator, not judge.** Rank hypotheses, represent temporal context, expose uncertainty, request additional tests, and propose candidate explanations. Test HDC/CfC and active test selection by ablation.

**Symtropy — scenario orchestration and model mismatch.** Organize deterministic scenario state, expected-versus-observed comparison, and counterfactual bookkeeping only where the modeled state and equations are validated.

**Authoritative domain tools — ground truth within their scope.** Use the selected CPU simulator for its configured microarchitecture, RTL simulator/formal engine for encoded design properties, firmware verification for image/update/recovery conditions, and calibrated measurement instrumentation for physical signals. Symtropy's generic physics engine is not a replacement for gem5, RTL, silicon, or electromagnetic measurement.

**Independent verifier — enforces the claim ceiling.** Recompute hashes, schema validity, split separation, metric arithmetic, witness replay and property-specific evidence. The classifier cannot certify its own output.

A failed replay, digest mismatch, unknown configuration, absent coverage, or verifier failure returns **inconclusive**; never silently coerce it to **secure**.

### Reuse cross-repository foundations rather than fork them

Before adding a second model-confidence or abstention subsystem, review Symtropy's existing [MINE-MCRED-001 model-credibility gate](https://github.com/Luminous-Dynamics/symtropy/issues/2179), which already treats stale telemetry, residual drift, degraded-model latching and recovery as explicit admission states. HARDSEC may reuse its general credibility vocabulary and test patterns, but must add hardware-specific assumptions, capture provenance and the strict distinction between attempted attack and verified security effect. Do not import an industrial process model's calibration envelope as if it were evidence about CPU behavior.

For reproducible execution when shared CI runners are unavailable, coordinate with Symtropy's [QUAL-002 hermetic qualification work](https://github.com/Luminous-Dynamics/symtropy/issues/1280). This is a potential independent execution path, not proof that the current PR has been compiled: the exact code/configuration and retained test output still have to match. Do not register a persistent personal workstation as a general public-PR runner.

### Related design-time lane: HARDSEC-002

[HARDSEC-002 / issue #9038](https://github.com/Luminous-Dynamics/symthaea/issues/9038) separately evaluates closed-loop secure RTL and firmware generation against [HardSecBench (IJCAI 2026)](https://doi.org/10.24963/ijcai.2026/57), which reports 924 Verilog/C tasks spanning 76 hardware-relevant CWEs. Its draft protocol is in [PR #9040](https://github.com/Luminous-Dynamics/symthaea/pull/9040). This complements HARDSEC-001 but must not be pooled into one leaderboard: design-time joint functional/security correctness and runtime telemetry detection have different datasets, ground truth, and metrics.

## 10. Future lanes (separate qualification)

### RTL and hardware-Trojan lane

Use a circuit-aware dataset manifest and hold out entire host circuits/base families. Compare structural features and formal/property-driven checks where available. Include clean circuits, known Trojan variants, and controlled mutations as distinct categories. Do not claim that success on Trust-Hub-like benchmark circuits generalizes to arbitrary silicon.

### Firmware protection and recovery lane

Exercise authorized update, rollback, integrity, recovery and unavailable-root-of-trust scenarios aligned to NIST SP 800-193. Record both failure detection and time / correctness of restoration. A clean initial boot does not establish secure recovery.

### Side-channel and fault-injection lane

Require a real capture manifest, instrument calibration, trigger alignment, device/board details, acquisition parameters, environmental controls, repeat sessions, and benign/positive controls. Synthetic signals can test the analysis pipeline but cannot stand in for measured emissions or establish physical susceptibility.

## 11. Phase plan

- **A — protocol + data audit:** verify licenses and schemas, pin hashes, create leakage-resistant split manifests, and reproduce published baselines.
- **B — model ablation:** compare the fixed baselines with Symthaea HDC/CfC under identical inputs; include calibration and resource cost.
- **C — targeted investigation:** evaluate active test selection against random and coverage-guided controls, counting only independently validated outcomes.
- **D — model replay integration:** connect Symtropy for trace/scenario orchestration and compare simulated predictions against held-out observations.
- **E — cross-layer / physical qualification:** add RTL, firmware and laboratory lanes only when each has its own ground truth, tooling, authorized target and evidence contract.

### Dataset-specific facts to preserve in Phase A

- **HARPY-V V2 (physical):** its repository describes 16 attack workloads and 16 benign programs on a Xuantie C910 RISC-V board. Some attack primitives failed on the tested device in some iterations. The dataset page says its time column is approximate and should not be used to train the agent; the physical study also reports per-counter collection to control measurement interference. Preserve raw file, counter and run identity, and do not manufacture synchronized timestamps from the approximate time column. Data DOI: [10.34810/DATA2538](https://doi.org/10.34810/DATA2538).
- **HARPY-V-GEM5 (simulated):** a separate CC0 1.0 dataset of 16 attack and 16 benign workloads, generated using a single-core gem5 O3 CPU model and statistics sampled every 10,000 instructions. The published ML study evaluates both hardware performance counters and committed-instruction mnemonic frequencies. It reports attack-family-dependent generalization for counter features and stronger unseen-program generalization for instruction-based features; the paper recommends a possible two-stage design (instruction-based detection followed by counter-based attribution). This is an empirical hypothesis to test, not an assumed winner. Keep `hpc`, `instruction_mnemonics`, and `hpc_plus_instruction_mnemonics` as separate prespecified modality arms and compare them at equal resource budgets. It is a separately versioned simulator domain, not a surrogate for a physical device. Data DOI: [10.34810/DATA2711](https://doi.org/10.34810/DATA2711).
- Both data repositories identify the datasets themselves as **CC0 1.0**. This does not automatically relicense source programs, attack implementations, binaries, or our derived distributions; record those rights separately before redistribution.

For these corpora, a classifier may learn to recognize an *attack workload or attempted primitive* even when that primitive did not succeed. The benchmark must distinguish (1) recognition of an attack workload, (2) an observed security effect, and (3) an independently established property violation. `security_outcome=unknown` remains unknown rather than being imputed from the workload label. The manifest auditor checks supplied metadata consistency; it cannot verify the underlying experimental truth.

### Phase A implementation slice

An opt-in Rust example is present on the HARDSEC-001 draft branch:

```bash
cargo fmt --check --package symthaea
cargo test --example hardsec_split_audit --features hardsec-audit
cargo run --example hardsec_split_audit --features hardsec-audit -- \
  --manifest /path/to/splits.tsv --root /path/to/dataset \
  --receipt /path/to/new-receipt.json
```

The strict TSV header is:

```text
dataset_id domain source_artifact_id source_path source_sha256 sample_id program_id execution_id attack_family_id device_config_id capture_config_id counter_set_id feature_modality_id security_outcome split_protocol split label
```

These are tab-separated columns, not space-delimited. The initial auditor intentionally supports only `physical_hpc` and `gem5_sim`; RTL/netlist and physical side-channel data require separate modality-specific schemas and auditors. Supported telemetry split protocols are `program_holdout`, `attack_family_holdout`, `execution_holdout`, `device_config_holdout`, `capture_config_holdout`, and `counter_set_holdout`. Preserve `capture_config_id` (acquisition command/tool/settings) and `counter_set_id` (the exact counter set captured together). The `feature_modality_id` must be exactly one of `hpc`, `instruction_mnemonics`, or `hpc_plus_instruction_mnemonics`, and must remain constant within a manifest. `physical_hpc` permits only `hpc`; `gem5_sim` can use the three modality arms. Instruction-only traces set `counter_set_id=NA`; counter-set holdout is not applicable to that arm. This prevents a classifier comparison from silently mixing different feature availability. Each manifest must represent one dataset, one domain, one feature modality, and one split protocol. Use `NA` only for metadata that is genuinely unavailable and is not the selected grouping key. The extra `security_outcome` field is `not_applicable` for benign samples or `success_observed`, `failed_observed`, or `unknown` for attack workloads.

The auditor hashes the exact manifest bytes and each referenced regular source artifact with SHA-256, and prints the manifest digest in its deterministic summary under a `HARDSEC_SPLIT_AUDIT_OK` status that must not be interpreted as a security pass. It fails closed on digest mismatch, malformed columns, duplicate sample IDs, identical source bytes reused across different splits, missing split groups, path traversal, symlinks, inconsistent source-artifact metadata, execution IDs crossing splits, or a held-out group crossing splits. Exact duplicate bytes under different artifact IDs inside the same split are counted as aliases and reported, not automatically rejected; they are not independent evidence and require deduplication or weighting before statistical evaluation. For attack-family holdout, attack rows are grouped by family and benign rows by program so a single generic benign label cannot accidentally collapse all benign controls into one group. Capture-configuration holdout is separately supported so evaluation can ask whether a monitor generalizes beyond a particular acquisition setup. It also requires non-missing program, execution, device/configuration, acquisition configuration, and counter-set identifiers; every attack-labeled row needs an attack-family ID. Validation and test must each contain both benign and attack labels so binary recall and false-positive metrics are defined; training may be one-class for an explicitly unsupervised anomaly detector. Repeated windows from a single artifact may remain in one split, but may not be presented as independent artifacts. The selected holdout key is enforced in addition to the invariant that one execution cannot cross splits. Exact source content under different artifact IDs must also agree on workload label and security outcome; an otherwise valid-looking duplicate with contradictory labels is rejected. The `counter_set_holdout` protocol is explicitly rejected for the `instruction_mnemonics` modality, where a counter-set grouping has no meaning.

**Receipt semantics:** the optional JSON receipt is deterministic metadata for the split audit and is created only at a new path; the tool refuses to overwrite an existing file. It includes the exact manifest SHA-256, split counts and a claim-scope label of `metadata_and_split_integrity_only`. The receipt is not signed, independently anchored, or proof of factual labels; an attacker who can rewrite both inputs and outputs can forge a self-consistent result.

**Scope limit:** this checks the metadata and file identities supplied to it; it does not prove that a program/family label is factually correct, that the data are licensed for the proposed use, or that an attack is exploitable. Those remain explicit data-audit and security-review tasks. Twenty-six unit tests are authored with the example but have not yet been executed in this environment.

Phase A is the first deliverable. The protocol intentionally does not depend on completion of runners to begin data and split audits; any execution claims still require retained exact-input and exact-code receipts.

## 12. Stop conditions and publication rule

Stop and repair the harness if the label-permutation test performs materially above chance, duplicate groups cross the split, feature transforms see test data, the benign negative control is called an attack without a documented threshold reason, or metrics cannot be independently recomputed.

Publish null results, failures and group-level outcomes. A SOTA claim requires same-data and same-split comparisons, equal feature access, disclosed compute/tuning budgets, appropriate uncertainty, and an independent rerun. Never compare scores from different datasets as if they formed one leaderboard.

## References

1. Pou, A. et al. (2026). [RISC-V hardware attack detection using on-chip hardware performance counters](https://link.springer.com/article/10.1186/s42400-026-00662-8). *Cybersecurity* 9, 222. Dataset: [HARPY-V, DOI 10.34810/DATA2538](https://doi.org/10.34810/DATA2538).
2. Pujol, J.-T. et al. (2026). [Lightweight detection and classification of RISC-V hardware exploits using machine learning techniques](https://link.springer.com/article/10.1007/s00521-026-12467-4). *Neural Computing and Applications*. Simulation dataset: [HARPY-V-GEM5, DOI 10.34810/data2711](https://doi.org/10.34810/data2711).
3. Xiao, H. et al. (2026). [Protocol effects on feature-based hardware-Trojan detection across Trust-Hub families](https://research.google/pubs/protocol-effects-on-feature-based-hardware-trojan-detection-across-trust-hub-families/).
4. Regenscheid, A. (2018). [NIST SP 800-193: Platform Firmware Resiliency Guidelines](https://doi.org/10.6028/NIST.SP.800-193).
5. [OpenTitan design verification methodology](https://opentitan.org/book/doc/contributing/dv/methodology/index.html) and [security overview](https://opentitan.org/book/doc/security/).
6. Barredo, J. et al. (2025). [GJALLARHORN: A framework for vulnerability detection via electromagnetic side-channel analysis in embedded systems](https://www.sciencedirect.com/science/article/pii/S0167404825003815). *Computers & Security* 159, 104692.
