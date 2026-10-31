# HARDSEC-004 — Operational, Open-Set Evaluation for Hardware Attack Detection

**Status:** research protocol / pre-registration draft  
**Scope:** evaluation design only; no trained detector, reproduced baseline, physical measurement, or security verdict is claimed  
**Related work:** [HARDSEC-001 split and provenance controls](https://github.com/Luminous-Dynamics/symthaea/pull/8992), [HARDSEC-003 deterministic corpus pipeline](https://github.com/Luminous-Dynamics/symthaea/issues/9045), [HARDSEC-002 secure RTL/firmware generation](https://github.com/Luminous-Dynamics/symthaea/pull/9040), [HARDSEC-004 issue](https://github.com/Luminous-Dynamics/symthaea/issues/9060)

## 1. Thesis: evaluate the monitoring service, not only the classifier

The benchmark's central question is not whether a model classifies telemetry windows accurately under a convenient split. It is whether a detector provides useful security evidence at a realistic alert, latency, measurement, and compute budget while remaining honest about unrecognized attacks and unavailable evidence.

High window-level accuracy can coexist with missed attack episodes, unacceptable false-alert volume, miscalibration, or failure on unseen programs. A classifier score is therefore a diagnostic metric—not the product-level outcome.

**Falsifiable hypothesis:** an inexpensive first-stage detector paired with selectively activated high-resolution attribution may lie on a better security–cost frontier than a single always-on classifier. Reject this hypothesis if the cascade fails to improve security outcomes or total measured cost under matched conditions.

## 2. Evidence motivating the protocol

The following are published findings, not locally reproduced results:

- The HARPY-V-GEM5 study published on 27 October 2026 reports known-workload detection accuracy above 99% for lightweight Decision Tree / Random Forest methods, but also large gaps under Leave-One-Program-Out evaluation and near-zero detection for some held-out attack families. It suggests instruction features for inexpensive screening and HPC features for attribution, while noting that its experiments are simulator-based and include only 16 proof-of-concept exploit implementations.
- The physical HARPY-V study published on 24 September 2026 reports that concurrently recording multiple hardware performance counters inflated some recorded values by over 300% in one example. It therefore collected each event separately on its test platform. In a separate multi-process experiment on that platform, dividing each raw counter value by retired instructions for the sample reportedly produced 98.56% accuracy versus 60% for unnormalized counts; it evaluated one attack plus one benign process and one attack plus three benign processes. The paper reports an average `perf` collection overhead of about 0.2 ms per 10 ms sample (2% of execution time). The platform was one Xuantie C910, direct energy was not measured, and these values are study-specific—not a portable deployment guarantee.

**Interpretation limit for the physical study's held-out-attack experiment:** its leave-one-attack-program-out procedure trains on the other benign and malicious programs while reserving one malicious program for testing. That held-out test fold is attack-only. Its reported per-attack accuracy is therefore an attack-conditional detection result; it cannot estimate the false-alert rate on unseen benign workloads, precision at deployment prevalence, or a complete operational alert burden. Treat it as an attack-transfer stress test, not as a mixed open-set deployment evaluation.

**Interpretation limit for the GEM5 leave-one-program-out experiment:** each test fold contains a single held-out program and thus only one class (benign or malicious). Per-fold accuracy is a class-conditional result; class metrics requiring both positive and negative examples in the same fold (such as ordinary per-fold MCC, or a conventional two-class F1 interpretation) are not informative there. Preserve every program's result and aggregate at the independent-program level. Do not let a pooled window score imply that each fold tested both benign false alarms and attack detection.

For an operational open-set claim, require distinct held-out strata for unseen malicious programs/families and unseen benign programs, then add a mixed benign/attack test where the data and monitored exposure support it. Report held-out-attack sensitivity and held-out-benign false-alert behavior separately. Precision and alerts per device-hour require an explicit prevalence/exposure model; they cannot be recovered from an attack-only fold.

**Model-selection provenance is a separate unresolved question:** the physical paper states that the leave-one-attack-program-out experiment reuses hyperparameters from the earlier experiment because “the previous validation was performed using the entire dataset.” The article text does not fully establish whether the held-out attack program also influenced feature ranking, feature subset choice, hyperparameter selection, or other learned preprocessing before the leave-one-attack-out test. This is not sufficient evidence to declare proven leakage, but it is not enough to certify a fully untouched confirmatory holdout either.

A clean reproduction must use an outer program-group holdout. Inside each outer-training partition only, perform feature selection (including mRMR/RFECV), normalization/transform fitting, model and hyperparameter selection, probability calibration, and threshold selection using group-aware inner validation. The outer held-out program must not influence any learned choice before final scoring. Report the paper-faithful setup and this fully nested setup separately if they differ; do not silently describe the former as leakage-free.

**Temporal-support and preprocessing caveat for HARPY-V-GEM5:** the 27 October 2026 study samples every 10,000 committed instructions, truncates each program trace from the beginning to 12,890 windows, and extends GhostWrite from 2,340 naturally produced windows by repeated iterations. Thus 412,480 windows are not 412,480 independent attack trials. The importer must retain original trace, execution, instruction-range and replay/iteration identity, and statistical uncertainty must be grouped at the program/execution level. This does not invalidate the paper; it limits what the row count can support. The article also describes variance filtering, Min-Max normalization and SelectKBest as preprocessing for LOPO but does not make the fold-local fitting boundary for each learned transform sufficiently explicit in the text. This is not proof of leakage. A clean reproduction must fit every learned transform, feature selector, calibration and threshold inside the outer-training programs only. Track the dependence and effective-sample work in [HARDSEC-014 / issue #9413](https://github.com/Luminous-Dynamics/symthaea/issues/9413).

**Instruction-only baseline clarification:** the GEM5 study reports known-workload detection above 99% for both HPC and instruction features, and reports instruction-only accuracy above 96% on its held-out attack-program cases, while instruction-based attack-family classification is substantially weaker (around 60%). That makes instruction-only DT/RF a mandatory binary-detection comparator in the *simulator evidence lane*, not an automatic physical-HARPY-V modality or proof of family attribution. Keep binary detection and attack classification as separate outcomes.

These results make four failure modes first-class: source/program leakage, held-out-family failure, false alarms in benign but unusual workloads, and measurement-induced distribution shift.

References:
- Pujol et al., “Lightweight detection and classification of RISC-V hardware exploits using machine learning techniques,” *Neural Computing and Applications* (published 27 October 2026): https://doi.org/10.1007/s00521-026-12467-4
- Pou et al., “RISC-V hardware attack detection using on-chip hardware performance counters,” *Cybersecurity* (published 24 September 2026): https://doi.org/10.1186/s42400-026-00662-8
- Physical HARPY-V V2, DOI 10.34810/DATA2538: https://doi.org/10.34810/DATA2538
- HARPY-V-GEM5, DOI 10.34810/DATA2711: https://doi.org/10.34810/DATA2711
- NIST SP 800-160 Rev. 1, systems security engineering context: https://doi.org/10.6028/NIST.SP.800-160v1r1

## 3. Claim boundaries and data preconditions

### 3.1 Keep evidence domains separate

Report physical HARPY-V and HARPY-V-GEM5 independently. A simulator result is not a physical-device result; a physical trace from one processor is not evidence of transfer to a different microarchitecture. Cross-domain evaluation may be reported only as an explicitly labeled transfer experiment with source/device identities and acquisition details intact. Do not publish a pooled headline score across domains.

### 3.2 Inspect source semantics before defining samples or episodes

Before computing any score, freeze the downloaded dataset's exact DOI/version, source URI, archive/file inventory and SHA-256, license evidence, schema, and importer revision. See HARDSEC-003.

Do not infer that one row equals an independent event, that adjacent windows are independent, or that a workload labeled as an attack succeeded. Keep workload/attack-family label separate from independently observed security outcome: success_observed, failed_observed, or unknown; benign rows use not_applicable.

**Time-to-detection is scoreable only if ground truth provides a defensible attack onset and the capture clock is aligned.** The approximate physical HARPY-V time field must not be used as ground-truth time merely because it is numeric. If onset or synchronized time is unavailable, mark event-latency metrics unavailable; do not synthesize them from row order and present them as measured latency.

### 3.3 No modality by assumption

Maintain separate artifacts and results for:
- instruction_mnemonics;
- hpc;
- hpc_plus_instruction_mnemonics.

A combined modality may be used only where both signals are available and alignment is evidenced. If the physical corpus supplies HPC but not aligned instruction mnemonics, no physical instruction-first cascade score can be claimed. Simulator and physical results remain separate, even when features share names.

### 3.4 Define the independent group

The split protocol must use the source-supported independent units: program, execution/run, attack family/primitive, device or microarchitecture, capture configuration, and counter set. Never split adjacent windows from one execution across train/validation/test. Use the HARDSEC-001 auditor, and resolve its documented source-artifact ambiguity through the HARDSEC-003 importer before scoring.

“Unseen attack family” is a held-out-family generalization experiment. It is not proof of real-world zero-day coverage.

## 4. Primary operational endpoints

Every metric must declare its numerator, denominator, aggregation unit, source modality and domain. Report group-wise values and uncertainty; do not treat highly correlated windows as independent trials.

### 4.1 False-alert burden

Report false alerts per monitored device-hour and device-day where exposure duration is known, plus the number of alert episodes after the frozen deduplication policy. Also report window-level false-positive rate for comparison with prior papers, but do not use it as a substitute for operational alert burden.

Predeclare the alarm merge/re-arm rule using validation data only. Count each resulting benign alert episode consistently. If monitored exposure duration is absent, label rates per unit time unavailable rather than estimating duration from sample count.

### 4.2 Attack-event detection

Where reliable ground-truth intervals exist, report:
- fraction of independent attack episodes alerted at least once;
- episode recall within each predeclared response deadline;
- delay from verified onset to first qualifying alert;
- missed episodes, repeated alerts per episode, and alerts after the episode ends.

If the corpus only supports sample labels or attack-workload tags, report sample/workload detection at that granularity and explicitly state that episode recall and latency cannot be computed from the supplied evidence.

### 4.3 Generalization, not just aggregate accuracy

Report:
- macro recall by held-out program and by attack family;
- worst-family recall, with family-level support counts;
- known-program, unseen-program, and unseen-family results as separate rows;
- physical and simulator domain results as separate sections;
- confusion matrix, precision, recall, F1 and MCC at the validation-frozen threshold;
- per-group uncertainty intervals and all failure strata.

Aggregate means must not obscure an attack family that is consistently missed. When a test fold has too few independent groups to support stable inference, mark the result descriptive and do not claim a statistically established improvement.

### 4.4 Calibration, abstention, and open-set behavior

If the model emits probabilities or comparable confidence scores, report reliability plots and Brier score or log loss where appropriate. Evaluate risk versus coverage as the system abstains on uncertain or out-of-distribution observations. Record abstention reason, frequency, and downstream policy.

Define a fail-closed telemetry policy before testing:
- missing, stale, malformed or unsupported-modality telemetry must not silently become “benign”;
- unknown distribution or unsupported device/configuration must trigger an explicit abstain/escalation state;
- a successful detector output must identify the model and configuration revision that produced it.

An abstention is not an attack detection, and a model's confidence score is not itself evidence that the system is secure.

### 4.5 Compute and measurement cost

Record per-stage and end-to-end:
- p50/p95 inference latency and, where meaningful, latency from observation availability to alert;
- CPU time/utilization, peak memory, trace bandwidth/storage, and the number of counters collected;
- frequency of expensive stage activation and fallback/abstention;
- measured energy only when a calibrated measurement source exists.

Use identical hardware/tool settings for paired comparisons. Separate inference cost from instrumentation cost. Report observer effect by comparing workload/counter behavior and detector output with and without collection; isolate counters when concurrent collection is known to perturb the platform.

### 4.6 Counter normalization and measurement invariance

Include a predeclared raw-versus-normalized feature ablation wherever the source schema supports the required denominator. The physical HARPY-V study's per-retired-instruction normalization result (98.56% versus 60% in its specific multi-process experiment) is a compelling baseline to reproduce, not a result to assume will transfer.

At minimum compare:
- raw event counts;
- event counts per retired instruction (the cited study's normalization);
- alternative physically meaningful denominators only where units and acquisition semantics are validated;
- no-denominator / zero-denominator / missing-denominator behavior.

Keep the denominator as a provenance-bearing input. Do not calculate a ratio when the denominator is zero, missing, saturated, or captured in a different acquisition interval; return an explicit invalid/abstain state. Fit any data-dependent transforms using training data only. Compare raw and normalized modes on the same group holdouts, benign interference conditions, and device/configuration strata. Normalization can suppress workload-scale variation, but it can also conceal genuine attacks or amplify denominator noise; treat this as a falsifiable invariance hypothesis, not an automatic improvement.

## 5. Frozen comparison matrix

The initial comparison should be deliberately modest and interpretable.

| Arm | Purpose | Required caveat |
|---|---|---|
| Simple threshold / non-ML rule | Establish a low-cost floor | Freeze the rule on validation data |
| Decision Tree (DT) | Published lightweight baseline | Report tree depth/model size and threshold selection |
| Random Forest (RF) | Strong tabular baseline | Measure actual end-to-end overhead |
| Instruction-only DT/RF (HARPY-V-GEM5) | Published low-cost binary-detection comparator; reported >96% on held-out attack programs | Simulator evidence only; attack-family classification is weaker; do not assume this modality exists in physical HARPY-V V2 |
| Instruction-only detector | Low-cost candidate where the modality exists | No use where aligned instruction telemetry is absent |
| HPC-only detector | Microarchitectural signal baseline | Counter acquisition may perturb the workload |
| Published workload-conditioned two-stage detector (T-SMADE) | Existing cascade comparator for evasive Spectre under varied workloads | Reproduce the workload classifier, selected HPC set, attack variants, platform and alert semantics; do not compare the abstract's rates across datasets |
| Instruction-first → HPC attribution cascade | Test selective monitoring hypothesis | Only meaningful where both signals and ordering are available; distinguish from T-SMADE's workload-first design |
| Always-on HPC detector | Compare against cascade's resource trade-off | Include the full instrumentation cost |
| Symthaea HDC/CfC candidate | Test the project-specific hypothesis | Must compete under the same groups, threshold policy, and budgets |

T-SMADE is a published two-stage baseline: its first stage identifies workload type and selects a second-stage detector using four HPC events plus a derived evasive-expanding feature. The paper's Table 6 reports **average attack TP across its three workload classes** of 95.42% for Evasive Spectre-NOP and 100% for Evasive Spectre-memory; the corresponding average non-attack TN rates are 87.49% and 88.89%. These are paired TP/TN outcomes and show a non-trivial benign false-alert trade-off.

Do not misread 95.42% and 100% as memory-stress-only results. The paper's per-workload tables/conclusion report Evasive Spectre-NOP TP improving from 59.80% to 97.32% under realistic applications, 0.84% to 98.16% under CPU stress, and 27.52% to 90.77% under memory stress. For Evasive Spectre-memory, the corresponding baseline-to-T-SMADE TP values are 95.14% to 100% (realistic), 11.73% to 100% (CPU stress), and 47.81% to 100% (memory stress). Its Table 4 reports per-realistic-workload TP/TN pairs (e.g. Evasive Spectre-NOP TP/TN 97.32%/75.04%, 96.98%/88.44%, and 90.95%/63.48% across three application bundles), illustrating why average attack TP alone is inadequate.

These are source-paper outcomes on its Intel Xeon Silver 4210 and workload setup, not results on HARPY-V or Symthaea. A fair reproduction must recover exact data, hardware, variants, per-workload TP **and TN**, precision/prevalence assumptions, sampling/threshold protocol and full acquisition cost: https://www.mdpi.com/2079-9292/13/20/4090.

A probabilistic baseline or other model may be added in a labeled extension, but the comparison cannot silently change the set after test results are seen. Hyperparameter search and model selection are confined to the training/validation partitions.

The cascade has two distinct outputs: (1) a suspiciousness/detection decision and (2) optional attack-family attribution. The second-stage labeler must never turn a missed first-stage event into a detection, and attribution accuracy must not be merged into detection recall.

## 6. Stress and negative-control matrix

Run the same frozen evaluation against:

1. **Benign diversity:** programs and workloads not seen during training, including CPU/memory-intensive benign work where the dataset or controlled testbed supports it.
2. **Execution/program holdouts:** prevent source program fingerprints and repeated traces from leaking into both train and test.
3. **Attack-family holdout:** hold out semantically distinct attack families; list each held-out family and its support rather than describing all held-out programs generically as zero-days.
4. **Capture/configuration shift:** counter subset/order, device, compiler/build, simulation configuration and capture configuration shifts when supported by the source.
5. **Interference and observer effect:** background-process interference, concurrent versus isolated counter capture, and resource-stress scenarios where reproducible evidence exists.
6. **Telemetry degradation:** missing counters, reordered fields, stale intervals, incomplete windows, and unsupported modality IDs; invalid input must not generate a benign certificate.
7. **Label and provenance mutation:** intentionally corrupt a label, group ID, artifact hash, or split assignment. The independent data/audit pipeline must fail before model scoring.
8. **Permutation/control check:** permute labels within defensible groups or use a known negative-control construction. An unexpectedly strong control score blocks interpretation pending leakage investigation.

Not every condition exists in each published corpus. Unsupported stress cases must be reported as not run, not simulated into existence, and not blended with real observations.

## 7. Pre-registration and statistical discipline

Before unlocking confirmatory test results, freeze:
- exact source hashes and parser revision;
- trace/window lineage, original-versus-replayed segment identity, window stride and independent group counts;
- train/validation/test grouping and holdout rationale;
- feature/preprocessing code and feature list, with every learned transform/feature selector fit inside outer-training groups only;
- model versions, seeds, search spaces, and attempt budgets;
- threshold selection rule and alert deduplication policy;
- primary endpoints, response deadlines, operating alert budget, equivalence/non-inferiority margin, and group-level statistical method;
- all exclusion rules, missing-data handling, and report tables.

Use at least three predetermined model-randomness seeds where the algorithm is stochastic. Use paired scenarios and the same held-out groups across model arms. Compute uncertainty by resampling independent groups (e.g. program/execution/family as supported), not by bootstrapping adjacent windows as though independent. Preserve null results and all failed runs.

Thresholds and calibration parameters are chosen on validation data only. Test results may not be used to select thresholds, modalities, model families, attack subsets, or deduplication windows. If the independent test set is small, publish effect sizes and group outcomes as exploratory; do not manufacture certainty with a large row count.

## 8. Proposed success gates

These are **project targets**, not published standards or established SOTA thresholds.

A new approach can be considered promising only after the data-quality and negative-control gates pass, and when one of the following is met on frozen confirmatory groups:

1. At the same predeclared false-alert budget, improve macro attack-episode recall by at least 5 percentage points over the strongest matched baseline, with group-aware uncertainty; or
2. Maintain recall within a predeclared 2-percentage-point non-inferiority margin while reducing measured end-to-end monitoring overhead by at least 25%.

The exact alert budget, deadline, group minimums, and margins must be selected before seeing confirmatory results based on the intended operating context. For this corpus, if exposure duration or reliable onset labels are missing, the event-rate/latency gates cannot be evaluated and must remain unscored. The benchmark must not redefine its primary endpoint after seeing the results.

No aggregate improvement passes if it conceals a reproducibly missed family, failed provenance control, material test leakage, or unsupported transfer from simulator to physical hardware. These gates are research decision rules, not a claim of certification or safe deployment.

## 9. Required results report

Every report should contain the following table, including unavailable metrics:

| Dimension | Required result |
|---|---|
| Evidence domain | Physical or simulator; device/simulator/configuration |
| Source integrity | DOI/version, artifact SHA-256, importer revision, split receipt |
| Independent test groups | Program, execution, family, and configuration counts |
| Modality | HPC, instruction mnemonics, or combined; evidence for alignment |
| Baseline | Exact model/setup and frozen threshold-selection procedure |
| Window metrics | Accuracy, precision, recall, F1, MCC, confusion matrix |
| Operational outcome | False alerts/device-hour, attack-episode recall, detection delay, or explicit reason unavailable |
| Generalization | Known program / unseen program / unseen family / domain transfer |
| Uncertainty | Group-aware interval/method and support counts |
| Open-set behavior | Calibration, abstention rate, unsupported/missing telemetry response |
| Cost | Inference and instrumentation overhead, p50/p95 latency, memory, bandwidth, energy only if measured |
| Controls | Leakage, label, permutation and telemetry-degradation control outcomes |
| Deviations | Every deviation from the preregistered protocol |
| Claim ceiling | Exactly what evidence supports and what remains untested |

## 10. Implementation order and decision gates

1. **Source qualification:** finish HARDSEC-003. Pin and inspect the downloaded files; decide which metrics the actual schema can support.
2. **Reference reproduction:** reproduce published DT/RF splits and feature handling before proposing a new model. Record every deviation from the paper.
3. **Operational metric evaluator:** implement deterministic event/alert aggregation and report generation only for metrics supported by source data. Include fixtures for overlapping alerts, missed events, missing intervals, unsupported timestamps, and group-aware summaries.
4. **Paired model comparison:** compare the frozen baselines under the same groups, validation threshold policy, resource measurement conditions, and test controls.
5. **Symthaea candidate:** add HDC/CfC only after baselines and evaluator are independently replayable.
6. **Physical transfer:** run only on a real, identified, authorized platform with instrumentation settings and counter effects recorded. Never relabel simulator results as on-device evidence.

**Stop conditions:** unresolved provenance/schema ambiguity, any test leakage, failed negative controls, fabricated timing alignment, absent outcome labels, or a broken baseline reproduction stop confirmatory scoring. Preserve the failure receipt and report the reason.

## 11. Final interpretation rule

The benchmark can establish comparative evidence for the tested datasets, holdouts, hardware/configuration, and acquisition policy. It cannot prove comprehensive zero-day coverage, absence of vulnerabilities, resistance to an adaptive attacker, or safe automatic mitigation. A promising model is a candidate to combine with explicit security policy, independent monitoring, and qualified platform-specific verification—not a replacement for those controls.
