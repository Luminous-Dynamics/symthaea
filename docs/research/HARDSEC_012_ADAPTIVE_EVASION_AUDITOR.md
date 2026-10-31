# HARDSEC-012 — Adaptive-Evasion Campaign Evaluator

**Status:** Rust-first evaluator and synthetic known-answer fixtures drafted; exact-head CI not yet observed  
**Parent:** [HARDSEC-012 issue #9346](https://github.com/Luminous-Dynamics/symthaea/issues/9346)  
**Evidence boundary:** synthetic fixture only; no attack has been generated, no physical or simulator campaign has been run, and no detector has been benchmarked.

## 1. Purpose

The evaluator scores whether a runtime detector identifies a malicious event **when the independently established attack effect actually occurs**, while reporting benign false alerts, adaptation-strategy generalization, telemetry integrity and measured resource cost separately.

It does not generate attack payloads, exploit hardware, infer effect from an attack label, or treat an attack attempt with no observed effect as a successful evasion.

## 2. Three-way evidence separation

The evaluator takes three separate inputs:

1. **Detector predictions** — case IDs, alert/clear/abstain/sensor-invalid response, optional calibrated score, model and threshold digests, and per-case measured costs.
2. **Evaluator case manifest** — hidden-to-detector split, program/execution/device groups, attack/adaptation family and level, and sensor state.
3. **Independent outcome oracle** — benign/malicious class, observed/not-observed/unknown attack effect, and an outcome-evidence digest.

The detector-facing predictions file does not contain ground-truth class, effect outcome, hidden split metadata or adaptation-family labels. The evaluator manifest and truth oracle must be protected by the campaign operator and not exposed to the model or explainer. Different files are not an access-control mechanism by themselves; the deployment runner must enforce the separation.

The evaluator requires exact equality between prediction IDs, oracle IDs and the manifest's test case IDs. It rejects duplicate IDs, missing outcomes, cross-campaign inputs, unsupported labels, and declared holdout-group leakage.

## 3. Implemented Rust evaluator

Source: examples/hardsec_adaptive_evasion_audit.rs  
Isolated locked harness: tools/hardsec-adaptive-evasion-audit-harness  
Focused workflow: .github/workflows/hardsec-adaptive-evasion-audit.yml

Run:

~~~bash
cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-adaptive-evasion-audit \
  --example hardsec_adaptive_evasion_audit -- \
  --predictions tests/fixtures/hardsec/adaptive-evasion-v1/predictions.json \
  --manifest tests/fixtures/hardsec/adaptive-evasion-v1/case-manifest.json \
  --truth tests/fixtures/hardsec/adaptive-evasion-v1/evaluator-truth.json \
  --receipt /tmp/hardsec-adaptive-evasion-receipt.json
~~~

### Primary outcomes

- Attack-effect-conditioned recall: only malicious cases with independently labelled attack_effect=observed enter the successful-attack denominator.
- Failed attack attempts (not_observed) are reported separately; they are never counted as successful evasion.
- Unknown-effect attempts remain unscored for successful-attack recall.
- Benign false-alert rate and valid-sensor true-negative rate are separate.
- Conservative effect-confirmed alert precision is reported against all detector alerts; it is not a deployment-prevalence estimate.
- Calibration (Brier, log loss, 10-bin ECE) is reported only when every test case has a numeric score, and its target is explicitly malicious-attempt versus benign—not successful attack effect.
- Attack-family, adaptation-family and adaptation-family/level recall are reported separately (avoiding severity-level pseudoreplication), plus program-group and execution-group recall, with macro and worst-group outcomes.
- Deterministic equal-group percentile bootstrap intervals (10,000 resamples) are emitted for attack-family, adaptation-family, program, execution and benign-program macro rates only when at least 10 declared independent groups exist. Resampling is at the group level, never at the window/case level. Seed derivation uses campaign ID, grouping axis and algorithm version—not prediction bytes—so matched model arms can use common resamples when group IDs match. These are descriptive intervals conditional on caller-declared independence/exchangeability, not proof of independence or family-wise population guarantees.
- When fewer than 10 independent groups exist, point estimates and per-group support remain visible but the bootstrap interval is explicitly unavailable. The nine-case known-answer fixture intentionally exercises this unavailable path.
- Exact one-sided Clopper–Pearson upper bounds are also reported for the group-level probability of at least one successful-attack miss or at least one benign false alert. The Bernoulli unit is the independent group, not a case/window. Six predeclared group-risk endpoints share a 5% family-wise alpha budget by Bonferroni (alpha=0.05/6 per endpoint). These bounds are conditional on caller-declared independent/exchangeable groups and are not per-case FNR/FPR.
- The evaluator checks class/oracle consistency: benign cases must use attack/adaptation family none; malicious cases must name an attack family; named adaptation families require positive levels.
- Benign false-alert rate is reported overall and on valid-sensor cases; valid-sensor true-negative rate excludes invalid telemetry.
- Holdout checks require disjoint program, execution, attack-family and adaptation-family IDs across train/validation/test; the sentinel value none is excluded from attack/adaptation-family grouping for benign/no-adaptation cases.
- Twenty-three Rust unit tests are authored, including direct attack-family and adaptation-family leakage mutations plus small-group interval suppression, reproducible equal-group bootstrap and known-answer support-count checks.
- Sensor-invalid cases are counted separately. Any clear response on a missing, stale, misaligned, multiplexed or untrusted sensor is reported as an integrity violation.
- Acquisition and inference nanoseconds, telemetry bytes and peak memory are summarized only where their required per-case measurements are complete. Energy is not inferred.

The receipt binds exact prediction, manifest and truth input bytes. It is unsigned, deterministic and no-overwrite. The tool reports result=evaluated; it does not automatically call a detector safe or successful.

## 4. Versioned input contracts

- schemas/hardsec-adaptive-predictions-v1.schema.json
- schemas/hardsec-adaptive-case-manifest-v1.schema.json
- schemas/hardsec-adaptive-oracle-v1.schema.json

The JSON schemas specify record shape; Rust performs cross-file joins and holdout integrity checks. Neither validates the truth authority or proves that declared source-group metadata is true.

## 5. Known-answer fixture

The fixture in tests/fixtures/hardsec/adaptive-evasion-v1/ contains nine test cases plus one training and one validation metadata case. Its synthetic cases deliberately include:
- three observed successful attack effects, of which one is alerted, one cleared and one abstained;
- one malicious attempt with no observed effect;
- one malicious attempt with unknown effect;
- four benign cases, including one false alert;
- three invalid sensor states, including one invalid-sensor clear verdict;
- complete per-case acquisition/inference cost values.

Expected evaluator-mechanics outputs:
- attack-effect-conditioned recall: 1/3;
- benign false-alert rate: 1/4;
- failed attack attempts excluded from successful-attack recall: 1;
- unknown-effect attempts excluded: 1;
- invalid-sensor cases: 3;
- invalid-sensor clear verdicts: 1;
- end-to-end measured cost sum: 1,080 ns.

These numbers are intentionally illustrative fixture behavior, not detection-performance results.

The fixture has fewer than ten independent groups on every reported axis. The expected statistical-support section therefore emits no confidence intervals, even though there are nine test cases. This is intentional: raw case count is not independent-group support.

Fourteen Rust unit tests cover known-answer outcomes, program/attack/adaptation-family holdout leakage, oracle/manifest class consistency, named-adaptation level semantics, attack-family macro reporting, duplicate/missing case IDs, failed/unknown effect semantics, invalid sensor clear verdicts, incomplete cost, calibration, adaptation-stratum reporting and evidence-domain separation.

## 6. Mandatory comparison arms

The evaluator is model-agnostic. Reproduction work should include:
- the original matched DT/RF HARPY-V/GEM5 baselines where the source protocol permits;
- T-SMADE's workload-classifier plus four-HPC detector;
- Tong et al. (2026) adversarial sample-augmentation method;
- a simple non-ML/statistical baseline;
- standard versus adversarial augmentation under matched tuning and group budgets;
- Symthaea HDC/CfC and any selective cascade as candidates, not presumed winners.

Important literature bounds:
- T-SMADE (2024) reports average attack TP rates of 95.42% for Evasive Spectre-NOP and 100% for Evasive Spectre-memory, with comparison-table non-attack TN rates of 87.49% and 88.89%; the paper also reports workload-specific TN reductions and explicitly leaves unseen evasion techniques / novel workloads unverified: https://www.mdpi.com/2079-9292/13/20/4090.
- Tong et al. (2026) report >99% detection accuracy and up to 1.7% runtime overhead on their tested Intel i7-7700 and AMD Ryzen 7-4800H platforms; combined hardware events reduce required adversarial samples by up to 50% and 94% on those respective platforms: https://doi.org/10.1016/j.engappai.2025.113634.
- Mandal et al. (2024) report 100% detection in their tested interleaving settings down to 10^-6; false-alert calibration, sample support and transfer must be independently audited: https://eprint.iacr.org/2024/1856.

These published results are not directly comparable across different processors, attack implementations, workloads, event sets and operating points. Reproduction requires source artifact, platform, kernel/perf, counter configuration, sampling, labels, benign TN/false-alert behavior, and full acquisition cost to be pinned.

## 7. Claim ceiling and current limitations

A successful receipt establishes conditional metrics for the exact supplied campaign, oracle, metadata, detector outputs and declared group IDs. It does not authenticate the oracle, prove attack-family completeness, prove true source lineage, establish physical transfer, or certify security.

No HARPY-V data was consumed. No Evasive Spectre or other attack was generated. No detector was trained, no published result was reproduced, and no physical telemetry was collected. Synthetic campaign evidence must remain separate from simulator and physical-hardware evidence.
