# SOH Adaptation Portfolio and Target-Data Budget Audit v0.1

**Status:** Rust-first benchmark protocol/evidence audit. It does not train experts or establish physical SOH transfer.
**Owner:** [EV-SOH-ADAPTATION-PORTFOLIO-001 #11489](https://github.com/Luminous-Dynamics/symthaea/issues/11489).
**Related:** [SOH benchmark manifest v0.1 #11342](https://github.com/Luminous-Dynamics/symthaea/pull/11342), [support-conditioned calibration #11390](https://github.com/Luminous-Dynamics/symthaea/pull/11390).

## Research motivation

Liu et al., *Domain-Adaptive Mixture-of-Experts for Cross-Dataset Lithium-Ion Battery State-of-Health Prediction via Adaptive Strategy Selection*, *Batteries* 2026, 12(9), 359 (10 September 2026): https://www.mdpi.com/2313-0105/12/9/359

The study evaluates 564 cells from seven public datasets under leave-one-domain-out cross-validation. It reports average R²=0.795 with the gate trained while the target domain is held out, and 0.864 with oracle strategy alignment. The gate has 32 parameters and selects among zero-shot, test-time adaptation, supervised fine-tuning and MAML. Hard argmax routing beats weighted fusion by an average 0.027 R² in the paper's experiment.

Crucial qualification: the four experts do not share one target-data budget. The methods include zero target updates for zero-shot, unlabeled target adaptation for TTT, an 80/20 labeled target split for fine-tuning, and a labeled target support set for MAML. Therefore the 0.795 result is a label-budget-conditioned strategy-selection result, not a pure zero-shot score. The target-data budget of every expert and the router must be described separately.

## Paradigm

Do not flatten the whole portfolio into one adaptation on/off flag. The router, each expert and the final evaluation partition have distinct data access and chronology.

This manifest records:
- source-domain groups used to fit the router;
- target-domain support groups available for adaptation;
- final target-domain evaluation groups, untouched by router and expert adaptation;
- per-router feature identity, evidence and availability time;
- per-expert strategy, model, group identities, labeled/unlabeled support budget, update budget and execution cost;
- router-selected strategy or explicit abstention;
- completion time for each candidate expert's adaptation/readiness, before the final test is first accessed.

Each expert is evaluated on exactly the same physical target group set. The final test groups must be disjoint from source training and every adaptation group.

## Strategy-specific contracts

| Strategy | Target labels | Unlabeled target groups | Required evidence |
|---|---:|---:|---|
| Zero-shot | 0 | 0 | Model digest; no target updates |
| Unlabeled TTA | 0 | Positive | Exact target support groups; no labeled target evidence; update budget |
| Supervised fine-tune | Positive | 0 in v0.1 | Exact labeled support groups and label evidence digest; update budget |
| Few-shot MAML | At least declared support minimum | 0 in v0.1 | Exact labeled support group IDs; minimum ≥2; label evidence |
| Physical/statistical | 0 | 0 | Fixed/reference estimator identity; no target adaptation |
| Abstain | 0 | 0 | Explicit no-eligible-expert outcome; no model result digest |

Budgets are counts of independent physical groups (e.g., cells for unseen-cell transfer), not arbitrary windows. If a method also uses within-cell samples or partial-charge prefixes, a future protocol version should record those observation budgets and chronology separately. A label evidence digest binds acquired target reference measurements; it does not independently prove their accuracy or rights.

## Hard gates

- Split is frozen before router selection; adaptation/readiness completes before final test access.
- Router training groups are a subset of declared source domains only.
- Router features are label-free, not derived from target reference labels, and available no later than selection time.
- Target adaptation groups belong to the declared target support pool and are disjoint from final evaluation groups and source training groups.
- Every candidate uses the same exact final evaluation group identities.
- Zero-shot has zero target support groups, zero target labels and zero model updates.
- Unlabeled TTA has positive unlabeled budget, zero labeled budget and no label evidence digest.
- Supervised fine-tuning and MAML have positive labeled budgets, label evidence digests and nonzero update budgets.
- MAML must meet its declared labeled support minimum.
- Exactly one predictive expert is selected, or the portfolio explicitly abstains. An abstain portfolio cannot claim a prediction result.
- Wall time, memory and optional energy are recorded for every expert.

## Frozen synthetic corpus

Sixteen cases: five accepted portfolios (zero-shot, unlabeled TTA, supervised fine-tuning, few-shot MAML and explicit abstention) and eleven rejected controls (router test-label leakage, target-label-derived router features, adaptation/test overlap, expert evaluation-set mismatch, zero-shot budget inflation, unlabeled TTA claiming labels, missing fine-tuning label evidence, feature available after selection, adaptation completing after test access, MAML below its support minimum and an abstention that improperly claims a prediction result).

These fixtures validate protocol gates only; they are not model-performance or physical battery evidence.

## Benchmark scorecards

Publish separate performance-vs-budget scorecards:
1. Zero target labels and zero adaptation.
2. Unlabeled target prefixes/TTA, with exact group and observation budget.
3. Few-shot labeled support, with exact reference-test burden.
4. Larger supervised target fine-tuning.
5. Physical/statistical baseline and explicit abstention.

For each regime report mean and worst-domain MAE/RMSE/R², support-conditioned 90/95 interval coverage and width, NLL/CRPS, OOD/selective risk-coverage, target-label/unlabeled data burden, adaptation compute/time, memory/energy, router selection performance, oracle gap, regret versus the best fixed eligible strategy and no-estimate frequency. Compare hard argmax with weighted fusion.

The oracle router is an offline upper bound, not a deployable baseline. Do not claim one SOTA number across incomparable target-label budgets or physical split protocols.

## Limits

The audit verifies internal consistency of producer-declared physical group digests and strategy budgets. It does not independently verify that digests map to real batteries, reproduce a published model, validate target labels, or establish cross-domain accuracy. Physical transfer claims require blind held-out assets and independently sourced labels.
