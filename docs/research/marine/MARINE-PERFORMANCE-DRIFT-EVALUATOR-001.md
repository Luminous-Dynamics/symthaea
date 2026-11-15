# MARINE-PERFORMANCE-DRIFT-EVALUATOR-001
## Independent candidate scoring for causal performance-drift monitoring

**Status:** research/evaluator contract v0.1.0  
**Parent:** MARINE-PERFORMANCE-DRIFT-001 #12198  
**Implementation:** #12227  
**Scope:** candidate outcome evaluation against a frozen, evaluator-only synthetic challenge corpus

> The v0.1.0 evaluator is a synthetic contract smoke test. It does not establish real-vessel accuracy, hydrodynamic validity, fouling diagnosis, cleaning efficacy, energy savings, safety, class approval, or operational permission.

## Why a second layer is necessary

The time-causal corpus auditor checks whether the benchmark itself is structurally usable: schema validity, model cutoff, observation and feature availability times, evidence timing, hidden-label isolation, and calibration status. Those checks do not answer a separate question:

**Did a candidate make the right decision, and did it have admissible evidence for the cause it claimed?**

A drift detector can achieve low residual error and still be dangerous if it labels weather, intake blockage, propulsor damage, sensor drift, or mixed conditions as biofouling. A model that abstains when evidence is insufficient should be evaluated differently from a model that guesses confidently.

The evaluator therefore scores candidate-visible outputs separately from evaluator-only challenge truth. Candidate output has a strict schema that rejects extra fields, including expected labels and truth values. It is a separate input file; truth is read only by the evaluator.

## Paradigm shift: residuals are observations, not explanations

Keep four layers distinct:

1. **Performance state:** drift detected, no drift detected, indeterminate, or out of domain.
2. **Candidate cause:** only a cause whose own evidence requirements are met.
3. **Evidence admissibility:** exact source reference, visible in the terminal decision snapshot, available by decision time, uniquely resolvable, and confirmed.
4. **Evaluation truth:** held outside the candidate artifact and used only by the evaluator.

Power, fuel-flow, speed, shaft-rpm, battery, and residual features can establish that performance changed. They are not, by themselves, independent proof of hull fouling or another physical cause.

## Candidate-output contract v0.1.0

One candidate result is required for every case, bound to the exact terminal decision timestamp. The evaluator rejects:

- missing, duplicate, or extra case predictions;
- corpus identity mismatch;
- non-matching terminal decision time;
- undeclared fields or evaluator-label leakage;
- a cause of `none` carrying causal evidence references.

The schema deliberately contains no confidence score. Confidence is not a substitute for calibration or evidence. A future version may add uncertainty only with a separately defined calibration and coverage protocol.

## Cause-specific evidence gates

| Candidate cause | Minimum evidence types |
|---|---|
| biofouling | confirmed hull inspection |
| weather_or_environment | confirmed weather observation |
| propulsor_damage_or_blockage | confirmed propulsor inspection |
| biofouling_cleaned | confirmed cleaning record **and** confirmed hull inspection |
| none | no cause claim; no supporting evidence references |

Every cited evidence ID must occur exactly once in the final decision point's candidate-visible evidence list, have a valid availability time no later than the decision, and carry confirmed status. Calibration-only evidence cannot support a biofouling attribution. Unknown, mixed, late, revoked, absent, or ambiguous evidence cannot be upgraded into a confident causal label.

The truth-side support flag is separately checked by the evaluator. This prevents a candidate from being credited for a cause that is not independently supported by the challenge record, even if it supplied a superficially relevant evidence reference.

## Metrics

The report includes:

- per-case expected/predicted disposition and cause result;
- disposition confusion matrix and exact disposition accuracy;
- abstention count;
- **coverage** = non-abstaining decisions / all cases;
- **selective accuracy** = correct dispositions among non-abstaining decisions / non-abstaining decisions;
- predicted causal attributions and attribution precision;
- unsupported attribution count;
- false biofouling attribution count.

Coverage must always be shown beside selective accuracy. High accuracy on a tiny selected subset must not be presented as fleet-wide performance. Every case remains in the denominator. Out-of-domain and indeterminate predictions count as abstentions, not silent exclusions.

## Frozen v0.1.0 smoke cases

The evaluator binds to the seven-case MARINE-PERFORMANCE-DRIFT-SYNTHETIC-0001 corpus:

- stable reference;
- weather/regime shift;
- biofouling drift;
- post-cleaning recovery;
- propulsor damage/blockage;
- sensor/calibration drift;
- mixed/unidentified causes.

The positive known-answer file demonstrates the report shape and correct evidence linkage. The negative-control file intentionally attributes weather, propulsor, sensor, and mixed cases to biofouling; it must be rejected and report all four false-fouling attributions. Neither file is a model prediction from a trained system.

## Comparison with current research and industry practice

- Gao, Wu & Tian, *Ocean Engineering* (15 October 2026), propose operating-condition recognition and event-triggered hierarchical hybrid-tug EMS. Their abstract reports 415.1 kg equivalent fuel versus 447.2 kg for REMS, fewer controller updates, and terminal SOC near 45% in their evaluated simulations. This is a relevant control/energy comparator, not an independent field result or a drift-attribution benchmark: https://doi.org/10.1016/j.oceaneng.2026.127569
- Lang et al., *Transportation Research Part D* 156 (July 2026), use physics-guided warm-up and incremental/sliding-window updates for ship biofouling performance assessment, compared with retraining and ISO 19030. The authors state that the underlying data cannot be shared, limiting independent reproduction: https://doi.org/10.1016/j.trd.2026.105364
- ISO 19030 is a relevant method reference for conventional ship performance monitoring, but its scope must not be transferred uncritically to PWC waterjets or tug azimuth/VSP propulsion.
- ISO 23247-6:2026 provides digital-twin composition guidance in manufacturing. It is a composition/interoperation reference, not a marine performance test method or vessel approval: https://www.iso.org/standard/87426.html

Our differentiation is not a claim to have a more accurate model than these studies. It is an independently executable evaluation contract for false-positive causal claims, evidence provenance, abstention, and selective risk. The evaluator will only become a model benchmark when separately developed candidates are run on held-out, rights-admitted physical data.

## Proposed future measured-data targets (not achieved)

After independent data admission, measurement-chain qualification, and a pre-registered test protocol:

- zero false biofouling attribution in every declared no-fouling confounder stratum;
- at least 95% precision among causal attributions, with coverage always disclosed;
- at least 90% drift recall on independently labelled, in-domain cases;
- worst-slice uncertainty coverage within a predeclared tolerance;
- zero future-data leakage and zero unsupported causal attributions.

These are aspirational gates, not current results. They must be revised if instrument uncertainty, sample support, class/operating domain, or independent labels make them statistically indefensible.

## Explicit v0.1 limitations and next increment

The current candidate file scores one terminal result per case. It does not yet score intermediate time-to-detection, false-alarm exposure over long durations, interval coverage, calibration drift, detection delay, or post-intervention counterfactual recovery. Those require per-decision-point truth labels with explicit label-availability semantics, held-out vessel/campaign partitions, and a larger adversarial temporal corpus. They are reserved for a versioned follow-on rather than inferred from seven synthetic terminal labels.

## Claim ceiling

A clean positive smoke fixture means only that the evaluator can parse the frozen candidate contract and score the known-answer synthetic cases. The negative control must demonstrate that unsupported fouling diagnoses are rejected. No result here proves vessel condition, model accuracy, hydrodynamic fidelity, energy savings, maintenance efficacy, or safe physical control.
