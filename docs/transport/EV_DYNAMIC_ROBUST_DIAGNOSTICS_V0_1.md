# Dynamic robust EV battery diagnostics v0.1

## Why this is a separate research increment

The existing EV battery decision stack already has:
- a sensor/observation provenance contract (#11249);
- a decision-focused one-step VOI selector (#11277);
- a finite-horizon, static latent-state planner (#11325);
- joint-likelihood support for correlated tests (#11512);
- battery SOH transfer/calibration work (#11252);
- a diagnostic outcome and intervention evidence programme (#11289);
- a fixed-pulse conservative power envelope (#11384) and a model-derived pack/cell β-power evaluator (#11438);
- an upcoming SOC-coupled 2-RC electrothermal/active-identification owner (#11485).

This issue (#11404) owns the missing dynamic regime: **the battery state and test context can change while diagnosis is taking place, and the model that predicts those changes may itself be uncertain.**

This v0.1 implementation is a bounded exact reference, not a general POMDP solver and not a learned diagnostic agent.

## Paradigm shift

A diagnostic plan is not merely a decision tree over a fixed latent state. It is a policy over a belief state that evolves with time and test actions:

`belief_t + elapsed time + operating context + test perturbation + observation -> belief_(t+1)`

A highly informative test can have negative decision value when its duration allows a battery to cross a service threshold. A test can change temperature/SOC/polarization. A timeout is an observation, not a row to discard. A test model can be too uncertain or outside its support domain to justify a confident decision.

The planner must therefore choose between **measure, act, continue monitoring, and stop/abstain** while accounting for the state transitions caused by elapsed time and by the test itself.

## v0.1 bounded exact model

To make every result independently replayable, v0.1 constrains the problem to:
- exactly two latent states;
- one or two terminal actions with non-negative declared losses;
- at most two total tests;
- a single optional follow-up test after an explicitly declared first-test outcome;
- two or three outcomes per test, including timeout/equipment failure where relevant;
- one to four finite transition/likelihood ambiguity models.

Each test binds duration, protocol, equipment, calibration evidence reference, cost, downtime cost, safety approval, calibration validity, supported domain and equipment availability.

Each world/model binds a version, model evidence reference and digest, domain, source class, prior, transition/perturbation models and outcome likelihoods. The v0.1 source class is deliberately restricted to `SYNTHETIC_KNOWN_TRUTH`.

For each test:
1. `time_transition` maps the pre-test state to the state after the exact declared test duration.
2. `perturbation_transition` maps the elapsed-time state to the state after the test has perturbed the system.
3. `observation_by_post_state` gives the outcome likelihood conditional on the final post-test state.

The transition model's duration must exactly match the test specification within the declared numerical tolerance. A test cannot silently borrow a transition calibrated for a different duration.

### Dependence is explicit

v0.1 only accepts the declared assumption `CONDITIONAL_INDEPENDENT_GIVEN_POST_STATE` and requires an empty shared-nuisance-factor set. If tests share calibration bias, sensor hardware, excitation, temperature, wiring or a feature pipeline, this model is not admissible. Do not approximate that situation by multiplying marginal likelihoods. Compose with the joint assignment/conditional likelihood owner in #11512 instead. Unknown fields are rejected, not ignored.

### Conditional follow-up availability

A follow-up test names:
- the exact predecessor test;
- the first-test outcomes under which it is available.

The enumerator cannot place it before the predecessor or on an undeclared outcome branch. v0.1 supports one follow-up only; it does not represent an arbitrary diagnostic campaign.

### Robust objective

A single policy is selected to minimize the **worst expected total loss** across the declared finite model set. The report preserves expected loss and test burden for each model, the robust loss, and comparison against a static-state policy evaluated in the dynamic worlds.

Hard test eligibility gates are non-compensable: safety approval, calibration validity, supported domain and available equipment. An ineligible test never enters policy enumeration.

## Frozen synthetic known-truth scenarios

The corpus includes five problems:

1. **Long test crosses service threshold.** During a 600-second test, a healthy state transitions to degraded. A static-state planner prefers the informative test; the dynamic planner chooses NO_TEST because the test delays the decision until service is required anyway. The synthetic dynamic loss is 3.0; the static test policy's loss when replayed in the dynamic world is 3.2.
2. **Test-induced state perturbation.** A diagnostic pulse perturbs healthy to degraded after the elapsed-time transition. The dynamic planner declines the pulse while the static comparator selects it. Synthetic dynamic loss is 3.0 versus 3.15 for the static policy replayed under perturbation.
3. **Likelihood ambiguity.** One ambiguity model has an informative screen; another has no decision-relevant signal. The robust policy chooses NO_TEST rather than relying on the nominal model alone.
4. **Timeout is observed.** A diagnostic timeout is an explicit observation branch. The policy must retain and assign a terminal decision to it.
5. **Conditional follow-up.** A weak initial screen enables a reference test only on the `flagged` branch. The policy uses the follow-up after `flagged` but stops after `normal`.

All values, models, test costs, protocol/equipment references, keys and evidence digests in this corpus are synthetic. They validate the exact planner contract only; they are not measured battery transition rates, real test costs, or physically qualified test likelihoods.

## Baselines and benchmark tournament

The first exact comparator set is:
- no test / current evidence only;
- static-state finite-horizon planner (#11325);
- static-state planner with explicit joint outcome likelihoods (#11512) where the relevant correlation model is admitted;
- dynamic finite-state oracle;
- robust dynamic policy over a finite model ambiguity set;
- greedy one-step EVSI (#11277);
- maximum information gain/entropy as a diagnostic-only baseline;
- fixed periodic and random eligible tests on equal test/downtime budgets.

The exact dynamic oracle is an upper-bound reference for the frozen finite model, not a deployable estimator. Report:
- expected and worst-model decision loss/regret;
- worst-case tail loss/CVaR once there are sufficient independent episodes;
- missed-service / false-escalation rates;
- test count, duration, downtime, direct cost and time-to-decision;
- timeout/equipment-fault and abstention rates;
- model-misspecification and OOD detection;
- policy-tree size, enumeration nodes, runtime and memory;
- calibration/support quality by asset, chemistry, temperature, SOC, age, firmware, protocol and test equipment.

A fair industry comparison must use the same physical packs, test protocols, temperature/SOC windows, evidence access, decision alternatives, safety constraints and cost accounting. No result from synthetic models should be ranked against physical fleet performance.

## Proposed research targets (not achieved results)

For a future physical, independently admitted corpus:
- zero selection of tests missing safety, equipment, calibration or domain prerequisites;
- zero false certainty when required transition/likelihood support is absent;
- at least 20% lower worst-case decision regret than the strongest static/greedy baseline at equal safe test and downtime budget, with no increase in hard-constraint violations;
- calibrated 90%/95% predictive transition and outcome intervals, reported per group-disjoint asset/domain rather than pooled only;
- explicit selective-risk/coverage curves and an honest abstention rate;
- demonstrate decision-value improvement over fixed HPPC-style scheduling before claiming active-identification superiority.

These are research gates to preregister, not certification criteria and not claims about the present synthetic fixture results.

## Relation to industry

The industry frontier is already broader than SOH regression:

- Bosch Battery in the Cloud describes fleet-connected digital battery twins, anomaly detection, ageing prediction, usage certificates, lifetime optimization and battery-passport support.
- AVILOO offers rapid and full-cycle EV battery assessment, pack/cell diagnostics and comparative vehicle-class certificates. Its website claims that at least 95% of reference measurements have maximum SOH deviation no greater than ±3%; it also says its test is a point-in-time state assessment and cannot detect defects not exposed through vehicle diagnostics. This is a vendor-published claim and must be independently reproduced under a matched protocol before comparison.
- Qnovo describes edge-centric, physics-plus-ML battery analytics, adaptive charging and closed-loop changes to charge rate and depth of discharge. Its own published safety/accuracy/lifetime percentages are vendor claims, not independent benchmark results.
- TWAICE markets BESS analytics for SoC imbalance, self-discharge, DC resistance, round-trip efficiency, availability and operational recommendations.
- Nature Energy (25 August 2026) reports a fleet-scale cell-inconsistency analysis across 116 passenger EVs and 17 buses, over 3+ years and up to 300,000 km, and defines pack/cell utilization metrics including β_power. Its dataset and physical cohort are an important external benchmark anchor, not a directly comparable score for this synthetic model.

Differentiation cannot be “we have a digital twin,” “we predict SOH,” or “we optimize charging.” Established vendors already describe those capabilities. A stronger measurable claim is **evidence-qualified decision reliability**: preserve data provenance, authenticate producer and freshness, keep sensor truth/calibration separate from cryptographic authenticity, model state evolution and test perturbation, choose tests by service-decision regret, measure intervention outcomes independently, and retain lifecycle/passport lineage. Those pieces must still be integrated and physically validated; the present PR is only one exact reference slice.

## Integration and claim boundaries

Promotion sequence:

physical event -> observation provenance (#11249) -> signature/producer authority and replay check (Symtropy #4467 / draft PR #4656) -> physical/calibration admissibility -> estimator/pack model support (#11252, #11271, #11485) -> dynamic diagnostic decision (#11404) -> independently approved intervention/outcome (#11289) -> lifecycle/passport projection (Mycelix #6062/#6063).

A valid signature does not establish physical measurement truth. A selected test is a proposal, not authorization or execution. The planner cannot alter BMS settings, contactors, charging, torque, thermal controls or vehicle operation.

Passing this exact synthetic oracle establishes decision-theory and software contract behavior for the declared two-state model. It does not establish physical transition-kernel accuracy, sensor/likelihood calibration, battery safety, reduced failure rates, SOTA performance, automotive qualification or standards conformance.
