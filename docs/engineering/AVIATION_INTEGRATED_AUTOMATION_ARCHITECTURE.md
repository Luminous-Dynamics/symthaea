# Airline-Wide Automation Architecture (Research Plan)

**Status:** Draft research architecture, not a deployment specification.  
**Companion implementation:** [Aviation enterprise benchmark PR #8586](https://github.com/Luminous-Dynamics/symthaea/pull/8586)  
**First evidence-integrity gate:** [AVIATION-ENTERPRISE-013 / issue #8840](https://github.com/Luminous-Dynamics/symthaea/issues/8840)  
**Claim ceiling:** human-review decision support. This document does not grant dispatch authority, maintenance release, airworthiness approval, or aircraft-control authority.

## 1. Objective: one coordinated airline, not one giant optimizer

The research direction is an airline-wide decision fabric that coordinates planning and recovery across the airline, airports, ground handlers, air-navigation partners, suppliers, and passengers. It should not replace every operational system with a monolithic model. It should connect specialist services through versioned state, explicit constraints, independently checkable recommendations, and auditable outcomes.

The key architecture shift is to treat every recommendation as a **versioned decision transaction**:

1. Observe a time-stamped state and record source, availability time, quality, and identity.
2. Construct a frozen decision snapshot from source data, configuration, policy, and model versions.
3. Generate multiple feasible candidates; preserve hard constraints as constraints, not objective penalties.
4. Recompute candidate feasibility and cost with an independent verification path.
5. Evaluate distribution shift, uncertainty, tail loss, network externalities, and per-stakeholder harm.
6. Bind the decision and its verifier receipts to an immutable evidence manifest.
7. Return a recommendation, complete blockers, expected consequences, and a safe fallback to the authorized human or existing system.
8. Record the selected action and observed outcome as new evidence, without rewriting the prior decision record.

The unit of trust is the entire decision transaction, not a model score, a dashboard status, or a passing local unit test.

## 2. Why this integration problem is worth solving

Airline operations already use collaborative systems. EUROCONTROL's Airport Collaborative Decision Making (A-CDM) work links inbound and outbound flights through operational milestones and shared turn-round information. Its Capacity and Weather-Based Operations (CWBO) programme coordinates forecast weather, demand, capacity, and network-instability triggers. These are important comparators and integration surfaces—not systems we should claim to have replaced.

EUROCONTROL reported that CWBO saved about 3.1 million ATFM delay minutes and an estimated €400 million in operational costs during 1 May–31 August 2026, with a reported 14% increase in network stability on challenging days. Those are reported programme outcomes under EUROCONTROL's context and costing assumptions. They are not Symthaea results, and they should not be compared directly to synthetic costs.

Research on airline recovery also demonstrates that the subsystems interact: aircraft recovery affects crew feasibility, which changes passenger reaccommodation, which changes downstream load and cost. The Transportation Science study by Hızır, Barnhart, and Vaze (online 23 December 2025; 2026 journal issue) evaluates learning-guided mixed-integer crew recovery on an actual major-U.S.-airline schedule exceeding 2,800 daily flights. That is a materially stronger comparator than a small synthetic fixture. Any comparison must preserve its own problem definition and data assumptions rather than imply a direct leaderboard score.

## 3. System boundary and capability map

The intended system is a federation of bounded decision services. Existing airline, airport, aircraft, maintenance, crew, and air-traffic systems remain authoritative for their respective facts and regulated approvals.

| Domain | Initial automation target | Hard boundaries and verification |
|---|---|---|
| Schedule and fleet planning | Forecast schedule fragility, identify robust aircraft rotations, surface capacity conflicts | Aircraft type and capability, airport/route constraints, approved operating limits, schedule version |
| Disruption prediction | Predict delay, cancellations, missed connections, capacity loss, and recovery time; detect when calibration is stale | Availability-time rules, held-out time blocks, calibrated uncertainty, explicit out-of-support alarm |
| Integrated recovery | Generate aircraft, crew, passenger, and cargo recovery candidates jointly or through coordinated subproblems | Crew qualification/duty legality, aircraft availability, maintenance restrictions, passenger connection feasibility; independent feasibility verification |
| Crew and rostering | Identify legal replacements and robust reserve positioning; explain why assignments changed | Contractual rules, qualification/recency, rest and duty constraints; final authority remains in approved crew systems |
| Maintenance and parts | Predict material shortages, plan qualified part positioning, sequence maintenance logistics | Configuration and serial/lot traceability, approved maintenance data, licensed personnel, maintenance-release authority |
| Airport turnaround | Predict milestone times and coordinate gate, stand, ground-handler, fueling, cleaning, catering, boarding, and baggage dependencies | Airport/handler ownership, access permissions, milestone lineage, resource capacity and safety zones |
| Network and airspace | Evaluate rerouting/re-timing candidates against weather, capacity, and network propagation | ATC/ANSP instructions and capacity restrictions remain authoritative; no autonomous airspace authority |
| Fuel and environment | Compare fuel, taxi, routing, delay, noise, and CO2e trade-offs | Approved performance data and operational limits; disclose units and conversion sources; do not hide CO2e in monetary cost without a declared conversion |
| Passenger service | Detect misconnect risk, propose reaccommodation and assistance, estimate arrival-time and access needs | Accessibility, protected personal data, policy and legal entitlements, transparent prioritization and human escalation |
| Cargo and special loads | Reallocate capacity and recover missed connections | Dangerous-goods requirements, temperature/handling limits, load and balance approvals |
| Commercial and finance | Reforecast revenue, disruption costs, compensation exposure, procurement, and supplier risk | Separate forecasts from accounting records; protect customer data; preserve auditability |
| Cybersecurity and resilience | Detect stale feeds, credential misuse, dependency failures, malicious data changes, and degraded modes | Least privilege, signed source receipts, provenance, incident response, fail-closed authorization |
| Workforce and governance | Explain decisions, measure workload, record overrides and adverse impacts | No hidden objective changes; role-based approval, appeal/escalation paths, privacy and labor constraints |

This table is a coverage map, not a claim that each capability already exists in the repository. The immediate implementation focus remains evidence integrity and synthetic decision composition.

## 4. Shared decision contract

Every domain adapter should map its data into a common, versioned contract without erasing domain-specific semantics.

### 4.1 Event and state identity

Each event should include, where applicable:

- A stable source event ID and source-system identity.
- Event time, observation/ingestion time, and the time the fact became available to the decision-maker.
- A schema version and source snapshot digest.
- Entity IDs with explicit namespaces; do not infer that equal strings mean equal entities across systems.
- Data-quality and missingness indicators, including known clock uncertainty.
- The policy/configuration, model, solver build, and data versions that consumed it.
- Correction or supersession links; never silently overwrite an earlier evidence event.

Reject duplicate IDs, invalid chronology, unsupported schema changes, future-available features, and incomplete actor universes. A source being syntactically valid is not proof that its content is fresh or true.

### 4.2 Immutable run manifest and trusted receipts

The next implementation increment is issue #8840. The manifest must bind the same source revision/tree, data snapshot, scenario, model, policy/configuration, plan, calibration artifact, drift report, feasibility verification, distributional-risk assessment, and network-harm assessment used by the readiness gate.

Each verifier receipt must bind the exact manifest digest and role-specific subject digest, identify the verifier binary and trusted key, include an expiry, and have a signature checked against an independently provisioned trust store. A hash establishes byte identity; it does not establish who created those bytes. A signature over a self-generated artifact is not trusted unless the public key and verifier identity are anchored outside the artifact itself.

Unknown, duplicate, missing, stale, future-dated, malformed, incorrectly bound, invalidly signed, untrusted-key, and binary-mismatched receipts must block promotion. Keep all blockers in deterministic order. The JSON manifest should make freshness ages, TTL policies, digest relationships, verifier identities, and verification mode auditable.

### 4.3 Gate semantics

The existing readiness gate is a *composition gate*. Its strongest positive state remains EligibleForHumanReview; this only means the supplied evidence passed declared checks. It is not dispatch authorization or proof of operational safety. Missing or indeterminate evidence is never converted to an acceptable default.

The eventual operator adapter must make the authorization boundary explicit:
- **Observe:** ingest facts, identify conflicts, and monitor evidence freshness.
- **Recommend:** produce ranked alternatives and counterfactuals with explanations.
- **Confirm:** let an authorized person approve a bounded action.
- **Pre-authorized bounded execution:** only for a specifically allowlisted, reversible operation with a tested rollback path, explicit scope, expiry, and independent policy check.
- **Regulated or safety-critical authority:** remains out of scope until the relevant operator, engineering, safety, and regulatory approval process establishes a separate basis for it.

Do not let a successful recommendation silently inherit authority to execute.

## 5. Optimization without hiding trade-offs

A single weighted cost can improve an airline's aggregate result while shifting harm onto passengers, crew, airports, or another airline. Preserve a vector of outcomes and use a declared policy to choose among eligible candidates.

**Hard feasibility constraints** include operational permissions and applicable legality/safety constraints; they are not softened to improve a score. The exact list must come from the relevant authoritative system and domain experts.

**Decision metrics** should be reported separately:
- Total operating cost under a frozen, sourced cost model.
- Cancellations and delay minutes, separating non-cancelled delay from cancellation penalties to avoid double counting.
- Passenger final-arrival delay, missed connections, overnight stranding, and reaccommodation burden.
- Crew reassignment cost, reserve usage, and legality failures (the target for legality violations is zero).
- Maintenance disruption, parts availability, and overdue or infeasible work (the optimization must never override release authority).
- Fuel burn and CO2e, with conversion assumptions explicit.
- Worst-scenario cost, upper-tail mean (CVaR/expected shortfall under a declared convention), worst stakeholder regression, and harm by stakeholder domain.
- Recovery duration and the time at which a stable schedule is restored.
- Solver runtime, feasibility rate, timeout rate, and quality versus a feasible baseline.
- Human workload, override rate, acceptance rate, explanation completeness, and cost of abstention.

Keep the Pareto frontier visible for cost, cancellations, passenger outcomes, CO2e, and worst-actor harm. If a policy selects one candidate from that frontier, publish the policy and its limits. A Pareto frontier does not itself determine what is fair.

## 6. Evaluation ladder and benchmark design

### Tier A — executable contracts and adversarial synthetic fixtures

Start with small cases whose correct answers can be exhaustively enumerated. Include zero-severity controls and inject each fault alone and in combination: missing/stale/future timestamps; wrong digests; invalid signatures; changed verifier binaries; duplicate identities; leakage between train/calibration/test episodes; out-of-support scenarios; feed outages; conflicting source versions; and a solver that returns a feasible-looking but invalid plan.

Initial acceptance target: **zero false promotions on the declared fault-injection suite**. This is a contract target, not a probability guarantee. Preserve all failures and make sure the positive fixture reaches EligibleForHumanReview only when every required evidence role is valid.

### Tier B — public-data forecasts and policy replay

Use only data for which source rights, schema, timestamps, coverage, and limitations are documented. Candidate sources include published EUROCONTROL taxi-time/CODA analysis and airport/network operations publications; an eventual adapter must verify access terms and the actual availability of raw fields before claiming replication.

For forecasting, compare against persistence, seasonal-naive, linear/regularized models, and a strong domain-specific model. Report MAE/RMSE where appropriate, quantile loss or weighted interval score for distributions, calibration coverage/width, and performance by airport, season, aircraft/wake category, weather regime, and disruption severity. A low average error cannot hide dangerous under-planning in a tail or a sparse airport cohort.

For policy replay, use:
1. Do nothing / current plan.
2. Transparent rule-based or greedy recovery.
3. A conventional integrated MIP/CP-SAT model where a matched formulation is available.
4. Learning-guided MIP and integrated recovery comparators from the literature.
5. A hindsight oracle only as an explicitly labeled upper bound—not a deployable baseline.
6. Our candidate pipeline, including ablations that remove drift checks, robust risk, network externality constraints, or evidence gates.

Split data by event episode and time, not random rows alone. Fit transforms and calibrators only on permitted partitions. Keep a final, locked temporal test set. Compare candidates on identical input snapshots, feasible-action sets, runtime budgets, and cost definitions.

### Tier C — preregistered shift and network stress tests

Freeze the event families, thresholds, alpha, detection deadline, reset policy, and metric aggregation before the final run. Include abrupt, gradual, seasonal, airport-cohort, compound-weather/capacity, latency, missing-feed, and outside-support shifts.

Compare the current frozen-reference mixture likelihood-ratio detector with CUSUM, Page–Hinkley, EWMA, predictive-rank martingales, and rolling conformal recalibration. Report:
- False-alarm behavior under each detector's stated assumptions.
- Detection delay and miss rate by shift family and magnitude.
- Time to recalibration and useful-intervention rate.
- Time abstaining and the cost of the safe fallback.
- Post-shift tail cost and harm.
- Runtime/memory overhead and sensitivity to serial dependence.

The proposed target of at least 90% detection of preregistered operationally material shifts within a fixed deadline is a goal only. The false-alarm budget, deadline, and tolerated abstention cost must be stated numerically and frozen before evaluation. An anytime-valid result under an i.i.d. null must not be presented as a guarantee under arbitrary aviation time-series dependence.

### Tier D — authorized shadow operations

Only with an airline/airport partner and explicit data authorization, run a time-synchronized shadow pipeline against the actual decisions and observed outcomes. Do not write into operational systems. Compare on matched route-day/airport/season or disruption episodes and account for clustering in confidence intervals. Record missingness, manual overrides, operational changes during the study, and all exclusions. Publish negative results and failed monotonicity checks.

### Tier E — bounded pilot

A pilot requires an approved safety assessment, cyber and privacy review, operational owner, rollback design, incident process, training, monitoring, and the relevant organizational/regulatory approval. Start with a narrow reversible administrative action—not dispatch or maintenance release. Expand the action scope only after an independently reviewed evidence package passes preregistered criteria.

## 7. Proposed initial targets (not achieved results)

| Dimension | Initial target | How to avoid gaming it |
|---|---|---|
| Evidence gate | 0 false promotions on the specified injected-fault matrix | Mutate every signed field and every subject artifact; independently recompute digests |
| Plan verification | 100% detection of intentionally introduced hard-constraint violations in the declared oracle suite | Use a separately written oracle and adversarial ID/cost/overflow cases |
| Drift detection | ≥90% detection of preregistered material shifts by the frozen deadline | Report false alarms, missed shifts, abstention cost, serial dependence, and all shift families |
| Forecast calibration | Meet preregistered coverage and interval-width targets on the locked time holdout | Report cohort and tail results; no test tuning or favorable airport-only selection |
| Recovery quality | Improve the declared policy objective against a feasible, matched baseline under identical runtime budgets | Report cost vector, Pareto front, feasibility, tail loss, and worst-actor regression |
| Operational speed | Meet a fixed decision latency budget at p50/p95/p99 | Include queueing, feed staleness, verification, signing, and fallback time |
| Robustness | No automatic promotion with stale/missing inputs or out-of-support events | Inject latency, clock skew, duplicate events, source disagreement, and feed silence |
| Human factors | Track override, acceptance, time-to-decision, and intervention burden | Predefine denominators and survey protocol; do not equate low override rate with correctness |
| Generalization | Retain performance on unseen airport/season/disruption cohorts | Hold out entire cohorts and time periods; state the scope of generalization |
| Runtime reliability | Deterministic manifest for the same frozen inputs/config; all actions idempotent | Repeat runs and independently compare canonical artifacts |

These targets are deliberately not collapsed into one success metric. A system that saves average cost but violates hard constraints, causes unreported tail harm, or promotes untrusted evidence fails.

## 8. Current research status and immediate sequence

The current aviation branch has a tiny exact planning fixture, separately written in-crate verifier/oracles, paired and network cost evaluation, finite-set distributionally robust assessment, calibration split audits, finite-sample TV-radius calibration, online drift monitoring, and a composed freshness gate. The public report says its Rust tests are defined but have not executed in this environment; PR-related workflow conclusions observed for head e7d96fff0b610eec065f927d23bc8a9bfc1127a7 were skipped, not passed. That is a hard evidence limitation, not a footnote.

Recommended order:

1. **Evidence authenticity first:** implement issue #8840: immutable manifest, signed role-specific receipts, independently provisioned trust anchors, exact digest binding, deterministic output, and negative controls.
2. **Execute the exact head:** use a Rust-equipped environment to run formatting, locked package tests, examples, and an independently written parser/verifier. Record the exact tested commit SHA and run artifact digests.
3. **Public-data adapter:** add a timestamp-safe source adapter with explicit field availability, licensing, event clustering, snapshot identity, and repeatable manifest generation.
4. **Baseline implementation:** compare the same recovery instances against feasible greedy and integrated MIP/CP-SAT baselines; report solution quality versus runtime and infeasibility.
5. **Event-stream simulation:** add rolling-horizon recourse, disruption arrival processes, value-of-information, and sensitivity to forecast error.
6. **Stakeholder and tail accounting:** retain actor-level outcomes and a visible Pareto frontier across the same frozen scenarios.
7. **External evidence:** seek authorized operator data and independent review only after public replay is reproducible.

Do not expand the code surface by adding more clever optimizers before the current evidence chain can be independently reproduced.

## 9. Reference points

- EUROCONTROL, [Capacity and Weather-Based Operations](https://www.eurocontrol.int/service/capacity-and-weather-based-operations-cwbo) and [2026 results paper](https://www.eurocontrol.int/publication/how-capacity-and-weather-based-operations-cwbo-are-building-network-resilience-capacity), 14 September 2026.
- EUROCONTROL, [Airport Collaborative Decision Making (A-CDM)](https://www.eurocontrol.int/concept/airport-collaborative-decision-making) and [A-CDM Specification](https://www.eurocontrol.int/publication/eurocontrol-specification-airport-collaborative-decision-making), 30 January 2025.
- EASA, [Artificial Intelligence Roadmap](https://www.easa.europa.eu/en/domains/research-innovation/ai) and [Concept Paper on Artificial Intelligence, Proposed Issue 03](https://www.easa.europa.eu/en/newsroom-and-events/news/easa-releases-latest-issue-its-concept-paper-artificial-intelligence), 3 June 2026.
- Hızır, Barnhart, and Vaze, [Large-Scale Airline Crew Recovery Using Mixed-Integer Optimization and Supervised Machine Learning](https://doi.org/10.1287/trsc.2025.0105), published online 23 December 2025; Transportation Science 60(2), 2026.
- Momtaz, [Rolling-Calibrated Conformal Decision Support for Seasonal Airport Taxi-Time Planning under Distribution Shift](https://doi.org/10.19139/soic-2310-5070-3782), 25 June 2026.
- Huang, Ma, and Michailidis, [Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift](https://proceedings.mlr.press/v337/huang26b.html), UAI 2026.
- [Addressing the systemic complexity of airline irregular operations: Toward integrated schedule recovery](https://doi.org/10.1016/j.jairtraman.2025.102925), Journal of Air Transport Management, March 2026.

These sources define research and operational comparators, not evidence that Symthaea has reproduced their results.
