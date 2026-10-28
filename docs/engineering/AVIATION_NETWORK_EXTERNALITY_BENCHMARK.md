# Aviation Network Externalities and Harm-Constrained Selection (AVNX-001)

**Status:** research implementation; Rust tests are defined but not executed in this environment. No operational policy or savings claim is made.

## Hypothesis

A local optimizer may reduce the initiating airline's cost while moving cost, delay, emissions or disruption to airports, passengers, air navigation providers, maintenance suppliers or other airlines. A paired actor ledger plus a predeclared harm constraint should make this displacement observable and prevent the candidate selector from choosing plans that violate that declared constraint.

This is not a claim that a single harm threshold defines social welfare. The maximum per-actor regression and minimum system benefit are explicit governance parameters; they require stakeholder review and preregistration. Compensation, rights-based constraints and distributional weights are separate future work.

## Current industry comparators

- **EUROCONTROL Capacity and Weather-Based Operations (CWBO):** the 14 September 2026 report describes network-wide weather/capacity/instability triggers and reports 3.1 million delay minutes saved from 1 May to 31 August 2026, €400 million in estimated operational cost savings using €130/minute, and 14% higher network stability on challenging days. These are EUROCONTROL-reported program results, not directly comparable targets for a synthetic local benchmark. https://www.eurocontrol.int/publication/how-capacity-and-weather-based-operations-cwbo-are-building-network-resilience-capacity
- **Integrated airline recovery:** a 2026 review characterizes irregular operations as an interconnected supply-demand imbalance with interdependent recovery actions and network propagation. https://doi.org/10.1016/j.jairtraman.2025.102925
- **Efficiency/fairness in ATFM delay assignment:** a 2020 Journal of Air Transport Management study explicitly analyzes trade-offs using both flight and passenger metrics. This supports separate distributional reporting rather than a single airline-cost score. https://doi.org/10.1016/j.jairtraman.2019.101758
- **Integrated air-rail recovery:** an August 2026 Journal of Air Transport Management study integrates high-speed rail as a schedule-based resource and uses Benders decomposition to generate air, rail and mixed itineraries. Include it as a comparator where multimodal alternatives are available; do not claim reproduction without equivalent data. https://doi.org/10.1016/j.jairtraman.2026.103019
- **Industry operations:** EUROCONTROL's Network Manager describes shared airspace/capacity data and pre-tactical/tactical coordination across airlines, airports, ANSPs and meteorological services. https://www.eurocontrol.int/network-operations

## Implemented Rust surface

File: `crates/domains/symthaea-operations-research/src/aviation_network_externalities.rs`

- `compare_network_outcomes` requires paired scenario and objective IDs, verified feasibility flags, unique actor IDs, identical actor sets and stable domain mappings.
- Actor costs use a versioned objective. Delay is explicitly `non_cancelled_delay_minutes`; canceled flights are counted separately to avoid double counting.
- Emissions are reported in kg CO2e as a separate physical ledger unless an explicit sourced conversion is added to a versioned objective.
- Report includes system net benefit, gross actor benefit/burden, spillover ratio, worst actor regression, actor deltas, delay/cancellation differences and emissions differences.
- `select_network_candidate` filters candidates using a predeclared maximum per-actor regression and minimum net-system-benefit requirement. It deterministically selects the lowest-cost eligible outcome, with tie-break by lower worst-actor regression and original index. Invalid candidates are recorded as ineligible. An invalid baseline fails closed. If no candidate meets the constraints, the selected index is `None`; callers must retain only a separately verified incumbent or explicitly escalate.
- The module scores supplied outcomes; it does not create a plan or independently verify physical feasibility. The feasibility label must originate from the chosen verifier and should later be checked by an independently maintained implementation.

## Required benchmark

Freeze actor IDs/domains, objective/cost mappings, event horizon, counterfactual and actor-level attribution before evaluation. Reuse identical frozen scenarios and action spaces across:

1. initiating-airline local objective;
2. network-total cost objective;
3. network-total objective with predeclared per-actor harm ceiling;
4. trigger-based network rules inspired by published operational coordination (reimplemented independently);
5. integrated MILP/CP-SAT or decomposition baseline when the same constraints/data are available;
6. do-nothing / incumbent plan;
7. hindsight reference, labeled non-deployable.

Report mean/median/95th/99th-percentile and CVaR-style cost; component ledger; system benefit; actor benefit and burden; worst actor regression; affected actors by domain; delay, cancellations, passenger outcomes and CO2e separately; hard-constraint failures; computational latency; and uncertainty intervals clustered at the disruption-episode level.

### Mandatory scenarios

- weather/capacity shocks with multiple flights sharing airspace constraints;
- gate/airport constraints and cascading delays;
- maintenance-grounding or qualified-part shortage;
- alternate-airport/rerouting choices;
- conflicting stakeholder objectives;
- missing/stale/contradictory actor evidence;
- compound disruptions, distribution shift, and cases where do-nothing is best.

### Proposed gates (targets, not results)

- 100% agreement with hand-computable outcomes and an independently written evaluator for small fixtures;
- zero silently omitted actors, mismatched objective versions, or scored rejected/indeterminate plans;
- zero selector choices that violate the preregistered per-actor harm ceiling;
- no unaccounted overlap between cancellation and non-cancelled-delay metrics;
- demonstrate a positive paired decision-value improvement over the strongest feasible baseline, with a 95% interval excluding no improvement, and no unpredeclared tail or subgroup regression;
- publish all exclusions, invalid candidate reasons, objective weights, data digests and test status.

A passing test demonstrates only the frozen model and fixture. It does not establish causal effects, airline savings, operational safety, airworthiness, maintenance release, dispatch authority or regulatory approval.

## Evidence status

The local Python fixture and synthetic stress checks for the preceding AVNX artifact passed 10 tests. This repository commit adds a Rust ledger and selector with 9 unit tests defined. The current environment still has no local `cargo` or `rustc`, so the Rust tests have not run. Workflow records for the earlier draft PR were skipped, not passed.


## Pareto-frontier extension (28 October 2026)

The Rust module now includes `network_pareto_frontier`, which keeps system cost, worst-actor regression, non-cancelled delay, cancellations, and kg CO2e as separate objectives. It returns non-dominated candidate indices and assessments for every candidate; invalid ledgers remain visible with explicit exclusion errors. This avoids inventing a weighted sum to choose among legitimate trade-offs. The frontier is descriptive, not a fairness decision by itself: a downstream policy should apply predeclared hard constraints or a stakeholder-approved choice rule.

Two additional unit tests cover retaining a genuine cost-versus-harm trade-off while removing a dominated candidate, and excluding malformed candidates without hiding the reason. The Rust module now defines 11 tests; no Rust compile/test claim is made because a Rust toolchain is still unavailable in the current execution environment.
