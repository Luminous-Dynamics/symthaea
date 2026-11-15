# Atelier FactoryBench — Counterfactual Evaluation Protocol v0.1.0

**Status:** proposed benchmark protocol  
**Scope:** reproducible comparison, causal attribution, out-of-distribution generalization and evidence progression  
**Authority:** research/benchmark methodology only; not a physical-factory or safety qualification

## 1. Why this exists

A factory optimizer can appear better than a baseline because it received easier orders, simpler garment topology, more skilled staffing, a different shift calendar, a more permissive inspection rule, or a different definition of productive time.

A paper's reported improvement is not a transferable factory improvement until the task, baseline, inputs, measurement boundary and uncertainty are matched.

FactoryBench therefore evaluates not only *which result is better*, but *what caused the difference, whether the difference survives held-out conditions, and whether the result remains feasible under disruptions*.

The protocol separates:
- **prediction:** what a model says will happen;
- **optimization:** what plan an algorithm proposes;
- **verification:** whether the plan satisfies hard constraints;
- **execution:** what physically happened;
- **acceptance:** what passed independent product inspection;
- **causal effect:** the measured difference attributable to the intervention under a matched comparison.

A simulation result cannot be upgraded into physical performance merely because its internal checks pass.

## 2. Frozen experiment identity

Each benchmark run must publish a canonical run manifest that includes:

- FactoryBench release and evaluator version;
- exact product, pattern, BOM, operation-graph and standard-work revisions;
- garment family, style, size and color mix;
- material classes, lots, conditioning and cut-plan revision;
- order quantities, lot-streaming and delivery promises;
- machine asset/configuration/capability references;
- process-cell and tooling revisions;
- operator skill requirements, available qualified headcount, shift calendar and paid-time policy;
- inspection plan, acceptance criteria, defect taxonomy and rework policy;
- facility/layout revision, geometry frame, clearances and utility assumptions;
- algorithm/solver version, parameters, hardware and deterministic seed;
- scenario-event sequence and random seed;
- metric-dictionary version and exact measurement boundaries;
- source references, evidence classes, exclusions, uncertainty and input digest.

The manifest is immutable. Any changed input creates a new run identity.

## 3. Evidence lanes

Never combine these lanes into one result table without an explicit evidence-class column.

### Lane S — synthetic deterministic

Use frozen, human-readable inputs and exact expected outputs. Appropriate for graph correctness, constraint logic, determinism, reason codes, arithmetic and negative controls.

Allowed claims: structural and algorithmic behavior on the synthetic fixture.

Not allowed: real capacity, cost, worker productivity, quality, energy, safety or commercial claims.

### Lane M — simulated stochastic

Use calibrated or explicitly assumed distributions, paired random seeds and documented model validation. Report distributions and uncertainty.

Allowed claims: conditional simulated behavior under the declared model.

Not allowed: physical improvement claims unless the simulator has been validated for that quantity and context.

### Lane P — physical pilot

Use surveyed facilities, identified machines, approved standard work, observed staffing, calibrated meters, defined inspection plans and timestamped physical events.

Allowed claims: measured behavior of the named pilot and sampled operating envelope.

Not allowed: universal or production-network claims from a small pilot.

### Lane I — independently measured production

Use a preregistered measurement plan, independent evidence review, repeated production periods, material/quality traceability, and a clearly stated external comparability boundary.

This is the only lane eligible for strong production-performance and industry-leadership claims.

## 4. Baselines and causal attribution

Every optimization experiment must include:

1. **Current/manual baseline:** the existing documented process, not an intentionally weakened baseline.
2. **Conventional engineering baseline:** Systematic Layout Planning, industrial engineering and current standard scheduling rules.
3. **Transparent heuristic baseline:** FIFO, EDD, earliest-finish, setup-aware assignment and other simple interpretable methods.
4. **Exact small-instance oracle:** MILP, exhaustive enumeration or another independently checked exact method where tractable.
5. **Advanced optimizer:** GA, PSO, multiobjective evolutionary search or other candidate method.
6. **Independent feasibility verifier:** rechecks hard constraints without trusting the optimizer's status or score.

The workload, product mix, worker requirements, resource availability, acceptance criteria, utility boundary and delivery promise must be held constant.

### Paired scenarios

For stochastic experiments, compare candidate and baseline on the same exogenous scenario seeds: arrivals, cancellations, absences, machine failures, material holds, power events and order-mix changes. This common-random-number design reduces noise from different scenario draws.

Publish paired per-scenario deltas as well as aggregate distributions. Do not compare the candidate's best seed to the baseline's average.

### Ablation and interaction

Change one factor at a time for initial attribution, then use a preregistered factorial design for interactions.

Examples:
- new layout, same scheduler;
- new scheduler, same layout;
- qualified capability matching enabled/disabled;
- reconfiguration procedure enabled/disabled;
- additional inspection or buffer policy;
- partner route enabled/disabled.

This reveals whether an improvement came from layout, better work content, an extra machine, different staffing, removed inspection, altered batch sizes, or actual algorithmic improvement.

## 5. Held-out generalization

Do not randomly split near-identical pieces or operation sequences across train and test.

Use family/site/material-aware splits:

- **Garment holdout:** reserve an entire construction family or topology for testing.
- **Fabric holdout:** reserve fabric types, weights, stretch, friction and drape ranges.
- **Pattern holdout:** reserve unseen pattern and seam-topology revisions.
- **Facility holdout:** test at a partner site not used during tuning.
- **Mix holdout:** test product-mix and lot-size distributions not used for optimization.
- **Disruption holdout:** reserve combined and compound failures for final evaluation.
- **Temporal holdout:** evaluate on later orders and later qualification states.

Report both interpolation performance and held-out generalization. The test set must not be used to tune objective weights, constraints, random seeds or stopping conditions.

## 6. Feasibility before score

Hard constraints are non-compensatory.

A solution with a missing operation, invalid material lot, expired process configuration, unqualified operator, unavailable equipment, missing inspection, unsafe/blocked egress or inaccessible maintenance path is not a low-scoring feasible solution. It is Blocked or Indeterminate.

Required sequence:

1. Optimizer produces a candidate plan.
2. Independent verifier checks identities, precedence, resource capability, skill/headcount, material, configuration validity, utility, layout and inspection.
3. Only independently verified feasible plans enter performance comparison.
4. Physical acceptance remains a distinct downstream gate.

Publish false-eligible rate and false-blocked rate separately. A zero false-eligible result on a small corpus is not a universal correctness proof; publish the corpus size and uncertainty.

## 7. Multiobjective and robust comparison

Do not publish a single weighted "factory efficiency" score.

Publish Pareto outcomes for:
- fully loaded cost per first-pass accepted product;
- order lead time P50/P95;
- accepted output per paid labour-hour and constrained machine-hour;
- first-pass yield, defects, rework and scrap;
- WIP, queue and flow distance;
- setup time and time to first conforming unit;
- energy/material intensity within measured boundaries;
- disruption recovery time and backlog;
- worker workload and task-specific ergonomic exposure.

Safety, statutory conditions, egress, qualification, inspection and worker stop/report rights are feasibility gates—not objectives to trade away.

For resilience and tail risk, report P50/P95 and predeclared CVaR or worst-case scenario metrics. Publish feasible Pareto sets and hypervolume only with a frozen reference point and objective normalization; always retain raw metric values.

## 8. Statistical reporting

For stochastic methods:
- at least 30 seeds for initial distributional comparison;
- publish every seed's outcome, not winner-only results;
- report median, P95, interquartile range and confidence intervals;
- use paired deltas where scenario seeds are matched;
- state the interval construction method and assumptions;
- report infeasible runs as failures, not omit them from the distribution;
- report runtime, hardware, memory, solver settings and stopping criteria.

For physical pilots:
- preregister the primary outcome and measurement boundary;
- collect repeated shifts/orders and enough observations to estimate variability;
- state calibration and sampling methods;
- use independent acceptance inspection;
- record missing/censored orders and rework rather than silently dropping them;
- use a control or matched-period design where randomization is not operationally possible;
- report confounders and limits to causal interpretation.

No universal sample size guarantees adequate power. Use a pilot to estimate variance, then size the confirmatory experiment for the declared minimum practically important effect.

## 9. Industry comparison admission test

A published or vendor result is admitted only with a comparability record:

| Dimension | Match required |
|---|---|
| Product | garment family, construction, size and complexity |
| Materials | fabric/trim properties, lot and conditioning |
| Work content | seam/operation graph and approved standard-time method |
| Workload | order size, mix, lot streaming and due-date policy |
| Resources | machine capabilities, count, tooling and uptime |
| Workforce | paid hours, skill mix, headcount, training and shift calendar |
| Quality | inspection gate, acceptance, FPY, defect taxonomy and rework |
| Time | planned, attended, run and productive-time definitions |
| Boundary | factory/partner, transport, queue, utility and cost scope |
| Evidence | measured, independent, paper-reported, simulated or vendor-claimed |

If key dimensions do not match, the comparator is descriptive context—not a head-to-head result.

Vendor-announced targets must be labeled as targets. Paper-reported case studies remain tied to their original problem. Do not normalize unlike products into a generic "garments/day" metric without a validated work-content conversion.

## 10. Proposed Atelier advancement gates

These are research targets, not standards, achieved results or promises.

### Gate A — correctness
- Zero false-eligible plans in the frozen adversarial corpus.
- Every operation assigned exactly once or the plan is rejected.
- Every required input and output lot has a disposition path.
- Exact small-instance answers agree with an independent oracle.
- Deterministic output for frozen inputs.

### Gate B — comparative optimization
- Candidate beats manual/SLP and transparent heuristic baselines on preregistered, matched cases.
- Report exact-oracle optimality gaps for small cases.
- Demonstrate a Pareto improvement, not merely a lower weighted score.
- No quality, worker, safety or traceability regression.

### Gate C — generalization
- Improvements survive held-out garment families, fabric types, facilities and order mixes.
- No material drop in feasibility or FPY under held-out conditions.
- Compound disruptions do not reveal hidden common-cause dependencies.

### Gate D — physical pilot
- At least 20% lower median time to first conforming unit than a frozen feasible baseline is a proposed target for a demonstrated reconfiguration task.
- At least 20% lower measured material/person travel is a proposed target only where the physical layout intervention causes it.
- At least 20% faster recovery is a proposed target on named disruptions.
- Improvement confidence intervals exclude zero for preregistered primary outcomes, with the minimum practically important effect also reported.
- No deterioration in first-pass acceptance, worker safeguards or traceability.

### Gate I — independently measured production
- Repeat the result across multiple periods and, where possible, multiple sites/partners.
- Publish full input/output traceability and measurement uncertainty.
- Publish raw anonymized per-scenario/per-period result records.
- Independent reviewer reproduces the published calculation and checks the claim boundary.

## 11. What would justify an SOTA claim?

A defensible claim must name:
- the specific task and product scope;
- the exact competitor/baseline and version;
- matched input and measurement boundaries;
- held-out evaluation design;
- feasibility and false-eligible rates;
- statistical method and uncertainty;
- raw result artifact digest;
- evidence class and physical scope;
- independent verification status.

A claim such as "Atelier is better than SOTA" is not an admissible result. A claim such as "on held-out woven overshirt and trouser cases, with identical staffing, equipment, inspection and order promises, method X reduced time-to-first-conforming-unit by Y with interval Z and no measured FPY or workload regression" is testable.

## 12. Claim ceiling

This protocol defines how to compare methods. It does not show that any Atelier optimizer is currently faster, cheaper, safer, more sustainable or more productive. The current corpus remains synthetic until actual source evidence and physical observations are supplied.