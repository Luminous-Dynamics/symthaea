# SAM Decision-Sufficient Action-Space Coverage v0.1

**Status:** analytic research protocol / golden fixtures  
**Issue:** [SAM-ACTION-SPACE-001 #10338](https://github.com/Luminous-Dynamics/symthaea/issues/10338)  
**Composes:** #10237, #10252, #10257, #10318, #10083, #10012, #10043; Symtropy #3387 and #3502  
**Claim ceiling:** bounded regret over the declared action domain and named model/reference context only.

## 1. Purpose

A finite counterfactual evaluator can correctly compare every action supplied to it and still miss the genuinely better action because the supplied candidates do not cover the maneuver space.

`best supplied action != best action in declared action domain`

This protocol distinguishes exhaustive enumeration from empirically sampled actions and from a decision-sufficient finite candidate set whose omitted regions are bounded by independently checkable enclosures.

The objective is not to enumerate every control sequence. It is to prove when the unenumerated parts of a declared domain cannot change the decision beyond a stated tolerance—or explicitly return unresolved.

## 2. V0.1 scope

The first qualification corpus is analytic:
- one discrete maneuver family at a time, with separate families for hybrid modes;
- one continuous parameter (x) in a closed bounded interval;
- one scalar maximization objective;
- one or more hard feasibility constraints;
- exact analytic functions and derivative/interval bounds;
- deterministic partition and refinement.

The later vehicle adapter may parameterize maneuver primitives such as a braking profile, yield-and-resume profile or bounded steering profile. Symthaea #6247 remains the single vehicle-dynamics owner. This protocol adds no vehicle dynamics, physics engine, simulator, or control path.

## 3. Domain identity and coverage

A domain is not merely a list of candidate points. It is a versioned object:

`A = M × X`

where (M) is an explicitly enumerated finite set of discrete maneuver modes and (X) is a bounded continuous parameter domain for each mode.

The protocol must bind:
- domain schema/version and exact mode IDs;
- every continuous parameter name, unit, frame, closed/open bounds and norm;
- admissible parameter combinations and cross-constraints;
- mode-specific discontinuities and switching surfaces;
- action generator / partition algorithm and configuration;
- scenario, initial state, environment, actor policy and decision context;
- dynamics/reference/utility/evaluator identities and qualification records;
- horizon, timestep, clock and exogenous stream;
- each evaluated representative action, exact command digest and execution receipt;
- partition cells, representative points, radii, boundaries and proof method;
- objective/constraint enclosures and proof provenance;
- refinement, pruning, unresolved cells, numerical tolerances and resource budget.

A point sample without a declared domain and a covering partition is `EmpiricalCoverageOnly`, never exhaustive or complete.

### 3.1 Partition invariants

For each discrete mode, partition cells must:
1. be contained in the declared admissible parameter domain;
2. cover the complete declared domain, including every boundary;
3. have disjoint interiors (shared boundaries are permitted under a canonical ownership rule);
4. identify their representative point and maximum radius in the declared norm;
5. reject duplicated, overlapping-interior, out-of-domain or missing cells;
6. remain stable under canonical input ordering.

For a box partition, every axis interval and endpoint inclusion is explicit. For non-box partitions, a separate verifiable coverage representation is required. Unknown gaps remain unresolved and block completeness.

## 4. Conservative enclosures

For each cell (C), objective (f), and hard constraint (g_k(x) le 0), retain source-supported enclosures:

`f(C) ⊆ [f_lower(C), f_upper(C)]`

`g_k(C) ⊆ [g_lower(C), g_upper(C)]`

For a maximization objective, a representative point (c_C), cell radius (r_C), and valid Lipschitz constant (L_C):

`f_upper(C) = f(c_C) + L_C r_C`

`f_lower(C) = f(c_C) - L_C r_C`

The bound is valid only when the named objective is (L_C)-Lipschitz over the entire cell under the declared norm. A fitted local slope, training residual, ensemble spread, empirical smoothness or confidence score does not establish this property.

For a hard constraint (g(x) le 0):

- `DefinitelyFeasible` only when (g_upper(C) le 0);
- `DefinitelyInfeasible` only when (g_lower(C) > 0);
- `FeasibilityUnresolved` otherwise.

A feasibility certificate is not a safety certificate: it is conditional on the declared constraint, model, reference and context.

### 4.1 Accepted V0 bound methods

V0.1 should admit only methods for which the verifier can independently recompute conservative enclosures:

- exact analytic extrema for a frozen function class;
- interval arithmetic using a named, versioned arithmetic profile and outward rounding;
- a supplied analytic derivative/Lipschitz proof object for a named function and domain, checked by the declared verifier.

A user-provided (L) without a verifiable proof/source is not accepted as a certificate. Unsupported methods yield `OutOfScope` or `Unresolved`.

Every bound binds function identity, parameter domain, norm, units, precision, rounding direction, overflow policy, and proof/verifier identity. A bound from one model/reference does not transfer to another by name or similarity.

### 4.2 Discontinuities and hybrid modes

A smoothness bound cannot cross:
- a discrete maneuver-mode boundary;
- a contact/collision event boundary;
- an actor-policy state transition threshold;
- a saturation, clamp or actuator mode switch;
- a rule/feasibility threshold;
- a sensor visibility/occlusion transition;
- an unknown or unqualified dynamics boundary,

unless the proof method explicitly covers that discontinuity.

The default is to split the domain at the boundary or mark the affected cell unresolved. No interpolation through discontinuities and no smooth bound inferred from neighboring samples.

## 5. Decision-sufficiency and regret

For one frozen maximization utility, a complete admissible partition and a selected feasible action (a^*) with evaluated value (U(a^*)), compute:

`U_domain_upper = max_C U_upper(C)`

`regret_upper(a*) = max(0, U_domain_upper - U(a*))`

If the selected-action value itself is interval-valued, use its lower bound in the conservative regret upper bound:

`regret_upper(a*) = max(0, U_domain_upper - U_lower(a*))`

The result is conditional on the action domain, objective, reference/model profile and all bound assumptions.

An unselected cell can be safely pruned from further refinement only when:
- it is definitely infeasible under every required hard constraint; or
- its objective upper bound cannot improve the selected action beyond the frozen regret tolerance (single-objective V0); or
- a separately specified interval-Pareto rule proves it cannot improve the frontier on any objective (later multi-objective work).

A cell with unresolved feasibility cannot be treated as infeasible. A cell whose objective upper bound exceeds the selected value plus tolerance cannot be pruned.

### 5.1 Typed result states

- `FiniteDomainExhaustive`: every action in a genuinely finite declared domain has been evaluated.
- `BoundedDecisionSufficient`: all omitted cells have valid bounds proving no improvement beyond the frozen tolerance.
- `BoundedRegret`: a conservative upper bound on regret over the entire declared domain is reported.
- `EmpiricalCoverageOnly`: samples are retained, but the omitted domain is not bounded.
- `Unresolved`: at least one cell could contain a better feasible action or change a hard-constraint disposition.
- `Incomplete`: required cells, domain boundaries, bounds, receipts or context identities are missing or invalid.
- `OutOfScope`: the domain/function/reference/bound method is outside the qualified context.

A non-dominated finite sample is not automatically a Pareto-optimal set. No multi-objective scalarization is permitted by this v0.1 scalar fixture.

## 6. Adaptive refinement

After a conservative verifier exists, refinement may use best-first branch-and-bound:

1. initialize a covering partition for each discrete mode;
2. evaluate deterministic representatives under the exact same frozen context;
3. obtain independently verifiable objective and constraint enclosures;
4. prune only cells satisfying the accepted rules above;
5. refine cells whose bound can change feasibility or improve the regret upper bound;
6. stop at the declared tolerance or resource/deadline budget;
7. preserve every evaluated, pruned, refined and unresolved cell.

Budget exhaustion is an incomplete/unresolved result, not a pass. A heuristic optimizer may propose representatives, but its failure to find a better action is not a completeness certificate.

Compare adaptive refinement against uniform grids, seeded random sampling, low-discrepancy sampling and CMA-ES under equal execution budgets.

## 7. Golden fixtures

The companion JSON at `fixtures/SAM_ACTION_SPACE_BOUNDS_V0_1.json` includes:
- a coarse grid whose best sample is not the analytic optimum;
- a valid Lipschitz enclosure that safely prunes an omitted region;
- an understated Lipschitz constant that would falsely prune and must be rejected;
- a feasibility boundary crossing a cell;
- a hybrid discontinuity that forbids cross-mode smoothness;
- an uncovered partition gap;
- an overlapping cell;
- a budget-exhausted unresolved cell;
- a context/dynamics mismatch;
- input-order invariance.

These fixtures are synthetic arithmetic tests, not vehicle dynamics measurements.

## 8. Benchmark ladder and comparison

| Stage | Workload | Main measure |
|---|---|---|
| Analytic | 1D/2D objectives with known extrema | Missed optimum, enclosure correctness, regret bound |
| Hybrid | Mode switches and feasibility thresholds | False pruning / false feasibility |
| SAM crossing | #10291 and #10257 analytic counterfactuals | Decision changes hidden by coarse action grids |
| Execution | Symtropy #3387 / #3502 | Exact action/receipt pairing and replay |
| Planning | CommonRoad / nuPlan tasks with frozen maneuver parameterization | Budget–regret frontier |
| Closed loop | CARLA Leaderboard 2.1 / Bench2Drive v0.0.4 | Task outcomes under matched protocols |
| OOD | Fail2Drive paired routes | OOD decision degradation and action-domain coverage |

Always report:
- false completeness and false pruning;
- missed better feasible action rate on known-answer domains;
- regret upper bound and its tightness against the analytic optimum;
- domain fraction certified, pruned, refined and unresolved;
- execution count and runtime/compute per decision;
- decision changes after refinement;
- constraint misclassification;
- exact action-domain and evaluator identities.

A reduction in regret bound is not useful if the verifier admits a false prune. Correctness is a hard gate; efficiency is optimized only after correctness.

## 9. Falsification / acceptance

The verifier must independently recompute:
- exact domain and partition coverage;
- cell radii and boundary ownership;
- accepted objective/constraint enclosures;
- feasibility classification;
- every pruning decision;
- domain-wide upper utility and regret bound;
- exact source/action/context/receipt identities.

Negative controls include missing cells, overlapping cells, understated Lipschitz constants, wrong norm/units, threshold crossings, hybrid switches, NaN/Inf, inward rounding, stale receipts, changed horizons, and budget exhaustion.

A future implementation is accepted only when all analytic fixtures pass, no false completeness or false pruning occurs, unsupported bound methods fail closed, and exact-head build/test/format evidence is retained. Skipped or queued workflows are not passes.

## 10. Claim ceiling and prospective targets

Passing establishes only exhaustive or decision-sufficient coverage of the named bounded action domain under the declared function bounds, constraints, objective, context and reference/model assumptions. It does not establish global optimality outside that domain, true-world model validity, physical safety, collision-risk reduction, roadworthiness, certification or SAE Level 4/5 capability.

Prospective targets (not results):
- zero false exhaustive/complete claims in the falsification suite;
- zero known-answer better feasible actions hidden by a purported certificate;
- 100% of declared domain covered by verified cells or explicitly unresolved gaps;
- at matched execution budget, 30% tighter worst-case regret bound than a uniform-grid baseline on the analytic corpus, with zero false pruning;
- zero hard-constraint false-feasible classifications;
- 100% independently recomputable bounds.
