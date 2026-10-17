# MARINE-CAT-002 — bounded design-space generation and multi-fidelity search

Status: deterministic grid generator and receipt contract authored; no Rust build, CFD model, resistance prediction, seakeeping evaluation or ship-design optimization result is claimed.

Research issue: [MARINE-CAT-002 #7737](https://github.com/Luminous-Dynamics/symthaea/issues/7737)  
Implementation PR: [PR #7722](https://github.com/Luminous-Dynamics/symthaea/pull/7722)

## Paradigm shift

The optimisation target is not a hull-spacing ratio or calm-water drag value in isolation. It is a set of designs that remain attractive across a declared mission envelope while uncertainty, evidence gaps, physical constraints, buildability and operating cost are visible. We therefore separate three tasks:

1. **Candidate generation** — deterministic geometry/operating points, unit consistency, explicit bounds and bounded resource usage.
2. **Physical evaluation** — separately versioned low-fidelity estimates, compatible CFD, model-basin tests, electrical/energy models and later full-scale observations. Each method declares what it can and cannot answer.
3. **Evidence-qualified decision** — compare only comparable cases, keep measurement/numerical/model uncertainty separate, and retain `indeterminate`/abstain when hard-gate evidence is missing.

The first executable generator cross-products five caller-provided axes: hull spacing/LWL, demihull beam/LWL, draft/LWL, wet-deck clearance/draft and Froude number. It checks cardinality with overflow-safe arithmetic and rejects a candidate count beyond the caller's cap before allocating. A receipt records the input spec, ordered candidate list, grid version, optional exact source revision, SHA-256 digest of the candidate list, and a digest over the complete receipt body.

These generated points are a geometric parameter grid. They are not a hull optimisation, resistance estimate, safe operating envelope or recommendation for hull ratios.

## Why multi-fidelity is essential

An expensive high-fidelity run should not be spent on every possible grid point. Start wide, then allocate better solvers/experiments to points that can alter the decision:

- **Level 0 — contract/arithmetic:** dimensions, ratios, unit conversions, candidate count, simple geometry consistency, route/draft/beam checks when input evidence is supplied.
- **Level 1 — low-fidelity screening:** clearly versioned analytical estimates and simplified powering/energy budgets. Report ranges and applicability; no hidden physical guarantee.
- **Level 2 — calibrated hydrodynamic methods:** CFD, panel/seakeeping codes, propulsor models or other tools only after verification, uncertainty analysis and validation on compatible reference cases.
- **Level 3 — experimental confirmation:** towing-tank/model tests for critical resistance, trim/sinkage, wave response, wet-deck impacts and structural load mechanisms where the chosen model needs physical validation.
- **Level 4 — full-scale evidence:** instrumented vessel trials across a predeclared mission envelope. Only this layer supports claims about the actual vessel's measured operation within that envelope.

Do not call a surrogate 'high fidelity' because it has many parameters. Fidelity is the demonstrated error and applicability for a defined quantity and regime.

## The evaluation loop

1. Freeze the mission, candidate axes, baseline ships, operating-point corpus, objective definitions and hard constraints.
2. Generate and hash the grid; keep a disjoint held-out corpus and record the exact seed/sampling procedure.
3. Evaluate cheap quantities for all admissible candidates. Reject only explicit contradictions or failed declared hard gates, not unknown physics disguised as failure.
4. Pick the next high-fidelity evaluation to reduce uncertainty relevant to the Pareto frontier or a hard constraint. Initially compare deterministic, fixed-grid and random selection baselines; report paired improvements on held-out scenarios.
5. Update candidate values with units, model/evaluator revision, condition, uncertainty and evidence class. Unknown values remain unknown.
6. Report robust dominance only where evidence and conditions are comparable. If uncertainties overlap or methods are out of domain, retain an unresolved trade-off.
7. Audit the frozen receipt and negative controls before claiming improvement.

Bayesian optimisation, expected information gain or active learning can be a future policy, but only when outcome/noise models and calibration support those claims. In the current synthetic selector, interval shrinkage is explicitly not probability or information gain.

## Research basis and scope limits

The ITTC's CFD-validation database procedure is an index of existing reference cases and their measured quantities, not proof that all listed data are downloadable or reusable. It explicitly includes different hull types and different levels of available evidence; case selection must be driven by the quantity and model being validated. Most conventional ship resistance cases cannot be treated as validation for catamaran hull interference or hydrofoil lift merely because they are maritime CFD cases. See [ITTC Benchmark Database for CFD Validation for Resistance and Propulsion](https://www.ittc.info/media/9771/75-03-02-02.pdf) and [the current ITTC ship resistance/propulsion index](https://ittc.info/benchmark-repository/ship-resistance-and-propulsion/).

Experiments on catamaran design have varied hull form, separation and longitudinal centre of gravity and recorded not only calm-water resistance but also sinkage/trim, added resistance, motion, deck acceleration and cross-member loads. This shows why a single calm-water scalar is insufficient: [TRID: Experimental Investigation of Resistance and Seakeeping Characteristics of a Catamaran Design](https://trid.trb.org/View/439786).

A July 2026 *Ocean Engineering* paper compares zero-emission fast-ferry feasibility for catamarans and hydrofoil vessels, including mass and energy-use estimation. That motivates a route-by-route energy feasibility gate rather than assuming batteries make any speed/range combination feasible: [Godø et al. (2026), DOI 10.1016/j.oceaneng.2026.126414](https://doi.org/10.1016/j.oceaneng.2026.126414). Its conclusions should be interpreted under its route and model assumptions; they do not validate this code.

## Proposed benchmarks — goals, not results

- **Determinism:** exact ordered candidate axes and source revision yield byte-identical candidate list digest and receipt digest.
- **Resource safety:** reject empty/non-finite/non-positive axes, negative Froude numbers, arithmetic overflow and over-cap grid size before candidate allocation.
- **Input binding:** receipt binds the full axis spec, candidate manifest, design-space generation version and source revision. Digest verification detects accidental mutation; it is not a signature or proof of physical truth.
- **Breadth:** once bounds are reviewed, generate at least 10,000 low-fidelity design points per archetype and verify axis coverage; do not claim the number alone establishes search quality.
- **Fidelity calibration:** report error and uncertainty per named reference condition and metric; no aggregate pass hides failed conditions or domain gaps.
- **Seakeeping:** publish response distributions and interval coverage by speed/Froude, wave heading/encounter, loading and hull spacing; calibrate both coverage and sharpness.
- **Decision safety:** zero false-safe decisions in prespecified boundary/fault controls for slamming, cross-deck loading, stability, foil failure, route limits and degraded energy/propulsion.
- **Policy efficiency:** report paired improvement and uncertainty versus fixed-grid and random baselines on frozen held-out cases, along with high-fidelity compute/test cost.
- **Pareto honesty:** show non-dominated sets on compatible conditions, keep unresolved hard constraints explicit, and never let cost, speed or emissions compensate for a safety failure.

## Claim ceiling

A schema pass, candidate-grid receipt or arithmetic pass establishes only deterministic generation and the declared geometric consistency checks. It does not establish lower resistance, comfortable motions, acceptable slamming, structural adequacy, energy feasibility, seaworthiness, class/flag approval or passenger-carrying authority.