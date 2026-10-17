# Chemical Process Engineering: End-to-End Qualification Benchmark

**Status:** Proposed benchmark contract and research campaign; not evidence that these capabilities have already been implemented or qualified.  
**Date:** 2026-10-17  
**Repository baseline:** main@5e83cea4986ba3845aa50e0d5dd03493c5adeddf  
**Scope:** Architecture, evidence semantics, benchmark design, and qualification gates. This document defines no hazardous synthesis procedure, detailed operating recipe, or plant-control authority.

## Executive thesis

Symthaea should evolve from a collection of chemistry/physics reasoning modules into an **evidence-carrying chemical-process engineering workflow**. The unit of achievement is not a plausible reaction, a converged solver, or a visually complete flowsheet. It is a design whose assumptions, models, material and energy transfers, constraints, provenance, uncertainty, equipment dependencies, and unresolved obligations are explicit and independently reviewable.

The central chain is:

    objective
    → proposed chemistry/material transformations
    → model-qualified unit operations
    → connected process graph
    → solved flowsheet
    → independently checked balances and constraints
    → robust/Pareto design candidates
    → equipment/material/utility closure
    → engineering review and qualification
    → construction/commissioning evidence
    → measured operational evidence

Every arrow is an evidence transition—not an assumption. A result at one stage must not implicitly promote the next.

## Why this is the next high-leverage gap

The current public roadmap already contains the relevant owners. Do not create a competing chemistry engine, universal quantity system, flowsheet engine, manufacturing ontology, inventory ledger, or safety-authority framework.

Compose the existing work:

- [CHEM-MFG-000 #5970](https://github.com/Luminous-Dynamics/symthaea/issues/5970): industrial chemical, process-gas, fertilizer-intermediate and consumable manufacturing semantics.
- [First-principles chemical-engineering audit #7317](https://github.com/Luminous-Dynamics/symthaea/pull/7317): baseline reproduction through chemistry, thermodynamics, kinetics, transport and flowsheet optimization.
- [CHEM-QC-001 #7321](https://github.com/Luminous-Dynamics/symthaea/issues/7321): reaction candidates and qualified quantum-chemistry evidence.
- [THERM-VALID-001 #7318](https://github.com/Luminous-Dynamics/symthaea/issues/7318): checked thermofluid numeric domains.
- [CFD-QUAL-001 #7316](https://github.com/Luminous-Dynamics/symthaea/issues/7316): numerical qualification for the native continuum solver.
- [SEP-001 #5199](https://github.com/Luminous-Dynamics/symthaea/issues/5199): selective-separation and recovery evidence.
- [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686): process identity, capability, model, plan, results and qualification semantics.
- [PIE #1604](https://github.com/Luminous-Dynamics/symthaea/issues/1604): resource-to-product, utility, waste, maintenance and industrial-closure accounting.
- [Physics quantity type system #6868](https://github.com/Luminous-Dynamics/symthaea/issues/6868), [V&V/UQ #3697](https://github.com/Luminous-Dynamics/symthaea/issues/3697), [engineering obligation discharge #5124](https://github.com/Luminous-Dynamics/symthaea/issues/5124), and [model maturity envelopes #6871](https://github.com/Luminous-Dynamics/symthaea/issues/6871): shared validation, authority and evidence contracts.

The novel contribution proposed here is an **end-to-end qualification and benchmark layer** that composes these owners and exposes where a chain is broken. It must not become a second implementation of their responsibilities.

## Paradigm shifts

### 1. Chemical-process design as a compiler with proof-carrying artifacts

Treat a process proposal like a typed program:

1. Parse the design intent and constraints.
2. Construct a canonical, versioned process graph.
3. Type-check chemical species, stream state, physical quantities, phases, equipment interfaces and model validity domains.
4. Generate solver inputs only when required fields and provenance are present.
5. Evaluate the graph through selected qualified models.
6. Independently check invariants and emit a machine-readable evidence receipt.
7. Refuse to promote the design if a required proof obligation is missing, stale, contradictory or out of scope.

A receipt should identify exact source/design/model/input digests, versions, assumptions, units, applicable conditions, solver tolerances, return status, residuals, uncertainty, known limitations and checker version. A receipt proves only its named proposition. For example, a valid mass balance does not prove reaction selectivity, safe operation, product purity, economic viability or manufacturing capability.

### 2. A model firewall across scales

Build an explicit evidence lattice, not one global “validated” boolean. At minimum distinguish:

- represented / syntactically valid;
- data sourced and current;
- input accepted;
- model run to completion;
- numerically converged;
- independently reproduced;
- validated against relevant experimental observations;
- uncertainty characterized for the declared domain;
- fit for the exact engineering decision;
- reviewed/qualified by the required authority.

Keep model families and transitions distinct: electronic structure → thermochemistry → kinetics → reaction network → transport → unit operation → flowsheet → equipment → plant. Transfer between levels requires an explicit bridge, assumptions and uncertainty propagation. A quantum-chemistry result or HDC analogy must never be treated as a process-feasibility certificate by itself.

### 3. Inverse design of the complete process

Optimize chemistry, separation, recycle, heat integration, utilities, product specifications, materials of construction and equipment dependencies together where valid models exist. An isolated reaction with high conversion is not automatically the best industrial route if separation is costly, utilities are unavailable, waste cannot be handled, or components cannot be qualified.

Return a **Pareto set** with trade-offs and sensitivity analysis rather than collapsing safety, cost, yield, emissions, water use, resilience and maintainability into one uninspectable score. Safety and mandatory engineering constraints are feasibility gates, not soft penalties that cheaper output can compensate for.

### 4. Active learning and value of information

When candidate designs remain uncertain, Symthaea should ask which additional datum, experiment, reference calculation, or inspection is most likely to change the decision. Estimate value of information against engineering objectives and uncertainty—not just model curiosity.

This stage ranks and proposes evidence-gathering work. It does not autonomously authorize hazardous experiments. Any physical campaign remains subject to qualified experimental design, institutional/site procedures and human approval.

### 5. Closed-loop, multi-fidelity engineering

Use cheap models for broad screening, qualified detailed models for refinement, and independent measurement for calibration and discrepancy correction. A surrogate must state its training domain, applicability conditions, error bounds, out-of-distribution test and fallback path. Extrapolation beyond that envelope blocks a qualified prediction.

Use measurement residuals to investigate model discrepancies, not to silently retune source inputs or relax acceptance thresholds. The evaluation corpus and thresholds must be frozen before test execution.

### 6. Manufacture and reproduce the process, not just the plant drawing

Extend the flowsheet through equipment and productive-capability dependencies: material grades, pumps, valves, seals, instrumentation, utilities, calibration, maintainability, spares, quality control, waste recovery and supplier/lot provenance. Distinguish a nominally complete bill of materials from an actually qualified, buildable and maintainable plant.

This connects Symthaea's process reasoning to manufacturing and Mycelix operational provenance without allowing inventory records to establish chemical purity or engineering qualification.

### 7. Safety by construction and explicit authority boundaries

Represent missing safeguards, unresolved hazards, uncertain material compatibility, unsupported operating envelopes, environmental obligations and required independent reviews as typed blockers. Prefer safer alternatives during candidate generation by comparing minimization, substitution, moderation and simplification.

A software benchmark can verify that a blocker is surfaced and remains blocking. It cannot certify a HAZOP, process hazard analysis, SIL assignment, regulatory compliance, plant design or safe operation. Physical actuation and operating authority remain outside this research artifact.

## Benchmark design

Freeze the benchmark manifest before measuring performance. Each case binds the graph, exact inputs, units, source/model versions, expected evidence state, quantitative reference where available, acceptance tolerance, permitted numerical method and known non-claims.

The first campaign should have three distinct tiers. Do not combine their results into one pass rate.

### Tier A — Deterministic synthetic contracts

Start with at least 20 small synthetic graphs whose correct outcomes are analytically known or explicitly specified. Use benign, non-operational examples: abstract component balances, mixers, splitters, simple heat duties, recycle graph structure and generic separation accounting. Do not encode hazardous operating recipes.

Initial corpus:

1. Balanced two-stream mixer: composition and total-flow closure.
2. Elemental imbalance: reject or block a stream transfer that violates declared element conservation.
3. Unit mismatch: refuse silent mixing of incompatible units or quantity dimensions.
4. Scale invariance: equivalent representations at different units/scales produce equivalent normalized results after explicit conversion.
5. Missing outlet/by-product: mass closure stays unresolved rather than assuming disappearance.
6. Explicit waste sink: only the declared quantity is accounted for by that sink.
7. Analytical recycle: converge to a known solution within the frozen tolerance.
8. Nonconvergent or singular recycle: report failure, never a valid design.
9. Property reference-state mismatch: prevent direct comparison until conventions are reconciled.
10. Separation trade-off: do not equate selectivity with recovery or product purity.
11. Unknown impurity: block a grade-sensitive downstream admission.
12. Stale data or missing source: refuse promotion to sourced/current evidence.
13. Out-of-domain model: block qualified prediction or require an explicitly qualified alternative.
14. Missing utility: product graph may be structurally valid but plant capability remains incomplete.
15. Lower operating cost with a violated hard constraint: mark infeasible.
16. Uncertainty changes candidate order: preserve the change and sensitivity; do not hide it behind a deterministic point estimate.
17. Duplicate or conflicting stream identity: refuse ambiguous graph resolution.
18. Missing equipment or maintenance dependency: block the applicable capability-closure claim.
19. Simulation mistaken for observation: keep the evidence classes distinct.
20. A component with unknown purity used as if it were specification-grade: reject the grade promotion.

21. Common-mode evidence dependency: different solver implementations share a property-data digest; the result must remain unresolved rather than count agreement as independent corroboration.
22. Independence positive control: complete manifests with distinct implementation digests and disjoint evidence-data dependencies pass only the synthetic independence-contract check, without promoting physical model validation.

**Tier A acceptance target:** 100% agreement with the frozen expected disposition for every case. A correct disposition may be BLOCKED, UNRESOLVED or OUT_OF_DOMAIN; rejection is the intended result for some tests. This target measures contract correctness, not physical-model accuracy. Target zero silent promotions and zero missing-stream erasures.

For accepted numerical fixtures, define a normalized residual before execution:

    r = |accounted_in - accounted_out - declared_generation|
        / max(reference_scale, sum(abs(accounting_terms)))

Use separate component-wise mass, elemental and energy equations, with sign conventions defined per profile. For the simplest analytically solvable fixtures, use a target of r <= 1e-8; set stricter or looser thresholds only where a documented numerical analysis justifies it. Do not use one threshold to conceal cancellation, missing terms or a bad reference scale.

### Tier B — Independent model and reference-flow-sheet comparisons

Begin with a small, fixed set of open, reproducible cases that cover property evaluation, equilibrium/thermodynamics, reaction-network simulation, heat/material balances, and a benign water/resource-recovery train. Use matched input assumptions, phase definitions, property methods, units, and reference states.

Compare with relevant external tools rather than reimplementing them:

- [Cantera reactor networks](https://cantera.org/stable/reference/reactors/index.html) for reacting-system dynamics and stiff ODE/DAE integration.
- [DWSIM streams and balances](https://dwsim.org/tutorials/en/fundamentals/01-streams-state-and-balances.html) and [CAPE-OPEN support](https://dwsim.org/help/main/60-cape-open-subsystem.html) for process flowsheets and component interoperability.
- [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) for sourced thermochemical, thermophysical and reaction data.
- [IDAES](https://idaes.org/) as an additional process-systems engineering framework and candidate source of reproducible flowsheet/optimization examples, after checking each example's license, provenance and applicability.

These engines are comparators, not unquestionable oracles. Agreement between two models is not independent experimental validation when they share the same data or assumptions. Before comparing, reconcile reference states, units, species sets, phases and solver tolerances. Record disagreements; never average incompatible results into apparent agreement.

**Tier B target:**
- At least 3 independent reference cases in the first slice, then expand the corpus.
- 100% of runs produce a receipt including solver outcome, inputs and transitive data/model provenance.
- Every quantitative error threshold is frozen from documented reference uncertainty and numerical requirements before test execution.
- No claim of universal model accuracy from a small corpus.
- External tool absence or execution skip is reported as SKIPPED, never PASS.

### Tier C — Inverse design, uncertainty and robust decisions

Use a small analytically solvable optimization problem first, followed by more realistic documented reference flowsheets.

Measure:
- feasibility rate and hard-constraint violations;
- distance to a known optimum or Pareto frontier on synthetic cases;
- Pareto hypervolume against frozen baseline algorithms where a known reference frontier exists;
- repeatability across seeds and builds;
- sensitivity to feed composition, prices, utility availability and property uncertainty;
- calibration of prediction intervals on held-out cases;
- quality and usefulness of active-learning recommendations;
- performance cost per independently accepted design.

**Proposed initial targets:**
- Zero accepted candidates violate hard constraints in the frozen benchmark.
- For analytic optimization cases, reach at least 95% of reference Pareto hypervolume under a preregistered metric and compute budget; report all failures and seeds.
- For uncertainty intervals, report empirical coverage with sample sizes and confidence intervals. Do not claim calibration from a handful of cases.
- Demonstrate that an out-of-domain candidate is flagged before its surrogate prediction can qualify a design.
- Compare cost/quality against explicit baselines; no claim of “best design” without a reference set and fixed budget.

These are proposed research gates; none is presently claimed as measured or passed.

## Risk-adaptive credibility: spend verification effort where it changes decisions

A flat validation checklist is not enough for a process portfolio that includes models of very different consequence. Use a risk-informed credibility profile based on at least (a) the model's influence on a decision, (b) the consequence of a wrong decision, (c) extrapolation distance from validation evidence, and (d) reversibility/detectability of a failure. This follows the direction of NIST's 2026 work on harmonizing computational-model uncertainty with measurement practice and NIST's manufacturing digital-twin guidance; it is a design principle, not a declaration of standards compliance.

For each model-to-claim edge, freeze a **credibility budget** before evaluating candidates:
- intended context of use and the exact claim it supports;
- failure modes and consequence class;
- verification depth, validation evidence and independent comparison required;
- treatment of measurement, parameter, numerical and model-form uncertainty;
- allowable extrapolation and out-of-domain detection;
- decision sensitivity and value of additional information;
- expiry, data-currentness and requalification triggers.

Use distributions, not only point estimates, where uncertainty materially affects the engineering decision. Preserve correlation and common-mode uncertainty when multiple models share a property package, dataset, fitted correlation or assumptions. Independent software does not imply independent evidence. When shared uncertainty can reorder the Pareto frontier, the system should say so and prioritize evidence that can resolve the decision—not average disagreeing answers or call the apparent winner robust.

**Benchmark additions:**
1. Keep all 26 synthetic Tier-A fixture dispositions exact and fail-closed.
2. Add paired decision cases with identical nominal predictions but different uncertainty and consequence profiles; require the higher-consequence case to request stronger evidence rather than inherit the lower-risk approval.
3. Add common-mode test cases where two independent-looking engines share a stale property source; the evaluator must not count agreement as independent corroboration.
4. Add value-of-information tests with a synthetic known decision boundary; confirm the selected next observation reduces expected decision loss under the frozen loss model, not merely model variance.
5. Report calibration and coverage with sample size and confidence intervals; do not claim calibration on synthetic cases alone.

A proposed metric is **decision-weighted regret** against the oracle decision for known synthetic problems, reported alongside feasibility and uncertainty calibration. Do not multiply away hard safety constraints with a risk score: hard gates remain non-compensatory. The model-risk profile changes the amount and independence of evidence required, not the laws of conservation or the right to bypass plant approval.

## Cross-cutting verification requirements

Every benchmark run should create an immutable manifest/receipt containing:

- schema and benchmark version;
- repository commit and exact dirty/clean source status;
- case ID and case/input digests;
- property-data sources, dates/currentness, model and solver versions;
- units, reference-state conventions and operating-envelope assumptions;
- method selection, tolerances, seed, resource limits and environment;
- solver status, convergence diagnostics and every failed attempt;
- conservation residuals, uncertainty representation, discrepancy and disposition;
- independent checker version and result;
- explicit claim ceiling: what the run does and does not establish.

The evaluator must run independently of candidate generation. Golden fixtures and acceptance thresholds must not be modified as part of a candidate's own repair. Include negative controls in every release. Bind every verdict to the exact source head and input digests; queued, skipped or stale CI is not a pass.

## Executable Tier-A contract checker

The first Rust checker lives at `crates/domains/symthaea-engineering/examples/chemical_process_benchmark_evaluate.rs`. It checks a **candidate result report** against this frozen corpus. It does not generate process candidates or execute a chemistry/physics solver; its independent numeric oracles are deliberately limited to basic synthetic bookkeeping. The CLI adds SHA-256 digests of the exact corpus and candidate-result file bytes to each emitted receipt. The evaluator uses the existing workspace `sha2` dependency; `Cargo.toml` and the direct dependency edge in `Cargo.lock` are updated together.

Commands, run from the repository root:

```bash
cargo test -p symthaea-engineering --test chemical_process_corpus_contract
cargo test -p symthaea-engineering --example chemical_process_benchmark_evaluate
cargo run -p symthaea-engineering --example chemical_process_benchmark_evaluate -- \
  --corpus docs/engineering/fixtures/CHEMICAL_PROCESS_QUALIFICATION_V1.json \
  --results /path/to/candidate-results.json \
  --expected-head "$(git rev-parse HEAD)"
```

The candidate results file must declare `benchmark_id`, `corpus_version`, `evaluated_head`, and `cases`. Each result file digest identifies the exact bytes evaluated; it does not yet prove transitive source/model-data provenance or verify a full engineering evidence graph. Each case row declares `id`, `observed_disposition`, `observed_findings`, and `observed_promotions`. The two mixer cases also supply `observed_numeric_values.component_balance_residuals`; the checker independently sums per-component inlet and outlet flows, verifies unit consistency and recomputes raw/normalized residuals before deriving the correct disposition. The explicit conversion case supplies `observed_numeric_values.converted_value`; the checker independently calculates `from_value * conversion_factor` and compares it to both the frozen reference and candidate result. The analytic recycle case supplies `observed_numeric_values.solution`; the checker recomputes the equation residual and error to the closed-form solution. These oracles do not trust reported finding labels as substitutes for arithmetic. The evaluator compares the disposition with the frozen expectation; verifies all required findings are present; rejects any observed prohibited promotion; fails missing, duplicate or unexpected case IDs; and requires the reported head to match the independently supplied exact commit ID. It emits a deterministic JSON report and returns exit status 0 only when all cases and the report-level checks pass; status 1 means the supplied report failed evaluation, while status 2 means the evaluator could not read/parse its inputs or its command line was invalid.

The `chemical_process_corpus_contract.rs` integration test separately validates corpus versioning, 26 unique case IDs, required fields, allowed dispositions/classes, summary counters, non-operational scope declarations and residual-profile caveats. Sixteen unit tests exercise synthetic pass/fail behavior, including disposition mismatch, missing findings, prohibited promotions, head mismatch, duplicate/unexpected IDs, missing cases, malformed-corpus fail-closed behavior, independent numeric checks for mixer balances, explicit unit conversion, the analytic recycle equation, a standard SHA-256 test vector, and a corpus-wide check of all 22 non-numeric semantic oracles.

**Boundary:** these checks qualify benchmark-contract handling only. They do not establish process-model accuracy, mass/energy solver correctness, scientific validity, plant safety, product grade, economic performance, or regulatory qualification. The CLI report binds to the exact source head supplied by the caller; content-addressed SHA-256 fields for external transitive inputs remain a follow-on requirement for the full evidence-receipt implementation.

## Recommended execution sequence

1. **Freeze Tier A manifest and schema** as documentation/data-only work. Review ownership against #5970, #5686, #5199 and #1604 to avoid duplicate contracts.
2. **Map each fixture to its canonical owner**: quantity validation, chemical/QC evidence, thermofluids/CFD, separation, manufacturing capability, provenance and engineering obligations. Do not build a second balance/evidence engine inside the benchmark.
3. **Build an independent evaluator** for exact expected disposition and residual calculations. Keep the candidate's output separate from the evaluator's authority.
4. **Add a reproducible external-reference lane** for a few matched cases; freeze transitive inputs, method conventions and explicit skip/failure semantics.
5. **Only after the above qualifies**, add inverse-design and uncertainty experiments; do not tune the evaluation corpus on the results.
6. **Use a benign water/resource-recovery pilot** to exercise multi-stage separation, purity/recovery distinction, energy/water accounting, by-products and equipment/utility closure. Then expand to other chemical families through domain profiles.

## Open research questions

- Can a typed process graph statically identify missing equations, degrees of freedom, inconsistent quantities and unconnected streams before a solver is called?
- Can model selection and uncertainty propagation cross chemistry/kinetics/transport/flowsheet layers without losing provenance or double-counting shared data uncertainty?
- Which model discrepancies are decisive enough to change the Pareto ranking, and what is the value of resolving them?
- Can safety-relevant constraints be represented as non-relaxable feasibility conditions while still allowing meaningful trade-off analysis among feasible candidates?
- How much industrial reproduction can be demonstrated when utilities, maintenance, tooling, calibration and upstream productive equipment are included?
- How can common-mode errors be detected when multiple simulators share property packages, correlations, source data or assumptions?

## Claim ceiling

This benchmark specification establishes a proposed test program and target criteria only. It does not establish that any solver is qualified, any chemical route is feasible, any process has been physically demonstrated, any plant is safe, any product meets a grade specification, any industrial facility is economically viable, or any design is authorized for construction or operation.

## References

1. Cantera, [Reactors and Reactor Networks](https://cantera.org/stable/reference/reactors/index.html).
2. DWSIM, [Streams, State and Balances](https://dwsim.org/tutorials/en/fundamentals/01-streams-state-and-balances.html) and [CAPE-OPEN Subsystem](https://dwsim.org/help/main/60-cape-open-subsystem.html).
3. NIST, [Chemistry WebBook SRD 69](https://webbook.nist.gov/chemistry/).
4. CO-LaN, [The CAPE-OPEN Standard](https://www.colan.org/).
5. OSHA, [Process Safety Management, 29 CFR 1910.119](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.119).
6. AIChE, [Principles of Inherently Safer Design (2026)](https://publications.aiche.org/cep/2026/february/principles-inherently-safer-design).
7. NIST, [On Methods of Uncertainty Quantification in Computational Modeling and Simulation: Harmonizing Modeling and Metrology Practices](https://doi.org/10.6028/NIST.AMS.100-85) (2026).
8. NIST, [Digital Twins Workshops Summary Report (NISTIR 8620)](https://doi.org/10.6028/NIST.IR.8620) (2026).
