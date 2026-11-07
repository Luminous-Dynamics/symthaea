# Aircraft Engineering Loop v0.1

**Parent:** [AERO-ENGINEERING-LOOP-001 #10902](https://github.com/Luminous-Dynamics/symthaea/issues/10902)  
**Symthaea transport parent:** [TRANSPORT-ENG-001 #6244](https://github.com/Luminous-Dynamics/symthaea/issues/6244)  
**Symtropy aerospace MDO owner:** [AERO-MDO-INTEROP-001 #3237](https://github.com/Luminous-Dynamics/symtropy/issues/3237)  
**Status:** architecture/contract work; no aircraft-specific physics or design result is established.

## 1. Decision

Symthaea should do more of aircraft engineering, but it should not become the CFD solver, finite-element solver, flight-dynamics authority, or design-release authority.

The appropriate role is an **evidence-aware adaptive engineering scientist**:

- structure a mission and its requirements without silently turning preferences into hard requirements;
- formulate candidate architectures and identify the unknowns that could change a decision;
- select the next analysis, measurement or test *and its fidelity* under a frozen engineering budget;
- coordinate exact solver requests through declared adapters;
- interpret discrepancy, applicability, uncertainty and constraint impact;
- preserve a robust set of alternatives rather than inventing a single weighted "best aircraft";
- explain why an evaluation is worth its cost, or abstain when it is not.

Symtropy and qualified backends remain responsible for deterministic engineering computations, exact inputs/outputs, conservation, residuals and replay. Human engineering review and competent authorities retain release, safety and certification decisions.

## 2. The loop

`Mission / intended use
  -> requirements and hard constraints
  -> candidate architecture + exact configuration
  -> required operating-case coverage
  -> select candidate + analysis method + fidelity
  -> solver / measurement request
  -> evidence-bound evaluation receipt
  -> independent checks + discrepancy / uncertainty update
  -> robust frontier and constraint status
  -> next decision-value evaluation
  -> human-reviewed design decision`

The loop is adaptive rather than a fixed one-way funnel. A cheap analysis can screen a candidate, but its cost must include any later validation required before a design decision. A low-fidelity optimum outside its validated domain is a request for more evidence, not a winner.

## 3. Authority and ownership

| Concern | Owner | Symthaea's role |
|---|---|---|
| Design intent, requirement decomposition, design study | `symthaea-engineering` / ENG-DESIGN-002 #6090 | Structure, trace and expose assumptions; do not invent acceptance limits |
| Fixed-wing aero, flight dynamics, loads and applicability | #6249 + domain model owners | Choose analyses and interpret qualified outputs |
| Candidate-specific mass-coupled mission | #7729 and aircraft-energy owners | Formulate comparable mission cases and identify decision-critical uncertainties |
| Design variables / objective semantics | #5635 | Supply domain-specific candidates and metrics to shared frontier methods |
| Interval robust Pareto relation | #7787 | Keep uncertain candidates separate; do not infer probability from interval endpoints |
| Decision-value benchmark / next experiment | #7803 and domain adapters | Select what to measure next under equal, explicit budgets |
| Cross-domain physical transfer | #6173 / #6180 | Request correct coupling profiles; reject field-name-only assumptions |
| Simulation request/result boundary | `symthaea-sim-bridge` | Orchestrate only explicitly registered, reviewed backends |
| MDO, solver interoperability, run artefacts | Symtropy #3237 | Consume exact receipts; do not reimplement discipline solvers |
| Aircraft chronology, loading, energy-plane arithmetic | Symtropy #4029 / #4036 | Bind state/phase identities; require independent replay |
| Safety case, release and certification | Human reviewers / competent authorities | Prepare evidence and unresolved obligations; never self-approve |

The machine-readable mapping is in [`owner-map-v0.1.json`](owner-map-v0.1.json).

## 4. Evaluation receipt boundary

A study references domain artefacts; it must not duplicate them into a second aircraft ontology. Each evaluation receipt should bind:

- candidate and exact geometry/configuration digest;
- mission and required operating-case identity;
- exact model, solver, data and protocol revisions;
- input/output artefact digests;
- evaluation method and evidence origin as separate fields;
- applicability disposition, convergence/residual state and model limitations;
- uncertainty semantics and dependence/common-mode notes;
- evaluation class/fidelity, cost basis, elapsed and consumed cost;
- failed, aborted, censored, missing and out-of-domain outcomes;
- exact claim ceiling.

A receipt does not make a solver valid, a source authentic or a model physically accurate. It makes the submitted result identifiable and reviewable.

## 5. Decision policy

Use hard-feasibility gates before objective comparison. Keep objective axes independent and retain infeasible/indeterminate counts visibly.

When defensible calibrated distributions exist, the decision policy may evaluate expected value of sample information or expected regret. With interval/set-valued evidence only, it must use distribution-free decision impact, robust feasibility or minimax regret. It must not convert interval midpoints or widths into probabilities.

Do not optimize for information gain alone. An experiment is valuable when it can change a consequential decision, resolve an important hard constraint, distinguish surviving robust alternatives, or falsify a model assumption. Unknown costs and unqualified experiments must not appear artificially cheap.

## 6. Industry comparisons

The aircraft-specific benchmark should compare under the same mission, payload, operating cases, reserve policy, environment, data, solver/fidelity budget and downstream validation cost:

- seeded random and space-filling / Latin-hypercube sampling;
- deterministic grid/sweep and sequential one-discipline-at-a-time design;
- evolutionary multi-objective search (NSGA-II/III);
- Bayesian optimization (only with defensible probabilistic assumptions);
- constrained and multi-fidelity Bayesian optimization;
- fixed-fidelity allocation and uncertainty/information-gain selection;
- decision-aware expected-regret selection;
- distribution-free robust-feasibility / minimax-regret selection;
- a conventional MDO workflow such as OpenMDAO/Dymos where the same disciplines are available.

Report time/cost to first feasible candidate, feasible-evaluation yield, false-feasible/missed-feasible rate, robust-frontier recovery, regret against hidden synthetic truth, high-fidelity follow-up cost, out-of-domain abstention, operating-case coverage, failure accounting and total engineering hours. A weighted score must not hide a failed hard constraint.

## 7. Initial gates

The first implementation should be a dry-run orchestration/receipt path, not an autonomous solver runner:

1. owner map is reviewed and duplicate ownership is rejected;
2. a frozen study manifest references exact candidate, mission, objective, constraint, model and operating-case profiles;
3. an evaluation proposal names the backend/method, fidelity, budget and decision it could change;
4. a dry-run produces no engineering result and cannot be counted as a solver evaluation;
5. only an allow-listed backend may return an evaluation receipt;
6. independent checks recompute the declared invariants;
7. model applicability, missing required phases, stale identities or hard violations prevent a feasible disposition;
8. a human review step remains mandatory before external design or physical-test decisions.

## 8. Claim ceiling

A successful synthetic policy benchmark establishes only that the frozen policy behaves as specified on the frozen synthetic oracle and budget. It does not validate aircraft physics, CFD/FEM, flight performance, safety, airworthiness, certification, passenger operations or commercial viability.

## 9. Executable dry-run boundary

Rust module: `crates/domains/symthaea-engineering/src/aircraft_design_loop.rs`

Wire schemas:
- `crates/domains/symthaea-engineering/schemas/aircraft-design-study.schema.json`
- `crates/domains/symthaea-engineering/schemas/aircraft-evaluation-proposal.schema.json`
- `crates/domains/symthaea-engineering/schemas/aircraft-evaluation-receipt.schema.json`

The bounded API currently:
- validates a study's exact mission, candidate, required operating-case, decision-question, objective, hard-constraint, backend, model-profile and cost-basis references;
- admits a proposal only when all references belong to the frozen study and the backend/method/model-profile combination is allow-listed;
- abstains when the proposed cost exceeds remaining budget;
- records actual failed, aborted, out-of-domain and non-converged outcomes without turning them into decision-eligible evidence;
- reports a completed, in-domain evaluation as within its declared model scope only—not as a physical validation or feasibility finding.

Every dry-run review has `execution_started = false`. No external process, solver, CAD system, network client or aircraft-control interface is invoked by this module. It cannot grant flight approval. The SHA-256 checks are syntax and reference-equality checks; artifact authentication and source acquisition are separate responsibilities.

## 10. Immediate follow-on gates

1. Add a canonical, independently recomputable digest for the serialized study and proposal payloads; do not use debug formatting or map iteration order.
2. Bind an evaluation receipt to the exact runner artifact, executable/environment identity, raw inputs, output bundle, applicability result and uncertainty profile.
3. Add independent fixtures for stale model profiles, case/configuration mismatch, budget overruns, missing run artifacts, non-convergence and false-feasible labels.
4. Compose with the existing Symtropy run manifest / receipt work. Do not create a second solver execution registry.
5. Keep human approval as a distinct, externally recorded review event before any engineering action outside the dry-run boundary.

The first Rust implementation is intentionally a study/receipt admission gate, not yet an adaptive experiment selector or a physics tool.
