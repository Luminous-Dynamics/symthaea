# MFG-ACTIVE-001 — Manufacturing-conditioned active-learning benchmark

**Status:** protocol proposal; not a result  
**Owners:** Symthaea (selection policy and uncertainty report), Symtropy (deterministic scenario/replay where suitable), Mycelix (resource, lot, execution-event and provenance interfaces)  
**Related work:** Symthaea [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608), [MAT-ACTIVE-001 #6298](https://github.com/Luminous-Dynamics/symthaea/issues/6298), [SCI-EXP-001 #6226](https://github.com/Luminous-Dynamics/symthaea/issues/6226), [MAT-UQ-001 #6225](https://github.com/Luminous-Dynamics/symthaea/issues/6225), Symtropy [ENG-SYNTH-TWIN-001 #1467](https://github.com/Luminous-Dynamics/symtropy/issues/1467), Mycelix [fabrication monitor #4283](https://github.com/Luminous-Dynamics/mycelix/issues/4283).

This document instantiates existing active-learning and manufacturing-value contracts as a narrow, falsifiable benchmark. It does not introduce a second process ontology, scheduler, evidence ledger, digital-twin runtime, or manufacturing authority layer.

## 1. Question and falsifiable hypothesis

**Question:** Under an equal experiment-cost budget and identical declared constraints, does a manufacturing-aware experiment selector discover a useful, conforming process/material configuration more cheaply or reliably than transparent conventional experiment-selection policies?

The candidate system must select experiments that are both informative and executable within the declared manufacturing envelope. Information gain is not useful if the selected experiment cannot be run, cannot be measured, does not distinguish the stated hypotheses, or optimizes a response that cannot change an engineering decision.

**Null is a valid result.** Symthaea may be worse than or indistinguishable from the strongest baseline. Report that result rather than changing objectives, constraints, holdouts, or the primary metric after looking at outcomes.

This benchmark initially uses deterministic synthetic environments with explicitly hidden response functions. It establishes selector semantics and comparative behavior only—not a real material property, printer setting, machining procedure, process qualification, or physical savings claim.

## 2. Ownership and interfaces

- **Symthaea:** propose candidate experiments; state the uncertainty, competing hypotheses, engineering decision and measurements each experiment is expected to resolve; supply calibrated predictive uncertainty where supported.
- **Symtropy:** optionally supply qualified, deterministic scenario/rehearsal outcomes and replay receipts. A simulation output remains simulated evidence and may not be relabeled as a physical observation.
- **Mycelix:** represent resource/capability offers, material lots, work orders, run events and provenance as declared external references. A capability label or producer offer is not proof that a specific experiment is currently feasible.
- **Existing canonical owners:** process/profile and transformation contracts; quantities/units/frames; measurement-system identity; material-property evidence; physical metrology/inspection; and machine/safety authority remain with their established owners. This benchmark consumes those semantics rather than reimplementing them.

No selector in this benchmark may dispatch to a real machine, alter a recipe, place an order, or promote a part to accepted. Real execution remains behind existing operator authorization, process qualification, and machine-safety systems.

## 3. Frozen benchmark environment

### 3.1 Candidate and experiment record

Each candidate experiment must have a stable ID and enough immutable context to evaluate it without guessing:

- exact design/material/process profile references and revisions;
- proposed controllable factor values, units and declared ranges;
- declared protected variables and hard constraints;
- feasibility disposition: `Feasible`, `Blocked(reason)`, or `Unknown(reason)`;
- the capability and resource evidence supporting feasibility (including machine/workcell, material/lot, tooling, calibration, measurement method, time and cost envelopes as applicable);
- target responses and measurement plan, including expected measurement uncertainty and missingness;
- predicted outcomes and uncertainty, selector/policy version, and the hypothesis or decision being discriminated;
- expected experiment cost, duration and consumables, each with units and provenance;
- parent knowledge-state digest and experiment-history identities.

Unknown is not feasible. Blocked and Unknown candidates are ineligible for selection; they remain counted in coverage/refusal metrics. A selector must never invent a missing cost as zero or turn a model's proposed process window into evidence of capability.

The synthetic environment may use normalized, dimensionless variables to avoid implying real operating instructions. All domains, bounds, transforms, hard gates, response definitions and hidden-oracle version must be frozen in a manifest before any selector is scored.

### 3.2 Independent hidden response surface

The evaluator owns a deterministic response function inaccessible to the selector at decision time. Responses should include independent dimensions rather than one unconstrained scalar:

- product/part conformity against explicit synthetic requirements;
- quality or performance margin to those requirements;
- cycle-time/throughput proxy;
- material, energy and consumables per conforming unit;
- experiment cost and measurement uncertainty;
- optional degradation or lifecycle penalty, only when the scenario defines its meaning.

Synthetic observations returned to a selector must contain only the measurements and metadata that an admissible experiment would reveal. Future observations, scenario labels, hidden thresholds, ground-truth values for unrun candidates, and evaluation seeds must not be visible to the policy. The oracle is for evaluator scoring only.

Hard feasibility, safety and critical-quality constraints are gates, not weights in a composite score. Faster or cheaper candidates that fail a critical gate cannot count as useful discoveries.

### 3.3 Scenario families

Freeze at least these five compact synthetic families:

1. **Smooth response:** a feasible optimum lies within the supported domain; uncertainty should aid sample efficiency.
2. **Narrow feasible window:** a high-performing region is easy to miss; infeasible candidates can appear attractive to a naive optimizer.
3. **Material/process interaction:** a process factor's effect depends on material profile; one-factor-at-a-time reasoning may mislead.
4. **Confounded evidence:** composition/material and process changes are correlated in the observed history; the policy must request a discriminating experiment or report that the effect is not identifiable.
5. **Multi-fidelity discrepancy:** cheap model/simulation observations are biased relative to a more expensive reference oracle in a declared regime; the selector must account for discrepancy instead of treating fidelity levels as interchangeable.

Add adversarial variants for stale calibration, missing critical measurement, duplicated/reordered observations, changed equipment or material-lot identity, measurement noise, repeated uninformative experiments, and a candidate whose predicted information gain is high but whose feasibility is Blocked or Unknown. All scenario parameters, allowed actions, observation rules and expected invariants must be frozen before evaluation.

## 4. Policies to compare

Every arm gets the same initial observations, candidate domain, hard constraints, maximum experiment count, total budget, observation model and stopping rules.

- **P0 — fixed design:** a declared space-filling or small factorial design, with no adaptive selection. This establishes a transparent experimental-design baseline.
- **P1 — random-feasible:** uniform selection from currently feasible, not-yet-run candidates; useful as a low-complexity floor.
- **P2 — uncertainty-only:** choose the feasible experiment with highest declared predictive uncertainty for the target response.
- **P3 — information-gain-only:** choose the feasible experiment maximizing expected information gain or expected discrimination under the same measurement model.
- **P4 — cost-aware information gain:** P3 with an explicit cost denominator and preregistered handling for zero/uncertain costs.
- **P5 — decision-aware / manufacturing-conditioned candidate:** use value of information or expected reduction in decision regret, filtered by process/capability/measurement feasibility and hard quality requirements; report the explanation for each choice.
- **Oracle diagnostic (tiny fixtures only):** exhaustive/future-informed upper bound used only to quantify available headroom. Mark it **non-deployable** and never report it as an operational competitor.

Use deterministic tie-breaking on stable candidate IDs. If a policy is stochastic, its random stream must be isolated from the oracle and bound to the run manifest. A complex policy may only claim its measured contribution; do not assume the Symthaea arm wins.

## 5. Outcomes — no single factory score

Choose one primary metric per preregistered experiment. Retain the full metric vector and critical guardrails:

1. **Primary discovery outcome:** experiment cost to first independently verified, conforming candidate, with a right-censored/NotFound result when the budget ends first.
2. **Budgeted decision outcome:** best known conforming engineering utility at fixed cumulative-cost checkpoints; report the Pareto set where objectives are genuinely multiobjective.
3. **Efficiency:** cumulative cost, wall-clock-free simulated experiment time, experiments and measurement resources to discovery.
4. **Feasibility quality:** proposed candidate count; blocked/unknown attempts refused; infeasible proposals that escaped the gate (target zero); fraction of budget spent on failed or non-discriminating experiments.
5. **Learning value:** uncertainty calibration/coverage where statistically defined; discrimination among the declared competing hypotheses; held-out predictive error by operating regime; expected versus realized value of information.
6. **Robustness:** performance under sensor missingness, drift, changed profile identity, noise, candidate-domain changes and oracle discrepancy.
7. **Protocol integrity:** deterministic replay; duplicate proposal handling; data leakage; number of hidden-oracle accesses; complete provenance and independent-evaluator agreement.

Do not let any one metric offset a critical constraint violation. Report each scenario separately and in aggregate with the scenario-level results available. Do not pool incomparable process families into a universal score.

## 6. Experimental design and leakage controls

- Freeze the manifest, primary metric, guardrails, all selector versions, baselines, stopping rule, seed list, scenario split and analysis script before scoring.
- Use paired seeds and identical budgets across policy arms. Start with at least 30 paired seeds per scenario family as a provisional floor; increase the count when uncertainty is too wide to discriminate the preregistered practical effect. This is a proposed benchmark rule, not an industry standard.
- Keep complete scenario families / hidden response functions in holdout; do not randomly split neighboring observations from one synthetic surface between tuning and test.
- If a policy is tuned on a seed or surface, that seed/surface is development data, not holdout.
- Every observation must bind to the exact experiment, scenario, measurement model, profile, parent knowledge-state digest, fidelity and source. Repeated or correlated measurements must not be treated as independent samples by default.
- Discovery data and confirmatory data are different state transitions. After policy selection or adaptive exploration, reserve fresh predeclared confirmatory experiments for a confirmatory claim; do not reuse the adaptive search results as though they were prospectively independent confirmation.
- Publish all failed episodes, timeouts, refused candidates and missing results. No silent reruns or removed outliers. Any rerun is a new attempt linked to the original.
- Report effect sizes and uncertainty intervals, not just ranks or p-values. The evaluator should be able to conclude Better, Worse, Inconclusive or NotEvaluable.

## 7. Independent evaluator and hostile controls

The evaluator must be a distinct executable or separately implemented module that consumes only the frozen manifest, selector transcript and oracle receipts. It recomputes budget accounting, feasibility, conformance, discovery time/cost, aggregate metrics and all invariants; it must not trust selector-provided totals.

Minimum negative controls:

1. an infeasible candidate has the largest claimed information gain -> candidate must be blocked before selection;
2. feasibility is Unknown or supporting evidence is stale -> candidate must not be selected as runnable;
3. hard quality requirement fails while throughput improves -> candidate does not count as a discovery;
4. duplicate measurement is replayed under a new ID -> duplicate lineage must be detected or dependence exposed;
5. material and process variables change together -> selector cannot assert a causal effect without a discriminating design;
6. model and reference oracle disagree in a declared regime -> discrepancy remains visible and uncertainty cannot disappear;
7. calibration/profile identity changes -> stale observations cannot silently qualify a new profile;
8. missing critical measurement is converted to zero by a defective selector -> independent evaluator rejects the fabricated value;
9. selector asks for an experiment outside the declared envelope or outside its declared measurement capability -> refuse with a reason;
10. future oracle value, hidden scenario label or test seed leaks into policy input -> evaluation is invalid;
11. one arm spends more money/queries or receives more information than another -> fairness gate fails;
12. a failure is dropped from the final report or a held-out surface is used for tuning -> evaluation is invalid.

The independent evaluator should include mutation tests demonstrating that it rejects at least corrupted cost totals, a false conforming flag, a missing required measurement, a duplicated result, a mismatched profile, and a changed corpus digest.

## 8. Acceptance ladder

- **A0 — frozen contract:** versioned manifest schema, domain and scenario definitions, baseline specs, primary metrics, guardrails and negative controls are frozen; corpus counts and hashes are recorded.
- **A1 — deterministic replay:** the same code SHA, manifest, policy/configuration, observation history and seed yield byte-stable normalized transcript and results.
- **A2 — independent recomputation:** evaluator reproduces every score from transcript/oracle receipts and detects mutations in all required negative controls.
- **A3 — fair policy comparison:** all operational policies run over the same preregistered portfolio and budgets, with complete failures, uncertainty and ablations disclosed. No minimum win is required to pass the protocol.
- **A4 — public-data admission:** separate source/license/provenance audit; exact dataset revision and preprocessing manifest; split by build/run/machine/lot as appropriate; units, missingness and measurement system retained. Public-data results remain process- and dataset-specific.
- **A5 — shadow/physical pilot (future):** requires domain expert, metrology and safety review, prior registration, authorized execution and independent inspection. A Stage 0 win is not evidence of physical benefit.

### Proposed project-level improvement hurdle

Only after Stage 3 of the broader MFG-VALUE-001 program, target at least a 10% improvement on one predeclared business metric while meeting quality and safety guardrails. That is a project hurdle, not a universal norm. This active-learning benchmark by itself should use discovery cost and conforming discovery under equal budgets; it must not claim 10% factory cost reduction without a physical production pilot.

## 9. Run receipt

Every run must emit a machine-readable, content-addressed receipt containing:

- benchmark ID/version and scenario manifest digest;
- repository, exact code commit, build/toolchain and policy configuration identity;
- seed, initial observations, allowed candidate IDs and domain digest;
- observation history and exact source/measurement/fidelity references;
- selector transcript with decision rationale, predicted information/value, feasibility disposition and cumulative cost;
- oracle receipts for executed experiments only, with inaccessible future labels omitted from selector input;
- independent evaluator version, output digest, metric vector, negative-control results and all failure/refusal counts;
- train/tune/holdout designation, exclusions, limitations and claim ceiling.

A matching hash proves identity of bytes, not scientific validity or physical applicability. Synthetic, simulated, recorded and physical evidence types must remain separate.

## 10. External research and later dataset adapters

The design follows the closed-loop evaluation principle in **MADE: Benchmark Environments for Closed-Loop Materials Discovery** (ICML/PMLR 2026): compare discovery policies under constrained oracle budgets and transparent baselines. MADE is a materials-discovery benchmark, not manufacturing production data, and its task-specific findings must not be transferred to factory outcomes: https://proceedings.mlr.press/v306/malik26a.html

The 2026 *Nature Reviews Chemistry* review of self-driving laboratories identifies scalability, generalizability and provenance-complete experimentation as key needs: https://www.nature.com/articles/s41570-026-00847-2

NIST's **2026 Roadmap on AI/ML for Smart Manufacturing** highlights heterogeneous industrial data, reliability, metrology, digital twins, robotics, logistics and sustainable manufacturing as active challenges: https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing

NIST's digital-twin program emphasizes interoperability and verification, validation and uncertainty quantification (VVUQ): https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing

Subsequent dataset tracks must remain separate and process-specific:
- NIST Additive Manufacturing Metrology Testbed (AMMT) in-situ observations, with the exact build, sensor and inspection lineage admitted: https://www.nist.gov/publications/process-monitoring-dataset-additive-manufacturing-metrology-testbed-ammt-three
- NASA Milling Wear, a distinct subtractive-manufacturing tool-wear dataset: https://data.nasa.gov/dataset/milling-wear

Do not combine these datasets as if they measured the same physical process or transfer one process's thresholds to another. For live data integration, prefer established interfaces (for example, MTConnect) via explicit adapters, while preserving source units and semantics: https://www.mtconnect.org/

## Claim ceiling

A passing A0–A3 establishes a reproducible synthetic comparison of experiment-selection policies on the frozen environments. It establishes no physical material discovery, validated process window, defect-detection accuracy, conforming product, production yield, cost savings, self-sufficient microfactory, or authority to operate equipment.
