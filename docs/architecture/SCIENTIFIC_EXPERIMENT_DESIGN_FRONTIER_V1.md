# SCI-009 — Scientific Experiment Design Frontier v1

Status: architecture contract only. Non-authorizing, non-qualifying, non-executing.

Parent: SCI-008 — Scientific Uncertainty-Bearing Observation v1.

## Purpose

Define the common experiment-selection boundary for Symthaea without collapsing heterogeneous scientific goals, uncertainty models, costs, risks, feasibility constraints, or falsification objectives into one universal scalar utility.

The repository already contains useful but intentionally different planning mechanisms:

- Conjecture Engine query-by-committee disagreement: deterministic prediction variance over candidate hypotheses;
- the #662 candidate: coverage-aware disagreement that makes unavailable/non-finite predictions explicit;
- Spark Engine: mutual-information expected information gain under an explicitly approximate signature-level outcome model;
- domain experiment programs with explicit cost, duration, instrumentation, controls, go/no-go semantics, and scientific success criteria;
- SCI-004 adaptive experiment contracts;
- SCI-007 preregistered falsifier semantics;
- SCI-008 uncertainty-bearing observation semantics.

SCI-009 preserves these as distinct planning models and defines the common contract needed to compare or compose them honestly.

## Core theorem

Experiment selection is not one scalar truth function.

The shared kernel must preserve at least the following distinctions:

- candidate experiment != executable experiment;
- feasibility != desirability;
- prediction coverage != information gain;
- disagreement != calibrated uncertainty;
- expected information gain != realized information gain;
- falsification power != posterior entropy reduction;
- scientific value != monetary cost;
- low cost != safe;
- risk constraint != negative utility term;
- novelty != scientific value;
- planner recommendation != experiment authorization;
- selected experiment != preregistered experiment;
- preregistered experiment != valid execution;
- successful execution != scientifically successful result;
- scientific result != action authority.

## Scope

SCI-009 owns only common experiment-design mechanics:

- exact candidate identity;
- exact hypothesis/proposition/falsifier target set;
- prediction/evaluation coverage;
- planner/model identity;
- planning-information snapshot;
- candidate feasibility state;
- multidimensional scientific-value coordinates;
- resource/cost/time coordinates;
- safety/risk admissibility coordinates;
- Pareto dominance/frontier semantics;
- optional explicitly registered scalar preference policies;
- sequential/adaptive replanning receipts;
- planner comparison and regret-style benchmark surfaces.

It does not define one universal scientific objective, Bayesian model, safety policy, cost function, experiment authorization policy, or optimality theorem.

## Existing baselines remain first-class

### Baseline A — query-by-committee disagreement

The Conjecture Engine's prediction variance is a useful deterministic baseline when hypotheses expose point predictions.

A stronger implementation may additionally retain:

- finite prediction count;
- total live hypothesis count;
- prediction coverage;
- identities of hypotheses with unavailable predictions;
- reason for unavailability when typed information exists.

The #662 direction remains explicitly not expected information gain.

### Baseline B — signature-model expected information gain

Spark Engine currently computes mutual information between hypothesis identity and discretized outcome class under a documented approximate likelihood model.

This is a legitimate EIG calculation relative to that exact model.

It must not silently become:

- calibrated physical-world information gain;
- proof that the hypothesis prior is calibrated;
- proof that the outcome model is complete;
- proof that the selected experiment is globally optimal.

The planning object should retain the exact belief model, likelihood/outcome model, discretization/profile, and planner implementation identity used to derive EIG.

### Baseline C — domain falsification objective

SCI-007 permits experiments whose primary value is testing an exact preregistered falsifier.

A candidate can have high falsification value even when a posterior/EIG model is unavailable or scientifically unjustified.

The common kernel therefore cannot require Bayesian probabilities as the only admissible experiment-design language.

## Candidate identity

A `ScientificExperimentCandidateV1`-like object should eventually bind exact identities for:

- candidate/profile version;
- target proposition/hypothesis/falsifier set;
- experimental action/intervention/setup;
- required measurement specifications;
- required SCI-008 observation/uncertainty profiles;
- required controls;
- required SCI-003 execution profiles where computational execution is part of the experiment;
- domain constraints;
- resource requirements;
- expected observation model references, if any;
- planner-visible metadata.

Human-readable names are navigation only.

Changing a scientifically material intervention, measurement, control, outcome model, or target creates a new candidate identity or an explicit candidate revision relation.

## Planning information snapshot

Experiment selection is conditional on information available at planning time.

A future `ExperimentPlanningSnapshotV1` should bind, where applicable:

- exact live hypothesis/proposition set;
- current scientific-view identity/cutoff;
- current model/belief state identity;
- prior/weight model identity;
- current evidence/dependency graph generation;
- current falsifier set;
- uncertainty/calibration artifacts;
- resource inventory;
- equipment availability;
- time constraints;
- safety/admissibility policy identity;
- planner implementation/profile identity.

A later planning decision produced after new evidence is a new planning event, not a mutation of the old one.

## Prediction coverage is a coordinate

Before ranking experiments on sophisticated epistemic value, the planner should know whether its live hypotheses can actually generate evaluable predictions.

Illustrative diagnostics include:

- hypotheses total;
- finite/evaluable predictions;
- missing predictions;
- invalid/non-finite predictions;
- prediction coverage ratio;
- coverage by hypothesis family or proposition scope;
- expected observation dimensions covered.

High variance among two predictions does not automatically outrank moderate discrimination across a complete hypothesis set.

Coverage itself is not scientific value; it is a diagnostic/admissibility coordinate.

## Expected information gain requires an exact probabilistic model

The term `ExpectedInformationGain` should be reserved for a calculation that actually specifies the probabilistic objects needed for the expectation.

At minimum the planning lineage should identify:

- prior/current hypothesis distribution or model state;
- outcome/likelihood model;
- candidate experiment;
- observation space/discretization or density representation;
- uncertainty/noise model where material;
- numerical approximation/integration method where used;
- implementation identity;
- planning execution capsule.

A heuristic uncertainty reduction, variance score, entropy proxy, ensemble spread, or prediction disagreement should retain its own name.

## Model uncertainty and misspecification

The experiment planner itself is a scientific model and can be wrong.

A robust SCI-009 implementation should permit diagnostics such as:

- alternative prior sensitivity;
- alternative likelihood-model sensitivity;
- alternative discretization sensitivity;
- model-family sensitivity;
- OOD applicability;
- hypothesis-set incompleteness;
- calibration state;
- adversarial/simple-baseline comparison;
- unmodeled-outcome mass or `Other/Unknown` handling where domain-appropriate.

A candidate that is optimal only under one fragile modeling choice should retain that dependence.

## Multidimensional scientific value

The common layer should support separately named coordinates such as:

- expected information gain;
- query-by-committee disagreement;
- falsification power;
- expected hypothesis discrimination;
- expected reduction in a declared uncertainty object;
- expected measurement-quality improvement;
- expected resolution of a qualified defeater;
- expected ability to discriminate competing causal models;
- replication/triangulation value;
- expected OOD boundary information;
- novelty/search value within an exact scope.

These coordinates are not automatically commensurate.

## Cost, time, resources, and feasibility

Candidate experiments may also carry independent coordinates such as:

- monetary cost;
- duration;
- personnel/time burden;
- compute budget;
- energy/material requirements;
- equipment availability;
- sample availability;
- destructive sample consumption;
- regulatory/ethical constraints;
- scheduling dependencies;
- setup/changeover costs;
- expected failure probability;
- opportunity cost where explicitly modeled.

A candidate can be scientifically valuable but currently infeasible.

`Infeasible` is not equivalent to low scientific value.

## Safety and rights are not ordinary utility penalties

SCI-009 must not assume every risk can be traded against information gain by increasing a scalar coefficient.

Some constraints may be hard admissibility boundaries imposed by domain safety, ethics, consent, governance, rights, environmental protection, or physical-agency policy.

The generic sequence is:

1. candidate exists;
2. exact feasibility/admissibility policies are evaluated;
3. inadmissible candidates are excluded with reasons retained;
4. admissible candidates enter the scientific-value/resource comparison surface;
5. a planner proposes a candidate or frontier;
6. separate authorization determines whether execution may occur.

No experiment-design score grants physical, clinical, governance, financial, or other action authority.

## Pareto frontier as the default shared comparison

Where multiple objectives are present and no qualified scalar preference policy exists, SCI-009 should prefer a Pareto representation.

For a declared set of maximize/minimize coordinates and exact comparison rules, candidate A dominates candidate B only when A is no worse on every declared coordinate and strictly better on at least one.

The resulting frontier should retain:

- nondominated candidate identities;
- dominated candidate identities;
- exact dominance witnesses/reasons;
- unavailable/unknown coordinates;
- comparison-profile identity;
- excluded/inadmissible candidates separately.

Unknown values must not silently become zero or best/worst.

## Optional scalar preference policies

A domain may legitimately need one selected experiment rather than a frontier.

A scalar or lexicographic policy is allowed only when it is explicit, versioned, and prospectively bound to the planning/experiment contract where confirmatory authority matters.

Examples include:

- EIG per dollar;
- lexicographic `safety -> feasibility -> falsification power -> cost`;
- constrained optimization with declared maximum budget;
- weighted utility over exact normalized coordinates;
- minimax/regret policies.

The resulting scalar value is a property of that policy, not a universal `experiment_quality` score.

Changing weights, normalizations, thresholds, or ordering creates a new policy identity.

## Adaptive / sequential experiment design

SCI-004 already permits preregistered adaptive science.

SCI-009 refines that path:

- freeze the candidate-generation policy or candidate action space;
- freeze the planner family/implementation identity where scientifically material;
- freeze objective/constraint semantics;
- freeze update/replanning policy;
- freeze stopping rules;
- execute one experiment;
- admit actual observation/evidence through SCI-008 and relevant qualification layers;
- create a new planning snapshot;
- re-evaluate the frontier or policy;
- retain every planning and execution generation.

The system must not update its belief using a simulated MAP outcome and then present the resulting plan as if real data were observed.

Simulated rollout planning is allowed when explicitly labeled as such.

## Expected value != realized value

After an experiment executes, SCI-009 should permit a later comparison between predicted and realized planning value without rewriting the historical decision.

Illustrative records may include:

- expected EIG at selection;
- realized posterior entropy change under the declared model;
- predicted vs realized duration/cost;
- expected vs realized measurement quality;
- expected vs realized falsifier applicability;
- candidate failure or protocol deviation;
- planner calibration/retrospective regret metrics.

A bad realized outcome does not by itself prove the original decision was irrational; a good outcome does not prove the planner was well-calibrated.

## Planner comparison and meta-science

SCI-009 should make planning algorithms experimentally comparable.

A benchmark can compare, for example:

- random candidate choice;
- cost-only choice;
- variance-only query-by-committee;
- coverage-aware disagreement;
- signature-model EIG;
- richer uncertainty-aware EIG;
- falsification-first policy;
- Pareto/constraint-aware planning.

Comparisons should bind exact candidate sets, priors/models, simulated or historical worlds, random seeds, budgets, and scoring criteria.

No single benchmark should establish universal planner superiority.

## Integration with SCI-007 and SCI-008

SCI-007 provides exact falsifier targets and outcomes.

SCI-008 provides uncertainty-bearing predicted/observed quantities and calibration/applicability state.

Together they enable honest experiment-design questions such as:

- Which experiment has the highest expected ability to trigger or exclude an exact falsifier under the declared uncertainty model?
- Which experiment most separates competing predicted observation distributions?
- Which candidate is nondominated once scientific value, cost, duration, risk, and measurement feasibility are kept separate?

## Integration with SCI-006

Experiment choices can share dependencies.

Two nominally different experiments may rely on the same instrument, calibration, trained model, reference corpus, or learned grammar.

SCI-006 dependency structure can therefore inform replication/triangulation objectives and prevent a planner from treating superficial multiplicity as independent information.

## No novelty scalar shortcut

Novelty may matter for exploration, but SCI-009 does not define `novelty = high` as scientific value.

Any novelty/search coordinate should bind the exact search corpus, cutoff, query method, model/tool identities, nearest prior work, and limitations, consistent with the broader Scientific Method Kernel novelty direction.

## Suggested first implementation slice

Do not begin with a universal optimizer.

The first shared Rust tranche should remain non-executing and non-authorizing:

- `ExperimentCandidateRefV1`;
- `ExperimentPlanningSnapshotRefV1`;
- `PlanningCoordinateV1` with explicit direction and availability;
- `ExperimentFeasibilityStateV1`;
- `ParetoComparisonProfileV1`;
- `ExperimentDesignFrontierV1`.

It should consume externally produced coordinate values rather than computing EIG, safety, cost, or scientific validity itself.

That gives the kernel a neutral comparison substrate before importing any domain planner.

## Later implementation sequence

After the neutral frontier qualifies:

1. adapt Conjecture Engine variance baseline;
2. adapt #662 coverage-aware discrimination if/when qualified;
3. adapt Spark signature-model EIG as an explicitly approximate EIG implementation;
4. add one SCI-008 uncertainty-aware EIG pilot;
5. add one SCI-007 falsification-power pilot;
6. compare planners under frozen synthetic/historical campaigns;
7. only then consider a richer autonomous experiment-planning policy.

## Anti-shortcuts

A shared SCI-009 implementation should reject or make impossible shortcuts such as:

- universal `best_experiment: bool`;
- universal `experiment_quality: f64`;
- calling prediction variance EIG;
- calling ensemble dispersion calibrated uncertainty;
- treating missing predictions as zero-valued predictions;
- treating unknown cost/risk as zero;
- using current wall clock/live environment as hidden planner input;
- allowing caller-supplied `safe=true` to bypass domain safety admission;
- letting planner output directly execute an experiment;
- post-hoc changing utility weights while retaining prospective authority;
- updating sequential beliefs from simulated outcomes without marking them simulated;
- dropping dominated, failed, infeasible, or rejected candidates from the audit record.

## Relationship to later SCI tranches

SCI-009 should feed rather than replace later layers.

- SCI-010 symbolic-discovery tournament can use SCI-009 to choose discriminating mathematical/data tests among symbolic hypotheses.
- SCI-012 Genesis causal laboratory can use SCI-009 for intervention selection under causal-model uncertainty.
- SCI-013 falsification campaigns can use SCI-009 to prioritize negative controls and boundary attacks.
- SCI-014 Theory Atlas can expose unresolved discriminating experiments and receive their resulting evidence.

## Deliberate non-claims

SCI-009 does not establish that any current planner is calibrated, Bayesian-optimal, globally optimal, safe, ethically admissible, cost accurate, causally identified, or scientifically superior.

It does not authorize any experiment, choose universal utility weights, define universal risk tolerance, or convert expected information into truth probability.

## Review boundary

Review SCI-009 on one question:

Does this contract let Symthaea compare and select experiments intelligently while preserving the exact models, uncertainty, coverage, falsification goals, dependencies, feasibility, resources, risk, preference policy, and adaptive-planning lineage that make an experiment-selection claim scientifically meaningful—without collapsing those dimensions into an unqualified universal score or allowing a planner recommendation to become execution authority?