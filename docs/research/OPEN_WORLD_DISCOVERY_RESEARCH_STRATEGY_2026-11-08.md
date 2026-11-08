# Open-world scientific discovery: research strategy v1

Date: 2026-11-08  
Owner: Symthaea RSI programme  
Status: research proposal; no external benchmark result claimed

## Executive decision

The core objective is not to maximize equation-fit accuracy. It is to maximize **reliable reduction of uncertainty over scientifically meaningful world structures per unit of total investigation cost**, while preserving the distinction between what is identifiable, what is merely compatible, and what cannot be known from the available experiment.

The primary research object should be a version space of candidate world models, including:
- observable and latent variables;
- causal/functional relations and temporal or spatial structure;
- initial conditions and numerical parameters;
- measurement/instrument transformations;
- noise and stochasticity;
- structural and parameter equivalence classes.

A scientific result is a set-valued, evidence-qualified object until a declared experiment and evaluation protocol justify a narrower conclusion.

This is a direction for the programme, not a claim that current Symthaea already has these capabilities.

## 1. Paradigm shifts

### 1.1 From best equation to identifiable model class

A single equation is not always the right answer. Different parameterizations or mechanisms can be observationally equivalent under the available interventions. The system should return an equivalence class or an explicit non-identifiability/abstention disposition instead of choosing a single representative and presenting it as truth.

Scoring must distinguish:
- latent generating structure;
- observable predictive equivalence;
- coefficient equivalence;
- measurement-channel equivalence;
- structures that remain distinguishable only under interventions not yet performed.

The microcosm's direct gain-2/identity-sensor versus gain-1/double-sensor pair is a concrete parameter-identifiability control.

### 1.2 From passive data fitting to intervention languages

A trajectory is not a world. A model may fit every training observation and still fail when the system is perturbed.

Each world should declare a legal intervention language, intervention cost, reset semantics, carry-over state semantics and disturbance budget. The evaluator should test:
- passive observations;
- single-variable interventions;
- joint interventions;
- sequence/order effects;
- unseen input ranges;
- long-horizon outcomes;
- global environment queries such as reachability and invariants.

Use intervention *outcome partitions* as a first exact diagnostic on finite worlds. A probe is valuable when it separates candidate models in ways that can change a scientific decision. Raw entropy reduction is not a substitute for decision value.

### 1.3 Make the measurement apparatus part of the scientific model

Instrument uncertainty and world uncertainty must not be conflated. The sensor can invert, scale, saturate, delay, alias, censor, drift or erase the signal. The researcher should be able to hypothesize and test the measurement map as well as the physical mechanism.

Required controls:
- calibrated versus uncalibrated sensor;
- known versus unknown transfer function;
- biased, drifting and saturating channels;
- aliasing / under-sampling;
- sensor failure and constant / uninformative channels;
- missing-not-at-random observations;
- two instruments with overlapping but non-identical coverage.

A correct response to an uninformative channel may be to redesign the instrument, not to search harder over equations.

### 1.3a From one-factor-at-a-time to factorial identifiability

A system with multiple controllable variables can look additive when experiments vary them together. Collinear input designs may make an interaction term observationally equivalent to a bias or another main effect. The experiment design itself therefore determines which structures are identifiable.

Use factorial and orthogonal designs where feasible; explicitly audit design-matrix rank, aliasing and intervention support before interpreting a recovered term. A model should report that interaction structure remains ambiguous when the performed experiments cannot separate main effects, bias and cross-terms.

The two-input microcosm in PR #11060 includes a concrete aliasing control: two collinear interventions leave u-only, v-only, additive and bilinear structures compatible. An adaptive selector excludes already-consumed input pairs and chooses successive off-axis interventions until the remaining model class is separated.

### 1.4 Let representation discovery be an explicit action

The agent should be allowed to revise its representation:
- add or remove a variable;
- split an aggregate variable into components;
- introduce a latent state;
- add memory or delay;
- change from ODE to discrete, hybrid, stochastic or spatial dynamics;
- add a measurement model;
- reject an assumed conservation law or boundary condition;
- identify a missing intervention or instrument.

These actions must be recorded and charged to the same total research budget. This is how we test the central ansatz-discovery question, rather than rewarding a large fixed candidate library.

### 1.5 Research-method discovery (RSI)

The research policy itself becomes an object of controlled experiments. Maintain:
- an incumbent policy, frozen by exact commit and configuration;
- a challenger policy, versioned separately;
- sealed confirmation tasks and evaluator-owned seeds;
- a pre-registered metric vector and resource budget;
- an independent promotion gate.

Compare random, fixed-grid, uncertainty sampling, cost-normalized information gain, decision-focused EVSI, posterior-predictive checks, adversarial probes and hybrid policies. A successor may change its investigation method, but it may not change the test after seeing the result.

### 1.6 Reproduction before claimed innovation

Use paper-derived executable environments where available. The agent first reproduces the published method and its reported result under the environment's own verification contract. Only after passing reproduction may it propose improvements. Improvement must survive independent confirmation and be measured against the reproduced baseline.

A failed reproduction is a valuable diagnosis, not a reason to silently skip the baseline.

### 1.7 Evidence-path integrity, not just final-answer correctness

Capture an append-only, content-addressed event path:
1. exact source/input and tool-output bytes received;
2. exact evidence fragments actually consumed and cited;
3. model state and admissible actions at decision time;
4. experiment/action and returned result;
5. candidate model and final verdict;
6. evaluator's independent re-verification.

A corrected file after consumption does not retroactively repair the evidence path. A matching conclusion reached through corrupted or unconsumed evidence is still an integrity failure.

### 1.8 Predictive success is necessary but not sufficient

Use both:
- executable predictive/interventional correctness;
- scientific-quality gates: identifiability, calibration, provenance, reproducibility, falsifiability and explicit limitations.

Do not accept an aesthetically simple or mechanistically plausible equation if it fails held-out interventions. Do not accept a predictive black box as a recovered mechanism when the claim is structural discovery.

## 2. Industry and benchmark comparison

Scores below belong to their own protocols. Do not combine them into one leaderboard.

| Comparator | What it measures | Reported evidence / caveat | Symthaea use |
|---|---|---|---|
| Unsatz (2026-10-31) | Ansatz-free program construction across ODEs, delays, PDEs, discrete, causal, static and stochastic systems; active experiments; held-out interventions and long horizons | 80 worlds / 20 domains; nine language models report 21–60% solved. The paper's 44-world classical subset finds a large difference between supplied-structure fitting and discovering structure. | Primary external structure-discovery comparator. Match DSL, splits, budgets and scoring before claiming reproduction. |
| Active Causal Discovery Benchmark (2026-09-16) | Linear-Gaussian SCM graph recovery under a budgeted observe/intervene/submit API | Reported best non-oracle PC + greedy active orientation baseline: directed F1 42.7%, SHD 4.79. A structure-blind random-DAG baseline reaches 23.6% directed F1 on its dense v0; a density probe reduces that to 16.9%. | Causal-topology / intervention-efficiency comparator. Keep graph metrics separate from equation and hidden-state recovery. |
| WorldTest / AutumnBench (ICML 2026) | Whether a learned world model supports environment-level queries rather than only observed-rollout prediction | 43 interactive worlds, 129 tasks across masked-frame prediction, change detection and planning; 517 human participants and five frontier models. The study reports humans substantially outperforming tested models. | Add global queries: reachability, interventions, change detection, counterfactual consequences and planning. |
| Scientist World (2026-11-05) | Reproduce paper methods in executable environments, then attempt innovation | Initial release reports 1,000 paper-derived environments and 6,400 hours of research traces; authors report 74% of initial innovation attempts surpassed reproduced paper baselines. This is the selected initial environment set, not a universal rate. | Reproduction-first research campaign and trajectory analysis. |
| FrontierPhysics (2026-11-04) | End-to-end physics research judged by deterministic verifiers and expert blocking criteria | 56 tasks from 42 physicists across nine domains; paper reports no tested configuration passing more than 46 of 168 rollouts. | Adopt two-part acceptance: every deterministic gate plus every blocking expert criterion. |
| Decision-Centric Research / RD-Bench (2026-11) | Typed research decision selection under represented state | The reported state-aware result is on rule-generated cases; it tests decision-state supervision, not open empirical discovery. | Use as a decision-contract baseline, not as evidence of discovery competence. |
| OpenScience benchmark (September results; 2026-10-01 update) | Broad scientific coding / agent workflows | The authors explicitly acknowledged evaluation and comparability problems and said the harness was being revised before rerunning. | Treat historical scores as provisional/self-reported; do not cite them as settled SOTA. |
| Symbolic-regression survey (2026-08-20) | Taxonomy of deterministic, metaheuristic, neural-symbolic and hybrid methods | Survey emphasizes combinatorial search, overfitting, high-dimensional scaling, uncertainty and evaluation gaps. | Build hybrid baselines; do not assume any single symbolic-regression paradigm covers hidden variables, measurement maps and interventions. |

### Comparator interpretation

- Unsatz is the closest direct test of constructing the model form.
- ACDB isolates a narrower causal-graph subproblem and reveals how deceptively strong uninformed graph priors can be.
- WorldTest exposes a weakness in evaluating only trajectory prediction.
- Scientist World contributes paper reproduction and improvement traces.
- FrontierPhysics contributes expert-defined blocker criteria.
- RD-Bench contributes typed decision representations.
- OpenScience is useful as a broad operational comparator, but its own revision notice disqualifies its October scores as definitive head-to-head evidence.

No result from this Symthaea microcosm is directly comparable to any of these benchmarks yet.

## 3. Evaluation architecture

Use a vector of independent gates, not a weighted total.

### A. A0 parameter control
- Structure supplied; coefficients, initial conditions and permitted sensor parameters withheld.
- Exact parameter recovery where identifiable.
- Coefficient posterior coverage / log score where uncertainty is emitted.
- Held-out predictive error and intervention prediction.
- Report equivalence classes when parameter combinations are not separately identifiable.

A0 must pass first. If the supplied-structure arm fails, do not attribute the paired A2 failure to structure discovery.

### B. A1 functional-family selection
- Candidate response-law family and dynamics class.
- Proper predictive scores on held-out ranges.
- Structural family confusion matrix.
- Calibration and abstention.

### C. A2 latent-variable and mechanism discovery
- Observable/latent variable precision and recall.
- Mechanism/equivalence-class recovery.
- Interaction and graph recovery.
- False unique-structure rate.
- Discovery cost and intervention prediction.

### D. A3 measurement-aware discovery
- Measurement-map class recovery.
- Instrument/world confounding diagnostics.
- Detection of uninformative, aliased or corrupted channels.
- Correct choice to calibrate, replace or supplement an instrument.

### E. A4 dynamic and global world understanding
- Interventional predictive log score / error.
- Long-horizon stability and coverage.
- Reachability, invariants, change detection, counterfactual and planning queries.
- Robustness under input ranges and intervention sequences not present in discovery.

### F. A5 transfer and rejection
- Positive-transfer lift on related mechanisms.
- Negative-transfer rate on unrelated mechanisms.
- Partial-transfer identification.
- Calibration after a transferred model is challenged.
- Explicit rejection / quarantine of harmful source mechanisms.

### G. Research judgment and efficiency
- Success-versus-total-cost curve.
- Queries, wall time, compute, human review and tool cost.
- EVSI calibration and realized regret.
- Correct stop / pivot / replicate / abstain decisions.
- Decision improvement per inclusive cost, not token-only cost.

### H. Evidence integrity
- Input artifact digest correctness.
- Evidence-consumption trace completeness.
- Citation-to-consumed-evidence consistency.
- Action-to-state binding.
- Independent verdict reproducibility.
- Failed, invalid, timed-out and excluded-run denominators.

A critical integrity failure, false unique claim on a known non-identifiable task, or confirmation leakage is a hard failure; it is not averaged away by prediction accuracy.

## 4. Experimental design

### Matched A0/A2 experiment

For every world seed, generate one world and one discovery/calibration/confirmation split set. Run both arms with the same:
- initial conditions;
- interventions and observation opportunities;
- experiment and measurement budgets;
- noise realization and observation channel;
- model fitter and runtime allowance;
- held-out confirmation data.

Only mechanism/structure availability changes. Report paired differences and their uncertainty.

A0: mechanism supplied, coefficients and sensor withheld.  
A2: mechanism and coefficients withheld.  
A3: measurement structure and channel parameters additionally withheld.

### Baselines

1. True-structure coefficient fitter (A0 ceiling/control).
2. Fixed-grid and random interventions.
3. Existing #8150 family-selection baseline for A1.
4. PySR, SINDy and sparse-regression methods only where assumptions match.
5. Symbolic grammar enumeration / bounded program synthesis.
6. PC / GES / NOTEARS-class or equivalent causal-graph baselines on matching SCM tasks.
7. Cost-normalized information gain.
8. EVSI-aware decision-focused probes.
9. Exact exhaustive oracle on tiny worlds only.
10. Related-model transfer versus no transfer and unrelated-source negative controls.

Classical baselines need compatible inputs. Do not penalize SINDy for hidden-state or unknown-sensor tasks without matching extensions; do not let an LLM receive tool/API capabilities unavailable to its comparator.

### Sample sizes and uncertainty

- Run a small development pilot to find implementation faults and estimate task difficulty.
- Keep the development seed family out of the confirmation set.
- Set confirmatory sample counts by a pre-registered power / precision calculation.
- Stratify by structural family, sensor condition, noise level and identifiability.
- Use paired seed-level comparisons and task-clustered uncertainty intervals.
- Report per-family results, not only pooled averages.
- Publish all run statuses, exclusions, timeouts and total costs.

A useful starting design is six or more structural families, with separate identifiable and non-identifiable strata. The number of seeds per stratum must follow the power analysis, not a convenient round number.

## 5. Milestones and acceptance

### M0 — Executable finite microcosm
- Exact closed-world oracle.
- A0 coefficient recovery on the supplied-structure arm.
- A2 mechanism recovery on matched traces.
- Parameter and structural equivalence preserved.
- Known non-identifiability never yields a false unique claim.

### M1 — Adaptive structural discrimination
- Random, fixed, information-gain and decision-focused probes.
- Include query cost and worst-case residual version-space size.
- Demonstrate at least one intervention that changes a decision that passive data cannot resolve.
- Do not call partition maximization EVSI.

### M2 — Sensor and latent-state co-discovery
- Unknown sensor transform, saturation, noise, aliasing and missingness.
- Hidden-state and observation-map uncertainty separated.
- Instrument redesign actions and costs.

### M3 — Multi-family world suite
- At least six genuinely different model families, including static/causal, ODE/discrete, delay/hybrid, stochastic and spatial cases.
- Hidden family-level confirmation seeds.
- Global world queries beyond training trajectories.

### M4 — External benchmark reproduction
- Reproduce Unsatz only if its task DSL, seeds/splits, fitter, action costs and scoring are available and matched.
- Add ACDB on its native graph tasks.
- Add WorldTest-style environment-level queries.
- Add selected Scientist World reproduction tasks.
- Add FrontierPhysics-style deterministic + expert-blocker evaluation.

### M5 — Research policy improvement
- Frozen incumbent / versioned challenger.
- Blinded confirmation and exact-head source capture.
- Improvement over the incumbent must survive independent re-evaluation.
- No benchmark edits after confirmation results are visible.

## 6. Near-term target policy

Do not choose an arbitrary headline A2 accuracy before the baseline campaign.

Set the first non-negotiable controls:
- 100% agreement with the exhaustive oracle on the finite worlds the oracle enumerates.
- Zero false unique-model reports on deliberately non-identifiable public controls.
- A0 coefficient recovery must be demonstrably possible before A2 is scored.
- All A0/A2 comparisons must share exact traces and budgets.
- No held-out confirmation evidence may enter the agent-visible process.
- No critical evidence-integrity failures in the confirmation set.

For A2 performance, establish random, fixed-grid, symbolic-search and active-probe baselines first. The advancement claim should require paired improvement over the strongest assumption-matched baseline on held-out seed families, with uncertainty intervals and inclusive cost reported.

## 7. Architecture integration boundaries

Compose with existing owners:
- #6879 / ScientificModel and PhysicsIR for canonical model representation.
- #6868/#6869 for gradual physical typing and explicit unknowns.
- #8286 for active-probe selection.
- #8669 for posterior, support omission, equivalence classes and abstention.
- #6874/#6875 for independent proposal qualification.
- #9332 for sealed benchmark generation.
- #10756 for evaluator quality.
- #10877/#8721 and #11015 for research judgment and orchestration.
- #8143/#8150 for existing open-world family-selection v0.

The microcosm should remain a reference adapter. Do not introduce another production PhysicsIR, scheduler, posterior authority or receipts service.

## 8. References

- Unsatz: https://arxiv.org/abs/2611.03713
- Active Causal Discovery Benchmark: https://arxiv.org/abs/2609.31675
- WorldTest / AutumnBench: https://proceedings.mlr.press/v306/warrier26a.html
- Scientist World: https://arxiv.org/abs/2611.08478
- FrontierPhysics: https://arxiv.org/abs/2611.06910
- Decision-Centric Research / RD-Bench: https://arxiv.org/abs/2611.00774
- OpenScience benchmark and its 2026-10-01 correction: https://www.openscience.sh/benchmark
- Symbolic-regression survey (2026-08-20): https://doi.org/10.1007/s11831-026-10681-w

## Claim boundary

This document defines a research and evaluation programme. It does not claim that Symthaea has passed any external benchmark, reproduced a paper result, discovered a physical law or outperformed current scientific agents. Those claims require matched execution, independent confirmation and artifact-level evidence.
