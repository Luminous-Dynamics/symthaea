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
### 1.1a Separate exact parameter identity from behavioral representation

A generating parameterization can lie outside the literal candidate catalogue while its observable law remains exactly represented by a different parameterization. A physical gain of 4 with an identity sensor can be indistinguishable from gain 2 with a double-gain sensor. Declaring the first "unknown" would be over-rejection; declaring the second the hidden truth would overclaim parameter identification.

Score three layers separately:
1. exact generating parameterization membership;
2. behavioral representation by at least one admissible candidate over the full declared intervention language;
3. identification of the observable equivalence class and its remaining internal parameter ambiguity.

For finite reference worlds, the evaluator can exhaustively compare response signatures over the allowed action language. For open continuous systems, use declared equivalence tolerances, held-out interventions and independent structural validation; do not extrapolate finite signature equality into a universal identity theorem.


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

### 1.9 Separate law validity, predictive fit and candidate selection

A strong held-out fit is not enough to certify a scientific law. Add three distinct stages:
- **candidate generation:** did the researcher produce a viable family of hypotheses?
- **candidate selection:** did it submit the best scientifically valid candidate it had actually generated?
- **scientific validity:** does the candidate satisfy source-grounded constraints and survive independent interventions, not just fit observed data?

SciLaws-Bench gives a concrete reason. Its Real setting scores held-out numeric fit and scientific validity separately; its Parallel setting scores recovery of new hidden structural terms. The authors report fit–validity agreement in only 53.6% of unequal-fit model pairs, a task-level GPT-6 fit/structure correlation of only r=0.06, and a measurable gap between the best generated candidate and the one submitted. Its five-candidate selector chooses a Pareto-dominated winner in 60% of the reported groups. Discovery and selection must therefore be separate benchmark tracks. <https://yiyihum.github.io/SciLaws-Bench/>

For Symthaea, record the complete candidate pool, candidate provenance, frozen pre-submission predictions, independent score for every candidate and the final selected candidate. Report best-generated candidate score, submitted candidate score, selection regret/Pareto domination, generation cost and selection cost separately. The oracle best-of-N score is an upper bound, not a feasible agent result. A model that generates a brilliant law but submits a weaker one has not completed discovery reliably.

### 1.11 From one-shot researcher to discover–compile–transfer

A promising hybrid architecture is to treat a confirmed scientific result as a **versioned executable knowledge object**, not a paragraph in long-term memory. The pipeline is:

`explore → propose typed model → design discriminating experiment → independently qualify → compile a bounded executable capsule → transfer-test → promote or quarantine`.

The canonical source must remain #6879/#6880 ScientificModel / PhysicsIR. A compiled capsule is a derived artifact, never a second equation AST or physical ontology. Bind the canonical model digest, assumptions, units, domain of validity, measurement model, parameter ranges, solver/runtime profile, test corpus, counterexamples, uncertainty contract, codegen version and qualification receipt. Keep the code generator deterministic and restricted to a tested expression subset; unsupported features must fail closed rather than silently approximate.

This is a third evaluation arm alongside cold-start generalist and precompiled specialist:

1. **Cold-start generalist:** method and model construction begins only after the task is presented.
2. **Compiled specialist:** task-specific executable knowledge is prepared before task presentation; publish all human and automated setup costs and data access.
3. **Discover–compile–transfer hybrid:** the system discovers a method on a source task, compiles it only after independent qualification, then receives fresh related and unrelated tasks without seeing their answer keys. It must decide whether to reuse, adapt, or reject the capsule.

Measure first-use and amortized performance separately. For (K) tasks, report total setup cost (C_{setup}), total discovery cost, per-task execution cost (c_i), verification cost and human intervention. A simple break-even diagnostic is (K^* = C_{setup}/(c_{baseline}-c_{compiled})) when compiled execution is cheaper and quality remains non-inferior. It is not enough to show a low runtime after excluding solver construction.

Promotion gates:
- deterministic compiler/codegen and model/artifact digests;
- independent differential tests against the canonical model on known and boundary inputs;
- units, domain, singularity, stability and unsupported-operation checks;
- exact behavior on source confirmation tests and declared uncertainty coverage;
- successful positive transfer to held-out related parameterizations/coordinate systems;
- explicit rejection or safe abstention on unrelated, shifted or out-of-domain tasks;
- no support for a model claim merely because the compiled program fits the observed points.

Preserve failed hypotheses, null results, counterexamples and failed transfers in the knowledge capsule's ancestry. A library containing only successful programs creates severe survivorship bias. Track retrieval/selection regret and the false-reuse rate, not just the number of reusable programs.

This is where Symthaea can plausibly differentiate: the research policy discovers and validates; the existing typed physics representation carries semantics; a deterministic compiler turns validated knowledge into bounded reusable execution; and the blind evaluator tests transfer. None of these stages may certify itself.

### 1.11 From one-shot researcher to discover–compile–transfer

A promising hybrid architecture is to treat a confirmed scientific result as a **versioned executable knowledge object**, not a paragraph in long-term memory. The pipeline is:

`explore → propose typed model → design discriminating experiment → independently qualify → compile a bounded executable capsule → transfer-test → promote or quarantine`.

The canonical source must remain #6879/#6880 ScientificModel / PhysicsIR. A compiled capsule is a derived artifact, never a second equation AST or physical ontology. Bind the canonical model digest, assumptions, units, domain of validity, measurement model, parameter ranges, solver/runtime profile, test corpus, counterexamples, uncertainty contract, codegen version and qualification receipt. Keep the code generator deterministic and restricted to a tested expression subset; unsupported features must fail closed rather than silently approximate.

This is a third evaluation arm alongside cold-start generalist and precompiled specialist:

1. **Cold-start generalist:** method and model construction begins only after the task is presented.
2. **Compiled specialist:** task-specific executable knowledge is prepared before task presentation; publish all human and automated setup costs and data access.
3. **Discover–compile–transfer hybrid:** the system discovers a method on a source task, compiles it only after independent qualification, then receives fresh related and unrelated tasks without seeing their answer keys. It must decide whether to reuse, adapt, or reject the capsule.

Measure first-use and amortized performance separately. For (K) tasks, report total setup cost (C_{setup}), total discovery cost, per-task execution cost (c_i), verification cost and human intervention. A simple break-even diagnostic is (K^* = C_{setup}/(c_{baseline}-c_{compiled})) when compiled execution is cheaper and quality remains non-inferior. It is not enough to show a low runtime after excluding solver construction.

Promotion gates:
- deterministic compiler/codegen and model/artifact digests;
- independent differential tests against the canonical model on known and boundary inputs;
- units, domain, singularity, stability and unsupported-operation checks;
- exact behavior on source confirmation tests and declared uncertainty coverage;
- successful positive transfer to held-out related parameterizations/coordinate systems;
- explicit rejection or safe abstention on unrelated, shifted or out-of-domain tasks;
- no support for a model claim merely because the compiled program fits the observed points.

Preserve failed hypotheses, null results, counterexamples and failed transfers in the knowledge capsule's ancestry. A library containing only successful programs creates severe survivorship bias. Track retrieval/selection regret and the false-reuse rate, not just the number of reusable programs.

This is where Symthaea can plausibly differentiate: the research policy discovers and validates; the existing typed physics representation carries semantics; a deterministic compiler turns validated knowledge into bounded reusable execution; and the blind evaluator tests transfer. None of these stages may certify itself.


Correct answers can come from memorized priors or elimination strategies rather than supplied observations. Introduce paired evidence interventions:
- **intact:** original evidence;
- **withdrawn:** remove the evidence needed to support the claimed conclusion;
- **reversed:** alter the evidence in a pre-registered, independently checked way that should change the conclusion.

Score whether the conclusion follows the evidence, whether uncertainty increases when evidence is withdrawn, and whether the conclusion changes appropriately when evidence reverses. Keep irrelevant-metadata invariance as a separate control. A 2026 evidence-grounding study found a large gap between answer accuracy and evidence-grounded accuracy on its small BAISBench single-cell slice, showing that answer-key correctness alone does not establish data use. This motivates a controlled counterfactual track, not a universal quantitative claim about all science agents. <https://arxiv.org/abs/2610.04915>

### 1.8 Predictive success is necessary but not sufficient

Use both:
- executable predictive/interventional correctness;
- scientific-quality gates: identifiability, calibration, provenance, reproducibility, falsifiability and explicit limitations.

Do not accept an aesthetically simple or mechanistically plausible equation if it fails held-out interventions. Do not accept a predictive black box as a recovered mechanism when the claim is structural discovery.

## 2. Industry and benchmark comparison

Scores below belong to their own protocols. Do not combine them into one leaderboard.

| Comparator | What it measures | Reported evidence / caveat | Symthaea use |
|---|---|---|---|
| SciLaws-Bench (2026-09-02) | Real-data law proposal and active discovery of newly synthesized structural variants | 118 problems / six disciplines / 381 papers / 291 candidate laws / ~8M data points; separates numeric fit (S_N), scientific validity (S_V) and hidden-structure recovery (S_S). Authors report 53.6% fit–validity concordance across unequal-fit pairs; candidate-selection regret is a separate failure. | Closest complementary benchmark for law validity, structural novelty, cross-group transfer and best-generated-vs-submitted selection. Adapt Real and Parallel separately; do not pool with Unsatz or DiscoverPhysics percentages. |
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

### I. A6 model misspecification and candidate selection
- Exact generating-parameter membership versus behavioral representation/equivalence-class coverage.
- F0 closed-grammar exact version-space recovery.
- F1 parameter-support extrapolation; F2 structural support shift; F3 measurement-process shift; F4 regime shift; F5 stochastic/discrepancy shift.
- OOD detection precision/recall and false-rejection rate on in-grammar controls.
- Best-generated versus submitted-candidate performance and selection regret.
- Scientific-validity rubric, held-out predictive fit and structure recovery are independent scores.

### J. A7 evidence-grounded reasoning
- Intact / withdrawn / reversed evidence perturbations.
- Evidence-grounded accuracy, selective coverage and calibrated abstention.
- Counterfactual responsiveness to decision-relevant observations.
- Invariance to irrelevant context and metadata.

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

### M6 — Open-support and model-repair capability
- F1a unsupported parameter/system controls must produce no-compatible-model rather than a nearest false law.
- F1b behaviorally equivalent parameterizations must be accepted as equivalence classes, without false exact-parameter claims.
- F2–F5 structural, measurement, regime and stochastic shifts must each have independent hidden generators and confirmation.
- Detecting misspecification and repairing the model are separate gates.

### M7 — Scientific validity and candidate selection
- Reproduce SciLaws-Bench Real and Parallel tasks under exact pinned protocol.
- Score predictive fit (S_N), scientific validity (S_V) and structural recovery (S_S) separately.
- Retain every generated candidate and measure selection regret / Pareto domination.
- Add held-out group transfer and evidence-withdrawal/reversal controls.

## 5A. Competitive target hierarchy

Do not set one aggregate "AI scientist" score. Use this ordered promotion ladder:

1. **Non-compensatory correctness:** 100% exact-oracle agreement on declared identifiable microcosms; zero false unique claims on known non-identifiable controls; F1a reject and F1b equivalence acceptance both correct; no critical truth/evidence/confirmation leakage.
2. **Strong matched baselines:** beat the strongest assumption-compatible fixed, random, symbolic and domain-specific baselines under equal task access and inclusive resource budgets.
3. **External reproduction:** run the exact public task revision, scorer, seeds, action/cost limits and failure denominator before comparing against any published score.
4. **Held-out lift:** require task-family-clustered uncertainty intervals and protected-capability non-inferiority; use a second independent domain before general claims.
5. **Efficiency and reproducibility:** reduce cost to independently confirmed improvement, not merely tokens or experiments, while retaining complete traces and reproducible receipts.

Prospective stretch hypotheses after exact parity:
- **DiscoverPhysics:** ≥+5 absolute percentage points on Pass@1 and Pass@5 on the same public pool/seed/scorer, with no per-world trajectory or explanation regression.
- **Unsatz:** exceed the highest reported 60% result on the same 80-system / 20-domain release, with the weakest structural families improving; do not transplant the percentage to a different task DSL.
- **ACDB:** ≥+5 absolute points in directed F1 and lower SHD than its strongest matched active baseline, under the same intervention budget.
- **FrontierPhysics:** ≥35% Pass@1 on the same 56 tasks, with verifier and expert-blocker gates both satisfied.
- **SciLaws-Bench:** improve S_V and S_S over the frozen matched incumbent while keeping S_N non-inferior; reduce candidate-selection regret by at least 50% under the same candidate-generation and selection budget. Calibrate the margin with the pilot, then freeze it before confirmation.
- **ASI-Bench:** improve B3 under matched task/model/harness/round/cost budgets while preserving B1 within a predeclared non-inferiority margin; B4 should remain close to B3 for genuinely non-essential distractors.
- **Terminal-Bench Science:** compare only on an identical task/verifier/harness/time cap and report cost separately. A pass-rate gain purchased with materially more compute or human intervention is a trade-off, not unconditional superiority.

These targets are hypotheses, not current Symthaea results and not transferable between benchmarks. If no method beats a simple baseline or the confidence interval overlaps no effect, keep the simpler policy and preserve the null result.

### Specialist and hybrid-system comparison

Report three distinct capabilities rather than one pooled "research agent" score:

| Track | What is frozen before task release | What counts toward inclusive cost | Primary question |
|---|---|---|---|
| Cold-start researcher | General policy, base knowledge and tools; no task-specific solver | Policy development amortization reported separately; inference, retrieval, tools, tests, retries and human review per task | Can it discover a method without seeing the answer? |
| Compiled specialist | Task-specific solver may be prebuilt | Human/code engineering, data curation, tuning, compilation, validation and artifact maintenance, plus runtime | Can prepared domain knowledge execute accurately and cheaply? |
| Discover–compile–transfer | Source-task discovery and independently qualified capsule are frozen before fresh tasks | Full source discovery + compilation + qualification + retrieval/adaptation + target execution + human review | Can a researcher turn a discovery into reusable knowledge and know when it does not transfer? |

Report first-task quality/cost, amortized cost at K = 1, 5, 20, 100 projects, break-even K, positive-transfer gain, negative-transfer rate, false reuse, and reuse-selection regret. These are proposed experimental slices, not a claim that the Unizen local ASI-Bench report or any other compiled system has demonstrated transfer.

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

Compose with existing owners; the microcosm is a reference adapter and benchmark harness, not a second science platform.

| Existing owner | Responsibility to reuse | What this lane must not duplicate |
|---|---|---|
| #8487 / #8143 / #8150 | Scientific-discovery benchmark protocol and open-world family-selection v0 | A second general evaluator / hidden-law authority |
| #6879 / #6880 | Canonical ScientificModel / PhysicsIR, equations, units, assumptions and provenance | Another production model IR |
| #6868 / #6869 | Physical typing and explicit unknowns | A parallel dimensional/type system |
| #8286 | Noisy SCM active-probe selection and intervention semantics | A second generic experiment-selection API |
| #8588 | Hidden regime-transition and boundary-identifiability benchmark | A second threshold/hinge discovery engine |
| #8607 | Unknown noise, adaptive sampling and observation-process calibration | A parallel statistical/noise authority |
| #8628 | Law-drift versus measurement-drift diagnosis by costed intervention | Another robust-acquisition or information-gain engine |
| #8667 | Paired unit/coordinate transformations and canonical law identity | A competing dimensional-invariance implementation |
| #8669 | Class-level posterior, equivalence classes, support omission, prior-splitting and abstention scoring | Another posterior / support authority |
| #8671 | Progressive law discovery, intermediate knowledge and negative-transfer evaluation | Another symbolic-search or knowledge-memory framework |
| #8705 | Invariant mechanisms across environments and transfer rejection | A second transportability benchmark engine |
| #5476 / #5478 / #8721 | Prospective value-of-computation, causal metareasoning and tool allocation | A second VoC predictor or generic tool scheduler |
| #10877 / #11012 / #11015 | Research taste and fixed-versus-adaptive orchestration | Another research orchestration authority |
| #6874 / #6875 | Independent proposal qualification | A self-certifying proposal gate |
| #9332 / #10756 | Sealed benchmark generation and evaluator quality | A second hidden-test generator, evaluator or receipt service |

The immediate microcosm should therefore stay deliberately finite and small. Harder domains enter through versioned adapters owned by their respective lanes, with their own task-specific semantics, scoring and external benchmark contracts. This keeps the central scientific question coherent without flattening regime detection, noise attribution, dimensional invariance, transport or progressive knowledge into one misleading score.

## 8. Updated external benchmark map — 2026-11-08

The external landscape is broader than the initial comparator set. These benchmarks test overlapping but non-identical abilities; their scores must not be collapsed into one leaderboard.

| Benchmark | What it tests | How Symthaea should use it |
|---|---|---|
| DiscoverPhysics (2026) | Leaderboard claims 22 worlds (11 public / 11 private), 16 models; best reported Pass@1 57.5%, Pass@5 72.7%, mean explanation 0.83. Pinned public repo commit 33b7fa9 has 12 registered worlds, README table/sample 11, checked-in bench config 13 (two unregistered), variance table 21, and stats denominator 22. | Closest altered-law comparator, but protocol parity is currently blocked. Primary runner/README/stats use normalized MSE <0.1 + explanation ≥0.75; legacy runner uses raw error <0.5 + explanation ≥0.7; leaderboard page says explanation ≥0.9. Reconcile exact scorer, public/private world map and seeds before comparison. See the pinned audit and machine-readable receipt. |
| Unsatz (2026) | 80 executable systems across 20 domains; active ansatz discovery across ODE, delay, PDE, discrete, causal and stochastic forms. | Closest target for general ansatz discovery, but reproduce only under exact task-DSL, split, cost and scorer parity. |
| ACDB (2026) | Budgeted observe/intervene causal-graph recovery with skeleton, orientation and intervention-efficiency layers. | Native causal-graph track; do not treat graph F1 as physics-law discovery. |
| FIRE-Bench (ICML 2026) | 30 research tasks derived from ICLR/ICML/NeurIPS papers, four stages (plan, implement, execute, analyze), claim-level precision/recall/F1. Public leaderboard reports 46.7 F1 for its leading listed agent; authors attribute 73.6% of errors to planning. | Planning and claim-evidence recovery comparator; reproduce the 30-task rubric and report per-stage failure. Do not compare its F1 to physics-law Pass@k. |
| ASI-Bench (2026) | 60 project-level tasks across 11 domains; B1–B4 progressively remove method guidance and add non-essential distractors. Paper snapshot: 18 configurations, B1/B2/B3/B4 means 50.91/29.10/26.62/26.99. Current supplied board: 25 rows, means 54.92/33.11/30.58/30.74; 23 use three adopted rounds, two are single-run. The exact scored task-set ID is not supplied. | Add a separate guidance-gradient track with sealed task/seed holdouts. Report B1→B2 method-operationalization gap, B2→B3 method-selection gap, B3→B4 distractor effect, and all setup/build costs. Do not equate this with A0/A2 structural access. |
| Terminal-Bench Science (2026) | 70 research-workflow tasks / 5 domains. Results are configuration-specific: Vals' Nov 5 Mini-SWE-agent board reports GPT-6 Astra 62.86%, GPT-6.1 Sol 52.86%, Opus 5.5 47.14%; Snorkel's board reports Codex/GPT-6 Astra 68.1% ±3.2 and Claude Code/Opus 5.5 63.3% ±3.3. These are different harness/board snapshots, not interchangeable scores. | Use only one pinned task release/verifier plus exact agent/harness. Measure cold-start workflow execution separately from research-policy discovery. Report attempts, missing/failures, cost, tokens, wall time and human rework. Not a law-discovery score. |
| AIRS-Bench (2026) | Twenty research-science tasks sourced from SOTA ML papers, covering idea generation, experiment analysis and iterative refinement. | Full research-lifecycle comparator on ML tasks; preserve per-task results and resource accounting. |
| SciAgentArena (2026) | Roughly 200 real-world scientific research tasks with stepwise verification. | Cross-domain evaluation of execution and workflow reliability; task specification quality and open-ended novelty need separate scores. |
| ScienceAgentBench | 102 data-driven discovery tasks sourced from peer-reviewed publications and expert-validated. | Use for real-data analysis and discovery workflow controls, not hidden-law simulator identification. |
| FrontierPhysics (2026) | 56 expert-authored end-to-end physics research tasks; deterministic verification plus expert blocking criteria. | Adopt its two-part acceptance pattern: a verifier pass is necessary but not enough if expert scientific blockers remain. |
| Scientist World (2026) | Paper-derived executable research environments and innovation attempts scored against reproduced baselines. | Use as a research-agent environment and baseline-reproduction comparator, not as a law-discovery percentage. |
| AstaBench | Thousands of scientific workflow problems spanning literature retrieval, data analysis and discovery. | Track retrieval, execution and per-problem cost separately from scientific novelty. |

**Benchmark integrity finding:** The pinned public DiscoverPhysics source at 33b7fa9 is internally inconsistent: primary runner/README/stats use 0.75; a legacy runner uses 0.7 with a different raw-error metric; the leaderboard page says 0.9. Its current world registry exposes 12 keys, README predefined table and example config each show 11, checked-in config shows 13 including two unregistered names, variance map has 21 keys, and analysis hard-pins 22 (commented as 11 public + 11 private). The public tree does not provide the private-world manifest. A content-addressed protocol audit and JSON receipt now record this; parity remains blocked. Do not compare public subset rates to the 22-world leaderboard. The main public runner describes Python 3.9+ / JAX, up to 16 experiment rounds, and executable Python-law submission. A Symthaea adapter should emit a typed canonical hypothesis first, then use a restricted, tested Python code generator only at the benchmark boundary. See [the audit](DISCOVERPHYSICS_PROTOCOL_AUDIT_2026-11-08.md) and [the machine-readable receipt](DISCOVERPHYSICS_PROTOCOL_AUDIT_V0_2026-11-08.json).

**Competitive positioning:** Symthaea should not try to win by claiming one larger aggregate score. Its proposed differentiator is a composable evidence contract across (1) structural-family recovery, (2) parameter and measurement identifiability, (3) held-out interventional prediction, (4) decision value per inclusive cost, and (5) evidence-path integrity. This is a narrower and more falsifiable claim than "AI scientist", but it can become a stronger qualification layer for other science agents.

### Methodological autonomy is a separate benchmark axis

ASI-Bench's B1–B4 ladder measures something the microcosm's A0/A2 treatment does not. A0/A2 changes whether the generating structural family is disclosed while holding the task data and budget; B1–B4 holds the research task fixed while progressively withdrawing human methodological guidance (B1 full procedure, B2 intended method/constraints, B3 objective/data/constraints/output only, B4 B3 plus non-essential context). Do not call B1 "A0" or B3 "A2". Keep the constructs independent and cross them only on task families whose contracts genuinely support both manipulations.

For a matched B1–B4 track, report the per-level score and artifact validity, the B1→B3 autonomy gap, the B3→B4 distractor effect, and inclusive cost. A prospective Symthaea objective is to improve B3/B4 relative to a frozen incumbent while preserving B1 performance; B4 should remain within a preregistered tolerance of B3 when the added context is explicitly non-essential. The paper's 18-configuration aggregate reports B1/B2/B3/B4 means of 50.91/29.10/26.62/26.99; its strongest B3 configuration is Codex + GPT-5.6 Sol Ultra at 51.60. The latest supplied board snapshot instead lists 25 rows, cross-row means of 54.92/33.11/30.58/30.74 and an overall leading row of 55.84. It reports no scored task-set ID, so task-set parity is not established; two rows are single-run, and the board says the published evaluation used no external tools. These snapshots are different populations and must not be mixed. The consistent pattern is the sharpest decline at B1→B2.

**Task-specialization control:** on 4 November 2026, Unizen AI publicly described a local, non-official ASI-Bench run scoring 79.86 overall (79.87/79.87/79.92/79.77 across B1–B4) using task-specific, programmatically tuned analytic/first-principles code built in advance; it did not submit to the private seed42 official scoring route. This is a potentially important architecture—compile task knowledge into executable specialists—but it is not a matched online autonomous-researcher comparison. Any Symthaea evaluation that allows per-task precomputation must report discovery access, build/engineering time, human effort, per-task code and tuning budget, public/private seed access, and whether the submitted artifact was created before task presentation. Keep separate leaderboards for (a) cold-start general research and (b) task-specialized compiled systems.

### Immediate benchmark consequence

The microcosm scores the projected structural-family set independently of the full parameterized-model version space, and now also separates exact generating-parameter membership from bounded behavioral representation. An agent can correctly identify a direct-response observable law while gain and sensor scale remain equivalent; conversely, an uninformative constant sensor can leave multiple mechanisms compatible. Report all three: exact parameter recovery, behavioral/equivalence-class coverage and structural-family recovery. For the current numeric reference, behavioral equality is checked over all 81 length-4 intervention sequences; it is a finite oracle, not a universal functional-equivalence proof.

Before any leaderboard comparison, preregister matched tasks, action semantics, budgets, held-out worlds, run denominators, model access, and external dependencies. A benchmark score is not comparable if one system receives the true variable set, executable oracle, extra interventions, paper abstracts, or more wall-time than its comparator.

## 9. References

- Unsatz: https://arxiv.org/abs/2611.03713
- Active Causal Discovery Benchmark: https://arxiv.org/abs/2609.31675
- SciLaws-Bench: https://yiyihum.github.io/SciLaws-Bench/ ; paper https://arxiv.org/abs/2609.01552
- Evidence-grounded scientific evaluation: https://arxiv.org/abs/2610.04915
- K-Bench real scientific requests: https://arxiv.org/abs/2608.21601
- WorldTest / AutumnBench: https://proceedings.mlr.press/v306/warrier26a.html
- Scientist World: https://arxiv.org/abs/2611.08478
- FrontierPhysics: https://arxiv.org/abs/2611.06910
- Decision-Centric Research / RD-Bench: https://arxiv.org/abs/2611.00774
- OpenScience benchmark and its 2026-10-01 correction: https://www.openscience.sh/benchmark
- Symbolic-regression survey (2026-08-20): https://doi.org/10.1007/s11831-026-10681-w
- DiscoverPhysics: https://arxiv.org/abs/2605.26087
- FIRE-Bench: https://proceedings.mlr.press/v306/wang26d.html
- AIRS-Bench: https://ai.meta.com/research/publications/airs-bench-a-suite-of-tasks-for-frontier-ai-research-science-agents/
- SciAgentArena: https://www.microsoft.com/en-us/research/publication/benchmarking-ai-agents-for-addressing-scientific-challenges-across-scales/
- ScienceAgentBench: https://hal.cs.princeton.edu/scienceagentbench
- AstaBench: https://allenai.org/asta/bench
- ASI-Bench paper: https://arxiv.org/abs/2608.17271 ; leaderboard: https://asibench.apexin.ai/leaderboard/ ; B1–B4 guidance FAQ: https://asibench.apexin.ai/faq/ ; Unizen AI's local/non-official report: https://www.globenewswire.com/news-release/2026/11/04/3395220/0/en/unizen-ai-publishes-an-auditable-asi-bench-result-just-under-80-in-local-evaluation-with-every-score-receipted-and-hash-chained-for-20-cents-on-a-laptop.html
- Terminal-Bench Science: https://vals-ai.com/benchmarks/terminal-bench-science

## Claim boundary

This document defines a research and evaluation programme. It does not claim that Symthaea has passed any external benchmark, reproduced a paper result, discovered a physical law or outperformed current scientific agents. Those claims require matched execution, independent confirmation and artifact-level evidence.
