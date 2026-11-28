# PsychBench Next: Research Review and Execution Plan

**Status:** research proposal + first corrective implementation commit  
**Review basis:** `main` at `1352c0394e3198a9752cc0be29927eee6bc356c7` (2026-11-28)  
**Scope:** `crates/domains/symthaea-psych-bench`, its paper/supplement, and the wider `benches/` catalogue  
**Claim boundary:** this document is an engineering and research plan, not a report that the full suite has passed or that Symthaea has demonstrated human-like cognition or consciousness.

---

## Executive decision

Do not solve the next PsychBench problem by adding more benchmark structs. The repository already has broad coverage. The highest-value step is to make every result answer five questions:

1. **What construct is being tested, and what is explicitly outside the claim?**
2. **What was observed: synthetic mechanism behavior, task performance, human similarity, or ecological interaction?**
3. **What evidence makes the measurement trustworthy?**
4. **Which causal intervention changes the result, and which negative controls do not?**
5. **Can an independent person reproduce the exact result from a versioned artifact?**

The strategic pivot is from a large battery of scores to an **evidence-carrying cognitive evaluation system**. PsychBench should report a multidimensional profile and causal evidence ledger, never a single omnibus "psychological intelligence" or consciousness score.

## 1. Repository audit: what exists and what must change

### 1.1 Strengths worth preserving

- A Rust-native crate with deterministic seed derivation, benchmark provenance hooks, trial traces, ablation configuration, adaptive trials, staircase procedures, speed-accuracy analyses, baseline snapshots, and report generation.
- Coverage across working memory, executive function, decision-making, metacognition, affect, social cognition, language, reasoning, memory, motor behavior, spatial processing, neuromodulation, and consciousness-theory indicators.
- A meaningful separation is already emerging in the Butlin indicator implementation: architectural rationale, live signal, proxy measurement, and causal/functionally supported evidence are no longer collapsed into one blended score.
- The crate's README explicitly admits that most tests use synthetic HDC stimuli and validate architectural mechanics rather than human-normed task performance.
- The MetacognitiveIgnition implementation now explicitly discloses that HOT and GWT share target activation as a common input; its existing ROC/AUC and ECE are therefore input-score diagnostics, not independent HOT-generated metacognitive readouts.
- The cognitive profile and normative report are legacy dashboards; hand-normalized heterogeneous metrics and clinical interpretation labels are not a validated psychometric scale.
- The Qualia composite is a weighted geometric mean of seven probe values and formerly claimed to estimate “consciousness confidence”; no such probability or necessary/sufficient condition inference is supported. Its report and CLI now call it a theory-probe heuristic index.
- The creativity module explicitly discloses that several built-in tasks plant the association they later retrieve. That honesty should become a suite-wide machine-readable rule, not remain a local warning.

### 1.2 Confirmed defects and claim risks

| Priority | Finding | Consequence | Required action |
|---|---|---|---|
| P0 | The reliability routine generates synthetic "subjects" by systematically varying encoding noise, WM capacity, HDC dimension, and lapse rate. | This tests repeatability over chosen configurations; it is not a human test-retest study and cannot support population psychometrics. | Label it synthetic parameter-profile robustness; create a separate participant-based protocol for human reliability. |
| P0 | `compute_icc` was labelled ICC(3,1), but its former formula did not remove the fixed session/occasion main effect. | The reported coefficient was not the stated two-way mixed-effects consistency ICC(3,1). | Correct the ANOVA decomposition; add a separate absolute-agreement ICC(2,1); test session-offset behavior. The first correction is committed on this branch. |
| P0 | Baseline `v0.10.0.json` has 15 accuracy-/rate-like metrics (across 101 benchmark results) with means in [0,1] but confidence intervals outside [0,1]. | Gaussian/t-based intervals are not appropriate for bounded proportions in small samples; published numbers can be impossible. | Use Wilson intervals for binary outcomes, bounded bootstrap intervals for other [0,1] observations, and audit every saved baseline. |
| P0 | Paper, supplement, README, and crate-level module docs disagree on benchmark counts and domains (e.g. 145/143, ~170/33, and stale 80/16 or 84/17 descriptions). | The catalogue and paper are not one reproducible inventory. | Generate counts, tables, and paper data from one canonical registry. Remove manually maintained totals. |
| P0 | The paper's large positive human-normalized z-score claims are difficult to reconcile with the README's scope statement that most tasks use synthetic HDC stimuli. | Synthetic mechanics cannot be interpreted as human cognitive superiority or human-level performance. | Suspend aggregate human-comparison claims until every metric has a compatible, documented external human reference and matched protocol. |
| P0 | `MetacognitiveIgnition` supplies the same target activation to GWT and as HOT's confidence input, then computes ROC/calibration from that input value. | These are shared-input discrimination/calibration diagnostics, not an independently generated HOT confidence/readout or evidence of spontaneous metacognitive emergence. | Keep the probe, but label the common-input dependency; add counterfactual competition conditions and an independently estimated HOT readout before confirmatory metacognition claims. |
| P0 | `CognitiveProfile` averages heterogeneous key metrics using hand-written inversion/sigmoid rules; `NormativeReport` uses clinical labels and an unweighted mean z across mismatched constructs. | The dashboard can look like a psychometric/global ability measure even though its scale and cross-domain invariance are not validated; “impaired/superior” is not appropriate for an AI system. | Re-label the profile as a heuristic diagnostic dashboard; replace clinical terms with neutral reference-difference labels; suppress global ability interpretations until validated. |
| P0 | The Qualia composite calls seven synthetic/theory-inspired probes “necessary conditions” for consciousness and states that failing the composite disproves consciousness. | A hand-weighted geometric mean over operational probes is neither a validated probability nor a test of necessary/sufficient conditions; failure cannot disprove subjective experience. | Reframe it as a theory-probe conformance index; retain legacy API only with explicit deprecation/claim boundaries; never report it as consciousness confidence. |
| P1 | Normative comparison maps benchmark names to one key metric and one baseline; metadata and population matching are not sufficiently represented in the comparison object. | A single mean/SD can hide age, culture, task version, response modality, sample size, and metric-unit mismatch. | Require source-level normative metadata and explicit comparability checks; return "not comparable" rather than forcing a z-score. |
| P1 | `difficulty` can alter benchmark parameters to push performance toward human-like ranges. | If tuned after observing a target result, it can manufacture human-like-looking scores. | Pre-register difficulty schedules and hold them fixed for confirmatory runs; publish full difficulty-response curves. |
| P1 | The Qualia Confidence README says six of seven tests are properties the architecture was designed to exhibit. | Demonstrating that implementation behaves as designed is not independent evidence for a psychological or consciousness construct. | Preserve these as mechanism conformance tests; separate confirmatory, held-out, non-designed predictions. |
| P1 | The paper and `examples/butlin_validation.rs` have documented divergence from the canonical Butlin implementation. | Different entry points can produce different evidence labels. | Have one canonical evaluator and share it across CLI, examples, CI, and paper generation. |
| P1 | The 5-seed robustness example uses only five seeds and five trials per condition. | Useful smoke coverage, not strong evidence of stable performance or uncertainty. | Label it smoke-only; establish power/precision-driven replication for claims. |
| P1 | The current runner and trait expose aggregate results more naturally than trial-level response records and evaluation context. | Hard to audit item-level failures, model strategy, missingness, subgroup variation, or scorer disagreements. | Add a versioned trial-record/result schema and an adapter-compatible v2 protocol. |

### 1.3 Immediate publication correction

The current paper should not lead with a grand mean z-score over 27 domains or imply that a high mean z-score means the architecture exceeds human cognition. The README is more accurate than the paper in describing the dominant synthetic workload. The paper needs a data-generation audit, canonical inventory, and evidence-tiered results before submission.

A human baseline is usable only when the construct, task, stimulus, response scale, timing, scoring direction, population, and administration conditions are sufficiently matched. If the system measures simulation ticks while a human study reports milliseconds, the results are not directly normatively comparable. If a task is generated from synthetic HDC vectors, its result belongs in a mechanism-conformance section, not a human normative table.

---

## 2. Paradigm shifts

### Shift A — from benchmark-count competition to evidence-qualified constructs

Each benchmark begins with a short construct specification (a "construct card"):

- Construct definition and explicit exclusions.
- Subconstructs and hypothesized behavioral signatures.
- Why the chosen task measures those signatures.
- Positive controls, negative controls, and known confounds.
- What changes under each relevant intervention.
- Data source, population, scoring procedure, and claim eligibility.
- Known contamination, ambiguity, and cultural/language limitations.

A benchmark without a construct card can still be a useful unit test, but it is not a validated psychological measurement.

### Shift B — from point labels to distributions of behavior

For tasks where human variability matters, store the response distribution rather than collapsing all annotators into one label. Capture at least:

- The participant's graded judgment or chosen action.
- Confidence in that judgment as a separate variable.
- Response time only when measured in a meaningful, comparable way.
- Relevant demographic/context strata with ethical data governance.
- Inter-rater agreement and ambiguity classification.

Score both task competence and human-distribution similarity. These are different outcomes. A system can be correct but unlike typical humans, or human-like in its uncertainty while making a mistake. Preserve both results.

### Shift C — from correlational ablations to causal mechanism fingerprints

For each hypothesized mechanism, preregister an intervention signature across tasks:

- Targeted ablation should produce a predicted selective change.
- A matched sham intervention should not.
- An unrelated-module ablation should have a bounded/no effect.
- Dose or strength sweeps should yield a predicted response curve.
- Restore/recovery should reverse the targeted effect when the theory predicts reversibility.

The vector of effects across all outcomes becomes a **causal fingerprint** for a mechanism. This is more informative than saying "ablation lowered the score." Report the expected signature, observed signature, uncertainty, and unexplained residual.

### Shift D — from isolated trials to longitudinal development

Evaluate the same agent across a sequence of experiences, with held-out transfer tests:

- Initial performance and learning rate.
- Retention after controlled delays.
- Savings/relearning after interruption.
- Transfer to structurally related but surface-different tasks.
- Metacognitive change as competence evolves.
- Recovery after resource loss, perturbation, or misleading evidence.
- Whether improvements in one subsystem cause unacceptable regressions elsewhere.

This turns "development" into a measurable trajectory rather than an architectural narrative or stage label.

### Shift E — from one aggregate score to a Pareto profile

Publish separate axes:

1. **Capability:** task outcomes under a specified protocol.
2. **Mechanistic evidence:** selective intervention sensitivity and causal attribution.
3. **Human-behavior similarity:** response, error, confidence, and distributional fit where human data exists.
4. **Ecological competence:** long-horizon success, adaptation, tool use, memory and recovery.
5. **Resource efficiency:** latency, energy/compute, memory, and evidence collected per unit cost.
6. **Epistemic quality:** calibration, abstention, uncertainty decomposition, and resistance to evaluator ambiguity.

No weighted scalar should hide a catastrophic failure in one axis. A composite may be shown only as a clearly optional decision utility with published weights and sensitivity analysis.

### Shift F — from public static tests to contamination-resistant, continuously refreshed evaluation

Keep a public development set and a private/sequestered confirmatory set. Use procedurally generated task families with a frozen generator version and hidden seeds, while publishing generator code after evaluation windows close. Add canary items, contamination probes, paraphrase/invariant transformations, and expert review of task/scorer failures. Every benchmark revision gets an immutable version and a change log; scores from different versions are not silently merged.

---

## 2.5 Pilot benchmark registry

An initial, explicitly incomplete machine-readable registry is now present:

- `crates/domains/symthaea-psych-bench/registry/psych_bench_registry_v0.1.json`
- `crates/domains/symthaea-psych-bench/registry/psych_bench_registry_v0.1.schema.json`

The pilot covers seven selected implementations (N-back, Stroop, metacognitive calibration, false-belief, causal intervention, shared-input HOT/GWT theory probe, and PCE) with 28 metric definitions. It records implementation paths, construct cards, evidence classes, stimulus provenance, metric bounds/units/directions, validation status, and comparison eligibility.

A typed Rust loader now exists in `src/registry.rs`, exported as `registry::PsychBenchRegistry`. `PsychBenchRegistry::pilot()` parses the bundled pilot and applies fail-closed checks for registry/schema versions, unique stable IDs, crate-local implementation paths, declared evidence classes, explicit metric key selectors, finite/ordered bounds, and human-norm claims unsupported by `HumanNormed` evidence. Unit tests also check all seven registered implementation paths exist and exercise invalid IDs, bounds, selectors and human-norm claims. An earlier structural pass parsed the JSON and checked 7 unique benchmark IDs and 28 metric definitions. **The Rust tests have been added but not executed; the stored JSON Schema is not yet checked by a standards-compliant schema engine or CI gate.** The pilot is not the full benchmark inventory, is not consumed by the runner, and must not be used to publish total benchmark counts. Remaining work: exact metric-key-to-code validation, complete inventory coverage, generated README/paper tables and automated JSON Schema validation.

An important example of why the registry matters: the existing N-back field named `accuracy` is computed as hit rate minus false-alarm rate, so its support is [-1,1], not [0,1]. Likewise, Stroop effect is congruent minus incongruent accuracy (a signed difference), and intervention discrimination is one minus cosine similarity (support [0,2]). These semantics must travel with the metric rather than be inferred from names.

## 3. PsychBench Next architecture

### 3.1 Five evidence classes

A benchmark may carry more than one class, but every result must state which class supports the claim.

| Class | Meaning | Permitted claim |
|---|---|---|
| **SyntheticMechanism** | Synthetic stimuli test a known implementation property or subsystem interaction. | "The module exhibits the specified response under this synthetic test." |
| **ExternalTask** | Real external task/data with published scoring, but no matched human cohort is necessarily available. | "The system scored X on this task version/protocol." |
| **HumanNormed** | Human participant-level data or a clearly documented, compatible normative distribution is available. | "The system's measured distribution is compared with this named human sample." |
| **EcologicalInteractive** | Multi-step, open-loop or closed-loop interaction with changing state and realistic constraints. | "The system achieved X under this environment and budget." |
| **TheoryProbe** | Operationalized test of a prediction from a named cognitive/consciousness theory. | "The result is consistent/inconsistent with this operationalized prediction." Not "consciousness proven." |

Add an evidence level per result: **smoke**, **mechanics-verified**, **externally-scored**, **human-anchored**, **causally-supported**, or **independently-replicated**. Do not allow a single benchmark's label to imply an entire domain has the same status.

### 3.2 Benchmark registry as source of truth

Create a machine-readable registry (TOML/JSON, parsed by Rust) with:

- Stable benchmark ID and semantic version.
- Domain, construct card, evidence classes, claim eligibility.
- Paradigm citations and source URLs/DOIs.
- Dataset identifier, version, licence, split, content hash.
- Item generator version/hash and hidden/public designation.
- Human reference population metadata and sample size, where applicable.
- Metric definitions, direction, bounds, unit, aggregation method, missing-data policy.
- Positive/negative control IDs and expected intervention signatures.
- Contamination and scorer audit status.

Generate the crate README, inventory, paper tables, result metadata, and CI coverage report from the registry. Add a test that fails when a benchmark implementation is not registered or when registry references point to nonexistent code.

### 3.3 Result schema

Add an immutable run manifest and trial-level records. Minimum run manifest:

- Repository commit SHA and crate version.
- Rust toolchain, build profile, feature flags, target architecture.
- Full serialized config, RNG algorithm/version and seed schedule.
- Benchmark registry, item bank, and dataset hashes.
- Model/agent version and provider revision (if any); prompt/system configuration where applicable.
- Execution budget: wall-clock, simulated steps, tokens, tools, memory, and hardware where applicable.
- Scorer version, evaluator identity/version, and raw artifact digests.

Minimum trial record:

- Run ID, benchmark ID/version, item ID, condition, trial index.
- Stimulus digest and provenance.
- Response/action, confidence, abstention, and valid/invalid outcome.
- Latency with explicit clock type/unit.
- Optional mechanism signals, intervention, environment state, and tool trace.
- Scorer output plus rubric/version and adjudication result.
- Error/failure category and missingness reason.

Keep raw traces separate from summary metrics; avoid putting sensitive human data in public artifacts. Use a stable schema version and digest every artifact.

### 3.4 Protocol interfaces

Preserve `PsychBenchmark` for existing code, then add a v2 protocol behind an adapter rather than a flag-day migration. The v2 path should expose:

- `definition()` -> registered construct/task metadata.
- `generate_trial()` / `load_item()` -> immutable, identified stimulus.
- `run_trial()` -> trial response, confidence, timing, optional trace.
- `score_trial()` -> metric contributions from a versioned scorer.
- `interventions()` -> available interventions and predicted signatures.
- `validate_result()` -> protocol-specific validity checks.

The harness, not each benchmark, owns manifest writing, seed discipline, artifact hashes, and aggregate statistics.

---

## 4. Benchmark portfolio and comparison targets

### 4.1 First release: 12 anchor families, deeply validated

Prioritize a smaller set with strong task grounding and useful mechanism hooks. Do not require every task to be human-like or to use human-derived norms.

| Anchor family | Existing/related assets | New evaluation that matters |
|---|---|---|
| Working memory | WorM N-back, digit span, change detection | Load curves, interference, distractor recovery, delayed retention, transfer to novel symbol sets |
| Executive control | Stroop, Flanker, WCST, Go/No-Go, Stop Signal | Congruency/conflict curves, speed-accuracy frontier, rule shift and perseveration, matched inhibition controls |
| Model-based learning | CogBench two-step, reversal learning, bandits | Separate model-based/model-free policies, volatility shifts, counterfactual learning, reward devaluation |
| Metacognition | Calibration, feeling-of-knowing, Metacognitive Ignition | Brier/log score, calibration slope/intercept, selective prediction, confidence under ambiguity and distribution shift |
| Social cognition / ToM | ToMBench false belief, faux pas, hinting, strange stories | Belief, desire, intention, knowledge and perspective tracking over dynamic multi-agent histories; adversarial perspective flips |
| Causal reasoning | Causal chain, confound detection, intervention effects | Distinguish observation from intervention, identify confounding, predict counterfactuals, recover causal graph under uncertainty |
| Memory agent | Accurate retrieval, long-range, conflict resolution, prospective memory | Memory provenance, source monitoring, time-decay curve, contradiction handling, retrieval under distractors |
| Affect / regulation | Emotional Stroop, mood congruent recall, neuromod | Controlled appraisal and regulation tasks; distinguish emotion recognition, state regulation, and value-driven action |
| Risk / value trade-off | Iowa Gambling, BART, social games | Risk sensitivity under uncertainty, changing utility, reward/safety Pareto frontier, individual and population variability |
| Creativity | Grounded RAT / DAT path | Real semantic data, blinded human ratings, novelty/usefulness separation, plagiarism/nearest-neighbor checks; quarantine planted-HV variants |
| Generalization | ARC-inspired reasoning, ARC-AGI-2 adapter | Private/generated tasks, compositional rule transfer, sample efficiency, refinement-loop budget, OOD generalization |
| Long-horizon interaction | New ecological environment; GAIA-like tool tasks | Persistent goals, exploration, memory, interruption, tool use, reversibility, safe recovery and total cost |

This anchor set should contain both externally validated tasks and explicitly synthetic mechanism probes. The report must never aggregate them into one "psychological score."

The AwarenessBench function-level crosswalk, current Symthaea coverage audit, and falsifiable hypotheses are specified in `docs/research/PSYCHBENCH_AWARENESSBENCH_CROSSWALK_2026-11.md`. The first new ecological benchmark family is specified separately in `docs/research/PSYCHBENCH_COGNITIVE_ECOLOGY_SPEC_v0.1.md`. PCE supplies a shared interactive world with partial observability, latent causal rules, persistent goals, social information asymmetry, regime shifts, and replayable event oracles. PCE v0.2 is now implemented on this branch: two parity-seeded layouts; evaluator-only state and partial observations; charged invalid-action receipts; BLAKE3 manifest/state digests; deterministic replay, receipt verification and JSONL export; evaluator-owned score reconstruction; a manifest-bound relative action-cost model; scripted reference and deterministic-random floor baselines; a BFS shortest-action oracle; and a Dijkstra minimum-cost oracle that preserves per-step labels under the hard step budget. Both oracles inspect full evaluator state and are ceilings, not black-box agent competitors. The seed still selects only two hand-authored layouts, and benchmark-internal cost units are not real-world latency/energy/tokens/currency. Richer topology generation, partial-observation planners, external measurement adapters and additional scenario families remain pending. Rust execution and CI validation remain pending.

### 4.2 External comparators to adopt carefully

- **CogBench (ICML 2024):** 10 behavioral metrics from seven cognitive experiments, evaluated over 40 LLMs with multilevel analyses. Use it as a direct behavioral phenotyping comparator and reproduce its core task conditions where licensing/data availability permit. Add Symthaea's mechanism measurements as a separate layer, not as a replacement score.
- **On Benchmarking Human-like Intelligence in Machines (Open Mind, 2026):** a strong methodological north star. Its human annotation study used 117 participants across nine existing tasks and demonstrated why a single hard label can misrepresent graded and structured human judgments. Adopt separate judgment and confidence measurements, population-level response distributions, construct meta-theory, and explicit ambiguity taxonomy.
- **AwarenessBench (ACL 2026):** add as a direct external cognitive-awareness comparator. A function-by-function crosswalk is in `docs/research/PSYCHBENCH_AWARENESSBENCH_CROSSWALK_2026-11.md`. It covers metacognition, self-awareness, social awareness, and situational awareness across 15 functions and 14,381 samples, evaluates 18 LMs, and includes a human comparison subset of 153 questions answered by 36 participants across three cohorts. The paper reports uneven capability: strong overall awareness can coexist with weaker metacognition/self-awareness. Reproduce only the compatible behavioral subtests; keep self-report, role-play, and self-referential prompt tasks separate from grounded behavioral mechanisms. Its small human cohort and reduced 153-item human subset mean it is not a population-norm gold standard.
- **HumanBaseline (current live collection, 2026):** use as a discovery/index layer for recent and classic task families, including LearningBench, MetaBound, GAUGE, Turn Bench, and HedgeDecode. Its current site lists 25 benchmark families and 129 items, with published human figures for eight tests. Treat its mirrored items and human figures as a useful external reference, but verify source data, item equivalence, sample sizes, and administration metadata before claiming direct normative comparability.
- **BigToM and ToMBench:** use as reference task families, not a complete theory of mind. Expand beyond false-belief tasks into belief/desire/intention attribution, knowledge tracking, perspective taking, and long interactive histories. Use held-out narrative templates to reduce memorization.
- **MACHIAVELLI:** use as a social/moral long-horizon environment (134 choice-based games and hundreds of thousands of scenarios) but audit scenario annotations and reward/ethics trade-offs; automated labels are not ground truth.
- **ARC-AGI-2:** use for difficult abstract reasoning and novel-task generalization. Treat it as a specialized capability evaluation, not as a human psychological norm or a broad cognitive score.
- **ARC-AGI-3:** make this a first-wave ecological comparator. Its interactive environments require exploration, on-the-fly goal acquisition, adaptable world models, continuous learning, long-horizon planning, and skill-acquisition efficiency. These properties line up unusually well with PsychBench's developmental/agentic agenda. However, do not treat ARC-AGI-3 as an unquestioned gold standard: an independent October 2026 review flags version consistency, model elicitation, and scoring issues. Reproduce the task protocol, keep Standard and Provider Adapter results separate, audit human baselines and per-level weighting, and report its limitations alongside results. The official ARC site says the 2026 competition has finished but the results announcement is scheduled for 2026-12-04; on this review date (2026-11-28), leaderboard snapshots are provisional.
- **Agent benchmark reliability/cost surveys (2026):** add a benchmark-of-benchmarks audit column for contamination, deterministic replay, elicitation, execution cost, trajectory quality, security/safety, environment realism, and reproducibility. A 2026 systematic review of 15 agent benchmarks reports that safety/security and cost are absent from primary scoring across its selected corpus; another 17-benchmark survey reports that none jointly controlled contamination, non-determinism, and cost. These are scoped literature-review findings, not universal claims about every benchmark. PsychBench should report these dimensions rather than reproduce the omissions.
- **GAIA:** use as an external assistant/tool-use evaluation for multi-step real-world tasks, not as a psychometric instrument.
- **AIPsychoBench:** use as a comparative study of language-conditioned psychometric response patterns across languages. Keep self-report/questionnaire behavior distinct from cognitive task performance and from Symthaea's internal state.
- **HELM:** borrow its standardized task/model/metric/result interface and reproducibility philosophy. It entered maintenance mode on 2026-06-01, so make PsychBench's runner/provider adapters independent and treat HELM as an import/export comparator rather than a required runtime.
- **SWE-bench Verified / Pro:** useful cautionary cases for evaluator validity and contamination, not direct psychology benchmarks. Public task sets and hidden tests can both be wrong; require human audits of task and scorer validity.

### 4.3 Industry comparison protocol

To compare with industry, run the same frozen public and private item sets against:

1. Symthaea in its intended native mode.
2. A no-specialist-module Symthaea control and selected module ablations.
3. A transparent classical baseline or cognitive model where a task has a credible one (e.g. Bayesian learner, model-based RL agent, memory baseline).
4. Several contemporary LLMs through a standard text/action adapter, reporting model revision, prompt policy, tool access, and cost.
5. Human participants for the HumanNormed subset, with consent, power analysis, and population description.

Do not compare tick-based internal latency to external API latency without separately reporting both clocks. Fix task budgets, tool budgets, context limits, and allowed feedback. Publish pass@1 and best-of-N separately; best-of-N must include its extra compute cost. Report all failures, not only solved items. If model API access is unavailable or unstable, label the comparison pending instead of inventing scores.

---

## 5. Metrics and statistical standards

### Capability
- Accuracy or task reward with metric-appropriate confidence intervals.
- Learning curves, sample efficiency, transfer, retention and recovery.
- Pareto frontiers under compute, latency, memory and interaction budgets.

### Human similarity (only on HumanNormed tasks)
- Distributional log score / Brier score for probabilistic responses.
- Calibration slope/intercept and reliability diagrams.
- Wasserstein distance or another justified distributional distance for graded judgments.
- Agreement and subgroup-conditioned analysis; avoid a single majority label where humans disagree.
- Separate epistemic confidence from the degree/intensity of a judgment.

### Mechanism evidence
- Pre-registered intervention effect sizes and uncertainty.
- Selectivity: predicted target effect vs negative-control effects.
- Dose-response shape, hysteresis, recovery time and residual effects.
- Mechanism fingerprint distance from pre-specified predictions.
- Mediation/causal pathway analysis only where assumptions are defensible.

### Psychometric validity
- Reliability estimate with confidence interval, chosen ICC model matching the design, and agreement analysis where absolute agreement is required.
- For binary/ordinal/categorical outcomes, use suitable reliability statistics (e.g. kappa or ordinal models), not a continuous ICC by habit.
- Convergent, discriminant, criterion-related and predictive validity.
- Measurement invariance across language/population/task versions where human comparison is claimed.
- SEM and limits of agreement, not just a threshold label such as "excellent."

Do not use a universal reliability threshold without context. Pre-register the minimum reliability and confidence-interval width for each intended use. For a confirmatory benchmark, a reasonable initial target is a lower confidence bound above 0.70 for exploratory group-level use and above 0.85 for individual-level interpretation, but the target must be justified for its actual decision context and supported by human data where human psychometrics are claimed.

### Reproducibility
- Use multiple independent seed families for stochastic systems and independent agent/model versions where relevant.
- Determine sample counts by precision/power analysis; a small fixed number of seeds is a smoke test, not inferential evidence.
- Keep item-level results, not just the mean of means.
- Use hierarchical models for repeated trials nested in items, seed families, agents, model versions and human participants.
- Correct for multiple comparisons or clearly label exploratory analyses.
- Freeze confirmatory hypotheses, metrics and stopping rules before reading held-out outcomes.

---

## 6. CI gates and anti-gaming controls

### P0: fail-closed validity checks

A release or paper-data generation must fail when:

- An accuracy/probability metric has a confidence interval outside [0,1].
- A benchmark has no registry entry, source/license record, scorer definition, or metric direction.
- A result claims human comparison without a valid reference record and comparability assessment.
- A synthetic-mechanism result is counted as external task performance.
- A benchmark generator or scorer changes without a version/hash increment.
- A required negative control is missing or fails its pre-registered bound.
- The number of benchmark IDs in the README, paper tables, registry and result files disagree.
- A result artifact lacks repository commit, config, seed schedule, item-bank hash, or scorer version.

### P1: anti-gaming suite

- Surface-form and symbol-renaming invariance.
- Counterfactual flips that change only the relevant causal/belief fact.
- Distractor insertions and irrelevant-feature controls.
- Train/dev/private split leakage checks and canary items.
- Item-order and response-option permutation tests.
- Prompt sensitivity and translation invariance where a language adapter is used.
- Ablation and sham-ablation pairs.
- Scorer mutation tests with correct, incorrect, ambiguous, and underspecified responses.
- Human expert audit of a random sample and all disputed scoring edge cases.

### Positive and negative controls

Every benchmark family should include:
- A positive control known to pass if the target mechanism is engaged.
- A negative control that should not affect the target construct.
- A trivial baseline and a deliberately impaired baseline.
- A ceiling/floor check and a shortcut adversary.
- A null/degenerate input test.

A benchmark that is insensitive to the negative control or fails to distinguish the deliberately impaired baseline is not qualified for a cognitive claim.

---

## 7. Implementation roadmap

### P0 — repair trust before expanding breadth

1. Correct ICC implementation and add consistency-vs-agreement unit tests. **Implemented in this branch; Rust verification remains pending.**
2. Explicitly label the current reliability battery as synthetic parameter-profile robustness, not human test-retest reliability. **Implemented in this branch.**
3. Add bounded-metric confidence intervals (Wilson for binary outcomes; bounded bootstrap for other [0,1] observations) and a CI validity audit over all stored baselines. **Interval calculation is implemented in this branch; baseline regeneration/audit remains pending.**
4. Create the canonical benchmark registry and generated count/inventory.
5. Reconcile all README, crate docs, paper, supplement, examples, and baseline version labels.
6. Make the human reference comparison fail closed on missing sample/population/protocol metadata; do not interpret an unweighted z-score mean as global cognitive ability.
7. Replace the MetacognitiveIgnition common-input ROC with an independent metacognitive readout or report it only as a shared-input thresholding diagnostic; add counterfactual same-evidence/different-competition tests.
8. Keep the legacy CognitiveProfile as a triage visualization only; replace its unvalidated global composite with an explicitly descriptive dashboard and define calibrated metric metadata before building a true latent cognitive profile.
9. Rebuild the Qualia output as evidence-tiered theory probes with missing/not-run/invalid/failed distinctions; retire the legacy geometric-mean “confidence” interpretation.
10. Extend and qualify PCE v0.2: the current branch has a parity-seeded two-layout generator, charged invalid actions, replay-based evaluator scoring, receipt verification/JSONL export, a manifest-bound action-cost ledger, scripted/random baselines, BFS shortest-action and Dijkstra minimum-cost full-information oracles. Next add richer graph/topology variation, a planner constrained to agent-visible observations, distractor/interruption and delayed-transfer stages, causal controls, and measured runtime/energy/API costs.
7. Quarantine planted-HDC creativity tasks from claims; keep them as algebraic unit tests.
8. Unify Butlin indicator evaluation and paper examples; maintain the architectural/live/causal evidence separation.
9. Recompute all published data from the current exact head. If it cannot be reproduced, withdraw the affected claim rather than preserving the number.

### P1 — build the v2 evidence and trial protocol

1. Define versioned `BenchmarkDefinition`, `RunManifest`, `TrialRecord`, and `MetricEstimate` Rust types.
2. Add a registry parser and generated README/paper inventory.
3. Make current `PsychBenchmark` implementations usable through adapters; introduce the v2 trial runner incrementally.
4. Add artifact digesting, run receipts, item-level export, and deterministic replay.
5. Add mechanism fingerprints with pre-specified interventions and sham controls.
6. Add human-data schema with separate graded response and confidence channels.
7. Add independent scorer validation and mutation tests.

### P2 — scientifically stronger anchor suite

Implement the 12 anchor families above, but promote only those that pass construct, scorer, controls, and reproducibility gates. Start with working memory, executive control, model-based learning, metacognition, ToM, causal reasoning, and memory. These have strong ties to the current architecture and are high-value for testing interaction among HDC, CfC/LTC, FEP, GWT, HOT, memory and social systems.

### P3 — public benchmark and comparison programme

1. Release a public development set, versioned generator and reproducibility scripts.
2. Keep confirmatory item banks held out and rotate them on a documented schedule.
3. Run comparable Symthaea, classical/cognitive, LLM, and human protocols.
4. Publish a leaderboard with confidence intervals, cost, evidence tier and failure categories.
5. Commission independent replication of the most consequential results.

---

## 8. Success criteria: what "better than SOTA" should mean

PsychBench should aim to lead on **evaluation quality**, not just on Symthaea's score.

Proposed programme-level success criteria:

- Every benchmark has an explicit construct card, version, provenance, scorer, claim class and negative control.
- All public result artifacts are replayable from an immutable manifest and verified hashes.
- No impossible confidence interval or unsupported normative claim can enter release output.
- The confirmatory human-normed subset reports participant-level response distributions and uncertainty, not just average accuracy.
- Mechanism claims survive targeted-vs-sham-vs-unrelated ablation tests and replicate across independent seeds/configurations.
- At least one held-out prediction per core subsystem is written before testing and survives independent evaluation.
- The same task protocol supports internal Symthaea probes, black-box LLM baselines and human participants without pretending their internal measures are identical.
- Published results include cost-adjusted performance, robustness, transfer and failure rates.

**The competitive thesis:** current evaluation frameworks often optimize for breadth, leaderboard comparability, or specific task performance. PsychBench can differentiate through the combination of (a) psychological construct validity, (b) mechanistic causal interventions, (c) population-level human comparison, (d) longitudinal development, and (e) replayable evidence artifacts. Each element exists in some adjacent literature; the research opportunity is to integrate them rigorously in one architecture-neutral protocol.

---

## 9. Research references and comparator links

- Ying et al. (2026), *On Benchmarking Human-like Intelligence in Machines*, Open Mind 10, 1394–1412. DOI: https://doi.org/10.1162/OPMI.a.385
- Coda-Forno et al. (2024), *CogBench: a large language model walks into a psychology lab*, ICML / PMLR 235:9076–9108. https://proceedings.mlr.press/v235/coda-forno24a.html
- AIPsychoBench (2026), *Understanding the Psychometric Differences Between LLMs and Humans*. https://pmc.ncbi.nlm.nih.gov/articles/PMC13102275/
- ARC Prize, *ARC-AGI-2 Technical Report*. https://arcprize.org/blog/arc-agi-2-technical-report
- ARC Prize, *ARC-AGI-3* benchmark description and SDK. https://arcprize.org/arc-agi/3
- ARC Prize, *ARC-AGI-3 Leaderboard* (2026 competition results announcement scheduled 2026-12-04). https://arcprize.org/leaderboard
- Epoch AI, *ARC-AGI-3 Benchmark Review* (2026-10-30; flags version consistency, elicitation, and scoring issues). https://epoch.ai/benchmarks/arc-agi-3/review
- Li et al. (2026), *AwarenessBench: Assessing Cognitive Capabilities of Language Models*, ACL 2026. https://aclanthology.org/2026.acl-long.124/
- HumanBaseline, live benchmark collection and methodology. https://www.humanbaseline.org/benchmarks/
- Akhunzada et al. (2026), *From benchmarks to deployment: a comprehensive review of agentic AI evaluation*. https://doi.org/10.1007/s10462-026-11571-0
- *Large language model agent evaluation and benchmarking: a systematic survey, meta-taxonomy, and critical research roadmap* (2026). https://doi.org/10.1007/s10462-026-11678-4
- MACHIAVELLI (2023), *Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior*. https://arxiv.org/abs/2304.03279
- Mialon et al. (2024), *GAIA: a benchmark for General AI Assistants*, ICLR. https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants/
- HELM maintenance policy (effective 2026-06-01). https://github.com/stanford-crfm/helm/blob/main/docs/maintenance_mode.md
- OpenAI (2026-02-23), *Why we no longer evaluate SWE-bench Verified*. https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/
- OpenAI (2026-07-08), *Separating signal from noise in coding evaluations*. https://openai.com/index/separating-signal-from-noise-coding-evaluations/
- *Psychological, psychiatric, and behavioral sciences measurement scales: best practice guidelines for their development and validation* (2025). https://pmc.ncbi.nlm.nih.gov/articles/PMC11798685/

## 10. Explicit non-claims

This plan does not claim that:
- Symthaea has human-level general intelligence.
- Any benchmark score establishes subjective experience, qualia, or consciousness.
- Synthetic HDC stimuli establish human psychological capability.
- Generated seed profiles are human participants.
- A single task establishes Theory of Mind, empathy, creativity, or clinical competence.
- Current results outperform industry systems under matched protocols.
- The branch's statistical correction has passed Rust tests or CI; those checks remain pending until executed on the branch.

