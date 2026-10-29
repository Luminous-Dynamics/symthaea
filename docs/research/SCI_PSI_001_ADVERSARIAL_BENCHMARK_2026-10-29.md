# SCI-PSI-001 — Adversarial, preregistered benchmark for anomalous-information claims

**Status:** research protocol proposal; no human experiment run, no psi result claimed.  
**Issue:** [#8967](https://github.com/Luminous-Dynamics/symthaea/issues/8967)  
**Date:** 2026-10-29

## Executive position

ESP/psi is an empirical question, not a premise of this benchmark. Existing findings are mixed and debated. The work here is to make any claim exposed to a serious, reproducible attempt to falsify it.

**Current answer to “could Chain Theory explain psi?”:** it may motivate a hypothesis, but no derivation currently connects Chain Theory's discrete order, entanglement, or structural grammar to a measurable ESP/psi effect. A plausible-sounding analogy is not a mechanism. Before testing a Chain Theory explanation, its proponents must define an observable, a causal/informational pathway, quantitative predictions, and results that would count against it. Time crystals or quantum nonlocality by themselves do not imply information transfer usable by a person or agent.

This proposal initially benchmarks one narrow claim: **blinded four-alternative forced-choice identification of a hidden target whose identity is fixed before the participant's response**. It is not, by itself, a test of precognition, psychokinesis, mediumship, survival, or every other phenomenon labelled psi. Each requires a separate protocol and endpoint.

## Why a disciplined test is worthwhile

The published literature does not support a one-line consensus summary:

- Tressoldi & Storm's 2024 Stage 2 Registered Report meta-analysis of Ganzfeld studies from 1974–2020 estimated an overall effect around Hedges' *g* = 0.08 (95% interval approximately 0.04–0.12), and reported that several publication-bias tests did not indicate strong bias. This is a reported pooled estimate; it does not itself establish a causal mechanism or substitute for a new independent replication. https://pmc.ncbi.nlm.nih.gov/articles/PMC11134153/
- Wiseman & Watt's 2026 *PeerJ* analysis reported significant supporting outcomes in 16% of preregistered Koestler Parapsychology Unit registry studies versus 46.6% of non-preregistered studies. The paper discusses remaining limits, including sensory leakage and unresolved outcomes. https://pmc.ncbi.nlm.nih.gov/articles/PMC13218336/
- A 2025 metascientific project attempting to replicate Bem's precognition Experiment 1 involved 26,483 participants and 420,472 critical trials across three studies. The original confirmatory effect did not replicate; an opposite-direction exploratory result in Study 2 was not confirmed by Study 3. https://pmc.ncbi.nlm.nih.gov/articles/PMC12588471/
- Acunzo et al.'s June 2026 systematic review of ESP neuroimaging studies examined 143 reports and qualitatively evaluated 129 studies. It identified recurrent issues with small sample sizes, multiple-comparison control, and analytical flexibility; explicit paradigms rarely showed above-chance behavior, while implicit effects were heterogeneous and seldom replicated. https://www.sciencedirect.com/science/article/pii/S1053811926002521
- Tressoldi & Storm's preprint version published 28 October 2026 reviewed preregistered ESP/micro-psychokinesis studies through October 2026. It reports 17 positive experiments out of 60 (28.3%) and 20 positive confirmatory hypotheses out of 97 (20.6%). This is a recent preprint and should be treated as a provisional synthesis, not settled consensus. Its units/inclusion criteria differ from Wiseman & Watt's 2026 registry comparison, so these percentages are not a head-to-head contradiction or directly comparable rates. https://sciety.org/articles/activity/10.31234/osf.io/ha4np_v4

These sources represent different paradigms, analyses, inclusion rules and research perspectives. They justify a high-integrity benchmark—not a presumption either that psi is real or that every report is invalid. Meta-analysis can increase precision for comparable studies but cannot automatically repair shared design flaws, selective reporting, dependence between studies, or ambiguous operational definitions.

## Paradigm shift: treat unusual-information claims as an information-flow security problem

The key engineering shift is to model every possible path from target to response, including the pathways the experimenter did not intend to create.

For each trial, record a directed information-flow graph:

1. target generation and custody;
2. randomization and assignment;
3. room, device, operator, and participant interfaces;
4. network/filesystem/metadata and timing surfaces;
5. response creation and immutable response lock;
6. scoring, feedback, exclusions, and analysis.

The claimed anomalous channel is not assumed to exist. Conventional channels are inventoried and adversarially tested. The experiment should be able to demonstrate that ordinary leakage is detectable when deliberately injected into a synthetic test, while the clean null pipeline remains calibrated.

### Initial paradigm

- Four candidate targets, exactly one correct target; nominal hit-rate chance baseline = 0.25.
- Target identity is drawn and escrowed before the session, while target content remains unavailable to the participant, session operator, scorer, and analyst until their roles no longer need blinding.
- Use a preregistered, independently held target manifest. Commit to the manifest by cryptographic hash before collection; reveal the manifest, salts/seeds where safe, and receipts after the data freeze. A hash commitment demonstrates integrity, not that the original random source was good or that information leakage was impossible.
- Freeze participant instructions, trial count, target-selection and replacement rules, feedback timing, exclusions, analysis code, and stopping rule.
- This first paradigm is a generic hidden-target test. Do not call it a successful Ganzfeld replication unless the sensory conditions and the rest of the registered Ganzfeld protocol actually match a chosen source study.

## Benchmark suite

### Tier 0 — offline validity before human trials

All tests at this tier use synthetic data. The outputs must be deterministic given recorded seeds.

| Fixture | Required behavior |
|---|---|
| Null-only data at 25% chance | Primary analysis must not systematically report an effect; estimate false-positive rate across many seeded runs and report a confidence interval. |
| Known planted effect | Analysis must recover direction and approximate magnitude; report power against the preregistered smallest effect of interest. |
| Injected metadata/timing/ordering leak | Red-team audit or baseline leak detector must flag the predeclared injected channel. |
| Labels permuted after response lock | Significance should disappear at the expected calibrated rate; any leakage invariant to permutations must be explained. |
| Missing rows, duplicates, changed target manifest, altered response lock | Fail closed with a specific integrity error; never silently drop or repair records. |
| Selective stopping / optional peeking simulation | Simulation should expose the inflated error rate of the invalid procedure; the production path must reject it. |
| Synthetic evaluator who can see target labels | Blinding/information-flow check must flag the violation. |
| Repeated participant/session clusters | Analysis must preserve the unit of randomization and account for dependence rather than treating correlated trials as independent observations. |

The null calibration should include enough seeded repetitions to estimate rare error rates meaningfully; a few passing unit tests cannot establish a 0.5% false-positive rate. Predeclare the number of simulations and provide binomial confidence intervals. Tests of an alpha threshold are tests of the analysis procedure, not evidence for or against psi.

### Tier 1 — field experiment controls

- **Positive instrument control:** known information is supplied through the ordinary channel to confirm the target rendering, response capture, and scoring work. Do not mix these trials into the psi endpoint.
- **Sham / no-information control:** preserve session shape but remove target-to-participant information, with target construction and role separation otherwise comparable.
- **Leakage red team:** try filenames, image dimensions, compression artefacts, file order, target reuse, timestamps, network requests, cache state, random-number seeds, operator cues, feedback artefacts, and evaluator/analyst access. The full attack list is preregistered; newly discovered routes are recorded as deviations and addressed in a new version.
- **Scoring control:** automated scoring where possible; otherwise independent raters blind to target identity and condition.
- **Randomization control:** independent review of the target-generation and allocation implementation, including mapping each category to each response position equally often where design permits.
- **Blinded labels:** primary analyst receives arm and target labels only after the analysis pipeline, data exclusions, and checksums are frozen.

### Tier 2 — confirmatory test

1. Publish a Stage 1-style protocol (or an equivalent dated preregistration) before collecting confirmatory data.
2. Set one primary endpoint: trial hit-rate / a preregistered hierarchical model, with participant/session clustering handled correctly.
3. Set the primary inferential framework in advance. If frequentist, use a prespecified type-I error threshold (proposed default: two-sided alpha = 0.005) and report the corresponding interval and the effect magnitude. If Bayesian, specify the prior and decision threshold in advance. A secondary analysis cannot replace a failed primary endpoint.
4. Choose the smallest effect size of scientific interest before the test, based on theory and the practical claim, not the observed sample. Estimate sample size via a simulation that reflects clustering, missingness, multiple layers of randomization, and the exact stopping rule. Aim for at least 90% power at that minimum effect under the planned design. No post hoc adjustment to the effect definition.
5. One independent monitor validates the locked manifest and data hashes; a second party executes the frozen primary analysis without changing the code.
6. Report the effect estimate, uncertainty interval, all registered outcomes, all deviations, every excluded trial and reason, null and sham results, and all code/data manifests.
7. A claimed discovery must survive a separate, independently operated confirmatory replication. The replication must use the locked primary endpoint and decision rule, not a redesigned analysis tailored to the initial result.

**Do not infer sample size from the reported Ganzfeld Hedges' *g* = 0.08 without a justified mapping.** It is not a direct estimate of the absolute increase over 25% hit-rate in this four-choice design. The benchmark must explicitly model the chosen outcome and design.

For scale only, a simple independent-trial binomial approximation with chance hit-rate 0.25, target hit-rate 0.26, two-sided alpha = 0.005, and 90% power gives roughly **31,600 independent trials** before allowing for participant/session clustering, missingness, and protocol exclusions. For a target rate of 0.255, the analogous rough figure is about 126,000 independent trials. These are illustrations of the cost of resolving small effects, not a preregistered sample size or a claim that either effect is real. Final sizing must use exact-design simulation and explicit dependence assumptions.

## Scoring and baselines

A valid report compares the frozen analysis to:

- **Chance baseline:** 25% expected hits under valid uniform four-choice allocation.
- **Conventional-channel baseline:** a classifier/audit restricted to known ordinary side-channel features; it should predict when the target is recoverable through leakage and fail when those features are absent.
- **Sham baseline:** the same protocol without the hypothesized target-information pathway.
- **Permutation baseline:** target labels permuted under the prespecified randomization scheme.
- **Positive instrument baseline:** ordinary disclosed targets to check software/hardware, not to assert psi.
- **Independent replication baseline:** same direction and preregistered decision criteria on a new apparatus/data collection, with effect heterogeneity reported.

Report at least the following:

| Measure | Purpose |
|---|---|
| Hit rate and absolute difference from 0.25 | Human-readable effect magnitude |
| Primary effect estimate and interval | Precision; do not report p-values alone |
| Type-I error calibration on synthetic nulls | Detect a broken/optimistic analysis |
| Power versus a predeclared minimum effect | Tell a null result from an underpowered one |
| Leakage attack detection rate and false alarms | Validate adversarial controls |
| Integrity failures detected / injected mutations | Demonstrate fail-closed provenance |
| All-trial accounting and missingness | Detect selective reporting |
| Independent replication outcome and heterogeneity | Evaluate reproducibility |
| Runtime, compute and researcher time | Cost to reproduce the claim |

The suite must not compress these metrics into a single leaderboard score that allows a large hit-rate to compensate for broken blinding or data integrity.

## Chain Theory / Symthaea testability gate

Before connecting a theory to the outcome, require a frozen theory card:

- **State variables:** define the model state and observable in the experiment.
- **Mechanism:** state the proposed information path and how it differs from known sensory, digital, social, and statistical channels.
- **Mapping:** derive the target-level prediction from model variables, with units and assumptions stated.
- **Quantitative prediction:** specify direction, effect size or probability distribution, conditions, and uncertainty before confirmatory data are revealed.
- **Distinctive prediction:** identify at least one result that separates the theory from chance, leakage-only models, and ordinary inference.
- **Failure condition:** predeclare the outcome that would lower confidence in the proposed mechanism.
- **Independent derivation:** do not choose equations, targets, windows, participants, or variants after inspecting the confirmatory results.

A generic claim about “shared grammar,” “nonlocality,” “entanglement,” “hidden order,” or “emergence” does not pass this gate without an operational derivation. If the theory only fits a post hoc result, mark it exploratory.

## Relation to Chain Theory and time crystals

The prior Chain Theory audit asks whether discrete construction order can be mapped to physical time and thermodynamics; issue [#8964](https://github.com/Luminous-Dynamics/symthaea/issues/8964) proposes a discrete-time-crystal stress test. Those are distinct questions. A recurring observable in a driven many-body system is not evidence that an observer can access hidden targets or anticipate future random bits. Any proposed bridge from time-crystal dynamics or Chain Theory to psi must predict a measurable effect under controlled conditions and must specify why standard quantum information constraints, causal structure, and ordinary channels do not already explain the result.

## Milestones

- [ ] **M0 — protocol:** choose a single primary claim, publish protocol and failure criteria.
- [ ] **M1 — synthetic harness:** generate null, known-effect, leakage, permutation, missing-data, and invalid-stopping fixtures with deterministic seeds.
- [ ] **M2 — analysis lock:** define and independently review the primary model, sample-size simulation, and reporting schema.
- [ ] **M3 — adversarial review:** independent red team tests the target-generation / custody / response / scoring pipeline.
- [ ] **M4 — preregistered field test:** only after M0–M3 pass and ethics/privacy review where applicable.
- [ ] **M5 — independent replication:** independently operated, preregistered confirmation.

## What would count as progress?

Progress includes a null result that is precise enough to exclude effects larger than the preregistered minimum, discovering and eliminating an ordinary information leak, or identifying a protocol flaw that would otherwise create a false positive. A technically invalid or underpowered trial is inconclusive, not evidence for or against psi.

**Success for the current research task is a reliable and auditable benchmark—not a positive psi result. No experiment has been run as part of this document.**
