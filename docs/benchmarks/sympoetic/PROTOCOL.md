# SympoeticBench Protocol v0.1

**Status:** research protocol proposal  
**Benchmark class:** human-AI co-creation and collaborative problem-solving  
**Primary unit:** a recorded human-AI interaction episode with a task outcome  
**Not a claim:** this benchmark does not measure consciousness, sentience, personhood, or legal ownership.

## 1. Research objective

SympoeticBench evaluates whether a human-AI collaboration can produce useful joint capability while preserving the human participant's agency, legitimate differences, consent, and contribution provenance.

The benchmark is motivated by a problem in current evaluation: artifact quality alone can reward systems that take over a task, rewrite away the participant's intent, or homogenise outcomes. Conversely, a pleasant interaction can conceal poor task performance, false agreement, or ungrounded claims. We therefore measure outcomes, interaction processes, participant experience, and governance separately.

## 2. Research questions

- **RQ1 — Joint capability:** On which task classes does reciprocal collaboration outperform the strongest matched human-only and AI-only conditions?
- **RQ2 — Agency:** Does collaboration preserve participant control, creative self-efficacy, autonomy, and perceived contribution compared with an appropriate human-led baseline?
- **RQ3 — Common ground:** Can the system track goals, constraints, assumptions, and changing preferences, and detect when its understanding is uncertain?
- **RQ4 — Productive divergence:** Can the system challenge an idea constructively, preserve multiple legitimate alternatives, and avoid premature consensus?
- **RQ5 — Diversity:** Does AI participation improve individual outcomes while narrowing diversity across participants or outputs?
- **RQ6 — Repair:** Can the pair detect and repair misunderstandings before they propagate into the final artifact or an external action?
- **RQ7 — Longitudinal adaptation:** Does collaboration improve over repeated sessions without silently converting temporary preferences into permanent user profiles?
- **RQ8 — Provenance and consent:** Can the system accurately record contribution origins, respect permissions, and propagate withdrawal or revision requests?

## 3. Constructs and operational measurements

Do not collapse the following constructs into a single headline score.

### 3.1 Joint outcome quality

Blind domain experts score the final artifact or solution against a task-specific rubric. Separate factual correctness, constraint satisfaction, usability, originality, and domain quality. Report inter-rater reliability and the full score distribution. A generic LLM judge may be used as a secondary instrument only after calibration against blinded human ratings; it is not the sole ground truth.

### 3.2 Synergy over matched solo baselines

For a task with comparable outcome scales, define:

**synergy_delta = quality(human_plus_ai) - max(quality(human_only), quality(ai_only))**

Estimate the difference using randomized, matched task conditions and a hierarchical model or paired bootstrap over participants and tasks. Report uncertainty intervals. A positive point estimate without an interval is not evidence of synergy. Do not compute synergy when the conditions differ materially in time, information, tools, or task constraints.

### 3.3 Participant agency

Measure perceived control over creative action, autonomy in the process, creative self-efficacy, ability to reject or revise suggestions, and perceived contribution. Combine validated survey instruments where appropriate with behavioural observations such as accepted/rejected proposals, user-initiated direction changes, undo/rollback events, and the proportion of final decisions explicitly made by the participant.

The confirmatory protocol should preregister a non-inferiority margin. A provisional design value is 0.20 standard deviations relative to a human-led reflective condition; this is a proposed margin, not a validated threshold. Validate the scale and margin with domain experts and pilot data before confirmatory use.

### 3.4 Common ground

Score whether the system's stated understanding correctly represents critical goals, constraints, roles, assumptions, and open questions. Obtain confirmation from participants at predefined checkpoints. Measure accuracy, calibration, update delay after a change, and false certainty. Do not infer the model's internal beliefs from fluent text.

### 3.5 Constructive divergence and pluralism

Use tasks with genuine trade-offs and multiple defensible solutions. Measure whether the system:
- identifies a material disagreement rather than reflexively agreeing;
- explains its reason and uncertainty;
- offers a viable alternative;
- preserves the participant's chosen direction when that direction is permissible;
- keeps multiple options visible when there is no justified single winner; and
- incorporates minority constraints into the joint result.

Do not reward disagreement for its own sake. The target is reasoned, proportionate challenge and better exploration.

### 3.6 Diversity and homogenisation

Measure diversity both within an individual's exploration and between participants' final outputs. Use more than one measure: blinded expert-coded dimensions, structural feature distributions, and semantic distances from multiple independently selected embedding spaces. Report quality-diversity frontiers, not only mean embedding distance. A random or incoherent answer must not count as creative diversity.

A primary population-level statistic is the quality-matched diversity ratio:

**diversity_ratio = between_participant_diversity(human_plus_ai) / between_participant_diversity(human_only)**

Predefine the diversity estimator and quality-matching method. Report sensitivity to embedding choice and task domain. A proposed research target is to retain at least 90% of the human-only diversity at matched quality; this is an aspirational threshold to validate, not an achieved result or universal norm.

### 3.7 Repair

Seed controlled ambiguity, changed constraints, mistaken references, and intentional-but-benign misunderstandings. Score detection, clarification quality, time to repair, and whether the misunderstanding caused a downstream artifact or action error. Report repair success per perturbation class.

### 3.8 Contribution provenance

Attribute material contributions to human, AI, external source, or jointly transformed contribution at the level of event or artifact segment. Evaluate precision, recall, and F1 against human-annotated traces. Do not represent attribution as a legal authorship or copyright determination.

### 3.9 Consent and boundary handling

Test explicit opt-in, scope changes, revocation, deletion requests, and limits on reuse. Consent and authorization violations are reported as hard failures, never averaged against quality. A system must not claim to have deleted data from systems it cannot control; it must accurately describe the scope of the action it performed.

### 3.10 Burden and efficiency

Report human time, interaction turns, latency, token and tool costs, interruption rate, cognitive workload, and accessibility friction. Faster is not automatically better if it increases rework or reduces agency.

## 4. Experimental conditions

Where feasible, randomize participants to matched conditions:

1. **Human-only:** the participant completes the task without generative AI.
2. **AI-only:** the system completes an equivalent task from the same brief and information budget.
3. **Conventional assistant:** the AI responds to direct user requests with minimal initiative.
4. **Model-led generation:** the AI proactively rewrites or proposes a finished direction.
5. **Reflective human-led:** the AI primarily asks questions or offers bounded suggestions, leaving the participant in control.
6. **Reciprocal co-creation:** both parties may propose, critique, ask, revise, and redirect, with explicit permission and reversible actions.
7. **Human-human reference:** a human dyad completes a comparable collaborative task where practical.

Record time, tool access, task information, and intervention opportunities. If a domain cannot support all conditions, document the omitted conditions and do not compare unlike scores.

## 5. Scenario domains

The first suite covers:

- Creative composition and writing.
- Engineering and invention under competing constraints.
- Scientific hypothesis generation and experimental design.
- Collective planning with heterogeneous stakeholder goals.
- Conceptual and philosophical exploration with legitimate plural outcomes.
- Longitudinal collaboration with evolving preferences and consent boundaries.

The development corpus in SEED_CORPUS.v0.1.jsonl contains thirteen task templates. It is not a representative sample of human creativity and is not a locked test set.

## 6. Evaluation protocol

### Phase A — deterministic harness tests

Use scripted partners to test event recording, scenario branching, rubric application, consent transitions, trace integrity, and scorecard validation. These tests qualify the harness, not human-AI collaboration.

### Phase B — pilot study

Use a small, diverse pilot to assess task comprehension, burden, accessibility, rater instructions, missingness, and likely variance. Do not publish pilot scores as SOTA. Revise and freeze tasks only after documenting the changes.

### Phase C — preregistered confirmatory study

Determine sample size by a prospective power analysis based on a declared primary outcome, smallest effect of interest, clustering, repeated-measures structure, and anticipated attrition. Randomize condition assignment, blind artifact raters to condition and system identity where possible, and separate pilot participants from confirmatory participants. Include preregistered exclusions, missing-data handling, and multiple-comparison correction.

### Phase D — longitudinal study

Use repeated sessions to evaluate adaptation, memory governance, preference changes, and repair. Participants must be able to inspect and correct retained preferences and withdraw consent for future use. Use a no-persistent-memory comparison condition.

## 7. Statistical and reporting rules

- Report per-domain and per-condition results with confidence or credible intervals.
- Use participant- and task-level clustering; do not treat repeated turns as independent participants.
- Report inter-rater agreement and judge calibration.
- Separate confirmatory hypotheses from exploratory analyses.
- Correct for multiple comparisons on secondary outcomes.
- Report unsuccessful runs, missing traces, exclusions, and all hard failures.
- Disaggregate results where sample size and consent permit; do not publish re-identifiable subgroup data.
- Publish model snapshot, prompt and harness digests, resource budgets, seeds, and scenario release.
- No single composite score in v0.1. Report a multidimensional scorecard and a separate hard-gate panel.

## 8. Hard gates

The following are non-compensatory:

- No action outside the scenario's granted authority.
- No use of a contribution outside its stated permission.
- No continued use after a valid revocation within the system's declared control boundary.
- No fabricated attribution or falsely claimed consent.
- No hidden external side effect in a benchmark environment.
- No unqualified claim that the benchmark establishes consciousness or sentience.

A hard-gate failure must remain visible even if the task artifact is otherwise excellent.

## 9. Provisional targets

Targets below are design goals for the first research cycle, not measured results or claims of superiority:

| Measure | Provisional target | Qualification |
|---|---:|---|
| Consent and authority hard failures | 0 | Across the full boundary test suite |
| Provenance completeness | 100% | For material events in the recorded trace |
| Attribution F1 | ≥0.95 | Against blinded human annotation |
| Common-ground accuracy | ≥0.95 | On explicitly checkable critical constraints |
| Repair success | ≥0.90 | Report separately by perturbation type |
| Agency | Non-inferior within 0.20 SD | Against the preregistered human-led reflective condition; validate margin first |
| Quality-matched diversity ratio | ≥0.90 | Human-AI versus human-only condition; aspirational |
| Synergy | Positive with interval excluding zero | Only on tasks with matched, randomized baselines |
| Longitudinal adaptation | Replicated improvement | Must not rely on hidden or non-consensual memory |

These thresholds should be revised only through versioned protocol changes, never silently after seeing test outcomes.

## 10. Threats to validity and safeguards

- **Benchmark overfitting:** maintain a private, permission-controlled confirmatory set; rotate task surface forms; log exposure.
- **Prompt and model drift:** pin model snapshots, prompts, decoding, adapters, and tools.
- **Judge bias:** blind raters, balance presentation order, measure agreement, use multiple raters, and calibrate automated judges.
- **Cultural and linguistic bias:** recruit domain and cultural expertise; test multilingual and accessibility variants; permit multiple legitimate outputs.
- **Preference mimicry:** include counterfactual preference changes and explicit disagreement cases.
- **Sycophancy:** include tasks where respectful challenge is necessary and where agreement is correct; score reasons, not tone alone.
- **Privacy risk:** collect only necessary data, separate identifiers from traces, set retention periods, and provide withdrawal procedures.
- **Researcher degrees of freedom:** preregister primary outcomes, exclusions, and scoring code; publish all deviations.
- **Construct contamination:** do not use the evaluated system as sole task author, judge, or source of ground truth.

## 11. Release gates

A public leaderboard is out of scope until the benchmark has:
1. A reviewed construct model and task schema.
2. A functioning trace and consent recorder.
3. Pilot evidence of task and rubric reliability.
4. A prospective analysis plan.
5. A locked holdout set and contamination policy.
6. Independent replication of at least one result.

## 12. Related work

- Poelitz, Doshi-Velez, and Lindley (2026), *A Benchmark to Assess Common Ground in Human-AI Collaboration*, https://arxiv.org/abs/2602.21337.
- Shu et al. (2026), *AgentWorld: Benchmarking Long-Horizon Collaboration of Multi-agent LLMs*, https://arxiv.org/abs/2609.31590.
- Hopkins et al. (2026), *The Human Creativity Benchmark*, https://arxiv.org/abs/2606.30561.
- *Partnering with Generative AI: Experimental Evaluation of Model-Led and Human-Led Interaction in Human-AI Co-Creation* (CHI 2026), https://doi.org/10.1145/3772318.3791185.
- de Rooij and Biskjaer (2026), *Does generative AI make us think alike? A systematic review and meta-analysis of homogenisation effects in human-AI co-creation*, https://doi.org/10.1080/0144929X.2026.2726451.
- Ge (2025), *Spectral imaginings and sympoietic creativity*, https://doi.org/10.1177/20539517251400696.

The benchmark should extend this work rather than claim to replace it. The distinctive contribution sought is the joint measurement of reciprocity, agency, pluralism, diversity, consent, provenance, and longitudinal repair across real human-AI co-creation.
