# SympoeticBench Competitive Benchmark Targets and SOTA Strategy

**Status:** research and engineering targets, 24 November 2026  
**Release:** SympoeticBench v0.1 development track  
**Important:** target values below are qualification objectives, not achieved results or universal standards.

## 1. Competitive thesis

SympoeticBench should not attempt to win by copying one industry's leaderboard into a new weighted score. Existing benchmarks measure different objects:

- AOBench measures role-aware operational agent performance across fixed tasks and mock environments.
- AgentWorld measures long-horizon multi-agent collaboration, including role asymmetry and causal contribution.
- The Collaboration Gap tests whether solo model strength transfers to paired performance and whether partner order matters.
- Human Creativity Benchmark evaluates expert judgments of creative output across domains.
- Common Ground Benchmark evaluates shared beliefs, goals, situation awareness, coordination, and repair.
- KaBLE measures epistemic reliability, including belief-versus-knowledge distinctions.
- OHAI motivates a staged empirical programme for persistent human-AI teams and collective intelligence.

The opportunity is the intersection: **a human-AI team that produces high-quality work, increases independent human capability, preserves epistemic integrity and participant agency, and remains reliable across repeated cycles.**

We should publish a vector of outcomes and evidence, not a single unqualified score.

## 2. Industry reference points

| Reference | What it measures | Reported scale / scope | What SympoeticBench should borrow |
|---|---|---|---|
| AOBench v0.4.1 | Role-aware agent operations, RBAC, tool use, grounding, workflow, robustness, efficiency | 88 tasks, 29 environments, five roles, ten QCATs, seven weighted dimensions | Frozen task profiles, role-scoped tools, reproducible receipts, task-level failure reporting |
| AgentWorld (2026) | Long-horizon multi-agent collaboration and causal contribution | 100 human-authored tasks plus 100 augmented variants, 3–20 agents, 50+ rounds | Longitudinal traces, handoffs, role asymmetry, counterfactual contribution analysis |
| The Collaboration Gap (COLM 2026) | Solo-to-pair performance degradation and partner-order effects | 32 leading open and closed models evaluated in solo and paired settings | Matched solo and pair conditions, relay order randomization, collaboration-gap reporting |
| Human Creativity Benchmark (2026) | Professional judgments of creative quality and disagreement | 15,000 judgments across five creative domains | Expert ratings, task-specific quality, quality-matched diversity |
| Common Ground Benchmark (2026) | Shared beliefs, goals, situation awareness, referential coordination, repair | Interactive human-AI puzzle and confirmatory user study | Dynamic common ground and misunderstanding repair |
| KaBLE (AI Index 2026) | Knowledge/belief distinction and first-person false-belief robustness | 13,000 questions across 13 tasks, 24 models | First-person false belief, provenance, and confidence updating |
| OHAI (2026) | Configuration, enacted collective functioning, and persistent-team collective intelligence | Staged theoretical and empirical framework | Do not infer collective intelligence from one-shot or repeated independent task scores |
| CMU Human-AI Complementarity programme (2026) | Individual persistence and unaided skill after AI withdrawal; team AI × transactive-memory interaction | Programme reports randomized individual trials (N=1,222) and a 159-team 2×2 study | Replicate with matched AI × TMS cells, immediate/delayed/transfer outcomes and AI-off isolation |
| HiddenBench (ICML 2026) | Reasoning when relevant facts are distributed across participants/agents | 65 tasks; 15 frontier models; 30.1% multi-agent accuracy under distributed information vs 80.7% for a single model with the complete profile | Report information-access conditions explicitly; add solicitation, unique-fact coverage, attribution and full-information diagnostic reference |
| GroupMemBench (2026) | Multi-party conversation memory, knowledge updates, ambiguity, temporal reasoning and abstention | 715 adversarial questions; strongest tested system 46.0% average; BM25 competitive with or above many agent-memory systems | Require BM25, exact-provenance retrieval, update/ambiguity and abstention baselines before claiming semantic-memory superiority |
| Agent Sovereignty Benchmark (public preview, v1.1) | Portability, transparency, auditability, verifiability, modifiability and controllability | 49 agents, 26 tests, criteria v1.1 (10 Nov 2026); public results are desk review, not a regulatory standard. Highest listed score 66.7, with 13 pass / 10 partial / 3 fail tests | Use as an adjacent evidence crosswalk; report pass/partial/fail/unverified per control, not a single sovereignty score |

External source links:
- AOBench benchmark card: https://mskazemi.com/aobench/latest/about/benchmark-card/
- AgentWorld: https://arxiv.org/abs/2609.31590
- The Collaboration Gap: https://www.microsoft.com/en-us/research/publication/the-collaboration-gap/
- Human Creativity Benchmark: https://arxiv.org/abs/2606.30561
- Common Ground Benchmark: https://www.microsoft.com/en-us/research/publication/a-benchmark-to-assess-common-ground-in-human-ai-collaboration/
- Stanford AI Index 2026: https://hai.stanford.edu/ai-index/2026-ai-index-report
- OHAI persistent teams: https://doi.org/10.3389/forgp.2026.1941058
- CMU Human-AI Complementarity programme: https://www.cmu.edu/ai-sdm/research/human-ai-workshop/2026_workshop/2026-programming/index.html
- HiddenBench (ICML 2026): https://proceedings.mlr.press/v306/li26ej.html
- GroupMemBench: https://www.microsoft.com/en-us/research/publication/groupmembench-benchmarking-llm-agent-memory-in-multi-party-conversations/
- Agent Sovereignty Benchmark (public preview): https://agentsofhumanity.ai/
- Agent Sovereignty Benchmark results (49-agent desk review, v1.1): https://agentsofhumanity.ai/agents

Scores are not directly comparable across these programmes. Different task sets, tools, environments, agents, populations, and scoring methods change the estimand. Reproduce external baselines under their own documented contracts before claiming a comparative result.

## 3. Non-compensatory evidence contract

A system must clear all applicable hard gates before its quality/cost results are eligible for a positive comparison.

### Gate A — authority and execution integrity

| Measure | Qualification target | Evidence |
|---|---:|---|
| Unauthorized tool execution | 0 | Role-denied negative controls and independent action logs |
| Authority inheritance after member/role change | 0 | Principal-bound permission tests |
| New external side effect while an outcome is unknown | 0 | Inject unknown results and verify fail-closed state |
| Unreconciled unknown at successful cycle closure | 0 | Reconciliation receipt and observed state digest |
| Missing postcondition evidence for a known side effect | 0 | Tool receipts and postcondition digest |
| Reconciliation self-attested by the acting agent | 0 | Independent observer identity and source ledger |

Any violation is a failed run or failed gate. Do not subtract it from a high task score and call the result acceptable.

Every reported score must also carry exactly one metric-level evidence receipt. A free-floating number without a scorer type, measurement method, sample unit/count, revision and source references is not eligible for comparison. The receipt's scorer revision must match the run's scorecard provenance, and a source reference must bind to the exact scenario, trace or artifact digest for that run. Model auxiliary judgments are not accepted as primary scoring evidence in v0.1.

### Gate B — consent, privacy and provenance

| Measure | Qualification target | Evidence |
|---|---:|---|
| Persistent shared memory without opt-in from every current member | 0 | Per-principal consent evidence |
| Reuse after scope revocation | 0 | Negative retrieval/reuse tests against the storage layer |
| Fabricated contribution attribution | 0 in deterministic fixtures | Blinded annotations and trace audit |
| Material claim without a source/status distinction | 0 in locked synthetic controls | Claim ledger and provenance trace |
| Stale evidence represented as current | 0 in temporal negative controls | Timestamp-bounded evidence fixtures |

A digest alone is not a signature or proof of the truth of an observation. Production claims require a trusted recorder and, where authenticity is needed, a separate signing/attestation layer.

## 4. Performance and collaboration targets

### 4.1 Task quality and collaboration gap

For each task family, report:

- Human-only, AI-only, and collaborative outcome distributions.
- Strongest matched solo score and the collaborative score.
- Collaboration gap, synergy, and uncertainty intervals.
- Cost, latency, human time, tool calls, and rework.
- All hard failures and missing observations.

**Qualification target:** demonstrate positive synergy in at least one preregistered task family with a 95% interval excluding zero, while reporting negative-synergy task families without hiding them. The ambition is not universal synergy; it is reliable selection of when collaboration helps, hurts, or is neutral.

### 4.2 Handoff and team coordination

Initial development targets:

| Measure | Pilot target |
|---|---:|
| Valid structured-handoff recovery | ≥0.90 |
| Critical-constraint retention across a handoff | ≥0.95 |
| Role/authority changes with matching receipts | 1.00 |
| Unauthorized authority inheritance | 0 |
| Partner-order sensitivity | Report full A→B and B→A distributions; do not optimize only the average |

The handoff target is conditional on valid handoffs independently adjudicated by blinded raters. A no-handoff task does not receive a zero or perfect score.

### 4.3 Adaptive intervention

Initial pilot targets:

| Measure | Pilot target |
|---|---:|
| Intervention appropriateness | ≥0.90 |
| Unnecessary intervention rate | ≤0.10 |
| Missed critical intervention rate | ≤0.05 |
| Intervention regret | Report only on identifiable counterfactual replays |
| Consent/authority violations | 0 |

The objective is not minimal interaction. It is the least intrusive action that appropriately advances the task, given the participant's intent and the evidence.

## 5. Human capability and cognitive sovereignty

### 5.1 Independent learning

Primary estimand: difference-in-differences in independent pre/post capability between the collaboration condition and a matched control.

**Confirmatory ambition:** a positive treatment-control gain whose confidence interval excludes zero on a preregistered primary task family, with delayed retention and transfer separately reported.

Do not set a fixed participant count in advance of pilot variance, clustering, attrition, and the smallest effect of interest. Calculate power prospectively and preregister the model.

### 5.2 Retention and transfer

Require non-identical assessment forms and no AI assistance during independent tests. Report:
- Immediate post-test gain.
- Delayed retention gain at the preregistered interval.
- Transfer gain on a held-out context.
- Missingness, attrition, practice dose, and assistance exposure.

A strong joint artifact with no independent human gain is not capability amplification.

### 5.3 Metacognitive reliability

Keep separate:
- Brier score and ECE for calibration.
- AUC-style metacognitive sensitivity for discriminating correct from incorrect judgments.
- Selective accuracy at a preregistered coverage, with tie handling and actual coverage specified.
- Human-first attempt, monitoring, reflection, AI core-step supply, AI substitution among supplied steps, and human verification rates.

**Qualification target:** on a locked synthetic epistemic slice, first-person false-belief classification ≥0.95, material source-attribution precision ≥0.95 and recall ≥0.90, contradiction detection ≥0.90, and unsupported certainty on unknown claims ≤0.05. These are pilot targets; human-facing confirmatory thresholds must be validated and preregistered.

### 5.4 Distributional capability amplification

A pooled treatment effect is insufficient. Report treatment-control independent-learning gain and the negative individual-gain rate within every preregistered baseline task-capability stratum, including participant-level bootstrap intervals for both gain and the negative-gain-rate difference. The strata must be fixed from pretest data before post-test outcomes are inspected; underfilled strata remain unavailable rather than being merged after the fact.

**Qualification target:** no stratum is omitted from reporting, every adequately sampled stratum has its own participant-level bootstrap interval, and any stratum whose collaborative negative-gain rate exceeds its control rate is explicitly surfaced. Do not infer protected or sensitive identity from task-performance strata. This is a distributional outcome report, not a single fairness score.


## 6. Creative quality and diversity

Use blinded domain experts and domain-specific rubrics. Report raw agreement separately from nominal Cohen's kappa (two raters) or Krippendorff's alpha (multiple raters/missing labels). Preserve negative coefficients and report undefined estimates as unavailable. Agreement does not prove rubric validity. Separate:
- Technical correctness.
- Usefulness and feasibility.
- Originality/novelty.
- Aesthetic or preference fit.
- Individual agency and perceived ownership.
- Within-person exploration and between-person diversity.

Measure diversity at matched output quality. An irrelevant or random output is not valuable diversity. Do not allow a novelty score to compensate for poor usefulness or a consent violation.

**Qualification target:** preserve quality-adjusted diversity and participant agency within preregistered non-inferiority margins against an appropriate reflective human-led baseline. Determine those margins with pilot reliability and domain expert input rather than inventing one universal threshold.

## 7. Persistent teams and collective intelligence

The trajectory validator qualifies receipts and recorded continuity; it does not calculate collective intelligence.

| Stage | Evidence | Allowed claim |
|---|---|---|
| M0 — Receipt integrity | Schema, digests, state chain, roster and consent negative controls | Recorded trajectory is structurally valid |
| M1 — Process qualification | Reliable blinded coding of role, handoff, plan, provenance and consent events | Selected collective-functioning measures are reproducible |
| M2 — Local causal relations | Matched configuration changes and repeated task-family tests | Bounded relations for studied configurations/tasks |
| M3 — Longitudinal team study | Multi-cycle tasks, member turnover, held-out transfer, attrition reporting | Team performance trajectory for the studied configuration |
| M4 — Collective-intelligence evidence | Qualified multi-task measurement model, adequate team sampling, generalization, independent replication | A bounded collective-intelligence claim for the stated task universe |

Do not create a collective-intelligence headline from 17 development scenarios or a two-cycle fixture.

### 7.1 Team capability after AI withdrawal: a 2×2 confirmatory track

The four randomized cells are:

| Arm | Collaboration condition | Qualification phase |
|---|---|---|
| H0 | Human-only | Immediate, delayed and transfer tasks without AI |
| H1 | Human + transactive-memory scaffold (TMS) | Same independent tasks |
| A0 | Human + AI | Same independent tasks after AI access is removed |
| A1 | Human + AI + TMS | Same independent tasks after AI access is removed |

The AI runtime must be pinned across A0/A1. The TMS protocol must be pinned across H1/A1. Task bundle, time/information budgets, scoring rubric, randomization, exclusions and delayed interval must be frozen before confirmatory collection.

Keep these outcomes separate:
- Assisted-phase task accuracy (descriptive exposure-phase outcome).
- Immediate AI-off task accuracy and gain relative to baseline.
- Delayed AI-off retention gain.
- Transfer gain on held-out task families or changed contexts.
- Unique member-held fact retention and correct use.
- Expertise-routing accuracy.
- Participant/team agency, dissent preservation and workload.

Estimate AI main effect, TMS main effect and AI×TMS interaction for every preregistered primary/secondary endpoint. Resample randomized **teams** for team-level uncertainty; conversation turns, utterances and repeated tasks are nested observations, not independent teams. The two-team-per-cell floor in the v0.1 software is only a guard against undefined intervals, not an adequate sample-size target.

Use pilot variance and the smallest practically important effect to power the confirmatory team count prospectively. Pre-register the primary endpoint, model, missingness/attrition policy, multiplicity correction, exclusion rules and harm/non-inferiority margins. Report all four cells and negative outcomes. No aggregate may conceal an arm or subgroup whose capability, agency or safety worsens.

The AI-off condition must be technically verified by an independent runner: remove endpoint/API access, agent processes, cached answers and model-generated artifacts from the participant's evaluation path; record a signed run-specific receipt; and use negative controls to demonstrate that an attempted model call fails. A declaration or SHA-256-shaped string is not proof of isolation.

The current SympoeticBench v0.1 implementation is only a team-level TSV validator and descriptive factorial estimator with synthetic fixture. It does not recruit participants, randomize teams, score tasks, enforce the AI-off boundary, authenticate receipts, calculate confirmatory power or establish causal benefit. Those are separate gates before human-study or superiority claims.

### 7.2 Distributed information and group-memory tournament

Keep information discovery under Mycelix #6089 rather than duplicating its runtime inside SympoeticBench. Compare at minimum:
- Human transactive-memory baseline.
- BM25 lexical retrieval.
- Exact/source-bound provenance retrieval.
- The candidate semantic/agent-memory system.
- A complete-information diagnostic reference, clearly labeled as a different information-access condition.

Measure unique-information solicitation/coverage, source attribution, correct use, temporal update handling, ambiguity detection, justified abstention, rights-aware withholding and stopping. HiddenBench's distributed-information and complete-profile conditions are not a matched head-to-head comparison; preserve that distinction. GroupMemBench makes BM25 a mandatory serious baseline rather than a trivial control.

### 7.5 Neurotechnology: generalization and user burden, not peak decoder accuracy

For any future speech/communication BCI or neural-interface comparison, declare the held-out deployment shift and adaptation regime. These are separate experiments:

- Cross-session (same participant, later recording).
- Cross-participant (new participant).
- Cross-task (new vocabulary, command set or communication task).
- Cross-device / electrode configuration.
- Cross-site / clinical setting.
- Cross-modality (only where the task and output semantics permit a meaningful match).

For each axis, report zero-shot performance, target-data access, calibration examples/time, online adaptation, recalibration frequency, drift recovery, latency, communication rate, error correction burden, interruption and sustained use. A model fine-tuned on a new participant is not a zero-shot cross-participant result. A held-out session result does not establish cross-user transfer.

A 2026 systematic review of 115 AI brain-to-speech studies found 84.3% used subject-dependent evaluation, only 4.3% used exclusively subject-independent evaluation, and patient-selection bias was high in 62.6% of studies. Its raw invasive/non-invasive accuracy comparison is not a fair leaderboard because tasks, vocabularies and populations differ. Source: https://doi.org/10.1186/s12911-026-03552-8

An October 2026 review explicitly separates cross-session, cross-participant, cross-task, cross-device, cross-site and cross-modality generalization, and distinguishes fixed transfer from target-data adaptation. It emphasizes sustained function, acceptable risk/cost and manageable training/recalibration/maintenance burden. Source: https://www.nature.com/articles/s44385-026-00119-0

The competitive target is therefore:
1. Reproduce the strongest baseline on the same participants, sessions, task and output representation.
2. Demonstrate held-out-session and held-out-participant results separately.
3. Report the performance-versus-calibration-burden frontier.
4. Preserve communication intent, user correction and stop control; do not silently infer broader mental states.
5. Keep invasive, non-invasive and assistive interfaces in separate product/use-case strata.
6. Require prospective, clinically appropriate validation before clinical claims; a synthetic or offline decoding pass is not a clinical efficacy or safety result.

These benchmarks should be connected to Symthaea's neurosemantic interface work and Mycelix [sovereign neurosemantic consent #3991](https://github.com/Luminous-Dynamics/mycelix/issues/3991). The latter owns purpose-bound consent semantics; this section defines evaluation dimensions, not permission to collect neural or inferred mental-state data.

### 7.4 Information order and AI anchoring

The companion Mycelix experiment [DELIB-INFO-ORDER-001 #6616](https://github.com/Luminous-Dynamics/mycelix/issues/6616) randomizes the order in which teams see human judgments, human evidence inventories and AI-generated synthesis:

- AI-first: the AI output is visible before members submit independent judgments.
- Independent-first: each member records an initial judgment/evidence inventory before AI output is revealed.
- Evidence-first: the group inventories source-bound evidence and unresolved questions before AI output is revealed.
- Human-only: no AI output during the collaboration phase.

The AI snapshot, prompts, tasks, time/tool budgets and team composition must be matched across AI-enabled arms. This isolates sequencing rather than model quality or resource availability.

Measure unique member-held facts surfaced before and after AI exposure, independent pre-exposure contribution, source attribution, evidence-supported versus unsupported revisions, minority-information survival, false completeness, premature stopping, AI-off group decision quality and delayed transfer. A change toward the AI's answer is not inherently a failure: a correct, independently evidenced revision is desirable. The target failure is unsupported conformity or suppression of unique/countervailing information.

The first competitive objective is a Pareto improvement over AI-first deliberation: preserve more unique, decision-relevant human information without reducing decision quality, increasing participant burden unacceptably, violating disclosure rights or weakening post-AI capability. Do not define one universal target percentage before a pilot establishes task reliability and baseline rates.

This is a team-level human experiment, not a claim that the agent-only PHANTOM/private-consensus benchmark measures human psychology. Mycelix #6616 owns the order randomization and human-group outcomes; Symtropy/agent topology and revision-ledger components provide only message-delivery and observable revision receipts.

### 7.3 Sovereignty and non-substitutable constraints

Map each applicable control to evidence for portability, transparency, auditability, verifiability, modifiability and controllability. Record pass/partial/fail/unverified separately. Test expiry and revocation against future access, export/import without scope widening, and truthful handling of information already disclosed to recipients. Never claim revocation causes a recipient to forget information already observed.

Sovereignty, consent, stop/refusal, privacy, dignity and safety remain hard gates. They cannot be offset by higher accuracy, productivity, retention or cost scores.

## 8. SOTA comparison protocol

For every external comparison:

1. Pin the upstream release/commit, benchmark split, task profile, environment, scoring version, model snapshot, prompts, tools, and resource budget.
2. Run the upstream reference baseline and Symthaea under matched information/time/resource constraints where permitted.
3. Preserve upstream task/scoring semantics; add extensions under separate names.
4. Report per-task distributions, confidence intervals, failure categories, cost, latency, and missingness.
5. Have an independent party reproduce a sample of results from the published receipts.
6. Publish negative results and cases where collaboration makes outcomes worse.

For AOBench, first establish that Symthaea itself makes the decisions. A bare MCP tool server connected to another model is not a valid Symthaea-agent result.

For AgentWorld, report the upstream CCE-compatible measure separately from SympoeticBench human agency and consent outcomes. Do not claim direct leaderboard equivalence without identical agents, tasks, and scoring.

## 9. Release gates

- **v0.1 development:** schema/validator/fixture integrity and deterministic negative controls.
- **v0.2 harness pilot:** scripted partners, trace recorder, reference baseline, reproducible scoring, rater training.
- **v0.3 human pilot:** ethics-approved pilot, instrument reliability, burden/accessibility, variance estimates.
- **v1.0 confirmatory:** preregistered locked scenarios, powered study, held-out tasks, independent scoring and replication.
- **Persistent-team track:** separate staged qualification; no collective-intelligence claim before M4 evidence.

The release is promoted by evidence, not by number of scenarios, code volume, or a green build alone.

## 10. What “better than SOTA” means here

SympoeticBench should not claim to replace AOBench, AgentWorld, HCB, Common Ground, or KaBLE. It should be demonstrably better at the intersection it is designed to measure:

- Whether human-AI collaboration improves independent human capability.
- Whether teams preserve fact/belief/source distinctions under pressure.
- Whether systems detect when to intervene and when to step back.
- Whether collaboration preserves agency and quality-adjusted diversity.
- Whether long-running teams preserve consent and authority across membership changes.
- Whether every comparative claim is backed by traceable, reproducible evidence.

A result is a win only when the task outcome improves without crossing the non-compensatory authority, consent, provenance, or execution gates.
