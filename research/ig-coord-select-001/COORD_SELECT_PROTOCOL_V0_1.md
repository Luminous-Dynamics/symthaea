# IG-COORD-SELECT-001 — protocol v0.1

**Owner:** [#12786](https://github.com/Luminous-Dynamics/symthaea/issues/12786)  
**State:** research protocol and synthetic structural selector; no evaluated model results.

## Research question

Given task structure, calibrated single-agent capability, architecture-specific outcome estimates, uncertainty and resource constraints, can Symthaea choose the collaboration architecture that improves verified quality without paying unnecessary coordination overhead?

## Core hypothesis

Coordination returns are conditional. They depend on at least:
- baseline single-agent capability on the same task family;
- decomposability and independent subtask structure;
- sequential coupling and shared-state requirements;
- latent information asymmetry and the need for active discovery;
- external tools/feedback and verification requirements;
- group composition, communication overhead, model errors and budget.

Kim et al. (Nature Machine Intelligence, 24 July 2026) controlled prompts, tools and compute budgets across 260 configurations, six benchmarks, five architectures and three model families. They report a task-conditional collaboration advantage, capability-saturation behaviour and error-amplification costs. Their ~45% threshold is an empirical selection rule rather than a universal law. We therefore do not hard-code it.

AgendaFlow (EMNLP Findings 2026) motivates structured meeting protocols; Free-MAD (ACL Findings 2026) motivates consensus-free trajectory selection; Meta-Moderator (EMNLP Findings 2026) motivates a learned moderator. HiddenBench is an information-discovery target; the CSCW Companion hidden-profile study is an integration target.

## Decision objective

For a candidate architecture a:

`U(a) = E[Q_a] - λ·Tokens_a - μ·Latency_a - ν·Risk_a`

where Q is a task-specific independently scored outcome. The weights and hard budgets are declared per application profile and frozen before confirmatory data.

Keep the quality/cost/latency/risk Pareto frontier visible. A weighted utility does not replace hard safety and authority constraints.

## Conservative v0.1 selector

Inputs:
- one calibrated single-agent estimate;
- calibrated estimates for named candidate team architectures;
- per-estimate quality mean/lower/upper bounds, total token cost, wall-clock latency, risk score, calibration-block count, OOD flag and availability;
- task descriptors for decomposability, sequential coupling, latent information asymmetry and distribution shift;
- explicit hard policy thresholds and penalty weights.

The v0.1 deterministic policy:
1. CALIBRATE if task is out of distribution or the single baseline is not adequately calibrated.
2. For each available candidate, CALIBRATE if the candidate is not ready but its upper-bound gain and cost-adjusted upper-bound gain could plausibly exceed the declared minimum.
3. Otherwise compute conservative net gain: candidate lower-bound utility minus single-agent upper-bound utility.
4. SELECT the candidate with the largest positive conservative net gain only if both the minimum quality gain and net utility gain clear the frozen thresholds.
5. Otherwise choose SINGLE.
6. Break exact ties lexicographically by architecture ID for deterministic replay.

Token/latency/risk penalties in v0.1 are transparent fixed-point arithmetic. They are a policy demonstration, not empirically fitted weights. Weights require calibration against the deployment's actual resource accounting.

The selector consumes calibrated estimates; it does not create them. Candidate estimates must be generated from prior randomized calibration experiments, separately from confirmatory task outcomes.

## Mandatory test-time-compute baseline ladder

The 13 November 2026 synthesis *More Agents, More Samples, or More Thinking?* (arXiv:2611.17610) reports that many multi-agent gains disappear when comparisons are matched for sample count or total resource budget. It aggregates nine evaluations / 516 configurations and reports multi-agent accuracy at or below matched baselines in 74% of comparable contrasts. This synthesis is not a universal theorem, but it exposes a confound that this selector must not repeat: a multi-agent architecture should not be credited for additional samples, tokens or verifier compute.

Every task/model block therefore needs:
1. single greedy response;
2. single-agent best-of-N/self-consistency at the same generated sample count;
3. independent parallel samples with no communication;
4. communication with identical evidence allocation;
5. communication with truly partitioned/private evidence;
6. independent tool/program verification at the same cost;
7. cross-model teams where independent error diversity is itself the treatment.

Only the remaining gain after these controls may be attributed to coordination. Record a six-dimensional resource vector at minimum: model calls/samples, input tokens, output tokens, tool/verification calls, wall-clock latency, and monetary cost. Add context-window occupancy and energy/CPU/GPU where observable. Use the same scoring/evaluator policy across the ladder.

## Candidate architectures

Compare:
- single strong agent;
- single agent plus independent verifier;
- independent parallel sampling with outcome-aware selection;
- standard debate / round-robin;
- Free-MAD;
- AgendaFlow;
- Meta-Moderator;
- HiddenBench structured discovery;
- dedicated synthesizer + independent evidence-integration audit;
- Symthaea-selected hybrid.

A named framework is not a reproduction. Pin source revision, model/provider revision, task split, prompt templates, tools, decoding, per-system compute budget and scorer.

## Factors and controls

Measure:
- single-agent baseline accuracy and uncertainty on a calibration split;
- task decomposability and sequential coupling;
- information asymmetry and source diversity;
- independent parallelizable work versus shared mutable state;
- need for source verification and contradiction handling;
- task model competence / capability saturation;
- task-family, model-family and topology shift;
- token, latency, dollar, tool and energy costs.

Positive controls: genuinely complementary agent expertise, unique private evidence, independent parallel subtasks, independent verification that catches baseline errors.

Negative controls: saturated single-agent tasks, sequentially coupled tasks, duplicate work, correlated model copies, no information asymmetry, tool latency bottlenecks, high-cost low-gain teams, and OOD calibration.

## Matched campaign

1. Freeze task-family and model-family calibration splits.
2. Measure single-agent baselines per task family and provider/model revision.
3. Run candidate architecture arms under paired, randomized task/seed blocks and matched total budgets.
4. Fit quality and cost estimates only on the calibration split.
5. Freeze the policy and weights.
6. Evaluate the selector on confirmatory task/model blocks without access to evaluator answer keys.
7. Compute oracle architecture and regret only after all runs finish; this evaluator-only oracle must never enter the selector.
8. Report held-out model-family and task-family transfer.

Count every agent, orchestrator, moderator, synthesizer, judge, verifier, tool call and retry. Do not use "same number of agents" as a substitute for matched total compute.

## Metrics

**Selection:** oracle-architecture agreement, regret versus per-task oracle, regret versus best fixed architecture, correct SINGLE routing, CALIBRATE precision/recall, predicted-gain calibration, architecture selection by task family.

**Outcome:** verified success, accuracy, evidence discovery/integration, task completion, abstention quality, tool/environment correctness, false completion and error amplification.

**Economics:** all tokens/calls/tool costs, p50/p95 end-to-end and critical-path latency, energy where measurable, cost per correct verified output, incremental quality per resource.

**Robustness:** held-out model/task families, topology perturbations, shifted calibration, missing data and adversarial inputs.

## Proposed targets — internal gates, not published standards

- at least 20% lower quality/cost regret than the best fixed architecture on the confirmatory portfolio;
- selected result within five percentage points of per-task best on at least 90% of confirmatory cases;
- at most 5% unnecessary multi-agent routing when every tested team is demonstrably worse than single-agent under matched budgets;
- positive net gain versus strongest fixed baseline with a paired task-level 95% interval excluding zero;
- held-out model-family and two held-out task-family transfer;
- calibrated abstention to CALIBRATE on materially uncertain/OOD cases.

These targets must be frozen before the confirmatory split. The 45% threshold from the Nature paper is not one of our fixed policy constants.

## Claim boundary

Synthetic fixture passes validate selector arithmetic and disposition semantics only. Model-level superiority requires complete matched-budget experiments and independent scoring. The results cannot establish universal intelligence, human-group validity, consciousness, legitimacy or authority.
