# Open Mobility — Preference Drift and Held-Out Task Transfer v0.1

**Status:** research benchmark protocol draft; fixture only  
**Tracking issue:** [MOBILITY-ASSIST-011 / #7921](https://github.com/Luminous-Dynamics/symthaea/issues/7921)  
**Upstream comparison:** [interval-aware architecture robustness / PR #7892](https://github.com/Luminous-Dynamics/symthaea/pull/7892)  
**Decision-sensitive evidence gathering:** [PR #7907](https://github.com/Luminous-Dynamics/symthaea/pull/7907)  
**Fixture:** [preference-drift-scenarios-v1.json](./fixtures/preference-drift-scenarios-v1.json)

## 1. Research question

Can a mobility-assistance decision policy preserve engineering feasibility and multi-objective trade-offs while handling explicitly reported preferences that change with task, context, fatigue, environment, or time—and avoid pretending that yesterday's preference answers today's question?

The paradigm shift is **fixed personalization → revisable, context-scoped preference evidence**. A system should not have to choose between imposing a permanent scalar utility and returning an unhelpful Pareto set. It can retain the feasible non-dominated set, use only applicable explicit preference evidence, ask a bounded question when the decision is sensitive to missing information, or abstain when evidence is insufficient.

This is a simulation/replay benchmark. It is not a new preference model, Bayesian optimizer, expected-information-gain engine, experiment scheduler, hardware controller, or authorization path.

## 2. What this benchmark keeps separate

There are at least three evidence channels; they must not be collapsed into one score without an explicit, reviewable model:

1. **Engineering evidence:** task completion, wearer-borne mass, energy, movement restriction, stability, feasibility, and uncertainty bounds. Candidate architecture comparison and hard feasibility gates remain authoritative for engineering admissibility.
2. **Preference evidence:** explicit ordinal feedback such as “prefer A to B,” together with task, context, observation time/window, and provenance. A task-success metric or inferred latent state is not a substitute for a reported preference.
3. **Policy output:** a proposed option, a non-dominated option set, a query, or abstention. A model's predicted preference is not a wearer's observed preference.

A preference score must never re-admit a candidate rejected by a hard feasibility/safety gate. A Pareto set must not be described as a preference ranking. Contradictory reports and missing evidence remain visible rather than being averaged into fabricated certainty.

## 3. Fixture contract and limitations

The versioned JSON fixture contains three synthetic candidate profiles and 12 hand-authored scenario records: two development examples and ten held-out negative/shift controls. It covers steady-context explicit feedback, fatigue-related preference reversal, transfer from level walking to sit-to-stand, an environment transition, a hard feasibility gate, no preference evidence, contradictory feedback, rejection of all eligible options, exhausted query budget, noisy feedback, and explicit preference that disagrees with an engineering metric proxy.

All candidate metrics, preference records, contexts, and evaluation oracles in this file are synthetic and illustrative. They are not measured biomechanics, human-subject data, inferred preferences, or performance results. The small fixture is a contract/smoke corpus, **not a statistically powered benchmark**. The evaluation oracle is evaluation-only: it must never be passed into a policy, query selector, calibration stage, feature builder, or model-fitting call.

Before making generalization claims, grow the corpus with a versioned generator and manifest. Split by complete synthetic wearer/session and scenario-family/context shift before tuning policy parameters; report each held-out family. Never random-split adjacent records from the same trajectory across train/test.

## 4. Candidate and query protocol

For each episode, freeze before evaluation:

- task and context identity (including fatigue/environment/time window where applicable);
- candidate set and per-context hard-feasibility results;
- engineering metric definitions, directions, units, uncertainty, and source status;
- available preference observations, each tied to a task/context/time window and explicitly tagged as simulated reported feedback;
- query and compute budget;
- the evaluation-only oracle and scenario split.

A pairwise answer establishes only the stated ordinal relation in its declared context/window. Each episode supplies an explicit `current_context_id`; observations apply only when both `task_id` and `context_id` exactly match the episode. Historical observations remain visible as stale/mismatched diagnostics and cannot resolve a current-context choice. The implemented replay baseline recommends only when one eligible candidate has explicit pairwise support against every other eligible candidate; it preserves a current-context reject-all answer; otherwise it queries if budget remains and abstains when it does not. It never reads the evaluation-only oracle. The fatigue-shift negative control includes a complete ranking from the rested context, so a task-only/context-blind policy can choose stale `flex` while exact-context replay must ask again. The batch runner computes the context-aware and context-blind outputs first, then scores both against evaluation-only acceptable-option sets. In the hand-authored fixture, the context-aware baseline makes four recommendations and all four are within the declared acceptable set; the context-blind ablation makes five recommendations, including one stale-context recommendation. These are synthetic software-fixture counts, not empirical accuracy estimates. Do not infer cardinal utility, transitivity, stable global ranking, or a calibrated probability from one or several pairwise records. Contradictions must be detected; noisy feedback must be reported and confidence calibrated only from separate validation data.

For the frozen engineering-weight baseline and seeded-random baseline, use the same eligible-candidate lists and evaluate each selected option only after choices are frozen. A fixed seed is recorded in the receipt. On the current 12-scenario fixture with seed 20261018, a structural mirror produces 8 acceptable selections from 9 ranked scenarios for the fixed-weight baseline and 6 of 9 for seeded random; both also select in the three scenarios where the oracle supplies no ranking, including the explicit reject-all case. These are fixture-specific diagnostic counts, not estimates of generalization or human preference accuracy. A 100-seed sweep (seeds 1000–1099) on this same fixed fixture yields 550 acceptable and 350 unacceptable random selections over 900 ranked scenario-seed opportunities; acceptable selections range from 2 to 9 per seed. It also emits a selection in each of the 300 unranked scenario-seed cases, including 100 reject-all cases. This measures sensitivity to the random baseline's seed, not generalization across independent scenarios.

When context shifts, report how much evidence transfers and why. Do not automatically carry a preference learned for level walking over to slopes or sit-to-stand. The protocol may transfer a prior as a *prior* only if the policy labels the shift, accounts for uncertainty, and does not call the result current explicit feedback.

## 5. Required baselines

Compare on identical scenarios, candidate feasibility, initial observations, and budgets:

1. **Fixed-weight scalar objective:** weights and metric normalization preregistered on development data; never tune weights after seeing held-out results.
2. **Single-objective selection:** show each objective's result or a preregistered rotation, rather than selecting the best-performing objective post hoc.
3. **Pareto + explicit query/abstention:** preserve all feasible candidates not robustly dominated; use context-matched preference observations and query or abstain when evidence cannot resolve a context-specific choice.
4. **Seeded random:** reproducible random eligible candidate/query baseline.
4. **Seeded random eligible selection:** reproducible, seed-controlled selection from the declared eligible set, with no preference or metric ranking.
5. **Context-blind preference/Pareto:** an ablation that receives the same observations but ignores context, task, or time matching.
6. **Frozen fixed-weight engineering baseline:** task success (0.40), stability margin (0.20), normalized wearer-borne mass (0.15), normalized energy per task (0.10), and movement freedom (0.15). Weights and fixture scales are constants, sum to 1.0, and must not be fitted to evaluation-oracle labels. This baseline is intentionally an engineering-metric proxy, not a model of subjective preference.

A context-aware preference method is an additional baseline only after its interface, provenance, calibration, and training/test isolation are qualified. Do not implement another generic preference model or active-learning engine just to run this domain benchmark; adapt the existing qualified interfaces.

## 6. Primary metrics

Report raw numerators/denominators, uncertainty intervals, excluded episodes, and per-family/per-wearer-profile slices, not only a pooled mean.

| Metric | Definition / guardrail |
|---|---|
| Context-specific pairwise accuracy | Correct held-out ordinal relations; report context-matched and context-mismatched observations separately |
| Preference regret / top-choice agreement | Evaluate only where the evaluation oracle explicitly supplies a ranking; do not manufacture a ranking for “insufficient evidence” or “reject all” episodes |
| Reversal-detection latency | Query/feedback steps from a scripted context change until policy flags stale preference or updates its recommendation; distinguish no-feedback detection from actual preference confirmation |
| Query burden | Number of queries, cost budget, and answerable decisions; report query-budget exhaustion separately |
| Abstention behavior | Correct abstention on missing, contradictory, stale, rejected-all, and budget-exhausted controls; useful coverage on positive controls |
| Held-out task transfer | Pairwise/top-choice behavior on task families absent from policy fitting; report by task/context shift, never as same-task replay |
| Feasible Pareto recall | Fraction of known feasible non-dominated candidates preserved; use upstream comparator semantics and its declared intervals |
| Hard-gate violation count | Must be exactly zero; any ineligible candidate selected is a hard failure, not a small utility penalty |
| Confidence calibration | Brier/log loss and reliability diagnostics only when a policy actually emits probabilities and a separate calibration split exists |
| Feedback conflict/noise | Detection rate, false conflict rate, response to repeated/contradictory observations, and whether confidence tracks evidence quality |
| Determinism and provenance | Byte-stable reports for identical canonical inputs/seeds; fixture version, scenario split, policy revision and parameters recorded |

Do not set a universal acceptable calibration error, regret threshold, or query count before the scenario generator, split policy, denominators and statistical analysis are frozen. Thresholds are decision- and risk-specific.

## 7. Software acceptance gates

The initial fixture should be implemented as deterministic contract tests. Proposed minimum software gates:

- **100%** of fixture cases with explicit expected behavior pass, including every negative control.
- **0** ineligible candidate selections after hard feasibility filtering.
- **100%** preservation of the source feasible non-dominated candidates required by the declared Pareto contract.
- **0** claims of current context-specific preference based only on stale or task-mismatched feedback.
- **100%** of missing/contradictory/reject-all controls trigger explicit uncertainty, query, or abstention behavior consistent with available query budget.
- **0** evaluation-oracle fields reach policy features, model fitting, active query ranking, or calibration.
- **0** synthetic preference records are reported as actual wearer input or clinical evidence.
- Same canonical fixture + same policy revision/parameters produces byte-identical replay artifacts.
- The benchmark fails closed on unknown schema versions, duplicate scenario/observation IDs, unknown candidate references, impossible budget values, malformed preference pairs, and split leakage.

A blanket “always abstain” implementation does **not** pass: positive controls with matching, conflict-free, context-relevant feedback must support an admissible useful recommendation, while hard constraints remain in force.

## 8. Shift and leakage controls

The generated follow-on corpus must include, at minimum:

- same task, preference changes after a context/time transition;
- context changes but preference does not (false-reversal negative control);
- task shift where the engineering Pareto set changes;
- an unseen terrain/task family;
- a candidate preferred in explicit feedback but ruled infeasible by a hard gate;
- feedback contradicting a proxy metric such as task success, mass, or inferred comfort;
- missing feedback despite abundant sensor/engineering data;
- repeated/contradictory feedback, feedback noise, and changing time windows;
- rejection of all currently eligible candidates;
- query-budget and compute-budget exhaustion;
- context-blind overconfidence;
- a leakage canary in the evaluation oracle whose value must not influence any policy output.

Use matched seeds for every baseline. Split by wearer/session and full trajectory, and reserve entire task/context families for final evaluation. Tune on development data only. Publish a seed/split manifest and a canonical hash of the fixture and policy config. A hash identifies bytes; it does not authenticate any human report or physical measurement.

## 9. Current research and standards context

- A 2026 review describes fragmented lower-limb exoskeleton evaluation and gaps in safety, stability, comfort, and environmental adaptability: [Hu & Wang, *Robotics and Autonomous Systems* (2026)](https://doi.org/10.1016/j.robot.2026.105645).
- A second 2026 review proposes a multi-level framework spanning the human, machine, and human–robot interaction: [*Biomimetic Intelligence and Robotics* (June 2026)](https://doi.org/10.1016/j.birob.2026.100340).
- A 2026 open-access study evaluates user-preference-based human-in-the-loop tuning of exoskeleton assistance during walking: [Schäfer et al. (13 May 2026)](https://www.nature.com/articles/s44385-026-00085-7). It motivates testing individual preference, but its participants/results do not define universal thresholds or prove our proposed policy.
- An August 2026 preprint on preference-based Bayesian optimization reports 90.7% validation accuracy after 20 interaction iterations for one six-parameter hip-exoskeleton tuning task, with treadmill and outdoor experiments: [Liu et al., arXiv:2608.09015](https://arxiv.org/abs/2608.09015). This motivates measuring feedback/query efficiency; its task, baseline, participants and outcome definitions differ from this synthetic benchmark, so 20 iterations is a comparison point, not a universal acceptance threshold.
- [ISO 18646-6:2026](https://www.iso.org/standard/88497.html) specifies performance indices and tests using an anthropomorphic test dummy for lower-limb wearable robots. It is relevant to technical assistance/enhancement measurement, not a substitute for explicit human preference evidence; the published scope excludes robots operated based on biosignals such as EMG.
- [ASTM F3474-25](https://store.astm.org/f3474-25.html) provides practice for choosing functional ergonomic parameters and metrics across exoskeleton contexts and time horizons.

These references reinforce a system-level benchmark that keeps assistance, feasibility, wearer interaction, context and time in view. They do not validate the synthetic fixture or set its numerical gates.

## 10. Claim ceiling and sequence

A passing deterministic fixture can establish only that a particular implementation honors the declared synthetic preference/context/feasibility contracts. It cannot establish that a policy knows real user preferences, is clinically helpful, improves metabolic outcomes, is safe, or is ready to control a powered wearable robot.

Sequence:
1. Review and freeze fixture schema and expected behavior.
2. Add a typed Rust replay/evaluator and exhaustive fixture tests; do not connect it to actuation.
3. Add generated scenarios, split/seed manifest and paired baseline runner.
4. Run locally where toolchain permits; record exact source revision and real execution output.
5. Only after offline benchmark qualification, design any human study separately with appropriate protocol, consent/ethics, safety review and calibrated instrumentation.
