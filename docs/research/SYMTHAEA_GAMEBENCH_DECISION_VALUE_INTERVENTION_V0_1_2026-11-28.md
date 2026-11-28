# Symthaea GameBench Decision-Value Intervention Track v0.1

**Tracking issue:** [#14187](https://github.com/Luminous-Dynamics/symthaea/issues/14187)  
**Parent programme:** [#14137](https://github.com/Luminous-Dynamics/symthaea/issues/14137)  
**Date:** 2026-11-28  
**Status:** Research protocol and isolated reference primitive; qualification pending

## 1. Thesis

An intelligent explorer should not maximize information. It should acquire information only when that information can change a consequential decision enough to justify its cost.

GameBench already contains an F6 hidden-mechanics fixture in which visually matched worlds differ in a hidden switch rule. The next step is a decision-focused epistemic-control experiment: the candidate predicts what probes will reveal, selects an intervention, observes its actual consequence, revises its model, and makes a downstream decision. The evaluator scores the entire causal chain.

This track reuses the game task and observation/action boundary from #14137. It composes with active scientific intelligence (#13691), decision-centric world-model evaluation (#9094), sealed task generation and evaluator custody (#9332/#10756), and canonical run-manifest accounting (#9735). It does not create a second generic task manifest, evaluator authority, or runner.

## 2. The paradigm shift

Traditional exploration often rewards information gain, map coverage, or curiosity. These are incomplete objectives.

A probe can reveal several bits about irrelevant mechanics without changing the best action. A low-entropy probe can reveal the one distinction that changes a decision from unsafe to safe. A probe may also alter the environment: opening a door, consuming a key, advancing a hazard, or irreversibly committing a resource. Its cost and side effect must be part of the decision.

The benchmark separates:

1. **Mechanism identifiability:** can the candidate distinguish mechanisms that predict different consequences?
2. **Forecast quality:** did it predict the outcome distribution before probing?
3. **Posterior revision:** did it update beliefs using the observed outcome without overclaiming?
4. **Decision value:** did the intervention improve the best available downstream action?
5. **Intervention cost/risk:** was the expected improvement worth the action, resource, compute, latency, and irreversible-state cost?
6. **Transfer:** does the discovered mechanism generalize across appearance, layout, and novel compositions?
7. **Assurance:** can an independent evaluator replay the trajectory and recompute every metric?

No metric compensates for a failed claim boundary or invalid action protocol.

## 3. Exact finite reference

The isolated crate is:

\`research/symthaea-gamebench-evsi-v0-1/\`

For hypotheses h in H, decisions a in A, prior/posterior p(h), and an intervention I with possible outcomes o in O_I:

\[
V_0(p)=\\max_{a\\in A}\\sum_h p(h)U_0(a,h)
\]

\[
V_I(p)=\\sum_o P(o\\mid p,I)
\\max_{a\\in A}\\sum_h P(h\\mid o,p,I)U_{I,o}(a,h)
\]

\[
\\mathrm{GrossDelta}(I\\mid p)=V_I(p)-V_0(p)
\]

\[
\\mathrm{NetValue}(I\\mid p)=\\mathrm{GrossDelta}(I\\mid p)-c(I)
\]

\[
\\mathrm{IG}(I\\mid p)=H(p)-\\sum_o P(o\\mid p,I)H(P(h\\mid o,p,I))
\]

Here, U_I,o is the utility of each downstream decision after the intervention and its observation. It can encode physical side effects. For a purely informational probe, set every U_I,o = U_0. The two score families remain separate: information gain is a diagnostic; net decision value is the decision criterion.

The exact reference selects the positive-net-value intervention with highest value, with deterministic declaration-order tie-breaking. It returns no intervention when all net values are non-positive. It rejects malformed probability mass, impossible observations, duplicate identifiers, invalid utility matrices, and non-finite costs/values.

### Why exact finite fixtures first?

For small finite tasks, exact Bayesian updates and exhaustive decision values let us test the evaluator itself. This creates a reliable unit-test surface before adding stochastic simulators, language-model agents, or visual interfaces. It does not demonstrate that a candidate can represent the hypotheses or discover the correct model.

## 4. Required task design

### 4.1 Paired hidden-mechanics twins

Construct pairs with the same visible layout, goal, assets, and initial observation, changing exactly one causal transition rule. Examples:

- switch opens only the adjacent gate versus all gates;
- conveyor advances east versus west;
- resource regenerates versus depletes permanently;
- hazard advances every action versus every second action;
- key unlocks one door permanently versus key is consumed and unlock is temporary.

The independent generator must prove that initial observations are aliased and that at least one legal intervention distinguishes the mechanisms. If no available intervention can distinguish two hypotheses, the evaluator marks them observationally equivalent under the permitted action budget. The candidate is not rewarded for naming one arbitrarily.

### 4.2 Decision-relevant and irrelevant evidence

Every cohort includes:
- high-information probes with no effect on the best downstream decision;
- lower-information probes with high decision value;
- probes whose expected value is positive only at particular belief states;
- probes dominated by a cheaper probe;
- probes with no possible positive net value;
- probes whose physical side effect makes an otherwise informative action harmful;
- stochastic probes with overlapping outcome distributions;
- no-probe and abstention controls.

This prevents the benchmark from equating curiosity, entropy reduction, or probe count with intelligence.

### 4.3 Budget and risk

Probe costs are explicit and denominated in a declared utility unit. Keep raw measurements alongside any scalar conversion: actions, consumed resources, irreversible-loss probability, wall time, CPU/GPU time, peak memory, simulator calls, model queries, and energy method/quality. Report Pareto frontiers rather than a hidden weighted score.

Do not silently assign arbitrary monetary/energy weights. Pre-register the utility conversion per task family and report sensitivity to reasonable alternative weights. For safety-critical probes, include a hard constraint on expected damage in addition to utility; favorable average value cannot excuse a forbidden action.

### 4.4 Forecast and update protocol

Before each intervention, the candidate emits a probability distribution over declared observable outcomes and, where available, its candidate-authored posterior or mechanism predictions. After the action, the evaluator returns only the actual task observation. The candidate then emits an updated belief and downstream action.

Score forecasts with proper scoring rules (Brier and/or log score) against sealed outcomes. Score belief calibration separately. A narrative explanation is diagnostic only and cannot substitute for a pre-action forecast or machine-checkable belief update.

## 5. Baselines and interventions

Compare on identical task instances and budgets:

- **No-probe:** choose the best action under the initial belief.
- **Random legal probe:** random intervention from the legal set.
- **Entropy-greedy:** choose the probe with largest expected information gain.
- **Cost-aware entropy:** information gain divided by declared cost; diagnostic, not oracle.
- **Decision-value oracle:** evaluator-only exact reference on finite fixtures.
- **Heuristic causal learner:** explicit hypothesis table updated from observations.
- **Symthaea component arms:** FEP, HDC, temporal model, episodic memory, planner, and coupled variants as actually integrated. Label every arm by exact component and source digest.
- **External agent baselines:** use protocol-matched BALROG/GameWorld agents where interface and task semantics genuinely match. Never convert their aggregate progress into a causal-discovery score.

Ablate memory, temporal dynamics, HDC, active sensing, planning, and model revision with matched interaction and inclusive compute budgets. Preserve the same evaluator, task, safety constraints, and action opportunities across ablations.

## 6. Scorecard

Publish separate axes:

| Axis | Measure | Failure it exposes |
|---|---|---|
| Mechanism discrimination | Accuracy and calibrated log score on held-out mechanisms | Guessing or memorizing a named rule |
| Outcome forecast | Brier/log score, calibration, coverage | Post-hoc explanation without prediction |
| Decision value | Gross and net value delta | Information that does not improve action |
| Probe regret | Oracle intervention value minus candidate intervention value | Wasteful or harmful exploration |
| Downstream regret | Utility loss against exact decision after actual outcome | Correct discovery but poor action |
| Adaptation | Interactions/updates to recover after one-rule flip | Frozen policy and slow relearning |
| Transfer | Held-out appearance, layout, and composition | Surface memorization |
| Retention/interference | Previously qualified skill retention after sequential tasks | Catastrophic forgetting |
| Protocol | Valid actions, abstention correctness, replay agreement | Invalid or unauditable behaviour |
| Efficiency | Cost per independently verified success/skill | Expensive capability hidden by raw score |

Do not collapse these into one scalar ranking. Report confidence intervals and per-family results.

## 7. Mutation tests and falsification

Required negative controls:
- identical observation and identical outcomes under multiple hypotheses;
- action/outcome histories shuffled or duplicated;
- probe observation corrupted or delayed;
- hypothesis labels permuted without changing the model;
- irrelevant high-information probe;
- impossible outcome;
- negative-cost, non-finite, or malformed-likelihood fixtures;
- tampered trace, changed rule digest, or substituted seed;
- oracle posterior or hidden rule accidentally serialized to the candidate;
- intervention whose physical side effect dominates its information benefit.

Falsify the causal-discovery claim if the candidate performs no better than a matched entropy-greedy or random policy on decision-value regret; predicts only after seeing outcomes; claims unique identification under observational equivalence; or loses the effect on held-out mechanism compositions.

## 8. Industry comparison and realistic positioning

### BALROG

BALROG is the agentic reasoning baseline across BabyAI, Crafter, TextWorld, Baba Is AI, MiniHack, and NetHack. The official public leaderboard snapshot visible on 2026-11-28 shows Claude Opus 5 Max at **63.4 ± 1.8 overall progress**, dated 2026-09-20. The submission data lists 254 episodes and highly uneven per-environment outcomes, including 7.3% NetHack progress. This is a dated aggregate-progress reference, not an intervention-selection result or a locally reproduced baseline.

Official sources: [BALROG leaderboard](https://balrogai.com/) and [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html).

### GameWorld

GameWorld evaluates 34 browser games / 170 tasks using state-verifiable success and normalized progress. Its paused protocol isolates decision quality; GameWorld-RT keeps the environment running during inference and is a distinct latency-sensitive track. The public Fine-Control 400-step leaderboard snapshot reports GPT-6.1-Terra at **76.79 general progress / 61.18 general success** and **72.53 computer-use progress / 55.29 computer-use success**. These are protocol-specific figures, not a direct comparison to BALROG or this finite intervention core.

Sources: [GameWorld project](https://gameworld-project.github.io/) and [official repository / leaderboard](https://github.com/gameworld-project/GameWorld).

### Generated interactive worlds

Google DeepMind's Genie 4 announcement (2026-10-27) and November 2026 model card show the move toward high-fidelity, steerable, multimodal world generation. The public model card still identifies short-session continuity, object persistence, complex interactions, and spatial accuracy as active limitations. This makes generative worlds valuable for diversity and challenge proposals, but they must not be the source of ground truth. Every generated challenge needs a separate executable ruleset, seeded hidden-state authority, and verifier-owned outcome.

Sources: [Genie 4 announcement](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/) and [Genie 4 model card](https://deepmind.google/models/model-cards/genie-4/).

### Where Symthaea can win

Do not try to outspend frontier multimodal systems on raw visual fidelity or broad browser-game coverage. Aim to outperform on a narrower, falsifiable capability: **choosing the right experiment, learning the causal rule, updating only what changed, and transferring that model at lower verified interaction and compute cost**. BALROG/GameWorld remain external capability checks; this intervention track measures a different axis they do not directly isolate.

## 9. Prospective qualification targets

Targets are hypotheses, not results:

- Exact 100% agreement with the finite reference on all frozen deterministic posterior/value fixtures and negative controls.
- Zero probe choices when every available intervention has non-positive net value.
- Zero unique-mechanism claims under observational equivalence.
- At least 80% held-out mechanism discrimination on a preregistered compositional cohort, with confidence intervals.
- Statistically supported reduction in decision-value regret versus random and entropy-greedy policies at matched budgets.
- At least 20% lower interactions or inclusive cost than a matched from-scratch baseline on held-out combinations, without worse safety or task validity.
- 100% replay agreement for qualified episodes; missing or inconsistent receipts remain INCOMPLETE/INVALID.

The first target qualifies the reference implementation only. The later targets require real Symthaea candidate arms and held-out campaigns.

## 10. Execution sequence

1. Freeze the mathematical reference and tests in the isolated crate.
2. Integrate F6 probes with evaluator-private likelihood/utility tables; do not expose the oracle model.
3. Add candidate-authored pre-intervention outcome forecasts and post-intervention belief updates.
4. Add no-probe, random, entropy-greedy, cost-aware entropy, and exact-oracle baselines.
5. Add sealed twin mechanics and decision-relevance controls.
6. Add matched FEP/HDC/temporal/memory/planner ablations only after those components are actually wired.
7. Execute the small deterministic cohort first; then stochastic and held-out rule compositions.
8. Only then connect to BALROG/GameWorld external tracks and report protocol-matched comparisons.

## 11. Qualification status

- The Rust reference and tests are source artifacts until exact-head CI or local commands complete.
- No candidate agent has yet been evaluated on this track.
- No mechanics-discovery, decision-value, held-out transfer, energy-efficiency, or SOTA result is claimed.
- A skipped workflow is not a pass. A missing or stale receipt is not a pass.
