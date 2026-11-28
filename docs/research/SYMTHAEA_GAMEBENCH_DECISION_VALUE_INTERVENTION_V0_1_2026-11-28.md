# GameBench hidden-mechanics bridge to shared decision-value evaluators

**Date:** 2026-11-28  
**Tracking issue:** [#14187](https://github.com/Luminous-Dynamics/symthaea/issues/14187)  
**Parent game benchmark:** [#14137](https://github.com/Luminous-Dynamics/symthaea/issues/14137)  
**Shared one-step decision evaluator:** [research/decision-predictability-benchmark](https://github.com/Luminous-Dynamics/symthaea/tree/main/research/decision-predictability-benchmark)  
**Sequential robust policy owner:** [#13567 / draft PR #13571](https://github.com/Luminous-Dynamics/symthaea/pull/13571)  
**Cross-domain intervention co-design:** [#14161](https://github.com/Luminous-Dynamics/symthaea/issues/14161)

## Scope correction: reuse the existing decision oracle

The repository already contains a finite-world Bayesian and prior-free decision-value evaluator. Its README records an exact-head run on commit `d66d216d77d7b9d70cbbd7e66fa75a4bda3f6738`: Rust 1.96.0 formatting, 16/16 tests, and strict Clippy. The Bayesian evaluator calculates posterior Bayes risk, gross/net EVSI, and mutual information separately, with permission, budget, deadline, and risk-admission checks.

The existing evaluator has known-answer controls for:
- passive observational equivalence and irreducible decision risk;
- a decision-changing probe with positive net EVSI;
- a one-bit nuisance probe with zero decision value;
- over-budget/deadline rejection and risk-gate vetoes.

The repository's #13571 contribution adds a separate prior-free sequential policy tree with a separately written exhaustive enumerator. It is currently a draft and has not earned exact-head Rust qualification; it is not the Bayesian EVSI engine.

Issue #14161 owns the wider co-design of measurement/intervention, robust test, stopping, and terminal decision. #13739 owns the integration tournament for existing Symthaea experiment selectors.

**Therefore this GameBench track does not implement a second EVSI engine, posterior authority, robust oracle, or generic policy planner.** It supplies the game-specific adapter, forecast score, and mechanics challenge contract.

## Research question

Can a game agent identify which hidden-mechanics observation is useful for a later decision, forecast its observable outcomes before interacting, and abstain from irrelevant or unsupported claims—without being rewarded merely for reducing entropy?

The benchmark is deliberately about causal mechanics and decision relevance, not visual world realism. A hidden switch rule, conveyor direction, resource regeneration rule, or hazard cadence may be indistinguishable from the initial observation. A candidate must choose a legal read-only probe, predict its result distribution, update from the actual observation, and select a decision.

## Existing shared API mapping

The bridge package is:

`research/symthaea-gamebench-decision-bridge-v0-1/`

It path-depends on `research/decision-predictability-benchmark` and calls its `evaluate_candidate` function.

| GameBench concept | Shared evaluator field |
|---|---|
| Hidden mechanic/world variant | EvaluatorWorld.states |
| Consequence of a terminal action | StateModel.decision_loss |
| Public game observation | CandidateEpisode.evidence |
| Legal terminal actions | CandidateEpisode.allowed_decision_ids |
| Read-only inspection action | ProbeModel |
| Publicly offered inspection | CandidateEpisode.permitted_probes |
| Observable probe result | OutcomeLikelihood |
| Candidate's current terminal choice | CandidateSubmission.decision_id |
| Candidate's selected probe | CandidateSubmission.requested_probe_id |
| Probe budget and deadline | CandidateEpisode budget / step window |
| Evaluator-side posterior and expected decision value | CandidateEvaluation / ProbeValue |

The adapter passes the requested probe through to the existing evaluator and returns its EVSI and information-gain fields unchanged. It does not duplicate those equations.

## New candidate forecast contract

The shared evaluator accepts a chosen probe but does not accept a candidate-authored forecast. GameBench requires the candidate to submit, before probe execution, a probability distribution over the public outcome vocabulary:
- probe ID;
- exact set of public outcome IDs;
- one finite probability in [0,1] for each outcome;
- normalized mass within the declared tolerance.

After the environment reveals the actual outcome, the bridge scores the pre-action forecast with the multiclass Brier score:

[
\mathrm{BS}(p,o)=\sum_{i=1}^{K}(p_i-\mathbf{1}[i=o])^2.
]

Forecast score remains separate from EVSI, mechanism discrimination, task success, and decision regret. The evaluator must not return its exact posterior, hidden likelihoods, decision-loss table, or probe-value estimate to the candidate during the episode.

## Hard v0.1 limitation: read-only probes only

The shared one-step Bayesian EVSI model scores an information-gathering action with an outcome likelihood and an unchanged terminal-loss table. It does not define outcome-specific state transitions or post-intervention utility matrices.

The adapter therefore accepts only read-only observations, such as inspecting a label or reading a panel. It fails closed for world-mutating actions such as pulling a lever, consuming a key, opening a door, or advancing a hazard. Those actions must not be smuggled into a read-only probe model.

A later state-changing track must define explicit transition and outcome-conditioned terminal-loss semantics, then compose with the existing sequential/robust owners. This PR does not claim that the current shared evaluator supports those semantics.

## Development known-answer challenge

The bridge's test fixture has four equally weighted hidden worlds:
- switch mechanic A and nuisance state 0;
- switch mechanic A and nuisance state 1;
- switch mechanic B and nuisance state 0;
- switch mechanic B and nuisance state 1.

Terminal decision loss depends on the switch mechanic, not the nuisance lamp state.

Two read-only probes are offered:
1. Inspect the lamp: reveals the nuisance state perfectly, giving 1 bit of information but zero expected decision-value improvement.
2. Inspect the switch trace: a noisy 75/25 observation of the decision-relevant mechanic, giving approximately 0.1887 bits of information but 0.15 gross EVSI and 0.11 net EVSI at the frozen synthetic cost of 0.04.

The lower-information probe must outrank the high-information nuisance probe on net EVSI. These values are drawn from the shared evaluator; this bridge does not reimplement the equations.

## Candidate/evaluator custody

Candidate-visible:
- current public observation and evidence;
- allowed decision IDs;
- legal probe IDs, declared costs and deadlines;
- public outcome labels;
- candidate-authored pre-probe forecast;
- actual observation after the selected read-only probe executes.

Evaluator-only:
- hidden world/mechanism IDs and generator state;
- prior and outcome likelihoods;
- decision-loss table;
- posterior, Bayes risk, EVSI and information-gain calculations;
- oracle policy/value references and sealed task labels.

Rust type visibility is not a process sandbox. Confirmatory runs require an independently permissioned evaluator boundary and replay receipts bound to task, candidate artifact, environment, seed, and evidence digests.

## Scorecard and negative controls

Report separate axes:
- game task success and terminal decision loss;
- mechanism discrimination and false-unique-claim rate;
- probe-choice regret against the shared decision-value reference;
- forecast Brier/log score and calibration;
- gross/net EVSI, cost, deadline and action count;
- abstention and unsupported-case rates;
- held-out rule-composition and appearance transfer;
- replay agreement and receipt completeness.

Required controls:
- high-information nuisance probe with zero decision value;
- low-information decision-changing probe;
- observationally equivalent mechanics with no available distinguishing probe;
- probe outcome vocabulary mismatch;
- malformed or unnormalized forecast;
- duplicate/unknown outcome IDs;
- candidate forecast emitted only after observing the result (invalid protocol);
- over-budget and deadline-missing probes;
- world-mutating probe sent to the read-only adapter (must reject);
- task IDs, asset names, outcome labels, and error messages that encode hidden mechanism (must be absent from sealed candidate packets).

No average score can compensate for a false unique-mechanism claim, evaluator leakage, or invalid evidence.

## Industry comparison

BALROG's official leaderboard snapshot shows Claude Opus 5 Max at 63.4 ± 1.8 overall progress (dated 2026-09-20), with uneven performance across game families, including 7.3% on NetHack. This is a broad agentic game-progress benchmark, not a decision-value-of-intervention result.

GameWorld has two distinct protocols. Its standard project leaderboard reports the top generalist at 41.9% progress / 21.2% success and a novice-player progress baseline of 64.1 under the same task budget. Its separate Fine-Control 400-step leaderboard has GPT-6.1-Terra at 76.79% general progress / 61.18% general success; those figures are not comparable to the standard leaderboard or BALROG without protocol matching.

Genie 4 demonstrates the frontier move toward photorealistic, steerable, multimodal generated worlds. Its November 2026 model card still lists object/visual drift during longer sessions, complex interaction fidelity, spatial accuracy, and long-term consistency as active limitations. Such models may generate appearance diversity, but they must not own ground truth: a separate executable task model and verifier must determine what actually happened.

This benchmark should not chase raw visual fidelity first. The differentiator is **decision-relevant exploration under hidden mechanics**: fewer wasted interactions, calibrated forecasts, reliable abstention under non-identifiability, transfer to unseen rule compositions, and exact replay. BALROG and GameWorld remain external capability checks, not interchangeable score columns.

## Prospective qualification targets

These are targets, not current results:
- 100% exact agreement on frozen adapter/reference fixtures.
- Zero decision-value credit for the nuisance-only probe.
- Positive decision-value probe selected over a higher-EIG irrelevant probe at matched declared costs.
- Zero unique-mechanism claims under observational equivalence.
- At least 80% held-out mechanism discrimination on a preregistered compositional cohort, with clustered confidence intervals.
- Statistically supported reduction in probe-choice regret against random and entropy-greedy baselines under matched budgets.
- At least 20% reduction in interactions to a verified terminal decision on held-out rule compositions, with no worse safety or task validity.
- 100% replay consistency for qualified episodes; missing receipts are INCOMPLETE, not pass.

Only the first item can be qualified by this adapter crate. The rest require an actual GameBench candidate integration, sealed task families, and matched campaigns.

## Qualification state

- The shared decision-evaluator library's previous exact-head pass is its own qualification and does not qualify this adapter.
- The new bridge has twelve authored Rust tests. They include post-observation posterior/terminal-decision known answers, nuisance observation non-effect, episode-bound receipt validation, pre-registration timing, result delay, budget and deadline rejection, forecast scoring, and rejection of world-mutating probes. No local Rust toolchain is available; this crate has not been locally formatted, compiled, tested or Clippy-checked.
- Its exact-head CI is required before any Rust qualification claim.
- The GameBench agent adapter and candidate campaign remain future integration work.
- No agent capability, transfer, physical validity, or SOTA superiority is claimed.
