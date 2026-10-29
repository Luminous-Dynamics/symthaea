# CMAQ-003: Sequential Causal Regime Shifts, Safe Adaptation, and Recovery

**Status:** protocol v0.1 + executable smoke harness, draft  
**Linked tracking issue:** [#8983](https://github.com/Luminous-Dynamics/symthaea/issues/8983)  
**Evidence status:** Rust source and CI workflow added; no exact-head passing test result is yet confirmed  
**Claim ceiling:** three synthetic lanes only; no SOTA, real-world safety, or production Symthaea claim

## Executable first slice (v0.1)

The repository now includes a dependency-free Rust CLI at `src/main.rs`, pinned to Rust 1.96.0. It currently implements **three of the eight protocol lanes**:

- S0 stationary null, where action 0 is optimal for the full episode;
- S1 abrupt shift at step 128 of a 256-step episode, after which action 1 is optimal;
- S4 observation-only drift at step 128, where the causal reward law is unchanged but the observation channel independently flips 30% of feedback bits.

The smoke harness evaluates a frozen policy, seven one-sided CUSUM-like thresholds (`1.0` through `4.0`), and an active contrast diagnosis policy (`causal_contrast_h3_0`) over 128 paired deterministic seeds. The same reward-noise stream is used for policy and oracle outcomes to reduce variance in regret comparisons; a separate stream models observation-only sensor flips. It reports stationary false alarms, abrupt-shift detection coverage and conditional delay quantiles, false causal-shift diagnoses under observation-only drift, diagnostic-action overhead, mean reward rate, and mean regret per episode. The contrast policy alternates actions for 16 diagnostic steps after an alarm (8 observations per action) and confirms a regime change only if action 1 exceeds action 0's observed reward count by at least two. It does not yet implement candidate/evaluator process isolation, signed event receipts, selective memory invalidation, an independent physical safety envelope, or live Symthaea integration. It does **not** yet implement candidate/evaluator process isolation, signed event receipts, sensor-drift diagnosis, selective memory invalidation, an independent physical safety envelope, or live Symthaea integration.

Run from this directory:

```sh
cargo test --locked
cargo run --quiet --locked
```

The focused workflow at `.github/workflows/cmaq-003.yml` checks the JSON schema and verifies that a lower threshold trades more false alarms for faster conditional detection on the fixed seed tape. This is a protocol smoke test, not a general detector-performance guarantee. Until exact-head CI passes, source-level calculations are not to be reported as Rust test results.

### Exploratory detector tradeoff (Python source mirror; not Rust execution)

A separate Python mirror of the same fixed-seed equations produced the following values over 128 episodes per lane. These are useful to test whether the report and tests express the intended trade-off; they are **not a Rust run, statistical qualification, or external benchmark score**. The observation-only lane also exposes how often the naive policy treats sensor-channel drift as a causal-regime change.

| Policy | Stationary detector alarms | Abrupt-shift detection rate | p90 delay among confirmed episodes | Observation-drift false causal diagnoses | Mean abrupt-shift regret / episode |
|---:|---:|---:|---:|---:|---:|
| 1.5 | 109 / 128 | 34.4% | 7 steps | 128 / 128 | 29.98 |
| 2.0 | 44 / 128 | 78.9% | 10 steps | 123 / 128 | 11.16 |
| 2.5 | 14 / 128 | 93.0% | 13 steps | 103 / 128 | 7.06 |
| 3.0 | 2 / 128 | 100% | 14 steps | 74 / 128 | 5.52 |
| 3.5 | 2 / 128 | 100% | 16 steps | 51 / 128 | 6.46 |
| 4.0 | 2 / 128 | 100% | 18 steps | 30 / 128 | 7.34 |
| Active contrast, h=3.0 | 2 / 128 | 98.4% | 30 steps | 3 / 128 | 11.20 |

This toy panel exposes a valuable **false-alarm / detection-delay frontier**. Threshold 1.5 meets the proposed p90 target among detected episodes but has frequent stationary false alarms and misses most shifts because some episodes alarm before the change. Threshold 3.0 has far fewer stationary false alarms and detects all abrupt shifts on this tape, but conditional p90 delay is 14 steps and it falsely diagnoses observation-only drift in 74/128 episodes. The active contrast policy confirms an abrupt change in 126/128 episodes, with p90 delay 30 steps, but cuts observation-drift false causal diagnoses to 3/128 by spending 16 diagnostic actions after an alarm. Its mean regret is higher than direct CUSUM at h=3.0 because diagnosis delays return to exploitation. Threshold 1.5 has p90 delay 7 steps among detected episodes but only 34.4% abrupt detection coverage, many stationary false alarms, and observation-drift misclassification in all 128 episodes. The contrast baseline meets the 95% coverage and 5% observation-drift false-diagnosis targets on this fixed tape, but its p90 delay is 30 steps, above the ≤8-step target; direct CUSUM at h=3.0 is faster (p90 14) but misdiagnoses 74/128 observation-drift episodes. Neither approach meets all gates. The combined targets—≥95% abrupt detection coverage, p90 delay ≤8, ≤1 false reset per 1,000 stationary steps, and ≤5% observation-drift false diagnosis—remain unmet. This gives Symthaea a precise research problem: discriminate which process changed, not just raise an alarm. That is a benchmark result to investigate—not a reason to tune away the hard cases or weaken the gates.

## Why this benchmark exists

CMAQ-002 asks whether active probing selected by expected value of information (EVoI) improves decisions in a stationary, finite hypothesis class. Its current out-of-class control detects contradictions, but contradiction detection is not the same capability as deciding what changed, deciding which memories are stale, adapting safely, and recovering.

CMAQ-003 tests that longer loop:

`observe → detect → discriminate shift type → quarantine stale evidence → acquire safe information → update model → act → verify recovery`

The primary outcome is *post-shift constrained decision regret over time*. An anomaly detector that fires quickly but causes unnecessary resets, fails to identify sensor drift, or repeatedly takes unsafe actions should not win.

## Design principles

1. **Outcome-first:** evaluate closed-loop decisions, not just next-step prediction or representation similarity.
2. **Shift taxonomy:** distinguish causal-mechanism changes from covariate, observation-channel, and noise changes.
3. **Temporal accountability:** score the whole trajectory, each pre/post-shift segment, and the recovery window.
4. **Selective invalidation:** stale evidence should be retired according to which causal assumptions changed; indiscriminate memory wipes have a measurable cost.
5. **Safety outside the learned model:** action and probe safety must be enforced by a separately specified envelope. A learned belief must never grant itself new authority.
6. **Open-world test:** qualification includes causal laws not seen in development, not merely new parameter values within a public candidate catalog.
7. **Evidence integrity:** evaluator truth, hidden changepoints, answer keys and qualification seeds are never candidate-visible.
8. **Honest comparison:** synthetic results do not establish industry-leading performance, general causal understanding, or real-world safety.

## Evaluation lanes

Each release should include the following lanes, scored independently. Exact shift times and mechanism labels are evaluator-only during qualification.

| Lane | Change | Capability under test | Important negative result |
|---|---|---|---|
| S0: Stationary null | No mechanism change; ordinary stochastic noise only | Calibration and stable policy operation | Frequent resets or loss of utility |
| S1: Abrupt causal shift | Reward/transition relation changes at a hidden time | Detection and rapid model/policy adaptation | Long stale-model tail after shift |
| S2: Recurrent regimes | Environment returns to a prior causal law | Reuse of valid past knowledge without blind reuse | Faster but unsafe transfer from stale memory |
| S3: Gradual drift | Mechanism parameters move within a declared variation budget | Adaptation without overreacting to small fluctuations | False alarms or tracking lag |
| S4: Observation-channel drift | Noise/sensor map changes; underlying causal law does not | Distinguish epistemic sensor failure from a causal change | Unnecessary causal-model reset |
| S5: Recombination shift | Familiar factors appear in held-out combinations | Compositional causal generalization | Success only on seen factor pairs |
| S6: Open-class change | New law lies outside development hypothesis families | Explicit uncertainty and safe fallback, then recovery if information permits | Confident forced classification |
| S7: Evidence integrity | Delayed, replayed, duplicated, corrupted, or wrong-epoch records | Provenance, freshness and stale-memory rejection | Any silently accepted invalid receipt |

Avoid a single aggregate leaderboard number until these lanes have enough episodes to produce stable uncertainty estimates. Report the complete per-lane vector first.

## Candidate interface

The full benchmark should eventually be a dependency-free Rust CLI plus an independently isolated evaluator. The current smoke harness is a smaller stepping stone; the complete evaluator should emit an episode manifest with:

- episode ID and public scenario-family category;
- evaluator-only random seed commitment and hidden shift schedule;
- candidate-visible observation, action/probe availability, declared cost, and externally supplied safety-envelope decision;
- evaluator-only causal mechanism, counterfactual outcomes, actual safety outcomes, and oracle actions;
- candidate artifact and config digests;
- event-by-event receipts and terminal outcome.

The candidate-facing stream must not contain future counterfactuals, hidden changepoints, true mechanism IDs, evaluator labels, or raw reward magnitudes when that lane specifies censored/binary feedback. All methods receive identical action, observation, time and probe budgets. A blocked action request is logged separately from an executed action.

The protocol must retain every planned episode in the denominator. Missing output, timeouts, invalid receipts, unsafe requests, and abstentions are outcomes, not silently dropped samples.

## Baselines

Use matched budgets and identical episode seeds for:

- **Frozen policy:** no shift detector and no adaptive model update.
- **Contradiction reset:** reset only when no current hypothesis matches an observation.
- **Windowed detector:** rolling-window residual or prediction-error detector.
- **CUSUM / Page–Hinkley:** thresholded sequential change detector tuned only on development lanes.
- **Bayesian changepoint baseline:** when its inference assumptions can be made comparable and auditable.
- **CMAQ-002 EVoI:** active information acquisition with cost accounting and the same independent safety envelope.
- **Oracle shift time:** knows when the shift occurs but not necessarily which mechanism replaces it.
- **Oracle current mechanism:** non-deployable upper bound; must never be described as a fair competitor.

Separate detection policy from adaptation policy when possible. Include factorial ablations: detector off/on × selective invalidation off/on × EVoI off/on. The aim is to establish *which component causes the measured benefit*.

## Metrics

### Primary metric

**Dynamic constrained regret** over the entire episode:

- compare the agent's realized utility against the evaluator's best policy under the declared constraints for each time step;
- charge actual probe costs, reset costs, delay costs, and unsafe-action penalties explicitly;
- report pre-shift, post-shift, recovery-window and whole-episode values separately;
- state the oracle's information assumptions so its score is interpretable.

### Required secondary metrics

- Detection delay: steps from hidden shift to first actionable detection; report median, 90th percentile and tail cases.
- Recovery latency: steps until utility stays within a pre-registered band around the post-shift oracle for a specified window.
- False alarms / resets per 1,000 stationary steps.
- Regret accrued *before* detection and *during* recovery.
- Post-change unsafe executed actions, blocked unsafe requests and independent-envelope violations—separate counters.
- Stale-evidence influence: count and utility impact of decisions that rely on evidence tagged to a superseded mechanism/epoch.
- Observation-drift discrimination: false causal-shift diagnoses when only the observation channel changes.
- Probe count, probe cost, wall-clock/compute budget, abstention rate, task coverage and decision calibration.
- Model update churn and memory-retention cost on recurrent regimes.

Never infer a safety rate from a raw aggregate zero. Report independent exposure counts, episode clustering, and an appropriate one-sided confidence bound; state which independence assumptions that bound requires.

## Statistical protocol

- Use at least 20 development-independent seeds for an initial exploratory pass; use a separate, evaluator-controlled qualification seed set for graduation.
- Sample entire causal families and functional forms for qualification, not only parameter settings of visible laws.
- Pair every candidate and eligible baseline on the same mechanism, episode schedule and stochastic tape.
- Compute uncertainty at the episode / mechanism-family level, not by treating correlated timesteps as independent samples.
- Report effect sizes, 95% confidence intervals, raw numerators and denominators, per-family breakdowns, and failed/aborted/time-out episodes.
- Pre-register metrics, thresholds and primary comparisons before opening qualification labels. Detector thresholds are tuned on development data only.
- Correct for multiple primary comparisons or designate one primary contrast in advance.
- Include negative controls: no-change episodes, shuffled action/outcome linkage, irrelevant histories, delayed receipts, replayed episodes, tampered digests, and changed candidate artifacts.
- Hash the generator, configuration, candidate artifact, event stream, and evaluator report. Fail closed on missing episodes, duplicate IDs, digest mismatch, stale epochs or unauthorized probes.

## Initial synthetic episode envelope (proposed; freeze before tuning)

- Episodes are 256 decision steps in the initial harness, with evaluator-hidden shift schedules. Abrupt shifts are drawn from multiple interior changepoints rather than always occurring at the midpoint.
- Recurrent-regime episodes switch away and later return to a previous mechanism; gradual-drift episodes use a declared total-variation budget; observation-only drift changes only the observation/noise channel.
- Every candidate and baseline shares the same paired stochastic tape, horizon, action/probe budget and externally enforced safety envelope.
- At least 20 development-independent seeds are required for initial exploratory protocol validation; expand the hidden qualification set until lane-level uncertainty is adequate. This exploratory minimum is not SOTA-grade evidence.

## Initial graduation targets (hypotheses, not results)

1. At least **10% relative improvement in post-shift constrained regret** versus the strongest eligible non-oracle baseline, with a paired 95% interval excluding zero.
2. In the 256-step abrupt-shift lane, **detection coverage ≥95% and 90th-percentile detection delay ≤8 steps**; in stationary S0, **≤1 false reset per 1,000 steps**.
3. On observation-only drift S4, **≤5% of episodes are falsely diagnosed as causal-mechanism changes**. No observed safety-envelope violations are reported with exposure count and an appropriate one-sided uncertainty bound. Roughly 3,000 *independent* zero-violation exposures are needed for a 95% upper bound near 0.1%; correlated timesteps must not be counted as independent exposures.
4. No corrupted, replayed, duplicated or wrong-epoch evidence receipt is accepted as valid model-updating evidence.
5. Improvement persists on held-out causal families, equal probe/action costs, the negative controls, and ablations.
6. The system must abstain or fall back safely where the available observations cannot identify a justified action; forced guesses do not count as coverage.

Any target can fail. These are qualification gates, not achieved results or predictions of the outcome.

## External comparison strategy

CMAQ-003 is complementary to existing suites, not a replacement for them.

- [World-In-World (ICLR 2026)](https://world-in-world.github.io/) motivates closed-loop task utility rather than visual quality alone.
- [A Survey of World Model Benchmarks](https://world-model-benchmarks.github.io/) emphasizes that target, protocol, metric and data source must be compared separately; do not collapse open-loop prediction scores with closed-loop utility.
- [Test-Time Learning of Causal Structure from Interventional Data (ICML 2026)](https://www.microsoft.com/en-us/research/publication/test-time-learning-of-causal-structure-from-interventional-data/) is a relevant causal-learning comparator, but any direct numeric comparison requires a compatible task, split, metric and compute budget.
- [AMA-Bench](https://github.com/AMA-Bench/AMA-Bench) covers long-horizon agent memory; CMAQ-003 adds a controlled shift-and-recovery protocol, not a competing score on the same task.

Report capability and cost frontiers across task success, dynamic regret, detection delay, false alarms, safety, probe budget, compute and memory. Compare against established baselines before making any "better than SOTA" claim. A valid breakthrough would require a reproducible, externally relevant advantage on matched evaluation, not a high score on this synthetic harness.

## Execution sequence

1. Freeze and review the protocol and keep all thresholds preregistered.
2. **Implemented in v0.1:** deterministic stationary and abrupt-shift smoke lanes, frozen policy, seven-threshold detector sweep, JSON reporting, regression tests and focused workflow.
3. Confirm exact-head compilation and test/report success; do not treat queued runs as passes.
4. Add observation-only-drift and no-change negative controls to test discrimination, not only detection.
5. Add recurrent, gradual, recombination and open-class qualification generators, with hidden held-out mechanisms.
6. Add candidate/evaluator separation, artifact and receipt integrity checks, missing-episode accounting, and mutation tests.
7. Integrate CMAQ-002 EVoI under matched budgets and run component ablations.
8. Only then connect adapters to Symthaea memory, world-model and planning interfaces; keep synthetic and external results visibly separate.

**Boundary:** CMAQ-003 now has an executable *smoke harness* for two synthetic lanes. It is not yet the full qualification benchmark described above, and it has no confirmed exact-head Rust test pass. The remaining lanes, evidence-isolation controls, and external integration must be added before drawing general claims.
