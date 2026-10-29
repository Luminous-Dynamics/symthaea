# CMAQ-004: World-Model Value-Add Under Matched Budgets

**Status:** executable synthetic smoke harness, v0.3 with cost frontier and paired uncertainty intervals  
**Tracking issue:** [#9010](https://github.com/Luminous-Dynamics/symthaea/issues/9010)  
**Evidence status:** exact-seed Python mirror available; Rust source and focused Actions workflow added, but Rust execution is not yet confirmed  
**Claim ceiling:** tiny finite held-out-composition harness only; no industry/SOTA, broad causal reasoning, or production Symthaea claim

## Research question

Does a learned action-conditioned model improve *closed-loop choice* enough to justify prediction cost, compared with a strong direct policy that uses the same support observations but does not fit a compositional reward law?

Prediction accuracy and decision value are different claims. A model can predict well but add too little utility to pay for inference; it can also fit training/support data yet fail on a held-out interaction or after a regime reversal.

This smoke harness isolates that bridge on a tiny, known-ground-truth family. It is not yet an integration with Symthaea's HDC/CfC/LTC, memory, world model, or planner.

## Executable scope

The dependency-free Rust CLI is pinned to Rust 1.96.0:

```sh
cargo test --locked
cargo run --quiet --locked
```

It observes support contexts `00`, `01`, and `10`, then evaluates decisions in the held-out combination `11`. Each support context/action cell has 1,024 deterministic Bernoulli samples. The final policy is evaluated on 4,096 paired seeded reward draws.

The direct baseline is a **nearest-context/tabular fallback**: it averages outcome rates from the two adjacent, observed factor combinations (`01` and `10`) for each action. It does not fit a compositional law. Both direct and model-assisted methods use exactly the same support observations.

The additive world-model predictor uses the three support means to extrapolate:

`p(1,1) = p(0,1) + p(1,0) - p(0,0)`

It predicts both actions, chooses the action with higher predicted reward, and reports a query-cost frontier from 0.00 to 0.16 utility units per model query (two model queries per decision). The direct policy pays no model-query fee. These are transparent synthetic bookkeeping costs, not measured hardware latency or energy estimates. The table below shows the mid-point (0.04/query); the CLI report covers all six cost levels.

## Families and controls

- **Additive positive:** held-out action 1 is genuinely better. This is the in-class transfer case.
- **Interaction only:** the optimal effect appears only when both factors are active. The additive extrapolator is deliberately misspecified on the held-out combination.
- **Additive reversed:** the sign of factor effects is reversed at evaluation. This supplies a stale-model negative control: an older model that predicts action 1 with reward 0.75 remains confident despite the current action-1 reward falling to 0.15.
- **Shuffled model:** swaps the two predicted action values to verify that model/action alignment matters.
- **Oracle evaluator-only policy:** knows the true held-out rewards; it is an upper bound, not an eligible deployable baseline.

The report includes chosen action, query count and cost, model-prediction MAE on the held-out context, mean reward, net utility, regret against the evaluator-only oracle, net regret, and paired net utility lift against the direct baseline with a standard error and approximate normal 95% confidence interval. The paired interval reuses the same reward draw for the model-assisted and direct choices, reducing variance in this synthetic within-family comparison. It does not account for uncertainty across unseen mechanism families or generator specifications.

The current harness generates both candidate decisions and evaluator metrics in one process. It does **not** yet provide candidate/evaluator process isolation, hidden family labels, artifact receipts/digests, a human or external task, or statistical qualification over held-out family draws. Keep every report labeled as a synthetic development result.

## Exploratory fixed-seed results (independent Python mirror, not Rust execution)

The following numbers were reproduced with a Python mirror of the exact SplitMix64 streams and equations in the Rust source. They validate the intended report/test thresholds but do not count as Rust compilation, CI evidence, external benchmark results, or SOTA evidence.

| World family | Strategy | Chosen action | Mean reward | Model query cost | Net utility | Net lift vs direct | Held-out prediction MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| Additive positive | Direct nearest-context | 0 | 0.4976 | 0.0000 | 0.4976 | 0.0000 | — |
| Additive positive | Learned additive model | 1 | 0.7600 | 0.0800 | 0.6800 | **+0.1825** | 0.0112 |
| Additive positive | Shuffled model | 0 | 0.4976 | 0.0800 | 0.4176 | -0.0800 | 0.2524 |
| Interaction only | Direct nearest-context | 0 | 0.4846 | 0.0000 | 0.4846 | 0.0000 | — |
| Interaction only | Learned additive model | 0 | 0.4846 | 0.0800 | 0.4046 | -0.0800 | 0.3159 |
| Additive reversed | Direct nearest-context | 0 | 0.5090 | 0.0000 | 0.5090 | 0.0000 | — |
| Additive reversed | Learned additive model | 0 | 0.5090 | 0.0800 | 0.4290 | -0.0800 | 0.0480 |
| Additive reversed | Stale model control | 1 | 0.1541 | 0.0800 | 0.0741 | **-0.4350** | 0.3000 |

### Query-cost frontier (independent Python mirror, not Rust execution)

For the additive-positive family, the gross reward lift before model-query costs is approximately 0.2625. With two predictions per decision, the world model crosses break-even near 0.1312 utility units per query on this fixed panel.

| Cost per model query | Total model cost | Net lift vs direct |
|---:|---:|---:|
| 0.00 | 0.00 | +0.2625 |
| 0.02 | 0.04 | +0.2225 |
| 0.04 | 0.08 | +0.1825 |
| 0.08 | 0.16 | +0.1025 |
| 0.12 | 0.24 | +0.0225 |
| 0.16 | 0.32 | −0.0575 |

The code now emits all six cost levels, standard errors, and paired 95% intervals. The focused workflow requires the additive-model lift interval to exclude zero at 0.04/query and the expensive cost point's interval to be negative, plus a negative upper interval for the interaction-only control. These values remain exact-seed Python-mirror results, not Rust output or measured real-world latency/energy. The interval is conditional on this single held-out context/family and the stated random-draw process; future qualification must resample mechanism families and generator parameters.

## What this says—and does not say

1. **The world model can earn its cost** when its compositional assumption matches the held-out task: net lift is about +0.18 here.
2. **Model use is not a universal win.** Under the interaction-only mismatch, the predictor does not choose the better held-out action, and its query cost makes it 0.08 worse than direct.
3. **Stale models can be costly.** On the reversed family, the stale model keeps action 1 and loses roughly 0.435 net utility compared with direct fallback.
4. **Prediction MAE is diagnostic, not the primary outcome.** Closed-loop reward and net utility are the headline measures; a model that is wrong on a high-value action can be harmful even if average prediction error looks moderate.

The first smoke test therefore suggests a design principle worth testing in Symthaea: **gate planning/model use on calibrated expected decision lift and explicit model freshness**, and fall back to direct policy when the estimated benefit does not repay model cost or when a mismatch/stale-model signal is present.

## Qualification gates before external claims

- [ ] Exact-head `cargo test --locked` and report-schema workflow pass.
- [ ] Paired episode/family confidence intervals for model-vs-direct net lift, with the family—not each correlated timestep—as the unit of generalization.
- [ ] Development and hidden qualification mechanism families are separate.
- [ ] Query-cost frontier is reported across several latency/compute values rather than using only 0.04 per query.
- [ ] Interaction, shuffled-action, stale-model and shift controls remain in the denominator.
- [ ] Candidate/evaluator separation, immutable artifact/event digests, full failure accounting and mutation tests are added.
- [ ] A real Symthaea adapter is evaluated against a strong direct baseline under matched budgets.
- [ ] External results are reproduced before claiming superiority over industry or SOTA.

## Research alignment

This is directly motivated by recent evaluation work arguing that world models should be assessed by whether they improve decisions, policy ranking, planning, and closed-loop outcomes—not only the quality of predictions or generated artifacts:

- [World-In-World (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html)
- [How Should World Models Be Evaluated? A Decision-Making-Centric Position (2026)](https://arxiv.org/abs/2606.15032)
- [Do World Models Make Better Robots? Survey of Predictive Embodied Intelligence Benchmarks (2026)](https://arxiv.org/abs/2609.29669)
- [TICL, ICML 2026](https://proceedings.mlr.press/v306/chen26t.html), a relevant causal-learning comparator when tasks and metrics are compatible.
