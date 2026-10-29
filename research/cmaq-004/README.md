# CMAQ-004: World-Model Value-Add Under Matched Budgets

**Status:** executable synthetic smoke harness; family-level held-out transfer v0.3 adds gross/net cost analysis and selective-trust margin gates  
**Tracking issue:** [#9010](https://github.com/Luminous-Dynamics/symthaea/issues/9010)  
**Evidence status:** original value-add binary passed exact-head Rust workflow at [`9eddcf5`](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40950457324); family-transfer companion and the integrated workflow passed exact-head run [#19 at `0a51bd2`](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40960013206): 7 original tests, 6 family-panel tests, and both report validators passed. Both panels remain synthetic smoke evidence, not blind qualification.  
**Claim ceiling:** tiny finite held-out-composition harness only; no industry/SOTA, broad causal reasoning, or production Symthaea claim

## Research question

Does a learned action-conditioned model improve *closed-loop choice* enough to justify prediction cost, compared with a strong direct policy that uses the same support observations but does not fit a compositional reward law?

Prediction accuracy and decision value are different claims. A model can predict well but add too little utility to pay for inference; it can also fit training/support data yet fail on a held-out interaction or after a regime reversal.

This smoke harness isolates that bridge on a tiny, known-ground-truth family. It is not yet an integration with Symthaea's HDC/CfC/LTC, memory, world model, or planner.

## Executable scope

The dependency-free Rust CLI is pinned to Rust 1.96.0:

```sh
cargo test --locked
cargo run --quiet --locked --bin cmaq-004-world-model-value-add
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

## Exact-head Rust smoke results (synthetic, not blind qualification)

The table below reproduces the deterministic JSON emitted by the Rust binary in the exact-head focused workflow linked above. This confirms these numbers are Rust smoke outputs for the declared synthetic panel. It does not establish performance on unseen mechanism families, measured hardware costs, external environments, or SOTA.

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

### Query-cost frontier (exact-head Rust smoke output)

For the additive-positive family, the gross reward lift before model-query costs is approximately 0.2625. With two predictions per decision, the world model crosses break-even near 0.1312 utility units per query on this fixed panel.

| Cost per model query | Total model cost | Net lift vs direct |
|---:|---:|---:|
| 0.00 | 0.00 | +0.2625 |
| 0.02 | 0.04 | +0.2225 |
| 0.04 | 0.08 | +0.1825 |
| 0.08 | 0.16 | +0.1025 |
| 0.12 | 0.24 | +0.0225 |
| 0.16 | 0.32 | −0.0575 |

The code emits all six cost levels, standard errors, and paired 95% intervals. The exact-head focused workflow requires the additive-model lift interval to exclude zero at 0.04/query and the expensive cost point's interval to be negative, plus a negative upper interval for the interaction-only control. These are deterministic synthetic Rust outputs—not measured real-world latency/energy. The intervals condition on this single held-out family/context and do not quantify generalization across mechanism families. The interval is conditional on this single held-out context/family and the stated random-draw process; future qualification must resample mechanism families and generator parameters.

## What this says—and does not say

1. **The world model can earn its cost** when its compositional assumption matches the held-out task: net lift is about +0.18 here.
2. **Model use is not a universal win.** Under the interaction-only mismatch, the predictor does not choose the better held-out action, and its query cost makes it 0.08 worse than direct.
3. **Stale models can be costly.** On the reversed family, the stale model keeps action 1 and loses roughly 0.435 net utility compared with direct fallback.
4. **Prediction MAE is diagnostic, not the primary outcome.** Closed-loop reward and net utility are the headline measures; a model that is wrong on a high-value action can be harmful even if average prediction error looks moderate.

The first smoke test therefore suggests a design principle worth testing in Symthaea: **gate planning/model use on calibrated expected decision lift and explicit model freshness**, and fall back to direct policy when the estimated benefit does not repay model cost or when a mismatch/stale-model signal is present.

## Family-level transfer companion (v0.2)

The `family_transfer` Rust binary widens the unit of analysis from repeated reward draws on one fixed law to **256 generated mechanism families** and, in v0.3, compares three selective-trust margin gates. For each family, the candidate receives support-outcome means for contexts `00`, `01`, and `10`; context `11` remains evaluator-only. The direct nearest-context policy and additive extrapolator use the same support outcomes. The generator balances three lanes: near-additive laws, strong held-out interaction laws, and a held-out sign-shift stress lane.

Run it explicitly (the crate now has two binaries):

```sh
cargo test --locked
cargo run --quiet --locked --bin cmaq-004-world-model-value-add
cargo run --quiet --locked --bin family_transfer
```

The companion crosses eight strategies—direct nearest-context, additive model, three margin gates, shuffled-model control, stale-model control and evaluator-only oracle—with four query-cost levels and four reporting groups (all families plus each mechanism lane). Its 128 JSON rows (schema `cmaq-004-family-transfer-v0.3`) report:

- mean **gross** lift before model-query fees, mean **net** lift after fees, and implied break-even fee per query (gross lift divided by two queries), with an approximate 95% interval on net lift at the generated-mechanism-family unit;
- fraction of families where the strategy is worse than the direct comparator;
- lower-tail CVaR05 of family-level lift and the single worst family;
- fraction of families showing positive gross lift before cost, action-one rate, query cost, action override rate versus direct, and (for additive/shuffled/margin-gated policies) held-out prediction MAE.

The three margin gates accept the additive-model action only when it disagrees with the direct comparator and its predicted advantage over that direct action is at least 0.005, 0.020, or 0.050 respectively. A stricter margin must never override direct more often than a looser margin. Because the margin itself is computed from model outputs, **all gate variants still incur the two-query inference fee**, even when they abstain and preserve the direct action. This deliberately tests decision selectivity separately from inference-cost savings; the latter would require a cheap pre-query uncertainty signal or cached model output.

The most important new question is not just whether the *average* family benefits, but whether the model's average advantage masks a bad tail or fails under interaction/held-out-shift lanes. The cost sweep charges two model queries per decision; this is declared synthetic utility accounting, **not** measured wall-clock latency, energy, or hardware cost.

The support-sample RNG is keyed by the family slot and observed context/action cell, not by hidden interaction or held-out-shift parameters. A regression test ensures that mechanisms with identical support laws produce identical support tapes under the same sample seed, even when their held-out laws differ. This prevents a subtle side-channel where held-out parameters could change observed sampling noise.

### Exact-head family-panel result and break-even cost (v0.2)

At 0.04 utility units per query (two model queries, 0.08 total cost per decision), the **additive-model policy has negative family-level net lift vs direct in all four reporting groups** on this deterministic 256-family sample. The v0.2 report also separates gross policy lift from query fees. At zero query cost:

| Lane | Families | Mean net lift | Approx. family-level 95% interval | Lower-tail CVaR05 | Families harmed |
|---|---:|---:|---:|---:|---:|
| All | 256 | -0.0927 | [-0.1056, -0.0797] | -0.4535 | 95.3% |
| Near-additive | 86 | -0.0770 | [-0.0839, -0.0701] | -0.1435 | 96.5% |
| Interaction | 85 | -0.0655 | [-0.0890, -0.0420] | -0.3218 | 89.4% |
| Held-out shift | 85 | -0.1357 | [-0.1640, -0.1073] | -0.5383 | 100.0% |

The following gross-lift figures and break-even prices are the exact Rust output from workflow run [40960943585](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40960943585):

| Lane | Mean gross lift before fees | Approx. family-level 95% interval at zero fee | Break-even cost per query | Families with positive gross lift |
|---|---:|---:|---:|---:|
| All | -0.0127 | [-0.0256, +0.0003] | -0.00633 | 7.0% |
| Near-additive | +0.0030 | [-0.0039, +0.0099] | +0.00150 | 10.5% |
| Interaction | +0.0145 | [-0.0090, +0.0380] | +0.00725 | 10.6% |
| Held-out shift | -0.0557 | [-0.0840, -0.0273] | -0.02783 | 0.0% |

A negative break-even fee means the mean policy is worse than direct even if queries are free; it would require a subsidy or other unmodeled benefit to tie the comparator. The small positive mean break-even in the additive lane is not a robust win: its family-level interval crosses zero and only 10.5% of families show positive gross lift. In the interaction lane the mean gross lift is higher but still uncertain across families, and the maximum break-even query cost is only about 0.00725 utility units—far below the 0.04/query test point.

The net-lift table above remains the exact Rust output from earlier integrated run [40960013206](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40960013206); the latest v0.2 workflow passed the new gross/net/break-even identities and all family validations at [40960943585](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40960943585). The interval unit is generated family, not a repeated reward timestep. This is a meaningful negative result: under this mixed family distribution and declared cost, the direct baseline is better on average, even on the near-additive lane. The correct response is **not** to retune the family mix until the model wins; it is to report the failure regions, improve decision-value/freshness gating or the model itself, and test against action-coupled environments. This public synthetic panel is not external qualification or SOTA evidence.

**Evidence boundary:** candidate and evaluator are still in one synthetic process; the generator and its public specification are visible. The report is a development/smoke study, not sealed qualification. The normal intervals are approximate over this finite generated family sample and do not establish population-wide transfer. The next gate is separate candidate/evaluator execution with frozen candidate artifacts, blinded mechanism keys, and an independently maintained environment.

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
