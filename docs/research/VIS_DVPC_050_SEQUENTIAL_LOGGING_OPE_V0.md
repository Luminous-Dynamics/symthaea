# VIS-DVPC-050 — Sequential logging OPE and target-drift falsification protocol v0

**Status:** implementation-in-progress research harness. Not a method reproduction, statistical qualification, or production authorization.

## Research gap

VIS-DVPC-046/#8909 and the fixed-horizon comparator work (#9011) evaluate confidence bounds and policy selection at preregistered looks. They do not directly evaluate a policy-update loop where the policy selected from prior observations determines the next logger and therefore the distribution of later observations.

The first experiment here is deliberately modest: expose the adaptive data-generating process, log the exact behavior propensity for every event, separate current from historical estimands, and include a stale-propensity negative control. Only after that substrate is trustworthy should it be used to reproduce the sequential LS / adjusted LS algorithm.

Relevant research:
- Haddouche & Sakhi, **Sequential Off-Policy Learning with Logarithmic Smoothing**, AISTATS 2026: https://proceedings.mlr.press/v300/haddouche26a.html
- Haddouche & Sakhi, **Logarithmic Smoothing for Adaptive PAC-Bayesian Off-Policy Learning**, 2025 preprint: https://arxiv.org/abs/2506.10664
- Sakhi et al., **Logarithmic Smoothing for Pessimistic Off-Policy Evaluation, Selection and Learning**, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/9379ea6ba7a61a402c7750833848b99f-Abstract-Conference.html
- Ryu et al., **Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing**, COLT 2025: https://proceedings.mlr.press/v291/ryu25a.html
- Broader exact-known-answer estimator, misspecification, and support stress tests: [VIS-DVPC-028 / #8730](https://github.com/Luminous-Dynamics/symthaea/issues/8730)
- Fixed-horizon policy portfolio benchmark: [VIS-DVPC-046 / #8909](https://github.com/Luminous-Dynamics/symthaea/issues/8909), [PR #9011](https://github.com/Luminous-Dynamics/symthaea/pull/9011)

## Experimental harness

Implementation: `tools/vision-dvpc-050-sequential-logging-ope.rs`, dependency-free Rust.

Each replicate runs a multi-round contextual-bandit deployment loop:
- Two contexts with known probability 1/2 each; two actions per context; Bernoulli rewards with exact potential-outcome means.
- Policy portfolios `M={4,8,32}`; common seed schedule for matched comparisons.
- At each round, the selected incumbent policy plus a declared exploration mixture determines the next logger. The logger changes as a consequence of the prior rounds.
- Each observation stores context, action, reward, round, incumbent/logger policy ID, and the **actual action propensity used at that event**.
- Stationary rewards with a changing logger, linear reward drift with an optimum reversal, weak overlap, and a structural-zero-support control.
- One large cumulative history versus a preregistered two-round recent window, both using row-wise propensities; an intentionally stale-propensity negative control that reuses the first logger's probabilities.

The negative control is a bug detector, not a candidate method. The recency window is a tracking baseline under drift, not a universally valid estimator of a stationary target. No intervals are currently claimed, so no sequential-coverage claim is emitted by this harness.

## Estimands must remain distinct

For target policy π at round (t), let (V_t(π)) be its value under that round's reward means. The harness reports:
- **Current-round value** (V_t(π)).
- **Pooled-history value** ((t+1)^{-1}∑_{s=0}^{t} V_s(π)), because batches have equal size.
- **Recent-window value** averaged over the registered last two rounds (or fewer near initialization).

The row-wise IPS point estimate in a stationary reward world uses the actual logging probability for each row. Under reward drift, the estimator for pooled historical data targets a historical mixture, not the current value; calling those the same target would make the comparison invalid. The recent-window estimator is evaluated against the recent-window estimand, not treated as a free correction for drift.

## Support / fail-closed boundary

Before producing numeric policy estimates for a portfolio, verify that the behavior logger had positive support for every action/context pair receiving positive target-policy mass in the selected data window. If not, emit no value and no selected policy for that replicate/look. In the structural-zero arm, the target portfolio is not identified and the correct behavior is abstention; finite estimates from unsupported actions must not be surfaced as facts.

Weak support remains distinct from zero support: estimates can be identifiable but high-variance. Report maximum importance weights and ESS fraction rather than letting a seemingly good point estimate override the support and variance evidence.

## Metrics and output

The program emits a stable TSV-like stream with explicit configuration rows and per-scenario, portfolio, method, round metrics:
- selection / abstention rate and best-policy top-1 accuracy conditional on selection;
- current-round regret;
- absolute error to current, pooled, and recent estimands separately;
- maximum importance weight and effective-sample-size fraction.

Run configurations:
```sh
rustc --edition=2024 -O --test tools/vision-dvpc-050-sequential-logging-ope.rs -o /tmp/vis-dvpc-050-tests
/tmp/vis-dvpc-050-tests
rustc --edition=2024 -O tools/vision-dvpc-050-sequential-logging-ope.rs -o /tmp/vis-dvpc-050
/tmp/vis-dvpc-050 --smoke
/tmp/vis-dvpc-050 > /tmp/vis-dvpc-050-full.tsv
```

The full target is 128 replicates per scenario/portfolio/method with 128 logged events per round; smoke uses 4 replicates and 8 events per round. These are engineering configuration choices, not standard sample sizes or power claims.

## Preregistered qualification gates

1. Rust tests pass and exact-head execution is recorded, with stable output on repeated smoke runs.
2. Row-wise propensity use matches independent expectations; the stale-propensity negative control differs on a known-answer fixture.
3. The incumbent actually changes future logging probabilities, with the chosen policy/version recorded for every event.
4. The structural-zero scenario emits zero selections and full abstention; it must never emit a numeric value or a certificate for unsupported policies.
5. Output rows are rectangular, finite where a numeric estimate exists, and have stable counts. Missing values are explicit `NA`, not zero.
6. Reward-drift evaluation reports its current, pooled, and recent targets separately. No theorem for stationary outcomes is silently applied to drifting rewards.
7. Coverage and false-promotion targets (at least 95% coverage, at most 5% false promotion where formally claimed) are evaluated only after the relevant inference method and monitoring protocol have been implemented and justified. Current harness makes no such claim.
8. Compare every method at equal total interaction budget and paired random seeds. Report Monte Carlo uncertainty for proportions and paired uncertainty for differences; no ranking from a single seed or a composite score.
9. Before calling this a SeqLS / adjusted SeqLS reproduction, add an equation-by-equation paper-to-code map, explicit assumptions/deviations, independently derived fixtures, and any available reference-code comparison. The published 2024 LS baseline and the 2026 sequential method are not interchangeable.
10. External data validation requires pinned data/configuration and a license/eligibility review, particularly before any commercial use.

## Hypotheses to test (not results)

- Correct row-wise propensity accounting should be invariant to changes in the logging policy when rewards are stationary and support holds.
- Reusing stale propensities should fail a constructed known-answer test and can introduce bias after logger updates.
- During reward drift, cumulative data may be more precise about a pooled historical target while recent-window estimation can track the current target faster; this is a bias/variance and lag trade-off to measure, not assume.
- Exploration creates an identifiable support/variance frontier: shrinking exploration can reduce immediate cost while increasing maximum weights, lowering ESS, or making comparisons unidentified.

A result counts as improvement only if it moves a preregistered risk–regret–adaptation–compute frontier under matched interaction budgets without hiding abstention or support failures.

## Evidence boundary

The source, test set, and configuration define the benchmark scaffold. Until exact-head Rust tests and the full deterministic replay succeed, the benchmark has no execution receipt. Even after those pass, the harness is synthetic-methodology evidence—not a real-data SOTA result, a proof of coverage, or deployment authorization.
