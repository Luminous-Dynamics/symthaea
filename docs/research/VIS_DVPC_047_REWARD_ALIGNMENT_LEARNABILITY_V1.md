# VIS-DVPC-047 — Reward alignment, online learnability and personalization headroom

**Status:** deterministic synthetic benchmark under exact-head Rust qualification. No statistical results are qualified until CI completes.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8915

## Research question

A single offline policy value cannot answer all of these questions:
1. Does a proxy reward rank choices in the same direction as the real north-star outcome?
2. Can a contextual learner identify a good proxy-optimal policy quickly enough?
3. Is context-dependent personalization better than choosing one global action?
4. How much of the delayed north-star evidence is available by the decision deadline?
5. Once the learner has trained on proxy feedback, how noisy is an independent offline estimate of its current greedy policy?

These are separate estimands. The benchmark reports them separately instead of treating proxy reward lift as business-outcome lift.

This follows the diagnostic distinction in Lee et al., *When Offline Evaluation Misleads: A Diagnostic Protocol for Reward and Policy Selection in Delayed-Feedback Contextual Bandits* (2026 workshop paper / arXiv:2608.11560). The authors highlight reward alignment versus learnability, interval miscalibration, and the possibility that a “personalization premium” is just robustness when the best global action is hard to identify:
https://arxiv.org/abs/2608.11560

## Frozen ground truth

The environment has two equally likely contexts and four actions. The north-star Bernoulli means for contextual personalization fixtures are:

| Context | Action 0 | Action 1 | Action 2 | Action 3 |
|---|---:|---:|---:|---:|
| 0 | 0.35 | 0.55 | 0.75 | 0.65 |
| 1 | 0.80 | 0.60 | 0.40 | 0.55 |

The context-specific optimal actions differ: action 2 in context 0 and action 0 in context 1. Therefore the oracle contextual value is 0.775. The best single global action is action 3 at value 0.600, giving true personalization headroom 0.175.

Five registered fixtures isolate the mechanisms:
- **aligned dense:** proxy cell means equal the north-star means, feedback is immediate;
- **aligned sparse:** proxy means are 0.12 times the north-star means, preserving the rank ordering but making positive proxy rewards less frequent;
- **misaligned dense:** proxy mean is 1 minus the north-star mean, deliberately reversing every pairwise action ranking within each context;
- **zero personalization headroom:** both contexts have the same north-star action means, so the best context-dependent policy and best global action have identical true value;
- **aligned sparse, delayed:** same aligned-sparse reward structure, but 70% of north-star feedback matures after 4 steps and 30% after 512 steps.

All outcome means, proxy means, action rankings, alignment, global-best value, contextual-oracle value and personalization headroom are computed directly from the fixture.

## Online learner and independent OPE log

A tabular contextual UCB learner receives the proxy reward immediately and chooses actions separately for each context. Every decision also produces a north-star outcome that matures according to the fixture’s delay schedule. The reported learner value and regret are calculated from the **true expected north-star mean of the chosen action**, so they do not need to infer business performance from proxy outcomes.

Separately, an independent uniform-randomized action log is used to evaluate the learner’s currently estimated greedy proxy policy. The target policy is fixed conditional on the online training stream, and the evaluation data use a separate seed. The known logger assigns probability 1/4 to each action, so for each of the 16 deterministic context-to-action policies the IPS score is 4R when the logged action matches that policy and 0 otherwise; every score lies in [0,4].

Two interval tracks are reported over the **north-star outcomes that have matured by that look**:

- The pointwise 95% Wald interval is a conventional variance-based diagnostic baseline. Reusing it across looks or selecting the currently learned policy does not give a time-uniform guarantee.
- The familywise Hoeffding confidence sequence maintains one nested sequence for each of the 16 deterministic policies, with alpha/16 allocated per policy and two-sided telescoping alpha spending over the count of matured observations. Every policy score is bounded by 4. The learner’s greedy policy may change over wall-clock time because of its independent training stream, but the simultaneous event that all 16 CSs cover their values at all maturity counts also covers whichever policy it selects. The evaluation log is processed in maturity order; this construction relies on the frozen fixture’s outcome-independent delays, known uniform action propensities and stationary reward means.

## Measurements

For each fixture and looks 128, 512 and 4,096, the output reports:
- expected north-star value attained by the contextual UCB choices;
- north-star regret per decision versus the context-specific oracle;
- expected proxy reward value attained and proxy regret per decision;
- fraction of contexts where the greedy proxy estimate identifies the true proxy-optimal action;
- fraction of contexts where that same estimated greedy action matches the north-star-optimal action;
- online value minus the best global-action value (with the true headroom emitted separately);
- online north-star outcome maturation fraction and mean among matured outcomes;
- absolute IPS error for the independently evaluated learned-greedy policy;
- pointwise 95% Wald coverage and interval width for that policy;
- fraction of independent logged outcomes that have matured by the look.

These are Monte Carlo means over 1,024 seeded trajectories; standard errors are emitted alongside each metric. The alignment sign-accuracy, true personalization headroom, oracle values and best-global-action value are deterministic fixture properties and are emitted separately from empirical estimates.

## Correctness controls

- SplitMix64 seed schedule is fixed; replay compares all statistical output and excludes wall-clock rows.
- Online regret uses context-specific oracle value and is nonnegative by construction.
- North-star reward feedback is generated and matured independently of proxy reward feedback; the learner only observes the proxy reward.
- The IPS evaluation log is independent of the online training stream and uses known uniform action propensities.
- The delayed evaluation estimator only uses north-star rewards matured by the registered look; its denominator is the number matured, not the total log length.
- The zero-headroom fixture asserts its exact personalization headroom is zero, and its expected online north-star value cannot exceed the best global action value.
- The deliberately misaligned proxy fixture must have pairwise alignment accuracy below 0.01; aligned dense and sparse fixtures preserve the same action ordering.
- The workflow checks exact row counts, rate ranges, nonnegative regret/error/width, empirical familywise-CS coverage, deterministic replay and source/output SHA-256 receipts.

## Evidence boundary and industry comparison

This is a synthetic diagnostic protocol, not a production result and not a state-of-the-art claim. The independent pointwise Wald interval is purposely not time-uniform. A future extension should add time-uniform OPE intervals, delayed-feedback correction when maturation depends on context/action, covariate/reward drift controls, and public-data replay.

SCOPE-RL benchmarks OPE and off-policy selection using estimators including IPS variants, doubly robust estimators and high-confidence Hoeffding / empirical-Bernstein intervals; its evaluation protocol reports MSE, rank correlation, regret, Type I/II errors, top-k policy performance and safety violations:
https://scope-rl.readthedocs.io/en/stable/documentation/index.html

Recent contextual-bandit theory also keeps *learnability* separate from fixed-policy value estimation: Qin and Zhang’s 2026 OE2D framework studies the coverage/exploitation trade-off and efficient calls to a regression oracle:
https://proceedings.mlr.press/v336/qin26a.html

The target for VIS-DVPC-047 is not one score that dominates a leaderboard. It is a falsifiable decomposition of why a proxy/policy succeeds or fails: alignment, learning speed, personalization headroom, and usable delayed evidence.