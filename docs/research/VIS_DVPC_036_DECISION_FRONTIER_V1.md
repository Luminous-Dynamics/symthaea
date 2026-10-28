# VIS-DVPC-036 — Exact-DP frontier for sequential OPE decision efficiency

**Status:** implementation draft; exact synthetic finite-sample benchmark only.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8803

## Research question

VIS-DVPC-034 asks whether fixed-time OPE intervals are calibrated. VIS-DVPC-035 demonstrates that repeatedly reusing fixed-time intervals can fail under optional stopping, while a conservative anytime-valid baseline may become nearly vacuous under weak overlap. The next decision-facing question is:

> At a predeclared error budget, how often and how quickly can an evaluator cross a policy-value threshold, and what width/calibration do we pay for that decision?

A single terminal RMSE or coverage number hides decision latency. This benchmark exposes the reliability–width–power–latency frontier.

## Paradigm shift: aggregate equivalent trajectories exactly

For a fixed logging policy with action-A probability p, fixed target policy with probability q, and Bernoulli reward means mu_A and mu_B, one logged IPS contribution is:

- X = q/p when A is logged and its reward is 1;
- X = (1-q)/(1-p) when B is logged and its reward is 1;
- X = 0 for either action when its reward is 0.

The three probabilities are p*mu_A, (1-p)*mu_B, and 1 - p*mu_A - (1-p)*mu_B. For this frozen model, the sample mean and sample variance depend only on the number of A-success contributions and B-success contributions. Their order does not matter. The DP can therefore merge histories with identical count states.

At look t there are (t+1)(t+2)/2 reachable count states. The program evaluates O(H^2) states per look, O(H^3) cumulative state visits/transitions over horizon H, and O(H^2) memory. This extends exact-distribution auditing from the 4^8 path count in VIS-DVPC-035 to horizon 256 without claiming that arbitrary contextual or adaptive logging processes admit this compression.

Each look's probability mass must sum to one within the stated floating-point tolerance. The program emits reachable-state counts and mass at each look so a faulty aggregation cannot silently inflate apparent power.

## Methods

### WALD_FIXED_TIME

A nominal 95% interval centered on the IPS sample mean, using sample variance and the normal quantile 1.9599639845. It is a deliberately weak comparator: checking it repeatedly does not grant sequence-wide 95% coverage.

### ALPHA_SPENDING_HOEFFDING_CS

A transparent anytime-valid baseline using a known upper bound W on each IPS contribution:

alpha_t = alpha / (t(t+1)), with sum alpha_t = alpha,

radius_t = W * sqrt(log(2/alpha_t) / (2t)).

This assigns a summable error budget over looks. Under the frozen iid, fixed-logger, correctly specified propensity and bounded-reward assumptions, the union bound controls simultaneous two-sided failure by alpha. It does not adapt to empirical variance, so under weak overlap its width can approach the full [0,1] range.

This is **not** a faithful implementation of the published Off-Policy Confidence Sequences (OFCS) algorithm; a direct implementation and comparison remain open work.

## Frozen design

The fixture includes a horizon-8 healthy null to reproduce the previous exact path-enumeration known answers, followed by horizon-256 cases:

| Scenario | Logger P(A) | Reward means (A, B) | Target value | Threshold |
|---|---:|---:|---:|---:|
| healthy null | 0.50 | (0.20, 0.80) | 0.50 | 0.50 |
| small positive margin | 0.50 | (0.30, 0.80) | 0.55 | 0.50 |
| larger positive margin | 0.50 | (0.40, 0.80) | 0.60 | 0.50 |
| weak-overlap null | 0.99 | (0.20, 0.80) | 0.50 | 0.50 |
| weak-overlap positive margin | 0.99 | (0.30, 0.80) | 0.55 | 0.50 |

The target policy is q=0.5 in these scenarios. The horizon-8 control has truth 0.5; independent enumeration in VIS-DVPC-035 gives fixed-time Wald sequence coverage 0.4375 and ever-false-promotion probability 0.28125 for looks n=2..8. Its alpha-spending baseline has sequence coverage 1.0 for n=1..8 in this finite control. The Rust DP's unit tests require those known answers within 1e-9.

The CI executes all horizon-256 scenarios from the frozen fixture; their per-look coverage, widths, crossing probabilities and conditional first-crossing times are emitted in the machine-readable TSV and artifact receipt.

## Metrics and interpretation

For every eligible look and method, the TSV reports:

- **pointwise_coverage:** probability that this look's interval contains the known target value;
- **mean_interval_width:** exact expected clipped interval width under the frozen DGP;
- **pointwise_threshold_crossing_probability:** probability its lower bound is above the threshold at this look;
- **first_threshold_crossing_probability:** probability this is the first look whose lower bound exceeds the threshold;
- **ever_threshold_crossing_probability:** cumulative probability of having crossed by this look;
- **survival_probability:** probability of not having crossed yet;
- **sequence_coverage_to_look:** probability the interval covered truth at every eligible look through this look.

At sequence end, the report adds sequence-wide coverage, cumulative crossing probability, and mean crossing look conditional on a crossing (or NOT_REACHED when no path crosses). These are distinct measures:
- when threshold equals truth, crossing is a **false-promotion** event;
- when truth is above threshold, crossing probability is **decision power**;
- conditional latency must always be read alongside crossing probability, or a method that almost never decides could look artificially fast.

Every look's exact DP probability mass, number of reachable states, total state-evaluation count, and the explicit `path_enumeration=NOT_USED` marker are recorded.

## Evidence and limitations

- Synthetic two-action Bernoulli DGP only; no real trace, production, or SOTA claim.
- Formal interpretation relies on independent stationary rows, a fixed known logger, correctly specified propensities, bounded rewards, and a fixed target policy.
- This DP does not cover adaptive contexts, dependent observations, reward drift, outcome-adaptive logging, hidden confounding, propensity misspecification, or zero support. Separate support gates remain mandatory.
- Wald is an intentionally unsafe optional-stopping comparator. The alpha-spending Hoeffding CS is a conservative transparent baseline, not the sharpest known method.
- A faithful published OFCS comparator should next be evaluated on the same frozen DGPs, then expanded only with explicit controls for changing loggers, contexts and reward drift.

## References

- Karampatziakis, Mineiro & Ramdas, *Off-Policy Confidence Sequences*, ICML 2021: https://proceedings.mlr.press/v139/karampatziakis21a.html
- Waudby-Smith et al., *Anytime-valid off-policy inference for contextual bandits* (2022): https://arxiv.org/abs/2210.10768
- SCOPE-RL HCOPE implementation notes: https://scope-rl.readthedocs.io/en/latest/documentation/evaluation_implementation.html
