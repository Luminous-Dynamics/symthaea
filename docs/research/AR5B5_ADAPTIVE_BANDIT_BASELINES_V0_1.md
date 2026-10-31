# AR-5B5 — adaptive bandit baselines v0.1

**Status:** two deterministic-seed, one-action synthetic policy baselines implemented; tests authored but not compiled or executed.  
**Tracking issue:** [#9392](https://github.com/Luminous-Dynamics/symthaea/issues/9392)  
**Implementation:** \`crates/domains/symthaea-architecture-engineering/src/direction_board_benchmark.rs\`  
**Campaign foundation:** [AR-5B direction board](./AR5B_DIRECTION_BOARD_V0_1.md)  
**Round contract:** [AR-5B4](./AR5B4_MATCHED_ROUND_SEMANTICS_V0_1.md)

## Why add these baselines

The existing comparison includes score-only, fixed round-robin, seeded random, global-score UCB, Symthaea's verifier-grounded direction board, and a synthetic oracle. It does not yet include an adaptive policy that discounts stale evidence or a posterior-sampling baseline.

Two useful controls are:

- **Discounted UCB** for rank reversals, changing payoff regimes, and non-stationary evidence.
- **Gaussian Thompson sampling** for uncertainty-aware arm selection using a frozen Normal-Normal reward model.

These are necessary controls before interpreting DirectionBoard performance. They are one-action-per-step policies in the existing synthetic benchmark. They are not four-seat Coral replications and not complete autonomous-search systems.

## Implemented policies

### Discounted UCB

The policy forms recency-weighted reward and effective-count estimates over prior actions. For an action \(j\) with age \(a_j\) and reward \(r_j\):

\[
w_j = \gamma^{a_j}, \qquad
S_i = \sum_{j \in i} w_j r_j, \qquad
N_i = \sum_{j \in i} w_j
\]

The v0.1 frozen discount is \(\gamma=0.90\). Direction arms use a 0.5 prior reward; NEW uses the same declared \`NEW_DIRECTION_CREDIT\` prior as the existing global-UCB baseline. The potential is:

\[
\hat \mu_i = \frac{S_i + \mu_{0,i}}{N_i + 1},
\qquad
U_i = \hat \mu_i +
c\sqrt{\frac{2\ln(T+2)}{N_i+1}}
\]

where \(c\) reuses the existing frozen exploration coefficient, and \(T\) counts prior actions. Ties are resolved lexically for determinism.

This is an explicit discounted-UCB heuristic, not a proof-optimal non-stationary bandit. The fixed discount should be preregistered and tested against multiple change rates; do not tune it on the confirmatory holdout.

### Gaussian Thompson sampling

The policy uses an independent Normal-Normal posterior for each arm. Reward is normalized to [0,1]; failed or non-admitted NEW proposals contribute zero, while an admitted NEW proposal earns the same fixed \`NEW_DIRECTION_CREDIT\` as the global-UCB control. That shared credit is important: an admission must not receive a hidden reward-scale advantage in one baseline.

For prior mean \(\mu_0\), prior variance \(\sigma_0^2\), observation variance \(\sigma^2\), and rewards \(r_1,\ldots,r_n\), the posterior is:

\[
\tau_n = \frac{1}{\sigma_0^2}+\frac{n}{\sigma^2},
\qquad
\mu_n = \frac{\mu_0/\sigma_0^2+\sum_j r_j/\sigma^2}{\tau_n},
\qquad
\sigma_n^2 = \frac{1}{\tau_n}
\]

Frozen v0.1 assumptions:

- prior mean 0.5 for existing directions;
- prior mean \`NEW_DIRECTION_CREDIT\` for NEW;
- prior variance 0.25;
- known observation variance 0.25;
- deterministic SplitMix64-seeded Box–Muller normal sampling.

This is a Gaussian approximation to bounded [0,1] rewards. It is not a calibrated posterior guarantee, and the sampled latent mean is not constrained to [0,1]. The variance assumptions are engineering baseline parameters, not estimated from the hidden landscape. A Beta/empirical-Bernstein or heteroscedastic model should be evaluated as a separate baseline, not silently substituted.

## Test coverage authored

Twelve tests are authored in the synthetic campaign module. New coverage verifies:

- the policies are included in exact unit-action budget conservation across the existing development landscapes;
- identical seeds reproduce identical campaigns for both policies;
- discounted evidence can recover after a rank reversal;
- Gaussian posterior mean moves toward observed rewards and variance contracts with evidence;
- NEW admission credit remains on the same frozen scale as global UCB;
- neither adaptive baseline manufactures a novel direction on the no-new-direction null.

Existing matrix-level paired-seed checks also include both policy IDs. These tests are authored but have **not** been compiled or executed in the authoring environment.

## What this does and does not compare

The policies receive the same deterministic synthetic payoff schedule and each selection consumes one unit-cost action. A NEW proposal also consumes one action slot and is accounted for as a proposal; a later evaluation in an admitted direction consumes a separate slot.

This preserves the existing synthetic campaign's action-count model. It is **not yet** the four-seat synchronous execution contract in AR-5B4. In particular, this tranche does not implement:

- four concurrent seats or pre-round frozen snapshots;
- Coral's exact four-agent claim-cap orchestration;
- atomic multi-worker resource reservation;
- MAP-Elites/CVT-MAP-Elites descriptors or archive replacement;
- a real LLM/agent proposal stack or independent semantic direction verifier.

Do not compare these one-step policies directly against a four-seat Coral run until all are driven through the common AR-5B4 round/action/cost contract.

## Development versus confirmatory benchmark

The current four landscape families—delayed breakthrough, no-new null, new-operator breakthrough and rank reversal—are development fixtures. They are not enough to support general claims. The confirmatory target remains at least 100 paired payoff instances across at least six independent landscape families, with frozen seeds and algorithm parameters.

Add later families for deceptive proxies, multiple resource optima, direction duplication, delayed operator invention, cross-task regression and abrupt/gradual drift. Report all families, including failures and negative controls. Cluster uncertainty by independent landscape/payoff template, not by individual action.

Primary metrics remain verified best-so-far capability and simple regret versus total charged cost, target-hit rate versus budget, cost/time to first verified gain, worst-family performance, NEW precision/duplicate/rejection cost, diversity of verifier-admitted directions, and multi-resource Pareto performance.

## Competitive positioning

The stronger comparison is not "our new bandit beats the old bandit." It is a staged evidence chain:

1. exact policy semantics and reproducible synthetic test fixtures;
2. matched scheduler mechanisms under common four-seat round semantics;
3. end-to-end direction discovery with proposal and evaluation costs separated;
4. full practical systems compared on held-out tasks, identical model/tool budgets and total resource cost;
5. independent replication and transfer to tasks not used to tune the search policy.

The benchmark should include global UCB, discounted UCB, Gaussian Thompson sampling, Coral's faithful scheduler, DirectionBoard, fixed/random controls, quality-diversity methods where descriptors are justified, and a non-deployable oracle on synthetic landscapes. Best-of-N may be shown as a secondary metric, but the primary result must include all predeclared seeds rather than reporting only the best run.

## Claim ceiling

This tranche adds two synthetic policy baselines and authored tests. It does not establish SOTA, superiority, production scheduling quality, an actual Coral reproduction, novel operator invention, or autonomous scientific discovery.
