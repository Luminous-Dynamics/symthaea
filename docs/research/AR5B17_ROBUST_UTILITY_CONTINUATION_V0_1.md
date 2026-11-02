# AR-5B17: Robust Alternative-Set Decision Utility with Evaluator-Cost Constraints v0.1

**Status:** finite reference draft; seven tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → #9814 → #9823 → #9825 → #9854 → #9856 → #9860 → this branch.  
**Tracking:** #9893.  
**Research anchors:**
- Chick, Forster & Pertile (2017), *A Bayesian Decision Theoretic Model of Sequential Experimentation with Delayed Response*, JRSSB 79(5), 1439–1462: https://doi.org/10.1111/rssb.12225
- Johari et al. (2021), *Always Valid Inference: Continuous Monitoring of A/B Tests*, Operations Research 70(3), 1806–1821: https://doi.org/10.1287/opre.2021.2135
- Spotify Engineering, *Choosing a Sequential Testing Framework* (updated 12 May 2026): https://confidence.spotify.com/blog/sequential-testing-comparison
- Spotify Confidence, *Accurate Sample Size for Always-Valid Inference* (14 July 2026): https://confidence.spotify.com/blog/accurate-sample-size-always-valid

Spotify's July 2026 report is a useful current industry benchmark: its corrected sample-size multiplier reduced required N by 8–20% in its simulation grid (14% at common alpha=0.05 / 80%-power WSKR settings) and delivered a median 9.5% saving across 713 metrics from 283 real experiments. It also reports empirical power within approximately three percentage points of target over its simulation grid. Treat these as Spotify's own reported results, not independent verification. Our benchmark should compare against this regime-specific production baseline with identical power, alpha, sample horizon, and cost accounting.

## 1. Research shift: optimize robust decision utility, not power alone

AR-5B16 optimizes power against one declared alternative distribution P1 while independently checking type-I size under P0. That isolates validity from power, but it still assumes a single alternative model and treats an extra observation as if it had no heterogeneous economic cost.

Real engineering experiments have:
- Several plausible ways the alternative can manifest.
- Different consequences for false rejection and missed detection.
- Different evaluator costs, delays, energy use and opportunity costs.
- A need to make one shared, non-anticipating policy rather than select a different policy after learning which alternative model is true.

AR-5B17 uses a finite alternative family P1_1,...,P1_K and selects one common deterministic policy by worst-case decision utility, while keeping null validity as a hard constraint.

## 2. Objective and constraints

For a policy pi and alternative model k, the reference computes:
- rejection power: Power_k(pi);
- expected evaluator cost: Cost_k(pi);
- null false-positive probability: Size_0(pi), evaluated under the single declared P0.

Utility under alternative k is:

```
U_k(pi) =
    benefit_true_rejection * Power_k(pi)
  - loss_false_negative * (1 - Power_k(pi))
  - loss_false_positive * Size_0(pi)
  - weight_evaluator_cost * Cost_k(pi)
```

The policy solves the finite maximin problem:

```
maximize_pi  min_k U_k(pi)

subject to:
    Size_0(pi) <= alpha
    max_k Cost_k(pi) <= declared_cost_cap  (when supplied)
```

The false-positive loss is included in utility, but it does not replace the hard type-I bound. The evaluator-cost cap is also separate from the continuous cost penalty.

The implementation enumerates every deterministic outer policy up to 16 internal decision histories. One policy is applied across every alternative model. Costs are incurred on observed first-stage edges and when confirmatory tests are launched. A leaf must launch its conditional continuation.

This is intentionally an exact, bounded reference—not a scalable robust MDP solver. It avoids claiming that a greedy local min/max recursion solves a global maximin problem.

## 3. Worked fixture (synthetic, independently checked arithmetic)

The first-stage P0 tree is:
- root interim value 0.05;
- early history: P0=0.5, interim value 0.08, edge cost 1;
- late history: P0=0.5, interim value 0.02, edge cost 1;
- early-good and early-bad children: P0=0.5 each, interim values 0.05 and 0.02, edge cost 1.

The null supermartingale checks:
- root: 0.5×0.08 + 0.5×0.02 = **0.05**;
- early: 0.5×0.05 + 0.5×0.02 = **0.035 <= 0.08**.

There are two alternatives:
- alt-a: root early/late probabilities 0.7/0.3; at early, good/bad probabilities 0.8/0.2.
- alt-b: root early/late probabilities 0.3/0.7; at early, good/bad probabilities 0.2/0.8.

Every history has a one-step confirmatory test with launch cost 2. Its P0 rejection probabilities are 0.05 (root), 0.06 (early), 0.01 (late), 0.05 (early-good), and 0.02 (early-bad). Each is no greater than that history's interim test value.

Use:
- true-rejection benefit = 10;
- false-negative loss = 2;
- false-positive loss = 100;
- evaluator-cost weight = 0.1;
- alpha = 0.05;
- no hard cost cap for the primary comparison.

Three non-duplicate outer policy classes produce:

| Policy | Power alt-a | Power alt-b | P0 size | Expected cost alt-a / alt-b | Worst-case utility |
|---|---:|---:|---:|---:|---:|
| Stop at root and launch | 0.50 | 0.20 | 0.0500 | 2.0 / 2.0 | -4.800 |
| Continue at root; launch at early/late | 0.66 | 0.68 | 0.0350 | 3.0 / 3.0 | 2.120 |
| Continue at root and early; launch at leaves | 0.576 | 0.755 | 0.0225 | 3.7 / 3.3 | **2.292** |

The selected robust-utility policy continues at root and early, achieving worst-case power 0.576, P0 size 0.0225, maximum expected cost 3.7 and worst-case utility 2.292. It does not maximize worst-case power: the middle policy has worst-case power 0.66. It wins only because the declared false-positive loss is high enough to reward its lower null size.

There are two internal decision histories, so the exhaustive engine enumerates four bit masks. When the root policy stops immediately, the early-history bit is unreachable; those two masks encode the same realized stopping rule. This is why the three distinct pathwise policy classes appear in the table while the report records four enumerated masks.

When the hard worst-case evaluator-cost cap is 3.5, the more expensive policy becomes infeasible and the selected policy changes to the middle policy (P0 size 0.035; maximum expected cost 3.0). This demonstrates that a hard resource constraint and a soft cost penalty do different jobs.

### Alternative-misspecification control

Reduce alt-a's early-good probability from 0.8 to 0.1 and increase early-bad from 0.2 to 0.9. The deeper continuation policy's alt-a power falls to 0.1595 and its worst-case utility becomes -2.706. The selected policy changes to launch at the early history instead. Its P0 size remains 0.035.

These are constructed finite-tree calculations, not empirical results or proof of robustness to alternatives outside the declared family.

## 4. Implementation and failure modes

The module:
- Requires at least two alternative models and a common node/outcome structure.
- Validates each P0/P1 transition distribution.
- Validates the P0 first-stage supermartingale at every internal history.
- Validates the conditional P0 rejection size of each continuation against its interim value.
- Enumerates all deterministic policy masks (bounded at 16 decision nodes).
- Computes each policy's P0 size and per-alternative power, expected cost and utility.
- Rejects policies over alpha or the optional worst-case cost cap.
- Maximizes the minimum per-alternative utility.
- Uses deterministic tie-breaking: higher worst-case utility, then lower maximum expected cost, lower null size, and lower canonical policy mask.
- Emits policy decisions, per-alternative metrics, model IDs and canonical plan/report digests.

Negative controls cover probability mass, support/model omissions, conditional budgets, invalid costs, impossible cost caps, policy sensitivity and report tampering.

## 5. Important claim boundaries

- Robustness is only over the explicitly declared finite alternative family. An omitted alternative may have much worse power and utility.
- P0 validity depends on the correctness of the supplied null kernel and the test-value supermartingale conditions.
- The policy class is deterministic. A randomized mixture can improve a maximin objective by convexification; this reference does not solve that LP.
- The cost model is expected cost under each supplied alternative. Heavy-tailed costs, delays, queue contention and correlated resource failures need a richer model.
- A single false-positive cost is not a substitute for domain-specific safety constraints or human authorization.
- This is not a general robust MDP, learned policy, production decision controller, or SOTA claim.

## 6. Industry and SOTA comparison

A useful benchmark suite must include several regimes.

**Known or estimable maximum horizon**
- Fixed-horizon Neyman–Pearson terminal tests.
- O'Brien–Fleming, Pocock and Lan–DeMets group-sequential spending.
- Conditional-error and inverse-normal/Fisher combination tests.

**Streaming / uncertain horizon**
- Always-valid inference and confidence-sequence approaches.
- Mixture SPRT / mSPRT.
- E-processes and exact sequential E-variables for k-sample testing.

**Decision-aware evaluation**
- A fixed-horizon decision-theoretic policy with the same utility and evaluator cost.
- A nominal-P1 power optimizer.
- A per-alternative oracle policy (upper bound, not a deployable baseline).
- Minimax power with no cost or asymmetric loss.
- Stop-and-discard and fixed alpha split.

References:
- Always-valid inference was implemented in a commercial A/B testing platform at large scale; it is a real production comparator, not merely a theoretical baseline: https://doi.org/10.1287/opre.2021.2135
- Spotify's comparison emphasizes regime dependence: group-sequential methods can be more powerful when horizon and variance structure are well understood; always-valid inference is useful for streaming and uncertain horizons: https://engineering.atspotify.com/2023/3/choosing-sequential-testing-framework-comparisons-and-discussions
- Spotify's July 2026 sample-size correction highlights an important benchmarking issue: sample-size calculations should account for early stopping rather than only the endpoint rejection probability: https://confidence.spotify.com/blog/accurate-sample-size-always-valid
- Decision-theoretic sequential experimentation explicitly balances technology-adoption benefit and sampling cost with delayed response: https://doi.org/10.1111/rssb.12225

## 7. Benchmark programme and target gates

Freeze a grid of:
- alpha values: 0.01, 0.025 and 0.05;
- alternative-family size: 2, 4 and 8;
- 2–16 internal decision histories;
- evaluator costs with low/high asymmetry;
- false-negative to false-positive loss ratios spanning 1:1 through 1:100;
- nominal, held-out and deliberately omitted alternatives;
- correlated, delayed and heteroskedastic observation regimes.

Report:
1. Exact P0 size for every policy and empirical false-positive rate across at least 10,000 independent seeds for scalable stochastic benchmarks.
2. Worst-case and per-model power and expected utility.
3. Expected and tail evaluator count, wall time, energy, memory and queue delay.
4. Worst-case utility regret relative to the best per-model oracle.
5. Cost-cap feasibility rate and unused alpha budget.
6. Robustness to held-out alternatives and omitted-family negative controls.
7. Exact DP/exhaustive enumeration agreement for every tiny policy tree.

Proposed stretch targets, not results:
- Zero finite-oracle discrepancies against exhaustive enumeration.
- Zero false-positive budget overruns in exact finite fixtures.
- At least 10% lower worst-case decision loss at matched alpha and evaluator-cost cap versus the strongest matched baseline.
- No more than 5% regret against the best single-model oracle across the registered alternative grid.
- A separately reported robustness penalty when the true alternative is outside the registered family.

Never claim universal superiority from a single synthetic tree. Comparisons must use the same null, alternative, sample horizon, stopping opportunities, cost assumptions and loss function.

## 8. Exact-head validation

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering robust_utility_continuation -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

Seven tests are authored. No compile, formatting, clippy, test, CI, benchmark or SOTA pass is claimed.
