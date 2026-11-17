# EGDDM ShiftBench D6.17.2.2.5-alpha — Goal-oriented one-step robust decision value of information

**Date:** 2026-11-17  
**Status:** exact one-step action-rule enumerator over shared-latent point kernels; synthetic known-answer fixture; Rust execution pending  
**Related:** issue #12684, programme #12630, shared-latent validator PR #12689, stochastic policy scorer PR #12686.

## 1. Research positioning

Information gain is not decision value. A measurement can reduce uncertainty about the world without changing the safest or most useful action. Conversely, a small amount of targeted evidence can change a high-consequence decision even when it has low global mutual information.

This distinction is already present in current research:
- GO-CBED (TMLR 2026) optimizes non-myopic intervention policies around a specified causal quantity of interest, rather than requiring the agent to learn an entire causal graph. https://uq.engin.umich.edu/causal-learning.html
- Cheng and Huan (2025, revised June 2026) formulate sequential Bayesian experimental design as an MDP in which the next experiment and whether to stop are joint choices. https://arxiv.org/abs/2509.21734
- Wen et al. (ICML 2026) study robust sequential experimental design under model misspecification in contextual and dynamic settings. https://proceedings.mlr.press/v306/wen26i.html
- NIST's 2026 agentic-evaluation-probe programme emphasizes workflow-integrated verification and machine-readable evidence trails. https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai

EGDDM should use those methods as important comparators, not claim their ideas as new. The contribution here is a decision-assurance evaluator: an evaluator-owned loss matrix, shared-latent joint evidence model, independently qualified post-observation actions, explicit safe defer, and a reproducible robust stopping decision.

## 2. Decision problem

An evaluator-only query specifies:
- a target decision and loss currency;
- a bounded prior over the registered hypotheses;
- an action-by-hypothesis loss matrix in micro-units;
- action safety constraints, including each action's safe hypotheses, residual-risk envelope, hard-gate and irreversibility flags;
- one evaluator-owned globally safe read-only defer action;
- evaluator-owned measurement costs for all candidate options.

The current stop decision chooses the minimum-worst-case-loss action among actions that pass the robust risk gate at the current evidence state. The global read-only defer remains available even when support is zero, but it is not treated as evidence that a consequential action is justified.

For a measurement option, the evaluator enumerates every deterministic mapping from each possible measurement outcome to an action. A mapping is retained only when every outcome's action independently passes the shared-latent robust posterior-risk gate. Thus, a high expected-utility action that is unsafe after one branch cannot be selected merely because it makes the average loss attractive.

For each legal rule \(\delta\), the post-measurement decision loss is:

\[
L(\delta;p,w)=
\sum_h p_h\sum_z w_z\sum_j q_{h,z}(j)
\,\ell\!\left(\delta(y_j),h\right).
\]

Here \(p\) belongs to the bounded prior set, \(w\) is the single shared latent-weight vector, \(q_{h,z}\) is the evaluator-owned point joint kernel, and \(y_j\) is the selected option's outcome in the complete joint row.

The minimax post-measurement loss is:

\[
L_{\mathrm{post}}^{\max}(o)=
\min_{\delta\in\mathcal D_o^{\mathrm{qualified}}}
\max_{p,w} L(\delta;p,w).
\]

The current stop loss is:

\[
L_{\mathrm{stop}}^{\max}=
\min_{a\in\mathcal A^{\mathrm{qualified}}}
\max_p\sum_h p_h\ell(a,h).
\]

The reported one-step net minimax decision value is:

\[
V_{\mathrm{net}}^{\mathrm{robust}}(o)=
L_{\mathrm{stop}}^{\max}
-L_{\mathrm{post}}^{\max}(o)-c_o.
\]

The next measurement is recommended only when the best net robust value is strictly positive. Otherwise the system stops at the robustly qualified action (in this fixture, the globally safe read-only defer).

**Interpretation limit:** this is the difference between two minimax objective values. It is not a claim that every individual prior/latent realization benefits from measuring. It is a one-step finite decision criterion, not a full multi-step dynamic program, a non-myopic GO-CBED policy, or an optimal stopping solution over an unbounded horizon.

## 3. Exact solution in the current finite model

The prior and latent-weight ambiguity sets are bounded simplexes. For a fixed deterministic outcome-to-action rule, expected loss is multilinear in prior mass and shared latent weights. Its maximum over the product of these two polytopes occurs at a pair of vertices. The implementation enumerates latent-simplex vertices and optimizes the linear prior objective at each vertex. All outcome-to-action rules are exhaustively enumerated, subject to an evaluator cap.

The current conditional kernels are point distributions. Extending to interval-valued conditional kernels, dynamic latent transitions, and shared-latent temporal dependencies requires additional work and is not claimed here.

To avoid rounding a positive value into existence, the displayed stop loss is floored to the micro-unit, post-measurement loss is ceiled to the micro-unit, and net value is computed from those conservative integer values. This can understate a small positive value by at most the rounding granularity, but does not create a positive decision value from rounding in this implementation.

## 4. Synthetic known-answer fixture

The fixture contains:
- two hypotheses: safe-world and common-cause-hazard;
- two gates with shared-latent joint kernels;
- a fixed 50/50 prior;
- two actions: release and globally safe read-only inspect/defer;
- release loss 2,000 micro-USD only under common-cause-hazard;
- inspect/defer loss 2,000 micro-USD only under safe-world;
- evaluator risk envelope for release: 4,500 bps;
- measurement costs: gate-a 100 micro-USD, gate-b 200 micro-USD.

At the current evidence state, release has 5,000 bps posterior hazard risk and is not qualified. The read-only defer has robust and nominal loss of 1,000 micro-USD.

For either gate, a pass can qualify release under the 4,500-bps envelope; a fail does not qualify release and must defer. The two legal rules are therefore:
- pass → release; fail → inspect/defer;
- both outcomes → inspect/defer.

The first rule has:
- robust post-measurement loss: 860 micro-USD;
- nominal post-measurement loss: 850 micro-USD.

The all-defer rule has 1,000 micro-USD loss and does not improve on stopping. Exactly two of the four possible binary outcome-to-action rules are qualified.

| Option | Measurement cost | Robust post-loss | Net robust value | Net nominal value |
|---|---:|---:|---:|---:|
| gate-a | 100 | 860 | +40 | +50 |
| gate-b | 200 | 860 | −60 | −50 |

The evaluator therefore recommends measuring gate-a and not gate-b. This is a synthetic expected-value calculation over the declared latent kernels and loss matrix, not measured sensor accuracy, operational savings, or a policy-generation benchmark win.

### Risk-envelope negative control

When the release maximum residual risk is tightened from 4,500 to 4,000 bps, the pass branch no longer qualifies release. The only qualified decision rule for either gate is to defer on both outcomes. Gate-a's robust net decision value becomes −100 micro-USD, so the recommendation changes from measure to stop/defer. This confirms that expected loss cannot bypass the robust action gate.

## 5. Implementation and adversarial controls

The shared-latent validator now:
- accepts evaluator-only decision-VOI queries, not candidate-authored losses, action permissions or measurement costs;
- requires an exact prior × hypothesis and loss × action matrix;
- requires a complete evaluator action-constraint registry and exactly one globally safe read-only defer;
- screens the immediate stop action with the same risk qualification used after measurement;
- enumerates every finite outcome-to-action mapping for each option;
- filters rules by positive event support and worst-case shared-latent posterior risk;
- calculates nominal and worst-case post-measurement loss, then subtracts evaluator-owned measurement cost;
- reports each option's decision rule, qualified-rule count, net values and the best next step;
- uses conservative micro-unit rounding for the net decision value.

Tests cover positive-value measurement, a more expensive dominated measurement, a tightened risk envelope that forces defer, candidate/oracle isolation and the requirement for a single globally safe read-only fallback.

## 6. Benchmark design and SOTA comparison

Compare on matched evaluator-owned instances:
1. Global-model expected information gain.
2. Target-causal-QoI information gain / GO-CBED-style design.
3. Myopic expected value of sample information.
4. Nominal one-step decision value of information.
5. This robust one-step minimax decision value.
6. Cheng–Huan style joint design/stopping.
7. Robust sequential experimental design under misspecification (Wen et al.).
8. Full revalidation, no-measurement stop, always-measure and always-defer controls.
9. Exact finite multi-step policy search from D6.17.2.2.4 on small instances.

Measure:
- decision regret and false-safe action count;
- robust/nominal post-decision loss;
- VOI ranking regret against exact multi-step finite trees;
- premature stopping and over-testing;
- cost per robustly justified action;
- time-to-safe-decision and delay-adjusted loss;
- sensitivity to prior widening, latent-weight widening and common-cause drift;
- calibration/coverage and interval sharpness on held-out latent families;
- measured run cost from D6.17.2.3 receipts rather than only nominal option costs.

### Proposed targets (not achieved)

- Exact agreement with exhaustive one-step rule enumeration on all small fixtures.
- Zero acceptance of a post-measurement action that violates its robust risk/hard-gate constraint.
- Zero positive net-VOI decisions from rounding alone.
- At least 95% correct next-step ranking versus exact multi-step oracle on a pre-registered validity pilot, with ties and ambiguity reported.
- At matched robust risk, at least 10% lower measured end-to-end cost than global-information-gain acquisition on decision-targeted challenge cases where both systems achieve equal justified-action coverage.
- Zero false-safe decisions on adjudicated common-cause and unknown-world controls.

Targets must be frozen before held-out evaluation. A failed target is reported as failed; do not weaken the evaluator or change the test set after seeing results.

## 7. Limitations and next steps

This alpha performs exact one-step rule enumeration over shared-latent point kernels and a bounded product of prior and shared-latent-weight sets. It does not:
- search multi-step adaptive policies;
- include stochastic measurement costs, queueing, retries or delays (join D6.17.2.3 receipts later);
- estimate calibration or latent weights from real data;
- prove latent-state or hypothesis completeness;
- represent conditional-kernel interval ambiguity or a time-varying latent process;
- establish value under every individual ambiguity realization.

Next:
1. Compare its one-step recommendations against D6.17.2.2.4's exact multi-step tree oracle.
2. Add multiple measurements with contingent stopping and shared-latent updates.
3. Add delay costs, unknown execution outcomes and receipt-based measurement costs.
4. Integrate D6.18 open-world reserve so unregistered outcomes cannot become false-safe.
5. Validate the loss matrix and latent kernels with independently adjudicated repeated measurements.

## 8. Claim ceiling

This is an exact one-step minimax decision-value calculation for the finite evaluator-provided outcome-to-action rules, point conditional kernels, shared latent-weight simplex, bounded prior and loss matrix. It is not a general optimal stopping solver or a multi-step POMDP policy. All present results are synthetic. Rust compilation, tests, formatting, Clippy and CI remain unexecuted; keep the PR draft until exact-head validation and independent measurement replay.
