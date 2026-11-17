# EGDDM ShiftBench D6.17.2.2-alpha — Joint stochastic outcome kernels and dependence-preserving acquisition

**Date:** 2026-11-17  
**Status:** joint-kernel validation prototype; synthetic known-answer fixture; Rust execution pending  
**Related:** [D6.17.2.2 issue #12630](https://github.com/Luminous-Dynamics/symthaea/issues/12630), [D6.17.2.1 exact adaptive oracle PR #12649](https://github.com/Luminous-Dynamics/symthaea/pull/12649), D6.17.1 partial dependence, D6.18 open-world adequacy.

## Why this increment exists

D6.17.2.1 provides an exact adaptive-policy oracle over deterministic finite worlds: each world maps each acquisition option to one fixed outcome. That is useful for search and policy-regret calibration, but it cannot represent a sensor that produces different outcomes on repeated trials under the same physical hypothesis.

A second limitation is more consequential: per-channel marginal likelihoods do not identify the joint distribution of multiple observations. Multiplying marginals silently assumes conditional independence. A generic Fréchet envelope is valid without an independence proof, but can be very wide. D6.17.2.2-alpha introduces an evaluator-owned credal distribution over **complete joint outcome vectors**. It retains stochasticity and common-cause dependence rather than reconstructing a joint distribution from marginal channels.

This first increment validates the joint model and its marginal consistency. It is not yet the stochastic adaptive-policy optimizer; that remains the next step.

## Relationship to current research and the novelty boundary

The underlying ideas are established. We should not claim that D6.17.2.2 invents robust Bayesian inference, stochastic optimal control, or optimal experimental design.

- Huan, Jagalur and Marzouk's 2024 *Acta Numerica* review covers modern optimal experimental design, including Bayesian/decision-theoretic criteria, combinatorial observation selection and sequential non-myopic design policies. D6.17.2 should be compared with these methods, not positioned as their replacement. [Optimal experimental design: Formulations and computations](https://doi.org/10.1017/S0962492924000023).
- Caprio et al. (UAI 2026) study Huber-style contamination of both priors and likelihoods, with tractable robust Bayesian predictive uncertainty. D6.18 and D6.17.2.2 should build on this literature and focus on the engineering policy/evidence integration. [Robust Predictive Uncertainty and Double Descent in Contaminated Bayesian Random Features](https://proceedings.mlr.press/v337/caprio26a.html).
- The 2026 review *From benchmarks to deployment* identifies gaps in agent-evaluation protocols around safety, security, cost efficiency and end-to-end workflow reliability. It is a broad review, not a claim that no benchmark elsewhere has these measures; its 15-benchmark sample found that the surveyed benchmark protocols rarely included them in primary scoring. [Review](https://doi.org/10.1007/s10462-026-11571-0).
- NIST's 2026 *Building Evaluation Probes into Agentic AI* programme builds adversarial verifiers and machine-readable evidence trails for agent claims. D6.17.2.2 can complement this direction by making stochastic model assumptions and cross-channel dependence machine-checkable. [NIST project](https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai).
- A 2026 Findings of ACL survey on LLM-agent evaluation emphasizes sequential decision-making, dynamic environments, and the need for safety, robustness and cost-efficiency measures beyond text-output accuracy. [Survey](https://aclanthology.org/2026.findings-acl.1330/).

The defensible contribution under evaluation is an operational combination:
1. evaluator-owned joint stochastic kernels over complete outcome vectors;
2. exact bounded-simplex projections to marginal and pairwise event intervals;
3. consistency checks against the D6.17 channel-likelihood registry;
4. eventual adaptive policy search over histories without independence shortcuts;
5. integration with D6.18 unknown-world reserves and D6.17.2.3 end-to-end cost receipts.

## Model

For one evaluator-owned hypothesis/world \(h\), let \(J_h\) be the finite set of complete outcome vectors across all registered acquisition options. Each row \(j\in J_h\) has an interval and a nominal mass:

\[
\underline q_{hj}\le q_{hj}\le\overline q_{hj},
\quad
q^0_{hj}\in[\underline q_{hj},\overline q_{hj}],
\quad
\sum_jq_{hj}=1,
\quad
\sum_jq^0_{hj}=1.
\]

The row bounds must define a non-empty bounded simplex:

\[
\sum_j\underline q_{hj}\le1\le\sum_j\overline q_{hj}.
\]

For any event \(A\subseteq J_h\), the exact probability interval under this box-constrained simplex is:

\[
\underline q_h(A)=
\max\left(\sum_{j\in A}\underline q_{hj},
1-\sum_{j\notin A}\overline q_{hj}\right),
\]

\[
\overline q_h(A)=
\min\left(\sum_{j\in A}\overline q_{hj},
1-\sum_{j\notin A}\underline q_{hj}\right).
\]

The nominal probability is \(q_h^0(A)=\sum_{j\in A}q^0_{hj}\).

D6.17.2.2-alpha applies these formulas to each option/outcome marginal and each pairwise outcome intersection. The induced joint-kernel marginal interval must be contained in the registered D6.17 channel interval, and the nominal marginal must match exactly. The evaluator rejects underclaimed channel bounds.

This is an explicit model contract: the joint kernel is evaluator-owned, valid at the decision cutoff, unrevoked, and complete over the registered option dimensions. A future version may support partial outcome vectors and missing/unavailable results, but they must be explicit outcome categories rather than silently omitted rows.

## Synthetic common-cause control

The fixture has two gates, each with the same marginal failure interval:

\[
P(A=\text{fail})\in[0.40,0.60],\qquad
P(B=\text{fail})\in[0.40,0.60],
\]

with nominal marginal failure probability 0.50 for each.

The evaluator-owned joint kernel is:

| Gate A | Gate B | Lower | Nominal | Upper |
|---|---|---:|---:|---:|
| pass | pass | 40% | 45% | 50% |
| pass | fail | 0% | 5% | 10% |
| fail | pass | 0% | 5% | 10% |
| fail | fail | 40% | 45% | 50% |

It induces:
- each marginal failure interval [40%, 60%], nominal 50%;
- joint fail/fail interval [40%, 50%], nominal 45%;
- Fréchet-only fail/fail envelope [0%, 60%], width 6,000 basis points;
- joint-kernel fail/fail interval width 1,000 basis points.

The joint interval is six times narrower in width than the generic Fréchet envelope in this synthetic fixture. The nominal joint fail/fail mass is 45%, not the 25% product implied by an independence shortcut. The purpose is to expose common-cause dependence and quantify how much information an independently qualified joint model could contribute. It is not evidence that any real sensor pair has these probabilities.

## Implemented alpha slice

The Rust binary stochastic-kernel-validator.rs currently:
- accepts a candidate-visible option/factor/channel model and a separate evaluator-only joint-kernel oracle;
- validates exact hypothesis coverage and option dimensions;
- rejects duplicate joint vectors, unknown outcomes, invalid dates and revoked/stale kernels;
- verifies bounded-simplex feasibility and nominal normalization;
- derives exact marginal event intervals and checks containment by each D6.17 likelihood interval;
- derives pairwise joint-event intervals and checks them against Fréchet envelopes;
- emits a machine-readable dependence report and explicit claim ceiling.

The candidate cannot supply or override world_kernels. The current fixture has one hypothesis, two binary channels and four joint rows. The validator does not yet calculate adaptive policy costs or choose a policy.

## Next mathematical step: exact stochastic policy cost on small trees

For a fixed deterministic policy tree, every complete outcome vector induces a terminal path and a total acquisition cost c_j. Conditional on hypothesis h, worst-case expected cost is a linear optimization over the evaluator-owned joint credal set:

\[
C_h^{\max}(\pi)=
\max_{q_h\in\mathcal Q_h}
\sum_{j\in J_h}q_{hj}c_j(\pi).
\]

Because \(\mathcal Q_h\) is a bounded simplex, this is solved exactly by starting at the lower masses and allocating remaining mass to the rows with highest path costs until their upper bounds are reached.

If the prior credal set is \(\mathcal P\), then under the explicitly stated product-set assumption (each world's kernel can vary independently conditional on the world):

\[
C^{\max}(\pi)=\max_{p\in\mathcal P}\sum_h p_hC_h^{\max}(\pi).
\]

That outer optimization is another bounded-simplex linear programme. The exact policy oracle can therefore score every finite policy without assuming that acquisition outcomes are conditionally independent within a world. The nominal expected cost uses the evaluator's nominal prior and nominal joint-kernel masses.

This is exact only for the stated product credal set. A future shared-latent or common-cause coupling across hypotheses needs a joint outer set and must not be approximated silently.

At each reachable policy history, action qualification still replays the observations through D6.17.1 (or a future joint-likelihood adapter) and D6.18's open-world reserve. The decision objective should be lexicographic: satisfy hard safety gates, maximize justified positive-control action coverage, then minimize worst-case expected cost, nominal cost and worst path cost.

## Benchmark design and proposed targets

The benchmark must compare policy quality, not only detector scores or information gain.

**Model adequacy and dependence**
- Calibrate joint-kernel marginal and pairwise interval coverage on held-out repeated measurements.
- Report interval coverage and sharpness separately by outcome, hypothesis, device, site and environment.
- Include shared power, clock, firmware, network, reference standard and environmental common causes.
- Include dependence drift and calibration revocation.
- Include candidate attempts to replace the oracle, remove options, duplicate vectors, understate marginals or relabel an outcome.

**Adaptive policy**
- Compare independence multiplication, Fréchet-only robust bounds, evaluator-owned joint kernels, nominal EVSI, robust one-step EVSI, receding-horizon heuristics, full revalidation, and exact small-tree policy search.
- Use the same model/task/harness/runner budgets and paired seeds.
- Report exact policy regret only when exhaustive search completes; otherwise mark the oracle bound unavailable.
- Preserve a human/operator plan and always-act / always-abstain controls.

**Metrics**
1. Joint-kernel interval coverage, nominal calibration and interval sharpness.
2. Marginal-consistency violation count (target zero).
3. Pairwise and higher-order common-cause miss rate.
4. Open-world and robust-risk false-safe rate (target zero on registered critical controls).
5. Justified action coverage (target at least 95% on positive controls).
6. Exact policy regret on small exhaustive cases (target zero).
7. Adaptive heuristic cost regret on small cases (research target at most 1% at matched coverage).
8. Full-run trajectory p50/p95/p99 from joint attempt receipts, separate from sums of per-attempt p95 bounds.
9. Measured cost per robustly justified action; research target at least 20% lower median and 10% lower p95 than full revalidation at matched safety.
10. Drift, revocation, unknown-outcome and escalation latency.

Start with 144 adjudicated decisions as a validity pilot, then a 400-episode development suite. Expand to thousands of independent or cluster-aware episodes before making rare-event rate claims. These are proposed experimental targets, not achieved results.

## Industry comparison: where to be better

| Area / baseline | Strong established capability | EGDDM target |
|---|---|---|
| Sequential optimal experimental design | Non-myopic Bayesian and decision-theoretic acquisition | Explicit robust action constraints, open-world reserve, replayable evidence and measured tail cost |
| Robust Bayes / contaminated models | Prior and likelihood uncertainty sets | Joint stochastic outcome kernels and action-policy regret |
| Fréchet fallback | Safe dependence bounds without independence assumptions | Tighter joint envelopes only when evaluator evidence qualifies them |
| OOD / open-set detection | Shift and unknown-state signals | Quantify downstream decision risk and choose the next discriminating measurement |
| Agent-evaluation probes | Adversarial grounding checks and audit trails | Include model, likelihood, dependence, action, cost and policy receipts in one evaluator-owned record |

The path to outperforming established practice is not another headline benchmark score. It is to demonstrate, on matched and independently adjudicated tasks, fewer unsafe decisions at the same or greater useful coverage, and lower measured cost at the same robust-risk envelope. If the results do not show that, we should report the negative result and revise the design.

## Claim ceiling and next steps

D6.17.2.2-alpha validates one finite evaluator-owned joint-kernel representation. It does not establish real-world calibration, estimate drift, prove model completeness, solve an adaptive policy, authorize actions or supply a safety guarantee.

Next:
1. Add a stochastic policy-tree scorer using complete joint outcome vectors and exact bounded-simplex expected-cost optimization.
2. Extend exhaustive search to stochastic histories on small trees and prove deterministic-limit parity with D6.17.2.1.
3. Add candidate/oracle negative controls and joint-kernel commitments before candidate execution.
4. Connect D6.18 contamination reserve to each history.
5. Replace synthetic kernels and costs with held-out real measurement and full-run receipts under D6.17.2.3.

Rust compilation, tests, rustfmt, Clippy and CI remain unexecuted in this environment. The alpha must stay draft until exact-head validation and independent repeated-measurement trials.
