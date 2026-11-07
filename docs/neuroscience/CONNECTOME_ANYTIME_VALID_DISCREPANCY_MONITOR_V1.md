# Anytime-valid response-model discrepancy monitoring V1

**Tracking:** #10897 — SCI-CONNECTOMICS-098  
**Composes:** #10880, #10762, #10831, #10781 and #9462.  
**Status:** exact finite synthetic monitor and tree oracle; no biological result is claimed.

## 1. Why this tranche exists

The active-probe benchmark (#10880) selects measurements that may expose omitted response mechanisms while preserving query value. The adaptive-policy work (#10762 / #10831) determines which action to take after each observed history. Neither by itself supplies an anytime-valid error process for repeatedly monitoring a campaign whose action choices and stopping time depend on the observations.

A conventional fixed-look p-value cannot simply be inspected after every adaptive stage and retain its nominal type-I error. This tranche composes sequential-testing theory with the connectomics response-model envelope.

## 2. Distinct from adjacent work

- #10880 owns which probe to acquire and the Pareto trade-off among query value, discrepancy value, audit coverage and cost.
- #10762 / #10831 own coherent adaptive intervention-policy search under one static, latent model identity.
- #10781 owns held-out trajectory calibration and out-of-support diagnostics.
- #9462 owns finite-population, design-based residual auditing over a frozen PSU frame. Its sampling estimand and SRSWOR assumptions differ from intervention-response likelihoods.
- V1 here owns the **sequential discrepancy evidence process** and exact finite-horizon false-alarm replay.

It is a monitor, not an intervention optimizer, causal estimator, or model-completeness certificate.

## 3. Robust null-envelope e-process

Freeze one campaign, binary outcome alphabet, horizon, finite null family M0, finite challenger family C, fixed prior over complete challenger models, deterministic history-dependent action policy, and every conditional response kernel before opening outcomes.

The action at stage t must be a predictable function of observed history only:
    a_t = pi(h_t)

The actual null response model is one fixed vertex m in M0 for the whole campaign. A null vertex cannot switch identity across histories or stages. The challenger family is also a fixed mixture over complete models. After observing history h, compute its posterior predictive distribution by updating the original challenger prior with the likelihood of h. Do not reset the prior at each stage.

For the selected action a and outcome y, let:
    q_C(y | h,a) = fixed-mixture challenger posterior predictive
    p_star(y | h,a) = max over m in M0 of p_m(y | h,a)

Define the one-step e-factor:
    e_t(y | h,a) = q_C(y | h,a) / p_star(y | h,a)

For every fixed null vertex m:
    E_m[e_t | h,a]
      = sum_y p_m(y | h,a) q_C(y | h,a) / max_j p_j(y | h,a)
      <= sum_y q_C(y | h,a)
      = 1

Thus the product E_t = product_{s<=t} e_s is a non-negative supermartingale under every declared null vertex, as long as the supplied kernels are valid and actions depend only on observed history. Ville's inequality gives:
    P_m(max_{t<=T} E_t >= 1/alpha) <= alpha

The bound is uniform over the finite declared null vertices and predictable action paths. It is not robust to an omitted null mechanism, measurement corruption, invalid kernels, or unregistered multiplicity. It does not prove that the finite model family is complete.

If p_star(y|h,a)=0 and q_C(y|h,a)>0, the observed outcome is outside every declared null vertex and has positive challenger probability; the evidence process crosses with infinite e-factor. If both are zero, the outcome is outside every declared model and is reported as OUT_OF_SUPPORT_UNMODELED, not converted into a clean rejection. If the observed history exhausts all challenger mixture support, monitoring stops with CHALLENGER_MIXTURE_EXHAUSTED rather than silently refreshing the prior.

## 4. Exact two-stage known-answer case

This fixture is deliberately small enough for exact rational arithmetic and exhaustive enumeration.

**Policy**
- Empty history -> action screen.
- History H -> action follow-H.
- History L -> action follow-L.

**Fixed null vertices**
- N0: H=1/2, L=1/2 at every reachable stage.
- N1: H=3/5, L=2/5 at every reachable stage.

**Fixed challenger mixture**
- C-strong: prior 1/2; H=19/20, L=1/20.
- C-weak: prior 1/2; H=17/20, L=3/20.

The mixture posterior is updated rather than refreshed:
- Before any observation: challenger q(H)=9/10, q(L)=1/10.
- After H: posterior weights (19/36, 17/36), so q(H|H)=65/72 and q(L|H)=7/72.
- After L: posterior weights (1/4, 3/4), so q(H|L)=7/8 and q(L|L)=1/8.

The robust null envelope is H=3/5 and L=1/2. Therefore:
- At empty history: e(H)=3/2, e(L)=1/5.
- After H: e(H)=325/216, e(L)=7/36.
- After L: e(H)=35/24, e(L)=1/4.

With alpha=1/2 and threshold 2, only path HH crosses. Its wealth is:
    E_2(HH) = (3/2) * (325/216) = 325/144 > 2

The exact probability of ever crossing is 1/4 under N0 and 9/25 under N1; both are at most alpha. The observed fixture path HH therefore yields MODEL_DISCREPANCY_DETECTED, while the null-envelope false-alarm bound is satisfied. Alpha=1/2 is chosen for transparent arithmetic, not as a biological operating threshold.

## 5. Required implementation invariants

1. Use exact rational probabilities and wealth for the finite oracle; no float tolerance for the known answer.
2. Validate every action/history/model kernel and require each categorical distribution to sum exactly to one.
3. Require complete action-policy coverage for every reachable history.
4. Policy rows may contain only history and action ID; hidden model identity is not an input.
5. Require every action selected by the frozen policy to have a synthetic feasibility receipt.
6. Compute challenger posterior weights from the original static prior and complete history likelihood.
7. Keep each null vertex fixed during the entire path; do not build a rectangular stage-wise adversary.
8. Enumerate every path under every null vertex and compute the probability of ever crossing the threshold.
9. Reject the fixture if any exact null crossing probability exceeds alpha.
10. Replay the observed path independently from the whole-tree calculation and compare exact wealth.
11. Out-of-alphabet or outside-all-model-support observations are not clean e-value rejections.
12. Non-crossing is not evidence of global adequacy; it is only NO_DISCREPANCY_DETECTED_WITHIN_DECLARED_CHALLENGE_FAMILY for this declared horizon and challenge set.

## 6. Statuses

- NOT_TESTED
- MONITORING_CONTINUES
- MODEL_DISCREPANCY_DETECTED
- NO_DISCREPANCY_DETECTED_WITHIN_DECLARED_CHALLENGE_FAMILY
- OUT_OF_SUPPORT_UNMODELED
- CHALLENGER_MIXTURE_EXHAUSTED
- INVALID_POLICY_OR_INCOMPLETE_KERNEL
- NO_FEASIBLE_DISCREPANCY_MONITOR

A threshold crossing rejects the declared null envelope at the registered anytime-valid level under the stated assumptions. It does not identify which challenger is true, prove a direct synaptic effect, or establish in-vivo intervention efficacy.

## 7. Baselines and benchmark targets

Compare on identical finite worlds, observation kernels, action policies and compute:
- repeated fixed-look tests without correction (negative control; expected to exceed its advertised rate in selected scenarios);
- fixed-horizon testing with preregistered alpha spending;
- a single-null likelihood-ratio e-process;
- ordinary Bayesian model evidence, reported separately from frequentist false-alarm control;
- a challenger mixture incorrectly reset at each look (negative control);
- the robust null-envelope e-process;
- exhaustive exact-tree replay as the oracle.

**Synthetic gates**
- exact rational agreement for predictive probabilities, e-factors, path wealth and false-alarm probability;
- exact false-alarm probability <= alpha under every fixed null vertex;
- zero hidden-model policy branches;
- zero missing or unnormalized kernels;
- zero silent alternative-prior refresh;
- exact separation of threshold crossing, support exhaustion and out-of-support outcomes;
- online-path wealth agrees with exact whole-tree replay;
- injecting a declared false-alarm probability above alpha must fail the audit; changing alpha recomputes the threshold and crossing event rather than reusing a previous event probability.

**Biological evaluation targets (not achieved)**
Use animal/session/target as the independent unit and report power-versus-cost, detection delay, false-alarm frequency, calibration, out-of-support rate and audit coverage. Choose thresholds only after a source-bound pilot estimates independent-unit variance and expected prevalence. Do not use neurons, pairs or time bins as IID biological replicates.

## 8. V1 limits and next research

V1 handles a small finite categorical campaign with deterministic observed-history policies. It does not yet cover randomized action selection, continuous outcomes, unbounded horizons, large-scale composite-null optimization, cross-query multiplicity, cluster-robust biological inference, or adaptive challenger discovery/confirmation. Those need separate theorem-matched extensions.

The result is an auditable finite monitor, not a new general sequential-testing theorem. It should be compared with established anytime-valid methods, including *Anytime validity is free: inducing sequential tests* (2026) and sequential Monte Carlo testing by betting (2025).
