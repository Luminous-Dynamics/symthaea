# Static-model robust sequential connectome intervention design V1

**Tracking:** #10762 — SCI-CONNECTOMICS-095
**Composes:** #10603, #10661, #10670, #10675, #10757, #9587, #9255
**Status:** finite two-stage synthetic protocol; no biological intervention or performance result is claimed.

## 1. Research question

How should an experimenter choose a sequence of interventions when the true causal/measurement model is unknown but remains fixed during the experimental campaign?

One-step robust design (#10761) chooses an intervention under a finite ambiguity set. Real experiments are sequential: an observation can reveal information about the world, the model family, or both, and the next intervention can depend on that history.

The model identity must remain latent. A deployable policy can condition on recorded observations, never on the hidden model-vertex ID.

## 2. Static versus rectangular uncertainty

Let `m` denote a causal/measurement model vertex, `w` the fixed biological world, and `h_t` the observed history before stage t.

**Static/non-rectangular model uncertainty:** nature chooses one `m` from the frozen vertex set before the campaign; that same `m` governs the prior and every response kernel along the full trajectory.

**Rectangular relaxation:** nature may choose a different model vertex independently at each future history/stage. This enlarges the adversary's power and can create hybrid trajectories that no single biological/measurement model generates.

Because the rectangular set contains every static trajectory but may contain impossible model-switching trajectories, its worst-case loss is at least the static-model worst-case loss for the same policy. The gap is a conservatism diagnostic, not a biological uncertainty estimate.

## 3. Non-anticipative policy tree

A policy tree contains:
- a root action;
- a mapping from each observed first-stage outcome to the next action or STOP;
- a terminal SUPPORT/REFUTE decision for each complete observed history.

The tree must be shared across all model vertices. It can depend on observable history only. No model-vertex identifier or hidden world identity may enter an action or terminal decision.

For a two-stage policy with root action `a0`, second-stage action map `a1(y1)` and terminal decision `d(y1,y2)`, the static risk under model vertex m is:

\[ R(\pi,m)=\sum_{w,y_1,y_2} p_m(w)P_m(y_1\mid w,a_0)P_m(y_2\mid w,y_1,a_1(y_1))\,\mathbf{1}[d(y_1,y_2)\ne q(w)] \]

The primary robust objective is:

\[ R^*_{\mathrm{static}}=\min_{\pi\in\Pi_{\mathrm{nonanticipative}}}\max_{m\in V}R(\pi,m) \]

The rectangular comparator permits a distinct second-stage model choice for each first-stage history while retaining the prior and first-stage kernel from the root model. It must be computed as a separate relaxation, not silently substituted for the static model.

## 4. Synthetic two-stage known-answer construction

Worlds `QUERY_TRUE` and `QUERY_FALSE` each have prior probability 0.5 under both model vertices. There are two model vertices, `MODEL-A` and `MODEL-B`.

Stage-one calibration action:
- `CALIBRATE` emits `CAL-A` under MODEL-A, independently of the world.
- `CALIBRATE` emits `CAL-B` under MODEL-B, independently of the world.

Two candidate discriminating actions:
- Under MODEL-A, `ACTION-A` emits POSITIVE for QUERY_TRUE and NEGATIVE for QUERY_FALSE; `ACTION-B` is reversed.
- Under MODEL-B, `ACTION-B` emits POSITIVE for QUERY_TRUE and NEGATIVE for QUERY_FALSE; `ACTION-A` is reversed.

Exact static-model policy:
1. Run CALIBRATE.
2. After CAL-A, choose ACTION-A.
3. After CAL-B, choose ACTION-B.
4. POSITIVE -> SUPPORT; NEGATIVE -> REFUTE.

For either fixed model, this policy has zero terminal loss. The calibration outcome is an observable proxy for which model vertex is operating; the policy never reads the vertex ID.

The rectangular relaxation may select MODEL-A for the calibration outcome and switch to MODEL-B for the second-stage ACTION-A kernel, or vice versa. It can thereby reverse the expected outcome semantics after calibration. The same adaptive tree cannot know whether this model switch occurred.

Expected known-answer outputs:

| Metric | Expected result |
|---|---:|
| Best one-shot robust risk (ACTION-A or ACTION-B) | 0.5 |
| Static two-stage robust risk | 0 |
| Rectangular two-stage robust risk | 0.5 |
| Static non-rectangularity advantage over one-shot | 0.5 risk units |
| Static robust certificate at tolerance 0.01 | Pass |
| Rectangular robust certificate at tolerance 0.01 | Fail |
| Total worst-branch cost for calibration + second action | 2 units |

The gap shows why a globally fixed model vertex must not be replaced with an adversary that switches kernels independently at each stage.

## 5. Required invariants

- The biological world identity is fixed along a trajectory.
- The static model vertex selected by nature is fixed for the entire campaign.
- The policy tree is common across all vertices and non-anticipative.
- Every transition/observation kernel binds the exact parent history, action, world and model vertex.
- Conditional independence is never assumed silently; history-conditioned kernels are explicit unless sufficiency is proved.
- The rectangular risk is computed and reported separately from static risk.
- The rectangular model switch is not described as a plausible biological event unless supported by a real mechanism.
- Stop/continue decisions and terminal certificates are bound to the full observed history.
- Costs are summed along each branch; the maximum branch cost is reported separately from expected cost.
- Inaccessible, failed, missing or ambiguous observations are explicit outcome categories, not negative biology.
- Infeasible actions are excluded before policy enumeration.

## 6. Baselines

- best one-shot robust action from #10761;
- best fixed, non-adaptive two-action sequence;
- expert-designed sequence;
- entropy-greedy adaptive policy;
- query-directed Bayesian sequential design such as GO-CBED;
- rectangular robust adaptive policy;
- clairvoyant model-aware sequence (diagnostic only);
- exact exhaustive optimum over all finite policy trees on the synthetic corpus.

All methods receive the same model vertices, world priors, conditional kernels, cost, horizon and terminal loss.

## 7. Industry and research comparison

Goal-oriented causal Bayesian experimental design now explicitly learns adaptive sequences for a specified causal quantity. Its reported tests include synthetic causal models and simulated gene-regulatory networks; laboratory campaign performance remains a separate question. We should compare under matched query, action set and budget, then extend the objective to static-model worst-case risk.

Non-rectangular robust MDP literature explains why ambiguity sets coupled across time are difficult: rectangularity supports dynamic programming but may not preserve the fact that one underlying model governs an entire trajectory. Recent 2026 work studies fixed stationary kernels and history-dependent policies without requiring the rectangular Bellman principle.

Relevant sources:
- Zhang, Dong, Liu and Huan (2026), *Goal-oriented sequential Bayesian experimental design for causal learning*, TMLR: https://uq.engin.umich.edu/causal-learning.html
- Wang and Si (2026), *Non-Rectangular Average-Reward Robust MDPs: Optimal Policies and Their Transient Values*, arXiv:2603.00945: https://arxiv.org/abs/2603.00945
- Li, Kuhn and Sutter (2026), *Policy Gradient Algorithms for Robust MDPs with Non-Rectangular Uncertainty Sets*, SIAM Journal on Optimization 36(1): https://doi.org/10.1137/24M1631250

## 8. Promotion targets

These are targets, not results:

- 100% exact two-stage policy-tree optimum on bounded known-answer cases;
- 0 policies that condition on hidden model vertex or world identity;
- 100% correct static-versus-rectangular risk distinction;
- 0 infeasible actions in selected policy trees;
- exact agreement with an independent exhaustive oracle;
- ≥20% lower static robust terminal loss per total branch cost than the best frozen non-adaptive baseline on preregistered held-out cases, with paired uncertainty supporting improvement;
- report worst-branch and expected cost separately;
- report the static/rectangular gap rather than hiding rectangular conservatism.

## 9. Claim ceiling

A passing solver establishes policy-tree semantics conditional on the supplied finite worlds, model vertices, priors and history-conditioned kernels. It does not establish that the model vertex is truly stationary in a real biological campaign, prove the vertex set is complete, authenticate observations, or establish that a selected intervention is feasible or effective in vivo.