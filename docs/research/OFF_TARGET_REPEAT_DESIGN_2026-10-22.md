# Repeatable vs. Distinct Off-Target Intervention Design — 2026-10-22

**Status:** exact local enumeration; embedded self-test and 4/4 unit tests passed. This is a matched-budget, small-SCM pilot—not an external benchmark or an SOTA claim.

## Question

Once an experimenter models unknown graph structure, target compliance q and off-target probability r, should every intervention request be unique, or can repeating an action be more valuable because it sharpens inference about the action kernel?

This matters because two goals can compete: distinct actions explore more parts of the graph, while repeated actions provide replicate evidence about whether the actuation process is reliable. The experiment compares these policies at the same two-intervention budget.

## Protocol

Implementation: validation/off_target_replicate_design.py; tests: validation/test_off_target_replicate_design.py.

- Candidate models: a three-node Boolean chain and fork.
- Nuisance parameter grid: q ∈ {0.50, 0.75, 1.00}; r ∈ {0.00, 0.25}.
- Four regimes: (q=.75, r=0, noise=.10), (q=.75, r=.25, noise=.10), (q=.50, r=.25, noise=.10), and (q=.75, r=.25, noise=.25).
- Two out-of-class worlds: collider XOR and reversed chain.
- Six policies: joint graph EIG, graph + compliance + off-target information, joint-state entropy, an r=0 scalar-compliance model, a naive q=1/r=0 hard-do model, and a true-kernel greedy graph-EIG reference policy.
- Two designs at identical cost: distinct_only removes an action after using it; repeatable keeps the full six-action menu available and can choose the same target/value again. Both use a maximum of two interventions.
- Full grid: 4 regimes × 2 held-out worlds × 6 policies × 2 designs = 96 exact rows.

Every binary outcome and adaptive path is enumerated exactly. Repeated observations are conditionally independent given graph, q and r. The simple cyclic spill kernel is unchanged from the preceding experiment, so this is not an arbitrary target-subset model.

## Results: same two-probe budget

AUC=0.50 is chance-level ranking. Values below average over the two held-out worlds for each regime/policy. Graph information is expected reduction in the initial one-bit chain/fork entropy; the off-target-information metric is expected reduction in r entropy.

### Joint graph-EIG policy

| Regime | Design | OOD AUC | Graph information (bits) | Compliance information | Off-target information | Zero-support mass |
|---|---|---:|---:|---:|---:|---:|
| Clean, noise .10 | Distinct actions | .8796 | .3300 | .3142 | .0093 | .0000 |
| Clean, noise .10 | Repeatable | .7139 | **.4980** | .3156 | **.0196** | .0000 |
| Matched off-target, noise .10 | Distinct actions | .8519 | .3019 | .3146 | .0088 | .0000 |
| Matched off-target, noise .10 | Repeatable | .7078 | **.4353** | .3162 | **.0169** | .0000 |
| Weak off-target, noise .10 | Distinct actions | .7727 | .2324 | .4752 | .0062 | .0000 |
| Weak off-target, noise .10 | Repeatable | .6780 | **.3377** | .4843 | **.0115** | .0000 |
| Noisy off-target, noise .25 | Distinct actions | .6784 | .1238 | .2277 | .0089 | .0000 |
| Noisy off-target, noise .25 | Repeatable | .6716 | **.1661** | .2358 | **.0112** | .0000 |

## Interpretation

1. **Repeats trade OOD discrimination for sharper inference in this policy.** Across these regimes, repeatable actions substantially increase the expected information about the graph and usually about r, while surprise AUC decreases. Repeated experiments provide extra information about a single actuation pathway, but can leave candidate-versus-alternative trajectories less distinguishable under the tested surprise score.
2. **More graph information does not necessarily mean better graph decisions.** This is a two-graph entropy-reduction measure, not SHD or a full discovery-accuracy evaluation. Several other policy rows shift differently, and the tiny designed world set cannot establish a general preference.
3. **The naive hard-do model remains fragile.** In the weak-off-target regime, its in-class zero-support mass rises from .4395 with distinct actions to .6519 with repeatable actions. Repeating a misspecified action model can reinforce the wrong likelihood model rather than repair it.
4. **There is no universal repeat or never-repeat rule.** The action policy should adapt to the current posterior over graph, compliance and spill. A better design can reserve a small fraction of budget for instrumented replicate/control trials, then use distinct interventions once actuation uncertainty is low enough.

## Paradigm shift

Treat probe selection as a sequential allocation problem across three uncertainties: graph structure, action-kernel reliability and model-set inadequacy. Replicates should be selected when their expected value for q/r identification exceeds the graph-coverage opportunity cost. Instrumentation that directly records realized targets can dominate both repeated and distinct probes in many settings; it should be tested as another explicit experimental resource.

## Prior art and industry comparison

- **Causal Discovery under Off-Target Interventions (AISTATS 2024)** explicitly models action-dependent distributions over arbitrary subsets of affected vertices and studies verification and search. Our q/r cyclic-spill kernel is narrower and should be compared against their target-set model: https://proceedings.mlr.press/v238/choo24a.html
- **MetaCaDI (UAI 2026)** uses multi-environment/meta-learning to identify unknown intervention targets: https://proceedings.mlr.press/v337/ong26a.html
- **SCOUT (ICML 2026)** studies cyclic causal discovery with soft interventions and unknown targets: https://proceedings.mlr.press/v306/turkoglu26a.html
- **ACDB (September 2026)** separates skeleton recovery, directed-graph recovery and intervention efficiency in an explicit common evaluation protocol; its current setting differs from this Boolean intervention-kernel test: https://arxiv.org/abs/2609.31675
- **CausalArena (September 2026)** spans synthetic, semantic-operational and formula-grounded SCMs, with rankings varying across families: https://arxiv.org/abs/2609.11897

No external baseline is run here. We should not compare these toy AUC values numerically to published graph-recovery metrics.

## Proposed qualification gates (not achieved)

- Expand from the cyclic single-collateral rule to action-dependent distributions over all intervention target subsets on small graphs.
- Evaluate repeatable, distinct-only, randomized, EIG and adaptive replicate-allocation policies at matched budgets 1–10.
- Add instrumented-control experiments that directly reveal the realized target set; quantify whether one direct telemetry observation is more valuable than one or more repeats.
- Report graph SHD, skeleton/directed precision-recall, OOD AUROC/AUPRC, q/r calibration, target-set log score/Brier, zero-support mass, false-alarm control and intervention cost.
- Use at least 1,000 independently seeded worlds across held-out topology, mechanism and action-kernel families with paired intervals clustered by SCM.
- Only claim superiority after a preregistered improvement over appropriate stochastic-intervention baselines with confidence intervals excluding zero and no validity-endpoint regression.

## Reproduction and provenance

    python3 validation/off_target_replicate_design.py --self-test
    python3 -m unittest -v validation/test_off_target_replicate_design.py
    python3 validation/off_target_replicate_design.py --source-git-blob 796dbcaaa61ba612239db9f4c65be029a381a555 --output /tmp/off-target-repeat-design.json

- Implementation Git blob: 796dbcaaa61ba612239db9f4c65be029a381a555; SHA-256 c13b88743199933088169d7cb68bc6d24c3f2c08328f9b476e9e9bee979bd82e; 9,398 bytes.
- Test Git blob: a7c8010faa2634e1bdfb5da58ab828582d6c0586; SHA-256 6cc8e537d1d5b355567bdb556859402ea60981befdbfcb3efa70af246ce1fd36; four tests.
- Embedded self-test and 4/4 tests passed locally on these exact source/test bytes.
- Full 96-row artifact: 65,497 bytes; SHA-256 428d17dffb74f48dfe8d1c0359bd5302d89ac2165106560fd09228f9d5386cd0; artifact binds the source blob above.
- Local evidence is not a GitHub Actions pass.
