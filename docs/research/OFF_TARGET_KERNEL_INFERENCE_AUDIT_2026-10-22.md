# Action-Dependent Off-Target Causal Inference Pilot — 2026-10-22

**Status:** exact local enumeration; embedded self-test and 8/8 unit tests passed on the exact current source and test Git blobs. This is a diagnostic pilot, not a general stochastic intervention solver, external benchmark entry, or state-of-the-art claim.

## Research question

The earlier experiments modeled intervention reliability with one scalar target-compliance probability q. That misses “fat-hand” effects: an attempted intervention can also set a collateral variable. This pilot adds a second parameter r, the probability of a collateral effect conditional on successful target actuation, and maintains a posterior over graph × q × r hypotheses.

This is established research territory. Choo, Shiragur and Uhler (AISTATS 2024) model an attempted intervention as affecting a random subset of variables drawn from an action-dependent distribution and study verification and search. Our contribution here is a compact exact-enumeration stress test that integrates a small off-target kernel with graph/compliance inference and action selection; it is not a new general theory of off-target discovery.

## Exact model and protocol

Implementation: validation/off_target_kernel_inference.py; tests: validation/test_off_target_kernel_inference.py.

- Candidate graph models: three-node Boolean chain and fork.
- Compliance q ∈ {0.50, 0.75, 1.00}; spill r ∈ {0.00, 0.25}.
- For attempted action on variable i at value v: with probability 1−q nothing is clamped; with probability q(1−r) only i is clamped to v; with probability qr both i and its cyclic successor (i+1 mod 3) are clamped to v. Remaining nodes follow noisy structural mechanisms.
- Reference mechanisms: collider AND and OR. Held-out worlds: collider XOR and a reversed chain.
- Four regimes: clean q=.75/r=0 at noise .10; off-target q=.75/r=.25 at noise .10; weaker compliance q=.50/r=.25 at noise .10; and q=.75/r=.25 at noise .25.
- Six policies: joint graph-EIG; graph + compliance + off-target information gain; joint-state entropy reduction; clean scalar-compliance (assumes r=0); naive perfect hard-do (q=1, r=0); and a known-kernel reference policy given true q/r but still using greedy graph-EIG, not a globally optimal oracle.
- Each policy has a two-distinct-action budget. All binary outcomes and adaptive paths are enumerated exactly. The full output has 48 policy/regime/held-out-world rows.

The spill mechanism is deliberately simple and hand-specified. It is not an arbitrary subset-of-targets model: it only permits no clamping, target-only clamping, or target plus one cyclic collateral variable. This makes exact enumeration tractable but limits external validity.

## Results

AUC 0.50 is chance-level ranking. Graph error is graph MAP error conditional on paths with nonzero predictive support. Zero-support mass is the true in-class probability mass assigned exactly zero likelihood by the learner.

| Regime | Policy | OOD AUC | Graph information (bits) | Graph MAP error | Zero-support mass |
|---|---|---:|---:|---:|---:|
| Clean, noise .10 | Joint graph-EIG | .8796 | .3300 | .2080 | .0000 |
| Clean, noise .10 | Joint graph + q/r | .9045 | .3300 | .2080 | .0000 |
| Off-target, noise .10 | Joint graph-EIG | .8519 | .3019 | .2723 | .0000 |
| Off-target, noise .10 | Joint graph + q/r | .8914 | .3019 | .2723 | .0000 |
| Weak off-target, noise .10 | Joint graph-EIG | .7727 | .2324 | .3236 | .0000 |
| Weak off-target, noise .10 | Joint graph + q/r | .8090 | .2324 | .3236 | .0000 |
| Noisy off-target, noise .25 | Joint graph-EIG | .6784 | .1238 | .3527 | .0000 |
| Noisy off-target, noise .25 | Joint graph + q/r | .7142 | .1238 | .3527 | .0000 |

The graph + q/r acquisition has the same reported graph-information score as graph-EIG in these scenarios, with slightly higher OOD-surprise AUC. Off-target-information gain remains low; the two-probe budget is not enough to reliably identify the spill probability. In the off-target regime, the restricted r=0 model reports greater graph-information reduction (.3927 bits) than joint inference (.3019), yet its conditional graph MAP error is slightly worse (.2762 versus .2723) and it is pinned to the wrong off-target model (r=0; posterior-mean squared error .0625). Entropy reduction can therefore reward a restricted model for becoming more certain without becoming more correct.

The naive perfect-hard-do model assigns zero likelihood to substantial valid in-class probability mass: mean .2249 in the matched off-target regime, .4395 with weaker compliance, and .2090 in the noisier regime. Joint graph/q/r inference has zero zero-support mass in this pilot because true q and r fall on its declared grid. That is a design-conditioned result, not a universal guarantee.

The known-kernel policy receives true q/r but chooses probes greedily for graph EIG; it is not a global optimality oracle for OOD detection.

## Paradigm shift

Model the intervention as a probabilistic actuation channel, not a Boolean command with guaranteed effect. Jointly represent graph structure, mechanism parameters, intervention compliance, action-dependent collateral targets, measurement noise and omitted-model uncertainty. Choose experiments on a Pareto surface across graph identification, actuator-model identification and model-set falsification. If observations lie outside every joint hypothesis, return NOT_IDENTIFIED / ACTION_OR_MODEL_MISMATCH rather than forcing a graph claim.

## External research and comparison

- **Causal Discovery under Off-Target Interventions (AISTATS 2024)** models random action-dependent subsets of intervened vertices and studies verification and search with approximation algorithms. This is richer than the current cyclic-spill model and should become a direct comparator: https://proceedings.mlr.press/v238/choo24a.html
- **MetaCaDI (UAI 2026)** treats unknown intervention-target discovery across environments as a meta-learning task: https://proceedings.mlr.press/v337/ong26a.html
- **SCOUT (ICML 2026)** studies cyclic causal discovery under soft interventions with unknown targets: https://proceedings.mlr.press/v306/turkoglu26a.html
- **ACDB (September 2026)** separates skeleton recovery, directed-graph recovery and intervention efficiency; its linear-Gaussian perfect-intervention results are not numerically comparable to this Boolean pilot: https://arxiv.org/abs/2609.31675
- **CausalArena (September 2026)** includes 1,200 executable SCMs across synthetic, semantic-operational and formula-grounded families and reports ranking shifts across families: https://arxiv.org/abs/2609.11897
- **TCD-Arena (ICLR 2026)** evaluates time-series discovery under 33 assumption violations and a much larger evaluation volume: https://tcd-arena.github.io/

## Proposed qualification gates (not achieved)

- Replace the single cyclic collateral rule with action-dependent distributions over target subsets, including no target and multiple off-target effects.
- Sweep intervention budgets 1–10 and test whether q/r posterior calibration improves with more probes. Report target-set log score/Brier, graph SHD, skeleton/directed precision-recall, OOD AUROC/AUPRC, false-alarm control, intervention cost and abstention.
- Use at least 1,000 independently seeded worlds partitioned by topology, mechanism, action kernel and measurement regime.
- Compare against established stochastic intervention/verification methods and matched-budget PC/FCI/GES/IGSP or other assumption-appropriate baselines.
- Use held-out action kernels, disjoint calibration worlds and paired intervals clustered by SCM. Do not tune the reference bank or coefficient on evaluation worlds.
- Claim superiority only when a preregistered graph-recovery or calibration metric improves with paired uncertainty excluding zero, without degrading model-adequacy and actuation-validity metrics.

## Reproduction and provenance

    python3 validation/off_target_kernel_inference.py --self-test
    python3 -m unittest -v validation/test_off_target_kernel_inference.py
    python3 validation/off_target_kernel_inference.py --source-git-blob aadfb064aa8369a1d1e5417bf089703ce9f5d0fb --output /tmp/off-target-kernel.json

- Implementation Git blob: aadfb064aa8369a1d1e5417bf089703ce9f5d0fb; SHA-256 cd57596385de170e1859d8de513e6daa241251b785d1c777b165e9b389c50032; 18,527 bytes.
- Test Git blob: 525858c74a83cb9476000e6c2ddafda92c5bed63; SHA-256 fc03f079b98b3073ff5162601b304fd10420e5c13f6aa4275f493e95a654c103; 3,285 bytes; 8 tests.
- Embedded self-test passed; 8/8 tests passed locally against the exact GitHub source/test blobs.
- Full 48-row result: 37,320 bytes; SHA-256 f54c7f56d21f4ba53e5fafe600e545cf1846310ccbc738877d8ba5cd2a92dc4d; JSON binds the source blob above.
- Local evidence is not a GitHub Actions pass; no remote pass is inferred.
