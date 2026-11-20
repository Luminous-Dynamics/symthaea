# Causal Discovery Qualification Targets — Symthaea, 2026-10-22

**Status:** proposed benchmark protocol and acceptance gates, not achieved results.

## Current position

The current Symthaea work is a small, exact, three-node Boolean SCM suite. It is useful for checking intervention semantics, posterior normalization, and model-misspecification failure modes. It is not currently comparable numerically to industrial or academic state of the art.

The new selective-discovery pilot makes this limitation concrete: under one out-of-class XOR collider, posterior-confidence gating and a calibrated predictive-surprise gate both still emit incorrect graph claims. This is not a qualified robust detector. The correct response is to expand the evaluation design rather than inflate the current claim.

## Relevant public comparison points

| Work | Reported scope/results | Symthaea implication |
|---|---|---|
| [ACDB, September 2026](https://arxiv.org/abs/2609.31675) | Active discovery under a shared observe/intervene/submit contract. The paper reports PC plus greedy active orientation at 42.7% directed F1 and SHD 4.79 on its current six-level linear-Gaussian ladder; the dense v0 random-DAG floor is 23.6% directed F1. | Reproduce a matched subset. Score skeleton, directed edges, SHD and intervention efficiency separately; do not compare against toy entropy. |
| [CausalArena, September 2026](https://arxiv.org/abs/2609.11897) | 1,200 executable SCM specifications reported: 1,000 synthetic, 100 semantic-operational and 100 formula-grounded scientific worlds. The release combines public and reserved held-out cases; performance shifts across families and protocols. | Use graph-family and mechanism-family holdouts. Add semantic/formula worlds rather than relying only on generated Boolean gates. |
| [CauScale, ICML 2026](https://proceedings.mlr.press/v306/peng26c.html) | Reports inference scaling up to 1,000 nodes, training on 500-node graphs, 99.6% mAP in-distribution and 84.4% out-of-distribution, with 4×–13,000× inference speedups over prior methods on its evaluated tasks. | Build a separate 50/100/500/1,000-node scale lane. mAP, directed F1, SHD and task-conditional graph claims are different quantities; no cross-metric ranking. |
| [TCD-Arena, ICLR 2026](https://tcd-arena.github.io/) | A large time-series causal-discovery stress-testing framework; its reported study covers 33 assumption violations over roughly 30 million discovery attempts. | Adopt a versioned shift/assumption matrix when adding temporal SCMs. |
| [Falsifying Causal Graphs With Outlier Events, UAI 2026](https://proceedings.mlr.press/v337/orchard26a.html) | Develops statistical tests to falsify candidate causal graphs from outlier propagation, reporting false-positive control and power guarantees under stated assumptions. | Benchmark against formal graph-falsification tests. Predictive surprise alone is not a new method and has weak power in our current toy fixture. |
| [CausalGuard, May 2026](https://arxiv.org/abs/2605.21928) | Calibrates treatment-effect inference under causal-graph uncertainty via conformal methods; its target is causal-effect coverage, not active graph recovery. | Relevant calibration prior art, but keep treatment-effect coverage separate from graph-discovery accuracy. |

All external scores above are source-reported results under their own contracts. They are contextual targets, not numbers Symthaea has reproduced.

## Benchmark tracks

### Track A — exact small-world correctness

- 3–8 node SCMs with analytically/exhaustively calculable outcomes.
- Verify hard-do semantics, graph identity, likelihoods, posterior updates, path mass, utility and intervention-policy objectives.
- Include disconnected/no-op edges, XOR synergy, colliders, chains, forks, cancellation/unfaithfulness, redundant paths and multiple state priors.
- Gate: absolute error ≤ 1e-12 where the reference is exactly enumerable and floating arithmetic permits it.

### Track B — intervention efficiency and graph recovery

- Implement an adapter for the ACDB interaction/scoring contract wherever feasible.
- Paired worlds and equal intervention budgets for fixed, random, cost-first, PC/standard causal-discovery baselines, greedy EIG and exact oracle on tractable cases.
- Report skeleton precision/recall, directed precision/recall, F1, SHD, intervention count, utility/decision quality and runtime separately.
- Proposed stretch target: ≥10% lower SHD than the strongest inexpensive baseline at the same budget on predeclared held-out worlds, with paired confidence intervals excluding zero improvement; alternatively, similar recovery with ≥25% fewer interventions. No target is met by current pilot evidence.

### Track C — model adequacy, selective prediction and falsification

- Keep in-class worlds, graph-out-of-class worlds, mechanism-out-of-class worlds, unknown-noise worlds, omitted-variable/latent-confounding worlds and intervention-failure worlds separate.
- Calibrate on disjoint reference worlds, then evaluate on untouched graph families and noise regimes. Never tune a threshold on its test world.
- Report coverage, selective error, unconditional false-claim probability mass, false-abstention rate, out-of-class detection power, posterior/credible-set coverage, prequential log loss and calibration curves.
- Proposed stretch target: ≤10% selective graph-claim error at ≥70% claim coverage on held-out OOD families; ≤5% false rejection on in-class calibration-matched worlds; nominal 95% uncertainty coverage empirically between 90% and 98%. These targets must be reported by family, not only pooled.
- Include formal graph-falsification baselines where assumptions and implementation allow. If the detector cannot separate graph shift from mechanism/noise shift, report that limit.

### Track D — scale and resource cost

- A distinct inference/evaluation lane at 50, 100, 500 and 1,000 nodes. Do not attempt exact enumeration beyond tractable small graphs.
- Record wall-clock time, peak memory, hardware, graph density, data sample count, observation/intervention budget, environment digest and accuracy/coverage. Include at least one strong scale baseline.
- Proposed target: ≥10× lower inference time or memory at equal error and coverage on a declared workload, or a clear Pareto improvement. Require identical inputs and hardware for a performance claim; CauScale's own speedup range cannot be transplanted onto a different workload.
- Retain a separate exact-small-world track as a guard against fast but wrong implementations.

### Track E — external validity and freshness

- Add CausalArena synthetic, semantic-operational and formula-grounded families under their documented license/interfaces.
- Add temporal assumptions from TCD-Arena when a time-series model and comparable scoring contract exist.
- Reserve fresh generated cases for final test rounds and version all generator code, seeds, prior distributions, SCM digests and held-out-family IDs to reduce leakage and benchmark overfitting.

## Reproducibility and fail-closed contract

Every run must bind source Git blob, test Git blob, world/mechanism/prior digest, generator version, seed, intervention sequence, noise assumptions, scoring schema, output byte count and SHA-256. Exact-source local tests and independent GitHub Actions results are separate evidence. queued, in_progress, skipped, empty combined status and missing checks are never a pass. Missing strata, invalid posterior mass, unsupported comparisons or missing provenance produce NOT QUALIFIED.

## Immediate sequence

1. Keep the exact 3-node falsification fixture as a regression control, not a leaderboard.
2. Build a deterministic family generator for 3–8-node SCMs and hidden test families; separate true-world and learner-model specifications.
3. Add disjoint calibration/test worlds and score graph-falsification detection/false-positive curves under matched-noise and shifted-noise conditions.
4. Implement an ACDB-compatible adapter and compare policies on the same worlds, action semantics and budget.
5. Add the 50–1,000-node lane and profile resource cost independently.
6. Only describe a method as superior when the paired, held-out metric and uncertainty result support that claim. If a custom aggregate does not outperform simpler edge effects, graph tests or established baselines, remove it instead of renaming it.

The paradigm shift is therefore not “maximize entropy reduction at any cost.” It is a causal-discovery system that searches efficiently, checks its own model class, falsifies wrong explanations, abstains in a calibrated way, and exposes a reproducible accuracy–coverage–cost frontier across families and scales.