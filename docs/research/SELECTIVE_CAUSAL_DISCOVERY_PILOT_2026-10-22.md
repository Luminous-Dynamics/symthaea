# Selective Causal Discovery Pilot — 2026-10-22

**Status:** locally tested exact-enumeration research prototype; **not qualified for external benchmark claims**.

## Research question

Can a causal-discovery agent reduce confidently wrong graph claims by combining active interventions with an explicit abstention decision, without silently optimizing away valid coverage?

This is the next layer after the out-of-class SCM audit. A learner receives two candidate SCMs (a chain and a fork) and can make at most two hard interventions. The in-class test world is the candidate chain. The out-of-class world is an XOR collider absent from the learner's candidate graph set. The actual noise rate is either the learner's assumed 0.1 or a shifted 0.25.

## Method

Source: validation/selective_causal_discovery.py; tests: validation/test_selective_causal_discovery.py.

Three acquisition policies are compared: fixed order, greedy expected information gain, and an exact two-step oracle under the learner's assumed model. Decision rules include always commit, posterior-confidence gating, fixed per-observation predictive-surprise cutoffs, and a calibrated surprise cutoff.

For each action, all compatible binary outcomes are enumerated exactly under the data-generating SCM. The learner computes predictive probabilities and Bayesian updates under its candidate set and assumed noise. A surprise gate abstains when \(-\log_2 P(\text{observed outcome}\mid\text{current candidate mixture})\) exceeds its cutoff. A separate calibration routine chooses a max-surprisal cutoff whose exact rejection mass is at most 5% on the known in-class chain fixture.

**Calibration caveat:** the toy calibration routine has access to the in-class ground-truth fixture. This is an optimistic benchmark calibration device, not a deployable calibration protocol and not a finite-sample guarantee for unseen systems. A real-world protocol would require disjoint calibration environments and explicit exchangeability/shift assumptions.

## Key exact results

The tables below use the assumed-model oracle for acquisition. Selective error is the fraction of issued claims that are wrong; when the true graph is out of candidate support, every issued graph claim is wrong by construction.

| World / actual noise | Decision rule | Claim coverage | Selective error | Wrong-claim probability mass | Early surprise abstention |
|---|---|---:|---:|---:|---:|
| In-class chain / 0.10 | Always commit | 100.00% | 8.29% | 8.29% | 0.00% |
| In-class chain / 0.10 | Posterior MAP ≥ 0.8 | 89.38% | 9.28% | 8.29% | 0.00% |
| In-class chain / 0.10 | Calibrated surprise | 99.00% | 8.36% | 8.28% | 1.00% |
| Out-of-class XOR collider / 0.10 | Always commit | 100.00% | 100.00% | 100.00% | 0.00% |
| Out-of-class XOR collider / 0.10 | Posterior MAP ≥ 0.8 | 89.38% | 100.00% | 89.38% | 0.00% |
| Out-of-class XOR collider / 0.10 | Calibrated surprise | 91.00% | 100.00% | 91.00% | 9.00% |
| Out-of-class XOR collider / 0.25 | Always commit | 100.00% | 100.00% | 100.00% | 0.00% |
| Out-of-class XOR collider / 0.25 | Posterior MAP ≥ 0.8 | 74.22% | 100.00% | 74.22% | 0.00% |
| Out-of-class XOR collider / 0.25 | Calibrated surprise | 81.25% | 100.00% | 81.25% | 18.75% |

For the calibrated rule, exact in-class rejection masses and cutoffs were:
- Fixed order: cutoff 4.3219 bits; in-class rejection 0.18%; out-of-class rejection 7.38% at actual noise 0.10.
- Greedy EIG: cutoff 3.6082 bits; in-class rejection 3.52%; out-of-class rejection 10.72% at actual noise 0.10.
- Assumed-model oracle: cutoff 4.3219 bits; in-class rejection 1.00%; out-of-class rejection 9.00% at actual noise 0.10.

Under noise shift to 0.25, rejection rates rise even when the graph is the in-class chain: the assumed-model oracle rejects 6.25% and greedy EIG rejects 17.97% in this specific fixture. That is evidence the monitor is sensitive to noise misspecification, not proof that it can distinguish graph misspecification from mechanism/noise shift.

## Interpretation: the honest result is a trade-off, not a win

1. **Posterior confidence is not a model-adequacy test.** A 0.8 MAP graph gate still emits claims in 89.38% of out-of-class paths for the assumed-model oracle; every emitted claim is wrong.
2. **Predictive surprise detects only a subset of misspecification.** Even the benchmark-calibrated gate fails to detect many out-of-class outcomes, leaving 91% claim coverage at matched noise for the assumed-model oracle.
3. **Calibration can expose, not erase, the trade-off.** On this tiny exact fixture, calibrated surprise reduces wrong-claim mass from 100% to 91% at matched noise, but does not reduce error conditional on committing: it remains 100%.
4. **Two probes are not enough to establish a useful robust detector.** The current fixture is intentionally too small to estimate general detection power or generalization.

The next algorithmic advance should not be a more aggressive fixed surprisal cutoff. It should combine diverse interventions, held-out predictive checks, graph-level falsification tests, and calibrated abstention. Candidate graph sets should be expandable or return NOT_IDENTIFIED; confidence should not be treated as evidence of model adequacy.

## Reproduction and provenance

From the repository root:

~~~sh
python3 validation/selective_causal_discovery.py --self-test
python3 -m unittest -v validation.test_selective_causal_discovery
python3 validation/selective_causal_discovery.py \
  --source-git-blob 1b17b9bf81c8519e5520e5c1a765c2cde64dcc62 \
  --output /tmp/selective-causal-results.json
sha256sum /tmp/selective-causal-results.json
~~~

Local run:
- Source Git blob: 1b17b9bf81c8519e5520e5c1a765c2cde64dcc62.
- Test Git blob: 1de489f9d75b475aa58cb4df66ac029e8fa2f585.
- Embedded self-test passed; all 8 unit tests passed locally.
- Full machine-readable output: 60,971 bytes; SHA-256 aa94e2aee4f6b6eed43eb1f0a1699bc0e3869669862e7bae48c71ba7b22c8068; the artifact binds the source Git blob above.
- The experiment is reproducible local evidence, not a GitHub Actions pass. Do not translate it into an ACDB/CausalArena ranking.

## How this connects to current research

- [ACDB (September 2026)](https://arxiv.org/abs/2609.31675) separates graph skeleton recovery, directed DAG recovery, and intervention efficiency; its reported linear-Gaussian results are not numerically comparable to this Boolean fixture.
- [CausalArena (September 2026)](https://arxiv.org/abs/2609.11897) emphasizes graph/mechanism-family diversity, semantic and formula-grounded worlds, and risks from benchmark/pretraining overlap.
- [Falsifying Causal Graphs With Outlier Events (UAI 2026)](https://proceedings.mlr.press/v337/orchard26a.html) studies statistical falsification of candidate graphs and reports false-positive control and power guarantees under its assumptions. It is a stronger methodological reference than raw posterior confidence.
- [CausalGuard (May 2026)](https://arxiv.org/abs/2605.21928) is relevant for graph uncertainty and coverage; its target is causal-effect inference with conformal calibration, not active graph discovery.

Next gates: a family-held-out SCM generator; disjoint calibration/test worlds; calibration under graph and noise shift; latent confounding/omitted variables; failed and off-target interventions; graph SHD and skeleton/direction precision-recall; false-positive control and detection power curves; and paired uncertainty intervals over generated worlds.