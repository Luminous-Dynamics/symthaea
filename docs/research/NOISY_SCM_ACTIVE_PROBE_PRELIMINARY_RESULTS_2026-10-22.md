# Noisy Multi-Node SCM Active-Probe Pilot — Preliminary Results

**Date:** 2026-10-22  
**Status:** exploratory, **NOT QUALIFIED** for external performance or SOTA claims.

## Question

Can adaptive interventions reduce uncertainty about a causal graph more efficiently than fixed, random, or cost-only probe selection when the candidate models have noisy structural equations?

This moves the earlier four-probe, noiseless two-input mechanism toy toward graph recovery. The pilot uses 3-node labeled DAG candidates, Boolean structural equations, known independent mechanism-noise probability, and exact enumeration of all possible outcomes.

## Contract

- 40 deterministic instances, generated from seeds 0–39.
- 3 nodes; 8 candidate SCM hypotheses per instance, with integer prior weights 1–5.
- Candidate DAGs alternate sparse/dense edge-generation settings; every declared parent is functionally essential to its node's Boolean mechanism.
- Noise probability 0.1, known to the inference policy.
- Six allowed interventions: hard (do(X_i=0)) and (do(X_i=1)) for each node. The intervention replaces the target mechanism; descendant mechanisms continue to evaluate with the assigned node value.
- Cost budget: 2 units. Action costs (1–2) and disturbance annotations (0–2) are deterministic synthetic labels independent of the hidden SCM. They are **not measured real-world cost or safety risk**.
- Exact enumeration, not Monte Carlo: posterior outcome probabilities and Bayesian updates enumerate all assignments consistent with each hard intervention.
- Policies: fixed order, seeded random order, cost first, greedy graph expected information gain, greedy graph information per synthetic cost-plus-disturbance, and a finite-horizon exact oracle.
- Primary objective: expected posterior entropy over distinct graph structures. The oracle uses synthetic disturbance, cost and posterior MAP-graph mass only as tie-breakers. Model/mechanism entropy is reported separately.

## Results

Means over the 40 instances. Lower graph/model entropy is better; higher MAP-graph posterior mass is better. Cost and disturbance are separate measures.

| Policy | Graph entropy (bits) ↓ | Model entropy (bits) ↓ | MAP-graph posterior mass ↑ | Cost ↓ | Synthetic disturbance ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed order | 1.470067 | 1.593791 | 0.597751 | 2.000000 | 1.725000 |
| Random | 1.461736 | 1.588061 | 0.588787 | 2.000000 | 1.625000 |
| Cost first | 1.341856 | 1.461405 | 0.627067 | 1.875000 | 1.225000 |
| Greedy expected information gain | 1.337145 | 1.487949 | 0.650999 | 1.975000 | 1.827364 |
| Greedy information / (cost + 0.25 × disturbance) | 1.253444 | 1.375766 | 0.666995 | 1.875000 | 1.459550 |
| Exact graph-entropy oracle | 1.213775 | 1.337179 | 0.685286 | 1.975000 | 1.973496 |

The exact oracle reduced mean graph entropy by **0.256292 bits (17.4%)** versus fixed order on this generated suite. The cost-efficiency heuristic ended 0.039669 bits above the oracle, with 0.100 less expected cost and 0.514 less synthetic disturbance. Cost-first had the lowest disturbance, but left more graph uncertainty. This is a visible trade-off, not a universal winner.

These numbers are **not comparable to an ACDB score**: the model space, inference contract, number of nodes, noise model, interventions and scoring are not yet a contract-matched reproduction. They also do not establish that this policy generalizes outside this generator.

## Reproduction command

From the repository root, after the source and suite have been aligned and reviewed:

```bash
python validation/noisy_scm_probe_benchmark.py --self-test
python -m unittest -v validation.test_noisy_scm_probe_benchmark
SYMTHAEA_SOURCE_SHA="$(git rev-parse HEAD)" \
  python validation/noisy_scm_probe_benchmark.py \
  --instances 40 --budget 2 --nodes 3 --hypotheses 8 --noise 0.1 \
  --source-sha "$(git rev-parse HEAD)" \
  --output /tmp/noisy-scm-probe.json
sha256sum /tmp/noisy-scm-probe.json
```

## Local validation and provenance caveat

- Embedded self-test passed on the local execution copy.
- All **10 unittest tests passed** locally. They cover determinism, hard-do behavior (including invariance to the target's replaced mechanism), exact probability normalization, parent essentiality, cost bounds, input validation, and the oracle's declared graph-entropy objective.
- Test file Git blob on the branch: `c69e398b53170d355fcbb9f834cd79b4e175f09f`. The locally run test file matched that blob.
- The earlier local mirror/artifact listed in the initial draft has been superseded by the **Receipt update** below. Use the later source-mirror blob and JSON digest there; both entries preserve the same qualification caveat that the execution mirror differs from the branch source.

## Limits and next gates

This is still a tiny finite candidate-world pilot. It assumes the correct SCM is among the hypotheses, knows the noise rate, contains no latent confounding, restricts interventions to hard do-actions, uses synthetic costs/disturbances, and evaluates only 3-node graphs in the reported run. It does not test observational discovery, unknown/noise-misspecified models, hidden variables, cycles/time-series, intervention failures, or held-out mechanism-family generalization.

Before qualification:
1. Match the executed source bytes to the committed blob and rerun all tests plus the 40-case report.
2. Add hidden-confounder / omitted-variable cases, misspecified noise, non-functional candidate edges, intervention failure, and held-out mechanism/topology families.
3. Compare random, fixed, cost-first, greedy-EIG and exact-oracle policies with ACDB under identical candidate models, action budgets and graph metrics.
4. Add graph skeleton/direction precision-recall, structural Hamming distance, calibration, graph posterior coverage and paired uncertainty intervals.
5. Test CausalArena's family/protocol-shift concern; do not combine representation-recovery scores with graph-recovery scores.
6. Replace synthetic disturbance only when operationally justified values or a clearly stated task-specific damage model exists.

Relevant prior art: [ACDB (2026)](https://arxiv.org/abs/2609.31675), [CausalArena (2026)](https://arxiv.org/abs/2609.11897), [TCD-Arena (ICLR 2026)](https://tcd-arena.github.io/), [active learning for intervention design (Nature Machine Intelligence, 2023)](https://www.nature.com/articles/s42256-023-00719-0).

No novelty, SOTA, general causal discovery, or real-world safety claim is made.


## Receipt update — local rerun

A fresh local run completed on 2026-10-22 after rerunning the embedded self-test and all ten unit tests. The test source matches branch blob `c69e398b53170d355fcbb9f834cd79b4e175f09f`. The executed local source mirror blob was `ae0678017d1b42b266d207b2869108ae92080284` (SHA-256 `18a672c4267e297e5ec7ff10c094aa52ee3d244f3f5a965c3a9fcd87dcc9295c`). The output JSON was 87,416 bytes, SHA-256 `8cd371d931f7d01c96fd8c8658bd9428d3f4067bf7e5bdb9bd9d8da9f3db1e48`; its `source_sha` field binds to that local mirror blob.

**Still not exact-current-branch evidence:** the source blob currently on the research branch is `7b92460b60c6722681389981d8cbfbaa80122cf5`, which differs from the executed mirror. The metric values are reproduced, but the committed-source byte mismatch remains a qualification blocker. Do not call this a test pass for the current GitHub source or combine this receipt with skipped CI runs.


## Receipt update — exact current-branch source rerun (2026-10-22)

The previous provenance mismatch is now resolved for the 40-case pilot by a fresh run against the **exact source bytes currently committed on this branch**. The older mismatched mirror receipt remains above as history; it is superseded by this one for reproduction purposes.

- Executed source Git blob: `7b92460b60c6722681389981d8cbfbaa80122cf5` (matches the branch source blob exactly).
- Executed test Git blob: `c69e398b53170d355fcbb9f834cd79b4e175f09f` (matches branch tests exactly).
- Embedded self-test: PASS locally.
- Unit suite: **10/10 passed locally** against the exact committed source and tests.
- Run: 40 seeds (0–39), 3 nodes, 8 hypotheses, noise 0.1, intervention budget 2; command and metrics match the original table above.
- Exact generated JSON: 87,416 bytes; SHA-256 `3fab6c0dbccba5986de9a36a46f1e2fdb5cd08201512bcb5f940a4dc004d1302`; `source_sha` is the committed source Git blob above.
- Local full result artifact is available as `validation/NOISY_SCM_PROBE_40_CASE_EXACT_BRANCH_2026-10-22.json` in the research workspace.

This closes the **source-byte mismatch for this local 40-case receipt**. It does not establish that the benchmark is externally qualified or superior to SOTA, does not fix the assumed-model limitation, and does not turn skipped CI runs into passing checks. The separate out-of-class audit documents that limitation directly.
