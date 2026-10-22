# Family-Held-Out SCM Benchmark Scaffold — 2026-10-22

**Status:** generated ground-truth test suite and graph-scoring utility. This is not a causal-discovery algorithm evaluation and contains no claimed performance advantage.

## Purpose

The earlier noisy-SCM and selective-prediction pilots were deliberately small and hand-designed. The next step is a deterministic generator that can separate generalization across graph topology from generalization across causal mechanism family. It also makes graph skeleton accuracy and edge-direction accuracy separate outputs, instead of allowing either to hide inside a single entropy metric.

## Implementation

- Generator: \`validation/family_heldout_scm_benchmark.py\`
- Tests: \`validation/test_family_heldout_scm_benchmark.py\`
- CLI emits versioned JSON with per-world IDs, graph parents, directed edges, Boolean truth tables, family labels, and split labels.
- Node count supports 4–8 for this exact finite-state generator; this run uses 6 nodes.
- All declared graph parents are checked to be functionally essential in their local Boolean mechanism; DAG structure, truth-table bounds, deterministic generation, graph score edge cases, and supplied-parent acyclicity are tested.
- All world IDs hash the schema version, topology/mechanism labels, seed, parent graph and truth tables.

## Generated run

Configuration: 6 nodes; 6 topology families; 5 mechanism families; 4 deterministic seeds per topology × mechanism cell; 120 SCMs total.

Topology families:
- Chain, fork, collider, diamond, sparse random DAG, dense random DAG.

Mechanism families:
- AND, OR, XOR/parity, threshold, random Boolean truth tables constrained so every declared parent is functionally essential.

The benchmark uses four separate split labels:

| Split | Cases | Definition |
|---|---:|---|
| \`train_id\` | 36 | Topology and mechanism families designated for training |
| \`topology_ood\` | 36 | Entire topology families held out; mechanisms are in the training family set |
| \`mechanism_ood\` | 24 | Entire mechanism families held out; topologies are in the training family set |
| \`combined_ood\` | 24 | Both topology and mechanism families held out |

These labels identify benchmark regimes. This generation step does not itself train a discovery algorithm or prevent an external user from leaking held-out examples; runners must enforce the split protocol.

## Graph scoring contract

The current helper scores a fully directed predicted graph and reports:
- Skeleton adjacency precision, recall and F1.
- Exact directed-edge precision, recall and F1.
- True/false positive and false negative counts.
- Structural Hamming distance (SHD), with a missing/extra adjacency costing one edit and a reversed common adjacency costing one edit.
- Orientation reversals, edge-count delta and total edge counts.

A reversed directed edge contributes one false positive and one false negative to exact directed-edge F1, even though SHD counts it as one graph edit. Skeleton and direction scores are deliberately reported separately. The helper rejects self-loops, out-of-range endpoints and duplicate directed edges.

This is a fully directed-DAG scoring contract, not an ACDB adapter: it does not yet score CPDAG/undirected predictions, compelled-edge recovery, intervention efficiency or interventional distance (SID).

## Local validation and provenance

Executed locally against exact committed source/test Git blobs:
- Source blob: \`36b6c75a0f00a3056ddaf282a2f0b2008eb97621\`
- Test blob: \`f9e8fb471f446d718477244b44ea05610fe4952b\`
- Embedded self-test: passed.
- Unit tests: **9/9 passed**.
- Generated artifact: 31,607 bytes; SHA-256 \`aeb54c62299144e28013be786d9311f0b8e6d92c2d95bf749f514419dd0f9117\`.
- Run manifest: \`validation/FAMILY_HELDOUT_SCM_2026-10-22_RECEIPT.json\`.

Local test/source execution is independent of GitHub Actions. No passing remote checks are claimed.

## Why the next gate is an external adapter, not another toy metric

The Active Causal Discovery Benchmark (ACDB) separates skeleton recovery, direction recovery, and intervention efficiency under a common observe/intervene/submit protocol. CausalArena highlights rank changes across SCM graph families, causal mechanisms, and evaluation protocols. CauScale reports a distinct scale target—up to 1,000-node inference—so scale should be measured separately from exact-small-world correctness. These results motivate our split and score contract but are not numbers this scaffold has reproduced.

- ACDB: https://arxiv.org/abs/2609.31675
- CausalArena: https://arxiv.org/abs/2609.11897
- CauScale: https://proceedings.mlr.press/v306/peng26c.html
- TCD-Arena (temporal assumption violations): https://tcd-arena.github.io/
- Falsifying Causal Graphs With Outlier Events (UAI 2026): https://proceedings.mlr.press/v337/orchard26a.html

## Limitations

This scaffold only generates deterministic Boolean SCM ground truths. It does not generate observational/interventional sample matrices, implement a causal discovery estimator, infer noisy mechanisms, model latent confounding, simulate failed interventions, compute statistical uncertainty intervals, or run the ACDB/CausalArena protocols. A six-family toy generator cannot establish external generalization. The generated cases are a start for method evaluation, not an evidence-backed performance result.

## Immediate next steps

1. Add a sample/rollout simulator with an explicit observation model, state prior, independent noise and hard-do semantics.
2. Implement the full ACDB score schema, including legitimate unresolved/undirected edges, baseline floors and intervention efficiency.
3. Evaluate fixed/random/cost-aware/greedy acquisition and a known-truth oracle on paired worlds with identical action budgets.
4. Add missing-variable, latent-confounding, intervention-failure and noise-shift variants. Preserve family-held-out tests and disclose each shift separately.
5. Add paired uncertainty intervals and a benchmark-level falsification/abstention lane; score selective error against coverage rather than hiding errors by abstaining.
6. Only then compare against established methods on contract-matched data. The objective is improved held-out graph recovery at equal budget, or similar recovery with fewer interventions—not lower entropy alone.