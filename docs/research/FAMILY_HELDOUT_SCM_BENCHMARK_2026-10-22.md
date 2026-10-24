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

## Layered mixed-graph scoring — follow-up

The scoring utility now also includes \`layered_graph_scores\`, which accepts directed predictions plus legitimate undirected predictions and separately reports:
- skeleton precision/recall/F1;
- compelled-direction precision/recall/F1 against caller-supplied CPDAG compelled arrows;
- exact directed-DAG precision/recall/F1;
- a documented DAG-SHD convention, including one error for each true adjacency left unresolved;
- orientation coverage and intervention count relative to a caller-supplied minimum intervention-set size.

This is **ACDB-inspired, not yet an exact ACDB adapter**. The caller must provide the true CPDAG's compelled arrows and benchmark-computed minimum intervention set; this code does not derive them. The unresolved-orientation contribution to \`dag_shd\` is an explicit local convention that still needs a parity check against the official implementation before being reported as an ACDB score. The current intervention output is a raw count ratio, not an assertion about ACDB's exact scalar efficiency calculation.

The suite now rejects a supplied SCM whose actual parent lists contain a directed cycle, and layered scoring rejects duplicate or reciprocal orientations and any adjacency that is declared as both directed and undirected.

Latest exact local validation:
- Source blob: \`48f7a9e1e6b25ec9be7e8f90f10f1bfa3418b94f\`
- Test blob: \`5cd2c2d7adc3caa78f6bc87b9903f35b2525c7c3\`
- Embedded self-test passed; **11/11 tests passed**.


## Deterministic sample simulator — follow-up

Added \`mechanism_output\`, a deterministic topological-order routine, and \`sample_scm\`. The sampler supports observational rows and a single hard intervention \(do(X_i=v)\), with optional independent mechanism-output flips. The intervention replaces the target equation and bypasses its mechanism-noise flip; descendants use the intervened value. Inputs are validated, sampling is deterministic under the fixed seed, and a canonical JSON SHA-256 binds each row set to the SCM ID, seed, sample count, noise setting, and intervention contract.

Smoke experiment: one 5-node AND-chain, 32 observational samples at noise probability 0.1, 32 samples for \(do(X_0=0)\), and 32 for \(do(X_0=1)\). Results are in the local sample artifact. The deterministic noiseless hard-do regression confirms all chain nodes become 0 under \(do(X_0=0)\) and all become 1 under \(do(X_0=1)\); this follows the deliberately constructed AND chain and is not evidence of discovery performance.

Exact local validation on matching committed blobs:
- Source blob: \`a67b65fe34fe40533bd3425f818b94b152c21acc\`
- Test blob: \`9879240cd7bc4e64f19d76350f263c0af8642160\`
- Embedded self-test passed; **14/14 unit tests passed**.
- Sample smoke artifact: 2,316 bytes, SHA-256 \`b3647893810cac0fa74e557885ea953c7e26ad8787341c9552058b33fd171cd5\`.

This enables a first reproducible sample-level evaluation protocol, but the code still does not infer a graph from samples, compute confidence intervals, handle latent variables/unknown interventions, or reproduce ACDB's exact scoring/efficiency contract. The benchmark is infrastructure only until discovery algorithms are evaluated.


## First sample-to-graph baseline now available

The next report, \`docs/research/FAMILY_HELDOUT_MDL_BASELINE_2026-10-22.md\`, adds a paired comparison of observational-only versus mixed observational + hard-interventional samples using an exhaustive four-node enumerative-MDL graph selector. It evaluates 120 worlds and keeps topology-OOD, mechanism-OOD and combined-OOD results separate. The result is mixed: observational F1 is 0.9486 overall versus 0.9448 for the mixed intervention arm, while mean SHD improves from 0.0667 to 0.0500. The F1 advantage is not established; 110/120 worlds tie and the family-specific deltas vary. This is baseline plumbing and a diagnostic result, not a superiority claim.

Source: \`validation/family_heldout_mdl_baseline.py\` (Git blob \`3698ecce55bbd02210c9221a8e31eb2c81ac7c05\`); tests: \`validation/test_family_heldout_mdl_baseline.py\` (Git blob \`1540b1d32f82c757b22cb4df66ac029e8fa2f585\`), 6/6 passed locally. The full output has SHA-256 \`7db7c8ceeb2438039636479c919ab5ea5cf7d618608d044320e3c9d0aca49707\`. The baseline now uses an explicit discrete-model MDL code instead of a misleading continuous-parameter BIC penalty.
