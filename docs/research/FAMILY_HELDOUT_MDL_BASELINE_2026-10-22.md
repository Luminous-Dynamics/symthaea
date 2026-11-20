# Family-Held-Out MDL Graph-Selection Baseline — 2026-10-22

**Status:** local exact-source baseline evaluation, not an SOTA claim and not an ACDB reproduction.

## What was implemented

The family-held-out SCM scaffold now has a sample-to-graph baseline in \`validation/family_heldout_mdl_baseline.py\` with six committed unit tests. It exhaustively considers all 543 labeled four-node DAGs and selects a parent graph using a finite-model minimum-description-length (MDL) score.

For each node, the score is:
- exact negative log-likelihood in bits under the supplied symmetric mechanism-noise probability;
- plus \(\log_2 {n-1 \choose k}\) bits to encode the parent subset of size \(k\);
- plus \(2^k\) bits to encode the Boolean truth table over that parent subset.

This is a transparent enumerative MDL objective for a discrete truth-table model. It deliberately replaces the initial BIC-named prototype: a standard continuous-parameter BIC penalty is not automatically justified when each mechanism truth table is a finite discrete hypothesis. The previous BIC prototype and tests were removed rather than leaving the incorrect criterion as an alternate implementation.

For each world, two arms have the **same total sample count**:
- Observational arm: 512 observational rows.
- Mixed arm: 256 observational rows plus 32 rows for each of eight hard interventions \(do(X_i=0)\), \(do(X_i=1)\), covering all four nodes.

Both arms use the same known noise probability (0.1), the same ground-truth SCM, and the same exhaustive DAG candidate class. For the mixed arm, observations where a node itself was intervened on are excluded from that node's structural-equation likelihood; interventional effects on the other nodes remain usable. The benchmark is paired at the SCM level.

## Exact-source provenance and local validation

- Source Git blob: \`3698ecce55bbd02210c9221a8e31eb2c81ac7c05\`
- Test Git blob: \`1540b1d32f82c757b22cb45af508f3f198dc28e3\`
- Embedded self-test: passed locally.
- Unit tests: **6/6 passed locally** against those exact committed source/test blobs.
- Worlds: 120 four-node SCMs, six topology families × five mechanism families × four deterministic seeds.
- Split counts: 36 train/ID, 36 topology-OOD, 24 mechanism-OOD, 24 combined-OOD.
- Sample totals: 512 rows per arm per SCM.
- Full JSON artifact: 146,963 bytes, SHA-256 \`7db7c8ceeb2438039636479c919ab5ea5cf7d618608d044320e3c9d0aca49707\`.
- The artifact's \`source_git_blob\` is bound to the exact committed source blob above.

The JSON includes every SCM's seed, family/split, both predicted graphs, graph scores, paired differences, and per-split cluster-bootstrap summaries.

## Results

All metrics below average over the deterministic 120-world suite. Higher F1 is better; lower SHD is better. The baseline supplies the data-generating noise probability, so unknown-noise estimation is not tested.

| Family regime | Worlds | Observational directed F1 | Observational + do directed F1 | Paired F1 delta (do − obs) | Observational SHD | Observational + do SHD |
|---|---:|---:|---:|---:|---:|---:|
| Train / in-distribution | 36 | 0.9071 | 0.9167 | +0.0095 | 0.0556 | 0.0000 |
| Topology OOD | 36 | 0.9857 | 0.9549 | −0.0309 | 0.1111 | 0.1667 |
| Mechanism OOD | 24 | 0.9167 | 0.9167 | 0.0000 | 0.0000 | 0.0000 |
| Combined OOD | 24 | 0.9870 | 1.0000 | +0.0130 | 0.0833 | 0.0000 |
| **All worlds** | **120** | **0.9486** | **0.9448** | **−0.0038** | **0.0667** | **0.0500** |

The global F1 change is slightly negative. Mean SHD is slightly lower with interventions, but 110 of the 120 paired worlds produce the same F1; seven favor the mixed arm and three favor observation-only. The per-family effects vary: interventions helped the combined-OOD subgroup slightly, but were worse on the topology-OOD subgroup under this particular MDL score and sampling design. Several cluster-bootstrap intervals touch or cross zero. There is no basis to claim a general intervention advantage from this run.

The unexpectedly high baseline scores on several families also show a benchmark-design limitation: the topologies are simple and the candidate mechanisms align closely with the generator's finite family catalog, giving a high ceiling. The near-perfect combined-OOD score is not evidence of general scientific transfer; it is one finite deterministic grid of four-node models.

## Interpretation

The useful result is that the pipeline now carries causal ground truth all the way to sample generation, an actual graph-selection decision, graph-recovery scoring, family-stratified reporting, and paired uncertainty summaries. The *scientific finding in this pilot* is mixed/neutral, not a win. This is exactly why equal-budget interventions and held-out family slices need to be mandatory.

## Limitations

- Four-node Boolean SCMs only; exact exhaustive search does not scale to the 6–8 node generator or industrial graph sizes.
- Noise is known to the learner and matched to the generator; there is no unknown-noise fit, latent confounding, omitted variable, nonstationarity, or intervention failure.
- Truth-table MDL is a transparent custom baseline, not PC, GES, NOTEARS, an LLM agent, or a reproduction of the ACDB contestants.
- Family splits here stratify evaluation worlds; there is no learned meta-model trained on the train split.
- Cluster bootstrap intervals resample the finite topology × mechanism cells; this is descriptive uncertainty for this grid, not a population-level guarantee.
- The mixed-graph scorer remains ACDB-inspired, not yet byte/metric-parity checked against the official ACDB scoring and efficiency implementation.
- Local test receipts are distinct from GitHub Actions; skipped workflows are not passes.

## Next gates

1. Add at least one strong conventional baseline (e.g. a contract-matched PC/causal-learn path with documented assumptions) and check CPDAG/SHD/efficiency parity directly against ACDB.
2. Add unknown-noise estimation and deliberately misspecified noise, hidden/omitted variables, unfaithful mechanisms, and intervention failures.
3. Grow the sample-budget curves (e.g. 64/128/256/512/1024 rows), paired on every world; include error/coverage/cost frontiers.
4. Test larger graph sizes using algorithms that scale; preserve exact four-node results as a correctness guard rather than claiming exact enumeration scales.
5. Keep the OOD result segregated by topology, mechanism, and combined shift; do not collapse them into one score or retune based on held-out results.
6. Make no superiority claim until a paired external benchmark adapter and its scoring semantics are independently verified.

## Research links

- [ACDB](https://arxiv.org/abs/2609.31675) — active intervention budget and graph-recovery contract.
- [CausalArena](https://arxiv.org/abs/2609.11897) — broad SCM and mechanism-family variation.
- [Falsifying Causal Graphs With Outlier Events (UAI 2026)](https://proceedings.mlr.press/v337/orchard26a.html) — formal candidate-graph rejection as a stronger comparator than raw posterior confidence.
- [CauScale (ICML 2026)](https://proceedings.mlr.press/v306/peng26c.html) — separate large-graph efficiency lane.

No SOTA, novelty, or general causal-discovery superiority claim is made.