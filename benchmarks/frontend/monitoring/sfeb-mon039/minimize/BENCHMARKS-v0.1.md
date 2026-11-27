# MON-039 Benchmark Plan v0.1

## Hypothesis

Exact-signature counterfactual reduction can lower the number of recovery fault replays needed to isolate a minimal reproducer without replacing independent verification or conflating "same verdict" with "same failure".

## Prospective targets (not achieved)

| Dimension | Target |
|---|---:|
| Exact failure preservation | 100% selected subsets match both target verdict and signature |
| Small search | 100% global candidate minimum for complete subset tables up to 12 faults |
| Large search | 100% 1-minimality certificate for fully observed single-deletion search |
| Missing rows | 0 missing/incomplete subset observations treated as pass |
| Boundary/revision/trial drift | 100% rejected before minimization |
| Receipt integrity | 100% observation field mutations rejected unless the receipt is recomputed |
| Determinism | byte-identical result for same sorted replay table and code revision |
| Search efficiency | p95 < 100 ms for 64 candidate faults and complete 1-deletion table on a pinned 2-vCPU host |
| Evidence | 100% of selected steps link to a recomputed outcome-receipt digest |

A deterministic synthetic table is a regression oracle, not a live fault trial. The minimizer does not convert 10,000 software mutations into independent reliability statistics.

## Baselines

Compare:
1. no reduction (report the entire fault schedule);
2. deterministic greedy single-fault deletion;
3. conventional ddmin with its assumptions explicitly stated;
4. exhaustive cardinality search for <=12 faults;
5. MON-039's exact-signature, trial/boundary/revision-pinned reduction;
6. independent SRE/maintainer adjudication.

Hold the scenario, recovery boundary, trial, code revision, fault scheduler, seed, oracle and outcome signature constant. Report number of replays, wall time, receipt count, minimal-set cardinality, minimality guarantee, incomplete-evidence rate and disagreement with independent adjudication.

## Research extensions

- pairwise and interaction-aware search for non-monotone faults;
- hierarchical reduction by fault families, then exact fault IDs;
- flakiness estimation with repeated replays before allowing a failure signature into the table;
- confidence-aware causal graphs that distinguish a minimal reproducer from a unique root cause.

All live comparisons are pending.
