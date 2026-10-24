# QOBLIB-MIS-PREFLIGHT-001: exploratory greedy-policy quality preflight

Status: **exploratory data-analysis artifact; not a Rust run or competitive solver benchmark**

## Question

Before implementing a more expensive portfolio, do cheap deterministic and seeded-randomized MIS policies provide a useful negative/control baseline, and how far can they get toward existing public classical results?

## Source and method

All eight DIMACS inputs and the ParallelILS result table are pinned to QOBLIB commit d61f234b9fc816deb3d4e1dea4c031595b35acd7. The JSON companion records each source graph's Git blob object ID and every policy's result.

An in-tool JavaScript reference implementation (not committed as production code) checked the DIMACS header and edge count, endpoint ranges, self-loops and duplicate undirected edges. It evaluated 18 policies per graph:
- deterministic minimum residual degree, ties by lowest vertex ID;
- deterministic maximum residual degree, ties by lowest vertex ID;
- 16 seeded randomized selections among vertices tied at minimum residual degree;
- each initial solution was improved by repeated feasible 1-for-2 exchanges, replacing one selected vertex with two unselected vertices that are nonadjacent to each other and to every remaining selected vertex.

The exact seeds, per-policy objectives, feasibility verdicts, exchange counts, graph source IDs, and public ParallelILS objective/success metadata are in QOBLIB-MIS-PREFLIGHT-001.json. No timing for this transient analysis was collected.

## Results

All candidate sets across all 8 graphs and 18 policies each were feasible under an independent adjacency check. The best of these simple policies matched the pinned public ParallelILS objective on 7 of 8 selected instances; on C125-9, the best result was 33 versus published objective 34. On farm (17 vertices / 39 edges), exhaustive subset enumeration independently establishes optimum 10, matching the published best-known objective. These matches are not a reproduced ParallelILS run and do not imply that the target objectives are proven optimal on the other seven instances.

| Instance | Vertices | Edges | Published ParallelILS objective | Best of 18 simple policies | Worst of 18 policies |
|---|---:|---:|---:|---:|---:|
| farm | 17 | 39 | 10 | 10 (exact optimum) | 5 |
| chesapeake | 39 | 170 | 17 | 17 | 16 |
| karate | 34 | 78 | 20 | 20 | 20 |
| C125-9 | 125 | 787 | 34 | 33 | 31 |
| ibm32 | 32 | 90 | 13 | 13 | 13 |
| brock200-1 | 200 | 14,834 | 6 | 6 | 5 |
| c-fat200-1 | 200 | 1,534 | 18 | 18 | 18 |
| es60fst03 | 113 | 142 | 55 | 55 | 50 |

These are objective-quality checks, not runtime comparisons. The QOBLIB submission reports its own machine, implementation and runtime conventions; this preflight intentionally records no comparable timing. The policies are simple controls, not a competitive solver and not an equivalent reproduction of ParallelILS or a quantum-informed method.

## Research implications

1. **Useful easy-instance controls:** several heterogeneous small/medium instances are already solved to the published target by a weak greedy portfolio. Any quantum-informer evaluation must include them rather than curate only cases that favor the proposed method.
2. **A real hard-case signal:** C125-9 remains one below the published target after 18 cheap policy variants plus 1-for-2 local improvement. This is a useful place to compare stronger classical portfolios and information-source ablations, but it is only one selected case—not evidence of systematic hardness.
3. **Portfolio diversity matters:** on farm, max-residual-degree greedy returns 5 while min-residual-degree returns the exact optimum 10. On chesapeake and C125-9, tie-breaking and local search alter the result. This motivates measuring marginal decision value and policy complementarity, not just average single-policy accuracy.
4. **Next experiment:** port the smoke baseline to executable Rust, then evaluate the exact same seeded arms across a prespecified slice of the 50-instance cohort. Only after reproducibility is established should a learned selector or quantum-informed policy be compared, at both matched decision budget and matched all-in cost.

## Limitations and claim boundaries

- Eight purposively selected instances are not a random or preregistered confirmatory sample.
- The 18 policies are cheap controls, not a state-of-the-art portfolio.
- QOBLIB's ParallelILS objectives are published records, not independently replayed in this work; best-known is not synonymous with proven optimum.
- Only farm is certified optimal by this preflight's independent exhaustive enumeration.
- This analysis was not performed by the new Rust parser or Rust smoke runner, which remain uncompiled/unexecuted in the authoring environment.
- No QPU or simulator was executed, and no quantum advantage or SOTA improvement is claimed.
