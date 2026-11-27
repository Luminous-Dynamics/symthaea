# MON-040 Benchmark Plan v0.1

## Hypothesis

A preregistered repeated-replay gate can distinguish one-off fault observations from reproducible recovery failure signatures and avoid overconfident reliability claims from a single run.

## Prospective targets (not achieved)

| Dimension | Target |
|---|---:|
| Receipt verification | 100% of altered replay rows rejected |
| Sample completeness | 0 missing/incomplete runs silently removed from denominator |
| Scope identity | 100% of trial/scenario/boundary/code/fault-set drift rejected |
| Signature identity | 100% different blocked signatures counted as non-matches |
| Statistical calibration | exact Clopper–Pearson known answers reproduced to 1e-6 |
| 29/29 confirmation | one-sided 95% lower bound >= 0.90 |
| 28/29 control | one-sided 95% lower bound < 0.90 |
| Runtime overhead | p95 < 5 ms for 200 replay rows on a pinned 2-vCPU host |

These are software and protocol targets, not live recovery measurements. The synthetic 29/29 fixture is not evidence about production failure rates.

## Baselines and study design

Compare:
1. single replay (no repeatability estimate);
2. fixed K repeated replay with raw observed proportion only;
3. fixed K with Wilson interval;
4. MON-040 exact one-sided Clopper–Pearson lower bound;
5. repeated replay clustered by execution host/failure domain;
6. independent SRE/maintainer adjudication of signature equivalence.

Use matched fault schedules, exact database/connector/broker versions, code revision, backup snapshot, recovery boundary, executor class, fault injection seed and oracle. Preregister run count, target signature, reliability threshold and confidence before observing results.

Report exact signature match counts, other-blocked signatures, qualified outcomes, indeterminate outcomes, missing runs, seed/replay identity collisions, confidence bounds, executor/environment domains, wall time and compute cost.

## Research extensions

- stratified/clustered intervals for shared hosts and common-cause domains;
- sequential probability ratio tests with preregistered stopping bounds;
- multi-signature discovery with a holdout confirmation set;
- repeated replay before MON-039 minimization, then independent confirmation of the minimized fault set;
- estimate flakiness separately from causal contribution.

All live comparisons and empirical reliability claims remain pending.
