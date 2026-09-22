# LQCD-021C — β=6.0 pre-production pilot protocol

Independent standard-library protocol subject separating throughput, scientific equilibration/slow-mode diagnostics, operator-overlap selection, and final production authority for the future EHK β=6.0 reproduction.

Exact executed subject SHA-256:

`0481ec58bb6ca118a3bb6bfd14e848c8d9072df0ecbbb8f153e3226b6c0f2783`

Canonical protocol/result SHA-256:

`5147c53e2f97a4a773006848cd5c9cdf40be84593f090ece1433c485374dcff4`

## Throughput lane

Binds integration PR #2531 / revision `d7569beecf8bb5c775939d28dba9b8e09ccb2e7b` as performance/feasibility only. Throughput may determine wall-time/resource budgets and whether a pilot is executable on a given environment. It is explicitly forbidden from selecting burn-in, stride, plateau windows, static-potential fit ranges, physics thresholds, or benchmark estimates.

## Equilibration / slow-mode lane

Target geometry is β=6.0 on `16^3×32` with four distinct start classes:

- cold identity, zero center phase;
- cold identity, positive `Z3` phase;
- cold identity, negative `Z3` phase;
- deterministic stress field v1.

Every chain requires a disjoint transition stream. Required diagnostic coverage includes plaquette, center-invariant/aligned Polyakov observables, categorical `Z3` mobility, flowed `Q`, and flowed `Q²`.

Frozen diagnostic revisions:

- chain statistics: `a3d631e56ccf052cbd1a9bf33d481e711378cf35`;
- center symmetry: `5e275c1d2e5e03ee27f592db17fc1a7e84692489`;
- topology diagnostics: `735b69f24b7cf6e9f7e56c9caace13e697bc1390`;
- RK3 flow: `93017e5207f7b0d52f3387fb1211f62fe10e804a`.

The diagnostic/qualification policy and machine-feasible pilot execution budget must both be frozen **before the first pilot cycle**. The pilot may produce production burn-in/stride candidates and block/equilibrium evidence, but may not produce final string tension, Sommer scales, or benchmark pass/fail claims.

## Operator-overlap lane

Binds the exact LQCD-021A measurement design result `df92cdfbbf882a8f3c37eb911476eb961bbc7790d6d6adae60bccc1f8c999c72`. Pilot-only equilibrated configurations may be used to freeze primary/diagnostic `V_eff` windows and static-potential `r` ranges. Those choices must be frozen before final production and cannot be modified using production data.

## Final benchmark firewall

The benchmark target is 4000 retained final configurations. After the pilot, the final campaign either freezes a 4000-configuration production plan or records that the benchmark reproduction is infeasible under the qualified design/environment; it must not quietly reduce the target.

All pilot configurations are permanently excluded from the final sample. Final seeds, burn-in, stride, windows, ranges, and policies are frozen before production authorization. Production data cannot relax or reselect them.

## Scientific boundary

This subject freezes pilot chronology and authority only. It does not choose pilot cycle counts, run any Markov chain, establish equilibration or topology mixing, select a physical plateau/range, or reproduce the external benchmark. A separate pilot execution manifest must bind actual resource budgets and thresholds after LQCD-021B throughput evidence and before any scientific pilot cycle.
