# PIE-009K black-start criticality oracle evidence

Date: 2026-09-12

## Scope

This note records the implementation-independent structural restart-criticality reference in `scripts/pie-009k-black-start-criticality-oracle.py`.

The oracle complements PIE-009J. PIE-009J accounts for island-local startup energy and consumables; PIE-009K asks a different question: which minimal structural seed capabilities are sufficient for restart, and which minimal seed losses make restart impossible?

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

## Frozen semantics

The oracle freezes these rules:

- restart nodes may have alternative dependency recipes;
- reachability grows monotonically when seed capabilities are added;
- unsupported dependency cycles do not self-bootstrap;
- essential targets and productive targets are analyzed separately;
- feasible seed sets are reduced to inclusion-minimal sets;
- cut sets are reduced to inclusion-minimal losses from an explicitly feasible baseline;
- a baseline that does not satisfy a target fails closed instead of returning misleading cut sets;
- duplicate recipes, duplicate dependencies, unknown nodes, and unknown seed candidates fail closed;
- restart waves expose dependency depth without pretending to be physical startup time.

## Executed synthetic fixture

The reference graph contains:

- two alternative habitat black-start seeds (`battery` and `rtg`);
- a separate industrial seed (`ind_seed`);
- control -> habitat generation -> life-support/water dependencies;
- industrial generation -> machine shop -> metrology dependencies;
- a disconnected `cycle_a <-> cycle_b` pair with no seed path.

The self-test proves:

1. either `battery` or `rtg` alone is an inclusion-minimal essential restart seed;
2. `ind_seed` alone is an inclusion-minimal productive restart seed in the synthetic graph;
3. losing both habitat seeds is the inclusion-minimal essential cut;
4. losing `ind_seed` is the inclusion-minimal productive cut;
5. the disconnected cycle remains unreachable;
6. adding seed capabilities cannot reduce the reachable closure;
7. supersets of already-minimal seed sets are not redundantly reported;
8. an infeasible baseline cannot be used for cut-set claims;
9. unknown dependencies fail at validation.

## Important limitations

This is structural reachability analysis only. It does not model startup energy, startup power, batteries, thermal transients, physical switchgear, protection, synchronization, repair time, reliability, common-mode probabilities, or economic cost. PIE-009J remains the independent reference for finite startup energy, island-local accounting, and startup-consumable semantics.

`restart_wave` is a dependency layer count, not seconds/minutes/hours. A later production implementation should compose structural criticality with PIE-009J physical startup accounting and PIE-009E/H/I resilience/authority semantics.

## Promotion boundary

The intended progression is:

`structural seed/cut-set oracle -> production restart-criticality analysis -> composition with PIE-009J energy/islanding -> compound-shock black-start campaign -> evidence-bounded Moon/Mars architecture comparison`.

Tracks #1932, #1924, #1754, #1791, #1849, #1857, #1647 and master #1604.
