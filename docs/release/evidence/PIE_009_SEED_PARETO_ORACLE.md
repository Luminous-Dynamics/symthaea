# PIE-009A synthetic seed-package / Pareto oracle

## Purpose

Freeze implementation-independent reference semantics for evaluating finite industrial seed packages without collapsing industrial planning into one hidden weighted score.

The oracle is `scripts/pie-seed-pareto-oracle.py` and imports no Symthaea code.

## Executed evidence

The final candidate self-test was executed locally on 2026-09-12 and returned `ok` before the checked-in reference was created.

The synthetic fixtures verify:

1. increasing the mass/volume budget cannot remove a previously feasible package;
2. over-budget packages fail closed;
3. removing a critical electronics/control import cannot improve critical or productive closure;
4. an explicit alternate local control route may improve closure without pretending the import still exists;
5. the Pareto frontier preserves multiple nondominated strategies instead of choosing one hidden weighted optimum;
6. a high-bulk-output package can remain weak in productive closure;
7. output-producing hardware without required power produces zero useful output in the synthetic dependency fixture;
8. duplicate/invalid item definitions fail closed;
9. each reported Pareto point is genuinely nondominated under the declared objective directions.

## Objective directions

Minimize:
- Earth seed mass;
- Earth seed volume;
- remaining imported blocker mass;
- commissioning time.

Maximize:
- useful local output;
- critical-capability closure;
- productive-equipment closure;
- resilience.

No weighted scalar score is computed.

## Important limitations

All seed masses, volumes, production rates, closure gains, blocker masses, commissioning times, and dependency caps are synthetic arithmetic fixtures. They are not lunar or Martian engineering estimates.

The oracle performs static subset enumeration rather than the future discrete-time PIE campaign. It does not yet model:
- process inventory over time;
- maintenance/failure histories;
- power dispatch or heat cascades;
- real transport campaigns;
- real Moon/Mars evidence lineages;
- process scale-up;
- financing/economics;
- autonomous plant authority.

## Non-claims

A Pareto point is a planning alternative under the synthetic fixture, not a recommended mission architecture. High bulk output does not imply self-sufficiency, and high closure metrics do not imply plant feasibility or qualification.

Tracks PIE-009 #1641, Phase-0 exit gates #1647, integrated synthetic audit #1700, and master program #1604.
