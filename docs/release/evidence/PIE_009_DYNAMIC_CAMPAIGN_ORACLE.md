# PIE-009D Dynamic Bootstrap Campaign Oracle

## Scope

Implementation-independent standard-library Python reference for temporal industrial-bootstrap bookkeeping. All values are synthetic fixtures. This is not a Moon/Mars production forecast, reliability prediction, economic optimizer, chemistry solver, qualification result, or hardware-control layer.

## Executed evidence

The candidate `scripts/pie-dynamic-campaign-oracle.py` self-test was executed locally on 2026-09-12 and returned `ok` before commit.

## Frozen semantics

- Maintenance occurs before production and consumes real spare inventory.
- When spares are scarce, maintenance allocation is explicit through declared maintenance targets; machine iteration order cannot silently choose winners.
- Production and construction consume one single-spend opening inventory ledger.
- Outputs created during step N are pending until the end-of-step boundary and cannot bootstrap another action in the same step.
- Locally built machines and locally built power capacity increase capability only from the following step onward.
- Scheduled imports arrive only at the opening of their declared step.
- Process capacity, builder capacity, material inventory, maintenance inventory, and power all fail closed if exceeded.
- An imported machine may remain operational even when missing imported controllers or other blockers prevent local reproduction.
- Cumulative imported material, locally built machines, and cumulative process outputs remain distinct.

## Synthetic acceptance fixtures

The executed self-test checks:

1. same-step production output cannot be consumed to build a machine;
2. a locally built refinery increases throughput only in the next step;
3. insufficient maintenance inventory fails the all-machines maintenance plan;
4. an explicit reduced maintenance target can preserve a selected machine without arbitrary hidden prioritization;
5. a missing controller prevents local refinery construction while an existing refinery can still operate;
6. newly built power capacity cannot support the build step that creates it, but is available next step;
7. two builds cannot spend the same controller or material inventory twice;
8. an import can support a build only when it arrives at that step boundary;
9. process-capacity and power violations fail closed;
10. a two-step campaign demonstrates delayed local machine growth and increased future throughput;
11. malformed machine/process cross-references fail during model construction.

## Important limitations

- Deterministic bookkeeping only; no stochastic failure model.
- One campaign step has abstract duration; no claim that it represents a day, month, or year.
- Builder throughput is one unit per operational builder per step in the synthetic fixture only.
- No process chemistry, composition, grade, recycle, transport, site, economics, or real equipment parameters are inferred here.
- The reference does not optimize plans; it validates declared plans against temporal resource constraints.

## Integration intent

This oracle is intended to compose later with:

- PIE-009A seed-package Pareto selection;
- PIE-009B frozen evidence lineages;
- PIE-009C decision sensitivity / experiment priority;
- PIE-P0 resource access, throughput, lifecycle, and integrated-chain oracles;
- production PIE implementations only after their Rust execution evidence clears.

Tracks #1752, #1641, #1647, #1700 and master #1604.
