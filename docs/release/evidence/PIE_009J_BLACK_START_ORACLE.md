# PIE-009J black-start / islanding oracle evidence

Date: 2026-09-12

## Scope

This note records the independent synthetic reference semantics in `scripts/pie-009j-black-start-oracle.py`.

The oracle models industrial black-start ordering after a total or partial shutdown. It is intentionally small, deterministic, and implementation-independent. It does not import Symthaea production code.

## Independent execution

The final candidate self-test was executed locally with Python 3 on 2026-09-12 and returned:

`ok`

## Frozen reference semantics

The oracle freezes these rules:

- startup energy is finite and tracked per island;
- opening operational equipment must be sustained before startup actions are credited;
- a component started in step `N` becomes operational only at the step boundary;
- newly started generation cannot bootstrap another component in the same step;
- `requires_operational` dependencies must be opening-operational;
- equipment that requires a live bus cannot start on a dark island;
- startup consumables are explicit and single-spend;
- one island's stored energy cannot silently start another island's equipment;
- circular dependency pairs remain unavailable without an independently started seed;
- essential-service restoration and productive-core restoration are separate campaign milestones;
- malformed references and insufficient energy/consumables fail closed.

## Executed synthetic fixtures

The self-test demonstrates:

1. stored island energy starts control infrastructure;
2. control infrastructure permits a local generator start;
3. life support and water become available only on a later boundary;
4. machine-shop/metrology productive closure occurs later than essential restoration;
5. a generator and dependent load cannot bootstrap one another in the same step;
6. an island with zero startup energy cannot conjure a generator start;
7. a circular `A -> B -> A` startup dependency does not self-start;
8. one startup consumable cannot be spent twice;
9. energy on another island cannot be borrowed implicitly;
10. malformed dependency references fail at model validation.

## Reference outputs

The campaign result exposes at least:

- per-step opening operational components;
- components started at the current boundary;
- per-island energy remaining;
- delivered service quantities;
- whether essential service is restored;
- whether the declared productive core is restored;
- first essential-service restoration step;
- first productive-core restoration step;
- final operational set;
- final startup-consumable inventory.

## Important limitations

This is not an AC/DC transient model, electrical protection model, inverter-control study, thermal-startup model, battery degradation model, relay model, power-quality model, hardware qualification result, lunar/Mars grid design, economic analysis, or mission-operations plan.

`generation_per_step`, `load_per_step`, and `startup_energy` are synthetic scalar bookkeeping quantities. A later production model should compose dedicated PIE utility/thermal accounting and any future electrical-network model rather than enlarging this oracle into a physical power-system simulator.

The oracle also does not yet model explicit interties, synchronization, partial-voltage states, startup-ramp curves, degraded service, or repairs during the black-start sequence. Those belong in later composition with PIE-009E/H/I and any dedicated power-system domain.

## Promotion boundary

The intended progression is:

`independent black-start oracle -> production black-start semantics -> cross-check with PIE utility/power accounting -> integrated shock recovery -> evidence-bounded Moon/Mars islanding campaigns`.

A later real campaign must preserve evidence roots for startup power, storage, equipment requirements, operating loads, and failure/repair assumptions.

Tracks #1924, #1754, #1791, #1849, #1857, #1647 and master #1604.
