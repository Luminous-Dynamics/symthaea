# PIE Phase-0 resource-access oracle

## Purpose

Freeze implementation-independent semantics for converting an in-place resource occurrence into recoverable material and then into feedstock actually deliverable to a downstream process.

The oracle is `scripts/pie-resource-access-oracle.py` and imports no Symthaea code.

## Semantics

- in-place occurrence mass is not inventory;
- acquisition/recovery efficiency is a separate process outcome;
- a resource at another site requires an explicit available transport edge;
- transport capacity, delivery loss and energy remain explicit;
- same-site use does not invent transport cost;
- conservative feasibility is `Guaranteed`, `Possible`, or `Impossible`;
- widening uncertainty cannot strengthen a conservative result;
- malformed fractions, invalid routes, non-finite values and missing remote transport fail closed.

## Executed synthetic fixtures

The final candidate self-test was executed locally on 2026-09-11 and returned `ok` before the checked-in reference was created.

Fixtures cover:

1. 100..120 kg in-place resource with 50..60% acquisition produces only 50..72 kg recoverable material;
2. same-site material needs no transport edge;
3. remote material without a transport edge is unavailable;
4. bounded transport capacity and delivery efficiency reduce delivered feedstock to 36..68.4 kg in the synthetic case;
5. a 30 kg demand is guaranteed, 50/60 kg demands are only possible, and 70 kg is impossible under those bounds;
6. widening acquisition/transport uncertainty weakens guaranteed feasibility to possible;
7. unavailable transport removes remote feedstock;
8. mismatched route and invalid recovery fraction fail closed.

## Important limitations

This oracle intentionally does not model real lunar/Martian abundance, excavation physics, terrain, route scheduling, storage, transport technology, economics, degradation, multiple windows, competing consumers, or time-varying resource depletion. `energy_j_per_shipped_kg` and all synthetic numbers are arithmetic fixtures only.

A later production layer should compose this contract with provenance-bound site/resource data, LETN transport edges, inventories, PIE-002 utilities, and process throughput/lifecycle semantics.

## Non-claims

This tranche does not establish that any specific lunar or Martian resource is accessible, recoverable, economic, or sufficient for a settlement.

Tracks #1648, #1647, and master #1604.
