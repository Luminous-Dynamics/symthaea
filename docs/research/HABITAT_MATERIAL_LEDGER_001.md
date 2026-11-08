# HABITAT-MATERIAL-LEDGER-001 — quantity-aware genealogy and conservation

**Status:** synthetic Rust accounting qualifier; no physical measurements or live ledger are connected.

## Paradigm shift

Identity-only genealogy answers "which lots may have contributed?" It cannot, by itself, answer "how much was consumed, what remains, what became scrap or rework, and can the inventory actually support the claimed outputs?" This work adds a deterministic quantity ledger where every transaction must conserve a declared material stream, each account has a replayed balance, and an output cannot be produced from stock that was not available.

That distinction matters for recall scope, stock accuracy, yield, circular recovery, product costing, and production replanning. A graph edge is not a mass balance.

## Research grounding

GS1 EPCIS 2.0.1's [TransformationEvent model](https://ref.gs1.org/epcis/TransformationEvent) explicitly represents input and output objects and quantity lists, including partial consumption. The [EPCIS 2.0.1 standard](https://ref.gs1.org/standards/epcis/2.0.1/) describes how quantities participate in transformations. This Rust fixture is a local research contract—not an EPCIS implementation or conformance claim.

## Implemented synthetic ledger

Fixture: `docs/research/fixtures/habitat-material-ledger-v0.1.json`. Rust qualifier: `crates/domains/symthaea-habitat-vertical-bench/src/material_ledger.rs`.

The example traces a single **textile-fiber component-mass stream** through nine transactions:
1. Cut 600,000 mg from a synthetic 1,000,000 mg opening balance. Account for 480,000 mg in cut panels, 20,000 mg in recoverable offcuts and 100,000 mg in scrap. The remaining raw-material stock is 400,000 mg.
2. Transform 480,000 mg of panels into two 215,000 mg cover components, 20,000 mg of scrap and 30,000 mg of rework WIP.
3. Recover the 30,000 mg rework pool into 18,000 mg recovered feedstock and 12,000 mg residue.
4. Move each cover component through final-inspection, packaging and dispatch accounts at unchanged tracked component mass.

The known answer is 1,000,000 mg opening mass and 1,000,000 mg closing mass across eight non-zero closing accounts. Accepted/delivered cover-component mass is 430,000 mg; recoverable material is 38,000 mg; scrap plus residue is 132,000 mg. All of these are constructed integer controls, not physical results.

**Important scope boundary:** the base identity fixture stores the source fabric lot as 12 `piece_equivalent`; the new ledger separately assigns a synthetic 1,000,000 mg opening mass and explicitly blocks conversion between those units. The component mass is not full product mass, and the ledger does not validate an actual bill of materials, scale reading, supplier quantity, or lifecycle assessment. Source event IDs anchor process context only and do not attest the ledger quantities.

## Fail-closed rules

- Every account and transaction ID is unique; transaction sequences are unique and sorted deterministically.
- All line quantities are positive integers in mg. Mixed units, unknown accounts/source events, unknown traceability nodes and mismatched source-operation IDs fail closed.
- Inputs must be available in the account ledger; no negative inventory, double allocation or consumption of future stock is accepted.
- For each transaction, total input component mass must equal all output, offcut, scrap, rework, recovery and residue accounts exactly. Integer overflow is an error, not a wraparound.
- After each valid transaction, global tracked mass is conserved; final account balances must exactly match a frozen known answer.
- Invalid/missing quantities are never imputed to zero; unit conversion remains blocked unless separately supplied and qualified.

## Proposed benchmarks (not measured)

1. **Conservation:** 100% exact integer balance for every admitted transaction and every tracked stream.
2. **Inventory feasibility:** zero overdrawn accounts, negative balances, duplicate allocations, or outputs from unavailable stock.
3. **Genealogy quantities:** exact quantity attribution for all declared input/output lots and component branches; unknown shares remain unknown, not proportionally guessed.
4. **Waste and recovery:** 100% disposition accounting for the chosen boundary—accepted component, reusable offcut, scrap, rework, recovered feedstock, residue and remaining inventory.
5. **Mutation detection:** reject 100% of seeded one-milligram imbalances, overdraws, mixed-unit lines, unknown accounts and false known-answer balances.
6. **Scale:** after profiling the qualified implementation, replay 10,000 transactions with checked integer arithmetic and report p50/p95 time, peak memory and deterministic replay hashes.
7. **Independent audit:** independently recompute the ledger and every closing balance from raw lines; do not trust the ledger module to certify itself.

No cost savings, mass yield, process efficiency, or environmental benefits should be inferred from this synthetic example. Before real use, add signed source measurements, calibrated unit conversions, scale uncertainty, rework genealogy, scrap custody, and a jointly governed adapter to Mycelix's manufacturing-evidence and process-registry owners.

## Next integration gate

Bind stable physical batch/component IDs and explicit measurement evidence to the event-replay stream; add a calibrated conversion record where count and mass units cross; then compare the ledger against independent weighing and inventory counts under a preregistered sampling protocol. Until then, this is only a synthetic correctness challenge.
