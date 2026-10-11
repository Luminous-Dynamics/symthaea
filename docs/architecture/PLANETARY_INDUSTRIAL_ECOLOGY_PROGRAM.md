# Planetary Industrial Ecology (PIE) Program

Status: Phase-0 program definition.

## Purpose

PIE models how an off-Earth settlement converts local resources, imported seed equipment, energy, utilities, recycled wastes, and maintenance capacity into useful products and replacement productive capacity.

PIE does **not** assume self-sufficiency, a preferred ISRU process, or that the Moon and Mars share the same industrial optimum. It exists to make resource/process dependencies explicit and evidence-bearing so competing industrial architectures can be compared.

## Architectural boundary

PIE complements rather than replaces existing systems:

- `symthaea-materials`: engineering material properties, design/search, aging, mining cognition;
- `symthaea-fabrication-kernel`: manufacturing and design-to-production twins;
- Mycelix Fabrication: designs, machines, jobs, material links and verification;
- Mycelix Supply Chain: provenance, inventory, logistics and settlement of flows;
- LETN/lunar transport: movement of resources, equipment and products;
- energy systems: electricity, storage and generation;
- future thermal network: process heat and waste-heat reuse.

PIE owns the **industrial process graph and closure accounting** between those systems.

## Program tranches

1. **PIE-000 — neutral ontology**: resource, material lot/grade, process, inputs/outputs, utilities, equipment, waste/recycle and evidence.
2. **PIE-001 — conservation**: mass and elemental-balance validation with explicit tolerances.
3. **PIE-002 — utilities**: energy, peak power, heat, process time, storage and duty-cycle accounting.
4. **PIE-003 — material quality**: composition, impurities, physical form, grade transitions and purification requirements.
5. **PIE-004 — lunar registry**: evidence-bearing lunar resources and candidate process chains.
6. **PIE-005 — Mars registry**: evidence-bearing atmospheric, water, regolith and biological process chains.
7. **PIE-006 — circularity**: recycling, waste streams, by-product exchange and heat cascading.
8. **PIE-007 — equipment dependencies**: machine tools, maintenance, consumables, catalysts, spares and replacement trees.
9. **PIE-008 — closure analysis**: mass/energy/chemical/spares/recycling/critical-path closure and import leverage.
10. **PIE-009 — industrial seed optimization**: compare seed equipment sets under fixed Earth-import mass/cost/power constraints.
11. **PIE-010 — fabrication/supply-chain bridges**: integrate with existing Mycelix and Symthaea manufacturing systems without duplicating them.
12. **PIE-011 — energy/logistics integration**: couple production chains to power, thermal and LETN transport networks.
13. **PIE-012 — growth campaigns**: multi-year simulations of expansion, failure, maintenance and increasing local reproductive capacity.

## Core invariants

- A resource occurrence is not an available feedstock until acquisition/beneficiation losses are modeled.
- A material name is insufficient; grade, composition, physical form and evidence matter.
- Local mass fraction is not equivalent to independence; critical imported items remain visible.
- Recycling competes with mining and import on the same accounting basis.
- Waste and by-products never disappear; they terminate in inventory, a sink, a recycle edge or explicit unknown.
- Simulation/literature/vendor evidence cannot become qualification evidence automatically.
- Missing process/equipment dependencies fail closed in closure analysis rather than being assumed locally available.
- Optimization never grants plant/control authority.

## Phase-0 decision questions

PIE should eventually answer, with explicit uncertainty:

- Which local processes unlock the largest number of downstream capabilities?
- Which imported kilograms have the highest industrial leverage?
- Which Moon/Mars process chains are Pareto-efficient under power, heat, logistics and maintenance limits?
- At what scale does recycling beat fresh extraction for each material family?
- Which critical dependencies prevent a settlement from reproducing its own productive equipment?
- What seed industrial package maximizes useful local productive capacity over a declared horizon?

## Non-claims

PIE-000 contains ontology only. It does not establish lunar or Martian resource abundance, process yields, industrial feasibility, self-sufficiency, economic viability, or safety qualification.
