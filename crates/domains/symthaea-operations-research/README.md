# symthaea-operations-research

Operations research for Symthaea — a practical decision-science layer: inventory
optimization, queueing, and shortest paths.

Pure `std`, zero deps, no `symthaea-core` link. Closed-form + exact algorithms,
checked vs known values.

- `inventory::economic_order_quantity` (EOQ) + total cost.
- `queue::MM1` — ρ, L, Lq, W, Wq (Little's law).
- `graph::dijkstra` — single-source shortest paths.

```bash
cargo test -p symthaea-operations-research
```


## Soft-first modular reconfiguration

`modular_reconfiguration::evaluate_reconfiguration` validates a target cell configuration and derives a deterministic transition from the current configuration.

It separates:
- `NoChange`: exact physical and assignment state is unchanged;
- `AllocationOnly`: operations are reassigned while module placement, connections and layout remain fixed;
- `PhysicalReconfiguration`: modules are added, removed, relocated, reconnected or the layout context changes.

The evaluator checks site-scoped qualified module profiles and validity intervals, independent qualification evidence, exact operation capability/process/digest/inspection bindings, typed interface/protocol/media/unit/direction compatibility, port capacity and connection limits, coordinate-frame equality, orthogonal module rotation, footprints, keepouts and layout evidence.

A physical transition is never production-ready. It returns `FeasibleForQualification` at most and emits explicit obligations for layout revalidation, interface inspection, utility/service checks, process requalification, workforce/skill reconfirmation, inspection-plan reconfirmation and independent commissioning review. `production_authorized` is always false.

The geometry kernel checks declared rectangles and keepouts only. It does not certify building/fire rules, egress, machine guarding, electrical installation or worker safety. Port connections are typed configuration declarations, not machine-control commands.

The initial deterministic ReconfigBench fixtures live in `research/atelier/factorybench/reconfigbench-cases.v0.1.0.json`. Rust tests are authored but have not been run in an environment with a Rust toolchain.
