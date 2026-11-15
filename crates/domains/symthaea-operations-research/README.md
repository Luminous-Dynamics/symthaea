# symthaea-operations-research

Operations research for Symthaea — a practical decision-science layer: inventory
optimization, queueing, shortest paths and facility-flow evaluation.

Pure std, zero dependencies, no symthaea-core link.

- `inventory::economic_order_quantity` — EOQ and total cost.
- `queue::MM1` — M/M/1 queue metrics.
- `graph::dijkstra` — floating-point shortest paths for existing callers.
- `graph::dijkstra_u64` — exact integer shortest-path trees with predecessor-edge receipts.
- `facility_layout::evaluate_layout` — deterministic declared-zone, adjacency, typed-flow and path-provenance evaluation.

The facility layout evaluator consumes caller-supplied zones, adjacency declarations and
directed travel edges with integer millimetre distances. It does not infer geometry,
approve protected egress, qualify machinery, schedule production, or certify a facility.
Distance edges require an explicit source reference and retain a basis
(`Synthetic`, `Measured`, or `Imported`). Return flows cannot traverse
new-material-only zones; new-material and WIP flows cannot enter returns-only
zones; protected egress edges are restricted to people movement.

For the full evaluator boundary and Rust tests, see
`src/facility_layout.rs`.

Run focused tests with:

```bash
cargo test -p symthaea-operations-research
```
