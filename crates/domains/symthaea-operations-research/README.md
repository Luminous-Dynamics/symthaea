# symthaea-operations-research

Operations research for Symthaea — a practical decision-science layer: inventory
optimization, queueing, and shortest paths.

Pure `std`, zero deps, no `symthaea-core` link. Closed-form + exact algorithms,
checked vs known values.

- `inventory::economic_order_quantity` (EOQ) + total cost.
- `queue::MM1` — ρ, L, Lq, W, Wq (Little's law).
- `graph::dijkstra` — single-source shortest paths.
- `factory::plan_factory_line` — deterministic skill/capability-constrained line scheduling with precedence, availability windows and a critical-path lower bound.
- `factory::verify_factory_schedule` — independent validation of assignments, precedence, non-overlap and reported schedule metrics.

```bash
cargo test -p symthaea-operations-research
```
