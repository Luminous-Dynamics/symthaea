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
