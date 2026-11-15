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


## Manufacturing KPI calculators

`manufacturing_metrics` contains dependency-free integer/fixed-point calculations for FactoryBench:
- ratio in parts per million (ppm), retaining numerator and denominator;
- quality-adjusted line efficiency;
- first-pass yield, OTIF, material yield and traceability coverage;
- OEE components and exact combined OEE for one named work unit/asset;
- defects per 100 inspected units;
- nearest-rank percentiles for P50/P95 and other declared quantiles.

Calculators reject zero denominators, impossible counts, inconsistent OEE time boundaries, performance above 100%, invalid percentiles and arithmetic overflow. They do not substitute zero for missing measurements. OEE is calculated from exact inputs to avoid compounding truncation in rounded component ratios.

These are arithmetic functions only. They do not validate SAM/SMV methods, source records, calibration, inspection truth, or production capacity.

Run focused tests with:

```bash
cargo test -p symthaea-operations-research manufacturing_metrics
```

The tests are authored but not run in the current environment because the Rust toolchain is unavailable.
