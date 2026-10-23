# QCOMP-AMORTIZED-COST-001: shared setup and marginal cost

Status: **Rust accounting primitive and fixtures authored; not yet compiled or benchmarked**

## Why it exists

Hybrid QAOA-informed workflows can have large shared setup expenses—building local-expectation tables, collecting circuits/shots for reusable subgraph topologies, calibration, parameter discovery, or compilation—that are then reused over many graph instances. Comparing only per-instance runtime unfairly ignores setup; charging every graph the entire setup also unfairly ignores reuse.

The new Rust primitive in symthaea-quantum-comp compares fixed setup and marginal per-instance cost separately, for a caller-declared cohort size N:

    total_cost(N) = fixed_setup_cost + N * marginal_cost_per_instance

It does this for each raw resource dimension separately, without summing unlike units into a universal scalar. The cost comparison is independent of solution quality: the method with lower cost is not automatically an equivalent or superior solver.

## API and failure behavior

- MisAmortizedCostProfile holds a method ID and one entry for each declared metric.
- Each entry has an optional fixed setup amount and marginal per-instance amount. None means missing/unrecorded and fails closed; Some(0) means explicitly recorded zero.
- Candidate and baseline must declare the same dimensions, no duplicates, and valid unit metadata. Monetary comparisons require matching three-letter uppercase currency codes.
- Raw dimensions include candidate observations, physical shots, circuit batches, compile/transpile time, calibration, preprocessing, classical compute, queue, end-to-end wall time, QPU execution time, energy and money.
- Metrics remain separate in the output and are sorted deterministically. Wall time overlaps stage times and must not be summed with its component timers into one total.
- For a candidate whose marginal cost is no greater than the baseline, the result reports the smallest positive cohort size at which its total is no greater. If marginal cost is higher, no durable break-even is claimed; the candidate may still be cheaper for a small cohort because its fixed setup is lower.

**Wall-clock caveat:** fixed + N × marginal wall time only models end-to-end latency under the declared serial execution regime. If graph jobs run concurrently or share batches, do not infer latency by multiplying a mean per-instance time; measure the actual end-to-end time at each cohort size. Cumulative CPU time, circuit counts and shots can still be additive when their accounting boundary is explicit. Stage times may overlap; do not sum them with wall time.

## Hand-checkable example

Suppose a quantum-informed method has fixed setup of 1,001 units and costs 2 units per graph, while the classical baseline has setup of 1 unit and costs 12 units per graph.

- At N = 99: candidate total = 1,199; baseline total = 1,189, so the candidate is more expensive.
- At N = 100: candidate total = 1,201; baseline total = 1,201, so this is the first cohort size where candidate total cost is no greater.
- At N = 101: candidate total = 1,203; baseline total = 1,213, so candidate is cheaper in that dimension.

The crossover does not show a quality win. Both methods must also be compared on paired graph instances and independently verified outcomes, including failure rates and probability of hitting a preregistered target.

## Connection to random-regular MIS research

The 2026 QAOA-informed greedy study precomputes and reuses local QAOA expectation information for recurring subgraph topologies. Symthaea should record:
1. reusable table/circuit acquisition expense,
2. per-graph marginal inference and classical-reduction expense,
3. exact number of instances actually served by the shared artifact,
4. break-even graph count against the strong classical baseline, and
5. the crossover at multiple workloads or a disclosed cohort size.

Physical shots, candidate observations, circuit batches and successful returned solutions remain separate quantities. For monetary entries use one currency. A claimed fixed setup must state exactly which graphs or service period reuse it; do not silently amortize across work that the artifact did not serve.

## Tests authored

Nine unit tests cover a hand-computed crossover, no-durable-break-even behavior, setup cheaper from the first instance, missing-value fail-closed behavior, currency mismatch, invalid currency metadata, duplicate/mismatched metrics, deterministic output ordering, distinct raw cost dimensions and zero/blank input.

**Validation boundary:** the Rust tests have not been compiled or executed in this authoring environment because cargo/rustc are unavailable. This is accounting infrastructure, not measured cost data, solver execution, a reproduction of a published method, or a quantum-advantage claim.
