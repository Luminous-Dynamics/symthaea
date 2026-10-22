# HEAVY-OIL-RESCUE-001 — V0 synthetic economics/admissibility kernel

**Status:** research prototype. **Evidence tier:** synthetic software fixtures only.

This is the first vertical slice of [issue #8408](https://github.com/Luminous-Dynamics/symthaea/issues/8408). It tests a narrow contract: do mass-balance, interval-economics, hard-gate and evidence-tier rules behave as specified? It does **not** model petroleum chemistry or demonstrate a process that works.

All fixture quantities, prices and costs are fabricated solely to exercise arithmetic and fail-closed behavior. They are not representative of any crude, refinery, reaction, vendor quote, market or project. A positive synthetic netback must never be cited as physical or commercial evidence.

## Run offline

No third-party Rust dependencies are used.

From a checkout of this repository:

```sh
cargo test --manifest-path research/heavy-oil-rescue/Cargo.toml --offline
cargo run --manifest-path research/heavy-oil-rescue/Cargo.toml --offline
```

The runner emits deterministic CSV rows for:
- a synthetic case whose interval netback clears a zero hurdle;
- a synthetic case that remains below the hurdle even at the optimistic bound;
- an input/output interval-overlap case that must remain uncertain;
- a point-mass imbalance that must block a positive economic screen;
- a case with unknown environmental evidence that cannot be promoted.

Every row carries an FNV-1a 64-bit regression checksum of its canonical CSV record. This checksum is **not cryptographic**, not a signature, and not provenance attestation; it exists only to detect accidental output drift. Mycelix / an approved content-digest mechanism should own later evidence receipts.

## V0 calculation contract

### Explicit basis and units

- Feed basis: kilograms of feedstock in the declared case.
- Boundary mass flows: kilograms per case basis, including feedstock, hydrogen/consumables, products, by-products, water, waste and residual streams.
- Product prices: USD per metric tonne of the named output.
- Costs and netback: USD per metric tonne of feedstock.
- Market revenue is credited only for output streams with an explicit price interval. No price means no revenue credit; handling/disposal costs must appear in the cost ledger.
- Intervals are conservative bounds, **not probability distributions**. Endpoint arithmetic may be pessimistic when variables are correlated. Do not silently interpret interval width as confidence or assign a probability without supporting data.

### Mass-balance statuses

- `ExactBalanced`: both total input and output are point values and close within the declared absolute + relative tolerance.
- `ExactUnbalanced`: point-valued totals differ beyond tolerance.
- `PossibleWithUncertainty`: uncertain total intervals overlap; that means closure is possible, not demonstrated.
- `ImpossibleWithinBounds`: intervals do not overlap within tolerance.

A wide overlapping interval can never upgrade the status to `ExactBalanced`. The model also will not infer elemental/species closure, thermodynamic feasibility, kinetics, product quality, operability, safety, lifecycle impact or economics from mass closure alone.

### Economic status versus route admissibility

The evaluator deliberately emits separate dimensions:

- `EconomicsStatus`: conservative netback above hurdle, optimistic netback at/below hurdle, or indeterminate.
- `Admissibility`: eligible only if mass balance is point-closed and every required energy, product-quality, environmental, process-safety and model-applicability gate is explicitly Pass. Failed gates block; unknown gates remain indeterminate.
- `Conclusion`: even an eligible, positive-margin case is only `SyntheticOnly`, `LiteratureScreenOnly`, or `BenchMeasuredNeedsIndependentReplication` until its evidence tier supports a stronger bounded claim.

A passed gate in a synthetic fixture checks code-path behavior only. It does not assert that a real process passes that gate.

## Tests / hostile controls

The unit corpus checks:
- hand-computable input and output mass totals;
- interval revenue, cost and lower/upper netback calculations;
- exact mass imbalance and overlap-only uncertainty;
- negative mass and non-finite/invalid costs;
- invalid balance tolerances;
- unknown environmental evidence;
- order-invariant calculations and repeatable receipt output;
- a synthetic profitable result never being labelled as independently validated or commercial.

This is a small software oracle, not the independent chemistry/process-model validator needed for later tiers. Additional controls must be added before any real-data integration, including missing/duplicated output streams, unit mismatches, hydrogen/utility flows hidden outside the process boundary, double-counted by-products, and infeasible process-model domains.

## Industry comparison plan

Do not use a single commercial simulator or one optimization score as ground truth.

1. Freeze source-versioned, openly accessible/licensed feedstock assay and process data; preserve assay method, conditions, units, uncertainty and provenance. Do not replace missing values with plausible invented values.
2. Compare each feedstock only with applicable established incumbent routes—e.g. blending, partial upgrading, coking or hydroprocessing where the feedstock and available evidence support it—plus a no-investment/retain/retire baseline.
3. Use a qualified flowsheet simulator as a reference implementation within its tested model scope. [DWSIM's thermodynamics documentation](https://dwsim.org/help/main/38-introduction.html) describes multicomponent petroleum/petrochemical systems, but model applicability and petroleum pseudo-component characterization must be checked for each case. Agreement between simulators is not empirical validation.
4. Compare decision and experiment selection with a fixed/rule-based baseline, a strong conventional optimizer/search method, matched-budget random selection and Symthaea with relevant components ablated. Match available data, simulator calls, retries and information budgets; hold out whole feedstock families or experimental campaigns, not adjacent rows.
5. Carry uncertainty through feed quality, yield, saleable specification, utility prices, catalyst deactivation/lifetime, downtime, throughput, capex/opex, product prices, waste handling and lifecycle emissions/water. Show each metric and the Pareto frontier separately; never hide a hard safety or environmental limit in a weighted average.
6. Publish all negative results and failure cases. Only claim a research advantage after a preregistered metric improves over the strongest applicable baseline on held-out evidence, with confidence/uncertainty intervals and successful independent verification.

Relevant scientific constraints include coke formation, metal deposition and sintering as recognized heavy-oil hydroprocessing catalyst-deactivation mechanisms ([review, 2025](https://doi.org/10.1016/j.fuproc.2024.108170)). Autonomous catalysis should retain human review and accountable experimental execution ([Nature Catalysis perspective, 2025](https://www.nature.com/articles/s41929-024-01275-5)).

## Evidence ladder and cross-repository boundaries

- **Symthaea:** hypothesis generation, uncertainty handling, decision/experiment ranking, evidence-bound interpretation. Reuse [PIE-001 mass-boundary work](https://github.com/Luminous-Dynamics/symthaea/issues/1606); do not fork it silently.
- **Symtropy:** may own resource/schedule/physical-layout campaign constraints if supported by a declared model. It is not treated here as a validated petroleum thermodynamics or reaction simulator.
- **Sol-Atlas:** site-specific supply, transport, energy/water access, industrial demand and spatial dependencies. Modeled capacity is not actual delivery.
- **Mycelix:** source lineage, custody, revision bindings and evidence receipts; signatures attest to a record/claim source, not automatically to the truth of a physical measurement.

B0 here is software correctness only. B1 is literature reproduction. B2 is flowsheet comparison within qualified model boundaries. B3 requires bench results and controls. B4 requires pilot/industrial operating evidence, lifetime/degradation, maintenance, off-spec output, capex/opex and lifecycle accounting.

The 2025 [IEA World Energy Outlook](https://www.iea.org/reports/world-energy-outlook-2025) presents multiple energy futures. All downstream investments must therefore be stress-tested against more than one future price/demand/policy scenario.

## Out of scope

No plant actuation/control, equipment certification, safety approval, reserve certification, proof of a new catalyst, emissions-offset claim, or assertion of commercial viability. No target percentage for beating industry baselines is frozen in V0: first establish correctness, evidence access and a fair baseline, then preregister a meaningful performance target.
