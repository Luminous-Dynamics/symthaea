# Nuclear TEA reference implementation

A small Rust reference package for the Luminous nuclear port/marine techno-economic programme.

## Scope

This package currently provides:

- structural validation of the frozen South African port shore-power / spare-capacity source dataset;
- SHA-256 input identity in a machine-readable verification receipt;
- discounted lifecycle-cost and discounted delivered-service calculations;
- explicit real/nominal cash-flow and discount-rate basis matching;
- explicit service unit and delivery-boundary matching before comparisons;
- scenario-level evidence references in every economic input series;
- mandatory feasibility-gate aggregation that blocks economic ranking on FAIL or UNKNOWN.

It does not model a reactor, grid load flow, ship hydrodynamics, thermal hydraulics, structural response, radiation, nuclear safety, insurance pricing, or actual project economics.

## Run

From the repository root:

```bash
cargo test --manifest-path tools/nuclear-tea/Cargo.toml
cargo run --release --manifest-path tools/nuclear-tea/Cargo.toml -- research/nuclear/port-reference-evidence-sa-2025.json
```

The executable prints a JSON receipt with the exact input SHA-256, package version, build commit (when run under GitHub Actions), row counts and the limited validation status. It exits nonzero on parsing or validation failure.

## Calculation semantics

The first annual flow is project year 0. Costs and delivered service are discounted using the declared rate. All flows must use one declared price basis. Real cash flows require a real discount rate; nominal cash flows require a nominal rate. The code refuses mixed bases.

Levelized cost is:

```
sum(discounted lifecycle costs)
--------------------------------
sum(discounted delivered service)
```

The delivered service unit and delivery boundary are mandatory. Comparisons are blocked unless both cases have matching currency, price basis, base year, discount rate, service unit and delivery boundary. Every mandatory feasibility gate must be explicitly PASS and carry a non-empty evidence reference.

Hard gates are not costs. A failed gate yields NO_GO; an unresolved mandatory gate, a blank gate ID, or a PASS without an evidence reference yields UNKNOWN. A set with no mandatory gates is UNKNOWN rather than vacuously eligible. Evidence references are locators only: this package does not authenticate issuers or determine legal authority.

## South African source dataset

The source dataset contains separate published estimates for:
- vessel-derived shore-power demand; and
- spare port-intake capacity using the source's stated 0.9 power-factor conversion.

These are not one synchronized load-flow record. The verifier does not join them, subtract one from the other, or infer a grid connection offer. The dataset remains PUBLISHED_ESTIMATE, not metered observation.

## Build reproducibility status

This initial slice does not yet include a committed Cargo.lock. The direct dependencies use semver ranges, so the resolved transitive dependency graph is not frozen. Before any production or long-lived benchmark use, generate and commit the exact lockfile, pin the toolchain, and bind the lockfile/toolchain digests into the build receipt. The current workflow checks source formatting, Clippy, tests and input structure, but does not by itself freeze the dependency graph.

## Evidence and claim ceiling

A successful port-data receipt establishes only that the exact input file meets the structural checks in this package and that the stated conversion arithmetic is internally consistent. It does not verify the original studies, prove source truth, validate a port electrical network, establish a nuclear site, or approve any infrastructure.

A successful arithmetic unit test establishes only the tested calculation semantics against the synthetic analytical fixtures. It does not establish actual levelized costs or comparative competitiveness.

The research parent is [NUCL-TEA-001 #11343](https://github.com/Luminous-Dynamics/symthaea/issues/11343); the implementation child is [#11363](https://github.com/Luminous-Dynamics/symthaea/issues/11363). The wider programme includes the UCL merchant-shipping TCO benchmark, the separate Lloyd's Register containership scenario, and South African port-readiness cases; these are not assumed to be directly comparable.
