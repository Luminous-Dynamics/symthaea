# SX-033: Uncertainty-Aware Cumulative Resource Budgets v0.1

**Status:** experimental software-contract verifier; synthetic evidence only  
**Implementation:** suit_resource_budget  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/7543  
**Predecessor:** SX-032 task proof chain, PR #7532

## Paradigm shift: budgets are uncertainty claims, not single numbers

A mission task graph can be connected and every task can have declared capability evidence while the resource plan still over-allocates oxygen, water, electrical energy or another cumulative resource. SX-033 adds an explicit worst-case accounting layer that binds resource supply to exact providers, revisions, evidence labels and operating environments, then compares declared demand against conservative availability and a reserve.

The v0.1 resource-kind enum encodes canonical units in its variant name:

- oxygen mass in micrograms;
- water volume in microliters;
- consumable mass in micrograms;
- electrical energy in microwatt-hours;
- data volume in bytes;
- crew time in milliseconds.

This avoids accidental unit mixing inside one budget calculation. Quantities are unsigned integers, not floating-point values, so NaN and fractional rounding cannot silently affect acceptance.

## Conservative interval semantics

Each supply and demand uses a declared interval: lower bound, nominal value and upper bound. Inputs are rejected unless lower <= nominal <= upper.

For each resource pool, the verifier sums the upper bounds of **required-task** demand, adds a caller-selected reserve rounded up to the next integer quantum, and compares that total to the **lower bound** of available supply:

required_capacity = required_demand_upper + ceil(required_demand_upper * reserve_basis_points / 10,000)

The required pool fits only when available.lower >= required_capacity.

Optional-task demand is accounted separately. It cannot turn an otherwise feasible required budget into a blocker: when required demand plus reserve fits but adding optional demand exceeds the lower supply bound, the resource plan is degraded and reports the optional shortfall. The v0.1 policy does not implicitly combine supplies across separate provider pools. Reusing the same provider/resource kind as two pools is rejected to prevent accidental double-counting.

## Binding and fail-closed conditions

The plan pins:
- mission ID and configuration revision;
- SX-031 configuration digest and SX-032 task-plan digest;
- exact resource pool IDs, resource kinds, provider component IDs and hardware revisions;
- supply and demand intervals, evidence maturity labels and evidence references;
- provider environment scope, evidence floor and reserve policy;
- each demand's exact task ID and pool ID.

Evaluation re-runs SX-032 over the exact task list and requires the supplied task receipt to match. Resource findings detect missing provider components, provider-revision mismatch, unreachable provider, environment-scope mismatch and supply/demand evidence below the caller's floor. Requirements never infer a missing pool or silently coerce one resource kind to another. Input ambiguity, bad digests, duplicate identities, unused pools, bad intervals and arithmetic overflow fail closed.

Receipt digests cover canonically ordered pool/demand declarations and the complete resource report. Receipt verification re-evaluates the configuration/task chain and compares exact typed results; the hash is change evidence, not a signature or proof that external evidence is authentic.

## Synthetic test coverage

The module has unit cases for:
- a conservative interval with a 20% reserve and exact reported margin;
- required-demand shortfall of one canonical quantum blocking the budget;
- optional overage degrading while required demand remains within budget;
- invalid interval ordering;
- provider hardware-revision mismatch;
- resource-kind mismatch;
- stale task receipt and configuration revision;
- vector-order determinism;
- reserve rounding and overflow;
- duplicate provider/resource pool rejection;
- receipt invalidation after demand mutation;
- a deterministic 2,000-case interval boundary corpus that checks reserve rounding, required capacity, block/pass classification, shortfall and margin; receipt verification is repeated at fixed seeds.

The integration replay independently reimplements the resource-plan, resource-report, task-receipt-binding and resource-receipt digest encodings, then checks both a successful report and a blocked report. This is an independent replay implementation in source; execution has not been observed in this environment.

The integration replay fixes a synthetic configuration with 1,000 units of conservative supply, 400 units of worst-case required demand and an 80-unit reserve. Expected required capacity is 480 and remaining required margin is 520. A second replay sets conservative supply to 479 and must block with a one-unit shortfall.

All IDs, values, capabilities and evidence references above are synthetic. The tests do not validate actual life-support consumption, spacecraft equipment, fluid dynamics, physiology, crew workload or hardware safety.

## Proposed benchmark targets

These are goals, not measured results. Freeze the CPU, OS, Rust toolchain, build profile, deterministic generator seed and corpus manifest before establishing CI performance gates.

| Benchmark | Initial target |
|---|---|
| Invalid quantity intervals | 100% rejected |
| Required demand + reserve above conservative supply | 100% blocked |
| Optional-only overage with required reserve satisfied | 100% degraded, not blocked |
| Wrong resource kind / missing provider / revision mismatch / unreachable provider / environment mismatch | 100% seeded detection |
| Receipt mutations to pool, demand, quantity, reserve, task or configuration | 100% rejection |
| Equivalent pool/demand vector permutations | Identical report and receipt across 1,000 permutations |
| Generated budget/fault corpus beyond the implemented 2,000-case interval sweep | 2,000 seeded cases, zero unexpected panics |
| 1,000 pools / 20,000 demand claims | p95 below 100 ms on a documented runner |

All "100%" correctness claims must report numerator and denominator and preserve positive controls alongside negative controls. Timings are secondary to conservative accounting, no-implicit-pooling semantics, and fail-closed mutation coverage.

## Research basis

NASA systems engineering guidance defines margin as an allowance in budgets and technical performance parameters for uncertainty and risk. NASA's software engineering handbook separately recommends establishing and maintaining quantitative margins for critical resources. NASA-STD-3001 Volume 2 Revision F supplies human-system requirements across crew-facing equipment, environments, operations and spacesuits. SX-033 uses these as motivation for explicit margin bookkeeping; it does not encode NASA's full standards or claim compliance.

- NASA Systems Engineering Handbook appendix: https://www.nasa.gov/reference/system-engineering-handbook-appendix/
- NASA SWEHB 9.12, Resource Margins: https://swehb.nasa.gov/spaces/SWEHBVC/pages/85426336/9.12%2BResource%2BMargins
- NASA-STD-3001 Volume 2 Revision F, active: https://standards.nasa.gov/node/237

## Scope deliberately excluded

This is cumulative accounting only. Peak power needs time-window concurrency and scheduling; thermal/coolant coupling needs physical models; resource correlations need a justified uncertainty model; crew activity needs scheduling and human-system validation. Those should be separate, evidence-backed extensions rather than assumptions hidden inside a generic sum.