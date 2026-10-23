# Solar Infrastructure Fabric — implementation notes

## Current tranche

Draft PR #1402 now contains two implementation slices:

- **SIF-001:** a leaf `symthaea-infrastructure` crate containing the domain-neutral coordination vocabulary and authority boundary. The workspace already includes `crates/domains/*`, so this avoids editing the large `symthaea-engineering` facade merely to expose foundational types.
- **SIF-002 (foundation only):** `symthaea-orbital::cislunar`, containing Earth-Moon CR3BP parameters, deterministic collinear Lagrange-point solving, rotating/inertial state transforms, and a differential third-body acceleration primitive.

The infrastructure crate intentionally has only `serde` as an internal dependency boundary. It contains no networking, flight control, robotics, physics, hardware drivers, HDC, or consciousness dependencies.

## Required verification before promotion from draft

1. run `cargo test -p symthaea-infrastructure --lib`;
2. run `cargo test -p symthaea-orbital --lib`;
3. run workspace dependency/cycle and orphan-module checks;
4. confirm no canonical `AssetId`, resource, reservation, or authorization type should replace the new leaf types;
5. review the scalar `unit: String` boundary and decide whether SIF-003 adapters should normalize into `uom` before domain execution;
6. add property tests if repository conventions require them for these value/window invariants;
7. verify that `CommandProposal` remains non-executable and that `Authorization` cannot bypass local controller acceptance;
8. compare CR3BP regression values with an independent reference implementation before treating them as evidence beyond unit-test scale.

## Evidence boundary

The CR3BP constants/solver are a circular mean-geometry model, not ephemeris-grade Earth-Moon state. The third-body primitive is a differential acceleration building block, not a complete n-body propagator. Simulation and unit-test results do not establish mission qualification.

SIF-003 and the higher-fidelity parts of SIF-002 remain tracked in issue #1403 so digital-twin composition, ephemerides, perturbations, and operational benchmarks can proceed as separate evidence lines.
