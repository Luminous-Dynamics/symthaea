# LQCD-020R — EHK beta=6.0 Wilson-action static-potential benchmark

This artifact freezes an **external reproduction target**, not a Symthaea result.

Canonical benchmark JSON SHA-256:

`bf35a192e337f12fdd4728de40f2bb04bb0f2e81cc1e796f2eb6a603774c18c3`

## Source identity

- R. G. Edwards, U. M. Heller, T. R. Klassen
- *Accurate Scale Determinations for the Wilson Gauge Action*
- arXiv:hep-lat/9711003v2
- Nucl. Phys. B517 (1998) 377-392

The source studies the pure SU(3) Wilson gauge action and uses correlated static-potential fits with the tree-level lattice Coulomb term. Its general ansatz is

`V(r) = V0 + sigma*r - e*C_lat(r) + l*(C_lat(r) - 1/r)`.

For string-tension stability it evaluates the free four-parameter form, a three-parameter form with `e = pi/12`, and a two-parameter form with `e = pi/12, l = 0`, while increasing the minimum fitted separation.

## Frozen beta=6.0 targets

- `a * sqrt(sigma) = 0.2189(9)`
- `r0 / a = 5.369(9)`
- `r4 / a = 8.831(21)`
- `r6 / a = 10.89(3)`

These values are frozen as target central values and quoted uncertainties only. This artifact does **not** assert that one successful beta=6.0 comparison establishes a physical prediction, a continuum result, or the correctness of Symthaea's operator construction.

## Comparison policy boundary

No universal pass/fail percentage is encoded here. A later preregistered comparison must preserve Symthaea's correlated uncertainties and report discrepancies against the quoted external uncertainties. Operator, smearing, fit-range and finite-volume differences must remain visible rather than being absorbed into a permissive tolerance.

The benchmark is intended as the first external static-potential reproduction gate before a multi-beta finite-volume/continuum program.
