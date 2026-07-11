# symthaea-structural

A self-contained **structural / mechanical engineering statics** layer for
Symthaea: section properties, Euler-Bernoulli beam analysis, axial members,
Euler buckling, and a 2D truss solver. Fills a gap — the workspace had robotics
and materials crates but no beam/section/buckling/truss statics.

Pure `std`, zero dependencies, no `symthaea-core` link. All results are
closed-form or exact linear-algebra, checked against textbook hand calculations.

## Capabilities

| Area | API |
|------|-----|
| Sections | `Section::{rectangular, circular, hollow_circular}`, `section_modulus` |
| Beams | `Beam::analyze(LoadCase)` → deflection, moment, stress, factor of safety |
| Members | `axial_stress/strain/elongation`, `euler_buckling_load` |
| Trusses | `Truss::solve` (method of joints), `check_truss_members` (stress + buckling) |
| Materials | `material::{steel_a36, aluminum_6061}` |

The four canonical statically-determinate beam cases are exposed via a combined
`LoadCase` enum so invalid support/load pairings are unrepresentable. Truss
analysis checks static determinacy (`members + reactions == 2·nodes`) and returns
errors for indeterminate, singular, or zero-length inputs.

## Example

```rust
use symthaea_structural::{Beam, LoadCase, Section, material::steel_a36};

let beam = Beam { length: 2.0, section: Section::rectangular(0.05, 0.1), material: steel_a36() };
let r = beam.analyze(LoadCase::CantileverEndPoint(1000.0));
assert!((r.max_deflection - 0.0032).abs() < 1e-6); // 3.2 mm
assert!(r.factor_of_safety > 10.0);
```

## Validation

```bash
cargo test -p symthaea-structural
```

## Not yet

Frames / method of sections, indeterminate structures (stiffness matrix),
dynamics.
