# ROB-BOOT C0 Independent Oracle v1

Status: **REFERENCE / QUALIFICATION FIXTURE ONLY**

Authority: independent analytical/software oracle for the C0 synthetic rehearsal.

This document does **not** establish material truth, physical structural validity,
fabrication quality, measurement validity, safety, or an R2 result. Production
optimization/evaluation code must not call this oracle.

Related work:

- ROB-DESIGN-001C0 #5084
- ROB-BOOT-000 #5078
- ROB-BOOT-000A #5107
- preregistration template `ROB_BOOT_C0_PREREGISTRATION_V1.md`

## 1. Frozen C0 analytical profile

C0 uses a solid rectangular beam/link coupon with:

- fixed span `L`;
- simply supported boundary condition;
- centered transverse point load `F`;
- midspan deflection measurement;
- fixed section orientation;
- one homogeneous isotropic linear-elastic material profile;
- constant density `rho` and modulus `E` across candidates.

Candidate variables are rectangular-section width `b` and height `h` only.

For this profile:

```text
A      = b h
I      = b h^3 / 12
M_max  = F L / 4
sigma  = 3 F L / (2 b h^2)
delta  = F L^3 / (4 E b h^3)
mass   = rho b h L
```

These equations are an oracle for this bounded profile only.

## 2. Dimensionless oracle

Let the baseline section be `(b0, h0)` and define:

```text
bw = b / b0
hh = h / h0
```

When `F`, `L`, `E`, `rho`, support conditions, orientation, and material remain
identical, the following ratios are independent of their absolute values:

```text
mass_ratio       = bw * hh
deflection_ratio = 1 / (bw * hh^3)
stress_ratio     = 1 / (bw * hh^2)
```

This dimensionless layer can therefore qualify parameter mapping, section
orientation, objective reproduction, feasibility logic, and Pareto selection
before absolute material/load evidence is available.

It cannot qualify physical prediction.

## 3. Exact baseline

```text
bw = 1.0
hh = 1.0
mass_ratio       = 1.0
deflection_ratio = 1.0
stress_ratio     = 1.0
```

## 4. Spot-check known answers

### Candidate A

```text
bw = 0.8
hh = 1.1
mass_ratio       = 0.88
deflection_ratio = 0.9391435011269719
stress_ratio     = 1.0330578512396693
```

Interpretation: lighter and slightly stiffer in the declared bending axis, but
slightly higher maximum bending stress than baseline.

### Candidate B

```text
bw = 0.7
hh = 1.2
mass_ratio       = 0.84
deflection_ratio = 0.8267195767195769
stress_ratio     = 0.9920634920634921
```

Interpretation: lighter, stiffer, and slightly lower idealized bending stress in
this bounded analytical model.

### Candidate C

```text
bw = 0.9
hh = 1.0
mass_ratio       = 0.9
deflection_ratio = 1.1111111111111112
stress_ratio     = 1.1111111111111112
```

Interpretation: lighter but more flexible and higher-stress.

## 5. Frozen exhaustive synthetic grid

For BOOT-001A synthetic rehearsal only, define the independent fixture domain:

```text
bw in {0.8, 0.9, 1.0, 1.1, 1.2}
hh in {0.8, 0.9, 1.0, 1.1, 1.2}
```

This is a 25-candidate grid. It is **not** the physical campaign search domain.
Physical bounds remain `UNSET` until the preregistration is instantiated.

The reference evaluator computes `mass_ratio` and `deflection_ratio` directly
from the equations above and minimizes both independently. No scalar weighted
fitness is permitted.

## 6. Expected nondominated frontier

The exact nondominated set for the frozen 5×5 fixture grid, minimizing both
`mass_ratio` and `deflection_ratio`, is:

| `bw` | `hh` | mass ratio | deflection ratio |
|---:|---:|---:|---:|
| 0.8 | 0.8 | 0.64 | 2.44140625 |
| 0.8 | 0.9 | 0.72 | 1.7146776406035664 |
| 0.8 | 1.0 | 0.80 | 1.25 |
| 0.8 | 1.1 | 0.88 | 0.9391435011269719 |
| 0.8 | 1.2 | 0.96 | 0.7233796296296298 |
| 0.9 | 1.2 | 1.08 | 0.6430041152263375 |
| 1.0 | 1.2 | 1.20 | 0.5787037037037038 |
| 1.1 | 1.2 | 1.32 | 0.5260942760942762 |
| 1.2 | 1.2 | 1.44 | 0.48225308641975323 |

Expected frontier cardinality:

```text
9
```

A production Pareto implementation that reports a different frontier under this
exact fixture requires investigation before C0 search authority advances.

## 7. Required synthetic checks

A future independent/reference test harness should verify at least:

1. all 25 fixture candidates are generated exactly once;
2. parameter insertion order does not change semantic candidate identity;
3. width and height are not silently transposed;
4. the evaluator reproduces the dimensionless equations within a frozen numeric tolerance;
5. the nondominated frontier contains exactly the 9 entries above;
6. dominated candidates never enter the reference frontier;
7. search order/random seed cannot change objective values;
8. a search algorithm may fail to discover a frontier point, but may not alter the oracle frontier;
9. production code cannot import/call the independent oracle implementation;
10. a successful synthetic oracle comparison creates no physical/material/safety claim.

## 8. Orientation hostile test

The section orientation is semantic.

For a non-square candidate, exchanging width and height changes the declared
bending inertia:

```text
I = b h^3 / 12
```

Therefore:

```text
(b, h) != (h, b)
```

for the C0 structural subject unless `b == h`.

A serializer that canonicalizes width/height by numeric sorting is physically
wrong and must fail this fixture.

## 9. Unit hostile test

The dimensionless oracle is deliberately immune to absolute-unit choice when
ratios are formed consistently. The absolute analytical adapter is not.

The implementation should therefore include a separate hostile fixture where a
millimetre/metre conversion error causes an obvious failure in absolute mass or
deflection while leaving the dimensionless ratio oracle unchanged. This helps
localize unit bugs rather than allowing two wrong conversions to cancel.

## 10. Claim ceiling

Passing this oracle may establish only that, for the frozen synthetic C0 profile:

- parameter mapping is consistent;
- the bounded analytical equations are reproduced;
- candidate enumeration is complete;
- objective vectors are stable;
- Pareto classification matches an independent oracle.

It does **not** establish:

```text
absolute material properties
absolute load calibration
fixture ideality
Euler-Bernoulli applicability to a physical article
physical mass prediction
physical deflection prediction
fabrication repeatability
measurement capability
physical safety
R2 improvement
C1 JointCarrier competence
humanoid competence
```

The oracle exists to make the first recursive-design pipeline easier to falsify,
not easier to overclaim.
