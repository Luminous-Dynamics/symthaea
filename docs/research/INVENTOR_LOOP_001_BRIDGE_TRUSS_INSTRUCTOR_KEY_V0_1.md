# INVENTOR-LOOP-001 — Bridge / Truss Paper Exercise A: Instructor Key

**Status:** DESIGN_ONLY / analytical teaching reference  
**Companion learner packet:** [Bridge / Truss Design Challenge v0.1](INVENTOR_LOOP_001_BRIDGE_TRUSS_LEARNER_PACKET_V0_1.md)  
**Programme:** [INVENTOR-LOOP-001](INVENTOR_LOOP_001_PROGRAM_V0_1.md)  
**Tracking:** [Symthaea #10442](https://github.com/Luminous-Dynamics/symthaea/issues/10442)  
**Date:** 2026-11-04

> This is an independently derived closed-form result for one idealized mathematical truss. It is not a solver test receipt, physical measurement, benchmark grader, structural design approval, code-compliance assessment or safety certification. It is public teaching material; do not use it as a hidden evaluation oracle.

## 1. Frozen case definition

Nodes (m):

- A = (0, 0)
- B = (4, 0)
- C = (2, 3)

Members: A–C and B–C.

Properties:

- Both members use E = 200 GPa = 2.0 × 10¹¹ Pa.
- Both members use A = 10 mm² = 1.0 × 10⁻⁵ m².
- EA = 2.0 × 10⁶ N.
- Both support nodes have ux = uy = 0.
- Apex load is FC = (0, −100) N.
- No self-weight; ideal pins; axial bar response only; static, small-displacement, linear-elastic 2D model.

Each member length is:

[
L=sqrt{2^2+3^2}=sqrt{13}=3.605551275ldots;\mathrm{m}
]

Direction cosines from the base node toward C:

[
n_A=\left(\frac{2}{\sqrt{13}},\frac{3}{\sqrt{13}}\right),
\qquad
n_B=\left(-\frac{2}{\sqrt{13}},\frac{3}{\sqrt{13}}\right)
]

## 2. Apex stiffness and displacement

For an axial bar, the contribution to the free-node stiffness is:

[
K_e=\frac{EA}{L}
\begin{bmatrix}
c^2 & cs\\
cs & s^2
\end{bmatrix}
]

The two mirrored members cancel the off-diagonal terms. The reduced apex stiffness is:

[
K_C=\frac{EA}{L}
\begin{bmatrix}
8/13 & 0\\
0 & 18/13
\end{bmatrix}
=
\begin{bmatrix}
8EA/(13L) & 0\\
0 & 18EA/(13L)
\end{bmatrix}
]

Numerically:

[
K_C\approx
\begin{bmatrix}
341353.9669 & 0\\
0 & 768046.4255
\end{bmatrix}
\;\mathrm{N/m}
]

With FC = (0, −100) N:

[
u_C=K_C^{-1}F_C
=
\left(
0,
-\frac{13LP}{18EA}
\right)
]

Therefore:

[
u_{Cx}=0
\]

[
u_{Cy}=-\frac{13\sqrt{13}\cdot100}{18\cdot2.0\times10^6}
=-0.0001302004627\ldots\;\mathrm{m}
]

or −0.13020046 mm vertically. The negative sign means downward displacement.

## 3. Member force, stress, strain and shortening

For either member, signed axial force is:

[
N=\frac{EA}{L}(n\cdot u_C)
=-\frac{13P}{6\sqrt{13}}
=-60.09252126\ldots\;\mathrm{N}
]

Negative force denotes compression under the declared sign convention.

Stress:

[
\sigma=\frac{N}{A}
=-6.00925213\ldots\;\mathrm{MPa}
]

Strain:

[
\epsilon=\frac{N}{EA}
=-3.00462606\ldots\times10^{-5}
]

Axial change in length:

[
\delta=\frac{NL}{EA}
=-\frac{13P}{6EA}
=-0.0001083333333\ldots\;\mathrm{m}
]

Each member shortens by approximately 0.10833333 mm. Both members have the same force, stress, strain and shortening because the geometry and properties are symmetric.

## 4. Support reactions

At A, the member force acting on the joint is (N n_A). The support reaction balances it:

[
R_A=-Nn_A
=
\left(
\frac{P}{3},
\frac{P}{2}
\right)
=
(33.3333333,50)\;\mathrm{N}
]

At B:

[
R_B=-Nn_B
=
\left(
-\frac{P}{3},
\frac{P}{2}
\right)
=
(-33.3333333,50)\;\mathrm{N}
]

Signs use +x right and +y up.

## 5. Independent equilibrium checks

Global horizontal force:

[
\sum F_x=33.3333333-33.3333333=0
]

Global vertical force:

[
\sum F_y=50+50-100=0
]

Moment about A:

[
\sum M_A=50\cdot4-100\cdot2=0
]

The two support reactions and the applied apex load satisfy global force and moment equilibrium.

## 6. Energy check

The external work for a linear load ramp is:

[
W_{ext}=\frac12 P|u_{Cy}|
=0.006510023136\ldots\;\mathrm{J}
]

The total member strain energy is:

[
U=\sum_e\frac12N_e\delta_e
=2\cdot\frac12N\delta
=0.006510023136\ldots\;\mathrm{J}
]

Thus (U=W_{ext}) for this idealized linear static model.

## 7. Summary values

| Quantity | Expected value |
|---|---:|
| Member length | 3.605551275 m |
| Apex horizontal displacement | 0 m |
| Apex vertical displacement | −0.130200463 mm |
| Axial force in each member | −60.09252126 N (compression) |
| Stress in each member | −6.00925213 MPa |
| Strain in each member | −3.00462606 × 10⁻⁵ |
| Axial change in each member length | −0.108333333 mm |
| Reaction at A | (+33.3333333, +50) N |
| Reaction at B | (−33.3333333, +50) N |
| Strain energy | 0.006510023136 J |
| External work | 0.006510023136 J |

Rounding tolerances should be selected for the learner's numerical method. These displayed values are not universal solver acceptance tolerances.

## 8. Metamorphic reasoning prompts

The following comparisons hold within the same idealized, linear-elastic, statically determinate model:

- Double P: member forces, stresses, reactions and displacements double; energy/work increase by a factor of four.
- Double E while holding A, geometry and load fixed: member forces and reactions are unchanged; displacement, strain and shortening halve; stress is unchanged.
- Double A while holding E, geometry and load fixed: member forces and reactions are unchanged; displacement, strain, shortening and stress halve.
- Mirror the geometry about the vertical centerline and mirror all loads: the x components reverse sign while corresponding magnitudes and vertical results remain symmetric.

A metamorphic relation tests consistency under a controlled transformation; it does not validate the physical model.

## 9. What this reference does not model

It excludes member bending, shear deformation, buckling, geometric nonlinearity, contact, connection eccentricity or failure, material plasticity, fracture, dynamics, construction defects, residual stress, fatigue, corrosion, temperature dependence, self-weight and support/soil flexibility.

The result is an equation-level teaching reference for one specified model. It does not establish that a real bridge, component, material, joint or fixture can safely carry 100 N—or any other real-world load.
