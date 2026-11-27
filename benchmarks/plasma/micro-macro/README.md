# High-field VPFP micro–macro scaling contract v0.1

This contract freezes the equation family, nondimensionalization, epsilon definition / sequence, field scaling, retained macroscopic variables, micro-space constraints, discrete collision operator identity, and limiting-model identity for one high-field VPFP cohort.

It is deliberately separate from the collisionless periodic 1D1V Vlasov–Poisson Landau / echo tournament.

## Why freeze this first

The notation and singular scaling vary across VPFP formulations. A micro–macro decomposition is meaningful only relative to the exact local Maxwellian, collision operator, discrete nullspace, moment constraints, and asymptotic limit of the selected model. The fixture therefore does not hard-code an assumed full set of collision invariants. It requires the selected reference to identify the discrete collision operator and its nullspace explicitly.

## Weighted micro projector

The Rust primitive `project_weighted_micro_fluctuation` solves

`min 0.5*(g_micro-g)^T W (g_micro-g), subject to C*g_micro=0`

for positive diagonal W and caller-supplied discrete constraint rows C. It verifies post-projection moments, reports the weighted correction norm, and fails closed on dependent constraints or invalid inputs.

This is a numerical projector primitive only. The caller is responsible for deriving C from the exact collision operator and its declared macro/micro decomposition. The synthetic contract does not claim that density is the only invariant for any physical VPFP operator.

## Claim boundary

The included fixture is schema-only and manufactured. It is not a VPFP trajectory, does not validate a high-field limit, and does not reproduce the Zhang–Hu Micro-ROM. Physical qualification requires the pinned reference, matched epsilon sequence, macro/limit residual, collision dissipation, moment / positivity gates, and complete offline/online cost accounting.

Research owner: [#13483](https://github.com/Luminous-Dynamics/symthaea/issues/13483).
