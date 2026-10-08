# RH-006 Multivariate Identified-Set Propagation — v1

## Status

Research diagnostic only.

- formal inference: disabled
- formal p-value: disabled
- formal confidence interval: disabled
- selector/authorization: unchanged

This layer generalizes the RH-006 identified-set propagation seam from a scalar nuisance interval to a two-parameter compact axis-aligned nuisance set.

## 1. Why 2-D first

The generalized RH-006 joint-DGP construction already implies polynomial risk/discriminant surfaces in multiple nuisance coordinates.

The first exact multivariate implementation is deliberately bounded to two nuisance dimensions. This is enough to expose cross-parameter curvature, interior stationary extrema, boundary extrema, and sign regions that cannot be certified from independent scalar intervals alone.

No claim is made for arbitrary-dimensional nonlinear global optimization.

## 2. Exact quadratic box image

For f(x,y)=a x^2 + b y^2 + c x y + d x + e y + f0 over a closed rectangle, global extrema occur at a corner, an edge stationary point, or an interior stationary point when the Hessian is nonsingular and the point lies inside the rectangle.

The implementation evaluates those candidates directly. Therefore the resulting image is not grid-defined.

A dense grid is used only by the independent falsification harness as a cross-check that sampled values lie inside the analytically computed interval.

## 3. RH-006 discriminant surface

For a two-parameter affine/quadratic nuisance slice:

B(x,y)=B0+B_x x+B_y y

C(x,y)=C0+C_x x+C_y y+C_xx x^2+C_xy x y+C_yy y^2

the ray equation remains Delta(tu;x,y)=A t^2+B(x,y)t+C(x,y).

Consequently D(x,y)=B(x,y)^2-4 A C(x,y) is itself quadratic in (x,y).

## 4. Decision certification

For A>0, positive-root count is governed by D, B, and C.

The 2-D gate computes exact interval images for these surfaces and certifies a constant positive-root count only when those ranges imply the same count throughout the full region.

The strict certification rules are:

- zero roots when D is strictly negative everywhere;
- one positive root when D is strictly positive and C is strictly negative everywhere;
- two positive roots when D is strictly positive, C is strictly positive, and B is strictly negative everywhere;
- zero positive roots when D is strictly positive, C is strictly positive, and B is strictly positive everywhere;
- unknown at any unresolved sign or boundary case.

The B=0 boundary is deliberately not classified as zero: with A>0 and C>0, B=0 gives one negative and one positive root.

Otherwise decision_count=unknown and the gate fails closed. This is deliberately conservative.

## 5. Canonical multivariate adversary

Use D(x,y)=0.25-x^2-y^2 over (x,y) in [-1,1]^2.

The exact discriminant image is [-1.75, 0.25].

Therefore all three regimes remain reachable: two real branches, double root, and no real branch.

## 6. Stable multivariate fixture

A separate box uses C(x,y)=-0.5+0.1x+0.05y with constant positive discriminant.

Its exact C image is [-0.65,-0.35].

Therefore one positive branch is certified throughout the box.

## 7. Approximation polarity remains mandatory

The RH-006 layer distinguishes witness-only, inner approximation, outer approximation, and sharp set.

An exact geometric image of a declared rectangular nuisance set does not prove that the rectangle is the sharp identified set. The geometric propagator certifies only what follows from the supplied set representation.

## 8. Joint-constraint preservation with convex polygons

A Cartesian rectangle can be only an outer representation of a genuinely joint
identified region. The branch therefore adds a second exact geometry:

ConvexPolygon2d.

For a compact convex polygon, quadratic extrema are obtained from:

- polygon vertices;
- stationary points along every polygon edge;
- an admissible interior stationary point.

This preserves constraints such as:

x >= 0, y >= 0, x + y <= 1

without enlarging the set to a rectangle.

The canonical triangular region (0,0), (1,0), (0,1) with
D(x,y)=0.25-x²-y² has exact image [-0.75, 0.25].

The same discriminant adapter is exposed from RaySurface, so the RH-006
polynomial surface can be propagated directly over a joint convex region.

This remains a geometric propagation result: it does not claim that the
polygon is the sharp identified set.
## 9. Independent falsification

The Python harness ran with seed 20261008 across 250 randomized quadratic surfaces and a 129x129 grid used solely as a falsification cross-check.

Observed: zero containment violations; canonical image matched exactly; stable fixture matched exactly.

The grid result is not used as the scientific definition.

## 10. Set-origin provenance

Approximation polarity is not sufficient by itself. The origin of the region is also typed:

- observational-equivalence-derived;
- maintained-restriction region;
- sensitivity-analysis region;
- unknown.

Only the observational-equivalence-derived origin can support an identified-set claim.
A mathematically exact sensitivity rectangle or polygon remains a sensitivity region unless
the identification argument derives that region from the observed law and maintained assumptions.

This prevents exact geometry from becoming a substitute for identification evidence.

The executable provenance micro-gate asserts:

- sharp + sensitivity region is not an identified set;
- outer + observational-equivalence can certify a stable identified-set decision;
- inner + observational-equivalence can certify an unstable identified-set decision;
- witness-only + observational-equivalence certifies neither.

## 11. Evidence boundary

This layer does not establish a sharp RH-006 multivariate identified set, point identification, bootstrap validity, uniform size control, confidence-set coverage, empirical predictive validity, arbitrary-dimensional global optimization, or formal inference.

The inference selector remains unchanged and closed.

## 12. Next gate

The next scientific extension should characterize the joint identified set geometry itself:

observed law -> admissible latent DGP class -> sharp/inner/outer joint set -> exact polynomial surface image -> decision region -> connected-component audit.

Only after that should higher-dimensional approximation be introduced, with explicit outer/inner guarantees rather than unconstrained gridding.

## 13. Literature boundary

Bontemps and Magnac (2017) emphasize geometric characterization of identified sets and inference under moment restrictions.

Schlemper and Moreira (2026) provide a current weak-identification example where grid inversion can miss disconnected or unbounded confidence-set components and show how polynomial structure can support reliable inversion.

These are methodological precedents, not a plug-in validity theorem for RH-006.