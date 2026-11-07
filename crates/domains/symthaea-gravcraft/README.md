# symthaea-gravcraft

GravCraft is an experimental metric-ansatz research sandbox. It is not a
demonstrated propulsion technology.

## Two different code paths

### Legacy kinematic simulator

`simulator::MetricPerturbationSimulator` is a heuristic toy. Its current
acceleration method uses the amplifier shift vector directly as a coordinate
acceleration. It does not derive that acceleration from a 4D metric and its
Christoffel symbols. The legacy path is useful only for controller, interface
and deterministic regression experiments; it must not be presented as a GR
geodesic simulation.

### ADM metric reference

`adm_reference::AmplifierAdmMetric` provides a mathematically explicit
static 3+1 ADM metric from the existing amplifier shift:

    ds² = -(c dt)² + δᵢⱼ (dxⁱ + βⁱ c dt)(dxʲ + βʲ c dt)

with unit lapse, flat Euclidean spatial metric, and βᵢ produced by the
three-amplifier shaping model. The coordinates are (ct, x, y, z) in metres;
the metric-geometry kernel uses G=c=1.

The module then provides:

- Lorentzian-inertia and inverse-metric checks.
- Christoffel symbols from metric derivatives.
- Properly normalized timelike initial four-velocities.
- Affine-parameter RK4 integration of the full 4D geodesic equation.
- Conservation checks for g(u,u) along the integrated trajectory.
- Einstein-implied stress-energy reconstruction,
  Tᵤᵥ = (Rᵤᵥ - ½ gᵤᵥ R)/(8π), in geometrized units.
- Eulerian energy density and finite-direction sampling of the null-energy
  contraction.

This is a fixed-source, static metric ansatz. The amplifier focal points do
not follow a moving craft; time-dependent drive, source motion, and a
self-consistent matter evolution are not implemented. The finite-difference
curvature and finite null-direction sample require convergence and
independent-coordinate checks before being used quantitatively.

A sampled negative null contraction demonstrates a local NEC violation
along that sampled null direction. A finite set of non-negative samples does
not prove the NEC globally. Similarly, the Einstein-implied stress-energy is
what the prescribed metric would require under the Einstein equation; it is
not evidence that such matter exists or can be engineered.

## Reproduction

    cargo fmt --check
    cargo test -p symthaea-gravcraft
    cargo test -p symthaea-metric-geometry-lab

## Next scientific gates

1. Compare metric derivatives and curvature against an independent symbolic
   or automatic-differentiation reference over a frozen coordinate corpus.
2. Sweep derivative step sizes; report observed convergence order for
   Christoffel, Ricci, Kretschmann and source diagnostics.
3. Compare the ADM constraints computed from the induced spatial metric and
   extrinsic curvature against the Eulerian projections of the Einstein
   tensor. Record sign and index conventions explicitly.
4. Evaluate NEC/WEC/SEC/DEC over observer and null-direction ensembles with
   angular refinement, not just one finite sample.
5. Propagate the full stress-energy profile and integrate positive/negative
   energy over a declared spatial domain; compare boundary conditions and
   asymptotic falloff.
6. Compare the same initial data and gauge against Einstein Toolkit / SpECTRE
   test problems where applicable. This small reference kernel is not a
   substitute for BSSNOK/Z4c evolution, matter evolution or a numerical
   relativity test suite.
7. Only after a self-consistent source and constraint solution exists should
   any trajectory be described as a solution of a physically sourced metric.

## Scientific non-claims

This crate does not demonstrate:
- warp-drive propulsion, metric engineering, or spacetime manipulation;
- a realizable stress-energy source;
- violation-free energy conditions;
- a dynamically generated or stable bubble;
- any role for IIT Phi or consciousness in gravitational coupling.

The ADM module is a stricter mathematical reference for a prescribed ansatz,
not a feasibility result.
