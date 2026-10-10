# Orbital-ring control-volume note — steady inertial circular-stream limit

**Status:** T0 analytical limiting case only; not an instantiated orbital-ring design.  
**Purpose:** provide a small independently checkable reference case before defining a moving-support rotor model.

## 1. Exact scope of this case

The example is a mass stream following a **fixed circular path in an inertial frame**. The stream has constant line density \(\lambda\), constant inertial speed \(v\), and path radius \(R\). Its speed magnitude is constant, so there is no tangential acceleration. The calculation asks for the **net normal resultant per unit path length** required to bend the stream's trajectory.

This is not a model of a ring held stationary over Earth, a launch loop, or an electromagnetic bearing. It does not attribute the required resultant to one force source.

## 2. Derivation

Let \(\mathbf{t}\) be the unit tangent, \(\mathbf{n}\) the principal normal, and \(\kappa=1/R\) the curvature of the fixed circular path. By the Frenet relation,

\[
\frac{d\mathbf{t}}{ds} = \kappa\mathbf{n},
\]

and along the stream \(ds/dt=v\). With constant speed, \(\mathbf{v}=v\mathbf{t}\), so

\[
\mathbf{a}
= \frac{d(v\mathbf{t})}{dt}
= v\frac{d\mathbf{t}}{dt}
= v\frac{d\mathbf{t}}{ds}\frac{ds}{dt}
= v^2\kappa\mathbf{n}.
\]

A material element of length \(ds\) has mass \(dm=\lambda ds\). Newton's second law therefore gives the net normal force on that element as \(d\mathbf{F}_n=dm\,\mathbf{a}\); dividing by \(ds\) gives the resultant per unit path length:

\[
\boxed{\mathbf{q}_n = \lambda v^2\kappa\mathbf{n}},
\qquad
\boxed{q_n = \lambda\frac{v^2}{R}}.
\]

Here \(q_n\) is a **resultant**, not the electromagnetic, tension, contact, or other individual contribution. In a real supported system, the vector sum of the modeled forces must produce the acceleration; no one mechanism can be assumed to carry the whole load.

### Dimensional check

\[
[\lambda v^2/R]
= (\mathrm{kg}/\mathrm{m})(\mathrm{m}^2/\mathrm{s}^2)/\mathrm{m}
= \mathrm{N}/\mathrm{m}.
\]

### Hand-checkable numerical limit

Take \(\lambda=1\,\mathrm{kg/m}\), \(v=3\,\mathrm{m/s}\) (inertial), and \(R=10\,\mathrm{m}\). Then

\[
q_n = 1\cdot 3^2/10 = 0.9\,\mathrm{N/m}.
\]

The reference checker exercises this case and the following implications: zero speed gives zero normal resultant; doubling speed multiplies the resultant by four; doubling radius halves it; doubling line density doubles it. These are tests of this limiting equation only.

## 3. Why this is not yet the orbital-ring equation

The formula is valid only for the stated fixed-curve, constant-speed, inertial-frame limit. It cannot be fed a support-relative speed as though that were the stream's inertial speed. For a translating frame, velocities must be transformed; for a rotating frame, the transform also depends on the support's angular velocity and the point's position. Relative-frame acceleration can include Coriolis, centrifugal, and angular-acceleration terms. Gravity, support acceleration, tension, electromagnetic/contact forces, payload exchanges, and any mass or momentum crossing a control boundary must be handled consistently.

For a material line element, the local acceleration equation above is appropriate for this restricted limit. For a **fixed or moving spatial control volume** with mass crossing its boundary, use a momentum balance that explicitly includes momentum flux; do not equate that flux-free material-element balance with a general control-volume equation.

The governing question for the next increment is not merely “what is \(v^2/R\)?” It is: **which physical bodies and fluxes are inside the control volume, in which frame is each velocity defined, and which modeled force terms sum to the inertial momentum change?**

## 4. Required fields before a coupled model is implemented

The model's exact scenario manifest must freeze all of these or mark them not applicable with a reason:

- **Architecture:** freely orbiting structure, dynamically supported stationary ring, launch loop, momentum-exchange tether, or accelerator-mounted system.
- **Geometry:** path \(\mathbf{r}(s,t)\), tangent/normal convention, curvature, path time dependence, ring plane/orientation, and segment/control-volume boundaries.
- **Frames:** inertial frame and origin; support origin and pose; support linear and angular velocity/acceleration; a complete transform for every relative velocity or force.
- **Stream:** line density or discrete masses, inertial velocity, support-relative velocity, mass-flow rate, and whether density/speed vary with position or time.
- **Momentum accounting:** all external forces by category; incoming/outgoing mass and momentum fluxes; contact/tension or actuator reactions; payload exchange terms.
- **Energy accounting:** inertial kinetic energy for each included mass, potential-energy convention, support work, actuator/electrical power, losses, and boundary energy fluxes.
- **Evidence:** canonical SI inputs, assumptions and units, versioned scenario manifest, residual tolerances, independent derivation, negative tests, deterministic output, and a claim no stronger than the evidence supports.

Fail closed if the chosen architecture, frame transform, mass representation, or control-volume boundary remains ambiguous. A small residual is not, by itself, evidence of local stability; that requires a separately specified equilibrium and perturbation/stability analysis.

## 5. Current evidence state

- The fixed-circle limit is algebraically derived above and numerically checked in scripts/engineering/orbital_ring_reference_check.py.
- The checker is a distinct Python implementation of the reference equations; it has not been compared against GMAT, Orekit, Basilisk, experimental data, or an independent human derivation.
- No moving-support, Earth-gravity-plus-support, electromagnetic, structural, thermal, debris, fault-response, stability, cost, or feasibility conclusion follows from this note.
- Rust compilation, Rust unit tests, formatting, and Clippy remain a separate outstanding verification gate. This does not prevent work on a clearer physical specification.
