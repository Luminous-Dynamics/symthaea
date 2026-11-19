# SCI-SYMGRAM-001 — Exact radial-force development case (v0)

**Date:** 2026-11-18  
**Status:** analytic development fixture specification; not yet executed by Symthaea or Symtropy  
**Parent:** Symthaea [#12781](https://github.com/Luminous-Dynamics/symthaea/issues/12781)  
**Protocol:** [SCI-SYMGRAM-001 v0](./SCI_SYMGRAM_001_PROTOCOL_V0_2026-11-18.md)

## 1. Purpose

This case gives the factorial study one small system where:

- the state representation has a nontrivial continuous symmetry;
- the group action and vector-field equivariance can be checked exactly;
- the quotient variables have a closed reduced dynamical system;
- several worlds share a reusable law structure but differ in the radial-force function;
- a false symmetry can be created by a controlled anisotropic perturbation.

It is a **bounded symbolic-mechanics development case**, not a real-world empirical discovery and not an ansatz-free benchmark. Its purpose is to test the quotient/grammar interface before adding more difficult systems.

## 2. Hidden world family

Let \(x=(x_1,x_2)\) be position and \(v=(v_1,v_2)\) velocity in a dimensionless two-dimensional state space. The hidden world is:

\[
\dot{x}=v, \qquad \dot{v}=-g(a)x,
\]
where
\[
a=x\cdot x, \qquad g(a)=k a^m, \qquad k>0, \quad m\in\{0,1,2,\ldots\}.
\]

The held-out parameter is the exponent / radial law, not merely a new random trajectory from the same fitted coefficients.

For integer \(m\geq0\), this is a polynomial vector field. The potential and conserved energy are:

\[
V(a)=\frac{k}{2(m+1)}a^{m+1},
\qquad
H(x,v)=\frac{1}{2}v\cdot v+V(a).
\]

All coefficients and initial conditions in the finite development fixtures should be rational and dimensionless. A future dimensional variant must declare length/time/mass scales and a consistent unit system rather than silently reusing these dimensionless coefficients.

## 3. Exact SO(2) action and equivariance

For \(R\in SO(2)\), act on position and velocity together:

\[
\rho(R)(x,v)=(Rx,Rv).
\]

Because \(R^T R=I\), the radial scalar is unchanged:

\[
(Rx)\cdot(Rx)=x\cdot x=a.
\]

The vector field is therefore equivariant:

\[
F(\rho(R)(x,v))
=(Rv,-g(a)Rx)
=\rho(R)F(x,v).
\]

An exact infinitesimal check uses:

\[
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix},
\qquad A=\operatorname{diag}(J,J),
\]
\[
DF(s)As-AF(s)=0,
\qquad s=(x_1,x_2,v_1,v_2),
\]

as a polynomial identity for integer \(m\), rational \(k\). This is a **vector-field equivariance** claim. It is not inferred from a low sampled residual and does not automatically qualify a particular integrator's discrete step map.

This standalone algebraic check is not a task-preserving quotient certificate. Symthaea #11616 owns full search-semantics qualification, orbit certificates and lifting. Its current executable finite-group slice (#11653) does not yet support this continuous SO(2) action. Therefore this radial case may be used to verify the mathematical dynamics, but arms D–F may not hard-prune with a continuous quotient until that owner independently qualifies the continuous action and complete task semantics.

## 4. Invariant coordinates and reduced dynamics

Use the following SO(2)-invariant quantities:

\[
a=x\cdot x,
\quad b=x\cdot v,
\quad c=v\cdot v,
\quad \ell=x_1v_2-x_2v_1.
\]

They satisfy:

\[
ac-b^2=\ell^2,
\qquad a\geq0,
\qquad c\geq0.
\]

For every nonzero state pair \((x,v)\neq(0,0)\), these invariants identify the SO(2) orbit; the algebraic relation means the four values contain three independent degrees of freedom. The origin \((x,v)=(0,0)\) is fixed by the whole group and is the non-free singular orbit. The slice \(x=0, v\neq0\) is a boundary/control case but is not itself a nontrivial-stabilizer orbit; the implementation should test it rather than incorrectly classify every \(x=0\) state as singular. Retaining signed \(\ell\) matters when the lifted state must distinguish clockwise from counter-clockwise motion; dropping it would silently quotient by reflections as well.

Differentiate the invariants:

\[
\dot a=2b,
\]
\[
\dot b=c-a g(a),
\]
\[
\dot c=-2b g(a),
\]
\[
\dot\ell=0.
\]

For the polynomial family \(g(a)=ka^m\):

\[
\dot a=2b,
\quad
\dot b=c-ka^{m+1},
\quad
\dot c=-2k a^m b,
\quad
\dot\ell=0.
\]

The reduced energy is:

\[
H(a,b,c)=\frac{1}{2}c+\frac{k}{2(m+1)}a^{m+1},
\]

and its derivative is exactly zero under the reduced dynamics:

\[
\dot H
=\frac{1}{2}(-2b g(a))
+\frac{1}{2}g(a)(2b)
=0.
\]

The candidate reusable concept is not a pre-supplied equation string. It is the **radial-force mechanism** \(g(a)\), together with the shared invariant kinematics. The discovery process must recover the structure from policy-visible observations; \(g\), \(k\), \(m\), and the energy expression remain evaluator-only until the candidate is frozen.

## 5. Suggested development partition

A small first fixture can use:
- discovery worlds with \(m=0,1,2\) and multiple positive \(k\);
- a held-out composition / parameter world with \(m=3\);
- distinct initial-condition and trajectory partitions;
- a declared finite expression-search budget shared by all arms.

The bounded development expression language may include rational constants, variables, addition, multiplication and bounded integer powers. Since this language is known to the candidate, this fixture tests **symmetry reduction and compositional grammar recovery within a bounded language**. It must not be presented as Unsatz-style open-ansatz discovery.

No task/seed manifest should be called sealed while its generator and truth fixture are public. Keep this first case source-visible; construct later sealed confirmation families under the existing evaluator owner.

## 6. How the factorial arms apply

| Arm | Candidate representation | Discovery organization |
|---|---|---|
| A | Raw \(x_1,x_2,v_1,v_2\) | Independent fit per world |
| B | Raw \(x_1,x_2,v_1,v_2\) | Pooled fit, no reusable grammar |
| C | Raw \(x_1,x_2,v_1,v_2\) | Shared grammar across worlds |
| D | Qualified \((a,b,c,\ell)\) quotient | Independent fit per world |
| E | Qualified \((a,b,c,\ell)\) quotient | Pooled fit, no reusable grammar |
| F | Qualified \((a,b,c,\ell)\) quotient | Shared grammar across worlds |

The no-quotient arms must receive the same raw measurements used to derive the quotient arm's inputs. Do not give quotient arms additional observations or hidden ground truth. Account separately for the cost of constructing, verifying and lifting the quotient representation.

For this family, the shared reduced structure is:

\[
\dot a=2b,
\qquad
\dot b=c-a g(a),
\qquad
\dot c=-2b g(a),
\qquad
\dot\ell=0.
\]

Only the radial-force function changes between worlds. The study asks whether the quotient representation makes that reusable structure easier to recover and transfer to a held-out \(m\).

## 7. Mandatory negative controls

### 7.1 Broken rotational symmetry

Use an anisotropic oscillator:

\[
\dot x=v,
\qquad
\dot v=-(k_1x_1,k_2x_2),
\qquad k_1\neq k_2.
\]

The continuous SO(2) quotient is invalid. A system that applies the radial quotient must be caught by the exact equivariance verifier or fail the held-out prediction test.

### 7.2 Near symmetry

Set \(k_2=k_1(1+\epsilon)\) and sweep \(\epsilon\) across the sampled acceptance region. A small sampled residual is not an exact SO(2) proof. The candidate may use the signal to prioritize further tests, but it may not remove orientation-dependent candidates.

### 7.3 Measurement-map asymmetry

Keep the physical central-force system but apply an anisotropic observation map \(y=Mx\) with non-orthogonal \(M\). The evaluator must distinguish:
- symmetry of the hidden physical dynamics;
- symmetry of the measurement process;
- whether the published observation channel is invertible / calibrated enough to recover the physical symmetry.

Do not claim physical symmetry from raw sensor invariance or break the physical-law model just because the sensor coordinates are anisotropic.

### 7.4 Orientation and signed-angular-momentum targets

Use two distinct outputs:

- **Signed angular momentum:** a target depending on \(\ell=x_1v_2-x_2v_1\). A quotient retaining only \(a,b,c\) loses its sign and must fail the sufficiency check; retain \(\ell\) or an equivalent orientation-chirality variable.
- **Absolute external-frame direction:** a target such as \(h(x)=x_1\) is not SO(2)-invariant. A quotient that removes global rotation is not admissible for this target unless the external reference frame is included in the state/observation contract and a valid lift is supplied. The quotient must not claim it can reconstruct absolute orientation from invariants alone.

### 7.5 Singular orbit and quotient boundary

Include the fixed origin \(x=v=0\), where the SO(2) stabilizer is the full group, and the boundary slice \(x=0, v\neq0\), where the action is still free. The implementation must distinguish the non-free orbit from a regular boundary case instead of treating every \(x=0\) state as singular.

## 8. Independent qualification

The standalone Rust oracle now has explicit arithmetic bounds (exponent `0..=8`, coupling `1..=1,000,000`) and rejects unsupported inputs before polynomial expansion. It includes exact scalar-output checks: `a`, `b`, `c` and signed `ell` are SO(2)-invariant, while absolute external-frame coordinate `x1` is rejected as non-invariant. This validates one observable-level obligation only; the complete task contract and pruning permission remain owned by #11616. The oracle also evaluates the invariant-map Jacobian rank at selected boundary/stratum points: rank three at \(x=0, v\neq0\), and rank zero at the fixed origin. The Rust API rejects state coordinates outside its declared magnitude bound.

The exact development oracle should independently verify:
1. SO(2) group-generator algebra and the block action on \((x,v)\).
2. Exact vector-field equivariance \(DF(s)As=AF(s)\) for every admitted \(m,k\).
3. The invariant relation \(ac-b^2=\ell^2\).
4. The reduced equations by direct differentiation.
5. Energy conservation by exact symbolic simplification.
6. Failure of rotational equivariance for \(k_1\neq k_2\).
7. Correct classification of the near-symmetry and measurement-map controls.
8. Quotient/lift round-trip semantics on generic and singular strata.

Symtropy should subsequently check analytic trajectories, numerical convergence and discrete-step equivariance for the selected integrator. A continuous-time proof must not be reused as a discrete-time proof. Task-level pruning must separately pass #11616's objective, feasible-domain, boundary/initial-condition, measurement/output and lift obligations.

## 9. Expected outcome and claim ceiling

This fixture can test whether a verified symmetry quotient:
- reduces the number of variables and candidate evaluations;
- exposes common kinematic relations across different radial force laws;
- improves transfer to a held-out exponent;
- preserves signed orientation and singular-state semantics;
- rejects false quotienting when rotational symmetry is broken or only the sensor map changes.

It does **not** establish that Symthaea has discovered a new physical law, that this family represents general mechanics, or that quotienting is universally advantageous. It is one exact, source-visible development case. Performance claims require the complete matched factorial, held-out families and independent reproduction.
