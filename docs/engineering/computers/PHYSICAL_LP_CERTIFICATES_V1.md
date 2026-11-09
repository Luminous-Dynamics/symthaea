# Physical LP Certificate Checker v1

Research issue: [PHYS-COMPUTE-OPT-001 #11172](https://github.com/Luminous-Dynamics/symthaea/issues/11172)

Source reference: research/physical_compute/lp_certificate_v1.rs

## Scope

The reference checker accepts certificates for the standard-form linear program

\[
\min_x c^T x \quad \text{subject to } Ax=b,\;x\geq 0.
\]

It checks three distinct claims against the original caller-supplied model:

1. **Optimality:** a primal point x and unrestricted equality-dual vector y satisfying primal feasibility, dual feasibility, small primal/dual objective gap and complementary slackness within explicit tolerances.
2. **Infeasibility:** a Farkas vector y with \(A^Ty\geq 0\) and \(b^Ty<0\), using a normalized strict-contradiction margin.
3. **Unboundedness:** a feasible base point x0 and improving ray d with \(Ad=0\), \(d\geq0\) and \(c^Td<0\).

The dual convention for this minimization form is

\[
\max_y b^T y \quad \text{subject to } A^T y\leq c,
\]

so the dual slack checked by the implementation is \(s=c-A^Ty\geq0\).

## Trust and numerical boundary

- The checker consumes only the original A, b, c and the candidate certificate. It does not accept a solver status string as evidence.
- It has no access to evaluator-only labels or known optimal solutions.
- It rejects dimension mismatches, non-finite inputs, invalid tolerances and detected non-finite arithmetic.
- Dot products use compensated summation.
- Primal equality residuals are checked with rowwise componentwise backward-error scales; dual-feasibility and Farkas-cone violations use columnwise componentwise scales. Unbounded base-point and nullspace residuals are checked rowwise. A large row or column cannot hide a violation in a small one.
- Nonnegativity, objective gap and complementarity use dimensionless relative scales. Denominators are not floored at 1, which would silently turn a uniformly tiny inconsistent model into an absolute-tolerance pass.
- Homogeneous Farkas multipliers and unbounded rays are normalized before checking, so positive rescaling of a valid certificate cannot change its disposition.
- This is a **floating-point numerical checker**, not an exact-arithmetic theorem prover. A successful result means the declared numerical certificate passed the declared tolerance contract. It must not be described as a formal proof over real arithmetic.
- It does not yet support inequality constraints, variable bounds, presolve transformations, integer variables, exact rational certificates or interval-arithmetic proof.

The verifier must always be applied to the canonical original model. If the solver uses scaling, presolve, row/column elimination or a transformed model, the candidate certificate must first be mapped back and then checked against the original data.

## Standalone test command

When a Rust toolchain is available, run:

    rustc --edition=2021 --test research/physical_compute/lp_certificate_v1.rs -o /tmp/lp_certificate_v1_tests
    /tmp/lp_certificate_v1_tests

The source currently authors 21 positive/negative unit tests covering:
- a known primal/dual optimality pair;
- primal infeasibility and dual infeasibility;
- excessive duality gap;
- valid and invalid Farkas certificates;
- valid and invalid improving rays;
- dimension mismatch;
- non-finite model input;
- invalid, non-finite and greater-than-one tolerance configurations;
- positive scale invariance of Farkas vectors and unbounded rays across very small and very large magnitudes;
- rejection of a tiny-scaled vector that is not actually in the null space;
- rejection of an infeasible certificate on a uniformly tiny LP;
- mixed-scale negative controls where large rows/columns would otherwise hide small primal, dual, Farkas, ray or base-point violations.

**Execution status:** tests are authored but have not been compiled or run in the current authoring environment. No CI, repository or hardware qualification is implied.

## Next qualification work

1. Compile and run the standalone tests on the exact PR head.
2. Add adversarial scale tests, nearly strict Farkas margins, cancellation-heavy dot products and overflow boundaries.
3. Cross-check against an independent exact-rational enumeration/checker on small integer/rational fixtures.
4. Add row/column scaling and certificate unscaling tests.
5. Extend to inequality/bound forms by translating certificates back to the canonical original problem.
6. Add interval or rational verification if a formal mathematical certificate is required.
7. Keep optimal, infeasible, unbounded, feasible-but-not-certified-optimal and unknown outcomes distinct in the execution receipt.
