# Physical Computing Metamorphic Benchmark v1

Research owner: [PHYS-COMPUTE-META-001 #11208](https://github.com/Luminous-Dynamics/symthaea/issues/11208)

Related work:
- [Physical computing compiler #11116](https://github.com/Luminous-Dynamics/symthaea/issues/11116)
- [Flexible physical preconditioners #11152](https://github.com/Luminous-Dynamics/symthaea/issues/11152)
- [Uncertainty-aware routing #11170](https://github.com/Luminous-Dynamics/symthaea/issues/11170)
- [Physical optimization #11172](https://github.com/Luminous-Dynamics/symthaea/issues/11172)
- [LP certificate checker #11174](https://github.com/Luminous-Dynamics/symthaea/pull/11174)

## 1. Purpose

A static benchmark asks whether a solver succeeds on a list of instances. A metamorphic benchmark asks whether the compiler, solver and certificate transport preserve the same mathematical result across *equivalent representations* of each instance.

This is a high-value stress test for scientific-computing compilers because a transformation can be algebraically valid while exposing numerical scaling, sign, ordering, presolve, sparse-storage or certificate-transport defects.

The SCS 3.3.1 public benchmark explicitly recomputes primal residual, dual residual and duality gap against original problem data. It also documents false convergence on badly scaled LPs where internal scaled residuals fail to track the unscaled residuals. The benchmark therefore motivates original-model metamorphic qualification:
https://www.cvxgrp.org/scs/benchmarks/index.html

## 2. Canonical problem and evidence classes

The first track uses standard-form LP:

\[
\min_x c^T x \quad \text{subject to } Ax=b,\;x\geq0.
\]

Every original problem is immutable and identified by a canonical digest over dimensions, coefficient encoding, A, b, c, objective sense, variable bounds and declared tolerances.

Keep separate evidence classes:
- **Exact binary64 dyadic:** every coefficient is the exact finite binary64 value supplied to the verifier; exact sign/equality checks can apply.
- **Rational source:** coefficients and certificates are exact rationals, represented canonically as numerator/denominator pairs.
- **Tolerance-based numerical:** floating-point residuals are checked under declared componentwise backward-error tolerances.
- **Synthetic/modelled:** transformations or perturbations are generated for stress tests and are not hardware measurements.
- **Hardware-measured:** only admitted when device identity, raw observations, calibration and complete measurement boundary are present.

Never promote one class into another silently.

## 3. Transformations and certificate maps

Each transformation must carry a typed forward map, inverse map, expected objective relation, proof obligation and transform digest.

### 3.1 Row permutation

For a permutation matrix P:

\[
A'=PA,\qquad b'=Pb,\qquad c'=c.
\]

A transformed dual vector y' maps to the original as:

\[
y=P^T y'.
\]

The primal vector is unchanged.

### 3.2 Nonzero row scaling

For nonsingular diagonal D:

\[
A'=DA,\qquad b'=Db.
\]

This includes negative row scales. The transformed equality-dual vector y' maps to:

\[
y=D^T y'.
\]

Do not replace D with |D| in the dual map. Sign mistakes must fail negative controls.

### 3.3 Column permutation

For a permutation matrix Q and x=Qz:

\[
A'=AQ,\qquad c'=Q^Tc.
\]

The transformed primal z maps back as x=Qz. Equality dual variables are unchanged.

### 3.4 Positive variable-unit scaling

For positive diagonal D and x=Dz:

\[
A'=AD,\qquad c'=D^Tc.
\]

The transformed variable z must map back as x=Dz. The equality dual vector is unchanged. Include scales spanning multiple powers of two and mixed units, subject to finite representability.

### 3.5 Positive objective scaling

For gamma>0:

\[
c'=\gamma c.
\]

The minimizers and feasible region are unchanged, the optimal objective scales by gamma, and the dual vector transforms as y'=gamma y.

A negative gamma is not an invariance of minimization; it reverses the optimization direction and is a required negative control.

### 3.6 Redundant equality insertion

For a declared coefficient vector r, append:

\[
a_{m+1}^T=r^TA,\qquad b_{m+1}=r^Tb.
\]

The row is redundant only when this relation is established for the exact source data. A certificate can be extended with a zero multiplier for the new row. Also test dependent rows that are not redundant because the appended right-hand side is inconsistent; these must change the feasibility disposition.

### 3.7 Sparse storage permutation

Reorder COO entries, CSR column indices within rows, CSC row indices within columns, and equivalent block-storage representations without changing the mathematical matrix. Compare operator and certificate results, not serialization order.

### 3.8 Exact power-of-two scaling

Use powers of two to vary magnitude while retaining exact binary64 representability, and restrict exponents so no coefficient overflows or underflows. These controls isolate algorithmic scaling sensitivity from decimal-to-binary input rounding.

### 3.9 Composed transforms

Hold out compositions of row permutations, signed row scales, column permutations, positive variable scales, objective scales, redundant rows and sparse-storage reorderings. A compiler that passes each individual transform but fails a composition is not qualified for formulation discovery.

## 4. Invariants to test

For each positive transformation:
1. Feasibility/infeasibility/unboundedness disposition is preserved.
2. Primal variables map back to the canonical original feasible region.
3. Dual and Farkas certificates map back with the declared sign and scale conventions.
4. The optimal objective changes only by the declared positive objective scale.
5. The canonical original-model verifier returns the expected disposition.
6. Exact-dyadic checks and tolerance checks are reported separately.
7. Solver status strings, internal residuals, original-model residuals and certificate results remain distinct.
8. The compiler's mathematical eligibility decision is unchanged when the underlying problem semantics are unchanged.

Do not require bitwise-identical numerical iterates. Require invariant mathematical disposition and declared error bounds.

## 5. Required negative controls

- Apply the wrong sign to a negative row scaling.
- Scale A but not b, or b but not A.
- Permute columns but not c or the primal vector.
- Omit D when mapping z back to x.
- Transform y with D instead of D^{-T} in the forward direction, or apply the wrong inverse map.
- Claim minimizer invariance after multiplying c by a negative scalar.
- Append a row that is not a proven linear combination of existing rows.
- Append a dependent row with an inconsistent right-hand side.
- Verify only the transformed model and skip the canonical original model.
- Reuse a stale or mismatched transform digest.
- Provide duplicate transform IDs or non-invertible maps where invertibility is required.
- Use a large unrelated row/column to hide a small feasibility violation.
- Include near-degenerate instances where the active set changes but the mathematical result does not.

## 6. Benchmark tracks

### Track A — finite exact fixtures

Hand-computable optimal, infeasible and unbounded LPs with binary64-exact coefficients/certificates. Apply each transform and verify both transformed and canonical forms.

### Track B — mixed-scale stress corpus

Generate matrices with controlled row/column dynamic range, small nonzero residuals, near-zero Farkas margins, cancellation-heavy dot products and mixed-unit variables. Pair each transformed instance with its original.

### Track C — solver parity

Run HiGHS, SCS 3.3.1, Clarabel, PIQP and OR-Tools PDLP where the canonicalized problem class is supported. Pin versions, tolerances, presolve/scaling modes, hardware, memory and time limits. Record unsupported cases rather than silently dropping them.

### Track D — physical/emulated execution

Run digital baselines and synthetic physical-emulator plans on the same semantic transform families. A physical plan must return through the canonical original-model verifier. No synthetic profile may be labelled a qualified physical backend.

### Track E — compiler strategy invariance

Compare deterministic plan selection before and after semantics-preserving transforms. If execution costs change because of sparsity/layout or conditioning, the selected execution strategy may legitimately change; mathematical eligibility and accepted-result semantics must not.

## 7. Metrics

Report separately:
- positive-transform disposition agreement;
- exact certificate pass/fail and tolerance-based pass/fail;
- certificate-transport success by transformation family;
- original-model primal/dual residuals and objective gap;
- solver false-convergence and false-certificate counts;
- routing decision flip rate for eligibility and mathematical method;
- route changes caused by legitimate cost/layout differences;
- p50/p95/p99 solve and verification cost across equivalent transforms;
- worst-stratum degradation;
- total cost per independently verified accepted result;
- held-out composition generalization.

Correctness invariance is a hard gate. Runtime invariance is an empirical objective, not a theorem: equivalent formulations can have different sparsity, conditioning and computational cost.

## 8. Reproducibility protocol

For every case, archive:
- original problem digest and canonical coefficient encoding;
- transformation sequence, parameters, forward/inverse maps and digest;
- solver/backend version and full configuration;
- random seed and generator version;
- transformed input digest;
- transformed output and transported certificate;
- canonical original-model verification receipt;
- evidence class and measurement boundary;
- exact disposition and failure reason.

Split by original problem family—not merely by transformed instance—so near-duplicate transforms do not leak across development and confirmatory sets. Freeze policies before confirmatory evaluation.

## 9. Proposed gates

Targets, not achievements:
- 100% expected disposition agreement on the finite exact corpus.
- Zero accepted certificates that fail exact checking when exact-dyadic inputs and certificates are applicable.
- Zero accepted outputs that violate the canonical original-model numerical contract.
- 100% correct transform-map behavior on enumerated positive and negative controls.
- 100% immutable linkage between original model, transform, output and receipt.
- At least one held-out composed-transform family passed without policy retuning.
- Publish every transform family on which any solver or compiler plan loses.

No universal runtime-variance target is set until the workload family, machine and measurement boundary are frozen.

## 10. Claim ceiling

Passing this suite qualifies the declared finite transformation families and verifier contracts. It does not prove every possible compiler transformation semantics-preserving, guarantee robust behavior on all LPs, qualify physical hardware, or establish universal superiority over digital solvers.
