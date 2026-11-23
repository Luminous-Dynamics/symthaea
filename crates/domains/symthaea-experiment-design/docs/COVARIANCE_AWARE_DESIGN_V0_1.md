# Covariance-aware experiment design v0.1

**Status:** research implementation slice and synthetic regression controls. Not yet Rust/CI-qualified, connected to the live Symtropy response atlas, calibrated on physical data, or validated as expected information gain.

## Why this extension exists

The v0.1 scalar planner in this crate computes standardized distances independently for every observable and adds their squares. That calculation is only justified when the declared observable channels are conditionally independent under the adopted uncertainty model.

For a response matrix, this is generally false. Matrix entries, determinant, singular values, condition number, basis-transformed quantities, and residual summaries can all be functions of the same small set of numerical values. Treating all of them as independent measurements can make the same information appear multiple times.

This module adds a vector-output contract and consumes a complete covariance matrix for each unordered rival-hypothesis pair. It is an additive extension of the existing package, not another selector service or scheduler.

## Mathematical contract

For hypothesis pair i,j and design d, let the frozen observable vector be Y(d). Define:

\[
\Delta\mu_{ij}(d)=\mu_i(d)-\mu_j(d).
\]

The input supplies the full covariance of the difference vector:

\[
S_{ij}(d)=\operatorname{Cov}\big(Y_i(d)-Y_j(d)\big).
\]

Where this is derived from marginal covariance matrices, the cross-hypothesis terms must be retained:

\[
S_{ij}=\Sigma_i+\Sigma_j-C_{ij}-C_{ij}^{T},
\qquad C_{ij}=\operatorname{Cov}(Y_i,Y_j).
\]

The covariance descriptor must state the adopted coupling/uncertainty semantics and identify which shared errors, numerical errors, calibration terms, and measurement terms are included. There is no universal way to infer cross-hypothesis covariance from two marginal standard deviations. The selector therefore consumes the difference covariance directly instead of guessing it.

For a positive-definite covariance, the screening statistic is:

\[
D_{ij}^{2}(d)=\Delta\mu_{ij}(d)^T S_{ij}(d)^{-1}\Delta\mu_{ij}(d).
\]

For singular positive-semidefinite covariance, the module computes the Moore–Penrose quadratic form only on positive-eigenvalue modes, while separately testing whether the mean difference lies in the covariance support. If a material component lies in a zero-variance direction, the candidate is **quarantined for support review** and omitted from automatic ranking. This avoids a dangerous property of the pseudoinverse alone: it can silently discard a mean difference in the null space and return a finite score.

For eligible candidates, the design score is the minimum D_ij across all rival pairs, divided by the caller's frozen relative cost units. Cost is not inferred from covariance or model confidence.

This is a prior-free covariance-geometry screen. It is **not** Bayesian expected information gain, a likelihood ratio, a Bayes factor, a posterior, or a truth oracle.

## Numerical treatment and limitations

- Covariance matrices are checked for dimensions, finiteness, symmetry, non-negative diagonal entries, positive semidefiniteness, and exact pair coverage.
- A descriptor SHA-256 binds the supplied provenance text. A separate canonical matrix digest binds the numeric matrix dimensions and row-major IEEE-754 values. Neither digest proves that an executable generated the data.
- Covariance eigendecomposition is applied to a diagonal-standardized correlation matrix. This makes numerical rank/support cutoffs invariant to positive diagonal unit changes (for example, metres to millimetres). The algorithm and tolerances are versioned in the output.
- No diagonal jitter, automatic shrinkage, or undocumented regularization is applied.
- The Jacobi eigensolver and tolerances are a deterministic v0.1 numerical implementation. They require independent comparison with a trusted linear-algebra implementation, adversarial conditioning tests, dimension-scaling tests, and exact-head Rust/CI qualification before production use.
- Covariance uncertainty sets, nuisance-parameter profiling, non-Gaussian predictive distributions, and calibrated Monte Carlo EIG are not implemented.
- The current schema does not yet encode the full basis/endpoint Gram-metric contract required by Symtropy #7131. It must not be used to make EGB response-atlas claims until an adapter carries that exact basis identity and metric transport semantics.

## Exact synthetic known-answer control

The fixture at fixtures/covariance-redundancy-v0.1.json is purely synthetic.

\[
\Sigma=
\begin{pmatrix}
1 & 1/2\\
1/2 & 1
\end{pmatrix},
\qquad
\Delta\mu=
\begin{pmatrix}
1\\2
\end{pmatrix}.
\]

The inverse and quadratic form are:

\[
\Sigma^{-1}=
\begin{pmatrix}
4/3 & -2/3\\
-2/3 & 4/3
\end{pmatrix},
\qquad
\Sigma^{-1}\Delta\mu=
\begin{pmatrix}
0\\2
\end{pmatrix},
\qquad D^2=4.
\]

Append x3=x1+x2. The augmented difference is (1,2,3)^T and its covariance is AΣA^T, where:

\[
A=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}.
\]

The augmented covariance has rank two. Its covariance-aware pseudoinverse score remains D²=4. The diagonal-only independence calculation instead changes from 5 to 8, falsely rewarding the duplicate.

A second hostile test uses S=[[1,1],[1,1]] and Δμ=(1,0)^T. Because the mean difference has a component outside range(S), the candidate must be quarantined rather than assigned the finite value returned by the pseudoinverse quadratic form alone.

These tests are linear-algebra controls, not EGB calculations, not physical covariance estimates, and not validation of the scalar #8034 planner.

## Run locally

    cargo test --locked -p symthaea-experiment-design
    cargo run --locked -p symthaea-experiment-design -- --covariance-demo

The second command emits the base and exactly-redundant candidates with the same D²=4 score and an empty support-review list.

## Required next qualifications

1. Compare the Jacobi result against an independent LAPACK/BLAS or high-precision reference over randomized SPD, PSD, nearly singular, badly conditioned, and indefinite matrices.
2. Add metamorphic tests for arbitrary invertible linear basis changes A, verifying AΔμ, ASA^T, and invariant quadratic form when numerically well-conditioned.
3. Add interval/uncertainty-set covariance bounds. A single point covariance is not robust to covariance-estimation uncertainty.
4. Add a predictive calibration campaign: coverage, standardized residuals, and covariance calibration on held-out synthetic and independently solved controls.
5. Add a Symtropy #7131 adapter that carries the response operator, endpoint bases and Gram metrics, chart identity, units, and uncertainty provenance without reconstructing or duplicating the physical response model.
6. Compare with Monte Carlo expected information gain only on tasks where predictive distributions and priors are independently justified. Keep the prior-free screen as a distinct method.
7. Freeze budgets and episodes before comparing against fixed, random, scalar-independence, covariance-aware, and exact tiny-world oracle arms.

## Acceptance and claim ceiling

Before merging, require exact-head formatting, locked tests, Clippy, the covariance demo, JSON contract tests, and the existing planner's regression suite. Preserve all failures and denominators.

The present branch proves no performance gain, no SOTA result, no live Symtropy integration, no finite-coupling EGB solution, no branch event, no theorem, and no physical measurement claim. It only adds a testable covariance-aware candidate method and synthetic negative controls.
