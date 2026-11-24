# Joint-covariance experiment design v0.2

**Status:** additive Stage-B screening implementation. The current exact-head focused workflow is the qualification authority. This module is not connected to the live Symtropy response atlas, calibrated on physical data, or validated as expected information gain.

Synthetic fixture: [covariance-redundancy-v0.1.json](../fixtures/covariance-redundancy-v0.1.json). Machine-readable contract: [covariance-design.schema.json](../fixtures/covariance-design.schema.json).

## Why v0.2 uses one joint covariance

A first prototype accepted one covariance matrix for every pair of rival hypotheses. This was expressive, but a generic "pairwise covariance completion" check based on double-centering the block matrix of pairwise difference covariances was not valid for arbitrary vector outputs: it implicitly discarded admissible antisymmetric cross-observable blocks and could reject valid joint covariance models.

The contract is therefore strengthened and simplified.

Each candidate experiment supplies one PSD covariance matrix \(\Gamma\) for the complete stacked output vector:

\[
Z=(Y_1^T,Y_2^T,\ldots,Y_H^T)^T,
\]

where hypotheses are sorted lexicographically and observables follow the candidate's declared order. The covariance descriptor must state the joint coupling semantics: how common inputs, model discrepancy, shared calibration, solver/background error, measurement noise and cross-hypothesis dependencies are represented.

The selector validates \(\Gamma\succeq0\) and derives every pairwise difference covariance from its blocks. This makes pairwise coherence structural instead of relying on a separate approximate matrix-completion test.

For hypothesis pair \(i,j\), the derived covariance is:

\[
S_{ij}=\operatorname{Cov}(Y_i-Y_j)
=\Gamma_{ii}+\Gamma_{jj}-\Gamma_{ij}-\Gamma_{ji}.
\]

The cross-hypothesis blocks \(\Gamma_{ij}\) are essential. Silently replacing them with zero is equivalent to asserting independence and can misstate uncertainty when hypotheses share model inputs, calibration, nuisance parameters, background solutions or numerical error.

A joint covariance is a *coupling model*, not something that can always be inferred from marginal variances alone. Where no defensible joint coupling exists, preserve interval/set-valued uncertainty or mark the design UNRESOLVED. Do not invent a convenient cross-covariance merely to satisfy the API.

## Scoring

For the mean difference:

\[
\Delta\mu_{ij}=\mu_i-\mu_j,
\]

the covariance-geometry screening statistic is:

\[
D_{ij}^{2}=\Delta\mu_{ij}^{T}S_{ij}^{\dagger}\Delta\mu_{ij}.
\]

For each candidate design, the selector takes the least-separated rival pair and divides the distance \(D_{ij}\) by caller-defined positive relative cost. The reported score is a prior-free maximin screening criterion. It is **not** Bayesian expected information gain, a Bayes factor, a posterior, an expected log-likelihood ratio, an optimal-design theorem, or a truth oracle.

The Moore–Penrose inverse is used only on positive-eigenvalue modes. The selector separately tests whether the mean difference lies in the numerical support of a singular \(S_{ij}\). A material component in a zero-variance direction quarantines the candidate for support review; the pseudoinverse alone would silently discard that component.

## Validation contract

- Canonical order is explicit: sorted hypothesis IDs, then observable IDs in the experiment declaration order.
- One joint covariance matrix is supplied per experiment, of dimension \((H p)\times(H p)\).
- Hypothesis and observable orders, dimensions, finite values, symmetry, non-negative diagonals, descriptor digest and numeric matrix digest are validated.
- The joint matrix is diagonal-standardized and checked for positive semidefiniteness before any pairwise difference is derived.
- The matrix digest hashes dimensions and row-major IEEE-754 f64 bit patterns. The descriptor digest binds provenance text. Neither digest proves executable execution or physical calibration.
- No diagonal jitter, silent shrinkage or undocumented covariance repair is performed.
- The derived pair covariance \(S_{ij}\) is computed directly from \(\Gamma\); it is not independently supplied, so mutually impossible pairwise covariance families cannot enter through the API.
- Resource guards currently cap 8 hypotheses, 8 observables, 16 stacked dimensions, 256 candidate experiments and 8 MiB of CLI JSON input. These are v0.2 implementation bounds, not theoretical limits.

## Exact synthetic redundancy control

The fixture has two hypotheses. H0 is deterministic and H1 has covariance:

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

The stacked covariance is block diagonal with a zero H0 block and \(\Sigma\) as the H1 block. Hence the derived difference covariance is exactly \(S_{01}=\Sigma\), and:

\[
D^2=\Delta\mu^T\Sigma^{-1}\Delta\mu=4.
\]

Append the exactly redundant output \(x_3=x_1+x_2\). Let

\[
A=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}.
\]

The H1 joint block becomes \(A\Sigma A^T\), with rank two, and the mean difference becomes \(A\Delta\mu=(1,2,3)^T\). The pseudoinverse score remains \(D^2=4\). The diagonal-only independence calculation incorrectly rises from 5 to 8.

This is a linear-algebra control, not an EGB computation, physical covariance estimate, or result from Symtropy.

## Additional hostile controls

- Singular joint covariance with a derived difference mean outside the support: candidate quarantined.
- Singular joint covariance with a derived difference mean inside the support: candidate scored on the supported subspace.
- Non-PSD joint matrix with positive pairwise difference variances: rejected at the joint input boundary.
- Wrong hypothesis/observable order, wrong matrix dimensions, non-finite values, stale descriptor or matrix digest: fail closed.
- Three-hypothesis joint models: all pairwise difference covariances are derived from the same \(\Gamma\).
- Positive diagonal unit rescaling and well-conditioned invertible observable-basis changes: transform means and \(\Gamma\) together; the quadratic form is invariant within numerical tolerance.

## Run locally

    cargo test --locked -p symthaea-experiment-design
    cargo run --locked -p symthaea-experiment-design -- --covariance-demo
    cargo run --locked -p symthaea-experiment-design -- --covariance < crates/domains/symthaea-experiment-design/fixtures/covariance-redundancy-v0.1.json

The last command exercises the public v0.2 JSON input contract. Replace the fixture with a campaign-specific input only after its joint coupling and provenance are justified.

## Remaining qualifications before physical use

1. Compare the Jacobi eigensolver against an independent LAPACK/high-precision reference over SPD, PSD, nearly singular, badly conditioned and indefinite matrices.
2. Expand metamorphic tests for general invertible observable transforms and Symtropy endpoint Gram-metric transports.
3. Add covariance uncertainty sets and interval/set-valued design for non-probabilistic truncation/discretization/model-form uncertainty.
4. Add an observable lineage DAG so determinant, singular values, condition number and other derived response diagnostics inherit their ancestry rather than being declared as new independent variables. This is owned by [#13472](https://github.com/Luminous-Dynamics/symthaea/issues/13472).
5. Add the typed Symtropy [#7131](https://github.com/Luminous-Dynamics/symtropy/issues/7131) adapter carrying response matrix, endpoint basis identities, Gram metrics, chart identity, normalization, solver provenance and shared error sources.
6. Compare with Monte Carlo EIG only where predictive distributions and priors are independently justified. Otherwise retain prior-free screening or robust/interval design.

## Claim ceiling

The v0.2 contract establishes that one caller-supplied joint covariance matrix is PSD under the declared numerical tolerance and that the pairwise difference covariances are derived consistently from it. It does not establish that the covariance is physically calibrated, that the coupling model is scientifically justified, that the experiment is worthwhile at Stage A, or that any EGB branch/event/theorem has been established.
