# AIRROAD Structured MIMO Generator Support — exact lower-dimensional simplex reference

**Status:** V0.1 exact-rational support proof and outer-cover fixture  
**Tracking issue:** [#12054](https://github.com/Luminous-Dynamics/symthaea/issues/12054)  
**Composes with:** #11926/#11928 shared-coordinate generator adapter, #11970 sampled-FRF IQC transcription, #11948 solver backend, #11946 matched benchmark

> A lower-dimensional generator cover is valid only for a lower-dimensional admitted set whose support is established by hard structural constraints. Sample covariance rank, a low-rank fit or an SVD threshold cannot establish that structural fact.

## 1. Baseline and objective

The sampled-FRF IQC paper by Spin, Heemels, van de Wouw and Toth (2026-10-14), [arXiv:2610.20248](https://arxiv.org/abs/2610.20248), constructs a simplex outer approximation for the complex matrix unit ball. The real dimension of \(\mathbb C^{n\times m}\) is \(d=2nm\), and the full-dimensional simplex baseline uses \(d+1\) vertices.

AIRROAD keeps that full-dimensional baseline. This reference adds one mathematically distinct option: prove that the *admitted* uncertainty matrices lie in a strict real support subspace, and construct a valid outer cover in that support. It does not claim fewer vertices cover the full ambient ball.

## 2. Frozen support proof

For a complex \(2\times2\) matrix, freeze real vectorization:
\[
\operatorname{vec}_{\mathbb R}(\Delta)=
[\Re\Delta_{00},\Re\Delta_{01},\Re\Delta_{10},\Re\Delta_{11},
\Im\Delta_{00},\Im\Delta_{01},\Im\Delta_{10},\Im\Delta_{11}]^T.
\]
Thus \(d=8\).

The synthetic structural contract declares that every admissible uncertainty has the form:
\[
\Delta(\alpha,\beta)=
\begin{bmatrix}\alpha&0\\0&\beta\end{bmatrix},
\qquad \alpha^2+\beta^2\leq1.
\]
Equivalently, in the frozen real coordinates:
\[
x=\alpha e_1+\beta e_4.
\]

The two real Frobenius-orthonormal basis vectors are \(B_1=E_{00}\) and \(B_2=E_{11}\). Their exact Gram matrix is \(I_2\). The six annihilator constraints set both off-diagonal real entries and all four imaginary entries to zero. Because the declared source class defines **every** member as this diagonal-real matrix family, the entire set—not merely a set of observed samples—is supported on a two-dimensional subspace.

The support certificate must bind to a structural source identity and the full vectorization order. If an admitted model can produce a nonzero off-diagonal or imaginary component, this proof fails and AIRROAD must fall back to the full ambient representation.

## 3. Rational outer simplex in support coordinates

Write support coordinates \(q=(\alpha,\beta)^T\). Use the half-spaces:
\[
n_i^Tq\leq1,
\]
where
\[
n_1=(1,0),\quad n_2=(-3/5,4/5),\quad n_3=(-3/5,-4/5).
\]
Every normal has exact squared Euclidean norm one.

The pairwise intersections give three rational vertices:
\[
v_1=(1,2),\quad v_2=(1,-2),\quad v_3=(-5/3,0).
\]
Each vertex satisfies all three half-space inequalities and lies on two of the boundary lines. The normals positively span \(\mathbb R^2\), so the intersection is the bounded triangle with those vertices.

For any \(q\) in the unit disk, Cauchy–Schwarz gives \(n_i^Tq\leq\|n_i\|_2\|q\|_2\leq1\). Hence:
\[
\{q:\|q\|_2\leq1\}\subseteq
\operatorname{co}\{(1,2),(1,-2),(-5/3,0)\}.
\]
This is a rational, non-regular circumscribed triangle. Its vertex norms are \(\sqrt5,\sqrt5,5/3\), so the maximum vertex norm is \(\sqrt5\). That is a simple conservatism diagnostic, not a claim of minimum-volume optimality.

The ambient full-dimensional baseline remains \(d+1=9\) vertices for the 8D complex \(2\times2\) ball. The structured three-vertex cover encloses only the 2D disk in the proved support subspace; it does **not** enclose the full 8D ambient ball.

## 4. Exact verifier contract

The Fraction-only oracle:
- checks the 8D ambient coordinate convention;
- builds the 8x2 support basis and proves exact rank 2;
- checks the basis Gram matrix is exactly \(I_2\);
- verifies all six hard annihilator constraints for every basis generator;
- checks the structural source says all admissible matrices have this form;
- derives the three rational vertices by pairwise line intersections;
- checks every normal norm, every vertex/half-space inequality and boundedness;
- verifies the exact Cauchy–Schwarz containment argument;
- demonstrates an allowed off-diagonal/imaginary perturbation invalidates the support declaration;
- rejects rank reduction based only on covariance/sample rank;
- rejects mismatched calibration coordinate identities;
- reports ambient dimension, support dimension, generator count and claim scope separately.

## 5. Claim ceiling

This is exact conformance for one synthetic hard structural support and one rational outer simplex. It does not prove any physical AIRROAD plant has this support, infer hard constraints from statistical covariance, implement the paper's complete \(\Gamma_{M,k}^{\pm}\) generator product construction, synthesize a controller, or establish robust stability/airworthiness.
