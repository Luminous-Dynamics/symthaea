# AIRROAD IQC Bilinear Generator Outer Product — exact Equation (65) conformance

**Status:** V0.1 exact scalar and noncommuting 2×2 matrix product reference  
**Tracking issue:** [#12058](https://github.com/Luminous-Dynamics/symthaea/issues/12058)  
**Source:** Spin et al. (2026-10-14), [arXiv:2610.20248](https://arxiv.org/abs/2610.20248), Section 5.3.3, Equations (63)–(67)

> The product-generator hull is an outer approximation of every bilinear product of the two interval-interpolated sets. It is not an inner approximation and it must preserve the matrix multiplication order and one-sided endpoint identities.

## 1. Source equation

For two piecewise-convex sets \(\mathscr C(\widehat A)(\omega)\) and \(\mathscr C(\widehat B)(\omega)\), define:
\[
\mathcal C(\omega)=
\{A_\omega B_\omega:
 A_\omega\in\mathscr C(\widehat A)(\omega),\
 B_\omega\in\mathscr C(\widehat B)(\omega)\}.
\]
The paper defines the interval product generator set:
\[
\Gamma_{A,B,k|k+1}=
\{\gamma_A\gamma_B:
\gamma_A\in\Gamma_{A,k}^{+}\cup\Gamma_{A,k+1}^{-},\
\gamma_B\in\Gamma_{B,k}^{+}\cup\Gamma_{B,k+1}^{-}\}.
\]
Every Cartesian pair is included, in the prescribed order \(A B\). The product family is enclosed by the convex hull of those generators, with:
\[
\Gamma_{C,k|k+1}=\Gamma_{C,k}^{+}
=\Gamma_{C,k+1}^{-}.
\]
Thus each interval owns a product generator set and populates the right-sided endpoint of its left knot and the left-sided endpoint of its right knot. The two sides at an interior knot may differ.

Primary-source passage: Section 5.3.3, Equations (63)–(67). [Read the sampled-FRF IQC paper](https://arxiv.org/abs/2610.20248).

## 2. Why all four endpoint products matter

For scalar or matrix endpoints:
\[
A(\lambda)=(1-\lambda)A_0+\lambda A_1,\qquad
B(\lambda)=(1-\lambda)B_0+\lambda B_1,
\]
expansion gives:
\[
\begin{aligned}
A(\lambda)B(\lambda)={}&
(1-\lambda)^2 A_0B_0+
\lambda(1-\lambda) A_0B_1\\
&+\lambda(1-\lambda) A_1B_0+
\lambda^2 A_1B_1.
\end{aligned}
\]
For all \(\lambda\in[0,1]\), the four coefficients are nonnegative and sum to one. Therefore the product is a convex combination of the four ordered endpoint products. For convex endpoint generator sets, the same argument multiplies the endpoint convex weights and interpolation weights.

Pairing only corresponding endpoints omits the two cross products and is not generally an outer cover.

## 3. Scalar exact counterexample

For interval \(I_0=[0,1]\):
\[
A_0=1,\ A_1=3,\qquad B_0=3,\ B_1=1.
\]
The full Cartesian endpoint-product multiset is:
\[
\{A_0B_0,A_0B_1,A_1B_0,A_1B_1\}
=\{3,1,9,3\},
\]
whose convex hull is \([1,9]\).

The aligned-only comparator retains \(\{A_0B_0,A_1B_1\}=\{3\}\). At \(\lambda=1/2\), \(A=B=2\), so the true product is \(4\): outside the aligned-only set, inside the full outer hull.

The exact polynomial identity is:
\[
(1-\lambda)^2 3+
\lambda(1-\lambda)1+
\lambda(1-\lambda)9+
\lambda^2 3
=(1+2\lambda)(3-2\lambda).
\]
This is checked coefficient-by-coefficient with Fraction arithmetic, not by sampling a grid.

## 4. Noncommuting 2×2 matrix case

Freeze:
\[
A_0=I,\quad A_1=
\begin{bmatrix}1&1\\0&1\end{bmatrix},\qquad
B_0=\begin{bmatrix}1&0\\1&1\end{bmatrix},\quad
B_1=\begin{bmatrix}2&0\\0&1\end{bmatrix}.
\]
At \(\lambda=1/2\):
\[
A(1/2)=\begin{bmatrix}1&1/2\\0&1\end{bmatrix},\quad
B(1/2)=\begin{bmatrix}3/2&0\\1/2&1\end{bmatrix},
\]
and:
\[
A(1/2)B(1/2)=
\begin{bmatrix}7/4&1/2\\1/2&1\end{bmatrix}.
\]
The four source-prescribed products \(A_iB_j\) are:
\[
A_0B_0=\begin{bmatrix}1&0\\1&1\end{bmatrix},\
A_0B_1=\begin{bmatrix}2&0\\0&1\end{bmatrix},
\]
\[
A_1B_0=\begin{bmatrix}2&1\\1&1\end{bmatrix},\
A_1B_1=\begin{bmatrix}2&1\\0&1\end{bmatrix}.
\]
Their equal-weight average is exactly the midpoint product above. The reverse product is not interchangeable:
\[
A_1B_0=
\begin{bmatrix}2&1\\1&1\end{bmatrix}
\ne
B_0A_1=
\begin{bmatrix}1&1\\1&2\end{bmatrix}.
\]
The oracle preserves the source order \(A B\).

## 5. One-sided shared-knot fixture

At shared knot \(\omega_1=1\):
- left interval \(I_0=[0,1]\) uses \(A_1^-=\{3\}\), \(B_1^-=\{1\}\), so its product-side generator set is \(\Gamma_{C,1}^-=\{1,3,9\}\);
- right interval \(I_1=[1,2]\) uses \(A_1^+=\{4\}\), \(B_1^+=\{2\}\), with \(A_2^-=\{2\}\), \(B_2^-=\{4\}\), so \(\Gamma_{C,1}^+=\{4,8,16\}\).

The left and right sets at \(\omega_1\) are intentionally distinct. A verifier that collapses them into a single endpoint set has lost interval identity.

## 6. Exact verifier

The Fraction-only oracle:
- constructs every Cartesian product of the one-sided endpoint unions;
- binds each product witness to (interval ID, A-generator ID, B-generator ID, product order A-then-B);
- calculates exact convex hull endpoints for the scalar known answers;
- verifies the polynomial identity symbolically and convex-combination weights;
- calculates all four matrix products in order and confirms the midpoint matrix is their exact convex combination;
- proves the reverse order changes the noncommuting matrix answer;
- checks the one-sided shared-knot identities remain distinct;
- rejects aligned-only pairing, missing cross-products, duplicate/stale IDs, reversed matrix product order and inner-approximation claims.

## 7. Claim ceiling

V0.1 verifies the finite Cartesian endpoint-product construction on synthetic scalar and noncommuting 2×2 examples. It does not construct the full \(\Gamma_{M,k}^{\pm}\) generalized-plant generator, solve the finite-dimensional IQC SDP, synthesize a controller, or establish robust stability/airworthiness.
