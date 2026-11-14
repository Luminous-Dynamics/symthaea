# AIRROAD IQC Paper Transcription Audit — arXiv:2610.20248 v0.1

**Status:** primary-source equation ledger + scalar RCF/LFT conformance oracle  
**Owner:** #11970 — generalized-plant and RCF pipeline  
**Solver dependency:** #11948 — Hermitian mapping / Clarabel / independent solver witnesses  
**Benchmark dependency:** #11946 — matched sampled-FRF IQC reproduction

Primary source: Spin, Heemels, van de Wouw and Toth (2026-10-14), [Robust Data-Driven Controller Synthesis from Sampled and Uncertain Frequency Response Estimates using Integral Quadratic Constraints](https://arxiv.org/abs/2610.20248), v1, Sections 2–6.

**Review status:** first-pass transcription against the primary arXiv HTML rendering. Independent second-person mathematical review and visual equation audit are still required before the full generator/SDP implementation is treated as source-conformant. This document does not claim the full paper has been reimplemented.

## 1. Scope and system conventions

The paper is discrete-time. The frequency domain is the unit circle \(\partial\mathbb D\), with \(\omega\in[-\pi,\pi]\) and the sampled positive half-spectrum:
\[
0=\omega_0<\omega_1<\cdots<\omega_{N-1}=\pi.
\]
The open-loop plant \(G\) is connected with prescribed feedback/feedforward wiring and weighting filters to form the generalized plant \(P\). The finite-data methods use uncertain estimates \(\widehat{\mathcal P}_k\), not point estimates treated as exact.

AIRROAD adapters must bind:
- discrete-time sample period / clock and angular-frequency units;
- all generalized input/output port identities and block dimensions;
- the feedback sign convention;
- weighting-filter identities and their frequency-domain enclosures;
- model, topology, acquisition, calibration and operating-point identities;
- whether each uncertainty is deterministic hard-set or statistical.

Do not silently interpret rad/s data as rad/sample or transfer the discrete-time unit-circle theorem to a continuous-time plant.

## 2. Equation ledger: generalized plant and closed-loop map

### Equation (1): state-space generalized plant

\[
\begin{aligned}
x_{t+1}&=Ax_t+B_ww_t+B_uu_t,\\
z_t&=C_zx_t+D_{zw}w_t+D_{zu}u_t,\\
y_t&=C_yx_t+D_{yw}w_t+D_{yu}u_t.
\end{aligned}
\]

Dimensions:
- \(x\in\mathbb R^{n_x}\), \(w\in\mathbb R^{n_w}\), \(u\in\mathbb R^{n_u}\);
- \(z\in\mathbb R^{n_z}\), \(y\in\mathbb R^{n_y}\);
- \(A:n_x\times n_x\), \(B_w:n_x\times n_w\), \(B_u:n_x\times n_u\);
- \(C_z:n_z\times n_x\), \(D_{zw}:n_z\times n_w\), \(D_{zu}:n_z\times n_u\);
- \(C_y:n_y\times n_x\), \(D_{yw}:n_y\times n_w\), \(D_{yu}:n_y\times n_u\).

### Equation (2): block partition of the generalized-plant FRF

\[
P(e^{j\omega})=
\begin{bmatrix}
P_{zw}(e^{j\omega})&P_{zu}(e^{j\omega})\\
P_{yw}(e^{j\omega})&P_{yu}(e^{j\omega})
\end{bmatrix}.
\]

Therefore:
- \(P_{zw}:n_z\times n_w\);
- \(P_{zu}:n_z\times n_u\);
- \(P_{yw}:n_y\times n_w\);
- \(P_{yu}:n_y\times n_u\);
- \(P:(n_z+n_y)\times(n_w+n_u)\).

### Equation (4): feedback closed-loop transfer

\[
T_{zw}=P\star K
=P_{zw}+P_{zu}K(I-P_{yu}K)^{-1}P_{yw}.
\]

The inverse has dimension \(n_y\times n_y\). This sign convention is source-specific and must be reflected in the port-order tests.

## 3. Equation ledger: right/left coprime factors and transformed loop

### Equation (13): controller right-coprime factorization

\[
K=UV^{-1},\qquad
V\in\mathcal{RH}_\infty^{n_y\times n_y},\quad
U\in\mathcal{RH}_\infty^{n_u\times n_y}.
\]

The RCF conditions include invertibility of \(V\) and the Bézout identity:
\[
XV+YU=I_{n_y},
\]
with \(X\in\mathcal{RH}_\infty^{n_y\times n_y}\) and \(Y\in\mathcal{RH}_\infty^{n_y\times n_u}\).

The controller factors are parameterized by a fixed stable basis:
\[
V(e^{j\omega})=C_V\phi(e^{j\omega})+D_V,\qquad
U(e^{j\omega})=C_U\phi(e^{j\omega})+D_U,
\]
\[
\phi(e^{j\omega})=(e^{j\omega}I-A_\phi)^{-1}B_\phi.
\]

Dimensions:
- \(\phi:n_\phi\times n_y\);
- \(C_V:n_y\times n_\phi\), \(D_V:n_y\times n_y\);
- \(C_U:n_u\times n_\phi\), \(D_U:n_u\times n_y\).

The optimization vector is
\[
\theta=\operatorname{vec}(C_V,C_U,D_V,D_U)\in\mathbb R^d,\quad
d=(n_\phi+n_y)(n_u+n_y).
\]
AIRROAD must preserve the paper's vectorization order, real-valued parameterization and fixed basis identity.

### Equation (15): plant-side left coprime factorization and \(\Lambda\)

For the measured \(P_{yu}\) block, the paper uses a left coprime factorization:
\[
P_{yu}=D^{-1}N,\qquad
D:n_y\times n_y,\quad N:n_y\times n_u.
\]
With controller RCF \((V,U)\),
\[
\Lambda=DV-NU\in\mathbb C^{n_y\times n_y}.
\]
The closed-loop expression becomes:
\[
T_{zw}=P_{zw}+P_{zu}U\Lambda^{-1}DP_{yw}.
\]

Do not swap this left factorization for a right factorization; the matrix dimensions and the order \(DV-NU\) depend on it.

### Equation (16): virtual-identity LFT object

The paper defines
\[
M=
\begin{bmatrix}
I-\Lambda&DP_{yw}\\
P_{zu}U&P_{zw}
\end{bmatrix},
\qquad \Delta_I:=I,
\]
and writes \(T_{zw}=\Delta_I\star M\).

Block dimensions:
- top-left \(n_y\times n_y\);
- top-right \(n_y\times n_w\);
- bottom-left \(n_z\times n_y\);
- bottom-right \(n_z\times n_w\);
- \(M:(n_y+n_z)\times(n_y+n_w)\).

**LFT port order is a critical conformance test.** The scalar fixture in Section 8 checks the source equality between Equations (4), (15) and (16). Do not substitute a generic lower-right Redheffer formula without preserving this order.

## 4. Equation ledger: IQC multiplier and SDP identities

### Equations (9)–(12): inertia and FDI

The paper defines an inertia-constrained multiplier class with upper-left block positive definite and lower-right block negative semidefinite. The robust-stability FDI is:
\[
\begin{bmatrix}H\\I\end{bmatrix}^*
\Pi^s
\begin{bmatrix}H\\I\end{bmatrix}\prec0
\quad\text{on }\partial\mathbb D.
\]

For simultaneous robust stability and performance, the augmented multiplier is a structured direct sum of stability and performance multipliers (Equation 11). AIRROAD's typed multiplier must keep the stability and performance channels/dimensions distinct.

### Equations (17)–(20): virtual identity multiplier and Schur complement

For \(\Delta_I=I\), the paper defines
\[
\Pi_I=\left\{\Pi\in\Pi_{\rm in}:
\begin{bmatrix}I\\I\end{bmatrix}^*
\Pi
\begin{bmatrix}I\\I\end{bmatrix}\succeq0
\text{ on }\partial\mathbb D\right\}.
\]
With
\[
\Pi=\begin{bmatrix}\Pi_{11}&\Pi_{12}\\
\Pi_{12}^*&\Pi_{22}\end{bmatrix},
\]
the FDI is transformed by Schur complement to:
\[
\begin{bmatrix}
\Pi_{22}+M^*\Pi_{12}+\Pi_{12}^*M&M^*\\
M&-\Pi_{11}^{-1}
\end{bmatrix}\prec0.
\]
This use requires the stated positive-definiteness of \(\Pi_{11}\). AIRROAD must check the sign, block dimensions and strictness before passing the matrix to Clarabel.

### Equations (25)–(28): plant-linear and one-sided set interpolation

The plant uncertainty uses linear set interpolation:
\[
\mathscr L(\widehat{\mathcal P})(\omega)=
\{(1-\lambda_\omega)\widetilde P_k+
\lambda_\omega\widetilde P_{k+1}:
\widetilde P_k\in\widehat{\mathcal P}_k,\ 
\widetilde P_{k+1}\in\widehat{\mathcal P}_{k+1}\}.
\]

The distinct piecewise-convex operator is:
\[
\mathscr C(\mathcal S)(\omega)=
\{(1-\lambda_\omega)S_k+\lambda_\omega S_{k+1}:
S_k\in\operatorname{co}(\Gamma_{\mathcal S,k}^{+}),\
S_{k+1}\in\operatorname{co}(\Gamma_{\mathcal S,k+1}^{-})\}.
\]
The one-sided endpoint sets \(\Gamma_{\mathcal S,k}^{+}\) and
\(\Gamma_{\mathcal S,k}^{-}\) are allowed to differ. AIRROAD must preserve this intentional interval-local discontinuity.

### Theorem 3 / Equations (29)–(35): finite interval certificate

The plant closed-loop class must be contained in the virtual-identity representation:
\[
\mathscr P\star K\subseteq\Delta_I\star\mathscr M.
\]
For each interval, the finite multiplier LMIs are quantified over every generator:
\[
\gamma_M\in\Gamma_{M,k}^{+}\cup\Gamma_{M,k+1}^{-}.
\]
The interval multipliers are then assembled by the paper's zero-order-hold operator and must belong to the prescribed stability/performance multiplier classes.

For controller optimization, the Schur-complement form is:
\[
\begin{bmatrix}
\Pi_{22,k|k+1}+
\operatorname{He}(\gamma_M^*\Pi_{12,k|k+1})&
\gamma_M^*\\
\gamma_M&-\Pi_{11,k|k+1}^{-1}
\end{bmatrix}\prec0.
\]
Here \(\operatorname{He}(X)=X^*+X\). The block sizes are:
- top-left \(m\times m\), where \(m\) is the identity/input side of the FDI;
- top-right \(m\times n\);
- bottom-left \(n\times m\);
- bottom-right \(n\times n\), matching \(\Pi_{11}\in\mathbb H^{n\times n}\).

The paper separates multiplier-only optimization for fixed \(\theta\) from controller-parameter optimization for fixed stability multipliers. These are alternating convex subproblems, not one jointly convex optimization over all variables.

## 5. Equation ledger: generator construction

### Equations (37)–(42): matrix unit-ball simplex

The complex \(n\times m\) matrix space has real dimension \(d=2nm\) under the real/imaginary embedding. A full-dimensional Euclidean ball requires at least \(d+1\) vertices for a polytope outer cover. The paper's regular simplex construction uses \(d+1\) vertices and has minimum volume among such circumscribed simplices.

### Equations (43)–(47): hard unstructured uncertainty

The paper models additive uncertainty as \(W_1\Delta_uW_2\), with \(\|\Delta_u\|_{\mathcal H_\infty}\leq1\), and constructs a Frobenius-norm outer cover scaled by \(1/\sqrt{\min(n,m)}\). AIRROAD should retain the proof direction:
\[
\|\Delta_u\|_{2,2}\leq\|\Delta_u\|_F
\leq\sqrt{\min(n,m)}\|\Delta_u\|_{2,2}.
\]
Reducing the generator count requires a verified lower-dimensional support or a different proven enclosure; it cannot be done for a full-dimensional ball by assumption.

### Equations (48)–(55): source statistical assumption

The paper's stochastic generator path assumes circular complex Gaussian FRF errors and statistical independence between frequency points. It uses a vectorized complex normal, real embedding, chi-square radius and simplex cover.

This is a scoped assumption of its stochastic branch, not a requirement of IQC theory generally. AIRROAD's #12013 joint \(C/P\) estimator can support a future improper/cross-frequency extension, but that is a deviation from the source method and must be benchmarked separately. Do not claim the source statistical generator theorem covers correlated or improper estimates.

### Equations (57)–(67): closed-loop generator construction

- Equation (57) introduces an interval-specific left-coprime factorization for the \(P_{yu}\) block around the midpoint.
- Equations (58)–(59) define the intermediate transformed set \(\mathscr F(\omega)\) and \(\Lambda(\omega)\).
- Equations (60)–(62) bound the controller basis interpolation error \(r_{k|k+1}\) and propagate the resulting set through \(V\) and \(U\).
- Important caveat: the paper states the supremum in Equation (60) can be approximated by sufficiently dense offline sampling. That is not, by itself, a certified global upper bound. AIRROAD must use the existing certified-extrema/interval engines and bind the result to the exact \(\phi\), controller iterate and interval.
- Equations (63)–(65) outer-approximate a bilinear product by the convex hull of all pairwise endpoint-generator products. The enclosure is an outer approximation; no inner approximation may be used as a robust-safety bound.
- Equations (66a)–(67f) form one-sided \(\Gamma_{\Lambda,k}^{\pm}\) and \(\Gamma_{M,k}^{\pm}\) generator matrices. Their \(\pm\) limits must stay distinct through the final LMI generator quantifiers.

### Industrial scope

The paper reports 247 logarithmically spaced measured frequencies, ten independent multisine realizations each repeated ten times, an approximately 95.45% (2-sigma) stochastic generator region, and a 740-point logarithmic interpolation/display grid. The exact sequence length and sampling frequency are withheld. Its Section 5.4 also omits inter-frequency weighting-filter behavior for simplicity. AIRROAD must report these limitations and must not claim to reproduce the wirebonder experiment without its data/setup.

## 6. Schur-complement source conformance

Equation (20) is checked independently of the RCF/LFT fixture. For real-symmetric matrices the source block is
[
Q=\begin{bmatrix}
S&M^T\\M&-\Pi_{11}^{-1}
\end{bmatrix},\quad
S=\Pi_{22}+M^T\Pi_{12}+\Pi_{12}^TM,
]
and the Schur-reduced matrix is
[
S+M^T\Pi_{11}M.
]
Since (-\Pi_{11}^{-1}\prec0) for (\Pi_{11}\succ0), the full block is strictly negative definite exactly when this reduced matrix is strictly negative definite. The verifier checks both via exact rational leading principal minors.

The feasible 2×2 case uses:
[
\Pi_{11}=\operatorname{diag}(2,1),\quad
\Pi_{12}=\operatorname{diag}(1/4,0),\quad
\Pi_{22}=\operatorname{diag}(-3,-2),\quad
M=\operatorname{diag}(1/2,1/3).
]
It yields (S=\operatorname{diag}(-11/4,-2)) and the Schur reduction (\operatorname{diag}(-9/4,-17/9)\prec0). The full 4×4 block has leading principal minors (-11/4,11/2,-9/4,17/8), so it is strictly negative definite.

The infeasible control uses (\Pi_{11}=2I), (\Pi_{12}=0), (\Pi_{22}=-(1/5)I), (M=(1/2)I). The correct Schur reduction is (+(3/10)I), so the full block is indefinite. **Reversing the Schur-complement sign** produces (-(7/10)I\prec0), a false feasibility result. This is a deliberate negative control for the exact sign in Equation (20).

The verifier rejects any substitution of non-strict negativity for the paper's strict (\prec0) condition. This fixture is algebraic conformance only; it does not imply a controller has been synthesized.

## 7. Exact scalar source-equation conformance fixture

Use a scalar \(1\times1\) generalized-plant instance:
\[
P_{zw}=1,\quad P_{zu}=2,\quad P_{yw}=3,\quad P_{yu}=1/4.
\]
Use the left factorization \(D=1,N=1/4\), so \(P_{yu}=D^{-1}N\). Use controller factors \(U=1,V=2\), hence \(K=UV^{-1}=1/2\). The Bézout witness \(X=0,Y=1\) satisfies \(XV+YU=1\).

Then:
\[
\Lambda=DV-NU=2-1/4=7/4.
\]
Equation (4) gives:
\[
T_{zw}=1+2(1/2)(1-(1/4)(1/2))^{-1}3=31/7.
\]
Equation (15) independently gives:
\[
T_{zw}=1+2(1)(7/4)^{-1}(1)3=31/7.
\]
Equation (16) yields:
\[
M=\begin{bmatrix}-3/4&3\\2&1\end{bmatrix}.
\]
The scalar LFT convention that preserves the source identity is:
\[
T_{zw}=M_{22}+M_{21}(1-M_{11})^{-1}M_{12}=31/7.
\]
A port-order or sign error changes this equality and must fail the exact oracle.

The fixture verifies algebra only. It does not check stable rational realizability for a nonconstant plant, the full matrix-valued LFT theorem, any interval generator inclusion, or robust stability.

## 8. Independent review checklist

Before the next synthesis increment:
- [ ] independently review equations (13)–(20), including the left/right coprime convention and LFT block order;
- [ ] visually inspect the original PDF figures/equation layouts where HTML extraction could obscure block alignment;
- [ ] review dimensions for the vector-valued generalization of the scalar fixture;
- [ ] independently verify one-sided endpoint construction (28), controller remainder (60) and bilinear outer product (65);
- [ ] verify the outer/inner inclusion direction at every generator stage;
- [ ] verify equation (35) Schur-complement sign and strictness for Hermitian blocks;
- [ ] review source assumptions on frequency-independent circular complex Gaussian samples before adapting with \(C/P\);
- [ ] have a second reviewer sign the equation/dimension audit before using source-derived constraints for AIRROAD claims.

## 9. Claim ceiling

This is a primary-source transcription ledger and scalar identity test. It does not implement the full generalized-plant generator path, controller iteration, finite-data SDP, continuous-frequency certificate, physical FRF, robust stability, performance, wirebonder reproduction or airworthiness.
