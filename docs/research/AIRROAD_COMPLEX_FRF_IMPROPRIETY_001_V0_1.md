# AIRROAD-COMPLEX-FRF-IMPROPRIETY-001 — complementary covariance and properness

**Status:** V0.1 exact-rational scalar complex-statistics reference  
**Tracking issue:** [#11990](https://github.com/Luminous-Dynamics/symthaea/issues/11990)  
**Integrates:** #11986 repeat-period covariance, #11988 MIMO excitation admissibility

> This reference demonstrates why ordinary covariance alone may be incomplete for complex-valued errors. It does not assert that a physical AIRROAD measurement is improper.

## 1. Statistical objects

For a zero-mean complex vector (z\in\mathbb C^n), retain both:
[
C=\mathbb E[zz^H],\qquad P=\mathbb E[zz^T].
]
Here (C) is Hermitian positive semidefinite and (P) is complex symmetric. The augmented covariance is:
[
C_a=\mathbb E\left[
\begin{bmatrix}z\\z^*\end{bmatrix}
\begin{bmatrix}z\\z^*\end{bmatrix}^{H}
\right]
=\begin{bmatrix}C&P\\P^*&C^*\end{bmatrix}.
]
For zero-mean complex variables, properness is (P=0). Properness is a model condition that should be tested or justified from the acquisition/noise model; do not set (P=0) by convenience.

## 2. Exact scalar mapping

Let (z=x+iy) and define the real covariance:
[
\Sigma_{xy}=
\begin{bmatrix}
\operatorname{var}(x)&\operatorname{cov}(x,y)\\
\operatorname{cov}(x,y)&\operatorname{var}(y)
\end{bmatrix}
=
\begin{bmatrix}a&b\\b&d\end{bmatrix}.
]

Then:
[
C=\mathbb E[zz^*]=(a+d)+0i,
\qquad
P=\mathbb E[z^2]=(a-d)+2bi.
]
The inverse scalar mapping is:
[
a=\frac{\operatorname{Re}C+\operatorname{Re}P}{2},\quad
d=\frac{\operatorname{Re}C-\operatorname{Re}P}{2},\quad
b=\frac{\operatorname{Im}P}{2}.
]

For the augmented covariance:
[
C_a=\begin{bmatrix}C&P\\P^*&C^*\end{bmatrix}.
]
The real and augmented representations are related by a nonsingular linear transform, so positive semidefiniteness must agree. For the scalar case:
[
\det(C_a)=4\det(\Sigma_{xy}).
]

## 3. Exact-rational discriminator

The improper fixture is:
[
\Sigma_{xy}=\begin{bmatrix}4&1/2\\1/2&1\end{bmatrix}.
]
It gives:
[
C=5,\qquad P=3+i,
\qquad
C_a=\begin{bmatrix}5&3+i\\3-i&5\end{bmatrix}.
]
The augmented determinant is (25-|3+i|^2=15>0). The real determinant is (4-1/4=15/4>0), and the exact congruence relation (\det(C_a)=4\det(\Sigma_{xy})) holds.

The proper control is:
[
\Sigma_{xy}^{\rm proper}=\begin{bmatrix}5/2&0\\0&5/2\end{bmatrix},
\qquad C=5,\qquad P=0.
]
Both models therefore have the same ordinary complex covariance (C=5), but different complementary covariance and different real/augmented joint covariance. A covariance-only representation cannot distinguish these models.

## 4. Exact two-coordinate MIMO fixture

V0.1 also includes a two-coordinate complex vector (z=x+iy\in\mathbb C^2). Its real vectorization is frozen as ([x_0,x_1,y_0,y_1]^T), corresponding to `f0:Re(z0), f1:Re(z1), f0:Im(z0), f1:Im(z1)`.

The improper real covariance is:
[
\Sigma_{\mathbb R}=\begin{bmatrix}
4&0&1/2&1/4\\
0&3&-1/4&1/2\\
1/2&-1/4&1&0\\
1/4&1/2&0&2
\end{bmatrix}.
]
It induces:
[
C=\begin{bmatrix}5&-i/2\\i/2&5\end{bmatrix},\qquad
P=\begin{bmatrix}3+i&0\\0&1+i\end{bmatrix}.
]
The (4\times4) augmented covariance is:
[
C_a=\begin{bmatrix}
5&-i/2&3+i&0\\
i/2&5&0&1+i\\
3-i&0&5&i/2\\
0&1-i&-i/2&5
\end{bmatrix}.
]
The exact leading principal minors of (\Sigma_{\mathbb R}) are (4,12,11,5353/256); those of (C_a) are (5,99/4,295/4,5353/16). All are positive. The determinant relation is:
[
\det(C_a)=16\det(\Sigma_{\mathbb R})=5353/16.
]

A matched proper control uses:
[
\Sigma_{\mathbb R}^{\rm proper}=\begin{bmatrix}
5/2&0&0&1/4\\
0&5/2&-1/4&0\\
0&-1/4&5/2&0\\
1/4&0&0&5/2
\end{bmatrix}.
]
It has the **same ordinary covariance matrix (C)** as the improper case, but (P=0). Its real and augmented leading principal minors are positive, with determinants (9801/256) and (9801/16), respectively. The same-(C), different-(P) counterexample therefore holds for a two-coordinate complex vector, not only for one scalar.

The verifier independently constructs (C), (P), the augmented matrix and the real covariance from each representation. It does not simply compare precomputed matrix literals.

## 5. Required V0.1 checks

Use exact Fraction arithmetic and pairs of rational real/imaginary components:
- derive (C,P) from the real covariance;
- derive the real covariance back from (C,P);
- construct (C_a) and check exact Hermitian symmetry;
- check nonnegative diagonal entries and exact determinant;
- verify (\det(C_a)=4\det(\Sigma_{xy}));
- compare proper and improper fixtures with equal (C) but unequal (P);
- reject a properness claim when (P\ne0);
- reject discarding (P\ne0);
- include a malformed augmented covariance with a negative determinant;
- reject conversion of statistical covariance into a deterministic hard set;
- bind all inputs to the same campaign/model/acquisition/clock identity.

## 5. MIMO extension requirements

For (z\in\mathbb C^n), the vector case must retain:
- Hermitian (C\in\mathbb C^{n\times n});
- symmetric (P\in\mathbb C^{n\times n});
- the complete (2n\times2n) augmented Hermitian covariance;
- a separately constructed (2n\times2n) covariance of ([\operatorname{Re}z^T,\operatorname{Im}z^T]^T).

Freeze vectorization order across frequency, output/input ports, and real/imaginary components. A MIMO implementation must compare the two representations with independent PSD checks and report numerical tolerances, not silently discard complementary covariance.

## 6. Claim ceiling

V0.1 is an exact scalar and two-coordinate MIMO statistical-representation reference. It does not establish that real AIRROAD FRF errors are improper, Gaussianity, a physical pseudo-covariance estimate, finite-sample coverage, deterministic uncertainty bounds, IQC feasibility, robust stability or airworthiness.
