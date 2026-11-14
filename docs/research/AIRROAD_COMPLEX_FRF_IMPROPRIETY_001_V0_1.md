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

## 4. Required V0.1 checks

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

V0.1 is an exact scalar statistical-representation reference. It does not establish that real AIRROAD FRF errors are improper, Gaussianity, a physical pseudo-covariance estimate, finite-sample coverage, deterministic uncertainty bounds, IQC feasibility, robust stability or airworthiness.
