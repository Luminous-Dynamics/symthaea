# SDCB vector continuous-time state-space oracle v1

**Status:** bounded exact numerical reference. Ten Rust tests are authored but not compiled or executed in this environment. All frozen numerical fixtures are analytical or independently quadrature-checked; no real sensor data or physical validation is claimed.

## 1. Purpose and relation to the existing programme

This increment advances DC1 from issue #13904. It extends #13903's scalar OU window covariance to a vector linear stochastic differential equation. It does not replace the scalar window kernel, the full-covariance subset oracle (#13881), finite-scenario maximin (#13882), or the categorical adaptive planners (#13793/#13804).

The purpose is to provide an exact small-instance reference for propagation of coupled physical-process, shared measurement-system, reference-chain and device-local states across irregular elapsed-time intervals.

## 2. Continuous-time model

For a vector state \(x(t)\in\mathbb R^n\):

\[
dx(t)=Ax(t)\,dt+G\,dW(t), \qquad Q_c=GG^\top.
\]

For elapsed time \(\Delta\geq0\):

\[
F(\Delta)=e^{A\Delta},
\]

\[
Q(\Delta)=\int_0^\Delta e^{Au}Q_c e^{A^\top u}\,du.
\]

The exact discrete transition is:

\[
x(t+\Delta)\mid x(t)\sim
\mathcal N(F(\Delta)x(t),Q(\Delta)).
\]

For a Gaussian belief \(x_t\sim\mathcal N(m_t,P_t)\):

\[
m_{t+\Delta}=F(\Delta)m_t,
\]

\[
P_{t+\Delta}=F(\Delta)P_tF(\Delta)^\top+Q(\Delta).
\]

Static scientific coordinates are represented by zero generator rows and zero diffusion rows. They may retain prior cross-covariance with dynamic nuisance coordinates; they must not be accidentally forced to evolve.

## 3. Van Loan block exponential

Construct:

\[
\mathcal V(\Delta)=
\exp\left(
\begin{bmatrix}
A & Q_c\\
0 & -A^\top
\end{bmatrix}\Delta
\right)
=
\begin{bmatrix}
F & U\\
0 & F^{-\top}
\end{bmatrix}.
\]

Then:

\[
Q(\Delta)=UF^\top.
\]

The reference uses the diagonal [13/13] Padé approximant with scaling and squaring for the matrix exponential and partial-pivoting linear solves. The state dimension is capped at eight (block dimension at most sixteen) to keep exhaustive numerical controls tractable.

This is an established method originating in Van Loan's work on matrix-exponential integrals. It is not claimed as a novel numerical algorithm. The contribution is its use as an exact SDCB reference across coupled scientific and metrological state coordinates.

## 4. Validation contract

- Generator and diffusion covariance dimensions must agree.
- Matrices must contain finite values.
- \(Q_c\) must be symmetric and positive semidefinite; semidefinite is allowed.
- Negative elapsed time is rejected; zero elapsed time returns identity \(F\) and zero \(Q\).
- No diagonal jitter or silent covariance repair is applied.
- The propagated Gaussian covariance is checked for symmetry and positive semidefiniteness.
- Non-finite matrix-exponential intermediates fail closed.
- The reference exposes transition \(F\), process covariance \(Q\), and elapsed time separately for independent inspection.

Numerical tolerances are scale-relative and are not metrological uncertainty tolerances.

## 5. Frozen fixtures

### Scalar OU

For \(A=-0.7\), stationary variance \(\sigma^2=1.5\), \(Q_c=2(0.7)(1.5)=2.1\), and \(\Delta=2.3\):

- \(F=0.19988761407514452\)
- \(Q=1.4400674126090192\)

This reproduces the scalar OU result used by #13903.

### Coupled two-factor state

For:

\[
A=\begin{bmatrix}-1&0.25\\0&-0.5\end{bmatrix},\quad
Q_c=\begin{bmatrix}0.4&0.12\\0.12&0.3\end{bmatrix},\quad \Delta=1.25,
\]

the expected transition is:

\[
F\approx
\begin{bmatrix}
0.2865047968601900&0.1243783158294001\\
0&0.5352614285189903
\end{bmatrix}
\]

and process covariance is:

\[
Q\approx
\begin{bmatrix}
0.1995091523166423&0.0900913798078700\\
0.0900913798078700&0.2140485609419429
\end{bmatrix}.
\]

The off-diagonal process covariance is material and must not be dropped.

### Non-normal generator

For:

\[
A=\begin{bmatrix}-1&10\\0&-2\end{bmatrix},\quad
Q_c=\begin{bmatrix}0.2&0.05\\0.05&0.3\end{bmatrix},\quad \Delta=0.1,
\]

the expected transition is:

\[
F\approx
\begin{bmatrix}
0.9048374180359595&0.8610666495797772\\
0&0.8187307530779818
\end{bmatrix}
\]

and:

\[
Q\approx
\begin{bmatrix}
0.0303703939111182&0.0162415101669830\\
0.0162415101669830&0.0247259965473271
\end{bmatrix}.
\]

The process covariance values were independently checked by numerical quadrature of the defining integral, rather than only by the block-exponential identity.

## 6. Tests and limitations

Ten authored tests cover scalar OU closed form, coupled cross-covariance, a non-normal generator, zero time, stable long gap, static state coordinates, Gaussian propagation, semigroup composition, invalid covariance/time, and zero/non-normal matrix exponentials.

This is not yet:
- a continuous-time observation/update filter;
- a vector integrated-window observation kernel;
- a parameter or covariance-model estimator;
- a nonlinear/non-Gaussian state estimator;
- a dynamic adaptive action planner;
- a physical calibration or traceability service.

Van Loan block exponentials can be ill-conditioned for very stiff systems or extreme elapsed times. Such cases must be explicitly stress-tested against alternative formulations and high-precision references before broadening the numerical domain. This bounded oracle must fail on non-finite intermediates rather than claiming universal stability.

## 7. Research integration

The next extensions should reuse this state transition:
1. interval-integrated observation operators for unequal mobile/reference windows;
2. predictive innovation likelihoods for finite covariance/dynamics model comparison;
3. mixture posterior moments including between-model variance;
4. robust support-set worst-case loss kept distinct from Bayesian model averaging;
5. elapsed-time-aware finite adaptive policies;
6. evidence-ledger binding with state versions and atomic rejection.

HDC/CfC may propose candidate experiments only after these exact evaluators exist. Learned policies remain subordinate to hard budgets, abstention, authorization and evidence qualification.
