# SDCB interval-integrated observation operator v1

**Status:** bounded exact linear-Gaussian reference. Seven Rust tests are authored but not compiled or executed in this environment. The OU known answers are independently analytical; no sensor data or physical test is claimed.

## 1. Why this bridge matters

#13903 computes covariance between averages of one stationary scalar OU process. #13937 propagates a coupled vector linear SDE through elapsed time. This increment composes them: each sensor/reference action has its own integration window and loading vector, and overlapping or nested observations share the same evolving process noise.

That is the central mathematical issue in short mobile-reference collocation. A fast mobile sensor may be collocated only briefly with a slower reference instrument, whose reported value averages over a longer interval. The two reported values are not necessarily averages over the same physical interval. Kok & Tabandeh (2026) explicitly derive variance and covariance expressions for such averages and show why process variability and unequal averaging windows must enter testing/calibration uncertainty (DOI 10.1002/qre.70319).

## 2. State and observation model

Let the state follow the continuous-time linear SDE:

\[
dx(t)=Ax(t)\,dt+G\,dW(t), \qquad Q_c=GG^\top.
\]

Measurement \(i\) has a declared time window \(I_i=[a_i,b_i]\), duration \(T_i=b_i-a_i\), and state loading row \(c_i^\top\):

\[
y_i=\frac{1}{T_i}\int_{a_i}^{b_i}c_i^\top x(t)\,dt+\epsilon_i.
\]

The initial state distribution is defined at the earliest requested window start. Each observation may have a different start, end and loading vector.

## 3. Exact augmented-state construction

For \(m\) measurements, augment the physical state with one integral accumulator per observation:

\[
r_i(t)=\int_{a_i}^{t} c_i^\top x(s)\,ds
\quad\text{while }t\in[a_i,b_i],
\]

and freeze \(r_i\) before \(a_i\) and after \(b_i\).

On each segment between the sorted set of all window boundaries, define the augmented state \(z=[x^\top,r_1,\ldots,r_m]^\top\) and piecewise-constant generator:

\[
A_{\mathrm{aug}}(t)=
\begin{bmatrix}
A&0\\
C_{\mathrm{active}}(t)&0
\end{bmatrix},
\qquad
Q_{c,\mathrm{aug}}=
\begin{bmatrix}
Q_c&0\\
0&0
\end{bmatrix}.
\]

The existing Van Loan vector-state oracle computes exact segment \(F\) and \(Q\). Compose segments using:

\[
F_{k:0}=F_kF_{k-1:0},\qquad
Q_{k:0}=F_kQ_{k-1:0}F_k^\top+Q_k.
\]

At the final boundary, extract:
- state endpoint transition and process covariance;
- each integrated observation operator divided by its own duration;
- the full covariance among interval-average process-noise contributions;
- cross-covariance between endpoint state noise and each integrated observation.

The accumulator dimension is bounded together with the physical state so the existing maximum augmented dimension remains eight. No numerical quadrature is required at run time for this bounded linear-SDE reference.

## 4. Endpoint filtering with interval observations

Let the initial state be \(x_0\sim N(m_0,P_0)\). The bundle returns:
- endpoint state: \(x_T=F x_0+w\);
- observation vector: \(y=H x_0+v\);
- \(\operatorname{Cov}(w)=Q_{xx}\);
- \(\operatorname{Cov}(v)=Q_{yy}+R_{\mathrm{acq}}\);
- \(\operatorname{Cov}(w,v)=Q_{xy}\).

Then:

\[
S_y=HP_0H^\top+Q_{yy}+R_{\mathrm{acq}},
\]

\[
C_{x_T,y}=FP_0H^\top+Q_{xy}.
\]

Condition the endpoint state using:

\[
m_T'=Fm_0+C_{x_T,y}S_y^{-1}(y-Hm_0),
\]

\[
P_T'=FP_0F^\top+Q_{xx}
-C_{x_T,y}S_y^{-1}C_{x_T,y}^\top.
\]

The \(Q_{xy}\) term is essential. Ignoring process-noise correlation between the final state and the interval averages is equivalent to treating an interval mean as a point observation or incorrectly independent noise.

Acquisition/electronics covariance is supplied separately. Shared process variation is not silently folded into independent per-device electronics noise.

## 5. Frozen OU cross-check against #13903

Use stationary OU \(dx=-x\,dt+\sqrt{2}\,dW\), stationary initial variance 1:

- slow reference window: [0, 2];
- fast mobile window: [1, 2].

The resulting observation covariance must be:

\[
S_y=
\begin{bmatrix}
0.5676676416183064&0.5676676416183064\\
0.5676676416183064&0.7357588823428847
\end{bmatrix}.
\]

This reproduces the nested-window known answer from the scalar OU reference #13903. Duplicate windows produce duplicate covariance rows. Disjoint windows recover the exponential gap covariance.

For acquisition noise \(R=\operatorname{diag}(0.01,0.01)\) and observed averages [1.0, 0.7], the endpoint posterior is:
- mean: 0.43660632952719985;
- variance: 0.4478182406823277.

These are constructed analytical values, not physical sensor results.

## 6. Test coverage and limits

Seven authored tests cover:
1. nested windows against the #13903 scalar OU reference;
2. duplicate interval covariance rows;
3. disjoint-window exponential covariance;
4. endpoint conditioning with process/observation cross-covariance;
5. unequal loadings across coupled state factors;
6. electronics/acquisition covariance kept separate;
7. invalid windows, duplicate IDs and bounded dimensions.

Current v1 uses rectangular averaging kernels and piecewise-constant state dynamics/loadings inside each declared interval. It does not yet handle arbitrary time-varying loadings, nonlinear dynamics, missing/irregular sub-samples, or general nonstationary kernels.

The covariance matrix can be positive semidefinite (e.g. duplicate windows), not necessarily positive definite. Downstream conditioning requires the total innovation covariance to be positive definite; repeated/duplicate data with zero independent acquisition noise may require deduplication or a rank-aware update.

## 7. Competitive protocol

Compare against:
- point-sample approximation at the window midpoint;
- naive independent-observation covariance;
- scalar OU averaging kernel (#13903);
- standard continuous-time Kalman filtering with correct interval integration;
- Gaussian-process temporal covariance and direct quadrature reference;
- OLS/GDR/C-GDR as applicable to the Kok–Tabandeh short-collocation task.

All methods receive identical window timestamps, loadings, acquisition-noise assumptions and reference budgets. Compare held-out target bias, coverage, interval width, calibration parameter error, and decision loss per reference cost.

No SOTA claim is justified until the paper's methods and assumptions are reproduced and the full matched tournament is run.
