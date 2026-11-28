# CMAQ-058 — Exact capital-restart renewal proof note v0.1

Status: theorem note for the capital-only restart construction. This is conditional on the per-expert factor contract; it does not prove the factor producer's assumptions from data.

## 1. Filtration and direct conditional null

Let \(\mathcal F_{t-1}\) contain all paired observations through time \(t-1\), all past covariates, and the complete state of the predictable policy. Let \(Z_t\) be revealed before the current response pair \((X_t,Y_t)\).

The direct conditional paired-equality null is:

\[
\mathcal L(X_t\mid Z_t,\mathcal F_{t-1})
=
\mathcal L(Y_t\mid Z_t,\mathcal F_{t-1}).
\]

For expert \(k\), a Fourier response map \(\phi_k\), a covariate map \(\psi_k\), and a direction \(u_{k,t}\) are fixed or selected using only pre-pair information. In particular, \(u_{k,t}\) is \(\mathcal F_{t-1}\)-measurable; the current pair cannot affect it.

A unit-norm product feature difference is

\[
r_{k,t}
=
\frac12\psi_k(Z_t)\otimes
\left(\phi_k(Y_t)-\phi_k(X_t)\right),
\qquad
\|r_{k,t}\|_2\leq 1.
\]

Under the null,

\[
\mathbb E[r_{k,t}\mid Z_t,\mathcal F_{t-1}]=0.
\]

For a predictable direction \(\|u_{k,t}\|_2\leq1\) and fixed \(0<\lambda<1\), define

\[
f_{k,t}=1+\lambda\langle u_{k,t},r_{k,t}\rangle.
\]

Then \(1-\lambda\leq f_{k,t}\leq1+\lambda\), so the factor is strictly positive, and

\[
\mathbb E[f_{k,t}\mid Z_t,\mathcal F_{t-1}]=1.
\]

The proof only needs the weaker inequality \(\mathbb E[f_{k,t}\mid\mathcal F_{t-1}]\leq1\), after integrating over the predictable covariate. No independence between expert streams is needed.

## 2. One e-process for each start/expert pair

Fix a positive start-time prior \((\nu_s)_{s\geq1}\) with \(\sum_s\nu_s=1\), and a positive kernel prior \((w_k)_{k=1}^K\) with \(\sum_kw_k=1\). Both are frozen before the monitored stream.

For start \(s\) and expert \(k\), define

\[
M_{s,k,t}=
\begin{cases}
1,&t<s,\\
\prod_{i=s}^t f_{k,i},&t\geq s.
\end{cases}
\]

For \(t\geq s\),

\[
\mathbb E[M_{s,k,t}\mid\mathcal F_{t-1}]
=
M_{s,k,t-1}
\mathbb E[f_{k,t}\mid\mathcal F_{t-1}]
\leq M_{s,k,t-1}.
\]

Thus each fixed-start component is a nonnegative supermartingale from unit initial wealth at time \(s-1\). The fact that \(f_{k,t}\)'s policy state may contain observations from before \(s\) does not invalidate this argument: the direction and factor are still predictable with respect to the full filtration \(\mathcal F_{t-1}\).

## 3. Exact renewal recurrence

Define the active restart wealth for expert \(k\):

\[
R_{k,t}=\sum_{s=1}^{t}\nu_s M_{s,k,t},
\qquad R_{k,0}=0.
\]

For \(s<t\), \(M_{s,k,t}=f_{k,t}M_{s,k,t-1}\). For the new start \(s=t\), \(M_{t,k,t}=f_{k,t}\). Therefore:

\[
\begin{aligned}
R_{k,t}
&=
\sum_{s=1}^{t-1}\nu_s f_{k,t}M_{s,k,t-1}
+\nu_t f_{k,t}\\
&=f_{k,t}\left(R_{k,t-1}+\nu_t\right).
\end{aligned}
\]

This is an algebraic identity, not an approximation. It compresses all active start components for one kernel into one scalar restart state.

The exact unstarted prior mass is

\[
U_t=\sum_{s=t+1}^{\infty}\nu_s.
\]

The complete kernel × start mixture is

\[
E_t=U_t+\sum_{k=1}^K w_kR_{k,t}.
\]

Because \(U_t=U_{t-1}-\nu_t\), and every factor has conditional expectation at most one,

\[
\begin{aligned}
\mathbb E[E_t\mid\mathcal F_{t-1}]
&\leq U_t+\sum_kw_k(R_{k,t-1}+\nu_t)\\
&=U_{t-1}-\nu_t
+\sum_kw_kR_{k,t-1}
+\nu_t\sum_kw_k\\
&=U_{t-1}+\sum_kw_kR_{k,t-1}\\
&=E_{t-1}.
\end{aligned}
\]

Here \(\sum_kw_k=1\) is essential. At \(t=0\), \(U_0=1\), so \(E_0=1\). Thus \((E_t)\) is a nonnegative supermartingale under the common null. By Ville's inequality,

\[
\Pr\left(\sup_{t\geq0}E_t\geq\frac1{\alpha_{\rm test}}\right)
\leq\alpha_{\rm test}.
\]

If the monitor is nested inside the CMAQ-053 fleet allocation, \(\alpha_{\rm test}\) is the already-composed local test budget. No additional alpha split is required across starts or kernels; the fixed prior mixtures are the multiplicity control.

## 4. Dyadic start prior

CMAQ-055 uses epochs

\[
I_j=\{2^{j-1},\ldots,2^j-1\},
\qquad
p_j=\frac1{j(j+1)},
\]

and splits each epoch mass uniformly:

\[
\nu_s=\frac1{2^{j-1}j(j+1)},\quad s\in I_j.
\]

The epoch masses telescope:

\[
\sum_{j=1}^{\infty}\frac1{j(j+1)}=1.
\]

The exact residual \(U_t\) is retained. Dropping it or renormalizing only active starts would change the process and invalidate the recurrence proof above.

## 5. Stable log implementation

Let

\[
C_{k,t}=\sum_{i=1}^t\log f_{k,i},
\qquad
A_{k,t}=\sum_{s=1}^t\nu_s e^{-C_{k,s-1}},
\quad C_{k,0}=0.
\]

Then

\[
\log A_{k,t}
=
\operatorname{logaddexp}
\left(
\log A_{k,t-1},
\log\nu_t-C_{k,t-1}
\right)
\]

and

\[
\log R_{k,t}=C_{k,t}+\log A_{k,t}.
\]

The total wealth is computed with log-sum-exp over \(\log U_t\) and \(\log w_k+\log R_{k,t}\). The rational start prior and kernel priors remain exact; the e-wealth arithmetic is floating point with explicit finite/overflow guards. This is not a formally verified floating-point implementation.

## 6. Critical distinction: capital-only versus full restart

CMAQ-058 resets the capital at every possible start but shares the same predictable policy-state trajectory across starts. A start-\(s\) component uses factors \(f_{k,t}\) for \(t\geq s\) whose directions may depend on observations before \(s\). That is valid under the full filtration because every factor is still predictable and conditionally mean-one.

CMAQ-055 instead resets both:
- the component's capital; and
- its own policy state.

Those are different procedures. CMAQ-058 is not an optimized implementation of CMAQ-055's full reset. The latter remains the exact comparator for the effect of resetting policy state.

## 7. Assumptions not proven by this note

This note does not prove:
- that the caller's covariate is timestamped and available before the response pair;
- that the observed pair really obeys the conditional equality null;
- that source identity, clocks or provenance are authentic;
- that data are physically independent across streams;
- that the Fourier inventory detects arbitrary conditional alternatives;
- that floating-point arithmetic exactly preserves the real-valued wealth at all horizons.

Those are separate statistical, data-governance and numerical-qualification obligations.
