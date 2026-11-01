# HARDSEC-025 — Exact rational restarted Markov portfolio oracle

**Status:** First finite-state implementation slice. It checks the restarted capital mixture and its conditional supermartingale inequality on one exact two-state fixture. It does **not** implement calibration confidence bounds or establish a runtime PFA guarantee for hardware telemetry.

Issue: [HARDSEC-025 / #9704](https://github.com/Luminous-Dynamics/symthaea/issues/9704)  
Parent research: [HARDSEC-024 / PR #9679](https://github.com/Luminous-Dynamics/symthaea/pull/9679)

## 1. Why a separate restart layer

The known-transition Markov oracle in HARDSEC-024 tests model-conditional betting. HARDSEC-025 adds a distinct question: can we distribute one unit of initial capital over every possible change start time, without resetting or minting evidence, so a portfolio begun near a late change is not burdened by the wealth history before that change?

Liu, Prinster and Orabona's 2026 parameter-free conditional conformal martingale work uses a restarting universal portfolio with heavy-tailed initial wealth allocation over start times. Their result is for their stated conditional-conformal setting; it is not automatically a theorem for PMU streams or Markov telemetry. This example is an exact small-state adaptation of the *capital-allocation pattern*, not a reproduction of their continuous three-asset algorithm.

## 2. Fixture model

The two-state transition matrix is

\[
P=\begin{pmatrix}3/4&1/4\\1/2&1/2\end{pmatrix}.
\]

The exact fixture upper envelope is

\[
U=\begin{pmatrix}4/5&1/2\\3/5&3/5\end{pmatrix},
\]

so every entry satisfies \(P_{ij}\le U_{ij}\). The envelope is supplied as an exact rational fixture constant. It is **not** estimated from calibration data and has no confidence level.

For a portfolio \(w=(w_{\rm cash},w_0,w_1)\) on the unit simplex and a transition from state \(i\) to state \(j\), the return is

\[
R_t(w)=w_{\rm cash}+\frac{w_j}{U_{ij}}.
\]

Under the stated fixture null,

\[
\mathbb E[R_t(w)\mid X_{t-1}=i]
=w_{\rm cash}+\sum_{j=0}^1w_j\frac{P_{ij}}{U_{ij}}
\le w_{\rm cash}+w_0+w_1=1.
\]

Thus each fixed portfolio's wealth is a nonnegative supermartingale for this known fixture model. A fixed mixture over the six rational simplex points with denominator two is also a nonnegative supermartingale.

## 3. Restart capital accounting

For start time \(s\ge1\), allocate

\[
\pi_s=\frac1{s(s+1)}.
\]

The exact telescoping identity is

\[
\sum_{s=1}^{t}\pi_s=\frac{t}{t+1},\qquad
r_t=1-\sum_{s=1}^{t}\pi_s=\frac1{t+1}.
\]

At time \(t\), each component started at \(s\le t\) receives the same current observation from its start onward. The aggregate is

\[
M_t=r_t+\sum_{s=1}^{t}\pi_s
\left(\frac1{|\mathcal W|}\sum_{w\in\mathcal W}
\prod_{u=s}^{t}R_u(w)\right).
\]

The unstarted residual is held as constant cash. The implementation checks the capital identity with exact rational arithmetic at every update. A restart transfers mass from the reserve into a new component; it does not mint another unit of wealth or another alpha budget.

Since each fixed-\(s,w\) component is a nonnegative supermartingale after its start, and is constant before its start, the nonnegative weighted sum plus residual cash is also a nonnegative supermartingale. Ville's inequality can then be applied to the aggregate under the model and envelope assumptions.

## 4. Exact known answer

The oracle enumerates every \(2^6=64\) path from each initial state. All transition path masses are integer numerators over \(4^6=4096\), and the total mass is exactly one in both episodes.

The deliberately high fixture threshold is \(1/\alpha=2\), with \(\alpha=1/2\), to make the short-horizon crossing calculation nontrivial:

| Initial state | Enumerated paths | Exact anytime crossing mass | Probability |
|---:|---:|---:|---:|
| 0 | 64 | 64 / 4096 | 1 / 64 |
| 1 | 64 | 128 / 4096 | 1 / 32 |

Both exact crossing probabilities are at most the declared \(\alpha=1/2\). This is a code/proof arithmetic check for this tiny fixture; the deliberately loose threshold is not an operational target.

## 5. What this does and does not establish

It checks:
- exact rational portfolio-grid construction;
- the conditional expected-return inequality for every fixture state and portfolio;
- exact restart-prior plus residual capital conservation;
- exact path-probability mass and anytime threshold-crossing mass.

It does not check:
- how the upper envelope is derived from independent calibration data;
- one-sided Clopper–Pearson or multinomial confidence-bound numerical certification;
- familywise calibration coverage or the combined \(\alpha+\delta\) guarantee;
- context assignment, state discretizer integrity, missingness/admission exogeneity, trace lineage or PMU acquisition;
- unknown transition matrices, hidden states, order-2 dependence, regime switching, transition drift or adversarially controlled transitions;
- physical hardware performance, false alarms per device-hour, detector superiority or SOTA.

The next implementation must add a frozen, digest-bound calibration receipt and certified outward confidence bounds. The row-wise calibration protocol must specify a fixed number of visits per source state (or use a separately qualified time-uniform calibration confidence sequence); random stopping and adaptive admission must not be silently treated as fixed-sample binomial sampling. The observation-admission decision must be predictable before the next-state outcome, or missingness must be modeled as an outcome.

## References

- Liu, Prinster & Orabona (2026), *Monitoring for Shifts at Unknown Times: Parameter-Free Conditional Conformal Martingales*: https://arxiv.org/abs/2610.17343
- Kueffner et al. (2026), *Confidence Sequences for Online Statistical Model Checking of Markov Decision Processes*: https://arxiv.org/abs/2606.25797
- HARDSEC-024: https://github.com/Luminous-Dynamics/symthaea/pull/9679
