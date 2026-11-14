# AIRROAD-MIMO-FINITE-SAMPLE-FRF-001 — excitation rank, aliasing and covariance-claim admissibility

**Status:** V0.1 exact-rational experiment-design reference  
**Tracking issue:** [#11988](https://github.com/Luminous-Dynamics/symthaea/issues/11988)  
**Research baseline:** González et al. (2025), [Finite Sample MIMO System Identification with Multisine Excitation](https://arxiv.org/abs/2510.26929)

> This reference checks the hypotheses required by a finite-sample MIMO multisine FRF estimator. It is not itself an FRF estimator or a physical uncertainty certificate.

## 1. Source-backed theorem boundary

The paper uses a multisine-equivalent discrete-time model with frequency lines (0,\pm\omega_1,\ldots,\pm\omega_M).

Its admissibility obligations are distinct:

1. **Experiment count:** (m\ge n_u).
2. **Input linear independence:** every excitation matrix (A_0,A_1,\ldots,A_M\) has full column rank.
3. **No overlapping aliased input frequencies:** the residues (0,\pm q_1,\ldots,\pm q_M\) must be distinct modulo one, where (q_\ell=\omega_\ell h/(2\pi)).
4. **Stationarity:** each experiment output is in the declared stationary regime.
5. **Sample count:** (N>2M).

The first four plus sample count are conditions for the paper's LS estimator to be well-defined and unbiased at the excited frequencies (Theorem 1). The covariance theorem (Theorem 2) gives a closed-form finite-sample covariance. Its additional no-spectral-leakage condition makes the positive/negative frequency estimates mutually uncorrelated. This is a stronger and separate gate; it must not be silently inferred from Theorem 1's unbiasedness conditions.

The no-overlap condition is equivalent to nonsingularity of the paper's Vandermonde matrix whose nodes are (1,e^{+i\omega_\ell h},e^{-i\omega_\ell h}). A Nyquist-rate condition is sufficient, but the paper also allows slow-sampled designs provided aliased frequency lines do not overlap.

## 2. Exact-rational V0.1 fixture

Use (n_u=2), (m=2), (M=2), and (N=15).

The normalized frequencies in cycles per sample are:
[
q_1=\frac15,\qquad q_2=\frac13.
]

The signed alias residues modulo one are:
[
\left\{0,\frac15,\frac45,\frac13,\frac23\right\}.
]
They are pairwise distinct. The Vandermonde determinant is therefore nonzero.

The exact excitation matrices are:
[
A_0=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
A_1=A_2=\begin{bmatrix}\frac12&\frac12\\\frac12&-\frac12\end{bmatrix}.
]
Their determinants are respectively (1,-1/2,-1/2). All have full column rank. For (A_1,A_2), the two experiments use nonzero amplitudes in both input channels, with phases 0 and \(\pi\) represented by signs.

The sample count satisfies (15>2M=4). For the separate no-spectral-leakage gate:
[
Nq_1=3,\qquad Nq_2=5,
]
both integers. Thus the record spans whole periods of both input lines.

V0.1 returns:
- admissible for excited-line LS unbiasedness: true, under the stipulated stationarity and model assumptions;
- eligible for Theorem 2 covariance: true;
- eligible for the mutually-uncorrelated-frequency consequence: true;
- continuous-frequency model adequacy: NOT_ESTABLISHED.

## 3. Exact aliasing rule

For each rational (q_\ell), form:
[
r_{\ell,+}=q_\ell\bmod1,\qquad
r_{\ell,-}=(-q_\ell)\bmod1.
]

Include the DC residue 0. Reject the no-overlap hypothesis if any residue is repeated. Examples:
- (q_1=1/5,q_2=4/5): positive line 2 collides with the negative alias of line 1.
- (q_1=1/2): the positive and negative aliases coincide at Nyquist.

Do this with exact rational modulo arithmetic, not floating tolerances.

## 4. Claim separation

The machine-readable receipt must keep these statuses separate:

- input design rank;
- alias non-overlap;
- sample-count eligibility;
- stationarity declaration;
- LS well-posedness/unbiasedness eligibility;
- finite-sample covariance eligibility;
- no-spectral-leakage / cross-frequency decorrelation eligibility;
- continuous-frequency adequacy.

A full-rank excitation and non-overlapping aliases do not bound unmeasured between-knot dynamics. A valid knotwise FRF estimate is not a full-band robust uncertainty model. That handoff remains with #11882/#11885/#11893.

The exact theorem covariance is statistical and depends on its noise assumptions. It is not a deterministic hard set. Repeat-period covariance estimation and shared campaign-fixed calibration are handled separately by #11986 and #11914.

## 5. Benchmark ladder

V0.1 checks the exact rational known-answer case and adversarial controls. Later matched tests should compare:
- orthogonal amplitude/phase designs;
- random-phase full-rank designs;
- D-optimal / condition-number-minimizing excitation;
- the paper's multisine baseline under identical energy and duration;
- below-Nyquist and permissible slow-sampled/non-overlapping cases;
- aliased collision, leakage and transient/stationarity violations.

Report experiment count, sample count, frequency lines, exact rank, minimum singular value/condition number where numerical, alias-residue separation, covariance structure and measurement time separately.

## 6. Claim ceiling

V0.1 is an exact-rational experiment-admissibility reference only. It does not compute the FRF, reproduce the full finite-sample estimator/covariance theorem, establish physical stationarity or white-noise assumptions, qualify physical measurements, bound dynamics between frequencies, solve an IQC, or establish aircraft robust stability/airworthiness.
