# HARDSEC-024 — Exact finite-state Markov conditional-rank oracle

**Status:** Small known-transition model-based oracle; not a PMU detector, not distribution-free, and not evidence that real PMU telemetry follows a Markov model. Tracked by [issue #9679](https://github.com/Luminous-Dynamics/symthaea/issues/9679).

## Purpose

HARDSEC-023's PRM uses the Pólya predictive rank law to handle dependence induced by reusing one finite calibration sample under iid calibration and online observations. It does not handle source-stream serial dependence. This example begins the separate HARDSEC-024 line with a fully observed finite-state Markov null whose conditional law is known exactly.

## Model

The state space and transition matrix are:

\[
\mathcal S=\{0,1\},\qquad
P=\begin{pmatrix}0.9&0.1\\0.2&0.8\end{pmatrix}.
\]

The next state is the rank category. The betting distribution is fixed and predictable:

\[
q=(1/2,1/2).
\]

At time \(t\), the current state \(X_{t-1}=i\) is observed. Therefore the exact conditional next-rank probabilities are \(p_{t,j}=P_{ij}\). The betting factor is

\[
E_t=\frac{q_{R_t}}{p_{t,R_t}}.
\]

For every current state i,

\[
\mathbb E[E_t\mid\mathcal F_{t-1}]
=\sum_{j=0}^1P_{ij}\frac{1/2}{P_{ij}}=1.
\]

Thus the product is a nonnegative martingale for this stated null model. The model is assumed known exactly; this is not distribution-free and does not establish arbitrary-PMU validity.

## Exact path enumeration

For each initial state, the executable enumerates all \(2^{12}=4096\) paths. Transition probabilities are rational with denominator 10. The wealth factor for an edge with transition numerator a is exactly \(5/a\), so the threshold test can be done with integer arithmetic rather than floating-point comparisons. The terminal path-probability denominator is \(10^{12}\).

| Initial state | Exact any-time crossing mass | Decimal | Alpha |
|---:|---:|---:|---:|
| 0 | 52,840,309,168 / 1,000,000,000,000 | 0.052840309168 | 0.10 |
| 1 | 55,866,239,952 / 1,000,000,000,000 | 0.055866239952 | 0.10 |

Both initial-state path distributions sum to exactly one in integer arithmetic. These probabilities are exact for the enumerated 12-step experiment, not estimated from Monte Carlo.

## Claim ceiling

The receipt states:

- known transition matrix: assumed true by construction;
- current state: fully observed;
- fixed bet: predictable;
- model-specific anytime PFA claim: valid for this finite-state null;
- arbitrary serial PMU PFA claim: false;
- unknown transition matrix and hidden-state support: false;
- physical-hardware evidence: false;
- effective sample size: null.

## Next step

For unknown transition matrices, HARDSEC-024 issue #9679 proposes a time-uniform confidence sequence \(C_t\) and predictable bin upper bounds \(u_{t,j}=\sup_{Q\in C_{t-1}}Q_{i_t,B_j}\). A proof using the analysis-only first confidence-set-exit time can allocate a detector budget alpha and confidence-sequence failure budget delta, giving an alpha+delta marginal bound, provided the confidence sequence is genuinely time-uniform and the conditional transition model is correctly specified. This requires its own theorem and finite-state exhaustive tests; it is not implemented here.

The next prototype should add:
1. exact integer confidence-sequence updates for the unknown 2-state transition row probabilities;
2. exhaustive path verification of the alpha+delta claim for every transition matrix in a small rational grid;
3. negative controls for model misspecification, selective admission, hidden states, and adversary-controlled transitions.
