# HARDSEC-024 — Unknown transition rows: beta-mixture confidence-sequence audit

**Status:** Experimental finite-grid audit. The mathematical construction has a time-uniform coverage argument under a correctly specified, fully observed, fixed finite-state Markov chain. The Rust endpoint implementation uses padded floating-point bisection and is **not yet formally interval-certified**; this must not be promoted to a runtime sequential-PFA guarantee.

## 1. Why this is the next step

The known-transition oracle demonstrates the conditional betting identity for a fully observed Markov model. The practical problem is that the transition matrix is unknown. A point estimate of P cannot be inserted into q_j/p_hat_j and called an e-process.

This audit implements a row-wise confidence-sequence primitive for a two-state Markov chain. It is kept separate from the PRM implementation and keeps the unknown model and numeric-certification limitations explicit.

## 2. Beta-mixture confidence sequence

For one transition row, collect the observed destinations on visits to that source state. Let s be transitions to state 1 and f transitions to state 0. Under a fixed row probability p, the Bernoulli likelihood is L_t(p)=p^s(1-p)^f.

Mix the alternative likelihood over a uniform Beta(1,1) prior:

\[
m_t=\int_0^1 \theta^s(1-\theta)^f\,d\theta
=B(s+1,f+1)=\frac{s!f!}{(s+f+1)!}.
\]

For each fixed p, the ratio M_t(p)=m_t/L_t(p) is a nonnegative martingale under the Bernoulli(p) row-transition process. Inverting Ville's inequality gives:

\[
C_t(\delta_i)=
\left\{p\in[0,1]:
\frac{B(s+1,f+1)}{p^s(1-p)^f}<\frac{1}{\delta_i}
\right\},
\qquad
\Pr_p\{\exists t:p\notin C_t(\delta_i)\}\le\delta_i.
\]

Only transitions made from the corresponding source state update its row process; at other times that row's confidence sequence is unchanged. Under the Markov property, the row indicator remains a conditional Bernoulli(p) observation at each visit, so the mixture process is a martingale in global time. For two rows, choosing delta_0=delta_1=delta/2 and applying a union bound yields simultaneous time-uniform row coverage at least 1-delta.

## 3. Predictable conditional-probability envelope

Before the next transition from source state i, use the already-computed row interval [L_(t-1,i),U_(t-1,i)]. The predictable upper bounds are:

\[
u_{t,1}=U_{t-1,i},\qquad
u_{t,0}=1-L_{t-1,i}.
\]

On the simultaneous coverage event, the true conditional probabilities satisfy p_(t,j) <= u_(t,j) at every time.

For any predictable betting distribution q_t on the two outcomes, the factor q_(t,R_t)/u_(t,R_t) has conditional expectation at most one on that event:

\[
\sum_j p_{t,j}\frac{q_{t,j}}{u_{t,j}}
\le\sum_jq_{t,j}=1.
\]

Let tau be the analysis-only first time that the true transition row exits its confidence sequence. Stop the proof process at tau by using factor 1 afterward. Ville gives probability at most alpha of crossing before tau, while the confidence-sequence failure event has probability at most delta. Therefore the actual non-stopped monitor's total crossing probability is bounded by alpha+delta. The monitor does not observe tau; this is a proof device, and any crossing after tau is included in the delta event.

The alpha and delta budgets must be declared separately. A fixed-time interval, plug-in estimate, empirical ACF, block bootstrap, or estimated effective sample size is not a replacement for this time-uniform confidence sequence.

## 4. Executable finite-grid audit

The example examples/hardsec_markov_transition_cs_audit.rs builds Beta-mixture row intervals, then exhaustively enumerates all paths for:

- 16 rational two-row transition matrices with row probabilities in {1/5, 2/5, 3/5, 4/5};
- both initial states;
- horizon 8;
- 256 paths per episode;
- exact integer path-probability numerators and denominator 5^8.

For every episode it verifies total path mass exactly one and computes the exact probability that either row confidence sequence excludes its true grid parameter at any time. This is a finite grid diagnostic of the implementation, not a universal proof. It also checks that the upper envelope dominates each grid conditional outcome probability whenever the true row parameter remains inside the interval.

The implementation uses 96-step bisection and 64-ULP endpoint padding. That is a conservative engineering attempt, not a machine-checked outward-rounding proof for platform libm. Receipt field floating_point_runtime_claim therefore remains not_established_until_outward_interval_arithmetic_is_certified. Before runtime PFA claims, use directed-rounding interval arithmetic or rational root bracketing that provides certified outer endpoints.

## 5. Assumptions and negative controls

The theorem requires:
- fully observed states and a correctly specified first-order Markov chain;
- fixed but unknown transition-row probabilities;
- no adversarial changes to the transition law within the claimed episode;
- every attempted transition is accounted for (no selective admission or silent drops);
- predictable context/row identity;
- correct Beta-mixture CS computation and outward endpoint enclosures.

Unsupported claim classes remain false: arbitrary serial PMU process, hidden-state process, misspecified Markov model, selective sensor admission, and effective sample size.

Required next controls include hidden Markov observations, regime switching, attacker-driven transitions, adaptive missingness, state-identity errors, and repeated/overlapping PMU windows. These should fail the theorem-valid eligibility audit rather than be silently coerced to the Markov class.

## References

- Kueffner, Meggendorfer, Weininger and Wienhöft (2026), https://arxiv.org/abs/2606.25797. The work studies sequential confidence sequences for unknown MDP transitions and reports a 50x sample-efficiency improvement over its evaluated previous pipeline; that is a model-checking result, not a PMU detector comparison.
- Howard et al., https://doi.org/10.1214/20-AOS1991.
- Barber and Pananjady (2026), https://proceedings.mlr.press/v313/barber26a.html. Their dependence-switch-coefficient results concern prediction coverage, not a general sequential e-process for PMU telemetry.
