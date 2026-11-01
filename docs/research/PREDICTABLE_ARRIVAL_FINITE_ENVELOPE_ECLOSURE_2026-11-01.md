# Predictable random arrivals with finite-envelope dynamic e-closure

**Status:** exact finite-catalogue research prototype plus a derived finite-envelope localization argument. It is not a proof for an unrestricted open-ended random catalogue, and the source e-process contracts remain external.

## 1. Research gap

Dynamic e-closure studies two clocks: hypotheses may arrive while evidence for existing hypotheses continues to evolve. Its stopped-FDR and SupFDR results rely on future-extension coherence, with time-monotonicity added for SupFDR. The paper's baseline theory takes deterministic finite active sets and explicitly leaves random predictable arrivals for future work because the active true-null intersection at a fixed horizon is random.

The missing step is not simply allowing the active mask to vary. We need to remove a data-dependent intersection index from the optional-stopping argument without selecting among e-processes after seeing their evidence.

This branch isolates one useful bounded case: at every finite horizon, all possible arrivals belong to a deterministic finite candidate envelope fixed before the evidence stream. The realised arrival schedule may still be random and predictable.

## 2. Finite-envelope localization proposition

Let \((\mathcal F_t)\) be one global filtration. Let \(I_t\) be a nested, finite, predictable active set, and suppose that for every deterministic finite horizon \(T\), there is a deterministic finite set \(U_T\) such that

\[
  \Pr(I_t\subseteq U_T\text{ for all }t\le T)=1.
\]

Let \(\mathbf E=(E_t^S)\) be a future-extension-coherent dynamic intersection e-process collection for every fixed finite intersection \(S\). Write \(\mathcal N(P)\) for the fixed set of true-null hypotheses under distribution \(P\), and define

\[
  S_T(P):=\mathcal N(P)\cap U_T.
\]

Although the realised set \(\mathcal N(P)\cap I_t\) is random, \(S_T(P)\) is deterministic under fixed \(P\), and

\[
  \mathcal N(P)\cap I_t=S_T(P)\cap I_t
  \qquad (t\le T).
\]

By future-extension coherence, pathwise,

\[
  E_t^{\mathcal N(P)\cap I_t}
  =E_t^{S_T(P)\cap I_t}
  \le E_t^{S_T(P)}.
\]

The right-hand side is one fixed e-process. This removes the random intersection index without selecting an intersection certificate using the observed path.

### Stopped-FDR conclusion

For a stopping time \(\tau\le T\), every reported \(R\subseteq I_\tau\) satisfies the closure inequality

\[
  \operatorname{FDP}_{\mathcal N(P)}(R)
  =\operatorname{FDP}_{\mathcal N(P)\cap I_\tau}(R)
  \le \alpha E_\tau^{S_T(P)}.
\]

Taking the maximum over all certified \(R\), then expectations and optional stopping for the fixed \(S_T(P)\) process, gives simultaneous stopped-FDR at level \(\alpha\). For an almost-surely finite unbounded \(\tau\), apply the bounded result to \(\tau\wedge T\), use nonnegativity on \(\{\tau\le T\}\), and pass to the limit by monotone convergence.

If \(S_T(P)=\varnothing\), then no true null can be active by time \(T\), so the loss is zero and no intersection process is needed.

### SupFDR conclusion

If every fixed-intersection certificate is also time-monotone, then for every \(t\le T\),

\[
 \max_{R\in\mathfrak R_{\alpha,t}}
 \operatorname{FDP}_{\mathcal N(P)}(R)
 \le \alpha E_t^{S_T(P)}
 \le \alpha E_T^{S_T(P)}.
\]

Taking the supremum over \(t\le T\) and expectations gives the finite-horizon SupFDR bound. Letting \(T\to\infty\) and applying monotone convergence gives the infinite-horizon bound.

**Scope:** this is a finite-envelope localization result. It does not solve arbitrary random arrivals from an unbounded catalogue with no deterministic finite horizon envelope. The proof assumes the dynamic closure constraints hold pathwise for the realised active family and that the fixed-intersection processes are valid in the common filtration.

## 3. Predictable self-financing construction

For every coordinate \(i\), let \(X_{i,t}\ge0\) be its one-step evidence factor and assume, under \(H_i\),

\[
  \mathbb E[X_{i,t}\mid\mathcal F_{t-1}]\le1.
\]

The arrival set \(I_t\) and portfolio shares \(p_{i,t}\) must be \(\mathcal F_{t-1}\)-measurable, with \(p_{i,t}=0\) for inactive coordinates and \(\sum_i p_{i,t}\le1\). For a fixed intersection \(S\), define

\[
 E_0^S=1,\qquad
 E_t^S=E_{t-1}^S
 \left(
  1-\sum_{i\in S\cap I_t}p_{i,t}
  +\sum_{i\in S\cap I_t}p_{i,t}X_{i,t}
 \right).
\]

The factor is nonnegative. Under \(H_S\), predictability and the coordinate conditional bounds imply

\[
 \mathbb E[G_t^S\mid\mathcal F_{t-1}]
 =1-\sum_{i\in S\cap I_t}p_{i,t}
  +\sum_{i\in S\cap I_t}p_{i,t}
    \mathbb E[X_{i,t}\mid\mathcal F_{t-1}]
 \le1.
\]

Thus every fixed-intersection product is a test supermartingale. Since the active sets are nested and the same global shares are used for every intersection, neutral padding is exact:

\[
  E_t^{S'}=E_t^{S'\cap I_t}.
\]

A common valid running-maximum adjuster makes the intersection collection time-monotone. Together with the finite-envelope localization proposition, this yields the stated finite-envelope SupFDR scope.

This conditional-expectation derivation is mathematical, but the code cannot infer the conditional expectation contract from a realised path. Evidence producers must establish it separately.

## 4. Event-ledger contract

The new validator, validation/predictable_arrival_eclosure.py, adds a canonical event ledger with two event kinds:

- activation_commit: commits a new hypothesis against the digest of the filtration-visible state before the current round.
- evidence_increment: records one nonnegative one-step factor for an already committed active hypothesis.

Within each round, all activation commits must precede every evidence increment. Every active coordinate must then have exactly one increment event; use factor one when no new evidence is observed. Sequence numbers are contiguous, rounds are ordered, and all IDs must belong to the predeclared finite catalogue. The catalogue hash binds IDs, priority scores, utilities and costs. State digests bind each activation to prior activations and prior evidence.

The ledger audit detects activation after current-round evidence, future-state or stale-state commitment digests, unknown or duplicate arrivals, increments for inactive coordinates, missing or duplicate active-coordinate increments, event reordering and malformed round horizons.

**Security boundary:** this is a structural and hash-chain audit, not a trusted timestamp, signature, or append-only storage service. A production deployment needs signed activation receipts, a trusted sequencer or append-only/WORM log, explicit clock semantics, and an external attestation that the catalogue was fixed before the stream began. A hash proves binding to supplied bytes; it does not prove when those bytes were created.

## 5. Exact fixture

The finite catalogue is h1, h2, h3, but only h1 and then h2 arrive. h2 is activated in round 2, after round-1 evidence for h1 has been observed. Its activation receipt binds to that exact pre-round state. With the all-success path:

- round 0: activate h1; factor \(X_{1,0}=1\);
- round 1: no new arrival; factor \(X_{1,1}=8\);
- round 2: commit h2; factors \(X_{1,2}=1\), \(X_{2,2}=16\).

Equal-active shares, \(\alpha=1/2\), adjuster thresholds \((2,4,8,16,32,64)\), unit adjuster increments, utilities \((10,8,1)\), costs \((1,1,1)\), and cumulative capacities \((0,1,2)\) give nested actions

\[
  \varnothing,\quad \{h1\},\quad \{h1,h2\}.
\]

The fixture checks pathwise receipts and exact optimizer replay; it is not a Monte Carlo estimate of FDR or a proof of the evidence generator's conditional law.

## 6. Negative controls and tests

The test module, validation/test_predictable_arrival_eclosure.py, covers a later arrival committed after earlier evidence; state-digest sensitivity to prior evidence; activation-after-evidence rejection; stale/tampered commitment rejection; arrivals outside the deterministic finite envelope; missing, duplicate and inactive-coordinate increments; contiguous sequence and fixed-horizon enforcement; result receipt tampering; prefix invariance under changes to future evidence; and exact base-selection feasibility and utility optimality replay.

Run from the repository root:

    PYTHONPATH=validation python -m unittest -v test_predictable_arrival_eclosure.py

The committed test suite has not been executed in this environment. The implementation is a small-family exact oracle and inherits the exponential action search of the wrapped selector. No CI pass is claimed.

## 7. Research comparisons and benchmarks

Compare on matched guarantees and the same evidence generators:

1. deterministic-arrival dynamic projective closure;
2. this finite-envelope predictable-arrival ledger;
3. fixed globally summable projective mergers;
4. global active-universe self-financing shares;
5. intersection-local AWAIRE linear, quadratic and largest-past-wealth mixtures;
6. online e-closure / compound e-values for one-final-e-value-per-arrival streams;
7. adjusted running-maximum closure for persistent SupFDR;
8. ordinary e-BH, Anchored e-BH and closed e-BH for fixed-family batch controls.

Do not rank methods by power if one method controls stopped FDR and another controls SupFDR. Report stopped FDR, SupFDR, persistence, power and utility separately.

### Statistical stress matrix

Use all-null, sparse, moderate and dense alternatives; independent, common-shock, block-correlated and adversarially dependent increments; predictable arrivals driven by past evidence, common covariates, resource availability and randomized pre-evidence policies; rare-large-jump null martingales and near-unit factors; delayed or absent increments (represented by one); adaptive stopping rules using the full global history; finite-envelope sizes 2, 4, 8, 16 and 32; catalogue misspecification and deliberately post-evidence activation as negative controls.

Use exact finite-state enumeration where possible. Then run at least 5,000 exploratory streams and 50,000 frozen confirmatory streams with fixed seeds and a preregistered analysis. Simulation is a diagnostic for implementation and power, never the source of the validity guarantee.

### Operational scorecard

Record true actionable discoveries, false investigations, precision@B and recall@B; utility and utility regret against exhaustive exact optimization; cumulative capacity utilization, queue delay and time-to-certification; activation-to-first-evidence latency; p50/p95 decision and verifier latency; peak memory, catalogue size, active-set size and frontier width; commitment-verification failures, rejected late activations and audit overhead.

Industry baselines must use the same evidence contract and error guarantee. A model confidence or severity score is not an e-value merely because it is calibrated as a probability; the upstream evidence producer must supply a valid conditional e-process.

## 8. Path to a stronger result

The next mathematical step is removing the deterministic finite-envelope assumption. Candidate directions include a countable deterministic catalogue with a summable horizon-localization penalty, or a predictable random finite catalogue controlled by a separate arrival-selection e-process. Either would need a new proof; neither is implemented or claimed here.

The next systems step is to make the event ledger independently verifiable across processes: signed canonical activation receipts, append-only sequencing, catalogue commitment before the first observation, and a Rust verifier that reconstructs the exact active masks and portfolio decisions.

## References

- de Heide (2026), [Dynamic e-closure for online hypotheses with any-time-valid evidence](https://arxiv.org/abs/2608.09927), especially Sections 2–3 and Appendix A.3–A.5.
- Xu, Fischer & Ramdas (2026), [Improving online FDR procedures via online analogs of e-closure and compound e-values](https://proceedings.mlr.press/v337/xu26a.html).
- Tavyrikov, Goeman & de Heide (2026), [Carefree multiple testing with e-processes](https://doi.org/10.1214/26-EJS2546).
- Ek, Stark, Stuckey & Vukcevic (2023), [Adaptively Weighted Audits of Instant-Runoff Voting Elections: AWAIRE](https://arxiv.org/abs/2307.10972).
