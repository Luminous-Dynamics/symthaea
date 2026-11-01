# Bounded-envelope localization for random predictable arrivals

**Date:** 2026-11-01  
**Status:** proof sketch complete under fixed-candidate / finite-envelope assumptions; integration with the full dynamic closure implementation remains to be reviewed.

## 1. Problem and scope

Dynamic e-closure combines evolving evidence with a growing active hypothesis family. The 2026 dynamic e-closure paper gives stopped-FDR localization for deterministic active sets and identifies random predictable arrivals as a future direction. Here we consider a restricted case that covers ordinary serial arrival systems with random activation times:

- There is a deterministic universe \(\mathcal I\) of candidate hypotheses.
- The truth/null status of every fixed candidate is fixed under a data-generating law \(P\).
- The active roster \(I_t\) is finite, nested, and predictable: it is known before the round-t evidence is revealed.
- At every deterministic horizon \(T\), a deterministic finite envelope \(J_T\) contains every candidate that could have activated by time \(T\). The envelopes are nested in T.

For one candidate arriving at most per round under serial indexing, use \(J_T=\{1,\ldots,T\}\). The actual roster can be random: delayed activation, event-triggered scheduling, and selection from a finite predeclared candidate pool are allowed, provided the roster choice is predictable and stays inside the deterministic envelope.

This does not cover hypotheses whose mathematical definitions are themselves generated from observed data without a fixed candidate universe, or a data-dependent identifier set for which no deterministic finite \(J_T\) exists.

## 2. Assumptions

Let \((\mathcal F_t)\) be the global filtration. For each fixed finite nonempty intersection S, let \(E_t^S\) be a nonnegative test supermartingale under \(H_S\), in this same filtration. Suppose the collection is future-extension coherent:
\[
E_t^{S\cap I_t}\le E_t^S
\]
pathwise for every finite deterministic S and every t. Empty true-null intersections have zero FDP and need no certificate.

A predictable portfolio from #9848 satisfies fixed-intersection validity if all coordinate inputs are test supermartingales in the common filtration. Predictable random activation changes which cash/asset coordinates are active before the next observation, while neutral value-one padding gives exact future-horizon projectivity. Its per-intersection validity does not itself prove the closure theorem; the localization below supplies the random-index step.

## 3. Bounded-envelope stopped-FDR localization

Fix a data-generating law P and deterministic horizon T. Because the candidate universe and null configuration are fixed under P, the terminal envelope true-null set
\[
S_T:=\mathcal N(P)\cap J_T
\]
is deterministic. Let \(\tau\) be any global stopping time and \(\sigma_T=\tau\wedge T\). Since \(I_t\subseteq J_T\) for all \(t\le T\),
\[
\mathcal N(P)\cap I_{\sigma_T}=S_T\cap I_{\sigma_T}.
\]

Let \(\mathcal C_t\) be the dynamic-closure candidate family formed from the active intersections at time t. Its feasibility constraints imply, for every candidate \(R\in\mathcal C_t\),
\[
\operatorname{FDP}_{\mathcal N(P)\cap I_t}(R)
\le \alpha E_t^{\mathcal N(P)\cap I_t}.
\]
Future-extension coherence then gives
\[
\max_{R\in\mathcal C_{\sigma_T}}
\operatorname{FDP}_{\mathcal N(P)\cap I_{\sigma_T}}(R)
\le \alpha E_{\sigma_T}^{S_T}.
\]
If \(S_T=\varnothing\), the left side is zero. Otherwise \(P\in H_{S_T}\), so the right-hand process is one fixed nonnegative test supermartingale evaluated at a bounded stopping time. Optional stopping yields
\[
\mathbb E_P\left[
\max_{R\in\mathcal C_{\sigma_T}}
\operatorname{FDP}_{\mathcal N(P)\cap I_{\sigma_T}}(R)
\right]\le\alpha.
\]

If \(\tau<\infty\) almost surely, then \(\sigma_T=\tau\) eventually on each path. The loss therefore converges almost surely to the loss at \(\tau\). Fatou's lemma yields simultaneous stopped-FDR control at level \(\alpha\).

The key trick is that the random active true-null intersection is localized at each finite horizon by the deterministic envelope true-null set S_T. We do not need one single terminal set for all horizons.

## 4. SupFDR extension

Add the assumption that every fixed-intersection certificate is time-monotone. At a deterministic horizon T, for all t≤T, future-extension coherence bounds the active true-null loss by \(\alpha E_t^{S_T}\). Time-monotonicity gives
\[
\sup_{t\le T}\max_{R\in\mathcal C_t}
\operatorname{FDP}_{\mathcal N(P)\cap I_t}(R)
\le \alpha E_T^{S_T}.
\]
Taking expectations gives the finite-horizon SupFDR bound. Monotone convergence as T increases gives the infinite-horizon bound.

This does not make raw running maxima valid e-processes. Time-monotonicity must come from a separately valid persistent certificate / adjuster, and the adjusted family must preserve future-extension coherence.

## 5. Why the finite-envelope assumption matters

If the set of candidate definitions itself is generated from the data and, by a fixed horizon T, could be any of infinitely many unrelated labels, \(J_T\) may not be finite or deterministic. Then \(\mathcal N(P)\cap J_T\) is not a usable fixed finite intersection for optional stopping. This note does not claim to solve that harder adaptive-hypothesis-generation problem.

Similarly, predictable timing alone does not lift locally valid coordinate streams into the global filtration: every fixed-intersection supermartingale must already be valid relative to the filtration used by the stopping rule.

## 6. Executable set-relation checks

`random_arrival_localization.py` checks nested rosters, finite-horizon envelope coverage, and the identity
\[
\mathcal N(P)\cap I_t=(\mathcal N(P)\cap J_T)\cap I_t.
\]
The property tests include 250 seeded random arrival schedules from a fixed 12-candidate pool, staggered envelopes, empty-null cases, and negative controls. These validate finite-set bookkeeping only; predictability and supermartingale validity remain statistical/model assumptions verified by the event ledger and process proof.

## 7. Required next steps

1. Turn the proof sketch into a formally reviewed extension of de Heide's stopped-FDR localization theorem.
2. Prove the exact measurability of the random candidate-family closure for predictable finite rosters.
3. Integrate with #9848 and use the independent predictable-arrival ledger from #9809 to verify pre-observation roster decisions.
4. Implement the production path in Rust; keep Python as an independent reference and test oracle only.
5. Run full stopped-FDR benchmarks under random predictable activation with finite envelopes, then separately run SupFDR only with a valid projective time-monotone adjuster.

## References

- de Heide (2026), [Dynamic e-closure for online hypotheses with any-time-valid evidence](https://arxiv.org/abs/2608.09927), Definition 2.5, Lemma 3.4, Theorems 3.6–3.7 and Section 1.5.
- Xu, Fischer & Ramdas (2026), [Improving online FDR procedures via online analogs of e-closure and compound e-values](https://arxiv.org/abs/2603.24792).
- Choe & Ramdas (2026), [Combining evidence across filtrations](https://academic.oup.com/jrsssb/article/88/4/1492/8627076).