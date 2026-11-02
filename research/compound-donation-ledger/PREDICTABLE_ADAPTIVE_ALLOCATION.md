# Predictable adaptive allocation: statistical admissibility versus horizon projectivity

## Executive finding

The next design decision should not force every router to be both maximally adaptive and exactly horizon-projective. These are different objectives.

A predictable online recipient allocator can select where a donor sends its new reserve increment using past recipient utility. If the shares are nonnegative and sum to at most one, then the donor cap, own-increment cap and forward-prefix budget remain unchanged. Under the existing valid-e-process and global-filtration assumptions, the same compound null-mass accounting argument remains available.

Such an allocator may nevertheless fail strong horizon projectivity: appending a higher-ID recipient with a different history can change the allocation to an older recipient. This is a coherence cost, not automatically a statistical-validity failure.

Keep two modes:

1. **Projective mode:** fixed global priors / prefix-local information; horizon-stable.
2. **Adaptive-allocation mode:** predictable use of currently visible recipient histories; valid per fixed online stream under the same filtration assumptions, but not guaranteed projective under retrospective horizon extension.

## 1. Pathwise theorem for time-varying routing shares

Fix a global candidate order and weights \(\gamma_i\ge0\), with \(\sum_i\gamma_i\le1\). Let \(X_{i,t}\) be a nonnegative, adapted, pathwise nondecreasing base e-process under its own null. The reserve process \(D_{i,t}\) is nondecreasing and satisfies

\[
0\le D_{i,t}\le\gamma_i\min(X_{i,t},1),\qquad
0\le\Delta D_{i,t}\le\gamma_i\Delta X_{i,t}.
\]

Before round \(t\)'s donor increment is observed, choose shares \(q_{i,j,t}\), for \(j>i\), such that

\[
q_{i,j,t}\ge0,\qquad
\sum_{j>i}q_{i,j,t}\le1,
\]

and the whole vector \(q_{i,\cdot,t}\) is \(\mathcal F_{t-1}\)-measurable with respect to the same global filtration that validates the base e-processes.

Allocate the new reserve by

\[
\Delta O_{i,j,t}=q_{i,j,t}\Delta D_{i,t}.
\]

Let \(O_{i,t}=\sum_{s\le t}\sum_{j>i}\Delta O_{i,j,s}\) be aggregate source withdrawal. Since each share vector has mass at most one,

\[
0\le O_{i,t}
\le\sum_{s\le t}\Delta D_{i,s}
=D_{i,t}
\le\gamma_i\min(X_{i,t},1).
\]

Likewise,

\[
0\le\Delta O_{i,t}
\le\Delta D_{i,t}
\le\gamma_i\Delta X_{i,t}.
\]

The route can change on every round. It may use the full past history of currently visible recipients. Neither the donor cap nor own-increment cap depends on which recipient receives the reserve.

For every credited unit, source \(i\) must be strictly earlier than recipient \(j\), and the token must be source-tagged and single-spend. For every finite prefix \([m]\),

\[
\sum_{i\le m}(I_{i,t}-O_{i,t})\le0.
\]

Every credit entering a recipient \(j\le m\) originated at a donor \(i<j\le m\); outflows from donors \(i\le m\) that go beyond the cut or remain escrowed only make the prefix balance more negative.

Define the weighted compound score

\[
Z_{i,t}=\gamma_iX_{i,t}+I_{i,t}-O_{i,t}.
\]

It is nonnegative because \(O_{i,t}\le\gamma_i\min(X_{i,t},1)\), and pathwise nondecreasing because

\[
\Delta Z_{i,t}
=\gamma_i\Delta X_{i,t}+\Delta I_{i,t}-\Delta O_{i,t}
\ge0.
\]

For true-null set \(N\subseteq[m]\) and alternatives \(A=[m]\setminus N\),

\[
\begin{aligned}
\sum_{i\in N}Z_{i,t}
&\le \sum_{i\in N}\gamma_iX_{i,t}
    +\sum_{i\in A}O_{i,t}\\
&\le \sum_{i\in N}\gamma_iX_{i,t}
    +\sum_{i\in A}\gamma_i.
\end{aligned}
\]

Taking expectations under the valid-null e-process assumptions gives the same compound null-mass bound used by #9894. The already-established e-BH / compound-e-value argument must still be invoked for the desired FDR or SupFDR statement. This is an allocation lemma; it is not a new general SupFDR theorem.

The predictability requirement is an operational guarantee that prevents current-round utility leakage. The pathwise inequalities themselves are deterministic once the shares are selected. Do not use this observation to relax the original global-filtration requirements.

## 2. Projectivity no-go witness

Suppose donor 1 can allocate to recipients 2 and 3.

- In a short universe \(\{1,2\}\), the router sees recipient 2 and selects \(q_{1,2,t}=1\).
- In an extended universe \(\{1,2,3\}\), the router observes high historical utility for recipient 3 and selects \(q_{1,3,t}=1\).

Let the reserve increment be \(a>0\). In both runs donor 1's total outflow is exactly \(a\), the donor cap is unchanged, and every transfer points forward. The prefix budget also holds in each run. Yet candidate 2 receives \(a\) in the short run and zero in the extended run. Thus its compound score differs and strong initial-segment projectivity fails.

This is a counterexample to requiring both (i) route adaptation to realized higher-ID recipient histories and (ii) exact old-prefix state equality for all horizons. A projective router must avoid that information, or route only using horizon-invariant metadata / a fixed prior. A statistically admissible adaptive router can instead accept non-projectivity and define its guarantee for each fixed online information flow.

Cut projection from #9991 remains necessary for tokens routed outside a finite prefix, but it does not fix this counterexample when an appended recipient competes with an old recipient that lies inside the prefix.

## 3. Prequential online allocation algorithms

Use recipient utility \(U_{j,t}\in[0,1]\), observed only after the round-t allocation. The route vector \(q_t\) must be computed from utility history through \(t-1\). This gives a clean full-information online-learning benchmark:

- Fixed telescoping prior: \(\rho_d=1/[d(d+1)]\).
- Fixed geometric prior: \(\rho_d=2^{-d}\).
- Nearest-only prior.
- Uniform over the current finite horizon (diagnostic and non-projective).
- Prior-weighted Hedge using cumulative past recipient utility.
- Fixed-share Hedge, which periodically mixes the updated weights back toward the original prior to adapt after regime shifts.
- Hindsight best fixed recipient and per-round best recipient as separate offline comparators, never as implementable policies.

For a prior \(p_j\), prior-weighted Hedge uses

\[
q_{j,t}
=\frac{p_j\exp(\eta\sum_{s<t}U_{j,s})}
       {\sum_kp_k\exp(\eta\sum_{s<t}U_{k,s})}.
\]

A fixed-share variant mixes the post-update distribution back with the fixed prior. Both produce nonnegative shares summing to one. They are predictable because current utility is revealed after choosing \(q_t\). They are not horizon-projective when the update reads histories for newly appended recipients.

The route choice determines capital allocation only. It must not change the base e-process construction, its filtration, or the global candidate weights. A utility model is not automatically a valid e-process.

## 4. Executed deterministic allocation benchmark

Committed runner: benchmarks/predictable_recipient_allocation.py.

Parameters:
- 32 recipient offsets;
- 100 rounds;
- seed 20261102;
- utility in [0,1];
- q_t uses only past utility;
- prior-Hedge learning rate \(\eta=\sqrt{8\log(33)/100}\);
- fixed-share alpha 0.05;
- exact-Fraction accounting screen for time-varying source shares.

Scenarios:
- nearest-recipient utility;
- mid-distance peak at offset 8;
- far-distance peak at offset 24;
- four regime switches;
- sparse high-utility recipients;
- diffuse random utility;
- smooth time-varying utility.

Representative results (mean utility per unit reserve):

| Scenario | Nearest | Telescope fixed | Geometric fixed | Uniform finite horizon | Prior Hedge | Fixed-share Hedge |
|---|---:|---:|---:|---:|---:|---:|
| Nearest recipient | 1.0000 | 0.6996 | 0.7791 | 0.1102 | 0.9853 | 0.9386 |
| Far-distance peak | 0.0101 | 0.0308 | 0.0129 | 0.2836 | 0.8742 | 0.7615 |
| Four-regime switching | 0.1780 | 0.1807 | 0.1970 | 0.1421 | 0.2318 | 0.4594 |
| Sparse opportunities | 0.1029 | 0.1444 | 0.1514 | 0.1755 | 0.8423 | 0.7401 |
| Diffuse random | 0.4599 | 0.4604 | 0.4730 | 0.5013 | 0.4835 | 0.4743 |

The fixed telescoping and geometric priors are projective because the remaining prior mass is retained in an outside-horizon sink, not renormalized using the visible family size. Uniform finite-horizon allocation and both adaptive policies are non-projective diagnostics.

The adaptive Hedge variants exploit the historical utility signal: prior-Hedge improves from 0.0308 to 0.8742 in the far-distance scenario, while fixed-share improves from 0.1807 to 0.4594 in the switching scenario. These numbers are entirely synthetic utility results; they are not discovery power, FDR, SupFDR or a claim that the allocation has better statistical validity.

This benchmark illustrates an important frontier:
- static projective priors give horizon coherence but cannot learn from realized recipient utility;
- adaptive predictable policies can learn and switch recipients but can change old-prefix state when the universe is extended;
- fixed-share policies trade some stationary performance for faster adaptation after regime changes.

## 5. Exact accounting stress test

The committed script also contains a separate Fraction-based check. It generates predictable share vectors before each donor's current evidence increment, allocates the source reserve only across strictly later IDs, and checks:

- nonnegative shares with total mass at most one; a zero-mass route leaves reserve unwithdrawn at the source;
- cumulative donor reserve and own-increment caps;
- prefix budgets after each round;
- exact equality of incoming credits plus escrow and total outgoing capital.

Default configuration: 1,000 traces, 8 donors, 25 rounds. A local execution of the same Fraction-based accounting logic completed 1,725,000 assertions. It checks the source and prefix inequalities, not an implementation test of the Rust writer or a statistical validity experiment. Re-run the committed script to reproduce the exact output.

## 6. Acceptance and benchmark programme

Before any statistical claim:

1. Implement the multi-recipient router in Rust with source-scoped token identities.
2. Preserve the fixed-destination writer and independent verifier as reference oracles.
3. Add adversarial tests for share mass >1, non-predictable route inputs, current-round utility lookahead, high-ID history lookahead, delayed recipients, partial token credit and attempted re-donation.
4. Replay every generated receipt with an independent linear verifier.
5. Prove the exact compound null-mass consequence using the operational global filtration.
6. Evaluate the statistical router against fixed priors on the same valid base-e-process streams, reporting FDR, SupFDR, stopped-FDR, FDX and P(any false) separately.
7. Report power/time-to-discovery, recipient concentration, worst-regime regret, reserve use, p50/p95/p99 latency, memory and verifier time.
8. Run at least 50,000 frozen paired streams per confirmatory statistical cell, with more for rare error-rate targets and simultaneous uncertainty over the prespecified grid.

## Claim boundary

The pathwise time-varying allocation lemma follows from the source cap and the simplex constraint on q. The no-go example shows why higher-ID recipient adaptation conflicts with strong horizon projectivity. The synthetic benchmark tests utility adaptation only. No Rust implementation, valid-null experiment, FDR/SupFDR result, new general theorem or SOTA claim is made.
