# PHOT-EVAL-006: verified incumbent-and-bound trajectories for maximum independent set

**Status:** research protocol draft; not an optimization result.  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8826  
**Depends on:** [PHOT-EVAL-005A verified anytime-quality trace](PHOT-EVAL-005A-VERIFIED-ANYTIME-QUALITY-PROTOCOL.md)  
**Related confirmation protocol:** [PHOT-CTRL-003 held-out MIS ablation](PHOT-CTRL-003A-HELDOUT-ABLATION-PROTOCOL.md)

## Purpose

A verified feasible independent set proves a lower bound on the optimum, not optimality. A solver can find a high-quality incumbent early and still have no defensible estimate of how much better the optimum could be. Conversely, a method may spend meaningful time producing evidence that closes an optimality gap.

PHOT-EVAL-005A's verified incumbent-quality curve remains valuable and unchanged. This protocol adds a separate upper-bound evidence stream, allowing reports to distinguish incumbent quality, uncertainty about the optimum, and a verified proof of optimality. It must not replace the primary failure-inclusive verified time-to-target endpoint.

Scope is **unweighted maximum independent set (MIS)** on the exact canonical graph bytes. Weighted MIS needs its own versioned objective, arithmetic, bound certificates, and metric.

## Mathematical contract

Let G=(V,E), n=|V|, and alpha(G) be the (unknown to the evaluator in general) maximum independent-set cardinality. Maintain two step functions over deployment elapsed time t:

- L(t): greatest independently verified feasible independent-set cardinality available by t. Initialize L(0)=0; the empty set is always feasible.
- U(t): least independently verified upper bound on alpha(G) available by t. Initialize U(0)=n, the trivial bound.

For every event and every interval between events, require:

0 <= L(t) <= alpha(G) <= U(t) <= n.

The evaluator does not need to know alpha(G) to check every part of this invariant: it independently checks each lower-bound witness for feasibility and each upper-bound certificate under a pinned sound verifier, and it checks L(t) <= U(t). On tiny reference instances, exhaustive enumeration additionally checks the result against the known exact alpha(G).

A lower-bound update can only increase L. An upper-bound update can only decrease U. Invalid, missing, untrusted, or post-horizon evidence never improves the within-horizon curve. If L(t)>U(t), reject the trace as contradictory evidence; do not clamp values, repair the trace, or award a score.

For a frozen horizon H>0, report the following independently:

1. **Verified incumbent-quality AUC** from PHOT-EVAL-005A, unchanged.
2. **Normalized certified-gap AUC**, lower is better:

   A_gap(H) = (1/H) integral from 0 to H of (U(t)-L(t))/n dt.

3. **Time to certified optimality**, the first elapsed time t for which L(t)=U(t), if that equality follows from independently verified evidence. Otherwise report “not certified within H”; never substitute a timeout or a best-known objective for a proof.
4. **Certified-optimality-at-horizon**: whether L(H)=U(H), plus the final certified interval [L(H), U(H)].
5. **Evidence and cost accounting**: number and type of attempted/accepted/rejected certificates, generation time, bytes persisted/transferred, independent-verification time, all-in elapsed time, terminal outcome, and the reason for every rejection. Values that cannot be measured must be marked unknown, not estimated silently.

The gap curve is a certificate-quality diagnostic, not a substitute for final objective quality. If U(t)=n for the whole run, then A_gap(H)=1-A_Q(H), where A_Q is the PHOT-EVAL-005A normalized incumbent-quality AUC. A meaningful new signal appears when independent certificates tighten U below n or close the gap; the additional machinery is not justified merely by renaming the existing curve.

## First upper-bound certificate: canonical clique partition

The first MVP upper-bound type should be a versioned **CliquePartitionV1** certificate against the original graph. It is a partition of all vertices into nonempty blocks, with each block certified to be a clique in G:

- Every vertex ID is in range.
- Every vertex appears in exactly one block.
- Every pair of distinct vertices within a block is an edge in the exact input graph.
- The certificate identifies the SHA-256 of the canonical graph bytes, the canonical encoding version, and the certificate type/version.
- The independent verifier recomputes the partition validity and sets U to the number of blocks. It must not trust a caller-supplied numeric bound.

Every independent set contains at most one vertex from each clique in the partition, so the number of blocks is a valid upper bound on alpha(G). A singleton partition gives the deliberately weak U=n baseline. A smaller verified clique partition can tighten the bound. The method must use only the graph and its operational inputs; known optima are permitted only in clearly labeled reference tests, never as operational evidence.

Additional upper-bound certificate types (for example, proof logs from exact solvers) must declare their own versioned format and independently checkable verifier. A solver-reported objective bound without a verifiable certificate is diagnostic metadata only and cannot update U(t).

## Event and timing semantics

- Bind the trace to the protocol digest, exact canonical graph digest, trial and pair IDs, seed, solver source/configuration identities, upper-bound verifier identity/build digest, environment, budget, horizon, and the same clock boundary declared by PHOT-EVAL-005A.
- An event becomes eligible only when its exact artifact bytes are persisted and the independent verifier has accepted them. Its elapsed timestamp must represent that boundary on a monotonic clock.
- Preserve proposal/generation, transfer, persistence, and verification times as separate fields when measured. The primary curve timestamp is when the evidence has become available and independently verified; do not retroactively credit solve time before verification.
- Reject non-finite, negative, decreasing, or beyond-contract timestamps; duplicate event identities; unsafe artifact paths; digest/byte mismatches; malformed certificates; graph or verifier substitutions; bound values outside [0,n]; and contradictory L>U states.
- Group same-time events and update L and U from all accepted events at that timestamp before integrating the next interval. Event order at identical timestamps must not change any score.
- Events verified after H may be retained in the raw all-events report but must not improve any within-horizon metric.

This boundary does not authenticate a compromised host or prove that the authoring process honestly measured elapsed time. Trusted-runner signatures or attested execution are a separate threat-model decision; a valid hash alone is not an attestation.

## Worked reference calculation

For a graph with n=6 and horizon H=600 ms, suppose:
- At t=0: L=0, U=6.
- At t=100 ms: independently verified incumbent L=2 and clique-partition certificate U=4.
- At t=300 ms: further verified evidence establishes L=3 and U=3.

Then the normalized gap is 1 on [0,100), 2/6 on [100,300), and 0 on [300,600]. Therefore:

A_gap(600) = (100*1 + 200*(2/6) + 300*0) / 600 = 5/18 ≈ 0.2777778.

The first certified-optimality time is 300 ms. The reference fixture must test the intermediate state and event-time semantics; a separate exact tiny-graph check must verify that the true optimum is 3. This arithmetic fixture does not count as experimental data.

## Benchmark protocol

### Cohorts and arms

- Tiny exact cohort: exhaustively enumerate all subsets for small graphs and validate every accepted L/U state against alpha(G).
- Structural cohort: include empty-edge, complete, disjoint-clique, bipartite, sparse, dense, and adversarially malformed certificate fixtures.
- Held-out evaluation: use the frozen QOBLIB MIS instances and separate graph-family cohorts already registered by PHOT-CTRL-003. Retain family-level holdouts where feasible; never tune certificates, schedules, or policy thresholds on confirmation families.
- Compare incumbent-only operation with the same solver plus certificate generation and independent checking. Include the strongest applicable pinned classical MIS portfolio and eligible exact solver arms; keep simulated/replayed/QPU evidence classes distinct.

### Required outputs

- Raw accepted and rejected lower/upper evidence events and their digests.
- Incumbent AUC, normalized gap AUC, time to certified optimality, final [L,U], fraction of trials certified optimal by H, failure-inclusive target result, and terminal outcome.
- Paired per-instance differences, graph-family stratification, win/tie/loss counts, effect sizes, and uncertainty under the preregistered cluster-aware analysis plan.
- All-in CPU/GPU/QPU time, measurement or trajectory costs, preprocessing, acquisition, transfer, certificate production and independent verification, retries, and energy/money only when actually measured at a matched boundary.
- An explicit “unknown/not measured” for any missing cost component.

### Fail-closed and falsification tests

At minimum, positive and negative tests must cover:
- Valid partitions; singleton partition producing U=n; cliques of different sizes; complete vertex coverage.
- Duplicate or missing vertices, empty blocks, out-of-range vertices, non-clique blocks, invalid edge lookup, graph digest mismatch, tampered certificate bytes, incorrect reported block count, and unsupported certificate versions.
- Duplicate event IDs, conflicting artifacts at one path, path traversal/symlink escape, invalid digests, non-finite/decreasing timestamps, same-time order invariance, exactly-at-horizon and post-horizon certificates.
- Bounds outside [0,n], non-monotonic purported best bounds, L>U contradictions, and correct handling of missing outputs, crash, abort and timeout.
- Hand-calculated gap AUC and exhaustive optimum comparison for every tiny graph fixture.
- Deterministic replay under fixed seeds and no dependence on filesystem iteration order or wall-clock randomness.

## Promotion and stop rules

This metric is qualified only after schema validation, independent certificate verification, exact-head tests, and a held-out preregistered campaign. CI passing proves the tested implementation behaves as tested; it does not establish solver performance.

Promote an upper-bound arm only if it improves certified-gap or proof-time outcomes enough to justify its measured all-in cost, without degrading the primary verified target-hit endpoint beyond a predeclared noninferiority margin. If a stronger certificate is too expensive, too rarely tightens U, or adds no held-out value over the incumbent-only baseline, retain the negative result and do not promote it.

No single scalar should collapse incumbent quality, target success, proof status, cost, and energy into an opaque leaderboard score.

## Claim boundaries and references

This protocol defines evidence and metrics, not a result. It makes no SOTA, speedup, quantum-advantage, or hardware claim.

- PHOT-EVAL-005A verified incumbent-quality curve: https://github.com/Luminous-Dynamics/symthaea/issues/8796
- PHOT-CTRL-003 held-out strong-baseline ablation: https://github.com/Luminous-Dynamics/symthaea/issues/8753
- The Quantum Optimization Benchmarking Library, including its model-independent comparison rationale and MIS references (Nature Computational Science, 2026): https://doi.org/10.1038/s43588-026-00991-1
- qReduMIS, quantum-informed reductions combined with classical kernelization (Physical Review Research, 2026): https://doi.org/10.1103/nh3b-1wv5
