# Capacity-aware selection inside online e-closure: exact small-stream oracle

**Status:** theorem-constrained research prototype. The optimizer is exhaustive and deliberately small. No production scalability or asynchronous e-process claim.

## Research idea

Xu, Fischer & Ramdas (2026), *Improving online FDR procedures via online analogs of e-closure and compound e-values*, establish an online e-closure theorem. Under their filtration and increasing e-collection assumptions, every measurable sequence of rejection sets \(R_t\) selected from the nested certified menus \(\mathcal C_t\) inherits the theorem's SupFDR guarantee.

This yields a useful separation of concerns:

1. The statistical procedure constructs the valid, nested menu \(\mathcal C_t\).
2. An operational policy chooses a useful action from that menu.
3. An independent verifier checks menu membership, resource feasibility, nesting, optimality for the encoded objective, and provenance.

The policy need not invent a new statistical guarantee if it always selects from the theorem-certified menu. This is an application/corollary of the existing theorem, not a new SupFDR theorem.

## Explicit increasing e-collection

For each finite intersection \(S\), the prototype uses

\[
E_S=\sum_{i\in S}\gamma_{|S\cap[i]|}E_i,
\qquad
\gamma_j\geq0,\quad\sum_{j\geq1}\gamma_j\leq1.
\]

For a rejection set \(R\subseteq[t]\), define

\[
\operatorname{FDP}_S(R)=
\begin{cases}
|R\cap S|/|R|,& |R|>0,\\
0,& |R|=0.
\end{cases}
\]

The set is in the certified menu exactly when

\[
\operatorname{FDP}_S(R)\leq\delta E_S
\qquad\text{for every }S\subseteq[t].
\]

The implementation enumerates every intersection \(S\) for each candidate \(R\) and uses exact rational arithmetic.

## Resource-aware action policy

At time \(t\), choose

\[
R_t\in\arg\max_R\sum_{i\in R}v_i
\]

subject to

\[
R_{t-1}\subseteq R\subseteq[t],\qquad
\sum_{i\in R}c_i\leq B_t,\qquad R\in\mathcal C_t.
\]

The first experiment requires:
- \(R_t\) irrevocable and nested;
- cumulative capacity \(B_t\) predictable before the decision and nondecreasing;
- all costs nonnegative;
- values and costs fixed before the corresponding e-value is observed (or otherwise predictable under a separately justified contract);
- one fixed increasing e-collection with predeclared \(\gamma\).

Because \(R_{t-1}\subseteq R_t\) and the menus are nested, an earlier certified action remains a feasible candidate as capacity accumulates. Selecting any \(R_t\in\mathcal C_t\) preserves the online e-closure theorem's SupFDR guarantee under that theorem's assumptions. The verifier does not prove incoming \(E_i\) are valid marginal e-values; this is an upstream evidence contract.

This is **cumulative** capacity, not a per-period service-rate or queue-stability model. It does not yet address asynchronous e-process evidence, repeated looks within a hypothesis, or time-varying stopping rules beyond the cited theorem's exact filtration.

## Independent validity and optimality verifier

The independent verifier in validation/online_eclosure_capacity_oracle.py has two distinct checks:

1. **Validity receipt:** recompute all \(2^t\) intersections for every selected \(R_t\), check closure slack, check \(R_{t-1}\subseteq R_t\), enforce cumulative capacity, and verify prefix digests and receipt fields.
2. **Optimality receipt:** independently enumerate every nested and capacity-feasible candidate \(R\), check its e-closure membership, and confirm the selected set maximizes total predeclared value. Ties resolve by lower resource cost, then lexicographically smaller selected-index tuple.

All exact values serialize as reduced numerator/denominator pairs. Prefix hashes include only evidence, weights, utility, costs, capacity and risk level available up to time \(t\). Changing future e-values therefore cannot change prior decisions or prior prefix hashes.

## Complexity and SOTA baselines

The reference implementation enumerates up to \(2^t\) candidate sets and checks up to \(2^t\) intersections per candidate. Its worst-case per-prefix cost is \(O(4^t)\), and the independent optimality verifier repeats exhaustive candidate validation. It therefore defaults to a hard limit of 10 hypotheses; this is an oracle for theorem/decision semantics, not a scalable algorithm.

A production implementation must be compared against:
- online e-LOND / closed online e-BH dynamic programming, with reported \(O(t^2)\) cost;
- donation-based online e-LOND / compound e-values, with reported \(O(\log t)\) decision-time algorithms;
- LORD, SAFFRON, and SCORE-LORD / SCORE-SAFFRON as online FDR baselines (while carefully matching the guarantee class);
- unconstrained utility ranking and no-capacity e-closure as operational controls.

Do not compare a SupFDR method to a fixed-time FDR method as if the validity targets were interchangeable.

## Tests and negative controls

The test file validation/test_online_eclosure_capacity_oracle.py covers:
- nested certified actions under cumulative capacity;
- rejecting a high-value but uncertified new hypothesis;
- prefix invariance when future e-values change;
- irrevocable decisions and nondecreasing cumulative capacity;
- malformed \(\gamma\) and nonmonotone capacity;
- JSON-safe decision receipts;
- rejection of a statistically valid but utility-suboptimal action by the independent verifier.

Reproduction from the repository root:

    PYTHONPATH=validation python -m unittest -v test_online_eclosure_capacity_oracle.py

The test suite has not been executed in this environment. Remote CI is not represented as passing.

## Benchmark programme

Start with exact streams of \(t\leq10\), then validate a scalable solver against the exact oracle for every frozen small fixture. Required dimensions:

- stream length 1–10 for exact oracle, then 10², 10³, 10⁴ for optimized candidates;
- \(\delta\in\{0.01,0.025,0.05,0.1\}\);
- \(\gamma\): flat, geometric donation, sparse, front-loaded and back-loaded (all nonnegative with sum at most one);
- null e-values: Bernoulli/rare-event likelihood ratios, heavy-tailed mean-one distributions, and calibrated domain e-processes;
- arbitrary cross-hypothesis dependence: independent, common-shock, block and adversarial coupling;
- values: uniform, predeclared priority, heterogeneous cost/value ratios, misspecified external priorities;
- capacities: zero, bursty cumulative increments, saturated and unsaturated;
- report SupFDR diagnostics, power/discoveries, expected declared utility, resource utilization, p50/p95 decision latency, memory, certificate verification cost and number of candidate/intersection checks.

Use common random numbers for paired comparisons, independent held-out seeds for confirmation, and Monte Carlo uncertainty. Empirical SupFDR checks are diagnostics; the theorem and e-value contracts provide validity.

## Boundaries and next gates

- SupFDR is inherited only if every selected \(R_t\) remains in the exact theorem's certified menu and all theorem assumptions hold.
- This is not a new proof for asynchronous e-process streams or per-period capacities.
- Incoming e-values are assumed valid; an independent upstream evidence verifier is required.
- Exact oracle complexity is exponential. Do not port it to production or advertise scalability.
- Next: execute committed tests on the exact branch head; add an independent combinatorial solver for the menu problem and differential-test it against this oracle; benchmark online donation algorithms; formalize predictable metadata and the domain evidence contract.

## References

- Xu, Fischer & Ramdas (2026), [Improving online FDR procedures via online analogs of e-closure and compound e-values](https://proceedings.mlr.press/v337/xu26a.html), [arXiv version](https://arxiv.org/abs/2603.24792).
- Bai & Ramdas (2026), [Recertifying decisions via e-closure](https://arxiv.org/abs/2610.35821).
- Kuang, Gang & Xia (2026), [SCORE: A Unified Framework for Overshoot Refund in Online FDR Control](https://proceedings.mlr.press/v306/kuang26c.html).
