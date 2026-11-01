# Countable prior-weighted dynamic e-closure

**Status:** candidate theorem, exact finite-family oracle, and prefix-code prior construction. No novelty claim or empirical power improvement is made. Validity of each supplied component e-process remains an upstream mathematical obligation.

## 1. Question and scope

Dynamic e-closure uses future-extension coherence to compare the true-null intersection active at a stopping time with one fixed terminal intersection. The finite-envelope companion track localizes predictable random arrivals to a deterministic finite candidate set. This track asks whether a single prior-weighted mixture can dominate every possible finite true-null active intersection directly.

Let the global hypothesis universe be countable and indexed once and for all by positive integers. At each reporting time t, let I_t be a finite active family and R_t a reported subset of I_t. I_t need not be nested or predictable for the pathwise argument below. Under a fixed data-generating distribution P, let N(P) be the fixed set of true-null indices and B_t(P)=N(P) intersect I_t.

For every deterministic finite nonempty S, assume a nonnegative process E_t^S adapted to one common filtration is an e-process under every P in the intersection null H_S. We use the maximal definition

\[
  \mathbb E_P[\sup_{t\geq 0} E_t^S]\leq 1
  \qquad\text{for every }P\in H_S.
\]

A test supermartingale is sufficient, but the proof does not require components to remain supermartingales after a running-maximum adjuster. For continuous time, assume separable/right-continuous versions so the supremum is measurable, or state the result on a countable monitoring grid.

Choose deterministic prior masses pi_S > 0, fixed before observing data, over all finite nonempty intersections, with

\[
  \sum_{\substack{S\subseteq\mathbb N\\0<|S|<\infty}}\pi_S\leq 1.
\]

Define the true-null mixture

\[
  M_t(P)=\sum_{\substack{S\subseteq N(P)\\0<|S|<\infty}}\pi_S E_t^S.
\]

For every time t, require the prior-weighted local closure inequalities for every nonempty S subset of I_t:

\[
  \operatorname{FDP}_{S}(R_t)
  :=\frac{|R_t\cap S|}{|R_t|\vee 1}
  \leq \alpha\pi_S E_t^S.
\]

These are pathwise decision constraints. They do not establish the stochastic validity of the component processes.

## 2. Candidate countable-mixture theorem

**Proposition (prior-weighted random-intersection domination).** If the assumptions above hold and every local closure inequality holds simultaneously for all reporting times and all nonempty S subset of I_t, then for every fixed P,

\[
  \operatorname{FDP}_{N(P)}(R_t)
  \leq \alpha M_t(P)\quad\text{for every }t,
\]

where the FDP is zero when B_t(P) is empty. Consequently,

\[
  \mathbb E_P\left[\sup_t\operatorname{FDP}_{N(P)}(R_t)\right]
  \leq \alpha.
\]

In particular, FDR at every stopping time is at most alpha, since stopped FDP is bounded by the pathwise supremum. This is a candidate direct route to SupFDR without future-extension coherence or a deterministic finite horizon envelope.

**Proof.** If B_t(P) is empty, the loss is zero. Otherwise B_t(P) is one nonempty finite subset of I_t, so the local constraint for S=B_t(P) gives

\[
  \operatorname{FDP}_{N(P)}(R_t)
  =\operatorname{FDP}_{B_t(P)}(R_t)
  \leq \alpha\pi_{B_t(P)}E_t^{B_t(P)}
  \leq \alpha M_t(P).
\]

The intersection B_t(P) may be random and may change with t; it is dominated pathwise by the same mixture containing every fixed finite true-null intersection. Since P belongs to H_S for every finite S subset of N(P), each E^S is an e-process under P. By nonnegativity and monotone convergence,

\[
\begin{aligned}
  \mathbb E_P[\sup_t M_t(P)]
  &\leq \mathbb E_P\left[\sum_{S\subseteq_f N(P)}\pi_S\sup_t E_t^S\right]\\
  &=\sum_{S\subseteq_f N(P)}\pi_S\mathbb E_P[\sup_t E_t^S]\\
  &\leq \sum_{S\subseteq_f N(P)}\pi_S\leq 1.
\end{aligned}
\]

Combining the inequalities proves the claim. No independence between component processes is used.

**Important correction to the initial issue sketch:** the mixture is an e-process in the maximal sense. It should not be described as a time-monotone supermartingale. A pathwise nondecreasing nonnegative supermartingale is generally constant; after a running-maximum adjuster, the maximal e-process property is the appropriate object.

## 3. What this removes—and what it does not

The argument removes the random-index selection problem by paying prior mass for every possible finite intersection. It does not require coherence between E^S and E^{S'} when S is a subset of S'. The proof also does not require I_t itself to be predictable or nested, provided that:

1. every fixed-S component E^S is valid in the common filtration under H_S;
2. each reported R_t is measurable at its declared reporting time;
3. all local closure inequalities are checked for every nonempty S subset of the realised finite I_t;
4. the prior is fixed ex ante and independent of the evidence path.

A post-evidence activation policy is not automatically safe. If it causes a component process to start betting with evidence already observed, that component may cease to be an e-process. The theorem delegates this obligation to the fixed-S process contract. Neutral-one padding before predictable activation is one safe construction; broader activation semantics need their own component-validity proof.

If I_t can be infinite, B_t(P) need not be finite and this proposition does not apply as written. A separate local-loss approximation or an infinite-intersection component is required.

## 4. A universal prefix-code prior

A concrete prior can be defined without choosing summable coordinate parameters. For S={i_1<...<i_k}, define positive gaps

\[
  g_1=i_1,\qquad g_j=i_j-i_{j-1}\quad(j\geq2).
\]

Let gamma(n) be the Elias gamma code of a positive integer n, with length

\[
  L_\gamma(n)=2\lfloor\log_2 n\rfloor+1.
\]

Encode S as

\[
  C(S)=\gamma(k)\,\gamma(g_1)\cdots\gamma(g_k),
  \qquad L(S)=L_\gamma(k)+\sum_{j=1}^kL_\gamma(g_j).
\]

This code is prefix-free: the first codeword identifies k, after which exactly k prefix-free gap codewords are decoded. Since the Elias gamma code has Kraft sum one,

\[
\begin{aligned}
  \sum_{S\subseteq_f\mathbb N,\,S\ne\varnothing}2^{-L(S)}
  &=\sum_{k\geq1}2^{-L_\gamma(k)}
    \left(\sum_{g\geq1}2^{-L_\gamma(g)}\right)^k\\
  &=1.
\end{aligned}
\]

Thus pi_S=2^{-L(S)} is a proper countable prior with positive mass on every finite nonempty intersection. The exact Python oracle implements the code and checks finite prefix Kraft sums; the countable normalization follows from the coding identity, not finite simulation.

This is a universal coding prior, not a universally optimal power allocation. For a singleton {i}, its mass is Theta(i^-2). Intersections with many or widely spaced indices pay a larger description-length penalty. Alternatives to benchmark include product priors proportional to the product of summable positive lambda_i, size-aware hierarchical mixtures, group priors, and metadata-based priors registered before evidence collection.

A data-adaptively selected prior is not covered. A hyperprior mixture of fixed priors with fixed top-level weights is valid when those weights sum to at most one.

## 5. Finite-family oracle

The implementation adds:

- exact Elias-gamma prefix-code weights over finite intersections;
- canonical finite-intersection IDs and exact rational arithmetic;
- exhaustive local closure checking over all nonempty subsets of the active family;
- fail-closed behavior for missing component processes, malformed IDs, negative values, and incomplete paths;
- enumeration of locally certified rejection sets for a finite active family;
- exact finite-truth fixtures checking pathwise mixture domination and a finite-prefix prior-budget audit.

The oracle is exponential by design: it checks 2^|I_t|-1 local constraints. It is a reference implementation, not a production algorithm. The component e-process law, common filtration, signed event provenance, and real-world activation timing are not inferred from supplied paths.

Run from the repository root:

    PYTHONPATH=validation python -m unittest -v test_countable_prior_weighted_eclosure.py

## 6. Further negative controls and stress tests

The current exact tests cover prefix-code uniqueness/Kraft mass on a finite prefix, finite prior-budget checks, missing intersection processes, data-dependent prior selection, random non-nested active sets, exact rational boundaries, and pathwise mixture domination.

The next controls should include a deliberately invalid component process, evidence-after-activation that invalidates a component, omitted true-intersection constraints, all-null/sparse/moderate/dense alternatives, independent/common-shock/block/adversarial dependence, singleton/clustered/widely spaced true-null sets, and fixed-seed preregistered simulations. Simulation diagnostics cannot establish the theorem.

## 7. Prior-art boundary and comparisons

The ingredients—e-process mixtures, prefix-code/Kraft priors, weighted closure, and online closed testing—are established. Relevant work includes:

- de Heide (2026), Dynamic e-closure for online hypotheses with any-time-valid evidence: closure principles and projective mergers, arXiv:2608.09927.
- Fischer and Ramdas (2024), Admissible online closed testing must employ e-values, arXiv:2407.15733.
- Xu, Solari, Fischer, de Heide, Ramdas and Goeman (2025), Bringing Closure to False Discovery Rate Control: A General Principle for Multiple Testing, arXiv:2509.02517.
- Tavyrikov, Goeman and de Heide (2026), Carefree multiple testing with e-processes, Electronic Journal of Statistics 20(2), 3178–3189, DOI 10.1214/26-EJS2546.
- Xu, Fischer and Ramdas (2026), Improving online FDR procedures via online analogs of e-closure and compound e-values, UAI 2026 / PMLR 337.
- Ek, Stark, Stuckey and Vukcevic (2023), Adaptively Weighted Audits of Instant-Runoff Voting Elections: AWAIRE, arXiv:2307.10972.

No novelty claim should be made until the proposition is compared theorem-by-theorem with online closed testing and the general e-closure principle. The candidate distinction is a countable prior over finite intersections used to dominate a time-varying random true-null active intersection without future-extension coherence or a deterministic finite envelope.

## 8. Benchmark programme

Compare against the finite-envelope predictable-arrival method, coherent dynamic e-closure, global projective mergers, intersection-local AWAIRE mixtures, online e-closure/compound e-values, Carefree multiple testing with adjusters, and weighted closed testing/general e-closure.

At matched guarantee levels report stopped-FDR and SupFDR separately, power, true actionable discoveries, false investigations, precision@B, recall@B, utility regret, prior-mass sensitivity by intersection rank/size, and computation. First compare exact finite families (n=2..12); then run 5,000 exploratory and 50,000 frozen confirmatory streams. Report p50/p95 closure latency, peak memory, constraint count, and verifier time. Do not compare ordinary FDR with SupFDR, or treat calibrated confidence as an e-process without a conditional-validity proof.

## 9. Next mathematical questions

1. Can the local-constraint family be reduced without losing the true-set guarantee, using structure of E^S or a general e-partitioning theorem?
2. What prior maximizes worst-case or average power under a declared distribution of intersection size and arrival rank?
3. Can a hierarchical mixture adapt to unknown intersection complexity while keeping all top-level weights fixed ex ante?
4. When is the countable-mixture bound sharper or weaker than finite-envelope coherence?
5. Can the method extend from FDP to bounded action losses whose true-configuration loss equals one of the locally certified losses?
6. What exact or approximate separation oracle can avoid all 2^n constraints while preserving an independently checkable certificate?
