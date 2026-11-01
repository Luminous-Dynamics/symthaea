# Countable global-projective e-closure for random arrivals

**Status:** candidate extension plus exact finite-path oracle. The underlying global-weight affine merger is established prior work. The research question is whether its countable neutral-padding form removes the finite-envelope restriction for predictable random arrivals while preserving an exact true-set certificate.

## 1. Core construction

Let the hypothesis universe be countable, indexed by positive integers. Fix global, ex-ante weights

    w_i >= 0,   sum_{i>=1} w_i <= 1.

For each coordinate i, let M_{i,t} be a nonnegative e-process under H_i in one common global filtration. Assume M_{i,t}=1 while i is dormant and that activation is predictable, so evidence begins only after the activation decision is available to the global filtration. Active sets I_t are finite and nested, but their membership and arrival times may be random predictable functions of the past.

For S subset of N define

    E_t^{S,w} = 1 + sum_{i in S} w_i (M_{i,t} - 1)
              = 1 - sum_{i in S} w_i + sum_{i in S} w_i M_{i,t}.

This is the global-weight affine/projective merger already studied in dynamic e-closure. The cash remainder makes it nonnegative; fixed global weights are not renormalized when new hypotheses arrive. Dormant-one padding gives exact projectivity:

    E_t^{S,w} = E_t^{S intersect I_t,w}.

For a fixed P with countable true-null set N(P), define the global certificate

    Z_t^{N(P),w} = 1 - sum_{i in N(P)} w_i
                     + sum_{i in N(P)} w_i M_{i,t}.

This nonnegative countable mixture is well-defined. By Tonelli and the common-filtration e-process contract, for every almost-surely finite global stopping time tau,

    E_P[Z_tau^{N(P),w}]
      <= 1 - sum_{i in N(P)} w_i + sum_{i in N(P)} w_i
      = 1.

Since every coordinate outside I_t is still exactly one,

    Z_t^{N(P),w} = E_t^{N(P) intersect I_t,w}.

The random active true-null intersection is the restriction of one fixed countable true-set process. Unlike the finite-envelope route, this argument does not need a deterministic finite candidate envelope at each horizon. Unlike the all-intersections prior route, it does not allocate a second prior weight to every finite intersection.

## 2. Candidate stopped-FDR proposition

At every reporting time t, let R_t be a subset of I_t. Require the local closure constraints for every nonempty S subset of I_t:

    FDP_S(R_t) = |R_t intersect S| / (|R_t| vee 1)
               <= alpha * E_t^{S,w}.

Let B_t(P)=N(P) intersect I_t. If B_t(P) is nonempty, it is one of the locally checked intersections and

    FDP_N(P)(R_t)
      = FDP_B_t(P)(R_t)
      <= alpha * E_t^{B_t(P),w}
      = alpha * Z_t^{N(P),w}.

If B_t(P) is empty, the loss is zero. Optional stopping of the single global countable certificate therefore gives stopped-FDR control at level alpha.

The proof does not require independence across coordinates; it requires every coordinate process to be valid in the same global filtration under its own null. The pathwise local constraint must hold at each report, and every active family must be finite so the local closure family is finite.

### Exact linear-time separator for the affine closure family

For a fixed candidate rejection set R with r=|R|>0 (the same expression also handles R empty), write

    d_i = w_i (M_{i,t}-1)
    c_i = 1{i in R}/r - alpha*d_i.

For any nonempty S subset of I_t, its local constraint is equivalent to

    sum_{i in S} c_i <= alpha.

The maximum over all subsets is separable: include every coordinate with c_i>0. Therefore all 2^|I_t|-1 local inequalities hold if and only if

    sum_{i in I_t} max(0, 1{i in R}/r - alpha*w_i*(M_{i,t}-1)) <= alpha.

This is an exact O(|I_t|) separation test. When the positive-part sum exceeds alpha, the set of positive-coefficient coordinates is an explicit violating intersection. If no coefficient is positive, all nonempty subset sums are nonpositive and the constraints pass because alpha>0. The implementation uses exact Fractions and checks this separator against exhaustive enumeration on every rejection mask for active-family sizes 1 through 5 and a rational evidence grid.

This reduces the cost of checking one proposed rejection set from O(2^n) to O(n). If an outer exact action optimizer still enumerates 2^n candidate rejection sets, its closure-checking component can fall from O(4^n) to O(n 2^n). The action-optimization problem itself is not yet solved by this separator; it still needs an exact branch-and-bound or MILP treatment under cumulative capacity and persistence constraints.

### Stronger result: an exact O(n)-size MILP for the full local closure

The separator can be embedded directly in the action optimizer. Let x_i in {0,1} encode rejection, r=sum_i x_i, and d_i=w_i(M_{i,t}-1). For any subset S, the local closure inequality is equivalent to

    sum_{i in S} (x_i - alpha*r*d_i) <= alpha*r.

Because the maximum over all subsets is the sum of positive parts, the entire local-closure family is equivalent to the existence of y_i satisfying

    y_i >= 0
    y_i >= x_i - alpha*r*d_i                for every i
    sum_i y_i <= alpha*r.

This remains valid when r=0: the binary/count constraints force x=0 and the only feasible auxiliary vector is y=0. The formulation is necessary and sufficient, not a relaxation.

The capacity-aware decision problem can therefore be written as a mixed-integer linear program:

    maximize   sum_i v_i*x_i
    subject to r = sum_i x_i
               x_i in {0,1}; 0 <= r <= n
               y_i >= 0
               y_i >= x_i - alpha*r*w_i*(M_i-1)
               sum_i y_i <= alpha*r
               sum_i c_i*x_i <= B_t
               x_i = 1 for every previously committed rejection.

There are n binary x variables, one continuous count variable r, and n continuous y variables. The count equality r=sum_i x_i makes r integral automatically, so declaring it integer is redundant. With capacity and persistence, the model has n+2+O(p) rows and O(n) nonzeros (6n+2 with one capacity row and no persistence rows). The closure logic no longer introduces 2^n rows. The overall problem is still a 0-1 optimization problem (NP-hard in general); this does not imply linear-time optimization. It gives a compact exact formulation suitable for HiGHS/CBC/SCIP/Gurobi comparisons, with exact rational post-solution verification.

The new `validation/affine_projective_closure_milp.py` builds the model with exact Fraction coefficients, emits CPLEX-LP-compatible text with each row scaled to exact integer coefficients, provides an O(n) exact binary-decision verifier that does not trust solver auxiliary values, and retains a separate exhaustive verifier for small-family differential checks. `validation/test_affine_projective_closure_milp.py` differentially compares the exact MILP feasibility witness against direct enumeration for all rejection masks and rational evidence grids for n=1..4, and checks that the LP export preserves rational coefficients. The builder does not call or depend on a solver, and no solver performance claim is made.

This is the key implementation target: preserve the small exhaustive verifier as the oracle, use the compact MILP for action selection, and independently replay every proposed solution against the original local-intersection definition before issuing a decision receipt.

### First solver-backed engineering sweep (exploratory)

A deterministic optional adapter has been added:

- `validation/benchmark_affine_projective_closure_solvers.py` builds the exact model, converts it to SciPy's sparse MILP interface, calls `scipy.optimize.milp` (HiGHS), records primal/dual bounds, MIP gap, nodes, model dimensions, build/solve/verifier times, and independently replays the returned binary action using exact Fraction arithmetic.
- `validation/test_benchmark_affine_projective_closure_solvers.py` checks deterministic instance manifests, linear model dimensions, exact exhaustive agreement on small cases, and optional solver results.
- The rejection-count variable r is now continuous; its integrality follows from r=sum_i x_i with binary x_i.

I also ran an independent SciPy 1.17.0 / HiGHS prototype in the current analysis environment on a seven-size synthetic sweep. All seven cases were proven optimal (solver status 0), all returned actions passed exact rational closure/capacity verification, and all reported zero MIP gap:

| Active n | Selected | Utility | Cost | HiGHS solve time | Nodes |
|---:|---:|---:|---:|---:|---:|
| 10 | 9 | 522 | 118 | 2.55 ms | 1 |
| 25 | 24 | 1,542 | 307 | 1.89 ms | 0 |
| 50 | 48 | 2,647 | 601 | 2.78 ms | 1 |
| 100 | 96 | 5,609 | 1,239 | 5.53 ms | 1 |
| 250 | 240 | 12,643 | 2,971 | 20.89 ms | 1 |
| 500 | 481 | 26,671 | 5,850 | 144.85 ms | 1 |
| 1,000 | 963 | 55,871 | 11,846 | 291.78 ms | 1 |

The generator sets d_i=w_i(M_i-1)=12/n and uses alpha=0.1, heterogeneous integer utilities and costs, and capacity at 95% of total cost. This creates a lower-cardinality closure requirement plus a knapsack trade-off. The exact verifier checks the output independently.

**Interpretation:** this demonstrates that the compact formulation can be solved at n=1,000 on one tractable synthetic family in this environment. It is not a publishable solver benchmark, does not establish general 1-second performance, and does not demonstrate superiority over another solver. The instances were solved at or near the root node, so substantially harder and more diverse families are required.

The checked-in benchmark runner and tests have been authored but were not executed against the repository checkout in this environment. Direct GitHub cloning was unavailable; the numerical sweep above used an independent local implementation of the same exact formulation, not the checked-in runner. Re-run the checked-in command in a connected development shell:

    PYTHONPATH=validation python validation/benchmark_affine_projective_closure_solvers.py --sizes 10,25,50,100,250,500,1000 --seed 20261101 --output solver-sweep.json

The next benchmark must include hard 0-1 knapsack instances, correlated utility/cost distributions, tight capacities, persistent-prefix constraints, high-precision rational coefficients, solver time limits, and at least three solvers under identical hardware/time/thread settings. Report both primal quality and proof-of-optimality time; finding a feasible incumbent is not the same as proving an optimum.



**Guarantee boundary:** this O(n) separator applies to the raw affine intersection certificates used for the stopped-FDR construction. It does not automatically apply after a nonlinear running-maximum adjuster is applied separately to every intersection: the adjusted value A(max_{s<=t} E_s^{S,w}) is no longer an additive function of the current coordinate values. For SupFDR, use the projectivity-preserving adjuster from the #9538 track and keep exhaustive local-constraint checking until a separate valid separator for the adjusted family is derived. The raw-coordinate monotonicity field in the oracle is only a path diagnostic, not evidence that the supplied coordinate paths are valid e-processes; in particular, a nontrivial valid e-process started at one is not generally pathwise nondecreasing.


## 3. SupFDR extension

The ordinary e-process condition does not by itself control the expected path supremum. For SupFDR, require the projective certificate family to be pathwise time-monotone e-processes, or apply a valid common running-maximum adjuster to the whole projective family while preserving dormant-one projectivity.

For nested arrivals, if s<=t then I_s is a subset of I_t. Any coordinate in I_t but not I_s is dormant at s. Hence

    E_s^{N(P) intersect I_t,w}
      = E_s^{N(P) intersect I_s,w}.

So the historical path of the terminal active-set certificate agrees with the global true-set certificate. A projectivity-preserving adjuster therefore has

    adjusted E_t^{N(P) intersect I_t,w}
      = adjusted Z_t^{N(P),w}.

If the global adjusted certificate is nondecreasing, then on a discrete monitoring grid monotone convergence yields

    E_P[sup_t adjusted Z_t^{N(P),w}]
      = lim_n E_P[adjusted Z_n^{N(P),w}]
      <= 1,

and the all-times local closure inequalities give SupFDR at level alpha.

The branch now includes an exact finite-path replay, `verify_adjusted_projective_supfdr_path`, that computes the same common running-maximum adjuster for (a) the fixed full true-set certificate and (b) the current active true-set intersection. It checks equality of the entire historical path prefix, verifies every adjusted local closure constraint by exhaustive finite-intersection enumeration, and checks pathwise dominance. This is the correct SupFDR replay route. It intentionally does **not** reuse the raw affine O(n) separator, because applying a nonlinear maximum/adjuster across time destroys the current-time additive form.

The fixture includes a raw certificate path that rises and then falls; the adjusted certificate remains nondecreasing and the active/full historical adjusted paths agree. As everywhere else, this fixture tests algebra only. The adjuster theorem supplies stochastic validity only when the raw global certificate is a valid e-process.


**Do not infer SupFDR from an ordinary test supermartingale alone.** The adjuster or monotonicity requirement is an additional certificate contract. A raw-path monotonicity check in the finite oracle is only a sufficient pathwise diagnostic, not an implementation of the adjuster theorem.

## 4. Why this differs from the all-intersections prior route

The countable prior route (Issue #9649 / draft PR #9684) assumes an arbitrary valid e-process E^S for each finite intersection and adds pi_S for all finite S. It is structurally broader but has an intersection-description penalty and still checks 2^|I_t|-1 constraints.

The global-projective route constrains the intersection merger to the established affine form generated by globally summable coordinate weights. It avoids a second pi_S penalty and the finite-envelope assumption, but inherits the model restriction that every intersection certificate is produced by the same global coordinate family and fixed weights. It is a sharper systems route if the evidence architecture supports global coordinates; it is not a universal replacement for arbitrary intersection-level processes.

## 5. Exact weight families

Two transparent ex-ante choices have exact total mass one:

- Geometric: w_i=2^-i; the first n coordinates have mass 1-2^-n.
- Telescoping: w_i=1/[i(i+1)]=1/i-1/(i+1); the first n coordinates have mass 1-1/(n+1).

The coordinate-to-index mapping and weight scheme must be registered before observing evidence. Do not renormalize weights over the currently active set: doing so changes the projective merger and can rewrite earlier certificates.

### The symmetry limit is structural, not an implementation bug

The dynamic e-closure paper's cross-horizon rigidity result shows that, within its pointwise arbitrary-dependence merger class, global summability is unavoidable. Its Corollary 4.13 further shows that a symmetric nontrivial merger over a countably infinite open-ended universe is impossible: symmetry forces equal coordinate weights, and summability then forces every weight to zero.

Therefore geometric and telescoping weights are not merely interchangeable engineering settings. Every useful countable projective design in this class must encode a pre-data priority, group structure, or other asymmetry. This can be an explicit registry order, a domain hierarchy, or a fixed group-weight × within-group-weight scheme. The benchmark must report how much weight lands on hypotheses by arrival rank and group, and measure sensitivity to reasonable pre-registered relabelings.

The route beyond this limitation is not to secretly renormalize or select weights after observing evidence. It is to leave at least one assumption of the rigidity theorem: exploit verified conditional dependence, use direct intersection betting, or build history-dependent mergers with their own filtration-validity proof. The paper identifies these as future research directions; they are the highest-upside theory track after the current affine arrival extension.


The finite oracle implements both schemes with exact rational arithmetic and verifies dormant-one equality between the active true-null certificate and the finite fixture's full true-null certificate.

## 6. Implementation and trust boundary

- `validation/countable_global_projective_eclosure.py` provides exact rational weights, projective certificate evaluation, exhaustive finite local-closure verification, dormant-one checks, full-true-set/active-true-set identity checks, and a pathwise monotonicity diagnostic.
- `validation/test_countable_global_projective_eclosure.py` covers both weight families, random nested arrivals, dormant padding, empty true sets, local closure, invalid inputs, and the SupFDR condition boundary.

The module is a finite exact oracle. It does not authenticate event timestamps, signatures, catalogues, or filtration completeness. It does not infer E[M_{i,t}|F_{t-1}]<=1 from a realised path. Predictable activation and the conditional e-process contract remain obligations of the upstream evidence producer. Its O(n) separator certifies the raw affine stopped-FDR closure only; the adjusted SupFDR route still needs an all-intersections verifier. The exhaustive differential oracle is exponential and is intentionally test-only. The production-facing local-constraint separator is O(n), but the exact action optimizer still needs a scalable solution; no production-scale claim is made.

Run from the repository root:

    PYTHONPATH=validation python -m unittest -v test_countable_global_projective_eclosure.py

The source test suites now contain 32 test methods across the global-projective oracle, MILP model, and solver benchmark. The 16-test raw-separator suite had previously passed locally; the adjusted-path, MILP, and solver-adapter tests have not yet been executed in this environment.

## 7. Prior-art boundary

The global-weight affine merger and projective neutral padding are established by the dynamic e-closure paper. Its cross-horizon rigidity theorem characterizes admissible pointwise arbitrary-dependence mergers via globally summable coordinate weights, and its subsequent corollary gives the global-weight projective construction. The same paper explicitly identifies random predictable arrivals as a future extension. We therefore make no novelty claim for the merger itself.

The candidate increment is the direct countable true-set process Z^{N(P),w}, using neutral-one padding to remove the random active-intersection index for predictable growing families without a deterministic finite envelope. Whether this is genuinely outside the paper's general closure theorem or existing online closed-testing theory requires a theorem-by-theorem review.

Relevant sources:

- de Heide (2026), *Dynamic e-closure for online hypotheses with any-time-valid evidence: closure principles and projective mergers*, arXiv:2608.09927, especially Theorem 4.8, Corollary 4.10, and its discussion of random predictable arrivals.
- Fischer and Ramdas (2024), *Admissible online closed testing must employ e-values*, arXiv:2407.15733.
- Xu et al. (2025), *Bringing Closure to False Discovery Rate Control: A General Principle for Multiple Testing*, arXiv:2509.02517.
- Xu, Fischer and Ramdas (2026), *Improving online FDR procedures via online analogs of e-closure and compound e-values*, PMLR 337.
- Tavyrikov, Goeman and de Heide (2026), *Carefree multiple testing with e-processes*, DOI 10.1214/26-EJS2546.

## 8. Benchmark programme

Compare at matched guarantees against #9538/#9539 coherent dynamic e-closure, #9648 finite-envelope predictable arrivals, #9684 countable prior-weighted closure, online closed testing, online e-closure/compound e-values, and Carefree multiple testing. Use exact finite enumeration for n=2..12, then 5,000 exploratory and 50,000 frozen confirmatory streams. Benchmark both the O(n) separator and the exhaustive reference, and then evaluate whether a binary optimization model can minimize utility regret under the separator constraints without enumerating every action.

Report stopped-FDR and SupFDR separately; power, true actionable discoveries, false investigations, precision@B, recall@B, utility regret, coordinate-weight sensitivity by arrival rank, cumulative capacity use, p50/p95 decision and verifier latency, peak memory, and local constraint counts. The global-projective method should also report what fraction of total coordinate budget lands on eventually active hypotheses. Do not compare ordinary FDR to SupFDR. Simulation is a diagnostic, not a proof of stochastic validity.

The target is not merely to beat a baseline on p-value thresholds. The engineering goal is a system where every action can be independently replayed from an immutable event ledger, while the statistical guarantee remains valid under predictable arrivals and arbitrary cross-coordinate dependence. The next theory frontier is a verified history-dependent or direct-intersection betting construction that can recover power without violating the global budget or the symmetry impossibility.
