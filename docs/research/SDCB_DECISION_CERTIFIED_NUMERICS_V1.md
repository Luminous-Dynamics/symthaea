# SDCB decision-certified numerical computation v1

**Status:** Research specification plus an authored finite outcome action-value enclosure reference (`scripts/research/sdcb_decision_value_bounds.rs`, nine Rust tests). No certified Gaussian probability enclosure implementation is claimed; the action-value reference consumes already supplied probability intervals. Rust tests remain uncompiled/unexecuted.

## 1. Motivation

The bivariate rectangle oracle in #13951 computes bounded conditional Gaussian rectangle probabilities and retains per-cell diagnostics. Its adaptive-Simpson refinement difference is an error indicator, not a certified total error bound. The scalar inner quadrature and floating-point operations also need an explicit error contract before the result can be called a numerical certificate.

The scientific decision system should not consume an unqualified point estimate when numerical uncertainty can change the selected action. The next increment is therefore not another Gaussian integrator or planner. It is a **validated enclosure layer and decision-sufficient refinement contract** that composes with:

- #13945 / #13951 for joint Gaussian rectangle probabilities;
- #13940 for interval-integrated dynamic observations;
- #13567 for the exact sequential policy oracle;
- #13739 for existing experiment-selector evaluation;
- #13805 for independently owned evidence transitions and receipts.

## 2. Core contract

For each probability cell, return an enclosure

\[
p_i \in [p_i^-,p_i^+], \qquad 0 \le p_i^- \le p_i^+ \le 1.
\]

The enclosure must contain the exact probability for the exact input values represented by the numerical API. It must separately report:

1. input/parameter interpretation (binary floating-point point inputs versus caller-supplied parameter intervals);
2. arithmetic-rounding contribution;
3. conditional normal-CDF / tail-evaluation contribution;
4. outer-integration remainder;
5. domain-truncation contribution;
6. resource use and termination reason.

The API must distinguish:
- **CertifiedEnclosure:** every listed error component has a defensible inclusion bound;
- **HeuristicEstimate:** useful estimate, but one or more components are not rigorously bounded;
- **UnresolvedWithinBudget:** enclosure is valid but too wide for the requested decision;
- **UnsupportedInput:** covariance/support or input class is not supported;
- **NumericalFailure:** no trustworthy result can be returned.

A refinement indicator, replicate spread, or agreement between two ordinary floating-point implementations must never be promoted to a certified error bound.

## 3. Bivariate reference problem

For a full-rank standardized bivariate Gaussian with correlation $\rho$, $|\rho|<1$, define $s=\sqrt{1-\rho^2}$. The rectangle probability can be written

\[
P = \int_{a_1}^{b_1} \phi(z)
\left[
\Phi\left(\frac{b_2-\rho z}{s}\right)
-
\Phi\left(\frac{a_2-\rho z}{s}\right)
\right] dz.
\]

The implementation in #13951 remains the fast heuristic baseline. The first validated reference should use a conservative enclosure of the nonnegative integrand on a partition of the finite integration domain, plus an independently bounded omitted-tail allowance. Because the integrand is nonnegative, panel lower/upper bounds can be accumulated without cancellation. A more efficient validated Taylor/Gauss-type method can follow only after its remainder proof and special-function enclosures are explicit.

This low-dimensional enclosure path is a correctness reference, not the proposed high-dimensional performance winner.

### 3.1 Required numerical components

- Establish certified bounds for the normal interval mass used by the conditional integrand, including far-tail and narrow-interval cases.
- Establish outward-rounded or arbitrary-precision interval arithmetic for all enclosure calculations. Ordinary f64 arithmetic with an empirical epsilon allowance is insufficient.
- Bound the outer integral by a mathematically valid remainder method. Do not use the Simpson |S2-S1|/15 expression alone as a certificate.
- Bound truncation separately. For a first-coordinate standard-normal cutoff K, a conservative omitted mass is bounded by P(|Z|>K); the tail bound itself must be evaluated with a certified method.
- Propagate all components to a final probability interval, clamp only by mathematically valid probability constraints, and record any loss of tightness.
- Preserve exact rank-zero and rank-one support handling; do not use diagonal jitter or eigenvalue flooring.
- Explicitly reject unsupported ill-conditioned inputs rather than silently regularizing them.


### 3.3 Preferred validated-integration reference: FLINT Acb

The current FLINT 3.7.0-dev numerical-integration documentation (generated 25 November 2026) provides a substantially stronger route than hand-authoring a new interval integrator:

- `acb_calc_integrate` returns a complex ball enclosure for a finite-path integral.
- It combines direct ball enclosures on subintervals, Gauss–Legendre quadrature with rigorous remainder bounds derived from the maximum modulus on Bernstein ellipses, and adaptive subdivision.
- The callback type is `acb_calc_func_t`: for `order=0` it encloses the function on the input complex ball; for `order=1` it must also establish holomorphy on that ball (otherwise return a non-finite enclosure). The current integrator normally needs only orders 0 and 1, not higher derivatives.
- Its return status is `ARB_CALC_SUCCESS` or `ARB_CALC_NO_CONVERGENCE`. Crucially, the returned result is still a valid enclosure in either case (when finite); no-convergence means the requested width was not reached, not that the result is an invalid point estimate. Pass the enclosure onward with its actual radius, and mark the decision unresolved/refine if it is too wide.
- Precision and tolerance must be set jointly: FLINT documents that total error may exceed the local goal because of the number of segments, cancellation and rounding. Record the actual returned radius and non-convergence status rather than equating `rel_goal` with a guaranteed final relative error.
- The documentation explicitly warns that quadrature-rule difference estimates can be fooled, which is exactly the limitation we must avoid with the #13951 Simpson refinement indicator.

For the full-rank bivariate conditional integral, the real-axis integrand is built from the Gaussian density and normal-CDF interval mass. Since the Gaussian CDF is expressible using the entire error function, this has an entire complex continuation for fixed covariance parameters when the conditional standard deviation is positive. That makes the FLINT analytic-path approach a plausible direct fit for rank-two SPD fixtures. This analyticity assumption must be implemented and checked in the Acb callback, not simply asserted in a comment. Rank-one and rank-zero covariance stay on their existing exact support paths.

The finite integration interval still requires a separately enclosed omitted-tail contribution. For a standard-normal first coordinate and K>0, a conservative two-sided Mills bound is

[
P(|Z|>K) le rac{2phi(K)}{K}.
]

Compute that bound in ball arithmetic and increase K until the tail contribution is below the requested probability-width budget. Do not hard-code K=12 as sufficient for every rare-tail target. This also makes the truncation budget decision-dependent without confusing it with the integration remainder.

**Implementation recommendation:** first build a minimal standalone C reference that calls FLINT `acb_calc_integrate` for the bivariate integrand, with Acb special-function evaluation and an explicit ball tail bound. Freeze its version and build configuration. Then call it independently from the Rust #13951 implementation for cross-validation. Only consider a production Rust/FFI dependency after measuring interface complexity, portability, performance, licence/build implications, and the value of a second independent implementation. A second validated method (for example a direct interval-panel or Taylor/Cauchy-bound implementation) should be kept as a cross-check rather than assuming one library is infallible.

FLINT's documentation describes the method and its assumptions: https://flintlib.org/doc/algorithms/ball-arithmetic/integration.html. FLINT integrates Arb/Acb ball arithmetic; the old standalone Arb documentation states that Arb was merged into FLINT in 2023. The FLINT 3.7.0-dev documentation notes that its documentation was undergoing a major revision in October 2026 and that some parts were generated by an LLM; therefore pin a released header/library and test against it instead of treating the development documentation as a stable API contract. The callback and return-status semantics above should be checked against the exact pinned `acb_calc.h` and source used for the reference build.

### 3.2 Independent reference strategy

Use two distinct paths:
- a low-dimensional enclosure implementation in the Rust research module, with explicit panel/tail bounds;
- an independent high-precision ball-arithmetic reference such as Arb/FLINT, or another documented interval package.

Arb/FLINT is a candidate oracle, not a presumed production dependency. Its documentation describes rigorous ball arithmetic and integration support, while also warning that interval dependency can overestimate and that generic integration facilities require care. Benchmark enclosure tightness and failure modes rather than assuming that interval arithmetic is automatically sharp.

## 4. Decision-sufficient refinement

For a finite action a, state s, and terminal utility u(a,s), probability enclosures induce bounds on expected value:

\[
V(a)=\sum_s p_s u(a,s)-c(a).
\]

The finite decision adapter should preserve constraints already known about the probability vector: normalization, non-negativity, and validated marginal identities. Compute conservative action-value bounds under those constraints; a small linear program is a suitable first exact evaluator for finite outcome tables.

An action a* is numerically certified as uniquely best only when

\[
V^-(a^*) > \max_{a\ne a^*} V^+(a).
\]

Otherwise:
1. identify the cells and numerical components with the largest possible contribution to the unresolved action-value gap;
2. allocate additional integration budget to those cells;
3. recompute the affected enclosures without changing the scientific model, evidence state, or measurement budget;
4. stop when the decision is separated, the hard computation budget is reached, or the result is otherwise unsupported;
5. report UnresolvedWithinBudget and abstain when the ranking remains unresolved.

This is numerical refinement, not a new experiment planner. #13567 remains the owner of policy-tree optimization, non-anticipativity, and resource budgets. #13739 remains the owner of selector benchmarking.


## 5. Authored finite action-value enclosure reference

`scripts/research/sdcb_decision_value_bounds.rs` implements the finite box-plus-normalization subproblem: given cell probability intervals `[l_i,u_i]`, and finite per-cell utilities `v_i`, bound

`min/max sum_i p_i v_i` subject to `l_i <= p_i <= u_i` and `sum_i p_i = 1`.

It uses the LP dual identity

`E[v] = lambda + sum_i (v_i-lambda) p_i`

and evaluates utility breakpoints as dual multipliers. Each multiplication/addition is expanded outward by one binary64 representable step. An exact binary64 probability-sum accumulator (integer units of 2^-1074) checks feasibility of the interval box plus exact unit-sum constraint without a rounded sum comparison. It selects a unique best action only when that action's lower value bound strictly exceeds every competing upper bound.

Nine Rust tests are authored for exact unit sums, decimal-float overage, a 3-cell utility fixture, exact probabilities, negative utilities, infeasible boxes, unique action separation, unresolved overlap and malformed inputs. A separate Python oracle using exact `Fraction.from_float` summation and `nextafter` outward arithmetic passed the fixed fixtures and 2,000 randomized box-plus-normalization LP-extrema checks; a separate exact-sum bit-accumulator check matched `Fraction.from_float` on 10,000 random vectors of up to 64 cells. This is independent authoring-runtime verification, not Rust execution or a formally verified compiler/arithmetic proof.

**Arithmetic qualification assumption:** the one-step outward expansion relies on correctly rounded IEEE-754 binary64 basic operations with gradual underflow and without flush-to-zero / fast-math transformations. The target architecture, compiler flags, and floating-point environment must be independently qualified before this helper is promoted from a conservative research bound to a certified numerical contract.

Current scope is deliberately limited to interval boxes plus one normalization equality. Extra pairwise marginal or cross-moment constraints are not handled by this module; they require a richer constrained LP. This is an evaluator only: it does not construct experiments, mutate #13805, authorize actions or validate the supplied probability enclosures themselves.

## 6. Invariants and negative controls

- Every certified interval contains every frozen analytical/high-precision known answer.
- Joint cells remain nonnegative and jointly normalized within the sum of their enclosure widths; marginal constraints are preserved.
- Duplicate channels and rank-one support never create independent-evidence multiplication.
- Perfect positive/negative correlation controls preserve support geometry.
- Probability complements and marginal recovery are checked with interval arithmetic.
- Rare-tail tests cover small probabilities where relative error matters; do not rely on absolute tolerance alone.
- Increasing precision or subdividing panels must not exclude a previously enclosed exact value.
- Budget exhaustion never returns a point result labelled certified.
- Numerical uncertainty alone never changes #13805 evidence state or authorizes a physical action.

## 7. Benchmark protocol

Compare on identical inputs, hardware, precision, and resource budgets:

1. #13951 adaptive-Simpson point estimate and its current refinement indicator;
2. deterministic low-dimensional quadrature references;
3. Genz–Bretz randomized QMC and supported Miwa/TVPACK cases as established MVN baselines;
4. Peterlin's 2026 doubly blocked Genz method on its published high-dimensional SPD workloads, with the exact workload and hardware reproduced before comparative claims;
5. Arb/FLINT or a second validated-arithmetic implementation as an independent enclosure reference.

Do not compare the bivariate support-aware method's runtime with a high-dimensional SPD throughput result as if the workloads were equivalent. The Peterlin paper's reported near/above-100x speedups apply to its tested large-dimensional settings; they do not establish singular-PSD correctness or decision quality for SDCB.

### Peterlin 2026 replication controls

The paper's experimental setup is specific enough to reproduce rather than treating the headline speedup as a generic target:

- It tests SPD covariance matrices only. Its four matrix families are random low-rank-plus-identity, equicorrelated, and two Matérn constructions. The authors explicitly note that singular-covariance support is available in mvtnorm but absent in the other compared packages.
- Its main systems are an Apple M2 Ultra and a dual-socket Intel Xeon server, with Apple Accelerate and Intel MKL respectively. The paper says the cross-system timings are not intended as controlled cross-system comparisons.
- It uses Julia 1.12.x and R 4.4/4.5.2, 16 Julia threads on the M2 Ultra, and one BLAS thread for the main integration kernel to avoid oversubscription.
- The main comparisons use FP64; FP32 variants are reported separately. Each configuration is repeated at least 100 times and the paper reports minimum runtime, estimate mean, empirical estimate standard deviation, and mean integration error.
- Preprocessing such as Cholesky factorization and variable reordering is included; package loading and first-use compilation are excluded.
- Normalize the total QMC work, not the nominal argument: the paper notes tlrmvnmvt interprets N as points per repetition and performs 20 repetitions, while the other implementations interpret qmc_pts as total points.
- The paper's reported near/above-100x advantage is strongest for large dimensions (n > 2^9) in the tested settings; its independent accuracy control is a special equicorrelation reduction, not arbitrary-covariance certification.

Our reproduction matrix should therefore separate (a) exact paper replication on the same CPU/BLAS conditions where feasible, (b) same-machine cross-method comparisons on our own target hardware, and (c) SDCB-specific singular-support and decision-certified tasks. Report minimum and median/quantile timings, not only a best-case minimum, and retain the paper's QMC/replication budget normalization.

### Metrics

- enclosure containment rate on frozen known answers;
- absolute and relative enclosure width, stratified by probability scale;
- certified probability cells per second and time-to-target-width;
- peak memory and hard-budget compliance;
- fraction of decisions certified without refinement;
- fraction unresolved at each budget;
- false action-selection rate against exact small-world evaluators (target zero);
- regret and abstention quality where the exact best action is known;
- Pareto frontier of decision certification rate, enclosure width, latency, memory and computation budget.

Publish all failed, unsupported, and unresolved cases. Do not collapse correctness, enclosure tightness, and throughput into one score.

## 8. Acceptance gates (targets, not achieved results)

- 100% containment of all frozen analytical and independently validated references.
- Zero results labelled certified when any error component is unbounded or heuristic.
- Zero incorrect certified action rankings on the exact finite-world benchmark.
- Zero false-independent-confirmation claims on duplicate/shared-support controls.
- Zero budget overruns and zero silent precision/covariance repair.
- A declared fraction of supported bivariate cases reach useful enclosure widths within a frozen resource cap; preregister that fraction after a pilot, before blind test evaluation.
- Decision refinement reduces median computation on clear-margin cases relative to a fixed stringent tolerance, without worsening false-ranking or containment gates.
- Unresolved cases remain abstentions, not forced rankings.

## 9. Implementation sequence

1. Audit all current numerical error sources in #13944/#13951 and write an explicit error ledger.
2. Implement certified normal interval/tail enclosures and validate them independently.
3. Implement the conservative panel enclosure for bivariate rectangles, including rank-aware exact support cases.
4. Add frozen analytical, high-precision, metamorphic, rare-tail, and adversarial near-singular fixtures.
5. Add the finite utility-bound evaluator and exact decision-separation rule.
6. Connect the bounded result to #13567 using an adapter only; do not create a second planner.
7. Benchmark against the matched numerical comparators and publish enclosure-width/latency Pareto results.
8. Only after the low-dimensional path is independently qualified, investigate validated higher-dimensional QMC/importance-sampling enclosures.

## 10. References

- Peterlin, J. (2026), “Doubly blocked variant of Genz's method for multivariate normal integration,” Computational Statistics 41(6), DOI: https://doi.org/10.1007/s00180-026-01794-8.
- Genz and Bretz, Computation of Multivariate Normal and t Probabilities, Springer, and the R package mvtnorm: https://cran.r-project.org/package=mvtnorm.
- Arb/FLINT documentation: https://flintlib.org/ and https://arblib.org/.
- Existing implementation and ownership: #13903, #13904, #13940, #13944, #13945, #13951, #13567, #13739, #13805.
