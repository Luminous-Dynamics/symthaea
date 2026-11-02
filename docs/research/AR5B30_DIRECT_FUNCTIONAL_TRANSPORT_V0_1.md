# AR-5B30 — Direct Simultaneous Transport for Frozen Decision Functionals v0.1

**Status:** implementation draft; Rust tests authored but not executed.  
**Tracking:** [Issue #10055](https://github.com/Luminous-Dynamics/symthaea/issues/10055)  
**Module:** **crates/domains/symthaea-architecture-engineering/src/direct_functional_transport.rs**  
**Branch:** **research/ar5b30-direct-functional-transport-v0.1-20261102**  
**Dependency:** AR-5B27 finite-support two-sample TV transport / PR #10045.

## 1. Paradigm shift: transport the decision, not always the whole distribution

AR-5B27 constructs a confidence bound on the entire finite-support distribution discrepancy:

\[
\operatorname{TV}(P,Q)
\le
\operatorname{TV}(\widehat P,\widehat Q)+r_P+r_Q.
\]

That is valuable because it applies to **every** bounded function on the frozen support. Its Weissman-style concentration term, however, depends on support size \(K\). If a decision only uses a small registered vector of \(d\) metrics and \(K\) is large, full TV may be needlessly conservative.

AR-5B30 asks a narrower question: can we certify the target-minus-calibration expectation shift for a frozen, complete family of decision functionals directly, without paying the full support-dimension penalty?

For each registered function \(g_j\), with known range \([a_j,b_j]\), calculate empirical means under the two category-count samples:

\[
\widehat\mu_{P,j}=\sum_{k=1}^K\widehat p_k g_j(k),
\qquad
\widehat\mu_{Q,j}=\sum_{k=1}^K\widehat q_k g_j(k).
\]

For \(d\) functions and family failure allocation \(\delta\), allocate:

\[
\delta_{\mathrm{row}}=\frac{\delta}{2d}
\]

to each two-sided sample/function statement. Hoeffding's inequality gives:

\[
r_{P,j}=(b_j-a_j)
\sqrt{\frac{\ln(2/\delta_{\mathrm{row}})}{2n}},
\qquad
r_{Q,j}=(b_j-a_j)
\sqrt{\frac{\ln(2/\delta_{\mathrm{row}})}{2m}}.
\]

By a union bound over both samples and all registered functions, all shift intervals simultaneously cover with probability at least \(1-\delta\):

\[
\mathbb E_Q[g_j]-\mathbb E_P[g_j]
\in
\left[
\widehat\mu_{Q,j}-\widehat\mu_{P,j}-r_{P,j}-r_{Q,j},
\widehat\mu_{Q,j}-\widehat\mu_{P,j}+r_{P,j}+r_{Q,j}
\right].
\]

The interval is intersected with the known mathematical shift range \([-(b_j-a_j),b_j-a_j]\).

A calibration-law confidence interval \([L_{P,j},U_{P,j}]\) can then be shifted directionally:

\[
L_{Q,j}=\max\{a_j,L_{P,j}+\ell_j\},
\qquad
U_{Q,j}=\min\{b_j,U_{P,j}+u_j\},
\]

where \([\ell_j,u_j]\) is the simultaneous shift interval.

This is a direct confidence-set transport result for a frozen finite family. It is **not** a full-distribution TV bound, does not cover arbitrary functions, and cannot protect an omitted safety metric.

## 2. Implemented contracts

### 2.1 Frozen family digest

The request binds the common finite support digest and exact category order; a sorted, unique roster of stable function IDs; each function's declared finite range; its exact per-category values; both sample count streams and their source artifact digests; calibration and target design identities; family failure probability; IID complete-path declarations; independent calibration/target sample declaration; reference-sample independence from confirmation; and a pre-outcome-only target reference sample declaration.

The functional-contract digest is recomputed from the support digest and sorted function roster. A function added, removed or modified after the contract is frozen is rejected. The report digest binds the counts and all source fields. Digests bind bytes; they do not authenticate the upstream data producer or prove that the asserted sampling design was followed.

### 2.2 Per-functional report

For each registered function, the report contains calibration and target empirical expectations, observed target-minus-calibration empirical shift, separate calibration and target Hoeffding radii, the simultaneous row failure allocation, lower and upper shift-confidence bounds, and status **INFORMATIVE**, **VACUOUS**, or **EXACT_CONSTANT**.

A constant function with range width zero has exact zero shift. A family containing only constant functions is not labelled vacuous.

### 2.3 Calibration interval adapter

The typed interval adapter requires the exact registered functional ID, matching functional-family SHA-256, a SHA-256 for the calibration-interval source artifact, and a finite ordered calibration interval within the registered function range.

It combines that interval with the directional shift interval and clips the result to the registered range. It rejects unknown IDs, mismatched contracts, malformed digests, invalid intervals and non-finite results.

### 2.4 Conditional-alpha composition

The error-budget composer unions calibration null-confidence failure probability, direct-functional transport family failure probability, and any additional declared null-side confidence failure probability.

It requires the calibration functional-family digest to match the transport certificate and every contributing evidence stream to be independent of the confirmatory stream. If \(\delta_{\mathrm{total}}\ge\alpha\), it fails closed. Otherwise it returns:

\[
\alpha_{\mathrm{conditional}}
=
\frac{\alpha-\delta_{\mathrm{total}}}{1-\delta_{\mathrm{total}}}.
\]

No independence among the failure events themselves is assumed; the confidence failures are composed by union bound. Independence from the confirmatory stream is a separate contract needed for the conditional Type-I argument.

## 3. What this does not claim

V1 assumes that each calibration and target reference stream is IID complete-path sampling from its own law; the two reference samples are independent; both reference samples are independent of confirmatory outcomes; both samples use the same exact finite support and category order; every registered function is fixed before observing the samples; every function's values are known and lie in its registered finite range; and the frozen roster contains every metric whose target expectation must be protected.

It does not prove any of those source/data/design declarations. It does not handle omitted or post-hoc functional metrics, target support outside calibration support, non-IID/serial/cluster dependence, unobserved or open-ended path categories, target reference data reusing confirmatory outcomes, causal identification or changing estimands, unbounded functionals or unknown ranges, or arbitrary distributional equivalence.

If a function is omitted, the certificate says nothing about it. Hashing the roster does not establish that the roster includes every mandatory harm, quality or access metric. That responsibility remains with the domain owner and the preregistered experiment contract.

## 4. Tests authored

Twelve Rust tests are authored:

1. Exact two-sample Hoeffding shift-radius formula.
2. Large-support / small-family regime where direct functional bounds avoid the full TV support penalty.
3. Exact constant-function zero radius.
4. Simultaneous Bonferroni allocation across multiple functionals and both samples.
5. Orthogonal-shift control: a registered function remains unchanged while an omitted category indicator changes; the omitted function cannot be queried from the certificate.
6. Directional calibration-to-target interval expansion and source receipt binding.
7. Late functional roster expansion, unknown IDs and report tampering rejection.
8. IID, calibration/target independence, and confirmation-independence failures.
9. Duplicate IDs, missing category values and out-of-range values rejection.
10. Conditional-alpha error-budget composition.
11. Functional-contract mismatch and over-budget alpha rejection.
12. Report/request digest binding of sample and functional data.

**Tests are authored, not run.** No cargo test, cargo check, rustfmt, Clippy, CI, or performance result is claimed.

## 5. Formula-level matched pilot (not Rust execution)

An independent NumPy pilot compared this direct functional construction against the AR-5B27 full finite-support TV formula on a frozen high-dimensional example:

- support size \(K=512\);
- registered functional count \(d=3\);
- \(n=m=10{,}000\);
- family failure probability \(\delta=0.01\);
- calibration law uniform on 512 categories;
- target law transfers 0.01 total probability mass from the first half to the second half;
- true distribution TV distance \(0.01\);
- three fixed functions: last-half indicator, category-zero indicator, and normalized category-rank score;
- 30,000 independent calibration/target replications;
- deterministic seed 20261102.

Observed:
- simultaneous direct-functional coverage: 30,000 / 30,000 = 100%;
- exact 95% Clopper–Pearson lower confidence limit: 99.9877%;
- each direct functional shift interval width: approximately 0.07531;
- mean full-TV radius: approximately 0.39577;
- corresponding normalized bounded-function full-TV shift interval width: approximately 0.79154;
- direct/full-TV interval-width ratio: approximately 0.0951.

This is a formula-level simulation result, not a Rust test or proof. It uses a known synthetic distribution and a fixed three-function family. It does not imply a 90% improvement in other workloads. The observed finite-run 100% coverage is not proof of exact nominal coverage.

## 6. Matched benchmark programme

Compare AR-5B27 distribution-level finite-support TV; AR-5B30 direct simultaneous functional Hoeffding transport; vector empirical-Bernstein or matrix-heat confidence sets when the exact theorem applies; bounded density-ratio transport in justified overlap regimes; and conservative recalibration/abstention controls.

Freeze workloads before running the benchmark: decision-aligned shift; orthogonal shift and omitted metric; highly correlated registered metrics; rare-event safety metrics; heterogeneous function ranges; increasing support \(K\) with fixed functional count \(d\); increasing functional count \(d\); unequal calibration/target sample sizes; low-data and vacuous regimes; and invalid non-IID, target-outcome reuse and incomplete-support controls.

Report simultaneous registered-functional coverage and exact binomial confidence intervals; per-functional interval width (max/median/p95); full-TV and direct-functional width ratio; downstream guardrail false-safe/false-block rate; omitted-metric false-qualification rate; utility regret, guardrail margins and recalibration cost; sample counts, compute time, memory and report size.

### Proposed internal qualification targets (not achieved results or industry standards)

- Zero false qualification on all preregistered roster, support and sampling negative controls.
- Simultaneous coverage at or above \(1-\delta\) on an untouched lockbox, with lower binomial confidence limits published.
- Zero positive mixture mass on a policy whose required functional bound is unavailable.
- At matched coverage, at least 20% median interval-width improvement over full TV on preregistered low-dimensional tasks where only the registered family is relevant.
- Explicit non-qualification whenever the roster or support is incomplete.
- No full-distribution transport claim from a functional-only report.

## 7. Integration path

Keep AR-5B27 full TV as the conservative all-bounded-functions certificate. Use AR-5B30 only where the domain has frozen and justified the complete mandatory function roster. Convert both certificate types into common typed per-policy upper/lower confidence bounds. Feed those bounds into AR-5B26's hard guardrail-constrained LP. Compose every calibration, transport, null-side confidence and policy-family failure event before conditional-alpha spending. Compare direct Hoeffding against AR-5B28's vector confidence-set work; variance-adaptive vector methods may reduce width further, but only under their exact theorem assumptions.

The target architecture is a portfolio of correctly scoped certificates: full-distribution TV when broad protection matters, direct functionals when the decision family is finite and complete, and vector covariance-adaptive sets when their statistical model fits. No single certificate is a universal substitute for the others.

## 8. Example command

From the repository root:

~~~bash
cargo test --locked -p symthaea-architecture-engineering direct_functional_transport
~~~

Run against the exact PR head when a compatible Rust toolchain is available.
