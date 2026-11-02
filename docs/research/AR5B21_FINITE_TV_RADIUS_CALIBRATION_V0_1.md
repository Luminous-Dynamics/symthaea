# AR-5B21: Finite-Sample TV Radius Calibration for Complete Experiment Paths v0.1

**Status:** dependent research draft. Eight Rust tests authored, not compiled or executed.  
**Base:** AR-5B20 / PR #9924.  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/9925  
**Branch:** research/ar5b21-finite-tv-calibration-v0.1-20261102.

## 1. Objective

AR-5B20 accepts path-space total-variation (TV) radii and accounts for their simultaneous null-family coverage failure in the type-I budget. It deliberately does not estimate those radii.

AR-5B21 adds a finite-sample calibration method for a restricted but mathematically explicit setting:
- a finite, predeclared categorical support of complete experiment paths;
- independent, identically distributed complete-path calibration draws for each policy/model pair;
- a fixed nominal probability vector on the same ordered support;
- calibration data independent of the confirmatory stream.

This is a reference calibration method, not a general solution for continuous, high-dimensional, dependent, adaptive or nonstationary trajectories.

## 2. Statistical construction

For a given policy/model pair, let:
- \(K\) be the number of categories in the complete-path support;
- \(n\) be the number of independent calibration paths;
- \(\widehat P_n\) be the empirical categorical distribution;
- \(P_0\) be the frozen nominal path distribution;
- \(P^\star\) be the unknown calibration-path distribution.

First compute empirical distance to the nominal model:

\[
d_{\mathrm{TV}}(\widehat P_n,P_0)
=\frac12\sum_{k=1}^{K}
|\widehat P_n(k)-P_0(k)|.
\]

Weissman, Ordentlich, Seroussi, Verdú and Weinberger (HP Labs, 2003) give the finite-alphabet concentration inequality:

\[
\Pr\left(\|\widehat P_n-P^\star\|_1\ge x\right)
\le 2^K \exp(-nx^2/2).
\]

Since TV is half the L1 distance, a conservative per-row failure probability \(\delta_i\) gives:

\[
r_i
=\sqrt{\frac{K\ln 2+\ln(1/\delta_i)}{2n}}.
\]

By the triangle inequality, with probability at least \(1-\delta_i\):

\[
d_{\mathrm{TV}}(P^\star,P_0)
\le
d_{\mathrm{TV}}(\widehat P_n,P_0)+r_i.
\]

The reported radius is capped at one:

\[
\epsilon_i=
\min\left(1,\,
d_{\mathrm{TV}}(\widehat P_n,P_0)+r_i
\right).
\]

The implementation uses \(2^K\), not the slightly tighter \(2^K-2\), for a simple conservative bound. For a one-category support the distribution is known exactly and the radius is zero.

## 3. Simultaneous coverage across policies and models

There are multiple calibration rows: one for every policy × null-model pair and one for every policy × alternative-model pair.

For a null-family failure budget \(\delta_0\), allocate:

\[
\delta_i=\frac{\delta_0}{N_0},
\]

where \(N_0\) is the total number of policy × null-model calibration rows. The union bound gives simultaneous coverage of every null row with probability at least \(1-\delta_0\), without requiring the rows themselves to be independent.

Take the maximum row radius over policies for each model. This produces one null-model radius that is valid for every supplied policy. The alternative family receives its own budget \(\delta_1\) and the same procedure. Its support bounds for utility and evaluator cost are carried forward into AR-5B20.

The null-family \(\delta_0\) must be smaller than nominal alpha. AR-5B20 then spends it using:

\[
\alpha_c=\frac{\alpha-\delta_0}{1-\delta_0},
\qquad
\delta_0+(1-\delta_0)\alpha_c=\alpha.
\]

The alternative failure budget is kept separate because it qualifies alternative performance bounds, not type-I validity.

## 4. Implementation and fail-closed controls

New module:

**crates/domains/symthaea-architecture-engineering/src/finite_tv_radius_calibration.rs**

The request binds:
- the canonical nominal plan SHA-256;
- unique policy and model IDs;
- a path-support SHA-256 for each row, including category ordering;
- a calibration-data SHA-256 for each row;
- observed categorical counts and nominal category probabilities;
- null and alternative simultaneous failure budgets;
- IID complete-path and confirmatory-independence declarations;
- alternative utility and cost support bounds.

The calibrator:
1. canonicalizes and validates the request;
2. checks exact policy/model coverage against the nominal plan;
3. verifies support sizes and support digests agree across policies for each model;
4. verifies count totals, probability simplices, and alternative utility/cost bounds;
5. allocates family failure budgets across every row;
6. computes empirical TV, concentration slack and per-row radius;
7. takes the maximum radius over policies for each model;
8. produces a typed AR-5B20 ambiguity envelope;
9. emits a report with every row's support size, sample size, empirical TV, allocated delta, slack, radius and source digest;
10. verifies reports by deterministic full reconstruction.

The module has eight authored tests:
- hand-calculated empirical-TV and concentration arithmetic;
- singleton-support zero-radius control;
- simultaneous delta allocation and maximum-over-policy aggregation;
- end-to-end handoff into AR-5B20;
- stale plan digest and missing-policy rejection;
- non-IID, dependent calibration and support mismatch rejection;
- invalid counts, probability simplex and support bounds;
- canonical input ordering and report tampering.

No Rust test has run. No compile, format, Clippy, CI or benchmark pass is claimed.

Exact-head commands:

~~~bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering finite_tv_radius_calibration -- --nocapture
cargo test -p symthaea-architecture-engineering tv_robust_policy_frontier -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
~~~

## 5. Hand-checkable fixture

For a binary complete-path support with \(K=2\), \(n=10{,}000\), empirical distribution equal to the nominal distribution, and four rows in each family:

- Null-family \(\delta_0=0.01\), so each null row receives \(\delta_i=0.0025\).
- Alternative-family \(\delta_1=0.02\), so each alternative row receives \(\delta_i=0.005\).
- Null TV radius is approximately 0.0192064558.
- Alternative TV radius is approximately 0.0182819744.
- At nominal \(\alpha=0.05\), conditional alpha is approximately 0.0404040404.
- With the largest nominal null size 0.015, the largest robust null size is approximately 0.0342064558, leaving the robust LP feasible.

These values were independently checked with direct high-precision arithmetic. They are analytical fixture values, not coverage simulations or observed experiment results.

## 6. Scaling limitation is fundamental

The bound has a \(\sqrt{K/n}\) dependence. Complete-path support size often grows exponentially with horizon and branching factor. This means the conservative confidence radius can reach one before the sample size becomes operationally useful.

For example, at fixed \(n=10{,}000\) and \(\delta_i=0.0025\), increasing \(K\) quickly expands the slack. A support of tens of thousands of possible paths can make the method uninformative. Capping the radius at one is correct and fail-closed; it does not solve the statistical curse of path enumeration.

This is a small finite-support reference method and an independent calibration baseline—not the production endpoint.

## 7. Validation and benchmark programme

Before qualification:
- independently reproduce the formula over randomized \(K,n,\delta\) grids;
- simulate multinomial coverage including rare and zero-probability cells;
- check empirical coverage against exact multinomial calculations for small \(K\);
- include clustered and dependent data as negative controls that must not be accepted under the IID request;
- vary policy count, model count, path support size, horizon and calibration size;
- measure radius width and fraction of cases where the robustified LP remains feasible;
- feed generated envelopes into AR-5B20 and reconstruct the final decision receipt;
- test calibration reuse, support leakage, category permutation and nominal-model version drift.

Compare the Weissman union-bound radius with exact multinomial confidence regions, likelihood-ratio regions, and bootstrap methods only when each comparator has a separate coverage argument. Do not compare radius width while ignoring actual coverage.

Research reference:
- T. Weissman, E. Ordentlich, G. Seroussi, S. Verdú and M. J. Weinberger, *Inequalities for the L1 Deviation of the Empirical Distribution*, HP Laboratories Technical Report HPL-2003-97R1 (2003). https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/2003/HPL-2003-97R1.pdf
- The empirical-distribution inequality is also summarized with its finite-alphabet assumptions in the formal theorem entry at https://prove2.me/theorems/c4a181b3-6f64-4da6-a10a-07996e06e1d1.

## 8. Next paradigm shift: stop enumerating complete paths

The complete-path method is deliberately a reference oracle. A more scalable next step is to calibrate conditional transition kernels or conditional outcome distributions and propagate uncertainty through a robust finite-horizon dynamic program.

That route must prove how local confidence sets compose into a global path-space guarantee. A naive per-step epsilon is invalid. A valid composition bound, confidence sequence or robust Bellman recursion could avoid explicitly enumerating every complete trajectory, while retaining a clear connection to the type-I error budget.

Other extensions, each requiring its own theorem and validation:
- clustered / block-IID path calibration;
- Markov or martingale concentration for dependent paths;
- adaptive discretization with an explicit approximation-error budget;
- cost-aware, support-aware ambiguity sets (including generalized TV);
- Wasserstein or KL ambiguity sets when geometry and support make them tighter;
- finite-support robust dynamic programming and column generation.

## 9. Claim boundary

The calibration certificate is valid only under the declared finite support, known category ordering, IID complete-path sampling, frozen nominal probabilities, and independent calibration/confirmatory streams. The implementation cannot prove those premises from metadata or hashes.

It does not establish validity for arbitrary continuous, unbounded, dependent, clustered or nonstationary trajectories. Large-support conservatism is expected. No SOTA claim or real-experiment performance claim is made.

**The intended contribution is a reproducible, fail-closed finite-support radius oracle that connects external statistical coverage to the robust sequential decision solver.**
