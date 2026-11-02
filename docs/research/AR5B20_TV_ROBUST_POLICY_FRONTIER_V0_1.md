# AR-5B20: TV-Robust Policy-Frontier Optimization with Explicit Coverage Spending v0.1

**Status:** dependent research draft. Eight Rust tests are authored but have not been compiled or executed.  
**Base:** AR-5B19 / PR #9899 at SHA 3ce2a927d05737c9862bc3b4ce6fcbf89f1d0610.  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/9924  
**Branch:** research/ar5b20-tv-robust-frontier-v0.1-20261102.

## 1. Why this layer is necessary

AR-5B19 solves an exact finite LP over a supplied finite policy frontier and finite model families. Its arithmetic can be exact while the declared model family is incomplete. A policy may look optimal under every supplied null and alternative but fail under a nearby data-generating process.

AR-5B20 adds an explicit distributional uncertainty envelope around the policy receipts. The first version uses total variation (TV) because its bounded-expectation inequality is transparent, auditable and computationally cheap. The radius is not inferred by this module; it is supplied with an external coverage receipt and remains an assumption until independently justified.

The envelope is defined over the **complete finite-horizon observable path distribution** for each model and must hold uniformly for every supplied policy. It is not a per-step radius. Reusing one epsilon at every observation would be invalid without a separate composition argument: repeated per-step perturbations can compound substantially over a long experiment.

## 2. TV convention and robust coefficients

Use the convention:

\[
d_{\mathrm{TV}}(P,Q)=\sup_A|P(A)-Q(A)|.
\]

For every measurable function \(f\) with \(L\le f\le U\),

\[
|\mathbb{E}_{P}f-\mathbb{E}_{Q}f|
\le d_{\mathrm{TV}}(P,Q)(U-L).
\]

For a policy \(\pi\), a nominal path law \(P_m\), and every true path law \(Q_m\) satisfying \(d_{\mathrm{TV}}(P_m,Q_m)\le\epsilon_m\), transform its receipts as follows.

**Null size, where the rejection function lies in [0,1]:**

\[
s^{\mathrm{rob}}_{\pi,m}
=\min(1,s_{\pi,m}+\epsilon_m).
\]

**Alternative power, where the rejection function lies in [0,1]:**

\[
p^{\mathrm{rob}}_{\pi,m}
=\max(0,p_{\pi,m}-\epsilon_m).
\]

**Evaluator cost, when the random cost is in [0,C_m]:**

\[
c^{\mathrm{rob}}_{\pi,m}
=\min(C_m,c_{\pi,m}+\epsilon_m C_m).
\]

**Decision utility, when the random utility is in [L_m,U_m]:**

\[
u^{\mathrm{rob}}_{\pi,m}
=\max(L_m,u_{\pi,m}-\epsilon_m(U_m-L_m)).
\]

These are deliberately conservative coefficient bounds. They preserve the LP structure in AR-5B19: substitute robust size, power, cost and utility coefficients, then solve the same finite policy-frontier problem.

The cost and utility support bounds must contain every nominal receipt. A digest binds the bounds and coverage evidence to a specific artifact but does not authenticate its origin or establish its truth.

## 3. The important correction: estimated radii consume alpha

A deterministic TV bound has no coverage failure. A radius estimated from a calibration sample does.

Let \(\delta_0\) be the failure probability of a **simultaneous coverage event for the complete declared null family**. The calibration data must be independent of the confirmatory data stream. On the coverage event, the robustified conditional type-I probability is bounded by \(\alpha_c\). Outside it, use the conservative bound one. Thus:

\[
\Pr_0(\mathrm{reject})
\le \delta_0+(1-\delta_0)\alpha_c.
\]

To preserve the nominal unconditional level \(\alpha\), spend the budget using:

\[
\boxed{\alpha_c=\frac{\alpha-\delta_0}{1-\delta_0}},
\qquad 0\le\delta_0<\alpha.
\]

Then \(\delta_0+(1-\delta_0)\alpha_c=\alpha\). If \(\delta_0\ge\alpha\), fail closed; there is no positive conditional error budget.

Alternative-family coverage failure \(\delta_1\) is reported separately. It qualifies the alternative power, utility and cost bounds, but is not silently charged to the null type-I budget. This separation is intentional: type-I validity and alternative performance are different claims.

This construction requires:
1. simultaneous coverage for all declared null path laws and all policies;
2. calibration/confirmatory independence;
3. a valid external derivation of the TV radii and their failure probability;
4. an appropriate finite-horizon path-space definition.

The module validates declared metadata and reconstructs arithmetic. It cannot prove these statistical premises from SHA-256 strings.

## 4. Implementation

New module:

**crates/domains/symthaea-architecture-engineering/src/tv_robust_policy_frontier.rs**

It provides:
- TvAmbiguityEnvelope and typed null/alternative path-radius records;
- canonical source-plan binding by SHA-256;
- exact model-family matching, finite TV-radius validation, and cost/utility support checks;
- independent-calibration requirement;
- null-family confidence-failure spending through conditional alpha;
- robust coefficient construction followed by the AR-5B19 frontier solver;
- a report containing nominal and conditional alpha, null/alternative coverage-failure probabilities, source/envelope/robustified-plan digests, and an unconditional type-I upper bound;
- deterministic full-report reconstruction.

The implementation does not mutate the nominal plan or AR-5B19. It is an adapter around that reference solver.

## 5. Tests authored

Eight tests cover:
1. exact TV coefficient transformation and frontier reconstruction;
2. spending null coverage failure from alpha;
3. zero-radius, zero-failure identity control;
4. stale nominal-plan digest rejection;
5. omitted-model and dependent-calibration rejection;
6. invalid confidence budgets and support bounds;
7. infeasible robustification rejection;
8. report-tampering detection.

### Independent LP fixture check

For the committed two-policy fixture, the independently encoded robust LP was solved with SciPy 1.17.0 / HiGHS. It returned:
- policy-a weight 0.6 and policy-b weight 0.4;
- worst-case decision utility 6.7;
- robust alternative powers 0.700 and 0.738;
- robust null sizes 0.0208 and 0.0320;
- nominal alpha 0.05, null-radius coverage failure 0.0025, conditional alpha 0.047619047619;
- unconditional type-I upper bound 0.03442;
- alternative evaluator costs 3.7 and 7.3, below the cap of 9.

The Rust regression test now locks these values. This independent solve checks the finite LP formulation and synthetic fixture; it does **not** execute the Rust basis enumerator, prove its numerical behavior on all fixtures, or establish empirical performance.

No Rust test has run. No compile, formatting, Clippy, CI or benchmark pass is claimed.

Exact-head commands:

~~~bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering tv_robust_policy_frontier -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
~~~

## 6. Research and industry comparison

### Statistical foundations

- Fontem and Ji (2026), *Distributionally robust optimization with generalized total variation ambiguity sets*, develops ambiguity geometry that depends on sample size, support, confidence level and cost structure. AR-5B20 is a bounded TV-envelope adapter, not an implementation or reproduction of GTV. https://doi.org/10.1016/j.ejor.2025.09.024
- Rahimian and coauthors' distributionally robust optimization review discusses TV ambiguity sets, Huber contamination and worst-case expectation formulations. https://optimization-online.org/wp-content/uploads/2019/08/7332.pdf
- *Statistical testing under distributional shifts* (JRSSB, 2023) formalizes testing for a target distribution when observations arise from a shifted distribution. Its setup and assumptions differ from this finite path-space envelope. https://doi.org/10.1093/jrsssb/qkad018
- *Safe testing* (JRSSB, 2024) develops e-value methods for optional continuation and composite nulls. It is a relevant sequential-validity baseline, not evidence that TV robustness is solved. https://doi.org/10.1093/jrsssb/qkae011

### Industry baseline

Spotify's July 2026 public report states that its always-valid sample-size correction produces 8–20% sample savings in simulations, with 14% at common alpha=0.05 / 80%-power WSKR settings and a 9.5% median saving across 713 metrics from 283 production experiments. The company reports these as its simulations and production evaluation; they are not independently replicated here. The article also reports empirical power within approximately three percentage points of target over its simulation grid. https://confidence.spotify.com/blog/accurate-sample-size-always-valid

Spotify's published methodology compares group-sequential tests, mSPRT/GAVI-style always-valid methods and corrected-alpha approaches; it emphasizes the dependence of method choice on streaming versus batch arrivals and sample-size knowledge. https://engineering.atspotify.com/2023/3/choosing-sequential-testing-framework-comparisons-and-discussions

Spotify's 2026 Experiments with Learning framework reports an approximate 64% learning rate versus a 12% win rate. That motivates evaluating decision-relevant learning, neutral conclusions and regressions—not optimizing only for win rate. These are Spotify-reported operational metrics. https://www.engineering.atspotify.com/2025/9/spotifys-experiments-with-learning-framework

## 7. Benchmark programme: what would count as better

Freeze scenarios, random seeds, estimands, stopping rules, costs, model families and baseline implementations before comparing methods. Use paired seeds and independent simulation code. Include:

- clean nulls and alternatives;
- Huber-style contamination at several levels;
- omitted-null and omitted-alternative models;
- support changes, tail reweighting and heavy tails;
- serial dependence, clustering and unequal allocation;
- delayed outcomes and asynchronous arrival;
- time-varying effects and nonstationarity;
- ratio metrics, low-base-rate guardrails and multiple metric families;
- deliberate calibration failure and calibration/test leakage negative controls.

Compare at matched unconditional type-I guarantees:
1. fixed-horizon z/t procedures;
2. Bonferroni-adjusted repeated looks;
3. Lan–DeMets group-sequential alpha spending;
4. always-valid mSPRT / confidence-sequence / e-process methods;
5. nominal AR-5B19;
6. TV-robust AR-5B20.

The primary outcomes should be:
- unconditional type-I error and a one-sided confidence upper bound;
- conditional power on the coverage event and unconditional power;
- false-ship and false-stop expected loss;
- expected and tail sample cost;
- evaluator compute cost and wall-clock latency;
- learning yield: the fraction of runs producing a valid, decision-changing conclusion (win, regression or conclusive neutral);
- coverage of the radius certificates;
- worst-case decision regret across declared stress families;
- robust utility versus cost frontier and the price paid for robustness.

### Proposed gates (targets, not results)

| Dimension | Qualification target |
|---|---:|
| Unconditional type-I | empirical upper confidence bound ≤ nominal alpha + 0.001 |
| Null-family coverage | ≥ 99% simultaneous coverage at the declared confidence level |
| TV arithmetic | exact agreement with independent high-precision fixtures |
| Randomized LP differential | 100% feasibility agreement; objective gap ≤ 1e-8 on bounded fixtures |
| Clean-regime efficiency | no more than 5% median sample-cost penalty versus the best valid baseline |
| Stress-regime decision loss | ≥ 10% lower worst-case loss than nominal AR-5B19 at matched error |
| Calibration leakage | every negative control detected; otherwise fail qualification |
| Reproducibility | identical report digest for identical canonical inputs |

The 10% decision-loss target is a research goal, not a claim of superiority. A robust method that merely becomes conservative and never rejects is not a success; its power, cost, learning yield and regret must all be reported.

## 8. Limitations and next research frontier

- This is TV robustness around supplied receipts, not a general distributionally robust sequential test.
- A TV radius over a complete path law may be hard to estimate and can be conservative. Per-observation contamination must not be substituted directly for a path-space radius; the joint law can move much farther from nominal as the horizon grows.
- The envelope only protects against deviations inside its declared set. Omitted model families remain a threat.
- The module does not authenticate calibration data, establish coverage, or generate the radii.
- The LP still optimizes over at most six supplied deterministic policies and does not generate new policies.
- Alternative-family coverage affects power/utility/cost claims and needs independent audit.
- The next stage is a data-backed radius-calibration method, then a finite-support DRO oracle using CVaR/TV geometry, then a scalable LP or column-generation solver cross-checked against the exact reference.

**No SOTA claim is made.** A SOTA comparison requires matched baselines, locked workloads and seeds, independent solver agreement, empirical calibration checks, and exact-head Rust execution.
