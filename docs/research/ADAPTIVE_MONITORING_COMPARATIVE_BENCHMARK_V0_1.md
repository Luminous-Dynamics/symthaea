# Adaptive Monitoring Comparative Benchmark v0.1

**Status:** executable, deterministic Rust integration benchmark proposed in PR #8457; exact-head Rust CI has not yet completed. The narrow simulation compares detector behavior, not full scientific-law discovery quality and not state-of-the-art performance.

## 1. Question being tested

Can a monitor detect a conditional predictive failure that is invisible to pooled directional and dispersion summaries, when experimental actions are known before the outcome and the monitoring null remains valid?

The benchmark compares three policies on the same paired synthetic streams:

1. **Pooled prequential PIT portfolio:** Order/Dispersion betting with only the intercept channel.
2. **Action-contextual PIT portfolio:** identical detector with one predeclared action-sign channel.
3. **Fixed-reference PRM:** an Order/Dispersion predictive-rank martingale fed the same PIT sequence, with a frozen uniform calibration reference.

The null generator provides conditionally Uniform(0,1) PITs and an alternating, predeclared action. The alternative draws PITs from opposite interior bands by action: action +1 receives U(0.65, 0.85); action -1 receives U(0.15, 0.35). By construction, pooled Order and Dispersion means are approximately zero; action-times-Order has positive mean. This is a focused contrast diagnostic, not an attempt to represent every physical model failure.

The PRM is evaluated under its appropriate clean null, where calibration and online PIT scores are jointly exchangeable. Under the contextual alternative, only the conditional PIT model-based contract is deliberately targeted; PRM is included as an empirical comparator for this fixed feature portfolio, not declared invalid or universally weak.

## 2. Current diagnostic replay

Before running the Rust integration test, a separate JavaScript implementation using the same fixed-seed PRM/PIT update equations generated the following reference outcomes for 256 streams, 300 observations per stream, a reference of 100 ranks, and alpha=0.05:

| Stream family | Pooled PIT alarm rate | Contextual PIT alarm rate | PRM alarm rate |
|---|---:|---:|---:|
| Conditional-uniform null | 2.7% | 4.3% | 3.5% |
| Marginally balanced context shift | 0% | 100% | 15.2% |

These values are **algorithmic replay results, not Rust CI results**. The Rust test uses a deterministic generator and broad regression bounds; it reports the exact rates and median detection delays. Reproduction on the exact branch head is pending. The 256-stream rates are too small to establish formal Type-I control, and the feature-aligned alternative was deliberately chosen to exercise the contextual channel. Independent confirmatory streams and confidence intervals are still required.

## 3. Reproducible entry point

Run the integration test with:

```sh
cargo test -p symthaea-physics-bridge --test adaptive_monitoring_benchmark -- --nocapture
```

The test freezes alpha=0.05, horizon=300, reference count=100, and 256 paired streams per world. It uses one deterministic xorshift64* stream for reproducible test fixtures only; it is not a production random generator. All detectors see the same policy-visible PIT path within a stream. Calibration reference samples and tie-breakers are separate draws, and hidden-world labels are used only by the benchmark harness to stratify results.

The test checks a wide null smoke envelope (all alarm rates <=10%) and checks that, on the constructed alternative, the contextual detector's detection rate is >=90%, pooled detection <=10%, and PRM detection <=35%. Those are regression bounds for this deliberately selected synthetic contrast, not general claims about detector rankings. The exact Rust results must be copied into the CI evidence record before the PR can be promoted.

## 4. Statistical contract and threat model

- **Pooled PIT:** under a correct conditional predictive CDF issued before each outcome, PITs should be conditionally Uniform(0,1). Pooled Order/Dispersion betting may miss miscalibration that cancels across contexts.
- **Contextual PIT:** a fixed, predictable context (g(a_t,\mathcal F_{t-1})\in[-1,1]) multiplies a zero-mean PIT feature. Its model-based guarantee requires the conditional PIT premise and the context to be frozen before the outcome. Context features and the number of portfolio channels must be preregistered; adding channels dilutes per-channel wealth.
- **PRM:** a fixed calibration reference is retained and the exact conditional rank law is used. Its stated distribution-free anytime guarantee is marginal over the jointly sampled reference and online stream, under the paper's exchangeability/rank-filtration assumptions; it is not generally conditional on the particular realized numeric reference sample.
- **No detector:** alarm means evidence of a shift/misfit under the detector's own null, not proof that a specific physical law is false. Non-alarm is not proof of adequacy.
- **Adaptive-query validity:** raw residuals from an adaptive EIG loop are not automatically exchangeable. Do not pass them directly to PRM or turn them into PITs after the outcome. The predictive CDF must be recorded before measurement, conditional on the selected action and permitted history.

## 5. Research and industry comparisons

The broader question—does a monitoring-aware discovery loop recover valid laws with fewer measurements?—needs a separate end-to-end evaluation. Relevant public benchmarks report complementary skills and should not be collapsed into one leaderboard:

- **SciLaws-Bench (2026):** 118 tasks, six scientific disciplines, 381 papers, 291 candidate laws and roughly eight million data points. Its Real setting measures held-out predictive fit and scientific validity; its Parallel setting measures structural recovery for synthesized equation variants under active queries. It explicitly notes that good fit, recovery of published formulas, and recovery of novel terms are different claims. Project: https://yiyihum.github.io/SciLaws-Bench/
- **LLM-AutoSciLab / ActiveSciBench (NeurIPS 2026 project):** reports 57 enzyme-kinetics and 45 gene-regulatory-network tasks and evaluates closed-loop hypothesis generation, experiment selection, and mechanism recovery. Its public report gives 67.6% symbolic accuracy on NewtonBench, 35.1% on ActiveSciBench-Chem and 31.1% exact graph recovery on ActiveSciBench-GRN, and reports 2–5x sample-efficiency gains for its hypothesis-guided design. Those figures are the authors' reported results and use a different system and protocol: https://scientific-discovery.github.io/llm-autoscilab-project/
- **LLM-SRBench (ICML 2025):** 239 equation-discovery problems across four domains, with transformed familiar physics models and synthetic discovery-driven tasks. Its paper reports a best symbolic accuracy of 31.5% on its split, emphasizing memorization-resistant evaluation: https://proceedings.mlr.press/v267/shojaee25a.html
- **NewtonBench:** an additional symbolic-law recovery reference used in both newer systems. Pin its actual version/task split and matching evaluator before comparing scores: https://github.com/HKUST-KnowComp/NewtonBench

These are not directly comparable scores. A valid paper-level comparison needs exact benchmark version/split, task-package licenses, allowed prior knowledge, compute/tools, number of oracle queries, wall time, failure/abstention policy, evaluator and source digest. For external benchmark claims, reproduce the published baseline settings before claiming an improvement.

## 6. End-to-end confirmatory protocol

The new integration test is Phase 1: a cheap detector contrast regression. Phase 2 must evaluate discovery policy and valid recovery:

- At least 60 held-out structural worlds across 6 or more families and at least 20 paired seeds per world; keep the truth generator and model/selector inputs isolated.
- Compare seeded uniform, raw variance, nominal EIG, scenario-robust EIG, batch conformal audits, finite-horizon alpha spending, PRM, pooled PIT, contextual PIT, and the combined discovery policies.
- Include correctly specified clean controls, omitted terms, location and scale misspecification, heavy-tailed/heteroscedastic noise, contextual shifts that cancel marginally, gradual and abrupt regimes, rare-region/extrapolation failure, and query-cost heterogeneity.
- Report structural exact recovery, predictive score on held-out experiments, scientific validity, false confident singleton rate, calibrated alarm rate, detection power vs shift magnitude, median and 90th percentile detection delay, unresolved rate, query/measurement cost to valid recovery, compute, and wall time separately.
- Use paired candidate pools and worlds, then bootstrap worlds first and seeds within worlds. Do not count multiple seeds of the same generator as independent structural families.
- Use at least 10,000 clean-null streams for Monte Carlo calibration checks; give confidence intervals and preserve every negative-control run.
- Have an evaluator independent of the policy code compute hidden-truth metrics from signed/digested receipts. The policy-visible process must never read the true law, discrepancy amplitude, or terminal labels.

## 7. Advancement targets (not current results)

Promote a monitoring-aware discovery policy only if it satisfies all relevant conditions on the held-out confirmatory split:

- Family-wise/anytime false-alarm behavior matches the stated null contract and uncertainty intervals.
- At least 80% detection for predeclared practically important alternatives at matched cost.
- At least 20% relative reduction in false confident singletons under misspecification versus nominal EIG.
- No more than a 5-percentage-point loss in clean-world structural recovery.
- At least 20% lower median measurement cost to valid recovery or explicit unresolved status than the strongest simple matched-budget baseline, with a paired world-clustered interval supporting the direction of the effect.
- No advantage obtained by over-abstaining, increasing budgets invisibly, using hidden-truth labels, or changing thresholds/features after looking at held-out outcomes.

If the gates fail, retain the detectors as research baselines and report where they fail. Do not weaken the gates to force a win.

## 8. Claim and validation boundary

The PR adds a deterministic smoke benchmark and CI entry point. It does not prove statistical validity, reproduce external SOTA, or establish that contextual PIT beats PRM on general distributions. The broader end-to-end protocol is a next milestone. Rust execution, rustfmt, compile and Clippy status must come from completed checks at the exact PR head; queued/pending/skipped actions count as no evidence.
