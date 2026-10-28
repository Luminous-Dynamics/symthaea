# Groupwise-Robust Coverage vs Decision Regret (2026-10-28)

## Result

This paired **synthetic finite-game pilot** compares pooled global conformal calibration, a max-over-known-regimes common radius, per-regime radius (oracle diagnostic), and an unrestricted-prior minimax baseline. It is not a real SCM integration or an external SOTA result.

| Target | Observation budget | Method | Overall coverage | Worst-regime coverage | Mean joint regret (bits) |
|---:|---:|---|---:|---:|---:|
| 90% | 120 | Pooled global | 91.7% | 83.3% | 0.01636 |
| 90% | 120 | Max-regime common radius | 97.2% | 93.3% | 0.01875 |
| 90% | 240 | Pooled global | 93.3% | 90.0% | 0.00968 |
| 90% | 240 | Max-regime common radius | 95.0% | 90.0% | 0.01035 |

The experiment used 600 calibration and 180 independent test episodes balanced across six synthetic regimes, with nested observation budgets of 60/120/240 and targets of 90%/95%. It used 16 latent hypotheses and eight deterministic horizon-two policies. Four local unit/smoke tests passed. Maximum independent primal/dual LP certificate gap: **2.00000594e-9 bits** (numerical checks with tolerance margins, not formal interval arithmetic).

At 120 observations, the max-regime radius increased the observed weakest-regime coverage at a small regret cost. At 240 observations the marginal coverage benefit was smaller. The unrestricted-prior baseline achieved 100% empirical coverage but had mean regret around 0.10423 bits and p95 regret around 0.21491 bits.

## Evidence boundary

There are only 30 test cases per regime, so coverage estimates remain noisy. The regime taxonomy is assumed known during calibration; this does not protect against unseen regimes. The regime-conditional strategy is an oracle unless the regime is known at decision time. The ten-feature polytope does not constrain every direction of the joint prior. A larger attempted run did not finish within the local execution window and emitted no result. Do not claim it as evidence.

Raw JSON SHA-256: **00e84c2de63b7e7615844cc0957bfbe57652f258a25b0a9565e975967fef9263**  
Benchmark source SHA-256: **7334f974506bde8ed14b7dae495438f33b575efcffe0e2ad15cd7e4563b72c1e**  
Test source SHA-256: **718f2af38c206613705657821b6f32da46565fdd1e9bd7f70ea24d2480cc5982**

The benchmark and test are committed in this PR branch. The raw output, compressed output, complete summary, and full research bundle remain downloadable artifacts rather than committed result files. Local passing tests are not represented as remote CI evidence.

## What the literature says about novelty

The max-over-regimes idea is **not itself novel**. Yang & Jin's ICML 2026 *Multi-Distribution Robust Conformal Prediction* proposes max-p aggregation with finite-sample multi-distribution coverage, plus efficiency optimization: https://proceedings.mlr.press/v306/yang26ad.html. Our scalar max-radius rule is a simple baseline; implement MDCP directly before arguing improvement.

Braun et al.'s ICML 2026 *Conditional Coverage Diagnostics for Conformal Prediction* introduces excess risk of target coverage (ERT), a classifier-based conditional-coverage diagnostic: https://proceedings.mlr.press/v306/braun26c.html. Add ERT or a similarly powered diagnostic; six-regime minimum coverage alone is not sufficient.

On ACDB's protocol-specific six-level ladder, PC with greedy active orientation reports directed F1 42.7% and SHD 4.79; Claude Sonnet 4.6 raw-active reports 31.7% and 7.25; GPT-5.4 raw-active reports 22.9% and 9.27: https://arxiv.org/abs/2609.31675. These metrics are not comparable to this toy game's cross-entropy regret. CausalArena finds rankings shift across SCM families and protocols, reinforcing the need for held-out heterogeneous worlds: https://arxiv.org/abs/2609.11897.

## Next benchmark gates

1. Implement multi-distribution robust conformal (including max-p) and ERT as direct baselines.
2. Use at least 1,000 held-out cases per regime across multiple seeds and held-out SCM/mechanism families; report coverage confidence intervals and paired regret intervals.
3. Choose the smallest ambiguity set that meets predeclared overall and worst-regime coverage floors, with an explicit regret budget; trigger safe fallback/abstention for unsupported regimes or failed shift checks.
4. Compare the full risk–coverage–cost frontier: mean/p95 regret, set-size proxy, intervention efficiency, latency, memory, calibration sample count, and fallback rate.
5. Run ACDB/CausalArena/TCD-Arena workloads under matched protocols. No SOTA claim until those results exist.
