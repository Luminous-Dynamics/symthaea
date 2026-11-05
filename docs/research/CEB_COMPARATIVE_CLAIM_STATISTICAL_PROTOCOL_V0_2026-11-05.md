# CEB Comparative Claim Qualification and Statistical Protocol V0 — 2026-11-05

**Status:** Design-only; zero empirical claims registered.  
**Registry:** [`benchmarks/ceb/comparative-claim-registry.v0.json`](../../benchmarks/ceb/comparative-claim-registry.v0.json)  
**Registry schema:** [`benchmarks/ceb/comparative-claim-registry.schema.json`](../../benchmarks/ceb/comparative-claim-registry.schema.json)  
**Individual claim schema:** [`benchmarks/ceb/comparative-claim.schema.json`](../../benchmarks/ceb/comparative-claim.schema.json)  
**Validator:** [`scripts/validate_ceb_comparative_claims.py`](../../scripts/validate_ceb_comparative_claims.py)

## 1. Claim classes

CEB distinguishes descriptive, exploratory, matched-comparison, non-inferiority, superiority and Pareto-dominance claims. Classification depends on experimental design and evidence, not headline score.

A contextual leaderboard result is not a direct comparison. Exact metric semantics make an external observation eligible for a future matched test; they do not prove task, model, harness, budget, environment or adjudication equivalence.

## 2. Pairing and sparse discordant outcomes

Binary task outcomes must preserve task pairing. The same task is evaluated by treatment and control, and the analysis reports discordant counts. McNemar's test is designed for paired binary observations; with sparse discordant pairs, exact McNemar/binomial inference is preferred to relying only on a large-sample approximation. Source: https://www.ncbi.nlm.nih.gov/books/NBK560699/

Seeds are repeated measurements nested within tasks. They do not multiply the task denominator or repository cluster count.

## 3. Repository/root-cause clustering

The 48-task Rust pilot has six repository placeholders. It is an exploratory feasibility/variance-estimation pilot, not a confirmatory SOTA study. Tasks from one repository or root-cause family are not automatically independent.

The 2023 *Cluster-robust inference: A guide to empirical practice* discusses how few or heterogeneous clusters can make conventional asymptotic inference unreliable and recommends small-sample-aware methods, including wild-cluster restricted bootstrap. Source: https://doi.org/10.1016/j.jeconom.2022.04.001

CEB V0 uses a conservative policy: **12 or fewer repository clusters remain exploratory**. A confirmatory claim requires at least 13 repository clusters, an explicit power analysis, and cluster-aware inference. Thirteen is a policy floor, not a guarantee of adequate power; the design still needs to justify the actual cluster count and sample size.

## 4. Preregistration, power and multiplicity

A confirmatory experiment must freeze one primary endpoint and one primary contrast before holdout access. It must set a minimum practically important effect, estimate sample size from pilot paired discordance and repository/root-cause variation, target at least 90% power for the preregistered effect unless justified otherwise, and preserve task, prompt, grader, environment, budget and metric digests.

Secondary metrics in the confirmatory family receive Holm step-down adjustment. Holm controls family-wise error under general dependence. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC2583789/

## 5. Non-inferiority and hard gates

The proposed -2 percentage-point functional non-inferiority margin is not approved by default. It must be justified and frozen before holdout access.

Critical security, evaluator-integrity, authorization and isolation failures are non-compensable. A qualified claim requires qualified isolation/trust, independent regrade qualification, reference-patch PASS, empty-patch FAIL and zero critical violations.

Zero observed failures do not prove zero risk. Under an independent Bernoulli model, zero failures in 300 trials yields an approximate one-sided 95% upper bound of 0.994%. The independence assumption must be justified; hundreds of mutations from one fault family are not hundreds of independent controls. Exact-binomial reference: https://www.govinfo.gov/content/pkg/GOVPUB-C13-477b14161dac9d9ab920bf2cf1d6d4ac5/pdf/GOVPUB-C13-477b14161dac9d9ab920bf2cf1d6d4ac5.pdf

## 6. Pareto reporting

Publish a metric vector across regression-safe correctness, security, cost per accepted repair, latency/resource consumption, human review burden, abstention/escalation and evidence completeness. Pareto dominance requires no worse performance on each preregistered dimension and a strict improvement on at least one, with uncertainty and hard gates visible. Do not replace the vector with an arbitrary weighted aggregate.

## 7. Current state

The V0 registry contains **zero empirical claims**. The Rust and human-escalation pilots remain design-only. No power result, matched superiority, non-inferiority or Pareto claim is asserted. #10064 and #10369 remain the execution and trust gates.
