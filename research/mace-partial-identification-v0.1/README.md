# MACE-PI-001 — finite-cohort partial identification for causal experience transfer

**Status:** standalone synthetic research reference; not production, not statistical SOTA, not an empirical causal finding.

Parent: [#9516 — causal experience transfer across interdependent agent sessions](https://github.com/Luminous-Dynamics/symthaea/issues/9516).

## Why this increment exists

The current MACE v0.1 reference admits structured observations only when their intervention-design and outcome-reconciliation receipts match. A matching digest proves payload identity, not that random assignment really happened, that the issuer is authentic, or that a point causal effect is identified.

A more defensible memory object is an **identified set of causal effects**, not an automatically trusted point estimate. The engine must preserve an observational association separately, expose the assumptions that narrow the effect set, and authorize a decision only when all compatible effects imply the same action.

## Executed reference

A dependency-free Go 1.23.2 reference enumerates binary potential-outcome completions for a finite cohort. It computes exact rational bounds for the average treatment effect under explicitly declared finite-cohort consistency, no-interference and well-defined-treatment assumptions. It requires an explicit monotone-treatment-response (MTR) violation budget; zero violations imposes individual MTR, while a budget equal to cohort size imposes no MTR restriction.

The robust decision rule compares the entire ATE interval [L,U] to a decision threshold τ:
- intervene only if L > τ;
- maintain if U ≤ τ;
- otherwise return NOT_IDENTIFIED.

The implementation uses an O(n²) dynamic programme over exact MTR violation counts, exact rational arithmetic, strict JSON input, exact regime filtering, and an independently written exhaustive counterfactual oracle for generated small cohorts.

## Canonical synthetic result

Four units in one regime produce:
- observed treated-minus-control success-rate difference: **+2/3**;
- sharp finite-cohort ATE bounds without MTR: **[−1/4, +3/4]**;
- bounds with zero MTR violations: **[0, +3/4]**;
- allowing one MTR violation restores the lower bound to **−1/4**.

At a declared intervention threshold of +1/10, both the unrestricted and MTR-only intervals cross the threshold. The correct outcome is **NOT_IDENTIFIED**. A plug-in policy that mistakes the +2/3 observed association for a causal effect would recommend intervention.

This is a four-unit deterministic illustration, not a measured population effect, not a confidence interval, and not an external benchmark result.

## Local verification

- 13/13 top-level test functions passed.
- 256 deterministic generated small cohorts matched an independent exhaustive counterfactual-completion oracle for every MTR budget.
- race detector, vet and formatting passed.
- statement coverage: 77.4%.
- standard library only.

The runnable reference and complete test suite are included in the downloadable research archive supplied with this work. This PR adds the protocol handoff, not the source archive itself.

## Research comparison

Do not claim novelty for evidence-linked skills or causal graphs:
- **MSCE (Findings EMNLP 2026)** already combines grounded traces, evidence-backed policies, callable skills, applicability boundaries, verification rules and reflection-weighted value backfilling; the paper reports statistically supported aggregate gains on EvoAgentBench and LoCoMo.
- **SkillLens / From Raw Experience to Skill Consumption (2026)** evaluates experience generation, skill extraction and skill consumption, finding that model-generated skills can help on average yet still produce negative transfer.
- **EvoAgentBench (Findings EMNLP 2026)** measures ability transfer across web research, algorithmic reasoning, software engineering and knowledge work; its paper reports curated ability content transfers positively across tested configurations while automatic methods are inconsistent.
- **AMA-Bench (ICML 2026)** is a strong long-horizon trajectory-memory baseline; AMA-Agent combines causality graphs with tool-augmented retrieval.
- **MemoryArena (ICML 2026)** evaluates interdependent multi-session Memory-Agent-Environment loops.

This increment's narrower contribution is an exact finite-cohort partial-identification oracle for the causal-transfer boundary. It does not reproduce the 2026 information-theoretic causal bounds estimator or the NeurIPS 2025 data-fusion partial-identification method.

## Next empirical qualification

Compare no-memory, full-history, BM25/dense/hybrid, similarity-only skills, MSCE, SkillLens/meta-skill, MACE typed admission, and MACE + partial-identification gate under matched models, splits, tool/action budgets and evaluators. Report negative-transfer rate, paired transfer gain, worst-cell regret, risk/coverage, causal-bound width/coverage, decision sign-stability under assumption relaxation, and complete lifecycle cost.

Proposed gates—not achieved results:
- 100% exact agreement on deterministic finite-cohort fixtures.
- Zero robust interventions when the identified set crosses the threshold.
- Zero implicit cross-regime transfer.
- Positive paired transfer improvement over the strongest matched automatic baseline, with domain/scaffold negative cells reported rather than averaged away.
- Statistical confidence intervals, identification bounds and predictive intervals must be reported as three separate objects.

## Boundaries

The reference assumes binary outcomes/treatment, a fully observed finite cohort, consistency, no interference and a well-defined treatment. Those assumptions are declarations, not authenticated truths. It has no sampling uncertainty and does not generalize beyond the enumerated cohort. A dataset digest establishes content identity only. No external benchmark, LLM, randomized trial, real-world intervention, Rust production integration or SOTA comparison was run.

Archive SHA-256: `435ce4a536d3df0d0bc1ebab11b5da6f999aa43de0203fe534868d9b3f301250`.
