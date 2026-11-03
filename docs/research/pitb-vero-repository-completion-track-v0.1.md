# PITB Vero repository-completion evaluation track v0.1

**Priority:** P1 external comparator; promote to P0 after artifact/license and reproduction feasibility are confirmed.  
**Status:** evaluation design; no Vero tasks executed.  
**Tracking:** [PITB Aegis integration #10272](https://github.com/Luminous-Dynamics/symthaea/issues/10272)

## 1. Why add Vero

Vero asks a different and more system-level question than Aegis: can an agent build and maintain a complete formally verified repository, not merely close individual theorem obligations?

The September 2026 Berkeley RDI report describes 43 multi-module Lean 4 instances curated from real-world projects originally written in Python, Dafny, Verus, and Coq. It reports 743 scored APIs and 2,705 formal specifications. The strongest reported GPT-5.5 (xhigh) + Codex configuration fully solved 27/43 instances in code-and-proof mode and 25/43 in proof-only mode, while passing 87.3% and 85.8% of individual specifications respectively.

The gap between per-specification success and full-instance completion is the key signal: a repository can have a high local pass rate and still fail as a coherent buildable, axiom-clean deliverable.

Primary source: https://rdi.berkeley.edu/blog/vero/

## 2. Distinct evaluation modes

Preserve Vero's two modes:

- **Proof-only:** reference implementations are supplied; the agent proves their specifications.
- **Code-and-proof:** the agent implements the required APIs and proves the supplied specifications.

Do not compare code-and-proof completion against proof-only completion as if they were the same task. Report both separately.

## 3. PITB research questions

1. Does proof insight transfer increase complete-instance success, not merely per-specification pass rate?
2. Do stable helper lemmas reduce duplicated proof work and improve repository completion?
3. Can dependency-aware scheduling identify global invariants and shared helpers early?
4. Can the controller prevent local proof success from masking repository build failures, axiom leakage, missing APIs, or stale proofs?
5. Can PITB preserve exact repository state and produce a replayable whole-repository receipt?

## 4. Required outcome hierarchy

Report all levels independently:

1. API implementation completeness.
2. Per-specification proof pass rate.
3. Build success for the exact graded repository.
4. Axiom-clean status under the benchmark's declared policy.
5. Full-instance completion (all required APIs and specifications pass, repository builds, and required axiom policy passes).
6. Independent reproduction from a clean checkout.

Full-instance completion is the primary headline metric. Per-specification pass rate is diagnostic and must not replace it.

## 5. PITB experiments

Run paired, matched-budget ablations:

- baseline agent/controller;
- typed proof retrieval disabled/enabled;
- stable helper lemma synthesis disabled/enabled;
- dependency-closure scheduling disabled/enabled;
- global-invariant discovery disabled/enabled;
- implementation planning disabled/enabled (code-and-proof mode only);
- full PITB controller.

Track helper theorem reuse, number of dependent specifications per helper, proof-line concentration in shared lemmas, duplicate proof effort, failed downstream obligations caused by helper changes, rebuild frequency, and rollback precision.

Use a clean checkout for final reproduction. Every accepted result binds repository commit, toolchain, manifests, all API/spec IDs, generated source/proof digests, build log digest, axiom audit, and replay receipt.

## 6. Metrics

- Full instances solved / total instances.
- APIs implemented / required APIs.
- Specifications proved / required specifications.
- Buildable instances / total instances.
- Axiom-clean instances / total instances.
- Independent clean-checkout reproduction rate.
- Helper reuse distribution and proof-graph centrality.
- Cost per full instance and cost per verified specification.
- p50/p95 completion time and repair iterations.
- Regression count after helper edits.
- Partial completion distribution, not just binary success.

Cluster confidence intervals by repository instance; APIs and specifications within one instance are not independent observations.

## 7. Strategic hypothesis

PITB should optimize for **coherent verified repository completion** rather than isolated theorem closure.

The most promising architectural differentiator is a persistent proof-and-implementation dependency graph that tracks:
- API obligations;
- specification dependencies;
- helper lemma dependencies;
- build/module dependencies;
- global invariants;
- exact source/proof version relationships.

A change to a shared helper should invalidate exactly the dependent proofs and build artifacts, not the entire repository and not an arbitrary subset. This invalidation precision should itself be benchmarked.

## 8. Relationship to the portfolio

| Benchmark | Primary capability |
|---|---|
| Aegis | Real Rust implementation-derived obligations |
| SorryDB | Proof retrieval and transfer integrity |
| Vero | Whole-repository code-and-proof completion |
| VeriSoftBench | Repository-context proof discovery |
| CoCo-Prover comparison | Cost-aware proof-graph scheduling |
| SIB | Adversarial semantic and evidence integrity |

Keep separate cohorts and scores. Vero is P1 until official artifacts, licensing, exact harness and baseline reproducibility are established; its whole-instance completion metric is strategically important enough to evaluate early.
