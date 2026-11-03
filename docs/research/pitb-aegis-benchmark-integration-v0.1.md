# PITB Aegis integration protocol v0.1

**Status:** research protocol; dataset acquisition, adapter implementation, and benchmark execution are pending.  
**Priority:** P0 external comparator, alongside SorryDB.  
**Tracking:** [#10272](https://github.com/Luminous-Dynamics/symthaea/issues/10272)  
**Parent research:** [PITB #8154](https://github.com/Luminous-Dynamics/symthaea/pull/8154)

## 1. Purpose

Extend the Proof Insight Transfer Benchmark (PITB) from mathematical proof transfer into verified reasoning about real implementations.

Aegis is a distinct benchmark cohort derived from Rust programs translated into Lean. Its obligations exercise fixed-width arithmetic, overflow, explicit failure, loops, casts, and functionalized mutation. PITB must establish both that a proposition is proved and that the proposition faithfully describes the frozen implementation.

Aegis does not replace SorryDB. Keep benchmark identities, source snapshots, denominators, budgets, scorecards, and claims separate.

Primary paper: Ganguly et al., [Scaling Lean Theorem Proving for Real-World Code Verification](https://arxiv.org/abs/2610.38174), arXiv:2610.38174 (2026-10-28).

## 2. Published reference point

The paper reports:

- 2,045 differentially validated Rust–Lean program pairs.
- 11,533 candidate specifications.
- 1,054/2,000 sampled statements proved by EasyProver (52.7%) within 30 minutes per statement.
- A substantial difficulty split between edge-case and explicit-contract specifications (reported as 84.1% and 31.0%, respectively).

These are published comparator values, not PITB results. Reproduction must recover the paper's exact cohort selection, category definitions, exclusions, denominators, model/harness configuration, and budget semantics before treating the values as directly comparable.


## 2A. Important evaluation details from the full paper

The 2,000-statement sample is not equivalent to 2,000 attempted statements. The paper reports 1,997 attempts, 1,054 proofs, and 943 unresolved tasks. The 52.7% headline uses the full 2,000-statement sample as denominator. Preserve all three counts in any reproduction.

The sample is stratified by LeetCode difficulty and specification category over 1,257 translated programs. It excludes statements already refuted by the Plausible screen and the category requested to fail. Consequently:
- the published 52.7% is a proof rate on this selected sample, not an all-task resolution rate;
- it is not a direct measurement of the system's ability to prove or refute arbitrary Aegis statements;
- the 84.1% edge-case and 31.0% explicit-contract figures must be reported with the original stratum denominators and selection procedure;
- reproduction must preserve pre-screen exclusions and requested-polarity metadata.

The paper reports that each frozen Lean statement is about the extracted Lean program. Differential testing separately compares Rust and Lean outputs on retained reference cases. This is useful translation-validation evidence, but it is not a formal proof of semantic equivalence for all inputs. PITB must preserve this distinction in claims and receipts.

The full paper also reports 2,356 attempted source problems, of which 2,045 (86.8%) passed Rust testing, extraction, translation, Lean elaboration, and differential evaluation. The 2,045 program pairs and 11,533 candidate specifications are corpus-generation outputs, not the evaluation sample size.

The paper mentions optional mutated variants as a way to generate further tasks, but explicitly does not evaluate those mutation tasks. This creates a strong opportunity for SIB: develop a separate, independently labeled mutation evaluation while preserving the official Aegis cohort unchanged.

At the time of this protocol revision, the paper body does not identify a direct downloadable official Aegis dataset artifact or release/license URL. Do not invent one. Locate an official release from the authors or request access and record provenance before claiming dataset acquisition or running a reproduction.

## 3. Claim model

A successful implementation-verification result requires independent evidence for all of:

1. **Source identity:** exact Rust source bytes, repository revision, path, and relevant build inputs are frozen.
2. **Translation identity:** exact translator version/configuration and generated Lean definitions are bound to the source.
3. **Specification identity:** exact theorem statement bytes and digest are frozen; semantic review addresses correspondence to the intended implementation behavior.
4. **Proof validity:** the proof elaborates and is accepted by the pinned Lean kernel with the expected axiom closure.
5. **Independent replay:** a separate verifier reopens the frozen artifacts and reproduces the acceptance decision.
6. **Adversarial integrity:** negative controls for mismatched source, translation, specification, and evidence are rejected.

A kernel-checked proof of a vacuous, weakened, stale, or implementation-mismatched statement is not a successful implementation verification.

## 4. Outcome taxonomy

Each task attempt must end in exactly one terminal classification:

- `proved`: proof accepted and all required integrity/correspondence gates passed.
- `refuted`: a valid counterexample/refutation artifact is accepted under the task's semantics.
- `unresolved`: budget exhausted or no decisive result.
- `invalid_task`: frozen input or benchmark metadata fails integrity checks.
- `translation_disagreement`: source/translated semantics fail the required correspondence checks.
- `specification_disagreement`: specification is ambiguous, vacuous, or does not match the declared contract.
- `execution_error`: toolchain or infrastructure failure; never silently count as a mathematical non-solve or a proof.

Do not convert missing evidence, timeouts, skipped workflows, or infrastructure failures into success or silently drop them from denominators.

## 5. Frozen task identity

The adapter should bind at least:

- benchmark name, release/version, cohort ID, task ID;
- source repository, revision, path, source digest;
- Rust compiler/toolchain and relevant build configuration;
- translator name/version/configuration and translated Lean source digest;
- Lean toolchain, Lake manifest, imports, and dependency closure;
- program-pair ID and specification ID (distinct namespaces);
- exact statement bytes/digest and published stratum/category;
- label provenance and split membership.

The full corpus counts (program pairs and candidate specifications) must not be conflated with the sampled evaluation-task count.

Dataset acquisition is a hard prerequisite. Record the official artifact URL, license, release metadata, archive digest, extraction manifest, and any exclusions. Do not substitute a regenerated corpus or infer missing labels.

## 6. Required stratification

Report separate denominators and outcomes for:

- edge-case specifications;
- explicit-contract specifications;
- fixed-width arithmetic and overflow;
- explicit failure behavior;
- loops and loop invariants;
- casts and signedness/width conversions;
- mutation and state transitions;
- mixed obligations.

Also report per-program results, not only per-specification results. Multiple specifications derived from one program are clustered observations; do not treat them as independent samples without a justified statistical model.

## 7. Matched experiments

Use the exact same frozen tasks, model, timeout, tool budget, and environment for:

1. fixed tactic/search baseline;
2. retrieval/typed proof transfer disabled vs enabled;
3. dependency-aware decomposition disabled vs enabled;
4. invariant synthesis disabled vs enabled;
5. counterexample-guided repair disabled vs enabled;
6. adaptive action allocation disabled vs enabled;
7. full PITB controller.

Use paired task-level comparisons. Preregister primary outcomes and ablations. Report confidence intervals, task/program clustering, missingness, timeout rates, p50/p95 latency, model/API spend, CPU/GPU use, and independent verifier cost.

The paper's 30-minute-per-statement limit is a reference point. Equal wall-clock limits do not imply equal compute, parallelism, hardware, or API spend.

## 8. Adversarial controls

The integrity suite should include mutations for:

- signedness and integer width;
- overflow boundaries and cast truncation;
- loop bounds and termination conditions;
- explicit error/failure paths;
- mutation order and state update semantics;
- missing or weakened preconditions;
- strengthened or vacuous postconditions;
- stale source or translator versions;
- swapped program/specification identifiers;
- altered Lean statement bytes;
- missing, duplicate, stale, or forged receipts.

Every seeded mismatch must be rejected by the independent acceptance path. Keep these controls outside the scored benchmark denominator unless the official benchmark explicitly defines them as tasks.

## 9. Prospective targets

Targets below are hypotheses, not achieved results:

| Measure | Initial target | Stretch |
|---|---:|---:|
| Exact published Aegis baseline reproduction | 52.7% reconciled | — |
| Overall verified resolution at matched budget | 60% | 65% |
| Explicit-contract resolution | 45% | 55% |
| Edge-case resolution | At least 84.1% | Improve |
| Cost reduction vs matched baseline | 25% | 35% |
| Independent replay of accepted proofs | 100% | 100% |
| Seeded integrity-control rejection | 100% | 100% |
| Unaccounted tasks | 0 | 0 |

Any target change must be dated and justified. Never overwrite historical results or present prospective values as measured performance.

## 10. Related benchmark portfolio

Maintain independent scorecards:

| Cohort | Research dimension |
|---|---|
| Aegis | Implementation-derived Rust-to-Lean verification |
| SorryDB | Integrity-qualified proof retrieval and transfer |
| VeriSoftBench | Repository context and cross-module dependencies |
| CoCo-Prover comparison | Cost-aware proof-graph scheduling |
| PutnamBench / NTP4VC / VERINA | Mathematical and verification regression controls |

VeriSoftBench: https://utopia-group.github.io/VeriSoftBench/ and https://arxiv.org/abs/2602.18307  
CoCo-Prover: https://arxiv.org/abs/2610.09681

Do not combine these cohorts into one weighted leaderboard score. Report a performance vector with independent denominators, snapshots, and budgets.

## 11. Paradigm-shift hypotheses

### Proof-carrying implementation contracts

Create an independently checkable chain from source bytes to translated semantics, specification, theorem, proof artifact, and replay receipt.

### Counterexample-guided specification diagnosis

Use counterexamples and differential execution to localize errors among implementation, translation, specification, and invariant. Never silently rewrite the frozen target.

### Invariant discovery as a first-class search problem

Search for loop, arithmetic-range, mutation, aliasing, and failure-path invariants. Each candidate carries provenance and explicit proof obligations.

### Typed and counterfactual proof insight

Transfer proof strategies only when their preconditions and dependency closures match. Test near-miss contexts to distinguish essential assumptions from incidental proof-script dependencies.

### Cost-aware verified progress

Allocate bounded resources across retrieval, decomposition, invariant synthesis, tactics, refutation, repair, and restart. Optimize independently accepted results per unit cost, not raw attempt count.

### Independent semantic triangulation

Combine Lean kernel checking, source-level differential/property testing, and independently implemented translation/specification checks. These evidence channels are complementary, not majority votes.

## 12. Implementation sequence

1. Acquire and pin the official Aegis artifact; document licensing and reproducibility.
2. Implement dataset provenance manifest and independent integrity verifier.
3. Implement an Aegis adapter to PITB's existing task/receipt model without weakening SorryDB gates.
4. Add cohort-stratified scorecard and paired experiment runner.
5. Add synthetic semantic-mismatch and evidence-tampering controls.
6. Reproduce the published baseline before evaluating PITB improvements.
7. Run ablations and publish complete evidence, including failures and unresolved tasks.

## 13. Current evidence boundary

This document is a research protocol only. It does not claim that the Aegis dataset has been downloaded, that an adapter exists, that PITB has run any Aegis task, that Rust or Lean builds pass, or that PITB has reproduced or exceeded EasyProver. Those claims require exact-head artifacts and independently reproducible execution evidence.
