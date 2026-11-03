# PITB industry benchmark landscape — 2026-11-03

**Purpose:** keep external comparators and their measurement boundaries explicit.  
**Status:** desk research; no external benchmark was run by PITB.

## 1. Current frontier

| System / benchmark | Published scope and result | What it tests | PITB response |
|---|---|---|---|
| EasyProver / Aegis (arXiv:2610.38174) | 1,054/2,000 Aegis statements proved (52.7%); 1,997 attempted, 943 unresolved. Also 669/672 PutnamBench, 597/600 NTP4VC resolved, 189/189 VERINA resolved. | Recursive agent search, isolated contexts, parallel tactic probes, decoupled Lean elaboration; implementation-derived Rust-to-Lean obligations. | Matched Aegis evaluation, explicit-contract improvement, independent semantic evidence, cost/reliability profile. |
| CoCo-Prover (arXiv:2610.09681) | Reports best solve rate on five Lean verification benchmarks and up to 30.9% lower cost than strongest baseline in its evaluation. | Two-level proof graphs, goal selection, priced specialist-agent routing. | Matched success/cost frontier and ablation of PITB scheduling/typed retrieval. |
| VeriSoftBench (arXiv:2602.18307) | 500 tasks from 23 repositories; project-specific dependency context; project page reports 41% best model with curated context and 35% with raw repository context. | Repository context, multi-hop dependencies, software proof retrieval. | Dependency-closure discovery, context precision, cross-file reuse, clean reproduction. |
| Vero (Berkeley RDI, Sep 2026) | 43 multi-module Lean instances, 743 APIs, 2,705 specs; strongest reported GPT-5.5 (xhigh)+Codex completed 27/43 code-and-proof and 25/43 proof-only instances, while per-spec pass rates were 87.3% and 85.8%. | Whole-repository implementation plus proof completion, build and axiom cleanliness. | Make full-instance completion and clean-checkout replay primary, not average theorem pass rate. |
| WybeCoder (arXiv:2603.29088) | Reports 74% on Verina and 62% on Clever in its evaluation. | Prove-as-you-generate imperative code, invariant synthesis and SMT/Lean interaction. | Compare invariant discovery and code/proof co-evolution under exact shared tasks/settings; don't compare headline percentages across different benchmark versions. |
| Aeneas | Rust-to-functional translation toolchain with formalized translation semantics for a supported Rust subset. | Translation and memory/ownership abstraction. | Pin translator, audit coverage, distinguish formal tool guarantees from empirical differential tests. |

## 2. Comparability rules

These figures are not a common leaderboard. They differ in:
- task definitions and benchmark versions;
- proof-only versus code-and-proof settings;
- per-theorem versus whole-instance success;
- proof versus proof-or-refutation counting;
- number of attempts, pass@k, wall-clock budgets, and model/tool configurations;
- available context, hardware, and API/compute costs.

Never compare percentages without aligning these factors. Report published values as context only, then run PITB's own matched experiment.

## 3. Aegis details that materially affect reproduction

The paper's 2,000-task sample includes 1,997 attempted statements, 1,054 proofs, and 943 unresolved; 52.7% is calculated over the full 2,000 sample. The sample spans 1,257 translated programs and excludes statements already refuted by Plausible and the category requested to fail. The paper reports 84.1% edge-case proof rate and 31.0% explicit-contract proof rate.

The paper describes 2,356 attempted source problems, of which 2,045 (86.8%) passed Rust testing, extraction, translation, Lean elaboration and differential evaluation. The 2,045 program-pair count, 11,533 candidate specifications and 2,000-task sample are different quantities.

The Rust/Lean comparison is differential testing on retained reference inputs, not a formal all-input semantic-equivalence theorem. The paper also notes mutation-generated variants as a possible extension but does not evaluate those variants.

The paper body does not link a direct official Aegis dataset artifact or state its release license. PITB must obtain an official author release or access confirmation before acquisition/reproduction claims.

## 4. Strategic conclusion

PITB should compete on a vector of capabilities, not a single aggregate:
1. implementation-derived theorem resolution (Aegis);
2. proof retrieval/transfer integrity (SorryDB);
3. repository-context proof discovery (VeriSoftBench);
4. whole-repository code-and-proof completion (Vero);
5. cost-aware proof scheduling (CoCo-Prover comparison);
6. adversarial semantic/evidence integrity (SIB);
7. safe proof maintenance after changes (CIPM).

The differentiator to pursue is **lifecycle assurance**: construct a proof, establish what it means about the source, maintain its dependency graph through changes, and independently reproduce the complete repository claim.

## 5. Sources

- EasyProver / Aegis: https://arxiv.org/abs/2610.38174
- CoCo-Prover: https://arxiv.org/abs/2610.09681
- VeriSoftBench: https://arxiv.org/abs/2602.18307 and https://utopia-group.github.io/VeriSoftBench/
- Vero: https://rdi.berkeley.edu/blog/vero/
- WybeCoder: https://arxiv.org/abs/2603.29088
- Aeneas: https://aeneasverif.github.io/ and https://doi.org/10.1145/3547647
