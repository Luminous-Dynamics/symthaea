# Mathematical proof comprehension — competitive landscape and benchmark ledger v1

**Parent:** [#13293 MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Date:** 2026-11-22  
**Status:** literature baseline, not Symthaea benchmark results.

## Executive positioning

Symthaea should not compete on a single blended theorem-proving score. The field's current benchmarks measure different capabilities:

- formal completion on a supplied statement;
- discovery of the theorem before proving it;
- repository-scale proof obligations and dependency retrieval;
- proof compression / restructuring;
- formalization of implementations;
- theorem-family transfer;
- human comprehension and explanation fidelity.

The last two, when linked to a formal evidence graph and a blind human study, are the proposed differentiation. We do not claim no one has studied explanation; we claim this exact joined evaluation—formal identity, proof-mechanism reconstruction, explanation support, counterfactual understanding and human transfer—needs to be benchmarked as a coherent capability.

## 1. Benchmark ledger

| Benchmark / work | Public reference | Published scale / headline result | Capability measured | Strategic implication |
|---|---|---|---|---|
| FormalMATH (EMNLP Findings 2026) | https://aclanthology.org/2026.findings-emnlp.579/ | 5,560 semantically validated Lean 4 statements; DeepSeek-Prover-V2 671B: 28.31% Pass@32 | Formal proof generation from formal statements | Use as a proving baseline, but measure theorem-family transfer, claim fidelity and explanation quality separately |
| EasyProver / Aegis (Oct 28, 2026 preprint) | https://arxiv.org/abs/2610.38174 | 669/672 PutnamBench; all six IMO 2026 formalizations; 597/600 NTP4VC; 189/189 VERINA; Aegis has 2,045 Rust–Lean program pairs and 11,533 candidate specs; 1,054/2,000 sampled specs proved within 30 minutes | Multi-agent Lean proving and implementation-derived verification | Contest theorem accuracy alone is saturated; add implementation semantics, mutations and hidden specifications |
| VeriSoftBench | https://utopia-group.github.io/VeriSoftBench/ | 500 tasks from 23 Lean repositories; 41% best with curated context, 35% full raw-repository context | Repository-context software verification | Dependency selection/context construction is a measurable bottleneck and useful baseline |
| LeanLean (2026) | https://www.sri.inf.ethz.ch/publications/rognvaldsson2026leanlean | 64 repositories up to 174k LOC; Opus 5 reports 48.3% average proof-size compression within 12 hours | Repository-scale proof compression and restructuring | Add an explanation-compression frontier, but do not equate shorter code with better mathematical understanding |
| TheoremBench (Jun 8, 2026) | https://arxiv.org/abs/2606.09450 | Nearly 100 classical theorems; plain and premised versions with subtheorem families | Main theorem and supporting-subtheorem coverage, token efficiency | Compare proof dependency coverage, not just final theorem pass/fail |
| Formal Conjectures (May 13, 2026 paper snapshot) | https://arxiv.org/abs/2605.13171 | 2,615 statements in the paper snapshot; 1,029 open research conjectures and 836 solved tasks for autoformalization | Research conjectures and formalization | Use immutable time-stamped family holdouts and contamination controls |
| Discover-and-Prove (ACL 2026) | https://aclanthology.org/2026.acl-long.3/ | Hard-Mode MiniF2F/FIMO; reports 36 PutnamBench theorems formally proved in Hard Mode | Discovery before proof generation | Use as discovery baseline; add conceptual and cross-family transfer |
| Aegis implementation lane | https://arxiv.org/abs/2610.38174 | Real Rust-to-Lean program pairs and candidate specifications | Correctness, safety and semantics of code | Useful transfer target for formal explanation and human-readable specification reasoning |

Scores and budgets are not directly comparable across rows. Cite the paper's own protocol. Do not create an aggregate rank across these tasks.

## 2. Where Symthaea should compete

### Track A — proof identity and formal trust

Baselines: Lean kernel, theorem prover output, #11362's claim-to-Lean fidelity checks.

Targets:
- exact theorem/source/toolchain identity on every task;
- zero false-positive theorem acceptance on changed/weakening/namespace-shadowing controls;
- explicit axioms, admissions, unknown and timeout;
- no HDC similarity or model confidence as proof evidence.

This is a hard gate, not a score that can be averaged.

### Track B — repository dependency comprehension

Baselines:
- raw repository context;
- curated context (VeriSoftBench-style);
- lexical/BM25 retrieval;
- dense retrieval;
- Lean dependency graph;
- symbolic/type-directed retrieval;
- hybrid graph + lexical + dense;
- hybrid plus HDC.

Targets:
- ≥90% precision / ≥85% recall on expert-labelled material dependencies;
- distinguish type dependency, proof dependency, imported infrastructure, inductive/recursor edges and mathematical role;
- do not claim every syntactic dependency is mathematically essential.

### Track C — proof mechanism reconstruction

For each proof slice, require:
- exact theorem claim and all assumptions;
- each key lemma's role and the direction in which it is used;
- a derivation skeleton in mathematical prose;
- one worked example / counterexample if available;
- a list of quoted or unresolved obligations;
- per-sentence evidence links for formal facts.

Target: ≥80% independent expert acceptance on whole held-out theorem families, with reviewer disagreement retained.

### Track D — counterfactual understanding

Mutate one binder, assumption, endpoint, definition or reduction guarantee.

Require:
- predict whether the changed statement is still true;
- provide a concrete counterexample when false;
- distinguish "proof no longer typechecks" from "the mathematical proposition is false";
- detect theorem substitutions and weakened claims.

Targets:
- ≥90% correct disposition on held-out controlled mutations;
- zero acceptance of altered statement as original;
- abstention is preferred over fabrication.

### Track E — human comprehension and transfer

Compare raw Lean/source, generic LLM summary, dependency graph only, and source-linked concept explanation.

Primary outcomes:
- exact restatement of theorem and assumptions;
- explanation of selected construction / lemma role;
- near-transfer problem accuracy;
- confidence calibration.

Secondary:
- time-to-correct-understanding;
- ability to detect a deliberately planted missing assumption;
- ability to write a formalizable paraphrase;
- explanation length / token and reviewer cost.

Targets:
- ≥20 percentage-point improvement in comprehension or near-transfer over raw-source and source-only baselines;
- ≥10 percentage-point improvement on unseen theorem families;
- no increase in false-fact rate.

### Track F — explanation compression (Minimum Sufficient Explanation)

Treat explanation length as a constrained optimization objective, not as a standalone score.

Find the Pareto frontier of:
- token count;
- dependency coverage;
- formal-fact support coverage;
- expert faithfulness;
- human comprehension;
- near-transfer.

Target: ≥25% fewer explanation tokens at matched comprehension within a preregistered 5 percentage-point non-inferiority margin. The frontier must include full-source and dependency-only baselines. A shorter explanation that drops assumptions or hurts transfer is a failure.

### Track G — mathematical mechanism transfer

Hold out entire theorem families. Train/tune on separate families. Ask whether a mechanism learned in one proof—recursion, quotient induction, local-to-global composition, interpolation, valuation arguments, symmetry reduction, witness reconstruction—helps solve or explain an unseen instance in another family.

Target: at least +10 percentage points over a source-only baseline without increased false-fact rate. Require a task-level explanation of the transferred mechanism.

## 3. HDC / CfC competitive ablations

Use identical corpus, model, prompt, context budget and compute:
1. BM25;
2. dense embeddings;
3. dependency graph only;
4. symbolic / type-directed retrieval;
5. lexical + dense + graph hybrid;
6. hybrid + HDC;
7. hybrid + HDC + CfC.

Report recall@k, MRR, material-edge precision/recall, statement fidelity, explanation support, transfer, wall-clock time, memory and tokens. Do not claim HDC/CfC advantage from one benchmark or a similarity score. If the improvement fails to reproduce on held-out theorem families, remove the component from this capability.

## 4. Industry comparison strategy

We do not have to beat EasyProver at PutnamBench to make a meaningful contribution. We should:
- reproduce published proof-generation baselines under compatible budgets;
- compare repository comprehension against VeriSoftBench-style context conditions;
- compare proof simplification against LeanLean under matched resource ceilings;
- compare subtheorem/coverage behavior against TheoremBench;
- compare research discovery against Formal Conjectures / Discover-and-Prove;
- introduce the proof comprehension + human-transfer suite as a separate track.

The new capability should be evaluated with open data, pinned environments, raw per-task rows and independent human review. We should publish negative results and false-positive rates. The ambition is to become a strong complementary research system, not to misrepresent one result as beating all of industry.

## 5. Claim policy

A leaderboard improvement on a supplied theorem is proof-generation evidence.  
A smaller file is proof-compression evidence.  
A correct dependency graph is structural evidence.  
A human-readable explanation is only explanation evidence when reviewed.  
A human transfer improvement is a learning/utility result.

Do not merge these into a single "mathematical intelligence score" until the component metrics are independently reliable, and even then keep the hard trust gates separate.
