# Mathematical proof comprehension — interoperation contract v1

**Parent:** [#13293 MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Related implementation:** [#13295 MATH-LEAN-EXTRACT-001](https://github.com/Luminous-Dynamics/symthaea/issues/13295)  
**Competitive ledger:** [MATH_PROOF_COMPREHENSION_COMPETITIVE_LEDGER_V1.md](MATH_PROOF_COMPREHENSION_COMPETITIVE_LEDGER_V1.md)

## 1. Positioning

Symthaea should not rebuild capabilities already demonstrated by adjacent systems:
- HumanLean: staged, source-faithful paper-to-Lean formalization and human review.
- APE-Bench / APE-Harness: repository-scale proof engineering with dual syntactic and semantic verification.
- LeanGraphSearch: semantic seed retrieval plus multi-hop traversal over Lean declaration dependencies.

Our research object begins after or alongside these capabilities:

`faithful source / formalization contract
→ exact Lean theorem identity and dependency graph
→ mathematical mechanism / concept graph
→ evidence-linked human explanation
→ counterfactual diagnosis
→ reader comprehension and transfer`

The contribution is not "we have proof fidelity" or "we use dependency graphs." It is a reproducible bridge from formal proof structure to conceptual understanding and measured transfer.

## 2. HumanLean interoperation

Reference paper: Yuhao et al., *Faithful Lean Formalization for Human Proof Review* (2026, under review): https://files.sri.inf.ethz.ch/website/papers/yuhao2026humanlean.pdf

HumanLean uses a staged formalization workflow and a structured goal contract. Its evaluation separates acceptance, formalized goal coverage, faithfulness, cost, dependency exposure, and verbosity. Its GPT-6-Astra result on nine paper case studies reports 60/60 accepted goals formalized and 100% faithfulness for those formalizations, with two independent human reviewers confirming the faithfulness result. This is 60 out of 150 eligible goals, not 100% coverage of all eligible goals. Its paper also reports a review-cost trade-off and a 5–20 minute expert review per goal.

### Consume, do not clone

Symthaea's intake adapter should preserve, when supplied:
- paper / source identity, revision and source spans;
- goal ID and natural-language theorem statement;
- formal Lean declaration and elaborated type identity;
- assumptions and domain restrictions;
- definitions and notation;
- formalized versus excluded goals;
- external dependency declarations / axioms;
- proof route and helper lemmas;
- source, definition, scaffold and final-proof review dispositions;
- clarification records and unresolved source obligations.

Do not turn an external dependency declared as an axiom into a locally proven theorem. Keep formalization acceptance, faithfulness and proof checking as separate evidence records.

### What Symthaea adds

- reconstruct the mathematical mechanism for an already formalized goal;
- show why each assumption, definition and helper matters;
- make an evidence-linked concept graph from the formal dependency graph;
- construct examples and valid counterfactuals;
- explain at several mathematical levels;
- measure whether readers can restate, critique and transfer the result.

HumanLean's human review of paper-to-formalization faithfulness is not a substitute for a comprehension study.

## 3. APE-Bench / APE-Harness interoperation

Reference paper: Xin et al., *APE-Bench: Evaluating Automated Proof Engineering for Formal Math Libraries*, ICML 2026 / PMLR 306: https://proceedings.mlr.press/v306/xin26c.html

Public reference implementation:
- Repository: https://github.com/xinhjBrant/APE-Bench
- Pinned inspected commit: `20017343657a7ce77fc67467c3dc9e443c81c075`
- License: MIT
- README reports 100 proof-engineering tasks and 64 semantic-judge validation tasks.
- APE uses dual verification: compilation in a pinned library environment plus semantic requirement satisfaction.

### Reuse its task boundary

Where feasible, adapt our proof-comprehension tasks to an APE-style task contract:
- immutable target repository and source commit;
- task input and source artifacts;
- expected theorem / semantic requirement;
- permitted tools and resource budget;
- candidate output;
- syntactic build result;
- semantic requirement / theorem-identity result;
- raw evidence and digests.

Do not duplicate APE-Harness execution, content-addressable builds, multi-version compile database, or isolated runtimes. Use APE-Bench as an external regression lane for proof engineering and semantic judgment. Keep our human-comprehension and transfer outcomes separate from APE's formal task acceptance.

## 4. LeanGraphSearch interoperation

Reference: Xu et al., *Lean Graph Search for Formal Theorem Proving* (2026-10-15): https://arxiv.org/abs/2610.22339

Reported results:
- MathlibQR Recall@100: 94.7% versus 82.7% for LeanSearch v2 on the 810-query fair subset.
- MathlibMPR complete-route coverage: 42.8% versus 39.1% at Covered@20, and 44.2% versus 41.3% at Covered@50.
- Downstream proof pass-rate improvements up to 40% over vanilla and up to 10% over LeanSearch v2 across the paper's evaluated settings.

LeanGraphSearch uses jixia static analysis to build Mathlib declaration edges and query-personalized PageRank (PPR) to expand from semantically selected seeds.

### Retrieval baseline requirement

Our evaluations must explicitly include:
1. lexical / BM25;
2. dense embedding retrieval;
3. LeanSearch v2 style semantic retrieval;
4. LeanGraphSearch-style semantic seeds + dependency PPR + reranking;
5. symbolic / type-directed retrieval;
6. hybrid retrieval;
7. hybrid + HDC;
8. hybrid + HDC + CfC.

Do not claim graph retrieval novelty or an HDC advantage without beating this baseline on a frozen, held-out corpus at matched budgets.

### Graph extraction comparison

The extractor in #13295 should compare three edge sources on identical declarations:
- Lean environment traversal from elaborated declaration type/value and inductive/recursor metadata;
- jixia / LeanGraphSearch-style static dependency edges;
- external kernel-export edges where available (e.g., RH proof dependency dataset).

Keep graph provenance and edge semantics explicit. Differences are research data:
- static source references may include references not present in elaborated kernel proof terms;
- type references and proof-value references are different edge classes;
- opaque declarations, inductives and recursors require special handling;
- a kernel dependency edge is not a human-reviewed mathematical role.

Do not blindly merge these into one graph or assign the highest authority to a candidate edge without provenance.

## 5. Unified capability matrix

| Capability | Upstream comparator / evidence | Symthaea responsibility |
|---|---|---|
| Source-to-Lean faithfulness | HumanLean contract / reviewer protocol | Consume provenance and audit boundaries; do not duplicate its formalization workflow |
| Formal proof acceptance | Lean kernel | Preserve exact declaration/type/toolchain and checker receipts |
| Repository-scale semantic task | APE dual verification | Adapt task contracts and compare; do not recreate its runner |
| Premise retrieval | LeanSearch v2 / LeanGraphSearch | Include multi-hop PPR baseline and report matched-budget retrieval metrics |
| Dependency extraction | Lean native environment / jixia / kernel export | Differentially compare typed edge semantics |
| Mathematical role interpretation | Human expert review | Build a reviewed concept layer over machine graph edges |
| Human explanation | HumanLean review supports inspectability, not reader transfer | Produce source-linked explanations and measure reader comprehension |
| Counterfactual understanding | Not established by proof acceptance alone | Test assumption/definition changes with Lean and counterexamples |
| Human and cross-family transfer | Separate research target | Run blinded near-transfer and held-out mechanism-transfer studies |

## 6. Revised pilot and acceptance

### Pilot 1: small Mathlib theorem

- Reuse a HumanLean-style goal/source/dependency contract where a matching artifact is available.
- Extract the formal declaration graph with #13295.
- Compare to LeanGraphSearch/jixia-style graph edges.
- Produce the concept graph and explanation.
- Check the explanation's factual claims against the exact source and theorem type.
- Use one counterfactual and one near-transfer question.
- Do not use it as a gold task before two independent reviewers approve it.

### Pilot 2: RH inequality (M)

- Use the externally published RH kernel dependency dataset as a reference map.
- Keep source-to-theorem fidelity separate.
- Reconstruct the bounded `ASK / MAIN → each_unit_bound_statement → ch_top → gapA` chain.
- Explain cell centering and finite-group translation separately from the uniform analytic estimates.
- Preserve unresolved theta-integrality, trace/rank and Galois-cohomology inputs.
- Compare the local extractor to external graph edges only after pinning exact declarations.

### Pilot 3: ETH / exact counting

- Use the finite 5-source / 5-correct / 6-mutated Tseitin fixture.
- Explain why decision-preserving is weaker than unique-extension / parsimonious.
- Compose the explanation with #10811 / #10835's typed guarantee registry.
- Transfer the mechanism to a different reduction family only after the first task is reviewed.

### Hard gates

- zero theorem identity substitution;
- zero unsupported formal facts;
- explicit external axioms;
- exact source/toolchain/graph provenance;
- no mathematical-role inference from PPR, PageRank, HDC similarity or centrality;
- unknown accepted over fabrication;
- human transfer results kept separate from formal validity.

## 7. What would count as a genuine Symthaea contribution?

A result that:
1. consumes an existing faithful formalization and a formally extracted proof graph;
2. identifies the mathematical mechanism and the exact assumptions it needs;
3. gives a human-readable explanation whose factual claims are traceable;
4. detects a subtle counterfactual change correctly;
5. improves a human reader's comprehension or transfer accuracy on a blinded holdout;
6. reduces explanation/reviewer cost at matched comprehension;
7. generalizes to a theorem family not used during development.

That would complement, rather than relabel, the achievements of HumanLean, APE-Bench and LeanGraphSearch.
