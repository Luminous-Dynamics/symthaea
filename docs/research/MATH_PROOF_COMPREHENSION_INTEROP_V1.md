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


## 8. TheoremGraph / LeanGraph interoperation (2026-11-23)

TheoremGraph is now a required external baseline, not merely adjacent context.

- Paper: [TheoremGraph: Bridging Formal and Informal Mathematics](https://arxiv.org/abs/2606.25363).
- Project and API: [TheoremSearch](https://github.com/uw-math-ai/TheoremSearch), [Theorem Graph API](https://www.theoremsearch.com/docs).
- Dataset: [Math-Graph](https://huggingface.co/datasets/uw-math-ai/math-graph), CC-BY-4.0; retain attribution, dataset revision and source metadata.
- LeanGraph source: [aurasoph/lean-graph](https://github.com/aurasoph/lean-graph), Apache-2.0.

The paper reports 11.7M informal mathematical statement nodes, 18.3M candidate informal dependency edges, 388,105 formal Lean declaration nodes and 11.3M typed formal dependency edges across 25 projects. Its formal graph distinguishes `sig`, `proof`, and `def` edges, alongside structural/documentation relations `extends`, `field`, and `docref`. Signature edges also preserve binder position/kind, function-versus-argument role, and projection metadata.

### Exact and inexact cross-formality relations

The headline set of 47,952 judge-affirmed cross-formality links must not be treated as 47,952 faithful formalizations. The authors' November 19, 2026 rebuttal separates 4,765 exact matches from 43,187 inexact/related matches (for example, special cases, generalizations or one direction of an equivalence). Inexact matches can be useful retrieval candidates, but they are not evidence of equivalence or paper-to-Lean fidelity. The reviewed evidence is limited: one expert reviewed twenty pairs; a separate 199-pair sample was assessed by two LLM judges. Reported judge agreement is repeatability, not correctness. These labels must remain candidate annotations until independently adjudicated.

### LeanGraph source-version distinction

Do not conflate the historical LeanGraph submodule revision recorded in the TheoremSearch corpus (`9823e3f8f45036b124905e541c7dc92c39620283`) with the later raw-graph implementation inspected at `3bf57c70f0326e45b9e84345c8fe7cf520919358`.

At the historical pinned revision, the unified graph applies signature filtering/parent redirection, proof/definition transitive expansion, and structure/field/docstring overlays. It is therefore a filtered/expanded view, not a direct raw `getUsedConstants` export. The later revision introduces `rawUnifiedGraph` and derives a readable view as a post-pass. The raw and readable outputs are different semantic artifacts and must be evaluated separately.

### Revised extractor strategy

Do not rebuild a second 388k-node graph extractor. Use the current LeanGraph raw mode as an external differential baseline where toolchain and environment pins are compatible. Keep #13295 focused on:

- exact local receipts binding Lean/Mathlib/Lake/module closure, declaration identity, type/value availability and digests, universes, axiom/`sorryAx` profile, source ranges and serializer version;
- direct type edges, direct value edges, filtered/expanded edges and overlays as distinct provenance classes;
- explicit opaque-value-unavailable status rather than an empty successful proof-dependency list;
- explicit axiom and inductive/constructor/recursor records;
- differential tests on the four pinned Mathlib and six Lean-core pilot declarations already in #13294.

The first differential should compare (1) the historical dataset graph, (2) current LeanGraph raw output, and (3) Symthaea's local extracted output. Explain each mismatch by filter/redirect, transitive expansion, type/value classification, opaque values, irreducible-definition helper, inductive/recursor, declaration filter, or unknown. Do not compare only edge counts.

### Role-aware informal/formal relations

The adapter should distinguish `exact_equivalent`, `formal_implies_informal`, `informal_implies_formal`, `specialization`, `generalization`, `related_not_equivalent`, `wrong_match`, and `unresolved`. Each relation needs explicit direction, assumptions/domain restrictions, translation provenance, and review disposition. #11362 remains the owner of the full paper-claim fidelity audit; this section specifies interoperation, not a duplicate audit system.

The new differential work is tracked in [#13325 — MATH-GRAPH-INTEROP-001](https://github.com/Luminous-Dynamics/symthaea/issues/13325). No local LeanGraph build, dataset ingest, or differential execution is claimed here; this is a source- and paper-grounded interoperation contract.


## 9. Proof-plan reconstruction is a separate semantic layer (2026-11-23)

The graph-interoperation work must not collapse proof understanding into dependency retrieval. Hamami & Morris (2024) characterize proof understanding as rational reconstruction of the underlying plan ([DOI](https://doi.org/10.1111/nous.12489)). Gao & Evans (2026-11-06) studied proof presentation in 143 undergraduates and found that strong presentation performance was only moderately associated with delayed near-transfer proof construction ([DOI](https://doi.org/10.1007/s40753-026-00319-4)). These motivate a separate `ProofPlanV1` artifact and outcomes for plan reconstruction, proof validation/presentation, immediate transfer, and delayed transfer.

A proof-plan graph records goal states, mathematical actions, justifications, side conditions, generated subgoals, witnesses/constructions, invariants and unresolved steps. Typed edges include `transforms`, `justified_by`, `requires`, `creates_subgoal`, `preserves`, `discharges`, `constructs`, `alternative_to`, and `evidenced_by`. This is not inferred mechanically from Lean dependency edges: the relationship is many-to-many and the mathematical action labels require review.

The same theorem may have several valid plans. Benchmarking must accept distinct valid routes and compare their side-condition burden, explanatory clarity and cost without turning one proof route into a claim of mathematical necessity. See [#13326 — MATH-PROOF-PLAN-001](https://github.com/Luminous-Dynamics/symthaea/issues/13326).
