# Exact minimum-cost connectome query-flipping witnesses V1

**Tracking:** [#9344 — Query robustness across plausible connectome reconstructions](https://github.com/Luminous-Dynamics/symthaea/issues/9344)  
**Related work:** [#9250 — correction interactions and portfolios](https://github.com/Luminous-Dynamics/symthaea/issues/9250), [#9987 — decision-value evidence selection](https://github.com/Luminous-Dynamics/symthaea/issues/9987), [#9255 — query-scoped reliability certificates](https://github.com/Luminous-Dynamics/symthaea/issues/9255).  
**Tool:** \`connectome-query-flipping-witness-audit\`.  
**Status:** standalone Rust exact enumerator, JSON Schema, synthetic synergy fixture, independent standard-library oracle and isolated CI authored. Biological validity and edit-universe completeness are not claimed.

## 1. Research question

Given a frozen baseline connectome graph, a frozen scientific query and decision rule, and a finite set of admissible, evidence-bound edit atoms:

> What is the minimum declared effort required for a compatible, precondition-valid edit set to reverse the baseline scientific decision?

This is a bounded scientific counterexample search. It is not a proof that the baseline graph is false, a recommendation to apply edits, or a certificate that the edit universe contains every biologically plausible alternative.

## 2. Formal problem

Let \(G_0\) be the frozen baseline graph, \(Q\) a query and \(D\) its binary decision rule. Let edit atoms be \(E=\{e_1,\ldots,e_n\}\), each with positive integer effort \(c_i\), a set of direct prerequisites \(R_i\), a symmetric conflict set \(X_i\), evidence digest \(h_i\), and zero or more absolute edge-state changes.

Find:

\[
S^* \in \arg\min_{S \subseteq E}
\sum_{e_i \in S} c_i
\]

subject to:

- every selected edit's prerequisites are also selected (transitive closure follows from direct closure);
- no selected edit conflicts with another selected edit;
- every graph mutation's expected edge state matches the same frozen baseline graph;
- no two selected edits mutate the same directed edge;
- total effort is at most the declared budget; and
- \(D(Q(G(S))) \ne D(Q(G_0))\).

The V1 semantics are deliberately simultaneous: every edit precondition is checked against \(G_0\), then all compatible selected replacements are applied to form one resulting graph. There is no hidden ordering effect. An edit sets one or more edge weights to absolute values; weight zero means delete/absent, positive means set that exact value. A graph-edit atom must contain at least one change. An evidence-gate atom has no graph mutations and represents a required adjudication/authorization prerequisite that contributes effort and evidence lineage.

Effort is an integer in a caller-declared unit. For example, \`review_minute_micro_units\` could encode micro-minutes. The verifier never infers units or treats them as probability.

## 3. Exact search and scope

V1 enumerates every non-empty subset of the declared edit atoms and evaluates the explicit prerequisite/conflict constraints and effort budget. It supports at most 20 edit atoms:

\[
2^{20}-1 = 1{,}048{,}575
\]

non-empty subsets. This is intentionally bounded and auditable. The report records the total subset count, budget-rejected subsets, constraint-rejected subsets, admissible subsets, decision-flipping subsets, minimum effort and all tied minimum witnesses (up to 128 returned, with an explicit truncation flag).

The input declares one of:

- \`exhaustive_declared_edit_universe\`: exhaustive only relative to the exact supplied finite universe;
- \`bounded_candidate_subset\`: a bounded subset of candidate edits, with no completeness claim.

A no-witness result is always:

\`NO_WITNESS_WITHIN_DECLARED_EDIT_UNIVERSE_AND_BUDGET\`

It must never be promoted to “no counterexample exists.” The scope note, edit-universe digest, query digest, baseline graph digest, budget and exact input digest are all retained. A search that cannot validate its source preconditions, dependency graph or conflict declarations fails closed instead of returning a reassuring negative result.

## 4. Edit admissibility contract

Every edit atom binds:

- stable edit ID and kind (\`graph_edit\` or \`evidence_gate\`);
- positive integer effort and an explicit effort unit at the input level;
- evidence SHA-256;
- direct prerequisite edit IDs;
- symmetric conflict IDs;
- absolute edge-state changes and exact expected baseline weights.

Validation rejects unknown/self/duplicate prerequisites and conflicts, cycles in the requirement DAG, asymmetric conflicts, stale expected edge states, multiple changes to one edge within an atom, non-finite/negative weights, unregistered same-edge alternatives that do not explicitly conflict, and edit sets larger than the exact-search bound.

The exact edit-universe digest canonicalizes edit order, prerequisite/conflict order and edge-change order. The graph digest uses the same domain-separated canonical graph encoding as the V1 finite query-ensemble evaluator.

## 5. Synthetic known-answer benchmark

The checked-in fixture is a weighted directed graph with three outgoing edges from node 1, each weight 0.6. The frozen query asks whether node 1's weighted out-degree is at least 1.0. Its baseline value is 1.8, so the baseline decision is met.

Each single graph correction leaves two 0.6 edges (or otherwise leaves the total above 1.0); no singleton flips the decision. Two edge removals leave only 0.6 and flip it. Each removal requires an evidence/adjudication gate costing one effort unit. The exact minimum witness therefore costs five effort units: gate + two removals.

The fixture deliberately includes:
- multiple equally cheap minimum witnesses;
- two alternative operations on the same directed edge, explicitly marked mutually conflicting;
- a higher-cost replacement alternative that conflicts with the removal;
- a cheap unrelated decoy edit;
- a prerequisite evidence gate;
- an effort budget exactly equal to the minimum witness cost.

Expected results:
- baseline out-degree: 1.8;
- baseline decision: met;
- all 127 non-empty subsets evaluated;
- five tied minimum witnesses at effort cost 5;
- every returned witness includes the evidence gate and two compatible graph edits;
- no witness contains both conflicting same-edge alternatives;
- no probability or posterior claim.

This fixture is synthetic software truth only.

## 6. Independent verification

The Python standard-library oracle uses:
- \`itertools.combinations\` rather than the Rust bit-mask enumeration;
- a Boolean reachability matrix and Floyd–Warshall rather than the Rust BFS query implementation;
- independent recomputation of the baseline and resulting graph digests;
- independent canonical query/edit-universe digests;
- exhaustive subset, prerequisite, conflict, effort and decision recalculation;
- negative controls that mutate the minimum cost, witness membership and resulting graph digest.

The oracle is test-only. It is not imported by production Rust code or used to produce the witness result. A passing oracle run establishes agreement for this exact finite fixture and supported V1 query/edit semantics, not a biological claim.

## 7. Required adversarial benchmark ladder

Permanent known-answer and negative controls should include:

1. **No singleton, pair synergy:** two edits jointly flip a query although neither does alone.
2. **Higher-order synergy:** no singleton or pair flips, but a three-edit set does.
3. **Conflicting alternatives:** mutually exclusive edits can never appear in one witness.
4. **Prerequisite closure:** a graph edit cannot be selected without all evidence gates it requires.
5. **Stale precondition:** expected edge weight differs from the frozen graph; fail closed.
6. **Cheap decoy:** a low-cost irrelevant edit must not be mistaken for a witness.
7. **Budget cutoff:** a real witness above the declared effort budget must not be returned.
8. **Tied optima:** every equal-minimum witness is counted, not just the first encountered.
9. **Permutation invariance:** edit input order does not change the canonical edit-universe digest or minimum witness set.
10. **No witness:** an invariant query returns the scoped no-witness disposition, never a global-robustness claim.
11. **Held-out edit:** add a scientifically admissible edit outside the declared universe; use it to demonstrate why a no-witness result is universe-bounded.
12. **Baseline revision:** change the baseline edge state; stale edit preconditions must reject the old universe.

V1 includes three executable known-answer inputs:
- Pair synergy plus an evidence gate: five tied minima at effort 5.
- Three-way synergy: four candidate edge removals, with no singleton or pair flipping the query; any three flip it at effort 3.
- Bounded no-witness: the declared edit subset contains only an evidence gate and an unrelated edge insertion, so no subset flips the query.

Each fixture is run through Rust and independently checked by the Python oracle. The bounded no-witness fixture deliberately omits the query-relevant removal alternatives present in the pair-synergy fixture, even though those alternatives would change the same baseline query. This is the executable counterexample to treating a negative search result as global robustness: the bounded search is correct about its declared subset while an omitted admissible alternative still exists. Unit tests also cover tied minima, preconditions, dependency cycles, symmetric conflicts, same-edge alternatives, budget cutoff, digest mismatch, input-order canonicalization and the 20-edit bound.

## 8. Position against graph-counterfactual minimization SOTA

A directly relevant comparator appeared in *Machine Learning* on 18 September 2026: García et al., “On the Minimization of Graph Counterfactual Explanations: Theory and a Local Bounded Search Algorithm” ([paper](https://doi.org/10.1007/s10994-026-07157-0)). It formalizes graph-counterfactual minimization as NP-hard and proposes Local Bounded Search (LBS), a model-agnostic generate-and-minimize heuristic. Across nine synthetic and real-world molecular, biomedical and social datasets, the paper reports structural edit-distance reductions of up to 98% from initial counterfactuals.

This changes our benchmark strategy. The current exact enumerator should be positioned as a **small-instance optimality oracle**, not as a scalable winner. The scientific assurance problem differs from generic graph explainability in several important ways:

- The target is a frozen scientific query and declared decision rule, not necessarily a black-box GNN prediction.
- Edits carry evidence digests, exact baseline preconditions, prerequisite gates and explicit conflicts.
- The objective is declared expert effort, not merely unweighted graph edit distance.
- A witness is bounded by an explicit biological edit universe and cannot be interpreted as an observed correction.
- The witness may reveal which observation or adjudication would falsify a scientific claim; generic graph-CF minimization does not itself establish evidential admissibility.

The fair comparator ladder should be:
1. Exact subset enumeration (ground-truth optimum on small cases).
2. García et al. LBS, adapted to the same admissible edit atoms and effort cost.
3. Greedy marginal query-sensitivity ranking.
4. Beam search / bounded best-first search.
5. ILP/SMT or branch-and-bound when a sound encoding and independently checkable optimality bound are available.
6. Random compatible subsets as a negative baseline.

Report validity, minimum-cost regret against the exact oracle, runtime, memory, witness diversity, dependency/conflict violations, human-adjudicated biological plausibility and reviewer time. Do not compare raw edit-count reductions when one method is allowed different edit atoms or a different cost model.

## 9. Benchmark targets (proposed, not results)

| Metric | Target |
|---|---:|
| Exact minimum-cost witness recovery on fully enumerable cases (≤20 atoms) | 100% |
| Regret against exact oracle on enumerable cases | 0 |
| False witness reports for inadmissible/stale/conflicting subsets | 0 |
| Missed tied minimum witnesses in returned count | 0 |
| Deterministic result under equivalent edit input permutations | 100% |
| Mutation controls detected by independent oracle | 100% |
| False “no witness” dispositions in exhaustive declared finite universes | 0 |
| Biological counterexample completeness | Not claimed; separately measured |

For larger candidate sets, compare exact search on smaller prefixes, admissible branch-and-bound/ILP/SMT solvers with independently checked optimality bounds, beam search, greedy marginal query sensitivity, and interaction-aware portfolio optimization. Report optimality gaps, witness validity, cost, runtime, peak memory and search coverage. Approximate methods must never label a result exact or return a global no-witness conclusion.

Scientific utility is not just minimum number of edits: future versions should compare expert minutes, evidence acquisition cost, expected scientific decision loss and robust worst-case regret. An effort-minimum witness is a review lead, not a posterior risk estimate.

## 10. Integration boundaries

- **#9344:** reconstruction-world uncertainty and query robustness. The witness search supplies a concrete falsifying edit set within a declared edit universe.
- **#9250:** correction interaction/portfolio semantics. The witness tool searches for decision-flipping sets; the portfolio tool optimizes utility under review budgets. Keep these estimands separate.
- **#9987:** value of next evidence. An observation can be prioritized by whether it can rule out a minimum witness or distinguish conclusion-changing worlds. Expected VOI requires justified probabilities/losses; otherwise report bounded or qualitative discrimination.
- **#9255:** a future certificate may bind query digest, baseline graph digest, edit-universe digest, exact witness report/input digests, verifier identity and scope. A witness result alone must never make a certificate certified.
- **#10435/#10453:** frame closure and execution-route observability remain upstream conjunctive prerequisites. This tool does not prove that all edit/evidence routes are registered or observable.
- **#10406:** evidence-dependence verification remains a separate prerequisite. The edit evidence digests identify artifacts but do not establish independent evidence.

## 11. Promotion boundary

This tool is research infrastructure and does not apply corrections to CAVE, H01, MICrONS, FlyWire or any production connectome. Every returned witness should route to human review and independent adjudication. Do not turn the result into an automated edit action, a biological error probability, or a scientific certificate until source, frame, dependence, query semantics, scenario coverage and prospective biological validation are separately qualified.
