# Quantifier-Aware Strategy Scopes — Research Note

Date: 2026-11-07  
Status: research extension and typed scope prototype; not a QBF solver  
Owner: Symthaea #10899

## 1. Why visible/hidden sets are not enough

The current finite reference checker supports propositional SAT, exact #SAT and projected model counting. For projected counting, a visible-variable scope and existential hidden witnesses are sufficient to express the basic support semantics:

`PMC(F,V) = |{v : exists h . F(v,h)}|`.

That representation is not sufficient for general quantified Boolean formulas. A prefix such as

`forall u exists x forall v exists y . F(u,x,v,y)`

contains dependency order. The Skolem function for x can depend on u, while y can depend on u and v. Flattening the prefix to unordered universal/existential sets loses the legal dependency scope and can make an invalid strategy appear valid.

The programme should therefore distinguish:
- a **variable universe**;
- an **ordered quantifier prefix**;
- the variables quantified in each block;
- the legal dependency scope of every existential variable;
- Skolem-function evidence for any claimed existential strategy.

The current prototype handles standard prenex QBF dependencies only. Dependency schemes, DQBF and non-prenex transformations are separate future work.

## 2. Current Rust scope prototype

Added the dependency-free module:

`crates/domains/symthaea-operations-research/src/quantified_scope.rs`

It exports:
- `Quantifier` (`ForAll`, `Exists`);
- `QuantifierBlock`;
- `QuantifierPrefixBudget`;
- `ValidatedQuantifierPrefix`;
- `QuantifierScopeError`.

The validator establishes:
1. the variable universe is bounded;
2. quantifier blocks are bounded and non-empty;
3. total variable occurrences equal the declared variable count;
4. each variable appears exactly once;
5. all variable indexes lie in the universe;
6. every existential variable's standard Skolem dependency scope is the ordered set of preceding universal variables;
7. any declared Skolem dependency set is a duplicate-free subset of that allowed scope.

It deliberately does **not** evaluate QBF, count quantified models, synthesize Skolem functions, or certify a compiled representation. No QBF correctness or performance claim follows from validating the prefix.

## 3. Research that makes this a timely extension

### Projected knowledge compilation

Bryant, Tan and Heule, *Certifying Projected Knowledge Compilation*, SAT 2025:
https://doi.org/10.4230/LIPIcs.SAT.2025.8

The work extends certified knowledge compilation to projected compilation by encoding Skolem assignments for quantified variables, and reports a soundness argument validated in HOL4. It provides both a high-performance C checker and a CakeML/HOL4-verified checker.

Research implication: projected-result evidence needs more than a visible support set. A certified projected representation should also explain how hidden variables can be assigned for a visible assignment, or provide an equivalent checkable witness relation.

### New QBF / quantified counting knowledge-compilation results

Chede, Krishan and Shukla, *On Knowledge Compilation Languages for QBFs*, ECCC TR26-279:
https://eccc.weizmann.ac.il/report/2026/279/

The report was posted 27 October 2026. It proposes QBF knowledge-compilation languages ranging from tree-based forms to more compact DAG forms, studies their expressive power, and connects them to #QBF proof systems. It reports a minor extension of Q-MICE that efficiently simulates the static systems considered, and an initial tree-like compiler/counting implementation. The work compares against d4-QBF and reports that its representation can handle doubly exponential model counts.

This is a new and directly relevant research frontier. The report should be treated as a recent research result, not as a mature industry benchmark or a result already reproduced by Symthaea.

### Certified counting and proof-system selection

The 2025 CPOG work presents proof-carrying knowledge compilation with a formally verified model counter; the July 2026 proof-system comparison establishes CPOG simulation strength over MICE and kcps, while those two are exponentially incomparable:
https://doi.org/10.1613/JAIR.1.15958
https://doi.org/10.1007/s10817-026-09757-w

Research implication: proof calculus, compiled representation and quantifier/dependency semantics must be treated as compatible typed capabilities. A proof system can only certify the claim types it actually defines.

## 4. Proposed quantifier-aware strategy pipeline

The longer-term strategy pipeline should add a quantified contract layer:

1. Parse the source QBF and its exact ordered prefix.
2. Validate the complete variable universe and quantifier blocks.
3. Derive legal Skolem dependency scopes.
4. Discover candidate tree/DAG/compiled representations.
5. Qualify transformations with quantifier-preservation and dependency-preservation guarantees.
6. Produce Skolem/witness evidence where the proof system requires it.
7. Execute the selected #QBF / QBF / MaxQBF solver or compiler adapter.
8. Independently check representation equivalence, quantified semantics, count/objective result and evidence binding.
9. Return a result contract that distinguishes truth, model count, optimum, completeness, and resource limitation.

The strategy interpreter must reject:
- quantifier block reordering without a qualified equivalence theorem;
- a Skolem function that depends on a future universal variable;
- an existential variable depending on another variable in its own simultaneous block under standard prenex semantics;
- missing/duplicated quantified variables;
- a proof backend that cannot certify the requested claim type;
- an incomplete quantified search presented as false, complete or optimal.

## 5. Benchmark plan for a future quantified track

Do not mix quantified results into SAT, #SAT or PMC scores. Define a separate QBF / #QBF / MaxQBF evaluation contract.

Required measurements:
- instances solved/counts verified by task type;
- quantifier alternation depth and block count;
- universal/existential variable counts;
- maximum and mean legal Skolem dependency width;
- Skolem circuit or witness size;
- compiled tree and DAG sizes;
- exact count bit width, including very large counts;
- proof generation bytes/time and checker time/memory;
- time to first valid strategy/witness and time to independently verified result;
- resource-limit and incomplete-proof rates;
- generalization across quantifier structures and formula families.

Baselines:
- d4-QBF and the solver/compilation approach in TR26-279 where reproducible artifacts are available;
- tree-like versus DAG-like compilation;
- the relevant static proof systems and Q-MICE variants;
- uncompiled QBF solvers for the same input contracts.

A fair comparison requires the same QBF semantics, prefix, instance set, hardware, memory and timeout. A compiler's ability to represent a doubly exponential count is not the same metric as its ability to solve every instance quickly.

## 6. Scope boundary

The current implementation is a structural type validator only. Keep QBF/#QBF execution out of the finite reference engine until:
- the prefix and dependency contract is reviewed;
- proof/Skolem artifact semantics are specified;
- a source-level independent checker exists;
- known-answer and adversarial quantified fixtures are frozen;
- d4-QBF and the new 2026 knowledge-compilation work can be reproduced.

This extension is intended to widen Symthaea's mathematical strategy language without weakening the correctness gates established for SAT, #SAT and PMC.
