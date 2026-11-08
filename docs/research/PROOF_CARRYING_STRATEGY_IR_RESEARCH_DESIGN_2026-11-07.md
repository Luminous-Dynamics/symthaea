# Proof-Carrying Strategy IR — Research Design v0

Date: 2026-11-07  
Status: design proposal; not a normative schema or a verified implementation  
Owner workstream: [Symthaea #10899](https://github.com/Luminous-Dynamics/symthaea/issues/10899)

## 1. Purpose

Symthaea's algorithm-discovery unit should be an executable **computational strategy**, not an isolated solver or heuristic.

A strategy composes:

1. a source problem contract;
2. representations and qualified transformations;
3. proof-carrying decomposition;
4. specialist solver invocations;
5. mathematically specified aggregation;
6. witness reconstruction or projection;
7. independent validation;
8. an anytime stopping policy.

This note proposes the smallest language that can execute the three finite examples in [#10868](https://github.com/Luminous-Dynamics/symthaea/issues/10868) without inventing another solver, proof calculus, benchmark receipt or provenance format.

This is a design proposal. The type rules below become normative only after review, counterexamples and an independently implemented checker agree.

## 2. Design constraints

- **Declarative before generative.** A strategy names registered transformations and pinned solver adapters. It does not embed arbitrary generated code in the trusted interpreter.
- **Proof obligations before execution.** A hyperedge with an unqualified aggregation law is not executable.
- **Source semantics remain authoritative.** Feasibility is checked against the original task, not only a transformed representation.
- **Evidence is not a Boolean.** A digest establishes identity; a producer-supplied `accepted` or `verified` field establishes nothing by itself.
- **Unknown is a first-class result.** Budget exhaustion, cancellation, crash and unsupported operations cannot become UNSAT, infeasible, complete or optimal.
- **Cost includes verification.** Solver runtime, proof generation, certificate bytes, independent checker runtime and checker memory are measured separately.
- **Reuse existing contracts.** Strategy execution reuses #10805 receipts, #10884 result semantics, #10811 reduction records, #10835 composition rules and #10868 hyperedges.

## 3. Core typed values

The exact Rust representation remains an implementation decision. The semantic type system should distinguish at least these dimensions:

| Type dimension | Examples | Why it cannot be erased |
|---|---|---|
| Task | SAT, #SAT, PMC, WMC, PWMC, AMC, MaxSAT, finite optimization | Decision, counting and optimization have different output semantics |
| Representation | CNF, factor graph, hypergraph, circuit, CSP | A solver's input language is not the source problem |
| Variable universe | constrained, free, visible, hidden, auxiliary | Omission and projection change counts |
| Guarantee | decision-preserving, parsimonious, unique-extension, known-factor, approximation-preserving, optimization-preserving | A weaker reduction cannot be silently upgraded |
| Algebra | Boolean support, naturals, rationals, declared exact field | Aggregation is domain-dependent |
| Objective | sense, exact value type, units, tie policy | Bounds and comparisons must agree on the same objective |
| Randomness | deterministic, or algorithm + seed/policy + error parameters | Exact arithmetic does not imply deterministic execution |
| Resource effect | decisions, propagations, clauses inspected, proof work, bytes, memory, CPU, wall time, energy | Reproducibility and cost composition require explicit units |

A future generalized value can be viewed as:

`Task<Kind, SourceContract, Representation, Universe, Algebra, Guarantee>`

This is conceptual notation, not a Rust API commitment. In particular, do not encode every semantic dimension in a single unvalidated string.

### Algebra must be explicit

Examples of distinct aggregation semantics:

- SAT branch existence: Boolean OR of child decisions.
- #SAT disjoint branching: integer addition after proving an exhaustive, disjoint partition.
- Independent #SAT components: integer multiplication after proving pairwise-disjoint variable scopes and accounting for free variables.
- Projected counting: union of visible supports / existential Boolean aggregation; hidden witnesses that produce the same visible assignment collapse to one support element.
- Weighted or algebraic counting: a declared coefficient domain and its operations, including negative weights or cancellation where supported.

Do not infer a valid algebra merely from the task label. For example, projected counting with arithmetic addition is wrong when hidden witnesses overlap.

## 4. Restricted v1 strategy graph

Use a strictly parsed, versioned, canonically serialized acyclic graph. V1 should have these node forms:

| Node | Input | Output | Static obligation |
|---|---|---|---|
| `Input` | source contract + input bytes | typed source artifact | schema, canonical digest, variable universe and objective agree |
| `Transform` | typed representation | target representation + mapping | registry entry is qualified for this task and guarantee |
| `Decompose` | source representation | ordered child tasks + hyperedge witness | partition/scope/separator and child identities satisfy the registry contract |
| `Solve` | child task + pinned adapter + budget | typed partial or complete solver outcome | adapter supports the exact task and representation |
| `Aggregate` | child outcomes + hyperedge | parent result | use the operator bound by the qualified hyperedge, never a caller-selected replacement |
| `Reconstruct` | child witnesses/results | source-level witness/result | reconstruction preserves the declared semantics |
| `Validate` | source contract + source-level artifact | independent checker receipt | checker is separate and validates against the original source contract |
| `Checkpoint` | candidate + checker receipt | admitted incumbent event | complete witness, exact request binding and validator acceptance |
| `Stop` | search/evidence state + budget | typed disposition | no status upgrade without qualifying evidence |

V1 supports one source per hyperedge, acyclic composition, deterministic topological execution and bounded fan-out. It does not support recursive strategies, general multi-source hyperedges, dynamic code loading or unconstrained parallel execution.

## 5. Static typing and capability checks

A strategy is accepted for execution only if every edge passes a type/effect check.

Illustrative rejection rules:

1. A decision-preserving transformation cannot feed an exact #SAT result claim unless a separate multiplicity/count-preservation proof is supplied.
2. A non-unique auxiliary encoding cannot be used as parsimonious counting without a proved extension multiplicity correction.
3. `IndependentProduct` is rejected if component variable scopes overlap or fail to cover the constrained universe.
4. Free variables must be represented in the universe and contribute their declared multiplicity.
5. `BranchSum` requires a proved exhaustive, pairwise-disjoint branch partition.
6. Projected aggregation must preserve visible support and collapse repeated hidden witnesses.
7. A WMC/PWMC/AMC plan must bind coefficient domain, weight semantics and exact/approximate mode.
8. An exact objective comparison cannot pass through binary floating point.
9. A randomized approximate result must bind its probability/error parameters and cannot be represented as exact.
10. A bounded solve that stops early yields UNKNOWN/resource-limited, not exhaustive UNSAT.

The interpreter should call the existing guarantee-composition rules from #10835 rather than implementing a second reduction calculus. The hyperedge registry in #10868 owns its aggregation law and proof obligations.

## 6. Resource/effect semantics

Keep declared limits and measured consumption separate.

For a sequential plan:

- wall-time upper envelope: checked sum of child wall-time envelopes plus transformation, aggregation and checking overhead;
- CPU work: sum of child CPU work plus overhead;
- memory: peak live allocation, calculated from the execution/liveness schedule, not blindly the maximum or sum of every child;
- proof output: emitted bytes and generation work;
- checking: separate checker wall time, CPU, memory and outcome.

For a bounded parallel fan-out:

- wall time: maximum child wall envelope plus dispatch, synchronization and aggregation overhead;
- CPU work and energy: additive across children;
- peak memory: maximum concurrent live-set sum plus coordinator/aggregation memory.

All arithmetic is checked. Overflow is a typed rejection/resource outcome. V1 may execute sequentially only; the parallel algebra is specified now to avoid accidentally treating concurrent costs as sequential.

Deterministic operation budgets (decisions, propagations, clause inspections, proof events, trail entries) are distinct from an outer wall-clock deadline. Every incomplete result carries the exact budget dimension that stopped it.

The strategy objective is multi-criteria. Preserve a Pareto set over verified task value, time-to-first-verified-result, completion/optimality, proof bytes, checker work, CPU, peak memory and energy. Do not compress these into one reward until a separately documented policy justifies the weights and sensitivity.

## 7. Interpreter state machine

Each execution maintains:

- immutable problem-contract identity;
- the typed strategy graph and its source digest;
- node states: pending, running, completed, resource-limited, failed, rejected;
- per-node input/output digests and pinned implementation/configuration identities;
- typed solver outputs, kept distinct from accepted evidence;
- admitted witness set, where every item has a source-level witness and independent validator receipt;
- accepted bounds/completion/infeasibility evidence, each bound to the same immutable request/objective/source tuple;
- deterministic work counters and measured resource counters.

The interpreter can emit a feasible incumbent as soon as the source-level witness is independently accepted. Later failure, timeout or failed proof generation must not erase that incumbent. Optimality and infeasibility require the evidence in #10884/#10887. A solver status or empty queue is never sufficient completion evidence by itself.

## 8. Three executable reference plans

These plans refer to the frozen finite known-answer cases in [#10868](https://github.com/Luminous-Dynamics/symthaea/issues/10868). The identifiers below are descriptive; final IDs and fields belong to the v1 schema.

### A. Disjoint #SAT cofactor branching

`Input(F, {x,y,z}) -> Decompose(BranchSum, split=x) -> Solve(F[x=0]), Solve(F[x=1]) -> Aggregate(sum) -> ValidateAgainst(F)`

Preconditions: the branch conditions are exhaustive and disjoint; each child preserves the remaining variable universe including free variables. The fixture expects child counts 2 and 2 and source count 4.

### B. Independent components with a free variable

`Input(F, {a,b,c,d,e}) -> Decompose(IndependentProduct, components={a,b},{c,d}, free={e}) -> Solve(child_1), Solve(child_2) -> Aggregate(product × 2) -> ValidateAgainst(F)`

Preconditions: component scopes are pairwise disjoint; constrained scopes are covered; free variable `e` contributes a factor of 2. The fixture expects (2 \times 3 \times 2 = 12).

### C. Projected support with overlapping hidden witnesses

`Input(F(v,h1,h2), visible={v}, hidden={h1,h2}) -> Decompose(ExistentialSupportUnion) -> Solve(children) -> Aggregate(BooleanSupportUnion) -> ValidateProjection(F)`

For (F=(v\land h_1)\lor(v\land h_2)), there are three satisfying full assignments but only one supported visible assignment. Summing hidden-witness counts is an intentionally invalid plan.

All three plans must be checked both by their local hyperedge verifier and by direct source enumeration for the finite fixtures. Those checks are reference tests, not scalable correctness proofs.

## 9. Minimum adversarial plan corpus

Every invalid plan should fail before it can produce a promoted result. Include at least:

- missing or duplicated child branch;
- overlapping branch conditions;
- overlapping component scopes;
- omitted free variable;
- wrong aggregation operator;
- visible/hidden scope overlap or omission;
- decision guarantee used for exact counting;
- non-unique Tseitin extensions used as parsimonious;
- target/input digest mismatch;
- stale objective or source-data digest;
- randomized result falsely labelled deterministic;
- budget exhausted but status set to UNSAT/complete;
- producer-provided `accepted=true` without independent checker evidence;
- a child failure after an earlier valid incumbent;
- resource arithmetic overflow;
- checker timeout or crash.

Every negative case should assert both the typed rejection reason and the absence of any prohibited result claim.

## 10. Qualification and promotion gates

1. **Schema gate:** strict parser, canonical serialization, unknown-field rejection.
2. **Static gate:** type/effect rejection of semantically invalid plans.
3. **Finite semantic gate:** 100% source-vs-child/aggregate agreement on the supported exhaustive fixture corpus plus mutation tests.
4. **Independent-check gate:** source-level models/results are accepted by a checker not implemented by the strategy producer.
5. **Budget gate:** no resource-limit path can manufacture completion, UNSAT, infeasibility or optimality.
6. **Differential gate:** reference solver and bounded trail solver agree with brute force on the exhaustive small corpus.
7. **Benchmark gate:** compare direct solver, fixed specialist portfolio, static selector and typed strategy under identical total budgets.
8. **Generalization gate:** hold out complete benchmark families, generator lineages and structural clusters; reserve a sealed confirmatory split inaccessible to strategy/generator search.
9. **Reproduction gate:** pin source, build, toolchain, machine, solver versions, seeds, input digests, receipts and independent checker versions.

Promotion from research to production requires all applicable gates. A benchmark win cannot compensate for a semantic failure; a proof of a finite example cannot establish scalable performance.

## 11. Research benchmark matrix

| Domain | External reference | Strategy-specific question |
|---|---|---|
| SAT decision | SAT Competition 2026 Main Track; pinned Kissat/CaDiCaL adapters | Can representation + decomposition + portfolio selection improve verified solved count and PAR-2? |
| #SAT / PMC / WMC / AMC | Model Counting Competition 2026; Ganak and appropriate exact/PAC solvers | Can the planner select algebra- and projection-correct decompositions and beat exact/approximate baselines without mixing their guarantees? |
| MaxSAT | MaxSAT Evaluation 2026 exact, anytime and incremental tracks | Does the strategy preserve better verified incumbents sooner, and does proof/bound checking change the best portfolio? |
| General combinatorial optimization | CO-Bench / LACE baselines and established domain solvers | Does the approach generalize to structurally new problem contracts rather than overfit a solver's training families? |

Exact, PAC, high-precision floating-point and heuristic results must remain distinct categories. A probabilistic algorithm using arbitrary-precision integers is not thereby a deterministic exact algorithm.

## 12. Open design questions for review

1. Should the graph be a strict DAG in v1, or should a single bounded `RepeatUntil` construct be admitted? Recommendation: DAG only until interpreter and evidence replay are qualified.
2. Should the type checker be a Rust module plus Lean theorem statements for composition rules? Recommendation: yes, but only prove a small core and keep executable semantic checks in Rust.
3. How should randomization be represented? Recommendation: explicit algorithm identity, seed stream derivation, error/confidence parameters and any theorem that connects repetitions to the final guarantee.
4. What is the smallest trusted computing base? Recommendation: parser, type checker, deterministic scheduler, evidence binder and verifier invocation adapter; keep solver algorithms and generators outside it.
5. When does a finite result become a proof? Only when a general proof object/checker or a clearly scoped finite exhaustive argument supports that exact claim.

## 13. Dependencies

- #10811 — representation/reduction registry.
- #10835 — guarantee composition and complexity-transfer calculus.
- #10848 — strategy/instance co-evolution.
- #10868 — proof-carrying AND/OR hyperedges.
- #10805 — reproducible benchmark runner and pinned result audit.
- #10810 — SAT decision and theorem-proof separation.
- #10884 — anytime result semantics.
- #10887 — independent semantic verifier.
- #10899 — executable strategy IR and interpreter workstream.


## 14. Proof calculus is a strategy choice, not a post-processing detail

A strategy's output semantics and its certificate calculus are related but distinct. Two strategies may compute the same exact #SAT result through different decompositions and emit very different certificates. The strategy plan therefore needs a proof-backend capability and cost profile, bound to the task and aggregation semantics. Do not hard-code one proof calculus as universally optimal.

The July 2026 proof-complexity result **The Relative Strength of #SAT Proof Systems** establishes that CPOG simulates both MICE and kcps, while MICE and kcps are exponentially incomparable; CLIP simulates all three and is theoretically very strong. The authors further discuss CPOG and restricted CPOG/Decision-DNNF or kcps+resolution variants as potentially practical proof-logging candidates. This is a proof-complexity simulation result, not an empirical claim that CPOG always produces the smallest or fastest certificate.

Source: Beyersdorff, Fichte, Hecher, Hoffmann & Kasche, *The Relative Strength of #SAT Proof Systems*, Journal of Automated Reasoning 70 (2026), 12, https://doi.org/10.1007/s10817-026-09757-w. Open full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC13385077/.

For SAT, CaDiCaL 3.0 (SAT 2026) is an important engineering baseline: it adapts Kissat's techniques to incremental solving, supports implied-literal extraction under assumptions and emphasizes efficient linear proof production with hints. That is evidence that solver features, incremental semantics and certificate production have to be co-designed. It does not mean a Symthaea strategy should replace CaDiCaL's core solver.

Source: Pollitt et al., *CaDiCaL 3.0*, SAT 2026, https://doi.org/10.4230/LIPIcs.SAT.2026.40.

### Required strategy proof-backend profile

For each supported task/representation pair, record:
- proof system and exact version / language revision;
- which result claims the system can certify (decision UNSAT, exact count, projected count, optimization bound, completion, etc.);
- checker implementation, independence and formal-verification status;
- whether checking can stream or run concurrently with production;
- certificate bytes, peak checker memory and checker wall/CPU time;
- proof-generation overhead and whether proof logging changes the solver's search behavior;
- supported transformations, preprocessing and incremental assumptions;
- known lower bounds or incomparability results relevant to the selected proof system.

A proof calculus is eligible only for the claims it actually supports. Do not infer that a #SAT proof system establishes projected counting, weighted counting, or optimization bounds without a separately established semantics and checker.

### Initial #SAT proof-backend experiment

Compare CPOG / Decision-DNNF-oriented certificates, MICE, and kcps or kcps+resolution where usable implementations and adapters exist. Use the same exact counting instances and pinned solver runs. Record count correctness, certificate size, generation overhead and independent checking throughput. Treat CLIP primarily as a theoretical comparison until a practical producer/checker pair exists.

Required ablations:
1. solver without certificate production;
2. solver with native proof production;
3. proof generation plus independent offline checking;
4. streaming/on-the-fly checking where supported;
5. representation/decomposition chosen without proof-cost awareness versus jointly chosen with proof-cost awareness.

The scientific question is whether proof-aware strategy synthesis can improve time-to-trusted-answer or extend the set of instances with practically checkable results, without degrading mathematical correctness.


## 15. Quantifier-aware scope extension

A plain visible/hidden variable split is adequate for the initial PMC examples but is not a sufficient semantic type for quantified Boolean formulas. Prenex QBF carries an ordered quantifier prefix; an existential variable's Skolem function can depend only on universal variables quantified before it. The strategy type system must preserve the prefix order and legal dependencies rather than flattening them into unordered role sets.

A structural prototype is now present in `symthaea-operations-research::quantified_scope`. It validates exact prefix coverage, block non-emptiness, uniqueness, bounds and standard prenex Skolem dependency scopes. It does not solve QBF/#QBF/MaxQBF or certify Skolem functions.

This extension is motivated by:
- Bryant, Tan & Heule, *Certifying Projected Knowledge Compilation*, SAT 2025: projected compilation certificates encode Skolem assignments; soundness was validated in HOL4, with C and CakeML/HOL4 checkers. https://doi.org/10.4230/LIPIcs.SAT.2025.8
- Chede, Krishan & Shukla, *On Knowledge Compilation Languages for QBFs*, ECCC TR26-279, posted 27 October 2026. It explores tree/DAG representations, #QBF proof systems and an initial model counter that handles doubly exponential counts, compared against d4-QBF. https://eccc.weizmann.ac.il/report/2026/279/

The next research objective is to compose quantifier-prefix validation, Skolem evidence, compiled representation guarantees and independently checked quantified result semantics. Keep actual quantified solving outside v1 until the proof objects and independent checker exist.


## 16. Claim-specific proof backend capability

A result status and a proof format are not interchangeable. The semantic verifier must reject evidence produced by a backend that cannot certify the exact claim, even if the producer says that verification succeeded.

Model proof capability as a typed adapter contract, not another universal proof language:

`ProofBackendCapability = (task kind, representation, claim kind, proof format/version, checker identity/version, input-binding rules, supported transformations, deterministic/randomized guarantee, resource profile)`

Every admitted evidence object must bind:
- immutable source problem / request identity;
- task semantics, objective or quantifier prefix;
- claim type (witness feasibility, UNSAT, exact count, projected count, weighted/algebraic count, quantified truth, #QBF count, optimization bound/optimality, completion);
- proof-system and certificate format revision;
- producer implementation/configuration identity;
- independent checker implementation/build identity and outcome;
- certificate digest/length and measured generation/checking resources.

A theoretical simulation theorem, a checker binary, and a producer adapter are separate facts. A proof system may be theoretically stronger without being the most practical producer or checker. CPOG's simulation result over MICE and kcps does not establish universal practical speed or certificate-size dominance.

Illustrative compatibility boundaries (subject to exact adapter/version review):
- SAT UNSAT: DRAT/LRAT-family certificates checked against the exact CNF/assumptions; a SAT witness is checked directly against the source formula.
- Exact #SAT / WMC: use a counting-specific proof system and a declared coefficient algebra; CPOG, MICE and kcps are research-relevant #SAT systems, not generic UNSAT proofs.
- Projected counting: carry visible-scope identity and projected-compilation/Skolem evidence; an ordinary #SAT count is not a projected count certificate.
- QBF / #QBF: preserve the quantifier prefix and Skolem dependency evidence; Q-MICE and the tree/DAG systems in TR26-279 are research targets, not currently qualified Symthaea backends.
- MaxSAT optimality: an UNSAT proof for one bound is not a complete optimum certificate unless the lower-bound derivation is also checked. MaxSAT-specific core-guided proof logging and general pseudo-Boolean proof systems such as VeriPB are distinct candidate backends.
- Approximate counting: validate the theorem/parameters, randomness source and confidence guarantee; arbitrary-precision or floating-point arithmetic alone is not an approximation guarantee.

The capability registry should only point to existing proof/checker formats and versioned adapters. It must not introduce a second proof language or treat an unverified checker assertion as a proof.

Research references:
- CPOG / MICE / kcps / CLIP strength: https://doi.org/10.1007/s10817-026-09757-w
- Projected knowledge compilation with Skolem assignments and HOL4-validated soundness: https://doi.org/10.4230/LIPIcs.SAT.2025.8
- Q-MICE for #QBF: https://eccc.weizmann.ac.il/report/2026/101/
- MaxSAT-specific proof logging for core-guided OLL: https://conf.researchr.org/details/ase-2025/ase-2025-papers/49/Efficient-and-Verifiable-Proof-Logging-for-MaxSAT-Solving
- VeriPB's pseudo-Boolean proof format and current solver ecosystem: https://veripb.org/


## 15. Quantifier-aware strategy synthesis

A quantifier-aware strategy is a stronger computational result than a Boolean decision. For a true prenex QBF, the result is a set of Skolem functions for existential variables, each depending only on preceding universal variables. For a false prenex QBF, the dual result is a set of Herbrand functions for universal variables, each depending only on preceding existential variables.

The dependency order is semantic data. Do not represent it as an unordered visible/hidden set, and do not let a generated function depend on a future opposing-player variable or another variable in its own quantifier block.

The bounded implementation workstream is #10991. It builds on the validated prefix in quantified_scope.rs and the source CNF type:
- recursive finite game semantics synthesizes the winning player's functions;
- strategy tables use explicit ordered dependency keys;
- an independent checker enumerates every opponent assignment, fills choices in prefix order and checks the original CNF matrix;
- budget exhaustion yields an error/unknown, never a TRUE/FALSE claim.

The implementation must keep QBF decision, QBF strategy extraction, proof certificates, DQBF dependencies and #QBF strategy counting as separate task/evidence types. SAT 2026 research on proof systems for QBF synthesis shows why proof-derived Skolem/Herbrand functions and intermediate game values deserve to be first-class outputs; the finite truth-table engine is only a reference oracle, not a competing scalable proof system.

Long-term strategy discovery should jointly optimize:
1. whether the quantified instance is solved;
2. the circuit/table size of the strategy;
3. dependency sparsity and circuit structure;
4. proof/strategy generation overhead;
5. independent verification cost;
6. robustness on held-out quantifier alternation depths and application families.

A smaller, independently checkable strategy is often the more useful synthesis result even when two strategies solve the same quantified instance.


## 16. Strategy-function optimization as a verifiable compilation pipeline

The strategy synthesis output should have two separate correctness levels:

- **Game certificate:** the Skolem/Herbrand functions satisfy the original CNF matrix for all assignments of the opposing player under the original ordered quantifier prefix.
- **Compiled representation certificate:** each compact BDD computes exactly the same function as the source strategy table over its declared semantic dependency scope.

Do not let the second certificate stand in for the first. A BDD can faithfully compile a function that is not a winning strategy.

The current bounded representation pipeline introduces:
- exhaustive semantic-support reduction (only inessential dependencies may be removed);
- a distinction between semantic dependency order and BDD decision order;
- a reduced ordered BDD with deterministic node IDs;
- exact order search when the small-scope permutation count fits the budget;
- deterministic sifting when it does not;
- bounded graph validation, reachable-node checks and exhaustive table equivalence.

The next research step is a structural seed order from the interaction graph of the strategy function. For a pair of input dependencies, an edge can be inferred when the second Boolean derivative is non-zero for some assignment to the remaining inputs. Deterministic min-fill/frontier-style orders can then be used as additional sifting starts. This graph is only an ordering heuristic; it is not a proof that higher-order interactions are absent.

The QBF search path now also measures partial-matrix checks and early decisive subtrees. When the already assigned prefix makes the CNF definitely TRUE or permanently FALSE, the producer can stop that subtree and fill the remaining winning-player functions with tables over the still-free dependencies. Each branch owns only rows consistent with its assigned dependency values, so sibling branches can merge without silently overwriting one another. The independent strategy checker remains the acceptance gate.

Evaluation should optimize a multi-objective tuple rather than a single benchmark score:

1. QBF decision correctness;
2. strategy extraction coverage;
3. strategy circuit/BDD size;
4. support size and dependency width;
5. proof or synthesis time;
6. construction peak memory;
7. independent checker time and simplicity;
8. robust performance on held-out quantifier patterns and application families.

A small certificate is valuable only if the original strategy is sound and the compiled object is independently checkable.


## 17. Frontier-aware interaction ordering and multi-start strategy compilation

The BDD optimizer now uses the exact pairwise Boolean interaction graph of each support-minimized strategy function. Its edges are computed from mixed Boolean derivatives over all assignments of the other support variables. This is deliberately distinct from the semantic QBF dependency scope.

A deterministic min-fill order seeds a bounded insertion refinement. Its score is lexicographic in maximum frontier, total frontier mass and the full frontier profile. Since refinement accepts only score improvements, the maximum frontier is guaranteed not to exceed the min-fill seed's maximum. Natural, reverse, min-fill, reverse min-fill, frontier-refined and reverse frontier-refined starts are deduplicated and passed through a globally budgeted sifting search.

This adds a structure-aware layer without making the structural graph a proof object. The acceptance path remains:
1. validate the source QBF strategy against the original matrix and ordered prefix;
2. remove only semantically inessential function dependencies and verify equivalence;
3. build and optimize an ROBDD under an explicit variable order;
4. verify the BDD's structural canonicality and every function row;
5. retain exact small-scope permutation search as the optimal-order oracle.

The recent frontier-aware OBDD study reports strong pathwidth/frontier results for CNF compilation. We adapt its structural principle to strategy functions, but the CNF theorem and measured gains do not transfer automatically. Our empirical question is whether pairwise strategy interaction plus frontier refinement predicts smaller verified strategy BDDs on held-out QBF application functions.

The higher-order limitation is important: pairwise mixed derivatives can reveal pairwise interactions induced by higher-order monomials, but a graph cannot encode the full Boolean interaction hypergraph. Future experiments should compare graph seeds against interaction hypergraphs, hypergraph partitioning, FORCE-like placement, clause/gate order where available, and counterexample-guided order selection.

A separate strategic direction is proof-derived synthesis. The 2026 SAT paper *Proof Systems for QBF Synthesis* provides a sound and complete QBF proof system from which Skolem/Herbrand functions and intermediate game values can be extracted in polynomial time. Symthaea's exhaustive engine should remain the finite oracle; a future adapter should target a proof system with explicit derivation/checking semantics rather than scaling recursive expansion. See https://doi.org/10.4230/LIPIcs.SAT.2026.3.


## 18. End-to-end verified strategy compilation

A safe entry point now composes the two independent semantic layers:

1. Check that the reported QBF truth value agrees with the declared winner (TRUE requires an existential Skolem strategy; FALSE requires a universal Herbrand strategy).
2. Re-run the independent winning-strategy checker against the original CNF matrix and ordered prefix. The checker recomputes opponent-assignment coverage and ignores any producer-supplied verification counters.
3. Only after source verification succeeds, minimize semantic support, compile each strategy function into an ROBDD, and independently compare every BDD against its reduced truth table.
4. Retain selected multi-start seed, interaction checks, frontier measurements, sifting evaluations, support-reduction counts and BDD node totals in the compiled bundle.

This does not make the proof format itself checked: the finite reference path still verifies strategy behavior directly. The future proof-derived adapter is tracked separately in [#11034](https://github.com/Luminous-Dynamics/symthaea/issues/11034), and must add an independent proof checker before its extraction output can be admitted.


## 19. Exact #SAT proof logging and the CPOG candidate

Exact #SAT proof evidence and QBF Skolem/Herbrand strategy evidence are separate mathematical objects.

The 2026 paper *The Relative Strength of #SAT Proof Systems* establishes that CPOG polynomially simulates both MICE and kcps, while MICE and kcps are exponentially incomparable. CLIP simulates all three but is a stronger theoretical reference. This creates a concrete, source-backed reason to evaluate CPOG as the common proof-logging target for exact #SAT rather than inventing a new counting-proof format.

The implementation boundary is tracked in #11036. Before coding:
- pin the paper's full formal CPOG rules and any public artifact;
- compare those rules with the current typed AND/OR hyperedge and exact decomposition semantics;
- demonstrate rule-by-rule mapping, not a name-based equivalence;
- independently check CPOG graph structure, partition/scope constraints, count aggregation and binding to the original CNF universe;
- preserve the distinction between #SAT, projected #SAT, weighted/algebraic counting, QBF truth, QBF strategies and #QBF strategy counting.

The CPOG simulation result is a proof-complexity relation, not an empirical runtime guarantee. Measure proof size/generation and independent checking on held-out exact #SAT families before promoting it as the practical common target.


## 17. Arithmetic coupling as an additional ROBDD ordering signal

The GF(2) interaction graph remains an exact description of non-zero second Boolean mixed derivatives. It has a known limitation as an ordering heuristic: affine XOR functions have zero pairwise GF(2) mixed derivatives even though their arithmetic (0/1)-embedded truth tables have non-zero mixed finite differences.

A separate weighted graph now defines
[
w_{ij}=\sum_{x\in\{0,1\}^{n-2}}
\left|f_{00}(x)-f_{01}(x)-f_{10}(x)+f_{11}(x)\right|.
]
This is a representation heuristic, never a support/dependency claim. It must not replace the original quantifier-derived dependency scope.

Weighted min-fill uses a deterministic lexicographic score: weighted induced fill cost, fill-edge count, weighted degree, active degree, then variable index. A newly induced fill edge receives the checked product of the two current coupling weights through the eliminated variable. Its weight is propagated to later elimination steps. Arithmetic overflow or context-budget exhaustion is an error; it cannot silently produce a partial matrix.

Multi-start ordering now preserves both GF(2) and arithmetic-weighted seeds, their reverse orders and frontier-refined variants. Each unique seed shares the same global BDD-order evaluation budget. The best candidate is selected only by actual ROBDD node count and deterministic tie-breaks, then exhaustively checked against the exact reduced truth table.

### Finite evidence and limitation

- Exhaustive oracle: all 256 three-input truth tables are checked against an independently written integer mixed-difference calculation.
- XOR3: pairwise GF(2) interaction graph is empty; arithmetic pair weights are 4, 4, 4.
- Four-variable witness: natural and GF(2) min-fill produce 7 BDD nodes; arithmetic weighted min-fill order ([2,3,0,1]) produces 6; exact permutation search finds 5; multi-start sifting reaches 5.

This is a finite known-answer demonstration, not evidence that the weighted heuristic wins on average. The next experiment is a preregistered ablation on held-out generated and application-derived strategy functions. Record source support, both interaction signals, candidate seeds, total search budget, peak construction size, final node count and checker cost. Include functions where weighted ordering is worse than the baseline, and require the multi-start selector to preserve the best already-verified candidate under budget exhaustion.


## 18. External QBF comparison: pin the output contract, not just the solver name

The first competitive baseline is CAQE, which SAT 2026 proof-theoretic work characterizes as the most successful practice-wise QBF algorithmic paradigm. Keep that description separate from any specific leaderboard score or build/version result.

CAQE's public interface exposes prenex-CNF/QDIMACS decision solving, a `--qdo` partial-assignment output mode, and a separate `dcaqe` executable for DQBF/DQDIMACS. This leads to a required adapter distinction:

- a QBF truth decision is not a full Skolem/Herbrand strategy;
- a partial assignment is not a total strategy function;
- a proof object is not a checked proof until a compatible independent checker accepts it;
- a proof is not an extracted strategy until a supported extractor returns functions and an independent strategy checker validates them against the original matrix/prefix;
- DQBF is not prenex QBF and must retain its explicit dependency semantics.

For each external run, bind solver version/configuration, input digest, track/format, command line, resource envelope, output capability class and checker identity. Separate decision, proof, extraction and independent validation times. This avoids overstating the external baseline and yields a fair time-to-verified-strategy metric.

The long-term production route should study the SAT 2026 synthesis proof system as an independent extraction adapter. Its reported contribution is a sound/complete proof system with polynomial-time extraction of Skolem/Herbrand functions and intermediate game values for proofs in that system; that theoretical claim does not automatically transfer to arbitrary existing QBF proof formats.


## 19. Representation portfolio: ROBDDs and AND-Inverter Graphs

A Boolean strategy function should be allowed to have multiple compiled representations. The bounded compiler now includes:
- an ROBDD path with support reduction, exact small-scope order search, bounded sifting and graph-derived seed orders;
- an AIG path with Shannon decomposition, complemented edges, two-input AND nodes, structural hashing and caller-selected decomposition order.

These representations have different strengths and their node counts are not directly comparable. A BDD's canonicality is conditional on variable order; an AIG is a structurally hashed Boolean circuit but is not globally canonical. AIG construction order can affect sharing and node count.

Keep two independent checks:
1. **Function equivalence:** the candidate graph evaluates to the exact source strategy table for every assignment to the declared support.
2. **Game validity:** the original Skolem/Herbrand function is independently checked against the original CNF matrix and ordered quantifier prefix for every opposing-player assignment.

The graph checker cannot stand in for the game checker. A compact AIG that computes an invalid strategy is still invalid.

The next compilation selection step should use one pinned serialization and a multi-objective cost: serialized bytes, compile peak memory, construction work, evaluation time, independent checker time/memory, and support width. Do not minimize raw node count across representation types. Test candidate BDD decision orders and AIG Shannon decomposition orders under a shared aggregate compute budget, with explicit incomplete-search metadata.


## 20. Portable game-proof artifacts and three-layer verification

The bounded QBF implementation now exposes a proof trace in addition to Skolem/Herbrand strategies. Its finite rules are:
- matrix leaf: evaluate the original CNF under a complete prefix assignment;
- early TRUE: every source clause has an already assigned satisfying literal;
- early FALSE: a cited clause is fully assigned and all of its literals are false;
- EXISTS decision: both children are checked and the parent value is their disjunction;
- FORALL decision: both children are checked and the parent value is their conjunction.

The verifier rebuilds the variable/quantifier order from the source `ValidatedQuantifierPrefix`, reconstructs each branch assignment, rejects non-postorder or unreachable tree structure, and independently evaluates each source-matrix witness. It does not trust stored intermediate node values.

The canonical `LD-QBFPROOF1` format is now a portable byte artifact. Its bounded decoder rejects invalid magic, truncated integers/nodes, invalid Boolean bytes, unknown node tags, out-of-budget variable/node counts, invalid tree edges and trailing bytes. The evidence-portfolio builder round-trips the in-memory proof through the encoder/decoder and re-verifies the decoded artifact against the original matrix and prefix before returning it.

The new `verify_and_compile_qbf_evidence_portfolio` makes the following trust boundaries mandatory and ordered:
1. independently verify the game proof against source CNF and ordered prefix;
2. require the proof root truth value to agree with the claimed QBF result;
3. serialize, decode and independently reverify the proof bytes;
4. independently verify the Skolem/Herbrand strategy against every opponent assignment;
5. compile support-minimized ROBDD and AIG circuits;
6. independently verify both graph representations against their source Boolean functions;
7. enforce the aggregate byte budget across game proof, BDD and AIG.

This deliberately separates three claims:
- **game validity:** the quantified formula is TRUE or FALSE;
- **strategy validity:** the extracted function family is winning for the appropriate player;
- **representation validity:** each graph computes exactly the corresponding source strategy function.

The current proof is a tree, not a DAG. A DAG is a future optimization only after a cache key captures the complete residual formula, assigned-prefix context and rule obligations. Naive structural hash-consing can merge the same-looking subproof under different contexts and is not acceptable.

For proof-system comparisons, include QBFcert/DepQBF syntactic QRP proof checking and extraction where available, CAQE partial QDIMACS outputs as a partial-assignment interface (never assumed complete), and the SAT 2026 synthesis proof system as a theoretical target for intermediate game values and extracted strategies. Keep #QBF counting separate from all these decision/strategy-evaluation claims.


## 21. Algebraic hypergraph ordering and workload-aware representation portfolios

The ROBDD ordering system should no longer rely on a single pairwise graph. It has three complementary structural signals:
- GF(2) mixed derivatives;
- arithmetic weighted mixed finite differences;
- exact ANF monomial supports obtained from the Boolean Möbius transform.

The ANF support hypergraph preserves higher-order monomials. It contributes deterministic min-fill and reverse seed orders to the same bounded multi-start ordering budget. All hypergraph vertices are local positions in the validated function dependency scope. These graph/hypergraph signals are heuristics only; the semantic support reduction and prefix-legal dependency list remain authoritative.

An exhaustive three-input function corpus checks ANF reconstruction, support reduction and ROBDD equivalence. This is a finite correctness base, not evidence that an ANF hypergraph heuristic beats SOTA at larger scale. A promotion experiment needs held-out generated functions and QBF strategies, exact order oracles on small scopes, and a full ablation of natural/reverse, GF(2), arithmetic, frontier and ANF starts.

### Representation portfolio is multi-objective

A verified strategy function is compiled into both a canonical ROBDD and a canonical AIG. The portfolio records:
- encoded byte size;
- internal decision/gate nodes;
- logic depth;
- worst-case evaluation work (the ROBDD follows one decision path; the current AIG evaluator traverses its complete straight-line node list);
- a deterministic structural estimate of independent verification work.

The nondominated set is preserved. The default chooses the smallest byte-size candidate within that frontier, but the API exposes both candidates for workload-specific storage/latency/checker decisions. The estimate is not wall-clock timing, and no target-hardware performance claim should be based on it.

Next, benchmark each candidate on the same pinned CPU and memory envelope:
1. compile time and peak memory;
2. cold and warm evaluation latency across the full strategy input distribution;
3. checker time/memory over all assignments (bounded finite scope) and on larger external proof-check workloads;
4. encoded size and transfer size;
5. energy if a stable hardware measurement is available.

### The fair QBF baseline rule

QBFGallery's documented 2026 command-line protocol returns exit 10 for TRUE and 20 for FALSE. This does not guarantee a complete winning function family or proof certificate. Compare decision-only CAQE/DepQBF/Qute runs separately from proof-logging and strategy-extraction modes. CAQE partial-assignment output must not be treated as a complete strategy without scope/completeness and source-matrix checks.

The 2026 Gallery's PCNF, crafted, random, 2QBF, prenex non-CNF and DQBF tracks are distinct. DQBF cannot be collapsed into the prenex dependency validator. The 2023 results can serve as a historical, track-specific calibration point, but current 2026 numerical SOTA must wait until the exact public benchmark and score artifacts are pinned.

The long-term scientific breakthrough remains proof-derived synthesis: the SAT 2026 synthesis proof system proposes extracting Skolem/Herbrand functions and intermediate game values in polynomial time from proofs in that system. The finite recursive engine is the oracle for small cases; the research workstream must study and implement an exact published rule subset, then compare proof-derived extraction against the finite oracle without letting the proof producer certify itself.


## 17. Residual-state proof DAGs and #QBF research boundary

The current proof tree deliberately rejects multiple parents and therefore cannot share equal contextual subproofs. Workstream #11130 adds a different proof representation, not a mutation of the trusted tree:

- each node is bound to an exact prefix position and canonical residual CNF;
- the canonicalizer removes satisfied clauses, assigned-false literals, duplicate literals/clauses and tautologies, while preserving empty clauses;
- identical `(position, residual)` pairs are interned into one node;
- the independent verifier recomputes both child residuals from the parent and the next quantified variable, checks the local OR/AND recurrence and validates the entire reachable DAG;
- multiple parents and identical low/high child references are valid and expected.

This is a direct research experiment in light of Chede, Krishan & Shukla, *On Knowledge Compilation Languages for QBFs*, ECCC TR26-279 (published 27 October 2026). That work already studies tree-based and DAG-based knowledge-compilation languages for QBF, including proof-system consequences for #QBF and an initial compiled #QBF solver compared against d4-QBF. Therefore the novelty claim is limited to our specific residual-CNF state identity, proof/checker contract, and its integration with independently verified Skolem/Herbrand strategy plus BDD/AIG compilation. General QBF tree-to-DAG compilation is not claimed as new.

The same paper makes a separate frontier visible: **#QBF counts winning strategies**, and should have a distinct task type. The initial Symthaea #QBF oracle should enumerate all bounded Skolem strategy profiles and independently count those that satisfy the source matrix for every universal assignment. It must not confuse the number of satisfying full assignments with the number of distinct winning functions, and it must not be folded into QBF truth or #SAT.

The research sequence is:
1. qualify the bounded residual-DAG producer and checker;
2. measure tree-vs-DAG reduction on exhaustive small CNF and held-out generated QBFs;
3. integrate verified DAG evidence into the existing strategy portfolio without duplicating formats;
4. build a distinct finite #QBF reference/counting lane;
5. only then compare with QBF Gallery benchmarks and #QBF knowledge-compilation / d4-QBF baselines.

## 18. Cross-representation QBF evidence portfolio

The strategy portfolio now has a second complete evidence path:

- **Tree proof path:** existing QbfGameProof, source-bound tree verifier and canonical LD-QBFPROOF1 encoding.
- **Residual DAG path:** QbfGameProofDag, source-bound independent residual verifier and canonical LD-QBFDAG1 encoding.
- **Strategy path:** independently checked Skolem/Herbrand functions.
- **Circuit path:** support-minimized ROBDD and AIG per strategy function, independently checked and canonically encoded.

The new DAG bundle builder verifies the residual proof DAG, checks its root value against the candidate, encodes and decodes the DAG, reruns the source-bound verifier on the decoded object, independently verifies the candidate strategy, compiles the BDD/AIG portfolio, and accounts the total evidence bytes. The tree and DAG evidence bundles remain distinct types; the portfolio can compare their size, verification work and construction costs without pretending they share a proof calculus.

The bounded #QBF reference counter is also implemented separately from decision and #SAT. It enumerates total Skolem strategy profiles over exact prefix-derived dependency tables and counts a profile once only if it wins for every universal assignment. It has known answers and differential exact-count oracles on generated small instances. It is not a scalable #QBF proof system.

The official SAT Competition 2026 results have since been corrected. The current official Main Track table reports 306/400 solved by the best individual listed configuration (kissat-sc2024) and 347/400 by the per-instance VBS; the website notes an August 10, 2026 update that changed a ranking. Historical local recomputations (276/269 and derived 324/344 VBS) are not the official table and must be reconciled against the updated scores.csv before any claim that uses them.


## 17. Full-history game values and Skolem/Herbrand synthesis

A winning strategy and a full-history Skolem/Herbrand function are not interchangeable. The winning strategy only needs to secure the initial game. Full-history functions must make the right move after every partial play, even one not reached when both players follow a winning strategy.

Following Akshay et al., SAT 2026, *Proof Systems for QBF Synthesis: Extracting Skolem and Herbrand Functions* (https://doi.org/10.4230/LIPIcs.SAT.2026.3), define (g_i(x_1,\ldots,x_i)) as the value of the remaining quantified suffix after the first i moves. Then:
- (g_n) is the source matrix;
- (g_{i-1} = g_i|_{x_i=0} \lor g_i|_{x_i=1}) for existential (x_i);
- (g_{i-1} = g_i|_{x_i=0} \land g_i|_{x_i=1}) for universal (x_i);
- the full-history Skolem function is (f_i(x_{<i}) = g_i(x_{<i},1)) for existential (x_i);
- the full-history Herbrand function is (f_i(x_{<i}) = g_i(x_{<i},0)) for universal (x_i).

Crucially, (f_i)'s dependency scope is every earlier prefix variable, including earlier variables quantified by the same player. This is deliberately different from the opponent-only dependency scope of the existing initial winning-strategy contract.

The new finite module `quantified_game_values.rs` extracts complete (g_0,\ldots,g_n) truth tables by traversing the already independently verified residual game DAG. It returns both full-history function families even if the root formula is true or false. An independent verifier rechecks the final matrix layer, all quantifier recurrences, all function scopes/rows and the root value.

This is a semantic profile, not yet a SynQBF proof. The SAT 2026 proof system uses a circuit (hat\varphi(x_i,\hat x_i)) subject to three explicit conditions (source-matrix equivalence, per-variable representability after the paper's prescribed substitutions, and truth-value agreement). Its CEGC constructor uses an AllSATRep routine to represent all satisfying assignments; the paper leaves implementation of AllSATRep outside its scope. Symthaea must implement and validate that subroutine separately before claiming a SynQBF proof constructor.

### Qualification boundary

- Existing `WinningStrategy`: one initial winning player's strategy, opponent-only dependencies.
- New `QuantifiedGameValueProfile`: every intermediate game value, plus both full-history function families, all-prior-prefix dependencies.
- Future SynQBF proof object: a circuit proof meeting the paper's three representability/equivalence conditions, independently checked from the source and prefix.

These are related but different evidence artifacts. A full-history profile does not become a SynQBF proof by renaming it.


## 17. Portable SynQBF artifacts and full-chain acceptance

The finite pipeline now has a deterministic versioned portable artifact for SynQbfProof, backed by the existing AIG representation. The v1 codec uses explicit lengths and tags, little-endian integer fields, canonical input/decomposition order and strict malformed/trailing-byte rejection.

The trust model is deliberately two-stage:

1. Decode checks byte/count budgets, prefix binding shape, AIG graph validity, and canonical input/decomposition order. It establishes only structural validity.
2. The source-bound artifact verifier decodes and calls the independent SynQBF verifier with the original CNF and validated ordered quantifier prefix. Source equivalence, every representability condition and the claimed root value must all pass.

The format does not embed the source CNF, make a producer-provided verification flag authoritative, or claim cryptographic authenticity. Source and prefix digests are recorded by the existing evidence manifest/receipt system.

The intended vertical acceptance chain is:

QDIMACS source -> parsed ordered prefix + CNF -> residual game proof DAG -> independently verified game-value profile -> SynQBF AIG proof -> deterministic artifact bytes -> decode -> source-bound SynQBF verification

The focused finite corpus now includes the full 2,184 two-variable CNF/prefix construction combinations, empty-prefix constant TRUE/FALSE cases and a QDIMACS-to-portable-artifact example. Until exact-head CI passes, these are committed tests, not qualified results.

The decoder preflights the minimum bytes needed for the declared AIG node count before reserving storage. This is separate from the absolute artifact-byte and node-count limits; a small malformed artifact must not induce a large node-vector allocation.

For external evaluation, follow QBFGallery's documented exit-code contract: 10=TRUE, 20=FALSE, any other status=error/unknown. Keep its decision-only execution separate from proof/artifact generation and verified strategy coverage. The current PCNF parser supports QDIMACS only; QCIR, prenex non-CNF and DQBF require separate typed input semantics and must not silently fall through to CNF parsing.
