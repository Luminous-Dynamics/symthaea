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
