# ProofPlanV1 — global obligation accounting v1

**Issue:** [#13641 — PROOF-OBLIGATION-ACCOUNTING-001](https://github.com/Luminous-Dynamics/symthaea/issues/13641)  
**Parent:** [#13293 — MATH-COMPREHENSION-001](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Base fixture:** [PROOF_PLAN_AND_COMM_V1.json](PROOF_PLAN_AND_COMM_V1.json)  
**Status:** Rust implementation and mutation tests authored; 11-case expected-behavior corpus committed. Rust compilation and tests have not been executed in this environment.

## 1. The failure mode: local well-formedness is not global completeness

A proof-plan graph can have valid node IDs, evidence references, goal-action edges and justifications yet still fail to account for the whole argument.

Examples:
- a split action creates two obligations, but only one is later discharged;
- two independent actions both claim to discharge the same goal;
- an action's declared output list differs from its actual creates-subgoal edges;
- a goal is labelled transformed, but no successor obligation is created;
- a cycle makes goals recursively depend on one another without a terminal discharge;
- an accepted plan contains an unresolved leaf;
- a candidate proof route is mistaken for the route actually being qualified.

The previous local checks are necessary, but they are not sufficient. Global obligation accounting checks whether the plan represents a complete path from its root to explicit terminal discharges.

## 2. Paradigm shift: actions are goal-consuming hyperedges

Treat a proof action as a hyperedge with:
- one or more input goal states;
- a disposition for each input: `discharges` or `transforms`;
- zero or more output subgoals;
- justifications, consumed side conditions and evidence references.

For each declared alternative, the accounting pass establishes:
1. The root goal is the alternative's root.
2. Every subgoal has exactly one producing action.
3. Every goal/subgoal is consumed exactly once by an action, or remains explicitly unresolved.
4. The action input list agrees with its discharge/transform edges.
5. Every action's output-subgoal list matches its creates-subgoal edges exactly.
6. Every transformed goal creates at least one successor subgoal.
7. Every transition path is acyclic.
8. Terminal discharged goals and unresolved goals are counted separately.

A goal that is not discharged or transformed is an unresolved obligation, not an implicit success.

### Why the hyperedge model matters

A single action can consume multiple input goals and create several output obligations. This represents proof steps such as combining premises, introducing cases or constructing a witness with multiple side conditions.

The current v1 accounting model records each input's disposition and conservatively connects every transformed input to the action's declared successor subgoals. More precise provenance can later add per-input/per-output lineage identifiers when proof extraction provides them.

## 3. Alternative routes are separate proof objects

ProofPlanV1 already supports multiple alternatives. Global accounting respects that.

- Only the explicitly selected primary alternative can qualify an accepted plan.
- A candidate alternative can remain incomplete and report unresolved obligations.
- A rejected alternative can be retained for provenance.
- An accepted plan requires exactly one primary alternative.
- That primary alternative's root must match ProofPlanV1.root_goal_id.
- Candidate alternatives cannot be silently promoted by a complete-looking union of nodes from several routes.

This avoids another subtle failure: combining the forward half of one route with the reverse half of another and reporting a complete proof that no single alternative actually contains.

## 4. State semantics

### Provisional / in-progress

The validator may return a structurally valid plan with unresolved goals. The CLI reports:
- produced subgoals;
- transformed goals;
- discharged/terminal goals;
- unresolved goal IDs;
- goal-transition cycles;
- primary alternative count.

An incomplete but well-formed research plan remains useful. It is not called complete.

### Accepted

An accepted plan must have:
- exactly one primary alternative;
- root identity matching the top-level ProofPlan root;
- every subgoal produced exactly once;
- every goal/subgoal consumed exactly once;
- zero unresolved obligations on the primary route;
- at least one terminal discharged goal;
- no transition cycle;
- all previously required graph, kernel, evidence and independent-review gates.

This is still a plan-accounting result. A discharge edge is not a kernel certificate. The exact local proposition and context still need their own proof artifact and replay evidence.

## 5. Implementation

Added to the existing Rust crate:

- `src/global_obligations.rs` — global obligation analysis, per-alternative reports and cycle detection.
- `src/lib.rs` — integrates the analyzer into ProofPlanV1 validation and exports the module.
- `src/main.rs` — CLI reports global accounting status and counts instead of only printing structural validity.
- `docs/research/PROOF_PLAN_GLOBAL_OBLIGATION_MUTATIONS_V1.json` — 11 frozen controls.
- `docs/research/PROOF_PLAN_GLOBAL_OBLIGATION_MUTATIONS_V1.schema.json` — strict fixture schema.

### Mutation corpus

| Fixture | Expected result | Failure mode |
|---|---|---|
| OBL-001 | Complete | And.comm two-direction positive control |
| OBL-002 | Incomplete | Missing discharge in provisional plan |
| OBL-003 | Reject | Double discharge |
| OBL-004 | Reject | Subgoal with no producer |
| OBL-005 | Reject | Orphan create-subgoal edge |
| OBL-006 | Reject | Declared output without create edge |
| OBL-007 | Reject | Transformed goal without successor |
| OBL-008 | Reject | Accepted plan with unresolved leaf |
| OBL-009 | Reject | Goal-transition cycle |
| OBL-010 | Primary complete; candidate incomplete | Alternative isolation |
| OBL-011 | Reject | Multiple primary alternatives |

Unit tests in the Rust module exercise the same core cases, including cycle detection and the distinction between provisional incompleteness and accepted-plan rejection.

## 6. ECPCB integration and metrics

ECPCB should evaluate global proof-plan accounting separately from local action certificate validity.

Report:
- subgoal producer precision/recall;
- goal-consumer accounting accuracy;
- double-discharge false acceptance;
- orphan-output detection;
- transformed-without-successor detection;
- unresolved-obligation rate;
- cycle detection;
- primary-route isolation;
- candidate-route contamination rate;
- accepted-plan false completeness.

Prospective hard gates:
- 100% of generated subgoals have exactly one producer;
- 100% of input goals have one explicit disposition on the primary route;
- zero accepted plans with unresolved goals;
- zero accepted primary routes with cycles;
- zero accepted plans with multiple primary alternatives;
- candidate-route incompleteness never changes primary-route status.

These are targets, not measured results. A zero failure count on a small test corpus is not proof of universal correctness.

## 7. Remaining limitations and next steps

1. Rust compilation and the existing ProofPlanV1 tests must be executed before this module is merged.
2. Add per-input/per-output lineage IDs so transformations with multiple inputs and outputs can identify the precise obligation mapping rather than the conservative hyperedge relation.
3. Add certificate-aware terminal nodes: the obligation pass should bind each terminal discharge to its exact local certificate ID, not merely a discharge edge.
4. Add independent plan adjudication and alternate-route tests across more than And.comm.
5. Add property-based generation of well-formed plan hypergraphs and mutations, then compare the Rust accounting report against a small reference oracle.
6. Extend ECPCB's gold corpus to include at least one theorem with nested case splits, one with induction, one witness-construction proof and one local-to-global argument.

## 8. Claim ceiling

Global obligation accounting verifies the internal completeness of a represented proof plan within a declared alternative. It does not establish:
- that a goal statement is mathematically true;
- that an action is a valid inference;
- that a discharge has a valid Lean proof;
- that the proof plan accurately reconstructs the source proof;
- that expert role annotations are correct;
- that a human learner understood the explanation.

Those remain separate formal, source, review and human-evaluation obligations.
