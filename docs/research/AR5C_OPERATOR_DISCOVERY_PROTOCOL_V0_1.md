# AR-5C — typed primitive invention and verifier-qualified grammar expansion v0.1

**Status:** research protocol / implementation plan; no operator invention or qualification result is claimed.  
**Tracking issue:** [#9319](https://github.com/Luminous-Dynamics/symthaea/issues/9319)  
**Parent programme:** [#9197](https://github.com/Luminous-Dynamics/symthaea/issues/9197)  
**Related:** #1318, #8536, #8705, #9230, #9235.

## Why this is a distinct research layer

Architecture search over a fixed operator set can find compositions, parameterizations and runtime schedules, but cannot discover a primitive outside the grammar. AR-5C makes grammar expansion an explicit research operation. It is separate from candidate scoring and separate from granting any production system new authority.

The hypothesis is not that arbitrary generated code is a new paradigm. A genuine candidate must provide a typed mechanism with semantics, falsifiable invariants, reproducible implementation, independently checkable behavior and held-out capability transfer beyond the strongest composition of the existing grammar.

## Three nested discovery spaces

1. **Composition:** search arrangements of existing operators.
2. **Regime:** search when/how operators run, including timescales, gating, memory, plasticity, compute allocation and switching policies.
3. **Grammar expansion:** propose a new representation, operator, update rule or information-flow primitive.

Results must state which space was searched. A composition win must not be described as discovery of a new primitive.

## Typed operator proposal

The authoritative proposal should be a typed AST / operator IR containing:

- parent grammar digest and schema version;
- canonical operator identity and typed input/output ports;
- units, shapes, domains, codomains and arity;
- preconditions, postconditions and formal or empirical invariants;
- complete semantics or an explicit partial-domain contract;
- reference implementation digest and candidate implementation digest;
- predicted capability delta and falsifiable counterexamples;
- complexity/resource model and declared unknowns;
- lineage, evaluator profile and verifier receipt.

Natural-language rationale can explain the candidate, but it is not the authoritative executable semantics.

## Verification sequence

| Gate | Evidence | Failure behavior |
|---|---|---|
| Type/schema | Valid AST, ports, dimensions, units and arity | Reject |
| Semantic definition | Total semantics or explicit preconditions | Reject/inconclusive |
| Invariants | Property and metamorphic tests | Reject on counterexample |
| Differential | Independent reference/implementation agreement | Reject |
| Adversarial | Boundary, fuzz, numerical and resource tests | Reject/inconclusive |
| Proof | SMT or other proof where decidable; assumptions labeled | No proof claim when infeasible |
| Held-out transfer | Hidden tasks/interventions against best existing composition | No admission if no transfer |
| Resource | Wall time, memory, operations, energy and scaling | Reject bound violations |
| Reproducibility | Replay pinned input, grammar, evaluator and build | No qualification without replay |

The verifier—not the proposer—returns one of Equivalent(existing), Novel, Rejected, or Inconclusive. Inconclusive is never counted as novel. Failed proposals and all evaluation costs remain in the denominators.

## Semantic novelty is not descriptor novelty

Case folding and string hashing are not enough. The verifier should combine canonical typed signatures, normalized semantics, equality saturation/e-graph rewriting where appropriate, exhaustive small-domain I/O comparison, metamorphic invariants and counterexample-guided refinement.

Finite-sample behavioral similarity is not proof of semantic equivalence. The receipt must say whether equivalence is proven, exhaustively checked over a bounded domain, or merely empirically supported.

A source refactor, prompt change, random seed, renamed operator, or reordered composition must not create a new direction/primitive identity.

## First experimental domains

Start where exhaustive small-domain ground truth is practical:

- finite-state transition systems and automata;
- temporal/state update operators;
- compositional sequence and memory mechanisms;
- causal intervention and invariant-mechanism operators;
- algebraic and relational reasoning operators;
- resource-bounded planning/control abstractions.

Each family should contain equivalent re-expressions, efficient equivalent implementations, invalid-but-plausible invariants, training-only operators, unidentifiable cases, and true novel primitives with held-out transfer.

## Baselines

Every candidate is compared against exhaustive or budget-matched composition search on the frozen existing grammar; typed-AST regularized evolution; grammar-guided synthesis; counterexample-guided synthesis where applicable; equality saturation for composition/equivalence; best human-authored primitives; simple classical controls; and proposal-generator baselines such as ShinkaEvolve, CodeEvolve or DeltaEvolve under the same verifier, model and resource budget.

No advantage is attributable to invention if the baseline had fewer candidate evaluations, weaker primitives, less compute, different prompts or less access to the evaluator.

## Qualification metrics

Primary: qualified novel primitives per total cost, novelty precision, false-admission rate, held-out capability lift over the strongest existing-grammar composition, cost to first verified primitive, worst-family transfer, and resource Pareto frontier.

Secondary: proof/property pass rates, counterexample yield, duplicate rate, reproducibility, robustness to task shift and cost-normalized proposal yield.

These should remain separate metrics; semantic validity and safety gates are non-compensatory.

## Proposed targets — not achieved

- Zero false admissions on exhaustive finite-domain equivalence controls.
- At least 95% precision on a blinded semantic-novelty challenge set, after independent verifier calibration.
- Exact reconciliation of proposal, IR, reference implementation, candidate implementation, verifier profile and evidence receipts.
- Held-out improvement on at least two independent task families, or a clearly specified capability that the existing grammar cannot express.
- No resource-bound violation or catastrophic regression on the challenge families.
- Deterministic replay from pinned grammar, candidate, evaluator, verifier and environment inputs.

These are candidate qualification gates to be calibrated and preregistered, not claims about the present system.

## Authority boundary

The search process runs in an isolated sandbox. It has no ability to modify production cognition, merge code, deploy candidate operators, or change acceptance thresholds. A successful qualification produces a research-grammar proposal and evidence bundle. A human-reviewed, versioned grammar change and a separate production qualification are still required.

## Connection to AR-5B and AR-5B1

AR-5B allocates research effort across verifier-defined directions. AR-5B1 tests whether that allocation improves discovery on synthetic delayed-payoff and grammar-ceiling landscapes. AR-5C supplies the missing object of paradigm discovery: a typed operator proposal with independently checkable semantics and transfer evidence.

The complete loop is therefore:

direction proposal → typed operator IR → independent novelty/equivalence check → property/proof/intervention tests → held-out transfer → cost/resource receipt → research grammar proposal

The claim ceiling remains bounded: passing these gates supports operator discovery in the tested domains, not universal scientific discovery, AGI, consciousness or general paradigm invention.
