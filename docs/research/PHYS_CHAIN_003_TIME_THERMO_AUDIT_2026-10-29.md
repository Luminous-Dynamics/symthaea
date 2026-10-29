# PHYS-CHAIN-003 — Time and thermodynamics audit

**Status:** Source-level audit and benchmark protocol. No independent reconstruction of the compendium's transition graphs or its Lean artifacts is claimed complete.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8941

## Primary source and locations

A. D. Wissner-Gross, *Chain Theory: Toward Fundamental Physics from Distilling Networks of Entangled Qudits — A Compendium of Preliminary Findings*, working draft dated 23 October 2026:
https://alexwg.org/publications/ChainTheoryCompendium.pdf

Relevant printed pages in the 754-page draft:
- Abstract / overall standing counts: pp. 19–20.
- Order coordinate, Theorem 50.1: p. 135.
- Thermodynamics overview: p. 537.
- Boundary-count second law, Theorems 217.1–217.2: p. 539.
- Thermal state and ensembles / open results: pp. 540–543.
- Thermodynamic remainder and arrow, Open Result 223.1 and Theorem 223.2: pp. 551–552.

## Source-level findings (not yet independently reproduced)

### 1. The theory's order coordinate is a count, not a flow

Theorem 50.1 says the order coordinate is the count of discrete one-steps of an order register, “a count and never a flow.” It increments at specified construction-arrow steps and at read-depth steps, but not at census moves. The construction's arrow operator is described as acyclic on the 64 catalogued networks. The result is conditional on the axiom and named prior results; the same passage states that the depth step is “posed and derived by no result.”

The compendium abstract separately lists parity, time reversal, and CPT among conditional theorems. That is logically distinct from an arrow of thermodynamic time. Microscopic time-reversal symmetry, the partial order of a chosen construction, and macroscopic irreversible behavior must not be conflated.

### 2. Two count functions are used in the thermodynamics section

The compendium distinguishes:
- **Boundary count:** the number of records across a cut/interface, interpreted as the entanglement entropy across a cut. Theorem 217.1 states this count is non-decreasing under every specified one-act read; the reasoning is combinatorial: a read appends a record and removes none.
- **Register count W:** a count over the whole common support, used as an order parameter and later named entropy.

Theorem 217.1 is called the second law at the counting face. However, the text says the correspondence to the conventional isolated-system statement ΔS ≥ 0 is “by name only.” Open Result 217.3 records that the boundary count required by some downstream results does not consistently coincide with the proposed site/free-direction bounds.

### 3. Thermal-state, equilibrium, and irreversible-thermodynamic correspondence is incomplete in this draft

- The thermal state as a twirl fixed point at a cap is Open Result 218.1; the bath-network correspondence is Open Result 218.2.
- The text says the thermodynamic limit is named at a cap rather than obtained as an actual limit.
- The summary table lists the equilibrium theorem as awaited.
- Open Result 223.1 reports failures and scope dependence: the boundary count and W diverge at the first wave in the tested closure; the proposed fluctuation relation is defined at only 16 of 111 move/inverse pairs in one comparison; and some Onsager readouts are asymmetric.
- A second reading of the graph without caps has W decreasing on many edges labelled forward. Thus universal W-monotonicity is not stable across the draft's own graph/stop-condition variants.
- Theorem 223.2 reports strictly increasing W along every forward edge in one large truncated closure. But its derivation inherits open results, and the text says the physical concepts it names—including entropy production, Crooks/Jarzynski fluctuation relations, and Onsager reciprocity—are correspondences “by name only.”

These statements reflect the current working draft's own theorem/open-result labels. They do not constitute an independent verification of the counts or a blanket derivation of standard thermodynamics.

## Research question

Can the discrete ordering/counting structure derived from Chain Theory be mapped to physical time, thermodynamic entropy and experimentally observed irreversibility without putting the desired physical meaning into the definitions?

The audit separates five objects:
1. **Construction order:** topological order of allowed moves.
2. **Physical time:** an operational clock or proper time, with the relevant Lorentz-covariant behavior.
3. **Microscopic time reversal:** which transitions/operators are mapped under time reversal, including valid inverse moves.
4. **Thermodynamic entropy:** a physical state function with a stated coarse-graining or density-matrix mapping.
5. **Thermodynamic arrow:** statistics of irreversible behavior, including fluctuation and response relations under specified conditions.

An acyclic directed graph establishes a construction order; by itself it does not establish a physical time coordinate or the second law for a physical system.

## Bounded work plan

### WP1 — Reproduce the order-coordinate theorem

- Pin the exact compendium source and associated code/artifacts.
- Independently reconstruct the bounded list of networks and permitted construction moves.
- Check acyclicity, step-count semantics, the four-coordinate count lattice, and move exclusions that should not increment the order coordinate.
- Inventory all premises and retain their actual theorem / conditional / open status.
- Treat the un-derived read-depth step as an explicit gap until proved from declared semantics or classified as an additional assumption.

### WP2 — Separate the three arrows

Create paired fixtures separating:
- construction-order orientation;
- reversible microscopic dynamics with inverse transitions;
- irreversible dynamics with a physically defined coarse-grained entropy.

Where possible, hold the state space fixed while changing move orientation or transition probabilities. The test must demonstrate that an algorithm cannot infer thermodynamic irreversibility from DAG orientation alone.

### WP3 — Reconstruct both counts and check edgewise claims

- Independently compute boundary-count and register-count W from their declared definitions.
- Reproduce every reported state, edge, forward/reverse split, and violating edge for each closure and stop condition.
- Keep all edges in the denominator; don't hide pointwise violations behind aggregate averages.
- Separate exact graph counting from the claim that the resulting quantity is a physical entropy.

### WP4 — Compare against explicit thermodynamic contracts

The reference definitions must be concrete, not just labels:
- von Neumann entropy / entanglement entropy with an explicitly constructed density matrix and subsystem;
- Gibbs state and partition function for a stated Hamiltonian and temperature;
- equilibrium and thermodynamic-limit behavior under a specified limiting sequence;
- entropy production for a stated stochastic/dynamical process;
- Crooks/Jarzynski fluctuation relations with a well-defined ensemble and forward/reverse trajectory measures;
- Onsager reciprocity only under its stated near-equilibrium and microscopic-reversibility assumptions.

For every alleged correspondence, specify the mapping from Chain Theory states/moves to physical states/transitions/observables, units, initial ensemble, boundary conditions, and regime of validity. If required quantities are absent, mark the correspondence incomplete rather than infer them by matching names.

### WP5 — Robustness and mutation matrix

Cross:
- closure/truncation and stop condition;
- forward-only and forward/reverse moves;
- boundary count versus W;
- exact enumeration and independent implementation;
- boundary/site/free-direction definitions;
- semantics-preserving basis or node enumeration changes;
- semantics-changing edits to the move table, record rule, orientation, or entropy definition.

Report the state and edge denominator for each cell; each monotonicity violation; the exact function tested; and whether the assertion remains proved, conditional, or open.

### WP6 — Prospective physical prediction

- Identify how a construction step would correspond to an operational clock reading or proper-time interval.
- Identify a physical model whose entropy can be calculated independently of the Chain Theory bookkeeping.
- Derive at least one quantitative prediction for a time/entropy/fluctuation observable that differs from a standard comparator, with uncertainty and a rejection criterion fixed before seeing new outcomes.
- If the present draft produces no independent discriminator, state that as the result. Do not invent a prediction to satisfy the gate.

## Metrics and acceptance gates

1. Exact agreement with independently generated tiny-graph and known-answer controls.
2. Every asserted monotonicity property tested edge-by-edge on the full enumerated graph.
3. Complete accounting of reverse edges, truncation, open results, and failed cells.
4. Correct separation of construction order, microscopic time reversal, physical clock time, and thermodynamic irreversibility.
5. No mapping to physical entropy is accepted solely from a name or monotonic count; the state/dynamics/observable contract must be explicit.
6. The method must fail on a negative control where a count decreases on a declared forward edge.
7. Every physical claim is assigned a status: formal combinatorial result, conditional theorem, open result, proposed correspondence, or empirically tested prediction.
8. No test is described as passing before exact-head artifacts and logs exist.

## Relationship to other work

- #8921: independent alpha-candidate reconstruction and proof-boundary audit.
- #8923: claim dependencies and prospective predictions across the compendium.
- #8919: general shared-generative-grammar benchmark.
- #8920: evidence-carrying derivation chains.

This issue owns the specific order/time and thermodynamic-arrow question. Reuse existing ScientificModel/PhysicsIR, provenance and receipt types after auditing them; do not introduce duplicate universal abstractions.

## Claim boundary

This is a preliminary source-level audit and proposed experiment protocol, not a conclusion that Chain Theory's time or thermodynamic claims are true or false. The core test is whether its discrete counts can be independently mapped to the physical objects and relations that standard thermodynamics and relativity actually define.
