# SAM Option-Preservation Measurement v0.1

Status: research evaluator / finite-envelope synthetic contract  
Implementation: `symthaea_vehicle::option_preservation`  
Canonical analytic conformance fixture: `fixtures/SAM_OPTION_PRESERVATION_CLAIRVOYANCE_TRAP_V0_1.json`  
Owner: Symthaea; consumes Symtropy execution and upstream qualification receipts  
Issue: #11291

## 1. The distinction

An action can be feasible now and still leave the vehicle with no usable next decision.

A second pitfall is **clairvoyance**. Suppose two hidden scene states produce the same observation at the next decision:

- in state A, only `brake` is feasible;
- in state B, only `coast` is feasible.

A simulator that knows the hidden state can report that each state has a feasible continuation. The actual controller cannot choose differently if it cannot distinguish A from B. The policy-level continuation set is empty.

If a later observation really distinguishes the states, the policy may then choose different actions. The observation partition must be backed by the exact delivered-observation receipt; a hidden-state ID, reference-model ID or evaluator label cannot create a new controller-visible class.

## 2. Set-valued computation

For one first action and one controller-visible observation class (o), let (C(o)) be the declared successor cells. Each cell represents a hidden-state / qualified-reference combination and contains one disposition for every action in the frozen next-action set (A_1).

The robust option lower set is:

[
L(o)=\{a\in A_1 : \forall c\in C(o),\;F(c,a)=\text{Feasible}\}.
]

The possible option upper set is:

[
U(o)=\{a\in A_1 : \forall c\in C(o),\;F(c,a)\ne\text{Infeasible}\}.
]

Therefore (L(o)\subseteq U(o)).

- A non-empty lower set establishes at least one common continuation for the declared class.
- An empty upper set establishes that no action in the declared continuation set remains possible for that class.
- An empty lower set with non-empty upper set is unresolved; unknown evidence cannot enter the lower set.
- Missing expected cells or action outcomes produce incomplete output, not a dead-end or a pass.

A first action is robustly recoverable over the declared envelope only when every expected controller-visible observation class has a non-empty lower set. If any expected class has an empty upper set, the action has no possible continuation for at least one declared observation class.

## 3. Clairvoyance gap

For each observation class, the evaluator also reports the union of actions that are feasible in at least one hidden cell. This is the statewise/clairvoyant union, not a policy-feasible set.

When every disposition is known:

[
G(o)=\left|\bigcup_{c\in C(o)} A_{\text{feasible}}(c)\right|-|L(o)|.
]

A positive gap exposes apparent options that depend on knowing which hidden state is true. The metric is descriptive and depends on the frozen finite action set; it is not a probability or a safety score.

## 4. Ownership and evidence

The evaluator consumes, but does not replace:

- Symthaea #10338 for action-domain coverage and decision-sufficient finite candidate sets;
- Symthaea #10266 / #10318 for reference qualification and reference-set envelopes;
- Symthaea #10256 / #10257 for prediction-set coverage and model discrepancy;
- Symthaea #10083 / MANIFOLD-006 #2022 for action-conditioned avoidability and generic reachability;
- Symthaea #10237 / #10252 for regret and cross-model action-ranking stability;
- Symthaea #10320 for decision-sensitive measurement;
- Symthaea #10012 / #3779 for runtime assurance and recovery authority;
- Symtropy #3387 / #3471 / #3532 for exact branches, causal observations, actors and replay receipts.

The qualification receipt fields are passed through as upstream evidence identities. This module checks their presence and status; it does not authenticate their origin or independently qualify an action domain, reference model, execution or observation partition.

## 5. Fail-closed behavior

- A missing expected successor cell is incomplete.
- A missing action disposition is incomplete.
- An explicit unknown disposition widens the upper set and never enters the lower set.
- A duplicate cell key, cell receipt or execution receipt is rejected.
- The expected-cell manifest digest is recomputed from a canonical key-sorted manifest; content mutation is rejected and input ordering does not affect the digest.
- An undeclared action or mismatched action-set/evaluator digest is rejected.
- Hidden-state/reference IDs cannot split one observation class.
- Different actions may be selected only across distinct receipt-backed observation classes.
- An unqualified upstream reference or action-domain receipt blocks the option claim.
- No probabilities are inferred from counts of references, hidden states, cells or actions.

The expected-cell manifest is the finite universe being measured. Its BLAKE3 digest is recomputed from canonical serialized entries (sorted by complete cell key) to detect mutation and ordering ambiguity. This does not prove that the upstream manifest is exhaustive; completeness remains the responsibility of its owning scenario, action-domain, observation-partition and reference-qualification systems.

### Persistent clairvoyance-trap fixture

The canonical fixture at `fixtures/SAM_OPTION_PRESERVATION_CLAIRVOYANCE_TRAP_V0_1.json` encodes two hidden successor states with the same delivered observation:

- Hidden state A: `brake` feasible, `coast` infeasible.
- Hidden state B: `brake` infeasible, `coast` feasible.
- Both cells share one observation-equivalence receipt and one class ID.

The expected policy-level result is an empty robust lower set and empty possible upper set, while the clairvoyant union contains both actions; the clairvoyance gap is 2. The Rust test loads this persistent fixture, derives its canonical BLAKE3 manifest digest, and checks the result against the declared expected values. The fixture is analytic/synthetic only.

## 6. Benchmark plan

Start with analytic finite-state fixtures, then consume exact Symtropy branches:

1. Common-continuation controls.
2. Same-observation clairvoyance traps.
3. Observation-resolved ambiguity.
4. Unknown and incomplete successor branches.
5. Reference disagreement and model-family shifts.
6. Delayed/stale observation delivery and two-sided policy interaction.

Compare a myopic selector, short-horizon MPC, a rule-based fallback, learned/VLA proposals and an option-preserving selector using the same candidate actions, observation partition, reference envelope and compute budget.

Report robust/possible option widths, clairvoyance gap, dead-end count, unresolved/incomplete fractions, intervention/fallback events, route progress, comfort and p50/p95/p99 evaluator latency separately. Never collapse them into a single hidden weighted score.

## 7. Relationship to current research

The September 2026 SURE-Plan paper combines evidential uncertainty with spatio-temporal reachable-set constraints for autonomous-vehicle planning. It is a relevant algorithm comparator, not an external benchmark result for SAM: [SURE-Plan, Transportation Research Part C, DOI 10.1016/j.trc.2026.105800](https://doi.org/10.1016/j.trc.2026.105800).

NIST's Measurement Science for Automated Vehicles programme separately distinguishes surrogate safety, forward simulation, counterfactual action comparison and systematic weakness analysis. This evaluator complements that decision-quality direction by measuring whether a selected action preserves observation-conditioned next-step options: [NIST MSAV](https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0).

Autoware 2.0's published generator-selector direction is another architecture comparator. SAM's proposed contribution is not simply another selector; it is an independently measured option-preservation property under an explicit observation/reference envelope: [Autoware Architecture 2.0](https://docs.autoware.org/main/design/autoware-architecture-v2/).

## 8. Claim ceiling

V0.1 is a finite, declared-envelope offline measurement. A test pass establishes the set computation for the declared inputs. It does not prove global viability, action-domain completeness, physical safety, real-world risk probability, or permission to execute a command.
