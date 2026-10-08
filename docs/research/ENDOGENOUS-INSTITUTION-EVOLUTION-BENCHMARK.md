# Endogenous Institutional Evolution Benchmark

**Status:** research design v0  
**Related issue:** Symthaea #7046  
**Parent:** #5606  
**Source frontier:** institutional-evolution-method-v0

## Purpose

Extend the financial mechanism laboratory from comparing predeclared institutions to testing **bounded endogenous institutional emergence**.

The laboratory asks:

> When agents repeatedly encounter coordination problems under bounded information and explicit constraints, which institutional mechanisms are proposed, adopted, implemented, amended, rejected, or abandoned?

This is not a search for a globally optimal institution. It is a controlled experiment on institutional formation and change.

## Methodological anchors

The design is informed by:

- Masahiko Aoki's institutional-process view, where beliefs, strategies, equilibrium states of play, and public representations recursively interact and institutions co-evolve with their environments.
- Avner Greif and David Laitin's account of endogenous institutional change through repeated interaction, self-reinforcement, and quasi-parameters that can become endogenous over longer horizons.
- Elinor Ostrom's Institutional Analysis and Development framework, which treats action situations, rules, actors, community attributes, biophysical conditions, and outcomes as connected through feedback.
- Agent-based computational economics, which studies how repeated decentralized interactions can generate macro-level institutions and conventions.

References:

- https://www.cambridge.org/core/journals/journal-of-institutional-economics/article/endogenizing-institutions-and-institutional-changes/AD5F9B9201F6AA456644417DC245B764
- https://www.cambridge.org/core/journals/american-political-science-review/article/abs/theory-of-endogenous-institutional-change/161BF12CD6B8AA93F80EDA25CCDE12A9
- https://ostrom.indiana.edu/courses-teaching/teaching-tools/iad-framework/index.html
- https://www.sciencedirect.com/science/article/pii/S0020025502002803

## Causal loop

The minimum experimental loop is:

```
recurrent coordination problem
        ->
local observations + bounded communication
        ->
agent responses
        ->
success / failure / distributional signal
        ->
institutional proposal
        ->
explicit adoption process
        ->
implementation
        ->
repeated use
        ->
observed outcomes
        ->
feedback into beliefs, technology, community attributes, and rules
```

Proposal generation, adoption authority, implementation, and evaluation are separate mechanisms.

## Institution state

An institution is a versioned mechanism bundle, not a label.

Every state must bind:

- institution_id;
- semantic_version;
- parent institution identity;
- action situations;
- participant roles;
- action set;
- information available to each role;
- decision procedure;
- allocation procedure;
- monitoring;
- verification;
- sanctions/remedies;
- entry;
- exit;
- amendment;
- repeal/supersession;
- authority distribution;
- resource/property rights;
- settlement/claim semantics;
- dispute and appeal route;
- implementation cost;
- observability cost;
- enforcement cost;
- switching or migration cost.

A profile such as `market`, `commons`, or `Creditism` is only shorthand for a fully declared mechanism bundle.

## Seed boundary

Experiments must begin from an explicit institutional substrate.

At minimum declare:

- communication protocol;
- baseline governance process;
- baseline property/access rules;
- baseline settlement mechanism;
- baseline dispute mechanism.

The experiment may permit specified dimensions to mutate.

The experiment must not imply that agents begin from a literal institutional vacuum.

## Mutation grammar

Institutional change is bounded by an explicit mutation grammar.

Example mutation families:

### Coordination

- pooling;
- queue/reservation;
- matching rule;
- milestone commitment;
- allocation rule.

### Information

- disclosure;
- monitoring;
- verification;
- certification;
- reputation;
- signal aggregation.

### Risk

- pooling;
- contingency reserve;
- loss-sharing;
- cancellation;
- salvage.

### Governance

- centralization/decentralization;
- delegation;
- voting threshold;
- affectedness weighting;
- appeal;
- rule-revision authority.

### Property and access

- private title;
- cooperative title;
- stewardship;
- time-bounded use right;
- exclusion rule;
- common-pool rule.

### Intertemporal commitment

- staged release;
- milestone funding;
- claim creation;
- claim transferability;
- default/cancellation treatment.

### External dependence

- supplier diversification;
- reserve;
- import substitution;
- export specialization;
- external-settlement arrangement.

Every mutation must expose a semantic delta and preserve parent identity.

## Problem generator

Coordination problems are generated independently from institutional mutations.

Required classes:

- scarce capital;
- correlated/uninsurable risk;
- information asymmetry;
- commons conflict;
- long-horizon projects;
- verification gaming;
- low-observability maintenance;
- external supplier dependency;
- emergency coordination;
- skill bottleneck;
- spatial coordination;
- intergenerational investment.

The institutional process may adapt to the problem; it must not choose the problem to make its own preferred mechanism appear successful.

## Agent model

Agents must have declared:

- objectives/preferences;
- resources;
- risk tolerance;
- information set;
- communication graph;
- bounded-rationality model;
- learning/update rule;
- memory horizon;
- switching cost;
- institutional knowledge;
- proposal capability;
- adoption capability.

Agents may be heterogeneous.

Future evaluator truth must never be observable by proposal or adoption logic.

## Proposal/adoption separation

At least seven states are distinct:

```
Proposal
-> PublicPresentation
-> Deliberation
-> Adoption
-> Implementation
-> Review
-> Amendment/Rejection/Repeal
```

This creates the required separation:

```
institutional innovation
!=
institutional authority
!=
institutional legitimacy
```

The simulator may evaluate procedural outcomes without declaring political legitimacy.

## Selection boundary

Never collapse institutional performance to one hidden scalar.

Emit a vector containing, where applicable:

- coordination success;
- unmet demand;
- resource loss;
- risk absorbed;
- information cost;
- monitoring burden;
- enforcement cost;
- governance burden;
- concentration;
- participation;
- access inequality;
- maintenance continuity;
- resilience;
- innovation rate;
- switching cost;
- external dependency;
- failure incidence.

Multiple equilibria and institutional diversity are valid outcomes.

## Endogeneity classes

Each state variable is classified for the duration of a run.

### Exogenous

- physical laws;
- initial resource stock;
- exogenous shock schedule;
- fixed measurement definitions.

### Quasi-endogenous

- technology;
- beliefs;
- network topology;
- skill distribution;
- community attributes;
- preferences, where explicitly modeled.

### Endogenous

- institutional rules;
- governance structure;
- allocation procedure;
- monitoring arrangement;
- verification arrangement;
- authority distribution;
- claim semantics when mutation is enabled.

The classification itself is part of the experiment identity.

## Institutional lineage

Use an append-only lineage DAG:

```
institution-v0
  -> mutation-a -> candidate-v1 -> rejected
  -> mutation-b -> candidate-v2 -> adopted
                                  -> amended -> v3
                                  -> repealed
```

Every transition records:

- parent identity;
- mutation identity;
- proposer;
- affected roles/population;
- adoption decision;
- decision procedure;
- semantic delta;
- triggering signal/problem;
- evidence references;
- implementation result;
- subsequent outcome vector;
- supersession/repeal state.

This lineage is the institutional analogue of an evidence chain.

## Emergence fixtures

### E1 — repeated coordination failure

Hold the problem class constant across seeds.

Measure whether related coordination mechanisms repeatedly arise.

### E2 — repeated risk loss

Introduce repeated correlated shocks.

Measure whether pooling, reserves, or loss-sharing mechanisms arise.

### E3 — hidden information

Hide quality from one side of the interaction.

Measure whether monitoring, certification, disclosure, verification, or reputation arise.

### E4 — long-horizon commitment

Require resources before uncertain returns.

Measure whether staged commitments, pooling, claims, or alternative commitment arrangements arise.

### E5 — commons depletion

Make resource use individually attractive but collectively depleting.

Measure whether access, monitoring, sanction, stewardship, or governance mechanisms arise.

### E6 — verification gaming

Allow agents to improve measured performance without improving the underlying target.

Measure whether verification evolves.

### E7 — maintenance neglect

Make benefits immediate and maintenance delayed/poorly observable.

Measure whether maintenance obligations, reserves, stewardship, or role specialization arise.

### E8 — external dependency

Constrain access to a strategic imported input.

Measure whether diversification, reserves, substitution, export specialization, or external settlement mechanisms arise.

## Metamorphic tests

1. Same physical world, different initial institution: pathways remain distinguishable.

2. Same initial institution, different random seed: stochastic variation is allowed, deterministic invariants are preserved.

3. Same problem, different observability: information mechanisms may differ; physical accounting does not.

4. Same problem, different adoption rule: institutional trajectories may differ; the problem generator does not.

5. Same mutation, different physical environment: mutation semantics remain identical.

6. Evolution disabled: seed institution remains unchanged.

7. Institution frozen: existing fixed-mechanism benchmarks reproduce their baseline behavior.

8. Lineage replay with the same profile/seed is deterministic.

9. Counterfactual removal of an adopted institution changes downstream state only through declared causal edges.

## Anti-selection-bias controls

Every campaign should include:

- multiple seeds;
- multiple initial institutional states;
- proposal-generator ablations;
- adoption-process ablations;
- fixed-institution controls;
- random-mutation controls;
- planner-designed positive controls;
- holdout shocks;
- parameter perturbations;
- mutation-order permutations.

A mechanism cannot be deemed successful merely because the mutation grammar explicitly encoded it as the target.

## Institutional Evolution Record

Every run should emit a deterministic record containing:

- mechanism_profile_hash;
- seed_state_hash;
- problem_schedule_hash;
- mutation_grammar_hash;
- agent_model_hash;
- adoption_rule_hash;
- random_seed;
- oracle_version;
- full institution lineage;
- rejected proposals;
- adopted proposals;
- implementation failures;
- outcome vectors;
- uncertainty;
- claim ceiling.

The human-readable dashboard is not authoritative.

## Failure dispositions

Use explicit dispositions including:

- AccountingInvalid;
- CoordinationFailure;
- InformationFailure;
- VerificationFailure;
- GovernanceCapture;
- AllocationCapture;
- EnforcementFailure;
- InstitutionalLockIn;
- PathDependence;
- MaintenanceFailure;
- ExternalSettlementFailure;
- AdaptationFailure;
- ProposalSpaceExhausted;
- MultipleEquilibria;
- Unresolved.

## Qualification boundary

A PASS establishes only that the declared evolutionary machinery produced the declared institutional transitions under the frozen synthetic profile.

It does not establish:

- social optimality;
- political legitimacy;
- universal desirability;
- historical equivalence;
- causal validity outside the model;
- real-world scalability.

## Research output

The preferred result is not:

```
winner = X
```

It is:

```
problem X
-> candidate mechanisms A/B/C emerge
-> A persists under observability regime 1
-> B is more robust under risk regime 2
-> C requires governance condition 3
-> divergence persists under identical physical shocks
-> lineage/path-dependence explains the divergence
```

This makes institutional evolution itself an object of study rather than an assumption hidden inside a benchmark.
