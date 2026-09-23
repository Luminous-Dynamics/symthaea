# Systemic Intelligence Authority Model

Status: proposed architecture decision  
Issue: #5578 (`SYS-ADR-001`)  
Scope: Symthaea, Mycelix, Futures Laboratory, Sol-Atlas

## Decision

Systemic intelligence is not one graph and not one authority domain.

The stack MUST preserve a typed separation between source-bound evidence, observed/derived structure, dynamic feedback models, causal hypotheses/estimates, simulated scenarios, and normative decisions.

The core invariant is:

```text
evidence
!= structural finding
!= dynamic model
!= causal claim
!= scenario result
!= normative recommendation
```

No API, bridge, visualization, or model may silently promote a value from one level to a stronger level.

## Why this is necessary

Complex social, economic, ecological, technical, and institutional systems contain cycles, delays, hidden variables, incomplete records, changing identities, conflicting sources, and boundary-dependent effects.

A dense network can arise from ordinary market structure. A feedback loop can exist without proving a particular causal direction. A causal model can be useful while remaining non-identifiable from observational data. A simulation can illuminate consequences without becoming evidence that those consequences will occur.

The architecture therefore treats uncertainty and provenance as data rather than presentation details.

## Ownership boundaries

### Mycelix: evidence and statement authority

Mycelix owns:

- source-bound claims and observations;
- provenance and derivation lineage;
- entity identifiers and explicit identity-resolution assertions;
- valid-time and knowledge-time history;
- contradictions, supersession, and source disagreement;
- evidence references and cryptographic/integrity bindings;
- jurisdiction and scope metadata.

Mycelix MUST be able to preserve mutually inconsistent statements without selecting one merely because it is newer or more convenient.

### Symthaea: analytic authority

Symthaea owns:

- structural analysis;
- multilayer network analysis;
- stocks, flows, delays, and feedback-loop analysis;
- dependency, concentration, bottleneck, resilience, and propagation analysis;
- causal discovery and causal reasoning under explicit assumptions;
- systemic hypotheses and falsification conditions;
- sensitivity, robustness, and null-model comparison.

Symthaea-derived output MUST retain the evidence lineage and assumptions needed to reproduce or challenge it.

### Futures Laboratory: scenario authority

The Futures Laboratory owns:

- scenario construction;
- intervention simulation;
- ensemble uncertainty;
- calibration against subsequently observed outcomes where available;
- sensitivity and regret analysis under model misspecification.

A scenario result MUST NOT be re-imported as an observation of the world.

### Sol-Atlas: presentation and interrogation authority

Sol-Atlas owns:

- human exploration of evidence, structure, dynamics, causal hypotheses, and scenarios;
- layer selection and boundary selection;
- explanation surfaces;
- comparison of competing models;
- uncertainty and contradiction visualization.

Visualization MUST NOT strengthen the epistemic authority of its inputs.

## Five graph/model classes

### 1. Evidence graph

Purpose: represent what a source asserted or measured.

Cycles: allowed.  
Authority: source-bound evidence only.

An edge means that a statement exists, not that the statement is true.

Examples:

- registry A reports that entity X has parent Y;
- procurement release R reports award A to supplier S;
- sensor S measured value V at time T;
- document D asserts relationship Q.

### 2. Structural graph

Purpose: represent typed observed or derived relationships across layers.

Cycles: allowed.  
Authority: structural finding.

Possible layers include:

- ownership;
- contracts/procurement;
- money/funding;
- supply/materials;
- labor;
- information;
- authority/regulation;
- energy/resources;
- risk;
- externalities.

A structural edge MUST state whether it is directly observed, source-asserted, resolved from multiple sources, or analytically derived.

### 3. System-dynamics graph

Purpose: represent stocks, flows, delays, reinforcing loops, balancing loops, and nonlinear response.

Cycles: expected and often required.  
Authority: dynamic model.

A feedback loop is not interchangeable with a causal DAG. The model MUST preserve its governing equations/rules, units where applicable, delay assumptions, and boundary choices.

### 4. Causal model

Purpose: reason about interventions and counterfactuals.

Representation: DAG, CPDAG, PAG, SCM, or another explicitly named formalism.  
Authority: causal hypothesis or identified/estimated effect, depending on evidence.

The model MUST expose:

- discovery/estimation algorithm;
- causal-sufficiency assumptions;
- conditional-independence assumptions;
- unresolved edge orientation;
- adjustment/identification assumptions;
- observational versus interventional evidence;
- known or suspected latent confounding;
- sample and temporal scope.

Unresolved orientation MUST remain unresolved. A heuristic direction MUST NOT be encoded as an identified causal direction.

### 5. Scenario ensemble

Purpose: explore possible trajectories under explicit model and intervention assumptions.

Cycles: arbitrary.  
Authority: simulated scenario.

Every result MUST retain:

- starting state;
- intervention;
- model version;
- objective(s);
- stochastic seed/ensemble identity where applicable;
- parameter uncertainty;
- structural uncertainty;
- sensitivity results.

Scenario output is never promoted to observation solely because it is high confidence or frequently reproduced by the same model family.

## Finding authority ladder

Systemic analysis uses the following monotone authority ladder:

```text
Observation
    |
RelationshipClaim
    |
StructuralFinding
    |
MechanismHypothesis
    |
CausalCandidate
    |
CausalEstimate
    |
ScenarioResult
```

This is not a simple numeric confidence scale. Each level has different semantics.

### Observation

A source-bound measurement or assertion with provenance.

### RelationshipClaim

A typed relationship asserted by one or more observations/statements.

### StructuralFinding

A derived property of a structural graph, such as concentration, dependency, community structure, bottleneck status, or feedback-loop membership.

### MechanismHypothesis

A falsifiable explanation for why a structural/dynamic pattern may arise.

### CausalCandidate

A causal orientation/mechanism supported by a declared causal model but not yet established as an identified effect.

### CausalEstimate

An identified/estimated causal effect under explicit assumptions and evidence.

### ScenarioResult

A model-derived outcome under a particular intervention/scenario specification.

`ScenarioResult` is not "stronger evidence" than `CausalEstimate`; it is a different kind of authority. Implementations MUST NOT treat this ladder as an ordinal score.

## Promotion rules

Promotion between authority types MUST be explicit and evidence-bearing.

Examples:

```text
Observation -> RelationshipClaim
requires:
  mapping semantics
  source lineage
  temporal scope

RelationshipClaim -> StructuralFinding
requires:
  declared graph projection
  boundary definition
  derivation method

StructuralFinding -> MechanismHypothesis
requires:
  explicit hypothesis
  alternatives
  falsifiers

MechanismHypothesis -> CausalCandidate
requires:
  causal formalism
  declared assumptions
  discovery/identification evidence

CausalCandidate -> CausalEstimate
requires:
  identification argument
  estimator
  uncertainty
  diagnostics/sensitivity
```

No promotion may be inferred from a visualization property such as line thickness, node centrality, cluster density, or color.

## Bitemporal semantics

Every time-varying real-world statement SHOULD distinguish at least:

```text
valid_time     = when the claim says the relationship/state held in the world
knowledge_time = when the system/source learned, published, or recorded the claim
```

Where source schemas provide additional times (retrieval, filing, effective date, observation time), they SHOULD be preserved rather than collapsed.

This supports questions such as:

- What did we believe on date K about the world on date V?
- What do we now believe was true on date V?
- When did a newly published fact become available to a decision maker?

Analyses MUST NOT mix future knowledge into historical decision reconstruction unless explicitly operating in hindsight mode.

## Entity resolution

Identity resolution is evidence, not cleanup.

```text
possible_same_entity != same_entity
shared_name != same_entity
shared_address != same_entity
shared_identifier_from_untrusted_source != same_entity
```

Resolution MUST retain:

- identifiers and namespaces;
- evidence used to resolve;
- algorithm/manual authority;
- confidence/ambiguity;
- validity interval where identity may change through mergers, splits, renaming, or reorganization.

Ambiguous candidates remain distinct until an explicit resolution assertion is admitted.

## Required systemic finding envelope

Every derived systemic finding SHOULD carry:

```text
finding_id
authority_kind
subject_scope
jurisdiction_scope
valid_time
knowledge_time
source_evidence
source_graph_version
derivation_method
assumptions
supporting_evidence
contradicting_evidence
missing_evidence
alternative_explanations
boundary_definition
sensitivity
robustness
falsification_conditions
model_version
```

Fields may be optional only where their absence is itself represented explicitly.

## Structural anomaly and null models

An unusual-looking graph is not necessarily unusual for the domain.

Structural anomaly detection SHOULD compare observations against one or more declared null/baseline models preserving relevant constraints such as:

- node degree;
- actor size;
- geography;
- sector;
- procurement volume;
- market concentration;
- temporal activity;
- known capacity constraints.

The output MUST distinguish:

```text
raw_structure
expected_structure_under_null
residual/anomalous_structure
```

`anomalous` means inconsistent with the declared baseline, not suspicious, malicious, illegal, or coordinated.

## Institutional and industrial-network invariants

The following implications are forbidden unless separately evidenced:

```text
network proximity != influence
influence != coordination
coordination != collusion
collusion != legal culpability

high centrality != control
concentration != capture
shared incentive != conspiracy
revolving employment != corrupt exchange
funding relationship != editorial/scientific control
regulatory contact != regulatory capture
externalized modeled cost != established legal liability
```

Named real-world actors MUST be evaluated using the same standards as synthetic or anonymized actors.

## Causal discovery guardrails

Observational causal discovery is hypothesis generation unless stronger identification evidence is present.

At minimum:

- unresolved edges remain unresolved;
- Markov-equivalent structures remain distinguishable from identified orientations;
- order dependence is tested or removed where the chosen method permits;
- latent-confounding assumptions are explicit;
- missing-data assumptions are explicit;
- effect estimation is separate from edge-discovery confidence;
- interventions are not inferred from topology alone.

## Intervention analysis

Systemic intelligence may identify leverage points and simulate interventions, but intervention evaluation MUST expose the objective function.

There is no universal scalar "better" hidden inside systemic analysis.

Intervention comparisons SHOULD support multiple outcomes, trade-offs, and distributional effects, for example:

- aggregate benefit;
- worst-case harm;
- reversibility;
- resilience;
- cost;
- time-to-effect;
- externality coverage;
- distribution across affected populations;
- uncertainty/model disagreement.

Changing objective weights MUST be treated as a normative choice, not a causal discovery result.

## Privacy and person-level analysis

Systemic analysis SHOULD prefer institutional and aggregate structure when person-level detail is unnecessary.

Person-level data MUST preserve source/legal-use constraints and SHOULD minimize personally identifying information in public outputs.

The system MUST NOT infer sensitive personal traits merely from network proximity, association, or membership unless such inference is explicitly in scope, lawfully sourced, and epistemically justified.

## Qualification before named-entity conclusions

Before systemic inference is relied on for named real-world organizations, the implementation SHOULD pass deterministic synthetic/adversarial fixtures covering:

- dense-but-benign networks;
- sparse coordinated structures;
- latent confounding;
- Simpson's paradox;
- collider/selection bias;
- missing-not-at-random observations;
- stale/conflicting sources;
- entity-resolution collisions;
- delayed feedback and hysteresis;
- nonlinear shock propagation;
- rebound/substitution effects after interventions.

Primary quality metrics include false-positive structural anomaly rate, false-positive causal orientation rate, uncertainty calibration/coverage, provenance retention, contradiction retention, boundary sensitivity, and abstention quality.

## Interoperability direction

External standards are evidence adapters, not semantic authorities over the whole model.

Priority mappings include:

- W3C PROV for entity/activity/agent provenance concepts;
- Beneficial Ownership Data Standard (BODS) for source-bound entity/relationship statements and temporal semantics;
- Open Contracting Data Standard (OCDS) for immutable procurement releases and versioned contracting histories;
- GLEIF LEI relationship data for legal-entity identity and direct/ultimate accounting-parent relationships.

Adapters MUST preserve source-native semantics and reporting exceptions instead of normalizing them into stronger Mycelix/Symthaea claims.

## Review checklist

A systemic-intelligence PR is incomplete if reviewers cannot answer:

1. What authority type does this output have?
2. Which graph/model class produced it?
3. Which evidence supports it?
4. What contradicts it?
5. What assumptions are required?
6. What time and jurisdiction does it cover?
7. What would falsify or materially change it?
8. Does it preserve unresolved uncertainty?
9. Could a visualization or bridge accidentally strengthen the claim?
10. Is any normative objective being presented as an empirical conclusion?

## Consequences

This decision intentionally makes some analyses more conservative. Symthaea will sometimes answer "unresolved," "boundary-dependent," or "insufficient evidence" where a simpler graph system would emit a confident edge or label.

That abstention is a feature. Systemic intelligence is useful only if people can distinguish what was observed, what was inferred, what was assumed, and what was simulated.