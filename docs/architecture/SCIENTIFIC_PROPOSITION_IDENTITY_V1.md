# Scientific Proposition Identity v1

Status: architecture contract candidate only.

This document refines the Scientific Method Kernel path established by SCI-001, the triangulation contract, and the defeater-aware disposition graph. It specifies the narrow identity theorem that a future Theory Atlas must satisfy before any evidence can be attached to a scientific proposition.

It adds no product implementation and grants no scientific, epistemic, governance, safety, or execution authority.

## 1. Core theorem

A future shared scientific kernel must preserve at least:

```text
proposition family
    != proposition semantic identity
    != human-readable rendering
    != measurement operationalization
    != experiment / falsifier contract
    != estimator / model implementation
    != evidence contribution
    != current scientific disposition
    != truth
```

The central rule is:

```text
same words != same proposition

different words != different proposition

same proposition != same assessment protocol

different assessment protocol != different proposition
```

A proposition identity proves only that two artifacts refer to the same registered scientific target under one exact semantic profile. It does not prove the proposition is true, meaningful, measurable, testable, causal, supported, or currently admissible.

## 2. Correction to an over-broad revision rule

The preceding defeater-aware disposition design correctly required proposition versions to be immutable, but its illustrative wording grouped some assessment choices too closely with proposition semantics.

This contract makes the split explicit.

A change to the scientific claim itself creates a new proposition semantic identity.

Examples include changes to domain-owned semantic dimensions such as:

- population or entity scope when that scope is part of the claim;
- intervention or treatment semantics;
- comparison / counterfactual contrast;
- outcome construct;
- effect target;
- temporal horizon when part of the claimed effect;
- regime / boundary conditions;
- quantifier or modality;
- sign / direction / inequality;
- claimed mechanism when mechanism is constitutive of the proposition;
- mathematical relation or conservation statement.

By contrast, the following normally create a new assessment, measurement, execution, or evidence lineage rather than a new proposition:

- changing data source;
- changing source vintage;
- changing measurement instrument;
- changing operationalization of the same theoretical construct;
- changing estimator;
- changing model implementation;
- changing code/toolchain/runtime;
- changing preregistered falsifier;
- changing diagnostic threshold;
- changing scoring rule;
- changing sampling design;
- changing uncertainty model;
- changing evidence-admission policy.

A domain may explicitly declare that one of these choices changes the semantic proposition in a particular scientific vocabulary, but the shared kernel must not assume that automatically.

Therefore:

```text
ScientificProposition
    + AssessmentContractA
    + AssessmentContractB
```

may represent two independent or dependent tests of one exact proposition rather than two different propositions.

## 3. Why this separation matters

If every experimental change minted a new proposition, the Theory Atlas could never accumulate evidence across multiple methods.

If every measurement or model change silently preserved the complete assessment identity, incompatible experiments could be collapsed into one evidence record.

The intended structure is:

```text
ScientificPropositionIdentity
        |
        +--> MeasurementBinding A
        +--> MeasurementBinding B
        |
        +--> FalsifierContract A
        +--> FalsifierContract B
        |
        +--> Estimator / Model A
        +--> Estimator / Model B
        |
        +--> EvidenceContribution ...
```

This allows one claim to be tested through multiple operationalizations, methods, and evidence lineages while keeping every assessment dependency explicit.

## 4. Three identity layers

### 4.1 Proposition family identity

A proposition family is a navigation/grouping relation only.

Examples:

```text
"minimum wage and employment"
"nuclear mass prediction"
"consciousness and integration"
```

A family identity may group many scientific targets.

It carries no evidence-transfer authority.

```text
same family != same proposition
```

### 4.2 Proposition semantic identity

This is the exact scientific target.

The shared kernel should represent it conceptually as:

```text
ScientificPropositionIdentityV1 {
    proposition_profile_id,
    domain_id,
    domain_semantic_schema_id,
    domain_semantic_payload_digest,
}
```

The shared kernel owns the envelope and identity contract.

The scientific domain owns the meaning of the semantic payload.

### 4.3 Assessment target identity

A concrete experiment, falsifier, forecast, causal design, measurement binding, or adjudication policy should bind the proposition plus its own exact contract identity.

Conceptually:

```text
AssessmentTargetIdentity {
    proposition_id,
    measurement_binding_id?,
    experiment_contract_id?,
    falsifier_contract_id?,
    estimator_or_model_id?,
    outcome_policy_id?,
    evaluation_protocol_id?,
}
```

No universal list is required in v1. The important theorem is that assessment identity is layered above proposition identity rather than silently encoded into prose or hidden implementation state.

## 5. Domain-owned semantic payload

The shared kernel must not attempt to parse English prose into universal scientific semantics.

Instead each domain supplies a registered semantic profile that defines canonical bytes for its proposition type.

Examples of possible domain-owned proposition semantics include:

### Economics

A causal economic proposition might bind:

```text
treatment construct
outcome construct
population
intervention contrast
horizon
effect target
regime / scope
```

### Nuclear science

A proposition might bind:

```text
nucleus identity
observable / property
model-independent relation or predicted interval
physical conditions
scope
```

### ALife

A proposition might bind:

```text
population / lineage scope
perturbation class
response construct
comparison condition
horizon
```

### Neuroscience

A proposition might bind:

```text
anatomical / functional target
measurement construct
subject / population scope
contrast
analysis-independent scientific relation
```

The shared kernel must not force these domains into one field vocabulary.

## 6. Canonical identity rules

A future implementation should derive proposition identity from exact canonical semantic bytes, not from incidental serialization.

The identity contract should require:

- explicit identity profile/version;
- explicit domain identity;
- explicit domain semantic-schema identity;
- explicit canonical semantic payload;
- explicit stable semantic tags;
- deterministic ordering for unordered sets;
- length-prefixed or otherwise unambiguous canonical encoding;
- domain-separated cryptographic commitment.

Identity must not depend on:

- JSON key ordering;
- serde representation;
- debug formatting;
- Rust enum discriminants;
- memory layout;
- hash-map iteration order;
- human title;
- explanatory prose;
- repository path;
- PR number;
- database row ID.

## 7. Presentation is not identity

A proposition may have multiple renderings:

```text
"X increases Y"
"Y rises when X is increased"
"Increasing X causes a positive change in Y"
```

If one registered domain semantic payload represents all three, they can share one proposition identity.

Conversely, two identical strings may refer to different propositions if hidden scientific scope differs.

Therefore:

```text
rendering -> annotation / communication
semantic payload -> identity
```

Human-readable text remains mandatory for auditability, but it does not define cryptographic scientific identity by itself.

## 8. Semantic revision

A registered proposition is immutable.

Changing its canonical semantic payload creates a new proposition identity.

The old proposition remains in history.

A revision may record a typed relation such as:

```text
Specializes
Generalizes
NarrowsScope
BroadensScope
ChangesPopulation
ChangesHorizon
ChangesMechanism
ChangesOutcomeConstruct
Contradicts
ReformulatesWithoutEstablishedEquivalence
```

These edges describe the revision relationship only.

They do not automatically migrate evidence.

## 9. Evidence migration requires a receipt

Evidence attached to proposition A may affect proposition B only through an explicit compatibility or transformation result.

Possible outcomes remain conservative:

```text
ExactSameTarget
ExplicitlyTransformable
RelatedButNotTransferable
Incomparable
```

For an explicit transformation, the receipt should retain:

- source proposition identity;
- destination proposition identity;
- transformation identity;
- assumptions;
- scope limitations;
- information loss;
- dependency inventory;
- qualification/currentness.

The existence of a proposition-family edge is never enough.

## 10. Measurement operationalization stays separate

The Economic Science measurement work established:

```text
theoretical variable != measurement binding
```

This contract preserves that theorem generically.

A scientific proposition may refer to an abstract construct while multiple measurement bindings operationalize it.

Changing from one valid measurement binding to another does not automatically change proposition semantic identity.

Instead the evidence contribution retains the exact measurement binding as a dependency.

If a domain determines that a particular operational definition is constitutive of the claim, it may include that definition in the domain semantic payload explicitly.

The shared kernel must not decide this globally.

## 11. Falsifier identity stays separate

A preregistered falsifier is a scientific assessment object targeting a proposition.

```text
Proposition P
    + Falsifier F1
    + Falsifier F2
```

can represent two distinct attempts to falsify the same proposition.

Changing F1 to F2 therefore normally creates a new falsifier/experiment lineage, not a new proposition.

This is required if SCI-013 falsification campaigns are to accumulate multiple attacks on one exact target.

Falsifier outcomes remain separate:

```text
Triggered
NotTriggered
Inconclusive
NotEvaluable
```

and do not mutate the proposition identity.

## 12. Causal estimand relation

For causal science, a causal estimand can be constitutive of proposition semantics.

For example:

```text
ATE != ATT
ATE[population A] != ATE[population B]
3-month effect != 12-month effect
```

if those dimensions define the scientific target.

But identification strategy and estimator remain assessment dependencies:

```text
same causal estimand
    + RCT
    + IV
    + RD
    + DiD
```

may provide multiple strategy-bound evidence contributions about one proposition.

Thus:

```text
estimand identity can define proposition semantics
identification strategy does not define proposition truth target by itself
```

## 13. Constraint, empirical, and normative propositions

A shared kernel may support multiple proposition classes, but classification itself does not grant authority.

Possible domain-owned classes include:

```text
Constraint
EmpiricalAssociational
EmpiricalMechanistic
Causal
Formal / Mathematical
Normative
```

The shared proposition envelope should bind the domain semantic schema/class identity when required.

Normative propositions must not become empirical merely because they have a content digest.

Formal statements must not become empirically established merely because simulations agree with them.

## 14. Legacy opaque proposition references

Existing Symthaea systems already contain opaque proposition digests or IDs.

Migration must be fail-closed.

A legacy ID can be wrapped conceptually as:

```text
LegacyOpaquePropositionRef {
    legacy_namespace,
    legacy_id,
}
```

without pretending its semantic payload is known.

A migration to `ScientificPropositionIdentityV1` requires an explicit semantic reconstruction/admission process.

```text
legacy id equality
    != shared-kernel semantic identity
```

## 15. Proposition identity does not imply compatibility

Two proposition identities can be different while a domain adapter later proves them explicitly transformable.

Two proposition identities can also be syntactically similar but scientifically incomparable.

The identity layer therefore exposes no automatic:

```text
is_equivalent()
can_pool()
shares_evidence()
is_more_general()
is_true()
```

Those belong to separately qualified relation/compatibility layers.

## 16. Argument graph integration

The scientific argument graph should target exact proposition identities.

Conceptually:

```text
EvidenceContribution
    --Supports--> PropositionId

EvidenceContribution
    --Opposes--> PropositionId

Defeater
    --Undercuts--> exact evidence relation / inference object

FalsifierResult
    --Targets--> PropositionId
```

A proposition revision therefore does not silently retarget existing graph edges.

New target edges require explicit compatibility/migration evidence.

## 17. Time-indexed Theory Atlas views

Because proposition identity is immutable, an Atlas can reconstruct historical state safely.

```text
AtlasView(proposition_id, time, policy_generation)
```

can derive a scientific state from the evidence and lifecycle events admissible at that time.

Today's proposition rewrite, correction, or revised scope cannot mutate yesterday's target identity.

## 18. No mutable truth record

The proposition object itself should contain no mutable fields such as:

```text
supported: bool
verified: bool
causal: bool
confidence: f64
truth_probability: f64
current_disposition
```

Disposition is always derived from exact proposition identity plus exact current evidence/argument/policy generation.

## 19. No evidence inheritance from identity

A proposition identity grants no evidence channel.

A proposition family grants no evidence channel.

A compatibility receipt grants no evidence channel by itself.

An evidence contribution must pass its own provenance, measurement, execution, dependency, and admission contracts.

## 20. Suggested future implementation order

Do not jump directly to a Theory Atlas database.

A narrow implementation sequence is:

```text
SCI-014a.1  ScientificPropositionIdentityV1 envelope
SCI-014a.2  domain semantic-profile registration / canonical payload contract
SCI-014a.3  immutable proposition registry + revision edges
SCI-014a.4  explicit target-compatibility receipts
SCI-014a.5  assessment-target binding
SCI-014a.6  argument-graph integration
```

Each tranche should receive independent qualification.

## 21. Required adversarial cases

A future qualification suite should prove at least:

1. presentation-only text changes do not change semantic identity when canonical payload is unchanged;
2. semantic payload changes do change proposition identity;
3. caller ordering cannot change identity;
4. JSON/debug/serde representation does not define identity;
5. same family does not imply same proposition;
6. different measurement bindings can target the same proposition;
7. different falsifiers can target the same proposition;
8. different estimators can target the same proposition;
9. different causal estimands produce different proposition targets when estimand semantics are constitutive;
10. old evidence does not automatically migrate to a revised proposition;
11. legacy opaque IDs cannot masquerade as registered semantic proposition IDs;
12. proposition identity contains no disposition/truth authority.

## 22. Authority boundary

The complete theorem remains:

```text
ScientificPropositionIdentityV1
    != semantic equivalence receipt
    != measurement admission
    != experiment qualification
    != evidence contribution
    != scientific disposition
    != canonical belief
    != recommendation
    != governance decision
    != execution authority
```

## 23. Theory Atlas path after this refinement

The intended SCI-014 path becomes:

```text
SCI-006 evidence dependency graph
    -> target compatibility / result comparison / triangulation
    -> defeater-aware scientific argument graph
    -> immutable scientific proposition identity       [this contract]
    -> evidence lifecycle / correction / retraction
    -> disposition assessment + full reason topology
    -> time-indexed Theory Atlas storage/query projection
```

The proposition identity layer is intentionally early because every later relation, lifecycle event, compatibility receipt, falsifier result, and disposition must target an immutable scientific object.

## 24. Important non-claims

This document does not:

- implement proposition identity in Rust;
- define one universal scientific proposition schema;
- define universal semantic equivalence;
- prove any current opaque proposition digest is semantically canonical;
- decide whether an operationalization is constitutive of a proposition in every domain;
- define a universal causal estimand;
- qualify any existing evidence;
- establish any scientific proposition as true or false;
- authorize a Theory Atlas implementation to influence action.

Review this contract only on the question:

> Does it give a future shared kernel a precise immutable target identity while keeping measurement, experiment, falsifier, estimator, evidence, and disposition lineages separate enough for real cross-method science?
