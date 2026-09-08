# Scientific Evidence Dependency Graph v1

**Status:** architecture contract only; non-authorizing; non-qualifying.

**Series:** SCI-006, stacked on SCI-005.

**Parent:** `architecture/exploratory-confirmatory-separation-v1@90305fef7c3a8adeb682492bbcbac66b9702fde9`

## 1. Purpose

SCI-006 defines the common dependency substrate needed to answer a deceptively difficult scientific question:

> When several results point in the same direction, how many genuinely distinct evidentiary lineages do we have, what do they share, and what remains unknown?

Symthaea already contains two complementary strong patterns.

RCA evidence lineage provides a closed ancestry DAG over exact evidence artifacts and derives overlap from root ancestry rather than accepting `independent: bool` from producers.

Economic Science adds a typed inventory of scientific dependencies such as source data, source vintage, measurement specification, sampling design, transformations, models, estimators, identification strategies, outcome policies, and evaluation protocols. Its strongest no-overlap language is deliberately `DeclaredDisjointWithinScope`, not `IndependentReplication`, and incomplete inventories fail closed.

SCI-006 preserves both strengths while making them composable with SCI-002 artifact identity, SCI-003 execution identity, SCI-004 experiment contracts, and SCI-005 exposure/use lineage.

---

## 2. Core theorem

SCI-006 preserves these distinctions:

- number of publications is not number of evidence lineages;
- number of evidence artifacts is not number of independent observations;
- different model implementations are not necessarily independent evidence;
- different authors or institutions are not scientific-independence oracles;
- distinct content identities are not necessarily distinct information sources;
- shared ancestry establishes dependence within the declared graph, but disjoint declared ancestry does not prove universal independence;
- root disjointness is not statistical independence;
- pairwise no-overlap is not transitive;
- known dependency connected components are not equivalence classes of identical evidence;
- methodological diversity is not the same thing as dependency independence;
- exact replication is not the same thing as conceptual triangulation;
- same target is not implied by similar prose;
- target compatibility is not dependency independence;
- incomplete inventory cannot support a universal no-overlap conclusion;
- qualified replication evidence is not scientific truth;
- scientific replication evidence is not action authority.

The central rule is:

> Evidence multiplicity must be derived from exact target-compatible lineages and their declared dependency topology, never from paper count, agent count, model count, organization count, or producer-supplied independence flags.

---

## 3. Three different graph questions

SCI-006 must not force all provenance into one undifferentiated graph.

### 3.1 Derivation ancestry

Question: Which exact artifacts were derived from which earlier artifacts?

Examples:

- raw observation -> cleaned table;
- dataset -> fitted model;
- theorem target -> generated proof witness;
- source documents -> extracted claim set;
- previous discovery -> learned grammar primitive;
- source image -> derived feature artifact.

This is a directed acyclic provenance graph when the domain declares derivation semantics acyclic.

### 3.2 Scientific dependency inventory

Question: Which scientific assumptions/resources/methods does a contribution depend on, even when there is no literal derivation edge?

Examples:

- same measurement specification;
- same sampling frame;
- same identification strategy;
- same calibration standard;
- same outcome policy;
- same evaluation protocol;
- same instrument family/profile;
- same model weights;
- same external database;
- same learned grammar or retrieval model.

These dependencies may connect artifacts that were produced separately.

### 3.3 Replication / triangulation assessment

Question: Given exact target-compatible contributions and sufficiently complete dependency inventories, what can be said about shared foundations, declared disjointness, methodological diversity, and replication scope?

This is a derived assessment layer. It must not be stored as a producer assertion.

---

## 4. Exact evidence target first

Two contributions must not be aggregated merely because they sound related.

Before dependency or replication comparison, bind an exact scientific target such as:

- proposition identity;
- causal estimand;
- forecast target;
- measurement claim;
- formal theorem target;
- model-performance claim;
- simulation claim;
- bounded method-validation target.

The safe default is exact target identity.

A later target-compatibility receipt may permit explicitly scoped comparisons such as equivalent, transformable, nested, related-but-not-transferable, or incomparable.

Target compatibility itself is not evidence independence.

---

## 5. Evidence contribution

A scientific contribution is not merely a paper or result row.

A future common envelope may include:

- contribution identity;
- exact target identity;
- contribution relation to target, such as support/opposition/falsification/replication attempt;
- exact evidence/observation identities;
- exact SCI-003 execution identities where applicable;
- exact SCI-004 contract/preregistration identities where applicable;
- exact SCI-005 exposure/use eligibility lineage where applicable;
- dependency inventory identity;
- assumptions/limitations;
- provenance/source metadata.

The contribution envelope itself does not decide independence or scientific disposition.

---

## 6. Dependency domains

SCI-006 should define a small extensible common vocabulary plus domain-owned extension namespaces.

A useful common baseline includes:

- `SourceData`;
- `SourceVintage`;
- `SamplingFrame`;
- `SamplingDesign`;
- `MeasurementSpecification`;
- `InstrumentOrCalibrationProfile`;
- `TransformationArtifact`;
- `FeatureConstructionArtifact`;
- `ModelArtifact`;
- `ModelTrainingData`;
- `EstimatorArtifact`;
- `IdentificationStrategy`;
- `ExecutionCapsule`;
- `AnalysisImplementation`;
- `OutcomeObservationPolicy`;
- `EvaluationProtocol`;
- `DecisionCriteria`;
- `LearnedGrammarArtifact`;
- `RetrievalOrEmbeddingArtifact`;
- `ExternalToolOrDatabase`;
- `PriorEvidenceOrTheoryArtifact`;
- `HumanOrAgentDecisionProcess` when scientifically relevant;
- `DomainSpecific(namespace, profile)`.

These categories are not a strength ranking.

A dependency in one category does not necessarily matter in the same way as a dependency in another.

---

## 7. Exact dependency identity

Each dependency entry should bind an exact SCI-002 artifact/semantic identity or another exact versioned scientific identity.

Friendly labels such as `AME2020`, `RandomForest`, `Z3`, `same survey`, or `same lab protocol` are insufficient as canonical dependency identity.

When exact underlying identity is unavailable, the inventory should retain the strongest honest reference state and mark coverage incomplete rather than fabricate precision.

---

## 8. Dependency relation kinds

A future dependency graph may need relation kinds such as:

- `DerivedFrom`;
- `TransformedFrom`;
- `MeasuredBy`;
- `CalibratedBy`;
- `SampledFrom`;
- `TrainedOn`;
- `FittedOn`;
- `CalibratedOn`;
- `SelectedUsing`;
- `EvaluatedOn`;
- `UsesModel`;
- `UsesEstimator`;
- `UsesIdentificationStrategy`;
- `UsesExecutionCapsule`;
- `UsesOutcomePolicy`;
- `UsesEvaluationProtocol`;
- `UsesDecisionCriteria`;
- `LearnedFrom`;
- `RetrievedFrom`;
- `Assumes`;
- `DependsOn` as a deliberately generic fallback only when a more precise relation is unavailable.

Relation kinds should be versioned/domain-separated.

A graph edge is a declared dependency relation, not proof that the scientific relation is causally important or exhaustive.

---

## 9. Inventory scope and completeness

“No shared dependencies found” is meaningful only relative to a declared inventory scope.

A future `DependencyInventoryScopeV1` should identify:

- dependency domains required for comparison;
- target/use profile;
- depth/closure requirements;
- treatment of unknown references;
- domain-specific required categories;
- whether transitive dependencies must be expanded;
- evidence required to claim coverage complete within each category.

An inventory should separately represent coverage per domain.

Possible states include:

- complete within declared scope;
- partial;
- unknown;
- not applicable under the exact profile;
- externally asserted but not verified.

Do not encode one global `complete: bool` if completeness differs by dependency domain.

---

## 10. Pairwise lineage relation

For two target-compatible contributions under an exact comparison scope, the strongest generic relation should remain conservative.

Illustrative outcomes:

- `SameScientificLineage`;
- `DirectOrTransitiveDerivation`;
- `SharedDeclaredDependencies`;
- `DeclaredDisjointWithinScope`;
- `IncompleteInventory`;
- `TargetCompatibilityNotEstablished`;
- `ComparisonProfileMismatch`.

Known overlap should dominate missing metadata: if one shared dependency is known, the pair is not declared-disjoint merely because other categories remain incomplete.

If no overlap is known but required inventory coverage is incomplete, the result is `IncompleteInventory`, not disjointness.

---

## 11. Why the generic layer should not emit `Independent`

RCA can derive an `Independent` relation inside one fully validated evidence-root graph under its exact semantics. That is a valid domain/governance theorem.

SCI-006 operates at a broader scientific dependency layer where undeclared external dependencies may remain possible.

Therefore the generic strongest default no-overlap conclusion is:

`DeclaredDisjointWithinScope`.

This does not weaken RCA. RCA's stronger local result may remain a typed input to a higher replication assessment.

The shared kernel should not rename a scoped disjointness result to universal scientific independence.

---

## 12. Statistical independence is separate

Lineage disjointness and statistical independence answer different questions.

Two studies can use disjoint datasets and implementations while sharing a common latent selection mechanism that makes their errors statistically correlated.

Conversely, statistically independent random samples can intentionally share the same measurement instrument, estimator implementation, or protocol.

Therefore SCI-006 should never infer statistical independence solely from provenance topology.

If statistical independence assumptions matter, they belong to the domain's statistical model/diagnostics and evidence lineage.

---

## 13. Pairwise independence is not transitive

Suppose:

- A shares dataset D with B;
- B shares estimator E with C;
- A and C have no direct declared overlap.

A/B/C form one known dependency-connected component, but that does not imply every pair shares the same dependency or should receive identical evidence weight.

SCI-006 should retain both:

- exact pairwise relations;
- conservative connected components for anti-double-counting / navigation.

A connected component is not a proof that all members are scientifically equivalent or pairwise dependent in the same way.

---

## 14. Dependency components

A future audit may compute known dependency components using edges that establish shared or derived lineage.

Useful outputs include:

- contribution identities;
- pairwise relations;
- shared dependency identities/categories;
- connected components;
- incomplete-inventory flags;
- exact target identity;
- comparison-scope identity.

There should be no default `independent_count`, `replication_score`, or weighted evidence total.

Component cardinality is descriptive topology, not a scientific confidence score.

---

## 15. Duplicate publication / representation protection

The same scientific lineage may appear as:

- preprint and journal article;
- conference and extended paper;
- replicated table in a review;
- duplicated repository result;
- reformatted evidence package;
- multiple agent summaries of one source;
- multiple generated critiques of one observation.

SCI-006 should retain contribution/publication identities for audit while deduplicating the underlying scientific lineage when exact lineage identity matches.

Conflicting dependency inventories under the same canonical lineage identity should fail closed rather than silently select one.

---

## 16. Different authors/institutions are metadata, not independence oracles

Social/organizational separation can matter, especially for collusion, shared incentives, operational replication, or external validation.

But scientific dependency independence cannot be inferred merely from:

- different author names;
- different institutions;
- different countries;
- different journals;
- different companies;
- different AI agents.

Those actors may still share the same data, code, methods, assumptions, instrument calibration, upstream database, or model weights.

SCI-006 may retain organizational provenance as another typed dimension, but it must not substitute for scientific dependency analysis.

---

## 17. Different implementations are not automatically independent

Two implementations may share:

- the same algorithm;
- generated code;
- common library;
- common reference implementation;
- same model weights;
- same training data;
- same compiler bug;
- same prompt/model;
- same solver backend;
- same preprocessing artifact.

Implementation diversity is useful evidence about some failure modes but should remain distinct from data/method/interpretation independence.

A future triangulation assessment may explicitly value implementation diversity without relabeling it independent replication.

---

## 18. Methodological diversity / triangulation

SCI-006 should support multi-dimensional triangulation rather than one replication scalar.

Two contributions may be:

- shared data + different estimators;
- different data + same measurement spec;
- different measurement modalities + same target;
- different identification strategies + overlapping data;
- simulation + empirical observation;
- formal proof + numerical evidence;
- different instruments + same calibration standard.

These configurations provide different kinds of corroboration and robustness.

A future `TriangulationAssessmentV1` should expose the dependency/diversity structure rather than rank every pair on one axis.

Methodological diversity does not erase shared dependencies.

---

## 19. Replication classes are separate from independence

A replication attempt may target different goals such as:

- exact/direct replication;
- computational reproduction;
- robustness replication;
- methodological replication;
- conceptual replication;
- cross-population replication;
- cross-instrument replication;
- independent external replication.

The exact taxonomy should be domain/profile controlled.

The replication class describes intended relationship to the original protocol/claim. Dependency topology separately describes what foundations are shared.

Thus an exact computational reproduction can deliberately share nearly everything, while an external conceptual replication may differ widely in method but still share key source data or theory dependencies.

---

## 20. Prospective cleanliness is separate from replication independence

SCI-005 determines whether evidence was prospectively eligible for its own declared use.

SCI-006 determines what that evidence shares with other contributions.

Therefore:

- a prospectively clean replication can still share major dependencies;
- a highly independent implementation can still be post-hoc against already revealed target outcomes.

A trustworthy replication assessment needs both axes.

---

## 21. Target compatibility and evidence transfer

Evidence aimed at different targets must not be merged merely because dependency graphs overlap.

Examples:

- ATE vs ATT;
- 3-month vs 12-month effect;
- one isotope vs neighboring isotope;
- one proposition semantic revision vs its predecessor;
- one forecast horizon vs another;
- one operationalization vs another when operationalization is target-constitutive.

SCI-006 should require exact target identity or a qualified target-compatibility receipt before cross-target aggregation/triangulation.

A compatibility receipt may say that evidence is transformable or informative without claiming targets are identical.

---

## 22. Dependency closure and learned information

SCI-006 is where indirect contamination from SCI-005 should become machine-auditable.

Examples:

- revealed outcomes -> trained model -> later predictions;
- prior discovery -> learned grammar -> later conjecture;
- post-cutoff papers -> embeddings/model -> historical discovery run;
- calibration set -> chosen threshold -> later evaluation policy;
- shared raw source -> differently transformed datasets;
- common instrument calibration -> nominally separate laboratories.

A lineage inventory that stops at the immediately visible artifact may be incomplete even if every listed direct parent is exact.

Profiles should state required transitive closure depth/semantics.

---

## 23. Negative, null, refuted, and failed evidence still has dependencies

Dependency analysis applies regardless of whether a contribution supports or opposes a target.

A failed replication, null result, refutation, protocol deviation, or incomplete execution still has provenance and may share dependencies with other results.

Do not drop dependency lineage simply because the scientific outcome was unfavorable.

This is important for publication-bias and falsification analysis later.

---

## 24. Lifecycle / supersession / retraction

Dependency identity is historical and should not be rewritten when an artifact is later superseded, retracted, invalidated, or made ineligible for a particular use.

A retraction/lifecycle layer should point to immutable contribution/dependency identities.

A retracted source does not cease to have been an ancestor of downstream work.

Therefore:

- current eligibility is not historical existence;
- supersession is not deletion;
- retraction is not proposition negation;
- dependency ancestry survives lifecycle changes.

SCI-014 / evidence-lifecycle work can consume this invariant.

---

## 25. Canonical graph identity

A scientific dependency graph/generation should eventually receive a SCI-002 composite semantic identity.

Identity should bind:

- graph/profile schema;
- target/comparison scope;
- every contribution identity;
- every dependency identity/category;
- every relation kind/direction;
- coverage/completeness state;
- canonical ordering rules;
- domain-extension profile identities.

Producer labels or incidental JSON ordering must not define the canonical graph identity.

RCA #578 is the leading precedent for serializer-independent complete-graph generation identity.

---

## 26. Closed vs open-world inventories

Some dependency graphs can be structurally closed because the scientific system owns all relevant artifacts under a narrow contract.

Other real-world scientific inventories are necessarily open-world.

SCI-006 must distinguish these cases.

A closed-world profile may require every parent/dependency to appear in the graph and reject unknown nodes.

An open-world profile may permit external references but cannot use absence of a listed edge as proof of no dependency.

The comparison result must reflect that difference.

---

## 27. Evidence for inventory completeness

A future positive replication-independence authority should require evidence about inventory completeness, not only syntactic inventory presence.

Possible mechanisms include:

- qualified provenance capture from controlled pipelines;
- independently audited manifests;
- signed/attested external execution/source records;
- institutional declarations with explicit limited scope;
- reproducible build/execution closure;
- source-data custody records;
- model/tool provenance attestations;
- explicit unknown-dependency declarations.

SCI-006 does not define one universal authority for completeness.

It defines where that authority would be consumed.

---

## 28. Proposed shared vocabulary

Names are illustrative; semantics are normative.

- `ScientificDependencyDomainV1`;
- `ScientificDependencyRefV1`;
- `ScientificDependencyRelationV1`;
- `DependencyInventoryScopeV1`;
- `DependencyInventoryCoverageV1`;
- `ScientificDependencyInventoryV1`;
- `ScientificEvidenceContributionV1`;
- `EvidenceDependencyGraphV1`;
- `EvidenceLineageRelationV1`;
- `KnownDependencyComponentV1`;
- later `TriangulationAssessmentV1`;
- later `ReplicationAssessmentV1`;
- later `QualifiedReplicationIndependenceV1` if a sufficiently strong external/closed-world authority exists.

The first shared implementation should not introduce the last three positive assessment types.

---

## 29. First implementation tranche

Start with non-authorizing inventory and topology only.

Suggested slice:

- `ScientificDependencyDomainV1`;
- `ScientificDependencyRefV1`;
- `DependencyInventoryScopeV1`;
- `DependencyInventoryCoverageV1`;
- `ScientificDependencyInventoryV1`.

Required properties:

1. exact SCI-002 dependency identities;
2. explicit target/use/scope;
3. per-domain coverage state;
4. duplicate/conflicting dependency rejection;
5. no `independent: bool`;
6. no replication count/score;
7. no universal completeness claim;
8. closed schema/versioning;
9. domain-specific extension mechanism;
10. no scientific/action authority.

---

## 30. Second implementation tranche

Add exact pairwise comparison and conservative graph topology:

- `SameScientificLineage`;
- `DirectOrTransitiveDerivation`;
- `SharedDeclaredDependencies`;
- `DeclaredDisjointWithinScope`;
- `IncompleteInventory`;
- target/profile mismatch states;
- known dependency components.

Still do not issue `IndependentReplication`.

Economic Science is the strongest candidate pilot because it already implements almost exactly this theorem.

The pilot must preserve its stronger domain-specific vocabulary and qualification status rather than importing it by name.

---

## 31. Third implementation tranche

Integrate RCA closed ancestry as one typed dependency source.

The adapter should preserve:

- RCA's closed validated DAG;
- exact root sets;
- `SameEvidence` / `Derived` / `SameRoot` / `PartiallyShared` / local `Independent` semantics;
- canonical graph generation identity.

The generic SCI-006 projection may conservatively report local RCA independence as `DeclaredDisjointWithinScope` unless a higher replication policy explicitly accepts the stronger closed-world theorem for its use.

This prevents semantic weakening in RCA while preventing its local vocabulary from being overgeneralized to all science.

---

## 32. Fourth implementation tranche

Only after target compatibility, prospective eligibility, inventory completeness, and domain replication semantics have qualified should a higher `ReplicationAssessmentV1` be attempted.

Its output should be multidimensional, retaining at least:

- replication class/use;
- target relation;
- prospective status;
- shared/disjoint dependency domains;
- incomplete domains;
- implementation diversity;
- measurement diversity;
- data/source diversity;
- interpretation/analysis diversity;
- organizational provenance where relevant;
- known connected components;
- limitations.

Do not reduce this to one score in v1.

---

## 33. Adversarial requirements

Future implementations should include at least the following.

### Counting attacks

- five papers sharing one dataset cannot become five independent replications;
- five AI agents summarizing one observation cannot become five confirmations;
- preprint + journal version of same lineage deduplicates scientifically while retaining publication metadata;
- same lineage identity with conflicting inventory fails closed.

### Dependency attacks

- different model artifacts sharing training data are recognized as shared-dependency lineages;
- different datasets sharing one transformation/model/calibration dependency retain that overlap;
- learned grammar/model carrying prior target evidence is represented as dependency;
- renamed/copied/transformed data does not imply disjoint lineage;
- one known overlap dominates otherwise incomplete inventories.

### Incomplete inventory attacks

- no known overlap + incomplete required domain returns `IncompleteInventory`;
- unknown external references cannot be silently treated as disjoint;
- open-world inventory absence of edge is not proof of independence;
- per-domain completeness cannot be laundered through one global boolean.

### Target attacks

- different target identities cannot be aggregated without compatibility receipt;
- related-but-not-transferable targets cannot share replication count;
- target compatibility does not change dependency relation.

### Topology attacks

- pairwise no-overlap is not inferred transitively;
- connected-component membership does not overwrite exact pairwise relation;
- duplicate edges/nodes fail or canonicalize under explicit rules;
- graph cycles are rejected where the selected relation semantics require a DAG.

### Social/implementation attacks

- different author/institution does not mint independence;
- different implementation does not mint independence;
- same institution does not automatically prove dependence if scientific dependencies are otherwise disjoint; social provenance remains a separate dimension.

### Authority attacks

- serialized `DeclaredDisjointWithinScope` cannot become `IndependentReplication` by deserialization;
- producer cannot set `independent=true`;
- dependency graph cannot grant scientific disposition/action authority;
- replication assessment cannot grant safety/effect authority.

---

## 34. Relationship to SCI-002

Every dependency/contribution/graph generation should bind canonical SCI-002 identities.

Content identity is necessary for exact joins but does not imply independence.

Different digests do not establish different information ancestry.

---

## 35. Relationship to SCI-003

Execution capsules and verifier/solver/program identities are dependency domains.

Two studies using different input data but the same buggy execution/runtime implementation may share an important failure mode.

SCI-003 execution identity therefore becomes one dependency axis, not an automatic independence criterion.

---

## 36. Relationship to SCI-004

Experiment contracts, outcome policies, analysis plans, thresholds, and stopping/adaptation policies can be scientific dependencies.

Exact protocol replication intentionally shares many SCI-004 dependencies.

That is not a flaw; SCI-006 should describe the shared structure honestly.

---

## 37. Relationship to SCI-005

SCI-005 exposure/use lineage is a dependency source.

A model, grammar, analyst, or benchmark may carry prior target information into a supposedly fresh campaign.

SCI-006 provides the transitive graph machinery needed to detect those indirect paths when sufficiently inventoried.

---

## 38. Relationship to SCI-011

Learned grammar/macro artifacts should be explicit dependency nodes linked to their originating discoveries, datasets, and verification lineage.

This makes macro-assisted rediscovery visible as shared ancestry rather than independent rediscovery.

---

## 39. Relationship to SCI-014 Theory Atlas

Theory Atlas should consume SCI-006 topology when presenting support/opposition/replication.

It should show:

- exact evidence contributions;
- shared dependency clusters/components;
- declared-disjoint relationships;
- incomplete inventories;
- methodological diversity;
- replication attempts and their scopes.

It should not display publication count as replication count.

---

## 40. Dependency order

SCI-001 audit -> SCI-002 artifact identity -> SCI-003 execution capsule -> SCI-004 experiment contract -> SCI-005 exposure/use separation -> SCI-006 dependency graph -> SCI-007 claim/falsifier graph and later replication/triangulation/disposition layers.

SCI-006 architecture can be reviewed while RCA/economics implementations remain draft. No qualification transfers from them.

---

## 41. Review boundary

Review SCI-006 on this question:

> Does this contract preserve enough exact lineage, dependency-domain coverage, target compatibility, and incomplete-inventory information to prevent evidence multiplicity from becoming fake replication, while still representing useful methodological diversity and triangulation without reducing them to one independence score?

A positive architecture review does not establish that any current pair of studies, agents, models, datasets, or publications are independently replicated.