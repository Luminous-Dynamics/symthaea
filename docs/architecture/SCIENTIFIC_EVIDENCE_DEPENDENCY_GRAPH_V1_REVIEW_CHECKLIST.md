# SCI-006 Review Checklist — Scientific Evidence Dependency Graph v1

**Status:** review aid only; non-authorizing; non-qualifying.

## A. Core anti-counting theorem

- [ ] Publication count is distinct from evidence-lineage count.
- [ ] Evidence-object count is distinct from independent observation count.
- [ ] Model/implementation count is distinct from independent replication.
- [ ] Author/institution/agent count is not an independence oracle.
- [ ] No producer-supplied `independent: bool` is authoritative.
- [ ] No default `independent_count`, `replication_count`, or scalar replication score exists.

## B. Graph separation

- [ ] Derivation ancestry is represented distinctly from broader scientific dependency inventory.
- [ ] Replication/triangulation assessment is derived from target-compatible lineage and inventory rather than stored as producer assertion.
- [ ] Open-world and closed-world graph semantics are explicit.
- [ ] Closed ancestry does not automatically imply complete real-world scientific dependency inventory.

## C. Exact target binding

- [ ] Contributions bind exact proposition/estimand/claim target identity.
- [ ] Similar prose is not enough for aggregation.
- [ ] Cross-target comparison requires an explicit compatibility relation/receipt.
- [ ] Target compatibility does not imply evidence independence.
- [ ] Related-but-not-transferable targets cannot be pooled as replications.

## D. Dependency domains

- [ ] Source data/vintage can be represented.
- [ ] Sampling frame/design can be represented.
- [ ] Measurement/instrument/calibration dependencies can be represented.
- [ ] Transform/feature artifacts can be represented.
- [ ] Model/training-data dependencies can be represented.
- [ ] Estimator/identification dependencies can be represented.
- [ ] SCI-003 execution/analysis dependencies can be represented.
- [ ] SCI-004 outcome/evaluation/decision-policy dependencies can be represented.
- [ ] Learned grammar/retrieval/embedding/tool dependencies can be represented.
- [ ] Domain-specific extension namespaces exist without changing common semantics.
- [ ] Dependency categories are not treated as strength rankings.

## E. Exact dependency identity

- [ ] Dependency references use exact SCI-002 or equivalent versioned identity where possible.
- [ ] Friendly names are not canonical identity.
- [ ] Unknown/external references remain explicitly incomplete rather than receiving invented precision.
- [ ] Duplicate/conflicting dependencies fail or canonicalize under explicit rules.

## F. Inventory scope/coverage

- [ ] Comparison binds an exact dependency-inventory scope/profile.
- [ ] Coverage is tracked per relevant dependency domain.
- [ ] Completeness is not reduced to one global boolean when domains differ.
- [ ] `NotApplicable` is distinct from `Unknown`/`Partial`.
- [ ] External assertion of completeness is distinct from verified completeness.
- [ ] Required transitive-closure depth/semantics are profile-defined.

## G. Pairwise relation semantics

- [ ] Same lineage is distinct from shared dependency.
- [ ] Direct/transitive derivation is explicit.
- [ ] Known shared dependency dominates otherwise missing metadata.
- [ ] No known overlap + incomplete required inventory => `IncompleteInventory`.
- [ ] Strongest generic no-overlap wording is `DeclaredDisjointWithinScope`.
- [ ] Generic SCI-006 does not emit universal `Independent` by default.
- [ ] Target/profile mismatch states remain explicit.

## H. Preserve RCA strength without overgeneralization

- [ ] RCA closed-DAG/local `Independent` semantics remain intact in the RCA domain.
- [ ] Generic projection may conservatively map local independence to scoped declared disjointness unless a higher policy accepts RCA's exact theorem.
- [ ] SCI-006 does not rewrite RCA canonical lineage identities.
- [ ] Complete RCA root sets remain available as typed evidence to later replication assessment.

## I. Statistical independence

- [ ] Provenance disjointness is distinct from statistical independence.
- [ ] Shared method does not automatically imply statistically dependent observations.
- [ ] Disjoint data does not automatically imply statistically independent errors.
- [ ] Statistical assumptions remain domain-model/diagnostic concerns.

## J. Pairwise vs component topology

- [ ] Pairwise no-overlap is not treated as transitive.
- [ ] Known dependency connected components retain exact pairwise edges/causes.
- [ ] Component membership does not mean every member shares the same dependency.
- [ ] Component cardinality is descriptive, not confidence/replication score.

## K. Duplicate scientific lineage

- [ ] Preprint/journal/reformatted package can remain multiple publication objects but one scientific lineage where exact identity proves it.
- [ ] Same lineage ID with conflicting inventory fails closed.
- [ ] Multiple summaries/critiques/encodings of one root do not become multiple confirmations.

## L. Social/organizational provenance

- [ ] Different organizations do not mint scientific independence.
- [ ] Same organization does not automatically prove scientific dependence.
- [ ] Organizational separation can remain a separate provenance/operational dimension where relevant.
- [ ] AI-agent identity does not substitute for evidence ancestry.

## M. Implementation diversity

- [ ] Different implementations can still share data, libraries, models, codegen, solver, compiler, prompts, or training lineage.
- [ ] Implementation diversity can be represented as methodological diversity without being labeled independent replication.
- [ ] Shared upstream implementation dependency remains visible.

## N. Triangulation

- [ ] Shared-data/different-method cases can be represented.
- [ ] Different-data/shared-method cases can be represented.
- [ ] Cross-instrument/cross-modality evidence can be represented.
- [ ] Formal/numerical/empirical evidence diversity can be represented.
- [ ] Methodological diversity does not erase shared dependency.
- [ ] No mandatory scalar triangulation score is introduced.

## O. Replication class vs independence

- [ ] Exact computational reproduction can intentionally share most dependencies.
- [ ] Conceptual/methodological replication can differ while still sharing hidden upstream dependencies.
- [ ] Replication intended class is distinct from dependency topology.
- [ ] Prospective cleanliness (SCI-005) is distinct from replication independence (SCI-006).

## P. Indirect information dependencies

- [ ] labels -> trained model -> later result can be represented.
- [ ] prior discovery -> learned grammar -> later conjecture can be represented.
- [ ] post-cutoff sources -> model/embedding -> historical replay can be represented.
- [ ] calibration outcome -> threshold/model selection -> evaluation can be represented.
- [ ] Direct-parent inventory alone is not assumed complete when deeper closure is required.

## Q. Outcome/lifecycle neutrality

- [ ] Positive, negative, null, refuted, failed, or incomplete contributions all retain dependency lineage.
- [ ] Retraction/supersession does not erase historical ancestry.
- [ ] Current eligibility is distinct from historical existence.
- [ ] Lifecycle metadata can point to immutable contribution/dependency identities.

## R. Canonical graph identity

- [ ] Graph identity eventually binds exact target/scope, nodes, dependencies, relation kinds, and coverage states.
- [ ] Producer graph labels do not define canonical scientific identity.
- [ ] Incidental serialization/order does not define graph identity.
- [ ] SCI-002 composite identity/profile semantics are used.

## S. First implementation gate

First tranche remains inventory-only:

- `ScientificDependencyDomainV1`;
- `ScientificDependencyRefV1`;
- `DependencyInventoryScopeV1`;
- `DependencyInventoryCoverageV1`;
- `ScientificDependencyInventoryV1`.

Reject first-tranche introduction of:

- `IndependentReplication`;
- replication scoring/counting;
- scientific disposition;
- action authority.

## T. Second/third tranche discipline

- [ ] Pairwise comparison remains conservative.
- [ ] Economic Science is a candidate pilot without qualification inheritance.
- [ ] RCA adapter preserves stronger closed-world theorem.
- [ ] Higher replication assessment waits for target compatibility, prospective eligibility, and inventory-completeness evidence.

## Review question

> Does SCI-006 prevent publication/model/agent multiplicity from becoming fake replication while retaining the exact dependency topology needed to reason about scoped disjointness, methodological diversity, triangulation, and later replication qualification?
