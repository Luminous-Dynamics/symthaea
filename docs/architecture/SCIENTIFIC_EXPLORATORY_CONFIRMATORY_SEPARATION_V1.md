# Scientific Exploratory / Confirmatory Separation v1

**Status:** architecture contract only; non-authorizing; non-qualifying.

**Series:** SCI-005, stacked on SCI-004.

**Parent:** `architecture/scientific-experiment-contract-v1@fc7d350a7c392b8247e1c716fa6021919653ce58`

## 1. Purpose

SCI-004 defines what a prospective confirmatory experiment contract must freeze. SCI-005 defines the complementary capability boundary: artifacts that have already participated in exploratory, tuning, reveal, or outcome-aware decision processes must not be relabeled as fresh confirmatory evidence merely by changing a field, serializing a new wrapper, or writing a later preregistration document.

Symthaea already contains concrete instances of this rule:

- VART preserves spent confirmatory material as `historical_only_no_tuning` and uses a benchmark firewall for the next campaign;
- Matter separates fit, calibration, and structural holdout roles so evaluation data cannot silently enter fitting;
- Futures separates precommitted outcome semantics from post-reveal resolution choices;
- Physical Agency requires claims and safety obligations to exist before the run whose outcome they judge;
- historical discovery work requires source/model/tool cutoffs so future knowledge cannot enter through a supposedly historical replay;
- learned-grammar work recognizes that a downstream discovery may inherit information from earlier discoveries even if the visible dataset looks new.

The common kernel needs to represent these exposure and use boundaries mechanically while avoiding an overbroad rule such as “an artifact used once can never be used again.” Exposure is contextual: what information was visible, to which decision process, for which target and evidentiary use, matters.

---

## 2. Core theorem

SCI-005 preserves the following distinctions:

- exploratory evidence is not confirmatory evidence;
- a frozen confirmatory contract is not fresh confirmatory evidence;
- hidden data is not necessarily unexposed data;
- unexposed data is not necessarily independent data;
- sealed custody is not proof of no model/tool leakage;
- outcome exposure is not identical to covariate exposure;
- data reuse is not automatically contamination;
- different file bytes are not automatically fresh information;
- the same raw artifact can support distinct scoped questions only when the relevant information exposure and dependency semantics permit it;
- a post-hoc reanalysis can be scientifically valuable without becoming prospective evidence;
- confirmatory failure is evidence and must not be converted into tuning without changing the next campaign lineage;
- replication attempt is not independent replication;
- confirmatory scientific evidence is not action authority.

The central prohibition is:

> An artifact whose relevant outcome information was already available to the decision process for a declared evidentiary scope cannot later mint fresh prospective confirmatory authority for that same scope merely through relabeling, reserialization, re-splitting, or a later timestamp.

---

## 3. Why a boolean is insufficient

A field such as `confirmatory: true` or `held_out: true` cannot encode the required semantics.

A defensible boundary must answer at least:

- Which exact information artifact was involved?
- Which projection of that artifact was visible?
- To which human/model/tool/agent decision process was it visible?
- For what target proposition or hypothesis family?
- For what use: fitting, tuning, calibration, model selection, threshold selection, evaluation, adjudication, replication, or publication?
- When did the exposure occur relative to contract freeze and outcome reveal?
- Did derived artifacts, embeddings, summaries, learned grammar, model weights, prompts, or caches carry equivalent information?
- Was access controlled or merely declared hidden?

Therefore SCI-005 should derive prospective eligibility from explicit lineage and exposure records rather than accept a producer assertion.

---

## 4. Four distinct concepts

### 4.1 Data / artifact identity

SCI-002 identifies the exact raw or semantic artifact.

Identity answers what the artifact is. It does not answer whether it was seen or how it was used.

### 4.2 Information projection

A decision process may see only part of an artifact.

Examples include:

- features/covariates without labels;
- blinded labels without semantic names;
- aggregate metrics without case-level outcomes;
- model outputs without hidden ground truth;
- an encrypted or committed artifact whose bytes are not readable;
- a derived summary that reveals some but not all outcome information.

Therefore exposure must bind a declared information projection, not only the parent file identity.

### 4.3 Exposure event

An exposure event records that a particular decision process obtained a particular information projection for a declared purpose/context.

Exposure is historical provenance. It should be append-only.

### 4.4 Prospective evidence eligibility

Prospective eligibility is a derived, scoped conclusion that the exact evidence input remained admissible for a declared confirmatory contract/use under the relevant exposure/custody/dependency policy.

Eligibility is not a property stored permanently on the data artifact itself.

---

## 5. Information projection

A future shared vocabulary should support domain-owned projection semantics while retaining a small common envelope.

Illustrative projection classes include:

- raw bytes;
- input features/covariates;
- treatment/assignment information;
- labels/outcomes;
- timestamps/vintages;
- case membership;
- aggregate statistics;
- score only;
- gradients/loss feedback;
- model-selection feedback;
- semantic annotations;
- source metadata;
- transformed/derived representation;
- unknown projection.

These are not universally equivalent across domains.

For example, revealing input features may be harmless in one fixed-prediction benchmark but may leak the answer in a dataset where feature construction embeds labels. The domain profile determines which projections are decision-relevant.

Unknown projection should fail closed for strong prospective eligibility when the missing information matters to the contract.

---

## 6. Decision-process identity

Exposure is relative to the process capable of adapting decisions.

A decision process may include:

- human analyst or research team;
- model weights/training pipeline;
- hyperparameter optimizer;
- automated scientific agent;
- retrieval system;
- learned grammar/macro store;
- prompt/context builder;
- benchmark leaderboard feedback loop;
- institutional pipeline whose downstream operators inherit upstream results.

The kernel should avoid assuming only direct human viewing counts as exposure.

A model trained on hidden outcomes may contaminate later evaluation even if the current human operator never saw those outcomes.

---

## 7. Exposure context

An exposure record should be scoped to the scientific context in which the information could influence a decision.

Suggested coordinates include:

- exact artifact/projection identity;
- decision-process identity;
- target proposition or target-family identity;
- experiment/campaign identity;
- evidentiary use;
- exposure event time/ordering evidence;
- access/custody source;
- derivation dependencies;
- optional reason/purpose.

This prevents two opposite mistakes:

1. Treating every historical access to a dataset as contamination for every future scientific question.
2. Treating the same information as fresh merely because it appears in a new file or under a new experiment name.

---

## 8. Exposure is append-only and monotonic as history

Historical exposure cannot be undone.

A later artifact may become eligible for a new use, but the old exposure event remains part of lineage.

Therefore the system should not implement mutable state transitions such as `Revealed -> Hidden`.

Instead:

- exposure events accumulate;
- a prospective-eligibility evaluator considers the complete relevant exposure lineage;
- a new scientific scope may receive a distinct eligibility conclusion if its semantics genuinely differ.

This preserves historical truth without imposing a universal one-use-only rule.

---

## 9. Suggested exposure event vocabulary

Names are illustrative.

A future common event taxonomy may include:

- `ArtifactRegistered`;
- `ProjectionReleased`;
- `ProjectionAccessed`;
- `OutcomeRevealed`;
- `LabelsRevealed`;
- `ScoreRevealed`;
- `LeaderboardFeedbackObserved`;
- `UsedForFitting`;
- `UsedForCalibration`;
- `UsedForModelSelection`;
- `UsedForThresholdSelection`;
- `UsedForHypothesisGeneration`;
- `UsedForConfirmatoryAdjudication`;
- `UsedForPublicationDecision`;
- `ImportedIntoModelOrMemory`;
- `DerivedArtifactCreated`;
- `CustodyTransferred`;
- `AccessRevoked`;
- `ExposureUnknown`.

`AccessRevoked` affects future access, not historical exposure. It must not erase prior `ProjectionAccessed` events.

---

## 10. Scientific-use classes

SCI-005 should keep use explicit rather than reduce everything to “seen/not seen.”

Illustrative uses:

- hypothesis generation;
- qualitative exploration;
- feature engineering;
- training/fitting;
- calibration;
- hyperparameter selection;
- model selection;
- metric selection;
- threshold selection;
- stopping-rule adaptation;
- confirmatory evaluation;
- replication evaluation;
- method validation;
- retrospective reanalysis;
- publication/reporting selection.

A projection may be admissible for one use and not another.

For example, a calibration set intentionally informs calibration but cannot then be described as untouched structural holdout evidence.

---

## 11. Evidence-role capability types

The strongest design should make scientific role difficult to counterfeit through ordinary data construction.

Illustrative layers:

- `ExploratoryEvidenceV1` — result/evidence explicitly outcome-aware or development-visible;
- `SealedEvidenceReferenceV1` — reference under a declared custody policy, not yet proof of prospective eligibility;
- `ProspectiveEvidenceEligibilityV1` — private-fielded scoped witness issued from exact contract + exposure/custody/dependency evidence;
- `ConfirmatoryObservationCandidateV1` — an observation admitted for one exact confirmatory contract/use;
- later `ConfirmatoryEvidenceContributionV1` — result after measurement/adjudication layers.

There should be no convenience conversion from `ExploratoryEvidenceV1` to `ProspectiveEvidenceEligibilityV1`.

---

## 12. No direct exploratory-to-confirmatory conversion

The following transition must be structurally absent:

`ExploratoryResult -> ConfirmatoryEvidence`

A legitimate future path is instead:

1. exploratory result proposes or refines a hypothesis;
2. a new SCI-004 contract is frozen;
3. preregistration chronology is established;
4. genuinely prospective evidence eligible for that contract is identified;
5. SCI-003 executions occur under the contract;
6. observations/measurements are adjudicated;
7. only those new prospective observations contribute confirmatory evidence.

The exploratory artifact remains a dependency in hypothesis-generation provenance.

---

## 13. Exploratory evidence remains valuable

SCI-005 must not encode “exploratory = bad.”

Exploratory evidence can legitimately support:

- hypothesis generation;
- mechanism discovery;
- visualization;
- model development;
- prior construction when transparently declared;
- experimental design;
- measurement design;
- falsifier design;
- candidate feature discovery;
- future sample-size planning;
- theory refinement;
- retrospective explanation.

The distinction concerns evidentiary authority, not intellectual value.

---

## 14. Exploratory evidence as prior information

A sophisticated confirmatory workflow may intentionally use exploratory results as prior information.

This does not necessarily invalidate prospective evidence from new data.

The required lineage is explicit:

- exploratory evidence contributes to prior/hypothesis/model/design;
- fresh confirmatory evidence contributes through a distinct prospective likelihood/evaluation path;
- the resulting inference retains both dependencies.

The system must not pretend the exploratory prior is independent of itself when the same data are reused in the confirmatory likelihood.

Thus SCI-005 avoids an incorrect rule that “no prior knowledge is allowed.” Science is cumulative; the requirement is honest dependency accounting.

---

## 15. Data reuse and scoped freshness

The same raw artifact can sometimes support more than one scientific question without invalid reuse.

Examples:

- previously visible covariates used with newly unrevealed outcomes;
- a dataset reused for a different target whose relevant labels were never exposed;
- raw sensor bytes reinterpreted under a newly preregistered measurement question when previous analysis did not expose the relevant target information.

But reuse is not automatically valid.

A future eligibility evaluator must consider:

- exact projection previously exposed;
- target overlap/equivalence;
- decision-process memory/model contamination;
- derived artifacts carrying the same information;
- whether analysis choices were tuned using relevant outcomes;
- dependency/independence requirements.

Where equivalence cannot be established, strong prospective eligibility should fail closed or be explicitly scoped as uncertain.

---

## 16. Byte changes do not create fresh information

Changing file representation does not reset exposure.

Examples:

- CSV converted to Parquet;
- relabeled columns;
- reordered rows;
- compressed/decompressed archives;
- screenshots or summaries of the same outcome;
- embeddings generated from revealed labels;
- transformed target values;
- model weights trained on the revealed data.

SCI-002 transformation lineage and SCI-006 dependency analysis should eventually make these relationships explicit.

The key theorem is:

> Different content identity does not imply informational independence or fresh confirmatory eligibility.

---

## 17. Derived-artifact contamination

Exposure can propagate through derived artifacts.

A future dependency graph should be able to represent paths such as:

- hidden labels -> trained model -> predictions;
- revealed benchmark -> hyperparameter choice -> new model;
- prior scientific result -> learned grammar -> downstream conjecture;
- post-cutoff literature -> embedding/model -> historical replay;
- held-out outcomes -> threshold choice -> evaluation policy.

The evaluator should not require literal byte equality to detect contamination.

SCI-005 therefore depends conceptually on SCI-006 for complete dependency analysis, while still defining the role/exposure theorem now.

---

## 18. Hidden benchmark lifecycle

A hidden benchmark should have a lifecycle, not a permanent `hidden: bool`.

Illustrative stages:

- registered under custody;
- sealed for a declared evaluation scope;
- partially exposed;
- score-only exposed;
- fully revealed;
- historical/spent for that declared scope;
- retained as development data for future work;
- superseded by a fresh benchmark generation.

Once relevant outcomes have been revealed to the decision process, the same benchmark generation cannot return to `sealed fresh` for the same scope.

A fresh confirmatory campaign normally requires fresh eligible evidence or a new independent sample/generation, not merely a renamed split.

---

## 19. Leaderboard and repeated-query leakage

A benchmark can leak information without revealing labels directly.

Repeated score queries can enable adaptation to the hidden set.

Therefore an exposure policy may need to bind:

- query count/budget;
- score precision;
- subgroup feedback;
- per-case feedback;
- public/private leaderboard semantics;
- adaptation allowed between queries;
- holdback set policy;
- final one-shot confirmation set.

Score-only exposure is not automatically equivalent to no exposure.

This is important for autonomous systems that can optimize rapidly against weak feedback.

---

## 20. Human and machine memory

A future confirmatory campaign cannot rely solely on file permissions if the relevant outcomes have already entered persistent human or model memory.

Potential mitigations include:

- fresh independent evidence;
- independent analysts/models without relevant exposure;
- sealed evaluation services that return bounded adjudication;
- new benchmark generations;
- explicit historical-reanalysis status;
- contamination-aware statistical designs.

SCI-005 does not claim memory erasure is verifiable in general.

Where prior exposure is known and cannot be removed, the architecture should record it rather than pretend the decision process is fresh.

---

## 21. Model/tool provenance and historical replay

For historical discovery benchmarks, visible literature cutoff is insufficient.

Prospective historical eligibility must also consider:

- model training cutoff;
- fine-tuning data;
- retrieval corpus cutoff;
- embedding/model weights;
- learned grammar/macros;
- tools or databases that encode post-cutoff facts;
- prior benchmark runs/results.

A historical replay is contaminated if future information reaches the decision process through any dependency channel relevant to the target.

This connects SCI-005 directly to SCI-011 and SCI-015.

---

## 22. Confirmatory failure and tuning

A failed/null confirmatory experiment is valid evidence for that campaign.

It may also motivate future changes, but once its outcome influences tuning, the tuned system belongs to a new development lineage.

A clean next confirmatory test requires a new prospective contract and eligible evidence.

Therefore:

- confirmatory failure -> retained evidence;
- confirmatory failure -> may inform exploration/development;
- tuned successor -> new model/method identity;
- tuned successor -> cannot reuse the same spent evaluation evidence as fresh confirmation for the affected scope without an explicit design that supports such reuse.

This prevents iterative benchmark overfitting from masquerading as repeated confirmation.

---

## 23. Cross-validation and resampling

Cross-validation is not automatically exploratory or confirmatory.

Its evidentiary meaning depends on the prospectively frozen design.

A confirmatory cross-validation protocol may be valid when:

- folds/partition policy are fixed prospectively;
- all model selection/tuning semantics are frozen or nested correctly;
- reported metric/aggregation policy is frozen;
- no fold outcome is used to alter the protocol outside allowed adaptation;
- the claim scope matches the resampling design.

Nested cross-validation may separate tuning and outer evaluation within one preregistered design.

SCI-005 should therefore avoid simplistic “each datum can only be viewed once” rules.

---

## 24. Sequential and adaptive experiments

SCI-004 permits prospectively specified adaptive experiment selection. SCI-005 must track exposure through that loop.

At each step:

- observations allowed by the adaptive policy become visible;
- planner state updates under the frozen algorithm;
- future actions may depend on prior observations;
- the final evidentiary interpretation is that of the preregistered adaptive design, not a collection of independent static tests.

The same observed data cannot then be detached from the campaign and presented as a fresh one-shot confirmation for a neighboring policy without new qualification.

---

## 25. Replication attempts

Replication has two separate questions:

1. Was the replication prospectively conducted relative to its own outcomes?
2. How independent is it from the original evidence lineage?

SCI-005 addresses the first through exposure/use eligibility.

SCI-006 addresses the second through dependency/independence analysis.

A team can run a prospectively clean replication that still shares major dependencies with the original study. Conversely, a methodologically independent implementation can still be post-hoc with respect to revealed target outcomes.

Do not collapse these axes.

---

## 26. Publication and selective reporting

Exposure may occur not only during model fitting but during decisions about what to report.

A reporting policy should make it possible to retain:

- all preregistered primary outcomes;
- null/negative/indeterminate results;
- failed runs;
- protocol deviations;
- declared exclusions;
- exploratory follow-ups clearly marked as such.

Selecting only favorable analyses after reveal is an exploratory/publication-selection dependency and must not inherit naive confirmatory authority.

---

## 27. Proposed shared vocabulary

Names are illustrative; semantics are normative.

- `InformationProjectionV1`
- `DecisionProcessIdentityV1`
- `EvidenceUseV1`
- `ExposureContextV1`
- `ExposureEventV1`
- `ExposureLedgerV1`
- `ExploratoryEvidenceV1`
- `SealedEvidenceReferenceV1`
- `ProspectiveEvidenceEligibilityV1`
- `ConfirmatoryObservationCandidateV1`
- `BenchmarkGenerationV1`
- `BenchmarkExposurePolicyV1`
- `HistoricalEvidenceUseV1`

The common layer should remain small; domains own projection semantics and policies that decide which exposures matter for which claims.

---

## 28. Positive eligibility must be derived

`ProspectiveEvidenceEligibilityV1` should be a private-fielded, scoped issued witness.

A future issuer/qualifier should consume at least:

- exact SCI-004 contract/preregistration identity;
- exact artifact and information-projection identity;
- exact exposure ledger/current relevant history;
- decision-process identity;
- declared evidence use;
- custody/reveal evidence where required;
- dependency-analysis evidence where required;
- policy/profile identity.

The positive wrapper should not be directly deserializable into live authority.

Serialized records are audit material; current eligibility must be re-derived from the relevant lineage/policy generation.

---

## 29. Failure states should be explicit

Do not reduce eligibility to `Option<bool>`.

Illustrative non-positive outcomes include:

- `EligibleWithinDeclaredScope`;
- `RelevantOutcomePreviouslyExposed`;
- `UsedForDevelopmentOrTuning`;
- `BenchmarkGenerationSpent`;
- `TargetOverlapUnresolved`;
- `ProjectionSemanticsUnknown`;
- `DecisionProcessExposureUnknown`;
- `DerivedContaminationDetected`;
- `CustodyNotEstablished`;
- `ChronologyNotEstablished`;
- `DependencyClosureIncomplete`;
- `PolicyMismatch`.

Exact vocabulary should be profile-specific enough to preserve meaning.

---

## 30. No evidence laundering through serialization

A persisted object containing fields such as:

- `held_out = true`;
- `fresh = true`;
- `confirmatory = true`;
- `unseen = true`;
- `independent = true`;

must not recreate prospective authority by itself.

These may be descriptive/audit fields only.

Positive eligibility comes from current validated ancestry and policy.

---

## 31. No evidence laundering through branching/copying

Copying a benchmark or artifact to a new branch/repository/path does not create a new evidence generation.

A new benchmark generation must differ in the scientific evidence source/sampling lineage or another explicitly qualified freshness mechanism—not merely in Git identity.

This applies equally to:

- copied files;
- regenerated archives;
- reordered datasets;
- derived representations;
- equivalent mirrors.

SCI-002 identity and SCI-006 lineage should expose the relationship.

---

## 32. First implementation tranche

The first Rust tranche should be non-authorizing with respect to actual confirmatory evidence.

Suggested slice:

- `InformationProjectionV1` envelope;
- `DecisionProcessIdentityV1`;
- `EvidenceUseV1`;
- `ExposureEventV1`;
- `ExposureLedgerV1`.

Required properties:

1. append-only exposure history;
2. exact SCI-002 artifact/projection references;
3. exact target/use/process context;
4. no mutable `hidden/fresh` authority field;
5. no prospective-eligibility constructor yet;
6. no confirmatory evidence issuance yet;
7. closed schema and persistence revalidation;
8. explicit unknown/incomplete exposure semantics.

This tranche records history but grants no positive prospective authority.

---

## 33. Second implementation tranche

Pilot `ProspectiveEvidenceEligibilityV1` for one narrow benchmark/campaign with existing custody/firewall semantics.

VART-style hidden benchmark generation is a strong candidate because it already distinguishes historical spent material from future development/confirmation and has explicit benchmark-firewall concepts.

The pilot must not transfer VART qualification into the shared kernel.

---

## 34. Third implementation tranche

Add one exact SCI-004 + SCI-005 + SCI-003 join:

- qualified preregistration receipt;
- prospective evidence eligibility witness;
- exact execution receipt;
- exact observation identity.

The output should only be a `ConfirmatoryObservationCandidateV1` or equivalently scoped artifact.

Measurement validity and final scientific adjudication remain later boundaries.

---

## 35. Adversarial requirements

Future implementation should include at least the following.

### Relabeling attacks

- exploratory result + `confirmatory=true` cannot mint eligibility;
- copied/renamed hidden dataset cannot become fresh;
- transformed/recompressed equivalent outcomes retain contamination ancestry;
- serialization/deserialization cannot reset exposure history.

### Exposure attacks

- previously revealed labels fail fresh eligibility for the same scoped target/use;
- score-only repeated-query adaptation is represented as exposure;
- training a model on hidden labels contaminates later evaluation through model ancestry;
- learned grammar derived from target outcomes is visible to dependency analysis;
- unknown decision-process exposure fails closed where required.

### Scope precision

- covariate-only exposure need not equal outcome exposure when the profile explicitly distinguishes them;
- genuinely different target/use can receive a separate eligibility assessment rather than global permanent rejection;
- target equivalence/overlap uncertainty cannot be silently treated as disjointness.

### Campaign lifecycle

- confirmatory failure remains evidence;
- tuning after confirmatory result creates a new development lineage;
- spent benchmark generation cannot become sealed-fresh again for the same scope;
- fresh independent generation can become eligible under a new contract.

### Adaptive/resampling designs

- preregistered nested CV remains admissible under its declared scope;
- post-hoc fold/metric/model selection cannot inherit confirmatory status;
- preregistered adaptive experiment selector may react to allowed observations;
- manual actions outside the frozen adaptation policy are protocol deviations.

### Authority

- prospective eligibility does not imply measurement validity;
- prospective eligibility does not imply independent replication;
- confirmatory observation does not imply scientific truth;
- no SCI-005 object grants action/effect authority.

---

## 36. Relationship to SCI-002

SCI-005 uses SCI-002 identities for artifacts and derived representations.

SCI-002 alone cannot establish fresh information or exposure independence.

Different digests are not evidence of informational independence.

---

## 37. Relationship to SCI-003

SCI-003 identifies the decision/execution machinery that may receive information and produce outputs.

Exposure lineage may therefore refer to exact model/tool/execution artifacts from SCI-003.

A clean input set with a contaminated model can still violate prospective eligibility.

---

## 38. Relationship to SCI-004

SCI-004 defines what was prospectively frozen.

SCI-005 defines whether the evidence and decision process remained prospectively eligible for that exact contract/use.

Neither substitutes for the other.

A perfect frozen contract applied to already-revealed target outcomes is not a prospective confirmation.

---

## 39. Relationship to SCI-006

SCI-005 records and evaluates exposure/use history.

SCI-006 should provide the more general dependency graph needed to detect shared ancestry, derived contamination, and replication dependence across artifacts, models, transformations, measurements, and analyses.

Positive SCI-005 eligibility may eventually require a sufficiently complete SCI-006 dependency closure for the profile being evaluated.

---

## 40. Relationship to SCI-011

Learned grammar is a persistent decision-process dependency.

If a macro/operator was learned from prior target-relevant evidence, downstream discovery inherits that exposure/dependency even when the macro contains no literal copy of the source data.

SCI-011 should issue the exact grammar lineage needed for SCI-005/006 to reason about that contamination.

---

## 41. Relationship to SCI-015

Historical discovery benchmarks require an especially strict SCI-005 profile.

Historical eligibility should cover not only visible source corpus cutoff but model/tool/retrieval/grammar prior knowledge and previous benchmark exposure.

The strongest safe conclusion is scoped historical eligibility under the declared dependency/exposure inventory, not proof that the system had no possible future knowledge.

---

## 42. Dependency order

SCI-001 audit -> SCI-002 artifact identity -> SCI-003 execution capsule -> SCI-004 experiment contract -> SCI-005 exposure/use separation -> SCI-006 dependency graph -> later measurement/adjudication/replication layers.

SCI-005 architecture may be reviewed before SCI-006 implementation exists, but strong positive eligibility should eventually consume dependency closure rather than pretend exposure history can detect every indirect information path by itself.

---

## 43. Review boundary

Review SCI-005 on this question:

> Does this contract make it mechanically difficult to relabel outcome-aware exploratory/tuning material as fresh confirmatory evidence, while remaining precise enough to allow legitimate scoped data reuse, cumulative prior knowledge, preregistered adaptive science, cross-validation, and genuinely new prospective evidence?

A positive architecture review does not establish that any current dataset, benchmark, model, analyst, or experiment is prospectively eligible.