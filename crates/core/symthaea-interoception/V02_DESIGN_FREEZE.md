# Affective Emergence v0.2 — Design Freeze and Implementation-Start Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines when v0.2 planning is sufficiently complete to stop changing the scientific design and begin implementation. Its purpose is to prevent endless design drift and to make the transition from research planning to code an explicit evidence event.

## Principle

A research design should not remain indefinitely editable while implementation and data generation begin.

Before runtime implementation starts, freeze the scientific/epistemic contracts that determine what will count as a candidate, what information it may use, which scenarios test it, how failures are classified, and how evidence is promoted.

Implementation may reveal a genuine design defect. Fixing such a defect is allowed, but it creates a new design-freeze identity and must occur before confirmatory data are generated under the revised design.

## Authoritative contract closure

The freeze must bind a validated `DesignContractRegistryManifest` defined by `V02_DESIGN_CONTRACT_REGISTRY.md`.

This registry is the authoritative closure over all active normative v0.2 design contracts. Selected individual contract digests may also be copied into the freeze for audit convenience, but they must equal the corresponding registry entries and cannot substitute for registry closure.

The freeze is invalid if:

- a required normative role is absent from the registry;
- an eligible active normative `V02_*` contract is omitted;
- a role that must be singular has multiple active normative entries;
- any path/content digest does not match the exact frozen source tree;
- registry source commit differs from the exact v0.2 design source commit;
- a selected individual digest disagrees with the registry;
- the registry or one of its required contracts is superseded/invalidated.

Adding or changing an active normative contract after freeze changes the registry identity and supersedes the old freeze.

## Proposed DesignFreezeManifest

Create a canonical machine-readable manifest binding at minimum:

- design-freeze schema/version;
- exact v0.1 parent source commit;
- v0.1 model-semantics version;
- current v0.1 qualification dependency state;
- required v0.1 `QualificationEvidenceBundle` schema/version;
- exact v0.2 design branch/source commit;
- `V02_DESIGN_CONTRACT_REGISTRY.md` content digest/version;
- validated `DesignContractRegistryManifest` SHA-256;
- top-level observational-affect plan digest, verified against the registry;
- information-firewall contract digest, verified against the registry;
- temporal-alignment contract digest, verified against the registry;
- candidate-definition contract digest, verified against the registry;
- candidate-factor-space contract digest, verified against the registry;
- weighting-decomposition contract digest, verified against the registry;
- allostatic-exposure-decomposition contract digest, verified against the registry;
- execution-mode contract digest, verified against the registry;
- scenario/holdout contract digest, verified against the registry;
- blinding/unblinding contract digest, verified against the registry;
- evidence-root contract digest, verified against the registry;
- capability-typed API contract digest, verified against the registry;
- adversarial-validation contract digest, verified against the registry;
- claim-boundary text/version;
- implementation tranche ordering/version;
- unresolved-design-question list;
- freeze status;
- canonical manifest SHA-256.

The registry digest is authoritative; the repeated contract fields are redundant audit anchors and must match it exactly.

## Freeze states

Use explicit states rather than a generic complete/incomplete flag:

- `Draft` — design still actively changing;
- `Reviewable` — all required contracts exist, but unresolved blocking questions remain or review has not completed;
- `FrozenBlockedOnV01` — v0.2 design is frozen, but implementation remains blocked on v0.1 qualification;
- `FrozenImplementationAuthorized` — v0.1 dependency satisfied by one exact valid qualification/evidence bundle and implementation may begin;
- `Superseded` — a later design freeze or contract registry replaces this one;
- `Invalidated` — a discovered internal contradiction makes this freeze unusable.

`FrozenImplementationAuthorized` must never be reachable while v0.1 qualification is unresolved/failed, while the supplied v0.1 qualification/evidence components do not validate as one exact bound lineage, or while the design-contract registry fails closure validation.

## Required design-completeness gates

Before entering `FrozenBlockedOnV01`, all of the following should be true.

### Contract-registry gate

The complete active normative design set is represented in one validated registry.

The current intended normative roles cover:

- observational-affect plan;
- information firewall;
- temporal alignment;
- candidate definition;
- candidate factor space;
- weighting decomposition;
- allostatic exposure decomposition;
- execution mode;
- scenario/holdout manifest;
- blinding custody;
- capability-typed API;
- adversarial validation;
- evidence root;
- design freeze.

An unregistered normative-looking contract is a hard freeze failure until explicitly classified.

### Scientific question gate

The primary v0.2 question is stated narrowly and does not presuppose emotion:

> Do reproducible, prefix-causal, label-free regulatory observables distinguish aspects of regulatory change, confidence, exposure, and forecast revision beyond simpler reactive and nuisance baselines?

No stronger claim is embedded in metric names or success criteria.

### Information / execution-mode gate

The design specifies:

- allowed information through time `t`;
- forbidden future/semantic information;
- `OfflinePrefixReplay` as the first evidence-bearing mode;
- later `OnlineShadowObservation` as a separately qualified engineering mode;
- retrospective and oracle classes as non-primary authority namespaces;
- prefix-equivalence/future-mutation invariants;
- separation of prefix-causal `CandidatePayload` from suffix-sensitive full-trace provenance envelope;
- capability-typed API shape preventing accidental escalation.

The candidate computation must never receive the full source-trace digest or any other suffix-sensitive identity.

### Mathematical relation gate

The design distinguishes at least:

- R0 current burden;
- R1 realized current change;
- R2 one-step forecast residual;
- R3 aligned overlapping-future revision;
- R4 rolling finite-horizon change;
- separate urgency family.

Rolling-horizon turnover is not silently interpreted as forecast revision.

### Weighting gate

The design keeps distinct:

- raw per-channel burden;
- importance-only viability-weighted burden;
- the legacy v0.1 precision×importance aggregate;
- explicit precision/confidence observables.

The exploratory scenario set must decorrelate burden and precision so these hypotheses can disagree.

### Temporal-exposure gate

The design keeps distinct:

- instantaneous burden;
- discounted mean burden;
- discounted cumulative exposure;
- undiscounted cumulative exposure;
- peak;
- terminal state;
- preferred/viability exposure duration;
- breach latency;
- recovery exposure.

v0.1 `discounted_debt` is treated under its actual legacy semantics as a discounted mean, not silently reinterpreted as cumulative exposure.

### Candidate-factorization gate

Each candidate has an explicit factor-space coordinate binding at least:

- relation basis;
- weighting basis;
- temporal aggregation;
- forecast policy when applicable;
- information class;
- channel projection when applicable;
- availability/numeric contract.

The factor axes do not authorize an unrestricted Cartesian search. A finite exploratory candidate set is prospectively frozen.

### Candidate-identity gate

Candidate identity prospectively includes formula, sign, temporal indices, factor coordinate, information class, forecast policy, horizon, discount, normalization, numeric rules, undefined semantics, source/model lineage, and reference fixtures.

### Scenario gate

Discovery and confirmatory scenario identity are separate and content-hash audited.

Required adversarial scenario families include neutral, nuisance-matched, crossed-sign, exact-prefix/divergent-future, future-mutation, forecast-agreement/disagreement, burden-vs-precision discriminators, temporal-aggregation discriminators, and channel-projection disagreements.

Comparison cut points/windows are prospective.

### Blinding gate

The design specifies:

- separate semantic arm mapping commitment;
- opaque codes;
- primary artifact flow without mapping contents;
- semantic-label canary tests;
- explicit unblinding receipt;
- honest blinding-strength declaration.

### Causal-isolation gate

For initial scientific evidence, native execution completes and freezes before observational candidate computation begins.

This removes observer→native feedback from the primary `OfflinePrefixReplay` path by construction.

A later `OnlineShadowObservation` implementation must separately prove exact native execution equality with and without observatory attachment and exact candidate-payload equivalence to offline replay.

No v0.2 measurement type is designed to become a drive/action/neuromodulator/memory/cognitive command.

### Baseline-qualification gate

The v0.2 design recognizes one authoritative v0.1 promotion object: `QualificationEvidenceBundle`.

Implementation authorization requires that the bundle:

- validates structurally;
- reports qualified;
- binds the exact v0.1 source commit named by the design/start receipt;
- binds the expected v0.1 model-semantics version;
- contains the passing qualification receipt and evidence capsule from that same source lineage.

Two independently valid but cross-paired v0.1 artifacts must never authorize v0.2 implementation.

### Evidence gate

The prospective root and realized package are distinct.

Every locked scenario is accounted for as included/excluded/indeterminate.

Qualified negative/null result is distinct from integrity failure.

The prospective and realized evidence structures bind the exact design-contract-registry identity.

### Adversarial-validation gate

The design has explicit attacks for future leakage, suffix-sensitive provenance leakage, semantic leakage, observer feedback, temporal indexing, weighting conflation, exposure/mean conflation, artifact substitution, scenario omission, exclusion manipulation, analysis mutation, and known-malicious fixtures.

### Deterministic-inference gate

The design does not manufacture stochastic significance from deterministic grids.

Held-out robustness, worst-case/minimum margins, equivalence bounds, paired baseline comparisons, explicit failure regions, and finite prospectively closed candidate sets are preferred unless stochasticity is separately introduced/qualified.

### Claim-boundary gate

Even a successful v0.2 remains explicitly insufficient to establish emotion, subjective valence, feeling, mood, suffering, sentience, consciousness, or unseen-future prediction.

## Unresolved question policy

A design freeze may contain unresolved questions only if each is classified:

- `ImplementationDetail` — does not change scientific meaning/evidence identity;
- `ExploratoryChoice` — may be chosen using exploratory data, after which confirmatory identity must be newly frozen;
- `ConfirmatoryBlocking` — must be resolved before confirmatory root lock;
- `ArchitectureBlocking` — must be resolved before implementation begins.

Examples:

- exact internal Rust collection type: usually `ImplementationDetail`;
- primary candidate after a prospectively closed exploratory comparison: `ExploratoryChoice`;
- minimum-effect threshold for confirmation: `ConfirmatoryBlocking`;
- whether full-trace provenance can enter candidate computation: resolved as forbidden and `ArchitectureBlocking` if reopened;
- whether primary evidence is offline replay or live co-resident observation: resolved as `OfflinePrefixReplay` and `ArchitectureBlocking` if reopened.

No unresolved `ArchitectureBlocking` item is permitted at `FrozenImplementationAuthorized`.

## Implementation tranche freeze

Before code begins, lock the initial implementation order so later attractive features do not enter v0.2 opportunistically.

Recommended sequence:

1. standalone observatory crate and one-way dependency boundary;
2. validated frozen-trace replay harness;
3. canonical prefix artifact/digest contract;
4. `ObservationPrefixView` and capability types;
5. separate prefix-causal `CandidatePayload` and outer `CandidateEvidenceEnvelope`;
6. prefix-causal forecast policy interfaces;
7. forecast trajectory artifact sufficient to reproduce exact legacy v0.1 allostasis;
8. weighting × temporal × relation candidate factorization and compatibility validator;
9. finite exploratory candidate-set manifest;
10. raw / viability-weighted / legacy-weighted / confidence candidate families;
11. mean / cumulative exposure / peak / terminal / duration / latency / recovery candidate families;
12. neutral R0/R1/R2/R3/R4/urgency definitions;
13. typed unavailable/undefined semantics;
14. prefix/suffix-mutation, weighting, temporal, and malicious-fixture adversarial tests;
15. semantic-label canary and mapping separation;
16. scenario/cohort manifests;
17. blinded candidate artifacts/comparison;
18. evidence-root/validation receipts;
19. exploratory `OfflinePrefixReplay` study only;
20. later, separately qualified `OnlineShadowObservation` equivalence.

Out of scope for this implementation lineage:

- neuromodulation;
- memory weighting;
- attention modulation;
- action selection;
- policy/control outputs;
- controllability/dominance;
- persistent mood states;
- attachment/social affect;
- learned emotion labels;
- consciousness/sentience inference.

Adding any of these requires a later tranche/design lineage.

## Design-change severity after freeze

### Class I — implementation-preserving

Examples:

- refactor with identical reference fixtures/artifacts;
- clearer error message that does not leak semantics/change machine artifact;
- internal allocation/performance improvement with identical qualified outputs.

May preserve design identity only when the contract registry and all canonical scientific identities remain unchanged.

### Class II — candidate/evidence semantic change

Examples:

- formula/sign/horizon change;
- factor-coordinate change;
- weighting or temporal aggregation change;
- information dependency change within the same authority class;
- temporal alignment change;
- scenario cohort change;
- threshold/comparison rule change;
- new baseline/removal of baseline.

Requires new design/candidate/evidence identity before confirmatory data.

### Class III — architecture-boundary change

Examples:

- allowing feedback to native execution;
- changing primary evidence from offline replay to co-resident live execution without a new lineage;
- allowing full-trace/suffix-sensitive provenance into candidate computation;
- allowing oracle information in primary prefix-causal code;
- adding semantic arm identity to candidate computation;
- adding causal affect outputs.

Invalidates v0.2 observational freeze and requires a new scientific tranche/design identity.

## ImplementationStartReceipt

When v0.1 qualification eventually passes, create a small receipt binding:

- `DesignFreezeManifest` SHA-256;
- `DesignContractRegistryManifest` SHA-256;
- exact qualified v0.1 source commit;
- v0.1 `QualificationEvidenceBundle` SHA-256;
- embedded v0.1 qualification-receipt SHA-256 and evidence-capsule SHA-256 for audit detail;
- v0.1 model-semantics version;
- v0.2 implementation branch starting SHA;
- implementation tranche version;
- authorization state `FrozenImplementationAuthorized`;
- canonical receipt SHA-256.

The start receipt validator must require that:

- the supplied design freeze validates;
- the supplied contract registry validates and exactly equals the registry bound by the freeze;
- the supplied `QualificationEvidenceBundle` validates and reports qualified;
- the v0.1 source/model-semantics identities match exactly;
- component digests equal the corresponding embedded bundle components.

This receipt marks the exact transition from design to implementation.

No observational primary data should predate the relevant implementation-start/design identities.

If v0.1 source changes after the start receipt is issued but before a later experiment claims the new source lineage, a new qualified v0.1 bundle and a new implementation/evidence lineage are required as appropriate. An older qualified bundle does not silently qualify a newer source head.

## Confirmatory lock occurs later

Design freeze is **not** the same as confirmatory preregistration.

After implementation and exploratory qualification:

- choose/freeze the primary candidate from the prospectively closed exploratory set;
- choose thresholds/equivalence bands;
- freeze confirmatory scenario cohort;
- freeze analysis plan;
- freeze mapping commitment;
- construct the prospective observational evidence root;
- only then generate confirmatory data.

This preserves legitimate exploratory learning while preventing confirmatory retrofitting.

## Review checklist

A reviewer should be able to answer yes to all of these before implementation begins:

1. Is every active normative v0.2 design contract represented by one validated contract registry?
2. Can primary candidate computation run from an immutable prefix without access to future schedules or full-trace provenance?
3. Can primary candidate computation run without semantic arm mapping?
4. Are R0/R1/R2/R3/R4/urgency mathematically distinct and adversarially tested?
5. Are weighting and temporal aggregation explicit orthogonal factors rather than hidden inside one burden/debt scalar?
6. Is the exploratory candidate set finite and prospectively closed?
7. Is oracle analysis structurally separate?
8. Are candidate and scenario identities immutable/hashable?
9. Can null/negative results survive without being reclassified as integrity failures?
10. Can every confirmatory scenario be accounted for?
11. Can malicious leakage/tampering fixtures be caught?
12. Is initial evidence observer→native isolated by offline replay, with live observation deferred to a separate equivalence gate?
13. Does implementation authorization bind one exact qualified v0.1 `QualificationEvidenceBundle` and one exact design-contract registry?
14. Are stronger affect/mood/consciousness claims explicitly out of scope?

## Claim boundary

A design freeze demonstrates that the intended experiment and evidence architecture were fixed before implementation/data generation and that the complete normative contract surface was closed under one registry identity. It does not validate the implementation, qualify v0.1, establish that a candidate succeeds, or support claims of emotion, feeling, mood, suffering, sentience, or consciousness.
