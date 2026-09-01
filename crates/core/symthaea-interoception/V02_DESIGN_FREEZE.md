# Affective Emergence v0.2 — Design Freeze and Implementation-Start Contract

Status: **design-only / blocked on Native Interoception v0.1 qualification**

This contract defines when v0.2 planning is complete enough to stop changing scientific semantics and begin implementation.

## 1. Principle

Implementation and primary data generation must not begin while the scientific design remains freely editable.

The design freeze binds one complete normative contract registry. If implementation reveals a genuine design defect, fixing it creates a new registry/freeze identity before confirmatory data are generated.

## 2. Authoritative design closure

The freeze binds one validated `DesignContractRegistryManifest` from `V02_DESIGN_CONTRACT_REGISTRY.md`.

The registry is authoritative over every active normative `V02_*` contract. A freeze fails when:

- a required role is missing or duplicated;
- an eligible normative file is omitted;
- path/content/source identities disagree;
- a required contract is superseded/invalidated;
- a repeated audit digest differs from its registry entry.

Adding or changing an active normative contract changes the registry and supersedes the prior freeze.

## 3. Freeze states

- `Draft` — design actively changing;
- `Reviewable` — contracts exist but review/blockers remain;
- `FrozenBlockedOnV01` — design frozen; implementation blocked only on v0.1 qualification;
- `FrozenImplementationAuthorized` — one exact qualified v0.1 bundle plus one exact frozen design authorize implementation;
- `Superseded` — replaced by a later freeze;
- `Invalidated` — internal contradiction makes the freeze unusable.

`FrozenImplementationAuthorized` is unreachable while v0.1 is unresolved/failed or any architecture-blocking design gate below is unresolved.

## 4. Proposed DesignFreezeManifest

Bind at minimum:

- freeze schema/version;
- exact v0.1 source commit/model-semantics version;
- required `QualificationEvidenceBundle` schema/version;
- exact v0.2 design source commit;
- registry-contract specification digest;
- validated registry-manifest SHA-256;
- selected audit digests for key contracts, required to match the registry;
- claim-boundary version;
- implementation-tranche version;
- unresolved-question list with severity;
- freeze state;
- canonical SHA-256.

The registry digest is authoritative; repeated individual contract digests are audit anchors only.

## 5. Scientific-question gate

The primary question remains narrow:

> Do reproducible, prefix-causal, label-free regulatory observables distinguish regulatory change, confidence, burden distribution, temporal exposure, historical information, and forecast revision beyond simpler baselines?

The question does not presuppose emotion.

## 6. Information / execution gate

The design fixes:

- `OfflinePrefixReplay` as the initial evidence-bearing mode;
- later `OnlinePrefixCausalShadow` as a separately qualified engineering mode;
- retrospective/oracle authority as diagnostic only;
- `CandidatePayload` separated from suffix-sensitive full-trace provenance;
- no future schedule/state, semantic mapping, mutable native state, or full-trace digest in prefix-causal candidate computation.

Future-suffix mutation must not alter a candidate payload at an earlier cut point.

## 7. Native-state/history gate

The design distinguishes:

- `H0CurrentNativeStateOnly`;
- `H1ReplayedPrefixHistory` supplied to the external observatory;
- future separately qualified native persisted memory.

External history-derived persistence is not native memory or mood.

Matched-state restart tests must verify that identical complete native state/configuration plus identical future inputs yields identical future native execution under the v0.1 deterministic contract.

## 8. Mathematical-decomposition gate

Keep explicit and independent:

- R0 current burden;
- R1 realized change;
- R2 one-step forecast residual;
- R3 aligned future-outlook revision;
- R4 rolling-horizon change;
- urgency/breach-imminence families;
- W0 raw, W1 viability-only, W2 legacy precision×importance, W3 confidence;
- cross-channel vector/channel/peak/mean/sum/breadth aggregation;
- T0–T9 instantaneous/mean/exposure/peak/terminal/duration/latency/recovery semantics.

The legacy v0.1 `discounted_debt` remains a discounted mean projected burden, not cumulative exposure.

## 9. Candidate-identity gate

Every candidate binds:

- factor coordinate;
- exact formula/sign/indices;
- forecast/horizon/discount semantics;
- aggregation/denominator semantics;
- history-access and information class;
- preprocessing/calibration manifest or explicit `None`;
- calibration/fitted-parameter identities when applicable;
- evaluator-isolation manifest and allowed persistent-state/cache class;
- numeric/undefined/out-of-range rules;
- source/fixture/native-lineage identities.

The candidate set is finite and prospectively closed. The factor space is not authorization for Cartesian metric fishing.

## 10. Calibration / preprocessing gate

Before implementation authorization, `V02_CALIBRATION_AND_PREPROCESSING.md` must be normative and unresolved architecture-level questions must be closed.

Before **confirmatory root lock**, every primary candidate must have all value-changing preprocessing parameters frozen from:

- structural constants;
- a prospectively identified discovery/calibration cohort;
- or an independently identified external reference.

Confirmatory outcomes cannot refit scaling, thresholds, clipping, smoothing, or normalization.

A calibration scenario becomes discovery/calibration evidence and cannot also be an untouched holdout case.

## 11. Observatory-state lifecycle gate

`V02_OBSERVATORY_STATE_LIFECYCLE.md` is architecture-blocking for initial implementation.

Primary offline evaluation must be scenario-local and replay-determined.

Required semantics include:

- no cross-scenario/arm mutable evaluator state;
- explicit create/evaluate/finalize/destroy lifecycle;
- cold/warm process equivalence;
- candidate/scenario order invariance;
- serial/parallel and batch-size invariance;
- cache hit/miss equivalence;
- from-scratch vs incremental prefix equivalence where incremental H1 computation exists.

The initial allowed persistent-state class is `NoneAcrossEvaluationCoordinates`.

## 12. Identifiability gate

Every confirmatory primary-vs-baseline superiority claim needs a prospectively registered discriminator.

A `CandidateDiscriminationManifest` binds candidate/baseline pairs, locked scenario/cut-point discriminators, equivalence tolerances, and parsimony rule.

If a complex candidate remains equivalent to a simpler baseline, report `EquivalentToBaseline` / `InsufficientDiscrimination` rather than promoting it semantically.

H1 history claims require matched-current-state H1-vs-H0 discriminators.

## 13. Causal-contrast gate

Mechanistic language requires a prospectively frozen `CausalContrastManifest` declaring:

- manipulated fields;
- pre-treatment equalities;
- allowed mediators;
- forbidden changes;
- contrast class;
- discrimination obligation.

A realized mechanistic claim also needs a passing manipulation-check artifact. Otherwise report a descriptive scenario difference.

## 14. Scenario / holdout gate

Discovery, calibration, diagnostic, and confirmatory holdout roles are explicit and content-hash audited.

Required scenario families include:

- neutral stability;
- equal-state/different-history;
- equal-current-state/different-current-load;
- equal-drive/different-margin;
- deterministic recovery;
- crossed R1/R2/R3/R4 signs;
- identical-prefix/divergent-future twins;
- forecast-policy agreement/disagreement;
- burden-vs-precision discriminators;
- mean-vs-cumulative/peak/duration discriminators;
- channel-aggregation/denominator discriminators;
- H0-vs-H1 matched-state history discriminators;
- evaluator order/isolation malicious fixtures;
- preprocessing/holdout-leakage malicious fixtures.

Primary comparison cut points/windows are prospective.

## 15. Blinding gate

The design fixes:

- opaque blind codes;
- separate semantic arm mapping commitment;
- primary artifact flow without semantic mapping contents;
- semantic-label canary tests;
- explicit unblinding receipt;
- honest blinding-strength declaration.

Preprocessing/calibration code used for primary candidate values must also be semantic-label blind.

## 16. Baseline-qualification gate

Implementation authorization requires one exact valid `QualificationEvidenceBundle` that:

- validates structurally;
- reports qualified;
- binds the exact v0.1 source/model-semantics lineage named by the start receipt;
- contains the passing qualification receipt and evidence capsule from that same lineage.

Cross-paired valid components cannot authorize implementation.

## 17. Evidence gate

Prospective root and realized evidence package are distinct.

The root binds the exact design registry/freeze, candidate definitions, preprocessing/calibration identities, evaluator-isolation identity, scenarios, analysis, blinding, toolchain, and qualification baseline.

The realized package accounts for every locked scenario and every required integrity report.

`QualifiedNegativeOrNullResult` remains distinct from `IntegrityFailure`.

## 18. Deterministic-inference gate

Do not manufacture stochastic significance from deterministic scenario grids.

Prefer:

- directional consistency;
- worst-case/minimum signed margins;
- equivalence bounds;
- paired candidate-vs-baseline margins;
- explicit failure regions;
- deterministic coverage summaries;
- equivalence classes.

If true stochastic sampling is later introduced, it requires a separately frozen generator/distribution/seed contract.

## 19. Claim boundary

Even a successful v0.2 remains insufficient to establish emotion, subjective valence, feeling, native mood, native memory, suffering, sentience, consciousness, or unseen-future prediction.

## 20. Unresolved-question policy

Each unresolved question is classified:

- `ImplementationDetail`;
- `ExploratoryChoice`;
- `ConfirmatoryBlocking`;
- `ArchitectureBlocking`.

No unresolved `ArchitectureBlocking` item is permitted at `FrozenImplementationAuthorized`.

Primary preprocessing choice may remain an `ExploratoryChoice` only when the exploratory candidate/preprocessing set is finite and prospectively locked; the chosen confirmatory preprocessing identity then becomes `ConfirmatoryBlocking` until frozen.

## 21. Initial implementation tranche

After authorization, implement in this order:

1. standalone read-only observatory crate + one-way dependency gate;
2. frozen-trace replay harness;
3. canonical prefix artifact/digest + `ObservationPrefixView`;
4. `CandidatePayload` / outer evidence-envelope separation;
5. evaluator lifecycle/isolation context with no cross-coordinate persistent state;
6. typed prefix-causal forecast policies;
7. trajectory artifact reproducing exact legacy v0.1 allostasis;
8. factor-space/compatibility validator including history and aggregation axes;
9. preprocessing/calibration manifest types and holdout firewall;
10. finite exploratory candidate-set manifest;
11. W/A/T/R/H candidate families;
12. typed unavailable/out-of-range semantics;
13. future-suffix, restart, evaluator-order, calibration-leakage, weighting, aggregation, temporal, semantic, and malicious-fixture gates;
14. scenario/cohort/calibration manifests and overlap audits;
15. identifiability/discrimination and causal-contrast manifests;
16. blinding/mapping separation;
17. blinded comparison + prospective/realized evidence receipts;
18. first exploratory `OfflinePrefixReplay` study;
19. only later separately qualify `OnlinePrefixCausalShadow`.

Out of scope: neuromodulation, memory/attention modulation, action selection, policy/control outputs, controllability/dominance, native persistent mood, attachment/social affect, emotion labels, consciousness/sentience inference.

## 22. Change severity after freeze

### Class I — implementation-preserving

Pure refactor/performance/error-message changes with identical canonical scientific identities and outputs.

### Class II — candidate/evidence semantic change

Includes formula, factor coordinate, preprocessing/calibration, evaluator-state lifecycle, weighting, aggregation, temporal semantics, history access, scenario cohort, threshold, analysis, or baseline changes.

Requires new relevant candidate/design/evidence identity before confirmatory use.

### Class III — architecture-boundary change

Includes feedback into native execution, live co-resident primary evidence without new qualification, future/oracle authority in primary code, semantic identity in candidate computation, or native causal affect/persistent-memory outputs.

Requires a new scientific tranche/design lineage.

## 23. ImplementationStartReceipt

When v0.1 qualifies, create a receipt binding:

- exact `DesignFreezeManifest` SHA-256;
- exact registry SHA-256;
- exact qualified v0.1 source/model-semantics identity;
- exact `QualificationEvidenceBundle` SHA-256 plus component audit digests;
- exact v0.2 implementation starting SHA;
- implementation-tranche version;
- `FrozenImplementationAuthorized` state;
- canonical receipt SHA-256.

The validator requires freeze/registry/bundle/source identities to match exactly.

No primary observational data may predate the applicable start receipt.

## 24. Confirmatory lock occurs later

Design freeze is not confirmatory preregistration.

After implementation and exploratory work:

- freeze primary candidate + exact preprocessing identity;
- freeze fitted calibration parameters/source cohort;
- freeze required baselines/sensitivity variants;
- freeze confirmatory scenario cohort;
- freeze discrimination and causal-contrast manifests;
- freeze analysis plan/mapping commitment;
- construct prospective confirmatory evidence root;
- only then generate confirmatory data.

## 25. Review checklist

Before implementation, a reviewer should be able to answer yes to all of these:

1. Is every active normative contract represented by the registry?
2. Can candidate computation run without future schedules, full-trace identity, semantic mapping, or mutable native state?
3. Is external history distinguished from native persisted memory?
4. Are weighting, channel aggregation, temporal aggregation, relation, forecast, information, and history axes explicit?
5. Is the exploratory candidate/preprocessing set finite?
6. Can confirmatory values be computed without fitting anything to the confirmatory cohort?
7. Is candidate evaluation invariant to scenario/candidate order and process/cache history?
8. Are primary-vs-baseline claims identifiable under locked discriminators?
9. Are causal claims backed by declared contrasts/manipulation checks?
10. Can null/negative/ambiguous outcomes survive intact?
11. Does implementation authorization bind one exact qualified v0.1 lineage?
12. Are stronger affect/mood/consciousness claims explicitly out of scope?

## 26. Claim boundary

A design freeze establishes only that the intended scientific/evidence architecture was fixed prospectively under one closed contract registry. It does not validate implementation or establish affect, emotion, native mood, sentience, or consciousness.