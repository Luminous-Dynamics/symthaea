# ROB-BENCHMARK-PROVENANCE-HOLDOUT-001

Status: first executable robotics-specific source-group holdout validator and descriptive transfer report. This is a deterministic Rust reference over caller-supplied manifests and paired hidden-evaluation outcomes. It is not a dataset importer, hidden-challenge custody service, experiment runner, inferential statistics library, or robot controller.

Parent: [ROB-BENCHMARK-VALIDITY-001 #11775](https://github.com/Luminous-Dynamics/symthaea/issues/11775). Compose with:
- [EMSR-002 #10814](https://github.com/Luminous-Dynamics/symthaea/issues/10814) for exact dataset bytes, rights, frozen splits, group-aware assignments and run receipts.
- [SYM-EVAL-CORE-008 #1300](https://github.com/Luminous-Dynamics/symtropy/issues/1300) for hidden challenge commitment, custody, post-run reveal, prior-exposure classes and no selective omission.
- [Paired independent-cluster comparison #11807](https://github.com/Luminous-Dynamics/symthaea/pull/11807) for confirmatory inference.
- [Shortcut and privileged-input probes #11829](https://github.com/Luminous-Dynamics/symthaea/pull/11829) for probe-control diagnostics.

## Research question

Do policy gains survive when evaluation is held out by the real data-generating sources and embodiment factors—not merely by random episodes drawn from the same collection process?

Random episode splits can place highly correlated trajectories, operator conventions, camera/calibration characteristics, serial-unit behavior or simulator artifacts in both train and test. Such a split can measure interpolation within one source rather than transfer to a new source.

## Robotics source taxonomy

The reference supports explicit applicability decisions for:
- collection site / facility;
- operator / teleoperator / demonstration collector;
- physical robot article / serial unit;
- robot morphology / kinematic and actuator family;
- sensor and calibration stack;
- simulator engine / physics backend / asset family;
- upstream dataset source / capture pipeline;
- task / object / fixture family;
- payload / load regime;
- environment / operating-domain regime.

A dimension must be marked Required or NotApplicable with a non-empty reason. A required dimension has whole-group assignments in training, calibration/model-selection and hidden-confirmatory partitions, with a predeclared minimum hidden-group count. NotApplicable dimensions cannot carry source-group assignments or case labels.

Sensitive real-world identifiers may be represented by stable access-controlled digests; public result reports need not expose operator identity or other personal information.

## Comparison contract

Candidate and baseline bind the same:
- benchmark/version and dataset/source manifest;
- grouping and split policy;
- hidden partition;
- task and environment;
- robot configuration / embodiment;
- evaluator and safety specification;
- resource-budget boundary.

Every hidden paired case binds an exact case manifest, independent cluster, independence receipt, disposition and exactly one group identity for every Required source dimension. The same source group cannot be assigned to more than one partition. Hidden cases cannot point to training or calibration groups.

Unknown outcomes, censored episodes or Unknown cluster independence make the relevant group Inconclusive. Rejected independence evidence rejects the input. Missing a hidden source group from results makes the dimension Inconclusive.

## Output and statistical boundaries

The report gives, per hidden source group:
- candidate and matched-baseline safe-success rates over every scheduled episode;
- successful-but-unsafe counts separately;
- Unknown/censored counts;
- distinct independent cluster count;
- descriptive candidate-minus-baseline group effect.

It also reports source-group coverage, worst observed group effect and candidate rate range separately for each required dimension.

The descriptive rates are not inferential statistics. They do not imply that multiple episodes or source groups create independent observations. Use #11807 for paired independent-cluster significance, multiplicity and uncertainty. The source-holdout report never emits a p-value, confidence interval, SOTA ranking or safety certification.

The initial reference validates the **hidden confirmatory partition**. A separate in-source reference arm and explicit transfer-gap estimate are not yet implemented; the current report must not be presented as a quantitative train-to-test transfer gap. Those require a separately frozen in-source reference partition and source-matched paired observations, not a comparison against training scores.

## Claim ceiling

A passing software reference shows only that the declared group assignments, arm scopes and synthetic paired results satisfy the deterministic contract. It does not prove source labels truthful, cluster independence, dataset representativeness, hidden challenge secrecy, generalization to physical hardware, or superiority over another robot.

The SHA/digest fields are identities supplied by upstream evidence owners. This crate does not recompute the dataset, source metadata or hidden partition hashes. Use EMSR-002 and the challenge custody owner for byte integrity and pre-run blinding.

## Running the reference

The package is dependency-free and isolated from the Symthaea root Cargo workspace:

    cargo fmt --manifest-path research/rob-benchmark-provenance-holdout-001/reference/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-benchmark-provenance-holdout-001/reference/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-benchmark-provenance-holdout-001/reference/Cargo.toml --all-targets -- -D warnings

The exact-head Rust 1.96 GitHub Actions workflow is the qualification gate. Authored tests are not PASS until the dedicated workflow succeeds.
