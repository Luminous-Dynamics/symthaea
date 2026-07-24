# Symthaea Aesthetic — Next-Wave Patch Bundles

This series builds on the aesthetic truth-layer patches. It does not add another opaque “beauty” scalar. Instead it separates artifact evidence, context, human preference, novelty, policy utility, temporal form, and study provenance so each claim remains auditable.

## Required baseline

Apply the previous **Aesthetic Truth Patch Series** first. The expected baseline commit for these patches is:

- `2cd3d24 fix: make audio feature destructuring edition-explicit`

The patch files are ordinary Git mail patches and may also apply to an equivalent tree with different commit IDs.

## Bundle 1 — Assessment and Policy Contract

Commits:

1. `feat: separate assessment evidence from policy utility`
2. `feat: make policy utility explainable and context-sensitive`
3. `feat: track policy utility without contaminating intrinsic memory`

Adds:

- `ArtifactEvidence` and per-channel `EvidenceReliability`
- `EvaluationContext`, `CreativeMode`, and `AestheticModality`
- `ContextAlignment`, `NoveltyEstimate`, and `PreferenceEstimate`
- explicit `AestheticPolicy` weights
- `AestheticAssessment` and `AestheticUtilityBreakdown`
- separate persisted intrinsic and policy-utility expectations
- `AestheticTracker::process_assessment`

Important semantic boundary:

- intrinsic expectation learns only from artifact evidence;
- policy utility can vary by task without rewriting intrinsic taste;
- missing preference evidence reduces confidence rather than becoming a negative rating.

The aesthetic-memory schema advances from version 2 to version 3. Older memory loads with a neutral `utility_ema = 0.5` and upgrades on save.

## Bundle 2 — Temporal Form and Cross-Modal Intent

Commits:

1. `feat: score temporal arcs and intentional cross-modal relations`
2. `feat: add deterministic cross-modal creative intent curves`
3. `fix: harden persisted intent curves and affect endpoints`

Adds:

- `AestheticTrajectory` and `TimedAestheticFrame`
- modality-independent `AestheticVector`
- trajectory summaries for continuity, arc span, and return-to-origin
- bounded-lag temporal correspondence
- explicit `CrossModalRelation` modes:
  - `Congruent`
  - `Complementary`
  - `Counterpoint`
  - `Transformation`
- versioned `CreativeIntent`
- deterministic energy, warmth, complexity, affect, and Eight-Harmony curves
- robust sampling of deserialized curves even when keyframes are unsorted

This replaces the assumption that all coherent modalities must express the same emotion at the same instant. Delayed correspondence, intentional opposition, and gradual convergence are now representable.

## Bundle 3 — Preference Science and Study Provenance

Commits:

1. `feat: quantify preference support and grounded uncertainty`
2. `feat: add replayable preference studies and calibration reports`
3. `fix: reject invalid preference-study evidence at ingestion`

Adds:

- `TastePrediction` with confidence, feature support, and grounded fraction
- `TasteModelHealth`
- pairwise preference probabilities
- training-weight and feature-exposure accounting
- replayable `PreferenceStudyLedger`
- absolute and pairwise study observations
- source-separated evidence counts
- contradiction detection
- atomic JSON persistence
- ten-bin calibration reports with Brier score and expected calibration error
- strict rejection of non-finite or out-of-range study evidence

Use a held-out ledger for real calibration claims. Evaluating the same ledger used for training is only a smoke test.

## Bundle 4 — Integration Closure

Commit:

1. `fix: preserve independent evidence while integrating policy feedback`

This patch connects `TastePrediction` to `PreferenceEstimate`, down-weights analyst or synthetic preference evidence without erasing it, and ensures complete assessments update both independent expectations from their proper evidence channels.

Apply this after Bundles 1 and 3.

## Recommended downstream flow

```text
modality extractor
    -> ArtifactEvidence
history
    -> NoveltyEstimate
intent / task evaluator
    -> ContextAlignment
TasteModel::predict_with_uncertainty
    -> PreferenceEstimate
AestheticPolicy::for_mode
    -> AestheticAssessment::evaluate
AestheticTracker::process_assessment
    -> AestheticFeedback
```

For multimodal works:

```text
CreativeIntent::sample(position)
    -> modality generators
modality feature trajectories
    -> temporal_relation_score(relation)
    -> ContextAlignment
```

For human studies:

```text
PreferenceStudyLedger
    -> train_model(training ledger)
    -> evaluate_calibration(held-out ledger)
    -> evidence bundle / dashboard
```

## Compatibility

Existing APIs remain available, including:

- `AestheticEvaluator`
- `AestheticScore`
- `AestheticTracker::process`
- `CreativeSession` and `derive_session`
- `NoveltyTracker`
- `TasteModel::predict`

New code should prefer the typed assessment contract for task-dependent decisions and `CreativeIntent` for cross-modal generation.

## Verification performed in the generation environment

- all Rust files parsed successfully with the tree-sitter Rust grammar;
- `git diff --check` passed after every patch;
- commit order was rebuilt into a clean dependency sequence;
- the reordered final Git tree exactly matched the authored final tree;
- all bundles are replayed onto a fresh baseline during packaging;
- SHA-256 manifests are generated for every archive and patch.

A Rust toolchain could not be downloaded because external DNS was unavailable in the sandbox. Run the parent workspace’s required gates before merge:

```text
cargo fmt --all -- --check
cargo test -p symthaea-aesthetic
cargo clippy -p symthaea-aesthetic --all-targets --all-features -- -D warnings
cargo test --workspace
```

## Next research layer

The next highest-value work is empirical rather than architectural:

- modality-specific extractors with declared evidence reliability;
- held-out pairwise studies across expertise and cultural cohorts;
- learned context alignment rather than static mappings;
- ablations proving which signals improve human preference prediction;
- long-form trajectory studies testing tension, release, recurrence, and delayed cross-modal correspondence.
