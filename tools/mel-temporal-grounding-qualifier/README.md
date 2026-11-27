# MEL temporal-grounding qualification harness

This nested Rust package imports the exact production temporal-grounding scorer and SHA-256 digest module by path. It remains isolated from the main workspace so focused qualification does not inherit unrelated legacy build/lint debt.

Run the scorer checks with:

    cargo +1.96.0 test --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --all-targets -- -D warnings

## Multi-annotator temporal grounding

The temporal_consensus module preserves every independent annotation record and runs the exact production temporal-grounding scorer once per annotator. It reports per-annotator TP/FP/FN, F1, occurrence count accuracy and boundary errors, plus panel summaries for positive grounding, negative claims, occurrence-count spread and boundary errors.

Ambiguous and not-estimable labels are retained in the report but excluded only from their own scored denominators. An absent label remains a scored negative reference. The evaluator does not create a majority-voted/merged consensus interval, rank annotators, or treat a small panel's score spread as a population confidence interval. It computes descriptive pairwise concordance, but not chance-corrected formal inter-annotator reliability.

Run the synthetic six-annotator panel with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-multi-annotator-grounding-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-multi-annotator-temporal.synthetic.json

All annotator identities are represented by opaque digests in the fixture/report. Raw personal identifiers must not be placed in public benchmark artifacts.


The strict input envelope is also described by `schemas/mel-multi-annotator-temporal-grounding-v1.schema.json`; cross-record identity, state/interval semantics and digest uniqueness are enforced by the Rust adapter.

Count-gated one-to-many F1 is computed only over Present annotations whose reference contains more than one occurrence. Single-event references remain in ordinary F1 and exact-count accuracy but do not dilute the one-to-many denominator.

The report separates eligible annotation count from model abstention, parse failure and prediction coverage. A present/absent annotation stays eligible even if the model returns Ambiguous, NotEstimable, NotDetected or Unparseable. Coverage is reported overall and separately for Present and Absent references. The panel score spread is human-reference variability; it is not model confidence. The adapter does not compute formal pairwise inter-annotator agreement or population confidence intervals.

Research protocol: `docs/research/MEL-GROUND-006-MULTI-ANNOTATOR-TEMPORAL-GROUNDING-V1.md`.


For each predicted occurrence, the report also derives a **per-event reference-boundary distribution** from the production scorer's existing prediction↔annotation match records. It reports the number and fraction of eligible Present annotators who match that event, plus min/median/max start and end frames and boundary spread. Events that match no eligible Present annotation appear in `unmatched_prediction_indices`. This is not an interval consensus: original annotation records remain intact, and the support denominator is visible.


## Pairwise annotation concordance (descriptive, not formal IAA)

Every annotator pair is compared. The report first measures exact disposition agreement across Present, Absent, Ambiguous and NotEstimable states. For Present/Present pairs only, it applies the production one-to-one interval matcher in both directions and reports symmetric event F1, mean matched interval IoU and exact occurrence-count agreement.

Pairs with different dispositions, including Present/Absent, do not receive fabricated interval scores. Ambiguous and NotEstimable states remain categorical. These pairwise metrics are descriptive for the current query panel and are not Cohen's kappa, Fleiss' kappa or Krippendorff's alpha. A defensible chance-corrected reliability estimate requires a multi-query corpus with preregistered item/rater structure, missingness policy and chance model. The report continues to set `pairwise_inter_annotator_agreement_computed = false`.


## Corpus-level annotation reliability

The temporal_reliability module consumes a corpus of independently annotated items and reports two separate profiles:

1. Nominal Krippendorff alpha for item-level disposition states (Present, Absent, Ambiguous, NotEstimable), including the coincidence-derived observed and expected disagreement.
2. Symmetric pairwise event F1, matched IoU and exact-count agreement at frozen IoU thresholds 0.25, 0.50, 0.75 and 0.90, reusing the production temporal-grounding scorer in both directions.

This is not Krippendorff's alpha for unitizing (alpha_U): temporal interval agreement is deliberately reported separately. The report does not score a model, create consensus truth, calculate a source-work bootstrap or claim population-level reliability.

Run the synthetic corpus with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-reliability-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-temporal-reliability-corpus.synthetic.json

See docs/research/MEL-GROUND-007-CORPUS-ANNOTATION-RELIABILITY-V1.md for the mathematical definition and claim limits.


## MusicListenBench paired CLEAN / FLIP / STAY adapter

The music_listen_bench module imports no audio model. It validates a complete triad and immutable dataset/scorer revision digests, enforces one model/prompt/runtime/decoding identity per run capsule, recomputes the question SHA-256 over exact UTF-8 bytes, decodes variant-specific A/B choices into Same/Different musical relations, validates the expected CLEAN/STAY equality and FLIP opposition, and reports CLEAN accuracy, FLIP miss rate, STAY false-flip rate, CLEAN-correct-conditioned error rates, Acc_FS, joint exact rate, A/B answer bias, task slices and factor slices.

The strict case schema is at schemas/mel-music-listen-bench-case-v1.schema.json. Run the synthetic four-task case with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-music-listen-bench-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-music-listen-bench.synthetic.json

The synthetic case deliberately includes one wrong CLEAN answer and one unparseable FLIP answer, uses a reversed A/B semantic mapping in one variant, and contains two source lineages. Its metrics are adapter-plumbing checks only. Audio bytes are not fetched or rehashed, model inference is not invoked, and transformation authenticity is not established.


## Performance-invariant musical identity (MEL-HEAR-002)

The `performance_invariance` module is a separate extension to the official MusicListenBench v1.0 evaluator. It preserves the official CLEAN/FLIP/STAY contract and adds its own CLEAN / Performance-STAY / Musical-FLIP triad. Each triad names exactly one target property:

- chord quality;
- root pitch class;
- chord voicing;
- relative interval sequence;
- melodic contour;
- rhythmic ratio pattern;
- pulse rate;
- instrument family.

The target property determines which variations are valid STAY factors. Examples: global transposition is a STAY for relative interval sequence and chord quality, but not root pitch class; inversion may preserve chord quality/root but not exact voicing; uniform tempo scaling may preserve rhythmic ratios but not pulse rate. Acoustic-only Room/EQ/Hiss factors are deliberately excluded from this performance-variation module and remain in the existing MusicListenBench/acoustic-invariance path.

CLEAN and Musical-FLIP share one performance-condition digest; Performance-STAY must bind a different condition. All three audio hashes must differ. The question text and its recomputed SHA-256 are identical across the matched triad. Semantic A/B normalization is reused from `music_listen_bench::predicted_relation_for_variant`, so letter reversal does not change the musical answer.

Run the synthetic six-target fixture:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-performance-invariance-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-performance-invariance.synthetic.json

The case explicitly labels its split as SyntheticRendered, PublicRealRecordingEvaluationOnly or MusicianRecordedConfirmatory; do not pool these splits. It also binds a source-permissions manifest digest consumed from the existing rights registry. The report sets source_permission_receipts_authenticated=false: this digest is not verified by this adapter.

The fixture is an evaluator plumbing control, not evidence about model capability. Performance-condition, source-work, composition, annotation and permission references are supplied metadata; this evaluator does not authenticate them, read audio bytes, or invoke a model. Real-data analysis must be performer-, composition- and source-work-disjoint and rights-approved.
