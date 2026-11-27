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


## Property-relative musical performance identity extension

The `music_performance_identity` module extends, but does not modify, the official MusicListenBench v1.0 CLEAN/FLIP/STAY protocol. Each case is a CLEAN / Performance-STAY / Musical-FLIP triad. It distinguishes chord quality, root pitch class, voicing, relative interval sequence, melodic contour, rhythmic ratio, pulse rate and instrument family.

Performance-STAY factors are typed as a different take by the same performer, a different performer on the same instrument family, strum/articulation change, expressive timing change, a different tempo with the same rhythmic-ratio pattern, or a different renderer for the same performance. The evaluator checks the corresponding identity metadata. Musical-FLIP must change exactly the declared target-property family; it cannot be treated as a generic waveform difference.

Run the synthetic eight-triad extension panel with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-music-performance-identity-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-music-performance-identity.synthetic.json

The report keeps CLEAN accuracy, performance-STAY false-difference rate and musical-FLIP miss rate separate, with conditional-on-CLEAN-correct rates and property/performance-factor/performer/take/renderer slices. It does not emit a composite music identity score.

The dataset, target-property references and pairing manifests are digest-shaped metadata only. The evaluator does not rehash audio bytes or authenticate musician annotations, performance identities or rights/consent. The synthetic panel is a deterministic evaluator control, not evidence of real-performance invariance or SOTA.
