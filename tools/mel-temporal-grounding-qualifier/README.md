# MEL temporal-grounding qualification harness

This nested Rust package imports the exact production scorer and digest module by path. It is intentionally isolated from the main workspace so scorer qualification does not inherit unrelated workspace build or lint debt.

Run:

    cargo +1.96.0 test --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --all-targets -- -D warnings

The base temporal-grounding tests exercise artifact/clock-bound interval scoring and count-gated one-to-many aggregation.

The temporal-revision V2 API accepts one frozen ground truth, one scoring plan, and raw draft/final predictions. It computes both scores using the exact production scorer. The caller cannot inject derived score fields. The resulting audit reports revision gain, harmful edits, perfect-draft preservation, count-gated one-to-many improvement, and explicit absence-classification changes.

Evidence conditions are distinct: no evidence, local evidence bound to the exact recording and in-range frame window, or yoked evidence from a different audio digest. Structural digest references do not authenticate evidence bytes or prove that a model consumed them.

The revision audit is an evaluation layer, not a model-training loop, waveform observer, capture verifier or musician study.
