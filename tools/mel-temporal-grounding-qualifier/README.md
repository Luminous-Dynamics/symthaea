# MEL temporal-grounding qualification harness

This nested Rust package imports the exact production scorer and digest module by path. It is intentionally isolated from the main workspace so scorer qualification does not inherit unrelated workspace build or lint debt.

Run:

    cargo +1.96.0 test --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --all-targets -- -D warnings

The base temporal-grounding tests exercise artifact/clock-bound interval scoring and count-gated one-to-many aggregation.

The temporal-revision V2 API accepts one frozen ground truth, one scoring plan, and raw draft/final predictions. It computes both scores using the exact production scorer. The caller cannot inject derived score fields. The resulting audit reports revision gain, harmful edits, perfect-draft preservation, count-gated one-to-many improvement, and explicit absence-classification changes.

Evidence conditions are distinct: no evidence, local evidence bound to the exact recording and in-range frame window, or yoked evidence from a different audio digest. Structural digest references do not authenticate evidence bytes or prove that a model consumed them.

The revision audit is an evaluation layer, not a model-training loop, waveform observer, capture verifier or musician study.


## Matched evidence-ablation triplets

The evaluator can compare one NoEvidence, one LocalRetrieved, and one YokedRetrieved V2 case under the same experiment_unit_id. All three arms must share the identical raw draft prediction, truth annotations, scoring plan, model, prompt, adapter and source recording. Only the final prediction and evidence condition may differ.

The contrast reports local-minus-no-evidence and local-minus-yoked final F1, count-gated F1, draft-to-final gains, and whether local evidence beats yoked evidence on both revision gain and final F1. Missing/duplicate arms and changed drafts are rejected. This does not itself prove causal effects in a population; it enforces the paired experimental design so subsequent statistics can estimate them correctly.


## CLI replay

Run a single V2 case or a matched evidence-triplet JSON file:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-revision-audit -- path/to/case-or-triplet.json

Use a hyphen as the path to read JSON from standard input. For an array input, the CLI audits each V2 case and emits matched evidence-ablation contrasts. The committed synthetic triplet fixture is at fixtures/mel-temporal-revision-triplet.synthetic.json.


## Matched-trial runtime and budget identity

V2 case inputs also bind a source-lineage identifier, immutable runtime identity digest, decoding-configuration digest, maximum output-token budget, actual output tokens, maximum wall-time budget and observed elapsed wall time. The evaluator rejects actual usage beyond the declared budget. A matched NoEvidence/LocalRetrieved/YokedRetrieved triplet must share source lineage, runtime, decoding configuration and maximum budgets; actual token/time consumption is reported separately for each arm.

These fields are self-reported run metadata until backed by an independently authenticated execution receipt. They make the paired protocol's intended controls explicit; they do not authenticate the runtime or prove that the model consumed retrieved audio.
