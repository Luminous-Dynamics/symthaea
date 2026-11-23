# MEL Blind Preference Assay v1

This standalone reference evaluator qualifies the measurement process used for pairwise music-preference studies. It is not a music generator, reward model, or musical-quality score.

## What it checks

- SHA-256 commitment/reveal of the assignment seed and deterministic per-presentation A-left/B-left assignment.
- Actual display order against that assignment.
- Counterbalancing within every declared stratum.
- Unique presentation IDs and one exposure per rater/case and per rater/candidate-pair.
- Byte-identical negative controls, used to estimate left/right selection tendency.
- Candidate-A preference separately when A appears left and right, with per-case order gaps.
- Wilson 95% interval for left choice on decisive identical-duplicate controls.

## Run

\`\`\`sh
cargo run --manifest-path tools/mel-preference-assay/Cargo.toml -- \
  tools/mel-preference-assay/fixtures/valid-assay-v1.json
cargo test --manifest-path tools/mel-preference-assay/Cargo.toml
cargo clippy --manifest-path tools/mel-preference-assay/Cargo.toml -- -D warnings
\`\`\`

The CLI exits non-zero for structural or assignment-integrity failures. A detected side-bias flag is a diagnostic, not a validity failure: it tells the study owner that raw preference scores need investigation.

## Assignment protocol

The operator commits to SHA256 of the UTF-8 assignment seed before collecting judgments, then reveals the 64-character seed in the completed assay. The assignment bit is the high bit of the SHA256 digest over these exact byte segments, concatenated in order:

1. The domain separator MEL-BLIND-PREFERENCE-ASSAY-V1 followed by a NUL byte.
2. The seed as 64 lowercase ASCII hex characters.
3. A NUL byte, then the assay ID as UTF-8.
4. A NUL byte, then the presentation ID as UTF-8.

A zero high bit means candidate A is displayed left; a one high bit means candidate B is displayed left. Presentation IDs must be frozen before assignment. The commitment is internally verifiable, but it is not proof of when the commitment was made. Production studies should anchor the commitment in an independent timestamped or signed receipt and keep the seed secret until collection closes.

The assay never treats ties, neither, or skips as left/right votes. They remain visible in the report and are not silently recoded. Repeated judge/case or judge/pair exposure is rejected to avoid memory contamination from showing the same pair twice to one judge.

## Scope limits

The included four-row fixture is synthetic and deliberately too small to flag side bias. It demonstrates only structural and deterministic replay. The crate does not authenticate raters, verify that a UI actually displayed the audio, assess audio rights, establish human preference, or prove that the seed commitment was externally timestamped.
