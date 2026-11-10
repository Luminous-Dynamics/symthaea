# Vocal pitch benchmark evaluator

A dependency-free Rust evaluator for comparing monophonic pitch trackers against aligned, independently annotated singing F0 references.

## Scope

This tool consumes frame-aligned F0 observations. It does not read audio, resample, run trackers, choose confidence thresholds, infer voiced frames, or construct statistical confidence intervals.

The primary track rejects score-conditioned pitch estimation. A tracker adapter must apply its frozen voicing policy and emit None for an unvoiced/abstained frame. The policy identifier, tracker artifact digest, reference manifest digest, preprocessing manifest and frame-alignment policy are all bound into the report.

## Metrics

- **FFE@threshold:** fraction of all frames with a voicing mismatch or a gross pitch error.
- **GPE@threshold:** gross pitch errors divided by jointly voiced reference/estimate frames; undefined when no jointly voiced frame exists.
- **VDE:** voicing mismatches divided by all frames.
- **Reference-voiced coverage:** jointly voiced frames divided by reference voiced frames.
- **Abstention rate:** reference-voiced frames where the tracker abstains.
- **False-voiced rate:** estimated voiced frames where the reference is unvoiced.
- **Octave error rate:** octave/subharmonic errors among jointly voiced frames.
- **Cents error:** median signed/absolute, p95 absolute and maximum absolute cents error on jointly voiced frames.

A gross pitch error uses the predeclared cents threshold (50 cents by default in the test fixtures). The threshold and voicing policy must be chosen on a disjoint calibration split before evaluating held-out segments. The evaluator does not tune them against the test split.

## Aggregation

The report preserves:
- every per-segment result;
- exact source-audio, reference-F0-track and estimated-F0-track digests;
- singer, song and recording-session identities;
- language, register, vocal-technique and SNR strata;
- corpus and recording-condition labels;
- micro metrics pooled over all frames;
- per-corpus × recording-condition groups;
- equal-weight macro metrics across those groups;
- median per-segment FFE and VDE.

The specification declares the complete expected corpus × condition group set. Missing or unexpected groups reject the run; a partial run cannot silently change the macro denominator. For the external mixed pitch benchmark's headline methodology, declare the eight non-clean conditions for each expected corpus and report clean audio separately. For Muse's singing-only study, define a separate group set appropriate to the licensed corpus.

Use group-level outputs for corpus/condition macro reporting. Do not use raw frames as independent bootstrap samples. Comparative confidence intervals should resample singer/song/session clusters and preserve matched source cases across trackers. The library intentionally does not emit confidence intervals because it does not own the sampling design or resampling policy.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/vocal-pitch-benchmark-eval/Cargo.toml
cargo +1.96.0 test --manifest-path tools/vocal-pitch-benchmark-eval/Cargo.toml
~~~

## Research-only data boundary

The external mixed monophonic-pitch benchmark dataset is CC BY-NC-SA 4.0 and marked research use only. Its aggregate ranking includes speech and music corpora, so it is a reference-tracker shortlist and methodology check, not a singing-only SOTA result. Do not redistribute the dataset, incorporate it into a commercial training corpus, or claim commercial validation from its scores without separate rights clearance.

SoulX-Singer-Eval/GMO-SVS and any real vocal references also require per-dataset licence, consent and redistribution review. The dataset's benchmark code licence does not override source-corpus terms.

## Claim boundary

Passing synthetic tests establishes only metric arithmetic, grouping and malformed-input rejection. A SOTA or production claim requires a separate licensed/consented singing corpus, manually reviewed sample-level F0, singer/song/session-disjoint splits, matched tracker adapters and cluster-aware uncertainty.
