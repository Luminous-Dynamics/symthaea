# Lyric edit assurance evaluator (VMEA-001)

A dependency-free Rust scorer for task-aware vocal and audio editing evaluation.

## Core principle

An edit must satisfy both:
1. Instruction realization — the requested operation happened.
2. Preservation — the declared properties outside or alongside the edit remain acceptable.

The evaluator does not collapse these into a single aesthetic score. It reports Instruction Following Rate (IFR), Consistency Rate (CR) and strict Exact Match Rate (EMR), plus every per-case and per-rubric decision.

## Supported task families

LyricEditBench operation values:
- Partial substitution
- Full substitution
- Deletion
- Insertion
- Translation
- Code-mixing

A general audio-edit operation is also representable for a separately declared MMAE-style rubric profile. This does not imply LyricEditBench and MMAE use the same tasks or official evaluators.

## Three-way interval decisions

Each required rubric declares:
- metric ID and unit;
- instruction or preservation role;
- frozen accepted interval.

An independently versioned analyzer or human panel supplies an interval:
- PASS — the entire interval is inside the accepted range;
- FAIL — the entire interval is outside the accepted range;
- INDETERMINATE — the interval overlaps a decision boundary.

Indeterminate does not count as a pass. Missing, duplicate, extra, stale or malformed observations reject the produced case.

## Denominators and failures

Every case in the frozen manifest requires exactly one execution record. Produced, unsupported, failed and abstained cases are retained. Unsupported/failed/abstained cases have no output metrics but all their required rubrics remain unscored in IFR and CR denominators and receive no EMR credit.

A produced case is an exact match only when every required instruction and preservation rubric passes. A single failed or indeterminate rubric prevents exact-match credit.

## Provenance

The run binds:
- benchmark and system identity/version/artifact digest;
- evaluation and calibration split IDs and distinct manifest digests;
- calibration receipt;
- complete predeclared operation × language × timbre-mode group matrix.

Each case binds:
- source audio, melody reference and optional timbre reference; same-timbre requires the timbre-reference digest to equal the melody-reference digest, while cross-timbre requires a distinct digest;
- original and requested lyric payload digests for lyric tasks;
- source singer/song/session and technique strata;
- task manifest digest;
- every rubric's evaluator ID/version/artifact digest and calibration receipt.

This crate validates digest syntax and consistency. It does not prove source-data rights, split independence or truthfulness of upstream analyzer claims; those require external audits.

## Metrics

- IFR = passing instruction rubrics / all required instruction rubrics.
- CR = passing preservation rubrics / all required preservation rubrics.
- EMR = exact-match cases / all manifest cases.
- Group macro results are the unweighted mean over the predeclared groups.
- Per-case and per-rubric reports retain failures and indeterminate decisions.

These are locally computed from the declared rubric profile. They are not official MMAE scores unless the exact official rubric mapping and official judge results are separately established.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/lyric-edit-assurance-eval/Cargo.toml
cargo +1.96.0 test --manifest-path tools/lyric-edit-assurance-eval/Cargo.toml
~~~

## Benchmark rights

LyricEditBench is CC BY-NC-SA 4.0. Do not use it for commercial training or redistribute it without separate permission. Verify underlying recording/source-corpus rights. The YingMusic-Singer-Plus project describes CC BY 4.0 for code/weights with Stable Audio-derived VAE components under the Stability AI Community License.

The ICASSP 2027 MMAE challenge has its own development/test data, official SDK, private submission credentials and eligibility rules. This local scorer does not submit to the competition and does not reproduce the official Qwen3-Omni judge.

## Claim boundary

Synthetic tests qualify only the rubric contract and aggregation logic. They do not establish real lyric intelligibility, melody adherence, identity preservation, naturalness, or SOTA performance.
