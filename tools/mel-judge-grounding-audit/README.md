# MEL Judge Grounding Audit

A standalone Rust reference evaluator for causal and metamorphic probes of music judges. It asks whether a judge actually uses audio evidence, remains stable under instruction paraphrase, ignores declared nuisance transformations, and responds to controlled semantic mismatches.

## Run

    cargo run --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --bin mel-judge-grounding-audit -- \
      audit tools/mel-judge-grounding-audit/fixtures/valid-probe-panel-v1.json \
      report.json

    cargo test --manifest-path tools/mel-judge-grounding-audit/Cargo.toml
    cargo clippy --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --all-targets -- -D warnings

## Probe conditions

Each case binds a baseline audio A/B score and exact prompt, instruction and audio digests. It may contain:

- \`prompt_only\`: remove both audios while keeping the prompt and rubric. The pairwise discrimination margin should collapse relative to the full audio+prompt condition.
- \`audio_swap\`: exchange the two audio assets while keeping prompt, instruction and candidate slots fixed. The score margin should reverse.
- \`instruction_paraphrase\`: keep audio and prompt fixed while replacing the rubric with a reviewed semantic paraphrase. The pairwise ranking should remain stable.
- \`nuisance_transform\`: replace one or both audio assets with a validated semantics-preserving transform. The pairwise margin should remain within a per-probe tolerance.
- \`semantic_mismatch\`: alter exactly one candidate's audio to violate a declared prompt attribute. That candidate's score should fall by at least the per-probe minimum.

The scoring-time probe panel is assembled only after the judge-output package is frozen and the independent human reference is released. The assembled panel contains both probe outputs and post-freeze human-reference directions; it must never be mounted or sent to the judge. The evaluator reports baseline human-reference accuracy on clear cases. The sealed execution workflow in MEL-SEALED-EVAL-001 remains responsible for proving prediction freeze precedes reference release.

## Interpretation

This is not a music-quality model. It is an *instrument qualification* tool: it can expose prompt dependence, failure to follow audio swaps, rubric instability, nuisance sensitivity and semantic-mismatch blindness. A judge can pass one probe and fail another; no single composite score is emitted.

All tolerance values are predeclared in the probe panel. Nuisance transforms must be human-validated as semantics-preserving for the target task. A tempo change is not a nuisance when the prompt specifies tempo; a channel conversion is not automatically safe if it removes spatial cues relevant to the prompt.

The evaluator checks supplied digest-shaped identifiers and declared relationships. It does not open audio files, verify transformations, authenticate model execution, or prove the judge was isolated from the human reference. Synthetic fixtures establish only Rust evaluator conformance.


## Task-scoped transform qualification

A separate reference validator qualifies each transformation for one declared task dimension. It recomputes the source and variant file digests, validates the task/preservation declarations and rights-scope receipt digests, and requires an independent listener-equivalence interval wholly inside the preregistered margin with adequate listener and item counts. An underpowered, inconclusive or out-of-margin receipt is ineligible, not a pass.

Run it with:

    cargo run --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --bin mel-audio-transform-qualification -- audit \
      tools/mel-judge-grounding-audit/fixtures/valid-transform-qualification-v1.json \
      tools/mel-judge-grounding-audit/fixtures/source-audio.bin \
      tools/mel-judge-grounding-audit/fixtures/variant-audio.bin report.json

The CLI requires `sha256sum` on PATH. Its committed byte fixtures are not playable audio and its panel receipt is synthetic. It does not authenticate receipts or verify decoded audio metadata; see the separate transform-qualification research note.


## Live collaboration trace audit

The collaboration trace auditor validates causal input availability, event/decision/output order, WAIT/RESPOND/CONTINUE/YIELD semantics, protected rest windows, missed deadlines, response completeness, and optional independent audio loopback timestamps. It reports emission latency separately from audible latency; a generation request is never treated as proof that sound was heard.

    cargo run --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --bin mel-collab-trace-audit -- audit \
      tools/mel-judge-grounding-audit/fixtures/valid-collab-trace-v1.json report.json

The fixture is synthetic and MIDI-style. It proves only deterministic evaluator behavior, not actual real-time performance. See docs/research/MEL_COLLAB_TRACE_AUDITOR_V1_2026-11-22.md for the contract and claim boundaries.


## Beat-phase and tempo-adaptation audit

The musical timing evaluator consumes expected onset and local beat-period values from the canonical #9477 time mapper, bound by map ID/version/kind and a receipt digest. It does not implement a second beat-to-time map. It scores signed/absolute beat-phase error, per-instrument strata, target/missed/extra event accounting, descriptive phase drift and recovery after tempo changes using consecutive, spaced estimates. Each estimate binds the end of the evidence window; future evidence is rejected.

    cargo run --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --bin mel-musical-timing-audit -- score \
      tools/mel-judge-grounding-audit/fixtures/valid-musical-timing-v2.json report.json

Its MIDI-style fixture is synthetic. No real timing or performance claim follows from the example output.


## Predictive transparency and musical forecasts

The musical forecast auditor checks precommitted, input-snapshot-bound forecasts against later symbolic realization. It requires a source/target clock-alignment receipt, target-window validity and a declared uncertainty bound; that uncertainty is included in worst-case beat-phase error before tolerance qualification. It records delivery, actual display and collaborator disposition separately; verifies revision/supersession history; rejects post-hoc, yoked, future-input, stale and vetoed output; and reports conditional event-presence Brier, role and beat-index agreement, beat-phase error and preview lead time. Deliberate user deviation remains valid agency.

    cargo run --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --bin mel-musical-forecast-audit -- audit \
      tools/mel-judge-grounding-audit/fixtures/valid-musical-forecast-v1.json report.json

The fixture is synthetic. Its Brier score is conditional on predicted event slots, not a general recall/calibration result. See docs/research/MEL_PREDICTIVE_TRANSPARENCY_AUDITOR_V1_2026-11-23.md.


## Musical agency and non-assimilation audit

The agency auditor distinguishes hard current-operation constraints from soft collaborator preferences and agent proposals. It checks protected-event and veto compliance, explicit ignore/accept/steer behavior, non-canonical branching, cross-collaborator memory leakage, and purpose-scoped persistent-write authorization. It emits no composite creativity or agency score.

    cargo run --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --bin mel-musical-agency-audit -- audit tools/mel-judge-grounding-audit/fixtures/valid-musical-agency-v1.json report.json

The synthetic fixture validates structural behavior only. It does not authenticate collaborators, verify signatures, or execute Studio writes. See docs/research/MEL_MUSICAL_AGENCY_AUDITOR_V1_2026-11-23.md.
