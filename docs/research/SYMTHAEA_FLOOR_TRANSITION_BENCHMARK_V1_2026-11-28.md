# Symthaea floor-transition benchmark v1 — 2026-11-28

## Motivation and external baseline

MultiPartyTurnBench (MPTB) evaluates multi-party floor control using 240 synthetic channel-separated conversations with two to five speakers and six interaction types. It separates end-of-turn (EOT) detection from interruption detection. The project reports EOT recall of 82.5–92.6% at false-positive rates of 4.3–11.8%; selective interruption handling for five full-duplex agents is only 46.9–54.7% balanced accuracy.

This is the distinction Symthaea must preserve: detecting that an utterance ended is not the same as deciding that the assistant should take the floor.

## Implemented evaluator

Source: `crates/domains/symthaea-soma/src/embodiment_floor_transition.rs`.

The evaluator consumes annotated event windows and probability/timestamp predictions. It maintains two independent tasks:

1. **End-of-turn detection:** positive on annotated EndOfTurn and SuccessfulTakeover events.
2. **Selective interruption:** positive only on SuccessfulTakeover events.

Temporary holds, backchannels, other-directed speech, ambient speech and no-floor-change events are negatives unless the annotation explicitly marks a floor event.

Each task reports:
- TP, FP, TN and FN counts;
- recall, false-positive rate, precision and balanced accuracy;
- threshold-free AUROC using the Mann–Whitney pairwise ranking estimator;
- whether the configured false-positive budget is met;
- signed detection latency in milliseconds (negative = early, positive = late).

A prediction crossing its configured probability threshold must include a decision timestamp; a timestamp below threshold is rejected as contradictory. The evaluator enforces one prediction per event, exact clock-domain binding, valid event chronology, finite probabilities and explicit handling of undefined denominators.

The evaluator uses two separately configured thresholds and a shared maximum FPR. The thresholds must be selected on development data and frozen before held-out evaluation; do not tune them on the test set.

## Proposed qualification targets (not results)

| Task | Target |
|---|---:|
| EOT recall | >=95% at FPR <=5% |
| Selective interruption balanced accuracy | >=80% at FPR <=5% |
| Selective interruption recall | >=85% at FPR <=5% |
| False takeover on holds/backchannels/other-directed/ambient speech | <=5% |
| Median signed detection latency | report, target within +/-100 ms |
| p95 positive (late) detection latency | <=250 ms |
| AUROC | >=0.95 EOT; >=0.90 selective interruption |

The latency target should be interpreted together with precision/recall and signed-latency distributions. A detector that acts 200 ms early on every pause is not a successful turn-taking system.

## Experimental protocol

- Split by speaker and conversation/session, not by windows from the same recording.
- Include two-, three-, four- and five-speaker conditions.
- Test temporary holds, breath pauses, backchannels, interruptions, side conversations, background speech, overlap and successful takeover.
- Use frozen hidden decision points and false-positive-constrained recall.
- Report EOT and interruption confusion matrices independently.
- Sweep thresholds on development data and publish test-set PR/ROC curves.
- Cluster-bootstrap by conversation/session; publish per-condition uncertainty and sample counts.
- Add grader mutation tests for missing predictions, duplicate IDs, wrong clocks, impossible timestamps and mismatched detection probabilities.

## Relation to the other Symthaea evaluators

- `embodiment_turn_control.rs` validates one-candidate/one-decision turn state transitions.
- `embodiment_multiparty_control.rs` scores addressee recognition, silence and human-resolution behavior.
- `embodiment_stream_engagement.rs` scores continuous participation windows, response quality and timing.
- This module isolates EOT and selective interruption with explicit FPR constraints.

Together these avoid collapsing distinct abilities into one turn-taking score.

## Evidence boundary

Twelve source-level tests are authored but not compiled or executed. No MPTB data, recordings, annotation files or model predictions were loaded. No EOT/INT score or latency measurement exists. The evaluator is not a speaker/turn detector. MPTB data rights and any dataset transformations must be reviewed through the benchmark provenance/license gate before use.
