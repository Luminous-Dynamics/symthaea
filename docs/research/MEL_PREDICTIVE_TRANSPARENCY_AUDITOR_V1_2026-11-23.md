# MEL Predictive Transparency Auditor v1

**Status:** standalone Rust evaluator with synthetic known-answer fixtures. No live forecast model, musician, display device, MIDI device or audio loopback has been connected.

## Objective

Measure whether a musical system exposes an inspectable commitment early enough for a collaborator to anticipate, steer, veto, branch, ignore or abstain before the corresponding musical output occurs.

A textual explanation shown after an event is not a forecast. A prediction generated in advance but never delivered or displayed is not usable transparency. A forecast from another session is not a valid contingent preview. A musician's deliberate deviation from a preview is not automatically a system failure.

This implements a deterministic evaluator slice for #10190, Predictive Transparency and Musician Anticipation.

## Evidence contract

Each forecast is append-only and binds:
- session, case, clock domain and unique forecast ID;
- commitment series, revision number and superseded forecast ID;
- exact input snapshot ID, digest and availability time;
- policy digest;
- target start/end window and expiry;
- predicted symbolic event commitments, role, beat index, expected onset, local beat period and presence probability;
- canonical time-map ID/version/kind, source clock-domain ID and receipt digest;
- a time-alignment receipt that explicitly names source and target clock domains, a validity interval and a maximum declared uncertainty bound.

All lifecycle timestamps and expected event times scored by this evaluator must be normalized to the declared evaluator clock domain. The canonical mapper's source clock domain is recorded separately. A time-alignment receipt must explicitly bind source to target domain, declare its validity interval and maximum uncertainty, and cover the forecast target window. The scorer adds that uncertainty to the absolute onset error before applying its tolerance gate. Notation beats, performance seconds, runner monotonic time and CPAL stream time are distinct domains; this evaluator does not derive a conversion between them.

A revision may use a newer input snapshot, but that snapshot must have become available before the new forecast was committed. The revision chain must be contiguous and explicitly supersede exactly the previous version. A changed policy requires a new series rather than silently changing the meaning of an existing forecast.

The system never needs to expose hidden chain-of-thought. The forecast contains actionable musical intent and uncertainty, not private reasoning.

## Display and human agency

The evaluator separates:
1. forecast commit;
2. delivery to the interface;
3. actual display;
4. collaborator disposition (accept, steer, veto, branch, ignore, abstain);
5. later realization and exact output artifact.

A display receipt records that the preview was rendered, not that a person understood it. Human anticipation and interpretability require separate participant-level measurement.

Steer and branch require a superseding forecast. A vetoed forecast must not be realized. User deviations are recorded as unforecasted output, but are not labelled disobedience or automatically counted as a failure of musical quality.

## Metrics

- Forecast integrity: precommitment, session-input binding, current revision, expiry, display lead, supersession continuity and disposition ordering.
- Contingent availability: whether a preview was rendered before its target window, and the distribution of preview lead time.
- Forecast realization: exact forecast-event links, role agreement, beat-index agreement, beat-phase/onset error and unfulfilled forecast commitments. Beat-index agreement is separate from onset error: an event can sound close in time but land on the wrong declared beat.
- Conditional presence calibration: Brier score over the declared forecast event slots, comparing predicted presence probability with whether that commitment was realized.
- Agency: counts of accept/steer/veto/branch/ignore/abstain and explicit user departures from the preview.

The presence Brier score is conditional on the forecast's declared event slots. It is not a complete event-recall or general probability-calibration study because it does not include a separate universe of all plausible but unforecasted musical events. A larger confirmatory benchmark needs an independently defined target-event universe and calibration analysis.

No opaque composite transparency score is emitted.

## Synthetic negative controls

The Rust suite covers:
- post-hoc forecast commit;
- a forecast whose input digest does not match its snapshot (yoked preview);
- a forecast citing input that became available only after commit;
- late display;
- stale realization after expiry;
- skipped/forked revision and output from a superseded forecast;
- veto followed by realization;
- steering without a new forecast revision;
- duplicate event realization;
- deliberate musician deviation from the forecast;
- overconfident versus low-probability event-presence predictions.

## Run

    cargo +1.96.0 run --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml \
      --bin mel-musical-forecast-audit -- audit \
      tools/mel-judge-grounding-audit/fixtures/valid-musical-forecast-v1.json \
      report.json

    cargo +1.96.0 test --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --all-targets -- -D warnings

## Competitive comparison

- JAM_BOT (NIME 2026) demonstrates expressive symbolic call-and-response, velocity, variable-tempo interaction and MIDI latency compensation. It is a relevant live-partner comparator, not a direct forecast-calibration leaderboard.
- Agents in Concert (IUI 2026) reports a two-month musician co-design process and a live concert with 97 audience members; one design intervention let performers anticipate upcoming AI responses through visual forecasts. The causal benefit of preview is still a hypothesis to test, not an established effect.
- HEARTBEAT 2027 is an adjacent streaming-turn-taking challenge. Its WAIT/RESPOND/CONTINUE/YIELD and no-lookahead discipline is relevant, but it is not a music-specific forecast or musical agency benchmark.
- Daydream's real-time music research reports separate latency, chord, melody, groove and accompaniment measurements. Those are output-control measurements, not musician-facing precommitment or steerability evaluations.

The next human study should compare no-preview, post-hoc, yoked-preview, single-preview, uncertainty-aware alternatives and steerable-preview arms under matched compute and content budgets. Outcomes must separately measure anticipation accuracy, response preparation/entry timing, ownership, perceived control, conformity/anchoring, creative diversity and willingness to continue. Use musician- and composition-held-out splits and cluster inference by participant/session.

## Claim ceiling

The committed fixture contains invented timestamps, mock canonical time-map provenance, mock time-alignment provenance and synthetic event commitments. The tool validates declared clock-order relationships and receipt shapes; it does not authenticate receipts, prove the musician saw/understood the screen, establish a real model forecast, measure acoustic output, or show that previews improve collaboration. Real claims require the actual canonical mapper, a qualified performance-to-runtime clock alignment, UI display receipts, independently measured runtime timing and a preregistered musician study.


## Output observation closure (MEL-OBS-CLOSE-001 / #13311)

A missing forecast-event link is not automatically an observed absence. Every forecast now requires exactly one observation-closure receipt. The receipt states the observation kind (symbolic score, MIDI event log or audio capture), clock domain, coverage interval, observed-through time, completion status, output-evidence digest where available, and observation-receipt digest.

Presence outcomes are tri-state:
- **present:** the exact forecast event was observed inside the captured interval;
- **absent:** a complete capture covers the full half-open forecast target interval and the event was not observed;
- **unknown:** observation is partial or unavailable.

Unknown outcomes are counted but excluded from Brier scoring. Unfulfilled forecasts count only observed-absent event slots. The report exposes observed-present, observed-absent, unknown and scored-event counts plus an explicit completeness flag. A complete observation's output-evidence digest must equal the realization artifact digest when realized events are present.

A digest-shaped receipt is still not authenticated capture evidence. This layer prevents missingness from being silently treated as a negative label; it does not prove that the capture device ran or that a waveform detector found every perceptual event.
