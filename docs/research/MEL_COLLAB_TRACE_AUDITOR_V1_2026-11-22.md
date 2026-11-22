# MEL Collaboration Trace Auditor v1

**Status:** standalone Rust trace auditor with a synthetic MIDI-style fixture. No live Symthaea/Melothaea audio engine is connected, and no musician session, physical latency, or audible result is claimed.

## Objective

Evaluate the interaction over time, not just the generated artefact. This first version checks whether a collaboration trace is causally ordered, whether each musical decision has its corresponding observable output, whether the system leaves instructed rest windows alone, and what timing can be calculated from supplied timestamps.

Each session binds a task family (call/response, adaptive accompaniment, phrase continuation, structural transition, silence/restraint, or recovery), a pseudonymous musician ID, instrument, agent role, hardware profile and evaluation-consent receipt digest.

## Event contract

All nanosecond timestamps in the trace are expressed in the declared clock_domain_id. Capture/context timestamps must already be mapped into this monotonic clock domain. Source frame offsets, if needed, belong in a future schema field and must not be compared directly with monotonic nanoseconds.

Events have a contiguous trace sequence:
1. input_chunk: input sequence, context start/end, availability time, and exact media digest.
2. decision: references an input chunk already available; carries WAIT/RESPOND/CONTINUE/YIELD, reason, confidence and (for active responses) deadline.
3. output: must reference a preceding RESPOND/CONTINUE decision and the same input sequence, with contiguous output sequence, output digest and first-emission time.

The auditor rejects non-monotonic trace time, event/input/output sequence gaps, duplicate event IDs/output sequence, unknown or future input references, context available after its event, output before its decision/input, output after WAIT/YIELD, and clock-domain mismatch for loopback data.

## Rest and latency

Protected rest windows must include at least one WAIT or YIELD decision. An active RESPOND/CONTINUE decision during a rest window is separately reported from an actual interruption.

For the MIDI track, output first-emission time is observable in the trace and is used to detect a note/event emitted during a rest. For rendered audio, an interruption is counted only when an audible onset timestamp is supplied from a separate loopback measurement in the same declared clock domain. If no audible timestamp exists, the report marks rendered-audio latency as incomplete; it does not infer sound from an audio-generation request.

The report calculates input-availability-to-first-emission latency and, when measured, input-availability-to-first-audible-onset latency with nearest-rank p50/p95/p99. It does not subtract timestamps from unrelated clocks. The receipt digest is checked for shape and scope only; it is not authenticated.

## Run

    cargo +1.96.0 run --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml \
      --bin mel-collab-trace-audit -- audit \
      tools/mel-judge-grounding-audit/fixtures/valid-collab-trace-v1.json \
      report.json

    cargo +1.96.0 test --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-judge-grounding-audit/Cargo.toml --all-targets -- -D warnings

## Negative controls

Tests cover backdated timestamps, future input access, input context extending beyond availability, active decisions and MIDI output during protected rest, missing WAIT/YIELD, output from WAIT, duplicate output sequencing, late response, unmeasured audible onset, loopback clock mismatch and a response decision with no output.

## Claim ceiling

The committed trace is a synthetic conformance fixture with invented timestamps and digests. The validator does not run a music model, capture audio/MIDI hardware, independently verify loopback receipts, authenticate consent receipts, prove model quality or establish musician preference. A passing trace proves only that the supplied event log satisfies this schema and its declared checks. Real-time performance needs a hardware-specific live capture and independently measured loopback.
