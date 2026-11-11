# SA-SING-001 singing benchmark manifest validator

A Rust metadata validator for a performer-consented South African multilingual singing evaluation set.

## Purpose and boundary

This crate admits **singing vocal recordings for evaluation only**. It does not download or ingest audio, train models, synthesize voices, grant voice-identity permissions, determine legal sufficiency, or create public datasets.

Every case references:
- an exact immutable language-profile digest (from the audited language-profile work);
- an exact corpus-registry snapshot and item-level evaluation-eligibility receipt (from #9509);
- a pseudonymous performer, song/work-family and recording-session identity;
- one or more time-bounded language spans;
- explicit supported evaluation tracks and their required reference artifacts.

The crate does not implement a second rights registry. The registry owns the detailed rights/consent decisions. This validator requires the case's upstream registry receipt to state approved benchmark-evaluation-only use.

## Why speech-only sources are rejected

SAMA, Lwazi and Swivuriso are speech resources, not singing recordings. They may be relevant to separate speech-side pronunciation/ASR tasks under their licences. They do not qualify sung F0, note fidelity, vocal naturalness, singing technique or local edit preservation.

Swivuriso / ZA-African Next Voices explicitly prohibits TTS, voice cloning and voice synthesis; it is excluded from Muse singer training and timbre-reference generation. An evaluation-only receipt is not training or synthesis permission.

Signed South African Sign Language is a separate modality, not an F0-based acoustic singing case. A future signed musical performance benchmark needs its own track.

## Required evidence by track

| Track | Required reference artifacts |
| --- | --- |
| Lyric / phonetic realization | Lyrics, native-reviewed phones, phone-time alignment |
| Score pitch fidelity | Score event map, independent measured F0 track |
| Local edit preservation | Target region map, protected region map, phone-time alignment |
| Code-switch boundaries | Native-reviewed phones, phone alignment, adjudicated switch-boundary labels, at least two language profiles |
| Human perceptual quality | Human-perceptual panel receipt |

An F0 reference cannot be derived from the candidate tracker or solely from the score MIDI. The score event map and measured F0 are separate references.

## Leakage controls

When enabled in the frozen spec, performer pseudonyms, song/work-family pseudonyms, recording-session pseudonyms and source-audio digests may not cross dataset splits. Supported split labels are feasibility pilot, calibration, development, confirmatory and locked test.

A tiny convenience cohort may be registered as a feasibility pilot. It must not be relabelled as an independent confirmatory test. Determine confirmatory sample sizes by singer/session-clustered power analysis after the pilot.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/south-african-singing-benchmark/Cargo.toml
cargo +1.96.0 test --manifest-path tools/south-african-singing-benchmark/Cargo.toml
~~~

## Claim ceiling

A valid manifest proves only that the supplied metadata and receipts satisfy this schema. It does not independently authenticate the registry, participant consent, rights-holder authority, language annotations, measured F0, cultural interpretation, audio quality or benchmark result. A rights receipt remains an upstream decision and must be auditable in #9509.
