# Open Audio Evidence Contract v1.0.0

**Status:** proposed common manifest and test-data policy; no hardware qualification implied.  
**Parent tracker:** [OPEN-AUDIO-EVIDENCE-001 #7861](https://github.com/Luminous-Dynamics/symthaea/issues/7861).  
**Flagship programme:** [#7848](https://github.com/Luminous-Dynamics/symthaea/issues/7848).

## Why this is a paradigm shift

A frequency-response image is not an auditable result by itself. It omits the device build, fit or angle, calibration, raw captures, analysis code, uncertainty and exclusions needed to reproduce a conclusion.

The shared contract treats a measurement as a small reproducible experiment:
1. identify the exact device/sample and source revision;
2. identify the fixture, instruments and calibration;
3. pin the protocol and settings;
4. preserve each original capture and its SHA-256 hash;
5. record the processing tool and immutable source revision;
6. report computed values with units, sample counts and uncertainty;
7. preserve deviations, limitations and failed/excluded runs;
8. separate a measured result from an internal gate decision and from any formal standards-compliance claim.

Files must be relative paths; absolute paths and parent-directory traversal are rejected by the schema. Missing values must not be silently invented as zero. Unknowns should be omitted or clearly described, and limitations must be explicit.

## Files

- `schemas/audio-evidence-manifest.schema.json`: JSON Schema Draft 2020-12.
- `schemas/examples/h1-fit-repeatability-plan.json`: an intentionally `planned` example, not a measurement report and not evidence of a prototype.
- `docs/research/OPEN_AUDIO_EVIDENCE_CONTRACT_2026-10-17.md`: this protocol.

The lifecycle is `planned`, `captured`, `analyzed`, `validated`, `invalid`, or `excluded`. Planned runs cannot carry captures/results or a pass decision. Any run that started must link its fixture, instruments, environment and at least one raw capture. The schema requires a raw capture role for every started run, and captured/analyzed states cannot claim pass or fail. Analyzed/validated states also require source-pinned analysis and structured results. Validated means the declared project gate was evaluated—it does not mean product safety or formal standard certification.

Schema v1 is an evidence index, not a universal interchange format for every waveform and response array. Keep native raw data in durable versioned files and reference it with path, SHA-256, media type and size. Derived CSV/JSON/plots are separate artifacts and never replace raw captures.

### Executable integrity checker

The Rust package `symthaea-audio-evidence` provides the offline CLI `audio-evidence-check`. It checks project-specific lifecycle and required-field invariants, validates artifact path syntax, rejects incomplete artifact references, verifies the exact SHA-256 and optional byte length of referenced bundle files, and rejects file symlinks that resolve outside the bundle root.

Example from the repository root:

~~~sh
cargo run -p symthaea-audio-evidence --bin audio-evidence-check -- schemas/examples/h1-fit-repeatability-plan.json
cargo test -p symthaea-audio-evidence
~~~

The checker deliberately does **not** claim to implement all JSON Schema Draft 2020-12 keywords. It verifies selected high-risk project invariants and file integrity; a standards-compliant JSON Schema validator remains a separate gate. A successful check does not prove calibration, physical validity or standards conformity.

## Headphone protocol: separate best fit from robustness

Use IEC 60268-7:2025 as the current standard reference and obtain the licensed text before claiming compliance. State the ear simulator/coupler model and its limits. Recent public work from Warren Labs is a useful example of why seating rules, capture counts, calibration, pad state and repeatability belong next to a curve: [method and repeatability protocol](https://warrenlabs.com/method/).

H1 should produce two different views:

**A. Deliberate listening fit.** Find a repeatable best-fit region using a declared positioning method. Capture every position and retain individual responses. Choose a representative real capture using a frozen rule (for example a medoid), not a synthetic average that may describe no actual fit. If fit distributions are multimodal—such as good seal versus broken seal—report them separately.

**B. Fit sensitivity.** Re-seat repeatedly and deliberately perturb factors likely to affect fit: headband/clamp setting, pad condition, head/ear position, and glasses-like seal interruptions where relevant. Label every perturbation and channel. Report distribution by band (at minimum bass/seal, midrange/control, and treble/pinna-sensitive), not one undifferentiated spread. Establish fixture repeatability controls before attributing changes to the headphone.

Do not blindly adopt one bench's capture count as a universal rule. Pilot the fixture and the specific headphone first, determine whether more placements materially reduce uncertainty, and freeze a justified capture plan before confirmatory comparison. Warren Labs' published 19-captures-per-side procedure is an example, not our universal requirement. Do not combine a well-sealed group and a broken-seal group into a curve that represents neither.

Keep left/right channels separate. Include pad type, age/condition, clamp setting, coupler and fixture IDs, calibration files, gain/drive settings, sample settling, sample/build ID, placement map and rerun reason. State where a coupler simulates an ear versus where it functions only as a repeatable acoustic coupler.

## Loudspeaker protocol: response, directivity and output are separate

Use the actual applicable CTA-2034-B procedure before making standard-based claims; its scope is residential loudspeaker-system response, directivity and maximum output, not raw-driver properties or input-survival safety: [CTA scope](https://www.cta.tech/standards/ansicta-2034-b/).

Retain:
- on-axis and listening-window responses;
- horizontal/vertical angular responses with exact angles;
- directivity, early-reflection and sound-power-derived data when the declared method supports them;
- low-frequency near-field/ground-plane or other method data, including transition/merging procedure;
- impedance magnitude/phase;
- distortion and compression over declared frequency/output points;
- thermal state, amplifier/load assumptions and limiter/failure conditions.

Do not call an exploratory angle raster CTA-2034-B-compliant unless it follows the actual procedure. Keep single-speaker and system/subwoofer measurements distinct. Keep anechoic, near-field, ground-plane and in-room traces labelled; do not merge incompatible measurements without a documented transition and uncertainty.

Use AES75-2023's Music-Noise procedure for maximum-linear-output claims when applicable: [AES75 resources and test signals](https://aes.org/standards/AES75/). Ordinary sweeps or a marketing peak-SPL number are not a substitute. Treat high-SPL testing as controlled lab work; do not expose listeners to hazardous levels.

For room correction, publish uncorrected and corrected response, each microphone position, correction limits and filter revision. A room-EQ result at one point does not establish a directivity fix or uniform performance across the listening area.

## Electronics protocol

For headphone amplifiers, DACs and power amplifiers, identify source/load impedances and measure frequency response, noise, THD+N, channel tracking, crosstalk, clipping, output impedance, thermal behavior and relevant startup/shutdown/fault behavior across the declared operating envelope. Store each load/power point as its own condition. A best-case 1 kHz result is not the full performance envelope and does not establish safety or compliance.

## Data-quality and claim rules

- Hash the exact file bytes; any re-encoding creates a new artifact and hash.
- Protocol and analysis revisions must be immutable. Mutable branch names may add context but cannot replace a commit SHA or versioned release.
- Failed or invalid runs remain in the record with a reason; never silently delete an inconvenient capture.
- Exclusion is a transparent decision, not permission to erase evidence.
- Report sample counts, repeated placements, per-unit variation, instrument/fixture limitations and uncertainty.
- Do not combine data from different fixtures as if they share an axis. Cross-calibration requires a published mapping and residual uncertainty.
- Record source and license provenance for third-party data. Do not redistribute material under a license that disallows intended use.
- Listening preference is a separate outcome requiring randomized, safely level-matched trials and appropriate participant safeguards.
- A product claim requires a preregistered common-protocol comparison, with the observed difference greater than relevant uncertainty.

## Future validator acceptance checks

1. Parse under a Draft 2020-12 validator and reject unknown fields.
2. Reject malformed SHA-256, absolute paths and parent traversal.
3. Reject a `planned` manifest containing captures/results or a pass/fail decision.
4. Reject `analyzed`/`validated` status without analysis revision and structured results.
5. Reject `validated` status with `not_evaluated` decision.
6. Require fixture, instruments, environment, start time and raw capture references once a run starts.
7. Verify artifact existence and actual byte hash in the release bundle.
8. Run known-answer fixture controls before using product traces in target optimization.
9. Keep schema validation separate from physical validity: a valid manifest can still describe a poor measurement.

## Research references

- [IEC 60268-7:2025](https://webstore.iec.ch/en/publication/86573) — headphone characteristics and measurement methods, published 2025-06-27.
- [Warren Labs measurement method](https://warrenlabs.com/method/) — public example for headphone reseating, capture provenance and fixture limits.
- [ANSI/CTA-2034-B](https://www.cta.tech/standards/ansicta-2034-b/) — residential loudspeaker response/directivity/output scope.
- [AES75-2023](https://aes.org/standards/AES75/) — maximum linear SPL using Music-Noise.
- [Spinorama measurement-quality FAQ](https://www.spinorama.org/docs/05_faq.html) and [data format/API documentation](https://www.spinorama.org/docs/09_api.html) — precedent for open measurement data with quality tiers and caveats.

This contract is a project proposal. First use should be the planned H1 fixture repeatability experiment; do not treat it as a measurement until the fixture and instruments are selected and limitations recorded.
