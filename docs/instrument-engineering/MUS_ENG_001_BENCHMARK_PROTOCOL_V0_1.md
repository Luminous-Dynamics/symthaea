# MUS-ENG-001 — Musical-instrument engineering benchmark protocol v0.1

Status: proposed contract and synthetic fixtures only. No physical measurements, solver validation, production yield, or musical-preference results are asserted by this document.

Parent work: https://github.com/Luminous-Dynamics/symthaea/issues/8036

## 1. Paradigm shift

Treat the musical instrument as a maintained, reproducible system—not just a waveform or a one-off prototype. The benchmark follows an explicit design-to-build-to-measure-to-revise loop:

Requirements and gestures → editable design + BOM → analytical/numerical models → Pareto candidate set → frozen build package → actual manufactured article and assembly deviations → calibrated measurements → separately governed playability/listening study → model-discrepancy analysis → revision and release.

The following identities and evidence classes are not interchangeable:

- design intent;
- analytical prediction;
- numerical-solver result;
- surrogate/model result;
- manufacturing plan;
- manufacturing execution;
- actual as-built article/configuration;
- calibrated physical measurement;
- human-perceptual result;
- release or repair disposition.

A completed toolpath, plausible render, digest, signed manifest, or successful CI job does not prove that a physical instrument exists, conforms to its design, sounds better, or is safe.

## 2. Reuse established owners

- Symthaea electro-acoustics and existing acoustic, circuit, simulation, digital-twin and fabrication-kernel facilities: https://github.com/Luminous-Dynamics/symthaea/issues/5630
- Objective and perceptual loudspeaker benchmark: https://github.com/Luminous-Dynamics/symthaea/issues/7708
- Muse instrument identity/rendering provenance: https://github.com/Luminous-Dynamics/symthaea/issues/7416 and https://github.com/Luminous-Dynamics/symthaea/pull/8019
- Symtropy whole-life optimization and measured repairability: https://github.com/Luminous-Dynamics/symtropy/issues/1773
- Mycelix hardware-release evidence profile: https://github.com/Luminous-Dynamics/mycelix/issues/2392

This proposal does not create a second CAD kernel, simulator, calibration ledger, BOM system, signing mechanism, or repair-task authority. Where a cross-project adapter is missing, record it as a gap rather than impersonating the existing owner.

## 3. Fixture corpus and claim gates

The corpus at fixtures/instrument-engineering/mus_eng_corpus_v0_1.json is a synthetic policy test set governed by schemas/instrument-engineering/mus_eng_corpus_v0_1.schema.json. Every case is explicitly synthetic. The fixture corpus contains no raw audio, measurements, calibration certificates, hardware, qualified parts, or licensable sound assets.

Expected dispositions:

- CONSISTENT_UNQUALIFIED: record/policy structure is internally consistent, but this does not qualify physical performance.
- INCOMPLETE: required evidence or trial denominator is absent; the claim stays unresolved.
- REJECT_CLAIM: requested claim exceeds or conflicts with its evidence.
- BLOCK_RELEASE: unresolved release-critical prerequisite such as asset rights prevents release.

The validator that may later consume this corpus must preserve those claim ceilings. Merely parsing JSON or passing schema validation must never promote a physical or human-performance claim.

## 4. Family-specific measurements

Do not force all instruments into one universal quality score. Each versioned instrument-family profile must specify hard feasibility constraints separately from named objectives, units, directions, target bands, acceptable uncertainty, measurement protocol, baseline and evidence class.

### Acoustic and mechanical

Depending on the instrument: resonance/modal frequencies, spectral response, decay/sustain, harmonic and inharmonic structure, tuning stability, output level, directivity, force/travel, key or trigger repeatability, mass, stiffness, structural drift and environmental sensitivity.

### Electronics and real-time interaction

Depending on the instrument: noise floor, SNR, distortion, dynamic range, clipping behavior, bandwidth, power and temperature, input-to-output latency distribution, jitter, event loss, controller resolution, calibration drift, interoperability and safe failure behavior.

For interactive digital/hybrid prototypes only, use a provisional design target of median end-to-end input-to-sound latency ≤5 ms, p95 ≤10 ms and p99 latency jitter ≤2 ms at a declared sample rate, buffer size, load and measurement boundary. These are engineering targets to test, not universal psychophysical thresholds or measured results. Report failures and tail distributions; do not average away dropped events.

### Manufacturing and lifecycle

Measure BOM completeness and dated cost evidence; part availability and approved substitutions; tooling; setup/assembly time; dimensional/process variation; failed builds; rework; material waste; energy where measurable; calibration burden; repair task time; part lead time; post-repair function; and recurrence over a stated observation window.

### Musical use

Define a preregistered family-specific gesture/task set before scoring. Measure range, tuning/gesture success, expressive controllability and fatigue/comfort where relevant. Keep blinded listening preference, task performance and objective physical metrics as separate outcomes. Human studies require an appropriate protocol and uncertainty analysis; a few informal listens are not universal preference evidence.

## 5. Proposed benchmark targets

These are targets, not observed results.

### Stage 0 — digital contracts

- 100% of required fields and enumerated evidence classes represented in valid fixtures.
- 100% of named negative controls reject the prohibited claim or block the affected release.
- Zero silent unit conversions, undocumented substitution acceptance, stale-revision acceptance, evidence-class promotion, or unknown-as-pass outcomes.
- Deterministic replay: identical input bytes and tool version produce byte-identical canonical report bytes.
- Any duplicate case IDs, unrecognized fields, malformed units, missing required artifacts or conflicting revision bindings reject.

### Stage 1 — one bench fixture

Freeze excitation, mounting, geometry, electrical gain, sensor position, sample rate, environment, raw-capture format, calibration references, analysis version and exclusion rules. For a first method pilot, target at least 10 repeated trials per declared condition and retain every valid and failed attempt. Use these to estimate repeatability before deciding acceptance bands.

### Stage 2 — replication

Target at least three independently assembled units and, where feasible, three builders. Publish between-unit and between-builder dispersion, not just the best prototype. Compare predicted, simulated and measured results with residuals and declared uncertainty. If the project only has one unit, label the result a single-unit demonstration, not manufacturing capability.

### Stage 3 — robust design search

Require Pareto fronts over separately named objectives. Report feasibility under a predeclared tolerance/environment campaign, worst-case results, sensitivity to uncertain inputs and measurement-to-model discrepancies. A proposed stretch gate is ≥95% feasibility over the declared non-safety operating envelope with zero observed safety-critical violations; the confidence and limitations of that sample must be reported. Simulation-only campaigns remain predictive.

### Stage 4 — lifecycle and musician validation

Compare against a frozen baseline with the same task/protocol and a complete denominator including failures, exclusions, repairs and rework. Preregister material, cost, repair-time or performance improvement thresholds after baseline characterization. Human-study sample sizes should follow the question and a power/precision plan, rather than asserting a fixed small sample is conclusive.

## 6. Standards and interoperability

- MIDI 2.0 extends rather than replaces MIDI 1.0. MIDI-CI is designed to negotiate supported features while preserving backward compatibility. See https://midi.org/midi-ci-specification and https://midi.org/details-about-midi-2-0-midi-ci-profiles-and-property-exchange-updated-june-2023. A first prototype should retain a practical MIDI 1.0 path where appropriate; do not require MIDI 2.0 for all instrument families.
- IEC 60268-21 covers output-based acoustical measurements for electro-acoustic transducers and sound systems; CTA-2034-B has an in-home loudspeaker scope. Use the actual relevant standard procedure and licensed text where applicable. Do not claim either standard qualifies guitars, drums, controllers or every instrument family. https://webstore.iec.ch/en/publication/28687 and https://www.cta.tech/standards/ansicta-2034-b/
- NIST manufacturing literature emphasizes process metrology, part qualification, exemplar data and digital-thread traceability. Adapt those principles proportionately; do not imply additive-manufacturing qualification transfers automatically to other processes. https://www.nist.gov/programs-projects/additive-manufacturing-part-qualification and https://www.nist.gov/publications/towards-digital-thread-and-data-package-metals-additive-manufacturing
- Consider CERN-OHL-S, CERN-OHL-W or CERN-OHL-P for the hardware source, intentionally chosen for the intended reciprocity model. Separately record licenses for firmware, sample libraries, soundfonts, CAD libraries and documentation. https://cern-ohl.web.cern.ch/home

## 7. First physical pilot

Use a guarded low-voltage modular electro-acoustic resonator fixture: replaceable sound-producing element, contact/acoustic sensor, simple low-voltage signal path and optional event/MIDI control. Select the exact geometry, sensor and parts after inventorying available tools and suppliers. The first deliverable is a repeatable test fixture and data contract, not a commercial instrument and not a custom-chip program.

Required bundle when physical work starts: exact editable sources; BOM and approved substitutes; fabrication/assembly files; build environment; serial/configuration identity; deviations; raw captures; calibration and instrument references; test protocol; uncertainty and exclusions; analysis/report version; license manifest; repair procedure; and post-repair test.

## 8. Claim ceilings and fail-closed rules

- Schema validity is not design correctness.
- Design or simulation is not physical measurement.
- A job-completed event is not an as-built conformity result.
- One conforming unit is not a repeatable manufacturing process.
- One successful test is not reliability.
- Acoustic response is not human preference.
- A successful repair attempt without post-repair testing is not restored capability.
- Source availability is not permission to redistribute.
- Unknown, stale, incomplete or contradictory evidence remains unresolved or blocks the claim.

## 9. Exit criteria

This draft is ready for implementation review when the schema and synthetic corpus agree; all case IDs are unique; every negative-control ID has a documented reason/disposition; no case claims physical or perceptual qualification; and the docs clearly distinguish proposed targets from evidence. A follow-on implementation should independently validate the manifest, exercise every negative control, and report exact input digests. No CI pass, Rust build, physical run or human-study result is claimed by this documentation-only draft.
