# MEL-INST-001: Instrument Vocabulary and Proficiency Benchmark v1

**Status:** Proposed acceptance contract for staged implementation. Targets below are goals, not measured outcomes.  
**Priority:** Instrument capability before further style-catalog expansion.  
**Implementation wave:** Recorder, Ocarina, Harmonica, and đàn tranh procedural profiles, plus explicit source identity and renderer-state boundaries.  
**Repository:** Public `Luminous-Dynamics/symthaea` / `symthaea-muse`.

## 1. Why vocabulary is not proficiency

A selectable instrument name does not prove that Melothaea can perform it. Instrument capability has several independent dimensions:

1. **Identity:** which instrument or source-native model is requested, with alias and provenance.
2. **Procedural model:** which synthesis parameters and excitation model are available.
3. **Source assets:** whether a named sample bank or external preset is declared and actually present.
4. **Playable coverage:** which pitches/registers, dynamics, articulations and tuning contexts are represented.
5. **Performance vocabulary:** which instrument-specific gestures the event model can express.
6. **Realization evidence:** whether a requested event rendered, produced valid signal, and stayed inside resource/output bounds.
7. **Perceptual evidence:** whether musicians/listeners can recognize a useful timbre or gesture in blind, level-matched evaluation.

These dimensions must not collapse into one `supported=true` flag. A source-tree directory is not a local asset; a decoded WAV is not a correctly mapped pitch; non-silent output is not an authentic timbre; a GM program number is an interoperability fallback, not source identity.

## 2. Wave 3: four useful vocabulary additions

The first increment adds four instrument variants to the existing vocabulary of 42, bringing the enum/inventory target to 46. It also gives the VCSL Baroque Alto Recorder mapping to **Recorder**, not **Ney**. That distinction matters: an available recorder recording cannot be presented as a faithful ney recording.

| Instrument | First-wave representation | Explicitly not yet claimed |
|---|---|---|
| Recorder | Derived procedural spectrum/envelope/attack plus a declared VCSL Baroque Alto Recorder source path | Local sample availability, full register coverage, accurate tonguing/ornaments or human-validated timbre |
| Ocarina | Derived procedural profile and GM Ocarina compatibility program | Dedicated sample-bank/pitch coverage, breath-to-pitch calibration, articulation fidelity |
| Harmonica | Derived free-reed procedural profile and GM Harmonica compatibility program | Blow/draw reed-pair semantics, bends/overblows, hand-cup/wah, chord blocks or source-backed realism |
| Đàn tranh (Dan Tranh) | Derived plucked profile, Karplus–Strong route and source-native identity in the related catalogue | String-specific bends/tuning, tremolo, instrument-specific plucking positions, mapped/decoded local bank |

These are vocabulary and model footholds, not a claim that each instrument is already proficiently performed. The three non-recorder instruments remain without asserted source sample mappings until the exact assets and naming/pitch conventions have been inspected. For all four new procedural tables, values are labelled as derived estimates and must not be cited as direct measurements.

## 3. Instrument capability contract

The follow-on capability record should be versioned and additive. It should distinguish fields equivalent to:

- stable `instrument_id`, canonical label, aliases and source-native names;
- family and pitched/unpitched event semantics;
- approximate or source-evidenced usable pitch range, including its evidence source;
- primary renderer type and optional sample/preset source;
- available articulations and expressive controls, each with evidence status;
- dynamic-layer/velocity behavior and polyphony limits;
- fallback policy, loss declarations, and source-license/provenance receipt;
- calibration status, benchmark version, renderer revision, and asset digest.

Do not add a single “proficiency level” inferred from enum membership. Capability records should allow unknown or not-yet-tested fields. Keep this sidecar compatible with the existing versioned instrument-inventory work in [issue #7416](https://github.com/Luminous-Dynamics/symthaea/issues/7416) rather than creating a competing identity registry.

## 4. Benchmark suite

### A. Registry and routing integrity

- Enumerate every supported variant from one explicit roster.
- Require successful name resolution, valid bounded synthesis parameters, renderer-family assignment, and either a truthful sample mapping or an explicit unmapped status.
- Ensure GM fallback IDs are unique for the instrument pairs Melothaea can place in the same ensemble, and visibly label approximate mappings.
- Unknown instruments and unsupported articulations must fail closed or produce a visible reason-coded fallback.

**Initial target:** 100% of declared instruments have a deterministic identity, renderer route/status, and explicit unsupported/fallback behavior.

### B. Acoustic-model robustness

For each procedural instrument, render a deterministic pitch × velocity × duration grid inside the declared operating range. Start with five pitches spanning low, middle and high register; three velocities (soft/medium/loud); and short/medium/sustained note durations where applicable. Use fixed seeds and a named sample rate.

Measure:
- finite output and no NaN/Inf;
- peak/RMS and near-silence status;
- clipping/headroom and DC offset;
- pitch tracking error and unwanted octave errors;
- onset delay, decay/sustain/release behavior;
- spectral centroid, rolloff, spectral flux and harmonic/noise profile;
- rerender determinism under pinned settings.

**Initial target:** 100% finite/profile-contract cases; ≥95% successful note coverage across each *declared* range; every uncovered cell reported rather than omitted from the denominator. Targets should be revised only with a versioned rationale and measured baseline.

### C. Instrument-specific gesture coverage

Evaluate these as explicit event-level capabilities, not genre presets:

- **Recorder:** tongued versus slurred attacks, breath phrasing, register transitions and ornament primitives.
- **Ocarina:** breath-pressure dynamics, phrase-boundary breath, stable pitch and deliberate pitch transitions.
- **Harmonica:** blow/draw selection, bend envelope, overblow/overdraw events, chord/double-stop behavior and hand-cup modulation.
- **Đàn tranh:** pluck position/strength, repeated-note variation, tremolo, controlled string bends and tuning/scale profiles.

The first code increment does **not** implement all these techniques. Start with identity and basic render probes, then add a typed gesture model only where event semantics and an appropriate renderer can represent the behavior. Do not fake a bend by changing instrument identity or relabel a stationary spectrum as a fully idiomatic performance.

**Initial target:** every marketed gesture has a positive control that audibly/structurally changes the intended feature and a negative control that proves the test can fail. Untested gestures remain labelled unavailable.

### D. Source-backed sample and mapping qualification

For each sample bank:

1. pin the upstream source revision, license/provenance, relative asset paths and manifest/content digests;
2. confirm the asset root is actually present locally;
3. decode all indexed WAVs and report corrupt, silent, duplicate or rejected files;
4. verify pitch tokens and ranges against known note/pitch references, not filenames alone;
5. measure coverage by pitch, dynamic layer, round-robin and articulation;
6. run round-trip and render probes against the selected notes and velocities;
7. record exact sample/renderer decisions in each note/part receipt.

VCSL states its assets are CC0; VSCO-2-CE also documents CC0 use and asks users for courtesy attribution. Pin and preserve the applicable source terms and revision rather than assuming all sample sources share a license. References: [VCSL](https://github.com/sgossner/VCSL), [VSCO-2-CE](https://github.com/sgossner/VSCO-2-CE).

**Initial target:** 100% of selected parts have truthful provenance/render-path receipts; zero silent source substitution; represented import/export assignments and timed changes round-trip exactly on supported paths.

### E. Blind timbre/gesture evaluation

Create equal-loudness, level-matched, randomized A/B packs. Compare the candidate instrument model against a named baseline and, only when properly licensed, suitable reference recordings. Split tests by instrument, register, velocity and gesture. Include both instrument-identification trials and naturalness/preference ratings; the two answer different questions.

Report listener counts, uncertainty, per-class confusion, source/baseline and failed trials. Do not promote a sample-decode or signal-health pass into an audio-quality claim. Where tradition-specific playing practices are involved, involve appropriately knowledgeable practitioners and separate that review from technical pass criteria.

**Proposed stretch goal:** for a named task and instrument family, >60% pairwise preference against the strongest matched baseline with a 95% confidence-interval lower bound above 50%, then replicate on held-out prompts/listeners. Do not use a pooled score to hide weak instruments.

## 5. Next implementation sequence

### P0 — This wave: vocabulary and no-fabrication boundary
- Add four explicit identities and derived procedural profiles.
- Keep source paths distinct from renderer/sample claims.
- Route the actual VCSL recorder bank to Recorder instead of treating it as Ney.
- Add roster and parameter-integrity regressions.

### P1 — Instrument capability matrix (implemented as a new local tool)
Run:

```sh
cargo run -p symthaea-muse --bin muse_instrument_qualification -- > mel-instrument-capability-matrix.json
```

- The tool consumes the existing VCSL / VSCO-2-CE inventory and emits 46 explicit per-instrument rows.
- It distinguishes no source mapping, missing directories, no pitch-indexed samples, decode failures, silence-only decoded assets, partial sample evidence and indexed non-silent source samples.
- It preserves filename-observed MIDI bounds and internal pitch holes but labels the usable/playable range as **not established**. These are source-file observations, not measured instrument capabilities.
- Four instrument-specific gesture queues (Recorder, Ocarina, Harmonica, DanTranh) are emitted as `planned_unvalidated`; this is a worklist, not an implemented capability claim.
- The inventory reads indexed WAVs for decode/silence evidence. `--hash-indexed-samples` additionally hashes the contents of every accepted WAV and may be I/O intensive.
- Missing roots remain unavailable evidence, not empty-but-supported coverage. No instrument is marked qualified by the source inventory alone.

### P2 — Per-instrument render receipts and probe grid
- First increment now exists as `src/bin/muse_instrument_probe.rs` and is auto-discovered by Cargo as `muse_instrument_probe`.
- It probes all 46 registered instrument variants on a default A3/A4/A5 × 3-velocity smoke grid, with two exact rerenders by default.
- For range-focused qualification, `--midi-range MIN:MAX` builds an inclusive semitone grid. For example, MIDI 60:84 probes 25 notes across three velocities for the selected instrument. JSON records both MIDI note and exact frequency for every case.
- Work is bounded to 2,000 render attempts per invocation to keep local sweeps responsive; wide sweeps should target one instrument and use source-inventory bounds as an investigation hint, not as proof of the playable range.
- Per-case JSON reports peak, RMS, DC mean, non-finite count, near-silence classification, clipped sample count, SHA-256 of the actual rendered float samples, whether repeated hashes match, and an exploratory YIN-periodicity pitch estimate with signed cents error and confidence when measurable.
- The probe aggregates RMS across the 0.2 / 0.5 / 0.9 velocity cells for each instrument/pitch, including adjacent level changes in dB and an RMS-monotonicity flag. This is a diagnostic—not a universal requirement that all sample layers or percussive timbres be strictly monotonic.
- The new `render_arrangement_with_evidence` API attaches the selected per-note renderer path and sample-selection outcome. The probe checks that receipts match the requested note controls and remain stable across repeated renders.
- Receipt paths distinguish named/default sample rendering, Karplus–Strong, FM, ordinary additive synthesis, and the legacy mood-derived path when no instrument identity is assigned. Sample fallback status differentiates a missing library from an unavailable bank, pitch/velocity selection, or short articulation.
- When an actual WAV is selected, the receipt includes source library, exact source-root-relative path, SHA-256 over the exact source WAV bytes, recorded root MIDI pitch, sample rate, filename-derived dynamic layer, and applied pitch shift in semitones. The absolute machine-specific asset root is excluded.
- The default pitch grid is a **signal-health smoke grid, not a valid range claim for every instrument**. A user-selected MIDI sweep measures the requested render cells, not whether the entire range is musically valid. Source filename bounds may guide the sweep but cannot certify playability.
- A selected sample path means the renderer picked and rendered a compatible sample; it does not prove pitch fidelity, full source-bank coverage, authentic technique, or perceptual quality.
- The pitch estimator searches a target-relative band (approximately half-to-double the requested pitch), capped below the sample-rate Nyquist region and by available window length. Out-of-band requests return an explicit unavailable status; low-periodicity or inharmonic tones must not be assigned a false pass or treated as a calibrated pitch failure. No pitch metric changes the current render outcome gate.
- Schema revision is `melothaea.instrument-probe.v4`. Run-level provenance records the running probe executable's SHA-256, byte length, and basename (never its absolute path), plus the source revision and the source of that revision value. The pitch calibration report uses `melothaea.pitch-estimator-calibration.v2` and carries the same run-level identity fields.
- The capability matrix now uses schema `melothaea.instrument-capability-matrix.v3` and emits each additional sample style's identity and bank evidence, including its indexed-content digest when hashing is enabled. This makes the matrix usable for bank-path cross-checks instead of only summarizing style counts.
- The binary digest is computed by streaming the executable bytes in bounded memory. Hash failures are explicit (`hash_failed` or `unavailable`) and do not get disguised as valid digests. Source revision values supplied through environment variables are caller-supplied metadata, not authenticated proof of the source tree used to build that binary.
- The MIDI-sweep feature creates per-note diagnostic coverage but does not yet enforce source-derived ranges or per-family calibration.
- The tool makes no network calls and performs no heavyweight asset download by default.

Example commands:

```sh
# Fast probe of one instrument (nine pitch/velocity cases, two render repetitions each).
cargo run -p symthaea-muse --bin muse_instrument_probe -- --instrument recorder

# Full registered-variant grid; redirect the machine-readable report.
cargo run -p symthaea-muse --bin muse_instrument_probe -- --instrument all > mel-instrument-probe.json

# One instrument over 25 semitones, with per-cell source and renderer receipts.
cargo run -p symthaea-muse --bin muse_instrument_probe -- --instrument recorder --midi-range 60:84 > recorder-semitone-sweep.json

# Calibrate the pitch estimator against known-frequency synthetic tones.
# Default grid: MIDI 36:96, 22050 Hz, 100 ms; no instrument rendering occurs.
cargo run -p symthaea-muse --bin muse_instrument_probe -- --calibrate-pitch-estimator > mel-pitch-calibration.json

# Narrow calibration when investigating one register.
cargo run -p symthaea-muse --bin muse_instrument_probe -- --calibrate-pitch-estimator --midi-range 48:84 > mel-pitch-calibration-48-84.json

# Keep runtime lower while inspecting signal health.
cargo run -p symthaea-muse --bin muse_instrument_probe -- --instrument harmonica --repeats 1

# Optional, I/O-intensive: put the selected bank's indexed-content digest
# in every selected-sample receipt. Must be set before the process starts.
SYMTHAEA_MUSE_HASH_SOURCE_BANK_MANIFEST=1 cargo run -p symthaea-muse --bin muse_instrument_probe -- --instrument harp > harp-probe-with-bank-manifest.json

# Generate indexed-bank and whole-root content digests for comparison.
# Root hashing streams every .wav found under both roots, including files
# rejected by the pitch indexer; enable only when the I/O cost is acceptable.
SYMTHAEA_SOURCE_REV="$(git rev-parse HEAD)" cargo run -p symthaea-muse --bin muse_instrument_qualification -- --hash-indexed-samples --hash-root-wav-contents > mel-instrument-capability-matrix.json
```

Use `cargo test -p symthaea-muse --bin muse_instrument_probe` for roster/bounds checks, synthetic sine pitch-estimator sanity tests, silence rejection and velocity-response summary tests.
Use `cargo test -p symthaea-muse --bin muse_instrument_qualification` for fail-closed source-status classification and the explicit unvalidated-gesture worklist.
These tests and commands are authored/documented but are not marked as executed by these changes.

The probe can optionally add the indexed-content digest for the selected mapped bank beside the selected-WAV byte hash. Compare `selected_sample.source_bank_content_manifest_sha256` in the render report against `indexed_sample_content_manifest_sha256` for the corresponding bank in the qualification matrix, using the same asset roots and code/indexer revision. The inventory can also emit a complete root-content digest via `--hash-root-wav-contents`; this streams all lowercase `.wav` files found by the root walker, including files rejected by the pitch indexer. Each root report distinguishes `not_requested`, `root_missing`, and `hashed`. The root digest uses a versioned, deterministic record encoding over sorted root-relative paths, sizes, and exact file-byte hashes. The probe now adds run-level executable identity so a result can be tied to the exact on-disk probe binary observed at run time.

For an explicit revision field alongside the compile-time `code_revision`, set the runtime revision before invoking the probe:

```sh
SYMTHAEA_SOURCE_REV="$(git rev-parse HEAD)" cargo run -p symthaea-muse --bin muse_instrument_probe -- --instrument recorder > recorder-probe.json
SYMTHAEA_SOURCE_REV="$(git rev-parse HEAD)" cargo run -p symthaea-muse --bin muse_instrument_probe -- --calibrate-pitch-estimator > mel-pitch-calibration.json
```

The report records `source_revision_status` as `runtime_symthaea_source_rev`, `runtime_github_sha`, `compile_time_embedded`, or `unavailable`. Values supplied at runtime are assertions by the caller; they are not verified against the executable's source tree and are not signed. The binary identity records the basename, SHA-256, byte length, and a state of `hashed`, `hash_failed`, or `unavailable`. The absolute executable path is deliberately omitted to avoid leaking local usernames/workspace paths. Hashing the process executable does not yet capture dynamically linked libraries, the OS/runtime, environment variables, asset licensing, or source authenticity.

These hashes and receipts are integrity evidence rather than a signature or authenticated attestation: they say what bytes were observed, not who supplied them or whether the source was trustworthy.

### P2b — Cross-report provenance verifier

The new `muse_instrument_provenance_verify` CLI cross-checks the run-level executable digest against a separately supplied binary file. It now also requires both concrete asset roots, independently recomputes each root-content manifest using the inventory's sorted path/size/byte-hash encoding, and rehashes every selected source WAV referenced by the probe report. Each selected sample path must be a normalized relative path discovered by the same WAV-root walker, remain under its canonical supplied root, and refer to a regular lowercase `.wav` file. Absolute, parent-traversal, and symlink paths excluded by the root walker are rejected. It compares source revision and bank manifest claims against the matching instrument/library/relative-directory entry in the capability matrix. It streams and hashes the exact JSON bytes consumed by the parser so a report digest cannot silently refer to a different on-disk version. The matrix schema v3 includes detailed additional-style bank evidence so those selected banks can be checked too. Executable and root arguments are not written into the report; only observed digests, counts, and findings are emitted.

Generate both inputs from a clean source checkout and preserve the outputs outside the checkout, so report generation itself does not make the recorded source revision ambiguous:

```sh
test -z "$(git status --porcelain)" || { echo "Use a clean checkout for provenance runs." >&2; exit 2; }
REV="$(git rev-parse HEAD)"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/melothaea-provenance.XXXXXX")"
SYMTHAEA_SOURCE_REV="$REV" cargo run -p symthaea-muse --bin muse_instrument_qualification -- --hash-indexed-samples --hash-root-wav-contents > "$OUT/mel-instrument-capability-matrix.json"
SYMTHAEA_SOURCE_REV="$REV" SYMTHAEA_MUSE_HASH_SOURCE_BANK_MANIFEST=1 cargo run -p symthaea-muse --bin muse_instrument_probe -- --instrument recorder > "$OUT/recorder-probe.json"
VCSL_ROOT="${SYMTHAEA_VCSL_DIR:-data/samples/vcsl}"
VSCO2_ROOT="${SYMTHAEA_VSCO2_DIR:-data/samples/vsco2-ce}"
cargo run -p symthaea-muse --bin muse_instrument_provenance_verify -- --probe "$OUT/recorder-probe.json" --inventory "$OUT/mel-instrument-capability-matrix.json" --renderer-binary target/debug/muse_instrument_probe --vcsl-root "$VCSL_ROOT" --vsco2-root "$VSCO2_ROOT" > "$OUT/provenance-verification.json"
printf 'Evidence reports retained in %s\n' "$OUT"
```

Before trusting a local result, run the verifier's regressions (including same-size byte-tamper and parent-path negative controls):

```sh
cargo test -p symthaea-muse --bin muse_instrument_provenance_verify
```

This command is documented but was not executed by this change.

The verifier streams SHA-256 over both input-report files, the supplied renderer executable, every discovered lowercase `.wav` file in both asset roots, and each selected source WAV. It compares the recomputed root manifest digest and WAV file count with the inventory, and checks selected-file byte digests against the per-note receipts. Adjust `--renderer-binary` to the exact executable used for the probe if Cargo uses a non-default target directory, and ensure the two root arguments point to the same directories used to generate the inventory. Root hashing scans the full WAV trees, so this verification is I/O-intensive for large sample libraries. The JSON report includes input/executable/root digests, verified counts, and mismatch findings. Exit code 0 means `consistent`; exit 1 means provenance is incomplete or inconsistent; exit 2 means the command or an input report could not be processed. An absent asset root, disabled hashing, schema mismatch, missing selected-bank mapping, or malformed digest cannot silently pass. The verifier's result proves only report consistency: it does not authenticate the claimed source revision, verify the build that produced the executable, validate licensing, or qualify audio quality. Run it for each probe report and keep the two source reports plus verification JSON together.

Then add instrument-specific range, pitch-accuracy, and gesture tests. Keep a per-note evidence record linking requested instrument, selected renderer, selected source sample, signal checks, and exact asset/renderer digests. Do not collapse these fields into a pooled “instrument score.”

### P2a — Calibrate the measurement instrument first
- Run MEL-PITCH-CAL-001 on the default synthetic-tone sweep (MIDI 36–96) before interpreting instrument pitch errors.
- Track numeric-estimate coverage and periodic-estimate coverage separately, plus median/p95/maximum absolute error and counts above 2, 5, and 15 cents.
- Treat this as a check of the estimator against known pure sine waves, not proof of instrument tuning or perceptual quality.
- Expand the calibration set later with harmonic-rich tones, controlled noise, vibrato, and known inharmonic stimuli; each stimulus class needs explicit limits and failure statuses.

### P3 — First-class gesture events
- Add versioned typed events for a small subset of distinctive gestures and prove target sensitivity plus non-target invariance.
- Start with the highest-value cases where the selected sample assets or renderer can actually realize them.
- Unsupported gestures remain explicit, instead of being approximated invisibly.

### P4 — Blind audition and musician review
- Produce reproducible instrument-specific listening packs from pinned outputs.
- Collect independent ratings and report confidence intervals and confusion matrices.
- Only then label instruments as perceptually reviewed.

### P5 — Broaden the vocabulary by measured capability gaps
Prioritize the next instrument based on a weighted combination of listener demand, source availability, ensemble utility, missing expressive capability and engineering cost. Candidate source families already identified for later evaluation include recorder variants, ocarina, harmonicas, Dan Tranh, percussion drum-hit/roll lanes, and additional culturally distinct plucked/wind families. Source catalogue entries alone do not count as playable instruments.

## 6. Evidence boundary for this increment

This plan is designed to let source work and local evaluation proceed without waiting for unrelated style PRs, while preserving honest qualification. The four procedural profiles and regression code can be inspected statically; compile, formatting, unit-test, local-asset decode, synthetic calibration, instrument-grid render, and blind-listening outcomes require execution. A queued or skipped runner never passes a gate. The new calibration mode exercises only clean, generated sine waves; it does not certify any instrument renderer or source bank.

## 7. Research anchors for evaluation design

- [MIREX 2026 Music Evaluation via CMI-RewardBench](https://www.music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench) explicitly focuses on the gap between automatic acoustic metrics and listener preference. In the [published 2026 results](https://www.music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results), leading preference-prediction systems scored in the low-to-mid 70% range on the core pairwise task. That is a *judge-versus-human agreement metric*, not a target score for the generated audio itself.
- The [YIN fundamental-frequency method](https://pubmed.ncbi.nlm.nih.gov/12002874/) provides a computationally efficient F0 diagnostic. Melothaea's first implementation is deliberately labelled exploratory, with confidence/status visible; it still needs instrument-family calibration and independent tests against known reference tones and acoustically inharmonic instruments.
- [MIREX 2026 Symbolic Music Generation](https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation) uses a double-blind subjective evaluation across prompt coherence, creativity, structuredness and musicality; its [audio-generation evaluation protocol](https://www.music-ir.org/mirex/wiki/2026%3AMusic_Audio_Generation) separately scores melody, harmony, key consistency, style consistency, and arrangement/production. Melothaea should follow the same separation rather than masking a broken instrument inside a high overall music score.
- [ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I) is the in-force recommendation for programme loudness and true-peak measurement. The initial probe only measures raw sample peak/RMS/DC and therefore must not label these as LUFS or true-peak compliance.
- [SLSA Provenance](https://slsa.dev/spec/v1.2/provenance) models evidence about how an artifact was produced and which inputs it consumed; [in-toto Statements](https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md) bind metadata to subjects by digest. Melothaea's current root manifest is an integrity digest, not yet a signed SLSA/in-toto attestation or a complete renderer-and-environment provenance record.



## 8. Provisional performance gates (not yet achieved)

These are engineering targets to adopt after the v3/v4 measurement reports are locally run and calibrated; they are not results.

| Dimension | Proposed target | Qualification note |
|---|---:|---|
| Identity / receipt fidelity | 100% of probe cells have correct per-note receipts; zero missing or mismatched instrument/control fields | Hard deterministic gate |
| Deterministic rendering | 100% exact PCM hashes match for deterministic paths under pinned environment/assets | Stochastic paths must declare seed/policy rather than be silently excluded |
| Signal health | Zero non-finite samples; zero unexplained near-silent cells in declared playable range | Unsupported cells must be explicit, not counted as passes |
| Periodic-tone pitch accuracy | Synthetic sine fixtures: median absolute error ≤ 2 cents; p95 ≤ 5 cents | Run MEL-PITCH-CAL-001; report coverage, median, p95, maximum, and counts beyond 2/5/15 cents before interpreting instrument pitch results |
| Instrument pitch accuracy | For calibrated, pitched instruments: median absolute error ≤ 5 cents and p95 ≤ 15 cents over supported range | Exclude or separately model inharmonic/percussive timbres and report low-confidence cells |
| Velocity controllability | ≥95% of declared velocity-sensitive cells show expected loudness direction; exceptions explicitly documented | Treat this as a target only after per-instrument baseline and within-velocity repeatability are measured |
| Source coverage | 100% of sample-backed cells trace to a relative source path, source WAV byte hash, source root pitch, and source rate; with bank hashing enabled every sample-backed cell carries a bank digest; root content digest captures all discovered .wav assets when enabled | Digests must match for identical assets, roots, and indexer revision; digest ≠ signature, source authenticity, licensing clearance, or renderer identity |
| Provenance consistency | 100% of selected sample-bank digests match the inventory bank at the same instrument/library/relative path; source revision matches; renderer and existing-root manifests have valid digests | The verifier must return exit 0 only for complete, internally consistent evidence; incomplete or conflicting reports return nonzero and do not imply audio quality |
| Asset-byte verification | 100% of selected source-file SHA-256 values match the supplied actual WAV bytes; all selected files are members of the recomputed root manifest; zero unsafe or unmanifested paths accepted | Negative controls mutate bytes without updating receipts and attempt parent-path traversal; each must fail closed |
| Gesture semantics | 100% of marketed gestures have positive controls, negative controls, and renderer receipts | Untested gestures remain unavailable |
| Human preference | Paired preference accuracy reported with confidence intervals; target ≥60% preference for improved vs matched baseline with lower 95% CI >50% | Separate instrumental identity, musicality, prompt alignment and production quality; blind review is required |

**Do not collapse the table into one weighted score.** Report coverage per instrument, renderer/source route, pitch region, velocity region, and gesture. An aggregate score can conceal one instrument that is completely broken.

**Definition of progress:** fewer unsupported musical requests, more reliably playable pitches/gestures, less undocumented fallback, and stronger audible distinction—not simply a larger instrument count. Style diversity expansion should resume after the instrument inventory, render grid, and first instrument-specific listening pack identify concrete gaps.
