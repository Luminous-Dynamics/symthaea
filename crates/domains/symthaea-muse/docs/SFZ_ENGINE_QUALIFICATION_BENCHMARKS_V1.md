# SFZ Engine Qualification Benchmarks v1

**Status:** proposed engineering targets; not yet a conformance standard.  
**Applies to:** Muse SFZ fixture generation, parser/sequencer checks, offline render comparison, and evidence-bundle verification.

## 1. Paradigm shift: replace the universal score with a claim-specific evidence graph

An SFZ engine should not receive one global "compatible" badge because it passed a few examples or produced a similar-looking waveform. SFZ behavior is a matrix of features, engine versions, build options, host conditions, and fixture conditions. A result must name the behavior it supports and the exact engine build against which it was measured.

Represent one result as:

`(claim, fixture, SFZ revision, engine/build hash, config, MIDI hash, sample hashes, capture method, metrics, verifier receipt)`

The graph links source assets to a deterministic fixture, an exact engine build, a capture, the comparison receipt, and the integrity verifier's output. A hash proves identity of bytes, not authorship, correct rendering, SFZ conformance, or perceptual quality. Keep those claims separate.

**Four qualification layers must remain independent:**

1. **Evidence integrity:** Do the declared inputs, executable/build config, captures, and receipts link together without tampering?
2. **Semantic behavior:** Did the engine parse the intended regions and choose the expected sample/round-robin/velocity/key-switch behavior?
3. **Signal behavior and performance:** Does a captured output meet a *predeclared, fixture-specific* numerical contract, and can the engine sustain the target load?
4. **Perceptual quality:** Can a controlled listening study detect an impairment, with uncertainty and test conditions reported?

Passing a lower layer never implies passing a higher one.

## 2. Integrity and adversarial mutation gates

### Required checks

- Recompute the fixture-pack tree digest independently; do not trust the pack's own receipt alone.
- Bind each fixture ID to the exact SFZ path/hash/length and each sample asset to its exact path/hash/length.
- Bind candidate/reference WAV hashes to the differential receipt; reject incomplete, invalid, empty, or non-comparable measurements.
- Reject absolute/parent paths, symlinks, non-regular files, oversized inputs, schema mismatches, duplicate asset paths, and output overwrite attempts.
- Preserve an explicit negative result when any link is missing; never convert unknown or not-run into pass.

### Targets

| Measure | Initial target | Qualification rule |
|---|---:|---|
| Known unsafe-path and symlink cases rejected | 100% | Every declared negative case must fail closed |
| Hash/length/path/schema mutations detected | 100% of seeded mutations | A clean baseline must also be accepted |
| False accepts in a 1,000-case seeded mutation campaign | 0 | Report mutation categories and seed; expand corpus after any escape |
| Repeat verification of unchanged bundle | Identical decision and stable manifest/pack digests | Verification receipt may differ only in explicitly variable metadata |
| Quality promotion by forged fields | 0 accepted | Input flags never override verifier policy |

The 1,000-case goal is a scaling target, not a claim that the current unit tests already cover 1,000 cases. Start with a small deterministic mutation corpus in CI, then grow it. Include single mutations and combinations: swap an asset, change one byte, replace a receipt, change a fixture ID, corrupt a comparison field, truncate a WAV, use a directory instead of a file, and redirect a path through a symlink. Keep each mutation's expected outcome explicit.

## 3. Semantic conformance: qualify features, not file extensions

A parser accepting an `.sfz` file is not proof that its instrument behaves correctly. Keep a versioned feature matrix, with a fixture and negative control for every behavior claimed.

Minimum fixture families:

- key/velocity zones and boundary conditions;
- sample path resolution, missing samples, and malformed paths;
- round-robin and sequence state (including repeat/restart behavior);
- envelopes, release tails, sustain loops, one-shot samples and choke groups;
- pitch tracking, tune offsets, key switches, MIDI CCs and velocity curves;
- stereo/multichannel sources, sample-rate conversion and unsupported opcodes.

**Targets:** 100% of expected parse/region-selection/sequencer outcomes for the explicitly declared supported subset; 100% of unsupported or ambiguous cases labeled accurately; zero silent fallback to a different sample for a declared supported fixture. Publish support as an engine/version-specific matrix rather than claiming universal SFZ compatibility.

When two engines disagree, do not automatically call either one the oracle. Inspect the written SFZ specification, current regression tests, fixture provenance, and exact build versions; classify the outcome as spec-defined, implementation-specific, under-specified, or a suspected engine defect.

## 4. Signal and numerical targets

All limits in this section are **proposed research targets**. Do not turn them into merge gates until the fixtures have analytical ground truth, the metric is defined precisely, and the threshold is validated on a calibration set distinct from the held-out qualification set.

| Fixture / metric | Initial target | Notes |
|---|---:|---|
| Identical WAV input | Identical SHA-256 | Integrity control; not a perceptual metric |
| Same engine/build/host/config deterministic rerender | Identical WAV hash for fixtures declared deterministic | Record any deliberate nondeterminism and its seed |
| Known pitch-ratio transform | Relative ratio error ≤ (10^{-6}) | Test multiple notes, sample rates and tuning offsets |
| Declared pitch tracking / stretch behavior | Absolute error ≤ 0.1 cents | Only when the fixture has an unambiguous reference pitch |
| Filter passband | Ripple ≤ 0.1 dB; relative amplitude error ≤ 0.04 | Predeclare frequency bands and edge treatment |
| Aliasing test with analytically specified alias band | Alias energy ≤ −70 dBc for the specified fixture, or ≥ 26 dB reduction against the declared non-oversampled baseline where applicable | Report both absolute result and baseline; don't generalize to all patches |
| Envelope, loop and release fixtures | No missing/extra events; release/loop timing within a predeclared frame tolerance | Use event-level checks plus waveform inspection |
| Comparator stability | Stable metrics for identical bytes, bounded by documented floating-point tolerance | Pin FFT parameters and comparator version |

Avoid one composite weighted score early on. It can conceal a catastrophic failure (wrong round-robin sample, clipped tail, or large pitch error) behind several good metrics. Report a vector of metrics and a per-fixture pass/fail/unknown state.

## 5. Runtime and resource benchmarks

Run performance tests on a named host profile and publish CPU model, compiler flags, OS, audio backend, sample rate, channel count, polyphony, sample-bank size, warm/cold state, and test duration. Separate offline rendering speed from realtime deadline behavior.

| Measure | Initial target |
|---|---:|
| Offline renderer throughput | ≥ 10× real time on a named reference host and declared fixture mix |
| Realtime 128-frame callback at 44.1 kHz | p99 callback duration ≤ 2.3 ms against a 2.90 ms block deadline |
| Sustained realtime test | 10 minutes with zero missed deadlines at declared polyphony |
| Resource reporting | p50/p95/p99 CPU time, peak RSS, allocation count if instrumented, startup/load latency, and deadline misses |
| Scaling curve | Sweep polyphony and active sample/voice count; publish the first failure knee, not only the maximum success point |

These are target thresholds for controlled reference hardware, not universal hardware requirements. Never compare runner-wall-clock numbers across unlike hosts or infer realtime capability from an offline benchmark.

## 6. Perceptual quality: calibrate objective metrics against blinded listening

Waveform RMSE, spectral distance, alias energy and correlation are useful diagnostics. They are not substitutes for listener judgments: timing, envelope, phase, stereo image and timbral differences can affect the metrics differently, and small numerical error does not guarantee perceptual equivalence.

Use ITU-R BS.1116-3 as the starting methodology for controlled subjective assessment of small impairments. Pre-register the hypothesis, reference, impairments, playback conditions, listener inclusion rules, statistical analysis, stopping criterion, and confidence interval. Choose sample size from a pilot variance estimate and desired resolution; do not assume a fixed panel size is universally adequate.

Keep a calibration set for exploring metric/threshold behavior and a held-out set for qualification. Do not tune thresholds using the held-out outcomes. Report listener-level and aggregate uncertainty; avoid claims of universal perceptual quality from a single group or setup.

## 7. Multi-engine compatibility and reference policy

The public [SFZ regression test collection](https://github.com/sfz/tests) is one useful input, not a complete universal oracle. Freeze its exact revision and retain fixture provenance. Add minimal synthetic fixtures with known ground truth and explicit negative controls.

Use independently developed engines as separate compatibility axes. [LiquidSFZ](https://github.com/swesterfeld/liquidsfz) is a candidate second implementation; pin the precise source revision, build flags and executable hash when adopted. Do not rely on an engine name or version string as authenticated provenance. Earlier sfizz releases can be retained as a frozen historical reference, but an archived engine must not be treated as a maintained, authoritative oracle.

For each behavior, publish:
- supported by this build;
- measured and passed against named fixtures;
- unsupported;
- ambiguous / engine-specific;
- not measured.

A "cross-engine common profile" should require agreement by at least two independently built engines on the *semantic outcomes* for the claimed feature set. It does not require byte-identical audio across different DSP implementations.

## 8. CI and governance policy

- Pin all fixtures, engines, reference assets, comparator settings, and benchmark profiles by content hash/revision.
- Keep deliberate negative controls permanently in the suite; don't remove a failing control to improve the pass rate.
- Separate calibration from held-out qualification data.
- Every number has a unit, definition, provenance, and uncertainty or tolerance.
- A queued, skipped, stale, unavailable or missing runner is **unknown**, never pass.
- Record exact head SHA and check conclusion alongside results.
- Requalify when a renderer, compiler, FFT implementation, fixture pack, sample asset, or metric implementation changes.

## References

- SFZ regression fixtures: https://github.com/sfz/tests
- SFZ format overview and feature model: https://sfzlab.github.io/sfz-website/documentation/getting-started/what_is_sfz/
- LiquidSFZ repository / supported opcodes / release history: https://github.com/swesterfeld/liquidsfz
- ITU-R BS.1116-3, *Methods for the subjective assessment of small impairments in audio systems* (status shown as in force): https://www.itu.int/rec/R-REC-BS.1116-3-201502-I/en
