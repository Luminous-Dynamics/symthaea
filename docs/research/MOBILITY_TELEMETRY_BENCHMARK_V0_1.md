# Mobility telemetry benchmark contract — v0.1

**Status:** proposed benchmark contract; thresholds below are engineering targets, not measured results or universal standards.  
**Owner:** [TYRE-DATA-001 #8076](https://github.com/Luminous-Dynamics/symthaea/issues/8076), composed with [TRANSPORT-ROAD-001 #6247](https://github.com/Luminous-Dynamics/symthaea/issues/6247).  
**Implementation baseline:** [telemetry manifest and raw-trace validator](MOBILITY_TELEMETRY_REFERENCE_ARCHITECTURE_2026-10-19.md).

## Paradigm shift

Do not optimize for “more telemetry.” Optimize for the smallest reproducible evidence package that can support a bounded engineering claim.

A telemetry record is not just time plus value. It is an observation with:
- a declared physical quantity, unit, coordinate frame, sign convention and acquisition configuration;
- sensor identity, calibration state, timestamp source and uncertainty;
- explicit validity and loss state, with no silent repair;
- provenance for raw bytes, decoder/filter transforms, dataset rights and protocol;
- a claim-specific evidence ceiling.

Keep raw capture immutable. Every filter, synchronization correction, decoded bus signal, resampling step and derived quantity is a new versioned artifact pointing back to the raw evidence. Missing is not zero; receive time is not necessarily sample time; decoded CAN is not a substitute for raw frames; and a valid file is not proof of physical accuracy.

Use an interoperable exporter rather than making our schema a format island: [ASAM MDF](https://www.asam.net/standards/detail/mdf/) is a suitable future interchange target. Linux [SocketCAN timestamp facilities](https://docs.rs/socketcan/latest/socketcan/timestamp/index.html) are a candidate optional adapter. Neither dependency should be required to validate a plain session manifest and trace.

## Benchmark ladder

| Level | Benchmark | Pass condition | What a pass does not establish |
|---|---|---|---|
| B0 | Schema and semantics | Unique IDs, valid references, finite positive sample rates, known units, explicit clock domain and sign/frame conventions | Sensor accuracy |
| B1 | Byte and custody integrity | Declared bytes, lengths and SHA-256 match; raw artifacts remain inside the session root; rights/calibration receipts are explicit | Permission where none was granted; truthful sensor calibration |
| B2 | Adversarial fault injection | Zero false accepts across the frozen mutation matrix; invalid samples stay visible; no implicit interpolation, zero-fill or row deletion | Hardware reliability |
| B3 | Determinism and replay | Same raw bytes + manifest + decoder versions yield the same canonical report digest; toolchain/build identity is recorded | Correct physics |
| B4 | Bench capture and timing | Known injected drops reconcile exactly with counters; clock offset/jitter and sample loss meet the pre-registered test budget; power-loss recovery preserves committed records | Calibration traceability unless independently evidenced |
| B5 | Metrology | Each claim's uncertainty budget is measured using traceable reference instruments over the declared operating envelope | Vehicle/tire performance outside that envelope |
| B6 | Claim-specific validation | Entire runs, days, vehicles or tire specimens are held out; prediction error and uncertainty are reported per unit/run, including outliers | Regulatory approval or safe public-road use |

### Initial software acceptance targets

These targets apply only to the software/fixture suite and must be reported separately from hardware results.

1. **Fail-closed corpus:** zero false accepts across every frozen negative control in the machine-readable fault matrix. A change to expected outcomes requires a reviewed rationale, not deleting a failing case.
2. **No silent alteration:** zero implicit zero-fills, interpolations, outlier deletions or unit conversions in the raw layer. Any transformation is opt-in, versioned and emits its own artifact.
3. **Content identity:** SHA-256 and byte length match for 100% of declared raw artifacts before a recorded-session report can pass.
4. **Replay determinism:** 100/100 replays of the same fixture on the same build produce the same canonical report digest. Volatile fields must be excluded from the canonical digest, not ignored after a mismatch.
5. **Claim separation:** 100% of synthetic, planned, rights-incomplete, uncalibrated or channel-incomplete sessions remain ineligible for physical claims.
6. **Gap observability:** every injected dropped-sample pattern remains detectable from sequence/counters/timestamps or an explicit event. A gap diagnostic may flag a likely gap; it must never synthesize a sample.

These are proposed gate definitions, not claims that the current implementation has met them. The Rust validator has authored unit tests, but the tests still need execution on an equipped runner.

## Initial hardware targets to measure, not assume

Treat these as revision-A goals that can be changed only before a test campaign and with a recorded reason.

- **Capture completeness:** at least 99.99% of expected IMU/wheel/power samples during a one-hour controlled bench run at the declared target rates, with every loss accounted for by independent reference counts or an explicit logger-overflow/error counter. Report per channel and interval; an aggregate percentage must not hide a failed channel.
- **Electrical time alignment:** voltage/current pairwise alignment p99 no worse than 1 ms and maximum no worse than 2 ms during a calibrated 100 Hz reference campaign; record timestamp source and uncertainty. If the instrumentation cannot meet this budget, restrict the claim or upgrade acquisition hardware.
- **Energy accounting:** relative error no worse than 1% against a calibrated reference energy meter on pre-registered steady and stepped-load profiles, including a declared integration method and boundary. Also publish the combined uncertainty; meeting a percentage target does not erase uncertainty.
- **Recovery:** after controlled power cuts, all reported committed records remain byte-identical, the final session is marked incomplete unless cleanly closed, and any lost tail is bounded/reported—not silently treated as a valid close.
- **Clock discipline:** test offset, drift, jitter and reboot behavior over the full session duration. GNSS UTC/PPS is an anchor, not an excuse to replace the monotonic clock or omit measured synchronization uncertainty.
- **Thermal/environmental coverage:** validate at specified temperatures, supply conditions, vibration and storage fill states before advertising those conditions as supported.

Do not select a generic force, grip or temperature accuracy threshold without specifying the sensor chain, calibration reference, range, loading geometry, uncertainty model and applicable test method. Freeze those per measurement programme.

## Clock integrity as an evidence gate

A physical claim is blocked unless the manifest uses one of the exact v0.1 synchronization methods (`gnss_pps_verified`, `hardware_timestamp_verified`, `ptp_verified`, or `shared_clock_verified`), declares a content-addressed `clock_sync_record` artifact, and states maximum alignment uncertainty at or below 2 ms. Artifact byte length and SHA-256 are checked for recorded/closed sessions. Free-form labels containing the word `verified` do not count. This is a conservative v0.1 engineering gate, not a universal tolerance; tighter claim-specific limits may be needed.

Physical eligibility also requires a claim-specific uncertainty budget (measurand and unit, standard uncertainty, expanded uncertainty, coverage factor, acceptance tolerance, evidence state and a content-addressed `uncertainty_budget_record`). The budget must be bench-characterized or independently reviewed and its expanded uncertainty must not exceed tolerance. A syntactically valid calibration receipt by itself is not enough. This follows the metrology principle that uncertainty contributes along the calibration chain and fitness for purpose is a separate requirement ([NIST policy](https://www.nist.gov/calibrations/traceability)).

## Fault injection matrix and conformance

The machine-readable matrix lists 32 required positive and negative controls, including clock-method spoofing, missing claim-specific uncertainty budgets, an expanded-uncertainty value that exceeds its pre-registered tolerance, a calculated-but-not-characterized budget, calibration/rights receipt references that do not match the declared artifacts, zero-width uncertainty budgets, and uncertainty receipt contents that disagree with the manifest. Include positive controls so a validator that rejects everything does not pass. At minimum, exercise:
- malformed or reordered headers, empty traces and unknown channels/statuses;
- VALID samples with missing values, non-finite numbers, sequence gaps, timestamp regression and repeated timestamps for a channel;
- repeated timestamps across different channels where simultaneous sampling is permitted;
- explicit sensor-fault rows with missing values, preserved as invalid observations;
- long inter-sample gaps: report the gap while retaining all original rows;
- duplicate IDs, unresolved sensor links, unsupported units, invalid sample rates and missing calibration receipts;
- traversal paths, missing raw artifacts and byte-length/digest tampering;
- absent permission, unverified clocks, missing claim-required channels and simulated evidence relabelled as physical.

A matrix case counts as detected only when the observed decision equals the frozen expectation and required diagnostics are present. “The command exited zero” is insufficient: compare the structured report and the artifact digests.

## Splits and leakage prevention

For predictive tire/vehicle models, partition at the independent unit that can leak identity or operating conditions:
- tire monitoring: group by tire/unit and complete recording segment;
- tire force/compound work: hold out tire specimen/batch, not random rows;
- vehicle dynamics: hold out complete runs and preferably track/day/vehicle;
- powertrain: hold out drive cycles, ambient conditions and test sessions.

Report per-group metrics, worst-group behavior, uncertainty calibration and out-of-distribution cases. A shuffled row split is a smoke test, not evidence of generalization. Fit normalizers and feature-selection only on training data.

### Per-claim uncertainty completeness

Every physical claim needs a separate uncertainty budget for every required measurand—not one budget for the headline score. For example, `ride_energy` needs budgets for bus voltage, bus current, net DC bus energy and ground speed; tire-force characterization needs separate budgets for normal/longitudinal/lateral forces plus slip ratio, slip angle and camber. The Rust schema checks claim/measurand/unit pairs, duplicate pairs, numeric consistency, positive nonzero uncertainty, evidence state and content-addressed budget records. It also requires sensor calibration receipt IDs to refer to the corresponding declared `calibration_record`, and data-rights receipt IDs to refer to a declared `rights_receipt` or `collection_authorization` of the expected kind. Missing even one required measurand keeps the claim ineligible.

## Output contract

Every benchmark case includes both the expected/observed outcome and the required diagnostic tokens; a correct exit class with a missing explanation does not pass. Reports include a canonical SHA-256 computed over the serialized report with its `canonical_report_sha256` field blank; a verifier blanks that field and recomputes it. Every benchmark report should include: benchmark version; source commit; toolchain/build identity; fixture and manifest hashes; case ID and expected/observed result; per-channel loss and timing summaries; calibration/protocol/rights references; software and physical gates separately; failed cases and reasons; claim ceiling. Preserve failing raw fixtures so a regression can be replayed.

## Executable entry point and next concrete work

Run from the workspace root once Cargo/Rust are available:

```sh
cargo test -p xtask frozen_fault_matrix_cases_match_expected_outcomes
cargo run -p xtask -- mobility-telemetry-benchmark \
  --source-commit "$(git rev-parse HEAD)" \
  --out reports/mobility_telemetry_benchmark.json
cargo run -p xtask -- mobility-telemetry-verify-report \
  --input reports/mobility_telemetry_benchmark.json
```

The benchmark command verifies that `--source-commit` exactly matches `git rev-parse HEAD`, refuses dirty tracked/untracked checkout state except the explicitly requested report path, captures the executing environment by running `rustc -Vv` plus `cargo --version` (failure to execute either command fails closed), validates the matrix schema and positive/negative controls, applies each mutation, checks expected outcomes and required diagnostic tokens, and emits a hash-bound JSON report with a canonical report digest. The standalone `mobility-telemetry-verify-report` command then recomputes that digest and cross-checks case counts, pass/fail totals, status, and required-diagnostic consistency without replaying the benchmark. The separate verifier recomputes the canonical SHA-256 after blanking its digest field and checks report consistency; it establishes integrity relative to the stored digest, not cryptographic authorship or independent physical truth. These tools have been authored but not compiled or run in this environment. It does not measure hardware. It has been authored but not compiled or run in this environment.

Next:

1. Execute the Rust tests and matrix on a host with the workspace toolchain; retain the exact-head report and build identity, then run `mobility-telemetry-verify-report` before sharing/archive.
2. Run the full suite 100 times on the same build to measure replay determinism; a unit test checks 100 repeated canonical digest computations and content-tamper detection.
3. Build a host recorder/replayer before MCU firmware; add SocketCAN only as a receive-only, fail-closed adapter and test unavailable timestamp modes.
4. Build a known-signal bench: pulse/step signal generator, synchronized voltage/current reference and independent sample counter.
5. Validate the physical sensor chain and uncertainty before any real-world efficiency, handling, tire-force or safety claim.
