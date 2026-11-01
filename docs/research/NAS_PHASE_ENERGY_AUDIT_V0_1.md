# Marker-synchronized application-event energy audit v0.1

Tracking: #9755  
Implementation: `symthaea-architecture-engineering::phase_energy_audit`  
Dependencies: #9740 raw PPK2 trace audit; #9744 metrology uncertainty; #9595 artifact identity; #9601 device readback.

## Objective

A per-inference energy trace is not a full application energy measurement. A deployed node can consume meaningful energy during wake, sensor acquisition, preprocessing, postprocessing, communication, storage, fallback and sleep transitions.

This v0.1 auditor aligns a separate trigger-marker stream with the raw PPK2 current samples. It integrates each declared phase and the complete outer event envelope from the same current trace. It does not infer phase boundaries from current-amplitude changes.

The event is:

`wake → acquire → preprocess → inference → postprocess → communicate/store/act → sleep`

The phase protocol can define the subset relevant to a particular application, but the first and last phases must be required so the event's measurement envelope is well-defined.

## Inputs

### Raw sample CSV

Exported from `ppk2_samples.parquet` using the upstream dataset helper. Required fields:

`segment_id,time_ms,current_uA`

The samples are first validated with #9740, including finite/non-negative current, strictly increasing timestamps, the nominal sample rate and gap policy.

### Marker CSV

Required columns:

`event_id,segment_id,phase_id,edge,marker_timestamp_ns,sequence,clock_domain_id,marker_source_id`

Each declared phase needs exactly one `START` and one `END`. Marker sequence numbers must be unique within the event and mapped timestamps must be non-decreasing in sequence order. Every event maps to exactly one PPK2 segment, and one segment cannot be reused for multiple application events.

The marker source is a separate GPIO/logic-analyzer or otherwise qualified trigger stream. Phase boundaries are never guessed from current changes.

### Clock mapping manifest

A strict, content-addressed manifest binds marker clock domain/source to the PPK2 time domain:
- marker and PPK2 clock identifiers;
- marker origin (ns), PPK2 origin (ms);
- calibrated milliseconds-per-marker-nanosecond scale;
- scale tolerance and maximum alignment error;
- calibration evidence digest and mapping digest.

The default unit conversion is (10^{-6}) ms/ns, but the manifest must declare a calibrated scale and explicit tolerance. This does not authenticate the clock-calibration evidence; its digest must be checked by the acquisition/provenance process.

### Phase protocol manifest

The strict manifest binds:
- benchmark and protocol identity;
- ordered phase list and required/optional declaration;
- required first and last boundary phases;
- maximum acceptable total unattributed duration;
- task-outcome evaluator digest;
- accepted-success outcome policy;
- protocol digest.

Accepted-success outcomes are explicit per benchmark. `incorrect`, `failed`, and `not_attempted` can never be configured as successful. A benchmark may explicitly choose whether a safe fallback or abstention constitutes success.

### Outcome CSV

Exactly one outcome row per marker event is required:

`event_id,outcome,artifact_sha256,device_session_sha256,input_sha256,output_sha256,quality_metric_id,quality_value,deadline_met,retry_count`

Allowed outcome values:
- `correct`
- `incorrect`
- `abstained`
- `fallback`
- `failed`
- `not_attempted`

The input digest is mandatory. Correct, incorrect and fallback outcomes require an output/result digest. Quality metric and value must be supplied together or both omitted. Deadline is an explicit boolean; retries are counted.

Each outcome row also binds the exact candidate artifact and physical device/session receipt that produced the event. The per-event outcome digest binds artifact, device session, input/result identities, task outcome, quality value, deadline and retries to the energy receipt. The device-session digest must resolve to physical device identity and readback evidence through the acquisition/provenance layer. It does not prove that the outcome is semantically correct; the pinned task-outcome evaluator and its independent verification are separate evidence requirements.

## Integration method

Markers are mapped into the PPK2 timebase through the frozen clock mapping. Each phase boundary is interpolated between its bracketing PPK2 samples. The current integral uses trapezoidal integration over the actual sample timestamps, including interpolated boundary points:

`energy_uJ = voltage_V / 1000 × integral(current_uA × time_ms)`

Every phase has its own duration, energy, time-weighted mean current/power, maximum observed current and conservative boundary-alignment error bound:

`alignment_energy_bound_uJ = 2 × max_current_uA × voltage_V × alignment_error_ms / 1000`

This is a worst-case bound from both phase boundaries, not a statistical uncertainty interval. It is included separately so downstream metrology can combine it only under a justified uncertainty model.

The event envelope integrates from the required first-phase start to the required last-phase end. Phase energy is summed separately. Their difference is an explicit unattributed energy residual, and gaps are reported as unattributed duration. The total event integral therefore includes transition energy rather than dropping it.

## Event statuses

- `COMPLETE_PHASE_COVERAGE`: all declared phases are marked and unattributed time is within the protocol limit.
- `UNATTRIBUTED_INTERVALS_PRESENT`: whole event energy is available, but a declared optional phase is missing or transition gaps exceed the permitted attribution limit. The residual remains visible.
- `MISSING_REQUIRED_PHASE`: a required phase is missing or incomplete. The whole-event energy is not certified and no energy-per-successful-event result is issued for that event.

A complete whole-event integral can still have an `UNATTRIBUTED_INTERVALS_PRESENT` status; the energy was integrated, but not every joule could be attributed to a named phase.

## Energy per successful event

An event is counted in `successful_completed_events` only when:
1. its outcome belongs to the protocol's accepted-success set;
2. its deadline was met; and
3. a complete required event envelope was measured (no missing required phase).

`mean_energy_per_successful_event_uj` is computed from those events only. Failed, incorrect, late, not-attempted and missing-required-phase events are not promoted as successful-energy observations. Unsuccessful event energy remains present in the individual event receipt for failure and retry accounting.

This is still a reported outcome record, not an independent semantic proof of task correctness. The task-specific evaluator digest is pinned; the evaluator itself must be reviewed and independently applied before an application-success claim.

## CLI

From repository root:

```bash
cargo run -p symthaea-architecture-engineering --bin phase_energy_audit -- \
  samples.csv markers.csv outcomes.csv <voltage_v> \
  clock-mapping.json phase-protocol.json phase-energy-receipt.json
```

The CLI currently uses the PPK2 audit defaults (nominal 100 kHz sampling, 20% median-rate tolerance, 2.5× maximum sample-gap ratio); its voltage is explicit. For empirical comparisons, freeze all trace and clock settings in the run protocol and bind them to the acquisition receipt.

## Receipt and replay

The receipt binds:
- exact sample, marker and outcome CSV hashes and byte counts;
- upstream PPK2 trace receipt digest;
- clock-mapping and phase-protocol digests;
- voltage, sample rate and gap policy;
- per-phase and whole-event energy, duration and boundary-alignment bounds;
- unattributed energy/time;
- per-event candidate artifact and device-session digests, outcome/input/output digests, quality value, deadline and retries;
- counts and mean energy of successful completed events;
- canonical, domain-separated receipt digest.

`validate_phase_energy_audit_receipt` revalidates the portable receipt, phase ordering, energy/power relationships, boundary uncertainty bounds, event/outcome digests and aggregate status counts. `verify_phase_energy_receipt_against_sources` rebuilds the complete receipt from the exact source sample, marker and outcome CSVs and requires exact equality.

Neither receipt validation nor source replay authenticates the physical trigger wiring, clock calibration or task evaluator by itself. Those require independent acquisition and provenance evidence.

## Authored controls

Synthetic tests currently cover:
- exact phase and whole-event energy integration;
- inclusion and reporting of transition-gap energy;
- missing required-phase refusal;
- optional missing-phase non-promotion to complete phase coverage;
- incorrect or deadline-missed outcomes excluded from successful-event energy;
- missing/duplicate outcome records;
- clock-domain substitution;
- overlapping phase windows;
- one PPK2 segment assigned to multiple application events;
- rehashed phase-energy field tampering and source replay.

Tests are authored but have not been compiled or executed in this environment because Cargo/rustc/rustfmt are unavailable. No raw Parquet trace, trigger acquisition, physical device event, task evaluator or energy result has been audited.

## Benchmark programme

1. Keep NATS-Bench-MCU v2 inference-only energy as a controlled architecture sub-benchmark.
2. Create one nRF5340 sensor event profile with real sensor/peripheral acquisition and explicit GPIO markers.
3. Add sleep/wake transitions and a declared duty-cycle schedule, aligned with EEMBC ULPMark-CP principles.
4. Add peripheral-inclusive measurements for application-relevant ADC/SPI/RTC operations, aligned with ULPMark-PP.
5. Use trigger-aligned pre/inference/post phases when comparing with Bartoli et al.'s IJCNN 2025 methodology.
6. Include radio/communications only when the actual radio and retransmission behaviour are measured in the same acquisition envelope.
7. Publish inference-only and whole-event Pareto fronts separately, and optimize energy per accepted correct, deadline-met event subject to task quality and fleet constraints.

Do not claim an official EEMBC ULPMark result unless the official rules, license and execution process are followed. Aligning with a benchmark's principles is not the same as submitting its score.

## Claim ceiling

- The marker stream records trigger events; it does not prove firmware semantic correctness.
- Clock mapping and boundary uncertainty are part of the evidence, not eliminated by interpolation.
- A missing required phase means no complete event-energy claim.
- Whole-event energy is not battery life.
- A battery-life projection requires a separate duty-cycle/workload, sleep residency, communications, battery and maintenance model with uncertainty.
- No SOTA improvement or physical measurement is claimed until the raw traces, marker/outcome streams, and target event are acquired and independently replayed.

## Sources

- EEMBC ULPMark-CP: https://www.eembc.org/ulpmark/ulp-cp/
- EEMBC ULPMark-PP: https://www.eembc.org/ulpmark/ulp-pp/
- EEMBC ULPMark-CM: https://www.eembc.org/ulpmark/ulp-cm/
- Bartoli et al., phase-aligned TinyML energy/latency methodology (IJCNN 2025): https://doi.org/10.1109/IJCNN64981.2025.11228997
- ITU-T L.1341 (2025): https://www.itu.int/epublications/publication/itu-t-l-1341-2025-12-functional-requirements-for-energy-efficiency-in-intelligent-internet-of-things-platforms
