# InstantClean physical-run evidence bundle

**Schema version:** 2  
**Status:** research data contract; not a certification, signed evidence format or physical result.

## Purpose

The observer and resource optimizer should never receive loosely joined CSV rows or manually reconstructed run identity. A physical wash cycle is the root entity. Sensor traces, assay results and resource measurements must be explicitly linked to that same run.

The Rust module `experiment_manifest` validates the in-memory `PhysicalRunEvidenceBundle` before downstream qualification. The normalized CSV files in `experiments/` define a portable acquisition format for capture systems and independent labs.

## Bundle contents

A version-2 bundle comprises:

1. One physical run row: unique run ID, service-request ID, explicit compartment/cell ID, compartment-configuration digest, shared/isolated utility-topology ID, exact machine/unit, programme and firmware, protocol, randomized block/treatment assignment, group identities, run start/end, partition, and top-level archive URI/digest.
2. Item rows: unique item IDs belonging to the run, each with material, surface finish, geometry, location stratum and item preparation batch.
3. Instrument calibrations: instrument ID, calibration ID, calibration validity interval, certificate URI/digest and calibration-bundle digest.
4. Sensor traces: instrument/channel/metric/unit, sample count/rate, start/end, calibration reference, source run ID and immutable raw archive reference/digest. Schema v2 treats these as regularly sampled traces and checks that the inclusive sample span `(sample_count - 1) / sample_rate_hz` agrees with the integer-second interval within one second.
5. Independent assay records: one required item × endpoint record, with criterion/metric/method/protocol, measured value, expanded uncertainty, unit, instrument/calibration, assay timestamp, source run and archive reference/digest.
6. Resource measurements: wall electricity, upstream hot-water energy, fresh water, detergent, rinse aid and active operator time, each with an explicit source run ID, declared physical `measurement_boundary_id`, unit, interval increment, measurement method, provenance and calibration where instrumented.

The tables are normalized: all child tables repeat the physical `run_id`. Never encode lists of item IDs or endpoints in a comma-delimited cell.

## Identity and time rules

- A run ID uniquely identifies one complete physical wash cycle. Do not reuse it for retries or rewashes; each attempt gets its own run ID and is linked at the service-request layer.
- Each run also records soil age at wash start, inlet-water hardness, inlet temperature, ambient temperature and relative humidity. The ambient relative-humidity value is expressed in permille and must be no greater than 1,000. All timestamps are Unix UTC seconds in this schema version. Capture systems should retain higher-resolution raw timestamps in trace payloads when available. Sensor traces are regularly sampled in schema v2: `(sample_count - 1) / sample_rate_hz` must agree with `end_utc_s - start_utc_s` within one second, reflecting integer-second interval storage. Irregular/event-driven traces need an explicit future representation rather than a misleading fixed sample rate.
- Run end must be strictly later than run start.
- Sensor-trace and resource-measurement intervals must be contained within the physical run interval. Every resource increment carries its source run ID, which must match the containing physical run.
- Assay measurements may occur after the wash ends, but must retain the exact source run ID and the assay timestamp.
- Machine unit, session, soil lot, detergent lot, water batch, operator, laboratory and site IDs are opaque grouping keys. Their meaning and uniqueness must be governed by the acquisition protocol. Include service request, compartment, compartment configuration and utility topology among possible dataset holdout axes. IDs alone do not prove independence.

## Calibration rules

- Every sensor trace must reference a calibration for the same instrument.
- That calibration must cover the entire trace interval.
- Instrumented resource measurements must reference a calibration for the same instrument covering the complete measurement interval.
- A manual active-operator-time log may omit an instrument/calibration, but still requires a versioned method ID and evidence record.
- Manual fresh-water volume is not accepted in schema v2: use a calibrated volumetric instrument. A future schema can add a separately qualified manual-volume method.
- Assay observation calibration-bundle digest must match the referenced instrument calibration record.

## Completeness and resource accounting

The caller supplies the required resource dimensions for a particular benchmark. A required dimension must have at least one measurement row, even when the measured value is exactly zero. Omission is not interpreted as zero.

Resource rows must use canonical units:
- appliance wall electricity: kWh;
- upstream hot-water energy: kWh;
- fresh water: L;
- detergent mass: g;
- rinse-aid mass: g;
- active operator time: s.

Resource values are interval increments (delta energy, delta water, delta mass or active seconds), not cumulative meter-register readings. Multiple non-overlapping events may be summed by type. Each row declares a stable `measurement_boundary_id` for the physical scope being measured. For a given resource kind and boundary ID, intervals must not overlap, even when different instruments measure that boundary; this prevents meter replacement or redundant instruments from bypassing the overlap guard. Distinct calibrated branch meters may be combined under distinct boundary IDs only when the acquisition protocol documents physically disjoint coverage. Boundary IDs are declarations, not proof: this validator cannot establish that two declared branches are physically disjoint or detect a false split of one physical boundary into multiple IDs. Independent audit of the boundary map is still required. This overlap check is local to one physical-run bundle; it does not detect the same aggregate meter interval repeated across separate concurrent run bundles. Cross-run/system-window reconciliation, including parent-meter versus branch-meter allocation, is tracked in #10300. For `ActiveOperatorTime`, use a pseudonymous per-person boundary/log identity rather than one shared team ID: concurrently active people contribute distinct person-seconds and may legitimately have overlapping clock intervals.

## Fail-closed rules

The validator rejects:
- unsupported schema version;
- missing/duplicate run, item, endpoint, calibration or artifact identity;
- invalid intervals, traces outside the run, or sample-count/rate/time-window contradictions;
- traces with another source run ID;
- missing, duplicate or unexpected item/endpoint assays;
- assays attributed to another run or protocol;
- proxy-only records presented as independent endpoint assays;
- instrument/calibration mismatch or expired calibration;
- resource units that differ from the canonical unit;
- missing resource dimensions, invalid values or missing resource provenance;
- overlapping increments for the same resource kind and declared physical measurement boundary.

It does not impute missing measurements, promote indeterminate assays to PASS, or infer a missing resource as zero.

## Digest and provenance ceiling

The validator checks SHA-256 string shape and internal references only. It **does not hash the referenced file bytes**, verify a digital signature, authenticate an uploader, or establish chain of custody. URIs and hashes supplied in this bundle are untrusted until an external archive verifier retrieves the bytes, recomputes digests, validates signed calibration/assay records and checks trusted issuer identity.

Do not describe a bundle as cryptographically verified merely because this validator accepts it. A separate signed-manifest / content-addressed archive verifier is required before evidence is treated as authenticated. The exact firmware SHA and model/configuration digests are identifiers, not proof of firmware safety or model validity.

## Downstream use

1. Validate the physical-run bundle.
2. Join its `RunPartitionRecord` to the dataset-level grouped-holdout manifest.
3. Verify archive bytes and source signatures in an independent evidence-verification layer.
4. Qualify assay methods and uncertainty using the outcome-evidence protocol.
5. Feed the frozen, complete item × endpoint outcome rows and predictions to `observer_evaluation`.
6. Feed all failed/interrupted/rewash attempts to service accounting; never compute resource-per-successful-item from only successful cycles.

A valid bundle is structurally complete, not necessarily scientifically valid. Assay fitness, experimental independence, generalization, model calibration, performance and product safety are separate evidence gates.

## Tests authored

The 22 Rust unit tests on this branch cover complete-bundle acceptance and resource summation; schema version rejection; source-run mismatch for assay and resource records; duplicate items and assay pairs; missing item/endpoints; invalid assay measurement/uncertainty; missing item material/geometry metadata; invalid ambient relative humidity; traces outside the run; sample-window inconsistency; cross-instrument measurement-boundary overlap and distinct-boundary accounting; blank boundary identity; stale calibration; assay-calibration digest mismatch; resource unit/value/provenance failures; required resource omission; manual-water rejection; and duplicate artifacts across tables.

Cargo execution, formatting and clippy are required on the exact PR head. The source-authored tests are not claimed to pass until run.
