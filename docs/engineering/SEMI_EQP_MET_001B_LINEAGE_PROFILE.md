# SEMI-EQP-MET-001B — Inspection-Bench Configuration Binding and Raw-Observation Lineage

Status: frozen documentation/data contract only  
Parent: #5918 / #5912  
Benchmark parent: #5914 / PR #5915  
Reference validator parent: #5916 / PR #5917

## Purpose

Freeze the inspection-bench evidence-binding profile without creating a second observation,
calibration, device-identity, clock, provenance, or evidence system.

This tranche defines **how canonical references are composed** for the first semiconductor
inspection/positioning/calibration bench. It does not define hardware construction, motion
control, imaging algorithms, semiconductor processing, or physical execution authority.

## Canonical ownership

- **SE-OBS #3695** owns physical observation envelopes, source/receive time semantics,
  physical-vs-derived observation class, raw/normalized artifact identity, calibration refs,
  uncertainty/quality metadata, and non-authority classification.
- **EXEC-ID #4620** owns canonical calibration identity/epoch semantics. A calibration ID
  proves content identity, not correctness/currentness.
- **ROB-INSTR #4866** owns evidence-grade raw-device -> decoder/parser -> typed observation
  discipline, including raw retention and clock distinctions.
- **ROB-REALIZE #4859 / SE configuration owners** own physical article/as-built/configuration
  lineage.
- **SENSE #5858/#5859** owns benchmark custody and anti-self-evidence constraints.

SEMI-EQP-MET adds only a semiconductor inspection-bench **binding profile** over those refs.

## Core laws

- friendly instrument name != physical device identity
- physical device identity != mounted/as-built configuration
- raw bytes/image != decoded or normalized observation
- decoded observation != calibrated measurement
- derived/enhanced image != new physical observation
- software replay != fresh physical acquisition
- same commanded coordinate != same observed physical location
- changed optics/mount/parser/calibration context != same evidence subject automatically

## Identity decomposition

Do not build one opaque mega-ID. Preserve separate identities for:

1. physical/as-built instrument configuration;
2. mounting/remount transform;
3. acquisition firmware/driver/parser profile;
4. calibration identity, epoch, and currentness;
5. raw physical observation;
6. raw artifact bytes/content root;
7. derived artifact / transform / checkpoint;
8. benchmark session/custody context.

A higher-level evidence capsule may bind these identities together without replacing them.

## Reference-only binding profile

A future thin adapter may resemble:

InspectionBenchEvidenceBindingV1

- article_or_as_built_configuration_ref
- sample_fixture_ref
- motion_stage_or_positioning_ref
- imager_or_sensor_ref
- optics_path_ref
- illumination_profile_ref (optional)
- mounting_or_transform_ref
- acquisition_firmware_ref (optional)
- driver_transport_ref (optional)
- parser_normalizer_ref
- source_clock_ref (optional)
- receive_clock_ref
- calibration_ref (optional)
- calibration_currentness_ref (optional)
- coverage_profile_ref
- raw_observation_ref
- raw_artifact_digest
- derived_artifact_refs
- benchmark_session_ref
- execution_authority = none

Exact Rust names must follow the qualified public APIs of the canonical owners.

## Raw-to-derived lineage

Physical sample/reference target
-> exact instrument/configuration
-> raw acquisition
-> SE-OBS physical observation + raw artifact root
-> parser/normalization receipt
-> calibration projection where applicable
-> derived measurement/image/feature artifact

Every derived artifact must retain parent physical-observation/root references.

A transform can improve usability or extract a quantity. It cannot increase the number of
independent physical witnesses.

## Calibration

Changed calibration epoch/currentness changes the measurement-admission context but does not
rewrite the raw historical observation.

Calibration identity and measurement currentness remain distinct:

CalibrationId exists
!= calibration current
!= calibration correct
!= measurement qualified

## Time semantics

Where available preserve independently:

- instrument/source timestamp;
- acquisition timestamp;
- host receive timestamp;
- clock/time-base identity;
- replay scheduling timestamp.

No missing clock relation may be coerced to zero delay or simultaneity.

## Coverage

Coverage is profile-relative. A local field, tile, or region remains local evidence unless a
separate scan/sampling theorem supports a broader claim.

## Synthetic known-answer corpus

Path:

`docs/release/evidence/semi-eqp-met-001b-lineage-corpus-v1.json`

Schema:

`semi-eqp-met-001b-lineage-corpus-v1`

Authority:

`representation_only_no_physical_execution_authority`

Exact case count: 16

Canonical SHA-256:

`df745926fb6e8eecb220f622ccd7cf995c80309a24667f22d7e14d4eae14235f`

The corpus covers:

1. same friendly name / different physical device;
2. changed optics and mount;
3. remount transform identity;
4. parser change with preserved raw root;
5. derived enhancement with one physical witness;
6. orphan derived measurement rejection;
7. calibration epoch change without raw-history rewrite;
8. stale calibration narrowing claims;
9. source/receive timestamp preservation;
10. replay vs fresh physical acquisition;
11. multiple derivations from one physical witness;
12. duplicate-reference rejection;
13. set-like ancillary-reference canonicalization;
14. sample/article lineage mismatch;
15. operational evidence with imported subsystem closure;
16. zero execution authority.

## Production gate

Do not implement production Rust behavior until:

- PR #5917's dedicated reference qualification passes on its exact head;
- required canonical owners expose a sufficiently stable/qualified public surface;
- the implementation can differentially reproduce the frozen corpus without duplicating
  SE-OBS/calibration/provenance semantics.

Until then this contract and corpus are source/data only.

## Claim ceiling

This tranche establishes no real instrument accuracy, optical resolution, positioning
repeatability, traceable calibration, commissioned hardware, semiconductor inspection
performance, process capability, yield, fabrication capability, productive closure, or
physical execution authority.
