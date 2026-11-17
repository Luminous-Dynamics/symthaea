# Atelier Lightshell — Coupon Evidence Contract v0.1

## Purpose

This contract links every physical coupon observation to the specimen and manufacturing configuration that produced it. It is the bridge between the spacer DoE matrix (#12724), candidate-level benchmark and qualification (#12688 / #12691), and generic measurement/uncertainty owners (#4919, #6225, #6226, #6298). It is not another generic metrology engine.

## Why specimen-level provenance is essential

Yan, Wang and Yu (2026) tested twelve weft-knitted spacer structures and reported construction-dependent compression energy absorption, impact force reduction, airflow resistance and vibration transmissibility. Their square-wave inlays produced multiple frequency-response peaks. They explicitly identify durability and washability as untested future work. Lightshell must therefore preserve the frequency response and report resonance assessment, not reduce vibration to a single average. https://doi.org/10.3390/polym18020151

Schwager et al. (2022) report that flat-textile methods can be unsuitable for thick 3D spacer structures: fixtures can pre-compress specimens or fail to accommodate sample geometry. Their work emphasizes spacer-specific sample preparation, thickness and air-permeability methods, and identifies compression, shear/tilting and abrasion as areas requiring further method development. https://doi.org/10.1007/s10443-021-09959-y

## Specimen record

Each specimen binds:
- unique specimen ID and preregistered DoE candidate ID;
- material lot, construction, process/machine program revision and manufactured article;
- cut orientation (coursewise/walewise/face) and dimensions;
- measured thickness and conditioning method, environment, temperature, relative humidity and duration;
- geometry/sample-preparation QC state and source record.

A specimen remains `proposed` until its actual lot, process, geometry and conditioning are known. It becomes `available` only when those references and measurements are complete and its preparation QC is passed.

## Test-run record

Every executed run binds a method/protocol revision, fixture identity/revision, support/contact boundary through the referenced protocol, environment, instruments, calibration evidence, operator, acquisition time, raw-data reference and SHA-256 digest, replicate/sample counts and status.

Passed runs additionally require a reviewer, acceptance rule, measured observations, method and uncertainty-model references. Failed and inconclusive runs retain the same raw-data and calibration provenance plus a reason. Invalidation retains a reason and does not erase the original record.

## Dynamic response and resonance

Vibration runs retain the ordered frequency-response curve (`frequency_response`, Hz, ratio) and an explicit resonance assessment: `none_observed` or `peaks_reported`. A passed run cannot use `not_assessed`; a peaks-reported result must include peak frequencies and transmissibility. Peaks must fall inside the sweep. The sweep band must be justified by the robot/joint excitation profile; the 0–1000 Hz range in the 2026 paper is a reproduction reference, not a universal robot band.

Compression runs retain a stress-strain curve; dynamic impact runs retain a force-time curve. Raw acquisition data remains authoritative for independent re-analysis; summary curves are not a substitute for the digest-bound raw record.

## Uncertainty and decision boundaries

Measurement records preserve standard or expanded uncertainty, coverage factor where expanded, replicate/sample counts, method and uncertainty-model references, units and measurement boundary. The validator checks record structure; it does not independently calculate uncertainty or certify its calibration model. Use the existing ENG-MEAS / MAT-UQ owners for uncertainty propagation, model discrepancy, and calibration assessment.

Do not compare paper-level results with Lightshell results unless specimen geometry, conditioning, fixture, support, loading, method and metric boundary are genuinely matched. A coupon pass is not robot-module qualification.

## P0 negative controls

Authored test cases cover empty campaigns; duplicate specimen/run IDs; unknown specimen links; available specimens missing lot/process/conditioning/geometry; missing instrument/calibration; malformed raw-data digest; vibration curve absence; frequency points not strictly increasing; resonance assessment omission; expanded uncertainty without coverage factor; failed/inconclusive reasons and candidate-scoped adverse evidence; explicit invalidated-candidate reporting; missing calibration; metric/family mismatch; planned runs accidentally carrying measurements; complete campaigns with pending runs; and unknown fields.

## Benchmark programme

1. **Evidence completeness:** fraction of physical runs with complete specimen, method, instrument, calibration, raw-data and uncertainty records.
2. **False-accept control:** zero acceptance of adversarial stale/mismatched specimen or configuration records in the fixed negative-control corpus.
3. **Dynamic fidelity:** preserve full frequency response, locate and explicitly report resonances, compare test/retest curves with uncertainty.
4. **Repeatability/reproducibility:** decompose variance across within-specimen repeats, specimens, material lots, machine/process revisions, fixture/instrument and operators.
5. **Uncertainty calibration:** preregister interval-coverage targets on held-out repeat measurements only after the measurement system and sample size are frozen.
6. **Change impact:** verify that changes to material lot, construction, process, conditioning, fixture or protocol invalidate dependent claims.

## Current state and claim ceiling

The example is synthetic and entirely proposed; all physical values and identities are intentionally absent. Rust tests are authored but not yet compiled or run. No material has been manufactured or tested under this contract. The module cannot issue a material grade, protection rating, candidate ranking or robot qualification.

## References

- Yan, Wang & Yu (2026), [Influence of Connective Architectures of Inlaid Weft-Knitted Spacer Fabric](https://doi.org/10.3390/polym18020151).
- Schwager et al. (2022), [Development and Standardization of Testing Equipment and Methods for Spacer Fabrics](https://doi.org/10.1007/s10443-021-09959-y).
- Existing engineering owners: Symthaea #4919, #6225, #6226, #6298; Lightshell #12724, #12688 and #12742.