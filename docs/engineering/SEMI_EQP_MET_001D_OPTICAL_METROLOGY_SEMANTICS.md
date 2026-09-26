# SEMI-EQP-MET-001D — Optical Scale, Distortion, Configuration, and Calibration Currentness

Parent: SEMI-EQP-MET-001 #5912
Issue: #5926
Semantic parent: SEMI-EQP-MET-001C #5922 / draft PR #5923

## Purpose

Freeze the optical-metrology evidence semantics for the bounded inspection bench before production image-metrology code or a claim-bearing physical optical campaign.

This tranche is documentation/data only. It introduces no camera driver, image-processing implementation, optical design solver, hardware parameters, vendor selection, or physical execution authority.

## Ownership

- PHOT-ENG #5668 owns optical-system design, simulation, alignment/tolerance hypotheses, and future solver adapters.
- SE-OBS #3695 owns physical optical observations, raw/derived distinction, frames, timestamps, calibration refs, uncertainty, and non-authority classification.
- EXEC-ID #4620 owns canonical calibration identity, epoch, and currentness semantics.
- SEMI-MET #5891 owns semiconductor metrology specialization and later process-gate composition.
- SENSE provenance/custody #5858/#5859 owns anti-self-evidence and replay/derived-representation constraints.
- SEMI-EQP-MET-001D only composes those owners for this inspection-bench profile.

Do not create another ray tracer, lens model, camera calibration ontology, image store, generic distortion engine, quantity/unit system, or execution-authority path.

## Core theorem

Sharp image is not calibrated scale.

Nominal magnification is not measured image scale.

Center-field calibration is not field-wide distortion characterization.

Same camera body plus changed optics/focus/mount is not automatically the same calibration context.

Distortion correction or super-resolution is derived evidence, not a new physical observation.

A current calibration record with a stale reference artifact does not establish a current metrology chain.

## Evidence planes

Keep independently attributable:

1. physical imager/detector identity;
2. optical-path/configuration identity;
3. focus/zoom/magnification setting identity where material;
4. mounting/remount configuration;
5. illumination profile where material;
6. raw optical observation root;
7. scale/reference artifact identity and currentness;
8. calibration identity/epoch/currentness;
9. local scale evidence;
10. distortion/spatial-coverage evidence;
11. derived correction/registration/resampling identity;
12. uncertainty/limits and unresolved regions.

## Spatial scope

Scale and distortion evidence are region/profile relative.

A center-field reference cannot silently establish edge/corner behavior. A local reference feature cannot establish whole-field or whole-sample dimensional accuracy unless the declared evidence actually covers that scope.

## Configuration changes

A materially changed optical path, focus/zoom profile, mount/remount configuration, detector/readout mode, parser/firmware identity, or calibration epoch creates a changed evidence context where the owning profile says the change matters.

Transfer is a proposition requiring evidence. Unsupported transfer remains unsupported.

## Derived-image semantics

Cropping, resampling, distortion correction, registration, denoising, deconvolution, enhancement, and super-resolution remain derived artifacts. They must preserve parentage to raw observation roots and cannot increase the number of independent physical witnesses.

## Frozen synthetic corpus

Path: `docs/release/evidence/semi-eqp-met-001d-optical-corpus-v1.json`

Canonical SHA-256: `8034e7d16f375fcfe69e210629bda9d6fdda6c29e5eb59878c5bbf93f3a3e64b`

The corpus contains exactly 16 benign synthetic cases:

1. sharp image without scale/reference -> qualitative only;
2. current scale reference + current calibration -> candidate local dimensional measurement;
3. nominal magnification without physical scale reference -> uncalibrated nominal scale only;
4. center-field calibration with edges uncharacterized -> coverage limited;
5. current calibration with stale reference artifact -> stronger measurement unavailable;
6. same friendly camera name with changed optics -> distinct configuration context;
7. material focus/zoom change with no transfer evidence -> calibration transfer unestablished;
8. remount changes optical transform -> distinct context;
9. distortion correction -> derived artifact, raw physical root preserved;
10. super-resolution/enhancement -> no new physical resolution witness;
11. crop/resample from one raw image -> one physical witness;
12. local scale reference requested for whole field -> coverage limited;
13. derived scalar with missing raw image -> lineage invalid;
14. recalibration creates new epoch while historical evidence is preserved;
15. replay of raw image/calibration processing -> software replay, no fresh optical acquisition;
16. optical/calibration evidence -> zero physical execution authority.

## Future independent-validator requirements

A future stdlib-only oracle must hard-bind digest/schema/authority/case identities and derive the expected relationships rather than echoing expected fields. It must preserve physical-vs-derived lineage, local-vs-whole-field coverage, configuration/calibration discontinuities, reference-artifact currentness, replay-vs-acquisition, and zero execution authority.

## Production/physical gate

Production Rust and claim-bearing optical qualification remain blocked until #5917, #5921, and #5925 dedicated reference workflows pass on their exact heads, canonical owners expose sufficiently stable public surfaces, and the production specialization can differentially reproduce the frozen A/B/C/D corpora.

## Prohibited content

No hardware dimensions, focal lengths, optical powers, exposure values, illumination powers, motion parameters, electrical operating values, vendor BOM, semiconductor process parameters, hazardous materials, or high-energy optical operation.

## Claim ceiling

This tranche establishes no real spatial resolution, scale accuracy, distortion bounds, calibrated optics, traceable reference chain, commissioned metrology equipment, wafer inspection performance, semiconductor process capability, fabrication capability, or physical execution authority.
