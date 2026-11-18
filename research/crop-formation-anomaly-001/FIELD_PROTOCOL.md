# CROP-INTEL-001 — Prospective crop-flow and plant-response field protocol

**Version:** 0.1 design  
**Status:** protocol proposal; no field site, samples, equipment or laboratory partner has been confirmed.  
**Primary aim:** measure crop posture and plant response reproducibly, without presupposing a cause or damaging a site.

## 1. Questions

1. What is the per-plant 3D stem/canopy posture field inside, at the boundary of, and outside a reported formation?
2. Does the field contain radial, inward, tangential, sectoral, locally aligned or multimodal directional structure?
3. Are bend, breakage, node dimensions, tissue state or seed outcomes different from matched controls?
4. Which observations discriminate between human mechanical flattening, wind/rain lodging, mixed mechanisms and other explicitly specified hypotheses?
5. Are any measured differences repeatable across independent formations and laboratories?

This protocol does not use the word "genuine" as an outcome. Authorship, physical mechanism, plant response, information content and evidence quality remain independent dispositions.

## 2. Permission, safety and ethics gate

Do not begin fieldwork until all are documented:
- written landowner/manager permission for site access, measurements, and any sampling;
- local aviation permission and applicable flight authorization if a drone is used;
- land-management / biosecurity requirements for entering crop fields;
- institutional ethics and biosafety review where required;
- sample volume, destructive sampling, disposal and compensation agreed in advance;
- named field lead, safety plan, privacy plan, and data-rights plan.

Default to non-destructive mapping. Never trespass, collect without permission, damage a crop to improve imagery, disclose private landowner details, or fly an unauthorized drone. A site can be studied without collecting biological material.

## 3. Study strata and comparison groups

Where available, recruit the following groups under the same crop and field-season context:

A. **Reported formation:** spatially resolved upright, bent, flattened and broken plants.  
B. **Adjacent upright controls:** plants from the same field, cultivar and phenological stage, outside any visible formation effect.  
C. **Mechanical-lodging controls:** controlled, documented hand/board/roller/foot or other specified flattening, only in a designated experimental plot with landowner permission.  
D. **Natural-lodging controls:** independently documented wind/rain or other naturally lodged crop, with weather and field conditions retained.  
E. **Unexposed field controls:** if possible, comparable nearby plots to estimate field-scale variation.

Do not treat the number of stems as the number of independent formation events. The event is the highest-level replicate; locations and plants are nested observations.

## 4. Spatial sampling design

### 4.1 Preserve the observed footprint

Before plant measurements:
- obtain a calibrated, rights-cleared orthomosaic or ground survey;
- register the outline and crop-flow mask with independent control/check points;
- define a field-local coordinate frame and origin;
- record uncertainty on the formation centre, boundary and scale;
- map standing, bent, flattened, broken and unknown regions separately.

Do not borrow a town centroid or approximate map location as the formation centre. If the centre cannot be determined to the precision needed for radial/tangential tests, report those tests as unavailable.

### 4.2 Transects and zones

Use a preregistered sampling grid or radial transects selected from the actual footprint. A practical design is:
- interior zones;
- multiple boundary-crossing transects;
- exterior control bands at predeclared distances;
- matched control plots sampled using the same spatial spacing.

Choose transect count and spacing from the field dimensions, smallest effect of interest, expected spatial correlation and feasible field time. Do not choose them after inspecting the response data.

Store every sampled position and every unmeasured planned position. Record exclusions and missingness reasons; do not silently drop difficult locations.

## 5. Individual-plant measurements

For each sampled plant or defined plant clump, record:

### Geometry and posture
- specimen/sample ID and field-local x/y position;
- stem base and tip coordinates or a calibrated 3D stem-axis vector, where feasible;
- stem inclination relative to gravity/vertical;
- flow azimuth in a declared reference frame;
- bend angle, bend location and curvature profile if measurable;
- breakage, kinking, crushing, uprooting and root-plate state;
- canopy/leaf orientation only as a separate measurement;
- measurement uncertainty, view angle, occlusion and quality flags.

### Plant and field context
- species and cultivar where known;
- growth stage / phenological stage;
- plant height, stem diameter and node/internode position;
- moisture / water status using a named protocol where practical;
- lodging state and the classification rule;
- local slope, soil/ground state and crop-row direction;
- collector, timestamp, instrument and calibration ID.

Do not substitute an image-space angle for a field azimuth unless the image-to-field transform and its uncertainty have passed independent spatial checks.

### Mechanical properties (optional laboratory track)
If a laboratory can perform a qualified and repeatable protocol, measure:
- flexural rigidity / stiffness;
- bending force and/or breaking moment under a declared test geometry;
- stem diameter and cross-section;
- moisture and tissue condition at test time.

These are separate measured outcomes. Do not infer them from a vegetation index, stem inclination or an overhead photograph. Record loading rate, support span, fixture geometry, calibration and failure mode for any mechanical test.

## 6. Imaging and 3D measurement hierarchy

Use the least destructive method that answers the question. Possible complementary methods:
1. calibrated ground-level photographs with scale and coordinate references;
2. multi-view photogrammetry or structured-light/depth capture;
3. terrestrial / mobile LiDAR where canopy density and instrument access permit;
4. permissioned UAV RGB / multispectral / LiDAR survey for wider context;
5. direct manual stem measurements as the reference or audit sample.

Aerial RGB or multispectral segmentation is a posture proxy, not a direct measurement of stem-axis orientation. LiDAR/photogrammetry can improve 3D reconstruction but must still be validated against manually measured stems. Retain raw imagery and instrument data under the declared rights policy; publish only permitted derivatives.

## 7. Blinding, randomization and custody

- Assign random sample labels before laboratory analysis.
- Keep formation/control labels and hypothesized origin hidden from assessors where possible.
- Randomize collection order and lab processing order.
- Use predeclared duplicate samples / repeat measurements to estimate measurement reliability.
- Record each custody transfer, seal, timestamp, storage condition and integrity check.
- Preserve raw files and hashes; derivatives must list parent hashes and software/configuration.
- Maintain an exclusion log and a protocol-deviation log.
- Have an independent analyst replay the geometry extraction and statistical analysis from the frozen input set.

A chain-of-custody record establishes documented handling, not the truth of a causal interpretation.

## 8. Preregistered outcomes and analysis

Declare one or a small set of primary endpoints before collecting or unblinding data. Candidate primary outcomes include:
- stem-axis angular difference between formation and matched controls;
- radial / tangential alignment against a surveyed formation centre;
- bent/flattened/broken state by predeclared spatial zone;
- a mechanical endpoint such as bending force under a fixed protocol.

Secondary outcomes can include node/internode dimensions, tissue histology, seed viability and subsequent germination.

Use hierarchical models with plants nested within transects/locations, plots and formation events. Account for spatial autocorrelation and repeated measurements. Do not run a stem-level test that treats hundreds of plants from one formation as hundreds of independent formations.

Power analysis must declare:
- smallest effect of scientific interest;
- variance and intraclass-correlation assumptions;
- number of independent events / plots;
- samples per event and spatial sampling plan;
- expected measurement error and missingness;
- multiplicity strategy for secondary outcomes.

The existing simple Gaussian power reference is not a substitute for a spatially correlated hierarchical field-study design.

## 9. Crop-flow analysis

For calibrated samples, calculate:
- circular mean direction and resultant concentration;
- direction histogram and normalized entropy;
- local nearest-neighbour alignment;
- centre-relative radial alignment: outward (+1) versus inward (-1);
- centre-relative tangential alignment: counter-clockwise (+1) versus clockwise (-1);
- bend-magnitude mean and spread.

Retain an undefined mean when circular directions cancel or are nearly isotropic. Centre-relative scores require a surveyed/calibrated centre in the same coordinate frame. Report uncertainty from centre location and individual stem-angle measurement. These scores describe geometry; they do not identify wind, human action, electromagnetic effects or any other cause.

## 10. Competing predictions

Before measurements are unblinded, each candidate mechanism must submit:
- predicted spatial distribution of bend directions;
- predicted boundary transition and crop-flow patterns;
- predicted break/bend/root state;
- predicted association with rows, tracks, terrain, wind direction and weather;
- physical plausibility / mechanics constraints;
- a result that would count against the mechanism.

A model is not supported because it can be tuned to imitate the observed circle. It must predict withheld measurements under frozen parameters or predeclared parameter-estimation rules.

## 11. Field-study stop gates

Stop or defer if:
- permission or safety conditions are unclear;
- the centre or coordinate frame is too uncertain for the planned directional statistic;
- the required independent controls are absent;
- the prospective design is underpowered for its primary outcome;
- custody or blinding is broken;
- the sampling would damage crops or disturb a site beyond agreed limits.

An inconclusive or unpowered study is an acceptable outcome. It must not be reframed as positive evidence.

## 12. Deliverables

A qualified field campaign should produce:
1. preregistration and versioned protocol;
2. permissions and ethics/rights record;
3. site map, coordinate frame and uncertainty report;
4. raw per-plant observations and instrument files;
5. control/negative-control provenance;
6. blinded lab results and custody log;
7. deviations, exclusions and missingness report;
8. frozen analysis code, configuration and exact input hashes;
9. independent replay and reviewer disposition;
10. a claim-limited report separating measurement, association and causal inference.

**Current status:** protocol only. No site has been selected, no permission obtained, no samples collected, and no experiment conducted.
