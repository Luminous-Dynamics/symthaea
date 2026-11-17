# CROP-INTEL-001 — Evidence and geometry data model

**Version:** 0.1 proposal  
**Status:** schema design; no ingestion service implemented.

## Design rule

The canonical unit is not a photograph and not a "crop circle" label. It is an **evidence assertion** about a reported formation event, linked to immutable source assets and reproducible transformations.

Do not overwrite disagreements. Two archives may give different dates, coordinates, dimensions, or descriptions; retain both source assertions and record the adjudication separately.

## Entity model

### 1. Formation event

A candidate physical occurrence:
- stable internal event ID;
- discovery/report date and time, separately from alleged formation time;
- first-seen / last-seen interval and temporal precision;
- country, locality and field/site reference;
- geolocation point or polygon with horizontal uncertainty;
- crop species/cultivar and phenological stage, nullable;
- measured footprint and measurement uncertainty;
- event deduplication state (distinct, possible_duplicate, merged_with_review, unresolved);
- authorship, mechanism, plant-response and information-content dispositions as separate fields.

The event record is a registry handle, not a claim that the event's origin or details are known.

### 2. Source assertion

A statement made by a specific source:
- assertion ID and event ID;
- source organization, author/reporter and source URL;
- original record identifier and archive retrieval timestamp;
- exact asserted value and unit;
- assertion type (date, location, crop, measurement, witness, authorship, interpretation, sample, etc.);
- source's own confidence or wording, preserved verbatim where lawful;
- reliability assessment and reviewer rationale, separately from the source's assertion;
- correction/supersession relationship, without deleting the original assertion.

### 3. Image asset

An image or video reference:
- immutable asset ID and source URL;
- original file hash when a lawful copy is held;
- publisher, photographer and credit;
- licence/permission state (redistributable, display_link_only, analysis_permitted, permission_required, unknown, prohibited);
- capture time versus publication time, separately;
- camera/view metadata if available;
- pixel dimensions, colour space and known edits;
- rights and attribution notes.

Do not copy or redistribute third-party aerial photographs merely because an archive exposes them. If only linking is permitted, retain a link and metadata rather than the pixels. Every crop, resize, threshold, perspective correction or annotation becomes a derived asset with a parent hash and transformation record.

### 4. Geometry reconstruction

A vector reconstruction derived from one or more image assets:
- geometry model ID, parent asset IDs and source event ID;
- coordinate system and units;
- image-to-field transformation and its fit/validation evidence;
- scale and orientation uncertainty;
- primitives (circle, arc, line, polygon, raster region, centreline, graph edge);
- primitive geometry and covariance / tolerance;
- standing/flattened/unknown mask with confidence;
- annotator or extractor identity and method/version;
- independent reconstruction ID and adjudication state.

A circle radius from an uncalibrated aerial photograph is image-space geometry, not a field measurement in metres.

### 5. Observation / environmental context

A time-bounded measurement with explicit provenance:
- observation ID, type, value, unit and uncertainty;
- location/geometry and time interval;
- instrument, calibration, data source and version;
- weather source, station distance, model/reanalysis version and representativeness notes;
- terrain, slope, soil and agricultural management context;
- missingness reason and known confounders.

Never turn absent weather, soil or crop metadata into zero or a guessed value.

### 6. Biological specimen and custody

A specimen record:
- specimen ID and parent event;
- exact sample coordinates, depth/plant part and sampling role (formation, boundary, control, mechanical control, natural-lodging control);
- crop/cultivar/phenology and collection time;
- collector, landowner permission reference and field protocol version;
- container, preservation, transport and storage conditions;
- every custody transfer with sender, receiver, time, seal and integrity status;
- blinded laboratory ID and analysis protocol;
- result, raw-data reference, uncertainty and exclusions.

A custody record proves documented handling, not that the biological interpretation is correct.

## Provenance graph

Expected relationships:

- formation_event ← source_assertion ← source_record
- formation_event ← image_asset ← derived_asset ← geometry_reconstruction
- formation_event ← environmental_observation
- formation_event ← specimen ← custody_transfer ← lab_result
- geometry_reconstruction ← descriptor_set ← decoder_run ← claim_record

Each derived object records:
- parent object IDs and hashes;
- code repository, commit and tool version;
- parameter/configuration hash;
- deterministic seed, if any;
- execution timestamp and environment;
- complete warnings, exclusions and failure state;
- reviewer / independent replay disposition.

## Unknowns and confidence

Use distinct states:
- unknown: no reliable value available;
- not_observed: the field was not measured;
- not_applicable: the field does not apply;
- conflicting: credible sources disagree;
- measured: value with units and uncertainty;
- asserted: a source claims a value but no independent measurement exists.

Confidence attaches to a specific assertion or derived estimate, never to the entire formation as a vague scalar. A source's confidence is not the programme's confidence.

## Claim record

Every scientific claim references the exact evidence and analysis that support it:
- claim ID, wording, scope and claim level (S0–S3 or physical/biological disposition);
- supporting and contradicting evidence IDs;
- model alternatives and prior assumptions;
- all decoder/model variants explored;
- null/control definition and statistical correction;
- estimated effect/metric and interval;
- unresolved confounders and known failure modes;
- replication status and independent reviewer;
- explicit claim ceiling and supersession history.

## Import policy

For external catalogues:
1. import source assertions as source assertions;
2. identify probable duplicate events without erasing separate source records;
3. preserve unresolved aliases;
4. retain geolocation precision class (exact field, approximate field, locality centroid, unknown);
5. record the rights status of every linked image;
6. keep source counts, canonical-event counts and reviewed distinct-event counts separate.

Do not optimize for the largest headline count. A defensible uncertain record is better than a false merge.
