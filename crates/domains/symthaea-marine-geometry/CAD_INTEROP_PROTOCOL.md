# Marine CAD Interoperability and Translation Validation Protocol (G3)

Status: G3a evidence contract / research protocol. No STEP reader/writer or production CAD kernel is implemented by this increment.

## 1. Standard baseline and version policy

The initial published exchange baseline is **ISO 10303-242:2025, Edition 4**. ISO records Edition 4 as published in August 2025. The next Edition 5 is a committee draft registered in February 2026; that draft is not the released baseline. A profile must identify the exact schema edition, conformance class and supported exchange content.

The current G3a Rust contract distinguishes:
- AP214: not accepted by the initial AP242:2025 qualification profile;
- AP242:2025 Edition 4: initial released baseline;
- AP242 Edition 5 committee draft: explicitly identified as draft/in-development and not interchangeable with Edition 4.

CAx-IF implementation guidance provides revisioned practices and test assets for exact B-rep, tessellated geometry, PMI, persistent IDs and validation properties. A claim that a standard supports a field does not prove a particular backend implements it. Every semantic field must be covered by backend/version-specific round-trip tests.

Open CASCADE documentation states that the STEP writer defaults to AP214 and AP242 must be explicitly selected. The backend configuration must therefore pin the selected STEP schema and the receipt must record the exact configuration digest. Never infer AP242 from the fact that a STEP file was written.

## 2. Representation and semantic classes

Keep the following distinct:

| Semantic | Representation examples | Claim ceiling |
|---|---|---|
| Displacement-volume shell | G2b closed mesh, tessellated shell, later exact B-rep | Volume below declared design waterplane only; cap is not deck/freeboard |
| Full external hull | Exact B-rep surfaces/solids plus deck/freeboard, transom/stem and declared appendages | External geometry only; not a vessel stability, structural or manufacturing pass |
| Legacy reconstruction | Point cloud, inferred curves/sections, reconstructed B-rep | Reconstructed hypothesis with source and uncertainty; not as-built unless independently surveyed |
| Component/subassembly | Exact B-rep, assembly structure, PMI | Component-level geometry and semantics only |
| Generic surface | B-spline/NURBS or trimmed surface | Mathematical surface only; may not enclose a valid solid |

A tessellation is not an exact B-rep. A B-rep is not a parametric feature-history model. AP242 may carry multiple related representations and metadata, but the receipt must say which representation was actually translated and validated.

## 3. Translation envelope

G3a implements a Rust receipt contract with:
- candidate identity and geometry semantic;
- semantic scope and an explicit deck/freeboard-presence flag;
- source artifact format, digest and source commit;
- source and target representation types;
- explicit AP242 edition;
- STEP file digest;
- exporter and importer backend family, version, build digest and configuration digest;
- frozen tolerance profile and required checks;
- measured translation metrics;
- a deterministic receipt digest.

The receipt digest covers the typed payload, including backend identities, source/STEP digests, semantics, tolerance profile, validation measurements and check dispositions. It is an integrity reference, not a digital signature or proof that the stated backend actually ran.

## 4. Independent validation requirements

A same-kernel export/import round-trip is a diagnostic control, **not independent validation**. A qualified profile requires distinct exporter/importer backend families and exact backend build/configuration identities. Future integration should include:
- Open CASCADE XDE/STEPCAF as an open baseline, with AP242 explicitly configured;
- at least one independently implemented commercial or open B-rep kernel;
- if possible, a second independent tessellation/mesh validation path.

Each profile must require relevant, explicit checks:
- schema/edition;
- source and STEP identity;
- backend independence;
- B-rep validity and shell/solid closure;
- unit/frame/handedness preservation;
- geometric equivalence;
- semantic preservation (PMI, IDs, materials, assemblies, revisions where in scope);
- waterplane/geometry semantic consistency.

The metric checks (length, volume, area, centroid, surface distance and edge gap) must include measured values. A PASS label with a missing metric or out-of-tolerance value is rejected by the G3a contract.

## 5. Suggested validation metrics

Do not use one universal set of tolerances for all vessel scales, surface classes or downstream uses. A profile must state the measurement method and tolerance source for each metric. The G3a receipt supports explicit required-semantic checks for product identity, configuration/revision, PMI/GD&T, materials, persistent IDs and validation properties; each must have an explicit PASS/FAIL/UNKNOWN result.

At minimum record:
- principal dimensions and bounding-box extents;
- absolute/relative volume and area deviations;
- centroid and inertia-property deviations;
- symmetric surface-distance distributions (not just a single point-to-surface average);
- maximum and percentile edge gap;
- shell/solid counts and validity result;
- degenerate face/edge and self-intersection outcomes;
- source/imported topology counts as diagnostics;
- units, coordinate frame, handedness and explicit transformation receipts;
- product structure, persistent IDs, PMI/GD&T and material preservation where required;
- downstream hydrostatic-property deviation on the imported representation.

Exact B-rep-to-B-rep comparisons and tessellation comparisons must be reported separately. A high-resolution tessellation that visually overlays a B-rep does not establish exact geometric equivalence or PMI preservation.

## 6. Benchmark corpus and negative controls

Start with deterministic, analytically known examples:
- boxes, wedges, prisms, cylinders, spheres and torus surfaces;
- the G2b triangular-prism displacement shell;
- convex section lofts and tapered section families;
- exact B-spline/NURBS surfaces with known continuity;
- trimmed periodic surfaces and singularities;
- multi-body assemblies with known transforms and metadata.

Adversarial fixtures:
- open shells, non-manifold edges/vertices, duplicate faces and degenerate entities;
- deliberately reversed orientation;
- micro-gaps, sliver faces, tiny edges and high-aspect-ratio faces;
- self-intersecting faces/shells and nested/inverted shells;
- inconsistent units, frame changes and mirrored handedness;
- AP214 output when AP242:2025 is required;
- AP242 draft schema mislabelled as released edition 4;
- lost PMI, persistent ID, materials, assembly placement or revision;
- silent tolerance widening;
- source file mutation after export;
- a waterplane-capped displacement volume falsely labelled a full external hull.

## 7. Claim dispositions

- QUALIFIED WITHIN PROFILE: every required supported check has a measured result, explicit PASS and within-profile metric.
- REJECTED: a required check fails or a measured metric is outside tolerance.
- INCOMPLETE: required checks/metrics are missing, unknown or unsupported.
- INCOMPARABLE: schema, unit, frame or representation semantics cannot be aligned.
- BLOCKED: rights, source identity, semantic mismatch or missing prerequisites prohibit use.

Unknown information must remain unknown. No implicit unit or coordinate transformation, auto-healing, topology repair, or tolerance widening is permitted.

## 8. Implementation phases

- G3a: typed Rust receipt, semantics, tolerances and deterministic integrity checks (implemented in the companion module; runner qualification pending).
- G3b: pin an Open CASCADE/XDE backend version/build and implement AP242:2025 export/import with the schema explicitly set.
- G3c: independent second-kernel import and geometric/semantic comparison.
- G3d: cross-kernel golden corpus, negative controls and repeatability campaign.
- G3e: full external hull/deck/freeboard surface hand-off; never repurpose the G2b displacement cap as a deck.
- G3f: downstream CAD/DFM and hydrostatic property round-trip qualification.

No G3a result establishes that a real STEP file has been read, written or independently validated.