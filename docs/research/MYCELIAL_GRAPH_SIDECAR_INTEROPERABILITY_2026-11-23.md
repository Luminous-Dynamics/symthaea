# Mycelial Graph Sidecar Interoperability v0.1
**Date:** 2026-11-23  
**Status:** Source-bound interchange contract and validation protocol; native vendor adapters and end-to-end vendor round trips remain future work

## 1. Why a sidecar instead of a proprietary project file

The image/volume remains in its native container. The graph, annotation provenance and application identity mappings live in a separate, versioned sidecar that references the source image by URI, group path and content digest.

This gives us a stable contract across OME-Zarr, Fiji/SNT, Imaris, ZEISS arivis and future tools without pretending that any one vendor's project format is the universal scientific source of truth.

The sidecar is not an OME-NGFF core-schema extension. It is a separately versioned JSON-serializable data contract which can be stored adjacent to an OME-Zarr dataset or in a repository / research-object bundle.

## 2. Current standard facts and compatibility decision

### OME-Zarr / NGFF

- OME-NGFF 0.6 was released on **14 September 2026**.
- Version 0.6 adopts RFC-5 coordinate systems, transformations and scene metadata. Transformations identify input and output coordinate systems, and the coordinate-transform graph is part of the scientific meaning of the spatial coordinates.
- RFC-8 Collections and Extensibility was still **R1 — Under review** in the official NGFF RFC index as of 19 October 2026. We therefore should not claim that a graph sidecar is already a standardized OME-Zarr node type or depend on RFC-8 becoming final.

The sidecar v1 source record therefore binds:
- immutable image-container URI, dataset/group path and dataset ID;
- OME-NGFF version where the source uses OME-Zarr;
- the NGFF coordinate-system name and transform metadata path for 0.6;
- a digest of the exact transform chain;
- source image digest, frame identity, application coordinate-frame identity and calibration digest.

The graph source digest, frame, coordinate frame and calibration must match the source reference. For OME-NGFF 0.6 the named coordinate system and transform path are mandatory. A missing or mismatched transform reference fails closed.

### Existing application capabilities

- **ZEISS arivis Pro 4.6** (released 2 October 2026) supports externally trained 3D ONNX semantic models in its DL Segmenter, annotation-to-cloud training workflows and 2D/4D interaction.
- **Imaris 11.1** (8 September 2026) adds native CZI reader integration, expanded file recognition and a Training Hub. Its product family provides mature scientific 3D visualization and filament tracing workflows.
- **Fiji/SNT** supports multidimensional tracing, semi-/automated routines, editing, scripted analysis, calibrated quality control and experimental TB-scale datasets; SNT documentation explicitly notes uses beyond neuronal structures for filamentous objects.

The strategy is not to reproduce every segmentation, visualization or vendor workflow. It is to preserve the graph and its evidence across those workflows.

## 3. Rust sidecar contract

Implemented in `mycelial_graph_sidecar.rs` and exported from `symthaea-canvas`.

### Source binding

`MycelialGraphImageReference` binds the graph to the exact source image and coordinate system. OME-Zarr references include format/version, group path, coordinate-system name, transform metadata path and transform digest.

The contract never copies image bytes into the graph sidecar. The source URI is a locator; the content digest is the identity check. A changed image at the same URI is a different observation.

### Graph identity and provenance

The sidecar embeds the typed `AnnotatedMycelialSkeleton` with calibrated junction positions, segment endpoint IDs, segment lengths, source observation digest, coordinate frame, calibration and producer provenance.

A sidecar source reference that does not match the graph's source digest, frame, coordinate frame or calibration is rejected.

### External object identity mappings

`MycelialExternalObjectMapping` maps application-specific object IDs to internal junction or segment IDs. Each mapping includes:
- target application identifier and version;
- target dataset identity;
- external object identity;
- internal entity kind and ID;
- digest of the import/export receipt or identity-adjudication evidence.

Mappings are canonically ordered, unique per external application/dataset/object identity, and must point to an existing graph entity. Multiple application identities can map to the same internal graph entity; provenance is retained rather than silently choosing one.

### Content identity is distinct from validity

The sidecar content digest hashes the exact fields and graph geometry, even if the object is semantically invalid. `validate()` is a separate operation. This is deliberate: changed/tampered content must still have a distinct digest so a receipt can identify it before an independent validator rejects it.

## 4. Topology-aware interchange and the SWC boundary

The interoperability assessment computes:
- junction and segment counts;
- connected components;
- cycle rank `beta_1 = E - V + C`;
- branch-junction count;
- whether the topology is a forest.

A forest can be topology-compatible with a tree-oriented encoding. A cyclic network is not losslessly representable as a single SWC tree/forest without omitting or duplicating edges.

Even an acyclic graph is **not currently ready for a geometrically faithful SWC export**. The current reconstruction schema stores junction endpoints and segment length, but not sampled centerlines or calibrated segment radii. A straight-line interpolation or invented radius would fabricate geometry. The assessment therefore records:
- topology compatibility separately from geometry readiness;
- the exact missing fields;
- an explicit cyclic-topology blocker when cycle rank is nonzero.

No SWC projection is silently emitted in v1. A future lossy export must state exactly which edges/geometry were omitted and retain the native graph sidecar as the authoritative record.

## 5. Round-trip assurance

`evaluate_mycelial_graph_sidecar_round_trip` compares the source and imported sidecars. It checks graph digests, source-image digest, coordinate frame, calibration digest, junction count, segment count and cycle rank.

A graph round trip is lossless only when all these checks match. A vendor's successful import message is not sufficient. The adapter must provide external-ID mapping receipts, preserve physical units and coordinate transforms, and return a graph that independently matches the source digest.

### Proposed round-trip benchmark

For every supported adapter and each fixture:
1. Import the source image reference and graph sidecar.
2. Export through the target tool to its supported interchange representation.
3. Re-import to a new sidecar.
4. Verify the graph and source image digests.
5. Verify coordinate-system and calibration binding.
6. Compare connected components, cycle rank, branch count and every external ID mapping.
7. Record any dropped, merged, split or synthesized object.

Proposed targets (not measured):
- 100% digest-preserving round trips for the native sidecar path.
- 100% explicit failure or loss reports when a target format cannot represent cycles.
- Zero silent edge deletions, ID remappings or coordinate-frame changes.
- Zero graph-to-pixel coordinate changes beyond a predeclared physical tolerance.
- 100% of external object IDs either mapped with a receipt or reported as unresolved.
- A round-trip manifest and result digest for every adapter/version pair.

Metrics must be reported per application version and source format. Do not pool successful tree fixtures with cyclic fixtures to conceal failure.

## 6. Proposed adapter sequence

1. **Native OME-Zarr sidecar**: read/write a JSON sidecar adjacent to NGFF 0.6, using its coordinate-system and transformation metadata as the spatial binding.
2. **Fiji/SNT**: import/export the graph-compatible subset and preserve SNT object/trace identifiers. Do not force cyclic graphs into SWC.
3. **Imaris**: preserve stable object identity and physical coordinate metadata through a supported interchange path; assess whether the target export represents actual filament centerlines.
4. **arivis**: consume/export object or segmentation references while maintaining the graph as a separate, cycle-capable structure.
5. **Atlas**: add a target adapter only after confirming its current input/output contract and licensing.

A connector should be released only when there is an executable, independently checked round-trip fixture. A file extension or successful UI import is not proof of semantic fidelity.

## 7. Dataset and benchmark priorities

- Use small, synthetic, version-controlled fixtures to exhaustively test trees, cycles, disconnected graphs, branchpoints, tip-to-side contacts and uncertain contacts.
- Use the Sten et al. 2025 Z-stack graph-reconstruction study as an academic reconstruction comparator, reproducing its image-level split and annotation tolerance before comparing F1.
- Use the 2026 Agaricus scan-imaging / cord-remodeling dataset for longitudinal sidecar tests only after auditing the exact raw-file license and coordinate metadata.
- Use the 2026 Hyphal Highways Device dataset for time-series and assay-local graph links, keeping liquid connectivity separate from fungal graph identity.
- Keep the AM-fungal flow dataset (about 25,000 trajectories) as a permission-request track; its publication states that the data are available upon request.

## 8. Claim boundary

Implemented: a versioned source-bound graph sidecar, external application object mapping validation, topology/forest assessment, explicit SWC geometry-readiness blockers and exact sidecar round-trip comparison.

Not yet implemented: native readers/writers for proprietary Imaris or arivis projects, SNT/Atlas plugins, actual vendor dataset round trips, OME-NGFF schema registration, centerline sampling, radius measurement or lossless SWC geometry export.

No measured adapter performance or industry superiority is claimed.

## 9. References

- OME-NGFF 0.6 version history (released 14 September 2026): https://ngff.openmicroscopy.org/specifications/0.6/version_history.html
- OME-NGFF RFC-8 status (R1, under review as of 19 October 2026): https://ngff.openmicroscopy.org/rfc/8/index.html
- ZEISS arivis Pro 4.6 release notes (2 October 2026): https://knowledge.zeiss.com/rms/en/arivis-pro/release-notes/arivis-pro-4-6
- Imaris 11.1 release notes (8 September 2026): https://imaris.oxinst.com/support/imaris-release-notes/
- Fiji/SNT documentation: https://imagej.net/plugins/snt/index
