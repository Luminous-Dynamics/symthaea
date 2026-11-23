# Mycelial Graph Interchange and Loss-Aware SWC Projection
**Date:** 2026-11-23
**Status:** Versioned research contract; no vendor round-trip certification claimed

## 1. Purpose

A mycelial network is not always a tree. Anastomoses create cycles; temporal observations change edge evidence; uncertain junctions carry epistemic state; the source volume and calibration determine what its coordinates mean.

The canonical interchange object must therefore preserve the full graph and bind it to the immutable source dataset. A target format that cannot encode those semantics must receive a declared projection, never a silently modified "equivalent" graph.

This work does not attempt to replace Imaris, ZEISS arivis Pro or Fiji SNT tracing and visualization. It provides a graph-assurance interchange layer that can sit beside those applications.

## 2. Implemented canonical sidecar v1

The Rust module `mycelial_graph_interchange.rs` defines `MycelialGraphInterchangeSidecar`.

### Source dataset reference

The source reference contains:
- a URI or relative store reference;
- source format identity (OME-Zarr, Imaris, ZEISS arivis, Fiji SNT or generic);
- dataset and acquisition-session IDs;
- immutable source-content digest;
- axes/multiscale metadata digest;
- spatial-transform-chain digest;
- explicit rights state and digest of file-level rights/access evidence.

The format field is descriptive, not a claim that all vendor-native containers have a public API or lossless adapter. OME-Zarr multiscales and transformations are referenced by digest rather than duplicating image bytes into the graph sidecar.

Rights status is a typed value: verified open, restricted, permission required, unclear or not applicable. A paper's licence is not substituted for file-level evidence.

### Canonical graph snapshot

The sidecar embeds the validated annotated skeleton and binds:
- coordinate frame;
- physical/model length unit;
- dimensionality;
- calibration digest;
- producer identity/version;
- source observation digest;
- junction IDs, types and positions;
- segment IDs, endpoints and scalar lengths.

The sidecar independently stores the graph digest, connected-component count and cycle rank. Validation recomputes these values and rejects mismatches. A graph with cycles is valid in the canonical sidecar.

### Foreign identity bindings

A sorted mapping can bind a canonical junction/segment ID to an external tool ID (for example a trace or filament ID). It is not a replacement for graph identity and cannot override source or calibration digests. Unknown local IDs, duplicates and non-canonical order are rejected.

## 3. Coordinate and provenance binding

The graph coordinates are valid only relative to the declared coordinate frame and calibration. The source axes metadata and spatial-transform digests make the binding auditable, but v1 deliberately does not parse every vendor's coordinate metadata. A future tool adapter must:
1. read axes and multiscale metadata from the source container;
2. resolve the exact transform chain to the graph coordinate frame;
3. verify physical units and orientation (including axis order and handedness);
4. compare its canonical transform digest with the sidecar;
5. reject or quarantine missing / contradictory metadata.

No tool adapter may silently convert pixel coordinates into micrometres from a guessed scale. The calibration manifest must be an independent source.

## 4. SWC is a projection, not a canonical graph format

SWC represents rooted neuronal morphology through rows containing a node ID, node type, coordinate, radius and parent ID. It cannot faithfully represent a cyclic mycelial graph with multiple anastomoses.

The v1 projection contract therefore:
- rejects any graph with cycle rank (eta_1 > 0);
- accepts acyclic forests only;
- requires one explicit root junction for every connected component;
- requires a positive, finite, explicitly supplied radius for every junction;
- orders roots and neighbours deterministically;
- maps branch junctions to SWC type 5 and tips to type 6;
- maps origins, anastomoses, crossings and uncertain junctions to custom/undefined type 0 rather than incorrectly labelling them as soma or a neuronal subtype;
- emits one SWC point per junction, with parent edges connecting endpoints.

A cyclic input returns `CyclicGraphNotRepresentable`. No edges are deleted to force a tree.

### Why even an acyclic projection is marked lossy

The current reconstruction schema stores junction coordinates and scalar segment lengths, not measured hypha centreline samples or per-position radii. An SWC line between two junction coordinates is an endpoint-only geometric representation; it is not a measured hyphal curve. The scalar arc length may differ from the Euclidean endpoint distance.

The export audit therefore always marks the projection lossy and records discarded semantics:
- scalar segment arc lengths;
- measured centreline geometry;
- anastomosis/fusion semantics;
- junction confidence and uncertainty;
- temporal evidence;
- native source/calibration provenance;
- original segment identities.

The text header carries graph and projection digests, coordinate-frame ID, length unit and endpoint-only semantics. The full sidecar remains the authoritative representation.

## 5. Conformance tests

The initial tests cover:
- sidecar graph digest, source digest and topology binding;
- rights evidence and canonical sidecar digest;
- source URI parent-directory traversal rejection;
- foreign IDs bound only to existing local graph entities;
- deterministic SWC projection of an acyclic fixture;
- explicit root requirement;
- explicit measured radius requirement and rejection of zero/invalid radii;
- cycle-rank 1 graph rejected by SWC rather than silently pruned;
- output marked lossy with an explicit discard list and digest header.

Before claiming interoperability, extend with:
- round-trip tests through actual SNT/Atlas trace files and native export APIs where available;
- OME-Zarr axes/transform fixtures covering 2D, 3D and anisotropic z spacing;
- multi-component forest root mapping and isolated junctions;
- multiple-edge, self-loop and malformed-ID adversarial inputs;
- foreign binding round-trip and conflict reconciliation;
- coordinate conversion tests against independent transform implementations.

## 6. Industry and open-tool comparison (checked 23 November 2026)

### ZEISS arivis Pro 4.6

ZEISS lists arivis Pro 4.6 as its current release, with 3D ONNX semantic segmentation and updated AI model-training workflows among the new capabilities. It remains a strong segmentation/visualization environment. Symthaea should import the resulting graph or trace with complete provenance, then add calibrated contact hypotheses, feasible graph posterior and decision receipts rather than claiming to replace the segmentation stack.

### Imaris 11.1

Oxford Instruments lists Imaris 11.1 (8 September 2026) with native CZI reader integration and expanded Arena file recognition/search/sort. Imaris' wider product line is a mature 3D visualization, filament-tracing and quantitative analysis baseline. The appropriate comparison is a reproducible graph/metadata round-trip and downstream graph metrics, not a claim that our current Rust module replaces Imaris visualization.

### Fiji SNT

SNT supports modern multidimensional microscopy data, semi-automated and automated tracing, trace editing, scripting, quantitative morphology analysis and experimental TB-scale datasets exceeding RAM. Its documentation explicitly notes applicability to filamentous structures beyond neurons. This makes it an important open interoperability target: import SNT trace identities and preserve them as foreign bindings while the canonical sidecar retains cycles and non-neuronal semantics.

### Standards direction

OME-Zarr is a cloud-friendly image representation with multiscale and coordinate-transform metadata; RFC-8's 2026 graph proposal is still an early proposal, not a finalized graph standard. The sidecar is therefore a versioned application-level contract that references OME-Zarr rather than assuming a graph standard is already settled.

## 7. Proposed benchmark ladder (not achieved)

1. **Integrity:** 100% digest and topology-summary agreement over all valid round-trip fixtures; fail closed on every source/transform/rights mismatch.
2. **Tree projection:** 100% deterministic SWC row identity and parent-edge recovery for tree fixtures; no projection for (eta_1>0).
3. **Cyclic preservation:** exact graph digest, (eta_0), (eta_1), junction/segment ID and cycle-witness preservation through canonical sidecar round-trip.
4. **Cross-tool identity:** at least 99.9% foreign-ID reconciliation on curated SNT/Atlas/Imaris/arivis adapter fixtures, with unmatched entities reported rather than guessed.
5. **Coordinate fidelity:** maximum transform residual below a predeclared calibrated tolerance, reported separately per axis and unit.
6. **Operational performance:** bounded streaming validation and export, with peak memory and runtime measured by graph size; never omit evidence to hit a speed target.

Targets must be measured on held-out fixtures with source files, tool versions, transform manifests and licence status recorded.

## 8. Claim boundary

The current implementation defines a canonical graph sidecar and a loss-aware SWC projection contract. It does not yet read OME-Zarr metadata itself, parse Imaris/arivis/SNT files, guarantee vendor round-trips or validate a live microscope transform chain. Those require adapters and independent fixtures. The full cyclic graph remains canonical; SWC is optional, tree-only and explicitly lossy.
