# Atelier Lightshell — Pattern and Panel Topology Contract v0.1

## Purpose

Bind robot compiler panel IDs to source-native textile pattern pieces, seam/attachment graph, articulation features and service openings. Preserve semantic mappings to APPAREL-INTEROP rather than implementing a parallel apparel pattern engine.

## Ownership and interoperability

- Robot Garment Compiler #12710/#12719 owns candidate robot panel geometry and its conservative zone/keep-out admission.
- APPAREL-INTEROP #11750 owns generic 2D pattern, 3D garment, source format and machine-independent Knitout semantic preservation.
- MFG-PROC #5686 and the Symtropy textile-process owner retain manufacturing-process capability and machine execution semantics.
- Lightshell #12785 owns only the robot-specific mapping between compiler panels, textile piece topology, joints and service access.

## Current format status matters

ASTM D6673-10 was withdrawn in 2019 without a replacement. Its official page describes 2D flat sewn pattern exchange and explicitly excludes pattern-piece relationships and 2D/3D geometry correspondence. A DXF file may still be a valid legacy source, but the repository must not imply current ASTM D6673 conformance or full garment engineering semantics. [Official ASTM metadata](https://store.astm.org/d6673-10.html).

Knitout v0.6 is a community-maintained, machine-independent low-level instruction format. It can bridge design tools, simulators and machine inputs, but it is not a high-level garment product model and does not establish the target machine's capabilities. [Knitout v0.6 specification](https://textiles-lab.github.io/knitout/knitout.html).

## P0 entities

- `CompilerBinding`: exact robot geometry, compiler revision, compilation identity and geometry state/digest.
- `InteropBinding`: exact #11750 contract revision and sidecar digest.
- `SourceArtifact`: native DXF/OBJ/USD/Knitout/pattern identity, format version and digest.
- `TopologyPanel`: pattern piece ↔ exact compiler panel/zone, 2D piece digest, 3D surface, correspondence ref/state, coordinate frame, grain and material orientation.
- `BoundaryEdge`: globally unique panel edge and semantic role.
- `PanelConnection`: paired edges, construction route, path, revision and explicit mismatch allowance.
- `ArticulationFeature`: affected panels, robot joint, owner-supplied motion envelope and feature region.
- `ServiceFeature`: opening edge(s), exact compiler service keep-out and access path.
- `SemanticObligation`: source-native, sidecar, external-owner, unsupported, lossy or unknown disposition for each required meaning.

## Topology invariants

1. Every topology panel must match a compiler panel ID, zone and coordinate frame exactly.
2. Every panel binds one source artifact/piece, pattern revision, piece digest and 2D↔3D correspondence.
3. Every seam/attachment connection uses two extant edges on two distinct panels.
4. Each seam/attachment boundary edge is consumed exactly once; no edge may belong to two connections.
5. Connected edge lengths must meet explicit mismatch plus source tolerances; no implicit ease is introduced.
6. A removable attachment binds an interface-design reference and remains distinct from proven retention or repairability.
7. Articulation features bind the exact robot joint and motion-envelope owner reference.
8. Service openings bind an actual service/emergency-access keep-out affecting the same panel zone.
9. Required semantic obligations are explicit. Unknown/lossy/unsupported semantics block source-bound topology unless declared out-of-scope with a deliberate disposition.
10. A topology preflight requires CAD-bound geometry and verified 2D↔3D correspondence. Synthetic boxes cannot be promoted.

## Synthetic elbow example

The example represents upper-arm, elbow-gusset and forearm panels, one sewn seam, one proposed removable attachment, an articulation feature and a forearm service opening. It is a graph/semantics fixture only. Its IDs, pattern mappings, source digests and route references are synthetic placeholders; no real DXF/Knitout file, CAD, machine or physical garment is asserted.

## P0 adversarial tests

- dangling edge and panel refs;
- duplicate edge/piece IDs and edge reuse;
- seam length mismatch;
- zone/frame mismatch between compiler and pattern panel;
- stale/invalid source artifact or piece digest;
- unknown/lossy 2D↔3D mapping;
- missing articulation joint/motion envelope;
- service opening not backed by compiler service keep-out;
- unsupported Knitout revision and implicit machine capability;
- synthetic geometry promoted to topology preflight;
- out-of-scope semantic without explicit disposition.

## Benchmark programme

Report semantic-preservation and geometry metrics separately:
- 100% compiler-panel ↔ pattern-piece identity preservation;
- 100% seam/attachment edge-graph preservation on exact round trips;
- zero silently dropped required semantics;
- zero dangling/duplicate edge and connection references;
- zero false topology preflight from synthetic/stale geometry;
- seam path/edge-length mismatch per operation and per construction family;
- 2D↔3D correspondence residual by panel and local region, not only global mesh Chamfer distance;
- number and type of human semantic repairs needed per garment;
- compare against expert-authored manual topology on identical CAD/zone constraints.

## Research comparator

An open-access robotic sewing study published 21 September 2026 reports a single-arm, vision/force-controlled system using an unmodified industrial sewing machine. Across its tested configurations it reports seam deviation within ±2.5 mm, mean deviation ~1.2 mm, SD 0.9 mm, and 79% of seam points inside its desired offset tolerance. This is a sewing-execution comparator—not pattern-topology accuracy, garment fit or robot-cover durability. [Anatoliotakis et al. (2026)](https://doi.org/10.1007/s41315-026-00590-3).

Lightshell should target a stronger complete-thread benchmark: preserve the topology exactly, compile the path with explicit tolerances, execute using an independently qualified cell, and measure the as-sewn article against both pattern and robot-mounted geometry. We must not compare unlike metrics or call a software topology pass a seam-quality pass.

## Claim boundary

A topology pass is structural/source consistency only. It does not prove physical fit, drape, seam strength, retention, articulation durability, machine compatibility, certification or robot safety.