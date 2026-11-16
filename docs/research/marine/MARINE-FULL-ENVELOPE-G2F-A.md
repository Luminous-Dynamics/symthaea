# MARINE-FULL-ENVELOPE-G2F-A
## Full external hull profile schema and semantic fixture

**Status:** schema/fixture contract v0.1.0  
**Parent issue:** [MARINE-FULL-ENVELOPE-GENERATION-001 #12434](https://github.com/Luminous-Dynamics/symthaea/issues/12434)  
**G2e dependency:** [MARINE-END-CLOSURE-001 #12385](https://github.com/Luminous-Dynamics/symthaea/issues/12385)  
**Schema:** [marine-full-envelope-spec.schema.json](./schemas/marine-full-envelope-spec.schema.json)  
**Synthetic known-answer input:** [marine-full-envelope-box-001.json](./fixtures/marine-full-envelope-box-001.json)

> This is an input-contract and synthetic-fixture increment. It does not generate or qualify a full external hull. No geometry or physical solver result is claimed.

## 1. Scope separation

G2a/G2b describes keel-to-design-waterline sections and closes the submerged displacement volume at z=0. G2e makes the underwater endpoint semantics explicit. Those representations remain valid for their declared purpose; neither contains the full topside/freeboard and deck envelope.

G2f introduces a **separate full-envelope input type**. It must not mutate the G2a/G2b artifact's meaning or allow a waterplane cap to be relabelled as a deck.

The full external envelope is a vessel-configuration object. A displacement shell is a derived, loading-condition-dependent control volume. One full-envelope revision can therefore generate many displacement shells for different drafts and, later, trim/heel conditions.

## 2. Coordinate and profile convention

The schema fixes metres and a right-handed local frame:
- x forward, with stations ordered from bow to stern;
- y starboard, with port represented by negative y;
- z down, with the declared design waterplane at z=0;
- positive z is below the design waterplane and negative z is above it.

Each station's starboard half-profile is ordered from keel toward the sheer/deck edge. Its points are named and carry a semantic role plus the region to the next point. Under `SYMMETRIC_ABOUT_Y0`, the port profile is derived by exact reflection and shares an explicit `MIRROR:<starboard-point-id>` identity. Under `EXPLICIT_PORT_AND_STARBOARD`, each station must supply a separate port profile; no symmetry is inferred.

Deck sections are separate from the side-shell half-profiles. They are ordered from port deck edge through camber/crown points to starboard deck edge. The deck surface is therefore not inferred by extending the underwater waterplane closure.

The design waterline is explicit and named. Its per-station intersection points must resolve to the profile's waterline points (or their declared mirrored equivalents) and must lie on the declared z=0 plane within a frozen tolerance. Mirrored point identities use the reserved `MIRROR:<starboard-point-id>` form; a mirrored reference never pretends that a separate port source point was supplied.

## 3. End conditions

Bow and stern are independent:
- `POINTED_STEM`: the end station collapses to the centreplane under the declared profile rule and shares port/starboard point identities; no planar end-cap fan is created.
- `TRANSOM`: the end is closed by an explicit transom surface and face-region identity.

The first slice only defines these semantic choices. It does not yet implement arbitrary bulbous stems, flared transoms, tunnel sterns, skegs, appendage cut-outs or superstructure intersections. Such forms must be typed unsupported/INCOMPLETE until dedicated profile and topology rules exist.

## 4. Surface inventory and boundary policy

Every admitted candidate must enumerate the regions it actually contains:
- bottom;
- topsides;
- deck;
- bow-end closure;
- stern-end closure;
- optional chine and keel breaklines.

The inventory is not an inferred feature list. The validator must reconcile it with actual station profile segments, deck sections and generated face-region assignments.

A closed external shell must have no unintended open boundary. Any opening or cut-out needs an explicit boundary loop with named point references and a supported disposition. Unknown openings, duplicate seams, missing deck sections, mismatched edge identities or a transom that does not meet the side/deck surfaces cause rejection or typed INCOMPLETE—not automatic filling.

Appendages and superstructure are versioned external components. They are not silently merged into the hull shell; their identity, source digest and assembly relation are recorded separately.

## 5. Runtime semantic validation beyond JSON Schema

JSON Schema checks shape and local field domains only. A future Rust validator must additionally check:

1. Station IDs are unique; x/L is strictly increasing, starts at 0 and ends at 1; principal dimensions are finite and positive.
2. Every station has exactly one supported starboard keel, waterline and sheer/deck-edge marker in the declared order; all profile points are finite and within dimensional envelopes.
3. Profile station point IDs are unique within their identity scope; point references resolve exactly once.
4. Symmetric mode omits explicit port coordinates and uses deterministic mirrored identities; explicit asymmetric mode requires a complete port profile with compatible semantic breaklines.
5. Mirrored profiles have equal point counts and exact reflected y coordinates; port/starboard identity is preserved at shared endpoints.
6. Design waterline point references resolve to the intended section points; z=0 and waterline continuity are checked without assuming a waterplane is a deck.
7. Deck section station IDs match the full station inventory; first/last deck points coincide with their corresponding port/starboard deck-edge points; deck crown/camber remains on the declared side of z=0.
8. Bow/stern end-condition labels agree with actual terminal profile topology; pointed stations cannot contain positive transverse breadth; transom surfaces must close with the station/deck perimeter.
9. Surface region inventory, edge-to-region declarations, component seams and boundary loops are complete, non-overlapping and bound in a manifest digest.
10. All source/profile/configuration, units, frame, dimensions, symmetry and end-condition fields are digest-bound. A stale digest or semantic change without a new manifest fails closed.

The G2f-a input is not itself a B-rep, triangle mesh or full external hull artifact. A separate output schema will be required for generated surface regions, face identity, topology and geometry digests.

## 6. Synthetic box fixture

`marine-full-envelope-box-001.json` is an intentionally simple symmetric box hull:
- length 100 m, beam 20 m, draft 6 m, freeboard 4 m;
- planar/transom bow and stern;
- rectangular underwater body with vertical topsides;
- flat deck section with a small declared crown/camber;
- two longitudinal stations with exact repeated profile control points;
- one named design waterline at z=0;
- explicit bottom, topsides, deck and both end-closure regions;
- no appendages or openings.

The fixture is an input contract test, not a produced mesh. Its dimensions provide an exact future geometry/hydrostatic known-answer target after the generator exists: external box envelope volume 20,000 m³ before accounting for the declared deck crown; the submerged rectangular prism below z=0 has 12,000 m³ displacement. The crown makes the full-envelope top not a simple rectangular box, so external-envelope volume must be computed from the explicit deck profile rather than reusing the 20,000 m³ rectangular-box bound.

## 7. Next implementation increments

- **G2f-a1:** add Rust typed spec and cross-field validator; malformed-reference, asymmetry, waterline, deck-edge and end-condition negative controls.
- **G2f-b:** loft bottom/topside surfaces with shared breakline identities and explicit port/starboard policy.
- **G2f-c:** generate deck, bow and transom surfaces with a complete face-region inventory.
- **G2f-d:** intersect the external shell with declared waterplanes and emit a separate loading-specific displacement artifact.
- **G2f-e:** qualify full-envelope and derived displacement artifacts independently through AP242 export/import and validation-property preservation.

## 8. Benchmark targets

Research targets, not achieved results:
- 100% surface inventory and boundary-loop reconciliation for every artifact admitted as a full external hull;
- zero silent symmetry inference for explicitly asymmetric inputs;
- zero unintended open/non-manifold edges for candidates declared closed;
- zero duplicate vertices at declared shared seams;
- exact agreement with box/prism geometry and displacement fixtures within a frozen tolerance profile;
- stable displacement/waterplane curves across a predeclared range of horizontal waterplanes;
- identical full-envelope geometry digest across distinct loading conditions, with distinct derived displacement-shell digests;
- preserved persistent IDs, surface-region semantics and validation properties after independent AP242 round-trip.

## 9. Claim ceiling

Schema validation proves only that an input record conforms to the schema. Semantic validation proves only the declared profile contract. Neither proves a generated hull is geometrically valid, fair, watertight, structurally adequate, hydrostatically stable, manufacturable, class-approved, seaworthy or safe to operate. Those claims require separate geometry, numerical, manufacturing, physical-test and competent-authority evidence.
