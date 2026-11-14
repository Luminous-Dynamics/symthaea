# Development-map federation v0.1

This directory contains **pinned references to independently owned project maps**. It is not a merged graph, event store, trust root, authority engine, source registry or deployment controller.

## Current members

The [federation manifest](federation-v0.1.json) pins four public project maps:

| Project | Map size | Subject snapshot scope |
|---|---:|---|
| Symthaea | 13 nodes / 18 edges | Root README + selected public issue/workstream observations |
| Mycelix | 17 nodes / 14 edges | Root README maturity projection |
| Symtropy | 20 nodes / 29 edges | README + architecture document |
| Sol Atlas | 41 nodes / 51 edges | DEMO, Cargo, selected Rust paths, decision-reliance, TUF and Earth-observability docs |

Each member separates two identities:

1. **Subject snapshot** — the exact source-repository revision and Git tree the map describes.
2. **Map artifact** — the exact branch commit and Git blob that contain the map itself.

These pins must not be conflated. Updating a map artifact does not, by itself, mean the source repository changed. Conversely, a source revision can change while the last published map remains pinned to the older revision.

## Shared contract

- `federation-v0.1.schema.json` defines the directory/relationship manifest.
- `../research/development-map-v0.1.schema.json` defines individual map snapshots.
- Both are provisional research contracts; selected structural checks are not equivalent to full Draft 2020-12 meta-validation or independent qualification.

## Federation rules

- A member map remains authoritative only within its owner project and declared source scope.
- Source-declared integration edges are not API compatibility receipts.
- Shared schema version strings are not conformance proofs.
- A contract join requires an exact owner-published contract identifier, version and blob digest, with two-sided evidence; independent qualification requires an independent conformance result.
- Authority, finance, governance, identity, sensor, physical actuation and deployment powers never transfer through a federation edge.
- Unknown, stale, proposed, contested, missing and unqualified claims remain distinct.
- A newer map artifact should not silently rewrite historical snapshots. Preserve the old artifact pin and mark the relevant projection stale when its subject source, extractor, schema or referenced owner contract changes.

## Refresh procedure

1. Fetch each repository's exact default-branch commit and Git tree OID.
2. Fetch the project map from its review branch and record the artifact commit, tree OID and Git blob OID.
3. Parse the map and run the structural validator on the exact fetched bytes.
4. Update only the relevant member entry and relationships; never copy its graph claims into another project's map.
5. Re-run manifest structural checks and independently confirm every pinned map path/blob/commit pair against GitHub.
6. Leave CI, compiler, runtime, physical, security and independent-adjudication states unchanged unless their own owner evidence has been obtained.

## Claim boundary

The manifest currently records bounded source projections and source-declared cross-project responsibilities. It does not establish that the four projects compile together, share compatible runtime contracts, pass end-to-end tests, or are jointly production-ready.
