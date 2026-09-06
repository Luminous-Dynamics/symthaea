# fsaverage5 → Glasser360 Map Compiler v1

Status: **compiler boundary implemented and synthetically qualified; no real atlas-derived mapping is qualified yet**

This document records the implemented compiler boundary for the broader
`FSAVERAGE5_GLASSER_TRANSFORM_QUALIFICATION_V1` profile.

The compiler does not make atlas bytes trustworthy. It takes a narrower input:
two already-decoded semantic-label vectors, one per hemisphere, and deterministically
compiles them into a provenance-bearing `FsAverage5GlasserMapV1` vertex→parcel
artifact. The future atlas extractor/derivation remains a separate qualification
boundary.

## Trust-boundary split

The intended pipeline is:

`controlled atlas source bytes`
`-> qualified extractor / resampling lineage`
`-> semantic fsaverage5 labels`
`-> deterministic compiler`
`-> FsAverage5GlasserMapV1`
`-> later surface-value aggregation`

The compiler consumes semantic labels, not raw `.annot` integers. This prevents
FreeSurfer color-table values or exporter-specific integer encodings from becoming
accidental parcel authority.

## Canonical semantic namespace

`data/neuroscience/hcp_mmp1_area_order_v1.json` contains exactly 180
hemisphere-neutral HCP-MMP1 base area names. Hemisphere determines whether a name
maps to parcel `1..180` or `181..360`.

The semantic namespace is pinned to both:

- upstream commit `e3a33a5a50d4ca86ab8fbaa0407d6c3296fcab12`
- exact XML Git blob `78a240b52845dd01c8676ecfabef59e2a0526a85`

The pinned XML is used only as the ordered semantic-name root. It is not itself a
qualified fsaverage5 spatial transform.

## Input contract

Each hemisphere input uses schema `symthaea-semantic-surface-labels-v1` and must
contain:

- `space = fsaverage5`
- exact hemisphere (`left` or `right`)
- exactly 10,242 vertex labels
- each label either the canonical hemisphere-prefixed HCP-MMP1 name or `null`
- immutable source identity/version/digest
- generator identity/version
- terms/reference metadata

Unknown fields, unknown area names, wrong hemisphere prefixes, malformed digests,
and wrong vector lengths fail closed.

`null` is explicit unassigned/medial-wall state. It is never converted to an
arbitrary parcel or zero-valued parcel observation.

## Output contract

The compiler emits `FsAverage5GlasserMapV1` with:

- 20,484 `vertex_to_parcel` entries
- canonical left-then-right hemisphere ordering
- parcel ids `1..180` for left and `181..360` for right
- explicit unassigned entries
- non-zero coverage for every one of 360 parcels
- source file and semantic-label digests
- canonical area-table digest and upstream source metadata
- per-parcel vertex census
- assigned/unassigned census per hemisphere
- zero cross-hemisphere violations
- fixed v1 aggregation policy metadata
- deterministic content digest

The v1 declared value-reduction policy is arithmetic mean, rejecting non-finite
values and empty parcels while excluding explicitly unassigned vertices. The
current compiler constructs and validates the label mapping; it does not yet
apply that aggregation to TRIBE activation values.

## Qualification status

The compiler currently has 14 synthetic contract tests covering:

- canonical 360-parcel mapping
- deterministic logical and byte output
- exact hemisphere vertex counts
- wrong-hemisphere label rejection
- unknown-area rejection
- complete parcel coverage
- explicit unassigned vertices
- artifact digest tampering
- semantic-label input tampering
- cross-hemisphere parcel leakage
- unknown input fields
- malformed nested source objects
- missing qualification fields

These tests qualify the deterministic compiler behavior against synthetic semantic
fixtures. They do **not** qualify a real HCP-MMP1→fsaverage5 derivation, a real
`.annot` extractor, or any checked-in atlas mapping artifact.

## What remains before first real mapping qualification

A real mapping remains blocked on:

1. controlled source atlas inputs and reviewed redistribution/data-use terms;
2. a pinned, reproducible HCP-MMP1→fsaverage/fsaverage5 derivation or extractor;
3. exact semantic-label source digests for both hemispheres;
4. generation of a real `FsAverage5GlasserMapV1` artifact;
5. per-parcel and medial-wall census review;
6. independent derivation comparison required by FMQ-010;
7. deterministic regeneration under the pinned toolchain;
8. review of the final evidence bundle.

Only after those gates pass may the mapping be used to de-quarantine a bounded
TRIBE `ExternalSurrogate` / `SurrogateAlignment` experiment.

## Non-authority invariant

Successful map compilation proves only that a declared semantic fsaverage5 label
assignment deterministically maps to the canonical Glasser360 namespace. It does
not establish that the input labels are scientifically correct, does not make a
TRIBE prediction empirical observation, and grants no consciousness or substrate
evidence authority.
