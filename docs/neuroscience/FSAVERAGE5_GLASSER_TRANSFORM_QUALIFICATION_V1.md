# fsaverage5 → Glasser360 Transform Qualification v1

Status: **design/qualification contract — no transform artifact qualified yet**

This profile defines the evidence and reproducibility requirements for reducing
TRIBE v2 cortical-surface output in FreeSurfer `fsaverage5` space to the HCP
MMP1.0 / Glasser 360-parcel atlas used by Symthaea.

It deliberately does **not** provide an approximate mapping. Until a mapping
artifact satisfies this profile, `fsaverage5` data must remain `fsaverage5`.

## Why this boundary exists

TRIBE v2 predicts on the cortical surface, not directly in Glasser parcels or
Symthaea's 12-region abstraction. A vector of the right length cannot be
reinterpreted as a parcel vector by truncation, index slicing, or relabeling.

The required path is:

`TRIBE fsaverage5 surface -> qualified Glasser360 parcellation -> optional qualified Symthaea12 projection`

Each arrow is independently versioned, digest-bound, and lossy where applicable.

## TRIBE surface contract

The released TRIBE v2 source defines `fsaverage5` as **10,242 vertices per
hemisphere**. Its surface handling treats the first hemisphere block as left and
the second as right.

Therefore the canonical input layout for this profile is:

- total vertices: `20_484`
- indices `0..10_242`: left hemisphere, local vertex `0..10_242`
- indices `10_242..20_484`: right hemisphere, local vertex `0..10_242`

Any artifact whose shape or ordering cannot prove this contract is rejected.

Reference source for the ordering contract:

- `facebookresearch/tribev2`
- reviewed source commit: `af58661791a351a448a489042a28f6c37e1c14b7`
- `tribev2/utils_fmri.py`
- `FmriTemplateSpace.FSAVERAGE_5 = (10242,)`
- 2-D surface handling slices left first and right second

The model revision used for an actual prediction must still be recorded
separately in neural-evidence provenance; this source commit only establishes the
reviewed layout rule used by this qualification profile.

## Glasser identity contract

Symthaea uses one canonical identity for HCP MMP1.0 / Glasser360:

- 180 cortical areas per hemisphere
- canonical parcel ids `1..180` for left hemisphere
- canonical parcel ids `181..360` for right hemisphere
- right parcel `180 + n` is the homologous right-hemisphere area for canonical
  left parcel `n`

Examples from the canonical ordered area table:

- `1 = L_V1`
- `2 = L_MST`
- `3 = L_V6`
- `4 = L_V2`
- `5 = L_V3`
- `6 = L_V4`
- `180 = L_p24`
- `181 = R_V1`

The existing `symthaea-core::hdc::glasser_parcellation::GlasserMapping` assumes
this canonical ordering for its 360→12 abstraction.

## Critical rule: map by semantic area identity, not raw annotation integer

FreeSurfer `.annot` files contain label/color-table values whose raw integer
representation must not be assumed to equal Symthaea's canonical parcel ids.
Different conversion/export pipelines may preserve names while changing raw
annotation values.

The qualified generator must therefore:

1. read each hemisphere annotation with a parser that exposes label names;
2. recover the area identity for every vertex;
3. normalize the hemisphere and HCP-MMP1 area name;
4. resolve that semantic name against an explicit canonical 180-area order;
5. assign Symthaea parcel `n` for left or `180 + n` for right;
6. represent medial-wall/unassigned vertices explicitly as unassigned, never as
   an arbitrary parcel.

Unknown, duplicate, or ambiguous area names fail closed.

## Source-data strategy

No third-party `fsaverage5` annotation should be copied into Symthaea merely
because it appears plausible.

A candidate qualification lineage should be reproducible from controlled source
inputs. Two useful lineages exist for cross-checking:

### Primary derivation candidate

1. HCP MMP1.0 parcellation from the Glasser/HCP BALSA study data in `fs_LR 32k`.
2. Reviewed HCP sphere-registration assets.
3. Resample labels to FreeSurfer `fsaverage` with Connectome Workbench using a
   label-preserving method.
4. Convert to FreeSurfer annotation form.
5. Downsample `fsaverage -> fsaverage5` with an explicitly pinned FreeSurfer
   version and label-preserving/nearest mapping.
6. Decode area names and emit the canonical vertex→parcel artifact described
   below.

BALSA data-use terms and redistribution conditions must be reviewed before any
source or derived atlas bytes are committed to the repository.

### Independent cross-check candidate

Kathryn Mills' published **HCP-MMP1.0 projected on fsaverage**, version 2,
provides a citable fsaverage projection under **CC BY 4.0** and documents its
projection procedure from HCP/BALSA source data.

A separately generated `fsaverage5` result from that lineage can be compared
against the primary derivation. Agreement is evidence about mapping
reproducibility; disagreement blocks qualification until resolved.

Reference:

- DOI: `10.6084/m9.figshare.3498446.v2`

A convenience repository containing pre-generated `fsaverage5` annotations may
be useful as a diagnostic third comparison, but it must not become the sole
provenance root for the canonical artifact.

## Canonical mapping artifact

The long-term checked artifact should encode meaning, not tool-specific `.annot`
bytes. Conceptually:

```text
FsAverage5GlasserMapV1
  schema_version
  input_space = fsaverage5
  output_space = glasser360
  vertices_per_hemisphere = 10242
  hemisphere_order = left_then_right
  canonical_area_table_version
  vertex_to_parcel[20484] -> optional parcel id 1..360
  source_inputs[]
    source_id
    source_version
    cryptographic_digest
    license_or_data_terms_reference
  generator_id
  generator_version
  generator_toolchain
  artifact_digest
```

The committed runtime representation may be binary/CBOR/JSON/Rust-generated, but
its logical contents and digest must be deterministic.

## Aggregation is part of the transform definition

A vertex→parcel label map alone does not define how surface values become parcel
values. The transform implementation must declare the statistic used for each
parcel, for example:

- arithmetic mean over assigned vertices, or
- surface-area-weighted mean using a separately qualified vertex-area artifact.

The first qualified profile should choose exactly one primary statistic rather
than silently switching between them. Alternative statistics may be evaluated as
sensitivity analyses under separate transform ids/versions.

NaN, infinity, empty parcels, and unassigned vertices require explicit policy.
They may not be silently converted to zero.

## Qualification invariants

### FMQ-001 — Exact input shape

Input has exactly 20,484 surface vertices per timepoint.

### FMQ-002 — Hemisphere ordering

Input is proven to be left 10,242 followed by right 10,242. The ordering root is
bound to the reviewed TRIBE surface contract/model artifact.

### FMQ-003 — Semantic label resolution

Vertex annotations are resolved through HCP-MMP1 area identity/name, not trusted
raw annotation integers.

### FMQ-004 — Canonical parcel order

Output uses one explicit 360-element order: left canonical 1..180 followed by
homologous right 181..360.

### FMQ-005 — No invented labels

Unknown/unassigned/medial-wall vertices remain explicitly unassigned. Unknown
area names fail qualification.

### FMQ-006 — Complete parcel coverage

Every one of the 360 canonical parcels has at least one assigned `fsaverage5`
vertex. Parcel vertex counts are emitted as qualification evidence.

### FMQ-007 — No cross-hemisphere leakage

A left surface vertex can only map to parcels 1..180; a right surface vertex can
only map to parcels 181..360.

### FMQ-008 — Deterministic artifact

Exact controlled inputs + exact generator/toolchain produce byte-identical
canonical mapping content or an explicitly canonicalized identical digest.

### FMQ-009 — Cryptographic provenance

Every source input and the generated mapping artifact carry approved 256-bit
cryptographic digests compatible with the neural-evidence provenance contract.

### FMQ-010 — Independent derivation check

Before first qualification, the canonical mapping is compared against at least
one independently sourced/generated HCP-MMP1→fsaverage lineage. Differences are
reported by hemisphere, parcel, and vertex count; unexplained disagreement is a
hard failure.

### FMQ-011 — Declared aggregation statistic

The surface→parcel reduction statistic and handling of missing/non-finite values
are versioned parts of the transform.

### FMQ-012 — Lossiness explicit

`fsaverage5 -> Glasser360` is a many-to-one lossy reduction. Downstream code and
reports must not imply invertibility or reconstruct the original surface from
parcel means.

### FMQ-013 — Mapping before Symthaea12

No direct `fsaverage5 -> Symthaea12` shortcut is qualified in v1. The evidence
chain must preserve the intermediate Glasser360 representation so each lossy
step can be inspected independently.

### FMQ-014 — No authority upgrade

A correctly mapped TRIBE prediction remains an `ExternalSurrogate`. Coordinate
qualification cannot turn it into `EmpiricalObserved` or consciousness evidence.

## Qualification evidence bundle

A first qualified artifact should emit at least:

- exact input source names/versions/digests,
- exact generator source revision,
- Workbench/FreeSurfer/parser versions where used,
- canonical 180-area name table + digest,
- final vertex→parcel artifact + digest,
- 20,484 input-position census,
- unassigned-vertex count per hemisphere,
- vertex count for every parcel,
- list of unknown/ambiguous labels (must be empty),
- cross-hemisphere violation count (must be zero),
- independent-lineage disagreement report,
- aggregation-method id/version,
- deterministic regeneration result.

## What this enables

Only after this profile is qualified should Symthaea implement the first bounded
external-surrogate experiment:

`TRIBE prediction (fsaverage5)`
`-> admitted ExternalSurrogate / SurrogateAlignment`
`-> qualified fsaverage5→Glasser360 transform`
`-> parcel-space similarity/RSA`

The later `Glasser360 -> Symthaea12` projection is a separate, more aggressive
semantic reduction and requires its own qualification. It should never be
smuggled into the surface-parcellation step.
