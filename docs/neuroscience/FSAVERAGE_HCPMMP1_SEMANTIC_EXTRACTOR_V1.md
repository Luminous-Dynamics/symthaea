# fsaverage HCP-MMP1 Semantic Extractor v1

Status: **synthetic parser/extraction qualification; real source bytes not yet qualified in-repo**

This profile defines the first controlled real-atlas lineage boundary for the
Symthaea NeuroBridge work.

The extractor converts a full FreeSurfer `fsaverage` HCP-MMP1 annotation into the
`10,242` semantic labels required by the deterministic `fsaverage5 -> Glasser360`
compiler introduced in PR #490.

It does **not** interpolate surfaces, infer parcel names, or accept arbitrary
annotation variants.

## Canonical path

```text
licensed/pinned fsaverage .annot bytes
        ↓
source hash verification
        ↓
strict FreeSurfer annotation decoding
        ↓
semantic HCP-MMP1 color-table names
        ↓
first 10,242 fsaverage vertices / hemisphere
        ↓
symthaea-semantic-surface-labels-v1
        ↓
PR #490 mapping compiler
```

## Why no `mri_surf2surf` is needed in this lineage

FreeSurfer's lower-resolution fsaverage meshes are nested icosahedral subsets of
high-resolution `fsaverage`. `fsaverage5` contains the first `10,242` vertices of
each `fsaverage` hemisphere.

This is independently documented by multiple neuroscience toolchains, including
Pycortex and MNE usage, and is used by ABCD surface data documentation.

Therefore the canonical v1 extraction rule is exact and inspection-friendly:

- input: exactly `163,842` fsaverage vertices per hemisphere;
- output: input vertices `0..10,241` in the same hemisphere;
- no nearest-neighbor interpolation;
- no spherical resampling;
- no reordering;
- no direct `fsaverage -> Symthaea12` shortcut.

A different surface or a file whose vertex count is not exactly `163,842` is
rejected.

## Lineage A: Mills / Figshare

`data/neuroscience/hcpmmp1_mills_figshare_fsaverage_v2.json` pins the candidate
first lineage:

- Kathryn Mills, **HCP-MMP1.0 projected on fsaverage**, Figshare version 2;
- DOI `10.6084/m9.figshare.3498446.v2`;
- left file URL `https://ndownloader.figshare.com/files/5528816`;
- left MD5 `46a102b59b2fb1bb4bd62d51bf02e975`;
- right file URL `https://ndownloader.figshare.com/files/5528819`;
- right MD5 `75e96b331940227bbcb07c1c791c2463`.

Those exact URLs and hashes are independently pinned by MNE-Python's HCP-MMP
fetcher. The reviewed MNE file currently has Git blob:

`a744af9c9ee9de83d997f17b0c12602b5be4854b`

The manifest is transport/provenance evidence. It does not by itself establish
atlas correctness.

## Terms boundary

The Figshare article advertises CC BY 4.0. MNE also explicitly states that use of
this parcellation is subject to the HCP-MMP terms on the HCP/BALSA page and asks
users to accept those terms before downloading.

For that reason this PR intentionally:

- does not vendor the `.annot` files;
- does not auto-download them;
- does not accept license/HCP terms in CI;
- records `acknowledgement_required = true`;
- requires the operator/user to obtain the files through an authorized process.

Once obtained, the extractor verifies their exact pinned MD5 values before any
semantic result is emitted.

## Annotation parser boundary

`scripts/extract_fsaverage_hcpmmp1_semantic_labels.py` implements the required
subset of the FreeSurfer `.annot` binary format using only Python's standard
library.

It supports the legacy positive-entry color table and version-2 color table,
and verifies:

- exact fsaverage vertex count;
- complete, unique vertex indices;
- color table presence;
- bounded string and color values;
- unique color-table structures where applicable;
- every annotation id resolves through the color table;
- no trailing uninterpreted bytes.

The implementation was independently checked locally against MNE's mature
annotation reader on synthetic v1 and v2 files: vertex annotation arrays and
semantic names agreed exactly.

MNE is **not** a runtime dependency of the canonical extractor.

## Semantic-name rules

For a left hemisphere artifact, every assigned atlas color-table name must be:

`L_<canonical-area>_ROI`

For a right hemisphere artifact:

`R_<canonical-area>_ROI`

The base area must occur in the pinned 180-area HCP-MMP1 namespace used by PR
#490.

The only name allowed to mean unassigned/medial wall is exactly:

`???`

It is converted to JSON `null`.

Names such as `unknown`, unexpected hemisphere prefixes, malformed `_ROI` names,
or unrecognized area names fail closed. This prevents unknown semantic content
from being laundered into medial-wall status.

## Output authority

Successful extraction emits `symthaea-semantic-surface-labels-v1` with:

- `space = fsaverage5`;
- `vertex_count = 10242`;
- canonical `L_<area>` / `R_<area>` values or `null`;
- SHA-256 of the exact source annotation bytes;
- source lineage id/version;
- extractor id/version;
- HCP terms reference.

Every one of the 180 HCP areas must have non-zero coverage inside the retained
`fsaverage5` subset. Empty areas are a hard error and are not repaired by
neighbor filling.

## Qualification status

The parser/extractor mechanism is synthetically qualified when its dedicated
contract suite passes.

That does **not** mean Lineage A is qualified until the actual licensed source
files are obtained, their MD5s verify, extraction succeeds, PR #490 compiles the
resulting semantic labels, and the evidence bundle is reviewed.

It also does not satisfy FMQ-010. Mills/Figshare-derived mirrors are the same
scientific lineage even when hosted in different repositories.

A separate derivation from controlled HCP/BALSA source and independently pinned
surface-registration inputs is still required for the PR #491 cross-check.

## Current synthetic gates

The tests require at least:

- FreeSurfer color-table v1 and v2 decoding;
- exact `163,842 -> 10,242` nested subset extraction;
- left/right semantic-name enforcement;
- exact `??? -> null` behavior;
- unknown names fail rather than become unassigned;
- all 180 areas survive the retained subset;
- source hash mismatch rejection;
- missing color ids rejection;
- duplicate vertex rejection;
- trailing-byte rejection;
- closed-world lineage manifest;
- wrong vertex-count rejection;
- deterministic output bytes.

## Non-goals

This extractor does not establish consciousness, cortical-mechanism identity, or
empirical human-neural alignment. It creates a reproducible semantic coordinate
artifact from one atlas lineage so that later representational analyses can be
performed without losing source authority.
