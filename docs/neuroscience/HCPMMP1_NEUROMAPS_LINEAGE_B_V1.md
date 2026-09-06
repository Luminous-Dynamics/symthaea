# HCP-MMP1 neuromaps-method Lineage B v1

Status: **synthetic mechanism qualification only; real BALSA/HCP source bytes not yet qualified**

This profile defines a candidate second transform lineage for the Symthaea NeuroBridge `fsaverage5 -> Glasser360` qualification theorem. It does not establish atlas truth, scientific independence, empirical neural evidence, or authority to de-quarantine a benchmark.

## Implementation trust boundaries

The implementation is deliberately split into three small modules:

- `hcpmmp_neuromaps_common.py` validates the closed-world method/run schema, the exact WN56 source-pair identity, input hashes, canonical area namespace, and pinned Workbench executable/version;
- `hcpmmp_neuromaps_gifti.py` decodes label GIFTI payloads, preserves label-table identity, applies the explicit target medial-wall mask, and performs strict HCP-MMP1 semantic normalization;
- `derive_hcpmmp1_neuromaps_lineage_b.py` orchestrates the transform, constructs the scientific-input commitment, emits semantic artifacts, and creates a non-authorizing evidence report.

These are separate review boundaries: provenance, surface-label semantics, and transform/evidence orchestration are not one opaque parser.

## Why this lineage exists

Lineage A (#501) begins from the Mills/Figshare HCP-MMP1 annotations already projected to `fsaverage`. The Mills path uses HCP/BALSA source material, Workbench `BARYCENTRIC` resampling to high-resolution fsaverage, and FreeSurfer annotation conversion. Because `fsaverage5` is nested in `fsaverage`, Lineage A then retains vertices `0..10241` per hemisphere.

The public `aliamrod/HCP-MMP1_FSAvg` workflow substantially reproduces that BALSA -> Workbench `BARYCENTRIC` -> fsaverage route. It is useful as an independent execution replay, but not as a meaningfully distinct transform-method lineage.

Lineage B instead pins the neuromaps transform generation at commit:

`ffcc2e0f657943ce00a1b6a968396f32250e495c`

with:

- `transforms.py` blob `cddd03f2f2f6da94119732d57b4e4d0f1f1563bd`;
- atlas registry blob `f56eebc42375c11b18d4a2fca6c9ea151e1b50af`;
- atlas fetcher blob `82259057dbeb847795b2b461699300e0c51f3b55`;
- software license `CC-BY-NC-SA-4.0`;
- Markello et al. DOI `10.1038/s41592-022-01625-w`.

The relevant label path uses Workbench `ADAP_BARY_AREA`, average vertex-area correction, and a source ROI. Neuromaps defines fsaverage `10k` as 10,242 vertices per hemisphere.

## Exact HCP-MMP1 source pair

The v1 source root is the original BALSA/HCP Figure-3 context:

- study id: `RVVG` — *A Multi-modal Parcellation of Human Cerebral Cortex*;
- scene id: `WN56`;
- provider: `BALSA/Human Connectome Project`;
- source acquisition status: `operator_pinned_required`;
- automatic acquisition: forbidden.

The scene publishes the two hemisphere-specific CIFTI dense-label files used by this profile:

### Left

- BALSA file id: `npz0`;
- filename: `Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii`;
- run role: `hcp_left_dlabel`.

### Right

- BALSA file id: `pkN9`;
- filename: `Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii`;
- run role: `hcp_right_dlabel`.

These two files form **one required scene/study source pair**. They are not two independent scientific lineages. A run must acquire the authorized bytes, bind each file independently by SHA-256, and use both exact roles. A left file from one source context and a right file from another cannot satisfy v1 even if both are valid HCP-MMP1 files.

The metadata above identifies the required source objects; it is not a substitute for the source bytes or for authorization to acquire/use them.

## Closed-world v1 input roles

The method and run manifests must contain exactly fourteen scientific input roles:

- `hcp_left_dlabel`;
- `hcp_right_dlabel`;
- left/right fsLR32k non-medial-wall ROI;
- left/right fsLR32k sphere-to-fsaverage registration;
- left/right fsLR32k average vertex-area metric;
- left/right fsaverage10k sphere;
- left/right fsaverage10k average vertex-area metric;
- left/right fsaverage10k non-medial-wall mask.

No role may be omitted, renamed or added while retaining the v1 profile. A different source shape or transform dependency requires a new profile/version.

## Independence is multidimensional

The theorem intentionally compares two transformations of the **same HCP-MMP1 atlas**. Shared atlas ancestry is therefore required, not evidence of failure.

| Dimension | Lineage-B contract |
| --- | --- |
| Same HCP-MMP1 atlas root | required |
| Separate execution/acquisition | required, proven externally |
| Transform method distinct from Mills | yes (`ADAP_BARY_AREA` vs `BARYCENTRIC`) |
| Transform implementation family independent | no; both use Connectome Workbench |
| Semantic normalizer independent | no |
| Glasser compiler independent | no |
| Independence established by manifest/tool | never |
| External provenance review | required |

A different repository, filename, execution ID, or metadata string is not proof of independence.

Every generated evidence report retains:

`independence_established = false`

and:

`status = requires_external_provenance_review`

## Machine-readable method root

`data/neuroscience/hcpmmp1_neuromaps_transform_method_v1.json` pins:

- the exact `WN56` / `RVVG` / `npz0` / `pkN9` source-pair identity;
- source bytes as `operator_pinned_required`;
- no automatic source acquisition;
- exact neuromaps source generation and Git blobs;
- Workbench `ADAP_BARY_AREA`;
- average vertex-area correction;
- required source ROI;
- target-mask profile `symthaea-positive-label-mask-v1`;
- fsLR32k neuromaps bundle MD5 `7932b4418f63d28935b5adf67150b16f`;
- fsaverage10k neuromaps bundle MD5 `c61384c271ee2e6b5449222281137414`;
- the exact fourteen-role input set.

The upstream bundle MD5s are transport/provenance anchors. A real run still binds every extracted scientific file individually by SHA-256.

## No automatic source acquisition

The derivation contains no HTTP client and performs no download. A real run requires an operator-created closed-world run manifest containing:

- exact method-manifest SHA-256;
- execution identity;
- descriptive authorization/provenance reference;
- exact Workbench executable SHA-256;
- SHA-256 of exact `wb_command -version` output;
- exact path and SHA-256 for every v1 scientific input.

The program re-hashes those bytes before transformation. `authorization_reference` is descriptive provenance only; it is not a legal-entitlement or scientific-independence oracle.

## Controlled transform

For the left hemisphere, the transform separates `CORTEX_LEFT` from the pinned `hcp_left_dlabel`; for the right hemisphere it separates `CORTEX_RIGHT` from the pinned `hcp_right_dlabel`.

Each hemisphere then:

1. produces an fsLR32k label surface from its exact WN56 CIFTI source;
2. runs Workbench `-label-resample ... ADAP_BARY_AREA` directly to fsaverage10k;
3. binds the pinned source/target average vertex-area metrics;
4. binds the pinned source non-medial-wall ROI;
5. retains the Workbench label GIFTI and label table;
6. decodes the pinned fsaverage10k medial-wall surface separately;
7. applies explicit positive/non-positive target-mask semantics;
8. normalizes only canonical HCP-MMP1 names;
9. emits exactly 10,242 semantic labels.

The target-mask step is a Symthaea label-preserving hardening of the neuromaps method. This profile is therefore **neuromaps-method-derived**, not a claim of byte-identical neuromaps output.

## GIFTI and semantic boundary

The dependency-free decoder accepts only:

- exactly one `NIFTI_INTENT_LABEL` DataArray;
- INT32 labels;
- exactly 10,242 values;
- ASCII, Base64Binary or GZipBase64Binary;
- explicit little/big endian;
- an in-file label table;
- no external binary payload.

Semantic normalization is closed-world:

- `L_<known-area>_ROI -> L_<known-area>`;
- `R_<known-area>_ROI -> R_<known-area>`;
- exact `??? -> null`;
- all other labels fail.

All 180 areas must retain non-zero coverage in each hemisphere.

## Scientific input identity

The semantic `source_digest` commits to:

- method-manifest digest, which includes the exact WN56 source-pair identity;
- canonical 180-area order digest;
- exact Workbench executable digest;
- exact Workbench version-output digest;
- every role-specific source/template SHA-256.

It deliberately excludes execution ID, authorization-note text and the overall run-manifest digest. Those are retained separately as execution/provenance metadata.

Therefore:

`scientific input identity != execution/provenance metadata`

and a source-pair, semantic-namespace, transform-input, or tool change cannot preserve the same scientific identity.

## Evidence report

`derivation-evidence.json` records output hashes, root commitments and the non-authorizing independence state. Its own content digest detects corruption, but PR #525 adds the stronger archival theorem that a self-hash is not its own retained authority root.

## Synthetic qualification

The dependency-free suite covers:

- exact method/source/role authority boundary;
- authority-escalation rejection;
- ASCII/base64/compressed GIFTI decoding;
- external binary rejection;
- exact vertex-count enforcement;
- strict semantic/hemisphere rules;
- exact run-input role closure;
- pre-transform hash failure;
- deterministic complete fake-Workbench orchestration;
- explicit target-medial-wall masking;
- execution metadata excluded from scientific identity;
- area-order digest included in scientific identity;
- Workbench version binding;
- evidence-report authority tamper rejection.

Synthetic tests qualify only the mechanism.

## Real qualification sequence

1. Acquire the authorized WN56 `npz0` and `pkN9` bytes and retain acquisition evidence.
2. Record their exact SHA-256 roots in a closed-world run manifest.
3. Extract and SHA-256 pin the exact neuromaps fsLR32k/fsaverage10k template resources.
4. Pin Workbench bytes and version output.
5. Execute Lineage B and retain the evidence root outside the generated bundle.
6. Verify the archive with #525.
7. Compile Lineage B with #490.
8. Process Mills/Figshare Lineage A through #501 and #490.
9. Compare the two maps through #491.
10. Review every disagreement spatially and by parcel.
11. Establish execution/provenance independence outside the comparison program.
12. Only then decide whether FMQ-010 is satisfied.

No public neural benchmark should be de-quarantined before that evidence exists.
