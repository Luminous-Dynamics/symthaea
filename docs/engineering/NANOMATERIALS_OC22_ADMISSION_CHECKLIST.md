# NANO-001A: OC22 IS2RE-Total Admission Checklist

**Status:** Source-level research complete; artifact admission **not yet complete**. The archive has not been downloaded, parsed, or independently checksum-verified in this workstream.  
**Tracking:** [Symthaea #7455](https://github.com/Luminous-Dynamics/symthaea/issues/7455)  
**Program:** [NANOMATERIALS_RESEARCH_PROGRAM.md](./NANOMATERIALS_RESEARCH_PROGRAM.md)  
**Representation/predictor suitability gate:** [NANO-001B #7460](https://github.com/Luminous-Dynamics/symthaea/issues/7460)  
**Prepared:** 2026-10-15

## 1. Candidate artifact record

| Field | Current research finding | Admission requirement |
|---|---|---|
| Dataset/task | Open Catalyst 2022 (OC22), IS2RE-Total | Confirm the artifact's internal schema and task metadata on download |
| Dataset paper scale | 62,331 DFT relaxations and approximately 9,854,504 single-point calculations (paper) | Do not confuse calculations/relaxations with independent candidate count; record actual row counts for this task's exact splits |
| Related-system metadata | mapping fields documented as bulk_id, bulk_symbols, miller_index, traj_id, slab_sid, ads_symbols, nads | Check field presence/coverage and use available lineage to audit cross-split related structures; unknown fields stay unknown |
| Target | DFT-derived relaxed total energy from initial structure | Verify units, finite values, and target semantics |
| Upstream source documentation | [FairChem OC22 docs at commit `57c71322d5a5be0a0fcdcf05592e1699b120dc79`](https://github.com/facebookresearch/fairchem/blob/57c71322d5a5be0a0fcdcf05592e1699b120dc79/docs/catalysts/datasets/oc22.md) | Preserve immutable source reference and access date in the run manifest |
| Direct archive | [is2res_total_train_val_test_lmdbs.tar.gz](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz) | Download to temporary research storage; do not commit or mirror |
| Expected compressed size | About 109 MB (upstream documentation) | Record exact byte count |
| Expected expanded size | About 424 MB (upstream documentation) | Record exact byte count and archive members |
| Upstream MD5 | `b35dc24e99ef3aeaee6c5c949903de94` | Verify as compatibility check only |
| Metadata map | [oc22_metadata.pkl](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl) | Admit only if required by analysis; record exact identity |
| Metadata map upstream MD5 | `13dc06c6510346d8a7f614d5b26c8ffa` | Verify if downloaded |
| Dataset terms | Upstream labels data as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Preserve attribution, citation, license text/URI, and modifications; confirm exact artifact scope |
| Loader/tooling terms | The dataset repo labels `ocdata` tooling MIT | Pin source commit, dependencies and license separately from dataset terms |
| Train/validation/test | Archive is documented as containing all three | Verify exact counts and split identifiers from archive metadata |
| ID/OOD | Official in-distribution and out-of-distribution subsplits documented | Preserve upstream definitions; do not invent semantics from folder names |
| SHA-256 | Not yet measured | Required after download; do not invent a digest |
| Actual local admission state | `not_downloaded` | Change only after recorded verification |

## 2. Safety rails for data handling

- Do not put OC22 raw records, LMDB files, trajectories, or the metadata pickle into this Git repository.
- Do not assume that the MIT license for the reader applies to the dataset. The dataset and code/tooling are separately licensed.
- Do not call the dataset admitted merely because the upstream page lists a checksum. The exact file must be retrieved, size-recorded, checksum-verified, and parsed under a pinned toolchain.
- MD5 is a legacy compatibility check, not a collision-resistant identity. Compute SHA-256 for every downloaded artifact and record the exact number of bytes.
- Keep the official test labels unavailable to the acquisition policy until evaluation. A single code path that can inspect test labels invalidates a claimed blind evaluation.
- If a dataset or split field is ambiguous, preserve it as unresolved; do not guess the unit, target meaning, or grouping.
- Any exported candidate or evaluation manifest must be reviewed for license and redistribution constraints before publication.

## 3. Required admission outputs

The admission run should emit an immutable, machine-readable manifest and a human-readable audit report. The manifest should include:

- manifest schema/version and exact Symthaea source commit;
- upstream docs source commit, retrieval timestamp, artifact URL, published license, dataset-paper citation and required attribution;
- local file name, exact byte count, SHA-256, upstream checksum algorithm/value and checksum outcome;
- tool/loader revision, dependency lock identity, invocation, OS/architecture, runtime/toolchain identity;
- archive member list, data/split identifiers, exact counts, declared units, label/structure alignment results;
- missing, non-finite, malformed, duplicate-ID and related-trajectory/group diagnostics;
- train/validation/test separation and official ID/OOD mapping results;
- exclusions and reasons, with no silent row drops;
- final status: admitted, or one of `data_access_unresolved`, `license_review_required`, `checksum_mismatch`, `schema_unresolved`, `split_integrity_failed`.

Do not place the full local filesystem path or machine-specific secrets in a public artifact manifest. A pseudonymous machine profile plus toolchain and resource details is sufficient.

## 4. Gate sequence

1. **Source freeze:** verify the pinned upstream documentation and dataset citation/license.
2. **Artifact identity:** download the exact archive and record byte count, upstream MD5 result, and SHA-256.
3. **Schema inspection:** list archive members before extraction/processing; validate IDs, field types, target units, structures and expected splits with a pinned reader.
4. **Independent audit:** use a checker separate from the data reader to recompute counts, split mapping, finite-label checks, duplicate IDs and structure/label alignment from exported records.
5. **Rights and disclosure:** review exactly what is planned for public output and ensure raw licensed data is not accidentally mirrored.
6. **Admission decision:** reviewer signs off on the manifest and report before any selection policy or predictive model is compared.

No benchmark accuracy or Symthaea value claim is allowed until this gate has a reviewed pass receipt. A passed data-integrity gate means the admitted artifact and its split metadata are understood; it does not mean the atomistic target is scientifically complete, the dataset is unbiased, or a learned model will generalize to physical chemistry.

## 5. Lineage and split-leakage metadata

The pinned FairChem OC22 documentation describes an optional mapping file whose per-system metadata can include `bulk_id`, `bulk_symbols`, `miller_index`, `traj_id`, `slab_sid`, `ads_symbols`, and `nads`. During admission, produce a field-presence and missingness report for these fields. Where present, use `traj_id`/`slab_sid` to detect related trajectory or surface records, and `bulk_id`/`miller_index`/adsorbate metadata to inspect whether nominally different system IDs share a source family across splits. This is a diagnostic, not permission to redefine the upstream official split. Any additional group holdout must be registered and reported separately. Do not assign synthetic lineage where the mapping is missing.

The OC22 paper reports 62,331 DFT relaxations and approximately 9,854,504 single-point calculations. These counts describe different scientific events and do not imply that the benchmark contains 9.85 million independent candidates. The exact usable record counts must come from the admitted IS2RE-Total archive and its split metadata, not be copied from the headline paper number.

## 6. Why total energy is only the first pipeline target

OC22 IS2RE-Total is a tractable starting point for verifying structural inputs, labels, splits, and label-acquisition evaluation. **Raw total energies across different compositions and system sizes must not be interpreted as a direct measure of catalyst quality.** The first benchmark is total-energy prediction under fixed label budgets only. A later benchmark using an adsorption-energy or reaction-specific target must be separately preregistered before claiming the system improves catalyst ranking or discovery.


## 7. Alternative hosted ColabFit representation (not admitted)

Admission research identified an OC22 collection on ColabFit Exchange / Hugging Face:
- [OC22 collection](https://huggingface.co/colabfit/collections)
- [IS2RE train](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train)
- [IS2RE validation in-domain](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Validation-in-domain)
- [IS2RE validation out-of-domain](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Validation-out-of-domain)

The visible cards say CC BY 4.0 and expose a ColabFit/Parquet representation with energies, forces, structures and metadata. The surfaced train dataset root revision is [fb892188938913025086056f6e29461fb49bcaf6](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train/commit/fb892188938913025086056f6e29461fb49bcaf6); the commit adding the large Parquet shards is [787bf4654488ae585cc98f19a74c3401706fa832](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train/commit/787bf4654488ae585cc98f19a74c3401706fa832). The card reports 7,861,269 unique molecular configurations, while the viewer estimates approximately 7.24 million rows / 30.6 GB. This discrepancy is not reconciled. The out-of-domain validation card reports about 521 thousand rows / 2.04 GB. These numbers differ materially from the official compact LMDB benchmark (~62K systems and a 109 MB compressed official IS2RE-Total/IS2RS archive).

Therefore this path is a **candidate alternate representation**, not a verified mirror or a drop-in replacement. The row-count difference could reflect configuration/trajectory-level records, but the exact interpretation must come from file metadata and source mapping. No files, rows, or labels were downloaded or independently verified as part of this source audit.

Before admitting any ColabFit data:
1. Freeze the exact dataset revision and every downloaded Parquet file's byte size and SHA-256; preserve uploader/processing provenance, license text and attribution.
2. Trace configurations back to source OC22 system identifiers (sid / lmdb-system-id) and establish which rows represent initial structures, relaxed targets, trajectory frames, or S2EF records.
3. Reconcile official train, validation-ID and validation-OOD membership; do not derive it from a label string alone.
4. Check whether an official test partition exists in this representation. The collection currently surfaced here lists train and two validation sets, not a test set; this is not evidence that hidden test labels are available.
5. Count configurations and unique systems separately and audit duplicates, trajectory-family leakage, units, missing values and structure/label alignment.
6. Where original access becomes available, compare a deterministic sample against records parsed by the pinned FairChem source reader. If exact task/split semantics cannot be established, keep this as a separate exploratory dataset with a separate contract and do not combine its scores with official OC22 results.

This discovery adds a possible retrieval/conversion route, but the canonical admission state remains not_downloaded and no benchmark is authorized.
