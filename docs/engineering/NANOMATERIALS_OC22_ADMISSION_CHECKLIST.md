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

The admission run should emit an immutable, machine-readable manifest and a human-readable audit report. Use the repository contract in [NANOMATERIALS_OC22_PAIR_MANIFEST.schema.json](./NANOMATERIALS_OC22_PAIR_MANIFEST.schema.json) for the source artifacts, split inventory, per-SID input/target pairs, independent checks, and explicit unresolved/excluded states. This is a document-level evidence contract, not a production runtime type and not proof that any manifest values are true. The manifest should include:

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

The visible cards say CC BY 4.0 and expose a ColabFit/Parquet representation with energies, forces, structures and metadata. The surfaced train dataset root revision is [fb892188938913025086056f6e29461fb49bcaf6](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train/commit/fb892188938913025086056f6e29461fb49bcaf6); the commit adding the large Parquet shards is [787bf4654488ae585cc98f19a74c3401706fa832](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train/commit/787bf4654488ae585cc98f19a74c3401706fa832). The train card reports 7,861,269 unique molecular configurations, while the auto-converted viewer estimates 7,236,212 rows (showing the first 718,431) and 30.6 GB across its Parquet representation. The out-of-domain validation card reports 521,827 unique configurations and the viewer estimates 521,308 rows; its auto-converted Parquet representation is listed at 2.44 GB, while the source ColabFit file tree reports 2.04 GB for the source data folder. These 2.04 GB and 2.44 GB values refer to different representations, not a verified disagreement in the byte size of one identical file. The card/viewer count differences remain unresolved. All counts/footprints differ substantially from the official compact LMDB benchmark (~62K oxide systems and a 109 MB compressed official IS2RE-Total/IS2RS archive). Visible rows carry both IS2RE and S2EF dataset membership labels. Because the hosted rows include individual configurations with per-configuration energy/forces, their identity as official initial-structure to relaxed-total-energy examples is not established; treat the hosted representation as a potential configuration/trajectory-level dataset until its row-level target semantics are independently proven.

Therefore this path is a **candidate alternate representation**, not a verified mirror or a drop-in replacement. The row-count difference could reflect configuration/trajectory-level records, but the exact interpretation must come from file metadata and source mapping. No files, rows, or labels were downloaded or independently verified as part of this source audit.

Before admitting any ColabFit data:
1. Freeze the exact dataset revision and every downloaded Parquet file's byte size and SHA-256; preserve uploader/processing provenance, license text and attribution.
2. Trace configurations back to source OC22 system identifiers (sid / lmdb-system-id) and establish which rows represent initial structures, relaxed targets, trajectory frames, or S2EF records.
3. Reconcile official train, validation-ID and validation-OOD membership; do not derive it from a label string alone.
4. Check whether an official test partition exists in this representation. The collection currently surfaced here lists train and two validation sets, not a test set; this is not evidence that hidden test labels are available.
5. Count configurations and unique systems separately and audit duplicates, trajectory-family leakage, units, missing values and structure/label alignment.
6. Where original access becomes available, compare a deterministic sample against records parsed by the pinned FairChem source reader. If exact task/split semantics cannot be established, keep this as a separate exploratory dataset with a separate contract and do not combine its scores with official OC22 results.

This discovery adds a possible retrieval/conversion route, but the canonical admission state remains not_downloaded and no benchmark is authorized.


## 8. ColabFit schema/lineage audit findings (source-level only)

The public Hugging Face Viewer exposes ColabFit fields `configuration_metadata`, `configuration_metadata_id`, `configuration_labels`, `configuration_dataset_ids`, `configuration_id`, `configuration_hash`, and `structure_hash`. The pinned IS2RE train dataset page identifies its aggregate `ds.parquet` file as 18.5 kB and provides SHA-256 `843156208bc225e1c7c77690c36975b45c6f0f85cf63600b7ca1dff850b9b8ef` plus Xet hash `5bcd29fdf37dc0809be1f8dbc50e6ce49cf9cbec51dc2ffe0784bbf22cb0aa2c` at revision `fb892188938913025086056f6e29461fb49bcaf6`. These are provider-reported metadata values, not a locally downloaded or independently measured hash of the large Parquet configuration shards; the small aggregate metadata file must not be mistaken for the actual data. The OOD validation `ds.parquet` metadata was updated at revision [`bac2ddf46c6b7399a8a796026fe711094c8d7027`](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Validation-out-of-domain/commit/bac2ddf46c6b7399a8a796026fe711094c8d7027), with Git-LFS pointer SHA-256 `3621b4c1261e81148afb280a66a89fd47327838e9e275254605eb4eb8c934faf` and size 18,677 bytes. This remains metadata-only provenance, not a measured hash of the multi-gigabyte source or auto-converted data. Visible OOD validation examples have `lmdb-system-id`, `traj-id`, and `num-adsorbates` in their configuration metadata. Some records carry both an `is2re_val_ood_*` label and an `s2ef_val_ood_*` label, with dataset IDs for both task collections. This demonstrates task-collection overlap for at least some configurations; do not count matching cross-task entries as independent systems if datasets are combined. The ColabFit OC22 family page lists six dataset members and 18.0M configurations in total; its IS2RE-train member reports 7,861,269 and S2EF-train reports 8,356,688. These are configuration-level collection counts, not 18 million distinct source systems. Count dataset memberships, configurations, source SIDs, and distinct structure hashes separately.

Observed source metadata, download representation and viewer estimates are different layers:
- IS2RE train: ColabFit Exchange catalog and Hugging Face card report 7,861,269 configurations (DOI `10.60732/722bcab6`); the Hugging Face auto-converted viewer is explicitly limited to the first 5 GB, shows 718,431 rows and estimates 7,236,212 rows total. Treat the latter as an estimate from the partial preview, not the source-level unique-configuration count.
- IS2RE train download choices on ColabFit: original files ~9.4 GB; Parquet ~30.6 GB; XYZ ~22.7 GB.
- IS2RE validation-OOD: ColabFit family catalog reports 520,744 configurations (DOI `10.60732/15fa94f2`), Hugging Face card reports 521,827, and its partial auto-converted viewer estimates 521,308 rows. The aggregate count differences remain unresolved.
- Validation-OOD download choices on ColabFit: original files ~2.04 GB; auto-converted Parquet ~2.44 GB. These are different representations, not verified conflicting sizes for the same file.

Pin exact revisions and file/shard identities, download the chosen representation, then independently calculate actual hashes and row/unique-configuration counts. Preserve card, catalog, and viewer values as separate observations; do not copy one provider estimate into the admission manifest as ground truth.

These values are not yet an admitted count/size manifest. Preserve each as a source observation, pin the exact dataset revision and file/shard identities, and compute counts from the actual frozen files. Never silently choose one aggregate value or use it as a training/evaluation denominator.

The audit must reconcile:
1. `lmdb-system-id` against OC22 `sid` on source-matched rows;
2. `traj-id`, source metadata and structure/configuration hashes for duplicated or related structures;
3. configuration labels and dataset IDs across IS2RE train, validation-ID, validation-OOD and parallel S2EF collections;
4. target property, units, input structure role and whether each row is the official initial-to-relaxed-energy example;
5. the presence of official test membership. The currently surfaced ColabFit collection lists six IS2RE/S2EF train/validation datasets and no IS2RE test collection. Until a rights-reviewed independent test set is defined and available, the ColabFit route cannot replace the official combined benchmark for the registered primary evaluation.

Sources:
- [ColabFit OC22 collection](https://huggingface.co/collections/colabfit/oc22-open-catalyst-2022-colabfit)
- [IS2RE train card and viewer](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train)
- [IS2RE OOD validation card and schema/examples](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Validation-out-of-domain)
- [FairChem OC22 source contract](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc22.md)

No Parquet data bytes or row-level crosswalk have been independently verified. These findings are source-audit observations only; admission state remains `not_downloaded`.


## 9. Target-pair semantics: a ColabFit configuration is not automatically an IS2RE example

The ColabFit schema defines every row in a configuration Parquet file as one structure plus its associated calculated properties. Its documented fields include structure arrays (`positions`, `atomic_numbers`, `cell`, `pbc`) and energy/force properties (`energy` in eV and `atomic_forces` in eV/Å). References: [ColabFit configuration schema](https://materials.colabfit.org/docs/configuration_schema) and [Parquet usage guide](https://materials.colabfit.org/docs/how_to_use_parquet).

The official FairChem OC22 task definition states that IS2RE-Total/IS2RS LMDB records contain initial input structures and output relaxed structures/energies. A ColabFit row with one structure and its energy/forces therefore does not alone establish the official IS2RE input-target pair. The visible inclusion of some records in both IS2RE and S2EF collections does not prove which configuration is the initial input or the relaxed endpoint.

Before treating ColabFit as an IS2RE dataset, require a pair-level conversion manifest with:
- OC22 `sid` / ColabFit `lmdb-system-id` crosswalk and provenance for every source record;
- explicit evidence selecting the initial structure and the relaxed endpoint/energy for each SID;
- no reliance on file order, label text, modified timestamps, or hash ordering to infer trajectory position;
- structure integrity comparisons over elements/atomic numbers, atom count/order, coordinates, cell, periodic-boundary flags and units;
- finite target-energy validation and explicit missing/ambiguous/multiple-endpoint dispositions;
- exact pair counts reconciled to the official train, validation-ID, validation-OOD and test membership where available;
- deterministic sample-level validation against the pinned FairChem loader, followed by a complete independent comparison.

If a unique input/endpoint pair cannot be established, retain ColabFit as a separate single-configuration energy/force collection; do not use it to claim IS2RE label efficiency. Its surfaced collection still lists train and validation sets only, so the absence of a separately identified official test partition remains a separate holdout blocker.

This is a data-contract gate, not an assertion that a correct conversion is impossible. No ColabFit pair manifest has been generated and no data rows have been independently validated.
