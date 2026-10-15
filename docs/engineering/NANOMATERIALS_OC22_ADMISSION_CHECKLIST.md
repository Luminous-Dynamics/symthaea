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
