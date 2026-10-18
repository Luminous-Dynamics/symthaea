# CropFoundry dataset discovery inventory v0

**Snapshot date:** 2026-10-18  
**Status:** discovery shortlist only; no external dataset files have been downloaded, parsed, normalized, or scored for this PR.  
**Purpose:** make the first data-ingestion decision auditable, rights-aware, and matched to a real benchmark question.

**Machine-readable registry:** [data-source-manifest-v0.json](./data-source-manifest-v0.json). The Rust validator at `cargo run -p xtask -- cropfoundry-validate` checks that these candidates remain non-trainable unless artifact-specific rights, access authorization, an exact pinned digest, intended use, and schema review are recorded.

## Decision

Do not start by bulk-ingesting every crop database. Prefer two sharply separated, small proofs with explicit license evidence:

1. **First tabular pipeline candidate:** [UdeA Mobile and Manual Tomato](https://zenodo.org/records/16745911). Zenodo states CC BY 4.0 and describes 135 plants, 27 treatment combinations and 20 weekly sessions. The accessible CSV preview exposes row-level plant/time identifiers, treatment labels, mixed sensor/manual variables and yield fields. Do not randomly split rows: hold out whole plants for one question and later dates for a separate temporal question. The related GitHub mirror says 2,664 rows × 44 columns and its README says CC BY 4.0, but the mirror's LICENSE file is CC0 1.0. The canonical Zenodo v1 record says 2,666 rows × 40 columns, identifies CC BY 4.0, and lists the CSV's MD5. Do not merge or substitute the mirror variant; resolve the exact row-count semantics and acquire only the canonical Zenodo artifact. The Zenodo record references a separate data dictionary/processing document, but that document is not included among the record's files, so the missingness procedure, units and treatment assignment details remain an evidence gap.
2. **Separate phenotyping/vision candidate:** [AerialYield-T2M](https://zenodo.org/records/22071809), a six-stage tomato-ripeness/mask/count dataset with 677 frames, 14,303 annotated tomato instances, explicit block-grouped splits, a checksum list and a bundled CC BY 4.0 LICENSE file. It is a single commercial greenhouse in the UAE, so it cannot establish yield, genotype, or South African transfer.
3. The **existing Mycelix lettuce/DWC profile and cultivar-admission work** as an integration/contract lane, not as measured yield evidence.
4. An **ARC partnership/access request** for one locally relevant underutilized leafy vegetable rather than assuming its trial records are openly downloadable.
5. A **germplasm discovery** lane through Genesys or relevant national/institutional collections, subject to access terms and material-transfer conditions.

The UdeA file preview has been inspected for column headings and example records through Zenodo's web preview, but no data bytes have been pinned locally, normalized, or scored. Its treatment labels are not proof of randomization or blocking, so CF-018 must remain unevaluable until the design is documented. These non-South-African studies are pipeline/benchmark candidates, not evidence that varieties or models transfer to Johannesburg.

## Candidate source register

| ID | Source and known contents | Potential role | Access / license status as checked | Decision |
|---|---|---|---|---|
| DS-001 | [LiveSeeding tomato breeding-line trials (Portugal, 2023), Zenodo 21789988](https://zenodo.org/records/21789988). Record describes agronomic, phenological, fruit morphology, yield-related, sensory and quality evaluations; XLSX is listed. | First structured-trial candidate; test schema extraction, trait harmonization, missingness, and trial grouping. | The rendered Zenodo record was marked Open, but its visible License field was blank. Inspect the exact record metadata and file terms before reuse or derived redistribution. | **P0 discovery; quarantine until rights confirmed.** |
| DS-002 | [LiveSeeding tomato shuttle-breeding trials (Portugal, 2024–2025), Zenodo 22008471](https://zenodo.org/records/22008471). Record describes agronomic, morphological, yield and sensory characteristics; XLSX is listed. | Complementary season/management dataset; possible independent study if definitions and experimental populations align. | The rendered record was marked Open, but its visible License field was blank. Exact rights and related-study dependencies need verification. | **P0 discovery; quarantine until rights confirmed.** |
| DS-003 | [Horti-M3-Tomato, DOI 10.5281/zenodo.17217565](https://doi.org/10.5281/zenodo.17217565). Paper describes three seasons (2023–2025), RGB images, sensor logs, soil/environment data, agronomic records and yield. It identifies a greenhouse in Northeast China with one integrated monitoring station. | Multimodal sensor/image alignment and within-site temporal evaluation, if permitted. It is **not** independent multi-location replication and cannot validate transfer to Gauteng. | The paper says the data are openly available for research/education; verify the exact dataset/artifact terms, subfile terms and any third-party content before training, derivative work, or redistribution. | **P0 discovery; legal/terms review required.** |
| DS-004 | [TomatoMAP: multi-angle/multi-pose tomato phenotyping](https://www.nature.com/articles/s41597-026-06926-9), *Scientific Data* (2026). The article describes a fine-grained tomato imaging/phenotyping dataset. | Image-only phenotyping tasks such as plant/fruit localization, segmentation, pose and annotation-quality evaluation—not yield prediction unless linked observations support it. | Article is described as CC BY 4.0; verify data availability, supplementary files, and any third-party image exceptions at the point of use. | **P1 after artifact/license verification.** |
| DS-005 | [ARC Indigenous/Traditional African Leafy Vegetables program](https://arc.agric.za/arc-vopi/Pages/Crop%20Science/Indigenous-Crops.aspx). Research scope includes *Amaranthus*, *Cleome gynandra*, *Corchorus*, *Solanum retroflexum*, *Vigna* and others; its overview discusses water/fertilizer needs, planting/harvest methods, nutrition and field/pot trials. | Local species/cultivation knowledge; identify candidate partners, variables and evidence gaps for an underutilized crop. | This is a program/research page, not a published downloadable trial dataset. Do not scrape researcher/contact data or imply records are open; pursue the official contact/partner path and agreements. | **P0 collaboration/access discovery.** |
| DS-006 | [Genesys PGR](https://www.genesys-pgr.org/) and its [API guidance](https://docs.genesys-pgr.org/docs/api/). The July 2026 API guide describes accession passport and trait-data query workflows and a token-based API. | Discover relevant accessions, origins, descriptors and available trait datasets; not automatically a usable yield dataset. | API access requires an authorized token; terms, attribution, institution-level permissions and material-transfer conditions apply. | **P1 germplasm discovery after access/terms review.** |
| DS-007 | [FAOSTAT](https://www.fao.org/faostat/). Country/region-level production, prices, inputs, trade and food-security statistics. | Context variables, plausible production/price baselines and food-system prioritization. | Public statistical portal; capture dataset vintage, series definition, geography, units and terms. Aggregated time series are not cultivar-level trial evidence. | **P0 contextual baseline only.** |
| DS-008 | [Plant Phenotyping Datasets](https://www.plant-phenotyping.org/datasets-home). Existing image/annotation benchmark resources for detection, localization, segmentation, leaf counting and related CV tasks. | Reproducible computer-vision baseline before new imaging models. | Each dataset's own license, annotation protocol and train/test conventions must be checked separately. | **P1 for CV work only.** |
| DS-009 | [Mycelix #3303 ProductiveLoopV1](https://github.com/Luminous-Dynamics/mycelix/issues/3303), [#3309 lettuce/DWC profile](https://github.com/Luminous-Dynamics/mycelix/issues/3309), [#3311 cultivar-admission profile](https://github.com/Luminous-Dynamics/mycelix/issues/3311), and [#3317 admission validator](https://github.com/Luminous-Dynamics/mycelix/issues/3317). | Reuse existing output/lot/cultivar semantics and negative controls for controlled-environment integration. | These are project design/implementation work items, not third-party biological validation datasets. Their references must keep claim ceilings intact. | **P0 integration; do not count as biological observations.** |
| DS-010 | [UdeA Mobile and Manual Dataset for Greenhouse Tomato Crop](https://zenodo.org/records/16745911). Zenodo's canonical CSV is described as 2,666 rows × 40 columns; 135 plants, 27 treatment combinations and 20 weekly sessions from November 2023–April 2024. The CSV preview reveals plant, date, treatment, sensor/manual variables and yield-related fields. | First tabular-contract and leakage-audit pilot; explore prediction feasibility only after direct artifact inspection. | The Zenodo record states CC BY 4.0. Its GitHub mirror README reports 2,664 rows × 44 columns, so the mirror and canonical artifact must not be silently mixed. The canonical file is not locally pinned or scored. | **P0 schema / split audit; no random-row split and no South African transfer claim.** |
| DS-011 | [AerialYield-T2M](https://zenodo.org/records/22071809). 677 RGB frames (535 close-range, 142 drone), 14,303 tomato instances, six ripeness stages, masks, count labels, split metadata and a SHA-256 checksum list. | Separate visual-phenotyping benchmark: ripeness classification, instance detection/segmentation and fruit counting. It is not a crop-yield or genetics dataset. | Public Zenodo release includes a LICENSE file explicitly stating CC BY 4.0. The 3.6 GB artifacts and checksum file have not been downloaded or verified locally. One commercial greenhouse in the UAE limits external validity. | **P1 after minimal artifact/split/checksum audit.** |

## Local-only CSV audit command

After the canonical DS-010 artifact is legitimately acquired to a local path, run the **schema audit only**:

```bash
cargo run -q -p xtask -- cropfoundry-audit-csv \
  --input /path/to/DB_Mobile_Manual_Tomato.csv \
  --source-id DS-010 \
  --source-uri https://zenodo.org/records/16745911 \
  --delimiter ';' \
  --out reports/cropfoundry/udea-csv-audit.json
```

If an independently trusted file-level checksum is available, supply it with `--expected-sha256 <64-hex-digest>`; the audit will fail when it does not match. A digest computed from the file itself is useful for reproducibility but does not prove that the file came from the claimed source. The current Zenodo page lists an MD5 digest rather than a SHA-256 for this CSV, so do not invent an expected SHA-256 or call the artifact identity verified merely because a local digest was produced.

```bash
# Optional only when a trustworthy SHA-256 exists for the exact artifact:
cargo run -q -p xtask -- cropfoundry-audit-csv \
  --input /path/to/DB_Mobile_Manual_Tomato.csv \
  --expected-sha256 <64-hex-digest>

```

Do not pass `--expected-data-records` until the canonical file's exact record-count semantics have been reconciled. The command binds to DS-010's canonical record URI and computes a SHA-256 digest over the exact local bytes. It reports header duplication, missing required fields, row-width anomalies, record count, per-column blank/zero/numeric counts, unique/missing Plant/Date/Treatment values, duplicate Plant-Date keys and unitless measurement headers. It writes no normalized training data and always reports `training_authorized: false`.

The audit is intentionally descriptive rather than a source-backed adjudication. A unitless weight field is a warning, not permission to infer grams or kilograms. A zero harvested weight may be a valid early-season observation, not missingness. An audit receipt does not validate license terms, agronomic quality, a model split, or crop performance.

The CSV auditor has deterministic synthetic unit tests for semicolon parsing, exact digest shape, repeated Plant-Date detection, malformed-width records, duplicate/missing columns, blank versus zero counts, unit warnings, count mismatch, empty input and source-identity mislabelling. Their presence is not a claim that the tests have passed on the current PR head.

## First ingestion gate

Before a file enters a training or evaluation dataset, record:

- stable source URI/identifier, version, retrieval timestamp, content digest, and file inventory;
- license text and intended-use decision, including any no-derivative/non-commercial or third-party-material limits;
- dataset authorship/citation and relevant access/material-transfer terms;
- experiment ID, biological material/cultivar/accession, site, season, management, replication, randomization/block structure where recorded;
- trait identifiers mapped to Crop Ontology or a documented local term, including method, scale, unit and missingness;
- train/test reuse restrictions and any published benchmark split rules;
- provenance of normalization, conversion, exclusion and derived fields.

**Fail closed:** unresolved rights mean quarantine; missing genotype identity means no genotype-level claim; incompatible trait definitions mean no naïve pooling; absent site-year context means no site-year transfer claim. A public landing page alone is not permission to copy or train on every artifact linked from it.

## Recommended initial work order

1. Use the canonical DS-010 Zenodo artifact as the first small tabular pipeline candidate: verify exact file identity, retrieve timestamp, local SHA-256 snapshot, headers/units, plant/date/treatment grouping, missingness and the row/column discrepancy before scoring. Zenodo lists an MD5 for the CSV, not SHA-256; a locally calculated SHA-256 is a reproducibility pin, not independent proof of source authenticity.
2. Design two separate leakage-resistant questions: unseen-plant generalization and future-date prediction. Never use random-row split as the only evaluation because repeated measures of the same plants can leak across partitions.
3. Start with descriptive QA and simple baselines only if the chosen target has enough non-missing observations and a valid group split. Report dataset coverage before any prediction result; if groups are too few, keep the result at schema-validation level.
4. Keep DS-011 as a separate image task: start with metadata, bundled license, checksum file and split-integrity review; only then download the minimum authorized components needed for a baseline.
5. Keep DS-001, DS-002 and DS-003 quarantined until their individual artifact rights/access terms have been resolved.
6. Open an ARC collaboration/access conversation for the indigenous-crop lane; choose the crop based on local relevance and real evidence/seed access, not on a preconceived model target.
7. Before treatment recommendations, separately verify randomization/allocation, experimental units, blocking, replication, treatment definitions, and source-backed outcomes. If those design details are unavailable, keep the task descriptive/predictive and do not claim causal treatment effects.
8. Freeze traits, baselines, splits, metrics and a run receipt only after the applicable inventory gate passes.

## Source notes

- The two LiveSeeding records were published in August 2026 and describe small, structured tomato trial workbooks. The record pages do not show a clearly filled license in the rendered metadata; this inventory intentionally does not assume reuse permission.
- The Horti-M3-Tomato paper describes multi-season, multimodal coverage, but article license and data-package license must be distinguished.
- ARC's program page documents an active research remit and local crop species; the existence of a research program is not proof of a public raw dataset.
- Genesys, FAOSTAT and the phenotype dataset portal serve different purposes. Germplasm discovery, macro-level agricultural context, imaging annotations and replicated field-trial outcomes must not be conflated.
