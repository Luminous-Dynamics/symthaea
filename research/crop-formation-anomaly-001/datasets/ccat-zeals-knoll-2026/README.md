# CCAT Zeals Knoll 2026 geometry fixture

This directory contains an attributed, unchanged copy of the 60-row circle-centre/radius CSV published with CCAT v1.0.0.

## Source and rights

- Author: Salah-Eddin Gherbi
- Project: [CCAT v1.0.0](https://github.com/salahealer9/ccat-crop-circle-analysis-toolkit)
- Source file: [data/circles_drag_60.csv](https://github.com/salahealer9/ccat-crop-circle-analysis-toolkit/blob/main/data/circles_drag_60.csv)
- Version DOI: [10.5281/zenodo.21527026](https://doi.org/10.5281/zenodo.21527026)
- Dataset licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

**Attribution:** Salah-Eddin Gherbi, *CCAT v1.0.0: Geometric Construction Grammar of the Zeals Knoll Formation*, Zenodo (2026), DOI 10.5281/zenodo.21527026. The original repository declares its digitized geometry CSV files CC BY 4.0. The source software is MIT-licensed; those are separate rights grants.

Only the derived geometry CSV is copied here. No source photograph, aerial image, map tile, cache, or unlicensed media is included.

## Data semantics

The CSV has 60 records and columns `id,x,y,radius`. The source analysis treats x/y/radius as pixels. These are image-space measurements, not field-local metres, GPS coordinates or surveyed ground truth. The CSV does not include per-circle error bars, digitizer confidence scores or independent annotator measurements.

The source project associates the case with Zeals Knoll and reports the formation date as 5 July 2026. That date is preserved as a source assertion, not independently verified by CROP-INTEL.

## Permitted benchmark use

Use this as a single-case **external geometry fixture** to:
- check CSV ingestion and circle-geometry algorithms;
- reproduce published descriptive geometry metrics;
- compare implementations using identical image-space inputs;
- perform declared sensitivity analysis on centre, radii, segmentation/measurement error and chosen geometric thresholds.

Do not:
- call the CSV a representative dataset;
- treat the source's radius classes, rings, boundary subset, or contact threshold as independent ground truth;
- claim field-measurement accuracy from pixel-space reconstruction;
- infer authorship, intent, symbolic meaning, non-human origin, or formation mechanism from geometric regularity;
- redistribute source photographs without separate permission.

The original data are retained without modifying row order, identifiers, precision, or values. Provenance and limitations are recorded in `manifest.json`.
