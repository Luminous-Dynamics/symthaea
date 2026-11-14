# EDLC External Raw-Data Reproduction Benchmark v0.1

**As of:** 2026-11-14  
**Status:** dataset metadata reviewed and registered; raw file download, checksum re-verification, format inspection and metric reproduction are still pending.

## 1. Why this is a separate benchmark lane

We do not need to wait for a laboratory partner to test the data pipeline and reproduce published electrochemical measurements. Public raw datasets can support:
- CSV ingestion, unit and sample-identity checks;
- cyclic-voltammetry (CV) curve reproduction;
- galvanostatic charge/discharge (GCD) capacitance and rate-performance reproduction;
- electrochemical impedance spectroscopy (EIS) model identification;
- cycle-ageing and pre-/post-ageing comparisons;
- uncertainty, group-holdout and reproducibility tests.

But these datasets do **not** replace application-specific testing on a complete EDLC module after standby. An electrode, flexible coin-style cell or material characterization result cannot establish module Wh/kg, installed-system Wh/L, standby reserve, repeated-call readiness, safety, or product lifetime.

Keep three independent evidence tracks:

1. **Published-data reproduction:** reproduce the source's own raw-data metrics and protocol.
2. **Physical-device reserve qualification:** standby, actual load waveform, repeated calls, recovery and installed-system parasitics.
3. **Commercial matched comparison:** test commercial reference devices and prototypes on the same frozen application profile and physical boundary.

No score should be averaged across these tracks.

## 2. Candidate dataset A — DataverseNO CC0 carbon materials

**Dataset:** Zhang and Wei, *Electrochemical and Structural Data of Carbon-Based Materials for Energy Storage*  
**DOI:** [10.18710/F4NFMJ](https://doi.org/10.18710/F4NFMJ)  
**Repository version:** 1.0  
**Publication date:** 2026-05-08  
**License:** CC0 1.0 Universal, as stated on the dataset record  
**Repository:** DataverseNO, University of South-Eastern Norway

The version 1.0 record lists 26 public files. The dataset description includes raw CSV CV, GCD, EIS and cycling data plus XRD, Raman, SEM and BET characterization.

Selected files discovered in the public manifest:

| File | Size | Repository MD5 | First eligible task |
|---|---:|---|---|
| `CyclicVoltammetry_PCNS_TemperatureComparison_50mVs.csv` | 220.4 KB | `42631fe50e35001dfcfd474f99db37dd` | CV curve reproduction at a named scan rate |
| `GalvanostaticChargeDischarge_PCNS_TemperatureComparison_0.1Ag.csv` | 3.9 MB | `88bc3dc26bd37971d7ca4a80ed79f290` | GCD discharge integration and capacitance reproduction |
| `ImpedanceSpectroscopy_PCNS_TemperatureComparison.csv` | 2.9 KB | `4ef9d5efe0519c599a03b6527511c54c` | EIS header/unit inspection and impedance-model fit |

The persistent file identifiers above are the file-level DOIs. The MD5 values are repository-provided checksums, **not locally re-computed verification**. The page also provides a per-file URL form `https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/F4NFMJ/RREE4O`; the manifest records the corresponding file DOIs. The files must be downloaded, byte-counted, hashed locally and their actual headers/units inspected before being admitted to the executable benchmark. The container download attempt has not produced a local file yet, so the three files remain metadata-only candidates.

The dataset is a material/electrochemical characterization resource. It is not a commercial module test. Do not infer complete-device energy density, installed-system power, standby service or sequential-reserve reliability.

## 3. Candidate dataset B — TiO2 ALD-coated activated-carbon flexible supercapacitors

**Dataset:** Vitto et al., *Charge-storage evolution of TiO2 ALD-coated AC flexible supercapacitors over long-term cycling*  
**DOI:** [10.5281/zenodo.22106814](https://doi.org/10.5281/zenodo.22106814)  
**Repository version:** v2  
**Publication date:** 2026-08-26  
**Repository:** Zenodo

The record describes pristine activated-carbon and TiO2-ALD-coated symmetric flexible supercapacitors, including 60- and 100-cycle ALD variants. It lists long-term GCD cycling to 10,000 cycles at low and high applied current densities, CV at varying scan rates, Trasatti analysis, before/after MACCOR data and publication analysis files. The versioned record lists these archive files and repository MD5 values: `0.1Ag Long-term Cycling.zip` (a4b3f9b0408c53726f631917056504a4), `1.0Ag Long-term Cycling.zip` (d15048417ce7f351fdf573c83edd8a36), `CV.zip` (edd70a94907c301b284e0965e80d5bf6), and pre-/post-ageing MACCOR archives (741e8d7c8d4099b1b5052606cc1e1a92 / 8e98f3445744d92cc3601d0d9ecbfcdf). These remain repository metadata, not locally re-computed checksums.

**Rights gate:** the public Zenodo Rights section did not expose a licence value at inspection time. Treat this dataset as discoverable and citeable metadata only until the licence/terms are confirmed. Do not commit or mirror raw archives before that check. The record is a research-device ageing comparator, not an installed-module or reserve-service benchmark.

## 4. Reproduction protocol

Before analysis, freeze:
- DOI, repository record ID, version and file-specific persistent ID;
- file name, byte count, repository checksum and locally verified SHA-256;
- licence and any data-use conditions;
- source paper/protocol and exact sample / variant / current-density / scan-rate identifiers;
- column names, units, sign convention, sample interval and missing-data rules;
- transformations, software revision, model configuration and analysis outputs.

Preserve raw files byte-for-byte. Every normalized trace must reference the raw file SHA-256 and describe the exact transformation, unit conversion, column mapping and parser version.

### CV lane

Reproduce current-versus-potential curves for each sample/scan-rate/temperature stratum. Report:
- exact voltage window and scan rate;
- integrated current/charge with the source's mass/area/normalization boundary;
- forward/reverse branch handling and cycle selection;
- numerical integration method and uncertainty/sensitivity to sample spacing.

Do not infer full-device Wh/kg from electrode-specific CV results.

### GCD lane

For each cycle and current-density condition:
- retain raw time/voltage/current values and exact sign convention;
- identify charge/discharge segments and voltage cutoffs;
- reproduce discharge capacitance using the paper's stated formula and boundary;
- calculate energy from measured terminal V×I where simultaneous terminal current is available;
- publish sensitivity to the selected discharge interval and any IR-drop treatment.

Do not silently substitute active-material mass for total-cell or complete-device mass.

### EIS lane

Retain frequency, real impedance, imaginary impedance, units, perturbation amplitude, DC bias and temperature if supplied. Compare one-branch RC, two-branch RC and higher-order/distributed models only where the measured frequency range and data support identifiability.

Report complex residuals by frequency, parameter uncertainty/correlation, fit stability and held-out condition performance. A lower residual from a more complex circuit is not automatically a better physical explanation.

### Ageing lane

Use the actual source's repeated-cycle and before/after records. Split by complete sample/device identity and material variant. Do not randomly place rows or adjacent cycles from the same device in both train and test.

Report:
- capacitance/energy retention by cycle and condition;
- ESR/EIS evolution where available;
- uncertainty and missingness;
- between-sample dispersion;
- held-out sample/variant predictions;
- whether the public data support calendar standby, repeated-call or recovery conclusions (normally they do not).

## 5. Model tournament and SOTA discipline

Models to compare when the data permit:
1. direct reproduction of source metrics;
2. one-branch RC / Thevenin baseline;
3. two-branch RC;
4. multi-branch RC;
5. transmission-line or porous-electrode models, only with sufficient frequency/geometry support;
6. fractional-order / distributed-time-constant candidates, only where identifiable;
7. physics-informed state-space models;
8. calibrated ML residuals as challengers, never assumed winners.

Primary evaluation split is by complete sample/device. Stronger splits hold out entire material variants, temperature conditions or source/lab cohorts. Hyperparameter selection must not inspect the held-out groups.

Metrics:
- CV/GCD source-metric reproduction error;
- capacitance and energy absolute/relative error;
- EIS complex residual and parameter identifiability;
- ageing-trajectory prediction error;
- 90% / 95% prediction interval coverage and width;
- calibration by material / temperature / rate / cycle stratum;
- abstention outside the measured domain;
- false-pass cost for any reserve-service decision.

The public datasets do not contain the complete standby-to-load sequence contract required for reserve readiness. Do not publish a false-ready score or mission pass rate from these data.

## 6. Current admission state

| Requirement | DataverseNO CC0 | Zenodo TiO2-AC |
|---|---|---|
| Versioned public record | Yes, v1.0 | Yes, v2 |
| Public files listed | 26 | Raw-data archive files listed |
| Explicit licence verified | CC0 1.0 | Not yet verified |
| File-level repository hashes | MD5 listed | Listed on archive inventory |
| Local SHA-256 recomputed | Pending | Pending |
| Raw headers/units inspected | Pending download | Pending download |
| Source metric reproduced | Not yet | Not yet |
| Complete-module reserve evidence | No | No |

No raw external files have been committed to this branch. No dataset reproduction result or external-model performance claim is made yet.

## 7. Next executable steps

1. Download the selected CC0 files and compute SHA-256 locally; verify file bytes and inspect headers.
2. Build a Rust CSV adapter and immutable data-admission receipt; retain the source MD5 as a source-provided cross-check.
3. Reproduce one CV curve, one GCD discharge and one EIS dataset with source-specific units.
4. Add grouped sample-level evaluation and the first one-/two-branch RC model comparison.
5. Confirm the Zenodo licence before accessing or mirroring its raw archives.
6. Feed actual lab captures through the separate [EDLC trace evidence receipt](EDLC_TRACE_EVIDENCE_RECEIPT_V0_1.md), never mix these published files with physical reserve-service records.
