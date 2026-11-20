# SC-PRECURSOR-GENOME-AUDIT-0.1

**Status:** source-locked public ledger profile and published holdout reference; no real-data onset model has been reproduced.  
**Research issue:** [#13015](https://github.com/Luminous-Dynamics/symthaea/issues/13015)  
**Upstream data:** [Precursor Genome](https://github.com/lauren-walters/precursor-genome), CC BY 4.0  
**Raw XRD/refinement deposit:** [Zenodo 10.5281/zenodo.21285546](https://doi.org/10.5281/zenodo.21285546)

## Source identity

This work inspected the ledger at upstream main commit `78cb026aa6c9b53aa0ec26ea5611d523bc3c320a`, Git blob `a8ef03cee29670b22ed04b9cebb1686fe26e1acd`. The 31 MB JSON ledger is not copied into Symthaea. The frozen source/profile identity is recorded in `data/research/precursor_genome_source_manifest_v0_1.json`.

The Rust auditor accepts a user-supplied local ledger file. It verifies the Git object blob SHA-1 before parsing, then checks sample and pair identity, categories, synthesis metadata, physical failures, XRD scans, refinements, active refinement selections, acceptance and quality scores. It emits explicit cohort ID lists and exclusion-reason lists. CI uses a tiny synthetic fixture; it never downloads or vendors the upstream ledger.

## Observed source snapshot

| Measure | Public ledger |
|---|---:|
| Samples / unique sample IDs | 1,035 / 1,035 |
| Unique precursor formulae / unique pairs | 46 / 1,035 |
| Outcome categories | transformed 384; unreacted 369; partial 113; complete 143; unclassified 26 |
| Synthesis objects / recorded heating setpoints | 995 / 981 |
| Samples with recorded setpoint + category + no physical-failure flag | 956 |
| XRD scans / Rietveld refinement cases | 1,351 / 1,950 |
| Samples with active refinement selection | 1,030 |
| Active refinement quality | score 1: 685; score 2: 320; score 3: 10; missing: 15; no active selection: 5 |
| Active refinement acceptance | accepted 1,005; rejected 11; missing 14 |
| Setpoint + category + no failure + accepted quality 1/2 | 944 |

Counts are computed from the pinned public JSON snapshot. They are not paper model training rows and not the onset validation set.

## Publication reconciliation: do not silently normalize

The onset paper reports 1,000 analyzable outcomes with 380 transformed, 367 unreacted, 110 partial and 143 complete. The pinned public ledger contains 1,009 categorized rows: 384 transformed, 369 unreacted, 113 partial and 143 complete, plus 26 unclassified physical-failure entries.

The ledger and manuscript therefore differ by 9 categorized rows (4 transformed, 2 unreacted, 3 partial, 0 complete). The 956-row “setpoint present + classified + no physical failure” view is yet another cohort. This repository records the mismatch as unresolved. It does not drop nine records, infer the paper’s exclusions or claim that 956 is the publication's training cohort.

Before any fitted model is scored:
1. Identify exact sample IDs in the paper's 1,000-row analyzable cohort.
2. Reconstruct all physical / characterization / metadata exclusion rules.
3. Explain the nine additional categorized rows using sample-level provenance.
4. Verify the 26 PG row exclusions against the paper's exact training table and recover the 218 PG-LIT row identifiers.
5. Freeze a cohort manifest with source commit, ledger blob, inclusion rules and exact pair IDs.

## Outcome semantics

- `unreacted`: original precursors recovered under the declared characterization.
- `transformed`: one or both precursors may have chemically transformed, but no cross-precursor product was identified. This is a negative for the specific *cross-precursor product* endpoint, not proof of chemical inertness.
- `partially_reacted` / `completely_reacted`: cross-precursor product evidence under the source categorization.
- unclassified / physical-failure / uncharacterizable entries are not to be silently converted to negative reaction labels.

## Published 26-pair held-out onset reference

`data/research/sc_onset_in_situ_validation_v0_1.json` records the 26 pairs and summary onset values from Supplementary Table S8 of arXiv:2611.23527v1. Every record is explicitly labelled `published_summary_only`; no in-situ sample ID, run ID or raw digest is fabricated. I joined the pair identities (unordered precursor formula pairs) against the pinned PG ledger: all 26 pairs map to 26 unique screening sample IDs. Those are preserved separately as `precursor_genome_screen_context`, including the screening setpoint/category and active refinement quality. They are not the in-situ onset measurements.

The paper explicitly states that the 26 corresponding Precursor Genome samples and 218 PG-LIT literature rows involving those same pairs were removed from training. `data/research/sc_onset_pair_holdout_manifest_v0_1.json` now freezes the 26 PG sample IDs and pair-level exclusion rule. The public ledger provides the 26 PG row identities; the exact 218 literature-row IDs are not available in the currently inspected public source. A faithful PG-LIT reproduction still needs the authors' exact training table / row identifiers and split implementation. The 26 pairs must remain excluded from every training fold, feature-selection pass and uncertainty-calibration pass.

The paper reports:
- Tammann heuristic: 131 °C MAE, 152 °C RMSE, R² −0.59.
- 3-feature log-normal AFT: 88 °C MAE, 101 °C RMSE, R² +0.30; nominal 95% prediction intervals cover 25/26.
- 32-feature log-normal AFT: 84 °C MAE, 122 °C RMSE, R² −0.02; intervals cover 21/26.

The 32-feature extension is not a demonstrated improvement over the compact model: its paired MAE difference was −4 °C with 95% interval [−35,+33]. The publication also reports that training on the Precursor Genome data alone degraded the 3-feature model to 180 °C MAE. Training-data breadth and leakage discipline matter more than descriptor count.

## Benchmark strategy

**Reproduce first:** match the 26-pair validation exactly, using the authors' train/test exclusions and data release. Then evaluate distinct frozen cohorts:
- full high-quality ledger cohort for cross-precursor outcome classification;
- temperature-known subset for setpoint-conditioned reaction outcome;
- precursor-held-out and chemistry-family-held-out onset generalization;
- prospective experiment selections under equal synthesis / characterization budget.

Metrics:
- exact-onset MAE/RMSE with paired bootstrap CIs;
- censored log-likelihood on binary dwell outcomes;
- 50/80/90/95% interval coverage and width;
- calibration and time-to-onset residuals;
- balanced accuracy / PR-AUC only for the binary cross-precursor-product event, stratified by characterization quality;
- experiment count and cost to reach target uncertainty or a verified product.
No synthetic and real metrics should be pooled.

## Claim ceiling

This is provenance and cohort infrastructure. It does not reproduce the paper's model, validate raw XRD patterns, establish a real reaction-onset distribution, synthesize a new compound or demonstrate superconductivity.
