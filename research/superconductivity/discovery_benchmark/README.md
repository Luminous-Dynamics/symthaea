# Superconductivity discovery benchmark contract

This workstream implements the benchmark contract and leakage auditor, not a predictor or a materials result.

## Why random-split performance is not enough

The November 2026 AutoSuperCon study reports 20,354 experimental parameter entries across 6,793 superconductors and eight properties. Its composition-based models report log-space R-squared of 0.83 for Tc, with median absolute error 1.6 K across 5,843 materials, and 0.69 for orientation-averaged Hc2 across 1,204 materials. In leave-one-family-out evaluation, the cuprate Tc score falls to -10.2, compared with +0.78 when cuprates are represented in training; the paper also reports severe failure for iron-based materials. The withheld-family cuprate Hc2 score is -9.0, improving to +0.4 when measured Tc is provided.

That is the central benchmark gap: excellent interpolation inside known families is not evidence of reliable extrapolation to a new superconducting family.

BEE-NET is a different and complementary baseline. Its reported 0.87 K MAE is against DFT-derived Allen-Dynes Tc labels, not measured experimental Tc. It predicts the Eliashberg spectral function and integrates ML screening with first-principles calculations, stability filtering and experiments. The 2026 report describes 741 computationally stable candidates and synthesis/experimental confirmation of two previously unreported compounds. This is a genuine workflow milestone, but it must remain a separate computational-label and experimental-confirmation track.

## Published baseline inventory

The machine-readable inventory at `data/research/superconductivity_published_baseline_registry_v0_1.json` pins the published AutoSuperCon, BEE-NET and HTSC-2025 baseline definitions, including label authority, dataset size, test split and reproduction status. It is an inventory, not a downloaded dataset or an independent reproduction.

BEE-NET's public repository README reports 5,241 calculated examples, a 1,049-entry held-out ID list and 100-model CSO/CPD ensembles. The selected database and model checkpoints are documented as not tracked in the repository. Its README warns that training/validation indices are DataFrame row labels but its test list contains Alexandria IDs; this alignment must be verified before replay. The associated Materials Cloud source archive is about 48.1 GiB, so acquisition and local compute should be an explicit, resource-aware decision. The archive metadata lists CC BY 4.0, but rights to reuse the BEE-NET code and model artifacts require separate review.

## Split contract

- Family holdout: no family may appear in both train and test.
- Source-paper holdout: all observations from a paper move together.
- Pressure-regime holdout: pressure bounds are explicit; the gap between train and test is excluded.
- Time-forward holdout: the training cutoff and test start dates are frozen.
- Prototype holdout: prototype identity is versioned and held out as a group.

Each split must group by both material_state_id and source_id. Additional grouping keys are declared per split. Formula grouping can be required for strict composition generalization. Pressure extrapolation may intentionally include the same composition at a different explicitly identified state.

## Evidence contract

The target authority is explicit and must match every observation in a task. Retain normalized formula, family/prototype taxonomy, material-state identity, pressure, temperature when reported, field orientation, sample form, measurement method and criterion, source paper and date, source locator, extraction method, reported-versus-derived status, and uncertainty.

Do not merge resistive onset, zero resistance, diamagnetic onset, bulk thermodynamic transition, or calculated Tc into one label. In particular, a calculated DFT/SCDFT/Allen-Dynes Tc is not experimentally measured, and an unspecified measurement condition is not known.

## Metrics for a real benchmark run

Report by property, measurement class, label authority, family and split regime: MAE/RMSE in physical units and log-space; median absolute error; paired confidence intervals; prediction-interval coverage and width; calibration; risk-versus-coverage for abstaining models; precision/recall and top-k enrichment at fixed candidate budgets; and cost per independently confirmed candidate. Bootstrap by material/source groups, not individual rows. Never pool resistive-onset, zero-resistance, diamagnetic, and bulk-thermodynamic Tc as one metric: score each measurement class separately, and only create a composite endpoint under a preregistered harmonization protocol.

A high global score must never conceal a catastrophic family-specific failure.

## Competitive benchmark plan and prospective targets

Run the same frozen train/test folds and label authority across the following baselines: a composition-only Magpie-descriptor model (including AutoSuperCon-style TabPFN/gradient-boosted baselines); a crystal-structure equivariant graph model; a BEE-NET-style spectral-function model scored against calculated alpha-squared-F/Allen-Dynes labels; and a multi-property, physics-informed model that predicts a distribution and can abstain. Do not compare BEE-NET's DFT-label error with an experimental-label MAE. Keep computational screening precision separate from experimentally confirmed yield.

The following are **prospective programme targets, not achieved results**:

- On source-paper, prototype and family-held-out folds, report paired group-bootstrap intervals and demonstrate calibrated uncertainty. For nominal 90% prediction intervals, target empirical coverage of 85–95% on each adequately represented held-out family; publish a coverage-versus-width curve rather than widening intervals after seeing the test set.
- At a fixed expensive-compute budget and a fixed top-K candidate count, target at least 1.5× independently validated candidate enrichment over the strongest matched baseline, with the paired 95% confidence interval lower bound above 1.0.
- Target at least 2× fewer expensive DFT/alpha-squared-F follow-up calculations per independently confirmed material, without reducing recall on the preregistered test candidates.
- Require explicit abstention on unknown families, missing structural descriptors, unidentifiable GL coefficients, and pressure/temperature conditions outside training support. Report selective risk and coverage; abstention must not be silently counted as a correct negative.
- For physical discovery, require independent confirmation in more than one characterization channel; distinguish a resistive onset from bulk thermodynamic evidence and record pressure, sample form and orientation.

HTSC-2025 is a useful structure/system-level test of theoretically proposed, ambient-pressure BCS candidates (155 samples); its authors already exclude test materials from training based on structural similarity and report MAE by system. It does not replace experimental family-generalization evaluation: its scope is explicitly theoretical, ambient-pressure and largely conventional BCS systems. Our benchmark should complement it with source-disjoint, state-aware, family-held-out and measured-property tracks.
## Current implementation boundary

The version 0.1 auditor validates a seven-row synthetic adversarial fixture. It proves only that the contract and split guardrails behave as coded. The fixture is fabricated, is not an experimental or computational dataset, must not train a model, and has no benchmark score. The Rust auditor deliberately rejects external datasets until a separate importer pins source version, licensing/terms, preprocessing and immutable digests.

## Research references

- Manium et al., AutoSuperCon, arXiv:2611.15627 (11 Nov 2026): https://arxiv.org/abs/2611.15627
- Gibson et al., BEE-NET, npj Computational Materials 12, 95 (27 Jan 2026): https://doi.org/10.1038/s41524-026-01964-8
- HTSC-2025 family/system-aware dataset: https://doi.org/10.1088/1674-1056/adf042
- Finite-momentum SCDFT framework, PRL 137, 196001 (3 Nov 2026): https://doi.org/10.1103/lrhl-j9bt


## Additional critical-field baseline (August 2026)

Geisler et al., *Towards the discovery of high critical magnetic field superconductors*, npj Computational Materials (12 August 2026), reports an approximately 7,300-material **calculated** electron-phonon database with Hc, Hc1, Hc2, coherence length, London penetration depth and Ginzburg-Landau parameters. It combines alpha-squared-F spectral functions and DFT Fermi surfaces with clean-limit Eliashberg theory, including strong-coupling and electron-phonon mass-renormalization effects. This is a separate theoretical baseline from AutoSuperCon's measured-property dataset; never combine their labels or scores. The publisher page identifies Supporting Information, but a row-level machine-readable archive and exact dataset digest have not yet been verified.

The published baseline registry now includes this source as `geisler-hcfield-2026`, with reproduction status `not_reproduced` and an explicit claim ceiling. The detailed reproduction contract is tracked in [SC-HIGH-FIELD-REPRO-001 #12916](https://github.com/Luminous-Dynamics/symthaea/issues/12916). Source: https://doi.org/10.1038/s41524-026-02266-9 (preprint: https://arxiv.org/abs/2601.21044).

## AutoSuperCon artifact-access status (19 November 2026)

The public arXiv HTML for AutoSuperCon describes the extraction method and evaluation in detail: 47,000+ arXiv superconductivity preprints through March 2026; 2,500-character overlapping text segments with 500-character overlap; structured extraction using the 2025-08-07 snapshot of gpt-5-mini; formula normalization with pymatgen; condition-aware duplicate aggregation; and 639 added Roberts (1976) records. The article's accessible HTML does not expose a machine-readable dataset URL or code repository, so the row-level dataset/checkpoint release is not yet admitted for reproduction. Do not recreate/redistribute the data from the paper; obtain the authors' artifact locations and use terms, or keep a paper-derived summary fixture strictly non-training. See [SC-AUTOSUPERCON-REPRO-001 #12908](https://github.com/Luminous-Dynamics/symthaea/issues/12908).


## NEMAD LLM baseline (August 2026)

Itani et al., *Large language models for superconductor discovery*, Journal of Magnetism and Magnetic Materials 652 (15 August 2026), reports 78,203 literature-derived records across 19,058 unique compositions, including experimentally reported superconductors and non-superconductors. Its tasks include SC/NSC classification, composition/structure-aware Tc prediction and inverse design. The paper reports that 28% of inverse-design outputs were unseen in its training data and identifies GNoME candidates with predicted Tc > 10 K; those outputs are unverified and are not discoveries. A count discrepancy (78,203 in the abstract versus 78,204 in one methods description) must be resolved against the official release before replay.

The paper identifies the official data site at https://www.nemad.org, code repository at https://github.com/sumanitani/LLM_for_Superconductors and model index at https://huggingface.co/sumanitani. Data, code and checkpoint reuse terms must be separately pinned before copying or retraining. Reproduction is tracked in [NEMAD-SC-REPRO-001 #12648](https://github.com/Luminous-Dynamics/symthaea/issues/12648). This remains a distinct label/extraction track from AutoSuperCon and BEE-NET.

## Additional 2026 baselines (19 November 2026)

Two recently published approaches expand the evaluation landscape:

- **Feasibility-first hierarchical inference** (Physical Review Materials, 25 August 2026, DOI 10.1103/bg57-bjdk) evaluates superconducting feasibility first using an electronic-compositional backbone, then predicts (T_c) conditionally using additional structural and magnetic/orbital descriptors. The abstract reports a reduction in end-to-end (T_c) error on a balanced dataset of more than 26,000 inorganic materials, including superconductors and non-superconductors. Exact metrics, split manifests, and artifact identities remain to be audited; the registry therefore marks it not reproduced.
- **DataG Non-Sc / genome-inspired extrapolation** (Scientific Reports, early access 29 October 2026, DOI 10.1038/s41598-026-76977-w) introduces a 53,196-material non-superconductor control dataset and compares ten ML with nine neural-network models. The paper reports Pu-based and hydride candidate predictions, including a predicted (T_c\sim89\,K) for Pu2Co17; these are predictions, not measured materials. The dataset's non-superconducting label criteria, row manifest, group splits, and model artifacts require independent qualification. Radioactive actinide candidates stay computational-only unless a qualified institution explicitly handles the relevant review.

These methods create complementary benchmark tracks rather than one unified leaderboard:
1. feasibility classification with calibrated abstention and hard negative controls;
2. conditional (T_c) regression given superconducting feasibility;
3. family/prototype/source/time extrapolation;
4. mechanism-conditioned observables and model discrimination;
5. synthesis/phase realization and same-specimen experimental confirmation.

Do not score a candidate merely because it is ranked highly. A useful discovery claim requires the identity, mechanism applicability, predicted and observed properties, synthesis pathway, and independent validation to be separately evidenced.


## Soft-mode false-negative / recovery benchmark

This child benchmark is linked to SC-SOFT-MODE-RECALL-001 (#13134) and composes the BEE-NET pipeline/VOI owner (#12644) with the harmonic dynamic-stability evidence owner (#5016).

Schema: `data/schemas/sc_soft_mode_triage_v0_1.schema.json`.

Synthetic fixture: `data/research/sc_soft_mode_triage_fixture_v0_1.json`.

Auditor: `tools/superconductivity-discovery-benchmark-auditor/src/bin/soft_mode_triage_audit.rs`.

Run:

```bash
cargo test --locked --manifest-path tools/superconductivity-discovery-benchmark-auditor/Cargo.toml --bin sc_soft_mode_triage_audit
cargo run --locked --manifest-path tools/superconductivity-discovery-benchmark-auditor/Cargo.toml --bin sc_soft_mode_triage_audit -- \
  --schema data/schemas/sc_soft_mode_triage_v0_1.schema.json \
  --fixture data/research/sc_soft_mode_triage_fixture_v0_1.json
```

The auditor separates four dispositions: harmonic stable for the declared 0 K method, marginal/unresolved, robust harmonic instability, and underconverged/unknown. Its screening action is distinct from the physical stability claim. Marginal cases are retained for re-evaluation; robustly unstable high-symmetry structures are routed to a lower-symmetry distortion pathway rather than globally labelling the nominal material impossible. Underconverged records stay unknown.

**Source reference and synthetic values are separate.** The LuRu3B2 literature reference records only the paper-reported weakly imaginary phonon observation and experimental bulk-superconductivity evidence (Tc=0.95 K); it explicitly records that numeric raw phonon artifacts are not admitted. Every numerical frequency, uncertainty and convergence value in the smoke fixture belongs to fabricated synthetic candidate controls and must never be attributed to LuRu3B2 or used as scientific data.

The test fixture includes an intentionally false-negative case for the simple any-imaginary-mode hard gate, plus stable, robustly unstable, underconverged and sign-sensitive controls. The receipt compares the hard-gate false negative with the triage queue's ability to preserve a positive case, while ensuring no unsupported harmonic-stability claim is emitted.

This proves only software disposition semantics on synthetic inputs and a source-linked positive-control citation. It does not reproduce material phonons, validate a universal imaginary-frequency threshold, establish anharmonic stabilization or prove an improved physical discovery rate.
