# NeuroTwin Dataset Intake Protocol v0.1

**Purpose:** admit a research dataset into a reproducible experiment without copying participant-level data into public repositories.
**BIDS target:** stable specification 1.11.2 (29 September 2026). Dataset-specific version is read from `dataset_description.json` and must be preserved.

## Intake stages

1. **Authority:** identify data controller, custodian, data-use agreement, licence, permitted purposes, cross-border constraints, ethics/IRB determination and expiry.
2. **Dataset identity:** record dataset name, stable release/version, source location, retrieval date, BIDS version, data dictionary and citation in a restricted registry.
3. **Privacy boundary:** keep raw imaging, EEG/MEG, identifiers, phenotypic rows and subject-level inclusion lists inside approved encrypted storage. Do not place them in a public DHT or public Git repository.
4. **Structural preflight:** run the local NeuroTwin cohort auditor. Preserve its aggregate receipt in the restricted study registry. Treat BLOCKED as no-go and REVIEW as requiring resolution/justification.
5. **Independent BIDS validation:** run the official BIDS Validator against the same immutable dataset copy. The NeuroTwin preflight is not a replacement.
6. **Scientific intake:** validate acquisition parameters, modality availability, phenotype/outcome dictionaries, visit timing, missingness, label construction and cohort selection with a domain expert.
7. **Experiment lock:** freeze the participant inclusion manifest, endpoint, estimand, split plan, preprocessing plan, baseline models, metric set, analysis plan and code revision before model development.

## Immutable experiment manifest

The restricted experiment registry should record:

- Study / protocol ID and version.
- Dataset source, release, BIDSVersion and retrieval date.
- Data-use agreement and licence identifiers.
- Ethics determination, consent scope and expiry.
- Custodian and secure-environment identity.
- Restricted source manifest digest and protected participant inclusion manifest commitment.
- BIDS Validator version and complete report digest.
- NeuroTwin auditor version, receipt and finding resolution.
- Modality / visit / outcome dictionary versions.
- Inclusion/exclusion criteria and cohort flow counts.
- Preprocessing, split, model and metric protocol versions.
- Model code commit, environment lock, seed and output artifact digest.
- Retention / destruction date and downstream revocation procedure.

Do not publish participant IDs, row-level phenotype data, individual scan hashes, diagnosis counts or other restricted fields without explicit permission. An aggregate count can still be disclosive in a small or rare cohort.

## Split and leakage controls

- Group repeated scans, sessions and visits by participant.
- Identify family/household relationships and related records where available.
- Use site/scanner-held-out evaluation when transportability is claimed.
- Use time-held-out evaluation for future prediction.
- Fit every learned preprocessing step inside the training partition.
- Keep test cohorts and threshold selection locked.
- Compare modality ablations on the same participant intersection.
- Report all missingness by site, visit, modality and outcome stratum.
- Never infer treatment efficacy from a model trained only on observational prognostic labels.

## No-go conditions

- Missing or expired data-use authority.
- Unclear consent / permitted purpose.
- An unresolved ethics determination.
- Unresolved participant identity collisions or invalid split leakage.
- Unavailable data dictionary / outcome definition.
- Unresolved critical BIDS validation errors.
- Inability to protect raw data in the approved environment.
- Inability to establish independent external validation for a generalization claim.

## Evidence ceiling

A successful intake proves only that the declared data source has passed documented authority, structural and study-specific checks. It does not establish that the cohort is representative, measurements are reliable, labels are valid, the model generalizes, a biomarker is causal, or a treatment is effective.
