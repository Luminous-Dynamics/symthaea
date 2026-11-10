// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-aware OLI benchmark adapters.
//!
//! This crate deliberately reports task-specific, evidence-bounded results.
//! It does not infer a universal intelligence, life, sentience, or consciousness score.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

pub const SOURCE_DATASET_DOI: &str = "10.5281/zenodo.18849382";
pub const SOURCE_DATASET_CONCEPT_DOI: &str = "10.5281/zenodo.18846348";
pub const SOURCE_GITHUB_TAG: &str = "v1.0.4";
pub const SOURCE_TRIAL_CSV_BLOB_SHA: &str = "92499810019fb631fffb508dbacc1de2e550797b";
pub const BOOTSTRAP_REPLICATES: usize = 12_000;
pub const BOOTSTRAP_SEED: u32 = 20_261_110;
pub const CONDITIONS: [&str; 5] = [
    "no_displacement",
    "rotate",
    "slide",
    "rotate-slide",
    "slide-rotate",
];

const EXPECTED_ROWS: usize = 375;
const EXPECTED_SUBJECTS: usize = 25;
const TRIALS_PER_SUBJECT_CONDITION: usize = 3;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Trial {
    pub study_id: String,
    pub subject_key: String,
    pub species: String,
    pub block_of_trials: u32,
    pub trial: u32,
    pub condition: String,
    pub initial: String,
    #[serde(rename = "final")]
    pub final_position: String,
    pub choice: String,
    pub correct: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConditionSummary {
    pub condition: String,
    pub subjects: usize,
    pub trials: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub cluster_bootstrap_95: [f64; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct PairedDifference {
    pub contrast: String,
    pub mean_difference: f64,
    pub cluster_bootstrap_95: [f64; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct BaselineScore {
    pub baseline: String,
    pub held_out_subjects: usize,
    pub predictions: usize,
    pub accuracy_at_0_5: f64,
    pub brier_score: f64,
    pub mean_log_loss: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub schema_version: &'static str,
    pub audit_id: &'static str,
    pub source_dataset_doi: &'static str,
    pub source_dataset_concept_doi: &'static str,
    pub source_github_tag: &'static str,
    pub source_trial_csv_blob_sha: &'static str,
    pub rows: usize,
    pub subjects: usize,
    pub species_counts: BTreeMap<String, usize>,
    pub condition_results: Vec<ConditionSummary>,
    pub paired_subject_differences: Vec<PairedDifference>,
    pub leave_one_subject_out_baselines: Vec<BaselineScore>,
    pub independent_unit: &'static str,
    pub split_policy: &'static str,
    pub claim_ceiling: Vec<&'static str>,
}

#[derive(Debug, Default)]
struct SubjectProfile {
    species: Option<String>,
    conditions: BTreeMap<String, Vec<bool>>,
}

#[derive(Debug, Default)]
struct ScoreAccumulator {
    correct: usize,
    predictions: usize,
    brier_sum: f64,
    log_loss_sum: f64,
}

impl ScoreAccumulator {
    fn push(&mut self, probability_correct: f64, outcome_correct: bool) {
        let probability = probability_correct.clamp(1.0e-15, 1.0 - 1.0e-15);
        let observed = if outcome_correct { 1.0 } else { 0.0 };
        self.predictions += 1;
        if (probability >= 0.5) == outcome_correct {
            self.correct += 1;
        }
        self.brier_sum += (probability - observed).powi(2);
        self.log_loss_sum += if outcome_correct {
            -probability.ln()
        } else {
            -(1.0 - probability).ln()
        };
    }

    fn finish(self, baseline: &'static str, held_out_subjects: usize) -> BaselineScore {
        let denominator = self.predictions.max(1) as f64;
        BaselineScore {
            baseline: baseline.to_owned(),
            held_out_subjects,
            predictions: self.predictions,
            accuracy_at_0_5: self.correct as f64 / denominator,
            brier_score: self.brier_sum / denominator,
            mean_log_loss: self.log_loss_sum / denominator,
        }
    }
}

/// Parse the minimized, pseudonymous trial CSV.
pub fn parse_trials(csv_text: &str) -> Result<Vec<Trial>, String> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(csv_text.as_bytes());

    let expected_headers = [
        "study_id",
        "subject_key",
        "species",
        "block_of_trials",
        "trial",
        "condition",
        "initial",
        "final",
        "choice",
        "correct",
    ];
    let headers = reader
        .headers()
        .map_err(|error| format!("CSV header error: {error}"))?;
    let actual_headers: Vec<&str> = headers.iter().collect();
    if actual_headers != expected_headers {
        return Err(format!(
            "CSV headers do not match the minimized OLI contract: {actual_headers:?}"
        ));
    }

    let mut trials = Vec::new();
    for (index, row) in reader.deserialize::<Trial>().enumerate() {
        let trial = row.map_err(|error| format!("CSV row {}: {error}", index + 2))?;
        trials.push(trial);
    }
    Ok(trials)
}

/// Validate source-derived records and calculate descriptive and LOSO baselines.
pub fn audit_csv(csv_text: &str) -> Result<AuditReport, String> {
    let trials = parse_trials(csv_text)?;
    validate_trials(&trials)?;

    let mut subjects: BTreeMap<String, SubjectProfile> = BTreeMap::new();
    let mut species_counts = BTreeMap::<String, usize>::new();

    for trial in &trials {
        let profile = subjects.entry(trial.subject_key.clone()).or_default();
        match &profile.species {
            Some(species) if species != &trial.species => {
                return Err(format!(
                    "subject {} has inconsistent species values",
                    trial.subject_key
                ));
            }
            None => profile.species = Some(trial.species.clone()),
            _ => {}
        }
        profile
            .conditions
            .entry(trial.condition.clone())
            .or_default()
            .push(trial.correct == "correct");
    }

    for profile in subjects.values() {
        let species = profile
            .species
            .as_ref()
            .ok_or_else(|| "subject profile is missing species".to_owned())?;
        *species_counts.entry(species.clone()).or_default() += 1;
    }

    let subject_profiles: Vec<&SubjectProfile> = subjects.values().collect();
    let mut rng = DeterministicLcg::new(BOOTSTRAP_SEED);
    let condition_results = condition_summaries(&subject_profiles, &mut rng);
    let paired_subject_differences = paired_differences(&subject_profiles, &mut rng);
    let leave_one_subject_out_baselines = loso_baselines(&subjects);

    Ok(AuditReport {
        schema_version: "oli-ape-spatial-descriptive-audit-v0.2",
        audit_id: "OLI-APE-SPATIAL-001-DESCRIPTIVE-002",
        source_dataset_doi: SOURCE_DATASET_DOI,
        source_dataset_concept_doi: SOURCE_DATASET_CONCEPT_DOI,
        source_github_tag: SOURCE_GITHUB_TAG,
        source_trial_csv_blob_sha: SOURCE_TRIAL_CSV_BLOB_SHA,
        rows: trials.len(),
        subjects: subjects.len(),
        species_counts,
        condition_results,
        paired_subject_differences,
        leave_one_subject_out_baselines,
        independent_unit: "subject",
        split_policy: "leave-one-subject-out; all trials from a subject remain in one held-out fold",
        claim_ceiling: vec![
            "One study and one spatial/object-tracking task only",
            "Descriptive reproduction and subject-held-out baseline comparison only",
            "No cross-study, cross-lab, or cross-protocol generalization claim",
            "No species ranking, universal intelligence score, or consciousness inference",
        ],
    })
}

fn validate_trials(trials: &[Trial]) -> Result<(), String> {
    if trials.len() != EXPECTED_ROWS {
        return Err(format!(
            "expected {EXPECTED_ROWS} trial rows, found {}",
            trials.len()
        ));
    }

    let mut subject_keys = BTreeSet::new();
    let mut study_ids = BTreeSet::new();
    let mut counts = BTreeMap::<(String, String), usize>::new();
    let mut condition_counts = BTreeMap::<String, usize>::new();
    let mut trial_keys = HashSet::new();

    for trial in trials {
        if trial.study_id != "albiach2010great" {
            return Err(format!("unexpected study_id: {}", trial.study_id));
        }
        study_ids.insert(trial.study_id.as_str());

        if !trial.subject_key.starts_with("subject-") {
            return Err("subject keys must be pseudonymous subject-* tokens".to_owned());
        }
        subject_keys.insert(trial.subject_key.as_str());

        if !matches!(
            trial.species.as_str(),
            "chimpanzee" | "bonobo" | "gorilla" | "orangutan"
        ) {
            return Err(format!("unsupported species category: {}", trial.species));
        }
        if !CONDITIONS.contains(&trial.condition.as_str()) {
            return Err(format!("unsupported condition: {}", trial.condition));
        }
        if trial.block_of_trials == 0 || trial.trial == 0 {
            return Err("block_of_trials and trial must be positive".to_owned());
        }
        for (field, value) in [
            ("initial", trial.initial.as_str()),
            ("final", trial.final_position.as_str()),
            ("choice", trial.choice.as_str()),
        ] {
            if !matches!(value, "left" | "middle" | "right") {
                return Err(format!("unsupported {field} location: {value}"));
            }
        }

        let expected_outcome = if trial.choice == trial.final_position {
            "correct"
        } else {
            "incorrect"
        };
        if trial.correct != expected_outcome {
            return Err(format!(
                "correctness label contradicts choice/final location for {}",
                trial.subject_key
            ));
        }

        if !matches!(trial.correct.as_str(), "correct" | "incorrect") {
            return Err(format!("unsupported outcome label: {}", trial.correct));
        }

        let key = (
            trial.subject_key.clone(),
            trial.block_of_trials,
            trial.trial,
        );
        if !trial_keys.insert(key) {
            return Err(format!(
                "duplicate trial key for {} block {} trial {}",
                trial.subject_key, trial.block_of_trials, trial.trial
            ));
        }

        *counts
            .entry((trial.subject_key.clone(), trial.condition.clone()))
            .or_default() += 1;
        *condition_counts.entry(trial.condition.clone()).or_default() += 1;
    }

    if study_ids.len() != 1 {
        return Err(format!("expected one study, found {}", study_ids.len()));
    }
    if subject_keys.len() != EXPECTED_SUBJECTS {
        return Err(format!(
            "expected {EXPECTED_SUBJECTS} pseudonymous subjects, found {}",
            subject_keys.len()
        ));
    }
    if condition_counts.len() != CONDITIONS.len() {
        return Err(format!(
            "expected {} conditions, found {}",
            CONDITIONS.len(),
            condition_counts.len()
        ));
    }

    for condition in CONDITIONS {
        let observed = condition_counts.get(condition).copied().unwrap_or(0);
        if observed != EXPECTED_SUBJECTS * TRIALS_PER_SUBJECT_CONDITION {
            return Err(format!(
                "condition {condition} has {observed} rows; expected {}",
                EXPECTED_SUBJECTS * TRIALS_PER_SUBJECT_CONDITION
            ));
        }
    }

    for subject in &subject_keys {
        for condition in CONDITIONS {
            let observed = counts
                .get(&(subject.to_string(), condition.to_owned()))
                .copied()
                .unwrap_or(0);
            if observed != TRIALS_PER_SUBJECT_CONDITION {
                return Err(format!(
                    "{subject} has {observed} trials for {condition}; expected {TRIALS_PER_SUBJECT_CONDITION}"
                ));
            }
        }
    }
    Ok(())
}

fn condition_summaries(
    subjects: &[&SubjectProfile],
    rng: &mut DeterministicLcg,
) -> Vec<ConditionSummary> {
    CONDITIONS
        .iter()
        .map(|condition| {
            let mut correct = 0usize;
            let mut total = 0usize;
            for subject in subjects {
                if let Some(outcomes) = subject.conditions.get(*condition) {
                    correct += outcomes.iter().filter(|value| **value).count();
                    total += outcomes.len();
                }
            }
            let accuracy = correct as f64 / total.max(1) as f64;
            ConditionSummary {
                condition: (*condition).to_owned(),
                subjects: subjects.len(),
                trials: total,
                correct,
                accuracy,
                cluster_bootstrap_95: bootstrap_condition(subjects, condition, rng),
            }
        })
        .collect()
}

fn bootstrap_condition(
    subjects: &[&SubjectProfile],
    condition: &str,
    rng: &mut DeterministicLcg,
) -> [f64; 2] {
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut correct = 0usize;
        let mut total = 0usize;
        for _ in 0..subjects.len() {
            let selected = subjects[rng.next_index(subjects.len())];
            if let Some(outcomes) = selected.conditions.get(condition) {
                correct += outcomes.iter().filter(|value| **value).count();
                total += outcomes.len();
            }
        }
        samples.push(correct as f64 / total.max(1) as f64);
    }
    percentile_interval(samples)
}

fn paired_differences(
    subjects: &[&SubjectProfile],
    rng: &mut DeterministicLcg,
) -> Vec<PairedDifference> {
    let contrasts = [
        ("no_displacement", "rotate"),
        ("no_displacement", "slide"),
        ("rotate", "rotate-slide"),
        ("slide", "slide-rotate"),
        ("no_displacement", "rotate-slide"),
    ];

    contrasts
        .iter()
        .map(|(left, right)| {
            let differences: Vec<f64> = subjects
                .iter()
                .map(|subject| {
                    subject_accuracy(subject, left) - subject_accuracy(subject, right)
                })
                .collect();
            let mean_difference = mean(&differences);
            let mut bootstrap = Vec::with_capacity(BOOTSTRAP_REPLICATES);
            for _ in 0..BOOTSTRAP_REPLICATES {
                let mut sample = 0.0;
                for _ in 0..subjects.len() {
                    sample += differences[rng.next_index(subjects.len())];
                }
                bootstrap.push(sample / subjects.len().max(1) as f64);
            }
            PairedDifference {
                contrast: format!("{left} - {right}"),
                mean_difference,
                cluster_bootstrap_95: percentile_interval(bootstrap),
            }
        })
        .collect()
}

fn subject_accuracy(subject: &SubjectProfile, condition: &str) -> f64 {
    match subject.conditions.get(condition) {
        Some(outcomes) => {
            outcomes.iter().filter(|value| **value).count() as f64
                / outcomes.len().max(1) as f64
        }
        None => 0.0,
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn percentile_interval(mut values: Vec<f64>) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let lower = ((values.len() as f64) * 0.025).floor() as usize;
    let upper = ((values.len() as f64) * 0.975).floor() as usize;
    [
        values[lower.min(values.len() - 1)],
        values[upper.min(values.len() - 1)],
    ]
}

fn loso_baselines(subjects: &BTreeMap<String, SubjectProfile>) -> Vec<BaselineScore> {
    let mut random_choice = ScoreAccumulator::default();
    let mut pooled_rate = ScoreAccumulator::default();
    let mut condition_rate = ScoreAccumulator::default();

    for (held_out_key, held_out) in subjects {
        let mut pooled_correct = 0usize;
        let mut pooled_trials = 0usize;
        let mut by_condition = BTreeMap::<String, (usize, usize)>::new();

        for (training_key, training_subject) in subjects {
            if training_key == held_out_key {
                continue;
            }
            for (condition, outcomes) in &training_subject.conditions {
                let entry = by_condition.entry(condition.clone()).or_default();
                entry.0 += outcomes.iter().filter(|value| **value).count();
                entry.1 += outcomes.len();
                pooled_correct += outcomes.iter().filter(|value| **value).count();
                pooled_trials += outcomes.len();
            }
        }

        let pooled_probability =
            (pooled_correct as f64 + 1.0) / (pooled_trials as f64 + 2.0);
        for (condition, outcomes) in &held_out.conditions {
            let (condition_correct, condition_trials) =
                by_condition.get(condition).copied().unwrap_or_default();
            let condition_probability =
                (condition_correct as f64 + 1.0) / (condition_trials as f64 + 2.0);

            for outcome in outcomes {
                random_choice.push(1.0 / 3.0, *outcome);
                pooled_rate.push(pooled_probability, *outcome);
                condition_rate.push(condition_probability, *outcome);
            }
        }
    }

    vec![
        random_choice.finish("random_choice_p_1_over_3", subjects.len()),
        pooled_rate.finish("condition_independent_laplace_rate", subjects.len()),
        condition_rate.finish("condition_only_laplace_rate", subjects.len()),
    ]
}

struct DeterministicLcg {
    state: u32,
}

impl DeterministicLcg {
    const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_index(&mut self, length: usize) -> usize {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        let unit = self.state as f64 / 4_294_967_296.0;
        ((unit * length as f64).floor() as usize).min(length - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/albiach2010great_pseudonymous.csv");

    #[test]
    fn source_fixture_has_expected_dimensions_and_no_named_subject_columns() {
        let trials = parse_trials(FIXTURE).expect("fixture should parse");
        assert_eq!(trials.len(), EXPECTED_ROWS);
        let subjects: BTreeSet<&str> =
            trials.iter().map(|trial| trial.subject_key.as_str()).collect();
        assert_eq!(subjects.len(), EXPECTED_SUBJECTS);
        assert!(trials.iter().all(|trial| trial.subject_key.starts_with("subject-")));
        assert!(!FIXTURE.contains("participant,"));
        assert!(!FIXTURE.contains("age_in_years"));
        assert!(!FIXTURE.contains("parent"));
    }

    #[test]
    fn source_fixture_passes_semantic_validation() {
        validate_trials(&parse_trials(FIXTURE).expect("fixture should parse"))
            .expect("fixture should satisfy the task contract");
    }

    #[test]
    fn descriptive_results_match_frozen_condition_ordering() {
        let report = audit_csv(FIXTURE).expect("audit should succeed");
        assert_eq!(report.rows, 375);
        assert_eq!(report.subjects, 25);
        assert_eq!(report.condition_results.len(), 5);
        let accuracy: Vec<f64> = report
            .condition_results
            .iter()
            .map(|result| result.accuracy)
            .collect();
        assert!((accuracy[0] - 0.92).abs() < 1.0e-12);
        assert!((accuracy[1] - 0.626_666_666_7).abs() < 1.0e-9);
        assert!((accuracy[2] - 0.613_333_333_3).abs() < 1.0e-9);
        assert!((accuracy[3] - 0.493_333_333_3).abs() < 1.0e-9);
        assert!((accuracy[4] - 0.493_333_333_3).abs() < 1.0e-9);
    }

    #[test]
    fn every_baseline_uses_all_subjects_as_held_out_folds() {
        let report = audit_csv(FIXTURE).expect("audit should succeed");
        assert_eq!(report.leave_one_subject_out_baselines.len(), 3);
        for baseline in report.leave_one_subject_out_baselines {
            assert_eq!(baseline.held_out_subjects, 25);
            assert_eq!(baseline.predictions, 375);
            assert!(baseline.brier_score.is_finite());
            assert!(baseline.mean_log_loss.is_finite());
        }
    }

    #[test]
    fn dropped_trial_is_rejected() {
        let mut rows: Vec<&str> = FIXTURE.lines().collect();
        rows.pop();
        let truncated = rows.join("\n");
        assert!(audit_csv(&truncated).is_err());
    }

    #[test]
    fn invalid_outcome_is_rejected() {
        let mut altered = FIXTURE.to_owned();
        let line = altered.lines().nth(1).expect("first data row");
        let replacement = line.rfind("correct").expect("outcome token");
        let mut changed = line.to_owned();
        changed.replace_range(replacement..replacement + "correct".len(), "unknown!");
        altered = altered.replacen(line, &changed, 1);
        assert!(audit_csv(&altered).is_err());
    }
}
