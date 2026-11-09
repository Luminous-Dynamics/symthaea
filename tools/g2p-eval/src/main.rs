// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent, reference-based G2P scoring.
//!
//! This tool scores candidate pronunciation hypotheses against separately
//! authored, versioned reference variants. It never treats another G2P
//! system's output as truth. Each input JSONL record is one lexical item or
//! fixed phrase, not an arbitrary token stream.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
struct ReferenceVariant {
    variant_id: String,
    /// Canonical IPA phone tokens, not Unicode characters from a raw IPA string.
    phones: Vec<String>,
    /// Exclusive phone indices ending each syllable; last boundary must equal
    /// phones.len(). Example [2, 4] means two syllables ending at phones 2 and 4.
    syllable_boundaries: Vec<usize>,
    /// One lexical stress value per syllable: 0 none, 1 primary, 2 secondary.
    syllable_stress: Vec<u8>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReferenceCase {
    case_id: String,
    language: String,
    /// Versioned orthographic/phonological variety (for example en-US-v1 or zu-ZA-standard-v1).
    profile_id: String,
    text: String,
    context: String,
    expected_oov: bool,
    accepted_variants: Vec<ReferenceVariant>,
}

#[derive(Debug, Clone, Deserialize)]
struct CandidateCase {
    case_id: String,
    candidate_id: String,
    source_id: String,
    language: String,
    /// Must match the exact reference profile used to evaluate this hypothesis.
    profile_id: String,
    phones: Vec<String>,
    syllable_boundaries: Option<Vec<usize>>,
    syllable_stress: Option<Vec<u8>>,
    detected_oov: bool,
    /// Probability that this complete case's pronunciation is exactly correct.
    /// Omit when the candidate system does not expose calibrated confidence.
    confidence_exact: Option<f64>,
}

#[derive(Debug, Serialize)]
struct EditCounts {
    substitutions: usize,
    deletions: usize,
    insertions: usize,
}

impl EditCounts {
    fn total(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }
}

#[derive(Debug, Serialize)]
struct BoundaryCounts {
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
    precision: Option<f64>,
    recall: Option<f64>,
    f1: Option<f64>,
}

#[derive(Debug, Serialize)]
struct CaseScore {
    case_id: String,
    language: String,
    profile_id: String,
    selected_reference_variant: Option<String>,
    reference_phone_count: usize,
    candidate_phone_count: usize,
    edits: EditCounts,
    phone_error_rate: Option<f64>,
    exact_pronunciation: Option<bool>,
    syllable_count_reference: usize,
    syllable_count_candidate: Option<usize>,
    boundaries: Option<BoundaryCounts>,
    stress_correct: Option<usize>,
    stress_compared: Option<usize>,
    stress_accuracy: Option<f64>,
    expected_oov: bool,
    detected_oov: bool,
    oov_detection_correct: bool,
    confidence_exact: Option<f64>,
}

#[derive(Debug, Serialize)]
struct CalibrationBin {
    lower_bound: f64,
    upper_bound: f64,
    observations: usize,
    mean_confidence: Option<f64>,
    observed_accuracy: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ConfidenceCalibration {
    observations: usize,
    brier_score: Option<f64>,
    expected_calibration_error: Option<f64>,
    bins: Vec<CalibrationBin>,
}

#[derive(Debug, Serialize)]
struct AggregateMetrics {
    cases: usize,
    candidate_id: String,
    source_id: String,
    reference_phone_count: usize,
    phone_edits: usize,
    phone_error_rate: Option<f64>,
    in_vocabulary_cases: usize,
    exact_pronunciation_cases: usize,
    exact_pronunciation_rate: Option<f64>,
    syllable_boundary_cases_compared: usize,
    boundary_true_positive: usize,
    boundary_false_positive: usize,
    boundary_false_negative: usize,
    syllable_boundary_precision: Option<f64>,
    syllable_boundary_recall: Option<f64>,
    syllable_boundary_f1: Option<f64>,
    stress_correct: usize,
    stress_compared: usize,
    stress_accuracy: Option<f64>,
    oov_true_positive: usize,
    oov_false_positive: usize,
    oov_false_negative: usize,
    oov_precision: Option<f64>,
    oov_recall: Option<f64>,
    oov_f1: Option<f64>,
    oov_accuracy: Option<f64>,
    confidence_calibration: Option<ConfidenceCalibration>,
}

#[derive(Debug, Serialize)]
struct LanguageMetrics {
    language: String,
    cases: usize,
    in_vocabulary_cases: usize,
    reference_phone_count: usize,
    phone_edits: usize,
    phone_error_rate: Option<f64>,
    exact_pronunciation_cases: usize,
    exact_pronunciation_rate: Option<f64>,
    oov_true_positive: usize,
    oov_false_positive: usize,
    oov_false_negative: usize,
    oov_precision: Option<f64>,
    oov_recall: Option<f64>,
    confidence_calibration: Option<ConfidenceCalibration>,
}

#[derive(Debug, Serialize)]
struct ProfileMetrics {
    profile_id: String,
    cases: usize,
    in_vocabulary_cases: usize,
    reference_phone_count: usize,
    phone_edits: usize,
    phone_error_rate: Option<f64>,
    exact_pronunciation_cases: usize,
    exact_pronunciation_rate: Option<f64>,
    oov_true_positive: usize,
    oov_false_positive: usize,
    oov_false_negative: usize,
    oov_precision: Option<f64>,
    oov_recall: Option<f64>,
    confidence_calibration: Option<ConfidenceCalibration>,
}

#[derive(Debug, Serialize)]
struct EvaluationReport {
    schema: &'static str,
    metrics: AggregateMetrics,
    by_language: Vec<LanguageMetrics>,
    by_profile: Vec<ProfileMetrics>,
    cases: Vec<CaseScore>,
}

#[derive(Default)]
struct LanguageTotals {
    cases: usize,
    in_vocabulary_cases: usize,
    reference_phone_count: usize,
    phone_edits: usize,
    exact_pronunciation_cases: usize,
    oov_true_positive: usize,
    oov_false_positive: usize,
    oov_false_negative: usize,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn confidence_calibration(samples: &[(f64, bool)]) -> Option<ConfidenceCalibration> {
    if samples.is_empty() {
        return None;
    }
    const BIN_COUNT: usize = 10;
    let mut sums = [0.0f64; BIN_COUNT];
    let mut correct = [0usize; BIN_COUNT];
    let mut counts = [0usize; BIN_COUNT];
    let mut squared_error = 0.0f64;

    for &(confidence, is_correct) in samples {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            continue;
        }
        let bin = ((confidence * BIN_COUNT as f64).floor() as usize).min(BIN_COUNT - 1);
        sums[bin] += confidence;
        correct[bin] += usize::from(is_correct);
        counts[bin] += 1;
        let outcome = if is_correct { 1.0 } else { 0.0 };
        squared_error += (confidence - outcome).powi(2);
    }

    let observations = counts.iter().sum::<usize>();
    if observations == 0 {
        return None;
    }
    let mut expected_calibration_error = 0.0f64;
    let bins = (0..BIN_COUNT)
        .map(|index| {
            let count = counts[index];
            let mean_confidence = (count > 0).then(|| sums[index] / count as f64);
            let observed_accuracy = (count > 0).then(|| correct[index] as f64 / count as f64);
            if let (Some(confidence), Some(accuracy)) = (mean_confidence, observed_accuracy) {
                expected_calibration_error +=
                    (count as f64 / observations as f64) * (confidence - accuracy).abs();
            }
            CalibrationBin {
                lower_bound: index as f64 / BIN_COUNT as f64,
                upper_bound: (index + 1) as f64 / BIN_COUNT as f64,
                observations: count,
                mean_confidence,
                observed_accuracy,
            }
        })
        .collect();

    Some(ConfidenceCalibration {
        observations,
        brier_score: Some(squared_error / observations as f64),
        expected_calibration_error: Some(expected_calibration_error),
        bins,
    })
}


fn validate_candidate_syllable_info(
    phone_count: usize,
    boundaries: &Option<Vec<usize>>,
    stress: &Option<Vec<u8>>,
    label: &str,
) -> Result<(), String> {
    if let Some(boundaries) = boundaries {
        if phone_count == 0 || boundaries.is_empty() || *boundaries.last().unwrap() != phone_count {
            return Err(invalid(format!(
                "{label}: final exclusive syllable boundary must equal non-empty phone count"
            )));
        }
        let mut previous = 0usize;
        for (index, boundary) in boundaries.iter().copied().enumerate() {
            if boundary <= previous || boundary > phone_count {
                return Err(invalid(format!(
                    "{label}: syllable boundary {index} is not strictly increasing or exceeds phone count"
                )));
            }
            previous = boundary;
        }
        if let Some(stress) = stress {
            if stress.len() != boundaries.len() {
                return Err(invalid(format!(
                    "{label}: stress count {} does not match syllable count {}",
                    stress.len(),
                    boundaries.len()
                )));
            }
        }
    }
    if stress.is_some() && boundaries.is_none() {
        return Err(invalid(format!(
            "{label}: syllable stress requires explicit syllable boundaries"
        )));
    }
    if let Some(stress) = stress {
        if stress.iter().any(|value| *value > 2) {
            return Err(invalid(format!(
                "{label}: stress values must be 0, 1, or 2"
            )));
        }
    }
    Ok(())
}

fn validate_phone_sequence(
    phones: &[String],
    boundaries: &[usize],
    stress: &[u8],
    label: &str,
) -> Result<(), String> {
    if phones.iter().any(|phone| {
        phone.trim().is_empty()
            || phone.chars().any(char::is_whitespace)
            || matches!(phone.as_str(), "ˈ" | "ˌ" | "ː")
    }) {
        return Err(invalid(format!(
            "{label}: phone tokens must be non-empty normalized IPA units without standalone stress/length markers"
        )));
    }
    if phones.is_empty() {
        if !boundaries.is_empty() || !stress.is_empty() {
            return Err(invalid(format!(
                "{label}: empty phone sequence cannot have syllable boundaries or stress"
            )));
        }
        return Ok(());
    }
    if boundaries.is_empty() || *boundaries.last().unwrap() != phones.len() {
        return Err(invalid(format!(
            "{label}: final exclusive syllable boundary must equal phone count"
        )));
    }
    let mut previous = 0usize;
    for (index, boundary) in boundaries.iter().copied().enumerate() {
        if boundary <= previous || boundary > phones.len() {
            return Err(invalid(format!(
                "{label}: syllable boundary {index} is not strictly increasing or exceeds phone count"
            )));
        }
        previous = boundary;
    }
    if stress.len() != boundaries.len() {
        return Err(invalid(format!(
            "{label}: stress count {} does not match syllable count {}",
            stress.len(),
            boundaries.len()
        )));
    }
    if stress.iter().any(|value| *value > 2) {
        return Err(invalid(format!(
            "{label}: stress values must be 0, 1, or 2"
        )));
    }
    Ok(())
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<Vec<T>, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| invalid(format!("cannot read {}: {error}", path.display())))?;
    content
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|error| {
                invalid(format!(
                    "{}:{}: invalid JSON: {error}",
                    path.display(),
                    index + 1
                ))
            })
        })
        .collect()
}

fn boundary_set(boundaries: &[usize]) -> BTreeSet<usize> {
    boundaries
        .iter()
        .copied()
        .take(boundaries.len().saturating_sub(1))
        .collect()
}

fn boundary_counts(reference: &[usize], candidate: &[usize]) -> BoundaryCounts {
    let expected = boundary_set(reference);
    let observed = boundary_set(candidate);
    let tp = expected.intersection(&observed).count();
    let fp = observed.difference(&expected).count();
    let fn_count = expected.difference(&observed).count();
    let precision = (tp + fp > 0).then(|| tp as f64 / (tp + fp) as f64);
    let recall = (tp + fn_count > 0).then(|| tp as f64 / (tp + fn_count) as f64);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    BoundaryCounts {
        true_positive: tp,
        false_positive: fp,
        false_negative: fn_count,
        precision,
        recall,
        f1,
    }
}

fn align(reference: &[String], candidate: &[String]) -> (EditCounts, usize) {
    let n = reference.len();
    let m = candidate.len();
    let mut cost = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in cost.iter_mut().enumerate().take(n + 1) {
        row[0] = i;
    }
    for j in 0..=m {
        cost[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let substitution = if reference[i - 1] != candidate[j - 1] {
                1
            } else {
                0
            };
            cost[i][j] = (cost[i - 1][j - 1] + substitution)
                .min(cost[i - 1][j] + 1)
                .min(cost[i][j - 1] + 1);
        }
    }

    let mut edits = EditCounts {
        substitutions: 0,
        deletions: 0,
        insertions: 0,
    };
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        // Deterministic tie-break: diagonal, deletion, insertion.
        if i > 0 && j > 0 {
            let substitution = if reference[i - 1] != candidate[j - 1] {
                1
            } else {
                0
            };
            if cost[i][j] == cost[i - 1][j - 1] + substitution {
                edits.substitutions += substitution;
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && cost[i][j] == cost[i - 1][j] + 1 {
            edits.deletions += 1;
            i -= 1;
        } else {
            edits.insertions += 1;
            j -= 1;
        }
    }
    (edits, cost[n][m])
}

struct VariantScore<'a> {
    variant: &'a ReferenceVariant,
    edits: EditCounts,
    distance: usize,
    boundaries: Option<BoundaryCounts>,
    stress_correct: Option<usize>,
    stress_compared: Option<usize>,
}

fn score_variant<'a>(variant: &'a ReferenceVariant, candidate: &CandidateCase) -> VariantScore<'a> {
    let (edits, distance) = align(&variant.phones, &candidate.phones);
    // Boundary indices refer to phone positions. Once phone edits exist,
    // reference and candidate indices are no longer in the same coordinate
    // system. Report boundary/stress metrics as undefined rather than
    // comparing unrelated offsets.
    let boundaries = (distance == 0)
        .then(|| candidate.syllable_boundaries.as_ref())
        .flatten()
        .map(|observed| boundary_counts(&variant.syllable_boundaries, observed));
    let (stress_correct, stress_compared) = if distance == 0 {
        candidate
            .syllable_stress
            .as_ref()
            .filter(|observed| variant.syllable_stress.len() == observed.len())
            .map(|observed| {
                let compared = variant.syllable_stress.len();
                let correct = variant
                    .syllable_stress
                    .iter()
                    .zip(observed)
                    .filter(|(expected, actual)| expected == actual)
                    .count();
                (Some(correct), Some(compared))
            })
            .unwrap_or((None, None))
    } else {
        (None, None)
    };
    VariantScore {
        variant,
        edits,
        distance,
        boundaries,
        stress_correct,
        stress_compared,
    }
}

fn choose_variant<'a>(
    variants: &'a [ReferenceVariant],
    candidate: &CandidateCase,
) -> Option<VariantScore<'a>> {
    variants
        .iter()
        .map(|variant| score_variant(variant, candidate))
        .min_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then_with(|| {
                    let ae = a
                        .boundaries
                        .as_ref()
                        .map(|counts| counts.false_positive + counts.false_negative)
                        .unwrap_or(0);
                    let be = b
                        .boundaries
                        .as_ref()
                        .map(|counts| counts.false_positive + counts.false_negative)
                        .unwrap_or(0);
                    ae.cmp(&be)
                })
                .then_with(|| {
                    let ae = a.stress_compared.unwrap_or(0) - a.stress_correct.unwrap_or(0);
                    let be = b.stress_compared.unwrap_or(0) - b.stress_correct.unwrap_or(0);
                    ae.cmp(&be)
                })
        })
}

fn validate_inputs(
    references: &[ReferenceCase],
    predictions: &[CandidateCase],
) -> Result<(), String> {
    let mut reference_ids = BTreeSet::new();
    for reference in references {
        if reference.case_id.trim().is_empty()
            || reference.language.trim().is_empty()
            || reference.profile_id.trim().is_empty()
            || reference.text.trim().is_empty()
        {
            return Err("reference case ID, language, and text are required".into());
        }
        if !reference_ids.insert(reference.case_id.as_str()) {
            return Err(format!("duplicate reference case_id {}", reference.case_id));
        }
        if reference.expected_oov && !reference.accepted_variants.is_empty() {
            return Err(format!(
                "OOV reference {} must not contain accepted pronunciation variants",
                reference.case_id
            ));
        }
        if !reference.expected_oov && reference.accepted_variants.is_empty() {
            return Err(format!(
                "in-vocabulary reference {} requires at least one accepted variant",
                reference.case_id
            ));
        }
        let mut variant_ids = BTreeSet::new();
        for variant in &reference.accepted_variants {
            if variant.variant_id.trim().is_empty() || !variant_ids.insert(variant.variant_id.as_str()) {
                return Err(format!(
                    "reference {} has an empty or duplicate variant ID",
                    reference.case_id
                ));
            }
            if variant.phones.is_empty() {
                return Err(format!(
                    "in-vocabulary reference {} variant {} must contain phones",
                    reference.case_id, variant.variant_id
                ));
            }
            validate_phone_sequence(
                &variant.phones,
                &variant.syllable_boundaries,
                &variant.syllable_stress,
                &format!(
                    "reference {} variant {}",
                    reference.case_id, variant.variant_id
                ),
            )?;
        }
    }

    let mut prediction_ids = BTreeSet::new();
    let mut identity: Option<(&str, &str)> = None;
    for prediction in predictions {
        if prediction.case_id.trim().is_empty()
            || prediction.candidate_id.trim().is_empty()
            || prediction.source_id.trim().is_empty()
            || prediction.language.trim().is_empty()
            || prediction.profile_id.trim().is_empty()
        {
            return Err(
                "candidate case_id, candidate_id, source_id, language, and profile_id are required".into(),
            );
        }
        if !prediction_ids.insert(prediction.case_id.as_str()) {
            return Err(format!(
                "duplicate prediction case_id {}",
                prediction.case_id
            ));
        }
        if !reference_ids.contains(prediction.case_id.as_str()) {
            return Err(format!(
                "prediction {} has no reference case",
                prediction.case_id
            ));
        }
        let reference = references
            .iter()
            .find(|reference| reference.case_id == prediction.case_id)
            .expect("prediction ID was checked against reference IDs");
        if prediction.language != reference.language {
            return Err(format!(
                "prediction {} language {:?} does not match reference language {:?}",
                prediction.case_id, prediction.language, reference.language
            ));
        }
        if prediction.profile_id != reference.profile_id {
            return Err(format!(
                "prediction {} profile {:?} does not match reference profile {:?}",
                prediction.case_id, prediction.profile_id, reference.profile_id
            ));
        }
        if let Some((candidate_id, source_id)) = identity {
            if candidate_id != prediction.candidate_id || source_id != prediction.source_id {
                return Err(
                    "one predictions file must contain exactly one candidate/source identity"
                        .into(),
                );
            }
        } else {
            identity = Some((&prediction.candidate_id, &prediction.source_id));
        }
        if prediction.detected_oov {
            if !prediction.phones.is_empty()
                || prediction.syllable_boundaries.is_some()
                || prediction.syllable_stress.is_some()
            {
                return Err(format!(
                    "prediction {} marked OOV must not contain guessed phones or syllable metadata",
                    prediction.case_id
                ));
            }
        } else {
            if prediction.phones.is_empty() {
                return Err(format!(
                    "in-vocabulary prediction {} must contain phones",
                    prediction.case_id
                ));
            }
            if prediction.phones.iter().any(|phone| {
                phone.trim().is_empty()
                    || phone.chars().any(char::is_whitespace)
                    || matches!(phone.as_str(), "ˈ" | "ˌ" | "ː")
            }) {
                return Err(format!(
                    "prediction {} contains an invalid normalized IPA phone token",
                    prediction.case_id
                ));
            }
            validate_candidate_syllable_info(
                prediction.phones.len(),
                &prediction.syllable_boundaries,
                &prediction.syllable_stress,
                &format!("prediction {}", prediction.case_id),
            )?;
        }
        if prediction
            .confidence_exact
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(format!(
                "prediction {} has invalid exact-confidence probability",
                prediction.case_id
            ));
        }
    }
    if reference_ids != prediction_ids {
        let missing: Vec<_> = reference_ids.difference(&prediction_ids).copied().collect();
        return Err(format!(
            "candidate/reference case sets differ; missing predictions: {missing:?}"
        ));
    }
    if predictions.is_empty() {
        return Err("candidate predictions must not be empty".into());
    }
    Ok(())
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn f1(precision: Option<f64>, recall: Option<f64>) -> Option<f64> {
    match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
}

fn evaluate(
    references: &[ReferenceCase],
    predictions: &[CandidateCase],
) -> Result<EvaluationReport, String> {
    validate_inputs(references, predictions)?;
    let prediction_by_id: BTreeMap<_, _> = predictions
        .iter()
        .map(|prediction| (prediction.case_id.as_str(), prediction))
        .collect();

    let mut cases = Vec::with_capacity(references.len());
    let mut language_totals: BTreeMap<String, LanguageTotals> = BTreeMap::new();
    let mut profile_totals: BTreeMap<String, LanguageTotals> = BTreeMap::new();
    let mut total_reference_phones = 0usize;
    let mut total_edits = 0usize;
    let mut in_vocabulary_cases = 0usize;
    let mut exact_cases = 0usize;
    let mut boundary_cases_compared = 0usize;
    let mut boundary_tp = 0usize;
    let mut boundary_fp = 0usize;
    let mut boundary_fn = 0usize;
    let mut stress_correct_total = 0usize;
    let mut stress_compared_total = 0usize;
    let mut oov_tp = 0usize;
    let mut oov_fp = 0usize;
    let mut oov_fn = 0usize;
    let mut oov_correct = 0usize;
    let mut confidence_samples: Vec<(f64, bool)> = Vec::new();
    let mut language_confidence: BTreeMap<String, Vec<(f64, bool)>> = BTreeMap::new();
    let mut profile_confidence: BTreeMap<String, Vec<(f64, bool)>> = BTreeMap::new();

    for reference in references {
        let language = language_totals
            .entry(reference.language.clone())
            .or_default();
        language.cases += 1;
        let profile = profile_totals
            .entry(reference.profile_id.clone())
            .or_default();
        profile.cases += 1;
        let candidate = prediction_by_id[reference.case_id.as_str()];
        let selected = choose_variant(&reference.accepted_variants, candidate);
        let (
            reference_phones,
            reference_syllables,
            edits,
            distance,
            boundaries,
            stress_correct,
            stress_compared,
            variant_id,
        ) = if let Some(score) = selected {
            (
                score.variant.phones.len(),
                score.variant.syllable_boundaries.len(),
                score.edits,
                score.distance,
                score.boundaries,
                score.stress_correct,
                score.stress_compared,
                Some(score.variant.variant_id.clone()),
            )
        } else {
            (
                0,
                0,
                EditCounts {
                    substitutions: 0,
                    deletions: 0,
                    insertions: 0,
                },
                0,
                None,
                None,
                None,
                None,
            )
        };
        let exact = (!reference.expected_oov).then_some(distance == 0 && !candidate.detected_oov);
        let candidate_phone_count = candidate.phones.len();
        let per = (reference_phones > 0).then(|| distance as f64 / reference_phones as f64);
        // OOV cases are evaluated only by OOV detection. Guessed phones on an
        // OOV word must not distort conventional in-vocabulary PER.
        if !reference.expected_oov {
            total_reference_phones += reference_phones;
            total_edits += edits.total();
            language.reference_phone_count += reference_phones;
            language.phone_edits += edits.total();
            profile.reference_phone_count += reference_phones;
            profile.phone_edits += edits.total();
        }
        if let Some(exact) = exact {
            in_vocabulary_cases += 1;
            language.in_vocabulary_cases += 1;
            profile.in_vocabulary_cases += 1;
            if exact {
                exact_cases += 1;
                language.exact_pronunciation_cases += 1;
                profile.exact_pronunciation_cases += 1;
            }
        }
        if let Some(boundaries) = &boundaries {
            boundary_cases_compared += 1;
            boundary_tp += boundaries.true_positive;
            boundary_fp += boundaries.false_positive;
            boundary_fn += boundaries.false_negative;
        }
        if let (Some(correct), Some(compared)) = (stress_correct, stress_compared) {
            stress_correct_total += correct;
            stress_compared_total += compared;
        }
        oov_tp += if reference.expected_oov && candidate.detected_oov {
            1
        } else {
            0
        };
        oov_fp += if !reference.expected_oov && candidate.detected_oov {
            1
        } else {
            0
        };
        oov_fn += if reference.expected_oov && !candidate.detected_oov {
            1
        } else {
            0
        };
        language.oov_true_positive +=
            if reference.expected_oov && candidate.detected_oov { 1 } else { 0 };
        language.oov_false_positive +=
            if !reference.expected_oov && candidate.detected_oov { 1 } else { 0 };
        language.oov_false_negative +=
            if reference.expected_oov && !candidate.detected_oov { 1 } else { 0 };
        profile.oov_true_positive +=
            if reference.expected_oov && candidate.detected_oov { 1 } else { 0 };
        profile.oov_false_positive +=
            if !reference.expected_oov && candidate.detected_oov { 1 } else { 0 };
        profile.oov_false_negative +=
            if reference.expected_oov && !candidate.detected_oov { 1 } else { 0 };
        oov_correct += if reference.expected_oov == candidate.detected_oov {
            1
        } else {
            0
        };
        if let (Some(confidence), Some(exact)) = (candidate.confidence_exact, exact) {
            confidence_samples.push((confidence, exact));
            language_confidence
                .entry(reference.language.clone())
                .or_default()
                .push((confidence, exact));
            profile_confidence
                .entry(reference.profile_id.clone())
                .or_default()
                .push((confidence, exact));
        }
        cases.push(CaseScore {
            case_id: reference.case_id.clone(),
            language: reference.language.clone(),
            profile_id: reference.profile_id.clone(),
            selected_reference_variant: variant_id,
            reference_phone_count: reference_phones,
            candidate_phone_count,
            edits,
            phone_error_rate: per,
            exact_pronunciation: exact,
            syllable_count_reference: reference_syllables,
            syllable_count_candidate: candidate.syllable_boundaries.as_ref().map(Vec::len),
            boundaries,
            stress_correct,
            stress_compared,
            stress_accuracy: stress_correct
                .zip(stress_compared)
                .and_then(|(correct, total)| ratio(correct, total)),
            expected_oov: reference.expected_oov,
            detected_oov: candidate.detected_oov,
            oov_detection_correct: reference.expected_oov == candidate.detected_oov,
            confidence_exact: candidate.confidence_exact,
        });
    }

    let oov_precision = ratio(oov_tp, oov_tp + oov_fp);
    let oov_recall = ratio(oov_tp, oov_tp + oov_fn);
    let (candidate_id, source_id) = {
        let prediction = &predictions[0];
        (
            prediction.candidate_id.clone(),
            prediction.source_id.clone(),
        )
    };
    let by_language = language_totals
        .into_iter()
        .map(|(language, totals)| {
            let oov_precision = ratio(
                totals.oov_true_positive,
                totals.oov_true_positive + totals.oov_false_positive,
            );
            let oov_recall = ratio(
                totals.oov_true_positive,
                totals.oov_true_positive + totals.oov_false_negative,
            );
            let calibration = language_confidence
                .get(&language)
                .and_then(|samples| confidence_calibration(samples));
            LanguageMetrics {
                language,
                cases: totals.cases,
                in_vocabulary_cases: totals.in_vocabulary_cases,
                reference_phone_count: totals.reference_phone_count,
                phone_edits: totals.phone_edits,
                phone_error_rate: ratio(totals.phone_edits, totals.reference_phone_count),
                exact_pronunciation_cases: totals.exact_pronunciation_cases,
                exact_pronunciation_rate: ratio(
                    totals.exact_pronunciation_cases,
                    totals.in_vocabulary_cases,
                ),
                oov_true_positive: totals.oov_true_positive,
                oov_false_positive: totals.oov_false_positive,
                oov_false_negative: totals.oov_false_negative,
                oov_precision,
                oov_recall,
                confidence_calibration: calibration,
            }
        })
        .collect();
    let by_profile = profile_totals
        .into_iter()
        .map(|(profile_id, totals)| {
            let oov_precision = ratio(
                totals.oov_true_positive,
                totals.oov_true_positive + totals.oov_false_positive,
            );
            let oov_recall = ratio(
                totals.oov_true_positive,
                totals.oov_true_positive + totals.oov_false_negative,
            );
            let calibration = profile_confidence
                .get(&profile_id)
                .and_then(|samples| confidence_calibration(samples));
            ProfileMetrics {
                profile_id,
                cases: totals.cases,
                in_vocabulary_cases: totals.in_vocabulary_cases,
                reference_phone_count: totals.reference_phone_count,
                phone_edits: totals.phone_edits,
                phone_error_rate: ratio(totals.phone_edits, totals.reference_phone_count),
                exact_pronunciation_cases: totals.exact_pronunciation_cases,
                exact_pronunciation_rate: ratio(
                    totals.exact_pronunciation_cases,
                    totals.in_vocabulary_cases,
                ),
                oov_true_positive: totals.oov_true_positive,
                oov_false_positive: totals.oov_false_positive,
                oov_false_negative: totals.oov_false_negative,
                oov_precision,
                oov_recall,
                confidence_calibration: calibration,
            }
        })
        .collect();
    Ok(EvaluationReport {
        schema: "g2p-reference-evaluation/v1",
        metrics: AggregateMetrics {
            cases: cases.len(),
            candidate_id,
            source_id,
            reference_phone_count: total_reference_phones,
            phone_edits: total_edits,
            phone_error_rate: ratio(total_edits, total_reference_phones),
            in_vocabulary_cases,
            exact_pronunciation_cases: exact_cases,
            exact_pronunciation_rate: ratio(exact_cases, in_vocabulary_cases),
            syllable_boundary_cases_compared: boundary_cases_compared,
            boundary_true_positive: boundary_tp,
            boundary_false_positive: boundary_fp,
            boundary_false_negative: boundary_fn,
            syllable_boundary_precision: ratio(boundary_tp, boundary_tp + boundary_fp),
            syllable_boundary_recall: ratio(boundary_tp, boundary_tp + boundary_fn),
            syllable_boundary_f1: f1(
                ratio(boundary_tp, boundary_tp + boundary_fp),
                ratio(boundary_tp, boundary_tp + boundary_fn),
            ),
            stress_correct: stress_correct_total,
            stress_compared: stress_compared_total,
            stress_accuracy: ratio(stress_correct_total, stress_compared_total),
            oov_true_positive: oov_tp,
            oov_false_positive: oov_fp,
            oov_false_negative: oov_fn,
            oov_precision,
            oov_recall,
            oov_f1: f1(oov_precision, oov_recall),
            oov_accuracy: ratio(oov_correct, cases.len()),
            confidence_calibration: confidence_calibration(&confidence_samples),
        },
        by_language,
        by_profile,
        cases,
    })
}

fn argument(args: &[String], flag: &str) -> Result<PathBuf, String> {
    let index = args
        .iter()
        .position(|argument| argument == flag)
        .ok_or_else(|| format!("missing required argument {flag}"))?;
    args.get(index + 1)
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing path after {flag}"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let reference_path = argument(&args, "--references")?;
    let prediction_path = argument(&args, "--predictions")?;
    let output_path = argument(&args, "--output")?;
    let references: Vec<ReferenceCase> = read_jsonl(&reference_path)?;
    let predictions: Vec<CandidateCase> = read_jsonl(&prediction_path)?;
    let report = evaluate(&references, &predictions)?;
    let json = serde_json::to_vec_pretty(&report)
        .map_err(|error| format!("cannot serialize report: {error}"))?;
    fs::write(&output_path, json)
        .map_err(|error| format!("cannot write {}: {error}", output_path.display()))?;
    let summary = serde_json::to_string_pretty(&report.metrics)
        .map_err(|error| format!("cannot serialize metric summary: {error}"))?;
    println!("{summary}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("g2p-eval: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variant(id: &str, phones: &[&str], boundaries: &[usize], stress: &[u8]) -> ReferenceVariant {
        ReferenceVariant {
            variant_id: id.into(),
            phones: phones.iter().map(|phone| (*phone).into()).collect(),
            syllable_boundaries: boundaries.to_vec(),
            syllable_stress: stress.to_vec(),
        }
    }

    fn reference(
        case_id: &str,
        expected_oov: bool,
        variants: Vec<ReferenceVariant>,
    ) -> ReferenceCase {
        ReferenceCase {
            case_id: case_id.into(),
            language: "en".into(),
            profile_id: "en-US-general-v1".into(),
            text: case_id.into(),
            context: "test context".into(),
            expected_oov,
            accepted_variants: variants,
        }
    }

    fn candidate(
        case_id: &str,
        phones: &[&str],
        boundaries: &[usize],
        stress: &[u8],
        detected_oov: bool,
    ) -> CandidateCase {
        CandidateCase {
            case_id: case_id.into(),
            candidate_id: "candidate-a".into(),
            source_id: "test-source".into(),
            language: "en".into(),
            profile_id: "en-US-general-v1".into(),
            phones: phones.iter().map(|phone| (*phone).into()).collect(),
            syllable_boundaries: (!detected_oov).then(|| boundaries.to_vec()),
            syllable_stress: (!detected_oov).then(|| stress.to_vec()),
            detected_oov,
            confidence_exact: None,
        }
    }

    #[test]
    fn exact_candidate_passes_phone_boundary_stress_and_oov_metrics() {
        let refs = vec![
            reference(
                "hello",
                false,
                vec![variant("v1", &["h", "ɛ", "l", "oʊ"], &[2, 4], &[0, 1])],
            ),
            reference("xqzpl", true, vec![]),
        ];
        let preds = vec![
            candidate("hello", &["h", "ɛ", "l", "oʊ"], &[2, 4], &[0, 1], false),
            candidate("xqzpl", &[], &[], &[], true),
        ];
        let report = evaluate(&refs, &preds).unwrap();
        assert_eq!(report.metrics.phone_error_rate, Some(0.0));
        assert_eq!(report.metrics.exact_pronunciation_rate, Some(1.0));
        assert_eq!(report.metrics.syllable_boundary_f1, Some(1.0));
        assert_eq!(report.metrics.stress_accuracy, Some(1.0));
        assert_eq!(report.metrics.oov_f1, Some(1.0));
    }

    #[test]
    fn heteronym_context_accepts_any_reviewed_pronunciation_variant() {
        let refs = vec![reference(
            "read-past",
            false,
            vec![
                variant("present", &["ɹ", "iː", "d"], &[3], &[1]),
                variant("past", &["ɹ", "ɛ", "d"], &[3], &[1]),
            ],
        )];
        let preds = vec![candidate("read-past", &["ɹ", "ɛ", "d"], &[3], &[1], false)];
        let report = evaluate(&refs, &preds).unwrap();
        assert_eq!(
            report.cases[0].selected_reference_variant.as_deref(),
            Some("past")
        );
        assert_eq!(report.metrics.phone_error_rate, Some(0.0));
    }

    #[test]
    fn contextual_heteronym_reference_rejects_the_wrong_read_pronunciation() {
        let mut past = reference(
            "read-past",
            false,
            vec![variant("past", &["ɹ", "ɛ", "d"], &[3], &[1])],
        );
        past.text = "read".into();
        past.context = "I read it yesterday".into();

        let refs = vec![past];
        let preds = vec![candidate(
            "read-past",
            &["ɹ", "iː", "d"],
            &[3],
            &[1],
            false,
        )];
        let report = evaluate(&refs, &preds).unwrap();
        assert_eq!(
            report.cases[0].selected_reference_variant.as_deref(),
            Some("past")
        );
        assert_eq!(report.cases[0].edits.substitutions, 1);
        assert_eq!(report.cases[0].phone_error_rate, Some(1.0 / 3.0));
        assert_eq!(report.cases[0].exact_pronunciation, Some(false));
    }

    #[test]
    fn edit_counts_and_phone_error_rate_expose_substitutions_insertions_and_deletions() {
        let refs = vec![reference(
            "case",
            false,
            vec![variant("v1", &["k", "æ", "t"], &[3], &[1])],
        )];
        let preds = vec![candidate("case", &["k", "ɛ", "t", "s"], &[4], &[1], false)];
        let report = evaluate(&refs, &preds).unwrap();
        assert_eq!(report.cases[0].edits.substitutions, 1);
        assert_eq!(report.cases[0].edits.insertions, 1);
        assert_eq!(report.cases[0].edits.deletions, 0);
        assert!((report.metrics.phone_error_rate.unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn syllable_metrics_are_undefined_when_phone_alignment_has_edits() {
        let refs = vec![reference(
            "word",
            false,
            vec![variant("v1", &["k", "æ", "t"], &[3], &[1])],
        )];
        let preds = vec![candidate("word", &["k", "ɛ", "t", "s"], &[4], &[1], false)];
        let report = evaluate(&refs, &preds).unwrap();
        assert_eq!(report.metrics.phone_error_rate, Some(2.0 / 3.0));
        assert_eq!(report.metrics.syllable_boundary_cases_compared, 0);
        assert_eq!(report.metrics.syllable_boundary_f1, None);
        assert_eq!(report.metrics.stress_compared, 0);
        assert_eq!(report.metrics.stress_accuracy, None);
    }

    #[test]
    fn prediction_language_must_match_reference_case_language() {
        let refs = vec![reference(
            "word",
            false,
            vec![variant("v1", &["w", "ɜ", "d"], &[3], &[1])],
        )];
        let mut pred = candidate("word", &["w", "ɜ", "d"], &[3], &[1], false);
        pred.language = "zu-ZA".into();
        let error = validate_inputs(&refs, &[pred]).unwrap_err();
        assert!(error.contains("does not match reference language"), "{error}");
    }

    #[test]
    fn prediction_profile_must_match_reference_profile() {
        let refs = vec![reference(
            "word",
            false,
            vec![variant("v1", &["w", "ɜ", "d"], &[3], &[1])],
        )];
        let mut pred = candidate("word", &["w", "ɜ", "d"], &[3], &[1], false);
        pred.profile_id = "en-GB-southern-v1".into();
        let error = validate_inputs(&refs, &[pred]).unwrap_err();
        assert!(error.contains("does not match reference profile"), "{error}");
    }

    #[test]
    fn aggregate_report_contains_language_stratified_metrics() {
        let mut refs = vec![reference(
            "hello",
            false,
            vec![variant("v1", &["h", "ə", "l", "oʊ"], &[2, 4], &[0, 1])],
        )];
        refs[0].language = "en-US".into();
        refs[0].profile_id = "en-US-general-v1".into();
        let mut pred = candidate("hello", &["h", "ə", "l", "oʊ"], &[2, 4], &[0, 1], false);
        pred.language = "en-US".into();
        pred.profile_id = "en-US-general-v1".into();
        let report = evaluate(&refs, &[pred]).unwrap();
        assert_eq!(report.by_language.len(), 1);
        assert_eq!(report.by_language[0].language, "en-US");
        assert_eq!(report.by_language[0].phone_error_rate, Some(0.0));
        assert_eq!(report.by_language[0].exact_pronunciation_rate, Some(1.0));
        assert_eq!(report.by_profile.len(), 1);
        assert_eq!(report.by_profile[0].profile_id, "en-US-general-v1");
        assert_eq!(report.by_profile[0].phone_error_rate, Some(0.0));
        assert_eq!(report.cases[0].profile_id, "en-US-general-v1");
    }

    #[test]
    fn final_syllable_terminator_is_not_counted_as_an_internal_boundary() {
        let refs = vec![reference(
            "one",
            false,
            vec![variant("v1", &["w", "ʌ", "n"], &[3], &[1])],
        )];
        let preds = vec![candidate("one", &["w", "ʌ", "n"], &[3], &[1], false)];
        let report = evaluate(&refs, &preds).unwrap();
        assert_eq!(report.metrics.boundary_true_positive, 0);
        assert_eq!(report.metrics.syllable_boundary_f1, None);
    }

    #[test]
    fn missing_syllable_prediction_keeps_phone_error_defined_and_syllable_metrics_undefined() {
        let refs = vec![reference(
            "hello",
            false,
            vec![variant("v1", &["h", "ə", "l", "oʊ"], &[2, 4], &[0, 1])],
        )];
        let mut prediction = candidate("hello", &["h", "ə", "l", "oʊ"], &[], &[], false);
        prediction.syllable_boundaries = None;
        prediction.syllable_stress = None;
        let report = evaluate(&refs, &[prediction]).unwrap();
        assert_eq!(report.metrics.phone_error_rate, Some(0.0));
        assert_eq!(report.metrics.syllable_boundary_f1, None);
        assert_eq!(report.metrics.stress_accuracy, None);
        assert_eq!(report.cases[0].syllable_count_candidate, None);
    }

    #[test]
    fn malformed_boundaries_and_stress_fail_closed() {
        let refs = vec![reference(
            "word",
            false,
            vec![variant("v1", &["w", "ɜ", "d"], &[2], &[1])],
        )];
        let preds = vec![candidate("word", &["w", "ɜ", "d"], &[2], &[3], false)];
        assert!(validate_inputs(&refs, &preds).is_err());
    }

    #[test]
    fn duplicate_or_missing_predictions_are_rejected() {
        let refs = vec![reference(
            "word",
            false,
            vec![variant("v1", &["w", "ɜ", "d"], &[3], &[1])],
        )];
        let duplicate = vec![
            candidate("word", &["w", "ɜ", "d"], &[3], &[1], false),
            candidate("word", &["w", "ɜ", "d"], &[3], &[1], false),
        ];
        assert!(validate_inputs(&refs, &duplicate).is_err());
        assert!(validate_inputs(&refs, &[]).is_err());
    }

    #[test]
    fn oov_guessed_phones_affect_oov_detection_not_in_vocabulary_per() {
        let refs = vec![
            reference(
                "known",
                false,
                vec![variant("known-v1", &["w", "ɜ", "d"], &[3], &[1])],
            ),
            reference("oov", true, vec![]),
        ];
        let preds = vec![
            candidate("known", &["w", "ɜ", "d"], &[3], &[1], false),
            candidate("oov", &["k", "s", "z", "p", "l"], &[5], &[1], false),
        ];
        let report = evaluate(&refs, &preds).unwrap();
        assert_eq!(report.metrics.phone_error_rate, Some(0.0));
        assert_eq!(report.metrics.exact_pronunciation_rate, Some(1.0));
        assert_eq!(report.metrics.oov_false_negative, 1);
        assert_eq!(report.metrics.oov_recall, Some(0.0));
        assert_eq!(report.cases[1].phone_error_rate, None);
        assert_eq!(report.cases[1].candidate_phone_count, 5);
    }

    #[test]
    fn stress_without_boundaries_and_duplicate_variant_ids_are_rejected() {
        let duplicate_refs = vec![reference(
            "word",
            false,
            vec![
                variant("duplicate", &["w", "ɜ", "d"], &[3], &[1]),
                variant("duplicate", &["w", "ɜ", "d"], &[3], &[1]),
            ],
        )];
        let valid_prediction = vec![candidate("word", &["w", "ɜ", "d"], &[3], &[1], false)];
        assert!(validate_inputs(&duplicate_refs, &valid_prediction).is_err());

        let valid_refs = vec![reference(
            "word",
            false,
            vec![variant("valid", &["w", "ɜ", "d"], &[3], &[1])],
        )];
        let mut invalid = candidate("word", &["w", "ɜ", "d"], &[3], &[1], false);
        invalid.syllable_boundaries = None;
        assert!(validate_inputs(&valid_refs, &[invalid]).is_err());
    }

    #[test]
    fn confidence_calibration_reports_brier_ece_and_reliability_bins() {
        let refs = vec![
            reference(
                "correct",
                false,
                vec![variant("v1", &["w", "ɜ", "d"], &[3], &[1])],
            ),
            reference(
                "incorrect",
                false,
                vec![variant("v1", &["k", "æ", "t"], &[3], &[1])],
            ),
        ];
        let mut correct = candidate("correct", &["w", "ɜ", "d"], &[3], &[1], false);
        correct.confidence_exact = Some(0.8);
        let mut incorrect = candidate("incorrect", &["k", "ɛ", "t"], &[3], &[1], false);
        incorrect.confidence_exact = Some(0.2);
        let report = evaluate(&refs, &[correct, incorrect]).unwrap();
        let calibration = report.metrics.confidence_calibration.unwrap();
        assert_eq!(calibration.observations, 2);
        assert!((calibration.brier_score.unwrap() - 0.04).abs() < 1e-12);
        assert!((calibration.expected_calibration_error.unwrap() - 0.2).abs() < 1e-12);
        assert_eq!(calibration.bins.iter().map(|bin| bin.observations).sum::<usize>(), 2);
        assert!(report.by_profile[0].confidence_calibration.is_some());
    }

    #[test]
    fn exact_confidence_brier_score_is_computed_only_when_supplied() {
        let refs = vec![reference(
            "word",
            false,
            vec![variant("v1", &["w", "ɜ", "d"], &[3], &[1])],
        )];
        let mut pred = candidate("word", &["w", "ɜ", "d"], &[3], &[1], false);
        pred.confidence_exact = Some(0.8);
        let report = evaluate(&refs, &[pred]).unwrap();
        let calibration = report.metrics.confidence_calibration.unwrap();
        assert!((calibration.brier_score.unwrap() - 0.04).abs() < 1e-12);
        assert!((calibration.expected_calibration_error.unwrap() - 0.2).abs() < 1e-12);
    }
}
