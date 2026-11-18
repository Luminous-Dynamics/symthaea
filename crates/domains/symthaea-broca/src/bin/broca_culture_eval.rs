// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic scorer for the original Broca literary/philosophy development seed.
//!
//! This is a structural development instrument, not an aesthetic judge or SOTA claim.
//! Evidence spans must quote the supplied source exactly. Claim and interpretation IDs
//! are human-authored rubric labels. Human expert evaluation remains necessary.

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Provenance {
    origin: String,
    author: String,
    license: String,
}

#[derive(Debug, Deserialize)]
struct Gold {
    required_evidence: Vec<String>,
    required_claim_ids: Vec<String>,
    acceptable_interpretation_ids: Vec<String>,
    ambiguity_expected: bool,
}

#[derive(Debug, Deserialize)]
struct EvalCase {
    schema_version: u32,
    id: String,
    track: String,
    language: String,
    split: String,
    source_text: String,
    prompt: String,
    provenance: Provenance,
    gold: Gold,
}

#[derive(Debug, Deserialize)]
struct Prediction {
    id: String,
    #[serde(default)]
    evidence_spans: Vec<String>,
    #[serde(default)]
    claim_ids: Vec<String>,
    #[serde(default)]
    interpretation_ids: Vec<String>,
    ambiguity_acknowledged: bool,
}

#[derive(Debug, Serialize)]
struct DatasetSummary {
    status: &'static str,
    schema_version: u32,
    cases: usize,
    tracks: BTreeMap<String, usize>,
    languages: BTreeMap<String, usize>,
    split: String,
    provenance: &'static str,
    limitation: &'static str,
}

#[derive(Debug, Default)]
struct Counts {
    cases: usize,
    predictions: usize,
    missing_predictions: usize,
    grounded_evidence: usize,
    predicted_evidence: usize,
    matched_evidence: usize,
    required_evidence: usize,
    correct_claims: usize,
    predicted_claims: usize,
    required_claims: usize,
    accepted_interpretations: usize,
    required_interpretations: usize,
    correct_ambiguity: usize,
}

#[derive(Debug, Serialize)]
struct ScoreSummary {
    cases: usize,
    predictions: usize,
    missing_predictions: usize,
    evidence_grounding_precision: Option<f64>,
    evidence_anchor_recall: Option<f64>,
    claim_precision: Option<f64>,
    claim_recall: Option<f64>,
    claim_f1: Option<f64>,
    interpretation_coverage_recall: Option<f64>,
    ambiguity_accuracy: Option<f64>,
}

#[derive(Debug, Serialize)]
struct EvaluationReport {
    status: &'static str,
    dataset_cases: usize,
    predictions: usize,
    overall: ScoreSummary,
    by_track: BTreeMap<String, ScoreSummary>,
    scoring_note: &'static str,
}

fn read_jsonl<T: DeserializeOwned>(path: &PathBuf) -> Result<Vec<T>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading JSONL file {}", path.display()))?;
    content
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("parsing {} line {}", path.display(), index + 1))
        })
        .collect()
}

fn unique_nonempty(values: &[String], label: &str, case_id: &str) -> Result<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        ensure!(!value.trim().is_empty(), "{case_id}: empty {label}");
        ensure!(seen.insert(value), "{case_id}: duplicate {label} {value:?}");
    }
    Ok(())
}

fn validate_dataset(cases: &[EvalCase]) -> Result<DatasetSummary> {
    ensure!(!cases.is_empty(), "dataset is empty");
    let mut ids = BTreeSet::new();
    let mut tracks = BTreeMap::new();
    let mut languages = BTreeMap::new();
    let mut split: Option<&str> = None;

    for case in cases {
        ensure!(case.schema_version == 1, "{}: unsupported schema_version {}", case.id, case.schema_version);
        ensure!(ids.insert(&case.id), "duplicate case id {}", case.id);
        ensure!(!case.id.trim().is_empty(), "case id must not be empty");
        ensure!(!case.track.trim().is_empty(), "{}: track must not be empty", case.id);
        ensure!(!case.language.trim().is_empty(), "{}: language must not be empty", case.id);
        ensure!(!case.prompt.trim().is_empty(), "{}: prompt must not be empty", case.id);
        ensure!(!case.source_text.trim().is_empty(), "{}: source_text must not be empty", case.id);
        ensure!(case.split == "dev", "{}: seed corpus must remain dev-only, got {:?}", case.id, case.split);
        if let Some(existing) = split {
            ensure!(existing == case.split, "mixed dataset splits are not allowed");
        } else {
            split = Some(&case.split);
        }

        ensure!(
            case.provenance.origin == "original-authored-fixture"
                && case.provenance.license == "CC0-1.0"
                && !case.provenance.author.trim().is_empty(),
            "{}: expected an original CC0 fixture with an attributed author field",
            case.id
        );
        ensure!(!case.gold.required_evidence.is_empty(), "{}: no evidence anchors", case.id);
        for (index, evidence) in case.gold.required_evidence.iter().enumerate() {
            ensure!(
                case.source_text.contains(evidence),
                "{}: gold evidence anchor {index} is not an exact substring of source_text",
                case.id
            );
        }
        unique_nonempty(&case.gold.required_evidence, "gold evidence anchor", &case.id)?;
        unique_nonempty(&case.gold.required_claim_ids, "required claim id", &case.id)?;
        unique_nonempty(
            &case.gold.acceptable_interpretation_ids,
            "acceptable interpretation id",
            &case.id,
        )?;

        *tracks.entry(case.track.clone()).or_insert(0) += 1;
        *languages.entry(case.language.clone()).or_insert(0) += 1;
    }

    Ok(DatasetSummary {
        status: "dataset_validated",
        schema_version: 1,
        cases: cases.len(),
        tracks,
        languages,
        split: split.unwrap_or("dev").to_string(),
        provenance: "original-authored-fixtures; CC0-1.0",
        limitation: "development seed only; not a representative sample and not a leaderboard result",
    })
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn summarize(counts: &Counts) -> ScoreSummary {
    let precision = ratio(counts.grounded_evidence, counts.predicted_evidence);
    let claim_precision = ratio(counts.correct_claims, counts.predicted_claims);
    let claim_recall = ratio(counts.correct_claims, counts.required_claims);
    let claim_f1 = match (claim_precision, claim_recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };

    ScoreSummary {
        cases: counts.cases,
        predictions: counts.predictions,
        missing_predictions: counts.missing_predictions,
        evidence_grounding_precision: precision,
        evidence_anchor_recall: ratio(counts.matched_evidence, counts.required_evidence),
        claim_precision,
        claim_recall,
        claim_f1,
        interpretation_coverage_recall: ratio(
            counts.accepted_interpretations,
            counts.required_interpretations,
        ),
        ambiguity_accuracy: ratio(counts.correct_ambiguity, counts.predictions),
    }
}

fn score_predictions(cases: &[EvalCase], predictions: &[Prediction]) -> Result<EvaluationReport> {
    let case_by_id: BTreeMap<&str, &EvalCase> =
        cases.iter().map(|case| (case.id.as_str(), case)).collect();
    let mut prediction_by_id = BTreeMap::new();

    for prediction in predictions {
        ensure!(
            prediction_by_id.insert(prediction.id.as_str(), prediction).is_none(),
            "duplicate prediction id {}",
            prediction.id
        );
        ensure!(
            case_by_id.contains_key(prediction.id.as_str()),
            "prediction references unknown case id {}",
            prediction.id
        );
    }

    let mut overall = Counts {
        cases: cases.len(),
        ..Counts::default()
    };
    let mut by_track: BTreeMap<String, Counts> = BTreeMap::new();

    for case in cases {
        let track = by_track.entry(case.track.clone()).or_default();
        track.cases += 1;

        overall.required_evidence += case.gold.required_evidence.len();
        overall.required_claims += case.gold.required_claim_ids.len();
        overall.required_interpretations += case.gold.acceptable_interpretation_ids.len();
        track.required_evidence += case.gold.required_evidence.len();
        track.required_claims += case.gold.required_claim_ids.len();
        track.required_interpretations += case.gold.acceptable_interpretation_ids.len();

        let Some(prediction) = prediction_by_id.get(case.id.as_str()) else {
            overall.missing_predictions += 1;
            track.missing_predictions += 1;
            continue;
        };

        overall.predictions += 1;
        track.predictions += 1;
        overall.correct_ambiguity += usize::from(
            prediction.ambiguity_acknowledged == case.gold.ambiguity_expected
        );
        track.correct_ambiguity += usize::from(
            prediction.ambiguity_acknowledged == case.gold.ambiguity_expected
        );

        let predicted_evidence: BTreeSet<&str> = prediction
            .evidence_spans
            .iter()
            .map(|span| span.trim())
            .filter(|span| !span.is_empty())
            .collect();
        let grounded = predicted_evidence
            .iter()
            .filter(|span| case.source_text.contains(**span))
            .count();
        let matched = case
            .gold
            .required_evidence
            .iter()
            .filter(|anchor| predicted_evidence.contains(anchor.as_str()))
            .count();

        overall.grounded_evidence += grounded;
        overall.predicted_evidence += predicted_evidence.len();
        overall.matched_evidence += matched;
        track.grounded_evidence += grounded;
        track.predicted_evidence += predicted_evidence.len();
        track.matched_evidence += matched;

        let required_claims: BTreeSet<&str> =
            case.gold.required_claim_ids.iter().map(String::as_str).collect();
        let predicted_claims: BTreeSet<&str> =
            prediction.claim_ids.iter().map(String::as_str).collect();
        let claim_hits = required_claims.intersection(&predicted_claims).count();
        overall.correct_claims += claim_hits;
        overall.predicted_claims += predicted_claims.len();
        track.correct_claims += claim_hits;
        track.predicted_claims += predicted_claims.len();

        let acceptable_interpretations: BTreeSet<&str> = case
            .gold
            .acceptable_interpretation_ids
            .iter()
            .map(String::as_str)
            .collect();
        let predicted_interpretations: BTreeSet<&str> = prediction
            .interpretation_ids
            .iter()
            .map(String::as_str)
            .collect();
        let interpretation_hits = acceptable_interpretations
            .intersection(&predicted_interpretations)
            .count();
        overall.accepted_interpretations += interpretation_hits;
        track.accepted_interpretations += interpretation_hits;
    }

    let by_track = by_track
        .into_iter()
        .map(|(track, counts)| (track, summarize(&counts)))
        .collect();

    Ok(EvaluationReport {
        status: "scored",
        dataset_cases: cases.len(),
        predictions: predictions.len(),
        overall: summarize(&overall),
        by_track,
        scoring_note: "exact quote grounding and rubric-ID coverage only; does not score prose quality, truth, originality, or expert-level philosophical adequacy",
    })
}

fn run() -> Result<()> {
    let mut dataset_path = None;
    let mut predictions_path = None;
    let mut output_path = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => dataset_path = args.next().map(PathBuf::from),
            "--predictions" => predictions_path = args.next().map(PathBuf::from),
            "--output" => output_path = args.next().map(PathBuf::from),
            "--help" | "-h" => {
                println!(
                    "broca-culture-eval --dataset <seed.jsonl> [--predictions <outputs.jsonl>] [--output <report.json>]"
                );
                return Ok(());
            }
            other => bail!("unknown argument {other:?}; use --help"),
        }
    }

    let dataset_path = dataset_path.context("--dataset is required")?;
    let cases: Vec<EvalCase> = read_jsonl(&dataset_path)?;
    let validation = validate_dataset(&cases)?;

    let report = if let Some(predictions_path) = predictions_path {
        let predictions: Vec<Prediction> = read_jsonl(&predictions_path)?;
        serde_json::to_string_pretty(&score_predictions(&cases, &predictions)?)?
    } else {
        serde_json::to_string_pretty(&validation)?
    };

    if let Some(path) = output_path {
        std::fs::write(&path, &report)
            .with_context(|| format!("writing report {}", path.display()))?;
    } else {
        println!("{report}");
    }

    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("broca-culture-eval: {error:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_case() -> EvalCase {
        EvalCase {
            schema_version: 1,
            id: "fixture-1".into(),
            track: "literary_interpretation".into(),
            language: "en".into(),
            split: "dev".into(),
            source_text: "A silver thread held the bowl together.".into(),
            prompt: "Interpret the repair.".into(),
            provenance: Provenance {
                origin: "original-authored-fixture".into(),
                author: "test".into(),
                license: "CC0-1.0".into(),
            },
            gold: Gold {
                required_evidence: vec!["A silver thread held the bowl together.".into()],
                required_claim_ids: vec!["repair-visible".into()],
                acceptable_interpretation_ids: vec!["care".into()],
                ambiguity_expected: true,
            },
        }
    }

    #[test]
    fn dataset_validator_accepts_original_dev_fixture() {
        let summary = validate_dataset(&[fixture_case()]).unwrap();
        assert_eq!(summary.cases, 1);
        assert_eq!(summary.split, "dev");
    }

    #[test]
    fn scorer_rewards_grounded_quotes_and_expected_rubric_ids() {
        let prediction = Prediction {
            id: "fixture-1".into(),
            evidence_spans: vec![
                "A silver thread held the bowl together.".into(),
                "The bowl was made of gold.".into(),
            ],
            claim_ids: vec!["repair-visible".into()],
            interpretation_ids: vec!["care".into()],
            ambiguity_acknowledged: true,
        };
        let report = score_predictions(&[fixture_case()], &[prediction]).unwrap();
        assert_eq!(report.overall.evidence_grounding_precision, Some(0.5));
        assert_eq!(report.overall.evidence_anchor_recall, Some(1.0));
        assert_eq!(report.overall.claim_f1, Some(1.0));
        assert_eq!(report.overall.interpretation_coverage_recall, Some(1.0));
        assert_eq!(report.overall.ambiguity_accuracy, Some(1.0));
    }

    #[test]
    fn scorer_rejects_unknown_prediction_ids() {
        let prediction = Prediction {
            id: "not-in-the-seed".into(),
            evidence_spans: vec![],
            claim_ids: vec![],
            interpretation_ids: vec![],
            ambiguity_acknowledged: false,
        };
        assert!(score_predictions(&[fixture_case()], &[prediction]).is_err());
    }
}
