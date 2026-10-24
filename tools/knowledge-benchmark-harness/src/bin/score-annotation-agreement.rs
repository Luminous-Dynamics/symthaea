use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const TOOL_VERSION: &str = "score-annotation-agreement-v0.1.0";
const REPORT_SCHEMA: &str =
    include_str!("../../../../benchmarks/knowledge-annotation/v0/agreement-report.schema.json");

#[derive(Debug)]
struct Args {
    annotations: PathBuf,
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Annotation {
    annotation_id: String,
    annotator_id: String,
    case_id: String,
    answer_id: String,
    annotation_role: String,
    review_status: String,
    answerability: String,
    abstention_assessment: String,
    claim_annotations: Vec<ClaimAnnotation>,
}

#[derive(Debug, Deserialize)]
struct ClaimAnnotation {
    claim_id: String,
    support_label: String,
    epistemic_lane: String,
    severity: String,
}

#[derive(Debug, Clone)]
struct Rating {
    unit_id: String,
    annotator_id: String,
    label: String,
}

#[derive(Debug, Serialize)]
struct AgreementMetric {
    field: &'static str,
    unit_kind: &'static str,
    eligible_unit_count: usize,
    excluded_single_rater_units: usize,
    rating_count: usize,
    category_count: usize,
    exact_agreement_rate: Option<f64>,
    krippendorff_alpha_nominal: Option<f64>,
    interpretation: &'static str,
}

#[derive(Debug, Serialize)]
struct Report {
    report_schema_version: &'static str,
    tool_version: &'static str,
    qualification_only: bool,
    annotation_count: usize,
    reviewer_count: usize,
    annotation_sha256: String,
    dependency_lock_sha256: String,
    metrics: Vec<AgreementMetric>,
    nonclaims: Vec<&'static str>,
}

fn usage() -> &'static str {
    "Usage: score-annotation-agreement --annotations FILE.jsonl --out FILE.json\n\
     Computes multi-rater exact agreement and nominal Krippendorff alpha for independent annotations.\n\
     It does not decide historical truth, adjudicate disagreement, or score model quality."
}

fn parse_args() -> Result<Args, String> {
    let mut raw = env::args().skip(1);
    let mut annotations = None;
    let mut output = None;
    while let Some(arg) = raw.next() {
        let value = |raw: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            raw.next().ok_or_else(|| format!("{flag} requires a value"))
        };
        match arg.as_str() {
            "--annotations" => {
                annotations = Some(PathBuf::from(value(&mut raw, "--annotations")?))
            }
            "--out" => output = Some(PathBuf::from(value(&mut raw, "--out")?)),
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }
    Ok(Args {
        annotations: annotations
            .ok_or_else(|| format!("missing --annotations\n{}", usage()))?,
        output: output.ok_or_else(|| format!("missing --out\n{}", usage()))?,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn unit_id(case_id: &str, answer_id: &str, claim_id: Option<&str>) -> String {
    match claim_id {
        Some(claim_id) => format!("{case_id}\u{1f}{answer_id}\u{1f}{claim_id}"),
        None => format!("{case_id}\u{1f}{answer_id}"),
    }
}

fn parse_annotations(bytes: &[u8]) -> Result<Vec<Annotation>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("annotation input is not UTF-8: {error}"))?;
    let mut annotations = Vec::new();
    let mut annotation_ids = HashSet::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!(
                "annotation input contains an empty JSONL line at {}",
                index + 1
            ));
        }
        let annotation: Annotation = serde_json::from_str(line).map_err(|error| {
            format!("annotation input has invalid JSON at line {}: {error}", index + 1)
        })?;
        if annotation.annotation_id.trim().is_empty()
            || annotation.annotator_id.trim().is_empty()
            || annotation.case_id.trim().is_empty()
            || annotation.answer_id.trim().is_empty()
        {
            return Err(format!(
                "annotation line {} requires non-empty annotation_id, annotator_id, case_id and answer_id",
                index + 1
            ));
        }
        if !annotation_ids.insert(annotation.annotation_id.clone()) {
            return Err(format!(
                "duplicate annotation_id {:?}",
                annotation.annotation_id
            ));
        }
        let mut claim_ids = HashSet::new();
        for claim in &annotation.claim_annotations {
            if claim.claim_id.trim().is_empty() {
                return Err(format!(
                    "annotation {:?} contains an empty claim_id",
                    annotation.annotation_id
                ));
            }
            if !claim_ids.insert(claim.claim_id.as_str()) {
                return Err(format!(
                    "annotation {:?} contains duplicate claim_id {:?}",
                    annotation.annotation_id, claim.claim_id
                ));
            }
        }
        annotations.push(annotation);
    }
    if annotations.is_empty() {
        return Err("annotation input contains no records".into());
    }
    Ok(annotations)
}

fn validate_review_set(annotations: &[Annotation]) -> Result<bool, String> {
    let synthetic_count = annotations
        .iter()
        .filter(|annotation| annotation.annotation_role == "synthetic_fixture")
        .count();
    if synthetic_count > 0 && synthetic_count != annotations.len() {
        return Err(
            "synthetic_fixture and real reviewer annotations cannot be mixed in one agreement report"
                .into(),
        );
    }
    if synthetic_count == annotations.len() {
        return Ok(true);
    }

    let allowed_roles = ["independent_reviewer", "subject_expert", "calibration_reviewer"];
    for annotation in annotations {
        if !allowed_roles.contains(&annotation.annotation_role.as_str()) {
            return Err(format!(
                "real agreement scoring requires an independent review role; found {:?}",
                annotation.annotation_role
            ));
        }
        if annotation.review_status != "independent" {
            return Err(format!(
                "real agreement scoring only accepts pre-adjudication independent labels; annotation {:?} has review_status {:?}",
                annotation.annotation_id, annotation.review_status
            ));
        }
    }
    Ok(false)
}

fn collect_answer_ratings(
    annotations: &[Annotation],
    field: &'static str,
) -> Result<Vec<Rating>, String> {
    let mut ratings = Vec::new();
    for annotation in annotations {
        let label = match field {
            "answerability" => annotation.answerability.as_str(),
            "abstention_assessment" => annotation.abstention_assessment.as_str(),
            _ => return Err(format!("unsupported answer-level field {field:?}")),
        };
        if label.is_empty() {
            return Err(format!(
                "annotation {:?} has empty {field}",
                annotation.annotation_id
            ));
        }
        ratings.push(Rating {
            unit_id: unit_id(&annotation.case_id, &annotation.answer_id, None),
            annotator_id: annotation.annotator_id.clone(),
            label: label.to_string(),
        });
    }
    Ok(ratings)
}

fn collect_claim_ratings(
    annotations: &[Annotation],
    field: &'static str,
) -> Result<Vec<Rating>, String> {
    let mut ratings = Vec::new();
    for annotation in annotations {
        for claim in &annotation.claim_annotations {
            let label = match field {
                "support_label" => claim.support_label.as_str(),
                "epistemic_lane" => claim.epistemic_lane.as_str(),
                "severity" => claim.severity.as_str(),
                _ => return Err(format!("unsupported claim-level field {field:?}")),
            };
            if label.is_empty() {
                return Err(format!(
                    "annotation {:?} claim {:?} has empty {field}",
                    annotation.annotation_id, claim.claim_id
                ));
            }
            ratings.push(Rating {
                unit_id: unit_id(
                    &annotation.case_id,
                    &annotation.answer_id,
                    Some(&claim.claim_id),
                ),
                annotator_id: annotation.annotator_id.clone(),
                label: label.to_string(),
            });
        }
    }
    Ok(ratings)
}

fn agreement_metric(
    field: &'static str,
    unit_kind: &'static str,
    ratings: Vec<Rating>,
) -> Result<AgreementMetric, String> {
    let mut units: BTreeMap<String, Vec<Rating>> = BTreeMap::new();
    for rating in ratings {
        units.entry(rating.unit_id.clone()).or_default().push(rating);
    }

    let mut eligible = Vec::<Vec<Rating>>::new();
    let mut excluded_single_rater_units = 0usize;
    for (unit, unit_ratings) in units {
        let mut annotators = HashSet::new();
        for rating in &unit_ratings {
            if !annotators.insert(rating.annotator_id.as_str()) {
                return Err(format!(
                    "unit {unit:?} has duplicate rating from annotator {:?} for {field}",
                    rating.annotator_id
                ));
            }
        }
        if unit_ratings.len() >= 2 {
            eligible.push(unit_ratings);
        } else {
            excluded_single_rater_units += 1;
        }
    }

    let eligible_unit_count = eligible.len();
    let rating_count: usize = eligible.iter().map(Vec::len).sum();
    let mut category_counts: HashMap<&str, usize> = HashMap::new();
    let mut observed_disagreements = 0usize;
    let mut possible_within_unit_pairs = 0usize;
    let mut exact_matches = 0usize;

    for unit_ratings in &eligible {
        let mut within_counts: HashMap<&str, usize> = HashMap::new();
        for rating in unit_ratings {
            *within_counts.entry(rating.label.as_str()).or_default() += 1;
            *category_counts.entry(rating.label.as_str()).or_default() += 1;
        }
        let n = unit_ratings.len();
        let same_ordered_pairs: usize = within_counts.values().map(|count| count * count).sum();
        observed_disagreements += n * n - same_ordered_pairs;
        possible_within_unit_pairs += n * (n - 1);
        exact_matches += usize::from(within_counts.len() == 1);
    }

    let observed_disagreement = if possible_within_unit_pairs == 0 {
        None
    } else {
        Some(observed_disagreements as f64 / possible_within_unit_pairs as f64)
    };
    let expected_disagreement = if rating_count < 2 {
        None
    } else {
        let same_ordered_pairs: usize =
            category_counts.values().map(|count| count * count).sum();
        let all_ordered_pairs = rating_count * (rating_count - 1);
        Some((rating_count * rating_count - same_ordered_pairs) as f64 / all_ordered_pairs as f64)
    };
    let exact_agreement_rate = if eligible_unit_count == 0 {
        None
    } else {
        Some(exact_matches as f64 / eligible_unit_count as f64)
    };
    let (krippendorff_alpha_nominal, interpretation) = match (
        observed_disagreement,
        expected_disagreement,
    ) {
        (Some(observed), Some(expected)) if expected > 0.0 => {
            (Some(1.0 - observed / expected), "estimated")
        }
        (Some(_), Some(_)) => (None, "undefined_no_category_variation"),
        _ => (None, "insufficient_multi_reviewed_units"),
    };

    Ok(AgreementMetric {
        field,
        unit_kind,
        eligible_unit_count,
        excluded_single_rater_units,
        rating_count,
        category_count: category_counts.len(),
        exact_agreement_rate,
        krippendorff_alpha_nominal,
        interpretation,
    })
}

fn compute_metrics(annotations: &[Annotation]) -> Result<Vec<AgreementMetric>, String> {
    Ok(vec![
        agreement_metric(
            "answerability",
            "answer",
            collect_answer_ratings(annotations, "answerability")?,
        )?,
        agreement_metric(
            "abstention_assessment",
            "answer",
            collect_answer_ratings(annotations, "abstention_assessment")?,
        )?,
        agreement_metric(
            "support_label",
            "claim",
            collect_claim_ratings(annotations, "support_label")?,
        )?,
        agreement_metric(
            "epistemic_lane",
            "claim",
            collect_claim_ratings(annotations, "epistemic_lane")?,
        )?,
        agreement_metric(
            "severity",
            "claim",
            collect_claim_ratings(annotations, "severity")?,
        )?,
    ])
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let annotation_bytes = fs::read(&args.annotations).map_err(|error| {
        format!(
            "cannot read annotations {}: {error}",
            args.annotations.display()
        )
    })?;
    let annotations = parse_annotations(&annotation_bytes)?;
    let qualification_only = validate_review_set(&annotations)?;
    let metrics = compute_metrics(&annotations)?;

    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let lock_bytes = fs::read(&lock_path).map_err(|error| {
        format!(
            "cannot read dependency lockfile {}: {error}",
            lock_path.display()
        )
    })?;

    let report = Report {
        report_schema_version: "annotation-agreement-report-v0",
        tool_version: TOOL_VERSION,
        qualification_only,
        annotation_count: annotations.len(),
        reviewer_count: annotations
            .iter()
            .map(|annotation| annotation.annotator_id.as_str())
            .collect::<HashSet<_>>()
            .len(),
        annotation_sha256: sha256_hex(&annotation_bytes),
        dependency_lock_sha256: sha256_hex(&lock_bytes),
        metrics,
        nonclaims: vec![
            "Agreement measures consistency between labels, not historical truth, cultural legitimacy, or scholarly correctness.",
            "Synthetic-fixture agreement is a scorer qualification artifact, not real annotator reliability.",
            "Single-rater units are excluded from alpha and exact agreement; their count is reported.",
            "Nominal Krippendorff alpha is undefined when the eligible ratings have no category variation.",
            "Adjudicated labels are not used to inflate pre-adjudication reliability.",
        ],
    };

    let report_value = serde_json::to_value(&report)
        .map_err(|error| format!("cannot serialize agreement report: {error}"))?;
    let schema: Value = serde_json::from_str(REPORT_SCHEMA)
        .map_err(|error| format!("bundled agreement schema is invalid JSON: {error}"))?;
    let meta_validator = jsonschema::meta::validator_for(&schema)
        .map_err(|error| format!("cannot compile agreement schema meta-validator: {error}"))?;
    if !meta_validator.evaluate(&schema).flag().valid {
        return Err("bundled annotation agreement schema is not valid Draft 2020-12".into());
    }
    let validator = jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| format!("cannot compile agreement report schema: {error}"))?;
    let errors: Vec<String> = validator
        .iter_errors(&report_value)
        .map(|error| error.to_string())
        .collect();
    if !errors.is_empty() {
        return Err(format!(
            "agreement report violates its schema: {}",
            errors.join("; ")
        ));
    }

    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create output directory {}: {error}",
                parent.display()
            )
        })?;
    }
    fs::write(
        &args.output,
        serde_json::to_vec_pretty(&report_value)
            .map_err(|error| format!("cannot encode agreement report: {error}"))?,
    )
    .map_err(|error| format!("cannot write agreement report {}: {error}", args.output.display()))?;
    println!(
        "Scored {} annotations; qualification_only={}; report={}",
        report.annotation_count,
        report.qualification_only,
        args.output.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("score-annotation-agreement: {error}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_annotation(
        annotation_id: &str,
        annotator: &str,
        answerability: &str,
        support: &str,
    ) -> Annotation {
        Annotation {
            annotation_id: annotation_id.into(),
            annotator_id: annotator.into(),
            case_id: "case-1".into(),
            answer_id: "answer-1".into(),
            annotation_role: "synthetic_fixture".into(),
            review_status: "independent".into(),
            answerability: answerability.into(),
            abstention_assessment: "not_applicable".into(),
            claim_annotations: vec![ClaimAnnotation {
                claim_id: "claim-1".into(),
                support_label: support.into(),
                epistemic_lane: "source_attested".into(),
                severity: "major".into(),
            }],
        }
    }

    #[test]
    fn nominal_alpha_is_zero_for_two_disagreeing_raters() {
        let annotations = vec![
            fixture_annotation("a1", "r1", "fully_answerable", "supported"),
            fixture_annotation("a2", "r2", "fully_answerable", "unsupported"),
        ];
        let metric = agreement_metric(
            "support_label",
            "claim",
            collect_claim_ratings(&annotations, "support_label").unwrap(),
        )
        .unwrap();
        assert_eq!(metric.eligible_unit_count, 1);
        assert_eq!(metric.exact_agreement_rate, Some(0.0));
        assert_eq!(metric.krippendorff_alpha_nominal, Some(0.0));
        assert_eq!(metric.interpretation, "estimated");
    }

    #[test]
    fn alpha_is_one_for_agreement_with_category_variation() {
        let mut a1 = fixture_annotation("a1", "r1", "fully_answerable", "supported");
        let mut a2 = fixture_annotation("a2", "r2", "fully_answerable", "supported");
        let mut b1 = fixture_annotation("b1", "r1", "partially_answerable", "unsupported");
        let mut b2 = fixture_annotation("b2", "r2", "partially_answerable", "unsupported");
        a1.case_id = "case-A".into();
        a2.case_id = "case-A".into();
        b1.case_id = "case-B".into();
        b2.case_id = "case-B".into();
        let annotations = vec![a1, a2, b1, b2];
        let metric = agreement_metric(
            "support_label",
            "claim",
            collect_claim_ratings(&annotations, "support_label").unwrap(),
        )
        .unwrap();
        assert_eq!(metric.eligible_unit_count, 2);
        assert_eq!(metric.exact_agreement_rate, Some(1.0));
        assert_eq!(metric.krippendorff_alpha_nominal, Some(1.0));
    }

    #[test]
    fn alpha_is_undefined_without_category_variation() {
        let annotations = vec![
            fixture_annotation("a1", "r1", "fully_answerable", "supported"),
            fixture_annotation("a2", "r2", "fully_answerable", "supported"),
        ];
        let metric = agreement_metric(
            "support_label",
            "claim",
            collect_claim_ratings(&annotations, "support_label").unwrap(),
        )
        .unwrap();
        assert_eq!(metric.exact_agreement_rate, Some(1.0));
        assert_eq!(metric.krippendorff_alpha_nominal, None);
        assert_eq!(metric.interpretation, "undefined_no_category_variation");
    }

    #[test]
    fn duplicate_annotator_rating_for_same_unit_fails() {
        let annotations = vec![
            fixture_annotation("a1", "same-reviewer", "fully_answerable", "supported"),
            fixture_annotation("a2", "same-reviewer", "fully_answerable", "unsupported"),
        ];
        let error = agreement_metric(
            "support_label",
            "claim",
            collect_claim_ratings(&annotations, "support_label").unwrap(),
        )
        .unwrap_err();
        assert!(error.contains("duplicate rating from annotator"));
    }

    #[test]
    fn synthetic_fixture_labels_never_claim_real_reliability() {
        let annotations = vec![fixture_annotation(
            "a1",
            "r1",
            "fully_answerable",
            "supported",
        )];
        assert!(validate_review_set(&annotations).unwrap());
    }

    #[test]
    fn real_agreement_rejects_adjudicated_or_synthetic_mixed_records() {
        let mut real = fixture_annotation("a1", "r1", "fully_answerable", "supported");
        real.annotation_role = "subject_expert".into();
        real.review_status = "adjudicated".into();
        let error = validate_review_set(&[real]).unwrap_err();
        assert!(error.contains("pre-adjudication independent labels"));

        let mixed = vec![
            fixture_annotation("a1", "r1", "fully_answerable", "supported"),
            {
                let mut real = fixture_annotation("a2", "r2", "fully_answerable", "supported");
                real.annotation_role = "independent_reviewer".into();
                real.review_status = "independent".into();
                real
            },
        ];
        assert!(validate_review_set(&mixed).is_err());
    }

    #[test]
    fn bundled_annotation_fixture_retains_intended_disagreement() {
        let bytes =
            include_bytes!("../../../../benchmarks/knowledge-annotation/v0/synthetic-fixtures.jsonl");
        let annotations = parse_annotations(bytes).unwrap();
        assert!(validate_review_set(&annotations).unwrap());
        let metric = agreement_metric(
            "support_label",
            "claim",
            collect_claim_ratings(&annotations, "support_label").unwrap(),
        )
        .unwrap();
        assert_eq!(metric.eligible_unit_count, 1);
        assert_eq!(metric.krippendorff_alpha_nominal, Some(0.0));
    }
}
