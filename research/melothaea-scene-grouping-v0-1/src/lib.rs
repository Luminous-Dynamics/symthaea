#![forbid(unsafe_code)]

//! Auditory scene-grouping evaluator with explicit ambiguity and abstention.
//!
//! The scoring unit is a source-attributed acoustic component, not a note or
//! a forced transcription. A component may be one partial track, transient,
//! or localized time-frequency event. A complex polyphonic event can contain
//! multiple source components.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

const MAX_COMPONENTS: usize = 10_000;
const DIGEST_LENGTH: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplitV1 {
    Train,
    Development,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceDispositionV1 {
    Resolved,
    Ambiguous,
    NotEstimable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleSupportV1 {
    pub start_sample: u64,
    pub end_sample: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticComponentV1 {
    pub component_id: String,
    pub support: SampleSupportV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneGroupingCaseV1 {
    pub schema_version: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub case_id: String,
    pub split: EvaluationSplitV1,
    pub source_audio_sha256: String,
    pub rights_manifest_sha256: String,
    pub components: Vec<AcousticComponentV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictedComponentV1 {
    pub component_id: String,
    pub predicted_source_group: Option<String>,
    /// Confidence is required for an assignment and absent for abstention.
    pub assignment_confidence: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneGroupingPredictionV1 {
    pub schema_version: String,
    pub case_id: String,
    pub source_audio_sha256: String,
    pub model_identity_sha256: String,
    pub analysis_manifest_sha256: String,
    pub components: Vec<PredictedComponentV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceComponentV1 {
    pub component_id: String,
    pub reference_source_group: Option<String>,
    pub disposition: ReferenceDispositionV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneGroupingReferenceV1 {
    pub schema_version: String,
    pub case_id: String,
    pub source_audio_sha256: String,
    pub annotation_manifest_sha256: String,
    pub components: Vec<ReferenceComponentV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneGroupingEvaluationBundleV1 {
    pub case: SceneGroupingCaseV1,
    pub prediction: SceneGroupingPredictionV1,
    pub reference: SceneGroupingReferenceV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupingError {
    InvalidSchemaVersion,
    InvalidIdentifier(&'static str),
    DuplicateIdentifier(&'static str),
    InvalidDigest(&'static str),
    EmptyComponentSet,
    TooManyComponents,
    InvalidSampleSupport,
    PredictionCaseMismatch,
    ReferenceCaseMismatch,
    SourceAudioMismatch,
    ComponentSetMismatch,
    InvalidAssignment,
}

impl fmt::Display for GroupingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion => write!(formatter, "unsupported scene-grouping schema version"),
            Self::InvalidIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::DuplicateIdentifier(field) => write!(formatter, "{field} values must be unique"),
            Self::InvalidDigest(field) => write!(formatter, "{field} must be lowercase SHA-256"),
            Self::EmptyComponentSet => write!(formatter, "scene case requires at least one acoustic component"),
            Self::TooManyComponents => write!(formatter, "component count exceeds the bounded evaluation limit"),
            Self::InvalidSampleSupport => write!(formatter, "component sample support must be a non-empty half-open interval"),
            Self::PredictionCaseMismatch => write!(formatter, "prediction case identity does not match the scene case"),
            Self::ReferenceCaseMismatch => write!(formatter, "reference case identity does not match the scene case"),
            Self::SourceAudioMismatch => write!(formatter, "case, prediction and reference must bind the same source audio digest"),
            Self::ComponentSetMismatch => write!(formatter, "case, prediction and reference component identities must match exactly"),
            Self::InvalidAssignment => write!(formatter, "source assignment, ambiguity and confidence fields are inconsistent"),
        }
    }
}

impl Error for GroupingError {}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairwiseMetricsV1 {
    pub true_positive_pairs: u64,
    pub false_positive_pairs: u64,
    pub false_negative_pairs: u64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BcubedMetricsV1 {
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SplitMergeMetricsV1 {
    pub reference_cluster_count: usize,
    pub predicted_cluster_count: usize,
    pub split_reference_cluster_count: usize,
    pub merge_predicted_cluster_count: usize,
    pub split_rate: Option<f64>,
    pub merge_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SceneGroupingReportV1 {
    pub schema_version: String,
    pub case_id: String,
    pub source_audio_sha256: String,
    pub resolved_component_count: usize,
    pub covered_component_count: usize,
    pub abstained_component_count: usize,
    pub ambiguous_component_count: usize,
    pub not_estimable_component_count: usize,
    pub coverage: Option<f64>,
    pub pairwise: PairwiseMetricsV1,
    pub bcubed: BcubedMetricsV1,
    pub split_merge: SplitMergeMetricsV1,
}

fn nonempty(value: &str, field: &'static str) -> Result<(), GroupingError> {
    if value.trim().is_empty() {
        Err(GroupingError::InvalidIdentifier(field))
    } else {
        Ok(())
    }
}

fn digest(value: &str, field: &'static str) -> Result<(), GroupingError> {
    if value.len() == DIGEST_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(GroupingError::InvalidDigest(field))
    }
}

fn validate_unique(
    identifiers: impl IntoIterator<Item = String>,
    field: &'static str,
) -> Result<(), GroupingError> {
    let mut seen = BTreeSet::new();
    for identifier in identifiers {
        nonempty(&identifier, field)?;
        if !seen.insert(identifier) {
            return Err(GroupingError::DuplicateIdentifier(field));
        }
    }
    Ok(())
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn f1(precision: Option<f64>, recall: Option<f64>) -> Option<f64> {
    match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
}

fn choose2(count: usize) -> u64 {
    if count < 2 {
        0
    } else {
        let n = count as u64;
        n.saturating_mul(n.saturating_sub(1)) / 2
    }
}

/// Validate source and annotation identities, then score source grouping.
///
/// Ambiguous and not-estimable reference components are reported separately.
/// Abstentions do not silently become source mismatches: they reduce coverage.
/// B-cubed and pairwise scores are computed conditionally on covered, resolved
/// components, with coverage always reported alongside them.
pub fn evaluate_scene_grouping(
    bundle: &SceneGroupingEvaluationBundleV1,
) -> Result<SceneGroupingReportV1, GroupingError> {
    let case = &bundle.case;
    let prediction = &bundle.prediction;
    let reference = &bundle.reference;

    if case.schema_version != "mel-hear-016-scene-grouping-case-v0.1"
        || prediction.schema_version != "mel-hear-016-scene-grouping-prediction-v0.1"
        || reference.schema_version != "mel-hear-016-scene-grouping-reference-v0.1"
        || case.benchmark_id != "MEL-AUDITORY-SCENE-GROUPING"
        || case.benchmark_version.trim().is_empty()
    {
        return Err(GroupingError::InvalidSchemaVersion);
    }
    nonempty(&case.case_id, "case_id")?;
    nonempty(&prediction.case_id, "prediction.case_id")?;
    nonempty(&reference.case_id, "reference.case_id")?;
    digest(&case.source_audio_sha256, "case.source_audio_sha256")?;
    digest(&case.rights_manifest_sha256, "case.rights_manifest_sha256")?;
    digest(&prediction.source_audio_sha256, "prediction.source_audio_sha256")?;
    digest(&prediction.model_identity_sha256, "prediction.model_identity_sha256")?;
    digest(&prediction.analysis_manifest_sha256, "prediction.analysis_manifest_sha256")?;
    digest(&reference.source_audio_sha256, "reference.source_audio_sha256")?;
    digest(&reference.annotation_manifest_sha256, "reference.annotation_manifest_sha256")?;

    if case.components.is_empty() {
        return Err(GroupingError::EmptyComponentSet);
    }
    if case.components.len() > MAX_COMPONENTS {
        return Err(GroupingError::TooManyComponents);
    }
    if prediction.case_id != case.case_id {
        return Err(GroupingError::PredictionCaseMismatch);
    }
    if reference.case_id != case.case_id {
        return Err(GroupingError::ReferenceCaseMismatch);
    }
    if prediction.source_audio_sha256 != case.source_audio_sha256
        || reference.source_audio_sha256 != case.source_audio_sha256
    {
        return Err(GroupingError::SourceAudioMismatch);
    }

    validate_unique(
        case.components.iter().map(|component| component.component_id.clone()),
        "case.component_id",
    )?;
    validate_unique(
        prediction.components.iter().map(|component| component.component_id.clone()),
        "prediction.component_id",
    )?;
    validate_unique(
        reference.components.iter().map(|component| component.component_id.clone()),
        "reference.component_id",
    )?;

    for component in &case.components {
        if component.support.start_sample >= component.support.end_sample {
            return Err(GroupingError::InvalidSampleSupport);
        }
    }
    for assignment in &prediction.components {
        match (&assignment.predicted_source_group, assignment.assignment_confidence) {
            (Some(group), Some(confidence)) => {
                nonempty(group, "predicted_source_group")?;
                if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                    return Err(GroupingError::InvalidAssignment);
                }
            }
            (None, None) => {}
            _ => return Err(GroupingError::InvalidAssignment),
        }
    }
    for item in &reference.components {
        match (item.disposition, item.reference_source_group.as_deref()) {
            (ReferenceDispositionV1::Resolved, Some(group)) if !group.trim().is_empty() => {}
            (ReferenceDispositionV1::Ambiguous, None)
            | (ReferenceDispositionV1::NotEstimable, None) => {}
            _ => return Err(GroupingError::InvalidAssignment),
        }
    }

    let case_ids: BTreeSet<&str> = case.components.iter()
        .map(|component| component.component_id.as_str()).collect();
    let prediction_ids: BTreeSet<&str> = prediction.components.iter()
        .map(|component| component.component_id.as_str()).collect();
    let reference_ids: BTreeSet<&str> = reference.components.iter()
        .map(|component| component.component_id.as_str()).collect();
    if case_ids != prediction_ids || case_ids != reference_ids {
        return Err(GroupingError::ComponentSetMismatch);
    }

    let prediction_by_id: BTreeMap<&str, &PredictedComponentV1> = prediction.components.iter()
        .map(|component| (component.component_id.as_str(), component)).collect();
    let mut resolved_component_count = 0usize;
    let mut covered_component_count = 0usize;
    let mut abstained_component_count = 0usize;
    let mut ambiguous_component_count = 0usize;
    let mut not_estimable_component_count = 0usize;

    let mut predicted_cluster_sizes: BTreeMap<&str, usize> = BTreeMap::new();
    let mut reference_cluster_sizes: BTreeMap<&str, usize> = BTreeMap::new();
    let mut intersection_sizes: BTreeMap<(&str, &str), usize> = BTreeMap::new();

    for item in &reference.components {
        match item.disposition {
            ReferenceDispositionV1::Ambiguous => {
                ambiguous_component_count += 1;
                continue;
            }
            ReferenceDispositionV1::NotEstimable => {
                not_estimable_component_count += 1;
                continue;
            }
            ReferenceDispositionV1::Resolved => {}
        }
        resolved_component_count += 1;
        let reference_group = item.reference_source_group.as_deref()
            .ok_or(GroupingError::InvalidAssignment)?;
        let prediction_item = prediction_by_id[item.component_id.as_str()];
        let Some(predicted_group) = prediction_item.predicted_source_group.as_deref() else {
            abstained_component_count += 1;
            continue;
        };

        covered_component_count += 1;
        *predicted_cluster_sizes.entry(predicted_group).or_default() += 1;
        *reference_cluster_sizes.entry(reference_group).or_default() += 1;
        *intersection_sizes.entry((predicted_group, reference_group)).or_default() += 1;
    }

    let mut true_positive_pairs = 0u64;
    let mut predicted_positive_pairs = 0u64;
    let mut reference_positive_pairs = 0u64;
    for size in predicted_cluster_sizes.values() {
        predicted_positive_pairs = predicted_positive_pairs.saturating_add(choose2(*size));
    }
    for size in reference_cluster_sizes.values() {
        reference_positive_pairs = reference_positive_pairs.saturating_add(choose2(*size));
    }
    for size in intersection_sizes.values() {
        true_positive_pairs = true_positive_pairs.saturating_add(choose2(*size));
    }
    let false_positive_pairs = predicted_positive_pairs.saturating_sub(true_positive_pairs);
    let false_negative_pairs = reference_positive_pairs.saturating_sub(true_positive_pairs);

    let pairwise_precision = ratio(true_positive_pairs, true_positive_pairs + false_positive_pairs);
    let pairwise_recall = ratio(true_positive_pairs, true_positive_pairs + false_negative_pairs);
    let pairwise = PairwiseMetricsV1 {
        true_positive_pairs,
        false_positive_pairs,
        false_negative_pairs,
        precision: pairwise_precision,
        recall: pairwise_recall,
        f1: f1(pairwise_precision, pairwise_recall),
    };

    let mut bcubed_precision_total = 0.0f64;
    let mut bcubed_recall_total = 0.0f64;
    for ((predicted_group, reference_group), intersection_count) in &intersection_sizes {
        let predicted_count = predicted_cluster_sizes[predicted_group];
        let reference_count = reference_cluster_sizes[reference_group];
        bcubed_precision_total += *intersection_count as f64
            / predicted_count as f64 * *intersection_count as f64;
        bcubed_recall_total += *intersection_count as f64
            / reference_count as f64 * *intersection_count as f64;
    }
    let bcubed = if covered_component_count == 0 {
        BcubedMetricsV1 {
            precision: None,
            recall: None,
            f1: None,
        }
    } else {
        let denominator = covered_component_count as f64;
        // Each intersection contributes one identical precision/recall value
        // per item in that intersection.
        let precision = Some(bcubed_precision_total / denominator);
        let recall = Some(bcubed_recall_total / denominator);
        BcubedMetricsV1 {
            precision,
            recall,
            f1: f1(precision, recall),
        }
    };

    let mut reference_to_predicted: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut predicted_to_reference: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for ((predicted_group, reference_group), _) in &intersection_sizes {
        reference_to_predicted.entry(reference_group).or_default().insert(predicted_group);
        predicted_to_reference.entry(predicted_group).or_default().insert(reference_group);
    }
    let split_reference_cluster_count = reference_to_predicted.values()
        .filter(|groups| groups.len() > 1).count();
    let merge_predicted_cluster_count = predicted_to_reference.values()
        .filter(|groups| groups.len() > 1).count();
    let reference_cluster_count = reference_to_predicted.len();
    let predicted_cluster_count = predicted_to_reference.len();
    let split_merge = SplitMergeMetricsV1 {
        reference_cluster_count,
        predicted_cluster_count,
        split_reference_cluster_count,
        merge_predicted_cluster_count,
        split_rate: if reference_cluster_count == 0 {
            None
        } else {
            Some(split_reference_cluster_count as f64 / reference_cluster_count as f64)
        },
        merge_rate: if predicted_cluster_count == 0 {
            None
        } else {
            Some(merge_predicted_cluster_count as f64 / predicted_cluster_count as f64)
        },
    };

    Ok(SceneGroupingReportV1 {
        schema_version: "mel-hear-016-scene-grouping-report-v0.1".to_owned(),
        case_id: case.case_id.clone(),
        source_audio_sha256: case.source_audio_sha256.clone(),
        resolved_component_count,
        covered_component_count,
        abstained_component_count,
        ambiguous_component_count,
        not_estimable_component_count,
        coverage: if resolved_component_count == 0 {
            None
        } else {
            Some(covered_component_count as f64 / resolved_component_count as f64)
        },
        pairwise,
        bcubed,
        split_merge,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_scene_grouping, AcousticComponentV1, EvaluationSplitV1,
        PredictedComponentV1, ReferenceComponentV1, ReferenceDispositionV1,
        SampleSupportV1, SceneGroupingCaseV1, SceneGroupingEvaluationBundleV1,
        SceneGroupingPredictionV1, SceneGroupingReferenceV1, GroupingError,
    };

    fn digest(byte: u8) -> String {
        format!("{:02x}", byte).repeat(32)
    }

    fn bundle(predicted: &[Option<&str>], references: &[(&str, ReferenceDispositionV1)]) -> SceneGroupingEvaluationBundleV1 {
        let ids = ["c0", "c1", "c2", "c3"];
        let components = ids.iter().enumerate().map(|(index, id)| AcousticComponentV1 {
            component_id: (*id).to_owned(),
            support: SampleSupportV1 {
                start_sample: index as u64 * 100,
                end_sample: index as u64 * 100 + 80,
            },
        }).collect();

        let predictions = ids.iter().enumerate().map(|(index, id)| {
            let group = predicted[index].map(str::to_owned);
            PredictedComponentV1 {
                component_id: (*id).to_owned(),
                assignment_confidence: group.as_ref().map(|_| 0.9),
                predicted_source_group: group,
            }
        }).collect();

        let reference_components = ids.iter().enumerate().map(|(index, id)| {
            let (group, disposition) = references[index];
            ReferenceComponentV1 {
                component_id: (*id).to_owned(),
                reference_source_group: if disposition == ReferenceDispositionV1::Resolved {
                    Some(group.to_owned())
                } else {
                    None
                },
                disposition,
            }
        }).collect();

        SceneGroupingEvaluationBundleV1 {
            case: SceneGroupingCaseV1 {
                schema_version: "mel-hear-016-scene-grouping-case-v0.1".to_owned(),
                benchmark_id: "MEL-AUDITORY-SCENE-GROUPING".to_owned(),
                benchmark_version: "0.1.0".to_owned(),
                case_id: "scene-001".to_owned(),
                split: EvaluationSplitV1::Development,
                source_audio_sha256: digest(1),
                rights_manifest_sha256: digest(2),
                components,
            },
            prediction: SceneGroupingPredictionV1 {
                schema_version: "mel-hear-016-scene-grouping-prediction-v0.1".to_owned(),
                case_id: "scene-001".to_owned(),
                source_audio_sha256: digest(1),
                model_identity_sha256: digest(3),
                analysis_manifest_sha256: digest(4),
                components: predictions,
            },
            reference: SceneGroupingReferenceV1 {
                schema_version: "mel-hear-016-scene-grouping-reference-v0.1".to_owned(),
                case_id: "scene-001".to_owned(),
                source_audio_sha256: digest(1),
                annotation_manifest_sha256: digest(5),
                components: reference_components,
            },
        }
    }

    #[test]
    fn permutation_invariant_exact_grouping_scores_one() {
        let sample = bundle(
            &[Some("x"), Some("x"), Some("y"), Some("y")],
            &[
                ("a", ReferenceDispositionV1::Resolved),
                ("a", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
            ],
        );
        let report = evaluate_scene_grouping(&sample).unwrap();
        assert_eq!(report.coverage, Some(1.0));
        assert_eq!(report.pairwise.f1, Some(1.0));
        assert_eq!(report.bcubed.f1, Some(1.0));
        assert_eq!(report.split_merge.split_reference_cluster_count, 0);
        assert_eq!(report.split_merge.merge_predicted_cluster_count, 0);
    }

    #[test]
    fn split_and_merge_errors_are_reported_separately() {
        let split = bundle(
            &[Some("x"), Some("y"), Some("z"), Some("w")],
            &[
                ("a", ReferenceDispositionV1::Resolved),
                ("a", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
            ],
        );
        let split_report = evaluate_scene_grouping(&split).unwrap();
        assert_eq!(split_report.split_merge.split_reference_cluster_count, 2);
        assert_eq!(split_report.split_merge.merge_predicted_cluster_count, 0);

        let merged = bundle(
            &[Some("x"), Some("x"), Some("x"), Some("x")],
            &[
                ("a", ReferenceDispositionV1::Resolved),
                ("a", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
            ],
        );
        let merged_report = evaluate_scene_grouping(&merged).unwrap();
        assert_eq!(merged_report.split_merge.split_reference_cluster_count, 0);
        assert_eq!(merged_report.split_merge.merge_predicted_cluster_count, 1);
    }

    #[test]
    fn abstention_reduces_coverage_without_becoming_a_false_cluster() {
        let sample = bundle(
            &[Some("x"), None, Some("y"), Some("y")],
            &[
                ("a", ReferenceDispositionV1::Resolved),
                ("a", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
            ],
        );
        let report = evaluate_scene_grouping(&sample).unwrap();
        assert_eq!(report.coverage, Some(0.75));
        assert_eq!(report.abstained_component_count, 1);
        assert_eq!(report.pairwise.f1, Some(1.0));
    }

    #[test]
    fn ambiguous_and_not_estimable_references_have_separate_denominators() {
        let sample = bundle(
            &[Some("x"), Some("x"), None, None],
            &[
                ("a", ReferenceDispositionV1::Resolved),
                ("a", ReferenceDispositionV1::Resolved),
                ("", ReferenceDispositionV1::Ambiguous),
                ("", ReferenceDispositionV1::NotEstimable),
            ],
        );
        let report = evaluate_scene_grouping(&sample).unwrap();
        assert_eq!(report.resolved_component_count, 2);
        assert_eq!(report.ambiguous_component_count, 1);
        assert_eq!(report.not_estimable_component_count, 1);
        assert_eq!(report.coverage, Some(1.0));
    }

    #[test]
    fn source_audio_digest_mismatch_rejects() {
        let mut sample = bundle(
            &[Some("x"), Some("x"), Some("y"), Some("y")],
            &[
                ("a", ReferenceDispositionV1::Resolved),
                ("a", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
            ],
        );
        sample.prediction.source_audio_sha256 = digest(9);
        assert_eq!(
            evaluate_scene_grouping(&sample),
            Err(GroupingError::SourceAudioMismatch)
        );
    }

    #[test]
    fn unknown_or_missing_component_predictions_reject() {
        let mut sample = bundle(
            &[Some("x"), Some("x"), Some("y"), Some("y")],
            &[
                ("a", ReferenceDispositionV1::Resolved),
                ("a", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
            ],
        );
        sample.prediction.components.pop();
        assert_eq!(
            evaluate_scene_grouping(&sample),
            Err(GroupingError::ComponentSetMismatch)
        );
    }

    #[test]
    fn reference_disposition_must_match_reference_assignment() {
        let mut sample = bundle(
            &[Some("x"), Some("x"), Some("y"), Some("y")],
            &[
                ("a", ReferenceDispositionV1::Resolved),
                ("a", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
                ("b", ReferenceDispositionV1::Resolved),
            ],
        );
        sample.reference.components[0].reference_source_group = None;
        assert_eq!(
            evaluate_scene_grouping(&sample),
            Err(GroupingError::InvalidAssignment)
        );
    }
}
