//! Corrected multi-rater unitizing-alpha profile for continuous musical time.
//!
//! This is a Rust transcription of the coincidence construction in
//! Krippendorff's 2019 fourth edition, section 12.6 (the corrected u-alpha
//! family developed in the 2016 paper). It is not the withdrawn simplified
//! first-printing c_u-alpha formula and it is not pairwise interval F1.
//!
//! The public audio adapter evaluates one event query/category per continuum.
//! Each eligible annotator partitions [0, total_frames) into event intervals
//! and explicit gap segments. Ambiguous and not-estimable records are retained
//! but excluded from the unitization panel; they are never converted to gaps.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::temporal_consensus::IndependentTemporalAnnotationV1;
use crate::temporal_grounding::{AudioArtifactClockV1, FrameIntervalV1, GroundTruthStateV1};
use crate::temporal_reliability::ReliabilityCorpusItemV1;

const CASE_SCHEMA_V1: &str = "mel-temporal-unitizing-alpha-u-corpus-v1";
const REPORT_SCHEMA_V1: &str = "mel-temporal-unitizing-alpha-u-report-v1";
const EVALUATOR_VERSION_V1: &str = "mel-temporal-unitizing-alpha-u-v1";
const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;
const NUMERIC_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitizingCorpusCaseV1 {
    pub schema: String,
    pub annotation_manifest_sha256: String,
    /// Minimum eligible Present/Absent unitizations per item. Ambiguous and
    /// NotEstimable annotations do not contribute to this count.
    pub minimum_eligible_unitizers_per_item: u16,
    pub items: Vec<ReliabilityCorpusItemV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UnitizingItemStatusV1 {
    Scored,
    InsufficientEligibleUnitizers,
    UndefinedExpectedDisagreement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum UnitizingExclusionReasonV1 {
    Ambiguous,
    NotEstimable,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitizingAnnotatorDispositionV1 {
    pub annotator_id_sha256: String,
    pub annotation_sha256: String,
    pub state: GroundTruthStateV1,
    /// Original per-annotator intervals are retained verbatim in frame coordinates.
    pub intervals: Vec<FrameIntervalV1>,
    pub status_reason: Option<String>,
    pub used_for_unitizing: bool,
    pub exclusion_reason: Option<UnitizingExclusionReasonV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitizingPerValueAlphaV1 {
    pub value: String,
    pub alpha: Option<f64>,
    /// Fraction of this value's coincidence mass that is paired with the
    /// same value in another annotator's non-gap region.
    pub overlap_coverage: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitizingAlphaProfileV1 {
    pub observers: usize,
    pub continuum_frames: u64,
    pub category_count_including_gap: usize,
    pub observed_disagreement_u_nominal: f64,
    pub expected_disagreement_u_nominal: f64,
    pub u_alpha_nominal: Option<f64>,
    pub observed_disagreement_u_binary: f64,
    pub expected_disagreement_u_binary: f64,
    pub u_alpha_binary: Option<f64>,
    pub observed_disagreement_cu_nominal: Option<f64>,
    pub expected_disagreement_cu_nominal: Option<f64>,
    pub cu_alpha_nominal: Option<f64>,
    pub per_value: Vec<UnitizingPerValueAlphaV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitizingItemReportV1 {
    pub item_id: String,
    pub source_work_cluster_sha256: String,
    pub query_id: String,
    pub query_text: String,
    pub audio_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub annotation_policy_version: String,
    pub continuum_frames: u64,
    pub annotation_records: usize,
    pub eligible_unitizers: usize,
    pub excluded_ambiguous: usize,
    pub excluded_not_estimable: usize,
    pub status: UnitizingItemStatusV1,
    pub undefined_reason: Option<String>,
    pub coefficients: Option<UnitizingAlphaProfileV1>,
    pub annotator_dispositions: Vec<UnitizingAnnotatorDispositionV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitizingSourceWorkSummaryV1 {
    pub source_work_cluster_sha256: String,
    pub item_count: usize,
    pub scored_item_count: usize,
    pub mean_u_alpha_nominal: Option<f64>,
    pub mean_u_alpha_binary: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitizingCorpusSummaryV1 {
    pub item_count: usize,
    pub scored_item_count: usize,
    pub insufficient_eligible_items: usize,
    pub undefined_items: usize,
    pub total_annotation_records: usize,
    pub total_eligible_unitizations: usize,
    pub excluded_ambiguous: usize,
    pub excluded_not_estimable: usize,
    pub source_work_clusters: usize,
    /// Descriptive unweighted summaries over item-level coefficients. No
    /// intervals or frames are pooled to calculate a single corpus alpha.
    pub mean_item_u_alpha_nominal: Option<f64>,
    pub median_item_u_alpha_nominal: Option<f64>,
    pub min_item_u_alpha_nominal: Option<f64>,
    pub max_item_u_alpha_nominal: Option<f64>,
    pub mean_item_u_alpha_binary: Option<f64>,
    pub mean_item_cu_alpha_nominal: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnitizingCorpusReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub annotation_manifest_sha256: String,
    pub minimum_eligible_unitizers_per_item: u16,
    pub summary: UnitizingCorpusSummaryV1,
    pub source_work_summaries: Vec<UnitizingSourceWorkSummaryV1>,
    pub item_results: Vec<UnitizingItemReportV1>,
    pub unitizing_alpha_computed: bool,
    pub consensus_ground_truth_created: bool,
    pub pooled_continuum_alpha_computed: bool,
    pub population_confidence_interval_computed: bool,
    pub source_work_bootstrap_computed: bool,
    pub model_performance_scored: bool,
    pub annotation_receipts_authenticated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CategoryV1 {
    Gap,
    Value(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UnitSegmentV1 {
    start_frame: u64,
    end_frame: u64,
    category: CategoryV1,
}

#[derive(Debug, Clone, PartialEq)]
struct DisagreementPairV1 {
    observed: f64,
    expected: f64,
    alpha: Option<f64>,
}

pub fn audit_unitizing_corpus_json(input: &str) -> Result<UnitizingCorpusReportV1, String> {
    let case: UnitizingCorpusCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid unitizing corpus JSON: {error}"))?;
    audit_unitizing_corpus(&case)
}

pub fn audit_unitizing_corpus(
    case: &UnitizingCorpusCaseV1,
) -> Result<UnitizingCorpusReportV1, String> {
    if case.schema != CASE_SCHEMA_V1 {
        return Err(format!("unsupported unitizing corpus schema: {}", case.schema));
    }
    validate_digest(
        &case.annotation_manifest_sha256,
        "annotation_manifest_sha256",
    )?;
    if case.minimum_eligible_unitizers_per_item < 2 {
        return Err("minimum_eligible_unitizers_per_item must be at least two".into());
    }
    if case.items.is_empty() {
        return Err("unitizing corpus must contain at least one item".into());
    }

    let mut item_ids = BTreeSet::new();
    let mut source_work_clusters = BTreeSet::new();
    let mut total_annotation_records = 0_usize;
    let mut total_eligible_unitizations = 0_usize;
    let mut excluded_ambiguous = 0_usize;
    let mut excluded_not_estimable = 0_usize;
    let mut results = Vec::with_capacity(case.items.len());

    for (item_index, item) in case.items.iter().enumerate() {
        validate_item(item, item_index)?;
        if !item_ids.insert(item.item_id.as_str()) {
            return Err(format!("duplicate unitizing item_id: {}", item.item_id));
        }
        source_work_clusters.insert(item.source_work_cluster_sha256.as_str());
        total_annotation_records = total_annotation_records
            .checked_add(item.annotations.len())
            .ok_or_else(|| "unitizing annotation record count overflow".to_string())?;

        let mut annotator_ids = BTreeSet::new();
        let mut annotation_digests = BTreeSet::new();
        let mut eligible_partitions = Vec::new();
        let mut dispositions = Vec::with_capacity(item.annotations.len());
        let mut ambiguous = 0_usize;
        let mut not_estimable = 0_usize;

        for (annotation_index, annotation) in item.annotations.iter().enumerate() {
            validate_digest(
                &annotation.annotator_id_sha256,
                &format!(
                    "items[{item_index}].annotations[{annotation_index}].annotator_id_sha256"
                ),
            )?;
            validate_digest(
                &annotation.annotation_sha256,
                &format!(
                    "items[{item_index}].annotations[{annotation_index}].annotation_sha256"
                ),
            )?;
            if !annotator_ids.insert(annotation.annotator_id_sha256.as_str()) {
                return Err(format!(
                    "item {} repeats an annotator identity",
                    item.item_id
                ));
            }
            if !annotation_digests.insert(annotation.annotation_sha256.as_str()) {
                return Err(format!(
                    "item {} repeats an annotation receipt digest",
                    item.item_id
                ));
            }

            match annotation.state {
                GroundTruthStateV1::Present => {
                    validate_annotation_intervals(
                        &annotation.intervals,
                        item.artifact.total_frames,
                        &item.item_id,
                        annotation_index,
                        true,
                    )?;
                    let labeled: Vec<_> = annotation
                        .intervals
                        .iter()
                        .map(|interval| {
                            (
                                interval.start_frame,
                                interval.end_frame,
                                "event".to_string(),
                            )
                        })
                        .collect();
                    eligible_partitions.push(build_partition(
                        &labeled,
                        item.artifact.total_frames,
                    )?);
                    dispositions.push(UnitizingAnnotatorDispositionV1 {
                        annotator_id_sha256: annotation.annotator_id_sha256.clone(),
                        annotation_sha256: annotation.annotation_sha256.clone(),
                        state: annotation.state,
                        intervals: annotation.intervals.clone(),
                        status_reason: annotation.status_reason.clone(),
                        used_for_unitizing: true,
                        exclusion_reason: None,
                    });
                }
                GroundTruthStateV1::Absent => {
                    if !annotation.intervals.is_empty() {
                        return Err(format!(
                            "item {} annotator {} is Absent but has interval records",
                            item.item_id, annotation_index
                        ));
                    }
                    eligible_partitions.push(build_partition(
                        &[],
                        item.artifact.total_frames,
                    )?);
                    dispositions.push(UnitizingAnnotatorDispositionV1 {
                        annotator_id_sha256: annotation.annotator_id_sha256.clone(),
                        annotation_sha256: annotation.annotation_sha256.clone(),
                        state: annotation.state,
                        intervals: annotation.intervals.clone(),
                        status_reason: annotation.status_reason.clone(),
                        used_for_unitizing: true,
                        exclusion_reason: None,
                    });
                }
                GroundTruthStateV1::Ambiguous => {
                    require_text(
                        annotation.status_reason.as_deref().unwrap_or(""),
                        &format!("items[{item_index}].annotations[{annotation_index}].status_reason"),
                    )?;
                    validate_annotation_intervals(
                        &annotation.intervals,
                        item.artifact.total_frames,
                        &item.item_id,
                        annotation_index,
                        false,
                    )?;
                    ambiguous += 1;
                    dispositions.push(UnitizingAnnotatorDispositionV1 {
                        annotator_id_sha256: annotation.annotator_id_sha256.clone(),
                        annotation_sha256: annotation.annotation_sha256.clone(),
                        state: annotation.state,
                        intervals: annotation.intervals.clone(),
                        status_reason: annotation.status_reason.clone(),
                        used_for_unitizing: false,
                        exclusion_reason: Some(UnitizingExclusionReasonV1::Ambiguous),
                    });
                }
                GroundTruthStateV1::NotEstimable => {
                    require_text(
                        annotation.status_reason.as_deref().unwrap_or(""),
                        &format!("items[{item_index}].annotations[{annotation_index}].status_reason"),
                    )?;
                    validate_annotation_intervals(
                        &annotation.intervals,
                        item.artifact.total_frames,
                        &item.item_id,
                        annotation_index,
                        false,
                    )?;
                    not_estimable += 1;
                    dispositions.push(UnitizingAnnotatorDispositionV1 {
                        annotator_id_sha256: annotation.annotator_id_sha256.clone(),
                        annotation_sha256: annotation.annotation_sha256.clone(),
                        state: annotation.state,
                        intervals: annotation.intervals.clone(),
                        status_reason: annotation.status_reason.clone(),
                        used_for_unitizing: false,
                        exclusion_reason: Some(UnitizingExclusionReasonV1::NotEstimable),
                    });
                }
            }
        }

        total_eligible_unitizations = total_eligible_unitizations
            .checked_add(eligible_partitions.len())
            .ok_or_else(|| "eligible unitization count overflow".to_string())?;
        excluded_ambiguous += ambiguous;
        excluded_not_estimable += not_estimable;

        let (status, undefined_reason, coefficients) =
            if eligible_partitions.len() < case.minimum_eligible_unitizers_per_item as usize {
                (
                    UnitizingItemStatusV1::InsufficientEligibleUnitizers,
                    Some(format!(
                        "eligible unitizations {} are below the required minimum {}",
                        eligible_partitions.len(),
                        case.minimum_eligible_unitizers_per_item
                    )),
                    None,
                )
            } else {
                let profile =
                    compute_unitizing_profile(&eligible_partitions, item.artifact.total_frames)?;
                if profile.u_alpha_nominal.is_some() {
                    (UnitizingItemStatusV1::Scored, None, Some(profile))
                } else {
                    (
                        UnitizingItemStatusV1::UndefinedExpectedDisagreement,
                        Some("u_alpha_nominal has zero expected disagreement but nonzero observed disagreement".into()),
                        Some(profile),
                    )
                }
            };

        results.push(UnitizingItemReportV1 {
            item_id: item.item_id.clone(),
            source_work_cluster_sha256: item.source_work_cluster_sha256.clone(),
            query_id: item.query_id.clone(),
            query_text: item.query_text.clone(),
            audio_sha256: item.artifact.audio_sha256.clone(),
            sample_rate_hz: item.artifact.sample_rate_hz,
            channels: item.artifact.channels,
            annotation_policy_version: item.annotation_policy_version.clone(),
            continuum_frames: item.artifact.total_frames,
            annotation_records: item.annotations.len(),
            eligible_unitizers: eligible_partitions.len(),
            excluded_ambiguous: ambiguous,
            excluded_not_estimable: not_estimable,
            status,
            undefined_reason,
            coefficients,
            annotator_dispositions: dispositions,
        });
    }

    let scored: Vec<_> = results
        .iter()
        .filter(|item| item.status == UnitizingItemStatusV1::Scored)
        .collect();
    let nominal: Vec<_> = scored
        .iter()
        .filter_map(|item| item.coefficients.as_ref()?.u_alpha_nominal)
        .collect();
    let binary: Vec<_> = scored
        .iter()
        .filter_map(|item| item.coefficients.as_ref()?.u_alpha_binary)
        .collect();
    let coding: Vec<_> = scored
        .iter()
        .filter_map(|item| item.coefficients.as_ref()?.cu_alpha_nominal)
        .collect();

    let mut by_source: BTreeMap<&str, Vec<&UnitizingItemReportV1>> = BTreeMap::new();
    for item in &results {
        by_source
            .entry(item.source_work_cluster_sha256.as_str())
            .or_default()
            .push(item);
    }
    let source_work_summaries = by_source
        .into_iter()
        .map(|(source, items)| {
            let values: Vec<_> = items
                .iter()
                .filter_map(|item| item.coefficients.as_ref()?.u_alpha_nominal)
                .collect();
            let binary_values: Vec<_> = items
                .iter()
                .filter_map(|item| item.coefficients.as_ref()?.u_alpha_binary)
                .collect();
            UnitizingSourceWorkSummaryV1 {
                source_work_cluster_sha256: source.to_string(),
                item_count: items.len(),
                scored_item_count: values.len(),
                mean_u_alpha_nominal: mean(&values),
                mean_u_alpha_binary: mean(&binary_values),
            }
        })
        .collect();

    Ok(UnitizingCorpusReportV1 {
        schema: REPORT_SCHEMA_V1,
        evaluator_version: EVALUATOR_VERSION_V1,
        annotation_manifest_sha256: case.annotation_manifest_sha256.clone(),
        minimum_eligible_unitizers_per_item: case.minimum_eligible_unitizers_per_item,
        summary: UnitizingCorpusSummaryV1 {
            item_count: results.len(),
            scored_item_count: scored.len(),
            insufficient_eligible_items: results
                .iter()
                .filter(|item| item.status == UnitizingItemStatusV1::InsufficientEligibleUnitizers)
                .count(),
            undefined_items: results
                .iter()
                .filter(|item| {
                    item.status == UnitizingItemStatusV1::UndefinedExpectedDisagreement
                })
                .count(),
            total_annotation_records,
            total_eligible_unitizations,
            excluded_ambiguous,
            excluded_not_estimable,
            source_work_clusters: source_work_clusters.len(),
            mean_item_u_alpha_nominal: mean(&nominal),
            median_item_u_alpha_nominal: median(&nominal),
            min_item_u_alpha_nominal: nominal.iter().copied().reduce(f64::min),
            max_item_u_alpha_nominal: nominal.iter().copied().reduce(f64::max),
            mean_item_u_alpha_binary: mean(&binary),
            mean_item_cu_alpha_nominal: mean(&coding),
        },
        source_work_summaries,
        item_results: results,
        unitizing_alpha_computed: !scored.is_empty(),
        consensus_ground_truth_created: false,
        pooled_continuum_alpha_computed: false,
        population_confidence_interval_computed: false,
        source_work_bootstrap_computed: false,
        model_performance_scored: false,
        annotation_receipts_authenticated: false,
    })
}

fn validate_item(item: &ReliabilityCorpusItemV1, index: usize) -> Result<(), String> {
    require_text(&item.item_id, &format!("items[{index}].item_id"))?;
    require_text(&item.query_id, &format!("items[{index}].query_id"))?;
    require_text(&item.query_text, &format!("items[{index}].query_text"))?;
    require_text(
        &item.annotation_policy_version,
        &format!("items[{index}].annotation_policy_version"),
    )?;
    validate_digest(
        &item.source_work_cluster_sha256,
        &format!("items[{index}].source_work_cluster_sha256"),
    )?;
    validate_artifact(&item.artifact)?;
    if item.artifact.total_frames > MAX_EXACT_F64_INTEGER {
        return Err(format!(
            "items[{index}].artifact.total_frames exceeds exact f64 integer range"
        ));
    }
    if item.annotations.is_empty() {
        return Err(format!("items[{index}] has no annotation records"));
    }
    if item.annotations.len() > u16::MAX as usize {
        return Err(format!(
            "items[{index}] exceeds the supported annotation count"
        ));
    }
    Ok(())
}

fn validate_artifact(artifact: &AudioArtifactClockV1) -> Result<(), String> {
    validate_digest(&artifact.audio_sha256, "artifact.audio_sha256")?;
    if artifact.sample_rate_hz == 0 || artifact.channels == 0 || artifact.total_frames == 0 {
        return Err("artifact sample rate, channels and total frames must be positive".into());
    }
    Ok(())
}

fn validate_annotation_intervals(
    intervals: &[FrameIntervalV1],
    continuum_frames: u64,
    item_id: &str,
    annotation_index: usize,
    require_nonempty: bool,
) -> Result<(), String> {
    if require_nonempty && intervals.is_empty() {
        return Err(format!(
            "item {item_id} annotator {annotation_index} is Present but has no intervals"
        ));
    }
    for (interval_index, interval) in intervals.iter().enumerate() {
        if interval.start_frame >= interval.end_frame
            || interval.end_frame > continuum_frames
        {
            return Err(format!(
                "item {item_id} annotation {annotation_index} interval {interval_index} is invalid or outside the continuum"
            ));
        }
        if require_nonempty && interval_index > 0 {
            if interval.start_frame < intervals[interval_index - 1].end_frame {
                return Err(format!(
                    "item {item_id} annotator {annotation_index} has overlapping or unsorted unit intervals"
                ));
            }
        }
    }
    Ok(())
}

fn build_partition(
    intervals: &[(u64, u64, String)],
    continuum_frames: u64,
) -> Result<Vec<UnitSegmentV1>, String> {
    if continuum_frames == 0 || continuum_frames > MAX_EXACT_F64_INTEGER {
        return Err("continuum length must be positive and exactly representable".into());
    }
    let mut cursor = 0_u64;
    let mut partition = Vec::with_capacity(intervals.len() * 2 + 1);
    for (index, (start, end, value)) in intervals.iter().enumerate() {
        if value.trim().is_empty() {
            return Err(format!("unit interval {index} has an empty category value"));
        }
        if start >= end || *end > continuum_frames {
            return Err(format!("unit interval {index} is invalid or outside the continuum"));
        }
        if *start < cursor {
            return Err(format!(
                "unit interval {index} overlaps or is out of order; no silent interval unions are allowed"
            ));
        }
        if *start > cursor {
            partition.push(UnitSegmentV1 {
                start_frame: cursor,
                end_frame: *start,
                category: CategoryV1::Gap,
            });
        }
        partition.push(UnitSegmentV1 {
            start_frame: *start,
            end_frame: *end,
            category: CategoryV1::Value(value.clone()),
        });
        cursor = *end;
    }
    if cursor < continuum_frames {
        partition.push(UnitSegmentV1 {
            start_frame: cursor,
            end_frame: continuum_frames,
            category: CategoryV1::Gap,
        });
    }
    if partition.is_empty() {
        return Err("unitizing partition cannot be empty".into());
    }
    Ok(partition)
}

fn compute_unitizing_profile(
    partitions: &[Vec<UnitSegmentV1>],
    continuum_frames: u64,
) -> Result<UnitizingAlphaProfileV1, String> {
    if partitions.len() < 2 {
        return Err("unitizing alpha requires at least two eligible observers".into());
    }
    if continuum_frames == 0 || continuum_frames > MAX_EXACT_F64_INTEGER {
        return Err("continuum length must be positive and exactly representable".into());
    }
    for (rater, partition) in partitions.iter().enumerate() {
        if partition.is_empty()
            || partition[0].start_frame != 0
            || partition.last().map(|segment| segment.end_frame) != Some(continuum_frames)
        {
            return Err(format!("observer {rater} does not partition the whole continuum"));
        }
        for (index, pair) in partition.windows(2).enumerate() {
            if pair[0].end_frame != pair[1].start_frame {
                return Err(format!(
                    "observer {rater} partition has a gap in its explicit segmentation at boundary {index}"
                ));
            }
        }
    }

    let mut category_set = BTreeSet::new();
    category_set.insert(CategoryV1::Gap);
    for partition in partitions {
        for segment in partition {
            category_set.insert(segment.category.clone());
        }
    }
    let categories: Vec<_> = category_set.into_iter().collect();
    let category_indices: BTreeMap<_, _> = categories
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, category)| (category, index))
        .collect();
    let category_count = categories.len();
    let observer_count = partitions.len();

    let mut observed = vec![vec![0.0_f64; category_count]; category_count];
    for left_observer in 0..observer_count {
        for right_observer in 0..observer_count {
            if left_observer == right_observer {
                continue;
            }
            for left_segment in &partitions[left_observer] {
                for right_segment in &partitions[right_observer] {
                    let overlap_start =
                        left_segment.start_frame.max(right_segment.start_frame);
                    let overlap_end = left_segment.end_frame.min(right_segment.end_frame);
                    if overlap_end <= overlap_start {
                        continue;
                    }
                    let left_category = category_indices[&left_segment.category];
                    let right_category = category_indices[&right_segment.category];
                    observed[left_category][right_category] +=
                        (overlap_end - overlap_start) as f64;
                }
            }
        }
    }
    let pair_scale = (observer_count - 1) as f64;
    for row in &mut observed {
        for value in row {
            *value /= pair_scale;
        }
    }

    let observed_total: f64 = observed.iter().flatten().sum();
    let expected_observed_total = observer_count as f64 * continuum_frames as f64;
    let tolerance = (expected_observed_total * 1.0e-12).max(1.0e-8);
    if !observed_total.is_finite()
        || (observed_total - expected_observed_total).abs() > tolerance
    {
        return Err("observed coincidence total does not equal m × continuum length".into());
    }

    let observed_row: Vec<f64> = observed.iter().map(|row| row.iter().sum()).collect();
    let observed_column: Vec<f64> = (0..category_count)
        .map(|column| observed.iter().map(|row| row[column]).sum())
        .collect();

    let mut correction_by_category = vec![0.0_f64; category_count];
    for partition in partitions {
        for segment in partition {
            let index = category_indices[&segment.category];
            let length = (segment.end_frame - segment.start_frame) as f64;
            let correction = match &segment.category {
                CategoryV1::Gap => length,
                CategoryV1::Value(_) => length * length,
            };
            correction_by_category[index] += correction;
        }
    }

    let expected = expected_matrix(
        &observed,
        &observed_row,
        &observed_column,
        &correction_by_category,
    )?;
    let nominal = disagreement_pair(&observed, &expected);
    let u_alpha_nominal = alpha_from_disagreements(nominal.observed, nominal.expected);

    let gap_index = category_indices[&CategoryV1::Gap];
    let binary_observed = collapse_binary_matrix(&observed, gap_index);
    let binary_expected = collapse_binary_matrix(&expected, gap_index);
    let binary = disagreement_pair(&binary_observed, &binary_expected);
    let u_alpha_binary = alpha_from_disagreements(binary.observed, binary.expected);

    let non_gap_indices: Vec<usize> = (0..category_count)
        .filter(|index| *index != gap_index)
        .collect();
    let (cu_nominal, per_value) = if non_gap_indices.is_empty() {
        (None, Vec::new())
    } else {
        conditional_coding_profile(
            partitions,
            &observed,
            &observed_column,
            &categories,
            &non_gap_indices,
        )?
    };

    Ok(UnitizingAlphaProfileV1 {
        observers: observer_count,
        continuum_frames,
        category_count_including_gap: category_count,
        observed_disagreement_u_nominal: nominal.observed,
        expected_disagreement_u_nominal: nominal.expected,
        u_alpha_nominal,
        observed_disagreement_u_binary: binary.observed,
        expected_disagreement_u_binary: binary.expected,
        u_alpha_binary,
        observed_disagreement_cu_nominal: cu_nominal
            .as_ref()
            .map(|value| value.observed),
        expected_disagreement_cu_nominal: cu_nominal
            .as_ref()
            .map(|value| value.expected),
        cu_alpha_nominal: cu_nominal.and_then(|value| value.alpha),
        per_value,
    })
}

fn expected_matrix(
    observed: &[Vec<f64>],
    row_margins: &[f64],
    column_margins: &[f64],
    correction_by_category: &[f64],
) -> Result<Vec<Vec<f64>>, String> {
    let total: f64 = observed.iter().flatten().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("coincidence total must be finite and positive".into());
    }
    let correction_total: f64 = correction_by_category.iter().sum();
    let denominator = total - correction_total / total;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("unitizing expected-coincidence denominator is non-positive".into());
    }
    let categories = row_margins.len();
    let mut expected = vec![vec![0.0_f64; categories]; categories];
    for row in 0..categories {
        for column in 0..categories {
            let mut numerator = row_margins[row] * column_margins[column];
            if row == column {
                numerator -= correction_by_category[row];
            }
            expected[row][column] = numerator / denominator;
            if !expected[row][column].is_finite() {
                return Err("expected coincidence is non-finite".into());
            }
        }
    }
    Ok(expected)
}

fn collapse_binary_matrix(matrix: &[Vec<f64>], gap_index: usize) -> Vec<Vec<f64>> {
    let category_count = matrix.len();
    let mut binary = vec![vec![0.0_f64; 2]; 2];
    for row in 0..category_count {
        for column in 0..category_count {
            let row_class = usize::from(row != gap_index);
            let column_class = usize::from(column != gap_index);
            binary[row_class][column_class] += matrix[row][column];
        }
    }
    binary
}

fn disagreement_pair(
    observed: &[Vec<f64>],
    expected: &[Vec<f64>],
) -> DisagreementPairV1 {
    let mut observed_total = 0.0_f64;
    let mut expected_total = 0.0_f64;
    let mut observed_diagonal = 0.0_f64;
    let mut expected_diagonal = 0.0_f64;
    for row in 0..observed.len() {
        for column in 0..observed.len() {
            if row == column {
                observed_diagonal += observed[row][column];
                expected_diagonal += expected[row][column];
            }
            observed_total += observed[row][column];
            expected_total += expected[row][column];
        }
    }
    let observed_disagreement = observed_total - observed_diagonal;
    let expected_disagreement = expected_total - expected_diagonal;
    DisagreementPairV1 {
        observed: observed_disagreement,
        expected: expected_disagreement,
        alpha: alpha_from_disagreements(observed_disagreement, expected_disagreement),
    }
}

fn alpha_from_disagreements(observed: f64, expected: f64) -> Option<f64> {
    if !observed.is_finite() || !expected.is_finite() {
        return None;
    }
    if expected.abs() <= NUMERIC_EPSILON {
        return if observed.abs() <= NUMERIC_EPSILON {
            Some(1.0)
        } else {
            None
        };
    }
    let value = 1.0 - observed / expected;
    value.is_finite().then_some(value)
}

fn conditional_coding_profile(
    partitions: &[Vec<UnitSegmentV1>],
    observed_full: &[Vec<f64>],
    observed_full_column: &[f64],
    categories: &[CategoryV1],
    non_gap_indices: &[usize],
) -> Result<(Option<DisagreementPairV1>, Vec<UnitizingPerValueAlphaV1>), String> {
    let size = non_gap_indices.len();
    let mut observed = vec![vec![0.0_f64; size]; size];
    for row in 0..size {
        for column in 0..size {
            observed[row][column] =
                observed_full[non_gap_indices[row]][non_gap_indices[column]];
        }
    }
    let observed_total: f64 = observed.iter().flatten().sum();
    if observed_total <= NUMERIC_EPSILON {
        return Ok((None, Vec::new()));
    }

    let mut pairable_correction = vec![0.0_f64; size];
    for (observer_index, partition) in partitions.iter().enumerate() {
        let other_units: Vec<(u64, u64)> = partitions
            .iter()
            .enumerate()
            .filter(|(other_index, _)| *other_index != observer_index)
            .flat_map(|(_, other_partition)| {
                other_partition.iter().filter_map(|segment| {
                    matches!(&segment.category, CategoryV1::Value(_))
                        .then_some((segment.start_frame, segment.end_frame))
                })
            })
            .collect();
        let other_union = merge_ranges(&other_units);
        for segment in partition {
            let CategoryV1::Value(value) = &segment.category else {
                continue;
            };
            let label_index = non_gap_indices
                .iter()
                .position(|index| categories[*index] == CategoryV1::Value(value.clone()))
                .ok_or_else(|| "unitizing category index is missing".to_string())?;
            let pairable_length =
                overlap_with_union(segment.start_frame, segment.end_frame, &other_union)?;
            let length = pairable_length as f64;
            pairable_correction[label_index] += length * length;
        }
    }

    let mut row_margins = vec![0.0_f64; size];
    let mut column_margins = vec![0.0_f64; size];
    for row in 0..size {
        row_margins[row] = observed[row].iter().sum();
        column_margins[row] = (0..size).map(|column| observed[column][row]).sum();
    }
    let expected = expected_matrix(
        &observed,
        &row_margins,
        &column_margins,
        &pairable_correction,
    )?;
    let disagreement = disagreement_pair(&observed, &expected);

    let mut per_value = Vec::with_capacity(size);
    for value_index in 0..size {
        let observed_pairable_column = column_margins[value_index];
        let observed_same_value = observed[value_index][value_index];
        let expected_column: f64 = expected
            .iter()
            .map(|row| row[value_index])
            .sum();
        let expected_same_value = expected[value_index][value_index];
        let full_column = observed_full_column[non_gap_indices[value_index]];
        let coverage = if full_column > 0.0 {
            (observed_pairable_column / full_column).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let observed_value_disagreement =
            observed_pairable_column - observed_same_value;
        let expected_value_disagreement = expected_column - expected_same_value;
        let alpha = if expected_value_disagreement.abs() <= NUMERIC_EPSILON
            || observed_pairable_column.abs() <= NUMERIC_EPSILON
        {
            if observed_value_disagreement.abs() <= NUMERIC_EPSILON {
                Some(1.0)
            } else {
                None
            }
        } else {
            let value = 1.0
                - (expected_column / observed_pairable_column)
                    * (observed_value_disagreement / expected_value_disagreement);
            value.is_finite().then_some(value)
        };
        let CategoryV1::Value(value) = &categories[non_gap_indices[value_index]] else {
            return Err("gap unexpectedly appears in per-value unitizing profile".into());
        };
        per_value.push(UnitizingPerValueAlphaV1 {
            value: value.clone(),
            alpha,
            overlap_coverage: coverage,
        });
    }

    Ok((Some(disagreement), per_value))
}

fn merge_ranges(ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(sorted.len());
    for (start, end) in sorted {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

fn overlap_with_union(start: u64, end: u64, union: &[(u64, u64)]) -> Result<u64, String> {
    let mut total = 0_u64;
    for (union_start, union_end) in union {
        let overlap_start = start.max(*union_start);
        let overlap_end = end.min(*union_end);
        if overlap_end > overlap_start {
            total = total
                .checked_add(overlap_end - overlap_start)
                .ok_or_else(|| "pairable unitizing length overflow".to_string())?;
        }
    }
    Ok(total)
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labeled_partition(
        intervals: &[(u64, u64, &str)],
        continuum: u64,
    ) -> Vec<UnitSegmentV1> {
        let intervals: Vec<_> = intervals
            .iter()
            .map(|(start, end, value)| (*start, *end, (*value).to_string()))
            .collect();
        build_partition(&intervals, continuum).unwrap()
    }

    fn published_book_example() -> Vec<Vec<UnitSegmentV1>> {
        let alex = labeled_partition(
            &[
                (0, 7, "1"),
                (7, 13, "2"),
                (30, 32, "5"),
                (32, 37, "6"),
                (37, 44, "6"),
                (44, 60, "7"),
            ],
            60,
        );
        let paul = labeled_partition(
            &[(0, 7, "1"), (7, 13, "3"), (13, 21, "5"), (32, 44, "6")],
            60,
        );
        let suzan = labeled_partition(
            &[
                (0, 7, "1"),
                (7, 13, "4"),
                (13, 21, "5"),
                (32, 37, "6"),
                (37, 41, "6"),
                (41, 44, "6"),
            ],
            60,
        );
        vec![alex, paul, suzan]
    }

    #[test]
    fn reproduces_krippendorff_2019_fourth_edition_figure_12_11() {
        let profile = compute_unitizing_profile(&published_book_example(), 60).unwrap();
        assert!((profile.u_alpha_nominal.unwrap() - 0.510).abs() < 0.001);
        assert!((profile.u_alpha_binary.unwrap() - 0.413).abs() < 0.002);
        assert!((profile.cu_alpha_nominal.unwrap() - 0.72835).abs() < 0.001);
        let by_value: BTreeMap<_, _> = profile
            .per_value
            .iter()
            .map(|value| (value.value.as_str(), value))
            .collect();
        assert_eq!(by_value["1"].alpha, Some(1.0));
        assert_eq!(by_value["2"].alpha, Some(0.0));
        assert_eq!(by_value["3"].alpha, Some(0.0));
        assert_eq!(by_value["4"].alpha, Some(0.0));
        assert_eq!(by_value["5"].alpha, Some(1.0));
        assert_eq!(by_value["6"].alpha, Some(1.0));
        assert_eq!(by_value["7"].overlap_coverage, 0.0);
        assert!((by_value["5"].overlap_coverage - 8.0 / 18.0).abs() < 0.01);
    }

    #[test]
    fn identical_two_rater_unitizations_have_perfect_overall_and_binary_alpha() {
        let intervals = [(0, 10, "event"), (20, 30, "event")];
        let partitions = vec![
            labeled_partition(&intervals, 40),
            labeled_partition(&intervals, 40),
        ];
        let profile = compute_unitizing_profile(&partitions, 40).unwrap();
        assert_eq!(profile.u_alpha_nominal, Some(1.0));
        assert_eq!(profile.u_alpha_binary, Some(1.0));
        assert_eq!(profile.cu_alpha_nominal, Some(1.0));
    }

    #[test]
    fn one_category_event_queries_report_nominal_and_binary_alpha_equally() {
        let partitions = vec![
            labeled_partition(&[(0, 10, "event"), (20, 30, "event")], 40),
            labeled_partition(&[(0, 10, "event"), (21, 31, "event")], 40),
            labeled_partition(&[(0, 10, "event"), (22, 32, "event")], 40),
        ];
        let profile = compute_unitizing_profile(&partitions, 40).unwrap();
        assert_eq!(profile.u_alpha_nominal, profile.u_alpha_binary);
    }

    #[test]
    fn all_absent_raters_agree_on_a_full_gap_partition() {
        let partitions = vec![build_partition(&[], 100).unwrap(); 3];
        let profile = compute_unitizing_profile(&partitions, 100).unwrap();
        assert_eq!(profile.u_alpha_nominal, Some(1.0));
        assert_eq!(profile.u_alpha_binary, Some(1.0));
        assert_eq!(profile.cu_alpha_nominal, None);
    }

    #[test]
    fn disjoint_and_partial_units_are_scored_without_merging_intervals() {
        let partitions = vec![
            labeled_partition(&[(0, 10, "event")], 30),
            labeled_partition(&[(20, 30, "event")], 30),
        ];
        let profile = compute_unitizing_profile(&partitions, 30).unwrap();
        assert!(profile.u_alpha_nominal.unwrap() < 0.0);
        assert_eq!(profile.cu_alpha_nominal, None);
    }

    #[test]
    fn a_single_observer_is_not_a_reliability_estimate() {
        let partition = labeled_partition(&[(0, 10, "event")], 30);
        assert!(compute_unitizing_profile(&[partition], 30).is_err());
    }

    #[test]
    fn observer_order_and_exact_frame_rescaling_do_not_change_unitizing_coefficients() {
        let original = published_book_example();
        let baseline = compute_unitizing_profile(&original, 60).unwrap();

        let mut reordered = original.clone();
        reordered.reverse();
        let reordered_profile = compute_unitizing_profile(&reordered, 60).unwrap();
        assert!((baseline.u_alpha_nominal.unwrap() - reordered_profile.u_alpha_nominal.unwrap()).abs() < 1e-12);
        assert!((baseline.u_alpha_binary.unwrap() - reordered_profile.u_alpha_binary.unwrap()).abs() < 1e-12);
        assert!((baseline.cu_alpha_nominal.unwrap() - reordered_profile.cu_alpha_nominal.unwrap()).abs() < 1e-12);

        let scale = 7_u64;
        let scaled: Vec<Vec<UnitSegmentV1>> = original
            .iter()
            .map(|partition| {
                partition
                    .iter()
                    .map(|segment| UnitSegmentV1 {
                        start_frame: segment.start_frame * scale,
                        end_frame: segment.end_frame * scale,
                        category: segment.category.clone(),
                    })
                    .collect()
            })
            .collect();
        let scaled_profile = compute_unitizing_profile(&scaled, 60 * scale).unwrap();
        assert!((baseline.u_alpha_nominal.unwrap() - scaled_profile.u_alpha_nominal.unwrap()).abs() < 1e-12);
        assert!((baseline.u_alpha_binary.unwrap() - scaled_profile.u_alpha_binary.unwrap()).abs() < 1e-12);
        assert!((baseline.cu_alpha_nominal.unwrap() - scaled_profile.cu_alpha_nominal.unwrap()).abs() < 1e-12);
    }

    fn test_digest(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn test_annotation(
        annotator: char,
        receipt: char,
        state: GroundTruthStateV1,
        intervals: Vec<FrameIntervalV1>,
        reason: Option<&str>,
    ) -> IndependentTemporalAnnotationV1 {
        IndependentTemporalAnnotationV1 {
            annotator_id_sha256: test_digest(annotator),
            annotation_sha256: test_digest(receipt),
            state,
            intervals,
            status_reason: reason.map(str::to_owned),
        }
    }

    fn test_item(
        item_id: &str,
        annotations: Vec<IndependentTemporalAnnotationV1>,
    ) -> ReliabilityCorpusItemV1 {
        ReliabilityCorpusItemV1 {
            item_id: item_id.into(),
            source_work_cluster_sha256: test_digest('1'),
            query_id: format!("query-{item_id}"),
            query_text: "Every piano entrance".into(),
            artifact: AudioArtifactClockV1 {
                audio_sha256: test_digest('a'),
                sample_rate_hz: 48_000,
                channels: 2,
                total_frames: 1_000,
            },
            annotation_policy_version: "unitizing-test-policy-v1".into(),
            annotations,
        }
    }

    #[test]
    fn corpus_report_preserves_original_intervals_and_exclusion_reasons() {
        let event = FrameIntervalV1 {
            start_frame: 100,
            end_frame: 200,
        };
        let ambiguous_interval = FrameIntervalV1 {
            start_frame: 90,
            end_frame: 220,
        };
        let item = test_item(
            "piano",
            vec![
                test_annotation('1', 'a', GroundTruthStateV1::Present, vec![event.clone()], None),
                test_annotation('2', 'b', GroundTruthStateV1::Absent, Vec::new(), None),
                test_annotation(
                    '3',
                    'c',
                    GroundTruthStateV1::Ambiguous,
                    vec![ambiguous_interval.clone()],
                    Some("gradual onset has no stable event boundary"),
                ),
                test_annotation(
                    '4',
                    'd',
                    GroundTruthStateV1::NotEstimable,
                    Vec::new(),
                    Some("masking prevents reliable event identification"),
                ),
            ],
        );
        let case = UnitizingCorpusCaseV1 {
            schema: CASE_SCHEMA_V1.into(),
            annotation_manifest_sha256: test_digest('f'),
            minimum_eligible_unitizers_per_item: 2,
            items: vec![item],
        };
        let report = audit_unitizing_corpus(&case).unwrap();
        let result = &report.item_results[0];
        assert_eq!(result.status, UnitizingItemStatusV1::Scored);
        assert_eq!(result.eligible_unitizers, 2);
        assert_eq!(result.excluded_ambiguous, 1);
        assert_eq!(result.excluded_not_estimable, 1);
        assert_eq!(result.query_text, "Every piano entrance");
        assert_eq!(result.sample_rate_hz, 48_000);
        assert_eq!(result.annotator_dispositions[0].intervals, vec![event]);
        assert_eq!(result.annotator_dispositions[2].intervals, vec![ambiguous_interval]);
        assert_eq!(
            result.annotator_dispositions[2].exclusion_reason,
            Some(UnitizingExclusionReasonV1::Ambiguous)
        );
        assert_eq!(
            result.annotator_dispositions[2].status_reason.as_deref(),
            Some("gradual onset has no stable event boundary")
        );
        assert!(report.unitizing_alpha_computed);
        assert!(!report.consensus_ground_truth_created);
        assert!(!report.annotation_receipts_authenticated);
    }

    #[test]
    fn all_ambiguous_or_not_estimable_panel_is_not_reported_as_absence() {
        let item = test_item(
            "insufficient",
            vec![
                test_annotation(
                    '1',
                    'a',
                    GroundTruthStateV1::Present,
                    vec![FrameIntervalV1 {
                        start_frame: 100,
                        end_frame: 200,
                    }],
                    None,
                ),
                test_annotation(
                    '2',
                    'b',
                    GroundTruthStateV1::Ambiguous,
                    Vec::new(),
                    Some("uncertain whether this is the target instrument"),
                ),
                test_annotation(
                    '3',
                    'c',
                    GroundTruthStateV1::NotEstimable,
                    Vec::new(),
                    Some("audio quality is insufficient"),
                ),
            ],
        );
        let case = UnitizingCorpusCaseV1 {
            schema: CASE_SCHEMA_V1.into(),
            annotation_manifest_sha256: test_digest('f'),
            minimum_eligible_unitizers_per_item: 2,
            items: vec![item],
        };
        let report = audit_unitizing_corpus(&case).unwrap();
        let result = &report.item_results[0];
        assert_eq!(result.status, UnitizingItemStatusV1::InsufficientEligibleUnitizers);
        assert_eq!(result.eligible_unitizers, 1);
        assert_eq!(result.excluded_ambiguous, 1);
        assert_eq!(result.excluded_not_estimable, 1);
        assert!(result.coefficients.is_none());
        assert!(!report.unitizing_alpha_computed);
    }

    #[test]
    fn same_rater_overlapping_or_unsorted_intervals_are_rejected() {
        let overlap = build_partition(
            &[(0, 10, "event".into()), (9, 15, "event".into())],
            30,
        );
        assert!(overlap.is_err());
        let unsorted = build_partition(
            &[(10, 15, "event".into()), (0, 5, "event".into())],
            30,
        );
        assert!(unsorted.is_err());
    }
}
