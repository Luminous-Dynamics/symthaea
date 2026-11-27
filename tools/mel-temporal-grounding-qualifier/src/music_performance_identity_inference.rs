//! Source-work-clustered paired inference for property-relative musical identity.
//!
//! This module composes the authoritative performance-identity reducer. It
//! never re-decodes answers or re-scores CLEAN / Performance-STAY / Musical-FLIP.
//! Its only statistical unit is the source-work cluster.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::music_performance_identity::{
    MusicalIdentityTriadResultV1, MusicalPerformanceIdentityCaseV1,
    MusicalPerformanceIdentityReportV1, audit_musical_performance_identity,
};

const CASE_SCHEMA_V1: &str = "mel-music-performance-identity-comparison-v1";
const REPORT_SCHEMA_V1: &str = "mel-music-performance-identity-comparison-report-v1";
const BOOTSTRAP_METHOD_V1: &str = "paired-source-work-cluster-percentile-bootstrap-v1";
const DEFAULT_MINIMUM_CLUSTERS_V1: usize = 30;
const DEFAULT_REPLICATES_V1: usize = 2_000;
const DEFAULT_CONFIDENCE_LEVEL_V1: f64 = 0.95;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedInferencePolicyV1 {
    pub minimum_source_work_clusters: usize,
    pub bootstrap_replicates: usize,
    pub confidence_level: f64,
    pub seed: u64,
}

impl Default for PairedInferencePolicyV1 {
    fn default() -> Self {
        Self {
            minimum_source_work_clusters: DEFAULT_MINIMUM_CLUSTERS_V1,
            bootstrap_replicates: DEFAULT_REPLICATES_V1,
            confidence_level: DEFAULT_CONFIDENCE_LEVEL_V1,
            seed: 0x4d45_4c48_4541_5234,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedMusicalIdentityComparisonCaseV1 {
    pub schema: String,
    pub baseline: MusicalPerformanceIdentityCaseV1,
    pub candidate: MusicalPerformanceIdentityCaseV1,
    pub inference_policy: PairedInferencePolicyV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairedInferenceStatusV1 {
    SourceWorkClusterFloorMet,
    InsufficientSourceWorkClusters,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairedMetricEstimateV1 {
    pub baseline: f64,
    pub candidate: f64,
    /// Positive values always mean the candidate improved the named metric.
    pub candidate_improvement: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PercentileIntervalV1 {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairedMetricInferenceV1 {
    pub estimate: PairedMetricEstimateV1,
    pub candidate_improvement_interval: Option<PercentileIntervalV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Debug, Clone, PartialEq, Serialize)]
pub struct PairedMusicalIdentityStratumV1 {
    pub target_property: crate::music_performance_identity::MusicalIdentityPropertyV1,
    pub performance_stay_factor: Option<crate::music_performance_identity::PerformanceStayFactorV1>,
    pub musical_flip_factor: Option<crate::music_performance_identity::MusicalFlipFactorV1>,
    pub item_count: usize,
    pub source_work_clusters: usize,
    pub status: PairedInferenceStatusV1,
    pub performance_stay_false_difference: PairedMetricInferenceV1,
    pub musical_flip_miss: PairedMetricInferenceV1,
    pub joint_exact: PairedMetricInferenceV1,
}

pub struct PairedMusicalIdentityComparisonReportV1 {
    pub schema: &'static str,
    pub bootstrap_method: &'static str,
    pub status: PairedInferenceStatusV1,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub dataset_revision_sha256: String,
    pub evaluator_protocol_sha256: String,
    pub baseline_model_identity_sha256: String,
    pub candidate_model_identity_sha256: String,
    pub item_count: usize,
    pub source_work_clusters: usize,
    pub minimum_source_work_clusters: usize,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed: u64,
    pub confidence_level: f64,
    /// Equal-source-work estimand: each cluster contributes one cluster mean,
    /// irrespective of how many triads that source contributes.
    pub performance_stay_false_difference: PairedMetricInferenceV1,
    pub musical_flip_miss: PairedMetricInferenceV1,
    pub joint_exact: PairedMetricInferenceV1,
    /// Separate strata; no pooled score can hide property/factor failures.
    pub by_target_property: Vec<PairedMusicalIdentityStratumV1>,
    pub by_performance_stay_factor: Vec<PairedMusicalIdentityStratumV1>,
    pub by_musical_flip_factor: Vec<PairedMusicalIdentityStratumV1>,
    pub source_work_lineage_authenticated: bool,
    pub target_property_annotations_authenticated: bool,
    pub audio_bytes_recomputed: bool,
    pub model_invoked_by_evaluator: bool,
    pub population_generalization_established: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct ClusterOutcomes {
    baseline_stay_error: f64,
    candidate_stay_error: f64,
    baseline_flip_error: f64,
    candidate_flip_error: f64,
    baseline_joint_exact: f64,
    candidate_joint_exact: f64,
}

#[derive(Debug, Clone, Copy)]
struct PointEstimates {
    baseline_stay_error: f64,
    candidate_stay_error: f64,
    baseline_flip_error: f64,
    candidate_flip_error: f64,
    baseline_joint_exact: f64,
    candidate_joint_exact: f64,
    stay_improvement: f64,
    flip_improvement: f64,
    joint_improvement: f64,
}

pub fn audit_paired_musical_identity_json(
    input: &str,
) -> Result<PairedMusicalIdentityComparisonReportV1, String> {
    let case: PairedMusicalIdentityComparisonCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid paired musical identity comparison JSON: {error}"))?;
    audit_paired_musical_identity_comparison(&case)
}

pub fn audit_paired_musical_identity_comparison(
    case: &PairedMusicalIdentityComparisonCaseV1,
) -> Result<PairedMusicalIdentityComparisonReportV1, String> {
    if case.schema != CASE_SCHEMA_V1 {
        return Err(format!(
            "unsupported paired comparison schema: {}",
            case.schema
        ));
    }
    validate_policy(&case.inference_policy)?;
    validate_pair_structure(&case.baseline, &case.candidate)?;

    // The existing reducer remains the only owner of musical semantics and
    // model-answer decoding. Invalid baseline or candidate input stops here.
    let baseline_report = audit_musical_performance_identity(&case.baseline)?;
    let candidate_report = audit_musical_performance_identity(&case.candidate)?;
    let baseline_model = case.baseline.triads[0].model_identity_sha256.clone();
    let candidate_model = case.candidate.triads[0].model_identity_sha256.clone();

    let baseline_results = index_results(&baseline_report)?;
    let candidate_results = index_results(&candidate_report)?;
    let clusters = paired_cluster_outcomes(&baseline_results, &candidate_results, &case.baseline)?;
    let cluster_count = clusters.len();
    if cluster_count == 0 {
        return Err("paired comparison has no source-work clusters".into());
    }

    let observed = summarize_cluster_means(&clusters);
    let status = if cluster_count >= case.inference_policy.minimum_source_work_clusters {
        PairedInferenceStatusV1::SourceWorkClusterFloorMet
    } else {
        PairedInferenceStatusV1::InsufficientSourceWorkClusters
    };

    let intervals = if status == PairedInferenceStatusV1::SourceWorkClusterFloorMet {
        Some(bootstrap_intervals(&clusters, &case.inference_policy)?)
    } else {
        None
    };

    let stay = PairedMetricInferenceV1 {
        estimate: PairedMetricEstimateV1 {
            baseline: observed.baseline_stay_error,
            candidate: observed.candidate_stay_error,
            candidate_improvement: observed.stay_improvement,
        },
        candidate_improvement_interval: intervals.as_ref().map(|v| v.0.clone()),
    };
    let flip = PairedMetricInferenceV1 {
        estimate: PairedMetricEstimateV1 {
            baseline: observed.baseline_flip_error,
            candidate: observed.candidate_flip_error,
            candidate_improvement: observed.flip_improvement,
        },
        candidate_improvement_interval: intervals.as_ref().map(|v| v.1.clone()),
    };
    let joint = PairedMetricInferenceV1 {
        estimate: PairedMetricEstimateV1 {
            baseline: observed.baseline_joint_exact,
            candidate: observed.candidate_joint_exact,
            candidate_improvement: observed.joint_improvement,
        },
        candidate_improvement_interval: intervals.as_ref().map(|v| v.2.clone()),
    };

    let by_target_property = case
        .baseline
        .triads
        .iter()
        .map(|triad| triad.target_property)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|property| {
            build_stratum(
                StratumSelection::Property(property),
                &baseline_results,
                &candidate_results,
                &case.baseline,
                &case.inference_policy,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;

    let by_performance_stay_factor = case
        .baseline
        .triads
        .iter()
        .map(|triad| (triad.target_property, triad.performance_stay_factor))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|(property, factor)| {
            build_stratum(
                StratumSelection::PerformanceStay(property, factor),
                &baseline_results,
                &candidate_results,
                &case.baseline,
                &case.inference_policy,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;

    let by_musical_flip_factor = case
        .baseline
        .triads
        .iter()
        .map(|triad| (triad.target_property, triad.musical_flip_factor))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|(property, factor)| {
            build_stratum(
                StratumSelection::MusicalFlip(property, factor),
                &baseline_results,
                &candidate_results,
                &case.baseline,
                &case.inference_policy,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(PairedMusicalIdentityComparisonReportV1 {
        schema: REPORT_SCHEMA_V1,
        bootstrap_method: BOOTSTRAP_METHOD_V1,
        status,
        benchmark_id: case.baseline.benchmark_id.clone(),
        benchmark_version: case.baseline.benchmark_version.clone(),
        dataset_revision_sha256: case.baseline.dataset_revision_sha256.clone(),
        evaluator_protocol_sha256: case.baseline.evaluator_protocol_sha256.clone(),
        baseline_model_identity_sha256: baseline_model,
        candidate_model_identity_sha256: candidate_model,
        item_count: case.baseline.triads.len(),
        source_work_clusters: cluster_count,
        minimum_source_work_clusters: case.inference_policy.minimum_source_work_clusters,
        bootstrap_replicates: case.inference_policy.bootstrap_replicates,
        bootstrap_seed: case.inference_policy.seed,
        confidence_level: case.inference_policy.confidence_level,
        performance_stay_false_difference: stay,
        musical_flip_miss: flip,
        joint_exact: joint,
        by_target_property,
        by_performance_stay_factor,
        by_musical_flip_factor,
        source_work_lineage_authenticated: false,
        target_property_annotations_authenticated: false,
        audio_bytes_recomputed: false,
        model_invoked_by_evaluator: false,
        population_generalization_established: false,
    })
}

fn validate_policy(policy: &PairedInferencePolicyV1) -> Result<(), String> {
    if policy.minimum_source_work_clusters < DEFAULT_MINIMUM_CLUSTERS_V1 {
        return Err("minimum_source_work_clusters must be at least 30".into());
    }
    if !(100..=100_000).contains(&policy.bootstrap_replicates) {
        return Err("bootstrap_replicates must be between 100 and 100000".into());
    }
    if !policy.confidence_level.is_finite() || !(0.80..0.999).contains(&policy.confidence_level) {
        return Err("confidence_level must be finite and in [0.80, 0.999)".into());
    }
    Ok(())
}

fn validate_pair_structure(
    baseline: &MusicalPerformanceIdentityCaseV1,
    candidate: &MusicalPerformanceIdentityCaseV1,
) -> Result<(), String> {
    if baseline.schema != candidate.schema
        || baseline.benchmark_id != candidate.benchmark_id
        || baseline.benchmark_version != candidate.benchmark_version
        || baseline.dataset_revision_sha256 != candidate.dataset_revision_sha256
        || baseline.evaluator_protocol_sha256 != candidate.evaluator_protocol_sha256
    {
        return Err("baseline and candidate benchmark/evaluator identities differ".into());
    }
    if baseline.triads.len() != candidate.triads.len() {
        return Err("baseline and candidate item counts differ".into());
    }

    let baseline_items = triad_map(baseline)?;
    let candidate_items = triad_map(candidate)?;
    if baseline_items.len() != candidate_items.len() {
        return Err("baseline and candidate item identity sets differ".into());
    }
    for (item_id, baseline_triad) in baseline_items {
        let candidate_triad = candidate_items
            .get(item_id)
            .ok_or_else(|| format!("candidate is missing item {item_id}"))?;
        if baseline_triad.pair_id != candidate_triad.pair_id
            || baseline_triad.source_work_lineage_sha256
                != candidate_triad.source_work_lineage_sha256
            || baseline_triad.target_property != candidate_triad.target_property
            || baseline_triad.performance_stay_factor != candidate_triad.performance_stay_factor
            || baseline_triad.musical_flip_factor != candidate_triad.musical_flip_factor
            || baseline_triad.target_property_reference_sha256
                != candidate_triad.target_property_reference_sha256
            || baseline_triad.pairing_manifest_sha256 != candidate_triad.pairing_manifest_sha256
            || !variant_structure_matches(&baseline_triad.clean, &candidate_triad.clean)
            || !variant_structure_matches(
                &baseline_triad.performance_stay,
                &candidate_triad.performance_stay,
            )
            || !variant_structure_matches(
                &baseline_triad.musical_flip,
                &candidate_triad.musical_flip,
            )
        {
            return Err(format!(
                "baseline and candidate do not share the exact frozen trial structure for item {item_id}"
            ));
        }
    }
    Ok(())
}

fn triad_map<'a>(
    case: &'a MusicalPerformanceIdentityCaseV1,
) -> Result<
    BTreeMap<&'a str, &'a crate::music_performance_identity::MusicalPerformanceIdentityTriadV1>,
    String,
> {
    let mut output = BTreeMap::new();
    for triad in &case.triads {
        if output.insert(triad.item_id.as_str(), triad).is_some() {
            return Err(format!(
                "duplicate item_id in paired comparison: {}",
                triad.item_id
            ));
        }
    }
    Ok(output)
}

fn variant_structure_matches(
    baseline: &crate::music_performance_identity::MusicalIdentityVariantV1,
    candidate: &crate::music_performance_identity::MusicalIdentityVariantV1,
) -> bool {
    let left = &baseline.response;
    let right = &candidate.response;
    left.variant_id == right.variant_id
        && left.audio_sha256 == right.audio_sha256
        && left.question == right.question
        && left.question_sha256 == right.question_sha256
        && left.option_a_relation == right.option_a_relation
        && left.option_b_relation == right.option_b_relation
        && left.expected_relation == right.expected_relation
        && baseline.take_identity_sha256 == candidate.take_identity_sha256
        && baseline.performance_identity_sha256 == candidate.performance_identity_sha256
        && baseline.performer_identity_sha256 == candidate.performer_identity_sha256
        && baseline.instrument_family_sha256 == candidate.instrument_family_sha256
        && baseline.renderer_identity_sha256 == candidate.renderer_identity_sha256
}

fn index_results<'a>(
    report: &'a MusicalPerformanceIdentityReportV1,
) -> Result<BTreeMap<&'a str, &'a MusicalIdentityTriadResultV1>, String> {
    let mut output = BTreeMap::new();
    for result in &report.triad_results {
        if output.insert(result.item_id.as_str(), result).is_some() {
            return Err(format!(
                "reducer returned duplicate item_id: {}",
                result.item_id
            ));
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy)]
enum StratumSelection {
    Property(crate::music_performance_identity::MusicalIdentityPropertyV1),
    PerformanceStay(
        crate::music_performance_identity::MusicalIdentityPropertyV1,
        crate::music_performance_identity::PerformanceStayFactorV1,
    ),
    MusicalFlip(
        crate::music_performance_identity::MusicalIdentityPropertyV1,
        crate::music_performance_identity::MusicalFlipFactorV1,
    ),
}

impl StratumSelection {
    fn includes(
        self,
        triad: &crate::music_performance_identity::MusicalPerformanceIdentityTriadV1,
    ) -> bool {
        match self {
            Self::Property(property) => triad.target_property == property,
            Self::PerformanceStay(property, factor) => {
                triad.target_property == property && triad.performance_stay_factor == factor
            }
            Self::MusicalFlip(property, factor) => {
                triad.target_property == property && triad.musical_flip_factor == factor
            }
        }
    }

    fn fields(
        self,
    ) -> (
        crate::music_performance_identity::MusicalIdentityPropertyV1,
        Option<crate::music_performance_identity::PerformanceStayFactorV1>,
        Option<crate::music_performance_identity::MusicalFlipFactorV1>,
    ) {
        match self {
            Self::Property(property) => (property, None, None),
            Self::PerformanceStay(property, factor) => (property, Some(factor), None),
            Self::MusicalFlip(property, factor) => (property, None, Some(factor)),
        }
    }
}

fn build_stratum(
    selection: StratumSelection,
    baseline_results: &BTreeMap<&str, &MusicalIdentityTriadResultV1>,
    candidate_results: &BTreeMap<&str, &MusicalIdentityTriadResultV1>,
    baseline_case: &MusicalPerformanceIdentityCaseV1,
    policy: &PairedInferencePolicyV1,
) -> Result<PairedMusicalIdentityStratumV1, String> {
    let selected_case = MusicalPerformanceIdentityCaseV1 {
        schema: baseline_case.schema.clone(),
        benchmark_id: baseline_case.benchmark_id.clone(),
        benchmark_version: baseline_case.benchmark_version.clone(),
        dataset_revision_sha256: baseline_case.dataset_revision_sha256.clone(),
        evaluator_protocol_sha256: baseline_case.evaluator_protocol_sha256.clone(),
        triads: baseline_case
            .triads
            .iter()
            .filter(|triad| selection.includes(triad))
            .cloned()
            .collect(),
    };
    let selected_ids: BTreeSet<&str> = selected_case
        .triads
        .iter()
        .map(|triad| triad.item_id.as_str())
        .collect();
    if selected_ids.is_empty() {
        return Err("stratum unexpectedly contains no items".into());
    }
    let baseline_subset: BTreeMap<&str, &MusicalIdentityTriadResultV1> = baseline_results
        .iter()
        .filter(|(item_id, _)| selected_ids.contains(*item_id))
        .map(|(item_id, result)| (*item_id, *result))
        .collect();
    let candidate_subset: BTreeMap<&str, &MusicalIdentityTriadResultV1> = candidate_results
        .iter()
        .filter(|(item_id, _)| selected_ids.contains(**item_id))
        .map(|(item_id, result)| (*item_id, *result))
        .collect();
    let clusters = paired_cluster_outcomes(&baseline_subset, &candidate_subset, &selected_case)?;
    let cluster_count = clusters.len();
    if cluster_count == 0 {
        return Err("stratum has no source-work clusters".into());
    }
    let observed = summarize_cluster_means(&clusters);
    let status = if cluster_count >= policy.minimum_source_work_clusters {
        PairedInferenceStatusV1::SourceWorkClusterFloorMet
    } else {
        PairedInferenceStatusV1::InsufficientSourceWorkClusters
    };
    let intervals = if status == PairedInferenceStatusV1::SourceWorkClusterFloorMet {
        Some(bootstrap_intervals(&clusters, policy)?)
    } else {
        None
    };
    let metrics = (
        PairedMetricInferenceV1 {
            estimate: PairedMetricEstimateV1 {
                baseline: observed.baseline_stay_error,
                candidate: observed.candidate_stay_error,
                candidate_improvement: observed.stay_improvement,
            },
            candidate_improvement_interval: intervals.as_ref().map(|value| value.0.clone()),
        },
        PairedMetricInferenceV1 {
            estimate: PairedMetricEstimateV1 {
                baseline: observed.baseline_flip_error,
                candidate: observed.candidate_flip_error,
                candidate_improvement: observed.flip_improvement,
            },
            candidate_improvement_interval: intervals.as_ref().map(|value| value.1.clone()),
        },
        PairedMetricInferenceV1 {
            estimate: PairedMetricEstimateV1 {
                baseline: observed.baseline_joint_exact,
                candidate: observed.candidate_joint_exact,
                candidate_improvement: observed.joint_improvement,
            },
            candidate_improvement_interval: intervals.as_ref().map(|value| value.2.clone()),
        },
    );
    let (target_property, performance_stay_factor, musical_flip_factor) = selection.fields();
    Ok(PairedMusicalIdentityStratumV1 {
        target_property,
        performance_stay_factor,
        musical_flip_factor,
        item_count: selected_ids.len(),
        source_work_clusters: cluster_count,
        status,
        performance_stay_false_difference: metrics.0,
        musical_flip_miss: metrics.1,
        joint_exact: metrics.2,
    })
}

fn paired_cluster_outcomes(
    baseline: &BTreeMap<&str, &MusicalIdentityTriadResultV1>,
    candidate: &BTreeMap<&str, &MusicalIdentityTriadResultV1>,
    baseline_case: &MusicalPerformanceIdentityCaseV1,
) -> Result<Vec<ClusterOutcomes>, String> {
    if baseline.len() != candidate.len() || baseline.len() != baseline_case.triads.len() {
        return Err("reducer result count does not match paired case".into());
    }

    let mut by_cluster: BTreeMap<&str, Vec<ClusterOutcomes>> = BTreeMap::new();
    let case_items = triad_map(baseline_case)?;
    for (item_id, baseline_result) in baseline {
        let candidate_result = candidate
            .get(item_id)
            .ok_or_else(|| format!("candidate reducer result missing item {item_id}"))?;
        if baseline_result.pair_id != candidate_result.pair_id
            || baseline_result.source_work_lineage_sha256
                != candidate_result.source_work_lineage_sha256
            || baseline_result.target_property != candidate_result.target_property
        {
            return Err(format!(
                "paired reducer result identity mismatch for item {item_id}"
            ));
        }
        let triad = case_items
            .get(item_id)
            .ok_or_else(|| format!("case triad missing reducer item {item_id}"))?;
        by_cluster
            .entry(triad.source_work_lineage_sha256.as_str())
            .or_default()
            .push(ClusterOutcomes {
                baseline_stay_error: if baseline_result.performance_stay_correct {
                    0.0
                } else {
                    1.0
                },
                candidate_stay_error: if candidate_result.performance_stay_correct {
                    0.0
                } else {
                    1.0
                },
                baseline_flip_error: if baseline_result.musical_flip_correct {
                    0.0
                } else {
                    1.0
                },
                candidate_flip_error: if candidate_result.musical_flip_correct {
                    0.0
                } else {
                    1.0
                },
                baseline_joint_exact: if baseline_result.joint_exact {
                    1.0
                } else {
                    0.0
                },
                candidate_joint_exact: if candidate_result.joint_exact {
                    1.0
                } else {
                    0.0
                },
            });
    }

    let mut output = Vec::with_capacity(by_cluster.len());
    for (_, values) in by_cluster {
        let count = values.len() as f64;
        if count == 0.0 {
            return Err("empty source-work cluster encountered".into());
        }
        output.push(ClusterOutcomes {
            baseline_stay_error: values.iter().map(|v| v.baseline_stay_error).sum::<f64>() / count,
            candidate_stay_error: values.iter().map(|v| v.candidate_stay_error).sum::<f64>()
                / count,
            baseline_flip_error: values.iter().map(|v| v.baseline_flip_error).sum::<f64>() / count,
            candidate_flip_error: values.iter().map(|v| v.candidate_flip_error).sum::<f64>()
                / count,
            baseline_joint_exact: values.iter().map(|v| v.baseline_joint_exact).sum::<f64>()
                / count,
            candidate_joint_exact: values.iter().map(|v| v.candidate_joint_exact).sum::<f64>()
                / count,
        });
    }
    Ok(output)
}

fn summarize_cluster_means(clusters: &[ClusterOutcomes]) -> PointEstimates {
    let count = clusters.len() as f64;
    let baseline_stay_error = clusters.iter().map(|v| v.baseline_stay_error).sum::<f64>() / count;
    let candidate_stay_error = clusters.iter().map(|v| v.candidate_stay_error).sum::<f64>() / count;
    let baseline_flip_error = clusters.iter().map(|v| v.baseline_flip_error).sum::<f64>() / count;
    let candidate_flip_error = clusters.iter().map(|v| v.candidate_flip_error).sum::<f64>() / count;
    let baseline_joint_exact = clusters.iter().map(|v| v.baseline_joint_exact).sum::<f64>() / count;
    let candidate_joint_exact = clusters
        .iter()
        .map(|v| v.candidate_joint_exact)
        .sum::<f64>()
        / count;
    PointEstimates {
        baseline_stay_error,
        candidate_stay_error,
        baseline_flip_error,
        candidate_flip_error,
        baseline_joint_exact,
        candidate_joint_exact,
        stay_improvement: baseline_stay_error - candidate_stay_error,
        flip_improvement: baseline_flip_error - candidate_flip_error,
        joint_improvement: candidate_joint_exact - baseline_joint_exact,
    }
}

fn bootstrap_intervals(
    clusters: &[ClusterOutcomes],
    policy: &PairedInferencePolicyV1,
) -> Result<
    (
        PercentileIntervalV1,
        PercentileIntervalV1,
        PercentileIntervalV1,
    ),
    String,
> {
    let cluster_count = u64::try_from(clusters.len())
        .map_err(|_| "source-work cluster count exceeds RNG range".to_string())?;
    if cluster_count == 0 {
        return Err("cannot bootstrap an empty cluster set".into());
    }
    let mut rng = SplitMix64::new(policy.seed);
    let mut stay = Vec::with_capacity(policy.bootstrap_replicates);
    let mut flip = Vec::with_capacity(policy.bootstrap_replicates);
    let mut joint = Vec::with_capacity(policy.bootstrap_replicates);

    for _ in 0..policy.bootstrap_replicates {
        let mut sums = [0.0_f64; 6];
        for _ in 0..clusters.len() {
            let index = rng.bounded(cluster_count)? as usize;
            let cluster = clusters[index];
            sums[0] += cluster.baseline_stay_error;
            sums[1] += cluster.candidate_stay_error;
            sums[2] += cluster.baseline_flip_error;
            sums[3] += cluster.candidate_flip_error;
            sums[4] += cluster.baseline_joint_exact;
            sums[5] += cluster.candidate_joint_exact;
        }
        let denominator = clusters.len() as f64;
        stay.push((sums[0] - sums[1]) / denominator);
        flip.push((sums[2] - sums[3]) / denominator);
        joint.push((sums[5] - sums[4]) / denominator);
    }

    let tail_probability = (1.0 - policy.confidence_level) / 2.0;
    Ok((
        percentile_interval(&mut stay, policy.confidence_level, tail_probability),
        percentile_interval(&mut flip, policy.confidence_level, tail_probability),
        percentile_interval(&mut joint, policy.confidence_level, tail_probability),
    ))
}

fn percentile_interval(
    samples: &mut [f64],
    confidence_level: f64,
    tail_probability: f64,
) -> PercentileIntervalV1 {
    samples.sort_by(f64::total_cmp);
    PercentileIntervalV1 {
        lower: quantile(samples, tail_probability),
        upper: quantile(samples, 1.0 - tail_probability),
        confidence_level,
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() <= 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

/// SplitMix64 is used only as a small, fully specified deterministic bootstrap
/// sampler. It is not a cryptographic PRNG and produces no security claims.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn bounded(&mut self, upper_exclusive: u64) -> Result<u64, String> {
        if upper_exclusive == 0 {
            return Err("bootstrap RNG upper bound must be positive".into());
        }
        let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return Ok(value % upper_exclusive);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PairedMusicalIdentityComparisonCaseV1 {
        serde_json::from_str(include_str!(
            "../fixtures/mel-music-performance-identity-comparison.synthetic.json"
        ))
        .expect("checked-in paired comparison fixture must parse")
    }

    #[test]
    fn paired_cluster_bootstrap_reports_three_noncollapsed_metrics() {
        let report = audit_paired_musical_identity_comparison(&fixture()).unwrap();
        assert_eq!(
            report.status,
            PairedInferenceStatusV1::SourceWorkClusterFloorMet
        );
        assert_eq!(report.item_count, 30);
        assert_eq!(report.source_work_clusters, 30);
        assert_eq!(report.by_target_property.len(), 1);
        assert_eq!(report.by_performance_stay_factor.len(), 1);
        assert_eq!(report.by_musical_flip_factor.len(), 1);
        assert_eq!(report.by_target_property[0].source_work_clusters, 30);
        assert_eq!(
            report.by_performance_stay_factor[0].source_work_clusters,
            30
        );
        assert_eq!(report.by_musical_flip_factor[0].source_work_clusters, 30);
        assert_eq!(
            report.by_target_property[0].status,
            PairedInferenceStatusV1::SourceWorkClusterFloorMet
        );
        assert!(
            report
                .performance_stay_false_difference
                .estimate
                .candidate_improvement
                > 0.0
        );
        assert!(report.musical_flip_miss.estimate.candidate_improvement > 0.0);
        assert!(report.joint_exact.estimate.candidate_improvement > 0.0);
        assert!(
            report
                .performance_stay_false_difference
                .candidate_improvement_interval
                .is_some()
        );
        assert!(
            report
                .musical_flip_miss
                .candidate_improvement_interval
                .is_some()
        );
        assert!(report.joint_exact.candidate_improvement_interval.is_some());
        assert!(!report.source_work_lineage_authenticated);
        assert!(!report.target_property_annotations_authenticated);
        assert!(!report.population_generalization_established);
    }

    #[test]
    fn identical_model_answers_produce_zero_paired_effect_and_zero_width_intervals() {
        let mut input = fixture();
        input.candidate = input.baseline.clone();
        let first = audit_paired_musical_identity_comparison(&input).unwrap();
        let second = audit_paired_musical_identity_comparison(&input).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first
                .performance_stay_false_difference
                .estimate
                .candidate_improvement,
            0.0
        );
        assert_eq!(first.musical_flip_miss.estimate.candidate_improvement, 0.0);
        assert_eq!(first.joint_exact.estimate.candidate_improvement, 0.0);
        assert_eq!(
            first
                .performance_stay_false_difference
                .candidate_improvement_interval
                .as_ref()
                .unwrap()
                .lower,
            0.0
        );
        assert_eq!(
            first
                .performance_stay_false_difference
                .candidate_improvement_interval
                .as_ref()
                .unwrap()
                .upper,
            0.0
        );
    }

    #[test]
    fn item_order_does_not_change_equal_source_work_estimate_or_replay() {
        let mut input = fixture();
        let original = audit_paired_musical_identity_comparison(&input).unwrap();
        input.baseline.triads.reverse();
        input.candidate.triads.reverse();
        let reversed = audit_paired_musical_identity_comparison(&input).unwrap();
        assert_eq!(original, reversed);
    }

    #[test]
    fn audio_or_lineage_substitution_is_rejected_before_bootstrap() {
        let mut input = fixture();
        input.candidate.triads[0].clean.response.audio_sha256 = "f".repeat(64);
        assert!(audit_paired_musical_identity_comparison(&input).is_err());

        let mut input = fixture();
        input.candidate.triads[0].source_work_lineage_sha256 = "e".repeat(64);
        assert!(audit_paired_musical_identity_comparison(&input).is_err());
    }

    #[test]
    fn adding_an_item_to_one_work_does_not_reweight_the_equal_work_estimand() {
        let original_case = fixture();
        let original = audit_paired_musical_identity_comparison(&original_case).unwrap();

        let mut expanded = original_case;
        let mut baseline_extra = expanded.baseline.triads[0].clone();
        let mut candidate_extra = expanded.candidate.triads[0].clone();
        baseline_extra.item_id = "additional-item-same-source-work".into();
        candidate_extra.item_id = baseline_extra.item_id.clone();
        baseline_extra.pair_id = "additional-pair-same-source-work".into();
        candidate_extra.pair_id = baseline_extra.pair_id.clone();
        baseline_extra.target_property_reference_sha256 = "1".repeat(64);
        candidate_extra.target_property_reference_sha256 =
            baseline_extra.target_property_reference_sha256.clone();
        baseline_extra.pairing_manifest_sha256 = "2".repeat(64);
        candidate_extra.pairing_manifest_sha256 = baseline_extra.pairing_manifest_sha256.clone();
        for (index, (left, right)) in [
            (&mut baseline_extra.clean, &mut candidate_extra.clean),
            (
                &mut baseline_extra.performance_stay,
                &mut candidate_extra.performance_stay,
            ),
            (
                &mut baseline_extra.musical_flip,
                &mut candidate_extra.musical_flip,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let audio = format!("{:064x}", 0x9000_u64 + index as u64);
            left.response.audio_sha256 = audio.clone();
            right.response.audio_sha256 = audio;
            left.response.variant_id.push_str("-additional");
            right.response.variant_id = left.response.variant_id.clone();
        }
        for extra in [&mut baseline_extra, &mut candidate_extra] {
            let take = "a".repeat(64);
            let performance = "b".repeat(64);
            extra.clean.take_identity_sha256 = take.clone();
            extra.musical_flip.take_identity_sha256 = take;
            extra.clean.performance_identity_sha256 = performance.clone();
            extra.musical_flip.performance_identity_sha256 = performance;
            extra.performance_stay.take_identity_sha256 = "c".repeat(64);
            extra.performance_stay.performance_identity_sha256 = "d".repeat(64);
        }
        expanded.baseline.triads.push(baseline_extra);
        expanded.candidate.triads.push(candidate_extra);

        let report = audit_paired_musical_identity_comparison(&expanded).unwrap();
        assert_eq!(report.item_count, 31);
        assert_eq!(report.source_work_clusters, 30);
        assert_eq!(
            report.performance_stay_false_difference.estimate,
            original.performance_stay_false_difference.estimate
        );
        assert_eq!(
            report.musical_flip_miss.estimate,
            original.musical_flip_miss.estimate
        );
        assert_eq!(report.joint_exact.estimate, original.joint_exact.estimate);
    }

    #[test]
    fn sparse_property_stratum_is_not_estimable_even_when_overall_floor_is_met() {
        let mut input = fixture();
        let property_question = "Is the root pitch class the same?".to_string();
        let property_question_sha256 =
            crate::music_listen_bench::sha256_hex(property_question.as_bytes());
        for case in [&mut input.baseline, &mut input.candidate] {
            let triad = case.triads.last_mut().unwrap();
            triad.target_property =
                crate::music_performance_identity::MusicalIdentityPropertyV1::RootPitchClass;
            triad.musical_flip_factor =
                crate::music_performance_identity::MusicalFlipFactorV1::RootPitchClassChange;
            triad.target_property_reference_sha256 = "1".repeat(64);
            triad.pairing_manifest_sha256 = "2".repeat(64);
            for variant in [
                &mut triad.clean,
                &mut triad.performance_stay,
                &mut triad.musical_flip,
            ] {
                variant.response.question = property_question.clone();
                variant.response.question_sha256 = property_question_sha256.clone();
            }
        }

        let report = audit_paired_musical_identity_comparison(&input).unwrap();
        assert_eq!(report.source_work_clusters, 30);
        let root = report
            .by_target_property
            .iter()
            .find(|stratum| {
                stratum.target_property
                    == crate::music_performance_identity::MusicalIdentityPropertyV1::RootPitchClass
            })
            .unwrap();
        assert_eq!(root.source_work_clusters, 1);
        assert_eq!(
            root.status,
            PairedInferenceStatusV1::InsufficientSourceWorkClusters
        );
        assert!(
            root.performance_stay_false_difference
                .candidate_improvement_interval
                .is_none()
        );
        assert!(
            root.musical_flip_miss
                .candidate_improvement_interval
                .is_none()
        );
        assert!(root.joint_exact.candidate_improvement_interval.is_none());
    }

    #[test]
    fn sparse_stay_and_flip_factor_strata_withhold_intervals_independently() {
        let mut input = fixture();
        let root_question = "Is the root pitch class the same?".to_string();
        let root_question_sha256 = crate::music_listen_bench::sha256_hex(root_question.as_bytes());

        for case in [&mut input.baseline, &mut input.candidate] {
            let triad = case.triads.last_mut().unwrap();
            triad.target_property =
                crate::music_performance_identity::MusicalIdentityPropertyV1::RootPitchClass;
            triad.performance_stay_factor =
                crate::music_performance_identity::PerformanceStayFactorV1::ExpressiveTimingChange;
            triad.musical_flip_factor =
                crate::music_performance_identity::MusicalFlipFactorV1::RootPitchClassChange;
            triad.target_property_reference_sha256 = "3".repeat(64);
            triad.pairing_manifest_sha256 = "4".repeat(64);
            for variant in [
                &mut triad.clean,
                &mut triad.performance_stay,
                &mut triad.musical_flip,
            ] {
                variant.response.question = root_question.clone();
                variant.response.question_sha256 = root_question_sha256.clone();
            }
        }

        let report = audit_paired_musical_identity_comparison(&input).unwrap();
        assert_eq!(report.source_work_clusters, 30);

        let stay_stratum = report
            .by_performance_stay_factor
            .iter()
            .find(|stratum| {
                stratum.target_property
                    == crate::music_performance_identity::MusicalIdentityPropertyV1::RootPitchClass
            })
            .unwrap();
        assert_eq!(stay_stratum.source_work_clusters, 1);
        assert_eq!(
            stay_stratum.status,
            PairedInferenceStatusV1::InsufficientSourceWorkClusters
        );
        assert!(
            stay_stratum
                .performance_stay_false_difference
                .candidate_improvement_interval
                .is_none()
        );

        let flip_stratum = report
            .by_musical_flip_factor
            .iter()
            .find(|stratum| {
                stratum.target_property
                    == crate::music_performance_identity::MusicalIdentityPropertyV1::RootPitchClass
            })
            .unwrap();
        assert_eq!(flip_stratum.source_work_clusters, 1);
        assert_eq!(
            flip_stratum.status,
            PairedInferenceStatusV1::InsufficientSourceWorkClusters
        );
        assert!(
            flip_stratum
                .musical_flip_miss
                .candidate_improvement_interval
                .is_none()
        );
    }

    #[test]
    fn stay_improvement_does_not_hide_a_flip_regression() {
        let mut input = fixture();
        for triad in &mut input.candidate.triads {
            triad.performance_stay.response.model_answer =
                crate::music_performance_identity::MusicListenAnswerV1::ChoiceA;
            triad.musical_flip.response.model_answer =
                crate::music_performance_identity::MusicListenAnswerV1::ChoiceA;
        }
        let report = audit_paired_musical_identity_comparison(&input).unwrap();
        assert!(
            report
                .performance_stay_false_difference
                .estimate
                .candidate_improvement
                > 0.0
        );
        assert!(report.musical_flip_miss.estimate.candidate_improvement < 0.0);
        assert_ne!(
            report
                .performance_stay_false_difference
                .estimate
                .candidate_improvement,
            report.musical_flip_miss.estimate.candidate_improvement
        );
    }

    #[test]
    fn fewer_than_thirty_source_works_suppresses_confirmatory_intervals() {
        let mut input = fixture();
        input.baseline.triads.pop();
        input.candidate.triads.pop();
        let report = audit_paired_musical_identity_comparison(&input).unwrap();
        assert_eq!(report.source_work_clusters, 29);
        assert_eq!(
            report.status,
            PairedInferenceStatusV1::InsufficientSourceWorkClusters
        );
        assert!(
            report
                .performance_stay_false_difference
                .candidate_improvement_interval
                .is_none()
        );
        assert!(
            report
                .musical_flip_miss
                .candidate_improvement_interval
                .is_none()
        );
        assert!(report.joint_exact.candidate_improvement_interval.is_none());
    }
}
