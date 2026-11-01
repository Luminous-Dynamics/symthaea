//! Benjamini-Yekutieli multiple-testing adjustment over calibrated detector evidence.
//!
//! This module consumes a detector-calibration report, not raw detector scores.
//! It deduplicates repeated item rows to one hypothesis per source cluster and
//! detector. BY controls FDR under arbitrary dependence only when every true-null
//! input p-value is super-uniform. It cannot repair failed conformal exchangeability,
//! contaminated clean calibration data, or invalid detector features.

use crate::{
    detector_calibration::{DetectorCalibrationReportV1, GroundTruthV1},
    hash_domain,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CONFORMAL_BY_CONFIG_SCHEMA: &str = "symthaea.conformal_by_config.v1";
pub const CONFORMAL_BY_REPORT_SCHEMA: &str = "symthaea.conformal_by_report.v1";
pub const CONFORMAL_BY_HARMONIC_SCALE: u64 = 1_000_000;
pub const MAX_BY_FAMILIES: usize = 100_000;
pub const MAX_BY_HYPOTHESES: usize = 2_000_000;
pub const MAX_BY_Q_LEVELS: usize = 16;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ByFamilyScopeV1 {
    PerDetectorPerGroup,
    AllDetectorsPerGroup,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalByConfigV1 {
    pub schema: String,
    pub adjustment_id: String,
    pub source_calibration_study_sha256: String,
    pub source_calibration_report_sha256: String,
    pub family_scopes: Vec<ByFamilyScopeV1>,
    /// Target false-discovery rates in permille, e.g. 10 = q=1%.
    pub target_fdr_permille: Vec<u16>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ByAdjustedPValueV1 {
    /// Numerator is the conservative adjusted-p ceiling in permille.
    pub numerator: u64,
    pub denominator: u64,
    pub ceiling_permille: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ByHypothesisResultV1 {
    pub calibration_group_sha256: String,
    pub family_id: String,
    pub family_scope: ByFamilyScopeV1,
    pub detector_id: String,
    pub detector_version: String,
    pub source_cluster_id: String,
    pub item_count: u64,
    pub item_sample_ids: Vec<String>,
    pub ground_truth: GroundTruthV1,
    pub conformal_p_value_numerator: u64,
    pub conformal_p_value_denominator: u64,
    pub by_rank: u64,
    pub by_adjusted_p_value: ByAdjustedPValueV1,
    /// q-level decisions are aligned to the configuration's sorted q list.
    pub rejected_at_q_permille: Vec<u16>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ByQLevelSummaryV1 {
    pub target_fdr_permille: u16,
    pub rejection_cutoff_rank: u64,
    pub discovery_count: u64,
    pub clean_discovery_count: u64,
    pub contaminated_discovery_count: u64,
    /// Empirical FDP against the supplied locked-test labels; descriptive only.
    pub empirical_fdp: Option<f64>,
    pub empirical_power: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ByFamilyReportV1 {
    pub family_id: String,
    pub calibration_group_sha256: String,
    pub family_scope: ByFamilyScopeV1,
    pub detector_count: u64,
    pub hypothesis_count: u64,
    /// ceil(1e6 * H_m), a conservative integer upper bound.
    pub harmonic_upper_micros: u64,
    pub q_level_summaries: Vec<ByQLevelSummaryV1>,
    pub hypotheses: Vec<ByHypothesisResultV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConformalByReportV1 {
    pub schema: String,
    pub adjustment_id: String,
    pub source_calibration_study_sha256: String,
    pub source_calibration_report_sha256: String,
    pub config_sha256: String,
    pub family_scopes: Vec<ByFamilyScopeV1>,
    pub target_fdr_permille: Vec<u16>,
    pub family_count: u64,
    pub hypothesis_count: u64,
    pub families: Vec<ByFamilyReportV1>,
    pub fdr_adjustment_applied: bool,
    pub arbitrary_dependence_adjustment: bool,
    pub input_p_values_assumed_valid: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct HypothesisKey {
    detector_id: String,
    detector_version: String,
    source_cluster_id: String,
}

#[derive(Debug, Clone)]
struct CanonicalHypothesis {
    key: HypothesisKey,
    item_sample_ids: Vec<String>,
    ground_truth: GroundTruthV1,
    cluster_max_normalized_risk_score: f64,
    numerator: u64,
    denominator: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct FamilyKey {
    group_sha256: String,
    scope: ByFamilyScopeV1,
    detector_id: String,
    detector_version: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_identifier(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:/-".contains(&byte))
    {
        Err(format!(
            "{label} must be a nonempty opaque ASCII identifier"
        ))
    } else {
        Ok(())
    }
}

fn canonical_json<T: Serialize>(value: &T, label: &str) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|error| format!("cannot serialize {label}: {error}"))
}

/// Domain-separated commitment to the exact calibration report passed to BY.
pub fn compute_calibration_report_sha256(
    report: &DetectorCalibrationReportV1,
) -> Result<String, String> {
    let bytes = canonical_json(report, "detector calibration report")?;
    Ok(hash_domain(b"detector-calibration-report.v1", &bytes))
}

/// Conservative integer upper bound for the harmonic number H_m.
/// Every term ceil(1e6/i) is >= 1e6/i, so the sum is an upper bound.
pub fn harmonic_upper_micros(hypothesis_count: u64) -> Result<u64, String> {
    if hypothesis_count == 0 || hypothesis_count > MAX_BY_HYPOTHESES as u64 {
        return Err(format!("BY family size must be in 1..={MAX_BY_HYPOTHESES}"));
    }
    let mut sum = 0_u64;
    for index in 1..=hypothesis_count {
        let term = CONFORMAL_BY_HARMONIC_SCALE.div_ceil(index);
        sum = sum
            .checked_add(term)
            .ok_or_else(|| "BY harmonic upper bound overflows u64".to_owned())?;
    }
    Ok(sum)
}

fn compare_rational(
    left_numerator: u64,
    left_denominator: u64,
    right_numerator: u64,
    right_denominator: u64,
) -> std::cmp::Ordering {
    (u128::from(left_numerator) * u128::from(right_denominator))
        .cmp(&(u128::from(right_numerator) * u128::from(left_denominator)))
}

/// Returns the largest BY step-up rank k satisfying
/// p_(k) <= q*k/(m*H_m), using the conservative integer H_m upper bound.
pub fn by_rejection_cutoff(
    sorted_p_values: &[(u64, u64)],
    target_fdr_permille: u16,
    harmonic_micros: u64,
) -> Result<u64, String> {
    if sorted_p_values.is_empty() {
        return Err("BY family cannot be empty".to_owned());
    }
    if !(1..=1_000).contains(&target_fdr_permille) {
        return Err("BY target FDR must be in [1, 1000] permille".to_owned());
    }
    let family_size = sorted_p_values.len() as u64;
    if sorted_p_values.windows(2).any(|pair| {
        compare_rational(pair[0].0, pair[0].1, pair[1].0, pair[1].1) == std::cmp::Ordering::Greater
    }) {
        return Err("BY p-values must be sorted in nondecreasing exact-rational order".to_owned());
    }
    if harmonic_micros == 0 {
        return Err("BY harmonic upper bound must be positive".to_owned());
    }
    let mut largest_rank = 0_u64;
    for (index, (numerator, denominator)) in sorted_p_values.iter().enumerate() {
        if *denominator == 0 || *numerator == 0 || numerator > denominator {
            return Err("BY input p-values must be rational values in (0, 1]".to_owned());
        }
        let rank = index as u64 + 1;
        let left = u128::from(*numerator) * u128::from(family_size) * u128::from(harmonic_micros);
        let right = u128::from(target_fdr_permille)
            * u128::from(rank)
            * u128::from(1_000_u64)
            * u128::from(*denominator);
        if left <= right {
            largest_rank = rank;
        }
    }
    Ok(largest_rank)
}

fn adjusted_p_value_upper(
    sorted_p_values: &[(u64, u64)],
    harmonic_micros: u64,
) -> Result<Vec<ByAdjustedPValueV1>, String> {
    if sorted_p_values.is_empty() {
        return Err("BY family cannot be empty".to_owned());
    }
    if sorted_p_values.windows(2).any(|pair| {
        compare_rational(pair[0].0, pair[0].1, pair[1].0, pair[1].1)
            == std::cmp::Ordering::Greater
    }) {
        return Err("BY p-values must be sorted in nondecreasing exact-rational order".to_owned());
    }
    let family_size = sorted_p_values.len() as u64;
    let mut output = vec![
        ByAdjustedPValueV1 {
            numerator: 1_000,
            denominator: 1_000,
            ceiling_permille: 1_000,
        };
        sorted_p_values.len()
    ];
    let mut suffix_min = 1_000_u64;
    for index in (0..sorted_p_values.len()).rev() {
        let (numerator, denominator) = sorted_p_values[index];
        if numerator == 0 || denominator == 0 || numerator > denominator {
            return Err("BY input p-values must be rational values in (0, 1]".to_owned());
        }
        let rank = index as u64 + 1;
        // adjusted-p scaled to permille is
        // 1000 * p * m * H_m / rank.
        // H_m <= harmonic_micros/1e6; ceiling makes this conservative.
        let scaled_numerator =
            u128::from(numerator) * u128::from(family_size) * u128::from(harmonic_micros);
        let scaled_denominator = u128::from(denominator) * u128::from(rank) * u128::from(1_000_u64);
        let adjusted_ceiling = scaled_numerator.div_ceil(scaled_denominator).min(1_000) as u64;
        suffix_min = suffix_min.min(adjusted_ceiling);
        output[index] = ByAdjustedPValueV1 {
            numerator: suffix_min,
            denominator: 1_000,
            ceiling_permille: suffix_min as u16,
        };
    }
    Ok(output)
}

fn build_families(
    source_report: &DetectorCalibrationReportV1,
    scopes: &[ByFamilyScopeV1],
) -> Result<BTreeMap<FamilyKey, Vec<CanonicalHypothesis>>, String> {
    let mut output = BTreeMap::<FamilyKey, Vec<CanonicalHypothesis>>::new();
    let mut canonical_by_group =
        BTreeMap::<String, BTreeMap<HypothesisKey, CanonicalHypothesis>>::new();

    for observation in &source_report.scored_test_p_values {
        if !valid_digest(&observation.calibration_group_sha256) {
            return Err(
                "calibration report contains an invalid calibration-group digest".to_owned(),
            );
        }
        valid_identifier(&observation.sample_id, "sample_id")?;
        valid_identifier(&observation.detector_id, "detector_id")?;
        valid_identifier(&observation.detector_version, "detector_version")?;
        valid_identifier(&observation.source_cluster_id, "source_cluster_id")?;
        if !observation.normalized_risk_score.is_finite()
            || !observation.cluster_max_normalized_risk_score.is_finite()
            || observation.normalized_risk_score
                > observation.cluster_max_normalized_risk_score + f64::EPSILON
        {
            return Err(
                "calibration report has invalid item or cluster-maximum risk scores".to_owned(),
            );
        }
        let numerator = observation.conformal_p_value.numerator;
        let denominator = observation.conformal_p_value.denominator;
        if numerator == 0 || denominator == 0 || numerator > denominator {
            return Err(
                "calibration report contains an invalid rational conformal p-value".to_owned(),
            );
        }
        if observation.calibration_clean_support + 1 != denominator {
            return Err(
                "conformal p-value denominator does not match clean calibration support + 1"
                    .to_owned(),
            );
        }
        let key = HypothesisKey {
            detector_id: observation.detector_id.clone(),
            detector_version: observation.detector_version.clone(),
            source_cluster_id: observation.source_cluster_id.clone(),
        };
        let group = canonical_by_group
            .entry(observation.calibration_group_sha256.clone())
            .or_default();
        if let Some(previous) = group.get_mut(&key) {
            if previous.numerator != numerator
                || previous.denominator != denominator
                || previous.ground_truth != observation.ground_truth
                || previous.cluster_max_normalized_risk_score
                    != observation.cluster_max_normalized_risk_score
            {
                return Err("repeated item rows in a source cluster have inconsistent p-values, cluster maxima or labels".to_owned());
            }
            if previous.item_sample_ids.contains(&observation.sample_id) {
                return Err("duplicate item sample ID within a detector/source cluster".to_owned());
            }
            previous.item_sample_ids.push(observation.sample_id.clone());
        } else {
            group.insert(
                key.clone(),
                CanonicalHypothesis {
                    key,
                    item_sample_ids: vec![observation.sample_id.clone()],
                    ground_truth: observation.ground_truth,
                    cluster_max_normalized_risk_score: observation
                        .cluster_max_normalized_risk_score,
                    numerator,
                    denominator,
                },
            );
        }
    }

    for (group_sha, hypotheses) in canonical_by_group {
        let mut detector_cluster_sets = BTreeMap::<(String, String), BTreeSet<String>>::new();
        for hypothesis in hypotheses.values() {
            detector_cluster_sets
                .entry((
                    hypothesis.key.detector_id.clone(),
                    hypothesis.key.detector_version.clone(),
                ))
                .or_default()
                .insert(hypothesis.key.source_cluster_id.clone());
        }
        let expected_clusters = detector_cluster_sets
            .values()
            .next()
            .cloned()
            .ok_or_else(|| "calibration report has an empty detector group".to_owned())?;
        if detector_cluster_sets
            .values()
            .any(|clusters| clusters != &expected_clusters)
        {
            return Err("detectors in one calibration group must have the same cluster hypothesis set for joint BY adjustment".to_owned());
        }
        let mut cluster_truth = BTreeMap::<String, GroundTruthV1>::new();
        let mut cluster_items = BTreeMap::<String, BTreeSet<String>>::new();
        for hypothesis in hypotheses.values() {
            if let Some(previous) = cluster_truth.insert(
                hypothesis.key.source_cluster_id.clone(),
                hypothesis.ground_truth,
            ) && previous != hypothesis.ground_truth
            {
                return Err(
                    "ground-truth label differs across detectors for one source cluster".to_owned(),
                );
            }
            let items = cluster_items
                .entry(hypothesis.key.source_cluster_id.clone())
                .or_default();
            let current: BTreeSet<String> = hypothesis.item_sample_ids.iter().cloned().collect();
            if items.is_empty() {
                *items = current;
            } else if *items != current {
                return Err("detectors in one calibration group must have the same item-ID set per source cluster".to_owned());
            }
        }

        for scope in scopes {
            match scope {
                ByFamilyScopeV1::PerDetectorPerGroup => {
                    for ((detector_id, detector_version), _) in &detector_cluster_sets {
                        let family_key = FamilyKey {
                            group_sha256: group_sha.clone(),
                            scope: *scope,
                            detector_id: detector_id.clone(),
                            detector_version: detector_version.clone(),
                        };
                        let family_items: Vec<CanonicalHypothesis> = hypotheses
                            .values()
                            .filter(|hypothesis| {
                                hypothesis.key.detector_id == *detector_id
                                    && hypothesis.key.detector_version == *detector_version
                            })
                            .cloned()
                            .collect();
                        output.insert(family_key, family_items);
                    }
                }
                ByFamilyScopeV1::AllDetectorsPerGroup => {
                    let family_key = FamilyKey {
                        group_sha256: group_sha.clone(),
                        scope: *scope,
                        detector_id: "all_detectors".to_owned(),
                        detector_version: "joint_family_v1".to_owned(),
                    };
                    output.insert(family_key, hypotheses.values().cloned().collect());
                }
            }
        }
    }
    Ok(output)
}

fn family_identifier(key: &FamilyKey) -> Result<String, String> {
    let bytes = canonical_json(key, "BY family identity")?;
    Ok(hash_domain(b"conformal-by-family.v1", &bytes))
}

fn analyze_family(
    key: &FamilyKey,
    hypotheses: &[CanonicalHypothesis],
    q_levels: &[u16],
) -> Result<ByFamilyReportV1, String> {
    if hypotheses.is_empty() || hypotheses.len() > MAX_BY_HYPOTHESES {
        return Err("BY family has an empty or oversized hypothesis set".to_owned());
    }
    let family_id = family_identifier(key)?;
    let mut sorted = hypotheses.to_vec();
    sorted.sort_by(|left, right| {
        compare_rational(
            left.numerator,
            left.denominator,
            right.numerator,
            right.denominator,
        )
        .then_with(|| left.key.detector_id.cmp(&right.key.detector_id))
        .then_with(|| left.key.detector_version.cmp(&right.key.detector_version))
        .then_with(|| left.key.source_cluster_id.cmp(&right.key.source_cluster_id))
    });
    let raw_p_values: Vec<(u64, u64)> = sorted
        .iter()
        .map(|hypothesis| (hypothesis.numerator, hypothesis.denominator))
        .collect();
    let harmonic_micros = harmonic_upper_micros(sorted.len() as u64)?;
    let adjusted = adjusted_p_value_upper(&raw_p_values, harmonic_micros)?;
    let mut q_level_summaries = Vec::with_capacity(q_levels.len());
    let mut cutoffs = Vec::with_capacity(q_levels.len());
    for q in q_levels {
        let cutoff = by_rejection_cutoff(&raw_p_values, *q, harmonic_micros)?;
        cutoffs.push(cutoff);
        let mut clean = 0_u64;
        let mut contaminated = 0_u64;
        for hypothesis in sorted.iter().take(cutoff as usize) {
            match hypothesis.ground_truth {
                GroundTruthV1::Clean => clean += 1,
                GroundTruthV1::Contaminated => contaminated += 1,
            }
        }
        let discoveries = cutoff;
        q_level_summaries.push(ByQLevelSummaryV1 {
            target_fdr_permille: *q,
            rejection_cutoff_rank: cutoff,
            discovery_count: discoveries,
            clean_discovery_count: clean,
            contaminated_discovery_count: contaminated,
            empirical_fdp: if discoveries == 0 {
                Some(0.0)
            } else {
                Some(clean as f64 / discoveries as f64)
            },
            empirical_power: {
                let total_contaminated = sorted
                    .iter()
                    .filter(|hypothesis| hypothesis.ground_truth == GroundTruthV1::Contaminated)
                    .count() as u64;
                if total_contaminated == 0 {
                    None
                } else {
                    Some(contaminated as f64 / total_contaminated as f64)
                }
            },
        });
    }

    let mut result_hypotheses = Vec::with_capacity(sorted.len());
    for (index, hypothesis) in sorted.into_iter().enumerate() {
        let rank = index as u64 + 1;
        let rejected_at_q = q_levels
            .iter()
            .zip(&cutoffs)
            .filter_map(|(q, cutoff)| (rank <= *cutoff).then_some(*q))
            .collect();
        let mut item_sample_ids = hypothesis.item_sample_ids;
        item_sample_ids.sort();
        result_hypotheses.push(ByHypothesisResultV1 {
            calibration_group_sha256: key.group_sha256.clone(),
            family_id: family_id.clone(),
            family_scope: key.scope,
            detector_id: hypothesis.key.detector_id,
            detector_version: hypothesis.key.detector_version,
            source_cluster_id: hypothesis.key.source_cluster_id,
            item_count: item_sample_ids.len() as u64,
            item_sample_ids,
            ground_truth: hypothesis.ground_truth,
            conformal_p_value_numerator: hypothesis.numerator,
            conformal_p_value_denominator: hypothesis.denominator,
            by_rank: rank,
            by_adjusted_p_value: adjusted[index].clone(),
            rejected_at_q_permille: rejected_at_q,
        });
    }

    let detector_count = if key.detector_id == "all_detectors" {
        result_hypotheses
            .iter()
            .map(|hypothesis| {
                (
                    hypothesis.detector_id.clone(),
                    hypothesis.detector_version.clone(),
                )
            })
            .collect::<BTreeSet<_>>()
            .len() as u64
    } else {
        1
    };
    Ok(ByFamilyReportV1 {
        family_id,
        calibration_group_sha256: key.group_sha256.clone(),
        family_scope: key.scope,
        detector_count,
        hypothesis_count: result_hypotheses.len() as u64,
        harmonic_upper_micros: harmonic_micros,
        q_level_summaries,
        hypotheses: result_hypotheses,
    })
}

/// Apply predeclared BY families to a previously calibrated detector report.
pub fn analyze_conformal_by(
    source_report: &DetectorCalibrationReportV1,
    config: &ConformalByConfigV1,
) -> Result<ConformalByReportV1, String> {
    if config.schema != CONFORMAL_BY_CONFIG_SCHEMA {
        return Err(format!("config schema must be {CONFORMAL_BY_CONFIG_SCHEMA}"));
    }
    if source_report.schema != crate::detector_calibration::CALIBRATION_REPORT_SCHEMA {
        return Err(
            "BY input does not use the supported detector calibration report schema".to_owned(),
        );
    }
    valid_identifier(&config.adjustment_id, "adjustment_id")?;
    if !valid_digest(&config.source_calibration_study_sha256)
        || !valid_digest(&source_report.study_sha256)
        || config.source_calibration_study_sha256 != source_report.study_sha256
    {
        return Err("BY config source calibration study digest does not match report".to_owned());
    }
    if config.family_scopes.is_empty() || config.family_scopes.len() > 2 {
        return Err("BY family scopes must contain one or both supported scopes".to_owned());
    }
    let unique_scopes: BTreeSet<ByFamilyScopeV1> = config.family_scopes.iter().copied().collect();
    if unique_scopes.len() != config.family_scopes.len() {
        return Err("BY family scopes must be unique".to_owned());
    }
    if config.target_fdr_permille.is_empty() || config.target_fdr_permille.len() > MAX_BY_Q_LEVELS {
        return Err("BY target FDR levels must be nonempty and bounded".to_owned());
    }
    let mut q_levels = config.target_fdr_permille.clone();
    q_levels.sort_unstable();
    q_levels.dedup();
    if q_levels != config.target_fdr_permille || q_levels.iter().any(|q| !(1..=1_000).contains(q)) {
        return Err(
            "BY target FDR levels must be unique, ascending and in [1, 1000] permille".to_owned(),
        );
    }
    if source_report.scored_test_p_values.is_empty()
        || source_report.scored_test_p_values.len() > MAX_BY_HYPOTHESES
        || source_report.scored_locked_test_observation_count
            != source_report.scored_test_p_values.len() as u64
    {
        return Err("source calibration report has no test p-values, mismatched score count or exceeds BY limits".to_owned());
    }
    let source_report_sha = compute_calibration_report_sha256(source_report)?;
    if !valid_digest(&config.source_calibration_report_sha256)
        || config.source_calibration_report_sha256 != source_report_sha
    {
        return Err(
            "BY config source calibration report digest does not match exact report content"
                .to_owned(),
        );
    }
    let config_bytes = canonical_json(config, "BY config")?;
    let config_sha = hash_domain(b"conformal-by-config.v1", &config_bytes);
    let family_map = build_families(source_report, &config.family_scopes)?;
    if family_map.is_empty() || family_map.len() > MAX_BY_FAMILIES {
        return Err("BY family count is empty or exceeds supported bound".to_owned());
    }
    let mut hypothesis_count = 0_u64;
    let mut families = Vec::with_capacity(family_map.len());
    for (key, hypotheses) in &family_map {
        hypothesis_count = hypothesis_count
            .checked_add(hypotheses.len() as u64)
            .ok_or_else(|| "BY hypothesis counter overflows u64".to_owned())?;
        if hypothesis_count > (MAX_BY_HYPOTHESES as u64) * 2 {
            return Err("BY total family hypothesis outputs exceed supported bound".to_owned());
        }
        families.push(analyze_family(key, hypotheses, &q_levels)?);
    }
    let issues = vec![
        "BY arbitrary-dependence FDR control requires every true-null input conformal p-value to be super-uniform; this adjustment does not validate those p-values.".to_owned(),
        "Input p-value validity remains conditional on the source calibration report's clean-cluster exchangeability and cluster construction assumptions.".to_owned(),
        "The per-detector and all-detectors families represent different multiplicity claims; the primary family must be preregistered before test labels are inspected.".to_owned(),
        "Empirical FDP and power use supplied locked-test labels and are descriptive benchmark diagnostics, not proof of population FDR.".to_owned(),
        "This report does not establish model/data custody, causal exposure, label truth, distribution-shift robustness or universal detection.".to_owned(),
    ];
    Ok(ConformalByReportV1 {
        schema: CONFORMAL_BY_REPORT_SCHEMA.to_owned(),
        adjustment_id: config.adjustment_id.clone(),
        source_calibration_study_sha256: config.source_calibration_study_sha256.clone(),
        source_calibration_report_sha256: source_report_sha,
        config_sha256: config_sha,
        family_scopes: config.family_scopes.clone(),
        target_fdr_permille: q_levels,
        family_count: families.len() as u64,
        hypothesis_count,
        families,
        fdr_adjustment_applied: true,
        arbitrary_dependence_adjustment: true,
        input_p_values_assumed_valid: true,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: "exploratory_by_adjustment".to_owned(),
        issues,
        claim_boundary: "This report applies Benjamini-Yekutieli step-up adjustment to cluster-level conformal p-values in the predeclared detector families. BY controls FDR under arbitrary dependence only when every true-null input p-value is super-uniform. Here that premise remains conditional on the source calibration report's exact-group clean source-cluster exchangeability. BY cannot repair invalid detector scores, contaminated calibration cohorts, or distribution shift. Per-detector and joint all-detector families support different multiplicity claims; only the preregistered family is confirmatory. The report does not authenticate data/model custody, prove training membership or causal exposure, or establish SOTA superiority.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector_calibration::{CalibratedTestScoreV1, DetectorCalibrationReportV1};

    fn q(numerator: u64, denominator: u64) -> (u64, u64) {
        (numerator, denominator)
    }

    fn source_report() -> DetectorCalibrationReportV1 {
        let group = "a".repeat(64);
        let mut rows = Vec::new();
        for (detector, version, values) in [
            (
                "det-a",
                "v1",
                vec![
                    (1, 101, GroundTruthV1::Contaminated),
                    (10, 101, GroundTruthV1::Clean),
                    (5, 101, GroundTruthV1::Contaminated),
                ],
            ),
            (
                "det-b",
                "v2",
                vec![
                    (2, 101, GroundTruthV1::Contaminated),
                    (20, 101, GroundTruthV1::Clean),
                    (6, 101, GroundTruthV1::Contaminated),
                ],
            ),
        ] {
            for (cluster, (numerator, denominator, truth)) in values.into_iter().enumerate() {
                for item in 0..2 {
                    rows.push(CalibratedTestScoreV1 {
                        calibration_group_sha256: group.clone(),
                        sample_id: format!("cluster-{cluster}-item-{item}"),
                        source_cluster_id: format!("cluster-{cluster}"),
                        detector_id: detector.to_owned(),
                        detector_version: version.to_owned(),
                        ground_truth: truth,
                        raw_score: numerator as f64,
                        normalized_risk_score: numerator as f64,
                        cluster_max_normalized_risk_score: numerator as f64,
                        calibration_clean_support: 100,
                        conformal_p_value: crate::detector_calibration::ConformalPValueV1 {
                            numerator,
                            denominator,
                            ceiling_permille: (numerator * 1_000).div_ceil(denominator) as u16,
                        },
                    });
                }
            }
        }
        DetectorCalibrationReportV1 {
            schema: crate::detector_calibration::CALIBRATION_REPORT_SCHEMA.to_owned(),
            study_id: "study-001".to_owned(),
            benchmark_id: "bench-001".to_owned(),
            benchmark_version: "v1".to_owned(),
            study_sha256: "b".repeat(64),
            calibration_group_count: 1,
            detector_count: 2,
            observation_count: rows.len() as u64,
            scored_locked_test_observation_count: rows.len() as u64,
            group_cells: Vec::new(),
            scored_test_p_values: rows,
            disagreement_matrix: Vec::new(),
            detector_costs: Vec::new(),
            macro_cells: Vec::new(),
            intervals_multiplicity_adjusted: false,
            independently_authenticated: false,
            confirmatory_eligible: false,
            analysis_status: "exploratory_calibration_audit".to_owned(),
            issues: vec!["test fixture".to_owned()],
            claim_boundary: "test fixture".to_owned(),
        }
    }

    fn config(scopes: Vec<ByFamilyScopeV1>) -> ConformalByConfigV1 {
        ConformalByConfigV1 {
            schema: CONFORMAL_BY_CONFIG_SCHEMA.to_owned(),
            adjustment_id: "by-test-001".to_owned(),
            source_calibration_study_sha256: "b".repeat(64),
            source_calibration_report_sha256: compute_calibration_report_sha256(&source_report())
                .unwrap(),
            family_scopes: scopes,
            target_fdr_permille: vec![10, 50, 100],
        }
    }

    #[test]
    fn harmonic_upper_bound_dominates_known_small_harmonics() {
        assert!(harmonic_upper_micros(1).unwrap() >= 1_000_000);
        assert!(harmonic_upper_micros(2).unwrap() >= 1_500_000);
        assert!(harmonic_upper_micros(3).unwrap() >= 1_833_333);
    }

    #[test]
    fn by_step_up_uses_arbitrary_dependence_harmonic_correction() {
        let pvalues = vec![q(1, 100), q(2, 100), q(3, 100), q(70, 100)];
        let harmonic = harmonic_upper_micros(4).unwrap();
        let cutoff = by_rejection_cutoff(&pvalues, 100, harmonic).unwrap();
        assert!(cutoff <= 4);
    }

    #[test]
    fn adjusted_p_values_are_monotone_and_conservative() {
        let values = vec![q(1, 100), q(2, 100), q(20, 100)];
        let harmonic = harmonic_upper_micros(values.len() as u64).unwrap();
        let adjusted = adjusted_p_value_upper(&values, harmonic).unwrap();
        assert_eq!(adjusted.len(), 3);
        assert!(adjusted[0].ceiling_permille <= adjusted[1].ceiling_permille);
        assert!(adjusted[1].ceiling_permille <= adjusted[2].ceiling_permille);
        assert!(adjusted.iter().all(|value| value.ceiling_permille >= 1));
        assert!(adjusted.iter().all(|value| value.denominator == 1_000));
    }

    #[test]
    fn by_exact_rational_comparison_does_not_use_float_boundaries() {
        let harmonic = harmonic_upper_micros(1).unwrap();
        assert_eq!(by_rejection_cutoff(&[q(1, 100)], 100, harmonic).unwrap(), 1);
        assert_eq!(
            by_rejection_cutoff(&[q(101, 10_000)], 10, harmonic).unwrap(),
            0
        );
    }

    #[test]
    fn by_kernel_rejects_unsorted_p_values() {
        let harmonic = harmonic_upper_micros(2).unwrap();
        assert!(by_rejection_cutoff(&[q(2, 10), q(1, 10)], 100, harmonic).is_err());
        assert!(adjusted_p_value_upper(&[q(2, 10), q(1, 10)], harmonic).is_err());
    }

    #[test]
    fn p_value_sort_comparison_is_exact() {
        assert_eq!(compare_rational(1, 3, 2, 6), std::cmp::Ordering::Equal);
        assert_eq!(compare_rational(1, 4, 1, 3), std::cmp::Ordering::Less);
    }

    #[test]
    fn repeated_items_are_deduplicated_to_one_cluster_hypothesis() {
        let report = source_report();
        let config = config(vec![ByFamilyScopeV1::PerDetectorPerGroup]);
        let families = build_families(&report, &config.family_scopes).unwrap();
        assert_eq!(families.len(), 2);
        assert!(families.values().all(|family| family.len() == 3));
        assert!(
            families
                .values()
                .all(|family| family[0].item_sample_ids.len() == 2)
        );
    }

    #[test]
    fn per_detector_and_joint_detector_scopes_are_distinct_families() {
        let report = source_report();
        let per = build_families(&report, &[ByFamilyScopeV1::PerDetectorPerGroup]).unwrap();
        let joint = build_families(&report, &[ByFamilyScopeV1::AllDetectorsPerGroup]).unwrap();
        assert_eq!(per.len(), 2);
        assert_eq!(joint.len(), 1);
        assert_eq!(joint.values().next().unwrap().len(), 6);
    }

    #[test]
    fn joint_scope_rejects_detector_cluster_set_mismatch() {
        let mut report = source_report();
        report
            .scored_test_p_values
            .retain(|row| !(row.detector_id == "det-b" && row.source_cluster_id == "cluster-2"));
        assert!(build_families(&report, &[ByFamilyScopeV1::AllDetectorsPerGroup]).is_err());
    }

    #[test]
    fn same_cluster_p_value_must_bind_same_cluster_maximum() {
        let mut report = source_report();
        report.scored_test_p_values[1].cluster_max_normalized_risk_score += 0.01;
        assert!(build_families(&report, &[ByFamilyScopeV1::PerDetectorPerGroup]).is_err());
    }

    #[test]
    fn detectors_must_share_cluster_item_ids_for_joint_family() {
        let mut report = source_report();
        report
            .scored_test_p_values
            .retain(|row| !(row.detector_id == "det-b" && row.sample_id == "cluster-0-item-1"));
        report.scored_locked_test_observation_count = report.scored_test_p_values.len() as u64;
        assert!(build_families(&report, &[ByFamilyScopeV1::AllDetectorsPerGroup]).is_err());
    }

    #[test]
    fn source_report_score_count_must_match_embedded_p_values() {
        let mut report = source_report();
        report.scored_locked_test_observation_count -= 1;
        let config = config(vec![ByFamilyScopeV1::AllDetectorsPerGroup]);
        assert!(analyze_conformal_by(&report, &config).is_err());
    }

    #[test]
    fn repeated_item_p_values_must_agree_within_cluster() {
        let mut report = source_report();
        report.scored_test_p_values[1].conformal_p_value.numerator = 9;
        assert!(build_families(&report, &[ByFamilyScopeV1::PerDetectorPerGroup]).is_err());
    }

    #[test]
    fn config_must_bind_to_exact_source_calibration_report_digest() {
        let report = source_report();
        let mut config = config(vec![ByFamilyScopeV1::AllDetectorsPerGroup]);
        config.source_calibration_report_sha256 = "c".repeat(64);
        assert!(analyze_conformal_by(&report, &config).is_err());
    }

    #[test]
    fn config_must_bind_to_source_calibration_study_digest() {
        let report = source_report();
        let mut config = config(vec![ByFamilyScopeV1::AllDetectorsPerGroup]);
        config.source_calibration_study_sha256 = "c".repeat(64);
        assert!(analyze_conformal_by(&report, &config).is_err());
    }

    #[test]
    fn joint_detector_family_controls_against_detector_selection() {
        let report = source_report();
        let mut config = config(vec![ByFamilyScopeV1::AllDetectorsPerGroup]);
        config.target_fdr_permille = vec![100];
        let result = analyze_conformal_by(&report, &config).unwrap();
        assert_eq!(result.family_count, 1);
        assert_eq!(result.hypothesis_count, 6);
        assert_eq!(
            result.families[0].family_scope,
            ByFamilyScopeV1::AllDetectorsPerGroup
        );
        assert!(result.arbitrary_dependence_adjustment);
        assert!(!result.confirmatory_eligible);
    }
}
