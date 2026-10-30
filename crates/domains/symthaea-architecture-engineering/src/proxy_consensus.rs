//! Target-free multi-proxy rank consensus and anti-correlation gate.
//!
//! Implements a CoRA-style prior:
//!   s(a) = sum_i ln(rank_i(a) / N)
//! where best rank is N, plus a leave-one-out Spearman agreement gate.
//!
//! The gate is a target-free heuristic, not evidence that proxy agreement predicts accuracy.
//! It catches clear sign-flips but does not detect all weak-signal dilution. No target labels,
//! oracle ranks or challenge-set performance are accepted by this API.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

use super::nasbench201_trajectory::{
    build_trajectory_archive, plan_promotions, ProxyDirection, PromotionMix,
    TrajectorySearchConfig,
};
use super::tabular_nas_firewall::ArchitectureFeatures;

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyConsensusConfig {
    pub benchmark_artifact_sha256: String,
    pub evaluation_split_id: String,
    pub proxy_calibration_split_id: String,
    pub proxy_calibration_sha256: String,
    pub target_metric_id: String,
    pub proxy_directions: BTreeMap<String, ProxyDirection>,
    pub anti_correlation_threshold: f64,
    pub minimum_retained_proxies: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyGateReason {
    SingleProxyNoAgreementTest,
    PassedAgreementThreshold,
    RejectedAntiCorrelation,
    UndefinedAgreement,
    RetainedByMinimumFallback,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyAgreement {
    pub proxy_id: String,
    pub spearman_vs_leave_one_out: Option<f64>,
    pub retained: bool,
    pub reason: ProxyGateReason,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyConsensusResult {
    pub config: ProxyConsensusConfig,
    pub kept_proxy_ids: Vec<String>,
    pub rejected_proxy_ids: Vec<String>,
    pub agreements: Vec<ProxyAgreement>,
    pub proxy_ranks: BTreeMap<String, BTreeMap<String, f64>>,
    pub consensus_scores: BTreeMap<String, f64>,
    pub prior_percentiles: BTreeMap<String, f64>,
    pub feature_catalog_sha256: String,
    pub result_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyConsensusError {
    pub field: String,
    pub message: String,
}

impl ProxyConsensusError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for ProxyConsensusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for ProxyConsensusError {}

/// Compute per-proxy best-N midranks, a leave-one-out agreement gate and log-rank consensus.
///
/// Ties receive average ranks. A proxy is dropped only when it falls below the configured
/// anti-correlation threshold, unless that would leave fewer than the configured minimum
/// number of proxies; then the highest-agreement proxies are retained as a deterministic
/// fallback. A single proxy is retained because no agreement statistic can be computed.
pub fn build_proxy_consensus(
    features: &[ArchitectureFeatures],
    config: &ProxyConsensusConfig,
) -> Result<ProxyConsensusResult, ProxyConsensusError> {
    validate_inputs(features, config)?;

    let architecture_ids: Vec<String> = features
        .iter()
        .map(|feature| feature.architecture_id.clone())
        .collect();
    let proxy_ids: Vec<String> = config.proxy_directions.keys().cloned().collect();
    let architecture_count = features.len();

    let mut proxy_ranks: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut rank_vectors: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for proxy_id in &proxy_ids {
        let direction = config.proxy_directions[proxy_id];
        let values: Vec<f64> = features
            .iter()
            .map(|feature| feature.proxy_scores[proxy_id])
            .collect();
        let ranks = best_n_ranks(&values, direction);
        let rank_map: BTreeMap<String, f64> = architecture_ids
            .iter()
            .cloned()
            .zip(ranks.iter().copied())
            .collect();
        proxy_ranks.insert(proxy_id.clone(), rank_map);
        rank_vectors.insert(proxy_id.clone(), ranks);
    }

    let mut raw_agreements: Vec<(String, Option<f64>)> = Vec::with_capacity(proxy_ids.len());
    for proxy_id in &proxy_ids {
        if proxy_ids.len() == 1 {
            raw_agreements.push((proxy_id.clone(), None));
            continue;
        }
        let loo_consensus: Vec<f64> = (0..architecture_count)
            .map(|candidate_index| {
                let mut sum = 0.0;
                let mut count = 0usize;
                for other_id in &proxy_ids {
                    if other_id != proxy_id {
                        sum += rank_vectors[other_id][candidate_index];
                        count += 1;
                    }
                }
                sum / count as f64
            })
            .collect();
        raw_agreements.push((
            proxy_id.clone(),
            spearman(&rank_vectors[proxy_id], &loo_consensus),
        ));
    }

    let mut selected: BTreeSet<String> = raw_agreements
        .iter()
        .filter_map(|(proxy_id, alpha)| {
            if (*alpha).is_some_and(|value| value >= config.anti_correlation_threshold) {
                Some(proxy_id.clone())
            } else {
                None
            }
        })
        .collect();

    let required_minimum = config.minimum_retained_proxies.min(proxy_ids.len());
    if selected.len() < required_minimum {
        let mut ordered = raw_agreements.clone();
        ordered.sort_by(|(left_id, left_alpha), (right_id, right_alpha)| {
            let left_value = left_alpha.unwrap_or(-1.0);
            let right_value = right_alpha.unwrap_or(-1.0);
            right_value
                .total_cmp(&left_value)
                .then_with(|| left_id.cmp(right_id))
        });
        for (proxy_id, _) in ordered {
            if selected.len() >= required_minimum {
                break;
            }
            selected.insert(proxy_id);
        }
    }

    let agreements: Vec<ProxyAgreement> = raw_agreements
        .iter()
        .map(|(proxy_id, alpha)| {
            let retained = selected.contains(proxy_id);
            let reason = if proxy_ids.len() == 1 {
                ProxyGateReason::SingleProxyNoAgreementTest
            } else if retained && (*alpha).is_some_and(|value| value >= config.anti_correlation_threshold) {
                ProxyGateReason::PassedAgreementThreshold
            } else if retained {
                ProxyGateReason::RetainedByMinimumFallback
            } else if alpha.is_none() {
                ProxyGateReason::UndefinedAgreement
            } else {
                ProxyGateReason::RejectedAntiCorrelation
            };
            ProxyAgreement {
                proxy_id: proxy_id.clone(),
                spearman_vs_leave_one_out: *alpha,
                retained,
                reason,
            }
        })
        .collect();

    let kept_proxy_ids: Vec<String> = selected.iter().cloned().collect();
    let rejected_proxy_ids: Vec<String> = proxy_ids
        .iter()
        .filter(|proxy_id| !selected.contains(*proxy_id))
        .cloned()
        .collect();

    let mut consensus_scores = BTreeMap::new();
    for architecture_id in &architecture_ids {
        let mut score = 0.0;
        for proxy_id in &kept_proxy_ids {
            let percentile_rank =
                proxy_ranks[architecture_id][proxy_id] / architecture_count as f64;
            if !percentile_rank.is_finite() || percentile_rank <= 0.0 {
                return Err(ProxyConsensusError::new(
                    "proxy_ranks",
                    format!("invalid percentile rank for {architecture_id:?}/{proxy_id:?}"),
                ));
            }
            score += percentile_rank.ln();
        }
        consensus_scores.insert(architecture_id.clone(), score);
    }

    let consensus_values: Vec<f64> = architecture_ids
        .iter()
        .map(|architecture_id| consensus_scores[architecture_id])
        .collect();
    let percentile_ranks = best_n_ranks(&consensus_values, ProxyDirection::Maximize);
    let prior_percentiles = architecture_ids
        .iter()
        .cloned()
        .zip(
            percentile_ranks
                .iter()
                .map(|rank| rank / architecture_count as f64),
        )
        .collect();

    let feature_catalog_sha256 = digest_feature_catalog(features, config);
    let result_sha256 = digest_result(
        config,
        &feature_catalog_sha256,
        &agreements,
        &proxy_ranks,
        &consensus_scores,
        &prior_percentiles,
    );

    Ok(ProxyConsensusResult {
        config: config.clone(),
        kept_proxy_ids,
        rejected_proxy_ids,
        agreements,
        proxy_ranks,
        consensus_scores,
        prior_percentiles,
        feature_catalog_sha256,
        result_sha256,
    })
}

/// Clone feature rows and add the consensus percentile as a derived proxy for the trajectory
/// planner. Its per-model evaluation time is recorded as zero because no new model forward or
/// backward pass is performed; the source proxy costs remain on the original feature columns.
/// The outer runner should still measure and report this aggregation's CPU time separately.
pub fn apply_consensus_prior(
    features: &[ArchitectureFeatures],
    result: &ProxyConsensusResult,
    derived_proxy_id: &str,
) -> Result<Vec<ArchitectureFeatures>, ProxyConsensusError> {
    if derived_proxy_id.trim().is_empty() || derived_proxy_id == result.config.target_metric_id.as_str() {
        return Err(ProxyConsensusError::new(
            "derived_proxy_id",
            "must be non-empty and distinct from the target metric ID",
        ));
    }
    let recomputed = build_proxy_consensus(features, &result.config)?;
    if recomputed.result_sha256 != result.result_sha256 || &recomputed != result {
        return Err(ProxyConsensusError::new(
            "result.result_sha256",
            "feature catalog or consensus result does not match its content digest",
        ));
    }

    let mut output = Vec::with_capacity(features.len());
    for feature in features {
        let Some(percentile) = result.prior_percentiles.get(&feature.architecture_id) else {
            return Err(ProxyConsensusError::new(
                "result.prior_percentiles",
                format!("missing prior for {:?}", feature.architecture_id),
            ));
        };
        if feature.proxy_scores.contains_key(derived_proxy_id)
            || feature.proxy_times_seconds.contains_key(derived_proxy_id)
        {
            return Err(ProxyConsensusError::new(
                "derived_proxy_id",
                format!("derived proxy {derived_proxy_id:?} already exists"),
            ));
        }
        let mut cloned = feature.clone();
        cloned
            .proxy_scores
            .insert(derived_proxy_id.to_owned(), *percentile);
        cloned
            .proxy_times_seconds
            .insert(derived_proxy_id.to_owned(), 0.0);
        output.push(cloned);
    }
    Ok(output)
}

fn validate_inputs(
    features: &[ArchitectureFeatures],
    config: &ProxyConsensusConfig,
) -> Result<(), ProxyConsensusError> {
    if features.is_empty() {
        return Err(ProxyConsensusError::new("features", "at least one architecture is required"));
    }
    if config.target_metric_id.trim().is_empty()
        || config.evaluation_split_id.trim().is_empty()
        || config.proxy_calibration_split_id.trim().is_empty()
    {
        return Err(ProxyConsensusError::new(
            "target_metric_id",
            "target metric and split IDs must be non-empty",
        ));
    }
    if config.evaluation_split_id == config.proxy_calibration_split_id {
        return Err(ProxyConsensusError::new(
            "proxy_calibration_split_id",
            "strict no-oracle consensus requires a disjoint calibration split",
        ));
    }
    for (field, digest) in [
        ("benchmark_artifact_sha256", config.benchmark_artifact_sha256.as_str()),
        ("proxy_calibration_sha256", config.proxy_calibration_sha256.as_str()),
    ] {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(ProxyConsensusError::new(
                field,
                "must be exactly 64 hexadecimal characters",
            ));
        }
    }
    if config.proxy_directions.is_empty() {
        return Err(ProxyConsensusError::new("proxy_directions", "at least one proxy is required"));
    }
    if !config.anti_correlation_threshold.is_finite()
        || !(-1.0..=1.0).contains(&config.anti_correlation_threshold)
    {
        return Err(ProxyConsensusError::new(
            "anti_correlation_threshold",
            "must be finite and in [-1,1]",
        ));
    }
    if config.minimum_retained_proxies == 0
        || config.minimum_retained_proxies > config.proxy_directions.len()
    {
        return Err(ProxyConsensusError::new(
            "minimum_retained_proxies",
            "must be between one and the declared proxy count",
        ));
    }
    for proxy_id in config.proxy_directions.keys() {
        if proxy_id.trim().is_empty() || proxy_id == &config.target_metric_id {
            return Err(ProxyConsensusError::new(
                "proxy_directions",
                "proxy names must be non-empty and distinct from the target metric",
            ));
        }
    }

    let mut architecture_ids = BTreeSet::new();
    let mut topology_encodings = BTreeSet::new();
    for (index, feature) in features.iter().enumerate() {
        if feature.architecture_id.trim().is_empty()
            || !architecture_ids.insert(feature.architecture_id.as_str())
        {
            return Err(ProxyConsensusError::new(
                format!("features[{index}].architecture_id"),
                "architecture IDs must be non-empty and unique",
            ));
        }
        if feature.topology_encoding.len() != 6
            || feature.topology_encoding.iter().any(|operation| *operation >= 5)
        {
            return Err(ProxyConsensusError::new(
                format!("features[{index}].topology_encoding"),
                "NAS-Bench-201 topology must contain six operation IDs in 0..5",
            ));
        }
        if feature.proxy_scores.contains_key(&config.target_metric_id)
            || feature.proxy_times_seconds.contains_key(&config.target_metric_id)
        {
            return Err(ProxyConsensusError::new(
                format!("features[{index}]"),
                "target metric cannot appear in target-free proxy values or runtimes",
            ));
        }
        let declared_proxy_ids: BTreeSet<&str> =
            config.proxy_directions.keys().map(String::as_str).collect();
        let score_proxy_ids: BTreeSet<&str> =
            feature.proxy_scores.keys().map(String::as_str).collect();
        let runtime_proxy_ids: BTreeSet<&str> =
            feature.proxy_times_seconds.keys().map(String::as_str).collect();
        if score_proxy_ids != declared_proxy_ids || runtime_proxy_ids != declared_proxy_ids {
            return Err(ProxyConsensusError::new(
                format!("features[{index}]"),
                "proxy score and runtime keys must exactly match the declared proxy-direction schema",
            ));
        }
        if feature.topology_encoding.len() != 6
            || feature.topology_encoding.iter().any(|operation| *operation >= 5)
        {
            return Err(ProxyConsensusError::new(
                format!("features[{index}].topology_encoding"),
                "NAS-Bench-201 topology must contain six operation IDs in 0..5",
            ));
        }
        if !topology_encodings.insert(feature.topology_encoding.as_slice()) {
            return Err(ProxyConsensusError::new(
                format!("features[{index}].topology_encoding"),
                "duplicate NAS-Bench-201 topology encoding",
            ));
        }
        for proxy_id in config.proxy_directions.keys() {
            let Some(value) = feature.proxy_scores.get(proxy_id) else {
                return Err(ProxyConsensusError::new(
                    format!("features[{index}].proxy_scores"),
                    format!("missing proxy {proxy_id:?}"),
                ));
            };
            if !value.is_finite() {
                return Err(ProxyConsensusError::new(
                    format!("features[{index}].proxy_scores.{proxy_id}"),
                    "proxy value must be finite",
                ));
            }
            let Some(runtime) = feature.proxy_times_seconds.get(proxy_id) else {
                return Err(ProxyConsensusError::new(
                    format!("features[{index}].proxy_times_seconds"),
                    format!("missing runtime for proxy {proxy_id:?}"),
                ));
            };
            if !runtime.is_finite() || *runtime < 0.0 {
                return Err(ProxyConsensusError::new(
                    format!("features[{index}].proxy_times_seconds.{proxy_id}"),
                    "proxy runtime must be finite and non-negative",
                ));
            }
        }
    }
    Ok(())
}

/// Convert raw values into average ranks with best=N and worst=1. Tied values receive a
/// midrank. Direction defines whether numerically high or low values are better.
fn best_n_ranks(values: &[f64], direction: ProxyDirection) -> Vec<f64> {
    let mut ordered: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    ordered.sort_by(|(left_index, left_value), (right_index, right_value)| {
        let order = match direction {
            ProxyDirection::Maximize => left_value.total_cmp(right_value),
            ProxyDirection::Minimize => right_value.total_cmp(left_value),
        };
        order.then_with(|| left_index.cmp(right_index))
    });

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && ordered[end].1 == ordered[start].1 {
            end += 1;
        }
        let average_rank = ((start + 1) as f64 + end as f64) / 2.0;
        for (original_index, _) in &ordered[start..end] {
            ranks[*original_index] = average_rank;
        }
        start = end;
    }
    ranks
}

fn spearman(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_ranks = best_n_ranks(left, ProxyDirection::Maximize);
    let right_ranks = best_n_ranks(right, ProxyDirection::Maximize);
    pearson(&left_ranks, &right_ranks)
}

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let left_delta = *left_value - left_mean;
        let right_delta = *right_value - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator == 0.0 || !denominator.is_finite() {
        None
    } else {
        Some((covariance / denominator).clamp(-1.0, 1.0))
    }
}

fn digest_feature_catalog(
    features: &[ArchitectureFeatures],
    config: &ProxyConsensusConfig,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nasbench201.proxy-consensus-catalog.v0.1");
    update_string(&mut hasher, &config.target_metric_id);
    update_string(&mut hasher, &config.benchmark_artifact_sha256);
    update_string(&mut hasher, &config.evaluation_split_id);
    update_string(&mut hasher, &config.proxy_calibration_split_id);
    update_string(&mut hasher, &config.proxy_calibration_sha256);
    hasher.update(config.anti_correlation_threshold.to_bits().to_be_bytes());
    hasher.update((config.minimum_retained_proxies as u64).to_be_bytes());
    hasher.update((config.proxy_directions.len() as u64).to_be_bytes());
    for (proxy_id, direction) in &config.proxy_directions {
        update_string(&mut hasher, proxy_id);
        hasher.update([direction_byte(*direction)]);
    }
    let mut ordered: Vec<&ArchitectureFeatures> = features.iter().collect();
    ordered.sort_by(|left, right| left.architecture_id.cmp(&right.architecture_id));
    hasher.update((ordered.len() as u64).to_be_bytes());
    for feature in ordered {
        update_string(&mut hasher, &feature.architecture_id);
        hasher.update((feature.topology_encoding.len() as u64).to_be_bytes());
        hasher.update(&feature.topology_encoding);
        for proxy_id in config.proxy_directions.keys() {
            update_string(&mut hasher, proxy_id);
            hasher.update(feature.proxy_scores[proxy_id].to_bits().to_be_bytes());
            let runtime = feature.proxy_times_seconds[proxy_id];
            hasher.update(runtime.to_bits().to_be_bytes());
        }
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_result(
    config: &ProxyConsensusConfig,
    catalog_sha256: &str,
    agreements: &[ProxyAgreement],
    proxy_ranks: &BTreeMap<String, BTreeMap<String, f64>>,
    consensus_scores: &BTreeMap<String, f64>,
    prior_percentiles: &BTreeMap<String, f64>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nasbench201.proxy-consensus-result.v0.1");
    update_string(&mut hasher, &config.target_metric_id);
    update_string(&mut hasher, &config.benchmark_artifact_sha256);
    update_string(&mut hasher, &config.evaluation_split_id);
    update_string(&mut hasher, &config.proxy_calibration_split_id);
    update_string(&mut hasher, &config.proxy_calibration_sha256);
    update_string(&mut hasher, catalog_sha256);
    hasher.update(config.anti_correlation_threshold.to_bits().to_be_bytes());
    hasher.update((config.minimum_retained_proxies as u64).to_be_bytes());
    hasher.update((agreements.len() as u64).to_be_bytes());
    for agreement in agreements {
        update_string(&mut hasher, &agreement.proxy_id);
        match agreement.spearman_vs_leave_one_out {
            Some(value) => {
                hasher.update([1]);
                hasher.update(value.to_bits().to_be_bytes());
            }
            None => hasher.update([0]),
        }
        hasher.update([agreement.retained as u8]);
        hasher.update([gate_reason_byte(agreement.reason)]);
    }
    hasher.update((proxy_ranks.len() as u64).to_be_bytes());
    for (architecture_id, ranks) in proxy_ranks {
        update_string(&mut hasher, architecture_id);
        hasher.update((ranks.len() as u64).to_be_bytes());
        for (proxy_id, rank) in ranks {
            update_string(&mut hasher, proxy_id);
            hasher.update(rank.to_bits().to_be_bytes());
        }
    }
    hasher.update((consensus_scores.len() as u64).to_be_bytes());
    for (architecture_id, score) in consensus_scores {
        update_string(&mut hasher, architecture_id);
        hasher.update(score.to_bits().to_be_bytes());
    }
    hasher.update((prior_percentiles.len() as u64).to_be_bytes());
    for (architecture_id, percentile) in prior_percentiles {
        update_string(&mut hasher, architecture_id);
        hasher.update(percentile.to_bits().to_be_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

fn direction_byte(direction: ProxyDirection) -> u8 {
    match direction {
        ProxyDirection::Maximize => 0,
        ProxyDirection::Minimize => 1,
    }
}

fn gate_reason_byte(reason: ProxyGateReason) -> u8 {
    match reason {
        ProxyGateReason::SingleProxyNoAgreementTest => 0,
        ProxyGateReason::PassedAgreementThreshold => 1,
        ProxyGateReason::RejectedAntiCorrelation => 2,
        ProxyGateReason::UndefinedAgreement => 3,
        ProxyGateReason::RetainedByMinimumFallback => 4,
    }
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().copied() {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 15) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features() -> Vec<ArchitectureFeatures> {
        (0..6)
            .map(|index| {
                let mut scores = BTreeMap::new();
                scores.insert("proxy_a".into(), index as f64);
                scores.insert("proxy_b".into(), index as f64);
                scores.insert("proxy_c".into(), index as f64);
                scores.insert("proxy_flip".into(), (5 - index) as f64);
                ArchitectureFeatures {
                    architecture_id: format!("arch-{index}"),
                    proxy_scores: scores,
                    proxy_times_seconds: BTreeMap::from([
                        ("proxy_a".into(), 0.01),
                        ("proxy_b".into(), 0.02),
                        ("proxy_c".into(), 0.03),
                        ("proxy_flip".into(), 0.04),
                    ]),
                    topology_encoding: vec![(index / 5) as u8, (index % 5) as u8, 0, 0, 0, 0],
                }
            })
            .collect()
    }

    fn config() -> ProxyConsensusConfig {
        let mut directions = BTreeMap::new();
        for proxy in ["proxy_a", "proxy_b", "proxy_c", "proxy_flip"] {
            directions.insert(proxy.into(), ProxyDirection::Maximize);
        }
        ProxyConsensusConfig {
            benchmark_artifact_sha256: "a".repeat(64),
            evaluation_split_id: "challenge-cifar10".into(),
            proxy_calibration_split_id: "development-cifar100".into(),
            proxy_calibration_sha256: "b".repeat(64),
            target_metric_id: "val_accuracy".into(),
            proxy_directions: directions,
            anti_correlation_threshold: -0.3,
            minimum_retained_proxies: 3,
        }
    }

    #[test]
    fn gate_drops_clear_sign_flipped_proxy_without_target_labels() {
        let result = build_proxy_consensus(&features(), &config()).unwrap();
        assert_eq!(result.kept_proxy_ids.len(), 3);
        assert!(result.rejected_proxy_ids.contains(&"proxy_flip".to_string()));
        let flip = result
            .agreements
            .iter()
            .find(|entry| entry.proxy_id == "proxy_flip")
            .unwrap();
        assert_eq!(flip.reason, ProxyGateReason::RejectedAntiCorrelation);
        assert!(flip.spearman_vs_leave_one_out.unwrap() < -0.3);
    }

    #[test]
    fn log_rank_consensus_rewards_jointly_high_ranks() {
        let result = build_proxy_consensus(&features(), &config()).unwrap();
        assert!(result.prior_percentiles["arch-5"] > result.prior_percentiles["arch-0"]);
    }

    #[test]
    fn minimize_direction_reverses_proxy_rank_order() {
        let mut rows = features();
        let mut directions = BTreeMap::new();
        directions.insert("proxy_a".into(), ProxyDirection::Minimize);
        for (index, row) in rows.iter_mut().enumerate() {
            row.proxy_scores.retain(|key, _| key == "proxy_a");
            row.proxy_scores.insert("proxy_a".into(), index as f64);
            row.proxy_times_seconds.retain(|key, _| key == "proxy_a");
            row.proxy_times_seconds.insert("proxy_a".into(), 0.01);
        }
        let cfg = ProxyConsensusConfig {
            benchmark_artifact_sha256: "a".repeat(64),
            evaluation_split_id: "challenge-latency".into(),
            proxy_calibration_split_id: "development-latency".into(),
            proxy_calibration_sha256: "b".repeat(64),
            target_metric_id: "latency".into(),
            proxy_directions: directions,
            anti_correlation_threshold: -1.0,
            minimum_retained_proxies: 1,
        };
        let result = build_proxy_consensus(&rows, &cfg).unwrap();
        assert!(result.proxy_ranks["arch-0"]["proxy_a"] > result.proxy_ranks["arch-5"]["proxy_a"]);
    }

    #[test]
    fn average_ranks_are_used_for_ties() {
        let mut rows = features();
        for row in &mut rows {
            row.proxy_scores.retain(|key, _| key == "p");
            row.proxy_times_seconds.retain(|key, _| key == "p");
            row.proxy_scores.insert("p".into(), 1.0);
            row.proxy_times_seconds.insert("p".into(), 0.01);
        }
        let mut directions = BTreeMap::new();
        directions.insert("p".into(), ProxyDirection::Maximize);
        let cfg = ProxyConsensusConfig {
            benchmark_artifact_sha256: "a".repeat(64),
            evaluation_split_id: "challenge".into(),
            proxy_calibration_split_id: "development".into(),
            proxy_calibration_sha256: "b".repeat(64),
            target_metric_id: "accuracy".into(),
            proxy_directions: directions,
            anti_correlation_threshold: -0.3,
            minimum_retained_proxies: 1,
        };
        let result = build_proxy_consensus(&rows, &cfg).unwrap();
        assert!(result.proxy_ranks.values().all(|ranks| ranks["p"] == 3.5));
    }

    #[test]
    fn minimum_retained_fallback_is_explicit() {
        let mut rows = features();
        for (index, row) in rows.iter_mut().enumerate() {
            row.proxy_scores.retain(|key, _| key == "proxy_a" || key == "proxy_b");
            row.proxy_times_seconds.retain(|key, _| key == "proxy_a" || key == "proxy_b");
            row.proxy_scores.insert("anti_a".into(), (5 - index) as f64);
            row.proxy_scores.insert("anti_b".into(), (5 - index) as f64);
            row.proxy_times_seconds.insert("anti_a".into(), 0.05);
            row.proxy_times_seconds.insert("anti_b".into(), 0.06);
        }
        let mut directions = BTreeMap::new();
        for proxy in ["proxy_a", "proxy_b", "anti_a", "anti_b"] {
            directions.insert(proxy.into(), ProxyDirection::Maximize);
        }
        let cfg = ProxyConsensusConfig {
            benchmark_artifact_sha256: "a".repeat(64),
            evaluation_split_id: "challenge-cifar10".into(),
            proxy_calibration_split_id: "development-cifar100".into(),
            proxy_calibration_sha256: "b".repeat(64),
            target_metric_id: "val_accuracy".into(),
            proxy_directions: directions,
            anti_correlation_threshold: 0.9,
            minimum_retained_proxies: 3,
        };
        let result = build_proxy_consensus(&rows, &cfg).unwrap();
        assert_eq!(result.kept_proxy_ids.len(), 3);
        assert!(result
            .agreements
            .iter()
            .any(|agreement| agreement.reason == ProxyGateReason::RetainedByMinimumFallback));
    }

    #[test]
    fn result_digest_is_deterministic_and_feature_bound() {
        let rows = features();
        let first = build_proxy_consensus(&rows, &config()).unwrap();
        let second = build_proxy_consensus(&rows, &config()).unwrap();
        assert_eq!(first.result_sha256, second.result_sha256);
        let mut changed = rows;
        changed[0].proxy_scores.insert("proxy_a".into(), 99.0);
        let other = build_proxy_consensus(&changed, &config()).unwrap();
        assert_ne!(first.feature_catalog_sha256, other.feature_catalog_sha256);
    }

    #[test]
    fn undefined_constant_proxy_is_not_treated_as_a_positive_agreement() {
        let mut rows = features();
        for row in &mut rows {
            row.proxy_scores.insert("constant".into(), 1.0);
            row.proxy_times_seconds.insert("constant".into(), 0.01);
        }
        let mut cfg = config();
        cfg.proxy_directions.insert("constant".into(), ProxyDirection::Maximize);
        cfg.minimum_retained_proxies = 3;
        let result = build_proxy_consensus(&rows, &cfg).unwrap();
        let constant = result
            .agreements
            .iter()
            .find(|agreement| agreement.proxy_id == "constant")
            .unwrap();
        assert_eq!(constant.reason, ProxyGateReason::UndefinedAgreement);
        assert!(!constant.retained);
    }

    #[test]
    fn strict_no_oracle_manifest_requires_separate_calibration_split_and_digests() {
        let rows = features();
        let mut invalid = config();
        invalid.proxy_calibration_split_id = invalid.evaluation_split_id.clone();
        assert!(build_proxy_consensus(&rows, &invalid).is_err());

        let mut invalid = config();
        invalid.proxy_calibration_sha256 = "invalid".into();
        assert!(build_proxy_consensus(&rows, &invalid).is_err());
    }

    #[test]
    fn apply_consensus_rejects_tampered_public_prior_fields() {
        let rows = features();
        let mut result = build_proxy_consensus(&rows, &config()).unwrap();
        result.prior_percentiles.insert("arch-5".into(), 0.0);
        assert!(apply_consensus_prior(&rows, &result, "tampered_prior").is_err());
    }

    #[test]
    fn target_metric_column_is_rejected_even_when_not_selected() {
        let mut rows = features();
        for row in &mut rows {
            row.proxy_scores.insert("val_accuracy".into(), 99.0);
        }
        assert!(build_proxy_consensus(&rows, &config()).is_err());
    }

    #[test]
    fn invalid_topology_and_missing_proxy_runtime_are_rejected() {
        let mut invalid_topology = features();
        invalid_topology[0].topology_encoding.pop();
        assert!(build_proxy_consensus(&invalid_topology, &config()).is_err());

        let mut missing_runtime = features();
        missing_runtime[0].proxy_times_seconds.remove("proxy_a");
        assert!(build_proxy_consensus(&missing_runtime, &config()).is_err());
    }

    #[test]
    fn target_metric_runtime_column_is_rejected() {
        let mut rows = features();
        for row in &mut rows {
            row.proxy_times_seconds.insert("val_accuracy".into(), 0.0);
        }
        assert!(build_proxy_consensus(&rows, &config()).is_err());
    }

    #[test]
    fn exact_proxy_schema_rejects_unused_or_missing_columns() {
        let mut rows = features();
        rows[0].proxy_scores.insert("unselected_extra".into(), 0.0);
        rows[0].proxy_times_seconds.insert("unselected_extra".into(), 0.01);
        assert!(build_proxy_consensus(&rows, &config()).is_err());

        let mut rows = features();
        rows[0].proxy_times_seconds.remove("proxy_a");
        assert!(build_proxy_consensus(&rows, &config()).is_err());
    }

    #[test]
    fn target_metric_cannot_be_selected_as_proxy() {
        let rows = features();
        let mut invalid = config();
        invalid
            .proxy_directions
            .insert("val_accuracy".into(), ProxyDirection::Maximize);
        assert!(build_proxy_consensus(&rows, &invalid).is_err());
    }

    #[test]
    fn consensus_trajectory_and_promotion_pipeline_composes_without_target_labels() {
        let mut rows = features();
        for (index, row) in rows.iter_mut().enumerate() {
            row.topology_encoding = vec![
                (index / 5) as u8,
                (index % 5) as u8,
                0,
                0,
                0,
                0,
            ];
        }
        let consensus = build_proxy_consensus(&rows, &config()).unwrap();
        let derived = apply_consensus_prior(&rows, &consensus, "cora_consensus").unwrap();
        let search = TrajectorySearchConfig {
            proxy_id: "cora_consensus".into(),
            target_metric_id: "val_accuracy".into(),
            proxy_direction: ProxyDirection::Maximize,
            seed: 19,
            max_walks: 3,
            max_unique_visits: 6,
            max_moves_per_walk: 8,
            max_escapes_per_walk: 1,
        };
        let archive = build_trajectory_archive(&derived, &search).unwrap();
        let mix = PromotionMix {
            promotion_count: 4,
            proxy_elite_quota: 1,
            trajectory_quota: 1,
            topology_diversity_quota: 2,
        };
        let plan = plan_promotions(&derived, &archive, &mix).unwrap();
        assert_eq!(plan.candidates.len(), 4);
        assert_eq!(plan.proxy_id, "cora_consensus");
        assert_eq!(plan.archive_sha256, archive.archive_sha256);
    }

    #[test]
    fn consensus_trajectory_promotion_runs_through_multi_fidelity_query_oracle() {
        use super::super::multi_fidelity_tabular_nas::{
            FidelityCostModel, FidelityLevel, FidelityTargetLabel, MultiFidelityManifest,
            MultiFidelityTabularBenchmark,
        };
        use super::super::proxy_acquisition::{
            ProxyAcquisitionBenchmark, ProxyAcquisitionManifest,
        };
        use super::super::tabular_nas_firewall::MetricDirection;

        let mut source_features = features();
        for (index, row) in source_features.iter_mut().enumerate() {
            row.topology_encoding = vec![
                (index / 5) as u8,
                (index % 5) as u8,
                0,
                0,
                0,
                0,
            ];
        }
        let consensus_config = config();
        let acquisition = ProxyAcquisitionBenchmark::new(
            ProxyAcquisitionManifest {
                benchmark_id: "synthetic-metered-proxy-acquisition".into(),
                target_metric_id: consensus_config.target_metric_id.clone(),
                artifact_sha256: consensus_config.benchmark_artifact_sha256.clone(),
                evaluation_split_id: consensus_config.evaluation_split_id.clone(),
                proxy_calibration_split_id: consensus_config.proxy_calibration_split_id.clone(),
                proxy_calibration_sha256: consensus_config.proxy_calibration_sha256.clone(),
                proxy_directions: consensus_config.proxy_directions.clone(),
                max_unique_proxy_queries: source_features.len() * consensus_config.proxy_directions.len(),
                max_total_proxy_seconds: 10.0,
            },
            source_features,
        )
        .unwrap()
        .run_policy(|candidates, oracle| {
            for candidate in candidates {
                for proxy_id in consensus_config.proxy_directions.keys() {
                    oracle.query(&candidate.architecture_id, proxy_id).unwrap();
                }
            }
        });
        assert_eq!(acquisition.receipt.unique_proxy_queries, 24);
        assert!((acquisition.receipt.total_proxy_seconds - 0.6).abs() < 1e-12);
        let rows = acquisition.to_complete_feature_table().unwrap();
        let consensus = build_proxy_consensus(&rows, &consensus_config).unwrap();
        let derived = apply_consensus_prior(&rows, &consensus, "cora_consensus").unwrap();
        let search = TrajectorySearchConfig {
            proxy_id: "cora_consensus".into(),
            target_metric_id: "val_accuracy".into(),
            proxy_direction: ProxyDirection::Maximize,
            seed: 23,
            max_walks: 3,
            max_unique_visits: 6,
            max_moves_per_walk: 8,
            max_escapes_per_walk: 1,
        };
        let archive = build_trajectory_archive(&derived, &search).unwrap();
        let mix = PromotionMix {
            promotion_count: 4,
            proxy_elite_quota: 1,
            trajectory_quota: 1,
            topology_diversity_quota: 2,
        };
        let plan = plan_promotions(&derived, &archive, &mix).unwrap();

        let target_labels = derived
            .iter()
            .flat_map(|feature| {
                let id = feature.architecture_id.clone();
                let value = feature.proxy_scores["cora_consensus"] * 10.0;
                [
                    FidelityTargetLabel {
                        architecture_id: id.clone(),
                        fidelity_index: 0,
                        value: 30.0 + value,
                    },
                    FidelityTargetLabel {
                        architecture_id: id,
                        fidelity_index: 1,
                        value: 70.0 + value,
                    },
                ]
            })
            .collect();
        let benchmark = MultiFidelityTabularBenchmark::new(
            MultiFidelityManifest {
                base: TabularBenchmarkManifest {
                    schema: "symthaea.tabular-nas.benchmark".into(),
                    version: "0.1.0".into(),
                    benchmark_id: "synthetic-consensus-trajectory-sh".into(),
                    metric_id: "val_accuracy".into(),
                    metric_direction: MetricDirection::Maximize,
                    split_id: "challenge-cifar10".into(),
                    proxy_calibration_split_id: "development-cifar100".into(),
                    proxy_calibration_sha256: "b".repeat(64),
                    source_description: "synthetic integration fixture".into(),
                    artifact_sha256: "a".repeat(64),
                    max_unique_queries: 2,
                    top_k: 6,
                },
                fidelity_levels: vec![
                    FidelityLevel {
                        id: "epoch_5".into(),
                        cumulative_cost_units: 5,
                    },
                    FidelityLevel {
                        id: "epoch_20".into(),
                        cumulative_cost_units: 20,
                    },
                ],
                fidelity_cost_model: FidelityCostModel::ResumablePrefix,
                resource_unit: "epoch_equivalent".into(),
                max_total_cost_units: 50,
            },
            derived,
            target_labels,
        )
        .unwrap();
        let initial_ids = plan
            .candidates
            .iter()
            .map(|candidate| candidate.architecture_id.clone())
            .collect();
        let run = benchmark
            .run_successive_halving(initial_ids, &[4, 2])
            .unwrap();

        assert_eq!(run.receipt.unique_architectures_touched, 4);
        assert_eq!(run.receipt.final_fidelity_queries, 2);
        assert_eq!(run.receipt.total_cost_units, 50);
        assert_eq!(run.evaluation.final_top_1.true_top_k_count, 1);
        assert!(run.evaluation.best_final_metric.is_some());
        assert_eq!(run.receipt.query_trace_sha256.len(), 64);
        assert_eq!(plan.archive_sha256, archive.archive_sha256);
    }

    #[test]
    fn derived_consensus_prior_can_be_applied_to_features() {
        let rows = features();
        let result = build_proxy_consensus(&rows, &config()).unwrap();
        let transformed = apply_consensus_prior(&rows, &result, "cora_consensus").unwrap();
        assert_eq!(transformed.len(), rows.len());
        for feature in transformed {
            assert!(feature.proxy_scores.contains_key("cora_consensus"));
            assert_eq!(feature.proxy_times_seconds["cora_consensus"], 0.0);
        }
    }
}
