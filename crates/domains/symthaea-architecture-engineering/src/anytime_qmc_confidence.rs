//! Anytime-valid bounded-mean certificates for adaptive NAS decisions.
//!
//! This module supplies a conservative sequential alternative to repeatedly inspecting
//! nominal normal intervals. A preregistered family alpha is Bonferroni-split across a
//! fixed number of decision streams. Each stream spends alpha over looks with
//! alpha_t = alpha_stream / (t * (t + 1)); the allocations sum to alpha_stream.
//! At each look, a two-sided Hoeffding bound is computed over replicate-level values
//! with a caller-declared finite support.
//!
//! For randomized QMC, the independent units are complete independently randomized
//! replicates, not the dependent integration points within one replicate. The bound is
//! finite-sample and time-uniform under the independent bounded-replicate assumption.
//! It does not prove independence, PRNG quality, posterior calibration, or data
//! representativeness. In particular, a deterministic PRNG seed gives reproducibility;
//! the statistical guarantee is conditional on the random-shift generator behaving as
//! independent uniform randomization. Treat this as an explicit assumption, not a
//! cryptographic or physical-randomness certificate.
//!
//! The caller must keep look indices monotone for each stream and must not reuse a
//! stream ID for a different hypothesis. The fixed family size and alpha allocation
//! must be committed before outcomes are inspected. For paired common-random-number
//! action comparisons, pass one paired difference per independently randomized
//! replicate and use support [-box_volume, +box_volume].

use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

const RECEIPT_DOMAIN: &[u8] = b"symthaea.nas.anytime-bounded-mean-receipt.v0.1";
const SAMPLE_DOMAIN: &[u8] = b"symthaea.nas.anytime-bounded-mean-samples.v0.1";
const MAX_FAMILY_SIZE: usize = 1_000_000;
const MAX_LOOK_INDEX: u64 = 1_000_000;
const MAX_REPLICATES: usize = 1_000_000;
const MIN_REPLICATES: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnytimeDecision {
    AtOrAboveThreshold,
    BelowThreshold,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnytimeBoundedMeanConfig {
    /// Preregistered family identity shared by all streams in one error-control family.
    pub family_id: String,
    /// Stable identity for this one hypothesis / action / portfolio comparison.
    pub stream_id: String,
    /// Family-wise error probability, before Bonferroni allocation.
    pub family_alpha: f64,
    /// Fixed number of streams in the family, declared before observing outcomes.
    pub family_size: usize,
    /// One-based look number for this stream.
    pub look_index: u64,
    /// Known lower support bound for every replicate-level observation.
    pub range_min: f64,
    /// Known upper support bound for every replicate-level observation.
    pub range_max: f64,
    /// Decision boundary in the same units as the replicate-level statistic.
    pub threshold: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnytimeBoundedMeanReceipt {
    pub schema: String,
    pub version: String,
    pub method: String,
    pub family_id: String,
    pub stream_id: String,
    pub family_alpha: f64,
    pub family_size: usize,
    pub stream_alpha: f64,
    pub look_index: u64,
    pub per_look_alpha: f64,
    pub alpha_spent_through_look: f64,
    pub sample_count: usize,
    pub range_min: f64,
    pub range_max: f64,
    pub threshold: f64,
    pub estimate: f64,
    pub half_width: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub decision: AnytimeDecision,
    pub samples_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnytimeBoundedMeanError {
    pub field: String,
    pub message: String,
}

impl AnytimeBoundedMeanError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for AnytimeBoundedMeanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for AnytimeBoundedMeanError {}

/// Compute an anytime-valid confidence receipt for bounded, independent replicates.
///
/// Hoeffding's two-sided bound at look t uses
/// radius = range * sqrt(ln(2 / alpha_t) / (2 * n)).
/// The allocations alpha_t = stream_alpha / (t * (t + 1)) sum to at most
/// stream_alpha over all looks. Bonferroni allocation across family_size
/// streams bounds the total family-wise error by family_alpha.
///
/// This is deliberately conservative. The receipt reports INCONCLUSIVE when its
/// interval crosses the decision threshold; callers must not convert that state into
/// a promotion, pruning, or resource-spend authorization.
pub fn evaluate_anytime_bounded_mean(
    replicate_values: &[f64],
    config: &AnytimeBoundedMeanConfig,
) -> Result<AnytimeBoundedMeanReceipt, AnytimeBoundedMeanError> {
    validate_config(config)?;
    if !(MIN_REPLICATES..=MAX_REPLICATES).contains(&replicate_values.len()) {
        return Err(AnytimeBoundedMeanError::new(
            "replicate_values",
            format!("sample count must be in {MIN_REPLICATES}..={MAX_REPLICATES}"),
        ));
    }

    for (index, value) in replicate_values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(AnytimeBoundedMeanError::new(
                format!("replicate_values[{index}]"),
                "replicate value must be finite",
            ));
        }
        if value < config.range_min || value > config.range_max {
            return Err(AnytimeBoundedMeanError::new(
                format!("replicate_values[{index}]"),
                "replicate value lies outside the declared support",
            ));
        }
    }

    let stream_alpha = config.family_alpha / config.family_size as f64;
    let look = config.look_index as f64;
    let per_look_alpha = stream_alpha / look / (look + 1.0);
    if !stream_alpha.is_finite() || stream_alpha <= 0.0
        || !per_look_alpha.is_finite()
        || per_look_alpha <= 0.0
    {
        return Err(AnytimeBoundedMeanError::new(
            "alpha_allocation",
            "alpha allocation underflowed or is not representable",
        ));
    }

    let sample_count = replicate_values.len();
    let divisor = sample_count as f64;
    // Divide each term before summation to avoid overflow for large finite support values.
    let estimate = replicate_values
        .iter()
        .map(|value| *value / divisor)
        .sum::<f64>()
        .clamp(config.range_min, config.range_max);
    let support_width = config.range_max - config.range_min;
    let concentration = ((2.0f64).ln() - per_look_alpha.ln())
        / (2.0 * sample_count as f64);
    let multiplier = concentration.sqrt();
    // The interval cannot be narrower than a point estimate if the Hoeffding radius
    // exceeds the entire support, so cap the radius before multiplication.
    let half_width = if multiplier >= 1.0 {
        support_width
    } else {
        support_width * multiplier
    };
    let lower_bound = (estimate - half_width).max(config.range_min);
    let upper_bound = (estimate + half_width).min(config.range_max);
    let decision = if lower_bound >= config.threshold {
        AnytimeDecision::AtOrAboveThreshold
    } else if upper_bound < config.threshold {
        AnytimeDecision::BelowThreshold
    } else {
        AnytimeDecision::Inconclusive
    };

    let samples_sha256 = compute_samples_sha256(replicate_values);
    let mut receipt = AnytimeBoundedMeanReceipt {
        schema: "symthaea.nas.anytime-bounded-mean".to_owned(),
        version: "0.1.0".to_owned(),
        method: "hoeffding-alpha-spending-bonferroni-v1".to_owned(),
        family_id: config.family_id.clone(),
        stream_id: config.stream_id.clone(),
        family_alpha: config.family_alpha,
        family_size: config.family_size,
        stream_alpha,
        look_index: config.look_index,
        per_look_alpha,
        alpha_spent_through_look: stream_alpha * (look / (look + 1.0)),
        sample_count,
        range_min: config.range_min,
        range_max: config.range_max,
        threshold: config.threshold,
        estimate,
        half_width,
        lower_bound,
        upper_bound,
        decision,
        samples_sha256,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = compute_receipt_sha256(&receipt);
    Ok(receipt)
}

/// Specialization for randomized-QMC hypervolume replicates in [0, box_volume].
pub fn evaluate_anytime_qmc_hypervolume(
    replicate_hypervolumes: &[f64],
    box_volume: f64,
    threshold: f64,
    mut config: AnytimeBoundedMeanConfig,
) -> Result<AnytimeBoundedMeanReceipt, AnytimeBoundedMeanError> {
    if !box_volume.is_finite() || box_volume <= 0.0 {
        return Err(AnytimeBoundedMeanError::new(
            "box_volume",
            "box volume must be finite and strictly positive",
        ));
    }
    config.range_min = 0.0;
    config.range_max = box_volume;
    config.threshold = threshold;
    evaluate_anytime_bounded_mean(replicate_hypervolumes, &config)
}

/// Specialization for paired action-value differences under common random numbers.
///
/// If each action's replicate value lies in [0, box_volume], each paired difference
/// lies in [-box_volume, +box_volume]. Positive differences favour the proposed action.
pub fn evaluate_anytime_paired_difference(
    paired_differences: &[f64],
    box_volume: f64,
    threshold: f64,
    mut config: AnytimeBoundedMeanConfig,
) -> Result<AnytimeBoundedMeanReceipt, AnytimeBoundedMeanError> {
    if !box_volume.is_finite() || box_volume <= 0.0 {
        return Err(AnytimeBoundedMeanError::new(
            "box_volume",
            "box volume must be finite and strictly positive",
        ));
    }
    config.range_min = -box_volume;
    config.range_max = box_volume;
    config.threshold = threshold;
    evaluate_anytime_bounded_mean(paired_differences, &config)
}

/// Independently recompute and compare every field of an anytime evidence receipt.
pub fn verify_anytime_bounded_mean_receipt(
    replicate_values: &[f64],
    config: &AnytimeBoundedMeanConfig,
    receipt: &AnytimeBoundedMeanReceipt,
) -> Result<(), AnytimeBoundedMeanError> {
    let expected = evaluate_anytime_bounded_mean(replicate_values, config)?;
    if &expected != receipt {
        return Err(AnytimeBoundedMeanError::new(
            "receipt",
            "receipt does not exactly match recomputed values and configuration",
        ));
    }
    if compute_receipt_sha256(receipt) != receipt.receipt_sha256 {
        return Err(AnytimeBoundedMeanError::new(
            "receipt_sha256",
            "receipt digest does not match the canonical receipt fields",
        ));
    }
    Ok(())
}

fn validate_config(
    config: &AnytimeBoundedMeanConfig,
) -> Result<(), AnytimeBoundedMeanError> {
    if config.family_id.trim().is_empty() || config.stream_id.trim().is_empty() {
        return Err(AnytimeBoundedMeanError::new(
            "identity",
            "family_id and stream_id must be non-empty",
        ));
    }
    if !config.family_alpha.is_finite()
        || config.family_alpha <= 0.0
        || config.family_alpha >= 1.0
    {
        return Err(AnytimeBoundedMeanError::new(
            "family_alpha",
            "family alpha must be finite and strictly between zero and one",
        ));
    }
    if !(1..=MAX_FAMILY_SIZE).contains(&config.family_size) {
        return Err(AnytimeBoundedMeanError::new(
            "family_size",
            format!("family size must be in 1..={MAX_FAMILY_SIZE}"),
        ));
    }
    if config.look_index == 0 || config.look_index > MAX_LOOK_INDEX {
        return Err(AnytimeBoundedMeanError::new(
            "look_index",
            format!("look index must be in 1..={MAX_LOOK_INDEX}"),
        ));
    }
    if !config.range_min.is_finite()
        || !config.range_max.is_finite()
        || config.range_max <= config.range_min
        || !(config.range_max - config.range_min).is_finite()
    {
        return Err(AnytimeBoundedMeanError::new(
            "range",
            "support bounds must be finite, ordered, and have a finite width",
        ));
    }
    if !config.threshold.is_finite()
        || config.threshold < config.range_min
        || config.threshold > config.range_max
    {
        return Err(AnytimeBoundedMeanError::new(
            "threshold",
            "threshold must be finite and lie inside the declared support",
        ));
    }
    Ok(())
}

fn compute_samples_sha256(values: &[f64]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((SAMPLE_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(SAMPLE_DOMAIN);
    hasher.update((values.len() as u64).to_be_bytes());
    for value in values {
        hasher.update(value.to_bits().to_be_bytes());
    }
    encode_sha256(hasher.finalize().as_slice())
}

fn compute_receipt_sha256(receipt: &AnytimeBoundedMeanReceipt) -> String {
    let mut hasher = Sha256::new();
    hasher.update((RECEIPT_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(RECEIPT_DOMAIN);
    hash_text(&mut hasher, &receipt.schema);
    hash_text(&mut hasher, &receipt.version);
    hash_text(&mut hasher, &receipt.method);
    hash_text(&mut hasher, &receipt.family_id);
    hash_text(&mut hasher, &receipt.stream_id);
    hasher.update(receipt.family_alpha.to_bits().to_be_bytes());
    hasher.update((receipt.family_size as u64).to_be_bytes());
    hasher.update(receipt.stream_alpha.to_bits().to_be_bytes());
    hasher.update(receipt.look_index.to_be_bytes());
    hasher.update(receipt.per_look_alpha.to_bits().to_be_bytes());
    hasher.update(receipt.alpha_spent_through_look.to_bits().to_be_bytes());
    hasher.update((receipt.sample_count as u64).to_be_bytes());
    hasher.update(receipt.range_min.to_bits().to_be_bytes());
    hasher.update(receipt.range_max.to_bits().to_be_bytes());
    hasher.update(receipt.threshold.to_bits().to_be_bytes());
    hasher.update(receipt.estimate.to_bits().to_be_bytes());
    hasher.update(receipt.half_width.to_bits().to_be_bytes());
    hasher.update(receipt.lower_bound.to_bits().to_be_bytes());
    hasher.update(receipt.upper_bound.to_bits().to_be_bytes());
    hasher.update(match receipt.decision {
        AnytimeDecision::AtOrAboveThreshold => [1u8],
        AnytimeDecision::BelowThreshold => [2u8],
        AnytimeDecision::Inconclusive => [3u8],
    });
    hash_text(&mut hasher, &receipt.samples_sha256);
    encode_sha256(hasher.finalize().as_slice())
}

fn hash_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn encode_sha256(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AnytimeBoundedMeanConfig {
        AnytimeBoundedMeanConfig {
            family_id: "nas-voc-benchmark-v1".into(),
            stream_id: "portfolio-A-vs-B".into(),
            family_alpha: 0.05,
            family_size: 4,
            look_index: 1,
            range_min: 0.0,
            range_max: 1.0,
            threshold: 0.5,
        }
    }

    #[test]
    fn interval_contains_the_empirical_mean_and_reports_inconclusive() {
        let values = [0.49, 0.50, 0.51, 0.50, 0.50, 0.49, 0.51, 0.50];
        let receipt = evaluate_anytime_bounded_mean(&values, &config()).unwrap();
        assert!((receipt.lower_bound..=receipt.upper_bound).contains(&receipt.estimate));
        assert_eq!(receipt.decision, AnytimeDecision::Inconclusive);
        assert!(receipt.half_width > 0.0);
    }

    #[test]
    fn decisively_high_mean_is_at_or_above_threshold() {
        let values = vec![0.9; 2048];
        let receipt = evaluate_anytime_bounded_mean(&values, &config()).unwrap();
        assert!(receipt.lower_bound >= 0.5);
        assert_eq!(receipt.decision, AnytimeDecision::AtOrAboveThreshold);
    }

    #[test]
    fn decisively_low_mean_is_below_threshold() {
        let values = vec![0.1; 2048];
        let receipt = evaluate_anytime_bounded_mean(&values, &config()).unwrap();
        assert!(receipt.upper_bound < 0.5);
        assert_eq!(receipt.decision, AnytimeDecision::BelowThreshold);
    }

    #[test]
    fn alpha_spending_is_bounded_by_stream_and_family_alpha() {
        let mut cfg = config();
        cfg.family_size = 4;
        for look in 1..=10_000 {
            cfg.look_index = look;
            let receipt = evaluate_anytime_bounded_mean(&[0.5], &cfg).unwrap();
            assert!(receipt.alpha_spent_through_look <= receipt.stream_alpha + 1e-15);
            assert!(receipt.stream_alpha <= receipt.family_alpha / 4.0 + 1e-15);
        }
        let final_receipt = evaluate_anytime_bounded_mean(
            &[0.5],
            &AnytimeBoundedMeanConfig {
                look_index: 10_000,
                ..config()
            },
        )
        .unwrap();
        assert!(final_receipt.alpha_spent_through_look < final_receipt.stream_alpha);
    }

    #[test]
    fn later_looks_receive_less_alpha_and_wider_bounds() {
        let values = vec![0.6; 128];
        let first = evaluate_anytime_bounded_mean(&values, &config()).unwrap();
        let later = evaluate_anytime_bounded_mean(
            &values,
            &AnytimeBoundedMeanConfig {
                look_index: 5,
                ..config()
            },
        )
        .unwrap();
        assert!(later.per_look_alpha < first.per_look_alpha);
        assert!(later.half_width > first.half_width);
    }

    #[test]
    fn bonferroni_family_size_reduces_stream_alpha_and_widens_bounds() {
        let values = vec![0.6; 128];
        let one = evaluate_anytime_bounded_mean(
            &values,
            &AnytimeBoundedMeanConfig {
                family_size: 1,
                ..config()
            },
        )
        .unwrap();
        let many = evaluate_anytime_bounded_mean(&values, &config()).unwrap();
        assert!(many.stream_alpha < one.stream_alpha);
        assert!(many.half_width > one.half_width);
    }

    #[test]
    fn receipt_verifies_and_tampering_fails_closed() {
        let values = vec![0.8, 0.7, 0.9, 0.6];
        let cfg = config();
        let mut receipt = evaluate_anytime_bounded_mean(&values, &cfg).unwrap();
        verify_anytime_bounded_mean_receipt(&values, &cfg, &receipt).unwrap();
        receipt.upper_bound = 1.0;
        assert!(verify_anytime_bounded_mean_receipt(&values, &cfg, &receipt).is_err());
    }

    #[test]
    fn sample_digest_binds_ordered_replicates() {
        let cfg = config();
        let first = evaluate_anytime_bounded_mean(&[0.2, 0.8], &cfg).unwrap();
        let second = evaluate_anytime_bounded_mean(&[0.8, 0.2], &cfg).unwrap();
        assert_ne!(first.samples_sha256, second.samples_sha256);
        assert_eq!(first.estimate, second.estimate);
    }

    #[test]
    fn qmc_hypervolume_specialization_uses_zero_to_box_support() {
        let mut cfg = config();
        cfg.range_min = -999.0;
        cfg.range_max = 999.0;
        let receipt = evaluate_anytime_qmc_hypervolume(&[0.04, 0.06, 0.05], 0.1, 0.045, cfg).unwrap();
        assert_eq!(receipt.range_min, 0.0);
        assert_eq!(receipt.range_max, 0.1);
        assert_eq!(receipt.threshold, 0.045);
    }

    #[test]
    fn paired_difference_specialization_uses_symmetric_box_support() {
        let receipt =
            evaluate_anytime_paired_difference(&[-0.02, 0.01, 0.03], 0.1, 0.0, config()).unwrap();
        assert_eq!(receipt.range_min, -0.1);
        assert_eq!(receipt.range_max, 0.1);
        assert_eq!(receipt.decision, AnytimeDecision::Inconclusive);
    }

    #[test]
    fn rejects_values_outside_the_declared_support() {
        assert!(evaluate_anytime_bounded_mean(&[1.01], &config()).is_err());
        assert!(evaluate_anytime_bounded_mean(&[-0.01], &config()).is_err());
    }

    #[test]
    fn rejects_non_finite_values_and_configuration() {
        assert!(evaluate_anytime_bounded_mean(&[f64::NAN], &config()).is_err());
        assert!(evaluate_anytime_bounded_mean(&[f64::INFINITY], &config()).is_err());
        assert!(evaluate_anytime_bounded_mean(
            &[0.5],
            &AnytimeBoundedMeanConfig {
                family_alpha: 1.0,
                ..config()
            }
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_range_threshold_and_look() {
        assert!(evaluate_anytime_bounded_mean(
            &[0.5],
            &AnytimeBoundedMeanConfig {
                range_min: 1.0,
                range_max: 1.0,
                ..config()
            }
        )
        .is_err());
        assert!(evaluate_anytime_bounded_mean(
            &[0.5],
            &AnytimeBoundedMeanConfig {
                threshold: 1.1,
                ..config()
            }
        )
        .is_err());
        assert!(evaluate_anytime_bounded_mean(
            &[0.5],
            &AnytimeBoundedMeanConfig {
                look_index: 0,
                ..config()
            }
        )
        .is_err());
    }

    #[test]
    fn rejects_empty_or_oversized_families() {
        assert!(evaluate_anytime_bounded_mean(&[], &config()).is_err());
        assert!(evaluate_anytime_bounded_mean(
            &[0.5],
            &AnytimeBoundedMeanConfig {
                family_size: 0,
                ..config()
            }
        )
        .is_err());
    }

    #[test]
    fn one_sample_remains_valid_but_usually_inconclusive() {
        let receipt = evaluate_anytime_bounded_mean(&[0.9], &config()).unwrap();
        assert_eq!(receipt.decision, AnytimeDecision::Inconclusive);
        assert_eq!(receipt.lower_bound, 0.0);
        assert_eq!(receipt.upper_bound, 1.0);
    }

    #[test]
    fn receipt_digest_binds_the_decision_and_sample_digest() {
        let receipt = evaluate_anytime_bounded_mean(&[0.9; 2048], &config()).unwrap();
        let expected = compute_receipt_sha256(&receipt);
        assert_eq!(receipt.receipt_sha256, expected);
        let mut changed = receipt.clone();
        changed.decision = AnytimeDecision::BelowThreshold;
        assert_ne!(compute_receipt_sha256(&changed), receipt.receipt_sha256);
    }
}
