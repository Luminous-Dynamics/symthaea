//! FTD-style empirical reference p-values, weighted Cauchy fusion, and BH.
//!
//! This module implements a reproducible statistical comparison layer. It does
//! not reproduce model likelihood features (PPL / Min-K / Min-K++) and does not
//! claim the paper's theorem for arbitrary inputs. In particular, p-value
//! super-uniformity under the contaminated null and the cross-hypothesis
//! dependence conditions must be independently qualified.
//!
//! The ACL paper's printed Eq. (4) shows a lower-tail reference probability,
//! while its stated score direction (low = more contaminated) and official
//! implementation use the upper tail for the experiments. Both directions are
//! explicit here; Greater is the official code direction for those experiments.

use crate::hmac_sha256;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const FTD_REPORT_SCHEMA: &str = "symthaea.contamination_panel_ftd_report.v1";
pub const MIN_SEED_BYTES: usize = 32;
const CAUCHY_EPS: f64 = 1.0e-15;
const PI: f64 = std::f64::consts::PI;
const TWO_POW_52: f64 = 4_503_599_627_370_496.0;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceTailV1 {
    /// p = P(reference_score >= observed_score); small p means unusually high
    /// observed scores relative to the contaminated reference.
    Greater,
    /// p = P(reference_score <= observed_score); small p means unusually low
    /// observed scores relative to the reference.
    Less,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FtdWeightPolicyV1 {
    /// The paper's Eq. (7): normalized per-detector BH rejection counts.
    PaperNormalizedRejectionCounts,
    /// The official repository applies temperature-scaled softmax sharpening
    /// after normalizing rejection counts. Temperature is in thousandths.
    OfficialRepositorySoftmaxSharpened,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreObservationV1 {
    pub pair_id: String,
    pub score: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairPValueV1 {
    pub pair_id: String,
    pub p_value: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FeaturePValuesV1 {
    pub feature_id: String,
    pub direction: ReferenceTailV1,
    pub reference_count: u64,
    pub reference_scores_sha256: String,
    pub p_values: Vec<PairPValueV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FtdFeatureWeightV1 {
    pub feature_id: String,
    pub direction: ReferenceTailV1,
    pub reference_count: u64,
    pub reference_scores_sha256: String,
    pub single_feature_bh_rejections: u64,
    pub weight: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CombinedPValueV1 {
    pub pair_id: String,
    pub p_value: f64,
    pub retained_as_clean: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FtdFusionReportV1 {
    pub schema: String,
    pub alpha: f64,
    pub weight_policy: FtdWeightPolicyV1,
    pub softmax_temperature_milli: Option<u16>,
    pub feature_count: u64,
    pub hypothesis_count: u64,
    pub bh_rejection_count: u64,
    pub retained_as_clean_count: u64,
    pub feature_weights: Vec<FtdFeatureWeightV1>,
    pub combined_p_values: Vec<CombinedPValueV1>,
    pub null_reference_validity: String,
    pub cross_hypothesis_dependence: String,
    pub nominal_fdr_guarantee: String,
    pub claim_boundary: String,
}

fn validate_probability(value: f64, label: &str) -> Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!("{label} must be finite and in [0, 1]"));
    }
    Ok(())
}

fn deterministic_uniform(seed: &[u8], feature_id: &str, pair_id: &str) -> f64 {
    let mut payload = Vec::with_capacity(feature_id.len() + pair_id.len() + 16);
    payload.extend_from_slice(&(feature_id.len() as u64).to_be_bytes());
    payload.extend_from_slice(feature_id.as_bytes());
    payload.extend_from_slice(&(pair_id.len() as u64).to_be_bytes());
    payload.extend_from_slice(pair_id.as_bytes());
    let digest = hmac_sha256(seed, &payload);
    let raw = u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ]);
    let mantissa = raw >> 12;
    (mantissa as f64 + 0.5) / TWO_POW_52
}

/// Compute deterministic, randomized-tie empirical p-values against a
/// caller-supplied contaminated-reference score distribution.
///
/// For Greater, lower p-values correspond to observed scores unusually high
/// relative to the reference. This is the direction used by the official FTD
/// repository for its listed experiments. The seed makes tie smoothing
/// reproducible; it is not a cryptographic custody or provenance receipt.
pub fn empirical_reference_p_values(
    feature_id: &str,
    direction: ReferenceTailV1,
    reference_scores: &[f64],
    test_scores: &[ScoreObservationV1],
    seed: &[u8],
) -> Result<FeaturePValuesV1, String> {
    if feature_id.trim().is_empty() || feature_id.len() > 128 {
        return Err("feature_id must be nonempty and <=128 bytes".to_owned());
    }
    if seed.len() < MIN_SEED_BYTES {
        return Err(format!(
            "p-value seed must contain at least {MIN_SEED_BYTES} bytes"
        ));
    }
    if reference_scores.is_empty() {
        return Err("contaminated-reference score sample must not be empty".to_owned());
    }
    if test_scores.is_empty() {
        return Err("test score set must not be empty".to_owned());
    }
    let mut sorted_reference = reference_scores.to_vec();
    if sorted_reference.iter().any(|score| !score.is_finite()) {
        return Err("reference scores must all be finite".to_owned());
    }
    sorted_reference.sort_by(f64::total_cmp);

    let mut seen_ids = BTreeSet::new();
    let mut p_values = Vec::with_capacity(test_scores.len());
    for observation in test_scores {
        if observation.pair_id.trim().is_empty() || observation.pair_id.len() > 256 {
            return Err("test score pair_id must be nonempty and <=256 bytes".to_owned());
        }
        if !seen_ids.insert(observation.pair_id.as_str()) {
            return Err(format!(
                "duplicate test score pair_id {}",
                observation.pair_id
            ));
        }
        if !observation.score.is_finite() {
            return Err(format!(
                "test score for {} must be finite",
                observation.pair_id
            ));
        }

        let count = match direction {
            ReferenceTailV1::Greater => {
                let first_ge = sorted_reference.partition_point(|value| *value < observation.score);
                sorted_reference.len() - first_ge
            }
            ReferenceTailV1::Less => {
                sorted_reference.partition_point(|value| *value <= observation.score)
            }
        };
        let tie_uniform = deterministic_uniform(seed, feature_id, &observation.pair_id);
        let p_value = (count as f64 + tie_uniform) / (sorted_reference.len() as f64 + 1.0);
        validate_probability(p_value, "empirical p-value")?;
        if p_value == 0.0 || p_value == 1.0 {
            return Err("smoothed empirical p-value unexpectedly reached a boundary".to_owned());
        }
        p_values.push(PairPValueV1 {
            pair_id: observation.pair_id.clone(),
            p_value,
        });
    }

    let reference_bytes = serde_json::to_vec(&sorted_reference)
        .map_err(|error| format!("cannot serialize reference scores: {error}"))?;
    Ok(FeaturePValuesV1 {
        feature_id: feature_id.to_owned(),
        direction,
        reference_count: u64::try_from(sorted_reference.len())
            .map_err(|_| "reference count exceeds u64")?,
        reference_scores_sha256: crate::hash_domain(b"ftd-reference-scores.v1", &reference_bytes),
        p_values,
    })
}

fn bh_rejections(p_values: &[f64], alpha: f64) -> Result<Vec<bool>, String> {
    if p_values.is_empty() {
        return Err("Benjamini-Hochberg input must not be empty".to_owned());
    }
    validate_probability(alpha, "BH alpha")?;
    if alpha <= 0.0 {
        return Err("BH alpha must be in (0, 1)".to_owned());
    }
    let mut indexed = Vec::with_capacity(p_values.len());
    for (index, value) in p_values.iter().copied().enumerate() {
        validate_probability(value, "BH p-value")?;
        indexed.push((value, index));
    }
    indexed.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    let n = p_values.len() as f64;
    let mut largest_valid_rank = 0_usize;
    for (zero_index, (p_value, _)) in indexed.iter().enumerate() {
        let rank = (zero_index + 1) as f64;
        if *p_value <= alpha * rank / n {
            largest_valid_rank = zero_index + 1;
        }
    }

    let mut rejected = vec![false; p_values.len()];
    for (_, original_index) in indexed.into_iter().take(largest_valid_rank) {
        rejected[original_index] = true;
    }
    Ok(rejected)
}

fn normalize_rejection_counts(counts: &[u64]) -> Result<Vec<f64>, String> {
    if counts.is_empty() {
        return Err("at least one detector feature is required".to_owned());
    }
    let total = counts.iter().try_fold(0_u64, |sum, count| {
        sum.checked_add(*count)
            .ok_or_else(|| "total individual BH rejection count overflows u64".to_owned())
    })?;
    if total == 0 {
        return Ok(vec![1.0 / counts.len() as f64; counts.len()]);
    }
    Ok(counts
        .iter()
        .map(|count| *count as f64 / total as f64)
        .collect())
}

fn sharpen_weights(weights: &[f64], temperature_milli: u16) -> Result<Vec<f64>, String> {
    if temperature_milli == 0 || temperature_milli > 10_000 {
        return Err("softmax temperature must be in (0, 10]".to_owned());
    }
    if weights.is_empty()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("softmax weights must be a nonempty finite nonnegative vector".to_owned());
    }
    let temperature = f64::from(temperature_milli) / 1_000.0;
    let logits: Vec<f64> = weights
        .iter()
        .map(|weight| (weight + 1.0e-12).ln() / temperature)
        .collect();
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let sum: f64 = exponentials.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("softmax normalization is not finite and positive".to_owned());
    }
    Ok(exponentials.iter().map(|value| value / sum).collect())
}

fn weighted_cauchy_p_value(p_values: &[f64], weights: &[f64]) -> Result<f64, String> {
    if p_values.is_empty() || p_values.len() != weights.len() {
        return Err("Cauchy p-values and weights must have the same nonzero length".to_owned());
    }
    let mut weight_sum = 0.0;
    let mut statistic = 0.0;
    for (index, (p_value, weight)) in p_values.iter().zip(weights).enumerate() {
        validate_probability(*p_value, "Cauchy input p-value")?;
        if !weight.is_finite() || *weight < 0.0 {
            return Err(format!(
                "Cauchy weight {index} must be finite and nonnegative"
            ));
        }
        weight_sum += *weight;
        let clipped = (*p_value).clamp(CAUCHY_EPS, 1.0 - CAUCHY_EPS);
        statistic += *weight * ((0.5 - clipped) * PI).tan();
    }
    if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 1.0e-9 {
        return Err("Cauchy weights must sum to one within 1e-9".to_owned());
    }
    if !statistic.is_finite() {
        return Err("Cauchy statistic is non-finite".to_owned());
    }
    let combined = 0.5 - statistic.atan() / PI;
    Ok(combined.clamp(CAUCHY_EPS, 1.0 - CAUCHY_EPS))
}

/// Combine detector p-values with FTD-style adaptive rejection-count weights,
/// weighted Cauchy aggregation and a final BH step.
///
/// The output is an auditable implementation of the calculation. It deliberately
/// does not claim FDR control: that requires calibrated super-uniform null
/// p-values, valid feature construction, and a justified cross-hypothesis
/// dependence argument. The paper's asymptotic theorem is not implied by
/// computing these formulas on an arbitrary benchmark.
pub fn ftd_weighted_cauchy_bh(
    features: &[FeaturePValuesV1],
    alpha: f64,
    weight_policy: FtdWeightPolicyV1,
    softmax_temperature_milli: Option<u16>,
) -> Result<FtdFusionReportV1, String> {
    validate_probability(alpha, "FDR alpha")?;
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err("FDR alpha must be in (0, 1)".to_owned());
    }
    if features.is_empty() {
        return Err("FTD fusion requires at least one detector feature".to_owned());
    }
    match (weight_policy, softmax_temperature_milli) {
        (FtdWeightPolicyV1::PaperNormalizedRejectionCounts, None) => {}
        (FtdWeightPolicyV1::PaperNormalizedRejectionCounts, Some(_)) => {
            return Err("paper weight policy must not specify a softmax temperature".to_owned());
        }
        (FtdWeightPolicyV1::OfficialRepositorySoftmaxSharpened, Some(value))
            if value > 0 && value <= 10_000 => {}
        (FtdWeightPolicyV1::OfficialRepositorySoftmaxSharpened, _) => {
            return Err(
                "official repository weight policy requires temperature in (0, 10]".to_owned(),
            );
        }
    }

    let mut feature_ids = BTreeSet::new();
    let first_feature = &features[0];
    if first_feature.p_values.is_empty() {
        return Err("feature p-value list must not be empty".to_owned());
    }
    let pair_order: Vec<String> = first_feature
        .p_values
        .iter()
        .map(|row| row.pair_id.clone())
        .collect();
    let pair_id_set: BTreeSet<&str> = pair_order.iter().map(String::as_str).collect();
    if pair_id_set.len() != pair_order.len() {
        return Err("first feature repeats a pair ID".to_owned());
    }

    let mut pvalue_maps = Vec::<BTreeMap<String, f64>>::with_capacity(features.len());
    let mut feature_rejection_counts = Vec::with_capacity(features.len());
    for feature in features {
        if feature.feature_id.trim().is_empty()
            || feature.feature_id.len() > 128
            || !feature_ids.insert(feature.feature_id.as_str())
        {
            return Err("feature IDs must be unique, nonempty and <=128 bytes".to_owned());
        }
        if feature.reference_count == 0 || !crate::valid_digest(&feature.reference_scores_sha256) {
            return Err(format!(
                "feature {} has invalid reference metadata",
                feature.feature_id
            ));
        }
        if feature.p_values.len() != pair_order.len() {
            return Err(format!(
                "feature {} does not cover the same number of hypotheses",
                feature.feature_id
            ));
        }
        let mut p_map = BTreeMap::new();
        let mut values = Vec::with_capacity(feature.p_values.len());
        for row in &feature.p_values {
            if !pair_id_set.contains(row.pair_id.as_str()) {
                return Err(format!(
                    "feature {} contains unknown pair ID {}",
                    feature.feature_id, row.pair_id
                ));
            }
            validate_probability(row.p_value, "feature p-value")?;
            if p_map.insert(row.pair_id.clone(), row.p_value).is_some() {
                return Err(format!(
                    "feature {} repeats pair ID {}",
                    feature.feature_id, row.pair_id
                ));
            }
            values.push(row.p_value);
        }
        if p_map.len() != pair_order.len() {
            return Err(format!(
                "feature {} has missing pair IDs",
                feature.feature_id
            ));
        }
        let rejected = bh_rejections(&values, alpha)?;
        feature_rejection_counts.push(
            u64::try_from(rejected.iter().filter(|value| **value).count())
                .map_err(|_| "single-feature BH rejection count overflows u64")?,
        );
        pvalue_maps.push(p_map);
    }

    let normalized_weights = normalize_rejection_counts(&feature_rejection_counts)?;
    let weights = match weight_policy {
        FtdWeightPolicyV1::PaperNormalizedRejectionCounts => normalized_weights,
        FtdWeightPolicyV1::OfficialRepositorySoftmaxSharpened => {
            sharpen_weights(&normalized_weights, softmax_temperature_milli.unwrap())?
        }
    };

    let mut combined_p_values = Vec::with_capacity(pair_order.len());
    let mut combined_values = Vec::with_capacity(pair_order.len());
    for pair_id in &pair_order {
        let row_p_values: Vec<f64> = pvalue_maps
            .iter()
            .map(|feature| {
                feature
                    .get(pair_id)
                    .copied()
                    .ok_or_else(|| format!("pair {pair_id} missing from a feature p-value map"))
            })
            .collect::<Result<_, _>>()?;
        let combined = weighted_cauchy_p_value(&row_p_values, &weights)?;
        combined_values.push(combined);
        combined_p_values.push((pair_id.clone(), combined));
    }

    let final_rejections = bh_rejections(&combined_values, alpha)?;
    let bh_rejection_count = u64::try_from(final_rejections.iter().filter(|value| **value).count())
        .map_err(|_| "final BH rejection count overflows u64")?;
    let combined_p_values: Vec<CombinedPValueV1> = combined_p_values
        .into_iter()
        .zip(final_rejections)
        .map(|((pair_id, p_value), retained_as_clean)| CombinedPValueV1 {
            pair_id,
            p_value,
            retained_as_clean,
        })
        .collect();

    let mut feature_weights = Vec::with_capacity(features.len());
    for ((feature, rejection_count), weight) in
        features.iter().zip(feature_rejection_counts).zip(weights)
    {
        feature_weights.push(FtdFeatureWeightV1 {
            feature_id: feature.feature_id.clone(),
            direction: feature.direction,
            reference_count: feature.reference_count,
            reference_scores_sha256: feature.reference_scores_sha256.clone(),
            single_feature_bh_rejections: rejection_count,
            weight,
        });
    }

    Ok(FtdFusionReportV1 {
        schema: FTD_REPORT_SCHEMA.to_owned(),
        alpha,
        weight_policy,
        softmax_temperature_milli,
        feature_count: u64::try_from(features.len()).map_err(|_| "feature count overflows u64")?,
        hypothesis_count: u64::try_from(pair_order.len()).map_err(|_| "hypothesis count overflows u64")?,
        bh_rejection_count,
        retained_as_clean_count: bh_rejection_count,
        feature_weights,
        combined_p_values,
        null_reference_validity: "not_assessed_by_this_fusion_kernel".to_owned(),
        cross_hypothesis_dependence: "not_assessed_by_this_fusion_kernel".to_owned(),
        nominal_fdr_guarantee: "not_claimed_without_independent_p_value_and_dependence_qualification".to_owned(),
        claim_boundary: "deterministic_FTD_style_calculation_only_not_a_verified_FDR_guarantee_or_training_exposure_oracle".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash_domain;

    fn feature(feature_id: &str, pair_values: &[(&str, f64)]) -> FeaturePValuesV1 {
        FeaturePValuesV1 {
            feature_id: feature_id.to_owned(),
            direction: ReferenceTailV1::Greater,
            reference_count: 100,
            reference_scores_sha256: hash_domain(b"fixture-reference", feature_id.as_bytes()),
            p_values: pair_values
                .iter()
                .map(|(pair_id, p_value)| PairPValueV1 {
                    pair_id: (*pair_id).to_owned(),
                    p_value: *p_value,
                })
                .collect(),
        }
    }

    #[test]
    fn official_greater_tail_makes_high_scores_more_clean_like() {
        let reference = [1.0, 2.0, 3.0, 4.0];
        let observations = vec![
            ScoreObservationV1 {
                pair_id: "high".to_owned(),
                score: 9.0,
            },
            ScoreObservationV1 {
                pair_id: "low".to_owned(),
                score: 0.0,
            },
        ];
        let values = empirical_reference_p_values(
            "ppl",
            ReferenceTailV1::Greater,
            &reference,
            &observations,
            &[0x42; 32],
        )
        .unwrap();
        assert!(values.p_values[0].p_value < values.p_values[1].p_value);
        assert!(values.p_values[0].p_value < 0.2);
        assert!(values.p_values[1].p_value > 0.8);
    }

    #[test]
    fn lower_tail_is_explicit_and_reverses_the_score_order() {
        let reference = [1.0, 2.0, 3.0, 4.0];
        let observations = vec![
            ScoreObservationV1 {
                pair_id: "high".to_owned(),
                score: 9.0,
            },
            ScoreObservationV1 {
                pair_id: "low".to_owned(),
                score: 0.0,
            },
        ];
        let values = empirical_reference_p_values(
            "ppl",
            ReferenceTailV1::Less,
            &reference,
            &observations,
            &[0x42; 32],
        )
        .unwrap();
        assert!(values.p_values[1].p_value < values.p_values[0].p_value);
    }

    #[test]
    fn randomized_tie_smoothing_is_deterministic_and_strictly_interior() {
        let reference = [1.0, 1.0, 1.0, 1.0];
        let observations = vec![
            ScoreObservationV1 {
                pair_id: "tie-a".to_owned(),
                score: 1.0,
            },
            ScoreObservationV1 {
                pair_id: "tie-b".to_owned(),
                score: 1.0,
            },
        ];
        let left = empirical_reference_p_values(
            "feature",
            ReferenceTailV1::Greater,
            &reference,
            &observations,
            &[0x33; 32],
        )
        .unwrap();
        let right = empirical_reference_p_values(
            "feature",
            ReferenceTailV1::Greater,
            &reference,
            &observations,
            &[0x33; 32],
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&left).unwrap(),
            serde_json::to_vec(&right).unwrap()
        );
        assert!(
            left.p_values
                .iter()
                .all(|row| row.p_value > 0.0 && row.p_value < 1.0)
        );
    }

    #[test]
    fn empirical_reference_rejects_duplicate_ids_nonfinite_and_empty_reference() {
        let duplicate = vec![
            ScoreObservationV1 {
                pair_id: "x".to_owned(),
                score: 1.0,
            },
            ScoreObservationV1 {
                pair_id: "x".to_owned(),
                score: 2.0,
            },
        ];
        assert!(
            empirical_reference_p_values(
                "f",
                ReferenceTailV1::Greater,
                &[1.0],
                &duplicate,
                &[0x44; 32]
            )
            .is_err()
        );
        assert!(
            empirical_reference_p_values(
                "f",
                ReferenceTailV1::Greater,
                &[],
                &duplicate[..1],
                &[0x44; 32]
            )
            .is_err()
        );
        assert!(
            empirical_reference_p_values(
                "f",
                ReferenceTailV1::Greater,
                &[f64::NAN],
                &duplicate[..1],
                &[0x44; 32]
            )
            .is_err()
        );
    }

    #[test]
    fn bh_selects_largest_valid_rank_and_is_order_invariant() {
        let values = [0.001, 0.01, 0.03, 0.9];
        let result = bh_rejections(&values, 0.05).unwrap();
        assert_eq!(result, vec![true, true, true, false]);

        let shuffled = [0.9, 0.03, 0.001, 0.01];
        let shuffled_result = bh_rejections(&shuffled, 0.05).unwrap();
        assert_eq!(shuffled_result, vec![false, true, true, true]);
    }

    #[test]
    fn paper_normalized_weights_and_official_softmax_variant_are_distinct() {
        let p1 = feature(
            "strong",
            &[("a", 0.0001), ("b", 0.001), ("c", 0.01), ("d", 0.8)],
        );
        let p2 = feature("weak", &[("a", 0.4), ("b", 0.5), ("c", 0.6), ("d", 0.7)]);
        let paper = ftd_weighted_cauchy_bh(
            &[p1.clone(), p2.clone()],
            0.05,
            FtdWeightPolicyV1::PaperNormalizedRejectionCounts,
            None,
        )
        .unwrap();
        assert_eq!(paper.feature_weights[0].weight, 1.0);
        assert_eq!(paper.feature_weights[1].weight, 0.0);
        assert_eq!(paper.feature_weights[0].single_feature_bh_rejections, 3);
        assert_eq!(paper.feature_weights[1].single_feature_bh_rejections, 0);

        let official = ftd_weighted_cauchy_bh(
            &[p1, p2],
            0.05,
            FtdWeightPolicyV1::OfficialRepositorySoftmaxSharpened,
            Some(500),
        )
        .unwrap();
        assert!(official.feature_weights[0].weight > 0.999_999);
        assert!(official.feature_weights[1].weight < 0.000_001);
        assert_eq!(
            official.nominal_fdr_guarantee,
            "not_claimed_without_independent_p_value_and_dependence_qualification"
        );
    }

    #[test]
    fn cauchy_and_bh_pipeline_is_deterministic_and_keeps_claims_bounded() {
        let p1 = feature("a", &[("x", 0.001), ("y", 0.02), ("z", 0.9)]);
        let p2 = feature("b", &[("x", 0.01), ("y", 0.03), ("z", 0.8)]);
        let first = ftd_weighted_cauchy_bh(
            &[p1.clone(), p2.clone()],
            0.05,
            FtdWeightPolicyV1::PaperNormalizedRejectionCounts,
            None,
        )
        .unwrap();
        let second = ftd_weighted_cauchy_bh(
            &[p1, p2],
            0.05,
            FtdWeightPolicyV1::PaperNormalizedRejectionCounts,
            None,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        assert_eq!(first.hypothesis_count, 3);
        assert_eq!(first.combined_p_values.len(), 3);
        assert!(
            first
                .combined_p_values
                .iter()
                .all(|row| row.p_value > 0.0 && row.p_value < 1.0)
        );
        assert_eq!(
            first.cross_hypothesis_dependence,
            "not_assessed_by_this_fusion_kernel"
        );
    }

    #[test]
    fn incomplete_or_misaligned_feature_hypotheses_fail_closed() {
        let p1 = feature("a", &[("x", 0.01), ("y", 0.02)]);
        let p2 = feature("b", &[("x", 0.03), ("other", 0.04)]);
        assert!(
            ftd_weighted_cauchy_bh(
                &[p1, p2],
                0.05,
                FtdWeightPolicyV1::PaperNormalizedRejectionCounts,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn p_values_and_fdr_alpha_are_range_checked() {
        let mut p1 = feature("a", &[("x", 0.01)]);
        p1.p_values[0].p_value = f64::NAN;
        assert!(
            ftd_weighted_cauchy_bh(
                &[p1],
                0.05,
                FtdWeightPolicyV1::PaperNormalizedRejectionCounts,
                None
            )
            .is_err()
        );
        assert!(
            ftd_weighted_cauchy_bh(
                &[feature("a", &[("x", 0.01)])],
                1.0,
                FtdWeightPolicyV1::PaperNormalizedRejectionCounts,
                None
            )
            .is_err()
        );
    }
}
