// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Finite-sample audit of conservative retrieval-utility estimates.
//!
//! The VOI gate is only safe to use when its predicted lower bounds are
//! empirically conservative on held-out data. This module audits that property
//! on a distinct, untouched audit split and issues an opaque capability only
//! when a one-sided Hoeffding lower confidence bound on coverage meets the
//! pre-registered target.
//!
//! For each paired audit query, coverage means:
//!     realized_net_utility >= predicted_net_utility_lower_bound.
//!
//! The coverage indicators are Bernoulli. Under an i.i.d. / exchangeable audit
//! sample, Hoeffding's inequality gives a finite-sample lower confidence bound.
//! A Bonferroni correction controls family-wise error across a predeclared
//! number of audit slices/settings. The estimator must be fixed before the
//! audit split is opened; repeated adaptive tuning invalidates the guarantee.
//!
//! Brier score and ECE for the sign-of-utility probability are reported as
//! diagnostics only. They do not authorize the calibration capability because
//! neither metric establishes lower-bound coverage on its own.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const RETRIEVAL_CALIBRATION_POLICY_VERSION: &str =
    "BROCA-RETRIEVAL-CALIBRATION-V1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalCalibrationObservation {
    pub query_id: String,
    /// Estimator's probability that the marginal retrieval utility is positive.
    pub predicted_positive_probability: f64,
    /// Conservative lower bound supplied to the VOI gate.
    pub predicted_net_value_lcb_micro: i64,
    /// Paired realized net utility after the retrieval and reader outcome.
    pub observed_net_value_micro: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalCalibrationAuditRequest {
    pub estimator_id: String,
    pub estimator_version: String,
    pub estimator_sha256: String,
    pub feature_schema_sha256: String,
    pub utility_schema_sha256: String,
    pub training_split_sha256: String,
    pub calibration_split_sha256: String,
    /// Must be a separate, untouched split not used for training/calibration.
    pub audit_split_sha256: String,
    /// Family-wise error budget shared across pre-registered slices/settings.
    pub family_alpha: f64,
    pub family_size: u32,
    pub target_lower_bound_coverage: f64,
    pub minimum_audit_observations: usize,
    /// Equal-width bins used only for diagnostic ECE.
    pub ece_bins: usize,
    pub observations: Vec<RetrievalCalibrationObservation>,
}

impl RetrievalCalibrationAuditRequest {
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("estimator_id", self.estimator_id.as_str()),
            ("estimator_version", self.estimator_version.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        for (field, value) in [
            ("estimator_sha256", self.estimator_sha256.as_str()),
            ("feature_schema_sha256", self.feature_schema_sha256.as_str()),
            ("utility_schema_sha256", self.utility_schema_sha256.as_str()),
            ("training_split_sha256", self.training_split_sha256.as_str()),
            ("calibration_split_sha256", self.calibration_split_sha256.as_str()),
            ("audit_split_sha256", self.audit_split_sha256.as_str()),
        ] {
            validate_sha256(value, field)?;
        }
        let splits = [
            self.training_split_sha256.as_str(),
            self.calibration_split_sha256.as_str(),
            self.audit_split_sha256.as_str(),
        ];
        if splits[0] == splits[1] || splits[0] == splits[2] || splits[1] == splits[2] {
            return Err(
                "training, calibration and audit split digests must be pairwise distinct".into(),
            );
        }
        if !self.family_alpha.is_finite()
            || self.family_alpha <= 0.0
            || self.family_alpha >= 1.0
        {
            return Err("family_alpha must be finite and strictly between 0 and 1".into());
        }
        if self.family_size == 0 {
            return Err("family_size must be positive".into());
        }
        if !self.target_lower_bound_coverage.is_finite()
            || self.target_lower_bound_coverage <= 0.0
            || self.target_lower_bound_coverage >= 1.0
        {
            return Err(
                "target_lower_bound_coverage must be finite and strictly between 0 and 1".into(),
            );
        }
        if self.minimum_audit_observations < 30 {
            return Err("minimum_audit_observations must be at least 30".into());
        }
        if self.ece_bins == 0 || self.ece_bins > 100 {
            return Err("ece_bins must be between 1 and 100".into());
        }
        if self.observations.len() < self.minimum_audit_observations {
            return Err(format!(
                "audit sample has {} observations, below required minimum {}",
                self.observations.len(),
                self.minimum_audit_observations
            ));
        }

        let mut query_ids = BTreeSet::new();
        for observation in &self.observations {
            require_non_empty(&observation.query_id, "query_id")?;
            if !query_ids.insert(observation.query_id.as_str()) {
                return Err(format!(
                    "duplicate audit query ID {:?}",
                    observation.query_id
                ));
            }
            if !observation.predicted_positive_probability.is_finite()
                || !(0.0..=1.0).contains(&observation.predicted_positive_probability)
            {
                return Err(format!(
                    "query {:?} has a probability outside [0, 1]",
                    observation.query_id
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalCalibrationBin {
    pub bin_index: usize,
    pub count: usize,
    pub mean_predicted_probability: f64,
    pub observed_positive_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalCalibrationAuditReport {
    pub policy_version: String,
    pub estimator_id: String,
    pub estimator_version: String,
    pub estimator_sha256: String,
    pub training_split_sha256: String,
    pub calibration_split_sha256: String,
    pub audit_split_sha256: String,
    pub family_alpha: f64,
    pub family_size: u32,
    pub per_test_alpha: f64,
    pub target_lower_bound_coverage: f64,
    pub sample_count: usize,
    pub covered_count: usize,
    pub observed_lower_bound_coverage: f64,
    pub hoeffding_radius: f64,
    pub lower_confidence_bound_coverage: f64,
    pub brier_score: f64,
    pub expected_calibration_error: f64,
    pub ece_bins: Vec<RetrievalCalibrationBin>,
    pub passed: bool,
    pub report_sha256: String,
}

#[derive(Debug, Serialize)]
struct RetrievalCalibrationReportPayload<'a> {
    policy_version: &'a str,
    estimator_id: &'a str,
    estimator_version: &'a str,
    estimator_sha256: &'a str,
    feature_schema_sha256: &'a str,
    utility_schema_sha256: &'a str,
    training_split_sha256: &'a str,
    calibration_split_sha256: &'a str,
    audit_split_sha256: &'a str,
    family_alpha: f64,
    family_size: u32,
    per_test_alpha: f64,
    target_lower_bound_coverage: f64,
    sample_count: usize,
    covered_count: usize,
    observed_lower_bound_coverage: f64,
    hoeffding_radius: f64,
    lower_confidence_bound_coverage: f64,
    brier_score: f64,
    expected_calibration_error: f64,
    ece_bins: &'a [RetrievalCalibrationBin],
    passed: bool,
}

/// Opaque capability created only by a passing held-out calibration audit.
/// It is deliberately not serializable; cross-process use requires a signed
/// calibration receipt and a separately specified trust/key-management model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedRetrievalCalibration {
    estimator_sha256: String,
    feature_schema_sha256: String,
    utility_schema_sha256: String,
    calibration_split_sha256: String,
    report_sha256: String,
    target_lower_bound_coverage_ppm: u32,
    lower_confidence_bound_coverage_ppm: u32,
}

impl ValidatedRetrievalCalibration {
    /// Bind one VOI estimate to the exact estimator and passing calibration report.
    pub fn verify_estimate_binding(
        &self,
        estimator_sha256: &str,
        calibration_split_sha256: &str,
        feature_schema_sha256: &str,
        utility_schema_sha256: &str,
        calibration_report_sha256: &str,
    ) -> Result<(), String> {
        if self.estimator_sha256 != estimator_sha256 {
            return Err("VOI estimator does not match the calibrated estimator".into());
        }
        if self.calibration_split_sha256 != calibration_split_sha256 {
            return Err("VOI calibration split does not match the passing audit".into());
        }
        if self.feature_schema_sha256 != feature_schema_sha256 {
            return Err("VOI feature schema does not match the passing audit".into());
        }
        if self.utility_schema_sha256 != utility_schema_sha256 {
            return Err("VOI utility schema does not match the passing audit".into());
        }
        if self.report_sha256 != calibration_report_sha256 {
            return Err("VOI calibration report digest does not match the passing audit".into());
        }
        if self.lower_confidence_bound_coverage_ppm
            < self.target_lower_bound_coverage_ppm
        {
            return Err("calibration capability has an insufficient coverage bound".into());
        }
        Ok(())
    }

    pub fn report_sha256(&self) -> &str {
        &self.report_sha256
    }
}

/// Audit the estimator and return its report plus an opaque capability only on
/// a passing finite-sample lower-bound-coverage result.
pub fn audit_retrieval_value_estimator(
    request: RetrievalCalibrationAuditRequest,
) -> Result<(RetrievalCalibrationAuditReport, Option<ValidatedRetrievalCalibration>), String> {
    request.validate()?;

    let sample_count = request.observations.len();
    let covered_count = request
        .observations
        .iter()
        .filter(|observation| {
            observation.observed_net_value_micro
                >= observation.predicted_net_value_lcb_micro
        })
        .count();
    let observed_coverage = covered_count as f64 / sample_count as f64;

    // Bonferroni correction: each pre-registered audit slice receives alpha/m.
    // A one-sided Hoeffding bound controls coverage under i.i.d./exchangeable
    // audit observations while the estimator is fixed before seeing this split.
    let per_test_alpha = request.family_alpha / f64::from(request.family_size);
    if per_test_alpha <= 0.0 || per_test_alpha >= 1.0 {
        return Err("family-wise alpha and family size produce invalid per-test alpha".into());
    }
    let hoeffding_radius =
        (per_test_alpha.ln().mul_add(-1.0, 0.0) / (2.0 * sample_count as f64)).sqrt();
    let lower_coverage = (observed_coverage - hoeffding_radius).max(0.0);

    let mut brier_sum = 0.0f64;
    let mut bin_counts = vec![0usize; request.ece_bins];
    let mut bin_probability_sums = vec![0.0f64; request.ece_bins];
    let mut bin_positive_counts = vec![0usize; request.ece_bins];
    for observation in &request.observations {
        let probability = observation.predicted_positive_probability;
        let observed_positive = if observation.observed_net_value_micro > 0 {
            1.0
        } else {
            0.0
        };
        brier_sum += (probability - observed_positive).powi(2);

        let bin_index = ((probability * request.ece_bins as f64).floor() as usize)
            .min(request.ece_bins - 1);
        bin_counts[bin_index] += 1;
        bin_probability_sums[bin_index] += probability;
        bin_positive_counts[bin_index] += observed_positive as usize;
    }
    let brier_score = brier_sum / sample_count as f64;
    let mut bins = Vec::with_capacity(request.ece_bins);
    let mut ece = 0.0f64;
    for bin_index in 0..request.ece_bins {
        let count = bin_counts[bin_index];
        if count == 0 {
            bins.push(RetrievalCalibrationBin {
                bin_index,
                count: 0,
                mean_predicted_probability: 0.0,
                observed_positive_rate: 0.0,
            });
            continue;
        }
        let mean_predicted = bin_probability_sums[bin_index] / count as f64;
        let observed_rate = bin_positive_counts[bin_index] as f64 / count as f64;
        ece += (count as f64 / sample_count as f64) * (mean_predicted - observed_rate).abs();
        bins.push(RetrievalCalibrationBin {
            bin_index,
            count,
            mean_predicted_probability: mean_predicted,
            observed_positive_rate: observed_rate,
        });
    }

    let target_coverage_ppm =
        (request.target_lower_bound_coverage * 1_000_000.0).ceil() as u32;
    let lower_coverage_ppm = (lower_coverage * 1_000_000.0).floor() as u32;
    // Do not mint a capability when micro-unit quantization would weaken the
    // exact floating-point target comparison.
    let passed = lower_coverage >= request.target_lower_bound_coverage
        && lower_coverage_ppm >= target_coverage_ppm;
    let mut report = RetrievalCalibrationAuditReport {
        policy_version: RETRIEVAL_CALIBRATION_POLICY_VERSION.to_string(),
        estimator_id: request.estimator_id,
        estimator_version: request.estimator_version,
        estimator_sha256: request.estimator_sha256,
        training_split_sha256: request.training_split_sha256,
        calibration_split_sha256: request.calibration_split_sha256,
        audit_split_sha256: request.audit_split_sha256,
        family_alpha: request.family_alpha,
        family_size: request.family_size,
        per_test_alpha,
        target_lower_bound_coverage: request.target_lower_bound_coverage,
        sample_count,
        covered_count,
        observed_lower_bound_coverage: observed_coverage,
        hoeffding_radius,
        lower_confidence_bound_coverage: lower_coverage,
        brier_score,
        expected_calibration_error: ece,
        ece_bins: bins,
        passed,
        report_sha256: String::new(),
    };
    report.report_sha256 = retrieval_calibration_report_digest(&report)?;

    let capability = if passed {
        Some(ValidatedRetrievalCalibration {
            estimator_sha256: report.estimator_sha256.clone(),
            feature_schema_sha256: report.feature_schema_sha256.clone(),
            utility_schema_sha256: report.utility_schema_sha256.clone(),
            calibration_split_sha256: report.calibration_split_sha256.clone(),
            report_sha256: report.report_sha256.clone(),
            target_lower_bound_coverage_ppm: target_coverage_ppm,
            lower_confidence_bound_coverage_ppm: lower_coverage_ppm,
        })
    } else {
        None
    };

    Ok((report, capability))
}

impl RetrievalCalibrationAuditReport {
    /// Recompute the report commitment after JSON transport or archival.
    pub fn verify_internal_consistency(&self) -> Result<(), String> {
        validate_sha256(&self.report_sha256, "report_sha256")?;
        if self.policy_version != RETRIEVAL_CALIBRATION_POLICY_VERSION {
            return Err("unsupported retrieval calibration report version".into());
        }
        if self.sample_count == 0 || self.covered_count > self.sample_count {
            return Err("calibration report sample counts are inconsistent".into());
        }
        let expected = retrieval_calibration_report_digest(self)?;
        if expected != self.report_sha256 {
            return Err("retrieval calibration report SHA-256 mismatch".into());
        }
        Ok(())
    }
}

fn retrieval_calibration_report_digest(
    report: &RetrievalCalibrationAuditReport,
) -> Result<String, String> {
    let payload = RetrievalCalibrationReportPayload {
        policy_version: &report.policy_version,
        estimator_id: &report.estimator_id,
        estimator_version: &report.estimator_version,
        estimator_sha256: &report.estimator_sha256,
        feature_schema_sha256: &report.feature_schema_sha256,
        utility_schema_sha256: &report.utility_schema_sha256,
        training_split_sha256: &report.training_split_sha256,
        calibration_split_sha256: &report.calibration_split_sha256,
        audit_split_sha256: &report.audit_split_sha256,
        family_alpha: report.family_alpha,
        family_size: report.family_size,
        per_test_alpha: report.per_test_alpha,
        target_lower_bound_coverage: report.target_lower_bound_coverage,
        sample_count: report.sample_count,
        covered_count: report.covered_count,
        observed_lower_bound_coverage: report.observed_lower_bound_coverage,
        hoeffding_radius: report.hoeffding_radius,
        lower_confidence_bound_coverage: report.lower_confidence_bound_coverage,
        brier_score: report.brier_score,
        expected_calibration_error: report.expected_calibration_error,
        ece_bins: &report.ece_bins,
        passed: report.passed,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize retrieval calibration report: {error}"))?;
    Ok(sha256_hex(&bytes))
}

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn request_with(observations: Vec<RetrievalCalibrationObservation>) -> RetrievalCalibrationAuditRequest {
        RetrievalCalibrationAuditRequest {
            estimator_id: "voi-estimator".into(),
            estimator_version: "v1".into(),
            estimator_sha256: digest("estimator artifact"),
            feature_schema_sha256: digest("feature schema"),
            utility_schema_sha256: digest("utility schema"),
            training_split_sha256: digest("training split"),
            calibration_split_sha256: digest("calibration split"),
            audit_split_sha256: digest("audit split"),
            family_alpha: 0.05,
            family_size: 1,
            target_lower_bound_coverage: 0.90,
            minimum_audit_observations: 30,
            ece_bins: 10,
            observations,
        }
    }

    fn covered_observations(count: usize) -> Vec<RetrievalCalibrationObservation> {
        (0..count)
            .map(|index| RetrievalCalibrationObservation {
                query_id: format!("query-{index:04}"),
                predicted_positive_probability: 0.9,
                predicted_net_value_lcb_micro: 100,
                observed_net_value_micro: 250,
            })
            .collect()
    }

    #[test]
    fn passing_held_out_coverage_audit_mints_opaque_calibration_capability() {
        let (report, capability) =
            audit_retrieval_value_estimator(request_with(covered_observations(300))).unwrap();
        assert!(report.passed);
        assert_eq!(report.covered_count, 300);
        assert!(report.lower_confidence_bound_coverage >= 0.90);
        assert!(report.brier_score >= 0.0 && report.brier_score <= 1.0);
        assert!(report.expected_calibration_error >= 0.0);
        report.verify_internal_consistency().unwrap();
        let capability = capability.unwrap();
        capability.verify_estimate_binding(
            &digest("estimator artifact"),
            &digest("calibration split"),
            &digest("feature schema"),
            &digest("utility schema"),
            &report.report_sha256,
        ).unwrap();
    }

    #[test]
    fn undercoverage_does_not_mint_calibration_capability() {
        let mut observations = covered_observations(300);
        for observation in observations.iter_mut().take(60) {
            observation.observed_net_value_micro = -100;
        }
        let (report, capability) =
            audit_retrieval_value_estimator(request_with(observations)).unwrap();
        assert!(!report.passed);
        assert!(capability.is_none());
        assert!(report.lower_confidence_bound_coverage < report.target_lower_bound_coverage);
    }

    #[test]
    fn family_wise_bonferroni_correction_can_fail_a_single_slice() {
        let mut request = request_with(covered_observations(300));
        request.family_size = 100;
        let (report, capability) = audit_retrieval_value_estimator(request).unwrap();
        assert!(!report.passed);
        assert!(capability.is_none());
        assert!(report.per_test_alpha < report.family_alpha);
    }

    #[test]
    fn refuses_training_calibration_or_audit_split_reuse() {
        let mut request = request_with(covered_observations(300));
        request.audit_split_sha256 = request.calibration_split_sha256.clone();
        assert!(audit_retrieval_value_estimator(request)
            .unwrap_err()
            .contains("pairwise distinct"));
    }

    #[test]
    fn refuses_too_few_samples_and_duplicate_query_ids() {
        let too_few = request_with(covered_observations(29));
        assert!(audit_retrieval_value_estimator(too_few)
            .unwrap_err()
            .contains("below required minimum"));

        let mut duplicate = covered_observations(30);
        duplicate[1].query_id = duplicate[0].query_id.clone();
        assert!(audit_retrieval_value_estimator(request_with(duplicate))
            .unwrap_err()
            .contains("duplicate audit query ID"));
    }

    #[test]
    fn rejects_invalid_probabilities_and_split_hashes() {
        let mut invalid_probability = covered_observations(30);
        invalid_probability[0].predicted_positive_probability = f64::NAN;
        assert!(audit_retrieval_value_estimator(request_with(invalid_probability)).is_err());

        let mut invalid_hash = request_with(covered_observations(30));
        invalid_hash.audit_split_sha256 = "not-a-sha256".into();
        assert!(audit_retrieval_value_estimator(invalid_hash).is_err());
    }

    #[test]
    fn report_digest_detects_post_audit_mutation() {
        let (mut report, _) =
            audit_retrieval_value_estimator(request_with(covered_observations(300))).unwrap();
        report.passed = false;
        assert!(report.verify_internal_consistency().unwrap_err().contains("SHA-256 mismatch"));
    }

    #[test]
    fn capability_rejects_a_different_estimator_or_calibration_split() {
        let (report, capability) =
            audit_retrieval_value_estimator(request_with(covered_observations(300))).unwrap();
        let capability = capability.unwrap();
        assert!(capability.verify_estimate_binding(
            &digest("different estimator"),
            &report.calibration_split_sha256,
            &report.report_sha256,
        ).unwrap_err().contains("estimator"));
        assert!(capability.verify_estimate_binding(
            &report.estimator_sha256,
            &digest("different calibration split"),
            &digest("feature schema"),
            &digest("utility schema"),
            &report.report_sha256,
        ).unwrap_err().contains("calibration split"));
    }
}
