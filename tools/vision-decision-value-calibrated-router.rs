// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-008: split-conformal lower bounds for decision-value escalation.
//!
//! Calibration rows contain outcomes and are consumed only by the offline profile
//! fitter. The online router accepts a frozen profile and prediction-only inputs.
//! This code is a research reference: the module boundary is not an OS process
//! boundary, and conformal coverage relies on exchangeability within the declared
//! condition/domain. Shifted or unknown domains never escalate through this gate.
//!
//! Run tests without Cargo or third-party dependencies:
//!   rustc --edition=2024 --test tools/vision-decision-value-calibrated-router.rs -o /tmp/vis-dvpc-008-tests
//!   /tmp/vis-dvpc-008-tests

#![deny(unsafe_code)]

#[path = "vision-decision-value-vector-budget.rs"]
mod vector_budget;

use std::collections::BTreeSet;
use vector_budget::{
    AllocationChoice, AllocationCommitV1, AllocationError, ObservedResourceCost,
    PreEscalationInputV1, PreEscalationValueEstimateV1, ResourceVector, VectorBudgetAllocatorV1,
};

pub type EvidenceDigest = [u8; 32];

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationRowV1 {
    /// Stable ID from the frozen calibration manifest, not an evaluator label.
    pub sample_id: String,
    pub split_id: String,
    /// Digest of the causal input used for the domain assessment and value estimate.
    pub input_digest: EvidenceDigest,
    /// Must equal input_digest; prevents accidental reuse of a stale domain result.
    pub domain_assessment_input_digest: EvidenceDigest,
    pub domain_id: String,
    pub domain_assessor_id: String,
    pub domain_assessor_artifact_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub estimator_id: String,
    pub estimator_artifact_digest: EvidenceDigest,
    pub predicted_mean: f64,
    pub predicted_scale: f64,
    /// Observed downstream benefit for calibration only; never available to decide().
    pub observed_benefit: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationConfigV1 {
    pub profile_id: String,
    pub calibration_split_id: String,
    pub domain_id: String,
    pub domain_assessor_id: String,
    pub domain_assessor_artifact_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub estimator_id: String,
    pub estimator_artifact_digest: EvidenceDigest,
    /// Digest of the externally frozen calibration manifest.
    pub calibration_manifest_digest: EvidenceDigest,
    /// One-sided miscoverage rate, e.g. 0.1 for a nominal 90% lower bound.
    pub alpha: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationError {
    InvalidConfig,
    EmptyCalibrationSet,
    UnsupportedQuantileRank { rank: usize, count: usize },
    DuplicateCalibrationSample(String),
    CalibrationMetadataMismatch(String),
    InvalidCalibrationValue(String),
    InvalidPrediction(String),
    InvalidProfile,
    Allocation(AllocationError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitConformalBenefitProfileV1 {
    profile_id: String,
    calibration_split_id: String,
    domain_id: String,
    domain_assessor_id: String,
    domain_assessor_artifact_digest: EvidenceDigest,
    loss_profile_id: String,
    estimator_id: String,
    estimator_artifact_digest: EvidenceDigest,
    calibration_manifest_digest: EvidenceDigest,
    calibration_sample_ids: Vec<String>,
    alpha: f64,
    standardized_overprediction_quantile: f64,
    quantile_rank_one_based: usize,
}

impl SplitConformalBenefitProfileV1 {
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn calibration_count(&self) -> usize {
        self.calibration_sample_ids.len()
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn target_marginal_coverage(&self) -> f64 {
        1.0 - self.alpha
    }

    pub fn standardized_overprediction_quantile(&self) -> f64 {
        self.standardized_overprediction_quantile
    }

    pub fn quantile_rank_one_based(&self) -> usize {
        self.quantile_rank_one_based
    }

    pub fn calibration_split_id(&self) -> &str {
        &self.calibration_split_id
    }

    pub fn domain_id(&self) -> &str {
        &self.domain_id
    }

    pub fn loss_profile_id(&self) -> &str {
        &self.loss_profile_id
    }

    pub fn estimator_id(&self) -> &str {
        &self.estimator_id
    }

    pub fn estimator_artifact_digest(&self) -> &EvidenceDigest {
        &self.estimator_artifact_digest
    }

    pub fn calibration_manifest_digest(&self) -> &EvidenceDigest {
        &self.calibration_manifest_digest
    }

    pub fn calibration_sample_ids(&self) -> &[String] {
        &self.calibration_sample_ids
    }

    fn validate(&self) -> Result<(), CalibrationError> {
        if self.profile_id.trim().is_empty()
            || self.calibration_split_id.trim().is_empty()
            || self.domain_id.trim().is_empty()
            || self.domain_assessor_id.trim().is_empty()
            || self
                .domain_assessor_artifact_digest
                .iter()
                .all(|byte| *byte == 0)
            || self.loss_profile_id.trim().is_empty()
            || self.estimator_id.trim().is_empty()
            || self.estimator_artifact_digest.iter().all(|byte| *byte == 0)
            || self
                .calibration_manifest_digest
                .iter()
                .all(|byte| *byte == 0)
            || !self.alpha.is_finite()
            || self.alpha <= 0.0
            || self.alpha >= 1.0
            || !self.standardized_overprediction_quantile.is_finite()
            || self.calibration_sample_ids.is_empty()
            || self.quantile_rank_one_based == 0
            || self.quantile_rank_one_based > self.calibration_sample_ids.len()
        {
            return Err(CalibrationError::InvalidProfile);
        }
        let mut ids = BTreeSet::new();
        for id in &self.calibration_sample_ids {
            if id.trim().is_empty() || !ids.insert(id) {
                return Err(CalibrationError::InvalidProfile);
            }
        }
        Ok(())
    }
}

/// Fit a one-sided split-conformal lower bound from a frozen, held-out calibration
/// split. The standardized residual is (prediction - observation) / predicted_scale.
/// The finite-sample rank is ceil((n + 1) * (1 - alpha)); if that rank exceeds n,
/// the requested coverage is unsupported and fitting fails closed.
pub fn fit_split_conformal_profile_v1(
    config: &CalibrationConfigV1,
    rows: &[CalibrationRowV1],
) -> Result<SplitConformalBenefitProfileV1, CalibrationError> {
    validate_config(config)?;
    if rows.is_empty() {
        return Err(CalibrationError::EmptyCalibrationSet);
    }

    let mut seen = BTreeSet::new();
    let mut residuals = Vec::with_capacity(rows.len());
    let mut sample_ids = Vec::with_capacity(rows.len());

    for row in rows {
        if row.sample_id.trim().is_empty() || !seen.insert(row.sample_id.as_str()) {
            return Err(CalibrationError::DuplicateCalibrationSample(
                row.sample_id.clone(),
            ));
        }
        if row.split_id != config.calibration_split_id
            || row.input_digest.iter().all(|byte| *byte == 0)
            || row.domain_assessment_input_digest != row.input_digest
            || row.domain_id != config.domain_id
            || row.domain_assessor_id != config.domain_assessor_id
            || row.domain_assessor_artifact_digest != config.domain_assessor_artifact_digest
            || row.loss_profile_id != config.loss_profile_id
            || row.estimator_id != config.estimator_id
            || row.estimator_artifact_digest != config.estimator_artifact_digest
        {
            return Err(CalibrationError::CalibrationMetadataMismatch(
                row.sample_id.clone(),
            ));
        }
        if !row.predicted_mean.is_finite()
            || !row.predicted_scale.is_finite()
            || row.predicted_scale <= 0.0
            || !row.observed_benefit.is_finite()
        {
            return Err(CalibrationError::InvalidCalibrationValue(
                row.sample_id.clone(),
            ));
        }

        let residual = (row.predicted_mean - row.observed_benefit) / row.predicted_scale;
        if !residual.is_finite() {
            return Err(CalibrationError::InvalidCalibrationValue(
                row.sample_id.clone(),
            ));
        }
        residuals.push(residual);
        sample_ids.push(row.sample_id.clone());
    }

    let rank = ((rows.len() + 1) as f64 * (1.0 - config.alpha)).ceil() as usize;
    if rank == 0 || rank > residuals.len() {
        return Err(CalibrationError::UnsupportedQuantileRank {
            rank,
            count: residuals.len(),
        });
    }

    residuals.sort_by(f64::total_cmp);
    sample_ids.sort();

    let profile = SplitConformalBenefitProfileV1 {
        profile_id: config.profile_id.clone(),
        calibration_split_id: config.calibration_split_id.clone(),
        domain_id: config.domain_id.clone(),
        domain_assessor_id: config.domain_assessor_id.clone(),
        domain_assessor_artifact_digest: config.domain_assessor_artifact_digest,
        loss_profile_id: config.loss_profile_id.clone(),
        estimator_id: config.estimator_id.clone(),
        estimator_artifact_digest: config.estimator_artifact_digest,
        calibration_manifest_digest: config.calibration_manifest_digest,
        calibration_sample_ids: sample_ids,
        alpha: config.alpha,
        standardized_overprediction_quantile: residuals[rank - 1],
        quantile_rank_one_based: rank,
    };
    profile.validate()?;
    Ok(profile)
}

fn validate_config(config: &CalibrationConfigV1) -> Result<(), CalibrationError> {
    if config.profile_id.trim().is_empty()
        || config.calibration_split_id.trim().is_empty()
        || config.domain_id.trim().is_empty()
        || config.domain_assessor_id.trim().is_empty()
        || config
            .domain_assessor_artifact_digest
            .iter()
            .all(|byte| *byte == 0)
        || config.loss_profile_id.trim().is_empty()
        || config.estimator_id.trim().is_empty()
        || config
            .estimator_artifact_digest
            .iter()
            .all(|byte| *byte == 0)
        || config
            .calibration_manifest_digest
            .iter()
            .all(|byte| *byte == 0)
        || !config.alpha.is_finite()
        || config.alpha <= 0.0
        || config.alpha >= 1.0
    {
        return Err(CalibrationError::InvalidConfig);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainStatusV1 {
    InDomain,
    OodSuspected,
    Unknown,
}

impl DomainStatusV1 {
    fn label(self) -> &'static str {
        match self {
            Self::InDomain => "in_domain",
            Self::OodSuspected => "ood_suspected",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateReasonV1 {
    CalibratedInDomain,
    ShiftDetected,
    DomainUnknown,
    DomainMismatch,
    DomainAssessorMismatch,
    DomainAssessorArtifactMismatch,
    DomainAssessmentInputMismatch,
    LossProfileMismatch,
    EstimatorMismatch,
    EstimatorArtifactMismatch,
    InvalidPrediction,
    InvalidScale,
    InvalidCalibrationProfile,
}

impl GateReasonV1 {
    fn label(self) -> &'static str {
        match self {
            Self::CalibratedInDomain => "calibrated_in_domain",
            Self::ShiftDetected => "shift_detected",
            Self::DomainUnknown => "domain_unknown",
            Self::DomainMismatch => "domain_mismatch",
            Self::DomainAssessorMismatch => "domain_assessor_mismatch",
            Self::DomainAssessorArtifactMismatch => "domain_assessor_artifact_mismatch",
            Self::DomainAssessmentInputMismatch => "domain_assessment_input_mismatch",
            Self::LossProfileMismatch => "loss_profile_mismatch",
            Self::EstimatorMismatch => "estimator_mismatch",
            Self::EstimatorArtifactMismatch => "estimator_artifact_mismatch",
            Self::InvalidPrediction => "invalid_prediction",
            Self::InvalidScale => "invalid_scale",
            Self::InvalidCalibrationProfile => "invalid_calibration_profile",
        }
    }
}

/// Prediction-only input exposed to the online gate. This type has no field for
/// observed benefit, evaluator labels, counterfactual loss, or expensive output.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedRouterInputV1 {
    pub trial_id: u64,
    pub input_digest: EvidenceDigest,
    pub domain_assessment_input_digest: EvidenceDigest,
    pub domain_status: DomainStatusV1,
    pub domain_id: String,
    pub domain_assessor_id: String,
    pub domain_assessor_artifact_digest: EvidenceDigest,
    pub loss_profile_id: String,
    pub estimator_id: String,
    pub estimator_artifact_digest: EvidenceDigest,
    pub predicted_mean_benefit: f64,
    pub predicted_scale: f64,
    pub cheap_path_safe: bool,
    pub escalation_cost_upper_bound: Option<ResourceVector>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedRouterDecisionV1 {
    pub gate_reason: GateReasonV1,
    pub calibrated_lower_bound: Option<f64>,
    pub calibration_profile_id: String,
    pub calibration_count: usize,
    pub target_marginal_coverage: f64,
    pub allocation: AllocationCommitV1,
}

impl CalibratedRouterDecisionV1 {
    /// Stable receipt combines calibration provenance, gate state and allocation.
    pub fn to_tsv(&self) -> String {
        let allocation = self.allocation.to_tsv();
        let fields = allocation.trim_end().strip_prefix("COMMIT\t").unwrap_or("");
        format!(
            "DVPC008\t{}\t{}\t{}\t{:.6}\t{}\t{}\n",
            self.gate_reason.label(),
            self.calibration_profile_id,
            self.calibration_count,
            self.target_marginal_coverage,
            self.calibrated_lower_bound
                .map(|value| format!("{value:.9}"))
                .unwrap_or_else(|| "NA".to_owned()),
            fields,
        )
    }
}

#[derive(Debug, Clone)]
pub struct CalibratedDecisionValueRouterV1 {
    profile: SplitConformalBenefitProfileV1,
    allocator: VectorBudgetAllocatorV1,
}

impl CalibratedDecisionValueRouterV1 {
    pub fn new(
        profile: SplitConformalBenefitProfileV1,
        minimum_net_benefit: f64,
        capacity: ResourceVector,
    ) -> Result<Self, CalibrationError> {
        profile.validate()?;
        let allocator = VectorBudgetAllocatorV1::new(
            profile.loss_profile_id.clone(),
            minimum_net_benefit,
            capacity,
        )
        .map_err(CalibrationError::Allocation)?;
        Ok(Self { profile, allocator })
    }

    pub fn budget(&self) -> &vector_budget::VectorBudgetLedgerV1 {
        self.allocator.budget()
    }

    pub fn settle(
        &mut self,
        reservation_id: u64,
        actual: ObservedResourceCost,
    ) -> Result<vector_budget::BudgetStatus, CalibrationError> {
        self.allocator
            .settle(reservation_id, actual)
            .map_err(CalibrationError::Allocation)
    }

    /// Make a policy decision from pre-escalation prediction-only information.
    /// Any shift, metadata mismatch or malformed prediction strips the value
    /// estimate before delegation to the vector-budget allocator.
    pub fn decide(
        &mut self,
        input: CalibratedRouterInputV1,
    ) -> Result<CalibratedRouterDecisionV1, CalibrationError> {
        let gate = self.gate_reason(&input);
        let lower_bound = if gate == GateReasonV1::CalibratedInDomain {
            let result = input.predicted_mean_benefit
                - self.profile.standardized_overprediction_quantile * input.predicted_scale;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        } else {
            None
        };
        let gate = if gate == GateReasonV1::CalibratedInDomain && lower_bound.is_none() {
            GateReasonV1::InvalidPrediction
        } else {
            gate
        };

        let value_estimate = match (gate, lower_bound) {
            (GateReasonV1::CalibratedInDomain, Some(bound)) => Some(PreEscalationValueEstimateV1 {
                estimator_id: input.estimator_id.clone(),
                estimator_artifact_digest: input.estimator_artifact_digest,
                loss_profile_id: self.profile.loss_profile_id.clone(),
                source_input_digest: input.input_digest,
                net_benefit_lower_bound: bound,
            }),
            _ => None,
        };

        // Use the frozen profile's loss contract for fallback commits, even when
        // the caller's requested loss profile is stale. GateReason preserves why.
        let policy_input = PreEscalationInputV1 {
            trial_id: input.trial_id,
            loss_profile_id: self.profile.loss_profile_id.clone(),
            input_digest: input.input_digest,
            cheap_path_safe: input.cheap_path_safe,
            value_estimate,
            escalation_cost_upper_bound: input.escalation_cost_upper_bound,
        };
        let allocation = self
            .allocator
            .decide(policy_input)
            .map_err(CalibrationError::Allocation)?;

        Ok(CalibratedRouterDecisionV1 {
            gate_reason: gate,
            calibrated_lower_bound: lower_bound
                .filter(|_| gate == GateReasonV1::CalibratedInDomain),
            calibration_profile_id: self.profile.profile_id.clone(),
            calibration_count: self.profile.calibration_count(),
            target_marginal_coverage: self.profile.target_marginal_coverage(),
            allocation,
        })
    }

    fn gate_reason(&self, input: &CalibratedRouterInputV1) -> GateReasonV1 {
        if self.profile.validate().is_err() {
            return GateReasonV1::InvalidCalibrationProfile;
        }
        match input.domain_status {
            DomainStatusV1::OodSuspected => return GateReasonV1::ShiftDetected,
            DomainStatusV1::Unknown => return GateReasonV1::DomainUnknown,
            DomainStatusV1::InDomain => {}
        }
        if input.domain_assessment_input_digest != input.input_digest {
            return GateReasonV1::DomainAssessmentInputMismatch;
        }
        if input.domain_id != self.profile.domain_id {
            return GateReasonV1::DomainMismatch;
        }
        if input.domain_assessor_id != self.profile.domain_assessor_id {
            return GateReasonV1::DomainAssessorMismatch;
        }
        if input.domain_assessor_artifact_digest != self.profile.domain_assessor_artifact_digest {
            return GateReasonV1::DomainAssessorArtifactMismatch;
        }
        if input.loss_profile_id != self.profile.loss_profile_id {
            return GateReasonV1::LossProfileMismatch;
        }
        if input.estimator_id != self.profile.estimator_id {
            return GateReasonV1::EstimatorMismatch;
        }
        if input.estimator_artifact_digest != self.profile.estimator_artifact_digest {
            return GateReasonV1::EstimatorArtifactMismatch;
        }
        if !input.predicted_mean_benefit.is_finite() {
            return GateReasonV1::InvalidPrediction;
        }
        if !input.predicted_scale.is_finite() || input.predicted_scale <= 0.0 {
            return GateReasonV1::InvalidScale;
        }
        GateReasonV1::CalibratedInDomain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOSS: &str = "collision-weighted-task-loss-v1";
    const ESTIMATOR: &str = "pre-escalation-value-v2";
    const DOMAIN: &str = "night-clear-condition-v1";
    const CAL_SPLIT: &str = "calibration-2026-10";
    const ARTIFACT: EvidenceDigest = [0xAB; 32];
    const MANIFEST: EvidenceDigest = [0xCD; 32];

    fn config(alpha: f64) -> CalibrationConfigV1 {
        CalibrationConfigV1 {
            profile_id: "dvpc-value-lcb-profile-v1".to_owned(),
            calibration_split_id: CAL_SPLIT.to_owned(),
            domain_id: DOMAIN.to_owned(),
            domain_assessor_id: "domain-shift-scorer-v1".to_owned(),
            domain_assessor_artifact_digest: [0xDC; 32],
            loss_profile_id: LOSS.to_owned(),
            estimator_id: ESTIMATOR.to_owned(),
            estimator_artifact_digest: ARTIFACT,
            calibration_manifest_digest: MANIFEST,
            alpha,
        }
    }

    fn calibration_rows() -> Vec<CalibrationRowV1> {
        (0..5)
            .map(|index| {
                let z = index as f64;
                CalibrationRowV1 {
                    sample_id: format!("cal-{index:02}"),
                    split_id: CAL_SPLIT.to_owned(),
                    input_digest: [index as u8 + 1; 32],
                    domain_assessment_input_digest: [index as u8 + 1; 32],
                    domain_id: DOMAIN.to_owned(),
                    domain_assessor_id: "domain-shift-scorer-v1".to_owned(),
                    domain_assessor_artifact_digest: [0xDC; 32],
                    loss_profile_id: LOSS.to_owned(),
                    estimator_id: ESTIMATOR.to_owned(),
                    estimator_artifact_digest: ARTIFACT,
                    predicted_mean: 0.5,
                    predicted_scale: 0.1,
                    observed_benefit: 0.5 - z * 0.1,
                }
            })
            .collect()
    }

    fn profile(alpha: f64) -> SplitConformalBenefitProfileV1 {
        fit_split_conformal_profile_v1(&config(alpha), &calibration_rows()).unwrap()
    }

    fn input(trial_id: u64, mean: f64, scale: f64) -> CalibratedRouterInputV1 {
        CalibratedRouterInputV1 {
            trial_id,
            input_digest: [trial_id as u8 + 1; 32],
            domain_assessment_input_digest: [trial_id as u8 + 1; 32],
            domain_status: DomainStatusV1::InDomain,
            domain_id: DOMAIN.to_owned(),
            domain_assessor_id: "domain-shift-scorer-v1".to_owned(),
            domain_assessor_artifact_digest: [0xDC; 32],
            loss_profile_id: LOSS.to_owned(),
            estimator_id: ESTIMATOR.to_owned(),
            estimator_artifact_digest: ARTIFACT,
            predicted_mean_benefit: mean,
            predicted_scale: scale,
            cheap_path_safe: true,
            escalation_cost_upper_bound: Some(ResourceVector::new(20, 10)),
        }
    }

    fn router() -> CalibratedDecisionValueRouterV1 {
        CalibratedDecisionValueRouterV1::new(profile(0.4), 0.05, ResourceVector::new(100, 100))
            .unwrap()
    }

    #[test]
    fn conformal_quantile_uses_finite_sample_one_based_rank() {
        let fitted = profile(0.4);
        assert_eq!(fitted.calibration_count(), 5);
        assert_eq!(fitted.quantile_rank_one_based(), 4);
        assert!((fitted.standardized_overprediction_quantile() - 3.0).abs() < 1e-12);
        assert!((fitted.target_marginal_coverage() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn calibration_result_is_invariant_to_input_order() {
        let cfg = config(0.4);
        let rows = calibration_rows();
        let forward = fit_split_conformal_profile_v1(&cfg, &rows).unwrap();
        let mut reversed = rows;
        reversed.reverse();
        let backward = fit_split_conformal_profile_v1(&cfg, &reversed).unwrap();
        assert_eq!(
            forward.standardized_overprediction_quantile(),
            backward.standardized_overprediction_quantile()
        );
        assert_eq!(
            forward.calibration_sample_ids(),
            backward.calibration_sample_ids()
        );
    }

    #[test]
    fn rejects_unsupported_finite_sample_quantile_rank() {
        let error = fit_split_conformal_profile_v1(&config(0.05), &calibration_rows()).unwrap_err();
        assert_eq!(
            error,
            CalibrationError::UnsupportedQuantileRank { rank: 6, count: 5 }
        );
    }

    #[test]
    fn rejects_duplicate_calibration_ids() {
        let mut rows = calibration_rows();
        rows[4].sample_id = rows[0].sample_id.clone();
        assert!(matches!(
            fit_split_conformal_profile_v1(&config(0.4), &rows),
            Err(CalibrationError::DuplicateCalibrationSample(_))
        ));
    }

    #[test]
    fn rejects_calibration_row_from_wrong_model_or_split() {
        let mut rows = calibration_rows();
        rows[0].estimator_artifact_digest = [0xFE; 32];
        assert!(matches!(
            fit_split_conformal_profile_v1(&config(0.4), &rows),
            Err(CalibrationError::CalibrationMetadataMismatch(_))
        ));
        let mut rows = calibration_rows();
        rows[1].split_id = "test".to_owned();
        assert!(matches!(
            fit_split_conformal_profile_v1(&config(0.4), &rows),
            Err(CalibrationError::CalibrationMetadataMismatch(_))
        ));
        let mut rows = calibration_rows();
        rows[2].domain_assessor_artifact_digest = [0xEF; 32];
        assert!(matches!(
            fit_split_conformal_profile_v1(&config(0.4), &rows),
            Err(CalibrationError::CalibrationMetadataMismatch(_))
        ));
    }

    #[test]
    fn rejects_nonpositive_uncertainty_scale_during_calibration() {
        let mut rows = calibration_rows();
        rows[0].predicted_scale = 0.0;
        assert!(matches!(
            fit_split_conformal_profile_v1(&config(0.4), &rows),
            Err(CalibrationError::InvalidCalibrationValue(_))
        ));
    }

    #[test]
    fn positive_calibrated_lower_bound_composes_with_vector_budget() {
        let mut policy = router();
        let decision = policy.decide(input(1, 0.30, 0.02)).unwrap();
        assert_eq!(decision.gate_reason, GateReasonV1::CalibratedInDomain);
        assert!((decision.calibrated_lower_bound.unwrap() - 0.24).abs() < 1e-12);
        assert_eq!(decision.allocation.choice, AllocationChoice::Escalate);
        assert_eq!(policy.budget().reserved(), ResourceVector::new(20, 10));
        policy
            .settle(
                decision.allocation.reservation_id.unwrap(),
                ObservedResourceCost::known(15, 8),
            )
            .unwrap();
        assert_eq!(policy.budget().charged(), ResourceVector::new(15, 8));
    }

    #[test]
    fn lower_bound_below_threshold_keeps_cheap() {
        let mut policy = router();
        let decision = policy.decide(input(1, 0.03, 0.02)).unwrap();
        assert!(decision.calibrated_lower_bound.unwrap() < 0.05);
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }

    #[test]
    fn suspected_shift_never_escalates_even_for_large_raw_benefit() {
        let mut policy = router();
        let mut candidate = input(1, 1000.0, 0.001);
        candidate.domain_status = DomainStatusV1::OodSuspected;
        let decision = policy.decide(candidate).unwrap();
        assert_eq!(decision.gate_reason, GateReasonV1::ShiftDetected);
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);
        assert_eq!(decision.calibrated_lower_bound, None);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }

    #[test]
    fn unknown_domain_abstains_when_cheap_path_is_not_safe() {
        let mut policy = router();
        let mut candidate = input(1, 1000.0, 0.001);
        candidate.domain_status = DomainStatusV1::Unknown;
        candidate.cheap_path_safe = false;
        let decision = policy.decide(candidate).unwrap();
        assert_eq!(decision.gate_reason, GateReasonV1::DomainUnknown);
        assert_eq!(decision.allocation.choice, AllocationChoice::Abstain);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }

    #[test]
    fn changed_estimator_digest_never_uses_stale_calibration() {
        let mut policy = router();
        let mut candidate = input(1, 1000.0, 0.001);
        candidate.estimator_artifact_digest = [0xEF; 32];
        let decision = policy.decide(candidate).unwrap();
        assert_eq!(
            decision.gate_reason,
            GateReasonV1::EstimatorArtifactMismatch
        );
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);
        assert_eq!(decision.calibrated_lower_bound, None);
    }

    #[test]
    fn mismatched_domain_and_loss_profile_block_the_bound() {
        let mut policy = router();
        let mut candidate = input(1, 100.0, 0.001);
        candidate.domain_id = "day-fog-v2".to_owned();
        let decision = policy.decide(candidate).unwrap();
        assert_eq!(decision.gate_reason, GateReasonV1::DomainMismatch);
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);

        let mut candidate = input(2, 100.0, 0.001);
        candidate.loss_profile_id = "different-task-loss".to_owned();
        let decision = policy.decide(candidate).unwrap();
        assert_eq!(decision.gate_reason, GateReasonV1::LossProfileMismatch);
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);
    }

    #[test]
    fn invalid_predictions_are_rejected_before_lower_bound_use() {
        let mut policy = router();
        let mut non_finite = input(1, f64::NAN, 0.01);
        let decision = policy.decide(non_finite.clone()).unwrap();
        assert_eq!(decision.gate_reason, GateReasonV1::InvalidPrediction);
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);

        non_finite.trial_id = 2;
        non_finite.input_digest = [3; 32];
        non_finite.domain_assessment_input_digest = [3; 32];
        non_finite.predicted_mean_benefit = 100.0;
        non_finite.predicted_scale = 0.0;
        let decision = policy.decide(non_finite).unwrap();
        assert_eq!(decision.gate_reason, GateReasonV1::InvalidScale);
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);
    }

    #[test]
    fn zero_point_benefit_with_uncertainty_does_not_escalate() {
        let mut policy = router();
        let decision = policy.decide(input(1, 0.0, 0.02)).unwrap();
        assert!(decision.calibrated_lower_bound.unwrap() < 0.05);
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }

    #[test]
    fn changed_domain_assessor_id_or_artifact_blocks_escalation() {
        let mut policy = router();
        let mut candidate = input(1, 1000.0, 0.001);
        candidate.domain_assessor_id = "new-domain-scorer".to_owned();
        let decision = policy.decide(candidate).unwrap();
        assert_eq!(decision.gate_reason, GateReasonV1::DomainAssessorMismatch);
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);

        let mut candidate = input(2, 1000.0, 0.001);
        candidate.domain_assessor_artifact_digest = [0xEE; 32];
        let decision = policy.decide(candidate).unwrap();
        assert_eq!(
            decision.gate_reason,
            GateReasonV1::DomainAssessorArtifactMismatch
        );
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }

    #[test]
    fn stale_domain_assessment_input_digest_blocks_escalation() {
        let mut policy = router();
        let mut candidate = input(1, 1000.0, 0.001);
        candidate.domain_assessment_input_digest = [0xFE; 32];
        let decision = policy.decide(candidate).unwrap();
        assert_eq!(
            decision.gate_reason,
            GateReasonV1::DomainAssessmentInputMismatch
        );
        assert_eq!(decision.allocation.choice, AllocationChoice::KeepCheap);
        assert_eq!(policy.budget().reserved(), ResourceVector::default());
    }

    #[test]
    fn receipts_are_deterministic_and_expose_calibration_state() {
        let mut policy = router();
        let decision = policy.decide(input(1, 0.30, 0.02)).unwrap();
        let first = decision.to_tsv();
        let second = decision.to_tsv();
        assert_eq!(first, second);
        assert!(first.starts_with(
            "DVPC008\tcalibrated_in_domain\tdvpc-value-lcb-profile-v1\t5\t0.600000\t0.240000000\t1\tescalate\tpositive_conservative_value"
        ));
    }
}
