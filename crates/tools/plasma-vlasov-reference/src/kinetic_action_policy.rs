//! Response-constrained policy for kinetic representation and horizon actions.
//!
//! This deterministic policy layer is not a predictor, certificate, or state
//! mutator. Callers supply independently calibrated upper bounds for future
//! response, branch and flux errors. The policy recommends an action or
//! abstains; it never applies a sponge, changes a grid, or advances a solver.
//!
//! A velocity-Fourier Nyquist encounter is a warning, not automatic expiry.
//! Actual guard-band occupancy, support validity, calibration coverage and
//! response-risk bounds determine the qualification status.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizonStatus {
    Qualified,
    Warning,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizonReason {
    NyquistEncountered,
    NearDiscreteRecurrence,
    HighVelocityGuardband,
    VelocityBoundaryOccupancy,
    FutureResponseRisk,
    TransferFluxRisk,
    CalibrationDomainMissing,
    VelocitySupportInvalid,
    DiscreteRecurrenceReached,
    GuardbandLimitExceeded,
    BoundaryLimitExceeded,
    FutureResponseLimitExceeded,
    TransferFluxLimitExceeded,
}

/// State evidence for one registered spatial mode / intervention / representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonEvidence {
    pub current_time: f64,
    pub nyquist_encounter_time: f64,
    pub discrete_recurrence_time: f64,
    pub high_velocity_guardband_fraction: f64,
    pub velocity_boundary_ratio: f64,
    pub future_response_error_upper_bound: f64,
    pub transfer_flux_error_upper_bound: f64,
    pub calibration_domain_covered: bool,
    pub velocity_support_valid: bool,
}

/// Warning thresholds are strictly below hard limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonLimits {
    pub warning_guardband_fraction: f64,
    pub maximum_guardband_fraction: f64,
    pub warning_boundary_ratio: f64,
    pub maximum_boundary_ratio: f64,
    pub warning_response_error: f64,
    pub maximum_response_error: f64,
    pub warning_transfer_flux_error: f64,
    pub maximum_transfer_flux_error: f64,
    /// Warn at this fraction of the predicted recurrence time; this is not a hard expiry.
    pub recurrence_warning_fraction: f64,
}

impl HorizonLimits {
    pub fn validate(self) -> Result<(), PolicyInputError> {
        let values = [
            self.warning_guardband_fraction,
            self.maximum_guardband_fraction,
            self.warning_boundary_ratio,
            self.maximum_boundary_ratio,
            self.warning_response_error,
            self.maximum_response_error,
            self.warning_transfer_flux_error,
            self.maximum_transfer_flux_error,
            self.recurrence_warning_fraction,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PolicyInputError::NonFiniteLimit);
        }
        if !(0.0..=1.0).contains(&self.warning_guardband_fraction)
            || !(0.0..=1.0).contains(&self.maximum_guardband_fraction)
            || self.warning_guardband_fraction >= self.maximum_guardband_fraction
        {
            return Err(PolicyInputError::InvalidGuardbandLimits);
        }
        if !(0.0..=1.0).contains(&self.warning_boundary_ratio)
            || !(0.0..=1.0).contains(&self.maximum_boundary_ratio)
            || self.warning_boundary_ratio >= self.maximum_boundary_ratio
        {
            return Err(PolicyInputError::InvalidBoundaryLimits);
        }
        if self.warning_response_error < 0.0
            || self.maximum_response_error <= self.warning_response_error
        {
            return Err(PolicyInputError::InvalidResponseLimits);
        }
        if self.warning_transfer_flux_error < 0.0
            || self.maximum_transfer_flux_error <= self.warning_transfer_flux_error
        {
            return Err(PolicyInputError::InvalidFluxLimits);
        }
        if !(0.0..1.0).contains(&self.recurrence_warning_fraction) {
            return Err(PolicyInputError::InvalidRecurrenceWarningFraction);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HorizonAssessment {
    pub status: HorizonStatus,
    pub reasons: Vec<HorizonReason>,
    pub predicted_recurrence_time: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyInputError {
    NonFiniteLimit,
    InvalidGuardbandLimits,
    InvalidBoundaryLimits,
    InvalidResponseLimits,
    InvalidFluxLimits,
    InvalidRecurrenceWarningFraction,
    NonFiniteEvidence,
    InvalidTimeScales,
    InvalidFraction,
    NegativeErrorBound,
}

impl HorizonEvidence {
    pub fn assess(self, limits: HorizonLimits) -> Result<HorizonAssessment, PolicyInputError> {
        limits.validate()?;
        self.validate()?;

        let mut reasons = Vec::new();
        let mut hard_failure = false;
        let mut warning = false;

        if !self.calibration_domain_covered {
            reasons.push(HorizonReason::CalibrationDomainMissing);
            hard_failure = true;
        }
        if !self.velocity_support_valid {
            reasons.push(HorizonReason::VelocitySupportInvalid);
            hard_failure = true;
        }

        if self.current_time >= self.discrete_recurrence_time {
            reasons.push(HorizonReason::DiscreteRecurrenceReached);
            hard_failure = true;
        } else if self.current_time
            >= limits.recurrence_warning_fraction * self.discrete_recurrence_time
        {
            reasons.push(HorizonReason::NearDiscreteRecurrence);
            warning = true;
        }

        // Nyquist is deliberately a warning only. It does not by itself
        // invalidate a measured response branch.
        if self.current_time >= self.nyquist_encounter_time {
            reasons.push(HorizonReason::NyquistEncountered);
            warning = true;
        }

        if self.high_velocity_guardband_fraction >= limits.maximum_guardband_fraction {
            reasons.push(HorizonReason::GuardbandLimitExceeded);
            hard_failure = true;
        } else if self.high_velocity_guardband_fraction
            >= limits.warning_guardband_fraction
        {
            reasons.push(HorizonReason::HighVelocityGuardband);
            warning = true;
        }

        if self.velocity_boundary_ratio >= limits.maximum_boundary_ratio {
            reasons.push(HorizonReason::BoundaryLimitExceeded);
            hard_failure = true;
        } else if self.velocity_boundary_ratio >= limits.warning_boundary_ratio {
            reasons.push(HorizonReason::VelocityBoundaryOccupancy);
            warning = true;
        }

        if self.future_response_error_upper_bound >= limits.maximum_response_error {
            reasons.push(HorizonReason::FutureResponseLimitExceeded);
            hard_failure = true;
        } else if self.future_response_error_upper_bound >= limits.warning_response_error {
            reasons.push(HorizonReason::FutureResponseRisk);
            warning = true;
        }

        if self.transfer_flux_error_upper_bound >= limits.maximum_transfer_flux_error {
            reasons.push(HorizonReason::TransferFluxLimitExceeded);
            hard_failure = true;
        } else if self.transfer_flux_error_upper_bound >= limits.warning_transfer_flux_error {
            reasons.push(HorizonReason::TransferFluxRisk);
            warning = true;
        }

        let status = if hard_failure {
            HorizonStatus::Unknown
        } else if warning {
            HorizonStatus::Warning
        } else {
            HorizonStatus::Qualified
        };

        Ok(HorizonAssessment {
            status,
            reasons,
            predicted_recurrence_time: self.discrete_recurrence_time,
        })
    }

    fn validate(self) -> Result<(), PolicyInputError> {
        let values = [
            self.current_time,
            self.nyquist_encounter_time,
            self.discrete_recurrence_time,
            self.high_velocity_guardband_fraction,
            self.velocity_boundary_ratio,
            self.future_response_error_upper_bound,
            self.transfer_flux_error_upper_bound,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PolicyInputError::NonFiniteEvidence);
        }
        if self.current_time < 0.0
            || self.nyquist_encounter_time <= 0.0
            || self.discrete_recurrence_time <= self.nyquist_encounter_time
        {
            return Err(PolicyInputError::InvalidTimeScales);
        }
        if !(0.0..=1.0).contains(&self.high_velocity_guardband_fraction)
            || !(0.0..=1.0).contains(&self.velocity_boundary_ratio)
        {
            return Err(PolicyInputError::InvalidFraction);
        }
        if self.future_response_error_upper_bound < 0.0
            || self.transfer_flux_error_upper_bound < 0.0
        {
            return Err(PolicyInputError::NegativeErrorBound);
        }
        Ok(())
    }
}

/// A possible representation action. This policy estimates and ranks actions
/// supplied by the caller; it does not execute them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KineticAction {
    RetainCurrentRepresentation,
    RefineVelocityGrid,
    ExpandVelocitySupport,
    VelocityFourierSponge,
    ConservativeHermiteTransfer,
    CharacteristicVelocityOutflow,
    ParticleMeasureHandoff,
}

/// Candidate upper bounds and costs. Estimates must be calibrated independently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionEstimate {
    pub action: KineticAction,
    pub future_response_error_upper_bound: f64,
    pub branch_leakage_upper_bound: f64,
    pub transfer_flux_error_upper_bound: f64,
    pub invariant_error_upper_bound: f64,
    pub dissipated_casimir_fraction: f64,
    pub estimated_wall_seconds: f64,
    pub estimated_peak_memory_bytes: u64,
    pub calibration_domain_covered: bool,
    pub velocity_support_valid: bool,
    pub positivity_preserved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionConstraints {
    pub maximum_future_response_error: f64,
    pub maximum_branch_leakage: f64,
    pub maximum_transfer_flux_error: f64,
    pub maximum_invariant_error: f64,
    pub maximum_dissipated_casimir_fraction: f64,
    pub maximum_wall_seconds: f64,
    pub maximum_peak_memory_bytes: u64,
}

impl ActionConstraints {
    pub fn validate(self) -> Result<(), PolicyInputError> {
        let values = [
            self.maximum_future_response_error,
            self.maximum_branch_leakage,
            self.maximum_transfer_flux_error,
            self.maximum_invariant_error,
            self.maximum_dissipated_casimir_fraction,
            self.maximum_wall_seconds,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PolicyInputError::NonFiniteLimit);
        }
        if values.iter().any(|value| *value < 0.0)
            || self.maximum_dissipated_casimir_fraction > 1.0
        {
            return Err(PolicyInputError::InvalidFraction);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionAbstentionReason {
    NoCandidates,
    NoCalibratedCandidate,
    NoCandidateMeetsConstraints,
}

/// Pareto-filtered recommendation. Indices address the original candidate
/// slice. The default recommendation is lowest wall time on the Pareto front;
/// callers retain the front for alternative cost / dissipation preferences.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDecision {
    pub recommended_index: Option<usize>,
    pub pareto_front_indices: Vec<usize>,
    pub abstention_reason: Option<ActionAbstentionReason>,
}

impl ActionDecision {
    pub fn select(
        candidates: &[ActionEstimate],
        constraints: ActionConstraints,
    ) -> Result<Self, PolicyInputError> {
        constraints.validate()?;
        if candidates.is_empty() {
            return Ok(Self::abstain(ActionAbstentionReason::NoCandidates));
        }

        let mut feasible = Vec::new();
        let mut calibrated_count = 0usize;
        for (index, candidate) in candidates.iter().enumerate() {
            if !candidate.calibration_domain_covered {
                continue;
            }
            calibrated_count += 1;
            if candidate.meets(constraints) {
                feasible.push(index);
            }
        }

        if feasible.is_empty() {
            let reason = if calibrated_count == 0 {
                ActionAbstentionReason::NoCalibratedCandidate
            } else {
                ActionAbstentionReason::NoCandidateMeetsConstraints
            };
            return Ok(Self::abstain(reason));
        }

        let mut pareto_front = Vec::new();
        for &candidate_index in &feasible {
            let dominated = feasible.iter().copied().any(|other_index| {
                other_index != candidate_index
                    && candidates[other_index].dominates(candidates[candidate_index])
            });
            if !dominated {
                pareto_front.push(candidate_index);
            }
        }

        let recommended_index = pareto_front.iter().copied().min_by(|left, right| {
            candidates[*left]
                .estimated_wall_seconds
                .total_cmp(&candidates[*right].estimated_wall_seconds)
                .then_with(|| {
                    candidates[*left]
                        .dissipated_casimir_fraction
                        .total_cmp(&candidates[*right].dissipated_casimir_fraction)
                })
                .then_with(|| {
                    candidates[*left]
                        .estimated_peak_memory_bytes
                        .cmp(&candidates[*right].estimated_peak_memory_bytes)
                })
        });

        Ok(Self {
            recommended_index,
            pareto_front_indices: pareto_front,
            abstention_reason: None,
        })
    }

    fn abstain(reason: ActionAbstentionReason) -> Self {
        Self {
            recommended_index: None,
            pareto_front_indices: Vec::new(),
            abstention_reason: Some(reason),
        }
    }
}

impl ActionEstimate {
    fn finite_nonnegative(self) -> bool {
        [
            self.future_response_error_upper_bound,
            self.branch_leakage_upper_bound,
            self.transfer_flux_error_upper_bound,
            self.invariant_error_upper_bound,
            self.dissipated_casimir_fraction,
            self.estimated_wall_seconds,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
            && self.dissipated_casimir_fraction <= 1.0
    }

    fn meets(self, limits: ActionConstraints) -> bool {
        self.finite_nonnegative()
            && self.calibration_domain_covered
            && self.velocity_support_valid
            && self.positivity_preserved
            && self.future_response_error_upper_bound <= limits.maximum_future_response_error
            && self.branch_leakage_upper_bound <= limits.maximum_branch_leakage
            && self.transfer_flux_error_upper_bound <= limits.maximum_transfer_flux_error
            && self.invariant_error_upper_bound <= limits.maximum_invariant_error
            && self.dissipated_casimir_fraction <= limits.maximum_dissipated_casimir_fraction
            && self.estimated_wall_seconds <= limits.maximum_wall_seconds
            && self.estimated_peak_memory_bytes <= limits.maximum_peak_memory_bytes
    }

    fn dominates(self, other: Self) -> bool {
        let no_worse = self.future_response_error_upper_bound
            <= other.future_response_error_upper_bound
            && self.branch_leakage_upper_bound <= other.branch_leakage_upper_bound
            && self.transfer_flux_error_upper_bound <= other.transfer_flux_error_upper_bound
            && self.invariant_error_upper_bound <= other.invariant_error_upper_bound
            && self.dissipated_casimir_fraction <= other.dissipated_casimir_fraction
            && self.estimated_wall_seconds <= other.estimated_wall_seconds
            && self.estimated_peak_memory_bytes <= other.estimated_peak_memory_bytes;
        let strictly_better = self.future_response_error_upper_bound
            < other.future_response_error_upper_bound
            || self.branch_leakage_upper_bound < other.branch_leakage_upper_bound
            || self.transfer_flux_error_upper_bound < other.transfer_flux_error_upper_bound
            || self.invariant_error_upper_bound < other.invariant_error_upper_bound
            || self.dissipated_casimir_fraction < other.dissipated_casimir_fraction
            || self.estimated_wall_seconds < other.estimated_wall_seconds
            || self.estimated_peak_memory_bytes < other.estimated_peak_memory_bytes;
        no_worse && strictly_better
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> HorizonLimits {
        HorizonLimits {
            warning_guardband_fraction: 0.20,
            maximum_guardband_fraction: 0.40,
            warning_boundary_ratio: 0.01,
            maximum_boundary_ratio: 0.05,
            warning_response_error: 0.01,
            maximum_response_error: 0.03,
            warning_transfer_flux_error: 0.01,
            maximum_transfer_flux_error: 0.03,
            recurrence_warning_fraction: 0.80,
        }
    }

    fn evidence(time: f64) -> HorizonEvidence {
        HorizonEvidence {
            current_time: time,
            nyquist_encounter_time: 134.0,
            discrete_recurrence_time: 268.0,
            high_velocity_guardband_fraction: 0.05,
            velocity_boundary_ratio: 0.001,
            future_response_error_upper_bound: 0.002,
            transfer_flux_error_upper_bound: 0.003,
            calibration_domain_covered: true,
            velocity_support_valid: true,
        }
    }

    fn action(
        action: KineticAction,
        response: f64,
        dissipation: f64,
        wall: f64,
        memory: u64,
    ) -> ActionEstimate {
        ActionEstimate {
            action,
            future_response_error_upper_bound: response,
            branch_leakage_upper_bound: 0.002,
            transfer_flux_error_upper_bound: 0.003,
            invariant_error_upper_bound: 0.001,
            dissipated_casimir_fraction: dissipation,
            estimated_wall_seconds: wall,
            estimated_peak_memory_bytes: memory,
            calibration_domain_covered: true,
            velocity_support_valid: true,
            positivity_preserved: true,
        }
    }

    fn constraints() -> ActionConstraints {
        ActionConstraints {
            maximum_future_response_error: 0.01,
            maximum_branch_leakage: 0.01,
            maximum_transfer_flux_error: 0.01,
            maximum_invariant_error: 0.01,
            maximum_dissipated_casimir_fraction: 0.01,
            maximum_wall_seconds: 100.0,
            maximum_peak_memory_bytes: 1_000_000_000,
        }
    }

    #[test]
    fn nyquist_encounter_warns_but_does_not_expire_a_measured_branch() {
        let assessment = evidence(162.2).assess(limits()).unwrap();
        assert_eq!(assessment.status, HorizonStatus::Warning);
        assert!(assessment.reasons.contains(&HorizonReason::NyquistEncountered));
        assert!(!assessment.reasons.contains(&HorizonReason::DiscreteRecurrenceReached));
    }

    #[test]
    fn discrete_recurrence_reached_returns_unknown() {
        let assessment = evidence(268.0).assess(limits()).unwrap();
        assert_eq!(assessment.status, HorizonStatus::Unknown);
        assert!(
            assessment
                .reasons
                .contains(&HorizonReason::DiscreteRecurrenceReached)
        );
    }

    #[test]
    fn out_of_calibration_domain_returns_unknown() {
        let mut state = evidence(20.0);
        state.calibration_domain_covered = false;
        let assessment = state.assess(limits()).unwrap();
        assert_eq!(assessment.status, HorizonStatus::Unknown);
        assert!(
            assessment
                .reasons
                .contains(&HorizonReason::CalibrationDomainMissing)
        );
    }

    #[test]
    fn high_guardband_or_invalid_support_returns_unknown() {
        let mut high_guardband = evidence(20.0);
        high_guardband.high_velocity_guardband_fraction = 0.45;
        assert_eq!(
            high_guardband.assess(limits()).unwrap().status,
            HorizonStatus::Unknown
        );

        let mut invalid_support = evidence(20.0);
        invalid_support.velocity_support_valid = false;
        assert_eq!(
            invalid_support.assess(limits()).unwrap().status,
            HorizonStatus::Unknown
        );
    }

    #[test]
    fn invalid_or_nonfinite_evidence_fails_closed() {
        let mut invalid = evidence(20.0);
        invalid.high_velocity_guardband_fraction = f64::NAN;
        assert_eq!(
            invalid.assess(limits()),
            Err(PolicyInputError::NonFiniteEvidence)
        );
        let mut invalid_times = evidence(20.0);
        invalid_times.discrete_recurrence_time = invalid_times.nyquist_encounter_time;
        assert_eq!(
            invalid_times.assess(limits()),
            Err(PolicyInputError::InvalidTimeScales)
        );
    }

    #[test]
    fn action_policy_abstains_when_no_calibrated_candidate_exists() {
        let mut candidate = action(KineticAction::VelocityFourierSponge, 0.002, 0.001, 2.0, 10);
        candidate.calibration_domain_covered = false;
        let decision = ActionDecision::select(&[candidate], constraints()).unwrap();
        assert_eq!(decision.recommended_index, None);
        assert_eq!(
            decision.abstention_reason,
            Some(ActionAbstentionReason::NoCalibratedCandidate)
        );
    }

    #[test]
    fn action_policy_abstains_when_all_candidates_violate_constraints() {
        let candidate = action(KineticAction::VelocityFourierSponge, 0.02, 0.001, 2.0, 10);
        let decision = ActionDecision::select(&[candidate], constraints()).unwrap();
        assert_eq!(decision.recommended_index, None);
        assert_eq!(
            decision.abstention_reason,
            Some(ActionAbstentionReason::NoCandidateMeetsConstraints)
        );
    }

    #[test]
    fn action_policy_returns_pareto_front_not_a_weighted_score() {
        let conservative = action(
            KineticAction::ConservativeHermiteTransfer,
            0.001,
            0.0,
            10.0,
            500_000_000,
        );
        let sponge = action(KineticAction::VelocityFourierSponge, 0.002, 0.003, 2.0, 100_000_000);
        let dominated = action(KineticAction::RefineVelocityGrid, 0.003, 0.004, 12.0, 600_000_000);
        let decision =
            ActionDecision::select(&[conservative, sponge, dominated], constraints()).unwrap();
        assert_eq!(decision.pareto_front_indices, vec![0, 1]);
        assert_eq!(decision.recommended_index, Some(1));
        assert_eq!(decision.abstention_reason, None);
    }

    #[test]
    fn invalid_candidate_metrics_are_never_selected() {
        let mut candidate =
            action(KineticAction::RefineVelocityGrid, 0.001, 0.0, 1.0, 100_000_000);
        candidate.future_response_error_upper_bound = f64::NAN;
        let decision = ActionDecision::select(&[candidate], constraints()).unwrap();
        assert_eq!(decision.recommended_index, None);
    }

    #[test]
    fn positivity_failure_blocks_an_otherwise_cheap_candidate() {
        let mut candidate =
            action(KineticAction::VelocityFourierSponge, 0.001, 0.001, 0.1, 10);
        candidate.positivity_preserved = false;
        let decision = ActionDecision::select(&[candidate], constraints()).unwrap();
        assert_eq!(decision.recommended_index, None);
    }
}
