//! Deterministic reference for a calibrated decision-sufficiency horizon.
//!
//! The evaluator consumes externally produced loss bounds and admissibility
//! evidence. It does not estimate beliefs, losses, calibration, viability or
//! physical safety, and it never authorizes actuator commands.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

/// Tri-state robust-admissibility evidence for an action at one future tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admissibility {
    /// The action is proven admissible within the declared evidence scope.
    Yes,
    /// The action is proven inadmissible within the declared evidence scope.
    No,
    /// Evidence is missing, stale, contradictory or otherwise insufficient.
    Unknown,
}

/// A closed interval for expected task loss in a caller-declared fixed-point unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossInterval {
    /// Lower bound on expected loss.
    pub lower_units: u64,
    /// Upper bound on expected loss.
    pub upper_units: u64,
}

impl LossInterval {
    /// Construct a loss interval. Ordering is validated by the evaluator.
    pub const fn new(lower_units: u64, upper_units: u64) -> Self {
        Self {
            lower_units,
            upper_units,
        }
    }

    fn validate(self) -> Result<(), ReferenceError> {
        if self.lower_units > self.upper_units {
            return Err(ReferenceError::InvalidLossInterval {
                lower_units: self.lower_units,
                upper_units: self.upper_units,
            });
        }
        Ok(())
    }
}

/// Semantics of the coverage guarantee for loss intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageSemantics {
    /// Joint/family-wise coverage for all action intervals in the declared set.
    SimultaneousAcrossActionSet,
    /// Separate marginal coverage for each action; insufficient for a minimum over alternatives.
    MarginalPerAction,
}

/// Loss-model calibration evidence supplied by an independent evaluator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationEvidence {
    /// Stable calibration identity.
    pub calibration_id: String,
    /// Digest of the independent validation evidence.
    pub validation_digest: String,
    /// Loss-model digest covered by this calibration.
    pub loss_model_digest: String,
    /// Versioned task-loss scale covered by this calibration.
    pub loss_scale_id: String,
    /// Declared empirical coverage in parts per million.
    pub coverage_ppm: u32,
    /// Whether coverage holds jointly across the candidate action set.
    pub coverage_semantics: CoverageSemantics,
    /// Largest candidate-set cardinality covered by the calibration study.
    pub maximum_action_count: u32,
    /// Maximum contiguous decision-horizon length covered by calibration.
    pub maximum_horizon_ticks: u64,
    /// Frozen candidate action-domain manifest digest covered by calibration.
    pub action_domain_digest: String,
    /// Last control tick for which this calibration evidence is valid.
    pub valid_through_tick: u64,
}

/// Evidence for one candidate action at one future control tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionEvidence {
    /// Stable action identity within the current candidate set.
    pub action_id: String,
    /// Robust-admissibility status under the upstream qualified contract.
    pub admissibility: Admissibility,
    /// Expected-loss interval. Required when the action is admissible.
    pub expected_loss: Option<LossInterval>,
}

/// Evidence available at one future control tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionTickEvidence {
    /// Monotonic control-tick index within the declared robot epoch.
    pub tick: u64,
    /// Identity of the action selected at the original decision point.
    pub selected_action_id: String,
    /// Whether the supplied candidate set is complete for this decision scope.
    pub action_set_complete: bool,
    /// Whether all bound/evidence digests match the declared input identities.
    pub evidence_digest_matches: bool,
    /// Digest of the evidence used for this tick.
    pub evidence_digest: String,
    /// Versioned expected-loss unit scale.
    pub loss_scale_id: String,
    /// Expected-loss model identity.
    pub loss_model_digest: String,
    /// Frozen candidate action-domain manifest digest for this decision scope.
    pub action_domain_digest: String,
    /// Independent calibration evidence for the supplied expected-loss bounds.
    pub calibration: Option<CalibrationEvidence>,
    /// Complete set of candidate actions and their evidence.
    pub actions: Vec<ActionEvidence>,
}

/// Contract governing a sequence of decision-sufficiency evaluations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HorizonContract {
    /// Maximum tolerated stale-action regret in fixed-point task-loss units.
    pub epsilon_task_loss_units: u64,
    /// Required task-loss scale identity.
    pub loss_scale_id: String,
    /// Required expected-loss model digest.
    pub loss_model_digest: String,
    /// Minimum declared calibration coverage in parts per million.
    pub minimum_coverage_ppm: u32,
}

impl HorizonContract {
    fn validate(&self) -> Result<(), ReferenceError> {
        if self.loss_scale_id.trim().is_empty() {
            return Err(ReferenceError::EmptyField { field: "loss_scale_id" });
        }
        if self.loss_model_digest.trim().is_empty() {
            return Err(ReferenceError::EmptyField {
                field: "loss_model_digest",
            });
        }
        if self.minimum_coverage_ppm == 0 || self.minimum_coverage_ppm > 1_000_000 {
            return Err(ReferenceError::InvalidContractCoverage {
                coverage_ppm: self.minimum_coverage_ppm,
            });
        }
        Ok(())
    }
}

/// Reason a tick cannot establish decision sufficiency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownReason {
    /// The action set was not declared complete.
    IncompleteActionSet,
    /// Evidence digest is absent or does not match the declared inputs.
    EvidenceDigestUnavailable,
    /// The current loss model differs from the contract.
    LossModelMismatch,
    /// No calibration evidence was supplied.
    MissingCalibration,
    /// Calibration identity or covered model/scale does not match.
    CalibrationIdentityMismatch,
    /// Calibration coverage is below the declared contract requirement.
    CalibrationCoverageInsufficient {
        /// Observed declared coverage.
        observed_ppm: u32,
        /// Minimum contract coverage.
        required_ppm: u32,
    },
    /// Calibration is only marginal per action, not simultaneous over the set.
    CalibrationCoverageNotSimultaneous,
    /// Candidate action set exceeds the calibration study's covered cardinality.
    CalibrationActionSetTooLarge {
        /// Number of actions in the supplied candidate set.
        observed_count: usize,
        /// Maximum calibrated candidate-set size.
        maximum_count: u32,
    },
    /// Candidate action-domain manifest does not match the calibration scope.
    CalibrationActionDomainMismatch,
    /// Evaluation horizon exceeds the calibrated horizon envelope.
    HorizonExceedsCalibrationScope {
        /// Number of supplied horizon ticks.
        observed_ticks: usize,
        /// Maximum calibrated horizon length.
        maximum_ticks: u64,
    },
    /// The originally selected action identity changed during the horizon.
    SelectedActionIdentityChanged {
        /// Action selected at the first horizon tick.
        expected_action_id: String,
        /// Action identity supplied at the changed tick.
        observed_action_id: String,
    },
    /// Calibration evidence has expired before this tick.
    CalibrationExpired {
        /// Last valid tick in the calibration receipt.
        valid_through_tick: u64,
        /// Tick being assessed.
        assessed_tick: u64,
    },
    /// The selected action has unknown admissibility.
    SelectedActionAdmissibilityUnknown,
    /// At least one alternative has unknown admissibility.
    AlternativeAdmissibilityUnknown,
    /// An admissible action is missing its expected-loss interval.
    MissingExpectedLossBound,
    /// The supplied tick sequence has a gap or is out of order.
    NonContiguousTick {
        /// Tick expected after the preceding assessment.
        expected_tick: u64,
        /// Next supplied tick, if one exists.
        observed_tick: Option<u64>,
    },
}

/// Reason the selected action is known not to remain decision-sufficient.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsufficiencyReason {
    /// The previously selected action is no longer robustly admissible.
    SelectedActionNotAdmissible,
    /// Conservative stale-action regret exceeds the task threshold.
    RegretExceeded {
        /// Conservative upper bound on stale-action regret.
        regret_upper_units: u64,
        /// Contract threshold.
        epsilon_units: u64,
    },
}

/// Outcome for one control tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TickDisposition {
    /// The selected action remains sufficiently good and robustly admissible.
    Sufficient {
        /// Conservative upper bound on stale-action regret.
        regret_upper_units: u64,
    },
    /// The selected action is known to fail a decision-sufficiency condition.
    Insufficient {
        /// Deterministic reason for ending the horizon.
        reason: InsufficiencyReason,
        /// Regret bound, when it could be evaluated.
        regret_upper_units: Option<u64>,
    },
    /// Required evidence is absent or inconclusive.
    Unknown {
        /// Deterministic evidence limitation.
        reason: UnknownReason,
    },
}

/// Assessment of one control tick in the requested horizon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TickAssessment {
    /// Tick index assessed (or expected, for a missing tick).
    pub tick: u64,
    /// Result for this tick.
    pub disposition: TickDisposition,
}

/// Why the contiguous decision-sufficiency horizon stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HorizonStopReason {
    /// Stale-action regret exceeded the task threshold.
    RegretExceeded,
    /// The selected action ceased to be robustly admissible.
    SelectedActionNotAdmissible,
    /// Required evidence was unknown or unavailable.
    EvidenceUnknown(UnknownReason),
    /// The supplied sequence skipped or reordered a control tick.
    NonContiguousTick,
}

/// Result of evaluating the longest contiguous decision-sufficient prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HorizonResult {
    /// First tick in the supplied sequence.
    pub first_tick: u64,
    /// Last tick established as decision-sufficient, if any.
    pub last_sufficient_tick: Option<u64>,
    /// First tick that stopped the horizon, if any.
    pub stopped_at_tick: Option<u64>,
    /// Stop reason, or None when every supplied tick was sufficient.
    pub stop_reason: Option<HorizonStopReason>,
    /// Maximum regret upper bound across evaluated sufficient ticks.
    pub max_regret_upper_units: Option<u64>,
    /// Per-tick evidence and disposition.
    pub assessments: Vec<TickAssessment>,
}

/// Structural or arithmetic error in the supplied reference inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceError {
    /// Contract contains an empty identity or invalid threshold.
    EmptyField {
        /// Name of the invalid field.
        field: &'static str,
    },
    /// Calibration threshold is outside the valid parts-per-million range.
    InvalidContractCoverage {
        /// Invalid coverage value.
        coverage_ppm: u32,
    },
    /// A supplied coverage value is outside the valid parts-per-million range.
    InvalidEvidenceCoverage {
        /// Invalid coverage value.
        coverage_ppm: u32,
    },
    /// Loss interval has its lower endpoint above its upper endpoint.
    InvalidLossInterval {
        /// Supplied lower endpoint.
        lower_units: u64,
        /// Supplied upper endpoint.
        upper_units: u64,
    },
    /// A candidate action has an empty identity.
    EmptyActionId,
    /// Candidate action identity is duplicated in one tick.
    DuplicateActionId {
        /// Duplicated identity.
        action_id: String,
    },
    /// Selected action is missing from a declared complete candidate set.
    SelectedActionMissing {
        /// Selected action identity.
        action_id: String,
    },
    /// The tick loss scale conflicts with the contract's task-loss units.
    LossScaleMismatch {
        /// Tick containing the mismatch.
        tick: u64,
    },
    /// The requested horizon contains no ticks.
    EmptyHorizon,
    /// Incrementing the final tick would overflow the tick representation.
    TickOverflow,
}

impl std::fmt::Display for ReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ReferenceError {}

fn assess_tick(
    contract: &HorizonContract,
    evidence: &DecisionTickEvidence,
) -> Result<TickDisposition, ReferenceError> {
    if evidence.loss_scale_id != contract.loss_scale_id {
        return Err(ReferenceError::LossScaleMismatch { tick: evidence.tick });
    }
    if evidence.action_domain_digest.trim().is_empty() {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::CalibrationActionDomainMismatch,
        });
    }
    if evidence.evidence_digest.trim().is_empty() || !evidence.evidence_digest_matches {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::EvidenceDigestUnavailable,
        });
    }
    if evidence.loss_model_digest != contract.loss_model_digest {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::LossModelMismatch,
        });
    }
    if !evidence.action_set_complete {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::IncompleteActionSet,
        });
    }

    let calibration = match &evidence.calibration {
        Some(value) => value,
        None => {
            return Ok(TickDisposition::Unknown {
                reason: UnknownReason::MissingCalibration,
            });
        }
    };
    if calibration.calibration_id.trim().is_empty()
        || calibration.validation_digest.trim().is_empty()
        || calibration.loss_model_digest != contract.loss_model_digest
        || calibration.loss_scale_id != contract.loss_scale_id
        || calibration.action_domain_digest.trim().is_empty()
        || calibration.action_domain_digest != evidence.action_domain_digest
    {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::CalibrationIdentityMismatch,
        });
    }
    if calibration.coverage_ppm > 1_000_000 {
        return Err(ReferenceError::InvalidEvidenceCoverage {
            coverage_ppm: calibration.coverage_ppm,
        });
    }
    if calibration.coverage_ppm < contract.minimum_coverage_ppm {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::CalibrationCoverageInsufficient {
                observed_ppm: calibration.coverage_ppm,
                required_ppm: contract.minimum_coverage_ppm,
            },
        });
    }
    if calibration.coverage_semantics != CoverageSemantics::SimultaneousAcrossActionSet {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::CalibrationCoverageNotSimultaneous,
        });
    }
    if evidence.actions.len() > calibration.maximum_action_count as usize {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::CalibrationActionSetTooLarge {
                observed_count: evidence.actions.len(),
                maximum_count: calibration.maximum_action_count,
            },
        });
    }
    if calibration.valid_through_tick < evidence.tick {
        return Ok(TickDisposition::Unknown {
            reason: UnknownReason::CalibrationExpired {
                valid_through_tick: calibration.valid_through_tick,
                assessed_tick: evidence.tick,
            },
        });
    }

    let mut ids = BTreeSet::new();
    for action in &evidence.actions {
        if action.action_id.trim().is_empty() {
            return Err(ReferenceError::EmptyActionId);
        }
        if !ids.insert(action.action_id.as_str()) {
            return Err(ReferenceError::DuplicateActionId {
                action_id: action.action_id.clone(),
            });
        }
        if let Some(interval) = action.expected_loss {
            interval.validate()?;
        }
    }

    let selected = match evidence
        .actions
        .iter()
        .find(|action| action.action_id == evidence.selected_action_id)
    {
        Some(value) => value,
        None => {
            return Err(ReferenceError::SelectedActionMissing {
                action_id: evidence.selected_action_id.clone(),
            });
        }
    };

    match selected.admissibility {
        Admissibility::No => {
            return Ok(TickDisposition::Insufficient {
                reason: InsufficiencyReason::SelectedActionNotAdmissible,
                regret_upper_units: None,
            });
        }
        Admissibility::Unknown => {
            return Ok(TickDisposition::Unknown {
                reason: UnknownReason::SelectedActionAdmissibilityUnknown,
            });
        }
        Admissibility::Yes => {}
    }

    let selected_loss = match selected.expected_loss {
        Some(value) => value,
        None => {
            return Ok(TickDisposition::Unknown {
                reason: UnknownReason::MissingExpectedLossBound,
            });
        }
    };

    let mut best_alternative_lower: Option<u64> = None;
    for action in &evidence.actions {
        if action.action_id == evidence.selected_action_id {
            continue;
        }
        match action.admissibility {
            Admissibility::No => {}
            Admissibility::Unknown => {
                return Ok(TickDisposition::Unknown {
                    reason: UnknownReason::AlternativeAdmissibilityUnknown,
                });
            }
            Admissibility::Yes => {
                let interval = match action.expected_loss {
                    Some(value) => value,
                    None => {
                        return Ok(TickDisposition::Unknown {
                            reason: UnknownReason::MissingExpectedLossBound,
                        });
                    }
                };
                best_alternative_lower = Some(match best_alternative_lower {
                    Some(current) => current.min(interval.lower_units),
                    None => interval.lower_units,
                });
            }
        }
    }

    let regret_upper_units = match best_alternative_lower {
        Some(lower) => selected_loss.upper_units.saturating_sub(lower),
        None => 0,
    };

    if regret_upper_units <= contract.epsilon_task_loss_units {
        Ok(TickDisposition::Sufficient {
            regret_upper_units,
        })
    } else {
        Ok(TickDisposition::Insufficient {
            reason: InsufficiencyReason::RegretExceeded {
                regret_upper_units,
                epsilon_units: contract.epsilon_task_loss_units,
            },
            regret_upper_units: Some(regret_upper_units),
        })
    }
}

/// Evaluate the longest contiguous prefix for which the old decision remains
/// robustly admissible and its calibrated regret upper bound stays within epsilon.
///
/// The evaluator stops at the first insufficient or unknown tick. A gap in the
/// tick sequence is represented as Unknown at the first missing tick. This
/// function returns no actuation permission or physical-safety certificate.
pub fn decision_sufficiency_horizon(
    contract: &HorizonContract,
    steps: &[DecisionTickEvidence],
) -> Result<HorizonResult, ReferenceError> {
    contract.validate()?;
    let first = steps.first().ok_or(ReferenceError::EmptyHorizon)?;
    if let Some(first_calibration) = first.calibration.as_ref() {
        if steps.len() as u64 > first_calibration.maximum_horizon_ticks {
            return Ok(HorizonResult {
            first_tick: first.tick,
            last_sufficient_tick: None,
            stopped_at_tick: Some(first.tick),
            stop_reason: Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::HorizonExceedsCalibrationScope {
                    observed_ticks: steps.len(),
                    maximum_ticks: first_calibration.maximum_horizon_ticks,
                },
            )),
            max_regret_upper_units: None,
                assessments: vec![TickAssessment {
                    tick: first.tick,
                    disposition: TickDisposition::Unknown {
                        reason: UnknownReason::HorizonExceedsCalibrationScope {
                            observed_ticks: steps.len(),
                            maximum_ticks: first_calibration.maximum_horizon_ticks,
                        },
                    },
                }],
            });
        }
    }

    let anchor_action_id = first.selected_action_id.as_str();
    let anchor_domain_digest = first.action_domain_digest.as_str();

    let mut result = HorizonResult {
        first_tick: first.tick,
        last_sufficient_tick: None,
        stopped_at_tick: None,
        stop_reason: None,
        max_regret_upper_units: None,
        assessments: Vec::with_capacity(steps.len()),
    };

    let mut previous_tick: Option<u64> = None;
    for evidence in steps {
        if let Some(previous) = previous_tick {
            let expected_tick = previous
                .checked_add(1)
                .ok_or(ReferenceError::TickOverflow)?;
            if evidence.tick != expected_tick {
                let disposition = TickDisposition::Unknown {
                    reason: UnknownReason::NonContiguousTick {
                        expected_tick,
                        observed_tick: Some(evidence.tick),
                    },
                };
                result.assessments.push(TickAssessment {
                    tick: expected_tick,
                    disposition,
                });
                result.stopped_at_tick = Some(expected_tick);
                result.stop_reason = Some(HorizonStopReason::NonContiguousTick);
                return Ok(result);
            }
        }

        if evidence.selected_action_id != anchor_action_id {
            let reason = UnknownReason::SelectedActionIdentityChanged {
                expected_action_id: anchor_action_id.to_owned(),
                observed_action_id: evidence.selected_action_id.clone(),
            };
            result.assessments.push(TickAssessment {
                tick: evidence.tick,
                disposition: TickDisposition::Unknown {
                    reason: reason.clone(),
                },
            });
            result.stopped_at_tick = Some(evidence.tick);
            result.stop_reason = Some(HorizonStopReason::EvidenceUnknown(reason));
            return Ok(result);
        }
        if evidence.action_domain_digest != anchor_domain_digest {
            let reason = UnknownReason::CalibrationActionDomainMismatch;
            result.assessments.push(TickAssessment {
                tick: evidence.tick,
                disposition: TickDisposition::Unknown {
                    reason: reason.clone(),
                },
            });
            result.stopped_at_tick = Some(evidence.tick);
            result.stop_reason = Some(HorizonStopReason::EvidenceUnknown(reason));
            return Ok(result);
        }

        if let Some(calibration) = evidence.calibration.as_ref() {
            if steps.len() as u64 > calibration.maximum_horizon_ticks {
                let reason = UnknownReason::HorizonExceedsCalibrationScope {
                    observed_ticks: steps.len(),
                    maximum_ticks: calibration.maximum_horizon_ticks,
                };
                result.assessments.push(TickAssessment {
                    tick: evidence.tick,
                    disposition: TickDisposition::Unknown {
                        reason: reason.clone(),
                    },
                });
                result.stopped_at_tick = Some(evidence.tick);
                result.stop_reason = Some(HorizonStopReason::EvidenceUnknown(reason));
                return Ok(result);
            }
        }

        let disposition = assess_tick(contract, evidence)?;
        let current_tick = evidence.tick;
        match &disposition {
            TickDisposition::Sufficient {
                regret_upper_units,
            } => {
                result.last_sufficient_tick = Some(current_tick);
                result.max_regret_upper_units = Some(
                    result
                        .max_regret_upper_units
                        .unwrap_or(0)
                        .max(*regret_upper_units),
                );
            }
            TickDisposition::Insufficient { reason, .. } => {
                result.stopped_at_tick = Some(current_tick);
                result.stop_reason = Some(match reason {
                    InsufficiencyReason::SelectedActionNotAdmissible => {
                        HorizonStopReason::SelectedActionNotAdmissible
                    }
                    InsufficiencyReason::RegretExceeded { .. } => {
                        HorizonStopReason::RegretExceeded
                    }
                });
                result.assessments.push(TickAssessment {
                    tick: current_tick,
                    disposition,
                });
                return Ok(result);
            }
            TickDisposition::Unknown { reason } => {
                result.stopped_at_tick = Some(current_tick);
                result.stop_reason =
                    Some(HorizonStopReason::EvidenceUnknown(reason.clone()));
                result.assessments.push(TickAssessment {
                    tick: current_tick,
                    disposition,
                });
                return Ok(result);
            }
        }
        result.assessments.push(TickAssessment {
            tick: current_tick,
            disposition,
        });
        previous_tick = Some(current_tick);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> HorizonContract {
        HorizonContract {
            epsilon_task_loss_units: 3,
            loss_scale_id: "task-loss-v1".to_owned(),
            loss_model_digest: "loss-model-sha256".to_owned(),
            minimum_coverage_ppm: 950_000,
        }
    }

    fn calibration(valid_through_tick: u64) -> CalibrationEvidence {
        CalibrationEvidence {
            calibration_id: "calibration-v1".to_owned(),
            validation_digest: "validation-sha256".to_owned(),
            loss_model_digest: "loss-model-sha256".to_owned(),
            loss_scale_id: "task-loss-v1".to_owned(),
            coverage_ppm: 990_000,
            coverage_semantics: CoverageSemantics::SimultaneousAcrossActionSet,
            maximum_action_count: 8,
            maximum_horizon_ticks: 100,
            action_domain_digest: "action-domain-v1".to_owned(),
            valid_through_tick,
        }
    }

    fn step(
        tick: u64,
        selected_admissibility: Admissibility,
        selected_loss: Option<LossInterval>,
        alternative: Option<ActionEvidence>,
    ) -> DecisionTickEvidence {
        let mut actions = vec![ActionEvidence {
            action_id: "a0".to_owned(),
            admissibility: selected_admissibility,
            expected_loss: selected_loss,
        }];
        if let Some(action) = alternative {
            actions.push(action);
        }
        DecisionTickEvidence {
            tick,
            selected_action_id: "a0".to_owned(),
            action_set_complete: true,
            evidence_digest_matches: true,
            evidence_digest: "decision-evidence-sha256".to_owned(),
            loss_scale_id: "task-loss-v1".to_owned(),
            loss_model_digest: "loss-model-sha256".to_owned(),
            action_domain_digest: "action-domain-v1".to_owned(),
            calibration: Some(calibration(100)),
            actions,
        }
    }

    fn alternative(
        admissibility: Admissibility,
        loss: Option<LossInterval>,
    ) -> ActionEvidence {
        ActionEvidence {
            action_id: "a1".to_owned(),
            admissibility,
            expected_loss: loss,
        }
    }

    #[test]
    fn old_observation_can_remain_sufficient_when_decision_regret_is_low() {
        let steps = vec![
            step(
                10,
                Admissibility::Yes,
                Some(LossInterval::new(8, 10)),
                Some(alternative(
                    Admissibility::Yes,
                    Some(LossInterval::new(8, 12)),
                )),
            ),
            step(
                11,
                Admissibility::Yes,
                Some(LossInterval::new(8, 10)),
                Some(alternative(
                    Admissibility::Yes,
                    Some(LossInterval::new(8, 12)),
                )),
            ),
        ];
        let result = decision_sufficiency_horizon(&contract(), &steps).unwrap();
        assert_eq!(result.last_sufficient_tick, Some(11));
        assert_eq!(result.stopped_at_tick, None);
        assert_eq!(result.max_regret_upper_units, Some(2));
    }

    #[test]
    fn fresh_decision_change_ends_the_old_action_horizon() {
        let steps = vec![
            step(
                1,
                Admissibility::Yes,
                Some(LossInterval::new(4, 5)),
                Some(alternative(
                    Admissibility::Yes,
                    Some(LossInterval::new(5, 7)),
                )),
            ),
            step(
                2,
                Admissibility::Yes,
                Some(LossInterval::new(8, 9)),
                Some(alternative(
                    Admissibility::Yes,
                    Some(LossInterval::new(2, 3)),
                )),
            ),
        ];
        let result = decision_sufficiency_horizon(&contract(), &steps).unwrap();
        assert_eq!(result.last_sufficient_tick, Some(1));
        assert_eq!(result.stopped_at_tick, Some(2));
        assert_eq!(result.stop_reason, Some(HorizonStopReason::RegretExceeded));
        assert_eq!(
            result.assessments[1].disposition,
            TickDisposition::Insufficient {
                reason: InsufficiencyReason::RegretExceeded {
                    regret_upper_units: 7,
                    epsilon_units: 3,
                },
                regret_upper_units: Some(7),
            }
        );
    }

    #[test]
    fn selected_action_must_remain_robustly_admissible() {
        let steps = vec![step(
            1,
            Admissibility::No,
            Some(LossInterval::new(1, 2)),
            None,
        )];
        let result = decision_sufficiency_horizon(&contract(), &steps).unwrap();
        assert_eq!(result.last_sufficient_tick, None);
        assert_eq!(
            result.stop_reason,
            Some(HorizonStopReason::SelectedActionNotAdmissible)
        );
    }

    #[test]
    fn unknown_selected_action_admissibility_is_not_a_pass() {
        let steps = vec![step(
            1,
            Admissibility::Unknown,
            Some(LossInterval::new(1, 2)),
            None,
        )];
        let result = decision_sufficiency_horizon(&contract(), &steps).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::SelectedActionAdmissibilityUnknown
            ))
        ));
    }

    #[test]
    fn unknown_alternative_admissibility_is_not_an_empty_action_set() {
        let steps = vec![step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            Some(alternative(Admissibility::Unknown, None)),
        )];
        let result = decision_sufficiency_horizon(&contract(), &steps).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::AlternativeAdmissibilityUnknown
            ))
        ));
    }

    #[test]
    fn incomplete_action_set_returns_unknown() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        evidence.action_set_complete = false;
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::IncompleteActionSet
            ))
        ));
    }

    #[test]
    fn one_proven_admissible_action_has_zero_regret_when_set_is_complete() {
        let evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(7, 10)),
            None,
        );
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert_eq!(result.last_sufficient_tick, Some(1));
        assert_eq!(result.max_regret_upper_units, Some(0));
    }

    #[test]
    fn exact_epsilon_boundary_is_sufficient() {
        let evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(5, 8)),
            Some(alternative(
                Admissibility::Yes,
                Some(LossInterval::new(5, 5)),
            )),
        );
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert_eq!(result.last_sufficient_tick, Some(1));
        assert_eq!(result.max_regret_upper_units, Some(3));
    }

    #[test]
    fn zero_evsi_or_aoi_are_not_used_as_regret_inputs() {
        // The evaluator intentionally consumes only action-loss/admissibility
        // evidence. AoI and EVSI are separate upstream quantities.
        let evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(4, 5)),
            Some(alternative(
                Admissibility::Yes,
                Some(LossInterval::new(5, 6)),
            )),
        );
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert_eq!(result.last_sufficient_tick, Some(1));
    }

    #[test]
    fn loss_scale_mismatch_is_rejected() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        evidence.loss_scale_id = "other-task-scale".to_owned();
        assert_eq!(
            decision_sufficiency_horizon(&contract(), &[evidence]),
            Err(ReferenceError::LossScaleMismatch { tick: 1 })
        );
    }

    #[test]
    fn missing_calibration_returns_unknown() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        evidence.calibration = None;
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::MissingCalibration
            ))
        ));
    }

    #[test]
    fn expired_calibration_returns_unknown() {
        let evidence = step(
            5,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        let mut evidence = evidence;
        evidence.calibration = Some(calibration(4));
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::CalibrationExpired { .. }
            ))
        ));
    }

    #[test]
    fn insufficient_calibration_coverage_returns_unknown() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        let mut receipt = calibration(100);
        receipt.coverage_ppm = 900_000;
        evidence.calibration = Some(receipt);
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::CalibrationCoverageInsufficient { .. }
            ))
        ));
    }

    #[test]
    fn marginal_action_coverage_cannot_bound_the_minimum_alternative() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            Some(alternative(
                Admissibility::Yes,
                Some(LossInterval::new(1, 2)),
            )),
        );
        evidence.calibration.as_mut().unwrap().coverage_semantics =
            CoverageSemantics::MarginalPerAction;
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::CalibrationCoverageNotSimultaneous
            ))
        ));
    }

    #[test]
    fn action_set_larger_than_calibrated_envelope_returns_unknown() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            Some(alternative(
                Admissibility::Yes,
                Some(LossInterval::new(1, 2)),
            )),
        );
        evidence.calibration.as_mut().unwrap().maximum_action_count = 1;
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::CalibrationActionSetTooLarge { .. }
            ))
        ));
    }

    #[test]
    fn changed_selected_action_identity_terminates_the_same_horizon() {
        let first = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        let mut second = step(
            2,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        second.selected_action_id = "a1".to_owned();
        second.actions.push(ActionEvidence {
            action_id: "a1".to_owned(),
            admissibility: Admissibility::Yes,
            expected_loss: Some(LossInterval::new(1, 2)),
        });
        let result = decision_sufficiency_horizon(&contract(), &[first, second]).unwrap();
        assert_eq!(result.last_sufficient_tick, Some(1));
        assert_eq!(result.stopped_at_tick, Some(2));
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::SelectedActionIdentityChanged { .. }
            ))
        ));
    }

    #[test]
    fn action_domain_digest_must_match_calibration() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        evidence.action_domain_digest = "different-domain".to_owned();
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::CalibrationActionDomainMismatch
            ))
        ));
    }

    #[test]
    fn horizon_cannot_exceed_calibrated_scope() {
        let mut first = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        first.calibration.as_mut().unwrap().maximum_horizon_ticks = 1;
        let second = step(
            2,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        let result = decision_sufficiency_horizon(&contract(), &[first, second]).unwrap();
        assert_eq!(result.last_sufficient_tick, None);
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::HorizonExceedsCalibrationScope { .. }
            ))
        ));
    }

    #[test]
    fn digest_mismatch_returns_unknown() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        evidence.evidence_digest_matches = false;
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::EvidenceDigestUnavailable
            ))
        ));
    }

    #[test]
    fn model_digest_mismatch_returns_unknown() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        evidence.loss_model_digest = "different-model".to_owned();
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::LossModelMismatch
            ))
        ));
    }

    #[test]
    fn inadmissible_alternatives_do_not_compete_with_selected_action() {
        let evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(4, 5)),
            Some(alternative(
                Admissibility::No,
                Some(LossInterval::new(0, 1)),
            )),
        );
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert_eq!(result.last_sufficient_tick, Some(1));
        assert_eq!(result.max_regret_upper_units, Some(0));
    }

    #[test]
    fn malformed_loss_interval_is_rejected() {
        let evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(10, 2)),
            None,
        );
        assert_eq!(
            decision_sufficiency_horizon(&contract(), &[evidence]),
            Err(ReferenceError::InvalidLossInterval {
                lower_units: 10,
                upper_units: 2,
            })
        );
    }

    #[test]
    fn sequence_gap_stops_at_first_missing_tick() {
        let first = step(
            10,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        let second = step(
            12,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        let result = decision_sufficiency_horizon(&contract(), &[first, second]).unwrap();
        assert_eq!(result.last_sufficient_tick, Some(10));
        assert_eq!(result.stopped_at_tick, Some(11));
        assert_eq!(result.stop_reason, Some(HorizonStopReason::NonContiguousTick));
        assert!(matches!(
            &result.assessments[1].disposition,
            TickDisposition::Unknown {
                reason: UnknownReason::NonContiguousTick { .. }
            }
        ));
    }

    #[test]
    fn empty_horizon_is_rejected() {
        assert_eq!(
            decision_sufficiency_horizon(&contract(), &[]),
            Err(ReferenceError::EmptyHorizon)
        );
    }

    #[test]
    fn duplicate_action_identity_is_rejected() {
        let mut evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            None,
        );
        evidence.actions.push(ActionEvidence {
            action_id: "a0".to_owned(),
            admissibility: Admissibility::No,
            expected_loss: None,
        });
        assert_eq!(
            decision_sufficiency_horizon(&contract(), &[evidence]),
            Err(ReferenceError::DuplicateActionId {
                action_id: "a0".to_owned(),
            })
        );
    }

    #[test]
    fn selected_action_missing_from_complete_set_is_rejected() {
        let evidence = DecisionTickEvidence {
            tick: 1,
            selected_action_id: "a0".to_owned(),
            action_set_complete: true,
            evidence_digest_matches: true,
            evidence_digest: "decision-evidence-sha256".to_owned(),
            loss_scale_id: "task-loss-v1".to_owned(),
            loss_model_digest: "loss-model-sha256".to_owned(),
            action_domain_digest: "action-domain-v1".to_owned(),
            calibration: Some(calibration(100)),
            actions: Vec::new(),
        };
        assert_eq!(
            decision_sufficiency_horizon(&contract(), &[evidence]),
            Err(ReferenceError::SelectedActionMissing {
                action_id: "a0".to_owned(),
            })
        );
    }

    #[test]
    fn alternative_loss_bound_is_required_when_alternative_is_admissible() {
        let evidence = step(
            1,
            Admissibility::Yes,
            Some(LossInterval::new(1, 2)),
            Some(alternative(Admissibility::Yes, None)),
        );
        let result = decision_sufficiency_horizon(&contract(), &[evidence]).unwrap();
        assert!(matches!(
            result.stop_reason,
            Some(HorizonStopReason::EvidenceUnknown(
                UnknownReason::MissingExpectedLossBound
            ))
        ));
    }
}
