//! Product-level service-continuity classification for Luminous Aqua.
//!
//! This is an evidence/state evaluator only. It does not actuate valves, implement
//! emergency isolation, determine safe temperatures, approve water quality, or replace
//! locally qualified hardware protections. A local device authority must supply the
//! qualified capability state; this crate classifies the supplied state and fails closed
//! when critical evidence or protection status is missing or uncertain.

use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaterUseQualification {
    QualifiedForIntendedUse,
    Unknown,
    Stale,
    Revoked,
    WrongUseClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalProtectionState {
    Qualified,
    Tripped,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceFreshness {
    Current,
    Missing,
    Stale,
    Contradictory,
    Revoked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureEnvelopeState {
    InsideQualifiedEnvelope,
    BelowQualifiedMinimum,
    AboveQualifiedMaximum,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainPathState {
    Clear,
    Blocked,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeakState {
    Clear,
    Detected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalServiceCapability {
    FullServiceQualified,
    ReducedServiceQualified,
    ManualOnlyQualified,
    NoQualifiedService,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigitalExperienceState {
    Available,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuityMode {
    QualifiedFull,
    QualifiedReduced,
    ManualOnly,
    IsolatedStop,
    RecoveryPending,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContinuityReason {
    WaterUseNotQualified,
    LocalProtectionNotQualified,
    CriticalEvidenceNotCurrent,
    PressureOutsideOrUnknown,
    DrainNotClear,
    LeakNotClear,
    RequalificationRequired,
    NoQualifiedLocalService,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContinuityObservation {
    pub water_use: WaterUseQualification,
    pub local_protection: LocalProtectionState,
    pub critical_evidence: EvidenceFreshness,
    pub pressure: PressureEnvelopeState,
    pub drain: DrainPathState,
    pub leak: LeakState,
    pub local_capability: LocalServiceCapability,
    pub digital_experience: DigitalExperienceState,
    /// Remains true after events requiring inspection, flush, cleaning, calibration,
    /// source reassessment or explicit local restart authorization.
    pub requalification_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuityAssessment {
    pub mode: ContinuityMode,
    /// Informational only. This cannot promote the continuity mode or clear a reason.
    pub recommendations_available: bool,
    pub reasons: Vec<ContinuityReason>,
}

impl ContinuityAssessment {
    pub fn is_service_admitted(&self) -> bool {
        matches!(
            self.mode,
            ContinuityMode::QualifiedFull
                | ContinuityMode::QualifiedReduced
                | ContinuityMode::ManualOnly
        )
    }
}

impl Display for ContinuityMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QualifiedFull => write!(f, "QUALIFIED_FULL"),
            Self::QualifiedReduced => write!(f, "QUALIFIED_REDUCED"),
            Self::ManualOnly => write!(f, "MANUAL_ONLY"),
            Self::IsolatedStop => write!(f, "ISOLATED_STOP"),
            Self::RecoveryPending => write!(f, "RECOVERY_PENDING"),
        }
    }
}

/// Classify the highest locally qualified service state.
///
/// This function never infers or creates a reduced/manual mode. Such a mode must already
/// be explicitly qualified by the local device authority. Missing, stale, contradictory,
/// revoked, or out-of-envelope critical information results in IsolatedStop.
///
/// RecoveryPending is returned only after immediate stop conditions have cleared and
/// the caller still marks requalification as required. Clearing a fault alone therefore
/// cannot automatically restore a higher service state.
pub fn assess_continuity(observation: ContinuityObservation) -> ContinuityAssessment {
    let mut reasons = Vec::new();

    if observation.water_use != WaterUseQualification::QualifiedForIntendedUse {
        reasons.push(ContinuityReason::WaterUseNotQualified);
    }
    if observation.local_protection != LocalProtectionState::Qualified {
        reasons.push(ContinuityReason::LocalProtectionNotQualified);
    }
    if observation.critical_evidence != EvidenceFreshness::Current {
        reasons.push(ContinuityReason::CriticalEvidenceNotCurrent);
    }
    if observation.pressure != PressureEnvelopeState::InsideQualifiedEnvelope {
        reasons.push(ContinuityReason::PressureOutsideOrUnknown);
    }
    if observation.drain != DrainPathState::Clear {
        reasons.push(ContinuityReason::DrainNotClear);
    }
    if observation.leak != LeakState::Clear {
        reasons.push(ContinuityReason::LeakNotClear);
    }

    let mode = if !reasons.is_empty() {
        ContinuityMode::IsolatedStop
    } else if observation.requalification_required {
        reasons.push(ContinuityReason::RequalificationRequired);
        ContinuityMode::RecoveryPending
    } else {
        match observation.local_capability {
            LocalServiceCapability::FullServiceQualified => ContinuityMode::QualifiedFull,
            LocalServiceCapability::ReducedServiceQualified => ContinuityMode::QualifiedReduced,
            LocalServiceCapability::ManualOnlyQualified => ContinuityMode::ManualOnly,
            LocalServiceCapability::NoQualifiedService => {
                reasons.push(ContinuityReason::NoQualifiedLocalService);
                ContinuityMode::IsolatedStop
            }
        }
    };

    ContinuityAssessment {
        mode,
        recommendations_available: observation.digital_experience
            == DigitalExperienceState::Available,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qualified_observation() -> ContinuityObservation {
        ContinuityObservation {
            water_use: WaterUseQualification::QualifiedForIntendedUse,
            local_protection: LocalProtectionState::Qualified,
            critical_evidence: EvidenceFreshness::Current,
            pressure: PressureEnvelopeState::InsideQualifiedEnvelope,
            drain: DrainPathState::Clear,
            leak: LeakState::Clear,
            local_capability: LocalServiceCapability::FullServiceQualified,
            digital_experience: DigitalExperienceState::Available,
            requalification_required: false,
        }
    }

    #[test]
    fn admits_full_service_when_all_critical_conditions_are_qualified() {
        let assessment = assess_continuity(qualified_observation());
        assert_eq!(assessment.mode, ContinuityMode::QualifiedFull);
        assert!(assessment.is_service_admitted());
        assert!(assessment.reasons.is_empty());
    }

    #[test]
    fn digital_service_loss_does_not_reduce_qualified_local_service() {
        let mut observation = qualified_observation();
        observation.digital_experience = DigitalExperienceState::Unavailable;

        let assessment = assess_continuity(observation);

        assert_eq!(assessment.mode, ContinuityMode::QualifiedFull);
        assert!(assessment.is_service_admitted());
        assert!(!assessment.recommendations_available);
    }

    #[test]
    fn admits_only_an_explicitly_qualified_reduced_mode() {
        let mut observation = qualified_observation();
        observation.local_capability = LocalServiceCapability::ReducedServiceQualified;

        let assessment = assess_continuity(observation);
        assert_eq!(assessment.mode, ContinuityMode::QualifiedReduced);
    }

    #[test]
    fn admits_manual_only_when_local_authority_qualifies_that_mode() {
        let mut observation = qualified_observation();
        observation.local_capability = LocalServiceCapability::ManualOnlyQualified;
        observation.digital_experience = DigitalExperienceState::Unavailable;

        let assessment = assess_continuity(observation);
        assert_eq!(assessment.mode, ContinuityMode::ManualOnly);
        assert!(assessment.is_service_admitted());
    }

    #[test]
    fn rejects_unqualified_local_service_capability() {
        let mut observation = qualified_observation();
        observation.local_capability = LocalServiceCapability::NoQualifiedService;

        let assessment = assess_continuity(observation);
        assert_eq!(assessment.mode, ContinuityMode::IsolatedStop);
        assert_eq!(
            assessment.reasons,
            vec![ContinuityReason::NoQualifiedLocalService]
        );
    }

    #[test]
    fn source_quality_unknown_fails_closed() {
        let mut observation = qualified_observation();
        observation.water_use = WaterUseQualification::Unknown;

        let assessment = assess_continuity(observation);
        assert_eq!(assessment.mode, ContinuityMode::IsolatedStop);
        assert!(assessment
            .reasons
            .contains(&ContinuityReason::WaterUseNotQualified));
    }

    #[test]
    fn source_quality_stale_fails_closed() {
        let mut observation = qualified_observation();
        observation.water_use = WaterUseQualification::Stale;

        assert_eq!(
            assess_continuity(observation).mode,
            ContinuityMode::IsolatedStop
        );
    }

    #[test]
    fn revoked_or_wrong_use_class_source_fails_closed() {
        for state in [
            WaterUseQualification::Revoked,
            WaterUseQualification::WrongUseClass,
        ] {
            let mut observation = qualified_observation();
            observation.water_use = state;
            assert_eq!(
                assess_continuity(observation).mode,
                ContinuityMode::IsolatedStop
            );
        }
    }

    #[test]
    fn unknown_or_tripped_protection_fails_closed() {
        for state in [
            LocalProtectionState::Unknown,
            LocalProtectionState::Tripped,
            LocalProtectionState::Unavailable,
        ] {
            let mut observation = qualified_observation();
            observation.local_protection = state;
            assert_eq!(
                assess_continuity(observation).mode,
                ContinuityMode::IsolatedStop
            );
        }
    }

    #[test]
    fn missing_stale_contradictory_or_revoked_evidence_fails_closed() {
        for state in [
            EvidenceFreshness::Missing,
            EvidenceFreshness::Stale,
            EvidenceFreshness::Contradictory,
            EvidenceFreshness::Revoked,
        ] {
            let mut observation = qualified_observation();
            observation.critical_evidence = state;
            assert_eq!(
                assess_continuity(observation).mode,
                ContinuityMode::IsolatedStop
            );
        }
    }

    #[test]
    fn below_above_or_unknown_pressure_fails_closed() {
        for state in [
            PressureEnvelopeState::BelowQualifiedMinimum,
            PressureEnvelopeState::AboveQualifiedMaximum,
            PressureEnvelopeState::Unknown,
        ] {
            let mut observation = qualified_observation();
            observation.pressure = state;
            assert_eq!(
                assess_continuity(observation).mode,
                ContinuityMode::IsolatedStop
            );
        }
    }

    #[test]
    fn blocked_or_unknown_drain_fails_closed() {
        for state in [DrainPathState::Blocked, DrainPathState::Unknown] {
            let mut observation = qualified_observation();
            observation.drain = state;
            assert_eq!(
                assess_continuity(observation).mode,
                ContinuityMode::IsolatedStop
            );
        }
    }

    #[test]
    fn detected_or_unknown_leak_fails_closed() {
        for state in [LeakState::Detected, LeakState::Unknown] {
            let mut observation = qualified_observation();
            observation.leak = state;
            assert_eq!(
                assess_continuity(observation).mode,
                ContinuityMode::IsolatedStop
            );
        }
    }

    #[test]
    fn cleared_fault_does_not_clear_requalification_requirement() {
        let mut observation = qualified_observation();
        observation.requalification_required = true;

        let assessment = assess_continuity(observation);
        assert_eq!(assessment.mode, ContinuityMode::RecoveryPending);
        assert!(!assessment.is_service_admitted());
        assert_eq!(
            assessment.reasons,
            vec![ContinuityReason::RequalificationRequired]
        );
    }

    #[test]
    fn active_fault_takes_precedence_over_recovery_pending() {
        let mut observation = qualified_observation();
        observation.requalification_required = true;
        observation.leak = LeakState::Detected;

        let assessment = assess_continuity(observation);
        assert_eq!(assessment.mode, ContinuityMode::IsolatedStop);
        assert!(assessment.reasons.contains(&ContinuityReason::LeakNotClear));
        assert!(!assessment
            .reasons
            .contains(&ContinuityReason::RequalificationRequired));
    }

    #[test]
    fn accumulates_all_reasons_without_allowing_one_to_cancel_another() {
        let mut observation = qualified_observation();
        observation.water_use = WaterUseQualification::Unknown;
        observation.pressure = PressureEnvelopeState::Unknown;
        observation.drain = DrainPathState::Blocked;
        observation.leak = LeakState::Detected;

        let assessment = assess_continuity(observation);

        assert_eq!(assessment.mode, ContinuityMode::IsolatedStop);
        assert_eq!(assessment.reasons.len(), 4);
    }
}


#[cfg(test)]
mod exhaustive_state_space_tests {
    use super::*;

    #[test]
    fn all_57_600_input_combinations_preserve_continuity_invariants() {
        let water_states = [
            WaterUseQualification::QualifiedForIntendedUse,
            WaterUseQualification::Unknown,
            WaterUseQualification::Stale,
            WaterUseQualification::Revoked,
            WaterUseQualification::WrongUseClass,
        ];
        let protection_states = [
            LocalProtectionState::Qualified,
            LocalProtectionState::Tripped,
            LocalProtectionState::Unavailable,
            LocalProtectionState::Unknown,
        ];
        let evidence_states = [
            EvidenceFreshness::Current,
            EvidenceFreshness::Missing,
            EvidenceFreshness::Stale,
            EvidenceFreshness::Contradictory,
            EvidenceFreshness::Revoked,
        ];
        let pressure_states = [
            PressureEnvelopeState::InsideQualifiedEnvelope,
            PressureEnvelopeState::BelowQualifiedMinimum,
            PressureEnvelopeState::AboveQualifiedMaximum,
            PressureEnvelopeState::Unknown,
        ];
        let drain_states = [
            DrainPathState::Clear,
            DrainPathState::Blocked,
            DrainPathState::Unknown,
        ];
        let leak_states = [LeakState::Clear, LeakState::Detected, LeakState::Unknown];
        let capabilities = [
            LocalServiceCapability::FullServiceQualified,
            LocalServiceCapability::ReducedServiceQualified,
            LocalServiceCapability::ManualOnlyQualified,
            LocalServiceCapability::NoQualifiedService,
        ];
        let digital_states = [
            DigitalExperienceState::Available,
            DigitalExperienceState::Unavailable,
        ];

        let mut combinations = 0_usize;

        for water_use in water_states {
            for local_protection in protection_states {
                for critical_evidence in evidence_states {
                    for pressure in pressure_states {
                        for drain in drain_states {
                            for leak in leak_states {
                                for local_capability in capabilities {
                                    for digital_experience in digital_states {
                                        for requalification_required in [false, true] {
                                            let observation = ContinuityObservation {
                                                water_use,
                                                local_protection,
                                                critical_evidence,
                                                pressure,
                                                drain,
                                                leak,
                                                local_capability,
                                                digital_experience,
                                                requalification_required,
                                            };
                                            let assessment = assess_continuity(observation);

                                            let hard_fault = water_use
                                                != WaterUseQualification::QualifiedForIntendedUse
                                                || local_protection
                                                    != LocalProtectionState::Qualified
                                                || critical_evidence != EvidenceFreshness::Current
                                                || pressure
                                                    != PressureEnvelopeState::InsideQualifiedEnvelope
                                                || drain != DrainPathState::Clear
                                                || leak != LeakState::Clear;

                                            let expected_mode = if hard_fault {
                                                ContinuityMode::IsolatedStop
                                            } else if requalification_required {
                                                ContinuityMode::RecoveryPending
                                            } else {
                                                match local_capability {
                                                    LocalServiceCapability::FullServiceQualified
                                                    => {
                                                        ContinuityMode::QualifiedFull
                                                    }
                                                    LocalServiceCapability::ReducedServiceQualified
                                                    => {
                                                        ContinuityMode::QualifiedReduced
                                                    }
                                                    LocalServiceCapability::ManualOnlyQualified => {
                                                        ContinuityMode::ManualOnly
                                                    }
                                                    LocalServiceCapability::NoQualifiedService => {
                                                        ContinuityMode::IsolatedStop
                                                    }
                                                }
                                            };

                                            assert_eq!(
                                                assessment.mode,
                                                expected_mode,
                                                "unexpected mode for state combination \
                                                {combinations}"
                                            );
                                            assert_eq!(
                                                assessment.recommendations_available,
                                                digital_experience
                                                    == DigitalExperienceState::Available
                                            );
                                            assert_eq!(
                                                assessment.is_service_admitted(),
                                                matches!(
                                                    expected_mode,
                                                    ContinuityMode::QualifiedFull
                                                        | ContinuityMode::QualifiedReduced
                                                        | ContinuityMode::ManualOnly
                                                )
                                            );
                                            if hard_fault {
                                                assert_ne!(
                                                    assessment.mode,
                                                    ContinuityMode::RecoveryPending
                                                );
                                            }
                                            combinations += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(combinations, 57_600);
    }
}
