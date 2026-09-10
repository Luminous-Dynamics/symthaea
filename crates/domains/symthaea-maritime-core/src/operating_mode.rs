// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::OperatingEnvelope;
use serde::{Deserialize, Serialize};

/// Explicit platform operating modes.
///
/// These are configuration/behavior states, not authority grants. A caller must
/// evaluate the requested mode against the current operating envelope, current
/// operational-limit evidence, local interlocks and an independently evaluated
/// authority decision before transitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaritimeOperatingMode {
    DockedSafe,
    HarborTransit,
    SurfaceTransit,
    SubmergedTransit,
    ResidentVehicleSupport,
    SurfaceAviation,
    Maintenance,
    ReturnToService,
    EmergencySafeState,
}

impl MaritimeOperatingMode {
    pub const ALL: [Self; 9] = [
        Self::DockedSafe,
        Self::HarborTransit,
        Self::SurfaceTransit,
        Self::SubmergedTransit,
        Self::ResidentVehicleSupport,
        Self::SurfaceAviation,
        Self::Maintenance,
        Self::ReturnToService,
        Self::EmergencySafeState,
    ];

    /// Most restrictive envelope under which this mode may still be requested.
    ///
    /// The mode does not broaden the envelope: callers must use the more
    /// restrictive of the platform baseline and the current operational-limit
    /// assessment.
    pub fn maximum_permitted_envelope(self) -> OperatingEnvelope {
        match self {
            Self::SurfaceAviation | Self::ReturnToService => OperatingEnvelope::Normal,
            Self::SubmergedTransit | Self::ResidentVehicleSupport | Self::Maintenance => {
                OperatingEnvelope::ReducedCapability
            }
            Self::HarborTransit | Self::SurfaceTransit => OperatingEnvelope::SafeTransit,
            Self::DockedSafe => OperatingEnvelope::RecoverOrSurface,
            Self::EmergencySafeState => OperatingEnvelope::FailStop,
        }
    }
}

/// Current relationship to externally or locally defined operational limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationalLimitStatus {
    WithinLimits,
    Degraded,
    Exceeded,
    Unknown,
}

/// Evidence-bearing assessment of the platform's current operational limits.
///
/// The actual environmental/plant limits remain owned by the responsible
/// platform or subsystem. Maritime-core only consumes their conservative
/// assessment and never invents a physical threshold itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationalLimitAssessment {
    pub status: OperationalLimitStatus,
    /// Opaque binding to the evidence/configuration used for this assessment.
    /// `Unknown` may be represented without a binding; every affirmative status
    /// must be evidence-backed.
    pub evidence_binding: Option<String>,
}

impl OperationalLimitAssessment {
    pub fn validate(&self) -> Result<(), &'static str> {
        match self.evidence_binding.as_deref() {
            Some(binding) => {
                if binding.trim().is_empty() {
                    return Err("operational-limit evidence binding must not be empty");
                }
                if binding.trim() != binding {
                    return Err(
                        "operational-limit evidence binding must not contain outer whitespace",
                    );
                }
                if binding.chars().any(char::is_control) {
                    return Err("operational-limit evidence binding must not contain controls");
                }
            }
            None if self.status != OperationalLimitStatus::Unknown => {
                return Err("known operational-limit status requires evidence binding");
            }
            None => {}
        }
        Ok(())
    }

    /// Conservative restriction implied by this assessment alone.
    pub fn required_envelope(&self) -> OperatingEnvelope {
        match self.status {
            OperationalLimitStatus::WithinLimits => OperatingEnvelope::Normal,
            OperationalLimitStatus::Degraded => OperatingEnvelope::SafeTransit,
            OperationalLimitStatus::Unknown => OperatingEnvelope::HoldOrLoiter,
            OperationalLimitStatus::Exceeded => OperatingEnvelope::FailStop,
        }
    }
}

/// Inputs required at the point where an operating-mode transition is decided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatingModeContext {
    pub baseline_envelope: OperatingEnvelope,
    pub limits: OperationalLimitAssessment,
    /// Result of independent authority evaluation. Maritime-core deliberately
    /// does not smuggle authority credentials into this structure.
    pub authority_permitted: bool,
    /// Evidence that transition-specific prerequisites have been observed.
    /// Examples include a completed docking sequence or return-to-service check.
    pub transition_evidence_present: bool,
    /// Physical/local interlock projection. A network or cognitive component may
    /// consume this fact but must not override the underlying local controller.
    pub local_interlocks_clear: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatingModeRefusal {
    MalformedLimitAssessment,
    AuthorityDenied,
    MissingTransitionEvidence,
    LocalInterlockBlocked,
    EnvelopeTooRestrictive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatingModeDecision {
    Permitted {
        effective_envelope: OperatingEnvelope,
    },
    Refused(OperatingModeRefusal),
}

/// Evaluate a requested mode without expanding authority or physical limits.
///
/// `EmergencySafeState` is deliberately always requestable: loss of authority,
/// time, communications, evidence or an interlock fault must never prevent a
/// caller from asking the local safety layer to move toward its safe state.
/// Whether/how that local layer achieves the state remains outside this crate.
pub fn evaluate_operating_mode(
    requested: MaritimeOperatingMode,
    context: &OperatingModeContext,
) -> OperatingModeDecision {
    if requested == MaritimeOperatingMode::EmergencySafeState {
        return OperatingModeDecision::Permitted {
            effective_envelope: OperatingEnvelope::FailStop,
        };
    }

    if context.limits.validate().is_err() {
        return OperatingModeDecision::Refused(OperatingModeRefusal::MalformedLimitAssessment);
    }
    if !context.authority_permitted {
        return OperatingModeDecision::Refused(OperatingModeRefusal::AuthorityDenied);
    }
    if !context.transition_evidence_present {
        return OperatingModeDecision::Refused(OperatingModeRefusal::MissingTransitionEvidence);
    }
    if !context.local_interlocks_clear {
        return OperatingModeDecision::Refused(OperatingModeRefusal::LocalInterlockBlocked);
    }

    let effective_envelope = context
        .baseline_envelope
        .max(context.limits.required_envelope());
    if effective_envelope > requested.maximum_permitted_envelope() {
        return OperatingModeDecision::Refused(OperatingModeRefusal::EnvelopeTooRestrictive);
    }

    OperatingModeDecision::Permitted { effective_envelope }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(status: OperationalLimitStatus) -> OperationalLimitAssessment {
        OperationalLimitAssessment {
            status,
            evidence_binding: if status == OperationalLimitStatus::Unknown {
                None
            } else {
                Some("evidence:limits-v1".into())
            },
        }
    }

    fn context(envelope: OperatingEnvelope, status: OperationalLimitStatus) -> OperatingModeContext {
        OperatingModeContext {
            baseline_envelope: envelope,
            limits: limits(status),
            authority_permitted: true,
            transition_evidence_present: true,
            local_interlocks_clear: true,
        }
    }

    #[test]
    fn unknown_limits_fail_closed_for_active_operation() {
        assert_eq!(
            evaluate_operating_mode(
                MaritimeOperatingMode::SubmergedTransit,
                &context(
                    OperatingEnvelope::Normal,
                    OperationalLimitStatus::Unknown
                ),
            ),
            OperatingModeDecision::Refused(OperatingModeRefusal::EnvelopeTooRestrictive)
        );
    }

    #[test]
    fn degraded_limits_compose_with_existing_restriction() {
        assert_eq!(
            evaluate_operating_mode(
                MaritimeOperatingMode::SurfaceTransit,
                &context(
                    OperatingEnvelope::ReducedCapability,
                    OperationalLimitStatus::Degraded,
                ),
            ),
            OperatingModeDecision::Permitted {
                effective_envelope: OperatingEnvelope::SafeTransit,
            }
        );
    }

    #[test]
    fn high_consequence_modes_require_normal_envelope() {
        assert_eq!(
            evaluate_operating_mode(
                MaritimeOperatingMode::SurfaceAviation,
                &context(
                    OperatingEnvelope::ReducedCapability,
                    OperationalLimitStatus::WithinLimits,
                ),
            ),
            OperatingModeDecision::Refused(OperatingModeRefusal::EnvelopeTooRestrictive)
        );
    }

    #[test]
    fn missing_authority_evidence_or_interlock_never_authorizes() {
        let mut ctx = context(
            OperatingEnvelope::Normal,
            OperationalLimitStatus::WithinLimits,
        );
        ctx.authority_permitted = false;
        assert_eq!(
            evaluate_operating_mode(MaritimeOperatingMode::SurfaceTransit, &ctx),
            OperatingModeDecision::Refused(OperatingModeRefusal::AuthorityDenied)
        );

        ctx.authority_permitted = true;
        ctx.transition_evidence_present = false;
        assert_eq!(
            evaluate_operating_mode(MaritimeOperatingMode::SurfaceTransit, &ctx),
            OperatingModeDecision::Refused(OperatingModeRefusal::MissingTransitionEvidence)
        );

        ctx.transition_evidence_present = true;
        ctx.local_interlocks_clear = false;
        assert_eq!(
            evaluate_operating_mode(MaritimeOperatingMode::SurfaceTransit, &ctx),
            OperatingModeDecision::Refused(OperatingModeRefusal::LocalInterlockBlocked)
        );
    }

    #[test]
    fn malformed_known_limit_assessment_fails_closed() {
        let mut ctx = context(
            OperatingEnvelope::Normal,
            OperationalLimitStatus::WithinLimits,
        );
        ctx.limits.evidence_binding = None;
        assert_eq!(
            evaluate_operating_mode(MaritimeOperatingMode::SurfaceTransit, &ctx),
            OperatingModeDecision::Refused(OperatingModeRefusal::MalformedLimitAssessment)
        );
    }

    #[test]
    fn emergency_safe_state_remains_requestable_when_everything_else_is_lost() {
        let ctx = OperatingModeContext {
            baseline_envelope: OperatingEnvelope::FailStop,
            limits: OperationalLimitAssessment {
                status: OperationalLimitStatus::Unknown,
                evidence_binding: None,
            },
            authority_permitted: false,
            transition_evidence_present: false,
            local_interlocks_clear: false,
        };
        assert_eq!(
            evaluate_operating_mode(MaritimeOperatingMode::EmergencySafeState, &ctx),
            OperatingModeDecision::Permitted {
                effective_envelope: OperatingEnvelope::FailStop,
            }
        );
    }

    #[test]
    fn stronger_degradation_never_turns_refusal_into_permission() {
        let envelopes = [
            OperatingEnvelope::Normal,
            OperatingEnvelope::ReducedCapability,
            OperatingEnvelope::SafeTransit,
            OperatingEnvelope::HoldOrLoiter,
            OperatingEnvelope::RecoverOrSurface,
            OperatingEnvelope::FailStop,
        ];

        for mode in MaritimeOperatingMode::ALL {
            let mut already_refused = false;
            for envelope in envelopes {
                let decision = evaluate_operating_mode(
                    mode,
                    &context(envelope, OperationalLimitStatus::WithinLimits),
                );
                let refused = matches!(decision, OperatingModeDecision::Refused(_));
                if already_refused {
                    assert!(refused, "{mode:?} broadened at {envelope:?}");
                }
                already_refused |= refused;
            }
        }
    }
}
