//! Bounded task-phase planning for Aqua's low-flow + user-authorized boost research.
//!
//! This module validates an advisory phase schedule against an exact, independently
//! qualified capability catalogue. It does not calculate water/energy, actuate a fixture,
//! authorize a valve operation, or replace the physical local hold-to-boost control.
//! Manufacturer declarations are evidence for comparator research, not executable
//! qualifications.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPhaseKind {
    Prepare,
    BaseDelivery,
    OptionalBoost,
    Pause,
    Rinse,
    ReturnToReady,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityRole {
    BaseDelivery,
    Boost,
    Rinse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityQualification {
    IndependentlyQualified,
    ManufacturerDeclaredOnly,
    Ambiguous,
    Stale,
    Revoked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationRequirement {
    LocalQualifiedControl,
    DirectUserHoldWithAutoReturn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedModeCapability {
    pub mode_id: String,
    pub market_variant_id: String,
    pub role: CapabilityRole,
    pub configuration_sha256: String,
    /// Frozen pressure, inlet-water, temperature, mounting and room applicability.
    pub delivery_conditions_sha256: String,
    pub qualification_method_sha256: String,
    pub qualification_receipt_sha256: String,
    pub qualification: CapabilityQualification,
    pub activation: ActivationRequirement,
    /// Test-qualified flow at the evidence-bound pressure point; not a universal curve.
    pub qualified_flow_lpm: String,
    /// Required only for boost modes; local hardware must independently enforce this.
    pub maximum_continuous_seconds: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoostConsent {
    NotRequired,
    ExplicitlyAuthorizedForThisTask,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPhase {
    pub kind: TaskPhaseKind,
    pub duration_seconds: u32,
    pub mode_id: Option<String>,
    pub boost_consent: BoostConsent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPhasePlan {
    pub task_profile_id: String,
    pub market_variant_id: String,
    pub configuration_sha256: String,
    pub delivery_conditions_sha256: String,
    pub protocol_sha256: String,
    pub phases: Vec<TaskPhase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskPhaseError {
    MissingTaskProfile,
    MissingMarketVariant,
    InvalidConfigurationDigest,
    InvalidDeliveryConditionsDigest,
    DeliveryConditionsMismatch,
    InvalidQualificationMethod,
    InvalidProtocolDigest,
    EmptyPhaseSequence,
    PrepareMustBeFirst,
    ReturnToReadyMustBeLast,
    InvalidPhaseOrder { phase_index: usize },
    MissingBaseDelivery,
    ZeroDuration { phase_index: usize },
    DurationOverflow,
    MissingMode { phase_index: usize },
    UnexpectedMode { phase_index: usize },
    MissingCapability { mode_id: String },
    DuplicateCapability { mode_id: String },
    ConfigurationMismatch { mode_id: String },
    InvalidQualificationReceipt { mode_id: String },
    CapabilityNotIndependentlyQualified { mode_id: String },
    RoleMismatch { phase_index: usize, mode_id: String },
    ActivationMismatch { mode_id: String },
    InvalidQualifiedFlow { mode_id: String },
    MissingBoostDurationLimit { mode_id: String },
    UnexpectedBoostDurationLimit { mode_id: String },
    BoostConsentRequired { phase_index: usize },
    BoostDurationExceeded {
        phase_index: usize,
        requested_seconds: u32,
        maximum_seconds: u32,
    },
    ConsentOnNonBoostPhase { phase_index: usize },
}

impl Display for TaskPhaseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTaskProfile => write!(f, "task profile identifier is missing"),
            Self::MissingMarketVariant => {
                write!(f, "exact market variant identifier is missing")
            }
            Self::InvalidConfigurationDigest => {
                write!(f, "invalid task configuration SHA-256")
            }
            Self::InvalidDeliveryConditionsDigest => {
                write!(f, "invalid delivery conditions SHA-256")
            }
            Self::DeliveryConditionsMismatch => {
                write!(
                    f,
                    "mode qualification does not cover the plan delivery conditions"
                )
            }
            Self::InvalidQualificationMethod => {
                write!(f, "invalid qualification method SHA-256")
            },
            Self::InvalidProtocolDigest => write!(f, "invalid task protocol SHA-256"),
            Self::EmptyPhaseSequence => write!(f, "task phase sequence is empty"),
            Self::PrepareMustBeFirst => write!(f, "PREPARE must be the first task phase"),
            Self::ReturnToReadyMustBeLast => {
                write!(f, "RETURN_TO_READY must be the final task phase")
            }
            Self::InvalidPhaseOrder { phase_index } => {
                write!(f, "phase {phase_index} violates the task phase ordering contract")
            }
            Self::MissingBaseDelivery => write!(f, "task plan must include base delivery"),
            Self::ZeroDuration { phase_index } => {
                write!(f, "phase {phase_index} has zero duration")
            }
            Self::DurationOverflow => write!(f, "total task duration overflowed"),
            Self::MissingMode { phase_index } => {
                write!(f, "phase {phase_index} requires an exact mode identifier")
            }
            Self::UnexpectedMode { phase_index } => {
                write!(f, "phase {phase_index} must not select a fixture mode")
            }
            Self::MissingCapability { mode_id } => {
                write!(f, "mode capability is absent: {mode_id}")
            }
            Self::DuplicateCapability { mode_id } => {
                write!(f, "mode capability identifier is duplicated: {mode_id}")
            }
            Self::ConfigurationMismatch { mode_id } => {
                write!(f, "mode belongs to a different hardware configuration: {mode_id}")
            }
            Self::InvalidQualificationReceipt { mode_id } => {
                write!(f, "mode qualification receipt is invalid: {mode_id}")
            }
            Self::CapabilityNotIndependentlyQualified { mode_id } => {
                write!(f, "mode is not independently qualified: {mode_id}")
            }
            Self::RoleMismatch { phase_index, mode_id } => {
                write!(f, "phase {phase_index} does not match capability role: {mode_id}")
            }
            Self::ActivationMismatch { mode_id } => {
                write!(f, "mode activation requirement is inconsistent: {mode_id}")
            }
            Self::InvalidQualifiedFlow { mode_id } => {
                write!(f, "declared mode flow is invalid: {mode_id}")
            }
            Self::MissingBoostDurationLimit { mode_id } => {
                write!(f, "boost mode has no local maximum continuous duration: {mode_id}")
            }
            Self::UnexpectedBoostDurationLimit { mode_id } => {
                write!(f, "non-boost mode declares a boost duration limit: {mode_id}")
            }
            Self::BoostConsentRequired { phase_index } => {
                write!(f, "boost phase {phase_index} lacks explicit task-specific consent")
            }
            Self::BoostDurationExceeded {
                phase_index,
                requested_seconds,
                maximum_seconds,
            } => write!(
                f,
                "boost phase {phase_index} duration {requested_seconds}s exceeds local mode limit {maximum_seconds}s"
            ),
            Self::ConsentOnNonBoostPhase { phase_index } => {
                write!(f, "phase {phase_index} carries boost consent but is not a boost")
            }
        }
    }
}

impl Error for TaskPhaseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedTaskPhaseSummary {
    pub task_profile_id: String,
    pub phase_count: usize,
    pub total_duration_seconds: u32,
    pub base_delivery_seconds: u32,
    pub boost_seconds: u32,
    pub pause_seconds: u32,
    pub rinse_seconds: u32,
    pub selected_mode_ids: Vec<String>,
    /// Always false: this contract is advisory and has no physical actuation authority.
    pub executable_by_ai: bool,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_qualified_flow(mode: &QualifiedModeCapability) -> Result<f64, TaskPhaseError> {
    let value = mode
        .qualified_flow_lpm
        .parse::<f64>()
        .map_err(|_| TaskPhaseError::InvalidQualifiedFlow {
            mode_id: mode.mode_id.clone(),
        })?;
    if !value.is_finite() || value <= 0.0 {
        return Err(TaskPhaseError::InvalidQualifiedFlow {
            mode_id: mode.mode_id.clone(),
        });
    }
    Ok(value)
}

fn expected_role(kind: TaskPhaseKind) -> Option<CapabilityRole> {
    match kind {
        TaskPhaseKind::BaseDelivery => Some(CapabilityRole::BaseDelivery),
        TaskPhaseKind::OptionalBoost => Some(CapabilityRole::Boost),
        TaskPhaseKind::Rinse => Some(CapabilityRole::Rinse),
        TaskPhaseKind::Prepare | TaskPhaseKind::Pause | TaskPhaseKind::ReturnToReady => None,
    }
}

/// Validate a proposed task sequence against exact, independently qualified device modes.
///
/// The summary reports time allocation only. It deliberately does not turn declared flow
/// into estimated task litres: Symtropy owns canonical resource closure and the actual
/// time-varying flow trace. Any boost remains a direct physical user hold enforced locally;
/// this plan never emits actuator commands.
pub fn validate_task_phase_plan(
    plan: &TaskPhasePlan,
    capabilities: &[QualifiedModeCapability],
) -> Result<ValidatedTaskPhaseSummary, TaskPhaseError> {
    if plan.task_profile_id.trim().is_empty() {
        return Err(TaskPhaseError::MissingTaskProfile);
    }
    if plan.market_variant_id.trim().is_empty() {
        return Err(TaskPhaseError::MissingMarketVariant);
    }
    if !is_sha256(&plan.configuration_sha256) {
        return Err(TaskPhaseError::InvalidConfigurationDigest);
    }
    if !is_sha256(&plan.delivery_conditions_sha256) {
        return Err(TaskPhaseError::InvalidDeliveryConditionsDigest);
    }
    if !is_sha256(&plan.protocol_sha256) {
        return Err(TaskPhaseError::InvalidProtocolDigest);
    }
    if plan.phases.is_empty() {
        return Err(TaskPhaseError::EmptyPhaseSequence);
    }
    if plan.phases.first().map(|phase| phase.kind) != Some(TaskPhaseKind::Prepare) {
        return Err(TaskPhaseError::PrepareMustBeFirst);
    }
    if plan.phases.last().map(|phase| phase.kind) != Some(TaskPhaseKind::ReturnToReady) {
        return Err(TaskPhaseError::ReturnToReadyMustBeLast);
    }

    let mut capability_ids = HashSet::with_capacity(capabilities.len());
    for capability in capabilities {
        if capability.mode_id.trim().is_empty()
            || !capability_ids.insert(capability.mode_id.as_str())
        {
            return Err(TaskPhaseError::DuplicateCapability {
                mode_id: capability.mode_id.clone(),
            });
        }
    }

    let mut base_delivery_seen = false;
    for (phase_index, phase) in plan.phases.iter().enumerate() {
        if phase_index > 0 && phase_index + 1 < plan.phases.len()
            && matches!(phase.kind, TaskPhaseKind::Prepare | TaskPhaseKind::ReturnToReady)
        {
            return Err(TaskPhaseError::InvalidPhaseOrder { phase_index });
        }
        if phase.kind == TaskPhaseKind::Rinse && !base_delivery_seen {
            return Err(TaskPhaseError::InvalidPhaseOrder { phase_index });
        }
        if phase.kind == TaskPhaseKind::BaseDelivery {
            base_delivery_seen = true;
        }
        if phase.kind == TaskPhaseKind::OptionalBoost
            && !matches!(
                plan.phases.get(phase_index.wrapping_sub(1)).map(|previous| previous.kind),
                Some(TaskPhaseKind::BaseDelivery | TaskPhaseKind::Rinse)
            )
        {
            return Err(TaskPhaseError::InvalidPhaseOrder { phase_index });
        }
    }

    let mut total_duration_seconds = 0_u32;
    let mut base_delivery_seconds = 0_u32;
    let mut boost_seconds = 0_u32;
    let mut pause_seconds = 0_u32;
    let mut rinse_seconds = 0_u32;
    let mut selected_mode_ids = Vec::new();
    let mut has_base_delivery = false;

    for (phase_index, phase) in plan.phases.iter().enumerate() {
        if phase.duration_seconds == 0 {
            return Err(TaskPhaseError::ZeroDuration { phase_index });
        }
        total_duration_seconds = total_duration_seconds
            .checked_add(phase.duration_seconds)
            .ok_or(TaskPhaseError::DurationOverflow)?;

        let expected = expected_role(phase.kind);
        match (expected, phase.mode_id.as_deref()) {
            (None, Some(_)) => {
                return Err(TaskPhaseError::UnexpectedMode { phase_index });
            }
            (Some(_), None) => {
                return Err(TaskPhaseError::MissingMode { phase_index });
            }
            (None, None) => {
                if phase.boost_consent != BoostConsent::NotRequired {
                    return Err(TaskPhaseError::ConsentOnNonBoostPhase { phase_index });
                }
                if phase.kind == TaskPhaseKind::Pause {
                    pause_seconds = pause_seconds
                        .checked_add(phase.duration_seconds)
                        .ok_or(TaskPhaseError::DurationOverflow)?;
                }
            }
            (Some(expected_role), Some(mode_id)) => {
                let capability = capabilities
                    .iter()
                    .find(|candidate| candidate.mode_id == mode_id)
                    .ok_or_else(|| TaskPhaseError::MissingCapability {
                        mode_id: mode_id.to_owned(),
                    })?;

                if capability.market_variant_id != plan.market_variant_id
                    || capability.configuration_sha256 != plan.configuration_sha256
                {
                    return Err(TaskPhaseError::ConfigurationMismatch {
                        mode_id: mode_id.to_owned(),
                    });
                }
                if !is_sha256(&capability.delivery_conditions_sha256) {
                    return Err(TaskPhaseError::InvalidDeliveryConditionsDigest);
                }
                if capability.delivery_conditions_sha256 != plan.delivery_conditions_sha256 {
                    return Err(TaskPhaseError::DeliveryConditionsMismatch);
                }
                if !is_sha256(&capability.qualification_method_sha256) {
                    return Err(TaskPhaseError::InvalidQualificationMethod);
                }
                if !is_sha256(&capability.qualification_receipt_sha256) {
                    return Err(TaskPhaseError::InvalidQualificationReceipt {
                        mode_id: mode_id.to_owned(),
                    });
                }
                if capability.qualification != CapabilityQualification::IndependentlyQualified {
                    return Err(TaskPhaseError::CapabilityNotIndependentlyQualified {
                        mode_id: mode_id.to_owned(),
                    });
                }
                if capability.role != expected_role {
                    return Err(TaskPhaseError::RoleMismatch {
                        phase_index,
                        mode_id: mode_id.to_owned(),
                    });
                }
                let _declared_flow = parse_qualified_flow(capability)?;

                match expected_role {
                    CapabilityRole::Boost => {
                        if capability.activation
                            != ActivationRequirement::DirectUserHoldWithAutoReturn
                        {
                            return Err(TaskPhaseError::ActivationMismatch {
                                mode_id: mode_id.to_owned(),
                            });
                        }
                        let maximum_seconds = capability.maximum_continuous_seconds.ok_or_else(
                            || TaskPhaseError::MissingBoostDurationLimit {
                                mode_id: mode_id.to_owned(),
                            },
                        )?;
                        if maximum_seconds == 0 {
                            return Err(TaskPhaseError::MissingBoostDurationLimit {
                                mode_id: mode_id.to_owned(),
                            });
                        }
                        if phase.boost_consent != BoostConsent::ExplicitlyAuthorizedForThisTask {
                            return Err(TaskPhaseError::BoostConsentRequired { phase_index });
                        }
                        if phase.duration_seconds > maximum_seconds {
                            return Err(TaskPhaseError::BoostDurationExceeded {
                                phase_index,
                                requested_seconds: phase.duration_seconds,
                                maximum_seconds,
                            });
                        }
                        boost_seconds = boost_seconds
                            .checked_add(phase.duration_seconds)
                            .ok_or(TaskPhaseError::DurationOverflow)?;
                    }
                    CapabilityRole::BaseDelivery | CapabilityRole::Rinse => {
                        if capability.activation
                            != ActivationRequirement::LocalQualifiedControl
                            || capability.maximum_continuous_seconds.is_some()
                        {
                            return Err(TaskPhaseError::ActivationMismatch {
                                mode_id: mode_id.to_owned(),
                            });
                        }
                        if phase.boost_consent != BoostConsent::NotRequired {
                            return Err(TaskPhaseError::ConsentOnNonBoostPhase { phase_index });
                        }
                        match expected_role {
                            CapabilityRole::BaseDelivery => {
                                has_base_delivery = true;
                                base_delivery_seconds = base_delivery_seconds
                                    .checked_add(phase.duration_seconds)
                                    .ok_or(TaskPhaseError::DurationOverflow)?;
                            }
                            CapabilityRole::Rinse => {
                                rinse_seconds = rinse_seconds
                                    .checked_add(phase.duration_seconds)
                                    .ok_or(TaskPhaseError::DurationOverflow)?;
                            }
                            CapabilityRole::Boost => {
                                return Err(TaskPhaseError::RoleMismatch {
                                    phase_index,
                                    mode_id: mode_id.to_owned(),
                                });
                            }
                        }
                    }
                }
                selected_mode_ids.push(mode_id.to_owned());
            }
        }
    }

    if !has_base_delivery {
        return Err(TaskPhaseError::MissingBaseDelivery);
    }

    Ok(ValidatedTaskPhaseSummary {
        task_profile_id: plan.task_profile_id.clone(),
        phase_count: plan.phases.len(),
        total_duration_seconds,
        base_delivery_seconds,
        boost_seconds,
        pause_seconds,
        rinse_seconds,
        selected_mode_ids,
        executable_by_ai: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn capability(
        mode_id: &str,
        role: CapabilityRole,
        activation: ActivationRequirement,
        qualification: CapabilityQualification,
        max_seconds: Option<u32>,
    ) -> QualifiedModeCapability {
        QualifiedModeCapability {
            mode_id: mode_id.to_owned(),
            market_variant_id: "AQS1-TEST-ARTICLE".to_owned(),
            role,
            configuration_sha256: digest('a'),
            delivery_conditions_sha256: digest('d'),
            qualification_method_sha256: digest('e'),
            qualification_receipt_sha256: digest('b'),
            qualification,
            activation,
            qualified_flow_lpm: match role {
                CapabilityRole::BaseDelivery => "4.5",
                CapabilityRole::Boost => "7.0",
                CapabilityRole::Rinse => "5.5",
            }
            .to_owned(),
            maximum_continuous_seconds: max_seconds,
        }
    }

    fn plan() -> TaskPhasePlan {
        TaskPhasePlan {
            task_profile_id: "HAIR_WASH_V1".to_owned(),
            market_variant_id: "AQS1-TEST-ARTICLE".to_owned(),
            configuration_sha256: digest('a'),
            delivery_conditions_sha256: digest('d'),
            protocol_sha256: digest('c'),
            phases: vec![
                TaskPhase {
                    kind: TaskPhaseKind::Prepare,
                    duration_seconds: 30,
                    mode_id: None,
                    boost_consent: BoostConsent::NotRequired,
                },
                TaskPhase {
                    kind: TaskPhaseKind::BaseDelivery,
                    duration_seconds: 120,
                    mode_id: Some("base".to_owned()),
                    boost_consent: BoostConsent::NotRequired,
                },
                TaskPhase {
                    kind: TaskPhaseKind::OptionalBoost,
                    duration_seconds: 20,
                    mode_id: Some("boost".to_owned()),
                    boost_consent: BoostConsent::ExplicitlyAuthorizedForThisTask,
                },
                TaskPhase {
                    kind: TaskPhaseKind::Pause,
                    duration_seconds: 15,
                    mode_id: None,
                    boost_consent: BoostConsent::NotRequired,
                },
                TaskPhase {
                    kind: TaskPhaseKind::Rinse,
                    duration_seconds: 90,
                    mode_id: Some("rinse".to_owned()),
                    boost_consent: BoostConsent::NotRequired,
                },
                TaskPhase {
                    kind: TaskPhaseKind::ReturnToReady,
                    duration_seconds: 10,
                    mode_id: None,
                    boost_consent: BoostConsent::NotRequired,
                },
            ],
        }
    }

    fn capabilities() -> Vec<QualifiedModeCapability> {
        vec![
            capability(
                "base",
                CapabilityRole::BaseDelivery,
                ActivationRequirement::LocalQualifiedControl,
                CapabilityQualification::IndependentlyQualified,
                None,
            ),
            capability(
                "boost",
                CapabilityRole::Boost,
                ActivationRequirement::DirectUserHoldRequired,
                CapabilityQualification::IndependentlyQualified,
                Some(30),
            ),
            capability(
                "rinse",
                CapabilityRole::Rinse,
                ActivationRequirement::LocalQualifiedControl,
                CapabilityQualification::IndependentlyQualified,
                None,
            ),
        ]
    }

    #[test]
    fn validates_task_phase_plan_with_explicit_boost_consent() {
        let result = validate_task_phase_plan(&plan(), &capabilities()).expect("valid plan");
        assert_eq!(result.phase_count, 6);
        assert_eq!(result.total_duration_seconds, 285);
        assert_eq!(result.base_delivery_seconds, 120);
        assert_eq!(result.boost_seconds, 20);
        assert_eq!(result.pause_seconds, 15);
        assert_eq!(result.rinse_seconds, 90);
        assert!(!result.executable_by_ai);
    }

    #[test]
    fn rejects_missing_exact_mode_capability() {
        let mut candidate = plan();
        candidate.phases[2].mode_id = Some("unregistered-boost".to_owned());
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::MissingCapability {
                mode_id: "unregistered-boost".to_owned()
            })
        );
    }

    #[test]
    fn rejects_different_hardware_configuration() {
        let mut caps = capabilities();
        caps[1].configuration_sha256 = digest('9');
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::ConfigurationMismatch {
                mode_id: "boost".to_owned()
            })
        );
    }

    #[test]
    fn rejects_manufacturer_declared_boost_without_independent_qualification() {
        let mut caps = capabilities();
        caps[1].qualification = CapabilityQualification::ManufacturerDeclaredOnly;
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::CapabilityNotIndependentlyQualified {
                mode_id: "boost".to_owned()
            })
        );
    }

    #[test]
    fn rejects_ambiguous_stale_and_revoked_capabilities() {
        for qualification in [
            CapabilityQualification::Ambiguous,
            CapabilityQualification::Stale,
            CapabilityQualification::Revoked,
        ] {
            let mut caps = capabilities();
            caps[1].qualification = qualification;
            assert_eq!(
                validate_task_phase_plan(&plan(), &caps),
                Err(TaskPhaseError::CapabilityNotIndependentlyQualified {
                    mode_id: "boost".to_owned()
                })
            );
        }
    }

    #[test]
    fn rejects_boost_without_explicit_task_consent() {
        let mut candidate = plan();
        candidate.phases[2].boost_consent = BoostConsent::NotRequired;
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::BoostConsentRequired { phase_index: 2 })
        );
    }

    #[test]
    fn rejects_boost_without_direct_user_hold_requirement() {
        let mut caps = capabilities();
        caps[1].activation = ActivationRequirement::LocalQualifiedControl;
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::ActivationMismatch {
                mode_id: "boost".to_owned()
            })
        );
    }

    #[test]
    fn rejects_boost_duration_above_local_qualified_limit() {
        let mut candidate = plan();
        candidate.phases[2].duration_seconds = 31;
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::BoostDurationExceeded {
                phase_index: 2,
                requested_seconds: 31,
                maximum_seconds: 30
            })
        );
    }

    #[test]
    fn rejects_boost_without_local_duration_limit() {
        let mut caps = capabilities();
        caps[1].maximum_continuous_seconds = None;
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::MissingBoostDurationLimit {
                mode_id: "boost".to_owned()
            })
        );
    }

    #[test]
    fn rejects_wrong_mode_role_for_phase() {
        let mut candidate = plan();
        candidate.phases[1].mode_id = Some("boost".to_owned());
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::RoleMismatch {
                phase_index: 1,
                mode_id: "boost".to_owned()
            })
        );
    }

    #[test]
    fn rejects_missing_base_delivery() {
        let mut candidate = plan();
        candidate.phases.remove(1);
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::MissingBaseDelivery)
        );
    }

    #[test]
    fn rejects_phase_sequence_without_prepare_or_return_to_ready() {
        let mut candidate = plan();
        candidate.phases.remove(0);
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::PrepareMustBeFirst)
        );

        let mut candidate = plan();
        candidate.phases.pop();
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::ReturnToReadyMustBeLast)
        );
    }

    #[test]
    fn rejects_fixture_mode_on_pause_or_prepare() {
        let mut candidate = plan();
        candidate.phases[3].mode_id = Some("base".to_owned());
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::UnexpectedMode { phase_index: 3 })
        );
    }

    #[test]
    fn rejects_zero_duration_and_total_duration_overflow() {
        let mut candidate = plan();
        candidate.phases[1].duration_seconds = 0;
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::ZeroDuration { phase_index: 1 })
        );

        let mut candidate = plan();
        candidate.phases[1].duration_seconds = u32::MAX;
        candidate.phases[2].duration_seconds = u32::MAX;
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::DurationOverflow)
        );
    }

    #[test]
    fn rejects_duplicate_and_invalid_mode_capability_records() {
        let mut caps = capabilities();
        caps.push(caps[0].clone());
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::DuplicateCapability {
                mode_id: "base".to_owned()
            })
        );

        let mut caps = capabilities();
        caps[1].qualification_receipt_sha256 = "unknown".to_owned();
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::InvalidQualificationReceipt {
                mode_id: "boost".to_owned()
            })
        );
    }

    #[test]
    fn rejects_non_finite_or_non_positive_declared_flow() {
        for flow in ["NaN", "inf", "-1", "0", "unknown"] {
            let mut caps = capabilities();
            caps[1].qualified_flow_lpm = flow.to_owned();
            assert_eq!(
                validate_task_phase_plan(&plan(), &caps),
                Err(TaskPhaseError::InvalidQualifiedFlow {
                    mode_id: "boost".to_owned()
                })
            );
        }
    }

    #[test]
    fn rejects_non_boost_capability_with_boost_only_fields() {
        let mut caps = capabilities();
        caps[0].maximum_continuous_seconds = Some(30);
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::ActivationMismatch {
                mode_id: "base".to_owned()
            })
        );
    }

    #[test]
    fn rejects_wrong_market_variant_and_delivery_conditions() {
        let mut caps = capabilities();
        caps[1].market_variant_id = "OTHER-SKU".to_owned();
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::ConfigurationMismatch {
                mode_id: "boost".to_owned()
            })
        );

        let mut caps = capabilities();
        caps[1].delivery_conditions_sha256 = digest('9');
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::DeliveryConditionsMismatch)
        );
    }

    #[test]
    fn rejects_invalid_qualification_method_and_plan_condition_digest() {
        let mut caps = capabilities();
        caps[1].qualification_method_sha256 = "unknown".to_owned();
        assert_eq!(
            validate_task_phase_plan(&plan(), &caps),
            Err(TaskPhaseError::InvalidQualificationMethod)
        );

        let mut candidate = plan();
        candidate.delivery_conditions_sha256 = "unknown".to_owned();
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::InvalidDeliveryConditionsDigest)
        );
    }

    #[test]
    fn rejects_boost_after_pause_or_an_interior_prepare_phase() {
        let mut candidate = plan();
        candidate.phases.swap(2, 3);
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::InvalidPhaseOrder { phase_index: 2 })
        );

        let mut candidate = plan();
        candidate.phases.insert(
            2,
            TaskPhase {
                kind: TaskPhaseKind::Prepare,
                duration_seconds: 1,
                mode_id: None,
                boost_consent: BoostConsent::NotRequired,
            },
        );
        assert_eq!(
            validate_task_phase_plan(&candidate, &capabilities()),
            Err(TaskPhaseError::InvalidPhaseOrder { phase_index: 2 })
        );
    }

}
