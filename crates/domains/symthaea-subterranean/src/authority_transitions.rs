// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-readable inventory of runtime surfaces that can affect authority.
//!
//! Widening is a semantic property of a transition, not a naming convention.
//! `set_*`, `reset`, restore, service, evidence-injection and recovery APIs can
//! all increase capability under some inputs. This registry states the proof
//! required by the widening branch before RA-18 migrates those APIs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityTransitionClass {
    /// Every admissible invocation preserves or reduces productive authority.
    RestrictiveOnly,
    /// The operation changes state but not productive authority.
    NeutralOnly,
    /// Some inputs are non-widening while other inputs can widen authority.
    InputDependent,
    /// The operation is intrinsically a capability-restoration transition.
    Widening,
    /// The operation supplies evidence used by a later authority decision. Raw
    /// caller assertions are not acceptable production evidence.
    EvidenceInjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideningRequirement {
    /// Appropriate only for surfaces proven never to widen authority.
    None,
    /// Transition is internal to the single authoritative owner and not a
    /// downstream-callable widening boundary.
    OwnerInternal,
    /// Current qualified observations are required, but no independent release
    /// credential is needed for the specific transition.
    FreshQualifiedEvidence,
    /// A verified exact-subject release/service/recovery capability is required.
    VerifiedCapability,
    /// Both explicit release authority and fresh evidence are required.
    VerifiedCapabilityAndFreshEvidence,
    /// Restore/restart must reconcile current and historical authority rather
    /// than treating persistence as permission.
    RecoveryReconciliation,
    /// The input must originate from a verified evidence producer rather than a
    /// caller-controlled primitive boolean/scalar.
    VerifiedEvidenceSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuthoritySourceKind {
    OperatorAuthority,
    SafetyOverride,
    ActuatorIsolation,
    DegradedOperations,
    RuntimeHealthEvidence,
    UpdateHealthEvidence,
    LifecycleReset,
    CheckpointRestore,
    PartitionRecovery,
    TemporalAssurance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthorityTransitionContract {
    /// Stable source-level surface name used by review/static tooling.
    pub surface: &'static str,
    pub class: AuthorityTransitionClass,
    pub sources: &'static [AuthoritySourceKind],
    pub widening_requirement: WideningRequirement,
    /// True when downstream code can currently reach the boundary directly.
    pub downstream_surface: bool,
}

const OPERATOR: &[AuthoritySourceKind] = &[AuthoritySourceKind::OperatorAuthority];
const SAFETY: &[AuthoritySourceKind] = &[AuthoritySourceKind::SafetyOverride];
const ACTUATOR: &[AuthoritySourceKind] = &[AuthoritySourceKind::ActuatorIsolation];
const DEGRADED: &[AuthoritySourceKind] = &[AuthoritySourceKind::DegradedOperations];
const RUNTIME_HEALTH: &[AuthoritySourceKind] = &[AuthoritySourceKind::RuntimeHealthEvidence];
const UPDATE_HEALTH: &[AuthoritySourceKind] = &[AuthoritySourceKind::UpdateHealthEvidence];
const RESET: &[AuthoritySourceKind] = &[
    AuthoritySourceKind::LifecycleReset,
    AuthoritySourceKind::OperatorAuthority,
    AuthoritySourceKind::DegradedOperations,
    AuthoritySourceKind::ActuatorIsolation,
    AuthoritySourceKind::PartitionRecovery,
    AuthoritySourceKind::TemporalAssurance,
];
const RESTORE: &[AuthoritySourceKind] = &[
    AuthoritySourceKind::CheckpointRestore,
    AuthoritySourceKind::OperatorAuthority,
    AuthoritySourceKind::DegradedOperations,
    AuthoritySourceKind::ActuatorIsolation,
    AuthoritySourceKind::PartitionRecovery,
    AuthoritySourceKind::TemporalAssurance,
];

/// Initial subterranean widening-surface inventory.
///
/// This table states the target authority contract; it does not imply the
/// current implementation already satisfies it. RA-18 adversarial tests and
/// implementation tranches close the gap surface-by-surface.
pub const AUTHORITY_TRANSITION_CONTRACTS: &[AuthorityTransitionContract] = &[
    AuthorityTransitionContract {
        surface: "SubterraneanEmbodiment::reset",
        class: AuthorityTransitionClass::InputDependent,
        sources: RESET,
        widening_requirement: WideningRequirement::RecoveryReconciliation,
        downstream_surface: true,
    },
    AuthorityTransitionContract {
        surface: "SubterraneanEmbodiment::load_operational_checkpoint",
        class: AuthorityTransitionClass::InputDependent,
        sources: RESTORE,
        widening_requirement: WideningRequirement::RecoveryReconciliation,
        downstream_surface: true,
    },
    AuthorityTransitionContract {
        surface: "SubterraneanEmbodiment::set_safety_override",
        class: AuthorityTransitionClass::InputDependent,
        sources: SAFETY,
        widening_requirement: WideningRequirement::VerifiedCapability,
        downstream_surface: true,
    },
    AuthorityTransitionContract {
        surface: "SubterraneanEmbodiment::clear_safety_override",
        class: AuthorityTransitionClass::Widening,
        sources: SAFETY,
        widening_requirement: WideningRequirement::VerifiedCapability,
        downstream_surface: true,
    },
    AuthorityTransitionContract {
        surface: "SubterraneanEmbodiment::service_isolated_actuator",
        class: AuthorityTransitionClass::Widening,
        sources: ACTUATOR,
        widening_requirement: WideningRequirement::VerifiedCapabilityAndFreshEvidence,
        downstream_surface: true,
    },
    AuthorityTransitionContract {
        surface: "SubterraneanEmbodiment::authorize_degraded_recovery_clear",
        class: AuthorityTransitionClass::Widening,
        sources: DEGRADED,
        widening_requirement: WideningRequirement::VerifiedCapabilityAndFreshEvidence,
        downstream_surface: true,
    },
    AuthorityTransitionContract {
        surface: "SubterraneanEmbodiment::set_runtime_health",
        class: AuthorityTransitionClass::EvidenceInjection,
        sources: RUNTIME_HEALTH,
        widening_requirement: WideningRequirement::VerifiedEvidenceSource,
        downstream_surface: true,
    },
    AuthorityTransitionContract {
        surface: "SubterraneanEmbodiment::observe_update_health",
        class: AuthorityTransitionClass::InputDependent,
        sources: UPDATE_HEALTH,
        widening_requirement: WideningRequirement::VerifiedEvidenceSource,
        downstream_surface: true,
    },
    AuthorityTransitionContract {
        surface: "OperatorAuthority::issue_recovery_proposal",
        class: AuthorityTransitionClass::Widening,
        sources: OPERATOR,
        widening_requirement: WideningRequirement::OwnerInternal,
        downstream_surface: false,
    },
    AuthorityTransitionContract {
        surface: "OperatorAuthority::approve_recovery",
        class: AuthorityTransitionClass::Widening,
        sources: OPERATOR,
        widening_requirement: WideningRequirement::OwnerInternal,
        downstream_surface: false,
    },
    AuthorityTransitionContract {
        surface: "OperatorAuthority::issue_qualified_recovery_proposal",
        class: AuthorityTransitionClass::Widening,
        sources: OPERATOR,
        widening_requirement: WideningRequirement::OwnerInternal,
        downstream_surface: false,
    },
    AuthorityTransitionContract {
        surface: "OperatorAuthority::approve_qualified_recovery",
        class: AuthorityTransitionClass::Widening,
        sources: OPERATOR,
        widening_requirement: WideningRequirement::OwnerInternal,
        downstream_surface: false,
    },
];

pub fn transition_contract(surface: &str) -> Option<&'static AuthorityTransitionContract> {
    AUTHORITY_TRANSITION_CONTRACTS
        .iter()
        .find(|contract| contract.surface == surface)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn widening_and_input_dependent_surfaces_declare_proof_requirements() {
        for contract in AUTHORITY_TRANSITION_CONTRACTS {
            if matches!(
                contract.class,
                AuthorityTransitionClass::Widening
                    | AuthorityTransitionClass::InputDependent
                    | AuthorityTransitionClass::EvidenceInjection
            ) {
                assert_ne!(
                    contract.widening_requirement,
                    WideningRequirement::None,
                    "{} can affect widening but declares no proof requirement",
                    contract.surface
                );
            }
        }
    }

    #[test]
    fn downstream_widening_never_uses_owner_internal_as_its_requirement() {
        for contract in AUTHORITY_TRANSITION_CONTRACTS
            .iter()
            .filter(|contract| contract.downstream_surface)
        {
            assert_ne!(
                contract.widening_requirement,
                WideningRequirement::OwnerInternal,
                "{} is downstream-callable and cannot rely on internal ownership as proof",
                contract.surface
            );
        }
    }

    #[test]
    fn registry_has_unique_surface_names() {
        let names = AUTHORITY_TRANSITION_CONTRACTS
            .iter()
            .map(|contract| contract.surface)
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), AUTHORITY_TRANSITION_CONTRACTS.len());
    }

    #[test]
    fn known_escape_hatches_are_explicitly_classified() {
        for surface in [
            "SubterraneanEmbodiment::reset",
            "SubterraneanEmbodiment::load_operational_checkpoint",
            "SubterraneanEmbodiment::set_safety_override",
            "SubterraneanEmbodiment::clear_safety_override",
            "SubterraneanEmbodiment::service_isolated_actuator",
            "SubterraneanEmbodiment::authorize_degraded_recovery_clear",
            "SubterraneanEmbodiment::set_runtime_health",
            "SubterraneanEmbodiment::observe_update_health",
        ] {
            assert!(
                transition_contract(surface).is_some(),
                "known authority-relevant public surface {surface} lacks a widening contract"
            );
        }
    }

    #[test]
    fn safety_override_setter_is_not_misclassified_as_restrictive_only() {
        let contract = transition_contract("SubterraneanEmbodiment::set_safety_override")
            .expect("safety setter contract");
        assert_eq!(contract.class, AuthorityTransitionClass::InputDependent);
        assert_eq!(
            contract.widening_requirement,
            WideningRequirement::VerifiedCapability
        );
    }

    #[test]
    fn raw_evidence_injection_requires_verified_producer() {
        let health = transition_contract("SubterraneanEmbodiment::set_runtime_health")
            .expect("runtime-health contract");
        assert_eq!(health.class, AuthorityTransitionClass::EvidenceInjection);
        assert_eq!(
            health.widening_requirement,
            WideningRequirement::VerifiedEvidenceSource
        );
    }
}
