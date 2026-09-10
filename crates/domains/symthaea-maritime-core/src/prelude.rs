// SPDX-License-Identifier: AGPL-3.0-or-later
//! Convenience imports for maritime platform adapters.

pub use crate::{
    AuthorityContext, AuthorityLease, ComponentHealth, DependencyState, DockingObservation,
    DockingPhase, FailureObservation, FleetAssuranceReport, FleetMemberReport, HealthSeverity,
    MaritimeCapability, MaritimeOperatingMode, MaritimePlatformKind, MaritimeServiceContract,
    MaritimeServiceKind, MaritimeState, NavigationQuality, OperatingEnvelope,
    OperatingModeContext, OperatingModeDecision, OperatingModeRefusal, OperationalLimitAssessment,
    OperationalLimitStatus, PlatformHealth, ProgressiveFailureTrace, ProgressiveFailureViolation,
    ReleaseDecision, ReleaseGateContext, ReleaseRefusal, ResidualUtilitySnapshot,
    ServiceGateContext, ServiceStartDecision, ServiceStartRefusal, UtilityAvailability,
    baseline_envelope, evaluate_operating_mode, evaluate_release, evaluate_service_start,
};
