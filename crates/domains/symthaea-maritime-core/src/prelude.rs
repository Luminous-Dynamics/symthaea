// SPDX-License-Identifier: AGPL-3.0-or-later
//! Convenience imports for maritime platform adapters.

pub use crate::{
    AuthorityContext, AuthorityLease, ComponentHealth, DependencyState, FailureObservation,
    FleetAssuranceReport, FleetMemberReport, HealthSeverity, MaritimeCapability,
    MaritimeOperatingMode, MaritimePlatformKind, MaritimeState, NavigationQuality,
    OperatingEnvelope, OperatingModeContext, OperatingModeDecision, OperatingModeRefusal,
    OperationalLimitAssessment, OperationalLimitStatus, PlatformHealth, ProgressiveFailureTrace,
    ProgressiveFailureViolation, ResidualUtilitySnapshot, UtilityAvailability, baseline_envelope,
    evaluate_operating_mode,
};
