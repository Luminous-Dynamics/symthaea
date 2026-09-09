// SPDX-License-Identifier: AGPL-3.0-or-later
//! Convenience imports for maritime platform adapters.

pub use crate::{
    AuthenticatedMachineSession, AuthorityContext, AuthorityLease, ComponentHealth,
    DependencyState, FailureObservation, FleetAssuranceReport, FleetMemberReport,
    HealthSeverity, MachineSessionContext, MachineSessionTrust, MaritimeCapability,
    MaritimePlatformKind, MaritimeState, NavigationQuality, OperatingEnvelope, PlatformHealth,
    ProgressiveFailureTrace, ProgressiveFailureViolation, ResidualUtilitySnapshot,
    UtilityAvailability, baseline_envelope, evaluate_machine_session,
};
