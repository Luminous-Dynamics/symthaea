// SPDX-License-Identifier: AGPL-3.0-or-later
//! Convenience imports for maritime platform adapters.

pub use crate::{
    AuthenticatedMachineSession, AuthorityContext, AuthorityLease, ComponentHealth,
    DependencyState, FailureObservation, FleetAssuranceReport, FleetMemberReport,
    HealthSeverity, HistoricalSessionHandoffError, HistoricallyQualifiedMachineSessionV1,
    MachineSessionContext, MachineSessionPolicy, MachineSessionTrust, MaritimeCapability,
    MaritimeObservationKind, MaritimePlatformKind, MaritimeState, NavigationQuality,
    ObservationBindingError, ObservationSourceGrantV1, ObservationSourcePolicyError,
    ObservationSourcePolicyV1, OperatingEnvelope, PlatformHealth, ProgressiveFailureTrace,
    ProgressiveFailureViolation, ResidualUtilitySnapshot, SESSION_BOUND_OBSERVATION_PREFIX_V1,
    SessionBoundObservationV1, UtilityAvailability, baseline_envelope,
    bind_observation_to_historically_qualified_session, bind_observation_to_trusted_session,
    evaluate_machine_session,
};
