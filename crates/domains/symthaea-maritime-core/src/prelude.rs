// SPDX-License-Identifier: AGPL-3.0-or-later
//! Convenience imports for maritime platform adapters.

pub use crate::{
    AuthorityContext, AuthorityLease, BayBoundary, BayBoundaryDecision, BayBoundaryGateContext,
    BayBoundaryRefusal, BayMediumState, BayOccupancy, BayPressureQualification, BoundaryPosition,
    ComponentHealth, DependencyState, DockingObservation, DockingPhase, FailureObservation,
    FleetAssuranceReport, FleetMemberReport, HealthSeverity, MaritimeBayObservation,
    MaritimeCapability, MaritimeOperatingMode, MaritimePlatformKind, MaritimeResourceEnvelope,
    MaritimeResourceRequest, MaritimeServiceContract, MaritimeServiceKind, MaritimeState,
    NavigationQuality, OperatingEnvelope, OperatingModeContext, OperatingModeDecision,
    OperatingModeRefusal, OperationalLimitAssessment, OperationalLimitStatus, PlatformHealth,
    ProgressiveFailureTrace, ProgressiveFailureViolation, ReleaseDecision, ReleaseGateContext,
    ReleaseRefusal, ResidualUtilitySnapshot, ResourceAllocationProjection, ResourceDecision,
    ResourceGateContext, ResourcePriority, ResourceRefusal, ServiceGateContext,
    ServiceStartDecision, ServiceStartRefusal, UtilityAvailability, baseline_envelope,
    evaluate_bay_boundary_open, evaluate_operating_mode, evaluate_release, evaluate_resource_request,
    evaluate_service_start,
};
