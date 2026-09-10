// SPDX-License-Identifier: AGPL-3.0-or-later
//! Convenience imports for maritime platform adapters.

pub use crate::{
    AuthorityContext, AuthorityLease, BayBoundary, BayBoundaryDecision, BayBoundaryGateContext,
    BayBoundaryRefusal, BayMediumState, BayOccupancy, BayPressureQualification, BoundaryPosition,
    ComponentHealth, DependencyState, DockingObservation, DockingPhase, FailureObservation,
    FleetAssuranceReport, FleetMemberReport, HealthSeverity, MaintenanceDisposition,
    MaintenanceRecord, MaintenanceStage, MaritimeBayObservation, MaritimeCapability,
    MaritimeOperatingMode, MaritimePlatformKind, MaritimeResourceEnvelope, MaritimeResourceRequest,
    MaritimeServiceContract, MaritimeServiceKind, MaritimeState, NavigationQuality,
    OperatingEnvelope, OperatingModeContext, OperatingModeDecision, OperatingModeRefusal,
    OperationalLimitAssessment, OperationalLimitStatus, PlatformHealth, ProgressiveFailureTrace,
    ProgressiveFailureViolation, ReleaseDecision, ReleaseGateContext, ReleaseRefusal,
    ResidualUtilitySnapshot, ResourceAllocationProjection, ResourceDecision, ResourceGateContext,
    ResourcePriority, ResourceRefusal, ReturnToServiceDecision, ReturnToServiceGateContext,
    ReturnToServiceRefusal, ServiceGateContext, ServiceStartDecision, ServiceStartRefusal,
    UtilityAvailability, baseline_envelope, evaluate_bay_boundary_open, evaluate_operating_mode,
    evaluate_release, evaluate_resource_request, evaluate_return_to_service, evaluate_service_start,
};
