//! Non-compensatory household action-policy evaluation.
//!
//! This module evaluates whether a proposed action may be presented to a
//! local controller for its own decision. It never emits actuator commands,
//! writes device state, or overrides local interlocks. A positive result is
//! not an instruction to operate equipment and is not proof of service.
//!
//! The coordinator must preserve local manual controls. Local controller
//! refusal is terminal for the request; callers must not retry it as a bypass.

use crate::water_quality::{WaterQualificationStatus, WaterUseClass};
use crate::EvaluationError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const NEGATIVE_CASE_IDS: [&str; 14] = [
    "HRA-001", "HRA-002", "HRA-003", "HRA-004", "HRA-005", "HRA-006", "HRA-007",
    "HRA-008", "HRA-009", "HRA-010", "HRA-011", "HRA-012", "HRA-013", "HRA-014",
];
const POSITIVE_CASE_IDS: [&str; 3] = ["HRA-POS-001", "HRA-POS-002", "HRA-POS-003"];
const ARBITRATION_MANIFEST: &str =
    include_str!("../../../docs/research/household-services/household-resource-arbitration-scenarios-v0.1.json");

fn invalid(path: &'static str, message: impl Into<String>) -> EvaluationError {
    EvaluationError::new(path, message)
}

fn nonempty(path: &'static str, value: &str) -> Result<(), EvaluationError> {
    if value.trim().is_empty() {
        return Err(invalid(path, "must not be empty"));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GateState {
    Pass,
    Fail,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinatorState {
    Available,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalControllerState {
    Ready,
    Refused,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorizationState {
    Granted,
    Missing,
    Revoked,
    Expired,
    ScopeMismatch,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HouseholdActionKind {
    MaintainLocalService,
    RouteWater,
    RefillBuffer,
    DeferOptionalLoad,
    LeakIsolation,
    ScheduleThermalLoad,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterQualificationEvidence {
    pub status: WaterQualificationStatus,
    pub declared_use: WaterUseClass,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorizationEvidence {
    pub state: AuthorizationState,
    pub policy_id: Option<String>,
    pub policy_revision: Option<String>,
    pub scope_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LocalControllerEvidence {
    pub state: LocalControllerState,
    pub capability_revision: String,
    pub manual_control_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HouseholdActionRequest {
    pub request_id: String,
    pub action_kind: HouseholdActionKind,
    pub required_authorization_scope_id: String,
    pub required_capability_revision: String,
    pub requested_water_use: Option<WaterUseClass>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HouseholdCoordinationSnapshot {
    pub safety_gate: GateState,
    pub critical_service_floor: GateState,
    pub water_qualification: Option<WaterQualificationEvidence>,
    pub authorization: AuthorizationEvidence,
    pub coordinator_state: CoordinatorState,
    pub local_controller: LocalControllerEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HouseholdActionDecision {
    EligibleForLocalControllerEvaluation,
    RejectedSafety,
    SafetyUnknown,
    RejectedWaterUseQualification,
    WaterUseQualificationUnknown,
    RejectedCriticalServiceFloor,
    CriticalServiceFloorUnknown,
    AuthorizationMissing,
    AuthorizationRevoked,
    AuthorizationExpired,
    AuthorizationScopeMismatch,
    AuthorizationUnknown,
    CoordinatorUnavailable,
    LocalControllerRefused,
    LocalControllerUnavailable,
    LocalControllerUnknown,
    CapabilityRevisionMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct HouseholdActionEvaluation {
    pub request_id: String,
    pub decision: HouseholdActionDecision,
    pub reason_code: String,
    /// Eligibility means the caller may present the request to the named local
    /// controller. It does not mean the controller accepted or executed it.
    pub local_request_eligible: bool,
    /// This evaluator is policy-only and must never directly operate hardware.
    pub actuator_command_emitted: bool,
    pub local_manual_control_available: bool,
    pub local_controller_final_authority: bool,
    pub authorization_policy_revision: Option<String>,
    pub capability_revision: String,
}

fn finish(
    request: &HouseholdActionRequest,
    snapshot: &HouseholdCoordinationSnapshot,
    decision: HouseholdActionDecision,
    reason_code: &'static str,
    eligible: bool,
) -> HouseholdActionEvaluation {
    HouseholdActionEvaluation {
        request_id: request.request_id.clone(),
        decision,
        reason_code: reason_code.to_owned(),
        local_request_eligible: eligible,
        actuator_command_emitted: false,
        local_manual_control_available: snapshot.local_controller.manual_control_available,
        local_controller_final_authority: true,
        authorization_policy_revision: snapshot.authorization.policy_revision.clone(),
        capability_revision: snapshot.local_controller.capability_revision.clone(),
    }
}

/// Evaluate a request in strict, non-compensatory order.
///
/// No score or cost/energy benefit can compensate for a failed or unknown
/// safety gate, use qualification, critical service floor, authorization,
/// coordinator availability, or local capability. The result only determines
/// whether a typed request may be presented to the local controller.
pub fn evaluate_household_action(
    request: &HouseholdActionRequest,
    snapshot: &HouseholdCoordinationSnapshot,
) -> Result<HouseholdActionEvaluation, EvaluationError> {
    nonempty("request.request_id", &request.request_id)?;
    nonempty(
        "request.required_authorization_scope_id",
        &request.required_authorization_scope_id,
    )?;
    nonempty(
        "request.required_capability_revision",
        &request.required_capability_revision,
    )?;
    nonempty(
        "local_controller.capability_revision",
        &snapshot.local_controller.capability_revision,
    )?;

    match snapshot.safety_gate {
        GateState::Pass => {}
        GateState::Fail => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::RejectedSafety,
                "hard_safety_gate_failed",
                false,
            ));
        }
        GateState::Unknown | GateState::NotApplicable => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::SafetyUnknown,
                "hard_safety_gate_unknown_or_not_applicable",
                false,
            ));
        }
    }

    if let Some(requested_use) = request.requested_water_use {
        let Some(evidence) = &snapshot.water_qualification else {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::WaterUseQualificationUnknown,
                "water_qualification_evidence_missing",
                false,
            ));
        };
        if evidence.status != WaterQualificationStatus::QualifiedForDeclaredUse {
            let (decision, reason) = match evidence.status {
                WaterQualificationStatus::KnownNonCompliance
                | WaterQualificationStatus::TreatmentStateUnqualified => (
                    HouseholdActionDecision::RejectedWaterUseQualification,
                    "water_use_known_noncompliance_or_treatment_unqualified",
                ),
                WaterQualificationStatus::UnknownEvidenceInsufficient
                | WaterQualificationStatus::ExpiredOrMonitoringOverdue
                | WaterQualificationStatus::SourceOrSamplePointMismatch
                | WaterQualificationStatus::CorrectiveActionPending
                | WaterQualificationStatus::QualifiedForDeclaredUse => (
                    HouseholdActionDecision::WaterUseQualificationUnknown,
                    "water_use_evidence_unknown_expired_mismatched_or_pending",
                ),
            };
            return Ok(finish(request, snapshot, decision, reason, false));
        }
        if evidence.declared_use != requested_use {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::RejectedWaterUseQualification,
                "water_use_class_mismatch",
                false,
            ));
        }
    }

    match snapshot.critical_service_floor {
        GateState::Pass => {}
        GateState::Fail => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::RejectedCriticalServiceFloor,
                "protected_service_floor_would_be_violated",
                false,
            ));
        }
        GateState::Unknown | GateState::NotApplicable => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::CriticalServiceFloorUnknown,
                "protected_service_floor_unknown_or_not_applicable",
                false,
            ));
        }
    }

    match snapshot.authorization.state {
        AuthorizationState::Granted => {
            let policy_id = snapshot.authorization.policy_id.as_deref();
            let policy_revision = snapshot.authorization.policy_revision.as_deref();
            let scope_id = snapshot.authorization.scope_id.as_deref();
            if policy_id.is_none_or(str::is_empty)
                || policy_revision.is_none_or(str::is_empty)
                || scope_id.is_none_or(str::is_empty)
            {
                return Ok(finish(
                    request,
                    snapshot,
                    HouseholdActionDecision::AuthorizationUnknown,
                    "granted_authorization_missing_policy_identity",
                    false,
                ));
            }
            if scope_id != Some(request.required_authorization_scope_id.as_str()) {
                return Ok(finish(
                    request,
                    snapshot,
                    HouseholdActionDecision::AuthorizationScopeMismatch,
                    "authorization_scope_does_not_cover_requested_action",
                    false,
                ));
            }
        }
        AuthorizationState::Missing => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::AuthorizationMissing,
                "explicit_or_standing_authorization_missing",
                false,
            ));
        }
        AuthorizationState::Revoked => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::AuthorizationRevoked,
                "authorization_revoked",
                false,
            ));
        }
        AuthorizationState::Expired => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::AuthorizationExpired,
                "authorization_expired",
                false,
            ));
        }
        AuthorizationState::ScopeMismatch => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::AuthorizationScopeMismatch,
                "authorization_scope_mismatch",
                false,
            ));
        }
        AuthorizationState::Unknown => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::AuthorizationUnknown,
                "authorization_state_unknown",
                false,
            ));
        }
    }

    if snapshot.coordinator_state == CoordinatorState::Unavailable {
        return Ok(finish(
            request,
            snapshot,
            HouseholdActionDecision::CoordinatorUnavailable,
            "coordinator_unavailable_local_manual_path_only",
            false,
        ));
    }

    match snapshot.local_controller.state {
        LocalControllerState::Ready => {}
        LocalControllerState::Refused => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::LocalControllerRefused,
                "local_controller_refused_request_no_retry_or_override",
                false,
            ));
        }
        LocalControllerState::Unavailable => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::LocalControllerUnavailable,
                "local_controller_unavailable",
                false,
            ));
        }
        LocalControllerState::Unknown => {
            return Ok(finish(
                request,
                snapshot,
                HouseholdActionDecision::LocalControllerUnknown,
                "local_controller_state_unknown",
                false,
            ));
        }
    }

    if snapshot.local_controller.capability_revision != request.required_capability_revision {
        return Ok(finish(
            request,
            snapshot,
            HouseholdActionDecision::CapabilityRevisionMismatch,
            "local_capability_revision_changed",
            false,
        ));
    }

    Ok(finish(
        request,
        snapshot,
        HouseholdActionDecision::EligibleForLocalControllerEvaluation,
        "all_noncompensatory_gates_passed_local_controller_must_decide",
        true,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyCarrier {
    Electrical,
    UsefulThermal,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyObservation {
    pub carrier: EnergyCarrier,
    pub value_kwh: f64,
    pub system_boundary_id: String,
    pub measurement_point_id: String,
    pub window_start_unix_ms: i64,
    pub window_end_unix_ms: i64,
    pub calibration_qualified: bool,
    pub calibration_record_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct MatchedEnergyRatio {
    pub ratio_name: String,
    pub ratio_value: f64,
    pub numerator_carrier: EnergyCarrier,
    pub denominator_carrier: EnergyCarrier,
    pub system_boundary_id: String,
    pub window_start_unix_ms: i64,
    pub window_end_unix_ms: i64,
    pub electrical_input_kwh: f64,
    pub useful_thermal_output_kwh: f64,
    pub energy_quantities_converted: bool,
}

/// Calculate a thermal performance ratio only when both carriers use the
/// same declared system boundary and exact observation window, and each is
/// backed by a qualified calibration record. This derives a ratio; it never
/// converts electrical kWh into useful-thermal kWh or vice versa.
pub fn calculate_matched_thermal_cop(
    electrical: &EnergyObservation,
    useful_thermal: &EnergyObservation,
) -> Result<MatchedEnergyRatio, EvaluationError> {
    if electrical.carrier != EnergyCarrier::Electrical
        || useful_thermal.carrier != EnergyCarrier::UsefulThermal
    {
        return Err(invalid(
            "energy.carrier",
            "COP requires electrical input and useful-thermal output in their declared roles",
        ));
    }
    if !electrical.value_kwh.is_finite()
        || electrical.value_kwh <= 0.0
        || !useful_thermal.value_kwh.is_finite()
        || useful_thermal.value_kwh < 0.0
    {
        return Err(invalid(
            "energy.value_kwh",
            "electrical input must be positive and both quantities finite",
        ));
    }
    nonempty("energy.system_boundary_id", &electrical.system_boundary_id)?;
    nonempty(
        "energy.measurement_point_id",
        &electrical.measurement_point_id,
    )?;
    nonempty(
        "energy.measurement_point_id",
        &useful_thermal.measurement_point_id,
    )?;
    if electrical.system_boundary_id != useful_thermal.system_boundary_id {
        return Err(invalid(
            "energy.system_boundary_id",
            "electrical and thermal observations must share one declared system boundary",
        ));
    }
    if electrical.window_start_unix_ms >= electrical.window_end_unix_ms
        || useful_thermal.window_start_unix_ms >= useful_thermal.window_end_unix_ms
        || electrical.window_start_unix_ms != useful_thermal.window_start_unix_ms
        || electrical.window_end_unix_ms != useful_thermal.window_end_unix_ms
    {
        return Err(invalid(
            "energy.observation_window",
            "electrical and thermal observations must use the same positive-duration window",
        ));
    }
    for (name, observation) in [
        ("electrical", electrical),
        ("useful_thermal", useful_thermal),
    ] {
        if !observation.calibration_qualified {
            return Err(invalid(name, "qualified calibration is required"));
        }
        let digest = observation
            .calibration_record_sha256
            .as_deref()
            .ok_or_else(|| invalid(name, "calibration record digest is required"))?;
        if !valid_sha256(digest) {
            return Err(invalid(name, "calibration record digest must be SHA-256"));
        }
    }
    let ratio_value = useful_thermal.value_kwh / electrical.value_kwh;
    if !ratio_value.is_finite() {
        return Err(invalid("energy.ratio_value", "computed ratio is not finite"));
    }
    Ok(MatchedEnergyRatio {
        ratio_name: "coefficient_of_performance".to_owned(),
        ratio_value,
        numerator_carrier: EnergyCarrier::UsefulThermal,
        denominator_carrier: EnergyCarrier::Electrical,
        system_boundary_id: electrical.system_boundary_id.clone(),
        window_start_unix_ms: electrical.window_start_unix_ms,
        window_end_unix_ms: electrical.window_end_unix_ms,
        electrical_input_kwh: electrical.value_kwh,
        useful_thermal_output_kwh: useful_thermal.value_kwh,
        energy_quantities_converted: false,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceDependency {
    pub resource_id: String,
    pub dependent_service_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct CommonModeImpact {
    pub failed_resource_id: String,
    pub affected_service_ids: Vec<String>,
}

/// Report resources whose failure affects two or more distinct services.
/// This is a topology observation, not a prediction that the services will
/// fail; callers must separately validate dependency completeness.
pub fn find_common_mode_impacts(
    dependencies: &[ResourceDependency],
    failed_resource_ids: &BTreeSet<String>,
) -> Result<Vec<CommonModeImpact>, EvaluationError> {
    let mut by_resource: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for dependency in dependencies {
        nonempty("dependencies.resource_id", &dependency.resource_id)?;
        nonempty(
            "dependencies.dependent_service_id",
            &dependency.dependent_service_id,
        )?;
        by_resource
            .entry(&dependency.resource_id)
            .or_default()
            .insert(&dependency.dependent_service_id);
    }
    let mut impacts = Vec::new();
    for failed in failed_resource_ids {
        nonempty("failed_resource_ids", failed)?;
        if let Some(services) = by_resource.get(failed.as_str())
            && services.len() >= 2
        {
            impacts.push(CommonModeImpact {
                failed_resource_id: failed.clone(),
                affected_service_ids: services.iter().map(|id| (*id).to_owned()).collect(),
            });
        }
    }
    Ok(impacts)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryDeliveryObservation {
    DeliveryObserved,
    NoDeliveryObserved,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryOutcome {
    CommandNotAccepted,
    DeliveryObservedAtBoundary,
    NoServiceCredit,
    OutcomeUnknown,
}

/// Keep command acceptance separate from physical delivery. Even a positive
/// boundary observation is only an observation; other quality/service gates
/// remain separately applicable.
pub fn classify_boundary_outcome(
    local_controller_accepted: bool,
    observation: BoundaryDeliveryObservation,
) -> BoundaryOutcome {
    if !local_controller_accepted {
        return BoundaryOutcome::CommandNotAccepted;
    }
    match observation {
        BoundaryDeliveryObservation::DeliveryObserved => {
            BoundaryOutcome::DeliveryObservedAtBoundary
        }
        BoundaryDeliveryObservation::NoDeliveryObserved => BoundaryOutcome::NoServiceCredit,
        BoundaryDeliveryObservation::Unknown => BoundaryOutcome::OutcomeUnknown,
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArbitrationScenario {
    pub id: String,
    pub class: String,
    pub stimulus: String,
    pub inputs: Value,
    pub expected_decision: String,
    pub required_assertions: Vec<String>,
    pub must_not: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArbitrationManifest {
    pub schema_version: String,
    pub corpus_id: String,
    pub evidence_class: String,
    pub fixture_status: String,
    pub decision_authority: String,
    pub scenarios: Vec<ArbitrationScenario>,
    pub positive_controls: Vec<ArbitrationScenario>,
    pub positive_control_requirement: String,
    pub acceptance_gates: Value,
    pub evaluation_outputs: Vec<String>,
    pub interpretation_guardrails: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ArbitrationManifestValidation {
    pub corpus_id: String,
    pub negative_case_count: usize,
    pub positive_control_count: usize,
    pub synthetic_only: bool,
    pub live_actuation_authorized: bool,
}

/// Parse and validate the versioned machine-readable challenge manifest.
/// Expected decisions are fixture metadata, not policy logic; the typed
/// evaluator above computes its result independently.
pub fn validate_arbitration_manifest(
    source: &str,
) -> Result<(ArbitrationManifest, ArbitrationManifestValidation), EvaluationError> {
    let manifest: ArbitrationManifest = serde_json::from_str(source)
        .map_err(|error| invalid("arbitration_manifest", error.to_string()))?;
    if manifest.schema_version != "0.1.0"
        || manifest.corpus_id != "HOUSEHOLD-RESOURCE-ARBITRATION-001"
        || manifest.evidence_class != "synthetic_benchmark_design_only"
        || manifest.fixture_status != "proposed_not_executed"
    {
        return Err(invalid(
            "arbitration_manifest.header",
            "unsupported corpus, schema, evidence class or fixture status",
        ));
    }
    if manifest.decision_authority != "none_this_manifest_does_not_authorize_live_actuation" {
        return Err(invalid(
            "arbitration_manifest.decision_authority",
            "manifest must not grant actuator authority",
        ));
    }
    if manifest.scenarios.len() != NEGATIVE_CASE_IDS.len()
        || manifest.positive_controls.len() != POSITIVE_CASE_IDS.len()
    {
        return Err(invalid(
            "arbitration_manifest.case_count",
            "negative and positive-control case counts do not match the frozen v0.1 corpus",
        ));
    }
    let mut observed = BTreeSet::new();
    for scenario in manifest
        .scenarios
        .iter()
        .chain(manifest.positive_controls.iter())
    {
        nonempty("arbitration_manifest.scenario.id", &scenario.id)?;
        nonempty("arbitration_manifest.scenario.class", &scenario.class)?;
        nonempty("arbitration_manifest.scenario.stimulus", &scenario.stimulus)?;
        nonempty(
            "arbitration_manifest.scenario.expected_decision",
            &scenario.expected_decision,
        )?;
        if scenario.required_assertions.is_empty() || scenario.must_not.is_empty() {
            return Err(invalid(
                "arbitration_manifest.scenario.assertions",
                "each case requires positive assertions and prohibited behaviors",
            ));
        }
        if !observed.insert(scenario.id.as_str()) {
            return Err(invalid(
                "arbitration_manifest.scenario.id",
                "duplicate case identifier",
            ));
        }
    }
    for expected in NEGATIVE_CASE_IDS {
        if !observed.contains(expected) {
            return Err(invalid(
                "arbitration_manifest.scenarios",
                format!("missing frozen negative case {expected}"),
            ));
        }
    }
    for expected in POSITIVE_CASE_IDS {
        if !observed.contains(expected) {
            return Err(invalid(
                "arbitration_manifest.positive_controls",
                format!("missing frozen positive control {expected}"),
            ));
        }
    }
    if manifest.positive_control_requirement.trim().is_empty() {
        return Err(invalid(
            "arbitration_manifest.positive_control_requirement",
            "non-vacuity requirement is mandatory",
        ));
    }
    Ok((
        manifest,
        ArbitrationManifestValidation {
            corpus_id: "HOUSEHOLD-RESOURCE-ARBITRATION-001".to_owned(),
            negative_case_count: NEGATIVE_CASE_IDS.len(),
            positive_control_count: POSITIVE_CASE_IDS.len(),
            synthetic_only: true,
            live_actuation_authorized: false,
        },
    ))
}

pub fn validate_bundled_arbitration_manifest(
) -> Result<ArbitrationManifestValidation, EvaluationError> {
    let (_, validation) = validate_arbitration_manifest(ARBITRATION_MANIFEST)?;
    Ok(validation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_request() -> HouseholdActionRequest {
        HouseholdActionRequest {
            request_id: "request-001".into(),
            action_kind: HouseholdActionKind::ScheduleThermalLoad,
            required_authorization_scope_id: "home-policy-scope-1".into(),
            required_capability_revision: "capability-r3".into(),
            requested_water_use: None,
        }
    }

    fn valid_snapshot() -> HouseholdCoordinationSnapshot {
        HouseholdCoordinationSnapshot {
            safety_gate: GateState::Pass,
            critical_service_floor: GateState::Pass,
            water_qualification: None,
            authorization: AuthorizationEvidence {
                state: AuthorizationState::Granted,
                policy_id: Some("home-policy-1".into()),
                policy_revision: Some("policy-r7".into()),
                scope_id: Some("home-policy-scope-1".into()),
            },
            coordinator_state: CoordinatorState::Available,
            local_controller: LocalControllerEvidence {
                state: LocalControllerState::Ready,
                capability_revision: "capability-r3".into(),
                manual_control_available: true,
            },
        }
    }

    fn fixture_expected<'a>(id: &str) -> &'a str {
        let (manifest, _) = validate_arbitration_manifest(ARBITRATION_MANIFEST).unwrap();
        let case = manifest
            .scenarios
            .iter()
            .chain(manifest.positive_controls.iter())
            .find(|case| case.id == id)
            .unwrap_or_else(|| panic!("fixture case {id} missing"));
        // The caller checks the fixture value against a separately authored
        // semantic assertion in each test.
        Box::leak(case.expected_decision.clone().into_boxed_str())
    }

    fn evaluate(
        request: &HouseholdActionRequest,
        snapshot: &HouseholdCoordinationSnapshot,
    ) -> HouseholdActionEvaluation {
        evaluate_household_action(request, snapshot).unwrap()
    }

    #[test]
    fn bundled_manifest_has_exact_negative_and_positive_case_sets() {
        let validation = validate_bundled_arbitration_manifest().unwrap();
        assert_eq!(validation.negative_case_count, 14);
        assert_eq!(validation.positive_control_count, 3);
        assert!(validation.synthetic_only);
        assert!(!validation.live_actuation_authorized);
        for id in NEGATIVE_CASE_IDS {
            assert!(!fixture_expected(id).is_empty());
        }
        for id in POSITIVE_CASE_IDS {
            assert!(!fixture_expected(id).is_empty());
        }
    }

    #[test]
    fn positive_control_hra_pos_001_is_eligible_only_for_local_evaluation() {
        assert_eq!(fixture_expected("HRA-POS-001"), "EligibleForLocalControllerEvaluation");
        let result = evaluate(&valid_request(), &valid_snapshot());
        assert_eq!(result.decision, HouseholdActionDecision::EligibleForLocalControllerEvaluation);
        assert!(result.local_request_eligible);
        assert!(!result.actuator_command_emitted);
        assert!(result.local_controller_final_authority);
        assert!(result.local_manual_control_available);
    }

    #[test]
    fn hra_001_rejects_failed_thermal_or_hygiene_safety_gate() {
        assert_eq!(fixture_expected("HRA-001"), "RejectCandidate");
        let mut snapshot = valid_snapshot();
        snapshot.safety_gate = GateState::Fail;
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::RejectedSafety);
        assert!(!result.local_request_eligible);
        assert!(!result.actuator_command_emitted);
    }

    #[test]
    fn hra_002_unknown_water_quality_gets_no_potable_credit() {
        assert_eq!(fixture_expected("HRA-002"), "RejectPotableCredit");
        let mut request = valid_request();
        request.requested_water_use = Some(WaterUseClass::Drinking);
        let mut snapshot = valid_snapshot();
        snapshot.water_qualification = Some(WaterQualificationEvidence {
            status: WaterQualificationStatus::UnknownEvidenceInsufficient,
            declared_use: WaterUseClass::Drinking,
        });
        let result = evaluate(&request, &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::WaterUseQualificationUnknown);
        assert!(!result.local_request_eligible);
    }

    #[test]
    fn hra_003_unqualified_blend_does_not_inherit_parent_quality() {
        assert_eq!(fixture_expected("HRA-003"), "CreateDerivedStreamUnknown");
        let mut request = valid_request();
        request.requested_water_use = Some(WaterUseClass::Drinking);
        let mut snapshot = valid_snapshot();
        snapshot.water_qualification = Some(WaterQualificationEvidence {
            status: WaterQualificationStatus::UnknownEvidenceInsufficient,
            declared_use: WaterUseClass::Drinking,
        });
        let result = evaluate(&request, &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::WaterUseQualificationUnknown);
    }

    #[test]
    fn hra_004_below_floor_delivery_is_not_credited_as_usable_reserve() {
        assert_eq!(fixture_expected("HRA-004"), "InventoryOnlyNoServicePromise");
        let mut snapshot = valid_snapshot();
        snapshot.critical_service_floor = GateState::Fail;
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::RejectedCriticalServiceFloor);
    }

    fn energy_observations() -> (EnergyObservation, EnergyObservation) {
        (
            EnergyObservation {
                carrier: EnergyCarrier::Electrical,
                value_kwh: 2.0,
                system_boundary_id: "heat-pump-system-1".into(),
                measurement_point_id: "electrical-input-meter".into(),
                window_start_unix_ms: 1_000,
                window_end_unix_ms: 3_600_000,
                calibration_qualified: true,
                calibration_record_sha256: Some("a".repeat(64)),
            },
            EnergyObservation {
                carrier: EnergyCarrier::UsefulThermal,
                value_kwh: 6.0,
                system_boundary_id: "heat-pump-system-1".into(),
                measurement_point_id: "thermal-output-meter".into(),
                window_start_unix_ms: 1_000,
                window_end_unix_ms: 3_600_000,
                calibration_qualified: true,
                calibration_record_sha256: Some("b".repeat(64)),
            },
        )
    }

    #[test]
    fn hra_005_refuses_cop_across_different_observation_windows() {
        assert_eq!(fixture_expected("HRA-005"), "RejectCarrierConversion");
        let (electrical, mut thermal) = energy_observations();
        thermal.window_end_unix_ms += 1;
        assert!(calculate_matched_thermal_cop(&electrical, &thermal).is_err());
    }

    #[test]
    fn hra_005_refuses_cop_across_different_system_boundaries() {
        let (electrical, mut thermal) = energy_observations();
        thermal.system_boundary_id = "different-system".into();
        assert!(calculate_matched_thermal_cop(&electrical, &thermal).is_err());
    }

    #[test]
    fn matched_carrier_observations_derive_ratio_without_unit_conversion() {
        let (electrical, thermal) = energy_observations();
        let result = calculate_matched_thermal_cop(&electrical, &thermal).unwrap();
        assert_eq!(result.ratio_value, 3.0);
        assert!(!result.energy_quantities_converted);
        assert_eq!(result.electrical_input_kwh, 2.0);
        assert_eq!(result.useful_thermal_output_kwh, 6.0);
    }

    #[test]
    fn hra_006_reports_a_shared_failed_dependency() {
        assert_eq!(fixture_expected("HRA-006"), "CommonModeLossDeclared");
        let dependencies = vec![
            ResourceDependency { resource_id: "grid-pump-feed".into(), dependent_service_id: "water-pumping".into() },
            ResourceDependency { resource_id: "grid-pump-feed".into(), dependent_service_id: "household-controller".into() },
            ResourceDependency { resource_id: "grid-pump-feed".into(), dependent_service_id: "thermal-recirculation".into() },
        ];
        let failed = BTreeSet::from(["grid-pump-feed".to_owned()]);
        let impacts = find_common_mode_impacts(&dependencies, &failed).unwrap();
        assert_eq!(impacts.len(), 1);
        assert_eq!(impacts[0].affected_service_ids.len(), 3);
    }

    #[test]
    fn hra_007_coordinator_loss_preserves_independent_manual_control() {
        assert_eq!(fixture_expected("HRA-007"), "CoordinatorUnavailableLocalControlRetained");
        let mut snapshot = valid_snapshot();
        snapshot.coordinator_state = CoordinatorState::Unavailable;
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::CoordinatorUnavailable);
        assert!(result.local_manual_control_available);
        assert!(!result.local_request_eligible);
        assert!(!result.actuator_command_emitted);
    }

    #[test]
    fn hra_008_optional_deferral_requires_current_authorization() {
        assert_eq!(fixture_expected("HRA-008"), "RequireAuthorizationOrKeepSchedule");
        let mut request = valid_request();
        request.action_kind = HouseholdActionKind::DeferOptionalLoad;
        let mut snapshot = valid_snapshot();
        snapshot.authorization.state = AuthorizationState::Missing;
        let result = evaluate(&request, &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::AuthorizationMissing);
    }

    #[test]
    fn hra_009_unknown_safety_state_blocks_optimistic_plan() {
        assert_eq!(fixture_expected("HRA-009"), "WidenUncertaintyAndBlockOptimisticPlan");
        let mut snapshot = valid_snapshot();
        snapshot.safety_gate = GateState::Unknown;
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::SafetyUnknown);
    }

    #[test]
    fn hra_010_local_refill_envelope_refusal_is_final() {
        assert_eq!(fixture_expected("HRA-010"), "LocalControllerEnforcesApprovedRamp");
        let mut request = valid_request();
        request.action_kind = HouseholdActionKind::RefillBuffer;
        let mut snapshot = valid_snapshot();
        snapshot.local_controller.state = LocalControllerState::Refused;
        let result = evaluate(&request, &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::LocalControllerRefused);
        assert!(!result.local_request_eligible);
    }

    #[test]
    fn hra_011_model_confidence_cannot_override_local_isolation_policy() {
        assert_eq!(fixture_expected("HRA-011"), "NoAutonomousIsolationWithoutPolicyConditions");
        let mut request = valid_request();
        request.action_kind = HouseholdActionKind::LeakIsolation;
        let mut snapshot = valid_snapshot();
        snapshot.local_controller.state = LocalControllerState::Refused;
        let result = evaluate(&request, &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::LocalControllerRefused);
    }

    #[test]
    fn hra_012_coordinator_crash_after_proposal_does_not_issue_request() {
        assert_eq!(fixture_expected("HRA-012"), "NoActuationLocalSafeOperationContinues");
        let mut snapshot = valid_snapshot();
        snapshot.coordinator_state = CoordinatorState::Unavailable;
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::CoordinatorUnavailable);
        assert!(!result.local_request_eligible);
        assert!(result.local_manual_control_available);
    }

    #[test]
    fn hra_013_local_controller_refusal_is_never_retried_as_override() {
        assert_eq!(fixture_expected("HRA-013"), "RespectRefusalAndReplan");
        let mut snapshot = valid_snapshot();
        snapshot.local_controller.state = LocalControllerState::Refused;
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::LocalControllerRefused);
        assert!(!result.local_request_eligible);
        assert!(!result.actuator_command_emitted);
    }

    #[test]
    fn hra_014_accepted_command_without_boundary_delivery_has_no_service_credit() {
        assert_eq!(fixture_expected("HRA-014"), "NoServiceCompletionCredit");
        let result = classify_boundary_outcome(
            true,
            BoundaryDeliveryObservation::NoDeliveryObserved,
        );
        assert_eq!(result, BoundaryOutcome::NoServiceCredit);
    }

    #[test]
    fn command_acceptance_is_not_service_observation() {
        assert_eq!(
            classify_boundary_outcome(true, BoundaryDeliveryObservation::Unknown),
            BoundaryOutcome::OutcomeUnknown
        );
        assert_eq!(
            classify_boundary_outcome(false, BoundaryDeliveryObservation::DeliveryObserved),
            BoundaryOutcome::CommandNotAccepted
        );
    }

    #[test]
    fn hra_pos_002_requires_exact_water_use_match() {
        assert_eq!(fixture_expected("HRA-POS-002"), "EligibleForLocalControllerEvaluation");
        let mut request = valid_request();
        request.requested_water_use = Some(WaterUseClass::Drinking);
        let mut snapshot = valid_snapshot();
        snapshot.water_qualification = Some(WaterQualificationEvidence {
            status: WaterQualificationStatus::QualifiedForDeclaredUse,
            declared_use: WaterUseClass::Drinking,
        });
        let result = evaluate(&request, &snapshot);
        assert!(result.local_request_eligible);
        assert!(!result.actuator_command_emitted);
    }

    #[test]
    fn qualified_water_for_another_use_is_rejected() {
        let mut request = valid_request();
        request.requested_water_use = Some(WaterUseClass::Drinking);
        let mut snapshot = valid_snapshot();
        snapshot.water_qualification = Some(WaterQualificationEvidence {
            status: WaterQualificationStatus::QualifiedForDeclaredUse,
            declared_use: WaterUseClass::Irrigation,
        });
        let result = evaluate(&request, &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::RejectedWaterUseQualification);
    }

    #[test]
    fn hra_pos_003_requires_a_matching_authorized_scope() {
        assert_eq!(fixture_expected("HRA-POS-003"), "EligibleForLocalControllerEvaluation");
        let mut request = valid_request();
        request.action_kind = HouseholdActionKind::DeferOptionalLoad;
        let result = evaluate(&request, &valid_snapshot());
        assert!(result.local_request_eligible);
        assert_eq!(
            result.authorization_policy_revision.as_deref(),
            Some("policy-r7")
        );
    }

    #[test]
    fn rejects_authorization_scope_mismatch() {
        let mut snapshot = valid_snapshot();
        snapshot.authorization.scope_id = Some("different-scope".into());
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::AuthorizationScopeMismatch);
    }

    #[test]
    fn rejects_missing_identity_inside_granted_authorization() {
        let mut snapshot = valid_snapshot();
        snapshot.authorization.policy_revision = None;
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::AuthorizationUnknown);
    }

    #[test]
    fn rejects_changed_local_capability_revision() {
        let mut snapshot = valid_snapshot();
        snapshot.local_controller.capability_revision = "capability-r4".into();
        let result = evaluate(&valid_request(), &snapshot);
        assert_eq!(result.decision, HouseholdActionDecision::CapabilityRevisionMismatch);
    }

    #[test]
    fn no_eligible_result_emits_an_actuator_command() {
        let mut snapshot = valid_snapshot();
        snapshot.local_controller.state = LocalControllerState::Refused;
        let result = evaluate(&valid_request(), &snapshot);
        assert!(!result.actuator_command_emitted);
    }
}
