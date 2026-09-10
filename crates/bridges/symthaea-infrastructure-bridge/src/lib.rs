// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composition bridge from `symthaea-infrastructure` records into existing
//! digital-twin and formal-safety primitives.
//!
//! This crate is intentionally one-way: descriptive infrastructure data may
//! become telemetry, safety obligations, or a preflight decision. Nothing here
//! emits hardware commands or bypasses a local controller.

#![deny(unsafe_code)]

use chrono::{TimeZone, Utc};
use symthaea_digital_twin::{AssetClass, TelemetryPoint, TwinState};
use symthaea_formal_safety::{
    EvidenceKind, ObligationStatus, ProofObligation, SafetyCase, SafetyCaseTemplate,
};
use symthaea_infrastructure::{
    AssetId, Authorization, CommandProposal, Hazard, HazardSeverity, InfrastructureObservation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfrastructureBridgeError {
    MalformedRecord,
    AssetMismatch,
    InvalidExpectation,
    TimestampOutOfRange,
}

/// Result of the deterministic authorization preflight.
///
/// `PermitForLocalReview` is deliberately not named `Execute` or `Authorized`:
/// this bridge cannot grant physical actuation authority. It only confirms that
/// the descriptive records passed this layer's checks and may proceed to the
/// independent local execution boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightDecision {
    PermitForLocalReview,
    DenyMalformed,
    DenyAuthority,
    DenyHazard,
}

/// Create an existing `TwinState` for an infrastructure asset without teaching
/// the leaf infrastructure crate about Symthaea's digital-twin implementation.
pub fn twin_for_asset(
    asset: &AssetId,
    class: AssetClass,
    label: impl Into<String>,
) -> Result<TwinState, InfrastructureBridgeError> {
    if !asset.is_well_formed() {
        return Err(InfrastructureBridgeError::MalformedRecord);
    }
    Ok(TwinState::new(asset.0.clone(), class, label))
}

/// Ingest a validated infrastructure observation into an existing digital twin.
///
/// `expected_value` and `tolerance` come from the domain model, not from this
/// bridge. Observation uncertainty is normalized by the absolute tolerance for
/// the digital twin's `[0,1]` aleatoric uncertainty channel.
pub fn ingest_observation(
    twin: &mut TwinState,
    observation: &InfrastructureObservation,
    expected_value: f64,
    tolerance: f64,
) -> Result<(), InfrastructureBridgeError> {
    if !observation.is_well_formed() {
        return Err(InfrastructureBridgeError::MalformedRecord);
    }
    if twin.id != observation.asset.0 {
        return Err(InfrastructureBridgeError::AssetMismatch);
    }
    if !expected_value.is_finite() || !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(InfrastructureBridgeError::InvalidExpectation);
    }

    let timestamp_ms = i64::try_from(observation.observed_at_unix_ms)
        .map_err(|_| InfrastructureBridgeError::TimestampOutOfRange)?;
    let timestamp = Utc
        .timestamp_millis_opt(timestamp_ms)
        .single()
        .ok_or(InfrastructureBridgeError::TimestampOutOfRange)?;

    let normalized_uncertainty = (observation.uncertainty / tolerance.abs()).clamp(0.0, 1.0);
    let point = TelemetryPoint {
        channel: observation.quantity.clone(),
        value: observation.value,
        unit: observation.unit.clone(),
        timestamp,
    };
    twin.ingest_with_uncertainty(
        point,
        expected_value,
        tolerance,
        normalized_uncertainty,
    );
    Ok(())
}

/// Construct the existing formal-safety template that corresponds to a digital
/// twin asset class.
pub fn safety_case_for_asset(
    asset: &AssetId,
    class: AssetClass,
) -> Result<SafetyCase, InfrastructureBridgeError> {
    if !asset.is_well_formed() {
        return Err(InfrastructureBridgeError::MalformedRecord);
    }
    Ok(SafetyCase::from_template(
        format!("asset:{}", asset.0),
        safety_template_for_class(class),
    ))
}

/// Add an identified infrastructure hazard as an explicit safety obligation.
///
/// Evidence references are preserved, but their existence does not discharge
/// the obligation. Critical/catastrophic hazards require review immediately.
pub fn add_hazard_obligation(
    safety_case: &mut SafetyCase,
    hazard: &Hazard,
) -> Result<(), InfrastructureBridgeError> {
    if !hazard.is_well_formed() {
        return Err(InfrastructureBridgeError::MalformedRecord);
    }

    let mut obligation = ProofObligation::new(
        format!(
            "hazard {} for asset {} is mitigated within its accepted operating envelope: {}",
            hazard.hazard_id, hazard.asset.0, hazard.description
        ),
        EvidenceKind::Telemetry,
    );
    obligation.evidence_refs = hazard.evidence_refs.clone();
    obligation.status = match hazard.severity {
        HazardSeverity::Critical | HazardSeverity::Catastrophic => ObligationStatus::ReviewRequired,
        HazardSeverity::Advisory | HazardSeverity::Caution => ObligationStatus::Open,
    };
    safety_case.add_obligation(obligation);
    Ok(())
}

/// Deterministic preflight for a narrow authorization record.
///
/// Passing this function means only that this composition layer found an exact,
/// live authorization and no known critical/catastrophic hazard for the target.
/// A local controller must still independently validate and accept any command.
pub fn authorization_preflight(
    proposal: &CommandProposal,
    authorization: &Authorization,
    now_unix_ms: u64,
    hazards: &[Hazard],
) -> PreflightDecision {
    if !proposal.is_well_formed()
        || !authorization.is_well_formed()
        || hazards.iter().any(|hazard| !hazard.is_well_formed())
    {
        return PreflightDecision::DenyMalformed;
    }
    if !authorization.authorizes(proposal, now_unix_ms) {
        return PreflightDecision::DenyAuthority;
    }
    if hazards.iter().any(|hazard| {
        hazard.asset == proposal.target
            && matches!(
                hazard.severity,
                HazardSeverity::Critical | HazardSeverity::Catastrophic
            )
    }) {
        return PreflightDecision::DenyHazard;
    }

    PreflightDecision::PermitForLocalReview
}

fn safety_template_for_class(class: AssetClass) -> SafetyCaseTemplate {
    match class {
        AssetClass::CivilStructure => SafetyCaseTemplate::CivilStructure,
        AssetClass::MechanicalSystem => SafetyCaseTemplate::MechanicalSystem,
        AssetClass::ElectricalSystem => SafetyCaseTemplate::ElectricalSystem,
        AssetClass::AerospaceSystem => SafetyCaseTemplate::AerospaceSystem,
        AssetClass::ProcessSystem => SafetyCaseTemplate::ChemicalProcess,
        AssetClass::RoboticSystem => SafetyCaseTemplate::Robotics,
        AssetClass::NuclearSystem => SafetyCaseTemplate::NuclearSystem,
        AssetClass::MaterialSystem => SafetyCaseTemplate::Materials,
        AssetClass::EnvironmentalSystem => SafetyCaseTemplate::EnvironmentalSystem,
        AssetClass::SystemOfSystems => SafetyCaseTemplate::SystemOfSystems,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_infrastructure::{
        ExecutionDisposition, ExecutionReceipt, ObservationSource,
    };

    fn observation(asset: &str) -> InfrastructureObservation {
        InfrastructureObservation {
            observation_id: "obs-1".into(),
            asset: AssetId::new(asset),
            quantity: "bus-voltage".into(),
            value: 119.0,
            unit: "V".into(),
            uncertainty: 0.5,
            observed_at_unix_ms: 1_700_000_000_000,
            source: ObservationSource {
                source_id: "sensor-a".into(),
                method: "adc".into(),
            },
        }
    }

    fn proposal() -> CommandProposal {
        CommandProposal {
            proposal_id: "proposal-1".into(),
            target: AssetId::new("rover-7"),
            operation: "hold-position".into(),
            rationale: "navigation uncertainty exceeded envelope".into(),
            evidence_refs: vec!["obs-1".into()],
        }
    }

    fn authorization(proposal: &CommandProposal) -> Authorization {
        Authorization {
            authorization_id: "auth-1".into(),
            proposal_id: proposal.proposal_id.clone(),
            target: proposal.target.clone(),
            operation: proposal.operation.clone(),
            issuer: "ops-a".into(),
            valid_from_unix_ms: 100,
            valid_until_unix_ms: 200,
            policy_ref: "policy-v1".into(),
            evidence_refs: vec!["approval-1".into()],
        }
    }

    #[test]
    fn observation_asset_must_match_twin() {
        let mut twin = twin_for_asset(
            &AssetId::new("grid-a"),
            AssetClass::ElectricalSystem,
            "Grid A",
        )
        .unwrap();
        assert_eq!(
            ingest_observation(&mut twin, &observation("grid-b"), 120.0, 5.0),
            Err(InfrastructureBridgeError::AssetMismatch)
        );
    }

    #[test]
    fn observation_becomes_existing_digital_twin_telemetry() {
        let mut twin = twin_for_asset(
            &AssetId::new("grid-a"),
            AssetClass::ElectricalSystem,
            "Grid A",
        )
        .unwrap();
        ingest_observation(&mut twin, &observation("grid-a"), 120.0, 5.0).unwrap();
        assert_eq!(twin.telemetry.len(), 1);
        assert_eq!(twin.telemetry[0].channel, "bus-voltage");
        assert!(twin.aleatoric_uncertainty.is_finite());
    }

    #[test]
    fn catastrophic_target_hazard_blocks_preflight() {
        let proposal = proposal();
        let auth = authorization(&proposal);
        let hazard = Hazard {
            asset: proposal.target.clone(),
            hazard_id: "haz-1".into(),
            description: "unverified obstacle inside keep-out zone".into(),
            severity: HazardSeverity::Catastrophic,
            evidence_refs: vec!["vision-2".into()],
        };
        assert_eq!(
            authorization_preflight(&proposal, &auth, 150, &[hazard]),
            PreflightDecision::DenyHazard
        );
    }

    #[test]
    fn exact_live_authority_only_permits_local_review() {
        let proposal = proposal();
        let auth = authorization(&proposal);
        assert_eq!(
            authorization_preflight(&proposal, &auth, 150, &[]),
            PreflightDecision::PermitForLocalReview
        );
        assert_eq!(
            authorization_preflight(&proposal, &auth, 200, &[]),
            PreflightDecision::DenyAuthority
        );
    }

    #[test]
    fn severe_hazard_is_not_silently_discharged() {
        let asset = AssetId::new("habitat-a");
        let mut safety = safety_case_for_asset(&asset, AssetClass::SystemOfSystems).unwrap();
        let hazard = Hazard {
            asset,
            hazard_id: "haz-pressure".into(),
            description: "pressure integrity uncertain".into(),
            severity: HazardSeverity::Critical,
            evidence_refs: vec!["obs-pressure".into()],
        };
        add_hazard_obligation(&mut safety, &hazard).unwrap();
        let last = safety.obligations.last().unwrap();
        assert_eq!(last.status, ObligationStatus::ReviewRequired);
        assert_eq!(last.evidence_refs, vec!["obs-pressure"]);
    }

    #[test]
    fn receipt_type_remains_external_to_preflight() {
        // Compile-level guard that execution evidence remains a separate record
        // created after the independent local boundary, not by preflight.
        let receipt = ExecutionReceipt {
            receipt_id: "receipt-1".into(),
            authorization_id: "auth-1".into(),
            target: AssetId::new("rover-7"),
            operation: "hold-position".into(),
            disposition: ExecutionDisposition::Completed,
            recorded_at_unix_ms: 160,
            evidence_refs: vec![],
        };
        assert!(receipt.is_well_formed());
    }
}
