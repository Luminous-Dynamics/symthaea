//! Versioned, evidence-carrying AtomFab message types.
//!
//! This module does not implement a solver or instrument driver. Deserialize a
//! record and call ContractEnvelope::validate before using it for decisions.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CONTRACT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Proposed,
    Predicted,
    Simulated,
    Measured,
    IndependentlyReproduced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockDomain {
    Simulation,
    MonotonicProcess,
    WallClock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintOperator {
    AtLeast,
    AtMost,
    EqualWithin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvergenceStatus {
    Converged,
    NotConverged,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TwinStatus {
    Feasible,
    Infeasible,
    Unknown,
    OutOfDomain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionOutcome {
    Accept,
    Reject,
    Abstain,
    Reevaluate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    pub component: String,
    pub version: String,
    pub source_revision: String,
    pub config_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceReference {
    pub evidence_id: String,
    pub sha256: String,
    pub role: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Uncertainty {
    pub standard_uncertainty: f64,
    pub unit: String,
    pub method: String,
    pub calibrated: bool,
    pub valid_domain: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantityEstimate {
    pub value: f64,
    pub unit: String,
    pub uncertainty: Option<Uncertainty>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamedEstimate {
    pub name: String,
    pub estimate: QuantityEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropertyConstraint {
    pub name: String,
    pub operator: ConstraintOperator,
    pub target: QuantityEstimate,
    pub tolerance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessAction {
    pub action_type: String,
    pub parameters: BTreeMap<String, QuantityEstimate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record_type", rename_all = "snake_case", deny_unknown_fields)]
pub enum AtomFabPayload {
    ManufacturingIntent {
        objective: String,
        material_family: String,
        target_properties: Vec<PropertyConstraint>,
        hard_constraints: Vec<String>,
    },
    AtomisticEvaluation {
        structure_fingerprint: String,
        composition: BTreeMap<String, u64>,
        solver_id: String,
        convergence: ConvergenceStatus,
        energy_per_atom: Option<QuantityEstimate>,
        observables: Vec<NamedEstimate>,
    },
    ProcessProposal {
        equipment_class: String,
        actions: Vec<ProcessAction>,
        preconditions: Vec<String>,
        abort_guards: Vec<String>,
    },
    TwinEvaluation {
        status: TwinStatus,
        hard_constraints_satisfied: bool,
        violations: Vec<String>,
        metrics: Vec<NamedEstimate>,
        replay_digest: String,
    },
    MeasurementRecord {
        instrument_id: String,
        calibration_digest: String,
        sample_id: String,
        raw_data_digest: String,
        readings: Vec<NamedEstimate>,
        operator_approval_id: Option<String>,
    },
    DecisionRecord {
        outcome: DecisionOutcome,
        reasons: Vec<String>,
        next_action: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractEnvelope {
    pub schema_version: u16,
    pub run_id: String,
    pub scenario_id: String,
    pub sequence: u64,
    /// Record creation timestamp in RFC 3339 wall-clock format.
    pub created_at: String,
    pub clock_domain: ClockDomain,
    pub source_kind: SourceKind,
    pub producer: Producer,
    pub parent_digest: Option<String>,
    pub evidence_refs: Vec<EvidenceReference>,
    pub limitations: Vec<String>,
    pub payload: AtomFabPayload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractError {
    UnsupportedSchemaVersion(u16),
    EmptyField(&'static str),
    InvalidSha256(&'static str),
    InvalidTimestamp,
    InvalidEstimate(&'static str),
    InvalidUncertainty(&'static str),
    InvalidPayload(&'static str),
    InvalidSourceKind,
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(v) => {
                write!(f, "unsupported AtomFab schema version: {v}")
            }
            Self::EmptyField(field) => write!(f, "required AtomFab field is empty: {field}"),
            Self::InvalidSha256(field) => {
                write!(f, "field is not a lowercase SHA-256 digest: {field}")
            }
            Self::InvalidTimestamp => write!(
                f,
                "created_at must be an RFC 3339 timestamp with a timezone"
            ),
            Self::InvalidEstimate(field) => write!(f, "invalid quantity estimate: {field}"),
            Self::InvalidUncertainty(field) => write!(f, "invalid uncertainty: {field}"),
            Self::InvalidPayload(field) => write!(f, "invalid payload: {field}"),
            Self::InvalidSourceKind => write!(f, "source_kind does not match the record type"),
        }
    }
}

impl std::error::Error for ContractError {}

impl ContractEnvelope {
    /// Validate shared provenance fields and type-specific invariants.
    ///
    /// Valid JSON is not necessarily a semantically valid manufacturing record.
    /// Sequence monotonicity is a stream-level invariant checked by the consumer.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != CONTRACT_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchemaVersion(self.schema_version));
        }
        require_non_empty(&self.run_id, "run_id")?;
        require_non_empty(&self.scenario_id, "scenario_id")?;
        if !looks_like_rfc3339(&self.created_at) {
            return Err(ContractError::InvalidTimestamp);
        }

        require_non_empty(&self.producer.component, "producer.component")?;
        require_non_empty(&self.producer.version, "producer.version")?;
        require_non_empty(&self.producer.source_revision, "producer.source_revision")?;
        validate_sha256(&self.producer.config_digest, "producer.config_digest")?;

        if let Some(digest) = self.parent_digest.as_deref() {
            validate_sha256(digest, "parent_digest")?;
        }
        for reference in &self.evidence_refs {
            require_non_empty(&reference.evidence_id, "evidence_refs.evidence_id")?;
            require_non_empty(&reference.role, "evidence_refs.role")?;
            validate_sha256(&reference.sha256, "evidence_refs.sha256")?;
        }

        self.validate_payload()
    }

    fn validate_payload(&self) -> Result<(), ContractError> {
        match &self.payload {
            AtomFabPayload::ManufacturingIntent {
                objective,
                material_family,
                target_properties,
                hard_constraints,
            } => {
                if self.source_kind != SourceKind::Proposed {
                    return Err(ContractError::InvalidSourceKind);
                }
                require_non_empty(objective, "objective")?;
                require_non_empty(material_family, "material_family")?;
                if target_properties.is_empty() && hard_constraints.is_empty() {
                    return Err(ContractError::InvalidPayload(
                        "intent has no targets or constraints",
                    ));
                }
                for constraint in target_properties {
                    validate_constraint(constraint)?;
                }
                for constraint in hard_constraints {
                    require_non_empty(constraint, "hard_constraint")?;
                }
            }
            AtomFabPayload::AtomisticEvaluation {
                structure_fingerprint,
                composition,
                solver_id,
                energy_per_atom,
                observables,
                ..
            } => {
                if !matches!(
                    self.source_kind,
                    SourceKind::Predicted | SourceKind::Simulated
                ) {
                    return Err(ContractError::InvalidSourceKind);
                }
                require_non_empty(structure_fingerprint, "structure_fingerprint")?;
                require_non_empty(solver_id, "solver_id")?;
                if composition.is_empty() || composition.values().any(|count| *count == 0) {
                    return Err(ContractError::InvalidPayload(
                        "composition must contain positive element counts",
                    ));
                }
                for element in composition.keys() {
                    require_non_empty(element, "composition.element")?;
                }
                if energy_per_atom.is_none() && observables.is_empty() {
                    return Err(ContractError::InvalidPayload(
                        "evaluation has no energy or observables",
                    ));
                }
                if let Some(energy) = energy_per_atom {
                    validate_estimate(energy, "energy_per_atom")?;
                }
                for observable in observables {
                    validate_named_estimate(observable)?;
                }
            }
            AtomFabPayload::ProcessProposal {
                equipment_class,
                actions,
                preconditions,
                abort_guards,
            } => {
                if self.source_kind != SourceKind::Proposed {
                    return Err(ContractError::InvalidSourceKind);
                }
                require_non_empty(equipment_class, "equipment_class")?;
                if actions.is_empty() {
                    return Err(ContractError::InvalidPayload(
                        "process proposal has no actions",
                    ));
                }
                if abort_guards.is_empty() {
                    return Err(ContractError::InvalidPayload(
                        "process proposal has no abort guards",
                    ));
                }
                for value in preconditions {
                    require_non_empty(value, "precondition")?;
                }
                for value in abort_guards {
                    require_non_empty(value, "abort_guard")?;
                }
                for action in actions {
                    require_non_empty(&action.action_type, "action_type")?;
                    for (name, estimate) in &action.parameters {
                        require_non_empty(name, "action.parameter_name")?;
                        validate_estimate(estimate, "action.parameter")?;
                    }
                }
            }
            AtomFabPayload::TwinEvaluation {
                status,
                hard_constraints_satisfied,
                violations,
                metrics,
                replay_digest,
            } => {
                if self.source_kind != SourceKind::Simulated {
                    return Err(ContractError::InvalidSourceKind);
                }
                validate_sha256(replay_digest, "replay_digest")?;
                for metric in metrics {
                    validate_named_estimate(metric)?;
                }
                match status {
                    TwinStatus::Feasible
                        if !hard_constraints_satisfied || !violations.is_empty() =>
                    {
                        return Err(ContractError::InvalidPayload(
                            "feasible twin result has failed hard constraints",
                        ));
                    }
                    TwinStatus::Infeasible
                        if *hard_constraints_satisfied || violations.is_empty() =>
                    {
                        return Err(ContractError::InvalidPayload(
                            "infeasible twin result lacks a violation",
                        ));
                    }
                    TwinStatus::Unknown | TwinStatus::OutOfDomain
                        if *hard_constraints_satisfied || self.limitations.is_empty() =>
                    {
                        return Err(ContractError::InvalidPayload(
                            "unknown/out-of-domain twin result needs limitations and cannot satisfy hard constraints",
                        ));
                    }
                    _ => {}
                }
                for value in violations {
                    require_non_empty(value, "violations")?;
                }
            }
            AtomFabPayload::MeasurementRecord {
                instrument_id,
                calibration_digest,
                sample_id,
                raw_data_digest,
                readings,
                ..
            } => {
                if self.source_kind != SourceKind::Measured {
                    return Err(ContractError::InvalidSourceKind);
                }
                require_non_empty(instrument_id, "instrument_id")?;
                require_non_empty(sample_id, "sample_id")?;
                validate_sha256(calibration_digest, "calibration_digest")?;
                validate_sha256(raw_data_digest, "raw_data_digest")?;
                if readings.is_empty() {
                    return Err(ContractError::InvalidPayload("measurement has no readings"));
                }
                for reading in readings {
                    validate_named_estimate(reading)?;
                }
            }
            AtomFabPayload::DecisionRecord {
                outcome, reasons, ..
            } => {
                if reasons.is_empty() {
                    return Err(ContractError::InvalidPayload(
                        "decision must state at least one reason",
                    ));
                }
                for reason in reasons {
                    require_non_empty(reason, "decision.reason")?;
                }
                if *outcome == DecisionOutcome::Accept
                    && (self.parent_digest.is_none() || self.evidence_refs.is_empty())
                {
                    return Err(ContractError::InvalidPayload(
                        "accept decision requires a parent digest and evidence references",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn require_non_empty(value: &str, field: &'static str) -> Result<(), ContractError> {
    if value.trim().is_empty() {
        return Err(ContractError::EmptyField(field));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), ContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ContractError::InvalidSha256(field));
    }
    Ok(())
}

fn looks_like_rfc3339(value: &str) -> bool {
    chrono::DateTime::parse_from_rfc3339(value).is_ok()
}

fn validate_estimate(
    estimate: &QuantityEstimate,
    field: &'static str,
) -> Result<(), ContractError> {
    if !estimate.value.is_finite() || estimate.unit.trim().is_empty() {
        return Err(ContractError::InvalidEstimate(field));
    }
    if let Some(uncertainty) = &estimate.uncertainty {
        if !uncertainty.standard_uncertainty.is_finite()
            || uncertainty.standard_uncertainty < 0.0
            || uncertainty.unit.trim().is_empty()
            || uncertainty.unit != estimate.unit
        {
            return Err(ContractError::InvalidUncertainty(field));
        }
        require_non_empty(&uncertainty.method, "uncertainty.method")?;
        require_non_empty(&uncertainty.valid_domain, "uncertainty.valid_domain")?;
    }
    Ok(())
}

fn validate_named_estimate(value: &NamedEstimate) -> Result<(), ContractError> {
    require_non_empty(&value.name, "estimate.name")?;
    validate_estimate(&value.estimate, "estimate.value")
}

fn validate_constraint(value: &PropertyConstraint) -> Result<(), ContractError> {
    require_non_empty(&value.name, "constraint.name")?;
    validate_estimate(&value.target, "constraint.target")?;
    match (value.operator, value.tolerance) {
        (ConstraintOperator::EqualWithin, Some(tolerance))
            if tolerance.is_finite() && tolerance >= 0.0 => {}
        (ConstraintOperator::EqualWithin, _) => {
            return Err(ContractError::InvalidPayload(
                "EqualWithin constraint needs a finite non-negative tolerance",
            ));
        }
        (_, Some(tolerance)) if !tolerance.is_finite() || tolerance < 0.0 => {
            return Err(ContractError::InvalidPayload(
                "constraint tolerance must be finite and non-negative",
            ));
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_envelope(payload: AtomFabPayload, source_kind: SourceKind) -> ContractEnvelope {
        ContractEnvelope {
            schema_version: CONTRACT_SCHEMA_VERSION,
            run_id: "run-0001".to_owned(),
            scenario_id: "workcell-smoke-01".to_owned(),
            sequence: 0,
            created_at: "2026-10-19T15:00:00Z".to_owned(),
            clock_domain: ClockDomain::WallClock,
            source_kind,
            producer: Producer {
                component: "symthaea-atomfab".to_owned(),
                version: "0.1.0".to_owned(),
                source_revision: "0123456789abcdef0123456789abcdef01234567".to_owned(),
                config_digest: "a".repeat(64),
            },
            parent_digest: None,
            evidence_refs: Vec::new(),
            limitations: Vec::new(),
            payload,
        }
    }

    fn valid_process_proposal() -> ContractEnvelope {
        valid_envelope(
            AtomFabPayload::ProcessProposal {
                equipment_class: "surrogate_workcell".to_owned(),
                actions: vec![ProcessAction {
                    action_type: "move_tool_to_pose".to_owned(),
                    parameters: BTreeMap::from([(
                        "travel_m".to_owned(),
                        QuantityEstimate {
                            value: 0.01,
                            unit: "m".to_owned(),
                            uncertainty: None,
                        },
                    )]),
                }],
                preconditions: vec!["fixture_clamped".to_owned()],
                abort_guards: vec!["travel_limit_exceeded".to_owned()],
            },
            SourceKind::Proposed,
        )
    }

    #[test]
    fn shared_process_proposal_fixture_validates() {
        let envelope: ContractEnvelope = serde_json::from_str(include_str!(
            "../fixtures/contract-v1-process-proposal.json"
        ))
        .expect("shared contract fixture should deserialize");
        envelope
            .validate()
            .expect("shared contract fixture should validate");
    }

    #[test]
    fn negative_fixture_deserializes_but_fails_semantic_validation() {
        let envelope: ContractEnvelope = serde_json::from_str(include_str!(
            "../fixtures/invalid/contract-v1-no-abort-guard.json"
        ))
        .expect("negative fixture is structurally valid JSON and Rust data");
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidPayload(
                "process proposal has no abort guards"
            ))
        );
    }

    #[test]
    fn valid_process_contract_round_trips_json() {
        let envelope = valid_process_proposal();
        envelope.validate().expect("valid contract should pass");
        let json = serde_json::to_string(&envelope).expect("serialization should succeed");
        let decoded: ContractEnvelope =
            serde_json::from_str(&json).expect("known record type should deserialize");
        decoded
            .validate()
            .expect("round-tripped contract should pass");
        assert_eq!(decoded, envelope);
    }

    #[test]
    fn rejects_unsupported_schema_and_bad_configuration_digest() {
        let mut envelope = valid_process_proposal();
        envelope.schema_version = 2;
        assert_eq!(
            envelope.validate(),
            Err(ContractError::UnsupportedSchemaVersion(2))
        );

        let mut envelope = valid_process_proposal();
        envelope.producer.config_digest = "not-a-digest".to_owned();
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidSha256("producer.config_digest"))
        );
    }

    #[test]
    fn rejects_unknown_json_fields_and_record_types() {
        let json = serde_json::to_string(&valid_process_proposal()).expect("serialization");
        let unknown_field = json.replace(
            "\"run_id\":",
            "\"unrecognized_critical_field\":true,\"run_id\":",
        );
        assert!(serde_json::from_str::<ContractEnvelope>(&unknown_field).is_err());

        let unknown_record = json.replace("process_proposal", "atomic_print_anything");
        assert!(serde_json::from_str::<ContractEnvelope>(&unknown_record).is_err());
    }

    #[test]
    fn rejects_proposal_without_abort_guards() {
        let mut envelope = valid_process_proposal();
        if let AtomFabPayload::ProcessProposal { abort_guards, .. } = &mut envelope.payload {
            abort_guards.clear();
        }
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidPayload(
                "process proposal has no abort guards"
            ))
        );
    }

    #[test]
    fn measurements_require_measured_provenance_and_calibration_digests() {
        let mut envelope = valid_envelope(
            AtomFabPayload::MeasurementRecord {
                instrument_id: "instrument-01".to_owned(),
                calibration_digest: "b".repeat(64),
                sample_id: "sample-01".to_owned(),
                raw_data_digest: "c".repeat(64),
                readings: vec![NamedEstimate {
                    name: "thickness".to_owned(),
                    estimate: QuantityEstimate {
                        value: 1.0e-9,
                        unit: "m".to_owned(),
                        uncertainty: Some(Uncertainty {
                            standard_uncertainty: 1.0e-11,
                            unit: "m".to_owned(),
                            method: "repeatability".to_owned(),
                            calibrated: true,
                            valid_domain: "validated instrument range".to_owned(),
                        }),
                    },
                }],
                operator_approval_id: Some("approval-001".to_owned()),
            },
            SourceKind::Simulated,
        );
        assert_eq!(envelope.validate(), Err(ContractError::InvalidSourceKind));
        envelope.source_kind = SourceKind::Measured;
        envelope
            .validate()
            .expect("measured record with calibration should pass");
        if let AtomFabPayload::MeasurementRecord {
            calibration_digest, ..
        } = &mut envelope.payload
        {
            *calibration_digest = "missing-calibration".to_owned();
        }
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidSha256("calibration_digest"))
        );
    }

    #[test]
    fn rejects_non_finite_values_and_uncertainty_unit_mismatch() {
        let mut envelope = valid_process_proposal();
        if let AtomFabPayload::ProcessProposal { actions, .. } = &mut envelope.payload {
            actions[0]
                .parameters
                .get_mut("travel_m")
                .expect("parameter")
                .value = f64::NAN;
        }
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidEstimate("action.parameter"))
        );

        let mut envelope = valid_process_proposal();
        if let AtomFabPayload::ProcessProposal { actions, .. } = &mut envelope.payload {
            actions[0]
                .parameters
                .get_mut("travel_m")
                .expect("parameter")
                .uncertainty = Some(Uncertainty {
                standard_uncertainty: 0.001,
                unit: "mm".to_owned(),
                method: "repeatability".to_owned(),
                calibrated: true,
                valid_domain: "fixture domain".to_owned(),
            });
        }
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidUncertainty("action.parameter"))
        );
    }

    #[test]
    fn accepted_decision_requires_parent_and_evidence() {
        let mut envelope = valid_envelope(
            AtomFabPayload::DecisionRecord {
                outcome: DecisionOutcome::Accept,
                reasons: vec!["acceptance criteria met".to_owned()],
                next_action: None,
            },
            SourceKind::IndependentlyReproduced,
        );
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidPayload(
                "accept decision requires a parent digest and evidence references"
            ))
        );
        envelope.parent_digest = Some("d".repeat(64));
        envelope.evidence_refs.push(EvidenceReference {
            evidence_id: "reference-001".to_owned(),
            sha256: "e".repeat(64),
            role: "accepted_reference".to_owned(),
        });
        envelope
            .validate()
            .expect("traceable accepted decision should pass");
    }

    #[test]
    fn twin_feasibility_requires_consistent_hard_constraint_outcome() {
        let mut envelope = valid_envelope(
            AtomFabPayload::TwinEvaluation {
                status: TwinStatus::Feasible,
                hard_constraints_satisfied: false,
                violations: vec!["workspace_limit".to_owned()],
                metrics: Vec::new(),
                replay_digest: "f".repeat(64),
            },
            SourceKind::Simulated,
        );
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidPayload(
                "feasible twin result has failed hard constraints"
            ))
        );
        if let AtomFabPayload::TwinEvaluation {
            status,
            hard_constraints_satisfied,
            violations,
            ..
        } = &mut envelope.payload
        {
            *status = TwinStatus::Infeasible;
            *hard_constraints_satisfied = false;
            violations.push("travel_limit".to_owned());
        }
        envelope
            .validate()
            .expect("explicitly infeasible twin result should pass");
    }
}
