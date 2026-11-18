//! Product-level benchmark adapter for Luminous Aqua.
//!
//! This crate validates matched test context and binds an Aqua run to canonical owner
//! evidence. It deliberately does not calculate water/energy ledgers, hydraulic or
//! thermal states, water quality, metrology admission, safety, hygiene, or comfort.
//! Symtropy, Mycelix, Sol-Atlas, and independent endpoint assessors remain authoritative
//! for those results. Missing evidence is an error; it must never be replaced by zero.

use std::error::Error;
use std::fmt::{Display, Formatter};

pub mod spray_field;

pub const CONTRACT_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceOwner {
    SymtropyWaterServiceT0,
    SymtropyResourceClosure,
    MycelixMetrology,
    MycelixWaterQuality,
    IndependentTaskOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRef {
    pub owner: EvidenceOwner,
    pub record_id: String,
    pub sha256: String,
}

impl EvidenceRef {
    fn validate(
        &self,
        field: &'static str,
        expected_owner: EvidenceOwner,
    ) -> Result<(), ContractError> {
        if self.owner != expected_owner {
            return Err(ContractError::WrongEvidenceOwner { field });
        }
        if self.record_id.trim().is_empty() {
            return Err(ContractError::MissingField { field });
        }
        if !is_sha256(&self.sha256) {
            return Err(ContractError::InvalidDigest { field });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    MissingField { field: &'static str },
    InvalidDigest { field: &'static str },
    WrongEvidenceOwner { field: &'static str },
    ZeroDuration,
    SameRunCompared,
    ContextMismatch { field: &'static str },
    MissingEnergyReceipt,
}

impl Display for ContractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => write!(f, "missing required Aqua field: {field}"),
            Self::InvalidDigest { field } => write!(f, "invalid SHA-256 digest for: {field}"),
            Self::WrongEvidenceOwner { field } => {
                write!(f, "evidence owner mismatch for: {field}")
            }
            Self::ZeroDuration => write!(f, "task duration must be greater than zero"),
            Self::SameRunCompared => write!(f, "a run cannot be compared with itself"),
            Self::ContextMismatch { field } => {
                write!(f, "matched benchmark context differs at: {field}")
            }
            Self::MissingEnergyReceipt => {
                write!(
                    f,
                    "energy comparison requires a Symtropy resource-closure receipt"
                )
            }
        }
    }
}

impl Error for ContractError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), ContractError> {
    if value.trim().is_empty() {
        return Err(ContractError::MissingField { field });
    }
    Ok(())
}

fn require_digest(field: &'static str, value: &str) -> Result<(), ContractError> {
    if !is_sha256(value) {
        return Err(ContractError::InvalidDigest { field });
    }
    Ok(())
}

/// Conditions that must be identical for a matched control/candidate comparison.
///
/// Digests bind the full versioned profiles and their source data. The adapter checks
/// equality and syntax only; it does not validate the underlying physical observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkConditions {
    pub task_profile_id: String,
    pub duration_seconds: u32,
    pub inlet_water_profile_sha256: String,
    pub delivery_conditions_sha256: String,
    pub room_conditions_sha256: String,
    pub measurement_protocol_sha256: String,
    pub accounting_boundary_sha256: String,
}

impl BenchmarkConditions {
    pub fn validate(&self) -> Result<(), ContractError> {
        require_nonempty("task_profile_id", &self.task_profile_id)?;
        if self.duration_seconds == 0 {
            return Err(ContractError::ZeroDuration);
        }
        require_digest(
            "inlet_water_profile_sha256",
            &self.inlet_water_profile_sha256,
        )?;
        require_digest(
            "delivery_conditions_sha256",
            &self.delivery_conditions_sha256,
        )?;
        require_digest("room_conditions_sha256", &self.room_conditions_sha256)?;
        require_digest(
            "measurement_protocol_sha256",
            &self.measurement_protocol_sha256,
        )?;
        require_digest(
            "accounting_boundary_sha256",
            &self.accounting_boundary_sha256,
        )?;
        Ok(())
    }

    fn compare_with(&self, other: &Self) -> Result<(), ContractError> {
        if self.task_profile_id != other.task_profile_id {
            return Err(ContractError::ContextMismatch {
                field: "task_profile_id",
            });
        }
        if self.duration_seconds != other.duration_seconds {
            return Err(ContractError::ContextMismatch {
                field: "duration_seconds",
            });
        }
        if self.inlet_water_profile_sha256 != other.inlet_water_profile_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "inlet_water_profile_sha256",
            });
        }
        if self.delivery_conditions_sha256 != other.delivery_conditions_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "delivery_conditions_sha256",
            });
        }
        if self.room_conditions_sha256 != other.room_conditions_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "room_conditions_sha256",
            });
        }
        if self.measurement_protocol_sha256 != other.measurement_protocol_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "measurement_protocol_sha256",
            });
        }
        if self.accounting_boundary_sha256 != other.accounting_boundary_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "accounting_boundary_sha256",
            });
        }
        Ok(())
    }
}

/// Device-specific identity. Configuration and cleaning protocol may differ between
/// comparators; they must be recorded and bound to each run, not forced to be identical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceConfiguration {
    pub market_variant_id: String,
    pub hardware_configuration_sha256: String,
    pub installation_sha256: String,
    pub cleaning_protocol_sha256: String,
    pub firmware_revision: Option<String>,
}

impl DeviceConfiguration {
    pub fn validate(&self) -> Result<(), ContractError> {
        require_nonempty("market_variant_id", &self.market_variant_id)?;
        require_digest(
            "hardware_configuration_sha256",
            &self.hardware_configuration_sha256,
        )?;
        require_digest("installation_sha256", &self.installation_sha256)?;
        require_digest("cleaning_protocol_sha256", &self.cleaning_protocol_sha256)?;
        if let Some(revision) = &self.firmware_revision {
            require_nonempty("firmware_revision", revision)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AquaRunEvidence {
    pub water_service_t0: EvidenceRef,
    pub metrology_admission: EvidenceRef,
    pub water_quality_assessment: EvidenceRef,
    pub task_outcome_assessment: EvidenceRef,
    pub energy_resource_closure: Option<EvidenceRef>,
}

impl AquaRunEvidence {
    pub fn validate(&self, scope: ComparisonScope) -> Result<(), ContractError> {
        self.water_service_t0
            .validate("water_service_t0", EvidenceOwner::SymtropyWaterServiceT0)?;
        self.metrology_admission
            .validate("metrology_admission", EvidenceOwner::MycelixMetrology)?;
        self.water_quality_assessment.validate(
            "water_quality_assessment",
            EvidenceOwner::MycelixWaterQuality,
        )?;
        self.task_outcome_assessment.validate(
            "task_outcome_assessment",
            EvidenceOwner::IndependentTaskOutcome,
        )?;

        match (&self.energy_resource_closure, scope) {
            (Some(receipt), _) => receipt.validate(
                "energy_resource_closure",
                EvidenceOwner::SymtropyResourceClosure,
            )?,
            (None, ComparisonScope::WaterAndEnergy) => {
                return Err(ContractError::MissingEnergyReceipt);
            }
            (None, ComparisonScope::WaterServiceOnly) => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonScope {
    WaterServiceOnly,
    WaterAndEnergy,
}

/// One Aqua test run bound to canonical subsystem and independent outcome evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AquaRunRecord {
    pub run_id: String,
    pub conditions: BenchmarkConditions,
    pub device: DeviceConfiguration,
    pub evidence: AquaRunEvidence,
}

impl AquaRunRecord {
    pub fn validate(&self, scope: ComparisonScope) -> Result<(), ContractError> {
        require_nonempty("run_id", &self.run_id)?;
        self.conditions.validate()?;
        self.device.validate()?;
        self.evidence.validate(scope)
    }
}

/// Check whether two independently evidenced runs are suitable for a matched comparison.
///
/// This returns no performance score and does not certify non-inferiority. It only checks
/// that the task, duration, supply, room, measurement protocol, and accounting boundary
/// match, and that both runs carry the required owner receipts for the requested scope.
pub fn validate_matched_comparison(
    control: &AquaRunRecord,
    candidate: &AquaRunRecord,
    scope: ComparisonScope,
) -> Result<(), ContractError> {
    control.validate(scope)?;
    candidate.validate(scope)?;
    if control.run_id == candidate.run_id {
        return Err(ContractError::SameRunCompared);
    }
    control.conditions.compare_with(&candidate.conditions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn evidence(owner: EvidenceOwner, id: &str, byte: char) -> EvidenceRef {
        EvidenceRef {
            owner,
            record_id: id.to_owned(),
            sha256: digest(byte),
        }
    }

    fn run(run_id: &str, market_variant_id: &str, cleaning_digest: char) -> AquaRunRecord {
        AquaRunRecord {
            run_id: run_id.to_owned(),
            conditions: BenchmarkConditions {
                task_profile_id: "STANDARD_SHOWER".to_owned(),
                duration_seconds: 600,
                inlet_water_profile_sha256: digest('a'),
                delivery_conditions_sha256: digest('b'),
                room_conditions_sha256: digest('c'),
                measurement_protocol_sha256: digest('d'),
                accounting_boundary_sha256: digest('e'),
            },
            device: DeviceConfiguration {
                market_variant_id: market_variant_id.to_owned(),
                hardware_configuration_sha256: digest('f'),
                installation_sha256: digest('1'),
                cleaning_protocol_sha256: digest(cleaning_digest),
                firmware_revision: Some("1.0.0".to_owned()),
            },
            evidence: AquaRunEvidence {
                water_service_t0: evidence(EvidenceOwner::SymtropyWaterServiceT0, "water-run", '2'),
                metrology_admission: evidence(
                    EvidenceOwner::MycelixMetrology,
                    "metrology-run",
                    '3',
                ),
                water_quality_assessment: evidence(
                    EvidenceOwner::MycelixWaterQuality,
                    "quality-run",
                    '4',
                ),
                task_outcome_assessment: evidence(
                    EvidenceOwner::IndependentTaskOutcome,
                    "outcome-run",
                    '5',
                ),
                energy_resource_closure: Some(evidence(
                    EvidenceOwner::SymtropyResourceClosure,
                    "energy-run",
                    '6',
                )),
            },
        }
    }

    #[test]
    fn allows_matched_task_conditions_with_different_device_and_cleaning_profiles() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterAndEnergy,),
            Ok(())
        );
    }

    #[test]
    fn rejects_different_task_profiles() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.task_profile_id = "HAIR_WASH".to_owned();
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly,),
            Err(ContractError::ContextMismatch {
                field: "task_profile_id",
            })
        );
    }

    #[test]
    fn rejects_different_delivery_conditions() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.delivery_conditions_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly,),
            Err(ContractError::ContextMismatch {
                field: "delivery_conditions_sha256",
            })
        );
    }

    #[test]
    fn rejects_zero_duration() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.duration_seconds = 0;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::ZeroDuration)
        );
    }

    #[test]
    fn rejects_missing_water_service_receipt() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.water_service_t0.record_id.clear();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::MissingField {
                field: "water_service_t0",
            })
        );
    }

    #[test]
    fn rejects_wrong_evidence_owner() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.metrology_admission.owner = EvidenceOwner::MycelixWaterQuality;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::WrongEvidenceOwner {
                field: "metrology_admission",
            })
        );
    }

    #[test]
    fn rejects_invalid_sha256_reference() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.water_quality_assessment.sha256 = "not-a-digest".to_owned();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::InvalidDigest {
                field: "water_quality_assessment",
            })
        );
    }

    #[test]
    fn energy_comparison_requires_canonical_resource_receipt() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.energy_resource_closure = None;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterAndEnergy),
            Err(ContractError::MissingEnergyReceipt)
        );
    }

    #[test]
    fn water_only_comparison_does_not_require_energy_receipt() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.energy_resource_closure = None;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Ok(())
        );
    }

    #[test]
    fn rejects_comparing_a_run_to_itself() {
        let candidate = run("same-run", "aqua-test-article", '8');
        assert_eq!(
            validate_matched_comparison(&candidate, &candidate, ComparisonScope::WaterServiceOnly,),
            Err(ContractError::SameRunCompared)
        );
    }

    #[test]
    fn rejects_different_task_duration() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.duration_seconds = 599;
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "duration_seconds"
            })
        );
    }

    #[test]
    fn rejects_different_inlet_water_profile() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.inlet_water_profile_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "inlet_water_profile_sha256"
            })
        );
    }

    #[test]
    fn rejects_different_room_conditions() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.room_conditions_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "room_conditions_sha256"
            })
        );
    }

    #[test]
    fn rejects_different_measurement_protocol() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.measurement_protocol_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "measurement_protocol_sha256"
            })
        );
    }

    #[test]
    fn rejects_different_accounting_boundary() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.accounting_boundary_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "accounting_boundary_sha256"
            })
        );
    }

    #[test]
    fn rejects_missing_independent_task_outcome_evidence() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.task_outcome_assessment.record_id.clear();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::MissingField {
                field: "task_outcome_assessment"
            })
        );
    }

    #[test]
    fn rejects_wrong_energy_resource_owner() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate
            .evidence
            .energy_resource_closure
            .as_mut()
            .unwrap()
            .owner = EvidenceOwner::SymtropyWaterServiceT0;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterAndEnergy),
            Err(ContractError::WrongEvidenceOwner {
                field: "energy_resource_closure"
            })
        );
    }

    #[test]
    fn rejects_invalid_device_cleaning_protocol_digest() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.device.cleaning_protocol_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::InvalidDigest {
                field: "cleaning_protocol_sha256"
            })
        );
    }
}
