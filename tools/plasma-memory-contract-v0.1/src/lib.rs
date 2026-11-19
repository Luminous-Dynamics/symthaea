//! Versioned contracts for distinct plasma-closure memory mechanisms.
//!
//! This crate validates declared memory semantics and physical-time contracts.
//! It does not prove that a model has learned the stated physics, execute a
//! closure, validate solver trajectories, or authenticate artifact issuers.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageSemantics {
    AcceptedStateOnly,
    StageExplicit,
    SourceReported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupPolicy {
    EquilibriumPadding,
    AbstainUntilFullWindow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateInitializer {
    Zero,
    Equilibrium,
    LearnedPrior,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateResetPolicy {
    AtTrajectoryBoundary,
    AtRegimeOrSolverChange,
    EveryIndependentRun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    pub numerator: u64,
    pub denominator: u64,
}

impl Rational {
    fn is_positive(self) -> bool {
        self.numerator > 0 && self.denominator > 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryMechanism {
    /// Instantaneous closure; retained as a matched baseline.
    Markovian {
        input_fields: Vec<String>,
    },
    /// Finite history of resolved fields, measured in physical time.
    TrailingHistory {
        input_fields: Vec<String>,
        sample_count: u32,
        sample_period: Rational,
        physical_horizon: Rational,
        time_unit: String,
        warmup_policy: WarmupPolicy,
    },
    /// Compact evolving state intended to summarize unresolved kinetic history.
    PrognosticLatent {
        state_dimension: u32,
        initializer: StateInitializer,
        update_law_sha256: String,
        reset_policy: StateResetPolicy,
        uncertainty_evidence_sha256: Option<String>,
    },
    /// Reservoir closure of the unresolved velocity-space Hermite hierarchy.
    FourierHermiteReservoir {
        retained_configuration_modes: u32,
        retained_velocity_modes: u32,
        first_closed_velocity_mode: u32,
        reservoir_artifact_sha256: String,
        reset_policy: StateResetPolicy,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryContract {
    pub contract_id: String,
    pub physics_model_id: String,
    pub regime_id: String,
    pub trajectory_id: String,
    pub trajectory_sha256: String,
    pub model_artifact_sha256: String,
    pub solver_build_sha256: String,
    pub tableau_sha256: String,
    pub stage_semantics: StageSemantics,
    pub mechanism: MemoryMechanism,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    EmptyContractId,
    EmptyPhysicsModelId,
    EmptyRegimeId,
    EmptyTrajectoryId,
    InvalidTrajectoryDigest,
    InvalidModelDigest,
    InvalidSolverDigest,
    InvalidTableauDigest,
    EmptyInputFields,
    EmptyInputField,
    DuplicateInputField(String),
    ZeroSampleCount,
    InvalidSamplePeriod,
    InvalidPhysicalHorizon,
    EmptyTimeUnit,
    PhysicalHorizonMismatch,
    TimeArithmeticOverflow,
    ZeroLatentDimension,
    InvalidUpdateLawDigest,
    InvalidUncertaintyEvidenceDigest,
    ZeroConfigurationModes,
    ZeroVelocityModes,
    InvalidFirstClosedVelocityMode,
    InvalidReservoirArtifactDigest,
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_fields(fields: &[String]) -> Result<(), ContractError> {
    if fields.is_empty() {
        return Err(ContractError::EmptyInputFields);
    }
    let mut names = BTreeSet::new();
    for field in fields {
        if !non_empty(field) {
            return Err(ContractError::EmptyInputField);
        }
        if !names.insert(field.as_str()) {
            return Err(ContractError::DuplicateInputField(field.clone()));
        }
    }
    Ok(())
}

/// Exact integer arithmetic for sample_count * sample_period == physical_horizon.
///
/// The rational values must use the same canonical time unit. No floating-point
/// tolerance or implicit conversion is used.
pub fn history_horizon_matches(
    sample_count: u32,
    sample_period: Rational,
    physical_horizon: Rational,
) -> Result<bool, ContractError> {
    if sample_count == 0 {
        return Err(ContractError::ZeroSampleCount);
    }
    if !sample_period.is_positive() {
        return Err(ContractError::InvalidSamplePeriod);
    }
    if !physical_horizon.is_positive() {
        return Err(ContractError::InvalidPhysicalHorizon);
    }

    let actual_numerator = u128::from(sample_count)
        .checked_mul(u128::from(sample_period.numerator))
        .and_then(|value| value.checked_mul(u128::from(physical_horizon.denominator)))
        .ok_or(ContractError::TimeArithmeticOverflow)?;
    let expected_numerator = u128::from(physical_horizon.numerator)
        .checked_mul(u128::from(sample_period.denominator))
        .ok_or(ContractError::TimeArithmeticOverflow)?;

    Ok(actual_numerator == expected_numerator)
}

pub fn validate_contract(contract: &MemoryContract) -> Result<(), ContractError> {
    if !non_empty(&contract.contract_id) {
        return Err(ContractError::EmptyContractId);
    }
    if !non_empty(&contract.physics_model_id) {
        return Err(ContractError::EmptyPhysicsModelId);
    }
    if !non_empty(&contract.regime_id) {
        return Err(ContractError::EmptyRegimeId);
    }
    if !non_empty(&contract.trajectory_id) {
        return Err(ContractError::EmptyTrajectoryId);
    }
    if !is_sha256(&contract.trajectory_sha256) {
        return Err(ContractError::InvalidTrajectoryDigest);
    }
    if !is_sha256(&contract.model_artifact_sha256) {
        return Err(ContractError::InvalidModelDigest);
    }
    if !is_sha256(&contract.solver_build_sha256) {
        return Err(ContractError::InvalidSolverDigest);
    }
    if !is_sha256(&contract.tableau_sha256) {
        return Err(ContractError::InvalidTableauDigest);
    }

    match &contract.mechanism {
        MemoryMechanism::Markovian { input_fields } => {
            validate_fields(input_fields)?;
        }
        MemoryMechanism::TrailingHistory {
            input_fields,
            sample_count,
            sample_period,
            physical_horizon,
            time_unit,
            ..
        } => {
            validate_fields(input_fields)?;
            if !non_empty(time_unit) {
                return Err(ContractError::EmptyTimeUnit);
            }
            if !history_horizon_matches(*sample_count, *sample_period, *physical_horizon)? {
                return Err(ContractError::PhysicalHorizonMismatch);
            }
        }
        MemoryMechanism::PrognosticLatent {
            state_dimension,
            update_law_sha256,
            uncertainty_evidence_sha256,
            ..
        } => {
            if *state_dimension == 0 {
                return Err(ContractError::ZeroLatentDimension);
            }
            if !is_sha256(update_law_sha256) {
                return Err(ContractError::InvalidUpdateLawDigest);
            }
            if uncertainty_evidence_sha256
                .as_deref()
                .is_some_and(|digest| !is_sha256(digest))
            {
                return Err(ContractError::InvalidUncertaintyEvidenceDigest);
            }
        }
        MemoryMechanism::FourierHermiteReservoir {
            retained_configuration_modes,
            retained_velocity_modes,
            first_closed_velocity_mode,
            reservoir_artifact_sha256,
            ..
        } => {
            if *retained_configuration_modes == 0 {
                return Err(ContractError::ZeroConfigurationModes);
            }
            if *retained_velocity_modes == 0 {
                return Err(ContractError::ZeroVelocityModes);
            }
            if *first_closed_velocity_mode != retained_velocity_modes.saturating_add(1) {
                return Err(ContractError::InvalidFirstClosedVelocityMode);
            }
            if !is_sha256(reservoir_artifact_sha256) {
                return Err(ContractError::InvalidReservoirArtifactDigest);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn base_contract(mechanism: MemoryMechanism) -> MemoryContract {
        MemoryContract {
            contract_id: "synthetic-memory-contract-v1".into(),
            physics_model_id: "synthetic-vlasov-poisson-1d".into(),
            regime_id: "synthetic-landau-damping".into(),
            trajectory_id: "synthetic-trajectory-001".into(),
            trajectory_sha256: DIGEST_A.into(),
            model_artifact_sha256: DIGEST_B.into(),
            solver_build_sha256: DIGEST_A.into(),
            tableau_sha256: DIGEST_B.into(),
            stage_semantics: StageSemantics::AcceptedStateOnly,
            mechanism,
        }
    }

    fn history_contract(
        sample_count: u32,
        sample_period: Rational,
        physical_horizon: Rational,
    ) -> MemoryContract {
        base_contract(MemoryMechanism::TrailingHistory {
            input_fields: vec!["density".into(), "velocity".into(), "pressure".into()],
            sample_count,
            sample_period,
            physical_horizon,
            time_unit: "omega_p^-1".into(),
            warmup_policy: WarmupPolicy::EquilibriumPadding,
        })
    }

    #[test]
    fn published_fno_history_contract_is_exact_in_physical_time() {
        let contract = history_contract(
            50,
            Rational { numerator: 1, denominator: 10 },
            Rational { numerator: 5, denominator: 1 },
        );
        assert_eq!(validate_contract(&contract), Ok(()));
    }

    #[test]
    fn resampling_preserves_physical_horizon_when_sample_count_changes() {
        let original = history_horizon_matches(
            50,
            Rational { numerator: 1, denominator: 10 },
            Rational { numerator: 5, denominator: 1 },
        ).unwrap();
        let resampled = history_horizon_matches(
            100,
            Rational { numerator: 1, denominator: 20 },
            Rational { numerator: 5, denominator: 1 },
        ).unwrap();
        assert!(original);
        assert!(resampled);
    }

    #[test]
    fn keeping_sample_count_while_changing_cadence_changes_horizon() {
        let matches = history_horizon_matches(
            50,
            Rational { numerator: 1, denominator: 5 },
            Rational { numerator: 5, denominator: 1 },
        ).unwrap();
        assert!(!matches);
        let contract = history_contract(
            50,
            Rational { numerator: 1, denominator: 5 },
            Rational { numerator: 5, denominator: 1 },
        );
        assert_eq!(validate_contract(&contract), Err(ContractError::PhysicalHorizonMismatch));
    }

    #[test]
    fn zero_or_invalid_history_time_is_rejected() {
        assert_eq!(
            history_horizon_matches(0, Rational { numerator: 1, denominator: 10 }, Rational { numerator: 5, denominator: 1 }),
            Err(ContractError::ZeroSampleCount)
        );
        assert_eq!(
            history_horizon_matches(50, Rational { numerator: 1, denominator: 0 }, Rational { numerator: 5, denominator: 1 }),
            Err(ContractError::InvalidSamplePeriod)
        );
        assert_eq!(
            history_horizon_matches(50, Rational { numerator: 1, denominator: 10 }, Rational { numerator: 0, denominator: 1 }),
            Err(ContractError::InvalidPhysicalHorizon)
        );
    }

    #[test]
    fn time_arithmetic_overflow_fails_closed() {
        assert_eq!(
            history_horizon_matches(
                u32::MAX,
                Rational { numerator: u64::MAX, denominator: 1 },
                Rational { numerator: u64::MAX, denominator: 1 },
            ),
            Err(ContractError::TimeArithmeticOverflow)
        );
    }

    #[test]
    fn markovian_baseline_has_explicit_input_fields() {
        let contract = base_contract(MemoryMechanism::Markovian {
            input_fields: vec!["density".into(), "velocity".into(), "pressure".into()],
        });
        assert_eq!(validate_contract(&contract), Ok(()));
    }

    #[test]
    fn duplicate_or_empty_input_fields_are_rejected() {
        let duplicate = base_contract(MemoryMechanism::Markovian {
            input_fields: vec!["density".into(), "density".into()],
        });
        assert_eq!(
            validate_contract(&duplicate),
            Err(ContractError::DuplicateInputField("density".into()))
        );

        let empty = base_contract(MemoryMechanism::Markovian {
            input_fields: vec!["".into()],
        });
        assert_eq!(validate_contract(&empty), Err(ContractError::EmptyInputField));
    }

    #[test]
    fn latent_state_requires_dimension_and_update_law_identity() {
        let valid = base_contract(MemoryMechanism::PrognosticLatent {
            state_dimension: 1,
            initializer: StateInitializer::Equilibrium,
            update_law_sha256: DIGEST_A.into(),
            reset_policy: StateResetPolicy::AtTrajectoryBoundary,
            uncertainty_evidence_sha256: Some(DIGEST_B.into()),
        });
        assert_eq!(validate_contract(&valid), Ok(()));

        let invalid = base_contract(MemoryMechanism::PrognosticLatent {
            state_dimension: 0,
            initializer: StateInitializer::Zero,
            update_law_sha256: DIGEST_A.into(),
            reset_policy: StateResetPolicy::AtTrajectoryBoundary,
            uncertainty_evidence_sha256: None,
        });
        assert_eq!(validate_contract(&invalid), Err(ContractError::ZeroLatentDimension));
    }

    #[test]
    fn invalid_latent_uncertainty_digest_is_rejected() {
        let contract = base_contract(MemoryMechanism::PrognosticLatent {
            state_dimension: 4,
            initializer: StateInitializer::LearnedPrior,
            update_law_sha256: DIGEST_A.into(),
            reset_policy: StateResetPolicy::AtRegimeOrSolverChange,
            uncertainty_evidence_sha256: Some("not-a-digest".into()),
        });
        assert_eq!(
            validate_contract(&contract),
            Err(ContractError::InvalidUncertaintyEvidenceDigest)
        );
    }

    #[test]
    fn fourier_hermite_reservoir_requires_contiguous_resolved_boundary() {
        let valid = base_contract(MemoryMechanism::FourierHermiteReservoir {
            retained_configuration_modes: 32,
            retained_velocity_modes: 24,
            first_closed_velocity_mode: 25,
            reservoir_artifact_sha256: DIGEST_B.into(),
            reset_policy: StateResetPolicy::AtTrajectoryBoundary,
        });
        assert_eq!(validate_contract(&valid), Ok(()));

        let invalid = base_contract(MemoryMechanism::FourierHermiteReservoir {
            retained_configuration_modes: 32,
            retained_velocity_modes: 24,
            first_closed_velocity_mode: 26,
            reservoir_artifact_sha256: DIGEST_B.into(),
            reset_policy: StateResetPolicy::AtTrajectoryBoundary,
        });
        assert_eq!(
            validate_contract(&invalid),
            Err(ContractError::InvalidFirstClosedVelocityMode)
        );
    }

    #[test]
    fn malformed_contract_and_artifact_identities_are_rejected() {
        let mut contract = history_contract(
            50,
            Rational { numerator: 1, denominator: 10 },
            Rational { numerator: 5, denominator: 1 },
        );
        contract.solver_build_sha256 = "unknown".into();
        assert_eq!(validate_contract(&contract), Err(ContractError::InvalidSolverDigest));
    }

    #[test]
    fn distinct_mechanism_variants_remain_distinct() {
        let markovian = base_contract(MemoryMechanism::Markovian {
            input_fields: vec!["density".into()],
        });
        let latent = base_contract(MemoryMechanism::PrognosticLatent {
            state_dimension: 1,
            initializer: StateInitializer::Equilibrium,
            update_law_sha256: DIGEST_A.into(),
            reset_policy: StateResetPolicy::AtTrajectoryBoundary,
            uncertainty_evidence_sha256: None,
        });
        let reservoir = base_contract(MemoryMechanism::FourierHermiteReservoir {
            retained_configuration_modes: 32,
            retained_velocity_modes: 24,
            first_closed_velocity_mode: 25,
            reservoir_artifact_sha256: DIGEST_B.into(),
            reset_policy: StateResetPolicy::AtTrajectoryBoundary,
        });
        assert_ne!(markovian.mechanism, latent.mechanism);
        assert_ne!(latent.mechanism, reservoir.mechanism);
    }
}
