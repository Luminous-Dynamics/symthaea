use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    apply_intervention, assess_homeostasis, DrivePhase, ExperimentPreregistration,
    HomeostaticReport, InteroceptiveDrive, InteroceptiveDynamicsConfig,
    InteroceptiveStepReport, InterventionRecord, NativeInteroceptiveModel,
    NativeInteroceptiveState, ScheduledIntervention, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
    INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
};

pub const EXECUTION_TRACE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionLimits {
    pub max_steps_per_arm: u64,
    pub max_total_steps: u64,
}

impl ExecutionLimits {
    pub fn try_validate(self) -> Result<(), String> {
        if self.max_steps_per_arm == 0 {
            return Err("max_steps_per_arm must be positive".into());
        }
        if self.max_total_steps == 0 {
            return Err("max_total_steps must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionStepTrace {
    pub step_index: u64,
    pub phase_index: u64,
    pub drive: InteroceptiveDrive,
    pub intervention_records: Vec<InterventionRecord>,
    pub transition: InteroceptiveStepReport,
    pub state: NativeInteroceptiveState,
    pub homeostasis: HomeostaticReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmExecutionTrace {
    /// Opaque protocol code; semantic arm identity is intentionally omitted.
    pub blind_code: String,
    pub dynamics_config: InteroceptiveDynamicsConfig,
    pub initial_state: NativeInteroceptiveState,
    pub steps: Vec<ExecutionStepTrace>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub snapshot_schema_version: u16,
    pub protocol_id: String,
    pub analysis_version: String,
    pub protocol_sha256: String,
    pub resolved_config_sha256: String,
    pub input_sequence_sha256: String,
    pub arms: Vec<ArmExecutionTrace>,
}

#[derive(Serialize)]
struct ResolvedConfigIdentity<'a> {
    model_semantics_version: u16,
    arms: Vec<ArmConfigIdentity<'a>>,
}

#[derive(Serialize)]
struct ArmConfigIdentity<'a> {
    blind_code: &'a str,
    initial_state: &'a NativeInteroceptiveState,
    dynamics_config: InteroceptiveDynamicsConfig,
}

#[derive(Serialize)]
struct InputSequenceIdentity<'a> {
    model_semantics_version: u16,
    arms: Vec<ArmInputIdentity<'a>>,
}

#[derive(Serialize)]
struct ArmInputIdentity<'a> {
    blind_code: &'a str,
    phases: &'a [DrivePhase],
    interventions: &'a [ScheduledIntervention],
}

pub fn execute_preregistration(
    protocol: &ExperimentPreregistration,
    limits: ExecutionLimits,
) -> Result<ExecutionTrace, Vec<String>> {
    protocol.validate()?;
    limits
        .try_validate()
        .map_err(|error| vec![format!("invalid execution limits: {error}")])?;

    let mut total_steps = 0_u64;
    for arm in &protocol.arms {
        let arm_steps = arm
            .total_steps()
            .ok_or_else(|| vec![format!("arm {} total step count overflows u64", arm.arm_id)])?;
        if arm_steps > limits.max_steps_per_arm {
            return Err(vec![format!(
                "arm {} requests {} steps, exceeding max_steps_per_arm {}",
                arm.arm_id, arm_steps, limits.max_steps_per_arm
            )]);
        }
        total_steps = total_steps
            .checked_add(arm_steps)
            .ok_or_else(|| vec!["total execution step count overflows u64".into()])?;
    }
    if total_steps > limits.max_total_steps {
        return Err(vec![format!(
            "protocol requests {total_steps} total steps, exceeding max_total_steps {}",
            limits.max_total_steps
        )]);
    }

    let protocol_sha256 = protocol.sha256()?;
    let resolved_config_sha256 = resolved_config_sha256(protocol)
        .map_err(|error| vec![format!("failed to hash resolved config: {error}")])?;
    let input_sequence_sha256 = input_sequence_sha256(protocol)
        .map_err(|error| vec![format!("failed to hash input sequence: {error}")])?;

    let mut arm_traces = Vec::with_capacity(protocol.arms.len());
    for arm in &protocol.arms {
        let mut model = NativeInteroceptiveModel::new(arm.initial_state.clone(), arm.dynamics_config);
        let mut intervention_index = 0_usize;
        let mut step_index = 0_u64;
        let mut steps = Vec::new();

        for (phase_index, phase) in arm.phases.iter().enumerate() {
            for _ in 0..phase.steps {
                let mut intervention_records = Vec::new();
                while intervention_index < arm.interventions.len()
                    && arm.interventions[intervention_index].before_step == step_index
                {
                    let record = apply_intervention(
                        &mut model,
                        arm.interventions[intervention_index].intervention,
                    );
                    intervention_records.push(record);
                    intervention_index += 1;
                }

                let transition = model.step(phase.drive);
                let state = model.state().clone();
                let homeostasis = assess_homeostasis(&state);
                steps.push(ExecutionStepTrace {
                    step_index,
                    phase_index: phase_index as u64,
                    drive: phase.drive,
                    intervention_records,
                    transition,
                    state,
                    homeostasis,
                });
                step_index = step_index.saturating_add(1);
            }
        }

        if intervention_index != arm.interventions.len() {
            return Err(vec![format!(
                "arm {} finished with unexecuted scheduled interventions",
                arm.arm_id
            )]);
        }

        arm_traces.push(ArmExecutionTrace {
            blind_code: arm.blind_code.clone(),
            dynamics_config: arm.dynamics_config,
            initial_state: arm.initial_state.clone(),
            steps,
        });
    }

    Ok(ExecutionTrace {
        schema_version: EXECUTION_TRACE_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        snapshot_schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
        protocol_id: protocol.protocol_id.clone(),
        analysis_version: protocol.analysis_version.clone(),
        protocol_sha256,
        resolved_config_sha256,
        input_sequence_sha256,
        arms: arm_traces,
    })
}

impl ExecutionTrace {
    pub fn validation_errors_against(
        &self,
        protocol: &ExperimentPreregistration,
        limits: ExecutionLimits,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema_version != EXECUTION_TRACE_SCHEMA_VERSION {
            errors.push(format!(
                "execution trace schema version mismatch: {}",
                self.schema_version
            ));
        }
        if self.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
            errors.push(format!(
                "execution trace model semantics version mismatch: {}",
                self.model_semantics_version
            ));
        }
        if self.snapshot_schema_version != INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION {
            errors.push(format!(
                "execution trace snapshot schema version mismatch: {}",
                self.snapshot_schema_version
            ));
        }

        match execute_preregistration(protocol, limits) {
            Ok(expected) if &expected == self => {}
            Ok(_) => errors.push(
                "execution trace does not exactly replay from the locked preregistration".into(),
            ),
            Err(replay_errors) => {
                errors.extend(replay_errors.into_iter().map(|error| format!("replay: {error}")));
            }
        }

        errors
    }

    pub fn validate_against(
        &self,
        protocol: &ExperimentPreregistration,
        limits: ExecutionLimits,
    ) -> Result<(), Vec<String>> {
        let errors = self.validation_errors_against(protocol, limits);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

fn resolved_config_sha256(protocol: &ExperimentPreregistration) -> Result<String, String> {
    let identity = ResolvedConfigIdentity {
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        arms: protocol
            .arms
            .iter()
            .map(|arm| ArmConfigIdentity {
                blind_code: arm.blind_code.as_str(),
                initial_state: &arm.initial_state,
                dynamics_config: arm.dynamics_config,
            })
            .collect(),
    };
    hash_json(&identity)
}

fn input_sequence_sha256(protocol: &ExperimentPreregistration) -> Result<String, String> {
    let identity = InputSequenceIdentity {
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        arms: protocol
            .arms
            .iter()
            .map(|arm| ArmInputIdentity {
                blind_code: arm.blind_code.as_str(),
                phases: &arm.phases,
                interventions: &arm.interventions,
            })
            .collect(),
    };
    hash_json(&identity)
}

fn hash_json<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let digest = Sha256::digest(&bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(encoded)
}
