//! DMC-specific complete branch-point restoration and factual-prefix admission.
//!
//! This validates supplied restore/prefix evidence for the exact dm_control
//! and MuJoCo versions pinned by the DMC individual-effect pilot. It does not
//! execute MuJoCo, hash external buffers, authenticate a replay receipt, or
//! replace #13377's generic causal trajectory-lineage validator.
//!
//! The important distinction is that dm_control Physics.get_state() with no
//! signature is a legacy compact qpos/qvel/act[/plugin_state] view. It is not a
//! complete branch-point snapshot. This module only admits an explicit
//! mjSTATE_INTEGRATION snapshot or a non-aliasing deep Physics copy, and it
//! separately requires the Python task, wrapper, driver and applicable RNG
//! state that live outside the Physics object.

use std::collections::BTreeSet;
use std::fmt;

pub const PINNED_DM_CONTROL_COMMIT: &str =
    "d7e52e3602a555df6cc20cc928ea5b41e9920158";
pub const PINNED_DM_CONTROL_VERSION: &str = "1.0.52";
pub const PINNED_MUJOCO_VERSION: &str = "4.0.1";
/// Errors are reported in integer nanounits to avoid floating tolerance drift.
/// A value of 1 is the paper's 1e-9 prefix-match threshold in source units.
pub const PREFIX_MATCH_MAX_ERROR_NANO: u64 = 1;
/// MuJoCo 4.0.1 mjSTATE_INTEGRATION = FULLPHYSICS | USER | WARMSTART
/// = (1 << 0) | ... | (1 << 14).
pub const PINNED_MJ_STATE_INTEGRATION_MASK: u32 = 0x7fff;
pub const MAX_DMC_BRANCHES: usize = 16;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PhysicsSnapshotKind {
    /// Legacy Physics.get_state() without a MuJoCo state signature.
    /// This is intentionally rejected as an incomplete snapshot.
    LegacyCompactGetState {
        state_sha256: [u8; 32],
    },
    /// Physics.get_state(sig=mujoco.mjtState.mjSTATE_INTEGRATION.value).
    IntegrationState {
        signature_name: String,
        signature_value: u32,
        state_size: u64,
        state_sha256: [u8; 32],
    },
    /// Physics.copy(share_model=False), which deep-copies both MjModel and MjData.
    DeepPhysicsCopy {
        share_model: bool,
        copied_model_sha256: [u8; 32],
        copied_data_sha256: [u8; 32],
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmcSnapshotManifest {
    pub dm_control_commit: String,
    pub dm_control_version: String,
    pub mujoco_version: String,
    pub runtime_platform: String,
    pub runtime_architecture: String,
    pub model_mjcf_sha256: [u8; 32],
    pub physics_snapshot: PhysicsSnapshotKind,

    /// Python-side task and driver state are separate from MjData.
    pub task_class: String,
    pub task_configuration_sha256: [u8; 32],
    pub task_private_state_sha256: [u8; 32],
    pub task_random_state_sha256: [u8; 32],
    pub wrapper_state_sha256: [u8; 32],
    pub driver_controller_state_sha256: [u8; 32],

    /// Explicit wrapper state from the pinned dm_control.rl.control.Environment.
    pub environment_step_count: u64,
    pub reset_next_step: bool,
    pub n_sub_steps: u32,
    /// Raw IEEE-754 bits of dm_env Environment._step_limit (positive finite or +infinity).
    pub step_limit_f64_bits: u64,
    pub legacy_step: bool,
    pub control_timestep_ns: u64,

    /// MuJoCo plugin state is part of the legacy compact list only when plugins
    /// exist, and must be present in a complete snapshot when nplugin > 0.
    pub model_plugin_count: u32,
    pub plugin_state_included: bool,

    /// Task RandomState and MuJoCo sensor-noise RNG are independent domains.
    pub sensor_noise_enabled: bool,
    pub mujoco_sensor_rng_state_sha256: Option<[u8; 32]>,

    /// Common exogenous schedule used by all sibling interventions.
    pub exogenous_schedule_sha256: [u8; 32],

    /// Branch point and factual-prefix evidence.
    pub branch_point_step: u64,
    pub prefix_trace_sha256: [u8; 32],
    pub prefix_state_snapshot_sha256: [u8; 32],
    pub prefix_observation_sha256: [u8; 32],

    /// An auxiliary rest/settling operation is only admissible if it is already
    /// part of the shared factual prefix. A branch-only rest is invalid.
    pub rest_operation_sha256: Option<[u8; 32]>,
    pub rest_operation_in_factual_prefix: bool,

    /// Canonical digest of this complete caller-supplied manifest.
    pub snapshot_manifest_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmcBranchRestoreEvidence {
    pub run_id: u64,
    pub action_id: String,
    pub action_sequence_sha256: [u8; 32],
    pub restored_snapshot_manifest_sha256: [u8; 32],
    pub exogenous_schedule_sha256: [u8; 32],
    pub prefix_trace_sha256: [u8; 32],
    pub prefix_step_count: u64,
    pub branch_point_step: u64,
    pub prefix_state_max_abs_error_nano: u64,
    pub prefix_observation_max_abs_error_nano: u64,
    pub fresh_process_replay_passed: bool,
    pub rest_operation_sha256: Option<[u8; 32]>,
    pub rest_operation_in_factual_prefix: bool,
    /// May differ across sibling actions after the intervention. Equality is
    /// also valid for a true null/action-invariant case.
    pub post_intervention_trajectory_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmcRestoreCohort {
    pub expected_action_ids: Vec<String>,
    pub snapshot: DmcSnapshotManifest,
    pub branches: Vec<DmcBranchRestoreEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmcAdmittedBranch {
    pub action_id: String,
    pub run_id: u64,
    pub post_intervention_trajectory_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DmcRestoreCertificate {
    pub dm_control_commit: String,
    pub model_mjcf_sha256: [u8; 32],
    pub snapshot_manifest_sha256: [u8; 32],
    pub exogenous_schedule_sha256: [u8; 32],
    pub branch_point_step: u64,
    pub prefix_trace_sha256: [u8; 32],
    /// Action/run/trajectory lineage remains bound as one canonical tuple.
    pub admitted_branches: Vec<DmcAdmittedBranch>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DmcRestoreError {
    WrongDmControlCommit,
    WrongDmControlVersion,
    WrongMujocoVersion,
    EmptyRuntimePlatform,
    EmptyRuntimeArchitecture,
    EmptyTaskClass,
    MissingDigest { field: &'static str },
    LegacyCompactStateRejected,
    WrongIntegrationSignature,
    EmptyIntegrationState,
    SharedModelCopyRejected,
    MissingPluginState,
    MissingSensorNoiseRngState,
    WrapperResetPending,
    ZeroActionRepeat,
    ZeroControlTimestep,
    BranchPointStepMismatch,
    StepLimitAlreadyReached,
    InvalidStepLimit,
    RestOperationNotInFactualPrefix,
    EmptyActionSet,
    TooManyBranches { requested: usize, maximum: usize },
    EmptyActionId,
    DuplicateExpectedActionId { action_id: String },
    WrongBranchCount { expected: usize, actual: usize },
    DuplicateBranchActionId { action_id: String },
    UnexpectedBranchActionId { action_id: String },
    DuplicateRunId { run_id: u64 },
    MissingExpectedBranch { action_id: String },
    ZeroRunId,
    ZeroActionSequenceDigest,
    SnapshotManifestMismatch { action_id: String },
    ExogenousScheduleMismatch { action_id: String },
    PrefixTraceMismatch { action_id: String },
    PrefixStepCountMismatch { action_id: String },
    PrefixBranchPointMismatch { action_id: String },
    PrefixStateMismatch {
        action_id: String,
        error_nano: u64,
        maximum_nano: u64,
    },
    PrefixObservationMismatch {
        action_id: String,
        error_nano: u64,
        maximum_nano: u64,
    },
    FreshProcessReplayMissing { action_id: String },
    BranchRestOperationMismatch { action_id: String },
    MissingPostInterventionTrajectory { action_id: String },
}

impl fmt::Display for DmcRestoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongDmControlCommit => write!(formatter, "dm_control source commit does not match the pinned DMC pilot"),
            Self::WrongDmControlVersion => write!(formatter, "dm_control package version does not match the pinned DMC pilot"),
            Self::WrongMujocoVersion => write!(formatter, "MuJoCo package version does not match the pinned DMC pilot"),
            Self::EmptyRuntimePlatform => write!(formatter, "runtime platform must be declared"),
            Self::EmptyRuntimeArchitecture => write!(formatter, "runtime architecture must be declared"),
            Self::EmptyTaskClass => write!(formatter, "DMC task class must be declared"),
            Self::MissingDigest { field } => write!(formatter, "required DMC restore digest {field} is missing"),
            Self::LegacyCompactStateRejected => write!(formatter, "legacy Physics.get_state() without sig is not a complete branch-point snapshot"),
            Self::WrongIntegrationSignature => write!(formatter, "physics state signature must be mjSTATE_INTEGRATION"),
            Self::EmptyIntegrationState => write!(formatter, "integration-state signature, size and digest must be nonzero"),
            Self::SharedModelCopyRejected => write!(formatter, "deep Physics copy must not share the MuJoCo model"),
            Self::MissingPluginState => write!(formatter, "MuJoCo model has plugins but the restore snapshot omits plugin state"),
            Self::MissingSensorNoiseRngState => write!(formatter, "sensor noise is enabled but its independent MuJoCo RNG state is not captured"),
            Self::WrapperResetPending => write!(formatter, "dm_env wrapper says the next step must reset; this is not an admissible mid-episode branch point"),
            Self::ZeroActionRepeat => write!(formatter, "n_sub_steps/action repeat must be nonzero"),
            Self::ZeroControlTimestep => write!(formatter, "control timestep must be nonzero"),
            Self::BranchPointStepMismatch => write!(formatter, "wrapper step count must equal the declared branch-point control step"),
            Self::StepLimitAlreadyReached => write!(formatter, "branch point is at or beyond the declared environment step limit"),
            Self::InvalidStepLimit => write!(formatter, "environment step limit must be a positive finite value or positive infinity"),
            Self::RestOperationNotInFactualPrefix => write!(formatter, "auxiliary rest/settling operation is not part of the common factual prefix"),
            Self::EmptyActionSet => write!(formatter, "DMC restore cohort requires at least two expected action IDs"),
            Self::TooManyBranches { requested, maximum } => write!(formatter, "DMC restore cohort has {requested} branches; maximum is {maximum}"),
            Self::EmptyActionId => write!(formatter, "DMC branch action ID must be nonempty"),
            Self::DuplicateExpectedActionId { action_id } => write!(formatter, "duplicate expected DMC action ID {action_id}"),
            Self::WrongBranchCount { expected, actual } => write!(formatter, "DMC branch count {actual} does not match expected action count {expected}"),
            Self::DuplicateBranchActionId { action_id } => write!(formatter, "duplicate DMC branch action ID {action_id}"),
            Self::UnexpectedBranchActionId { action_id } => write!(formatter, "unexpected DMC branch action ID {action_id}"),
            Self::DuplicateRunId { run_id } => write!(formatter, "duplicate DMC branch run ID {run_id}"),
            Self::MissingExpectedBranch { action_id } => write!(formatter, "missing DMC branch action ID {action_id}"),
            Self::ZeroRunId => write!(formatter, "DMC branch run ID must be nonzero"),
            Self::ZeroActionSequenceDigest => write!(formatter, "DMC action-sequence digest must be nonzero"),
            Self::SnapshotManifestMismatch { action_id } => write!(formatter, "branch {action_id} does not restore the common complete snapshot manifest"),
            Self::ExogenousScheduleMismatch { action_id } => write!(formatter, "branch {action_id} does not share the frozen exogenous schedule"),
            Self::PrefixTraceMismatch { action_id } => write!(formatter, "branch {action_id} factual-prefix digest differs from the common reference"),
            Self::PrefixStepCountMismatch { action_id } => write!(formatter, "branch {action_id} factual-prefix step count is inconsistent"),
            Self::PrefixBranchPointMismatch { action_id } => write!(formatter, "branch {action_id} starts its intervention at a different control step"),
            Self::PrefixStateMismatch { action_id, error_nano, maximum_nano } => write!(formatter, "branch {action_id} prefix state error {error_nano} nano-units exceeds {maximum_nano}"),
            Self::PrefixObservationMismatch { action_id, error_nano, maximum_nano } => write!(formatter, "branch {action_id} prefix observation error {error_nano} nano-units exceeds {maximum_nano}"),
            Self::FreshProcessReplayMissing { action_id } => write!(formatter, "branch {action_id} has no passing fresh-process replay"),
            Self::BranchRestOperationMismatch { action_id } => write!(formatter, "branch {action_id} rest/settling operation differs from the common factual prefix"),
            Self::MissingPostInterventionTrajectory { action_id } => write!(formatter, "branch {action_id} has no post-intervention trajectory digest"),
        }
    }
}

impl std::error::Error for DmcRestoreError {}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn require_digest(digest: &[u8; 32], field: &'static str) -> Result<(), DmcRestoreError> {
    if digest_is_empty(digest) {
        return Err(DmcRestoreError::MissingDigest { field });
    }
    Ok(())
}

fn validate_snapshot(snapshot: &DmcSnapshotManifest) -> Result<(), DmcRestoreError> {
    if snapshot.dm_control_commit != PINNED_DM_CONTROL_COMMIT {
        return Err(DmcRestoreError::WrongDmControlCommit);
    }
    if snapshot.dm_control_version != PINNED_DM_CONTROL_VERSION {
        return Err(DmcRestoreError::WrongDmControlVersion);
    }
    if snapshot.mujoco_version != PINNED_MUJOCO_VERSION {
        return Err(DmcRestoreError::WrongMujocoVersion);
    }
    if snapshot.runtime_platform.is_empty() {
        return Err(DmcRestoreError::EmptyRuntimePlatform);
    }
    if snapshot.runtime_architecture.is_empty() {
        return Err(DmcRestoreError::EmptyRuntimeArchitecture);
    }
    if snapshot.task_class.is_empty() {
        return Err(DmcRestoreError::EmptyTaskClass);
    }

    for (digest, field) in [
        (&snapshot.model_mjcf_sha256, "model_mjcf_sha256"),
        (&snapshot.task_configuration_sha256, "task_configuration_sha256"),
        (&snapshot.task_private_state_sha256, "task_private_state_sha256"),
        (&snapshot.task_random_state_sha256, "task_random_state_sha256"),
        (&snapshot.wrapper_state_sha256, "wrapper_state_sha256"),
        (&snapshot.driver_controller_state_sha256, "driver_controller_state_sha256"),
        (&snapshot.exogenous_schedule_sha256, "exogenous_schedule_sha256"),
        (&snapshot.prefix_trace_sha256, "prefix_trace_sha256"),
        (&snapshot.prefix_state_snapshot_sha256, "prefix_state_snapshot_sha256"),
        (&snapshot.prefix_observation_sha256, "prefix_observation_sha256"),
        (&snapshot.snapshot_manifest_sha256, "snapshot_manifest_sha256"),
    ] {
        require_digest(digest, field)?;
    }

    match &snapshot.physics_snapshot {
        PhysicsSnapshotKind::LegacyCompactGetState { .. } => {
            return Err(DmcRestoreError::LegacyCompactStateRejected);
        }
        PhysicsSnapshotKind::IntegrationState {
            signature_name,
            signature_value,
            state_size,
            state_sha256,
        } => {
            if signature_name != "mjSTATE_INTEGRATION"
                || *signature_value != PINNED_MJ_STATE_INTEGRATION_MASK
            {
                return Err(DmcRestoreError::WrongIntegrationSignature);
            }
            if *state_size == 0 || digest_is_empty(state_sha256) {
                return Err(DmcRestoreError::EmptyIntegrationState);
            }
        }
        PhysicsSnapshotKind::DeepPhysicsCopy {
            share_model,
            copied_model_sha256,
            copied_data_sha256,
        } => {
            if *share_model {
                return Err(DmcRestoreError::SharedModelCopyRejected);
            }
            require_digest(copied_model_sha256, "copied_model_sha256")?;
            require_digest(copied_data_sha256, "copied_data_sha256")?;
        }
    }

    if snapshot.model_plugin_count > 0 && !snapshot.plugin_state_included {
        return Err(DmcRestoreError::MissingPluginState);
    }
    if snapshot.sensor_noise_enabled
        && snapshot
            .mujoco_sensor_rng_state_sha256
            .as_ref()
            .is_none_or(digest_is_empty)
    {
        return Err(DmcRestoreError::MissingSensorNoiseRngState);
    }
    if snapshot.reset_next_step {
        return Err(DmcRestoreError::WrapperResetPending);
    }
    let step_limit = f64::from_bits(snapshot.step_limit_f64_bits);
    if step_limit.is_nan() || step_limit <= 0.0 {
        return Err(DmcRestoreError::InvalidStepLimit);
    }
    if step_limit.is_finite() && snapshot.environment_step_count as f64 >= step_limit {
        return Err(DmcRestoreError::StepLimitAlreadyReached);
    }
    if snapshot.n_sub_steps == 0 {
        return Err(DmcRestoreError::ZeroActionRepeat);
    }
    if snapshot.control_timestep_ns == 0 {
        return Err(DmcRestoreError::ZeroControlTimestep);
    }
    if snapshot.environment_step_count != snapshot.branch_point_step {
        return Err(DmcRestoreError::BranchPointStepMismatch);
    }
    if snapshot
        .step_limit
        .is_some_and(|step_limit| snapshot.environment_step_count >= step_limit)
    {
        return Err(DmcRestoreError::StepLimitAlreadyReached);
    }
    if snapshot.rest_operation_sha256.as_ref().is_some_and(digest_is_empty) {
        return Err(DmcRestoreError::MissingDigest {
            field: "rest_operation_sha256",
        });
    }
    if snapshot.rest_operation_sha256.is_some() && !snapshot.rest_operation_in_factual_prefix {
        return Err(DmcRestoreError::RestOperationNotInFactualPrefix);
    }
    if snapshot.rest_operation_sha256.is_none() && snapshot.rest_operation_in_factual_prefix {
        return Err(DmcRestoreError::RestOperationNotInFactualPrefix);
    }
    Ok(())
}

/// Validate the DMC full-state and factual-prefix contract.
///
/// This checks caller-supplied commitments only. It does not hash MjData or
/// verify the dm_control Python object graph. The pinned Python adapter and
/// fresh-process replay must produce the supplied receipts independently.
pub fn validate_dmc_restore_cohort(
    cohort: &DmcRestoreCohort,
) -> Result<DmcRestoreCertificate, DmcRestoreError> {
    validate_snapshot(&cohort.snapshot)?;

    if cohort.expected_action_ids.len() < 2 {
        return Err(DmcRestoreError::EmptyActionSet);
    }
    if cohort.expected_action_ids.len() > MAX_DMC_BRANCHES {
        return Err(DmcRestoreError::TooManyBranches {
            requested: cohort.expected_action_ids.len(),
            maximum: MAX_DMC_BRANCHES,
        });
    }

    let mut expected = BTreeSet::new();
    for action_id in &cohort.expected_action_ids {
        if action_id.is_empty() {
            return Err(DmcRestoreError::EmptyActionId);
        }
        if !expected.insert(action_id.clone()) {
            return Err(DmcRestoreError::DuplicateExpectedActionId {
                action_id: action_id.clone(),
            });
        }
    }

    if cohort.branches.len() != expected.len() {
        return Err(DmcRestoreError::WrongBranchCount {
            expected: expected.len(),
            actual: cohort.branches.len(),
        });
    }

    let mut action_ids = BTreeSet::new();
    let mut run_ids = BTreeSet::new();
    for branch in &cohort.branches {
        if branch.run_id == 0 {
            return Err(DmcRestoreError::ZeroRunId);
        }
        if !run_ids.insert(branch.run_id) {
            return Err(DmcRestoreError::DuplicateRunId {
                run_id: branch.run_id,
            });
        }
        if branch.action_id.is_empty() {
            return Err(DmcRestoreError::EmptyActionId);
        }
        if !action_ids.insert(branch.action_id.clone()) {
            return Err(DmcRestoreError::DuplicateBranchActionId {
                action_id: branch.action_id.clone(),
            });
        }
        if !expected.contains(&branch.action_id) {
            return Err(DmcRestoreError::UnexpectedBranchActionId {
                action_id: branch.action_id.clone(),
            });
        }
        if branch.action_sequence_sha256.iter().all(|b| *b == 0) {
            return Err(DmcRestoreError::ZeroActionSequenceDigest);
        }
        if branch.restored_snapshot_manifest_sha256 != cohort.snapshot.snapshot_manifest_sha256 {
            return Err(DmcRestoreError::SnapshotManifestMismatch {
                action_id: branch.action_id.clone(),
            });
        }
        if branch.exogenous_schedule_sha256 != cohort.snapshot.exogenous_schedule_sha256 {
            return Err(DmcRestoreError::ExogenousScheduleMismatch {
                action_id: branch.action_id.clone(),
            });
        }
        if branch.prefix_trace_sha256 != cohort.snapshot.prefix_trace_sha256 {
            return Err(DmcRestoreError::PrefixTraceMismatch {
                action_id: branch.action_id.clone(),
            });
        }
        if branch.prefix_step_count != cohort.snapshot.branch_point_step {
            return Err(DmcRestoreError::PrefixStepCountMismatch {
                action_id: branch.action_id.clone(),
            });
        }
        if branch.branch_point_step != cohort.snapshot.branch_point_step {
            return Err(DmcRestoreError::PrefixBranchPointMismatch {
                action_id: branch.action_id.clone(),
            });
        }
        if branch.prefix_state_max_abs_error_nano > PREFIX_MATCH_MAX_ERROR_NANO {
            return Err(DmcRestoreError::PrefixStateMismatch {
                action_id: branch.action_id.clone(),
                error_nano: branch.prefix_state_max_abs_error_nano,
                maximum_nano: PREFIX_MATCH_MAX_ERROR_NANO,
            });
        }
        if branch.prefix_observation_max_abs_error_nano > PREFIX_MATCH_MAX_ERROR_NANO {
            return Err(DmcRestoreError::PrefixObservationMismatch {
                action_id: branch.action_id.clone(),
                error_nano: branch.prefix_observation_max_abs_error_nano,
                maximum_nano: PREFIX_MATCH_MAX_ERROR_NANO,
            });
        }
        if !branch.fresh_process_replay_passed {
            return Err(DmcRestoreError::FreshProcessReplayMissing {
                action_id: branch.action_id.clone(),
            });
        }
        if branch.rest_operation_sha256 != cohort.snapshot.rest_operation_sha256
            || branch.rest_operation_in_factual_prefix != cohort.snapshot.rest_operation_in_factual_prefix
        {
            return Err(DmcRestoreError::BranchRestOperationMismatch {
                action_id: branch.action_id.clone(),
            });
        }
        if digest_is_empty(&branch.post_intervention_trajectory_sha256) {
            return Err(DmcRestoreError::MissingPostInterventionTrajectory {
                action_id: branch.action_id.clone(),
            });
        }
    }

    for action_id in &expected {
        if !action_ids.contains(action_id) {
            return Err(DmcRestoreError::MissingExpectedBranch {
                action_id: action_id.clone(),
            });
        }
    }

    let mut admitted_branches: Vec<DmcAdmittedBranch> = cohort
        .branches
        .iter()
        .map(|branch| DmcAdmittedBranch {
            action_id: branch.action_id.clone(),
            run_id: branch.run_id,
            post_intervention_trajectory_sha256: branch.post_intervention_trajectory_sha256,
        })
        .collect();
    admitted_branches.sort_unstable_by(|left, right| left.action_id.cmp(&right.action_id));

    Ok(DmcRestoreCertificate {
        dm_control_commit: cohort.snapshot.dm_control_commit.clone(),
        model_mjcf_sha256: cohort.snapshot.model_mjcf_sha256,
        snapshot_manifest_sha256: cohort.snapshot.snapshot_manifest_sha256,
        exogenous_schedule_sha256: cohort.snapshot.exogenous_schedule_sha256,
        branch_point_step: cohort.snapshot.branch_point_step,
        prefix_trace_sha256: cohort.snapshot.prefix_trace_sha256,
        admitted_branches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn valid_snapshot() -> DmcSnapshotManifest {
        DmcSnapshotManifest {
            dm_control_commit: PINNED_DM_CONTROL_COMMIT.to_owned(),
            dm_control_version: PINNED_DM_CONTROL_VERSION.to_owned(),
            mujoco_version: PINNED_MUJOCO_VERSION.to_owned(),
            runtime_platform: "linux-x86_64".to_owned(),
            runtime_architecture: "x86_64".to_owned(),
            model_mjcf_sha256: digest(1),
            physics_snapshot: PhysicsSnapshotKind::IntegrationState {
                signature_name: "mjSTATE_INTEGRATION".to_owned(),
                signature_value: PINNED_MJ_STATE_INTEGRATION_MASK,
                state_size: 42,
                state_sha256: digest(2),
            },
            task_class: "cartpole.swingup".to_owned(),
            task_configuration_sha256: digest(3),
            task_private_state_sha256: digest(4),
            task_random_state_sha256: digest(5),
            wrapper_state_sha256: digest(6),
            driver_controller_state_sha256: digest(7),
            environment_step_count: 12,
            reset_next_step: false,
            n_sub_steps: 1,
            step_limit_f64_bits: 1000.0_f64.to_bits(),
            legacy_step: true,
            control_timestep_ns: 10_000_000,
            model_plugin_count: 0,
            plugin_state_included: false,
            sensor_noise_enabled: false,
            mujoco_sensor_rng_state_sha256: None,
            exogenous_schedule_sha256: digest(8),
            branch_point_step: 12,
            prefix_trace_sha256: digest(9),
            prefix_state_snapshot_sha256: digest(10),
            prefix_observation_sha256: digest(11),
            rest_operation_sha256: None,
            rest_operation_in_factual_prefix: false,
            snapshot_manifest_sha256: digest(12),
        }
    }

    fn branch(action_id: &str, run_id: u64, snapshot: &DmcSnapshotManifest) -> DmcBranchRestoreEvidence {
        DmcBranchRestoreEvidence {
            run_id,
            action_id: action_id.to_owned(),
            action_sequence_sha256: digest(run_id as u8 + 20),
            restored_snapshot_manifest_sha256: snapshot.snapshot_manifest_sha256,
            exogenous_schedule_sha256: snapshot.exogenous_schedule_sha256,
            prefix_trace_sha256: snapshot.prefix_trace_sha256,
            prefix_step_count: snapshot.branch_point_step,
            branch_point_step: snapshot.branch_point_step,
            prefix_state_max_abs_error_nano: 1,
            prefix_observation_max_abs_error_nano: 1,
            fresh_process_replay_passed: true,
            rest_operation_sha256: snapshot.rest_operation_sha256,
            rest_operation_in_factual_prefix: snapshot.rest_operation_in_factual_prefix,
            post_intervention_trajectory_sha256: digest(run_id as u8 + 40),
        }
    }

    fn cohort() -> DmcRestoreCohort {
        let snapshot = valid_snapshot();
        DmcRestoreCohort {
            expected_action_ids: vec!["a0".to_owned(), "a1".to_owned()],
            branches: vec![branch("a0", 100, &snapshot), branch("a1", 101, &snapshot)],
            snapshot,
        }
    }

    #[test]
    fn complete_integration_state_and_fresh_prefix_admit() {
        let result = validate_dmc_restore_cohort(&cohort()).unwrap();
        assert_eq!(result.branch_point_step, 12);
        assert_eq!(result.admitted_branches.len(), 2);
        assert_eq!(result.admitted_branches[0].action_id, "a0");
        assert_eq!(result.admitted_branches[0].run_id, 100);
        assert_eq!(result.admitted_branches[1].action_id, "a1");
        assert_eq!(result.admitted_branches[1].run_id, 101);
    }

    #[test]
    fn legacy_compact_get_state_is_rejected() {
        let mut input = cohort();
        input.snapshot.physics_snapshot = PhysicsSnapshotKind::LegacyCompactGetState {
            state_sha256: digest(2),
        };
        let error = validate_dmc_restore_cohort(&input).unwrap_err();
        assert_eq!(error, DmcRestoreError::LegacyCompactStateRejected);
    }

    #[test]
    fn wrong_dm_control_source_commit_rejects() {
        let mut input = cohort();
        input.snapshot.dm_control_commit = "0000000000000000000000000000000000000000".to_owned();
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::WrongDmControlCommit
        );
    }

    #[test]
    fn wrong_mujoco_version_rejects() {
        let mut input = cohort();
        input.snapshot.mujoco_version = "3.3.2".to_owned();
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::WrongMujocoVersion
        );
    }

    #[test]
    fn wrong_integration_signature_value_rejects() {
        let mut input = cohort();
        if let PhysicsSnapshotKind::IntegrationState {
            signature_value,
            ..
        } = &mut input.snapshot.physics_snapshot
        {
            *signature_value = PINNED_MJ_STATE_INTEGRATION_MASK - 1;
        }
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::WrongIntegrationSignature
        );
    }

    #[test]
    fn invalid_step_limit_rejects_nan() {
        let mut input = cohort();
        input.snapshot.step_limit_f64_bits = f64::NAN.to_bits();
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::InvalidStepLimit
        );
    }

    #[test]
    fn positive_infinite_step_limit_is_admissible() {
        let mut input = cohort();
        input.snapshot.step_limit_f64_bits = f64::INFINITY.to_bits();
        assert!(validate_dmc_restore_cohort(&input).is_ok());
    }

    #[test]
    fn invalid_integration_signature_rejects() {
        let mut input = cohort();
        input.snapshot.physics_snapshot = PhysicsSnapshotKind::IntegrationState {
            signature_name: "mjSTATE_FULLPHYSICS".to_owned(),
            signature_value: PINNED_MJ_STATE_INTEGRATION_MASK,
            state_size: 42,
            state_sha256: digest(2),
        };
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::WrongIntegrationSignature
        );
    }

    #[test]
    fn plugin_state_is_mandatory_when_model_has_plugins() {
        let mut input = cohort();
        input.snapshot.model_plugin_count = 1;
        input.snapshot.plugin_state_included = false;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::MissingPluginState
        );
    }

    #[test]
    fn sensor_noise_requires_separate_mujoco_rng_state() {
        let mut input = cohort();
        input.snapshot.sensor_noise_enabled = true;
        input.snapshot.mujoco_sensor_rng_state_sha256 = None;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::MissingSensorNoiseRngState
        );
    }

    #[test]
    fn reset_pending_wrapper_is_not_a_mid_episode_branch_point() {
        let mut input = cohort();
        input.snapshot.reset_next_step = true;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::WrapperResetPending
        );
    }

    #[test]
    fn wrapper_step_count_must_equal_branch_point() {
        let mut input = cohort();
        input.snapshot.environment_step_count = 11;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::BranchPointStepMismatch
        );
    }

    #[test]
    fn zero_action_repeat_rejects() {
        let mut input = cohort();
        input.snapshot.n_sub_steps = 0;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::ZeroActionRepeat
        );
    }

    #[test]
    fn branch_only_rest_operation_rejects() {
        let mut input = cohort();
        input.snapshot.rest_operation_sha256 = Some(digest(13));
        input.snapshot.rest_operation_in_factual_prefix = false;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::RestOperationNotInFactualPrefix
        );
    }

    #[test]
    fn symmetric_rest_operation_in_factual_prefix_is_admissible() {
        let mut input = cohort();
        input.snapshot.rest_operation_sha256 = Some(digest(13));
        input.snapshot.rest_operation_in_factual_prefix = true;
        for branch in &mut input.branches {
            branch.rest_operation_sha256 = Some(digest(13));
            branch.rest_operation_in_factual_prefix = true;
        }
        assert!(validate_dmc_restore_cohort(&input).is_ok());
    }

    #[test]
    fn branch_snapshot_mismatch_rejects() {
        let mut input = cohort();
        input.branches[1].restored_snapshot_manifest_sha256 = digest(99);
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::SnapshotManifestMismatch {
                action_id: "a1".to_owned()
            }
        );
    }

    #[test]
    fn prefix_state_error_above_one_e_minus_nine_rejects() {
        let mut input = cohort();
        input.branches[1].prefix_state_max_abs_error_nano = 2;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::PrefixStateMismatch {
                action_id: "a1".to_owned(),
                error_nano: 2,
                maximum_nano: 1
            }
        );
    }

    #[test]
    fn prefix_observation_error_above_one_e_minus_nine_rejects() {
        let mut input = cohort();
        input.branches[0].prefix_observation_max_abs_error_nano = 2;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::PrefixObservationMismatch {
                action_id: "a0".to_owned(),
                error_nano: 2,
                maximum_nano: 1
            }
        );
    }

    #[test]
    fn exogenous_schedule_mismatch_rejects() {
        let mut input = cohort();
        input.branches[1].exogenous_schedule_sha256 = digest(77);
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::ExogenousScheduleMismatch {
                action_id: "a1".to_owned()
            }
        );
    }

    #[test]
    fn fresh_process_replay_is_required() {
        let mut input = cohort();
        input.branches[0].fresh_process_replay_passed = false;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::FreshProcessReplayMissing {
                action_id: "a0".to_owned()
            }
        );
    }

    #[test]
    fn branch_local_trajectories_may_diverge_after_intervention() {
        let mut input = cohort();
        input.branches[0].post_intervention_trajectory_sha256 = digest(90);
        input.branches[1].post_intervention_trajectory_sha256 = digest(91);
        assert!(validate_dmc_restore_cohort(&input).is_ok());
    }

    #[test]
    fn identical_post_intervention_trajectories_are_allowed_for_null_case() {
        let mut input = cohort();
        input.branches[1].post_intervention_trajectory_sha256 =
            input.branches[0].post_intervention_trajectory_sha256;
        assert!(validate_dmc_restore_cohort(&input).is_ok());
    }

    #[test]
    fn duplicate_or_missing_action_branch_rejects() {
        let mut input = cohort();
        input.branches[1].action_id = "a0".to_owned();
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::DuplicateBranchActionId {
                action_id: "a0".to_owned()
            }
        );
    }

    #[test]
    fn integration_snapshot_with_zero_digest_rejects() {
        let mut input = cohort();
        input.snapshot.physics_snapshot = PhysicsSnapshotKind::IntegrationState {
            signature_name: "mjSTATE_INTEGRATION".to_owned(),
            signature_value: PINNED_MJ_STATE_INTEGRATION_MASK,
            state_size: 42,
            state_sha256: [0; 32],
        };
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::EmptyIntegrationState
        );
    }

    #[test]
    fn duplicate_run_id_rejects() {
        let mut input = cohort();
        input.branches[1].run_id = input.branches[0].run_id;
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::DuplicateRunId { run_id: 100 }
        );
    }

    #[test]
    fn deep_copy_must_not_share_mutable_model() {
        let mut input = cohort();
        input.snapshot.physics_snapshot = PhysicsSnapshotKind::DeepPhysicsCopy {
            share_model: true,
            copied_model_sha256: digest(2),
            copied_data_sha256: digest(3),
        };
        assert_eq!(
            validate_dmc_restore_cohort(&input).unwrap_err(),
            DmcRestoreError::SharedModelCopyRejected
        );
    }

    #[test]
    fn deep_physics_copy_is_accepted_when_model_and_data_are_bound() {
        let mut input = cohort();
        input.snapshot.physics_snapshot = PhysicsSnapshotKind::DeepPhysicsCopy {
            share_model: false,
            copied_model_sha256: digest(2),
            copied_data_sha256: digest(3),
        };
        assert!(validate_dmc_restore_cohort(&input).is_ok());
    }
}
