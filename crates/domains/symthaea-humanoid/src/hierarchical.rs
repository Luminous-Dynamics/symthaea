// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical humanoid control: contact-aware baseline, balance correction,
//! and bounded learned residual authority.
//!
//! The learned HDC-LTC policy should not have to rediscover every low-level
//! stability reflex. This module keeps a deterministic stabilizing substrate
//! while allowing the learned policy to specialize through bounded residuals.

use serde::{Deserialize, Serialize};

use crate::centroidal::CentroidalMomentumController;
use crate::contact::{BipedSupport, ContactFrame};
use crate::contact_inverse_dynamics::ReducedOrderContactInverseDynamicsController;
use crate::dynamics::RigidBodyDynamicsProvider;
use crate::floating_base::FloatingBaseDynamicsProvider;
use crate::floating_base_inverse_dynamics::{
    FloatingBaseInverseDynamicsController, FloatingBaseInverseDynamicsReport,
};
use crate::footstep::{FootstepPlan, ModelPredictiveFootstepPlanner};
use crate::full_dynamics::FullRigidBodyDynamicsProvider;
use crate::inverse_dynamics::SparseQpInverseDynamicsController;
use crate::morphology::HumanoidMorphology;
use crate::recovery::{CapturePointRecoveryController, RecoveryMode};
use crate::terrain::{FlatTerrain, SwingTrajectory, TerrainAwareSwingPlanner, TerrainProbe};
use crate::terrain_mpc::RecedingHorizonTerrainPlanner;
use crate::types::{HumanoidCommand, HumanoidState, HumanoidTask};
use crate::whole_body::WholeBodyObjective;

/// Estimated bilateral support state from foot height and vertical motion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportPhase {
    Flight,
    RightStance,
    LeftStance,
    DoubleSupport,
}

impl SupportPhase {
    pub const fn stance_count(self) -> usize {
        match self {
            Self::Flight => 0,
            Self::RightStance | Self::LeftStance => 1,
            Self::DoubleSupport => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ContactEstimate {
    pub phase: SupportPhase,
    pub right_contact: bool,
    pub left_contact: bool,
    pub right_foot_height: f64,
    pub left_foot_height: f64,
}

impl ContactEstimate {
    pub fn from_state(state: &HumanoidState, threshold_m: f64) -> Self {
        let right_foot_height = state.extremities.get(8).copied().unwrap_or(f64::INFINITY);
        let left_foot_height = state.extremities.get(11).copied().unwrap_or(f64::INFINITY);
        let right_contact = right_foot_height.is_finite() && right_foot_height <= threshold_m;
        let left_contact = left_foot_height.is_finite() && left_foot_height <= threshold_m;
        let phase = match (right_contact, left_contact) {
            (false, false) => SupportPhase::Flight,
            (true, false) => SupportPhase::RightStance,
            (false, true) => SupportPhase::LeftStance,
            (true, true) => SupportPhase::DoubleSupport,
        };
        Self {
            phase,
            right_contact,
            left_contact,
            right_foot_height,
            left_foot_height,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalControlConfig {
    /// Maximum learned residual mixed into the deterministic baseline.
    pub max_residual_authority: f32,
    /// Minimum deterministic baseline retained after curriculum handoff.
    pub baseline_floor: f32,
    /// Residual authority retained while standing.
    pub standing_residual_authority: f32,
    /// Foot height treated as contact.
    pub contact_threshold_m: f64,
    /// Roll correction gain applied to abdomen/hips/ankles.
    pub roll_balance_gain: f64,
    /// Pitch correction gain applied to abdomen/hips/ankles.
    pub pitch_balance_gain: f64,
    /// Angular-rate damping gain.
    pub angular_damping_gain: f64,
    /// Additional authority suppression as free energy rises.
    pub free_energy_suppression: f64,
}

impl Default for HierarchicalControlConfig {
    fn default() -> Self {
        Self {
            max_residual_authority: 0.45,
            baseline_floor: 0.55,
            standing_residual_authority: 0.20,
            contact_threshold_m: 0.04,
            roll_balance_gain: 0.55,
            pitch_balance_gain: 0.40,
            angular_damping_gain: 0.10,
            free_energy_suppression: 0.35,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalControlReport {
    pub support_phase: SupportPhase,
    pub residual_authority: f32,
    pub balance_effort: f32,
    pub baseline_weight: f32,
    pub recovery_mode: RecoveryMode,
    pub capture_margin_m: f64,
    pub recovery_effort: f32,
    /// Predictive recovery step selected when the capture point leaves support.
    pub planned_step: Option<FootstepPlan>,
    /// Terrain-shaped swing trajectory for the active recovery step.
    pub planned_swing: Option<SwingTrajectory>,
    /// Minimum terrain confidence along the active swing path.
    pub terrain_confidence: f64,
    /// Vertical swing clearance selected above the support surface.
    pub terrain_clearance_m: f64,
    /// Number of active projected dynamics constraints.
    pub whole_body_active_constraints: usize,
    /// Maximum normalized joint-range utilization observed by the allocator.
    pub whole_body_joint_utilization: f64,
    /// Residual torso objective error after allocation.
    pub whole_body_objective_residual: f64,
    /// Whether the projected whole-body allocation remained finite and feasible.
    pub whole_body_feasible: bool,
    /// Iterations used by the sparse projected-QP allocator.
    pub inverse_dynamics_iterations: usize,
    /// Maximum remaining sparse-QP constraint violation.
    pub inverse_dynamics_max_violation: f64,
    /// Whether the inverse-dynamics layer fell back to the seed allocator.
    pub inverse_dynamics_fallback: bool,
    /// Number of feasible future footsteps retained by the terrain MPC.
    pub terrain_horizon_steps: usize,
    /// Aggregate receding-horizon terrain cost.
    pub terrain_mpc_cost: f64,
    /// Candidate footholds evaluated during this replanning tick.
    pub terrain_mpc_candidates: usize,
    /// Whether the reduced-order contact dynamics solve satisfied its contracts.
    pub contact_dynamics_converged: bool,
    /// Whether the contact dynamics layer retained the sparse-QP seed command.
    pub contact_dynamics_fallback: bool,
    /// Maximum residual of the reduced dynamics equation in N m.
    pub contact_dynamics_residual_nm: f64,
    /// Maximum stance-foot acceleration residual.
    pub contact_acceleration_residual: f64,
    /// Maximum estimated tangential-friction utilization.
    pub contact_friction_utilization: f64,
    /// Whether the contact solver exceeded its admitted operation or time budget.
    pub contact_solver_budget_missed: bool,
    /// Measured solver wall time for budget evidence.
    pub contact_solver_elapsed_us: u64,
    /// Largest terrain height uncertainty retained in the active horizon.
    pub terrain_max_height_std_m: f64,
    /// Oldest terrain evidence retained in the active horizon.
    pub terrain_max_evidence_age_s: f64,
    /// Confidence and freshness gate derived from the contact frame.
    pub contact_trust: f32,
    /// Whether a dimensionally valid centroidal model informed this command.
    pub centroidal_model_valid: bool,
    /// Authority admitted for centroidal momentum damping.
    pub centroidal_authority: f64,
    /// Euclidean norm of the bounded centroidal correction.
    pub centroidal_correction_norm: f64,
    /// Current angular centroidal momentum norm.
    pub angular_momentum_norm: f64,
    /// Current linear centroidal momentum norm.
    pub linear_momentum_norm: f64,
    /// Whether a validated floating-base model was admitted this tick.
    pub floating_base_model_available: bool,
    /// Whether all six base equations and stance constraints converged.
    pub floating_base_dynamics_converged: bool,
    /// Whether the last bounded command was retained after solver rejection.
    pub floating_base_dynamics_fallback: bool,
    /// Maximum generalized Newton-Euler equation residual.
    pub floating_base_dynamics_residual: f64,
    /// Wall time consumed by the floating-base solve.
    pub floating_base_solver_elapsed_us: u64,
    /// Whether the floating-base solver missed admission or its deadline.
    pub floating_base_solver_budget_missed: bool,
    /// Whether a compatible previous solution seeded active bounds.
    #[serde(default)]
    pub floating_base_warm_start_used: bool,
    /// Number of active bounds recovered from the previous solution.
    #[serde(default)]
    pub floating_base_warm_start_active_bounds: usize,
    /// Whether the backend reused its symbolic sparsity pattern.
    #[serde(default)]
    pub floating_base_symbolic_pattern_reused: bool,
    /// Stable QP backend identity.
    #[serde(default)]
    pub floating_base_solver_backend_id: Option<String>,
    /// Stable backend model identifier used by the solve.
    pub floating_base_model_id: Option<String>,
}

pub struct HierarchicalHumanoidController {
    morphology: HumanoidMorphology,
    config: HierarchicalControlConfig,
    recovery: CapturePointRecoveryController,
    footstep_planner: ModelPredictiveFootstepPlanner,
    terrain_mpc: RecedingHorizonTerrainPlanner,
    swing_planner: TerrainAwareSwingPlanner,
    inverse_dynamics: SparseQpInverseDynamicsController,
    contact_inverse_dynamics: ReducedOrderContactInverseDynamicsController,
    floating_base_inverse_dynamics: FloatingBaseInverseDynamicsController,
    centroidal: CentroidalMomentumController,
}

impl HierarchicalHumanoidController {
    pub fn new(morphology: HumanoidMorphology) -> Self {
        Self::with_config(morphology, HierarchicalControlConfig::default())
    }

    pub fn with_config(morphology: HumanoidMorphology, config: HierarchicalControlConfig) -> Self {
        Self {
            morphology,
            config,
            recovery: CapturePointRecoveryController::new(morphology),
            footstep_planner: ModelPredictiveFootstepPlanner::new(),
            terrain_mpc: RecedingHorizonTerrainPlanner::new(),
            swing_planner: TerrainAwareSwingPlanner::new(),
            inverse_dynamics: SparseQpInverseDynamicsController::new(morphology),
            contact_inverse_dynamics: ReducedOrderContactInverseDynamicsController::new(morphology),
            floating_base_inverse_dynamics: FloatingBaseInverseDynamicsController::new(morphology),
            centroidal: CentroidalMomentumController::new(morphology),
        }
    }

    /// Blend deterministic task control with a learned residual, then add a
    /// small contact-aware balance correction. All outputs remain normalized.
    pub fn synthesize(
        &self,
        task: HumanoidTask,
        state: &HumanoidState,
        baseline: &HumanoidCommand,
        learned_residual: &HumanoidCommand,
        baseline_weight: f32,
        free_energy: f64,
    ) -> (HumanoidCommand, HierarchicalControlReport) {
        let contacts = ContactFrame::estimated_from_state(state, self.config.contact_threshold_m);
        self.synthesize_with_environment(
            task,
            state,
            &contacts,
            &FlatTerrain,
            baseline,
            learned_residual,
            baseline_weight,
            free_energy,
        )
    }

    pub fn synthesize_with_contacts(
        &self,
        task: HumanoidTask,
        state: &HumanoidState,
        contacts: &ContactFrame,
        baseline: &HumanoidCommand,
        learned_residual: &HumanoidCommand,
        baseline_weight: f32,
        free_energy: f64,
    ) -> (HumanoidCommand, HierarchicalControlReport) {
        self.synthesize_with_environment(
            task,
            state,
            contacts,
            &FlatTerrain,
            baseline,
            learned_residual,
            baseline_weight,
            free_energy,
        )
    }

    /// Synthesize control while querying the embodiment's support surface.
    /// Backends that do not expose terrain retain the flat-ground behavior.
    pub fn synthesize_with_environment<
        T: TerrainProbe
            + RigidBodyDynamicsProvider
            + FullRigidBodyDynamicsProvider
            + FloatingBaseDynamicsProvider
            + ?Sized,
    >(
        &self,
        task: HumanoidTask,
        state: &HumanoidState,
        contacts: &ContactFrame,
        terrain: &T,
        baseline: &HumanoidCommand,
        learned_residual: &HumanoidCommand,
        baseline_weight: f32,
        free_energy: f64,
    ) -> (HumanoidCommand, HierarchicalControlReport) {
        let n = self.morphology.num_actuators();
        if baseline.num_actuators() != n
            || learned_residual.num_actuators() != n
            || state.joint_angles.len() != n
        {
            return (
                HumanoidCommand::zero_for(n),
                HierarchicalControlReport {
                    support_phase: SupportPhase::Flight,
                    residual_authority: 0.0,
                    balance_effort: 0.0,
                    baseline_weight: 1.0,
                    recovery_mode: RecoveryMode::Fallen,
                    capture_margin_m: f64::NEG_INFINITY,
                    recovery_effort: 0.0,
                    planned_step: None,
                    planned_swing: None,
                    terrain_confidence: 0.0,
                    terrain_clearance_m: 0.0,
                    whole_body_active_constraints: n,
                    whole_body_joint_utilization: f64::INFINITY,
                    whole_body_objective_residual: f64::INFINITY,
                    whole_body_feasible: false,
                    inverse_dynamics_iterations: 0,
                    inverse_dynamics_max_violation: f64::INFINITY,
                    inverse_dynamics_fallback: true,
                    terrain_horizon_steps: 0,
                    terrain_mpc_cost: f64::INFINITY,
                    terrain_mpc_candidates: 0,
                    contact_dynamics_converged: false,
                    contact_dynamics_fallback: true,
                    contact_dynamics_residual_nm: f64::INFINITY,
                    contact_acceleration_residual: f64::INFINITY,
                    contact_friction_utilization: f64::INFINITY,
                    contact_solver_budget_missed: true,
                    contact_solver_elapsed_us: 0,
                    terrain_max_height_std_m: f64::INFINITY,
                    terrain_max_evidence_age_s: f64::INFINITY,
                    contact_trust: 0.0,
                    centroidal_model_valid: false,
                    centroidal_authority: 0.0,
                    centroidal_correction_norm: 0.0,
                    angular_momentum_norm: 0.0,
                    linear_momentum_norm: 0.0,
                    floating_base_model_available: false,
                    floating_base_dynamics_converged: false,
                    floating_base_dynamics_fallback: true,
                    floating_base_dynamics_residual: f64::INFINITY,
                    floating_base_solver_elapsed_us: 0,
                    floating_base_solver_budget_missed: true,
                    floating_base_warm_start_used: false,
                    floating_base_warm_start_active_bounds: 0,
                    floating_base_symbolic_pattern_reused: false,
                    floating_base_solver_backend_id: None,
                    floating_base_model_id: None,
                },
            );
        }

        let contact = ContactEstimate {
            phase: match contacts.support() {
                BipedSupport::Flight => SupportPhase::Flight,
                BipedSupport::Right => SupportPhase::RightStance,
                BipedSupport::Left => SupportPhase::LeftStance,
                BipedSupport::Double => SupportPhase::DoubleSupport,
            },
            right_contact: contacts.right.in_contact,
            left_contact: contacts.left.in_contact,
            right_foot_height: contacts.right.point_world_m[2],
            left_foot_height: contacts.left.point_world_m[2],
        };
        let contact_trust = contacts.control_trust(state.timestamp, 0.06);
        let curriculum_baseline_weight = baseline_weight.clamp(0.0, 1.0);
        let baseline_weight = self.config.baseline_floor.clamp(0.0, 1.0)
            + (1.0 - self.config.baseline_floor.clamp(0.0, 1.0)) * curriculum_baseline_weight;
        let task_authority = match task {
            HumanoidTask::Stand => self.config.standing_residual_authority,
            HumanoidTask::Walk => self.config.max_residual_authority,
            HumanoidTask::Run => self.config.max_residual_authority,
            HumanoidTask::Reach | HumanoidTask::Grasp => {
                self.config.max_residual_authority.min(0.35)
            }
        };
        let upright_gate = ((state.uprightness() - 0.15) / 0.75).clamp(0.0, 1.0) as f32;
        let support_gate = match contact.phase {
            SupportPhase::Flight => 0.45,
            SupportPhase::RightStance | SupportPhase::LeftStance => 0.85,
            SupportPhase::DoubleSupport => 1.0,
        };
        let fe_gate = (1.0 / (1.0 + self.config.free_energy_suppression * free_energy.max(0.0)))
            .clamp(0.15, 1.0) as f32;
        let curriculum_residual_gate = 1.0 - 0.5 * curriculum_baseline_weight;
        let residual_authority = task_authority
            * curriculum_residual_gate
            * upright_gate
            * support_gate
            * fe_gate
            * contact_trust;

        let mut out = HumanoidCommand::zero_for(n);
        for i in 0..n {
            let baseline_term = baseline_weight * baseline.torques[i];
            let residual_term = residual_authority * learned_residual.torques[i];
            out.torques[i] = (baseline_term + residual_term).clamp(-1.0, 1.0);
        }

        let balance_effort = self.apply_balance_correction(&mut out, state, contact.phase);
        let (recovery_command, recovery_report) = self.recovery.correction(state, contacts);
        for (value, correction) in out.torques.iter_mut().zip(recovery_command.torques.iter()) {
            *value = (*value + contact_trust * *correction).clamp(-1.0, 1.0);
        }

        let terrain_mpc_plan =
            if recovery_report.mode == RecoveryMode::CaptureStep && contact_trust >= 0.45 {
                Some(self.terrain_mpc.plan(state, contacts, terrain))
            } else {
                None
            };
        let raw_planned_step = terrain_mpc_plan
            .as_ref()
            .filter(|plan| plan.feasible)
            .and_then(|plan| plan.first_step())
            .or_else(|| {
                (recovery_report.mode == RecoveryMode::CaptureStep && contact_trust >= 0.45)
                    .then(|| self.footstep_planner.plan(state, contacts))
            });
        let planned_swing = raw_planned_step.map(|plan| {
            let start_world_m = swing_foot_position(state, plan.swing_foot);
            self.swing_planner.plan(&plan, start_world_m, terrain)
        });
        let planned_step = match (raw_planned_step, planned_swing) {
            (Some(plan), Some(swing)) => Some(FootstepPlan {
                target_world_m: swing.target_world_m,
                clearance_m: swing.clearance_m,
                feasible: plan.feasible && swing.feasible,
                confidence: (plan.confidence * swing.terrain_confidence).clamp(0.0, 1.0),
                ..plan
            }),
            (plan, None) => plan,
            (None, Some(_)) => None,
        };
        let support_center = contacts
            .center_of_pressure_world_m()
            .unwrap_or([state.root_position[0], state.root_position[1]]);
        let whole_body_objective = WholeBodyObjective {
            desired_sagittal_com_accel_mps2: -2.0
                * (recovery_report.capture_point_world_m[0] - support_center[0]),
            desired_lateral_com_accel_mps2: -2.0
                * (recovery_report.capture_point_world_m[1] - support_center[1]),
            desired_torso_pitch: 0.0,
            desired_torso_roll: 0.0,
            desired_support_ratio: match contact.phase {
                SupportPhase::DoubleSupport => 1.0,
                SupportPhase::RightStance | SupportPhase::LeftStance => 0.82,
                SupportPhase::Flight => 0.55,
            },
            planned_step,
        };
        let (mut sparse_command, inverse_dynamics_report) =
            self.inverse_dynamics
                .allocate(state, contacts, &out, whole_body_objective);
        let full_dynamics = terrain.full_dynamics_snapshot(state, contacts);
        let (centroidal_correction, centroidal_report) = match full_dynamics.as_ref() {
            Some(snapshot) => self.centroidal.correction(state, contacts, snapshot),
            None => (
                HumanoidCommand::zero_for(n),
                crate::centroidal::CentroidalMomentumReport {
                    valid_model: false,
                    support: contacts.support(),
                    current_angular_momentum: [0.0; 3],
                    current_linear_momentum: [0.0; 3],
                    target_angular_momentum_rate: [0.0; 3],
                    target_linear_momentum_rate: [0.0; 3],
                    correction_norm: 0.0,
                    authority: 0.0,
                },
            ),
        };
        for (command, correction) in sparse_command
            .torques
            .iter_mut()
            .zip(centroidal_correction.torques.iter())
        {
            *command = (*command + *correction).clamp(-1.0, 1.0);
        }
        let floating_base_snapshot = terrain.floating_base_dynamics_snapshot();
        let (floating_base_command, floating_base_report) = match floating_base_snapshot.as_ref() {
            Some(snapshot) => self.floating_base_inverse_dynamics.allocate(
                state,
                contacts,
                &sparse_command,
                snapshot,
            ),
            None => (
                sparse_command.clone(),
                FloatingBaseInverseDynamicsReport::unavailable(),
            ),
        };
        let dynamics_snapshot = terrain.dynamics_snapshot(state, contacts);
        let (out, contact_dynamics_report) = if floating_base_report.converged {
            // The floating-base solve already enforces all generalized
            // equations and stance constraints. Re-running the reduced joint
            // model would discard those guarantees.
            (
                floating_base_command,
                crate::contact_inverse_dynamics::ContactInverseDynamicsReport {
                    converged: true,
                    used_fallback: false,
                    active_set_iterations: floating_base_report.active_set_iterations,
                    active_bound_count: 0,
                    maximum_dynamics_residual_nm: floating_base_report.maximum_dynamics_residual,
                    maximum_contact_acceleration_residual: floating_base_report
                        .maximum_contact_acceleration_residual,
                    maximum_bound_violation: 0.0,
                    maximum_friction_utilization: floating_base_report.maximum_friction_utilization,
                    objective: floating_base_report.objective,
                    budget: floating_base_report.budget,
                },
            )
        } else {
            match dynamics_snapshot.as_ref() {
                Some(snapshot) => self.contact_inverse_dynamics.allocate_with_snapshot(
                    state,
                    contacts,
                    &floating_base_command,
                    snapshot,
                ),
                None => {
                    self.contact_inverse_dynamics
                        .allocate(state, contacts, &floating_base_command)
                }
            }
        };
        let whole_body_report = inverse_dynamics_report.seed_report;
        (
            out,
            HierarchicalControlReport {
                support_phase: contact.phase,
                residual_authority,
                balance_effort,
                baseline_weight,
                recovery_mode: recovery_report.mode,
                capture_margin_m: recovery_report.support_margin_m,
                recovery_effort: recovery_report.effort,
                planned_step,
                planned_swing,
                terrain_confidence: planned_swing
                    .map(|trajectory| trajectory.terrain_confidence)
                    .unwrap_or(1.0),
                terrain_clearance_m: planned_swing
                    .map(|trajectory| trajectory.clearance_m)
                    .unwrap_or(0.0),
                whole_body_active_constraints: whole_body_report.active_constraints,
                whole_body_joint_utilization: whole_body_report.maximum_joint_utilization,
                whole_body_objective_residual: whole_body_report.objective_residual,
                whole_body_feasible: whole_body_report.feasible
                    && (inverse_dynamics_report.converged || inverse_dynamics_report.used_fallback)
                    && (contact_dynamics_report.converged || contact_dynamics_report.used_fallback),
                inverse_dynamics_iterations: inverse_dynamics_report.solver_iterations,
                inverse_dynamics_max_violation: inverse_dynamics_report
                    .maximum_constraint_violation,
                inverse_dynamics_fallback: inverse_dynamics_report.used_fallback,
                terrain_horizon_steps: terrain_mpc_plan
                    .as_ref()
                    .map(|plan| plan.footsteps.len())
                    .unwrap_or(0),
                terrain_mpc_cost: terrain_mpc_plan
                    .as_ref()
                    .map(|plan| plan.total_cost)
                    .unwrap_or(0.0),
                terrain_mpc_candidates: terrain_mpc_plan
                    .as_ref()
                    .map(|plan| plan.evaluated_candidates)
                    .unwrap_or(0),
                contact_dynamics_converged: contact_dynamics_report.converged,
                contact_dynamics_fallback: contact_dynamics_report.used_fallback,
                contact_dynamics_residual_nm: contact_dynamics_report.maximum_dynamics_residual_nm,
                contact_acceleration_residual: contact_dynamics_report
                    .maximum_contact_acceleration_residual,
                contact_friction_utilization: contact_dynamics_report.maximum_friction_utilization,
                contact_solver_budget_missed: contact_dynamics_report.budget.deadline_missed,
                contact_solver_elapsed_us: contact_dynamics_report.budget.elapsed_micros,
                terrain_max_height_std_m: terrain_mpc_plan
                    .as_ref()
                    .map(|plan| plan.maximum_height_std_m)
                    .unwrap_or(0.0),
                terrain_max_evidence_age_s: terrain_mpc_plan
                    .as_ref()
                    .map(|plan| plan.maximum_evidence_age_s)
                    .unwrap_or(0.0),
                contact_trust,
                centroidal_model_valid: centroidal_report.valid_model,
                centroidal_authority: centroidal_report.authority,
                centroidal_correction_norm: centroidal_report.correction_norm,
                angular_momentum_norm: vector_norm3(centroidal_report.current_angular_momentum),
                linear_momentum_norm: vector_norm3(centroidal_report.current_linear_momentum),
                floating_base_model_available: floating_base_snapshot.is_some(),
                floating_base_dynamics_converged: floating_base_report.converged,
                floating_base_dynamics_fallback: floating_base_report.used_fallback,
                floating_base_dynamics_residual: floating_base_report.maximum_dynamics_residual,
                floating_base_solver_elapsed_us: floating_base_report.budget.elapsed_micros,
                floating_base_solver_budget_missed: floating_base_report.budget.deadline_missed,
                floating_base_warm_start_used: floating_base_report.warm_start_used,
                floating_base_warm_start_active_bounds: floating_base_report
                    .warm_start_active_bounds,
                floating_base_symbolic_pattern_reused: floating_base_report.symbolic_pattern_reused,
                floating_base_solver_backend_id: floating_base_report.solver_backend_id,
                floating_base_model_id: floating_base_report.model_id,
            },
        )
    }

    fn apply_balance_correction(
        &self,
        command: &mut HumanoidCommand,
        state: &HumanoidState,
        support: SupportPhase,
    ) -> f32 {
        let names = self.morphology.joint_names();
        let roll_error = state.torso_vertical[0];
        let pitch_error = state.torso_vertical[1];
        let roll_rate = state.root_angular_velocity[0];
        let pitch_rate = state.root_angular_velocity[1];
        let roll = -(self.config.roll_balance_gain * roll_error
            + self.config.angular_damping_gain * roll_rate) as f32;
        let pitch = -(self.config.pitch_balance_gain * pitch_error
            + self.config.angular_damping_gain * pitch_rate) as f32;

        let mut effort = 0.0f32;
        for (index, name) in names.iter().enumerate() {
            let correction = if name == "abdomen_x" {
                roll
            } else if name == "abdomen_y" {
                pitch
            } else if name == "right_hip_x" || name == "right_ankle_x" {
                if matches!(
                    support,
                    SupportPhase::RightStance | SupportPhase::DoubleSupport
                ) {
                    0.35 * roll
                } else {
                    0.0
                }
            } else if name == "left_hip_x" || name == "left_ankle_x" {
                if matches!(
                    support,
                    SupportPhase::LeftStance | SupportPhase::DoubleSupport
                ) {
                    0.35 * roll
                } else {
                    0.0
                }
            } else if name == "right_hip_y" || name == "right_ankle_y" {
                if matches!(
                    support,
                    SupportPhase::RightStance | SupportPhase::DoubleSupport
                ) {
                    0.25 * pitch
                } else {
                    0.0
                }
            } else if name == "left_hip_y" || name == "left_ankle_y" {
                if matches!(
                    support,
                    SupportPhase::LeftStance | SupportPhase::DoubleSupport
                ) {
                    0.25 * pitch
                } else {
                    0.0
                }
            } else {
                0.0
            };
            if correction != 0.0 {
                command.torques[index] = (command.torques[index] + correction).clamp(-1.0, 1.0);
                effort += correction.abs();
            }
        }
        effort / self.morphology.num_actuators().max(1) as f32
    }
}

fn vector_norm3(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn swing_foot_position(state: &HumanoidState, foot: crate::footstep::FootSide) -> [f64; 3] {
    let offset = match foot {
        crate::footstep::FootSide::Right => 6,
        crate::footstep::FootSide::Left => 9,
    };
    if state.extremities.len() >= offset + 3 {
        [
            state.extremities[offset],
            state.extremities[offset + 1],
            state.extremities[offset + 2],
        ]
    } else {
        [state.root_position[0], state.root_position[1], 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_double_support_for_standing_state() {
        let state = HumanoidState::standing();
        let contact = ContactEstimate::from_state(&state, 0.04);
        assert_eq!(contact.phase, SupportPhase::DoubleSupport);
    }

    #[test]
    fn residual_authority_collapses_when_fallen() {
        let mut state = HumanoidState::standing();
        state.torso_vertical = [1.0, 0.0, 0.0];
        let controller = HierarchicalHumanoidController::new(HumanoidMorphology::Dmc21);
        let baseline = HumanoidCommand::zero();
        let residual = HumanoidCommand::from_raw(&vec![1.0; 21]);
        let (_, report) =
            controller.synthesize(HumanoidTask::Stand, &state, &baseline, &residual, 1.0, 0.0);
        assert_eq!(report.residual_authority, 0.0);
    }

    #[test]
    fn output_stays_normalized() {
        let state = HumanoidState::standing();
        let controller = HierarchicalHumanoidController::new(HumanoidMorphology::Dmc21);
        let baseline = HumanoidCommand::from_raw(&vec![1.0; 21]);
        let residual = HumanoidCommand::from_raw(&vec![1.0; 21]);
        let (command, _) =
            controller.synthesize(HumanoidTask::Walk, &state, &baseline, &residual, 1.0, 0.0);
        assert!(
            command
                .torques
                .iter()
                .all(|value| (-1.0..=1.0).contains(value))
        );
    }
    #[test]
    fn contact_dynamics_layer_is_reported() {
        let state = HumanoidState::standing();
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let controller = HierarchicalHumanoidController::new(HumanoidMorphology::Dmc21);
        let zero = HumanoidCommand::zero();
        let (_, report) = controller.synthesize_with_contacts(
            HumanoidTask::Stand,
            &state,
            &contacts,
            &zero,
            &zero,
            1.0,
            0.0,
        );
        assert!(report.contact_dynamics_converged || report.contact_dynamics_fallback);
        assert!(
            report.contact_dynamics_residual_nm.is_finite() || report.contact_dynamics_fallback
        );
    }

    #[test]
    fn environment_path_shapes_recovery_swing() {
        let mut state = HumanoidState::standing();
        state.com_velocity[0] = 0.8;
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let terrain = crate::terrain::HeightFieldTerrain {
            origin_world_m: [-0.5, -0.5],
            cell_size_m: 0.1,
            width: 11,
            height: 11,
            heights_m: (0..121)
                .map(|index| {
                    if index / 11 == 5 && index % 11 == 6 {
                        0.08
                    } else {
                        0.0
                    }
                })
                .collect(),
            friction: 0.8,
            compliance: 0.0,
        };
        let controller = HierarchicalHumanoidController::new(HumanoidMorphology::Dmc21);
        let baseline = HumanoidCommand::zero();
        let residual = HumanoidCommand::zero();
        let (_, report) = controller.synthesize_with_environment(
            HumanoidTask::Walk,
            &state,
            &contacts,
            &terrain,
            &baseline,
            &residual,
            1.0,
            0.0,
        );
        if let Some(swing) = report.planned_swing {
            assert!(swing.clearance_m >= 0.055);
            assert!(swing.terrain_confidence > 0.0);
        }
    }
}
