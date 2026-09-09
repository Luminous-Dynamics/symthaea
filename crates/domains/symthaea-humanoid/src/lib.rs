// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

pub mod actuation;
pub mod backend_action_evidence;
pub mod centroidal;
pub mod contact;
pub mod contact_inverse_dynamics;
pub mod continuous_predictor;
pub mod contextual_cfc;
pub mod control;
pub mod control_budget;
pub mod control_release;
pub mod controller;
pub mod csc_qp;
pub mod dynamics;
pub mod dynamics_oracle;
pub mod embodied_certification;
pub mod embodiment;
pub mod encoder;
pub mod equality_qp;
pub mod evaluation;
pub mod evolution;
pub mod external_solver_certification;
pub mod fall_protection;
pub mod fep_agent;
pub mod floating_base;
pub mod floating_base_inverse_dynamics;
pub mod footstep;
pub mod full_dynamics;
pub mod gait;
pub mod hardware_state;
pub mod hierarchical;
pub mod inprocess_solver_certification;
pub mod inprocess_sparse_qp;
pub mod inverse_dynamics;
pub mod morphology;
pub mod mujoco_oracle_protocol;
pub mod multi_contact;
pub mod oracle_dataset;
pub mod oracle_generator_certification;
#[cfg(feature = "osqp-backend")]
pub mod osqp_backend;
pub mod plugin;
pub mod predictive_baseline;
pub mod predictive_qualification;
pub mod predictor_context;
pub mod qp_certification;
pub mod recovery;
pub mod recovery_benchmark;
pub mod recovery_certification;
pub mod release_pipeline;
pub mod reproducible_oracle_build;
pub mod reward;
pub mod safety;
pub mod semantic_action;
pub mod semantic_action_causality;
pub mod semantic_state;
pub mod simulator;
pub mod sparse_qp_backend;
pub mod state_estimation;
pub mod terrain;
pub mod terrain_mpc;
pub mod training;
pub mod transition_evidence;
pub mod types;
pub mod vision_terrain;
pub mod whole_body;
pub use actuation::*;
pub use backend_action_evidence::*;
pub use centroidal::*;
pub use contact_inverse_dynamics::*;
pub use continuous_predictor::*;
pub use contextual_cfc::*;
pub use control_budget::*;
pub use control_release::*;
pub use controller::*;
pub use csc_qp::*;
pub use dynamics::*;
pub use dynamics_oracle::*;
pub use embodied_certification::*;
pub use encoder::*;
pub use equality_qp::*;
pub use evaluation::*;
pub use external_solver_certification::*;
pub use fall_protection::*;
pub use floating_base::*;
pub use floating_base_inverse_dynamics::*;
pub use footstep::*;
pub use full_dynamics::*;
pub use hardware_state::*;
pub use hierarchical::*;
pub use inprocess_solver_certification::*;
pub use inprocess_sparse_qp::*;
pub use inverse_dynamics::*;
pub use mujoco_oracle_protocol::*;
pub use multi_contact::*;
pub use oracle_dataset::*;
pub use oracle_generator_certification::*;
#[cfg(feature = "osqp-backend")]
pub use osqp_backend::*;
pub use predictive_baseline::*;
pub use predictive_qualification::*;
pub use predictor_context::*;
pub use qp_certification::*;
pub use recovery_certification::*;
pub use release_pipeline::*;
pub use reproducible_oracle_build::*;
pub use safety::*;
pub use semantic_action::*;
pub use semantic_action_causality::*;
pub use semantic_state::*;
pub use simulator::*;
pub use sparse_qp_backend::*;
pub use state_estimation::*;
pub use terrain::*;
pub use terrain_mpc::*;
pub use transition_evidence::*;
pub use types::*;
pub use vision_terrain::*;
pub use whole_body::*;

// The predicted-value enum contains only copyable scalar/reason payloads. Making
// that semantic explicit avoids moving from borrowed prediction records during
// deterministic digest validation.
impl Copy for HumanoidPredictedValueV1 {}

// R4.4 reports are self-validating, but promotion decisions must also bind the
// report back to the exact frozen experiment that produced it. Keeping this
// cross-artifact check at the crate boundary avoids making the report module
// depend on private fields of the experiment while preserving an explicit API.
impl HumanoidPredictiveQualificationReportV1 {
    pub fn validate_against_experiment_v1(
        &self,
        experiment: &HumanoidPredictiveExperimentV1,
    ) -> Result<(), HumanoidPredictiveQualificationErrorV1> {
        self.validate()?;
        experiment.validate()?;
        let descriptor = experiment.predictor().descriptor();
        if self.experiment_digest_hex != experiment.experiment_digest_hex()
            || self.predictor_descriptor_digest_hex != descriptor.descriptor_digest_hex()?
            || self.split_manifest_digest_hex != experiment.manifest().manifest_digest_hex
            || self.output_contract_digest_hex != descriptor.output_role_digest_hex
        {
            return Err(HumanoidPredictiveQualificationErrorV1::ReportSummaryMismatch);
        }
        Ok(())
    }
}

pub use crate::control::GaitControlProfile;
pub use control::{
    CfcCpg, GenerativePrior, HdcWorkspace, LocomotionModule, MachineState, execute_modular_gait,
};
pub use evolution::{GaitGenome, Rng};
