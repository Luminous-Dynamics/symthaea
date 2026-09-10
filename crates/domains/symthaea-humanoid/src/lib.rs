// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

pub mod actuation;
pub mod actuation_capability;
pub mod actuator_controllability;
// Internal compatibility layer. Public motor authority uses the v2 verified
// source boundary below, which binds scheme identity and the full verification
// decision into the runtime evidence identity.
mod authority_source_attestation;
pub mod verified_authority_source;
pub mod authority_source_signing;
pub mod authority_evidence_artifacts;
pub mod verified_authority_artifacts;
pub mod authority_evidence_policy;
pub mod cognitive_authority_evidence;
pub mod live_authority_kernel;
pub mod grasp_contact_evidence;
pub mod grasp_retention_evidence;
pub mod grasp_acquisition_intent;
pub mod grasp_controller_qualification;
pub mod grasp_measurement_coverage;
pub mod capability_envelope;
pub mod capability_request;
pub mod cartesian_hand_reference;
pub mod centroidal;
pub mod contact;
pub mod contact_inverse_dynamics;
pub mod contact_site;
pub mod control;
pub mod control_budget;
pub mod control_capability;
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
pub mod evidence_digest;
pub mod evolution;
pub mod execution;
pub mod execution_authority_scope;
pub mod external_solver_certification;
pub mod fall_protection;
pub mod fep_agent;
pub mod floating_base;
pub mod floating_base_inverse_dynamics;
pub mod footstep;
pub mod frozen_dynamics;
pub mod full_dynamics;
pub mod gait;
pub mod guarded_skill_executive;
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
pub mod payload_capability;
pub mod physical_health;
pub mod plugin;
pub mod qp_certification;
pub mod qualification;
// Internal v1 attested facade. Public authority uses the v2 source verifier.
mod reach_attested_authority;
// Internal verified-source facade. The exported operational path additionally
// precommits accepted evidence policies and verifier trust roots.
mod reach_verified_authority;
pub mod reach_runtime_authority;
pub mod reach_authority_commitment;
pub mod reach_authority_committed_evidence;
// Internal implementation detail: authority-committed stages are necessary, but
// public operational promotion additionally requires collision-resistant evidence
// diversity for spatial goals and authority decisions.
mod reach_authority_committed_promotion;
// Internal implementation detail: protocol/corpus SHA-256 identity is necessary,
// but public operational promotion must additionally bind finalized authority
// commitments for every contributing trial and episode step.
mod reach_cryptographic_authority;
pub mod reach_episode_evidence;
// Internal implementation detail: episode-complete evidence is necessary, but
// operational promotion must additionally bind the exact qualification protocols.
mod reach_episode_promotion;
pub mod reach_execution;
pub mod reach_execution_evidence;
// Internal implementation detail: manifest-aware motor authority is necessary,
// but the exported path additionally requires authenticated source verification
// and runtime evidence-policy precommitment.
mod reach_manifest_authority;
// Internal implementation detail: manifest-bound qualification/promotion is
// necessary, but the actual motor receipt must record the strongest identities.
mod reach_manifest_bound_promotion;
// Internal implementation detail: operator/policy binding remains as deterministic
// defense-in-depth; the public authority path requires SHA-256 commitments too.
mod reach_operational_authority;
// Internal implementation detail: protocol-bound promotion is necessary, but the
// public operational authority path requires cryptographic protocol/corpus identity.
mod reach_operational_promotion;
pub mod reach_outcome_evidence;
pub mod reach_perturbation_manifest;
pub mod reach_policy_identity;
pub mod reach_qualification_campaign;
pub mod reach_qualification_lineage;
// Internal implementation detail: step-level qualification is necessary evidence,
// but it is not sufficient to expose an operational Reach capability on its own.
mod reach_qualification_promotion;
pub mod reach_spatial_goal_commitment;
// Internal implementation detail: SHA-256 diversity is mandatory and is consumed
// by the manifest-bound promotion layer rather than exposed as an alternate path.
mod reach_strong_diversity_promotion;
pub mod recovery;
pub mod recovery_benchmark;
pub mod recovery_certification;
pub mod release_pipeline;
pub mod reproducible_oracle_build;
pub mod reward;
pub mod safety;
pub mod simulator;
pub mod skill_actuation_guard;
pub mod skill_authority_receipt;
pub mod skill_executive;
pub mod skill_permit;
pub mod skill_runtime;
pub mod sparse_qp_backend;
pub mod spatial_goal;
mod spatial_goal_debug;
pub mod spatial_goal_identity;
pub mod state_estimation;
pub mod state_uncertainty;
pub mod terrain;
pub mod terrain_capability;
pub mod terrain_mpc;
pub mod training;
pub mod typed_actuation_capability;
pub mod types;
pub mod vision_terrain;
pub mod whole_body;
pub mod whole_body_intent;
pub mod whole_body_lowering;
pub use actuation::*;
pub use actuation_capability::*;
pub use actuator_controllability::*;
pub use verified_authority_source::*;
pub use authority_source_signing::*;
pub use authority_evidence_artifacts::*;
pub use verified_authority_artifacts::*;
pub use authority_evidence_policy::*;
pub use cognitive_authority_evidence::*;
pub use live_authority_kernel::*;
pub use grasp_contact_evidence::*;
pub use grasp_retention_evidence::*;
pub use grasp_acquisition_intent::*;
pub use grasp_controller_qualification::*;
pub use grasp_measurement_coverage::*;
pub use capability_envelope::*;
pub use capability_request::*;
pub use cartesian_hand_reference::*;
pub use centroidal::*;
pub use contact_inverse_dynamics::*;
pub use contact_site::*;
pub use control_budget::*;
pub use control_capability::*;
pub use control_release::*;
pub use controller::*;
pub use csc_qp::*;
pub use dynamics::*;
pub use dynamics_oracle::*;
pub use embodied_certification::*;
pub use encoder::*;
pub use equality_qp::*;
pub use evaluation::*;
pub use evidence_digest::*;
pub use execution::*;
pub use execution_authority_scope::*;
pub use external_solver_certification::*;
pub use fall_protection::*;
pub use floating_base::*;
pub use floating_base_inverse_dynamics::*;
pub use footstep::*;
pub use frozen_dynamics::*;
pub use full_dynamics::*;
pub use guarded_skill_executive::*;
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
pub use payload_capability::*;
pub use physical_health::*;
pub use qp_certification::*;
pub use qualification::*;
pub use reach_runtime_authority::*;
// These two public types occur in the runtime authority API even though their
// implementation module is intentionally private.
pub use reach_manifest_authority::{
    HumanoidReachManifestOperationalArtifact, HumanoidReachManifestOperationalPolicy,
};
pub use reach_authority_commitment::*;
pub use reach_authority_committed_evidence::*;
pub use reach_episode_evidence::*;
pub use reach_execution::*;
pub use reach_execution_evidence::*;
pub use reach_outcome_evidence::*;
pub use reach_perturbation_manifest::*;
pub use reach_policy_identity::*;
pub use reach_qualification_campaign::*;
pub use reach_qualification_lineage::*;
pub use reach_spatial_goal_commitment::*;
pub use recovery_certification::*;
pub use release_pipeline::*;
pub use reproducible_oracle_build::*;
pub use safety::*;
pub use simulator::*;
pub use skill_actuation_guard::*;
pub use skill_authority_receipt::*;
pub use skill_executive::*;
pub use skill_permit::*;
pub use skill_runtime::*;
pub use sparse_qp_backend::*;
pub use spatial_goal::*;
pub use spatial_goal_identity::*;
pub use state_estimation::*;
pub use state_uncertainty::*;
pub use terrain::*;
pub use terrain_capability::*;
pub use terrain_mpc::*;
pub use typed_actuation_capability::*;
pub use types::*;
pub use vision_terrain::*;
pub use whole_body::*;
pub use whole_body_intent::*;
pub use whole_body_lowering::*;

pub use crate::control::GaitControlProfile;
pub use control::{
    CfcCpg, GenerativePrior, HdcWorkspace, LocomotionModule, MachineState, execute_modular_gait,
};
pub use evolution::{GaitGenome, Rng};
