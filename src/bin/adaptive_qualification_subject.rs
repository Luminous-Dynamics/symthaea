// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic executable for the RQ-006Z AdaptiveReasoner control-surface qualifier.
//!
//! This binary deliberately accepts no task target or reward. It emits one content-bound
//! qualification receipt from the exact checkout selected by the outer subject verifier.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use symthaea::consciousness::adaptive_reasoning::qualification::{
    AdaptiveQualificationConfig, ForcedAdaptiveAction, ADAPTIVE_QUALIFICATION_CONTROL_VERSION,
};
use symthaea::consciousness::adaptive_reasoning::AdaptiveReasoner;
use symthaea::consciousness::primitive_reasoning::TransformationType;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::PrimitiveTier;

const DEFAULT_OUTPUT: &str = "target/rq-006z/adaptive-qualification-receipt.json";
const QUALIFICATION_STEPS: usize = 5;
const POLICY_SEED: u64 = 0xC0FF_EE42;
const QUESTION_SEED: u64 = 0xA11C_E700;

fn main() -> Result<(), Box<dyn Error>> {
    let subject_revision = env::var("SYMTHAEA_SUBJECT_REVISION")
        .map_err(|_| "SYMTHAEA_SUBJECT_REVISION must contain the exact checked-out commit SHA")?;
    let output = env::var("SYMTHAEA_ADAPTIVE_RECEIPT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_OUTPUT));

    let reasoner = AdaptiveReasoner::new(PrimitiveTier::NSM);
    let mut config =
        AdaptiveQualificationConfig::frozen(QUALIFICATION_STEPS, POLICY_SEED, &subject_revision);
    config.forced_actions.push(ForcedAdaptiveAction {
        step: 2,
        primitive_name: "ground".to_string(),
        transformation: TransformationType::Bundle,
    });

    let run = reasoner.reason_adaptive_qualification_frozen(
        BinaryHV::random(QUESTION_SEED),
        &config,
    )?;

    if !run.receipt.commitment_valid() {
        return Err("adaptive qualification receipt failed its nested commitment validation".into());
    }
    if !run.receipt.learner_state_unchanged() {
        return Err("frozen adaptive qualifier changed learner state".into());
    }
    if run.receipt.subject_revision != subject_revision {
        return Err("emitted receipt subject revision differs from requested subject".into());
    }
    if run.receipt.control_version != ADAPTIVE_QUALIFICATION_CONTROL_VERSION {
        return Err("emitted receipt control version differs from compiled qualifier".into());
    }
    if run.receipt.actions.len() != QUALIFICATION_STEPS {
        return Err("adaptive qualifier did not consume the exact preregistered action budget".into());
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&run.receipt)?)?;

    println!("subject_revision={}", run.receipt.subject_revision);
    println!("control_version={}", run.receipt.control_version);
    println!("action_space_commitment={}", run.receipt.action_space_commitment);
    println!("forced_schedule_commitment={}", run.receipt.forced_schedule_commitment);
    println!("receipt_commitment={}", run.receipt.receipt_commitment);
    println!("receipt_path={}", output.display());

    Ok(())
}
