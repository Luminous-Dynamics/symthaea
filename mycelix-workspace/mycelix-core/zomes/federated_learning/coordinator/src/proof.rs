// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mandatory gradient proof enforcement for trustless FL.
//!
//! Gradients must include a valid zkSTARK proof before acceptance.

use hdk::prelude::*;
use federated_learning_integrity::*;

use crate::signals::Signal;
use crate::matl::{GradientWithPoGQInput, PoGQResult, submit_gradient_with_pogq};

/// Submit gradient with mandatory proof (trustless path)
///
/// This is the REQUIRED submission path for trustless FL. The gradient
/// must include a valid zkSTARK proof. The proof is verified and linked
/// before the gradient is accepted.
#[hdk_extern]
pub fn submit_gradient_with_proof(input: GradientWithPoGQInput) -> ExternResult<PoGQResult> {
    // Verify that a proof hash is provided in the gradient hash field
    // The gradient_hash should reference an existing GradientProof entry
    let gradient_hash = &input.gradient_hash;

    // Validate gradient hash format
    if gradient_hash.len() != 64 || !gradient_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Gradient hash must be 64 hex characters".to_string()
        )));
    }

    // Check if a GradientProof exists for this hash
    let proof_path = Path::from(format!("gradient_proof.{}", gradient_hash));
    let proof_exists = match proof_path.clone().typed(LinkTypes::GradientToProof) {
        Ok(typed) => typed.exists()?,
        Err(_) => false,
    };

    if !proof_exists {
        // Also check via RoundToProofs link
        let round_proof_path = Path::from(format!("round_proofs.{}", input.round));
        let has_proof = match round_proof_path.clone().typed(LinkTypes::RoundToProofs) {
            Ok(typed) => {
                if !typed.exists()? {
                    false
                } else {
                    let proof_hash = typed.path_entry_hash()?;
                    let links = get_links(
                        LinkQuery::new(
                            proof_hash,
                            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToProofs as u8).into()),
                        ),
                        GetStrategy::default(),
                    )?;
                    // Check if any proof matches this gradient hash
                    let mut found = false;
                    for link in links {
                        if let Some(action_hash) = link.target.into_action_hash() {
                            if let Some(record) = get(action_hash, GetOptions::default())? {
                                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                                    // Try to deserialize as GradientProof via serde
                                    if let Ok(proof) = serde_json::from_slice::<GradientProof>(bytes.bytes()) {
                                        // Compare gradient_hash strings directly
                                        if proof.gradient_hash == *gradient_hash {
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    found
                }
            }
            Err(_) => false,
        };

        if !has_proof {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Mandatory: gradient proof required. Submit a GradientProof first via the proof system.".to_string()
            )));
        }
    }

    // Emit gossip signal: gradient is ready for this round
    let agent = agent_info()?.agent_initial_pubkey;
    emit_signal(Signal::GradientReady {
        node_id: agent.to_string(),
        round: input.round as u64,
        gradient_hash: gradient_hash.to_string(),
        source: Some(agent.to_string()),
        signature: None,
    })?;

    // Delegate to existing PoGQ submission (proof is verified above)
    submit_gradient_with_pogq(input)
}
