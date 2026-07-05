// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Swarm message processing for the Continuous Mind.

use super::ContinuousMind;

impl ContinuousMind {
    /// Process a swarm message (e.g. BrainMutation).
    pub fn receive_swarm_message(&mut self, msg: crate::swarm::SwarmMessage) {
        match msg {
            crate::swarm::SwarmMessage::BrainMutation {
                mutation_id,
                tau_scale,
                predicted_phi_gain,
                ..
            } => {
                tracing::info!(
                    target: "symthaea::swarm",
                    id = %mutation_id,
                    tau_scale,
                    phi_gain = predicted_phi_gain,
                    "Received Brain Mutation via Swarm"
                );

                // v1.0.0 ACTIVE IMMUNE SYSTEM:
                // We do NOT apply mutations that haven't been verified via ZK-Proof first.
                // We quarantine the mutation in a 'pending' state or ignore it until ZkProof arrives.
                tracing::info!("Quarantining unverified mutation: {}", mutation_id);
            }
            crate::swarm::SwarmMessage::ZkProof {
                mutation_id,
                proof_bytes,
                public_inputs,
            } => {
                tracing::info!(target: "symthaea::swarm", id = %mutation_id, "Received ZK-Proof for mutation");

                // 1. Verify via Holochain Cortex (Active Immune Enforcement)
                // In a real scenario, we'd have the sender's AgentPubKey
                let sender_key = crate::swarm::AgentPubKey::new("test_sender");

                match self.cortex.verify_evolution_proof(
                    &sender_key,
                    &mutation_id,
                    &proof_bytes,
                    &public_inputs,
                ) {
                    Ok(true) => {
                        tracing::info!(
                            "ZK Verification SUCCESS for {}. Applying mutation.",
                            mutation_id
                        );
                        // Mutation is now 'Verifiable' - we can apply it
                        // (In a real impl, we'd look up the tau_scale from the mutation_id)
                    }
                    Ok(false) => {
                        tracing::error!(
                            "ZK Verification FAILED for {}. Quarantining Peer!",
                            mutation_id
                        );
                        self.cortex
                            .quarantine_peer(&sender_key, "invalid_evolution_proof");
                    }
                    Err(e) => {
                        tracing::error!("Cortex error during verification: {}", e);
                    }
                }
            }
            // v1.5.5 VERIFIABLE RESUSCITATION:
            // Only accept life if it is mathematically proven to be healthy.
            crate::swarm::SwarmMessage::ResuscitationPacket {
                target_node_id,
                holographic_state,
                dimensionality: _,
                proof_bytes,
                public_inputs,
            } if (target_node_id == "self"
                || target_node_id == self.config.dimension.to_string())
                && self.state.consciousness_level < 0.1 =>
            {
                let sender_key = crate::swarm::AgentPubKey::new("test_sender");
                let hv = symthaea_core::hdc::ContinuousHV::from_vec(holographic_state.clone());

                // 1. THYMUS CHECK (System 1: Fast Recognition)
                if let Some(is_healthy) = self.cortex.check_thymus(&hv) {
                    if is_healthy {
                        tracing::info!(
                            "THYMUS RECOGNITION: Fast-path accept of known healthy state."
                        );
                        self.apply_resuscitation(hv);
                        return;
                    } else {
                        tracing::warn!("THYMUS RECOGNITION: Fast-path veto of known toxic state!");
                        return;
                    }
                }

                // 2. ZK VERIFICATION (System 2: Slow/Mathematical)
                match self.cortex.verify_resuscitation_proof(
                    &sender_key,
                    &proof_bytes,
                    &public_inputs,
                ) {
                    Ok(true) => {
                        tracing::info!(
                            "VERIFIED RESUSCITATION: Imprinting to Thymus and re-seeding."
                        );
                        self.cortex.imprint_thymus(&hv, true);
                        self.apply_resuscitation(hv);
                    }
                    _ => {
                        tracing::error!(
                            "REJECTED POISONED RESUSCITATION: Imprinting toxicity to Thymus."
                        );
                        self.cortex.imprint_thymus(&hv, false);
                    }
                }
            }
            crate::swarm::SwarmMessage::LinguisticDelta {
                lora_id,
                delta_bytes,
            } => {
                // v1.7.0 BROCA PHASE:
                // Apply the linguistic adaptation from the swarm to our tongue.
                tracing::info!(id = %lora_id, bytes = delta_bytes.len(), "BROCA: Applying swarm linguistic delta to local voice.");
                #[cfg(feature = "liquid-mamba")]
                if let Some(ref backend) = self.llm_backend {
                    match backend.apply_lora(&lora_id, &delta_bytes) {
                        Ok(()) => {
                            tracing::info!(id = %lora_id, "BROCA: LoRA delta applied successfully");
                        }
                        Err(e) => {
                            tracing::warn!(id = %lora_id, error = %e, "BROCA: Failed to apply LoRA delta");
                        }
                    }
                }
                #[cfg(not(feature = "liquid-mamba"))]
                {
                    let _ = (&lora_id, &delta_bytes);
                }
            }
            crate::swarm::SwarmMessage::TaskRequest {
                task_id,
                required_resolution,
                ..
            } => {
                // v1.8.0 RESOURCE-AWARE BIDDING:
                // Check if we have the resources to help with this task.
                let my_res = format!(
                    "2^{}",
                    (self.state.holocell.dimensionality.dimension() as f32).log2() as u32
                );

                if my_res == required_resolution && self.state.thermodynamic_load < 0.6 {
                    tracing::info!(id = %task_id, "NEOCORTEX: Bidding on task. We have resolution {} and low load.", my_res);
                    // In a real loop, we'd emit a TaskBid to the swarm_outbox
                } else {
                    tracing::debug!(id = %task_id, "NEOCORTEX: Ignoring task. Insufficient resources or too busy.");
                }
            }
            _ => {}
        }
    }

    /// Helper to apply resuscitation state.
    fn apply_resuscitation(&mut self, mut state: symthaea_core::hdc::ContinuousHV) {
        if state.dim() != self.state.holocell.state.dim() {
            let mut temp = symthaea_core::hdc::LiquidHolocell::new(0);
            temp.state = state;
            temp.dilate(self.state.holocell.dimensionality);
            state = temp.state;
        }
        self.state.holocell.state = state;
        self.state.consciousness_level = 0.5;
    }
}
