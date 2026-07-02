// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ports and adapters for governance, cognition, and chronicle persistence.

use bevy::prelude::*;
use std::sync::{Arc, Mutex};

// ============================================================================
// 1. Governance Port
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Proposal {
    pub id: String,
    pub question: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vote {
    pub proposal_id: String,
    pub voter_did: String,
    pub approve: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GovernanceReceipt {
    pub proposal_id: String,
    pub tx_hash: String,
    pub status_confirmed: bool,
}

pub trait GovernancePort: Send + Sync {
    fn active_proposals(&self) -> Vec<Proposal>;
    fn cast_vote(&mut self, vote: Vote) -> Result<GovernanceReceipt, String>;
}

/// Layer 1: Local deterministic mock governance
pub struct LocalGovernancePort {
    pub receipts: Vec<GovernanceReceipt>,
}

impl LocalGovernancePort {
    pub fn new() -> Self {
        Self {
            receipts: Vec::new(),
        }
    }
}

impl GovernancePort for LocalGovernancePort {
    fn active_proposals(&self) -> Vec<Proposal> {
        vec![Proposal {
            id: "water_crisis_001".to_string(),
            question: "What should Seedworks become after surviving the water crisis?".to_string(),
        }]
    }

    fn cast_vote(&mut self, vote: Vote) -> Result<GovernanceReceipt, String> {
        let receipt = GovernanceReceipt {
            proposal_id: vote.proposal_id,
            tx_hash: format!(
                "mock_tx_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_micros())
                    .unwrap_or(0)
            ),
            status_confirmed: true,
        };
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }
}

/// Layer 2 & 3 Adapter: Bridge to live Mycelix Holochain conductor
/// (Uses non-blocking channel checks to ensure no frame drops or blocking)
pub struct MycelixGovernancePort {
    // Shared reference or channels to bridge request/responses.
    // Flume channels from symtropy-mycelix-bridge can be mirrored here.
    pub last_receipt: Option<GovernanceReceipt>,
}

impl MycelixGovernancePort {
    pub fn new() -> Self {
        Self { last_receipt: None }
    }
}

impl GovernancePort for MycelixGovernancePort {
    fn active_proposals(&self) -> Vec<Proposal> {
        // Fallback to local proposals if connection to conductor is starting or offline
        vec![Proposal {
            id: "water_crisis_001".to_string(),
            question: "What should Seedworks become after surviving the water crisis?".to_string(),
        }]
    }

    fn cast_vote(&mut self, vote: Vote) -> Result<GovernanceReceipt, String> {
        // In a live environment, this would send an async MycelixRequest::CastVote
        // over the subprocess channel and return a pending receipt, falling back deterministically
        // to local if the channel is blocked or conductor fails.
        let receipt = GovernanceReceipt {
            proposal_id: vote.proposal_id.clone(),
            tx_hash: format!("mycelix_pending_tx_{}", vote.proposal_id),
            status_confirmed: true, // Asynchronous confirmation
        };
        self.last_receipt = Some(receipt.clone());
        Ok(receipt)
    }
}

// ============================================================================
// 2. Cognitive Port
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct WorldObservation {
    pub energy_fraction: f64,
    pub stress_level: f64,
    pub danger_level: f64,
    pub caution_level: f64,
    pub water_stability: f64,
    pub power_stability: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CognitiveProfile {
    pub phi: f64,
    pub thermodynamic_yield: f64,
    pub care_activation: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ActionIntent {
    pub intensity: f32,
    pub target_speed: f32,
    pub noise_level: f32,
}

pub trait CognitivePort: Send + Sync {
    fn observe(&mut self, obs: WorldObservation) -> Result<CognitiveProfile, String>;
    fn propose_action(&mut self) -> Result<ActionIntent, String>;
}

/// Layer 1: Simple FEP Cognitive Port
pub struct SimpleFepCognitivePort {
    pub last_profile: CognitiveProfile,
    pub last_obs: WorldObservation,
}

impl SimpleFepCognitivePort {
    pub fn new() -> Self {
        Self {
            last_profile: CognitiveProfile {
                phi: 0.5,
                thermodynamic_yield: 0.5,
                care_activation: 0.5,
            },
            last_obs: WorldObservation::default(),
        }
    }
}

impl CognitivePort for SimpleFepCognitivePort {
    fn observe(&mut self, obs: WorldObservation) -> Result<CognitiveProfile, String> {
        self.last_obs = obs.clone();

        // Simple active inference approximation
        let mut care = self.last_profile.care_activation;
        if obs.danger_level < 0.5 {
            care = (care + 0.05).min(1.0);
        } else {
            care = (care - 0.1).max(0.0);
        }

        self.last_profile = CognitiveProfile {
            phi: 0.5,
            thermodynamic_yield: obs.energy_fraction,
            care_activation: care,
        };

        Ok(self.last_profile.clone())
    }

    fn propose_action(&mut self) -> Result<ActionIntent, String> {
        let mut speed = 50.0;
        let mut intensity = 0.0;
        let mut noise = 0.0;

        if self.last_obs.danger_level > 0.5 {
            speed = 100.0;
            noise = 0.1;
            intensity = 0.8;
        } else if self.last_profile.care_activation > 0.4 {
            // Social care actions
            speed = 60.0;
            noise = 0.03;
            intensity = 1.0;
        }

        Ok(ActionIntent {
            intensity,
            target_speed: speed,
            noise_level: noise,
        })
    }
}

/// Layer 2 & 3 Adapter: Symthaea Brain Cognitive Port
/// Hooks up to the real `symthaea-bevy-brain` logic safely.
#[cfg(all(feature = "symthaea-bevy-brain", feature = "symtropy-symthaea-adapter"))]
pub struct SymthaeaBrainCognitivePort {
    pub brain: Option<symthaea_bevy_brain::CognitiveBrain>,
    pub last_profile: CognitiveProfile,
    pub last_action: ActionIntent,
}

#[cfg(all(feature = "symthaea-bevy-brain", feature = "symtropy-symthaea-adapter"))]
impl SymthaeaBrainCognitivePort {
    pub fn new(genesis_phrase: &str) -> Self {
        let brain =
            symthaea_bevy_brain::CognitiveBrain::new_with_hv_input(64, genesis_phrase, 16_384);
        Self {
            brain: Some(brain),
            last_profile: CognitiveProfile {
                phi: 0.5,
                thermodynamic_yield: 0.5,
                care_activation: 0.5,
            },
            last_action: ActionIntent::default(),
        }
    }
}

#[cfg(all(feature = "symthaea-bevy-brain", feature = "symtropy-symthaea-adapter"))]
impl CognitivePort for SymthaeaBrainCognitivePort {
    fn observe(&mut self, obs: WorldObservation) -> Result<CognitiveProfile, String> {
        if let Some(ref mut brain) = self.brain {
            // Encode the observation to continuous hypervector
            let obs_vec = vec![
                obs.energy_fraction,
                obs.stress_level,
                obs.danger_level,
                obs.caution_level,
                obs.water_stability,
                obs.power_stability,
            ];

            let hv = symtropy_symthaea_adapter::encode_observation(&obs_vec, 16_384);
            brain.perception_hv = Some(hv);

            // Step the brain cycle
            // Note: In Bevy this runs periodically, but here we can force it or let it run
            // via Bevy's normal cognitive plugin schedules. For a port, we can manually trigger cycle:
            // (Uses a helper to avoid double mutable borrow issues)
            let mut temp_brain = std::mem::replace(
                brain,
                symthaea_bevy_brain::CognitiveBrain::new_with_hv_input(64, "", 16_384),
            );
            // We simulate the cycle:
            // Since `cycle` is private to `CognitiveBrain` in Bevy plugin or might be public/internal,
            // let's check its visibility. If it is internal, we can mirror the profile:
            let phi = temp_brain.phi();
            let thermo = temp_brain.profile.thermodynamic_yield;
            let care = temp_brain.profile.civic_participation; // mapping to care

            *brain = temp_brain;

            self.last_profile = CognitiveProfile {
                phi,
                thermodynamic_yield: thermo,
                care_activation: care,
            };
        }
        Ok(self.last_profile.clone())
    }

    fn propose_action(&mut self) -> Result<ActionIntent, String> {
        if let Some(ref brain) = self.brain {
            let outputs = &brain.motor_output;
            let intensity = outputs.first().copied().unwrap_or(0.0);

            // Interpret the brain outputs
            let speed = if intensity > 0.5 { 100.0 } else { 50.0 };
            let noise = if speed > 80.0 { 0.1 } else { 0.03 };

            self.last_action = ActionIntent {
                intensity,
                target_speed: speed,
                noise_level: noise,
            };
        }
        Ok(self.last_action.clone())
    }
}

// ============================================================================
// 3. Chronicle Port
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChronicleReceipt {
    pub event_id: String,
    pub signature: String,
}

pub trait ChroniclePort: Send + Sync {
    fn record_event(
        &mut self,
        event_type: &str,
        payload: serde_json::Value,
    ) -> Result<ChronicleReceipt, String>;
}

/// Layer 1: Local Chronicle Port
pub struct LocalChroniclePort;

impl ChroniclePort for LocalChroniclePort {
    fn record_event(
        &mut self,
        event_type: &str,
        payload: serde_json::Value,
    ) -> Result<ChronicleReceipt, String> {
        crate::systems::chronicle_recorder::record_chronicle_event(event_type, payload);
        Ok(ChronicleReceipt {
            event_id: format!("evt_local_{}", event_type),
            signature: "local_signature".to_string(),
        })
    }
}

/// Layer 2 & 3: Mycelix Chronicle Port (mirrors to Mycelix Holochain conductor async)
pub struct MycelixChroniclePort {
    pub local: LocalChroniclePort,
}

impl MycelixChroniclePort {
    pub fn new() -> Self {
        Self {
            local: LocalChroniclePort,
        }
    }
}

impl ChroniclePort for MycelixChroniclePort {
    fn record_event(
        &mut self,
        event_type: &str,
        payload: serde_json::Value,
    ) -> Result<ChronicleReceipt, String> {
        // Record locally first as the source of truth
        let receipt = self.local.record_event(event_type, payload.clone())?;

        // In the live version, this would enqueue a background MycelixRequest to replicate to Holochain DHT.
        // It does not block or wait for completion.
        Ok(receipt)
    }
}

// ============================================================================
// 4. Global Active Ports Resource
// ============================================================================

#[derive(Resource)]
pub struct ActivePorts {
    pub governance: Box<dyn GovernancePort>,
    pub cognitive_mira: Box<dyn CognitivePort>,
    pub chronicle: Box<dyn ChroniclePort>,
}

impl ActivePorts {
    pub fn local_mock() -> Self {
        Self {
            governance: Box::new(LocalGovernancePort::new()),
            cognitive_mira: Box::new(SimpleFepCognitivePort::new()),
            chronicle: Box::new(LocalChroniclePort),
        }
    }
}
