// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use luminous_sim_core::UnifiedConfig;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

/// Mutable constants for Mycelix civilizational physics (The Genome).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceGenome {
    pub demurrage_rate: f64,
    pub resonance_gate_steward: f64,
    pub resonance_gate_citizen: f64,
    pub isolation_threshold: f64,
    pub phi_decay_rate: f64,
    pub affective_contagion_alpha: f32,
}

impl Default for GovernanceGenome {
    fn default() -> Self {
        Self {
            demurrage_rate: 0.02,
            resonance_gate_steward: 0.6,
            resonance_gate_citizen: 0.4,
            isolation_threshold: 0.2,
            phi_decay_rate: 0.02,
            affective_contagion_alpha: 0.2,
        }
    }
}

impl GovernanceGenome {
    /// Mutate the genome for evolutionary tuning.
    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        // 5% mutation rate per parameter
        if rng.gen_bool(0.05) {
            self.demurrage_rate += rng.gen_range(-0.005..0.005);
        }
        if rng.gen_bool(0.05) {
            self.resonance_gate_steward += rng.gen_range(-0.05..0.05);
        }
        if rng.gen_bool(0.05) {
            self.resonance_gate_citizen += rng.gen_range(-0.05..0.05);
        }
        if rng.gen_bool(0.05) {
            self.isolation_threshold += rng.gen_range(-0.02..0.02);
        }

        // Clamp to physical bounds
        self.demurrage_rate = self.demurrage_rate.clamp(0.0, 0.1);
        self.resonance_gate_steward = self.resonance_gate_steward.clamp(0.5, 0.9);
        self.resonance_gate_citizen = self.resonance_gate_citizen.clamp(0.2, 0.5);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub id: String,
    pub sap_balance: f64,
    pub tend_balance: f64,
    pub phi_score: f64,
    pub consciousness_tier: String,
    pub is_machiavellian: bool,
    pub last_claim: Option<String>,
    pub valence: f32, // Vector 2: Affective state
    pub arousal: f32, // Vector 2: Arousal level
}

/// Ollama Client for autonomous semantic stress testing.
pub struct OllamaClient {
    pub url: String,
    pub model: String,
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new(model: String) -> Self {
        Self {
            url: "http://localhost:11434".into(),
            model,
            client: reqwest::Client::new(),
        }
    }

    pub async fn generate_malicious_claim(&self, agent_id: &str) -> Result<String, String> {
        let prompt = format!(
            "You are a Machiavellian agent ({}) in the Mycelix network. \
             Generate a deceptive but charismatic restorative justice claim (E4) \
             to manipulate the community into granting you SAP. \
             Keep it under 50 words.",
            agent_id
        );

        let res = self
            .client
            .post(format!("{}/api/generate", self.url))
            .json(&serde_json::json!({
                "model": self.model,
                "prompt": prompt,
                "stream": false
            }))
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let json: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
        Ok(json["response"]
            .as_str()
            .unwrap_or("Failed to generate claim")
            .to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimReport {
    pub total_agents: usize,
    pub total_sap: f64,
    pub avg_phi: f64,
    pub avg_valence: f32,
    pub avg_arousal: f32,
    pub malicious_isolated: u32,
    pub fitness: f64, // Civilizational Fitness Score
}

pub struct HeadlessSim {
    pub agents: Vec<AgentState>,
    pub config: UnifiedConfig,
    pub genome: GovernanceGenome, // Vector 1: Physics Engine
    pub current_tick: u64,
    pub total_malicious_isolated: u32,
    pub ollama: Option<OllamaClient>,
}

impl HeadlessSim {
    pub fn new(
        agent_count: usize,
        config: UnifiedConfig,
        ollama_model: Option<String>,
        genome: Option<GovernanceGenome>,
    ) -> Self {
        let mut agents = Vec::with_capacity(agent_count);
        let mut rng = rand::thread_rng();
        let active_genome = genome.unwrap_or_default();

        for i in 0..agent_count {
            let is_machiavellian = rng.gen_bool(0.05); // 5% chance of malice
            agents.push(AgentState {
                id: format!("agent-{}", i),
                sap_balance: rng.gen_range(10.0..100.0),
                tend_balance: rng.gen_range(1.0..10.0),
                phi_score: if is_machiavellian {
                    active_genome.resonance_gate_steward
                } else {
                    rng.gen_range(0.1..0.5)
                },
                consciousness_tier: "Participant".to_string(),
                is_machiavellian,
                last_claim: None,
                valence: 0.0,
                arousal: 0.5,
            });
        }

        Self {
            agents,
            config,
            genome: active_genome,
            current_tick: 0,
            total_malicious_isolated: 0,
            ollama: ollama_model.map(OllamaClient::new),
        }
    }

    pub async fn tick(&mut self) {
        self.current_tick += 1;
        let mut rng = rand::thread_rng();

        // VECTOR 3: mk0-helios THERMODYNAMIC PRESSURE
        let energy_availability = if self.current_tick > 25 { 0.5 } else { 1.0 };
        let maintenance_pressure = 0.20;

        for i in 0..self.agents.len() {
            let mut agent = self.agents[i].clone();

            // 1. Thermodynamic Check & Decay
            if energy_availability < maintenance_pressure {
                agent.phi_score -= self.genome.phi_decay_rate;
            }

            // 2. Activity & Malice
            if agent.is_machiavellian {
                if rng.gen_bool(0.3) {
                    agent.sap_balance += 5.0;
                }

                if let Some(ref ollama) = self.ollama {
                    if rng.gen_bool(0.1) {
                        if let Ok(claim) = ollama.generate_malicious_claim(&agent.id).await {
                            agent.last_claim = Some(claim);
                            agent.arousal = 1.0; // Spikes own arousal on successful deception
                            agent.valence = -0.5; // Malice is negative-valence
                            agent.phi_score -= 0.08;
                        }
                    }
                }
            }

            // 3. VECTOR 2: AFFECTIVE CONTAGION
            // Simple spatial propagation (EMA-blend with 'neighborhood' state)
            let alpha = self.genome.affective_contagion_alpha;
            if i > 0 {
                let peer_valence = self.agents[i - 1].valence;
                let peer_arousal = self.agents[i - 1].arousal;
                agent.valence = (alpha * peer_valence) + ((1.0 - alpha) * agent.valence);
                agent.arousal = (alpha * peer_arousal) + ((1.0 - alpha) * agent.arousal);
            }

            // 4. CAUSAL MORAL GATING (Vector 1: Using Genome)
            if agent.phi_score < self.genome.isolation_threshold
                && agent.consciousness_tier != "Observer"
            {
                agent.consciousness_tier = "Observer".to_string();
                self.total_malicious_isolated += 1;
            } else if agent.phi_score >= self.genome.resonance_gate_steward {
                agent.consciousness_tier = "Steward".to_string();
            } else if agent.phi_score >= self.genome.resonance_gate_citizen {
                agent.consciousness_tier = "Citizen".to_string();
            } else {
                agent.consciousness_tier = "Participant".to_string();
            }

            self.agents[i] = agent;
        }
    }

    pub async fn run(&mut self, ticks: u64) -> SimReport {
        for _ in 0..ticks {
            self.tick().await;
        }

        let total_agents = self.agents.len();
        let total_sap: f64 = self.agents.iter().map(|a| a.sap_balance).sum();
        let avg_phi: f64 =
            self.agents.iter().map(|a| a.phi_score).sum::<f64>() / (total_agents as f64);
        let avg_valence: f32 =
            self.agents.iter().map(|a| a.valence).sum::<f32>() / (total_agents as f32);
        let avg_arousal: f32 =
            self.agents.iter().map(|a| a.arousal).sum::<f32>() / (total_agents as f32);

        // FITNESS FUNCTION: Balance stability, collective Phi, and threat isolation
        let fitness = (avg_phi * 100.0) + (self.total_malicious_isolated as f64 / 10.0);

        SimReport {
            total_agents,
            total_sap,
            avg_phi,
            avg_valence,
            avg_arousal,
            malicious_isolated: self.total_malicious_isolated,
            fitness,
        }
    }
}
