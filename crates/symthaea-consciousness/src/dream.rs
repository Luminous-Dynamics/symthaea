/*!
Counterfactual Dream Engine
===========================

Implements "Counterfactual Dreaming" - the ability to simulate alternative pasts
to learn from mistakes not made.

Algorithm:
1. Record high-surprise events (wake).
2. During low-load (sleep/idle), sample an event.
3. Generate N alternative actions.
4. Simulate outcomes using CfC World Model.
5. If alternative yields higher Φ than reality -> Consolidate as "Wisdom".
*/

use std::sync::{Arc, Mutex};
use symthaea_dynamics::world_model::HierarchicalCfCWorldModel;
use symthaea_math::RealHV;
use anyhow::Result;

pub struct DreamEngine {
    world_model: Arc<Mutex<HierarchicalCfCWorldModel>>,
    memory: Vec<DreamEvent>,
}

struct DreamEvent {
    state: Vec<f32>,
    action: Vec<f32>,
    actual_outcome: Vec<f32>,
    surprise: f32,
}

impl DreamEngine {
    pub fn new(model: Arc<Mutex<HierarchicalCfCWorldModel>>) -> Self {
        Self {
            world_model: model,
            memory: Vec::new(),
        }
    }

    /// Record a waking event for later dreaming
    pub fn record(&mut self, state: &[f32], action: &[f32], outcome: &[f32], surprise: f32) {
        if surprise > 0.1 { // Only dream about surprising things
            self.memory.push(DreamEvent {
                state: state.to_vec(),
                action: action.to_vec(),
                actual_outcome: outcome.to_vec(),
                surprise,
            });
        }
    }

    /// Run a dream cycle (Counterfactual Simulation)
    pub fn dream(&mut self) -> Result<usize> {
        if self.memory.is_empty() {
            return Ok(0);
        }

        let mut insights = 0;
        let mut model = self.world_model.lock().expect("lock poisoned");

        // Process a few events
        // (In real impl, use random sampling)
        let event_idx = 0; 
        if event_idx >= self.memory.len() { return Ok(0); }
        let event = &self.memory[event_idx];

        // 1. Evaluate actual Φ
        let actual_phi = model.estimate_phi(&event.actual_outcome);

        // 2. Counterfactuals: Try random alternative actions
        for _ in 0..5 {
            // Generate random action (hallucination)
            // In production, perturb the original action
            let dim = event.action.len();
            let alt_action = RealHV::random(dim, rand::random()).values;

            // 3. Simulate outcome O(1)
            let predicted_outcome = model.predict(&event.state, &alt_action, 1.0)?;
            let alt_phi = model.estimate_phi(&predicted_outcome);

            // 4. "Wisdom": Did we find a better path?
            if alt_phi > actual_phi + 0.05 {
                // Yes! This action would have been more conscious.
                // We should learn from this hypothetical.
                // Train the model on this "synthetic" success? 
                // Actually, we train the Policy (not implemented here), 
                // but we can update the World Model's "Concept of Value".
                insights += 1;
                
                // For now, just log it. In full system, this feeds LearningEngine.
                // println!("Dream Insight: Alternative action yielded higher Φ ({:.3} vs {:.3})", alt_phi, actual_phi);
            }
        }
        
        Ok(insights)
    }
}
