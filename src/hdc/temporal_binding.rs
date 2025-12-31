//! Temporal Binding for Consciousness
//!
//! # The Binding Problem in Time
//!
//! Consciousness requires binding experiences across time into a coherent stream.
//! This module implements temporal binding through:
//!
//! 1. **Present Moment Window**: The "specious present" (~3 seconds)
//! 2. **Temporal Integration**: How past influences present
//! 3. **Anticipatory Processing**: How future expectations shape now
//! 4. **Narrative Coherence**: Creating a continuous stream of experience
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - **Temporal Integration Window** (Poppel): ~3 second subjective present
//! - **Predictive Processing**: Past predicts future, shapes present
//! - **Memory Consolidation**: How working memory becomes episodic
//! - **Neural Oscillations**: Theta-gamma coupling for temporal binding
//!
//! # Architecture
//!
//! ```text
//!   PAST ─────────────> PRESENT ─────────────> FUTURE
//!     │                    │                     │
//!     │   ┌────────────────┼────────────────┐   │
//!     │   │          BINDING WINDOW          │   │
//!     │   │         (~3 seconds)             │   │
//!     │   │  ┌───────────────────────────┐  │   │
//!     │   │  │    Integrated Experience   │  │   │
//!     │   │  │  [memory + perception +    │  │   │
//!     │   │  │   anticipation = NOW]      │  │   │
//!     │   │  └───────────────────────────┘  │   │
//!     │   └────────────────┼────────────────┘   │
//!     │                    │                     │
//!     └───── decays ───────┴───── predicts ─────┘
//! ```

use super::real_hv::RealHV;
use super::unified_consciousness_engine::{ConsciousnessUpdate, ConsciousnessDimensions};
use std::collections::VecDeque;

/// Temporal binding window configuration
#[derive(Clone, Debug)]
pub struct TemporalBindingConfig {
    /// Window size in steps (representing ~3 seconds)
    pub window_size: usize,
    /// Decay rate for past experiences
    pub decay_rate: f64,
    /// Weight for anticipatory influence
    pub anticipation_weight: f64,
    /// HDC dimension
    pub dim: usize,
}

impl Default for TemporalBindingConfig {
    fn default() -> Self {
        Self {
            window_size: 30, // ~3 seconds at 100ms steps
            decay_rate: 0.1,
            anticipation_weight: 0.3,
            dim: 2048,
        }
    }
}

/// A moment of experience with temporal context
#[derive(Clone, Debug)]
pub struct TemporalMoment {
    /// The raw experience vector
    pub experience: RealHV,
    /// Integrated with temporal context
    pub bound_experience: RealHV,
    /// Timestamp (step number)
    pub step: usize,
    /// Integration strength with past
    pub past_integration: f64,
    /// Anticipation match (how well it matched prediction)
    pub anticipation_match: f64,
    /// Narrative continuity score
    pub continuity: f64,
}

/// Temporal integration summary
#[derive(Clone, Debug)]
pub struct TemporalIntegration {
    /// How strongly bound to past
    pub past_binding: f64,
    /// How strongly bound to anticipated future
    pub future_binding: f64,
    /// Overall temporal coherence
    pub coherence: f64,
    /// Length of continuous narrative
    pub narrative_length: usize,
    /// Dominant rhythm (simulated neural oscillation)
    pub dominant_rhythm: f64,
}

/// Temporal binding engine
pub struct TemporalBindingEngine {
    /// Configuration
    config: TemporalBindingConfig,
    /// Past moments buffer
    past_moments: VecDeque<TemporalMoment>,
    /// Current prediction of next moment
    anticipation: Option<RealHV>,
    /// Integrated temporal context
    temporal_context: RealHV,
    /// Step counter
    step: usize,
    /// Narrative vector (accumulates coherent experience)
    narrative: RealHV,
    /// Theta phase (for oscillation simulation)
    theta_phase: f64,
}

impl TemporalBindingEngine {
    /// Create new temporal binding engine
    pub fn new(config: TemporalBindingConfig) -> Self {
        let dim = config.dim;
        Self {
            config,
            past_moments: VecDeque::new(),
            anticipation: None,
            temporal_context: RealHV::zero(dim),
            step: 0,
            narrative: RealHV::zero(dim),
            theta_phase: 0.0,
        }
    }

    /// Bind a new experience into the temporal stream
    pub fn bind(&mut self, experience: &RealHV) -> TemporalMoment {
        self.step += 1;

        // Update theta phase (4-8Hz oscillation at ~100ms steps)
        self.theta_phase = (self.theta_phase + 0.6) % (2.0 * std::f64::consts::PI);
        let theta_weight = (self.theta_phase.sin() + 1.0) / 2.0;

        // 1. Compute anticipation match
        let anticipation_match = if let Some(ref predicted) = self.anticipation {
            experience.similarity(predicted).max(0.0) as f64
        } else {
            0.5 // Neutral if no prediction
        };

        // 2. Compute past integration
        let past_integration = self.compute_past_integration(experience);

        // 3. Create bound experience (integrate past + present + anticipated)
        let bound_experience = self.create_bound_experience(experience, theta_weight);

        // 4. Update temporal context
        self.update_temporal_context(&bound_experience);

        // 5. Update narrative
        self.update_narrative(&bound_experience);

        // 6. Generate next anticipation
        self.generate_anticipation();

        // 7. Compute continuity
        let continuity = self.compute_continuity(&bound_experience);

        // 8. Create moment
        let moment = TemporalMoment {
            experience: experience.clone(),
            bound_experience: bound_experience.clone(),
            step: self.step,
            past_integration,
            anticipation_match,
            continuity,
        };

        // 9. Store in buffer
        self.past_moments.push_back(moment.clone());
        if self.past_moments.len() > self.config.window_size {
            self.past_moments.pop_front();
        }

        moment
    }

    /// Compute how strongly this experience integrates with past
    fn compute_past_integration(&self, experience: &RealHV) -> f64 {
        if self.past_moments.is_empty() {
            return 0.5;
        }

        // Weighted average of similarity to recent past
        let mut total_sim = 0.0;
        let mut total_weight = 0.0;

        for (i, moment) in self.past_moments.iter().rev().enumerate() {
            let age = i as f64;
            let weight = (-age * self.config.decay_rate).exp();
            let sim = experience.similarity(&moment.bound_experience).max(0.0) as f64;
            total_sim += sim * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_sim / total_weight
        } else {
            0.5
        }
    }

    /// Create experience bound with temporal context
    fn create_bound_experience(&self, experience: &RealHV, theta_weight: f64) -> RealHV {
        let mut components = vec![experience.clone()];

        // Add decayed past contributions
        for (i, moment) in self.past_moments.iter().rev().enumerate() {
            let age = i as f64;
            let decay = (-age * self.config.decay_rate).exp();

            // Theta modulation: past is more accessible at theta peaks
            let theta_modulation = 0.5 + 0.5 * theta_weight;
            let weight = decay * theta_modulation;

            if weight > 0.05 {
                components.push(moment.bound_experience.scale(weight as f32));
            }
        }

        // Add anticipated future
        if let Some(ref anticipated) = self.anticipation {
            let antic_weight = self.config.anticipation_weight * (1.0 - theta_weight);
            components.push(anticipated.scale(antic_weight as f32));
        }

        // Bundle and normalize
        RealHV::bundle(&components)
    }

    /// Update running temporal context
    fn update_temporal_context(&mut self, bound: &RealHV) {
        let lr = 0.2;
        self.temporal_context = self.temporal_context
            .scale((1.0 - lr) as f32)
            .add(&bound.scale(lr as f32))
            .normalize();
    }

    /// Update narrative vector
    fn update_narrative(&mut self, bound: &RealHV) {
        // Narrative accumulates slowly (episodic integration)
        let lr = 0.05;
        self.narrative = self.narrative
            .scale((1.0 - lr) as f32)
            .add(&bound.scale(lr as f32))
            .normalize();
    }

    /// Generate anticipation of next moment
    fn generate_anticipation(&mut self) {
        if self.past_moments.len() < 3 {
            self.anticipation = Some(self.temporal_context.clone());
            return;
        }

        // Simple: predict as continuation of recent trend
        let recent: Vec<&TemporalMoment> = self.past_moments.iter().rev().take(5).collect();

        let mut predicted = RealHV::zero(self.config.dim);

        // Weighted recent history projection
        for (i, moment) in recent.iter().enumerate() {
            let recency = 1.0 / (i as f32 + 1.0);
            predicted = predicted.add(&moment.bound_experience.scale(recency));
        }

        // Blend with temporal context
        predicted = predicted.add(&self.temporal_context.scale(0.5));

        self.anticipation = Some(predicted.normalize());
    }

    /// Compute continuity with previous moment
    fn compute_continuity(&self, bound: &RealHV) -> f64 {
        if let Some(prev) = self.past_moments.back() {
            // Continuity = similarity to previous + narrative coherence
            let immediate = bound.similarity(&prev.bound_experience).max(0.0) as f64;
            let narrative = bound.similarity(&self.narrative).max(0.0) as f64;
            0.7 * immediate + 0.3 * narrative
        } else {
            0.5
        }
    }

    /// Get temporal integration summary
    pub fn integration_summary(&self) -> TemporalIntegration {
        let past_binding = if self.past_moments.is_empty() {
            0.0
        } else {
            self.past_moments.iter()
                .map(|m| m.past_integration)
                .sum::<f64>() / self.past_moments.len() as f64
        };

        let future_binding = if self.past_moments.is_empty() {
            0.0
        } else {
            self.past_moments.iter()
                .map(|m| m.anticipation_match)
                .sum::<f64>() / self.past_moments.len() as f64
        };

        let continuity_avg = if self.past_moments.is_empty() {
            0.0
        } else {
            self.past_moments.iter()
                .map(|m| m.continuity)
                .sum::<f64>() / self.past_moments.len() as f64
        };

        let coherence = (past_binding + future_binding + continuity_avg) / 3.0;

        TemporalIntegration {
            past_binding,
            future_binding,
            coherence,
            narrative_length: self.past_moments.len(),
            dominant_rhythm: 6.0, // Theta band center
        }
    }

    /// Get the current narrative vector
    pub fn narrative(&self) -> &RealHV {
        &self.narrative
    }

    /// Get the current temporal context
    pub fn temporal_context(&self) -> &RealHV {
        &self.temporal_context
    }

    /// Get anticipation for next moment
    pub fn anticipation(&self) -> Option<&RealHV> {
        self.anticipation.as_ref()
    }

    /// Current theta phase
    pub fn theta_phase(&self) -> f64 {
        self.theta_phase
    }

    /// Get stream health metrics
    pub fn stream_health(&self) -> StreamHealth {
        let integration = self.integration_summary();

        StreamHealth {
            coherence: integration.coherence,
            temporal_integration: (integration.past_binding + integration.future_binding) / 2.0,
            narrative_length: integration.narrative_length,
            is_flowing: integration.coherence > 0.4 && self.past_moments.len() >= 3,
            theta_rhythm: integration.dominant_rhythm,
        }
    }
}

/// Stream of consciousness - higher-level temporal binding
pub struct StreamOfConsciousness {
    /// Temporal binding engine
    binding: TemporalBindingEngine,
    /// Consciousness updates history
    consciousness_history: VecDeque<ConsciousnessUpdate>,
    /// Maximum history
    max_history: usize,
    /// Current stream coherence
    coherence: f64,
}

impl StreamOfConsciousness {
    /// Create new stream of consciousness
    pub fn new(dim: usize) -> Self {
        let config = TemporalBindingConfig {
            dim,
            ..Default::default()
        };

        Self {
            binding: TemporalBindingEngine::new(config),
            consciousness_history: VecDeque::new(),
            max_history: 100,
            coherence: 0.5,
        }
    }

    /// Process a consciousness update into the stream
    pub fn process(&mut self, update: &ConsciousnessUpdate) -> StreamUpdate {
        // Create experience vector from consciousness dimensions
        let experience = self.dimensions_to_vector(&update.dimensions);

        // Bind into temporal stream
        let moment = self.binding.bind(&experience);

        // Store update
        self.consciousness_history.push_back(update.clone());
        if self.consciousness_history.len() > self.max_history {
            self.consciousness_history.pop_front();
        }

        // Update coherence
        self.coherence = 0.9 * self.coherence + 0.1 * moment.continuity;

        let integration = self.binding.integration_summary();

        StreamUpdate {
            moment,
            integration,
            stream_coherence: self.coherence,
            phi_history: self.consciousness_history.iter().map(|u| u.phi).collect(),
        }
    }

    /// Convert consciousness dimensions to HDC vector
    fn dimensions_to_vector(&self, dims: &ConsciousnessDimensions) -> RealHV {
        let dim = self.binding.config.dim;

        // Create basis vectors for each dimension
        let phi_vec = RealHV::random(dim, (dims.phi * 10000.0) as u64);
        let workspace_vec = RealHV::random(dim, (dims.workspace * 10000.0 + 1000.0) as u64);
        let attention_vec = RealHV::random(dim, (dims.attention * 10000.0 + 2000.0) as u64);
        let temporal_vec = RealHV::random(dim, (dims.temporal * 10000.0 + 3000.0) as u64);

        // Weighted bundle
        RealHV::bundle(&[
            phi_vec.scale(dims.phi as f32),
            workspace_vec.scale(dims.workspace as f32),
            attention_vec.scale(dims.attention as f32),
            temporal_vec.scale(dims.temporal as f32),
        ])
    }

    /// Get overall stream health
    pub fn stream_health(&self) -> StreamHealth {
        let integration = self.binding.integration_summary();

        StreamHealth {
            coherence: self.coherence,
            temporal_integration: integration.coherence,
            narrative_length: integration.narrative_length,
            is_flowing: self.coherence > 0.5 && integration.coherence > 0.4,
            theta_rhythm: integration.dominant_rhythm,
        }
    }
}

/// Update from stream processing
#[derive(Clone, Debug)]
pub struct StreamUpdate {
    /// The temporal moment
    pub moment: TemporalMoment,
    /// Integration summary
    pub integration: TemporalIntegration,
    /// Overall stream coherence
    pub stream_coherence: f64,
    /// Recent Phi history
    pub phi_history: Vec<f64>,
}

/// Stream health metrics
#[derive(Clone, Debug)]
pub struct StreamHealth {
    /// How coherent is the stream
    pub coherence: f64,
    /// Temporal integration strength
    pub temporal_integration: f64,
    /// How long is the current narrative
    pub narrative_length: usize,
    /// Is consciousness "flowing"
    pub is_flowing: bool,
    /// Dominant theta rhythm
    pub theta_rhythm: f64,
}

impl std::fmt::Display for StreamHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let flow_status = if self.is_flowing { "FLOWING" } else { "FRAGMENTED" };
        write!(f, "Stream[{}: coherence={:.1}%, integration={:.1}%, narrative={}steps, theta={:.1}Hz]",
               flow_status,
               self.coherence * 100.0,
               self.temporal_integration * 100.0,
               self.narrative_length,
               self.theta_rhythm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_binding() {
        let config = TemporalBindingConfig {
            dim: 1024,
            window_size: 20,
            ..Default::default()
        };

        let mut engine = TemporalBindingEngine::new(config);

        println!("\nTemporal Binding Test:");
        for i in 0..25 {
            let experience = RealHV::random(1024, i * 100);
            let moment = engine.bind(&experience);

            if i % 5 == 0 {
                println!("Step {}: past_int={:.3}, antic_match={:.3}, continuity={:.3}",
                        moment.step, moment.past_integration,
                        moment.anticipation_match, moment.continuity);
            }
        }

        let summary = engine.integration_summary();
        println!("\nIntegration Summary:");
        println!("  Past binding: {:.3}", summary.past_binding);
        println!("  Future binding: {:.3}", summary.future_binding);
        println!("  Coherence: {:.3}", summary.coherence);
    }

    #[test]
    fn test_stream_continuity() {
        let config = TemporalBindingConfig {
            dim: 1024,
            window_size: 20,
            ..Default::default()
        };

        let mut engine = TemporalBindingEngine::new(config);

        // Create a sequence of similar experiences (should show high continuity)
        let base = RealHV::random(1024, 42);

        println!("\nStream Continuity Test:");
        for i in 0..15 {
            // Add small noise to base (simulating continuous experience)
            let noise = RealHV::random(1024, i * 1000).scale(0.1);
            let experience = base.add(&noise).normalize();

            let moment = engine.bind(&experience);
            println!("Step {}: continuity={:.3}", moment.step, moment.continuity);
        }

        let summary = engine.integration_summary();
        assert!(summary.coherence > 0.5, "Similar experiences should create coherent stream");
    }

    #[test]
    fn test_stream_of_consciousness() {
        use super::super::unified_consciousness_engine::{UnifiedConsciousnessEngine, EngineConfig};

        let engine_config = EngineConfig {
            hdc_dim: 1024,
            n_processes: 16,
            ..Default::default()
        };

        let mut engine = UnifiedConsciousnessEngine::new(engine_config);
        let mut stream = StreamOfConsciousness::new(1024);

        println!("\nStream of Consciousness Test:");
        for i in 0..20 {
            let input = RealHV::random(1024, i * 100);
            let update = engine.process(&input);
            let stream_update = stream.process(&update);

            if i % 4 == 0 {
                println!("Step {}: {}", i, stream.stream_health());
            }
        }

        let health = stream.stream_health();
        println!("\nFinal: {}", health);
    }
}
