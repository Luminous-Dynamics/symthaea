//! # Autopoietic Consciousness: Self-Maintaining Awareness
//!
//! Implements autopoietic (self-creating, self-maintaining) consciousness
//! based on Maturana and Varela's theory of autopoiesis.
//!
//! Key concepts:
//! - Operational closure: The system produces its own components
//! - Structural coupling: Interaction with environment while maintaining identity
//! - Self-referential dynamics: The system observes and modifies itself

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::RealHV;

/// Configuration for autopoietic consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticConfig {
    /// Dimension of state vectors
    pub dimension: usize,
    /// Self-reference strength
    pub self_reference_strength: f32,
    /// Boundary maintenance threshold
    pub boundary_threshold: f32,
    /// Adaptation rate
    pub adaptation_rate: f32,
    /// History size
    pub history_size: usize,
    /// Perturbation sensitivity
    pub sensitivity: f32,
}

impl Default for AutopoieticConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            self_reference_strength: 0.8,
            boundary_threshold: 0.5,
            adaptation_rate: 0.1,
            history_size: 100,
            sensitivity: 0.3,
        }
    }
}

/// The current state of autopoietic consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticState {
    /// Overall integrity (0.0-1.0)
    pub integrity: f32,
    /// Closure level (operational closure)
    pub closure: f32,
    /// Coupling strength (with environment)
    pub coupling: f32,
    /// Self-reference level
    pub self_reference: f32,
    /// Boundary strength
    pub boundary_strength: f32,
    /// Adaptation level
    pub adaptation: f32,
    /// Current phase
    pub phase: AutopoieticPhase,
}

impl Default for AutopoieticState {
    fn default() -> Self {
        Self {
            integrity: 0.8,
            closure: 0.7,
            coupling: 0.5,
            self_reference: 0.6,
            boundary_strength: 0.7,
            adaptation: 0.5,
            phase: AutopoieticPhase::Maintaining,
        }
    }
}

/// Phases of autopoietic operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutopoieticPhase {
    /// Self-production of components
    Producing,
    /// Maintaining organization
    Maintaining,
    /// Adapting to perturbations
    Adapting,
    /// Boundary regulation
    Regulating,
    /// Self-observation
    Observing,
    /// Integration/consolidation
    Integrating,
}

/// A perturbation from the environment
#[derive(Debug, Clone)]
pub struct Perturbation {
    /// Perturbation identifier
    pub id: u64,
    /// Source of perturbation
    pub source: String,
    /// Intensity (0.0-1.0)
    pub intensity: f32,
    /// Direction/content as hypervector
    pub content: RealHV,
    /// Timestamp
    pub timestamp: u64,
}

impl Perturbation {
    /// Create a new perturbation
    pub fn new(id: u64, source: impl Into<String>, intensity: f32, content: RealHV) -> Self {
        Self {
            id,
            source: source.into(),
            intensity: intensity.clamp(0.0, 1.0),
            content,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }
}

/// Internal component produced by the system
#[derive(Debug, Clone)]
pub struct Component {
    /// Component identifier
    pub id: u64,
    /// Component type
    pub component_type: ComponentType,
    /// State vector
    pub state: RealHV,
    /// Health/viability
    pub health: f32,
    /// Generation (when it was produced)
    pub generation: u64,
}

/// Types of internal components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    /// Boundary component
    Boundary,
    /// Processing component
    Processing,
    /// Memory component
    Memory,
    /// Integration component
    Integration,
    /// Self-model component
    SelfModel,
}

/// The autopoietic consciousness system
#[derive(Debug)]
pub struct AutopoieticConsciousness {
    /// Configuration
    config: AutopoieticConfig,
    /// Current state
    state: AutopoieticState,
    /// Internal components
    components: HashMap<u64, Component>,
    /// Next component ID
    next_component_id: u64,
    /// Current generation
    generation: u64,
    /// State history
    history: VecDeque<AutopoieticState>,
    /// Self-model (how the system sees itself)
    self_model: RealHV,
    /// Boundary representation
    boundary: RealHV,
    /// Statistics
    stats: AutopoieticStats,
}

/// Statistics for the system
#[derive(Debug, Clone, Default)]
pub struct AutopoieticStats {
    /// Total perturbations processed
    pub perturbations_processed: u64,
    /// Components produced
    pub components_produced: u64,
    /// Components decayed
    pub components_decayed: u64,
    /// Boundary violations
    pub boundary_violations: u64,
    /// Self-observations
    pub self_observations: u64,
}

impl AutopoieticConsciousness {
    /// Create a new autopoietic consciousness
    pub fn new(config: AutopoieticConfig) -> Self {
        let dim = config.dimension;
        Self {
            config,
            state: AutopoieticState::default(),
            components: HashMap::new(),
            next_component_id: 1,
            generation: 0,
            history: VecDeque::new(),
            self_model: RealHV::random(dim, 42),
            boundary: RealHV::random(dim, 42),
            stats: AutopoieticStats::default(),
        }
    }

    /// Initialize with basic components
    pub fn initialize(&mut self) {
        // Create initial components
        self.produce_component(ComponentType::Boundary);
        self.produce_component(ComponentType::Processing);
        self.produce_component(ComponentType::Memory);
        self.produce_component(ComponentType::SelfModel);
    }

    /// Produce a new internal component
    pub fn produce_component(&mut self, component_type: ComponentType) -> u64 {
        let id = self.next_component_id;
        self.next_component_id += 1;

        let state = match component_type {
            ComponentType::Boundary => self.boundary.perturb(0.1),
            ComponentType::SelfModel => self.self_model.perturb(0.1),
            _ => RealHV::random(self.config.dimension, 42),
        };

        let component = Component {
            id,
            component_type,
            state,
            health: 1.0,
            generation: self.generation,
        };

        self.components.insert(id, component);
        self.stats.components_produced += 1;

        id
    }

    /// Process a perturbation from the environment
    pub fn process_perturbation(&mut self, perturbation: &Perturbation) -> bool {
        self.stats.perturbations_processed += 1;

        // Check if perturbation crosses boundary
        let boundary_similarity = self.boundary.similarity(&perturbation.content);

        if boundary_similarity < self.config.boundary_threshold {
            // Perturbation blocked at boundary
            self.stats.boundary_violations += 1;

            // Strengthen boundary
            self.boundary = RealHV::bundle(&[self.boundary.clone(), perturbation.content.clone()]);
            self.state.boundary_strength = (self.state.boundary_strength + 0.1).min(1.0);

            return false;
        }

        // Perturbation accepted - adapt to it
        self.state.phase = AutopoieticPhase::Adapting;

        // Update internal state based on perturbation
        let adaptation_strength = perturbation.intensity * self.config.adaptation_rate;

        // Perturb processing components
        for component in self.components.values_mut() {
            if component.component_type == ComponentType::Processing {
                let perturbation_effect = perturbation.content.clone().scale(adaptation_strength);
                component.state = RealHV::bundle(&[component.state.clone(), perturbation_effect]);
            }
        }

        // Update coupling
        self.state.coupling = (self.state.coupling + adaptation_strength).min(1.0);

        true
    }

    /// Perform self-observation (metacognition)
    pub fn self_observe(&mut self) -> AutopoieticState {
        self.stats.self_observations += 1;
        self.state.phase = AutopoieticPhase::Observing;

        // Calculate integrity from component health
        let total_health: f32 = self.components.values().map(|c| c.health).sum();
        let component_count = self.components.len() as f32;
        self.state.integrity = if component_count > 0.0 {
            total_health / component_count
        } else {
            0.0
        };

        // Calculate operational closure
        let boundary_count = self.components.values()
            .filter(|c| c.component_type == ComponentType::Boundary)
            .count();
        self.state.closure = (boundary_count as f32 / self.components.len().max(1) as f32)
            .min(1.0);

        // Update self-model
        let component_states: Vec<RealHV> = self.components.values()
            .map(|c| c.state.clone())
            .collect();

        if !component_states.is_empty() {
            self.self_model = RealHV::bundle(&component_states);
        }

        // Calculate self-reference
        let model_components: Vec<_> = self.components.values()
            .filter(|c| c.component_type == ComponentType::SelfModel)
            .collect();

        if let Some(model_comp) = model_components.first() {
            self.state.self_reference = self.self_model.similarity(&model_comp.state);
        }

        // Record in history
        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(self.state.clone());

        self.state.clone()
    }

    /// Maintain the system (decay and regenerate)
    pub fn maintain(&mut self) {
        self.state.phase = AutopoieticPhase::Maintaining;
        self.generation += 1;

        // Decay components
        let mut to_remove = Vec::new();
        for (id, component) in self.components.iter_mut() {
            component.health -= 0.01;
            if component.health <= 0.0 {
                to_remove.push(*id);
            }
        }

        // Remove dead components
        for id in to_remove {
            self.components.remove(&id);
            self.stats.components_decayed += 1;
        }

        // Regenerate essential components
        let boundary_count = self.components.values()
            .filter(|c| c.component_type == ComponentType::Boundary)
            .count();

        if boundary_count < 2 {
            self.produce_component(ComponentType::Boundary);
        }

        let processing_count = self.components.values()
            .filter(|c| c.component_type == ComponentType::Processing)
            .count();

        if processing_count < 3 {
            self.produce_component(ComponentType::Processing);
        }

        // Update adaptation based on history
        if self.history.len() >= 2 {
            let recent: Vec<_> = self.history.iter().rev().take(5).collect();
            let integrity_trend: f32 = recent.windows(2)
                .map(|w| w[0].integrity - w[1].integrity)
                .sum::<f32>() / (recent.len() - 1).max(1) as f32;

            self.state.adaptation = (self.state.adaptation + integrity_trend * 0.1).clamp(0.0, 1.0);
        }
    }

    /// Integrate all components into coherent whole
    pub fn integrate(&mut self) {
        self.state.phase = AutopoieticPhase::Integrating;

        // Bundle all component states
        let states: Vec<RealHV> = self.components.values()
            .map(|c| c.state.clone())
            .collect();

        if states.len() >= 2 {
            let integrated = RealHV::bundle(&states);

            // Update self-model with integrated state
            self.self_model = RealHV::bundle(&[self.self_model.clone(), integrated]);
        }

        // Update closure based on integration
        self.state.closure = self.calculate_closure();
    }

    /// Calculate operational closure
    fn calculate_closure(&self) -> f32 {
        if self.components.len() < 2 {
            return 0.0;
        }

        // Closure = how much components reference each other
        let component_states: Vec<_> = self.components.values()
            .map(|c| &c.state)
            .collect();

        let mut total_similarity = 0.0;
        let mut count = 0;

        for i in 0..component_states.len() {
            for j in (i + 1)..component_states.len() {
                let sim = component_states[i].similarity(component_states[j]);
                total_similarity += sim;
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f32
        } else {
            0.0
        }
    }

    /// Get current state
    pub fn state(&self) -> &AutopoieticState {
        &self.state
    }

    /// Get self-model
    pub fn self_model(&self) -> &RealHV {
        &self.self_model
    }

    /// Get statistics
    pub fn stats(&self) -> &AutopoieticStats {
        &self.stats
    }

    /// Get component count
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Get current generation
    pub fn generation(&self) -> u64 {
        self.generation
    }

    // =========================================================================
    // Neuro-Bridge Compatibility Methods
    // =========================================================================

    /// Create a new autopoietic consciousness with config (alias for new + initialize)
    pub fn with_config(config: AutopoieticConfig) -> Self {
        let mut system = Self::new(config);
        system.initialize();
        system
    }

    /// Get overall health score (0.0-1.0) based on system integrity and component health
    ///
    /// This is a composite metric combining:
    /// - Component integrity (average health of all components)
    /// - Operational closure (how well-connected the system is)
    /// - Boundary strength (system's ability to maintain identity)
    pub fn health_score(&self) -> f64 {
        // Component health contribution
        let component_health = if self.components.is_empty() {
            0.0
        } else {
            let total: f32 = self.components.values().map(|c| c.health).sum();
            total / self.components.len() as f32
        };

        // Combine with state metrics (weighted average)
        let score = 0.4 * component_health
            + 0.2 * self.state.integrity
            + 0.2 * self.state.closure
            + 0.2 * self.state.boundary_strength;

        score as f64
    }

    /// Update the autopoietic system based on external signals
    ///
    /// This is a high-level update method for neuro-bridge integration that:
    /// 1. Incorporates phi/coherence signals as metabolic fuel
    /// 2. Applies perturbation stress
    /// 3. Runs maintenance cycle
    ///
    /// # Arguments
    /// * `phi` - Integrated information from neural processing (0.0-1.0)
    /// * `coherence` - Neural coherence level (0.0-1.0)
    /// * `perturbation_stress` - External stress factor (0.0-1.0)
    pub fn update(&mut self, phi: f64, coherence: f64, perturbation_stress: f64) {
        // 1. Neural coherence fuels component health (upward causation)
        let fuel = ((phi + coherence) / 2.0) as f32;
        for component in self.components.values_mut() {
            // High coherence regenerates health
            component.health = (component.health + fuel * 0.05).min(1.0);
        }

        // 2. Apply perturbation stress as boundary challenge
        if perturbation_stress > 0.1 {
            // Create a stress perturbation
            let stress_perturbation = Perturbation::new(
                self.generation,
                "neural_stress",
                perturbation_stress as f32,
                self.boundary.perturb(perturbation_stress as f32),
            );
            self.process_perturbation(&stress_perturbation);
        }

        // 3. Run maintenance cycle
        self.maintain();

        // 4. Update adaptation based on phi (higher phi = better adaptation)
        self.state.adaptation = (self.state.adaptation + phi as f32 * 0.1).clamp(0.0, 1.0);

        // 5. Perform self-observation to update metrics
        self.self_observe();
    }
}

impl Default for AutopoieticConsciousness {
    fn default() -> Self {
        let mut system = Self::new(AutopoieticConfig::default());
        system.initialize();
        system
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let system = AutopoieticConsciousness::default();
        assert!(system.component_count() > 0);
    }

    #[test]
    fn test_component_production() {
        let mut system = AutopoieticConsciousness::new(AutopoieticConfig::default());
        let id = system.produce_component(ComponentType::Processing);
        assert!(system.components.contains_key(&id));
    }

    #[test]
    fn test_self_observation() {
        let mut system = AutopoieticConsciousness::default();
        let state = system.self_observe();
        assert!(state.integrity > 0.0);
    }

    #[test]
    fn test_perturbation() {
        let mut system = AutopoieticConsciousness::default();
        let perturbation = Perturbation::new(
            1,
            "environment",
            0.5,
            RealHV::random(512, 42)
        );
        system.process_perturbation(&perturbation);
        assert!(system.stats.perturbations_processed > 0);
    }

    #[test]
    fn test_maintenance() {
        let mut system = AutopoieticConsciousness::default();
        let initial_gen = system.generation();
        system.maintain();
        assert_eq!(system.generation(), initial_gen + 1);
    }
}
