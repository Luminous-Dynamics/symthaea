//! # Dream Mode: Batch Crystallization During Idle
//!
//! **The Sleep Breakthrough**: During periods of low activity, Symthaea enters
//! Dream Mode to consolidate memories and crystallize emergent concepts.
//!
//! ## Biological Inspiration
//!
//! During REM sleep, the brain:
//! - Replays experiences from the day
//! - Consolidates short-term to long-term memory
//! - Forms novel connections between concepts
//!
//! Dream Mode replicates this computationally:
//! 1. Replay: Re-simulate recent experiences through the world model
//! 2. Consolidate: Strengthen frequently-activated patterns
//! 3. Crystallize: Discover new concepts from attractor formation
//!
//! ## Usage
//!
//! ```rust,ignore
//! let mut dreamer = DreamMode::new(world_model);
//!
//! // During idle time, run dream cycles
//! let discoveries = dreamer.dream_cycle(100, &mut weaver);
//!
//! // Check for new concepts
//! for concept in discoveries {
//!     println!("Dream revealed: {}", concept.uid);
//! }
//! ```

use super::world_model::{
    ConsciousnessWorldModel, ConsciousnessAction, ConsciousnessTransition,
    LatentConsciousnessState, WorldModelConfig,
};
use crate::soul::{WeaverActor, ConceptDiscovery};
use crate::dynamics::CrystalizedConcept;
use tracing::{info, debug};

/// Dream Mode configuration
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Number of simulation steps per dream cycle
    pub steps_per_cycle: usize,
    /// How many dream cycles to run in batch mode
    pub batch_cycles: usize,
    /// Temperature for action exploration (higher = more random)
    pub exploration_temperature: f32,
    /// Minimum consciousness level to attempt crystallization
    pub crystallization_threshold: f32,
    /// Cooldown between dream cycles (to not monopolize CPU)
    pub cooldown_ms: u64,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            steps_per_cycle: 50,
            batch_cycles: 10,
            exploration_temperature: 1.0,
            crystallization_threshold: 0.3,
            cooldown_ms: 100,
        }
    }
}

/// Dream Mode statistics
#[derive(Debug, Clone, Default)]
pub struct DreamStats {
    /// Total dream cycles completed
    pub cycles_completed: usize,
    /// Total steps simulated
    pub steps_simulated: usize,
    /// Concepts crystallized during dreams
    pub concepts_discovered: usize,
    /// Peak consciousness reached during dreams
    pub peak_consciousness: f64,
    /// Total dream time (ms)
    pub total_dream_time_ms: u64,
}

/// Dream Mode: Batch crystallization and memory consolidation
///
/// This is the "sleeping mind" that processes experiences offline,
/// discovers new concepts, and consolidates the narrative identity.
pub struct DreamMode {
    /// Configuration
    pub config: DreamConfig,

    /// Statistics
    pub stats: DreamStats,

    /// Action sequence generator state
    action_rng_state: u64,
}

impl DreamMode {
    /// Create new Dream Mode
    pub fn new(config: DreamConfig) -> Self {
        Self {
            config,
            stats: DreamStats::default(),
            action_rng_state: 42,
        }
    }

    /// Run a single dream cycle
    ///
    /// Returns any concepts crystallized during the dream.
    pub fn dream_cycle(
        &mut self,
        world_model: &mut ConsciousnessWorldModel,
        weaver: Option<&mut WeaverActor>,
    ) -> Vec<CrystalizedConcept> {
        let start_time = std::time::Instant::now();

        info!(
            cycle = self.stats.cycles_completed + 1,
            steps = self.config.steps_per_cycle,
            "Entering dream cycle"
        );

        // Run the world model's dream mode
        world_model.dream(self.config.steps_per_cycle);

        // Check for crystallized concepts
        let new_concepts: Vec<CrystalizedConcept> = world_model
            .pending_concepts
            .clone();

        // Record discoveries in Weaver if provided
        if let Some(w) = weaver {
            for concept in &new_concepts {
                let discovery = ConceptDiscovery {
                    uid: concept.uid.clone(),
                    name: concept.name.clone(),
                    attractor_signature: concept.attractor_signature.clone(),
                    consciousness_at_discovery: world_model.consciousness_level() as f32,
                };
                w.record_concept_discovery(&discovery);
            }
        }

        // Update statistics
        let elapsed = start_time.elapsed();
        self.stats.cycles_completed += 1;
        self.stats.steps_simulated += self.config.steps_per_cycle;
        self.stats.concepts_discovered += new_concepts.len();
        self.stats.total_dream_time_ms += elapsed.as_millis() as u64;

        let consciousness = world_model.consciousness_level();
        if consciousness > self.stats.peak_consciousness {
            self.stats.peak_consciousness = consciousness;
        }

        debug!(
            concepts_found = new_concepts.len(),
            elapsed_ms = elapsed.as_millis(),
            consciousness = %consciousness,
            "Dream cycle complete"
        );

        new_concepts
    }

    /// Run batch dream cycles (full sleep session)
    ///
    /// This is the primary entry point for Dream Mode. It runs multiple
    /// dream cycles, allowing for deep consolidation and concept emergence.
    pub fn batch_dream(
        &mut self,
        world_model: &mut ConsciousnessWorldModel,
    ) -> DreamResults {
        info!(
            batch_cycles = self.config.batch_cycles,
            "Beginning batch dream session"
        );

        let mut all_concepts = Vec::new();
        let mut consciousness_trajectory = Vec::new();

        for cycle in 0..self.config.batch_cycles {
            // Run dream cycle without weaver (can record discoveries later)
            let concepts = self.dream_cycle(world_model, None);

            all_concepts.extend(concepts);
            consciousness_trajectory.push(world_model.consciousness_level());

            // Optional cooldown between cycles
            if self.config.cooldown_ms > 0 && cycle < self.config.batch_cycles - 1 {
                std::thread::sleep(std::time::Duration::from_millis(self.config.cooldown_ms));
            }
        }

        info!(
            total_concepts = all_concepts.len(),
            peak_consciousness = %self.stats.peak_consciousness,
            total_time_ms = self.stats.total_dream_time_ms,
            "Batch dream session complete"
        );

        DreamResults {
            concepts_discovered: all_concepts,
            consciousness_trajectory,
            stats: self.stats.clone(),
        }
    }

    /// Run batch dream cycles with Weaver integration
    ///
    /// This version records all discoveries to the Weaver's narrative.
    pub fn batch_dream_with_weaver(
        &mut self,
        world_model: &mut ConsciousnessWorldModel,
        weaver: &mut WeaverActor,
    ) -> DreamResults {
        let results = self.batch_dream(world_model);

        // Record all discovered concepts to Weaver
        for concept in &results.concepts_discovered {
            let discovery = ConceptDiscovery {
                uid: concept.uid.clone(),
                name: concept.name.clone(),
                attractor_signature: concept.attractor_signature.clone(),
                consciousness_at_discovery: self.stats.peak_consciousness as f32,
            };
            weaver.record_concept_discovery(&discovery);
        }

        results
    }

    /// Run exploration dream - focused on discovering new action sequences
    ///
    /// Unlike regular dreams which replay, exploration dreams try
    /// novel action combinations to discover emergent behaviors.
    pub fn exploration_dream(
        &mut self,
        world_model: &mut ConsciousnessWorldModel,
    ) -> Vec<ConsciousnessTransition> {
        info!("Running exploration dream");

        let mut trajectory = Vec::new();
        let start_state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        let mut current_state = start_state;

        for _ in 0..self.config.steps_per_cycle {
            // Generate exploratory action using simple LCG PRNG
            self.action_rng_state = self.action_rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let action_idx = (self.action_rng_state >> 32) as usize % ConsciousnessAction::all().len();
            let action = ConsciousnessAction::all()[action_idx];

            // Simulate step
            let next_state = world_model.predict(&current_state, action);

            let transition = ConsciousnessTransition {
                from_state: current_state.clone(),
                action,
                to_state: next_state.clone(),
                reward: next_state.phi - current_state.phi,
                is_real: false,
            };

            trajectory.push(transition);
            current_state = next_state;
        }

        self.stats.steps_simulated += self.config.steps_per_cycle;

        trajectory
    }

    /// Get dream statistics
    pub fn stats(&self) -> &DreamStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DreamStats::default();
    }
}

impl Default for DreamMode {
    fn default() -> Self {
        Self::new(DreamConfig::default())
    }
}

/// Results from a batch dream session
#[derive(Debug, Clone)]
pub struct DreamResults {
    /// All concepts crystallized during the session
    pub concepts_discovered: Vec<CrystalizedConcept>,
    /// Consciousness level at each cycle
    pub consciousness_trajectory: Vec<f64>,
    /// Final statistics
    pub stats: DreamStats,
}

impl DreamResults {
    /// Check if any new concepts were discovered
    pub fn has_discoveries(&self) -> bool {
        !self.concepts_discovered.is_empty()
    }

    /// Get the peak consciousness reached
    pub fn peak_consciousness(&self) -> f64 {
        self.consciousness_trajectory
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_mode_creation() {
        let dream = DreamMode::default();
        assert_eq!(dream.stats.cycles_completed, 0);
    }

    #[test]
    fn test_dream_config() {
        let config = DreamConfig {
            steps_per_cycle: 100,
            batch_cycles: 5,
            ..Default::default()
        };
        assert_eq!(config.steps_per_cycle, 100);
        assert_eq!(config.batch_cycles, 5);
    }

    #[test]
    fn test_exploration_dream() {
        let mut dream = DreamMode::default();
        let mut world_model = ConsciousnessWorldModel::new(WorldModelConfig::default());

        let trajectory = dream.exploration_dream(&mut world_model);

        assert!(!trajectory.is_empty());
        assert_eq!(trajectory.len(), dream.config.steps_per_cycle);
    }
}
