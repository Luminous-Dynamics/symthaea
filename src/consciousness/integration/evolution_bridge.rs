// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Evolution Bridge
//!
//! Connects primitive evolution with recursive self-improvement.
//!
//! ## Purpose
//!
//! The system has two evolutionary mechanisms:
//! 1. **Primitive Evolution** (`primitive_evolution.rs`): Evolves optimal reasoning primitives
//! 2. **Recursive Improvement** (`recursive_improvement/`): Evolves architecture and routing
//!
//! This bridge connects them so they can co-evolve.
//!
//! ## Co-Evolution Strategy
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    CO-EVOLUTIONARY LOOP                              │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  Primitive Evolution          ◄──────►        Recursive Improvement │
//! │  ┌─────────────────┐                   ┌─────────────────────────┐ │
//! │  │ Discover optimal │    Feedback     │ Improve architecture    │ │
//! │  │ primitives based │ ◄─────────────► │ using evolved           │ │
//! │  │ on Φ fitness     │                 │ primitives              │ │
//! │  └────────┬────────┘                   └──────────┬──────────────┘ │
//! │           │                                       │                │
//! │           ▼                                       ▼                │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │              Evolution Coordinator                          │   │
//! │  │  - Schedules primitive vs architecture evolution            │   │
//! │  │  - Shares fitness signals between systems                   │   │
//! │  │  - Prevents catastrophic interference                       │   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use crate::consciousness::primitive_evolution::{EvolutionConfig, PrimitiveEvolution};
use crate::consciousness::primitive_reasoning::{
    AdaptivePrimitiveSelector, ReasoningChain, TaskType,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::primitive_system::PrimitiveTier;

// =============================================================================
// EVOLUTION COORDINATOR
// =============================================================================

/// Coordinates co-evolution between primitives and architecture
pub struct EvolutionCoordinator {
    /// Primitive evolution state
    primitive_evolver: Option<PrimitiveEvolution>,

    /// Currently evolving tier (tracked separately since config is private)
    active_tier: Option<PrimitiveTier>,

    /// Learning from primitive usage
    selector: AdaptivePrimitiveSelector,

    /// Evolution schedule
    schedule: EvolutionSchedule,

    /// Current generation
    generation: usize,

    /// Evolution history
    history: Vec<EvolutionSnapshot>,

    /// Configuration
    config: CoordinatorConfig,
}

impl std::fmt::Debug for EvolutionCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvolutionCoordinator")
            .field(
                "primitive_evolver",
                &self
                    .primitive_evolver
                    .as_ref()
                    .map(|_| "<PrimitiveEvolution>"),
            )
            .field("active_tier", &self.active_tier)
            .field("selector", &self.selector)
            .field("schedule", &self.schedule)
            .field("generation", &self.generation)
            .field("history_len", &self.history.len())
            .field("config", &self.config)
            .finish()
    }
}

/// Configuration for the coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// How often to run primitive evolution (every N generations)
    pub primitive_evolution_frequency: usize,

    /// How many primitive generations per architectural generation
    pub primitive_generations_per_arch: usize,

    /// Minimum phi improvement to accept evolved primitives
    pub min_phi_improvement: f64,

    /// Maximum primitives to evolve simultaneously
    pub max_evolving_primitives: usize,

    /// Enable cross-tier evolution
    pub cross_tier_evolution: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            primitive_evolution_frequency: 5,
            primitive_generations_per_arch: 3,
            min_phi_improvement: 0.05,
            max_evolving_primitives: 10,
            cross_tier_evolution: true,
        }
    }
}

/// Schedule for interleaving evolutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvolutionSchedule {
    /// Run primitive evolution every N architecture generations
    Interleaved { frequency: usize },

    /// Run primitive evolution when architecture improvement stalls
    OnPlateau { stall_generations: usize },

    /// Run both in parallel
    Parallel,

    /// Only run primitive evolution
    PrimitiveOnly,

    /// Only run architecture evolution
    ArchitectureOnly,

    /// CfC-HDC neuroevolution via FEP fitness (feature: neuroevolution)
    Neuroevolution,
}

impl Default for EvolutionSchedule {
    fn default() -> Self {
        Self::Interleaved { frequency: 5 }
    }
}

/// Snapshot of evolution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionSnapshot {
    pub generation: usize,
    pub primitive_psi: f64,
    pub architecture_phi: f64,
    pub evolved_primitives: usize,
    pub active_tier: Option<PrimitiveTier>,
    pub timestamp: f64,
}

impl EvolutionCoordinator {
    /// Create a new coordinator
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            primitive_evolver: None,
            active_tier: None,
            selector: AdaptivePrimitiveSelector::new(),
            schedule: EvolutionSchedule::Interleaved {
                frequency: config.primitive_evolution_frequency,
            },
            generation: 0,
            history: Vec::new(),
            config,
        }
    }

    /// Step the evolution process
    pub fn step(&mut self) -> Result<EvolutionStepResult> {
        self.generation += 1;

        let should_evolve_primitives = match &self.schedule {
            EvolutionSchedule::Interleaved { frequency } => {
                self.generation.is_multiple_of(*frequency)
            }
            EvolutionSchedule::OnPlateau { stall_generations } => {
                self.detect_plateau(*stall_generations)
            }
            EvolutionSchedule::Parallel => true,
            EvolutionSchedule::PrimitiveOnly => true,
            EvolutionSchedule::ArchitectureOnly => false,
            EvolutionSchedule::Neuroevolution => false, // Handled by NeuroevolutionManager
        };

        let mut result = EvolutionStepResult {
            generation: self.generation,
            primitive_evolution_ran: false,
            primitive_psi_delta: 0.0,
            architecture_evolution_ran: false,
            architecture_phi_delta: 0.0,
            new_primitives: Vec::new(),
        };

        if should_evolve_primitives {
            let primitive_result = self.evolve_primitives()?;
            result.primitive_evolution_ran = true;
            result.primitive_psi_delta = primitive_result.phi_delta;
            result.new_primitives = primitive_result.new_primitives;
        }

        // Record snapshot
        self.history.push(EvolutionSnapshot {
            generation: self.generation,
            primitive_psi: self.get_current_primitive_psi(),
            architecture_phi: 0.0, // Would be filled by architecture evolution
            evolved_primitives: result.new_primitives.len(),
            active_tier: self.get_active_evolution_tier(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        });

        Ok(result)
    }

    /// Run primitive evolution
    fn evolve_primitives(&mut self) -> Result<PrimitiveEvolutionResult> {
        // Determine which tier to evolve
        let tier = self.select_evolution_tier();
        self.active_tier = Some(tier);

        // Create or update evolver
        let config = EvolutionConfig {
            tier,
            population_size: self.config.max_evolving_primitives,
            num_generations: self.config.primitive_generations_per_arch,
            mutation_rate: 0.1,
            crossover_rate: 0.3,
            elitism_count: 2,
            fitness_tasks: Vec::new(),
            convergence_threshold: 0.001,
            phi_weight: 0.4,
            harmonic_weight: 0.3,
            epistemic_weight: 0.3,
        };

        let mut evolver = PrimitiveEvolution::new(config)?;

        // Run evolution
        let evolution_result = evolver.evolve()?;

        // Filter by minimum improvement
        let improved_primitives: Vec<String> = evolution_result
            .final_primitives
            .iter()
            .filter(|p| {
                // Check if this primitive improved phi
                let baseline = self
                    .selector
                    .all_stats()
                    .iter()
                    .filter(|((name, _), _)| *name == p.name)
                    .map(|(_, stats)| stats.mean_phi())
                    .next()
                    .unwrap_or(0.0);

                p.fitness > baseline + self.config.min_phi_improvement
            })
            .map(|p| p.name.clone())
            .collect();

        Ok(PrimitiveEvolutionResult {
            phi_delta: evolution_result.phi_improvement_percent / 100.0,
            new_primitives: improved_primitives,
            tier,
        })
    }

    /// Select which tier to evolve based on current performance
    fn select_evolution_tier(&self) -> PrimitiveTier {
        // Find tier with lowest average phi
        let tier_stats = self.compute_tier_stats();

        tier_stats
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(tier, _)| tier)
            .unwrap_or(PrimitiveTier::Physical) // Default to Physical
    }

    /// Compute average phi by tier using adaptive selector statistics
    ///
    /// This method aggregates per-primitive statistics by tier to inform
    /// evolution decisions. Lower-performing tiers are prioritized for evolution.
    fn compute_tier_stats(&self) -> Vec<(PrimitiveTier, f64)> {
        use std::collections::HashMap;

        // Aggregate phi stats by tier
        let mut tier_stats: HashMap<PrimitiveTier, (f64, usize)> = HashMap::new();

        // Initialize all tiers
        for tier in [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ] {
            tier_stats.insert(tier, (0.0, 0));
        }

        // Get primitive system for tier lookups (uses cached global instance)
        let prim_system = crate::hdc::primitive_system::PrimitiveSystem::global();

        // Aggregate stats from selector by tier
        for ((prim_name, _task), stats) in self.selector.all_stats() {
            // Look up the primitive's tier
            if let Some(primitive) = prim_system.get(prim_name) {
                let tier = primitive.tier;
                let entry = tier_stats.entry(tier).or_insert((0.0, 0));
                entry.0 += stats.total_phi;
                entry.1 += stats.use_count;
            }
        }

        // Compute average phi per tier (lower = needs more evolution)
        tier_stats
            .into_iter()
            .map(|(tier, (total_phi, count))| {
                let avg_phi = if count > 0 {
                    total_phi / count as f64
                } else {
                    0.5 // Default for unused tiers
                };
                (tier, avg_phi)
            })
            .collect()
    }

    /// Detect if evolution has plateaued
    fn detect_plateau(&self, stall_generations: usize) -> bool {
        if self.history.len() < stall_generations {
            return false;
        }

        let recent = &self.history[self.history.len() - stall_generations..];
        let first_phi = recent.first().map(|s| s.primitive_psi).unwrap_or(0.0);
        let last_phi = recent.last().map(|s| s.primitive_psi).unwrap_or(0.0);

        // Plateau if improvement < 1%
        (last_phi - first_phi).abs() < 0.01
    }

    /// Get current primitive phi
    fn get_current_primitive_psi(&self) -> f64 {
        self.history.last().map(|s| s.primitive_psi).unwrap_or(0.5)
    }

    /// Get currently evolving tier
    fn get_active_evolution_tier(&self) -> Option<PrimitiveTier> {
        self.active_tier
    }

    /// Update selector from reasoning chain
    pub fn update_from_reasoning(&mut self, chain: &ReasoningChain, task: TaskType) {
        self.selector.update_from_chain(chain, task);
    }

    /// Get evolution history
    pub fn history(&self) -> &[EvolutionSnapshot] {
        &self.history
    }

    /// Get current generation
    pub fn generation(&self) -> usize {
        self.generation
    }
}

impl Default for EvolutionCoordinator {
    fn default() -> Self {
        Self::new(CoordinatorConfig::default())
    }
}

/// Result of a single evolution step
#[derive(Debug, Clone)]
pub struct EvolutionStepResult {
    pub generation: usize,
    pub primitive_evolution_ran: bool,
    pub primitive_psi_delta: f64,
    pub architecture_evolution_ran: bool,
    pub architecture_phi_delta: f64,
    pub new_primitives: Vec<String>,
}

/// Result of primitive evolution
#[derive(Debug, Clone)]
struct PrimitiveEvolutionResult {
    phi_delta: f64,
    new_primitives: Vec<String>,
    #[allow(dead_code)]
    tier: PrimitiveTier,
}

// =============================================================================
// FEEDBACK MECHANISMS
// =============================================================================

/// Feedback from architecture evolution to primitive evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureFeedback {
    /// Which primitives performed well in current architecture
    pub high_performers: Vec<(String, f64)>,

    /// Which primitives underperformed
    pub low_performers: Vec<(String, f64)>,

    /// Routing decisions that used primitives
    pub routing_primitive_usage: std::collections::HashMap<String, usize>,

    /// Overall architecture phi
    pub architecture_phi: f64,
}

/// Feedback from primitive evolution to architecture evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveFeedback {
    /// Newly evolved primitives
    pub new_primitives: Vec<String>,

    /// Deprecated primitives (underperforming)
    pub deprecated_primitives: Vec<String>,

    /// Tier statistics update
    pub tier_stats: std::collections::HashMap<PrimitiveTier, f64>,

    /// Overall primitive fitness
    pub primitive_fitness: f64,
}

impl EvolutionCoordinator {
    /// Process feedback from architecture evolution.
    ///
    /// This implements the bidirectional feedback loop where architecture evolution
    /// informs primitive selection weights. High-performing primitives get boosted,
    /// low-performing ones get penalized, and consistently underperforming tiers
    /// may trigger evolution runs.
    ///
    /// # Feedback Processing
    ///
    /// 1. **High performers**: Recorded with positive phi to boost Thompson sampling weights
    /// 2. **Low performers**: Recorded with negative phi to reduce selection probability
    /// 3. **Routing usage**: Tracked to understand which primitives are actually used in routing
    /// 4. **Tier evolution**: Low-performing tiers are flagged for evolution
    pub fn receive_architecture_feedback(&mut self, feedback: ArchitectureFeedback) {
        let prim_system = crate::hdc::primitive_system::PrimitiveSystem::global();

        // Boost selectors for high performers across all task types
        // High performers get positive observations to increase their Thompson sampling weights
        for (name, phi) in &feedback.high_performers {
            // Record for the General task type (architecture-level feedback applies broadly)
            self.selector
                .record_observation(name, TaskType::General, *phi);

            // Also update task-specific stats based on primitive tier
            if let Some(primitive) = prim_system.get(name) {
                let task_type = match primitive.tier {
                    PrimitiveTier::Mathematical => TaskType::Logical,
                    PrimitiveTier::Physical => TaskType::Causal,
                    PrimitiveTier::Geometric => TaskType::Spatial,
                    PrimitiveTier::Strategic => TaskType::Social,
                    PrimitiveTier::MetaCognitive => TaskType::MetaCognitive,
                    PrimitiveTier::Consciousness => TaskType::MetaCognitive,
                    _ => TaskType::General,
                };
                self.selector.record_observation(name, task_type, *phi);
            }
        }

        // Penalize low performers to reduce their selection probability
        for (name, phi) in &feedback.low_performers {
            // Record negative contribution (or zero if phi was already near zero)
            let penalty = if *phi < 0.1 { -0.1 } else { 0.0 };
            self.selector
                .record_observation(name, TaskType::General, penalty);

            // Also record for tier-specific task type
            if let Some(primitive) = prim_system.get(name) {
                let task_type = match primitive.tier {
                    PrimitiveTier::Mathematical => TaskType::Logical,
                    PrimitiveTier::Physical => TaskType::Causal,
                    PrimitiveTier::Geometric => TaskType::Spatial,
                    PrimitiveTier::Strategic => TaskType::Social,
                    PrimitiveTier::MetaCognitive => TaskType::MetaCognitive,
                    PrimitiveTier::Consciousness => TaskType::MetaCognitive,
                    _ => TaskType::General,
                };
                self.selector.record_observation(name, task_type, penalty);
            }
        }

        // Track routing usage to understand primitive utilization patterns
        // Primitives that are heavily used in routing but underperform need attention
        for (name, usage_count) in &feedback.routing_primitive_usage {
            if *usage_count > 10 {
                // High usage: check if in low performers
                let is_underperforming = feedback.low_performers.iter().any(|(n, _)| n == name);

                if is_underperforming {
                    // High usage + low performance = candidate for evolution
                    if let Some(primitive) = prim_system.get(name) {
                        // Schedule evolution for this tier if not already active
                        if self.active_tier.is_none() {
                            self.schedule_tier_evolution(primitive.tier);
                        }
                    }
                }
            }
        }

        // Check if overall architecture phi is below threshold - may need evolution
        if feedback.architecture_phi < 0.3 && self.active_tier.is_none() {
            // Find the worst-performing tier and schedule evolution
            if let Some(worst_tier) = self.find_worst_tier() {
                self.schedule_tier_evolution(worst_tier);
            }
        }
    }

    /// Schedule evolution for a specific tier
    fn schedule_tier_evolution(&mut self, tier: PrimitiveTier) {
        // Only schedule if not already evolving
        if self.active_tier.is_some() {
            return;
        }

        // Mark this tier as the active evolution target
        // The actual evolution will be triggered on next evolution step
        self.active_tier = Some(tier);
    }

    /// Find the worst-performing tier based on selector statistics
    fn find_worst_tier(&self) -> Option<PrimitiveTier> {
        let tier_phi = self.compute_tier_stats();

        tier_phi
            .into_iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(tier, _)| tier)
    }

    /// Generate feedback for architecture evolution.
    ///
    /// This provides the reverse direction of the bidirectional feedback loop,
    /// informing architecture evolution about primitive performance and evolution status.
    pub fn generate_primitive_feedback(&self) -> PrimitiveFeedback {
        // Find newly discovered high-performing primitives from selector stats
        // (primitives that have been used recently and show positive phi)
        let new_primitives: Vec<String> = self
            .selector
            .all_stats()
            .iter()
            .filter(|((_, _), stats)| {
                // Recently discovered: few uses but high mean phi
                stats.use_count >= 3 && stats.use_count <= 10 && stats.mean_phi() > 0.5
            })
            .map(|((name, _), _)| name.clone())
            .collect();

        // Find underperforming primitives that might be deprecated
        let deprecated_primitives: Vec<String> = self
            .selector
            .all_stats()
            .iter()
            .filter(|((_, _), stats)| stats.use_count > 10 && stats.mean_phi() < 0.1)
            .map(|((name, _), _)| name.clone())
            .collect();

        // Compute tier statistics
        let tier_stats = self.compute_tier_stats().into_iter().collect();

        PrimitiveFeedback {
            new_primitives,
            deprecated_primitives,
            tier_stats,
            primitive_fitness: self.get_current_primitive_psi(),
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = EvolutionCoordinator::new(CoordinatorConfig::default());
        assert_eq!(coordinator.generation(), 0);
    }

    #[test]
    fn test_coordinator_step() {
        let mut coordinator = EvolutionCoordinator::new(CoordinatorConfig {
            primitive_evolution_frequency: 2,
            ..Default::default()
        });

        // First step - no primitive evolution
        let result = coordinator.step().unwrap();
        assert!(!result.primitive_evolution_ran);

        // Second step - should run primitive evolution
        let result = coordinator.step().unwrap();
        assert!(result.primitive_evolution_ran);
    }

    #[test]
    fn test_config_default() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.primitive_evolution_frequency, 5);
        assert!(config.cross_tier_evolution);
    }

    #[test]
    fn test_schedule_variants() {
        let interleaved = EvolutionSchedule::Interleaved { frequency: 5 };
        let _plateau = EvolutionSchedule::OnPlateau {
            stall_generations: 10,
        };
        let _parallel = EvolutionSchedule::Parallel;

        // Just verify they can be created
        match interleaved {
            EvolutionSchedule::Interleaved { frequency } => assert_eq!(frequency, 5),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_feedback_structures() {
        let arch_feedback = ArchitectureFeedback {
            high_performers: vec![("BIND".to_string(), 0.8)],
            low_performers: vec![("BAD".to_string(), 0.1)],
            routing_primitive_usage: std::collections::HashMap::new(),
            architecture_phi: 0.75,
        };

        assert_eq!(arch_feedback.high_performers.len(), 1);
    }
}
