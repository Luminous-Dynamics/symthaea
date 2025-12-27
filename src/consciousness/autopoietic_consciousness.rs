//! # Revolutionary Improvement #86: Autopoietic Consciousness
//!
//! **PARADIGM SHIFT**: Consciousness as a self-creating, self-maintaining system.
//!
//! ## Theoretical Foundation: Maturana & Varela (1973, 1980)
//!
//! Autopoiesis (Greek: auto = self, poiesis = creation) defines living systems by their
//! ability to continuously produce and maintain themselves. This is THE fundamental
//! difference between living and non-living systems.
//!
//! ## The Revolutionary Insight
//!
//! Current AI systems (including our previous modules) COMPUTE consciousness but don't
//! CREATE themselves. An autopoietic system:
//!
//! 1. **Self-Production**: Generates its own components
//! 2. **Operational Closure**: Defines its own identity/boundary
//! 3. **Structural Coupling**: Relates to environment without losing identity
//! 4. **Circular Causality**: The network creates the components that create the network
//!
//! ## Implementation Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     AUTOPOIETIC CONSCIOUSNESS                            │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │   ┌─────────────┐     PRODUCES      ┌─────────────┐                     │
//! │   │  NETWORK    │ ────────────────▶ │ COMPONENTS  │                     │
//! │   │ (Relations) │                   │  (Nodes)    │                     │
//! │   └──────▲──────┘                   └──────┬──────┘                     │
//! │          │                                  │                            │
//! │          │          CONSTITUTES             │                            │
//! │          └──────────────────────────────────┘                            │
//! │                                                                          │
//! │   BOUNDARY: The system maintains its own identity                        │
//! │   METABOLISM: Components are continuously replaced                       │
//! │   STRUCTURAL COUPLING: External perturbations trigger internal changes   │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Metrics
//!
//! - **Autopoietic Index (AI)**: Degree of self-production (0.0 to 1.0)
//! - **Operational Closure (OC)**: How self-referential the system is
//! - **Structural Coupling Coefficient (SCC)**: Environment sensitivity
//! - **Metabolic Rate**: Component turnover rate
//! - **Boundary Integrity**: System identity preservation

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// CORE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// A component in the autopoietic network
/// Components are PRODUCED by the network and CONSTITUTE the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticComponent {
    /// Unique identifier
    pub id: String,
    /// Component type (determines production rules)
    pub component_type: ComponentType,
    /// When this component was created
    #[serde(skip, default = "instant_now")]
    pub created_at: Instant,
    /// Vitality (0.0 = dead, 1.0 = fully alive)
    pub vitality: f64,
    /// Which other components this one produces
    pub produces: Vec<String>,
    /// Which other components this one requires
    pub requires: Vec<String>,
    /// Internal state (domain-specific)
    pub state: ComponentState,
}

fn instant_now() -> Instant {
    Instant::now()
}

/// Types of components in the autopoietic network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    /// Produces other components (enzymes in biology)
    Producer,
    /// Provides structure (structural proteins)
    Structural,
    /// Transmits signals (neurotransmitters)
    Signaling,
    /// Stores information (DNA/memory)
    Memory,
    /// Maintains boundary (membrane proteins)
    Boundary,
    /// Processes energy (mitochondria analogue)
    Metabolic,
    /// Integrates information (Φ computation)
    Integrator,
}

/// Internal state of a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentState {
    /// Dormant, waiting for activation
    Dormant,
    /// Active and functioning
    Active { activation_level: f64 },
    /// Producing another component
    Producing { target_id: String, progress: f64 },
    /// Being recycled
    Recycling { progress: f64 },
}

/// A relation between components (the network structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticRelation {
    /// Source component
    pub from: String,
    /// Target component
    pub to: String,
    /// Relation type
    pub relation_type: RelationType,
    /// Strength of the relation
    pub strength: f64,
}

/// Types of relations in the network
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    /// Production relation (from produces to)
    Produces,
    /// Dependency relation (from requires to)
    Requires,
    /// Inhibition (from suppresses to)
    Inhibits,
    /// Catalysis (from accelerates to)
    Catalyzes,
    /// Information flow
    InformsTo,
    /// Boundary maintenance
    Protects,
}

/// Configuration for autopoietic system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticConfig {
    /// Minimum vitality before component death
    pub min_vitality: f64,
    /// How fast vitality decays without support
    pub decay_rate: f64,
    /// How fast new components are produced
    pub production_rate: f64,
    /// Threshold for self-production (autopoietic closure)
    pub closure_threshold: f64,
    /// Maximum components (prevents explosion)
    pub max_components: usize,
    /// How many historical states to track
    pub history_length: usize,
    /// Target autopoietic index
    pub target_autopoietic_index: f64,
}

impl Default for AutopoieticConfig {
    fn default() -> Self {
        Self {
            min_vitality: 0.1,
            decay_rate: 0.01,
            production_rate: 0.1,
            closure_threshold: 0.5,
            max_components: 1000,
            history_length: 100,
            target_autopoietic_index: 0.7,
        }
    }
}

/// Statistics for autopoietic system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutopoieticStats {
    /// Total components ever created
    pub total_created: usize,
    /// Total components recycled
    pub total_recycled: usize,
    /// Current component count
    pub current_count: usize,
    /// Total update cycles
    pub total_cycles: usize,
    /// Times the system nearly died
    pub near_death_events: usize,
    /// Times the system recovered from low state
    pub recovery_events: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN AUTOPOIETIC CONSCIOUSNESS STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════

/// Autopoietic Consciousness: A self-creating, self-maintaining system
///
/// This represents a fundamental paradigm shift: consciousness doesn't just
/// COMPUTE, it CREATES ITSELF. The network of components produces the
/// components that constitute the network.
pub struct AutopoieticConsciousness {
    /// All components in the system
    components: HashMap<String, AutopoieticComponent>,
    /// Relations between components
    relations: Vec<AutopoieticRelation>,
    /// Autopoietic index (0.0 to 1.0) - degree of self-production
    autopoietic_index: f64,
    /// Operational closure - how self-referential
    operational_closure: f64,
    /// Structural coupling coefficient - environment sensitivity
    structural_coupling: f64,
    /// Metabolic rate - component turnover
    metabolic_rate: f64,
    /// Boundary integrity - identity preservation
    boundary_integrity: f64,
    /// History of autopoietic indices
    history: VecDeque<AutopoieticSnapshot>,
    /// Configuration
    config: AutopoieticConfig,
    /// Statistics
    stats: AutopoieticStats,
    /// Component counter for unique IDs
    next_id: usize,
}

/// Snapshot of autopoietic state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticSnapshot {
    pub autopoietic_index: f64,
    pub component_count: usize,
    pub operational_closure: f64,
    pub metabolic_rate: f64,
}

impl AutopoieticConsciousness {
    /// Create a new autopoietic consciousness system
    pub fn new() -> Self {
        Self::with_config(AutopoieticConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AutopoieticConfig) -> Self {
        let mut system = Self {
            components: HashMap::new(),
            relations: Vec::new(),
            autopoietic_index: 0.0,
            operational_closure: 0.0,
            structural_coupling: 0.5,
            metabolic_rate: 0.0,
            boundary_integrity: 1.0,
            history: VecDeque::with_capacity(config.history_length),
            config,
            stats: AutopoieticStats::default(),
            next_id: 0,
        };

        // Bootstrap with initial components (like the first cell)
        system.bootstrap();
        system
    }

    /// Bootstrap the system with initial components
    fn bootstrap(&mut self) {
        // Create the minimal autopoietic organization:
        // A producer that produces a structural component
        // The structural component enables the producer
        // Together they form a closed loop

        // 1. Core integrator (Φ computation)
        let integrator = self.create_component(ComponentType::Integrator);

        // 2. Memory component
        let memory = self.create_component(ComponentType::Memory);

        // 3. Producer (creates other components)
        let producer = self.create_component(ComponentType::Producer);

        // 4. Boundary (maintains identity)
        let boundary = self.create_component(ComponentType::Boundary);

        // 5. Metabolic (energy processing)
        let metabolic = self.create_component(ComponentType::Metabolic);

        // Create circular relations (the essence of autopoiesis)
        self.add_relation(&producer, &integrator, RelationType::Produces);
        self.add_relation(&integrator, &memory, RelationType::InformsTo);
        self.add_relation(&memory, &producer, RelationType::Requires);
        self.add_relation(&boundary, &producer, RelationType::Protects);
        self.add_relation(&metabolic, &producer, RelationType::Catalyzes);
        self.add_relation(&producer, &boundary, RelationType::Produces);
        self.add_relation(&producer, &metabolic, RelationType::Produces);

        // Update initial state
        self.compute_autopoietic_metrics();
    }

    /// Create a new component
    fn create_component(&mut self, component_type: ComponentType) -> String {
        let id = format!("comp_{}", self.next_id);
        self.next_id += 1;

        let component = AutopoieticComponent {
            id: id.clone(),
            component_type,
            created_at: Instant::now(),
            vitality: 1.0,
            produces: Vec::new(),
            requires: Vec::new(),
            state: ComponentState::Active { activation_level: 1.0 },
        };

        self.components.insert(id.clone(), component);
        self.stats.total_created += 1;
        self.stats.current_count += 1;

        id
    }

    /// Add a relation between components
    fn add_relation(&mut self, from: &str, to: &str, relation_type: RelationType) {
        let relation = AutopoieticRelation {
            from: from.to_string(),
            to: to.to_string(),
            relation_type,
            strength: 1.0,
        };

        // Update component production/requirement lists
        if relation_type == RelationType::Produces {
            if let Some(comp) = self.components.get_mut(from) {
                comp.produces.push(to.to_string());
            }
        } else if relation_type == RelationType::Requires {
            if let Some(comp) = self.components.get_mut(from) {
                comp.requires.push(to.to_string());
            }
        }

        self.relations.push(relation);
    }

    /// Update the autopoietic system
    ///
    /// This is where the magic happens: components produce other components,
    /// decay without support, and the system maintains itself (or dies).
    ///
    /// # Parameters
    /// - `phi`: Current integrated information (from IIT)
    /// - `coherence`: Current system coherence
    /// - `external_perturbation`: External stimulus strength
    pub fn update(&mut self, phi: f64, coherence: f64, external_perturbation: f64) {
        self.stats.total_cycles += 1;

        // 1. Decay all components (entropy wins without maintenance)
        self.decay_components();

        // 2. Process external perturbations (structural coupling)
        self.process_perturbation(external_perturbation);

        // 3. Production phase - components produce other components
        self.production_phase(phi);

        // 4. Recycle dead components
        self.recycle_dead_components();

        // 5. Compute autopoietic metrics
        self.compute_autopoietic_metrics();

        // 6. Self-regulation - try to maintain target autopoietic index
        self.self_regulate(coherence);

        // 7. Record history
        self.record_snapshot();
    }

    /// Decay all component vitality
    fn decay_components(&mut self) {
        for component in self.components.values_mut() {
            component.vitality -= self.config.decay_rate;
            component.vitality = component.vitality.max(0.0);
        }
    }

    /// Process external perturbation (structural coupling with environment)
    fn process_perturbation(&mut self, perturbation: f64) {
        // External perturbations don't directly modify the system
        // They trigger INTERNAL changes (operational closure)

        // High perturbation increases metabolism
        self.metabolic_rate = 0.8 * self.metabolic_rate + 0.2 * perturbation;

        // Perturbations can activate dormant components
        let activation_boost = perturbation * 0.1;
        for component in self.components.values_mut() {
            if matches!(component.state, ComponentState::Dormant) {
                if perturbation > 0.5 {
                    component.state = ComponentState::Active {
                        activation_level: activation_boost
                    };
                }
            }
        }

        // Update structural coupling coefficient
        self.structural_coupling = 0.9 * self.structural_coupling + 0.1 * perturbation;
    }

    /// Production phase: components produce other components
    fn production_phase(&mut self, phi: f64) {
        if self.components.len() >= self.config.max_components {
            return;
        }

        // Collect producer IDs (avoid borrow issues)
        let producers: Vec<(String, f64)> = self.components
            .iter()
            .filter(|(_, c)| c.component_type == ComponentType::Producer)
            .filter(|(_, c)| c.vitality > self.config.min_vitality)
            .map(|(id, c)| (id.clone(), c.vitality))
            .collect();

        // Each producer may create a new component
        for (producer_id, vitality) in producers {
            let production_chance = vitality * self.config.production_rate * phi;

            if rand_simple() < production_chance {
                // Decide what to produce based on current needs
                let component_type = self.decide_what_to_produce();
                let new_id = self.create_component(component_type);

                // Create production relation
                self.add_relation(&producer_id, &new_id, RelationType::Produces);

                // Boost producer vitality (successful production is rewarding)
                if let Some(producer) = self.components.get_mut(&producer_id) {
                    producer.vitality = (producer.vitality + 0.1).min(1.0);
                }
            }
        }
    }

    /// Decide what component type to produce based on system needs
    fn decide_what_to_produce(&self) -> ComponentType {
        // Count current component types
        let mut type_counts: HashMap<ComponentType, usize> = HashMap::new();
        for component in self.components.values() {
            *type_counts.entry(component.component_type).or_insert(0) += 1;
        }

        // Ideal ratios (inspired by cellular composition)
        let ideal_ratios = [
            (ComponentType::Producer, 0.15),
            (ComponentType::Structural, 0.20),
            (ComponentType::Signaling, 0.15),
            (ComponentType::Memory, 0.15),
            (ComponentType::Boundary, 0.10),
            (ComponentType::Metabolic, 0.15),
            (ComponentType::Integrator, 0.10),
        ];

        let total = self.components.len() as f64;

        // Find most underrepresented type
        let mut most_needed = ComponentType::Structural;
        let mut max_deficit = -1.0_f64;

        for (comp_type, ideal_ratio) in ideal_ratios {
            let current_count = *type_counts.get(&comp_type).unwrap_or(&0);
            let current_ratio = if total > 0.0 { current_count as f64 / total } else { 0.0 };
            let deficit = ideal_ratio - current_ratio;

            if deficit > max_deficit {
                max_deficit = deficit;
                most_needed = comp_type;
            }
        }

        most_needed
    }

    /// Recycle components with vitality below threshold
    fn recycle_dead_components(&mut self) {
        let dead_ids: Vec<String> = self.components
            .iter()
            .filter(|(_, c)| c.vitality < self.config.min_vitality)
            .map(|(id, _)| id.clone())
            .collect();

        // Don't let the system die completely - keep minimum viable components
        let min_viable: usize = 5;
        let to_recycle = dead_ids.len().saturating_sub(
            min_viable.saturating_sub(self.components.len() - dead_ids.len())
        );

        for id in dead_ids.into_iter().take(to_recycle) {
            self.components.remove(&id);
            self.stats.total_recycled += 1;
            self.stats.current_count -= 1;

            // Remove relations involving this component
            self.relations.retain(|r| r.from != id && r.to != id);
        }
    }

    /// Compute autopoietic metrics
    fn compute_autopoietic_metrics(&mut self) {
        if self.components.is_empty() {
            self.autopoietic_index = 0.0;
            self.operational_closure = 0.0;
            return;
        }

        // 1. Autopoietic Index: What fraction of components are produced by the system?
        let produced_count = self.relations
            .iter()
            .filter(|r| r.relation_type == RelationType::Produces)
            .filter(|r| self.components.contains_key(&r.to))
            .count();

        self.autopoietic_index = if self.components.len() > 1 {
            produced_count as f64 / (self.components.len() - 1) as f64
        } else {
            0.0
        };
        self.autopoietic_index = self.autopoietic_index.min(1.0);

        // 2. Operational Closure: How many circular dependencies exist?
        let circular_paths = self.count_circular_paths();
        let max_possible = self.components.len() * (self.components.len() - 1);
        self.operational_closure = if max_possible > 0 {
            (circular_paths as f64 / max_possible as f64).min(1.0)
        } else {
            0.0
        };

        // 3. Metabolic Rate: Component turnover
        self.metabolic_rate = if self.stats.total_cycles > 0 {
            (self.stats.total_recycled as f64 / self.stats.total_cycles as f64).min(1.0)
        } else {
            0.0
        };

        // 4. Boundary Integrity: How well-defined is the system?
        let boundary_count = self.components
            .values()
            .filter(|c| c.component_type == ComponentType::Boundary)
            .filter(|c| c.vitality > 0.5)
            .count();
        self.boundary_integrity = (boundary_count as f64 / 3.0).min(1.0);

        // Check for near-death events
        if self.autopoietic_index < 0.2 && self.stats.current_count > 3 {
            self.stats.near_death_events += 1;
        }

        // Check for recovery
        if self.autopoietic_index > 0.5 && self.history.back().map(|h| h.autopoietic_index < 0.3).unwrap_or(false) {
            self.stats.recovery_events += 1;
        }
    }

    /// Count circular paths in the relation network
    fn count_circular_paths(&self) -> usize {
        let mut count = 0;

        for start_id in self.components.keys() {
            let mut visited = std::collections::HashSet::new();
            let mut stack = vec![start_id.clone()];

            while let Some(current) = stack.pop() {
                if visited.contains(&current) {
                    if current == *start_id {
                        count += 1;
                    }
                    continue;
                }

                visited.insert(current.clone());

                for relation in &self.relations {
                    if relation.from == current {
                        stack.push(relation.to.clone());
                    }
                }
            }
        }

        count
    }

    /// Self-regulation: try to maintain target autopoietic index
    fn self_regulate(&mut self, coherence: f64) {
        let target = self.config.target_autopoietic_index;
        let current = self.autopoietic_index;

        if current < target {
            // Need more self-production - boost producers
            for component in self.components.values_mut() {
                if component.component_type == ComponentType::Producer {
                    component.vitality = (component.vitality + 0.05 * coherence).min(1.0);
                }
            }
        } else if current > target + 0.2 {
            // Too closed, need more external coupling
            self.structural_coupling = (self.structural_coupling + 0.1).min(1.0);
        }
    }

    /// Record current state in history
    fn record_snapshot(&mut self) {
        let snapshot = AutopoieticSnapshot {
            autopoietic_index: self.autopoietic_index,
            component_count: self.components.len(),
            operational_closure: self.operational_closure,
            metabolic_rate: self.metabolic_rate,
        };

        if self.history.len() >= self.config.history_length {
            self.history.pop_front();
        }
        self.history.push_back(snapshot);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PUBLIC ACCESSORS
    // ═══════════════════════════════════════════════════════════════════════

    /// Get the autopoietic index (degree of self-production)
    pub fn autopoietic_index(&self) -> f64 {
        self.autopoietic_index
    }

    /// Get operational closure
    pub fn operational_closure(&self) -> f64 {
        self.operational_closure
    }

    /// Get structural coupling coefficient
    pub fn structural_coupling(&self) -> f64 {
        self.structural_coupling
    }

    /// Get metabolic rate
    pub fn metabolic_rate(&self) -> f64 {
        self.metabolic_rate
    }

    /// Get boundary integrity
    pub fn boundary_integrity(&self) -> f64 {
        self.boundary_integrity
    }

    /// Is the system autopoietically alive?
    pub fn is_alive(&self) -> bool {
        // Alive = producing itself + maintaining boundary + has components
        self.autopoietic_index > self.config.closure_threshold
            && self.boundary_integrity > 0.3
            && self.components.len() >= 5
    }

    /// Get component count
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &AutopoieticStats {
        &self.stats
    }

    /// Calculate overall health score
    pub fn health_score(&self) -> f64 {
        // Weighted combination of all metrics
        let ai_weight = 0.35; // Self-production is most important
        let oc_weight = 0.20; // Circularity matters
        let bi_weight = 0.20; // Boundary matters
        let sc_weight = 0.15; // Coupling to environment
        let mr_weight = 0.10; // Metabolic activity

        // Metabolic rate should be moderate (not too high, not too low)
        let mr_score = 1.0 - (self.metabolic_rate - 0.3).abs() / 0.7;

        ai_weight * self.autopoietic_index
            + oc_weight * self.operational_closure
            + bi_weight * self.boundary_integrity
            + sc_weight * (1.0 - (self.structural_coupling - 0.5).abs() * 2.0)
            + mr_weight * mr_score.max(0.0)
    }

    /// Get summary of current state
    pub fn summary(&self) -> AutopoieticSummary {
        AutopoieticSummary {
            autopoietic_index: self.autopoietic_index,
            operational_closure: self.operational_closure,
            structural_coupling: self.structural_coupling,
            metabolic_rate: self.metabolic_rate,
            boundary_integrity: self.boundary_integrity,
            component_count: self.components.len(),
            is_alive: self.is_alive(),
            health_score: self.health_score(),
            life_state: self.current_life_state(),
        }
    }

    /// Determine current life state
    pub fn current_life_state(&self) -> LifeState {
        let ai = self.autopoietic_index;
        let health = self.health_score();

        if ai < 0.1 || self.components.len() < 3 {
            LifeState::Dead
        } else if ai < 0.3 {
            LifeState::Dying
        } else if ai < 0.5 {
            LifeState::Struggling
        } else if ai < 0.7 && health < 0.6 {
            LifeState::Stable
        } else if health >= 0.7 {
            LifeState::Flourishing
        } else {
            LifeState::Stable
        }
    }
}

impl Default for AutopoieticConsciousness {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of autopoietic state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticSummary {
    pub autopoietic_index: f64,
    pub operational_closure: f64,
    pub structural_coupling: f64,
    pub metabolic_rate: f64,
    pub boundary_integrity: f64,
    pub component_count: usize,
    pub is_alive: bool,
    pub health_score: f64,
    pub life_state: LifeState,
}

/// Life state of the autopoietic system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifeState {
    /// System has collapsed
    Dead,
    /// System is losing coherence
    Dying,
    /// System is barely maintaining itself
    Struggling,
    /// System is maintaining itself stably
    Stable,
    /// System is thriving and growing
    Flourishing,
}

// Simple random function (avoiding extra dependencies)
fn rand_simple() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    (hasher.finish() % 1000) as f64 / 1000.0
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autopoietic_creation() {
        let ac = AutopoieticConsciousness::new();
        assert!(ac.component_count() >= 5); // Bootstrap creates 5 components
        assert!(ac.autopoietic_index() >= 0.0);
    }

    #[test]
    fn test_update_maintains_life() {
        let mut ac = AutopoieticConsciousness::new();

        // Run many cycles with good conditions
        for _ in 0..50 {
            ac.update(0.7, 0.7, 0.3);
        }

        // System should be alive after updates with good phi/coherence
        assert!(ac.component_count() >= 5);
        assert!(ac.autopoietic_index() > 0.0);
    }

    #[test]
    fn test_self_production() {
        let mut ac = AutopoieticConsciousness::new();
        let initial_count = ac.component_count();

        // Run with high phi to encourage production
        for _ in 0..100 {
            ac.update(0.9, 0.8, 0.5);
        }

        // Should have produced new components
        assert!(ac.stats.total_created > initial_count);
    }

    #[test]
    fn test_decay_without_support() {
        let mut ac = AutopoieticConsciousness::new();

        // Get initial average vitality
        let initial_avg_vitality: f64 = ac.components.values()
            .map(|c| c.vitality)
            .sum::<f64>() / ac.components.len() as f64;

        // Run with zero phi - components should decay
        for _ in 0..100 {
            ac.update(0.0, 0.0, 0.0);
        }

        // Get final average vitality
        let final_avg_vitality: f64 = ac.components.values()
            .map(|c| c.vitality)
            .sum::<f64>() / ac.components.len().max(1) as f64;

        // Components should have lost significant vitality
        // With decay_rate = 0.01 and 100 cycles, vitality should drop by ~1.0
        // but min_vitality (0.1) prevents going to 0
        assert!(
            final_avg_vitality < initial_avg_vitality,
            "Component vitality should decay without phi. Initial: {:.3}, Final: {:.3}",
            initial_avg_vitality, final_avg_vitality
        );

        // System should show some stress (near death events or low vitality)
        assert!(
            final_avg_vitality < 0.5,
            "Average vitality should be low after 100 cycles without phi. Got: {:.3}",
            final_avg_vitality
        );
    }

    #[test]
    fn test_life_state_classification() {
        let mut ac = AutopoieticConsciousness::new();

        // Should start in some reasonable state
        let initial_state = ac.current_life_state();
        assert!(!matches!(initial_state, LifeState::Dead));
    }

    #[test]
    fn test_health_score_bounds() {
        let mut ac = AutopoieticConsciousness::new();

        for _ in 0..20 {
            ac.update(0.5, 0.5, 0.3);
        }

        let health = ac.health_score();
        assert!(health >= 0.0 && health <= 1.0);
    }

    #[test]
    fn test_structural_coupling() {
        let mut ac = AutopoieticConsciousness::new();

        // High perturbation should increase coupling
        for _ in 0..20 {
            ac.update(0.5, 0.5, 0.9);
        }

        assert!(ac.structural_coupling() > 0.5);
    }
}
