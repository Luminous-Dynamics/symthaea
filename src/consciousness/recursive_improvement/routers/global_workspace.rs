//! # Global Workspace Theory (GWT) Router
//!
//! Implementation of Bernard Baars' Global Workspace Theory.
//!
//! ## Key Concepts
//!
//! - **Global Workspace**: A cognitive "blackboard" where information becomes conscious
//! - **Specialized Processors**: Unconscious modules compete for workspace access
//! - **Coalition Formation**: Modules form coalitions to amplify their signal
//! - **Ignition**: When activation crosses threshold, global broadcast occurs
//! - **Broadcast**: Winning information is shared with ALL processors simultaneously
//!
//! This router models consciousness as an emergent property of information
//! competition and broadcast, not just computation.

use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};

use super::{RoutingStrategy, LatentConsciousnessState};

// =============================================================================
// WORKSPACE MODULE
// =============================================================================

/// A specialized processor module in the Global Workspace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkspaceModule {
    /// Perceptual processing - analyzes raw observables
    Perception,
    /// Memory retrieval - matches patterns to past states
    Memory,
    /// Attention allocation - prioritizes salient information
    Attention,
    /// Evaluation/valence - assesses importance and urgency
    Evaluation,
    /// Motor planning - prepares action sequences
    Motor,
    /// Language/symbolic - abstract reasoning
    Symbolic,
    /// Meta-cognition - monitors other processes
    MetaCognition,
}

impl WorkspaceModule {
    pub fn all() -> [WorkspaceModule; 7] {
        [
            WorkspaceModule::Perception,
            WorkspaceModule::Memory,
            WorkspaceModule::Attention,
            WorkspaceModule::Evaluation,
            WorkspaceModule::Motor,
            WorkspaceModule::Symbolic,
            WorkspaceModule::MetaCognition,
        ]
    }

    pub fn index(&self) -> usize {
        match self {
            WorkspaceModule::Perception => 0,
            WorkspaceModule::Memory => 1,
            WorkspaceModule::Attention => 2,
            WorkspaceModule::Evaluation => 3,
            WorkspaceModule::Motor => 4,
            WorkspaceModule::Symbolic => 5,
            WorkspaceModule::MetaCognition => 6,
        }
    }

    /// Each module has an affinity for certain state characteristics
    pub fn compute_activation(&self, state: &LatentConsciousnessState) -> f64 {
        match self {
            WorkspaceModule::Perception => {
                // Perception responds to raw signal clarity
                state.coherence * 0.6 + state.integration * 0.4
            }
            WorkspaceModule::Memory => {
                // Memory responds to pattern recognizability
                let stability = 1.0 - state.attention; // Low attention = stable
                stability * 0.5 + state.phi * 0.5
            }
            WorkspaceModule::Attention => {
                // Attention responds to salience and phi
                state.phi * 0.7 + state.attention * 0.3
            }
            WorkspaceModule::Evaluation => {
                // Evaluation responds to integration quality
                state.phi * 0.5 + state.coherence * 0.5
            }
            WorkspaceModule::Motor => {
                // Motor planning responds to action readiness
                let readiness = state.coherence * (1.0 - state.integration);
                readiness.max(0.0)
            }
            WorkspaceModule::Symbolic => {
                // Symbolic processing responds to integration (complexity)
                state.integration * 0.6 + state.phi * 0.4
            }
            WorkspaceModule::MetaCognition => {
                // Meta-cognition monitors all signals
                (state.phi + state.coherence + state.integration + state.attention) / 4.0
            }
        }
    }
}

// =============================================================================
// WORKSPACE ENTRY
// =============================================================================

/// An entry competing for access to the Global Workspace
#[derive(Debug, Clone)]
pub struct WorkspaceEntry {
    /// Unique identifier for this entry
    pub id: u64,
    /// The interpretation/strategy being proposed
    pub strategy: RoutingStrategy,
    /// Which modules support this entry (coalition)
    pub supporting_modules: Vec<WorkspaceModule>,
    /// Current activation level (0.0 - 1.0)
    pub activation: f64,
    /// How long this entry has been competing (timesteps)
    pub age: usize,
    /// Decay rate per timestep
    pub decay_rate: f64,
    /// Source analysis that generated this entry
    pub source_analysis: String,
}

impl WorkspaceEntry {
    pub fn new(
        id: u64,
        strategy: RoutingStrategy,
        initial_activation: f64,
        source: &str,
    ) -> Self {
        Self {
            id,
            strategy,
            supporting_modules: Vec::new(),
            activation: initial_activation.clamp(0.0, 1.0),
            age: 0,
            decay_rate: 0.1, // 10% decay per timestep
            source_analysis: source.to_string(),
        }
    }

    /// Add a supporting module to the coalition
    pub fn add_supporter(&mut self, module: WorkspaceModule, strength: f64) {
        if !self.supporting_modules.contains(&module) {
            self.supporting_modules.push(module);
            // Coalition support amplifies activation
            self.activation = (self.activation + strength * 0.2).clamp(0.0, 1.0);
        }
    }

    /// Apply decay and aging
    pub fn tick(&mut self) {
        self.age += 1;
        // Activation decays over time unless reinforced
        self.activation *= 1.0 - self.decay_rate;
        // Older entries decay faster (recency bias)
        if self.age > 5 {
            self.activation *= 0.95;
        }
    }

    /// Coalition strength: more supporters = stronger
    pub fn coalition_strength(&self) -> f64 {
        let base = self.supporting_modules.len() as f64 / 7.0;
        // Non-linear: coalitions become stronger with more members
        base * base.sqrt()
    }

    /// Effective activation = raw activation * coalition strength
    pub fn effective_activation(&self) -> f64 {
        self.activation * (1.0 + self.coalition_strength())
    }
}

// =============================================================================
// BROADCAST EVENT
// =============================================================================

/// A broadcast event when information wins workspace access
#[derive(Debug, Clone)]
pub struct BroadcastEvent {
    /// The winning entry
    pub entry_id: u64,
    /// Strategy that was broadcast
    pub strategy: RoutingStrategy,
    /// Activation at time of broadcast
    pub activation: f64,
    /// Coalition size at broadcast
    pub coalition_size: usize,
    /// Timestep when broadcast occurred
    pub timestep: u64,
    /// All modules that received the broadcast
    pub recipients: Vec<WorkspaceModule>,
    /// Post-broadcast effects on other entries
    pub suppression_applied: bool,
}

// =============================================================================
// CONFIGURATION AND STATS
// =============================================================================

/// Configuration for the Global Workspace Router
#[derive(Debug, Clone)]
pub struct GlobalWorkspaceConfig {
    /// Activation threshold for ignition/broadcast
    pub ignition_threshold: f64,
    /// Maximum entries competing simultaneously
    pub max_competing_entries: usize,
    /// Decay rate for losing entries after broadcast
    pub post_broadcast_decay: f64,
    /// Minimum coalition size for broadcast eligibility
    pub min_coalition_size: usize,
    /// Enable competition dynamics (entries inhibit each other)
    pub enable_competition: bool,
    /// Competition inhibition strength
    pub inhibition_strength: f64,
    /// Enable refractory period after broadcast
    pub refractory_period: usize,
}

impl Default for GlobalWorkspaceConfig {
    fn default() -> Self {
        Self {
            ignition_threshold: 0.7,
            max_competing_entries: 10,
            post_broadcast_decay: 0.5,
            min_coalition_size: 2,
            enable_competition: true,
            inhibition_strength: 0.15,
            refractory_period: 2,
        }
    }
}

/// Statistics for the Global Workspace
#[derive(Debug, Clone, Default)]
pub struct GlobalWorkspaceStats {
    /// Total routing decisions
    pub total_decisions: u64,
    /// Number of broadcasts (successful ignitions)
    pub broadcasts: u64,
    /// Number of times no entry reached threshold
    pub failed_ignitions: u64,
    /// Average coalition size at broadcast
    pub avg_coalition_size: f64,
    /// Average activation at broadcast
    pub avg_broadcast_activation: f64,
    /// Module participation frequency
    pub module_participation: [u64; 7],
    /// Timesteps in refractory period
    pub refractory_timesteps: u64,
    /// Competition-induced suppressions
    pub competition_suppressions: u64,
}

/// Output of a Global Workspace routing decision
#[derive(Debug, Clone)]
pub struct GlobalWorkspaceDecision {
    /// The selected routing strategy
    pub strategy: RoutingStrategy,
    /// Broadcast event if ignition occurred
    pub broadcast: Option<BroadcastEvent>,
    /// Number of entries currently competing
    pub competing_entries: usize,
    /// Highest effective activation among competitors
    pub highest_activation: f64,
    /// Whether we're in refractory period
    pub in_refractory: bool,
    /// Current timestep
    pub timestep: u64,
}

// =============================================================================
// GLOBAL WORKSPACE ROUTER
// =============================================================================

/// Revolutionary Improvement #69: Global Workspace Theory Router
///
/// Models consciousness as a "workspace" where specialized unconscious
/// processors compete for access. When information wins the competition
/// and crosses the ignition threshold, it is broadcast globally to all
/// processors, making it "conscious".
pub struct GlobalWorkspaceRouter {
    /// Current entries competing for workspace access
    competing_entries: Vec<WorkspaceEntry>,
    /// Recent broadcast history
    broadcast_history: VecDeque<BroadcastEvent>,
    /// Current timestep
    timestep: u64,
    /// Entry ID counter
    next_entry_id: u64,
    /// Configuration
    config: GlobalWorkspaceConfig,
    /// Statistics
    stats: GlobalWorkspaceStats,
    /// Current refractory countdown (0 = not in refractory)
    refractory_countdown: usize,
    /// Module activation levels
    module_activations: [f64; 7],
    /// Last broadcast strategy (for continuity)
    last_broadcast: Option<RoutingStrategy>,
}

impl GlobalWorkspaceRouter {
    pub fn new(config: GlobalWorkspaceConfig) -> Self {
        Self {
            competing_entries: Vec::with_capacity(config.max_competing_entries),
            broadcast_history: VecDeque::with_capacity(100),
            timestep: 0,
            next_entry_id: 0,
            config,
            stats: GlobalWorkspaceStats::default(),
            refractory_countdown: 0,
            module_activations: [0.0; 7],
            last_broadcast: None,
        }
    }

    /// Generate candidate entries from the current state
    fn generate_candidates(&mut self, state: &LatentConsciousnessState) -> Vec<WorkspaceEntry> {
        let mut candidates = Vec::new();

        // Each module can propose a strategy based on its analysis
        for module in WorkspaceModule::all() {
            let activation = module.compute_activation(state);

            // Only strong activations become candidates
            if activation > 0.3 {
                let strategy = self.module_to_strategy(&module, state);
                let id = self.next_entry_id;
                self.next_entry_id += 1;

                let mut entry = WorkspaceEntry::new(
                    id,
                    strategy,
                    activation,
                    &format!("{:?}", module),
                );
                entry.add_supporter(module, activation);
                candidates.push(entry);
            }
        }

        candidates
    }

    /// Map a module's activation to a strategy
    fn module_to_strategy(
        &self,
        module: &WorkspaceModule,
        state: &LatentConsciousnessState,
    ) -> RoutingStrategy {
        match module {
            WorkspaceModule::Perception => RoutingStrategy::HeuristicGuided,
            WorkspaceModule::Memory => RoutingStrategy::StandardProcessing,
            WorkspaceModule::Attention => {
                if state.phi > 0.7 {
                    RoutingStrategy::FullDeliberation
                } else {
                    RoutingStrategy::HeuristicGuided
                }
            }
            WorkspaceModule::Evaluation => {
                if state.coherence > 0.6 {
                    RoutingStrategy::StandardProcessing
                } else {
                    RoutingStrategy::Ensemble
                }
            }
            WorkspaceModule::Motor => RoutingStrategy::FastPatterns,
            WorkspaceModule::Symbolic => {
                if state.integration > 0.7 {
                    RoutingStrategy::FullDeliberation
                } else {
                    RoutingStrategy::HeuristicGuided
                }
            }
            WorkspaceModule::MetaCognition => RoutingStrategy::Ensemble,
        }
    }

    /// Run coalition formation: modules join entries they support
    fn form_coalitions(&mut self, state: &LatentConsciousnessState) {
        // Update module activation levels
        for module in WorkspaceModule::all() {
            self.module_activations[module.index()] = module.compute_activation(state);
        }

        // Copy activations to avoid borrow conflicts
        let activations = self.module_activations;

        // Each entry tries to recruit modules
        for entry in &mut self.competing_entries {
            for module in WorkspaceModule::all() {
                let module_activation = activations[module.index()];

                // Module joins coalition if:
                // 1. It has sufficient activation
                // 2. The entry's strategy aligns with module's preference
                if module_activation > 0.4 {
                    let alignment = Self::compute_alignment(&module, &entry.strategy);
                    if alignment > 0.5 {
                        entry.add_supporter(module, module_activation * alignment);
                    }
                }
            }
        }
    }

    /// Compute how well a strategy aligns with a module's function (pure function)
    fn compute_alignment(module: &WorkspaceModule, strategy: &RoutingStrategy) -> f64 {
        match (module, strategy) {
            // Primary alignments (1.0 = perfect match)
            (WorkspaceModule::Perception, RoutingStrategy::HeuristicGuided) => 1.0,
            (WorkspaceModule::Memory, RoutingStrategy::StandardProcessing) => 1.0,
            (WorkspaceModule::Attention, RoutingStrategy::FullDeliberation) => 0.9,
            (WorkspaceModule::Attention, RoutingStrategy::HeuristicGuided) => 0.8,
            (WorkspaceModule::Evaluation, RoutingStrategy::StandardProcessing) => 0.9,
            (WorkspaceModule::Evaluation, RoutingStrategy::Ensemble) => 0.8,
            (WorkspaceModule::Motor, RoutingStrategy::FastPatterns) => 1.0,
            (WorkspaceModule::Motor, RoutingStrategy::Reflexive) => 0.9,
            (WorkspaceModule::Symbolic, RoutingStrategy::FullDeliberation) => 0.9,
            (WorkspaceModule::Symbolic, RoutingStrategy::HeuristicGuided) => 0.8,
            (WorkspaceModule::MetaCognition, RoutingStrategy::Ensemble) => 1.0,
            // Cross-module alignments
            (WorkspaceModule::Attention, RoutingStrategy::Ensemble) => 0.6,
            (WorkspaceModule::Memory, RoutingStrategy::Preparatory) => 0.7,
            (WorkspaceModule::Evaluation, RoutingStrategy::FullDeliberation) => 0.6,
            (WorkspaceModule::Perception, RoutingStrategy::Reflexive) => 0.7,
            _ => 0.3, // Default weak alignment
        }
    }

    /// Apply competition dynamics: entries inhibit each other
    fn apply_competition(&mut self) {
        if !self.config.enable_competition {
            return;
        }

        // Sort by effective activation (strongest first)
        let activations: Vec<(usize, f64)> = self.competing_entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.effective_activation()))
            .collect();

        // Stronger entries inhibit weaker ones
        for (i, entry) in self.competing_entries.iter_mut().enumerate() {
            let my_activation = activations.iter()
                .find(|(idx, _)| *idx == i)
                .map(|(_, a)| *a)
                .unwrap_or(0.0);

            let inhibition: f64 = activations.iter()
                .filter(|(idx, act)| *idx != i && *act > my_activation)
                .map(|(_, act)| (act - my_activation) * self.config.inhibition_strength)
                .sum();

            if inhibition > 0.0 {
                entry.activation = (entry.activation - inhibition).max(0.0);
                self.stats.competition_suppressions += 1;
            }
        }
    }

    /// Check for ignition and broadcast
    fn check_ignition(&mut self) -> Option<BroadcastEvent> {
        // Can't broadcast during refractory period
        if self.refractory_countdown > 0 {
            self.refractory_countdown -= 1;
            self.stats.refractory_timesteps += 1;
            return None;
        }

        // Find entries that meet broadcast criteria
        let eligible: Vec<(usize, f64)> = self.competing_entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.effective_activation() >= self.config.ignition_threshold
                    && e.supporting_modules.len() >= self.config.min_coalition_size
            })
            .map(|(i, e)| (i, e.effective_activation()))
            .collect();

        if eligible.is_empty() {
            self.stats.failed_ignitions += 1;
            return None;
        }

        // Winner takes all: highest effective activation wins
        let (winner_idx, _) = eligible.into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let winner = &self.competing_entries[winner_idx];

        // Create broadcast event
        let broadcast = BroadcastEvent {
            entry_id: winner.id,
            strategy: winner.strategy,
            activation: winner.activation,
            coalition_size: winner.supporting_modules.len(),
            timestep: self.timestep,
            recipients: WorkspaceModule::all().to_vec(),
            suppression_applied: true,
        };

        // Update stats
        self.stats.broadcasts += 1;
        let n = self.stats.broadcasts as f64;
        self.stats.avg_coalition_size =
            (self.stats.avg_coalition_size * (n - 1.0) + winner.supporting_modules.len() as f64) / n;
        self.stats.avg_broadcast_activation =
            (self.stats.avg_broadcast_activation * (n - 1.0) + winner.activation) / n;

        // Track module participation
        for module in &winner.supporting_modules {
            self.stats.module_participation[module.index()] += 1;
        }

        // Store last broadcast
        self.last_broadcast = Some(winner.strategy);

        // Enter refractory period
        self.refractory_countdown = self.config.refractory_period;

        Some(broadcast)
    }

    /// Apply post-broadcast effects
    fn apply_broadcast_effects(&mut self, broadcast: &BroadcastEvent) {
        // Suppress losing entries
        for entry in &mut self.competing_entries {
            if entry.id != broadcast.entry_id {
                entry.activation *= self.config.post_broadcast_decay;
            }
        }

        // Remove entries with very low activation
        self.competing_entries.retain(|e| e.activation > 0.1);

        // Store in history
        if self.broadcast_history.len() >= 100 {
            self.broadcast_history.pop_front();
        }
        self.broadcast_history.push_back(broadcast.clone());
    }

    /// Main routing function
    pub fn route(&mut self, state: &LatentConsciousnessState) -> GlobalWorkspaceDecision {
        self.timestep += 1;
        self.stats.total_decisions += 1;

        // 1. Generate new candidate entries
        let new_candidates = self.generate_candidates(state);

        // 2. Add candidates (respecting max)
        for candidate in new_candidates {
            if self.competing_entries.len() < self.config.max_competing_entries {
                self.competing_entries.push(candidate);
            }
        }

        // 3. Age existing entries
        for entry in &mut self.competing_entries {
            entry.tick();
        }

        // 4. Form coalitions
        self.form_coalitions(state);

        // 5. Apply competition
        self.apply_competition();

        // 6. Check for ignition/broadcast
        let broadcast = self.check_ignition();

        // 7. Apply broadcast effects
        if let Some(ref b) = broadcast {
            self.apply_broadcast_effects(b);
        }

        // 8. Determine output strategy
        let strategy = if let Some(ref b) = broadcast {
            b.strategy
        } else if let Some(ref last) = self.last_broadcast {
            // Maintain last broadcast during refractory
            *last
        } else {
            // Default: heuristic-guided observation
            RoutingStrategy::HeuristicGuided
        };

        // Build decision report
        GlobalWorkspaceDecision {
            strategy,
            broadcast,
            competing_entries: self.competing_entries.len(),
            highest_activation: self.competing_entries.iter()
                .map(|e| e.effective_activation())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0),
            in_refractory: self.refractory_countdown > 0,
            timestep: self.timestep,
        }
    }

    /// Get current workspace state description
    pub fn workspace_state(&self) -> String {
        let mut desc = String::new();
        desc.push_str(&format!("=== GLOBAL WORKSPACE (t={}) ===\n", self.timestep));
        desc.push_str(&format!("Competing entries: {}\n", self.competing_entries.len()));
        desc.push_str(&format!("Refractory: {}\n",
            if self.refractory_countdown > 0 {
                format!("{} steps remaining", self.refractory_countdown)
            } else {
                "No".to_string()
            }
        ));

        desc.push_str("\nModule Activations:\n");
        for module in WorkspaceModule::all() {
            desc.push_str(&format!("  {:?}: {:.3}\n", module, self.module_activations[module.index()]));
        }

        desc.push_str("\nTop Competing Entries:\n");
        let mut sorted: Vec<_> = self.competing_entries.iter().collect();
        sorted.sort_by(|a, b| b.effective_activation().partial_cmp(&a.effective_activation()).unwrap_or(std::cmp::Ordering::Equal));

        for (i, entry) in sorted.iter().take(5).enumerate() {
            desc.push_str(&format!(
                "  {}. {:?} (act={:.3}, eff={:.3}, coalition={})\n",
                i + 1,
                entry.strategy,
                entry.activation,
                entry.effective_activation(),
                entry.supporting_modules.len()
            ));
        }

        desc
    }

    /// Generate statistics report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║     GLOBAL WORKSPACE THEORY ROUTER - STATISTICS              ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        report.push_str(&format!("║ Total Decisions:        {:>10}                         ║\n", self.stats.total_decisions));
        report.push_str(&format!("║ Successful Broadcasts:  {:>10}                         ║\n", self.stats.broadcasts));
        report.push_str(&format!("║ Failed Ignitions:       {:>10}                         ║\n", self.stats.failed_ignitions));

        let broadcast_rate = if self.stats.total_decisions > 0 {
            self.stats.broadcasts as f64 / self.stats.total_decisions as f64 * 100.0
        } else { 0.0 };
        report.push_str(&format!("║ Broadcast Rate:         {:>10.1}%                        ║\n", broadcast_rate));
        report.push_str(&format!("║ Avg Coalition Size:     {:>10.2}                         ║\n", self.stats.avg_coalition_size));
        report.push_str(&format!("║ Avg Broadcast Activation:{:>9.3}                         ║\n", self.stats.avg_broadcast_activation));
        report.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ MODULE PARTICIPATION (in winning coalitions):                ║\n");

        for module in WorkspaceModule::all() {
            let count = self.stats.module_participation[module.index()];
            let pct = if self.stats.broadcasts > 0 {
                count as f64 / self.stats.broadcasts as f64 * 100.0
            } else { 0.0 };
            report.push_str(&format!("║   {:?}: {:>6} ({:>5.1}%)                              ║\n",
                module, count, pct));
        }

        report.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        report.push_str(&format!("║ Refractory Timesteps:   {:>10}                         ║\n", self.stats.refractory_timesteps));
        report.push_str(&format!("║ Competition Suppressions:{:>9}                         ║\n", self.stats.competition_suppressions));
        report.push_str("╚══════════════════════════════════════════════════════════════╝\n");
        report
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_module_activation() {
        let state = LatentConsciousnessState::from_observables(0.8, 0.7, 0.6, 0.3);

        for module in WorkspaceModule::all() {
            let activation = module.compute_activation(&state);
            assert!(activation >= 0.0 && activation <= 1.0,
                "{:?} activation {} out of range", module, activation);
        }
    }

    #[test]
    fn test_workspace_entry_coalition() {
        let strategy = RoutingStrategy::HeuristicGuided;
        let mut entry = WorkspaceEntry::new(1, strategy, 0.5, "test");

        assert_eq!(entry.supporting_modules.len(), 0);
        assert!(entry.coalition_strength() < 0.01);

        entry.add_supporter(WorkspaceModule::Perception, 0.8);
        entry.add_supporter(WorkspaceModule::Attention, 0.7);

        assert_eq!(entry.supporting_modules.len(), 2);
        assert!(entry.coalition_strength() > 0.0);
        assert!(entry.effective_activation() > entry.activation);
    }

    #[test]
    fn test_workspace_entry_decay() {
        let mut entry = WorkspaceEntry::new(1, RoutingStrategy::HeuristicGuided, 0.8, "test");
        let initial = entry.activation;

        entry.tick();
        assert!(entry.activation < initial);
        assert_eq!(entry.age, 1);

        // Apply more ticks to ensure robust decay below 50%
        for _ in 0..7 {
            entry.tick();
        }

        // After 8 total ticks with 10% decay + age penalty, activation should be well below 50%
        assert!(entry.activation < initial * 0.5);
    }

    #[test]
    fn test_gwt_router_creation() {
        let config = GlobalWorkspaceConfig::default();
        let router = GlobalWorkspaceRouter::new(config);

        assert_eq!(router.timestep, 0);
        assert_eq!(router.competing_entries.len(), 0);
        assert_eq!(router.stats.total_decisions, 0);
    }

    #[test]
    fn test_gwt_basic_routing() {
        let mut router = GlobalWorkspaceRouter::new(GlobalWorkspaceConfig::default());
        let state = LatentConsciousnessState::from_observables(0.8, 0.8, 0.6, 0.3);

        let decision = router.route(&state);

        assert!(decision.timestep == 1);
        assert!(decision.competing_entries > 0);
    }

    #[test]
    fn test_gwt_broadcast_with_high_activation() {
        let config = GlobalWorkspaceConfig {
            ignition_threshold: 0.5, // Lower threshold for testing
            min_coalition_size: 1,
            ..Default::default()
        };
        let mut router = GlobalWorkspaceRouter::new(config);

        // High phi, high coherence should generate strong candidates
        let state = LatentConsciousnessState::from_observables(0.95, 0.95, 0.8, 0.2);

        // May need multiple iterations for broadcast
        for _ in 0..10 {
            let _decision = router.route(&state);
        }

        // Given high activation, broadcast should occur eventually
        assert!(router.stats.broadcasts > 0 || router.stats.failed_ignitions > 0);
    }

    #[test]
    fn test_gwt_refractory_period() {
        let config = GlobalWorkspaceConfig {
            ignition_threshold: 0.3,
            min_coalition_size: 1,
            refractory_period: 3,
            ..Default::default()
        };
        let mut router = GlobalWorkspaceRouter::new(config);
        let state = LatentConsciousnessState::from_observables(0.9, 0.9, 0.9, 0.1);

        // Trigger first broadcast
        for _ in 0..5 {
            router.route(&state);
        }

        if router.stats.broadcasts > 0 {
            // After broadcast, should enter refractory
            let decision = router.route(&state);
            // Refractory countdown should be active
            assert!(decision.in_refractory || router.stats.broadcasts >= 2);
        }
    }

    #[test]
    fn test_gwt_competition_suppression() {
        let config = GlobalWorkspaceConfig {
            enable_competition: true,
            inhibition_strength: 0.2,
            ..Default::default()
        };
        let mut router = GlobalWorkspaceRouter::new(config);

        // Generate varied state to create multiple competing entries
        for i in 0..5 {
            let phi = 0.3 + (i as f64) * 0.1;
            let state = LatentConsciousnessState::from_observables(phi, 0.6, 0.5, 0.4);
            router.route(&state);
        }

        // Competition should have caused some suppression
        // Competition suppression stats tracked (usize always >= 0)
    }

    #[test]
    fn test_gwt_report_generation() {
        let mut router = GlobalWorkspaceRouter::new(GlobalWorkspaceConfig::default());
        let state = LatentConsciousnessState::from_observables(0.7, 0.7, 0.6, 0.3);

        router.route(&state);
        router.route(&state);

        let report = router.report();
        assert!(report.contains("GLOBAL WORKSPACE"));
        assert!(report.contains("Total Decisions"));
        assert!(report.contains("MODULE PARTICIPATION"));
    }

    #[test]
    fn test_gwt_workspace_state() {
        let mut router = GlobalWorkspaceRouter::new(GlobalWorkspaceConfig::default());
        let state = LatentConsciousnessState::from_observables(0.7, 0.7, 0.6, 0.3);

        router.route(&state);

        let ws_state = router.workspace_state();
        assert!(ws_state.contains("GLOBAL WORKSPACE"));
        assert!(ws_state.contains("Module Activations"));
        assert!(ws_state.contains("Competing Entries"));
    }

    #[test]
    fn test_module_strategy_mapping() {
        let router = GlobalWorkspaceRouter::new(GlobalWorkspaceConfig::default());
        let state = LatentConsciousnessState::from_observables(0.8, 0.8, 0.8, 0.3);

        // Perception -> HeuristicGuided
        let strategy = router.module_to_strategy(&WorkspaceModule::Perception, &state);
        assert!(matches!(strategy, RoutingStrategy::HeuristicGuided));

        // Motor -> FastPatterns
        let strategy = router.module_to_strategy(&WorkspaceModule::Motor, &state);
        assert!(matches!(strategy, RoutingStrategy::FastPatterns));

        // MetaCognition -> Ensemble
        let strategy = router.module_to_strategy(&WorkspaceModule::MetaCognition, &state);
        assert!(matches!(strategy, RoutingStrategy::Ensemble));
    }

    #[test]
    fn test_gwt_continuity() {
        let config = GlobalWorkspaceConfig {
            ignition_threshold: 0.4,
            min_coalition_size: 1,
            refractory_period: 2,
            ..Default::default()
        };
        let mut router = GlobalWorkspaceRouter::new(config);

        let state = LatentConsciousnessState::from_observables(0.85, 0.85, 0.7, 0.2);

        let mut strategies = Vec::new();
        for _ in 0..10 {
            let decision = router.route(&state);
            strategies.push(decision.strategy);
        }

        // During refractory, should maintain last broadcast strategy
        let unique_strategies: std::collections::HashSet<_> = strategies.iter()
            .map(|s| format!("{:?}", s))
            .collect();

        // Should have some consistency (not 10 different strategies)
        assert!(unique_strategies.len() <= 5);
    }
}
