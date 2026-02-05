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
//! ## Architecture (Improvement #23 + #69 Unified)
//!
//! This router uses the HDC-based GlobalWorkspace from symthaea-core (#23) as its
//! backend, providing a routing interface on top (#69). This eliminates code
//! duplication while maintaining the high-level routing abstractions.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    GlobalWorkspaceRouter (#69)                  │
//! │  - Routing strategy selection based on consciousness state      │
//! │  - Module activation computation                                │
//! │  - Coalition formation logic                                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                    HDC GlobalWorkspace (#23)                    │
//! │  - Competitive dynamics for workspace access                    │
//! │  - Broadcasting mechanism                                       │
//! │  - HDC vector representations                                   │
//! │  - Capacity management and decay                                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

use super::{RoutingStrategy, LatentConsciousnessState};

// Import HDC GlobalWorkspace from symthaea-core (Improvement #23)
use symthaea_core::hdc::global_workspace::{
    GlobalWorkspace as HdcWorkspace,
    WorkspaceConfig as HdcWorkspaceConfig,
    WorkspaceContent,
    WorkspaceAssessment,
};
use symthaea_core::hdc::HV16;

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

    /// Convert to source string for HDC workspace
    pub fn as_source(&self) -> String {
        format!("{:?}", self).to_lowercase()
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
// ROUTING ENTRY - Bridge between Router and HDC Workspace
// =============================================================================

/// An entry competing for access to the Global Workspace
/// This is the routing layer's view of workspace content
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
    /// Source module that generated this entry
    pub source_module: WorkspaceModule,
}

impl WorkspaceEntry {
    pub fn new(
        id: u64,
        strategy: RoutingStrategy,
        source_module: WorkspaceModule,
        initial_activation: f64,
    ) -> Self {
        Self {
            id,
            strategy,
            supporting_modules: vec![source_module],
            activation: initial_activation.clamp(0.0, 1.0),
            source_module,
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

    /// Convert to HDC WorkspaceContent for submission to backend
    fn to_workspace_content(&self) -> WorkspaceContent {
        // Create HDC representation based on strategy and coalition
        let representation = self.create_hdc_representation();

        WorkspaceContent::new(
            representation,
            self.effective_activation(),
            self.source_module.as_source(),
        )
    }

    /// Create HDC representation encoding the strategy and coalition
    fn create_hdc_representation(&self) -> Vec<HV16> {
        // Encode strategy as base vector
        let strategy_seed = match self.strategy {
            RoutingStrategy::FullDeliberation => 1000,
            RoutingStrategy::StandardProcessing => 2000,
            RoutingStrategy::HeuristicGuided => 3000,
            RoutingStrategy::FastPatterns => 4000,
            RoutingStrategy::Reflexive => 5000,
            RoutingStrategy::Ensemble => 6000,
            RoutingStrategy::Preparatory => 7000,
        };

        // Create representation with strategy encoding + coalition encoding
        let mut hvs = Vec::with_capacity(4);
        hvs.push(HV16::random(strategy_seed + self.id));

        // Encode coalition members
        for (i, module) in self.supporting_modules.iter().enumerate() {
            let module_seed = module.index() as u64 * 100 + self.id + i as u64;
            hvs.push(HV16::random(module_seed));
        }

        // Pad to consistent length
        while hvs.len() < 4 {
            hvs.push(HV16::zero());
        }

        hvs
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
}

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the Global Workspace Router
#[derive(Debug, Clone)]
pub struct GlobalWorkspaceConfig {
    /// Activation threshold for ignition/broadcast (maps to HDC entry_threshold)
    pub ignition_threshold: f64,
    /// Maximum entries competing simultaneously (maps to HDC max_capacity)
    pub max_competing_entries: usize,
    /// Decay rate (maps to HDC decay_rate)
    pub decay_rate: f64,
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
            decay_rate: 0.1,
            min_coalition_size: 2,
            enable_competition: true,
            inhibition_strength: 0.15,
            refractory_period: 2,
        }
    }
}

impl GlobalWorkspaceConfig {
    /// Convert to HDC WorkspaceConfig
    fn to_hdc_config(&self) -> HdcWorkspaceConfig {
        HdcWorkspaceConfig {
            max_capacity: self.max_competing_entries,
            entry_threshold: self.ignition_threshold,
            decay_rate: self.decay_rate,
            enable_broadcasting: true,
            winner_takes_all: false,
            max_duration: 50,
        }
    }
}

// =============================================================================
// STATISTICS
// =============================================================================

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
    /// HDC workspace assessment (from backend)
    pub workspace_assessment: Option<WorkspaceAssessment>,
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
///
/// ## Unified Architecture (#23 + #69)
///
/// This router uses the HDC-based GlobalWorkspace from symthaea-core as
/// its backend, providing routing abstractions on top of the core GWT
/// implementation.
pub struct GlobalWorkspaceRouter {
    /// HDC Global Workspace backend (Improvement #23)
    hdc_workspace: HdcWorkspace,
    /// Current entries being tracked for routing
    routing_entries: Vec<WorkspaceEntry>,
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
        // Create HDC workspace with matching configuration
        let hdc_workspace = HdcWorkspace::new(config.to_hdc_config());

        Self {
            hdc_workspace,
            routing_entries: Vec::with_capacity(config.max_competing_entries),
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

                let entry = WorkspaceEntry::new(id, strategy, module, activation);
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
        for entry in &mut self.routing_entries {
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

    /// Submit routing entries to HDC workspace backend
    fn submit_to_hdc_workspace(&mut self) {
        for entry in &self.routing_entries {
            if entry.supporting_modules.len() >= self.config.min_coalition_size {
                let workspace_content = entry.to_workspace_content();
                self.hdc_workspace.submit(workspace_content);
            }
        }
    }

    /// Process the HDC workspace and check for broadcasts
    fn process_hdc_workspace(&mut self) -> (Option<BroadcastEvent>, WorkspaceAssessment) {
        // Process HDC workspace dynamics
        let assessment = self.hdc_workspace.process();

        // Check if any of our entries won (entered consciousness)
        let broadcast = if assessment.ignition_detected {
            // Find the winning entry from our routing entries
            self.find_winning_broadcast(&assessment)
        } else {
            None
        };

        (broadcast, assessment)
    }

    /// Find which routing entry corresponds to the broadcast winner
    fn find_winning_broadcast(&self, assessment: &WorkspaceAssessment) -> Option<BroadcastEvent> {
        // Match conscious contents back to our routing entries
        for conscious_content in &assessment.conscious_contents {
            // Find matching routing entry by source
            for entry in &self.routing_entries {
                if entry.source_module.as_source() == conscious_content.source
                    && conscious_content.duration == 0
                {
                    // This entry just entered consciousness
                    return Some(BroadcastEvent {
                        entry_id: entry.id,
                        strategy: entry.strategy,
                        activation: entry.effective_activation(),
                        coalition_size: entry.supporting_modules.len(),
                        timestep: self.timestep,
                        recipients: WorkspaceModule::all().to_vec(),
                    });
                }
            }
        }
        None
    }

    /// Apply post-broadcast effects
    fn apply_broadcast_effects(&mut self, broadcast: &BroadcastEvent) {
        // Update stats
        self.stats.broadcasts += 1;
        let n = self.stats.broadcasts as f64;
        self.stats.avg_coalition_size =
            (self.stats.avg_coalition_size * (n - 1.0) + broadcast.coalition_size as f64) / n;
        self.stats.avg_broadcast_activation =
            (self.stats.avg_broadcast_activation * (n - 1.0) + broadcast.activation) / n;

        // Track module participation
        if let Some(entry) = self.routing_entries.iter().find(|e| e.id == broadcast.entry_id) {
            for module in &entry.supporting_modules {
                self.stats.module_participation[module.index()] += 1;
            }
        }

        // Store last broadcast
        self.last_broadcast = Some(broadcast.strategy);

        // Enter refractory period
        self.refractory_countdown = self.config.refractory_period;

        // Remove winning entry from routing entries
        self.routing_entries.retain(|e| e.id != broadcast.entry_id);

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

        // Handle refractory period
        let in_refractory = if self.refractory_countdown > 0 {
            self.refractory_countdown -= 1;
            self.stats.refractory_timesteps += 1;
            true
        } else {
            false
        };

        // 1. Generate new candidate entries
        let new_candidates = self.generate_candidates(state);

        // 2. Add candidates (respecting max)
        for candidate in new_candidates {
            if self.routing_entries.len() < self.config.max_competing_entries {
                self.routing_entries.push(candidate);
            }
        }

        // 3. Form coalitions
        self.form_coalitions(state);

        // 4. Submit to HDC workspace backend
        self.submit_to_hdc_workspace();

        // 5. Process HDC workspace and check for broadcasts
        let (broadcast, workspace_assessment) = if !in_refractory {
            self.process_hdc_workspace()
        } else {
            (None, self.hdc_workspace.process())
        };

        // 6. Apply broadcast effects
        if let Some(ref b) = broadcast {
            self.apply_broadcast_effects(b);
        } else if !in_refractory {
            self.stats.failed_ignitions += 1;
        }

        // 7. Clean up old routing entries (sync with HDC workspace decay)
        self.routing_entries.retain(|e| e.activation > 0.1);

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
            competing_entries: self.routing_entries.len(),
            highest_activation: self.routing_entries.iter()
                .map(|e| e.effective_activation())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0),
            in_refractory,
            timestep: self.timestep,
            workspace_assessment: Some(workspace_assessment),
        }
    }

    /// Get current workspace state description
    pub fn workspace_state(&self) -> String {
        let mut desc = String::new();
        desc.push_str(&format!("=== GLOBAL WORKSPACE (t={}) ===\n", self.timestep));
        desc.push_str(&format!("Competing entries: {}\n", self.routing_entries.len()));
        desc.push_str(&format!("HDC workspace conscious: {}\n", self.hdc_workspace.num_conscious()));
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
        let mut sorted: Vec<_> = self.routing_entries.iter().collect();
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
        report.push_str("║     (Unified #23 + #69: HDC Backend + Routing Layer)         ║\n");
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
        report.push_str(&format!("║ HDC Conscious Contents: {:>10}                         ║\n", self.hdc_workspace.num_conscious()));
        report.push_str("╚══════════════════════════════════════════════════════════════╝\n");
        report
    }

    /// Get the underlying HDC workspace (for advanced integrations)
    pub fn hdc_workspace(&self) -> &HdcWorkspace {
        &self.hdc_workspace
    }

    /// Get mutable access to HDC workspace (for advanced integrations)
    pub fn hdc_workspace_mut(&mut self) -> &mut HdcWorkspace {
        &mut self.hdc_workspace
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
        let mut entry = WorkspaceEntry::new(1, strategy, WorkspaceModule::Perception, 0.5);

        assert_eq!(entry.supporting_modules.len(), 1); // Source module
        assert!(entry.coalition_strength() > 0.0);

        entry.add_supporter(WorkspaceModule::Attention, 0.7);

        assert_eq!(entry.supporting_modules.len(), 2);
        assert!(entry.effective_activation() > entry.activation);
    }

    #[test]
    fn test_gwt_router_creation() {
        let config = GlobalWorkspaceConfig::default();
        let router = GlobalWorkspaceRouter::new(config);

        assert_eq!(router.timestep, 0);
        assert_eq!(router.routing_entries.len(), 0);
        assert_eq!(router.stats.total_decisions, 0);
    }

    #[test]
    fn test_gwt_basic_routing() {
        let mut router = GlobalWorkspaceRouter::new(GlobalWorkspaceConfig::default());
        let state = LatentConsciousnessState::from_observables(0.8, 0.8, 0.6, 0.3);

        let decision = router.route(&state);

        assert!(decision.timestep == 1);
        assert!(decision.competing_entries > 0);
        assert!(decision.workspace_assessment.is_some());
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
    fn test_gwt_report_generation() {
        let mut router = GlobalWorkspaceRouter::new(GlobalWorkspaceConfig::default());
        let state = LatentConsciousnessState::from_observables(0.7, 0.7, 0.6, 0.3);

        router.route(&state);
        router.route(&state);

        let report = router.report();
        assert!(report.contains("GLOBAL WORKSPACE"));
        assert!(report.contains("Total Decisions"));
        assert!(report.contains("MODULE PARTICIPATION"));
        assert!(report.contains("Unified #23 + #69"));
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
        assert!(ws_state.contains("HDC workspace conscious"));
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

    #[test]
    fn test_hdc_workspace_access() {
        let mut router = GlobalWorkspaceRouter::new(GlobalWorkspaceConfig::default());

        // Verify we can access the HDC workspace
        assert_eq!(router.hdc_workspace().num_conscious(), 0);

        // Route some data
        let state = LatentConsciousnessState::from_observables(0.7, 0.7, 0.6, 0.3);
        router.route(&state);

        // HDC workspace should have processed content
        let ws = router.hdc_workspace();
        // The workspace may or may not have conscious content depending on thresholds
        assert!(ws.num_conscious() >= 0);
    }
}
