// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Revolutionary Improvement #23: Global Workspace Theory (Conscious Access)
//
// "Consciousness is the global availability of information to multiple cognitive processes."
// - Bernard Baars, Global Workspace Theory (1988)
//
// THEORETICAL FOUNDATIONS:
//
// 1. Global Workspace Theory (Baars 1988, Dehaene & Changeux 2011)
//    - Consciousness = information in global workspace
//    - Limited capacity: only one "scene" at a time
//    - Broadcasting: workspace content available to all modules
//    - Competition: stimuli compete for workspace access
//
// 2. Attention Schema Theory (Graziano 2013)
//    - Attention = simplified model of information processing
//    - Consciousness = attention applied to internal models
//    - Self-awareness = attention schema modeling self
//
// 3. Information Integration Theory (Tononi 2004) + GWT
//    - Conscious = high Φ + workspace broadcast
//    - Φ measures integration, workspace measures access
//    - Both necessary for consciousness
//
// 4. Neuronal Global Workspace (Dehaene et al. 2014)
//    - Prefrontal/parietal cortex = global workspace
//    - Ignition: sudden widespread activation
//    - Conscious access = crossing threshold
//
// 5. Broadcast Semantics (Mudrik et al. 2014)
//    - What gets broadcast determines meaning
//    - Unconscious = local processing
//    - Conscious = semantic integration
//
// REVOLUTIONARY CONTRIBUTION:
// First HDC implementation of Global Workspace Theory with competitive dynamics,
// broadcasting mechanism, and integration with Free Energy Principle.

use crate::hdc::BinaryHV;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Content that can enter global workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceContent {
    /// Content representation (HDC vector)
    pub representation: Vec<BinaryHV>,

    /// Activation strength (competition currency)
    pub activation: f64,

    /// Source module (where content comes from)
    pub source: String,

    /// Semantic category
    pub category: String,

    /// Time in workspace (how long has it been conscious)
    pub duration: usize,
}

impl WorkspaceContent {
    /// Create new workspace content
    pub fn new(representation: Vec<BinaryHV>, activation: f64, source: String) -> Self {
        Self {
            representation,
            activation,
            source,
            category: "unknown".to_string(),
            duration: 0,
        }
    }

    /// Decay activation over time
    pub fn decay(&mut self, rate: f64) {
        self.activation *= 1.0 - rate;
        self.duration += 1;
    }
}

/// Workspace access state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessState {
    /// Content not in workspace (unconscious)
    Unconscious,

    /// Content competing for access (preconscious)
    Competing,

    /// Content in workspace (conscious)
    Conscious,

    /// Content being removed from workspace
    Fading,
}

/// Broadcasting event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BroadcastEvent {
    /// Content being broadcast
    pub content: Vec<BinaryHV>,

    /// Strength of broadcast
    pub strength: f64,

    /// Time of broadcast
    pub timestamp: usize,

    /// Recipient modules
    pub recipients: Vec<String>,
}

/// Workspace capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityMetrics {
    /// Current workspace occupancy [0,1]
    pub occupancy: f64,

    /// Number of contents in workspace
    pub num_contents: usize,

    /// Maximum capacity
    pub max_capacity: usize,

    /// Competition intensity
    pub competition: f64,

    /// Average content duration
    pub avg_duration: f64,
}

/// Configuration for global workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    /// Maximum workspace capacity
    pub max_capacity: usize,

    /// Activation threshold for entry
    pub entry_threshold: f64,

    /// Activation decay rate
    pub decay_rate: f64,

    /// Broadcasting enabled
    pub enable_broadcasting: bool,

    /// Competition mode (winner-takes-all vs capacity-limited)
    pub winner_takes_all: bool,

    /// Maximum content duration (auto-removal)
    pub max_duration: usize,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            max_capacity: 3,      // Typical: 3-4 items
            entry_threshold: 0.3, // Dehaene (2011): ignition should fire for
            // most salient content, not just peaks.
            // 0.5 was too high — Phi-weighted activations
            // from the cognitive loop are typically 0.2-0.5.
            decay_rate: 0.1, // 10% per timestep
            enable_broadcasting: true,
            winner_takes_all: false, // Allow multiple contents
            max_duration: 50,        // Auto-remove after 50 steps
        }
    }
}

/// Global workspace assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceAssessment {
    /// Capacity metrics
    pub capacity: CapacityMetrics,

    /// Currently conscious contents
    pub conscious_contents: Vec<WorkspaceContent>,

    /// Competing (preconscious) contents
    pub competing_contents: Vec<WorkspaceContent>,

    /// Recent broadcasts
    pub broadcasts: Vec<BroadcastEvent>,

    /// Ignition detected (sudden conscious access)
    pub ignition_detected: bool,

    /// Competition winner (if any)
    pub winner: Option<String>,

    /// Explanation
    pub explanation: String,
}

/// Broadcast handler callback type
pub type GWBroadcastHandler = Box<dyn Fn(&[BinaryHV]) + Send + Sync>;

/// Global Workspace
/// Implements Baars' Global Workspace Theory with competitive dynamics
pub struct GlobalWorkspace {
    /// Configuration
    config: WorkspaceConfig,

    /// Current workspace contents (conscious)
    workspace: Vec<WorkspaceContent>,

    /// Competing contents (preconscious)
    competitors: Vec<WorkspaceContent>,

    /// Broadcast history
    broadcasts: VecDeque<BroadcastEvent>,

    /// Timestep counter
    timestep: usize,

    /// Registered recipient modules
    recipients: Vec<String>,

    /// Registered broadcast handlers (module_name → callback)
    handlers: HashMap<String, GWBroadcastHandler>,

    /// Binding coherence from oscillatory network [0, 1].
    /// When `ctc_wiring` is enabled, gates workspace entry per Fries (2015)
    /// Communication Through Coherence: phase-locked oscillations control
    /// which neural populations can communicate.
    #[cfg(feature = "ctc_wiring")]
    binding_coherence: f64,

    /// Phase-Amplitude Coupling modulation index [0, 1].
    /// High PAC MI means workspace (low-freq) successfully modulates
    /// binding (high-freq) — boosting coalition activation.
    #[cfg(feature = "ctc_wiring")]
    pac_mi: f64,

    /// Ignition strength from last cycle [0, 1].
    /// Fed back to binding network as Kuramoto coupling boost.
    #[cfg(feature = "ctc_wiring")]
    last_ignition_strength: f64,

    /// Hysteresis state: true if currently in ignited state.
    /// Prevents threshold oscillation (Mashour et al. 2020).
    #[cfg(feature = "ctc_wiring")]
    ignition_hysteresis: bool,
}

impl std::fmt::Debug for GlobalWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalWorkspace")
            .field("config", &self.config)
            .field("workspace", &self.workspace)
            .field("competitors", &self.competitors)
            .field("broadcasts", &self.broadcasts)
            .field("timestep", &self.timestep)
            .field("recipients", &self.recipients)
            .field("handlers", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl GlobalWorkspace {
    /// Create new global workspace
    pub fn new(config: WorkspaceConfig) -> Self {
        Self {
            config,
            workspace: Vec::new(),
            competitors: Vec::new(),
            broadcasts: VecDeque::new(),
            timestep: 0,
            recipients: vec![
                "perception".to_string(),
                "memory".to_string(),
                "planning".to_string(),
                "language".to_string(),
                "action".to_string(),
            ],
            handlers: HashMap::new(),
            #[cfg(feature = "ctc_wiring")]
            binding_coherence: 0.0,
            #[cfg(feature = "ctc_wiring")]
            pac_mi: 0.0,
            #[cfg(feature = "ctc_wiring")]
            last_ignition_strength: 0.0,
            #[cfg(feature = "ctc_wiring")]
            ignition_hysteresis: false,
        }
    }

    /// Register a broadcast handler for a module.
    /// The handler fires whenever the workspace broadcasts content.
    pub fn register_handler(&mut self, module: &str, handler: GWBroadcastHandler) {
        self.handlers.insert(module.to_string(), handler);
    }

    /// Submit content for workspace entry (competition)
    pub fn submit(&mut self, content: WorkspaceContent) {
        self.competitors.push(content);
    }

    /// Process workspace dynamics (competition, decay, broadcasting)
    pub fn process(&mut self) -> WorkspaceAssessment {
        self.timestep += 1;

        // 1. Decay existing workspace contents
        for content in &mut self.workspace {
            content.decay(self.config.decay_rate);
        }

        // 2. Remove stale or low-activation contents
        self.workspace
            .retain(|c| c.activation > 0.1 && c.duration < self.config.max_duration);

        // 3. Competition for workspace access
        let (new_contents, ignition) = self.compete();

        // 4. Broadcasting
        let broadcasts = if self.config.enable_broadcasting {
            self.broadcast()
        } else {
            Vec::new()
        };

        // 4b. Dispatch registered handlers to matching broadcast recipients
        if !broadcasts.is_empty() {
            for broadcast in &broadcasts {
                for recipient in &broadcast.recipients {
                    if let Some(handler) = self.handlers.get(recipient) {
                        handler(&broadcast.content);
                    }
                }
            }
        }

        // 5. Compute metrics
        let capacity = self.compute_capacity_metrics();

        // 6. Detect winner
        let winner = self.workspace.first().map(|c| c.source.clone());

        // 7. Generate explanation
        let explanation = self.generate_explanation(&capacity, ignition);

        WorkspaceAssessment {
            capacity,
            conscious_contents: self.workspace.clone(),
            competing_contents: self.competitors.clone(),
            broadcasts,
            ignition_detected: ignition,
            winner,
            explanation,
        }
    }

    /// Competition for workspace access
    fn compete(&mut self) -> (Vec<WorkspaceContent>, bool) {
        if self.competitors.is_empty() {
            return (Vec::new(), false);
        }

        // Sort competitors by activation (strongest first)
        self.competitors.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut new_contents = Vec::new();
        let mut ignition = false;

        if self.config.winner_takes_all {
            // Winner-takes-all: only strongest enters
            if let Some(winner) = self.competitors.first() {
                if winner.activation > self.config.entry_threshold {
                    // Clear workspace, install winner
                    self.workspace.clear();
                    self.workspace.push(winner.clone());
                    new_contents.push(winner.clone());
                    ignition = true;
                }
            }
            self.competitors.clear();
        } else {
            // Capacity-limited: fill up to max_capacity
            let available_slots = self
                .config
                .max_capacity
                .saturating_sub(self.workspace.len());

            // ── CTC-gated entry (Fries 2015) ─────────────────────────
            // With `ctc_wiring`: binding coherence lowers the effective
            // entry threshold, and PAC MI boosts activation. High binding
            // means oscillatory phase-locking allows information flow into
            // the workspace. Ignition uses a sigmoid with hysteresis
            // instead of a hard threshold.
            #[cfg(feature = "ctc_wiring")]
            let effective_threshold = {
                // High binding → lower barrier (up to 30% reduction)
                let bc = self.binding_coherence.clamp(0.0, 1.0);
                self.config.entry_threshold * (1.0 - 0.3 * bc)
            };
            #[cfg(not(feature = "ctc_wiring"))]
            let effective_threshold = self.config.entry_threshold;

            let mut entered = 0;
            self.competitors.retain(|content| {
                // PAC modulation: high MI boosts coalition activation
                #[cfg(feature = "ctc_wiring")]
                let effective_activation =
                    content.activation * (0.5 + 0.5 * self.pac_mi.clamp(0.0, 1.0));
                #[cfg(not(feature = "ctc_wiring"))]
                let effective_activation = content.activation;

                if entered < available_slots && effective_activation > effective_threshold {
                    self.workspace.push(content.clone());
                    new_contents.push(content.clone());
                    entered += 1;

                    // Ignition detection
                    #[cfg(feature = "ctc_wiring")]
                    {
                        // Nonlinear sigmoid ignition with hysteresis (Mashour et al. 2020)
                        let ignition_input = effective_activation;
                        let threshold = if self.ignition_hysteresis { 0.65 } else { 0.70 };
                        let igp = 1.0 / (1.0 + (-10.0 * (ignition_input - threshold)).exp());
                        if igp > 0.5 {
                            ignition = true;
                            self.ignition_hysteresis = true;
                            self.last_ignition_strength = igp;
                        } else if igp < 0.3 {
                            // Hysteresis: must drop significantly to de-ignite
                            self.ignition_hysteresis = false;
                            self.last_ignition_strength = 0.0;
                        }
                    }
                    #[cfg(not(feature = "ctc_wiring"))]
                    {
                        // Original: hard threshold ignition
                        if content.activation > 0.8 {
                            ignition = true;
                        }
                    }

                    false // Remove from competitors
                } else {
                    true // Keep in competitors
                }
            });
        }

        (new_contents, ignition)
    }

    /// Broadcast workspace contents to all modules
    fn broadcast(&mut self) -> Vec<BroadcastEvent> {
        let mut events = Vec::new();

        for content in &self.workspace {
            let event = BroadcastEvent {
                content: content.representation.clone(),
                strength: content.activation,
                timestamp: self.timestep,
                recipients: self.recipients.clone(),
            };

            events.push(event.clone());
            self.broadcasts.push_back(event);
        }

        // Limit broadcast history
        while self.broadcasts.len() > 100 {
            self.broadcasts.pop_front();
        }

        events
    }

    // ── Communication Through Coherence (CTC) API ─────────────────
    // Fries (2015): phase-locked oscillations control which neural
    // populations can communicate. These methods wire binding coherence
    // from the oscillatory network into workspace entry gating.

    /// Set binding coherence from oscillatory network [0, 1].
    /// High coherence → phase-locked → lower GWT entry threshold.
    #[cfg(feature = "ctc_wiring")]
    pub fn set_binding_coherence(&mut self, coherence: f64) {
        self.binding_coherence = coherence.clamp(0.0, 1.0);
    }

    /// Set Phase-Amplitude Coupling modulation index [0, 1].
    /// High PAC MI → workspace successfully orchestrates binding
    /// → coalition activation is boosted.
    #[cfg(feature = "ctc_wiring")]
    pub fn set_pac_mi(&mut self, mi: f64) {
        self.pac_mi = mi.clamp(0.0, 1.0);
    }

    /// Get ignition strength from last cycle [0, 1].
    /// Fed back to binding network as Kuramoto coupling boost
    /// to sustain the CTC feedback loop.
    #[cfg(feature = "ctc_wiring")]
    pub fn last_ignition_strength(&self) -> f64 {
        self.last_ignition_strength
    }

    /// Compute capacity metrics
    fn compute_capacity_metrics(&self) -> CapacityMetrics {
        let num_contents = self.workspace.len();
        let occupancy = num_contents as f64 / self.config.max_capacity as f64;

        let competition = if self.competitors.is_empty() {
            0.0
        } else {
            self.competitors.len() as f64 / (self.config.max_capacity as f64 + 1.0)
        };

        let avg_duration = if num_contents > 0 {
            self.workspace
                .iter()
                .map(|c| c.duration as f64)
                .sum::<f64>()
                / num_contents as f64
        } else {
            0.0
        };

        CapacityMetrics {
            occupancy,
            num_contents,
            max_capacity: self.config.max_capacity,
            competition,
            avg_duration,
        }
    }

    /// Generate human-readable explanation
    fn generate_explanation(&self, capacity: &CapacityMetrics, ignition: bool) -> String {
        let mut parts = Vec::new();

        // Occupancy
        if capacity.occupancy > 0.8 {
            parts.push("Workspace nearly full".to_string());
        } else if capacity.occupancy > 0.5 {
            parts.push("Workspace partially occupied".to_string());
        } else if capacity.occupancy > 0.0 {
            parts.push("Workspace sparse".to_string());
        } else {
            parts.push("Workspace empty".to_string());
        }

        // Competition
        if capacity.competition > 1.0 {
            parts.push(format!(
                "High competition ({:.1}x capacity)",
                capacity.competition
            ));
        } else if capacity.competition > 0.5 {
            parts.push("Moderate competition".to_string());
        }

        // Ignition
        if ignition {
            parts.push("Ignition detected (sudden conscious access)".to_string());
        }

        // Duration
        if capacity.avg_duration > 30.0 {
            parts.push(format!(
                "Contents stable (avg {:.0} steps)",
                capacity.avg_duration
            ));
        }

        // Contents
        if capacity.num_contents > 0 {
            parts.push(format!("{} conscious contents", capacity.num_contents));
        }

        parts.join(". ")
    }

    /// Get current conscious contents
    pub fn get_conscious_contents(&self) -> &[WorkspaceContent] {
        &self.workspace
    }

    /// Check if content is conscious
    pub fn is_conscious(&self, content: &[BinaryHV]) -> bool {
        self.workspace
            .iter()
            .any(|c| self.similarity(&c.representation, content) > 0.9)
    }

    /// Compute similarity between HDC vectors using Hamming-based similarity
    fn similarity(&self, a: &[BinaryHV], b: &[BinaryHV]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let avg_sim: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x.similarity(y) as f64)
            .sum::<f64>()
            / a.len() as f64;
        avg_sim
    }

    /// Clear workspace
    pub fn clear(&mut self) {
        self.workspace.clear();
        self.competitors.clear();
    }

    /// Get number of conscious contents
    pub fn num_conscious(&self) -> usize {
        self.workspace.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let ws = GlobalWorkspace::new(WorkspaceConfig::default());
        assert_eq!(ws.num_conscious(), 0);
    }

    #[test]
    fn test_submit_content() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig::default());
        let content =
            WorkspaceContent::new(vec![BinaryHV::ones(); 10], 0.8, "perception".to_string());
        ws.submit(content);
        assert!(ws.competitors.len() > 0);
    }

    #[test]
    fn test_competition_entry() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            entry_threshold: 0.5,
            ..Default::default()
        });

        // High activation content should enter
        let high_activation =
            WorkspaceContent::new(vec![BinaryHV::ones(); 10], 0.9, "perception".to_string());
        ws.submit(high_activation);

        let assessment = ws.process();
        assert_eq!(assessment.conscious_contents.len(), 1);
    }

    #[test]
    fn test_competition_rejection() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            entry_threshold: 0.7,
            ..Default::default()
        });

        // Low activation content should not enter
        let low_activation =
            WorkspaceContent::new(vec![BinaryHV::zero(); 10], 0.3, "perception".to_string());
        ws.submit(low_activation);

        let assessment = ws.process();
        assert_eq!(assessment.conscious_contents.len(), 0);
    }

    #[test]
    fn test_capacity_limit() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            max_capacity: 2,
            entry_threshold: 0.5,
            ..Default::default()
        });

        // Submit 5 high-activation contents
        for i in 0..5 {
            let content = WorkspaceContent::new(
                vec![BinaryHV::random(i as u64); 10],
                0.9,
                format!("module{}", i),
            );
            ws.submit(content);
        }

        let assessment = ws.process();
        // Should only accept up to max_capacity (2)
        assert!(assessment.conscious_contents.len() <= 2);
    }

    #[test]
    fn test_decay() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            decay_rate: 0.3, // Higher decay rate
            entry_threshold: 0.5,
            ..Default::default()
        });

        let content =
            WorkspaceContent::new(vec![BinaryHV::ones(); 10], 0.9, "perception".to_string());
        ws.submit(content);
        ws.process(); // Enter workspace

        // Initial state: should be in workspace
        assert_eq!(ws.num_conscious(), 1);

        // Process multiple times to decay
        for _ in 0..20 {
            ws.process();
        }

        // After sufficient decay, content should be removed (activation < 0.1 threshold)
        assert_eq!(ws.num_conscious(), 0);
    }

    #[test]
    fn test_broadcasting() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            enable_broadcasting: true,
            entry_threshold: 0.5,
            ..Default::default()
        });

        let content =
            WorkspaceContent::new(vec![BinaryHV::ones(); 10], 0.9, "perception".to_string());
        ws.submit(content);

        let assessment = ws.process();
        assert!(!assessment.broadcasts.is_empty());
    }

    #[test]
    fn test_ignition_detection() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig::default());

        // Very high activation should trigger ignition
        let content =
            WorkspaceContent::new(vec![BinaryHV::ones(); 10], 0.95, "perception".to_string());
        ws.submit(content);

        let assessment = ws.process();
        assert!(assessment.ignition_detected);
    }

    #[test]
    fn test_winner_takes_all() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            winner_takes_all: true,
            entry_threshold: 0.5,
            ..Default::default()
        });

        // Submit multiple contents
        for i in 0..3 {
            let content = WorkspaceContent::new(
                vec![BinaryHV::random(i as u64); 10],
                0.7 + i as f64 * 0.1,
                format!("module{}", i),
            );
            ws.submit(content);
        }

        let assessment = ws.process();
        // Winner-takes-all: only one content
        assert_eq!(assessment.conscious_contents.len(), 1);
        // Should be the strongest (module2)
        assert_eq!(assessment.winner, Some("module2".to_string()));
    }

    #[test]
    fn test_is_conscious() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            entry_threshold: 0.5,
            ..Default::default()
        });

        let content_vec = vec![BinaryHV::ones(); 10];
        let content = WorkspaceContent::new(content_vec.clone(), 0.9, "perception".to_string());
        ws.submit(content);
        ws.process();

        assert!(ws.is_conscious(&content_vec));
    }

    #[test]
    fn test_similar_workspace_contents_detected() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            entry_threshold: 0.5,
            ..Default::default()
        });

        // Use the same BinaryHV vectors for content
        let shared_vec = vec![BinaryHV::ones(); 10];
        let content = WorkspaceContent::new(shared_vec.clone(), 0.9, "perception".to_string());
        ws.submit(content);
        ws.process();

        // Identical content should be detected as conscious
        assert!(ws.is_conscious(&shared_vec));

        // Completely different content should NOT be detected
        let different_vec = vec![BinaryHV::zero(); 10];
        assert!(!ws.is_conscious(&different_vec));
    }

    #[test]
    fn test_clear() {
        let mut ws = GlobalWorkspace::new(WorkspaceConfig::default());
        let content =
            WorkspaceContent::new(vec![BinaryHV::ones(); 10], 0.9, "perception".to_string());
        ws.submit(content);
        ws.process();

        ws.clear();
        assert_eq!(ws.num_conscious(), 0);
    }

    #[test]
    fn test_register_handler_dispatches_on_broadcast() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            enable_broadcasting: true,
            entry_threshold: 0.3,
            ..Default::default()
        });
        // Register handler for "memory" — one of the default recipients
        ws.register_handler(
            "memory",
            Box::new(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            }),
        );

        // Submit high-activation content that will enter workspace and be broadcast
        let content =
            WorkspaceContent::new(vec![BinaryHV::random(42); 3], 0.9, "source".to_string());
        ws.submit(content);
        let assessment = ws.process();

        let count = counter.load(Ordering::Relaxed);
        // Handler fires once per broadcast event (each workspace content broadcasts)
        assert_eq!(
            count,
            assessment.broadcasts.len(),
            "handler should fire once per broadcast"
        );
    }

    #[test]
    fn test_handler_only_fires_for_matching_recipients() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let mut ws = GlobalWorkspace::new(WorkspaceConfig {
            enable_broadcasting: true,
            entry_threshold: 0.3,
            ..Default::default()
        });
        // Register handler for "custom_module" — NOT in default recipients
        ws.register_handler(
            "custom_module",
            Box::new(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            }),
        );

        let content =
            WorkspaceContent::new(vec![BinaryHV::random(99); 3], 0.9, "source".to_string());
        ws.submit(content);
        let assessment = ws.process();

        // Broadcasts happened but handler should NOT fire (no matching recipient)
        assert!(!assessment.broadcasts.is_empty(), "broadcasts should occur");
        assert_eq!(
            counter.load(Ordering::Relaxed),
            0,
            "handler should not fire for non-matching recipient"
        );
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// Strategy: generate a list of recipient names for the workspace
        fn recipient_names() -> impl Strategy<Value = Vec<String>> {
            prop::collection::vec(
                prop::sample::select(vec![
                    "perception",
                    "memory",
                    "planning",
                    "language",
                    "action",
                    "custom_a",
                    "custom_b",
                    "custom_c",
                ]),
                1..=5,
            )
            .prop_map(|v| v.into_iter().map(|s| s.to_string()).collect())
        }

        /// Strategy: generate a list of handler keys to register
        fn handler_keys() -> impl Strategy<Value = Vec<String>> {
            prop::collection::vec(
                prop::sample::select(vec![
                    "perception",
                    "memory",
                    "planning",
                    "language",
                    "action",
                    "custom_a",
                    "custom_b",
                    "custom_c",
                    "unrelated",
                ]),
                0..=4,
            )
            .prop_map(|v| v.into_iter().map(|s| s.to_string()).collect())
        }

        proptest! {
            #[test]
            fn filtered_dispatch_only_fires_matching_handlers(
                recipients in recipient_names(),
                handlers in handler_keys(),
            ) {
                // Deduplicate handler keys (HashMap.insert overwrites duplicates)
                let mut seen = std::collections::HashSet::new();
                let unique_handlers: Vec<String> = handlers
                    .into_iter()
                    .filter(|k| seen.insert(k.clone()))
                    .collect();

                let counters: Vec<(String, Arc<AtomicUsize>)> = unique_handlers
                    .iter()
                    .map(|k| (k.clone(), Arc::new(AtomicUsize::new(0))))
                    .collect();

                let mut ws = GlobalWorkspace::new(WorkspaceConfig {
                    enable_broadcasting: true,
                    entry_threshold: 0.3,
                    ..Default::default()
                });

                // Override recipients
                ws.recipients = recipients.clone();

                // Register handlers
                for (key, counter) in &counters {
                    let c = counter.clone();
                    ws.register_handler(key, Box::new(move |_| {
                        c.fetch_add(1, Ordering::Relaxed);
                    }));
                }

                // Submit high-activation content to trigger broadcast
                let content = WorkspaceContent::new(
                    vec![BinaryHV::random(42); 3],
                    0.9,
                    "source".to_string(),
                );
                ws.submit(content);
                let assessment = ws.process();
                let num_broadcasts = assessment.broadcasts.len();

                // Verify: each handler fires exactly once per broadcast IF its key
                // is in the recipients list, and zero times otherwise.
                for (key, counter) in &counters {
                    let count = counter.load(Ordering::Relaxed);
                    let expected_per_broadcast = if recipients.contains(key) { 1 } else { 0 };
                    // Key may appear multiple times in recipients → fires multiple times
                    let recipient_occurrences = recipients.iter().filter(|r| *r == key).count();
                    let expected = recipient_occurrences * num_broadcasts;
                    prop_assert_eq!(
                        count, expected,
                        "handler '{}': expected {} fires (recipients={:?}, broadcasts={}), got {}",
                        key, expected, recipients, num_broadcasts, count
                    );
                }
            }
        }
    }
}
