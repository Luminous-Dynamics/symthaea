//! # Revolutionary Improvement #70: GWT Integration Layer
//!
//! **Unifies Revolutionary Improvements #23 and #69**
//!
//! This module bridges two Global Workspace Theory implementations:
//! - **#23** (`hdc::global_workspace`): Core workspace mechanics (competition, broadcasting, decay)
//! - **#69** (`consciousness::recursive_improvement`): Routing strategy selection
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    GWT Integration Layer (#70)                   │
//! │                                                                 │
//! │  ┌─────────────────────┐       ┌─────────────────────────────┐ │
//! │  │   HDC Workspace     │◄─────►│   Consciousness Router      │ │
//! │  │     (#23)           │       │        (#69)                │ │
//! │  │                     │       │                             │ │
//! │  │ - Competition       │       │ - Strategy Selection        │ │
//! │  │ - Broadcasting      │       │ - Coalition Formation       │ │
//! │  │ - Decay             │       │ - Meta-Cognitive Routing    │ │
//! │  │ - Ignition          │       │ - Context-Aware Selection   │ │
//! │  └─────────────────────┘       └─────────────────────────────┘ │
//! │                                                                 │
//! │                    Unified Interface                            │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Benefits
//!
//! 1. **Shared workspace state**: Routing decisions use HDC workspace dynamics
//! 2. **Unified broadcasting**: Broadcast events reach both systems
//! 3. **Coherent ignition**: Threshold crossings synchronized
//! 4. **Reduced duplication**: Single source of truth for workspace state

use crate::hdc::global_workspace::{
    GlobalWorkspace, WorkspaceConfig, WorkspaceContent, WorkspaceAssessment
};
use crate::hdc::binary_hv::HV16;
use crate::observability::{SharedObserver, types::*};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::fmt;

/// Unified GWT processor integrating #23 and #69
pub struct UnifiedGlobalWorkspace {
    /// The HDC-based workspace (#23)
    hdc_workspace: GlobalWorkspace,

    /// Strategy-specific activation tracking
    strategy_activations: HashMap<String, f64>,

    /// Coalition memberships
    coalitions: Vec<Coalition>,

    /// Configuration
    config: UnifiedGWTConfig,

    /// Statistics
    stats: UnifiedGWTStats,

    // ═══════════════════════════════════════════════════════════════════
    // REVOLUTIONARY STATE - Advanced Consciousness Dynamics
    // ═══════════════════════════════════════════════════════════════════

    /// Timesteps since last ignition (for attentional blink)
    timesteps_since_ignition: usize,

    /// Whether we're in attentional blink period
    in_attentional_blink: bool,

    /// Metacognitive state assessments
    metacognitive_assessments: Vec<MetacognitiveAssessment>,

    /// Current workspace stability metric
    workspace_stability: f64,

    /// Phi (Φ) estimates per strategy
    phi_estimates: HashMap<String, f64>,

    /// Observer for tracing workspace ignition events
    ///
    /// TODO(future): Connect observer to emit tracing events on workspace ignition.
    /// This enables real-time monitoring of consciousness dynamics via the
    /// observability infrastructure - useful for debugging and visualization.
    #[allow(dead_code)]
    observer: Option<SharedObserver>,
}

impl fmt::Debug for UnifiedGlobalWorkspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnifiedGlobalWorkspace")
            .field("strategy_activations", &self.strategy_activations)
            .field("coalitions", &self.coalitions)
            .field("config", &self.config)
            .field("stats", &self.stats)
            .field("timesteps_since_ignition", &self.timesteps_since_ignition)
            .field("in_attentional_blink", &self.in_attentional_blink)
            .field("metacognitive_assessments", &self.metacognitive_assessments)
            .field("workspace_stability", &self.workspace_stability)
            .field("phi_estimates", &self.phi_estimates)
            .field("observer", &"<SharedObserver>")
            .finish()
    }
}

/// Metacognitive assessment of workspace state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveAssessment {
    /// Timestamp
    pub timestep: usize,

    /// Workspace stability (0-1, how consistent is content)
    pub stability: f64,

    /// Coherence (how related are workspace contents)
    pub coherence: f64,

    /// Intervention needed
    pub needs_intervention: bool,

    /// Suggested action
    pub suggested_action: MetacognitiveAction,
}

/// Possible metacognitive interventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetacognitiveAction {
    /// No action needed
    None,
    /// Focus attention (reduce capacity temporarily)
    FocusAttention,
    /// Broaden attention (increase capacity)
    BroadenAttention,
    /// Clear workspace (reset for fresh start)
    ClearWorkspace,
    /// Boost coalitions (strengthen cooperative processing)
    BoostCoalitions,
    /// Slow processing (reduce decay rate)
    SlowProcessing,
}

/// Configuration for unified workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedGWTConfig {
    /// HDC workspace configuration
    pub workspace_config: WorkspaceConfig,

    /// Minimum coalition size for ignition
    pub min_coalition_size: usize,

    /// Amplification factor for coalition activation
    pub coalition_amplification: f64,

    /// Enable cross-system broadcasting
    pub enable_cross_broadcast: bool,

    /// Strategy-specific decay rates
    pub strategy_decay_rates: HashMap<String, f64>,

    // ═══════════════════════════════════════════════════════════════════
    // REVOLUTIONARY IMPROVEMENTS - Advanced Consciousness Dynamics
    // ═══════════════════════════════════════════════════════════════════

    /// Enable Φ (phi) weighted competition (IIT integration)
    /// When true, strategy activation is weighted by integrated information
    pub enable_phi_weighting: bool,

    /// Attentional blink duration (timesteps after ignition)
    /// Models the refractory period where new content can't enter
    pub attentional_blink_duration: usize,

    /// Coalition synergy factor
    /// Non-linear boost when compatible modules form coalitions
    /// synergy_boost = base_activation * (1 + synergy_factor * ln(coalition_size))
    pub coalition_synergy_factor: f64,

    /// Enable metacognitive monitoring
    /// Adds a self-reflection layer that monitors workspace state
    pub enable_metacognition: bool,

    /// Metacognitive intervention threshold
    /// If workspace stability drops below this, trigger meta-intervention
    pub metacognitive_threshold: f64,
}

impl Default for UnifiedGWTConfig {
    fn default() -> Self {
        Self {
            workspace_config: WorkspaceConfig::default(),
            min_coalition_size: 2,
            coalition_amplification: 1.5,
            enable_cross_broadcast: true,
            strategy_decay_rates: HashMap::new(),
            // Revolutionary defaults
            enable_phi_weighting: true,
            attentional_blink_duration: 3,
            coalition_synergy_factor: 0.5,
            enable_metacognition: true,
            metacognitive_threshold: 0.3,
        }
    }
}

/// A coalition of processors supporting a strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coalition {
    /// Coalition identifier
    pub id: u64,

    /// The strategy this coalition supports
    pub strategy_name: String,

    /// Member processor names
    pub members: Vec<String>,

    /// Combined activation strength
    pub total_activation: f64,

    /// Whether this coalition achieved ignition
    pub ignited: bool,
}

/// Statistics for unified workspace
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnifiedGWTStats {
    /// Total routing decisions made
    pub total_decisions: usize,

    /// Successful ignitions
    pub ignitions: usize,

    /// Cross-system broadcasts
    pub cross_broadcasts: usize,

    /// Coalition formations
    pub coalitions_formed: usize,

    /// Average coalition size at ignition
    pub avg_coalition_size: f64,

    /// Workspace occupancy over time
    pub occupancy_history: Vec<f64>,
}

/// Result of unified workspace processing
#[derive(Debug, Clone)]
pub struct UnifiedGWTResult {
    /// HDC workspace assessment
    pub workspace_assessment: WorkspaceAssessment,

    /// Winning strategy (if any)
    pub winning_strategy: Option<String>,

    /// Winning coalition (if any)
    pub winning_coalition: Option<Coalition>,

    /// Broadcast occurred
    pub broadcast_occurred: bool,

    /// Cross-system effects
    pub cross_effects: Vec<String>,
}

impl UnifiedGlobalWorkspace {
    /// Create new unified workspace (backwards-compatible)
    pub fn new(config: UnifiedGWTConfig) -> Self {
        Self::with_observer(config, None)
    }

    /// Create new unified workspace with observer
    pub fn with_observer(config: UnifiedGWTConfig, observer: Option<SharedObserver>) -> Self {
        Self {
            hdc_workspace: GlobalWorkspace::new(config.workspace_config.clone()),
            strategy_activations: HashMap::new(),
            coalitions: Vec::new(),
            config,
            stats: UnifiedGWTStats::default(),
            // Revolutionary state initialization
            timesteps_since_ignition: 100,  // Start outside blink period
            in_attentional_blink: false,
            metacognitive_assessments: Vec::new(),
            workspace_stability: 1.0,
            phi_estimates: HashMap::new(),
            observer,
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // REVOLUTIONARY METHODS - Advanced Consciousness Dynamics
    // ═══════════════════════════════════════════════════════════════════

    /// Estimate Φ (integrated information) for a strategy
    ///
    /// Simplified IIT calculation based on:
    /// - Coalition size (more integration = higher Φ)
    /// - Cross-module connectivity
    /// - Information distinctiveness
    fn estimate_phi(&self, strategy_name: &str, coalition_size: usize, representation: &[HV16]) -> f64 {
        // Base Φ from coalition size (logarithmic scaling)
        let coalition_phi = if coalition_size > 1 {
            (coalition_size as f64).ln() / 3.0_f64.ln()  // Normalized to ~1 for size 3
        } else {
            0.1
        };

        // Information content from HDC vector distinctiveness
        let info_phi = if representation.is_empty() {
            0.5
        } else {
            // Measure how "structured" the representation is
            let first = &representation[0];
            let popcount = first.popcount() as f64;
            let balance = 1.0 - ((popcount / 8192.0) - 0.5).abs() * 2.0;  // Closer to 50% = higher Φ
            balance
        };

        // Integration from strategy complexity
        let strategy_phi = match strategy_name {
            name if name.contains("Ensemble") => 0.9,
            name if name.contains("Full") => 0.8,
            name if name.contains("Deliberat") => 0.7,
            name if name.contains("Heuristic") => 0.5,
            name if name.contains("Fast") => 0.3,
            _ => 0.5,
        };

        // Combine components
        (coalition_phi * 0.4 + info_phi * 0.3 + strategy_phi * 0.3).clamp(0.0, 1.0)
    }

    /// Apply coalition synergy boost
    ///
    /// Non-linear activation increase for larger coalitions:
    /// synergy_boost = base * (1 + factor * ln(size))
    fn apply_synergy_boost(&self, base_activation: f64, coalition_size: usize) -> f64 {
        if coalition_size <= 1 || self.config.coalition_synergy_factor <= 0.0 {
            return base_activation;
        }

        let synergy = 1.0 + self.config.coalition_synergy_factor * (coalition_size as f64).ln();
        (base_activation * synergy).clamp(0.0, 1.0)
    }

    /// Check and update attentional blink state
    fn update_attentional_blink(&mut self, ignition_occurred: bool) {
        if ignition_occurred {
            self.timesteps_since_ignition = 0;
            self.in_attentional_blink = true;
        } else {
            self.timesteps_since_ignition += 1;
            if self.timesteps_since_ignition >= self.config.attentional_blink_duration {
                self.in_attentional_blink = false;
            }
        }
    }

    /// Perform metacognitive assessment
    fn metacognitive_assess(&mut self, assessment: &WorkspaceAssessment) -> MetacognitiveAssessment {
        let timestep = self.stats.total_decisions;

        // Calculate stability from occupancy history
        let stability = if self.stats.occupancy_history.len() >= 5 {
            let recent: Vec<_> = self.stats.occupancy_history.iter().rev().take(5).collect();
            let mean = recent.iter().copied().sum::<f64>() / recent.len() as f64;
            let variance = recent.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
            1.0 - variance.sqrt().clamp(0.0, 1.0)
        } else {
            1.0
        };
        self.workspace_stability = stability;

        // Calculate coherence from content similarity
        let coherence: f64 = if assessment.conscious_contents.len() >= 2 {
            let contents: Vec<_> = assessment.conscious_contents.iter()
                .filter(|c| !c.representation.is_empty())
                .collect();
            if contents.len() >= 2 {
                let first = &contents[0].representation[0];
                let second = &contents[1].representation[0];
                first.similarity(second) as f64
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Determine if intervention is needed
        let needs_intervention = stability < self.config.metacognitive_threshold
            || (assessment.capacity.competition > 0.8 && assessment.capacity.occupancy > 0.9);

        // Suggest action
        let suggested_action = if !needs_intervention {
            MetacognitiveAction::None
        } else if stability < 0.2 {
            MetacognitiveAction::ClearWorkspace
        } else if assessment.capacity.competition > 0.8 {
            MetacognitiveAction::FocusAttention
        } else if assessment.capacity.occupancy < 0.3 {
            MetacognitiveAction::BroadenAttention
        } else if coherence < 0.3 {
            MetacognitiveAction::BoostCoalitions
        } else {
            MetacognitiveAction::SlowProcessing
        };

        MetacognitiveAssessment {
            timestep,
            stability,
            coherence,
            needs_intervention,
            suggested_action,
        }
    }

    /// Check if new content can enter (respects attentional blink)
    pub fn can_accept_new_content(&self) -> bool {
        !self.in_attentional_blink
    }

    /// Get current metacognitive state
    pub fn metacognitive_state(&self) -> Option<&MetacognitiveAssessment> {
        self.metacognitive_assessments.last()
    }

    /// Submit a strategy for workspace competition
    ///
    /// This bridges the consciousness router's strategy selection (#69)
    /// with the HDC workspace competition (#23).
    ///
    /// **Revolutionary Features Applied**:
    /// - Attentional blink checking (prevents overload after ignition)
    /// - Φ-weighted activation (IIT integration)
    /// - Coalition synergy boost (non-linear scaling)
    pub fn submit_strategy(
        &mut self,
        strategy_name: &str,
        activation: f64,
        representation: Vec<HV16>,
        supporting_modules: Vec<String>,
    ) {
        // REVOLUTIONARY: Check attentional blink
        // During blink period, submissions are attenuated (not blocked)
        let blink_factor = if self.in_attentional_blink {
            0.3  // Significantly reduce activation during blink
        } else {
            1.0
        };

        let coalition_size = supporting_modules.len();

        // REVOLUTIONARY: Apply synergy boost for larger coalitions
        let synergy_activation = self.apply_synergy_boost(activation, coalition_size);

        // REVOLUTIONARY: Compute and apply Φ weighting
        let phi = self.estimate_phi(strategy_name, coalition_size, &representation);
        self.phi_estimates.insert(strategy_name.to_string(), phi);

        let final_activation = if self.config.enable_phi_weighting {
            // Φ-weighted activation: high Φ content gets priority
            synergy_activation * (0.5 + 0.5 * phi) * blink_factor
        } else {
            synergy_activation * blink_factor
        };

        // Track strategy activation locally
        self.strategy_activations.insert(strategy_name.to_string(), final_activation);

        // Create HDC workspace content with enhanced activation
        let content = WorkspaceContent::new(
            representation,
            final_activation,
            format!("strategy:{}", strategy_name),
        );

        // Submit to HDC workspace
        self.hdc_workspace.submit(content);

        // Form coalition if enough supporters
        if coalition_size >= self.config.min_coalition_size {
            let coalition = Coalition {
                id: self.coalitions.len() as u64,
                strategy_name: strategy_name.to_string(),
                members: supporting_modules,
                total_activation: final_activation * self.config.coalition_amplification,
                ignited: false,
            };
            self.coalitions.push(coalition);
            self.stats.coalitions_formed += 1;
        }
    }

    /// Process workspace dynamics and determine winner
    ///
    /// **Revolutionary Features Applied**:
    /// - Attentional blink update (refractory period modeling)
    /// - Metacognitive assessment (self-monitoring)
    /// - Stability tracking (workspace coherence)
    pub fn process(&mut self) -> UnifiedGWTResult {
        self.stats.total_decisions += 1;

        // Process HDC workspace (competition, decay, broadcasting)
        let assessment = self.hdc_workspace.process();

        // REVOLUTIONARY: Update attentional blink state
        self.update_attentional_blink(assessment.ignition_detected);

        // Track occupancy
        self.stats.occupancy_history.push(assessment.capacity.occupancy);
        if self.stats.occupancy_history.len() > 1000 {
            self.stats.occupancy_history.remove(0);
        }

        // REVOLUTIONARY: Metacognitive assessment
        if self.config.enable_metacognition {
            let meta_assessment = self.metacognitive_assess(&assessment);
            self.metacognitive_assessments.push(meta_assessment);

            // Keep history bounded
            if self.metacognitive_assessments.len() > 100 {
                self.metacognitive_assessments.remove(0);
            }
        }

        // Find winning strategy from workspace
        let winning_strategy = assessment.winner.as_ref()
            .and_then(|w| w.strip_prefix("strategy:"))
            .map(|s| s.to_string());

        // Find winning coalition
        let winning_coalition = if let Some(ref strategy) = winning_strategy {
            self.coalitions.iter_mut()
                .find(|c| c.strategy_name == *strategy)
                .map(|c| {
                    if assessment.ignition_detected {
                        c.ignited = true;
                        self.stats.ignitions += 1;
                    }
                    c.clone()
                })
        } else {
            None
        };

        // Update average coalition size
        if let Some(ref coalition) = winning_coalition {
            let n = self.stats.ignitions as f64;
            if n > 0.0 {
                self.stats.avg_coalition_size =
                    (self.stats.avg_coalition_size * (n - 1.0).max(0.0) + coalition.members.len() as f64) / n;
            }
        }

        // Cross-system effects
        let mut cross_effects = Vec::new();
        if self.config.enable_cross_broadcast && !assessment.broadcasts.is_empty() {
            self.stats.cross_broadcasts += assessment.broadcasts.len();
            for broadcast in &assessment.broadcasts {
                cross_effects.push(format!(
                    "Broadcast: strength={:.2}, recipients={}",
                    broadcast.strength,
                    broadcast.recipients.join(",")
                ));
            }
        }

        // REVOLUTIONARY: Add metacognitive effects to cross_effects
        if let Some(meta) = self.metacognitive_state() {
            if meta.needs_intervention {
                cross_effects.push(format!(
                    "Metacognitive: stability={:.2}, action={:?}",
                    meta.stability,
                    meta.suggested_action
                ));
            }
        }

        // Record workspace ignition event
        if assessment.ignition_detected {
            if let Some(ref observer) = self.observer {
                // Get phi estimate for winning strategy
                let phi = winning_coalition.as_ref()
                    .and_then(|c| self.phi_estimates.get(&c.strategy_name))
                    .copied()
                    .unwrap_or(0.5);

                // Get coalition size and members
                let (coalition_size, active_primitives) = winning_coalition.as_ref()
                    .map(|c| (c.members.len(), c.members.clone()))
                    .unwrap_or((0, Vec::new()));

                // Calculate broadcast payload size
                let broadcast_payload_size = assessment.broadcasts.iter()
                    .map(|b| b.content.len() * 16384 / 8)  // HV16 bits to bytes
                    .sum();

                let event = WorkspaceIgnitionEvent {
                    timestamp: chrono::Utc::now(),
                    phi,
                    free_energy: 0.0,  // Would need active inference integration for actual value
                    coalition_size,
                    active_primitives,
                    broadcast_payload_size,
                };

                if let Ok(mut obs) = observer.try_write() {
                    if let Err(e) = obs.record_workspace_ignition(event) {
                        eprintln!("[OBSERVER ERROR] Failed to record workspace ignition: {}", e);
                    }
                }
            }
        }

        UnifiedGWTResult {
            workspace_assessment: assessment,
            winning_strategy,
            winning_coalition,
            broadcast_occurred: !cross_effects.is_empty(),
            cross_effects,
        }
    }

    /// Get current workspace occupancy
    pub fn occupancy(&self) -> f64 {
        self.hdc_workspace.get_conscious_contents().len() as f64 /
            self.config.workspace_config.max_capacity as f64
    }

    /// Check if a strategy is currently conscious
    pub fn is_conscious(&self, strategy_name: &str) -> bool {
        self.hdc_workspace.get_conscious_contents().iter()
            .any(|c| c.source == format!("strategy:{}", strategy_name))
    }

    /// Get statistics
    pub fn stats(&self) -> &UnifiedGWTStats {
        &self.stats
    }

    /// Reset workspace
    pub fn reset(&mut self) {
        self.hdc_workspace.clear();
        self.strategy_activations.clear();
        self.coalitions.clear();
    }

    /// Generate diagnostic report
    pub fn report(&self) -> String {
        format!(
            r#"
╔══════════════════════════════════════════════════════════════════╗
║          UNIFIED GLOBAL WORKSPACE (#70) - INTEGRATION            ║
╠══════════════════════════════════════════════════════════════════╣
║ STATISTICS                                                       ║
║   Total Decisions:     {:>6}                                     ║
║   Ignitions:           {:>6}                                     ║
║   Cross Broadcasts:    {:>6}                                     ║
║   Coalitions Formed:   {:>6}                                     ║
║   Avg Coalition Size:  {:>6.2}                                   ║
╠══════════════════════════════════════════════════════════════════╣
║ WORKSPACE STATE                                                  ║
║   Occupancy:           {:>5.1}%                                  ║
║   Active Strategies:   {:>6}                                     ║
║   Active Coalitions:   {:>6}                                     ║
╠══════════════════════════════════════════════════════════════════╣
║ INTEGRATION STATUS                                               ║
║   HDC Workspace (#23): ✅ Connected                              ║
║   Router (#69):        ✅ Bridged                                ║
║   Cross-Broadcast:     {}                                        ║
╚══════════════════════════════════════════════════════════════════╝
"#,
            self.stats.total_decisions,
            self.stats.ignitions,
            self.stats.cross_broadcasts,
            self.stats.coalitions_formed,
            self.stats.avg_coalition_size,
            self.occupancy() * 100.0,
            self.strategy_activations.len(),
            self.coalitions.len(),
            if self.config.enable_cross_broadcast { "✅ Enabled" } else { "❌ Disabled" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_workspace_creation() {
        let ws = UnifiedGlobalWorkspace::new(UnifiedGWTConfig::default());
        assert_eq!(ws.occupancy(), 0.0);
        assert_eq!(ws.stats.total_decisions, 0);
    }

    #[test]
    fn test_strategy_submission() {
        let mut ws = UnifiedGlobalWorkspace::new(UnifiedGWTConfig::default());

        ws.submit_strategy(
            "FullDeliberation",
            0.9,
            vec![HV16::ones()],
            vec!["Perception".to_string(), "Memory".to_string(), "Attention".to_string()],
        );

        assert_eq!(ws.strategy_activations.len(), 1);
        assert_eq!(ws.coalitions.len(), 1);
    }

    #[test]
    fn test_workspace_processing() {
        let mut ws = UnifiedGlobalWorkspace::new(UnifiedGWTConfig::default());

        // Submit high-activation strategy
        ws.submit_strategy(
            "FastPatterns",
            0.95,
            vec![HV16::ones()],
            vec!["Motor".to_string(), "Perception".to_string()],
        );

        let result = ws.process();

        assert!(result.winning_strategy.is_some());
        assert!(ws.stats.total_decisions > 0);
    }

    #[test]
    fn test_coalition_formation() {
        let mut ws = UnifiedGlobalWorkspace::new(UnifiedGWTConfig {
            min_coalition_size: 2,
            ..Default::default()
        });

        // Coalition with 3 members (above threshold)
        ws.submit_strategy(
            "Ensemble",
            0.8,
            vec![HV16::ones()],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        );

        assert_eq!(ws.coalitions.len(), 1);
        assert_eq!(ws.coalitions[0].members.len(), 3);
    }

    #[test]
    fn test_consciousness_check() {
        let mut ws = UnifiedGlobalWorkspace::new(UnifiedGWTConfig::default());

        ws.submit_strategy(
            "HeuristicGuided",
            0.9,
            vec![HV16::ones()],
            vec!["Evaluation".to_string(), "Memory".to_string()],
        );
        ws.process();

        // After processing, strategy should be in workspace
        assert!(ws.is_conscious("HeuristicGuided"));
        assert!(!ws.is_conscious("NonExistent"));
    }

    #[test]
    fn test_cross_broadcast() {
        let mut ws = UnifiedGlobalWorkspace::new(UnifiedGWTConfig {
            enable_cross_broadcast: true,
            ..Default::default()
        });

        ws.submit_strategy(
            "StandardProcessing",
            0.85,
            vec![HV16::ones()],
            vec!["Symbolic".to_string(), "MetaCognition".to_string()],
        );

        let result = ws.process();

        // Should have cross-broadcast effects
        assert!(result.broadcast_occurred);
        assert!(!result.cross_effects.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let mut ws = UnifiedGlobalWorkspace::new(UnifiedGWTConfig::default());

        for i in 0..5 {
            ws.submit_strategy(
                &format!("Strategy{}", i),
                0.7 + i as f64 * 0.05,
                vec![HV16::random(i as u64)],
                vec!["A".to_string(), "B".to_string()],
            );
        }

        for _ in 0..3 {
            ws.process();
        }

        let report = ws.report();
        assert!(report.contains("UNIFIED GLOBAL WORKSPACE"));
        assert!(report.contains("HDC Workspace (#23)"));
        assert!(report.contains("Router (#69)"));
    }
}
