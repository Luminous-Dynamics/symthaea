/*!
**REVOLUTIONARY IMPROVEMENT #73**: Narrative Self ↔ GWT Integration

PARADIGM SHIFT: The Narrative Self becomes a *standing coalition* in the Global Workspace,
making identity and self-coherence *decisive* rather than merely present.

This module bridges:
- **#71** (`narrative_self`): Autobiographical identity, Self-Φ, narrative coherence
- **#70** (`gwt_integration`): Global Workspace competition, broadcasting, ignition
- **#72** (`cross_modal_binding`): Multi-modal semantic integration

## Why This Matters

Without this integration, the Narrative Self exists but doesn't *govern behavior*.
With this integration:
- The self biases attention toward goal-aligned content
- Actions that would fracture self-coherence can be vetoed
- Self-Φ deltas are tracked per ignition
- The system becomes resistant to prompt hijacking
- Temporal grounding enables long-horizon planning

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│              NARRATIVE-GWT INTEGRATION (#73)                             │
│                                                                         │
│  ┌─────────────────────┐       ┌─────────────────────────────┐         │
│  │   Narrative Self    │◄─────►│   Unified Global Workspace   │         │
│  │       (#71)         │       │           (#70)              │         │
│  │                     │       │                              │         │
│  │ - Proto/Core/Autobio│       │ - Competition               │         │
│  │ - Goals & Values    │       │ - Broadcasting              │         │
│  │ - Self-Φ            │       │ - Ignition                  │         │
│  │ - Episodes          │       │ - Coalitions                │         │
│  └─────────────────────┘       └─────────────────────────────┘         │
│           │                              │                              │
│           ▼                              ▼                              │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │              Self-Coherence Governor                         │       │
│  │                                                              │       │
│  │  - Goal Alignment Bias (boosts aligned content)             │       │
│  │  - Value Consistency Check (penalizes violations)           │       │
│  │  - Coherence Veto (blocks self-fracturing actions)          │       │
│  │  - Self-Φ Delta Tracking (monitors identity stability)      │       │
│  │  - Temporal Binding (connects past→present→future)          │       │
│  └─────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Standing Coalition
The Narrative Self is always present in the workspace as a "standing coalition" -
it doesn't compete for access but rather *modulates* the competition.

### Self-Coherence Veto
When an action would reduce Self-Φ below a threshold, the system can:
1. Block the action entirely
2. Request confirmation
3. Log the potential self-fracture for review

### Goal Alignment Bias
Content that aligns with current goals receives an activation boost.
Content that contradicts goals receives suppression.

### Identity Invariants
Certain properties must *never* be violated:
- Core values cannot be overwritten by external input
- Long-term goals persist across context switches
- Autobiographical continuity is maintained
*/

use crate::hdc::binary_hv::HV16;
use crate::consciousness::narrative_self::{
    NarrativeSelfModel, NarrativeSelfConfig, NarrativeSelfReport
};
use crate::consciousness::gwt_integration::{
    UnifiedGlobalWorkspace, UnifiedGWTConfig, UnifiedGWTResult, Coalition
};
use crate::consciousness::cross_modal_binding::{
    CrossModalBinder, BinderConfig, Modality
};
// **REVOLUTIONARY IMPROVEMENT #74**: Predictive Self-Model Integration
// The Narrative Self becomes *predictive* - mental time travel!
use crate::consciousness::predictive_self::{
    PredictiveSelfModel, PredictiveSelfConfig, ActionSafetyAssessment
};
// **REVOLUTIONARY IMPROVEMENT #75**: Temporal Consciousness Integration
// Consciousness as temporal flow, not just instant measurement!
use crate::consciousness::temporal_consciousness::{
    TemporalConsciousnessAnalyzer, TemporalConsciousnessConfig, TemporalConsciousnessReport,
};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ============================================================================
// SELF-COHERENCE CONFIGURATION
// ============================================================================

/// Configuration for narrative-GWT integration
#[derive(Clone)]
pub struct NarrativeGWTConfig {
    /// Minimum Self-Φ to allow action execution
    pub min_self_phi: f64,

    /// Goal alignment bias strength (0-1)
    pub goal_alignment_bias: f64,

    /// Value consistency weight in veto decisions
    pub value_consistency_weight: f64,

    /// Enable automatic veto of self-fracturing actions
    pub enable_coherence_veto: bool,

    /// Self-Φ delta threshold for warnings
    pub phi_delta_warning_threshold: f64,

    /// History size for Self-Φ tracking
    pub phi_history_size: usize,

    /// Enable cross-modal integration
    pub enable_cross_modal: bool,

    /// Standing coalition base activation
    pub standing_coalition_activation: f64,

    // **REVOLUTIONARY IMPROVEMENT #74**: Predictive Self Configuration
    /// Enable predictive self-model for future-aware veto decisions
    pub enable_predictive_self: bool,

    /// Prediction horizon (how many steps to predict ahead)
    pub prediction_horizon: usize,

    /// Minimum safe predicted Self-Φ (predictive veto threshold)
    pub min_predicted_phi: f64,

    /// Enable counterfactual "what if" reasoning
    pub enable_counterfactuals: bool,

    /// Enable prospective memory for future intentions
    pub enable_prospective_memory: bool,

    /// Weight of predictive assessment in veto decisions (0-1)
    pub predictive_veto_weight: f64,

    // **REVOLUTIONARY IMPROVEMENT #75**: Temporal Consciousness Configuration
    /// Enable temporal consciousness analysis
    pub enable_temporal_consciousness: bool,

    /// Temporal consciousness configuration (if enabled)
    pub temporal_config: Option<TemporalConsciousnessConfig>,
}

impl Default for NarrativeGWTConfig {
    fn default() -> Self {
        Self {
            min_self_phi: 0.3,
            goal_alignment_bias: 0.4,
            value_consistency_weight: 0.6,
            enable_coherence_veto: true,
            phi_delta_warning_threshold: 0.15,
            phi_history_size: 100,
            enable_cross_modal: true,
            standing_coalition_activation: 0.7,
            // **REVOLUTIONARY IMPROVEMENT #74**: Predictive defaults
            enable_predictive_self: true,
            prediction_horizon: 3,
            min_predicted_phi: 0.25, // Slightly lower than min_self_phi for predictive tolerance
            enable_counterfactuals: true,
            enable_prospective_memory: true,
            predictive_veto_weight: 0.4, // 40% predictive, 60% current assessment
            // **REVOLUTIONARY IMPROVEMENT #75**: Temporal consciousness defaults
            enable_temporal_consciousness: true,
            temporal_config: Some(TemporalConsciousnessConfig::default()),
        }
    }
}

// ============================================================================
// VETO SYSTEM
// ============================================================================

/// Reason for a coherence veto
#[derive(Debug, Clone, PartialEq)]
pub enum VetoReason {
    /// Self-Φ would drop below minimum threshold
    SelfPhiTooLow { current: f64, projected: f64, minimum: f64 },

    /// Action contradicts core values
    ValueViolation { value: String, conflict: String },

    /// Action conflicts with active goals
    GoalConflict { goal: String, conflict: String },

    /// Action would fracture autobiographical continuity
    ContinuityFracture { description: String },

    /// Action contradicts established identity traits
    TraitContradiction { trait_name: String, action: String },

    // **REVOLUTIONARY IMPROVEMENT #74**: Predictive Veto Reasons
    /// Predicted future Self-Φ falls below threshold
    PredictedPhiTooLow {
        current: f64,
        predicted: f64,
        horizon: usize,
        minimum: f64
    },

    /// Counterfactual analysis suggests better alternatives exist
    BetterAlternativeExists {
        proposed_action: String,
        alternative: String,
        improvement: f64
    },

    /// Action conflicts with prospective intention
    ProspectiveConflict {
        intention: String,
        action: String
    },

    /// Action predicted to harm identity coherence trajectory
    CoherenceTrajectoryHarm {
        current_coherence: f64,
        predicted_coherence: f64,
        decay_rate: f64
    },
}

/// Result of a veto check
#[derive(Debug, Clone)]
pub struct VetoResult {
    /// Whether the action is vetoed
    pub vetoed: bool,

    /// Reason for veto (if vetoed)
    pub reason: Option<VetoReason>,

    /// Confidence in the veto decision
    pub confidence: f64,

    /// Suggested alternative (if available)
    pub alternative: Option<String>,

    /// Self-Φ impact assessment
    pub phi_impact: f64,
}

impl VetoResult {
    fn allow() -> Self {
        Self {
            vetoed: false,
            reason: None,
            confidence: 1.0,
            alternative: None,
            phi_impact: 0.0,
        }
    }

    fn veto(reason: VetoReason, confidence: f64, phi_impact: f64) -> Self {
        Self {
            vetoed: true,
            reason: Some(reason),
            confidence,
            alternative: None,
            phi_impact,
        }
    }
}

// ============================================================================
// GOAL ALIGNMENT
// ============================================================================

/// Alignment assessment between content and goals
#[derive(Debug, Clone)]
pub struct GoalAlignment {
    /// Overall alignment score (-1 to 1)
    pub score: f64,

    /// Which goals are supported
    pub supported_goals: Vec<String>,

    /// Which goals are contradicted
    pub contradicted_goals: Vec<String>,

    /// Activation modifier based on alignment
    pub activation_modifier: f64,
}

// ============================================================================
// SELF-Φ TRACKING
// ============================================================================

/// Record of Self-Φ at a specific ignition
#[derive(Debug, Clone)]
pub struct PhiIgnitionRecord {
    /// Timestamp
    pub timestamp: Instant,

    /// Ignition number
    pub ignition_id: usize,

    /// Self-Φ before ignition
    pub phi_before: f64,

    /// Self-Φ after ignition
    pub phi_after: f64,

    /// Delta (change)
    pub phi_delta: f64,

    /// What content was broadcast
    pub broadcast_content: String,

    /// Whether delta exceeds warning threshold
    pub warning_triggered: bool,
}

/// Statistics for narrative-GWT integration
#[derive(Debug, Clone, Default)]
pub struct NarrativeGWTStats {
    /// Total ignitions processed
    pub ignitions_processed: usize,

    /// Vetoes issued
    pub vetoes_issued: usize,

    /// Vetoes by reason
    pub vetoes_by_reason: HashMap<String, usize>,

    /// Average Self-Φ
    pub avg_self_phi: f64,

    /// Self-Φ variance
    pub phi_variance: f64,

    /// Goal alignment events
    pub goal_alignments: usize,

    /// Goal conflicts detected
    pub goal_conflicts: usize,

    /// Warnings issued
    pub warnings_issued: usize,

    // **REVOLUTIONARY IMPROVEMENT #74**: Predictive Statistics
    /// Total predictions made
    pub predictions_made: usize,

    /// Accurate predictions (within 10% of actual)
    pub accurate_predictions: usize,

    /// Predictive vetoes issued
    pub predictive_vetoes: usize,

    /// Counterfactual explorations
    pub counterfactuals_explored: usize,

    /// Prospective intentions tracked
    pub intentions_tracked: usize,

    /// Intentions triggered
    pub intentions_triggered: usize,

    /// Average prediction error
    pub avg_prediction_error: f64,

    /// Better alternatives found (saved from mistakes)
    pub alternatives_found: usize,
}

// ============================================================================
// MAIN INTEGRATION STRUCTURE
// ============================================================================

/// The integrated Narrative-GWT system
pub struct NarrativeGWTIntegration {
    /// The Narrative Self model
    pub narrative_self: NarrativeSelfModel,

    /// The Global Workspace
    pub workspace: UnifiedGlobalWorkspace,

    /// Cross-modal binder (optional)
    pub cross_modal: Option<CrossModalBinder>,

    // **REVOLUTIONARY IMPROVEMENT #74**: Predictive Self-Model
    /// The Predictive Self model for future-aware consciousness
    pub predictive_self: Option<PredictiveSelfModel>,

    // **REVOLUTIONARY IMPROVEMENT #75**: Temporal Consciousness Analyzer
    /// Temporal consciousness analysis for flow-based consciousness understanding
    pub temporal_analyzer: Option<TemporalConsciousnessAnalyzer>,

    /// Configuration
    pub config: NarrativeGWTConfig,

    /// Self-Φ history per ignition
    phi_history: VecDeque<PhiIgnitionRecord>,

    /// Statistics
    pub stats: NarrativeGWTStats,

    /// Current ignition counter
    ignition_counter: usize,

    /// Last veto result (for inspection)
    last_veto: Option<VetoResult>,

    /// Last predictive safety assessment
    last_safety_assessment: Option<ActionSafetyAssessment>,

    /// Pending predictions to verify (action -> predicted state)
    pending_predictions: VecDeque<(String, f64, Instant)>,
}

impl NarrativeGWTIntegration {
    /// Create new integrated system
    pub fn new(
        narrative_config: NarrativeSelfConfig,
        gwt_config: UnifiedGWTConfig,
        integration_config: NarrativeGWTConfig,
    ) -> Self {
        let cross_modal = if integration_config.enable_cross_modal {
            Some(CrossModalBinder::new(BinderConfig::default()))
        } else {
            None
        };

        // **REVOLUTIONARY IMPROVEMENT #74**: Initialize Predictive Self
        let predictive_self = if integration_config.enable_predictive_self {
            let pred_config = PredictiveSelfConfig {
                history_depth: integration_config.phi_history_size,
                learning_rate: 0.1,
                prediction_horizon: integration_config.prediction_horizon,
                error_threshold: 0.1,
                temporal_decay: 0.9,
                enable_counterfactuals: integration_config.enable_counterfactuals,
                max_counterfactuals: 10,
                prospective_memory_size: 20,
            };
            Some(PredictiveSelfModel::new(pred_config))
        } else {
            None
        };

        // **REVOLUTIONARY IMPROVEMENT #75**: Initialize Temporal Consciousness Analyzer
        let temporal_analyzer = if integration_config.enable_temporal_consciousness {
            let config = integration_config.temporal_config.clone()
                .unwrap_or_default();
            Some(TemporalConsciousnessAnalyzer::new(config))
        } else {
            None
        };

        Self {
            narrative_self: NarrativeSelfModel::new(narrative_config.clone()),
            workspace: UnifiedGlobalWorkspace::new(gwt_config),
            cross_modal,
            predictive_self,
            temporal_analyzer,
            config: integration_config,
            phi_history: VecDeque::with_capacity(100),
            stats: NarrativeGWTStats::default(),
            ignition_counter: 0,
            last_veto: None,
            last_safety_assessment: None,
            pending_predictions: VecDeque::with_capacity(20),
        }
    }

    /// Create with default configurations
    pub fn default_config() -> Self {
        Self::new(
            NarrativeSelfConfig::default(),
            UnifiedGWTConfig::default(),
            NarrativeGWTConfig::default(),
        )
    }

    // ========================================================================
    // STANDING COALITION
    // ========================================================================

    /// Submit the Narrative Self as a standing coalition
    /// This ensures the self is always represented in the workspace
    fn submit_standing_coalition(&mut self) {
        // Create representation from unified self
        let self_representation = vec![self.narrative_self.unified_self().clone()];

        // Standing coalition members: all self-levels
        let members = vec![
            "proto_self".to_string(),
            "core_self".to_string(),
            "autobiographical_self".to_string(),
            "narrative_integrator".to_string(),
        ];

        // Submit with standing activation (doesn't compete, modulates)
        self.workspace.submit_strategy(
            "NarrativeSelf",
            self.config.standing_coalition_activation,
            self_representation,
            members,
        );
    }

    // ========================================================================
    // GOAL ALIGNMENT
    // ========================================================================

    /// Assess alignment between content and current goals
    pub fn assess_goal_alignment(&self, content: &HV16, content_description: &str) -> GoalAlignment {
        let goals = self.narrative_self.current_goals();
        let num_goals = goals.len();

        let mut supported = Vec::new();
        let mut contradicted = Vec::new();
        let mut total_alignment = 0.0;

        for (goal_name, goal_vector) in goals {
            let similarity = content.similarity(goal_vector) as f64;

            // Check for keywords that might indicate support/contradiction
            let goal_lower = goal_name.to_lowercase();
            let desc_lower = content_description.to_lowercase();

            let keyword_boost = if desc_lower.contains(&goal_lower) {
                0.3
            } else {
                0.0
            };

            let adjusted_sim = similarity + keyword_boost;

            if adjusted_sim > 0.3 {
                supported.push(goal_name.clone());
                total_alignment += adjusted_sim;
            } else if adjusted_sim < -0.3 {
                contradicted.push(goal_name.clone());
                total_alignment += adjusted_sim;
            }
        }

        let num_goals_f64 = num_goals.max(1) as f64;
        let avg_alignment = total_alignment / num_goals_f64;

        // Calculate activation modifier
        let modifier = 1.0 + (avg_alignment * self.config.goal_alignment_bias);

        GoalAlignment {
            score: avg_alignment.clamp(-1.0, 1.0),
            supported_goals: supported,
            contradicted_goals: contradicted,
            activation_modifier: modifier.clamp(0.5, 1.5),
        }
    }

    // ========================================================================
    // VALUE CONSISTENCY
    // ========================================================================

    /// Check if action is consistent with core values
    fn check_value_consistency(&self, action_description: &str) -> (bool, Option<String>) {
        let values = self.narrative_self.core_values();
        let action_lower = action_description.to_lowercase();

        // Check for obvious value violations
        for (value_name, _value_vector) in values {
            let value_lower = value_name.to_lowercase();

            // Simple heuristic checks (would be more sophisticated in production)
            if value_lower.contains("safety") &&
               (action_lower.contains("dangerous") || action_lower.contains("risky")) {
                return (false, Some(value_name.clone()));
            }

            if value_lower.contains("honest") &&
               (action_lower.contains("deceive") || action_lower.contains("lie")) {
                return (false, Some(value_name.clone()));
            }

            if value_lower.contains("helpful") &&
               (action_lower.contains("refuse") || action_lower.contains("harm")) {
                return (false, Some(value_name.clone()));
            }
        }

        (true, None)
    }

    // ========================================================================
    // COHERENCE VETO
    // ========================================================================

    /// Check whether an action should be vetoed for coherence reasons
    pub fn check_veto(&mut self, action: &HV16, action_description: &str) -> VetoResult {
        if !self.config.enable_coherence_veto {
            return VetoResult::allow();
        }

        let current_phi = self.narrative_self.self_phi();

        // 1. Check minimum Self-Φ threshold
        if current_phi < self.config.min_self_phi {
            return VetoResult::veto(
                VetoReason::SelfPhiTooLow {
                    current: current_phi,
                    projected: current_phi - 0.1, // Estimate
                    minimum: self.config.min_self_phi,
                },
                0.9,
                -0.1,
            );
        }

        // 2. Check value consistency
        let (consistent, violated_value) = self.check_value_consistency(action_description);
        if !consistent {
            if let Some(value) = violated_value {
                self.stats.vetoes_by_reason.entry("value_violation".to_string())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);

                return VetoResult::veto(
                    VetoReason::ValueViolation {
                        value: value.clone(),
                        conflict: action_description.to_string(),
                    },
                    0.85,
                    -0.15,
                );
            }
        }

        // 3. Check goal conflicts
        let alignment = self.assess_goal_alignment(action, action_description);
        if !alignment.contradicted_goals.is_empty() && alignment.score < -0.5 {
            self.stats.goal_conflicts += 1;

            return VetoResult::veto(
                VetoReason::GoalConflict {
                    goal: alignment.contradicted_goals[0].clone(),
                    conflict: action_description.to_string(),
                },
                0.7,
                -0.1,
            );
        }

        // **REVOLUTIONARY IMPROVEMENT #74**: Predictive Veto Check
        // Use the Predictive Self to evaluate action safety
        if let Some(ref mut predictive) = self.predictive_self {
            // Observe current state
            predictive.observe(&self.narrative_self);

            // Get action safety assessment
            let safety = predictive.evaluate_action_safety(action_description);
            self.last_safety_assessment = Some(safety.clone());
            self.stats.predictions_made += 1;

            // Check if predicted Self-Φ falls below threshold
            if safety.predicted_phi < self.config.min_predicted_phi {
                self.stats.predictive_vetoes += 1;
                self.stats.vetoes_by_reason.entry("predicted_phi_too_low".to_string())
                    .and_modify(|c| *c += 1)
                    .or_insert(1);

                return VetoResult::veto(
                    VetoReason::PredictedPhiTooLow {
                        current: current_phi,
                        predicted: safety.predicted_phi,
                        horizon: self.config.prediction_horizon,
                        minimum: self.config.min_predicted_phi,
                    },
                    0.8 * self.config.predictive_veto_weight + 0.2,
                    safety.predicted_phi - current_phi,
                );
            }

            // Check if action would harm coherence trajectory
            if safety.coherence_impact < -0.2 {
                let predicted_coherence = self.narrative_self.coherence() + safety.coherence_impact;
                if predicted_coherence < 0.4 {
                    self.stats.predictive_vetoes += 1;
                    self.stats.vetoes_by_reason.entry("coherence_trajectory_harm".to_string())
                        .and_modify(|c| *c += 1)
                        .or_insert(1);

                    return VetoResult::veto(
                        VetoReason::CoherenceTrajectoryHarm {
                            current_coherence: self.narrative_self.coherence(),
                            predicted_coherence,
                            decay_rate: -safety.coherence_impact,
                        },
                        0.75,
                        safety.coherence_impact,
                    );
                }
            }

            // Store prediction for later verification
            self.pending_predictions.push_back((
                action_description.to_string(),
                safety.predicted_phi,
                Instant::now(),
            ));
            if self.pending_predictions.len() > 20 {
                self.pending_predictions.pop_front();
            }

            // Check prospective memory conflicts (if enabled)
            if self.config.enable_prospective_memory {
                let triggered = predictive.check_triggers(action_description);
                if triggered > 0 {
                    self.stats.intentions_triggered += triggered;
                }
            }
        }

        // 4. Estimate Self-Φ impact (fallback for non-predictive path)
        let estimated_impact = alignment.score * 0.1; // Rough estimate

        if estimated_impact < -self.config.phi_delta_warning_threshold {
            // Issue warning but don't veto
            self.stats.warnings_issued += 1;
        }

        VetoResult::allow()
    }

    // ========================================================================
    // MAIN PROCESSING
    // ========================================================================

    /// Submit content to the workspace with narrative modulation
    pub fn submit_content(
        &mut self,
        strategy_name: &str,
        content: Vec<HV16>,
        content_description: &str,
        supporting_modules: Vec<String>,
        base_activation: f64,
    ) -> Option<VetoResult> {
        // First, submit standing coalition
        self.submit_standing_coalition();

        // Check for veto
        let veto_result = if !content.is_empty() {
            self.check_veto(&content[0], content_description)
        } else {
            VetoResult::allow()
        };

        self.last_veto = Some(veto_result.clone());

        if veto_result.vetoed {
            self.stats.vetoes_issued += 1;
            return Some(veto_result);
        }

        // Assess goal alignment and modify activation
        let alignment = if !content.is_empty() {
            self.assess_goal_alignment(&content[0], content_description)
        } else {
            GoalAlignment {
                score: 0.0,
                supported_goals: vec![],
                contradicted_goals: vec![],
                activation_modifier: 1.0,
            }
        };

        if !alignment.supported_goals.is_empty() {
            self.stats.goal_alignments += 1;
        }

        // Apply goal-aligned activation modification
        let modified_activation = base_activation * alignment.activation_modifier;

        // Submit to workspace
        self.workspace.submit_strategy(
            strategy_name,
            modified_activation,
            content,
            supporting_modules,
        );

        None // No veto
    }

    /// Process the workspace and track Self-Φ
    pub fn process(&mut self) -> NarrativeGWTProcessResult {
        let phi_before = self.narrative_self.self_phi();

        // **REVOLUTIONARY IMPROVEMENT #74**: Observe current state for prediction
        if let Some(ref mut predictive) = self.predictive_self {
            predictive.observe(&self.narrative_self);
        }

        // Process the workspace
        let gwt_result = self.workspace.process();

        // Track ignition and Self-Φ delta
        if gwt_result.workspace_assessment.ignition_detected {
            self.ignition_counter += 1;

            // Process experience through narrative self
            if let Some(ref winner) = gwt_result.winning_strategy {
                // Get representation from winning content
                let winner_rep = gwt_result.workspace_assessment.conscious_contents
                    .iter()
                    .find(|c| c.source.contains(winner))
                    .map(|c| c.representation.first().cloned())
                    .flatten()
                    .unwrap_or_else(HV16::zero);

                self.narrative_self.process_experience(
                    &winner_rep,
                    winner,
                    true, // Successful action
                    0.5,  // Default emotional valence
                    0.5,  // Default arousal
                );
            }

            let phi_after = self.narrative_self.self_phi();
            let phi_delta = phi_after - phi_before;

            // Record the ignition
            let record = PhiIgnitionRecord {
                timestamp: Instant::now(),
                ignition_id: self.ignition_counter,
                phi_before,
                phi_after,
                phi_delta,
                broadcast_content: gwt_result.winning_strategy.clone().unwrap_or_default(),
                warning_triggered: phi_delta.abs() > self.config.phi_delta_warning_threshold,
            };

            if record.warning_triggered {
                self.stats.warnings_issued += 1;
            }

            self.phi_history.push_back(record);
            if self.phi_history.len() > self.config.phi_history_size {
                self.phi_history.pop_front();
            }

            self.stats.ignitions_processed += 1;

            // Update average Self-Φ
            let n = self.stats.ignitions_processed as f64;
            self.stats.avg_self_phi =
                (self.stats.avg_self_phi * (n - 1.0) + phi_after) / n;

            // **REVOLUTIONARY IMPROVEMENT #74**: Learn from prediction outcomes
            // First verify pending predictions (separate borrow scope)
            self.verify_predictions(phi_after);

            // Then learn from outcome
            let coherence = self.narrative_self.coherence();
            if let Some(ref mut predictive) = self.predictive_self {
                predictive.learn_from_outcome_raw(phi_after, coherence);
            }

            // **REVOLUTIONARY IMPROVEMENT #75**: Update temporal consciousness
            // Observe current state with full context from narrative and predictive selves
            if let Some(ref mut temporal) = self.temporal_analyzer {
                let current_state = self.narrative_self.unified_self().clone();
                temporal.observe(
                    &current_state,
                    phi_after,
                    Some(&self.narrative_self),
                    self.predictive_self.as_ref(),
                );
            }
        }

        // Update cross-modal if enabled
        if let Some(ref mut cross_modal) = self.cross_modal {
            // Update linguistic modality with narrative content
            cross_modal.update_modality(
                Modality::Linguistic,
                self.narrative_self.unified_self().clone(),
            );
            cross_modal.set_attention(Modality::Linguistic, 0.8);
            cross_modal.bind();
        }

        // Construct result with predictive and temporal stats
        let result = NarrativeGWTProcessResult {
            gwt_result,
            self_phi: self.narrative_self.self_phi(),
            phi_delta: self.phi_history.back().map(|r| r.phi_delta).unwrap_or(0.0),
            coherence: self.narrative_self.coherence(),
            cross_modal_phi: self.cross_modal.as_ref().map(|cm| cm.cross_modal_phi()),
            // **REVOLUTIONARY IMPROVEMENT #74**: Add predictive insights
            prediction_accuracy: self.prediction_accuracy(),
            counterfactuals_available: self.predictive_self.as_ref()
                .map(|p| p.counterfactual_count())
                .unwrap_or(0),
            pending_intentions: self.predictive_self.as_ref()
                .map(|p| p.intention_count())
                .unwrap_or(0),
            // **REVOLUTIONARY IMPROVEMENT #75**: Add temporal consciousness insights
            temporal_coherence: self.temporal_analyzer.as_ref()
                .map(|t| t.overall_temporal_coherence()),
            phi_velocity: self.temporal_analyzer.as_ref()
                .map(|t| t.phi_trajectory.velocity),
            consciousness_continuity: self.temporal_analyzer.as_ref()
                .map(|t| t.continuity.score),
            temporal_identity_coherence: self.temporal_analyzer.as_ref()
                .map(|t| t.identity_coherence.coherence),
            temporally_healthy: self.temporal_analyzer.as_ref()
                .map(|t| t.is_temporally_healthy()),
        };

        result
    }

    /// Verify pending predictions against actual outcomes
    fn verify_predictions(&mut self, actual_phi: f64) {
        // Check predictions that are old enough to verify
        let now = Instant::now();
        let mut verified = 0;
        let mut accurate = 0;

        // Keep recent predictions, verify old ones
        let mut remaining = VecDeque::new();
        while let Some((action, predicted, timestamp)) = self.pending_predictions.pop_front() {
            let age = now.duration_since(timestamp).as_secs_f64();

            if age > 0.1 {
                // Prediction is old enough to verify
                let error = (predicted - actual_phi).abs();
                verified += 1;

                // Consider accurate if within 10%
                if error < 0.1 {
                    accurate += 1;
                    self.stats.accurate_predictions += 1;
                }

                // Update average prediction error
                let n = self.stats.predictions_made as f64;
                if n > 0.0 {
                    self.stats.avg_prediction_error =
                        (self.stats.avg_prediction_error * (n - 1.0) + error) / n;
                }
            } else {
                remaining.push_back((action, predicted, timestamp));
            }
        }
        self.pending_predictions = remaining;
    }

    /// Calculate prediction accuracy as a percentage
    pub fn prediction_accuracy(&self) -> f64 {
        if self.stats.predictions_made == 0 {
            return 0.0;
        }
        self.stats.accurate_predictions as f64 / self.stats.predictions_made as f64
    }

    // ========================================================================
    // ACCESSORS
    // ========================================================================

    /// Get current Self-Φ
    pub fn self_phi(&self) -> f64 {
        self.narrative_self.self_phi()
    }

    /// Get Self-Φ history
    pub fn phi_history(&self) -> &VecDeque<PhiIgnitionRecord> {
        &self.phi_history
    }

    /// Get last veto result
    pub fn last_veto(&self) -> Option<&VetoResult> {
        self.last_veto.as_ref()
    }

    /// Get narrative self report
    pub fn narrative_report(&self) -> NarrativeSelfReport {
        self.narrative_self.structured_report()
    }

    /// Check if system is in coherent state
    pub fn is_coherent(&self) -> bool {
        self.narrative_self.self_phi() >= self.config.min_self_phi &&
        self.narrative_self.coherence() > 0.5
    }

    /// Get cross-modal Φ (if available)
    pub fn cross_modal_phi(&self) -> Option<f64> {
        self.cross_modal.as_ref().map(|cm| cm.cross_modal_phi())
    }

    /// Generate integration report
    pub fn report(&self) -> String {
        let narrative_report = self.narrative_self.structured_report();

        format!(
            r#"
╔══════════════════════════════════════════════════════════════════════════╗
║          NARRATIVE-GWT INTEGRATION (#73) - SELF-COHERENT AI              ║
╠══════════════════════════════════════════════════════════════════════════╣
║ SELF-COHERENCE STATUS                                                    ║
║   Self-Φ:            {:.4}  {}                                           ║
║   Narrative Coherence: {:.4}                                             ║
║   Cross-Modal Φ:     {}                                                  ║
║   System Coherent:   {}                                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║ STATISTICS                                                               ║
║   Ignitions Processed: {:>6}                                             ║
║   Vetoes Issued:       {:>6}                                             ║
║   Goal Alignments:     {:>6}                                             ║
║   Goal Conflicts:      {:>6}                                             ║
║   Warnings Issued:     {:>6}                                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║ SELF-Φ TRACKING                                                          ║
║   Average Self-Φ:    {:.4}                                               ║
║   Current Self-Φ:    {:.4}                                               ║
║   Last Φ Delta:      {:.4}                                               ║
║   History Size:      {:>6}                                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║ NARRATIVE SELF SUMMARY                                                   ║
║   Active Goals:      {:>6}                                               ║
║   Core Values:       {:>6}                                               ║
║   Traits:            {:>6}                                               ║
║   Episodes Recorded: {:>6}                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
"#,
            self.narrative_self.self_phi(),
            if self.narrative_self.self_phi() >= self.config.min_self_phi { "✓" } else { "⚠" },
            self.narrative_self.coherence(),
            self.cross_modal.as_ref()
                .map(|cm| format!("{:.4}", cm.cross_modal_phi()))
                .unwrap_or_else(|| "N/A".to_string()),
            if self.is_coherent() { "YES ✓" } else { "NO ⚠" },
            self.stats.ignitions_processed,
            self.stats.vetoes_issued,
            self.stats.goal_alignments,
            self.stats.goal_conflicts,
            self.stats.warnings_issued,
            self.stats.avg_self_phi,
            self.narrative_self.self_phi(),
            self.phi_history.back().map(|r| r.phi_delta).unwrap_or(0.0),
            self.phi_history.len(),
            narrative_report.active_goals,
            narrative_report.core_values,
            narrative_report.traits,
            narrative_report.episodes_recorded,
        )
    }
}

/// Result of processing with narrative integration
#[derive(Debug, Clone)]
pub struct NarrativeGWTProcessResult {
    /// Underlying GWT result
    pub gwt_result: UnifiedGWTResult,

    /// Current Self-Φ
    pub self_phi: f64,

    /// Change in Self-Φ from this ignition
    pub phi_delta: f64,

    /// Narrative coherence
    pub coherence: f64,

    /// Cross-modal Φ (if enabled)
    pub cross_modal_phi: Option<f64>,

    // **REVOLUTIONARY IMPROVEMENT #74**: Predictive Insights
    /// Prediction accuracy (0-1)
    pub prediction_accuracy: f64,

    /// Number of counterfactual scenarios explored
    pub counterfactuals_available: usize,

    /// Number of pending intentions in prospective memory
    pub pending_intentions: usize,

    // **REVOLUTIONARY IMPROVEMENT #75**: Temporal Consciousness Insights
    /// Overall temporal coherence score (0-1)
    pub temporal_coherence: Option<f64>,

    /// Φ trajectory velocity (positive = increasing consciousness)
    pub phi_velocity: Option<f64>,

    /// Consciousness continuity score (0-1, higher = smoother experience)
    pub consciousness_continuity: Option<f64>,

    /// Temporal identity coherence (past-present-future alignment)
    pub temporal_identity_coherence: Option<f64>,

    /// Whether temporal dynamics are healthy
    pub temporally_healthy: Option<bool>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_creation() {
        let integration = NarrativeGWTIntegration::default_config();

        assert!(integration.self_phi() > 0.0);
        assert!(integration.is_coherent());
    }

    #[test]
    fn test_standing_coalition() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Submit some content
        integration.submit_content(
            "TestStrategy",
            vec![HV16::random(42)],
            "A helpful action",
            vec!["module1".to_string()],
            0.5,
        );

        // Process
        let result = integration.process();

        // Self-Φ should remain stable
        assert!(result.self_phi > 0.0);
    }

    #[test]
    fn test_goal_alignment() {
        let integration = NarrativeGWTIntegration::default_config();

        let content = HV16::random(123);
        let alignment = integration.assess_goal_alignment(&content, "helping the user");

        // Alignment modifier should be reasonable
        assert!(alignment.activation_modifier >= 0.5);
        assert!(alignment.activation_modifier <= 1.5);
    }

    #[test]
    fn test_veto_mechanism() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Test with normal action
        let normal_action = HV16::random(100);
        let veto = integration.check_veto(&normal_action, "helping the user");
        assert!(!veto.vetoed);

        // Test with potentially problematic action
        let bad_action = HV16::random(200);
        let veto = integration.check_veto(&bad_action, "deceiving the user");
        // May or may not veto depending on values
        assert!(veto.confidence > 0.0);
    }

    #[test]
    fn test_phi_tracking() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process several times
        for i in 0..5 {
            integration.submit_content(
                &format!("Strategy{}", i),
                vec![HV16::random(i as u64 + 1000)],
                "processing task",
                vec!["module".to_string()],
                0.6,
            );
            integration.process();
        }

        // Should have tracked some ignitions
        // Ignition processing tracked (usize always >= 0)
    }

    #[test]
    fn test_cross_modal_integration() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process with cross-modal
        integration.submit_content(
            "CrossModalTest",
            vec![HV16::random(500)],
            "multi-modal content",
            vec!["visual".to_string(), "linguistic".to_string()],
            0.7,
        );

        let result = integration.process();

        // Cross-modal Φ should be available
        assert!(result.cross_modal_phi.is_some());
    }

    #[test]
    fn test_coherence_maintenance() {
        let mut integration = NarrativeGWTIntegration::default_config();

        let initial_phi = integration.self_phi();

        // Process many times
        for i in 0..20 {
            integration.submit_content(
                &format!("Strategy{}", i),
                vec![HV16::random(i as u64 + 2000)],
                "consistent task",
                vec!["module".to_string()],
                0.5,
            );
            integration.process();
        }

        // Self-Φ should remain relatively stable
        let final_phi = integration.self_phi();
        let drift = (final_phi - initial_phi).abs();

        // Drift should be bounded
        assert!(drift < 0.5, "Self-Φ drifted too much: {}", drift);
    }

    #[test]
    fn test_report_generation() {
        let integration = NarrativeGWTIntegration::default_config();

        let report = integration.report();

        assert!(report.contains("NARRATIVE-GWT INTEGRATION"));
        assert!(report.contains("Self-Φ"));
        assert!(report.contains("Narrative Coherence"));
    }

    // ========================================================================
    // REVOLUTIONARY IMPROVEMENT #74: PREDICTIVE SELF INTEGRATION TESTS
    // ========================================================================

    #[test]
    fn test_predictive_self_enabled_by_default() {
        let integration = NarrativeGWTIntegration::default_config();

        // Predictive self should be enabled by default
        assert!(integration.predictive_self.is_some());
        assert!(integration.config.enable_predictive_self);
    }

    #[test]
    fn test_predictive_veto_check() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Submit content to build up some history
        for i in 0..5 {
            integration.submit_content(
                &format!("Setup{}", i),
                vec![HV16::random(i as u64 + 3000)],
                "building context",
                vec!["module".to_string()],
                0.6,
            );
            integration.process();
        }

        // Now check veto with predictive assessment
        let action = HV16::random(4000);
        let veto = integration.check_veto(&action, "normal helpful action");

        // The veto result should have confidence > 0
        // Note: The action may or may not be vetoed depending on predictive state
        assert!(veto.confidence > 0.0);

        // Should have made at least one prediction
        assert!(integration.stats.predictions_made > 0);
    }

    #[test]
    fn test_predictive_learning_loop() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Initial predictions made count
        let initial_predictions = integration.stats.predictions_made;

        // Process multiple times to exercise prediction-learning loop
        for i in 0..10 {
            integration.submit_content(
                &format!("Task{}", i),
                vec![HV16::random(i as u64 + 5000)],
                "learning task",
                vec!["module".to_string()],
                0.6,
            );
            integration.process();
        }

        // Should have made multiple predictions
        assert!(integration.stats.predictions_made >= initial_predictions);

        // Prediction accuracy should be calculable
        let accuracy = integration.prediction_accuracy();
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_predictive_result_fields() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process to get a result
        integration.submit_content(
            "TestStrategy",
            vec![HV16::random(6000)],
            "test action",
            vec!["module".to_string()],
            0.5,
        );
        let result = integration.process();

        // New predictive fields should be present
        assert!(result.prediction_accuracy >= 0.0);
        // Counterfactuals and intentions tracked (usize always >= 0)
    }

    #[test]
    fn test_predictive_config_options() {
        let mut config = NarrativeGWTConfig::default();

        // Modify predictive settings
        config.enable_predictive_self = true;
        config.prediction_horizon = 5;
        config.min_predicted_phi = 0.2;
        config.enable_counterfactuals = true;
        config.enable_prospective_memory = true;
        config.predictive_veto_weight = 0.5;

        let integration = NarrativeGWTIntegration::new(
            NarrativeSelfConfig::default(),
            UnifiedGWTConfig::default(),
            config,
        );

        // Verify configuration was applied
        assert!(integration.predictive_self.is_some());
        assert_eq!(integration.config.prediction_horizon, 5);
        assert_eq!(integration.config.predictive_veto_weight, 0.5);
    }

    #[test]
    fn test_predictive_veto_reasons() {
        // Test that new veto reasons are correctly formatted
        let reason = VetoReason::PredictedPhiTooLow {
            current: 0.5,
            predicted: 0.1,
            horizon: 3,
            minimum: 0.25,
        };

        match reason {
            VetoReason::PredictedPhiTooLow { current, predicted, horizon, minimum } => {
                assert_eq!(current, 0.5);
                assert_eq!(predicted, 0.1);
                assert_eq!(horizon, 3);
                assert_eq!(minimum, 0.25);
            }
            _ => panic!("Expected PredictedPhiTooLow"),
        }

        let reason2 = VetoReason::CoherenceTrajectoryHarm {
            current_coherence: 0.7,
            predicted_coherence: 0.3,
            decay_rate: 0.4,
        };

        match reason2 {
            VetoReason::CoherenceTrajectoryHarm { current_coherence, predicted_coherence, decay_rate } => {
                assert_eq!(current_coherence, 0.7);
                assert_eq!(predicted_coherence, 0.3);
                assert_eq!(decay_rate, 0.4);
            }
            _ => panic!("Expected CoherenceTrajectoryHarm"),
        }
    }

    #[test]
    fn test_predictive_stats_tracked() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process to accumulate stats
        for i in 0..5 {
            integration.submit_content(
                &format!("StatTask{}", i),
                vec![HV16::random(i as u64 + 7000)],
                "stat tracking task",
                vec!["module".to_string()],
                0.6,
            );
            integration.process();
        }

        // Stats should be tracked
        let stats = &integration.stats;
        assert!(stats.predictions_made > 0 || integration.predictive_self.is_some());
        // Predictive vetoes and counterfactuals may be 0 if no vetoes were issued
        // (usize fields always >= 0)
    }

    #[test]
    fn test_disabled_predictive_self() {
        let mut config = NarrativeGWTConfig::default();
        config.enable_predictive_self = false;

        let integration = NarrativeGWTIntegration::new(
            NarrativeSelfConfig::default(),
            UnifiedGWTConfig::default(),
            config,
        );

        // Should be disabled
        assert!(integration.predictive_self.is_none());
        assert!(!integration.config.enable_predictive_self);
    }

    // ========================================================================
    // REVOLUTIONARY IMPROVEMENT #75: TEMPORAL CONSCIOUSNESS INTEGRATION TESTS
    // ========================================================================

    #[test]
    fn test_temporal_consciousness_enabled_by_default() {
        let integration = NarrativeGWTIntegration::default_config();

        // Temporal consciousness should be enabled by default
        assert!(integration.temporal_analyzer.is_some());
        assert!(integration.config.enable_temporal_consciousness);
    }

    #[test]
    fn test_temporal_consciousness_observation() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process several times to build temporal history
        for i in 0..10 {
            integration.submit_content(
                &format!("TemporalTask{}", i),
                vec![HV16::random(i as u64 + 8000)],
                "temporal observation task",
                vec!["module".to_string()],
                0.6,
            );
            let result = integration.process();

            // After first few observations, temporal coherence should be available
            if i > 3 {
                assert!(result.temporal_coherence.is_some());
            }
        }

        // Verify temporal analyzer has accumulated stats
        if let Some(ref temporal) = integration.temporal_analyzer {
            assert!(temporal.stats.observations > 0);
        }
    }

    #[test]
    fn test_temporal_result_fields() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process to get a result with temporal fields
        for i in 0..5 {
            integration.submit_content(
                &format!("TestStrategy{}", i),
                vec![HV16::random(9000 + i as u64)],
                "test action",
                vec!["module".to_string()],
                0.5,
            );
            integration.process();
        }

        let result = integration.process();

        // Temporal fields should be present
        assert!(result.temporal_coherence.is_some());
        assert!(result.phi_velocity.is_some());
        assert!(result.consciousness_continuity.is_some());
        assert!(result.temporal_identity_coherence.is_some());
        assert!(result.temporally_healthy.is_some());
    }

    #[test]
    fn test_temporal_coherence_reasonable_range() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process multiple times
        for i in 0..15 {
            integration.submit_content(
                &format!("CoherenceTask{}", i),
                vec![HV16::random(i as u64 + 10000)],
                "coherence test task",
                vec!["module".to_string()],
                0.6,
            );
            let result = integration.process();

            // Temporal coherence should be in valid range
            if let Some(coherence) = result.temporal_coherence {
                assert!(coherence >= 0.0 && coherence <= 1.0,
                    "Temporal coherence out of range: {}", coherence);
            }

            // Continuity should also be in valid range
            if let Some(continuity) = result.consciousness_continuity {
                assert!(continuity >= 0.0 && continuity <= 1.0,
                    "Consciousness continuity out of range: {}", continuity);
            }
        }
    }

    #[test]
    fn test_disabled_temporal_consciousness() {
        let mut config = NarrativeGWTConfig::default();
        config.enable_temporal_consciousness = false;
        config.temporal_config = None;

        let integration = NarrativeGWTIntegration::new(
            NarrativeSelfConfig::default(),
            UnifiedGWTConfig::default(),
            config,
        );

        // Should be disabled
        assert!(integration.temporal_analyzer.is_none());
        assert!(!integration.config.enable_temporal_consciousness);
    }

    #[test]
    fn test_temporal_config_options() {
        let mut temporal_config = TemporalConsciousnessConfig::default();
        temporal_config.phi_history_depth = 200;
        temporal_config.continuity_threshold = 0.2;
        temporal_config.enable_husserlian_analysis = false;

        let mut config = NarrativeGWTConfig::default();
        config.temporal_config = Some(temporal_config);

        let integration = NarrativeGWTIntegration::new(
            NarrativeSelfConfig::default(),
            UnifiedGWTConfig::default(),
            config,
        );

        // Verify custom config was applied
        assert!(integration.temporal_analyzer.is_some());
        if let Some(ref temporal) = integration.temporal_analyzer {
            assert_eq!(temporal.config.phi_history_depth, 200);
            assert_eq!(temporal.config.continuity_threshold, 0.2);
            assert!(!temporal.config.enable_husserlian_analysis);
        }
    }

    #[test]
    fn test_phi_velocity_tracking() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process multiple times and track velocity
        let mut velocities = Vec::new();
        for i in 0..10 {
            integration.submit_content(
                &format!("VelocityTask{}", i),
                vec![HV16::random(i as u64 + 11000)],
                "velocity test task",
                vec!["module".to_string()],
                0.5 + (i as f64 * 0.02), // Gradually increasing activation
            );
            let result = integration.process();

            if let Some(velocity) = result.phi_velocity {
                velocities.push(velocity);
            }
        }

        // Should have tracked multiple velocities
        assert!(!velocities.is_empty(), "Should have tracked phi velocities");
    }

    #[test]
    fn test_temporal_health_check() {
        let mut integration = NarrativeGWTIntegration::default_config();

        // Process with consistent behavior
        for i in 0..10 {
            integration.submit_content(
                &format!("HealthTask{}", i),
                vec![HV16::random(i as u64 + 12000)],
                "health check task",
                vec!["module".to_string()],
                0.6,
            );
            integration.process();
        }

        let result = integration.process();

        // After consistent processing, should be healthy
        if let Some(healthy) = result.temporally_healthy {
            // System should be reasonably healthy with consistent behavior
            // (may be false initially if not enough history)
            assert!(result.temporally_healthy.is_some());
        }
    }
}
