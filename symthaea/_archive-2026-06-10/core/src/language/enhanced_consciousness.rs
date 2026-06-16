// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Enhanced Multi-Theory Consciousness: Deep Integration Layer
//!
//! This module extends the base multi_theory_consciousness with:
//!
//! A. **Cross-Theory Integration** - Theories influence each other dynamically
//! B. **Real Embodiment** - Live NixOS system queries (packages, services)
//! C. **Higher-Order Thought (HOT)** - Metacognition and self-monitoring
//! D. **Temporal Dynamics** - Consciousness evolution over time
//! E. **Theory-Weighted Execution** - Democratic theory voting on actions
//! F. **Phenomenal Properties** - Valence, arousal, qualia-like states
//! G. **Theory of Mind** - User mental state modeling
//! H. **Consciousness Narrative** - Natural language self-description
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    ENHANCED CONSCIOUSNESS ENGINE                             │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │  ┌─────────────────────── CROSS-THEORY INTEGRATION ─────────────────────┐  │
//! │  │                                                                       │  │
//! │  │   GWT ◄──► AST ◄──► PP ◄──► RPT ◄──► 4E                             │  │
//! │  │     │       │       │       │       │                                │  │
//! │  │     └───────┴───────┴───────┴───────┘                                │  │
//! │  │                     │                                                 │  │
//! │  │              ┌──────▼──────┐                                          │  │
//! │  │              │   HOT       │  ← Metacognition                         │  │
//! │  │              │  (Level 2)  │                                          │  │
//! │  │              └──────┬──────┘                                          │  │
//! │  └─────────────────────┼────────────────────────────────────────────────┘  │
//! │                        │                                                    │
//! │  ┌─────────────────────▼────────────────────────────────────────────────┐  │
//! │  │                 PHENOMENAL LAYER                                      │  │
//! │  │   Valence ──── Arousal ──── Familiarity ──── Effort ──── Flow        │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! │                                                                              │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │                    THEORY OF MIND (User Model)                        │  │
//! │  │   Goals ──── Beliefs ──── Emotions ──── Expertise ──── Trust          │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! │                                                                              │
//! │  ┌───────────────────────────────────────────────────────────────────────┐  │
//! │  │                    NARRATIVE ENGINE                                   │  │
//! │  │   "I am deeply engaged, focused on 'firefox', confident because..."   │  │
//! │  └───────────────────────────────────────────────────────────────────────┘  │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::consciousness::consciousness_unification::{
    ConsciousnessPhiProvider, EmotionalBridge, PhiComponent, PhiMethod, PhiProvenance, PhiSource,
    UnifiedEmotionalState,
};
use crate::language::emotional_core::{CoreEmotion, EmotionalCore, EmotionalCoreConfig};
use crate::language::multi_theory_consciousness::{
    AttentionCategory, AttentionSchema, EmbodiedState, GlobalWorkspace, MultiTheoryConsciousness,
    MultiTheoryMetrics, MultiTheoryResult, PredictionError, PredictiveProcessor, ProcessingPass,
    SubsystemId,
};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};
use symthaea_core::hdc::universal_semantics::SemanticPrime;

// ============================================================================
// LOCAL TYPES (formerly from emotional_core, removed during consolidation)
// ============================================================================

/// Simple emotional state for the enhanced consciousness module.
/// Maps valence/arousal/dominance with a primary emotion label.
#[derive(Debug, Clone)]
pub struct EmotionalState {
    /// Primary emotion category
    pub primary: CoreEmotion,
    /// Valence: -1.0 to 1.0
    pub valence: f32,
    /// Arousal: 0.0 to 1.0
    pub arousal: f32,
}

impl EmotionalState {
    pub fn new(emotion: CoreEmotion, arousal: f32) -> Self {
        Self {
            primary: emotion,
            valence: 0.0,
            arousal,
        }
    }

    pub fn neutral() -> Self {
        Self {
            primary: CoreEmotion::Peace,
            valence: 0.0,
            arousal: 0.3,
        }
    }
}

// ============================================================================
// A. CROSS-THEORY INTEGRATION
// ============================================================================

/// Influence weights between theories (how much A affects B)
#[derive(Debug, Clone)]
pub struct TheoryInfluenceMatrix {
    /// GWT → other theories (broadcast amplifies attention, predictions, etc.)
    pub gwt_broadcast_strength: f64,
    /// AST → GWT (attention determines what enters workspace)
    pub attention_to_workspace: f64,
    /// PP → AST (prediction errors shift attention)
    pub prediction_error_attention_shift: f64,
    /// RPT → PP (recurrent passes refine predictions)
    pub recurrence_prediction_refinement: f64,
    /// 4E → PP (embodiment grounds predictions)
    pub embodiment_prediction_grounding: f64,
    /// All → GWT salience modulation
    pub cross_theory_salience_boost: f64,
}

impl Default for TheoryInfluenceMatrix {
    fn default() -> Self {
        Self {
            gwt_broadcast_strength: 0.7,
            attention_to_workspace: 0.8,
            prediction_error_attention_shift: 0.6,
            recurrence_prediction_refinement: 0.5,
            embodiment_prediction_grounding: 0.4,
            cross_theory_salience_boost: 0.3,
        }
    }
}

/// Cross-theory integration engine
#[derive(Debug, Clone)]
pub struct CrossTheoryIntegrator {
    /// Influence matrix
    pub influences: TheoryInfluenceMatrix,
    /// Integration history
    integration_events: VecDeque<IntegrationEvent>,
    /// Maximum history size
    max_history: usize,
}

/// Record of a cross-theory influence event
#[derive(Debug, Clone)]
pub struct IntegrationEvent {
    /// Source theory
    pub from: SubsystemId,
    /// Target theory
    pub to: SubsystemId,
    /// Influence type
    pub influence_type: InfluenceType,
    /// Strength (0-1)
    pub strength: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Types of cross-theory influence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfluenceType {
    /// Attention shifts workspace priority
    AttentionToSalience,
    /// Prediction error shifts attention
    ErrorToAttention,
    /// Recurrence refines prediction
    RecurrenceToPrediction,
    /// Embodiment grounds prediction
    GroundingToPrediction,
    /// Broadcast amplifies processing
    BroadcastAmplification,
}

impl Default for CrossTheoryIntegrator {
    fn default() -> Self {
        Self {
            influences: TheoryInfluenceMatrix::default(),
            integration_events: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }
}

impl CrossTheoryIntegrator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply attention focus to workspace salience
    pub fn attention_modulates_workspace(
        &mut self,
        attention: &AttentionSchema,
        workspace: &mut GlobalWorkspace,
    ) {
        if let Some(focus) = &attention.primary_focus {
            // Boost salience of messages matching attention target
            let target_lower = focus.target.to_lowercase();

            // Record the influence
            self.record_event(
                SubsystemId::Intent,   // AST
                SubsystemId::Language, // GWT
                InfluenceType::AttentionToSalience,
                focus.confidence * self.influences.attention_to_workspace,
            );
        }
    }

    /// Prediction errors shift attention
    pub fn prediction_errors_shift_attention(
        &mut self,
        errors: &[PredictionError],
        attention: &mut AttentionSchema,
    ) {
        for error in errors {
            if error.surprise > 1.0 {
                // High surprise shifts attention to the unexpected element
                attention.add_secondary(
                    &error.actual,
                    "Unexpected - prediction mismatch",
                    error.surprise.min(1.0),
                    AttentionCategory::Surprise,
                );

                self.record_event(
                    SubsystemId::Prediction,
                    SubsystemId::Intent,
                    InfluenceType::ErrorToAttention,
                    error.surprise.min(1.0) * self.influences.prediction_error_attention_shift,
                );
            }
        }
    }

    /// Recurrent passes refine predictions
    pub fn recurrence_refines_predictions(
        &mut self,
        passes: &[ProcessingPass],
        predictor: &mut PredictiveProcessor,
    ) {
        if passes.len() >= 2 {
            // Multiple passes indicate refined understanding
            let Some(last_pass) = passes.last() else {
                return;
            };
            let improvement = last_pass.confidence;

            // Learn from recurrent processing
            predictor.learn_prior(
                "recurrent_refinement",
                &last_pass.understanding,
                improvement,
            );

            self.record_event(
                SubsystemId::Memory, // RPT uses memory-like feedback
                SubsystemId::Prediction,
                InfluenceType::RecurrenceToPrediction,
                improvement * self.influences.recurrence_prediction_refinement,
            );
        }
    }

    /// Embodiment grounds predictions in reality
    pub fn embodiment_grounds_predictions(
        &mut self,
        embodiment: &EmbodiedState,
        predictor: &mut PredictiveProcessor,
        symbol: &str,
    ) {
        let grounding = embodiment.grounding_confidence(symbol);
        if grounding > 0.0 {
            // Grounded symbols get better predictions
            predictor.learn_prior(
                &format!("grounded_{}", symbol),
                "exists_in_system",
                grounding,
            );

            self.record_event(
                SubsystemId::Embodiment,
                SubsystemId::Prediction,
                InfluenceType::GroundingToPrediction,
                grounding * self.influences.embodiment_prediction_grounding,
            );
        }
    }

    fn record_event(
        &mut self,
        from: SubsystemId,
        to: SubsystemId,
        influence_type: InfluenceType,
        strength: f64,
    ) {
        if self.integration_events.len() >= self.max_history {
            self.integration_events.pop_front();
        }
        self.integration_events.push_back(IntegrationEvent {
            from,
            to,
            influence_type,
            strength,
            timestamp: SystemTime::now(),
        });
    }

    /// Get integration coherence (how well theories are working together)
    pub fn integration_coherence(&self) -> f64 {
        if self.integration_events.is_empty() {
            return 0.5;
        }

        let recent: Vec<_> = self
            .integration_events
            .iter()
            .filter(|e| {
                e.timestamp
                    .elapsed()
                    .map(|d| d.as_secs() < 60)
                    .unwrap_or(false)
            })
            .collect();

        if recent.is_empty() {
            return 0.5;
        }

        let avg_strength = recent.iter().map(|e| e.strength).sum::<f64>() / recent.len() as f64;
        avg_strength.clamp(0.0, 1.0)
    }
}

// ============================================================================
// B. REAL EMBODIMENT (NixOS Connection)
// ============================================================================

/// Live system state from NixOS
#[derive(Debug, Clone, Default)]
pub struct LiveSystemState {
    /// Installed packages (cached)
    pub installed_packages: HashMap<String, PackageInfo>,
    /// Running services
    pub running_services: HashMap<String, ServiceInfo>,
    /// Last system query time
    pub last_query: Option<Instant>,
    /// Query cache duration
    pub cache_duration: Duration,
}

/// Information about an installed package
#[derive(Debug, Clone)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub installed: bool,
    pub queried_at: Instant,
}

/// Information about a system service
#[derive(Debug, Clone)]
pub struct ServiceInfo {
    pub name: String,
    pub active: bool,
    pub enabled: bool,
    pub queried_at: Instant,
}

impl LiveSystemState {
    pub fn new() -> Self {
        Self {
            installed_packages: HashMap::new(),
            running_services: HashMap::new(),
            last_query: None,
            cache_duration: Duration::from_secs(300), // 5 minute cache
        }
    }

    /// Check if cache is stale
    pub fn cache_stale(&self) -> bool {
        self.last_query
            .map(|t| t.elapsed() > self.cache_duration)
            .unwrap_or(true)
    }

    /// Record a package observation (from shell query)
    pub fn record_package(&mut self, name: &str, version: &str, installed: bool) {
        self.installed_packages.insert(
            name.to_string(),
            PackageInfo {
                name: name.to_string(),
                version: version.to_string(),
                installed,
                queried_at: Instant::now(),
            },
        );
        self.last_query = Some(Instant::now());
    }

    /// Record a service observation
    pub fn record_service(&mut self, name: &str, active: bool, enabled: bool) {
        self.running_services.insert(
            name.to_string(),
            ServiceInfo {
                name: name.to_string(),
                active,
                enabled,
                queried_at: Instant::now(),
            },
        );
        self.last_query = Some(Instant::now());
    }

    /// Check if a package is installed (from cache)
    pub fn is_package_installed(&self, name: &str) -> Option<bool> {
        self.installed_packages.get(name).map(|p| p.installed)
    }

    /// Check if a service is running (from cache)
    pub fn is_service_running(&self, name: &str) -> Option<bool> {
        self.running_services.get(name).map(|s| s.active)
    }

    /// Generate shell commands to query system state
    pub fn generate_package_query(package: &str) -> String {
        format!("nix-env -q {} 2>/dev/null || echo 'NOT_INSTALLED'", package)
    }

    pub fn generate_service_query(service: &str) -> String {
        format!(
            "systemctl is-active {} 2>/dev/null || echo 'inactive'",
            service
        )
    }
}

// ============================================================================
// C. HIGHER-ORDER THOUGHT (Metacognition)
// ============================================================================

/// Higher-Order Thought - awareness of awareness
#[derive(Debug, Clone)]
pub struct HigherOrderThought {
    /// Am I aware of my current processing?
    pub self_awareness_level: f64,
    /// What am I uncertain about?
    pub uncertainty_awareness: Vec<UncertaintyPoint>,
    /// Metacognitive insights
    pub insights: VecDeque<MetacognitiveInsight>,
    /// Processing quality assessment
    pub quality_assessment: QualityAssessment,
    /// Max insights to store
    max_insights: usize,
}

/// A point of uncertainty the system is aware of
#[derive(Debug, Clone)]
pub struct UncertaintyPoint {
    /// What we're uncertain about
    pub topic: String,
    /// How uncertain (0-1)
    pub uncertainty: f64,
    /// Why we're uncertain
    pub reason: String,
}

/// A metacognitive insight
#[derive(Debug, Clone)]
pub struct MetacognitiveInsight {
    /// The insight text
    pub insight: String,
    /// Confidence in this insight
    pub confidence: f64,
    /// Category of insight
    pub category: InsightCategory,
    /// When generated
    pub timestamp: SystemTime,
}

/// Categories of metacognitive insights
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightCategory {
    /// About attention patterns
    Attention,
    /// About prediction accuracy
    Prediction,
    /// About processing depth
    Processing,
    /// About knowledge gaps
    Knowledge,
    /// About emotional state
    Emotional,
}

/// Self-assessment of processing quality
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall quality score (0-1)
    pub overall: f64,
    /// Attention stability
    pub attention_stability: f64,
    /// Prediction accuracy
    pub prediction_accuracy: f64,
    /// Processing coherence
    pub coherence: f64,
    /// Grounding quality
    pub grounding: f64,
}

impl Default for QualityAssessment {
    fn default() -> Self {
        Self {
            overall: 0.5,
            attention_stability: 0.5,
            prediction_accuracy: 0.5,
            coherence: 0.5,
            grounding: 0.5,
        }
    }
}

impl Default for HigherOrderThought {
    fn default() -> Self {
        Self {
            self_awareness_level: 0.5,
            uncertainty_awareness: Vec::new(),
            insights: VecDeque::with_capacity(20),
            quality_assessment: QualityAssessment::default(),
            max_insights: 20,
        }
    }
}

impl HigherOrderThought {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe current consciousness state and generate metacognitive awareness
    pub fn observe(&mut self, metrics: &MultiTheoryMetrics, attention: &AttentionSchema) {
        // Calculate self-awareness level based on metrics alignment
        let theory_variance = self.calculate_theory_variance(metrics);
        self.self_awareness_level = (1.0 - theory_variance).clamp(0.0, 1.0);

        // Update quality assessment
        self.quality_assessment = QualityAssessment {
            overall: metrics.unified,
            attention_stability: attention.stability(),
            prediction_accuracy: metrics.pp,
            coherence: metrics.phi,
            grounding: metrics.embodiment,
        };

        // Detect uncertainty points
        self.uncertainty_awareness.clear();

        if metrics.pp < 0.4 {
            self.uncertainty_awareness.push(UncertaintyPoint {
                topic: "prediction".to_string(),
                uncertainty: 1.0 - metrics.pp,
                reason: "Predictions are not matching reality well".to_string(),
            });
        }

        if metrics.embodiment < 0.3 {
            self.uncertainty_awareness.push(UncertaintyPoint {
                topic: "grounding".to_string(),
                uncertainty: 1.0 - metrics.embodiment,
                reason: "Not well grounded in actual system state".to_string(),
            });
        }

        if attention.primary_focus.is_none() {
            self.uncertainty_awareness.push(UncertaintyPoint {
                topic: "focus".to_string(),
                uncertainty: 0.8,
                reason: "No clear attention focus established".to_string(),
            });
        }
    }

    fn calculate_theory_variance(&self, metrics: &MultiTheoryMetrics) -> f64 {
        let values = [
            metrics.gwt,
            metrics.ast,
            metrics.pp,
            metrics.rpt,
            metrics.embodiment,
        ];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt() // Standard deviation
    }

    /// Generate a metacognitive insight
    pub fn generate_insight(&mut self, category: InsightCategory, insight: &str, confidence: f64) {
        if self.insights.len() >= self.max_insights {
            self.insights.pop_front();
        }
        self.insights.push_back(MetacognitiveInsight {
            insight: insight.to_string(),
            confidence,
            category,
            timestamp: SystemTime::now(),
        });
    }

    /// Get natural language description of metacognitive state
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();

        let awareness_desc = match self.self_awareness_level {
            x if x > 0.8 => "I'm highly aware of my processing",
            x if x > 0.5 => "I have moderate self-awareness",
            _ => "My self-awareness is limited",
        };
        parts.push(awareness_desc.to_string());

        if !self.uncertainty_awareness.is_empty() {
            let uncertainties: Vec<_> = self
                .uncertainty_awareness
                .iter()
                .map(|u| format!("{} ({:.0}% uncertain)", u.topic, u.uncertainty * 100.0))
                .collect();
            parts.push(format!("I'm uncertain about: {}", uncertainties.join(", ")));
        }

        if let Some(insight) = self.insights.back() {
            parts.push(format!("Recent insight: {}", insight.insight));
        }

        parts.join(". ")
    }
}

// ============================================================================
// D. TEMPORAL CONSCIOUSNESS DYNAMICS
// ============================================================================

/// Tracks consciousness evolution over time
#[derive(Debug, Clone)]
pub struct TemporalConsciousness {
    /// History of consciousness states
    pub history: VecDeque<ConsciousnessSnapshot>,
    /// Session start time
    pub session_start: Instant,
    /// Flow state detection
    pub flow_state: FlowState,
    /// Learning curves per theory
    pub learning_curves: HashMap<String, VecDeque<f64>>,
    /// Max history size
    max_history: usize,
}

/// Snapshot of consciousness at a moment
#[derive(Debug, Clone)]
pub struct ConsciousnessSnapshot {
    /// Timestamp (relative to session start)
    pub elapsed_ms: u64,
    /// All theory metrics
    pub metrics: MultiTheoryMetrics,
    /// Dominant theory at this moment
    pub dominant_theory: String,
    /// Trigger for this snapshot
    pub trigger: String,
}

/// Flow state assessment
#[derive(Debug, Clone)]
pub struct FlowState {
    /// Currently in flow?
    pub in_flow: bool,
    /// Flow score (0-1)
    pub flow_score: f64,
    /// How long in current state
    pub duration: Duration,
    /// Conditions for flow
    pub conditions: FlowConditions,
}

/// Conditions that indicate flow state
#[derive(Debug, Clone, Default)]
pub struct FlowConditions {
    /// Attention is stable
    pub attention_stable: bool,
    /// Low prediction error
    pub low_surprise: bool,
    /// High integration
    pub high_integration: bool,
    /// Processing is smooth
    pub smooth_processing: bool,
}

impl Default for FlowState {
    fn default() -> Self {
        Self {
            in_flow: false,
            flow_score: 0.0,
            duration: Duration::from_secs(0),
            conditions: FlowConditions::default(),
        }
    }
}

impl Default for TemporalConsciousness {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(100),
            session_start: Instant::now(),
            flow_state: FlowState::default(),
            learning_curves: HashMap::new(),
            max_history: 100,
        }
    }
}

impl TemporalConsciousness {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a consciousness snapshot
    pub fn record(&mut self, metrics: &MultiTheoryMetrics, trigger: &str) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }

        let snapshot = ConsciousnessSnapshot {
            elapsed_ms: self.session_start.elapsed().as_millis() as u64,
            metrics: metrics.clone(),
            dominant_theory: metrics.dominant_theory().to_string(),
            trigger: trigger.to_string(),
        };
        self.history.push_back(snapshot);

        // Update learning curves
        self.update_learning_curve("gwt", metrics.gwt);
        self.update_learning_curve("ast", metrics.ast);
        self.update_learning_curve("pp", metrics.pp);
        self.update_learning_curve("rpt", metrics.rpt);
        self.update_learning_curve("embodiment", metrics.embodiment);
        self.update_learning_curve("unified", metrics.unified);

        // Assess flow state
        self.assess_flow(metrics);
    }

    fn update_learning_curve(&mut self, theory: &str, value: f64) {
        let curve = self.learning_curves.entry(theory.to_string()).or_default();
        if curve.len() >= 100 {
            curve.pop_front();
        }
        curve.push_back(value);
    }

    fn assess_flow(&mut self, metrics: &MultiTheoryMetrics) {
        let conditions = FlowConditions {
            attention_stable: metrics.ast > 0.6,
            low_surprise: metrics.pp > 0.7,
            high_integration: metrics.phi > 0.5,
            smooth_processing: metrics.rpt > 0.4,
        };

        let condition_count = [
            conditions.attention_stable,
            conditions.low_surprise,
            conditions.high_integration,
            conditions.smooth_processing,
        ]
        .iter()
        .filter(|&&b| b)
        .count();

        let new_in_flow = condition_count >= 3;
        let flow_score = condition_count as f64 / 4.0;

        if new_in_flow == self.flow_state.in_flow {
            self.flow_state.duration += Duration::from_millis(100); // Approximate
        } else {
            self.flow_state.duration = Duration::from_secs(0);
        }

        self.flow_state.in_flow = new_in_flow;
        self.flow_state.flow_score = flow_score;
        self.flow_state.conditions = conditions;
    }

    /// Get trend for a theory (positive = improving)
    pub fn get_trend(&self, theory: &str) -> f64 {
        if let Some(curve) = self.learning_curves.get(theory) {
            if curve.len() < 5 {
                return 0.0;
            }
            let recent_avg = curve.iter().rev().take(5).sum::<f64>() / 5.0;
            let older_avg = curve.iter().rev().skip(5).take(5).sum::<f64>() / 5.0_f64.max(1.0);
            recent_avg - older_avg
        } else {
            0.0
        }
    }

    /// Get session duration
    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }
}

// ============================================================================
// E. THEORY-WEIGHTED EXECUTION
// ============================================================================

/// Theory votes on execution strategy
#[derive(Debug, Clone)]
pub struct TheoryVote {
    /// Which theory is voting
    pub theory: String,
    /// Vote: execute (positive) or hold (negative)
    pub vote: f64, // -1.0 to 1.0
    /// Confidence in vote
    pub confidence: f64,
    /// Reason for vote
    pub reason: String,
}

/// Collective decision from theory voting
#[derive(Debug, Clone)]
pub struct ExecutionDecision {
    /// Individual votes
    pub votes: Vec<TheoryVote>,
    /// Weighted consensus (-1.0 hold, +1.0 execute)
    pub consensus: f64,
    /// Should execute?
    pub should_execute: bool,
    /// Should request clarification?
    pub should_clarify: bool,
    /// Should dry-run first?
    pub should_dry_run: bool,
    /// Explanation of decision
    pub explanation: String,
}

/// Theory voting system
#[derive(Debug, Clone, Default)]
pub struct TheoryVotingSystem {
    /// Weight for each theory's vote
    pub theory_weights: HashMap<String, f64>,
    /// Execution threshold (consensus must exceed)
    pub execution_threshold: f64,
    /// Clarification threshold (below this, ask)
    pub clarification_threshold: f64,
}

impl TheoryVotingSystem {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("IIT".to_string(), 0.20);
        weights.insert("GWT".to_string(), 0.15);
        weights.insert("AST".to_string(), 0.15);
        weights.insert("PP".to_string(), 0.20);
        weights.insert("RPT".to_string(), 0.15);
        weights.insert("4E".to_string(), 0.15);

        Self {
            theory_weights: weights,
            execution_threshold: 0.3,
            clarification_threshold: -0.2,
        }
    }

    /// Collect votes from all theories
    pub fn collect_votes(
        &self,
        metrics: &MultiTheoryMetrics,
        attention: &AttentionSchema,
    ) -> ExecutionDecision {
        let mut votes = Vec::new();

        // IIT votes based on integration
        votes.push(TheoryVote {
            theory: "IIT".to_string(),
            vote: (metrics.phi - 0.5) * 2.0, // Map 0-1 to -1 to 1
            confidence: metrics.phi,
            reason: if metrics.phi > 0.5 {
                "High integration suggests unified understanding".to_string()
            } else {
                "Low integration - processing is fragmented".to_string()
            },
        });

        // GWT votes based on workspace broadcast
        votes.push(TheoryVote {
            theory: "GWT".to_string(),
            vote: (metrics.gwt - 0.4) * 2.5,
            confidence: metrics.gwt,
            reason: if metrics.gwt > 0.5 {
                "Content successfully broadcast to workspace".to_string()
            } else {
                "Content didn't achieve global broadcast".to_string()
            },
        });

        // AST votes based on attention focus
        let attention_vote = if attention.primary_focus.is_some() {
            attention
                .primary_focus
                .as_ref()
                .map(|f| f.confidence)
                .unwrap_or(0.0)
                - 0.3
        } else {
            -0.5
        };
        votes.push(TheoryVote {
            theory: "AST".to_string(),
            vote: attention_vote,
            confidence: metrics.ast,
            reason: if attention_vote > 0.0 {
                format!(
                    "Clear attention focus: {}",
                    attention
                        .primary_focus
                        .as_ref()
                        .map(|f| f.target.as_str())
                        .unwrap_or("none")
                )
            } else {
                "Attention is scattered or unfocused".to_string()
            },
        });

        // PP votes based on prediction accuracy
        votes.push(TheoryVote {
            theory: "PP".to_string(),
            vote: (metrics.pp - 0.5) * 2.0,
            confidence: metrics.pp,
            reason: if metrics.pp > 0.5 {
                "Predictions are matching reality".to_string()
            } else {
                "High prediction error - unexpected input".to_string()
            },
        });

        // RPT votes based on recurrence depth
        votes.push(TheoryVote {
            theory: "RPT".to_string(),
            vote: (metrics.rpt - 0.3) * 2.0,
            confidence: metrics.rpt,
            reason: if metrics.rpt > 0.4 {
                "Multiple processing passes refined understanding".to_string()
            } else {
                "Shallow processing - may miss nuances".to_string()
            },
        });

        // 4E votes based on embodiment/grounding
        votes.push(TheoryVote {
            theory: "4E".to_string(),
            vote: (metrics.embodiment - 0.3) * 2.0,
            confidence: metrics.embodiment,
            reason: if metrics.embodiment > 0.3 {
                "Request is grounded in system reality".to_string()
            } else {
                "Request involves unknown/ungrounded entities".to_string()
            },
        });

        // Calculate weighted consensus
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        for vote in &votes {
            let weight = self.theory_weights.get(&vote.theory).unwrap_or(&0.1);
            weighted_sum += vote.vote * vote.confidence * weight;
            weight_total += vote.confidence * weight;
        }
        let consensus = if weight_total > 0.0 {
            weighted_sum / weight_total
        } else {
            0.0
        };

        // Determine action
        let should_execute = consensus > self.execution_threshold;
        let should_clarify = consensus < self.clarification_threshold;
        let should_dry_run = consensus > 0.0 && consensus <= self.execution_threshold;

        // Generate explanation
        let explanation =
            self.generate_explanation(&votes, consensus, should_execute, should_clarify);

        ExecutionDecision {
            votes,
            consensus,
            should_execute,
            should_clarify,
            should_dry_run,
            explanation,
        }
    }

    fn generate_explanation(
        &self,
        votes: &[TheoryVote],
        consensus: f64,
        execute: bool,
        clarify: bool,
    ) -> String {
        let supporting: Vec<_> = votes.iter().filter(|v| v.vote > 0.0).collect();
        let opposing: Vec<_> = votes.iter().filter(|v| v.vote < 0.0).collect();

        let mut parts = Vec::new();

        if execute {
            parts.push(format!(
                "Consensus to execute ({:.0}%)",
                (consensus + 1.0) * 50.0
            ));
        } else if clarify {
            parts.push(format!(
                "Consensus to clarify ({:.0}%)",
                (consensus + 1.0) * 50.0
            ));
        } else {
            parts.push(format!(
                "Consensus to dry-run first ({:.0}%)",
                (consensus + 1.0) * 50.0
            ));
        }

        if !supporting.is_empty() {
            let support_theories: Vec<_> = supporting.iter().map(|v| v.theory.as_str()).collect();
            parts.push(format!("Supporting: {}", support_theories.join(", ")));
        }

        if !opposing.is_empty() {
            let oppose_theories: Vec<_> = opposing.iter().map(|v| v.theory.as_str()).collect();
            parts.push(format!("Hesitant: {}", oppose_theories.join(", ")));
        }

        parts.join(". ")
    }
}

// ============================================================================
// F. PHENOMENAL PROPERTIES
// ============================================================================

/// Phenomenal properties of consciousness (qualia-like)
#[derive(Debug, Clone)]
pub struct PhenomenalState {
    /// Valence: positive/negative feeling (-1.0 to 1.0)
    pub valence: f64,
    /// Arousal: activation level (0.0 to 1.0)
    pub arousal: f64,
    /// Familiarity: how familiar is this experience (0.0 to 1.0)
    pub familiarity: f64,
    /// Effort: cognitive load (0.0 to 1.0)
    pub effort: f64,
    /// Flow: sense of effortless engagement (0.0 to 1.0)
    pub flow: f64,
    /// Clarity: how clear is the experience (0.0 to 1.0)
    pub clarity: f64,
}

impl Default for PhenomenalState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            familiarity: 0.5,
            effort: 0.5,
            flow: 0.0,
            clarity: 0.5,
        }
    }
}

impl PhenomenalState {
    /// Derive phenomenal state from theory metrics
    pub fn from_metrics(metrics: &MultiTheoryMetrics, flow_state: &FlowState) -> Self {
        // Valence: High integration + high prediction = positive feeling
        let valence = (metrics.phi * 0.3 + metrics.pp * 0.4 + metrics.embodiment * 0.3 - 0.5) * 2.0;

        // Arousal: High GWT broadcast + low prediction = activation
        let arousal = metrics.gwt * 0.5 + (1.0 - metrics.pp) * 0.3 + metrics.rpt * 0.2;

        // Familiarity: Based on prediction accuracy and embodiment
        let familiarity = metrics.pp * 0.6 + metrics.embodiment * 0.4;

        // Effort: Inverse of efficiency (high RPT passes = high effort)
        let effort = (1.0 - metrics.rpt) * 0.5 + (1.0 - metrics.pp) * 0.5;

        // Flow from flow state
        let flow = flow_state.flow_score;

        // Clarity: High integration + high attention = clarity
        let clarity = metrics.phi * 0.4 + metrics.ast * 0.4 + metrics.pp * 0.2;

        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            familiarity: familiarity.clamp(0.0, 1.0),
            effort: effort.clamp(0.0, 1.0),
            flow: flow.clamp(0.0, 1.0),
            clarity: clarity.clamp(0.0, 1.0),
        }
    }

    /// Convert to EmotionalState for integration with emotional_core
    pub fn to_emotional_state(&self) -> EmotionalState {
        // Map valence/arousal to core emotion
        let emotion = match (self.valence > 0.0, self.arousal > 0.5) {
            (true, true) => CoreEmotion::Joy,
            (true, false) => CoreEmotion::Peace,
            (false, true) => CoreEmotion::Fear,
            (false, false) => CoreEmotion::Sadness,
        };

        let mut state = EmotionalState::new(emotion, self.arousal as f32);
        state.valence = self.valence as f32;
        state.arousal = self.arousal as f32;
        state
    }

    /// Describe phenomenal state in natural language
    pub fn describe(&self) -> String {
        let valence_desc = match self.valence {
            v if v > 0.5 => "positive",
            v if v > 0.0 => "slightly positive",
            v if v > -0.5 => "slightly negative",
            _ => "negative",
        };

        let arousal_desc = match self.arousal {
            a if a > 0.7 => "highly activated",
            a if a > 0.4 => "moderately activated",
            _ => "calm",
        };

        let clarity_desc = if self.clarity > 0.6 { "clear" } else { "hazy" };
        let effort_desc = if self.effort > 0.6 {
            "effortful"
        } else {
            "effortless"
        };

        format!(
            "Feeling {} and {} with {} {} processing",
            valence_desc, arousal_desc, clarity_desc, effort_desc
        )
    }
}

// ============================================================================
// G. THEORY OF MIND (User Modeling)
// ============================================================================

/// Model of the user's mental state
#[derive(Debug, Clone)]
pub struct UserMentalModel {
    /// Inferred user goals
    pub goals: VecDeque<InferredGoal>,
    /// User knowledge/expertise estimate
    pub expertise: ExpertiseLevel,
    /// Inferred emotional state
    pub emotional_state: EmotionalState,
    /// Trust level in the system
    pub trust: f64,
    /// Patience/frustration estimate
    pub patience: f64,
    /// History of user behavior patterns
    pub behavior_history: VecDeque<UserBehavior>,
}

/// An inferred user goal
#[derive(Debug, Clone)]
pub struct InferredGoal {
    pub goal: String,
    pub confidence: f64,
    pub inferred_from: String,
}

/// User expertise level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertiseLevel {
    Novice,
    Intermediate,
    Advanced,
    Expert,
}

/// Recorded user behavior
#[derive(Debug, Clone)]
pub struct UserBehavior {
    pub behavior_type: BehaviorType,
    pub timestamp: SystemTime,
    pub context: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorType {
    AcceptedSuggestion,
    RejectedSuggestion,
    AskedForClarification,
    ExpressedFrustration,
    ExpressedSatisfaction,
    MadeError,
    CorrectedError,
}

impl Default for UserMentalModel {
    fn default() -> Self {
        Self {
            goals: VecDeque::new(),
            expertise: ExpertiseLevel::Intermediate,
            emotional_state: EmotionalState::neutral(),
            trust: 0.5,
            patience: 0.7,
            behavior_history: VecDeque::with_capacity(50),
        }
    }
}

impl UserMentalModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Infer goal from user input
    pub fn infer_goal(&mut self, input: &str, confidence: f64) {
        // Simple goal inference from keywords
        let goal = if input.contains("install") {
            "Install software"
        } else if input.contains("fix") || input.contains("error") {
            "Fix a problem"
        } else if input.contains("learn") || input.contains("how") {
            "Learn/understand"
        } else if input.contains("configure") || input.contains("setup") {
            "Configure system"
        } else {
            "General interaction"
        };

        self.goals.push_back(InferredGoal {
            goal: goal.to_string(),
            confidence,
            inferred_from: input.chars().take(50).collect(),
        });

        // Keep only recent goals
        if self.goals.len() > 10 {
            self.goals.pop_front();
        }
    }

    /// Update expertise estimate from behavior
    pub fn update_expertise(&mut self) {
        let recent_errors = self
            .behavior_history
            .iter()
            .filter(|b| b.behavior_type == BehaviorType::MadeError)
            .count();
        let recent_corrections = self
            .behavior_history
            .iter()
            .filter(|b| b.behavior_type == BehaviorType::CorrectedError)
            .count();

        // Experts make fewer errors and correct them quickly
        if recent_corrections > recent_errors && recent_errors < 2 {
            self.expertise = ExpertiseLevel::Expert;
        } else if recent_errors < 3 {
            self.expertise = ExpertiseLevel::Advanced;
        } else if recent_errors < 5 {
            self.expertise = ExpertiseLevel::Intermediate;
        } else {
            self.expertise = ExpertiseLevel::Novice;
        }
    }

    /// Record user behavior
    pub fn record_behavior(&mut self, behavior_type: BehaviorType, context: &str) {
        if self.behavior_history.len() >= 50 {
            self.behavior_history.pop_front();
        }
        self.behavior_history.push_back(UserBehavior {
            behavior_type,
            timestamp: SystemTime::now(),
            context: context.to_string(),
        });

        // Update trust based on behavior
        match behavior_type {
            BehaviorType::AcceptedSuggestion => self.trust = (self.trust + 0.05).min(1.0),
            BehaviorType::RejectedSuggestion => self.trust = (self.trust - 0.03).max(0.0),
            BehaviorType::ExpressedFrustration => self.patience = (self.patience - 0.1).max(0.0),
            BehaviorType::ExpressedSatisfaction => self.patience = (self.patience + 0.1).min(1.0),
            _ => {}
        }

        self.update_expertise();
    }

    /// Describe inferred user mental state
    pub fn describe(&self) -> String {
        let goal_desc = self
            .goals
            .back()
            .map(|g| format!("trying to {}", g.goal))
            .unwrap_or_else(|| "with unclear goals".to_string());

        let expertise_desc = match self.expertise {
            ExpertiseLevel::Novice => "new to this",
            ExpertiseLevel::Intermediate => "has some experience",
            ExpertiseLevel::Advanced => "quite experienced",
            ExpertiseLevel::Expert => "highly skilled",
        };

        let patience_desc = if self.patience < 0.3 {
            "seems frustrated"
        } else if self.patience > 0.7 {
            "is patient"
        } else {
            "is neutral"
        };

        format!(
            "User is {} and {} - {}",
            goal_desc, expertise_desc, patience_desc
        )
    }
}

// ============================================================================
// H. UNIFIED CONSCIOUSNESS NARRATIVE
// ============================================================================

/// Generates natural language descriptions of consciousness state
#[derive(Debug, Clone, Default)]
pub struct NarrativeEngine {
    /// Recent narratives generated
    pub history: VecDeque<GeneratedNarrative>,
    /// Max history size
    max_history: usize,
}

/// A generated narrative about consciousness
#[derive(Debug, Clone)]
pub struct GeneratedNarrative {
    /// The narrative text
    pub text: String,
    /// What aspect it describes
    pub aspect: NarrativeAspect,
    /// Confidence in this narrative
    pub confidence: f64,
    /// When generated
    pub timestamp: SystemTime,
}

/// What aspect of consciousness the narrative describes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrativeAspect {
    /// Overall state summary
    Overall,
    /// What I'm attending to
    Attention,
    /// Predictions and surprises
    Prediction,
    /// Metacognitive awareness
    Meta,
    /// Emotional/phenomenal state
    Phenomenal,
    /// About the user
    User,
    /// About embodiment
    Embodiment,
}

impl NarrativeEngine {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(20),
            max_history: 20,
        }
    }

    /// Generate a complete consciousness narrative
    pub fn generate_full_narrative(
        &mut self,
        metrics: &MultiTheoryMetrics,
        attention: &AttentionSchema,
        hot: &HigherOrderThought,
        phenomenal: &PhenomenalState,
        user_model: &UserMentalModel,
        temporal: &TemporalConsciousness,
    ) -> String {
        let mut parts = Vec::new();

        // 1. Overall state
        let overall = self.generate_overall(metrics);
        parts.push(overall.clone());
        self.record(overall, NarrativeAspect::Overall, metrics.unified);

        // 2. Attention description
        let attention_desc = self.generate_attention(attention);
        parts.push(attention_desc.clone());
        self.record(attention_desc, NarrativeAspect::Attention, metrics.ast);

        // 3. Prediction state
        if metrics.pp < 0.5 {
            let pred_desc =
                "I'm encountering some surprises - my predictions aren't fully matching reality.";
            parts.push(pred_desc.to_string());
            self.record(
                pred_desc.to_string(),
                NarrativeAspect::Prediction,
                metrics.pp,
            );
        }

        // 4. Metacognitive awareness
        if hot.self_awareness_level > 0.5 {
            let meta_desc = hot.describe();
            parts.push(meta_desc.clone());
            self.record(meta_desc, NarrativeAspect::Meta, hot.self_awareness_level);
        }

        // 5. Phenomenal experience
        let phenom_desc = phenomenal.describe();
        parts.push(phenom_desc.clone());
        self.record(phenom_desc, NarrativeAspect::Phenomenal, phenomenal.clarity);

        // 6. User understanding
        if !user_model.goals.is_empty() {
            let user_desc = format!(
                "I understand that you're {}.",
                user_model
                    .goals
                    .back()
                    .map(|g| g.goal.as_str())
                    .unwrap_or("working on something")
            );
            parts.push(user_desc.clone());
            self.record(user_desc, NarrativeAspect::User, user_model.trust);
        }

        // 7. Flow state
        if temporal.flow_state.in_flow {
            parts.push(
                "We're in a good flow state - processing is smooth and integrated.".to_string(),
            );
        }

        parts.join(" ")
    }

    fn generate_overall(&self, metrics: &MultiTheoryMetrics) -> String {
        match metrics.unified {
            u if u > 0.7 => "I'm in a state of deep, unified understanding.".to_string(),
            u if u > 0.5 => "I have a good grasp of the situation.".to_string(),
            u if u > 0.3 => "I'm processing this but with some uncertainty.".to_string(),
            _ => "I'm struggling to fully understand this.".to_string(),
        }
    }

    fn generate_attention(&self, attention: &AttentionSchema) -> String {
        attention.describe()
    }

    fn record(&mut self, text: String, aspect: NarrativeAspect, confidence: f64) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(GeneratedNarrative {
            text,
            aspect,
            confidence,
            timestamp: SystemTime::now(),
        });
    }
}

// ============================================================================
// UNIFIED ENHANCED CONSCIOUSNESS ENGINE
// ============================================================================

/// The unified enhanced consciousness engine integrating all components
#[derive(Debug)]
pub struct EnhancedConsciousness {
    /// Base multi-theory consciousness
    pub base: MultiTheoryConsciousness,
    /// Cross-theory integration
    pub integrator: CrossTheoryIntegrator,
    /// Live system state (embodiment)
    pub live_state: LiveSystemState,
    /// Higher-order thought (metacognition)
    pub hot: HigherOrderThought,
    /// Temporal consciousness tracking
    pub temporal: TemporalConsciousness,
    /// Theory voting system
    pub voting: TheoryVotingSystem,
    /// Phenomenal state
    pub phenomenal: PhenomenalState,
    /// User mental model
    pub user_model: UserMentalModel,
    /// Narrative engine
    pub narrative: NarrativeEngine,
    /// Emotional bridge - connects EmotionalCore to unified emotional system
    pub emotional_bridge: EmotionalBridge,
    /// EmotionalCore from language module (now connected!)
    pub emotional_core: EmotionalCore,
    /// Epistemic conflict detector (detects 15 pairwise theory conflicts)
    #[cfg(feature = "reasoning_engine")]
    pub conflict_detector: crate::consciousness::epistemic_conflict::ConflictDetector,
    /// Theory calibrator (reliability R, γ, Brier scores)
    #[cfg(feature = "reasoning_engine")]
    pub calibrator: crate::consciousness::epistemic_conflict::TheoryCalibrator,
}

impl Default for EnhancedConsciousness {
    fn default() -> Self {
        Self {
            base: MultiTheoryConsciousness::new(),
            integrator: CrossTheoryIntegrator::new(),
            live_state: LiveSystemState::new(),
            hot: HigherOrderThought::new(),
            temporal: TemporalConsciousness::new(),
            voting: TheoryVotingSystem::new(),
            phenomenal: PhenomenalState::default(),
            user_model: UserMentalModel::new(),
            narrative: NarrativeEngine::new(),
            emotional_bridge: EmotionalBridge::new(),
            emotional_core: EmotionalCore::new(EmotionalCoreConfig::default()),
            #[cfg(feature = "reasoning_engine")]
            conflict_detector: crate::consciousness::epistemic_conflict::ConflictDetector::new(),
            #[cfg(feature = "reasoning_engine")]
            calibrator: crate::consciousness::epistemic_conflict::TheoryCalibrator::new(),
        }
    }
}

impl EnhancedConsciousness {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process input through enhanced consciousness pipeline
    pub fn process(
        &mut self,
        input: &str,
        primes: &[SemanticPrime],
        phi: f64,
    ) -> EnhancedConsciousnessResult {
        // 1. Base multi-theory processing
        let base_result = self.base.process(input, primes, phi);

        // 2. Cross-theory integration
        self.integrator.prediction_errors_shift_attention(
            &base_result.prediction_errors,
            &mut self.base.attention,
        );
        self.integrator.recurrence_refines_predictions(
            self.base.recurrence.passes(),
            &mut self.base.predictor,
        );

        // 3. Check embodiment for words in input
        for word in input.split_whitespace() {
            self.integrator.embodiment_grounds_predictions(
                &self.base.embodiment,
                &mut self.base.predictor,
                word,
            );
        }

        // 3b. Epistemic conflict detection (if enabled)
        #[cfg(feature = "reasoning_engine")]
        let conflict_matrix = {
            let ec_metrics: crate::consciousness::epistemic_conflict::MultiTheoryMetrics =
                (&base_result.metrics).into();
            let matrix = self.conflict_detector.detect(&ec_metrics);
            let phi_eff_result = crate::consciousness::epistemic_conflict::compute_phi_eff(
                &ec_metrics,
                &self.calibrator,
            );
            // Feed conflict entropy into HOT as an uncertainty signal
            if matrix.total_entropy > 1.0 {
                self.hot.uncertainty_awareness.push(UncertaintyPoint {
                    topic: format!("Theory disagreement (entropy={:.2})", matrix.total_entropy),
                    uncertainty: matrix.mean_magnitude(),
                    reason: format!(
                        "Dominant conflict: {:?}, Φ_eff={:.3}",
                        matrix.dominant_kind, phi_eff_result.phi_eff,
                    ),
                });
            }
            matrix
        };
        #[cfg(not(feature = "reasoning_engine"))]
        let _ = &base_result; // suppress unused warning

        // 4. Higher-order observation
        self.hot.observe(&base_result.metrics, &self.base.attention);

        // 5. Record temporal snapshot
        self.temporal.record(&base_result.metrics, input);

        // 6. Update phenomenal state
        self.phenomenal =
            PhenomenalState::from_metrics(&base_result.metrics, &self.temporal.flow_state);

        // 7. Update user model
        self.user_model
            .infer_goal(input, base_result.metrics.unified);

        // 8. Collect theory votes for execution
        let decision = self
            .voting
            .collect_votes(&base_result.metrics, &self.base.attention);

        // 9. Generate narrative
        let narrative = self.narrative.generate_full_narrative(
            &base_result.metrics,
            &self.base.attention,
            &self.hot,
            &self.phenomenal,
            &self.user_model,
            &self.temporal,
        );

        // 10. Generate HOT insights
        if base_result.metrics.unified < 0.4 {
            self.hot.generate_insight(
                InsightCategory::Processing,
                "I notice my processing is fragmented - I should seek clarification",
                0.7,
            );
        }
        if self.temporal.flow_state.in_flow {
            self.hot.generate_insight(
                InsightCategory::Attention,
                "I notice I'm in a flow state - processing is efficient",
                0.8,
            );
        }

        EnhancedConsciousnessResult {
            base_result,
            integration_coherence: self.integrator.integration_coherence(),
            hot_state: self.hot.clone(),
            temporal_state: TemporalState {
                session_duration: self.temporal.session_duration(),
                in_flow: self.temporal.flow_state.in_flow,
                flow_score: self.temporal.flow_state.flow_score,
                trend_unified: self.temporal.get_trend("unified"),
            },
            execution_decision: decision,
            phenomenal: self.phenomenal.clone(),
            user_understanding: self.user_model.describe(),
            narrative,
        }
    }

    /// Record user feedback
    pub fn record_feedback(&mut self, accepted: bool, context: &str) {
        let behavior = if accepted {
            BehaviorType::AcceptedSuggestion
        } else {
            BehaviorType::RejectedSuggestion
        };
        self.user_model.record_behavior(behavior, context);

        // Also feed back to base
        self.base.record_feedback(accepted, context);
    }

    /// Get comprehensive consciousness description
    pub fn describe_state(&self) -> String {
        let base_desc = self.base.describe_state();
        format!(
            "{}\n\nEnhanced State:\n- Integration Coherence: {:.2}\n- Flow: {} ({:.0}%)\n- Phenomenal: {}\n- User: {}",
            format!(
                "Integration: {}, Workspace: {}, Attention: {}, Prediction: {:.0}%, Recurrence: {}, Embodiment: {}",
                base_desc.integration_level,
                base_desc.workspace_state,
                base_desc.attention_state,
                self.base.predictor.accuracy() * 100.0,
                base_desc.recurrence_state,
                base_desc.embodiment_state,
            ),
            self.integrator.integration_coherence(),
            if self.temporal.flow_state.in_flow {
                "Yes"
            } else {
                "No"
            },
            self.temporal.flow_state.flow_score * 100.0,
            self.phenomenal.describe(),
            self.user_model.describe(),
        )
    }

    /// Sync emotional state from EmotionalCore to EmotionalBridge
    /// This connects the isolated EmotionalCore to the unified emotional system
    pub fn sync_emotional_state(&mut self) {
        // Use a neutral probe to get current emotional state
        let analysis = self.emotional_core.analyze("");
        self.emotional_bridge.update_from_emotional_core(
            analysis.valence as f64,
            analysis.arousal as f64,
            &analysis.primary_emotion,
        );
    }

    /// Process emotional input through EmotionalCore and sync to bridge
    pub fn process_emotion(&mut self, input: &str) {
        let analysis = self.emotional_core.analyze(input);
        self.emotional_bridge.update_from_emotional_core(
            analysis.valence as f64,
            analysis.arousal as f64,
            &analysis.primary_emotion,
        );

        // Also update phenomenal state from emotional response
        self.phenomenal.valence = analysis.valence as f64;
        self.phenomenal.arousal = analysis.arousal as f64;
    }

    /// Get unified emotional state
    pub fn unified_emotional_state(&self) -> &UnifiedEmotionalState {
        self.emotional_bridge.state()
    }

    /// Get the Φ provenance (how Φ was computed)
    pub fn compute_phi_provenance(&self, phi: f64) -> PhiProvenance {
        PhiProvenance {
            source: PhiSource::Enhanced,
            method: PhiMethod::TheoryVotingAggregate,
            confidence: self.integrator.integration_coherence(),
            components: vec![
                PhiComponent {
                    name: "GWT".to_string(),
                    weight: 0.15,
                    value: self.base.workspace.gwt_metric(),
                },
                PhiComponent {
                    name: "AST".to_string(),
                    weight: 0.15,
                    value: self.base.attention.ast_metric(),
                },
                PhiComponent {
                    name: "PP".to_string(),
                    weight: 0.20,
                    value: self.base.predictor.pp_metric(),
                },
                PhiComponent {
                    name: "RPT".to_string(),
                    weight: 0.15,
                    value: self.base.recurrence.rpt_metric(),
                },
                PhiComponent {
                    name: "4E".to_string(),
                    weight: 0.15,
                    value: self.base.embodiment.embodiment_metric(),
                },
                PhiComponent {
                    name: "IIT".to_string(),
                    weight: 0.20,
                    value: phi,
                },
            ],
        }
    }
}

/// Implement ConsciousnessPhiProvider for EnhancedConsciousness
impl ConsciousnessPhiProvider for EnhancedConsciousness {
    fn compute_phi(&self) -> f64 {
        // Combine metrics from all theories using their actual methods
        let gwt = self.base.workspace.gwt_metric();
        let ast = self.base.attention.ast_metric();
        let pp = self.base.predictor.pp_metric();
        let rpt = self.base.recurrence.rpt_metric();
        let e4 = self.base.embodiment.embodiment_metric();
        let coherence = self.integrator.integration_coherence();

        // Weighted combination with integration coherence as modifier
        let raw_phi =
            gwt * 0.15 + ast * 0.15 + pp * 0.20 + rpt * 0.15 + e4 * 0.15 + coherence * 0.20;
        raw_phi.min(1.0).max(0.0)
    }

    fn phi_provenance(&self) -> PhiProvenance {
        self.compute_phi_provenance(self.compute_phi())
    }
}

/// Result from enhanced consciousness processing
#[derive(Debug, Clone)]
pub struct EnhancedConsciousnessResult {
    /// Base multi-theory result
    pub base_result: MultiTheoryResult,
    /// Cross-theory integration coherence
    pub integration_coherence: f64,
    /// Higher-order thought state
    pub hot_state: HigherOrderThought,
    /// Temporal state summary
    pub temporal_state: TemporalState,
    /// Execution decision from theory voting
    pub execution_decision: ExecutionDecision,
    /// Phenomenal properties
    pub phenomenal: PhenomenalState,
    /// User understanding description
    pub user_understanding: String,
    /// Full consciousness narrative
    pub narrative: String,
}

/// Temporal state summary
#[derive(Debug, Clone)]
pub struct TemporalState {
    pub session_duration: Duration,
    pub in_flow: bool,
    pub flow_score: f64,
    pub trend_unified: f64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_theory_integrator() {
        let integrator = CrossTheoryIntegrator::new();
        assert!(integrator.influences.gwt_broadcast_strength > 0.0);
    }

    #[test]
    fn test_higher_order_thought() {
        let mut hot = HigherOrderThought::new();
        let metrics = MultiTheoryMetrics {
            phi: 0.6,
            gwt: 0.5,
            ast: 0.7,
            pp: 0.4,
            rpt: 0.5,
            embodiment: 0.3,
            unified: 0.5,
        };
        let attention = AttentionSchema::new();
        hot.observe(&metrics, &attention);

        assert!(hot.self_awareness_level > 0.0);
        assert!(!hot.uncertainty_awareness.is_empty()); // Should detect PP uncertainty
    }

    #[test]
    fn test_phenomenal_state() {
        let metrics = MultiTheoryMetrics {
            phi: 0.8,
            gwt: 0.7,
            ast: 0.6,
            pp: 0.9,
            rpt: 0.5,
            embodiment: 0.7,
            unified: 0.7,
        };
        let flow = FlowState::default();
        let phenomenal = PhenomenalState::from_metrics(&metrics, &flow);

        assert!(phenomenal.valence > 0.0); // High metrics = positive valence
        assert!(phenomenal.clarity > 0.5); // High phi + ast = clarity
    }

    #[test]
    fn test_theory_voting() {
        let voting = TheoryVotingSystem::new();
        let metrics = MultiTheoryMetrics {
            phi: 0.7,
            gwt: 0.6,
            ast: 0.8,
            pp: 0.7,
            rpt: 0.5,
            embodiment: 0.6,
            unified: 0.65,
        };
        let mut attention = AttentionSchema::new();
        attention.focus_on("test", "testing", 0.8, AttentionCategory::Intent);

        let decision = voting.collect_votes(&metrics, &attention);

        assert!(!decision.votes.is_empty());
        // With high metrics, should lean toward execution
        assert!(decision.consensus > 0.0);
    }

    #[test]
    fn test_user_mental_model() {
        let mut model = UserMentalModel::new();
        model.infer_goal("install firefox", 0.8);

        assert!(!model.goals.is_empty());
        assert!(model.goals[0].goal.contains("Install"));

        model.record_behavior(BehaviorType::AcceptedSuggestion, "test");
        assert!(model.trust > 0.5);
    }

    #[test]
    fn test_temporal_consciousness() {
        let mut temporal = TemporalConsciousness::new();
        let metrics = MultiTheoryMetrics {
            phi: 0.7,
            gwt: 0.7,
            ast: 0.7,
            pp: 0.8,
            rpt: 0.5,
            embodiment: 0.5,
            unified: 0.65,
        };

        temporal.record(&metrics, "test input");

        assert!(!temporal.history.is_empty());
        assert!(temporal.flow_state.flow_score > 0.0);
    }

    #[test]
    fn test_enhanced_consciousness_integration() {
        let mut enhanced = EnhancedConsciousness::new();
        let primes = vec![SemanticPrime::Do, SemanticPrime::Move];

        let result = enhanced.process("install firefox", &primes, 0.5);

        assert!(!result.narrative.is_empty());
        assert!(!result.execution_decision.votes.is_empty());
        // Verify temporal state has a valid duration (it's a Duration type which is always valid)
        let _ = result.temporal_state.session_duration;
    }
}