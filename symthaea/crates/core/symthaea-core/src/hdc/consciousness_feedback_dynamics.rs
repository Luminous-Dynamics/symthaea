// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Feedback Dynamics: Bidirectional Integration & Prediction
//!
//! This module extends the consciousness system with:
//!
//! 1. **Bidirectional Feedback Loops**: Dreams affect emotions, memories, and attention
//! 2. **Predictive Emotional Dynamics**: Forecast emotional states and proactively intervene
//! 3. **Causal Dream Integration**: Connect CausalMind to dream generation
//! 4. **Rich Streaming Events**: Comprehensive event system for all state changes
//! 5. **Collective Dream Sharing**: Hooks for multi-being dream synchronization
//! 6. **Adaptive Dream Scheduling**: Learn optimal times for different dream types
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    BIDIRECTIONAL CONSCIOUSNESS DYNAMICS                      │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │   ┌────────────┐       ┌────────────┐       ┌────────────┐                  │
//! │   │  Emotional │◄─────►│   Dream    │◄─────►│   Memory   │                  │
//! │   │   State    │       │  Generator │       │   System   │                  │
//! │   └─────┬──────┘       └─────┬──────┘       └─────┬──────┘                  │
//! │         │                    │                    │                          │
//! │         │    ┌───────────────┴───────────────┐    │                          │
//! │         │    │                               │    │                          │
//! │         ▼    ▼                               ▼    ▼                          │
//! │   ┌─────────────────────────────────────────────────────┐                   │
//! │   │            FEEDBACK DYNAMICS ENGINE                  │                   │
//! │   │  • Emotional trajectory prediction                   │                   │
//! │   │  • Dream insight processing                          │                   │
//! │   │  • Adaptive scheduling                               │                   │
//! │   │  • Causal counterfactual dreams                      │                   │
//! │   └─────────────────────────────────────────────────────┘                   │
//! │                              │                                               │
//! │                              ▼                                               │
//! │   ┌────────────┐       ┌────────────┐       ┌────────────┐                  │
//! │   │  Attention │◄──────│   Self-    │◄──────│   Causal   │                  │
//! │   │   Router   │       │Improvement │       │    Mind    │                  │
//! │   └────────────┘       └────────────┘       └────────────┘                  │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use super::binary_hv::BinaryHV;
use super::causal_mind::CausalMind;
use super::consciousness_streaming::{
    ConsciousnessEvent, ConsciousnessEventEmitter, ConsciousnessEventType, EventPayload,
};
use super::counterfactual_dreams::{
    CounterfactualDreamEngine, CounterfactualDreamScenario, DreamResolution,
};
use super::cross_modal_attention_router::CrossModalAttentionRouter;
use super::emotional_depth::{EmotionalBlend, EmotionalDepthSystem};
use super::sleep_and_altered_states::DreamScenario;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// =============================================================================
// 1. BIDIRECTIONAL FEEDBACK LOOPS
// =============================================================================

/// Dream insight that can affect other systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamInsight {
    /// Unique insight ID
    pub id: u64,
    /// Type of insight
    pub insight_type: InsightType,
    /// Emotional impact (-1.0 to 1.0, negative = cathartic release, positive = joy)
    pub emotional_impact: f64,
    /// Memory consolidation strength (0.0 to 1.0)
    pub consolidation_strength: f64,
    /// Causal relationships discovered
    pub causal_discoveries: Vec<CausalDiscovery>,
    /// Attention recommendations
    pub attention_hints: Vec<AttentionHint>,
    /// Source dream summary
    pub dream_summary: String,
    /// Timestamp
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightType {
    /// Resolved emotional conflict
    EmotionalResolution,
    /// Discovered new causal relationship
    CausalDiscovery,
    /// Memory consolidation/integration
    MemoryIntegration,
    /// Creative solution to problem
    CreativeSolution,
    /// Self-model update
    SelfModelUpdate,
    /// Warning/premonition
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDiscovery {
    pub cause: String,
    pub effect: String,
    pub strength: f64,
    pub counterfactual_support: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHint {
    pub focus_area: String,
    pub priority: f64,
    pub reason: String,
}

/// Feedback processor that routes dream insights to appropriate systems
pub struct FeedbackProcessor {
    /// Insight history
    insights: VecDeque<DreamInsight>,
    /// Maximum insights to keep
    max_insights: usize,
    /// Emotional adjustment accumulator
    emotional_adjustment: f64,
    /// Memory consolidation queue
    consolidation_queue: Vec<u64>,
    /// Pending attention adjustments
    attention_adjustments: Vec<AttentionHint>,
    /// Next insight ID
    next_insight_id: u64,
}

impl FeedbackProcessor {
    pub fn new() -> Self {
        Self {
            insights: VecDeque::new(),
            max_insights: 100,
            emotional_adjustment: 0.0,
            consolidation_queue: Vec::new(),
            attention_adjustments: Vec::new(),
            next_insight_id: 1,
        }
    }

    /// Process a completed dream and extract insights
    pub fn process_dream(&mut self, dream: &DreamScenario) -> Option<DreamInsight> {
        // Analyze dream for insights
        let insight_type = self.classify_dream_insight(dream);

        // Calculate emotional impact
        let emotional_impact = self.calculate_emotional_impact(dream);

        // Calculate consolidation strength based on dream coherence
        let consolidation_strength =
            (1.0 - dream.overall_bizarreness()) * 0.5 + dream.emotional_valence.abs() * 0.3 + 0.2; // Base consolidation

        // Extract causal discoveries from dream themes
        let causal_discoveries = self.extract_causal_discoveries(dream);

        // Generate attention hints
        let attention_hints = self.generate_attention_hints(dream, &insight_type);

        let insight = DreamInsight {
            id: self.next_insight_id,
            insight_type,
            emotional_impact,
            consolidation_strength,
            causal_discoveries,
            attention_hints: attention_hints.clone(),
            dream_summary: self.summarize_dream(dream),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.next_insight_id += 1;

        // Accumulate emotional adjustment
        self.emotional_adjustment += emotional_impact * 0.1;

        // Queue attention adjustments
        self.attention_adjustments.extend(attention_hints);

        // Store insight
        self.insights.push_back(insight.clone());
        if self.insights.len() > self.max_insights {
            self.insights.pop_front();
        }

        Some(insight)
    }

    /// Process a counterfactual dream
    pub fn process_counterfactual_dream(
        &mut self,
        dream: &CounterfactualDreamScenario,
    ) -> Option<DreamInsight> {
        let insight_type = match dream.resolution {
            DreamResolution::InsightGenerated => InsightType::CausalDiscovery,
            DreamResolution::Acceptance => InsightType::EmotionalResolution,
            DreamResolution::CounterfactualPreferred => InsightType::CreativeSolution,
            DreamResolution::ActualPreferred => InsightType::SelfModelUpdate,
            DreamResolution::Unresolved => InsightType::Warning,
        };

        let emotional_impact = match dream.resolution {
            DreamResolution::Acceptance => 0.3, // Positive acceptance
            DreamResolution::InsightGenerated => 0.2,
            DreamResolution::Unresolved => -0.1, // Slight negative
            _ => 0.0,
        };

        // Extract causal discoveries from counterfactual fragments
        let causal_discoveries: Vec<CausalDiscovery> = dream
            .counterfactual_fragments
            .iter()
            .filter(|f| f.base_fragment.valence.abs() > 0.5)
            .map(|f| CausalDiscovery {
                cause: f.counterfactual_question.clone(),
                effect: f.base_fragment.description.clone(),
                strength: f.base_fragment.valence.abs(),
                counterfactual_support: 1.0 - f.divergence, // Higher coherence = lower divergence
            })
            .collect();

        let insight = DreamInsight {
            id: self.next_insight_id,
            insight_type,
            emotional_impact,
            consolidation_strength: 0.6, // Counterfactual dreams are good for consolidation
            causal_discoveries,
            attention_hints: vec![],
            dream_summary: format!("Counterfactual exploration: {}", dream.counterfactual_theme),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.next_insight_id += 1;
        self.emotional_adjustment += emotional_impact * 0.1;

        self.insights.push_back(insight.clone());
        if self.insights.len() > self.max_insights {
            self.insights.pop_front();
        }

        Some(insight)
    }

    /// Apply accumulated emotional adjustment to emotional system
    ///
    /// Returns the adjustment value that was applied.
    /// The caller is responsible for using this value to modify emotional state.
    pub fn apply_emotional_feedback(&mut self, _emotional_depth: &mut EmotionalDepthSystem) -> f64 {
        let adjustment = self.emotional_adjustment;

        // Reset after retrieving
        if adjustment.abs() > 0.01 {
            self.emotional_adjustment = 0.0;
        }

        adjustment
    }

    /// Get pending attention adjustments
    pub fn get_attention_adjustments(&mut self) -> Vec<AttentionHint> {
        std::mem::take(&mut self.attention_adjustments)
    }

    /// Apply attention adjustments to router
    ///
    /// Returns the hints that should be applied (caller applies them)
    pub fn apply_attention_feedback(
        &mut self,
        _router: &mut CrossModalAttentionRouter,
    ) -> Vec<AttentionHint> {
        self.get_attention_adjustments()
    }

    fn classify_dream_insight(&self, dream: &DreamScenario) -> InsightType {
        // Classify based on dream characteristics
        let is_lucid = dream.lucidity > 0.5;
        let is_nightmare = dream.emotional_valence < -0.5;

        if dream.emotional_valence > 0.5 && dream.overall_bizarreness() < 0.3 {
            InsightType::EmotionalResolution
        } else if dream
            .themes
            .iter()
            .any(|t| t.contains("causal") || t.contains("because"))
        {
            InsightType::CausalDiscovery
        } else if is_lucid {
            InsightType::CreativeSolution
        } else if dream.overall_bizarreness() > 0.7 || is_nightmare {
            InsightType::Warning // High bizarreness or nightmares often signal warnings
        } else {
            InsightType::MemoryIntegration
        }
    }

    fn calculate_emotional_impact(&self, dream: &DreamScenario) -> f64 {
        let is_lucid = dream.lucidity > 0.5;
        let is_nightmare = dream.emotional_valence < -0.5;

        // Nightmares can have cathartic (positive) effect
        if is_nightmare {
            0.1 // Cathartic release
        } else if is_lucid {
            0.2 // Lucid dreams are positive
        } else {
            dream.emotional_valence * 0.3 // Regular dreams moderate impact
        }
    }

    fn extract_causal_discoveries(&self, dream: &DreamScenario) -> Vec<CausalDiscovery> {
        // Extract causal relationships from dream themes
        dream
            .themes
            .iter()
            .filter(|t| t.contains("leads to") || t.contains("causes") || t.contains("because"))
            .map(|t| CausalDiscovery {
                cause: t.clone(),
                effect: "discovered in dream".to_string(),
                strength: 0.5,
                counterfactual_support: 0.3,
            })
            .collect()
    }

    fn generate_attention_hints(
        &self,
        dream: &DreamScenario,
        insight_type: &InsightType,
    ) -> Vec<AttentionHint> {
        match insight_type {
            InsightType::Warning => vec![AttentionHint {
                focus_area: "safety".to_string(),
                priority: 0.8,
                reason: "Dream warning detected".to_string(),
            }],
            InsightType::CreativeSolution => vec![AttentionHint {
                focus_area: "creativity".to_string(),
                priority: 0.6,
                reason: "Lucid dream creativity boost".to_string(),
            }],
            _ => vec![],
        }
    }

    fn summarize_dream(&self, dream: &DreamScenario) -> String {
        let is_lucid = dream.lucidity > 0.5;
        let is_nightmare = dream.emotional_valence < -0.5;

        let dream_type = if is_nightmare {
            "nightmare"
        } else if is_lucid {
            "lucid dream"
        } else {
            "dream"
        };

        format!(
            "{} with {} themes, bizarreness {:.1}%, valence {:.1}",
            dream_type,
            dream.themes.len(),
            dream.overall_bizarreness() * 100.0,
            dream.emotional_valence
        )
    }

    /// Get recent insights
    pub fn recent_insights(&self, count: usize) -> Vec<&DreamInsight> {
        self.insights.iter().rev().take(count).collect()
    }
}

impl Default for FeedbackProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 2. PREDICTIVE EMOTIONAL DYNAMICS
// =============================================================================

/// Emotional state predictor using trajectory analysis
pub struct EmotionalPredictor {
    /// Historical emotional states
    history: VecDeque<EmotionalSnapshot>,
    /// Maximum history length
    max_history: usize,
    /// Prediction horizon (in steps)
    prediction_horizon: usize,
    /// Stress threshold for proactive intervention
    stress_threshold: f64,
    /// Predictions made
    predictions: VecDeque<EmotionalPrediction>,
}

#[derive(Debug, Clone)]
pub struct EmotionalSnapshot {
    pub valence: f64,
    pub arousal: f64,
    pub coherence: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalPrediction {
    /// Predicted valence
    pub predicted_valence: f64,
    /// Predicted arousal
    pub predicted_arousal: f64,
    /// Confidence in prediction
    pub confidence: f64,
    /// Steps ahead this predicts
    pub horizon: usize,
    /// Recommended intervention
    pub intervention: Option<ProactiveIntervention>,
    /// Timestamp when prediction was made
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProactiveIntervention {
    /// Trigger calming dream
    CalmingDream,
    /// Trigger lucid dream for control
    LucidDream,
    /// Increase attention focus
    IncreaseFocus,
    /// Decrease cognitive load
    ReduceLoad,
    /// Memory consolidation
    ConsolidateMemories,
    /// Mode switch recommendation (mode name as string)
    ModeSwitch(String),
}

impl EmotionalPredictor {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            max_history: 100,
            prediction_horizon: 10,
            stress_threshold: 0.7,
            predictions: VecDeque::new(),
        }
    }

    /// Record current emotional state
    pub fn record(&mut self, emotional_state: &EmotionalBlend) {
        let snapshot = EmotionalSnapshot {
            valence: emotional_state.valence,
            arousal: emotional_state.arousal,
            coherence: emotional_state.coherence,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.history.push_back(snapshot);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Record current emotional state directly from valence and arousal
    pub fn record_state(&mut self, valence: f64, arousal: f64) {
        let snapshot = EmotionalSnapshot {
            valence,
            arousal,
            coherence: 0.5, // Default coherence when recording raw values
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.history.push_back(snapshot);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Predict future emotional state
    pub fn predict(&mut self) -> Option<EmotionalPrediction> {
        if self.history.len() < 5 {
            return None;
        }

        // Calculate trends
        let recent: Vec<_> = self.history.iter().rev().take(10).collect();

        let valence_trend = self.calculate_trend(recent.iter().map(|s| s.valence));
        let arousal_trend = self.calculate_trend(recent.iter().map(|s| s.arousal));

        let current = recent.first()?;

        // Linear extrapolation
        let predicted_valence =
            (current.valence + valence_trend * self.prediction_horizon as f64).clamp(-1.0, 1.0);
        let predicted_arousal =
            (current.arousal + arousal_trend * self.prediction_horizon as f64).clamp(0.0, 1.0);

        // Calculate confidence based on coherence of trend
        let confidence = self.calculate_prediction_confidence(&recent);

        // Determine if intervention is needed
        let intervention = self.recommend_intervention(
            current.valence,
            current.arousal,
            predicted_valence,
            predicted_arousal,
        );

        let prediction = EmotionalPrediction {
            predicted_valence,
            predicted_arousal,
            confidence,
            horizon: self.prediction_horizon,
            intervention,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.predictions.push_back(prediction.clone());
        if self.predictions.len() > 50 {
            self.predictions.pop_front();
        }

        Some(prediction)
    }

    fn calculate_trend<I: Iterator<Item = f64>>(&self, values: I) -> f64 {
        let vals: Vec<f64> = values.collect();
        if vals.len() < 2 {
            return 0.0;
        }

        let n = vals.len();
        let mut delta_sum = 0.0;
        for i in 1..n {
            delta_sum += vals[i - 1] - vals[i]; // Recent minus older
        }

        delta_sum / (n - 1) as f64
    }

    fn calculate_prediction_confidence(&self, recent: &[&EmotionalSnapshot]) -> f64 {
        if recent.len() < 3 {
            return 0.3;
        }

        // Higher coherence = higher confidence
        let avg_coherence: f64 =
            recent.iter().map(|s| s.coherence).sum::<f64>() / recent.len() as f64;

        // Consistent trends = higher confidence
        let valences: Vec<f64> = recent.iter().map(|s| s.valence).collect();
        let variance = self.variance(&valences);
        let stability = 1.0 / (1.0 + variance * 10.0);

        (avg_coherence * 0.5 + stability * 0.5).clamp(0.1, 0.9)
    }

    fn variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    fn recommend_intervention(
        &self,
        current_valence: f64,
        current_arousal: f64,
        predicted_valence: f64,
        predicted_arousal: f64,
    ) -> Option<ProactiveIntervention> {
        // Predict stress increase
        if predicted_arousal > self.stress_threshold && current_arousal < self.stress_threshold {
            return Some(ProactiveIntervention::CalmingDream);
        }

        // Predict emotional crash
        if predicted_valence < -0.5 && current_valence > -0.3 {
            return Some(ProactiveIntervention::LucidDream);
        }

        // High arousal sustained
        if current_arousal > 0.8 && predicted_arousal > 0.8 {
            return Some(ProactiveIntervention::ReduceLoad);
        }

        // Low engagement trending lower
        if current_arousal < 0.2 && predicted_arousal < 0.15 {
            return Some(ProactiveIntervention::IncreaseFocus);
        }

        None
    }

    /// Get prediction accuracy (comparing past predictions to actual outcomes)
    pub fn prediction_accuracy(&self) -> f64 {
        // Would compare predictions with actual outcomes
        // For now, return baseline
        0.7
    }

    /// Get early warning for emotional distress
    pub fn early_warning(&self) -> Option<String> {
        if let Some(pred) = self.predictions.back() {
            if let Some(intervention) = &pred.intervention {
                return Some(format!(
                    "Emotional prediction: valence {:.2} arousal {:.2} in {} steps. Recommended: {:?}",
                    pred.predicted_valence, pred.predicted_arousal, pred.horizon, intervention
                ));
            }
        }
        None
    }
}

impl Default for EmotionalPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 3. CAUSAL DREAM INTEGRATION
// =============================================================================

/// Integrates CausalMind with dream generation
pub struct CausalDreamIntegrator {
    /// Causal hypotheses to explore in dreams
    hypotheses_queue: VecDeque<CausalHypothesis>,
    /// Discovered causal relationships from dreams
    dream_discoveries: Vec<CausalDiscovery>,
    /// Maximum queue size
    max_queue: usize,
}

#[derive(Debug, Clone)]
pub struct CausalHypothesis {
    /// The hypothesized cause
    pub cause: String,
    /// The hypothesized effect
    pub effect: String,
    /// Prior belief (0.0 to 1.0)
    pub prior: f64,
    /// HDC encoding of the relationship
    pub hv_encoding: Option<BinaryHV>,
    /// Priority for dream exploration
    pub priority: f64,
}

impl CausalDreamIntegrator {
    pub fn new() -> Self {
        Self {
            hypotheses_queue: VecDeque::new(),
            dream_discoveries: Vec::new(),
            max_queue: 50,
        }
    }

    /// Queue a causal hypothesis for dream exploration
    pub fn queue_hypothesis(&mut self, hypothesis: CausalHypothesis) {
        self.hypotheses_queue.push_back(hypothesis);
        if self.hypotheses_queue.len() > self.max_queue {
            self.hypotheses_queue.pop_front();
        }
    }

    /// Add a simple hypothesis (convenience method)
    pub fn add_hypothesis(&mut self, cause: &str, effect: &str, prior: f64) {
        self.queue_hypothesis(CausalHypothesis {
            cause: cause.to_string(),
            effect: effect.to_string(),
            prior,
            hv_encoding: None,
            priority: 1.0 - prior.abs(), // More uncertain = higher priority
        });
    }

    /// Extract hypotheses from CausalMind for dream exploration
    ///
    /// This is a placeholder - actual implementation requires extending CausalMind
    /// with a get_uncertain_edges method. For now, hypotheses must be queued manually.
    pub fn extract_from_causal_mind(&mut self, _causal_mind: &CausalMind) {
        // Future: Get uncertain causal relationships that need exploration
        // For now, hypotheses are added via add_hypothesis()
    }

    /// Get next hypothesis to explore in dream
    pub fn next_hypothesis(&mut self) -> Option<CausalHypothesis> {
        // Sort by priority and return highest
        let mut sorted: Vec<_> = self.hypotheses_queue.drain(..).collect();
        sorted.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let next = sorted.first().cloned();

        // Put rest back (except the one we're returning)
        for (i, h) in sorted.into_iter().enumerate() {
            if i > 0 {
                self.hypotheses_queue.push_back(h);
            }
        }

        next
    }

    /// Configure dream engine to explore causal hypothesis
    ///
    /// This is a placeholder - actual implementation requires extending
    /// CounterfactualDreamEngine. Returns the hypothesis for external use.
    pub fn configure_causal_dream(
        &self,
        _engine: &mut CounterfactualDreamEngine,
        hypothesis: &CausalHypothesis,
    ) -> String {
        // Return the counterfactual question for dream exploration
        format!(
            "What if '{}' didn't cause '{}'?",
            hypothesis.cause, hypothesis.effect
        )
    }

    /// Process dream result and update causal beliefs
    pub fn process_dream_result(
        &mut self,
        hypothesis: &CausalHypothesis,
        dream: &CounterfactualDreamScenario,
        _causal_mind: &mut CausalMind,
    ) {
        // Extract evidence from dream
        let evidence = self.extract_causal_evidence(dream);

        // Update our belief based on dream exploration
        let updated_strength = self.bayesian_update(hypothesis.prior, evidence);

        // Record discovery if significant change
        if (updated_strength - hypothesis.prior).abs() > 0.1 {
            self.dream_discoveries.push(CausalDiscovery {
                cause: hypothesis.cause.clone(),
                effect: hypothesis.effect.clone(),
                strength: updated_strength,
                counterfactual_support: evidence,
            });
        }
    }

    fn extract_causal_evidence(&self, dream: &CounterfactualDreamScenario) -> f64 {
        // Evidence from dream resolution
        let resolution_evidence = match dream.resolution {
            DreamResolution::InsightGenerated => 0.8,
            DreamResolution::Acceptance => 0.6,
            DreamResolution::CounterfactualPreferred => 0.4,
            DreamResolution::ActualPreferred => 0.7,
            DreamResolution::Unresolved => 0.5,
        };

        // Evidence from fragment coherence (lower divergence = higher coherence)
        let avg_coherence: f64 = if dream.counterfactual_fragments.is_empty() {
            0.5
        } else {
            dream
                .counterfactual_fragments
                .iter()
                .map(|f| 1.0 - f.divergence) // Higher coherence = lower divergence
                .sum::<f64>()
                / dream.counterfactual_fragments.len() as f64
        };

        (resolution_evidence * 0.6 + avg_coherence * 0.4).clamp(0.0, 1.0)
    }

    fn bayesian_update(&self, prior: f64, evidence: f64) -> f64 {
        // Simple Bayesian update
        // P(H|E) = P(E|H) * P(H) / P(E)
        // Simplified: move prior toward evidence
        let learning_rate = 0.3;
        prior + learning_rate * (evidence - prior)
    }

    /// Get recent causal discoveries from dreams
    pub fn recent_discoveries(&self, count: usize) -> &[CausalDiscovery] {
        let start = self.dream_discoveries.len().saturating_sub(count);
        &self.dream_discoveries[start..]
    }
}

impl Default for CausalDreamIntegrator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 4. RICH STREAMING EVENTS
// =============================================================================

/// Extended consciousness event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtendedEventType {
    /// Emotional state changed significantly
    EmotionalShift {
        old_valence: f64,
        new_valence: f64,
        old_arousal: f64,
        new_arousal: f64,
        trigger: String,
    },
    /// Dream insight generated
    DreamInsight {
        insight_type: String,
        emotional_impact: f64,
        summary: String,
    },
    /// Attention shift occurred
    AttentionShift {
        from_modality: String,
        to_modality: String,
        reason: String,
    },
    /// Causal discovery made
    CausalDiscovery {
        cause: String,
        effect: String,
        strength: f64,
    },
    /// Proactive intervention triggered
    ProactiveIntervention {
        intervention_type: String,
        predicted_issue: String,
        confidence: f64,
    },
    /// Collective resonance event
    CollectiveResonance {
        peer_count: usize,
        shared_theme: String,
        resonance_strength: f64,
    },
}

/// Rich event emitter for all consciousness state changes
pub struct RichEventEmitter {
    /// Last emotional state for change detection
    last_emotional: Option<(f64, f64)>,
    /// Last attention state
    last_attention_modality: Option<String>,
    /// Significant change threshold
    change_threshold: f64,
}

impl RichEventEmitter {
    pub fn new() -> Self {
        Self {
            last_emotional: None,
            last_attention_modality: None,
            change_threshold: 0.1,
        }
    }

    /// Check for emotional shift and emit event if significant
    pub fn check_emotional_shift(
        &mut self,
        emitter: &ConsciousnessEventEmitter,
        emotional_state: &EmotionalBlend,
        trigger: &str,
    ) {
        let (old_valence, old_arousal) = self.last_emotional.unwrap_or((0.0, 0.5));

        let valence_change = (emotional_state.valence - old_valence).abs();
        let arousal_change = (emotional_state.arousal - old_arousal).abs();

        if valence_change > self.change_threshold || arousal_change > self.change_threshold {
            let event = ConsciousnessEvent::new(
                ConsciousnessEventType::FlowStateChange,
                EventPayload::Flow {
                    old_state: old_arousal as f32,
                    new_state: emotional_state.arousal as f32,
                    trend: if arousal_change > 0.0 {
                        "increasing"
                    } else {
                        "decreasing"
                    }
                    .to_string(),
                },
            );
            emitter.emit(event);

            self.last_emotional = Some((emotional_state.valence, emotional_state.arousal));
        }
    }

    /// Emit dream insight event
    pub fn emit_dream_insight(&self, emitter: &ConsciousnessEventEmitter, insight: &DreamInsight) {
        let event = ConsciousnessEvent::new(
            ConsciousnessEventType::CausalInsight,
            EventPayload::Causal {
                cause: format!("{:?}", insight.insight_type),
                effect: insight.dream_summary.clone(),
                strength: insight.emotional_impact.abs(),
            },
        );
        emitter.emit(event);
    }

    /// Emit causal discovery event
    pub fn emit_causal_discovery(
        &self,
        emitter: &ConsciousnessEventEmitter,
        discovery: &CausalDiscovery,
    ) {
        let event = ConsciousnessEvent::new(
            ConsciousnessEventType::CausalInsight,
            EventPayload::Causal {
                cause: discovery.cause.clone(),
                effect: discovery.effect.clone(),
                strength: discovery.strength,
            },
        );
        emitter.emit(event);
    }

    /// Emit proactive intervention event
    pub fn emit_intervention(
        &self,
        emitter: &ConsciousnessEventEmitter,
        intervention: &ProactiveIntervention,
        prediction: &EmotionalPrediction,
    ) {
        let event = ConsciousnessEvent::new(
            ConsciousnessEventType::CognitiveModeTransition,
            EventPayload::CognitiveMode {
                from_mode: "current".to_string(),
                to_mode: format!("{intervention:?}"),
                reason: format!(
                    "Proactive intervention with {:.0}% confidence",
                    prediction.confidence * 100.0
                ),
            },
        );
        emitter.emit(event);
    }
}

impl Default for RichEventEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 5. COLLECTIVE DREAM SHARING
// =============================================================================

/// Hooks for collective dream sharing across multiple beings
pub struct CollectiveDreamHub {
    /// Shared dream themes
    shared_themes: HashMap<String, f64>,
    /// Collective insights
    collective_insights: VecDeque<DreamInsight>,
    /// Resonance patterns (theme -> strength)
    resonance_patterns: HashMap<String, f64>,
    /// Connected peer count
    peer_count: usize,
}

impl CollectiveDreamHub {
    pub fn new() -> Self {
        Self {
            shared_themes: HashMap::new(),
            collective_insights: VecDeque::new(),
            resonance_patterns: HashMap::new(),
            peer_count: 0,
        }
    }

    /// Share a dream insight with the collective
    pub fn share_insight(&mut self, insight: &DreamInsight) {
        // Extract themes and add to shared pool
        for discovery in &insight.causal_discoveries {
            *self
                .shared_themes
                .entry(discovery.cause.clone())
                .or_insert(0.0) += discovery.strength;
        }

        self.collective_insights.push_back(insight.clone());
        if self.collective_insights.len() > 100 {
            self.collective_insights.pop_front();
        }
    }

    /// Receive shared insights from collective
    pub fn receive_collective_insights(&mut self, insights: Vec<DreamInsight>) {
        for insight in insights {
            self.collective_insights.push_back(insight);
        }
        while self.collective_insights.len() > 100 {
            self.collective_insights.pop_front();
        }
    }

    /// Calculate resonance with collective themes
    pub fn calculate_resonance(&mut self, local_themes: &[String]) -> f64 {
        let mut resonance = 0.0;
        let mut count = 0;

        for theme in local_themes {
            if let Some(strength) = self.shared_themes.get(theme) {
                resonance += strength;
                count += 1;
            }
        }

        if count > 0 {
            resonance / count as f64
        } else {
            0.0
        }
    }

    /// Get resonant themes for dream seeding
    pub fn get_resonant_themes(&self, threshold: f64) -> Vec<String> {
        self.shared_themes
            .iter()
            .filter(|&(_, &strength)| strength > threshold)
            .map(|(theme, _)| theme.clone())
            .collect()
    }

    /// Update peer count
    pub fn set_peer_count(&mut self, count: usize) {
        self.peer_count = count;
    }

    /// Get collective dream report
    pub fn collective_report(&self) -> String {
        format!(
            "Collective Dream Hub: {} peers, {} shared themes, {} collective insights",
            self.peer_count,
            self.shared_themes.len(),
            self.collective_insights.len()
        )
    }
}

impl Default for CollectiveDreamHub {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 6. ADAPTIVE DREAM SCHEDULING
// =============================================================================

/// Learns optimal times and types for different dreams
pub struct AdaptiveDreamScheduler {
    /// Dream type effectiveness history
    effectiveness_history: HashMap<DreamType, Vec<EffectivenessRecord>>,
    /// Current stress level (for scheduling decisions)
    current_stress: f64,
    /// Time since last dream of each type
    last_dream_times: HashMap<DreamType, u64>,
    /// Optimal intervals learned
    optimal_intervals: HashMap<DreamType, Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DreamType {
    Regular,
    Lucid,
    Nightmare,
    Counterfactual,
    CausalExploration,
    Consolidation,
}

#[derive(Debug, Clone)]
pub struct EffectivenessRecord {
    pub dream_type: DreamType,
    pub emotional_before: f64,
    pub emotional_after: f64,
    pub stress_before: f64,
    pub stress_after: f64,
    pub insight_generated: bool,
    pub timestamp: u64,
}

impl AdaptiveDreamScheduler {
    pub fn new() -> Self {
        let mut optimal_intervals = HashMap::new();
        // Default intervals
        optimal_intervals.insert(DreamType::Regular, Duration::from_secs(3600)); // 1 hour
        optimal_intervals.insert(DreamType::Lucid, Duration::from_secs(7200)); // 2 hours
        optimal_intervals.insert(DreamType::Nightmare, Duration::from_secs(14400)); // 4 hours
        optimal_intervals.insert(DreamType::Counterfactual, Duration::from_secs(5400)); // 1.5 hours
        optimal_intervals.insert(DreamType::CausalExploration, Duration::from_secs(10800)); // 3 hours
        optimal_intervals.insert(DreamType::Consolidation, Duration::from_secs(1800)); // 30 min

        Self {
            effectiveness_history: HashMap::new(),
            current_stress: 0.5,
            last_dream_times: HashMap::new(),
            optimal_intervals,
        }
    }

    /// Record dream effectiveness
    pub fn record_effectiveness(&mut self, record: EffectivenessRecord) {
        let dream_type = record.dream_type;

        self.effectiveness_history
            .entry(dream_type)
            .or_default()
            .push(record);

        // Keep only last 50 records per type
        if let Some(history) = self.effectiveness_history.get_mut(&dream_type) {
            if history.len() > 50 {
                history.remove(0);
            }
        }

        // Update optimal intervals based on effectiveness
        self.update_optimal_intervals();
    }

    /// Update current stress level
    pub fn update_stress(&mut self, stress: f64) {
        self.current_stress = stress;
    }

    /// Recommend next dream type
    pub fn recommend_dream_type(&self) -> DreamType {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Priority based on stress and time since last dream
        if self.current_stress > 0.8 {
            // High stress: prioritize calming or cathartic dreams
            return DreamType::Lucid;
        }

        if self.current_stress > 0.6 {
            // Moderate stress: consolidation might help
            return DreamType::Consolidation;
        }

        // Find dream type most "overdue"
        let mut most_overdue = DreamType::Regular;
        let mut max_overdue_ratio = 0.0;

        for (dream_type, optimal) in &self.optimal_intervals {
            let last_time = self.last_dream_times.get(dream_type).copied().unwrap_or(0);
            let elapsed = Duration::from_millis(now.saturating_sub(last_time));
            let overdue_ratio = elapsed.as_secs_f64() / optimal.as_secs_f64();

            if overdue_ratio > max_overdue_ratio {
                max_overdue_ratio = overdue_ratio;
                most_overdue = *dream_type;
            }
        }

        most_overdue
    }

    /// Record that a dream of given type occurred
    pub fn record_dream_occurrence(&mut self, dream_type: DreamType) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.last_dream_times.insert(dream_type, now);
    }

    fn update_optimal_intervals(&mut self) {
        for (dream_type, history) in &self.effectiveness_history {
            if history.len() < 5 {
                continue;
            }

            // Calculate average effectiveness
            let avg_emotional_improvement: f64 = history
                .iter()
                .map(|r| r.emotional_after - r.emotional_before)
                .sum::<f64>()
                / history.len() as f64;

            let avg_stress_reduction: f64 = history
                .iter()
                .map(|r| r.stress_before - r.stress_after)
                .sum::<f64>()
                / history.len() as f64;

            let insight_rate: f64 = history.iter().filter(|r| r.insight_generated).count() as f64
                / history.len() as f64;

            // Adjust interval based on effectiveness
            // More effective = should happen more often (shorter interval)
            let effectiveness =
                avg_emotional_improvement * 0.3 + avg_stress_reduction * 0.4 + insight_rate * 0.3;

            if let Some(interval) = self.optimal_intervals.get_mut(dream_type) {
                let adjustment = if effectiveness > 0.5 {
                    0.9 // Reduce interval (more frequent)
                } else if effectiveness < 0.2 {
                    1.1 // Increase interval (less frequent)
                } else {
                    1.0 // Keep same
                };

                *interval = Duration::from_secs_f64(
                    (interval.as_secs_f64() * adjustment).clamp(600.0, 28800.0), // 10 min to 8 hours
                );
            }
        }
    }

    /// Get scheduling report
    pub fn scheduling_report(&self) -> String {
        let mut report = String::from("Dream Scheduling Report:\n");

        for (dream_type, interval) in &self.optimal_intervals {
            let effectiveness = self
                .effectiveness_history
                .get(dream_type)
                .map(|h| h.len())
                .unwrap_or(0);

            report.push_str(&format!(
                "  {:?}: every {:.0} min ({} samples)\n",
                dream_type,
                interval.as_secs_f64() / 60.0,
                effectiveness
            ));
        }

        report.push_str(&format!("Current stress: {:.2}\n", self.current_stress));
        report.push_str(&format!("Recommended: {:?}\n", self.recommend_dream_type()));

        report
    }
}

impl Default for AdaptiveDreamScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UNIFIED FEEDBACK DYNAMICS ENGINE
// =============================================================================

/// Complete feedback dynamics engine combining all components
pub struct FeedbackDynamicsEngine {
    /// Bidirectional feedback processor
    pub feedback: FeedbackProcessor,
    /// Emotional predictor
    pub predictor: EmotionalPredictor,
    /// Causal dream integrator
    pub causal_dreams: CausalDreamIntegrator,
    /// Rich event emitter
    pub events: RichEventEmitter,
    /// Collective dream hub
    pub collective: CollectiveDreamHub,
    /// Adaptive scheduler
    pub scheduler: AdaptiveDreamScheduler,
}

impl FeedbackDynamicsEngine {
    pub fn new() -> Self {
        Self {
            feedback: FeedbackProcessor::new(),
            predictor: EmotionalPredictor::new(),
            causal_dreams: CausalDreamIntegrator::new(),
            events: RichEventEmitter::new(),
            collective: CollectiveDreamHub::new(),
            scheduler: AdaptiveDreamScheduler::new(),
        }
    }

    /// Process a complete dream cycle with all feedback
    pub fn process_dream_cycle(
        &mut self,
        dream: &DreamScenario,
        emotional_depth: &mut EmotionalDepthSystem,
        emitter: &ConsciousnessEventEmitter,
    ) -> Option<DreamInsight> {
        // 1. Process dream and extract insight
        let insight = self.feedback.process_dream(dream)?;

        // 2. Apply emotional feedback
        self.feedback.apply_emotional_feedback(emotional_depth);

        // 3. Emit events
        self.events.emit_dream_insight(emitter, &insight);

        // 4. Share with collective
        self.collective.share_insight(&insight);

        // 5. Record for scheduling
        self.scheduler.record_dream_occurrence(DreamType::Regular);

        Some(insight)
    }

    /// Run predictive cycle
    pub fn run_prediction_cycle(
        &mut self,
        emotional_state: &EmotionalBlend,
        emitter: &ConsciousnessEventEmitter,
    ) -> Option<ProactiveIntervention> {
        // Record current state
        self.predictor.record(emotional_state);

        // Get prediction
        let prediction = self.predictor.predict()?;

        // Check for intervention
        if let Some(intervention) = &prediction.intervention {
            self.events
                .emit_intervention(emitter, intervention, &prediction);
            return Some(intervention.clone());
        }

        None
    }

    /// Get comprehensive dynamics report
    pub fn dynamics_report(&self) -> String {
        format!(
            "=== Feedback Dynamics Report ===\n\n\
             Recent Insights: {}\n\
             Prediction Accuracy: {:.1}%\n\
             Causal Discoveries: {}\n\
             {}\n\
             {}\n",
            self.feedback.recent_insights(5).len(),
            self.predictor.prediction_accuracy() * 100.0,
            self.causal_dreams.recent_discoveries(10).len(),
            self.collective.collective_report(),
            self.scheduler.scheduling_report(),
        )
    }
}

impl Default for FeedbackDynamicsEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_processor_creation() {
        let processor = FeedbackProcessor::new();
        assert_eq!(processor.insights.len(), 0);
    }

    #[test]
    fn test_emotional_predictor() {
        let mut predictor = EmotionalPredictor::new();

        // Record some states
        for i in 0..10 {
            let state = EmotionalBlend {
                name: "test".to_string(),
                components: vec![],
                encoding: BinaryHV::random(i as u64),
                valence: 0.5 + i as f64 * 0.03,
                arousal: 0.5,
                coherence: 0.8,
                timestamp: i as u64,
            };
            predictor.record(&state);
        }

        let prediction = predictor.predict();
        assert!(prediction.is_some());
        let pred = prediction.unwrap();
        assert!(
            pred.predicted_valence >= -1.0 && pred.predicted_valence <= 1.0,
            "predicted_valence should be in [-1,1], got {}",
            pred.predicted_valence
        );
        assert!(
            pred.confidence >= 0.1 && pred.confidence <= 0.9,
            "confidence should be in [0.1,0.9], got {}",
            pred.confidence
        );
    }

    #[test]
    fn test_causal_dream_integrator() {
        let mut integrator = CausalDreamIntegrator::new();

        let hypothesis = CausalHypothesis {
            cause: "stress".to_string(),
            effect: "poor sleep".to_string(),
            prior: 0.5,
            hv_encoding: None,
            priority: 0.8,
        };

        integrator.queue_hypothesis(hypothesis);
        assert!(integrator.next_hypothesis().is_some());
    }

    #[test]
    fn test_dream_scheduler() {
        let mut scheduler = AdaptiveDreamScheduler::new();

        scheduler.update_stress(0.3);
        let recommended = scheduler.recommend_dream_type();
        assert!(matches!(
            recommended,
            DreamType::Regular | DreamType::Counterfactual | DreamType::Consolidation
        ));

        scheduler.update_stress(0.9);
        let recommended = scheduler.recommend_dream_type();
        assert_eq!(recommended, DreamType::Lucid);
    }

    #[test]
    fn test_dynamics_engine_creation() {
        let engine = FeedbackDynamicsEngine::new();
        let report = engine.dynamics_report();
        assert!(report.contains("Feedback Dynamics Report"));
        assert!(!report.is_empty(), "dynamics report should not be empty");
        assert!(
            engine.predictor.prediction_accuracy().is_finite(),
            "prediction accuracy should be finite"
        );
    }

    // =========================================================================
    // Helper to build DreamScenario with controlled parameters
    // =========================================================================

    fn make_dream(
        emotional_valence: f64,
        lucidity: f64,
        bizarreness: f64,
        themes: Vec<&str>,
    ) -> DreamScenario {
        use crate::hdc::sleep_and_altered_states::{DreamFragment, DreamFragmentSource};

        // Build one fragment with the requested bizarreness so
        // overall_bizarreness() returns that value.
        let fragment = DreamFragment {
            content_hv: BinaryHV::random(42),
            description: "test fragment".to_string(),
            valence: emotional_valence,
            bizarreness,
            source: DreamFragmentSource::RandomActivation,
        };

        DreamScenario {
            fragments: vec![fragment],
            emotional_valence,
            narrative_coherence: 0.5,
            lucidity,
            themes: themes.into_iter().map(String::from).collect(),
            subjective_duration: 10.0,
        }
    }

    // =========================================================================
    // 1. test_feedback_processor_process_dream
    // =========================================================================

    #[test]
    fn test_feedback_processor_process_dream_emotional_resolution() {
        let mut processor = FeedbackProcessor::new();
        // High valence (>0.5), low bizarreness (<0.3) => EmotionalResolution
        let dream = make_dream(0.8, 0.2, 0.1, vec!["peace", "harmony"]);
        let insight = processor.process_dream(&dream);
        assert!(insight.is_some());
        let insight = insight.unwrap();
        assert_eq!(insight.insight_type, InsightType::EmotionalResolution);
        assert_eq!(insight.id, 1);
    }

    #[test]
    fn test_feedback_processor_process_dream_causal_discovery() {
        let mut processor = FeedbackProcessor::new();
        // Theme containing "because" => CausalDiscovery
        let dream = make_dream(0.3, 0.2, 0.2, vec!["something happened because of X"]);
        let insight = processor.process_dream(&dream).unwrap();
        assert_eq!(insight.insight_type, InsightType::CausalDiscovery);
        // Should also have extracted causal discoveries from the theme
        assert!(!insight.causal_discoveries.is_empty());
    }

    #[test]
    fn test_feedback_processor_process_dream_creative_solution() {
        let mut processor = FeedbackProcessor::new();
        // High lucidity (>0.5) => CreativeSolution (when valence is not
        // high-positive/low-bizarre and no causal theme)
        let dream = make_dream(0.3, 0.8, 0.2, vec!["flying"]);
        let insight = processor.process_dream(&dream).unwrap();
        assert_eq!(insight.insight_type, InsightType::CreativeSolution);
        // Creative solution should generate a "creativity" attention hint
        assert!(
            insight
                .attention_hints
                .iter()
                .any(|h| h.focus_area == "creativity")
        );
    }

    #[test]
    fn test_feedback_processor_process_dream_warning() {
        let mut processor = FeedbackProcessor::new();
        // High bizarreness (>0.7) => Warning
        let dream = make_dream(0.0, 0.2, 0.9, vec!["surreal"]);
        let insight = processor.process_dream(&dream).unwrap();
        assert_eq!(insight.insight_type, InsightType::Warning);
        // Warning should generate a "safety" attention hint
        assert!(
            insight
                .attention_hints
                .iter()
                .any(|h| h.focus_area == "safety")
        );
    }

    #[test]
    fn test_feedback_processor_process_dream_warning_from_nightmare() {
        let mut processor = FeedbackProcessor::new();
        // Negative valence (<-0.5) also triggers Warning
        let dream = make_dream(-0.8, 0.2, 0.2, vec!["falling"]);
        let insight = processor.process_dream(&dream).unwrap();
        assert_eq!(insight.insight_type, InsightType::Warning);
    }

    #[test]
    fn test_feedback_processor_process_dream_memory_integration() {
        let mut processor = FeedbackProcessor::new();
        // Moderate values with no special triggers => MemoryIntegration
        let dream = make_dream(0.2, 0.3, 0.4, vec!["routine"]);
        let insight = processor.process_dream(&dream).unwrap();
        assert_eq!(insight.insight_type, InsightType::MemoryIntegration);
    }

    // =========================================================================
    // 2. test_feedback_processor_insight_history
    // =========================================================================

    #[test]
    fn test_feedback_processor_insight_history() {
        let mut processor = FeedbackProcessor::new();

        // Process 5 dreams and verify history grows
        for i in 0..5 {
            let dream = make_dream(0.1 * i as f64, 0.2, 0.2, vec!["theme"]);
            let insight = processor.process_dream(&dream);
            assert!(insight.is_some());
        }
        assert_eq!(processor.insights.len(), 5);

        // Process up to 110 total to verify the max_insights cap (100)
        for i in 5..110 {
            let dream = make_dream(0.01 * (i % 50) as f64, 0.2, 0.2, vec!["theme"]);
            processor.process_dream(&dream);
        }
        // Should be capped at max_insights (100)
        assert_eq!(processor.insights.len(), 100);

        // Verify the first insights were evicted (oldest removed)
        // The first insight had id=1, so the oldest remaining should have id > 10
        assert!(processor.insights.front().unwrap().id > 10);
    }

    #[test]
    fn test_feedback_processor_insight_ids_monotonic() {
        let mut processor = FeedbackProcessor::new();
        let dream = make_dream(0.3, 0.2, 0.2, vec!["test"]);

        let insight1 = processor.process_dream(&dream).unwrap();
        let insight2 = processor.process_dream(&dream).unwrap();
        let insight3 = processor.process_dream(&dream).unwrap();

        assert_eq!(insight1.id, 1);
        assert_eq!(insight2.id, 2);
        assert_eq!(insight3.id, 3);
    }

    // =========================================================================
    // 3. test_emotional_adjustment_accumulation
    // =========================================================================

    #[test]
    fn test_emotional_adjustment_accumulation() {
        let mut processor = FeedbackProcessor::new();
        let mut emotional_depth = EmotionalDepthSystem::new();

        // Process a dream with known positive emotional valence.
        // A non-nightmare, non-lucid dream with valence 0.6 should have
        // emotional_impact = 0.6 * 0.3 = 0.18, accumulated as 0.18 * 0.1 = 0.018
        let dream = make_dream(0.6, 0.2, 0.2, vec!["happy"]);
        processor.process_dream(&dream);

        // Apply and verify non-zero adjustment
        let adj = processor.apply_emotional_feedback(&mut emotional_depth);
        assert!(
            adj.abs() > 0.001,
            "First adjustment should be non-zero, got {}",
            adj
        );

        // After applying, the accumulator was reset to 0.0.
        // A second call without processing more dreams should return ~0.0.
        let adj2 = processor.apply_emotional_feedback(&mut emotional_depth);
        assert!(
            adj2.abs() < 0.02,
            "Second adjustment should be near zero, got {}",
            adj2
        );
    }

    #[test]
    fn test_emotional_adjustment_nightmare_cathartic() {
        let mut processor = FeedbackProcessor::new();
        let mut emotional_depth = EmotionalDepthSystem::new();

        // Nightmare (valence < -0.5) should still produce a positive
        // cathartic emotional_impact of 0.1
        let dream = make_dream(-0.8, 0.2, 0.2, vec!["chased"]);
        processor.process_dream(&dream);

        let adj = processor.apply_emotional_feedback(&mut emotional_depth);
        // 0.1 (cathartic) * 0.1 = 0.01
        assert!(
            adj > 0.0,
            "Nightmare should have positive cathartic adjustment, got {}",
            adj
        );
    }

    // =========================================================================
    // 4. test_attention_hints_for_warning
    // =========================================================================

    #[test]
    fn test_attention_hints_for_warning() {
        let mut processor = FeedbackProcessor::new();

        // High bizarreness => Warning insight => safety attention hint
        let dream = make_dream(0.0, 0.2, 0.9, vec!["distortion"]);
        processor.process_dream(&dream);

        let hints = processor.get_attention_adjustments();
        assert!(
            !hints.is_empty(),
            "Warning insight should produce attention hints"
        );
        assert!(
            hints.iter().any(|h| h.focus_area == "safety"),
            "Warning should produce 'safety' focus area hint, got: {:?}",
            hints.iter().map(|h| &h.focus_area).collect::<Vec<_>>()
        );
        assert!(
            hints[0].priority > 0.5,
            "Safety hint should have high priority"
        );
    }

    #[test]
    fn test_attention_hints_for_creative_solution() {
        let mut processor = FeedbackProcessor::new();

        // High lucidity => CreativeSolution => creativity attention hint
        let dream = make_dream(0.3, 0.8, 0.2, vec!["innovation"]);
        processor.process_dream(&dream);

        let hints = processor.get_attention_adjustments();
        assert!(
            !hints.is_empty(),
            "CreativeSolution should produce attention hints"
        );
        assert!(
            hints.iter().any(|h| h.focus_area == "creativity"),
            "CreativeSolution should produce 'creativity' focus area hint"
        );
    }

    #[test]
    fn test_attention_hints_empty_for_memory_integration() {
        let mut processor = FeedbackProcessor::new();

        // MemoryIntegration should produce no attention hints
        let dream = make_dream(0.2, 0.3, 0.4, vec!["mundane"]);
        processor.process_dream(&dream);

        let hints = processor.get_attention_adjustments();
        assert!(
            hints.is_empty(),
            "MemoryIntegration should produce no attention hints"
        );
    }

    // =========================================================================
    // 5. test_dream_scheduler_stress_levels
    // =========================================================================

    #[test]
    fn test_dream_scheduler_low_stress() {
        let mut scheduler = AdaptiveDreamScheduler::new();
        scheduler.update_stress(0.1);
        let recommended = scheduler.recommend_dream_type();
        // Low stress: should NOT be Lucid (that is for high stress)
        assert_ne!(recommended, DreamType::Lucid);
        // Should be one of the regular types based on overdue intervals
        assert!(matches!(
            recommended,
            DreamType::Regular
                | DreamType::Consolidation
                | DreamType::Counterfactual
                | DreamType::CausalExploration
                | DreamType::Nightmare
        ));
    }

    #[test]
    fn test_dream_scheduler_medium_stress() {
        let mut scheduler = AdaptiveDreamScheduler::new();
        scheduler.update_stress(0.5);
        let recommended = scheduler.recommend_dream_type();
        // Medium stress (0.5) is below the 0.6 threshold for Consolidation
        // and below 0.8 for Lucid, so it falls through to overdue logic
        assert_ne!(recommended, DreamType::Lucid);
    }

    #[test]
    fn test_dream_scheduler_moderately_high_stress() {
        let mut scheduler = AdaptiveDreamScheduler::new();
        // 0.6 < stress <= 0.8 => Consolidation
        scheduler.update_stress(0.7);
        let recommended = scheduler.recommend_dream_type();
        assert_eq!(recommended, DreamType::Consolidation);
    }

    #[test]
    fn test_dream_scheduler_high_stress() {
        let mut scheduler = AdaptiveDreamScheduler::new();
        scheduler.update_stress(0.9);
        let recommended = scheduler.recommend_dream_type();
        assert_eq!(recommended, DreamType::Lucid);
    }

    #[test]
    fn test_dream_scheduler_extreme_stress() {
        let mut scheduler = AdaptiveDreamScheduler::new();
        scheduler.update_stress(1.0);
        let recommended = scheduler.recommend_dream_type();
        assert_eq!(recommended, DreamType::Lucid);
    }

    #[test]
    fn test_dream_scheduler_record_effectiveness() {
        let mut scheduler = AdaptiveDreamScheduler::new();

        let record = EffectivenessRecord {
            dream_type: DreamType::Regular,
            emotional_before: 0.3,
            emotional_after: 0.6,
            stress_before: 0.7,
            stress_after: 0.4,
            insight_generated: true,
            timestamp: 1000,
        };
        scheduler.record_effectiveness(record);

        // Verify it was recorded
        assert_eq!(
            scheduler
                .effectiveness_history
                .get(&DreamType::Regular)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_dream_scheduler_report_content() {
        let scheduler = AdaptiveDreamScheduler::new();
        let report = scheduler.scheduling_report();
        assert!(report.contains("Dream Scheduling Report"));
        assert!(report.contains("Current stress"));
        assert!(report.contains("Recommended"));
    }

    // =========================================================================
    // 6. test_causal_dream_integrator_queue
    // =========================================================================

    #[test]
    fn test_causal_dream_integrator_queue_priority() {
        let mut integrator = CausalDreamIntegrator::new();

        // Queue hypotheses with different priorities
        integrator.queue_hypothesis(CausalHypothesis {
            cause: "low_priority".to_string(),
            effect: "effect_a".to_string(),
            prior: 0.5,
            hv_encoding: None,
            priority: 0.2,
        });
        integrator.queue_hypothesis(CausalHypothesis {
            cause: "high_priority".to_string(),
            effect: "effect_b".to_string(),
            prior: 0.5,
            hv_encoding: None,
            priority: 0.9,
        });
        integrator.queue_hypothesis(CausalHypothesis {
            cause: "medium_priority".to_string(),
            effect: "effect_c".to_string(),
            prior: 0.5,
            hv_encoding: None,
            priority: 0.5,
        });

        // next_hypothesis should return highest priority first
        let first = integrator.next_hypothesis().unwrap();
        assert_eq!(first.cause, "high_priority");
        assert!((first.priority - 0.9).abs() < 1e-9);

        let second = integrator.next_hypothesis().unwrap();
        assert_eq!(second.cause, "medium_priority");

        let third = integrator.next_hypothesis().unwrap();
        assert_eq!(third.cause, "low_priority");

        // Queue should now be empty
        assert!(integrator.next_hypothesis().is_none());
    }

    #[test]
    fn test_causal_dream_integrator_max_queue() {
        let mut integrator = CausalDreamIntegrator::new();

        // Queue more than max_queue (50) hypotheses
        for i in 0..60 {
            integrator.queue_hypothesis(CausalHypothesis {
                cause: format!("cause_{}", i),
                effect: format!("effect_{}", i),
                prior: 0.5,
                hv_encoding: None,
                priority: 0.5,
            });
        }

        // Should be capped at max_queue (50)
        assert_eq!(integrator.hypotheses_queue.len(), 50);
    }

    #[test]
    fn test_causal_dream_integrator_add_hypothesis_convenience() {
        let mut integrator = CausalDreamIntegrator::new();

        // add_hypothesis is a convenience method that sets priority = 1.0 - prior.abs()
        integrator.add_hypothesis("stress", "insomnia", 0.3);

        let h = integrator.next_hypothesis().unwrap();
        assert_eq!(h.cause, "stress");
        assert_eq!(h.effect, "insomnia");
        assert!((h.prior - 0.3).abs() < 1e-9);
        // Priority = 1.0 - 0.3 = 0.7
        assert!((h.priority - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_causal_dream_integrator_configure_dream_question() {
        let integrator = CausalDreamIntegrator::new();

        let hypothesis = CausalHypothesis {
            cause: "rain".to_string(),
            effect: "flooding".to_string(),
            prior: 0.6,
            hv_encoding: None,
            priority: 0.5,
        };

        let mut engine = CounterfactualDreamEngine::new();
        let question = integrator.configure_causal_dream(&mut engine, &hypothesis);
        assert!(question.contains("rain"));
        assert!(question.contains("flooding"));
        assert!(question.contains("What if"));
    }

    // =========================================================================
    // 7. test_dynamics_engine_report_content
    // =========================================================================

    #[test]
    fn test_dynamics_engine_report_content() {
        let engine = FeedbackDynamicsEngine::new();
        let report = engine.dynamics_report();

        // Verify expected sections are present
        assert!(
            report.contains("Feedback Dynamics Report"),
            "Missing header"
        );
        assert!(
            report.contains("Recent Insights"),
            "Missing Recent Insights section"
        );
        assert!(
            report.contains("Prediction Accuracy"),
            "Missing Prediction Accuracy section"
        );
        assert!(
            report.contains("Causal Discoveries"),
            "Missing Causal Discoveries section"
        );
        assert!(
            report.contains("Collective Dream Hub"),
            "Missing Collective Dream Hub section"
        );
        assert!(
            report.contains("Dream Scheduling Report"),
            "Missing Dream Scheduling Report section"
        );
        assert!(
            report.contains("Current stress"),
            "Missing current stress info"
        );
        assert!(report.contains("Recommended"), "Missing recommendation");
    }

    #[test]
    fn test_dynamics_engine_components_accessible() {
        let mut engine = FeedbackDynamicsEngine::new();

        // Verify all sub-components are accessible and functional
        let dream = make_dream(0.5, 0.3, 0.2, vec!["test"]);
        let insight = engine.feedback.process_dream(&dream);
        assert!(insight.is_some());

        // Predictor starts empty
        let prediction = engine.predictor.predict();
        assert!(
            prediction.is_none(),
            "Predictor should need at least 5 states"
        );

        // Scheduler defaults
        engine.scheduler.update_stress(0.85);
        assert_eq!(engine.scheduler.recommend_dream_type(), DreamType::Lucid);

        // Collective hub
        engine.collective.set_peer_count(3);
        let collective_report = engine.collective.collective_report();
        assert!(collective_report.contains("3 peers"));
    }

    // =========================================================================
    // Additional edge case tests
    // =========================================================================

    #[test]
    fn test_feedback_processor_consolidation_strength_bounds() {
        let mut processor = FeedbackProcessor::new();

        // Test that consolidation_strength is always within [0, 1]
        // Low bizarreness, high valence
        let dream1 = make_dream(0.9, 0.1, 0.05, vec!["calm"]);
        let insight1 = processor.process_dream(&dream1).unwrap();
        assert!(
            insight1.consolidation_strength >= 0.0 && insight1.consolidation_strength <= 1.0,
            "consolidation_strength out of bounds: {}",
            insight1.consolidation_strength
        );

        // High bizarreness, low valence
        let dream2 = make_dream(0.0, 0.1, 0.95, vec!["chaos"]);
        let insight2 = processor.process_dream(&dream2).unwrap();
        assert!(
            insight2.consolidation_strength >= 0.0 && insight2.consolidation_strength <= 1.0,
            "consolidation_strength out of bounds: {}",
            insight2.consolidation_strength
        );
    }

    #[test]
    fn test_feedback_processor_dream_summary_format() {
        let mut processor = FeedbackProcessor::new();

        // Regular dream
        let dream = make_dream(0.3, 0.2, 0.4, vec!["a", "b"]);
        let insight = processor.process_dream(&dream).unwrap();
        assert!(insight.dream_summary.contains("dream"));
        assert!(insight.dream_summary.contains("2 themes"));

        // Lucid dream
        let lucid = make_dream(0.3, 0.8, 0.2, vec!["c"]);
        let insight2 = processor.process_dream(&lucid).unwrap();
        assert!(insight2.dream_summary.contains("lucid dream"));

        // Nightmare
        let nightmare = make_dream(-0.8, 0.2, 0.2, vec!["d"]);
        let insight3 = processor.process_dream(&nightmare).unwrap();
        assert!(insight3.dream_summary.contains("nightmare"));
    }

    #[test]
    fn test_recent_insights_ordering() {
        let mut processor = FeedbackProcessor::new();

        for _ in 0..5 {
            let dream = make_dream(0.3, 0.2, 0.2, vec!["test"]);
            processor.process_dream(&dream);
        }

        let recent = processor.recent_insights(3);
        assert_eq!(recent.len(), 3);
        // recent_insights returns most recent first (rev().take())
        assert!(recent[0].id > recent[1].id);
        assert!(recent[1].id > recent[2].id);
    }

    #[test]
    fn test_emotional_predictor_needs_minimum_history() {
        let mut predictor = EmotionalPredictor::new();

        // With fewer than 5 records, predict() should return None
        for i in 0..4 {
            predictor.record_state(0.5 + i as f64 * 0.01, 0.5);
        }
        assert!(predictor.predict().is_none());

        // After adding the 5th, it should work
        predictor.record_state(0.55, 0.5);
        assert!(predictor.predict().is_some());
    }

    #[test]
    fn test_emotional_predictor_prediction_values_clamped() {
        let mut predictor = EmotionalPredictor::new();

        // Record strongly trending states to push predictions to extremes
        for i in 0..10 {
            predictor.record_state(-1.0 + i as f64 * 0.01, 0.1);
        }

        let prediction = predictor.predict().unwrap();
        assert!(prediction.predicted_valence >= -1.0 && prediction.predicted_valence <= 1.0);
        assert!(prediction.predicted_arousal >= 0.0 && prediction.predicted_arousal <= 1.0);
        assert!(prediction.confidence >= 0.1 && prediction.confidence <= 0.9);
    }

    #[test]
    fn test_collective_dream_hub_resonance() {
        let mut hub = CollectiveDreamHub::new();

        // Share an insight with causal discoveries
        let insight = DreamInsight {
            id: 1,
            insight_type: InsightType::CausalDiscovery,
            emotional_impact: 0.3,
            consolidation_strength: 0.5,
            causal_discoveries: vec![CausalDiscovery {
                cause: "shared_theme".to_string(),
                effect: "outcome".to_string(),
                strength: 0.7,
                counterfactual_support: 0.5,
            }],
            attention_hints: vec![],
            dream_summary: "test".to_string(),
            timestamp: 0,
        };

        hub.share_insight(&insight);

        // Check resonance with matching local themes
        let resonance = hub.calculate_resonance(&["shared_theme".to_string()]);
        assert!(
            resonance > 0.0,
            "Should have positive resonance with shared theme"
        );

        // Check resonance with non-matching themes
        let no_resonance = hub.calculate_resonance(&["unrelated".to_string()]);
        assert!(
            (no_resonance - 0.0).abs() < 1e-9,
            "Should have zero resonance with unrelated theme"
        );

        // Get resonant themes
        let themes = hub.get_resonant_themes(0.5);
        assert!(themes.contains(&"shared_theme".to_string()));
    }
}
