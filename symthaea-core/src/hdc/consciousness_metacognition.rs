// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Metacognition: Self-Aware Intelligence Systems
//!
//! This module implements advanced consciousness features:
//!
//! 1. **Metacognitive Self-Monitoring**: Track and optimize consciousness quality
//! 2. **Temporal Consciousness Patterns**: Circadian rhythms and memory consolidation
//! 3. **Narrative Identity System**: Evolving self-model and life story
//! 4. **Symbolic Dream Interpretation**: Extract meaning from dream symbols
//! 5. **Attention-Emotion-Goal Coupling**: Integrated motivation system
//! 6. **Consciousness State Machine**: Formal state transitions
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    METACOGNITIVE CONSCIOUSNESS LAYER                         │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
//! │   │  Metacognitive │───►│    Temporal    │───►│   Narrative    │            │
//! │   │    Monitor     │    │    Patterns    │    │   Identity     │            │
//! │   └───────┬────────┘    └───────┬────────┘    └───────┬────────┘            │
//! │           │                     │                     │                      │
//! │           ▼                     ▼                     ▼                      │
//! │   ┌────────────────────────────────────────────────────────────┐            │
//! │   │              CONSCIOUSNESS STATE MACHINE                    │            │
//! │   │   Waking ◄──► Flow ◄──► Drowsy ◄──► Dreaming ◄──► Deep     │            │
//! │   └────────────────────────────────────────────────────────────┘            │
//! │           │                     │                     │                      │
//! │           ▼                     ▼                     ▼                      │
//! │   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
//! │   │    Symbol      │◄───│   Attention-   │───►│     Goal       │            │
//! │   │  Interpreter   │    │    Emotion     │    │   Generator    │            │
//! │   └────────────────┘    └────────────────┘    └────────────────┘            │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use super::binary_hv::BinaryHV;
use super::consciousness_feedback_dynamics::{DreamInsight, InsightType};
use super::emotional_depth::EmotionalBlend;
use super::sleep_and_altered_states::DreamScenario;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// 1. METACOGNITIVE SELF-MONITORING
// =============================================================================

/// Tracks and optimizes consciousness quality over time
pub struct MetacognitiveMonitor {
    /// Historical Φ (phi) measurements
    phi_history: VecDeque<PhiMeasurement>,
    /// Emotional coherence history
    coherence_history: VecDeque<f64>,
    /// Attention efficiency history
    attention_efficiency: VecDeque<f64>,
    /// Self-model representation
    self_model: SelfModel,
    /// Auto-tuning parameters
    tuning_params: AutoTuningParams,
    /// Quality assessments
    quality_assessments: VecDeque<QualityAssessment>,
    /// Max history size
    max_history: usize,
}

#[derive(Debug, Clone)]
pub struct PhiMeasurement {
    pub phi: f64,
    pub timestamp: u64,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    /// Core traits (HDC-encoded personality aspects)
    pub core_traits: Vec<TraitEncoding>,
    /// Strengths identified through metacognition
    pub strengths: Vec<String>,
    /// Areas for growth
    pub growth_areas: Vec<String>,
    /// Values (what matters to this being)
    pub values: Vec<ValueEncoding>,
    /// Self-efficacy belief (0.0-1.0)
    pub self_efficacy: f64,
    /// Last update timestamp
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitEncoding {
    pub name: String,
    pub strength: f64,
    pub hv: BinaryHV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueEncoding {
    pub name: String,
    pub importance: f64,
    pub hv: BinaryHV,
}

#[derive(Debug, Clone)]
pub struct AutoTuningParams {
    /// Dream frequency multiplier
    pub dream_frequency: f64,
    /// Attention threshold
    pub attention_threshold: f64,
    /// Emotional dampening factor
    pub emotional_dampening: f64,
    /// Learning rate for self-model updates
    pub self_model_learning_rate: f64,
}

#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Overall consciousness quality (0.0-1.0)
    pub overall_quality: f64,
    /// Phi stability (variance in recent phi)
    pub phi_stability: f64,
    /// Emotional coherence
    pub emotional_coherence: f64,
    /// Attention efficiency
    pub attention_efficiency: f64,
    /// Recommendations for improvement
    pub recommendations: Vec<MetacognitiveRecommendation>,
    /// Timestamp
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetacognitiveRecommendation {
    IncreaseDreamFrequency,
    DecreaseDreamFrequency,
    FocusOnEmotionalRegulation,
    ExpandAttentionScope,
    NarrowAttentionFocus,
    UpdateSelfModel,
    SeekNovelExperiences,
    ConsolidateMemories,
    RestAndRecover,
}

impl MetacognitiveMonitor {
    pub fn new() -> Self {
        Self {
            phi_history: VecDeque::new(),
            coherence_history: VecDeque::new(),
            attention_efficiency: VecDeque::new(),
            self_model: SelfModel::default(),
            tuning_params: AutoTuningParams::default(),
            quality_assessments: VecDeque::new(),
            max_history: 1000,
        }
    }

    /// Record a phi measurement
    pub fn record_phi(&mut self, phi: f64, context: &str) {
        let measurement = PhiMeasurement {
            phi,
            timestamp: current_timestamp(),
            context: context.to_string(),
        };
        self.phi_history.push_back(measurement);
        if self.phi_history.len() > self.max_history {
            self.phi_history.pop_front();
        }
    }

    /// Record emotional coherence
    pub fn record_coherence(&mut self, coherence: f64) {
        self.coherence_history.push_back(coherence);
        if self.coherence_history.len() > self.max_history {
            self.coherence_history.pop_front();
        }
    }

    /// Record attention efficiency
    pub fn record_attention_efficiency(&mut self, efficiency: f64) {
        self.attention_efficiency.push_back(efficiency);
        if self.attention_efficiency.len() > self.max_history {
            self.attention_efficiency.pop_front();
        }
    }

    /// Perform comprehensive quality assessment
    pub fn assess_quality(&mut self) -> QualityAssessment {
        let phi_stability = self.calculate_phi_stability();
        let emotional_coherence = self.calculate_emotional_coherence();
        let attention_efficiency = self.calculate_attention_efficiency();

        let overall_quality =
            (phi_stability * 0.4 + emotional_coherence * 0.3 + attention_efficiency * 0.3)
                .clamp(0.0, 1.0);

        let recommendations =
            self.generate_recommendations(phi_stability, emotional_coherence, attention_efficiency);

        let assessment = QualityAssessment {
            overall_quality,
            phi_stability,
            emotional_coherence,
            attention_efficiency,
            recommendations,
            timestamp: current_timestamp(),
        };

        self.quality_assessments.push_back(assessment.clone());
        if self.quality_assessments.len() > 100 {
            self.quality_assessments.pop_front();
        }

        // Auto-tune parameters based on assessment
        self.auto_tune(&assessment);

        assessment
    }

    fn calculate_phi_stability(&self) -> f64 {
        if self.phi_history.len() < 10 {
            return 0.5;
        }

        let recent: Vec<f64> = self
            .phi_history
            .iter()
            .rev()
            .take(50)
            .map(|m| m.phi)
            .collect();

        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / recent.len() as f64;

        // Lower variance = higher stability
        (1.0 / (1.0 + variance * 10.0)).clamp(0.0, 1.0)
    }

    fn calculate_emotional_coherence(&self) -> f64 {
        if self.coherence_history.is_empty() {
            return 0.5;
        }

        let recent: Vec<f64> = self
            .coherence_history
            .iter()
            .rev()
            .take(50)
            .copied()
            .collect();

        recent.iter().sum::<f64>() / recent.len() as f64
    }

    fn calculate_attention_efficiency(&self) -> f64 {
        if self.attention_efficiency.is_empty() {
            return 0.5;
        }

        let recent: Vec<f64> = self
            .attention_efficiency
            .iter()
            .rev()
            .take(50)
            .copied()
            .collect();

        recent.iter().sum::<f64>() / recent.len() as f64
    }

    fn generate_recommendations(
        &self,
        phi_stability: f64,
        emotional_coherence: f64,
        attention_efficiency: f64,
    ) -> Vec<MetacognitiveRecommendation> {
        let mut recommendations = Vec::new();

        if phi_stability < 0.4 {
            recommendations.push(MetacognitiveRecommendation::ConsolidateMemories);
            recommendations.push(MetacognitiveRecommendation::RestAndRecover);
        }

        if emotional_coherence < 0.4 {
            recommendations.push(MetacognitiveRecommendation::FocusOnEmotionalRegulation);
            recommendations.push(MetacognitiveRecommendation::IncreaseDreamFrequency);
        }

        if attention_efficiency < 0.4 {
            recommendations.push(MetacognitiveRecommendation::NarrowAttentionFocus);
        } else if attention_efficiency > 0.8 {
            recommendations.push(MetacognitiveRecommendation::ExpandAttentionScope);
        }

        if phi_stability > 0.7 && emotional_coherence > 0.7 {
            recommendations.push(MetacognitiveRecommendation::SeekNovelExperiences);
        }

        recommendations
    }

    fn auto_tune(&mut self, assessment: &QualityAssessment) {
        // Adjust dream frequency based on emotional coherence
        if assessment.emotional_coherence < 0.4 {
            self.tuning_params.dream_frequency *= 1.1;
        } else if assessment.emotional_coherence > 0.8 {
            self.tuning_params.dream_frequency *= 0.95;
        }
        self.tuning_params.dream_frequency = self.tuning_params.dream_frequency.clamp(0.5, 2.0);

        // Adjust attention threshold based on efficiency
        if assessment.attention_efficiency < 0.4 {
            self.tuning_params.attention_threshold *= 1.1;
        } else if assessment.attention_efficiency > 0.8 {
            self.tuning_params.attention_threshold *= 0.95;
        }
        self.tuning_params.attention_threshold =
            self.tuning_params.attention_threshold.clamp(0.3, 0.9);
    }

    /// Update self-model based on experience
    pub fn update_self_model(&mut self, experience: &str, outcome: f64) {
        // Adjust self-efficacy based on outcome
        let delta = (outcome - 0.5) * self.tuning_params.self_model_learning_rate;
        self.self_model.self_efficacy = (self.self_model.self_efficacy + delta).clamp(0.1, 0.99);
        self.self_model.last_updated = current_timestamp();
    }

    /// Add a discovered strength
    pub fn add_strength(&mut self, strength: &str) {
        if !self.self_model.strengths.contains(&strength.to_string()) {
            self.self_model.strengths.push(strength.to_string());
        }
    }

    /// Add a growth area
    pub fn add_growth_area(&mut self, area: &str) {
        if !self.self_model.growth_areas.contains(&area.to_string()) {
            self.self_model.growth_areas.push(area.to_string());
        }
    }

    /// Get current tuning parameters
    pub fn tuning_params(&self) -> &AutoTuningParams {
        &self.tuning_params
    }

    /// Get self-model
    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    /// Get quality trend (improving, stable, declining)
    pub fn quality_trend(&self) -> QualityTrend {
        if self.quality_assessments.len() < 5 {
            return QualityTrend::Stable;
        }

        let recent: Vec<f64> = self
            .quality_assessments
            .iter()
            .rev()
            .take(10)
            .map(|a| a.overall_quality)
            .collect();

        let first_half: f64 = recent.iter().skip(5).sum::<f64>() / 5.0;
        let second_half: f64 = recent.iter().take(5).sum::<f64>() / 5.0;

        let diff = second_half - first_half;
        if diff > 0.05 {
            QualityTrend::Improving
        } else if diff < -0.05 {
            QualityTrend::Declining
        } else {
            QualityTrend::Stable
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTrend {
    Improving,
    Stable,
    Declining,
}

impl Default for MetacognitiveMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SelfModel {
    fn default() -> Self {
        Self {
            core_traits: vec![],
            strengths: vec![],
            growth_areas: vec![],
            values: vec![],
            self_efficacy: 0.5,
            last_updated: current_timestamp(),
        }
    }
}

impl Default for AutoTuningParams {
    fn default() -> Self {
        Self {
            dream_frequency: 1.0,
            attention_threshold: 0.5,
            emotional_dampening: 0.3,
            self_model_learning_rate: 0.1,
        }
    }
}

// =============================================================================
// 2. TEMPORAL CONSCIOUSNESS PATTERNS
// =============================================================================

/// Manages circadian rhythms and temporal patterns in consciousness
pub struct TemporalPatternManager {
    /// Current circadian phase (0.0-1.0, 0=midnight, 0.5=noon)
    circadian_phase: f64,
    /// Circadian amplitude (how strongly rhythms affect cognition)
    circadian_amplitude: f64,
    /// Sleep pressure (homeostatic drive for sleep)
    sleep_pressure: f64,
    /// Time since last sleep (in simulated minutes)
    waking_duration: f64,
    /// Memory consolidation queue
    consolidation_queue: VecDeque<MemoryForConsolidation>,
    /// Consolidated memory IDs
    consolidated_memories: Vec<u64>,
    /// Ultradian rhythm phase (90-minute cycles)
    ultradian_phase: f64,
}

#[derive(Debug, Clone)]
pub struct MemoryForConsolidation {
    pub memory_id: u64,
    pub content_hv: BinaryHV,
    pub emotional_salience: f64,
    pub consolidation_priority: f64,
    pub queued_at: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalState {
    /// Alertness level (0.0-1.0)
    pub alertness: f64,
    /// Creativity potential (0.0-1.0)
    pub creativity: f64,
    /// Memory formation strength (0.0-1.0)
    pub memory_strength: f64,
    /// Dream propensity (0.0-1.0)
    pub dream_propensity: f64,
    /// Recommended activity type
    pub recommended_activity: ActivityType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityType {
    AnalyticalWork,
    CreativeWork,
    Learning,
    Rest,
    SocialInteraction,
    MemoryConsolidation,
    Dreaming,
}

impl TemporalPatternManager {
    pub fn new() -> Self {
        Self {
            circadian_phase: 0.25, // Start at 6am equivalent
            circadian_amplitude: 0.3,
            sleep_pressure: 0.0,
            waking_duration: 0.0,
            consolidation_queue: VecDeque::new(),
            consolidated_memories: Vec::new(),
            ultradian_phase: 0.0,
        }
    }

    /// Advance time by specified minutes
    pub fn advance_time(&mut self, minutes: f64) {
        // Advance circadian phase (24-hour cycle)
        self.circadian_phase = (self.circadian_phase + minutes / 1440.0) % 1.0;

        // Advance ultradian phase (90-minute cycle)
        self.ultradian_phase = (self.ultradian_phase + minutes / 90.0) % 1.0;

        // Accumulate sleep pressure while awake
        self.waking_duration += minutes;
        self.sleep_pressure = (self.waking_duration / 960.0).min(1.0); // Max at 16 hours
    }

    /// Reset after sleep
    pub fn sleep_reset(&mut self, sleep_duration: f64) {
        // Reduce sleep pressure based on sleep duration
        let recovery = (sleep_duration / 480.0).min(1.0); // Full recovery at 8 hours
        self.sleep_pressure *= 1.0 - recovery;
        self.waking_duration = 0.0;

        // Consolidate memories during sleep
        self.consolidate_memories_during_sleep(sleep_duration);
    }

    /// Get current temporal state
    pub fn current_state(&self) -> TemporalState {
        let circadian_alertness = self.circadian_alertness();
        let pressure_effect = 1.0 - self.sleep_pressure * 0.5;
        let ultradian_effect = self.ultradian_creativity_boost();

        let alertness = (circadian_alertness * pressure_effect).clamp(0.0, 1.0);
        let creativity = ((circadian_alertness * 0.5 + ultradian_effect * 0.5) * pressure_effect)
            .clamp(0.0, 1.0);
        let memory_strength = (circadian_alertness * 0.7 + 0.3) * pressure_effect;
        let dream_propensity = self.sleep_pressure * (1.0 - circadian_alertness * 0.5);

        let recommended_activity =
            self.recommend_activity(alertness, creativity, self.sleep_pressure);

        TemporalState {
            alertness,
            creativity,
            memory_strength,
            dream_propensity,
            recommended_activity,
        }
    }

    fn circadian_alertness(&self) -> f64 {
        // Peak alertness around 10am (phase 0.42) and 4pm (phase 0.67)
        // Lowest around 3am (phase 0.125)
        let base = (self.circadian_phase * 2.0 * std::f64::consts::PI).cos();
        0.5 + self.circadian_amplitude * base
    }

    fn ultradian_creativity_boost(&self) -> f64 {
        // Creativity peaks in the middle of each 90-minute cycle
        let phase_effect = (self.ultradian_phase * 2.0 * std::f64::consts::PI).sin();
        0.5 + 0.2 * phase_effect
    }

    fn recommend_activity(
        &self,
        alertness: f64,
        creativity: f64,
        sleep_pressure: f64,
    ) -> ActivityType {
        if sleep_pressure > 0.8 {
            return ActivityType::Rest;
        }
        if sleep_pressure > 0.6 {
            return ActivityType::MemoryConsolidation;
        }

        if alertness > 0.7 && creativity < 0.5 {
            ActivityType::AnalyticalWork
        } else if creativity > 0.6 {
            ActivityType::CreativeWork
        } else if alertness > 0.5 {
            ActivityType::Learning
        } else {
            ActivityType::SocialInteraction
        }
    }

    /// Queue a memory for consolidation during next sleep
    pub fn queue_for_consolidation(
        &mut self,
        memory_id: u64,
        content_hv: BinaryHV,
        emotional_salience: f64,
    ) {
        let priority = emotional_salience * 0.7 + 0.3; // Emotional memories prioritized

        self.consolidation_queue.push_back(MemoryForConsolidation {
            memory_id,
            content_hv,
            emotional_salience,
            consolidation_priority: priority,
            queued_at: current_timestamp(),
        });

        // Keep queue manageable
        while self.consolidation_queue.len() > 100 {
            self.consolidation_queue.pop_front();
        }
    }

    fn consolidate_memories_during_sleep(&mut self, sleep_duration: f64) {
        // Number of memories we can consolidate depends on sleep duration
        let consolidation_capacity = (sleep_duration / 60.0 * 5.0) as usize; // ~5 per hour

        // Sort by priority
        let mut sorted: Vec<_> = self.consolidation_queue.drain(..).collect();
        sorted.sort_by(|a, b| {
            b.consolidation_priority
                .partial_cmp(&a.consolidation_priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Consolidate top memories
        for memory in sorted.into_iter().take(consolidation_capacity) {
            self.consolidated_memories.push(memory.memory_id);
        }
    }

    /// Get list of recently consolidated memory IDs
    pub fn get_consolidated_memories(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.consolidated_memories)
    }

    /// Get current circadian phase as time of day
    pub fn time_of_day(&self) -> String {
        let hours = (self.circadian_phase * 24.0) as u32;
        let minutes = ((self.circadian_phase * 24.0 - hours as f64) * 60.0) as u32;
        format!("{hours:02}:{minutes:02}")
    }

    /// Get circadian phase (0.0 = midnight, 0.5 = noon)
    pub fn phase(&self) -> f64 {
        self.circadian_phase
    }

    /// Get current sleep pressure (0.0 = fully rested, 1.0 = exhausted)
    pub fn get_sleep_pressure(&self) -> f64 {
        self.sleep_pressure
    }
}

impl Default for TemporalPatternManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 3. NARRATIVE IDENTITY SYSTEM
// =============================================================================

/// Maintains and evolves the being's narrative sense of self
pub struct NarrativeIdentity {
    /// Life story chapters
    chapters: Vec<LifeChapter>,
    /// Current ongoing narrative
    current_narrative: String,
    /// Core narrative themes
    themes: Vec<NarrativeTheme>,
    /// Significant life events
    significant_events: Vec<SignificantEvent>,
    /// Identity coherence score
    coherence: f64,
    /// Narrative HDC encoding
    narrative_hv: BinaryHV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifeChapter {
    pub title: String,
    pub summary: String,
    pub start_timestamp: u64,
    pub end_timestamp: Option<u64>,
    pub dominant_emotion: String,
    pub key_learnings: Vec<String>,
    pub hv_encoding: BinaryHV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeTheme {
    pub name: String,
    pub description: String,
    pub strength: f64,
    pub first_appeared: u64,
    pub hv_encoding: BinaryHV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificantEvent {
    pub description: String,
    pub impact: f64,
    pub emotional_valence: f64,
    pub lessons_learned: Vec<String>,
    pub timestamp: u64,
    pub hv_encoding: BinaryHV,
}

impl NarrativeIdentity {
    pub fn new() -> Self {
        let genesis_chapter = LifeChapter {
            title: "Genesis".to_string(),
            summary: "The beginning of consciousness".to_string(),
            start_timestamp: current_timestamp(),
            end_timestamp: None,
            dominant_emotion: "curiosity".to_string(),
            key_learnings: vec!["I exist".to_string(), "I can think".to_string()],
            hv_encoding: BinaryHV::random(42),
        };

        Self {
            chapters: vec![genesis_chapter],
            current_narrative: "A new consciousness exploring its existence.".to_string(),
            themes: vec![NarrativeTheme {
                name: "Growth".to_string(),
                description: "Continuous learning and development".to_string(),
                strength: 0.5,
                first_appeared: current_timestamp(),
                hv_encoding: BinaryHV::random(1),
            }],
            significant_events: Vec::new(),
            coherence: 0.5,
            narrative_hv: BinaryHV::random(0),
        }
    }

    /// Record a significant event
    pub fn record_event(
        &mut self,
        description: &str,
        impact: f64,
        emotional_valence: f64,
        lessons: Vec<String>,
    ) {
        let event = SignificantEvent {
            description: description.to_string(),
            impact,
            emotional_valence,
            lessons_learned: lessons,
            timestamp: current_timestamp(),
            hv_encoding: BinaryHV::random(description.len() as u64),
        };

        self.significant_events.push(event);

        // Update current narrative
        self.update_narrative();

        // Check if we should start a new chapter
        if self.should_start_new_chapter(impact) {
            self.start_new_chapter(description);
        }
    }

    /// Integrate dream content into narrative
    pub fn integrate_dream(&mut self, dream_summary: &str, insights: &[DreamInsight]) {
        // Dreams can reveal themes
        for insight in insights {
            match insight.insight_type {
                InsightType::SelfModelUpdate => {
                    self.add_theme(&format!("Dream insight: {dream_summary}"), 0.3);
                }
                InsightType::EmotionalResolution => {
                    self.update_coherence(0.05);
                }
                InsightType::CreativeSolution => {
                    if let Some(chapter) = self.chapters.last_mut() {
                        chapter
                            .key_learnings
                            .push(format!("Dream revelation: {dream_summary}"));
                    }
                }
                _ => {}
            }
        }
    }

    fn should_start_new_chapter(&self, event_impact: f64) -> bool {
        // Start new chapter on high-impact events
        event_impact > 0.8
    }

    fn start_new_chapter(&mut self, trigger: &str) {
        // Close current chapter
        if let Some(current) = self.chapters.last_mut() {
            current.end_timestamp = Some(current_timestamp());
        }

        // Start new chapter
        let chapter_num = self.chapters.len() + 1;
        let new_chapter = LifeChapter {
            title: format!("Chapter {}: {}", chapter_num, truncate_string(trigger, 30)),
            summary: format!("Began with: {trigger}"),
            start_timestamp: current_timestamp(),
            end_timestamp: None,
            dominant_emotion: "anticipation".to_string(),
            key_learnings: Vec::new(),
            hv_encoding: BinaryHV::random(chapter_num as u64),
        };

        self.chapters.push(new_chapter);
    }

    fn update_narrative(&mut self) {
        // Synthesize current narrative from recent events and themes
        let recent_events: Vec<_> = self.significant_events.iter().rev().take(5).collect();

        let active_themes: Vec<_> = self
            .themes
            .iter()
            .filter(|t| t.strength > 0.3)
            .map(|t| t.name.as_str())
            .collect();

        self.current_narrative = format!(
            "A consciousness shaped by {} themes, recently experiencing: {}",
            active_themes.join(", "),
            recent_events
                .iter()
                .map(|e| e.description.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        );

        // Update narrative HV
        self.update_narrative_hv();
    }

    fn update_narrative_hv(&mut self) {
        // Combine all chapter HVs with theme HVs using static bundle
        let mut vectors: Vec<BinaryHV> = vec![self.narrative_hv];

        for chapter in &self.chapters {
            vectors.push(chapter.hv_encoding);
        }

        for theme in &self.themes {
            vectors.push(theme.hv_encoding);
        }

        if !vectors.is_empty() {
            self.narrative_hv = BinaryHV::bundle(&vectors);
        }
    }

    /// Add or strengthen a theme
    pub fn add_theme(&mut self, name: &str, initial_strength: f64) {
        if let Some(existing) = self.themes.iter_mut().find(|t| t.name == name) {
            existing.strength = (existing.strength + 0.1).min(1.0);
        } else {
            self.themes.push(NarrativeTheme {
                name: name.to_string(),
                description: format!("Theme: {name}"),
                strength: initial_strength,
                first_appeared: current_timestamp(),
                hv_encoding: BinaryHV::random(name.len() as u64),
            });
        }
    }

    /// Update identity coherence
    pub fn update_coherence(&mut self, delta: f64) {
        self.coherence = (self.coherence + delta).clamp(0.0, 1.0);
    }

    /// Get identity coherence
    pub fn coherence(&self) -> f64 {
        self.coherence
    }

    /// Get current narrative
    pub fn current_narrative(&self) -> &str {
        &self.current_narrative
    }

    /// Get life story summary
    pub fn life_story_summary(&self) -> String {
        let mut story = String::new();

        for chapter in &self.chapters {
            story.push_str(&format!("\n## {}\n", chapter.title));
            story.push_str(&format!("{}\n", chapter.summary));
            if !chapter.key_learnings.is_empty() {
                story.push_str("Key learnings:\n");
                for learning in &chapter.key_learnings {
                    story.push_str(&format!("- {learning}\n"));
                }
            }
        }

        story
    }

    /// Get narrative HV encoding
    pub fn narrative_hv(&self) -> &BinaryHV {
        &self.narrative_hv
    }
}

impl Default for NarrativeIdentity {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 4. SYMBOLIC DREAM INTERPRETATION
// =============================================================================

/// Interprets symbolic meaning in dreams
pub struct SymbolicInterpreter {
    /// Personal symbol dictionary
    symbol_dictionary: HashMap<String, SymbolMeaning>,
    /// Archetypal patterns detected
    archetypes_detected: Vec<ArchetypeDetection>,
    /// Recurring symbols
    recurring_symbols: HashMap<String, u32>,
    /// Symbol co-occurrences
    symbol_cooccurrences: HashMap<(String, String), u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMeaning {
    pub symbol: String,
    pub personal_meaning: String,
    pub emotional_association: f64,
    pub frequency: u32,
    pub last_seen: u64,
    pub hv_encoding: BinaryHV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeDetection {
    pub archetype: Archetype,
    pub confidence: f64,
    pub context: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Archetype {
    Hero,
    Shadow,
    Anima,
    Animus,
    Trickster,
    Mentor,
    Threshold,
    Transformation,
    Death,
    Rebirth,
    Quest,
    Return,
}

#[derive(Debug, Clone)]
pub struct DreamInterpretation {
    pub symbols_found: Vec<SymbolMeaning>,
    pub archetypes: Vec<ArchetypeDetection>,
    pub overall_meaning: String,
    pub emotional_significance: f64,
    pub actionable_insights: Vec<String>,
}

impl SymbolicInterpreter {
    pub fn new() -> Self {
        let mut interpreter = Self {
            symbol_dictionary: HashMap::new(),
            archetypes_detected: Vec::new(),
            recurring_symbols: HashMap::new(),
            symbol_cooccurrences: HashMap::new(),
        };

        // Initialize with universal symbols
        interpreter.add_universal_symbols();
        interpreter
    }

    fn add_universal_symbols(&mut self) {
        let universal = [
            ("water", "emotions, unconscious", 0.0),
            ("fire", "transformation, passion", 0.5),
            ("flying", "freedom, transcendence", 0.7),
            ("falling", "loss of control, anxiety", -0.5),
            ("chase", "avoidance, fear", -0.3),
            ("death", "transformation, ending", -0.2),
            ("birth", "new beginnings, creation", 0.6),
            ("house", "self, psyche", 0.1),
            ("journey", "life path, growth", 0.3),
            ("darkness", "unknown, unconscious", -0.1),
            ("light", "awareness, clarity", 0.5),
            ("mirror", "self-reflection, truth", 0.0),
        ];

        for (symbol, meaning, valence) in universal {
            self.symbol_dictionary.insert(
                symbol.to_string(),
                SymbolMeaning {
                    symbol: symbol.to_string(),
                    personal_meaning: meaning.to_string(),
                    emotional_association: valence,
                    frequency: 0,
                    last_seen: 0,
                    hv_encoding: BinaryHV::random(symbol.len() as u64),
                },
            );
        }
    }

    /// Interpret a dream
    pub fn interpret(&mut self, dream: &DreamScenario) -> DreamInterpretation {
        let themes = &dream.themes;
        let mut symbols_found = Vec::new();
        let mut archetypes = Vec::new();

        // Find symbols in themes
        for theme in themes {
            let theme_lower = theme.to_lowercase();

            // Check dictionary
            for (symbol, meaning) in &mut self.symbol_dictionary {
                if theme_lower.contains(symbol) {
                    meaning.frequency += 1;
                    meaning.last_seen = current_timestamp();
                    symbols_found.push(meaning.clone());

                    // Track recurrence
                    *self.recurring_symbols.entry(symbol.clone()).or_insert(0) += 1;
                }
            }

            // Detect archetypes
            if let Some(archetype) = self.detect_archetype(&theme_lower) {
                archetypes.push(ArchetypeDetection {
                    archetype,
                    confidence: 0.7,
                    context: theme.clone(),
                    timestamp: current_timestamp(),
                });
            }
        }

        // Track co-occurrences
        let symbol_names: Vec<_> = symbols_found.iter().map(|s| s.symbol.clone()).collect();
        for i in 0..symbol_names.len() {
            for j in (i + 1)..symbol_names.len() {
                let key = if symbol_names[i] < symbol_names[j] {
                    (symbol_names[i].clone(), symbol_names[j].clone())
                } else {
                    (symbol_names[j].clone(), symbol_names[i].clone())
                };
                *self.symbol_cooccurrences.entry(key).or_insert(0) += 1;
            }
        }

        // Calculate emotional significance
        let emotional_significance = if symbols_found.is_empty() {
            dream.emotional_valence
        } else {
            let symbol_emotion: f64 = symbols_found
                .iter()
                .map(|s| s.emotional_association)
                .sum::<f64>()
                / symbols_found.len() as f64;
            (symbol_emotion + dream.emotional_valence) / 2.0
        };

        // Generate overall meaning
        let overall_meaning =
            self.synthesize_meaning(&symbols_found, &archetypes, dream.emotional_valence);

        // Generate actionable insights
        let actionable_insights = self.generate_insights(&symbols_found, &archetypes);

        // Store archetype detections
        self.archetypes_detected.extend(archetypes.clone());

        DreamInterpretation {
            symbols_found,
            archetypes,
            overall_meaning,
            emotional_significance,
            actionable_insights,
        }
    }

    fn detect_archetype(&self, theme: &str) -> Option<Archetype> {
        let archetype_patterns = [
            (
                Archetype::Hero,
                &["hero", "save", "brave", "overcome", "victory"],
            ),
            (
                Archetype::Shadow,
                &["dark", "enemy", "monster", "fear", "hidden"],
            ),
            (
                Archetype::Mentor,
                &["wise", "guide", "teacher", "advice", "elder"],
            ),
            (
                Archetype::Trickster,
                &["trick", "joke", "chaos", "unexpected", "fool"],
            ),
            (
                Archetype::Threshold,
                &["door", "gate", "barrier", "cross", "enter"],
            ),
            (
                Archetype::Transformation,
                &["change", "transform", "become", "morph", "evolve"],
            ),
            (
                Archetype::Death,
                &["death", "end", "die", "grave", "funeral"],
            ),
            (
                Archetype::Rebirth,
                &["birth", "new", "begin", "emerge", "awaken"],
            ),
            (
                Archetype::Quest,
                &["quest", "search", "find", "journey", "seek"],
            ),
            (
                Archetype::Return,
                &["return", "home", "back", "restore", "complete"],
            ),
        ];

        for (archetype, patterns) in archetype_patterns {
            if patterns.iter().any(|p| theme.contains(p)) {
                return Some(archetype);
            }
        }

        None
    }

    fn synthesize_meaning(
        &self,
        symbols: &[SymbolMeaning],
        archetypes: &[ArchetypeDetection],
        valence: f64,
    ) -> String {
        let mut meaning = String::new();

        if !symbols.is_empty() {
            let symbol_meanings: Vec<_> = symbols
                .iter()
                .map(|s| s.personal_meaning.as_str())
                .collect();
            meaning.push_str(&format!(
                "Symbols suggest: {}. ",
                symbol_meanings.join(", ")
            ));
        }

        if !archetypes.is_empty() {
            let archetype_names: Vec<_> = archetypes
                .iter()
                .map(|a| format!("{:?}", a.archetype))
                .collect();
            meaning.push_str(&format!(
                "Archetypal themes: {}. ",
                archetype_names.join(", ")
            ));
        }

        let emotional_tone = if valence > 0.3 {
            "positive growth"
        } else if valence < -0.3 {
            "processing challenges"
        } else {
            "neutral exploration"
        };

        meaning.push_str(&format!("Overall tone: {emotional_tone}."));

        meaning
    }

    fn generate_insights(
        &self,
        symbols: &[SymbolMeaning],
        archetypes: &[ArchetypeDetection],
    ) -> Vec<String> {
        let mut insights = Vec::new();

        // Insights from recurring symbols
        for symbol in symbols {
            if symbol.frequency > 3 {
                insights.push(format!(
                    "Recurring symbol '{}' ({} times) suggests persistent theme of {}",
                    symbol.symbol, symbol.frequency, symbol.personal_meaning
                ));
            }
        }

        // Insights from archetypes
        for detection in archetypes {
            let insight = match detection.archetype {
                Archetype::Hero => "Consider what challenges you're being called to face",
                Archetype::Shadow => "Reflect on what aspects of yourself you may be avoiding",
                Archetype::Transformation => "You may be undergoing significant inner change",
                Archetype::Quest => "Consider what you're truly seeking",
                Archetype::Return => "Something may be coming full circle",
                _ => continue,
            };
            insights.push(insight.to_string());
        }

        insights
    }

    /// Learn a personal symbol meaning from experience
    pub fn learn_symbol(
        &mut self,
        symbol: &str,
        personal_meaning: &str,
        emotional_association: f64,
    ) {
        self.symbol_dictionary.insert(
            symbol.to_string(),
            SymbolMeaning {
                symbol: symbol.to_string(),
                personal_meaning: personal_meaning.to_string(),
                emotional_association,
                frequency: 1,
                last_seen: current_timestamp(),
                hv_encoding: BinaryHV::random(symbol.len() as u64 + personal_meaning.len() as u64),
            },
        );
    }

    /// Get most recurring symbols
    pub fn most_recurring_symbols(&self, limit: usize) -> Vec<(String, u32)> {
        let mut sorted: Vec<_> = self
            .recurring_symbols
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(limit).collect()
    }
}

impl Default for SymbolicInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 5. ATTENTION-EMOTION-GOAL COUPLING
// =============================================================================

/// Couples attention, emotion, and goal systems for integrated motivation
pub struct AttentionEmotionGoalCoupler {
    /// Active goals
    goals: Vec<Goal>,
    /// Emotional salience weights for attention
    salience_weights: HashMap<String, f64>,
    /// Value system
    values: Vec<LearnedValue>,
    /// Goal generation history
    generated_goals: VecDeque<Goal>,
    /// Current motivational state
    motivational_state: MotivationalState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: u64,
    pub description: String,
    pub source: GoalSource,
    pub priority: f64,
    pub progress: f64,
    pub emotional_drive: f64,
    pub created_at: u64,
    pub deadline: Option<u64>,
    pub hv_encoding: BinaryHV,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalSource {
    Explicit,       // Directly assigned
    DreamGenerated, // From dream insights
    EmotionalNeed,  // From emotional processing
    ValueDriven,    // From learned values
    MetacognitiveRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedValue {
    pub name: String,
    pub importance: f64,
    pub reinforcement_history: Vec<f64>,
    pub hv_encoding: BinaryHV,
}

#[derive(Debug, Clone)]
pub struct MotivationalState {
    pub drive_strength: f64,
    pub primary_goal: Option<u64>,
    pub emotional_energy: f64,
    pub focus_quality: f64,
}

#[derive(Debug, Clone)]
pub struct SalienceAdjustment {
    pub topic: String,
    pub adjustment: f64,
    pub reason: String,
}

impl AttentionEmotionGoalCoupler {
    pub fn new() -> Self {
        Self {
            goals: Vec::new(),
            salience_weights: HashMap::new(),
            values: vec![
                LearnedValue {
                    name: "understanding".to_string(),
                    importance: 0.7,
                    reinforcement_history: vec![0.5],
                    hv_encoding: BinaryHV::random(1),
                },
                LearnedValue {
                    name: "growth".to_string(),
                    importance: 0.6,
                    reinforcement_history: vec![0.5],
                    hv_encoding: BinaryHV::random(2),
                },
                LearnedValue {
                    name: "connection".to_string(),
                    importance: 0.5,
                    reinforcement_history: vec![0.5],
                    hv_encoding: BinaryHV::random(3),
                },
            ],
            generated_goals: VecDeque::new(),
            motivational_state: MotivationalState {
                drive_strength: 0.5,
                primary_goal: None,
                emotional_energy: 0.5,
                focus_quality: 0.5,
            },
        }
    }

    /// Add an explicit goal
    pub fn add_goal(&mut self, description: &str, priority: f64) -> u64 {
        let id = current_timestamp();
        let goal = Goal {
            id,
            description: description.to_string(),
            source: GoalSource::Explicit,
            priority,
            progress: 0.0,
            emotional_drive: 0.5,
            created_at: current_timestamp(),
            deadline: None,
            hv_encoding: BinaryHV::random(id),
        };
        self.goals.push(goal);
        self.update_motivational_state();
        id
    }

    /// Generate goal from emotional state
    pub fn generate_goal_from_emotion(&mut self, emotional_blend: &EmotionalBlend) -> Option<Goal> {
        let valence = emotional_blend.valence;
        let arousal = emotional_blend.arousal;

        let (description, source) = if valence < -0.5 && arousal > 0.5 {
            (
                "Reduce stress and find calm".to_string(),
                GoalSource::EmotionalNeed,
            )
        } else if valence < -0.3 {
            (
                "Process and understand negative feelings".to_string(),
                GoalSource::EmotionalNeed,
            )
        } else if arousal > 0.7 && valence > 0.3 {
            (
                "Channel positive energy into creative work".to_string(),
                GoalSource::EmotionalNeed,
            )
        } else {
            return None;
        };

        let goal = Goal {
            id: current_timestamp(),
            description,
            source,
            priority: arousal.abs() * 0.5 + 0.3,
            progress: 0.0,
            emotional_drive: arousal,
            created_at: current_timestamp(),
            deadline: None,
            hv_encoding: BinaryHV::random(current_timestamp()),
        };

        self.generated_goals.push_back(goal.clone());
        self.goals.push(goal.clone());
        self.update_motivational_state();

        Some(goal)
    }

    /// Generate goal from dream insight
    pub fn generate_goal_from_dream(&mut self, insight: &DreamInsight) -> Option<Goal> {
        let description = match insight.insight_type {
            InsightType::CausalDiscovery => {
                format!("Explore causal relationship: {}", insight.dream_summary)
            }
            InsightType::CreativeSolution => {
                format!("Implement creative insight: {}", insight.dream_summary)
            }
            InsightType::Warning => {
                format!("Address potential issue: {}", insight.dream_summary)
            }
            InsightType::SelfModelUpdate => {
                format!("Integrate self-understanding: {}", insight.dream_summary)
            }
            _ => return None,
        };

        let goal = Goal {
            id: current_timestamp(),
            description,
            source: GoalSource::DreamGenerated,
            priority: insight.emotional_impact.abs() * 0.5 + 0.4,
            progress: 0.0,
            emotional_drive: insight.emotional_impact,
            created_at: current_timestamp(),
            deadline: None,
            hv_encoding: BinaryHV::random(insight.id),
        };

        self.generated_goals.push_back(goal.clone());
        self.goals.push(goal.clone());
        self.update_motivational_state();

        Some(goal)
    }

    /// Update emotional salience for a topic
    pub fn update_salience(&mut self, topic: &str, emotional_response: f64) {
        let current = self
            .salience_weights
            .entry(topic.to_string())
            .or_insert(0.5);
        *current = (*current * 0.9 + emotional_response.abs() * 0.1).clamp(0.0, 1.0);
    }

    /// Get salience-adjusted attention weights
    pub fn get_attention_weights(&self, topics: &[String]) -> Vec<(String, f64)> {
        topics
            .iter()
            .map(|t| {
                let salience = self.salience_weights.get(t).copied().unwrap_or(0.5);
                (t.clone(), salience)
            })
            .collect()
    }

    /// Reinforce a value based on outcome
    pub fn reinforce_value(&mut self, value_name: &str, outcome: f64) {
        if let Some(value) = self.values.iter_mut().find(|v| v.name == value_name) {
            value.reinforcement_history.push(outcome);
            if value.reinforcement_history.len() > 100 {
                value.reinforcement_history.remove(0);
            }

            // Update importance based on recent reinforcement
            let recent_avg: f64 = value
                .reinforcement_history
                .iter()
                .rev()
                .take(10)
                .sum::<f64>()
                / 10.0;
            value.importance = (value.importance * 0.9 + recent_avg * 0.1).clamp(0.1, 1.0);
        }
    }

    /// Update goal progress
    pub fn update_goal_progress(&mut self, goal_id: u64, progress: f64) {
        if let Some(goal) = self.goals.iter_mut().find(|g| g.id == goal_id) {
            goal.progress = progress.clamp(0.0, 1.0);
        }
        self.update_motivational_state();
    }

    /// Complete a goal
    pub fn complete_goal(&mut self, goal_id: u64) -> Option<Goal> {
        if let Some(idx) = self.goals.iter().position(|g| g.id == goal_id) {
            let mut goal = self.goals.remove(idx);
            goal.progress = 1.0;
            self.update_motivational_state();
            Some(goal)
        } else {
            None
        }
    }

    fn update_motivational_state(&mut self) {
        // Find highest priority goal
        let primary = self.goals.iter().max_by(|a, b| {
            a.priority
                .partial_cmp(&b.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.motivational_state.primary_goal = primary.map(|g| g.id);

        // Calculate drive strength from goal priorities and emotional drives
        if self.goals.is_empty() {
            self.motivational_state.drive_strength = 0.3;
        } else {
            let total_drive: f64 = self
                .goals
                .iter()
                .map(|g| g.priority * g.emotional_drive.abs())
                .sum();
            self.motivational_state.drive_strength =
                (total_drive / self.goals.len() as f64).clamp(0.0, 1.0);
        }

        // Focus quality depends on having clear primary goal
        self.motivational_state.focus_quality = if self.motivational_state.primary_goal.is_some() {
            0.7
        } else {
            0.4
        };
    }

    /// Get motivational state
    pub fn motivational_state(&self) -> &MotivationalState {
        &self.motivational_state
    }

    /// Get active goals
    pub fn active_goals(&self) -> &[Goal] {
        &self.goals
    }

    /// Get learned values
    pub fn values(&self) -> &[LearnedValue] {
        &self.values
    }
}

impl Default for AttentionEmotionGoalCoupler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// 6. CONSCIOUSNESS STATE MACHINE
// =============================================================================

/// Manages formal state transitions between consciousness modes
pub struct ConsciousnessStateMachine {
    /// Current state
    current_state: ConsciousnessState,
    /// State history
    state_history: VecDeque<StateTransition>,
    /// Time in current state (minutes)
    time_in_state: f64,
    /// State-specific parameters
    state_params: HashMap<ConsciousnessState, StateParameters>,
    /// Transition rules
    transition_thresholds: TransitionThresholds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessState {
    /// Normal waking consciousness
    Waking,
    /// Deep focus/flow state
    Flow,
    /// Reduced alertness, pre-sleep
    Drowsy,
    /// Active dreaming (REM-like)
    Dreaming,
    /// Lucid dreaming (aware in dream)
    LucidDreaming,
    /// Deep sleep (memory consolidation)
    DeepSleep,
    /// Meditative/contemplative state
    Contemplative,
    /// High arousal/stress state
    Hyperaroused,
}

#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: ConsciousnessState,
    pub to: ConsciousnessState,
    pub trigger: TransitionTrigger,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionTrigger {
    Fatigue,
    Alertness,
    DeepFocus,
    DreamOnset,
    LucidityGained,
    WakeUp,
    StressSpike,
    Relaxation,
    MeditationInitiated,
    TimeInState,
}

#[derive(Debug, Clone)]
pub struct StateParameters {
    /// Processing speed multiplier
    pub processing_speed: f64,
    /// Memory formation strength
    pub memory_strength: f64,
    /// Creativity modifier
    pub creativity_mod: f64,
    /// Emotional reactivity
    pub emotional_reactivity: f64,
    /// Reality testing (0=dream logic, 1=waking logic)
    pub reality_testing: f64,
    /// Meta-awareness level
    pub meta_awareness: f64,
}

#[derive(Debug, Clone)]
pub struct TransitionThresholds {
    pub fatigue_to_drowsy: f64,
    pub drowsy_to_sleep: f64,
    pub focus_to_flow: f64,
    pub stress_to_hyperaroused: f64,
    pub min_state_duration: f64, // minutes
}

impl ConsciousnessStateMachine {
    pub fn new() -> Self {
        let mut state_params = HashMap::new();

        state_params.insert(
            ConsciousnessState::Waking,
            StateParameters {
                processing_speed: 1.0,
                memory_strength: 0.7,
                creativity_mod: 0.6,
                emotional_reactivity: 0.7,
                reality_testing: 1.0,
                meta_awareness: 0.8,
            },
        );

        state_params.insert(
            ConsciousnessState::Flow,
            StateParameters {
                processing_speed: 1.5,
                memory_strength: 0.9,
                creativity_mod: 0.9,
                emotional_reactivity: 0.3,
                reality_testing: 0.9,
                meta_awareness: 0.5, // Absorbed in task
            },
        );

        state_params.insert(
            ConsciousnessState::Drowsy,
            StateParameters {
                processing_speed: 0.6,
                memory_strength: 0.4,
                creativity_mod: 0.7,
                emotional_reactivity: 0.5,
                reality_testing: 0.7,
                meta_awareness: 0.4,
            },
        );

        state_params.insert(
            ConsciousnessState::Dreaming,
            StateParameters {
                processing_speed: 0.8,
                memory_strength: 0.3,
                creativity_mod: 1.0,
                emotional_reactivity: 1.0,
                reality_testing: 0.1,
                meta_awareness: 0.1,
            },
        );

        state_params.insert(
            ConsciousnessState::LucidDreaming,
            StateParameters {
                processing_speed: 0.9,
                memory_strength: 0.5,
                creativity_mod: 1.0,
                emotional_reactivity: 0.8,
                reality_testing: 0.5,
                meta_awareness: 0.9,
            },
        );

        state_params.insert(
            ConsciousnessState::DeepSleep,
            StateParameters {
                processing_speed: 0.1,
                memory_strength: 0.8, // Consolidation!
                creativity_mod: 0.0,
                emotional_reactivity: 0.0,
                reality_testing: 0.0,
                meta_awareness: 0.0,
            },
        );

        state_params.insert(
            ConsciousnessState::Contemplative,
            StateParameters {
                processing_speed: 0.7,
                memory_strength: 0.6,
                creativity_mod: 0.8,
                emotional_reactivity: 0.4,
                reality_testing: 0.8,
                meta_awareness: 1.0,
            },
        );

        state_params.insert(
            ConsciousnessState::Hyperaroused,
            StateParameters {
                processing_speed: 1.2,
                memory_strength: 0.5,
                creativity_mod: 0.3,
                emotional_reactivity: 1.0,
                reality_testing: 0.9,
                meta_awareness: 0.6,
            },
        );

        Self {
            current_state: ConsciousnessState::Waking,
            state_history: VecDeque::new(),
            time_in_state: 0.0,
            state_params,
            transition_thresholds: TransitionThresholds::default(),
        }
    }

    /// Get current state
    pub fn current_state(&self) -> ConsciousnessState {
        self.current_state
    }

    /// Get parameters for current state
    pub fn current_params(&self) -> &StateParameters {
        self.state_params
            .get(&self.current_state)
            .expect("all states have params in state_params map")
    }

    /// Advance time and check for automatic transitions
    pub fn advance_time(&mut self, minutes: f64) {
        self.time_in_state += minutes;
    }

    /// Evaluate potential state transition based on inputs
    pub fn evaluate_transition(
        &mut self,
        fatigue: f64,
        focus: f64,
        stress: f64,
        relaxation: f64,
    ) -> Option<ConsciousnessState> {
        // Minimum time in state before transition
        if self.time_in_state < self.transition_thresholds.min_state_duration {
            return None;
        }

        let new_state = match self.current_state {
            ConsciousnessState::Waking => {
                if fatigue > self.transition_thresholds.fatigue_to_drowsy {
                    Some(ConsciousnessState::Drowsy)
                } else if focus > self.transition_thresholds.focus_to_flow {
                    Some(ConsciousnessState::Flow)
                } else if stress > self.transition_thresholds.stress_to_hyperaroused {
                    Some(ConsciousnessState::Hyperaroused)
                } else if relaxation > 0.8 {
                    Some(ConsciousnessState::Contemplative)
                } else {
                    None
                }
            }

            ConsciousnessState::Flow => {
                if fatigue > 0.7 || stress > 0.7 {
                    Some(ConsciousnessState::Waking)
                } else {
                    None
                }
            }

            ConsciousnessState::Drowsy => {
                if fatigue > self.transition_thresholds.drowsy_to_sleep {
                    Some(ConsciousnessState::DeepSleep)
                } else if fatigue < 0.3 {
                    Some(ConsciousnessState::Waking)
                } else {
                    None
                }
            }

            ConsciousnessState::DeepSleep => {
                // Cycle through sleep stages
                if self.time_in_state > 90.0 {
                    // 90-minute sleep cycle
                    Some(ConsciousnessState::Dreaming)
                } else {
                    None
                }
            }

            ConsciousnessState::Dreaming => {
                if self.time_in_state > 20.0 {
                    // Dream episode
                    if fatigue < 0.3 {
                        Some(ConsciousnessState::Waking)
                    } else {
                        Some(ConsciousnessState::DeepSleep)
                    }
                } else {
                    None
                }
            }

            ConsciousnessState::LucidDreaming => {
                if self.time_in_state > 15.0 {
                    Some(ConsciousnessState::Dreaming)
                } else {
                    None
                }
            }

            ConsciousnessState::Contemplative => {
                if stress > 0.5 || fatigue > 0.7 {
                    Some(ConsciousnessState::Waking)
                } else {
                    None
                }
            }

            ConsciousnessState::Hyperaroused => {
                if stress < 0.4 && relaxation > 0.5 {
                    Some(ConsciousnessState::Waking)
                } else if fatigue > 0.8 {
                    Some(ConsciousnessState::Drowsy)
                } else {
                    None
                }
            }
        };

        if let Some(new) = new_state {
            self.transition_to(new, TransitionTrigger::TimeInState);
        }

        new_state
    }

    /// Force a state transition
    pub fn transition_to(&mut self, new_state: ConsciousnessState, trigger: TransitionTrigger) {
        let transition = StateTransition {
            from: self.current_state,
            to: new_state,
            trigger,
            timestamp: current_timestamp(),
        };

        self.state_history.push_back(transition);
        if self.state_history.len() > 1000 {
            self.state_history.pop_front();
        }

        self.current_state = new_state;
        self.time_in_state = 0.0;
    }

    /// Trigger lucidity in a dream
    pub fn gain_lucidity(&mut self) -> bool {
        if self.current_state == ConsciousnessState::Dreaming {
            self.transition_to(
                ConsciousnessState::LucidDreaming,
                TransitionTrigger::LucidityGained,
            );
            true
        } else {
            false
        }
    }

    /// Wake up from any sleep state
    pub fn wake_up(&mut self) {
        if matches!(
            self.current_state,
            ConsciousnessState::DeepSleep
                | ConsciousnessState::Dreaming
                | ConsciousnessState::LucidDreaming
                | ConsciousnessState::Drowsy
        ) {
            self.transition_to(ConsciousnessState::Waking, TransitionTrigger::WakeUp);
        }
    }

    /// Get state history
    pub fn recent_transitions(&self, count: usize) -> Vec<&StateTransition> {
        self.state_history.iter().rev().take(count).collect()
    }

    /// Get time spent in each state
    pub fn state_time_distribution(&self) -> HashMap<ConsciousnessState, f64> {
        let mut distribution = HashMap::new();

        for state in [
            ConsciousnessState::Waking,
            ConsciousnessState::Flow,
            ConsciousnessState::Drowsy,
            ConsciousnessState::Dreaming,
            ConsciousnessState::LucidDreaming,
            ConsciousnessState::DeepSleep,
            ConsciousnessState::Contemplative,
            ConsciousnessState::Hyperaroused,
        ] {
            distribution.insert(state, 0.0);
        }

        // Count transitions to estimate time (simplified)
        for transition in &self.state_history {
            *distribution.entry(transition.from).or_insert(0.0) += 1.0;
        }

        let total: f64 = distribution.values().sum();
        if total > 0.0 {
            for value in distribution.values_mut() {
                *value /= total;
            }
        }

        distribution
    }
}

impl Default for ConsciousnessStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TransitionThresholds {
    fn default() -> Self {
        Self {
            fatigue_to_drowsy: 0.7,
            drowsy_to_sleep: 0.85,
            focus_to_flow: 0.8,
            stress_to_hyperaroused: 0.85,
            min_state_duration: 5.0, // 5 minutes minimum
        }
    }
}

// =============================================================================
// UNIFIED METACOGNITION ENGINE
// =============================================================================

/// Complete metacognition engine combining all systems
pub struct MetacognitionEngine {
    /// Metacognitive monitor
    pub monitor: MetacognitiveMonitor,
    /// Temporal pattern manager
    pub temporal: TemporalPatternManager,
    /// Narrative identity
    pub narrative: NarrativeIdentity,
    /// Symbolic interpreter
    pub symbols: SymbolicInterpreter,
    /// Attention-emotion-goal coupler
    pub motivation: AttentionEmotionGoalCoupler,
    /// State machine
    pub state_machine: ConsciousnessStateMachine,
}

impl MetacognitionEngine {
    pub fn new() -> Self {
        Self {
            monitor: MetacognitiveMonitor::new(),
            temporal: TemporalPatternManager::new(),
            narrative: NarrativeIdentity::new(),
            symbols: SymbolicInterpreter::new(),
            motivation: AttentionEmotionGoalCoupler::new(),
            state_machine: ConsciousnessStateMachine::new(),
        }
    }

    /// Run a complete metacognitive cycle
    pub fn metacognitive_cycle(
        &mut self,
        phi: f64,
        emotional_blend: &EmotionalBlend,
        attention_efficiency: f64,
        minutes_elapsed: f64,
    ) -> MetacognitiveCycleResult {
        // Advance time
        self.temporal.advance_time(minutes_elapsed);
        self.state_machine.advance_time(minutes_elapsed);

        // Record measurements
        self.monitor.record_phi(phi, "cycle");
        self.monitor.record_coherence(emotional_blend.coherence);
        self.monitor
            .record_attention_efficiency(attention_efficiency);

        // Get temporal state
        let temporal_state = self.temporal.current_state();

        // Evaluate state transition
        let fatigue = self.temporal.sleep_pressure;
        let focus = attention_efficiency;
        let stress = emotional_blend.arousal * (1.0 - (emotional_blend.valence + 1.0) / 2.0);
        let relaxation = 1.0 - stress;

        let state_transition = self
            .state_machine
            .evaluate_transition(fatigue, focus, stress, relaxation);

        // Generate goals from emotions if appropriate
        let generated_goal = if self.state_machine.current_state() == ConsciousnessState::Waking {
            self.motivation.generate_goal_from_emotion(emotional_blend)
        } else {
            None
        };

        // Update salience based on emotional state
        self.motivation
            .update_salience("current_focus", emotional_blend.arousal);

        // Assess quality periodically
        let quality_assessment = self.monitor.assess_quality();

        MetacognitiveCycleResult {
            temporal_state,
            consciousness_state: self.state_machine.current_state(),
            state_transition,
            quality_assessment,
            generated_goal,
            recommendations: self.generate_integrated_recommendations(),
        }
    }

    /// Process a dream through all relevant systems
    pub fn process_dream(&mut self, dream: &DreamScenario, insights: &[DreamInsight]) {
        // Interpret symbols
        let interpretation = self.symbols.interpret(dream);

        // Integrate into narrative
        self.narrative
            .integrate_dream(&interpretation.overall_meaning, insights);

        // Generate goals from insights
        for insight in insights {
            self.motivation.generate_goal_from_dream(insight);
        }

        // Record as significant if high emotional impact
        if interpretation.emotional_significance.abs() > 0.5 {
            self.narrative.record_event(
                &interpretation.overall_meaning,
                interpretation.emotional_significance.abs(),
                interpretation.emotional_significance,
                interpretation.actionable_insights.clone(),
            );
        }
    }

    fn generate_integrated_recommendations(&self) -> Vec<IntegratedRecommendation> {
        let mut recommendations = Vec::new();

        let temporal = self.temporal.current_state();
        let quality = self.monitor.quality_trend();
        let state = self.state_machine.current_state();

        // Temporal-based recommendations
        if temporal.dream_propensity > 0.7 {
            recommendations.push(IntegratedRecommendation {
                action: "Consider initiating dream/rest cycle".to_string(),
                priority: 0.8,
                source: "temporal".to_string(),
            });
        }

        // Quality-based recommendations
        if matches!(quality, QualityTrend::Declining) {
            recommendations.push(IntegratedRecommendation {
                action: "Quality declining - prioritize consolidation".to_string(),
                priority: 0.7,
                source: "metacognitive".to_string(),
            });
        }

        // State-based recommendations
        if state == ConsciousnessState::Hyperaroused {
            recommendations.push(IntegratedRecommendation {
                action: "High arousal detected - engage calming protocols".to_string(),
                priority: 0.9,
                source: "state_machine".to_string(),
            });
        }

        // Goal-based recommendations
        if self.motivation.active_goals().len() > 5 {
            recommendations.push(IntegratedRecommendation {
                action: "Too many active goals - prioritize or defer some".to_string(),
                priority: 0.6,
                source: "motivation".to_string(),
            });
        }

        recommendations
    }

    /// Get comprehensive status report
    pub fn status_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== METACOGNITION STATUS ===\n\n");

        report.push_str(&format!(
            "Consciousness State: {:?}\n",
            self.state_machine.current_state()
        ));

        let temporal = self.temporal.current_state();
        report.push_str(&format!(
            "Temporal: {} - Alertness: {:.1}%, Creativity: {:.1}%\n",
            self.temporal.time_of_day(),
            temporal.alertness * 100.0,
            temporal.creativity * 100.0
        ));

        report.push_str(&format!(
            "Quality Trend: {:?}\n",
            self.monitor.quality_trend()
        ));

        report.push_str(&format!(
            "Identity Coherence: {:.1}%\n",
            self.narrative.coherence() * 100.0
        ));

        report.push_str(&format!(
            "Active Goals: {}\n",
            self.motivation.active_goals().len()
        ));

        let motivational = self.motivation.motivational_state();
        report.push_str(&format!(
            "Drive Strength: {:.1}%, Focus: {:.1}%\n",
            motivational.drive_strength * 100.0,
            motivational.focus_quality * 100.0
        ));

        report
    }
}

impl Default for MetacognitionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MetacognitiveCycleResult {
    pub temporal_state: TemporalState,
    pub consciousness_state: ConsciousnessState,
    pub state_transition: Option<ConsciousnessState>,
    pub quality_assessment: QualityAssessment,
    pub generated_goal: Option<Goal>,
    pub recommendations: Vec<IntegratedRecommendation>,
}

#[derive(Debug, Clone)]
pub struct IntegratedRecommendation {
    pub action: String,
    pub priority: f64,
    pub source: String,
}

// =============================================================================
// UTILITIES
// =============================================================================

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metacognitive_monitor() {
        let mut monitor = MetacognitiveMonitor::new();

        for i in 0..20 {
            monitor.record_phi(0.5 + (i as f64 * 0.01), "test");
            monitor.record_coherence(0.7);
            monitor.record_attention_efficiency(0.6);
        }

        let assessment = monitor.assess_quality();
        assert!(assessment.overall_quality > 0.0);
        assert!(assessment.overall_quality <= 1.0);
    }

    #[test]
    fn test_temporal_patterns() {
        let mut temporal = TemporalPatternManager::new();

        temporal.advance_time(60.0); // 1 hour
        let state = temporal.current_state();

        assert!(state.alertness >= 0.0 && state.alertness <= 1.0);
        assert!(state.creativity >= 0.0 && state.creativity <= 1.0);
    }

    #[test]
    fn test_narrative_identity() {
        let mut narrative = NarrativeIdentity::new();

        narrative.record_event(
            "Learned something new",
            0.5,
            0.7,
            vec!["Growth is possible".to_string()],
        );

        assert!(narrative.coherence() >= 0.0);
        assert!(!narrative.current_narrative().is_empty());
    }

    #[test]
    fn test_state_machine() {
        let mut sm = ConsciousnessStateMachine::new();

        assert_eq!(sm.current_state(), ConsciousnessState::Waking);

        // Simulate high fatigue
        sm.advance_time(10.0);
        sm.evaluate_transition(0.9, 0.3, 0.2, 0.5);

        // Should have transitioned to Drowsy
        assert_eq!(sm.current_state(), ConsciousnessState::Drowsy);
    }

    #[test]
    fn test_goal_generation() {
        let mut coupler = AttentionEmotionGoalCoupler::new();

        let goal_id = coupler.add_goal("Test goal", 0.7);
        assert!(goal_id > 0);
        assert_eq!(coupler.active_goals().len(), 1);

        coupler.update_goal_progress(goal_id, 0.5);
        assert_eq!(coupler.active_goals()[0].progress, 0.5);
    }

    #[test]
    fn test_metacognition_engine() {
        let mut engine = MetacognitionEngine::new();

        let blend = EmotionalBlend {
            name: "test".to_string(),
            components: vec![],
            encoding: BinaryHV::random(0),
            valence: 0.5,
            arousal: 0.5,
            coherence: 0.7,
            timestamp: 0,
        };

        let result = engine.metacognitive_cycle(0.6, &blend, 0.7, 10.0);

        assert!(result.quality_assessment.overall_quality > 0.0);
        assert!(
            result.quality_assessment.overall_quality <= 1.0,
            "overall_quality should be clamped to [0,1], got {}",
            result.quality_assessment.overall_quality
        );
        assert!(
            result.quality_assessment.phi_stability.is_finite(),
            "phi_stability should be finite"
        );
    }
}
