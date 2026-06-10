// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Extended consciousness subsystem integrations.
//!
//! Delegation methods for emotional depth, cross-modal attention routing,
//! self-improvement, counterfactual dreams, feedback dynamics, and metacognition.

use super::super::adaptive_topology::CognitiveMode;
use super::super::binary_hv::BinaryHV;
use super::super::consciousness_feedback_dynamics::{
    AttentionHint, CausalDiscovery, DreamInsight, DreamType, EmotionalPrediction,
    ProactiveIntervention,
};
use super::super::consciousness_metacognition::{
    ConsciousnessState, Goal, MetacognitiveCycleResult, QualityTrend,
};
use super::super::counterfactual_dreams::{CounterfactualDreamScenario, DreamResolution};
use super::super::cross_modal_attention_router::{ModalityInput, RoutingResult};
use super::super::cross_modal_binding::Modality;
use super::super::emotional_depth::{CompoundEmotion, EmotionalBlend, WeightedComponent};
use super::super::full_stack_consciousness::ConsciousComprehension;
use super::super::self_improvement_integration::{
    CognitiveSnapshot, ImprovementRecommendation, ImprovementType,
};
use super::being::UnifiedConsciousBeing;

impl UnifiedConsciousBeing {
    // =========================================================================
    // EMOTIONAL DEPTH SYSTEM
    // =========================================================================

    /// Feel a compound emotion (predefined complex blend)
    ///
    /// # Example
    /// ```ignore
    /// being.feel_compound(CompoundEmotion::Nostalgia, Some("Looking at old photographs"));
    /// being.feel_compound(CompoundEmotion::Awe, Some("Witnessing the stars"));
    /// ```
    pub fn feel_compound(&mut self, compound: CompoundEmotion, trigger: Option<&str>) {
        self.emotional_depth
            .feel_compound(compound, trigger.map(String::from));
    }

    /// Feel a custom emotional blend
    ///
    /// Create custom emotional states by blending weighted components.
    ///
    /// # Example
    /// ```ignore
    /// let components = vec![
    ///     WeightedComponent::new(EmotionalComponent::Joy, 0.6),
    ///     WeightedComponent::new(EmotionalComponent::Curiosity, 0.4),
    ///     WeightedComponent::new(EmotionalComponent::Anticipation, 0.3),
    /// ];
    /// being.feel_custom("Eager Discovery", components, Some("New project starting"));
    /// ```
    pub fn feel_custom(
        &mut self,
        name: &str,
        components: Vec<WeightedComponent>,
        trigger: Option<&str>,
    ) {
        self.emotional_depth
            .feel_blend(name, components, trigger.map(String::from));
    }

    /// Blend current emotional state with new emotions
    ///
    /// Useful for gradual emotional transitions rather than abrupt changes.
    ///
    /// # Arguments
    /// * `additional` - New emotional components to blend in
    /// * `blend_ratio` - 0.0 = keep current, 1.0 = fully replace with new
    pub fn blend_emotion(&mut self, additional: Vec<WeightedComponent>, blend_ratio: f64) {
        self.emotional_depth.blend_with(additional, blend_ratio);
    }

    /// Get current emotional blend
    pub fn current_emotion(&self) -> &EmotionalBlend {
        self.emotional_depth.current()
    }

    /// Get emotional valence trend (are emotions improving or declining?)
    ///
    /// Returns value from -1.0 (declining) to 1.0 (improving)
    pub fn emotional_trend(&self) -> f64 {
        self.emotional_depth.trend()
    }

    /// Get emotional volatility (how much are emotions fluctuating?)
    ///
    /// High volatility indicates emotional instability.
    pub fn emotional_volatility(&self) -> f64 {
        self.emotional_depth.volatility()
    }

    /// Check similarity between current state and a compound emotion
    ///
    /// Useful for emotion recognition: "How close am I to feeling nostalgia?"
    pub fn emotion_similarity(&self, compound: CompoundEmotion) -> f64 {
        self.emotional_depth.similarity_to(compound)
    }

    /// Get HDC encoding of current emotional trajectory
    ///
    /// This vector encodes the recent emotional history, useful for:
    /// - Emotional memory retrieval
    /// - Mood-based content generation
    /// - Cross-agent emotional synchronization
    pub fn emotional_trajectory_hv(&self) -> BinaryHV {
        self.emotional_depth.trajectory_encoding()
    }

    /// Get comprehensive emotional report
    pub fn emotional_report(&self) -> String {
        self.emotional_depth.report()
    }

    /// Detect emotional response to comprehension
    ///
    /// Based on the comprehension result, infer an appropriate emotional response.
    pub fn emotional_response_to(&mut self, comprehension: &ConsciousComprehension) {
        // Use valence from understanding to guide emotional response
        let valence = comprehension
            .understanding
            .speaker_model
            .emotional_state
            .valence;
        let confidence = comprehension.consciousness_phi;

        // High phi + positive valence = awe/joy
        // High phi + negative valence = thoughtful sadness (melancholy)
        // Low phi + positive = contentment
        // Low phi + negative = anxiety/dread

        let compound = match (valence, confidence) {
            (v, phi) if v > 0.3 && phi > 0.7 => CompoundEmotion::Awe,
            (v, phi) if v > 0.3 && phi > 0.4 => CompoundEmotion::Delight,
            (v, _) if v > 0.3 => CompoundEmotion::Serenity,
            (v, phi) if v < -0.3 && phi > 0.6 => CompoundEmotion::Melancholy,
            (v, phi) if v < -0.3 && phi > 0.3 => CompoundEmotion::Grief,
            (v, _) if v < -0.3 => CompoundEmotion::Dread,
            (v, _) if v > 0.0 => CompoundEmotion::Serenity,
            _ => CompoundEmotion::Nostalgia, // Default for neutral/mixed
        };

        self.feel_compound(compound, Some("response to comprehension"));
    }

    // =========================================================================
    // CROSS-MODAL ATTENTION ROUTING
    // =========================================================================

    /// Route attention across multiple modalities
    ///
    /// Produces a unified representation weighted by salience and Φ level.
    ///
    /// # Example
    /// ```ignore
    /// let inputs = vec![
    ///     ModalityInput::new(Modality::Visual, visual_hv, 0.8),
    ///     ModalityInput::new(Modality::Auditory, audio_hv, 0.5),
    ///     ModalityInput::new(Modality::Semantic, text_hv, 0.7),
    /// ];
    /// let result = being.route_attention(&inputs);
    /// ```
    pub fn route_attention(&mut self, inputs: &[ModalityInput]) -> RoutingResult {
        let phi = self.stats.avg_phi;
        self.attention_router.route(inputs, phi)
    }

    /// Route attention with explicit Φ level
    ///
    /// Use this when you want to override the internal Φ measurement.
    pub fn route_attention_with_phi(
        &mut self,
        inputs: &[ModalityInput],
        phi: f64,
    ) -> RoutingResult {
        self.attention_router.route(inputs, phi)
    }

    /// Set attention routing context
    ///
    /// The context vector biases attention toward modalities with similar representations.
    pub fn set_attention_context(&mut self, context: BinaryHV) {
        self.attention_router.set_context(context);
    }

    /// Set attention routing goal
    ///
    /// The goal vector biases attention toward goal-relevant modalities.
    pub fn set_attention_goal(&mut self, goal: BinaryHV) {
        self.attention_router.set_goal(goal);
    }

    /// Check if attention is stable (same dominant modality for N steps)
    pub fn is_attention_stable(&self, n: usize) -> bool {
        self.attention_router.is_attention_stable(n)
    }

    /// Get the sequence of dominant modalities over time
    pub fn dominant_modality_sequence(&self) -> Vec<Modality> {
        self.attention_router.dominant_modality_sequence()
    }

    /// Create a multi-modal unified representation from comprehension
    ///
    /// This is the high-level integration of cross-modal routing with comprehension.
    pub fn multi_modal_comprehend(
        &mut self,
        semantic_hv: BinaryHV,
        emotional_hv: Option<BinaryHV>,
        temporal_hv: Option<BinaryHV>,
    ) -> RoutingResult {
        let mut inputs = vec![ModalityInput::new(Modality::Semantic, semantic_hv, 0.8)];

        if let Some(emo) = emotional_hv {
            inputs.push(
                ModalityInput::new(Modality::Emotional, emo, 0.6).with_label("emotional_context"),
            );
        }

        if let Some(temp) = temporal_hv {
            inputs.push(
                ModalityInput::new(Modality::Temporal, temp, 0.5).with_label("temporal_context"),
            );
        }

        self.route_attention(&inputs)
    }

    /// Reset attention routing state
    pub fn reset_attention(&mut self) {
        self.attention_router.reset();
    }

    // =========================================================================
    // SELF-IMPROVEMENT SYSTEM
    // =========================================================================

    /// Observe current cognitive state for self-improvement
    ///
    /// Call this periodically (e.g., after each interaction) to enable
    /// the self-improvement system to track and optimize performance.
    pub fn observe_self(&mut self) {
        let snapshot =
            CognitiveSnapshot::now(self.stats.avg_phi, self.current_mode, self.flow_state);
        self.self_improvement.observe(snapshot);
    }

    /// Get top self-improvement recommendation
    ///
    /// Returns the highest-priority recommendation for improving consciousness.
    pub fn get_improvement_recommendation(&self) -> ImprovementRecommendation {
        self.self_improvement.top_recommendation()
    }

    /// Get all self-improvement recommendations
    pub fn get_all_recommendations(&self) -> Vec<ImprovementRecommendation> {
        self.self_improvement.all_recommendations().to_vec()
    }

    /// Apply an improvement recommendation
    ///
    /// Automatically applies the recommended improvement and records it.
    pub fn apply_recommendation(&mut self, recommendation: &ImprovementRecommendation) {
        match recommendation.improvement_type {
            ImprovementType::ModeSwitch(mode) => {
                self.current_mode = mode;
            }
            ImprovementType::IncreaseFocus => {
                // Increase flow state target
                self.flow_state = (self.flow_state + 0.1).min(1.0);
            }
            ImprovementType::DecreaseFocus => {
                // Decrease flow state target
                self.flow_state = (self.flow_state - 0.1).max(0.0);
            }
            ImprovementType::ResetAttention => {
                self.attention_router.reset();
            }
            ImprovementType::ConsolidateMemory => {
                // Trigger memory consolidation (would integrate with hippocampus)
            }
            ImprovementType::IncreaseIntegration => {
                // Increase binding strength (would affect cross-modal router)
            }
            ImprovementType::ReduceLoad => {
                // Reduce cognitive load
                self.flow_state = (self.flow_state - 0.2).max(0.3);
            }
            ImprovementType::None => {}
        }

        self.self_improvement
            .record_improvement(recommendation.improvement_type);
    }

    /// Auto-improve: automatically apply top recommendation if priority > threshold
    ///
    /// Returns true if an improvement was applied.
    pub fn auto_improve(&mut self, priority_threshold: f64) -> bool {
        let recommendation = self.get_improvement_recommendation();

        if recommendation.priority > priority_threshold {
            self.apply_recommendation(&recommendation);
            true
        } else {
            false
        }
    }

    /// Get self-improvement report
    pub fn self_improvement_report(&self) -> String {
        self.self_improvement.report()
    }

    /// Get current cognitive mode
    pub fn current_cognitive_mode(&self) -> CognitiveMode {
        self.current_mode
    }

    /// Set cognitive mode manually
    pub fn set_cognitive_mode(&mut self, mode: CognitiveMode) {
        self.current_mode = mode;
    }

    /// Get self-model accuracy (how well the system predicts itself)
    pub fn self_model_accuracy(&self) -> f64 {
        self.self_improvement.model_accuracy()
    }

    /// Get Φ trend from self-improvement module (positive = improving, negative = declining)
    pub fn self_improvement_phi_trend(&self) -> f64 {
        self.self_improvement.current_phi_trend()
    }

    /// Evaluate effectiveness of last improvement
    pub fn evaluate_last_improvement(&mut self) -> Option<f64> {
        self.self_improvement.evaluate_improvement()
    }

    // =========================================================================
    // COUNTERFACTUAL DREAMS
    // =========================================================================

    /// Add a counterfactual memory for dreaming
    ///
    /// The system can later explore this "what-if" scenario during sleep.
    ///
    /// # Arguments
    /// * `label` - Description of the memory
    /// * `actual` - HDC representation of what actually happened
    /// * `question` - The counterfactual question (e.g., "What if I had studied more?")
    /// * `counterfactual` - HDC representation of the counterfactual outcome
    /// * `intensity` - Emotional intensity (0.0-1.0) - higher = more likely to dream about
    pub fn add_counterfactual_memory(
        &mut self,
        label: &str,
        actual: BinaryHV,
        question: &str,
        counterfactual: BinaryHV,
        intensity: f64,
    ) -> u64 {
        self.counterfactual_dreams.add_memory_with_counterfactual(
            label,
            actual,
            question,
            counterfactual,
            intensity,
        )
    }

    /// Add a counterfactual memory with valence (positive or negative)
    ///
    /// Valence affects dream character:
    /// - Positive (> 0): Hope-oriented dreams
    /// - Negative (< 0): Regret-oriented dreams
    pub fn add_counterfactual_memory_with_valence(
        &mut self,
        label: &str,
        actual: BinaryHV,
        question: &str,
        counterfactual: BinaryHV,
        intensity: f64,
        valence: f64,
    ) -> u64 {
        self.counterfactual_dreams.add_memory_with_valence(
            label,
            actual,
            question,
            counterfactual,
            intensity,
            valence,
        )
    }

    /// Generate a counterfactual dream
    ///
    /// Creates a dream scenario that explores "what-if" possibilities.
    pub fn dream_counterfactually(&mut self, duration_minutes: f64) -> CounterfactualDreamScenario {
        self.counterfactual_dreams
            .generate_counterfactual_dream(duration_minutes)
    }

    /// Generate a lucid counterfactual dream
    ///
    /// A more controlled dream with higher awareness, optionally focused
    /// on a specific memory.
    pub fn dream_lucid_counterfactual(
        &mut self,
        duration_minutes: f64,
        focus_memory_id: Option<u64>,
    ) -> CounterfactualDreamScenario {
        self.counterfactual_dreams
            .generate_lucid_counterfactual_dream(duration_minutes, focus_memory_id)
    }

    /// Generate a nightmare based on negative counterfactuals
    ///
    /// Explores worst-case scenarios - can be cathartic for processing fears.
    pub fn dream_nightmare(&mut self, duration_minutes: f64) -> CounterfactualDreamScenario {
        self.counterfactual_dreams
            .generate_counterfactual_nightmare(duration_minutes)
    }

    /// Set dream bizarreness factor
    ///
    /// 0.0 = realistic dreams, 1.0 = very bizarre/surreal
    pub fn set_dream_bizarreness(&mut self, bizarreness: f64) {
        self.counterfactual_dreams.set_bizarreness(bizarreness);
    }

    /// Get dream report
    pub fn dream_report(&self) -> String {
        self.counterfactual_dreams.report()
    }

    /// Get number of counterfactual memories available for dreaming
    pub fn counterfactual_memory_count(&self) -> usize {
        self.counterfactual_dreams.memory_count()
    }

    /// Check if a dream provided insight
    pub fn dream_provided_insight(dream: &CounterfactualDreamScenario) -> bool {
        dream.resolution == DreamResolution::InsightGenerated
    }

    /// Get insight from dream (if any)
    pub fn get_dream_insight(dream: &CounterfactualDreamScenario) -> Option<&String> {
        dream.insight.as_ref()
    }

    // =========================================================================
    // INTEGRATED DREAM GENERATION (using ConsciousnessIntegrationBridge)
    // =========================================================================

    /// Generate a fully integrated dream influenced by emotional state and cognitive stress
    ///
    /// This is the most sophisticated dream generation method - it automatically:
    /// 1. Reads current emotional state (valence, arousal)
    /// 2. Calculates cognitive stress from self-improvement system
    /// 3. Adjusts dream parameters based on both factors
    /// 4. Generates appropriate dream type (nightmare, lucid, or regular)
    ///
    /// # Arguments
    /// * `duration_minutes` - Dream duration in minutes
    /// * `seed` - Random seed for reproducibility
    pub fn dream_integrated(
        &mut self,
        duration_minutes: f64,
        seed: u64,
    ) -> super::super::sleep_and_altered_states::DreamScenario {
        self.integration_bridge.generate_integrated_dream(
            &self.emotional_depth,
            &self.self_improvement,
            duration_minutes,
            seed,
        )
    }

    /// Generate an integrated counterfactual dream
    ///
    /// Combines emotional state and cognitive stress with counterfactual exploration.
    pub fn dream_integrated_counterfactual(
        &mut self,
        duration_minutes: f64,
    ) -> CounterfactualDreamScenario {
        self.integration_bridge.generate_integrated_counterfactual(
            &mut self.counterfactual_dreams,
            &self.emotional_depth,
            &self.self_improvement,
            duration_minutes,
        )
    }

    /// Get integration report showing cross-module correlations
    pub fn integration_report(&self) -> String {
        self.integration_bridge.integration_report()
    }

    // =========================================================================
    // FEEDBACK DYNAMICS ENGINE
    // =========================================================================

    /// Process dream and apply bidirectional feedback
    ///
    /// After a dream completes, call this to:
    /// 1. Adjust emotional state based on dream content
    /// 2. Queue memory consolidation for dream-highlighted memories
    /// 3. Generate attention hints for waking cognition
    pub fn process_dream_feedback(
        &mut self,
        dream: &super::super::sleep_and_altered_states::DreamScenario,
    ) -> Option<DreamInsight> {
        let insight = self.feedback_dynamics.feedback.process_dream(dream)?;

        // Apply emotional feedback
        self.feedback_dynamics
            .feedback
            .apply_emotional_feedback(&mut self.emotional_depth);

        Some(insight)
    }

    /// Process counterfactual dream feedback
    pub fn process_counterfactual_dream_feedback(
        &mut self,
        dream: &CounterfactualDreamScenario,
    ) -> Option<DreamInsight> {
        self.feedback_dynamics
            .feedback
            .process_counterfactual_dream(dream)
    }

    /// Get dream insights extracted from recent dreams
    pub fn dream_insights(&self, count: usize) -> Vec<DreamInsight> {
        self.feedback_dynamics
            .feedback
            .recent_insights(count)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get attention hints from dream processing
    pub fn dream_attention_hints(&mut self) -> Vec<AttentionHint> {
        self.feedback_dynamics.feedback.get_attention_adjustments()
    }

    // =========================================================================
    // EMOTIONAL PREDICTION
    // =========================================================================

    /// Record current emotional state for trajectory prediction
    ///
    /// Call this periodically to build emotional history for prediction.
    pub fn record_emotional_state(&mut self) {
        let blend = self.emotional_depth.current().clone();
        self.feedback_dynamics.predictor.record(&blend);
    }

    /// Predict next emotional state
    ///
    /// Returns prediction with confidence and optional intervention recommendation.
    pub fn predict_emotional_state(&mut self) -> Option<EmotionalPrediction> {
        self.feedback_dynamics.predictor.predict()
    }

    /// Check if proactive intervention is recommended
    ///
    /// Returns the intervention if predicted emotional trajectory suggests one.
    pub fn check_emotional_intervention(&mut self) -> Option<ProactiveIntervention> {
        let prediction = self.predict_emotional_state()?;
        prediction.intervention
    }

    /// Get prediction accuracy
    pub fn emotional_prediction_accuracy(&self) -> f64 {
        self.feedback_dynamics.predictor.prediction_accuracy()
    }

    // =========================================================================
    // CAUSAL DREAM INTEGRATION
    // =========================================================================

    /// Queue a causal hypothesis from CausalMind for dream exploration
    ///
    /// Dreams can explore hypotheses that are too uncertain for
    /// waking cognition to commit to.
    pub fn queue_causal_hypothesis_for_dream(&mut self, cause: &str, effect: &str, prior: f64) {
        self.feedback_dynamics
            .causal_dreams
            .add_hypothesis(cause, effect, prior);
    }

    /// Get causal discoveries from dream exploration
    pub fn causal_discoveries_from_dreams(&self, count: usize) -> Vec<CausalDiscovery> {
        self.feedback_dynamics
            .causal_dreams
            .recent_discoveries(count)
            .to_vec()
    }

    /// Integrate dream discoveries back into CausalMind
    ///
    /// This closes the loop: CausalMind → Dreams → Discoveries → CausalMind
    pub fn integrate_dream_discoveries(&mut self) {
        for discovery in self.causal_discoveries_from_dreams(10) {
            // Update CausalMind with dream-discovered relationships
            if discovery.counterfactual_support > 0.5 {
                self.causal_mind.learn_from_text(&format!(
                    "{} causes {} (dream-discovered, strength: {:.2})",
                    discovery.cause, discovery.effect, discovery.strength
                ));
            }
        }
    }

    // =========================================================================
    // COLLECTIVE DREAM SHARING
    // =========================================================================

    /// Share a dream insight with the collective
    pub fn share_dream_insight(&mut self, insight: &DreamInsight) {
        self.feedback_dynamics.collective.share_insight(insight);
    }

    /// Receive insights from collective consciousness
    pub fn receive_collective_insights(&mut self, insights: Vec<DreamInsight>) {
        self.feedback_dynamics
            .collective
            .receive_collective_insights(insights);
    }

    /// Calculate resonance with collective themes
    pub fn calculate_collective_resonance(&mut self, themes: &[String]) -> f64 {
        self.feedback_dynamics
            .collective
            .calculate_resonance(themes)
    }

    /// Get themes resonating in the collective
    pub fn resonant_collective_themes(&self, threshold: f64) -> Vec<String> {
        self.feedback_dynamics
            .collective
            .get_resonant_themes(threshold)
    }

    /// Get collective dream report
    pub fn collective_dream_report(&self) -> String {
        self.feedback_dynamics.collective.collective_report()
    }

    // =========================================================================
    // ADAPTIVE DREAM SCHEDULING
    // =========================================================================

    /// Record that a dream of a specific type occurred
    pub fn record_dream_occurrence(&mut self, dream_type: DreamType) {
        self.feedback_dynamics
            .scheduler
            .record_dream_occurrence(dream_type);
    }

    /// Get recommended dream type for current state
    ///
    /// Returns the optimal dream type based on:
    /// - Current stress level
    /// - Time since last dream of each type
    pub fn recommended_dream_type(&self) -> DreamType {
        self.feedback_dynamics.scheduler.recommend_dream_type()
    }

    /// Update current stress level for scheduling
    pub fn update_scheduling_stress(&mut self, stress: f64) {
        self.feedback_dynamics.scheduler.update_stress(stress);
    }

    /// Get scheduling report
    pub fn scheduling_report(&self) -> String {
        self.feedback_dynamics.scheduler.scheduling_report()
    }

    // =========================================================================
    // UNIFIED FEEDBACK DYNAMICS STEP
    // =========================================================================

    /// Run a complete feedback dynamics step
    ///
    /// This is the main integration point that:
    /// 1. Records current emotional state for prediction
    /// 2. Checks if intervention is needed
    /// 3. Updates scheduling stress from emotional arousal
    ///
    /// Call this periodically (e.g., after each interaction or on a timer).
    pub fn feedback_dynamics_step(&mut self) -> Option<ProactiveIntervention> {
        // Record emotional state for prediction
        self.record_emotional_state();

        // Estimate stress from emotional arousal (high arousal + negative valence = stress)
        let blend = self.emotional_depth.current();
        let stress = (blend.arousal * (1.0 - (blend.valence + 1.0) / 2.0)).clamp(0.0, 1.0);
        self.update_scheduling_stress(stress);

        // Check for proactive intervention needs
        self.check_emotional_intervention()
    }

    /// Get comprehensive feedback dynamics report
    pub fn feedback_dynamics_report(&self) -> String {
        self.feedback_dynamics.dynamics_report()
    }

    // =========================================================================
    // METACOGNITION ENGINE (Advanced Self-Monitoring & State Management)
    // =========================================================================

    /// Run a complete metacognitive cycle
    ///
    /// This integrates all metacognition subsystems:
    /// 1. Self-monitoring (quality tracking, auto-tuning)
    /// 2. Temporal patterns (circadian rhythms, memory consolidation)
    /// 3. Narrative identity (self-narrative, coherence)
    /// 4. Symbolic interpretation (archetypes, symbols)
    /// 5. Motivation (goals, salience, values)
    /// 6. State machine (consciousness state transitions)
    ///
    /// # Arguments
    /// * `current_phi` - Current integrated information level
    /// * `attention_efficiency` - How efficiently attention is being used
    /// * `minutes_elapsed` - Time elapsed since last cycle
    pub fn metacognitive_cycle(
        &mut self,
        current_phi: f64,
        attention_efficiency: f64,
        minutes_elapsed: f64,
    ) -> MetacognitiveCycleResult {
        let blend = self.emotional_depth.current().clone();
        self.metacognition.metacognitive_cycle(
            current_phi,
            &blend,
            attention_efficiency,
            minutes_elapsed,
        )
    }

    /// Get current consciousness state from state machine
    pub fn consciousness_state(&self) -> ConsciousnessState {
        self.metacognition.state_machine.current_state()
    }

    /// Transition consciousness state based on conditions
    ///
    /// The state machine tracks: Waking, Flow, Drowsy, Dreaming,
    /// LucidDreaming, DeepSleep, Contemplative, Hyperaroused
    ///
    /// # Arguments
    /// * `fatigue` - Sleep pressure / tiredness (0.0-1.0)
    /// * `focus` - Focus/phi level (0.0-1.0)
    /// * `stress` - Stress level (0.0-1.0)
    /// * `relaxation` - Relaxation level (0.0-1.0)
    pub fn transition_consciousness_state(
        &mut self,
        fatigue: f64,
        focus: f64,
        stress: f64,
        relaxation: f64,
    ) {
        self.metacognition
            .state_machine
            .evaluate_transition(fatigue, focus, stress, relaxation);
    }

    /// Get state-specific cognitive parameters
    ///
    /// Different consciousness states have different processing characteristics
    pub fn state_cognitive_params(
        &self,
    ) -> super::super::consciousness_metacognition::StateParameters {
        self.metacognition.state_machine.current_params().clone()
    }

    /// Get circadian phase (0.0 = midnight, 0.5 = noon)
    pub fn circadian_phase(&self) -> f64 {
        self.metacognition.temporal.phase()
    }

    /// Get current sleep pressure (0.0 = fully rested, 1.0 = exhausted)
    pub fn current_sleep_pressure(&self) -> f64 {
        self.metacognition.temporal.get_sleep_pressure()
    }

    /// Advance temporal patterns (call periodically, e.g., every "hour")
    pub fn advance_temporal(&mut self, minutes: f64) {
        self.metacognition.temporal.advance_time(minutes);
    }

    /// Queue a memory for consolidation during sleep
    pub fn queue_memory_consolidation(
        &mut self,
        memory_id: u64,
        content_hv: BinaryHV,
        emotional_salience: f64,
    ) {
        self.metacognition.temporal.queue_for_consolidation(
            memory_id,
            content_hv,
            emotional_salience,
        );
    }

    /// Get consolidated memories after sleep
    pub fn get_consolidated_memories(&mut self) -> Vec<u64> {
        self.metacognition.temporal.get_consolidated_memories()
    }

    /// Reset sleep pressure after sleep period
    pub fn reset_after_sleep(&mut self, sleep_duration_hours: f64) {
        self.metacognition
            .temporal
            .sleep_reset(sleep_duration_hours);
    }

    /// Get time of day description
    pub fn time_of_day(&self) -> String {
        self.metacognition.temporal.time_of_day()
    }

    /// Get current narrative identity summary
    pub fn narrative_summary(&self) -> &str {
        self.metacognition.narrative.current_narrative()
    }

    /// Get narrative coherence (how consistent is our self-story?)
    pub fn narrative_coherence(&self) -> f64 {
        self.metacognition.narrative.coherence()
    }

    /// Record a significant life event
    pub fn record_narrative_event(
        &mut self,
        description: &str,
        impact: f64,
        emotional_valence: f64,
        lessons: Vec<String>,
    ) {
        self.metacognition
            .narrative
            .record_event(description, impact, emotional_valence, lessons);
    }

    /// Integrate dream insights into narrative identity
    pub fn integrate_dream_narrative(&mut self, dream_summary: &str, insights: &[DreamInsight]) {
        self.metacognition
            .narrative
            .integrate_dream(dream_summary, insights);
    }

    /// Get life story summary
    pub fn life_story_summary(&self) -> String {
        self.metacognition.narrative.life_story_summary()
    }

    /// Interpret a dream symbolically
    ///
    /// Extracts symbols, archetypes, and personal meanings from dream content.
    pub fn interpret_dream(
        &mut self,
        dream: &super::super::sleep_and_altered_states::DreamScenario,
    ) -> super::super::consciousness_metacognition::DreamInterpretation {
        self.metacognition.symbols.interpret(dream)
    }

    /// Learn a personal symbol meaning from experience
    pub fn learn_symbol(
        &mut self,
        symbol: &str,
        personal_meaning: &str,
        emotional_association: f64,
    ) {
        self.metacognition
            .symbols
            .learn_symbol(symbol, personal_meaning, emotional_association);
    }

    /// Get recurring symbols and their frequencies
    pub fn recurring_symbols(&self, limit: usize) -> Vec<(String, u32)> {
        self.metacognition.symbols.most_recurring_symbols(limit)
    }

    /// Add an explicit goal
    pub fn add_goal(&mut self, description: &str, priority: f64) -> u64 {
        self.metacognition
            .motivation
            .add_goal(description, priority)
    }

    /// Generate a goal from emotional state
    pub fn generate_goal_from_emotion(&mut self) -> Option<Goal> {
        let blend = self.emotional_depth.current().clone();
        self.metacognition
            .motivation
            .generate_goal_from_emotion(&blend)
    }

    /// Generate a goal from dream insight
    pub fn generate_goal_from_dream(&mut self, insight: &DreamInsight) -> Option<Goal> {
        self.metacognition
            .motivation
            .generate_goal_from_dream(insight)
    }

    /// Get active goals
    pub fn active_goals(&self) -> &[Goal] {
        self.metacognition.motivation.active_goals()
    }

    /// Update goal progress
    pub fn update_goal_progress(&mut self, goal_id: u64, progress: f64) {
        self.metacognition
            .motivation
            .update_goal_progress(goal_id, progress);
    }

    /// Complete a goal
    pub fn complete_goal(&mut self, goal_id: u64) -> Option<Goal> {
        self.metacognition.motivation.complete_goal(goal_id)
    }

    /// Reinforce a learned value based on outcome
    pub fn reinforce_value(&mut self, value_name: &str, outcome: f64) {
        self.metacognition
            .motivation
            .reinforce_value(value_name, outcome);
    }

    /// Update emotional salience for a topic
    pub fn update_salience(&mut self, topic: &str, emotional_response: f64) {
        self.metacognition
            .motivation
            .update_salience(topic, emotional_response);
    }

    /// Get motivational state
    pub fn motivational_state(
        &self,
    ) -> &super::super::consciousness_metacognition::MotivationalState {
        self.metacognition.motivation.motivational_state()
    }

    /// Get quality trend from self-monitoring
    pub fn quality_trend(&self) -> QualityTrend {
        self.metacognition.monitor.quality_trend()
    }

    /// Get auto-tuning parameters
    pub fn auto_tuning_params(
        &self,
    ) -> &super::super::consciousness_metacognition::AutoTuningParams {
        self.metacognition.monitor.tuning_params()
    }

    /// Record a phi measurement with context
    pub fn record_phi_measurement(&mut self, phi: f64, context: &str) {
        self.metacognition.monitor.record_phi(phi, context);
    }

    /// Assess current cognitive quality
    pub fn assess_quality(
        &mut self,
    ) -> super::super::consciousness_metacognition::QualityAssessment {
        self.metacognition.monitor.assess_quality()
    }

    /// Update self-model based on experience
    pub fn update_self_model(&mut self, experience: &str, outcome: f64) {
        self.metacognition
            .monitor
            .update_self_model(experience, outcome);
    }

    /// Get self-model
    pub fn self_model(&self) -> &super::super::consciousness_metacognition::SelfModel {
        self.metacognition.monitor.self_model()
    }

    /// Get comprehensive metacognition report
    pub fn metacognition_report(&self) -> String {
        self.metacognition.status_report()
    }

    // =========================================================================
    // CONSCIOUSNESS STATE MACHINE
    // =========================================================================

    /// Trigger lucidity during a dream
    pub fn gain_lucidity(&mut self) -> bool {
        self.metacognition.state_machine.gain_lucidity()
    }

    /// Wake up from any sleep state
    pub fn wake_up(&mut self) {
        self.metacognition.state_machine.wake_up();
    }

    /// Get recent state transitions
    pub fn recent_state_transitions(
        &self,
        count: usize,
    ) -> Vec<&super::super::consciousness_metacognition::StateTransition> {
        self.metacognition.state_machine.recent_transitions(count)
    }

    /// Get time distribution across consciousness states
    pub fn state_time_distribution(&self) -> std::collections::HashMap<ConsciousnessState, f64> {
        self.metacognition.state_machine.state_time_distribution()
    }

    // =========================================================================
    // INTEGRATED METACOGNITIVE STEP
    // =========================================================================

    /// Run integrated metacognitive step
    ///
    /// This is the main integration point that:
    /// 1. Runs a metacognitive cycle with current metrics
    /// 2. Updates consciousness state machine
    /// 3. Advances temporal patterns
    /// 4. Generates goals from emotional state
    /// 5. Returns comprehensive result
    ///
    /// Call this after each major interaction or on a timer.
    /// Uses a default of 1 minute elapsed.
    pub fn metacognitive_step(&mut self) -> MetacognitiveCycleResult {
        self.metacognitive_step_with_time(1.0)
    }

    /// Run integrated metacognitive step with explicit time
    pub fn metacognitive_step_with_time(
        &mut self,
        minutes_elapsed: f64,
    ) -> MetacognitiveCycleResult {
        // Get current metrics
        let phi = self.stats.avg_phi;
        let attention_efficiency = self.flow_state as f64;
        let blend = self.emotional_depth.current().clone();

        // Run metacognitive cycle
        let result = self.metacognition.metacognitive_cycle(
            phi,
            &blend,
            attention_efficiency,
            minutes_elapsed,
        );

        // Update consciousness state
        // Map to evaluate_transition(fatigue, focus, stress, relaxation)
        let arousal = blend.arousal;
        let fatigue = self.metacognition.temporal.get_sleep_pressure();
        let focus = phi; // Higher phi = better focus
        let stress = arousal.max(0.0); // Use positive arousal as stress indicator
        let relaxation = (1.0 - arousal).max(0.0); // Inverse of arousal
        self.metacognition
            .state_machine
            .evaluate_transition(fatigue, focus, stress, relaxation);

        // Queue memory consolidation if important interaction
        if phi > 0.6 {
            let emotional_weight = blend.valence.abs();
            let memory_hv = BinaryHV::random((phi * 1000.0) as u64);
            let memory_id = (phi * 10000.0) as u64;
            self.metacognition.temporal.queue_for_consolidation(
                memory_id,
                memory_hv,
                emotional_weight,
            );
        }

        // Generate goal from current emotional state if strong emotion
        if blend.valence.abs() > 0.5 {
            let _ = self
                .metacognition
                .motivation
                .generate_goal_from_emotion(&blend);
        }

        result
    }

    /// Process a dream through all metacognition systems
    pub fn process_dream_metacognitively(
        &mut self,
        dream: &super::super::sleep_and_altered_states::DreamScenario,
        insights: &[DreamInsight],
    ) {
        self.metacognition.process_dream(dream, insights);
    }
}
