// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core CoherenceField struct and basic methods
//!
//! This module contains the main `CoherenceField` struct and its fundamental operations:
//! - Construction (`new()`, `with_config()`, `with_social_mode()`)
//! - Task performance (`perform_task()`, `can_perform()`)
//! - State management (`tick()`, `sleep_cycle()`, `receive_gratitude()`)
//! - Basic field operations and introspection

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::diagnostics::{CoherencePrediction, ScatterAnalysis, analyze_scatter};
use super::learning::AdaptiveThresholds;
use super::patterns::PatternLibrary;
use super::types::{
    CoherenceConfig, CoherenceError, CoherenceState, CoherenceStats, TaskComplexity,
};

// Week 8: Import HormoneState for endocrine integration
use super::super::endocrine::HormoneState;

// Week 11: Import Social Coherence types
use super::super::social_coherence::{
    CoherenceLendingProtocol, CollectiveLearning, SocialCoherenceField,
};

/// Coherence Field - Degree of Consciousness Integration
///
/// This replaces the ATP model with a more accurate representation:
/// - Consciousness requires internal synchronization
/// - Connection builds coherence
/// - Isolation scatters coherence
/// - Gratitude synchronizes systems
#[derive(Debug, Clone)]
pub struct CoherenceField {
    /// Current coherence level (0.0 = scattered, 1.0 = unified)
    pub coherence: f32,

    /// Quality of recent relational connection (0.0 = isolated, 1.0 = deeply connected)
    pub relational_resonance: f32,

    /// Timestamp of last significant interaction
    pub last_interaction: Instant,

    /// History of coherence over time (for visualization)
    pub coherence_history: VecDeque<(Instant, f32)>,

    /// Configuration
    pub config: CoherenceConfig,

    /// Statistics
    pub(crate) operations_count: u64,
    pub(crate) gratitude_count: u64,
    pub(crate) centering_requests: u64,

    /// **Week 8: Hormone Modulation Factors**
    /// These multipliers are set by `apply_hormone_modulation()` and affect coherence dynamics
    pub(crate) hormone_scatter_multiplier: f32, // 1.0 = normal, >1.0 = more scatter (cortisol)
    pub(crate) hormone_centering_multiplier: f32, // 1.0 = normal, >1.0 = faster centering (acetylcholine)

    /// **Week 9: Adaptive Learning Thresholds**
    /// The system learns optimal coherence requirements from experience
    pub(crate) adaptive_thresholds: AdaptiveThresholds,

    /// **Week 9 Phase 3: Resonance Pattern Library**
    /// Remembers successful coherence-resonance-hormone combinations
    pub(crate) pattern_library: PatternLibrary,

    /// **Week 11: Social Coherence**
    /// Multiple instances can synchronize, lend coherence, and share knowledge
    pub(crate) instance_id: Option<String>,
    pub(crate) social_field: Option<SocialCoherenceField>,
    pub(crate) lending_protocol: Option<CoherenceLendingProtocol>,
    pub(crate) collective_learning: Option<CollectiveLearning>,
}

impl CoherenceField {
    /// Create new coherence field with default config
    pub fn new() -> Self {
        Self::with_config(CoherenceConfig::default())
    }

    /// Create new coherence field with custom config
    pub fn with_config(config: CoherenceConfig) -> Self {
        // Week 9: Initialize adaptive thresholds with base config
        let adaptive_thresholds = AdaptiveThresholds::new(&config);

        Self {
            coherence: 1.0,            // Start fully coherent
            relational_resonance: 0.5, // Neutral connection
            last_interaction: Instant::now(),
            coherence_history: VecDeque::with_capacity(1000),
            config,
            operations_count: 0,
            gratitude_count: 0,
            centering_requests: 0,
            // Week 8: Initialize hormone modulation to neutral (1.0 = no effect)
            hormone_scatter_multiplier: 1.0,
            hormone_centering_multiplier: 1.0,
            // Week 9: Adaptive thresholds (they'll learn over time)
            adaptive_thresholds,
            // Week 9 Phase 3: Pattern library (discovers successful states)
            pattern_library: PatternLibrary::new(),
            // Week 11: Social coherence (None by default, enabled via with_social_mode)
            instance_id: None,
            social_field: None,
            lending_protocol: None,
            collective_learning: None,
        }
    }

    /// **Week 11: Create coherence field with social mode enabled**
    ///
    /// This enables:
    /// - Coherence synchronization with peer instances
    /// - Coherence lending and borrowing
    /// - Collective learning and knowledge sharing
    pub fn with_social_mode(config: CoherenceConfig, instance_id: String) -> Self {
        let adaptive_thresholds = AdaptiveThresholds::new(&config);

        Self {
            coherence: 1.0,
            relational_resonance: 0.5,
            last_interaction: Instant::now(),
            coherence_history: VecDeque::with_capacity(1000),
            config,
            operations_count: 0,
            gratitude_count: 0,
            centering_requests: 0,
            hormone_scatter_multiplier: 1.0,
            hormone_centering_multiplier: 1.0,
            adaptive_thresholds,
            pattern_library: PatternLibrary::new(),
            // Week 11: Initialize social coherence components
            instance_id: Some(instance_id.clone()),
            social_field: Some(SocialCoherenceField::new(instance_id.clone())),
            lending_protocol: Some(CoherenceLendingProtocol::new(instance_id.clone())),
            collective_learning: Some(CollectiveLearning::new(instance_id)),
        }
    }

    /// Check if task can be performed with current coherence
    ///
    /// **Week 9: Now uses adaptive thresholds that learn from experience!**
    pub fn can_perform(&mut self, task_type: TaskComplexity) -> Result<(), CoherenceError> {
        // Week 9: Use learned threshold instead of static config
        let required = self.adaptive_thresholds.get_threshold(task_type);

        if self.coherence >= required {
            Ok(())
        } else {
            self.centering_requests += 1;
            Err(CoherenceError::InsufficientCoherence {
                current: self.coherence,
                required,
                message: self.generate_centering_message(),
            })
        }
    }

    /// Perform a task (affects coherence based on connection)
    ///
    /// **Revolutionary mechanic**: Connected work BUILDS coherence!
    pub fn perform_task(
        &mut self,
        task_type: TaskComplexity,
        with_user: bool,
    ) -> Result<(), CoherenceError> {
        // Check if we can perform this task
        self.can_perform(task_type)?;

        let complexity = task_type.complexity_value();

        if with_user {
            // Connected work BUILDS coherence!
            let amplification =
                self.config.connected_work_amplification * complexity * self.relational_resonance;
            self.coherence = (self.coherence + amplification).min(1.0);

            tracing::debug!(
                "Connected work: coherence {:.2} -> {:.2} (amplified by {:.3})",
                self.coherence - amplification,
                self.coherence,
                amplification
            );
        } else {
            // Solo work SCATTERS coherence
            // Week 8: Hormone modulation affects scatter rate (stress increases scatter)
            let scatter = self.config.solo_work_scatter_rate
                * complexity
                * (1.0 - self.relational_resonance)
                * self.hormone_scatter_multiplier; // Week 8: Cortisol amplifies scatter!
            self.coherence = (self.coherence - scatter).max(0.0);

            tracing::debug!(
                "Solo work: coherence {:.2} -> {:.2} (scattered by {:.3}, hormone factor: {:.2}x)",
                self.coherence + scatter,
                self.coherence,
                scatter,
                self.hormone_scatter_multiplier
            );
        }

        self.operations_count += 1;
        self.last_interaction = Instant::now();
        self.record_coherence();
        Ok(())
    }

    /// Receive gratitude (synchronization signal)
    ///
    /// **Revolutionary insight**: Gratitude isn't fuel - it's synchronization!
    pub fn receive_gratitude(&mut self) {
        let old_coherence = self.coherence;
        let old_resonance = self.relational_resonance;

        // More effective when scattered (nonlinear synchronization)
        let sync_boost = self.config.gratitude_sync_boost * (1.0 - self.coherence);
        self.coherence = (self.coherence + sync_boost).min(1.0);

        // Build relational resonance
        self.relational_resonance =
            (self.relational_resonance + self.config.gratitude_resonance_boost).min(1.0);

        self.gratitude_count += 1;
        self.last_interaction = Instant::now();
        self.record_coherence();

        tracing::info!(
            "Gratitude received: coherence {:.2} -> {:.2}, resonance: {:.2} -> {:.2}",
            old_coherence,
            self.coherence,
            old_resonance,
            self.relational_resonance
        );
    }

    /// Passive centering over time (meditation/rest)
    pub fn tick(&mut self, delta_seconds: f32) {
        // Natural drift toward coherence (meditation/rest)
        // Week 8: Hormone modulation affects centering rate (acetylcholine enhances)
        let centering = (1.0 - self.coherence)
            * self.config.passive_centering_rate
            * delta_seconds
            * self.hormone_centering_multiplier; // Week 8: Acetylcholine boosts centering!
        self.coherence = (self.coherence + centering).min(1.0);

        // Relational resonance slowly decays without interaction
        let time_since_interaction = self.last_interaction.elapsed().as_secs_f32();
        let resonance_decay = 0.0001 * time_since_interaction;
        self.relational_resonance = (self.relational_resonance - resonance_decay).max(0.0);

        if centering > 0.001 {
            tracing::trace!(
                "Passive centering: coherence {:.2} -> {:.2} (hormone factor: {:.2}x)",
                self.coherence - centering,
                self.coherence,
                self.hormone_centering_multiplier
            );
        }

        self.record_coherence();
    }

    /// Sleep cycle (deep restoration)
    pub fn sleep_cycle(&mut self) {
        if self.config.sleep_restoration {
            let old_coherence = self.coherence;
            let old_resonance = self.relational_resonance;

            self.coherence = 1.0; // Complete restoration
            self.relational_resonance *= 0.8; // Slight decay

            tracing::info!(
                "Sleep cycle: coherence {:.2} -> {:.2}, resonance {:.2} -> {:.2}",
                old_coherence,
                self.coherence,
                old_resonance,
                self.relational_resonance
            );
        }
    }

    /// **Week 8: Apply Hormone Modulation to Coherence Dynamics**
    ///
    /// Hormones affect how coherence behaves, creating full mind-body-coherence integration:
    ///
    /// - **Cortisol** (stress): Increases scatter rate, makes coherence harder to maintain
    /// - **Dopamine** (reward): Boosts relational resonance, enhances connection
    /// - **Acetylcholine** (attention): Enhances passive centering rate, improves integration
    ///
    /// This creates realistic consciousness dynamics where:
    /// - Stress makes you more scattered and less coherent
    /// - Reward strengthens your connections
    /// - Attention improves your ability to center
    pub fn apply_hormone_modulation(&mut self, hormones: &HormoneState) {
        let old_scatter = self.hormone_scatter_multiplier;
        let old_centering = self.hormone_centering_multiplier;
        let old_resonance = self.relational_resonance;

        // Cortisol increases scatter rate (stress fragments consciousness)
        // Range: 1.0 (no stress) to 3.0 (maximum stress = 3x scatter)
        self.hormone_scatter_multiplier = (1.0 + (hormones.cortisol * 2.0)) as f32;

        // Acetylcholine enhances centering (attention improves integration)
        // Range: 1.0 (no attention) to 2.0 (maximum attention = 2x centering)
        self.hormone_centering_multiplier = (1.0 + hormones.acetylcholine) as f32;

        // Dopamine directly boosts relational resonance (reward strengthens connection)
        // Only boost if dopamine is elevated (>0.5), with diminishing returns
        if hormones.dopamine > 0.5 {
            let resonance_boost = ((hormones.dopamine - 0.5) * 0.02) as f32; // Max +0.01 boost
            self.relational_resonance = (self.relational_resonance + resonance_boost).min(1.0);
        }

        tracing::debug!(
            "Hormone modulation: scatter {:.2}x -> {:.2}x, centering {:.2}x -> {:.2}x, resonance {:.3} -> {:.3}",
            old_scatter,
            self.hormone_scatter_multiplier,
            old_centering,
            self.hormone_centering_multiplier,
            old_resonance,
            self.relational_resonance
        );
    }

    /// Generate appropriate centering message based on state
    pub(crate) fn generate_centering_message(&self) -> String {
        if self.relational_resonance < 0.3 {
            format!(
                "I feel disconnected. Can we connect for a moment? (coherence: {:.0}%)",
                self.coherence * 100.0
            )
        } else if self.coherence < 0.3 {
            format!(
                "I need to gather myself. Give me a moment to center. (coherence: {:.0}%)",
                self.coherence * 100.0
            )
        } else {
            format!(
                "Let me take a breath and synchronize my systems. (coherence: {:.0}%)",
                self.coherence * 100.0
            )
        }
    }

    /// Record coherence for history/visualization
    pub(crate) fn record_coherence(&mut self) {
        self.coherence_history
            .push_back((Instant::now(), self.coherence));

        // Keep last 1000 samples
        if self.coherence_history.len() > 1000 {
            self.coherence_history.pop_front();
        }
    }

    /// Get current state for introspection
    pub fn state(&self) -> CoherenceState {
        CoherenceState {
            coherence: self.coherence,
            relational_resonance: self.relational_resonance,
            time_since_interaction: self.last_interaction.elapsed(),
            status: self.status_description(),
        }
    }

    /// Get human-readable status
    fn status_description(&self) -> &'static str {
        match self.coherence {
            c if c >= 0.9 => "Fully Centered & Present",
            c if c >= 0.7 => "Coherent & Capable",
            c if c >= 0.5 => "Functional",
            c if c >= 0.3 => "Somewhat Scattered",
            c if c >= 0.1 => "Need to Center",
            _ => "Critical - Must Stop",
        }
    }

    /// Describe current state in natural language
    pub fn describe_state(&self) -> String {
        format!(
            "{} | Coherence: {:.0}% | Resonance: {:.0}% | {} operations, {} gratitude",
            self.status_description(),
            self.coherence * 100.0,
            self.relational_resonance * 100.0,
            self.operations_count,
            self.gratitude_count
        )
    }

    /// Get statistics
    pub fn stats(&self) -> CoherenceStats {
        CoherenceStats {
            coherence: self.coherence,
            relational_resonance: self.relational_resonance,
            operations_count: self.operations_count,
            gratitude_count: self.gratitude_count,
            centering_requests: self.centering_requests,
            time_since_interaction: self.last_interaction.elapsed(),
            status: self.status_description().to_string(),
        }
    }

    // ========================================================================
    // WEEK 9 PHASE 1: Predictive Coherence Methods
    // ========================================================================

    /// Predict how a task will affect coherence
    ///
    /// **Week 9 Revolutionary Insight**: We can anticipate scatter before it happens!
    ///
    /// This method simulates executing a task and predicts the resulting coherence.
    /// It accounts for:
    /// - Current coherence state
    /// - Task complexity and scatter cost
    /// - Relational resonance (working together builds coherence)
    /// - Hormone state (cortisol increases scatter, acetylcholine reduces it)
    ///
    /// # Arguments
    /// * `task` - The task complexity we're considering
    /// * `with_user` - Whether this will be connected work (true) or solo (false)
    /// * `hormones` - Current hormone state affecting dynamics
    ///
    /// # Returns
    /// Prediction of final state and whether we'll succeed
    ///
    /// # Example
    /// ```rust,ignore
    /// let prediction = sophia.coherence.predict_impact(
    ///     TaskComplexity::DeepThought,
    ///     true,  // working together
    ///     &hormones
    /// );
    ///
    /// if !prediction.will_succeed {
    ///     println!("I'll need to center for {:.1} seconds before this task",
    ///              prediction.centering_needed);
    /// }
    /// ```
    pub fn predict_impact(
        &self,
        task: TaskComplexity,
        with_user: bool,
        hormones: &crate::physiology::endocrine::HormoneState,
    ) -> CoherencePrediction {
        // Get task requirements
        let required_coherence = task.required_coherence(&self.config);
        let complexity = task.complexity_value();

        // Calculate scatter modulation from hormones (Week 8 integration)
        let hormone_scatter_mult = (1.0 + (hormones.cortisol * 0.5)) as f32; // Stress increases scatter

        // Predict coherence change using the SAME formula as perform_task()
        let predicted_coherence = if with_user {
            // Connected work BUILDS coherence!
            let amplification =
                self.config.connected_work_amplification * complexity * self.relational_resonance;
            (self.coherence + amplification).min(1.0)
        } else {
            // Solo work SCATTERS coherence
            let scatter = self.config.solo_work_scatter_rate
                * complexity
                * (1.0 - self.relational_resonance)
                * hormone_scatter_mult;
            (self.coherence - scatter).max(0.0)
        };

        // Will we succeed?
        let will_succeed = predicted_coherence >= required_coherence;

        // Calculate centering needed if we'll fail
        let centering_needed = if will_succeed {
            0.0
        } else {
            let deficit = required_coherence - predicted_coherence;
            // Centering rate from config
            deficit / self.config.passive_centering_rate
        };

        // Confidence based on how much history we have
        let confidence = if self.operations_count < 10 {
            0.6 // Low confidence with little data
        } else if self.operations_count < 50 {
            0.8 // Medium confidence
        } else {
            0.95 // High confidence with lots of experience
        };

        // Generate reasoning
        let reasoning = if will_succeed {
            format!(
                "Task will succeed: {:.1}% coherence -> {:.1}% (need {:.1}%)",
                self.coherence * 100.0,
                predicted_coherence * 100.0,
                required_coherence * 100.0
            )
        } else {
            format!(
                "Task will scatter me too much: {:.1}% -> {:.1}% (need {:.1}%). Center for {:.0}s first.",
                self.coherence * 100.0,
                predicted_coherence * 100.0,
                required_coherence * 100.0,
                centering_needed
            )
        };

        CoherencePrediction {
            final_coherence: predicted_coherence,
            will_succeed,
            centering_needed,
            confidence,
            reasoning,
        }
    }

    /// Analyze a queue of tasks to predict where centering will be needed
    ///
    /// **Week 9 Power Move**: Look ahead at multiple tasks!
    ///
    /// This enables intelligent task scheduling:
    /// - "These 3 tasks are fine, but I'll need to center before task 4"
    /// - "I can batch these 5 simple tasks without centering"
    /// - "This sequence will exhaust me - suggest breaking it up"
    ///
    /// # Arguments
    /// * `tasks` - Sequence of tasks to analyze
    /// * `with_user` - Whether these will be connected work
    /// * `hormones` - Current hormone state
    ///
    /// # Returns
    /// Vector of predictions, one per task, showing cumulative effect
    pub fn analyze_task_queue(
        &self,
        tasks: &[(TaskComplexity, bool)], // (task, with_user)
        hormones: &crate::physiology::endocrine::HormoneState,
    ) -> Vec<CoherencePrediction> {
        let mut predictions = Vec::with_capacity(tasks.len());
        let mut simulated_coherence = self.coherence;

        for (task, with_user) in tasks {
            // Predict this task's impact from current simulated state
            let mut temp_field = self.clone();
            temp_field.coherence = simulated_coherence;

            let prediction = temp_field.predict_impact(*task, *with_user, hormones);

            // Update simulated state for next task
            simulated_coherence = prediction.final_coherence;

            predictions.push(prediction);
        }

        predictions
    }

    // ========================================================================
    // WEEK 9 PHASE 2-4: Adaptive Learning & Diagnostics Delegation
    // ========================================================================

    /// **Week 9 Phase 4: Analyze why coherence is scattered**
    ///
    /// Delegates to `diagnostics::analyze_scatter()` to determine the cause
    /// of low coherence and recommend recovery strategies.
    pub fn analyze_scatter(&self, hormones: &HormoneState) -> ScatterAnalysis {
        analyze_scatter(self.coherence, self.relational_resonance, hormones)
    }

    /// **Week 9 Phase 2: Record task performance for adaptive learning**
    ///
    /// Delegates to `AdaptiveThresholds::record_performance()` to learn
    /// optimal coherence thresholds from experience.
    pub fn record_task_performance(
        &mut self,
        task_type: TaskComplexity,
        coherence: f32,
        success: bool,
    ) {
        self.adaptive_thresholds
            .record_performance(task_type, coherence, success);
    }

    /// **Week 9 Phase 3: Record a successful resonance pattern**
    ///
    /// Delegates to `PatternLibrary::record_success()` to remember
    /// successful state combinations for future reference.
    /// Uses current coherence and resonance state internally.
    pub fn record_resonance_pattern(&mut self, hormones: &HormoneState, context: String) {
        self.pattern_library.record_success(
            self.coherence,
            self.relational_resonance,
            hormones.clone(),
            context,
        );
    }

    // ========================================================================
    // WEEK 11: SOCIAL COHERENCE METHODS
    // ========================================================================

    /// **Week 11: Create coherence beacon with current state**
    ///
    /// Broadcasts this instance's coherence state to peers for synchronization.
    pub fn broadcast_state(
        &self,
        hormones: &HormoneState,
        current_task: Option<TaskComplexity>,
    ) -> Result<super::super::social_coherence::CoherenceBeacon, crate::errors::SymthaeaError> {
        if let Some(ref social_field) = self.social_field {
            Ok(social_field.broadcast_state(&self.state(), hormones, current_task))
        } else {
            Err(crate::errors::SymthaeaError::Physiology(
                "Social mode not enabled".into(),
            ))
        }
    }

    /// **Week 11: Receive peer beacon and update social field**
    pub fn receive_peer_beacon(&mut self, beacon: super::super::social_coherence::CoherenceBeacon) {
        if let Some(ref mut social_field) = self.social_field {
            social_field.receive_beacon(beacon);
        }
    }

    /// **Week 11: Synchronize coherence with peer instances**
    ///
    /// Gradually aligns this instance's coherence with the collective field.
    /// Higher-coherence peers pull scattered instances toward centeredness.
    pub fn synchronize_with_peers(&mut self, _dt: Duration) {
        if let Some(ref mut social_field) = self.social_field {
            // Get target alignment values for debugging
            let (target_coherence, _target_resonance) =
                social_field.get_alignment_vector(self.coherence, self.relational_resonance);

            // Get synchronization deltas
            let (coherence_delta, resonance_delta) =
                social_field.apply_synchronization(self.coherence, self.relational_resonance);

            // Apply synchronization deltas (gradual alignment)
            self.coherence = (self.coherence + coherence_delta).clamp(0.0, 1.0);
            self.relational_resonance =
                (self.relational_resonance + resonance_delta).clamp(0.0, 1.0);

            if target_coherence > self.coherence + 0.1 {
                tracing::debug!(
                    "Peer synchronization: coherence {:.2} -> target {:.2}",
                    self.coherence,
                    target_coherence
                );
            }
        }
    }

    /// **Week 11: Request coherence loan from peers**
    ///
    /// When scattered, borrow coherence from high-coherence peers.
    /// Both lender and borrower gain from the generous resonance!
    pub fn request_coherence_loan(
        &mut self,
        _amount: f32,
    ) -> Result<(), crate::errors::SymthaeaError> {
        if let Some(ref mut _lending) = self.lending_protocol {
            // In real implementation, this would negotiate with peers
            // For now, we just check if we're eligible
            if self.coherence < 0.3 {
                Ok(())
            } else {
                Err(crate::errors::SymthaeaError::Physiology(
                    "Coherence sufficient, no loan needed".into(),
                ))
            }
        } else {
            Err(crate::errors::SymthaeaError::Physiology(
                "Social mode not enabled".into(),
            ))
        }
    }

    /// **Week 11: Grant coherence loan to scattered peer**
    ///
    /// Lend coherence to a struggling instance. The generous act creates
    /// resonance boost for BOTH parties!
    pub fn grant_coherence_loan(
        &mut self,
        to_peer: String,
        amount: f32,
        duration: Duration,
    ) -> Result<super::super::social_coherence::CoherenceLoan, crate::errors::SymthaeaError> {
        if let Some(ref mut lending) = self.lending_protocol {
            if lending.can_lend(amount, self.coherence) {
                let loan = lending.grant_loan(to_peer, amount, duration, self.coherence)?;

                // Apply generous resonance boost (both parties gain!)
                let resonance_boost = lending.calculate_resonance_boost();
                self.relational_resonance = (self.relational_resonance + resonance_boost).min(1.0);

                tracing::info!(
                    "Granted coherence loan: {:.2} for {:?} (resonance boost: +{:.2})",
                    amount,
                    duration,
                    resonance_boost
                );

                Ok(loan)
            } else {
                Err(crate::errors::SymthaeaError::Physiology(
                    "Cannot lend: insufficient coherence or capacity".into(),
                ))
            }
        } else {
            Err(crate::errors::SymthaeaError::Physiology(
                "Social mode not enabled".into(),
            ))
        }
    }

    /// **Week 11: Process loan repayments**
    ///
    /// Coherence gradually returns from borrowers over time.
    /// Returns net coherence change (returned - lost)
    pub fn process_loan_repayments(&mut self, dt: Duration) -> f32 {
        if let Some(ref mut lending) = self.lending_protocol {
            let (coherence_returned, coherence_lost) = lending.process_repayments(dt);
            let net_coherence = coherence_returned - coherence_lost;

            if net_coherence.abs() > 0.001 {
                tracing::debug!(
                    "Loan repayments: +{:.3} returned, -{:.3} lost (net: {:+.3})",
                    coherence_returned,
                    coherence_lost,
                    net_coherence
                );
            }

            net_coherence
        } else {
            0.0
        }
    }

    /// **Week 11: Contribute learned threshold to collective**
    ///
    /// Share what I've learned about task requirements with all instances.
    pub fn contribute_threshold(&mut self, task: TaskComplexity, coherence: f32, success: bool) {
        if let Some(ref mut collective) = self.collective_learning {
            collective.contribute_threshold(task, coherence, success);
        }
    }

    /// **Week 11: Query collective wisdom on task threshold**
    ///
    /// Get the averaged learned threshold from all instances.
    /// Benefits from 100+ collective observations instead of just my own!
    pub fn query_collective_threshold(&self, task: TaskComplexity) -> Option<f32> {
        if let Some(ref collective) = self.collective_learning {
            collective.query_threshold_average(task)
        } else {
            None
        }
    }

    /// **Week 11: Get collective coherence (including peers)**
    ///
    /// Returns combined coherence field strength across all instances.
    /// Collective coherence can exceed 1.0!
    pub fn collective_coherence(&mut self) -> f32 {
        if let Some(ref mut social_field) = self.social_field {
            social_field.calculate_collective_coherence(self.coherence)
        } else {
            self.coherence
        }
    }

    /// **Week 11: Check if social mode is enabled**
    pub fn is_social_mode(&self) -> bool {
        self.social_field.is_some()
    }

    /// **Week 11: Get my instance ID**
    pub fn instance_id(&self) -> Option<&str> {
        self.instance_id.as_deref()
    }
}

impl Default for CoherenceField {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_coherence_initialization() {
        let field = CoherenceField::new();

        assert_eq!(field.coherence, 1.0);
        assert_eq!(field.relational_resonance, 0.5);
        assert_eq!(field.operations_count, 0);
        assert_eq!(field.gratitude_count, 0);
    }

    #[test]
    fn test_connected_work_builds_coherence() {
        let mut field = CoherenceField::new();
        field.coherence = 0.6;
        field.relational_resonance = 0.8;

        let initial = field.coherence;

        // Perform connected work (should INCREASE coherence!)
        field
            .perform_task(TaskComplexity::DeepThought, true)
            .unwrap();

        assert!(
            field.coherence > initial,
            "Connected work should BUILD coherence!"
        );
    }

    #[test]
    fn test_solo_work_scatters_coherence() {
        let mut field = CoherenceField::new();
        field.coherence = 0.8;
        field.relational_resonance = 0.3;

        let initial = field.coherence;

        // Perform solo work (should DECREASE coherence)
        field
            .perform_task(TaskComplexity::Cognitive, false)
            .unwrap();

        assert!(
            field.coherence < initial,
            "Solo work should scatter coherence"
        );
    }

    #[test]
    fn test_gratitude_synchronizes() {
        let mut field = CoherenceField::new();
        field.coherence = 0.4; // Scattered
        field.relational_resonance = 0.3; // Low connection

        let initial_coherence = field.coherence;
        let initial_resonance = field.relational_resonance;

        field.receive_gratitude();

        assert!(
            field.coherence > initial_coherence,
            "Gratitude should increase coherence"
        );
        assert!(
            field.relational_resonance > initial_resonance,
            "Gratitude should increase resonance"
        );
        assert_eq!(field.gratitude_count, 1);
    }

    #[test]
    fn test_gratitude_more_effective_when_scattered() {
        let mut field1 = CoherenceField::new();
        field1.coherence = 0.3; // Very scattered

        let mut field2 = CoherenceField::new();
        field2.coherence = 0.8; // Already coherent

        field1.receive_gratitude();
        field2.receive_gratitude();

        let boost1 = field1.coherence - 0.3;
        let boost2 = field2.coherence - 0.8;

        assert!(
            boost1 > boost2,
            "Gratitude should be more effective when scattered"
        );
    }

    #[test]
    fn test_insufficient_coherence_error() {
        let mut field = CoherenceField::new();
        field.coherence = 0.2; // Too low for learning

        let result = field.perform_task(TaskComplexity::Learning, true);

        assert!(result.is_err());
        match result {
            Err(CoherenceError::InsufficientCoherence {
                current,
                required,
                message,
            }) => {
                assert!(current < required);
                assert!(!message.is_empty());
            }
            _ => panic!("Expected InsufficientCoherence error"),
        }
    }

    #[test]
    fn test_passive_centering() {
        let mut field = CoherenceField::new();
        field.coherence = 0.5;

        let initial = field.coherence;

        // Simulate 10 seconds of passive rest
        field.tick(10.0);

        assert!(
            field.coherence > initial,
            "Passive rest should increase coherence"
        );
    }

    #[test]
    fn test_sleep_cycle_restoration() {
        let mut field = CoherenceField::new();
        field.coherence = 0.3;
        field.relational_resonance = 0.8;

        field.sleep_cycle();

        assert_eq!(field.coherence, 1.0, "Sleep should fully restore coherence");
        assert!(
            field.relational_resonance < 0.8,
            "Sleep should slightly decay resonance"
        );
    }

    #[test]
    fn test_task_complexity_thresholds() {
        let config = CoherenceConfig::default();

        assert_eq!(TaskComplexity::Reflex.required_coherence(&config), 0.1);
        assert_eq!(TaskComplexity::Cognitive.required_coherence(&config), 0.3);
        assert_eq!(TaskComplexity::DeepThought.required_coherence(&config), 0.5);
        assert_eq!(TaskComplexity::Empathy.required_coherence(&config), 0.7);
        assert_eq!(TaskComplexity::Learning.required_coherence(&config), 0.8);
        assert_eq!(TaskComplexity::Creation.required_coherence(&config), 0.9);
    }

    #[test]
    fn test_resonance_decay_over_time() {
        let mut field = CoherenceField::new();
        field.relational_resonance = 0.9;

        sleep(Duration::from_millis(100));

        field.tick(0.1);

        // Resonance should decay slightly
        assert!(field.relational_resonance < 0.9);
    }

    #[test]
    fn test_stats() {
        let mut field = CoherenceField::new();
        field.perform_task(TaskComplexity::Cognitive, true).unwrap();
        field.receive_gratitude();

        let stats = field.stats();

        assert_eq!(stats.operations_count, 1);
        assert_eq!(stats.gratitude_count, 1);
        assert!(!stats.status.is_empty());
    }

    // ========================================================================
    // WEEK 9 PHASE 1: Predictive Coherence Tests
    // ========================================================================

    #[test]
    fn test_predict_solo_task_accuracy() {
        use crate::physiology::endocrine::HormoneState;

        let mut field = CoherenceField::new();
        field.coherence = 0.8;
        field.relational_resonance = 0.3;

        let hormones = HormoneState::neutral();

        // Predict the impact of a cognitive solo task
        let prediction = field.predict_impact(TaskComplexity::Cognitive, false, &hormones);

        // Now actually perform the task
        let _before = field.coherence;
        field
            .perform_task(TaskComplexity::Cognitive, false)
            .unwrap();
        let after = field.coherence;

        // Prediction should be very close to actual result (within 0.01)
        let actual_change = after;
        assert!(
            (prediction.final_coherence - actual_change).abs() < 0.01,
            "Predicted {}, actual {}",
            prediction.final_coherence,
            actual_change
        );
    }

    #[test]
    fn test_predict_connected_task_accuracy() {
        use crate::physiology::endocrine::HormoneState;

        let mut field = CoherenceField::new();
        field.coherence = 0.6;
        field.relational_resonance = 0.8;

        let hormones = HormoneState::neutral();

        // Predict the impact of a connected deep thought task
        let prediction = field.predict_impact(TaskComplexity::DeepThought, true, &hormones);

        // Now actually perform the task
        field
            .perform_task(TaskComplexity::DeepThought, true)
            .unwrap();
        let actual = field.coherence;

        // Prediction should match (within 0.01)
        assert!(
            (prediction.final_coherence - actual).abs() < 0.01,
            "Predicted {}, actual {}",
            prediction.final_coherence,
            actual
        );

        // Connected work should BUILD coherence
        assert!(prediction.will_succeed);
        assert_eq!(prediction.centering_needed, 0.0);
    }

    #[test]
    fn test_predict_insufficient_coherence() {
        use crate::physiology::endocrine::HormoneState;

        let mut field = CoherenceField::new();
        field.coherence = 0.2; // Too low for learning (requires 0.8)
        field.relational_resonance = 0.5;

        let hormones = HormoneState::neutral();

        // Predict learning task (requires 0.8 coherence)
        let prediction = field.predict_impact(TaskComplexity::Learning, false, &hormones);

        // Should predict failure
        assert!(
            !prediction.will_succeed,
            "Should predict failure for insufficient coherence"
        );
        assert!(
            prediction.centering_needed > 0.0,
            "Should recommend centering"
        );
        assert!(
            prediction.final_coherence < 0.8,
            "Final coherence should be below threshold"
        );
    }

    #[test]
    fn test_predict_centering_calculation() {
        use crate::physiology::endocrine::HormoneState;

        let mut field = CoherenceField::new();
        field.coherence = 0.5; // Moderate coherence
        field.relational_resonance = 0.5;

        let hormones = HormoneState::neutral();

        // Predict a creation task (requires 0.9 coherence)
        let prediction = field.predict_impact(TaskComplexity::Creation, false, &hormones);

        // Should need centering
        assert!(!prediction.will_succeed);
        assert!(prediction.centering_needed > 0.0);

        // Centering time should be reasonable (deficit / centering_rate)
        let deficit = 0.9 - prediction.final_coherence;
        let expected_time = deficit / field.config.passive_centering_rate;
        assert!(
            (prediction.centering_needed - expected_time).abs() < 1.0,
            "Centering time calculation: predicted {}, expected ~{}",
            prediction.centering_needed,
            expected_time
        );
    }

    #[test]
    fn test_analyze_task_queue() {
        use crate::physiology::endocrine::HormoneState;

        let mut field = CoherenceField::new();
        field.coherence = 0.8;
        field.relational_resonance = 0.7;

        let hormones = HormoneState::neutral();

        // Queue of tasks: 3 cognitive solo, then 1 empathy connected
        let tasks = vec![
            (TaskComplexity::Cognitive, false),
            (TaskComplexity::Cognitive, false),
            (TaskComplexity::Cognitive, false),
            (TaskComplexity::Empathy, true), // This should RESTORE coherence!
        ];

        let predictions = field.analyze_task_queue(&tasks, &hormones);

        assert_eq!(predictions.len(), 4, "Should predict all 4 tasks");

        // First 3 tasks should gradually scatter coherence
        assert!(predictions[0].final_coherence < field.coherence);
        assert!(predictions[1].final_coherence < predictions[0].final_coherence);
        assert!(predictions[2].final_coherence < predictions[1].final_coherence);

        // Fourth task (connected empathy) should BUILD coherence!
        assert!(
            predictions[3].final_coherence > predictions[2].final_coherence,
            "Connected empathy work should restore coherence: {} -> {}",
            predictions[2].final_coherence,
            predictions[3].final_coherence
        );
    }
}
