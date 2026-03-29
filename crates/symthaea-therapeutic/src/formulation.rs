// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Clinical case formulation (CBT 4P model).
//!
//! Structures client information into a comprehensive understanding of
//! predisposing, precipitating, perpetuating, and protective factors.
//!
//! Science: Persons (2008) case formulation, Johnstone & Dallos (2013) 5P model,
//! Beck (1979) cognitive model (core beliefs → intermediate → automatic thoughts).

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::BinaryHV;

// ── Factor Types ───────────────────────────────────────────────────────────

/// A formulation factor with text description and HDC encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormulationFactor {
    /// Human-readable description.
    pub description: String,
    /// Confidence in this factor (0.0–1.0).
    pub confidence: f32,
    /// HDC encoding for similarity search.
    #[serde(skip)]
    pub encoding: Option<BinaryHV>,
}

impl FormulationFactor {
    /// Create a new formulation factor with HDC encoding.
    pub fn new(description: &str, confidence: f32) -> Self {
        let hash = blake3::hash(format!("factor:{}", description).as_bytes());
        let seed = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        Self {
            description: description.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
            encoding: Some(BinaryHV::random(seed)),
        }
    }
}

// ── CBT Belief Chain ───────────────────────────────────────────────────────

/// CBT cognitive model: core belief → intermediate belief → automatic thought → behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbtBeliefChain {
    /// Deep schema (e.g., "I am unlovable")
    pub core_belief: String,
    /// Conditional rule (e.g., "If I show vulnerability, I'll be rejected")
    pub intermediate_belief: String,
    /// Situation-triggered thought (e.g., "They'll think I'm weak")
    pub automatic_thought: String,
    /// Resulting behavior (e.g., "Emotional suppression")
    pub behavioral_consequence: String,
    /// Confidence in this chain (0.0–1.0)
    pub confidence: f32,
}

// ── Case Formulation ───────────────────────────────────────────────────────

/// 4P clinical case formulation.
///
/// Organizes understanding of the client's difficulties into:
/// - **Predisposing**: Historical vulnerability factors
/// - **Precipitating**: Recent triggers
/// - **Perpetuating**: Factors maintaining the current difficulties
/// - **Protective**: Strengths and resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseFormulation {
    /// Historical vulnerability factors (genetics, early experience, temperament).
    pub predisposing: Vec<FormulationFactor>,
    /// Recent triggers that activated the vulnerability.
    pub precipitating: Vec<FormulationFactor>,
    /// Factors that maintain current difficulties (avoidance, rumination, etc.).
    pub perpetuating: Vec<FormulationFactor>,
    /// Strengths and protective factors (social support, coping skills, resilience).
    pub protective: Vec<FormulationFactor>,
    /// CBT belief chains (if using cognitive formulation).
    pub belief_chains: Vec<CbtBeliefChain>,
}

impl CaseFormulation {
    /// Create an empty formulation.
    pub fn new() -> Self {
        Self {
            predisposing: Vec::new(),
            precipitating: Vec::new(),
            perpetuating: Vec::new(),
            protective: Vec::new(),
            belief_chains: Vec::new(),
        }
    }

    /// Add a predisposing factor.
    pub fn add_predisposing(&mut self, description: &str, confidence: f32) {
        self.predisposing
            .push(FormulationFactor::new(description, confidence));
    }

    /// Add a precipitating factor.
    pub fn add_precipitating(&mut self, description: &str, confidence: f32) {
        self.precipitating
            .push(FormulationFactor::new(description, confidence));
    }

    /// Add a perpetuating factor.
    pub fn add_perpetuating(&mut self, description: &str, confidence: f32) {
        self.perpetuating
            .push(FormulationFactor::new(description, confidence));
    }

    /// Add a protective factor.
    pub fn add_protective(&mut self, description: &str, confidence: f32) {
        self.protective
            .push(FormulationFactor::new(description, confidence));
    }

    /// Add a CBT belief chain.
    pub fn add_belief_chain(&mut self, chain: CbtBeliefChain) {
        self.belief_chains.push(chain);
    }

    /// Total number of factors across all categories.
    pub fn total_factors(&self) -> usize {
        self.predisposing.len()
            + self.precipitating.len()
            + self.perpetuating.len()
            + self.protective.len()
    }

    /// Protective-to-risk ratio: higher = more resilience resources.
    pub fn resilience_ratio(&self) -> f32 {
        let risk = self.predisposing.len() + self.precipitating.len() + self.perpetuating.len();
        let protective = self.protective.len();
        if risk == 0 {
            return 1.0;
        }
        protective as f32 / risk as f32
    }

    /// Whether the formulation has enough information to guide treatment.
    ///
    /// Requires at least one factor in precipitating + perpetuating + protective.
    pub fn is_actionable(&self) -> bool {
        !self.precipitating.is_empty()
            && !self.perpetuating.is_empty()
            && !self.protective.is_empty()
    }

    /// Auto-detect formulation factors from affect trajectory patterns.
    ///
    /// Detects three clinically meaningful patterns:
    /// - **Depression-like**: Sustained low valence + low arousal (negative valence mean < -0.3, arousal mean < 0.4)
    /// - **Anxiety-like**: Sustained negative valence + high arousal (negative valence mean < -0.2, arousal mean > 0.65)
    /// - **Dysregulation**: High affect variability (valence std dev > 0.4)
    ///
    /// Only adds factors not already present. Requires at least `min_window` snapshots.
    ///
    /// Science: Wichers et al. (2015) — affect dynamics as early warning signals.
    pub fn detect_patterns(
        &mut self,
        trajectory: &std::collections::VecDeque<super::client_model::CoreAffectSnapshot>,
        min_window: usize,
    ) {
        if trajectory.len() < min_window {
            return;
        }

        let n = trajectory.len() as f32;
        let mean_valence = trajectory.iter().map(|a| a.valence).sum::<f32>() / n;
        let mean_arousal = trajectory.iter().map(|a| a.arousal).sum::<f32>() / n;
        let valence_var = trajectory
            .iter()
            .map(|a| (a.valence - mean_valence).powi(2))
            .sum::<f32>()
            / n;
        let valence_std = valence_var.sqrt();

        let has_factor = |factors: &[FormulationFactor], pattern: &str| -> bool {
            factors.iter().any(|f| f.description.contains(pattern))
        };

        // Depression-like: sustained low valence + low arousal
        if mean_valence < -0.3 && mean_arousal < 0.4 && !has_factor(&self.perpetuating, "low mood")
        {
            let confidence = ((-mean_valence - 0.3) * 2.0).clamp(0.3, 0.9);
            self.add_perpetuating("persistent low mood and low activation pattern", confidence);
        }

        // Anxiety-like: negative valence + high arousal
        if mean_valence < -0.2
            && mean_arousal > 0.65
            && !has_factor(&self.perpetuating, "anxious activation")
        {
            let confidence = ((mean_arousal - 0.65) * 3.0 + 0.3).clamp(0.3, 0.9);
            self.add_perpetuating("sustained anxious activation pattern", confidence);
        }

        // Dysregulation: high variability in valence
        if valence_std > 0.4 && !has_factor(&self.perpetuating, "affect dysregulation") {
            let confidence = ((valence_std - 0.4) * 2.0 + 0.3).clamp(0.3, 0.9);
            self.add_perpetuating(
                "affect dysregulation — high emotional variability",
                confidence,
            );
        }

        // Protective: if recent trend is improving (last quarter vs first quarter)
        if trajectory.len() >= 20 {
            let quarter = trajectory.len() / 4;
            let early_mean: f32 = trajectory
                .iter()
                .take(quarter)
                .map(|a| a.valence)
                .sum::<f32>()
                / quarter as f32;
            let late_mean: f32 = trajectory
                .iter()
                .rev()
                .take(quarter)
                .map(|a| a.valence)
                .sum::<f32>()
                / quarter as f32;
            if late_mean - early_mean > 0.2
                && !has_factor(&self.protective, "improving affect trajectory")
            {
                let confidence = ((late_mean - early_mean - 0.2) * 2.0 + 0.5).clamp(0.5, 0.9);
                self.add_protective("improving affect trajectory", confidence);
            }
        }
    }
}

impl Default for CaseFormulation {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_formulation() {
        let form = CaseFormulation::new();
        assert_eq!(form.total_factors(), 0);
        assert!(!form.is_actionable());
    }

    #[test]
    fn test_add_factors() {
        let mut form = CaseFormulation::new();
        form.add_predisposing("family history of depression", 0.8);
        form.add_precipitating("job loss", 0.9);
        form.add_perpetuating("social withdrawal", 0.7);
        form.add_protective("supportive partner", 0.9);
        assert_eq!(form.total_factors(), 4);
        assert!(form.is_actionable());
    }

    #[test]
    fn test_resilience_ratio() {
        let mut form = CaseFormulation::new();
        form.add_perpetuating("avoidance", 0.8);
        form.add_perpetuating("rumination", 0.7);
        form.add_protective("social support", 0.9);
        form.add_protective("exercise routine", 0.8);
        form.add_protective("therapy engagement", 0.9);
        // 3 protective / 2 risk = 1.5
        assert!((form.resilience_ratio() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_resilience_ratio_no_risk() {
        let form = CaseFormulation::new();
        assert_eq!(form.resilience_ratio(), 1.0);
    }

    #[test]
    fn test_factor_encoding() {
        let factor = FormulationFactor::new("social withdrawal", 0.8);
        assert!(factor.encoding.is_some());
    }

    #[test]
    fn test_factor_confidence_clamped() {
        let factor = FormulationFactor::new("test", 1.5);
        assert_eq!(factor.confidence, 1.0);
    }

    #[test]
    fn test_belief_chain() {
        let mut form = CaseFormulation::new();
        form.add_belief_chain(CbtBeliefChain {
            core_belief: "I am worthless".to_string(),
            intermediate_belief: "If I fail, it proves I'm worthless".to_string(),
            automatic_thought: "I'll mess this up too".to_string(),
            behavioral_consequence: "Avoidance of challenging tasks".to_string(),
            confidence: 0.7,
        });
        assert_eq!(form.belief_chains.len(), 1);
    }

    // ── Pattern detection tests ──────────────────────────────────────────

    use crate::client_model::CoreAffectSnapshot;
    use std::collections::VecDeque;

    fn make_trajectory(valence: f32, arousal: f32, n: usize) -> VecDeque<CoreAffectSnapshot> {
        (0..n)
            .map(|i| CoreAffectSnapshot::new(valence, arousal, i as u64))
            .collect()
    }

    #[test]
    fn test_detect_depression_pattern() {
        let mut form = CaseFormulation::new();
        let traj = make_trajectory(-0.5, 0.2, 40);
        form.detect_patterns(&traj, 20);
        assert!(
            form.perpetuating
                .iter()
                .any(|f| f.description.contains("low mood")),
            "should detect depression-like pattern"
        );
    }

    #[test]
    fn test_detect_anxiety_pattern() {
        let mut form = CaseFormulation::new();
        let traj = make_trajectory(-0.4, 0.8, 40);
        form.detect_patterns(&traj, 20);
        assert!(
            form.perpetuating
                .iter()
                .any(|f| f.description.contains("anxious activation")),
            "should detect anxiety-like pattern"
        );
    }

    #[test]
    fn test_detect_dysregulation_pattern() {
        let mut form = CaseFormulation::new();
        // Alternating extreme valence → high std dev
        let mut traj: VecDeque<CoreAffectSnapshot> = VecDeque::new();
        for i in 0..40 {
            let v = if i % 2 == 0 { 0.8 } else { -0.8 };
            traj.push_back(CoreAffectSnapshot::new(v, 0.5, i as u64));
        }
        form.detect_patterns(&traj, 20);
        assert!(
            form.perpetuating
                .iter()
                .any(|f| f.description.contains("dysregulation")),
            "should detect dysregulation pattern"
        );
    }

    #[test]
    fn test_detect_improving_trajectory() {
        let mut form = CaseFormulation::new();
        let mut traj: VecDeque<CoreAffectSnapshot> = VecDeque::new();
        for i in 0..40 {
            let v = -0.5 + (i as f32 / 40.0); // -0.5 to +0.5
            traj.push_back(CoreAffectSnapshot::new(v, 0.5, i as u64));
        }
        form.detect_patterns(&traj, 20);
        assert!(
            form.protective
                .iter()
                .any(|f| f.description.contains("improving")),
            "should detect improving trajectory as protective factor"
        );
    }

    #[test]
    fn test_detect_no_pattern_insufficient_data() {
        let mut form = CaseFormulation::new();
        let traj = make_trajectory(-0.5, 0.2, 5); // too few
        form.detect_patterns(&traj, 20);
        assert_eq!(
            form.perpetuating.len(),
            0,
            "should not detect with insufficient data"
        );
    }

    #[test]
    fn test_detect_no_duplicate_patterns() {
        let mut form = CaseFormulation::new();
        let traj = make_trajectory(-0.5, 0.2, 40);
        form.detect_patterns(&traj, 20);
        form.detect_patterns(&traj, 20); // second call
        let low_mood_count = form
            .perpetuating
            .iter()
            .filter(|f| f.description.contains("low mood"))
            .count();
        assert_eq!(low_mood_count, 1, "should not duplicate pattern factors");
    }

    #[test]
    fn test_neutral_trajectory_no_patterns() {
        let mut form = CaseFormulation::new();
        let traj = make_trajectory(0.0, 0.5, 40);
        form.detect_patterns(&traj, 20);
        assert!(
            form.perpetuating.is_empty(),
            "neutral trajectory should not trigger patterns"
        );
        assert!(form.protective.is_empty());
    }
}
