// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Client psychological state tracking.
//!
//! Maintains a longitudinal model of the client's psychological state including
//! current affect, RDoC dimensional profile, diagnostic hypotheses, presenting
//! concerns, and CBT/narrative formulation elements.
//!
//! Science: Persons (2008) case formulation, Beck (1979) cognitive model,
//! Borsboom (2017) network theory, Fried & Nesse (2015) symptom specificity.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use symthaea_clinical::{DiagnosticProfile, RDocDomain, RDocProfile, SymptomProfile};
use symthaea_core::hdc::BinaryHV;

/// Maximum number of affect snapshots retained in trajectory.
const MAX_AFFECT_TRAJECTORY: usize = 256;

/// Maximum number of symptom snapshots retained.
const MAX_SYMPTOM_TRAJECTORY: usize = 64;

// ── Core Affect Snapshot ───────────────────────────────────────────────────

/// Snapshot of client's affective state at a point in time.
///
/// Mirrors Symthaea's CoreAffect but from the *client's* perspective.
/// Note: neutral arousal is 0.5, not 0.0 (matching CoreAffect::neutral()).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoreAffectSnapshot {
    /// Positive-negative (-1 to +1)
    pub valence: f32,
    /// Low-high activation (0 to 1, neutral = 0.5)
    pub arousal: f32,
    /// Cycle number when captured
    pub cycle: u64,
}

impl CoreAffectSnapshot {
    /// Create a new affect snapshot.
    pub fn new(valence: f32, arousal: f32, cycle: u64) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            cycle,
        }
    }

    /// Neutral baseline.
    pub fn neutral(cycle: u64) -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            cycle,
        }
    }

    /// Distress level (0-1): combines negative valence with high arousal.
    pub fn distress(&self) -> f32 {
        let neg_valence = (-self.valence).max(0.0);
        let high_arousal = (self.arousal - 0.5).max(0.0) * 2.0;
        (neg_valence * 0.6 + high_arousal * 0.4).clamp(0.0, 1.0)
    }
}

// ── Risk Level ─────────────────────────────────────────────────────────────

/// Client risk assessment level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum RiskLevel {
    /// No identified risk factors
    None,
    /// Some risk factors present but no imminent danger
    Low,
    /// Significant risk factors, needs monitoring
    Moderate,
    /// Imminent risk, requires immediate intervention
    High,
    /// Active crisis — engage emergency protocols
    Critical,
}

// ── Client Model ───────────────────────────────────────────────────────────

/// Longitudinal model of client psychological state.
///
/// Tracks current state, historical trajectory, CBT formulation elements,
/// and narrative fragments for a holistic therapeutic picture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientModel {
    // ── Current State ──
    /// Most recent affect snapshot.
    pub current_affect: CoreAffectSnapshot,
    /// Dimensional RDoC profile.
    pub rdoc_profile: RDocProfile,
    /// Active diagnostic hypotheses (ranked by confidence).
    pub diagnostic_hypotheses: Vec<DiagnosticProfile>,
    /// Presenting concerns as HDC-encoded vectors.
    #[serde(skip)]
    pub presenting_concerns: Vec<BinaryHV>,

    // ── Longitudinal Tracking ──
    /// Affect trajectory (ring buffer).
    pub affect_trajectory: VecDeque<CoreAffectSnapshot>,
    /// Symptom profile trajectory.
    pub symptom_trajectory: VecDeque<SymptomProfile>,
    /// Total session count.
    pub session_count: u32,
    /// Total cycles processed.
    pub cycle_count: u64,

    // ── CBT Formulation Elements ──
    /// Identified automatic thoughts (HDC-encoded).
    #[serde(skip)]
    pub automatic_thoughts: Vec<BinaryHV>,
    /// Core beliefs (HDC-encoded).
    #[serde(skip)]
    pub core_beliefs: Vec<BinaryHV>,
    /// Behavioral patterns (text labels).
    pub behavioral_patterns: Vec<String>,

    // ── Safety ──
    /// Current risk assessment level.
    pub risk_level: RiskLevel,
}

impl ClientModel {
    /// Create a new client model with neutral baseline.
    pub fn new() -> Self {
        Self {
            current_affect: CoreAffectSnapshot::neutral(0),
            rdoc_profile: RDocProfile::default(),
            diagnostic_hypotheses: Vec::new(),
            presenting_concerns: Vec::new(),
            affect_trajectory: VecDeque::new(),
            symptom_trajectory: VecDeque::new(),
            session_count: 0,
            cycle_count: 0,
            automatic_thoughts: Vec::new(),
            core_beliefs: Vec::new(),
            behavioral_patterns: Vec::new(),
            risk_level: RiskLevel::None,
        }
    }

    /// Update client affect from a new observation.
    pub fn update_affect(&mut self, snapshot: CoreAffectSnapshot) {
        self.current_affect = snapshot;
        self.affect_trajectory.push_back(snapshot);
        if self.affect_trajectory.len() > MAX_AFFECT_TRAJECTORY {
            self.affect_trajectory.pop_front();
        }
        self.cycle_count = snapshot.cycle;
    }

    /// Update symptom profile.
    pub fn update_symptoms(&mut self, profile: SymptomProfile) {
        self.symptom_trajectory.push_back(profile);
        if self.symptom_trajectory.len() > MAX_SYMPTOM_TRAJECTORY {
            self.symptom_trajectory.pop_front();
        }
    }

    /// Current distress level (0-1).
    pub fn distress(&self) -> f32 {
        self.current_affect.distress()
    }

    /// Affect trend: positive = improving, negative = worsening.
    ///
    /// Compares recent valence (last 10) to earlier valence (10 before that).
    pub fn affect_trend(&self) -> f32 {
        let len = self.affect_trajectory.len();
        if len < 20 {
            return 0.0;
        }
        let recent: f32 = self
            .affect_trajectory
            .iter()
            .rev()
            .take(10)
            .map(|a| a.valence)
            .sum::<f32>()
            / 10.0;
        let earlier: f32 = self
            .affect_trajectory
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .map(|a| a.valence)
            .sum::<f32>()
            / 10.0;
        recent - earlier
    }

    /// Mean arousal over the trajectory.
    pub fn mean_arousal(&self) -> f32 {
        if self.affect_trajectory.is_empty() {
            return 0.5;
        }
        let sum: f32 = self.affect_trajectory.iter().map(|a| a.arousal).sum();
        sum / self.affect_trajectory.len() as f32
    }

    /// Update RDoC profile for a specific domain.
    pub fn update_rdoc(&mut self, domain: RDocDomain, score: f32) {
        self.rdoc_profile.set_score(domain, score);
    }

    /// Add a diagnostic hypothesis.
    pub fn add_hypothesis(&mut self, profile: DiagnosticProfile) {
        self.diagnostic_hypotheses.push(profile);
        // Keep sorted by confidence (descending)
        self.diagnostic_hypotheses.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Start a new session.
    pub fn begin_session(&mut self) {
        self.session_count += 1;
    }

    /// Clinical severity from RDoC profile.
    pub fn clinical_severity(&self) -> f32 {
        self.rdoc_profile.clinical_severity()
    }

    /// Update RDoC profile from sustained affect patterns.
    ///
    /// Called periodically to make the RDoC profile responsive to ongoing
    /// emotional patterns rather than staying at defaults.
    ///
    /// Mapping (EMA blend, alpha=0.02 for slow adaptation):
    /// - Sustained negative valence → NegativeValence ↑
    /// - Sustained positive valence → PositiveValence ↑
    /// - Arousal dysregulation (far from 0.5) → ArousalRegulatory ↑
    /// - Low arousal variance → CognitiveSystems stable (no change)
    ///
    /// Science: Insel et al. (2010) — RDoC dimensions track continuous state.
    pub fn update_rdoc_from_affect(&mut self) {
        let window = 30;
        if self.affect_trajectory.len() < window {
            return;
        }

        let alpha = 0.02_f32; // Slow EMA for stability
        let recent: Vec<&CoreAffectSnapshot> =
            self.affect_trajectory.iter().rev().take(window).collect();

        // Mean negative valence in window (higher = more negative affect)
        let mean_neg = recent.iter().map(|a| (-a.valence).max(0.0)).sum::<f32>() / window as f32;
        let current_neg = self.rdoc_profile.score(RDocDomain::NegativeValence);
        self.rdoc_profile.set_score(
            RDocDomain::NegativeValence,
            current_neg * (1.0 - alpha) + mean_neg * alpha,
        );

        // Mean positive valence (higher = more positive affect)
        let mean_pos = recent.iter().map(|a| a.valence.max(0.0)).sum::<f32>() / window as f32;
        let current_pos = self.rdoc_profile.score(RDocDomain::PositiveValence);
        self.rdoc_profile.set_score(
            RDocDomain::PositiveValence,
            current_pos * (1.0 - alpha) + mean_pos * alpha,
        );

        // Arousal dysregulation: deviation from 0.5 baseline
        let mean_arousal_dev = recent
            .iter()
            .map(|a| (a.arousal - 0.5).abs() * 2.0)
            .sum::<f32>()
            / window as f32;
        let current_arousal = self.rdoc_profile.score(RDocDomain::ArousalRegulatory);
        self.rdoc_profile.set_score(
            RDocDomain::ArousalRegulatory,
            current_arousal * (1.0 - alpha) + mean_arousal_dev * alpha,
        );
    }

    // ── Outcome Measurement ─────────────────────────────────────────────

    /// PHQ-9 analogue score (0–27): depression severity from affect trajectory.
    ///
    /// Maps sustained negative valence, low positive valence, arousal dysregulation,
    /// and functional impairment to the PHQ-9 scale. NOT a clinical instrument —
    /// this is a computational analogue for tracking treatment progress.
    ///
    /// Scoring: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe.
    ///
    /// Science: Kroenke, Spitzer & Williams (2001). The PHQ-9: validity of a brief
    /// depression severity measure. *Journal of General Internal Medicine*, 16(9), 606-613.
    pub fn phq9_analogue(&self) -> f32 {
        if self.affect_trajectory.len() < 10 {
            return 0.0; // Insufficient data
        }

        let window = self.affect_trajectory.len().min(50);
        let recent: Vec<&CoreAffectSnapshot> =
            self.affect_trajectory.iter().rev().take(window).collect();

        // PHQ-9 domains mapped to computational signals:
        // 1. Little interest/pleasure → low positive valence (anhedonia)
        let mean_pos = recent.iter().map(|a| a.valence.max(0.0)).sum::<f32>() / window as f32;
        let anhedonia = (1.0 - mean_pos * 2.0).clamp(0.0, 1.0); // 0=engaged, 1=anhedonic

        // 2. Feeling down/depressed → sustained negative valence
        let mean_neg = recent.iter().map(|a| (-a.valence).max(0.0)).sum::<f32>() / window as f32;
        let depressed_mood = mean_neg.clamp(0.0, 1.0);

        // 3. Sleep disturbance → arousal dysregulation (proxy)
        let arousal_variance = {
            let mean_arousal = recent.iter().map(|a| a.arousal).sum::<f32>() / window as f32;
            let var = recent
                .iter()
                .map(|a| (a.arousal - mean_arousal).powi(2))
                .sum::<f32>()
                / window as f32;
            var.sqrt()
        };
        let sleep_disturbance = (arousal_variance * 4.0).clamp(0.0, 1.0);

        // 4. Fatigue → low arousal (sustained)
        let mean_arousal = recent.iter().map(|a| a.arousal).sum::<f32>() / window as f32;
        let fatigue = ((0.5 - mean_arousal).max(0.0) * 3.0).clamp(0.0, 1.0);

        // 5. Poor concentration → from RDoC CognitiveSystems
        let cognitive_impairment = self.rdoc_profile.score(RDocDomain::CognitiveSystems);

        // 6. Psychomotor → extreme arousal (too high OR too low)
        let psychomotor = ((mean_arousal - 0.5).abs() * 2.5).clamp(0.0, 1.0);

        // 7. Worthlessness → RDoC NegativeValence + low social
        let neg_val = self.rdoc_profile.score(RDocDomain::NegativeValence);
        let social = self.rdoc_profile.score(RDocDomain::SocialProcesses);
        let worthlessness = ((neg_val + (1.0 - social)) * 0.5).clamp(0.0, 1.0);

        // 8-9. Suicidal ideation → risk level
        let suicidal = match self.risk_level {
            RiskLevel::None | RiskLevel::Low => 0.0,
            RiskLevel::Moderate => 0.33,
            RiskLevel::High => 0.67,
            RiskLevel::Critical => 1.0,
        };

        // Each domain scored 0-3 (PHQ-9 Likert), sum across 9 items → 0-27
        let items = [
            anhedonia,
            depressed_mood,
            sleep_disturbance,
            fatigue,
            cognitive_impairment,
            psychomotor,
            worthlessness,
            suicidal,
            // 9th item: overall functional impairment (mean distress)
            self.distress(),
        ];
        items.iter().map(|d| d * 3.0).sum::<f32>().clamp(0.0, 27.0)
    }

    /// GAD-7 analogue score (0–21): anxiety severity from affect trajectory.
    ///
    /// Maps sustained high arousal, arousal dysregulation, and RDoC negative valence
    /// to the GAD-7 scale. NOT a clinical instrument — computational analogue.
    ///
    /// Scoring: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe.
    ///
    /// Science: Spitzer, Kroenke, Williams & Löwe (2006). A brief measure for
    /// assessing generalized anxiety disorder. *Archives of Internal Medicine*, 166(10), 1092-1097.
    pub fn gad7_analogue(&self) -> f32 {
        if self.affect_trajectory.len() < 10 {
            return 0.0;
        }

        let window = self.affect_trajectory.len().min(50);
        let recent: Vec<&CoreAffectSnapshot> =
            self.affect_trajectory.iter().rev().take(window).collect();

        // GAD-7 domains:
        // 1. Feeling nervous/anxious → high arousal + negative valence
        let mean_arousal = recent.iter().map(|a| a.arousal).sum::<f32>() / window as f32;
        let mean_neg = recent.iter().map(|a| (-a.valence).max(0.0)).sum::<f32>() / window as f32;
        let nervous = ((mean_arousal - 0.5).max(0.0) * 2.0 * 0.5 + mean_neg * 0.5).clamp(0.0, 1.0);

        // 2. Not being able to stop worrying → RDoC NegativeValence persistence
        let neg_val = self.rdoc_profile.score(RDocDomain::NegativeValence);
        let worry = neg_val.clamp(0.0, 1.0);

        // 3. Worrying too much → arousal + negative valence combined
        let excessive_worry = (nervous * 0.7 + worry * 0.3).clamp(0.0, 1.0);

        // 4. Trouble relaxing → arousal dysregulation
        let arousal_dysreg = self.rdoc_profile.score(RDocDomain::ArousalRegulatory);
        let trouble_relaxing = arousal_dysreg.clamp(0.0, 1.0);

        // 5. Restlessness → high arousal sustained
        let restless = ((mean_arousal - 0.6).max(0.0) * 3.0).clamp(0.0, 1.0);

        // 6. Easily annoyed/irritable → high arousal + negative valence + arousal variance
        let arousal_var = {
            let var = recent
                .iter()
                .map(|a| (a.arousal - mean_arousal).powi(2))
                .sum::<f32>()
                / window as f32;
            var.sqrt()
        };
        let irritable = (mean_neg * 0.4 + arousal_var * 2.0 * 0.3 + restless * 0.3).clamp(0.0, 1.0);

        // 7. Feeling afraid → RDoC NegativeValence (fear component)
        let afraid = (neg_val * 0.6 + mean_neg * 0.4).clamp(0.0, 1.0);

        let items = [
            nervous,
            worry,
            excessive_worry,
            trouble_relaxing,
            restless,
            irritable,
            afraid,
        ];
        items.iter().map(|d| d * 3.0).sum::<f32>().clamp(0.0, 21.0)
    }

    /// Outcome Rating Scale analogue (0-40): overall functioning.
    ///
    /// Maps positive valence, low distress, social engagement, and arousal
    /// regulation to a composite functioning score. Higher = better.
    ///
    /// Science: Miller, Duncan, Brown, Sparks & Claud (2003). The Outcome Rating
    /// Scale: A preliminary study of the reliability, validity, and feasibility.
    /// *Journal of Brief Therapy*, 2(2), 91-100.
    pub fn ors_analogue(&self) -> f32 {
        if self.affect_trajectory.len() < 10 {
            return 20.0; // Neutral midpoint
        }

        let window = self.affect_trajectory.len().min(50);
        let recent: Vec<&CoreAffectSnapshot> =
            self.affect_trajectory.iter().rev().take(window).collect();

        // ORS 4 domains (each 0-10):
        // 1. Individual wellbeing → valence
        let mean_valence = recent.iter().map(|a| a.valence).sum::<f32>() / window as f32;
        let individual = ((mean_valence + 1.0) / 2.0 * 10.0).clamp(0.0, 10.0);

        // 2. Interpersonal → RDoC SocialProcesses
        let social = self.rdoc_profile.score(RDocDomain::SocialProcesses);
        let interpersonal = (social * 10.0).clamp(0.0, 10.0);

        // 3. Social role → inverse of distress + arousal regulation
        let distress = self.distress();
        let social_role = ((1.0 - distress) * 10.0).clamp(0.0, 10.0);

        // 4. Overall → composite of above
        let pos_valence = self.rdoc_profile.score(RDocDomain::PositiveValence);
        let overall = ((pos_valence * 0.5 + (1.0 - distress) * 0.5) * 10.0).clamp(0.0, 10.0);

        individual + interpersonal + social_role + overall
    }

    /// Therapeutic outcome summary combining all analogue measures.
    ///
    /// Returns (phq9, gad7, ors, trend) where trend is computed from
    /// comparing current vs earlier window means.
    pub fn outcome_summary(&self) -> OutcomeSummary {
        OutcomeSummary {
            phq9: self.phq9_analogue(),
            gad7: self.gad7_analogue(),
            ors: self.ors_analogue(),
            affect_trend: self.affect_trend(),
            cycles_observed: self.cycle_count,
            sessions: self.session_count,
        }
    }
}

/// Therapeutic outcome summary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OutcomeSummary {
    /// PHQ-9 analogue (0-27). Lower = better. <5 minimal, 5-9 mild, 10-14 moderate.
    pub phq9: f32,
    /// GAD-7 analogue (0-21). Lower = better. <5 minimal, 5-9 mild, 10-14 moderate.
    pub gad7: f32,
    /// ORS analogue (0-40). Higher = better. <25 clinical concern, >25 adequate functioning.
    pub ors: f32,
    /// Affect trend (positive = improving).
    pub affect_trend: f32,
    /// Cycles observed (data quality indicator).
    pub cycles_observed: u64,
    /// Sessions completed.
    pub sessions: u32,
}

impl Default for ClientModel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_clinical::{DiagnosticCategory, Severity};

    #[test]
    fn test_new_client_neutral() {
        let client = ClientModel::new();
        assert_eq!(client.current_affect.valence, 0.0);
        assert_eq!(client.current_affect.arousal, 0.5);
        assert_eq!(client.risk_level, RiskLevel::None);
        assert_eq!(client.session_count, 0);
    }

    #[test]
    fn test_update_affect() {
        let mut client = ClientModel::new();
        let snap = CoreAffectSnapshot::new(-0.5, 0.8, 1);
        client.update_affect(snap);
        assert_eq!(client.current_affect.valence, -0.5);
        assert_eq!(client.affect_trajectory.len(), 1);
        assert_eq!(client.cycle_count, 1);
    }

    #[test]
    fn test_affect_trajectory_bounded() {
        let mut client = ClientModel::new();
        for i in 0..300 {
            client.update_affect(CoreAffectSnapshot::new(0.0, 0.5, i));
        }
        assert!(client.affect_trajectory.len() <= MAX_AFFECT_TRAJECTORY);
    }

    #[test]
    fn test_distress_high_for_negative_aroused() {
        let snap = CoreAffectSnapshot::new(-0.8, 0.9, 0);
        assert!(snap.distress() > 0.5);
    }

    #[test]
    fn test_distress_low_for_positive_calm() {
        let snap = CoreAffectSnapshot::new(0.5, 0.3, 0);
        assert!(snap.distress() < 0.2);
    }

    #[test]
    fn test_affect_trend_no_data() {
        let client = ClientModel::new();
        assert_eq!(client.affect_trend(), 0.0);
    }

    #[test]
    fn test_affect_trend_improving() {
        let mut client = ClientModel::new();
        // 10 negative snapshots then 10 positive
        for i in 0..10 {
            client.update_affect(CoreAffectSnapshot::new(-0.5, 0.5, i));
        }
        for i in 10..20 {
            client.update_affect(CoreAffectSnapshot::new(0.5, 0.5, i));
        }
        assert!(client.affect_trend() > 0.0);
    }

    #[test]
    fn test_add_hypothesis_sorted() {
        let mut client = ClientModel::new();
        client.add_hypothesis(DiagnosticProfile::new(
            DiagnosticCategory::Anxiety,
            Severity::Mild,
            0.5,
        ));
        client.add_hypothesis(DiagnosticProfile::new(
            DiagnosticCategory::Mood,
            Severity::Moderate,
            0.8,
        ));
        assert_eq!(client.diagnostic_hypotheses[0].confidence, 0.8);
    }

    #[test]
    fn test_begin_session() {
        let mut client = ClientModel::new();
        client.begin_session();
        client.begin_session();
        assert_eq!(client.session_count, 2);
    }

    #[test]
    fn test_mean_arousal() {
        let mut client = ClientModel::new();
        client.update_affect(CoreAffectSnapshot::new(0.0, 0.2, 0));
        client.update_affect(CoreAffectSnapshot::new(0.0, 0.8, 1));
        assert!((client.mean_arousal() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_update_rdoc_from_affect_needs_window() {
        let mut client = ClientModel::new();
        let pre = client.rdoc_profile.score(RDocDomain::NegativeValence);
        // Not enough data (< 30 snapshots)
        for i in 0..10 {
            client.update_affect(CoreAffectSnapshot::new(-0.9, 0.9, i));
        }
        client.update_rdoc_from_affect();
        // Should not change — insufficient window
        assert_eq!(client.rdoc_profile.score(RDocDomain::NegativeValence), pre);
    }

    #[test]
    fn test_update_rdoc_from_affect_negative_valence() {
        let mut client = ClientModel::new();
        let pre_neg = client.rdoc_profile.score(RDocDomain::NegativeValence);
        // 40 cycles of strong negative affect
        for i in 0..40 {
            client.update_affect(CoreAffectSnapshot::new(-0.8, 0.5, i));
        }
        client.update_rdoc_from_affect();
        let post_neg = client.rdoc_profile.score(RDocDomain::NegativeValence);
        assert!(
            post_neg > pre_neg,
            "NegativeValence should increase with sustained negative affect: {} → {}",
            pre_neg,
            post_neg,
        );
    }

    #[test]
    fn test_update_rdoc_from_affect_positive_valence() {
        let mut client = ClientModel::new();
        let pre_pos = client.rdoc_profile.score(RDocDomain::PositiveValence);
        // 40 cycles of strong positive affect
        for i in 0..40 {
            client.update_affect(CoreAffectSnapshot::new(0.8, 0.5, i));
        }
        client.update_rdoc_from_affect();
        let post_pos = client.rdoc_profile.score(RDocDomain::PositiveValence);
        assert!(
            post_pos > pre_pos,
            "PositiveValence should increase with sustained positive affect: {} → {}",
            pre_pos,
            post_pos,
        );
    }

    // ── Outcome Measure Tests ──────────────────────────────────────────

    #[test]
    fn test_phq9_minimal_for_positive_affect() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(0.6, 0.5, i));
        }
        let phq9 = client.phq9_analogue();
        assert!(
            phq9 < 5.0,
            "Positive affect should yield minimal PHQ-9: {}",
            phq9
        );
    }

    #[test]
    fn test_phq9_elevated_for_depressed_affect() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(-0.8, 0.3, i));
        }
        let phq9 = client.phq9_analogue();
        assert!(
            phq9 > 8.0,
            "Depressed affect should yield elevated PHQ-9: {}",
            phq9
        );
    }

    #[test]
    fn test_phq9_range() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(-0.9, 0.9, i));
        }
        client.risk_level = RiskLevel::High;
        let phq9 = client.phq9_analogue();
        assert!(
            phq9 >= 0.0 && phq9 <= 27.0,
            "PHQ-9 must be in 0-27: {}",
            phq9
        );
    }

    #[test]
    fn test_phq9_insufficient_data() {
        let mut client = ClientModel::new();
        for i in 0..5 {
            client.update_affect(CoreAffectSnapshot::new(-0.9, 0.9, i));
        }
        assert_eq!(client.phq9_analogue(), 0.0, "Insufficient data returns 0");
    }

    #[test]
    fn test_gad7_minimal_for_calm_affect() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(0.3, 0.4, i));
        }
        let gad7 = client.gad7_analogue();
        assert!(
            gad7 < 5.0,
            "Calm positive affect should yield minimal GAD-7: {}",
            gad7
        );
    }

    #[test]
    fn test_gad7_elevated_for_anxious_affect() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(-0.6, 0.9, i));
        }
        client.update_rdoc(RDocDomain::NegativeValence, 0.8);
        let gad7 = client.gad7_analogue();
        assert!(
            gad7 > 8.0,
            "Anxious affect should yield elevated GAD-7: {}",
            gad7
        );
    }

    #[test]
    fn test_gad7_range() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(-0.9, 0.95, i));
        }
        client.update_rdoc(RDocDomain::NegativeValence, 1.0);
        client.update_rdoc(RDocDomain::ArousalRegulatory, 1.0);
        let gad7 = client.gad7_analogue();
        assert!(
            gad7 >= 0.0 && gad7 <= 21.0,
            "GAD-7 must be in 0-21: {}",
            gad7
        );
    }

    #[test]
    fn test_ors_high_for_positive_functioning() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(0.7, 0.5, i));
        }
        client.update_rdoc(RDocDomain::SocialProcesses, 0.8);
        client.update_rdoc(RDocDomain::PositiveValence, 0.7);
        let ors = client.ors_analogue();
        assert!(
            ors > 25.0,
            "Good functioning should yield high ORS: {}",
            ors
        );
    }

    #[test]
    fn test_ors_low_for_impaired_functioning() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(-0.8, 0.9, i));
        }
        client.update_rdoc(RDocDomain::SocialProcesses, 0.1);
        let ors = client.ors_analogue();
        assert!(
            ors < 20.0,
            "Impaired functioning should yield low ORS: {}",
            ors
        );
    }

    #[test]
    fn test_outcome_summary() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(-0.3, 0.6, i));
        }
        client.begin_session();
        let summary = client.outcome_summary();
        assert!(summary.phq9 >= 0.0);
        assert!(summary.gad7 >= 0.0);
        assert!(summary.ors >= 0.0 && summary.ors <= 40.0);
        assert_eq!(summary.sessions, 1);
        assert_eq!(summary.cycles_observed, 49);
    }

    #[test]
    fn test_update_rdoc_from_affect_arousal_dysregulation() {
        let mut client = ClientModel::new();
        let pre_arousal = client.rdoc_profile.score(RDocDomain::ArousalRegulatory);
        // 40 cycles of extreme arousal (far from 0.5 baseline)
        for i in 0..40 {
            client.update_affect(CoreAffectSnapshot::new(0.0, 0.95, i));
        }
        client.update_rdoc_from_affect();
        let post_arousal = client.rdoc_profile.score(RDocDomain::ArousalRegulatory);
        assert!(
            post_arousal > pre_arousal,
            "ArousalRegulatory should increase with dysregulated arousal: {} → {}",
            pre_arousal,
            post_arousal,
        );
    }
}
