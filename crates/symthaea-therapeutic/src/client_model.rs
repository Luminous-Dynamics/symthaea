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
