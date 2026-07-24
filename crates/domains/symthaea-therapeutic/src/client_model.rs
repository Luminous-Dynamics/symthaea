// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Client psychological state tracking.
//!
//! Maintains a longitudinal model of the client's psychological state including
//! current affect, RDoC dimensional profile, presenting concerns, and
//! CBT/narrative formulation elements. Named diagnostic hypotheses are an
//! explicit research-only feature and are absent from default builds.
//!
//! Science: Persons (2008) case formulation, Beck (1979) cognitive model,
//! Borsboom (2017) network theory, Fried & Nesse (2015) symptom specificity.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
#[cfg(feature = "experimental-diagnostic-hypotheses")]
use symthaea_clinical::DiagnosticProfile;
use symthaea_clinical::{RDocDomain, RDocProfile, SymptomProfile};
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
    /// Research-only named diagnostic hypotheses (ranked by confidence).
    #[cfg(feature = "experimental-diagnostic-hypotheses")]
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
            #[cfg(feature = "experimental-diagnostic-hypotheses")]
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

    /// Add a named diagnostic hypothesis to the research-only model surface.
    #[cfg(feature = "experimental-diagnostic-hypotheses")]
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

    /// Continuous RDoC burden index reported by the dimensional profile.
    ///
    /// This is a model-derived tracking value, not a diagnosis or administered
    /// clinical assessment.
    pub fn rdoc_burden_index(&self) -> f32 {
        self.rdoc_profile.clinical_severity()
    }

    /// Compatibility alias for [`Self::rdoc_burden_index`].
    #[deprecated(note = "use rdoc_burden_index; this value is not clinical severity")]
    pub fn clinical_severity(&self) -> f32 {
        self.rdoc_burden_index()
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

    // ── Model-Inferred Outcome Tracking ─────────────────────────────────

    /// Model-inferred negative-affect burden in the range 0.0–1.0.
    ///
    /// Returns `None` until at least ten affect observations are available.
    /// The value is derived from affect and RDoC state; no questionnaire was
    /// administered and no diagnostic interpretation is valid.
    pub fn negative_affect_burden(&self) -> Option<f32> {
        if self.affect_trajectory.len() < 10 {
            return None;
        }

        let window = self.affect_trajectory.len().min(50);
        let recent: Vec<&CoreAffectSnapshot> =
            self.affect_trajectory.iter().rev().take(window).collect();

        let mean_pos = recent.iter().map(|a| a.valence.max(0.0)).sum::<f32>() / window as f32;
        let reduced_positive_affect = (1.0 - mean_pos * 2.0).clamp(0.0, 1.0);

        let mean_neg = recent.iter().map(|a| (-a.valence).max(0.0)).sum::<f32>() / window as f32;
        let sustained_negative_affect = mean_neg.clamp(0.0, 1.0);

        let arousal_variance = {
            let mean_arousal = recent.iter().map(|a| a.arousal).sum::<f32>() / window as f32;
            let variance = recent
                .iter()
                .map(|a| (a.arousal - mean_arousal).powi(2))
                .sum::<f32>()
                / window as f32;
            variance.sqrt()
        };
        let arousal_instability = (arousal_variance * 4.0).clamp(0.0, 1.0);

        let mean_arousal = recent.iter().map(|a| a.arousal).sum::<f32>() / window as f32;
        let low_activation = ((0.5 - mean_arousal).max(0.0) * 3.0).clamp(0.0, 1.0);
        let cognitive_burden = self.rdoc_profile.score(RDocDomain::CognitiveSystems);
        let activation_extremity = ((mean_arousal - 0.5).abs() * 2.5).clamp(0.0, 1.0);

        let negative_valence = self.rdoc_profile.score(RDocDomain::NegativeValence);
        let social_processes = self.rdoc_profile.score(RDocDomain::SocialProcesses);
        let social_negative_burden =
            ((negative_valence + (1.0 - social_processes)) * 0.5).clamp(0.0, 1.0);

        let safety_concern = match self.risk_level {
            RiskLevel::None | RiskLevel::Low => 0.0,
            RiskLevel::Moderate => 0.33,
            RiskLevel::High => 0.67,
            RiskLevel::Critical => 1.0,
        };

        let dimensions = [
            reduced_positive_affect,
            sustained_negative_affect,
            arousal_instability,
            low_activation,
            cognitive_burden,
            activation_extremity,
            social_negative_burden,
            safety_concern,
            self.distress(),
        ];
        Some((dimensions.iter().sum::<f32>() / dimensions.len() as f32).clamp(0.0, 1.0))
    }

    /// Model-inferred anxious-activation burden in the range 0.0–1.0.
    ///
    /// Returns `None` until at least ten affect observations are available.
    pub fn anxious_activation_burden(&self) -> Option<f32> {
        if self.affect_trajectory.len() < 10 {
            return None;
        }

        let window = self.affect_trajectory.len().min(50);
        let recent: Vec<&CoreAffectSnapshot> =
            self.affect_trajectory.iter().rev().take(window).collect();

        let mean_arousal = recent.iter().map(|a| a.arousal).sum::<f32>() / window as f32;
        let mean_neg = recent.iter().map(|a| (-a.valence).max(0.0)).sum::<f32>() / window as f32;
        let activated_negative_affect =
            (((mean_arousal - 0.5).max(0.0) * 2.0) * 0.5 + mean_neg * 0.5).clamp(0.0, 1.0);

        let negative_valence = self.rdoc_profile.score(RDocDomain::NegativeValence);
        let persistent_negative_affect = negative_valence.clamp(0.0, 1.0);
        let combined_activation =
            (activated_negative_affect * 0.7 + persistent_negative_affect * 0.3).clamp(0.0, 1.0);
        let regulatory_burden = self
            .rdoc_profile
            .score(RDocDomain::ArousalRegulatory)
            .clamp(0.0, 1.0);
        let restlessness = ((mean_arousal - 0.6).max(0.0) * 3.0).clamp(0.0, 1.0);

        let arousal_variability = {
            let variance = recent
                .iter()
                .map(|a| (a.arousal - mean_arousal).powi(2))
                .sum::<f32>()
                / window as f32;
            variance.sqrt()
        };
        let irritability_proxy =
            (mean_neg * 0.4 + arousal_variability * 2.0 * 0.3 + restlessness * 0.3).clamp(0.0, 1.0);
        let fear_burden = (negative_valence * 0.6 + mean_neg * 0.4).clamp(0.0, 1.0);

        let dimensions = [
            activated_negative_affect,
            persistent_negative_affect,
            combined_activation,
            regulatory_burden,
            restlessness,
            irritability_proxy,
            fear_burden,
        ];
        Some((dimensions.iter().sum::<f32>() / dimensions.len() as f32).clamp(0.0, 1.0))
    }

    /// Model-inferred functional wellbeing in the range 0.0–1.0.
    ///
    /// Returns `None` until at least ten affect observations are available.
    pub fn functional_wellbeing(&self) -> Option<f32> {
        if self.affect_trajectory.len() < 10 {
            return None;
        }

        let window = self.affect_trajectory.len().min(50);
        let recent: Vec<&CoreAffectSnapshot> =
            self.affect_trajectory.iter().rev().take(window).collect();
        let mean_valence = recent.iter().map(|a| a.valence).sum::<f32>() / window as f32;

        let individual_wellbeing = ((mean_valence + 1.0) / 2.0).clamp(0.0, 1.0);
        let interpersonal_resources = self
            .rdoc_profile
            .score(RDocDomain::SocialProcesses)
            .clamp(0.0, 1.0);
        let role_functioning = (1.0 - self.distress()).clamp(0.0, 1.0);
        let positive_valence = self.rdoc_profile.score(RDocDomain::PositiveValence);
        let overall_wellbeing =
            (positive_valence * 0.5 + (1.0 - self.distress()) * 0.5).clamp(0.0, 1.0);

        Some(
            ((individual_wellbeing
                + interpersonal_resources
                + role_functioning
                + overall_wellbeing)
                / 4.0)
                .clamp(0.0, 1.0),
        )
    }

    /// Return outcome metrics with explicit model provenance and instrument status.
    pub fn inferred_outcome_metrics(&self) -> InferredOutcomeMetrics {
        InferredOutcomeMetrics {
            negative_affect_burden: self.negative_affect_burden(),
            anxious_activation_burden: self.anxious_activation_burden(),
            functional_wellbeing: self.functional_wellbeing(),
            affect_trend: self.affect_trend(),
            observations_available: self.affect_trajectory.len(),
            observations_used: self.affect_trajectory.len().min(50),
            cycles_observed: self.cycle_count,
            sessions: self.session_count,
            source: OutcomeMetricSource::ModelInferenceFromAffectAndRDoc,
            instrument_status: InstrumentAdministrationStatus::NotAdministered,
        }
    }

    /// Compatibility-only mapping onto the PHQ-9 numerical range.
    #[cfg(feature = "legacy-clinical-scale-analogues")]
    #[deprecated(note = "not a PHQ-9; use negative_affect_burden")]
    pub fn phq9_analogue(&self) -> f32 {
        self.negative_affect_burden().unwrap_or(0.0) * 27.0
    }

    /// Compatibility-only mapping onto the GAD-7 numerical range.
    #[cfg(feature = "legacy-clinical-scale-analogues")]
    #[deprecated(note = "not a GAD-7; use anxious_activation_burden")]
    pub fn gad7_analogue(&self) -> f32 {
        self.anxious_activation_burden().unwrap_or(0.0) * 21.0
    }

    /// Compatibility-only mapping onto the ORS numerical range.
    #[cfg(feature = "legacy-clinical-scale-analogues")]
    #[deprecated(note = "not an ORS administration; use functional_wellbeing")]
    pub fn ors_analogue(&self) -> f32 {
        self.functional_wellbeing().unwrap_or(0.5) * 40.0
    }

    /// Compatibility-only summary using clinical-instrument-like ranges.
    #[cfg(feature = "legacy-clinical-scale-analogues")]
    #[deprecated(note = "use inferred_outcome_metrics with explicit provenance")]
    pub fn outcome_summary(&self) -> OutcomeSummary {
        OutcomeSummary {
            phq9: self.negative_affect_burden().unwrap_or(0.0) * 27.0,
            gad7: self.anxious_activation_burden().unwrap_or(0.0) * 21.0,
            ors: self.functional_wellbeing().unwrap_or(0.5) * 40.0,
            affect_trend: self.affect_trend(),
            cycles_observed: self.cycle_count,
            sessions: self.session_count,
        }
    }
}

/// Provenance of values in [`InferredOutcomeMetrics`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutcomeMetricSource {
    /// Derived from the internal affect trajectory and RDoC dimensional state.
    ModelInferenceFromAffectAndRDoc,
}

/// Whether a validated questionnaire was actually administered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstrumentAdministrationStatus {
    /// No questionnaire was administered; values are model-derived only.
    NotAdministered,
}

/// Neutral, normalized outcome-tracking metrics with explicit provenance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InferredOutcomeMetrics {
    /// Model-derived negative-affect burden (0.0–1.0), or `None` if insufficient data.
    pub negative_affect_burden: Option<f32>,
    /// Model-derived anxious activation (0.0–1.0), or `None` if insufficient data.
    pub anxious_activation_burden: Option<f32>,
    /// Model-derived functional wellbeing (0.0–1.0), or `None` if insufficient data.
    pub functional_wellbeing: Option<f32>,
    /// Affect trend (positive = improving).
    pub affect_trend: f32,
    /// Total affect observations currently retained.
    pub observations_available: usize,
    /// Number of observations used by the bounded calculation window.
    pub observations_used: usize,
    /// Cycles observed.
    pub cycles_observed: u64,
    /// Sessions completed.
    pub sessions: u32,
    /// How these values were produced.
    pub source: OutcomeMetricSource,
    /// Explicit statement that no named questionnaire produced these values.
    pub instrument_status: InstrumentAdministrationStatus,
}

/// Legacy clinical-scale-like outcome summary.
#[cfg(feature = "legacy-clinical-scale-analogues")]
#[deprecated(note = "use InferredOutcomeMetrics")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OutcomeSummary {
    /// Compatibility-only PHQ-9-range mapping; not an administered PHQ-9.
    pub phq9: f32,
    /// Compatibility-only GAD-7-range mapping; not an administered GAD-7.
    pub gad7: f32,
    /// Compatibility-only ORS-range mapping; not an administered ORS.
    pub ors: f32,
    /// Affect trend (positive = improving).
    pub affect_trend: f32,
    /// Cycles observed.
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
    #[cfg(feature = "experimental-diagnostic-hypotheses")]
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

    #[cfg(feature = "experimental-diagnostic-hypotheses")]
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

    // ── Model-Inferred Outcome Tests ───────────────────────────────────

    #[test]
    fn inferred_metrics_report_insufficient_data_as_missing() {
        let mut client = ClientModel::new();
        for i in 0..5 {
            client.update_affect(CoreAffectSnapshot::new(-0.9, 0.9, i));
        }
        let metrics = client.inferred_outcome_metrics();
        assert_eq!(metrics.negative_affect_burden, None);
        assert_eq!(metrics.anxious_activation_burden, None);
        assert_eq!(metrics.functional_wellbeing, None);
        assert_eq!(
            metrics.instrument_status,
            InstrumentAdministrationStatus::NotAdministered
        );
        assert_eq!(
            metrics.source,
            OutcomeMetricSource::ModelInferenceFromAffectAndRDoc
        );
    }

    #[test]
    fn inferred_metrics_are_normalized_and_explicitly_not_instruments() {
        let mut client = ClientModel::new();
        for i in 0..50 {
            client.update_affect(CoreAffectSnapshot::new(-0.4, 0.7, i));
        }
        let metrics = client.inferred_outcome_metrics();
        for value in [
            metrics.negative_affect_burden,
            metrics.anxious_activation_burden,
            metrics.functional_wellbeing,
        ] {
            let value = value.expect("sufficient observations should produce a value");
            assert!((0.0..=1.0).contains(&value));
        }
        assert_eq!(metrics.observations_used, 50);
        assert_eq!(
            metrics.instrument_status,
            InstrumentAdministrationStatus::NotAdministered
        );
    }

    // ── Legacy Clinical-Scale Compatibility Tests ───────────────────────

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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

    #[cfg(feature = "legacy-clinical-scale-analogues")]
    #[test]
    fn test_phq9_insufficient_data() {
        let mut client = ClientModel::new();
        for i in 0..5 {
            client.update_affect(CoreAffectSnapshot::new(-0.9, 0.9, i));
        }
        assert_eq!(client.phq9_analogue(), 0.0, "Insufficient data returns 0");
    }

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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

    #[cfg(feature = "legacy-clinical-scale-analogues")]
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
