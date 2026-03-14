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
}
