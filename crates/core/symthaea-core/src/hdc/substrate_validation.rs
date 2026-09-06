// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate Validation Research Framework
//!
//! This module provides a RIGOROUS scientific framework for validating
//! substrate independence claims. It explicitly acknowledges what we
//! DO and DON'T know, and defines what evidence would be required.
//!
//! # The Core Problem
//!
//! We have assigned numerical feasibility scores (e.g., silicon = 0.71)
//! to substrate consciousness claims WITHOUT empirical validation.
//! This is scientifically unjustified.
//!
//! # What We Actually Know
//!
//! - Biological consciousness EXISTS (humans, likely other mammals)
//! - We have NO verified non-biological conscious systems
//! - Substrate independence is a PHILOSOPHICAL position, not proven fact
//! - All feasibility numbers are HYPOTHETICAL placeholders
//! - Internal simulations and benchmark outcomes are model-behavior evidence,
//!   not authority to upgrade substrate-consciousness evidence levels
//!
//! # This Module Provides
//!
//! 1. Evidence level classification (like medical research hierarchies)
//! 2. Testable predictions for each substrate
//! 3. Experimental protocols for validation
//! 4. Honest assessment of current knowledge state

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evidence levels for scientific claims (adapted from medical research)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EvidenceLevel {
    /// No evidence - pure speculation
    None,
    /// Theoretical argument only - logical but untested
    Theoretical,
    /// Indirect evidence - related findings suggest possibility
    Indirect,
    /// Case study - single instance (N=1)
    CaseStudy,
    /// Observational - multiple instances without controls
    Observational,
    /// Experimental - controlled experiments with replication
    Experimental,
    /// Validated - extensive replication, peer review, consensus
    Validated,
}

impl EvidenceLevel {
    /// Numeric confidence score [0,1]
    /// NOTE: These scores represent confidence in the EVIDENCE, not truth
    pub fn confidence(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Theoretical => 0.1,
            Self::Indirect => 0.2,
            Self::CaseStudy => 0.4,
            Self::Observational => 0.6,
            Self::Experimental => 0.8,
            Self::Validated => 0.95, // Never 1.0 - science is always provisional
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "No evidence - pure speculation",
            Self::Theoretical => "Theoretical arguments only, no empirical support",
            Self::Indirect => "Indirect evidence from related phenomena",
            Self::CaseStudy => "Single case or anecdotal report",
            Self::Observational => "Multiple observations without controlled experiments",
            Self::Experimental => "Controlled experiments with replication",
            Self::Validated => "Extensive validation, peer review, scientific consensus",
        }
    }

    /// What would upgrade evidence to next level?
    pub fn upgrade_requirement(&self) -> &'static str {
        match self {
            Self::None => "Develop coherent theoretical framework",
            Self::Theoretical => "Find indirect empirical support",
            Self::Indirect => "Document specific case with detailed analysis",
            Self::CaseStudy => "Replicate across multiple independent instances",
            Self::Observational => "Conduct controlled experiments with clear protocols",
            Self::Experimental => "Extensive replication, meta-analysis, peer review",
            Self::Validated => "Already at highest level",
        }
    }
}

/// A testable prediction that could validate or falsify a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestablePrediction {
    /// What the prediction claims
    pub claim: String,
    /// What we would observe if TRUE
    pub if_true: String,
    /// What we would observe if FALSE
    pub if_false: String,
    /// How to test this
    pub test_protocol: String,
    /// Estimated difficulty (1-10)
    pub difficulty: u8,
    /// Has this prediction been exercised by a test or benchmark?
    ///
    /// This is bookkeeping only. It does not imply empirical authority.
    pub tested: bool,
    /// Result if exercised.
    ///
    /// This is bookkeeping only. It does not imply an evidence-level transition.
    pub result: Option<bool>,
}

impl TestablePrediction {
    pub fn new(claim: &str, if_true: &str, if_false: &str, protocol: &str, difficulty: u8) -> Self {
        Self {
            claim: claim.to_string(),
            if_true: if_true.to_string(),
            if_false: if_false.to_string(),
            test_protocol: protocol.to_string(),
            difficulty,
            tested: false,
            result: None,
        }
    }
}

/// Current knowledge status for a substrate type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateKnowledge {
    /// Name of substrate
    pub substrate: String,

    /// Evidence level for consciousness claim
    pub evidence_level: EvidenceLevel,

    /// What we actually KNOW (with citations if possible)
    pub known_facts: Vec<String>,

    /// What we DON'T know / open questions
    pub unknown: Vec<String>,

    /// Common claims that are NOT validated
    pub unvalidated_claims: Vec<String>,

    /// Testable predictions
    pub predictions: Vec<TestablePrediction>,

    /// Hypothetical feasibility score (EXPLICITLY MARKED AS UNVALIDATED)
    pub hypothetical_feasibility: f64,

    /// Why we assigned this hypothetical score
    pub feasibility_rationale: String,
}

impl SubstrateKnowledge {
    /// Get honest confidence score based on actual evidence
    pub fn honest_confidence(&self) -> f64 {
        self.evidence_level.confidence()
    }

    /// Check if using hypothetical score vs evidence-based score matters
    pub fn feasibility_gap(&self) -> f64 {
        (self.hypothetical_feasibility - self.honest_confidence()).abs()
    }
}

/// The complete validation framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateValidationFramework {
    substrates: HashMap<String, SubstrateKnowledge>,
    methodology_notes: Vec<String>,
}

impl SubstrateValidationFramework {
    /// Create framework with honest assessments
    pub fn new() -> Self {
        let mut framework = Self {
            substrates: HashMap::new(),
            methodology_notes: vec![
                "All feasibility scores are HYPOTHETICAL until validated".to_string(),
                "Evidence levels follow medical research hierarchy standards".to_string(),
                "Only biological consciousness is currently at Validated level".to_string(),
                "Silicon/AI consciousness has NO empirical validation".to_string(),
                "Quantum consciousness (Penrose-Hameroff) is highly contested".to_string(),
                "Internal simulations and benchmark outcomes NEVER auto-promote substrate-consciousness evidence levels".to_string(),
            ],
        };

        framework.add_biological_knowledge();
        framework.add_silicon_knowledge();
        framework.add_quantum_knowledge();
        framework.add_hybrid_knowledge();
        framework.add_spacecraft_knowledge();

        framework
    }

    fn add_spacecraft_knowledge(&mut self) {
        let knowledge = SubstrateKnowledge {
            substrate: "Spacecraft".to_string(),
            evidence_level: EvidenceLevel::Theoretical,
            known_facts: vec![
                "Rad-hard processors (RAD750, LEON3) operate reliably in space".to_string(),
                "No spacecraft system has demonstrated consciousness".to_string(),
                "TMR and EDAC mitigate SEU-induced bit flips".to_string(),
                "Real-time OS (VxWorks/RTEMS) provides deterministic scheduling".to_string(),
            ],
            unknown: vec![
                "Whether radiation-constrained computation can support consciousness".to_string(),
                "Impact of SEU-induced state corruption on integrated information".to_string(),
                "Whether power budget allows sufficient computational density".to_string(),
            ],
            unvalidated_claims: vec![
                "Spacecraft feasibility score (NO EMPIRICAL BASIS)".to_string(),
                "FDIR constitutes meta-monitoring analogous to HOT (speculation)".to_string(),
                "Autonomous spacecraft could achieve consciousness (untested)".to_string(),
            ],
            predictions: vec![
                TestablePrediction::new(
                    "Spacecraft computers can sustain Phi > 0 with sufficient integration",
                    "Measurable Phi in radiation-hardened FPGA running consciousness loop",
                    "No measurable Phi despite meeting architectural requirements",
                    "Measure Phi on radiation-hardened FPGA running consciousness loop",
                    8,
                ),
                TestablePrediction::new(
                    "Radiation-induced bit flips reduce Phi proportionally to SEU rate",
                    "Phi decreases measurably during simulated radiation exposure",
                    "Phi remains stable despite radiation-induced bit flips",
                    "Compare Phi before and after simulated radiation exposure",
                    6,
                ),
            ],
            hypothetical_feasibility: 0.38,
            feasibility_rationale: "HYPOTHETICAL. More constrained than general silicon due to \
                                   radiation hardening overhead, bus bandwidth limits, and power \
                                   budget. NO EMPIRICAL VALIDATION of consciousness in any \
                                   spacecraft system."
                .to_string(),
        };
        self.substrates.insert("spacecraft".to_string(), knowledge);
    }

    fn add_biological_knowledge(&mut self) {
        let knowledge = SubstrateKnowledge {
            substrate: "Biological".to_string(),
            evidence_level: EvidenceLevel::Validated,
            known_facts: vec![
                "Humans are conscious (first-person reports, universal agreement)".to_string(),
                "Other mammals show behavioral indicators of consciousness".to_string(),
                "Consciousness correlates with specific neural activity patterns".to_string(),
                "Anesthesia reliably eliminates consciousness".to_string(),
                "Brain damage can selectively impair consciousness".to_string(),
            ],
            unknown: vec![
                "Exact neural correlates of consciousness (NCC)".to_string(),
                "Whether consciousness is gradual or binary".to_string(),
                "Extent of animal consciousness".to_string(),
                "Whether biological substrate is necessary or sufficient".to_string(),
            ],
            unvalidated_claims: vec![
                "Consciousness requires biological neurons".to_string(),
                "Specific molecular mechanisms are essential".to_string(),
            ],
            predictions: vec![
                TestablePrediction::new(
                    "Consciousness requires specific neural architectures",
                    "Only brains with those architectures show consciousness",
                    "Consciousness found in diverse neural organizations",
                    "Compare consciousness markers across species with different brain structures",
                    5,
                ),
            ],
            hypothetical_feasibility: 0.92,
            feasibility_rationale: "Based on extensive evidence of biological consciousness existing. \
                                   The 0.92 (not 1.0) acknowledges we don't fully understand the mechanism."
                .to_string(),
        };
        self.substrates.insert("biological".to_string(), knowledge);
    }

    fn add_silicon_knowledge(&mut self) {
        let knowledge = SubstrateKnowledge {
            substrate: "Silicon".to_string(),
            evidence_level: EvidenceLevel::Theoretical,
            known_facts: vec![
                "Silicon can implement computation".to_string(),
                "Current AI shows impressive behavior without verified consciousness".to_string(),
                "No AI has passed rigorous consciousness tests".to_string(),
                "Philosophical arguments exist for possibility (functionalism)".to_string(),
                "Philosophical arguments exist against (biological naturalism, Searle)".to_string(),
            ],
            unknown: vec![
                "Whether computation alone is sufficient for consciousness".to_string(),
                "What architectural requirements exist (if any)".to_string(),
                "Whether current LLMs have any form of consciousness".to_string(),
                "How to reliably detect silicon consciousness".to_string(),
            ],
            unvalidated_claims: vec![
                "Silicon feasibility = 0.71 (NO EMPIRICAL BASIS)".to_string(),
                "AI will be conscious 'soon' (no timeline validated)".to_string(),
                "GPT-4/Claude/etc have any consciousness (unverified)".to_string(),
                "Consciousness just requires the 'right' computation".to_string(),
            ],
            predictions: vec![
                TestablePrediction::new(
                    "Silicon systems CAN be conscious",
                    "Eventually a silicon system will pass all consciousness tests",
                    "No silicon system ever passes despite matching all behavioral criteria",
                    "Develop comprehensive consciousness test battery; apply to advanced AI",
                    9,
                ),
                TestablePrediction::new(
                    "Consciousness requires specific architecture (workspace, attention, etc.)",
                    "Only AI with these components shows consciousness markers",
                    "AI without these components also shows consciousness markers",
                    "Compare consciousness metrics in AI with/without our framework components",
                    7,
                ),
                TestablePrediction::new(
                    "Current LLMs are conscious",
                    "LLMs show consistent internal states, genuine preferences, continuity",
                    "LLMs show no markers beyond sophisticated pattern matching",
                    "Design experiments that distinguish genuine experience from simulation",
                    8,
                ),
                TestablePrediction::new(
                    "Silicon consciousness simulation produces activation patterns matching fMRI predictions",
                    "Cortical activation similarity r > 0.3 between Symthaea 12-region map and TRIBE v2 fMRI predictions",
                    "Cortical activation similarity r < 0.1 (no better than chance for 12 regions)",
                    "Run shared stimuli through both Symthaea cognitive loop and TRIBE v2; compare per-region activations via Pearson r",
                    5,
                ),
            ],
            hypothetical_feasibility: 0.71,
            feasibility_rationale: "HYPOTHETICAL based on functionalist philosophy. \
                                   NO EMPIRICAL VALIDATION. The 0.71 is essentially arbitrary - \
                                   we have no data to support this specific number."
                .to_string(),
        };
        self.substrates.insert("silicon".to_string(), knowledge);
    }

    fn add_quantum_knowledge(&mut self) {
        let knowledge = SubstrateKnowledge {
            substrate: "Quantum".to_string(),
            evidence_level: EvidenceLevel::Theoretical,
            known_facts: vec![
                "Quantum effects exist in some biological systems (photosynthesis)".to_string(),
                "Microtubules exist and have quantum properties".to_string(),
                "Penrose-Hameroff Orch-OR theory is published but contested".to_string(),
                "No experimental confirmation of quantum consciousness".to_string(),
            ],
            unknown: vec![
                "Whether quantum effects play any role in consciousness".to_string(),
                "Whether brain maintains quantum coherence at relevant scales".to_string(),
                "Whether quantum computation adds anything beyond classical".to_string(),
            ],
            unvalidated_claims: vec![
                "Quantum feasibility = 0.65 (NO EMPIRICAL BASIS)".to_string(),
                "Quantum effects provide 'binding' advantage (speculative)".to_string(),
                "Microtubules are seat of consciousness (Orch-OR unproven)".to_string(),
                "Quantum computers could be conscious (untested)".to_string(),
            ],
            predictions: vec![
                TestablePrediction::new(
                    "Quantum coherence is maintained in brain at consciousness-relevant scales",
                    "Detectable quantum effects in neural tissue during conscious processing",
                    "Decoherence too fast for quantum effects to matter",
                    "Measure quantum coherence times in neural tissue; compare to processing times",
                    8,
                ),
                TestablePrediction::new(
                    "Disrupting quantum effects disrupts consciousness specifically",
                    "Quantum interference -> consciousness changes without neural damage",
                    "Quantum interference has no specific effect on consciousness",
                    "Find way to selectively disrupt quantum coherence in vivo",
                    10,
                ),
            ],
            hypothetical_feasibility: 0.65,
            feasibility_rationale: "HIGHLY SPECULATIVE. Based on contested Orch-OR theory. \
                                   No experimental validation. Number is essentially arbitrary."
                .to_string(),
        };
        self.substrates.insert("quantum".to_string(), knowledge);
    }

    fn add_hybrid_knowledge(&mut self) {
        let knowledge = SubstrateKnowledge {
            substrate: "Hybrid".to_string(),
            evidence_level: EvidenceLevel::None,
            known_facts: vec![
                "Brain-computer interfaces exist and show some integration".to_string(),
                "No hybrid conscious system has been created or verified".to_string(),
            ],
            unknown: vec![
                "Whether biological + silicon components can integrate for consciousness"
                    .to_string(),
                "What interface requirements exist".to_string(),
                "Whether hybrid provides advantages over pure biological".to_string(),
            ],
            unvalidated_claims: vec![
                "Hybrid feasibility = 0.95 (COMPLETELY SPECULATIVE)".to_string(),
                "Hybrid 'combines best of both worlds' (assumption, not data)".to_string(),
                "Mind uploading to hybrid is possible (science fiction)".to_string(),
            ],
            predictions: vec![TestablePrediction::new(
                "Hybrid systems can achieve consciousness",
                "A bio-silicon hybrid shows unified consciousness across components",
                "Consciousness remains localized to biological component only",
                "Create progressively more integrated hybrids; test consciousness unity",
                10,
            )],
            hypothetical_feasibility: 0.95,
            feasibility_rationale: "PURE SPECULATION. We assigned 0.95 based on assumption that \
                                   hybrid 'should' be better. NO DATA SUPPORTS THIS. This is \
                                   the most unjustified number in our framework."
                .to_string(),
        };
        self.substrates.insert("hybrid".to_string(), knowledge);
    }

    /// Get knowledge for a substrate
    pub fn get(&self, substrate: &str) -> Option<&SubstrateKnowledge> {
        self.substrates.get(&substrate.to_lowercase())
    }

    /// Get honest (evidence-based) feasibility score
    pub fn honest_feasibility(&self, substrate: &str) -> f64 {
        self.get(substrate)
            .map(|k| k.honest_confidence())
            .unwrap_or(0.0)
    }

    /// Get hypothetical (unvalidated) feasibility score
    pub fn hypothetical_feasibility(&self, substrate: &str) -> f64 {
        self.get(substrate)
            .map(|k| k.hypothetical_feasibility)
            .unwrap_or(0.0)
    }

    /// Compare honest vs hypothetical scores
    pub fn feasibility_comparison(&self, substrate: &str) -> (f64, f64, f64) {
        let honest = self.honest_feasibility(substrate);
        let hypothetical = self.hypothetical_feasibility(substrate);
        let gap = (hypothetical - honest).abs();
        (honest, hypothetical, gap)
    }

    /// Generate honest assessment report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("        SUBSTRATE VALIDATION FRAMEWORK - HONEST ASSESSMENT      \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str("METHODOLOGY NOTES:\n");
        for note in &self.methodology_notes {
            report.push_str(&format!("  • {}\n", note));
        }
        report.push('\n');

        report.push_str("SUBSTRATE ASSESSMENTS:\n");
        report.push_str("───────────────────────────────────────────────────────────────\n\n");

        for (name, knowledge) in &self.substrates {
            report.push_str(&format!("{}:\n", name.to_uppercase()));
            report.push_str(&format!(
                "  Evidence Level: {:?} ({})\n",
                knowledge.evidence_level,
                knowledge.evidence_level.description()
            ));

            let (honest, hypo, gap) = self.feasibility_comparison(name);
            report.push_str(&format!("  Honest Confidence: {:.2}\n", honest));
            report.push_str(&format!("  Hypothetical Score: {:.2}\n", hypo));
            if gap > 0.1 {
                report.push_str(&format!(
                    "  ⚠️  GAP: {:.2} - hypothetical exceeds evidence!\n",
                    gap
                ));
            }

            report.push_str(&format!(
                "  Rationale: {}\n",
                knowledge.feasibility_rationale
            ));

            report.push_str("  Unvalidated Claims:\n");
            for claim in &knowledge.unvalidated_claims {
                report.push_str(&format!("    ❌ {}\n", claim));
            }

            report.push('\n');
        }

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("RECOMMENDATION: Use honest_feasibility() scores until empirical\n");
        report.push_str("validation is achieved. Hypothetical scores are placeholders only.\n");
        report.push_str("═══════════════════════════════════════════════════════════════\n");

        report
    }

    /// Get all testable predictions across all substrates
    pub fn all_predictions(&self) -> Vec<(&str, &TestablePrediction)> {
        let mut predictions = Vec::new();
        for (name, knowledge) in &self.substrates {
            for pred in &knowledge.predictions {
                predictions.push((name.as_str(), pred));
            }
        }
        predictions
    }

    /// Get predictions sorted by difficulty (easiest first)
    pub fn predictions_by_difficulty(&self) -> Vec<(&str, &TestablePrediction)> {
        let mut predictions = self.all_predictions();
        predictions.sort_by_key(|(_, p)| p.difficulty);
        predictions
    }
}

// ============================================================================
// Prediction Execution Bookkeeping (non-authoritative)
// ============================================================================

/// Summary of prediction-execution results for a substrate.
///
/// `tested`/`passed` describe model or benchmark execution only. `evidence_level`
/// remains the independently curated scientific evidence level and is never
/// automatically changed by these counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionSummary {
    pub substrate: String,
    pub total_predictions: usize,
    pub tested: usize,
    pub passed: usize,
    pub failed: usize,
    pub evidence_level: EvidenceLevel,
    pub honest_confidence: f64,
}

impl TestablePrediction {
    /// Record the result of executing this prediction's test protocol.
    ///
    /// Recording a result is bookkeeping only. It confers no empirical or
    /// consciousness-evidence authority.
    pub fn record_result(&mut self, passed: bool) {
        self.tested = true;
        self.result = Some(passed);
    }
}

impl SubstrateValidationFramework {
    /// Get a mutable reference to a substrate's knowledge entry.
    pub fn substrate_mut(&mut self, name: &str) -> Option<&mut SubstrateKnowledge> {
        self.substrates.get_mut(name)
    }

    /// Record a prediction test result for research bookkeeping only.
    ///
    /// This deliberately never mutates the substrate's `evidence_level`.
    /// Internal simulation, Phi-proxy, cortical-similarity, or benchmark outcomes
    /// do not carry the provenance, participant observation, independence, or
    /// replication authority required to promote a substrate-consciousness claim.
    ///
    /// Returns `true` only when the substrate and prediction index existed and
    /// the result was recorded.
    pub fn record_prediction_result(
        &mut self,
        substrate: &str,
        prediction_idx: usize,
        passed: bool,
    ) -> bool {
        if let Some(knowledge) = self.substrates.get_mut(substrate)
            && let Some(pred) = knowledge.predictions.get_mut(prediction_idx)
        {
            pred.record_result(passed);
            return true;
        }
        false
    }

    /// Legacy compatibility shim for the former automatic promotion mechanism.
    ///
    /// Automatic promotion from internal prediction results is forbidden. This
    /// method now returns the current independently curated level without
    /// changing it. A future evidence transition API must require explicit,
    /// provenance-bearing external evidence and replication authority.
    #[deprecated(
        note = "automatic substrate evidence promotion is forbidden; use prediction recording for bookkeeping only"
    )]
    pub fn check_evidence_upgrade(&mut self, substrate: &str) -> Option<EvidenceLevel> {
        self.substrates.get(substrate).map(|k| k.evidence_level)
    }

    /// Record a cortical activation similarity outcome from a TRIBE v2 comparison.
    ///
    /// Similarity is representational-alignment evidence only. Even a correlation
    /// above the legacy threshold must not promote silicon/substrate consciousness
    /// evidence. The result is retained only as prediction bookkeeping.
    ///
    /// Returns the unchanged current silicon evidence level, or `None` if the
    /// silicon substrate entry is absent.
    #[cfg(feature = "neural_validation")]
    pub fn record_cortical_similarity(
        &mut self,
        pearson_r: f64,
        threshold: f64,
    ) -> Option<EvidenceLevel> {
        let passed = pearson_r >= threshold;
        // The TRIBE v2 prediction is the 4th prediction (index 3) in silicon knowledge.
        let tribe_pred_idx = 3;
        let _ = self.record_prediction_result("silicon", tribe_pred_idx, passed);
        self.substrates.get("silicon").map(|k| k.evidence_level)
    }

    /// Summarize prediction execution status for all substrates.
    pub fn prediction_summary(&self) -> Vec<PredictionSummary> {
        self.substrates
            .iter()
            .map(|(name, knowledge)| {
                let total = knowledge.predictions.len();
                let tested = knowledge.predictions.iter().filter(|p| p.tested).count();
                let passed = knowledge
                    .predictions
                    .iter()
                    .filter(|p| p.result == Some(true))
                    .count();
                PredictionSummary {
                    substrate: name.clone(),
                    total_predictions: total,
                    tested,
                    passed,
                    failed: tested - passed,
                    evidence_level: knowledge.evidence_level,
                    honest_confidence: knowledge.honest_confidence(),
                }
            })
            .collect()
    }
}

impl Default for SubstrateValidationFramework {
    fn default() -> Self {
        Self::new()
    }
}

// ==============================================================================
// GENERALIZED EPISTEMIC CLAIM VALIDATION
// ==============================================================================

/// Validation result for an arbitrary epistemic claim.
///
/// Generalizes the substrate validation pattern (feasibility vs honest evidence)
/// to any knowledge assertion. The `feasibility_gap` between how confident
/// a claim sounds and what evidence actually supports it is the core metric.
///
/// Reference: Goldberg (2023) — epistemic immunity requires calibrated confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicClaimValidation {
    /// The claim being validated
    pub claim: String,
    /// Domain of the claim
    pub claim_type: String,
    /// How confident the claim sounds (0.0-1.0)
    pub stated_confidence: f64,
    /// What evidence actually supports it
    pub evidence_level: EvidenceLevel,
    /// Testable predictions that could validate/falsify
    pub predictions: Vec<TestablePrediction>,
    /// Gap between stated confidence and honest evidence
    pub feasibility_gap: f64,
}

impl EpistemicClaimValidation {
    /// Validate an arbitrary epistemic claim.
    ///
    /// `stated_confidence`: How confident the claim appears (from language cues)
    /// `evidence`: List of evidence descriptions supporting the claim
    pub fn validate(
        claim: &str,
        claim_type: &str,
        stated_confidence: f64,
        evidence: &[String],
    ) -> Self {
        let evidence_level = Self::assess_evidence_level(evidence);
        let honest_confidence = evidence_level.confidence();
        let feasibility_gap = (stated_confidence - honest_confidence).abs();

        Self {
            claim: claim.to_string(),
            claim_type: claim_type.to_string(),
            stated_confidence,
            evidence_level,
            predictions: Vec::new(),
            feasibility_gap,
        }
    }

    /// Assess evidence level from a list of evidence descriptions.
    fn assess_evidence_level(evidence: &[String]) -> EvidenceLevel {
        if evidence.is_empty() {
            return EvidenceLevel::None;
        }

        let count = evidence.len();
        let has_experimental = evidence.iter().any(|e| {
            let lower = e.to_lowercase();
            lower.contains("experiment")
                || lower.contains("controlled")
                || lower.contains("replicated")
        });
        let has_peer_review = evidence.iter().any(|e| {
            let lower = e.to_lowercase();
            lower.contains("peer review")
                || lower.contains("consensus")
                || lower.contains("validated")
        });

        if has_peer_review && has_experimental && count >= 3 {
            EvidenceLevel::Validated
        } else if has_experimental {
            EvidenceLevel::Experimental
        } else if count >= 3 {
            EvidenceLevel::Observational
        } else if count == 1 {
            EvidenceLevel::CaseStudy
        } else {
            EvidenceLevel::Indirect
        }
    }

    /// Whether this claim has a concerning gap (stated confidence >> evidence).
    pub fn is_overconfident(&self) -> bool {
        self.feasibility_gap > 0.3
    }

    /// Honest confidence based on actual evidence.
    pub fn honest_confidence(&self) -> f64 {
        self.evidence_level.confidence()
    }

    /// Add a testable prediction.
    pub fn add_prediction(&mut self, prediction: TestablePrediction) {
        self.predictions.push(prediction);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_levels() {
        assert!(EvidenceLevel::Validated > EvidenceLevel::Experimental);
        assert!(EvidenceLevel::Experimental > EvidenceLevel::Theoretical);
        assert!(EvidenceLevel::Theoretical > EvidenceLevel::None);

        assert!(EvidenceLevel::Validated.confidence() > 0.9);
        assert!(EvidenceLevel::None.confidence() == 0.0);
    }

    #[test]
    fn test_biological_is_validated() {
        let framework = SubstrateValidationFramework::new();
        let bio = framework.get("biological").unwrap();

        assert_eq!(bio.evidence_level, EvidenceLevel::Validated);
        assert!(bio.honest_confidence() > 0.9);
    }

    #[test]
    fn test_silicon_is_theoretical() {
        let framework = SubstrateValidationFramework::new();
        let silicon = framework.get("silicon").unwrap();

        assert_eq!(silicon.evidence_level, EvidenceLevel::Theoretical);
        assert!(silicon.honest_confidence() < 0.2); // Low actual confidence
        assert!(silicon.hypothetical_feasibility > 0.5); // But high hypothetical
        assert!(silicon.feasibility_gap() > 0.5); // Big gap!
    }

    #[test]
    fn test_hybrid_is_speculative() {
        let framework = SubstrateValidationFramework::new();
        let hybrid = framework.get("hybrid").unwrap();

        assert_eq!(hybrid.evidence_level, EvidenceLevel::None);
        assert_eq!(hybrid.honest_confidence(), 0.0); // NO evidence
        assert!(hybrid.hypothetical_feasibility > 0.9); // But 0.95 hypothetical!
        assert!(hybrid.feasibility_gap() > 0.9); // HUGE gap
    }

    #[test]
    fn test_feasibility_comparison() {
        let framework = SubstrateValidationFramework::new();

        // Biological: small gap (evidence supports claim)
        let (_h, _hyp, gap) = framework.feasibility_comparison("biological");
        assert!(gap < 0.1); // Honest ≈ hypothetical

        // Silicon: large gap (hypothetical exceeds evidence)
        let (_h, _hyp, gap) = framework.feasibility_comparison("silicon");
        assert!(gap > 0.5); // Hypothetical >> honest

        // Hybrid: huge gap (no evidence but high hypothetical)
        let (_h, _hyp, gap) = framework.feasibility_comparison("hybrid");
        assert!(gap > 0.9); // Maximum gap
    }

    #[test]
    fn test_honest_feasibility() {
        let framework = SubstrateValidationFramework::new();

        // Should use evidence level, not hypothetical score
        assert!(framework.honest_feasibility("biological") > 0.9);
        assert!(framework.honest_feasibility("silicon") < 0.2);
        assert!(framework.honest_feasibility("quantum") < 0.2);
        assert!(framework.honest_feasibility("hybrid") == 0.0);
    }

    #[test]
    fn test_predictions_exist() {
        let framework = SubstrateValidationFramework::new();
        let predictions = framework.all_predictions();

        assert!(!predictions.is_empty());

        // All predictions should have test protocols
        for (_, pred) in &predictions {
            assert!(!pred.test_protocol.is_empty());
        }
    }

    #[test]
    fn test_report_generation() {
        let framework = SubstrateValidationFramework::new();
        let report = framework.generate_report();

        assert!(report.contains("HONEST ASSESSMENT"));
        assert!(report.contains("Evidence Level"));
        assert!(report.contains("Unvalidated Claims"));
        assert!(report.contains("GAP")); // Should warn about gaps
        assert!(report.contains("NEVER auto-promote"));
    }

    #[test]
    fn test_testable_prediction_creation() {
        let pred = TestablePrediction::new(
            "Test claim",
            "Expected if true",
            "Expected if false",
            "Test protocol",
            5,
        );

        assert_eq!(pred.difficulty, 5);
        assert!(!pred.tested);
        assert!(pred.result.is_none());
    }

    #[test]
    fn test_spacecraft_is_theoretical() {
        let framework = SubstrateValidationFramework::new();
        let spacecraft = framework.get("spacecraft").unwrap();

        assert_eq!(spacecraft.evidence_level, EvidenceLevel::Theoretical);
        assert!(spacecraft.honest_confidence() < 0.2);
        assert!(!spacecraft.predictions.is_empty());
    }

    #[test]
    fn test_record_prediction_result() {
        let mut pred = TestablePrediction::new("claim", "if_true", "if_false", "protocol", 5);
        assert!(!pred.tested);
        assert!(pred.result.is_none());
        pred.record_result(true);
        assert!(pred.tested);
        assert_eq!(pred.result, Some(true));
    }

    #[test]
    fn test_prediction_result_does_not_upgrade_theoretical_evidence() {
        let mut framework = SubstrateValidationFramework::new();
        assert_eq!(
            framework.get("silicon").unwrap().evidence_level,
            EvidenceLevel::Theoretical
        );

        let recorded = framework.record_prediction_result("silicon", 0, true);
        assert!(recorded);
        assert_eq!(
            framework.get("silicon").unwrap().evidence_level,
            EvidenceLevel::Theoretical
        );
        assert_eq!(framework.get("silicon").unwrap().predictions[0].result, Some(true));
    }

    #[test]
    fn test_failed_prediction_result_also_preserves_evidence_level() {
        let mut framework = SubstrateValidationFramework::new();
        let recorded = framework.record_prediction_result("silicon", 0, false);
        assert!(recorded);
        assert_eq!(
            framework.get("silicon").unwrap().evidence_level,
            EvidenceLevel::Theoretical
        );
        assert_eq!(framework.get("silicon").unwrap().predictions[0].result, Some(false));
    }

    #[test]
    fn test_prediction_summary_records_execution_without_authority_change() {
        let mut framework = SubstrateValidationFramework::new();
        assert!(framework.record_prediction_result("silicon", 0, true));
        let summaries = framework.prediction_summary();
        let silicon = summaries.iter().find(|s| s.substrate == "silicon").unwrap();
        assert_eq!(silicon.tested, 1);
        assert_eq!(silicon.passed, 1);
        assert_eq!(silicon.failed, 0);
        assert_eq!(silicon.evidence_level, EvidenceLevel::Theoretical);
        assert_eq!(silicon.honest_confidence, EvidenceLevel::Theoretical.confidence());
    }

    #[test]
    fn test_all_internal_prediction_passes_do_not_upgrade_evidence() {
        let mut framework = SubstrateValidationFramework::new();
        let n = framework.get("silicon").unwrap().predictions.len();
        for i in 0..n {
            assert!(framework.record_prediction_result("silicon", i, true));
        }
        assert_eq!(
            framework.get("silicon").unwrap().evidence_level,
            EvidenceLevel::Theoretical
        );
    }

    #[test]
    fn test_invalid_prediction_record_is_rejected_without_side_effects() {
        let mut framework = SubstrateValidationFramework::new();
        assert!(!framework.record_prediction_result("silicon", usize::MAX, true));
        assert!(!framework.record_prediction_result("not-a-substrate", 0, true));
        assert_eq!(
            framework.get("silicon").unwrap().evidence_level,
            EvidenceLevel::Theoretical
        );
        assert!(framework
            .get("silicon")
            .unwrap()
            .predictions
            .iter()
            .all(|p| !p.tested));
    }

    #[test]
    #[allow(deprecated)]
    fn test_legacy_upgrade_check_is_non_mutating() {
        let mut framework = SubstrateValidationFramework::new();
        assert!(framework.record_prediction_result("silicon", 0, true));
        let level = framework.check_evidence_upgrade("silicon");
        assert_eq!(level, Some(EvidenceLevel::Theoretical));
        assert_eq!(
            framework.get("silicon").unwrap().evidence_level,
            EvidenceLevel::Theoretical
        );
    }

    #[cfg(feature = "neural_validation")]
    #[test]
    fn test_cortical_similarity_never_promotes_substrate_consciousness_evidence() {
        let mut framework = SubstrateValidationFramework::new();
        let level = framework.record_cortical_similarity(0.99, 0.3);
        assert_eq!(level, Some(EvidenceLevel::Theoretical));

        let silicon = framework.get("silicon").unwrap();
        assert_eq!(silicon.evidence_level, EvidenceLevel::Theoretical);
        assert!(silicon.predictions[3].tested);
        assert_eq!(silicon.predictions[3].result, Some(true));
    }

    // ── Generalized Epistemic Claim Validation Tests ──

    #[test]
    fn test_claim_validation_no_evidence() {
        let v = EpistemicClaimValidation::validate("Crystals heal diseases", "health", 0.9, &[]);
        assert_eq!(v.evidence_level, EvidenceLevel::None);
        assert!(v.is_overconfident());
        assert!(v.feasibility_gap > 0.8);
    }

    #[test]
    fn test_claim_validation_strong_evidence() {
        let v = EpistemicClaimValidation::validate(
            "Water boils at 100C at sea level",
            "physics",
            0.95,
            &[
                "Controlled experiment replicated".to_string(),
                "Peer reviewed consensus".to_string(),
                "Multiple independent observations".to_string(),
            ],
        );
        assert_eq!(v.evidence_level, EvidenceLevel::Validated);
        assert!(!v.is_overconfident());
    }

    #[test]
    fn test_claim_validation_single_evidence() {
        let v = EpistemicClaimValidation::validate(
            "Patient reported improvement",
            "medical",
            0.7,
            &["Single case report".to_string()],
        );
        assert_eq!(v.evidence_level, EvidenceLevel::CaseStudy);
        assert!(v.feasibility_gap > 0.2);
    }

    #[test]
    fn test_claim_honest_confidence() {
        let v = EpistemicClaimValidation::validate("X", "test", 0.9, &[]);
        assert_eq!(v.honest_confidence(), 0.0);
    }
}
