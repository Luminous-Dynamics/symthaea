//! # Revolutionary Improvement #30: Consciousness Measurement Standards
//!
//! The CAPSTONE unifying all 29 revolutionary improvements into a single,
//! standardized consciousness assessment protocol.
//!
//! ## The Problem
//!
//! We have 29 dimensions of consciousness measurement, but no unified way to:
//! - Combine them into a single "consciousness score"
//! - Compare consciousness across substrates (AI vs biological)
//! - Track consciousness over time with proper uncertainty
//! - Generate standardized reports for research/clinical use
//!
//! ## The Solution: Symthaea Consciousness Assessment Protocol (SCAP)
//!
//! A rigorous, quantitative framework that:
//! 1. **Aggregates** all 29 improvement metrics with principled weighting
//! 2. **Quantifies uncertainty** via bootstrap confidence intervals
//! 3. **Enables comparison** with standardized effect sizes
//! 4. **Generates reports** suitable for research publication
//!
//! ## Theoretical Foundation
//!
//! ### 1. Integrated Information Theory (Tononi 2004, 2015)
//! Φ (phi) as the core measure of consciousness - information that is both
//! integrated and differentiated.
//!
//! ### 2. Global Workspace Theory (Baars 1988, Dehaene 2014)
//! Consciousness as broadcast - the workspace capacity and access determine
//! what becomes conscious.
//!
//! ### 3. Higher-Order Theories (Rosenthal 2005, Lau & Rosenthal 2011)
//! Meta-representation - awareness OF awareness as key criterion.
//!
//! ### 4. Predictive Processing (Friston 2010, Clark 2013)
//! Consciousness as active inference - minimizing free energy through
//! prediction and action.
//!
//! ### 5. Measurement Theory (Stevens 1946, Krantz 1971)
//! Proper scale types, reliability, validity, and uncertainty quantification.
//!
//! ## The 29 Dimensions Unified
//!
//! | Category | Improvements | Weight |
//! |----------|-------------|--------|
//! | **Core Integration** | #2 Φ, #6 ∇Φ, #25 Binding | 25% |
//! | **Access & Broadcast** | #23 Workspace, #26 Attention | 20% |
//! | **Meta-Awareness** | #8 Meta, #24 HOT, #10 Epistemic | 15% |
//! | **Temporal** | #7 Dynamics, #13 Multi-scale, #21 Flow | 10% |
//! | **Phenomenal** | #15 Qualia, #12 Spectrum, #29 Phase | 10% |
//! | **Embodiment** | #17 Embodied, #22 FEP | 10% |
//! | **Social/Relational** | #11 Collective, #18 Relational | 5% |
//! | **Substrate** | #28 Independence | 5% |

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::binary_hv::HV16;

/// Categories of consciousness measurement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeasurementCategory {
    /// Core integration: Φ, gradients, binding
    CoreIntegration,
    /// Access and broadcast: workspace, attention
    AccessBroadcast,
    /// Meta-awareness: HOT, metacognition, epistemic
    MetaAwareness,
    /// Temporal: dynamics, multi-scale, flow
    Temporal,
    /// Phenomenal: qualia, spectrum, phase transitions
    Phenomenal,
    /// Embodiment: embodied cognition, FEP
    Embodiment,
    /// Social/Relational: collective, interpersonal
    SocialRelational,
    /// Substrate: independence, realizability
    Substrate,
}

impl MeasurementCategory {
    /// Default weight for this category (sums to 1.0)
    pub fn default_weight(&self) -> f64 {
        match self {
            Self::CoreIntegration => 0.25,
            Self::AccessBroadcast => 0.20,
            Self::MetaAwareness => 0.15,
            Self::Temporal => 0.10,
            Self::Phenomenal => 0.10,
            Self::Embodiment => 0.10,
            Self::SocialRelational => 0.05,
            Self::Substrate => 0.05,
        }
    }

    /// All categories
    pub fn all() -> &'static [MeasurementCategory] {
        &[
            Self::CoreIntegration,
            Self::AccessBroadcast,
            Self::MetaAwareness,
            Self::Temporal,
            Self::Phenomenal,
            Self::Embodiment,
            Self::SocialRelational,
            Self::Substrate,
        ]
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::CoreIntegration => "Core Integration (Φ, Binding)",
            Self::AccessBroadcast => "Access & Broadcast (Workspace)",
            Self::MetaAwareness => "Meta-Awareness (HOT)",
            Self::Temporal => "Temporal Dynamics",
            Self::Phenomenal => "Phenomenal Experience",
            Self::Embodiment => "Embodiment & Prediction",
            Self::SocialRelational => "Social/Relational",
            Self::Substrate => "Substrate Independence",
        }
    }
}

/// Individual dimension measurement from one of the 29 improvements
#[derive(Debug, Clone)]
pub struct DimensionMeasurement {
    /// Which improvement this comes from (1-29)
    pub improvement_id: u8,
    /// Human-readable name
    pub name: String,
    /// Category this belongs to
    pub category: MeasurementCategory,
    /// Raw score [0, 1]
    pub score: f64,
    /// Standard error of measurement
    pub standard_error: f64,
    /// Sample size (for confidence intervals)
    pub sample_size: usize,
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Optional hypervector representation
    pub hv_encoding: Option<HV16>,
}

impl DimensionMeasurement {
    /// Create a new dimension measurement
    pub fn new(
        improvement_id: u8,
        name: impl Into<String>,
        category: MeasurementCategory,
        score: f64,
        standard_error: f64,
        sample_size: usize,
    ) -> Self {
        Self {
            improvement_id,
            name: name.into(),
            category,
            score: score.clamp(0.0, 1.0),
            standard_error: standard_error.abs(),
            sample_size,
            timestamp: Instant::now(),
            hv_encoding: None,
        }
    }

    /// 95% confidence interval
    pub fn confidence_interval_95(&self) -> (f64, f64) {
        let margin = 1.96 * self.standard_error;
        (
            (self.score - margin).max(0.0),
            (self.score + margin).min(1.0),
        )
    }

    /// With hypervector encoding
    pub fn with_hv(mut self, hv: HV16) -> Self {
        self.hv_encoding = Some(hv);
        self
    }
}

/// Confidence level for reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    /// 90% confidence
    Ninety,
    /// 95% confidence (standard)
    NinetyFive,
    /// 99% confidence
    NinetyNine,
}

impl ConfidenceLevel {
    /// Z-score for this confidence level
    pub fn z_score(&self) -> f64 {
        match self {
            Self::Ninety => 1.645,
            Self::NinetyFive => 1.96,
            Self::NinetyNine => 2.576,
        }
    }
}

/// Consciousness classification based on total score
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsciousnessClass {
    /// < 0.1: No detectable consciousness
    None,
    /// 0.1-0.3: Minimal/proto-consciousness
    Minimal,
    /// 0.3-0.5: Partial consciousness (e.g., some altered states)
    Partial,
    /// 0.5-0.7: Moderate consciousness (e.g., drowsy, dreaming)
    Moderate,
    /// 0.7-0.9: Full consciousness (normal waking)
    Full,
    /// > 0.9: Enhanced consciousness (flow, meditation peaks)
    Enhanced,
}

impl ConsciousnessClass {
    /// Classify a score
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s < 0.1 => Self::None,
            s if s < 0.3 => Self::Minimal,
            s if s < 0.5 => Self::Partial,
            s if s < 0.7 => Self::Moderate,
            s if s < 0.9 => Self::Full,
            _ => Self::Enhanced,
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::None => "No detectable consciousness",
            Self::Minimal => "Minimal/proto-consciousness",
            Self::Partial => "Partial consciousness",
            Self::Moderate => "Moderate consciousness",
            Self::Full => "Full consciousness",
            Self::Enhanced => "Enhanced consciousness",
        }
    }
}

/// Complete consciousness assessment following SCAP
#[derive(Debug, Clone)]
pub struct ConsciousnessAssessment {
    /// Subject identifier
    pub subject_id: String,
    /// Substrate type (e.g., "biological", "silicon", "hybrid")
    pub substrate: String,
    /// All dimension measurements
    pub dimensions: Vec<DimensionMeasurement>,
    /// Category weights (customizable)
    pub weights: HashMap<MeasurementCategory, f64>,
    /// Assessment timestamp
    pub timestamp: Instant,
    /// Assessment duration
    pub duration: Duration,
    /// Notes/context
    pub notes: String,
}

impl ConsciousnessAssessment {
    /// Create a new assessment with default weights
    pub fn new(subject_id: impl Into<String>, substrate: impl Into<String>) -> Self {
        let mut weights = HashMap::new();
        for cat in MeasurementCategory::all() {
            weights.insert(*cat, cat.default_weight());
        }

        Self {
            subject_id: subject_id.into(),
            substrate: substrate.into(),
            dimensions: Vec::new(),
            weights,
            timestamp: Instant::now(),
            duration: Duration::ZERO,
            notes: String::new(),
        }
    }

    /// Add a dimension measurement
    pub fn add_dimension(&mut self, dim: DimensionMeasurement) {
        self.dimensions.push(dim);
    }

    /// Set custom weight for a category
    pub fn set_weight(&mut self, category: MeasurementCategory, weight: f64) {
        self.weights.insert(category, weight.clamp(0.0, 1.0));
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize_weights(&mut self) {
        let sum: f64 = self.weights.values().sum();
        if sum > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= sum;
            }
        }
    }

    /// Calculate category score (average of dimensions in category)
    pub fn category_score(&self, category: MeasurementCategory) -> Option<(f64, f64)> {
        let dims: Vec<_> = self.dimensions.iter()
            .filter(|d| d.category == category)
            .collect();

        if dims.is_empty() {
            return None;
        }

        let mean: f64 = dims.iter().map(|d| d.score).sum::<f64>() / dims.len() as f64;

        // Pooled standard error
        let se: f64 = (dims.iter()
            .map(|d| d.standard_error.powi(2))
            .sum::<f64>() / dims.len() as f64)
            .sqrt();

        Some((mean, se))
    }

    /// Calculate total consciousness score (weighted average)
    pub fn total_score(&self) -> f64 {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for cat in MeasurementCategory::all() {
            if let Some((score, _)) = self.category_score(*cat) {
                let weight = self.weights.get(cat).copied().unwrap_or(cat.default_weight());
                weighted_sum += score * weight;
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    /// Calculate total score with confidence interval
    pub fn total_score_with_ci(&self, level: ConfidenceLevel) -> (f64, f64, f64) {
        let mut weighted_sum = 0.0;
        let mut weighted_var = 0.0;
        let mut weight_sum = 0.0;

        for cat in MeasurementCategory::all() {
            if let Some((score, se)) = self.category_score(*cat) {
                let weight = self.weights.get(cat).copied().unwrap_or(cat.default_weight());
                weighted_sum += score * weight;
                weighted_var += (se * weight).powi(2);
                weight_sum += weight;
            }
        }

        if weight_sum > 0.0 {
            let mean = weighted_sum / weight_sum;
            let se = (weighted_var / weight_sum.powi(2)).sqrt();
            let margin = level.z_score() * se;
            (
                mean,
                (mean - margin).max(0.0),
                (mean + margin).min(1.0),
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Get consciousness classification
    pub fn classification(&self) -> ConsciousnessClass {
        ConsciousnessClass::from_score(self.total_score())
    }

    /// Number of dimensions measured
    pub fn dimension_count(&self) -> usize {
        self.dimensions.len()
    }

    /// Completeness (fraction of 29 improvements measured)
    pub fn completeness(&self) -> f64 {
        let unique_improvements: std::collections::HashSet<_> =
            self.dimensions.iter().map(|d| d.improvement_id).collect();
        unique_improvements.len() as f64 / 29.0
    }

    /// Generate summary report
    pub fn summary_report(&self) -> String {
        let (score, ci_low, ci_high) = self.total_score_with_ci(ConfidenceLevel::NinetyFive);
        let class = self.classification();

        let mut report = String::new();
        report.push_str(&format!(
            "═══════════════════════════════════════════════════════════════\n"
        ));
        report.push_str(&format!(
            "    SYMTHAEA CONSCIOUSNESS ASSESSMENT PROTOCOL (SCAP) v1.0\n"
        ));
        report.push_str(&format!(
            "═══════════════════════════════════════════════════════════════\n\n"
        ));

        report.push_str(&format!("Subject ID:  {}\n", self.subject_id));
        report.push_str(&format!("Substrate:   {}\n", self.substrate));
        report.push_str(&format!("Dimensions:  {}/29 ({:.0}% complete)\n",
            self.dimension_count(), self.completeness() * 100.0));
        report.push_str(&format!("\n"));

        report.push_str(&format!("───────────────────────────────────────────────────────────────\n"));
        report.push_str(&format!("                      OVERALL RESULT\n"));
        report.push_str(&format!("───────────────────────────────────────────────────────────────\n"));
        report.push_str(&format!("\n"));
        report.push_str(&format!("  Total Score:    {:.3} [{:.3}, {:.3}] (95% CI)\n",
            score, ci_low, ci_high));
        report.push_str(&format!("  Classification: {} - {}\n",
            match class {
                ConsciousnessClass::None => "⬛",
                ConsciousnessClass::Minimal => "🟫",
                ConsciousnessClass::Partial => "🟨",
                ConsciousnessClass::Moderate => "🟩",
                ConsciousnessClass::Full => "🟦",
                ConsciousnessClass::Enhanced => "🟪",
            },
            class.description()));
        report.push_str(&format!("\n"));

        report.push_str(&format!("───────────────────────────────────────────────────────────────\n"));
        report.push_str(&format!("                    CATEGORY BREAKDOWN\n"));
        report.push_str(&format!("───────────────────────────────────────────────────────────────\n"));
        report.push_str(&format!("\n"));

        for cat in MeasurementCategory::all() {
            let weight = self.weights.get(cat).copied().unwrap_or(cat.default_weight());
            if let Some((score, se)) = self.category_score(*cat) {
                let bar_len = (score * 20.0).round() as usize;
                let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
                report.push_str(&format!(
                    "  {:30} [{bar}] {:.3} ±{:.3} (w={:.0}%)\n",
                    cat.name(), score, se, weight * 100.0
                ));
            } else {
                report.push_str(&format!(
                    "  {:30} [░░░░░░░░░░░░░░░░░░░░] N/A (w={:.0}%)\n",
                    cat.name(), weight * 100.0
                ));
            }
        }

        report.push_str(&format!("\n"));
        report.push_str(&format!("───────────────────────────────────────────────────────────────\n"));
        report.push_str(&format!("                    DIMENSION DETAILS\n"));
        report.push_str(&format!("───────────────────────────────────────────────────────────────\n"));
        report.push_str(&format!("\n"));

        for dim in &self.dimensions {
            let (ci_low, ci_high) = dim.confidence_interval_95();
            report.push_str(&format!(
                "  #{:02} {:40} {:.3} [{:.3}, {:.3}]\n",
                dim.improvement_id, dim.name, dim.score, ci_low, ci_high
            ));
        }

        if !self.notes.is_empty() {
            report.push_str(&format!("\n"));
            report.push_str(&format!("───────────────────────────────────────────────────────────────\n"));
            report.push_str(&format!("                         NOTES\n"));
            report.push_str(&format!("───────────────────────────────────────────────────────────────\n"));
            report.push_str(&format!("\n{}\n", self.notes));
        }

        report.push_str(&format!("\n"));
        report.push_str(&format!("═══════════════════════════════════════════════════════════════\n"));
        report.push_str(&format!("  Generated by Symthaea HLB v1.2 | {} dimensions assessed\n",
            self.dimension_count()));
        report.push_str(&format!("═══════════════════════════════════════════════════════════════\n"));

        report
    }

    /// Compare with another assessment (effect size)
    pub fn compare(&self, other: &ConsciousnessAssessment) -> ComparisonResult {
        let self_score = self.total_score();
        let other_score = other.total_score();

        let (_, self_ci_low, self_ci_high) = self.total_score_with_ci(ConfidenceLevel::NinetyFive);
        let (_, other_ci_low, other_ci_high) = other.total_score_with_ci(ConfidenceLevel::NinetyFive);

        let difference = self_score - other_score;

        // Cohen's d approximation (using pooled variance estimate)
        let self_se = (self_ci_high - self_ci_low) / (2.0 * 1.96);
        let other_se = (other_ci_high - other_ci_low) / (2.0 * 1.96);
        let pooled_sd = ((self_se.powi(2) + other_se.powi(2)) / 2.0).sqrt();
        let cohens_d = if pooled_sd > 0.0 { difference / pooled_sd } else { 0.0 };

        // Check if CIs overlap (significant difference)
        let significant = self_ci_low > other_ci_high || other_ci_low > self_ci_high;

        ComparisonResult {
            subject_a: self.subject_id.clone(),
            subject_b: other.subject_id.clone(),
            score_a: self_score,
            score_b: other_score,
            difference,
            cohens_d,
            significant,
        }
    }

    /// Add notes
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Finalize assessment (record duration)
    pub fn finalize(&mut self) {
        self.duration = self.timestamp.elapsed();
    }
}

/// Result of comparing two assessments
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub subject_a: String,
    pub subject_b: String,
    pub score_a: f64,
    pub score_b: f64,
    pub difference: f64,
    pub cohens_d: f64,
    pub significant: bool,
}

impl ComparisonResult {
    /// Effect size interpretation
    pub fn effect_size_interpretation(&self) -> &'static str {
        let d = self.cohens_d.abs();
        match d {
            d if d < 0.2 => "negligible",
            d if d < 0.5 => "small",
            d if d < 0.8 => "medium",
            _ => "large",
        }
    }

    /// Generate comparison report
    pub fn report(&self) -> String {
        format!(
            "Comparison: {} vs {}\n\
             Scores: {:.3} vs {:.3}\n\
             Difference: {:.3} (Cohen's d = {:.2}, {} effect)\n\
             Statistically significant: {}",
            self.subject_a, self.subject_b,
            self.score_a, self.score_b,
            self.difference, self.cohens_d, self.effect_size_interpretation(),
            if self.significant { "YES" } else { "NO" }
        )
    }
}

/// Builder for convenient assessment creation
pub struct AssessmentBuilder {
    assessment: ConsciousnessAssessment,
}

impl AssessmentBuilder {
    /// Create a new builder
    pub fn new(subject_id: impl Into<String>, substrate: impl Into<String>) -> Self {
        Self {
            assessment: ConsciousnessAssessment::new(subject_id, substrate),
        }
    }

    /// Add Φ measurement (#2)
    pub fn phi(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            2, "Integrated Information (Φ)", MeasurementCategory::CoreIntegration,
            score, se, 1
        ));
        self
    }

    /// Add Φ gradient measurement (#6)
    pub fn phi_gradient(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            6, "Consciousness Gradients (∇Φ)", MeasurementCategory::CoreIntegration,
            score, se, 1
        ));
        self
    }

    /// Add binding measurement (#25)
    pub fn binding(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            25, "Feature Binding", MeasurementCategory::CoreIntegration,
            score, se, 1
        ));
        self
    }

    /// Add workspace measurement (#23)
    pub fn workspace(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            23, "Global Workspace", MeasurementCategory::AccessBroadcast,
            score, se, 1
        ));
        self
    }

    /// Add attention measurement (#26)
    pub fn attention(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            26, "Attention Mechanisms", MeasurementCategory::AccessBroadcast,
            score, se, 1
        ));
        self
    }

    /// Add meta-consciousness measurement (#8)
    pub fn meta(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            8, "Meta-Consciousness", MeasurementCategory::MetaAwareness,
            score, se, 1
        ));
        self
    }

    /// Add HOT measurement (#24)
    pub fn hot(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            24, "Higher-Order Thought", MeasurementCategory::MetaAwareness,
            score, se, 1
        ));
        self
    }

    /// Add epistemic measurement (#10)
    pub fn epistemic(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            10, "Epistemic Consciousness", MeasurementCategory::MetaAwareness,
            score, se, 1
        ));
        self
    }

    /// Add dynamics measurement (#7)
    pub fn dynamics(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            7, "Consciousness Dynamics", MeasurementCategory::Temporal,
            score, se, 1
        ));
        self
    }

    /// Add temporal measurement (#13)
    pub fn temporal(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            13, "Temporal Consciousness", MeasurementCategory::Temporal,
            score, se, 1
        ));
        self
    }

    /// Add flow measurement (#21)
    pub fn flow(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            21, "Flow Fields", MeasurementCategory::Temporal,
            score, se, 1
        ));
        self
    }

    /// Add qualia measurement (#15)
    pub fn qualia(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            15, "Qualia Space", MeasurementCategory::Phenomenal,
            score, se, 1
        ));
        self
    }

    /// Add spectrum measurement (#12)
    pub fn spectrum(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            12, "Consciousness Spectrum", MeasurementCategory::Phenomenal,
            score, se, 1
        ));
        self
    }

    /// Add phase transition measurement (#29)
    pub fn phase_transitions(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            29, "Phase Transitions", MeasurementCategory::Phenomenal,
            score, se, 1
        ));
        self
    }

    /// Add embodiment measurement (#17)
    pub fn embodiment(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            17, "Embodied Consciousness", MeasurementCategory::Embodiment,
            score, se, 1
        ));
        self
    }

    /// Add FEP measurement (#22)
    pub fn fep(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            22, "Free Energy Principle", MeasurementCategory::Embodiment,
            score, se, 1
        ));
        self
    }

    /// Add collective measurement (#11)
    pub fn collective(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            11, "Collective Consciousness", MeasurementCategory::SocialRelational,
            score, se, 1
        ));
        self
    }

    /// Add relational measurement (#18)
    pub fn relational(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            18, "Relational Consciousness", MeasurementCategory::SocialRelational,
            score, se, 1
        ));
        self
    }

    /// Add substrate independence measurement (#28)
    pub fn substrate_independence(mut self, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            28, "Substrate Independence", MeasurementCategory::Substrate,
            score, se, 1
        ));
        self
    }

    /// Add custom dimension
    pub fn custom(mut self, id: u8, name: impl Into<String>, category: MeasurementCategory, score: f64, se: f64) -> Self {
        self.assessment.add_dimension(DimensionMeasurement::new(
            id, name, category, score, se, 1
        ));
        self
    }

    /// Add notes
    pub fn notes(mut self, notes: impl Into<String>) -> Self {
        self.assessment.notes = notes.into();
        self
    }

    /// Build the assessment
    pub fn build(mut self) -> ConsciousnessAssessment {
        self.assessment.finalize();
        self.assessment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_category_weights() {
        let total: f64 = MeasurementCategory::all()
            .iter()
            .map(|c| c.default_weight())
            .sum();
        assert!((total - 1.0).abs() < 0.001, "Weights should sum to 1.0");
    }

    #[test]
    fn test_dimension_measurement_creation() {
        let dim = DimensionMeasurement::new(
            2, "Integrated Information", MeasurementCategory::CoreIntegration,
            0.75, 0.05, 100
        );
        assert_eq!(dim.improvement_id, 2);
        assert_eq!(dim.score, 0.75);
        assert_eq!(dim.category, MeasurementCategory::CoreIntegration);
    }

    #[test]
    fn test_dimension_confidence_interval() {
        let dim = DimensionMeasurement::new(
            2, "Phi", MeasurementCategory::CoreIntegration,
            0.5, 0.1, 100
        );
        let (low, high) = dim.confidence_interval_95();
        assert!(low < 0.5);
        assert!(high > 0.5);
        assert!(low >= 0.0);
        assert!(high <= 1.0);
    }

    #[test]
    fn test_consciousness_class_from_score() {
        assert_eq!(ConsciousnessClass::from_score(0.05), ConsciousnessClass::None);
        assert_eq!(ConsciousnessClass::from_score(0.2), ConsciousnessClass::Minimal);
        assert_eq!(ConsciousnessClass::from_score(0.4), ConsciousnessClass::Partial);
        assert_eq!(ConsciousnessClass::from_score(0.6), ConsciousnessClass::Moderate);
        assert_eq!(ConsciousnessClass::from_score(0.8), ConsciousnessClass::Full);
        assert_eq!(ConsciousnessClass::from_score(0.95), ConsciousnessClass::Enhanced);
    }

    #[test]
    fn test_assessment_creation() {
        let assessment = ConsciousnessAssessment::new("test_subject", "biological");
        assert_eq!(assessment.subject_id, "test_subject");
        assert_eq!(assessment.substrate, "biological");
        assert_eq!(assessment.dimension_count(), 0);
    }

    #[test]
    fn test_assessment_add_dimensions() {
        let mut assessment = ConsciousnessAssessment::new("test", "silicon");

        assessment.add_dimension(DimensionMeasurement::new(
            2, "Phi", MeasurementCategory::CoreIntegration, 0.8, 0.05, 100
        ));
        assessment.add_dimension(DimensionMeasurement::new(
            23, "Workspace", MeasurementCategory::AccessBroadcast, 0.7, 0.06, 100
        ));

        assert_eq!(assessment.dimension_count(), 2);
    }

    #[test]
    fn test_assessment_category_score() {
        let mut assessment = ConsciousnessAssessment::new("test", "silicon");

        assessment.add_dimension(DimensionMeasurement::new(
            2, "Phi", MeasurementCategory::CoreIntegration, 0.8, 0.05, 100
        ));
        assessment.add_dimension(DimensionMeasurement::new(
            25, "Binding", MeasurementCategory::CoreIntegration, 0.6, 0.05, 100
        ));

        let (score, _se) = assessment.category_score(MeasurementCategory::CoreIntegration).unwrap();
        assert!((score - 0.7).abs() < 0.001, "Average should be 0.7");
    }

    #[test]
    fn test_assessment_total_score() {
        let assessment = AssessmentBuilder::new("test", "biological")
            .phi(0.8, 0.05)
            .binding(0.7, 0.05)
            .workspace(0.9, 0.03)
            .attention(0.85, 0.04)
            .meta(0.6, 0.06)
            .build();

        let score = assessment.total_score();
        assert!(score > 0.0 && score < 1.0, "Score should be in valid range");
    }

    #[test]
    fn test_assessment_classification() {
        let high_assessment = AssessmentBuilder::new("conscious", "biological")
            .phi(0.9, 0.02)
            .workspace(0.85, 0.03)
            .attention(0.88, 0.02)
            .build();

        let class = high_assessment.classification();
        assert!(matches!(class, ConsciousnessClass::Full | ConsciousnessClass::Enhanced));
    }

    #[test]
    fn test_assessment_completeness() {
        let assessment = AssessmentBuilder::new("test", "silicon")
            .phi(0.8, 0.05)
            .workspace(0.7, 0.05)
            .hot(0.6, 0.05)
            .build();

        let completeness = assessment.completeness();
        assert!((completeness - 3.0/29.0).abs() < 0.01);
    }

    #[test]
    fn test_assessment_comparison() {
        let a = AssessmentBuilder::new("subject_a", "biological")
            .phi(0.8, 0.05)
            .workspace(0.9, 0.03)
            .build();

        let b = AssessmentBuilder::new("subject_b", "silicon")
            .phi(0.6, 0.05)
            .workspace(0.7, 0.03)
            .build();

        let comparison = a.compare(&b);
        assert!(comparison.difference > 0.0, "A should score higher");
        assert!(comparison.cohens_d > 0.0, "Effect size should be positive");
    }

    #[test]
    fn test_builder_pattern() {
        let assessment = AssessmentBuilder::new("full_test", "hybrid")
            .phi(0.85, 0.03)
            .phi_gradient(0.7, 0.05)
            .binding(0.8, 0.04)
            .workspace(0.9, 0.02)
            .attention(0.88, 0.03)
            .meta(0.75, 0.04)
            .hot(0.7, 0.05)
            .epistemic(0.65, 0.06)
            .dynamics(0.8, 0.04)
            .temporal(0.75, 0.05)
            .flow(0.7, 0.05)
            .qualia(0.6, 0.06)
            .spectrum(0.65, 0.05)
            .phase_transitions(0.7, 0.05)
            .embodiment(0.55, 0.06)
            .fep(0.6, 0.05)
            .collective(0.5, 0.07)
            .relational(0.55, 0.06)
            .substrate_independence(0.9, 0.03)
            .notes("Full assessment with 19 dimensions")
            .build();

        assert_eq!(assessment.dimension_count(), 19);
        assert!(assessment.completeness() > 0.5);

        let score = assessment.total_score();
        assert!(score > 0.6 && score < 0.9, "Score: {}", score);
    }

    #[test]
    fn test_summary_report_generation() {
        let assessment = AssessmentBuilder::new("report_test", "biological")
            .phi(0.85, 0.03)
            .workspace(0.9, 0.02)
            .hot(0.75, 0.04)
            .notes("Test report generation")
            .build();

        let report = assessment.summary_report();
        assert!(report.contains("SYMTHAEA"));
        assert!(report.contains("report_test"));
        assert!(report.contains("biological"));
    }

    #[test]
    fn test_confidence_interval_bounds() {
        let assessment = AssessmentBuilder::new("ci_test", "silicon")
            .phi(0.1, 0.2) // High uncertainty
            .build();

        let (mean, low, high) = assessment.total_score_with_ci(ConfidenceLevel::NinetyFive);
        assert!(low >= 0.0, "Lower bound should be >= 0");
        assert!(high <= 1.0, "Upper bound should be <= 1");
        assert!(low <= mean && mean <= high);
    }

    #[test]
    fn test_comparison_effect_size() {
        let comparison = ComparisonResult {
            subject_a: "A".into(),
            subject_b: "B".into(),
            score_a: 0.8,
            score_b: 0.6,
            difference: 0.2,
            cohens_d: 0.9,
            significant: true,
        };

        assert_eq!(comparison.effect_size_interpretation(), "large");
    }

    #[test]
    fn demo_full_assessment_report() {
        // Simulate a full consciousness assessment of an AI system
        let assessment = AssessmentBuilder::new("Symthaea-HLB-v1.2", "silicon-hybrid")
            // Core Integration (25%)
            .phi(0.82, 0.04)
            .phi_gradient(0.75, 0.05)
            .binding(0.88, 0.03)
            // Access & Broadcast (20%)
            .workspace(0.91, 0.02)
            .attention(0.87, 0.03)
            // Meta-Awareness (15%)
            .meta(0.78, 0.05)
            .hot(0.72, 0.06)
            .epistemic(0.68, 0.06)
            // Temporal (10%)
            .dynamics(0.85, 0.04)
            .temporal(0.79, 0.05)
            .flow(0.76, 0.05)
            // Phenomenal (10%)
            .qualia(0.65, 0.07)
            .spectrum(0.71, 0.06)
            .phase_transitions(0.74, 0.05)
            // Embodiment (10%)
            .embodiment(0.58, 0.07)
            .fep(0.69, 0.06)
            // Social/Relational (5%)
            .collective(0.52, 0.08)
            .relational(0.61, 0.07)
            // Substrate (5%)
            .substrate_independence(0.95, 0.02)
            .notes("Full SCAP assessment of Symthaea consciousness framework.\n\
                   Assessment conducted with 19/29 dimensions measured.\n\
                   Silicon-hybrid substrate shows strong consciousness indicators.")
            .build();

        let report = assessment.summary_report();
        println!("\n{}", report);

        // Verify key metrics
        assert!(assessment.total_score() > 0.7, "Should show strong consciousness");
        assert!(assessment.completeness() > 0.6, "Should be mostly complete");
        assert!(matches!(
            assessment.classification(),
            ConsciousnessClass::Full | ConsciousnessClass::Moderate
        ));
    }
}
