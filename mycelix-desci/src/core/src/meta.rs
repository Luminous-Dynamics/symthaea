// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Meta-Claims & Systematic Reviews
//!
//! Implements higher-order claims about collections of claims - meta-analyses,
//! systematic reviews, and aggregate findings synthesis.

use std::collections::HashMap;
use uuid::Uuid;

/// Type of meta-claim
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaClaimType {
    /// Statistical combination of multiple study results
    MetaAnalysis,
    /// Comprehensive review following systematic methodology
    SystematicReview,
    /// Narrative synthesis of related claims
    NarrativeReview,
    /// Statistical replication of findings
    ReplicationStudy,
    /// Umbrella review (review of reviews)
    UmbrellaReview,
    /// Rapid evidence assessment
    RapidReview,
    /// Scoping review of field
    ScopingReview,
}

impl MetaClaimType {
    /// Get evidence strength multiplier for this meta-claim type
    pub fn evidence_multiplier(&self) -> f64 {
        match self {
            MetaClaimType::MetaAnalysis => 1.5,
            MetaClaimType::SystematicReview => 1.4,
            MetaClaimType::UmbrellaReview => 1.6,
            MetaClaimType::ReplicationStudy => 1.3,
            MetaClaimType::NarrativeReview => 1.1,
            MetaClaimType::ScopingReview => 1.0,
            MetaClaimType::RapidReview => 0.9,
        }
    }

    /// Get minimum required source claims
    pub fn min_sources(&self) -> usize {
        match self {
            MetaClaimType::MetaAnalysis => 3,
            MetaClaimType::SystematicReview => 5,
            MetaClaimType::UmbrellaReview => 3,
            MetaClaimType::ReplicationStudy => 1,
            MetaClaimType::NarrativeReview => 3,
            MetaClaimType::ScopingReview => 5,
            MetaClaimType::RapidReview => 2,
        }
    }
}

/// Quality assessment of source studies
#[derive(Debug, Clone)]
pub struct SourceQualityAssessment {
    /// Source claim ID
    pub source_id: Uuid,
    /// Quality score (0.0-1.0)
    pub quality_score: f64,
    /// Risk of bias assessment
    pub risk_of_bias: RiskOfBias,
    /// Weight in meta-analysis
    pub weight: f64,
    /// Sample size (if applicable)
    pub sample_size: Option<usize>,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
    /// Standard error (if applicable)
    pub standard_error: Option<f64>,
}

/// Risk of bias classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskOfBias {
    /// Low risk of bias
    Low,
    /// Some concerns
    SomeConcerns,
    /// High risk of bias
    High,
    /// Not assessed
    NotAssessed,
}

impl RiskOfBias {
    /// Get weight modifier for bias risk
    pub fn weight_modifier(&self) -> f64 {
        match self {
            RiskOfBias::Low => 1.0,
            RiskOfBias::SomeConcerns => 0.8,
            RiskOfBias::High => 0.5,
            RiskOfBias::NotAssessed => 0.7,
        }
    }
}

/// PRISMA-style flow diagram for systematic reviews
#[derive(Debug, Clone)]
pub struct PrismaFlow {
    /// Records identified from databases
    pub records_identified: usize,
    /// Duplicate records removed
    pub duplicates_removed: usize,
    /// Records screened
    pub records_screened: usize,
    /// Records excluded at screening
    pub records_excluded_screening: usize,
    /// Full-text articles assessed
    pub full_texts_assessed: usize,
    /// Full-text articles excluded
    pub full_texts_excluded: usize,
    /// Reasons for exclusion
    pub exclusion_reasons: HashMap<String, usize>,
    /// Studies included in review
    pub studies_included: usize,
    /// Studies included in meta-analysis
    pub studies_in_meta_analysis: usize,
}

impl PrismaFlow {
    /// Create a new PRISMA flow
    pub fn new(records_identified: usize) -> Self {
        Self {
            records_identified,
            duplicates_removed: 0,
            records_screened: 0,
            records_excluded_screening: 0,
            full_texts_assessed: 0,
            full_texts_excluded: 0,
            exclusion_reasons: HashMap::new(),
            studies_included: 0,
            studies_in_meta_analysis: 0,
        }
    }

    /// Calculate inclusion rate
    pub fn inclusion_rate(&self) -> f64 {
        if self.records_identified == 0 {
            return 0.0;
        }
        self.studies_included as f64 / self.records_identified as f64
    }
}

/// A meta-claim that synthesizes multiple source claims
#[derive(Debug, Clone)]
pub struct MetaClaim {
    /// Unique identifier
    pub id: Uuid,
    /// Type of meta-claim
    pub meta_type: MetaClaimType,
    /// Source claim IDs included
    pub source_claims: Vec<Uuid>,
    /// Quality assessments for each source
    pub source_assessments: Vec<SourceQualityAssessment>,
    /// PRISMA flow (for systematic reviews)
    pub prisma_flow: Option<PrismaFlow>,
    /// Overall synthesis result
    pub synthesis: SynthesisResult,
    /// Heterogeneity assessment
    pub heterogeneity: HeterogeneityAssessment,
    /// GRADE quality of evidence
    pub grade_quality: GradeQuality,
    /// Confidence in the meta-claim
    pub confidence: f64,
    /// Creator/author
    pub creator: String,
    /// Creation timestamp
    pub created_at: i64,
}

impl MetaClaim {
    /// Create a new meta-claim
    pub fn new(meta_type: MetaClaimType, creator: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            meta_type,
            source_claims: Vec::new(),
            source_assessments: Vec::new(),
            prisma_flow: None,
            synthesis: SynthesisResult::default(),
            heterogeneity: HeterogeneityAssessment::default(),
            grade_quality: GradeQuality::VeryLow,
            confidence: 0.0,
            creator,
            created_at: 0,
        }
    }

    /// Add a source claim with assessment
    pub fn add_source(&mut self, source_id: Uuid, assessment: SourceQualityAssessment) {
        self.source_claims.push(source_id);
        self.source_assessments.push(assessment);
    }

    /// Check if minimum sources are met
    pub fn meets_minimum_sources(&self) -> bool {
        self.source_claims.len() >= self.meta_type.min_sources()
    }

    /// Calculate weighted average quality
    pub fn average_source_quality(&self) -> f64 {
        if self.source_assessments.is_empty() {
            return 0.0;
        }

        let total_weight: f64 = self.source_assessments.iter().map(|a| a.weight).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        let weighted_sum: f64 = self
            .source_assessments
            .iter()
            .map(|a| a.quality_score * a.weight)
            .sum();

        weighted_sum / total_weight
    }

    /// Calculate overall confidence
    pub fn calculate_confidence(&mut self) {
        let source_quality = self.average_source_quality();
        let heterogeneity_penalty = self.heterogeneity.confidence_penalty();
        let grade_factor = self.grade_quality.confidence_factor();
        let source_count_factor = (self.source_claims.len() as f64 / 10.0).min(1.0);
        let type_multiplier = self.meta_type.evidence_multiplier();

        self.confidence = ((source_quality * 0.4)
            + (source_count_factor * 0.2)
            + (grade_factor * 0.3)
            + (0.1)) // Base
            * type_multiplier
            * (1.0 - heterogeneity_penalty);

        self.confidence = self.confidence.clamp(0.0, 1.0);
    }
}

/// Result of meta-analysis synthesis
#[derive(Debug, Clone, Default)]
pub struct SynthesisResult {
    /// Pooled effect size
    pub pooled_effect: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
    /// P-value (if applicable)
    pub p_value: Option<f64>,
    /// Direction of effect
    pub effect_direction: EffectDirection,
    /// Synthesis method used
    pub method: SynthesisMethod,
    /// Narrative conclusion
    pub conclusion: String,
}

/// Direction of effect in synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffectDirection {
    /// Positive/supportive effect
    Positive,
    /// Negative/contrary effect
    Negative,
    /// No clear effect
    #[default]
    Neutral,
    /// Mixed effects
    Mixed,
}

/// Method used for synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SynthesisMethod {
    /// Fixed-effects meta-analysis
    FixedEffects,
    /// Random-effects meta-analysis
    RandomEffects,
    /// Bayesian meta-analysis
    Bayesian,
    /// Vote counting
    VoteCounting,
    /// Narrative synthesis
    #[default]
    Narrative,
}

/// Heterogeneity assessment
#[derive(Debug, Clone, Default)]
pub struct HeterogeneityAssessment {
    /// I² statistic (0-100%)
    pub i_squared: f64,
    /// Q statistic
    pub q_statistic: f64,
    /// Tau² (between-study variance)
    pub tau_squared: f64,
    /// Classification of heterogeneity
    pub classification: HeterogeneityLevel,
    /// Potential sources of heterogeneity
    pub potential_sources: Vec<String>,
}

impl HeterogeneityAssessment {
    /// Get confidence penalty for heterogeneity
    pub fn confidence_penalty(&self) -> f64 {
        match self.classification {
            HeterogeneityLevel::None => 0.0,
            HeterogeneityLevel::Low => 0.05,
            HeterogeneityLevel::Moderate => 0.15,
            HeterogeneityLevel::Substantial => 0.25,
            HeterogeneityLevel::Considerable => 0.35,
        }
    }
}

/// Level of heterogeneity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeterogeneityLevel {
    /// I² < 25%
    #[default]
    None,
    /// I² 25-49%
    Low,
    /// I² 50-74%
    Moderate,
    /// I² 75-89%
    Substantial,
    /// I² >= 90%
    Considerable,
}

impl HeterogeneityLevel {
    /// Classify from I² value
    pub fn from_i_squared(i_squared: f64) -> Self {
        if i_squared < 25.0 {
            HeterogeneityLevel::None
        } else if i_squared < 50.0 {
            HeterogeneityLevel::Low
        } else if i_squared < 75.0 {
            HeterogeneityLevel::Moderate
        } else if i_squared < 90.0 {
            HeterogeneityLevel::Substantial
        } else {
            HeterogeneityLevel::Considerable
        }
    }
}

/// GRADE quality of evidence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradeQuality {
    /// High quality evidence
    High,
    /// Moderate quality evidence
    Moderate,
    /// Low quality evidence
    Low,
    /// Very low quality evidence
    #[default]
    VeryLow,
}

impl GradeQuality {
    /// Get confidence factor for this grade
    pub fn confidence_factor(&self) -> f64 {
        match self {
            GradeQuality::High => 1.0,
            GradeQuality::Moderate => 0.8,
            GradeQuality::Low => 0.5,
            GradeQuality::VeryLow => 0.3,
        }
    }
}

/// Meta-claim synthesizer
pub struct MetaSynthesizer {
    /// Minimum sources for valid synthesis
    pub min_sources: usize,
    /// Default synthesis method
    pub default_method: SynthesisMethod,
    /// Whether to require PRISMA flow
    pub require_prisma: bool,
}

impl MetaSynthesizer {
    /// Create a new synthesizer with defaults
    pub fn new() -> Self {
        Self {
            min_sources: 3,
            default_method: SynthesisMethod::RandomEffects,
            require_prisma: true,
        }
    }

    /// Perform fixed-effects meta-analysis
    pub fn fixed_effects_analysis(&self, assessments: &[SourceQualityAssessment]) -> SynthesisResult {
        if assessments.is_empty() {
            return SynthesisResult::default();
        }

        // Simple weighted average of effect sizes
        let valid: Vec<_> = assessments
            .iter()
            .filter(|a| a.effect_size.is_some() && a.standard_error.is_some())
            .collect();

        if valid.is_empty() {
            return SynthesisResult {
                method: SynthesisMethod::Narrative,
                conclusion: "Insufficient data for quantitative synthesis".into(),
                ..Default::default()
            };
        }

        // Inverse variance weights
        let mut total_weight = 0.0;
        let mut weighted_effect = 0.0;

        for assessment in &valid {
            let se = assessment.standard_error.unwrap();
            let effect = assessment.effect_size.unwrap();
            let weight = 1.0 / (se * se);
            total_weight += weight;
            weighted_effect += weight * effect;
        }

        let pooled = weighted_effect / total_weight;
        let pooled_se = (1.0 / total_weight).sqrt();

        // 95% CI
        let z = 1.96;
        let ci_lower = pooled - z * pooled_se;
        let ci_upper = pooled + z * pooled_se;

        let direction = if pooled > 0.1 {
            EffectDirection::Positive
        } else if pooled < -0.1 {
            EffectDirection::Negative
        } else {
            EffectDirection::Neutral
        };

        SynthesisResult {
            pooled_effect: pooled,
            ci_lower,
            ci_upper,
            p_value: None, // Would need full calculation
            effect_direction: direction,
            method: SynthesisMethod::FixedEffects,
            conclusion: format!(
                "Pooled effect: {:.3} (95% CI: {:.3} to {:.3})",
                pooled, ci_lower, ci_upper
            ),
        }
    }

    /// Calculate heterogeneity
    pub fn calculate_heterogeneity(
        &self,
        assessments: &[SourceQualityAssessment],
        pooled_effect: f64,
    ) -> HeterogeneityAssessment {
        if assessments.len() < 2 {
            return HeterogeneityAssessment::default();
        }

        let valid: Vec<_> = assessments
            .iter()
            .filter(|a| a.effect_size.is_some() && a.standard_error.is_some())
            .collect();

        if valid.len() < 2 {
            return HeterogeneityAssessment::default();
        }

        // Calculate Q statistic
        let mut q = 0.0;
        for assessment in &valid {
            let se = assessment.standard_error.unwrap();
            let effect = assessment.effect_size.unwrap();
            let weight = 1.0 / (se * se);
            q += weight * (effect - pooled_effect).powi(2);
        }

        let df = valid.len() as f64 - 1.0;

        // I² = (Q - df) / Q * 100
        let i_squared = if q > df { (q - df) / q * 100.0 } else { 0.0 };

        let classification = HeterogeneityLevel::from_i_squared(i_squared);

        HeterogeneityAssessment {
            i_squared,
            q_statistic: q,
            tau_squared: 0.0, // Would need DerSimonian-Laird calculation
            classification,
            potential_sources: Vec::new(),
        }
    }

    /// Assess GRADE quality
    pub fn assess_grade(&self, meta_claim: &MetaClaim) -> GradeQuality {
        let mut grade_score = 4; // Start at high

        // Risk of bias
        let high_bias_count = meta_claim
            .source_assessments
            .iter()
            .filter(|a| a.risk_of_bias == RiskOfBias::High)
            .count();
        if high_bias_count > meta_claim.source_claims.len() / 2 {
            grade_score -= 1;
        }

        // Heterogeneity
        if meta_claim.heterogeneity.classification == HeterogeneityLevel::Substantial
            || meta_claim.heterogeneity.classification == HeterogeneityLevel::Considerable
        {
            grade_score -= 1;
        }

        // Small sample size
        let total_samples: usize = meta_claim
            .source_assessments
            .iter()
            .filter_map(|a| a.sample_size)
            .sum();
        if total_samples < 100 {
            grade_score -= 1;
        }

        // Imprecision (wide CIs)
        let ci_width = meta_claim.synthesis.ci_upper - meta_claim.synthesis.ci_lower;
        if ci_width > 0.5 {
            grade_score -= 1;
        }

        match grade_score.max(0) {
            4 => GradeQuality::High,
            3 => GradeQuality::Moderate,
            2 => GradeQuality::Low,
            _ => GradeQuality::VeryLow,
        }
    }

    /// Synthesize a complete meta-claim
    pub fn synthesize(&self, mut meta_claim: MetaClaim) -> MetaClaim {
        if !meta_claim.meets_minimum_sources() {
            meta_claim.confidence = 0.0;
            meta_claim.synthesis.conclusion = "Insufficient sources for synthesis".into();
            return meta_claim;
        }

        // Perform synthesis
        meta_claim.synthesis = self.fixed_effects_analysis(&meta_claim.source_assessments);

        // Calculate heterogeneity
        meta_claim.heterogeneity = self.calculate_heterogeneity(
            &meta_claim.source_assessments,
            meta_claim.synthesis.pooled_effect,
        );

        // Assess GRADE quality
        meta_claim.grade_quality = self.assess_grade(&meta_claim);

        // Calculate overall confidence
        meta_claim.calculate_confidence();

        meta_claim
    }
}

impl Default for MetaSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_assessment(effect: f64, se: f64, quality: f64) -> SourceQualityAssessment {
        SourceQualityAssessment {
            source_id: Uuid::new_v4(),
            quality_score: quality,
            risk_of_bias: RiskOfBias::Low,
            weight: 1.0,
            sample_size: Some(100),
            effect_size: Some(effect),
            standard_error: Some(se),
        }
    }

    #[test]
    fn test_meta_claim_creation() {
        let meta = MetaClaim::new(MetaClaimType::MetaAnalysis, "researcher@test.com".into());

        assert_eq!(meta.meta_type, MetaClaimType::MetaAnalysis);
        assert!(meta.source_claims.is_empty());
        assert!(!meta.meets_minimum_sources());
    }

    #[test]
    fn test_fixed_effects_synthesis() {
        let synthesizer = MetaSynthesizer::new();

        let assessments = vec![
            create_test_assessment(0.5, 0.1, 0.8),
            create_test_assessment(0.6, 0.15, 0.9),
            create_test_assessment(0.4, 0.12, 0.7),
        ];

        let result = synthesizer.fixed_effects_analysis(&assessments);

        assert!(result.pooled_effect > 0.0);
        assert!(result.ci_lower < result.pooled_effect);
        assert!(result.ci_upper > result.pooled_effect);
        assert_eq!(result.method, SynthesisMethod::FixedEffects);
    }

    #[test]
    fn test_heterogeneity_calculation() {
        let synthesizer = MetaSynthesizer::new();

        // Similar effects = low heterogeneity
        let similar = vec![
            create_test_assessment(0.5, 0.1, 0.8),
            create_test_assessment(0.52, 0.1, 0.8),
            create_test_assessment(0.48, 0.1, 0.8),
        ];

        let het = synthesizer.calculate_heterogeneity(&similar, 0.5);
        assert!(het.i_squared < 50.0);
    }

    #[test]
    fn test_grade_assessment() {
        let synthesizer = MetaSynthesizer::new();
        let mut meta = MetaClaim::new(MetaClaimType::SystematicReview, "test@test.com".into());

        // Add high-quality sources
        for _ in 0..5 {
            meta.add_source(
                Uuid::new_v4(),
                SourceQualityAssessment {
                    source_id: Uuid::new_v4(),
                    quality_score: 0.9,
                    risk_of_bias: RiskOfBias::Low,
                    weight: 1.0,
                    sample_size: Some(200),
                    effect_size: Some(0.5),
                    standard_error: Some(0.1),
                },
            );
        }

        meta.synthesis.ci_lower = 0.3;
        meta.synthesis.ci_upper = 0.7;
        meta.heterogeneity.classification = HeterogeneityLevel::Low;

        let grade = synthesizer.assess_grade(&meta);
        assert!(grade == GradeQuality::High || grade == GradeQuality::Moderate);
    }

    #[test]
    fn test_full_synthesis() {
        let synthesizer = MetaSynthesizer::new();
        let mut meta = MetaClaim::new(MetaClaimType::MetaAnalysis, "test@test.com".into());

        for i in 0..5 {
            let effect = 0.4 + (i as f64 * 0.05);
            meta.add_source(Uuid::new_v4(), create_test_assessment(effect, 0.1, 0.8));
        }

        let result = synthesizer.synthesize(meta);

        assert!(result.meets_minimum_sources());
        assert!(result.confidence > 0.0);
        assert!(!result.synthesis.conclusion.is_empty());
    }
}
