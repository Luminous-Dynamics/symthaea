//! Moral Uncertainty Types
//!
//! Implements the tripartite model of moral uncertainty from GIS v4.0:
//! - Epistemic: Uncertainty about the facts
//! - Axiological: Uncertainty about which values apply
//! - Deontic: Uncertainty about what action is right

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Tripartite moral uncertainty model
///
/// Based on MacAskill's moral uncertainty research, these three dimensions
/// are orthogonal - an agent can be certain in one dimension while uncertain
/// in others.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MoralUncertainty {
    /// Epistemic uncertainty - "I'm unsure what will happen"
    /// Uncertainty about the facts of the moral situation.
    /// Range: 0.0 (certain) to 1.0 (completely uncertain)
    pub epistemic: f32,

    /// Axiological uncertainty - "I'm unsure which values apply"
    /// Uncertainty about which values/goods are at stake.
    /// Range: 0.0 (certain) to 1.0 (completely uncertain)
    pub axiological: f32,

    /// Deontic uncertainty - "I'm unsure what I should do"
    /// Uncertainty about what action is right.
    /// Range: 0.0 (certain) to 1.0 (completely uncertain)
    pub deontic: f32,
}

impl MoralUncertainty {
    /// Create with all dimensions fully certain
    pub fn certain() -> Self {
        Self {
            epistemic: 0.0,
            axiological: 0.0,
            deontic: 0.0,
        }
    }

    /// Create with all dimensions maximally uncertain
    pub fn maximally_uncertain() -> Self {
        Self {
            epistemic: 1.0,
            axiological: 1.0,
            deontic: 1.0,
        }
    }

    /// Create with specified values
    pub fn new(epistemic: f32, axiological: f32, deontic: f32) -> Self {
        Self {
            epistemic,
            axiological,
            deontic,
        }
    }

    /// Calculate average uncertainty across all dimensions
    pub fn average(&self) -> f32 {
        (self.epistemic + self.axiological + self.deontic) / 3.0
    }

    /// Calculate maximum uncertainty (most uncertain dimension)
    pub fn max(&self) -> f32 {
        self.epistemic.max(self.axiological).max(self.deontic)
    }

    /// Calculate minimum uncertainty (most certain dimension)
    pub fn min(&self) -> f32 {
        self.epistemic.min(self.axiological).min(self.deontic)
    }

    /// Check if epistemically certain but axiologically uncertain
    ///
    /// This is a common case: knowing the facts but unsure which values matter.
    pub fn facts_certain_values_uncertain(&self) -> bool {
        self.epistemic < 0.3 && self.axiological > 0.5
    }

    /// Check if values certain but action uncertain
    ///
    /// Another common case: knowing what matters but unsure how to act.
    pub fn values_certain_action_uncertain(&self) -> bool {
        self.axiological < 0.3 && self.deontic > 0.5
    }

    /// Check if this represents a novel situation (high across all dimensions)
    pub fn is_novel_situation(&self) -> bool {
        self.epistemic > 0.5 && self.axiological > 0.5 && self.deontic > 0.5
    }

    /// Check if action is warranted given uncertainty levels
    ///
    /// Conservative approach: require low deontic uncertainty to act.
    pub fn action_warranted(&self, threshold: f32) -> bool {
        self.deontic < threshold
    }

    /// Validate all values are in [0.0, 1.0]
    pub fn is_valid(&self) -> bool {
        let in_range = |v: f32| (0.0..=1.0).contains(&v) && v.is_finite();
        in_range(self.epistemic) && in_range(self.axiological) && in_range(self.deontic)
    }

    /// Clamp all values to [0.0, 1.0]
    pub fn clamp(&self) -> Self {
        Self {
            epistemic: self.epistemic.clamp(0.0, 1.0),
            axiological: self.axiological.clamp(0.0, 1.0),
            deontic: self.deontic.clamp(0.0, 1.0),
        }
    }

    /// Get the most uncertain dimension
    pub fn most_uncertain_dimension(&self) -> UncertaintyDimension {
        if self.epistemic >= self.axiological && self.epistemic >= self.deontic {
            UncertaintyDimension::Epistemic
        } else if self.axiological >= self.deontic {
            UncertaintyDimension::Axiological
        } else {
            UncertaintyDimension::Deontic
        }
    }

    /// Get the most certain dimension
    pub fn most_certain_dimension(&self) -> UncertaintyDimension {
        if self.epistemic <= self.axiological && self.epistemic <= self.deontic {
            UncertaintyDimension::Epistemic
        } else if self.axiological <= self.deontic {
            UncertaintyDimension::Axiological
        } else {
            UncertaintyDimension::Deontic
        }
    }

    /// Blend two uncertainty states (e.g., combining perspectives)
    pub fn blend(&self, other: &MoralUncertainty, weight: f32) -> Self {
        Self {
            epistemic: self.epistemic * (1.0 - weight) + other.epistemic * weight,
            axiological: self.axiological * (1.0 - weight) + other.axiological * weight,
            deontic: self.deontic * (1.0 - weight) + other.deontic * weight,
        }
    }

    /// Format as short string (e.g., "E:0.3,A:0.5,D:0.2")
    pub fn short_format(&self) -> String {
        format!(
            "E:{:.2},A:{:.2},D:{:.2}",
            self.epistemic, self.axiological, self.deontic
        )
    }
}

impl Default for MoralUncertainty {
    fn default() -> Self {
        // Default to moderate uncertainty in all dimensions
        Self {
            epistemic: 0.5,
            axiological: 0.5,
            deontic: 0.5,
        }
    }
}

/// The three dimensions of moral uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UncertaintyDimension {
    /// About the facts
    Epistemic,
    /// About the values
    Axiological,
    /// About the action
    Deontic,
}

impl UncertaintyDimension {
    /// Get the guiding question for this dimension
    pub fn guiding_question(&self) -> &'static str {
        match self {
            Self::Epistemic => "What will happen?",
            Self::Axiological => "Which values apply?",
            Self::Deontic => "What should I do?",
        }
    }

    /// Get a short description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Epistemic => "Uncertainty about facts and consequences",
            Self::Axiological => "Uncertainty about applicable values",
            Self::Deontic => "Uncertainty about right action",
        }
    }
}

/// Builder for MoralUncertainty
#[derive(Debug, Default)]
pub struct MoralUncertaintyBuilder {
    epistemic: Option<f32>,
    axiological: Option<f32>,
    deontic: Option<f32>,
}

impl MoralUncertaintyBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn epistemic(mut self, value: f32) -> Self {
        self.epistemic = Some(value);
        self
    }

    pub fn axiological(mut self, value: f32) -> Self {
        self.axiological = Some(value);
        self
    }

    pub fn deontic(mut self, value: f32) -> Self {
        self.deontic = Some(value);
        self
    }

    /// Build with defaults of 0.5 for unset values
    pub fn build(self) -> MoralUncertainty {
        MoralUncertainty {
            epistemic: self.epistemic.unwrap_or(0.5),
            axiological: self.axiological.unwrap_or(0.5),
            deontic: self.deontic.unwrap_or(0.5),
        }
    }
}

/// Moral uncertainty assessment for a specific action
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ActionAssessment {
    /// The action being assessed
    pub action_description: String,
    /// Uncertainty levels for this action
    pub uncertainty: MoralUncertainty,
    /// Confidence in the assessment itself
    pub meta_confidence: f32,
    /// Recommended approach given uncertainty
    pub recommendation: ActionRecommendation,
}

/// Recommended approach given moral uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ActionRecommendation {
    /// Proceed with action
    Proceed,
    /// Proceed with caution
    ProceedWithCaution,
    /// Seek more information first
    GatherInformation,
    /// Consult others before acting
    ConsultOthers,
    /// Defer decision
    Defer,
    /// Do not proceed
    Abstain,
}

impl ActionRecommendation {
    /// Get recommendation based on uncertainty levels
    pub fn from_uncertainty(uncertainty: &MoralUncertainty) -> Self {
        let avg = uncertainty.average();
        let max = uncertainty.max();

        if avg < 0.2 && max < 0.3 {
            Self::Proceed
        } else if avg < 0.4 && max < 0.5 {
            Self::ProceedWithCaution
        } else if uncertainty.epistemic > 0.6 {
            Self::GatherInformation
        } else if uncertainty.axiological > 0.6 {
            Self::ConsultOthers
        } else if uncertainty.deontic > 0.7 {
            Self::Defer
        } else if max > 0.8 {
            Self::Abstain
        } else {
            Self::ConsultOthers
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certain_vs_uncertain() {
        let certain = MoralUncertainty::certain();
        assert!((certain.average() - 0.0).abs() < 0.001);

        let uncertain = MoralUncertainty::maximally_uncertain();
        assert!((uncertain.average() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_facts_certain_values_uncertain() {
        let mu = MoralUncertainty::new(0.1, 0.8, 0.5);
        assert!(mu.facts_certain_values_uncertain());

        let not_mu = MoralUncertainty::new(0.5, 0.3, 0.5);
        assert!(!not_mu.facts_certain_values_uncertain());
    }

    #[test]
    fn test_novel_situation() {
        let novel = MoralUncertainty::new(0.7, 0.8, 0.6);
        assert!(novel.is_novel_situation());

        let familiar = MoralUncertainty::new(0.2, 0.3, 0.2);
        assert!(!familiar.is_novel_situation());
    }

    #[test]
    fn test_recommendation() {
        let low = MoralUncertainty::new(0.1, 0.1, 0.1);
        assert_eq!(
            ActionRecommendation::from_uncertainty(&low),
            ActionRecommendation::Proceed
        );

        let high_epistemic = MoralUncertainty::new(0.8, 0.3, 0.3);
        assert_eq!(
            ActionRecommendation::from_uncertainty(&high_epistemic),
            ActionRecommendation::GatherInformation
        );
    }

    #[test]
    fn test_builder() {
        let mu = MoralUncertaintyBuilder::new()
            .epistemic(0.3)
            .deontic(0.7)
            .build();

        assert!((mu.epistemic - 0.3).abs() < 0.001);
        assert!((mu.axiological - 0.5).abs() < 0.001); // Default
        assert!((mu.deontic - 0.7).abs() < 0.001);
    }
}
