// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # 3D Uncertainty Quantification
//!
//! Implements the three-dimensional uncertainty model:
//!
//! 1. **Epistemic Uncertainty**: What we don't know but could
//! 2. **Aleatoric Uncertainty**: Inherent randomness/noise
//! 3. **Structural Uncertainty**: Model/framework limitations
//!
//! ## Visualization
//!
//! ```text
//!           Epistemic
//!              │
//!              │    ●  High epistemic,
//!              │       medium aleatoric
//!              │
//!              └─────────── Aleatoric
//!             /
//!            /
//!           /
//!      Structural
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::mycelix::gis::Uncertainty3D;
//!
//! let uncertainty = Uncertainty3D::new(0.3, 0.1, 0.2);
//! println!("Total uncertainty: {:.2}", uncertainty.total());
//! println!("Confidence: {:.2}", uncertainty.confidence());
//! ```

/// 3D Uncertainty quantification
///
/// Each dimension is in range [0.0, 1.0] where:
/// - 0.0 = No uncertainty (complete certainty)
/// - 1.0 = Maximum uncertainty (complete ignorance)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uncertainty3D {
    /// Epistemic uncertainty: What we don't know but could learn
    ///
    /// This is reducible through gathering more information.
    /// Examples:
    /// - Not knowing the answer to a factual question
    /// - Missing data
    /// - Incomplete research
    pub epistemic: f32,

    /// Aleatoric uncertainty: Inherent randomness/noise
    ///
    /// This is NOT reducible - it's inherent to the phenomenon.
    /// Examples:
    /// - Quantum measurement outcomes
    /// - Chaotic systems
    /// - Human behavior unpredictability
    pub aleatoric: f32,

    /// Structural uncertainty: Model/framework limitations
    ///
    /// This reflects limitations in our conceptual framework.
    /// Examples:
    /// - Using Newtonian physics where relativity matters
    /// - Linear models for nonlinear phenomena
    /// - Cultural biases in interpretation
    pub structural: f32,
}

impl Uncertainty3D {
    /// Create a new uncertainty with explicit values
    pub fn new(epistemic: f32, aleatoric: f32, structural: f32) -> Self {
        Self {
            epistemic: epistemic.clamp(0.0, 1.0),
            aleatoric: aleatoric.clamp(0.0, 1.0),
            structural: structural.clamp(0.0, 1.0),
        }
    }

    /// Create with no uncertainty (complete certainty)
    pub fn certain() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Create with maximum uncertainty
    pub fn maximum() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Create from a single confidence value (inverse mapping)
    pub fn from_confidence(confidence: f32) -> Self {
        let uncertainty = 1.0 - confidence.clamp(0.0, 1.0);
        // Distribute uncertainty across dimensions (heuristic)
        Self::new(
            uncertainty * 0.5, // Half is epistemic
            uncertainty * 0.3, // 30% is aleatoric
            uncertainty * 0.2, // 20% is structural
        )
    }

    /// Calculate total uncertainty (L2 norm, scaled)
    pub fn total(&self) -> f32 {
        let sum_sq = self.epistemic.powi(2) + self.aleatoric.powi(2) + self.structural.powi(2);
        // Normalize so maximum is 1.0 (when all are 1.0, sqrt(3) ≈ 1.732)
        (sum_sq.sqrt() / 3.0_f32.sqrt()).clamp(0.0, 1.0)
    }

    /// Calculate confidence (inverse of total uncertainty)
    pub fn confidence(&self) -> f32 {
        1.0 - self.total()
    }

    /// Get the dominant uncertainty type
    pub fn dominant_type(&self) -> UncertaintyType {
        if self.epistemic >= self.aleatoric && self.epistemic >= self.structural {
            UncertaintyType::Epistemic
        } else if self.aleatoric >= self.structural {
            UncertaintyType::Aleatoric
        } else {
            UncertaintyType::Structural
        }
    }

    /// Can this uncertainty be reduced?
    pub fn is_reducible(&self) -> bool {
        // Epistemic and structural can be reduced, aleatoric cannot
        self.epistemic > 0.1 || self.structural > 0.1
    }

    /// What fraction of uncertainty is reducible?
    pub fn reducible_fraction(&self) -> f32 {
        let total = self.epistemic + self.aleatoric + self.structural;
        if total < 0.001 {
            0.0
        } else {
            (self.epistemic + self.structural) / total
        }
    }

    /// Combine with another uncertainty (takes maximum of each dimension)
    pub fn combine_max(&self, other: &Self) -> Self {
        Self::new(
            self.epistemic.max(other.epistemic),
            self.aleatoric.max(other.aleatoric),
            self.structural.max(other.structural),
        )
    }

    /// Average with another uncertainty
    pub fn combine_avg(&self, other: &Self) -> Self {
        Self::new(
            (self.epistemic + other.epistemic) / 2.0,
            (self.aleatoric + other.aleatoric) / 2.0,
            (self.structural + other.structural) / 2.0,
        )
    }

    /// Apply a reduction factor (e.g., from gathering evidence)
    pub fn reduce_epistemic(&self, factor: f32) -> Self {
        Self::new(
            self.epistemic * (1.0 - factor.clamp(0.0, 1.0)),
            self.aleatoric,
            self.structural,
        )
    }

    /// Apply model improvement (reduces structural uncertainty)
    pub fn improve_model(&self, factor: f32) -> Self {
        Self::new(
            self.epistemic,
            self.aleatoric,
            self.structural * (1.0 - factor.clamp(0.0, 1.0)),
        )
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Uncertainty: {:.1}% (epistemic: {:.1}%, aleatoric: {:.1}%, structural: {:.1}%)",
            self.total() * 100.0,
            self.epistemic * 100.0,
            self.aleatoric * 100.0,
            self.structural * 100.0,
        )
    }

    /// Generate recommendations for reducing uncertainty
    pub fn recommendations(&self) -> Vec<UncertaintyRecommendation> {
        let mut recs = Vec::new();

        if self.epistemic > 0.3 {
            recs.push(UncertaintyRecommendation {
                uncertainty_type: UncertaintyType::Epistemic,
                action: "Gather more data or research the topic".to_string(),
                potential_reduction: self.epistemic * 0.7,
            });
        }

        if self.structural > 0.3 {
            recs.push(UncertaintyRecommendation {
                uncertainty_type: UncertaintyType::Structural,
                action: "Consider alternative frameworks or models".to_string(),
                potential_reduction: self.structural * 0.5,
            });
        }

        if self.aleatoric > 0.3 {
            recs.push(UncertaintyRecommendation {
                uncertainty_type: UncertaintyType::Aleatoric,
                action: "Accept inherent variability; use probabilistic reasoning".to_string(),
                potential_reduction: 0.0, // Cannot reduce aleatoric
            });
        }

        recs
    }
}

impl Default for Uncertainty3D {
    fn default() -> Self {
        // Default to moderate uncertainty
        Self::new(0.3, 0.2, 0.2)
    }
}

/// Type of uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UncertaintyType {
    Epistemic,
    Aleatoric,
    Structural,
}

impl UncertaintyType {
    /// Can this type of uncertainty be reduced?
    pub fn is_reducible(&self) -> bool {
        match self {
            Self::Epistemic => true,
            Self::Aleatoric => false,
            Self::Structural => true,
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Epistemic => "What we don't know but could learn",
            Self::Aleatoric => "Inherent randomness that cannot be reduced",
            Self::Structural => "Limitations in our conceptual framework",
        }
    }
}

/// Recommendation for reducing uncertainty
#[derive(Debug, Clone)]
pub struct UncertaintyRecommendation {
    pub uncertainty_type: UncertaintyType,
    pub action: String,
    pub potential_reduction: f32,
}

// =============================================================================
// GIS v4.0: Moral Uncertainty (Tripartite Model)
// =============================================================================

/// Moral Uncertainty: Three orthogonal dimensions of moral doubt
///
/// Based on MacAskill's moral uncertainty research, these three dimensions
/// are orthogonal. An agent can be:
/// - Epistemically certain, axiologically uncertain: Knows facts, unsure which values apply
/// - Axiologically certain, deontically uncertain: Knows values at stake, unsure what to do
/// - All three uncertain: Humble about everything (appropriate for novel situations)
///
/// ## Example
///
/// ```rust,ignore
/// use symthaea::mycelix::gis::MoralUncertainty;
///
/// let moral = MoralUncertainty::new(0.3, 0.7, 0.5);
/// // Low epistemic (knows the facts), high axiological (unsure which values),
/// // moderate deontic (somewhat unsure what to do)
///
/// println!("Dominant: {:?}", moral.dominant());
/// println!("Should pause? {}", moral.should_pause_for_reflection());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoralUncertainty {
    /// Epistemic moral uncertainty: Uncertainty about the facts of the moral situation
    ///
    /// "I'm unsure what will happen if I do X"
    /// "I'm unsure who will be affected"
    /// "I'm unsure about the causal chain"
    pub epistemic: f32,

    /// Axiological moral uncertainty: Uncertainty about which values/goods are at stake
    ///
    /// "I'm unsure which values apply here"
    /// "I'm unsure how to weigh competing goods"
    /// "I'm unsure if this is even a moral question"
    pub axiological: f32,

    /// Deontic moral uncertainty: Uncertainty about what action is right
    ///
    /// "I'm unsure what I should do"
    /// "I'm unsure which framework to apply"
    /// "I'm unsure about my moral obligations here"
    pub deontic: f32,
}

impl MoralUncertainty {
    /// Create a new moral uncertainty with explicit values
    pub fn new(epistemic: f32, axiological: f32, deontic: f32) -> Self {
        Self {
            epistemic: epistemic.clamp(0.0, 1.0),
            axiological: axiological.clamp(0.0, 1.0),
            deontic: deontic.clamp(0.0, 1.0),
        }
    }

    /// Create with minimal moral uncertainty (morally clear)
    pub fn clear() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Create with maximum moral uncertainty (morally opaque)
    pub fn opaque() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Create from a situation assessment
    pub fn assess(
        facts_known: f32,  // How well we understand the facts (0 = unknown, 1 = known)
        values_clear: f32, // How clear the relevant values are (0 = unclear, 1 = clear)
        action_clear: f32, // How clear the right action is (0 = unclear, 1 = clear)
    ) -> Self {
        Self::new(
            1.0 - facts_known.clamp(0.0, 1.0),
            1.0 - values_clear.clamp(0.0, 1.0),
            1.0 - action_clear.clamp(0.0, 1.0),
        )
    }

    /// Calculate total moral uncertainty
    pub fn total(&self) -> f32 {
        let sum_sq = self.epistemic.powi(2) + self.axiological.powi(2) + self.deontic.powi(2);
        (sum_sq.sqrt() / 3.0_f32.sqrt()).clamp(0.0, 1.0)
    }

    /// Calculate moral confidence (inverse of total)
    pub fn confidence(&self) -> f32 {
        1.0 - self.total()
    }

    /// Average of all three dimensions (simpler metric)
    pub fn average(&self) -> f32 {
        (self.epistemic + self.axiological + self.deontic) / 3.0
    }

    /// Get the dominant type of moral uncertainty
    pub fn dominant(&self) -> MoralUncertaintyType {
        if self.epistemic >= self.axiological && self.epistemic >= self.deontic {
            MoralUncertaintyType::Epistemic
        } else if self.axiological >= self.deontic {
            MoralUncertaintyType::Axiological
        } else {
            MoralUncertaintyType::Deontic
        }
    }

    /// Should we pause for reflection before acting?
    ///
    /// Returns true if:
    /// - Any dimension exceeds 0.7, OR
    /// - Total exceeds 0.5, OR
    /// - Axiological uncertainty is high (values unclear)
    pub fn should_pause_for_reflection(&self) -> bool {
        self.epistemic > 0.7
            || self.axiological > 0.7
            || self.deontic > 0.7
            || self.total() > 0.5
            || self.axiological > 0.5 // Values uncertainty warrants extra caution
    }

    /// Should we seek input from others?
    ///
    /// Returns true if axiological or deontic uncertainty is high.
    pub fn should_seek_input(&self) -> bool {
        self.axiological > 0.5 || self.deontic > 0.5
    }

    /// Should we defer to a known authority/framework?
    ///
    /// Returns true if deontic uncertainty is very high but axiological is low.
    /// (We know the values but don't know how to act.)
    pub fn should_defer_to_authority(&self) -> bool {
        self.deontic > 0.7 && self.axiological < 0.4
    }

    /// Generate recommendations based on moral uncertainty profile
    pub fn recommendations(&self) -> Vec<MoralRecommendation> {
        let mut recs = Vec::new();

        if self.epistemic > 0.4 {
            recs.push(MoralRecommendation {
                dimension: MoralUncertaintyType::Epistemic,
                action: "Gather more information about consequences and affected parties"
                    .to_string(),
                priority: (self.epistemic * 10.0).clamp(0.0, 255.0) as u8,
            });
        }

        if self.axiological > 0.4 {
            recs.push(MoralRecommendation {
                dimension: MoralUncertaintyType::Axiological,
                action: "Identify which values and principles are at stake".to_string(),
                priority: (self.axiological * 12.0).clamp(0.0, 255.0) as u8, // Higher priority for value uncertainty
            });
        }

        if self.deontic > 0.4 {
            recs.push(MoralRecommendation {
                dimension: MoralUncertaintyType::Deontic,
                action: "Consider multiple ethical frameworks and their recommendations"
                    .to_string(),
                priority: (self.deontic * 10.0).clamp(0.0, 255.0) as u8,
            });
        }

        // Sort by priority (highest first)
        recs.sort_by(|a, b| b.priority.cmp(&a.priority));
        recs
    }

    /// Combine with another moral uncertainty (takes maximum of each)
    pub fn combine_max(&self, other: &Self) -> Self {
        Self::new(
            self.epistemic.max(other.epistemic),
            self.axiological.max(other.axiological),
            self.deontic.max(other.deontic),
        )
    }

    /// Reduce epistemic uncertainty (through research/investigation)
    pub fn reduce_epistemic(&self, factor: f32) -> Self {
        Self::new(
            self.epistemic * (1.0 - factor.clamp(0.0, 1.0)),
            self.axiological,
            self.deontic,
        )
    }

    /// Reduce axiological uncertainty (through value clarification)
    pub fn reduce_axiological(&self, factor: f32) -> Self {
        Self::new(
            self.epistemic,
            self.axiological * (1.0 - factor.clamp(0.0, 1.0)),
            self.deontic,
        )
    }

    /// Reduce deontic uncertainty (through ethical reasoning)
    pub fn reduce_deontic(&self, factor: f32) -> Self {
        Self::new(
            self.epistemic,
            self.axiological,
            self.deontic * (1.0 - factor.clamp(0.0, 1.0)),
        )
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Moral Uncertainty: {:.0}% (epistemic: {:.0}%, axiological: {:.0}%, deontic: {:.0}%)",
            self.total() * 100.0,
            self.epistemic * 100.0,
            self.axiological * 100.0,
            self.deontic * 100.0,
        )
    }

    /// Generate a simple action guidance based on profile
    pub fn action_guidance(&self) -> MoralActionGuidance {
        let total = self.total();

        if total < 0.2 {
            MoralActionGuidance::ProceedConfidently
        } else if total < 0.4 {
            MoralActionGuidance::ProceedWithCaution
        } else if total < 0.6 {
            MoralActionGuidance::SeekGuidance
        } else if total < 0.8 {
            MoralActionGuidance::PauseAndReflect
        } else {
            MoralActionGuidance::AvoidAction
        }
    }
}

impl Default for MoralUncertainty {
    fn default() -> Self {
        // Default to moderate uncertainty (humble by default)
        Self::new(0.3, 0.3, 0.3)
    }
}

/// Type of moral uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoralUncertaintyType {
    /// Uncertainty about facts and consequences
    Epistemic,
    /// Uncertainty about values and goods
    Axiological,
    /// Uncertainty about right action
    Deontic,
}

impl MoralUncertaintyType {
    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Epistemic => "Uncertainty about the facts and consequences",
            Self::Axiological => "Uncertainty about which values are at stake",
            Self::Deontic => "Uncertainty about what action is right",
        }
    }

    /// Suggested approach for this type of uncertainty
    pub fn approach(&self) -> &'static str {
        match self {
            Self::Epistemic => "Gather more information, investigate consequences",
            Self::Axiological => "Clarify values, consult stakeholders, reflect on principles",
            Self::Deontic => "Apply multiple ethical frameworks, seek wise counsel",
        }
    }
}

/// Recommendation for addressing moral uncertainty
#[derive(Debug, Clone)]
pub struct MoralRecommendation {
    pub dimension: MoralUncertaintyType,
    pub action: String,
    pub priority: u8,
}

/// Action guidance based on moral uncertainty level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoralActionGuidance {
    /// Low uncertainty - proceed with confidence
    ProceedConfidently,
    /// Mild uncertainty - proceed but monitor
    ProceedWithCaution,
    /// Moderate uncertainty - seek guidance before acting
    SeekGuidance,
    /// High uncertainty - pause and reflect deeply
    PauseAndReflect,
    /// Very high uncertainty - avoid action until resolved
    AvoidAction,
}

impl MoralActionGuidance {
    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::ProceedConfidently => "The moral path is clear - proceed with confidence",
            Self::ProceedWithCaution => "Proceed, but remain attentive to moral dimensions",
            Self::SeekGuidance => "Seek input from others before acting",
            Self::PauseAndReflect => "Pause for deep reflection - consult wise counsel",
            Self::AvoidAction => "Avoid action until moral clarity emerges",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_creation() {
        let u = Uncertainty3D::new(0.5, 0.3, 0.2);
        assert!((u.epistemic - 0.5).abs() < 0.001);
        assert!((u.aleatoric - 0.3).abs() < 0.001);
        assert!((u.structural - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_clamping() {
        let u = Uncertainty3D::new(1.5, -0.5, 0.5);
        assert!((u.epistemic - 1.0).abs() < 0.001);
        assert!((u.aleatoric - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_certain() {
        let u = Uncertainty3D::certain();
        assert!((u.total() - 0.0).abs() < 0.001);
        assert!((u.confidence() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_maximum() {
        let u = Uncertainty3D::maximum();
        assert!((u.total() - 1.0).abs() < 0.001);
        assert!((u.confidence() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_dominant_type() {
        let u1 = Uncertainty3D::new(0.8, 0.2, 0.3);
        assert_eq!(u1.dominant_type(), UncertaintyType::Epistemic);

        let u2 = Uncertainty3D::new(0.2, 0.8, 0.3);
        assert_eq!(u2.dominant_type(), UncertaintyType::Aleatoric);

        let u3 = Uncertainty3D::new(0.2, 0.3, 0.8);
        assert_eq!(u3.dominant_type(), UncertaintyType::Structural);
    }

    #[test]
    fn test_reducibility() {
        let u1 = Uncertainty3D::new(0.5, 0.8, 0.1);
        assert!(u1.is_reducible()); // High epistemic

        let u2 = Uncertainty3D::new(0.05, 0.8, 0.05);
        assert!(!u2.is_reducible()); // Only aleatoric is high
    }

    #[test]
    fn test_reduce_epistemic() {
        let u = Uncertainty3D::new(0.8, 0.3, 0.2);
        let reduced = u.reduce_epistemic(0.5);
        assert!((reduced.epistemic - 0.4).abs() < 0.001);
        assert!((reduced.aleatoric - 0.3).abs() < 0.001); // Unchanged
    }

    #[test]
    fn test_recommendations() {
        let u = Uncertainty3D::new(0.8, 0.1, 0.5);
        let recs = u.recommendations();

        assert!(
            recs.iter()
                .any(|r| r.uncertainty_type == UncertaintyType::Epistemic)
        );
        assert!(
            recs.iter()
                .any(|r| r.uncertainty_type == UncertaintyType::Structural)
        );
    }

    // === GIS v4.0 Moral Uncertainty Tests ===

    #[test]
    fn test_moral_uncertainty_creation() {
        let m = MoralUncertainty::new(0.3, 0.7, 0.5);
        assert!((m.epistemic - 0.3).abs() < 0.001);
        assert!((m.axiological - 0.7).abs() < 0.001);
        assert!((m.deontic - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_moral_clear_and_opaque() {
        let clear = MoralUncertainty::clear();
        assert!((clear.total() - 0.0).abs() < 0.001);

        let opaque = MoralUncertainty::opaque();
        assert!((opaque.total() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_moral_dominant() {
        let m1 = MoralUncertainty::new(0.8, 0.2, 0.3);
        assert_eq!(m1.dominant(), MoralUncertaintyType::Epistemic);

        let m2 = MoralUncertainty::new(0.2, 0.8, 0.3);
        assert_eq!(m2.dominant(), MoralUncertaintyType::Axiological);

        let m3 = MoralUncertainty::new(0.2, 0.3, 0.8);
        assert_eq!(m3.dominant(), MoralUncertaintyType::Deontic);
    }

    #[test]
    fn test_moral_should_pause() {
        // High axiological uncertainty should trigger pause
        let m = MoralUncertainty::new(0.3, 0.8, 0.3);
        assert!(m.should_pause_for_reflection());

        // Low overall uncertainty should not trigger pause
        let m2 = MoralUncertainty::new(0.2, 0.2, 0.2);
        assert!(!m2.should_pause_for_reflection());
    }

    #[test]
    fn test_moral_action_guidance() {
        let clear = MoralUncertainty::new(0.1, 0.1, 0.1);
        assert_eq!(
            clear.action_guidance(),
            MoralActionGuidance::ProceedConfidently
        );

        let opaque = MoralUncertainty::new(0.9, 0.9, 0.9);
        assert_eq!(opaque.action_guidance(), MoralActionGuidance::AvoidAction);
    }

    #[test]
    fn test_moral_recommendations() {
        let m = MoralUncertainty::new(0.2, 0.8, 0.6);
        let recs = m.recommendations();

        // Should have recommendations for axiological and deontic (both > 0.4)
        assert!(
            recs.iter()
                .any(|r| r.dimension == MoralUncertaintyType::Axiological)
        );
        assert!(
            recs.iter()
                .any(|r| r.dimension == MoralUncertaintyType::Deontic)
        );

        // Axiological should be higher priority
        let axio_rec = recs
            .iter()
            .find(|r| r.dimension == MoralUncertaintyType::Axiological);
        let deon_rec = recs
            .iter()
            .find(|r| r.dimension == MoralUncertaintyType::Deontic);
        assert!(axio_rec.unwrap().priority >= deon_rec.unwrap().priority);
    }

    #[test]
    fn test_moral_reduction() {
        let m = MoralUncertainty::new(0.8, 0.8, 0.8);

        let reduced = m.reduce_epistemic(0.5);
        assert!((reduced.epistemic - 0.4).abs() < 0.001);
        assert!((reduced.axiological - 0.8).abs() < 0.001); // Unchanged

        let clarified = m.reduce_axiological(0.5);
        assert!((clarified.axiological - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_moral_assess() {
        // Facts well known, values unclear, action unclear
        let m = MoralUncertainty::assess(0.9, 0.2, 0.3);
        assert!(m.epistemic < 0.2);
        assert!(m.axiological > 0.7);
        assert!(m.deontic > 0.6);
    }
}
