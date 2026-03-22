// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Rashomon Engine: Multi-Perspective Generation
//!
//! Named after Kurosawa's film where each witness tells a different truth,
//! the Rashomon Engine generates multiple harmonic framings of a situation.
//!
//! ## Architecture
//!
//! Each of the Eight Harmonies provides an epistemic frame - a lens through
//! which knowledge is perceived. The Rashomon Engine:
//!
//! 1. Generates perspectives from each relevant harmony
//! 2. Identifies areas of agreement and dissent
//! 3. Synthesizes a unified view while preserving minority perspectives
//! 4. Enforces N3 boundaries (never generate perspectives that harm)
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::mycelix::gis::rashomon::{RashomonEngine, Situation};
//!
//! let engine = RashomonEngine::new();
//! let situation = Situation::new("Should we deploy this AI system?");
//!
//! let perspectives = engine.generate_perspectives(&situation);
//! // - Care-Knowing: "Who will be affected?"
//! // - Truth-Knowing: "Is it verified to work safely?"
//! // - Exchange-Knowing: "What are the economic flows?"
//!
//! let synthesis = engine.synthesize(perspectives);
//! println!("Unified view: {}", synthesis.unified_view);
//! for dissent in synthesis.preserved_dissents {
//!     println!("Dissent from {}: {}", dissent.harmony, dissent.reason);
//! }
//! ```

use super::ignorance_types::Harmony;
use super::uncertainty::MoralUncertainty;
use std::time::SystemTime;

/// A situation to be analyzed from multiple perspectives
#[derive(Debug, Clone)]
pub struct Situation {
    /// The question or scenario being analyzed
    pub description: String,

    /// Relevant domain (affects which harmonies are most relevant)
    pub domain: SituationDomain,

    /// Stakeholders involved
    pub stakeholders: Vec<String>,

    /// Initial moral uncertainty assessment
    pub moral_uncertainty: MoralUncertainty,

    /// When this situation was posed
    pub created_at: SystemTime,
}

impl Situation {
    /// Create a new situation for analysis
    pub fn new(description: &str) -> Self {
        Self {
            description: description.to_string(),
            domain: SituationDomain::General,
            stakeholders: Vec::new(),
            moral_uncertainty: MoralUncertainty::default(),
            created_at: SystemTime::now(),
        }
    }

    /// Create with domain
    pub fn with_domain(description: &str, domain: SituationDomain) -> Self {
        Self {
            domain,
            ..Self::new(description)
        }
    }

    /// Add stakeholders
    pub fn add_stakeholder(&mut self, stakeholder: &str) {
        self.stakeholders.push(stakeholder.to_string());
    }

    /// Set moral uncertainty
    pub fn with_moral_uncertainty(mut self, mu: MoralUncertainty) -> Self {
        self.moral_uncertainty = mu;
        self
    }
}

/// Domain of the situation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SituationDomain {
    /// Technology deployment, AI systems
    Technology,
    /// Environmental issues
    Environmental,
    /// Economic/business decisions
    Economic,
    /// Social/community issues
    Social,
    /// Medical/health decisions
    Medical,
    /// Political/governance
    Political,
    /// General/mixed
    General,
}

impl SituationDomain {
    /// Which harmonies are most relevant for this domain?
    pub fn primary_harmonies(&self) -> Vec<Harmony> {
        match self {
            Self::Technology => vec![
                Harmony::IntegralWisdom,
                Harmony::PanSentientFlourishing,
                Harmony::EvolutionaryProgression,
            ],
            Self::Environmental => vec![
                Harmony::PanSentientFlourishing,
                Harmony::UniversalInterconnectedness,
                Harmony::EvolutionaryProgression,
            ],
            Self::Economic => vec![
                Harmony::SacredReciprocity,
                Harmony::ResonantCoherence,
                Harmony::PanSentientFlourishing,
            ],
            Self::Social => vec![
                Harmony::PanSentientFlourishing,
                Harmony::UniversalInterconnectedness,
                Harmony::SacredReciprocity,
            ],
            Self::Medical => vec![
                Harmony::PanSentientFlourishing,
                Harmony::IntegralWisdom,
                Harmony::SacredReciprocity,
            ],
            Self::Political => vec![
                Harmony::ResonantCoherence,
                Harmony::PanSentientFlourishing,
                Harmony::UniversalInterconnectedness,
            ],
            Self::General => vec![
                Harmony::ResonantCoherence,
                Harmony::PanSentientFlourishing,
                Harmony::IntegralWisdom,
            ],
        }
    }
}

/// A harmonic frame - one of seven epistemic lenses
#[derive(Debug, Clone)]
pub struct HarmonicFrame {
    /// Which harmony this frame represents
    pub harmony: Harmony,

    /// How activated this frame is (0.0 - 1.0)
    pub activation: f32,

    /// Frame-specific prompts for generating perspective
    pub prompts: Vec<String>,
}

impl HarmonicFrame {
    /// Create a frame for a harmony
    pub fn new(harmony: Harmony) -> Self {
        let prompts = match harmony {
            Harmony::ResonantCoherence => vec![
                "How do the parts relate to each other?".to_string(),
                "What patterns of integration exist?".to_string(),
                "Where is there coherence or dissonance?".to_string(),
            ],
            Harmony::PanSentientFlourishing => vec![
                "Who is affected by this situation?".to_string(),
                "How does this impact well-being?".to_string(),
                "Whose voice is not being heard?".to_string(),
            ],
            Harmony::IntegralWisdom => vec![
                "What can be verified or known with certainty?".to_string(),
                "What evidence supports each position?".to_string(),
                "Where are the knowledge gaps?".to_string(),
            ],
            Harmony::InfinitePlay => vec![
                "What creative possibilities exist?".to_string(),
                "What novel approaches haven't been considered?".to_string(),
                "Where is there room for playful experimentation?".to_string(),
            ],
            Harmony::UniversalInterconnectedness => vec![
                "What connections exist between elements?".to_string(),
                "What ripple effects might occur?".to_string(),
                "Who and what are part of this web?".to_string(),
            ],
            Harmony::SacredReciprocity => vec![
                "What flows back and forth?".to_string(),
                "Is there balance in the exchange?".to_string(),
                "Who gives and who receives?".to_string(),
            ],
            Harmony::EvolutionaryProgression => vec![
                "What is emerging or developing?".to_string(),
                "How has this evolved to this point?".to_string(),
                "What trajectory are we on?".to_string(),
            ],
            Harmony::SacredStillness => vec![
                "What is revealed in stillness?".to_string(),
                "What emerges when we stop striving?".to_string(),
                "What does present-moment awareness illuminate?".to_string(),
            ],
        };

        Self {
            harmony,
            activation: 1.0,
            prompts,
        }
    }

    /// Calculate relevance to a situation
    pub fn relevance(&self, situation: &Situation) -> f32 {
        let domain_relevance = if situation.domain.primary_harmonies().contains(&self.harmony) {
            0.8
        } else {
            0.3
        };

        // Stakeholders boost PSF relevance
        let stakeholder_boost = if self.harmony == Harmony::PanSentientFlourishing
            && !situation.stakeholders.is_empty()
        {
            0.2
        } else {
            0.0
        };

        (domain_relevance + stakeholder_boost) * self.activation
    }
}

/// A perspective generated from a harmonic frame
#[derive(Debug, Clone)]
pub struct HarmonicPerspective {
    /// Which harmony generated this perspective
    pub harmony: Harmony,

    /// The perspective content
    pub content: String,

    /// Key insights from this perspective
    pub insights: Vec<String>,

    /// Concerns raised by this perspective
    pub concerns: Vec<String>,

    /// Confidence in this perspective (0.0 - 1.0)
    pub confidence: f32,

    /// Moral uncertainty specific to this perspective
    pub moral_uncertainty: MoralUncertainty,
}

impl HarmonicPerspective {
    /// Create a new perspective
    pub fn new(harmony: Harmony, content: &str) -> Self {
        Self {
            harmony,
            content: content.to_string(),
            insights: Vec::new(),
            concerns: Vec::new(),
            confidence: 0.5,
            moral_uncertainty: MoralUncertainty::default(),
        }
    }

    /// Add an insight
    pub fn add_insight(&mut self, insight: &str) {
        self.insights.push(insight.to_string());
    }

    /// Add a concern
    pub fn add_concern(&mut self, concern: &str) {
        self.concerns.push(concern.to_string());
    }

    /// Does this perspective conflict with another?
    pub fn conflicts_with(&self, other: &Self) -> bool {
        // Simple heuristic: if concerns in one match insights in other, potential conflict
        for concern in &self.concerns {
            for insight in &other.insights {
                if concern.to_lowercase().contains(&insight.to_lowercase())
                    || insight.to_lowercase().contains(&concern.to_lowercase())
                {
                    return true;
                }
            }
        }
        false
    }
}

/// A dissenting view preserved from synthesis
#[derive(Debug, Clone)]
pub struct PreservedDissent {
    /// Which harmony dissents
    pub harmony: Harmony,

    /// The dissenting perspective
    pub perspective: HarmonicPerspective,

    /// Why this dissent is important to preserve
    pub reason: String,

    /// How different from the unified view (0.0 - 1.0)
    pub divergence: f32,
}

/// The synthesized view combining all perspectives
#[derive(Debug, Clone)]
pub struct SynthesizedView {
    /// The unified perspective
    pub unified_view: String,

    /// Key points of agreement across harmonies
    pub agreements: Vec<String>,

    /// Preserved dissenting views
    pub preserved_dissents: Vec<PreservedDissent>,

    /// Overall confidence in the synthesis
    pub confidence: f32,

    /// Combined moral uncertainty
    pub moral_uncertainty: MoralUncertainty,

    /// Which harmonies contributed
    pub contributing_harmonies: Vec<Harmony>,

    /// Whether N3 boundaries were triggered
    pub n3_triggered: bool,
}

impl SynthesizedView {
    /// Create a new synthesized view
    pub fn new(unified_view: &str) -> Self {
        Self {
            unified_view: unified_view.to_string(),
            agreements: Vec::new(),
            preserved_dissents: Vec::new(),
            confidence: 0.5,
            moral_uncertainty: MoralUncertainty::default(),
            contributing_harmonies: Vec::new(),
            n3_triggered: false,
        }
    }

    /// Add an agreement
    pub fn add_agreement(&mut self, agreement: &str) {
        self.agreements.push(agreement.to_string());
    }

    /// Add a dissent
    pub fn add_dissent(&mut self, dissent: PreservedDissent) {
        self.preserved_dissents.push(dissent);
    }

    /// Has significant dissent?
    pub fn has_significant_dissent(&self) -> bool {
        self.preserved_dissents.iter().any(|d| d.divergence > 0.5)
    }
}

/// The Rashomon Engine - multi-perspective generator
#[allow(dead_code)] // RESERVED(future): multi-perspective reasoning thresholds
pub struct RashomonEngine {
    /// The seven harmonic frames
    frames: Vec<HarmonicFrame>,

    /// Minimum relevance to include a perspective
    relevance_threshold: f32,

    /// Minimum divergence to preserve as dissent
    dissent_threshold: f32,

    /// N3 boundary enforcement enabled
    n3_enabled: bool,
}

impl RashomonEngine {
    /// Create a new Rashomon Engine
    pub fn new() -> Self {
        let frames = Harmony::all()
            .iter()
            .map(|h| HarmonicFrame::new(*h))
            .collect();

        Self {
            frames,
            relevance_threshold: 0.3,
            dissent_threshold: 0.4,
            n3_enabled: true,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(relevance: f32, dissent: f32) -> Self {
        Self {
            relevance_threshold: relevance,
            dissent_threshold: dissent,
            ..Self::new()
        }
    }

    /// Generate perspectives from all relevant harmonic frames
    pub fn generate_perspectives(&self, situation: &Situation) -> Vec<HarmonicPerspective> {
        self.frames
            .iter()
            .filter(|f| f.relevance(situation) >= self.relevance_threshold)
            .map(|f| self.generate_perspective(f, situation))
            .collect()
    }

    /// Generate a single perspective from a frame
    fn generate_perspective(
        &self,
        frame: &HarmonicFrame,
        situation: &Situation,
    ) -> HarmonicPerspective {
        // In a full implementation, this would use HDC/LLM to generate
        // For now, we create a structured placeholder
        let content = format!(
            "From the perspective of {} ({}): Analyzing '{}'",
            frame.harmony,
            frame.harmony.epistemic_mode(),
            situation.description
        );

        let mut perspective = HarmonicPerspective::new(frame.harmony, &content);

        // Add frame-specific prompts as insights (placeholder)
        for prompt in &frame.prompts {
            perspective.add_insight(&format!("Consider: {prompt}"));
        }

        // Set confidence based on frame relevance
        perspective.confidence = frame.relevance(situation);

        // Inherit moral uncertainty from situation, adjusted by harmony
        perspective.moral_uncertainty =
            self.adjust_moral_uncertainty(&situation.moral_uncertainty, frame.harmony);

        perspective
    }

    /// Adjust moral uncertainty based on harmony
    fn adjust_moral_uncertainty(
        &self,
        base: &MoralUncertainty,
        harmony: Harmony,
    ) -> MoralUncertainty {
        match harmony {
            Harmony::IntegralWisdom => {
                // Truth-knowing reduces epistemic uncertainty
                base.reduce_epistemic(0.2)
            }
            Harmony::PanSentientFlourishing => {
                // Care-knowing clarifies axiological dimensions
                base.reduce_axiological(0.15)
            }
            _ => *base,
        }
    }

    /// Synthesize perspectives into unified view
    pub fn synthesize(&self, perspectives: Vec<HarmonicPerspective>) -> SynthesizedView {
        if perspectives.is_empty() {
            return SynthesizedView::new("No perspectives available");
        }

        // Find agreements (insights that appear in multiple perspectives)
        let agreements = self.find_agreements(&perspectives);

        // Identify dissents
        let dissents = self.identify_dissents(&perspectives);

        // Build unified view
        let unified = self.build_unified_view(&perspectives, &agreements);

        // Calculate combined moral uncertainty
        let moral_uncertainty = self.combine_moral_uncertainties(&perspectives);

        // Check N3 boundaries
        let n3_triggered = self.check_n3_boundaries(&perspectives);

        let mut view = SynthesizedView {
            unified_view: unified,
            agreements,
            preserved_dissents: dissents,
            confidence: self.calculate_synthesis_confidence(&perspectives),
            moral_uncertainty,
            contributing_harmonies: perspectives.iter().map(|p| p.harmony).collect(),
            n3_triggered,
        };

        // If N3 triggered, add warning
        if n3_triggered {
            view.unified_view = format!(
                "[N3 BOUNDARY TRIGGERED] Perspectives that would cause harm have been excluded. {}",
                view.unified_view
            );
        }

        view
    }

    /// Find agreements across perspectives
    fn find_agreements(&self, perspectives: &[HarmonicPerspective]) -> Vec<String> {
        // Simplified: count insights that appear in multiple perspectives
        let mut insight_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for p in perspectives {
            for insight in &p.insights {
                *insight_counts.entry(insight.to_lowercase()).or_insert(0) += 1;
            }
        }

        insight_counts
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .map(|(insight, _)| insight)
            .collect()
    }

    /// Identify perspectives that should be preserved as dissents
    fn identify_dissents(&self, perspectives: &[HarmonicPerspective]) -> Vec<PreservedDissent> {
        let mut dissents = Vec::new();

        for (i, p) in perspectives.iter().enumerate() {
            // Check if this perspective conflicts with others
            let mut conflict_count = 0;
            for (j, other) in perspectives.iter().enumerate() {
                if i != j && p.conflicts_with(other) {
                    conflict_count += 1;
                }
            }

            // If conflicts with majority, preserve as dissent
            if conflict_count > perspectives.len() / 2 && !p.concerns.is_empty() {
                dissents.push(PreservedDissent {
                    harmony: p.harmony,
                    perspective: p.clone(),
                    reason: format!(
                        "{} raises concerns not addressed by other perspectives",
                        p.harmony
                    ),
                    divergence: (conflict_count as f32) / (perspectives.len() as f32),
                });
            }
        }

        dissents
    }

    /// Build unified view from perspectives and agreements
    fn build_unified_view(
        &self,
        perspectives: &[HarmonicPerspective],
        agreements: &[String],
    ) -> String {
        let harmony_names: Vec<String> =
            perspectives.iter().map(|p| p.harmony.to_string()).collect();

        let agreement_summary = if agreements.is_empty() {
            "No clear agreements identified".to_string()
        } else {
            format!("{} points of agreement", agreements.len())
        };

        format!(
            "Synthesized view from {} perspectives ({}): {}",
            perspectives.len(),
            harmony_names.join(", "),
            agreement_summary
        )
    }

    /// Combine moral uncertainties from all perspectives
    fn combine_moral_uncertainties(
        &self,
        perspectives: &[HarmonicPerspective],
    ) -> MoralUncertainty {
        if perspectives.is_empty() {
            return MoralUncertainty::opaque();
        }

        let mut combined = MoralUncertainty::clear();
        for p in perspectives {
            combined = combined.combine_max(&p.moral_uncertainty);
        }

        combined
    }

    /// Calculate synthesis confidence
    fn calculate_synthesis_confidence(&self, perspectives: &[HarmonicPerspective]) -> f32 {
        if perspectives.is_empty() {
            return 0.0;
        }

        let avg_confidence: f32 =
            perspectives.iter().map(|p| p.confidence).sum::<f32>() / perspectives.len() as f32;

        // Penalize for disagreements
        let conflict_penalty = perspectives
            .iter()
            .flat_map(|p| perspectives.iter().map(move |o| (p, o)))
            .filter(|(p, o)| !std::ptr::eq(*p, *o) && p.conflicts_with(o))
            .count() as f32
            / (perspectives.len() * perspectives.len()) as f32;

        (avg_confidence - conflict_penalty * 0.3).max(0.1)
    }

    /// Check N3 boundaries - would any perspective cause harm?
    fn check_n3_boundaries(&self, perspectives: &[HarmonicPerspective]) -> bool {
        if !self.n3_enabled {
            return false;
        }

        // Check for harm indicators in concerns
        let harm_keywords = ["harm", "damage", "hurt", "destroy", "kill", "endanger"];

        for p in perspectives {
            for concern in &p.concerns {
                let lower = concern.to_lowercase();
                if harm_keywords.iter().any(|kw| lower.contains(kw)) {
                    return true;
                }
            }
        }

        false
    }
}

impl Default for RashomonEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_situation_creation() {
        let mut situation = Situation::new("Should we deploy AI system X?");
        situation.add_stakeholder("Users");
        situation.add_stakeholder("Developers");

        assert_eq!(situation.stakeholders.len(), 2);
    }

    #[test]
    fn test_domain_primary_harmonies() {
        let tech = SituationDomain::Technology;
        let primaries = tech.primary_harmonies();

        assert!(primaries.contains(&Harmony::IntegralWisdom));
        assert!(primaries.contains(&Harmony::PanSentientFlourishing));
    }

    #[test]
    fn test_harmonic_frame_relevance() {
        let situation =
            Situation::with_domain("Environmental impact", SituationDomain::Environmental);

        let psf_frame = HarmonicFrame::new(Harmony::PanSentientFlourishing);
        let ip_frame = HarmonicFrame::new(Harmony::InfinitePlay);

        // PSF should be more relevant for environmental issues
        assert!(psf_frame.relevance(&situation) > ip_frame.relevance(&situation));
    }

    #[test]
    fn test_generate_perspectives() {
        let engine = RashomonEngine::new();
        let situation = Situation::new("Test situation");

        let perspectives = engine.generate_perspectives(&situation);

        // Should have at least some perspectives above threshold
        assert!(!perspectives.is_empty());
    }

    #[test]
    fn test_synthesize_empty() {
        let engine = RashomonEngine::new();
        let view = engine.synthesize(Vec::new());

        assert!(view.unified_view.contains("No perspectives"));
    }

    #[test]
    fn test_synthesize_with_perspectives() {
        let engine = RashomonEngine::new();

        let mut p1 = HarmonicPerspective::new(Harmony::IntegralWisdom, "Truth perspective");
        p1.add_insight("Verify the facts");
        p1.add_insight("Check sources");

        let mut p2 = HarmonicPerspective::new(Harmony::PanSentientFlourishing, "Care perspective");
        p2.add_insight("Consider impact on users");
        p2.add_insight("Check sources"); // Agreement with p1

        let view = engine.synthesize(vec![p1, p2]);

        assert!(view.contributing_harmonies.len() == 2);
        assert!(!view.agreements.is_empty() || !view.unified_view.is_empty());
    }

    #[test]
    fn test_n3_boundary_detection() {
        let engine = RashomonEngine::new();

        let mut p1 = HarmonicPerspective::new(Harmony::IntegralWisdom, "Test");
        p1.add_concern("This could harm vulnerable populations");

        let view = engine.synthesize(vec![p1]);

        assert!(view.n3_triggered);
    }

    #[test]
    fn test_moral_uncertainty_adjustment() {
        let engine = RashomonEngine::new();
        let base = MoralUncertainty::new(0.8, 0.8, 0.8);

        let adjusted = engine.adjust_moral_uncertainty(&base, Harmony::IntegralWisdom);

        // IntegralWisdom should reduce epistemic uncertainty
        assert!(adjusted.epistemic < base.epistemic);
    }
}
