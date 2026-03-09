//! Harmonic Types for GIS v4.0 "Kosmic Song"
//!
//! The Eight Harmonies represent eight fundamental ways of knowing,
//! each providing a distinct epistemic lens through which knowledge is perceived.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The Eight Harmonies - fundamental epistemic lenses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Harmony {
    /// Resonant Coherence (RC) - Integration-Knowing
    /// "How do the parts relate?"
    /// Weight: 0.20
    ResonantCoherence,

    /// Pan-Sentient Flourishing (PSF) - Care-Knowing
    /// "Who is affected?"
    /// Weight: 0.20
    PanSentientFlourishing,

    /// Integral Wisdom (IW) - Truth-Knowing
    /// "What is verifiable?"
    /// Weight: 0.15
    IntegralWisdom,

    /// Infinite Play (IP) - Creative-Knowing
    /// "What possibilities exist?"
    /// Weight: 0.10
    InfinitePlay,

    /// Universal Interconnectedness (UI) - Relational-Knowing
    /// "What connections exist?"
    /// Weight: 0.15
    UniversalInterconnectedness,

    /// Sacred Reciprocity (SR) - Exchange-Knowing
    /// "What flows back?"
    /// Weight: 0.10
    SacredReciprocity,

    /// Evolutionary Progression (EP) - Developmental-Knowing
    /// "What is emerging?"
    /// Weight: 0.09
    EvolutionaryProgression,

    /// Sacred Stillness (SS) - Apophatic-Knowing
    /// "What must be released?"
    /// Weight: 0.13
    SacredStillness,
}

impl Harmony {
    /// Get all harmonies in order
    pub const ALL: [Harmony; 8] = [
        Harmony::ResonantCoherence,
        Harmony::PanSentientFlourishing,
        Harmony::IntegralWisdom,
        Harmony::InfinitePlay,
        Harmony::UniversalInterconnectedness,
        Harmony::SacredReciprocity,
        Harmony::EvolutionaryProgression,
        Harmony::SacredStillness,
    ];

    /// Get the base weight for this harmony
    pub fn base_weight(&self) -> f32 {
        match self {
            Harmony::ResonantCoherence => 0.17,
            Harmony::PanSentientFlourishing => 0.17,
            Harmony::IntegralWisdom => 0.13,
            Harmony::InfinitePlay => 0.09,
            Harmony::UniversalInterconnectedness => 0.13,
            Harmony::SacredReciprocity => 0.09,
            Harmony::EvolutionaryProgression => 0.09,
            Harmony::SacredStillness => 0.13,
        }
    }

    /// Get the short code (e.g., "RC", "PSF")
    pub fn code(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "RC",
            Harmony::PanSentientFlourishing => "PSF",
            Harmony::IntegralWisdom => "IW",
            Harmony::InfinitePlay => "IP",
            Harmony::UniversalInterconnectedness => "UI",
            Harmony::SacredReciprocity => "SR",
            Harmony::EvolutionaryProgression => "EP",
            Harmony::SacredStillness => "SS",
        }
    }

    /// Get the epistemic mode name
    pub fn epistemic_mode(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "Integration-Knowing",
            Harmony::PanSentientFlourishing => "Care-Knowing",
            Harmony::IntegralWisdom => "Truth-Knowing",
            Harmony::InfinitePlay => "Creative-Knowing",
            Harmony::UniversalInterconnectedness => "Relational-Knowing",
            Harmony::SacredReciprocity => "Exchange-Knowing",
            Harmony::EvolutionaryProgression => "Developmental-Knowing",
            Harmony::SacredStillness => "Apophatic-Knowing",
        }
    }

    /// Get the guiding question for this harmony
    pub fn guiding_question(&self) -> &'static str {
        match self {
            Harmony::ResonantCoherence => "How do the parts relate?",
            Harmony::PanSentientFlourishing => "Who is affected?",
            Harmony::IntegralWisdom => "What is verifiable?",
            Harmony::InfinitePlay => "What possibilities exist?",
            Harmony::UniversalInterconnectedness => "What connections exist?",
            Harmony::SacredReciprocity => "What flows back?",
            Harmony::EvolutionaryProgression => "What is emerging?",
            Harmony::SacredStillness => "What must be released?",
        }
    }

    /// Parse from short code
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_uppercase().as_str() {
            "RC" => Some(Harmony::ResonantCoherence),
            "PSF" => Some(Harmony::PanSentientFlourishing),
            "IW" => Some(Harmony::IntegralWisdom),
            "IP" => Some(Harmony::InfinitePlay),
            "UI" => Some(Harmony::UniversalInterconnectedness),
            "SR" => Some(Harmony::SacredReciprocity),
            "EP" => Some(Harmony::EvolutionaryProgression),
            "SS" => Some(Harmony::SacredStillness),
            _ => None,
        }
    }
}

/// Harmonic impact - how much each harmony is affected by a piece of knowledge/ignorance
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HarmonicImpact {
    /// Impact on Resonant Coherence (0.0-1.0)
    pub resonant_coherence: f32,
    /// Impact on Pan-Sentient Flourishing (0.0-1.0)
    pub pan_sentient_flourishing: f32,
    /// Impact on Integral Wisdom (0.0-1.0)
    pub integral_wisdom: f32,
    /// Impact on Infinite Play (0.0-1.0)
    pub infinite_play: f32,
    /// Impact on Universal Interconnectedness (0.0-1.0)
    pub universal_interconnectedness: f32,
    /// Impact on Sacred Reciprocity (0.0-1.0)
    pub sacred_reciprocity: f32,
    /// Impact on Evolutionary Progression (0.0-1.0)
    pub evolutionary_progression: f32,
    /// Impact on Sacred Stillness (0.0-1.0)
    pub sacred_stillness: f32,
}

impl HarmonicImpact {
    /// Create with all impacts set to zero
    pub fn zero() -> Self {
        Self {
            resonant_coherence: 0.0,
            pan_sentient_flourishing: 0.0,
            integral_wisdom: 0.0,
            infinite_play: 0.0,
            universal_interconnectedness: 0.0,
            sacred_reciprocity: 0.0,
            evolutionary_progression: 0.0,
            sacred_stillness: 0.0,
        }
    }

    /// Create with all impacts set to a uniform value
    pub fn uniform(value: f32) -> Self {
        Self {
            resonant_coherence: value,
            pan_sentient_flourishing: value,
            integral_wisdom: value,
            infinite_play: value,
            universal_interconnectedness: value,
            sacred_reciprocity: value,
            evolutionary_progression: value,
            sacred_stillness: value,
        }
    }

    /// Get impact for a specific harmony
    pub fn get(&self, harmony: Harmony) -> f32 {
        match harmony {
            Harmony::ResonantCoherence => self.resonant_coherence,
            Harmony::PanSentientFlourishing => self.pan_sentient_flourishing,
            Harmony::IntegralWisdom => self.integral_wisdom,
            Harmony::InfinitePlay => self.infinite_play,
            Harmony::UniversalInterconnectedness => self.universal_interconnectedness,
            Harmony::SacredReciprocity => self.sacred_reciprocity,
            Harmony::EvolutionaryProgression => self.evolutionary_progression,
            Harmony::SacredStillness => self.sacred_stillness,
        }
    }

    /// Set impact for a specific harmony
    pub fn set(&mut self, harmony: Harmony, value: f32) {
        let target = match harmony {
            Harmony::ResonantCoherence => &mut self.resonant_coherence,
            Harmony::PanSentientFlourishing => &mut self.pan_sentient_flourishing,
            Harmony::IntegralWisdom => &mut self.integral_wisdom,
            Harmony::InfinitePlay => &mut self.infinite_play,
            Harmony::UniversalInterconnectedness => &mut self.universal_interconnectedness,
            Harmony::SacredReciprocity => &mut self.sacred_reciprocity,
            Harmony::EvolutionaryProgression => &mut self.evolutionary_progression,
            Harmony::SacredStillness => &mut self.sacred_stillness,
        };
        *target = value;
    }

    /// Get as array (ordered by Harmony::ALL)
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.resonant_coherence,
            self.pan_sentient_flourishing,
            self.integral_wisdom,
            self.infinite_play,
            self.universal_interconnectedness,
            self.sacred_reciprocity,
            self.evolutionary_progression,
            self.sacred_stillness,
        ]
    }

    /// Create from array (ordered by Harmony::ALL)
    pub fn from_array(values: [f32; 8]) -> Self {
        Self {
            resonant_coherence: values[0],
            pan_sentient_flourishing: values[1],
            integral_wisdom: values[2],
            infinite_play: values[3],
            universal_interconnectedness: values[4],
            sacred_reciprocity: values[5],
            evolutionary_progression: values[6],
            sacred_stillness: values[7],
        }
    }

    /// Calculate total weighted impact
    pub fn total_weighted_impact(&self) -> f32 {
        Harmony::ALL
            .iter()
            .map(|h| h.base_weight() * self.get(*h))
            .sum()
    }

    /// Get the primary (most affected) harmony
    pub fn primary_harmony(&self) -> Harmony {
        Harmony::ALL
            .iter()
            .max_by(|a, b| {
                self.get(**a)
                    .partial_cmp(&self.get(**b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(Harmony::IntegralWisdom)
    }

    /// Get harmonies with impact above threshold
    pub fn affected_harmonies(&self, threshold: f32) -> Vec<(Harmony, f32)> {
        Harmony::ALL
            .iter()
            .filter_map(|h| {
                let impact = self.get(*h);
                if impact > threshold {
                    Some((*h, impact))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Format as short string (e.g., "RC:0.8,PSF:0.5")
    pub fn short_format(&self) -> String {
        Harmony::ALL
            .iter()
            .filter_map(|h| {
                let impact = self.get(*h);
                if impact > 0.01 {
                    Some(format!("{}:{:.1}", h.code(), impact))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Validate all impacts are in [0.0, 1.0]
    pub fn is_valid(&self) -> bool {
        let in_range = |v: f32| (0.0..=1.0).contains(&v) && v.is_finite();
        in_range(self.resonant_coherence)
            && in_range(self.pan_sentient_flourishing)
            && in_range(self.integral_wisdom)
            && in_range(self.infinite_play)
            && in_range(self.universal_interconnectedness)
            && in_range(self.sacred_reciprocity)
            && in_range(self.evolutionary_progression)
            && in_range(self.sacred_stillness)
    }

    /// Clamp all values to [0.0, 1.0]
    pub fn clamp(&self) -> Self {
        Self {
            resonant_coherence: self.resonant_coherence.clamp(0.0, 1.0),
            pan_sentient_flourishing: self.pan_sentient_flourishing.clamp(0.0, 1.0),
            integral_wisdom: self.integral_wisdom.clamp(0.0, 1.0),
            infinite_play: self.infinite_play.clamp(0.0, 1.0),
            universal_interconnectedness: self.universal_interconnectedness.clamp(0.0, 1.0),
            sacred_reciprocity: self.sacred_reciprocity.clamp(0.0, 1.0),
            evolutionary_progression: self.evolutionary_progression.clamp(0.0, 1.0),
            sacred_stillness: self.sacred_stillness.clamp(0.0, 1.0),
        }
    }
}

impl Default for HarmonicImpact {
    fn default() -> Self {
        Self::zero()
    }
}

/// Harmonic ignorance - tracks what we don't know through harmonic lenses
///
/// This is the H-dimension extension to the E/N/M classification system.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HarmonicIgnorance {
    /// Base ignorance type (what kind of gap in knowledge)
    pub ignorance_type: IgnoranceType,
    /// How much each harmony is affected by this ignorance
    pub harmonic_impact: HarmonicImpact,
    /// Possible resolution paths, weighted by harmony
    pub resolution_paths: Vec<HarmonicResolution>,
}

/// Types of ignorance in the GIS framework
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IgnoranceType {
    /// Known unknown - we know what we don't know
    KnownUnknown,
    /// Unknown unknown - potential gaps we haven't identified
    UnknownUnknown,
    /// Temporary - will be resolved with more data
    Temporary,
    /// Fundamental - may be unresolvable
    Fundamental,
    /// Ethical - moral uncertainty
    Ethical,
    /// Methodological - limits of our approach
    Methodological,
}

/// A resolution path for harmonic ignorance
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HarmonicResolution {
    /// Which harmony this resolution primarily serves
    pub target_harmony: Harmony,
    /// Description of the resolution path
    pub description: String,
    /// Estimated difficulty (0.0-1.0)
    pub difficulty: f32,
    /// Expected information gain if resolved
    pub expected_gain: f32,
}

impl HarmonicIgnorance {
    /// Calculate total harmonic impact (weighted sum)
    pub fn total_impact(&self) -> f32 {
        self.harmonic_impact.total_weighted_impact()
    }

    /// Get the primary harmony gap (most affected)
    pub fn primary_gap(&self) -> Harmony {
        self.harmonic_impact.primary_harmony()
    }

    /// Calculate harmonic EIG (Expected Information Gain)
    ///
    /// Formula: H-EIG = BaseEIG × (1 + Σ(w_h × impact_h × relevance_h))
    pub fn harmonic_eig(&self, base_eig: f32, context_relevance: &HarmonicImpact) -> f32 {
        let harmonic_weight: f32 = Harmony::ALL
            .iter()
            .map(|h| h.base_weight() * self.harmonic_impact.get(*h) * context_relevance.get(*h))
            .sum();

        base_eig * (1.0 + harmonic_weight)
    }
}

/// Harmonic profile - current activation levels of all harmonies
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HarmonicProfile {
    /// Activation level for each harmony (0.0-1.0)
    pub activations: HarmonicImpact,
    /// Currently dominant harmony
    pub dominant: Harmony,
    /// How aligned the profile is (coherence measure)
    pub alignment: f32,
}

impl HarmonicProfile {
    /// Create a balanced profile (all harmonies equally active)
    pub fn balanced() -> Self {
        Self {
            activations: HarmonicImpact::uniform(0.5),
            dominant: Harmony::IntegralWisdom,
            alignment: 1.0,
        }
    }

    /// Recalculate dominant harmony and alignment
    pub fn recalculate(&mut self) {
        self.dominant = self.activations.primary_harmony();

        // Alignment is cosine similarity to a balanced profile
        let n = Harmony::ALL.len() as f32;
        let balanced = HarmonicImpact::uniform(1.0 / n);
        let dot: f32 = Harmony::ALL
            .iter()
            .map(|h| self.activations.get(*h) * balanced.get(*h))
            .sum();
        let mag_self: f32 = Harmony::ALL
            .iter()
            .map(|h| self.activations.get(*h).powi(2))
            .sum::<f32>()
            .sqrt();
        let mag_balanced: f32 = (n * (1.0 / n).powi(2)).sqrt();

        self.alignment = if mag_self > 0.0 {
            dot / (mag_self * mag_balanced)
        } else {
            0.0
        };
    }
}

impl Default for HarmonicProfile {
    fn default() -> Self {
        Self::balanced()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmony_weights_sum_to_one() {
        let sum: f32 = Harmony::ALL.iter().map(|h| h.base_weight()).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_harmonic_impact_primary() {
        let mut impact = HarmonicImpact::zero();
        impact.pan_sentient_flourishing = 0.9;
        impact.integral_wisdom = 0.3;

        assert_eq!(impact.primary_harmony(), Harmony::PanSentientFlourishing);
    }

    #[test]
    fn test_harmonic_impact_format() {
        let mut impact = HarmonicImpact::zero();
        impact.resonant_coherence = 0.8;
        impact.infinite_play = 0.3;

        let formatted = impact.short_format();
        assert!(formatted.contains("RC:0.8"));
        assert!(formatted.contains("IP:0.3"));
    }

    #[test]
    fn test_harmony_from_code() {
        assert_eq!(Harmony::from_code("RC"), Some(Harmony::ResonantCoherence));
        assert_eq!(
            Harmony::from_code("psf"),
            Some(Harmony::PanSentientFlourishing)
        );
        assert_eq!(Harmony::from_code("invalid"), None);
    }
}
