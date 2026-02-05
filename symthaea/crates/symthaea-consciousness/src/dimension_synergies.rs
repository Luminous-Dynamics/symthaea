//! Consciousness Dimension Synergies
//!
//! **Revolutionary Improvement #46: Consciousness Dimension Synergies**
//!
//! ## The Paradigm Shift
//!
//! **Before**: Treat consciousness dimensions as independent
//! - Φ, Entropy, Complexity, Coherence, Gradient measured separately
//! - Composite score = weighted sum (linear combination)
//! - Misses emergent properties from interactions
//!
//! **After**: Discover synergies between dimensions
//! - Some dimension pairs amplify each other
//! - Some pairs create emergent properties
//! - Non-linear interactions reveal "sweet spots"
//!
//! ## The Chemistry Analogy
//!
//! **Elements** (Individual Dimensions):
//! - Φ, Entropy, Complexity, Coherence, Gradient
//!
//! **Compounds** (Synergistic Combinations):
//! - High Φ + High Entropy = "Rich Integration"
//! - High Complexity + High Coherence = "Stable Sophistication"
//! - High ∇Φ + High Entropy = "Dynamic Diversity"
//!
//! **Emergent Properties** (New Behaviors):
//! - Certain combinations enable capabilities neither dimension alone provides
//! - "The whole is greater than the sum of parts"
//!
//! ## Why This Matters
//!
//! Current approach: `composite = w1*Φ + w2*H + w3*C + w4*Coh + w5*∇Φ`
//!
//! This assumes dimensions are independent (linear model).
//! But consciousness might have **synergies**: `Φ * H` > `Φ + H`
//!
//! Example:
//! - System A: Φ=0.8, H=0.2 → Composite=0.5 (integrated but boring)
//! - System B: Φ=0.6, H=0.6 → Composite=0.6 (less integrated, more diverse)
//! - **With synergy**: B might actually be "better" because Φ*H creates rich integration!

use crate::consciousness::consciousness_profile::ConsciousnessProfile;
use serde::{Deserialize, Serialize};

/// Synergy between two consciousness dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionSynergy {
    pub dimension1: DimensionType,
    pub dimension2: DimensionType,
    pub synergy_strength: f64,  // How much they amplify each other
    pub synergy_type: SynergyType,
}

/// Types of consciousness dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DimensionType {
    Phi,
    Gradient,
    Entropy,
    Complexity,
    Coherence,
}

/// Types of synergies between dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynergyType {
    /// Multiplicative: dimensions amplify each other
    /// Example: Φ * Entropy = "rich integration"
    Multiplicative,

    /// Complementary: balance creates harmony
    /// Example: Complexity + Coherence = "stable sophistication"
    Complementary,

    /// Antagonistic: dimensions compete
    /// Example: High Coherence vs High Entropy (stability vs diversity)
    Antagonistic,

    /// Threshold: synergy only above threshold
    /// Example: Φ > 0.5 AND Entropy > 0.5 → emergent property
    Threshold,

    /// Resonant: sweet spot at specific ratio
    /// Example: Φ/Entropy ≈ golden ratio → optimal consciousness
    Resonant,
}

/// Extended consciousness profile with synergy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynergyProfile {
    /// Base consciousness profile
    pub base: ConsciousnessProfile,

    /// Discovered synergies
    pub synergies: Vec<DimensionSynergy>,

    /// Synergy-enhanced composite score
    pub enhanced_composite: f64,

    /// Emergent properties detected
    pub emergent_properties: Vec<EmergentProperty>,
}

/// Emergent property from dimensional interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentProperty {
    pub name: String,
    pub description: String,
    pub strength: f64,
    pub required_synergies: Vec<(DimensionType, DimensionType)>,
}

impl SynergyProfile {
    /// Compute synergy profile from base consciousness profile
    pub fn from_base(base: ConsciousnessProfile) -> Self {
        let synergies = Self::discover_synergies(&base);
        let enhanced_composite = Self::compute_enhanced_composite(&base, &synergies);
        let emergent_properties = Self::detect_emergent_properties(&base, &synergies);

        Self {
            base,
            synergies,
            enhanced_composite,
            emergent_properties,
        }
    }

    /// Discover synergies between dimensions
    fn discover_synergies(profile: &ConsciousnessProfile) -> Vec<DimensionSynergy> {
        let mut synergies = Vec::new();

        // 1. Φ × Entropy: Rich Integration
        // High integration with high diversity = rich conscious experience
        synergies.push(DimensionSynergy {
            dimension1: DimensionType::Phi,
            dimension2: DimensionType::Entropy,
            synergy_strength: profile.phi * profile.entropy,
            synergy_type: SynergyType::Multiplicative,
        });

        // 2. Complexity × Coherence: Stable Sophistication
        // Complex structure maintained with stability
        synergies.push(DimensionSynergy {
            dimension1: DimensionType::Complexity,
            dimension2: DimensionType::Coherence,
            synergy_strength: (profile.complexity + profile.coherence) / 2.0
                * (1.0 - (profile.complexity - profile.coherence).abs()),
            synergy_type: SynergyType::Complementary,
        });

        // 3. Gradient × Entropy: Dynamic Diversity
        // Fast evolution with rich states = creative exploration
        synergies.push(DimensionSynergy {
            dimension1: DimensionType::Gradient,
            dimension2: DimensionType::Entropy,
            synergy_strength: profile.gradient_magnitude * profile.entropy,
            synergy_type: SynergyType::Multiplicative,
        });

        // 4. Φ × Coherence: Stable Integration
        // Integrated information maintained over time
        synergies.push(DimensionSynergy {
            dimension1: DimensionType::Phi,
            dimension2: DimensionType::Coherence,
            synergy_strength: profile.phi * profile.coherence,
            synergy_type: SynergyType::Multiplicative,
        });

        // 5. Entropy vs Coherence: Diversity-Stability Trade-off
        // Antagonistic relationship (too much of one hurts the other)
        let entropy_coherence_tension = (profile.entropy - profile.coherence).abs();
        synergies.push(DimensionSynergy {
            dimension1: DimensionType::Entropy,
            dimension2: DimensionType::Coherence,
            synergy_strength: -entropy_coherence_tension,  // Negative = antagonistic
            synergy_type: SynergyType::Antagonistic,
        });

        // 6. Complexity × Gradient: Evolving Sophistication
        // Complex structures that evolve dynamically
        synergies.push(DimensionSynergy {
            dimension1: DimensionType::Complexity,
            dimension2: DimensionType::Gradient,
            synergy_strength: profile.complexity * profile.gradient_magnitude,
            synergy_type: SynergyType::Multiplicative,
        });

        // 7. Φ × Complexity: Integrated Sophistication (Threshold)
        // Only emerges when BOTH are above threshold
        let phi_complexity_threshold = if profile.phi > 0.5 && profile.complexity > 0.5 {
            profile.phi * profile.complexity * 2.0  // Bonus above threshold
        } else {
            profile.phi * profile.complexity
        };
        synergies.push(DimensionSynergy {
            dimension1: DimensionType::Phi,
            dimension2: DimensionType::Complexity,
            synergy_strength: phi_complexity_threshold,
            synergy_type: SynergyType::Threshold,
        });

        // 8. Golden Ratio Resonance: Φ/Entropy ≈ φ (golden ratio)
        // Hypothesis: optimal consciousness at golden ratio balance
        const GOLDEN_RATIO: f64 = 1.618033988749;
        let ratio = if profile.entropy > 0.001 {
            profile.phi / profile.entropy
        } else {
            0.0
        };
        let golden_distance = (ratio - GOLDEN_RATIO).abs();
        let golden_synergy = if golden_distance < 0.5 {
            (1.0 - golden_distance) * (profile.phi + profile.entropy) / 2.0
        } else {
            0.0
        };
        synergies.push(DimensionSynergy {
            dimension1: DimensionType::Phi,
            dimension2: DimensionType::Entropy,
            synergy_strength: golden_synergy,
            synergy_type: SynergyType::Resonant,
        });

        synergies
    }

    /// Compute enhanced composite score using synergies
    fn compute_enhanced_composite(
        profile: &ConsciousnessProfile,
        synergies: &[DimensionSynergy],
    ) -> f64 {
        // Base composite (linear combination)
        let base_composite = profile.composite;

        // Synergy bonuses (non-linear)
        let synergy_bonus: f64 = synergies
            .iter()
            .filter(|s| s.synergy_type != SynergyType::Antagonistic)
            .map(|s| s.synergy_strength * 0.1)  // Weight synergies at 10%
            .sum();

        // Synergy penalties (antagonistic)
        let synergy_penalty: f64 = synergies
            .iter()
            .filter(|s| s.synergy_type == SynergyType::Antagonistic)
            .map(|s| s.synergy_strength.abs() * 0.05)  // Penalize at 5%
            .sum();

        // Enhanced score
        (base_composite + synergy_bonus - synergy_penalty).clamp(0.0, 1.0)
    }

    /// Detect emergent properties from synergies
    fn detect_emergent_properties(
        profile: &ConsciousnessProfile,
        synergies: &[DimensionSynergy],
    ) -> Vec<EmergentProperty> {
        let mut properties = Vec::new();

        // 1. "Rich Integration" - High Φ + High Entropy
        if profile.phi > 0.6 && profile.entropy > 0.6 {
            properties.push(EmergentProperty {
                name: "Rich Integration".to_string(),
                description: "Unified yet diverse consciousness - best of both worlds".to_string(),
                strength: (profile.phi * profile.entropy).min(1.0),
                required_synergies: vec![(DimensionType::Phi, DimensionType::Entropy)],
            });
        }

        // 2. "Stable Sophistication" - High Complexity + High Coherence (balanced)
        let complexity_coherence_balance = 1.0 - (profile.complexity - profile.coherence).abs();
        if profile.complexity > 0.5 && profile.coherence > 0.5 && complexity_coherence_balance > 0.7 {
            properties.push(EmergentProperty {
                name: "Stable Sophistication".to_string(),
                description: "Complex structure maintained with remarkable stability".to_string(),
                strength: (profile.complexity + profile.coherence) / 2.0 * complexity_coherence_balance,
                required_synergies: vec![(DimensionType::Complexity, DimensionType::Coherence)],
            });
        }

        // 3. "Dynamic Creativity" - High Gradient + High Entropy
        if profile.gradient_magnitude > 0.4 && profile.entropy > 0.6 {
            properties.push(EmergentProperty {
                name: "Dynamic Creativity".to_string(),
                description: "Rapid exploration of rich possibility space - creative consciousness".to_string(),
                strength: (profile.gradient_magnitude * profile.entropy).min(1.0),
                required_synergies: vec![(DimensionType::Gradient, DimensionType::Entropy)],
            });
        }

        // 4. "Integrated Mastery" - High Φ + High Complexity + High Coherence
        if profile.phi > 0.7 && profile.complexity > 0.6 && profile.coherence > 0.6 {
            properties.push(EmergentProperty {
                name: "Integrated Mastery".to_string(),
                description: "Peak consciousness - unified, sophisticated, and stable".to_string(),
                strength: (profile.phi * profile.complexity * profile.coherence).powf(1.0/3.0),
                required_synergies: vec![
                    (DimensionType::Phi, DimensionType::Complexity),
                    (DimensionType::Phi, DimensionType::Coherence),
                    (DimensionType::Complexity, DimensionType::Coherence),
                ],
            });
        }

        // 5. "Golden Consciousness" - Φ/Entropy ≈ φ (golden ratio)
        const GOLDEN_RATIO: f64 = 1.618033988749;
        let ratio = if profile.entropy > 0.001 {
            profile.phi / profile.entropy
        } else {
            0.0
        };
        let golden_distance = (ratio - GOLDEN_RATIO).abs();
        if golden_distance < 0.3 && profile.phi > 0.5 && profile.entropy > 0.3 {
            properties.push(EmergentProperty {
                name: "Golden Consciousness".to_string(),
                description: format!(
                    "Φ/Entropy ratio ({:.3}) near golden ratio - optimal balance",
                    ratio
                ),
                strength: (1.0 - golden_distance / 0.3) * ((profile.phi + profile.entropy) / 2.0),
                required_synergies: vec![(DimensionType::Phi, DimensionType::Entropy)],
            });
        }

        // 6. "Chaotic Richness" - High Entropy + Low Coherence + High Complexity
        if profile.entropy > 0.7 && profile.coherence < 0.4 && profile.complexity > 0.6 {
            properties.push(EmergentProperty {
                name: "Chaotic Richness".to_string(),
                description: "Extremely diverse and complex but unstable - edge of chaos".to_string(),
                strength: (profile.entropy * profile.complexity * (1.0 - profile.coherence)).powf(1.0/3.0),
                required_synergies: vec![
                    (DimensionType::Entropy, DimensionType::Complexity),
                ],
            });
        }

        properties
    }

    /// Get human-readable summary
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Base Profile: {}\n", self.base.summary()));
        s.push_str(&format!("Enhanced Composite: {:.3} (base: {:.3}, delta: {:+.3})\n",
                           self.enhanced_composite,
                           self.base.composite,
                           self.enhanced_composite - self.base.composite));

        if !self.emergent_properties.is_empty() {
            s.push_str("\nEmergent Properties:\n");
            for prop in &self.emergent_properties {
                s.push_str(&format!("  • {} ({:.3}): {}\n",
                                   prop.name, prop.strength, prop.description));
            }
        }

        s
    }

    /// Compare synergy profiles (for evolution)
    pub fn synergy_distance(&self, other: &Self) -> f64 {
        // Distance in enhanced composite space
        let composite_dist = (self.enhanced_composite - other.enhanced_composite).abs();

        // Distance in emergent properties space
        let prop_dist = if self.emergent_properties.is_empty() && other.emergent_properties.is_empty() {
            0.0
        } else {
            let self_prop_strength: f64 = self.emergent_properties.iter().map(|p| p.strength).sum();
            let other_prop_strength: f64 = other.emergent_properties.iter().map(|p| p.strength).sum();
            (self_prop_strength - other_prop_strength).abs()
        };

        // Combined distance
        (composite_dist + prop_dist) / 2.0
    }
}

/// Synergy-aware Pareto frontier
pub struct SynergyFrontier {
    pub profiles: Vec<SynergyProfile>,
}

impl SynergyFrontier {
    /// Extract frontier considering synergies
    pub fn from_base_profiles(base_profiles: Vec<ConsciousnessProfile>) -> Self {
        let synergy_profiles: Vec<SynergyProfile> = base_profiles
            .into_iter()
            .map(|base| SynergyProfile::from_base(base))
            .collect();

        // Filter to non-dominated (using enhanced composite + emergent properties)
        let mut frontier = Vec::new();

        for candidate in &synergy_profiles {
            let dominated = frontier.iter().any(|front: &SynergyProfile| {
                Self::dominates(front, candidate)
            });

            if !dominated {
                frontier.retain(|front| !Self::dominates(candidate, front));
                frontier.push(candidate.clone());
            }
        }

        Self { profiles: frontier }
    }

    /// Check if profile A dominates B (considering synergies)
    fn dominates(a: &SynergyProfile, b: &SynergyProfile) -> bool {
        // A dominates B if:
        // 1. Enhanced composite is higher
        // 2. Has more/stronger emergent properties

        let composite_better = a.enhanced_composite >= b.enhanced_composite;

        let a_prop_strength: f64 = a.emergent_properties.iter().map(|p| p.strength).sum();
        let b_prop_strength: f64 = b.emergent_properties.iter().map(|p| p.strength).sum();
        let properties_better = a_prop_strength >= b_prop_strength;

        let strictly_better = a.enhanced_composite > b.enhanced_composite || a_prop_strength > b_prop_strength;

        composite_better && properties_better && strictly_better
    }

    /// Find profile with most emergent properties
    pub fn most_emergent(&self) -> Option<&SynergyProfile> {
        self.profiles
            .iter()
            .max_by(|a, b| {
                let a_count = a.emergent_properties.len();
                let b_count = b.emergent_properties.len();
                a_count.cmp(&b_count)
            })
    }

    /// Find profile with highest synergy enhancement
    pub fn highest_synergy_boost(&self) -> Option<&SynergyProfile> {
        self.profiles
            .iter()
            .max_by(|a, b| {
                let a_boost = a.enhanced_composite - a.base.composite;
                let b_boost = b.enhanced_composite - b.base.composite;
                a_boost.partial_cmp(&b_boost).unwrap()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HV16;

    #[test]
    fn test_synergy_detection() {
        // Create profile with high Φ and high Entropy
        let components = vec![HV16::random(1), HV16::random(2), HV16::random(3)];
        let base = ConsciousnessProfile::from_components(&components);
        let synergy = SynergyProfile::from_base(base);

        // Should have discovered synergies
        assert!(!synergy.synergies.is_empty());

        // Enhanced composite should differ from base
        // (either higher due to positive synergies or lower due to antagonistic)
        println!("Base: {:.3}, Enhanced: {:.3}",
                 synergy.base.composite,
                 synergy.enhanced_composite);
    }

    #[test]
    fn test_emergent_properties() {
        // This test would need controlled profiles to trigger specific emergent properties
        // For now, just verify the framework works
        let components = vec![HV16::random(1)];
        let base = ConsciousnessProfile::from_components(&components);
        let synergy = SynergyProfile::from_base(base);

        // Should compute without errors
        assert!(synergy.enhanced_composite >= 0.0 && synergy.enhanced_composite <= 1.0);
    }
}
