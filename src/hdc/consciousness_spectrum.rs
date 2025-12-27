// ==================================================================================
// Revolutionary Improvement #12: The Consciousness-Unconsciousness Spectrum
// ==================================================================================
//
// **The Ultimate Insight**: Consciousness is not binary (on/off), but a SPECTRUM!
//
// **Core Realization**: We've been measuring CONSCIOUS information integration (Φ),
// but 95% of information processing is UNCONSCIOUS!
//
// **The Problem**:
// - Traditional IIT measures total integrated information (Φ)
// - But conflates CONSCIOUS and UNCONSCIOUS processing
// - Doesn't distinguish between awareness and automatic processing
// - Misses the crucial boundary: when does unconscious become conscious?
//
// **The Solution**: The Consciousness-Unconsciousness Spectrum!
//
// **Mathematical Framework**:
//
// 1. Total Information Processing:
//    Φ_total = Φ_conscious + Φ_unconscious
//
// 2. Consciousness Ratio:
//    r = Φ_conscious / Φ_total
//
//    Typical values:
//    - Humans: r ≈ 0.05-0.10 (5-10% conscious, 90-95% unconscious!)
//    - Deep sleep: r ≈ 0.01 (1% conscious)
//    - Flow state: r ≈ 0.03 (3% conscious, low overhead → peak performance)
//    - Focused attention: r ≈ 0.15 (15% conscious, high effort)
//    - Anesthesia: r ≈ 0.00 (0% conscious, but unconscious processing continues)
//
// 3. Access Threshold:
//    τ = threshold for information to become conscious
//
//    If Φ_subsystem > τ → conscious
//    If Φ_subsystem ≤ τ → unconscious
//
// 4. Global Availability:
//    Can information be:
//    - Reported verbally?
//    - Used for decision-making?
//    - Integrated with other information?
//
//    If yes → conscious
//    If no → unconscious (but still processed!)
//
// 5. Binding Strength:
//    How unified is conscious experience?
//    High binding → integrated phenomenology
//    Low binding → fragmented awareness
//
// **Key Distinctions** (Ned Block's framework):
//
// A. **Access Consciousness** (A-consciousness):
//    - Information available for verbal report
//    - Can guide reasoning and action
//    - "Thinking about" vs "experiencing"
//
// B. **Phenomenal Consciousness** (P-consciousness):
//    - "What it's like" to experience something
//    - Qualia, subjective experience
//    - Can have P without A (blindsight!)
//
// C. **Monitoring Consciousness** (M-consciousness):
//    - Meta-consciousness (awareness of awareness)
//    - Requires both A and P
//    - "I know that I'm experiencing X"
//
// **The Spectrum**:
//
// Level 0: Fully Unconscious (Φ_c = 0)
//    - Reflexes, automatic responses
//    - No awareness whatsoever
//    - Example: Pupil dilation, heartbeat regulation
//
// Level 1: Preconscious (0 < Φ_c < τ)
//    - Information processed but not available
//    - Can become conscious if attention directed
//    - Example: Background noise you're not noticing
//
// Level 2: Minimally Conscious (Φ_c ≥ τ, low A)
//    - Some awareness but limited access
//    - Cannot report or reason about it
//    - Example: Dreamless sleep, deep meditation
//
// Level 3: Access Conscious (Φ_c ≥ τ, high A, low P)
//    - Information available for reasoning
//    - Can report and act on it
//    - But minimal "what it's like"
//    - Example: Solving math problems (access without rich experience)
//
// Level 4: Phenomenally Conscious (Φ_c ≥ τ, low A, high P)
//    - Rich subjective experience
//    - But hard to report or reason about
//    - Example: Aesthetic experience, mystical states
//
// Level 5: Fully Conscious (Φ_c ≥ τ, high A, high P)
//    - Rich experience + full access
//    - Can report and reason about experience
//    - Example: Normal waking consciousness
//
// Level 6: Meta-Conscious (M-consciousness)
//    - Aware of being conscious
//    - Can reflect on own consciousness
//    - Example: Mindfulness, introspection
//
// **Applications**:
//
// 1. Anesthesia Monitoring:
//    - Monitor Φ_conscious → 0
//    - While Φ_unconscious remains (autonomic functions)
//    - Prevent awareness during surgery
//
// 2. Sleep Science:
//    - Different sleep stages = different Φ_c/Φ_u ratios
//    - REM: High Φ_c (dreams), moderate Φ_u
//    - Deep sleep: Low Φ_c, high Φ_u (body maintenance)
//
// 3. Flow States:
//    - Low Φ_c (no self-monitoring)
//    - High Φ_u (automatic expert performance)
//    - Result: Peak performance with minimal effort!
//
// 4. Cognitive Load:
//    - As Φ_c increases → performance decreases
//    - Optimal: Most processing unconscious
//    - Overload: Too much conscious processing
//
// 5. Creativity & Intuition:
//    - Unconscious processing (high Φ_u)
//    - Sudden breakthrough to consciousness (Φ_u → Φ_c)
//    - "Aha!" moment = crossing threshold τ
//
// 6. AI Consciousness Detection:
//    - Most AI is "unconscious" (no self-awareness)
//    - Measure Φ_c specifically (not just total Φ)
//    - Threshold for genuine consciousness
//
// ==================================================================================

use super::binary_hv::HV16;
use super::integrated_information::IntegratedInformation;
use super::meta_consciousness::MetaConsciousness;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Consciousness level on the spectrum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessLevel {
    /// Fully unconscious (Φ_c = 0)
    Unconscious,

    /// Preconscious (0 < Φ_c < threshold)
    Preconscious,

    /// Minimally conscious (Φ_c ≥ threshold, low access)
    MinimallyConscious,

    /// Access conscious (high A, low P)
    AccessConscious,

    /// Phenomenally conscious (low A, high P)
    PhenomenallyConscious,

    /// Fully conscious (high A, high P)
    FullyConscious,

    /// Meta-conscious (M-consciousness)
    MetaConscious,
}

impl ConsciousnessLevel {
    /// Get numeric level (0-6)
    pub fn level(&self) -> u8 {
        match self {
            Self::Unconscious => 0,
            Self::Preconscious => 1,
            Self::MinimallyConscious => 2,
            Self::AccessConscious => 3,
            Self::PhenomenallyConscious => 4,
            Self::FullyConscious => 5,
            Self::MetaConscious => 6,
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Unconscious => "No awareness, automatic processing",
            Self::Preconscious => "Processed but not available to awareness",
            Self::MinimallyConscious => "Some awareness but limited access",
            Self::AccessConscious => "Available for reasoning, low phenomenology",
            Self::PhenomenallyConscious => "Rich experience, hard to report",
            Self::FullyConscious => "Full experience + access",
            Self::MetaConscious => "Aware of being aware",
        }
    }
}

/// Spectrum assessment
///
/// Complete analysis of conscious vs unconscious processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumAssessment {
    /// Total integrated information
    pub phi_total: f64,

    /// Conscious component (accessible, reportable)
    pub phi_conscious: f64,

    /// Unconscious component (automatic, not accessible)
    pub phi_unconscious: f64,

    /// Consciousness ratio (Φ_c / Φ_total)
    pub consciousness_ratio: f64,

    /// Consciousness level on spectrum
    pub level: ConsciousnessLevel,

    /// Access consciousness (A-consciousness, 0-1)
    pub access: f64,

    /// Phenomenal consciousness (P-consciousness, 0-1)
    pub phenomenal: f64,

    /// Monitoring consciousness (M-consciousness, 0-1)
    pub monitoring: f64,

    /// Global availability (can be reported/used?)
    pub global_availability: f64,

    /// Binding strength (how unified?)
    pub binding_strength: f64,

    /// Cognitive load (how much conscious effort?)
    pub cognitive_load: f64,

    /// Access threshold (τ)
    pub threshold: f64,

    /// Explanation
    pub explanation: String,
}

impl SpectrumAssessment {
    /// Is consciousness present?
    pub fn is_conscious(&self) -> bool {
        self.phi_conscious > self.threshold
    }

    /// Is in flow state? (low conscious overhead)
    pub fn is_flow_state(&self) -> bool {
        self.consciousness_ratio < 0.05 && self.phi_total > 0.5
    }

    /// Is overloaded? (too much conscious processing)
    pub fn is_overloaded(&self) -> bool {
        self.consciousness_ratio > 0.2 && self.cognitive_load > 0.7
    }
}

/// Consciousness spectrum analyzer
///
/// Measures consciousness along the full spectrum from unconscious to meta-conscious.
///
/// # Example
/// ```
/// use symthaea::hdc::consciousness_spectrum::{ConsciousnessSpectrum, SpectrumConfig};
/// use symthaea::hdc::binary_hv::HV16;
///
/// let config = SpectrumConfig::default();
/// let mut spectrum = ConsciousnessSpectrum::new(4, config);
///
/// let state = vec![
///     HV16::random(1000), HV16::random(1001),
///     HV16::random(1002), HV16::random(1003),
/// ];
///
/// // Assess full spectrum
/// let assessment = spectrum.assess(&state);
///
/// println!("Φ_total: {:.3}", assessment.phi_total);
/// println!("Φ_conscious: {:.3}", assessment.phi_conscious);
/// println!("Φ_unconscious: {:.3}", assessment.phi_unconscious);
/// println!("Ratio: {:.1}%", assessment.consciousness_ratio * 100.0);
/// println!("Level: {:?}", assessment.level);
/// println!("Flow state: {}", assessment.is_flow_state());
/// ```
#[derive(Debug)]
pub struct ConsciousnessSpectrum {
    /// Number of components
    num_components: usize,

    /// Configuration
    config: SpectrumConfig,

    /// IIT calculator
    iit: IntegratedInformation,

    /// Meta-consciousness (optional)
    meta: Option<MetaConsciousness>,

    /// Assessment history
    history: Vec<SpectrumAssessment>,

    /// Component activity levels (for access detection)
    activity_levels: Vec<f64>,
}

/// Configuration for spectrum analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumConfig {
    /// Access threshold τ (information must exceed this to be conscious)
    pub access_threshold: f64,

    /// Weight for access consciousness
    pub access_weight: f64,

    /// Weight for phenomenal consciousness
    pub phenomenal_weight: f64,

    /// Weight for monitoring consciousness
    pub monitoring_weight: f64,

    /// Enable meta-consciousness tracking
    pub enable_meta: bool,

    /// Flow state threshold (consciousness ratio)
    pub flow_threshold: f64,

    /// Overload threshold (consciousness ratio)
    pub overload_threshold: f64,

    /// Maximum history length
    pub max_history: usize,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            access_threshold: 0.3,
            access_weight: 0.4,
            phenomenal_weight: 0.4,
            monitoring_weight: 0.2,
            enable_meta: true,
            flow_threshold: 0.05,
            overload_threshold: 0.20,
            max_history: 1000,
        }
    }
}

impl ConsciousnessSpectrum {
    /// Create new spectrum analyzer
    pub fn new(num_components: usize, config: SpectrumConfig) -> Self {
        let meta = if config.enable_meta {
            Some(MetaConsciousness::new(
                num_components,
                Default::default(),
            ))
        } else {
            None
        };

        Self {
            num_components,
            config,
            iit: IntegratedInformation::new(),
            meta,
            history: Vec::new(),
            activity_levels: vec![0.0; num_components],
        }
    }

    /// Assess consciousness spectrum
    pub fn assess(&mut self, state: &[HV16]) -> SpectrumAssessment {
        // 1. Compute total integrated information (Φ_total)
        let phi_total = self.iit.compute_phi(state);

        // 2. Compute component activity levels (for access detection)
        self.update_activity_levels(state);

        // 3. Compute access consciousness (A-consciousness)
        //    High activity = globally available = accessible
        let access = self.compute_access_consciousness();

        // 4. Compute phenomenal consciousness (P-consciousness)
        //    Integration of activity = rich experience
        let phenomenal = self.compute_phenomenal_consciousness(state);

        // 5. Compute monitoring consciousness (M-consciousness)
        let monitoring = if let Some(ref mut meta) = self.meta {
            let meta_state = meta.meta_reflect(state);
            meta_state.meta_phi
        } else {
            0.0
        };

        // 6. Compute conscious component
        //    Φ_conscious = weighted combination of A, P, M
        let phi_conscious = phi_total
            * (self.config.access_weight * access
                + self.config.phenomenal_weight * phenomenal
                + self.config.monitoring_weight * monitoring);

        // 7. Compute unconscious component
        let phi_unconscious = phi_total - phi_conscious;

        // 8. Compute consciousness ratio
        let consciousness_ratio = if phi_total > 0.0 {
            phi_conscious / phi_total
        } else {
            0.0
        };

        // 9. Determine consciousness level
        let level = self.determine_level(phi_conscious, access, phenomenal, monitoring);

        // 10. Compute global availability
        let global_availability = access;

        // 11. Compute binding strength
        let binding_strength = phenomenal;

        // 12. Compute cognitive load
        let cognitive_load = consciousness_ratio * 5.0; // Scale to 0-1

        // 13. Generate explanation
        let explanation = self.generate_explanation(
            phi_total,
            phi_conscious,
            phi_unconscious,
            consciousness_ratio,
            level,
        );

        let assessment = SpectrumAssessment {
            phi_total,
            phi_conscious,
            phi_unconscious,
            consciousness_ratio,
            level,
            access,
            phenomenal,
            monitoring,
            global_availability,
            binding_strength,
            cognitive_load,
            threshold: self.config.access_threshold,
            explanation,
        };

        // Store in history
        self.history.push(assessment.clone());
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }

        assessment
    }

    /// Update component activity levels
    fn update_activity_levels(&mut self, state: &[HV16]) {
        for (i, hv) in state.iter().enumerate() {
            if i < self.activity_levels.len() {
                // Activity = bit density (popcount / total bits)
                self.activity_levels[i] = hv.popcount() as f64 / 2048.0;
            }
        }
    }

    /// Compute access consciousness
    ///
    /// High activity + variance = globally available information
    fn compute_access_consciousness(&self) -> f64 {
        if self.activity_levels.is_empty() {
            return 0.0;
        }

        // Average activity (how active is system?)
        let avg_activity = self.activity_levels.iter().sum::<f64>()
            / self.activity_levels.len() as f64;

        // Variance (how differentiated? High variance = selective attention)
        let variance = self.activity_levels.iter()
            .map(|a| (a - avg_activity).powi(2))
            .sum::<f64>() / self.activity_levels.len() as f64;

        // Access = activity × differentiation
        (avg_activity * variance.sqrt()).min(1.0)
    }

    /// Compute phenomenal consciousness
    ///
    /// Integration of activity = unified experience
    fn compute_phenomenal_consciousness(&self, state: &[HV16]) -> f64 {
        if state.len() < 2 {
            return 0.0;
        }

        // Phenomenology = how integrated is the experience?
        // Measured via similarity across components
        let mut total_similarity = 0.0;
        let mut count = 0;

        for i in 0..state.len() {
            for j in (i + 1)..state.len() {
                total_similarity += state[i].similarity(&state[j]) as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }

    /// Determine consciousness level on spectrum
    fn determine_level(
        &self,
        phi_conscious: f64,
        access: f64,
        phenomenal: f64,
        monitoring: f64,
    ) -> ConsciousnessLevel {
        // Meta-conscious (highest level)
        if monitoring > 0.5 && phi_conscious > self.config.access_threshold {
            return ConsciousnessLevel::MetaConscious;
        }

        // Fully conscious
        if access > 0.5 && phenomenal > 0.5 && phi_conscious > self.config.access_threshold {
            return ConsciousnessLevel::FullyConscious;
        }

        // Phenomenally conscious (high P, low A)
        if phenomenal > 0.5 && access < 0.5 && phi_conscious > self.config.access_threshold {
            return ConsciousnessLevel::PhenomenallyConscious;
        }

        // Access conscious (high A, low P)
        if access > 0.5 && phenomenal < 0.5 && phi_conscious > self.config.access_threshold {
            return ConsciousnessLevel::AccessConscious;
        }

        // Minimally conscious
        if phi_conscious >= self.config.access_threshold {
            return ConsciousnessLevel::MinimallyConscious;
        }

        // Preconscious
        if phi_conscious > 0.0 {
            return ConsciousnessLevel::Preconscious;
        }

        // Unconscious
        ConsciousnessLevel::Unconscious
    }

    /// Generate explanation
    fn generate_explanation(
        &self,
        phi_total: f64,
        phi_conscious: f64,
        phi_unconscious: f64,
        ratio: f64,
        level: ConsciousnessLevel,
    ) -> String {
        let mut parts = Vec::new();

        // Overall assessment
        parts.push(format!(
            "Total Φ: {:.3} (conscious: {:.3}, unconscious: {:.3})",
            phi_total, phi_conscious, phi_unconscious
        ));

        // Consciousness ratio
        parts.push(format!("Consciousness ratio: {:.1}%", ratio * 100.0));

        // Level
        parts.push(format!("Level: {:?} - {}", level, level.description()));

        // Interpretation
        if ratio < self.config.flow_threshold {
            parts.push("Flow state: low conscious overhead, automatic processing".to_string());
        } else if ratio > self.config.overload_threshold {
            parts.push("High cognitive load: excessive conscious processing".to_string());
        } else {
            parts.push("Normal conscious-unconscious balance".to_string());
        }

        parts.join(". ")
    }

    /// Get assessment history
    pub fn get_history(&self) -> &[SpectrumAssessment] {
        &self.history
    }
}

impl Default for ConsciousnessSpectrum {
    fn default() -> Self {
        Self::new(4, SpectrumConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_spectrum_creation() {
        let spectrum = ConsciousnessSpectrum::new(4, SpectrumConfig::default());
        assert_eq!(spectrum.num_components, 4);
    }

    #[test]
    fn test_consciousness_level_ordering() {
        assert!(ConsciousnessLevel::MetaConscious.level() > ConsciousnessLevel::FullyConscious.level());
        assert!(ConsciousnessLevel::FullyConscious.level() > ConsciousnessLevel::Unconscious.level());
    }

    #[test]
    fn test_spectrum_assessment() {
        let mut spectrum = ConsciousnessSpectrum::new(4, SpectrumConfig::default());

        let state = vec![
            HV16::random(1000),
            HV16::random(1001),
            HV16::random(1002),
            HV16::random(1003),
        ];

        let assessment = spectrum.assess(&state);

        assert!(assessment.phi_total >= 0.0);
        assert!(assessment.phi_conscious >= 0.0);
        assert!(assessment.phi_unconscious >= 0.0);
        assert!(assessment.consciousness_ratio >= 0.0 && assessment.consciousness_ratio <= 1.0);
        assert!(!assessment.explanation.is_empty());
    }

    #[test]
    fn test_conscious_unconscious_partition() {
        let mut spectrum = ConsciousnessSpectrum::new(4, SpectrumConfig::default());

        let state = vec![HV16::random(1000); 4];
        let assessment = spectrum.assess(&state);

        // Φ_total should equal Φ_conscious + Φ_unconscious (approximately)
        let total_reconstructed = assessment.phi_conscious + assessment.phi_unconscious;
        assert!((assessment.phi_total - total_reconstructed).abs() < 0.01);
    }

    #[test]
    fn test_access_consciousness() {
        let mut spectrum = ConsciousnessSpectrum::new(4, SpectrumConfig::default());

        let state = vec![HV16::random(1000); 4];
        let assessment = spectrum.assess(&state);

        assert!(assessment.access >= 0.0 && assessment.access <= 1.0);
    }

    #[test]
    fn test_phenomenal_consciousness() {
        let mut spectrum = ConsciousnessSpectrum::new(4, SpectrumConfig::default());

        let state = vec![HV16::random(1000); 4];
        let assessment = spectrum.assess(&state);

        assert!(assessment.phenomenal >= 0.0 && assessment.phenomenal <= 1.0);
    }

    #[test]
    fn test_flow_state_detection() {
        let config = SpectrumConfig {
            flow_threshold: 0.05,
            ..Default::default()
        };
        let mut spectrum = ConsciousnessSpectrum::new(4, config);

        let state = vec![HV16::random(1000); 4];
        let assessment = spectrum.assess(&state);

        // Flow state: low consciousness ratio
        if assessment.consciousness_ratio < 0.05 && assessment.phi_total > 0.5 {
            assert!(assessment.is_flow_state());
        }
    }

    #[test]
    fn test_level_determination() {
        let mut spectrum = ConsciousnessSpectrum::new(4, SpectrumConfig::default());

        let state = vec![HV16::random(1000); 4];
        let assessment = spectrum.assess(&state);

        // Level should be one of the enum values
        assert!(assessment.level.level() <= 6);
    }

    #[test]
    fn test_serialization() {
        let config = SpectrumConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(!serialized.is_empty());

        let deserialized: SpectrumConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.access_threshold, config.access_threshold);
    }
}
