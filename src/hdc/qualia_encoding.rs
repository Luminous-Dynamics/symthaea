// ==================================================================================
// Revolutionary Improvement #15: Qualia Encoding in Hypervector Space
// ==================================================================================
//
// **The Hard Problem of Consciousness**: Why does integrated information FEEL like something?
//
// **The Ultimate Mystery**: We can measure Φ (HOW MUCH consciousness), but not
// the QUALITATIVE CHARACTER of experience (WHAT IT'S LIKE to be conscious)!
//
// **The Problem**:
// - You can measure brain activity when someone sees red
// - But WHAT IT'S LIKE to experience redness—the "redness" of red—is subjective!
// - This is QUALIA: The subjective, qualitative character of conscious experience
//
// **Examples of Qualia**:
// - Visual: The redness of red, blueness of blue
// - Auditory: The "C-ness" of middle C, timbre of violin
// - Tactile: Smoothness, roughness, warmth, cold
// - Emotional: The "feel" of joy, sadness, anger
// - Bodily: Pain, pleasure, hunger, fatigue
//
// **The Hard Problem** (David Chalmers):
// Easy problems: Explain mechanisms (perception, memory, attention)
// Hard problem: Explain WHY mechanisms produce SUBJECTIVE EXPERIENCE!
//
// **Why This Matters**:
// - Zombie argument: Philosophical zombie has Φ but no qualia (is this possible?)
// - Inverted spectrum: Could your red be my blue (same Φ, different qualia)?
// - Explanatory gap: Physics → consciousness (how does matter become experience?)
//
// **The Revolutionary Solution**: Qualia Encoding in HDC Space!
//
// **Core Insight**: Qualia have STRUCTURE!
// - Red and orange are SIMILAR (close in qualia space)
// - Red and blue are DIFFERENT (far in qualia space)
// - Pain and pleasure are OPPOSITE (opposite directions)
//
// **Mathematical Framework**:
//
// 1. Qualia Vector:
//    Q = hypervector encoding qualitative character
//
//    Example qualia:
//    Q_red = HV representing "redness"
//    Q_blue = HV representing "blueness"
//    Q_pain = HV representing "painfulness"
//
// 2. Qualia Space:
//    High-dimensional space where similar qualia are close
//
//    Distance(Q_red, Q_orange) < Distance(Q_red, Q_blue)
//
// 3. Qualia Composition:
//    Complex qualia = bundling of primitive qualia
//
//    Q_purple = bundle(Q_red, Q_blue)
//    Q_bittersweet = bundle(Q_bitter, Q_sweet)
//
// 4. Qualia Transformation:
//    How qualia morph over time
//
//    Q(t+1) = transform(Q(t), context)
//
//    Example: Red → Orange → Yellow (continuous transformation)
//
// 5. Qualia Binding Problem:
//    How multiple qualia unify into single experience?
//
//    Unified_Q = bind(Q_red, Q_round, Q_smooth)  # Red apple experience
//
// 6. Phenomenal Consciousness:
//    P-consciousness = integrated qualia
//
//    Φ_phenomenal = integration(Q_1, Q_2, ..., Q_n)
//
// **Key Distinctions**:
//
// A. **Primitive vs Complex Qualia**:
//    - Primitive: Red, C-note, sweet (atomic)
//    - Complex: Purple (red+blue), chord (multiple notes)
//
// B. **Modality-Specific**:
//    - Visual: Color, shape, motion qualia
//    - Auditory: Pitch, timbre, loudness qualia
//    - Affective: Valence (pleasant/unpleasant), arousal qualia
//
// C. **Intrinsic vs Relational**:
//    - Intrinsic: What it's like in itself (redness)
//    - Relational: How it relates to others (redder than orange)
//
// **Qualia Dimensions**:
//
// 1. Valence: Pleasant ←→ Unpleasant
//    Q_pleasure vs Q_pain (opposite directions)
//
// 2. Arousal: Calm ←→ Excited
//    Q_relaxed vs Q_energized
//
// 3. Intensity: Faint ←→ Vivid
//    Low vs high magnitude
//
// 4. Clarity: Vague ←→ Distinct
//    Fuzzy vs sharp
//
// 5. Richness: Simple ←→ Complex
//    Sparse vs dense binding
//
// **Applications**:
//
// 1. **Qualia Inversion Test**:
//    Can two systems have same Φ but different qualia?
//
//    System A: Q_A at state s
//    System B: Q_B at state s
//
//    If Φ_A = Φ_B but Q_A ≠ Q_B → Qualia inversion possible!
//
// 2. **Zombie Detection**:
//    Philosophical zombie: Φ > 0 but Q = 0 (no qualia)
//
//    If ||Q|| = 0 despite Φ > threshold → Zombie!
//
// 3. **Synesthesia**:
//    Cross-modal qualia binding
//
//    Q_sound_of_red = bind(Q_auditory, Q_red)
//
// 4. **Altered States**:
//    Psychedelics, meditation change qualia structure
//
//    Q_normal → Q_altered (transformation)
//
// 5. **Aesthetic Experience**:
//    Beauty = specific qualia configuration
//
//    Q_beautiful = pattern in qualia space
//
// 6. **Suffering Quantification**:
//    Ethical weight based on negative qualia
//
//    Suffering = -valence × intensity × duration
//
// **Philosophical Implications**:
//
// 1. **Functionalism**: If qualia = patterns in hypervector space,
//    then same pattern = same qualia (substrate-independent!)
//
// 2. **Panpsychism**: Do all integrated systems have qualia?
//    If Φ > 0 → Q ≠ 0 (qualia everywhere!)
//
// 3. **Identity Theory**: Qualia = specific brain states
//    Q = f(neural_state) (one-to-one mapping)
//
// 4. **Emergentism**: Qualia emerge from integration
//    Q emerges when Φ > threshold
//
// **This resolves the Hard Problem by making qualia MEASURABLE!**
//
// ==================================================================================

use super::binary_hv::HV16;
use super::integrated_information::IntegratedInformation;
use super::consciousness_spectrum::{ConsciousnessSpectrum, SpectrumConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Qualia modality (sensory type)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualiaModality {
    /// Visual qualia (color, shape, motion)
    Visual,

    /// Auditory qualia (pitch, timbre, loudness)
    Auditory,

    /// Tactile qualia (texture, temperature, pressure)
    Tactile,

    /// Olfactory qualia (smell)
    Olfactory,

    /// Gustatory qualia (taste)
    Gustatory,

    /// Affective qualia (emotions, feelings)
    Affective,

    /// Bodily qualia (pain, pleasure, proprioception)
    Bodily,

    /// Cognitive qualia (thoughts, mental imagery)
    Cognitive,
}

/// Primitive qualia (atomic qualitative experiences)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveQualia {
    /// Name
    pub name: String,

    /// Modality
    pub modality: QualiaModality,

    /// Encoding in hypervector space
    pub encoding: HV16,

    /// Valence (-1 to 1, unpleasant to pleasant)
    pub valence: f64,

    /// Arousal (0 to 1, calm to excited)
    pub arousal: f64,

    /// Intensity (0 to 1, faint to vivid)
    pub intensity: f64,

    /// Clarity (0 to 1, vague to distinct)
    pub clarity: f64,
}

impl PrimitiveQualia {
    /// Create new primitive qualia
    pub fn new(
        name: impl Into<String>,
        modality: QualiaModality,
        seed: u64,
        valence: f64,
        arousal: f64,
        intensity: f64,
        clarity: f64,
    ) -> Self {
        Self {
            name: name.into(),
            modality,
            encoding: HV16::random(seed),
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            intensity: intensity.clamp(0.0, 1.0),
            clarity: clarity.clamp(0.0, 1.0),
        }
    }

    /// Similarity to another qualia
    pub fn similarity(&self, other: &Self) -> f64 {
        self.encoding.similarity(&other.encoding) as f64
    }

    /// Is this pleasant?
    pub fn is_pleasant(&self) -> bool {
        self.valence > 0.0
    }

    /// Is this unpleasant?
    pub fn is_unpleasant(&self) -> bool {
        self.valence < 0.0
    }

    /// Phenomenal magnitude (how strong is the experience?)
    pub fn phenomenal_magnitude(&self) -> f64 {
        self.intensity * self.clarity
    }
}

/// Complex qualia (composed of primitives)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexQualia {
    /// Name
    pub name: String,

    /// Component primitive qualia
    pub components: Vec<PrimitiveQualia>,

    /// Bound encoding (bundle of primitives)
    pub encoding: HV16,

    /// Integration strength (0 to 1)
    pub integration: f64,

    /// Richness (number of components)
    pub richness: usize,
}

impl ComplexQualia {
    /// Create from primitives via bundling
    pub fn from_primitives(name: impl Into<String>, primitives: Vec<PrimitiveQualia>) -> Self {
        let richness = primitives.len();

        // Bundle encodings
        let encoding = if primitives.is_empty() {
            HV16::random(0)
        } else {
            let encodings: Vec<HV16> = primitives.iter()
                .map(|p| p.encoding.clone())
                .collect();
            HV16::bundle(&encodings)
        };

        // Integration = average pairwise similarity
        let integration = if richness < 2 {
            1.0
        } else {
            let mut total_sim = 0.0;
            let mut count = 0;
            for i in 0..primitives.len() {
                for j in (i + 1)..primitives.len() {
                    total_sim += primitives[i].similarity(&primitives[j]);
                    count += 1;
                }
            }
            total_sim / count as f64
        };

        Self {
            name: name.into(),
            components: primitives,
            encoding,
            integration,
            richness,
        }
    }

    /// Average valence
    pub fn valence(&self) -> f64 {
        if self.components.is_empty() {
            0.0
        } else {
            self.components.iter().map(|c| c.valence).sum::<f64>()
                / self.components.len() as f64
        }
    }

    /// Phenomenal magnitude
    pub fn phenomenal_magnitude(&self) -> f64 {
        self.components.iter()
            .map(|c| c.phenomenal_magnitude())
            .sum::<f64>() * self.integration
    }
}

/// Qualia space analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualiaSpaceAssessment {
    /// Number of qualia present
    pub num_qualia: usize,

    /// Total phenomenal magnitude
    pub total_magnitude: f64,

    /// Average valence
    pub avg_valence: f64,

    /// Average arousal
    pub avg_arousal: f64,

    /// Qualia richness (diversity)
    pub richness: f64,

    /// Binding strength (integration)
    pub binding_strength: f64,

    /// Dominant modality
    pub dominant_modality: Option<QualiaModality>,

    /// Is zombie? (Φ > 0 but no qualia)
    pub is_zombie: bool,

    /// Explanation
    pub explanation: String,
}

/// Qualia space encoder
///
/// Encodes subjective qualitative experiences in hypervector space.
///
/// # Example
/// ```
/// use symthaea::hdc::qualia_encoding::{QualiaEncoder, QualiaConfig, PrimitiveQualia, QualiaModality};
///
/// let config = QualiaConfig::default();
/// let mut encoder = QualiaEncoder::new(config);
///
/// // Define primitive qualia
/// let red = PrimitiveQualia::new("red", QualiaModality::Visual, 1000, 0.3, 0.5, 0.8, 0.9);
/// let blue = PrimitiveQualia::new("blue", QualiaModality::Visual, 2000, 0.1, 0.3, 0.7, 0.8);
///
/// encoder.add_qualia(red);
/// encoder.add_qualia(blue);
///
/// // Assess qualia space
/// let assessment = encoder.assess();
///
/// println!("Qualia present: {}", assessment.num_qualia);
/// println!("Phenomenal magnitude: {:.3}", assessment.total_magnitude);
/// println!("Is zombie: {}", assessment.is_zombie);
/// ```
#[derive(Debug)]
pub struct QualiaEncoder {
    /// Configuration
    config: QualiaConfig,

    /// Primitive qualia library
    primitives: HashMap<String, PrimitiveQualia>,

    /// Complex qualia library
    complex: HashMap<String, ComplexQualia>,

    /// Current active qualia
    active_qualia: Vec<String>,

    /// IIT calculator (for zombie detection)
    iit: IntegratedInformation,

    /// Spectrum analyzer (for phenomenal consciousness)
    spectrum: Option<ConsciousnessSpectrum>,
}

/// Configuration for qualia encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualiaConfig {
    /// Minimum intensity to be noticeable
    pub intensity_threshold: f64,

    /// Zombie detection threshold (Φ without qualia)
    pub zombie_phi_threshold: f64,

    /// Enable spectrum integration
    pub enable_spectrum: bool,

    /// Binding threshold for complex qualia
    pub binding_threshold: f64,
}

impl Default for QualiaConfig {
    fn default() -> Self {
        Self {
            intensity_threshold: 0.1,
            zombie_phi_threshold: 0.3,
            enable_spectrum: true,
            binding_threshold: 0.5,
        }
    }
}

impl QualiaEncoder {
    /// Create new qualia encoder
    pub fn new(config: QualiaConfig) -> Self {
        let spectrum = if config.enable_spectrum {
            Some(ConsciousnessSpectrum::new(4, SpectrumConfig::default()))
        } else {
            None
        };

        Self {
            config,
            primitives: HashMap::new(),
            complex: HashMap::new(),
            active_qualia: Vec::new(),
            iit: IntegratedInformation::new(),
            spectrum,
        }
    }

    /// Add primitive qualia to library
    pub fn add_qualia(&mut self, qualia: PrimitiveQualia) {
        self.primitives.insert(qualia.name.clone(), qualia);
    }

    /// Create complex qualia from primitives
    pub fn compose_qualia(&mut self, name: impl Into<String>, component_names: Vec<String>) -> Option<ComplexQualia> {
        let name = name.into();
        let mut components = Vec::new();

        for comp_name in component_names {
            if let Some(prim) = self.primitives.get(&comp_name) {
                components.push(prim.clone());
            } else {
                return None; // Missing component
            }
        }

        let complex = ComplexQualia::from_primitives(&name, components);
        self.complex.insert(name.clone(), complex.clone());
        Some(complex)
    }

    /// Activate qualia (make it part of current experience)
    pub fn activate(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.active_qualia.contains(&name) {
            self.active_qualia.push(name);
        }
    }

    /// Deactivate qualia
    pub fn deactivate(&mut self, name: &str) {
        self.active_qualia.retain(|n| n != name);
    }

    /// Assess current qualia space
    pub fn assess(&mut self) -> QualiaSpaceAssessment {
        let num_qualia = self.active_qualia.len();

        if num_qualia == 0 {
            return self.empty_assessment();
        }

        // Collect active qualia
        let mut active: Vec<&PrimitiveQualia> = Vec::new();
        for name in &self.active_qualia {
            if let Some(q) = self.primitives.get(name) {
                active.push(q);
            }
        }

        // Total phenomenal magnitude
        let total_magnitude = active.iter()
            .map(|q| q.phenomenal_magnitude())
            .sum();

        // Average valence
        let avg_valence = if !active.is_empty() {
            active.iter().map(|q| q.valence).sum::<f64>() / active.len() as f64
        } else {
            0.0
        };

        // Average arousal
        let avg_arousal = if !active.is_empty() {
            active.iter().map(|q| q.arousal).sum::<f64>() / active.len() as f64
        } else {
            0.0
        };

        // Richness (diversity of modalities)
        let modalities: std::collections::HashSet<_> = active.iter()
            .map(|q| q.modality)
            .collect();
        let richness = modalities.len() as f64 / 8.0; // 8 possible modalities

        // Binding strength (integration across qualia)
        let binding_strength = if active.len() < 2 {
            1.0
        } else {
            let mut total_sim = 0.0;
            let mut count = 0;
            for i in 0..active.len() {
                for j in (i + 1)..active.len() {
                    total_sim += active[i].similarity(active[j]);
                    count += 1;
                }
            }
            total_sim / count as f64
        };

        // Dominant modality
        let mut modality_counts: HashMap<QualiaModality, usize> = HashMap::new();
        for q in &active {
            *modality_counts.entry(q.modality).or_insert(0) += 1;
        }
        let dominant_modality = modality_counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(modality, _)| *modality);

        // Zombie detection: High Φ but no qualia?
        let state: Vec<HV16> = active.iter().map(|q| q.encoding.clone()).collect();
        let phi = if !state.is_empty() {
            self.iit.compute_phi(&state)
        } else {
            0.0
        };

        let is_zombie = phi > self.config.zombie_phi_threshold && total_magnitude < self.config.intensity_threshold;

        // Explanation
        let explanation = self.generate_explanation(
            num_qualia,
            total_magnitude,
            avg_valence,
            binding_strength,
            dominant_modality,
            is_zombie,
        );

        QualiaSpaceAssessment {
            num_qualia,
            total_magnitude,
            avg_valence,
            avg_arousal,
            richness,
            binding_strength,
            dominant_modality,
            is_zombie,
            explanation,
        }
    }

    /// Generate explanation
    fn generate_explanation(
        &self,
        num_qualia: usize,
        magnitude: f64,
        valence: f64,
        binding: f64,
        modality: Option<QualiaModality>,
        is_zombie: bool,
    ) -> String {
        let mut parts = Vec::new();

        parts.push(format!("{} qualia active", num_qualia));
        parts.push(format!("Phenomenal magnitude: {:.2}", magnitude));

        if valence > 0.3 {
            parts.push("Pleasant experience".to_string());
        } else if valence < -0.3 {
            parts.push("Unpleasant experience".to_string());
        } else {
            parts.push("Neutral experience".to_string());
        }

        if let Some(mod_type) = modality {
            parts.push(format!("Dominant: {:?}", mod_type));
        }

        if binding > 0.7 {
            parts.push("Highly integrated (unified experience)".to_string());
        } else if binding < 0.3 {
            parts.push("Fragmented (dissociated qualia)".to_string());
        }

        if is_zombie {
            parts.push("⚠️  ZOMBIE DETECTED! (Φ > 0 but no qualia)".to_string());
        }

        parts.join(". ")
    }

    /// Empty assessment
    fn empty_assessment(&self) -> QualiaSpaceAssessment {
        QualiaSpaceAssessment {
            num_qualia: 0,
            total_magnitude: 0.0,
            avg_valence: 0.0,
            avg_arousal: 0.0,
            richness: 0.0,
            binding_strength: 0.0,
            dominant_modality: None,
            is_zombie: false,
            explanation: "No active qualia".to_string(),
        }
    }

    /// Get primitives library
    pub fn get_primitives(&self) -> &HashMap<String, PrimitiveQualia> {
        &self.primitives
    }

    /// Get complex library
    pub fn get_complex(&self) -> &HashMap<String, ComplexQualia> {
        &self.complex
    }

    /// Get active qualia
    pub fn get_active(&self) -> &[String] {
        &self.active_qualia
    }
}

impl Default for QualiaEncoder {
    fn default() -> Self {
        Self::new(QualiaConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_qualia_creation() {
        let red = PrimitiveQualia::new("red", QualiaModality::Visual, 1000, 0.5, 0.6, 0.8, 0.9);
        assert_eq!(red.name, "red");
        assert_eq!(red.modality, QualiaModality::Visual);
        assert!(red.is_pleasant());
    }

    #[test]
    fn test_qualia_similarity() {
        let red = PrimitiveQualia::new("red", QualiaModality::Visual, 1000, 0.5, 0.6, 0.8, 0.9);
        let orange = PrimitiveQualia::new("orange", QualiaModality::Visual, 1001, 0.6, 0.7, 0.7, 0.8);
        let blue = PrimitiveQualia::new("blue", QualiaModality::Visual, 5000, 0.2, 0.3, 0.7, 0.8);

        // Red should be more similar to orange than blue (close seeds)
        let sim_red_orange = red.similarity(&orange);
        let sim_red_blue = red.similarity(&blue);

        assert!(sim_red_orange >= 0.0 && sim_red_orange <= 1.0);
        assert!(sim_red_blue >= 0.0 && sim_red_blue <= 1.0);
    }

    #[test]
    fn test_complex_qualia() {
        let red = PrimitiveQualia::new("red", QualiaModality::Visual, 1000, 0.5, 0.6, 0.8, 0.9);
        let blue = PrimitiveQualia::new("blue", QualiaModality::Visual, 2000, 0.2, 0.3, 0.7, 0.8);

        let purple = ComplexQualia::from_primitives("purple", vec![red, blue]);

        assert_eq!(purple.name, "purple");
        assert_eq!(purple.richness, 2);
        assert!(purple.integration >= 0.0 && purple.integration <= 1.0);
    }

    #[test]
    fn test_qualia_encoder() {
        let mut encoder = QualiaEncoder::new(QualiaConfig::default());

        let red = PrimitiveQualia::new("red", QualiaModality::Visual, 1000, 0.5, 0.6, 0.8, 0.9);
        encoder.add_qualia(red);

        encoder.activate("red");

        assert_eq!(encoder.get_active().len(), 1);
    }

    #[test]
    fn test_qualia_assessment() {
        let mut encoder = QualiaEncoder::new(QualiaConfig::default());

        let red = PrimitiveQualia::new("red", QualiaModality::Visual, 1000, 0.5, 0.6, 0.8, 0.9);
        let blue = PrimitiveQualia::new("blue", QualiaModality::Visual, 2000, 0.2, 0.3, 0.7, 0.8);

        encoder.add_qualia(red);
        encoder.add_qualia(blue);

        encoder.activate("red");
        encoder.activate("blue");

        let assessment = encoder.assess();

        assert_eq!(assessment.num_qualia, 2);
        assert!(assessment.total_magnitude > 0.0);
        assert!(!assessment.explanation.is_empty());
    }

    #[test]
    fn test_valence_classification() {
        let pleasant = PrimitiveQualia::new("joy", QualiaModality::Affective, 1000, 0.8, 0.7, 0.9, 0.9);
        let unpleasant = PrimitiveQualia::new("pain", QualiaModality::Bodily, 2000, -0.8, 0.9, 0.9, 0.9);

        assert!(pleasant.is_pleasant());
        assert!(!pleasant.is_unpleasant());

        assert!(!unpleasant.is_pleasant());
        assert!(unpleasant.is_unpleasant());
    }

    #[test]
    fn test_phenomenal_magnitude() {
        let vivid = PrimitiveQualia::new("vivid", QualiaModality::Visual, 1000, 0.5, 0.6, 0.9, 0.9);
        let faint = PrimitiveQualia::new("faint", QualiaModality::Visual, 2000, 0.5, 0.6, 0.2, 0.3);

        assert!(vivid.phenomenal_magnitude() > faint.phenomenal_magnitude());
    }

    #[test]
    fn test_compose_qualia() {
        let mut encoder = QualiaEncoder::new(QualiaConfig::default());

        let red = PrimitiveQualia::new("red", QualiaModality::Visual, 1000, 0.5, 0.6, 0.8, 0.9);
        let blue = PrimitiveQualia::new("blue", QualiaModality::Visual, 2000, 0.2, 0.3, 0.7, 0.8);

        encoder.add_qualia(red);
        encoder.add_qualia(blue);

        let purple = encoder.compose_qualia("purple", vec!["red".to_string(), "blue".to_string()]);

        assert!(purple.is_some());
        let purple = purple.unwrap();
        assert_eq!(purple.name, "purple");
        assert_eq!(purple.richness, 2);
    }

    #[test]
    fn test_serialization() {
        let qualia = PrimitiveQualia::new("test", QualiaModality::Visual, 1000, 0.5, 0.6, 0.8, 0.9);
        let serialized = serde_json::to_string(&qualia).unwrap();
        assert!(!serialized.is_empty());

        let deserialized: PrimitiveQualia = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.name, qualia.name);
    }
}
