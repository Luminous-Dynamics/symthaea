//! Thought-to-HDC encoder: 20 scalar channels → ContinuousHV (16,384D).
//!
//! Follows the `VocalTractHdcEncoder` pattern from `crates/symthaea-vocal-tract/src/encoder.rs`.
//! Each of 20 thought channels gets a genesis-seeded base vector. Values are
//! level-encoded (thermometer coding) then bound with the base vector. The result is
//! bundled into a single 16,384D ContinuousHV.
//!
//! # 20-Channel Thought State
//!
//! | # | Channel               | Range      | Source                         |
//! |---|-----------------------|------------|--------------------------------|
//! | 0-7 | semantic_intent     | [0,1]      | SemanticIntent one-hot (8 var) |
//! | 8  | epistemic_status     | [0,4]      | EpistemicStatus ordinal        |
//! | 9  | valence              | [-1,1]     | EmotionalTone                  |
//! | 10 | arousal              | [0,1]      | EmotionalTone                  |
//! | 11 | warmth               | [0,1]      | EmotionalTone                  |
//! | 12 | psi                  | [0,1]      | Consciousness metric           |
//! | 13 | meta_awareness       | [0,1]      | Consciousness metric           |
//! | 14 | coherence            | [0,1]      | Consciousness metric           |
//! | 15 | relationship_stage   | [0,6]      | Relational context             |
//! | 16 | trust                | [0,1]      | Relational context             |
//! | 17 | mood_temperature     | [0.5,2.0]  | Generation control             |
//! | 18 | has_computed_answer   | [0,1]      | Boolean indicator              |
//! | 19 | concept_count        | [0,10]     | Number of activated concepts   |

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

/// Number of thought channels.
pub const NUM_CHANNELS: usize = 20;

/// Number of thermometer coding levels.
const DEFAULT_NUM_LEVELS: usize = 32;

/// Channel names for genesis seeding.
const CHANNEL_NAMES: [&str; NUM_CHANNELS] = [
    "intent_acknowledge",
    "intent_answer",
    "intent_clarify",
    "intent_propose",
    "intent_uncertainty",
    "intent_reflect",
    "intent_continue",
    "intent_unknown",
    "epistemic_status",
    "valence",
    "arousal",
    "warmth",
    "psi",
    "meta_awareness",
    "coherence",
    "relationship_stage",
    "trust",
    "mood_temperature",
    "has_computed_answer",
    "concept_count",
];

/// Channel ranges [min, max] for normalization to [0, 1].
const CHANNEL_RANGES: [[f32; 2]; NUM_CHANNELS] = [
    [0.0, 1.0],   // intent_acknowledge
    [0.0, 1.0],   // intent_answer
    [0.0, 1.0],   // intent_clarify
    [0.0, 1.0],   // intent_propose
    [0.0, 1.0],   // intent_uncertainty
    [0.0, 1.0],   // intent_reflect
    [0.0, 1.0],   // intent_continue
    [0.0, 1.0],   // intent_unknown
    [0.0, 4.0],   // epistemic_status (ordinal: 0=Certain..4=OutOfDomain)
    [-1.0, 1.0],  // valence
    [0.0, 1.0],   // arousal
    [0.0, 1.0],   // warmth
    [0.0, 1.0],   // psi
    [0.0, 1.0],   // meta_awareness
    [0.0, 1.0],   // coherence
    [0.0, 6.0],   // relationship_stage (ordinal: 0-6)
    [0.0, 1.0],   // trust
    [0.5, 2.0],   // mood_temperature
    [0.0, 1.0],   // has_computed_answer
    [0.0, 10.0],  // concept_count
];

/// Decoupled thought state: 20 scalar channels extracted from StructuredThought.
///
/// Conversion from `StructuredThought -> ThoughtChannels` happens at the
/// integration layer (Phase 3), avoiding circular dependency.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ThoughtChannels {
    pub channels: [f32; NUM_CHANNELS],
}

impl Default for ThoughtChannels {
    fn default() -> Self {
        Self {
            channels: [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // intent: Unknown
                3.0,   // epistemic: Unknown
                0.0,   // valence: neutral
                0.5,   // arousal: mid
                0.5,   // warmth: mid
                0.5,   // psi: mid
                0.5,   // meta_awareness: mid
                0.5,   // coherence: mid
                0.0,   // relationship_stage: 0
                0.5,   // trust: mid
                1.0,   // mood_temperature: neutral
                0.0,   // has_computed_answer: false
                0.0,   // concept_count: none
            ],
        }
    }
}

impl ThoughtChannels {
    /// Create channels with a specific semantic intent one-hot.
    pub fn with_intent(intent_index: usize) -> Self {
        let mut channels = Self::default();
        // Clear all intent channels
        for i in 0..8 {
            channels.channels[i] = 0.0;
        }
        // Set the active intent
        if intent_index < 8 {
            channels.channels[intent_index] = 1.0;
        }
        channels
    }

    /// Set epistemic status (0=Certain, 1=Probable, 2=Uncertain, 3=Unknown, 4=OutOfDomain).
    pub fn set_epistemic(&mut self, ordinal: f32) {
        self.channels[8] = ordinal.clamp(0.0, 4.0);
    }

    /// Set emotional tone.
    pub fn set_emotion(&mut self, valence: f32, arousal: f32, warmth: f32) {
        self.channels[9] = valence.clamp(-1.0, 1.0);
        self.channels[10] = arousal.clamp(0.0, 1.0);
        self.channels[11] = warmth.clamp(0.0, 1.0);
    }

    /// Set consciousness metrics.
    pub fn set_consciousness(&mut self, psi: f32, meta_awareness: f32, coherence: f32) {
        self.channels[12] = psi.clamp(0.0, 1.0);
        self.channels[13] = meta_awareness.clamp(0.0, 1.0);
        self.channels[14] = coherence.clamp(0.0, 1.0);
    }

    /// Get epistemic status as ordinal.
    pub fn epistemic_ordinal(&self) -> f32 {
        self.channels[8]
    }

    /// Get psi (consciousness level).
    pub fn psi(&self) -> f32 {
        self.channels[12]
    }

    /// Get arousal.
    pub fn arousal(&self) -> f32 {
        self.channels[10]
    }

    /// Get warmth.
    pub fn warmth(&self) -> f32 {
        self.channels[11]
    }

    /// Get valence.
    pub fn valence(&self) -> f32 {
        self.channels[9]
    }

    /// Get coherence.
    pub fn coherence(&self) -> f32 {
        self.channels[14]
    }
}

/// HDC encoder for thought channels.
///
/// Encodes a 20D `ThoughtChannels` into a full 16,384D `ContinuousHV` via:
/// 1. Per-channel normalization to [0, 1] using known ranges
/// 2. Level encoding (thermometer coding via bundled levels 0..k)
/// 3. Binding with genesis-seeded channel base vectors
/// 4. Bundling all bound channel HVs
pub struct ThoughtLanguageEncoder {
    /// Base vectors for each of 20 channels.
    base_vectors: Vec<ContinuousHV>,
    /// Level codebook (num_levels entries).
    level_vectors: Vec<ContinuousHV>,
    /// Number of levels in the codebook.
    num_levels: usize,
}

impl ThoughtLanguageEncoder {
    /// Create a new encoder from a genesis seed.
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self::with_levels(genesis, DEFAULT_NUM_LEVELS)
    }

    /// Create with a custom number of thermometer levels.
    pub fn with_levels(genesis: &GenesisSeed, num_levels: usize) -> Self {
        let base_vectors: Vec<ContinuousHV> = CHANNEL_NAMES
            .iter()
            .map(|name| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::channel::{name}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        let level_vectors: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| {
                ContinuousHV::from_genesis(
                    genesis,
                    &format!("broca::level::{i}"),
                    HDC_DIMENSION,
                )
            })
            .collect();

        Self {
            base_vectors,
            level_vectors,
            num_levels,
        }
    }

    /// Normalize a raw channel value to [0, 1] using known ranges.
    pub fn normalize_channel(channel: usize, value: f32) -> f32 {
        let [min, max] = CHANNEL_RANGES[channel];
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Level-encode a normalized [0,1] value using thermometer coding.
    fn encode_level(&self, normalized: f32) -> ContinuousHV {
        let k = ((normalized * self.num_levels as f32) as usize).min(self.num_levels - 1);
        if k == 0 {
            self.level_vectors[0].clone()
        } else {
            let refs: Vec<&ContinuousHV> = self.level_vectors[..=k].iter().collect();
            ContinuousHV::bundle(&refs)
        }
    }

    /// Encode thought channels into a full 16,384D ContinuousHV.
    pub fn encode(&self, channels: &ThoughtChannels) -> ContinuousHV {
        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(NUM_CHANNELS);

        for i in 0..NUM_CHANNELS {
            let normalized = Self::normalize_channel(i, channels.channels[i]);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(self.base_vectors[i].bind(&level_hv));
        }

        let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Number of levels in the codebook.
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-broca-encoder")
    }

    #[test]
    fn test_encoder_determinism() {
        let genesis = test_genesis();
        let enc1 = ThoughtLanguageEncoder::new(&genesis);
        let enc2 = ThoughtLanguageEncoder::new(&genesis);

        let channels = ThoughtChannels::default();
        let hv1 = enc1.encode(&channels);
        let hv2 = enc2.encode(&channels);

        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "Same genesis -> identical encoding"
        );
    }

    #[test]
    fn test_intent_discrimination() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);

        let answer = ThoughtChannels::with_intent(1); // Answer
        let clarify = ThoughtChannels::with_intent(2); // Clarify

        let hv_answer = enc.encode(&answer);
        let hv_clarify = enc.encode(&clarify);

        let sim = hv_answer.similarity(&hv_clarify);
        assert!(
            sim < 0.95,
            "Different intents should produce dissimilar HVs: sim={sim}"
        );
        assert!(sim > 0.0, "Should share structure: sim={sim}");
    }

    #[test]
    fn test_channel_sensitivity() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);

        let calm = {
            let mut c = ThoughtChannels::default();
            c.set_emotion(0.2, 0.1, 0.3);
            c
        };
        let excited = {
            let mut c = ThoughtChannels::default();
            c.set_emotion(0.8, 0.9, 0.8);
            c
        };

        let hv_calm = enc.encode(&calm);
        let hv_excited = enc.encode(&excited);

        let sim = hv_calm.similarity(&hv_excited);
        // With 20 channels and only 3 differing (valence, arousal, warmth),
        // the 17 shared channels dominate — expect similarity ~0.97-0.99.
        assert!(
            sim < 0.995,
            "Different emotional states should produce different HVs: sim={sim}"
        );
        assert!(
            sim < 1.0 - 1e-4,
            "Should not be identical: sim={sim}"
        );
    }

    #[test]
    fn test_normalization() {
        assert!((ThoughtLanguageEncoder::normalize_channel(9, -1.0) - 0.0).abs() < 1e-6);
        assert!((ThoughtLanguageEncoder::normalize_channel(9, 1.0) - 1.0).abs() < 1e-6);
        assert!((ThoughtLanguageEncoder::normalize_channel(9, 0.0) - 0.5).abs() < 1e-6);
        assert!((ThoughtLanguageEncoder::normalize_channel(8, 0.0) - 0.0).abs() < 1e-6);
        assert!((ThoughtLanguageEncoder::normalize_channel(8, 4.0) - 1.0).abs() < 1e-6);
        // Clamped above max
        assert!((ThoughtLanguageEncoder::normalize_channel(9, 5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_epistemic_separation() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);

        let certain = {
            let mut c = ThoughtChannels::default();
            c.set_epistemic(0.0); // Certain
            c
        };
        let unknown = {
            let mut c = ThoughtChannels::default();
            c.set_epistemic(3.0); // Unknown
            c
        };

        let hv_certain = enc.encode(&certain);
        let hv_unknown = enc.encode(&unknown);

        let sim = hv_certain.similarity(&hv_unknown);
        assert!(
            sim < 0.99,
            "Different epistemic states should differ: sim={sim}"
        );
    }

    #[test]
    fn test_output_dimension() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);
        let channels = ThoughtChannels::default();
        let hv = enc.encode(&channels);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }
}
