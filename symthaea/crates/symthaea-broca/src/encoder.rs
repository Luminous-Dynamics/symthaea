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
#[cfg(not(feature = "therapeutic"))]
pub const NUM_CHANNELS: usize = 24;
#[cfg(feature = "therapeutic")]
pub const NUM_CHANNELS: usize = 28;

/// Number of channels in legacy (v2) training data format.
pub const LEGACY_NUM_CHANNELS: usize = 20;

/// Number of thermometer coding levels.
const DEFAULT_NUM_LEVELS: usize = 32;

/// Channel names for genesis seeding.
#[cfg(not(feature = "therapeutic"))]
const CHANNEL_NAMES: [&str; NUM_CHANNELS] = [
    "intent_acknowledge",  // 0
    "intent_answer",       // 1
    "intent_clarify",      // 2
    "intent_propose",      // 3
    "intent_uncertainty",  // 4
    "intent_reflect",      // 5
    "intent_continue",     // 6
    "intent_unknown",      // 7
    "epistemic_status",    // 8
    "valence",             // 9
    "arousal",             // 10
    "warmth",              // 11
    "psi",                 // 12
    "meta_awareness",      // 13
    "coherence",           // 14
    "relationship_stage",  // 15
    "trust",               // 16
    "mood_temperature",    // 17
    "has_computed_answer", // 18
    "concept_count",       // 19
    // New channels (v3)
    "time_pressure",       // 20: urgency/deadline proximity (0=relaxed, 1=urgent)
    "domain_familiarity",  // 21: how well-known the topic is (0=novel, 1=expert)
    "social_context",      // 22: formality of interaction (0=intimate, 1=formal)
    "response_confidence", // 23: confidence in the generated response (0=unsure, 1=confident)
];

/// Channel names for genesis seeding (therapeutic variant with 4 extra channels).
#[cfg(feature = "therapeutic")]
const CHANNEL_NAMES: [&str; NUM_CHANNELS] = [
    "intent_acknowledge",  // 0
    "intent_answer",       // 1
    "intent_clarify",      // 2
    "intent_propose",      // 3
    "intent_uncertainty",  // 4
    "intent_reflect",      // 5
    "intent_continue",     // 6
    "intent_unknown",      // 7
    "epistemic_status",    // 8
    "valence",             // 9
    "arousal",             // 10
    "warmth",              // 11
    "psi",                 // 12
    "meta_awareness",      // 13
    "coherence",           // 14
    "relationship_stage",  // 15
    "trust",               // 16
    "mood_temperature",    // 17
    "has_computed_answer", // 18
    "concept_count",       // 19
    // New channels (v3)
    "time_pressure",       // 20
    "domain_familiarity",  // 21
    "social_context",      // 22
    "response_confidence", // 23
    // Therapeutic channels (v4)
    "therapeutic_intent",      // 24
    "alliance_quality",        // 25
    "client_distress_level",   // 26
    "intervention_depth",      // 27
];

/// Channel ranges [min, max] for normalization to [0, 1].
#[cfg(not(feature = "therapeutic"))]
const CHANNEL_RANGES: [[f32; 2]; NUM_CHANNELS] = [
    [0.0, 1.0],  // intent_acknowledge
    [0.0, 1.0],  // intent_answer
    [0.0, 1.0],  // intent_clarify
    [0.0, 1.0],  // intent_propose
    [0.0, 1.0],  // intent_uncertainty
    [0.0, 1.0],  // intent_reflect
    [0.0, 1.0],  // intent_continue
    [0.0, 1.0],  // intent_unknown
    [0.0, 4.0],  // epistemic_status (ordinal: 0=Certain..4=OutOfDomain)
    [-1.0, 1.0], // valence
    [0.0, 1.0],  // arousal
    [0.0, 1.0],  // warmth
    [0.0, 1.0],  // psi
    [0.0, 1.0],  // meta_awareness
    [0.0, 1.0],  // coherence
    [0.0, 6.0],  // relationship_stage (ordinal: 0-6)
    [0.0, 1.0],  // trust
    [0.5, 2.0],  // mood_temperature
    [0.0, 1.0],  // has_computed_answer
    [0.0, 10.0], // concept_count
    // New channels (v3)
    [0.0, 1.0], // time_pressure
    [0.0, 1.0], // domain_familiarity
    [0.0, 1.0], // social_context
    [0.0, 1.0], // response_confidence
];

/// Channel ranges [min, max] for normalization to [0, 1] (therapeutic variant).
#[cfg(feature = "therapeutic")]
const CHANNEL_RANGES: [[f32; 2]; NUM_CHANNELS] = [
    [0.0, 1.0],  // intent_acknowledge
    [0.0, 1.0],  // intent_answer
    [0.0, 1.0],  // intent_clarify
    [0.0, 1.0],  // intent_propose
    [0.0, 1.0],  // intent_uncertainty
    [0.0, 1.0],  // intent_reflect
    [0.0, 1.0],  // intent_continue
    [0.0, 1.0],  // intent_unknown
    [0.0, 4.0],  // epistemic_status
    [-1.0, 1.0], // valence
    [0.0, 1.0],  // arousal
    [0.0, 1.0],  // warmth
    [0.0, 1.0],  // psi
    [0.0, 1.0],  // meta_awareness
    [0.0, 1.0],  // coherence
    [0.0, 6.0],  // relationship_stage
    [0.0, 1.0],  // trust
    [0.5, 2.0],  // mood_temperature
    [0.0, 1.0],  // has_computed_answer
    [0.0, 10.0], // concept_count
    // New channels (v3)
    [0.0, 1.0], // time_pressure
    [0.0, 1.0], // domain_familiarity
    [0.0, 1.0], // social_context
    [0.0, 1.0], // response_confidence
    // Therapeutic channels (v4)
    [0.0, 7.0], // therapeutic_intent
    [0.0, 1.0], // alliance_quality
    [0.0, 1.0], // client_distress_level
    [0.0, 1.0], // intervention_depth
];

/// Default values for the 4 new channels (used when loading legacy 20-channel data).
pub const NEW_CHANNEL_DEFAULTS: [f32; 4] = [
    0.0, // time_pressure: relaxed
    0.5, // domain_familiarity: mid
    0.5, // social_context: mid
    0.5, // response_confidence: mid
];

/// Decoupled thought state: 20 scalar channels extracted from StructuredThought.
///
/// Conversion from `StructuredThought -> ThoughtChannels` happens at the
/// integration layer (Phase 3), avoiding circular dependency.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ThoughtChannels {
    pub channels: [f32; NUM_CHANNELS],
}

#[cfg(not(feature = "therapeutic"))]
impl Default for ThoughtChannels {
    fn default() -> Self {
        Self {
            channels: [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // intent: Unknown
                1.0, // epistemic: Probable (not Unknown — avoids overly pessimistic default)
                0.0, // valence: neutral
                0.5, // arousal: mid
                0.5, // warmth: mid
                0.5, // psi: mid
                0.5, // meta_awareness: mid
                0.5, // coherence: mid
                0.0, // relationship_stage: 0
                0.5, // trust: mid
                1.0, // mood_temperature: neutral
                0.0, // has_computed_answer: false
                0.0, // concept_count: none
                // New channels (v3)
                0.0, // time_pressure: relaxed
                0.5, // domain_familiarity: mid
                0.5, // social_context: mid
                0.5, // response_confidence: mid
            ],
        }
    }
}

#[cfg(feature = "therapeutic")]
impl Default for ThoughtChannels {
    fn default() -> Self {
        Self {
            channels: [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // intent: Unknown
                1.0, // epistemic: Probable
                0.0, // valence: neutral
                0.5, // arousal: mid
                0.5, // warmth: mid
                0.5, // psi: mid
                0.5, // meta_awareness: mid
                0.5, // coherence: mid
                0.0, // relationship_stage: 0
                0.5, // trust: mid
                1.0, // mood_temperature: neutral
                0.0, // has_computed_answer: false
                0.0, // concept_count: none
                // New channels (v3)
                0.0, // time_pressure: relaxed
                0.5, // domain_familiarity: mid
                0.5, // social_context: mid
                0.5, // response_confidence: mid
                // Therapeutic channels (v4)
                0.0, // therapeutic_intent
                0.5, // alliance_quality
                0.0, // client_distress_level
                0.0, // intervention_depth
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

    /// Set contextual channels (new v3 channels).
    pub fn set_context(
        &mut self,
        time_pressure: f32,
        domain_familiarity: f32,
        social_context: f32,
        response_confidence: f32,
    ) {
        self.channels[20] = time_pressure.clamp(0.0, 1.0);
        self.channels[21] = domain_familiarity.clamp(0.0, 1.0);
        self.channels[22] = social_context.clamp(0.0, 1.0);
        self.channels[23] = response_confidence.clamp(0.0, 1.0);
    }

    /// Get time pressure (0=relaxed, 1=urgent).
    pub fn time_pressure(&self) -> f32 {
        self.channels[20]
    }

    /// Get domain familiarity (0=novel, 1=expert).
    pub fn domain_familiarity(&self) -> f32 {
        self.channels[21]
    }

    /// Get social context (0=intimate, 1=formal).
    pub fn social_context(&self) -> f32 {
        self.channels[22]
    }

    /// Get response confidence (0=unsure, 1=confident).
    pub fn response_confidence(&self) -> f32 {
        self.channels[23]
    }

    /// Construct from a legacy 20-channel array, padding new channels with defaults.
    pub fn from_legacy(legacy: &[f32; LEGACY_NUM_CHANNELS]) -> Self {
        let mut tc = Self::default();
        tc.channels[..LEGACY_NUM_CHANNELS].copy_from_slice(legacy);
        tc.channels[20] = NEW_CHANNEL_DEFAULTS[0];
        tc.channels[21] = NEW_CHANNEL_DEFAULTS[1];
        tc.channels[22] = NEW_CHANNEL_DEFAULTS[2];
        tc.channels[23] = NEW_CHANNEL_DEFAULTS[3];
        #[cfg(feature = "therapeutic")]
        {
            tc.channels[24] = THERAPEUTIC_CHANNEL_DEFAULTS[0];
            tc.channels[25] = THERAPEUTIC_CHANNEL_DEFAULTS[1];
            tc.channels[26] = THERAPEUTIC_CHANNEL_DEFAULTS[2];
            tc.channels[27] = THERAPEUTIC_CHANNEL_DEFAULTS[3];
        }
        tc
    }

    #[cfg(feature = "therapeutic")]
    pub fn set_therapeutic(&mut self, intent: f32, alliance: f32, distress: f32, depth: f32) {
        self.channels[24] = intent;
        self.channels[25] = alliance;
        self.channels[26] = distress;
        self.channels[27] = depth;
    }

    #[cfg(feature = "therapeutic")]
    pub fn therapeutic_intent(&self) -> f32 { self.channels[24] }

    #[cfg(feature = "therapeutic")]
    pub fn alliance_quality(&self) -> f32 { self.channels[25] }

    #[cfg(feature = "therapeutic")]
    pub fn client_distress_level(&self) -> f32 { self.channels[26] }

    #[cfg(feature = "therapeutic")]
    pub fn intervention_depth(&self) -> f32 { self.channels[27] }
}

#[cfg(feature = "therapeutic")]
pub const THERAPEUTIC_CHANNEL_NAMES: &[&str] = &[
    "therapeutic_intent",
    "alliance_quality",
    "client_distress_level",
    "intervention_depth",
];

#[cfg(feature = "therapeutic")]
pub const THERAPEUTIC_CHANNEL_RANGES: &[[f32; 2]] = &[
    [0.0, 7.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [0.0, 1.0],
];

#[cfg(feature = "therapeutic")]
pub const THERAPEUTIC_CHANNEL_DEFAULTS: &[f32] = &[0.0, 0.5, 0.0, 0.0];

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
                ContinuousHV::from_genesis(genesis, &format!("broca::level::{i}"), HDC_DIMENSION)
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
    ///
    /// NaN/Inf channel values are replaced with sensible defaults (mid-range)
    /// to prevent garbage propagation through the HDC pipeline.
    pub fn encode(&self, channels: &ThoughtChannels) -> ContinuousHV {
        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(NUM_CHANNELS);

        for (i, (&raw, base)) in channels
            .channels
            .iter()
            .zip(self.base_vectors.iter())
            .enumerate()
        {
            let value = if raw.is_finite() {
                raw
            } else {
                // Default to midpoint of the channel's range
                let [min, max] = CHANNEL_RANGES[i];
                (min + max) * 0.5
            };
            let normalized = Self::normalize_channel(i, value);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(base.bind(&level_hv));
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
        assert!(sim < 1.0 - 1e-4, "Should not be identical: sim={sim}");
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

    #[test]
    fn test_nan_inf_channels_produce_finite_output() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);

        // All NaN channels
        let mut channels = ThoughtChannels::default();
        for c in &mut channels.channels {
            *c = f32::NAN;
        }
        let hv = enc.encode(&channels);
        assert_eq!(hv.dim(), HDC_DIMENSION);
        assert!(
            hv.as_slice().iter().all(|v| v.is_finite()),
            "NaN input channels should produce finite HV output"
        );

        // All Inf channels
        for c in &mut channels.channels {
            *c = f32::INFINITY;
        }
        let hv = enc.encode(&channels);
        assert!(
            hv.as_slice().iter().all(|v| v.is_finite()),
            "Inf input channels should produce finite HV output"
        );
    }

    #[test]
    fn test_nan_channels_use_midpoint_defaults() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);

        // NaN channels should produce same encoding as midpoint values
        let mut nan_channels = ThoughtChannels::default();
        nan_channels.channels[9] = f32::NAN; // valence (range [-1, 1], midpoint = 0.0)

        let mut mid_channels = ThoughtChannels::default();
        mid_channels.channels[9] = 0.0; // explicit midpoint

        let hv_nan = enc.encode(&nan_channels);
        let hv_mid = enc.encode(&mid_channels);

        let sim = hv_nan.similarity(&hv_mid);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "NaN channel should fall back to midpoint: sim={sim}"
        );
    }

    // ── Channel Integration Tests ────────────────────────────────────────

    #[test]
    fn test_all_24_channels_produce_distinct_encodings() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);
        let baseline = enc.encode(&ThoughtChannels::default());
        let mut distinct = 0;
        for i in 0..NUM_CHANNELS {
            let mut ch = ThoughtChannels::default();
            ch.channels[i] = CHANNEL_RANGES[i][1];
            let sim = baseline.similarity(&enc.encode(&ch));
            if (sim - 1.0).abs() > 1e-4 { distinct += 1; }
        }
        assert!(distinct >= 20, "At least 20/24 channels should be distinct, got {distinct}");
    }

    #[test]
    fn test_out_of_range_channels_clamped() {
        let enc = ThoughtLanguageEncoder::new(&test_genesis());
        let mut ch = ThoughtChannels::default();
        ch.channels[8] = 100.0; ch.channels[9] = -50.0; ch.channels[10] = 999.0;
        assert!(enc.encode(&ch).as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_legacy_20_channel_conversion() {
        let mut legacy = [0.0f32; 20];
        legacy[0] = 1.0; legacy[9] = -0.5; legacy[12] = 0.8;
        let tc = ThoughtChannels::from_legacy(&legacy);
        assert_eq!(tc.channels[0], 1.0);
        assert_eq!(tc.channels[9], -0.5);
        assert_eq!(tc.channels[20], NEW_CHANNEL_DEFAULTS[0]);
        assert_eq!(tc.channels[23], NEW_CHANNEL_DEFAULTS[3]);
    }

    #[test]
    fn test_consciousness_channels_affect_encoding() {
        let enc = ThoughtLanguageEncoder::new(&test_genesis());
        let mut low = ThoughtChannels::default();
        low.set_consciousness(0.0, 0.0, 0.0);
        let mut high = ThoughtChannels::default();
        high.set_consciousness(1.0, 1.0, 1.0);
        let sim = enc.encode(&low).similarity(&enc.encode(&high));
        assert!(sim < 0.95, "Low vs high consciousness should differ, sim={sim}");
    }

    #[test]
    fn test_emotion_channels_affect_encoding() {
        let enc = ThoughtLanguageEncoder::new(&test_genesis());
        let mut neg = ThoughtChannels::default();
        neg.set_emotion(-1.0, 1.0, 0.0);
        let mut pos = ThoughtChannels::default();
        pos.set_emotion(1.0, 0.0, 1.0);
        let sim = enc.encode(&neg).similarity(&enc.encode(&pos));
        assert!(sim < 0.95, "Opposite emotions should differ, sim={sim}");
    }
}
