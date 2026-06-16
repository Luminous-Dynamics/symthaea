// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thought-to-HDC encoder: scalar channels → ContinuousHV (16,384D).
//!
//! Follows the `VocalTractHdcEncoder` pattern from `crates/symthaea-vocal-tract/src/encoder.rs`.
//! Each thought channel gets a genesis-seeded base vector. Values are
//! level-encoded (thermometer coding) then bound with the base vector. The result is
//! bundled into a single 16,384D ContinuousHV.
//!
//! # Channel Layout
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
//! | 20-23 | v3 context        | [0,1]      | Time/domain/social/confidence  |
//! | 24-27 | v5 code           | [0,1]      | Syntax/types/algo/error        |
//! | 28-32 | e_tier (one-hot)  | [0,1]      | Epistemic Cube: empirical E0-E4 |
//! | 33-36 | n_tier (one-hot)  | [0,1]      | Epistemic Cube: normative N0-N3 |
//! | 37-40 | m_tier (one-hot)  | [0,1]      | Epistemic Cube: materiality M0-M3 |
//! | 41 | h_tier               | [0,1]      | Epistemic Cube: harmonic (scalar) |
//! | 42 | epistemic_quality    | [0,1]      | Cube quality score E×0.4+N×0.35+M×0.25 |

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

/// Number of thought channels.
#[cfg(not(feature = "therapeutic"))]
pub const NUM_CHANNELS: usize = 43;
#[cfg(feature = "therapeutic")]
pub const NUM_CHANNELS: usize = 47;

/// Number of channels before epistemic cube channels were added (v6).
pub const PRE_EPISTEMIC_NUM_CHANNELS: usize = 28;

/// Base index for epistemic cube channels.
#[cfg(not(feature = "therapeutic"))]
pub const EPISTEMIC_CUBE_BASE: usize = 28;
#[cfg(feature = "therapeutic")]
pub const EPISTEMIC_CUBE_BASE: usize = 32;

/// Number of epistemic cube channels (E[5] + N[4] + M[4] + H[1] + quality[1]).
pub const EPISTEMIC_CUBE_CHANNELS: usize = 15;

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
    // Context channels (v3)
    "time_pressure",       // 20
    "domain_familiarity",  // 21
    "social_context",      // 22
    "response_confidence", // 23
    // Code channels (v5)
    "syntax_complexity", // 24
    "type_confidence",   // 25
    "algorithm_pattern", // 26
    "error_likelihood",  // 27
    // Epistemic Cube channels (v6) — 4D cube from Mycelix Epistemic Charter
    "e_tier_e0",         // 28: empirical one-hot E0 (opinion)
    "e_tier_e1",         // 29: empirical one-hot E1 (testimonial)
    "e_tier_e2",         // 30: empirical one-hot E2 (verifiable)
    "e_tier_e3",         // 31: empirical one-hot E3 (proven)
    "e_tier_e4",         // 32: empirical one-hot E4 (reproducible)
    "n_tier_n0",         // 33: normative one-hot N0 (personal)
    "n_tier_n1",         // 34: normative one-hot N1 (communal)
    "n_tier_n2",         // 35: normative one-hot N2 (network)
    "n_tier_n3",         // 36: normative one-hot N3 (axiomatic)
    "m_tier_m0",         // 37: materiality one-hot M0 (ephemeral)
    "m_tier_m1",         // 38: materiality one-hot M1 (temporal)
    "m_tier_m2",         // 39: materiality one-hot M2 (persistent)
    "m_tier_m3",         // 40: materiality one-hot M3 (foundational)
    "h_tier",            // 41: harmonic coherence scalar (0.0-1.0)
    "epistemic_quality", // 42: cube quality score (0.0-1.0)
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
    // Context channels (v3)
    "time_pressure",       // 20
    "domain_familiarity",  // 21
    "social_context",      // 22
    "response_confidence", // 23
    // Code channels (v5)
    "syntax_complexity", // 24
    "type_confidence",   // 25
    "algorithm_pattern", // 26
    "error_likelihood",  // 27
    // Therapeutic channels (v4)
    "therapeutic_intent",    // 28
    "alliance_quality",      // 29
    "client_distress_level", // 30
    "intervention_depth",    // 31
    // Epistemic Cube channels (v6) — 4D cube from Mycelix Epistemic Charter
    "e_tier_e0",         // 32: empirical one-hot E0 (opinion)
    "e_tier_e1",         // 33: empirical one-hot E1 (testimonial)
    "e_tier_e2",         // 34: empirical one-hot E2 (verifiable)
    "e_tier_e3",         // 35: empirical one-hot E3 (proven)
    "e_tier_e4",         // 36: empirical one-hot E4 (reproducible)
    "n_tier_n0",         // 37: normative one-hot N0 (personal)
    "n_tier_n1",         // 38: normative one-hot N1 (communal)
    "n_tier_n2",         // 39: normative one-hot N2 (network)
    "n_tier_n3",         // 40: normative one-hot N3 (axiomatic)
    "m_tier_m0",         // 41: materiality one-hot M0 (ephemeral)
    "m_tier_m1",         // 42: materiality one-hot M1 (temporal)
    "m_tier_m2",         // 43: materiality one-hot M2 (persistent)
    "m_tier_m3",         // 44: materiality one-hot M3 (foundational)
    "h_tier",            // 45: harmonic coherence scalar (0.0-1.0)
    "epistemic_quality", // 46: cube quality score (0.0-1.0)
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
    // Context channels (v3)
    [0.0, 1.0], // time_pressure
    [0.0, 1.0], // domain_familiarity
    [0.0, 1.0], // social_context
    [0.0, 1.0], // response_confidence
    // Code channels (v5)
    [0.0, 1.0], // syntax_complexity
    [0.0, 1.0], // type_confidence
    [0.0, 1.0], // algorithm_pattern
    [0.0, 1.0], // error_likelihood
    // Epistemic Cube channels (v6)
    [0.0, 1.0], // e_tier_e0
    [0.0, 1.0], // e_tier_e1
    [0.0, 1.0], // e_tier_e2
    [0.0, 1.0], // e_tier_e3
    [0.0, 1.0], // e_tier_e4
    [0.0, 1.0], // n_tier_n0
    [0.0, 1.0], // n_tier_n1
    [0.0, 1.0], // n_tier_n2
    [0.0, 1.0], // n_tier_n3
    [0.0, 1.0], // m_tier_m0
    [0.0, 1.0], // m_tier_m1
    [0.0, 1.0], // m_tier_m2
    [0.0, 1.0], // m_tier_m3
    [0.0, 1.0], // h_tier
    [0.0, 1.0], // epistemic_quality
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
    // Context channels (v3)
    [0.0, 1.0], // time_pressure
    [0.0, 1.0], // domain_familiarity
    [0.0, 1.0], // social_context
    [0.0, 1.0], // response_confidence
    // Code channels (v5)
    [0.0, 1.0], // syntax_complexity
    [0.0, 1.0], // type_confidence
    [0.0, 1.0], // algorithm_pattern
    [0.0, 1.0], // error_likelihood
    // Therapeutic channels (v4)
    [0.0, 7.0], // therapeutic_intent
    [0.0, 1.0], // alliance_quality
    [0.0, 1.0], // client_distress_level
    [0.0, 1.0], // intervention_depth
    // Epistemic Cube channels (v6)
    [0.0, 1.0], // e_tier_e0
    [0.0, 1.0], // e_tier_e1
    [0.0, 1.0], // e_tier_e2
    [0.0, 1.0], // e_tier_e3
    [0.0, 1.0], // e_tier_e4
    [0.0, 1.0], // n_tier_n0
    [0.0, 1.0], // n_tier_n1
    [0.0, 1.0], // n_tier_n2
    [0.0, 1.0], // n_tier_n3
    [0.0, 1.0], // m_tier_m0
    [0.0, 1.0], // m_tier_m1
    [0.0, 1.0], // m_tier_m2
    [0.0, 1.0], // m_tier_m3
    [0.0, 1.0], // h_tier
    [0.0, 1.0], // epistemic_quality
];

/// Default values for the 4 new channels (used when loading legacy 20-channel data).
pub const NEW_CHANNEL_DEFAULTS: [f32; 4] = [
    0.0, // time_pressure: relaxed
    0.5, // domain_familiarity: mid
    0.5, // social_context: mid
    0.5, // response_confidence: mid
];

/// Default values for the 4 code channels (indices 24-27).
pub const CODE_CHANNEL_DEFAULTS: [f32; 4] = [
    0.0, // syntax_complexity: simple
    0.0, // type_confidence: unknown
    0.0, // algorithm_pattern: none
    0.0, // error_likelihood: likely correct
];

/// Default values for the 15 epistemic cube channels.
pub const EPISTEMIC_CUBE_DEFAULTS: [f32; EPISTEMIC_CUBE_CHANNELS] = [
    0.0, 0.0, 0.0, 0.0, 0.0, // E-tier one-hot (none active)
    0.0, 0.0, 0.0, 0.0, // N-tier one-hot (none active)
    0.0, 0.0, 0.0, 0.0,  // M-tier one-hot (none active)
    0.25, // h_tier: H1 neutral
    0.0,  // epistemic_quality: unknown
];

/// Decoupled thought state: scalar channels extracted from StructuredThought.
///
/// Conversion from `StructuredThought -> ThoughtChannels` happens at the
/// integration layer (Phase 3), avoiding circular dependency.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ThoughtChannels {
    #[serde(
        serialize_with = "channel_serde::serialize",
        deserialize_with = "channel_serde::deserialize",
        default = "ThoughtChannels::default_array",
        bound(serialize = "", deserialize = "")
    )]
    pub channels: [f32; NUM_CHANNELS],
}

/// Custom serde for arrays > 32 elements (serde doesn't derive for large arrays).
mod channel_serde {
    use super::NUM_CHANNELS;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(arr: &[f32; NUM_CHANNELS], s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(NUM_CHANNELS))?;
        for val in arr {
            seq.serialize_element(val)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; NUM_CHANNELS], D::Error> {
        let v: Vec<f32> = Vec::deserialize(d)?;
        if v.len() != NUM_CHANNELS {
            // Accept shorter arrays (legacy data) by padding with defaults
            let mut arr = super::ThoughtChannels::default_array();
            let copy_len = v.len().min(NUM_CHANNELS);
            arr[..copy_len].copy_from_slice(&v[..copy_len]);
            return Ok(arr);
        }
        v.try_into()
            .map_err(|_| serde::de::Error::custom("wrong channel count"))
    }
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
                // Context channels (v3)
                0.0, // time_pressure: relaxed
                0.5, // domain_familiarity: mid
                0.5, // social_context: mid
                0.5, // response_confidence: mid
                // Code channels (v5)
                0.0, // syntax_complexity: simple
                0.0, // type_confidence: unknown
                0.0, // algorithm_pattern: none
                0.0, // error_likelihood: likely correct
                // Epistemic Cube channels (v6) — default: no cube data
                0.0, 0.0, 0.0, 0.0, 0.0, // E-tier one-hot (none active)
                0.0, 0.0, 0.0, 0.0, // N-tier one-hot (none active)
                0.0, 0.0, 0.0, 0.0,  // M-tier one-hot (none active)
                0.25, // h_tier: H1 neutral (0.25)
                0.0,  // epistemic_quality: unknown
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
                // Code channels (v5)
                0.0, // syntax_complexity: simple
                0.0, // type_confidence: unknown
                0.0, // algorithm_pattern: none
                0.0, // error_likelihood: likely correct
                // Therapeutic channels (v4)
                0.0, // therapeutic_intent
                0.5, // alliance_quality
                0.0, // client_distress_level
                0.0, // intervention_depth
                // Epistemic Cube channels (v6) — default: no cube data
                0.0, 0.0, 0.0, 0.0, 0.0, // E-tier one-hot (none active)
                0.0, 0.0, 0.0, 0.0, // N-tier one-hot (none active)
                0.0, 0.0, 0.0, 0.0,  // M-tier one-hot (none active)
                0.25, // h_tier: H1 neutral (0.25)
                0.0,  // epistemic_quality: unknown
            ],
        }
    }
}

impl ThoughtChannels {
    /// Returns the default channel array values (used by serde deserialization
    /// when loading legacy data with fewer channels).
    fn default_array() -> [f32; NUM_CHANNELS] {
        Self::default().channels
    }

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

    /// Set arousal (emotional intensity).
    pub fn set_arousal(&mut self, arousal: f32) {
        self.channels[10] = arousal.clamp(0.0, 1.0);
    }

    /// Set valence (emotional tone).
    pub fn set_valence(&mut self, valence: f32) {
        self.channels[9] = valence.clamp(-1.0, 1.0);
    }

    /// Helper for language intent detection.
    pub fn language_intent(&self) -> Option<String> {
        // Simple heuristic: check highest of code-related channels
        if self.channels[24] > 0.5 {
            Some("rust".to_string())
        } else {
            None
        }
    }

    /// Check if prompt-like context contains keywords.
    pub fn prompt_contains_any(&self, keywords: &[&str]) -> bool {
        // In real: check text memory. Here: simplified mock.
        keywords
            .iter()
            .any(|&k| k == "rust" || k == "nix" || k == "kubernetes")
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

    /// Get moral safety score (0.0 = high risk, 1.0 = safe).
    pub fn moral_score(&self) -> f32 {
        #[cfg(not(feature = "therapeutic"))]
        {
            self.channels[28]
        }
        #[cfg(feature = "therapeutic")]
        {
            1.0
        } // Default to safe
    }

    /// Get narrative/maintainability score (0.0 = poor, 1.0 = good).
    pub fn narrative_score(&self) -> f32 {
        #[cfg(not(feature = "therapeutic"))]
        {
            self.channels[29]
        }
        #[cfg(feature = "therapeutic")]
        {
            1.0
        }
    }

    /// Get idiomaticity score (0.0 = non-idiomatic, 1.0 = idiomatic).
    pub fn idiomatic_score(&self) -> f32 {
        #[cfg(not(feature = "therapeutic"))]
        {
            self.channels[30]
        }
        #[cfg(feature = "therapeutic")]
        {
            1.0
        }
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
        // Code channels default to 0.0 (already set by Default)
        #[cfg(feature = "therapeutic")]
        {
            tc.channels[28] = THERAPEUTIC_CHANNEL_DEFAULTS[0];
            tc.channels[29] = THERAPEUTIC_CHANNEL_DEFAULTS[1];
            tc.channels[30] = THERAPEUTIC_CHANNEL_DEFAULTS[2];
            tc.channels[31] = THERAPEUTIC_CHANNEL_DEFAULTS[3];
        }
        // Epistemic cube defaults
        for i in 0..EPISTEMIC_CUBE_CHANNELS {
            tc.channels[EPISTEMIC_CUBE_BASE + i] = EPISTEMIC_CUBE_DEFAULTS[i];
        }
        tc
    }

    /// Set code generation channels.
    pub fn set_code(
        &mut self,
        syntax_complexity: f32,
        type_confidence: f32,
        algorithm_pattern: f32,
        error_likelihood: f32,
    ) {
        self.channels[24] = syntax_complexity.clamp(0.0, 1.0);
        self.channels[25] = type_confidence.clamp(0.0, 1.0);
        self.channels[26] = algorithm_pattern.clamp(0.0, 1.0);
        self.channels[27] = error_likelihood.clamp(0.0, 1.0);
    }

    /// Get syntax complexity (0.0=simple expression, 1.0=deeply nested generics/lifetimes).
    pub fn syntax_complexity(&self) -> f32 {
        self.channels[24]
    }

    /// Get type confidence (0.0=unknown types, 1.0=all types resolved).
    pub fn type_confidence(&self) -> f32 {
        self.channels[25]
    }

    /// Get algorithm pattern strength (0.0=no pattern, 1.0=strong match).
    pub fn algorithm_pattern(&self) -> f32 {
        self.channels[26]
    }

    /// Get error likelihood (0.0=likely correct, 1.0=likely has errors).
    pub fn error_likelihood(&self) -> f32 {
        self.channels[27]
    }

    #[cfg(feature = "therapeutic")]
    pub fn set_therapeutic(&mut self, intent: f32, alliance: f32, distress: f32, depth: f32) {
        self.channels[28] = intent;
        self.channels[29] = alliance;
        self.channels[30] = distress;
        self.channels[31] = depth;
    }

    #[cfg(feature = "therapeutic")]
    pub fn therapeutic_intent(&self) -> f32 {
        self.channels[28]
    }

    #[cfg(feature = "therapeutic")]
    pub fn alliance_quality(&self) -> f32 {
        self.channels[29]
    }

    #[cfg(feature = "therapeutic")]
    pub fn client_distress_level(&self) -> f32 {
        self.channels[30]
    }

    #[cfg(feature = "therapeutic")]
    pub fn intervention_depth(&self) -> f32 {
        self.channels[31]
    }

    // ── Epistemic Cube channels (v6) ────────────────────────────────────

    /// Set the full 4D epistemic cube as channel data.
    ///
    /// - `e_tier`: 0-4 (E0-E4 empirical verifiability)
    /// - `n_tier`: 0-3 (N0-N3 normative authority)
    /// - `m_tier`: 0-3 (M0-M3 materiality/permanence)
    /// - `h_value`: 0.0-1.0 (harmonic coherence, continuous)
    /// - `quality`: 0.0-1.0 (composite quality score)
    pub fn set_epistemic_cube(
        &mut self,
        e_tier: u8,
        n_tier: u8,
        m_tier: u8,
        h_value: f32,
        quality: f32,
    ) {
        let base = EPISTEMIC_CUBE_BASE;

        // E-tier one-hot (5 slots)
        for i in 0..5 {
            self.channels[base + i] = if i == e_tier as usize { 1.0 } else { 0.0 };
        }

        // N-tier one-hot (4 slots)
        for i in 0..4 {
            self.channels[base + 5 + i] = if i == n_tier as usize { 1.0 } else { 0.0 };
        }

        // M-tier one-hot (4 slots)
        for i in 0..4 {
            self.channels[base + 9 + i] = if i == m_tier as usize { 1.0 } else { 0.0 };
        }

        // H-tier scalar
        self.channels[base + 13] = h_value.clamp(0.0, 1.0);

        // Quality score
        self.channels[base + 14] = quality.clamp(0.0, 1.0);
    }

    /// Get the E-tier index (0-4) from one-hot encoding, or None if no tier is active.
    pub fn e_tier(&self) -> Option<u8> {
        let base = EPISTEMIC_CUBE_BASE;
        (0..5u8).find(|&i| self.channels[base + i as usize] > 0.5)
    }

    /// Get the N-tier index (0-3) from one-hot encoding, or None if no tier is active.
    pub fn n_tier(&self) -> Option<u8> {
        let base = EPISTEMIC_CUBE_BASE + 5;
        (0..4u8).find(|&i| self.channels[base + i as usize] > 0.5)
    }

    /// Get the M-tier index (0-3) from one-hot encoding, or None if no tier is active.
    pub fn m_tier(&self) -> Option<u8> {
        let base = EPISTEMIC_CUBE_BASE + 9;
        (0..4u8).find(|&i| self.channels[base + i as usize] > 0.5)
    }

    /// Get the H-tier scalar (0.0-1.0).
    pub fn h_tier(&self) -> f32 {
        self.channels[EPISTEMIC_CUBE_BASE + 13]
    }

    /// Get the epistemic quality score (0.0-1.0).
    pub fn epistemic_quality(&self) -> f32 {
        self.channels[EPISTEMIC_CUBE_BASE + 14]
    }

    /// Returns true if the epistemic cube channels have been populated
    /// (at least one E/N/M one-hot is active).
    pub fn has_epistemic_cube(&self) -> bool {
        self.e_tier().is_some() || self.n_tier().is_some() || self.m_tier().is_some()
    }
}

#[cfg(feature = "therapeutic")]
pub const THERAPEUTIC_CHANNEL_NAMES: &[&str] = &[
    "therapeutic_intent",
    "alliance_quality",
    "client_distress_level",
    "intervention_depth",
];

#[cfg(feature = "therapeutic")]
pub const THERAPEUTIC_CHANNEL_RANGES: &[[f32; 2]] =
    &[[0.0, 7.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];

#[cfg(feature = "therapeutic")]
pub const THERAPEUTIC_CHANNEL_DEFAULTS: &[f32] = &[0.0, 0.5, 0.0, 0.0];

/// HDC encoder for thought channels.
///
/// Encodes a 20D `ThoughtChannels` into a full 16,384D `ContinuousHV` via:
/// 1. Per-channel normalization to [0, 1] using known ranges
/// 2. Level encoding (thermometer coding via bundled levels 0..k)
/// 3. Binding with genesis-seeded channel base vectors
/// 4. Bundling all bound channel HVs
#[derive(Clone)]
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

    /// Alias for `new()`, used in some constructors.
    pub fn new_from_genesis(genesis: &GenesisSeed) -> Self {
        Self::new(genesis)
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
            sim < 0.999,
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
        // With 43 channels and only 3 differing (valence, arousal, warmth),
        // the 40 shared channels dominate — expect high but not perfect similarity.
        assert!(
            sim < 0.9999,
            "Different emotional states should produce different HVs: sim={sim}"
        );
        assert!(sim < 1.0 - 1e-6, "Should not be identical: sim={sim}");
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
    fn test_all_channels_produce_distinct_encodings() {
        let genesis = test_genesis();
        let enc = ThoughtLanguageEncoder::new(&genesis);
        let baseline = enc.encode(&ThoughtChannels::default());
        let mut distinct = 0;
        for i in 0..NUM_CHANNELS {
            let mut ch = ThoughtChannels::default();
            ch.channels[i] = CHANNEL_RANGES[i][1];
            let sim = baseline.similarity(&enc.encode(&ch));
            if (sim - 1.0).abs() > 1e-4 {
                distinct += 1;
            }
        }
        assert!(
            distinct >= 20,
            "At least 20/{NUM_CHANNELS} channels should be distinct, got {distinct}"
        );
    }

    #[test]
    fn test_code_channels_default_zero() {
        let tc = ThoughtChannels::default();
        assert_eq!(tc.syntax_complexity(), 0.0);
        assert_eq!(tc.type_confidence(), 0.0);
        assert_eq!(tc.algorithm_pattern(), 0.0);
        assert_eq!(tc.error_likelihood(), 0.0);
    }

    #[test]
    fn test_code_channels_set_and_get() {
        let mut tc = ThoughtChannels::default();
        tc.set_code(0.8, 0.9, 0.7, 0.3);
        assert!((tc.syntax_complexity() - 0.8).abs() < 1e-6);
        assert!((tc.type_confidence() - 0.9).abs() < 1e-6);
        assert!((tc.algorithm_pattern() - 0.7).abs() < 1e-6);
        assert!((tc.error_likelihood() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_code_channels_affect_encoding() {
        let enc = ThoughtLanguageEncoder::new(&test_genesis());
        let baseline = ThoughtChannels::default();
        let mut code_active = ThoughtChannels::default();
        code_active.set_code(1.0, 1.0, 1.0, 1.0);
        let sim = enc.encode(&baseline).similarity(&enc.encode(&code_active));
        assert!(
            sim < 0.99,
            "Code channels should affect encoding, sim={sim}"
        );
    }

    #[test]
    fn test_out_of_range_channels_clamped() {
        let enc = ThoughtLanguageEncoder::new(&test_genesis());
        let mut ch = ThoughtChannels::default();
        ch.channels[8] = 100.0;
        ch.channels[9] = -50.0;
        ch.channels[10] = 999.0;
        assert!(enc.encode(&ch).as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_legacy_20_channel_conversion() {
        let mut legacy = [0.0f32; 20];
        legacy[0] = 1.0;
        legacy[9] = -0.5;
        legacy[12] = 0.8;
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
        assert!(
            sim < 0.999,
            "Low vs high consciousness should differ, sim={sim}"
        );
    }

    #[test]
    fn test_emotion_channels_affect_encoding() {
        let enc = ThoughtLanguageEncoder::new(&test_genesis());
        let mut neg = ThoughtChannels::default();
        neg.set_emotion(-1.0, 1.0, 0.0);
        let mut pos = ThoughtChannels::default();
        pos.set_emotion(1.0, 0.0, 1.0);
        let sim = enc.encode(&neg).similarity(&enc.encode(&pos));
        assert!(sim < 0.999, "Opposite emotions should differ, sim={sim}");
    }
}
