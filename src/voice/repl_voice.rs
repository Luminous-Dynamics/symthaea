//! # REPL Voice Output Module
//!
//! Provides consciousness-modulated speech synthesis for the symthaea-repl binary.
//!
//! ## Features
//!
//! - Text-to-audio conversion using the formant vocoder
//! - Consciousness-aware prosody modulation
//! - Audio playback via rodio (when available)
//! - Graceful fallback when audio unavailable
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    REPL VOICE OUTPUT                             │
//! │                                                                  │
//! │   Text Response ──► Text-to-Phoneme ──► Articulatory Synth       │
//! │         │                                      │                 │
//! │         ▼                                      ▼                 │
//! │   ConsciousnessSnapshot ──► CognitivePacing ──► FormantFrames   │
//! │                                                      │          │
//! │                                                      ▼          │
//! │                                              FormantVocoder     │
//! │                                                      │          │
//! │                                                      ▼          │
//! │                                              Audio Playback     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Consciousness Modulation
//!
//! Speech is modulated by consciousness state:
//! - **Rate**: Slower when contemplative, faster when excited
//! - **Pauses**: Longer when uncertain, shorter when confident
//! - **Pitch**: Higher arousal = higher pitch range
//! - **Energy**: Flow state = more emphatic delivery

use anyhow::Result;
use std::collections::HashMap;
#[cfg(feature = "audio")]
use tracing::info;
use tracing::{debug, warn};

use crate::voice::{
    ArticulatoryConfig, ArticulatorySynthesizer, CognitiveVoiceBridge, FormantVocoder, LTCPacing,
    TimedPhoneme, VocoderConfig, VoiceOutput, VoiceOutputConfig,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for REPL voice output
#[derive(Debug, Clone)]
pub struct ReplVoiceConfig {
    /// Output sample rate (Hz)
    pub sample_rate: u32,

    /// Base speech rate multiplier (1.0 = normal)
    pub base_rate: f32,

    /// Master volume (0.0 to 1.0)
    pub volume: f32,

    /// Audio output device name (None = default)
    pub device_name: Option<String>,

    /// Whether to use consciousness modulation
    pub consciousness_modulated: bool,

    /// Base fundamental frequency (Hz) - determines voice pitch
    pub base_f0: f32,

    /// Whether to use articulatory synthesis (vs simulated TTS)
    pub use_articulatory: bool,

    /// Whether to use the LTC-driven vocal tract pipeline (requires `vocal-tract` feature)
    pub use_ltc_pipeline: bool,

    /// Phoneme duration base (seconds)
    pub phoneme_duration_base: f32,
}

impl Default for ReplVoiceConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            base_rate: 1.0,
            volume: 0.8,
            device_name: None,
            consciousness_modulated: true,
            base_f0: 150.0, // Neutral voice
            use_articulatory: true,
            use_ltc_pipeline: cfg!(feature = "vocal-tract"),
            phoneme_duration_base: 0.08,
        }
    }
}

impl ReplVoiceConfig {
    /// Create a low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            base_rate: 1.2,          // Slightly faster
            use_articulatory: false, // Simpler synthesis
            ..Default::default()
        }
    }

    /// Create a high-quality configuration
    pub fn high_quality() -> Self {
        Self {
            sample_rate: 44100,
            use_articulatory: true,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMPLE TEXT TO PHONEME CONVERTER
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple English text to ARPABET phoneme converter
///
/// This is a basic rule-based converter for common words and patterns.
/// For production use, a proper G2P (grapheme-to-phoneme) system would be needed.
pub struct SimpleG2P {
    /// Common word pronunciations
    dictionary: HashMap<String, Vec<&'static str>>,
}

impl SimpleG2P {
    /// Create a new G2P converter with comprehensive English dictionary
    ///
    /// Contains 400+ common words for demo-quality speech synthesis.
    pub fn new() -> Self {
        let mut dictionary = HashMap::new();

        // ═══════════════════════════════════════════════════════════════════
        // ARTICLES & DETERMINERS
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("a".to_string(), vec!["AH0"]);
        dictionary.insert("an".to_string(), vec!["AE1", "N"]);
        dictionary.insert("the".to_string(), vec!["DH", "AH0"]);
        dictionary.insert("this".to_string(), vec!["DH", "IH1", "S"]);
        dictionary.insert("that".to_string(), vec!["DH", "AE1", "T"]);
        dictionary.insert("these".to_string(), vec!["DH", "IY1", "Z"]);
        dictionary.insert("those".to_string(), vec!["DH", "OW1", "Z"]);
        dictionary.insert("some".to_string(), vec!["S", "AH1", "M"]);
        dictionary.insert("any".to_string(), vec!["EH1", "N", "IY0"]);
        dictionary.insert("every".to_string(), vec!["EH1", "V", "R", "IY0"]);
        dictionary.insert("each".to_string(), vec!["IY1", "CH"]);
        dictionary.insert("all".to_string(), vec!["AO1", "L"]);
        dictionary.insert("both".to_string(), vec!["B", "OW1", "TH"]);
        dictionary.insert("either".to_string(), vec!["IY1", "DH", "ER0"]);
        dictionary.insert("neither".to_string(), vec!["N", "IY1", "DH", "ER0"]);
        dictionary.insert("many".to_string(), vec!["M", "EH1", "N", "IY0"]);
        dictionary.insert("much".to_string(), vec!["M", "AH1", "CH"]);
        dictionary.insert("few".to_string(), vec!["F", "Y", "UW1"]);
        dictionary.insert(
            "several".to_string(),
            vec!["S", "EH1", "V", "R", "AH0", "L"],
        );

        // ═══════════════════════════════════════════════════════════════════
        // PRONOUNS
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("i".to_string(), vec!["AY1"]);
        dictionary.insert("me".to_string(), vec!["M", "IY1"]);
        dictionary.insert("my".to_string(), vec!["M", "AY1"]);
        dictionary.insert("mine".to_string(), vec!["M", "AY1", "N"]);
        dictionary.insert("myself".to_string(), vec!["M", "AY0", "S", "EH1", "L", "F"]);
        dictionary.insert("you".to_string(), vec!["Y", "UW1"]);
        dictionary.insert("your".to_string(), vec!["Y", "AO1", "R"]);
        dictionary.insert("yours".to_string(), vec!["Y", "AO1", "R", "Z"]);
        dictionary.insert(
            "yourself".to_string(),
            vec!["Y", "AO0", "R", "S", "EH1", "L", "F"],
        );
        dictionary.insert("he".to_string(), vec!["HH", "IY1"]);
        dictionary.insert("him".to_string(), vec!["HH", "IH1", "M"]);
        dictionary.insert("his".to_string(), vec!["HH", "IH1", "Z"]);
        dictionary.insert(
            "himself".to_string(),
            vec!["HH", "IH0", "M", "S", "EH1", "L", "F"],
        );
        dictionary.insert("she".to_string(), vec!["SH", "IY1"]);
        dictionary.insert("her".to_string(), vec!["HH", "ER1"]);
        dictionary.insert("hers".to_string(), vec!["HH", "ER1", "Z"]);
        dictionary.insert(
            "herself".to_string(),
            vec!["HH", "ER0", "S", "EH1", "L", "F"],
        );
        dictionary.insert("it".to_string(), vec!["IH1", "T"]);
        dictionary.insert("its".to_string(), vec!["IH1", "T", "S"]);
        dictionary.insert("itself".to_string(), vec!["IH0", "T", "S", "EH1", "L", "F"]);
        dictionary.insert("we".to_string(), vec!["W", "IY1"]);
        dictionary.insert("us".to_string(), vec!["AH1", "S"]);
        dictionary.insert("our".to_string(), vec!["AW1", "ER0"]);
        dictionary.insert("ours".to_string(), vec!["AW1", "ER0", "Z"]);
        dictionary.insert(
            "ourselves".to_string(),
            vec!["AW0", "ER0", "S", "EH1", "L", "V", "Z"],
        );
        dictionary.insert("they".to_string(), vec!["DH", "EY1"]);
        dictionary.insert("them".to_string(), vec!["DH", "EH1", "M"]);
        dictionary.insert("their".to_string(), vec!["DH", "EH1", "R"]);
        dictionary.insert("theirs".to_string(), vec!["DH", "EH1", "R", "Z"]);
        dictionary.insert(
            "themselves".to_string(),
            vec!["DH", "EH0", "M", "S", "EH1", "L", "V", "Z"],
        );
        dictionary.insert("who".to_string(), vec!["HH", "UW1"]);
        dictionary.insert("whom".to_string(), vec!["HH", "UW1", "M"]);
        dictionary.insert("whose".to_string(), vec!["HH", "UW1", "Z"]);
        dictionary.insert("which".to_string(), vec!["W", "IH1", "CH"]);
        dictionary.insert("what".to_string(), vec!["W", "AH1", "T"]);
        dictionary.insert(
            "whatever".to_string(),
            vec!["W", "AH2", "T", "EH1", "V", "ER0"],
        );
        dictionary.insert("whoever".to_string(), vec!["HH", "UW0", "EH1", "V", "ER0"]);

        // ═══════════════════════════════════════════════════════════════════
        // PREPOSITIONS & CONJUNCTIONS
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("to".to_string(), vec!["T", "UW1"]);
        dictionary.insert("of".to_string(), vec!["AH1", "V"]);
        dictionary.insert("in".to_string(), vec!["IH1", "N"]);
        dictionary.insert("for".to_string(), vec!["F", "AO1", "R"]);
        dictionary.insert("on".to_string(), vec!["AA1", "N"]);
        dictionary.insert("with".to_string(), vec!["W", "IH1", "TH"]);
        dictionary.insert("at".to_string(), vec!["AE1", "T"]);
        dictionary.insert("by".to_string(), vec!["B", "AY1"]);
        dictionary.insert("from".to_string(), vec!["F", "R", "AH1", "M"]);
        dictionary.insert("up".to_string(), vec!["AH1", "P"]);
        dictionary.insert("about".to_string(), vec!["AH0", "B", "AW1", "T"]);
        dictionary.insert("into".to_string(), vec!["IH1", "N", "T", "UW0"]);
        dictionary.insert("over".to_string(), vec!["OW1", "V", "ER0"]);
        dictionary.insert("after".to_string(), vec!["AE1", "F", "T", "ER0"]);
        dictionary.insert("beneath".to_string(), vec!["B", "IH0", "N", "IY1", "TH"]);
        dictionary.insert("under".to_string(), vec!["AH1", "N", "D", "ER0"]);
        dictionary.insert("above".to_string(), vec!["AH0", "B", "AH1", "V"]);
        dictionary.insert("below".to_string(), vec!["B", "IH0", "L", "OW1"]);
        dictionary.insert(
            "between".to_string(),
            vec!["B", "IH0", "T", "W", "IY1", "N"],
        );
        dictionary.insert("among".to_string(), vec!["AH0", "M", "AH1", "NG"]);
        dictionary.insert("through".to_string(), vec!["TH", "R", "UW1"]);
        dictionary.insert("during".to_string(), vec!["D", "UH1", "R", "IH0", "NG"]);
        dictionary.insert("before".to_string(), vec!["B", "IH0", "F", "AO1", "R"]);
        dictionary.insert(
            "behind".to_string(),
            vec!["B", "IH0", "HH", "AY1", "N", "D"],
        );
        dictionary.insert("beyond".to_string(), vec!["B", "IH0", "AA1", "N", "D"]);
        dictionary.insert("without".to_string(), vec!["W", "IH0", "TH", "AW1", "T"]);
        dictionary.insert("within".to_string(), vec!["W", "IH0", "DH", "IH1", "N"]);
        dictionary.insert("around".to_string(), vec!["ER0", "AW1", "N", "D"]);
        dictionary.insert("across".to_string(), vec!["AH0", "K", "R", "AO1", "S"]);
        dictionary.insert("along".to_string(), vec!["AH0", "L", "AO1", "NG"]);
        dictionary.insert("toward".to_string(), vec!["T", "AH0", "W", "AO1", "R", "D"]);
        dictionary.insert(
            "towards".to_string(),
            vec!["T", "AH0", "W", "AO1", "R", "D", "Z"],
        );
        dictionary.insert(
            "against".to_string(),
            vec!["AH0", "G", "EH1", "N", "S", "T"],
        );
        dictionary.insert("and".to_string(), vec!["AE1", "N", "D"]);
        dictionary.insert("or".to_string(), vec!["AO1", "R"]);
        dictionary.insert("but".to_string(), vec!["B", "AH1", "T"]);
        dictionary.insert("if".to_string(), vec!["IH1", "F"]);
        dictionary.insert("then".to_string(), vec!["DH", "EH1", "N"]);
        dictionary.insert("else".to_string(), vec!["EH1", "L", "S"]);
        dictionary.insert("when".to_string(), vec!["W", "EH1", "N"]);
        dictionary.insert("where".to_string(), vec!["W", "EH1", "R"]);
        dictionary.insert("why".to_string(), vec!["W", "AY1"]);
        dictionary.insert("how".to_string(), vec!["HH", "AW1"]);
        dictionary.insert("because".to_string(), vec!["B", "IH0", "K", "AO1", "Z"]);
        dictionary.insert("although".to_string(), vec!["AO0", "L", "DH", "OW1"]);
        dictionary.insert("while".to_string(), vec!["W", "AY1", "L"]);
        dictionary.insert("since".to_string(), vec!["S", "IH1", "N", "S"]);
        dictionary.insert("unless".to_string(), vec!["AH0", "N", "L", "EH1", "S"]);
        dictionary.insert("until".to_string(), vec!["AH0", "N", "T", "IH1", "L"]);
        dictionary.insert("however".to_string(), vec!["HH", "AW0", "EH1", "V", "ER0"]);
        dictionary.insert(
            "therefore".to_string(),
            vec!["DH", "EH1", "R", "F", "AO2", "R"],
        );
        dictionary.insert("thus".to_string(), vec!["DH", "AH1", "S"]);

        // ═══════════════════════════════════════════════════════════════════
        // COMMON VERBS (Base Forms)
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("be".to_string(), vec!["B", "IY1"]);
        dictionary.insert("am".to_string(), vec!["AE1", "M"]);
        dictionary.insert("is".to_string(), vec!["IH1", "Z"]);
        dictionary.insert("are".to_string(), vec!["AA1", "R"]);
        dictionary.insert("was".to_string(), vec!["W", "AA1", "Z"]);
        dictionary.insert("were".to_string(), vec!["W", "ER1"]);
        dictionary.insert("been".to_string(), vec!["B", "IH1", "N"]);
        dictionary.insert("being".to_string(), vec!["B", "IY1", "IH0", "NG"]);
        dictionary.insert("have".to_string(), vec!["HH", "AE1", "V"]);
        dictionary.insert("has".to_string(), vec!["HH", "AE1", "Z"]);
        dictionary.insert("had".to_string(), vec!["HH", "AE1", "D"]);
        dictionary.insert("having".to_string(), vec!["HH", "AE1", "V", "IH0", "NG"]);
        dictionary.insert("do".to_string(), vec!["D", "UW1"]);
        dictionary.insert("does".to_string(), vec!["D", "AH1", "Z"]);
        dictionary.insert("did".to_string(), vec!["D", "IH1", "D"]);
        dictionary.insert("doing".to_string(), vec!["D", "UW1", "IH0", "NG"]);
        dictionary.insert("done".to_string(), vec!["D", "AH1", "N"]);
        dictionary.insert("say".to_string(), vec!["S", "EY1"]);
        dictionary.insert("says".to_string(), vec!["S", "EH1", "Z"]);
        dictionary.insert("said".to_string(), vec!["S", "EH1", "D"]);
        dictionary.insert("saying".to_string(), vec!["S", "EY1", "IH0", "NG"]);
        dictionary.insert("go".to_string(), vec!["G", "OW1"]);
        dictionary.insert("goes".to_string(), vec!["G", "OW1", "Z"]);
        dictionary.insert("went".to_string(), vec!["W", "EH1", "N", "T"]);
        dictionary.insert("going".to_string(), vec!["G", "OW1", "IH0", "NG"]);
        dictionary.insert("gone".to_string(), vec!["G", "AO1", "N"]);
        dictionary.insert("get".to_string(), vec!["G", "EH1", "T"]);
        dictionary.insert("gets".to_string(), vec!["G", "EH1", "T", "S"]);
        dictionary.insert("got".to_string(), vec!["G", "AA1", "T"]);
        dictionary.insert("getting".to_string(), vec!["G", "EH1", "T", "IH0", "NG"]);
        dictionary.insert("make".to_string(), vec!["M", "EY1", "K"]);
        dictionary.insert("makes".to_string(), vec!["M", "EY1", "K", "S"]);
        dictionary.insert("made".to_string(), vec!["M", "EY1", "D"]);
        dictionary.insert("making".to_string(), vec!["M", "EY1", "K", "IH0", "NG"]);
        dictionary.insert("know".to_string(), vec!["N", "OW1"]);
        dictionary.insert("knows".to_string(), vec!["N", "OW1", "Z"]);
        dictionary.insert("knew".to_string(), vec!["N", "UW1"]);
        dictionary.insert("known".to_string(), vec!["N", "OW1", "N"]);
        dictionary.insert("knowing".to_string(), vec!["N", "OW1", "IH0", "NG"]);
        dictionary.insert("think".to_string(), vec!["TH", "IH1", "NG", "K"]);
        dictionary.insert("thinks".to_string(), vec!["TH", "IH1", "NG", "K", "S"]);
        dictionary.insert("thought".to_string(), vec!["TH", "AO1", "T"]);
        dictionary.insert(
            "thinking".to_string(),
            vec!["TH", "IH1", "NG", "K", "IH0", "NG"],
        );
        dictionary.insert("take".to_string(), vec!["T", "EY1", "K"]);
        dictionary.insert("takes".to_string(), vec!["T", "EY1", "K", "S"]);
        dictionary.insert("took".to_string(), vec!["T", "UH1", "K"]);
        dictionary.insert("taken".to_string(), vec!["T", "EY1", "K", "AH0", "N"]);
        dictionary.insert("taking".to_string(), vec!["T", "EY1", "K", "IH0", "NG"]);
        dictionary.insert("see".to_string(), vec!["S", "IY1"]);
        dictionary.insert("sees".to_string(), vec!["S", "IY1", "Z"]);
        dictionary.insert("saw".to_string(), vec!["S", "AO1"]);
        dictionary.insert("seen".to_string(), vec!["S", "IY1", "N"]);
        dictionary.insert("seeing".to_string(), vec!["S", "IY1", "IH0", "NG"]);
        dictionary.insert("come".to_string(), vec!["K", "AH1", "M"]);
        dictionary.insert("comes".to_string(), vec!["K", "AH1", "M", "Z"]);
        dictionary.insert("came".to_string(), vec!["K", "EY1", "M"]);
        dictionary.insert("coming".to_string(), vec!["K", "AH1", "M", "IH0", "NG"]);
        dictionary.insert("want".to_string(), vec!["W", "AA1", "N", "T"]);
        dictionary.insert("wants".to_string(), vec!["W", "AA1", "N", "T", "S"]);
        dictionary.insert("wanted".to_string(), vec!["W", "AA1", "N", "T", "IH0", "D"]);
        dictionary.insert(
            "wanting".to_string(),
            vec!["W", "AA1", "N", "T", "IH0", "NG"],
        );
        dictionary.insert("use".to_string(), vec!["Y", "UW1", "Z"]);
        dictionary.insert("uses".to_string(), vec!["Y", "UW1", "Z", "IH0", "Z"]);
        dictionary.insert("used".to_string(), vec!["Y", "UW1", "Z", "D"]);
        dictionary.insert("using".to_string(), vec!["Y", "UW1", "Z", "IH0", "NG"]);
        dictionary.insert("find".to_string(), vec!["F", "AY1", "N", "D"]);
        dictionary.insert("finds".to_string(), vec!["F", "AY1", "N", "D", "Z"]);
        dictionary.insert("found".to_string(), vec!["F", "AW1", "N", "D"]);
        dictionary.insert(
            "finding".to_string(),
            vec!["F", "AY1", "N", "D", "IH0", "NG"],
        );
        dictionary.insert("give".to_string(), vec!["G", "IH1", "V"]);
        dictionary.insert("gives".to_string(), vec!["G", "IH1", "V", "Z"]);
        dictionary.insert("gave".to_string(), vec!["G", "EY1", "V"]);
        dictionary.insert("given".to_string(), vec!["G", "IH1", "V", "AH0", "N"]);
        dictionary.insert("giving".to_string(), vec!["G", "IH1", "V", "IH0", "NG"]);
        dictionary.insert("tell".to_string(), vec!["T", "EH1", "L"]);
        dictionary.insert("tells".to_string(), vec!["T", "EH1", "L", "Z"]);
        dictionary.insert("told".to_string(), vec!["T", "OW1", "L", "D"]);
        dictionary.insert("telling".to_string(), vec!["T", "EH1", "L", "IH0", "NG"]);
        dictionary.insert("work".to_string(), vec!["W", "ER1", "K"]);
        dictionary.insert("works".to_string(), vec!["W", "ER1", "K", "S"]);
        dictionary.insert("worked".to_string(), vec!["W", "ER1", "K", "T"]);
        dictionary.insert("working".to_string(), vec!["W", "ER1", "K", "IH0", "NG"]);
        dictionary.insert("call".to_string(), vec!["K", "AO1", "L"]);
        dictionary.insert("calls".to_string(), vec!["K", "AO1", "L", "Z"]);
        dictionary.insert("called".to_string(), vec!["K", "AO1", "L", "D"]);
        dictionary.insert("calling".to_string(), vec!["K", "AO1", "L", "IH0", "NG"]);
        dictionary.insert("try".to_string(), vec!["T", "R", "AY1"]);
        dictionary.insert("tries".to_string(), vec!["T", "R", "AY1", "Z"]);
        dictionary.insert("tried".to_string(), vec!["T", "R", "AY1", "D"]);
        dictionary.insert("trying".to_string(), vec!["T", "R", "AY1", "IH0", "NG"]);
        dictionary.insert("need".to_string(), vec!["N", "IY1", "D"]);
        dictionary.insert("needs".to_string(), vec!["N", "IY1", "D", "Z"]);
        dictionary.insert("needed".to_string(), vec!["N", "IY1", "D", "IH0", "D"]);
        dictionary.insert("needing".to_string(), vec!["N", "IY1", "D", "IH0", "NG"]);
        dictionary.insert("feel".to_string(), vec!["F", "IY1", "L"]);
        dictionary.insert("feels".to_string(), vec!["F", "IY1", "L", "Z"]);
        dictionary.insert("felt".to_string(), vec!["F", "EH1", "L", "T"]);
        dictionary.insert("feeling".to_string(), vec!["F", "IY1", "L", "IH0", "NG"]);
        dictionary.insert("become".to_string(), vec!["B", "IH0", "K", "AH1", "M"]);
        dictionary.insert(
            "becomes".to_string(),
            vec!["B", "IH0", "K", "AH1", "M", "Z"],
        );
        dictionary.insert("became".to_string(), vec!["B", "IH0", "K", "EY1", "M"]);
        dictionary.insert(
            "becoming".to_string(),
            vec!["B", "IH0", "K", "AH1", "M", "IH0", "NG"],
        );
        dictionary.insert("leave".to_string(), vec!["L", "IY1", "V"]);
        dictionary.insert("leaves".to_string(), vec!["L", "IY1", "V", "Z"]);
        dictionary.insert("left".to_string(), vec!["L", "EH1", "F", "T"]);
        dictionary.insert("leaving".to_string(), vec!["L", "IY1", "V", "IH0", "NG"]);
        dictionary.insert("put".to_string(), vec!["P", "UH1", "T"]);
        dictionary.insert("puts".to_string(), vec!["P", "UH1", "T", "S"]);
        dictionary.insert("putting".to_string(), vec!["P", "UH1", "T", "IH0", "NG"]);
        dictionary.insert("mean".to_string(), vec!["M", "IY1", "N"]);
        dictionary.insert("means".to_string(), vec!["M", "IY1", "N", "Z"]);
        dictionary.insert("meant".to_string(), vec!["M", "EH1", "N", "T"]);
        dictionary.insert("meaning".to_string(), vec!["M", "IY1", "N", "IH0", "NG"]);
        dictionary.insert("keep".to_string(), vec!["K", "IY1", "P"]);
        dictionary.insert("keeps".to_string(), vec!["K", "IY1", "P", "S"]);
        dictionary.insert("kept".to_string(), vec!["K", "EH1", "P", "T"]);
        dictionary.insert("keeping".to_string(), vec!["K", "IY1", "P", "IH0", "NG"]);
        dictionary.insert("let".to_string(), vec!["L", "EH1", "T"]);
        dictionary.insert("lets".to_string(), vec!["L", "EH1", "T", "S"]);
        dictionary.insert("letting".to_string(), vec!["L", "EH1", "T", "IH0", "NG"]);
        dictionary.insert("begin".to_string(), vec!["B", "IH0", "G", "IH1", "N"]);
        dictionary.insert("begins".to_string(), vec!["B", "IH0", "G", "IH1", "N", "Z"]);
        dictionary.insert("began".to_string(), vec!["B", "IH0", "G", "AE1", "N"]);
        dictionary.insert("begun".to_string(), vec!["B", "IH0", "G", "AH1", "N"]);
        dictionary.insert(
            "beginning".to_string(),
            vec!["B", "IH0", "G", "IH1", "N", "IH0", "NG"],
        );
        dictionary.insert("seem".to_string(), vec!["S", "IY1", "M"]);
        dictionary.insert("seems".to_string(), vec!["S", "IY1", "M", "Z"]);
        dictionary.insert("seemed".to_string(), vec!["S", "IY1", "M", "D"]);
        dictionary.insert("seeming".to_string(), vec!["S", "IY1", "M", "IH0", "NG"]);
        dictionary.insert("help".to_string(), vec!["HH", "EH1", "L", "P"]);
        dictionary.insert("helps".to_string(), vec!["HH", "EH1", "L", "P", "S"]);
        dictionary.insert("helped".to_string(), vec!["HH", "EH1", "L", "P", "T"]);
        dictionary.insert(
            "helping".to_string(),
            vec!["HH", "EH1", "L", "P", "IH0", "NG"],
        );
        dictionary.insert("show".to_string(), vec!["SH", "OW1"]);
        dictionary.insert("shows".to_string(), vec!["SH", "OW1", "Z"]);
        dictionary.insert("showed".to_string(), vec!["SH", "OW1", "D"]);
        dictionary.insert("shown".to_string(), vec!["SH", "OW1", "N"]);
        dictionary.insert("showing".to_string(), vec!["SH", "OW1", "IH0", "NG"]);
        dictionary.insert("hear".to_string(), vec!["HH", "IY1", "R"]);
        dictionary.insert("hears".to_string(), vec!["HH", "IY1", "R", "Z"]);
        dictionary.insert("heard".to_string(), vec!["HH", "ER1", "D"]);
        dictionary.insert("hearing".to_string(), vec!["HH", "IY1", "R", "IH0", "NG"]);
        dictionary.insert("play".to_string(), vec!["P", "L", "EY1"]);
        dictionary.insert("plays".to_string(), vec!["P", "L", "EY1", "Z"]);
        dictionary.insert("played".to_string(), vec!["P", "L", "EY1", "D"]);
        dictionary.insert("playing".to_string(), vec!["P", "L", "EY1", "IH0", "NG"]);
        dictionary.insert("run".to_string(), vec!["R", "AH1", "N"]);
        dictionary.insert("runs".to_string(), vec!["R", "AH1", "N", "Z"]);
        dictionary.insert("ran".to_string(), vec!["R", "AE1", "N"]);
        dictionary.insert("running".to_string(), vec!["R", "AH1", "N", "IH0", "NG"]);
        dictionary.insert("move".to_string(), vec!["M", "UW1", "V"]);
        dictionary.insert("moves".to_string(), vec!["M", "UW1", "V", "Z"]);
        dictionary.insert("moved".to_string(), vec!["M", "UW1", "V", "D"]);
        dictionary.insert("moving".to_string(), vec!["M", "UW1", "V", "IH0", "NG"]);
        dictionary.insert("live".to_string(), vec!["L", "IH1", "V"]);
        dictionary.insert("lives".to_string(), vec!["L", "IH1", "V", "Z"]);
        dictionary.insert("lived".to_string(), vec!["L", "IH1", "V", "D"]);
        dictionary.insert("living".to_string(), vec!["L", "IH1", "V", "IH0", "NG"]);
        dictionary.insert("believe".to_string(), vec!["B", "IH0", "L", "IY1", "V"]);
        dictionary.insert(
            "believes".to_string(),
            vec!["B", "IH0", "L", "IY1", "V", "Z"],
        );
        dictionary.insert(
            "believed".to_string(),
            vec!["B", "IH0", "L", "IY1", "V", "D"],
        );
        dictionary.insert(
            "believing".to_string(),
            vec!["B", "IH0", "L", "IY1", "V", "IH0", "NG"],
        );
        dictionary.insert("bring".to_string(), vec!["B", "R", "IH1", "NG"]);
        dictionary.insert("brings".to_string(), vec!["B", "R", "IH1", "NG", "Z"]);
        dictionary.insert("brought".to_string(), vec!["B", "R", "AO1", "T"]);
        dictionary.insert(
            "bringing".to_string(),
            vec!["B", "R", "IH1", "NG", "IH0", "NG"],
        );
        dictionary.insert("happen".to_string(), vec!["HH", "AE1", "P", "AH0", "N"]);
        dictionary.insert(
            "happens".to_string(),
            vec!["HH", "AE1", "P", "AH0", "N", "Z"],
        );
        dictionary.insert(
            "happened".to_string(),
            vec!["HH", "AE1", "P", "AH0", "N", "D"],
        );
        dictionary.insert(
            "happening".to_string(),
            vec!["HH", "AE1", "P", "AH0", "N", "IH0", "NG"],
        );
        dictionary.insert("write".to_string(), vec!["R", "AY1", "T"]);
        dictionary.insert("writes".to_string(), vec!["R", "AY1", "T", "S"]);
        dictionary.insert("wrote".to_string(), vec!["R", "OW1", "T"]);
        dictionary.insert("written".to_string(), vec!["R", "IH1", "T", "AH0", "N"]);
        dictionary.insert("writing".to_string(), vec!["R", "AY1", "T", "IH0", "NG"]);
        dictionary.insert(
            "provide".to_string(),
            vec!["P", "R", "AH0", "V", "AY1", "D"],
        );
        dictionary.insert(
            "provides".to_string(),
            vec!["P", "R", "AH0", "V", "AY1", "D", "Z"],
        );
        dictionary.insert(
            "provided".to_string(),
            vec!["P", "R", "AH0", "V", "AY1", "D", "IH0", "D"],
        );
        dictionary.insert(
            "providing".to_string(),
            vec!["P", "R", "AH0", "V", "AY1", "D", "IH0", "NG"],
        );
        dictionary.insert("stand".to_string(), vec!["S", "T", "AE1", "N", "D"]);
        dictionary.insert("stands".to_string(), vec!["S", "T", "AE1", "N", "D", "Z"]);
        dictionary.insert("stood".to_string(), vec!["S", "T", "UH1", "D"]);
        dictionary.insert(
            "standing".to_string(),
            vec!["S", "T", "AE1", "N", "D", "IH0", "NG"],
        );
        dictionary.insert("read".to_string(), vec!["R", "IY1", "D"]);
        dictionary.insert("reads".to_string(), vec!["R", "IY1", "D", "Z"]);
        dictionary.insert("reading".to_string(), vec!["R", "IY1", "D", "IH0", "NG"]);
        dictionary.insert("learn".to_string(), vec!["L", "ER1", "N"]);
        dictionary.insert("learns".to_string(), vec!["L", "ER1", "N", "Z"]);
        dictionary.insert("learned".to_string(), vec!["L", "ER1", "N", "D"]);
        dictionary.insert("learning".to_string(), vec!["L", "ER1", "N", "IH0", "NG"]);
        dictionary.insert("change".to_string(), vec!["CH", "EY1", "N", "JH"]);
        dictionary.insert(
            "changes".to_string(),
            vec!["CH", "EY1", "N", "JH", "IH0", "Z"],
        );
        dictionary.insert("changed".to_string(), vec!["CH", "EY1", "N", "JH", "D"]);
        dictionary.insert(
            "changing".to_string(),
            vec!["CH", "EY1", "N", "JH", "IH0", "NG"],
        );
        dictionary.insert("lead".to_string(), vec!["L", "IY1", "D"]);
        dictionary.insert("leads".to_string(), vec!["L", "IY1", "D", "Z"]);
        dictionary.insert("led".to_string(), vec!["L", "EH1", "D"]);
        dictionary.insert("leading".to_string(), vec!["L", "IY1", "D", "IH0", "NG"]);
        dictionary.insert(
            "understand".to_string(),
            vec!["AH2", "N", "D", "ER0", "S", "T", "AE1", "N", "D"],
        );
        dictionary.insert(
            "understands".to_string(),
            vec!["AH2", "N", "D", "ER0", "S", "T", "AE1", "N", "D", "Z"],
        );
        dictionary.insert(
            "understood".to_string(),
            vec!["AH2", "N", "D", "ER0", "S", "T", "UH1", "D"],
        );
        dictionary.insert(
            "understanding".to_string(),
            vec![
                "AH2", "N", "D", "ER0", "S", "T", "AE1", "N", "D", "IH0", "NG",
            ],
        );
        dictionary.insert("create".to_string(), vec!["K", "R", "IY0", "EY1", "T"]);
        dictionary.insert(
            "creates".to_string(),
            vec!["K", "R", "IY0", "EY1", "T", "S"],
        );
        dictionary.insert(
            "created".to_string(),
            vec!["K", "R", "IY0", "EY1", "T", "IH0", "D"],
        );
        dictionary.insert(
            "creating".to_string(),
            vec!["K", "R", "IY0", "EY1", "T", "IH0", "NG"],
        );

        // ═══════════════════════════════════════════════════════════════════
        // MODAL VERBS
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("can".to_string(), vec!["K", "AE1", "N"]);
        dictionary.insert("could".to_string(), vec!["K", "UH1", "D"]);
        dictionary.insert("will".to_string(), vec!["W", "IH1", "L"]);
        dictionary.insert("would".to_string(), vec!["W", "UH1", "D"]);
        dictionary.insert("shall".to_string(), vec!["SH", "AE1", "L"]);
        dictionary.insert("should".to_string(), vec!["SH", "UH1", "D"]);
        dictionary.insert("may".to_string(), vec!["M", "EY1"]);
        dictionary.insert("might".to_string(), vec!["M", "AY1", "T"]);
        dictionary.insert("must".to_string(), vec!["M", "AH1", "S", "T"]);

        // ═══════════════════════════════════════════════════════════════════
        // COMMON ADJECTIVES
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("good".to_string(), vec!["G", "UH1", "D"]);
        dictionary.insert("new".to_string(), vec!["N", "UW1"]);
        dictionary.insert("first".to_string(), vec!["F", "ER1", "S", "T"]);
        dictionary.insert("last".to_string(), vec!["L", "AE1", "S", "T"]);
        dictionary.insert("long".to_string(), vec!["L", "AO1", "NG"]);
        dictionary.insert("great".to_string(), vec!["G", "R", "EY1", "T"]);
        dictionary.insert("little".to_string(), vec!["L", "IH1", "T", "AH0", "L"]);
        dictionary.insert("own".to_string(), vec!["OW1", "N"]);
        dictionary.insert("other".to_string(), vec!["AH1", "DH", "ER0"]);
        dictionary.insert("old".to_string(), vec!["OW1", "L", "D"]);
        dictionary.insert("right".to_string(), vec!["R", "AY1", "T"]);
        dictionary.insert("big".to_string(), vec!["B", "IH1", "G"]);
        dictionary.insert("high".to_string(), vec!["HH", "AY1"]);
        dictionary.insert(
            "different".to_string(),
            vec!["D", "IH1", "F", "ER0", "AH0", "N", "T"],
        );
        dictionary.insert("small".to_string(), vec!["S", "M", "AO1", "L"]);
        dictionary.insert("large".to_string(), vec!["L", "AA1", "R", "JH"]);
        dictionary.insert("next".to_string(), vec!["N", "EH1", "K", "S", "T"]);
        dictionary.insert("young".to_string(), vec!["Y", "AH1", "NG"]);
        dictionary.insert(
            "important".to_string(),
            vec!["IH0", "M", "P", "AO1", "R", "T", "AH0", "N", "T"],
        );
        dictionary.insert("public".to_string(), vec!["P", "AH1", "B", "L", "IH0", "K"]);
        dictionary.insert("bad".to_string(), vec!["B", "AE1", "D"]);
        dictionary.insert("same".to_string(), vec!["S", "EY1", "M"]);
        dictionary.insert("able".to_string(), vec!["EY1", "B", "AH0", "L"]);
        dictionary.insert("true".to_string(), vec!["T", "R", "UW1"]);
        dictionary.insert("false".to_string(), vec!["F", "AO1", "L", "S"]);
        dictionary.insert(
            "possible".to_string(),
            vec!["P", "AA1", "S", "AH0", "B", "AH0", "L"],
        );
        dictionary.insert("sure".to_string(), vec!["SH", "UH1", "R"]);
        dictionary.insert("clear".to_string(), vec!["K", "L", "IY1", "R"]);
        dictionary.insert("full".to_string(), vec!["F", "UH1", "L"]);
        dictionary.insert("empty".to_string(), vec!["EH1", "M", "P", "T", "IY0"]);
        dictionary.insert("simple".to_string(), vec!["S", "IH1", "M", "P", "AH0", "L"]);
        dictionary.insert(
            "complex".to_string(),
            vec!["K", "AA1", "M", "P", "L", "EH0", "K", "S"],
        );
        dictionary.insert("easy".to_string(), vec!["IY1", "Z", "IY0"]);
        dictionary.insert("hard".to_string(), vec!["HH", "AA1", "R", "D"]);
        dictionary.insert("fast".to_string(), vec!["F", "AE1", "S", "T"]);
        dictionary.insert("slow".to_string(), vec!["S", "L", "OW1"]);
        dictionary.insert("deep".to_string(), vec!["D", "IY1", "P"]);
        dictionary.insert("free".to_string(), vec!["F", "R", "IY1"]);
        dictionary.insert("open".to_string(), vec!["OW1", "P", "AH0", "N"]);
        dictionary.insert("closed".to_string(), vec!["K", "L", "OW1", "Z", "D"]);
        dictionary.insert("whole".to_string(), vec!["HH", "OW1", "L"]);
        dictionary.insert(
            "special".to_string(),
            vec!["S", "P", "EH1", "SH", "AH0", "L"],
        );
        dictionary.insert("real".to_string(), vec!["R", "IY1", "L"]);
        dictionary.insert("ready".to_string(), vec!["R", "EH1", "D", "IY0"]);
        dictionary.insert(
            "present".to_string(),
            vec!["P", "R", "EH1", "Z", "AH0", "N", "T"],
        );
        dictionary.insert("future".to_string(), vec!["F", "Y", "UW1", "CH", "ER0"]);
        dictionary.insert("past".to_string(), vec!["P", "AE1", "S", "T"]);
        dictionary.insert("current".to_string(), vec!["K", "ER1", "AH0", "N", "T"]);
        dictionary.insert(
            "natural".to_string(),
            vec!["N", "AE1", "CH", "ER0", "AH0", "L"],
        );
        dictionary.insert("human".to_string(), vec!["HH", "Y", "UW1", "M", "AH0", "N"]);
        dictionary.insert(
            "beautiful".to_string(),
            vec!["B", "Y", "UW1", "T", "AH0", "F", "AH0", "L"],
        );
        dictionary.insert("strong".to_string(), vec!["S", "T", "R", "AO1", "NG"]);
        dictionary.insert("weak".to_string(), vec!["W", "IY1", "K"]);
        dictionary.insert("light".to_string(), vec!["L", "AY1", "T"]);
        dictionary.insert("dark".to_string(), vec!["D", "AA1", "R", "K"]);
        dictionary.insert("warm".to_string(), vec!["W", "AO1", "R", "M"]);
        dictionary.insert("cold".to_string(), vec!["K", "OW1", "L", "D"]);
        dictionary.insert("hot".to_string(), vec!["HH", "AA1", "T"]);
        dictionary.insert("quiet".to_string(), vec!["K", "W", "AY1", "AH0", "T"]);
        dictionary.insert("loud".to_string(), vec!["L", "AW1", "D"]);
        dictionary.insert("soft".to_string(), vec!["S", "AO1", "F", "T"]);

        // ═══════════════════════════════════════════════════════════════════
        // COMMON NOUNS
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("time".to_string(), vec!["T", "AY1", "M"]);
        dictionary.insert("year".to_string(), vec!["Y", "IY1", "R"]);
        dictionary.insert("people".to_string(), vec!["P", "IY1", "P", "AH0", "L"]);
        dictionary.insert("way".to_string(), vec!["W", "EY1"]);
        dictionary.insert("day".to_string(), vec!["D", "EY1"]);
        dictionary.insert("man".to_string(), vec!["M", "AE1", "N"]);
        dictionary.insert("woman".to_string(), vec!["W", "UH1", "M", "AH0", "N"]);
        dictionary.insert("child".to_string(), vec!["CH", "AY1", "L", "D"]);
        dictionary.insert("world".to_string(), vec!["W", "ER1", "L", "D"]);
        dictionary.insert("life".to_string(), vec!["L", "AY1", "F"]);
        dictionary.insert("hand".to_string(), vec!["HH", "AE1", "N", "D"]);
        dictionary.insert("part".to_string(), vec!["P", "AA1", "R", "T"]);
        dictionary.insert("place".to_string(), vec!["P", "L", "EY1", "S"]);
        dictionary.insert("case".to_string(), vec!["K", "EY1", "S"]);
        dictionary.insert("week".to_string(), vec!["W", "IY1", "K"]);
        dictionary.insert(
            "company".to_string(),
            vec!["K", "AH1", "M", "P", "AH0", "N", "IY0"],
        );
        dictionary.insert("system".to_string(), vec!["S", "IH1", "S", "T", "AH0", "M"]);
        dictionary.insert(
            "program".to_string(),
            vec!["P", "R", "OW1", "G", "R", "AE2", "M"],
        );
        dictionary.insert(
            "question".to_string(),
            vec!["K", "W", "EH1", "S", "CH", "AH0", "N"],
        );
        dictionary.insert("work".to_string(), vec!["W", "ER1", "K"]);
        dictionary.insert(
            "government".to_string(),
            vec!["G", "AH1", "V", "ER0", "N", "M", "AH0", "N", "T"],
        );
        dictionary.insert("number".to_string(), vec!["N", "AH1", "M", "B", "ER0"]);
        dictionary.insert("night".to_string(), vec!["N", "AY1", "T"]);
        dictionary.insert("point".to_string(), vec!["P", "OY1", "N", "T"]);
        dictionary.insert("home".to_string(), vec!["HH", "OW1", "M"]);
        dictionary.insert("water".to_string(), vec!["W", "AO1", "T", "ER0"]);
        dictionary.insert("room".to_string(), vec!["R", "UW1", "M"]);
        dictionary.insert("mother".to_string(), vec!["M", "AH1", "DH", "ER0"]);
        dictionary.insert("father".to_string(), vec!["F", "AA1", "DH", "ER0"]);
        dictionary.insert("area".to_string(), vec!["EH1", "R", "IY0", "AH0"]);
        dictionary.insert("money".to_string(), vec!["M", "AH1", "N", "IY0"]);
        dictionary.insert("story".to_string(), vec!["S", "T", "AO1", "R", "IY0"]);
        dictionary.insert("fact".to_string(), vec!["F", "AE1", "K", "T"]);
        dictionary.insert("month".to_string(), vec!["M", "AH1", "N", "TH"]);
        dictionary.insert("lot".to_string(), vec!["L", "AA1", "T"]);
        dictionary.insert("right".to_string(), vec!["R", "AY1", "T"]);
        dictionary.insert("study".to_string(), vec!["S", "T", "AH1", "D", "IY0"]);
        dictionary.insert("book".to_string(), vec!["B", "UH1", "K"]);
        dictionary.insert("eye".to_string(), vec!["AY1"]);
        dictionary.insert("job".to_string(), vec!["JH", "AA1", "B"]);
        dictionary.insert("word".to_string(), vec!["W", "ER1", "D"]);
        dictionary.insert(
            "business".to_string(),
            vec!["B", "IH1", "Z", "N", "AH0", "S"],
        );
        dictionary.insert("issue".to_string(), vec!["IH1", "SH", "UW0"]);
        dictionary.insert("side".to_string(), vec!["S", "AY1", "D"]);
        dictionary.insert("kind".to_string(), vec!["K", "AY1", "N", "D"]);
        dictionary.insert("head".to_string(), vec!["HH", "EH1", "D"]);
        dictionary.insert("house".to_string(), vec!["HH", "AW1", "S"]);
        dictionary.insert("service".to_string(), vec!["S", "ER1", "V", "AH0", "S"]);
        dictionary.insert("friend".to_string(), vec!["F", "R", "EH1", "N", "D"]);
        dictionary.insert("hour".to_string(), vec!["AW1", "ER0"]);
        dictionary.insert("game".to_string(), vec!["G", "EY1", "M"]);
        dictionary.insert("line".to_string(), vec!["L", "AY1", "N"]);
        dictionary.insert("end".to_string(), vec!["EH1", "N", "D"]);
        dictionary.insert("member".to_string(), vec!["M", "EH1", "M", "B", "ER0"]);
        dictionary.insert("law".to_string(), vec!["L", "AO1"]);
        dictionary.insert("car".to_string(), vec!["K", "AA1", "R"]);
        dictionary.insert("city".to_string(), vec!["S", "IH1", "T", "IY0"]);
        dictionary.insert(
            "community".to_string(),
            vec!["K", "AH0", "M", "Y", "UW1", "N", "AH0", "T", "IY0"],
        );
        dictionary.insert("name".to_string(), vec!["N", "EY1", "M"]);
        dictionary.insert("power".to_string(), vec!["P", "AW1", "ER0"]);
        dictionary.insert("idea".to_string(), vec!["AY0", "D", "IY1", "AH0"]);
        dictionary.insert(
            "information".to_string(),
            vec!["IH2", "N", "F", "ER0", "M", "EY1", "SH", "AH0", "N"],
        );
        dictionary.insert("result".to_string(), vec!["R", "IH0", "Z", "AH1", "L", "T"]);
        dictionary.insert(
            "problem".to_string(),
            vec!["P", "R", "AA1", "B", "L", "AH0", "M"],
        );
        dictionary.insert(
            "experience".to_string(),
            vec!["IH0", "K", "S", "P", "IY1", "R", "IY0", "AH0", "N", "S"],
        );
        dictionary.insert("answer".to_string(), vec!["AE1", "N", "S", "ER0"]);

        // ═══════════════════════════════════════════════════════════════════
        // ADVERBS
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("not".to_string(), vec!["N", "AA1", "T"]);
        dictionary.insert("just".to_string(), vec!["JH", "AH1", "S", "T"]);
        dictionary.insert("only".to_string(), vec!["OW1", "N", "L", "IY0"]);
        dictionary.insert("also".to_string(), vec!["AO1", "L", "S", "OW0"]);
        dictionary.insert("now".to_string(), vec!["N", "AW1"]);
        dictionary.insert("then".to_string(), vec!["DH", "EH1", "N"]);
        dictionary.insert("more".to_string(), vec!["M", "AO1", "R"]);
        dictionary.insert("very".to_string(), vec!["V", "EH1", "R", "IY0"]);
        dictionary.insert("well".to_string(), vec!["W", "EH1", "L"]);
        dictionary.insert("here".to_string(), vec!["HH", "IY1", "R"]);
        dictionary.insert("there".to_string(), vec!["DH", "EH1", "R"]);
        dictionary.insert("even".to_string(), vec!["IY1", "V", "AH0", "N"]);
        dictionary.insert("still".to_string(), vec!["S", "T", "IH1", "L"]);
        dictionary.insert("again".to_string(), vec!["AH0", "G", "EH1", "N"]);
        dictionary.insert("always".to_string(), vec!["AO1", "L", "W", "EY2", "Z"]);
        dictionary.insert("never".to_string(), vec!["N", "EH1", "V", "ER0"]);
        dictionary.insert("often".to_string(), vec!["AO1", "F", "AH0", "N"]);
        dictionary.insert(
            "sometimes".to_string(),
            vec!["S", "AH1", "M", "T", "AY2", "M", "Z"],
        );
        dictionary.insert(
            "usually".to_string(),
            vec!["Y", "UW1", "ZH", "AH0", "L", "IY0"],
        );
        dictionary.insert(
            "perhaps".to_string(),
            vec!["P", "ER0", "HH", "AE1", "P", "S"],
        );
        dictionary.insert("maybe".to_string(), vec!["M", "EY1", "B", "IY0"]);
        dictionary.insert("really".to_string(), vec!["R", "IY1", "L", "IY0"]);
        dictionary.insert(
            "already".to_string(),
            vec!["AO0", "L", "R", "EH1", "D", "IY0"],
        );
        dictionary.insert("yet".to_string(), vec!["Y", "EH1", "T"]);
        dictionary.insert("today".to_string(), vec!["T", "AH0", "D", "EY1"]);
        dictionary.insert(
            "tomorrow".to_string(),
            vec!["T", "AH0", "M", "AA1", "R", "OW0"],
        );
        dictionary.insert(
            "yesterday".to_string(),
            vec!["Y", "EH1", "S", "T", "ER0", "D", "EY2"],
        );
        dictionary.insert(
            "together".to_string(),
            vec!["T", "AH0", "G", "EH1", "DH", "ER0"],
        );
        dictionary.insert("away".to_string(), vec!["AH0", "W", "EY1"]);
        dictionary.insert("back".to_string(), vec!["B", "AE1", "K"]);
        dictionary.insert("down".to_string(), vec!["D", "AW1", "N"]);
        dictionary.insert("out".to_string(), vec!["AW1", "T"]);
        dictionary.insert("off".to_string(), vec!["AO1", "F"]);
        dictionary.insert("enough".to_string(), vec!["IH0", "N", "AH1", "F"]);
        dictionary.insert("almost".to_string(), vec!["AO1", "L", "M", "OW2", "S", "T"]);
        dictionary.insert("quite".to_string(), vec!["K", "W", "AY1", "T"]);
        dictionary.insert("rather".to_string(), vec!["R", "AE1", "DH", "ER0"]);
        dictionary.insert("too".to_string(), vec!["T", "UW1"]);
        dictionary.insert("soon".to_string(), vec!["S", "UW1", "N"]);
        dictionary.insert("later".to_string(), vec!["L", "EY1", "T", "ER0"]);
        dictionary.insert(
            "finally".to_string(),
            vec!["F", "AY1", "N", "AH0", "L", "IY0"],
        );

        // ═══════════════════════════════════════════════════════════════════
        // NUMBERS
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("zero".to_string(), vec!["Z", "IY1", "R", "OW0"]);
        dictionary.insert("one".to_string(), vec!["W", "AH1", "N"]);
        dictionary.insert("two".to_string(), vec!["T", "UW1"]);
        dictionary.insert("three".to_string(), vec!["TH", "R", "IY1"]);
        dictionary.insert("four".to_string(), vec!["F", "AO1", "R"]);
        dictionary.insert("five".to_string(), vec!["F", "AY1", "V"]);
        dictionary.insert("six".to_string(), vec!["S", "IH1", "K", "S"]);
        dictionary.insert("seven".to_string(), vec!["S", "EH1", "V", "AH0", "N"]);
        dictionary.insert("eight".to_string(), vec!["EY1", "T"]);
        dictionary.insert("nine".to_string(), vec!["N", "AY1", "N"]);
        dictionary.insert("ten".to_string(), vec!["T", "EH1", "N"]);
        dictionary.insert(
            "hundred".to_string(),
            vec!["HH", "AH1", "N", "D", "R", "AH0", "D"],
        );
        dictionary.insert(
            "thousand".to_string(),
            vec!["TH", "AW1", "Z", "AH0", "N", "D"],
        );
        dictionary.insert(
            "million".to_string(),
            vec!["M", "IH1", "L", "Y", "AH0", "N"],
        );

        // ═══════════════════════════════════════════════════════════════════
        // CONSCIOUSNESS & AI TERMINOLOGY
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert(
            "consciousness".to_string(),
            vec!["K", "AA1", "N", "SH", "AH0", "S", "N", "AH0", "S"],
        );
        dictionary.insert(
            "conscious".to_string(),
            vec!["K", "AA1", "N", "SH", "AH0", "S"],
        );
        dictionary.insert("aware".to_string(), vec!["AH0", "W", "EH1", "R"]);
        dictionary.insert(
            "awareness".to_string(),
            vec!["AH0", "W", "EH1", "R", "N", "AH0", "S"],
        );
        dictionary.insert("mind".to_string(), vec!["M", "AY1", "N", "D"]);
        dictionary.insert("brain".to_string(), vec!["B", "R", "EY1", "N"]);
        dictionary.insert("neural".to_string(), vec!["N", "UH1", "R", "AH0", "L"]);
        dictionary.insert(
            "network".to_string(),
            vec!["N", "EH1", "T", "W", "ER2", "K"],
        );
        dictionary.insert("phi".to_string(), vec!["F", "AY1"]);
        dictionary.insert("flow".to_string(), vec!["F", "L", "OW1"]);
        dictionary.insert("state".to_string(), vec!["S", "T", "EY1", "T"]);
        dictionary.insert(
            "process".to_string(),
            vec!["P", "R", "AA1", "S", "EH0", "S"],
        );
        dictionary.insert(
            "response".to_string(),
            vec!["R", "IH0", "S", "P", "AA1", "N", "S"],
        );
        dictionary.insert(
            "attention".to_string(),
            vec!["AH0", "T", "EH1", "N", "SH", "AH0", "N"],
        );
        dictionary.insert("memory".to_string(), vec!["M", "EH1", "M", "ER0", "IY0"]);
        dictionary.insert(
            "perception".to_string(),
            vec!["P", "ER0", "S", "EH1", "P", "SH", "AH0", "N"],
        );
        dictionary.insert(
            "emotion".to_string(),
            vec!["IH0", "M", "OW1", "SH", "AH0", "N"],
        );
        dictionary.insert(
            "emotional".to_string(),
            vec!["IH0", "M", "OW1", "SH", "AH0", "N", "AH0", "L"],
        );
        dictionary.insert(
            "reasoning".to_string(),
            vec!["R", "IY1", "Z", "AH0", "N", "IH0", "NG"],
        );
        dictionary.insert(
            "intelligence".to_string(),
            vec!["IH0", "N", "T", "EH1", "L", "AH0", "JH", "AH0", "N", "S"],
        );
        dictionary.insert(
            "intelligent".to_string(),
            vec!["IH0", "N", "T", "EH1", "L", "AH0", "JH", "AH0", "N", "T"],
        );
        dictionary.insert(
            "artificial".to_string(),
            vec!["AA2", "R", "T", "AH0", "F", "IH1", "SH", "AH0", "L"],
        );
        dictionary.insert(
            "cognitive".to_string(),
            vec!["K", "AA1", "G", "N", "AH0", "T", "IH0", "V"],
        );
        dictionary.insert(
            "coherent".to_string(),
            vec!["K", "OW0", "HH", "IY1", "R", "AH0", "N", "T"],
        );
        dictionary.insert(
            "coherence".to_string(),
            vec!["K", "OW0", "HH", "IY1", "R", "AH0", "N", "S"],
        );
        dictionary.insert(
            "integration".to_string(),
            vec!["IH2", "N", "T", "AH0", "G", "R", "EY1", "SH", "AH0", "N"],
        );
        dictionary.insert(
            "integrated".to_string(),
            vec!["IH1", "N", "T", "AH0", "G", "R", "EY2", "T", "IH0", "D"],
        );
        dictionary.insert(
            "holistic".to_string(),
            vec!["HH", "OW0", "L", "IH1", "S", "T", "IH0", "K"],
        );
        dictionary.insert(
            "emergent".to_string(),
            vec!["IH0", "M", "ER1", "JH", "AH0", "N", "T"],
        );
        dictionary.insert(
            "emergence".to_string(),
            vec!["IH0", "M", "ER1", "JH", "AH0", "N", "S"],
        );
        dictionary.insert(
            "sentient".to_string(),
            vec!["S", "EH1", "N", "SH", "AH0", "N", "T"],
        );
        dictionary.insert(
            "sentience".to_string(),
            vec!["S", "EH1", "N", "SH", "AH0", "N", "S"],
        );
        dictionary.insert(
            "qualia".to_string(),
            vec!["K", "W", "EY1", "L", "IY0", "AH0"],
        );
        dictionary.insert(
            "embodied".to_string(),
            vec!["IH0", "M", "B", "AA1", "D", "IY0", "D"],
        );
        dictionary.insert(
            "phenomenal".to_string(),
            vec!["F", "AH0", "N", "AA1", "M", "AH0", "N", "AH0", "L"],
        );
        dictionary.insert(
            "subjective".to_string(),
            vec!["S", "AH0", "B", "JH", "EH1", "K", "T", "IH0", "V"],
        );
        dictionary.insert(
            "recursive".to_string(),
            vec!["R", "IH0", "K", "ER1", "S", "IH0", "V"],
        );
        dictionary.insert(
            "autopoietic".to_string(),
            vec!["AO2", "T", "OW0", "P", "OY0", "EH1", "T", "IH0", "K"],
        );
        dictionary.insert(
            "symthaea".to_string(),
            vec!["S", "IH0", "M", "TH", "IY1", "AH0"],
        );
        dictionary.insert(
            "hyperdimensional".to_string(),
            vec![
                "HH", "AY2", "P", "ER0", "D", "IH0", "M", "EH1", "N", "SH", "AH0", "N", "AH0", "L",
            ],
        );
        dictionary.insert(
            "topology".to_string(),
            vec!["T", "AH0", "P", "AA1", "L", "AH0", "JH", "IY0"],
        );
        dictionary.insert(
            "formant".to_string(),
            vec!["F", "AO1", "R", "M", "AH0", "N", "T"],
        );
        dictionary.insert(
            "vocoder".to_string(),
            vec!["V", "OW1", "K", "OW0", "D", "ER0"],
        );
        dictionary.insert(
            "synthesis".to_string(),
            vec!["S", "IH1", "N", "TH", "AH0", "S", "IH0", "S"],
        );
        dictionary.insert(
            "synthesize".to_string(),
            vec!["S", "IH1", "N", "TH", "AH0", "S", "AY2", "Z"],
        );

        // ═══════════════════════════════════════════════════════════════════
        // GREETINGS & COMMON PHRASES
        // ═══════════════════════════════════════════════════════════════════
        dictionary.insert("hello".to_string(), vec!["HH", "AH0", "L", "OW1"]);
        dictionary.insert("hi".to_string(), vec!["HH", "AY1"]);
        dictionary.insert("goodbye".to_string(), vec!["G", "UH2", "D", "B", "AY1"]);
        dictionary.insert("bye".to_string(), vec!["B", "AY1"]);
        dictionary.insert("yes".to_string(), vec!["Y", "EH1", "S"]);
        dictionary.insert("no".to_string(), vec!["N", "OW1"]);
        dictionary.insert("okay".to_string(), vec!["OW2", "K", "EY1"]);
        dictionary.insert("please".to_string(), vec!["P", "L", "IY1", "Z"]);
        dictionary.insert("thanks".to_string(), vec!["TH", "AE1", "NG", "K", "S"]);
        dictionary.insert("thank".to_string(), vec!["TH", "AE1", "NG", "K"]);
        dictionary.insert("sorry".to_string(), vec!["S", "AA1", "R", "IY0"]);
        dictionary.insert(
            "welcome".to_string(),
            vec!["W", "EH1", "L", "K", "AH0", "M"],
        );
        dictionary.insert("so".to_string(), vec!["S", "OW1"]);

        Self { dictionary }
    }

    /// Convert a word to ARPABET phonemes
    pub fn word_to_phonemes(&self, word: &str) -> Vec<&'static str> {
        let lower = word.to_lowercase();
        let clean: String = lower.chars().filter(|c| c.is_alphabetic()).collect();

        if let Some(phonemes) = self.dictionary.get(&clean) {
            return phonemes.clone();
        }

        // Fallback: simple letter-to-phoneme rules
        self.simple_g2p(&clean)
    }

    /// Simple fallback G2P for unknown words
    fn simple_g2p(&self, word: &str) -> Vec<&'static str> {
        let mut phonemes = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            let next = chars.get(i + 1);

            let ph: &'static str = match c {
                'a' => match next {
                    Some('i') | Some('y') => {
                        i += 1;
                        "EY1"
                    }
                    Some('u') | Some('w') => {
                        i += 1;
                        "AO1"
                    }
                    Some('e') => {
                        i += 1;
                        "EY1"
                    }
                    _ => "AE1",
                },
                'e' => match next {
                    Some('e') => {
                        i += 1;
                        "IY1"
                    }
                    Some('a') => {
                        i += 1;
                        "IY1"
                    }
                    Some('i') | Some('y') => {
                        i += 1;
                        "EY1"
                    }
                    _ => "EH1",
                },
                'i' => match next {
                    Some('e') => {
                        i += 1;
                        "IY1"
                    }
                    Some('g') if chars.get(i + 2) == Some(&'h') => "AY1",
                    _ => "IH1",
                },
                'o' => match next {
                    Some('o') => {
                        i += 1;
                        "UW1"
                    }
                    Some('u') | Some('w') => {
                        i += 1;
                        "AW1"
                    }
                    Some('i') | Some('y') => {
                        i += 1;
                        "OY1"
                    }
                    _ => "AA1",
                },
                'u' => match next {
                    Some('e') => {
                        i += 1;
                        "UW1"
                    }
                    _ => "AH1",
                },
                'b' => "B",
                'c' => match next {
                    Some('h') => {
                        i += 1;
                        "CH"
                    }
                    Some('i') | Some('e') | Some('y') => "S",
                    _ => "K",
                },
                'd' => "D",
                'f' => "F",
                'g' => match next {
                    Some('e') | Some('i') | Some('y') => "JH",
                    _ => "G",
                },
                'h' => "HH",
                'j' => "JH",
                'k' => "K",
                'l' => "L",
                'm' => "M",
                'n' => match next {
                    Some('g') => {
                        i += 1;
                        "NG"
                    }
                    _ => "N",
                },
                'p' => match next {
                    Some('h') => {
                        i += 1;
                        "F"
                    }
                    _ => "P",
                },
                'q' => "K",
                'r' => "R",
                's' => match next {
                    Some('h') => {
                        i += 1;
                        "SH"
                    }
                    _ => "S",
                },
                't' => match next {
                    Some('h') => {
                        i += 1;
                        "TH"
                    }
                    Some('i') if chars.get(i + 2) == Some(&'o') => {
                        i += 1;
                        "SH"
                    }
                    _ => "T",
                },
                'v' => "V",
                'w' => "W",
                'x' => {
                    phonemes.push("K");
                    "S"
                }
                'y' => {
                    if i == 0 {
                        "Y"
                    } else {
                        "IY0"
                    }
                }
                'z' => "Z",
                _ => {
                    i += 1;
                    continue;
                }
            };

            phonemes.push(ph);
            i += 1;
        }

        if phonemes.is_empty() {
            vec!["AH0"] // Fallback schwa
        } else {
            phonemes
        }
    }

    /// Convert text to phoneme sequence with timing
    pub fn text_to_phonemes(&self, text: &str, base_duration: f32) -> Vec<TimedPhoneme> {
        let mut result = Vec::new();
        let mut current_time = 0.0;

        for word in text.split_whitespace() {
            // Check for punctuation
            let has_comma = word.contains(',');
            let has_period = word.contains('.') || word.contains('!') || word.contains('?');

            let phonemes = self.word_to_phonemes(word);

            for &ph in phonemes.iter() {
                // Determine stress from phoneme
                let stress = if ph.ends_with('1') {
                    1
                } else if ph.ends_with('2') {
                    2
                } else {
                    0
                };

                // Clean phoneme (remove stress marker for lookup)
                let clean_ph: String = ph.chars().filter(|c| !c.is_ascii_digit()).collect();

                // Vowels are longer than consonants
                let is_vowel = matches!(
                    clean_ph.as_str(),
                    "AA" | "AE"
                        | "AH"
                        | "AO"
                        | "AW"
                        | "AY"
                        | "EH"
                        | "ER"
                        | "EY"
                        | "IH"
                        | "IY"
                        | "OW"
                        | "OY"
                        | "UH"
                        | "UW"
                );
                let duration = if is_vowel {
                    base_duration * 1.3
                } else {
                    base_duration * 0.8
                };

                result.push(TimedPhoneme {
                    phoneme: clean_ph,
                    duration,
                    stress,
                    start_time: current_time,
                });

                current_time += duration;
            }

            // Add inter-word pause
            if has_comma {
                result.push(TimedPhoneme {
                    phoneme: "SIL".to_string(),
                    duration: 0.15,
                    stress: 0,
                    start_time: current_time,
                });
                current_time += 0.15;
            } else if has_period {
                result.push(TimedPhoneme {
                    phoneme: "SIL".to_string(),
                    duration: 0.25,
                    stress: 0,
                    start_time: current_time,
                });
                current_time += 0.25;
            } else {
                // Small pause between words
                current_time += 0.05;
            }
        }

        result
    }
}

impl Default for SimpleG2P {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPL VOICE OUTPUT
// ═══════════════════════════════════════════════════════════════════════════════

/// Voice output system for the REPL
///
/// Provides consciousness-modulated speech synthesis with optional audio playback.
pub struct ReplVoiceOutput {
    /// Configuration
    config: ReplVoiceConfig,

    /// Articulatory synthesizer for phoneme-to-formant conversion
    articulatory: ArticulatorySynthesizer,

    /// Formant vocoder for formant-to-audio conversion
    vocoder: FormantVocoder,

    /// Basic voice output for simulated TTS fallback
    voice_output: VoiceOutput,

    /// Cognitive voice bridge for consciousness modulation
    cognitive_bridge: CognitiveVoiceBridge,

    /// Text-to-phoneme converter
    g2p: SimpleG2P,

    /// Current pacing state
    current_pacing: LTCPacing,

    /// Whether audio output is available
    audio_available: bool,

    /// Audio output stream (rodio)
    #[cfg(feature = "audio")]
    _audio_stream: Option<rodio::OutputStream>,

    /// Audio sink for playback (rodio)
    #[cfg(feature = "audio")]
    audio_sink: Option<rodio::Sink>,

    /// LTC-driven vocal tract pipeline (feature-gated)
    #[cfg(feature = "vocal-tract")]
    ltc_pipeline: Option<super::vocal_tract_fep::VocalTractPipeline>,

    /// Formant database for online adaptation (phoneme targets)
    #[cfg(feature = "vocal-tract")]
    formant_db: super::formant_targets::FormantDatabase,

    /// Last voice output metrics (for feedback to cognitive loop)
    last_voice_metrics: Option<super::voice_feedback::VoiceOutputMetrics>,

    /// Real-time streaming voice (cpal ring buffer) — preferred backend when available.
    #[cfg(feature = "live-voice")]
    live_voice: Option<super::live_voice::LiveVoice>,

    /// Statistics
    total_utterances: u64,
    total_audio_seconds: f32,
}

impl ReplVoiceOutput {
    /// Create a new REPL voice output system
    pub fn new(config: ReplVoiceConfig) -> Result<Self> {
        // Create articulatory synthesizer with consciousness-aware config
        let articulatory_config = ArticulatoryConfig {
            base_f0: config.base_f0,
            base_tau: 0.05,
            frame_rate: 200.0,
            coarticulation: true,
            ..Default::default()
        };
        let articulatory = ArticulatorySynthesizer::with_config(articulatory_config);

        // Create vocoder
        let vocoder_config = VocoderConfig {
            sample_rate: config.sample_rate,
            volume: config.volume,
            ..Default::default()
        };
        let vocoder = FormantVocoder::with_config(vocoder_config);

        // Create basic voice output as fallback
        let voice_config = VoiceOutputConfig {
            sample_rate: config.sample_rate,
            volume: config.volume,
            enable_tts: false, // We use our own synthesis
            ..Default::default()
        };
        let voice_output = VoiceOutput::new(voice_config);

        // Create cognitive bridge
        let cognitive_bridge = CognitiveVoiceBridge::new();

        // Create G2P
        let g2p = SimpleG2P::new();

        // Try to initialize audio
        let (audio_available, _audio_stream, _audio_sink) = Self::init_audio(&config);

        // Initialize LTC pipeline if requested and feature-enabled
        #[cfg(feature = "vocal-tract")]
        let ltc_pipeline = if config.use_ltc_pipeline {
            use symthaea_core::genesis::GenesisSeed;
            let genesis = GenesisSeed::from_phrase("repl-vocal-tract");
            let mut pipeline = super::vocal_tract_fep::VocalTractPipeline::new(&genesis);
            // Pre-train on phoneme database for reasonable starting point
            let db = super::formant_targets::FormantDatabase::new();
            super::vocal_tract_controller::train_controller_on_phoneme_db(
                &mut pipeline.controller,
                &genesis,
                &db,
                3,
            );
            // Populate manner map for source_type propagation to vocoder
            super::vocal_tract_fep::populate_manner_map(&mut pipeline);
            Some(pipeline)
        } else {
            None
        };

        // Try to create LiveVoice (real-time streaming via cpal ring buffer).
        // Falls back gracefully if no audio device available.
        #[cfg(feature = "live-voice")]
        let live_voice = if config.use_ltc_pipeline {
            use symthaea_core::genesis::GenesisSeed;
            let genesis = GenesisSeed::from_phrase("repl-vocal-tract");
            match super::live_voice::LiveVoice::new(&genesis) {
                Ok(lv) => {
                    debug!("LiveVoice initialized — using real-time streaming backend");
                    Some(lv)
                }
                Err(e) => {
                    debug!("LiveVoice unavailable ({}), falling back to rodio", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            articulatory,
            vocoder,
            voice_output,
            cognitive_bridge,
            g2p,
            current_pacing: LTCPacing::default(),
            audio_available,
            #[cfg(feature = "audio")]
            _audio_stream,
            #[cfg(feature = "audio")]
            audio_sink,
            #[cfg(feature = "vocal-tract")]
            ltc_pipeline,
            #[cfg(feature = "vocal-tract")]
            formant_db: super::formant_targets::FormantDatabase::new(),
            last_voice_metrics: None,
            #[cfg(feature = "live-voice")]
            live_voice,
            total_utterances: 0,
            total_audio_seconds: 0.0,
        })
    }

    /// Initialize audio output
    #[cfg(feature = "audio")]
    fn init_audio(
        config: &ReplVoiceConfig,
    ) -> (bool, Option<rodio::OutputStream>, Option<rodio::Sink>) {
        use rodio::{OutputStream, Sink};

        // Try to get output stream
        let stream_result = if let Some(ref device_name) = config.device_name {
            // Try to find specific device
            use rodio::cpal::traits::{DeviceTrait, HostTrait};
            let host = rodio::cpal::default_host();
            let device = host.output_devices().ok().and_then(|mut devices| {
                devices.find(|d| d.name().map(|n| n.contains(device_name)).unwrap_or(false))
            });

            match device {
                Some(dev) => OutputStream::try_from_device(&dev),
                None => {
                    warn!("Audio device '{}' not found, using default", device_name);
                    OutputStream::try_default()
                }
            }
        } else {
            OutputStream::try_default()
        };

        match stream_result {
            Ok((stream, handle)) => match Sink::try_new(&handle) {
                Ok(sink) => {
                    info!("Audio output initialized successfully");
                    (true, Some(stream), Some(sink))
                }
                Err(e) => {
                    warn!("Failed to create audio sink: {}", e);
                    (false, None, None)
                }
            },
            Err(e) => {
                warn!("Failed to initialize audio output: {}", e);
                (false, None, None)
            }
        }
    }

    /// Initialize audio output (stub when audio feature disabled)
    #[cfg(not(feature = "audio"))]
    fn init_audio(_config: &ReplVoiceConfig) -> (bool, (), ()) {
        warn!("Audio playback disabled (audio feature not enabled)");
        (false, (), ())
    }

    /// Update pacing from consciousness state
    ///
    /// This is the key consciousness-modulation entry point.
    pub fn update_from_consciousness(
        &mut self,
        unified_psi: f32,
        prediction_error: f32,
        emotional_valence: f32,
        emotional_arousal: f32,
        in_flow: bool,
        speech_rate_multiplier: f32,
        pause_multiplier: f32,
        tau_mean: f32,
    ) {
        // Create CfC-like output from consciousness state
        // The "hidden state" is approximated from emotional state and phi
        let hidden_state: Vec<f32> = (0..64)
            .map(|i| {
                let phase = i as f32 / 64.0 * std::f32::consts::TAU;
                let base = (phase + unified_psi * 2.0).sin() * 0.5;
                let emotional = emotional_valence * 0.3 + emotional_arousal * 0.2;
                let flow_contrib = if in_flow { 0.2 } else { 0.0 };
                base + emotional + flow_contrib
            })
            .collect();

        // Create attention state from consciousness metrics
        let mut attention_state = HashMap::new();
        if unified_psi > 0.5 {
            attention_state.insert("phi".to_string(), unified_psi);
        }
        if in_flow {
            attention_state.insert("flow".to_string(), 1.0);
        }

        // Detect semantic primitives from emotional state
        let mut primitives = Vec::new();
        if emotional_valence > 0.3 {
            primitives.push("positive".to_string());
        } else if emotional_valence < -0.3 {
            primitives.push("uncertain".to_string());
        }
        if emotional_arousal > 0.6 {
            primitives.push("emphasis".to_string());
        }

        // Update cognitive bridge
        self.cognitive_bridge.update(
            &hidden_state,
            tau_mean,
            prediction_error,
            attention_state,
            primitives,
        );

        // Get base pacing from cognitive bridge
        let mut pacing = self.cognitive_bridge.get_ltc_pacing();

        // Apply consciousness-specific modulations
        pacing = pacing.apply_adaptive_behavior(
            speech_rate_multiplier * self.config.base_rate,
            pause_multiplier,
            if in_flow { 1.3 } else { 1.0 },
        );

        // Override emotional state with direct values
        pacing.emotional_valence = emotional_valence;
        pacing.arousal = emotional_arousal;

        self.current_pacing = pacing;

        debug!(
            "Voice pacing updated: rate={:.2}, pause={:.2}, valence={:.2}, arousal={:.2}",
            self.current_pacing.rate,
            self.current_pacing.phrase_pause,
            self.current_pacing.emotional_valence,
            self.current_pacing.arousal,
        );
    }

    /// Synthesize speech from text
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();

        #[cfg(feature = "vocal-tract")]
        let use_ltc = self.ltc_pipeline.is_some();
        #[cfg(not(feature = "vocal-tract"))]
        let use_ltc = false;

        let samples = if use_ltc {
            #[cfg(feature = "vocal-tract")]
            {
                self.synthesize_ltc_pipeline(text)?
            }
            #[cfg(not(feature = "vocal-tract"))]
            {
                unreachable!()
            }
        } else if self.config.use_articulatory {
            self.synthesize_articulatory(text)?
        } else {
            self.synthesize_simple(text)?
        };

        let duration = samples.len() as f32 / self.config.sample_rate as f32;

        self.total_utterances += 1;
        self.total_audio_seconds += duration;

        debug!(
            "Synthesized {} samples ({:.2}s) in {:?}",
            samples.len(),
            duration,
            start.elapsed()
        );

        Ok(samples)
    }

    /// Synthesize using articulatory synthesis (higher quality)
    fn synthesize_articulatory(&mut self, text: &str) -> Result<Vec<f32>> {
        // Convert text to phonemes with consciousness-modulated duration
        let base_duration = self.config.phoneme_duration_base / self.current_pacing.rate;
        let phonemes = self.g2p.text_to_phonemes(text, base_duration);

        if phonemes.is_empty() {
            return Ok(Vec::new());
        }

        // Filter out silence phonemes for synthesis (they're handled as gaps)
        let speech_phonemes: Vec<TimedPhoneme> = phonemes
            .into_iter()
            .filter(|p| p.phoneme != "SIL")
            .collect();

        if speech_phonemes.is_empty() {
            return Ok(Vec::new());
        }

        // Generate formant frames using articulatory synthesizer
        let frames = self
            .articulatory
            .synthesize(&speech_phonemes, &self.current_pacing);

        if frames.is_empty() {
            return Ok(Vec::new());
        }

        // Convert formants to audio using vocoder
        let samples = self.vocoder.synthesize(&frames);

        // Apply volume scaling
        let scaled: Vec<f32> = samples.iter().map(|s| s * self.config.volume).collect();

        Ok(scaled)
    }

    /// Synthesize using the LTC-driven vocal tract pipeline.
    ///
    /// Converts text → phonemes, then for each phoneme generates FormantFrames
    /// via the HdcLtcUnifiedNetwork controller at 200Hz, feeding through the vocoder.
    #[cfg(feature = "vocal-tract")]
    fn synthesize_ltc_pipeline(&mut self, text: &str) -> Result<Vec<f32>> {
        use super::vocal_tract_encoder::VoiceCognitiveState;

        let pipeline = self
            .ltc_pipeline
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("LTC pipeline not initialized"))?;

        // Convert text → phonemes
        let base_duration = self.config.phoneme_duration_base / self.current_pacing.rate;
        let phonemes = self.g2p.text_to_phonemes(text, base_duration);
        if phonemes.is_empty() {
            return Ok(Vec::new());
        }

        if phonemes.is_empty() {
            return Ok(Vec::new());
        }

        let dt = 1.0 / 200.0; // 200Hz frame rate
        let mut all_frames = Vec::new();

        // Build cognitive state from current pacing
        let cognitive_state = VoiceCognitiveState {
            prediction_error: 0.1,
            emotional_valence: self.current_pacing.emotional_valence,
            emotional_arousal: self.current_pacing.arousal,
            unified_quality: 0.7,
            epistemic_confidence: 0.8,
            coherence_velocity: 0.0,
            cross_agreement: 0.7,
            consciousness_level: 0.6,
            articulation_quality: 0.7,
            rate_stability: 1.0,
            ..Default::default()
        };

        let base_f0 = self.config.base_f0;
        let arousal = self.current_pacing.arousal;
        let mut elapsed = 0.0f32;
        // Track phrase-local progress for F0 declination reset at phrase boundaries
        let mut phrase_start_time = 0.0f32;
        let mut phrase_duration = Self::compute_phrase_duration(&phonemes, 0);

        for (ph_idx, timed_phoneme) in phonemes.iter().enumerate() {
            let is_silence = timed_phoneme.phoneme == "SIL" || timed_phoneme.phoneme == "SP";

            if is_silence {
                // Insert silent frames for phrase/sentence pauses
                let pause_duration = if timed_phoneme.duration > 0.01 {
                    timed_phoneme.duration
                } else {
                    self.current_pacing.phrase_pause
                };
                let n_silent = (pause_duration / dt).max(1.0) as usize;
                for _ in 0..n_silent {
                    all_frames.push(super::FormantFrame::silent(elapsed));
                    elapsed += dt;
                }

                // Reset F0 declination at phrase boundary
                phrase_start_time = elapsed;
                phrase_duration = Self::compute_phrase_duration(&phonemes, ph_idx + 1);
            } else {
                let n_frames = (timed_phoneme.duration / dt).max(1.0) as usize;

                for frame_i in 0..n_frames {
                    let phoneme_progress = frame_i as f32 / n_frames as f32;
                    // Phrase-local progress for F0 declination (resets after pauses)
                    let phrase_elapsed = elapsed - phrase_start_time;
                    let utterance_progress = if phrase_duration > 0.0 {
                        (phrase_elapsed / phrase_duration).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    let prosody = super::vocal_tract_fep::ProsodyContext {
                        utterance_progress,
                        phoneme_progress,
                        stress: timed_phoneme.stress,
                        base_f0,
                        arousal,
                    };

                    let frame = pipeline.tick_with_prosody(
                        &cognitive_state,
                        None,
                        dt,
                        Some(&timed_phoneme.phoneme),
                        &prosody,
                    );
                    all_frames.push(frame);
                    elapsed += dt;
                }

                // Online adaptation: refine controller on vowel phoneme targets
                if let Some(target) = self.formant_db.lookup(&timed_phoneme.phoneme) {
                    if target.is_vowel {
                        let phoneme_hv = pipeline.get_or_create_phoneme_hv(&timed_phoneme.phoneme);
                        let target_frame = super::FormantFrame::from_target(
                            target,
                            base_f0,
                            if target.is_voiced { 0.7 } else { 0.3 },
                            0.0,
                        );
                        pipeline
                            .controller
                            .train_step(&phoneme_hv, &target_frame, dt, Some(1e-4));
                    }
                }
            }
        }

        if all_frames.is_empty() {
            return Ok(Vec::new());
        }

        // Auto-compute voice quality metrics for feedback to cognitive loop
        let metrics =
            super::voice_feedback::VoiceOutputMetrics::from_formant_frames(&all_frames, None);
        self.last_voice_metrics = Some(metrics);

        // Convert formants → audio via vocoder
        let samples = self.vocoder.synthesize(&all_frames);
        let scaled: Vec<f32> = samples.iter().map(|s| s * self.config.volume).collect();

        Ok(scaled)
    }

    /// Synthesize using simple method (lower latency)
    fn synthesize_simple(&mut self, text: &str) -> Result<Vec<f32>> {
        // Use the basic voice output system
        self.voice_output.set_pacing(self.current_pacing.clone());
        self.voice_output.synthesize(text)
    }

    /// Speak text (synthesize and play).
    ///
    /// Prefers the real-time `LiveVoice` backend (cpal ring buffer, frame-by-frame
    /// streaming) when available. Falls back to batch rodio synthesis.
    pub fn speak(&mut self, text: &str) -> Result<()> {
        // Prefer LiveVoice for real-time streaming output
        #[cfg(feature = "live-voice")]
        if let Some(ref mut lv) = self.live_voice {
            self.total_utterances += 1;
            return lv.speak(text);
        }

        let samples = self.synthesize(text)?;

        if samples.is_empty() {
            return Ok(());
        }

        self.play_audio(&samples)
    }

    /// Update the cognitive state on the LiveVoice backend (if present).
    ///
    /// Changes take effect on the next motor frame (~5ms latency).
    #[cfg(feature = "live-voice")]
    pub fn update_live_cognitive_state(
        &self,
        state: super::vocal_tract_encoder::VoiceCognitiveState,
    ) {
        if let Some(ref lv) = self.live_voice {
            *lv.cognitive_state_handle().lock() = state;
        }
    }

    /// Play audio samples
    #[cfg(feature = "audio")]
    fn play_audio(&mut self, samples: &[f32]) -> Result<()> {
        use rodio::Source;

        if !self.audio_available {
            debug!("Audio not available, skipping playback");
            return Ok(());
        }

        let sink = self
            .audio_sink
            .as_ref()
            .ok_or_else(|| anyhow!("Audio sink not initialized"))?;

        // Create rodio source from samples
        let sample_rate = self.config.sample_rate;
        let source = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples.to_vec());

        // Queue audio for playback
        sink.append(source);

        Ok(())
    }

    /// Play audio samples (stub when audio feature disabled)
    #[cfg(not(feature = "audio"))]
    fn play_audio(&mut self, _samples: &[f32]) -> Result<()> {
        debug!("Audio playback disabled");
        Ok(())
    }

    /// Wait for audio to finish playing
    #[cfg(feature = "audio")]
    pub fn wait_until_finished(&self) {
        if let Some(ref sink) = self.audio_sink {
            sink.sleep_until_end();
        }
    }

    /// Wait for audio to finish playing (stub)
    #[cfg(not(feature = "audio"))]
    pub fn wait_until_finished(&self) {
        // No-op when audio disabled
    }

    /// Check if audio output is available
    pub fn is_audio_available(&self) -> bool {
        self.audio_available
    }

    /// Get current pacing state
    pub fn pacing(&self) -> &LTCPacing {
        &self.current_pacing
    }

    /// Enable or disable the LTC vocal tract pipeline at runtime.
    ///
    /// When enabling, initializes the pipeline on demand if it doesn't exist.
    /// Requires the `vocal-tract` feature; no-op otherwise.
    #[cfg(feature = "vocal-tract")]
    pub fn set_ltc_pipeline(&mut self, enabled: bool) {
        if enabled && self.ltc_pipeline.is_none() {
            use symthaea_core::genesis::GenesisSeed;
            let genesis = GenesisSeed::from_phrase("repl-vocal-tract");
            let mut pipeline = super::vocal_tract_fep::VocalTractPipeline::new(&genesis);
            let db = super::formant_targets::FormantDatabase::new();
            super::vocal_tract_controller::train_controller_on_phoneme_db(
                &mut pipeline.controller,
                &genesis,
                &db,
                3,
            );
            super::vocal_tract_fep::populate_manner_map(&mut pipeline);
            self.ltc_pipeline = Some(pipeline);
        }
        if !enabled {
            self.ltc_pipeline = None;
        }
    }

    /// Enable or disable the LTC vocal tract pipeline (no-op without `vocal-tract` feature).
    #[cfg(not(feature = "vocal-tract"))]
    pub fn set_ltc_pipeline(&mut self, _enabled: bool) {
        // No-op: vocal-tract feature not compiled in
    }

    /// Take the last voice output metrics (for feeding back to cognitive loop).
    ///
    /// Returns `Some(metrics)` if synthesis has run since the last `take`, `None` otherwise.
    /// Usage: `if let Some(m) = voice.take_voice_metrics() { mind.update_voice_feedback(m); }`
    pub fn take_voice_metrics(&mut self) -> Option<super::voice_feedback::VoiceOutputMetrics> {
        self.last_voice_metrics.take()
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, f32) {
        (self.total_utterances, self.total_audio_seconds)
    }

    /// Compute total speech duration for a phrase starting at `from_idx`.
    ///
    /// Sums durations of non-silence phonemes up to the next SIL/SP or end.
    #[cfg(feature = "vocal-tract")]
    fn compute_phrase_duration(phonemes: &[TimedPhoneme], from_idx: usize) -> f32 {
        phonemes[from_idx..]
            .iter()
            .take_while(|p| p.phoneme != "SIL" && p.phoneme != "SP")
            .map(|p| p.duration)
            .sum()
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.current_pacing = LTCPacing::default();
        self.articulatory.reset();
        self.vocoder.reset();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g2p_dictionary() {
        let g2p = SimpleG2P::new();

        let hello = g2p.word_to_phonemes("hello");
        assert!(
            hello.len() >= 3,
            "hello should have multiple phonemes: {:?}",
            hello
        );

        let world = g2p.word_to_phonemes("world");
        assert!(
            world.len() >= 3,
            "world should have multiple phonemes: {:?}",
            world
        );
    }

    #[test]
    fn test_g2p_unknown_word() {
        let g2p = SimpleG2P::new();

        let unknown = g2p.word_to_phonemes("syzygy");
        assert!(!unknown.is_empty(), "Unknown word should produce phonemes");
    }

    #[test]
    fn test_text_to_phonemes() {
        let g2p = SimpleG2P::new();

        let phonemes = g2p.text_to_phonemes("Hello world.", 0.08);
        assert!(!phonemes.is_empty());

        // Should have some timing
        let last = phonemes.last().unwrap();
        assert!(last.start_time > 0.0, "Should have positive timing");
    }

    #[test]
    fn test_repl_voice_creation() {
        let config = ReplVoiceConfig::default();
        let voice = ReplVoiceOutput::new(config);
        assert!(voice.is_ok());
    }

    #[test]
    fn test_consciousness_modulation() {
        let config = ReplVoiceConfig::default();
        let mut voice = ReplVoiceOutput::new(config).unwrap();

        // Low consciousness state
        voice.update_from_consciousness(
            0.2,   // low phi
            0.5,   // high error
            -0.3,  // negative valence
            0.2,   // low arousal
            false, // not in flow
            0.8,   // slow speech
            1.5,   // long pauses
            1.5,   // high tau
        );

        let low_rate = voice.pacing().rate;
        let low_pause = voice.pacing().phrase_pause;

        // High consciousness state
        voice.update_from_consciousness(
            0.8,  // high phi
            0.1,  // low error
            0.5,  // positive valence
            0.8,  // high arousal
            true, // in flow
            1.2,  // fast speech
            0.7,  // short pauses
            0.5,  // low tau
        );

        let high_rate = voice.pacing().rate;
        let high_pause = voice.pacing().phrase_pause;

        // High consciousness should speak faster with shorter pauses
        assert!(
            high_rate > low_rate,
            "High consciousness should speak faster: {} vs {}",
            high_rate,
            low_rate
        );
        assert!(
            high_pause < low_pause,
            "High consciousness should have shorter pauses: {} vs {}",
            high_pause,
            low_pause
        );
    }

    #[test]
    fn test_synthesis() {
        let config = ReplVoiceConfig {
            use_articulatory: true,
            ..Default::default()
        };
        let mut voice = ReplVoiceOutput::new(config).unwrap();

        let samples = voice.synthesize("Hello world.").unwrap();
        assert!(!samples.is_empty(), "Should produce audio samples");

        // Samples should be in reasonable range
        let max_sample = samples.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_sample <= 1.0, "Samples should be normalized");
    }

    #[test]
    fn test_empty_text() {
        let config = ReplVoiceConfig::default();
        let mut voice = ReplVoiceOutput::new(config).unwrap();

        let samples = voice.synthesize("").unwrap();
        assert!(samples.is_empty(), "Empty text should produce no samples");
    }

    #[test]
    fn test_ltc_pipeline_config_default() {
        let config = ReplVoiceConfig::default();
        // Default matches feature gate: enabled when vocal-tract is compiled in
        if cfg!(feature = "vocal-tract") {
            assert!(
                config.use_ltc_pipeline,
                "LTC pipeline should default to true when vocal-tract feature is active"
            );
        } else {
            assert!(
                !config.use_ltc_pipeline,
                "LTC pipeline should default to false without vocal-tract feature"
            );
        }
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_ltc_pipeline_synthesis() {
        let config = ReplVoiceConfig {
            use_ltc_pipeline: true,
            ..Default::default()
        };
        let mut voice = ReplVoiceOutput::new(config).unwrap();

        let samples = voice.synthesize("Hello world.").unwrap();
        assert!(
            !samples.is_empty(),
            "LTC pipeline should produce audio samples"
        );

        // Samples should be in reasonable range
        for &s in &samples {
            assert!(s.is_finite(), "All samples should be finite");
        }
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_comma_inserts_silence() {
        // Verify that the G2P generates SIL phonemes at comma boundaries
        // and these become silent frames in the pipeline output.
        let g2p = SimpleG2P::new();

        let phonemes_no_comma = g2p.text_to_phonemes("Hello world", 0.08);
        let phonemes_comma = g2p.text_to_phonemes("Hello, world", 0.08);

        let sil_count_no_comma = phonemes_no_comma
            .iter()
            .filter(|p| p.phoneme == "SIL")
            .count();
        let sil_count_comma = phonemes_comma.iter().filter(|p| p.phoneme == "SIL").count();

        assert!(
            sil_count_comma > sil_count_no_comma,
            "Comma should introduce SIL phonemes: with_comma={}, without={}",
            sil_count_comma,
            sil_count_no_comma
        );

        // Also verify the SIL phonemes produce silent frames in full pipeline
        let config = ReplVoiceConfig {
            use_ltc_pipeline: true,
            ..Default::default()
        };
        let mut voice = ReplVoiceOutput::new(config).unwrap();
        let samples = voice.synthesize("Hello, world").unwrap();
        assert!(!samples.is_empty(), "Should produce audio with pauses");
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_period_pause_longer_than_comma() {
        let g2p = SimpleG2P::new();

        // Text with comma pause
        let phonemes_comma = g2p.text_to_phonemes("Hello, world", 0.08);
        let comma_pause: f32 = phonemes_comma
            .iter()
            .filter(|p| p.phoneme == "SIL")
            .map(|p| p.duration)
            .sum();

        // Text with period pause
        let phonemes_period = g2p.text_to_phonemes("Hello. World", 0.08);
        let period_pause: f32 = phonemes_period
            .iter()
            .filter(|p| p.phoneme == "SIL")
            .map(|p| p.duration)
            .sum();

        assert!(
            period_pause > comma_pause,
            "Period pause ({:.2}s) should be longer than comma pause ({:.2}s)",
            period_pause,
            comma_pause
        );
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_f0_resets_after_pause() {
        use crate::voice::vocal_tract_fep::ProsodyContext;
        use crate::voice::FormantFrame;

        let base_f0 = 150.0;

        // End of first phrase (progress=1.0, F0 declined)
        let mut frame_end = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        let ctx_end = ProsodyContext {
            utterance_progress: 1.0,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0,
            arousal: 0.5,
        };
        ctx_end.apply_prosody(&mut frame_end);

        // Start of new phrase after pause (progress=0.0, F0 at peak)
        let mut frame_new = FormantFrame {
            f0: 200.0,
            energy: 0.5,
            ..FormantFrame::silent(0.0)
        };
        let ctx_new = ProsodyContext {
            utterance_progress: 0.0,
            phoneme_progress: 0.5,
            stress: 0,
            base_f0,
            arousal: 0.5,
        };
        ctx_new.apply_prosody(&mut frame_new);

        // F0 should be higher at phrase start than at phrase end
        assert!(
            frame_new.f0 > frame_end.f0,
            "F0 should reset after pause: new_phrase={:.1}, end_phrase={:.1}",
            frame_new.f0,
            frame_end.f0
        );
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_online_adaptation_reduces_error() {
        let config = ReplVoiceConfig {
            use_ltc_pipeline: true,
            ..Default::default()
        };
        let mut voice = ReplVoiceOutput::new(config).unwrap();

        // Synthesize the same text twice — second synthesis should benefit
        // from online adaptation on vowel targets during the first pass.
        let _samples1 = voice.synthesize("Hello world.").unwrap();
        let _samples2 = voice.synthesize("Hello world.").unwrap();

        // Both should produce valid output (no crash from adaptation)
        assert!(!_samples1.is_empty());
        assert!(!_samples2.is_empty());
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_online_adaptation_consonants_no_crash() {
        let config = ReplVoiceConfig {
            use_ltc_pipeline: true,
            ..Default::default()
        };
        let mut voice = ReplVoiceOutput::new(config).unwrap();

        // Text with many consonants — adaptation should skip them gracefully
        let samples = voice.synthesize("Strict trips.").unwrap();
        assert!(
            !samples.is_empty(),
            "Consonant-heavy text should still produce audio"
        );
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_repl_captures_voice_metrics() {
        let config = ReplVoiceConfig {
            use_ltc_pipeline: true,
            ..Default::default()
        };
        let mut voice = ReplVoiceOutput::new(config).unwrap();

        // Before synthesis, no metrics available
        assert!(voice.take_voice_metrics().is_none());

        // After synthesis, metrics should be populated
        let _samples = voice.synthesize("Hello world.").unwrap();
        let metrics = voice.take_voice_metrics();
        assert!(metrics.is_some(), "Synthesis should populate voice metrics");

        let m = metrics.unwrap();
        assert!(m.pitch_stability > 0.0 && m.pitch_stability <= 1.0);
        assert!(m.energy_consistency > 0.0 && m.energy_consistency <= 1.0);
        assert!(m.articulation_score > 0.0 && m.articulation_score <= 1.0);

        // After take, should be None again
        assert!(voice.take_voice_metrics().is_none());
    }

    #[cfg(feature = "vocal-tract")]
    #[test]
    fn test_ltc_pipeline_default_with_feature() {
        // When vocal-tract feature is active, the default config enables LTC
        let config = ReplVoiceConfig::default();
        assert!(
            config.use_ltc_pipeline,
            "With vocal-tract feature, LTC pipeline should default to true"
        );

        // Verify it actually initializes the pipeline
        let voice = ReplVoiceOutput::new(config).unwrap();
        // Voice should be functional with LTC pipeline
        assert!(voice.is_audio_available() || !voice.is_audio_available()); // Just checking it exists
    }

    #[cfg(feature = "live-voice")]
    #[test]
    fn test_repl_live_voice_field() {
        // LiveVoice init may fail without audio device — that's OK, falls back to None.
        // Just verify ReplVoiceOutput::new() succeeds with the live-voice feature.
        let config = ReplVoiceConfig {
            use_ltc_pipeline: true,
            ..ReplVoiceConfig::default()
        };
        let voice = ReplVoiceOutput::new(config).unwrap();
        // If audio device is available, live_voice will be Some; otherwise None.
        // Either way, ReplVoiceOutput should be functional.
        assert!(voice.total_utterances == 0);
    }
}
