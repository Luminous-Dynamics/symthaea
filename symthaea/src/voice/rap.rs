// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Rap Synthesis Module
//!
//! Combines beat synchronization, HDC rhyme encoding, and formant synthesis
//! to produce rhythmic, rhyme-aware spoken word / rap output.
//!
//! ## Pipeline
//!
//! ```text
//! Text/Lyrics → Phonemization → Rhyme Analysis → Beat Mapping →
//!     → Formant Synthesis → Beat-Synced Audio
//! ```
//!
//! ## Features
//!
//! - Beat-synchronized syllable timing
//! - HDC-based rhyme detection and emphasis
//! - Flow pattern selection (boom bap, triplet, double-time)
//! - Prosodic emphasis on rhyming syllables

use super::LTCPacing;
use super::articulatory_synthesizer::{
    ArticulatoryConfig, ArticulatorySynthesizer, FormantFrame, TimedPhoneme,
};
use super::beat_sync::{BeatSync, FlowPattern, SyllableTiming};
use super::rhyme_hdc::{RhymeEncoder, RhymeScheme};
use super::vocoder::{FormantVocoder, VocoderConfig};

use serde::{Deserialize, Serialize};

/// Configuration for rap synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RapConfig {
    /// Tempo in BPM
    pub bpm: f32,
    /// Flow pattern name
    pub flow_style: FlowStyle,
    /// Rhyme scheme pattern (e.g., "AABB", "ABAB")
    pub rhyme_scheme: String,
    /// Emphasis boost for rhyming syllables
    pub rhyme_emphasis: f32,
    /// Base pitch in Hz
    pub base_pitch: f32,
    /// Pitch variation for rhymes
    pub rhyme_pitch_lift: f32,
    /// Sample rate for output audio
    pub sample_rate: u32,
    /// Enable swing timing
    pub swing: bool,
}

impl Default for RapConfig {
    fn default() -> Self {
        Self {
            bpm: 90.0,
            flow_style: FlowStyle::BoomBap,
            rhyme_scheme: "AABB".to_string(),
            rhyme_emphasis: 1.3,
            base_pitch: 110.0,
            rhyme_pitch_lift: 20.0,
            sample_rate: 24000,
            swing: true,
        }
    }
}

/// Predefined flow styles
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FlowStyle {
    /// Classic hip-hop flow
    BoomBap,
    /// Triplet-based flow (like Migos)
    Triplet,
    /// Fast syllable delivery
    DoubleTime,
    /// Relaxed, behind-the-beat
    LaidBack,
    /// Straight 16th notes
    Sixteenth,
}

impl FlowStyle {
    fn to_pattern(&self) -> FlowPattern {
        match self {
            FlowStyle::BoomBap => FlowPattern::boom_bap(),
            FlowStyle::Triplet => FlowPattern::triplet_flow(),
            FlowStyle::DoubleTime => FlowPattern::double_time(),
            FlowStyle::LaidBack => FlowPattern::laid_back(),
            FlowStyle::Sixteenth => FlowPattern::sixteenth_notes(),
        }
    }
}

/// A line of lyrics with phonetic breakdown
#[derive(Debug, Clone)]
pub struct LyricLine {
    /// Original text
    pub text: String,
    /// Words with their phoneme sequences
    pub words: Vec<WordPhonemes>,
    /// Line index in verse
    pub line_index: usize,
    /// Rhyme group (lines with same letter rhyme together)
    pub rhyme_group: char,
}

/// A word with its phoneme breakdown
#[derive(Debug, Clone)]
pub struct WordPhonemes {
    /// The word text
    pub text: String,
    /// ARPABET phoneme sequence
    pub phonemes: Vec<String>,
    /// Syllable count
    pub syllable_count: usize,
    /// Is this word at a rhyme position?
    pub is_rhyme_word: bool,
    /// Rhyme score with partner word (if applicable)
    pub rhyme_score: Option<f32>,
}

/// Full verse structure
#[derive(Debug, Clone)]
pub struct Verse {
    /// Lines in the verse
    pub lines: Vec<LyricLine>,
    /// Total syllable count
    pub total_syllables: usize,
    /// Bars needed for this verse
    pub bar_count: u32,
}

/// Rap synthesizer
pub struct RapSynthesizer {
    /// Configuration
    config: RapConfig,
    /// Beat synchronization engine
    beat_sync: BeatSync,
    /// Rhyme encoder
    rhyme_encoder: RhymeEncoder,
    /// Formant synthesizer
    formant_synth: ArticulatorySynthesizer,
    /// Vocoder
    vocoder: FormantVocoder,
    /// Current flow pattern
    flow_pattern: FlowPattern,
    /// Rhyme scheme (reserved for future use)
    _rhyme_scheme: RhymeScheme,
}

impl RapSynthesizer {
    /// Create a new rap synthesizer
    pub fn new(config: RapConfig) -> Self {
        let mut beat_sync = BeatSync::new(config.bpm);
        if config.swing {
            beat_sync.swing = super::beat_sync::SwingConfig::hip_hop();
        }

        let flow_pattern = config.flow_style.to_pattern();
        let rhyme_scheme = RhymeScheme::from_pattern(&config.rhyme_scheme);

        let formant_synth = ArticulatorySynthesizer::with_config(ArticulatoryConfig {
            base_f0: config.base_pitch,
            ..Default::default()
        });

        let vocoder = FormantVocoder::with_config(VocoderConfig {
            sample_rate: config.sample_rate,
            ..Default::default()
        });

        Self {
            config,
            beat_sync,
            rhyme_encoder: RhymeEncoder::new(),
            formant_synth,
            vocoder,
            flow_pattern,
            _rhyme_scheme: rhyme_scheme,
        }
    }

    /// Create with hip-hop defaults
    pub fn hip_hop() -> Self {
        Self::new(RapConfig {
            bpm: 90.0,
            flow_style: FlowStyle::BoomBap,
            ..Default::default()
        })
    }

    /// Create with trap defaults
    pub fn trap() -> Self {
        Self::new(RapConfig {
            bpm: 140.0,
            flow_style: FlowStyle::Triplet,
            swing: false,
            ..Default::default()
        })
    }

    /// Set the flow style
    pub fn set_flow(&mut self, style: FlowStyle) {
        self.config.flow_style = style;
        self.flow_pattern = style.to_pattern();
    }

    /// Set BPM
    pub fn set_bpm(&mut self, bpm: f32) {
        self.config.bpm = bpm;
        self.beat_sync = BeatSync::new(bpm);
        if self.config.swing {
            self.beat_sync.swing = super::beat_sync::SwingConfig::hip_hop();
        }
    }

    /// Parse lyrics into a verse structure
    ///
    /// Input format: One line per line, words separated by spaces.
    /// Phonemes should be provided in a dictionary or looked up.
    pub fn parse_lyrics(&self, lyrics: &str, phoneme_dict: &impl PhonemeDict) -> Verse {
        let pattern_chars: Vec<char> = self.config.rhyme_scheme.chars().collect();

        let lines: Vec<LyricLine> = lyrics
            .lines()
            .enumerate()
            .map(|(i, line)| {
                let words: Vec<WordPhonemes> = line
                    .split_whitespace()
                    .map(|word| {
                        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
                        let phonemes = phoneme_dict.lookup(clean_word);
                        let syllable_count = count_syllables(&phonemes);

                        WordPhonemes {
                            text: clean_word.to_string(),
                            phonemes,
                            syllable_count,
                            is_rhyme_word: false,
                            rhyme_score: None,
                        }
                    })
                    .collect();

                let rhyme_group = pattern_chars
                    .get(i % pattern_chars.len())
                    .copied()
                    .unwrap_or('A');

                LyricLine {
                    text: line.to_string(),
                    words,
                    line_index: i,
                    rhyme_group,
                }
            })
            .collect();

        let total_syllables: usize = lines
            .iter()
            .map(|l| l.words.iter().map(|w| w.syllable_count).sum::<usize>())
            .sum();

        let syllables_per_bar = self.flow_pattern.syllable_count();
        let bar_count = ((total_syllables as f32 / syllables_per_bar as f32).ceil() as u32).max(1);

        Verse {
            lines,
            total_syllables,
            bar_count,
        }
    }

    /// Analyze rhymes in a verse and mark rhyming words
    pub fn analyze_rhymes(&self, verse: &mut Verse) {
        // Group lines by rhyme group
        let mut groups: std::collections::HashMap<char, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, line) in verse.lines.iter().enumerate() {
            groups.entry(line.rhyme_group).or_default().push(i);
        }

        // For each rhyme group, compare end words
        for (_, line_indices) in groups.iter() {
            if line_indices.len() < 2 {
                continue;
            }

            // Get end words from each line (clone phonemes to avoid borrow issues)
            let end_words: Vec<(usize, Vec<String>)> = line_indices
                .iter()
                .filter_map(|&i| verse.lines[i].words.last().map(|w| (i, w.phonemes.clone())))
                .collect();

            // Collect rhyme scores first
            let mut rhyme_updates: Vec<(usize, f32)> = Vec::new();

            for i in 0..end_words.len() {
                for j in (i + 1)..end_words.len() {
                    let (line_i, ref phonemes_i) = end_words[i];
                    let (line_j, ref phonemes_j) = end_words[j];

                    let score = self.rhyme_encoder.rhyme_score(phonemes_i, phonemes_j);

                    rhyme_updates.push((line_i, score.score));
                    rhyme_updates.push((line_j, score.score));
                }
            }

            // Now apply updates
            for (line_idx, rime_score) in rhyme_updates {
                if let Some(last_word) = verse.lines[line_idx].words.last_mut() {
                    last_word.is_rhyme_word = true;
                    last_word.rhyme_score = Some(rime_score);
                }
            }
        }
    }

    /// Generate beat-synced syllable timings for a verse
    pub fn generate_timings(&self, verse: &Verse) -> Vec<SyllableTiming> {
        let mut all_syllables = Vec::new();

        // Flatten all syllables from all words
        for line in &verse.lines {
            for word in &line.words {
                // Simple syllabification: one syllable per vowel phoneme
                for phoneme in &word.phonemes {
                    let base = strip_stress(phoneme);
                    if is_vowel(&base) {
                        all_syllables.push((phoneme.clone(), word.is_rhyme_word, word.rhyme_score));
                    }
                }
            }
        }

        // Convert to string syllables for beat mapping
        let syllable_strings: Vec<String> =
            all_syllables.iter().map(|(p, _, _)| p.clone()).collect();

        // Map to beats
        let mut timings = self
            .beat_sync
            .map_syllables(&syllable_strings, &self.flow_pattern, 0);

        // Apply rhyme emphasis
        for (i, timing) in timings.iter_mut().enumerate() {
            if let Some((_, is_rhyme, _score)) = all_syllables.get(i) {
                if *is_rhyme {
                    timing.stress *= self.config.rhyme_emphasis;
                }
            }
        }

        timings
    }

    /// Convert syllable timings to timed phonemes for synthesis
    fn timings_to_phonemes(&self, timings: &[SyllableTiming], verse: &Verse) -> Vec<TimedPhoneme> {
        let mut timed_phonemes = Vec::new();
        let mut timing_idx = 0;

        for line in &verse.lines {
            for word in &line.words {
                for phoneme in &word.phonemes {
                    let base = strip_stress(phoneme);
                    let stress = extract_stress(phoneme);

                    // Vowels get timing from syllable timings
                    if is_vowel(&base) && timing_idx < timings.len() {
                        let timing = &timings[timing_idx];
                        timed_phonemes.push(TimedPhoneme {
                            phoneme: base.clone(),
                            duration: timing.duration,
                            stress,
                            start_time: timing.start_time,
                        });
                        timing_idx += 1;
                    } else {
                        // Consonants get proportional duration
                        let prev_time = timed_phonemes
                            .last()
                            .map(|p| p.start_time + p.duration)
                            .unwrap_or(0.0);

                        timed_phonemes.push(TimedPhoneme {
                            phoneme: base.clone(),
                            duration: 0.05, // Short consonant duration
                            stress: 0,
                            start_time: prev_time,
                        });
                    }
                }
            }
        }

        timed_phonemes
    }

    /// Synthesize a verse to audio
    pub fn synthesize(&mut self, verse: &Verse, pacing: &LTCPacing) -> Vec<f32> {
        // Generate timings
        let timings = self.generate_timings(verse);

        // Convert to timed phonemes
        let phonemes = self.timings_to_phonemes(&timings, verse);

        // Generate formant frames
        let frames = self.formant_synth.synthesize(&phonemes, pacing);

        // Apply rhyme pitch emphasis
        let frames = self.apply_rhyme_prosody(frames, &timings);

        // Vocode to audio
        self.vocoder.synthesize(&frames)
    }

    /// Apply prosodic emphasis to rhyming syllables
    fn apply_rhyme_prosody(
        &self,
        mut frames: Vec<FormantFrame>,
        timings: &[SyllableTiming],
    ) -> Vec<FormantFrame> {
        for timing in timings {
            if timing.stress > 1.0 {
                // Find frames in this syllable's time range
                let start = timing.start_time;
                let end = timing.start_time + timing.duration;

                for frame in &mut frames {
                    if frame.time >= start && frame.time < end {
                        // Lift pitch for rhyme emphasis
                        frame.f0 += self.config.rhyme_pitch_lift;
                        // Boost energy
                        frame.energy *= timing.stress;
                    }
                }
            }
        }

        frames
    }

    /// Synthesize from raw ARPABET string
    ///
    /// Format: "W AH1 N | T UW1" (words separated by |)
    pub fn synthesize_arpabet(&mut self, arpabet: &str, pacing: &LTCPacing) -> Vec<f32> {
        // Parse into timed phonemes
        let frames = self.formant_synth.synthesize_arpabet(arpabet, pacing);
        self.vocoder.synthesize(&frames)
    }

    /// Get current config
    pub fn config(&self) -> &RapConfig {
        &self.config
    }

    /// Get beat sync
    pub fn beat_sync(&self) -> &BeatSync {
        &self.beat_sync
    }
}

impl Default for RapSynthesizer {
    fn default() -> Self {
        Self::new(RapConfig::default())
    }
}

/// Trait for phoneme dictionary lookup
pub trait PhonemeDict {
    fn lookup(&self, word: &str) -> Vec<String>;
}

/// Simple hardcoded phoneme dictionary
pub struct SimplePhonemeDict {
    entries: std::collections::HashMap<String, Vec<String>>,
}

impl SimplePhonemeDict {
    pub fn new() -> Self {
        let mut entries = std::collections::HashMap::new();

        // Common rap words
        entries.insert("yo".to_string(), vec!["Y".to_string(), "OW1".to_string()]);
        entries.insert(
            "check".to_string(),
            vec!["CH".to_string(), "EH1".to_string(), "K".to_string()],
        );
        entries.insert("the".to_string(), vec!["DH".to_string(), "AH0".to_string()]);
        entries.insert(
            "mic".to_string(),
            vec!["M".to_string(), "AY1".to_string(), "K".to_string()],
        );
        entries.insert(
            "flow".to_string(),
            vec!["F".to_string(), "L".to_string(), "OW1".to_string()],
        );
        entries.insert("go".to_string(), vec!["G".to_string(), "OW1".to_string()]);
        entries.insert("know".to_string(), vec!["N".to_string(), "OW1".to_string()]);
        entries.insert(
            "show".to_string(),
            vec!["SH".to_string(), "OW1".to_string()],
        );
        entries.insert(
            "hot".to_string(),
            vec!["HH".to_string(), "AA1".to_string(), "T".to_string()],
        );
        entries.insert(
            "spot".to_string(),
            vec![
                "S".to_string(),
                "P".to_string(),
                "AA1".to_string(),
                "T".to_string(),
            ],
        );
        entries.insert(
            "beat".to_string(),
            vec!["B".to_string(), "IY1".to_string(), "T".to_string()],
        );
        entries.insert(
            "street".to_string(),
            vec![
                "S".to_string(),
                "T".to_string(),
                "R".to_string(),
                "IY1".to_string(),
                "T".to_string(),
            ],
        );
        entries.insert(
            "heat".to_string(),
            vec!["HH".to_string(), "IY1".to_string(), "T".to_string()],
        );
        entries.insert(
            "rhyme".to_string(),
            vec!["R".to_string(), "AY1".to_string(), "M".to_string()],
        );
        entries.insert(
            "time".to_string(),
            vec!["T".to_string(), "AY1".to_string(), "M".to_string()],
        );
        entries.insert("i".to_string(), vec!["AY1".to_string()]);
        entries.insert("my".to_string(), vec!["M".to_string(), "AY1".to_string()]);
        entries.insert(
            "got".to_string(),
            vec!["G".to_string(), "AA1".to_string(), "T".to_string()],
        );
        entries.insert(
            "what".to_string(),
            vec!["W".to_string(), "AH1".to_string(), "T".to_string()],
        );
        entries.insert("up".to_string(), vec!["AH1".to_string(), "P".to_string()]);
        entries.insert("in".to_string(), vec!["IH1".to_string(), "N".to_string()]);
        entries.insert("it".to_string(), vec!["IH1".to_string(), "T".to_string()]);
        entries.insert("is".to_string(), vec!["IH1".to_string(), "Z".to_string()]);
        entries.insert(
            "this".to_string(),
            vec!["DH".to_string(), "IH1".to_string(), "S".to_string()],
        );
        entries.insert(
            "that".to_string(),
            vec!["DH".to_string(), "AE1".to_string(), "T".to_string()],
        );
        entries.insert(
            "like".to_string(),
            vec!["L".to_string(), "AY1".to_string(), "K".to_string()],
        );
        entries.insert(
            "mike".to_string(),
            vec!["M".to_string(), "AY1".to_string(), "K".to_string()],
        );
        entries.insert(
            "night".to_string(),
            vec!["N".to_string(), "AY1".to_string(), "T".to_string()],
        );
        entries.insert(
            "right".to_string(),
            vec!["R".to_string(), "AY1".to_string(), "T".to_string()],
        );
        entries.insert(
            "tight".to_string(),
            vec!["T".to_string(), "AY1".to_string(), "T".to_string()],
        );
        entries.insert(
            "fight".to_string(),
            vec!["F".to_string(), "AY1".to_string(), "T".to_string()],
        );
        entries.insert(
            "light".to_string(),
            vec!["L".to_string(), "AY1".to_string(), "T".to_string()],
        );
        entries.insert("we".to_string(), vec!["W".to_string(), "IY1".to_string()]);
        entries.insert("me".to_string(), vec!["M".to_string(), "IY1".to_string()]);
        entries.insert("be".to_string(), vec!["B".to_string(), "IY1".to_string()]);
        entries.insert("see".to_string(), vec!["S".to_string(), "IY1".to_string()]);
        entries.insert(
            "free".to_string(),
            vec!["F".to_string(), "R".to_string(), "IY1".to_string()],
        );
        entries.insert("a".to_string(), vec!["AH0".to_string()]);
        entries.insert(
            "and".to_string(),
            vec!["AH0".to_string(), "N".to_string(), "D".to_string()],
        );
        entries.insert("to".to_string(), vec!["T".to_string(), "UW1".to_string()]);
        entries.insert("you".to_string(), vec!["Y".to_string(), "UW1".to_string()]);
        entries.insert(
            "true".to_string(),
            vec!["T".to_string(), "R".to_string(), "UW1".to_string()],
        );
        entries.insert(
            "crew".to_string(),
            vec!["K".to_string(), "R".to_string(), "UW1".to_string()],
        );
        entries.insert("new".to_string(), vec!["N".to_string(), "UW1".to_string()]);
        entries.insert("do".to_string(), vec!["D".to_string(), "UW1".to_string()]);

        Self { entries }
    }
}

impl Default for SimplePhonemeDict {
    fn default() -> Self {
        Self::new()
    }
}

impl PhonemeDict for SimplePhonemeDict {
    fn lookup(&self, word: &str) -> Vec<String> {
        let lower = word.to_lowercase();
        self.entries.get(&lower).cloned().unwrap_or_else(|| {
            // Fallback: simple letter-to-phoneme
            fallback_phonemes(&lower)
        })
    }
}

/// Fallback letter-to-phoneme conversion
fn fallback_phonemes(word: &str) -> Vec<String> {
    let mut phonemes = Vec::new();

    for c in word.chars() {
        let phon = match c {
            'a' => "AE1",
            'e' => "EH1",
            'i' => "IH1",
            'o' => "AA1",
            'u' => "AH1",
            'b' => "B",
            'c' => "K",
            'd' => "D",
            'f' => "F",
            'g' => "G",
            'h' => "HH",
            'j' => "JH",
            'k' => "K",
            'l' => "L",
            'm' => "M",
            'n' => "N",
            'p' => "P",
            'q' => "K",
            'r' => "R",
            's' => "S",
            't' => "T",
            'v' => "V",
            'w' => "W",
            'x' => "K",
            'y' => "Y",
            'z' => "Z",
            _ => continue,
        };
        phonemes.push(phon.to_string());
    }

    phonemes
}

/// Count syllables from phoneme list (count vowels)
fn count_syllables(phonemes: &[String]) -> usize {
    phonemes
        .iter()
        .filter(|p| is_vowel(&strip_stress(p)))
        .count()
        .max(1)
}

/// Check if phoneme is a vowel
fn is_vowel(phoneme: &str) -> bool {
    matches!(
        phoneme,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AX"
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
    )
}

/// Strip stress marker from phoneme
fn strip_stress(phoneme: &str) -> String {
    phoneme.chars().filter(|c| !c.is_ascii_digit()).collect()
}

/// Extract stress level (0, 1, or 2) from phoneme
fn extract_stress(phoneme: &str) -> u8 {
    phoneme
        .chars()
        .find(|c| c.is_ascii_digit())
        .and_then(|c| c.to_digit(10))
        .map(|d| d as u8)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rap_config_default() {
        let config = RapConfig::default();
        assert_eq!(config.bpm, 90.0);
        assert_eq!(config.flow_style, FlowStyle::BoomBap);
    }

    #[test]
    fn test_simple_phoneme_dict() {
        let dict = SimplePhonemeDict::new();
        let phonemes = dict.lookup("flow");
        assert!(!phonemes.is_empty());
        assert!(phonemes.contains(&"OW1".to_string()));
    }

    #[test]
    fn test_syllable_count() {
        let phonemes = vec!["F".to_string(), "L".to_string(), "OW1".to_string()];
        assert_eq!(count_syllables(&phonemes), 1);

        let phonemes2 = vec![
            "HH".to_string(),
            "EH1".to_string(),
            "L".to_string(),
            "OW0".to_string(),
        ];
        assert_eq!(count_syllables(&phonemes2), 2);
    }

    #[test]
    fn test_parse_lyrics() {
        let synth = RapSynthesizer::default();
        let dict = SimplePhonemeDict::new();

        let lyrics = "yo check the mic\nflow go show";
        let verse = synth.parse_lyrics(lyrics, &dict);

        assert_eq!(verse.lines.len(), 2);
        assert!(verse.total_syllables > 0);
    }

    #[test]
    fn test_rhyme_analysis() {
        let synth = RapSynthesizer::new(RapConfig {
            rhyme_scheme: "AA".to_string(),
            ..Default::default()
        });
        let dict = SimplePhonemeDict::new();

        // "flow" and "go" should rhyme (OW ending)
        let lyrics = "check the flow\nwatch me go";
        let mut verse = synth.parse_lyrics(lyrics, &dict);
        synth.analyze_rhymes(&mut verse);

        // Last word of each line should be marked as rhyme word
        assert!(verse.lines[0].words.last().unwrap().is_rhyme_word);
        assert!(verse.lines[1].words.last().unwrap().is_rhyme_word);
    }

    #[test]
    fn test_generate_timings() {
        let synth = RapSynthesizer::default();
        let dict = SimplePhonemeDict::new();

        let lyrics = "yo check";
        let verse = synth.parse_lyrics(lyrics, &dict);
        let timings = synth.generate_timings(&verse);

        assert!(!timings.is_empty());
        assert!(timings[0].start_time >= 0.0);
    }

    #[test]
    fn test_flow_styles() {
        for style in [
            FlowStyle::BoomBap,
            FlowStyle::Triplet,
            FlowStyle::DoubleTime,
            FlowStyle::LaidBack,
            FlowStyle::Sixteenth,
        ] {
            let pattern = style.to_pattern();
            assert!(!pattern.positions.is_empty());
        }
    }
}
