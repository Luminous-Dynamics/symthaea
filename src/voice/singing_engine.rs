// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Backend-neutral singing performance model and renderer contract.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use symthaea_muse::Note;

/// Version of the JSON-compatible vocal performance contract.
pub const VOCAL_PERFORMANCE_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PitchPoint {
    /// Seconds from the beginning of the performance.
    pub time_s: f32,
    pub frequency_hz: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExpressionPoint {
    /// Seconds from the beginning of the performance.
    pub time_s: f32,
    /// Backend-neutral controls in the inclusive range [0, 1].
    pub energy: f32,
    pub breathiness: f32,
    pub tension: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VocalArticulation {
    Legato,
    Neutral,
    Staccato,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocalPhrase {
    pub syllable_start: usize,
    /// Exclusive syllable index.
    pub syllable_end: usize,
    pub breath_before_s: f32,
    pub breath_after_s: f32,
    /// Phrase-level arc. Backends interpolate between these points.
    pub expression_curve: Vec<ExpressionPoint>,
}

/// Relative shape of a phrase-level expression arc.
///
/// Energy multipliers scale the phrase's mean syllable energy. Breathiness
/// offsets are added to its mean breathiness. Tension is an absolute [0, 1]
/// control. Each array contains start, peak and release values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhraseExpressionProfile {
    pub energy_multiplier: [f32; 3],
    pub breathiness_offset: [f32; 3],
    pub tension: [f32; 3],
}

impl Default for PhraseExpressionProfile {
    fn default() -> Self {
        Self {
            energy_multiplier: [0.86, 1.12, 0.82],
            breathiness_offset: [0.06, 0.0, 0.08],
            tension: [0.28, 0.48, 0.22],
        }
    }
}

/// Authoring inputs for one phrase. The phrase end is an exclusive
/// syllable index in the flattened lyric pronunciation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VocalPhraseSpec {
    pub syllable_end: usize,
    pub breath_before_s: f32,
    pub breath_after_s: f32,
    pub expression: PhraseExpressionProfile,
}

impl VocalPhraseSpec {
    /// Default phrasing preserves the previous restrained expression arc.
    pub fn with_end(syllable_end: usize) -> Self {
        Self {
            syllable_end,
            breath_before_s: 0.12,
            breath_after_s: 0.12,
            expression: PhraseExpressionProfile::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VocalPhoneme {
    pub ipa: String,
    pub is_vowel: bool,
    pub stress: u8,
    /// Natural (un-stretched) duration requested for consonant preservation.
    pub natural_duration_s: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalSyllable {
    /// Stable lyric fragment for diagnostics and backend export. Until the
    /// language frontend exposes grapheme spans this is the IPA concatenation.
    pub lyric: String,
    pub phonemes: Vec<VocalPhoneme>,
    pub note: Note,
    /// Additional notes sung on the same vowel without re-articulating the
    /// consonants. This is real melisma rather than repeated syllables.
    pub melisma_notes: Vec<Note>,
    pub slur: bool,
    pub articulation: VocalArticulation,
    /// Start consonants slightly before the notated vowel onset.
    pub consonant_advance_s: f32,
    pub phrase_index: usize,
    /// Portamento duration as a fraction of the note duration.
    pub portamento: f32,
    /// Vibrato begins this far through the note, as a fraction [0, 1].
    pub vibrato_onset: f32,
    pub vibrato_rate_hz: f32,
    pub vibrato_depth_cents: f32,
    /// Breathiness and energy are backend-neutral expression controls [0, 1].
    pub breathiness: f32,
    pub energy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalPerformance {
    pub version: u32,
    pub lyrics: String,
    pub language: String,
    pub syllables: Vec<VocalSyllable>,
    pub phrases: Vec<VocalPhrase>,
    pub pitch_curve: Vec<PitchPoint>,
}

/// Reject text that the current English-only G2P frontend would silently
/// skip or partially map. Punctuation at token boundaries is stripped by the
/// frontend itself; internal punctuation, digits and non-ASCII graphemes are
/// not covered by its dictionary/letter fallback and must remain explicit errors.
fn validate_english_orthography(lyrics: &str) -> Result<()> {
    for character in lyrics.chars() {
        if !(character.is_ascii_alphabetic()
            || character.is_ascii_whitespace()
            || character.is_ascii_punctuation())
        {
            bail!(
                "current English G2P accepts ASCII letters, whitespace and boundary punctuation only; unsupported character {character:?}"
            );
        }
    }

    for token in lyrics.split_whitespace() {
        let core = token.trim_matches(|character: char| !character.is_alphanumeric());
        for grapheme in core.chars() {
            if !grapheme.is_ascii_alphabetic() {
                bail!(
                    "current English G2P accepts ASCII alphabetic words with boundary punctuation only; unsupported character {grapheme:?} in token {token:?}"
                );
            }
        }
    }
    Ok(())
}

impl VocalPerformance {
    /// Bind English lyrics to a performed Muse melody and derive restrained expression.
    ///
    /// This constructor uses the current English-only Muse G2P frontend. It
    /// refuses non-English language tags and insufficient note counts rather
    /// than attaching English phones to another language or silently dropping
    /// trailing lyric syllables. Surplus notes are represented as melisma.
    pub fn from_melody(lyrics: &str, melody: &[Note], language: impl Into<String>) -> Result<Self> {
        let syllable_count = symthaea_muse::singing_bridge::syllable_count(lyrics);
        Self::from_melody_with_phrase_ends(lyrics, melody, language, &[syllable_count])
    }

    /// Compatibility constructor using explicit phrase-ending syllable indices.
    /// Phrase expression and breath values use the restrained defaults.
    pub fn from_melody_with_phrase_ends(
        lyrics: &str,
        melody: &[Note],
        language: impl Into<String>,
        phrase_ends: &[usize],
    ) -> Result<Self> {
        let phrase_specs: Vec<VocalPhraseSpec> = phrase_ends
            .iter()
            .copied()
            .enumerate()
            .map(|(index, syllable_end)| VocalPhraseSpec {
                syllable_end,
                breath_before_s: if index == 0 { 0.0 } else { 0.12 },
                breath_after_s: if index + 1 == phrase_ends.len() {
                    0.10
                } else {
                    0.12
                },
                expression: PhraseExpressionProfile::default(),
            })
            .collect();
        Self::from_melody_with_phrase_specs(lyrics, melody, language, &phrase_specs)
    }

    /// Bind lyrics and notes to fully specified vocal phrases.
    ///
    /// Each phrase owns an exclusive syllable end, explicit breath durations,
    /// and an independent start/peak/release expression profile. This avoids
    /// imposing the same generic arc on every phrase in a song.
    pub fn from_melody_with_phrase_specs(
        lyrics: &str,
        melody: &[Note],
        language: impl Into<String>,
        phrase_specs: &[VocalPhraseSpec],
    ) -> Result<Self> {
        let language = language.into();
        if language != "en" {
            bail!(
                "VocalPerformance::from_melody currently supports language='en' only; got {language:?}"
            );
        }
        if lyrics.trim().is_empty() || melody.is_empty() {
            bail!("a vocal performance requires lyrics and at least one aligned note");
        }
        validate_english_orthography(lyrics)?;
        let lexical_syllables = symthaea_muse::singing_bridge::syllable_count(lyrics);
        if lexical_syllables == 0 {
            bail!("the lyric frontend did not produce any pronounceable syllables");
        }
        if lexical_syllables > melody.len() {
            bail!(
                "insufficient melody notes: {lexical_syllables} lyric syllables require at least that many notes, got {}",
                melody.len()
            );
        }
        if phrase_specs.is_empty()
            || phrase_specs.last().map(|spec| spec.syllable_end) != Some(lexical_syllables)
        {
            bail!(
                "phrase specs must be non-empty and terminate at the final lyric syllable ({lexical_syllables})"
            );
        }
        let mut previous_phrase_end = 0usize;
        for (phrase_index, spec) in phrase_specs.iter().enumerate() {
            if spec.syllable_end <= previous_phrase_end || spec.syllable_end > lexical_syllables {
                bail!(
                    "phrase end {phrase_index} must be strictly increasing and within the lyric syllable count"
                );
            }
            if !spec.breath_before_s.is_finite()
                || !(0.0..=5.0).contains(&spec.breath_before_s)
                || !spec.breath_after_s.is_finite()
                || !(0.0..=5.0).contains(&spec.breath_after_s)
            {
                bail!(
                    "phrase {phrase_index} breath durations must be finite and within [0, 5] seconds"
                );
            }
            for value in spec.expression.energy_multiplier {
                if !value.is_finite() || !(0.0..=2.0).contains(&value) {
                    bail!(
                        "phrase {phrase_index} energy multipliers must be finite and within [0, 2]"
                    );
                }
            }
            for value in spec.expression.breathiness_offset {
                if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                    bail!(
                        "phrase {phrase_index} breathiness offsets must be finite and within [-1, 1]"
                    );
                }
            }
            for value in spec.expression.tension {
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    bail!("phrase {phrase_index} tension values must be finite and within [0, 1]");
                }
            }
            previous_phrase_end = spec.syllable_end;
        }
        let sung = symthaea_muse::singing_bridge::sing(lyrics, melody);
        if sung.len() < lexical_syllables {
            bail!(
                "lyric frontend produced {lexical_syllables} syllables but singing bridge aligned only {}",
                sung.len()
            );
        }

        // `sing` sustains the last vowel across surplus notes. Collapse those
        // notes back into one syllable with an explicit melisma representation.
        let syllables: Vec<VocalSyllable> = sung
            .into_iter()
            .take(lexical_syllables)
            .enumerate()
            .map(|(index, aligned)| {
                let duration = aligned.note.duration.max(0.06);
                let sustained = duration >= 0.28;
                let phonemes: Vec<VocalPhoneme> = aligned
                    .phonemes
                    .into_iter()
                    .map(|p| VocalPhoneme {
                        ipa: p.ipa.to_string(),
                        is_vowel: p.is_vowel,
                        stress: p.stress,
                        natural_duration_s: (p.base_duration_ms / 1000.0).clamp(0.025, 0.22),
                    })
                    .collect();
                let lyric = phonemes.iter().map(|p| p.ipa.as_str()).collect::<String>();
                let melisma_notes = if index + 1 == lexical_syllables {
                    melody.iter().skip(lexical_syllables).copied().collect()
                } else {
                    Vec::new()
                };
                VocalSyllable {
                    lyric,
                    phonemes,
                    note: aligned.note,
                    slur: !melisma_notes.is_empty(),
                    melisma_notes,
                    articulation: VocalArticulation::Legato,
                    consonant_advance_s: 0.018,
                    phrase_index: phrase_specs.partition_point(|spec| index >= spec.syllable_end),
                    portamento: if sustained { 0.12 } else { 0.06 },
                    vibrato_onset: if sustained { 0.38 } else { 1.0 },
                    vibrato_rate_hz: 5.2 + aligned.note.velocity.clamp(0.0, 1.0) * 0.5,
                    vibrato_depth_cents: if sustained {
                        16.0 + 12.0 * aligned.note.velocity
                    } else {
                        0.0
                    },
                    breathiness: (0.35 - aligned.note.velocity * 0.18).clamp(0.08, 0.4),
                    energy: aligned.note.velocity.clamp(0.0, 1.0),
                }
            })
            .collect();

        let mut pitch_curve = Vec::new();
        for (i, syllable) in syllables.iter().enumerate() {
            let previous = i
                .checked_sub(1)
                .filter(|&j| syllables[j].phrase_index == syllable.phrase_index)
                .map(|j| syllables[j].last_note().frequency)
                .unwrap_or(syllable.note.frequency);
            for (note_index, note) in syllable.notes().enumerate() {
                let from = if note_index == 0 {
                    previous
                } else {
                    syllable
                        .notes()
                        .nth(note_index - 1)
                        .map(|n| n.frequency)
                        .unwrap_or(note.frequency)
                };
                let transition = note.duration * syllable.portamento;
                pitch_curve.push(PitchPoint {
                    time_s: note.start_time,
                    frequency_hz: from,
                });
                pitch_curve.push(PitchPoint {
                    time_s: note.start_time + transition,
                    frequency_hz: note.frequency,
                });
                pitch_curve.push(PitchPoint {
                    time_s: note.start_time + note.duration.max(0.0),
                    frequency_hz: note.frequency,
                });
            }
        }

        let mut phrases = Vec::with_capacity(phrase_specs.len());
        let mut phrase_start_index = 0usize;
        for (phrase_index, phrase_spec) in phrase_specs.iter().enumerate() {
            let phrase_end_index = phrase_spec.syllable_end;
            let phrase_syllables = &syllables[phrase_start_index..phrase_end_index];
            let start = phrase_syllables
                .first()
                .map(|syllable| syllable.note.start_time)
                .unwrap_or(0.0);
            let end = phrase_syllables
                .last()
                .map(VocalSyllable::end_time_s)
                .unwrap_or(start);
            let base_energy = phrase_syllables
                .iter()
                .map(|syllable| syllable.energy)
                .sum::<f32>()
                / phrase_syllables.len() as f32;
            let base_breath = phrase_syllables
                .iter()
                .map(|syllable| syllable.breathiness)
                .sum::<f32>()
                / phrase_syllables.len() as f32;
            phrases.push(VocalPhrase {
                syllable_start: phrase_start_index,
                syllable_end: phrase_end_index,
                breath_before_s: phrase_spec.breath_before_s,
                breath_after_s: phrase_spec.breath_after_s,
                expression_curve: vec![
                    ExpressionPoint {
                        time_s: start,
                        energy: (base_energy * phrase_spec.expression.energy_multiplier[0])
                            .clamp(0.0, 1.0),
                        breathiness: (base_breath + phrase_spec.expression.breathiness_offset[0])
                            .clamp(0.0, 1.0),
                        tension: phrase_spec.expression.tension[0],
                    },
                    ExpressionPoint {
                        time_s: start + (end - start) * 0.68,
                        energy: (base_energy * phrase_spec.expression.energy_multiplier[1])
                            .clamp(0.0, 1.0),
                        breathiness: (base_breath + phrase_spec.expression.breathiness_offset[1])
                            .clamp(0.0, 1.0),
                        tension: phrase_spec.expression.tension[1],
                    },
                    ExpressionPoint {
                        time_s: end,
                        energy: (base_energy * phrase_spec.expression.energy_multiplier[2])
                            .clamp(0.0, 1.0),
                        breathiness: (base_breath + phrase_spec.expression.breathiness_offset[2])
                            .clamp(0.0, 1.0),
                        tension: phrase_spec.expression.tension[2],
                    },
                ],
            });
            phrase_start_index = phrase_end_index;
        }

        let performance = Self {
            version: VOCAL_PERFORMANCE_VERSION,
            lyrics: lyrics.trim().to_string(),
            language,
            syllables,
            phrases,
            pitch_curve,
        };
        performance.validate()?;
        Ok(performance)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != VOCAL_PERFORMANCE_VERSION {
            bail!("unsupported vocal performance version {}", self.version);
        }
        if self.lyrics.trim().is_empty() || self.syllables.is_empty() {
            bail!("lyrics and syllables are required");
        }
        if self.language != "en" {
            bail!(
                "VocalPerformance v2 currently supports language='en' only; got {:?}",
                self.language
            );
        }
        validate_english_orthography(&self.lyrics)?;
        for token in self.lyrics.split_whitespace() {
            let core = token.trim_matches(|character: char| !character.is_alphanumeric());
            if !core.is_empty() && symthaea_muse::singing_bridge::syllable_count(core) == 0 {
                bail!("English G2P produced no pronounceable syllables for lyric token {token:?}");
            }
        }
        let expected_syllables = symthaea_muse::singing_bridge::syllable_count(&self.lyrics);
        if expected_syllables == 0 || self.syllables.len() != expected_syllables {
            bail!(
                "lyric/performance syllable mismatch: G2P produced {expected_syllables} syllable(s), performance contains {}",
                self.syllables.len()
            );
        }
        let primary_notes: Vec<Note> = self
            .syllables
            .iter()
            .map(|syllable| syllable.note)
            .collect();
        let expected_alignment = symthaea_muse::singing_bridge::sing(&self.lyrics, &primary_notes);
        if expected_alignment.len() != self.syllables.len() {
            bail!(
                "lyric/G2P alignment produced {} syllables for a {}-syllable performance",
                expected_alignment.len(),
                self.syllables.len()
            );
        }
        for (syllable_index, (expected, observed)) in
            expected_alignment.iter().zip(&self.syllables).enumerate()
        {
            if expected.phonemes.len() != observed.phonemes.len()
                || expected.phonemes.iter().zip(&observed.phonemes).any(
                    |(expected_phone, observed_phone)| {
                        expected_phone.ipa != observed_phone.ipa
                            || expected_phone.is_vowel != observed_phone.is_vowel
                            || expected_phone.stress != observed_phone.stress
                    },
                )
            {
                bail!(
                    "vocal syllable {syllable_index} phone sequence does not match the selected English G2P result; use a reviewed pronunciation-plan override rather than mutating v2 phones"
                );
            }
        }

        let duration_s = self.duration_s();
        if !duration_s.is_finite() || duration_s <= 0.0 {
            bail!("vocal performance duration must be finite and positive");
        }

        let mut previous_syllable_end = 0.0f32;
        for (syllable_index, syllable) in self.syllables.iter().enumerate() {
            if syllable.lyric.trim().is_empty() || syllable.phonemes.is_empty() {
                bail!("vocal syllable {syllable_index} requires a lyric and phonemes");
            }
            let note_end = validate_note(
                &syllable.note,
                &format!("vocal syllable {syllable_index} note"),
            )?;
            if syllable.note.start_time < previous_syllable_end {
                bail!("vocal syllable {syllable_index} overlaps the preceding syllable or melisma");
            }

            if syllable.phrase_index >= self.phrases.len()
                || !syllable.consonant_advance_s.is_finite()
                || !(0.0..=0.15).contains(&syllable.consonant_advance_s)
            {
                bail!("invalid phrase or consonant timing in vocal syllable {syllable_index}");
            }
            if !syllable.portamento.is_finite()
                || !(0.0..=1.0).contains(&syllable.portamento)
                || !syllable.vibrato_onset.is_finite()
                || !(0.0..=1.0).contains(&syllable.vibrato_onset)
                || !syllable.vibrato_rate_hz.is_finite()
                || syllable.vibrato_rate_hz < 0.0
                || !syllable.vibrato_depth_cents.is_finite()
                || syllable.vibrato_depth_cents < 0.0
                || (syllable.vibrato_depth_cents > 0.0 && syllable.vibrato_rate_hz <= 0.0)
                || !syllable.breathiness.is_finite()
                || !(0.0..=1.0).contains(&syllable.breathiness)
                || !syllable.energy.is_finite()
                || !(0.0..=1.0).contains(&syllable.energy)
            {
                bail!("invalid vocal expression controls in syllable {syllable_index}");
            }

            for (phoneme_index, phoneme) in syllable.phonemes.iter().enumerate() {
                if phoneme.ipa.trim().is_empty()
                    || !phoneme.natural_duration_s.is_finite()
                    || phoneme.natural_duration_s <= 0.0
                    || phoneme.stress > 2
                {
                    bail!("invalid phoneme {phoneme_index} in vocal syllable {syllable_index}");
                }
            }

            let mut previous_note_end = note_end;
            for (melisma_index, melisma) in syllable.melisma_notes.iter().enumerate() {
                let melisma_end = validate_note(
                    melisma,
                    &format!("melisma note {melisma_index} in vocal syllable {syllable_index}"),
                )?;
                if melisma.start_time < previous_note_end - 0.000_01 {
                    bail!(
                        "melisma note {melisma_index} overlaps the preceding note in vocal syllable {syllable_index}"
                    );
                }
                // A positive gap is an intentional rest. DiffSinger emits it
                // as an SP event; Kokoro preserves it in the absolute note
                // timeline. Do not compress it or reinterpret it as vowel.
                previous_note_end = melisma_end;
            }
            previous_syllable_end = previous_note_end;
        }

        let mut next_syllable = 0usize;
        for (phrase_index, phrase) in self.phrases.iter().enumerate() {
            if phrase.syllable_start != next_syllable
                || phrase.syllable_start >= phrase.syllable_end
                || phrase.syllable_end > self.syllables.len()
                || phrase.expression_curve.is_empty()
                || !phrase.breath_before_s.is_finite()
                || phrase.breath_before_s < 0.0
                || !phrase.breath_after_s.is_finite()
                || phrase.breath_after_s < 0.0
            {
                bail!("invalid or non-partitioning vocal phrase {phrase_index}");
            }
            for syllable_index in phrase.syllable_start..phrase.syllable_end {
                if self.syllables[syllable_index].phrase_index != phrase_index {
                    bail!(
                        "vocal syllable {syllable_index} does not match its declared phrase range"
                    );
                }
            }

            let mut previous_expression_time = None;
            for (point_index, point) in phrase.expression_curve.iter().enumerate() {
                if !point.time_s.is_finite()
                    || point.time_s < 0.0
                    || point.time_s > duration_s
                    || !point.energy.is_finite()
                    || !(0.0..=1.0).contains(&point.energy)
                    || !point.breathiness.is_finite()
                    || !(0.0..=1.0).contains(&point.breathiness)
                    || !point.tension.is_finite()
                    || !(0.0..=1.0).contains(&point.tension)
                    || previous_expression_time.is_some_and(|time| point.time_s < time)
                {
                    bail!(
                        "invalid or non-chronological expression point {point_index} in phrase {phrase_index}"
                    );
                }
                previous_expression_time = Some(point.time_s);
            }
            next_syllable = phrase.syllable_end;
        }
        if next_syllable != self.syllables.len() {
            bail!("vocal phrase ranges must cover every syllable exactly once");
        }

        let mut previous_pitch_time = None;
        for (point_index, point) in self.pitch_curve.iter().enumerate() {
            if !point.time_s.is_finite()
                || point.time_s < 0.0
                || point.time_s > duration_s
                || !point.frequency_hz.is_finite()
                || point.frequency_hz <= 0.0
                || previous_pitch_time.is_some_and(|time| point.time_s < time)
            {
                bail!("invalid or non-chronological pitch curve point {point_index}");
            }
            previous_pitch_time = Some(point.time_s);
        }
        Ok(())
    }
    fn pitch_control_at(&self, time_s: f32) -> Option<(f32, &VocalSyllable, &Note)> {
        if !time_s.is_finite() || time_s < 0.0 {
            return None;
        }
        const REST_EPSILON_S: f32 = 0.000_01;
        let mut previous_frequency: Option<f32> = None;
        let mut previous_end_s = 0.0f32;
        let mut previous_phrase_index: Option<usize> = None;

        for syllable in &self.syllables {
            let begins_new_phrase =
                previous_phrase_index.is_some_and(|index| index != syllable.phrase_index);
            for note in syllable.notes() {
                let note_start = note.start_time;
                let note_end = note_start + note.duration;
                if time_s >= note_start && time_s < note_end {
                    let gap_before = note_start - previous_end_s;
                    let from_frequency = if begins_new_phrase || gap_before > REST_EPSILON_S {
                        note.frequency
                    } else {
                        previous_frequency.unwrap_or(note.frequency)
                    };
                    let elapsed = (time_s - note_start).max(0.0);
                    let transition_s = note.duration * syllable.portamento;
                    let transition = if transition_s > f32::EPSILON {
                        (elapsed / transition_s).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    let center_frequency = if from_frequency > 0.0 && note.frequency > 0.0 {
                        from_frequency * (note.frequency / from_frequency).powf(transition)
                    } else {
                        note.frequency
                    };
                    return Some((center_frequency, syllable, note));
                }
                previous_frequency = Some(note.frequency);
                previous_end_s = note_end;
            }
            previous_phrase_index = Some(syllable.phrase_index);
        }
        None
    }

    /// Evaluate the portamento-only score pitch target at an absolute time.
    ///
    /// This centre frequency excludes vibrato and backend attack/release
    /// envelopes. It is the reference around which the per-note vibrato is
    /// defined and is useful for independent waveform measurements.
    pub fn target_pitch_center_at(&self, time_s: f32) -> Option<f32> {
        self.pitch_control_at(time_s)
            .map(|(center_frequency, _, _)| center_frequency)
    }

    /// Evaluate the canonical score-conditioned pitch target at an absolute time.
    ///
    /// Portamento is interpolated in log-frequency (uniform cents), resets
    /// after an explicit score rest, and vibrato uses the frozen per-syllable
    /// onset/rate/depth controls. Renderers apply their own attack/release
    /// envelopes around this shared musical target; evaluators use the same
    /// target so control adherence is not scored against a different curve.
    pub fn target_pitch_at(&self, time_s: f32) -> Option<f32> {
        const VIBRATO_ONSET_RAMP_S: f32 = 0.05;
        let (center_frequency, syllable, note) = self.pitch_control_at(time_s)?;
        if syllable.vibrato_depth_cents <= 0.0 || syllable.vibrato_rate_hz <= 0.0 {
            return Some(center_frequency);
        }
        let vibrato_start =
            note.start_time + note.duration * syllable.vibrato_onset.clamp(0.0, 1.0);
        if time_s < vibrato_start {
            return Some(center_frequency);
        }
        let onset_gate = ((time_s - vibrato_start) / VIBRATO_ONSET_RAMP_S).clamp(0.0, 1.0);
        let phase =
            std::f32::consts::TAU * syllable.vibrato_rate_hz * (time_s - vibrato_start).max(0.0);
        let cents = syllable.vibrato_depth_cents * phase.sin() * onset_gate;
        Some(center_frequency * 2.0f32.powf(cents / 1200.0))
    }

    pub fn duration_s(&self) -> f32 {
        self.syllables
            .iter()
            .map(VocalSyllable::end_time_s)
            .fold(0.0, f32::max)
    }
}

fn validate_note(note: &Note, context: &str) -> Result<f32> {
    if !note.frequency.is_finite()
        || note.frequency <= 0.0
        || !note.start_time.is_finite()
        || note.start_time < 0.0
        || !note.duration.is_finite()
        || note.duration <= 0.0
        || !note.velocity.is_finite()
        || !(0.0..=1.0).contains(&note.velocity)
    {
        bail!("{context} has invalid pitch, timing, duration, or velocity");
    }
    let end_time = note.start_time + note.duration;
    if !end_time.is_finite() || end_time <= note.start_time {
        bail!("{context} has an invalid end time");
    }
    Ok(end_time)
}

impl VocalSyllable {
    pub fn notes(&self) -> impl Iterator<Item = &Note> {
        std::iter::once(&self.note).chain(self.melisma_notes.iter())
    }

    pub fn last_note(&self) -> &Note {
        self.melisma_notes.last().unwrap_or(&self.note)
    }

    pub fn end_time_s(&self) -> f32 {
        let note = self.last_note();
        note.start_time + note.duration
    }
}

#[derive(Debug, Clone)]
pub struct VocalStem {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub backend: String,
}

impl VocalStem {
    pub fn validate(&self) -> Result<()> {
        if self.samples.is_empty() || self.sample_rate < 8_000 || self.backend.trim().is_empty() {
            bail!("invalid or empty vocal stem");
        }
        if !self.samples.iter().all(|sample| sample.is_finite()) {
            bail!("vocal stem contains non-finite samples");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SingingControl {
    LyricText,
    Language,
    ExactPhoneSequence,
    SyllableAlignment,
    NotePitch,
    NoteDuration,
    AbsoluteOnsetAndRests,
    ConsonantAnticipation,
    Melisma,
    ContinuousPitchCurve,
    PhraseExpressionCurve,
    PhraseBreathTiming,
    Breathiness,
    Energy,
    Tension,
    Vibrato,
    Portamento,
    Articulation,
    VoiceIdentity,
    LocalEdits,
}

impl SingingControl {
    pub const ALL: [Self; 20] = [
        Self::LyricText,
        Self::Language,
        Self::ExactPhoneSequence,
        Self::SyllableAlignment,
        Self::NotePitch,
        Self::NoteDuration,
        Self::AbsoluteOnsetAndRests,
        Self::ConsonantAnticipation,
        Self::Melisma,
        Self::ContinuousPitchCurve,
        Self::PhraseExpressionCurve,
        Self::PhraseBreathTiming,
        Self::Breathiness,
        Self::Energy,
        Self::Tension,
        Self::Vibrato,
        Self::Portamento,
        Self::Articulation,
        Self::VoiceIdentity,
        Self::LocalEdits,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySupport {
    Native,
    Approximated,
    Unsupported,
    Conditional,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlCapability {
    pub control: SingingControl,
    pub support: CapabilitySupport,
    /// Must explain the behavior or the condition; never imply a control is
    /// fulfilled when the backend ignores it.
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SingingBackendCapabilities {
    pub schema: String,
    pub backend_id: String,
    pub controls: Vec<ControlCapability>,
}

impl SingingBackendCapabilities {
    pub const SCHEMA: &'static str = "singing-backend-capabilities/v1";

    pub fn new(backend_id: impl Into<String>, controls: Vec<ControlCapability>) -> Result<Self> {
        let capabilities = Self {
            schema: Self::SCHEMA.to_string(),
            backend_id: backend_id.into(),
            controls,
        };
        capabilities.validate()?;
        Ok(capabilities)
    }

    /// A renderer that has not declared its behavior is never assumed capable.
    pub fn undeclared(backend_id: impl Into<String>) -> Self {
        Self {
            schema: Self::SCHEMA.to_string(),
            backend_id: backend_id.into(),
            controls: SingingControl::ALL
                .into_iter()
                .map(|control| ControlCapability {
                    control,
                    support: CapabilitySupport::Unsupported,
                    rationale: "backend has not declared this control".to_string(),
                })
                .collect(),
        }
    }

    pub fn support(&self, control: SingingControl) -> CapabilitySupport {
        self.controls
            .iter()
            .find(|entry| entry.control == control)
            .map(|entry| entry.support)
            .unwrap_or(CapabilitySupport::Unsupported)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != Self::SCHEMA || self.backend_id.trim().is_empty() {
            bail!("invalid singing backend capability schema or backend ID");
        }
        if self.controls.len() != SingingControl::ALL.len() {
            bail!(
                "backend {} must declare all {} singing controls, got {}",
                self.backend_id,
                SingingControl::ALL.len(),
                self.controls.len()
            );
        }
        let mut seen = std::collections::HashSet::new();
        for entry in &self.controls {
            if !seen.insert(entry.control) {
                bail!(
                    "backend {} declares {:?} more than once",
                    self.backend_id,
                    entry.control
                );
            }
            if entry.rationale.trim().is_empty() {
                bail!(
                    "backend {} has an empty rationale for {:?}",
                    self.backend_id,
                    entry.control
                );
            }
        }
        if SingingControl::ALL
            .iter()
            .any(|control| !seen.contains(control))
        {
            bail!(
                "backend {} has an incomplete singing capability manifest",
                self.backend_id
            );
        }
        Ok(())
    }
}

/// Requirements a caller places on one render. This makes backend control
/// claims executable instead of leaving the capability manifest as metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SingingRenderPolicy {
    pub required_controls: Vec<SingingControl>,
    pub allow_approximated: bool,
    pub allow_conditional: bool,
}

impl SingingRenderPolicy {
    pub fn new(
        required_controls: Vec<SingingControl>,
        allow_approximated: bool,
        allow_conditional: bool,
    ) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        for control in &required_controls {
            if !seen.insert(*control) {
                bail!("render policy declares {control:?} more than once");
            }
        }
        Ok(Self {
            required_controls,
            allow_approximated,
            allow_conditional,
        })
    }

    /// Additional controls for comparisons specifically targeting phrase
    /// expression curves. Backends may qualify with documented approximations.
    pub fn phrase_expression_comparison() -> Self {
        Self {
            required_controls: vec![SingingControl::PhraseExpressionCurve],
            allow_approximated: true,
            allow_conditional: true,
        }
    }

    /// Separate qualification for breath-before/after timing and audible
    /// inhalation behavior. Explicit score rests do not satisfy this policy.
    pub fn phrase_breath_comparison() -> Self {
        Self {
            required_controls: vec![SingingControl::PhraseBreathTiming],
            allow_approximated: false,
            allow_conditional: true,
        }
    }

    /// Minimum common contract for controlled score-conditioned comparisons.
    /// Lyric text is deliberately absent: a phoneme-native backend may consume
    /// an independently selected pronunciation plan rather than raw text.
    pub fn score_conditioned_comparison() -> Self {
        Self {
            required_controls: vec![
                SingingControl::Language,
                SingingControl::ExactPhoneSequence,
                SingingControl::SyllableAlignment,
                SingingControl::NotePitch,
                SingingControl::NoteDuration,
                SingingControl::AbsoluteOnsetAndRests,
                SingingControl::Melisma,
            ],
            allow_approximated: true,
            allow_conditional: true,
        }
    }

    pub fn validate_against(&self, capabilities: &SingingBackendCapabilities) -> Result<()> {
        capabilities.validate()?;
        for control in &self.required_controls {
            let support = capabilities.support(*control);
            let accepted = match support {
                CapabilitySupport::Native => true,
                CapabilitySupport::Approximated => self.allow_approximated,
                CapabilitySupport::Conditional => self.allow_conditional,
                CapabilitySupport::Unsupported => false,
            };
            if !accepted {
                bail!(
                    "backend {} cannot satisfy required control {control:?}: declared {support:?}",
                    capabilities.backend_id
                );
            }
        }
        Ok(())
    }
}

pub trait SingingVoiceEngine {
    fn id(&self) -> &str;

    /// Backends must override this to make supported and unsupported controls
    /// explicit. The default is fail-closed and treats every control as
    /// unsupported rather than assuming parity between implementations.
    fn capabilities(&self) -> SingingBackendCapabilities {
        SingingBackendCapabilities::undeclared(self.id())
    }

    /// Validate the performance and requested control contract before calling
    /// the renderer. Legacy render remains available for existing callers;
    /// controlled comparisons should use this policy-aware entrypoint.
    fn render_with_policy(
        &mut self,
        performance: &VocalPerformance,
        policy: &SingingRenderPolicy,
    ) -> Result<VocalStem> {
        performance.validate()?;
        let capabilities = self.capabilities();
        if capabilities.backend_id != self.id() {
            bail!(
                "backend capability ID {:?} does not match renderer ID {:?}",
                capabilities.backend_id,
                self.id()
            );
        }
        policy.validate_against(&capabilities)?;
        self.render(performance)
    }

    fn render(&mut self, performance: &VocalPerformance) -> Result<VocalStem>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undeclared_backend_controls_are_fail_closed() {
        let capabilities = SingingBackendCapabilities::undeclared("unverified-backend");
        capabilities.validate().unwrap();
        assert_eq!(
            capabilities.support(SingingControl::ExactPhoneSequence),
            CapabilitySupport::Unsupported
        );
        assert_eq!(capabilities.controls.len(), SingingControl::ALL.len());
    }

    #[test]
    fn capability_manifest_rejects_missing_duplicate_or_empty_entries() {
        let mut controls: Vec<_> = SingingControl::ALL
            .into_iter()
            .map(|control| ControlCapability {
                control,
                support: CapabilitySupport::Unsupported,
                rationale: "not supported by fixture".to_string(),
            })
            .collect();
        controls.pop();
        assert!(SingingBackendCapabilities::new("fixture", controls).is_err());

        let mut controls: Vec<_> = SingingControl::ALL
            .into_iter()
            .map(|control| ControlCapability {
                control,
                support: CapabilitySupport::Native,
                rationale: "fixture support".to_string(),
            })
            .collect();
        controls[0].control = controls[1].control;
        assert!(SingingBackendCapabilities::new("fixture", controls).is_err());

        let controls = SingingControl::ALL
            .into_iter()
            .map(|control| ControlCapability {
                control,
                support: CapabilitySupport::Native,
                rationale: String::new(),
            })
            .collect();
        assert!(SingingBackendCapabilities::new("fixture", controls).is_err());
    }

    fn capability_fixture(support: CapabilitySupport) -> SingingBackendCapabilities {
        SingingBackendCapabilities::new(
            "fixture",
            SingingControl::ALL
                .into_iter()
                .map(|control| ControlCapability {
                    control,
                    support,
                    rationale: "test fixture declaration".to_string(),
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn render_policy_admits_only_declared_control_levels() {
        let native = capability_fixture(CapabilitySupport::Native);
        let policy = SingingRenderPolicy::new(
            vec![
                SingingControl::ExactPhoneSequence,
                SingingControl::NotePitch,
            ],
            false,
            false,
        )
        .unwrap();
        assert!(policy.validate_against(&native).is_ok());

        let approximated = capability_fixture(CapabilitySupport::Approximated);
        assert!(policy.validate_against(&approximated).is_err());
        let permissive =
            SingingRenderPolicy::new(vec![SingingControl::ExactPhoneSequence], true, false)
                .unwrap();
        assert!(permissive.validate_against(&approximated).is_ok());

        let conditional = capability_fixture(CapabilitySupport::Conditional);
        assert!(permissive.validate_against(&conditional).is_err());
        let conditional_policy =
            SingingRenderPolicy::new(vec![SingingControl::Language], false, true).unwrap();
        assert!(conditional_policy.validate_against(&conditional).is_ok());

        let unsupported = SingingBackendCapabilities::undeclared("unverified");
        assert!(conditional_policy.validate_against(&unsupported).is_err());
    }

    struct MismatchedCapabilityBackend {
        render_called: bool,
    }

    impl SingingVoiceEngine for MismatchedCapabilityBackend {
        fn id(&self) -> &str {
            "actual-renderer-id"
        }

        fn capabilities(&self) -> SingingBackendCapabilities {
            capability_fixture(CapabilitySupport::Native)
        }

        fn render(&mut self, _performance: &VocalPerformance) -> Result<VocalStem> {
            self.render_called = true;
            bail!("render must not be dispatched after a capability identity mismatch");
        }
    }

    #[test]
    fn policy_render_rejects_a_manifest_for_a_different_backend_id() {
        let note = Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        let performance = VocalPerformance::from_melody("light", &[note], "en").unwrap();
        let mut backend = MismatchedCapabilityBackend {
            render_called: false,
        };
        let error = backend
            .render_with_policy(
                &performance,
                &SingingRenderPolicy::score_conditioned_comparison(),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not match renderer ID"), "{error}");
        assert!(!backend.render_called);
    }

    #[test]
    fn render_policy_rejects_duplicate_required_controls() {
        let error = SingingRenderPolicy::new(
            vec![SingingControl::NotePitch, SingingControl::NotePitch],
            true,
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("more than once"), "{error}");
    }

    #[test]
    fn performance_is_serializable_and_validated() {
        let melody = [Note {
            frequency: 261.63,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        }];
        let performance = VocalPerformance::from_melody("light", &melody, "en").unwrap();
        let json = serde_json::to_string(&performance).unwrap();
        let decoded: VocalPerformance = serde_json::from_str(&json).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.syllables.len(), 1);
        assert!(decoded.pitch_curve.len() >= 3);
        assert_eq!(decoded.version, 2);
        assert_eq!(decoded.phrases.len(), 1);
    }

    fn performance_with_melisma() -> VocalPerformance {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.2,
                velocity: 0.7,
            },
            Note {
                frequency: 246.94,
                start_time: 0.2,
                duration: 0.2,
                velocity: 0.7,
            },
            Note {
                frequency: 261.63,
                start_time: 0.4,
                duration: 0.3,
                velocity: 0.7,
            },
        ];
        VocalPerformance::from_melody("light", &notes, "en").unwrap()
    }

    #[test]
    fn validate_rejects_non_english_or_unpronounceable_serialized_performance() {
        let mut non_english = performance_with_melisma();
        non_english.language = "zu".into();
        let error = non_english.validate().unwrap_err().to_string();
        assert!(error.contains("language='en' only"), "{error}");

        let mut no_g2p = performance_with_melisma();
        no_g2p.lyrics = "q light".into();
        let error = no_g2p.validate().unwrap_err().to_string();
        assert!(error.contains("no pronounceable syllables"), "{error}");
    }

    #[test]
    fn validate_rejects_phone_identity_changes_even_when_syllable_count_matches() {
        let mut performance = performance_with_melisma();
        performance.syllables[0].phonemes[0].ipa = "z".into();
        let error = performance.validate().unwrap_err().to_string();
        assert!(error.contains("phone sequence does not match"), "{error}");
    }

    #[test]
    fn validate_rejects_lyric_syllable_count_mismatch() {
        let mut performance = performance_with_melisma();
        performance.lyrics = "light bright".into();
        let error = performance.validate().unwrap_err().to_string();
        assert!(
            error.contains("lyric/performance syllable mismatch"),
            "{error}"
        );
    }

    #[test]
    fn non_finite_melisma_values_are_rejected() {
        let mut performance = performance_with_melisma();
        performance.syllables[0].melisma_notes[0].duration = f32::NAN;
        assert!(performance.validate().is_err());
    }

    #[test]
    fn syllable_cannot_overlap_a_preceding_melisma_span() {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.2,
                velocity: 0.7,
            },
            Note {
                frequency: 246.94,
                start_time: 0.5,
                duration: 0.2,
                velocity: 0.7,
            },
        ];
        let mut performance = VocalPerformance::from_melody("light bright", &notes, "en").unwrap();
        performance.syllables[0].melisma_notes.push(Note {
            frequency: 233.08,
            start_time: 0.2,
            duration: 0.4,
            velocity: 0.7,
        });
        let error = performance.validate().unwrap_err().to_string();
        assert!(
            error.contains("overlaps the preceding syllable or melisma"),
            "{error}"
        );
    }

    #[test]
    fn explicit_melisma_gaps_are_preserved_as_rests() {
        let mut performance = performance_with_melisma();
        performance.syllables[0].melisma_notes[1].start_time += 0.03;
        performance.validate().unwrap();
        let first_end = performance.syllables[0].melisma_notes[0].start_time
            + performance.syllables[0].melisma_notes[0].duration;
        assert!(
            (performance.syllables[0].melisma_notes[1].start_time - first_end - 0.03).abs() < 1e-5
        );
    }

    #[test]
    fn overlapping_melisma_notes_are_rejected() {
        let mut performance = performance_with_melisma();
        performance.syllables[0].melisma_notes[0].start_time = 0.19;
        let error = performance.validate().unwrap_err().to_string();
        assert!(error.contains("overlap"), "{error}");
    }

    #[test]
    fn invalid_note_velocity_and_phoneme_duration_are_rejected() {
        let mut invalid_velocity = performance_with_melisma();
        invalid_velocity.syllables[0].note.velocity = f32::INFINITY;
        assert!(invalid_velocity.validate().is_err());

        let mut invalid_phone = performance_with_melisma();
        invalid_phone.syllables[0].phonemes[0].natural_duration_s = f32::NAN;
        assert!(invalid_phone.validate().is_err());
    }

    #[test]
    fn non_finite_and_non_chronological_pitch_curves_are_rejected() {
        let mut non_finite = performance_with_melisma();
        non_finite.pitch_curve[0].frequency_hz = f32::NAN;
        assert!(non_finite.validate().is_err());

        let mut backwards = performance_with_melisma();
        backwards.pitch_curve[0].time_s = 0.05;
        backwards.pitch_curve[1].time_s = 0.0;
        let error = backwards.validate().unwrap_err().to_string();
        assert!(error.contains("pitch curve point"), "{error}");
    }

    #[test]
    fn expression_curves_must_be_chronological_and_phrase_ranges_complete() {
        let mut backwards = performance_with_melisma();
        backwards.phrases[0].expression_curve[0].time_s = 0.1;
        backwards.phrases[0].expression_curve[1].time_s = 0.0;
        assert!(backwards.validate().is_err());

        let mut missing_syllable = performance_with_melisma();
        missing_syllable.phrases[0].syllable_start = 1;
        assert!(missing_syllable.validate().is_err());
    }

    #[test]
    fn surplus_notes_become_melisma() {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.2,
                velocity: 0.7,
            },
            Note {
                frequency: 246.94,
                start_time: 0.2,
                duration: 0.2,
                velocity: 0.7,
            },
            Note {
                frequency: 261.63,
                start_time: 0.4,
                duration: 0.3,
                velocity: 0.7,
            },
        ];
        let performance = VocalPerformance::from_melody("light", &notes, "en").unwrap();
        assert_eq!(performance.syllables.len(), 1);
        assert_eq!(performance.syllables[0].melisma_notes.len(), 2);
        assert!((performance.duration_s() - 0.7).abs() < 1e-4);
    }
    #[test]
    fn shared_pitch_target_applies_log_portamento_and_performance_vibrato() {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 440.0,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.7,
            },
        ];
        let mut performance = VocalPerformance::from_melody("light bright", &notes, "en").unwrap();
        performance.syllables[0].vibrato_depth_cents = 0.0;
        performance.syllables[1].portamento = 0.5;
        performance.syllables[1].vibrato_onset = 0.5;
        performance.syllables[1].vibrato_rate_hz = 5.0;
        performance.syllables[1].vibrato_depth_cents = 20.0;

        let at_glide = performance.target_pitch_at(0.6).unwrap();
        let expected_glide = 220.0 * 2.0f32.powf(0.4);
        assert!(
            (at_glide - expected_glide).abs() < 0.01,
            "expected log-frequency glide {expected_glide}, got {at_glide}"
        );
        assert!(
            (performance.target_pitch_at(0.75).unwrap() - 440.0).abs() < 0.01,
            "portamento must reach target before vibrato onset"
        );
        let at_vibrato_onset = performance.target_pitch_at(0.75).unwrap();
        let after_onset = performance.target_pitch_at(0.80).unwrap();
        assert!(
            (after_onset - at_vibrato_onset).abs() > 0.1,
            "per-syllable vibrato must alter the shared score target after onset"
        );
        assert_eq!(performance.target_pitch_at(1.0), None);
        assert_eq!(performance.target_pitch_at(f32::NAN), None);
    }

    #[test]
    fn shared_pitch_target_resets_portamento_across_explicit_rest() {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 440.0,
                start_time: 0.8,
                duration: 0.5,
                velocity: 0.7,
            },
        ];
        let mut performance = VocalPerformance::from_melody("light bright", &notes, "en").unwrap();
        performance.syllables[1].portamento = 1.0;
        assert!(
            (performance.target_pitch_at(0.81).unwrap() - 440.0).abs() < 0.01,
            "a score rest must reset the preceding pitch for portamento"
        );
    }

    #[test]
    fn from_melody_supports_explicit_multiple_phrase_boundaries() {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 246.94,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.7,
            },
        ];
        let performance =
            VocalPerformance::from_melody_with_phrase_ends("light bright", &notes, "en", &[1, 2])
                .unwrap();
        assert_eq!(performance.phrases.len(), 2);
        assert_eq!(performance.syllables[0].phrase_index, 0);
        assert_eq!(performance.syllables[1].phrase_index, 1);
        assert_eq!(performance.phrases[0].syllable_start, 0);
        assert_eq!(performance.phrases[0].syllable_end, 1);
        assert_eq!(performance.phrases[1].syllable_start, 1);
        assert_eq!(performance.phrases[1].syllable_end, 2);
        assert!(performance.phrases[0].breath_after_s > 0.0);
        assert!(performance.phrases[1].breath_before_s > 0.0);
        performance.validate().unwrap();
    }

    #[test]
    fn phrase_specs_apply_independent_breath_and_expression_arcs() {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 246.94,
                start_time: 1.12,
                duration: 0.5,
                velocity: 0.7,
            },
        ];
        let phrase_specs = [
            VocalPhraseSpec {
                syllable_end: 1,
                breath_before_s: 0.0,
                breath_after_s: 0.25,
                expression: PhraseExpressionProfile {
                    energy_multiplier: [0.5, 1.5, 0.6],
                    breathiness_offset: [-0.1, 0.2, -0.2],
                    tension: [0.1, 0.9, 0.2],
                },
            },
            VocalPhraseSpec {
                syllable_end: 2,
                breath_before_s: 0.37,
                breath_after_s: 0.18,
                expression: PhraseExpressionProfile {
                    energy_multiplier: [1.0, 0.8, 0.4],
                    breathiness_offset: [0.0, -0.1, 0.3],
                    tension: [0.3, 0.4, 0.05],
                },
            },
        ];
        let performance = VocalPerformance::from_melody_with_phrase_specs(
            "light bright",
            &notes,
            "en",
            &phrase_specs,
        )
        .unwrap();

        assert_eq!(performance.phrases.len(), 2);
        assert_eq!(performance.phrases[0].breath_after_s, 0.25);
        assert_eq!(performance.phrases[1].breath_before_s, 0.37);
        assert_eq!(performance.phrases[1].breath_after_s, 0.18);

        let first = &performance.phrases[0].expression_curve;
        assert!((first[0].energy - 0.35).abs() < 1e-5);
        assert_eq!(first[0].tension, 0.1);
        assert_eq!(first[1].tension, 0.9);
        assert_eq!(first[2].tension, 0.2);

        let second = &performance.phrases[1].expression_curve;
        assert_eq!(second[0].tension, 0.3);
        assert_eq!(second[1].tension, 0.4);
        assert_eq!(second[2].tension, 0.05);
        performance.validate().unwrap();
    }

    #[test]
    fn pitch_curve_does_not_glide_across_a_phrase_boundary() {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 440.0,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.7,
            },
        ];
        let phrase_ends = [1, 2];
        let performance = VocalPerformance::from_melody_with_phrase_ends(
            "light bright",
            &notes,
            "en",
            &phrase_ends,
        )
        .unwrap();

        let second_phrase_onset = performance
            .pitch_curve
            .iter()
            .find(|point| (point.time_s - 1.0).abs() < 1e-6)
            .expect("second phrase onset must be represented in the pitch curve");
        assert_eq!(second_phrase_onset.frequency_hz, 440.0);
        assert_eq!(
            performance.target_pitch_center_at(1.0),
            Some(440.0),
            "the canonical renderer/evaluator target must also reset at a phrase boundary"
        );
    }

    #[test]
    fn phrase_expression_and_phrase_breath_policies_are_separate() {
        let expression = SingingRenderPolicy::phrase_expression_comparison();
        assert_eq!(
            expression.required_controls,
            vec![SingingControl::PhraseExpressionCurve]
        );
        let breath = SingingRenderPolicy::phrase_breath_comparison();
        assert_eq!(
            breath.required_controls,
            vec![SingingControl::PhraseBreathTiming]
        );
        assert!(
            !SingingRenderPolicy::score_conditioned_comparison()
                .required_controls
                .contains(&SingingControl::PhraseBreathTiming)
        );
    }

    #[test]
    fn phrase_specs_reject_non_finite_and_out_of_range_expression_controls() {
        let note = Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        let mut invalid = VocalPhraseSpec::with_end(1);
        invalid.expression.energy_multiplier[1] = f32::NAN;
        assert!(
            VocalPerformance::from_melody_with_phrase_specs("light", &[note], "en", &[invalid],)
                .unwrap_err()
                .to_string()
                .contains("energy multipliers")
        );

        let mut invalid = VocalPhraseSpec::with_end(1);
        invalid.breath_after_s = 6.0;
        assert!(
            VocalPerformance::from_melody_with_phrase_specs("light", &[note], "en", &[invalid],)
                .unwrap_err()
                .to_string()
                .contains("breath durations")
        );
    }

    #[test]
    fn from_melody_rejects_non_partitioning_phrase_boundaries() {
        let notes = [
            Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.7,
            },
            Note {
                frequency: 246.94,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.7,
            },
        ];
        for phrase_ends in [&[][..], &[1][..], &[2, 1][..], &[1, 3][..]] {
            assert!(
                VocalPerformance::from_melody_with_phrase_ends(
                    "light bright",
                    &notes,
                    "en",
                    phrase_ends
                )
                .is_err()
            );
        }
    }

    #[test]
    fn from_melody_rejects_non_english_g2p_profiles() {
        let note = Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        let error = VocalPerformance::from_melody("light", &[note], "zu")
            .unwrap_err()
            .to_string();
        assert!(error.contains("language='en' only"));
    }

    #[test]
    fn from_melody_rejects_uncovered_graphemes_internal_punctuation_and_digits() {
        let note = Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        for lyrics in ["café", "don't", "2026", "light 🎵 bright"] {
            let error = VocalPerformance::from_melody(lyrics, &[note], "en")
                .unwrap_err()
                .to_string();
            assert!(error.contains("unsupported character"), "{lyrics}: {error}");
        }
    }

    #[test]
    fn from_melody_accepts_boundary_punctuation_around_supported_words() {
        let note = Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        assert!(VocalPerformance::from_melody("\"light,\"", &[note], "en").is_ok());
    }

    #[test]
    fn from_melody_rejects_insufficient_notes_instead_of_dropping_lyrics() {
        let note = Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        let error = VocalPerformance::from_melody("light bright", &[note], "en")
            .unwrap_err()
            .to_string();
        assert!(error.contains("insufficient melody notes"), "{error}");
    }
}
