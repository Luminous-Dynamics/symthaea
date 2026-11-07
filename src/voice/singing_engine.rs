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

impl VocalPerformance {
    /// Bind English lyrics to a performed Muse melody and derive restrained expression.
    ///
    /// This constructor uses the current English-only Muse G2P frontend. It
    /// refuses non-English language tags and insufficient note counts rather
    /// than attaching English phones to another language or silently dropping
    /// trailing lyric syllables. Surplus notes are represented as melisma.
    pub fn from_melody(lyrics: &str, melody: &[Note], language: impl Into<String>) -> Result<Self> {
        let language = language.into();
        if language != "en" {
            bail!(
                "VocalPerformance::from_melody currently supports language='en' only; got {language:?}"
            );
        }
        if lyrics.trim().is_empty() || melody.is_empty() {
            bail!("a vocal performance requires lyrics and at least one aligned note");
        }
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
                    phrase_index: 0,
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

        let start = syllables.first().map(|s| s.note.start_time).unwrap_or(0.0);
        let end = syllables
            .last()
            .map(VocalSyllable::end_time_s)
            .unwrap_or(start);
        let base_energy = syllables.iter().map(|s| s.energy).sum::<f32>() / syllables.len() as f32;
        let base_breath =
            syllables.iter().map(|s| s.breathiness).sum::<f32>() / syllables.len() as f32;
        let phrases = vec![VocalPhrase {
            syllable_start: 0,
            syllable_end: syllables.len(),
            breath_before_s: 0.0,
            breath_after_s: 0.10,
            expression_curve: vec![
                ExpressionPoint {
                    time_s: start,
                    energy: base_energy * 0.86,
                    breathiness: (base_breath + 0.06).min(1.0),
                    tension: 0.28,
                },
                ExpressionPoint {
                    time_s: start + (end - start) * 0.68,
                    energy: (base_energy + 0.12).min(1.0),
                    breathiness: base_breath,
                    tension: 0.48,
                },
                ExpressionPoint {
                    time_s: end,
                    energy: base_energy * 0.82,
                    breathiness: (base_breath + 0.08).min(1.0),
                    tension: 0.22,
                },
            ],
        }];

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
        if self.lyrics.trim().is_empty()
            || self.language.trim().is_empty()
            || self.syllables.is_empty()
        {
            bail!("lyrics, language, and syllables are required");
        }

        let duration_s = self.duration_s();
        if !duration_s.is_finite() || duration_s <= 0.0 {
            bail!("vocal performance duration must be finite and positive");
        }

        let mut previous_syllable_start = 0.0f32;
        for (syllable_index, syllable) in self.syllables.iter().enumerate() {
            if syllable.lyric.trim().is_empty() || syllable.phonemes.is_empty() {
                bail!("vocal syllable {syllable_index} requires a lyric and phonemes");
            }
            let note_end = validate_note(
                &syllable.note,
                &format!("vocal syllable {syllable_index} note"),
            )?;
            if syllable.note.start_time < previous_syllable_start {
                bail!("vocal syllables must be chronological");
            }
            previous_syllable_start = syllable.note.start_time;

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
                if melisma.start_time < previous_note_end {
                    bail!(
                        "melisma notes overlap or are not chronological in vocal syllable {syllable_index}"
                    );
                }
                previous_note_end = melisma_end;
            }
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

pub trait SingingVoiceEngine {
    fn id(&self) -> &str;
    fn render(&mut self, performance: &VocalPerformance) -> Result<VocalStem>;
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn non_finite_melisma_values_are_rejected() {
        let mut performance = performance_with_melisma();
        performance.syllables[0].melisma_notes[0].duration = f32::NAN;
        assert!(performance.validate().is_err());
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
