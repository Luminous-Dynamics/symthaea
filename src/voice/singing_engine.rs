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
    /// Bind lyrics to a performed Muse melody and derive restrained expression.
    pub fn from_melody(lyrics: &str, melody: &[Note], language: impl Into<String>) -> Result<Self> {
        let sung = symthaea_muse::singing_bridge::sing(lyrics, melody);
        if lyrics.trim().is_empty() || sung.is_empty() {
            bail!("a vocal performance requires lyrics and at least one aligned note");
        }

        // `sing` sustains the last vowel across surplus notes. Collapse those
        // notes back into one syllable with an explicit melisma representation.
        let lexical_syllables = symthaea_muse::singing_bridge::syllable_count(lyrics)
            .min(sung.len())
            .max(1);
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
            language: language.into(),
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
        let mut previous_start = 0.0f32;
        for (i, syllable) in self.syllables.iter().enumerate() {
            let note = syllable.note;
            if !note.frequency.is_finite()
                || note.frequency <= 0.0
                || !note.start_time.is_finite()
                || note.start_time < 0.0
                || !note.duration.is_finite()
                || note.duration <= 0.0
                || syllable.phonemes.is_empty()
            {
                bail!("invalid vocal syllable {i}");
            }
            if syllable.phrase_index >= self.phrases.len()
                || !syllable.consonant_advance_s.is_finite()
                || syllable.consonant_advance_s < 0.0
                || syllable.consonant_advance_s > 0.15
            {
                bail!("invalid phrase or consonant timing in vocal syllable {i}");
            }
            let mut note_start = note.start_time;
            for melisma in &syllable.melisma_notes {
                if !melisma.frequency.is_finite()
                    || melisma.frequency <= 0.0
                    || melisma.start_time < note_start
                    || melisma.duration <= 0.0
                {
                    bail!("invalid melisma note in vocal syllable {i}");
                }
                note_start = melisma.start_time;
            }
            if i > 0 && note.start_time < previous_start {
                bail!("vocal syllables must be chronological");
            }
            previous_start = note.start_time;
        }
        for (i, phrase) in self.phrases.iter().enumerate() {
            if phrase.syllable_start >= phrase.syllable_end
                || phrase.syllable_end > self.syllables.len()
                || phrase.expression_curve.is_empty()
                || phrase.breath_before_s < 0.0
                || phrase.breath_after_s < 0.0
            {
                bail!("invalid vocal phrase {i}");
            }
            if phrase.expression_curve.iter().any(|point| {
                !point.time_s.is_finite()
                    || !point.energy.is_finite()
                    || !(0.0..=1.0).contains(&point.energy)
                    || !(0.0..=1.0).contains(&point.breathiness)
                    || !(0.0..=1.0).contains(&point.tension)
            }) {
                bail!("invalid expression curve in vocal phrase {i}");
            }
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
}
