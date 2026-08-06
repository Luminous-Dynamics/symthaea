// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Score: the symbolic output of composition — voiced notes with absolute
//! timing, dynamics, and structural annotations. This is the boundary
//! `symthaea-muse` consumes and *realizes* as audio.
//!
//! Still no Hz: pitches are symbolic [`Pitch`] (MIDI), time is in beats plus a
//! tempo. muse chooses tuning, timbre, expressive micro-timing, and rendering.

use crate::harmony::Key;
use crate::meter::TimeSignature;
use crate::pitch::Pitch;
use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};

/// Which continuing musical line produced a note.
///
/// Deliberately separate from [`VoiceRole`], and the distinction is the point:
///
/// - `PartId` answers **"which persistent agent is playing this?"**
/// - `VoiceRole` answers **"what function is that line performing here?"**
///
/// One part can change role (a fugue's answer becomes an accompanying line);
/// two parts can share a role (two contrapuntal lines both labelled `Melody`).
/// Grouping notes by role and calling the result "voices" conflates them — a
/// mistake this codebase made in a style probe on 2026-07-29, reporting
/// "textbook staggered exposition" from what was only role-class entry.
///
/// # Never infer this
///
/// A `PartId` must be assigned by the generator that knows which line it is
/// writing. Reconstructing it afterwards from register, role, MIDI channel or
/// temporal overlap produces a plausible answer that is wrong exactly when it
/// matters — in dense counterpoint, which is the case you built it for.
/// Generators that genuinely do not track parts must emit
/// [`PartId::UNASSIGNED`] rather than a guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PartId(pub u16);

impl PartId {
    /// "This generator does not track parts." NOT part zero.
    ///
    /// Analysis that needs real part identity must check for this and refuse
    /// to answer, rather than treating every unassigned note as one shared
    /// line — see [`Score::has_part_identity`].
    pub const UNASSIGNED: PartId = PartId(u16::MAX);

    pub const fn is_assigned(self) -> bool {
        self.0 != u16::MAX
    }
}

/// Which contrapuntal voice a note belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceRole {
    /// The tune — the developed motivic line.
    Melody,
    /// Inner harmony / accompaniment chord tones.
    Harmony,
    /// The bass line (chord roots).
    Bass,
    /// A second, slower melodic line in dialogue with the melody (voice-led
    /// chord tones, collision-avoided) — appears in the RETURN section,
    /// where classical practice earns the reprise new interest.
    CounterMelody,
}

/// Structural role of the moment a note occurs in — lets muse apply expressive
/// timing/dynamics tied to STRUCTURE (rubato at cadences, swell to the climax)
/// rather than the old random jitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Emphasis {
    Normal,
    /// This note is the phrase/section climax (the melodic goal).
    Climax,
    /// This note voices a cadence (phrase close) — invites ritardando.
    Cadential,
    /// Phrase beginning — invites a small breath before it.
    PhraseStart,
}

/// One note in the score, with absolute timing in beats from the start.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoreNote {
    /// Which continuing line produced this note. [`PartId::UNASSIGNED`] means
    /// the generator did not track parts — it does NOT mean "part zero".
    pub part: PartId,
    pub pitch: Pitch,
    /// Onset in beats from the start of the score.
    pub onset: Duration,
    pub duration: Duration,
    /// Loudness [0, 1].
    pub velocity: f32,
    pub role: VoiceRole,
    pub emphasis: Emphasis,
    /// This note's section's long-range dynamic-intensity multiplier (see
    /// [`crate::form::SectionRole::intensity`]) — already folded into
    /// `velocity`, but exposed separately so a consumer (muse) can react
    /// structurally too, e.g. switching to a more intense instrument during
    /// the piece's peak section, not just playing it louder.
    pub section_intensity: f32,
}

/// A complete symbolic composition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub key: Key,
    pub tempo_bpm: f32,
    /// Beats per bar (numerator of the time signature).
    pub meter: u8,
    pub notes: Vec<ScoreNote>,
    /// Total length in beats.
    pub total_beats: Duration,
}

impl Score {
    pub fn new(key: Key, tempo_bpm: f32, meter: u8) -> Self {
        Score {
            key,
            tempo_bpm,
            meter,
            notes: Vec::new(),
            total_beats: Duration::zero(),
        }
    }

    /// Does every note carry a real part identity?
    ///
    /// Analysis that depends on knowing which line a note belongs to — fugue
    /// subject recurrence, passacaglia ground invariance, true voice-entry
    /// stagger — must gate on this and decline rather than answer from
    /// [`VoiceRole`], which is a different thing.
    pub fn has_part_identity(&self) -> bool {
        !self.notes.is_empty() && self.notes.iter().all(|n| n.part.is_assigned())
    }

    /// The distinct parts present, in first-onset order. Empty when the score
    /// carries no part identity at all.
    pub fn parts(&self) -> Vec<PartId> {
        let mut seen: Vec<PartId> = Vec::new();
        for n in &self.notes {
            if n.part.is_assigned() && !seen.contains(&n.part) {
                seen.push(n.part);
            }
        }
        seen
    }

    pub fn push(&mut self, note: ScoreNote) {
        let end = note.onset + note.duration;
        if end.beats() > self.total_beats.beats() {
            self.total_beats = end;
        }
        self.notes.push(note);
    }

    /// Notes belonging to a given voice, in onset order.
    pub fn voice(&self, role: VoiceRole) -> Vec<ScoreNote> {
        let mut v: Vec<ScoreNote> = self
            .notes
            .iter()
            .copied()
            .filter(|n| n.role == role)
            .collect();
        v.sort_by(|a, b| a.onset.beats().partial_cmp(&b.onset.beats()).unwrap());
        v
    }

    /// All notes sorted by onset (the event stream muse renders).
    pub fn events(&self) -> Vec<ScoreNote> {
        let mut v = self.notes.clone();
        v.sort_by(|a, b| a.onset.beats().partial_cmp(&b.onset.beats()).unwrap());
        v
    }

    /// Interpret the legacy `meter` field as a quarter-note denominator.
    ///
    /// New code that needs compound or additive meter should carry an explicit
    /// [`TimeSignature`] alongside the score until the score schema migrates.
    pub fn time_signature(&self) -> TimeSignature {
        TimeSignature::quarter_note_meter(self.meter)
    }

    /// Total duration in seconds at this score's tempo.
    pub fn seconds(&self) -> f64 {
        self.total_beats.seconds(self.tempo_bpm as f64)
    }

    /// True if within each voice, notes do not overlap (monophonic melody /
    /// bass). Harmony may overlap (chords), so it is excluded.
    pub fn melody_is_monophonic(&self) -> bool {
        let mel = self.voice(VoiceRole::Melody);
        mel.windows(2).all(|w| {
            let end0 = (w[0].onset + w[0].duration).beats();
            end0 <= w[1].onset.beats() + 1e-9
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn c4() -> Pitch {
        Pitch::new(PitchClass::C, 4)
    }

    #[test]
    fn push_tracks_total_length() {
        let mut s = Score::new(Key::major(PitchClass::C), 120.0, 4);
        s.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: c4(),
            onset: Duration::zero(),
            duration: Duration::half(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        s.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: c4(),
            onset: Duration::half(),
            duration: Duration::whole(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        // Total = 2 (onset of last) + 4 (its duration) = 6 beats.
        assert_eq!(s.total_beats, Duration::new(6, 1));
    }

    #[test]
    fn voice_filters_and_sorts() {
        let mut s = Score::new(Key::major(PitchClass::C), 120.0, 4);
        s.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: c4(),
            onset: Duration::quarter(),
            duration: Duration::quarter(),
            velocity: 0.5,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        s.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: c4(),
            onset: Duration::zero(),
            duration: Duration::quarter(),
            velocity: 0.5,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        assert_eq!(s.voice(VoiceRole::Melody).len(), 1);
        assert_eq!(s.voice(VoiceRole::Bass).len(), 1);
        assert_eq!(s.events().len(), 2);
        assert_eq!(s.events()[0].onset, Duration::zero()); // sorted by onset
    }

    #[test]
    fn seconds_from_beats_and_tempo() {
        let mut s = Score::new(Key::major(PitchClass::C), 120.0, 4);
        s.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: c4(),
            onset: Duration::zero(),
            duration: Duration::whole(), // 4 beats
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        // 4 beats at 120 BPM = 2 seconds.
        assert!((s.seconds() - 2.0).abs() < 1e-9);
    }
}
