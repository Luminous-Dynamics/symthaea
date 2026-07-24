// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! A culturally qualified modal-arc grammar.
//!
//! This is **not** a claim to implement Khayal or any complete Hindustani
//! tradition. It supplies the reusable architecture Muse lacked: pitch
//! hierarchy over an invariant drone, sparse unmetered-feeling exposition,
//! entry into pulse, and cumulative rhythmic intensification. Catalog and UI
//! policy keep the result labelled “Hindustani-informed” pending expert review.

use crate::composer::MusicalIntent;
use crate::harmony::Key;
use crate::motif::Motif;
use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};
use crate::spec::{Attitude, CompositionSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalArcProfile {
    ExpansiveOpening,
    Balanced,
    EarlyPulse,
    ExtendedIntensification,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModalArcPlan {
    pub profile: ModalArcProfile,
    pub stage_repetitions: [usize; 3],
    pub alap_end: Duration,
    pub jor_end: Duration,
    pub jhala_end: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModalArcRealization {
    pub score: Score,
    pub plan: ModalArcPlan,
}

/// Realize a generic exposition→pulse→intensification modal arc.
pub fn realize_modal_arc(intent: &MusicalIntent, spec: &CompositionSpec) -> ModalArcRealization {
    let key = spec
        .mode
        .and_then(|mode| Key::modal(intent.tonic, mode))
        .unwrap_or_else(|| Key::minor(intent.tonic));
    let tempo = spec.tempo(intent.arousal)
        * match spec.attitude {
            Some(Attitude::Joy) => 1.08,
            Some(Attitude::Grief) => 0.85,
            _ => 1.0,
        };
    let meter = spec.meter;
    let motif = spec.motif(intent.arousal, intent.seed);
    let motif = if motif.notes.iter().any(|note| note.degree.is_some()) {
        motif
    } else {
        Motif::from_degrees(&[
            (1, Duration::half()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
        ])
    };
    let scale = key.scale();
    let mut score = Score::new(key, tempo, meter);
    let bars = intent.bars.max(1);
    let profile = match intent.seed % 4 {
        0 => ModalArcProfile::ExpansiveOpening,
        1 => ModalArcProfile::Balanced,
        2 => ModalArcProfile::EarlyPulse,
        _ => ModalArcProfile::ExtendedIntensification,
    };
    let stage_repetitions = match profile {
        ModalArcProfile::ExpansiveOpening => [bars + 2, bars.saturating_sub(2).max(1), bars * 2],
        ModalArcProfile::Balanced => [bars, bars, bars * 2],
        ModalArcProfile::EarlyPulse => [(bars * 2 / 3).max(1), bars + bars / 3, bars * 2],
        ModalArcProfile::ExtendedIntensification => [bars, (bars * 3 / 4).max(1), bars * 3],
    };
    let mut onset = Duration::zero();

    // Exposition: augmented tones separated by breath. There is no chord
    // progression and no universal Western cadence correction.
    for repetition in 0..stage_repetitions[0] {
        for (index, event) in motif.notes.iter().enumerate() {
            let duration = event.duration.scale(2, 1);
            if let Some(degree) = event.degree {
                score.push(ScoreNote {
                    pitch: scale.degree_pitch(degree, 4),
                    onset,
                    duration,
                    velocity: 0.34 + repetition as f32 * 0.025,
                    role: VoiceRole::Melody,
                    emphasis: if repetition == 0 && index == 0 {
                        Emphasis::PhraseStart
                    } else {
                        Emphasis::Normal
                    },
                    section_intensity: 0.45,
                });
            }
            onset = onset + duration + Duration::eighth();
        }
    }
    let alap_end = onset;

    // Pulse enters: the same pitch identity is now carried by a regular gait.
    for repetition in 0..stage_repetitions[1] {
        for (index, event) in motif.notes.iter().enumerate() {
            let duration = event.duration;
            if let Some(degree) = event.degree {
                score.push(ScoreNote {
                    pitch: scale.degree_pitch(degree, 4),
                    onset,
                    duration,
                    velocity: 0.52 + repetition as f32 * 0.025,
                    role: VoiceRole::Melody,
                    emphasis: if index == 0 {
                        Emphasis::PhraseStart
                    } else {
                        Emphasis::Normal
                    },
                    section_intensity: 0.7,
                });
            }
            onset = onset + duration;
        }
    }
    let jor_end = onset;

    // Intensification: diminution and tonic/fifth punctuation make the
    // process audibly denser while the modal center never changes.
    for repetition in 0..stage_repetitions[2] {
        for event in &motif.notes {
            let duration = event.duration.scale(1, 2);
            if let Some(degree) = event.degree {
                score.push(ScoreNote {
                    pitch: scale.degree_pitch(degree + if repetition % 2 == 0 { 0 } else { 7 }, 4),
                    onset,
                    duration,
                    velocity: (0.64 + repetition as f32 * 0.018).min(0.88),
                    role: VoiceRole::Melody,
                    emphasis: Emphasis::Normal,
                    section_intensity: 0.95,
                });
            }
            onset = onset + duration;
            let punctuation = duration.scale(1, 2);
            score.push(ScoreNote {
                pitch: scale.degree_pitch(if repetition % 2 == 0 { 1 } else { 5 }, 5),
                onset,
                duration: punctuation,
                velocity: 0.58,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 0.95,
            });
            onset = onset + punctuation;
        }
    }
    score.push(ScoreNote {
        pitch: scale.degree_pitch(1, 4),
        onset,
        duration: Duration::new(meter as i64, 1),
        velocity: 0.66,
        role: VoiceRole::Melody,
        emphasis: Emphasis::Normal,
        section_intensity: 0.8,
    });
    onset = onset + Duration::new(meter as i64, 1);

    // One continuous tonic/fifth field spans all stages. These are a drone,
    // not a chord progression: their pitch classes and onset never change.
    for (role, degree, octave, velocity) in [
        (VoiceRole::Bass, 1, 2, 0.56),
        (VoiceRole::Harmony, 1, 3, 0.34),
        (VoiceRole::Harmony, 5, 3, 0.3),
        (VoiceRole::Harmony, 8, 3, 0.26),
    ] {
        score.push(ScoreNote {
            pitch: scale.degree_pitch(degree, octave),
            onset: Duration::zero(),
            duration: onset,
            velocity,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 0.6,
        });
    }

    ModalArcRealization {
        score,
        plan: ModalArcPlan {
            profile,
            stage_repetitions,
            alap_end,
            jor_end,
            jhala_end: onset,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;
    use crate::style::Style;

    fn intent() -> MusicalIntent {
        MusicalIntent {
            tonic: PitchClass::D,
            bars: 2,
            seed: 7,
            ..MusicalIntent::default()
        }
    }

    #[test]
    fn arc_has_ordered_stages_and_a_continuous_drone() {
        let realized = realize_modal_arc(&intent(), &Style::HindustaniInspired.spec());
        assert!(realized.plan.alap_end.beats() < realized.plan.jor_end.beats());
        assert!(realized.plan.jor_end.beats() < realized.plan.jhala_end.beats());
        for note in realized.score.voice(VoiceRole::Harmony) {
            assert_eq!(note.onset, Duration::zero());
            assert_eq!(note.duration, realized.plan.jhala_end);
        }
        assert_eq!(realized.score.voice(VoiceRole::Bass).len(), 1);
    }

    #[test]
    fn density_increases_without_a_western_cadential_pass() {
        let realized = realize_modal_arc(&intent(), &Style::HindustaniInspired.spec());
        let melody = realized.score.voice(VoiceRole::Melody);
        let alap_density = melody
            .iter()
            .filter(|note| note.onset.beats() < realized.plan.alap_end.beats())
            .count() as f64
            / realized.plan.alap_end.beats();
        let jhala_notes = melody
            .iter()
            .filter(|note| note.onset.beats() >= realized.plan.jor_end.beats())
            .count() as f64;
        let jhala_density =
            jhala_notes / (realized.plan.jhala_end.beats() - realized.plan.jor_end.beats());
        assert!(jhala_density > alap_density);
        assert!(
            melody
                .iter()
                .all(|note| note.emphasis != Emphasis::Cadential)
        );
        assert!(realized.score.melody_is_monophonic());
    }

    #[test]
    fn seeds_vary_stage_proportions_without_changing_stage_order() {
        let spec = Style::HindustaniInspired.spec();
        let profiles: std::collections::BTreeSet<_> = (0..4)
            .map(|seed| {
                let mut intent = intent();
                intent.bars = 8;
                intent.seed = seed;
                let plan = realize_modal_arc(&intent, &spec).plan;
                assert!(plan.alap_end.beats() < plan.jor_end.beats());
                assert!(plan.jor_end.beats() < plan.jhala_end.beats());
                plan.stage_repetitions
            })
            .collect();
        assert_eq!(profiles.len(), 4);
    }
}
