// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Process/additive composition in which the audible rule is the form.
//!
//! Unlike the legacy additive late pass, this engine never builds a period,
//! contrast section, cadence, or coda and then paints a process onto it. The
//! growing/shrinking cell determines duration, density, and large-scale shape.

use crate::composer::MusicalIntent;
use crate::harmony::Key;
use crate::motif::{Motif, MotifNote};
use crate::rhythm::Duration;
use crate::score::{Emphasis, PartId, Score, ScoreNote, VoiceRole};
use crate::spec::{Attitude, CompositionSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessTrajectoryKind {
    Arch,
    Terraced,
    PeakHold,
    AcceleratingRelease,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProcessPlan {
    pub trajectory_kind: ProcessTrajectoryKind,
    /// Number of cell events sounded in each successive cycle.
    pub prefix_lengths: Vec<usize>,
    pub cycle_starts: Vec<Duration>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessRealization {
    pub score: Score,
    pub plan: ProcessPlan,
}

pub fn realize_additive_process(
    intent: &MusicalIntent,
    spec: &CompositionSpec,
) -> ProcessRealization {
    let key = spec
        .mode
        .and_then(|mode| Key::modal(intent.tonic, mode))
        .unwrap_or_else(|| {
            if intent.valence >= 0.0 {
                Key::major(intent.tonic)
            } else {
                Key::minor(intent.tonic)
            }
        });
    let tempo = spec.tempo(intent.arousal)
        * match spec.attitude {
            Some(Attitude::Joy) => 1.08,
            Some(Attitude::Grief) => 0.85,
            _ => 1.0,
        };
    let mut cell = spec.motif(intent.arousal, intent.seed);
    cell.notes.retain(|event| event.degree.is_some());
    if cell.notes.is_empty() {
        cell = Motif::new(vec![
            MotifNote::new(1, Duration::eighth()),
            MotifNote::new(3, Duration::eighth()),
            MotifNote::new(5, Duration::quarter()),
        ]);
    }

    let n = cell.notes.len();
    let trajectory_kind = match intent.seed % 4 {
        1 => ProcessTrajectoryKind::Arch,
        0 => ProcessTrajectoryKind::Terraced,
        2 => ProcessTrajectoryKind::PeakHold,
        _ => ProcessTrajectoryKind::AcceleratingRelease,
    };
    let contour: Vec<usize> = match trajectory_kind {
        ProcessTrajectoryKind::Arch => (1..=n).chain((1..n).rev()).collect(),
        ProcessTrajectoryKind::Terraced => (1..=n)
            .flat_map(|length| [length, length])
            .chain((1..n).rev().flat_map(|length| [length, length]))
            .collect(),
        ProcessTrajectoryKind::PeakHold => (1..=n)
            .chain(std::iter::repeat_n(n, 3))
            .chain((1..n).rev())
            .collect(),
        ProcessTrajectoryKind::AcceleratingRelease => {
            (1..=n).chain((1..n).rev().step_by(2)).chain([1]).collect()
        }
    };
    let cycle_count = (intent.bars.max(1) * 2).max(contour.len());
    let prefix_lengths: Vec<usize> = (0..cycle_count)
        .map(|index| contour[index % contour.len()])
        .collect();

    let scale = key.scale();
    let mut score = Score::new(key, tempo, spec.meter);
    let mut onset = Duration::zero();
    let mut cycle_starts = Vec::with_capacity(prefix_lengths.len());
    for (cycle, &prefix_len) in prefix_lengths.iter().enumerate() {
        cycle_starts.push(onset);
        let intensity = 0.42 + 0.44 * prefix_len as f32 / n as f32;
        for (index, event) in cell.notes.iter().take(prefix_len).enumerate() {
            let duration = event.duration;
            score.push(ScoreNote {
                part: PartId::UNASSIGNED,
                pitch: scale.degree_pitch(event.degree.unwrap(), 5),
                onset,
                duration,
                velocity: 0.46 + 0.24 * prefix_len as f32 / n as f32,
                role: VoiceRole::Melody,
                emphasis: if cycle == 0 && index == 0 {
                    Emphasis::PhraseStart
                } else {
                    Emphasis::Normal
                },
                section_intensity: intensity,
            });
            onset = onset + duration;
        }
    }

    // The accompaniment is a clock and spectral field, not harmonic
    // argument. Tonic/fifth/octave repeat without progression or cadence.
    let total = onset;
    let mut pulse = Duration::zero();
    let mut pulse_index = 0usize;
    while pulse.beats() < total.beats() - 1e-9 {
        let duration = if (pulse + Duration::eighth()).beats() > total.beats() {
            total.saturating_sub(pulse)
        } else {
            Duration::eighth()
        };
        score.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: scale.degree_pitch([1, 5, 8, 5][pulse_index % 4], 4),
            onset: pulse,
            duration,
            velocity: 0.34,
            role: VoiceRole::Harmony,
            emphasis: Emphasis::Normal,
            section_intensity: 0.62,
        });
        pulse = pulse + duration;
        pulse_index += 1;
    }
    let bar = Duration::new(spec.meter as i64, 1);
    let mut bass_onset = Duration::zero();
    while bass_onset.beats() < total.beats() - 1e-9 {
        let duration = if (bass_onset + bar).beats() > total.beats() {
            total.saturating_sub(bass_onset)
        } else {
            bar
        };
        score.push(ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: scale.degree_pitch(1, 2),
            onset: bass_onset,
            duration,
            velocity: 0.43,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: 0.58,
        });
        bass_onset = bass_onset + duration;
    }

    ProcessRealization {
        score,
        plan: ProcessPlan {
            trajectory_kind,
            prefix_lengths,
            cycle_starts,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Style;

    #[test]
    fn the_process_grows_and_shrinks_without_cadences() {
        let intent = MusicalIntent {
            bars: 4,
            seed: 9,
            ..MusicalIntent::default()
        };
        let realized = realize_additive_process(&intent, &Style::Minimalism.spec());
        let lengths = &realized.plan.prefix_lengths;
        let peak = lengths
            .iter()
            .position(|&length| length == *lengths.iter().max().unwrap())
            .unwrap();
        assert!(
            lengths[..=peak]
                .windows(2)
                .all(|window| window[0] <= window[1])
        );
        assert!(
            lengths[peak..]
                .windows(2)
                .all(|window| window[0] >= window[1])
        );
        assert!(
            realized
                .score
                .notes
                .iter()
                .all(|note| note.emphasis != Emphasis::Cadential)
        );
        assert!(realized.score.melody_is_monophonic());
    }

    #[test]
    fn bass_is_a_static_tonic_clock() {
        let realized =
            realize_additive_process(&MusicalIntent::default(), &Style::Minimalism.spec());
        let bass = realized.score.voice(VoiceRole::Bass);
        assert!(!bass.is_empty());
        assert!(
            bass.iter()
                .all(|note| note.pitch.pitch_class() == realized.score.key.tonic)
        );
    }

    #[test]
    fn seeds_select_multiple_process_trajectories() {
        let spec = Style::Minimalism.spec();
        let kinds: std::collections::BTreeSet<_> = (0..8)
            .map(|seed| {
                realize_additive_process(
                    &MusicalIntent {
                        seed,
                        ..MusicalIntent::default()
                    },
                    &spec,
                )
                .plan
                .trajectory_kind as u8
            })
            .collect();
        assert_eq!(kinds.len(), 4);
    }
}
