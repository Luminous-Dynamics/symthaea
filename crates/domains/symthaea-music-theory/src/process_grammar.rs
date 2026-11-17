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


/// Versioned score-side audit schema for additive-process realization.
pub const PROCESS_REALIZATION_AUDIT_SCHEMA_VERSION: u16 = 1;

/// A successful score-side audit of a planned additive process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessRealizationAuditV1 {
    pub schema_version: u16,
    pub cycle_count: usize,
    pub observed_melody_event_count: usize,
    pub maximum_prefix_length: usize,
}

/// A fail-closed mismatch between an additive-process plan and its realized score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessRealizationAuditError {
    EmptyPlan,
    CycleStartCountMismatch {
        prefix_count: usize,
        cycle_start_count: usize,
    },
    FirstCycleDoesNotStartAtZero {
        actual: Duration,
    },
    NonPositivePrefixLength {
        cycle_index: usize,
    },
    TrajectoryContourMismatch {
        cycle_index: usize,
        expected: usize,
        actual: usize,
    },
    NonIncreasingCycleStart {
        cycle_index: usize,
    },
    InvalidCycleWindow {
        cycle_index: usize,
    },
    MelodyEventOutsideTrajectory {
        event_index: usize,
        onset: Duration,
    },
    MelodyEventCrossesCycleBoundary {
        cycle_index: usize,
        event_index: usize,
    },
    CycleStartDoesNotMatchFirstEvent {
        cycle_index: usize,
        expected: Duration,
        actual: Duration,
    },
    CyclePrefixLengthMismatch {
        cycle_index: usize,
        expected: usize,
        actual: usize,
    },
    PrefixContentMismatch {
        cycle_index: usize,
        event_index: usize,
    },
    MelodySequenceHasGapOrOverlap {
        previous_event_index: usize,
        event_index: usize,
    },
    FinalMelodyEndMismatch {
        expected: Duration,
        actual: Duration,
    },
}

/// Independently verify that the score actually realizes the declared
/// additive-process trajectory.
///
/// This verifier does not trust structural-stage labels in GrammarStructuralTrace.
/// It re-derives the canonical prefix contour from the declared trajectory kind
/// and maximum prefix length, partitions the melody by the exact cycle starts,
/// checks the note count and exact pitch/rhythm prefix in every cycle, and
/// rejects gaps, overlaps, boundary-crossing notes or unplanned melody events.
///
/// The audit qualifies symbolic score realization only. It does not qualify
/// timbre, renderer behavior, audible process perception or aesthetic quality.
pub fn verify_additive_process_score_v1(
    plan: &ProcessPlan,
    score: &Score,
) -> Result<ProcessRealizationAuditV1, ProcessRealizationAuditError> {
    if plan.prefix_lengths.is_empty() {
        return Err(ProcessRealizationAuditError::EmptyPlan);
    }
    if plan.prefix_lengths.len() != plan.cycle_starts.len() {
        return Err(ProcessRealizationAuditError::CycleStartCountMismatch {
            prefix_count: plan.prefix_lengths.len(),
            cycle_start_count: plan.cycle_starts.len(),
        });
    }
    if plan.cycle_starts[0] != Duration::zero() {
        return Err(ProcessRealizationAuditError::FirstCycleDoesNotStartAtZero {
            actual: plan.cycle_starts[0],
        });
    }
    for (cycle_index, prefix_length) in plan.prefix_lengths.iter().enumerate() {
        if *prefix_length == 0 {
            return Err(ProcessRealizationAuditError::NonPositivePrefixLength { cycle_index });
        }
    }

    let maximum_prefix_length = *plan.prefix_lengths.iter().max().expect("non-empty plan");
    let expected_contour = independent_prefix_contour(plan.trajectory_kind, maximum_prefix_length);
    for (cycle_index, actual) in plan.prefix_lengths.iter().copied().enumerate() {
        let expected = expected_contour[cycle_index % expected_contour.len()];
        if actual != expected {
            return Err(ProcessRealizationAuditError::TrajectoryContourMismatch {
                cycle_index,
                expected,
                actual,
            });
        }
    }

    for cycle_index in 1..plan.cycle_starts.len() {
        if !process_duration_less_than(
            plan.cycle_starts[cycle_index - 1],
            plan.cycle_starts[cycle_index],
        ) {
            return Err(ProcessRealizationAuditError::NonIncreasingCycleStart { cycle_index });
        }
    }

    let cycle_end = |cycle_index: usize| {
        plan.cycle_starts
            .get(cycle_index + 1)
            .copied()
            .unwrap_or(score.total_beats)
    };
    for cycle_index in 0..plan.cycle_starts.len() {
        if !process_duration_less_than(plan.cycle_starts[cycle_index], cycle_end(cycle_index)) {
            return Err(ProcessRealizationAuditError::InvalidCycleWindow { cycle_index });
        }
    }

    let melody: Vec<&ScoreNote> = score
        .notes
        .iter()
        .filter(|note| note.role == VoiceRole::Melody)
        .collect();
    if melody.is_empty() {
        return Err(ProcessRealizationAuditError::CyclePrefixLengthMismatch {
            cycle_index: 0,
            expected: plan.prefix_lengths[0],
            actual: 0,
        });
    }

    let mut cycle_events: Vec<Vec<(usize, &ScoreNote)>> =
        (0..plan.prefix_lengths.len()).map(|_| Vec::new()).collect();
    for (event_index, note) in melody.iter().enumerate() {
        let owner = (0..plan.cycle_starts.len()).find(|cycle_index| {
            let start = plan.cycle_starts[*cycle_index];
            let end = cycle_end(*cycle_index);
            !process_duration_less_than(note.onset, start)
                && process_duration_less_than(note.onset, end)
        });
        let Some(cycle_index) = owner else {
            return Err(ProcessRealizationAuditError::MelodyEventOutsideTrajectory {
                event_index,
                onset: note.onset,
            });
        };
        let note_end = note.onset + note.duration;
        if process_duration_less_than(cycle_end(cycle_index), note_end) {
            return Err(ProcessRealizationAuditError::MelodyEventCrossesCycleBoundary {
                cycle_index,
                event_index,
            });
        }
        cycle_events[cycle_index].push((event_index, *note));
    }

    for (cycle_index, events) in cycle_events.iter().enumerate() {
        let expected = plan.prefix_lengths[cycle_index];
        if events.len() != expected {
            return Err(ProcessRealizationAuditError::CyclePrefixLengthMismatch {
                cycle_index,
                expected,
                actual: events.len(),
            });
        }
        let actual_start = events[0].1.onset;
        let expected_start = plan.cycle_starts[cycle_index];
        if actual_start != expected_start {
            return Err(ProcessRealizationAuditError::CycleStartDoesNotMatchFirstEvent {
                cycle_index,
                expected: expected_start,
                actual: actual_start,
            });
        }
    }

    for (event_index, pair) in melody.windows(2).enumerate() {
        if pair[0].onset + pair[0].duration != pair[1].onset {
            return Err(ProcessRealizationAuditError::MelodySequenceHasGapOrOverlap {
                previous_event_index: event_index,
                event_index: event_index + 1,
            });
        }
    }

    let template_cycle_index = plan
        .prefix_lengths
        .iter()
        .position(|length| *length == maximum_prefix_length)
        .expect("maximum prefix has an index");
    let template = &cycle_events[template_cycle_index];
    for (cycle_index, events) in cycle_events.iter().enumerate() {
        for event_index in 0..events.len() {
            let observed = events[event_index].1;
            let expected = template[event_index].1;
            if observed.pitch != expected.pitch || observed.duration != expected.duration {
                return Err(ProcessRealizationAuditError::PrefixContentMismatch {
                    cycle_index,
                    event_index,
                });
            }
        }
    }

    let last = melody[melody.len() - 1];
    let actual_end = last.onset + last.duration;
    if actual_end != score.total_beats {
        return Err(ProcessRealizationAuditError::FinalMelodyEndMismatch {
            expected: score.total_beats,
            actual: actual_end,
        });
    }

    Ok(ProcessRealizationAuditV1 {
        schema_version: PROCESS_REALIZATION_AUDIT_SCHEMA_VERSION,
        cycle_count: cycle_events.len(),
        observed_melody_event_count: melody.len(),
        maximum_prefix_length,
    })
}

/// A separately implemented definition of the four declared prefix contours.
/// Keep this independent from the planner's construction so a planner drift
/// cannot automatically make its own score audit pass.
fn independent_prefix_contour(
    trajectory_kind: ProcessTrajectoryKind,
    cell_length: usize,
) -> Vec<usize> {
    match trajectory_kind {
        ProcessTrajectoryKind::Arch => {
            (1..=cell_length).chain((1..cell_length).rev()).collect()
        }
        ProcessTrajectoryKind::Terraced => (1..=cell_length)
            .flat_map(|length| [length, length])
            .chain((1..cell_length).rev().flat_map(|length| [length, length]))
            .collect(),
        ProcessTrajectoryKind::PeakHold => (1..=cell_length)
            .chain(std::iter::repeat_n(cell_length, 3))
            .chain((1..cell_length).rev())
            .collect(),
        ProcessTrajectoryKind::AcceleratingRelease => (1..=cell_length)
            .chain((1..cell_length).rev().step_by(2))
            .chain([1])
            .collect(),
    }
}

fn process_duration_less_than(left: Duration, right: Duration) -> bool {
    (left.num() as i128) * (right.den() as i128)
        < (right.num() as i128) * (left.den() as i128)
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
    #[test]
    fn score_side_audit_accepts_all_four_declared_trajectory_kinds() {
        for seed in 0..4 {
            let intent = MusicalIntent {
                bars: 4,
                seed,
                ..MusicalIntent::default()
            };
            let realization = realize_additive_process(&intent, &Style::Minimalism.spec());
            let report = verify_additive_process_score_v1(&realization.plan, &realization.score)
                .expect("the unmodified score should realize its declared process");
            assert_eq!(report.schema_version, PROCESS_REALIZATION_AUDIT_SCHEMA_VERSION);
            assert_eq!(report.cycle_count, realization.plan.prefix_lengths.len());
            assert_eq!(
                report.maximum_prefix_length,
                *realization.plan.prefix_lengths.iter().max().unwrap()
            );
        }
    }

    #[test]
    fn score_side_audit_rejects_a_plan_with_a_false_cycle_count() {
        let realization = realize_additive_process(
            &MusicalIntent { seed: 1, ..MusicalIntent::default() },
            &Style::Minimalism.spec(),
        );
        let mut altered_plan = realization.plan.clone();
        altered_plan.prefix_lengths[0] += 1;
        assert!(matches!(
            verify_additive_process_score_v1(&altered_plan, &realization.score),
            Err(ProcessRealizationAuditError::TrajectoryContourMismatch { .. })
                | Err(ProcessRealizationAuditError::CyclePrefixLengthMismatch { .. })
        ));
    }

    #[test]
    fn score_side_audit_rejects_mutated_note_content_inside_a_cycle() {
        let mut realization = realize_additive_process(
            &MusicalIntent { seed: 1, ..MusicalIntent::default() },
            &Style::Minimalism.spec(),
        );
        let melody_indices: Vec<usize> = realization
            .score
            .notes
            .iter()
            .enumerate()
            .filter_map(|(index, note)| (note.role == VoiceRole::Melody).then_some(index))
            .collect();
        let changed_note = melody_indices[realization.plan.prefix_lengths[0]];
        realization.score.notes[changed_note].duration =
            realization.score.notes[changed_note].duration + Duration::eighth();
        assert!(verify_additive_process_score_v1(&realization.plan, &realization.score).is_err());
    }

    #[test]
    fn score_side_audit_rejects_a_shifted_cycle_boundary() {
        let realization = realize_additive_process(
            &MusicalIntent { seed: 1, ..MusicalIntent::default() },
            &Style::Minimalism.spec(),
        );
        let mut altered_plan = realization.plan.clone();
        altered_plan.cycle_starts[1] =
            altered_plan.cycle_starts[1] + Duration::eighth();
        assert!(verify_additive_process_score_v1(&altered_plan, &realization.score).is_err());
    }

    #[test]
    fn score_side_audit_rejects_missing_melody_events() {
        let mut realization = realize_additive_process(
            &MusicalIntent { seed: 1, ..MusicalIntent::default() },
            &Style::Minimalism.spec(),
        );
        let remove_index = realization
            .score
            .notes
            .iter()
            .position(|note| note.role == VoiceRole::Melody)
            .unwrap();
        realization.score.notes.remove(remove_index);
        assert!(verify_additive_process_score_v1(&realization.plan, &realization.score).is_err());
    }

}
