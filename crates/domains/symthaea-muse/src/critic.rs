// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Music self-critic: evaluates compositional quality across multiple dimensions.
//!
//! Parallel to the visual critic in symthaea-atelier. Scores melodic interest,
//! rhythmic regularity, harmonic alignment, voice balance, and form coherence.
//! Weights shift with neuromodulator state (moral-aesthetic binding).
//!
//! **Honesty note**: [`evaluate_composition`] is a hand-authored heuristic
//! scorer over statistical properties of the note sequence (pitch variety,
//! inter-onset timing regularity, a section pitch-centroid contrast proxy
//! for form, a rise-then-fall arc detector) — it does not listen to audio
//! or use any learned/perceptual judge. [`music_auto_improve`] closes a
//! REAL loop (generate → score → mutate state → regenerate), but because
//! the loop optimizes against this same fixed heuristic, "improvement"
//! means "scores higher on these specific proxies," not verified perceptual
//! quality. Treat trajectories as evidence the loop is doing SOMETHING
//! directional, not as proof the music got more beautiful.

use crate::{Composition, MusicalState, Note, form};

/// Onsets closer together than this are one rhythmic event, not several.
///
/// 30 ms matches `creative_bench::extract_melody`'s onset-grouping tolerance, and
/// sits comfortably below the ~100 ms at which listeners begin to hear two attacks
/// as separate rather than as one chord.
const SIMULTANEOUS_ONSET_EPSILON: f32 = 0.03;

/// Music critic verdict.
#[derive(Debug, Clone)]
pub struct MusicVerdict {
    /// Melodic interest: pitch variety and contour quality.
    pub melodic_interest: f32,
    /// Rhythmic regularity: `1 − CV(inter-onset intervals)`, so **1.0 means
    /// perfectly metronomic** — higher is not better.
    ///
    /// Was pinned at exactly 0.0 until 2026-07-31 because it differenced the note
    /// list in *voice-major* order; see the comment at the computation site. It
    /// measures rhythm now, which means the downstream threshold matters for the
    /// first time.
    ///
    /// Still a *different* computation from `creative_bench::onset_evenness`
    /// despite the shared idea: that one filters section-boundary gaps
    /// (IOI > 3× median) and this one does not, so the two disagree on music with
    /// pauses. Real repertoire measures ~0.60 on the filtered version
    /// (`examples/rhythm_gate_calibration.rs`); **this unfiltered one has never
    /// been calibrated, so the 0.4 threshold in `update_state_from_verdict` below
    /// remains a guess.** Calibrate it the same way before trusting it.
    pub rhythmic_regularity: f32,
    /// Harmonic alignment: harmony activation mean.
    pub harmonic_alignment: f32,
    /// Voice balance: how evenly distributed are note velocities.
    pub voice_balance: f32,
    /// Form coherence: do adjacent sections actually contrast.
    pub form_coherence: f32,
    /// Melodic contour: does the melody have arc (rise-fall pattern).
    pub melodic_contour: f32,
    /// Weighted composite.
    pub composite: f32,
}

/// Evaluate a composition's musical quality.
pub fn evaluate_composition(comp: &Composition, state: &MusicalState) -> MusicVerdict {
    let notes = &comp.notes;

    // Melodic interest: unique pitch ratio
    let melodic_interest = if notes.len() >= 2 {
        let mut pitches: Vec<i32> = notes.iter().map(|n| (n.frequency * 10.0) as i32).collect();
        pitches.sort();
        pitches.dedup();
        (pitches.len() as f32 / notes.len() as f32).clamp(0.0, 1.0)
    } else {
        0.3
    };

    // Rhythmic regularity: low CV of inter-onset intervals.
    //
    // FIXED 2026-07-31. This used to difference `comp.notes` in LIST order, which
    // is voice-major: the melody is emitted first, then the accompaniment restarts
    // the clock at t=0, and chord tones share an onset. So the difference at each
    // voice boundary was NEGATIVE and each chord contributed ZEROes, and the
    // `.max(0.001)` collapsed all of them to one tiny constant. The CV exploded and
    // `1 − CV` clamped to 0.0 for *every* composition either generation path
    // produces — the metric was reading note-list ordering, not rhythm.
    //
    // That was not confined to the score: `update_state_from_verdict` decrements
    // arousal whenever this reads below 0.4, so with the value pinned at 0.0 that
    // branch fired unconditionally on every composition.
    //
    // Two things are needed, and sorting alone is not enough. A chord is ONE
    // rhythmic event, so simultaneous onsets are collapsed before differencing;
    // otherwise a densely voiced passage reads as maximally irregular however
    // steady its pulse.
    let rhythmic_regularity = {
        let mut onsets: Vec<f32> = notes.iter().map(|n| n.start_time).collect();
        onsets.sort_by(|a, b| a.total_cmp(b));
        onsets.dedup_by(|a, b| (*a - *b).abs() < SIMULTANEOUS_ONSET_EPSILON);

        if onsets.len() >= 3 {
            let intervals: Vec<f32> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
            let mean = intervals.iter().sum::<f32>() / intervals.len() as f32;
            if mean > 0.001 {
                let variance = intervals.iter().map(|&i| (i - mean).powi(2)).sum::<f32>()
                    / intervals.len() as f32;
                let cv = variance.sqrt() / mean;
                (1.0 - cv).clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.3
        }
    };

    // Harmonic alignment
    let harmonic_alignment = state.harmony_activations.iter().sum::<f32>() / 8.0;

    // Voice balance: velocity standard deviation (low = balanced)
    let voice_balance = if notes.len() >= 2 {
        let mean_vel: f32 = notes.iter().map(|n| n.velocity).sum::<f32>() / notes.len() as f32;
        let vel_var: f32 = notes
            .iter()
            .map(|n| (n.velocity - mean_vel).powi(2))
            .sum::<f32>()
            / notes.len() as f32;
        (1.0 - vel_var.sqrt() * 2.0).clamp(0.0, 1.0)
    } else {
        0.5
    };

    // Form coherence: check if the composition's section creates actual contrast
    let form_coherence = evaluate_form_coherence(comp, state);

    // Melodic contour: check for rise-fall patterns (arc)
    let melodic_contour = evaluate_contour(notes);

    // Weighted composite (stress/care binding would be applied by caller)
    let composite = (0.20 * melodic_interest
        + 0.15 * rhythmic_regularity
        + 0.20 * harmonic_alignment
        + 0.10 * voice_balance
        + 0.15 * form_coherence
        + 0.20 * melodic_contour)
        .clamp(0.0, 1.0);

    MusicVerdict {
        melodic_interest,
        rhythmic_regularity,
        harmonic_alignment,
        voice_balance,
        form_coherence,
        melodic_contour,
        composite,
    }
}

/// Evaluate form coherence: do sections contrast appropriately.
fn evaluate_form_coherence(comp: &Composition, state: &MusicalState) -> f32 {
    let song_form = form::plan_form(state, comp.duration_secs);
    if song_form.sections.len() < 2 {
        return 0.5; // Single section = neutral
    }

    // Check that notes in different sections have different pitch centroids
    let mut section_centroids = Vec::new();
    for section in &song_form.sections {
        let sec_notes: Vec<&Note> = comp
            .notes
            .iter()
            .filter(|n| {
                n.start_time >= section.start_time
                    && n.start_time < section.start_time + section.duration
            })
            .collect();
        if !sec_notes.is_empty() {
            let centroid =
                sec_notes.iter().map(|n| n.frequency).sum::<f32>() / sec_notes.len() as f32;
            section_centroids.push(centroid);
        }
    }

    if section_centroids.len() < 2 {
        return 0.5;
    }

    // Measure variance between section centroids (higher = more contrast)
    let mean = section_centroids.iter().sum::<f32>() / section_centroids.len() as f32;
    let variance = section_centroids
        .iter()
        .map(|c| (c - mean).powi(2))
        .sum::<f32>()
        / section_centroids.len() as f32;
    // Normalize: typical centroid range ~100-600 Hz, so variance up to ~60000
    let normalized = (variance.sqrt() / 200.0).clamp(0.0, 1.0);
    // Berlyne: moderate contrast is best (not identical, not random)
    1.0 - (normalized - 0.4).abs() * 2.0
}

/// Evaluate melodic contour: presence of rise-fall arc.
fn evaluate_contour(notes: &[Note]) -> f32 {
    if notes.len() < 4 {
        return 0.5;
    }

    // Split melody into halves, check if first half rises and second falls
    let half = notes.len() / 2;
    let first_half = &notes[..half];
    let second_half = &notes[half..];

    let first_direction: f32 = first_half
        .windows(2)
        .map(|w| {
            if w[1].frequency > w[0].frequency {
                1.0
            } else {
                -1.0
            }
        })
        .sum::<f32>()
        / (first_half.len() - 1).max(1) as f32;

    let second_direction: f32 = second_half
        .windows(2)
        .map(|w| {
            if w[1].frequency > w[0].frequency {
                1.0
            } else {
                -1.0
            }
        })
        .sum::<f32>()
        / (second_half.len() - 1).max(1) as f32;

    // Ideal arc: first half ascending (positive), second half descending (negative)
    let arc_quality = (first_direction - second_direction) / 2.0;
    arc_quality.clamp(0.0, 1.0)
}

// ── Music Auto-Improve ──────────────────────────────────────────────────────

/// Score improvement below this triggers Sacred Stillness for music.
pub const MUSIC_STILLNESS_THRESHOLD: f32 = 0.01;

/// Result of a music practice session.
#[derive(Debug, Clone)]
pub struct MusicPracticeResult {
    /// Rounds completed.
    pub rounds: usize,
    /// Composite score per round.
    pub trajectory: Vec<f32>,
    /// Whether stillness was reached.
    pub reached_stillness: bool,
    /// Best score achieved.
    pub best_score: f32,
}

/// Autonomous music practice: compose, critique, evolve state, repeat.
///
/// Each round: compose → evaluate → apply musical wisdom → repeat.
/// Stops at stillness (delta < threshold) or max_rounds.
pub fn music_auto_improve(
    config: &crate::MuseConfig,
    state: &mut MusicalState,
    max_rounds: usize,
    seed_base: u64,
) -> MusicPracticeResult {
    run_auto_improve(state, max_rounds, |round, state| {
        crate::compose(config, state, seed_base + round as u64)
    })
}

/// Like [`music_auto_improve`], but each round composes AND realizes
/// through the real pipeline (`symthaea-music-theory`'s structural
/// composer + muse's per-voice-instrument synthesis + mastering) instead
/// of the legacy neural-melody `crate::compose` path. This is what makes
/// the self-listening loop actually reflect on the SAME music a listener
/// hears today, not a disconnected older pipeline.
///
/// Only `seed` varies per round on the composed side (mirroring
/// `music_auto_improve`'s `seed_base + round`); `state` is what evolves via
/// [`apply_music_wisdom`], shaping the additive-voice harmonic doubling,
/// reverb depth, and dynamics each round.
#[cfg(feature = "theory")]
pub fn music_auto_improve_theory(
    intent: &symthaea_music_theory::MusicalIntent,
    style: symthaea_music_theory::Style,
    state: &mut MusicalState,
    sample_rate: u32,
    max_rounds: usize,
    seed_base: u64,
) -> MusicPracticeResult {
    run_auto_improve(state, max_rounds, |round, state| {
        let mut round_intent = *intent;
        round_intent.seed = seed_base + round as u64;
        crate::theory_realize::compose_and_realize_styled(&round_intent, style, state, sample_rate)
    })
}

/// Shared round loop: generate → evaluate → check stillness → apply wisdom.
/// `generate` gets the round index and the CURRENT (possibly wisdom-evolved)
/// state, and produces this round's [`Composition`].
fn run_auto_improve(
    state: &mut MusicalState,
    max_rounds: usize,
    mut generate: impl FnMut(usize, &MusicalState) -> Composition,
) -> MusicPracticeResult {
    let mut trajectory = Vec::with_capacity(max_rounds);
    let mut best_score = 0.0f32;
    let mut prev_score = 0.0f32;

    for round in 0..max_rounds {
        let comp = generate(round, state);
        let verdict = evaluate_composition(&comp, state);
        trajectory.push(verdict.composite);

        if verdict.composite > best_score {
            best_score = verdict.composite;
        }

        // Stillness check
        if round > 2 && (verdict.composite - prev_score).abs() < MUSIC_STILLNESS_THRESHOLD {
            return MusicPracticeResult {
                rounds: round + 1,
                trajectory,
                reached_stillness: true,
                best_score,
            };
        }
        prev_score = verdict.composite;

        // Apply musical wisdom
        apply_music_wisdom(state, &verdict);
    }

    MusicPracticeResult {
        rounds: max_rounds,
        trajectory,
        reached_stillness: false,
        best_score,
    }
}

/// Apply music verdict as wisdom to evolve the musical state.
fn apply_music_wisdom(state: &mut MusicalState, verdict: &MusicVerdict) {
    let lr = 0.05;

    // High melodic interest + good contour → boost consciousness (reward creative phrasing)
    if verdict.melodic_contour > 0.5 && verdict.melodic_interest > 0.5 {
        state.consciousness_level = (state.consciousness_level + lr * 0.5).clamp(0.0, 1.0);
    }

    // Low rhythmic regularity → reduce arousal (calm down, find the groove)
    if verdict.rhythmic_regularity < 0.4 {
        state.arousal = (state.arousal - lr).clamp(0.0, 1.0);
    }

    // High harmonic alignment → boost serotonin (satisfaction)
    if verdict.harmonic_alignment > 0.5 {
        state.serotonin = (state.serotonin + lr * 0.3).clamp(0.0, 1.0);
    }

    // Low voice balance → reduce consciousness slightly (simplify)
    if verdict.voice_balance < 0.3 {
        state.consciousness_level = (state.consciousness_level - lr * 0.3).clamp(0.0, 1.0);
    }

    // Good form coherence → boost progression harmony
    if verdict.form_coherence > 0.5 {
        state.harmony_activations[6] = (state.harmony_activations[6] + lr * 0.3).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::SectionType;

    fn test_comp() -> Composition {
        crate::compose(
            &crate::MuseConfig {
                duration_secs: 4.0,
                max_notes: 16,
                ..Default::default()
            },
            &crate::MusicalState::default(),
            42,
        )
    }

    #[test]
    fn verdict_bounded() {
        let comp = test_comp();
        let state = crate::MusicalState::default();
        let v = evaluate_composition(&comp, &state);
        assert!(v.composite >= 0.0 && v.composite <= 1.0);
        assert!(v.melodic_interest >= 0.0 && v.melodic_interest <= 1.0);
        assert!(v.rhythmic_regularity >= 0.0 && v.rhythmic_regularity <= 1.0);
        assert!(v.form_coherence >= 0.0 && v.form_coherence <= 1.0);
    }

    #[test]
    fn contour_detects_arc() {
        // Rising then falling melody = good arc
        let arc_notes = vec![
            Note {
                frequency: 200.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 300.0,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 400.0,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 500.0,
                start_time: 1.5,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 400.0,
                start_time: 2.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 300.0,
                start_time: 2.5,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 200.0,
                start_time: 3.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 100.0,
                start_time: 3.5,
                duration: 0.5,
                velocity: 0.8,
            },
        ];
        let contour = evaluate_contour(&arc_notes);
        assert!(
            contour > 0.5,
            "rising-falling arc should score high: {contour}"
        );
    }

    #[test]
    fn form_coherence_with_real_composition() {
        let state = crate::MusicalState {
            harmony_activations: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.1],
            ..Default::default()
        };
        let comp = crate::compose(
            &crate::MuseConfig {
                duration_secs: 8.0,
                max_notes: 32,
                ..Default::default()
            },
            &state,
            42,
        );
        let coherence = evaluate_form_coherence(&comp, &state);
        assert!(
            coherence >= 0.0 && coherence <= 1.0,
            "coherence={coherence}"
        );
    }

    #[test]
    fn empty_composition_safe() {
        let comp = Composition {
            audio: crate::AudioData::I16(vec![]),
            sample_rate: 44100,
            notes: vec![],
            duration_secs: 1.0,
            section: SectionType::Developmental,
        };
        let state = crate::MusicalState::default();
        let v = evaluate_composition(&comp, &state);
        assert!(v.composite >= 0.0 && v.composite <= 1.0);
    }

    #[test]
    fn high_variety_high_melodic_interest() {
        let diverse_notes = vec![
            Note {
                frequency: 100.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 200.0,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 300.0,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 400.0,
                start_time: 1.5,
                duration: 0.5,
                velocity: 0.8,
            },
        ];
        let repetitive_notes = vec![
            Note {
                frequency: 261.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 261.0,
                start_time: 0.5,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 261.0,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.8,
            },
            Note {
                frequency: 261.0,
                start_time: 1.5,
                duration: 0.5,
                velocity: 0.8,
            },
        ];

        let diverse_comp = Composition {
            audio: crate::AudioData::I16(vec![]),
            sample_rate: 44100,
            notes: diverse_notes,
            duration_secs: 2.0,
            section: SectionType::Developmental,
        };
        let repetitive_comp = Composition {
            audio: crate::AudioData::I16(vec![]),
            sample_rate: 44100,
            notes: repetitive_notes,
            duration_secs: 2.0,
            section: SectionType::Developmental,
        };

        let state = crate::MusicalState::default();
        let v_diverse = evaluate_composition(&diverse_comp, &state);
        let v_repetitive = evaluate_composition(&repetitive_comp, &state);
        assert!(
            v_diverse.melodic_interest > v_repetitive.melodic_interest,
            "diverse {} should beat repetitive {}",
            v_diverse.melodic_interest,
            v_repetitive.melodic_interest
        );
    }

    #[test]
    fn music_auto_improve_produces_trajectory() {
        let config = crate::MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };
        let mut state = crate::MusicalState::default();
        let result = music_auto_improve(&config, &mut state, 10, 42);
        assert!(result.rounds > 0);
        assert!(!result.trajectory.is_empty());
        assert!(result.best_score >= 0.0 && result.best_score <= 1.0);
    }

    #[test]
    fn music_auto_improve_reaches_stillness() {
        let config = crate::MuseConfig {
            duration_secs: 2.0,
            max_notes: 8,
            ..Default::default()
        };
        let mut state = crate::MusicalState::default();
        let result = music_auto_improve(&config, &mut state, 15, 42);
        assert!(
            result.reached_stillness || result.rounds == 15,
            "should terminate"
        );
    }

    #[test]
    fn music_wisdom_modifies_state() {
        let mut state = crate::MusicalState::default();
        let original_serotonin = state.serotonin;
        let verdict = MusicVerdict {
            melodic_interest: 0.7,
            rhythmic_regularity: 0.8,
            harmonic_alignment: 0.8,
            voice_balance: 0.6,
            form_coherence: 0.7,
            melodic_contour: 0.8,
            composite: 0.7,
        };
        apply_music_wisdom(&mut state, &verdict);
        // High harmonic alignment should boost serotonin
        assert!(
            state.serotonin > original_serotonin,
            "serotonin should increase: {} → {}",
            original_serotonin,
            state.serotonin
        );
    }

    #[cfg(feature = "theory")]
    #[test]
    fn music_auto_improve_theory_produces_trajectory() {
        use symthaea_music_theory::{MusicalIntent, Style};
        let intent = MusicalIntent {
            bars: 2,
            ..Default::default()
        };
        let mut state = crate::MusicalState::default();
        let result = music_auto_improve_theory(&intent, Style::Classical, &mut state, 44100, 4, 42);
        assert!(result.rounds > 0);
        assert!(!result.trajectory.is_empty());
        assert!(result.best_score >= 0.0 && result.best_score <= 1.0);
    }

    #[cfg(feature = "theory")]
    #[test]
    fn music_auto_improve_theory_rounds_use_different_seeds() {
        // Two consecutive rounds must not compose the identical piece --
        // otherwise "practice" would just be re-scoring the same music.
        use symthaea_music_theory::{MusicalIntent, Style};
        let intent = MusicalIntent {
            bars: 4,
            ..Default::default()
        };
        let a = symthaea_music_theory::compose(&MusicalIntent {
            seed: 100,
            ..intent
        });
        let b = symthaea_music_theory::compose(&MusicalIntent {
            seed: 101,
            ..intent
        });
        assert_ne!(a, b, "different seeds must produce different scores");
    }
}

#[cfg(test)]
mod onset_ordering_tests {
    use super::*;
    use crate::{AudioData, Composition, MusicalState, Note, structure::SectionType};

    /// A composition as the generator actually builds one: VOICE-MAJOR. The melody
    /// is emitted first, then the accompaniment restarts at t=0, and chord tones
    /// share an onset. `comp.notes` is therefore not in time order.
    fn voice_major_composition() -> Composition {
        let mut notes = Vec::new();
        // Melody: 8 notes, evenly spaced over 4s. Perfectly regular.
        for i in 0..8 {
            notes.push(Note {
                frequency: 440.0 + i as f32 * 20.0,
                start_time: i as f32 * 0.5,
                duration: 0.45,
                velocity: 0.7,
            });
        }
        // Accompaniment: two 3-note chords, restarting the clock at t=0.
        for (chord, t) in [(0, 0.0f32), (1, 2.0f32)] {
            for v in 0..3 {
                notes.push(Note {
                    frequency: 220.0 + chord as f32 * 10.0 + v as f32 * 30.0,
                    start_time: t,
                    duration: 1.9,
                    velocity: 0.5,
                });
            }
        }
        Composition {
            audio: AudioData::F32(Vec::new()),
            sample_rate: 48_000,
            notes,
            duration_secs: 4.0,
            section: SectionType::Developmental,
        }
    }

    /// The melody in the fixture is PERFECTLY regular — 8 notes at a fixed 0.5s
    /// spacing. Any honest onset-regularity measure must score it high.
    ///
    /// Before the 2026-07-31 fix this returned 0.0000, because
    /// `evaluate_composition` differenced `comp.notes` in list order. At the
    /// melody→accompaniment boundary that difference is NEGATIVE (3.5 → 0.0) and
    /// the three chord tones are SIMULTANEOUS (difference 0.0); `.max(0.001)`
    /// collapsed all of them to the same tiny constant, so the CV exploded and
    /// `1 − CV` clamped to zero.
    ///
    /// The consequence was not confined to the score. `update_state_from_verdict`
    /// decrements arousal whenever this reads below 0.4 — so with the metric pinned
    /// at 0.0 that branch fired on *every* composition, unconditionally, which is
    /// not a rhythm response at all.
    #[test]
    fn rhythmic_regularity_survives_voice_major_note_order() {
        let v = evaluate_composition(&voice_major_composition(), &MusicalState::default());
        assert!(
            v.rhythmic_regularity > 0.5,
            "rhythmic_regularity {:.4} on a fixture whose melody is perfectly \
             regular — the note list is voice-major, so this is measuring list \
             ORDER rather than rhythm.",
            v.rhythmic_regularity,
        );
    }

    /// Sorting alone is not enough: a chord is ONE rhythmic event, not three
    /// zero-length intervals. If simultaneous onsets are kept, a densely voiced
    /// passage reads as maximally irregular no matter how steady its pulse.
    #[test]
    fn simultaneous_chord_tones_are_one_rhythmic_event() {
        let mut c = voice_major_composition();
        // Thicken every chord to 6 voices. The PULSE is unchanged, so the metric
        // must not move much.
        let extra: Vec<Note> = c
            .notes
            .iter()
            .filter(|n| n.frequency < 400.0)
            .map(|n| Note {
                frequency: n.frequency + 7.0,
                ..*n
            })
            .collect();
        let thin = evaluate_composition(&c, &MusicalState::default()).rhythmic_regularity;
        c.notes.extend(extra);
        let thick = evaluate_composition(&c, &MusicalState::default()).rhythmic_regularity;
        assert!(
            (thin - thick).abs() < 0.15,
            "doubling chord voicing moved rhythmic_regularity {thin:.4} -> {thick:.4}; \
             the pulse did not change, so simultaneous onsets are being counted as \
             separate rhythmic events.",
        );
    }
}
