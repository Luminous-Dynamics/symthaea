// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Music taste benchmark: scores output against Tristan's taste profile.
//!
//! Derived from 5,972 Spotify liked songs and music theory research.
//! Each dimension is scored 0-100, with a composite score.
//! Use this to measure and optimize toward musical quality that
//! matches the target listener's preferences.

use crate::Note;
use std::collections::HashMap;

/// Taste benchmark result.
#[derive(Debug, Clone)]
pub struct TasteScore {
    /// Melodic movement: penalizes repeated notes, rewards steps + variety.
    pub melodic_movement: f32,
    /// Pitch diversity: unique pitches, no single note dominance.
    pub pitch_diversity: f32,
    /// Interval balance: right mix of steps, thirds, leaps.
    pub interval_balance: f32,
    /// Direction balance: roughly equal ascending/descending.
    pub direction_balance: f32,
    /// Dynamic range: velocity variety.
    pub dynamic_range: f32,
    /// Pitch register: notes in singable/playable range.
    pub pitch_register: f32,
    /// Composite score (weighted average).
    pub composite: f32,
    /// Per-dimension details for diagnosis.
    pub details: Vec<(String, f32, String)>,
}

/// Target profile (derived from Spotify analysis + music theory).
pub struct TasteProfile {
    pub target_repeated_pct: f32,  // 10%
    pub target_steps_pct: f32,     // 60%
    pub target_thirds_pct: f32,    // 15%
    pub target_leaps_pct: f32,     // 15%
    pub target_ascending_pct: f32, // 50%
    pub min_unique_pitches: usize, // 20
    pub max_top_note_pct: f32,     // 12%
    pub target_pitch_range: f32,   // 14 semitones
    pub target_mean_midi: f32,     // 66 (F#4)
    pub min_velocity_range: f32,   // 60
}

impl Default for TasteProfile {
    fn default() -> Self {
        // Tristan's profile: prog/jazz/atmospheric listener
        Self::tristan()
    }
}

impl TasteProfile {
    /// Tristan's taste (5,972 Spotify songs: prog/jazz/atmospheric).
    pub fn tristan() -> Self {
        Self {
            target_repeated_pct: 10.0,
            target_steps_pct: 55.0,
            target_thirds_pct: 15.0,
            target_leaps_pct: 15.0,
            target_ascending_pct: 50.0,
            min_unique_pitches: 20,
            max_top_note_pct: 12.0,
            target_pitch_range: 14.0,
            target_mean_midi: 66.0,
            min_velocity_range: 60.0,
        }
    }

    /// Universal baseline — musical universals from research.
    /// Jakubowski (2017), Huron (2006): preferences shared across cultures.
    pub fn universal() -> Self {
        Self {
            target_repeated_pct: 15.0, // some repetition is universal
            target_steps_pct: 50.0,    // stepwise motion dominates everywhere
            target_thirds_pct: 15.0,
            target_leaps_pct: 10.0,     // fewer leaps universally
            target_ascending_pct: 50.0, // balanced
            min_unique_pitches: 15,     // less demanding
            max_top_note_pct: 18.0,     // more tolerance
            target_pitch_range: 12.0,   // octave — most singable range
            target_mean_midi: 64.0,     // E4 — center of vocal range
            min_velocity_range: 40.0,   // less demanding
        }
    }

    /// Classical listener (Bach, Beethoven, Chopin).
    pub fn classical() -> Self {
        Self {
            target_repeated_pct: 8.0, // classical avoids exact repetition
            target_steps_pct: 60.0,   // very stepwise (voice-leading rules)
            target_thirds_pct: 18.0,  // arpeggiated thirds
            target_leaps_pct: 10.0,   // controlled leaps
            target_ascending_pct: 50.0,
            min_unique_pitches: 25,   // chromatic variety
            max_top_note_pct: 10.0,   // well-distributed
            target_pitch_range: 18.0, // wider range (keyboard)
            target_mean_midi: 64.0,   // E4
            min_velocity_range: 80.0, // extreme dynamics (ppp to fff)
        }
    }

    /// Metal/rock listener.
    pub fn metal() -> Self {
        Self {
            target_repeated_pct: 12.0, // riff-based repetition
            target_steps_pct: 40.0,    // more leaps, power chords
            target_thirds_pct: 12.0,
            target_leaps_pct: 25.0,     // wide intervals
            target_ascending_pct: 48.0, // slightly descending bias
            min_unique_pitches: 18,
            max_top_note_pct: 15.0,   // riff root repetition OK
            target_pitch_range: 20.0, // wide range (shredding)
            target_mean_midi: 52.0,   // E3 — guitar range
            min_velocity_range: 50.0, // compressed dynamics
        }
    }

    /// Jazz listener.
    pub fn jazz() -> Self {
        Self {
            target_repeated_pct: 8.0,
            target_steps_pct: 48.0,  // chromatic approach tones
            target_thirds_pct: 22.0, // chord tone arpeggiation
            target_leaps_pct: 15.0,  // interval variety
            target_ascending_pct: 50.0,
            min_unique_pitches: 28,   // chromatic richness
            max_top_note_pct: 8.0,    // very distributed
            target_pitch_range: 16.0, // wide but controlled
            target_mean_midi: 67.0,   // G4 (trumpet/sax range)
            min_velocity_range: 70.0, // expressive dynamics
        }
    }

    /// Electronic/ambient listener.
    pub fn ambient() -> Self {
        Self {
            target_repeated_pct: 25.0, // repetition is a feature
            target_steps_pct: 55.0,    // smooth movement
            target_thirds_pct: 10.0,
            target_leaps_pct: 5.0, // minimal leaps
            target_ascending_pct: 50.0,
            min_unique_pitches: 10,   // limited palette OK
            max_top_note_pct: 25.0,   // drone notes OK
            target_pitch_range: 10.0, // narrow, focused
            target_mean_midi: 60.0,   // C4
            min_velocity_range: 30.0, // subtle dynamics
        }
    }

    /// Pop listener.
    pub fn pop() -> Self {
        Self {
            target_repeated_pct: 18.0, // hook repetition
            target_steps_pct: 55.0,    // singable
            target_thirds_pct: 15.0,
            target_leaps_pct: 8.0,      // mostly stepwise
            target_ascending_pct: 52.0, // slight upward bias (hooks rise)
            min_unique_pitches: 12,     // simple palette
            max_top_note_pct: 20.0,     // tonic emphasis OK
            target_pitch_range: 10.0,   // singable octave
            target_mean_midi: 65.0,     // F4 (vocal sweet spot)
            min_velocity_range: 35.0,   // compressed
        }
    }

    /// Derive a profile from genre preset.
    pub fn for_genre(genre: &str) -> Self {
        match genre.to_lowercase().as_str() {
            "classical" | "bach" | "beethoven" => Self::classical(),
            "metal" | "prog-metal" | "rock" => Self::metal(),
            "jazz" => Self::jazz(),
            "ambient" | "electronic" => Self::ambient(),
            "pop" => Self::pop(),
            _ => Self::universal(),
        }
    }
}

/// Score a set of notes against the taste profile.
pub fn score(notes: &[Note], profile: &TasteProfile) -> TasteScore {
    if notes.len() < 10 {
        return TasteScore {
            melodic_movement: 0.0,
            pitch_diversity: 0.0,
            interval_balance: 0.0,
            direction_balance: 0.0,
            dynamic_range: 0.0,
            pitch_register: 0.0,
            composite: 0.0,
            details: vec![("Too few notes".into(), 0.0, "Need ≥10".into())],
        };
    }

    let midi_pitches: Vec<u8> = notes
        .iter()
        .map(|n| freq_to_midi(n.frequency))
        .filter(|&p| p >= 36 && p <= 96) // filter degenerate pitches
        .collect();
    let velocities: Vec<f32> = notes.iter().map(|n| n.velocity).collect();
    let intervals: Vec<i32> = midi_pitches
        .windows(2)
        .map(|w| w[1] as i32 - w[0] as i32)
        .collect();

    let n = intervals.len() as f32;
    let mut details = Vec::new();

    // 1. Melodic movement (penalize repeated notes)
    let repeated_pct = intervals.iter().filter(|&&iv| iv == 0).count() as f32 / n * 100.0;
    let movement_score = score_proximity(repeated_pct, profile.target_repeated_pct, 30.0);
    details.push((
        "Repeated notes".into(),
        repeated_pct,
        format!(
            "{:.0}% (target {:.0}%)",
            repeated_pct, profile.target_repeated_pct
        ),
    ));

    // 2. Pitch diversity
    let unique = midi_pitches
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    let unique_score = (unique as f32 / profile.min_unique_pitches as f32 * 100.0).min(100.0);

    let mut pitch_counts: HashMap<u8, usize> = HashMap::new();
    for &p in &midi_pitches {
        *pitch_counts.entry(p).or_insert(0) += 1;
    }
    let top_pct =
        pitch_counts.values().max().copied().unwrap_or(0) as f32 / notes.len() as f32 * 100.0;
    let dominance_score = if top_pct <= profile.max_top_note_pct {
        100.0
    } else {
        (1.0 - (top_pct - profile.max_top_note_pct) / 50.0).max(0.0) * 100.0
    };

    let diversity = (unique_score * 0.5 + dominance_score * 0.5).min(100.0);
    details.push((
        "Unique pitches".into(),
        unique as f32,
        format!("{} (target ≥{})", unique, profile.min_unique_pitches),
    ));
    details.push((
        "Top note %".into(),
        top_pct,
        format!("{:.0}% (target ≤{:.0}%)", top_pct, profile.max_top_note_pct),
    ));

    // 3. Interval balance
    let steps_pct = intervals
        .iter()
        .filter(|&&iv| iv != 0 && iv.abs() <= 2)
        .count() as f32
        / n
        * 100.0;
    let thirds_pct = intervals
        .iter()
        .filter(|&&iv| iv.abs() >= 3 && iv.abs() <= 5)
        .count() as f32
        / n
        * 100.0;
    let leaps_pct = intervals.iter().filter(|&&iv| iv.abs() > 7).count() as f32 / n * 100.0;

    let steps_score = score_proximity(steps_pct, profile.target_steps_pct, 25.0);
    let thirds_score = score_proximity(thirds_pct, profile.target_thirds_pct, 15.0);
    let leaps_score = score_proximity(leaps_pct, profile.target_leaps_pct, 15.0);
    let interval_bal = (steps_score + thirds_score + leaps_score) / 3.0;

    details.push((
        "Steps %".into(),
        steps_pct,
        format!(
            "{:.0}% (target {:.0}%)",
            steps_pct, profile.target_steps_pct
        ),
    ));
    details.push((
        "Thirds %".into(),
        thirds_pct,
        format!(
            "{:.0}% (target {:.0}%)",
            thirds_pct, profile.target_thirds_pct
        ),
    ));
    details.push((
        "Leaps %".into(),
        leaps_pct,
        format!(
            "{:.0}% (target {:.0}%)",
            leaps_pct, profile.target_leaps_pct
        ),
    ));

    // 4. Direction balance
    let ascending_pct = intervals.iter().filter(|&&iv| iv > 0).count() as f32 / n * 100.0;
    let direction = score_proximity(ascending_pct, profile.target_ascending_pct, 20.0);
    details.push((
        "Ascending %".into(),
        ascending_pct,
        format!(
            "{:.0}% (target {:.0}%)",
            ascending_pct, profile.target_ascending_pct
        ),
    ));

    // 5. Dynamic range
    let vel_min = velocities.iter().cloned().fold(f32::MAX, f32::min);
    let vel_max = velocities.iter().cloned().fold(f32::MIN, f32::max);
    let vel_range = (vel_max - vel_min) * 127.0; // scale to MIDI range
    let dynamics = (vel_range / profile.min_velocity_range * 100.0).min(100.0);
    details.push((
        "Velocity range".into(),
        vel_range,
        format!(
            "{:.0} (target ≥{:.0})",
            vel_range, profile.min_velocity_range
        ),
    ));

    // 6. Pitch register
    let mean_midi = midi_pitches.iter().map(|&p| p as f32).sum::<f32>() / midi_pitches.len() as f32;
    let pitch_range =
        *midi_pitches.iter().max().unwrap() as f32 - *midi_pitches.iter().min().unwrap() as f32;
    let register_score = score_proximity(mean_midi, profile.target_mean_midi, 12.0);
    let range_score = score_proximity(pitch_range, profile.target_pitch_range, 10.0);
    let register = (register_score * 0.5 + range_score * 0.5).min(100.0);
    details.push((
        "Mean MIDI".into(),
        mean_midi,
        format!("{:.0} (target {:.0})", mean_midi, profile.target_mean_midi),
    ));
    details.push((
        "Pitch range".into(),
        pitch_range,
        format!(
            "{:.0} semi (target {:.0})",
            pitch_range, profile.target_pitch_range
        ),
    ));

    // Composite: weighted by importance
    let composite = movement_score * 0.25    // repeated notes are the #1 problem
        + diversity * 0.20                    // pitch variety
        + interval_bal * 0.20                 // step/leap balance
        + direction * 0.10                    // ascending/descending
        + dynamics * 0.10                     // velocity range
        + register * 0.15; // pitch register

    TasteScore {
        melodic_movement: movement_score,
        pitch_diversity: diversity,
        interval_balance: interval_bal,
        direction_balance: direction,
        dynamic_range: dynamics,
        pitch_register: register,
        composite,
        details,
    }
}

/// Score how close a value is to a target (0-100, higher = closer).
fn score_proximity(actual: f32, target: f32, tolerance: f32) -> f32 {
    let distance = (actual - target).abs();
    (1.0 - distance / tolerance).max(0.0) * 100.0
}

fn freq_to_midi(freq: f32) -> u8 {
    ((12.0 * (freq / 440.0).log2() + 69.0).round() as i32).clamp(0, 127) as u8
}

/// Print a formatted benchmark report.
pub fn print_report(score: &TasteScore) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Music Taste Benchmark                                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Melodic Movement:  {:5.1}/100  {}║",
        score.melodic_movement,
        rating(score.melodic_movement)
    );
    println!(
        "║  Pitch Diversity:   {:5.1}/100  {}║",
        score.pitch_diversity,
        rating(score.pitch_diversity)
    );
    println!(
        "║  Interval Balance:  {:5.1}/100  {}║",
        score.interval_balance,
        rating(score.interval_balance)
    );
    println!(
        "║  Direction Balance: {:5.1}/100  {}║",
        score.direction_balance,
        rating(score.direction_balance)
    );
    println!(
        "║  Dynamic Range:     {:5.1}/100  {}║",
        score.dynamic_range,
        rating(score.dynamic_range)
    );
    println!(
        "║  Pitch Register:    {:5.1}/100  {}║",
        score.pitch_register,
        rating(score.pitch_register)
    );
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  COMPOSITE:         {:5.1}/100  {}║",
        score.composite,
        rating(score.composite)
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\nDetails:");
    for (name, _value, desc) in &score.details {
        println!("  {name:20}: {desc}");
    }
}

fn rating(score: f32) -> &'static str {
    if score >= 80.0 {
        "Excellent                    "
    } else if score >= 60.0 {
        "Good                         "
    } else if score >= 40.0 {
        "Fair                         "
    } else if score >= 20.0 {
        "Poor                         "
    } else {
        "Critical                     "
    }
}

/// Score against multiple taste profiles simultaneously.
/// Returns individual scores plus a consensus (average) score.
pub fn score_multi(notes: &[Note], profiles: &[(&str, TasteProfile)]) -> Vec<(String, TasteScore)> {
    profiles
        .iter()
        .map(|(name, profile)| (name.to_string(), score(notes, profile)))
        .collect()
}

/// Score from a MIDI file directly.
pub fn score_midi_file(path: &std::path::Path) -> Result<TasteScore, String> {
    let data = std::fs::read(path).map_err(|e| format!("{e}"))?;
    let mut notes = Vec::new();

    for i in 0..data.len().saturating_sub(2) {
        if 0x90 <= data[i] && data[i] <= 0x9F && data[i + 2] > 0 {
            let pitch = data[i + 1];
            let vel = data[i + 2];
            notes.push(Note {
                frequency: 440.0 * 2.0f32.powf((pitch as f32 - 69.0) / 12.0),
                start_time: notes.len() as f32 * 0.2, // approximate
                duration: 0.3,
                velocity: vel as f32 / 127.0,
            });
        }
    }

    Ok(score(&notes, &TasteProfile::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scale() -> Vec<Note> {
        // C major scale up and down — should score well on movement
        let freqs = [
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25, 493.88, 440.00, 392.00,
            349.23, 329.63, 293.66, 261.63,
        ];
        freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| Note {
                frequency: f,
                start_time: i as f32 * 0.3,
                duration: 0.3,
                velocity: 0.3 + (i as f32 / 15.0) * 0.5,
            })
            .collect()
    }

    fn make_monotone() -> Vec<Note> {
        // Same note 20 times — should score poorly
        (0..20)
            .map(|i| Note {
                frequency: 440.0,
                start_time: i as f32 * 0.3,
                duration: 0.3,
                velocity: 0.7,
            })
            .collect()
    }

    #[test]
    fn scale_scores_better_than_monotone() {
        let profile = TasteProfile::default();
        let scale = score(&make_scale(), &profile);
        let mono = score(&make_monotone(), &profile);
        assert!(
            scale.composite > mono.composite,
            "scale ({:.1}) should beat monotone ({:.1})",
            scale.composite,
            mono.composite
        );
    }

    #[test]
    fn monotone_has_zero_movement() {
        let profile = TasteProfile::default();
        let s = score(&make_monotone(), &profile);
        assert!(
            s.melodic_movement < 5.0,
            "monotone should have near-zero movement: {:.1}",
            s.melodic_movement
        );
    }

    #[test]
    fn scale_has_good_movement() {
        let profile = TasteProfile::default();
        let s = score(&make_scale(), &profile);
        assert!(
            s.melodic_movement > 50.0,
            "scale should have good movement: {:.1}",
            s.melodic_movement
        );
    }

    #[test]
    fn proximity_exact_is_100() {
        assert_eq!(score_proximity(50.0, 50.0, 10.0), 100.0);
    }

    #[test]
    fn proximity_out_of_range_is_0() {
        assert_eq!(score_proximity(100.0, 50.0, 10.0), 0.0);
    }
}
