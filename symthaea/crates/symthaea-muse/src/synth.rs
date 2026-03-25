// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hybrid additive + FM synthesis with chord voicing and feedback delay reverb.
//!
//! Timbre is modulated by neuromodulator levels:
//! - Dopamine → FM modulation depth (brightness/metallic quality)
//! - Serotonin → softer, rounder tones (reduces FM, boosts fundamental)
//! - Noradrenaline → edgier overtones + faster attack
//! - Consciousness → sustain length + reverb amount

use crate::{MusicalState, Note};

/// ADSR envelope parameters.
struct Adsr {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
}

/// Render a sequence of notes into PCM samples with FM synthesis + reverb.
pub fn render_notes(
    notes: &[Note],
    sample_rate: u32,
    total_samples: usize,
    state: &MusicalState,
) -> Vec<i16> {
    let mut buffer = vec![0.0f32; total_samples];
    let sr = sample_rate as f32;

    let partials = compute_timbre(state);
    let adsr = compute_adsr(state);
    let fm_depth = state.dopamine * 3.0; // FM modulation index (0-3)
    let fm_ratio = 2.0 + state.noradrenaline; // carrier:modulator ratio

    // Chord intervals: active harmonies suggest chord voicing
    let chord_intervals = compute_chord_intervals(state);

    for note in notes {
        // Render the root note + chord tones
        for &interval_ratio in &chord_intervals {
            let freq = note.frequency * interval_ratio;
            let chord_vol = if (interval_ratio - 1.0).abs() < 0.01 {
                1.0 // root is full volume
            } else {
                0.35 // chord tones quieter
            };

            render_single_tone(
                &mut buffer,
                sr,
                note,
                freq,
                chord_vol,
                &partials,
                &adsr,
                fm_depth,
                fm_ratio,
            );
        }
    }

    // Apply feedback delay reverb
    apply_reverb(&mut buffer, sr, state);

    // Convert to i16 with soft clipping
    buffer
        .iter()
        .map(|&s| {
            let clipped = soft_clip(s);
            (clipped * i16::MAX as f32) as i16
        })
        .collect()
}

/// Render a single tone (additive + FM) into the buffer.
fn render_single_tone(
    buffer: &mut [f32],
    sr: f32,
    note: &Note,
    freq: f32,
    volume_scale: f32,
    partials: &[f32; 4],
    adsr: &Adsr,
    fm_depth: f32,
    fm_ratio: f32,
) {
    let start_sample = (note.start_time * sr) as usize;
    let duration_samples = (note.duration * sr) as usize;
    let release_samples = (adsr.release * sr) as usize;
    let total_note_samples = duration_samples + release_samples;

    let modulator_freq = freq * fm_ratio;

    for i in 0..total_note_samples {
        let sample_idx = start_sample + i;
        if sample_idx >= buffer.len() {
            break;
        }

        let t = i as f32 / sr;
        let env = envelope(adsr, t, note.duration);

        // FM synthesis: modulator modulates carrier phase
        let mod_phase = std::f32::consts::TAU * modulator_freq * t;
        let fm_offset = fm_depth * mod_phase.sin();

        // Additive synthesis with FM-modulated carrier
        let mut sample = 0.0f32;
        for (harmonic, &amplitude) in partials.iter().enumerate() {
            let carrier_freq = freq * (harmonic + 1) as f32;
            let phase = std::f32::consts::TAU * carrier_freq * t + fm_offset;
            sample += amplitude * phase.sin();
        }

        buffer[sample_idx] += sample * env * note.velocity * volume_scale * 0.2;
    }
}

/// Compute chord intervals from harmony activations.
///
/// Returns frequency ratios relative to root (1.0 = root always included).
/// Active harmonies add their characteristic intervals.
fn compute_chord_intervals(state: &MusicalState) -> Vec<f32> {
    let mut intervals = vec![1.0]; // root always

    // PanSentientFlourishing → major 3rd (5/4)
    if state.harmony_activations[1] > 0.5 {
        intervals.push(1.25);
    }
    // IntegralWisdom → perfect 5th (3/2)
    if state.harmony_activations[2] > 0.4 {
        intervals.push(1.5);
    }
    // InfinitePlay → minor 7th (9/5)
    if state.harmony_activations[3] > 0.6 {
        intervals.push(1.8);
    }
    // UniversalInterconnectedness → perfect 4th (4/3)
    if state.harmony_activations[4] > 0.5 {
        intervals.push(1.333);
    }

    // Limit to triads (max 3 notes) to avoid muddiness
    intervals.truncate(3);
    intervals
}

/// Apply feedback delay reverb to the buffer.
///
/// Simple comb filter reverb: delayed copy mixed back into buffer.
/// Consciousness level controls reverb amount (more conscious = more spacious).
fn apply_reverb(buffer: &mut [f32], sr: f32, state: &MusicalState) {
    let reverb_amount = 0.1 + state.consciousness_level * 0.3; // 0.1-0.4
    let delay_samples = (0.12 * sr) as usize; // 120ms delay
    let feedback = 0.3 + state.harmony_activations[7] * 0.2; // SacredStillness → more reverb tail

    if delay_samples >= buffer.len() {
        return;
    }

    // Apply comb filter in-place (two taps for richer reverb)
    for tap in [delay_samples, delay_samples * 3 / 2] {
        if tap >= buffer.len() {
            continue;
        }
        for i in tap..buffer.len() {
            buffer[i] += buffer[i - tap] * reverb_amount * feedback;
        }
    }
}

/// Compute harmonic partial amplitudes from neuromodulators.
fn compute_timbre(state: &MusicalState) -> [f32; 4] {
    // Fundamental always present
    let fundamental = 1.0;
    // Dopamine adds brightness (more upper partials)
    let second = 0.3 + state.dopamine * 0.4;
    // Serotonin rounds the tone (fewer overtones)
    let third = 0.15 * (1.0 - state.serotonin * 0.5);
    // NE adds edge
    let fourth = 0.08 + state.noradrenaline * 0.2;

    [fundamental, second, third, fourth]
}

/// Compute ADSR parameters from cognitive state.
fn compute_adsr(state: &MusicalState) -> Adsr {
    Adsr {
        // Fast attack when aroused, slow when calm
        attack: 0.01 + (1.0 - state.arousal) * 0.05,
        decay: 0.05 + state.serotonin * 0.1,
        // Higher consciousness → more sustain
        sustain: 0.4 + state.consciousness_level * 0.4,
        // Longer release for contemplative states
        release: 0.1 + state.harmony_activations[7] * 0.3,
    }
}

/// ADSR envelope value at time t for a note of given duration.
fn envelope(adsr: &Adsr, t: f32, note_duration: f32) -> f32 {
    if t < adsr.attack {
        // Attack: ramp up
        t / adsr.attack
    } else if t < adsr.attack + adsr.decay {
        // Decay: ramp down to sustain
        let decay_progress = (t - adsr.attack) / adsr.decay;
        1.0 - decay_progress * (1.0 - adsr.sustain)
    } else if t < note_duration {
        // Sustain
        adsr.sustain
    } else {
        // Release
        let release_progress = (t - note_duration) / adsr.release;
        adsr.sustain * (1.0 - release_progress).max(0.0)
    }
}

/// Soft clipping to prevent harsh distortion.
fn soft_clip(x: f32) -> f32 {
    if x > 1.0 {
        1.0 - (-x + 1.0).exp() * 0.5
    } else if x < -1.0 {
        -1.0 + (x + 1.0).exp() * 0.5
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_single_note() {
        let notes = vec![Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        }];
        let state = MusicalState::default();
        let samples = render_notes(&notes, 44100, 44100, &state);
        assert_eq!(samples.len(), 44100);
        // Should have non-zero samples
        assert!(samples.iter().any(|&s| s != 0));
    }

    #[test]
    fn render_silence_for_no_notes() {
        let state = MusicalState::default();
        let samples = render_notes(&[], 44100, 44100, &state);
        assert!(samples.iter().all(|&s| s == 0));
    }

    #[test]
    fn soft_clip_bounds() {
        assert!((soft_clip(0.5) - 0.5).abs() < 0.01);
        assert!(soft_clip(2.0) <= 1.0);
        assert!(soft_clip(-2.0) >= -1.0);
    }

    #[test]
    fn envelope_shape() {
        let adsr = Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.7,
            release: 0.1,
        };
        // Attack peak
        let peak = envelope(&adsr, 0.01, 0.5);
        assert!((peak - 1.0).abs() < 0.1);

        // Sustain level
        let sus = envelope(&adsr, 0.2, 0.5);
        assert!((sus - 0.7).abs() < 0.1);

        // After release
        let rel = envelope(&adsr, 0.6, 0.5);
        assert!(rel < 0.7);
    }

    #[test]
    fn timbre_dopamine_effect() {
        let low_da = compute_timbre(&MusicalState {
            dopamine: 0.1,
            ..Default::default()
        });
        let high_da = compute_timbre(&MusicalState {
            dopamine: 0.9,
            ..Default::default()
        });
        // More dopamine → brighter (more upper partials)
        assert!(high_da[1] > low_da[1]);
    }
}
