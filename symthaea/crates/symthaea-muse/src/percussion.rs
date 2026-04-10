// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Percussion synthesis: kick, snare, hihat from noise + resonance.
//!
//! Consciousness-coupled: arousal drives pattern density, dopamine drives
//! accent strength, Sacred Stillness gates percussion on/off.

/// Percussion hit type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrumType {
    /// Kick drum: sine sweep + low-pass noise burst.
    Kick,
    /// Snare: noise burst + resonant body.
    Snare,
    /// Hi-hat: high-pass noise with fast decay.
    HiHat,
}

/// A single percussion hit.
#[derive(Debug, Clone, Copy)]
pub struct DrumHit {
    pub drum: DrumType,
    pub time: f32,
    pub velocity: f32,
}

/// Render a drum hit into a mono sample buffer.
pub fn render_drum(hit: &DrumHit, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    match hit.drum {
        DrumType::Kick => render_kick(sr, hit.velocity),
        DrumType::Snare => render_snare(sr, hit.velocity),
        DrumType::HiHat => render_hihat(sr, hit.velocity),
    }
}

fn render_kick(sr: f32, velocity: f32) -> Vec<f32> {
    let duration = 0.15; // 150ms
    let n = (duration * sr) as usize;
    let mut buf = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        let env = (-t * 20.0).exp(); // fast decay
                                     // Frequency sweep: 150Hz → 50Hz
        let freq = 50.0 + 100.0 * (-t * 30.0).exp();
        let phase = std::f32::consts::TAU * freq * t;
        let sample = phase.sin() * env * velocity * 0.8;
        buf.push(sample);
    }
    buf
}

fn render_snare(sr: f32, velocity: f32) -> Vec<f32> {
    let duration = 0.12;
    let n = (duration * sr) as usize;
    let mut buf = Vec::with_capacity(n);
    let mut noise_state = 0.0f32;
    let mut rng = 42u32; // simple LCG
    for i in 0..n {
        let t = i as f32 / sr;
        let env = (-t * 25.0).exp();
        // Body: 200Hz resonance
        let body = (std::f32::consts::TAU * 200.0 * t).sin() * env * 0.4;
        // Noise: filtered white noise
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let white = (rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        let alpha = 0.15; // LP coefficient
        noise_state += alpha * (white - noise_state);
        let noise = noise_state * env * 0.6;
        buf.push((body + noise) * velocity);
    }
    buf
}

fn render_hihat(sr: f32, velocity: f32) -> Vec<f32> {
    let duration = 0.05; // 50ms
    let n = (duration * sr) as usize;
    let mut buf = Vec::with_capacity(n);
    let mut rng = 99u32;
    let mut hp_state = 0.0f32;
    for i in 0..n {
        let t = i as f32 / sr;
        let env = (-t * 60.0).exp(); // very fast decay
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let white = (rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        // High-pass: subtract LP
        let alpha = 0.05;
        hp_state += alpha * (white - hp_state);
        let hp = white - hp_state;
        buf.push(hp * env * velocity * 0.4);
    }
    buf
}

/// Generate a basic drum pattern based on consciousness state.
///
/// Returns drum hits for one bar (duration in seconds at given BPM).
pub fn generate_pattern(bpm: f32, consciousness: f32, arousal: f32) -> Vec<DrumHit> {
    let beat_dur = 60.0 / bpm;
    let mut hits = Vec::new();

    // Kick on beats 1 and 3
    hits.push(DrumHit {
        drum: DrumType::Kick,
        time: 0.0,
        velocity: 0.8,
    });
    hits.push(DrumHit {
        drum: DrumType::Kick,
        time: beat_dur * 2.0,
        velocity: 0.7,
    });

    // Snare on beats 2 and 4
    if consciousness > 0.3 {
        hits.push(DrumHit {
            drum: DrumType::Snare,
            time: beat_dur,
            velocity: 0.6,
        });
        hits.push(DrumHit {
            drum: DrumType::Snare,
            time: beat_dur * 3.0,
            velocity: 0.65,
        });
    }

    // Hi-hats: density from arousal
    if consciousness > 0.5 {
        let subdivisions = if arousal > 0.7 {
            8
        } else if arousal > 0.4 {
            4
        } else {
            2
        };
        let step = beat_dur * 4.0 / subdivisions as f32;
        for i in 0..subdivisions {
            let vel = if i % 2 == 0 { 0.4 } else { 0.25 };
            hits.push(DrumHit {
                drum: DrumType::HiHat,
                time: step * i as f32,
                velocity: vel,
            });
        }
    }

    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kick_renders() {
        let hit = DrumHit {
            drum: DrumType::Kick,
            time: 0.0,
            velocity: 0.8,
        };
        let buf = render_drum(&hit, 44100);
        assert!(!buf.is_empty());
        assert!(
            buf.iter().any(|&s| s.abs() > 0.01),
            "kick should have signal"
        );
        // Last quarter should be quieter than first quarter (decay)
        let first_rms: f32 = (buf[..buf.len() / 4].iter().map(|s| s * s).sum::<f32>()
            / (buf.len() / 4) as f32)
            .sqrt();
        let last_rms: f32 = (buf[3 * buf.len() / 4..].iter().map(|s| s * s).sum::<f32>()
            / (buf.len() / 4) as f32)
            .sqrt();
        assert!(
            last_rms < first_rms,
            "kick should decay: first={first_rms:.4}, last={last_rms:.4}"
        );
    }

    #[test]
    fn snare_has_noise() {
        let hit = DrumHit {
            drum: DrumType::Snare,
            time: 0.0,
            velocity: 0.7,
        };
        let buf = render_drum(&hit, 44100);
        // Snare should have variance (noise component)
        let mean: f32 = buf.iter().sum::<f32>() / buf.len() as f32;
        let variance: f32 = buf.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / buf.len() as f32;
        assert!(variance > 0.001, "snare should have noise variance");
    }

    #[test]
    fn hihat_is_short() {
        let hit = DrumHit {
            drum: DrumType::HiHat,
            time: 0.0,
            velocity: 0.5,
        };
        let buf = render_drum(&hit, 44100);
        assert!(
            buf.len() < 3000,
            "hihat should be short: {} samples",
            buf.len()
        );
    }

    #[test]
    fn pattern_density_scales_with_arousal() {
        let calm = generate_pattern(120.0, 0.8, 0.2);
        let excited = generate_pattern(120.0, 0.8, 0.9);
        assert!(
            excited.len() > calm.len(),
            "high arousal should produce denser pattern"
        );
    }

    #[test]
    fn low_consciousness_sparse_pattern() {
        let low = generate_pattern(120.0, 0.2, 0.5);
        let high = generate_pattern(120.0, 0.8, 0.5);
        assert!(
            low.len() < high.len(),
            "low consciousness should produce sparse pattern"
        );
    }
}
