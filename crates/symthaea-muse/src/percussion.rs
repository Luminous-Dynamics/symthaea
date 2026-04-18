// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Percussion synthesis: kick, snare, hihat from noise + resonance.
//!
//! Each drum voice uses velocity-shaped envelopes: loud hits are punchy with
//! fast attacks and longer sustain; soft hits are gentle with slower attacks
//! and faster decay. Emotional state modulates character — noradrenaline
//! tightens envelopes, dopamine brightens noise, valence shifts timbre.
//!
//! References:
//! - Puckette, M. (2007). *The Theory and Technique of Electronic Music*
//! - Smith, J. O. (2010). *Physical Audio Signal Processing*

/// Percussion hit type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrumType {
    /// Kick drum: sine sweep + sub-bass punch + noise transient.
    Kick,
    /// Snare: noise burst + body resonance + wire buzz.
    Snare,
    /// Hi-hat: metallic noise, high-pass filtered.
    HiHat,
}

/// A single percussion hit.
#[derive(Debug, Clone, Copy)]
pub struct DrumHit {
    pub drum: DrumType,
    pub time: f32,
    pub velocity: f32,
}

/// Emotional modulation for percussion rendering.
///
/// Passed from the streaming engine so drums respond to consciousness state.
#[derive(Debug, Clone, Copy)]
pub struct DrumColor {
    /// Noradrenaline [0,1]: tightens envelopes, adds urgency.
    pub tightness: f32,
    /// Dopamine [0,1]: brightens noise, adds shimmer.
    pub brightness: f32,
    /// Valence [-1,1]: negative = darker body, positive = open ring.
    pub warmth: f32,
}

impl Default for DrumColor {
    fn default() -> Self {
        Self {
            tightness: 0.3,
            brightness: 0.5,
            warmth: 0.0,
        }
    }
}

/// Render a drum hit into a mono sample buffer.
pub fn render_drum(hit: &DrumHit, sample_rate: u32) -> Vec<f32> {
    render_drum_colored(hit, sample_rate, &DrumColor::default())
}

/// Render a drum hit with emotional coloring.
pub fn render_drum_colored(hit: &DrumHit, sample_rate: u32, color: &DrumColor) -> Vec<f32> {
    let sr = sample_rate as f32;
    match hit.drum {
        DrumType::Kick => render_kick(sr, hit.velocity, color),
        DrumType::Snare => render_snare(sr, hit.velocity, color),
        DrumType::HiHat => render_hihat(sr, hit.velocity, color),
    }
}

// ─── Kick Drum ───────────────────────────────────────────────────────────────

fn render_kick(sr: f32, velocity: f32, color: &DrumColor) -> Vec<f32> {
    let vel = velocity.clamp(0.0, 1.0);

    // Velocity shapes duration: louder hits ring longer
    let duration = 0.10 + vel * 0.12; // 100-220ms (was fixed 150ms)
    let n = (duration * sr) as usize;
    let mut buf = Vec::with_capacity(n);

    // Tightness from noradrenaline speeds up decay
    let decay_rate = 15.0 + color.tightness * 20.0; // 15-35 (was fixed 20)

    // Velocity shapes attack: loud hits have a sharper click
    let click_samples = ((0.003 + (1.0 - vel) * 0.005) * sr) as usize; // 3-8ms click

    // Frequency sweep: louder hits start higher (more punch)
    let freq_start = 40.0 + vel * 140.0; // 40-180Hz (was fixed 150Hz)
    let freq_end = 38.0 + color.warmth.max(0.0) * 12.0; // 38-50Hz (warmth opens)
    let freq_decay = 25.0 + color.tightness * 15.0; // faster sweep when tight

    let mut phase = 0.0f32;

    for i in 0..n {
        let t = i as f32 / sr;

        // Two-stage envelope: attack click + body decay.
        // FIX: removed overshoot (1.0 + 0.3*vel) that pushed envelope above 1.0
        // causing distortion when summed with other voices.
        let env = if i < click_samples {
            let attack_t = i as f32 / click_samples as f32;
            1.0 - (-attack_t * 8.0).exp()
        } else {
            (-t * decay_rate).exp()
        };

        // Pitch sweep with velocity-dependent contour
        let freq = freq_end + (freq_start - freq_end) * (-t * freq_decay).exp();
        phase += std::f32::consts::TAU * freq / sr;
        let body = phase.sin();

        // Sub-bass harmonic (adds weight)
        let sub = (phase * 0.5).sin() * 0.3;

        // Transient click: short noise burst at attack
        let click = if i < click_samples {
            let click_env = 1.0 - (i as f32 / click_samples as f32);
            let noise = lcg_noise(i as u32 + 7777);
            noise * click_env * click_env * vel * 0.4
        } else {
            0.0
        };

        let sample = (body + sub + click) * env * vel * 0.7;
        buf.push(sample);
    }
    buf
}

// ─── Snare Drum ──────────────────────────────────────────────────────────────

fn render_snare(sr: f32, velocity: f32, color: &DrumColor) -> Vec<f32> {
    let vel = velocity.clamp(0.0, 1.0);

    // Louder snares sustain longer (wire rattle)
    let duration = 0.08 + vel * 0.10; // 80-180ms (was fixed 120ms)
    let n = (duration * sr) as usize;
    let mut buf = Vec::with_capacity(n);

    // Envelope shaping: loud snares emphasize body, soft ones are mostly noise
    let body_mix = 0.2 + vel * 0.35; // body: 20-55% (loud = more body)
    let noise_mix = 0.8 - vel * 0.15; // noise: 65-80% (always present)

    let decay_body = 20.0 + color.tightness * 15.0; // body decays faster when tight
    let decay_noise = 15.0 + color.tightness * 10.0;

    // Body resonance: velocity shifts fundamental (loud = lower, more authoritative)
    let body_freq = 220.0 - vel * 40.0 + color.warmth * (-15.0); // 180-220Hz

    // Noise filter: brightness from dopamine controls LP cutoff
    let lp_alpha = 0.10 + color.brightness * 0.20; // 0.10-0.30 (brighter = more HF)

    // Wire buzz: high-frequency resonance (the "snare" sound)
    let wire_freq = 1800.0 + color.brightness * 1200.0; // 1800-3000Hz

    let mut noise_state = 0.0f32;
    let mut wire_phase = 0.0f32;
    let click_samples = ((0.002 + (1.0 - vel) * 0.003) * sr) as usize;

    for i in 0..n {
        let t = i as f32 / sr;

        // Attack transient
        let attack = if i < click_samples {
            let at = i as f32 / click_samples as f32;
            (1.0 + 0.2 * vel) * (1.0 - (-at * 10.0).exp())
        } else {
            1.0
        };

        // Body envelope (fast attack, shaped decay)
        let env_body = (-t * decay_body).exp() * attack;

        // Noise envelope (slightly slower decay for sustain)
        let env_noise = (-t * decay_noise).exp() * attack;

        // Body: resonant tone
        let body = (std::f32::consts::TAU * body_freq * t).sin() * env_body * body_mix;

        // Noise: filtered white noise (the main snare character)
        let white = lcg_noise(i as u32 + 4242);
        noise_state += lp_alpha * (white - noise_state);
        let noise = noise_state * env_noise * noise_mix;

        // Wire buzz: adds the characteristic snare rattle
        wire_phase += std::f32::consts::TAU * wire_freq / sr;
        let wire = wire_phase.sin() * (-t * 40.0).exp() * 0.08 * vel;

        // Transient pop
        let pop = if i < click_samples {
            let pop_env = 1.0 - (i as f32 / click_samples as f32);
            lcg_noise(i as u32 + 9999) * pop_env * pop_env * vel * 0.3
        } else {
            0.0
        };

        let sample = (body + noise + wire + pop) * vel;
        buf.push(sample);
    }
    buf
}

// ─── Hi-Hat ──────────────────────────────────────────────────────────────────

fn render_hihat(sr: f32, velocity: f32, color: &DrumColor) -> Vec<f32> {
    let vel = velocity.clamp(0.0, 1.0);

    // Velocity shapes open/closed: loud = longer ring (open hat), soft = tight
    let duration = 0.02 + vel * 0.08; // 20-100ms (was fixed 50ms)
    let n = (duration * sr) as usize;
    let mut buf = Vec::with_capacity(n);

    let decay = 40.0 + (1.0 - vel) * 40.0 + color.tightness * 20.0; // faster at low vel

    // High-pass cutoff: brightness drives metallic shimmer
    let hp_alpha = 0.03 + color.brightness * 0.06; // 0.03-0.09

    // Metallic resonances (6 square-root-related frequencies for inharmonic spectrum)
    let metal_freqs = [
        3500.0 + color.brightness * 2000.0,
        5100.0 + color.brightness * 1500.0,
        7200.0 + color.brightness * 1000.0,
    ];

    let mut hp_state = 0.0f32;
    let mut metal_phases = [0.0f32; 3];

    for i in 0..n {
        let t = i as f32 / sr;
        let env = (-t * decay).exp();

        // Noise component (dominant)
        let white = lcg_noise(i as u32 + 1337);
        hp_state += hp_alpha * (white - hp_state);
        let hp_noise = (white - hp_state) * 0.7;

        // Metallic resonances (adds the "ting" character)
        let mut metal = 0.0f32;
        for (k, &freq) in metal_freqs.iter().enumerate() {
            metal_phases[k] += std::f32::consts::TAU * freq / sr;
            metal += metal_phases[k].sin() * 0.08 / (k as f32 + 1.0);
        }

        let sample = (hp_noise + metal) * env * vel * 0.35;
        buf.push(sample);
    }
    buf
}

// ─── Noise Generator ─────────────────────────────────────────────────────────

/// Fast deterministic noise via LCG. Returns [-1, 1].
#[inline]
fn lcg_noise(seed: u32) -> f32 {
    let s = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (s as f32 / u32::MAX as f32) * 2.0 - 1.0
}

// ─── Pattern Generation ──────────────────────────────────────────────────────

/// Generate a drum pattern based on consciousness state.
///
/// Returns drum hits for one bar. Velocity varies with arousal and position
/// in the bar (downbeats accented, ghost notes on weak beats).
pub fn generate_pattern(bpm: f32, consciousness: f32, arousal: f32) -> Vec<DrumHit> {
    generate_pattern_full(bpm, consciousness, arousal, 0.5, 0.0)
}

/// Generate a drum pattern with full emotional modulation.
pub fn generate_pattern_full(
    bpm: f32,
    consciousness: f32,
    arousal: f32,
    dopamine: f32,
    valence: f32,
) -> Vec<DrumHit> {
    let beat_dur = 60.0 / bpm;
    let mut hits = Vec::new();

    // Dynamic range: arousal widens the gap between loud and soft hits
    let dynamic_range = 0.15 + arousal * 0.35; // 0.15-0.50
    let base_vel = 0.45 + dopamine * 0.15; // dopamine lifts overall energy

    // Kick on beats 1 and 3 — downbeat accented
    hits.push(DrumHit {
        drum: DrumType::Kick,
        time: 0.0,
        velocity: (base_vel + dynamic_range).clamp(0.2, 1.0), // strong downbeat
    });
    if arousal > 0.3 {
        hits.push(DrumHit {
            drum: DrumType::Kick,
            time: beat_dur * 2.0,
            velocity: (base_vel + dynamic_range * 0.7).clamp(0.2, 1.0), // slightly weaker
        });
    }

    // Snare on beats 2 and 4
    if consciousness > 0.3 {
        hits.push(DrumHit {
            drum: DrumType::Snare,
            time: beat_dur,
            velocity: (base_vel + dynamic_range * 0.8).clamp(0.2, 1.0),
        });
        hits.push(DrumHit {
            drum: DrumType::Snare,
            time: beat_dur * 3.0,
            velocity: (base_vel + dynamic_range * 0.85).clamp(0.2, 1.0),
        });

        // Ghost notes: soft snare drags on the "e" of 2 and 4 (adds groove)
        if consciousness > 0.6 && arousal > 0.4 {
            hits.push(DrumHit {
                drum: DrumType::Snare,
                time: beat_dur * 1.5,
                velocity: (base_vel * 0.35).clamp(0.1, 0.4), // ghost — very soft
            });
        }
    }

    // Hi-hats: density from arousal, velocity alternates (swing feel)
    if consciousness > 0.5 {
        let subdivisions = if arousal > 0.7 {
            8 // sixteenth notes
        } else if arousal > 0.4 {
            4 // eighth notes
        } else {
            2 // quarter notes
        };
        let step = beat_dur * 4.0 / subdivisions as f32;

        for i in 0..subdivisions {
            // Accent pattern: downbeat strong, offbeat soft, creates swing
            let accent = if i % 4 == 0 {
                1.0 // beat
            } else if i % 2 == 0 {
                0.7 // offbeat
            } else {
                0.4 // "and"
            };
            let vel = (base_vel * 0.5 * accent + dynamic_range * 0.2).clamp(0.1, 0.7);

            // Skip some weak hats at low arousal for breathing room
            if arousal < 0.5 && accent < 0.5 {
                continue;
            }

            hits.push(DrumHit {
                drum: DrumType::HiHat,
                time: step * i as f32,
                velocity: vel,
            });
        }
    }

    // Negative valence: add extra kick on the "and" of 4 (tension)
    if valence < -0.3 && consciousness > 0.5 {
        hits.push(DrumHit {
            drum: DrumType::Kick,
            time: beat_dur * 3.5,
            velocity: (base_vel * 0.6).clamp(0.2, 0.7),
        });
    }

    hits
}

/// Humanize drum hit timing with consciousness-driven jitter.
///
/// Adds subtle timing variations that make the pattern feel played by a human.
/// Higher consciousness = tighter timing (more precise internal clock).
pub fn humanize_hits(hits: &mut [DrumHit], consciousness: f32, seed: u32) {
    let max_jitter = 0.008 * (1.0 - consciousness * 0.7); // 2-8ms jitter

    for (i, hit) in hits.iter_mut().enumerate() {
        let noise = lcg_noise(seed.wrapping_add(i as u32 * 31));
        let jitter = noise * max_jitter;

        // Don't jitter the very first hit (downbeat anchor)
        if hit.time > 0.001 {
            hit.time = (hit.time + jitter).max(0.0);
        }

        // Subtle velocity humanization: ±5%
        let vel_noise = lcg_noise(seed.wrapping_add(i as u32 * 97 + 500));
        hit.velocity = (hit.velocity + vel_noise * 0.05).clamp(0.05, 1.0);
    }
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
            buf.len() < 6000,
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

    #[test]
    fn velocity_shapes_kick_duration() {
        let soft = render_drum_colored(
            &DrumHit { drum: DrumType::Kick, time: 0.0, velocity: 0.2 },
            44100, &DrumColor::default(),
        );
        let loud = render_drum_colored(
            &DrumHit { drum: DrumType::Kick, time: 0.0, velocity: 0.9 },
            44100, &DrumColor::default(),
        );
        assert!(
            loud.len() > soft.len(),
            "loud kick should be longer: loud={}, soft={}",
            loud.len(), soft.len()
        );
    }

    #[test]
    fn tightness_shortens_decay() {
        let loose = render_drum_colored(
            &DrumHit { drum: DrumType::Snare, time: 0.0, velocity: 0.7 },
            44100, &DrumColor { tightness: 0.0, ..Default::default() },
        );
        let tight = render_drum_colored(
            &DrumHit { drum: DrumType::Snare, time: 0.0, velocity: 0.7 },
            44100, &DrumColor { tightness: 1.0, ..Default::default() },
        );
        // Tight snare should have less energy in the tail
        let loose_tail: f32 = loose[loose.len() / 2..].iter().map(|s| s * s).sum::<f32>();
        let tight_tail: f32 = tight[tight.len() / 2..].iter().map(|s| s * s).sum::<f32>();
        assert!(
            tight_tail < loose_tail,
            "tight snare should have less tail energy"
        );
    }

    #[test]
    fn humanize_adds_jitter() {
        let mut hits = generate_pattern(120.0, 0.8, 0.6);
        let original_times: Vec<f32> = hits.iter().map(|h| h.time).collect();
        humanize_hits(&mut hits, 0.5, 42);
        let jittered_times: Vec<f32> = hits.iter().map(|h| h.time).collect();
        // At least some times should differ (all except beat 0)
        let diffs: usize = original_times.iter().zip(&jittered_times)
            .filter(|(&a, &b)| (a - b).abs() > 0.0001)
            .count();
        assert!(diffs > 0, "humanize should jitter at least some hit times");
    }

    #[test]
    fn ghost_notes_at_high_consciousness() {
        let pattern = generate_pattern_full(120.0, 0.8, 0.6, 0.5, 0.0);
        let ghost_snares: Vec<_> = pattern.iter()
            .filter(|h| h.drum == DrumType::Snare && h.velocity < 0.4)
            .collect();
        assert!(!ghost_snares.is_empty(), "high consciousness should produce ghost notes");
    }
}
