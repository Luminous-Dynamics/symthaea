// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Auto-mastering: LUFS measurement, corrective EQ, loudness normalization.
//!
//! Professional audio goes through mastering before release. This module
//! automates the process: measure loudness (LUFS), analyze frequency balance,
//! apply corrective EQ, normalize to -14 LUFS (streaming standard), compress,
//! and brick-wall limit.

use crate::mixing::{Limiter, ParametricEQ};

/// LUFS measurement result.
#[derive(Debug, Clone)]
pub struct LufsResult {
    pub integrated: f32,
    pub peak_db: f32,
    pub rms_db: f32,
    pub crest_factor_db: f32,
}

/// Frequency balance analysis.
#[derive(Debug, Clone)]
pub struct FrequencyBalance {
    pub low_energy: f32,  // 0-200 Hz
    pub mid_energy: f32,  // 200-2000 Hz
    pub high_energy: f32, // 2000+ Hz
}

impl FrequencyBalance {
    pub fn total(&self) -> f32 {
        self.low_energy + self.mid_energy + self.high_energy
    }
    pub fn low_ratio(&self) -> f32 {
        self.low_energy / self.total().max(1e-10)
    }
    pub fn mid_ratio(&self) -> f32 {
        self.mid_energy / self.total().max(1e-10)
    }
    pub fn high_ratio(&self) -> f32 {
        self.high_energy / self.total().max(1e-10)
    }
}

/// Mastering configuration.
pub struct MasteringConfig {
    pub target_lufs: f32,          // -14.0
    pub limiter_ceiling_db: f32,   // -1.0
    pub target_low_ratio: f32,     // 0.30
    pub target_mid_ratio: f32,     // 0.45
    pub target_high_ratio: f32,    // 0.25
    pub eq_max_correction_db: f32, // 6.0
    /// Treble BOOSTS are capped separately and much lower: acoustic chamber
    /// material naturally carries far less than the pop-normalized 25%
    /// high-band target, so a symmetric cap meant the master slammed +6dB
    /// of treble onto close-mic'd strings on essentially every render —
    /// heard as harshness. Cuts stay at the full correction range.
    pub eq_max_high_boost_db: f32, // 1.5
    /// Maximum dB a quiet/loud SECTION may be pulled toward the piece's
    /// integrated loudness (0 disables). A listening review flagged the
    /// section-level jumps as reading "rendered hot rather than intentional"
    /// — this rider narrows them gently instead of letting the limiter do
    /// it violently at the loud end.
    pub section_leveling_db: f32, // 1.5
}

impl Default for MasteringConfig {
    fn default() -> Self {
        Self {
            // -16 LUFS, not -14: this engine's output is dynamic acoustic
            // chamber material, and the hotter pop target pushed every
            // climax into the limiter (heard as "loud returns feel rendered
            // hot"). -16 matches the streaming target for exactly this kind
            // of material and leaves the limiter as a safety, not a sound.
            target_lufs: -16.0,
            limiter_ceiling_db: -1.5,
            target_low_ratio: 0.30,
            target_mid_ratio: 0.45,
            target_high_ratio: 0.25,
            eq_max_correction_db: 6.0,
            eq_max_high_boost_db: 1.5,
            section_leveling_db: 1.5,
        }
    }
}

/// Mastering result report.
#[derive(Debug, Clone)]
pub struct MasteringResult {
    pub input_lufs: f32,
    pub output_lufs: f32,
    pub gain_applied_db: f32,
    pub eq_corrections: [f32; 3],
    pub peak_before: f32,
    pub peak_after: f32,
}

/// A single biquad filter section (Direct Form I).
#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

const fn biquad(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Biquad {
    Biquad {
        b0,
        b1,
        b2,
        a1,
        a2,
        x1: 0.0,
        x2: 0.0,
        y1: 0.0,
        y2: 0.0,
    }
}

/// Build the two-stage ITU-R BS.1770 K-weighting filter for `sample_rate`.
///
/// Stage 1 is a high-shelf (+~4 dB above ~1.5 kHz, the head/torso model);
/// stage 2 is the RLB high-pass (~38 Hz). For the two sample rates that
/// matter to this crate (44.1 / 48 kHz) we use the EXACT published
/// coefficients (libebur128 / ITU reference). For any other rate we fall
/// back to an RBJ-cookbook derivation, which reproduces the reference to
/// within ~3% (< 0.3 dB) — documented as approximate.
fn k_weighting_filters(sample_rate: u32) -> [Biquad; 2] {
    match sample_rate {
        48000 => [
            // ITU-R BS.1770-4 reference (48 kHz)
            biquad(
                1.53512485958697,
                -2.69169618940638,
                1.19839281085285,
                -1.69065929318241,
                0.73248077421585,
            ),
            biquad(1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621),
        ],
        44100 => [
            // libebur128 reference (44.1 kHz)
            biquad(
                1.5308412300503478,
                -2.6509799951536985,
                1.1690790799210682,
                -1.6636551132560204,
                0.7125954280732254,
            ),
            biquad(1.0, -2.0, 1.0, -1.9891696736297957, 0.9891990357870394),
        ],
        _ => derive_k_weighting_rbj(sample_rate),
    }
}

/// RBJ-cookbook fallback for non-standard sample rates (approximate: within
/// ~3% of the ITU reference at 48 kHz). Only reached for unusual rates.
fn derive_k_weighting_rbj(sample_rate: u32) -> [Biquad; 2] {
    let fs = sample_rate as f64;
    let stage1 = {
        let f0 = 1681.974450955533;
        let gain_db = 3.999843853973347;
        let q = 0.7071752369554196;
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = std::f64::consts::TAU * f0 / fs;
        let (sn, cs) = w0.sin_cos();
        let alpha = sn / (2.0 * q);
        let sqrt_a = a.sqrt();
        let b0 = a * ((a + 1.0) + (a - 1.0) * cs + 2.0 * sqrt_a * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cs);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cs - 2.0 * sqrt_a * alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cs + 2.0 * sqrt_a * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cs);
        let a2 = (a + 1.0) - (a - 1.0) * cs - 2.0 * sqrt_a * alpha;
        biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    };
    let stage2 = {
        let f0 = 38.13547087602444;
        let q = 0.5003270373238773;
        let w0 = std::f64::consts::TAU * f0 / fs;
        let (sn, cs) = w0.sin_cos();
        let alpha = sn / (2.0 * q);
        let b0 = (1.0 + cs) / 2.0;
        let b1 = -(1.0 + cs);
        let b2 = (1.0 + cs) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cs;
        let a2 = 1.0 - alpha;
        biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    };
    [stage1, stage2]
}

/// Apply the K-weighting cascade to one channel.
fn k_weight_channel(samples: &[[f32; 2]], channel: usize, sample_rate: u32) -> Vec<f64> {
    let [mut s1, mut s2] = k_weighting_filters(sample_rate);
    samples
        .iter()
        .map(|frame| s2.process(s1.process(frame[channel] as f64)))
        .collect()
}

/// Measure integrated loudness per ITU-R BS.1770 (K-weighted, gated).
///
/// Full BS.1770-4 chain: two-stage K-weighting pre-filter → 400 ms blocks →
/// channel-summed mean square (L+R, each weight 1.0 for stereo) → −0.691 dB
/// offset → −70 LUFS absolute gate → −10 dB relative gate. `integrated` is a
/// genuine LUFS value comparable to streaming targets (−14 LUFS etc.).
/// (`peak_db`/`rms_db` remain un-weighted sample-domain measures.)
pub fn measure_lufs(samples: &[[f32; 2]], sample_rate: u32) -> LufsResult {
    if samples.is_empty() {
        return LufsResult {
            integrated: -70.0,
            peak_db: -70.0,
            rms_db: -70.0,
            crest_factor_db: 0.0,
        };
    }

    // K-weight each channel (BS.1770 stage 1: perceptual pre-filter).
    let kl = k_weight_channel(samples, 0, sample_rate);
    let kr = k_weight_channel(samples, 1, sample_rate);

    let window_size = (sample_rate as f32 * 0.4) as usize; // 400ms
    let mut block_powers: Vec<f32> = Vec::new();
    let mut peak = 0.0f32;
    let mut total_power = 0.0f64;

    for start in (0..samples.len()).step_by(window_size.max(1)) {
        let end = (start + window_size).min(samples.len());
        let mut power = 0.0f64;
        for i in start..end {
            // BS.1770 channel SUM of K-weighted powers (L+R, weight 1.0 each).
            power += kl[i] * kl[i] + kr[i] * kr[i];
            // Peak stays in the un-weighted sample domain (true-peak proxy).
            peak = peak.max(samples[i][0].abs()).max(samples[i][1].abs());
        }
        let mean_power = power / (end - start).max(1) as f64;
        total_power += power;
        block_powers.push(mean_power as f32);
    }

    let total_mean_power = total_power / samples.len() as f64;
    let rms = (total_mean_power as f32).sqrt();
    let rms_db = if rms > 1e-10 {
        20.0 * rms.log10()
    } else {
        -70.0
    };
    let peak_db = if peak > 1e-10 {
        20.0 * peak.log10()
    } else {
        -70.0
    };

    // Absolute gate: exclude blocks below -70 LUFS
    let gated: Vec<f32> = block_powers
        .iter()
        .copied()
        .filter(|&p| p > 1e-7) // ~-70 dBFS
        .collect();

    let integrated = if gated.is_empty() {
        -70.0
    } else {
        let mean = gated.iter().sum::<f32>() / gated.len() as f32;
        // Relative gate: exclude blocks below -10 dB of ungated mean
        let relative_threshold = mean * 0.1; // -10 dB
        let final_blocks: Vec<f32> = gated
            .iter()
            .copied()
            .filter(|&p| p >= relative_threshold)
            .collect();
        if final_blocks.is_empty() {
            -70.0
        } else {
            let final_mean = final_blocks.iter().sum::<f32>() / final_blocks.len() as f32;
            -0.691 + 10.0 * final_mean.max(1e-10).log10()
        }
    };

    LufsResult {
        integrated,
        peak_db,
        rms_db,
        crest_factor_db: peak_db - rms_db,
    }
}

/// Analyze frequency balance using simple band-pass energy measurement.
pub fn analyze_balance(samples: &[[f32; 2]], sample_rate: u32) -> FrequencyBalance {
    // Simple approach: measure energy in three bands using one-pole filters
    let mut low = 0.0f64;
    let mut mid = 0.0f64;
    let mut high = 0.0f64;

    // State for filters
    let mut lp_state = 0.0f32; // 200Hz low-pass
    let mut hp_state = 0.0f32; // 2000Hz high-pass
    let sr = sample_rate as f32;
    let lp_coeff = (std::f32::consts::TAU * 200.0 / sr).min(0.99);
    let hp_coeff = (std::f32::consts::TAU * 2000.0 / sr).min(0.99);

    for s in samples {
        let mono = (s[0] + s[1]) * 0.5;

        // Low-pass for bass
        lp_state += lp_coeff * (mono - lp_state);
        low += (lp_state * lp_state) as f64;

        // High-pass for treble
        hp_state += hp_coeff * (mono - hp_state);
        let hp_out = mono - hp_state;
        high += (hp_out * hp_out) as f64;

        // Mid = total - low - high
        let mid_out = mono - lp_state - hp_out;
        mid += (mid_out * mid_out) as f64;
    }

    let n = samples.len().max(1) as f64;
    FrequencyBalance {
        low_energy: (low / n).sqrt() as f32,
        mid_energy: (mid / n).sqrt() as f32,
        high_energy: (high / n).sqrt() as f32,
    }
}

/// Auto-master a stereo audio buffer in-place.
pub fn auto_master(
    samples: &mut Vec<[f32; 2]>,
    sample_rate: u32,
    config: &MasteringConfig,
) -> MasteringResult {
    // 1. Measure input
    let input = measure_lufs(samples, sample_rate);
    let balance = analyze_balance(samples, sample_rate);

    // 2. Compute corrective EQ
    let low_correction = ratio_to_db(config.target_low_ratio / balance.low_ratio().max(0.01))
        .clamp(-config.eq_max_correction_db, config.eq_max_correction_db);
    let mid_correction = ratio_to_db(config.target_mid_ratio / balance.mid_ratio().max(0.01))
        .clamp(-config.eq_max_correction_db, config.eq_max_correction_db);
    let high_correction = ratio_to_db(config.target_high_ratio / balance.high_ratio().max(0.01))
        .clamp(-config.eq_max_correction_db, config.eq_max_high_boost_db);

    // 3. Apply EQ
    let mut eq_l = ParametricEQ::new(low_correction, mid_correction, high_correction);
    let mut eq_r = ParametricEQ::new(low_correction, mid_correction, high_correction);
    let sr = sample_rate as f32;
    for s in samples.iter_mut() {
        s[0] = eq_l.process(s[0], sr);
        s[1] = eq_r.process(s[1], sr);
    }

    // 4. Loudness normalization
    let post_eq = measure_lufs(samples, sample_rate);
    let gain_db = config.target_lufs - post_eq.integrated;
    let gain_linear = 10.0f32.powf(gain_db / 20.0);

    for s in samples.iter_mut() {
        s[0] *= gain_linear;
        s[1] *= gain_linear;
    }

    // 4.5 Section leveling: pull each ~3s window's loudness gently toward
    // the piece's own integrated loudness, capped at ±section_leveling_db,
    // with the gain curve smoothed over ~2s so it rides sections, not notes.
    if config.section_leveling_db > 0.0 {
        apply_section_leveling(samples, sample_rate, config.section_leveling_db);
    }

    // 5. Brick-wall limiting
    let mut limiter = Limiter::new(sample_rate, config.limiter_ceiling_db);
    for s in samples.iter_mut() {
        let (l, r) = limiter.process(s[0], s[1]);
        s[0] = l;
        s[1] = r;
    }

    // 6. Measure output
    let output = measure_lufs(samples, sample_rate);

    MasteringResult {
        input_lufs: input.integrated,
        output_lufs: output.integrated,
        gain_applied_db: gain_db,
        eq_corrections: [low_correction, mid_correction, high_correction],
        peak_before: input.peak_db,
        peak_after: output.peak_db,
    }
}

fn ratio_to_db(ratio: f32) -> f32 {
    20.0 * ratio.max(1e-10).log10()
}

/// Two-pass slow gain rider. Pass 1 measures RMS loudness per ~3s window
/// and derives a per-window correction pulling it HALFWAY toward the
/// piece-wide mean, clamped to ±`max_db`. Pass 2 applies the curve through
/// a one-pole smoother with a ~2s time constant — far too slow to pump on
/// notes, fast enough to narrow section-to-section jumps. Silence (below
/// -60dB RMS) is left untouched so pauses and the final decay don't get
/// dragged upward.
fn apply_section_leveling(samples: &mut [[f32; 2]], sample_rate: u32, max_db: f32) {
    const WINDOW_SECS: f32 = 3.0;
    const SMOOTH_SECS: f32 = 2.0;
    const SILENCE_FLOOR_DB: f32 = -60.0;
    let win = (WINDOW_SECS * sample_rate as f32) as usize;
    if win == 0 || samples.len() < win * 2 {
        return; // too short to have "sections"
    }
    let rms_db: Vec<f32> = samples
        .chunks(win)
        .map(|chunk| {
            let energy: f32 = chunk
                .iter()
                .map(|s| 0.5 * (s[0] * s[0] + s[1] * s[1]))
                .sum::<f32>()
                / chunk.len() as f32;
            10.0 * energy.max(1e-12).log10()
        })
        .collect();
    let audible: Vec<f32> = rms_db
        .iter()
        .copied()
        .filter(|&db| db > SILENCE_FLOOR_DB)
        .collect();
    if audible.len() < 2 {
        return;
    }
    let mean_db = audible.iter().sum::<f32>() / audible.len() as f32;
    let target_gain_db: Vec<f32> = rms_db
        .iter()
        .map(|&db| {
            if db <= SILENCE_FLOOR_DB {
                0.0
            } else {
                (0.5 * (mean_db - db)).clamp(-max_db, max_db)
            }
        })
        .collect();
    // One-pole smoothing toward the current window's target gain.
    let alpha = 1.0 - (-1.0 / (SMOOTH_SECS * sample_rate as f32)).exp();
    let mut gain_db = target_gain_db[0];
    for (i, s) in samples.iter_mut().enumerate() {
        let target = target_gain_db[(i / win).min(target_gain_db.len() - 1)];
        gain_db += alpha * (target - gain_db);
        let g = 10.0f32.powf(gain_db / 20.0);
        s[0] *= g;
        s[1] *= g;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn treble_boost_is_capped_for_dark_material() {
        // One-pole low-passed noise ≈ acoustic chamber balance: the old
        // symmetric ±6dB clamp let the high band boost to +6dB here.
        let mut state = 0.0f32;
        let mut rng = 12345u32;
        let mut samples: Vec<[f32; 2]> = (0..44100)
            .map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let white = (rng >> 8) as f32 / 8388608.0 - 1.0;
                state += 0.02 * (white - state); // dark
                [state * 0.5, state * 0.5]
            })
            .collect();
        let result = super::auto_master(&mut samples, 44100, &super::MasteringConfig::default());
        assert!(
            result.eq_corrections[2]
                <= super::MasteringConfig::default().eq_max_high_boost_db + 1e-3,
            "high-band boost {} exceeds the cap",
            result.eq_corrections[2]
        );
    }

    use super::*;

    #[test]
    fn section_leveling_narrows_the_gap_without_inverting_it() {
        let sr = 44100u32;
        // 6s quiet section, 6s loud section (sine so RMS is well-defined).
        let mut samples: Vec<[f32; 2]> = (0..(12 * sr) as usize)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let amp = if t < 6.0 { 0.05 } else { 0.5 }; // 20dB apart
                let s = amp * (t * 440.0 * std::f32::consts::TAU).sin();
                [s, s]
            })
            .collect();
        let rms = |s: &[[f32; 2]]| {
            let e: f32 = s.iter().map(|x| x[0] * x[0]).sum::<f32>() / s.len() as f32;
            10.0 * e.max(1e-12).log10()
        };
        let n = samples.len();
        let (q0, l0) = (rms(&samples[..n / 2]), rms(&samples[n / 2..]));
        apply_section_leveling(&mut samples, sr, 1.5);
        let (q1, l1) = (rms(&samples[..n / 2]), rms(&samples[n / 2..]));
        let gap0 = l0 - q0;
        let gap1 = l1 - q1;
        assert!(gap1 < gap0 - 0.5, "gap must narrow: {gap0:.2} -> {gap1:.2}");
        // Bounded: each side moved at most max_db (+ smoothing slack), and
        // the loud section is still the loud section.
        assert!(gap1 > gap0 - 2.0 * 1.5 - 0.5);
        assert!(gap1 > 3.0, "dynamics must survive leveling");
    }

    #[test]
    fn section_leveling_leaves_silence_alone() {
        let sr = 44100u32;
        // 6s tone then 6s of silence (a final decay/pause).
        let mut samples: Vec<[f32; 2]> = (0..(12 * sr) as usize)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let amp = if t < 6.0 { 0.3 } else { 0.0 };
                let s = amp * (t * 440.0 * std::f32::consts::TAU).sin();
                [s, s]
            })
            .collect();
        apply_section_leveling(&mut samples, sr, 1.5);
        let tail_peak = samples[samples.len() - 44100..]
            .iter()
            .map(|s| s[0].abs())
            .fold(0.0f32, f32::max);
        assert_eq!(tail_peak, 0.0, "silence must not be gained up");
    }

    fn sine_wave(freq: f32, duration_secs: f32, amplitude: f32, sr: u32) -> Vec<[f32; 2]> {
        let n = (duration_secs * sr as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let s = amplitude * (t * freq * std::f32::consts::TAU).sin();
                [s, s]
            })
            .collect()
    }

    #[test]
    fn lufs_calibration_sine() {
        // -20 dBFS sine at 1kHz should measure approximately -23 LUFS
        let amplitude = 10.0f32.powf(-20.0 / 20.0); // -20 dBFS
        let samples = sine_wave(1000.0, 3.0, amplitude, 44100);
        let result = measure_lufs(&samples, 44100);
        // LUFS = -0.691 + 10*log10(mean_power), for sine: mean_power = A²/2
        // Expected: -0.691 + 10*log10(0.01/2) ≈ -0.691 + (-23.01) ≈ -23.7
        assert!(
            result.integrated < -18.0,
            "should be quiet: {}",
            result.integrated
        );
        assert!(
            result.integrated > -28.0,
            "should be audible: {}",
            result.integrated
        );
    }

    #[test]
    fn auto_master_normalizes() {
        let mut samples = sine_wave(440.0, 2.0, 0.01, 44100); // very quiet
        let config = MasteringConfig::default();
        let result = auto_master(&mut samples, 44100, &config);

        assert!(
            result.output_lufs > result.input_lufs,
            "should be louder after mastering"
        );
        assert!(
            result.gain_applied_db > 0.0,
            "should have applied positive gain"
        );
    }

    #[test]
    fn limiter_respects_ceiling() {
        let mut samples = sine_wave(440.0, 1.0, 0.9, 44100);
        let config = MasteringConfig {
            target_lufs: -6.0,
            ..Default::default()
        };
        auto_master(&mut samples, 44100, &config);

        let ceiling = 10.0f32.powf(-1.0 / 20.0);
        let max_peak = samples
            .iter()
            .flat_map(|s| [s[0].abs(), s[1].abs()])
            .fold(0.0f32, f32::max);
        assert!(
            max_peak <= ceiling + 0.01,
            "peak should be below ceiling: {max_peak}"
        );
    }

    #[test]
    fn k_weighting_attenuates_low_frequencies() {
        // BS.1770 K-weighting must read equal-amplitude low bass as QUIETER
        // than a 1 kHz reference (the RLB high-pass + head-model shelf).
        let amp = 10.0f32.powf(-20.0 / 20.0);
        let ref_1k = measure_lufs(&sine_wave(1000.0, 3.0, amp, 44100), 44100);
        let bass_40 = measure_lufs(&sine_wave(40.0, 3.0, amp, 44100), 44100);
        let high_6k = measure_lufs(&sine_wave(6000.0, 3.0, amp, 44100), 44100);
        assert!(
            bass_40.integrated < ref_1k.integrated - 6.0,
            "40Hz should be >6 LU quieter than 1kHz: 40Hz={} 1kHz={}",
            bass_40.integrated,
            ref_1k.integrated
        );
        // High-shelf: 6 kHz should read LOUDER than 1 kHz (the +4 dB lift)
        assert!(
            high_6k.integrated > ref_1k.integrated + 1.0,
            "6kHz should be lifted vs 1kHz: 6kHz={} 1kHz={}",
            high_6k.integrated,
            ref_1k.integrated
        );
    }

    #[test]
    fn k_weighting_coeffs_exact_at_standard_rates() {
        // 44.1 and 48 kHz use the exact published tables — assert bit-close.
        let [s1_48, s2_48] = k_weighting_filters(48000);
        assert!((s1_48.b0 - 1.53512485958697).abs() < 1e-9);
        assert!((s1_48.a1 - -1.69065929318241).abs() < 1e-9);
        assert!((s2_48.a2 - 0.99007225036621).abs() < 1e-9);
        let [s1_44, s2_44] = k_weighting_filters(44100);
        assert!((s1_44.b0 - 1.5308412300503478).abs() < 1e-9);
        assert!((s1_44.a1 - -1.6636551132560204).abs() < 1e-9);
        assert!((s2_44.a2 - 0.9891990357870394).abs() < 1e-9);
    }

    #[test]
    fn frequency_balance_detects_bass() {
        let samples = sine_wave(80.0, 1.0, 0.5, 44100); // pure bass
        let balance = analyze_balance(&samples, 44100);
        assert!(
            balance.low_energy > balance.high_energy,
            "bass sine should have more low energy: low={} high={}",
            balance.low_energy,
            balance.high_energy
        );
    }
}
