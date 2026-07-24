// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Formant-track extraction from raw audio: LPC analysis → [`FormantFrame`]s.
//!
//! Built for the voice plan's oracle test and Kokoro-teacher distillation
//! (2026-07-16): given verified-intelligible reference audio (e.g. Kokoro
//! output), recover the formant/f0/energy/voicing trajectory in the exact
//! frame format the vocal-tract controller and `FormantVocoder` consume.
//!
//! Method (classic speech DSP, deliberately dependency-free):
//! 1. Resample to 10 kHz (formant band of interest is F1–F3 ≤ ~3.5 kHz).
//! 2. 25 ms Hamming frames at the pipeline's 200 Hz frame rate (5 ms hop).
//! 3. Pre-emphasis → autocorrelation → Levinson-Durbin LPC (order 12).
//! 4. Formants = peaks of the LPC envelope evaluated on a frequency grid
//!    (peak-picking instead of polynomial root-finding: simpler, robust
//!    enough for resynthesis targets).
//! 5. f0 + voicing from normalized autocorrelation in the 60–400 Hz lag
//!    range; energy from frame RMS (peak-normalized per utterance).

use crate::types::{FormantFrame, SourceType};

/// Configuration for formant-track extraction.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Output frame rate (Hz). 200 matches the vocal-tract pipeline/vocoder.
    pub frame_rate: f32,
    /// Analysis window length in seconds (25 ms is the classic choice).
    pub window_secs: f32,
    /// Internal analysis sample rate for LPC (formants ≤ ~sr/2 − margin).
    pub analysis_sr: u32,
    /// LPC order (rule of thumb: 2 + sr_kHz, so ~12 at 10 kHz).
    pub lpc_order: usize,
    /// Pre-emphasis coefficient.
    pub preemphasis: f32,
    /// f0 search range (Hz).
    pub f0_min: f32,
    pub f0_max: f32,
    /// Periodicity threshold for the voiced decision (0..1).
    pub voicing_threshold: f32,
    /// Relative energy floor below which a frame is Silent.
    pub silence_rel_energy: f32,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            frame_rate: 200.0,
            window_secs: 0.025,
            analysis_sr: 10_000,
            lpc_order: 16,
            preemphasis: 0.97,
            f0_min: 60.0,
            f0_max: 400.0,
            voicing_threshold: 0.30,
            silence_rel_energy: 0.02,
        }
    }
}

/// Extract a 200 Hz [`FormantFrame`] track from mono audio.
///
/// Returns an empty vec for empty/too-short input. Frames carry F1–F3 with
/// estimated bandwidths, f0, per-utterance-normalized energy, a voicing
/// score, and a coarse `SourceType` (Vowel / Fricative / Silent) — enough to
/// drive [`crate::speech::vocoder`] or the root crate's `FormantVocoder` for
/// resynthesis, and to serve as distillation targets.
pub fn extract_formant_frames(
    samples: &[f32],
    sample_rate: u32,
    config: &ExtractionConfig,
) -> Vec<FormantFrame> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    let audio = resample_linear(samples, sample_rate, config.analysis_sr);
    let sr = config.analysis_sr as f32;
    let hop = (sr / config.frame_rate).round() as usize;
    let win_len = (config.window_secs * sr).round() as usize;
    if audio.len() < win_len || hop == 0 {
        return Vec::new();
    }

    let hamming: Vec<f32> = (0..win_len)
        .map(|n| 0.54 - 0.46 * (std::f32::consts::TAU * n as f32 / (win_len - 1) as f32).cos())
        .collect();

    let n_frames = (audio.len() - win_len) / hop + 1;
    let mut frames = Vec::with_capacity(n_frames);
    let mut peak_rms = 0.0f32;

    for fi in 0..n_frames {
        let start = fi * hop;
        let frame = &audio[start..start + win_len];

        // Energy (pre-normalization) on the raw frame.
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / win_len as f32).sqrt();
        peak_rms = peak_rms.max(rms);

        // f0 + periodicity from normalized autocorrelation (no pre-emphasis:
        // pitch lives in the low band).
        let windowed_raw: Vec<f32> = frame.iter().zip(&hamming).map(|(s, w)| s * w).collect();
        let (f0, periodicity) = estimate_f0(&windowed_raw, sr, config.f0_min, config.f0_max);

        // LPC on the pre-emphasized, windowed frame.
        let mut emph = vec![0.0f32; win_len];
        emph[0] = frame[0];
        for n in 1..win_len {
            emph[n] = frame[n] - config.preemphasis * frame[n - 1];
        }
        for (e, w) in emph.iter_mut().zip(&hamming) {
            *e *= w;
        }
        let lpc = lpc_coefficients(&emph, config.lpc_order);
        let formants = lpc_envelope_peaks(&lpc, sr, 3);

        let (f1, b1) = formants.first().copied().unwrap_or((500.0, 90.0));
        let (f2, b2) = formants.get(1).copied().unwrap_or((1500.0, 110.0));
        let (f3, b3) = formants.get(2).copied().unwrap_or((2500.0, 150.0));

        frames.push(FormantFrame {
            f1,
            f2,
            f3,
            b1,
            b2,
            b3,
            f0,
            energy: rms, // normalized below once peak_rms is known
            voicing: periodicity.clamp(0.0, 1.0),
            time: fi as f32 / config.frame_rate,
            source_type: SourceType::Vowel, // finalized below
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        });
    }

    // Per-utterance energy normalization + source-type decision.
    let norm = if peak_rms > 1e-9 {
        0.85 / peak_rms
    } else {
        0.0
    };
    for frame in &mut frames {
        frame.energy = (frame.energy * norm).clamp(0.0, 1.0);
        let voiced = frame.voicing >= config.voicing_threshold;
        frame.source_type = if frame.energy < config.silence_rel_energy {
            frame.voicing = 0.0;
            frame.f0 = 0.0;
            frame.energy = 0.0;
            SourceType::Silent
        } else if voiced {
            SourceType::Vowel
        } else {
            frame.f0 = 0.0;
            SourceType::Fricative
        };
    }

    median3_smooth(&mut frames);
    frames
}

/// Median-of-3 smoothing on F1–F3 and f0 to suppress single-frame LPC jumps.
fn median3_smooth(frames: &mut [FormantFrame]) {
    fn med3(a: f32, b: f32, c: f32) -> f32 {
        a.max(b.min(c)).min(b.max(c))
    }
    if frames.len() < 3 {
        return;
    }
    let snapshot: Vec<(f32, f32, f32, f32)> =
        frames.iter().map(|f| (f.f1, f.f2, f.f3, f.f0)).collect();
    for i in 1..frames.len() - 1 {
        let (p, c, n) = (snapshot[i - 1], snapshot[i], snapshot[i + 1]);
        frames[i].f1 = med3(p.0, c.0, n.0);
        frames[i].f2 = med3(p.1, c.1, n.1);
        frames[i].f3 = med3(p.2, c.2, n.2);
        if frames[i].source_type == SourceType::Vowel {
            frames[i].f0 = med3(p.3, c.3, n.3);
        }
    }
}

/// Normalized-autocorrelation f0 estimate. Returns (f0_hz, periodicity 0..1).
fn estimate_f0(windowed: &[f32], sr: f32, f0_min: f32, f0_max: f32) -> (f32, f32) {
    let n = windowed.len();
    let r0: f32 = windowed.iter().map(|s| s * s).sum();
    if r0 <= 1e-12 {
        return (0.0, 0.0);
    }
    let lag_min = (sr / f0_max).floor().max(2.0) as usize;
    let lag_max = ((sr / f0_min).ceil() as usize).min(n - 1);
    let mut best_lag = 0usize;
    let mut best_r = 0.0f32;
    for lag in lag_min..=lag_max {
        let mut r = 0.0f32;
        for i in 0..n - lag {
            r += windowed[i] * windowed[i + lag];
        }
        if r > best_r {
            best_r = r;
            best_lag = lag;
        }
    }
    if best_lag == 0 {
        return (0.0, 0.0);
    }
    (sr / best_lag as f32, (best_r / r0).clamp(0.0, 1.0))
}

/// Levinson-Durbin LPC coefficients `a[1..=order]` (a\[0\] = 1 implied);
/// returned vec has length `order` and represents A(z) = 1 - Σ a_k z^-k.
fn lpc_coefficients(frame: &[f32], order: usize) -> Vec<f32> {
    let n = frame.len();
    let mut r = vec![0.0f32; order + 1];
    for (lag, r_lag) in r.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for i in 0..n - lag {
            acc += frame[i] * frame[i + lag];
        }
        *r_lag = acc;
    }
    if r[0] <= 1e-12 {
        return vec![0.0; order];
    }

    let mut a = vec![0.0f32; order + 1];
    let mut err = r[0];
    for i in 1..=order {
        let mut acc = r[i];
        for j in 1..i {
            acc -= a[j] * r[i - j];
        }
        let k = acc / err.max(1e-12);
        // Update coefficients.
        let mut new_a = a.clone();
        new_a[i] = k;
        for j in 1..i {
            new_a[j] = a[j] - k * a[i - j];
        }
        a = new_a;
        err *= 1.0 - k * k;
        if err <= 0.0 {
            break;
        }
    }
    a[1..=order].to_vec()
}

/// Peaks of the LPC spectral envelope 1/|A(e^{-jω})| on a frequency grid,
/// with half-power bandwidth estimates. Returns up to `max_peaks` ascending
/// (freq, bandwidth) pairs in the 150 Hz .. 0.45·sr band.
fn lpc_envelope_peaks(lpc: &[f32], sr: f32, max_peaks: usize) -> Vec<(f32, f32)> {
    const GRID: usize = 512;
    let f_lo = 150.0f32;
    let f_hi = sr * 0.45;
    let mut mag = [0.0f32; GRID];
    for (g, m) in mag.iter_mut().enumerate() {
        let f = f_lo + (f_hi - f_lo) * g as f32 / (GRID - 1) as f32;
        let w = std::f32::consts::TAU * f / sr;
        // A(e^{-jw}) = 1 - Σ a_k e^{-jwk}
        let mut re = 1.0f32;
        let mut im = 0.0f32;
        for (k, &ak) in lpc.iter().enumerate() {
            let phase = w * (k + 1) as f32;
            re -= ak * phase.cos();
            im += ak * phase.sin();
        }
        *m = 1.0 / (re * re + im * im).sqrt().max(1e-9);
    }

    let bin_hz = (f_hi - f_lo) / (GRID - 1) as f32;
    let mut peaks: Vec<(f32, f32)> = Vec::new();
    for g in 1..GRID - 1 {
        if mag[g] > mag[g - 1] && mag[g] >= mag[g + 1] {
            let f = f_lo + bin_hz * g as f32;
            // Half-power bandwidth: scan outwards to mag/sqrt(2).
            let half = mag[g] / std::f32::consts::SQRT_2;
            let mut lo = g;
            while lo > 0 && mag[lo] > half {
                lo -= 1;
            }
            let mut hi = g;
            while hi < GRID - 1 && mag[hi] > half {
                hi += 1;
            }
            let bw = ((hi - lo) as f32 * bin_hz).clamp(60.0, 400.0);
            peaks.push((f, bw));
        }
    }
    // Keep the `max_peaks` lowest-frequency peaks with a minimum separation,
    // preferring prominent ones when crowded: sort by frequency, greedily
    // accept peaks ≥ 200 Hz apart.
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(f32, f32)> = Vec::new();
    for p in peaks {
        if out.last().is_none_or(|l| p.0 - l.0 >= 200.0) {
            out.push(p);
            if out.len() == max_peaks {
                break;
            }
        }
    }
    out
}

/// Linear resampler (shared convention with the rest of the voice stack).
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    (0..output_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let idx = src as usize;
            let frac = (src - idx as f64) as f32;
            match (input.get(idx), input.get(idx + 1)) {
                (Some(&a), Some(&b)) => a * (1.0 - frac) + b * frac,
                (Some(&a), None) => a,
                _ => 0.0,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::{self, VoiceProsody};

    #[test]
    fn silence_yields_silent_frames() {
        let samples = vec![0.0f32; 24_000];
        let frames = extract_formant_frames(&samples, 24_000, &ExtractionConfig::default());
        assert!(!frames.is_empty());
        assert!(
            frames
                .iter()
                .all(|f| f.source_type == SourceType::Silent && f.energy == 0.0)
        );
    }

    #[test]
    fn pure_tone_f0_recovered() {
        // 150 Hz tone at 24 kHz: f0 estimate within a few Hz, voiced.
        let sr = 24_000u32;
        let samples: Vec<f32> = (0..sr as usize)
            .map(|i| (std::f32::consts::TAU * 150.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let frames = extract_formant_frames(&samples, sr, &ExtractionConfig::default());
        let voiced: Vec<&FormantFrame> = frames
            .iter()
            .filter(|f| f.source_type == SourceType::Vowel)
            .collect();
        assert!(
            voiced.len() > frames.len() / 2,
            "a pure tone should be mostly voiced ({}/{})",
            voiced.len(),
            frames.len()
        );
        let mean_f0: f32 = voiced.iter().map(|f| f.f0).sum::<f32>() / voiced.len() as f32;
        assert!(
            (mean_f0 - 150.0).abs() < 10.0,
            "f0 estimate off: {mean_f0} vs 150"
        );
    }

    #[test]
    fn synthetic_vowel_formants_recovered() {
        // Synthesize a sustained /ɑ/-like vowel with the in-crate speech
        // vocoder (known targets F1=730, F2=1090) and check the extractor
        // recovers F1/F2 in the right neighborhoods. Generous tolerances:
        // the small vocoder adds noise/reverb and LPC peak-picking is coarse
        // — this is a resynthesis-target extractor, not a phonetics lab.
        let phonemes = vec![speech::g2p::Phoneme {
            ipa: "ɑ",
            is_vowel: true,
            stress: 1,
            base_duration_ms: 600.0,
        }];
        let prosody = VoiceProsody {
            arousal: 0.4,
            valence: 0.0,
            consciousness: 0.5,
            serotonin: 0.3,
        };
        let frames_in = speech::formants::phonemes_to_frames(&phonemes, &prosody, 24_000);
        let audio = speech::vocoder::synthesize(&frames_in, 24_000);
        assert!(!audio.is_empty());

        let frames = extract_formant_frames(&audio, 24_000, &ExtractionConfig::default());
        let voiced: Vec<&FormantFrame> = frames
            .iter()
            .filter(|f| f.source_type == SourceType::Vowel && f.energy > 0.2)
            .collect();
        assert!(
            !voiced.is_empty(),
            "sustained vowel should have voiced frames"
        );

        let mean_f1: f32 = voiced.iter().map(|f| f.f1).sum::<f32>() / voiced.len() as f32;
        let mean_f2: f32 = voiced.iter().map(|f| f.f2).sum::<f32>() / voiced.len() as f32;
        assert!(
            (500.0..=950.0).contains(&mean_f1),
            "F1 should be near 730 for /ɑ/: got {mean_f1}"
        );
        assert!(
            (850.0..=1400.0).contains(&mean_f2),
            "F2 should be near 1090 for /ɑ/: got {mean_f2}"
        );
        assert!(mean_f2 > mean_f1 + 150.0, "F2 must sit above F1");
    }

    #[test]
    fn frame_rate_and_timing() {
        let samples = vec![0.1f32; 24_000]; // 1s
        let frames = extract_formant_frames(&samples, 24_000, &ExtractionConfig::default());
        // ~200 frames per second (window shaves a few off the tail).
        assert!(
            (185..=200).contains(&frames.len()),
            "expected ~200 frames for 1s, got {}",
            frames.len()
        );
        let dt = frames[1].time - frames[0].time;
        assert!((dt - 0.005).abs() < 1e-4, "5ms hop expected, got {dt}");
    }
}
