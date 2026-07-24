// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The user listened to `kokoro_singing_intelligibility_gate`'s output
//! ("sounded like random fart noises") -- a stronger, more diagnostic
//! signal than the WER numbers alone: the OLD formant vocoder sounded like
//! "a real attempt at singing" to the same listener, so this isn't just
//! "singing confuses Whisper," it's a real quality regression in the new
//! Kokoro phase-vocoder pipeline. This tool inspects the actual generated
//! WAVs to find the mechanism: objective render-cleanliness diagnostics
//! (clicks, clipping, DC offset -- reusing `singing_quality.rs`, already
//! built for exactly this) plus a coarse autocorrelation pitch track over
//! time, since a buzzy/low-frequency "fart" character is the classic
//! symptom of either (a) a pitch-shift ratio gone wrong (target frequency
//! far below the source's natural range) or (b) phase-vocoder overlap-add
//! breaking down (near-zero normalization denominators, see `phase_vocoder`
//! in kokoro_singing.rs).
//!
//! Registered as `[[bin]]` (same reason as `kokoro_singing_intelligibility_gate`:
//! symthaea-humanoid, an unconditional dev-dependency, is broken at HEAD).
//!
//! ```bash
//! nix develop -c cargo run --bin kokoro_singing_diagnose --features voice-tts -- <path/to.wav> [...]
//! ```

use std::path::PathBuf;

fn estimate_f0(samples: &[f32], sample_rate: u32) -> Option<f32> {
    if samples.len() < 64 {
        return None;
    }
    let min_lag = (sample_rate as usize / 800).max(2);
    let max_lag = (sample_rate as usize / 50).min(samples.len() / 2);
    if min_lag >= max_lag {
        return None;
    }
    let mut best = (0usize, f64::NEG_INFINITY);
    for lag in min_lag..=max_lag {
        let (dot, ea, eb) = samples[..samples.len() - lag]
            .iter()
            .zip(&samples[lag..])
            .fold((0.0f64, 0.0f64, 0.0f64), |(dot, ea, eb), (&a, &b)| {
                (
                    dot + (a * b) as f64,
                    ea + (a * a) as f64,
                    eb + (b * b) as f64,
                )
            });
        let denom = (ea * eb).sqrt();
        if denom < 1e-9 {
            continue;
        }
        let score = dot / denom;
        if score > best.1 {
            best = (lag, score);
        }
    }
    (best.0 > 0 && best.1 > 0.2).then(|| sample_rate as f32 / best.0 as f32)
}

fn dc_offset(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f32>() / samples.len() as f32
}

fn peak_and_rms(samples: &[f32]) -> (f32, f32) {
    let peak = samples.iter().fold(0.0f32, |p, &x| p.max(x.abs()));
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    (peak, rms)
}

fn clipped_fraction(samples: &[f32]) -> f32 {
    let clipped = samples.iter().filter(|&&x| x.abs() >= 0.999).count();
    clipped as f32 / samples.len().max(1) as f32
}

/// Fraction of samples near-silent relative to this file's OWN peak --
/// added 2026-07-22 in response to direct listening feedback ("most of
/// the audio is silent"). Relative, not absolute, so quiet-but-legitimate
/// passages aren't miscounted as silence in a loud file.
fn silence_fraction(samples: &[f32], peak: f32) -> f32 {
    let threshold = (peak * 0.01).max(1e-4);
    let silent = samples.iter().filter(|&&x| x.abs() < threshold).count();
    silent as f32 / samples.len().max(1) as f32
}

/// Longest run of consecutive near-silent samples, in seconds -- a single
/// short dropout reads very differently from one long dead patch even at
/// the same overall silence_fraction.
fn longest_silent_run_s(samples: &[f32], peak: f32, sample_rate: u32) -> f32 {
    let threshold = (peak * 0.01).max(1e-4);
    let mut longest = 0usize;
    let mut current = 0usize;
    for &x in samples {
        if x.abs() < threshold {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    longest as f32 / sample_rate as f32
}

fn max_sample_delta(samples: &[f32]) -> f32 {
    samples
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f32::max)
}

/// Top-N largest sample-to-sample deltas with their sample index / time --
/// pinpoints exactly where in the phrase (and thus which piece/segment
/// boundary in kokoro_singing.rs) the worst clicks live.
fn top_deltas(samples: &[f32], sample_rate: u32, n: usize) -> Vec<(usize, f32, f32)> {
    let mut deltas: Vec<(usize, f32)> = samples
        .windows(2)
        .enumerate()
        .map(|(i, w)| (i, (w[1] - w[0]).abs()))
        .collect();
    deltas.sort_by(|a, b| b.1.total_cmp(&a.1));
    deltas.truncate(n);
    deltas
        .into_iter()
        .map(|(i, d)| (i, i as f32 / sample_rate as f32, d))
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let paths: Vec<PathBuf> = if args.is_empty() {
        std::fs::read_dir("audio_output/kokoro_singing_2026-07-21")?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "wav"))
            .collect()
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    for path in paths {
        let mut reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.unwrap_or(0) as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect()
            }
        };

        let (peak, rms) = peak_and_rms(&samples);
        let dc = dc_offset(&samples);
        let clip = clipped_fraction(&samples);
        let max_delta = max_sample_delta(&samples);

        println!(
            "\n=== {} ===  {:.2}s @ {sample_rate}Hz",
            path.display(),
            samples.len() as f32 / sample_rate as f32
        );
        let rms_dbfs = 20.0 * rms.max(1e-9).log10();
        println!(
            "  peak={peak:.3}  rms={rms:.4}  rms_dbfs={rms_dbfs:.1}  dc_offset={dc:+.5}  clipped_frac={clip:.4}  max_sample_delta={max_delta:.3}"
        );

        let silence_frac = silence_fraction(&samples, peak);
        let longest_silent = longest_silent_run_s(&samples, peak, sample_rate);
        println!("  silence_fraction={silence_frac:.3}  longest_silent_run={longest_silent:.2}s");

        let cleanliness =
            symthaea::voice::singing_quality::analyze_render_cleanliness(&samples, sample_rate);
        println!(
            "  [crate's own gate] peak_sample_delta={:.3} (pass<=0.12)  contextual_click_events={}  max_click_score={:.3}  normalized_first_difference_rms={:.3} (pass<=0.75)  render_cleanliness_pass={}",
            cleanliness.peak_sample_delta,
            cleanliness.contextual_click_events,
            cleanliness.max_contextual_click_score,
            cleanliness.normalized_first_difference_rms,
            cleanliness.render_cleanliness_pass
        );

        println!("  top 8 largest sample-to-sample deltas (sample_idx, time_s, delta):");
        for (idx, t, d) in top_deltas(&samples, sample_rate, 8) {
            println!("    idx={idx:>8}  t={t:6.3}s  delta={d:.3}");
        }

        // Coarse pitch track: 40ms windows, 20ms hop, over the whole file.
        let frame = (sample_rate as f32 * 0.04) as usize;
        let hop = (sample_rate as f32 * 0.02) as usize;
        if samples.len() >= frame {
            print!("  pitch track (Hz, '-' = unvoiced):");
            let mut voiced_count = 0usize;
            let mut total = 0usize;
            let mut below_human = 0usize; // < 60 Hz: below any real voice, "buzz" territory
            let mut freqs: Vec<f32> = Vec::new();
            for start in (0..=samples.len() - frame).step_by(hop.max(1)) {
                total += 1;
                match estimate_f0(&samples[start..start + frame], sample_rate) {
                    Some(f0) => {
                        voiced_count += 1;
                        if f0 < 60.0 {
                            below_human += 1;
                        }
                        freqs.push(f0);
                    }
                    None => {}
                }
            }
            // Print a compact summary rather than every frame (files run 8-14s).
            if !freqs.is_empty() {
                freqs.sort_by(f32::total_cmp);
                let median = freqs[freqs.len() / 2];
                let min = freqs[0];
                let max = freqs[freqs.len() - 1];
                println!(
                    " voiced {voiced_count}/{total} frames, median {median:.0}Hz, range [{min:.0}, {max:.0}]Hz, {below_human} frames < 60Hz"
                );
            } else {
                println!(" no voiced frames detected at all");
            }
        }
    }

    Ok(())
}
