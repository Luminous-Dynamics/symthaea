// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio quality analyzer: detects crackling, measures dynamics, spectral content,
//! percussion quality, and compares against professional benchmarks.
//!
//! Run: cargo run --release -p symthaea-muse --example audio_analyzer -- audio_output/*.wav

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: audio_analyzer <wav_files...>");
        eprintln!(
            "Example: cargo run --release -p symthaea-muse --example audio_analyzer -- audio_output/*.wav"
        );
        return;
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Audio Quality Analyzer                               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    for path in &args {
        if path.ends_with(".wav") {
            analyze_wav(Path::new(path));
        }
    }
}

fn analyze_wav(path: &Path) {
    let reader = match hound::WavReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  Error reading {}: {e}", path.display());
            return;
        }
    };

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as f32;
    let channels = spec.channels as usize;

    // Read all samples as f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    // Extract left channel
    let left: Vec<f32> = samples.iter().step_by(channels).copied().collect();
    let right: Vec<f32> = if channels > 1 {
        samples.iter().skip(1).step_by(channels).copied().collect()
    } else {
        left.clone()
    };

    let duration = left.len() as f32 / sample_rate;
    let peak = left.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let rms = (left.iter().map(|s| s * s).sum::<f32>() / left.len() as f32).sqrt();
    let crest_db = if rms > 0.0 {
        20.0 * (peak / rms).log10()
    } else {
        0.0
    };
    let peak_db = 20.0 * (peak + 0.0001).log10();
    let rms_db = 20.0 * (rms + 0.0001).log10();

    // === Crackling Detection ===
    // Crackling = sudden large sample-to-sample jumps (discontinuities)
    let mut crackle_count = 0u32;
    let crackle_threshold = 0.3; // 30% of full scale jump in one sample
    for i in 1..left.len() {
        let diff = (left[i] - left[i - 1]).abs();
        if diff > crackle_threshold {
            crackle_count += 1;
        }
    }
    let crackle_rate = crackle_count as f32 / duration;

    // === Zero-crossing analysis (spectral centroid proxy) ===
    let crossings: u32 = left
        .windows(2)
        .map(|w| if (w[0] > 0.0) != (w[1] > 0.0) { 1 } else { 0 })
        .sum();
    let zcr = crossings as f32 / duration;

    // === Silence Analysis ===
    let silence_threshold = 0.005; // -46 dBFS
    let silent_samples = left
        .iter()
        .filter(|&&s| s.abs() < silence_threshold)
        .count();
    let silence_pct = silent_samples as f32 / left.len() as f32 * 100.0;

    // === Note onset detection ===
    let window_size = (sample_rate * 0.01) as usize; // 10ms windows
    let mut onsets = Vec::new();
    let mut prev_rms = 0.0f32;
    for i in (0..left.len().saturating_sub(window_size)).step_by(window_size) {
        let chunk = &left[i..i + window_size];
        let chunk_rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        if chunk_rms > prev_rms * 2.5 && chunk_rms > 0.01 {
            let t = i as f32 / sample_rate;
            if onsets.is_empty() || (t - onsets.last().unwrap()) > 0.05 {
                onsets.push(t);
            }
        }
        prev_rms = chunk_rms.max(0.001);
    }

    let note_density = onsets.len() as f32 / duration;
    let iois: Vec<f32> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
    let ioi_mean = if !iois.is_empty() {
        iois.iter().sum::<f32>() / iois.len() as f32
    } else {
        0.0
    };
    let ioi_sd = if iois.len() > 1 {
        let mean = ioi_mean;
        (iois.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / iois.len() as f32).sqrt()
    } else {
        0.0
    };
    let ioi_cv = if ioi_mean > 0.0 {
        ioi_sd / ioi_mean
    } else {
        0.0
    };

    // === Dynamic range per second ===
    let chunk_size = sample_rate as usize;
    let mut rms_per_sec = Vec::new();
    for i in (0..left.len()).step_by(chunk_size) {
        let chunk = &left[i..(i + chunk_size).min(left.len())];
        if chunk.len() > 100 {
            let r = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            rms_per_sec.push(r);
        }
    }
    let dyn_range = if !rms_per_sec.is_empty() {
        let max_r = rms_per_sec.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_r = rms_per_sec.iter().fold(f32::MAX, |a, &b| a.min(b));
        if min_r > 0.0001 { max_r / min_r } else { 0.0 }
    } else {
        0.0
    };

    // === Stereo width ===
    let n = left.len().min(right.len());
    let l_mean: f32 = left[..n].iter().sum::<f32>() / n as f32;
    let r_mean: f32 = right[..n].iter().sum::<f32>() / n as f32;
    let mut cov = 0.0f32;
    let mut l_var = 0.0f32;
    let mut r_var = 0.0f32;
    for i in 0..n {
        let ld = left[i] - l_mean;
        let rd = right[i] - r_mean;
        cov += ld * rd;
        l_var += ld * ld;
        r_var += rd * rd;
    }
    let correlation = if l_var > 0.0 && r_var > 0.0 {
        cov / (l_var.sqrt() * r_var.sqrt())
    } else {
        1.0
    };
    let stereo_width = 1.0 - correlation.abs();

    // === Percussion detection ===
    // Percussion = short, loud transients with fast decay
    let mut percussion_events = 0u32;
    for i in 1..onsets.len() {
        let onset_t = onsets[i];
        let onset_sample = (onset_t * sample_rate) as usize;
        if onset_sample + (sample_rate as usize / 10) < left.len() {
            // Check if energy drops by 50% within 100ms (percussive decay)
            let attack_rms = {
                let s = onset_sample;
                let e = (s + (sample_rate * 0.01) as usize).min(left.len());
                (left[s..e].iter().map(|x| x * x).sum::<f32>() / (e - s) as f32).sqrt()
            };
            let decay_rms = {
                let s = onset_sample + (sample_rate * 0.05) as usize;
                let e = (s + (sample_rate * 0.01) as usize).min(left.len());
                if e > s && s < left.len() {
                    (left[s..e].iter().map(|x| x * x).sum::<f32>() / (e - s).max(1) as f32).sqrt()
                } else {
                    attack_rms
                }
            };
            if decay_rms < attack_rms * 0.5 {
                percussion_events += 1;
            }
        }
    }

    // === Print Report ===
    let name = path.file_name().unwrap().to_str().unwrap();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  Duration:       {:.1}s | {}Hz {}ch {}bit",
        duration, spec.sample_rate, channels, spec.bits_per_sample
    );
    println!("  Peak:           {:.3} ({:.1} dBFS)", peak, peak_db);
    println!("  RMS:            {:.4} ({:.1} dBFS)", rms, rms_db);
    println!("  Crest Factor:   {:.1} dB", crest_db);
    println!();

    // Crackling verdict
    let crackle_verdict = if crackle_count == 0 {
        "CLEAN"
    } else if crackle_rate < 0.5 {
        "MINOR"
    } else if crackle_rate < 5.0 {
        "NOTICEABLE"
    } else {
        "SEVERE"
    };
    println!(
        "  Crackling:      {} events ({:.1}/sec) — {}",
        crackle_count, crackle_rate, crackle_verdict
    );

    println!("  Silence:        {:.1}%", silence_pct);
    println!("  Stereo Width:   {:.3} (0=mono, 1=wide)", stereo_width);
    println!("  Dynamic Range:  {:.1}x", dyn_range);
    println!();
    println!(
        "  Note Onsets:    {} ({:.2}/sec)",
        onsets.len(),
        note_density
    );
    println!("  IOI Mean:       {:.0}ms", ioi_mean * 1000.0);
    println!("  IOI SD:         {:.0}ms", ioi_sd * 1000.0);
    println!("  IOI CV:         {:.3} (timing variability)", ioi_cv);
    println!(
        "  Percussion:     {} events ({:.0}% of onsets)",
        percussion_events,
        if !onsets.is_empty() {
            percussion_events as f32 / onsets.len() as f32 * 100.0
        } else {
            0.0
        }
    );
    println!("  ZCR:            {:.0} Hz (brightness proxy)", zcr);

    // Benchmarks
    println!();
    println!("  ── Benchmarks ──");
    if rms_db < -28.0 {
        println!("  ⚠ RMS too quiet ({:.0} dBFS, target -20 to -14)", rms_db);
    } else if rms_db > -10.0 {
        println!("  ⚠ RMS too loud ({:.0} dBFS, may clip)", rms_db);
    } else {
        println!("  ✓ RMS level OK ({:.0} dBFS)", rms_db);
    }

    if crackle_count > 0 {
        println!("  ⚠ Crackling detected ({} events)", crackle_count);
    } else {
        println!("  ✓ No crackling");
    }

    if stereo_width < 0.15 {
        println!("  ⚠ Nearly mono (width {:.3}, target >0.25)", stereo_width);
    } else if stereo_width > 0.25 {
        println!("  ✓ Good stereo width ({:.3})", stereo_width);
    } else {
        println!("  ~ Moderate stereo width ({:.3})", stereo_width);
    }

    if percussion_events == 0 && note_density > 1.0 {
        println!("  ⚠ No percussion detected despite note activity");
    } else if percussion_events > 0 {
        println!("  ✓ Percussion present ({} events)", percussion_events);
    }

    if ioi_cv < 0.1 && note_density > 0.5 {
        println!(
            "  ⚠ Very uniform timing (CV={:.3}, may sound mechanical)",
            ioi_cv
        );
    } else if ioi_cv > 0.3 {
        println!("  ✓ Good timing variation (CV={:.3})", ioi_cv);
    }

    println!();
}
