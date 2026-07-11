// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Train a linear valence/arousal regressor on REAL DEAM audio features.
//!
//! This is the honest replacement for the earlier scaffold, which never
//! decoded audio and synthesized its "features" directly from the V/A labels
//! it was fit to predict (target leakage — any R² it printed was circular).
//! This version:
//!
//! 1. Decodes each DEAM MP3 with symphonia (real audio, ~25s excerpt).
//! 2. Extracts 6 signal features: RMS, spectral centroid, zero-crossing
//!    rate, spectral flux, onset rate, low/high band-energy ratio.
//! 3. Fits ridge least squares on a SONG-LEVEL train split and reports R²
//!    and MAE on the held-out 10% — numbers a mean-predictor baseline
//!    contextualizes. Linear models on hand features are genuinely weak at
//!    valence (literature: arousal R² ~0.3-0.6, valence ~0.05-0.3); expect
//!    numbers in that range, not miracles.
//!
//! Usage:
//! ```bash
//! ./scripts/download_deam.sh   # once (~1.4 GB)
//! cargo run --release -p symthaea-muse --example train_deam_regressor
//! ```
//!
//! Output: `data/deam/va_regressor_weights.json` (weights + embedded
//! provenance/metrics).

use std::io::BufRead;
use std::path::{Path, PathBuf};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const SKIP_SECS: f32 = 5.0;
const KEEP_SECS: f32 = 25.0;
const N_WORKERS: usize = 10;

fn main() {
    println!("=== DEAM V-A Regressor (REAL audio features) ===\n");

    let audio_dir = Path::new("data/deam/MEMD_audio");
    let annot_path = "data/deam/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv";

    if !audio_dir.exists() {
        eprintln!("DEAM audio not found at {}", audio_dir.display());
        eprintln!("Run: ./scripts/download_deam.sh");
        return;
    }

    // 1. Load annotations
    let annotations = match load_annotations(annot_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load annotations: {e}");
            return;
        }
    };
    println!("Loaded {} annotations", annotations.len());

    // 2. Decode audio + extract features (parallel workers)
    println!("Decoding MP3s and extracting real features ({N_WORKERS} workers)...");
    let jobs: Vec<(u32, f32, f32, PathBuf)> = annotations
        .iter()
        .filter_map(|&(id, v, a)| {
            let p = audio_dir.join(format!("{id}.mp3"));
            p.exists().then_some((id, v, a, p))
        })
        .collect();
    println!("  {} tracks have audio on disk", jobs.len());

    let results = std::sync::Mutex::new(Vec::<(u32, f32, f32, [f32; 6])>::new());
    let next_job = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..N_WORKERS {
            scope.spawn(|| {
                loop {
                    let i = next_job.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(id, v, a, ref path)) = jobs.get(i) else {
                        break;
                    };
                    if let Some((samples, sr)) = decode_mp3_mono(path) {
                        if samples.len() > (sr as f32 * 5.0) as usize {
                            let feats = extract_features(&samples, sr);
                            let mut guard = results.lock().unwrap();
                            guard.push((id, v, a, feats));
                            if guard.len() % 200 == 0 {
                                println!("  {} tracks processed...", guard.len());
                            }
                        }
                    }
                }
            });
        }
    });
    let mut rows = results.into_inner().unwrap();
    rows.sort_by_key(|r| r.0); // deterministic order
    println!("  Extracted features from {} tracks\n", rows.len());
    if rows.len() < 300 {
        eprintln!("Too few decoded tracks for a trustworthy fit");
        return;
    }

    // 3. Song-level split: FNV(id) bucket 0 of 10 → test
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for row in &rows {
        if fnv(row.0) % 10 == 0 {
            test.push(row);
        } else {
            train.push(row);
        }
    }
    println!("Split: {} train / {} test songs", train.len(), test.len());

    let feats_of =
        |set: &[&(u32, f32, f32, [f32; 6])]| -> Vec<[f32; 6]> { set.iter().map(|r| r.3).collect() };
    let norm_v = |v: f32| (v - 5.0) / 4.0; // DEAM static valence 1..9 → [-1,1]
    let norm_a = |a: f32| (a - 5.0) / 4.0; // arousal likewise
    let tr_x = feats_of(&train);
    let tr_v: Vec<f32> = train.iter().map(|r| norm_v(r.1)).collect();
    let tr_a: Vec<f32> = train.iter().map(|r| norm_a(r.2)).collect();

    // 4. Fit
    let v_weights = fit_linear_regression(&tr_x, &tr_v);
    let a_weights = fit_linear_regression(&tr_x, &tr_a);

    // 5. Held-out evaluation vs mean-predictor baseline
    let te_x = feats_of(&test);
    let te_v: Vec<f32> = test.iter().map(|r| norm_v(r.1)).collect();
    let te_a: Vec<f32> = test.iter().map(|r| norm_a(r.2)).collect();
    let pv: Vec<f32> = te_x.iter().map(|f| predict(&v_weights, f)).collect();
    let pa: Vec<f32> = te_x.iter().map(|f| predict(&a_weights, f)).collect();

    let (v_r2, v_mae) = (r_squared(&te_v, &pv), mae(&te_v, &pv));
    let (a_r2, a_mae) = (r_squared(&te_a, &pa), mae(&te_a, &pa));
    let v_base_mae = mae_of_mean_predictor(&tr_v, &te_v);
    let a_base_mae = mae_of_mean_predictor(&tr_a, &te_a);

    println!("\n── Held-out evaluation ({} songs) ──", test.len());
    println!("             R²      MAE     mean-baseline-MAE");
    println!("  valence  {v_r2:>6.3}  {v_mae:>6.3}   {v_base_mae:>6.3}");
    println!("  arousal  {a_r2:>6.3}  {a_mae:>6.3}   {a_base_mae:>6.3}");
    println!("  (targets normalized to [-1,1]; R² vs test variance; a useful");
    println!("   model needs R² > 0 and MAE below the mean baseline)");

    // 6. Save weights with embedded provenance
    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let output = format!(
        "{{\n  \"provenance\": {{\n    \"trainer\": \"examples/train_deam_regressor.rs\",\n    \
         \"dataset\": \"DEAM (real decoded audio; symphonia mp3)\",\n    \"unix_time\": {unix},\n    \
         \"n_train\": {}, \"n_test\": {},\n    \
         \"split\": \"song-level FNV bucket 0 of 10\",\n    \
         \"features\": [\"rms\", \"centroid_hz\", \"zcr\", \"flux\", \"onset_rate\", \"low_high_ratio\"],\n    \
         \"held_out_valence_r2\": {v_r2:.4}, \"held_out_valence_mae\": {v_mae:.4},\n    \
         \"held_out_arousal_r2\": {a_r2:.4}, \"held_out_arousal_mae\": {a_mae:.4},\n    \
         \"mean_baseline_valence_mae\": {v_base_mae:.4}, \"mean_baseline_arousal_mae\": {a_base_mae:.4}\n  }},\n  \
         \"valence_weights\": {v_weights:?},\n  \"arousal_weights\": {a_weights:?}\n}}\n",
        train.len(),
        test.len(),
    );
    let out_path = "data/deam/va_regressor_weights.json";
    match std::fs::write(out_path, &output) {
        Ok(()) => println!("\nSaved to {out_path}"),
        Err(e) => eprintln!("Failed to save: {e}"),
    }
}

// ─── Audio decode + features ────────────────────────────────────────────────

/// Decode an MP3 to mono f32, keeping ~KEEP_SECS after skipping SKIP_SECS.
fn decode_mp3_mono(path: &Path) -> Option<(Vec<f32>, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?.clone();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;

    let mut samples = Vec::new();
    let mut sample_rate = 0u32;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break, // EOF or error — use what we have
        };
        if packet.track_id() != track.id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue, // skip corrupt frame
        };
        if sample_rate == 0 {
            sample_rate = decoded.spec().rate;
        }
        append_mono(&decoded, &mut samples);
        if sample_rate > 0
            && samples.len() >= ((SKIP_SECS + KEEP_SECS) * sample_rate as f32) as usize
        {
            break; // we have enough
        }
    }
    if sample_rate == 0 || samples.is_empty() {
        return None;
    }
    let skip = (SKIP_SECS * sample_rate as f32) as usize;
    if samples.len() <= skip {
        return None;
    }
    Some((samples.split_off(skip), sample_rate))
}

/// Downmix any decoded buffer to mono f32 and append.
fn append_mono(buf: &AudioBufferRef, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => {
            let chans = b.spec().channels.count();
            let frames = b.frames();
            for i in 0..frames {
                let mut s = 0.0f32;
                for c in 0..chans {
                    s += b.chan(c)[i];
                }
                out.push(s / chans as f32);
            }
        }
        AudioBufferRef::S16(b) => {
            let chans = b.spec().channels.count();
            let frames = b.frames();
            for i in 0..frames {
                let mut s = 0.0f32;
                for c in 0..chans {
                    s += b.chan(c)[i] as f32 / 32768.0;
                }
                out.push(s / chans as f32);
            }
        }
        AudioBufferRef::S32(b) => {
            let chans = b.spec().channels.count();
            let frames = b.frames();
            for i in 0..frames {
                let mut s = 0.0f32;
                for c in 0..chans {
                    s += b.chan(c)[i] as f32 / 2147483648.0;
                }
                out.push(s / chans as f32);
            }
        }
        _ => {} // other formats not produced by the mp3 decoder
    }
}

/// Extract 6 real audio features: [rms, centroid_hz, zcr, flux, onset_rate,
/// low/high band ratio]. Frame 2048 / hop 512, Hann window.
fn extract_features(samples: &[f32], sr: u32) -> [f32; 6] {
    use rustfft::{FftPlanner, num_complex::Complex};
    const N_FFT: usize = 2048;
    const HOP: usize = 512;

    // Time-domain: RMS + ZCR
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    let zc = samples
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    let zcr = zc as f32 / samples.len() as f32;

    // Spectral frames
    let fft = FftPlanner::new().plan_fft_forward(N_FFT);
    let hann: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (std::f32::consts::TAU * i as f32 / (N_FFT - 1) as f32).cos()))
        .collect();
    let bin_hz = sr as f32 / N_FFT as f32;

    let mut centroids = Vec::new();
    let mut fluxes = Vec::new();
    let (mut low_energy, mut high_energy) = (0.0f64, 0.0f64);
    let mut prev_mag: Vec<f32> = Vec::new();
    let mut pos = 0;
    while pos + N_FFT <= samples.len() {
        let mut buf: Vec<Complex<f32>> = samples[pos..pos + N_FFT]
            .iter()
            .zip(&hann)
            .map(|(&s, &w)| Complex { re: s * w, im: 0.0 })
            .collect();
        fft.process(&mut buf);
        let mag: Vec<f32> = buf[..N_FFT / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        let total: f32 = mag.iter().sum();
        if total > 1e-6 {
            let weighted: f32 = mag
                .iter()
                .enumerate()
                .map(|(k, &m)| k as f32 * bin_hz * m)
                .sum();
            centroids.push(weighted / total);
        }
        if !prev_mag.is_empty() {
            let flux: f32 = mag
                .iter()
                .zip(&prev_mag)
                .map(|(&m, &p)| (m - p).max(0.0))
                .sum();
            fluxes.push(flux / N_FFT as f32);
        }
        for (k, &m) in mag.iter().enumerate() {
            let f = k as f32 * bin_hz;
            if f < 250.0 {
                low_energy += (m * m) as f64;
            } else if f > 2000.0 {
                high_energy += (m * m) as f64;
            }
        }
        prev_mag = mag;
        pos += HOP;
    }

    let centroid = mean(&centroids);
    let flux_mean = mean(&fluxes);
    // Onset rate: flux peaks above 1.5x median, per second
    let onset_rate = if fluxes.len() > 4 {
        let mut sorted = fluxes.clone();
        sorted.sort_by(f32::total_cmp);
        let median = sorted[sorted.len() / 2];
        let thresh = median * 1.5 + 1e-6;
        let peaks = fluxes
            .windows(3)
            .filter(|w| w[1] > thresh && w[1] > w[0] && w[1] > w[2])
            .count();
        peaks as f32 / (samples.len() as f32 / sr as f32)
    } else {
        0.0
    };
    let low_high_ratio = (low_energy / (low_energy + high_energy + 1e-9)) as f32;

    [rms, centroid, zcr, flux_mean, onset_rate, low_high_ratio]
}

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

fn mae(actual: &[f32], predicted: &[f32]) -> f32 {
    actual
        .iter()
        .zip(predicted)
        .map(|(&a, &p)| (a - p).abs())
        .sum::<f32>()
        / actual.len().max(1) as f32
}

fn mae_of_mean_predictor(train_targets: &[f32], test_targets: &[f32]) -> f32 {
    let m = mean(train_targets);
    test_targets.iter().map(|&t| (t - m).abs()).sum::<f32>() / test_targets.len().max(1) as f32
}

fn fnv(id: u32) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in id.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ─── Annotations + linear algebra (unchanged from the scaffold) ─────────────

fn load_annotations(path: &str) -> Result<Vec<(u32, f32, f32)>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let reader = std::io::BufReader::new(file);
    let mut results = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line: String = line.map_err(|e| e.to_string())?;
        if i == 0 {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            if let (Ok(id), Ok(v), Ok(a)) = (
                parts[0].trim().parse::<u32>(),
                parts[1].trim().parse::<f32>(),
                parts[3].trim().parse::<f32>(),
            ) {
                results.push((id, v, a));
            }
        }
    }
    Ok(results)
}

/// Fit ordinary least squares linear regression.
/// Returns [bias, w1, w2, ..., w6] (7 weights).
fn fit_linear_regression(features: &[[f32; 6]], targets: &[f32]) -> Vec<f32> {
    let n = features.len();
    let d = 7; // bias + 6 features

    let mut xtx = vec![0.0f64; d * d];
    let mut xty = vec![0.0f64; d];

    for i in 0..n {
        let x = scaled_row(&features[i]);
        let y = targets[i] as f64;
        for r in 0..d {
            for c in 0..d {
                xtx[r * d + c] += x[r] * x[c];
            }
            xty[r] += x[r] * y;
        }
    }
    for i in 0..d {
        xtx[i * d + i] += 0.001 * n as f64;
    }
    solve_linear_system(&xtx, &xty, d)
        .into_iter()
        .map(|w| w as f32)
        .collect()
}

fn scaled_row(f: &[f32; 6]) -> [f64; 7] {
    [
        1.0,
        f[0] as f64,          // rms ~0-0.5
        f[1] as f64 / 1000.0, // centroid Hz → kHz
        f[2] as f64 * 10.0,   // zcr
        f[3] as f64 * 100.0,  // flux
        f[4] as f64,          // onset rate /s
        f[5] as f64,          // band ratio 0-1
    ]
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        x[i] = if diag.abs() > 1e-12 { sum / diag } else { 0.0 };
    }
    x
}

fn predict(weights: &[f32], features: &[f32; 6]) -> f32 {
    let x = scaled_row(features);
    let mut y = 0.0f64;
    for i in 0..7 {
        y += weights[i] as f64 * x[i];
    }
    y as f32
}

fn r_squared(actual: &[f32], predicted: &[f32]) -> f32 {
    let n = actual.len();
    let mean = actual.iter().sum::<f32>() / n as f32;
    let ss_tot: f32 = actual.iter().map(|&a| (a - mean).powi(2)).sum();
    let ss_res: f32 = actual
        .iter()
        .zip(predicted)
        .map(|(&a, &p)| (a - p).powi(2))
        .sum();
    if ss_tot > 1e-8 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}
