// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Render mel frames to WAV via additive-oscillator synthesis.
//!
//! Bridges the decoder (which produces log-mel frames) to something we can
//! actually listen to. Each mel bin becomes a sine oscillator at its
//! triangular filter's center frequency, amplitude = `exp(mel_val)`, crossfaded
//! between frames with a Hann ramp.
//!
//! This is a crude (mel → audio is lossy and the phase is made up), but it's
//! enough to hear whether the decoder's predicted mel contours track the real
//! performance.
//!
//! Handles both file formats:
//! - `.pairs.bin`  (state + mel) — extracts mel frames only
//! - `.pred.bin`   (mel frames only)
//!
//! ```sh
//! cargo run --release -p symthaea-muse --bin mel_to_wav -- \
//!     /opt/datasets/maestro/training_pairs/2017/<name>.pairs.pred.bin \
//!     /tmp/predicted.wav
//! ```

use std::io::{Read, Write};
use std::path::PathBuf;

const SAMPLE_RATE: u32 = 44100;
const HOP_LEN: usize = 512; // matches MelConfig default
const N_FFT: usize = 2048; // matches MelConfig default
const F_MIN: f32 = 20.0;
const F_MAX: f32 = 16000.0;

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Center frequencies for each of `n_mels` bins, matching MelConfig defaults.
fn mel_center_hz(n_mels: usize) -> Vec<f32> {
    let mel_min = hz_to_mel(F_MIN);
    let mel_max = hz_to_mel(F_MAX);
    // Mel points: n_mels + 2 including boundary edges; center is index m+1
    (0..n_mels)
        .map(|m| {
            let mel_point = mel_min + (mel_max - mel_min) * (m + 1) as f32 / (n_mels + 1) as f32;
            mel_to_hz(mel_point)
        })
        .collect()
}

/// Load a pairs.bin or pred.bin file and return the mel frames.
fn load_mel_frames(path: &std::path::Path) -> std::io::Result<Vec<Vec<f32>>> {
    let mut f = std::fs::File::open(path)?;
    let name = path.to_string_lossy().to_string();
    // .pairs.bin has a 17-float state block before each mel frame; any other
    // .bin we treat as a flat (count, mel_dim, mel_frames) file.
    let is_pairs = name.ends_with(".pairs.bin");

    let mut header = [0u8; 8];
    f.read_exact(&mut header)?;
    let count = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let mel_dim = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    let mut frames = Vec::with_capacity(count);

    // .pairs.bin stores 17 state floats before each mel frame; .pred.bin is
    // just mel frames.
    let state_bytes = if is_pairs { 17 * 4 } else { 0 };
    let mut pair_buf = vec![0u8; state_bytes + mel_dim * 4];
    for _ in 0..count {
        f.read_exact(&mut pair_buf)?;
        let mel_start = state_bytes;
        let mut mel = Vec::with_capacity(mel_dim);
        for chunk in pair_buf[mel_start..].chunks_exact(4) {
            mel.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        frames.push(mel);
    }
    Ok(frames)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: mel_to_wav <in.pairs.bin | in.pred.bin> <out.wav> [--max-frames N]");
        std::process::exit(1);
    }
    let in_path = PathBuf::from(&args[0]);
    let out_path = PathBuf::from(&args[1]);
    let max_frames = if args.len() >= 4 && args[2] == "--max-frames" {
        args[3].parse::<usize>().ok()
    } else {
        None
    };

    let mut frames = load_mel_frames(&in_path).expect("load mel frames");
    if let Some(n) = max_frames {
        frames.truncate(n);
    }
    if frames.is_empty() {
        eprintln!("No frames");
        std::process::exit(1);
    }
    let n_mels = frames[0].len();
    let centers = mel_center_hz(n_mels);

    println!("Input:   {}", in_path.display());
    println!("Frames:  {}", frames.len());
    println!("Mel dim: {}", n_mels);
    println!(
        "Center freqs: {:.1}..{:.1} Hz",
        centers.first().unwrap(),
        centers.last().unwrap()
    );

    // Additive synthesis: one oscillator per mel bin.
    // Amplitude = exp(log_mel) clipped to a reasonable range.
    // Between frames, linearly interpolate amplitudes (zero-order would click).
    let total_samples = frames.len() * HOP_LEN + N_FFT;
    let mut out = vec![0.0f32; total_samples];
    let mut phases = vec![0.0f32; n_mels];
    let sr = SAMPLE_RATE as f32;

    // Skip bins above Nyquist (shouldn't happen with default config, but safe)
    let active_bins: Vec<usize> = (0..n_mels).filter(|&b| centers[b] < sr * 0.5).collect();

    // Normalize mel values: typical log-mel is [-5, 5]; our baseline sees
    // values around -2 ± 1.5. Shift+scale into amplitudes [0, 1].
    let to_amp = |log_mel: f32| -> f32 {
        let shifted = (log_mel + 4.0).max(0.0) * 0.1; // -4 → 0, +6 → 1
        shifted.min(1.0)
    };

    for (frame_idx, frame) in frames.iter().enumerate() {
        let frame_start = frame_idx * HOP_LEN;
        let next_frame = frames.get(frame_idx + 1);

        for i in 0..HOP_LEN {
            let t = i as f32 / HOP_LEN as f32;
            let mut sample = 0.0f32;
            for &bin in &active_bins {
                let amp_now = to_amp(frame[bin]);
                let amp_next = next_frame.map(|f| to_amp(f[bin])).unwrap_or(amp_now);
                let amp = amp_now * (1.0 - t) + amp_next * t;

                let freq = centers[bin];
                phases[bin] += freq / sr * std::f32::consts::TAU;
                if phases[bin] > std::f32::consts::TAU {
                    phases[bin] -= std::f32::consts::TAU;
                }
                sample += phases[bin].sin() * amp;
            }
            // Scale down — sum of up to 128 sines needs aggressive attenuation.
            out[frame_start + i] = (sample * (1.0 / active_bins.len() as f32).sqrt()).tanh();
        }
    }

    // Write WAV (16-bit PCM mono, SAMPLE_RATE)
    let mut f = std::fs::File::create(&out_path).expect("create wav");
    let data_bytes = out.len() * 2;
    let riff_size = 36 + data_bytes as u32;
    // RIFF header
    f.write_all(b"RIFF").unwrap();
    f.write_all(&riff_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    // fmt chunk
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    f.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    f.write_all(&SAMPLE_RATE.to_le_bytes()).unwrap();
    f.write_all(&(SAMPLE_RATE * 2).to_le_bytes()).unwrap(); // byte rate
    f.write_all(&2u16.to_le_bytes()).unwrap(); // block align
    f.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    // data chunk
    f.write_all(b"data").unwrap();
    f.write_all(&(data_bytes as u32).to_le_bytes()).unwrap();
    for &s in &out {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes()).unwrap();
    }

    println!("Wrote {} samples → {}", out.len(), out_path.display());
    println!("Duration: {:.2}s", out.len() as f32 / sr);
}
