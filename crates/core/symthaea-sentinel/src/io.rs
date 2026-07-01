// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio I/O for sentinel processing.
//!
//! This module provides:
//! - `AudioPump`: Real-time audio capture (requires `live-audio` feature)
//! - `FileAudioPump`: File-based audio processing
//! - `DatasetProcessor`: Batch processing of audio directories

use std::f32::consts::PI;

#[cfg(feature = "live-audio")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "live-audio")]
use ringbuf::{
    HeapRb,
    traits::{Consumer, Producer, Split},
};
use rustfft::{FftPlanner, num_complex::Complex};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use walkdir::WalkDir;

// =============================================================================
// Audio Configuration
// =============================================================================

/// Configuration for audio capture
#[derive(Clone, Copy)]
pub struct AudioConfig {
    pub window_size: usize,
    pub hop_size: usize,
    pub apply_window: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            window_size: 2048,
            hop_size: 1024,
            apply_window: true,
        }
    }
}

/// Configuration for file audio processing
#[derive(Clone)]
pub struct FileAudioConfig {
    pub window_size: usize,
    pub hop_size: usize,
    pub apply_window: bool,
}

impl Default for FileAudioConfig {
    fn default() -> Self {
        Self {
            window_size: 2048,
            hop_size: 1024,
            apply_window: true,
        }
    }
}

// =============================================================================
// Live Audio Pump (requires live-audio feature)
// =============================================================================

#[cfg(feature = "live-audio")]
pub struct AudioPump {
    _stream: cpal::Stream,
    consumer: ringbuf::HeapCons<f32>,
    fft_planner: FftPlanner<f32>,
    config: AudioConfig,
    sample_rate: f32,
    sample_buffer: Vec<f32>,
    window_coeffs: Vec<f32>,
}

#[cfg(feature = "live-audio")]
impl AudioPump {
    pub fn new(config: AudioConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No audio input device found")?;

        let supported_config = device.default_input_config()?;
        let sample_rate = u32::from(supported_config.sample_rate()) as f32;
        let channels = supported_config.channels() as usize;

        let device_name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "Unknown".to_string());
        println!(
            "Audio Device: {} ({} Hz, {} ch)",
            device_name, sample_rate, channels
        );

        let rb = HeapRb::<f32>::new(config.window_size * 4);
        let (mut producer, consumer) = rb.split();

        let window_coeffs: Vec<f32> = (0..config.window_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (config.window_size - 1) as f32).cos()))
            .collect();

        let stream_config: cpal::StreamConfig = supported_config.into();

        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if channels == 1 {
                    for &sample in data {
                        let _ = producer.try_push(sample);
                    }
                } else {
                    for chunk in data.chunks(channels) {
                        let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                        let _ = producer.try_push(mono);
                    }
                }
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?;

        stream.play()?;

        Ok(Self {
            _stream: stream,
            consumer,
            fft_planner: FftPlanner::new(),
            config,
            sample_rate,
            sample_buffer: Vec::with_capacity(config.window_size * 2),
            window_coeffs,
        })
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn control_rate(&self) -> f32 {
        self.sample_rate / self.config.hop_size as f32
    }

    pub fn window_size(&self) -> usize {
        self.config.window_size
    }

    fn pull_samples(&mut self) {
        while let Some(sample) = self.consumer.try_pop() {
            self.sample_buffer.push(sample);
        }
    }

    pub fn next_spectrum(&mut self) -> Option<Vec<f32>> {
        self.pull_samples();

        if self.sample_buffer.len() < self.config.window_size {
            return None;
        }

        let window: Vec<f32> = self.sample_buffer[..self.config.window_size].to_vec();
        self.sample_buffer.drain(..self.config.hop_size);

        let windowed: Vec<f32> = if self.config.apply_window {
            window
                .iter()
                .zip(&self.window_coeffs)
                .map(|(s, w)| s * w)
                .collect()
        } else {
            window
        };

        let fft = self.fft_planner.plan_fft_forward(self.config.window_size);
        let mut complex_buffer: Vec<Complex<f32>> = windowed
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();

        fft.process(&mut complex_buffer);

        let n_bins = self.config.window_size / 2 + 1;
        let magnitude: Vec<f32> = complex_buffer
            .iter()
            .take(n_bins)
            .map(|c| c.norm())
            .collect();

        Some(magnitude)
    }

    pub fn next_power_spectrum(&mut self) -> Option<Vec<f32>> {
        self.next_spectrum()
            .map(|mag| mag.iter().map(|m| m * m).collect())
    }

    #[allow(dead_code)]
    pub fn ready(&self) -> bool {
        self.sample_buffer.len() >= self.config.window_size
    }

    #[allow(dead_code)]
    pub fn buffer_fill(&self) -> usize {
        self.sample_buffer.len()
    }
}

// =============================================================================
// File Audio Pump
// =============================================================================

pub struct FileAudioPump {
    samples: Vec<f32>,
    position: usize,
    sample_rate: f32,
    fft_planner: FftPlanner<f32>,
    config: FileAudioConfig,
    window_coeffs: Vec<f32>,
    file_path: String,
}

impl FileAudioPump {
    pub fn new<P: AsRef<Path>>(path: P, config: FileAudioConfig) -> Result<Self, String> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        let file = File::open(&path).map_err(|e| format!("Failed to open {path_str}: {e}"))?;

        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.as_ref().extension() {
            hint.with_extension(&ext.to_string_lossy());
        }

        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| format!("Failed to probe {path_str}: {e}"))?;

        let mut format = probed.format;

        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| format!("No audio track found in {path_str}"))?;

        let track_id = track.id;

        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| "Unknown sample rate".to_string())? as f32;

        let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

        let dec_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .map_err(|e| format!("Failed to create decoder: {e}"))?;

        let mut all_samples: Vec<f32> = Vec::new();

        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => {
                    eprintln!("    Warning: decode error: {e}");
                    continue;
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("    Warning: decode error: {e}");
                    continue;
                }
            };

            let samples = Self::audio_buffer_to_f32_mono(&decoded, channels);
            all_samples.extend(samples);
        }

        if all_samples.is_empty() {
            return Err(format!("No audio data decoded from {path_str}"));
        }

        let duration_secs = all_samples.len() as f32 / sample_rate;

        println!(
            "  Loaded: {} ({:.1}s, {:.0} Hz, {} samples)",
            path_str,
            duration_secs,
            sample_rate,
            all_samples.len()
        );

        let window_coeffs: Vec<f32> = (0..config.window_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (config.window_size - 1) as f32).cos()))
            .collect();

        Ok(Self {
            samples: all_samples,
            position: 0,
            sample_rate,
            fft_planner: FftPlanner::new(),
            config,
            window_coeffs,
            file_path: path_str,
        })
    }

    fn audio_buffer_to_f32_mono(buffer: &AudioBufferRef, channels: usize) -> Vec<f32> {
        match buffer {
            AudioBufferRef::F32(buf) => {
                if channels == 1 {
                    buf.chan(0).to_vec()
                } else {
                    let frames = buf.frames();
                    (0..frames)
                        .map(|i| {
                            let sum: f32 = (0..channels.min(buf.spec().channels.count()))
                                .map(|ch| buf.chan(ch)[i])
                                .sum();
                            sum / channels as f32
                        })
                        .collect()
                }
            }
            AudioBufferRef::S16(buf) => {
                let frames = buf.frames();
                let ch_count = buf.spec().channels.count();
                (0..frames)
                    .map(|i| {
                        let sum: f32 = (0..ch_count)
                            .map(|ch| buf.chan(ch)[i] as f32 / 32768.0)
                            .sum();
                        sum / ch_count as f32
                    })
                    .collect()
            }
            AudioBufferRef::S32(buf) => {
                let frames = buf.frames();
                let ch_count = buf.spec().channels.count();
                (0..frames)
                    .map(|i| {
                        let sum: f32 = (0..ch_count)
                            .map(|ch| buf.chan(ch)[i] as f32 / 2147483648.0)
                            .sum();
                        sum / ch_count as f32
                    })
                    .collect()
            }
            AudioBufferRef::U8(buf) => {
                let frames = buf.frames();
                let ch_count = buf.spec().channels.count();
                (0..frames)
                    .map(|i| {
                        let sum: f32 = (0..ch_count)
                            .map(|ch| (buf.chan(ch)[i] as f32 - 128.0) / 128.0)
                            .sum();
                        sum / ch_count as f32
                    })
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        Self::new(path, FileAudioConfig::default())
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn control_rate(&self) -> f32 {
        self.sample_rate / self.config.hop_size as f32
    }

    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate
    }

    #[allow(dead_code)]
    pub fn position_secs(&self) -> f32 {
        self.position as f32 / self.sample_rate
    }

    pub fn progress(&self) -> f32 {
        if self.samples.is_empty() {
            return 1.0;
        }
        self.position as f32 / self.samples.len() as f32
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    #[allow(dead_code)]
    pub fn seek(&mut self, time_secs: f32) {
        let sample = (time_secs * self.sample_rate) as usize;
        self.position = sample.min(self.samples.len());
    }

    pub fn next_spectrum(&mut self) -> Option<Vec<f32>> {
        if self.position + self.config.window_size > self.samples.len() {
            return None;
        }

        let window_slice = &self.samples[self.position..self.position + self.config.window_size];

        let mut complex_buffer: Vec<Complex<f32>> = if self.config.apply_window {
            window_slice
                .iter()
                .zip(&self.window_coeffs)
                .map(|(&s, &w)| Complex { re: s * w, im: 0.0 })
                .collect()
        } else {
            window_slice
                .iter()
                .map(|&s| Complex { re: s, im: 0.0 })
                .collect()
        };

        let fft = self.fft_planner.plan_fft_forward(self.config.window_size);
        fft.process(&mut complex_buffer);

        self.position += self.config.hop_size;

        let n_bins = self.config.window_size / 2 + 1;
        let magnitude: Vec<f32> = complex_buffer
            .iter()
            .take(n_bins)
            .map(|c| c.norm())
            .collect();

        Some(magnitude)
    }

    pub fn next_power_spectrum(&mut self) -> Option<Vec<f32>> {
        self.next_spectrum()
            .map(|mag| mag.iter().map(|m| m * m).collect())
    }

    #[allow(dead_code)]
    pub fn process_all(&mut self) -> Vec<Vec<f32>> {
        self.reset();
        let mut spectrums = Vec::new();
        while let Some(spectrum) = self.next_spectrum() {
            spectrums.push(spectrum);
        }
        spectrums
    }

    #[allow(dead_code)]
    pub fn process_all_power(&mut self) -> Vec<Vec<f32>> {
        self.reset();
        let mut spectrums = Vec::new();
        while let Some(spectrum) = self.next_power_spectrum() {
            spectrums.push(spectrum);
        }
        spectrums
    }

    #[allow(dead_code)]
    pub fn current_rms(&self) -> f32 {
        if self.position + self.config.window_size > self.samples.len() {
            return 0.0;
        }
        let window = &self.samples[self.position..self.position + self.config.window_size];
        (window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32).sqrt()
    }

    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    pub fn window_size(&self) -> usize {
        self.config.window_size
    }
}

// =============================================================================
// Dataset Processor
// =============================================================================

pub struct DatasetProcessor {
    pub files: Vec<String>,
    pub current_index: usize,
}

impl DatasetProcessor {
    pub fn from_directory<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let mut files = Vec::new();

        for entry in WalkDir::new(&dir) {
            let entry = entry.map_err(|e| format!("Walk error: {e}"))?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if matches!(ext.as_str(), "wav" | "mp3" | "ogg" | "flac") {
                        files.push(path.to_string_lossy().to_string());
                    }
                }
            }
        }

        files.sort();
        println!(
            "  Found {} audio files in {}",
            files.len(),
            dir.as_ref().display()
        );

        Ok(Self {
            files,
            current_index: 0,
        })
    }

    pub fn next_file(&mut self, config: FileAudioConfig) -> Option<Result<FileAudioPump, String>> {
        if self.current_index >= self.files.len() {
            return None;
        }

        let path = &self.files[self.current_index];
        self.current_index += 1;

        Some(FileAudioPump::new(path, config))
    }

    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

// =============================================================================
// Helper Functions (from original audio_pump.rs)
// =============================================================================

/// Compute onset strength from consecutive power spectrums
pub fn compute_onset_strength(prev_spectrum: &[f32], curr_spectrum: &[f32]) -> f32 {
    if prev_spectrum.len() != curr_spectrum.len() {
        return 0.0;
    }

    let flux: f32 = curr_spectrum
        .iter()
        .zip(prev_spectrum)
        .map(|(curr, prev)| (curr - prev).max(0.0))
        .sum();

    flux / curr_spectrum.len() as f32
}

/// Compute spectral centroid (brightness indicator)
pub fn compute_spectral_centroid(spectrum: &[f32], sample_rate: f32, fft_size: usize) -> f32 {
    let total: f32 = spectrum.iter().sum();
    if total < 1e-10 {
        return 0.0;
    }

    let freq_per_bin = sample_rate / fft_size as f32;

    let weighted_sum: f32 = spectrum
        .iter()
        .enumerate()
        .map(|(i, &mag)| i as f32 * freq_per_bin * mag)
        .sum();

    weighted_sum / total
}

/// Compute RMS energy from time-domain samples
#[allow(dead_code)]
pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Compute spectral flatness (Wiener entropy)
pub fn compute_spectral_flatness(spectrum: &[f32]) -> f32 {
    if spectrum.is_empty() {
        return 0.0;
    }

    let n = spectrum.len() as f32;
    let epsilon = 1e-10;

    let log_sum: f32 = spectrum.iter().map(|&x| (x + epsilon).ln()).sum();
    let geometric_mean = (log_sum / n).exp();

    let arithmetic_mean = spectrum.iter().sum::<f32>() / n;

    (geometric_mean / (arithmetic_mean + epsilon))
        .min(1.0)
        .max(0.0)
}

/// Compute burst density from onset history (DEPRECATED - use compute_ioi_variance)
///
/// Measures the rate of significant energy bursts per second.
/// NOTE: This metric fails to discriminate clock from dog because both have
/// similar burst counts. Use IOI variance instead.
pub fn compute_burst_density(onset_history: &[f32], control_rate: f32) -> f32 {
    if onset_history.len() < 10 {
        return 0.5;
    }

    let mean: f32 = onset_history.iter().sum::<f32>() / onset_history.len() as f32;
    let variance: f32 = onset_history
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / onset_history.len() as f32;
    let std_dev = variance.sqrt();
    let threshold = mean + 0.5 * std_dev;

    let mut burst_count = 0;
    let mut in_burst = false;
    for &onset in onset_history {
        if onset > threshold && !in_burst {
            burst_count += 1;
            in_burst = true;
        } else if onset <= threshold * 0.8 {
            in_burst = false;
        }
    }

    let window_duration = onset_history.len() as f32 / control_rate;
    let bursts_per_second = burst_count as f32 / window_duration;
    (bursts_per_second / 8.0).min(1.0)
}

/// Compute Inter-Onset-Interval (IOI) Variance
///
/// Measures the regularity of rhythm by computing the coefficient of variation
/// of intervals between onset peaks.
///
/// Returns a value normalized to `[0, 1]` where:
/// - 0.0-0.2: Very regular/mechanical (clock ticks, metronome)
/// - 0.2-0.4: Moderately regular (rhythmic music)
/// - 0.4-0.6: Mixed regularity (speech, some birdsong)
/// - 0.6-0.8: Irregular (dog barking, random events)
/// - 0.8-1.0: Highly irregular/chaotic (noise, unpredictable)
///
/// Key insight: Clock ticks at t=0,1,2,3 have intervals `[1,1,1]` → variance ≈ 0.
///              Dog barks at t=0,0.2,0.9,1.5 have intervals `[0.2,0.7,0.6]` → variance >> 0.
pub fn compute_ioi_variance(onset_history: &[f32], _control_rate: f32) -> f32 {
    if onset_history.len() < 20 {
        return 0.5; // Default for insufficient data
    }

    // Calculate threshold for peak detection
    let mean: f32 = onset_history.iter().sum::<f32>() / onset_history.len() as f32;
    let variance: f32 = onset_history
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / onset_history.len() as f32;
    let std_dev = variance.sqrt();

    // Threshold: mean + 0.75 * std_dev (detect clear peaks)
    let threshold = mean + 0.75 * std_dev;

    // Find peak positions (frame indices where onset exceeds threshold)
    let mut peak_positions: Vec<usize> = Vec::new();
    let mut in_peak = false;
    for (i, &onset) in onset_history.iter().enumerate() {
        if onset > threshold && !in_peak {
            peak_positions.push(i);
            in_peak = true;
        } else if onset <= threshold * 0.7 {
            in_peak = false;
        }
    }

    // Need at least 3 peaks to compute meaningful IOI variance
    if peak_positions.len() < 3 {
        return 0.5; // Not enough peaks - return neutral
    }

    // Compute inter-onset intervals
    let intervals: Vec<f32> = peak_positions
        .windows(2)
        .map(|w| (w[1] - w[0]) as f32)
        .collect();

    if intervals.is_empty() {
        return 0.5;
    }

    // Compute coefficient of variation (CV = std / mean)
    let ioi_mean: f32 = intervals.iter().sum::<f32>() / intervals.len() as f32;
    if ioi_mean < 1e-6 {
        return 0.5;
    }

    let ioi_variance: f32 = intervals
        .iter()
        .map(|x| (x - ioi_mean).powi(2))
        .sum::<f32>()
        / intervals.len() as f32;
    let ioi_std = ioi_variance.sqrt();

    // Coefficient of variation: 0 = perfectly regular, higher = more irregular
    let cv = ioi_std / ioi_mean;

    // Normalize: CV of 0-2 → 0-1
    // CV of 0 means perfect regularity (clock), CV > 1 means very irregular
    (cv / 2.0).min(1.0)
}

/// Compute temporal regularity from onset strength history
pub fn compute_temporal_regularity(onset_history: &[f32]) -> f32 {
    if onset_history.len() < 20 {
        return 0.5;
    }

    let n = onset_history.len();
    let mean: f32 = onset_history.iter().sum::<f32>() / n as f32;
    let variance: f32 = onset_history
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>()
        / n as f32;

    if variance < 1e-10 {
        return 0.0;
    }

    let max_lag = n / 2;
    let mut max_autocorr = 0.0f32;

    for lag in 2..max_lag {
        let mut autocorr = 0.0f32;
        let mut count = 0;

        for i in 0..(n - lag) {
            autocorr += (onset_history[i] - mean) * (onset_history[i + lag] - mean);
            count += 1;
        }

        if count > 0 {
            autocorr /= count as f32 * variance;
            if autocorr > max_autocorr {
                max_autocorr = autocorr;
            }
        }
    }

    max_autocorr.max(0.0).min(1.0)
}

/// Apply simple mel-scale approximation to spectrum
pub fn spectrum_to_mel_bands(spectrum: &[f32], n_mels: usize, sample_rate: f32) -> Vec<f32> {
    let n_fft = (spectrum.len() - 1) * 2;
    let hz_per_bin = sample_rate / n_fft as f32;

    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(20.0);
    let mel_max = hz_to_mel(sample_rate / 2.0);

    let mel_edges: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_edges: Vec<f32> = mel_edges.iter().map(|&m| mel_to_hz(m)).collect();

    let bin_edges: Vec<usize> = hz_edges
        .iter()
        .map(|&hz| (hz / hz_per_bin).round() as usize)
        .collect();

    let mut mel_bands = vec![0.0f32; n_mels];

    for (i, band) in mel_bands.iter_mut().enumerate() {
        let start = bin_edges[i].min(spectrum.len() - 1);
        let end = bin_edges[i + 2].min(spectrum.len());
        let center = bin_edges[i + 1].min(spectrum.len() - 1);

        if start < end {
            for j in start..end {
                let weight = if j <= center {
                    if center > start {
                        (j - start) as f32 / (center - start) as f32
                    } else {
                        1.0
                    }
                } else if end > center {
                    (end - j) as f32 / (end - center) as f32
                } else {
                    1.0
                };
                *band += spectrum[j] * weight;
            }
        }
    }

    mel_bands.iter().map(|&e| (e.max(1e-10)).log10()).collect()
}

/// Result of processing a single file through the sentinel
#[derive(Clone)]
pub struct FileProcessingResult {
    pub file_path: String,
    pub duration_secs: f32,
    pub num_frames: usize,
    pub mean_rhythm_signature: [f32; 8],
    pub mean_timbre_signature: [f32; 8],
    pub dominant_rhythm_freq: f32,
    pub dominant_timbre_freq: f32,
    pub estimated_bpm: f32,
}

impl FileProcessingResult {
    pub fn print_summary(&self) {
        println!("  File: {}", self.file_path);
        println!(
            "    Duration: {:.1}s, {} frames",
            self.duration_secs, self.num_frames
        );
        println!("    Estimated BPM: {:.0}", self.estimated_bpm);
        print!("    Rhythm Sig: [");
        for (i, &v) in self.mean_rhythm_signature.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            if v > 0.05 {
                print!("\x1b[33m{v:.3}\x1b[0m");
            } else {
                print!("{v:.3}");
            }
        }
        println!("]");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onset_strength() {
        let prev = vec![0.1, 0.2, 0.3];
        let curr = vec![0.2, 0.1, 0.5];

        let onset = compute_onset_strength(&prev, &curr);

        assert!((onset - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_spectral_centroid() {
        let low_spec = vec![1.0, 0.5, 0.1, 0.0];
        let centroid_low = compute_spectral_centroid(&low_spec, 44100.0, 8);

        let high_spec = vec![0.0, 0.1, 0.5, 1.0];
        let centroid_high = compute_spectral_centroid(&high_spec, 44100.0, 8);

        assert!(centroid_high > centroid_low);
    }

    #[test]
    fn test_mel_bands() {
        let spectrum = vec![0.1; 1025];
        let mel_bands = spectrum_to_mel_bands(&spectrum, 40, 44100.0);

        assert_eq!(mel_bands.len(), 40);
        assert!(mel_bands.iter().all(|&b| b.is_finite()));
    }

    #[test]
    fn test_file_audio_config_default() {
        let config = FileAudioConfig::default();
        assert_eq!(config.window_size, 2048);
        assert_eq!(config.hop_size, 1024);
        assert!(config.apply_window);
    }
}
