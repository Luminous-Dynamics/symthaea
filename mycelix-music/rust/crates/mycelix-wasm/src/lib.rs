// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix WebAssembly Module
//!
//! Browser-side audio processing capabilities including:
//! - Real-time waveform generation
//! - Audio analysis (loudness, BPM, frequency)
//! - Audio buffer manipulation
//! - Visualization data generation
//! - Real-time audio effects (EQ, reverb, spatial, dynamics)

pub mod effects;

pub use effects::*;

use wasm_bindgen::prelude::*;
use web_sys::{AudioContext, AnalyserNode, CanvasRenderingContext2d, HtmlCanvasElement};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

#[cfg(feature = "console_error_panic_hook")]
pub use console_error_panic_hook::set_once as set_panic_hook;

// ============================================================================
// Initialization
// ============================================================================

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ============================================================================
// Core Types
// ============================================================================

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationType {
    Waveform,
    FrequencyBars,
    FrequencyLine,
    Circular,
    Spectrogram,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub viz_type: VisualizationType,
    pub fft_size: u32,
    pub smoothing: f32,
    pub bar_width: f32,
    pub bar_gap: f32,
    pub mirror: bool,
    pub gradient: bool,
}

#[wasm_bindgen]
impl VisualizationConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen]
    pub fn with_type(mut self, viz_type: VisualizationType) -> Self {
        self.viz_type = viz_type;
        self
    }

    #[wasm_bindgen]
    pub fn with_fft_size(mut self, fft_size: u32) -> Self {
        self.fft_size = fft_size;
        self
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            viz_type: VisualizationType::Waveform,
            fft_size: 2048,
            smoothing: 0.8,
            bar_width: 3.0,
            bar_gap: 1.0,
            mirror: false,
            gradient: true,
        }
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveformData {
    peaks: Vec<f32>,
    rms: Vec<f32>,
    duration_ms: f64,
    sample_rate: u32,
    channels: u32,
}

#[wasm_bindgen]
impl WaveformData {
    #[wasm_bindgen(getter)]
    pub fn peaks(&self) -> Vec<f32> {
        self.peaks.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn rms(&self) -> Vec<f32> {
        self.rms.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u32 {
        self.channels
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioAnalysisResult {
    loudness_lufs: f32,
    peak_db: f32,
    dynamic_range: f32,
    estimated_bpm: Option<f32>,
    spectral_centroid: f32,
    zero_crossing_rate: f32,
}

#[wasm_bindgen]
impl AudioAnalysisResult {
    #[wasm_bindgen(getter)]
    pub fn loudness_lufs(&self) -> f32 {
        self.loudness_lufs
    }

    #[wasm_bindgen(getter)]
    pub fn peak_db(&self) -> f32 {
        self.peak_db
    }

    #[wasm_bindgen(getter)]
    pub fn dynamic_range(&self) -> f32 {
        self.dynamic_range
    }

    #[wasm_bindgen(getter)]
    pub fn estimated_bpm(&self) -> Option<f32> {
        self.estimated_bpm
    }

    #[wasm_bindgen(getter)]
    pub fn spectral_centroid(&self) -> f32 {
        self.spectral_centroid
    }

    #[wasm_bindgen(getter)]
    pub fn zero_crossing_rate(&self) -> f32 {
        self.zero_crossing_rate
    }

    #[wasm_bindgen]
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(self)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ============================================================================
// Waveform Generator
// ============================================================================

#[wasm_bindgen]
pub struct WaveformGenerator {
    samples_per_peak: usize,
}

#[wasm_bindgen]
impl WaveformGenerator {
    #[wasm_bindgen(constructor)]
    pub fn new(target_peaks: usize) -> Self {
        Self {
            samples_per_peak: target_peaks.max(100),
        }
    }

    #[wasm_bindgen]
    pub fn generate(&self, audio_data: &[f32], sample_rate: u32, channels: u32) -> WaveformData {
        let samples_per_channel = audio_data.len() / channels as usize;
        let samples_per_peak = (samples_per_channel / self.samples_per_peak).max(1);

        let mut peaks = Vec::new();
        let mut rms_values = Vec::new();

        // Mix to mono if stereo
        let mono_data: Vec<f32> = if channels == 2 {
            audio_data
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
                .collect()
        } else {
            audio_data.to_vec()
        };

        // Generate peaks and RMS
        for chunk in mono_data.chunks(samples_per_peak) {
            let peak = chunk.iter()
                .map(|s| s.abs())
                .fold(0.0f32, f32::max);

            let rms = (chunk.iter()
                .map(|s| s * s)
                .sum::<f32>() / chunk.len() as f32)
                .sqrt();

            peaks.push(peak);
            rms_values.push(rms);
        }

        let duration_ms = (samples_per_channel as f64 / sample_rate as f64) * 1000.0;

        WaveformData {
            peaks,
            rms: rms_values,
            duration_ms,
            sample_rate,
            channels,
        }
    }

    #[wasm_bindgen]
    pub fn generate_from_audio_buffer(&self, buffer: &web_sys::AudioBuffer) -> Result<WaveformData, JsValue> {
        let channels = buffer.number_of_channels();
        let sample_rate = buffer.sample_rate() as u32;
        let length = buffer.length() as usize;

        let mut audio_data = Vec::with_capacity(length * channels as usize);

        for c in 0..channels {
            let mut channel_data = vec![0.0f32; length];
            buffer.copy_from_channel(&mut channel_data, c as i32)?;
            audio_data.extend(channel_data);
        }

        Ok(self.generate(&audio_data, sample_rate, channels))
    }
}

// ============================================================================
// Audio Analyzer
// ============================================================================

#[wasm_bindgen]
pub struct AudioAnalyzer {
    fft_size: usize,
}

#[wasm_bindgen]
impl AudioAnalyzer {
    #[wasm_bindgen(constructor)]
    pub fn new(fft_size: usize) -> Self {
        let fft_size = fft_size.next_power_of_two().min(32768);
        Self { fft_size }
    }

    #[wasm_bindgen]
    pub fn analyze(&self, audio_data: &[f32], sample_rate: u32) -> AudioAnalysisResult {
        let peak = audio_data.iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        let peak_db = if peak > 0.0 { 20.0 * peak.log10() } else { -96.0 };

        // Calculate RMS for loudness estimation
        let rms = (audio_data.iter()
            .map(|s| s * s)
            .sum::<f32>() / audio_data.len() as f32)
            .sqrt();

        // Rough LUFS estimation (simplified, actual LUFS requires K-weighting)
        let loudness_lufs = if rms > 0.0 {
            -0.691 + 10.0 * (rms * rms).log10()
        } else {
            -70.0
        };

        // Dynamic range estimation
        let dynamic_range = self.calculate_dynamic_range(audio_data);

        // Spectral centroid
        let spectral_centroid = self.calculate_spectral_centroid(audio_data, sample_rate);

        // Zero crossing rate
        let zero_crossing_rate = self.calculate_zcr(audio_data, sample_rate);

        // BPM estimation (simplified onset detection)
        let estimated_bpm = self.estimate_bpm(audio_data, sample_rate);

        AudioAnalysisResult {
            loudness_lufs,
            peak_db,
            dynamic_range,
            estimated_bpm,
            spectral_centroid,
            zero_crossing_rate,
        }
    }

    fn calculate_dynamic_range(&self, audio_data: &[f32]) -> f32 {
        let window_size = 4410; // ~100ms at 44.1kHz
        let mut loudness_values: Vec<f32> = Vec::new();

        for chunk in audio_data.chunks(window_size) {
            let rms = (chunk.iter()
                .map(|s| s * s)
                .sum::<f32>() / chunk.len() as f32)
                .sqrt();

            if rms > 0.0001 { // Gate silence
                loudness_values.push(20.0 * rms.log10());
            }
        }

        if loudness_values.len() < 2 {
            return 0.0;
        }

        loudness_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p10_idx = loudness_values.len() / 10;
        let p90_idx = loudness_values.len() * 9 / 10;

        loudness_values[p90_idx] - loudness_values[p10_idx]
    }

    fn calculate_spectral_centroid(&self, audio_data: &[f32], sample_rate: u32) -> f32 {
        // Simplified DFT for spectral centroid
        let n = self.fft_size.min(audio_data.len());
        let data: Vec<f32> = audio_data.iter().take(n).copied().collect();

        let mut magnitude_sum = 0.0f32;
        let mut weighted_sum = 0.0f32;

        for k in 1..n/2 {
            let freq = k as f32 * sample_rate as f32 / n as f32;

            // Simple DFT bin calculation
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (i, sample) in data.iter().enumerate() {
                let angle = -2.0 * PI * k as f32 * i as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            let magnitude = (real * real + imag * imag).sqrt();
            magnitude_sum += magnitude;
            weighted_sum += freq * magnitude;
        }

        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }

    fn calculate_zcr(&self, audio_data: &[f32], sample_rate: u32) -> f32 {
        let mut crossings = 0;

        for window in audio_data.windows(2) {
            if (window[0] >= 0.0) != (window[1] >= 0.0) {
                crossings += 1;
            }
        }

        let duration_sec = audio_data.len() as f32 / sample_rate as f32;
        crossings as f32 / duration_sec
    }

    fn estimate_bpm(&self, audio_data: &[f32], sample_rate: u32) -> Option<f32> {
        if audio_data.len() < sample_rate as usize * 4 {
            return None; // Need at least 4 seconds
        }

        // Onset detection using energy difference
        let hop_size = sample_rate as usize / 20; // 50ms windows
        let mut onsets = Vec::new();
        let mut prev_energy = 0.0f32;

        for (i, chunk) in audio_data.chunks(hop_size).enumerate() {
            let energy: f32 = chunk.iter().map(|s| s * s).sum();
            let diff = energy - prev_energy;

            if diff > prev_energy * 0.5 && energy > 0.001 {
                onsets.push(i);
            }

            prev_energy = energy;
        }

        if onsets.len() < 4 {
            return None;
        }

        // Calculate inter-onset intervals
        let intervals: Vec<usize> = onsets.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        // Find most common interval
        let mut interval_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &interval in &intervals {
            // Quantize to 10ms bins
            let quantized = (interval / 2) * 2;
            *interval_counts.entry(quantized).or_insert(0) += 1;
        }

        let (best_interval, _) = interval_counts.into_iter()
            .max_by_key(|(_, count)| *count)?;

        let seconds_per_beat = best_interval as f32 * hop_size as f32 / sample_rate as f32;
        let bpm = 60.0 / seconds_per_beat;

        // Sanity check BPM range
        if bpm >= 60.0 && bpm <= 200.0 {
            Some(bpm)
        } else if bpm >= 30.0 && bpm < 60.0 {
            Some(bpm * 2.0) // Double time
        } else if bpm > 200.0 && bpm <= 400.0 {
            Some(bpm / 2.0) // Half time
        } else {
            None
        }
    }

    #[wasm_bindgen]
    pub fn analyze_audio_buffer(&self, buffer: &web_sys::AudioBuffer) -> Result<AudioAnalysisResult, JsValue> {
        let length = buffer.length() as usize;
        let sample_rate = buffer.sample_rate() as u32;
        let channels = buffer.number_of_channels();

        // Mix to mono
        let mut mono_data = vec![0.0f32; length];

        for c in 0..channels {
            let mut channel_data = vec![0.0f32; length];
            buffer.copy_from_channel(&mut channel_data, c as i32)?;

            for (i, sample) in channel_data.iter().enumerate() {
                mono_data[i] += sample / channels as f32;
            }
        }

        Ok(self.analyze(&mono_data, sample_rate))
    }
}

// ============================================================================
// Real-time Visualizer
// ============================================================================

#[wasm_bindgen]
pub struct RealtimeVisualizer {
    config: VisualizationConfig,
    history: Vec<Vec<u8>>,
    history_length: usize,
    primary_color: String,
    secondary_color: String,
    background_color: String,
}

#[wasm_bindgen]
impl RealtimeVisualizer {
    #[wasm_bindgen(constructor)]
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            history_length: 100,
            primary_color: "#818cf8".to_string(),   // Indigo
            secondary_color: "#c084fc".to_string(), // Purple
            background_color: "#0f0f0f".to_string(),
        }
    }

    #[wasm_bindgen]
    pub fn set_colors(&mut self, primary: &str, secondary: &str, background: &str) {
        self.primary_color = primary.to_string();
        self.secondary_color = secondary.to_string();
        self.background_color = background.to_string();
    }

    #[wasm_bindgen]
    pub fn draw_waveform(
        &self,
        ctx: &CanvasRenderingContext2d,
        data: &[f32],
        width: f64,
        height: f64,
    ) -> Result<(), JsValue> {
        // Clear canvas
        ctx.set_fill_style_str(&self.background_color);
        ctx.fill_rect(0.0, 0.0, width, height);

        if data.is_empty() {
            return Ok(());
        }

        let mid_y = height / 2.0;
        let step = width / data.len() as f64;

        // Draw waveform
        ctx.begin_path();
        ctx.set_stroke_style_str(&self.primary_color);
        ctx.set_line_width(2.0);

        for (i, &sample) in data.iter().enumerate() {
            let x = i as f64 * step;
            let y = mid_y - (sample as f64 * mid_y * 0.9);

            if i == 0 {
                ctx.move_to(x, y);
            } else {
                ctx.line_to(x, y);
            }
        }

        ctx.stroke();

        // Mirror if enabled
        if self.config.mirror {
            ctx.begin_path();
            ctx.set_stroke_style_str(&self.secondary_color);
            ctx.set_global_alpha(0.5);

            for (i, &sample) in data.iter().enumerate() {
                let x = i as f64 * step;
                let y = mid_y + (sample as f64 * mid_y * 0.9);

                if i == 0 {
                    ctx.move_to(x, y);
                } else {
                    ctx.line_to(x, y);
                }
            }

            ctx.stroke();
            ctx.set_global_alpha(1.0);
        }

        Ok(())
    }

    #[wasm_bindgen]
    pub fn draw_frequency_bars(
        &self,
        ctx: &CanvasRenderingContext2d,
        frequency_data: &[u8],
        width: f64,
        height: f64,
    ) -> Result<(), JsValue> {
        // Clear canvas
        ctx.set_fill_style_str(&self.background_color);
        ctx.fill_rect(0.0, 0.0, width, height);

        if frequency_data.is_empty() {
            return Ok(());
        }

        let bar_count = (width / (self.config.bar_width as f64 + self.config.bar_gap as f64)) as usize;
        let bars_to_use = bar_count.min(frequency_data.len());
        let samples_per_bar = frequency_data.len() / bars_to_use;

        for i in 0..bars_to_use {
            // Average frequency values for this bar
            let start = i * samples_per_bar;
            let end = (start + samples_per_bar).min(frequency_data.len());
            let avg: f32 = frequency_data[start..end].iter()
                .map(|&v| v as f32)
                .sum::<f32>() / samples_per_bar as f32;

            let bar_height = (avg / 255.0) as f64 * height * 0.9;
            let x = i as f64 * (self.config.bar_width as f64 + self.config.bar_gap as f64);
            let y = height - bar_height;

            if self.config.gradient {
                // Create gradient effect based on height
                let hue = 250.0 + (avg / 255.0) * 60.0; // Indigo to purple range
                let color = format!("hsl({}, 80%, {}%)", hue, 50.0 + (avg / 255.0) * 30.0);
                ctx.set_fill_style_str(&color);
            } else {
                ctx.set_fill_style_str(&self.primary_color);
            }

            ctx.fill_rect(x, y, self.config.bar_width as f64, bar_height);
        }

        Ok(())
    }

    #[wasm_bindgen]
    pub fn draw_circular(
        &self,
        ctx: &CanvasRenderingContext2d,
        frequency_data: &[u8],
        width: f64,
        height: f64,
    ) -> Result<(), JsValue> {
        // Clear canvas
        ctx.set_fill_style_str(&self.background_color);
        ctx.fill_rect(0.0, 0.0, width, height);

        if frequency_data.is_empty() {
            return Ok(());
        }

        let center_x = width / 2.0;
        let center_y = height / 2.0;
        let base_radius = width.min(height) / 4.0;

        ctx.begin_path();
        ctx.set_stroke_style_str(&self.primary_color);
        ctx.set_line_width(2.0);

        let points = frequency_data.len().min(360);
        let angle_step = (2.0 * std::f64::consts::PI) / points as f64;

        for (i, &value) in frequency_data.iter().take(points).enumerate() {
            let angle = i as f64 * angle_step - std::f64::consts::PI / 2.0;
            let radius = base_radius + (value as f64 / 255.0) * base_radius;

            let x = center_x + angle.cos() * radius;
            let y = center_y + angle.sin() * radius;

            if i == 0 {
                ctx.move_to(x, y);
            } else {
                ctx.line_to(x, y);
            }
        }

        ctx.close_path();
        ctx.stroke();

        // Draw inner circle
        ctx.begin_path();
        ctx.set_stroke_style_str(&self.secondary_color);
        ctx.set_global_alpha(0.3);
        ctx.arc(center_x, center_y, base_radius * 0.5, 0.0, 2.0 * std::f64::consts::PI)?;
        ctx.stroke();
        ctx.set_global_alpha(1.0);

        Ok(())
    }

    #[wasm_bindgen]
    pub fn draw_spectrogram(
        &mut self,
        ctx: &CanvasRenderingContext2d,
        frequency_data: &[u8],
        width: f64,
        height: f64,
    ) -> Result<(), JsValue> {
        // Add current frame to history
        self.history.push(frequency_data.to_vec());
        if self.history.len() > self.history_length {
            self.history.remove(0);
        }

        // Clear canvas
        ctx.set_fill_style_str(&self.background_color);
        ctx.fill_rect(0.0, 0.0, width, height);

        let column_width = width / self.history_length as f64;
        let row_height = height / 128.0; // Use first 128 frequency bins

        for (col, frame) in self.history.iter().enumerate() {
            for (row, &value) in frame.iter().take(128).enumerate() {
                let intensity = value as f64 / 255.0;

                // Color mapping: dark blue -> cyan -> yellow -> red
                let (r, g, b) = if intensity < 0.25 {
                    let t = intensity / 0.25;
                    (0.0, t * 0.3, 0.3 + t * 0.3)
                } else if intensity < 0.5 {
                    let t = (intensity - 0.25) / 0.25;
                    (0.0, 0.3 + t * 0.7, 0.6 - t * 0.2)
                } else if intensity < 0.75 {
                    let t = (intensity - 0.5) / 0.25;
                    (t, 1.0, 0.4 - t * 0.4)
                } else {
                    let t = (intensity - 0.75) / 0.25;
                    (1.0, 1.0 - t * 0.5, 0.0)
                };

                let color = format!(
                    "rgb({}, {}, {})",
                    (r * 255.0) as u8,
                    (g * 255.0) as u8,
                    (b * 255.0) as u8
                );

                ctx.set_fill_style_str(&color);
                ctx.fill_rect(
                    col as f64 * column_width,
                    height - (row + 1) as f64 * row_height,
                    column_width + 1.0,
                    row_height + 1.0,
                );
            }
        }

        Ok(())
    }

    #[wasm_bindgen]
    pub fn draw(&mut self, ctx: &CanvasRenderingContext2d, analyser: &AnalyserNode) -> Result<(), JsValue> {
        let canvas = ctx.canvas().ok_or_else(|| JsValue::from_str("No canvas"))?;
        let width = canvas.width() as f64;
        let height = canvas.height() as f64;

        match self.config.viz_type {
            VisualizationType::Waveform => {
                let mut time_data = vec![0u8; analyser.fft_size() as usize];
                analyser.get_byte_time_domain_data(&mut time_data);

                // Convert to float
                let float_data: Vec<f32> = time_data.iter()
                    .map(|&v| (v as f32 - 128.0) / 128.0)
                    .collect();

                self.draw_waveform(ctx, &float_data, width, height)
            }
            VisualizationType::FrequencyBars => {
                let mut freq_data = vec![0u8; analyser.frequency_bin_count() as usize];
                analyser.get_byte_frequency_data(&mut freq_data);
                self.draw_frequency_bars(ctx, &freq_data, width, height)
            }
            VisualizationType::FrequencyLine => {
                let mut freq_data = vec![0u8; analyser.frequency_bin_count() as usize];
                analyser.get_byte_frequency_data(&mut freq_data);

                // Convert to float for waveform rendering
                let float_data: Vec<f32> = freq_data.iter()
                    .map(|&v| v as f32 / 255.0)
                    .collect();

                self.draw_waveform(ctx, &float_data, width, height)
            }
            VisualizationType::Circular => {
                let mut freq_data = vec![0u8; analyser.frequency_bin_count() as usize];
                analyser.get_byte_frequency_data(&mut freq_data);
                self.draw_circular(ctx, &freq_data, width, height)
            }
            VisualizationType::Spectrogram => {
                let mut freq_data = vec![0u8; analyser.frequency_bin_count() as usize];
                analyser.get_byte_frequency_data(&mut freq_data);
                self.draw_spectrogram(ctx, &freq_data, width, height)
            }
        }
    }
}

// ============================================================================
// Audio Buffer Utilities
// ============================================================================

#[wasm_bindgen]
pub struct AudioBufferUtils;

#[wasm_bindgen]
impl AudioBufferUtils {
    #[wasm_bindgen]
    pub fn normalize(audio_data: &mut [f32], target_peak: f32) {
        let current_peak = audio_data.iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        if current_peak > 0.0 {
            let gain = target_peak / current_peak;
            for sample in audio_data.iter_mut() {
                *sample *= gain;
            }
        }
    }

    #[wasm_bindgen]
    pub fn fade_in(audio_data: &mut [f32], fade_samples: usize) {
        let fade_len = fade_samples.min(audio_data.len());
        for i in 0..fade_len {
            let gain = i as f32 / fade_len as f32;
            audio_data[i] *= gain;
        }
    }

    #[wasm_bindgen]
    pub fn fade_out(audio_data: &mut [f32], fade_samples: usize) {
        let len = audio_data.len();
        let fade_len = fade_samples.min(len);
        for i in 0..fade_len {
            let gain = 1.0 - (i as f32 / fade_len as f32);
            audio_data[len - fade_len + i] *= gain;
        }
    }

    #[wasm_bindgen]
    pub fn mix_stereo_to_mono(left: &[f32], right: &[f32]) -> Vec<f32> {
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| (l + r) / 2.0)
            .collect()
    }

    #[wasm_bindgen]
    pub fn apply_gain(audio_data: &mut [f32], gain_db: f32) {
        let linear_gain = 10.0f32.powf(gain_db / 20.0);
        for sample in audio_data.iter_mut() {
            *sample *= linear_gain;
        }
    }

    #[wasm_bindgen]
    pub fn detect_silence(audio_data: &[f32], threshold_db: f32) -> Vec<u32> {
        let threshold = 10.0f32.powf(threshold_db / 20.0);
        let window_size = 1024;
        let mut silence_regions = Vec::new();
        let mut in_silence = false;
        let mut silence_start = 0u32;

        for (i, chunk) in audio_data.chunks(window_size).enumerate() {
            let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
            let is_silent = rms < threshold;

            if is_silent && !in_silence {
                silence_start = (i * window_size) as u32;
                in_silence = true;
            } else if !is_silent && in_silence {
                silence_regions.push(silence_start);
                silence_regions.push((i * window_size) as u32);
                in_silence = false;
            }
        }

        if in_silence {
            silence_regions.push(silence_start);
            silence_regions.push(audio_data.len() as u32);
        }

        silence_regions
    }

    #[wasm_bindgen]
    pub fn compute_rms(audio_data: &[f32]) -> f32 {
        (audio_data.iter().map(|s| s * s).sum::<f32>() / audio_data.len() as f32).sqrt()
    }

    #[wasm_bindgen]
    pub fn compute_peak(audio_data: &[f32]) -> f32 {
        audio_data.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
    }
}

// ============================================================================
// Audio Worklet Processor (data preparation)
// ============================================================================

#[wasm_bindgen]
pub struct WorkletProcessor {
    buffer_size: usize,
    sample_rate: f32,
    ring_buffer: Vec<f32>,
    write_pos: usize,
}

#[wasm_bindgen]
impl WorkletProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(buffer_size: usize, sample_rate: f32) -> Self {
        Self {
            buffer_size,
            sample_rate,
            ring_buffer: vec![0.0; buffer_size * 4], // 4x buffer for safety
            write_pos: 0,
        }
    }

    #[wasm_bindgen]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        // Copy input to ring buffer
        for &sample in input {
            self.ring_buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.ring_buffer.len();
        }

        // Return processed chunk (passthrough for now, can add effects)
        input.to_vec()
    }

    #[wasm_bindgen]
    pub fn get_analysis_buffer(&self) -> Vec<f32> {
        // Return the last buffer_size samples for analysis
        let start = if self.write_pos >= self.buffer_size {
            self.write_pos - self.buffer_size
        } else {
            self.ring_buffer.len() - (self.buffer_size - self.write_pos)
        };

        let mut result = Vec::with_capacity(self.buffer_size);
        for i in 0..self.buffer_size {
            let idx = (start + i) % self.ring_buffer.len();
            result.push(self.ring_buffer[idx]);
        }
        result
    }

    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    #[wasm_bindgen(getter)]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

#[wasm_bindgen]
pub fn samples_to_time(samples: u32, sample_rate: u32) -> f64 {
    samples as f64 / sample_rate as f64
}

#[wasm_bindgen]
pub fn time_to_samples(time_seconds: f64, sample_rate: u32) -> u32 {
    (time_seconds * sample_rate as f64) as u32
}

#[wasm_bindgen]
pub fn db_to_linear(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

#[wasm_bindgen]
pub fn linear_to_db(linear: f32) -> f32 {
    if linear > 0.0 {
        20.0 * linear.log10()
    } else {
        -96.0
    }
}

#[wasm_bindgen]
pub fn frequency_to_midi(frequency: f32) -> f32 {
    69.0 + 12.0 * (frequency / 440.0).log2()
}

#[wasm_bindgen]
pub fn midi_to_frequency(midi_note: f32) -> f32 {
    440.0 * 2.0f32.powf((midi_note - 69.0) / 12.0)
}

#[wasm_bindgen]
pub fn bpm_to_ms(bpm: f32) -> f32 {
    60000.0 / bpm
}

#[wasm_bindgen]
pub fn ms_to_samples(ms: f32, sample_rate: u32) -> u32 {
    (ms * sample_rate as f32 / 1000.0) as u32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waveform_generation() {
        let generator = WaveformGenerator::new(100);
        let audio_data: Vec<f32> = (0..44100)
            .map(|i| (i as f32 / 44100.0 * 440.0 * 2.0 * PI).sin())
            .collect();

        let waveform = generator.generate(&audio_data, 44100, 1);
        assert!(!waveform.peaks.is_empty());
        assert_eq!(waveform.sample_rate, 44100);
    }

    #[test]
    fn test_audio_analysis() {
        let analyzer = AudioAnalyzer::new(2048);
        let audio_data: Vec<f32> = (0..44100)
            .map(|i| (i as f32 / 44100.0 * 440.0 * 2.0 * PI).sin() * 0.5)
            .collect();

        let result = analyzer.analyze(&audio_data, 44100);
        assert!(result.peak_db < 0.0);
        assert!(result.spectral_centroid > 0.0);
    }

    #[test]
    fn test_conversions() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 0.001);
        assert!((db_to_linear(-6.0) - 0.5).abs() < 0.01);
        assert!((linear_to_db(1.0) - 0.0).abs() < 0.001);

        assert!((frequency_to_midi(440.0) - 69.0).abs() < 0.001);
        assert!((midi_to_frequency(69.0) - 440.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize() {
        let mut data = vec![0.5, -0.3, 0.8, -0.2];
        AudioBufferUtils::normalize(&mut data, 1.0);
        assert!((data.iter().map(|s| s.abs()).fold(0.0f32, f32::max) - 1.0).abs() < 0.001);
    }
}
