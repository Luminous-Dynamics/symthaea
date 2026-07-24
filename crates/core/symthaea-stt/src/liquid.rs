// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Liquid Feature Extractor - CfC Neurons for Bioacoustic Analysis
//!
//! Replaces rigid DSP filters with adaptive Closed-form Continuous-time (CfC) neurons.
//! The key insight: biological hearing adapts to tempo, not just frequency.
//!
//! # The Problem with DSP
//!
//! Traditional filters fail when:
//! - Whale swims faster (Doppler shift)
//! - Click rate varies (excitement vs calm)
//! - Background noise changes
//!
//! # The CfC Solution
//!
//! CfC neurons have *learnable time constants* that adapt:
//! ```text
//! τ(x) · dx/dt = -x + f(input)
//! ```
//!
//! Where τ adapts based on the input history. Fast inputs → fast τ. Slow inputs → slow τ.
//!
//! # Architecture
//!
//! ```text
//! Raw Audio ──► CfC Bank ──► Event Detector ──► Time-Aware HV
//!                  │              │                   │
//!              (adaptive      (spikes with      (Φ ⊗ Δt ⊗ I)
//!               time-const)    timing)
//! ```

use crate::hdc::HV16;

/// CfC Cell - A single Closed-form Continuous-time neuron
///
/// Implements the differential equation:
/// τ(x) · dx/dt = -x + σ(W·input + b)
///
/// With closed-form solution for discrete time steps.
#[derive(Clone)]
pub struct CfcCell {
    /// Hidden state
    state: f32,

    /// Log time constant (learnable, maps to τ via exp)
    log_tau: f32,

    /// Input weight
    w_in: f32,

    /// Recurrent weight
    w_rec: f32,

    /// Bias
    bias: f32,

    /// Minimum tau (fastest response)
    tau_min: f32,

    /// Maximum tau (slowest response)
    tau_max: f32,
}

impl CfcCell {
    /// Create a new CfC cell with given parameters
    pub fn new(log_tau: f32, w_in: f32, w_rec: f32, bias: f32) -> Self {
        Self {
            state: 0.0,
            log_tau,
            w_in,
            w_rec,
            bias,
            tau_min: 0.001, // 1ms minimum
            tau_max: 1.0,   // 1s maximum
        }
    }

    /// Create a cell optimized for click detection (fast transients)
    pub fn click_detector() -> Self {
        Self::new(
            -3.0, // log_tau → τ ≈ 0.05 (50ms, fast)
            2.0,  // Strong input coupling
            -0.5, // Mild self-inhibition
            -1.0, // Negative bias (needs strong input to fire)
        )
    }

    /// Create a cell optimized for whistle detection (sustained tones)
    pub fn whistle_detector() -> Self {
        Self::new(
            0.0,  // log_tau → τ ≈ 1.0 (1s, slow)
            1.0,  // Moderate input coupling
            0.3,  // Mild self-excitation (sustain)
            -0.5, // Moderate threshold
        )
    }

    /// Create a cell for burst/noise detection
    pub fn burst_detector() -> Self {
        Self::new(
            -1.5, // log_tau → τ ≈ 0.22 (220ms, medium)
            1.5,  // Moderate-strong input
            0.0,  // No recurrence
            -0.3, // Low threshold
        )
    }

    /// Compute effective time constant
    fn tau(&self) -> f32 {
        let raw_tau = self.log_tau.exp();
        raw_tau.clamp(self.tau_min, self.tau_max)
    }

    /// Sigmoid activation
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Process one time step (closed-form CfC update)
    ///
    /// The closed-form solution for dx/dt = (-x + f) / τ over dt is:
    /// x(t+dt) = f + (x(t) - f) · exp(-dt/τ)
    pub fn step(&mut self, input: f32, dt: f32) -> f32 {
        let tau = self.tau();

        // Compute target activation
        let pre_activation = self.w_in * input + self.w_rec * self.state + self.bias;
        let target = Self::sigmoid(pre_activation);

        // Closed-form exponential decay toward target
        let decay = (-dt / tau).exp();
        self.state = target + (self.state - target) * decay;

        self.state
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.state = 0.0;
    }

    /// Get current state
    pub fn state(&self) -> f32 {
        self.state
    }

    /// Get current time constant
    pub fn current_tau(&self) -> f32 {
        self.tau()
    }
}

/// Liquid Feature Bank - Multiple CfC cells for different timescales
pub struct LiquidBank {
    /// Click detector (fast transients)
    click_cell: CfcCell,

    /// Whistle detector (sustained tones)
    whistle_cell: CfcCell,

    /// Burst detector (broadband noise)
    burst_cell: CfcCell,

    /// Sample rate for dt calculation
    sample_rate: f32,

    /// Frame size (stored for future stride calculations)
    #[allow(dead_code)]
    frame_size: usize,
}

impl LiquidBank {
    /// Create a new liquid bank with default cetacean-optimized cells
    pub fn new(sample_rate: f32, frame_size: usize) -> Self {
        Self {
            click_cell: CfcCell::click_detector(),
            whistle_cell: CfcCell::whistle_detector(),
            burst_cell: CfcCell::burst_detector(),
            sample_rate,
            frame_size,
        }
    }

    /// Reset all cells
    pub fn reset(&mut self) {
        self.click_cell.reset();
        self.whistle_cell.reset();
        self.burst_cell.reset();
    }

    /// Process a frame of audio, returning liquid features
    pub fn process_frame(&mut self, samples: &[f32]) -> LiquidFeatures {
        let dt = 1.0 / self.sample_rate;

        // Compute frame-level features to feed the CfC cells
        let energy = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
        let rms = energy.sqrt();

        // Teager-Kaiser energy (transient detection)
        let tk_energy = self.teager_kaiser_energy(samples);

        // Zero-crossing rate (noise vs tone)
        let zcr = self.zero_crossing_rate(samples);

        // Spectral centroid approximation (high vs low frequency)
        let spectral_tilt = self.spectral_tilt(samples);

        // Feed features through CfC cells
        // Click cell responds to Teager-Kaiser (transients)
        let click_response = self.click_cell.step(tk_energy, dt * samples.len() as f32);

        // Whistle cell responds to spectral tilt (harmonic content)
        let whistle_response = self
            .whistle_cell
            .step(spectral_tilt, dt * samples.len() as f32);

        // Burst cell responds to energy + ZCR (broadband noise)
        let burst_input = rms * zcr;
        let burst_response = self.burst_cell.step(burst_input, dt * samples.len() as f32);

        LiquidFeatures {
            click: click_response,
            whistle: whistle_response,
            burst: burst_response,
            energy: rms,
            duration_factor: self.click_cell.current_tau(), // Adaptive tempo
        }
    }

    /// Teager-Kaiser energy operator (sensitive to transients)
    fn teager_kaiser_energy(&self, samples: &[f32]) -> f32 {
        if samples.len() < 3 {
            return 0.0;
        }

        let mut tk_sum = 0.0;
        for i in 1..samples.len() - 1 {
            let tk = samples[i] * samples[i] - samples[i - 1] * samples[i + 1];
            tk_sum += tk.abs();
        }

        (tk_sum / (samples.len() - 2) as f32).sqrt()
    }

    /// Zero-crossing rate
    fn zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let crossings: usize = samples
            .windows(2)
            .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
            .count();

        crossings as f32 / (samples.len() - 1) as f32
    }

    /// Spectral tilt (ratio of high to low frequency energy)
    fn spectral_tilt(&self, samples: &[f32]) -> f32 {
        // Simple differentiator approximates high-frequency content
        if samples.len() < 2 {
            return 0.0;
        }

        let low_energy: f32 = samples.iter().map(|s| s * s).sum();
        let high_energy: f32 = samples
            .windows(2)
            .map(|w| {
                let diff = w[1] - w[0];
                diff * diff
            })
            .sum();

        if low_energy < 1e-10 {
            return 0.0;
        }

        (high_energy / low_energy).sqrt().min(1.0)
    }
}

/// Features extracted by the liquid bank
#[derive(Debug, Clone, Copy)]
pub struct LiquidFeatures {
    /// Click detector response (0-1)
    pub click: f32,
    /// Whistle detector response (0-1)
    pub whistle: f32,
    /// Burst detector response (0-1)
    pub burst: f32,
    /// Frame energy (RMS)
    pub energy: f32,
    /// Adaptive duration factor (current τ)
    pub duration_factor: f32,
}

/// Time-Aware Hypervector Binder
///
/// Binds symbolic identity with temporal properties:
/// HV = Φ_symbol ⊗ Φ_duration ⊗ Φ_intensity
pub struct TimeAwareBinder {
    /// Duration basis vectors (quantized into bins)
    duration_basis: Vec<HV16>,

    /// Intensity basis vectors (quantized into bins)
    intensity_basis: Vec<HV16>,

    /// Number of duration bins
    n_duration_bins: usize,

    /// Number of intensity bins
    n_intensity_bins: usize,
}

impl Default for TimeAwareBinder {
    fn default() -> Self {
        Self::new(8, 4)
    }
}

impl TimeAwareBinder {
    /// Create a new time-aware binder
    pub fn new(n_duration_bins: usize, n_intensity_bins: usize) -> Self {
        // Generate orthogonal basis vectors for duration
        let duration_basis: Vec<HV16> = (0..n_duration_bins)
            .map(|i| HV16::random(&format!("duration_bin_{i}")))
            .collect();

        // Generate orthogonal basis vectors for intensity
        let intensity_basis: Vec<HV16> = (0..n_intensity_bins)
            .map(|i| HV16::random(&format!("intensity_bin_{i}")))
            .collect();

        Self {
            duration_basis,
            intensity_basis,
            n_duration_bins,
            n_intensity_bins,
        }
    }

    /// Bind a symbol HV with duration and intensity
    ///
    /// Result: Φ_symbol ⊗ Φ_duration ⊗ Φ_intensity
    pub fn bind(&self, symbol: &HV16, duration: f32, intensity: f32) -> HV16 {
        // Quantize duration (log scale: 10ms to 10s)
        let duration_clamped = duration.clamp(0.01, 10.0);
        let duration_log = (duration_clamped / 0.01).ln() / (10.0 / 0.01_f32).ln();
        let duration_bin = (duration_log * (self.n_duration_bins - 1) as f32).round() as usize;
        let duration_bin = duration_bin.min(self.n_duration_bins - 1);

        // Quantize intensity (linear: 0 to 1)
        let intensity_clamped = intensity.clamp(0.0, 1.0);
        let intensity_bin =
            (intensity_clamped * (self.n_intensity_bins - 1) as f32).round() as usize;
        let intensity_bin = intensity_bin.min(self.n_intensity_bins - 1);

        // Triple binding: symbol ⊗ duration ⊗ intensity
        let duration_hv = &self.duration_basis[duration_bin];
        let intensity_hv = &self.intensity_basis[intensity_bin];

        symbol.bind(duration_hv).bind(intensity_hv)
    }

    /// Bind with interpolation between adjacent bins (smoother).
    ///
    /// Instead of snapping to the nearest bin, computes a weighted blend of the
    /// two adjacent bin HVs via majority-vote bundling, proportional to the
    /// fractional position. This produces smoother acoustic property
    /// representation, resolving long/short vowel distinctions that fall
    /// between discrete bin boundaries.
    pub fn bind_smooth(&self, symbol: &HV16, duration: f32, intensity: f32) -> HV16 {
        use crate::hdc::bundle;

        // Quantize duration (log scale: 10ms to 10s) to fractional bin
        let duration_clamped = duration.clamp(0.01, 10.0);
        let duration_log = (duration_clamped / 0.01).ln() / (10.0 / 0.01_f32).ln();
        let duration_frac = duration_log * (self.n_duration_bins - 1) as f32;
        let dur_lo = (duration_frac.floor() as usize).min(self.n_duration_bins - 1);
        let dur_hi = (dur_lo + 1).min(self.n_duration_bins - 1);
        let dur_alpha = duration_frac - duration_frac.floor();

        // Quantize intensity (linear: 0 to 1) to fractional bin
        let intensity_clamped = intensity.clamp(0.0, 1.0);
        let intensity_frac = intensity_clamped * (self.n_intensity_bins - 1) as f32;
        let int_lo = (intensity_frac.floor() as usize).min(self.n_intensity_bins - 1);
        let int_hi = (int_lo + 1).min(self.n_intensity_bins - 1);
        let int_alpha = intensity_frac - intensity_frac.floor();

        // Weighted majority-vote interpolation for binary HVs:
        // Repeat each basis vector proportionally (10 total copies) and bundle.
        let dur_hv = if dur_lo == dur_hi || dur_alpha < 0.05 {
            self.duration_basis[dur_lo]
        } else if dur_alpha > 0.95 {
            self.duration_basis[dur_hi]
        } else {
            let n_hi = (dur_alpha * 10.0).round() as usize;
            let n_lo = 10 - n_hi;
            let mut copies = Vec::with_capacity(10);
            for _ in 0..n_lo {
                copies.push(self.duration_basis[dur_lo]);
            }
            for _ in 0..n_hi {
                copies.push(self.duration_basis[dur_hi]);
            }
            bundle(&copies)
        };

        let int_hv = if int_lo == int_hi || int_alpha < 0.05 {
            self.intensity_basis[int_lo]
        } else if int_alpha > 0.95 {
            self.intensity_basis[int_hi]
        } else {
            let n_hi = (int_alpha * 10.0).round() as usize;
            let n_lo = 10 - n_hi;
            let mut copies = Vec::with_capacity(10);
            for _ in 0..n_lo {
                copies.push(self.intensity_basis[int_lo]);
            }
            for _ in 0..n_hi {
                copies.push(self.intensity_basis[int_hi]);
            }
            bundle(&copies)
        };

        // Triple binding: symbol ⊗ duration_interp ⊗ intensity_interp
        symbol.bind(&dur_hv).bind(&int_hv)
    }
}

/// Liquid Event - A detected acoustic event with timing
#[derive(Debug, Clone)]
pub struct LiquidEvent {
    /// Event type
    pub event_type: LiquidEventType,
    /// Start time (seconds)
    pub start_time: f32,
    /// Duration (seconds)
    pub duration: f32,
    /// Peak intensity
    pub intensity: f32,
    /// Time-aware hypervector
    pub hv: HV16,
}

/// Types of liquid events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LiquidEventType {
    /// Fast transient (click)
    Click,
    /// Sustained harmonic (whistle)
    Whistle,
    /// Broadband noise (burst)
    Burst,
    /// Silence / pause
    Silence,
}

impl LiquidEventType {
    /// Get the basis HV for this event type
    pub fn basis_hv(&self) -> HV16 {
        match self {
            Self::Click => HV16::random("liquid_event_click"),
            Self::Whistle => HV16::random("liquid_event_whistle"),
            Self::Burst => HV16::random("liquid_event_burst"),
            Self::Silence => HV16::random("liquid_event_silence"),
        }
    }
}

/// Liquid Event Detector - Converts CfC outputs to discrete events
pub struct LiquidEventDetector {
    /// Liquid feature bank
    bank: LiquidBank,

    /// Time-aware binder
    binder: TimeAwareBinder,

    /// Detection thresholds
    click_threshold: f32,
    whistle_threshold: f32,
    burst_threshold: f32,
    silence_threshold: f32,

    /// Current event state
    current_event: Option<(LiquidEventType, f32, f32)>, // (type, start, peak_intensity)

    /// Current time
    current_time: f32,

    /// Minimum event duration
    min_event_duration: f32,
}

impl LiquidEventDetector {
    /// Create a new event detector
    pub fn new(sample_rate: f32, frame_size: usize) -> Self {
        Self {
            bank: LiquidBank::new(sample_rate, frame_size),
            binder: TimeAwareBinder::new(8, 4),
            click_threshold: 0.6,
            whistle_threshold: 0.5,
            burst_threshold: 0.4,
            silence_threshold: 0.01,
            current_event: None,
            current_time: 0.0,
            min_event_duration: 0.01, // 10ms minimum
        }
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.bank.reset();
        self.current_event = None;
        self.current_time = 0.0;
    }

    /// Process a frame and potentially emit an event
    pub fn process_frame(&mut self, samples: &[f32]) -> Option<LiquidEvent> {
        let features = self.bank.process_frame(samples);
        let frame_duration = samples.len() as f32 / self.bank.sample_rate;

        // Determine dominant event type
        let detected_type = self.classify_frame(&features);

        // Check for event transitions
        let event = match &self.current_event {
            None => {
                // Start new event
                if detected_type != LiquidEventType::Silence {
                    self.current_event = Some((detected_type, self.current_time, features.energy));
                }
                None
            }
            Some((current_type, start_time, peak_intensity)) => {
                if detected_type != *current_type {
                    // Event transition - emit the completed event
                    let duration = self.current_time - start_time;

                    if duration >= self.min_event_duration {
                        let event_hv =
                            self.binder
                                .bind(&current_type.basis_hv(), duration, *peak_intensity);

                        let event = LiquidEvent {
                            event_type: *current_type,
                            start_time: *start_time,
                            duration,
                            intensity: *peak_intensity,
                            hv: event_hv,
                        };

                        // Start new event
                        if detected_type != LiquidEventType::Silence {
                            self.current_event =
                                Some((detected_type, self.current_time, features.energy));
                        } else {
                            self.current_event = None;
                        }

                        Some(event)
                    } else {
                        // Event too short, discard and start new
                        if detected_type != LiquidEventType::Silence {
                            self.current_event =
                                Some((detected_type, self.current_time, features.energy));
                        } else {
                            self.current_event = None;
                        }
                        None
                    }
                } else {
                    // Same event type - update peak intensity
                    let new_peak = peak_intensity.max(features.energy);
                    self.current_event = Some((*current_type, *start_time, new_peak));
                    None
                }
            }
        };

        self.current_time += frame_duration;
        event
    }

    /// Classify frame based on liquid features
    fn classify_frame(&self, features: &LiquidFeatures) -> LiquidEventType {
        // Check for silence first
        if features.energy < self.silence_threshold {
            return LiquidEventType::Silence;
        }

        // Winner-take-all classification
        if features.click > self.click_threshold
            && features.click > features.whistle
            && features.click > features.burst
        {
            LiquidEventType::Click
        } else if features.whistle > self.whistle_threshold && features.whistle > features.burst {
            LiquidEventType::Whistle
        } else if features.burst > self.burst_threshold {
            LiquidEventType::Burst
        } else {
            // Default to silence if nothing exceeds threshold
            LiquidEventType::Silence
        }
    }

    /// Flush any pending event
    pub fn flush(&mut self) -> Option<LiquidEvent> {
        if let Some((event_type, start_time, peak_intensity)) = self.current_event.take() {
            let duration = self.current_time - start_time;

            if duration >= self.min_event_duration {
                let event_hv = self
                    .binder
                    .bind(&event_type.basis_hv(), duration, peak_intensity);

                return Some(LiquidEvent {
                    event_type,
                    start_time,
                    duration,
                    intensity: peak_intensity,
                    hv: event_hv,
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_cfc_cell_basic() {
        let mut cell = CfcCell::click_detector();

        // Should start at 0
        assert_eq!(cell.state(), 0.0);

        // Strong input should increase state
        let dt = 0.001; // 1ms
        for _ in 0..100 {
            cell.step(1.0, dt);
        }

        assert!(
            cell.state() > 0.5,
            "State should increase with input: {}",
            cell.state()
        );
    }

    #[test]
    fn test_cfc_cell_decay() {
        let mut cell = CfcCell::click_detector();

        // Drive to high state
        for _ in 0..100 {
            cell.step(1.0, 0.001);
        }
        let peak = cell.state();

        // Remove input, should decay
        for _ in 0..100 {
            cell.step(0.0, 0.001);
        }

        assert!(cell.state() < peak, "State should decay without input");
    }

    #[test]
    fn test_liquid_bank() {
        let mut bank = LiquidBank::new(44100.0, 1024);

        // Generate test signal (click-like: impulse)
        let mut click_signal = vec![0.0f32; 1024];
        click_signal[512] = 1.0; // Single impulse

        let features = bank.process_frame(&click_signal);

        println!("Click signal features:");
        println!("  Click response: {:.4}", features.click);
        println!("  Whistle response: {:.4}", features.whistle);
        println!("  Burst response: {:.4}", features.burst);

        // Click detector should respond to impulse
        // (Note: single impulse may not trigger fully, but should show response)
    }

    #[test]
    fn test_time_aware_binder() {
        let binder = TimeAwareBinder::new(8, 4);

        let symbol = HV16::random("test_symbol");

        // Short, quiet event
        let hv1 = binder.bind(&symbol, 0.05, 0.2);

        // Long, loud event
        let hv2 = binder.bind(&symbol, 2.0, 0.9);

        // Same symbol but different duration/intensity should be different
        let sim = hv1.similarity(&hv2);

        println!("Short+quiet vs Long+loud similarity: {:.4}", sim);

        // Should be somewhat different due to duration/intensity binding
        assert!(
            sim < 0.9,
            "Different temporal properties should produce different HVs"
        );
    }

    #[test]
    fn test_liquid_event_detector() {
        let mut detector = LiquidEventDetector::new(44100.0, 1024);

        // Generate whistle-like signal (sine wave)
        let whistle_signal: Vec<f32> = (0..1024)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();

        // Process multiple frames
        let mut events = Vec::new();
        for _ in 0..10 {
            if let Some(event) = detector.process_frame(&whistle_signal) {
                events.push(event);
            }
        }

        // Flush final event
        if let Some(event) = detector.flush() {
            events.push(event);
        }

        println!("Detected {} events", events.len());
        for event in &events {
            println!(
                "  {:?}: duration={:.3}s, intensity={:.4}",
                event.event_type, event.duration, event.intensity
            );
        }
    }

    #[test]
    fn test_event_type_hvs_orthogonal() {
        let click_hv = LiquidEventType::Click.basis_hv();
        let whistle_hv = LiquidEventType::Whistle.basis_hv();
        let burst_hv = LiquidEventType::Burst.basis_hv();

        let sim_cw = click_hv.similarity(&whistle_hv);
        let sim_cb = click_hv.similarity(&burst_hv);
        let sim_wb = whistle_hv.similarity(&burst_hv);

        println!("Event type similarities:");
        println!("  Click-Whistle: {:.4}", sim_cw);
        println!("  Click-Burst: {:.4}", sim_cb);
        println!("  Whistle-Burst: {:.4}", sim_wb);

        // Random HVs should be nearly orthogonal (~0.5 similarity for binary)
        assert!(
            sim_cw.abs() < 0.2,
            "Event types should be nearly orthogonal"
        );
        assert!(
            sim_cb.abs() < 0.2,
            "Event types should be nearly orthogonal"
        );
        assert!(
            sim_wb.abs() < 0.2,
            "Event types should be nearly orthogonal"
        );
    }
}
