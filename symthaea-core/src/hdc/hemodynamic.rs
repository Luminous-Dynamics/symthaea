// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hemodynamic Response Function (HRF) — bridging neural dynamics to fMRI BOLD.
//!
//! Symthaea's cognitive loop runs at ~31 Hz, but fMRI measures the Blood-Oxygen-Level
//! Dependent (BOLD) signal at ~0.5-2 Hz with a hemodynamic lag of 4-6 seconds.
//! To compare simulated cortical activations against fMRI predictions (TRIBE v2),
//! we must convolve our fast dynamics with the canonical HRF and downsample.
//!
//! # Canonical Double-Gamma HRF (Glover 1999)
//!
//! h(t) = A₁ · (t/τ₁)^(α₁-1) · exp(-(t-τ₁)/β₁) / Γ(α₁)
//!      - c · A₂ · (t/τ₂)^(α₂-1) · exp(-(t-τ₂)/β₂) / Γ(α₂)
//!
//! Default parameters from SPM12 (Wellcome Trust Centre):
//!   Peak delay: 6s, undershoot delay: 16s, ratio c = 1/6
//!
//! # References
//!
//! - Glover, G.H. (1999). Deconvolution of impulse response in event-related
//!   BOLD fMRI. *NeuroImage*, 9(4), 416-429.
//! - Friston et al. (1998). Event-related fMRI: characterizing differential
//!   responses. *NeuroImage*, 7(1), 30-40.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::cortical_activation::{
    ActivationSource, CorticalActivationMap, CorticalActivationTimeseries,
};
use super::substrate_independence::CorticalRegion;

// ============================================================================
// Named constants (SPM12 canonical HRF)
// ============================================================================

/// Peak delay in seconds (positive BOLD response peak).
const HRF_PEAK_DELAY: f64 = 6.0;
/// Undershoot delay in seconds.
const HRF_UNDERSHOOT_DELAY: f64 = 16.0;
/// Peak dispersion (shape of the rise).
const HRF_PEAK_DISPERSION: f64 = 1.0;
/// Undershoot dispersion.
const HRF_UNDERSHOOT_DISPERSION: f64 = 1.0;
/// Undershoot-to-peak amplitude ratio (c in the formula).
const HRF_UNDERSHOOT_RATIO: f64 = 1.0 / 6.0;
/// Duration of the HRF kernel in seconds (after which it's ~0).
const HRF_KERNEL_DURATION: f64 = 32.0;

/// Configuration for the HRF convolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrfConfig {
    /// Peak delay in seconds.
    pub peak_delay: f64,
    /// Undershoot delay in seconds.
    pub undershoot_delay: f64,
    /// Undershoot/peak amplitude ratio.
    pub undershoot_ratio: f64,
    /// Duration of the HRF kernel in seconds.
    pub kernel_duration: f64,
    /// Target output sample rate in Hz (fMRI TR⁻¹, e.g., 0.5 for TR=2s).
    pub output_rate_hz: f64,
}

impl Default for HrfConfig {
    fn default() -> Self {
        Self {
            peak_delay: HRF_PEAK_DELAY,
            undershoot_delay: HRF_UNDERSHOOT_DELAY,
            undershoot_ratio: HRF_UNDERSHOOT_RATIO,
            kernel_duration: HRF_KERNEL_DURATION,
            output_rate_hz: 0.5, // TR = 2s
        }
    }
}

/// Precomputed HRF kernel at a given sample rate.
#[derive(Debug, Clone)]
pub struct HrfKernel {
    /// Kernel samples (normalized so sum of absolute values = 1).
    pub samples: Vec<f64>,
    /// Sample rate the kernel was computed at.
    pub sample_rate_hz: f64,
}

impl HrfKernel {
    /// Compute the HRF kernel at the given sample rate.
    pub fn new(config: &HrfConfig, sample_rate_hz: f64) -> Self {
        let n_samples = (config.kernel_duration * sample_rate_hz).ceil() as usize;
        let dt = 1.0 / sample_rate_hz;

        let mut samples = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 * dt;
            let peak = gamma_pdf(t, config.peak_delay, HRF_PEAK_DISPERSION);
            let undershoot = gamma_pdf(t, config.undershoot_delay, HRF_UNDERSHOOT_DISPERSION);
            samples.push(peak - config.undershoot_ratio * undershoot);
        }

        // Normalize: peak amplitude = 1.
        let max_abs = samples.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        if max_abs > 1e-12 {
            for s in &mut samples {
                *s /= max_abs;
            }
        }

        Self {
            samples,
            sample_rate_hz,
        }
    }

    /// Convolve a 1D signal with this HRF kernel (causal, same-length output).
    pub fn convolve(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let k = self.samples.len();
        let mut output = vec![0.0; n];

        for i in 0..n {
            let mut sum = 0.0;
            let jmax = k.min(i + 1);
            for j in 0..jmax {
                sum += signal[i - j] * self.samples[j];
            }
            output[i] = sum;
        }
        output
    }
}

/// Apply HRF convolution and downsampling to a `CorticalActivationTimeseries`.
///
/// Returns a new timeseries at the fMRI output rate with BOLD-like signals.
pub fn convolve_timeseries(
    ts: &CorticalActivationTimeseries,
    config: &HrfConfig,
) -> CorticalActivationTimeseries {
    let input_rate = ts.sample_rate_hz as f64;
    let kernel = HrfKernel::new(config, input_rate);
    let output_rate = config.output_rate_hz;

    // Downsample stride (how many input samples per output sample).
    let stride = (input_rate / output_rate).round() as usize;
    let stride = stride.max(1);

    let n_input = ts.maps.len();
    let n_output = if stride > 0 {
        (n_input + stride - 1) / stride
    } else {
        n_input
    };

    // Convolve each region independently.
    let mut convolved_regions: HashMap<CorticalRegion, Vec<f64>> = HashMap::new();
    for region in CorticalRegion::ALL {
        let signal: Vec<f64> = ts
            .region_timeseries(region)
            .into_iter()
            .map(|x| x as f64)
            .collect();
        let conv = kernel.convolve(&signal);
        convolved_regions.insert(region, conv);
    }

    // Downsample and build output maps.
    let mut output = CorticalActivationTimeseries::new(output_rate as f32);
    for out_idx in 0..n_output {
        let in_idx = out_idx * stride;
        if in_idx >= n_input {
            break;
        }
        let mut map = CorticalActivationMap::zeros(ActivationSource::Simulated, out_idx as u64);
        for region in CorticalRegion::ALL {
            let val = convolved_regions[&region][in_idx];
            // BOLD can be negative (undershoot); clamp to [0, 1] for activation map.
            map.set(region, val.clamp(0.0, 1.0) as f32);
        }
        output.push(map);
    }

    output
}

// ============================================================================
// Internal: Gamma probability density function
// ============================================================================

/// Gamma PDF: f(t; k, θ) = t^(k-1) * exp(-t/θ) / (θ^k * Γ(k))
/// where k = delay/dispersion, θ = dispersion.
///
/// Simplified for HRF where we use delay as the shape parameter.
fn gamma_pdf(t: f64, delay: f64, dispersion: f64) -> f64 {
    if t <= 0.0 || delay <= 0.0 || dispersion <= 0.0 {
        return 0.0;
    }
    let k = delay / dispersion;
    let theta = dispersion;
    let log_val = (k - 1.0) * t.ln() - t / theta - k * theta.ln() - ln_gamma(k);
    log_val.exp()
}

/// Natural log of the gamma function (Stirling approximation for large values,
/// Lanczos approximation for small values).
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation (g=7, n=9) — accurate to ~15 digits.
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(x) = π / (sin(πx) Γ(1-x))
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut ag = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        ag += c / (x + i as f64);
    }
    let tmp = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * tmp.ln() - tmp + ag.ln()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hrf_kernel_shape() {
        let config = HrfConfig::default();
        let kernel = HrfKernel::new(&config, 31.0);
        // Kernel should have ~32s * 31Hz = ~992 samples.
        assert!(kernel.samples.len() > 900 && kernel.samples.len() < 1100);
        // Peak should be normalized to ~1.0.
        let max = kernel
            .samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 1.0).abs() < 0.01, "Peak should be ~1.0, got {max}");
    }

    #[test]
    fn test_hrf_kernel_peaks_near_5s() {
        let config = HrfConfig::default();
        let kernel = HrfKernel::new(&config, 100.0); // High rate for precision
        let peak_idx = kernel
            .samples
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let peak_time = peak_idx as f64 / 100.0;
        // Gamma PDF peak is at (k-1)*theta; with delay=6, dispersion=1: peak ~5s.
        // The undershoot subtraction shifts it slightly.
        assert!(
            (peak_time - 5.0).abs() < 1.5,
            "HRF should peak near 5s, peaked at {peak_time}s"
        );
    }

    #[test]
    fn test_hrf_has_undershoot() {
        let config = HrfConfig::default();
        let kernel = HrfKernel::new(&config, 31.0);
        // There should be negative values after the peak (undershoot).
        let has_negative = kernel.samples.iter().any(|&x| x < -0.01);
        assert!(
            has_negative,
            "HRF should have an undershoot (negative values)"
        );
    }

    #[test]
    fn test_convolve_impulse() {
        let config = HrfConfig::default();
        let kernel = HrfKernel::new(&config, 10.0);
        // Impulse at t=0 → output should match kernel.
        let mut signal = vec![0.0; 320]; // 32s at 10Hz
        signal[0] = 1.0;
        let output = kernel.convolve(&signal);
        assert_eq!(output.len(), signal.len());
        // Output should be the kernel itself.
        for i in 0..kernel.samples.len().min(output.len()) {
            assert!(
                (output[i] - kernel.samples[i]).abs() < 1e-10,
                "Impulse response mismatch at sample {i}"
            );
        }
    }

    #[test]
    fn test_convolve_timeseries_downsamples() {
        let config = HrfConfig {
            output_rate_hz: 0.5, // TR = 2s
            ..Default::default()
        };
        // 10s of data at 31Hz = 310 samples.
        let mut ts = CorticalActivationTimeseries::new(31.0);
        for i in 0..310 {
            let mut map = CorticalActivationMap::zeros(ActivationSource::Simulated, i as u64);
            // Brief visual flash at t=1s (samples 31-61).
            if i >= 31 && i < 62 {
                map.set(CorticalRegion::Visual, 1.0);
            }
            ts.push(map);
        }
        let bold = convolve_timeseries(&ts, &config);
        // Output should be at 0.5 Hz over 10s → ~5 samples.
        assert!(bold.maps.len() >= 4 && bold.maps.len() <= 6);
        assert!((bold.sample_rate_hz - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ln_gamma_known_values() {
        // Γ(1) = 1 → ln(1) = 0
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);
        // Γ(5) = 24 → ln(24) ≈ 3.178
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-8);
        // Γ(0.5) = √π → ln(√π) ≈ 0.5724
        assert!((ln_gamma(0.5) - (std::f64::consts::PI.sqrt().ln())).abs() < 1e-8);
    }

    #[test]
    fn test_gamma_pdf_basic() {
        // Peak of gamma(t; k=6, θ=1) should be near t=5.
        let peak_t = (0..1000)
            .map(|i| {
                let t = i as f64 * 0.01;
                (i, gamma_pdf(t, 6.0, 1.0))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0 as f64
            * 0.01;
        assert!(
            (peak_t - 5.0).abs() < 0.5,
            "gamma PDF peak should be near 5.0, got {peak_t}"
        );
    }
}
