// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phase-Amplitude Coupling (PAC) for Cross-Frequency Analysis
//!
//! Measures how the phase of a low-frequency oscillation modulates the amplitude
//! of a high-frequency oscillation. This is a key signature of neural information
//! integration across cortical scales.
//!
//! ## Consciousness Relevance
//!
//! - **Theta-gamma PAC**: Memory encoding and retrieval (hippocampal)
//! - **Alpha-gamma PAC**: Attention and perceptual binding
//! - **Delta-beta PAC**: Sleep stage transitions
//! - **PAC strength correlates with consciousness level**: Reduced under anesthesia
//!
//! ## Algorithms
//!
//! - **Modulation Index (MI)**: KL divergence between amplitude distribution and
//!   uniform distribution (Tort et al. 2010)
//! - **Mean Vector Length (MVL)**: Magnitude of mean phase-amplitude vector
//!   (Canolty et al. 2006)
//! - **Surrogate testing**: Phase-shuffle surrogates for statistical significance
//! - **Comodulogram**: MI computed across all frequency pairs

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use super::spectral_analysis::{Complex, SpectralAnalyzer};

// ---------------------------------------------------------------------------
// Local complex IFFT helper (avoids depending on private fft_in_place)
// ---------------------------------------------------------------------------

/// Compute complex IFFT: conjugate → FFT → conjugate → scale.
///
/// Uses `SpectralAnalyzer::fft` (which takes real input) indirectly by
/// implementing a minimal Cooley-Tukey radix-2 DIT FFT for complex data.
fn complex_ifft(spectrum: &[Complex]) -> Vec<Complex> {
    let n = spectrum.len();
    // IFFT = conj → FFT → conj → (1/N)
    let mut buf: Vec<Complex> = spectrum.iter().map(|z| z.conj()).collect();
    fft_radix2(&mut buf);
    let inv = 1.0 / n as f64;
    buf.iter()
        .map(|z| Complex {
            re: z.re * inv,
            im: -z.im * inv,
        })
        .collect()
}

/// In-place Cooley-Tukey radix-2 DIT FFT for complex data.
fn fft_radix2(buf: &mut [Complex]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    // Butterfly operations
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wn = Complex {
            re: angle.cos(),
            im: angle.sin(),
        };

        let mut i = 0;
        while i < n {
            let mut w = Complex { re: 1.0, im: 0.0 };
            for k in 0..half {
                let u = buf[i + k];
                let t = Complex {
                    re: w.re * buf[i + k + half].re - w.im * buf[i + k + half].im,
                    im: w.re * buf[i + k + half].im + w.im * buf[i + k + half].re,
                };
                buf[i + k] = Complex {
                    re: u.re + t.re,
                    im: u.im + t.im,
                };
                buf[i + k + half] = Complex {
                    re: u.re - t.re,
                    im: u.im - t.im,
                };
                let new_w = Complex {
                    re: w.re * wn.re - w.im * wn.im,
                    im: w.re * wn.im + w.im * wn.re,
                };
                w = new_w;
            }
            i += len;
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// PAC analysis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacConfig {
    /// Sampling rate in Hz.
    pub sample_rate: f64,
    /// Number of phase bins for Modulation Index computation.
    pub n_phase_bins: usize,
    /// Number of surrogate shuffles for significance testing.
    pub n_surrogates: usize,
    /// Significance threshold (p-value).
    pub significance_level: f64,
    /// Bandpass filter order (number of FFT taps).
    pub filter_order: usize,
}

impl Default for PacConfig {
    fn default() -> Self {
        Self {
            sample_rate: 256.0,
            n_phase_bins: 18, // 20° bins
            n_surrogates: 200,
            significance_level: 0.05,
            filter_order: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// PAC Result types
// ---------------------------------------------------------------------------

/// Result of a single PAC computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacResult {
    /// Modulation Index (MI): 0 = no coupling, higher = stronger coupling.
    /// Normalized by log(n_bins) so range is approximately [0, 1].
    pub modulation_index: f64,
    /// Mean Vector Length (MVL): magnitude of mean phase-amplitude vector.
    pub mean_vector_length: f64,
    /// Preferred phase (radians): phase at which amplitude is maximal.
    pub preferred_phase: f64,
    /// P-value from surrogate testing (if computed). None if no surrogates.
    pub p_value: Option<f64>,
    /// Phase of low-frequency component.
    pub phase_freq_range: (f64, f64),
    /// Amplitude of high-frequency component.
    pub amp_freq_range: (f64, f64),
}

/// Comodulogram: PAC computed across a grid of frequency pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comodulogram {
    /// MI values. Dimensions: n_phase_freqs × n_amp_freqs.
    pub mi_matrix: Vec<Vec<f64>>,
    /// Phase frequency centers in Hz.
    pub phase_frequencies: Vec<f64>,
    /// Amplitude frequency centers in Hz.
    pub amp_frequencies: Vec<f64>,
    /// Phase frequency bandwidths in Hz.
    pub phase_bandwidths: Vec<f64>,
    /// Amplitude frequency bandwidths in Hz.
    pub amp_bandwidths: Vec<f64>,
}

// ---------------------------------------------------------------------------
// PAC Analyzer
// ---------------------------------------------------------------------------

/// Phase-Amplitude Coupling analyzer.
pub struct PacAnalyzer {
    pub config: PacConfig,
}

impl PacAnalyzer {
    pub fn new(config: PacConfig) -> Self {
        Self { config }
    }

    // -----------------------------------------------------------------------
    // Bandpass filtering via FFT
    // -----------------------------------------------------------------------

    /// Bandpass filter a signal using FFT multiplication.
    ///
    /// Zero-phase filtering: FFT → zero out-of-band → IFFT.
    fn bandpass(&self, signal: &[f64], f_low: f64, f_high: f64) -> Vec<f64> {
        let n = signal.len().next_power_of_two();
        let df = self.config.sample_rate / n as f64;

        let spectrum = SpectralAnalyzer::fft(signal);
        let mut filtered = vec![Complex::zero(); spectrum.len()];

        for k in 0..spectrum.len() {
            let freq = if k <= n / 2 {
                k as f64 * df
            } else {
                (n - k) as f64 * df
            };

            if freq >= f_low && freq <= f_high {
                filtered[k] = spectrum[k];
            }
        }

        SpectralAnalyzer::ifft(&filtered)
            .into_iter()
            .take(signal.len())
            .collect()
    }

    /// Extract instantaneous phase using the Hilbert transform (via FFT).
    ///
    /// Analytic signal: z(t) = x(t) + i*H[x(t)]
    /// Phase: φ(t) = atan2(Im(z), Re(z))
    fn hilbert_phase(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len().next_power_of_two();
        let mut spectrum = SpectralAnalyzer::fft(signal);

        // Create analytic signal by zeroing negative frequencies
        // and doubling positive frequencies
        for k in 0..spectrum.len() {
            if k == 0 || k == n / 2 {
                // DC and Nyquist: keep as-is
            } else if k < n / 2 {
                // Positive frequencies: double
                spectrum[k] = spectrum[k] * 2.0;
            } else {
                // Negative frequencies: zero
                spectrum[k] = Complex::zero();
            }
        }

        // Complex IFFT to get full analytic signal (need both re and im)
        let analytic = complex_ifft(&spectrum);

        // Extract phase from complex analytic signal
        analytic
            .iter()
            .take(signal.len())
            .map(|z| z.im.atan2(z.re))
            .collect()
    }

    /// Extract instantaneous amplitude envelope using the Hilbert transform.
    fn hilbert_amplitude(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len().next_power_of_two();
        let mut spectrum = SpectralAnalyzer::fft(signal);

        for k in 0..spectrum.len() {
            if k == 0 || k == n / 2 {
                // keep
            } else if k < n / 2 {
                spectrum[k] = spectrum[k] * 2.0;
            } else {
                spectrum[k] = Complex::zero();
            }
        }

        // Complex IFFT to get full analytic signal
        let analytic = complex_ifft(&spectrum);

        analytic
            .iter()
            .take(signal.len())
            .map(|z| (z.re * z.re + z.im * z.im).sqrt())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Modulation Index (Tort et al. 2010)
    // -----------------------------------------------------------------------

    /// Compute the Modulation Index between a phase signal and an amplitude signal.
    ///
    /// MI = KL(P || U) / log(N_bins)
    ///
    /// where P is the distribution of mean amplitude per phase bin, and U is
    /// the uniform distribution. MI ∈ [0, 1].
    fn modulation_index_from_signals(&self, phase: &[f64], amplitude: &[f64]) -> (f64, f64) {
        let n_bins = self.config.n_phase_bins;
        let bin_width = 2.0 * PI / n_bins as f64;

        // Bin amplitudes by phase
        let mut bin_sums = vec![0.0; n_bins];
        let mut bin_counts = vec![0usize; n_bins];

        let len = phase.len().min(amplitude.len());
        for i in 0..len {
            let mut p = phase[i];
            // Normalize to [0, 2π)
            while p < 0.0 {
                p += 2.0 * PI;
            }
            while p >= 2.0 * PI {
                p -= 2.0 * PI;
            }
            let bin = (p / bin_width) as usize;
            let bin = bin.min(n_bins - 1);
            bin_sums[bin] += amplitude[i];
            bin_counts[bin] += 1;
        }

        // Mean amplitude per bin
        let mean_amps: Vec<f64> = bin_sums
            .iter()
            .zip(bin_counts.iter())
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect();

        // Normalize to probability distribution
        let total: f64 = mean_amps.iter().sum();
        if total <= 0.0 {
            return (0.0, 0.0);
        }

        let p_dist: Vec<f64> = mean_amps.iter().map(|&a| a / total).collect();

        // KL divergence from uniform
        let uniform = 1.0 / n_bins as f64;
        let mut kl = 0.0;
        for &p in &p_dist {
            if p > 0.0 {
                kl += p * (p / uniform).ln();
            }
        }

        let ln_bins = (n_bins as f64).ln();
        let mi = if ln_bins > 0.0 {
            (kl / ln_bins).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Preferred phase: phase bin with max mean amplitude
        let preferred_bin = mean_amps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let preferred_phase = (preferred_bin as f64 + 0.5) * bin_width;

        (mi, preferred_phase)
    }

    /// Compute Mean Vector Length (Canolty et al. 2006).
    ///
    /// MVL = |mean(amplitude * exp(i*phase))| / mean(amplitude)
    fn mean_vector_length(phase: &[f64], amplitude: &[f64]) -> f64 {
        let len = phase.len().min(amplitude.len());
        if len == 0 {
            return 0.0;
        }

        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        let mut sum_amp = 0.0;

        for i in 0..len {
            sum_re += amplitude[i] * phase[i].cos();
            sum_im += amplitude[i] * phase[i].sin();
            sum_amp += amplitude[i];
        }

        if sum_amp <= 0.0 {
            return 0.0;
        }

        let mean_re = sum_re / len as f64;
        let mean_im = sum_im / len as f64;
        let mvl = (mean_re * mean_re + mean_im * mean_im).sqrt();
        let mean_amp = sum_amp / len as f64;

        if mean_amp > 0.0 { mvl / mean_amp } else { 0.0 }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Compute Phase-Amplitude Coupling between two frequency bands.
    ///
    /// - `phase_band`: (f_low, f_high) for the phase-providing oscillation
    /// - `amp_band`: (f_low, f_high) for the amplitude-providing oscillation
    ///
    /// The phase band should be lower frequency than the amplitude band.
    pub fn compute_pac(
        &self,
        signal: &[f64],
        phase_band: (f64, f64),
        amp_band: (f64, f64),
    ) -> PacResult {
        // Bandpass filter for phase and amplitude bands
        let phase_signal = self.bandpass(signal, phase_band.0, phase_band.1);
        let amp_signal = self.bandpass(signal, amp_band.0, amp_band.1);

        // Extract phase and amplitude envelope
        let phase = self.hilbert_phase(&phase_signal);
        let amplitude = self.hilbert_amplitude(&amp_signal);

        // Compute MI
        let (mi, preferred_phase) = self.modulation_index_from_signals(&phase, &amplitude);

        // Compute MVL
        let mvl = Self::mean_vector_length(&phase, &amplitude);

        // Surrogate testing
        let p_value = if self.config.n_surrogates > 0 {
            Some(self.surrogate_test(&phase, &amplitude, mi))
        } else {
            None
        };

        PacResult {
            modulation_index: mi,
            mean_vector_length: mvl,
            preferred_phase,
            p_value,
            phase_freq_range: phase_band,
            amp_freq_range: amp_band,
        }
    }

    /// Surrogate testing via time-shifted surrogates.
    ///
    /// Shuffles the amplitude time series relative to the phase time series
    /// by random circular shifts, computing MI for each surrogate. The p-value
    /// is the fraction of surrogates with MI >= observed MI.
    fn surrogate_test(&self, phase: &[f64], amplitude: &[f64], observed_mi: f64) -> f64 {
        let len = phase.len().min(amplitude.len());
        if len < 10 {
            return 1.0;
        }

        let mut n_exceeding = 0;
        let mut rng_state: u64 = 0xdeadbeef;

        for _ in 0..self.config.n_surrogates {
            // Simple LCG for deterministic pseudo-random shifts
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let shift = (rng_state >> 33) as usize % (len - 2) + 1;

            // Circular shift amplitude
            let shifted_amp: Vec<f64> = amplitude[shift..len]
                .iter()
                .chain(amplitude[..shift].iter())
                .copied()
                .collect();

            let (surr_mi, _) = self.modulation_index_from_signals(phase, &shifted_amp);

            if surr_mi >= observed_mi {
                n_exceeding += 1;
            }
        }

        (n_exceeding as f64 + 1.0) / (self.config.n_surrogates as f64 + 1.0)
    }

    /// Compute a comodulogram: PAC across a grid of frequency pairs.
    ///
    /// Phase frequencies are scanned from `phase_range.0` to `phase_range.1`
    /// in steps of `phase_step`. Similarly for amplitude frequencies.
    pub fn comodulogram(
        &self,
        signal: &[f64],
        phase_range: (f64, f64),
        phase_step: f64,
        phase_bandwidth: f64,
        amp_range: (f64, f64),
        amp_step: f64,
        amp_bandwidth: f64,
    ) -> Comodulogram {
        let mut phase_freqs = Vec::new();
        let mut f = phase_range.0;
        while f <= phase_range.1 {
            phase_freqs.push(f);
            f += phase_step;
        }

        let mut amp_freqs = Vec::new();
        let mut f = amp_range.0;
        while f <= amp_range.1 {
            amp_freqs.push(f);
            f += amp_step;
        }

        let mut mi_matrix = vec![vec![0.0; amp_freqs.len()]; phase_freqs.len()];

        // Create a no-surrogate config for speed
        let fast_config = PacConfig {
            n_surrogates: 0,
            ..self.config.clone()
        };
        let fast_analyzer = PacAnalyzer::new(fast_config);

        for (i, &pf) in phase_freqs.iter().enumerate() {
            let p_lo = (pf - phase_bandwidth / 2.0).max(0.5);
            let p_hi = pf + phase_bandwidth / 2.0;

            for (j, &af) in amp_freqs.iter().enumerate() {
                let a_lo = (af - amp_bandwidth / 2.0).max(0.5);
                let a_hi = af + amp_bandwidth / 2.0;

                let result = fast_analyzer.compute_pac(signal, (p_lo, p_hi), (a_lo, a_hi));
                mi_matrix[i][j] = result.modulation_index;
            }
        }

        let n_phase = mi_matrix.len();
        let n_amp = mi_matrix.first().map(|r| r.len()).unwrap_or(0);

        Comodulogram {
            mi_matrix,
            phase_frequencies: phase_freqs,
            amp_frequencies: amp_freqs,
            phase_bandwidths: vec![phase_bandwidth; n_phase],
            amp_bandwidths: vec![amp_bandwidth; n_amp],
        }
    }

    /// Compute standard consciousness-relevant PAC pairs.
    ///
    /// Returns PAC for: theta-gamma, alpha-gamma, delta-beta, delta-gamma.
    pub fn consciousness_pac(&self, signal: &[f64]) -> Vec<PacResult> {
        vec![
            self.compute_pac(signal, (4.0, 8.0), (30.0, 80.0)), // Theta-Gamma
            self.compute_pac(signal, (8.0, 13.0), (30.0, 80.0)), // Alpha-Gamma
            self.compute_pac(signal, (0.5, 4.0), (13.0, 30.0)), // Delta-Beta
            self.compute_pac(signal, (0.5, 4.0), (30.0, 80.0)), // Delta-Gamma
        ]
    }
}

// ===========================================================================
// Unit tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn coupled_signal(
        phase_freq: f64,
        amp_freq: f64,
        sr: f64,
        n: usize,
        coupling_strength: f64,
    ) -> Vec<f64> {
        // Generate a signal where the amplitude of the fast oscillation is
        // modulated by the phase of the slow oscillation.
        (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                let slow_phase = 2.0 * PI * phase_freq * t;
                // Amplitude modulation: higher amp when slow phase is positive
                let modulation = 1.0 + coupling_strength * slow_phase.cos();
                let fast = (2.0 * PI * amp_freq * t).sin();
                let slow = slow_phase.sin();
                slow + modulation * fast
            })
            .collect()
    }

    #[test]
    fn test_pac_detects_coupling() {
        let sr = 256.0;
        let n = 2048;

        let analyzer = PacAnalyzer::new(PacConfig {
            sample_rate: sr,
            n_surrogates: 0,
            ..Default::default()
        });

        // Strong theta-gamma coupling
        let coupled = coupled_signal(6.0, 40.0, sr, n, 0.8);
        let result = analyzer.compute_pac(&coupled, (4.0, 8.0), (30.0, 50.0));

        assert!(
            result.modulation_index > 0.01,
            "Should detect coupling, MI = {}",
            result.modulation_index
        );
    }

    #[test]
    fn test_pac_no_coupling_low_mi() {
        let sr = 256.0;
        let n = 2048;

        let analyzer = PacAnalyzer::new(PacConfig {
            sample_rate: sr,
            n_surrogates: 0,
            ..Default::default()
        });

        // Independent signals: no coupling
        let uncoupled: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 6.0 * t).sin() + (2.0 * PI * 40.0 * t).sin()
            })
            .collect();

        let result = analyzer.compute_pac(&uncoupled, (4.0, 8.0), (30.0, 50.0));

        // MI should be low for uncoupled signals
        assert!(
            result.modulation_index < 0.1,
            "Uncoupled signals should have low MI, got {}",
            result.modulation_index
        );
    }

    #[test]
    fn test_mvl_basic() {
        // Phase locked to 0: all amplitude at phase 0
        let phase = vec![0.0; 100];
        let amplitude = vec![1.0; 100];

        let mvl = PacAnalyzer::mean_vector_length(&phase, &amplitude);
        assert!(
            (mvl - 1.0).abs() < 0.01,
            "Perfect phase locking should give MVL ≈ 1, got {}",
            mvl
        );
    }

    #[test]
    fn test_mvl_random_phase() {
        // Random phases: MVL should be low
        let phase: Vec<f64> = (0..1000)
            .map(|i| (i as f64 * 2.0 * PI / 7.0) % (2.0 * PI))
            .collect();
        let amplitude = vec![1.0; 1000];

        let mvl = PacAnalyzer::mean_vector_length(&phase, &amplitude);
        assert!(mvl < 0.3, "Random phases should give low MVL, got {}", mvl);
    }

    #[test]
    fn test_surrogate_significance() {
        let sr = 256.0;
        let n = 2048;

        let analyzer = PacAnalyzer::new(PacConfig {
            sample_rate: sr,
            n_surrogates: 50,
            ..Default::default()
        });

        // Strong coupling should be significant
        let coupled = coupled_signal(6.0, 40.0, sr, n, 0.9);
        let result = analyzer.compute_pac(&coupled, (4.0, 8.0), (30.0, 50.0));

        if let Some(p) = result.p_value {
            assert!(p < 0.2, "Strong coupling should be significant, p = {}", p);
        }
    }

    #[test]
    fn test_consciousness_pac() {
        let sr = 256.0;
        let n = 2048;

        let analyzer = PacAnalyzer::new(PacConfig {
            sample_rate: sr,
            n_surrogates: 0,
            ..Default::default()
        });

        // Generate a simple mixed signal
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 6.0 * t).sin()
                    + 0.5 * (2.0 * PI * 10.0 * t).sin()
                    + 0.3 * (2.0 * PI * 40.0 * t).sin()
            })
            .collect();

        let results = analyzer.consciousness_pac(&signal);
        assert_eq!(results.len(), 4, "Should return 4 PAC pairs");
    }

    #[test]
    fn test_comodulogram_shape() {
        let sr = 256.0;
        let n = 1024;

        let analyzer = PacAnalyzer::new(PacConfig {
            sample_rate: sr,
            n_surrogates: 0,
            ..Default::default()
        });

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (2.0 * PI * 6.0 * t).sin() + (2.0 * PI * 40.0 * t).sin()
            })
            .collect();

        let comod = analyzer.comodulogram(
            &signal,
            (2.0, 12.0),  // Phase range
            2.0,          // Phase step
            4.0,          // Phase bandwidth
            (20.0, 60.0), // Amp range
            10.0,         // Amp step
            10.0,         // Amp bandwidth
        );

        assert_eq!(comod.phase_frequencies.len(), 6); // 2,4,6,8,10,12
        assert_eq!(comod.amp_frequencies.len(), 5); // 20,30,40,50,60
        assert_eq!(comod.mi_matrix.len(), 6);
        assert_eq!(comod.mi_matrix[0].len(), 5);
    }
}
