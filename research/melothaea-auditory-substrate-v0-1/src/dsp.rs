//! Deterministic CPU reference transforms for MEL-HEAR-011.
//!
//! This module provides a transparent reference implementation, not a
//! calibrated cochlear model. The ERB view is a triangular partition of
//! one-sided STFT power on the ERB-rate axis. The log-frequency view is a
//! triangular partition on a logarithmic frequency axis; it is intentionally
//! not named or claimed as a true constant-Q transform.

use std::error::Error;
use std::f64::consts::TAU;
use std::fmt;

/// A complex spectral coefficient. Both components are retained so phase is
/// available to downstream verification.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }

    fn multiply(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }

    /// Magnitude of this coefficient.
    pub fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared magnitude, avoiding an unnecessary square root.
    pub fn power(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Errors returned by the reference DSP transforms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DspError {
    InvalidConfiguration(&'static str),
    NonFiniteSample(usize),
    SourceTooShort,
    NoFrames,
}

impl fmt::Display for DspError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(reason) => {
                write!(formatter, "invalid DSP configuration: {reason}")
            }
            Self::NonFiniteSample(index) => {
                write!(formatter, "source contains a non-finite sample at index {index}")
            }
            Self::SourceTooShort => write!(formatter, "source is shorter than the analysis window"),
            Self::NoFrames => write!(formatter, "analysis produced no complete frames"),
        }
    }
}

impl Error for DspError {}

/// Half-open source-sample support for one analysis frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSupport {
    pub start_sample: usize,
    pub end_sample: usize,
}

/// Configuration for the phase-preserving short-time Fourier transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StftConfig {
    pub sample_rate_hz: u32,
    pub window_size: usize,
    pub hop_size: usize,
    pub fft_size: usize,
}

/// One complex, one-sided STFT frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexStftFrame {
    pub support: FrameSupport,
    /// DC through Nyquist, inclusive.
    pub bins: Vec<Complex64>,
}

/// Complex one-sided STFT with explicit analysis configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexStft {
    pub config: StftConfig,
    pub frames: Vec<ComplexStftFrame>,
}

impl ComplexStft {
    /// Frequency in hertz represented by a one-sided bin.
    pub fn bin_frequency_hz(&self, bin_index: usize) -> Option<f64> {
        if bin_index >= self.config.fft_size / 2 + 1 {
            return None;
        }
        Some(
            bin_index as f64 * f64::from(self.config.sample_rate_hz)
                / self.config.fft_size as f64,
        )
    }
}

/// Calculate a deterministic Hann-windowed, phase-preserving STFT.
///
/// Only complete windows are emitted. The one-sided coefficients are scaled
/// to recover the peak amplitude of an isolated bin-centred sinusoid.
pub fn extract_complex_stft(
    samples: &[f32],
    config: StftConfig,
) -> Result<ComplexStft, DspError> {
    if config.sample_rate_hz == 0 {
        return Err(DspError::InvalidConfiguration("sample rate must be positive"));
    }
    if config.window_size < 3 {
        return Err(DspError::InvalidConfiguration("window size must be at least three samples"));
    }
    if config.hop_size == 0 || config.hop_size > config.window_size {
        return Err(DspError::InvalidConfiguration("hop size must be in 1..=window_size"));
    }
    if config.fft_size < config.window_size || !config.fft_size.is_power_of_two() {
        return Err(DspError::InvalidConfiguration(
            "FFT size must be a power of two and at least the window size",
        ));
    }
    if samples.len() < config.window_size {
        return Err(DspError::SourceTooShort);
    }
    if let Some((index, _)) = samples.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(DspError::NonFiniteSample(index));
    }

    let window: Vec<f64> = (0..config.window_size)
        .map(|index| {
            0.5 * (1.0 - (TAU * index as f64 / (config.window_size - 1) as f64).cos())
        })
        .collect();
    let window_sum: f64 = window.iter().sum();
    if !window_sum.is_finite() || window_sum <= f64::EPSILON {
        return Err(DspError::InvalidConfiguration("analysis window has zero coherent gain"));
    }

    let mut frames = Vec::new();
    let max_start = samples.len() - config.window_size;
    for start in (0..=max_start).step_by(config.hop_size) {
        let mut spectrum = vec![Complex64::default(); config.fft_size];
        for (index, (&sample, &weight)) in samples[start..start + config.window_size]
            .iter()
            .zip(window.iter())
            .enumerate()
        {
            spectrum[index] = Complex64::new(f64::from(sample) * weight, 0.0);
        }
        fft_in_place(&mut spectrum)?;

        let mut bins = Vec::with_capacity(config.fft_size / 2 + 1);
        for (index, coefficient) in spectrum
            .iter()
            .take(config.fft_size / 2 + 1)
            .copied()
            .enumerate()
        {
            let is_edge_bin = index == 0 || index == config.fft_size / 2;
            let scale = if is_edge_bin {
                1.0 / window_sum
            } else {
                2.0 / window_sum
            };
            bins.push(coefficient.scale(scale));
        }
        frames.push(ComplexStftFrame {
            support: FrameSupport {
                start_sample: start,
                end_sample: start + config.window_size,
            },
            bins,
        });
    }

    if frames.is_empty() {
        return Err(DspError::NoFrames);
    }
    Ok(ComplexStft { config, frames })
}

fn fft_in_place(values: &mut [Complex64]) -> Result<(), DspError> {
    let length = values.len();
    if length < 2 || !length.is_power_of_two() {
        return Err(DspError::InvalidConfiguration(
            "internal FFT length must be a power of two",
        ));
    }

    let mut reversed = 0usize;
    for index in 1..length {
        let mut bit = length >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut block_size = 2usize;
    loop {
        let angle = -TAU / block_size as f64;
        let twiddle_step = Complex64::new(angle.cos(), angle.sin());
        let half_block = block_size / 2;

        for block_start in (0..length).step_by(block_size) {
            let mut twiddle = Complex64::new(1.0, 0.0);
            for offset in 0..half_block {
                let even_index = block_start + offset;
                let odd_index = even_index + half_block;
                let even = values[even_index];
                let odd = values[odd_index].multiply(twiddle);
                values[even_index] = even.add(odd);
                values[odd_index] = even.sub(odd);
                twiddle = twiddle.multiply(twiddle_step);
            }
        }

        if block_size == length {
            break;
        }
        block_size *= 2;
    }
    Ok(())
}

/// Configuration for an ERB-rate triangular power partition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErbConfig {
    pub band_count: usize,
    pub min_hz: f64,
    pub max_hz: f64,
}

/// One ERB-rate band-energy frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ErbEnergyFrame {
    pub support: FrameSupport,
    pub band_energies: Vec<f64>,
}

/// ERB-rate centres and their aligned frame energies.
#[derive(Debug, Clone, PartialEq)]
pub struct ErbEnergy {
    pub center_frequencies_hz: Vec<f64>,
    pub frames: Vec<ErbEnergyFrame>,
}

/// Convert frequency in hertz to the commonly used Glasberg-Moore ERB-rate
/// approximation. This coordinate transform does not itself model cochlear
/// filtering.
pub fn hz_to_erb_rate(frequency_hz: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * frequency_hz).log10()
}

/// Compute mean one-sided spectral power in triangular ERB-rate bands.
///
/// This is deliberately a transparent ERB-axis filterbank baseline, not a
/// gammatone filterbank or a complete model of human hearing.
pub fn compute_erb_energy(
    stft: &ComplexStft,
    config: ErbConfig,
) -> Result<ErbEnergy, DspError> {
    let nyquist = f64::from(stft.config.sample_rate_hz) / 2.0;
    if config.band_count < 2 {
        return Err(DspError::InvalidConfiguration("ERB band count must be at least two"));
    }
    if !config.min_hz.is_finite()
        || !config.max_hz.is_finite()
        || config.min_hz <= 0.0
        || config.max_hz <= config.min_hz
        || config.max_hz > nyquist
    {
        return Err(DspError::InvalidConfiguration(
            "ERB frequency range must be finite, positive, ordered and below Nyquist",
        ));
    }

    let erb_min = hz_to_erb_rate(config.min_hz);
    let erb_max = hz_to_erb_rate(config.max_hz);
    let erb_step = (erb_max - erb_min) / (config.band_count - 1) as f64;
    let center_frequencies_hz = (0..config.band_count)
        .map(|index| erb_rate_to_hz(erb_min + index as f64 * erb_step))
        .collect::<Vec<_>>();

    let mut frames = Vec::with_capacity(stft.frames.len());
    for frame in &stft.frames {
        let mut energies = vec![0.0f64; config.band_count];
        let mut weights = vec![0.0f64; config.band_count];

        for (bin_index, coefficient) in frame.bins.iter().enumerate() {
            let frequency_hz = stft.bin_frequency_hz(bin_index).unwrap_or(0.0);
            if frequency_hz < config.min_hz || frequency_hz > config.max_hz {
                continue;
            }
            let coordinate = ((hz_to_erb_rate(frequency_hz) - erb_min) / erb_step)
                .clamp(0.0, (config.band_count - 1) as f64);
            let lower = coordinate.floor() as usize;
            let upper = (lower + 1).min(config.band_count - 1);
            let fraction = coordinate - lower as f64;
            let power = coefficient.power();

            let lower_weight = if upper == lower { 1.0 } else { 1.0 - fraction };
            energies[lower] += power * lower_weight;
            weights[lower] += lower_weight;

            if upper != lower {
                energies[upper] += power * fraction;
                weights[upper] += fraction;
            }
        }

        for (energy, weight) in energies.iter_mut().zip(weights.iter()) {
            if *weight > 0.0 {
                *energy /= *weight;
            }
        }
        frames.push(ErbEnergyFrame {
            support: frame.support,
            band_energies: energies,
        });
    }

    Ok(ErbEnergy {
        center_frequencies_hz,
        frames,
    })
}

fn erb_rate_to_hz(erb_rate: f64) -> f64 {
    (10.0f64.powf(erb_rate / 21.4) - 1.0) / 0.00437
}

/// Configuration for logarithmically spaced frequency-energy bands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogFrequencyConfig {
    pub bins_per_octave: usize,
    pub min_hz: f64,
    pub max_hz: f64,
}

/// One log-frequency energy frame.
#[derive(Debug, Clone, PartialEq)]
pub struct LogFrequencyFrame {
    pub support: FrameSupport,
    pub band_energies: Vec<f64>,
}

/// Log-frequency band centres and their aligned frame energies.
#[derive(Debug, Clone, PartialEq)]
pub struct LogFrequencyEnergy {
    pub center_frequencies_hz: Vec<f64>,
    pub frames: Vec<LogFrequencyFrame>,
}

/// Compute a logarithmic frequency-energy representation by partitioning
/// one-sided STFT power between adjacent log-frequency centres.
///
/// This view is useful for pitch-oriented analysis but is not a true
/// constant-Q transform: it inherits the fixed linear-frequency STFT grid.
pub fn compute_log_frequency_energy(
    stft: &ComplexStft,
    config: LogFrequencyConfig,
) -> Result<LogFrequencyEnergy, DspError> {
    let nyquist = f64::from(stft.config.sample_rate_hz) / 2.0;
    if config.bins_per_octave == 0 || config.bins_per_octave > 96 {
        return Err(DspError::InvalidConfiguration(
            "log-frequency bins per octave must be in 1..=96",
        ));
    }
    if !config.min_hz.is_finite()
        || !config.max_hz.is_finite()
        || config.min_hz <= 0.0
        || config.max_hz <= config.min_hz
        || config.max_hz > nyquist
    {
        return Err(DspError::InvalidConfiguration(
            "log-frequency range must be finite, positive, ordered and below Nyquist",
        ));
    }

    let octave_span = (config.max_hz / config.min_hz).log2();
    let band_count = (octave_span * config.bins_per_octave as f64).floor() as usize + 1;
    if band_count < 2 || band_count > 4096 {
        return Err(DspError::InvalidConfiguration(
            "log-frequency configuration produces an unsupported number of bands",
        ));
    }
    let center_frequencies_hz = (0..band_count)
        .map(|index| config.min_hz * 2.0f64.powf(index as f64 / config.bins_per_octave as f64))
        .collect::<Vec<_>>();
    let log_min = config.min_hz.log2();
    let bin_step = 1.0 / config.bins_per_octave as f64;

    let mut frames = Vec::with_capacity(stft.frames.len());
    for frame in &stft.frames {
        let mut energies = vec![0.0f64; band_count];
        let mut weights = vec![0.0f64; band_count];

        for (bin_index, coefficient) in frame.bins.iter().enumerate() {
            let frequency_hz = stft.bin_frequency_hz(bin_index).unwrap_or(0.0);
            if frequency_hz < config.min_hz || frequency_hz > config.max_hz {
                continue;
            }
            let coordinate = ((frequency_hz.log2() - log_min) / bin_step)
                .clamp(0.0, (band_count - 1) as f64);
            let lower = coordinate.floor() as usize;
            let upper = (lower + 1).min(band_count - 1);
            let fraction = coordinate - lower as f64;
            let power = coefficient.power();

            let lower_weight = if upper == lower { 1.0 } else { 1.0 - fraction };
            energies[lower] += power * lower_weight;
            weights[lower] += lower_weight;

            if upper != lower {
                energies[upper] += power * fraction;
                weights[upper] += fraction;
            }
        }

        for (energy, weight) in energies.iter_mut().zip(weights.iter()) {
            if *weight > 0.0 {
                *energy /= *weight;
            }
        }
        frames.push(LogFrequencyFrame {
            support: frame.support,
            band_energies: energies,
        });
    }

    Ok(LogFrequencyEnergy {
        center_frequencies_hz,
        frames,
    })
}

/// Configuration and aligned outputs for the three materialized baseline
/// views in the first DSP tranche.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiViewConfig {
    pub stft: StftConfig,
    pub erb: ErbConfig,
    pub log_frequency: LogFrequencyConfig,
}

/// Joint output. Each view retains its own numerical representation while
/// sharing exact source-frame support.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiViewAnalysis {
    pub complex_stft: ComplexStft,
    pub erb_energy: ErbEnergy,
    pub log_frequency_energy: LogFrequencyEnergy,
}

/// Compute phase-preserving STFT, ERB-axis power and log-frequency power views.
pub fn analyze_multi_view(
    samples: &[f32],
    config: MultiViewConfig,
) -> Result<MultiViewAnalysis, DspError> {
    let complex_stft = extract_complex_stft(samples, config.stft)?;
    let erb_energy = compute_erb_energy(&complex_stft, config.erb)?;
    let log_frequency_energy =
        compute_log_frequency_energy(&complex_stft, config.log_frequency)?;
    Ok(MultiViewAnalysis {
        complex_stft,
        erb_energy,
        log_frequency_energy,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        analyze_multi_view, compute_erb_energy, compute_log_frequency_energy,
        extract_complex_stft, ComplexStft, DspError, ErbConfig, LogFrequencyConfig,
        MultiViewConfig, StftConfig,
    };

    fn tone(sample_rate_hz: u32, frequency_hz: f64, sample_count: usize) -> Vec<f32> {
        (0..sample_count)
            .map(|index| {
                (std::f64::consts::TAU * frequency_hz * index as f64
                    / f64::from(sample_rate_hz))
                    .sin() as f32
            })
            .collect()
    }

    fn stft_for_tone(samples: &[f32]) -> ComplexStft {
        extract_complex_stft(
            samples,
            StftConfig {
                sample_rate_hz: 8192,
                window_size: 1024,
                hop_size: 1024,
                fft_size: 1024,
            },
        )
        .unwrap()
    }

    #[test]
    fn stft_retains_complex_phase_and_recovers_bin_centered_tone_amplitude() {
        let samples = tone(8192, 1000.0, 1024);
        let result = stft_for_tone(&samples);
        let frame = &result.frames[0];
        let (peak_index, peak) = frame
            .bins
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .magnitude()
                    .total_cmp(&right.1.magnitude())
            })
            .unwrap();

        assert_eq!(peak_index, 125);
        assert!((peak.magnitude() - 1.0).abs() < 1.0e-5);
        assert!(peak.im < -0.99);
        assert!(peak.re.abs() < 1.0e-4);
        assert_eq!(frame.support.start_sample, 0);
        assert_eq!(frame.support.end_sample, 1024);
    }

    #[test]
    fn stft_rejects_invalid_fft_geometry_and_non_finite_samples() {
        let invalid = StftConfig {
            sample_rate_hz: 8192,
            window_size: 1024,
            hop_size: 512,
            fft_size: 1000,
        };
        assert!(matches!(
            extract_complex_stft(&vec![0.0; 1024], invalid),
            Err(DspError::InvalidConfiguration(_))
        ));

        let mut samples = vec![0.0; 1024];
        samples[17] = f32::NAN;
        assert_eq!(
            extract_complex_stft(
                &samples,
                StftConfig {
                    fft_size: 1024,
                    ..invalid
                }
            ),
            Err(DspError::NonFiniteSample(17))
        );
    }

    #[test]
    fn erb_energy_localizes_a_pure_tone_on_the_erb_axis() {
        let result = stft_for_tone(&tone(8192, 1000.0, 8192));
        let erb = compute_erb_energy(
            &result,
            ErbConfig {
                band_count: 24,
                min_hz: 100.0,
                max_hz: 4000.0,
            },
        )
        .unwrap();
        let frame = &erb.frames[0];
        let peak_band = frame
            .band_energies
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap();
        assert!((erb.center_frequencies_hz[peak_band] - 1000.0).abs() < 250.0);
    }

    #[test]
    fn log_frequency_energy_localizes_a_pure_tone() {
        let result = stft_for_tone(&tone(8192, 1000.0, 8192));
        let log_frequency = compute_log_frequency_energy(
            &result,
            LogFrequencyConfig {
                bins_per_octave: 12,
                min_hz: 125.0,
                max_hz: 4000.0,
            },
        )
        .unwrap();
        let frame = &log_frequency.frames[0];
        let peak_band = frame
            .band_energies
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap();
        assert!((log_frequency.center_frequencies_hz[peak_band] - 1000.0).abs() < 1.0);
    }

    #[test]
    fn multiview_outputs_share_exact_source_frame_support() {
        let samples = tone(8192, 1000.0, 8192);
        let analysis = analyze_multi_view(
            &samples,
            MultiViewConfig {
                stft: StftConfig {
                    sample_rate_hz: 8192,
                    window_size: 1024,
                    hop_size: 256,
                    fft_size: 1024,
                },
                erb: ErbConfig {
                    band_count: 24,
                    min_hz: 100.0,
                    max_hz: 4000.0,
                },
                log_frequency: LogFrequencyConfig {
                    bins_per_octave: 12,
                    min_hz: 125.0,
                    max_hz: 4000.0,
                },
            },
        )
        .unwrap();

        assert_eq!(
            analysis.complex_stft.frames.len(),
            analysis.erb_energy.frames.len()
        );
        assert_eq!(
            analysis.complex_stft.frames.len(),
            analysis.log_frequency_energy.frames.len()
        );
        for ((stft_frame, erb_frame), log_frame) in analysis
            .complex_stft
            .frames
            .iter()
            .zip(analysis.erb_energy.frames.iter())
            .zip(analysis.log_frequency_energy.frames.iter())
        {
            assert_eq!(erb_frame.support, stft_frame.support);
            assert_eq!(log_frame.support, stft_frame.support);
        }
    }
}
