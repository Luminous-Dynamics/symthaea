//! Independent actual-CPP candidate for comparison with the locked Praat reference.

use anyhow::{Context, Result, bail};
use rustfft::{FftPlanner, num_complex::Complex};

const ANALYSIS_RATE: usize = 10_000;
const FFT_SIZE: usize = 1024;

fn read_mono(path: &str) -> Result<(Vec<f64>, usize)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        bail!("expected mono 16-bit PCM WAV");
    }
    let samples = reader
        .samples::<i16>()
        .map(|sample| Ok(sample? as f64 / 32768.0))
        .collect::<Result<Vec<_>>>()?;
    Ok((samples, spec.sample_rate as usize))
}

fn resample_sinc50(input: &[f64], input_rate: usize) -> Vec<f64> {
    if input_rate == ANALYSIS_RATE {
        return input.to_vec();
    }
    let output_len = input.len() * ANALYSIS_RATE / input_rate;
    let rate_ratio = ANALYSIS_RATE as f64 / input_rate as f64;
    let cutoff = rate_ratio.min(1.0);
    let radius = 50_i64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 / rate_ratio;
            let centre = position.floor() as i64;
            let mut weighted = 0.0;
            let mut normalization = 0.0;
            for source_index in centre - radius + 1..=centre + radius {
                if source_index < 0 || source_index >= input.len() as i64 {
                    continue;
                }
                let distance = position - source_index as f64;
                let phase = std::f64::consts::PI * cutoff * distance;
                let sinc = if phase.abs() < 1e-12 {
                    cutoff
                } else {
                    cutoff * phase.sin() / phase
                };
                let window = 0.5
                    + 0.5
                        * (std::f64::consts::PI * distance / radius as f64)
                            .cos()
                            .max(-1.0);
                let weight = sinc * window;
                weighted += input[source_index as usize] * weight;
                normalization += weight;
            }
            weighted / normalization.max(1e-12)
        })
        .collect()
}

fn smooth(values: &[Vec<f64>], time_bins: usize, quefrency_bins: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; values[0].len()]; values.len()];
    for (time, row) in result.iter_mut().enumerate() {
        let t0 = time.saturating_sub(time_bins / 2);
        let t1 = (time + time_bins / 2 + 1).min(values.len());
        for (quefrency, value) in row.iter_mut().enumerate() {
            let q0 = quefrency.saturating_sub(quefrency_bins / 2);
            let q1 = (quefrency + quefrency_bins / 2).min(values[0].len() - 1);
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for source in &values[t0..t1] {
                for (index, sample) in source[q0..=q1].iter().enumerate() {
                    // Sampled_getMean integrates to the exact half-window;
                    // samples on its two boundaries contribute half weight.
                    let weight =
                        if (q0 > 0 && index == 0) || (q1 + 1 < source.len() && q0 + index == q1) {
                            0.5
                        } else {
                            1.0
                        };
                    sum += weight * sample;
                    weight_sum += weight;
                }
            }
            *value = sum / weight_sum;
        }
    }
    result
}

fn robust_line(points: &[(f64, f64)]) -> (f64, f64) {
    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        let middle = values.len() / 2;
        if values.len() % 2 == 0 {
            0.5 * (values[middle - 1] + values[middle])
        } else {
            values[middle]
        }
    }

    // Praat's "Robust" trend is the incomplete Theil estimator: split the
    // ordered points in half, pair corresponding points, then take the median
    // of those slopes. This deliberately reimplements, rather than calls,
    // Praat so cross-implementation agreement remains meaningful.
    let half = points.len() / 2;
    let mut slopes: Vec<_> = (0..half)
        .filter_map(|index| {
            let left = points[index];
            let right = points[half + index];
            let dx = right.0 - left.0;
            (dx > 0.0).then_some((right.1 - left.1) / dx)
        })
        .collect();
    let slope = median(&mut slopes);
    let mut intercepts: Vec<_> = points.iter().map(|(x, y)| y - slope * x).collect();
    (median(&mut intercepts), slope)
}

fn cpp_frames(path: &str) -> Result<Vec<(f64, f64)>> {
    let (input, input_rate) = read_mono(path)?;
    let mut signal = resample_sinc50(&input, input_rate);
    let alpha = (-2.0 * std::f64::consts::PI * 50.0 / ANALYSIS_RATE as f64).exp();
    for index in (1..signal.len()).rev() {
        signal[index] -= alpha * signal[index - 1];
    }
    // Praat uses a Gaussian window twice the three-period analysis width.
    let frame_len = ANALYSIS_RATE / 10; // 2 * 3 / 60 Hz = 100 ms
    let hop = ANALYSIS_RATE / 500; // 2 ms
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(FFT_SIZE);
    let inverse = planner.plan_fft_inverse(FFT_SIZE);
    let mut frames = Vec::new();
    for start in (0..=signal.len().saturating_sub(frame_len)).step_by(hop) {
        let mut spectrum = vec![Complex::new(0.0, 0.0); FFT_SIZE];
        let mean = signal[start..start + frame_len].iter().sum::<f64>() / frame_len as f64;
        for index in 0..frame_len {
            let x = index as f64 / (frame_len - 1) as f64 - 0.5;
            let gaussian = (-48.0 * x * x).exp();
            spectrum[index].re = (signal[start + index] - mean) * gaussian;
        }
        forward.process(&mut spectrum);
        for bin in &mut spectrum {
            bin.re = bin.norm_sqr().max(1e-30).ln();
            bin.im = 0.0;
        }
        inverse.process(&mut spectrum);
        let normalizer = FFT_SIZE as f64;
        frames.push(
            spectrum
                .iter()
                .take(FFT_SIZE / 2)
                .map(|value| (value.re / normalizer).powi(2))
                .collect::<Vec<_>>(),
        );
    }
    if frames.is_empty() {
        bail!("recording is too short for CPP analysis");
    }
    // 10-ms temporal and 1-ms quefrency averaging windows.
    let frames = smooth(&frames, 5, 10);
    let q_min = (ANALYSIS_RATE as f64 / 330.0).ceil() as usize;
    let q_max = (ANALYSIS_RATE as f64 / 60.0).floor() as usize;
    let trend_min = (0.001 * ANALYSIS_RATE as f64).ceil() as usize;
    let trend_max = FFT_SIZE / 2 - 1;
    let mut prominences = Vec::new();
    for frame in frames {
        let db: Vec<_> = frame
            .iter()
            .map(|power| 10.0 * power.max(1e-30).log10())
            .collect();
        let peak = (q_min..=q_max)
            .max_by(|left, right| db[*left].total_cmp(&db[*right]))
            .context("empty pitch-search region")?;
        let (peak_position, peak_db) = if peak > 0 && peak + 1 < db.len() {
            let left = db[peak - 1];
            let centre = db[peak];
            let right = db[peak + 1];
            let curvature = left - 2.0 * centre + right;
            if curvature.abs() > 1e-12 {
                let offset = (0.5 * (left - right) / curvature).clamp(-1.0, 1.0);
                (
                    peak as f64 + offset,
                    centre - 0.25 * (left - right) * offset,
                )
            } else {
                (peak as f64, centre)
            }
        } else {
            (peak as f64, db[peak])
        };
        let points: Vec<_> = (trend_min..=trend_max)
            .map(|index| (index as f64 / ANALYSIS_RATE as f64, db[index]))
            .collect();
        let (intercept, slope) = robust_line(&points);
        let q = peak_position / ANALYSIS_RATE as f64;
        prominences.push((q, peak_db - (intercept + slope * q)));
    }
    Ok(prominences)
}

fn cpps(path: &str) -> Result<f64> {
    let frames = cpp_frames(path)?;
    Ok(frames.iter().map(|(_, prominence)| prominence).sum::<f64>() / frames.len() as f64)
}

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.first().map(String::as_str) == Some("--frames") {
        let path = arguments
            .get(1)
            .context("usage: vowel_truth_cpp --frames INPUT.wav")?;
        println!("frame\tquefrency_s\tcpp_db");
        for (index, (quefrency, prominence)) in cpp_frames(path)?.iter().enumerate() {
            println!("{}\t{:.9}\t{:.16}", index + 1, quefrency, prominence);
        }
    } else {
        let path = arguments
            .first()
            .context("usage: vowel_truth_cpp INPUT.wav")?;
        println!("{:.6}", cpps(path)?);
    }
    Ok(())
}
