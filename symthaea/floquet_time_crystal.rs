// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use rustfft::{num_complex::Complex, FftPlanner};

/// Toy Floquet/discrete-time-crystal-like simulator.
///
/// This is not a physical quantum simulator. It is an exploratory classical surrogate
/// for testing whether Symthaea can detect persistent period-doubled response.
pub struct TimeCrystalSimulator {
    n_spins: usize,
    spins: Vec<f64>,
}

impl TimeCrystalSimulator {
    pub fn new(n_spins: usize) -> Self {
        let n_spins = n_spins.max(1);
        let spins = (0..n_spins)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        Self { n_spins, spins }
    }

    /// Step the simulation with an imperfect periodic drive.
    ///
    /// `epsilon = 0` is a perfect pi-pulse. Larger epsilon mixes the flipped state
    /// with the pre-drive state. The update is saturating so that the toy DTC phase
    /// can persist instead of trivially decaying.
    pub fn step(&mut self, epsilon: f64, j_interaction: f64) {
        let epsilon = epsilon.clamp(0.0, 1.0);
        let j_interaction = j_interaction.clamp(-1.0, 1.0);

        let driven: Vec<f64> = self
            .spins
            .iter()
            .map(|&spin| {
                let flipped = -spin.signum();
                let retained = spin.signum();
                ((1.0 - epsilon) * flipped + epsilon * retained).clamp(-1.0, 1.0)
            })
            .collect();

        let mut next_spins = driven.clone();

        for i in 0..self.n_spins {
            let prev = if i == 0 { self.n_spins - 1 } else { i - 1 };
            let next = (i + 1) % self.n_spins;
            let local_field = 0.5 * (driven[prev] + driven[next]);
            let updated = driven[i] + j_interaction * local_field;

            next_spins[i] = updated.tanh().clamp(-1.0, 1.0);
        }

        self.spins = next_spins;
    }

    /// Global magnetization. This is often zero for antiferromagnetic states.
    pub fn magnetization(&self) -> f64 {
        self.spins.iter().sum::<f64>() / self.n_spins as f64
    }

    /// Staggered magnetization, a better order parameter for alternating spin chains.
    pub fn staggered_magnetization(&self) -> f64 {
        self.spins
            .iter()
            .enumerate()
            .map(|(i, &spin)| if i % 2 == 0 { spin } else { -spin })
            .sum::<f64>()
            / self.n_spins as f64
    }

    pub fn signal(&mut self, steps: usize, epsilon: f64, j_interaction: f64) -> Vec<f64> {
        let mut signal = Vec::with_capacity(steps);

        for _ in 0..steps {
            signal.push(self.staggered_magnetization());
            self.step(epsilon, j_interaction);
        }

        signal
    }
}

/// Detector for subharmonic response and persistence.
pub struct TimeCrystalDetector;

impl TimeCrystalDetector {
    /// Fraction of spectral power concentrated at the period-2 / Nyquist bin.
    pub fn subharmonic_score(&self, signal: &[f64]) -> f64 {
        let n = signal.len();

        if n < 4 || n % 2 != 0 {
            return 0.0;
        }

        let spectrum = fft(signal);
        let subharmonic_idx = n / 2;
        let subharmonic_power = spectrum[subharmonic_idx].norm_sqr();

        let total_power: f64 = spectrum
            .iter()
            .skip(1)
            .take(n / 2)
            .map(|c| c.norm_sqr())
            .sum();

        if total_power > f64::EPSILON {
            subharmonic_power / total_power
        } else {
            0.0
        }
    }

    /// Absolute period-2 amplitude. Unlike subharmonic concentration, this falls when
    /// a damped alternating signal dies out.
    pub fn subharmonic_amplitude(&self, signal: &[f64]) -> f64 {
        let n = signal.len();

        if n < 4 || n % 2 != 0 {
            return 0.0;
        }

        let spectrum = fft(signal);
        spectrum[n / 2].norm() / n as f64
    }

    /// Persistence = late-window subharmonic amplitude / early-window amplitude.
    pub fn persistence_score(&self, signal: &[f64]) -> f64 {
        let n = signal.len();

        if n < 16 {
            return 0.0;
        }

        let half = n / 2;
        let early = self.subharmonic_amplitude(&signal[..half]);
        let late = self.subharmonic_amplitude(&signal[half..]);

        if early > f64::EPSILON {
            (late / early).clamp(0.0, 2.0)
        } else {
            0.0
        }
    }

    /// Combined score: period-2 concentration times persistence.
    pub fn time_crystal_likeness(&self, signal: &[f64]) -> f64 {
        self.subharmonic_score(signal) * self.persistence_score(signal)
    }
}

fn fft(signal: &[f64]) -> Vec<Complex<f64>> {
    let n = signal.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();

    fft.process(&mut buffer);
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staggered_magnetization_nonzero_for_antiferromagnet() {
        let sim = TimeCrystalSimulator::new(10);

        assert!(sim.magnetization().abs() < 1e-9);
        assert!(sim.staggered_magnetization().abs() > 0.9);
    }

    #[test]
    fn test_time_crystal_staggered_signal_is_persistent() {
        let mut sim = TimeCrystalSimulator::new(10);
        let signal = sim.signal(128, 0.0, 0.05);

        let detector = TimeCrystalDetector;

        assert!(detector.subharmonic_score(&signal) > 0.7);
        assert!(detector.persistence_score(&signal) > 0.5);
        assert!(detector.time_crystal_likeness(&signal) > 0.4);
    }

    #[test]
    fn test_constant_signal_has_no_subharmonic_score() {
        let signal = vec![1.0; 128];
        let detector = TimeCrystalDetector;

        assert!(detector.subharmonic_score(&signal) < 0.01);
    }

    #[test]
    fn test_damped_oscillator_fails_persistence() {
        let mut signal = Vec::new();
        let mut val = 1.0;

        for _ in 0..128 {
            signal.push(val);
            val *= -0.80;
        }

        let detector = TimeCrystalDetector;

        assert!(detector.subharmonic_score(&signal) > 0.5);
        assert!(detector.persistence_score(&signal) < 0.25);
    }
}
