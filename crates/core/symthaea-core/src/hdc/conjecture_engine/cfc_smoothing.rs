// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC-style closed-form smoothing for noisy/irregularly-sampled observations.
//!
//! Fourth capability in the agreed longer sequence for the Ramanujan Protocol (HDC →
//! constrained physical reasoning → FEP active-experiment-selection → **this** → IIT).
//! Every prior discovery pass in this arc (Stage A, Stage B/M2, both closed) fit against
//! clean, exactly-integrated, uniformly-sampled synthetic trajectories -- the easy case,
//! deliberately, per the agreed sequence ("CfC for noisy/irregular real observations,
//! deliberately after clean-data discovery works, not before"). Real observations (or a
//! future non-synthetic data source) won't have either property: sampling times are
//! irregular, and values carry measurement noise.
//!
//! ## Why not reuse `symthaea_hdc_ltc::HdcLtcUnifiedNeuron` directly
//!
//! `symthaea-hdc-ltc` already implements the real closed-form continuous-time (CfC) update
//! this module is named after: `sigma = 1 - exp(-dt/tau)`, `state <- (1-sigma)*state +
//! sigma*x_inf`, letting a neuron jump to *any* time horizon in O(1) instead of stepping an
//! ODE integrator. That crate's neuron is real, tested production code -- but it operates on
//! 16,384-dimensional `ContinuousHV` hypervectors with weight-hypervector-driven dynamics
//! (`x_inf = f(W . x + U . u)`, HDC binding), built for the cognitive loop's own
//! representation. Smoothing a handful of noisy scalar physical observations (e.g. a
//! wave-chain-style `[u1..un, v1..vn]` state) through that machinery would mean constructing
//! an inappropriate 16,384-D embedding just to filter a few numbers -- the same domain-
//! mismatch already found and documented for `symthaea-fep`'s `ExpectedFreeEnergyComputer`
//! in `experiment_selection.rs`. What's reused here is the **closed-form update rule
//! itself** (the actual CfC mathematics, cited and applied honestly), not the HDC neuron
//! code -- this module implements it directly over low-dimensional `Vec<f64>` state, which
//! is what physical-observation smoothing actually needs.
//!
//! ## Why closed-form matters for *irregular* sampling specifically
//!
//! A naive fixed-`alpha` exponential moving average (`state <- (1-alpha)*state +
//! alpha*obs`) implicitly assumes uniform sampling -- the same `alpha` is applied whether
//! the last observation was 0.01s or 10s ago, systematically over-trusting old estimates
//! after a long gap and over-reacting to noise after a short one. The closed-form update
//! recomputes the *effective* mixing weight `sigma = 1 - exp(-dt/tau)` from the actual
//! elapsed time every step, so it responds identically to observations regardless of
//! whether they arrive on a regular grid or not. See
//! `cfc_beats_naive_ema_under_irregular_sampling` for the measured effect.

/// Sequential CfC-style closed-form state estimator. Absorbs one observation at a time,
/// `dt` seconds after the previous one, and produces a smoothed running estimate. See
/// module docs for why this is dt-aware in a way a fixed-`alpha` EMA is not.
pub struct CfcSmoother {
    state: Vec<f64>,
    /// Time constant: larger = more smoothing (slower to trust a new observation).
    tau: f64,
    initialized: bool,
}

impl CfcSmoother {
    pub fn new(dim: usize, tau: f64) -> Self {
        Self {
            state: vec![0.0; dim],
            tau,
            initialized: false,
        }
    }

    /// Absorb a new observation. The very first call initializes the state to that
    /// observation directly (`dt` is ignored -- there is no prior estimate to blend with).
    pub fn observe(&mut self, dt: f64, observation: &[f64]) {
        debug_assert_eq!(observation.len(), self.state.len());
        if !self.initialized {
            self.state.copy_from_slice(observation);
            self.initialized = true;
            return;
        }
        let sigma = 1.0 - (-dt / self.tau).exp();
        for (s, &o) in self.state.iter_mut().zip(observation) {
            *s = (1.0 - sigma) * *s + sigma * o;
        }
    }

    pub fn state(&self) -> &[f64] {
        &self.state
    }
}

/// Naive fixed-`alpha` exponential moving average, ignoring `dt` entirely -- the baseline
/// [`CfcSmoother`]'s diagnostic compares against. Same interface deliberately, so the two
/// can be driven by identical observation sequences for a fair comparison.
pub struct NaiveEmaSmoother {
    state: Vec<f64>,
    alpha: f64,
    initialized: bool,
}

impl NaiveEmaSmoother {
    pub fn new(dim: usize, alpha: f64) -> Self {
        Self {
            state: vec![0.0; dim],
            alpha,
            initialized: false,
        }
    }

    pub fn observe(&mut self, observation: &[f64]) {
        debug_assert_eq!(observation.len(), self.state.len());
        if !self.initialized {
            self.state.copy_from_slice(observation);
            self.initialized = true;
            return;
        }
        for (s, &o) in self.state.iter_mut().zip(observation) {
            *s = (1.0 - self.alpha) * *s + self.alpha * o;
        }
    }

    pub fn state(&self) -> &[f64] {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift_next(rng: &mut u64) -> u64 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        *rng
    }

    fn rand_unit(rng: &mut u64) -> f64 {
        (xorshift_next(rng) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately-Gaussian noise via Box-Muller, using the module's own small RNG
    /// (matching this crate family's established no-new-dependency convention).
    fn rand_gaussian(rng: &mut u64, sigma: f64) -> f64 {
        let u1 = rand_unit(rng).max(1e-12);
        let u2 = rand_unit(rng);
        sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Simple harmonic oscillator ground truth: `x(t) = A cos(wt+phi)`, `v(t) =
    /// -A*w*sin(wt+phi)`. Deliberately a fresh, simple analytic toy domain -- not the
    /// closed wave-chain problem -- chosen because it has a known-exact closed-form
    /// solution (no numerical integrator needed to generate ground truth) and a
    /// trivially-checkable conserved quantity, `E = 0.5*v^2 + 0.5*w^2*x^2`.
    fn sho_state(t: f64, amplitude: f64, omega: f64, phase: f64) -> (f64, f64) {
        let x = amplitude * (omega * t + phase).cos();
        let v = -amplitude * omega * (omega * t + phase).sin();
        (x, v)
    }

    fn sho_energy(x: f64, v: f64, omega: f64) -> f64 {
        0.5 * v * v + 0.5 * omega * omega * x * x
    }

    fn variance(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    /// Generate `n` irregularly-spaced sample times in `[0, t_max]` (uniform random gaps,
    /// not a fixed grid), plus Gaussian-noised `(x, v)` observations at each.
    fn irregular_noisy_samples(
        rng: &mut u64,
        n: usize,
        t_max: f64,
        amplitude: f64,
        omega: f64,
        phase: f64,
        noise_sigma: f64,
    ) -> Vec<(f64, [f64; 2])> {
        let mut times: Vec<f64> = (0..n).map(|_| rand_unit(rng) * t_max).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times
            .into_iter()
            .map(|t| {
                let (x, v) = sho_state(t, amplitude, omega, phase);
                let noisy = [
                    x + rand_gaussian(rng, noise_sigma),
                    v + rand_gaussian(rng, noise_sigma),
                ];
                (t, noisy)
            })
            .collect()
    }

    #[test]
    fn cfc_smoother_reduces_energy_residual_vs_raw_noisy_observations() {
        let (amplitude, omega, phase) = (1.0, 1.5, 0.3);
        let mut rng = 0xC0FF_EE01_u64;
        let samples = irregular_noisy_samples(&mut rng, 200, 40.0, amplitude, omega, phase, 0.15);

        let raw_energies: Vec<f64> = samples
            .iter()
            .map(|(_, [x, v])| sho_energy(*x, *v, omega))
            .collect();

        let mut cfc = CfcSmoother::new(2, 0.5);
        let mut cfc_energies = Vec::with_capacity(samples.len());
        let mut last_t = samples[0].0;
        for (t, obs) in &samples {
            cfc.observe(t - last_t, obs);
            last_t = *t;
            let s = cfc.state();
            cfc_energies.push(sho_energy(s[0], s[1], omega));
        }

        let raw_var = variance(&raw_energies);
        // Skip the first ~20 samples (smoother still converging from its arbitrary init).
        let cfc_var = variance(&cfc_energies[20..]);
        println!(
            "energy residual variance: raw_noisy={raw_var:.4}, cfc_smoothed={cfc_var:.4} \
             (true energy is exactly constant: {:.4})",
            sho_energy(amplitude, 0.0, omega)
        );
        assert!(
            cfc_var < raw_var * 0.5,
            "CfC-smoothed energy estimate should be meaningfully more consistent (lower \
             variance) than the raw noisy observations' energy, got raw={raw_var:.4} vs \
             cfc={cfc_var:.4}"
        );
    }

    #[test]
    fn cfc_beats_naive_ema_under_irregular_sampling() {
        // The core claim this module exists to test: closed-form dt-awareness matters
        // specifically because sampling is IRREGULAR here (uniform random gaps, not a
        // fixed grid) -- a fixed-alpha EMA applies the same mixing weight regardless of
        // how much time actually elapsed, which is the wrong thing to do when gaps vary.
        let (amplitude, omega, phase) = (1.0, 1.5, 0.3);
        let mut rng = 0xBEEF_5EED_u64;
        let samples = irregular_noisy_samples(&mut rng, 300, 60.0, amplitude, omega, phase, 0.15);

        let mean_dt = {
            let mut total = 0.0;
            let mut last_t = samples[0].0;
            for (t, _) in &samples[1..] {
                total += t - last_t;
                last_t = *t;
            }
            total / (samples.len() - 1) as f64
        };
        let tau = 0.5;
        // Convert tau to the alpha a fixed-step EMA would use AT THE MEAN dt -- the most
        // charitable possible fixed alpha for the naive baseline, since it's calibrated
        // to this exact dataset's average spacing (a real irregular-sampling deployment
        // wouldn't even have this advantage).
        let equivalent_alpha = 1.0 - (-mean_dt / tau).exp();

        let mut cfc = CfcSmoother::new(2, tau);
        let mut naive = NaiveEmaSmoother::new(2, equivalent_alpha);
        let mut cfc_energies = Vec::with_capacity(samples.len());
        let mut naive_energies = Vec::with_capacity(samples.len());
        let mut last_t = samples[0].0;
        for (t, obs) in &samples {
            cfc.observe(t - last_t, obs);
            naive.observe(obs);
            last_t = *t;
            let cs = cfc.state();
            let ns = naive.state();
            cfc_energies.push(sho_energy(cs[0], cs[1], omega));
            naive_energies.push(sho_energy(ns[0], ns[1], omega));
        }

        let cfc_var = variance(&cfc_energies[30..]);
        let naive_var = variance(&naive_energies[30..]);
        println!(
            "under irregular sampling (mean_dt={mean_dt:.3}): cfc_var={cfc_var:.4}, \
             naive_ema_var={naive_var:.4} (naive using its best-case mean-dt-calibrated alpha)"
        );
        assert!(
            cfc_var < naive_var,
            "CfC's dt-aware update should outperform a fixed-alpha EMA under irregular \
             sampling even when the EMA's alpha is charitably calibrated to the dataset's \
             mean spacing, got cfc={cfc_var:.4} vs naive={naive_var:.4}"
        );
    }

    #[test]
    fn first_observation_initializes_without_blending() {
        let mut smoother = CfcSmoother::new(2, 1.0);
        smoother.observe(0.0, &[5.0, -3.0]);
        assert_eq!(smoother.state(), &[5.0, -3.0]);
    }

    #[test]
    fn larger_dt_trusts_the_new_observation_more() {
        let mut short_gap = CfcSmoother::new(1, 1.0);
        short_gap.observe(0.0, &[0.0]);
        short_gap.observe(0.01, &[10.0]);

        let mut long_gap = CfcSmoother::new(1, 1.0);
        long_gap.observe(0.0, &[0.0]);
        long_gap.observe(10.0, &[10.0]);

        assert!(
            long_gap.state()[0] > short_gap.state()[0],
            "a longer gap since the last observation should pull the estimate closer to \
             the new observation, got short_gap={:.4}, long_gap={:.4}",
            short_gap.state()[0],
            long_gap.state()[0]
        );
    }
}
