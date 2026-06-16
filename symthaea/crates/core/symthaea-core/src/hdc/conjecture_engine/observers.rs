// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use super::{MathDomain, ObservedSequence};

// ═══════════════════════════════════════════════════════════════════════════
// DATA COLLECTORS (observe math engine outputs)
// ═══════════════════════════════════════════════════════════════════════════

/// Collect partition count sequence p(1)..p(n).
pub fn observe_partitions(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::partition_count;
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, partition_count(n as u64) as f64))
        .collect();
    ObservedSequence::new("partition_count(n)", MathDomain::Combinatorics, data)
}

/// Collect Fibonacci ratio sequence F(n)/F(n-1) for n=2..max_n.
pub fn observe_fibonacci_ratios(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::fibonacci;
    let data: Vec<(f64, f64)> = (2..=max_n)
        .filter_map(|n| {
            let prev = fibonacci(n as u64 - 1);
            let curr = fibonacci(n as u64);
            if prev > 0 {
                Some((n as f64, curr as f64 / prev as f64))
            } else {
                None
            }
        })
        .collect();
    ObservedSequence::new("fibonacci_ratio(n)", MathDomain::Combinatorics, data)
}

/// Collect GCT obstruction ratio for perm_n vs det_{n²} for n=2..max_n.
///
/// This is potentially novel mathematics: the scaling of Kronecker coefficient
/// zeros as a function of permanent dimension has not been systematically mapped.
pub fn observe_gct_obstruction(max_n: usize) -> ObservedSequence {
    use crate::hdc::gct::check_obstruction_conjecture;
    let data: Vec<(f64, f64)> = (2..=max_n.min(6))
        .map(|n| {
            let result = check_obstruction_conjecture(n, n * n);
            (n as f64, result.obstruction_ratio)
        })
        .collect();
    ObservedSequence::new(
        "gct_obstruction_ratio(n)",
        MathDomain::AlgebraicComplexity,
        data,
    )
}

/// Detailed GCT obstruction report — returns raw counts + survivor triples.
pub fn observe_gct_detailed(max_n: usize) -> Vec<GctObservation> {
    use crate::hdc::gct::check_obstruction_conjecture;
    (2..=max_n.min(6))
        .map(|n| {
            let r = check_obstruction_conjecture(n, n * n);
            GctObservation {
                n,
                obstructions: r.obstructions_found,
                total: r.total_tested,
                ratio: r.obstruction_ratio,
                survivors: r.survivors,
            }
        })
        .collect()
}

/// Detailed observation for one dimension of the GCT obstruction scan.
#[derive(Debug, Clone)]
pub struct GctObservation {
    pub n: usize,
    pub obstructions: usize,
    pub total: usize,
    pub ratio: f64,
    /// The surviving (non-zero) triples: (lambda, mu, nu, coefficient)
    pub survivors: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, u64)>,
}

/// Collect prime gap sequence: gap(k) = p_{k+1} - p_k.
pub fn observe_prime_gaps(max_prime: u64) -> ObservedSequence {
    let mut primes = Vec::new();
    let mut is_prime = vec![true; max_prime as usize + 1];
    if max_prime >= 2 {
        for i in 2..=max_prime as usize {
            if is_prime[i] {
                primes.push(i as u64);
                let mut j = i * 2;
                while j <= max_prime as usize {
                    is_prime[j] = false;
                    j += i;
                }
            }
        }
    }
    let data: Vec<(f64, f64)> = primes
        .windows(2)
        .enumerate()
        .map(|(i, w)| (i as f64 + 1.0, (w[1] - w[0]) as f64))
        .collect();
    ObservedSequence::new("prime_gap(k)", MathDomain::NumberTheory, data)
}

/// Observe maximal prime gap below n: G(n) = max(p_{k+1} - p_k) for p_k ≤ n.
pub fn observe_maximal_prime_gap(max_n: u64) -> ObservedSequence {
    let mut is_prime = vec![true; max_n as usize + 1];
    for i in 2..=(max_n as f64).sqrt() as usize {
        if is_prime[i] {
            let mut j = i * i;
            while j <= max_n as usize {
                is_prime[j] = false;
                j += i;
            }
        }
    }

    let mut max_gap = 0u64;
    let mut prev_prime = 2u64;
    let mut data = Vec::new();
    let checkpoints: Vec<u64> = (1..=20).map(|i| max_n * i / 20).collect();
    let mut next_cp = 0;

    for n in 3..=max_n {
        if is_prime[n as usize] {
            let gap = n - prev_prime;
            if gap > max_gap {
                max_gap = gap;
            }
            prev_prime = n;
        }
        if next_cp < checkpoints.len() && n >= checkpoints[next_cp] {
            if max_gap > 0 {
                data.push((n as f64, max_gap as f64));
            }
            next_cp += 1;
        }
    }
    ObservedSequence::new("max_prime_gap(n)", MathDomain::NumberTheory, data)
}

/// Collect permanent/determinant ratio for random n×n matrices.
pub fn observe_perm_det_ratio(max_n: usize) -> ObservedSequence {
    use crate::hdc::gct::permanent_determinant_ratio;
    let data: Vec<(f64, f64)> = (1..=max_n.min(6))
        .map(|n| (n as f64, permanent_determinant_ratio(n, 200, 42)))
        .collect();
    ObservedSequence::new("perm_det_ratio(n)", MathDomain::AlgebraicComplexity, data)
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE INVARIANT DISCOVERY
// ═══════════════════════════════════════════════════════════════════════════

/// Inline RK4 trajectory generation for small invariant-mining probes.
fn rk45_trajectory(
    f: impl Fn(&[f64], f64) -> Vec<f64>,
    y0: &[f64],
    t_end: f64,
    dt: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut t = 0.0;
    let mut y = y0.to_vec();
    let mut times = vec![t];
    let mut states = vec![y.clone()];
    let dim = y0.len();

    while t < t_end {
        let h = dt.min(t_end - t);
        let k1 = f(&y, t);
        let y2: Vec<f64> = (0..dim).map(|i| y[i] + h * 0.5 * k1[i]).collect();
        let k2 = f(&y2, t + 0.5 * h);
        let y3: Vec<f64> = (0..dim).map(|i| y[i] + h * 0.5 * k2[i]).collect();
        let k3 = f(&y3, t + 0.5 * h);
        let y4: Vec<f64> = (0..dim).map(|i| y[i] + h * k3[i]).collect();
        let k4 = f(&y4, t + h);
        for i in 0..dim {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        times.push(t);
        states.push(y.clone());
    }
    (times, states)
}

fn lorenz_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let (x, y, z) = (s[0], s[1], s[2]);
    let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
    vec![sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
}

/// Observe time-averaged Lorenz statistics for conjecture discovery.
pub fn observe_lorenz_time_averages(n_samples: usize) -> ObservedSequence {
    let (_, states) = rk45_trajectory(lorenz_rhs, &[1.0, 1.0, 1.0], 50.0, 0.01);
    let attractor_states = if states.len() > 1000 {
        &states[1000..]
    } else {
        &states
    };
    let total = attractor_states.len();
    let step = total / n_samples.max(1);
    let mut data = Vec::new();
    let mut z_sum = 0.0;
    let mut count = 0usize;

    for (i, state) in attractor_states.iter().enumerate() {
        z_sum += state[2];
        count += 1;
        if (i + 1) % step == 0 && data.len() < n_samples {
            data.push((data.len() as f64 + 1.0, z_sum / count as f64));
        }
    }

    ObservedSequence::new(
        "lorenz_time_avg_z(samples)",
        MathDomain::DynamicalSystems,
        data,
    )
}

/// Observe Lorenz attractor candidate invariants for later GP search.
pub fn observe_lorenz_invariant_candidates(n_points: usize) -> Vec<ObservedSequence> {
    let (times, states) = rk45_trajectory(lorenz_rhs, &[1.0, 1.0, 1.0], 50.0, 0.01);
    let skip = if states.len() > 1000 { 1000 } else { 0 };
    let attractor = &states[skip..];
    let attractor_t = &times[skip..];
    let step = attractor.len() / n_points.max(1);

    let mut seqs = Vec::new();

    let z_data: Vec<(f64, f64)> = attractor
        .iter()
        .zip(attractor_t)
        .step_by(step.max(1))
        .take(n_points)
        .map(|(s, &t)| (t, s[2]))
        .collect();
    seqs.push(ObservedSequence::new(
        "lorenz_z(t)",
        MathDomain::DynamicalSystems,
        z_data,
    ));

    let xy_data: Vec<(f64, f64)> = attractor
        .iter()
        .zip(attractor_t)
        .step_by(step.max(1))
        .take(n_points)
        .map(|(s, &t)| (t, s[0] * s[0] + s[1] * s[1]))
        .collect();
    seqs.push(ObservedSequence::new(
        "lorenz_x2_y2(t)",
        MathDomain::DynamicalSystems,
        xy_data,
    ));

    let r2_data: Vec<(f64, f64)> = attractor
        .iter()
        .zip(attractor_t)
        .step_by(step.max(1))
        .take(n_points)
        .map(|(s, &t)| (t, s[0] * s[0] + s[1] * s[1] + s[2] * s[2]))
        .collect();
    seqs.push(ObservedSequence::new(
        "lorenz_r2(t)",
        MathDomain::DynamicalSystems,
        r2_data,
    ));

    seqs
}

/// Observe Bell numbers B(n) for n=0..max_n.
pub fn observe_bell_numbers(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::bell;
    let data: Vec<(f64, f64)> = (0..=max_n).map(|n| (n as f64, bell(n) as f64)).collect();
    ObservedSequence::new("bell(n)", MathDomain::Combinatorics, data)
}

/// Observe the Stirling-sum Σ_{k=0}^{n} S(n,k) for each n.
pub fn observe_stirling_sum(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::stirling_second;
    let data: Vec<(f64, f64)> = (0..=max_n)
        .map(|n| {
            let sum: u64 = (0..=n).map(|k| stirling_second(n, k)).sum();
            (n as f64, sum as f64)
        })
        .collect();
    ObservedSequence::new("stirling_sum(n)", MathDomain::Combinatorics, data)
}

/// Observe the difference B(n) - Σ S(n,k) to verify the Bell-Stirling identity.
pub fn observe_bell_stirling_residual(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::{bell, stirling_second};
    let data: Vec<(f64, f64)> = (0..=max_n)
        .map(|n| {
            let b = bell(n) as f64;
            let s_sum: f64 = (0..=n).map(|k| stirling_second(n, k) as f64).sum();
            (n as f64, (b - s_sum).abs())
        })
        .collect();
    ObservedSequence::new("bell_stirling_residual(n)", MathDomain::Combinatorics, data)
}

/// Observe Catalan numbers C(n) = C(2n,n)/(n+1).
pub fn observe_catalan(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::catalan;
    let data: Vec<(f64, f64)> = (0..=max_n)
        .map(|n| (n as f64, catalan(n as u64) as f64))
        .collect();
    ObservedSequence::new("catalan(n)", MathDomain::Combinatorics, data)
}

/// Observe derangement numbers D(n).
pub fn observe_derangements(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::derangement;
    let data: Vec<(f64, f64)> = (0..=max_n)
        .map(|n| (n as f64, derangement(n as u64) as f64))
        .collect();
    ObservedSequence::new("derangement(n)", MathDomain::Combinatorics, data)
}

/// Observe D(n)/n! ratio (should converge to 1/e ≈ 0.3679).
pub fn observe_derangement_ratio(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::derangement;
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| {
            let d = derangement(n as u64) as f64;
            let nfact: f64 = (1..=n as u64).map(|i| i as f64).product();
            (n as f64, d / nfact)
        })
        .collect();
    ObservedSequence::new("derangement_ratio(n)", MathDomain::Combinatorics, data)
}

/// Observe prime counting function π(n) for n = 1..max_n.
pub fn observe_prime_counting(max_n: usize) -> ObservedSequence {
    let mut is_prime = vec![true; max_n + 1];
    if max_n >= 1 {
        is_prime[0] = false;
    }
    if max_n >= 2 {
        is_prime[1] = false;
    }
    for i in 2..=max_n {
        if is_prime[i] {
            let mut j = i * 2;
            while j <= max_n {
                is_prime[j] = false;
                j += i;
            }
        }
    }
    let mut count = 0u64;
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| {
            if is_prime[n] {
                count += 1;
            }
            (n as f64, count as f64)
        })
        .collect();
    ObservedSequence::new("prime_counting(n)", MathDomain::NumberTheory, data)
}

fn harmonic_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    vec![s[1], -s[0]]
}

/// Observe harmonic oscillator invariant candidates.
pub fn observe_harmonic_invariants(n_points: usize) -> Vec<ObservedSequence> {
    let (times, states) = rk45_trajectory(harmonic_rhs, &[1.0, 0.0], 20.0, 0.01);
    let step = states.len() / n_points.max(1);
    let mut seqs = Vec::new();

    let energy: Vec<(f64, f64)> = states
        .iter()
        .zip(&times)
        .step_by(step.max(1))
        .take(n_points)
        .map(|(s, &t)| (t, s[0] * s[0] + s[1] * s[1]))
        .collect();
    seqs.push(ObservedSequence::new(
        "harmonic_E(t)",
        MathDomain::DynamicalSystems,
        energy,
    ));

    let x2: Vec<(f64, f64)> = states
        .iter()
        .zip(&times)
        .step_by(step.max(1))
        .take(n_points)
        .map(|(s, &t)| (t, s[0] * s[0]))
        .collect();
    seqs.push(ObservedSequence::new(
        "harmonic_x²(t)",
        MathDomain::DynamicalSystems,
        x2,
    ));

    seqs
}

/// Score invariant by variance. Zero variance = exact conservation law.
pub fn invariant_variance(data: &[(f64, f64)]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, f64::MAX);
    }
    let values: Vec<f64> = data.iter().map(|(_, v)| *v).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    (mean, var)
}

/// Observe the central binomial normalization: C(2n,n) · √n / 4^n → 1/√π.
pub fn observe_central_binomial_limit(max_n: usize) -> ObservedSequence {
    use crate::hdc::combinatorics::binomial;

    let data: Vec<(f64, f64)> = (2..=max_n)
        .filter_map(|n| {
            let cbn = binomial(2 * n as u64, n as u64) as f64;
            let val = cbn * (n as f64).sqrt() / 4.0_f64.powi(n as i32);
            if val.is_finite() && val > 0.0 {
                Some((n as f64, val))
            } else {
                None
            }
        })
        .collect();
    ObservedSequence::new("central_binom_limit(n)", MathDomain::Combinatorics, data)
}

/// Observe hydrogen atom energy levels: E_n = -13.6 / n² eV.
pub fn observe_hydrogen_energy_levels(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=max_n)
        .map(|n| (n as f64, -13.6 / (n as f64).powi(2)))
        .collect();
    ObservedSequence::new("hydrogen_E(n)", MathDomain::Physics, data)
}

/// Observe quantum harmonic oscillator energy levels: E_n = n + 0.5.
pub fn observe_quantum_harmonic_oscillator(max_n: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (0..=max_n).map(|n| (n as f64, n as f64 + 0.5)).collect();
    ObservedSequence::new("qho_E(n)", MathDomain::Physics, data)
}

/// Observe blackbody radiation peak wavelength vs temperature using Wien's law.
pub fn observe_blackbody_peak(n_temps: usize) -> ObservedSequence {
    let wien_b = 2.898e-3;
    let data: Vec<(f64, f64)> = (1..=n_temps)
        .map(|i| {
            let frac = (i as f64) / (n_temps as f64);
            let t = 300.0 * (10000.0_f64 / 300.0).powf(frac);
            (t, wien_b / t)
        })
        .collect();
    ObservedSequence::new("blackbody_peak(T)", MathDomain::Physics, data)
}

/// Observe Balmer series wavelengths: 1/λ = R_H (1/4 - 1/n²).
pub fn observe_balmer_series(max_n: usize) -> ObservedSequence {
    let rydberg = 1.0973731568539e7;
    let data: Vec<(f64, f64)> = (3..=max_n.max(3))
        .map(|n| {
            let inv_lambda = rydberg * (0.25 - 1.0 / (n as f64).powi(2));
            let lambda_nm = 1.0e9 / inv_lambda;
            (n as f64, lambda_nm)
        })
        .collect();
    ObservedSequence::new("balmer_λ(n)", MathDomain::Physics, data)
}

/// Observe Kepler's third law: T = r^(3/2) (normalized units GM = 4π²).
pub fn observe_kepler_third_law(n_orbits: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=n_orbits)
        .map(|i| {
            let r = i as f64;
            let t = r.powf(1.5);
            (r, t)
        })
        .collect();
    ObservedSequence::new("kepler_T(r)", MathDomain::Physics, data)
}

/// Observe Stefan-Boltzmann law: P ∝ T⁴ (normalized units σA = 1).
pub fn observe_stefan_boltzmann(n_temps: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=n_temps)
        .map(|i| {
            let t = i as f64 * 100.0;
            let p = t.powi(4);
            (t, p)
        })
        .collect();
    ObservedSequence::new("stefan_boltzmann_P(T)", MathDomain::Physics, data)
}

/// Observe relativistic kinetic energy in natural units.
pub fn observe_relativistic_kinetic_energy(n_samples: usize) -> ObservedSequence {
    let data: Vec<(f64, f64)> = (1..=n_samples)
        .map(|i| {
            let v = 0.1 + 0.85 * (i as f64) / (n_samples as f64);
            let gamma = 1.0 / (1.0 - v * v).sqrt();
            let ke = gamma - 1.0;
            (v, ke)
        })
        .collect();
    ObservedSequence::new("relativistic_KE(v)", MathDomain::Physics, data)
}
