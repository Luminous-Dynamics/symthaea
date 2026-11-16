use std::f64::consts::TAU;

pub const MAX_SAMPLES: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixtureId {
    StableLinear,
    LogisticFixed,
    LogisticPeriod2,
    LogisticChaotic,
    Henon,
    QuasiperiodicTorus,
    Ar1ColoredNoise,
    Lorenz63,
}

impl FixtureId {
    pub const ALL: [Self; 8] = [
        Self::StableLinear,
        Self::LogisticFixed,
        Self::LogisticPeriod2,
        Self::LogisticChaotic,
        Self::Henon,
        Self::QuasiperiodicTorus,
        Self::Ar1ColoredNoise,
        Self::Lorenz63,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::StableLinear => "stable_linear",
            Self::LogisticFixed => "logistic_fixed",
            Self::LogisticPeriod2 => "logistic_period2",
            Self::LogisticChaotic => "logistic_chaotic",
            Self::Henon => "henon",
            Self::QuasiperiodicTorus => "quasiperiodic_torus",
            Self::Ar1ColoredNoise => "ar1_colored_noise",
            Self::Lorenz63 => "lorenz63",
        }
    }

    pub fn regime(self) -> &'static str {
        match self {
            Self::StableLinear => "stable",
            Self::LogisticFixed => "stable_fixed_point",
            Self::LogisticPeriod2 => "stable_periodic",
            Self::LogisticChaotic | Self::Henon => "chaotic_deterministic_map",
            Self::QuasiperiodicTorus => "quasiperiodic",
            Self::Ar1ColoredNoise => "stochastic_linear_process",
            Self::Lorenz63 => "chaotic_continuous_flow",
        }
    }

    pub fn burn_in(self) -> usize {
        match self {
            Self::StableLinear => 128,
            Self::LogisticFixed => 512,
            Self::LogisticPeriod2 => 1024,
            Self::LogisticChaotic => 512,
            Self::Henon => 2048,
            Self::QuasiperiodicTorus => 0,
            Self::Ar1ColoredNoise => 512,
            Self::Lorenz63 => 5000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub step: usize,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixtureError {
    ZeroSamples,
    TooManySamples { requested: usize, maximum: usize },
    NonFiniteState,
}

pub fn generate(id: FixtureId, samples: usize, seed: u64) -> Result<Vec<Sample>, FixtureError> {
    if samples == 0 {
        return Err(FixtureError::ZeroSamples);
    }
    if samples > MAX_SAMPLES {
        return Err(FixtureError::TooManySamples { requested: samples, maximum: MAX_SAMPLES });
    }

    let mut output = Vec::with_capacity(samples);
    match id {
        FixtureId::StableLinear => {
            let mut x = 0.4_f64;
            let total = id.burn_in() + samples;
            for step in 0..total {
                x *= 0.7;
                if step >= id.burn_in() {
                    push(&mut output, step - id.burn_in(), x, 0.0, 0.0)?;
                }
            }
        }
        FixtureId::LogisticFixed => generate_logistic(2.5, 0.2, id.burn_in(), samples, &mut output)?,
        FixtureId::LogisticPeriod2 => generate_logistic(3.2, 0.2, id.burn_in(), samples, &mut output)?,
        FixtureId::LogisticChaotic => generate_logistic(4.0, 0.123_456_789, id.burn_in(), samples, &mut output)?,
        FixtureId::Henon => {
            let (a, b) = (1.4_f64, 0.3_f64);
            let (mut x, mut y) = (0.0_f64, 0.0_f64);
            let total = id.burn_in() + samples;
            for step in 0..total {
                let next_x = 1.0 - a * x * x + y;
                let next_y = b * x;
                x = next_x;
                y = next_y;
                if step >= id.burn_in() {
                    push(&mut output, step - id.burn_in(), x, y, 0.0)?;
                }
            }
        }
        FixtureId::QuasiperiodicTorus => {
            for step in 0..samples {
                let n = (step + 1) as f64;
                let phase_a = n * 2.0_f64.sqrt();
                let phase_b = n * 3.0_f64.sqrt();
                push(&mut output, step, phase_a.cos(), phase_a.sin(), phase_b.cos())?;
            }
        }
        FixtureId::Ar1ColoredNoise => {
            let mut rng = SplitMix64::new(seed);
            let (phi, innovation_scale) = (0.8_f64, (1.0_f64 - 0.8_f64 * 0.8).sqrt());
            let mut x = 0.0_f64;
            let total = id.burn_in() + samples;
            for step in 0..total {
                let innovation = rng.normal();
                x = phi * x + innovation_scale * innovation;
                if step >= id.burn_in() {
                    push(&mut output, step - id.burn_in(), x, 0.0, 0.0)?;
                }
            }
        }
        FixtureId::Lorenz63 => {
            let (sigma, rho, beta, dt) = (10.0_f64, 28.0_f64, 8.0_f64 / 3.0, 0.01_f64);
            let mut state = [1.0_f64, 1.0_f64, 1.0_f64];
            let total = id.burn_in() + samples;
            for step in 0..total {
                state = lorenz_rk4(state, sigma, rho, beta, dt);
                if step >= id.burn_in() {
                    push(&mut output, step - id.burn_in(), state[0], state[1], state[2])?;
                }
            }
        }
    }
    Ok(output)
}

fn generate_logistic(
    r: f64,
    mut x: f64,
    burn_in: usize,
    samples: usize,
    output: &mut Vec<Sample>,
) -> Result<(), FixtureError> {
    for step in 0..(burn_in + samples) {
        x = r * x * (1.0 - x);
        if step >= burn_in {
            push(output, step - burn_in, x, 0.0, 0.0)?;
        }
    }
    Ok(())
}

fn push(out: &mut Vec<Sample>, step: usize, x: f64, y: f64, z: f64) -> Result<(), FixtureError> {
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return Err(FixtureError::NonFiniteState);
    }
    out.push(Sample { step, x, y, z });
    Ok(())
}

fn lorenz_derivative(s: [f64; 3], sigma: f64, rho: f64, beta: f64) -> [f64; 3] {
    [
        sigma * (s[1] - s[0]),
        s[0] * (rho - s[2]) - s[1],
        s[0] * s[1] - beta * s[2],
    ]
}

fn add_scaled(a: [f64; 3], b: [f64; 3], scale: f64) -> [f64; 3] {
    [a[0] + scale * b[0], a[1] + scale * b[1], a[2] + scale * b[2]]
}

fn lorenz_rk4(s: [f64; 3], sigma: f64, rho: f64, beta: f64, dt: f64) -> [f64; 3] {
    let k1 = lorenz_derivative(s, sigma, rho, beta);
    let k2 = lorenz_derivative(add_scaled(s, k1, dt * 0.5), sigma, rho, beta);
    let k3 = lorenz_derivative(add_scaled(s, k2, dt * 0.5), sigma, rho, beta);
    let k4 = lorenz_derivative(add_scaled(s, k3, dt), sigma, rho, beta);
    [
        s[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        s[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        s[2] + dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    ]
}

/// Seeded normal noise generator. Deterministic within the generator's
/// integer-arithmetic contract; it is not a CSPRNG.
struct SplitMix64 { state: u64 }

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform_open01(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform_open01();
        let u2 = self.uniform_open01();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// Compact non-cryptographic digest for fixture regression only.
pub fn digest_non_security(samples: &[Sample]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for sample in samples {
        for bytes in [
            (sample.step as u64).to_le_bytes(),
            sample.x.to_bits().to_le_bytes(),
            sample.y.to_bits().to_le_bytes(),
            sample.z.to_bits().to_le_bytes(),
        ] {
            for byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "a={a:.12e}, b={b:.12e}, tol={tol:.3e}");
    }

    #[test]
    fn all_fixtures_emit_the_requested_finite_sample_count() {
        for id in FixtureId::ALL {
            let samples = generate(id, 128, 2026).unwrap();
            assert_eq!(samples.len(), 128, "{}", id.as_str());
            assert!(samples.iter().all(|s| s.x.is_finite() && s.y.is_finite() && s.z.is_finite()));
        }
    }

    #[test]
    fn stable_linear_fixture_matches_analytic_contraction() {
        let samples = generate(FixtureId::StableLinear, 8, 0).unwrap();
        near(samples[0].x, 0.4 * 0.7_f64.powi(129), 1e-14);
        near(0.7_f64.ln(), -0.356_674_943_938_732_45, 1e-15);
    }

    #[test]
    fn logistic_fixed_point_matches_known_solution() {
        let samples = generate(FixtureId::LogisticFixed, 8, 0).unwrap();
        for sample in samples {
            near(sample.x, 0.6, 1e-12);
        }
    }

    #[test]
    fn logistic_period_two_fixture_has_two_step_recurrence() {
        let samples = generate(FixtureId::LogisticPeriod2, 16, 0).unwrap();
        for i in 2..samples.len() {
            near(samples[i].x, samples[i - 2].x, 1e-10);
        }
        assert!((samples[15].x - samples[14].x).abs() > 1e-3);
    }

    #[test]
    fn quasiperiodic_fixture_stays_on_unit_circle_in_first_plane() {
        let samples = generate(FixtureId::QuasiperiodicTorus, 128, 0).unwrap();
        for s in samples {
            near(s.x * s.x + s.y * s.y, 1.0, 1e-12);
        }
    }

    #[test]
    fn seeded_ar1_is_reproducible_but_seed_sensitive() {
        let a = generate(FixtureId::Ar1ColoredNoise, 256, 123).unwrap();
        let b = generate(FixtureId::Ar1ColoredNoise, 256, 123).unwrap();
        let c = generate(FixtureId::Ar1ColoredNoise, 256, 124).unwrap();
        assert_eq!(digest_non_security(&a), digest_non_security(&b));
        assert_ne!(digest_non_security(&a), digest_non_security(&c));
    }

    #[test]
    fn lorenz_trajectory_is_finite_and_bounded_for_frozen_fixture() {
        let samples = generate(FixtureId::Lorenz63, 256, 0).unwrap();
        assert!(samples.iter().all(|s| s.x.abs() < 100.0 && s.y.abs() < 100.0 && s.z.abs() < 100.0));
    }

    #[test]
    fn sample_budget_is_enforced() {
        assert_eq!(generate(FixtureId::StableLinear, 0, 0), Err(FixtureError::ZeroSamples));
        assert!(matches!(
            generate(FixtureId::StableLinear, MAX_SAMPLES + 1, 0),
            Err(FixtureError::TooManySamples { .. })
        ));
    }
}
