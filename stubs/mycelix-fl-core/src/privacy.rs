//! Differential Privacy (stub)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DifferentialPrivacyConfig {
    pub clip_norm: f32,
    pub noise_multiplier: f32,
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            clip_norm: 1.0,
            noise_multiplier: 1.1,
        }
    }
}

impl DifferentialPrivacyConfig {
    pub fn new(clip_norm: f32, noise_multiplier: f32) -> Self {
        Self {
            clip_norm,
            noise_multiplier,
        }
    }

    pub fn sigma(&self) -> f32 {
        self.clip_norm * self.noise_multiplier
    }
}

pub fn clip_gradient(gradient: &mut [f32], clip_norm: f32) {
    let norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > clip_norm {
        let scale = clip_norm / norm;
        for g in gradient.iter_mut() {
            *g *= scale;
        }
    }
}

pub fn add_gaussian_noise(gradient: &mut [f32], sigma: f32) {
    if sigma <= 0.0 {
        return;
    }
    // Simple Box-Muller transform with deterministic seed per element.
    // Uses a hash-based seed so repeated calls with different gradient values
    // produce different noise, while being reproducible.
    for (i, g) in gradient.iter_mut().enumerate() {
        // Mix the element index and current value into a pseudo-random seed
        let seed = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(g.to_bits() as u64)
            .wrapping_mul(2862933555777941757);
        let u1 = ((seed & 0xFFFFFFFF) as f64 + 1.0) / (0x100000000u64 as f64);
        let u2 = (((seed >> 32) & 0xFFFFFFFF) as f64 + 1.0) / (0x100000000u64 as f64);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        *g += (sigma as f64 * z) as f32;
    }
}

pub fn apply_dp(gradient: &mut [f32], config: &DifferentialPrivacyConfig) {
    clip_gradient(gradient, config.clip_norm);
    add_gaussian_noise(gradient, config.sigma());
}

pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[derive(Debug, Clone)]
pub struct RdpBudgetTracker {
    pub rounds: usize,
    delta: f64,
}

impl RdpBudgetTracker {
    pub fn new(delta: f64) -> Self {
        Self { rounds: 0, delta }
    }

    pub fn record_round(&mut self, _sigma: f32) {
        self.rounds += 1;
    }

    pub fn epsilon(&self) -> f64 {
        (self.rounds as f64) * 0.1 + (-self.delta.ln()).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct PrivacyReport {
    pub dp_applied: bool,
    pub clip_norm: f32,
    pub sigma: f32,
    pub epsilon_estimate: Option<f64>,
    pub rounds_tracked: usize,
}
