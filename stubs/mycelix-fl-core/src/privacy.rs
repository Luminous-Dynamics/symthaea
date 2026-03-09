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
    for (i, g) in gradient.iter_mut().enumerate() {
        let noise = sigma * 0.01 * if i % 2 == 0 { 1.0 } else { -1.0 };
        *g += noise;
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
