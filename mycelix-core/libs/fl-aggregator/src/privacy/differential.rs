//! Differential privacy for federated learning.
//!
//! Implements gradient clipping and noise addition for (ε, δ)-differential privacy.
//!
//! Based on:
//! - Abadi et al., "Deep Learning with Differential Privacy" (2016)
//! - McMahan et al., "Learning Differentially Private Recurrent Language Models" (2017)

use std::collections::HashMap;

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

#[cfg(feature = "privacy")]
use rand::prelude::*;
#[cfg(feature = "privacy")]
use rand_distr::{Distribution, Normal};

/// Type of noise mechanism.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NoiseType {
    /// Gaussian noise for (ε, δ)-DP.
    Gaussian,
    /// Laplace noise for ε-DP.
    Laplace,
}

impl Default for NoiseType {
    fn default() -> Self {
        Self::Gaussian
    }
}

/// Configuration for differential privacy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DPConfig {
    /// Privacy budget (lower = more private).
    pub epsilon: f32,
    /// Failure probability (typically 1e-5 to 1e-7).
    pub delta: f32,
    /// Maximum L2 norm for gradient clipping.
    pub max_grad_norm: f32,
    /// Noise scale multiplier.
    pub noise_multiplier: f32,
    /// Type of noise mechanism.
    pub noise_type: NoiseType,
    /// Target epsilon for privacy accountant.
    pub target_epsilon: Option<f32>,
    /// Target delta for privacy accountant.
    pub target_delta: Option<f32>,
}

impl Default for DPConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            max_grad_norm: 1.0,
            noise_multiplier: 1.0,
            noise_type: NoiseType::Gaussian,
            target_epsilon: None,
            target_delta: None,
        }
    }
}

impl DPConfig {
    /// Set epsilon (privacy budget).
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set delta (failure probability).
    pub fn with_delta(mut self, delta: f32) -> Self {
        self.delta = delta;
        self
    }

    /// Set maximum gradient norm for clipping.
    pub fn with_max_grad_norm(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = max_norm;
        self
    }

    /// Set noise multiplier.
    pub fn with_noise_multiplier(mut self, multiplier: f32) -> Self {
        self.noise_multiplier = multiplier;
        self
    }

    /// Set noise type.
    pub fn with_noise_type(mut self, noise_type: NoiseType) -> Self {
        self.noise_type = noise_type;
        self
    }

    /// Set target epsilon for budget enforcement.
    pub fn with_target_epsilon(mut self, target: f32) -> Self {
        self.target_epsilon = Some(target);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.epsilon <= 0.0 {
            return Err("epsilon must be positive");
        }
        if self.delta <= 0.0 || self.delta >= 1.0 {
            return Err("delta must be in (0, 1)");
        }
        if self.max_grad_norm <= 0.0 {
            return Err("max_grad_norm must be positive");
        }
        if self.noise_multiplier < 0.0 {
            return Err("noise_multiplier must be non-negative");
        }
        Ok(())
    }
}

/// Metadata from privacy operations.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PrivacyMetadata {
    /// Original gradient norm before clipping.
    pub original_norm: f32,
    /// Norm after clipping.
    pub clipped_norm: f32,
    /// Whether gradient was clipped.
    pub was_clipped: bool,
    /// Noise scale used.
    pub noise_scale: f32,
    /// Current cumulative epsilon.
    pub current_epsilon: f32,
}

/// Clip gradient to maximum L2 norm.
///
/// Returns (clipped_gradient, original_norm).
pub fn clip_gradient(gradient: ArrayView1<f32>, max_norm: f32) -> (Array1<f32>, f32) {
    let norm = gradient.dot(&gradient).sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        (gradient.mapv(|x| x * scale), norm)
    } else {
        (gradient.to_owned(), norm)
    }
}

/// Clip multiple gradients.
pub fn clip_gradients_batch(
    gradients: &HashMap<String, Array1<f32>>,
    max_norm: f32,
) -> (HashMap<String, Array1<f32>>, HashMap<String, f32>) {
    let mut clipped = HashMap::new();
    let mut norms = HashMap::new();

    for (node_id, gradient) in gradients {
        let (clipped_grad, original_norm) = clip_gradient(gradient.view(), max_norm);
        clipped.insert(node_id.clone(), clipped_grad);
        norms.insert(node_id.clone(), original_norm);
    }

    (clipped, norms)
}

/// Gaussian mechanism for (ε, δ)-differential privacy.
#[derive(Clone, Debug)]
pub struct GaussianMechanism {
    /// Privacy parameter epsilon.
    pub epsilon: f32,
    /// Failure probability delta.
    pub delta: f32,
    /// L2 sensitivity.
    pub sensitivity: f32,
    /// Computed noise scale (sigma).
    sigma: f32,
}

impl GaussianMechanism {
    /// Create a new Gaussian mechanism.
    pub fn new(epsilon: f32, delta: f32, sensitivity: f32) -> Self {
        let sigma = Self::compute_sigma(epsilon, delta, sensitivity);
        Self {
            epsilon,
            delta,
            sensitivity,
            sigma,
        }
    }

    /// Compute noise scale for (ε, δ)-DP.
    /// σ ≥ √(2 ln(1.25/δ)) * Δf / ε
    fn compute_sigma(epsilon: f32, delta: f32, sensitivity: f32) -> f32 {
        let ln_term = (1.25 / delta).ln();
        (2.0 * ln_term).sqrt() * sensitivity / epsilon
    }

    /// Get the noise scale (sigma).
    pub fn noise_scale(&self) -> f32 {
        self.sigma
    }

    /// Add Gaussian noise to a value.
    #[cfg(feature = "privacy")]
    pub fn add_noise(&self, value: ArrayView1<f32>) -> Array1<f32> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, self.sigma as f64).unwrap();

        value.mapv(|x| x + normal.sample(&mut rng) as f32)
    }

    #[cfg(not(feature = "privacy"))]
    pub fn add_noise(&self, value: ArrayView1<f32>) -> Array1<f32> {
        value.to_owned()
    }

    /// Add noise with provided RNG.
    #[cfg(feature = "privacy")]
    pub fn add_noise_with_rng<R: Rng>(&self, value: ArrayView1<f32>, rng: &mut R) -> Array1<f32> {
        let normal = Normal::new(0.0, self.sigma as f64).unwrap();
        value.mapv(|x| x + normal.sample(rng) as f32)
    }

    #[cfg(not(feature = "privacy"))]
    pub fn add_noise_with_rng<R>(&self, value: ArrayView1<f32>, _rng: &mut R) -> Array1<f32> {
        value.to_owned()
    }
}

/// Laplace mechanism for ε-differential privacy.
#[derive(Clone, Debug)]
pub struct LaplaceMechanism {
    /// Privacy parameter epsilon.
    pub epsilon: f32,
    /// L1 sensitivity.
    pub sensitivity: f32,
    /// Noise scale (b parameter).
    scale: f32,
}

impl LaplaceMechanism {
    /// Create a new Laplace mechanism.
    pub fn new(epsilon: f32, sensitivity: f32) -> Self {
        let scale = sensitivity / epsilon;
        Self {
            epsilon,
            sensitivity,
            scale,
        }
    }

    /// Get the noise scale (b parameter).
    pub fn noise_scale(&self) -> f32 {
        self.scale
    }

    /// Add Laplace noise to a value.
    #[cfg(feature = "privacy")]
    pub fn add_noise(&self, value: ArrayView1<f32>) -> Array1<f32> {
        let mut rng = thread_rng();

        // Laplace distribution: sample from exponential and randomize sign
        value.mapv(|x| {
            let u: f64 = rng.gen::<f64>() - 0.5;
            let noise = -self.scale as f64 * u.signum() * (1.0 - 2.0 * u.abs()).ln();
            x + noise as f32
        })
    }

    #[cfg(not(feature = "privacy"))]
    pub fn add_noise(&self, value: ArrayView1<f32>) -> Array1<f32> {
        value.to_owned()
    }
}

/// Add noise to a gradient using specified mechanism.
#[cfg(feature = "privacy")]
pub fn add_noise(
    gradient: ArrayView1<f32>,
    noise_multiplier: f32,
    sensitivity: f32,
    noise_type: NoiseType,
) -> Array1<f32> {
    let noise_scale = noise_multiplier * sensitivity;
    let mut rng = thread_rng();

    match noise_type {
        NoiseType::Gaussian => {
            let normal = Normal::new(0.0, noise_scale as f64).unwrap();
            gradient.mapv(|x| x + normal.sample(&mut rng) as f32)
        }
        NoiseType::Laplace => {
            gradient.mapv(|x| {
                let u: f64 = rng.gen::<f64>() - 0.5;
                let noise = -noise_scale as f64 * u.signum() * (1.0 - 2.0 * u.abs()).ln();
                x + noise as f32
            })
        }
    }
}

#[cfg(not(feature = "privacy"))]
pub fn add_noise(
    gradient: ArrayView1<f32>,
    _noise_multiplier: f32,
    _sensitivity: f32,
    _noise_type: NoiseType,
) -> Array1<f32> {
    gradient.to_owned()
}

/// Compute privacy budget using simplified RDP accounting.
///
/// For production, use a full RDP accountant.
pub fn compute_privacy_budget(
    noise_multiplier: f32,
    sample_rate: f32,
    num_steps: u64,
    delta: f32,
) -> f32 {
    // Simplified composition bound (not tight, but safe)
    let per_step_epsilon = sample_rate / noise_multiplier;
    let composed_epsilon =
        (2.0 * num_steps as f32 * (1.0 / delta).ln()).sqrt() * per_step_epsilon;

    composed_epsilon
}

/// Privacy budget accountant.
#[derive(Clone, Debug)]
pub struct PrivacyAccountant {
    /// Base noise multiplier.
    noise_multiplier: f32,
    /// Sampling rate per round.
    sample_rate: f32,
    /// Target delta.
    delta: f32,
    /// Steps recorded.
    steps: Vec<f32>,
}

impl PrivacyAccountant {
    /// Create a new privacy accountant.
    pub fn new(noise_multiplier: f32, sample_rate: f32, delta: f32) -> Self {
        Self {
            noise_multiplier,
            sample_rate,
            delta,
            steps: Vec::new(),
        }
    }

    /// Record a training step.
    pub fn step(&mut self, noise_multiplier: Option<f32>) {
        let nm = noise_multiplier.unwrap_or(self.noise_multiplier);
        self.steps.push(nm);
    }

    /// Get current privacy budget spent (epsilon).
    pub fn get_epsilon(&self) -> f32 {
        if self.steps.is_empty() {
            return 0.0;
        }

        let avg_nm: f32 = self.steps.iter().sum::<f32>() / self.steps.len() as f32;

        compute_privacy_budget(avg_nm, self.sample_rate, self.steps.len() as u64, self.delta)
    }

    /// Get remaining budget.
    pub fn get_remaining_budget(&self, target_epsilon: f32) -> f32 {
        (target_epsilon - self.get_epsilon()).max(0.0)
    }

    /// Check if training can continue within budget.
    pub fn can_continue(&self, target_epsilon: f32) -> bool {
        self.get_epsilon() < target_epsilon
    }

    /// Reset the accountant.
    pub fn reset(&mut self) {
        self.steps.clear();
    }

    /// Get number of steps recorded.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }
}

/// High-level differential privacy wrapper.
pub struct DifferentialPrivacy {
    config: DPConfig,
    accountant: PrivacyAccountant,
    gaussian: Option<GaussianMechanism>,
    laplace: Option<LaplaceMechanism>,
}

impl DifferentialPrivacy {
    /// Create a new DP wrapper.
    pub fn new(config: DPConfig) -> Self {
        let accountant = PrivacyAccountant::new(config.noise_multiplier, 1.0, config.delta);

        let gaussian = if config.noise_type == NoiseType::Gaussian {
            Some(GaussianMechanism::new(
                config.epsilon,
                config.delta,
                config.max_grad_norm,
            ))
        } else {
            None
        };

        let laplace = if config.noise_type == NoiseType::Laplace {
            Some(LaplaceMechanism::new(config.epsilon, config.max_grad_norm))
        } else {
            None
        };

        Self {
            config,
            accountant,
            gaussian,
            laplace,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &DPConfig {
        &self.config
    }

    /// Apply differential privacy to a gradient.
    pub fn privatize_gradient(
        &mut self,
        gradient: ArrayView1<f32>,
        clip: bool,
        add_noise: bool,
    ) -> (Array1<f32>, PrivacyMetadata) {
        let original_norm = gradient.dot(&gradient).sqrt();
        let mut metadata = PrivacyMetadata {
            original_norm,
            ..Default::default()
        };

        // Clip gradient
        let (mut result, _clipped_norm) = if clip {
            let (clipped, norm) = clip_gradient(gradient, self.config.max_grad_norm);
            metadata.was_clipped = norm > self.config.max_grad_norm;
            metadata.clipped_norm = norm.min(self.config.max_grad_norm);
            (clipped, norm)
        } else {
            (gradient.to_owned(), original_norm)
        };

        // Add noise
        if add_noise {
            result = match (&self.gaussian, &self.laplace) {
                (Some(g), _) => {
                    metadata.noise_scale = g.noise_scale();
                    g.add_noise(result.view())
                }
                (_, Some(l)) => {
                    metadata.noise_scale = l.noise_scale();
                    l.add_noise(result.view())
                }
                _ => result,
            };
        }

        // Record step
        self.accountant.step(None);
        metadata.current_epsilon = self.accountant.get_epsilon();

        (result, metadata)
    }

    /// Apply DP to multiple gradients.
    pub fn privatize_gradients(
        &mut self,
        gradients: &HashMap<String, Array1<f32>>,
        clip: bool,
        add_noise: bool,
    ) -> (HashMap<String, Array1<f32>>, HashMap<String, PrivacyMetadata>) {
        let mut privatized = HashMap::new();
        let mut metadata = HashMap::new();

        for (node_id, gradient) in gradients {
            let (private_grad, meta) = self.privatize_gradient(gradient.view(), clip, add_noise);
            privatized.insert(node_id.clone(), private_grad);
            metadata.insert(node_id.clone(), meta);
        }

        (privatized, metadata)
    }

    /// Get total privacy budget spent.
    pub fn get_privacy_spent(&self) -> f32 {
        self.accountant.get_epsilon()
    }

    /// Get remaining privacy budget.
    pub fn get_privacy_remaining(&self) -> Option<f32> {
        self.config
            .target_epsilon
            .map(|target| self.accountant.get_remaining_budget(target))
    }

    /// Check if we can continue within budget.
    pub fn can_continue(&self) -> bool {
        match self.config.target_epsilon {
            Some(target) => self.accountant.can_continue(target),
            None => true,
        }
    }

    /// Reset privacy accountant.
    pub fn reset(&mut self) {
        self.accountant.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_config_validation() {
        let valid = DPConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_epsilon = DPConfig {
            epsilon: -1.0,
            ..Default::default()
        };
        assert!(invalid_epsilon.validate().is_err());

        let invalid_delta = DPConfig {
            delta: 1.5,
            ..Default::default()
        };
        assert!(invalid_delta.validate().is_err());
    }

    #[test]
    fn test_gradient_clipping() {
        let gradient = array![3.0f32, 4.0]; // Norm = 5.0
        let (clipped, original_norm) = clip_gradient(gradient.view(), 1.0);

        assert!((original_norm - 5.0).abs() < 1e-6);

        let clipped_norm = clipped.dot(&clipped).sqrt();
        assert!((clipped_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_clipping_when_below_threshold() {
        let gradient = array![0.3f32, 0.4]; // Norm = 0.5
        let (clipped, original_norm) = clip_gradient(gradient.view(), 1.0);

        assert!((original_norm - 0.5).abs() < 1e-6);
        assert_eq!(clipped, gradient);
    }

    #[test]
    fn test_gaussian_mechanism_sigma() {
        let mechanism = GaussianMechanism::new(1.0, 1e-5, 1.0);
        let sigma = mechanism.noise_scale();

        // σ = √(2 ln(1.25/1e-5)) * 1.0 / 1.0 ≈ 4.89
        assert!(sigma > 4.0 && sigma < 6.0);
    }

    #[test]
    fn test_laplace_mechanism_scale() {
        let mechanism = LaplaceMechanism::new(1.0, 1.0);
        let scale = mechanism.noise_scale();

        // b = 1.0 / 1.0 = 1.0
        assert!((scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_privacy_accountant() {
        let mut accountant = PrivacyAccountant::new(1.0, 1.0, 1e-5);

        assert_eq!(accountant.get_epsilon(), 0.0);
        assert!(accountant.can_continue(10.0));

        accountant.step(None);
        accountant.step(None);
        accountant.step(None);

        let epsilon = accountant.get_epsilon();
        assert!(epsilon > 0.0);
        assert_eq!(accountant.num_steps(), 3);

        accountant.reset();
        assert_eq!(accountant.num_steps(), 0);
    }

    #[test]
    fn test_differential_privacy_wrapper() {
        let config = DPConfig::default()
            .with_epsilon(1.0)
            .with_max_grad_norm(1.0);

        let mut dp = DifferentialPrivacy::new(config);
        let gradient = array![3.0f32, 4.0];

        let (privatized, metadata) = dp.privatize_gradient(gradient.view(), true, false);

        // Should be clipped to norm 1.0
        let norm = privatized.dot(&privatized).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(metadata.was_clipped);
        assert!((metadata.original_norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_clipping() {
        let mut gradients = HashMap::new();
        gradients.insert("node_1".to_string(), array![3.0f32, 4.0]); // Norm 5
        gradients.insert("node_2".to_string(), array![0.3f32, 0.4]); // Norm 0.5

        let (clipped, norms) = clip_gradients_batch(&gradients, 1.0);

        assert!((norms["node_1"] - 5.0).abs() < 1e-6);
        assert!((norms["node_2"] - 0.5).abs() < 1e-6);

        let norm_1 = clipped["node_1"].dot(&clipped["node_1"]).sqrt();
        let norm_2 = clipped["node_2"].dot(&clipped["node_2"]).sqrt();

        assert!((norm_1 - 1.0).abs() < 1e-6); // Clipped
        assert!((norm_2 - 0.5).abs() < 1e-6); // Not clipped
    }

    #[test]
    fn test_privacy_budget_computation() {
        let epsilon = compute_privacy_budget(1.0, 1.0, 10, 1e-5);
        assert!(epsilon > 0.0);

        // Higher noise multiplier = lower epsilon
        let epsilon_high_noise = compute_privacy_budget(2.0, 1.0, 10, 1e-5);
        assert!(epsilon_high_noise < epsilon);

        // More steps = higher epsilon
        let epsilon_more_steps = compute_privacy_budget(1.0, 1.0, 100, 1e-5);
        assert!(epsilon_more_steps > epsilon);
    }

    #[cfg(feature = "privacy")]
    #[test]
    fn test_noise_addition() {
        let gradient = array![1.0f32, 2.0, 3.0];
        let noisy = add_noise(gradient.view(), 0.1, 1.0, NoiseType::Gaussian);

        // Noisy gradient should be different
        assert_ne!(gradient, noisy);
        assert_eq!(gradient.len(), noisy.len());
    }
}
