//! Differential Privacy for Federated Learning
//!
//! Gradient clipping and noise addition for privacy-preserving FL.
//! Supports Gaussian noise (Box-Muller transform) and Rényi DP composition.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for differential privacy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DifferentialPrivacyConfig {
    /// L2 norm bound for gradient clipping
    pub clip_norm: f32,
    /// Noise multiplier (sigma = clip_norm * noise_multiplier)
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

    /// Get noise standard deviation
    pub fn sigma(&self) -> f32 {
        self.clip_norm * self.noise_multiplier
    }

    /// High privacy (epsilon ~0.1)
    pub fn high_privacy() -> Self {
        Self {
            clip_norm: 0.5,
            noise_multiplier: 2.0,
        }
    }

    /// Moderate privacy (epsilon ~1.0)
    pub fn moderate_privacy() -> Self {
        Self {
            clip_norm: 1.0,
            noise_multiplier: 1.1,
        }
    }

    /// Low privacy (epsilon ~10.0)
    pub fn low_privacy() -> Self {
        Self {
            clip_norm: 2.0,
            noise_multiplier: 0.5,
        }
    }
}

/// Clip gradient to have at most the specified L2 norm (in-place)
pub fn clip_gradient(gradient: &mut [f32], clip_norm: f32) {
    let norm: f32 = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > clip_norm {
        let scale = clip_norm / norm;
        for g in gradient.iter_mut() {
            *g *= scale;
        }
    }
}

/// Add Gaussian noise to gradient for differential privacy (in-place)
///
/// Uses Box-Muller transform to generate Gaussian noise without
/// requiring external distribution crates.
///
/// On native targets (with `std` feature), uses `thread_rng()`.
/// On WASM (without `std`), uses a seeded RNG based on gradient content.
pub fn add_gaussian_noise(gradient: &mut [f32], sigma: f32) {
    if sigma <= 0.0 {
        return;
    }

    #[cfg(feature = "std")]
    let mut rng = rand::thread_rng();

    #[cfg(not(feature = "std"))]
    let mut rng = {
        use rand::SeedableRng;
        // Derive seed from gradient content for reproducibility
        let seed: u64 = gradient.iter().enumerate().fold(0u64, |acc, (i, &v)| {
            acc.wrapping_add((v.to_bits() as u64).wrapping_mul(i as u64 + 1))
        });
        rand::rngs::SmallRng::seed_from_u64(seed)
    };

    let mut i = 0;
    while i + 1 < gradient.len() {
        let (n1, n2) = box_muller_pair(&mut rng, sigma);
        gradient[i] += n1;
        gradient[i + 1] += n2;
        i += 2;
    }

    if i < gradient.len() {
        let (n1, _) = box_muller_pair(&mut rng, sigma);
        gradient[i] += n1;
    }
}

/// Apply full DP pipeline: clip + noise
pub fn apply_dp(gradient: &mut [f32], config: &DifferentialPrivacyConfig) {
    clip_gradient(gradient, config.clip_norm);
    add_gaussian_noise(gradient, config.sigma());
}

/// Generate a pair of Gaussian random numbers using Box-Muller transform
fn box_muller_pair<R: Rng>(rng: &mut R, sigma: f32) -> (f32, f32) {
    let u1: f32 = rng.gen_range(1e-10_f32..1.0);
    let u2: f32 = rng.gen();

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;

    let z1 = r * theta.cos() * sigma;
    let z2 = r * theta.sin() * sigma;

    (z1, z2)
}

/// Compute L2 norm
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ============================================================================
// RDP Composition Tracking
// ============================================================================

/// Rényi Differential Privacy (RDP) budget tracker
///
/// Tracks cumulative privacy loss across multiple FL rounds using
/// Rényi Differential Privacy composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdpBudgetTracker {
    /// Accumulated RDP budget per alpha
    pub accumulated_rdp: Vec<f32>,
    /// Alpha values for RDP
    pub alphas: Vec<f32>,
    /// Total rounds tracked
    pub rounds: u32,
    /// Target delta for (epsilon, delta)-DP conversion
    pub target_delta: f64,
}

impl RdpBudgetTracker {
    pub fn new(target_delta: f64) -> Self {
        let alphas = vec![2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
        let accumulated_rdp = vec![0.0; alphas.len()];
        Self {
            accumulated_rdp,
            alphas,
            rounds: 0,
            target_delta,
        }
    }

    /// Record one round of Gaussian mechanism
    pub fn record_round(&mut self, sigma: f32) {
        self.rounds += 1;
        for (i, &alpha) in self.alphas.iter().enumerate() {
            // RDP for Gaussian mechanism: alpha / (2 * sigma^2)
            let rdp = alpha / (2.0 * sigma * sigma);
            self.accumulated_rdp[i] += rdp;
        }
    }

    /// Convert accumulated RDP to (epsilon, delta)-DP
    ///
    /// Returns the tightest epsilon across all alpha values.
    pub fn epsilon(&self) -> f64 {
        let mut best_eps = f64::INFINITY;
        for (i, &alpha) in self.alphas.iter().enumerate() {
            let rdp = self.accumulated_rdp[i] as f64;
            // RDP to (eps, delta)-DP conversion:
            // eps = rdp - (log(delta) + log(alpha-1)) / (alpha - 1) - log(1 - 1/alpha)
            let alpha_f64 = alpha as f64;
            if alpha_f64 <= 1.0 {
                continue;
            }
            let eps = rdp + (self.target_delta.ln() + (alpha_f64 - 1.0).ln()) / (alpha_f64 - 1.0)
                - (1.0 - 1.0 / alpha_f64).ln();
            // Note: when delta is very small, log(delta) is very negative,
            // which can make eps negative. Clamp to 0.
            if eps < best_eps && eps > 0.0 {
                best_eps = eps;
            }
        }
        if best_eps.is_infinite() {
            0.0
        } else {
            best_eps
        }
    }

    /// Check if budget is exhausted (exceeds target epsilon)
    pub fn is_exhausted(&self, target_epsilon: f64) -> bool {
        self.epsilon() > target_epsilon
    }
}

/// Privacy report for a pipeline execution
#[derive(Debug, Clone)]
pub struct PrivacyReport {
    /// Whether DP was applied
    pub dp_applied: bool,
    /// Clip norm used
    pub clip_norm: f32,
    /// Noise sigma used
    pub sigma: f32,
    /// Current epsilon estimate (if tracking)
    pub epsilon_estimate: Option<f64>,
    /// Total rounds tracked
    pub rounds_tracked: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_gradient() {
        let mut gradient = vec![3.0, 4.0]; // norm = 5
        clip_gradient(&mut gradient, 1.0);
        let norm = l2_norm(&gradient);
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_clip_no_change() {
        let mut gradient = vec![0.3, 0.4]; // norm = 0.5
        clip_gradient(&mut gradient, 1.0);
        assert!((gradient[0] - 0.3).abs() < 0.001);
        assert!((gradient[1] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_gaussian_noise_changes_gradient() {
        let mut gradient = vec![0.0; 100];
        add_gaussian_noise(&mut gradient, 1.0);
        let changed = gradient.iter().any(|&g| g != 0.0);
        assert!(changed, "Noise should change at least some values");
    }

    #[test]
    fn test_dp_presets() {
        let high = DifferentialPrivacyConfig::high_privacy();
        let moderate = DifferentialPrivacyConfig::moderate_privacy();
        let low = DifferentialPrivacyConfig::low_privacy();
        // High privacy has more noise_multiplier but lower clip_norm
        // The noise_multiplier is higher for stronger privacy
        assert!(high.noise_multiplier > moderate.noise_multiplier);
        assert!(moderate.noise_multiplier > low.noise_multiplier);
        // Clip norm is tighter for stronger privacy
        assert!(high.clip_norm < moderate.clip_norm);
        assert!(moderate.clip_norm < low.clip_norm);
    }

    #[test]
    fn test_apply_dp() {
        let mut gradient = vec![3.0, 4.0]; // norm = 5
        let config = DifferentialPrivacyConfig::new(1.0, 0.0);
        apply_dp(&mut gradient, &config);
        let norm = l2_norm(&gradient);
        assert!((norm - 1.0).abs() < 0.001, "Should be clipped to 1.0");
    }

    #[test]
    fn test_rdp_tracker() {
        let mut tracker = RdpBudgetTracker::new(1e-5);
        for _ in 0..100 {
            tracker.record_round(1.0);
        }
        assert_eq!(tracker.rounds, 100);
        let eps = tracker.epsilon();
        assert!(eps > 0.0, "Epsilon should be positive after 100 rounds");
    }

    #[test]
    fn test_rdp_budget_increases() {
        let mut tracker = RdpBudgetTracker::new(1e-5);
        tracker.record_round(1.0);
        let eps1 = tracker.epsilon();
        tracker.record_round(1.0);
        let eps2 = tracker.epsilon();
        assert!(eps2 >= eps1, "Budget should increase with more rounds");
    }
}
