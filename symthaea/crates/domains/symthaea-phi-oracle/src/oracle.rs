// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use symthaea_core::consciousness_metrics::{SpectralMIPConfig, SpectralMIPFinder};

use crate::encoder::SystemEncoder;
use crate::error::OracleError;
use crate::result::{HierarchicalReport, IntegrationReport};
use crate::temporal::TemporalProber;
use crate::window::ObservationWindow;

/// Configuration for the integration oracle.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OracleConfig {
    /// Number of observations in the sliding window. Default: 50.
    pub window_size: usize,

    /// Diagonal regularization for the covariance matrix. Default: 1e-6.
    pub regularization: f64,

    /// Timescale values for CfC temporal probing. Set to empty to disable
    /// temporal coherence measurement. Default: `[0.01, 0.1, 1.0, 10.0]`.
    pub temporal_probes: Vec<f64>,

    /// Seed for deterministic random projections (used by encoder and temporal
    /// prober). Default: 42.
    pub seed: u64,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            regularization: 1e-6,
            temporal_probes: vec![0.01, 0.1, 1.0, 10.0],
            seed: 42,
        }
    }
}

impl OracleConfig {
    fn validate(&self) -> Result<(), OracleError> {
        if self.window_size < 3 {
            return Err(OracleError::InvalidConfig(
                "window_size must be >= 3".into(),
            ));
        }
        Ok(())
    }
}

/// Integration Spectrometer — measures the structural coherence of external
/// systems via spectral MIP analysis and CfC temporal probing.
///
/// # Pipeline
///
/// 1. **Observe**: raw observations accumulate in a sliding window
/// 2. **Evolve** (optional): encoded HVs are fed through CfC temporal probes
/// 3. **Measure**: build variable-level covariance → spectral MIP → integration index
///
/// The key difference from v0.1.0: covariance is computed over the **original
/// system variables** (not random-projection HV dimensions), preserving the
/// correlation structure that the spectral MIP algorithm needs.
///
/// # Naming
///
/// Externally, the measured quantity is called "integration index" or
/// "structural coherence." Internally it uses IIT-inspired spectral MIP
/// algorithms.
pub struct IntegrationOracle {
    encoder: Box<dyn SystemEncoder>,
    /// Used only for `compute_from_covariance()` — no longer receives `push()`.
    mip_finder: SpectralMIPFinder,
    temporal_prober: Option<TemporalProber>,
    /// Sliding window of raw observations for variable-level covariance.
    obs_window: ObservationWindow,
    config: OracleConfig,
    num_observations: usize,
    /// When set, the MIP was fed via `observe_covariance()` and we have a
    /// pre-built covariance matrix ready.
    covariance_mode: Option<CovarianceState>,
}

struct CovarianceState {
    cov: Vec<f64>,
    n: usize,
    window_used: usize,
}

impl IntegrationOracle {
    /// Create a new integration oracle with the given encoder and
    /// configuration.
    pub fn new(encoder: Box<dyn SystemEncoder>, config: OracleConfig) -> Result<Self, OracleError> {
        config.validate()?;

        let num_vars = encoder.num_variables();

        // MIP finder is used only for compute_from_covariance() now.
        // num_components = num_vars (we operate on original variables).
        let mip_config = SpectralMIPConfig {
            num_components: num_vars,
            window_size: config.window_size,
            min_samples: 3.min(config.window_size),
            regularization: config.regularization,
            normalize_variance: true,
            temporal_decorrelation: true,
        };
        let mip_finder = SpectralMIPFinder::new(mip_config);

        let obs_window =
            ObservationWindow::new(num_vars, config.window_size, config.regularization);

        // Use the encoder's HV dimension so the temporal prober neurons
        // match the encoded observation vectors.
        let hv_dim = encoder.hv_dimension();
        let temporal_prober = if config.temporal_probes.is_empty() {
            None
        } else {
            Some(TemporalProber::new(
                hv_dim,
                &config.temporal_probes,
                config.seed,
            ))
        };

        Ok(Self {
            encoder,
            mip_finder,
            temporal_prober,
            obs_window,
            config,
            num_observations: 0,
            covariance_mode: None,
        })
    }

    /// Create an oracle for a system with `num_vars` variables, without
    /// requiring a custom encoder. Uses a no-op encoder internally and disables
    /// temporal probing. This is the simplest path for raw observation data.
    ///
    /// Requires `num_vars >= 2`.
    pub fn new_simple(num_vars: usize, mut config: OracleConfig) -> Result<Self, OracleError> {
        if num_vars < 2 {
            return Err(OracleError::InvalidConfig("num_vars must be >= 2".into()));
        }
        // SimpleEncoder doesn't support meaningful temporal probing
        config.temporal_probes = vec![];
        let encoder = crate::encoder::SimpleEncoder::new(num_vars);
        Self::new(Box::new(encoder), config)
    }

    /// Create an oracle with default configuration.
    pub fn with_defaults(encoder: Box<dyn SystemEncoder>) -> Result<Self, OracleError> {
        Self::new(encoder, OracleConfig::default())
    }

    /// Feed one observation. The observation slice must have length equal to
    /// the encoder's [`num_variables`](SystemEncoder::num_variables).
    ///
    /// Internally: push raw observation into the sliding window (for
    /// variable-level covariance). If temporal probes are configured, also
    /// encode → CfC evolve.
    pub fn observe(&mut self, observation: &[f64]) -> Result<(), OracleError> {
        let expected = self.encoder.num_variables();
        if observation.len() != expected {
            return Err(OracleError::DimensionMismatch {
                expected,
                got: observation.len(),
            });
        }

        // Clear any previous covariance-mode state
        self.covariance_mode = None;

        // Push raw observation into sliding window (variable-level covariance)
        self.obs_window.push(observation);

        // Feed temporal prober via encoded HV (lazy — only if probes configured)
        if let Some(prober) = &mut self.temporal_prober {
            let encoded = self.encoder.encode(observation);
            prober.observe(&encoded);
        }

        self.num_observations += 1;

        Ok(())
    }

    /// Feed a pre-built covariance matrix directly, bypassing encoding and CfC.
    ///
    /// - `covariance`: row-major n x n covariance matrix (n*n elements).
    /// - `n`: dimension of the covariance matrix.
    /// - `num_observations`: how many data points were used to build this
    ///   covariance matrix.
    ///
    /// After calling this, [`measure`](Self::measure) will use the covariance
    /// directly via `SpectralMIPFinder::compute_from_covariance()`.
    pub fn observe_covariance(
        &mut self,
        covariance: &[f64],
        n: usize,
        num_observations: usize,
    ) -> Result<(), OracleError> {
        if covariance.len() != n * n {
            return Err(OracleError::InvalidCovariance {
                expected: n * n,
                got: covariance.len(),
            });
        }

        self.covariance_mode = Some(CovarianceState {
            cov: covariance.to_vec(),
            n,
            window_used: num_observations,
        });
        self.num_observations = num_observations;

        Ok(())
    }

    /// Compute the integration report from accumulated observations.
    ///
    /// Returns `None` if insufficient data (need at least 3 observations and
    /// 2 variables for meaningful integration measurement).
    pub fn measure(&self) -> Option<IntegrationReport> {
        // Get covariance matrix and dimension
        let (cov, n) = if let Some(cov_state) = &self.covariance_mode {
            (cov_state.cov.clone(), cov_state.n)
        } else {
            if self.obs_window.len() < 3 || self.obs_window.num_vars() < 2 {
                return None;
            }
            (
                self.obs_window.build_covariance(),
                self.obs_window.num_vars(),
            )
        };

        let window_used = self
            .covariance_mode
            .as_ref()
            .map_or(self.obs_window.len(), |cs| cs.window_used);
        let mip_result = self
            .mip_finder
            .compute_from_covariance(&cov, n, window_used)?;

        let temporal_coherence = self.temporal_prober.as_ref().and_then(|p| p.compute());

        let normalized_index = if mip_result.total_mi > 1e-12 {
            (mip_result.phi / mip_result.total_mi).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let variable_contributions =
            Self::compute_variable_contributions(&cov, n, mip_result.total_mi);

        let betti_numbers = self.compute_betti_numbers(&cov, n);
        let persistent_cycles = self.compute_persistence(&cov, n);

        Some(IntegrationReport {
            integration_index: mip_result.phi,
            total_mutual_information: mip_result.total_mi,
            minimum_information_partition: (
                mip_result.mip.part_a.clone(),
                mip_result.mip.part_b.clone(),
            ),
            spectral_order: mip_result.spectral_order.clone(),
            temporal_coherence,
            normalized_index,
            variable_contributions,
            betti_numbers,
            persistent_cycles,
            num_observations: self.num_observations,
        })
    }

    /// Compute topological Betti numbers from the covariance structure.
    ///
    /// beta_0: number of connected components.
    /// beta_1: number of cycles (Euler characteristic approach).
    fn compute_betti_numbers(&self, cov: &[f64], n: usize) -> [usize; 3] {
        if n == 0 {
            return [0, 0, 0];
        }

        // Build adjacency matrix based on significant correlation
        let mut adj = vec![vec![false; n]; n];
        let threshold = 0.3; // Minimum correlation to consider an edge
        let mut edge_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let var_i = cov[i * n + i];
                let var_j = cov[j * n + j];
                if var_i > 1e-10 && var_j > 1e-10 {
                    let r = (cov[i * n + j] / (var_i * var_j).sqrt()).abs();
                    if r > threshold {
                        adj[i][j] = true;
                        adj[j][i] = true;
                        edge_count += 1;
                    }
                }
            }
        }

        // beta_0: Connected Components (BFS)
        let mut visited = vec![false; n];
        let mut beta_0 = 0;
        for i in 0..n {
            if !visited[i] {
                beta_0 += 1;
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(i);
                visited[i] = true;
                while let Some(u) = queue.pop_front() {
                    for v in 0..n {
                        if adj[u][v] && !visited[v] {
                            visited[v] = true;
                            queue.push_back(v);
                        }
                    }
                }
            }
        }

        // beta_1: number of cycles in the graph
        // For a general graph, beta_1 = edges - nodes + beta_0
        let beta_1 = if edge_count + beta_0 >= n {
            edge_count + beta_0 - n
        } else {
            0
        };

        [beta_0, beta_1, 0]
    }

    /// Compute persistent topological cycles across a filtration spectrum.
    ///
    /// Sweeps correlation threshold from 1.0 down to 0.0 and tracks
    /// the 'lifespan' of one-dimensional holes (beta_1 cycles).
    fn compute_persistence(&self, cov: &[f64], n: usize) -> Vec<crate::result::PersistentCycle> {
        use crate::result::PersistentCycle;
        if n < 3 {
            return Vec::new();
        }

        let mut results = Vec::new();
        let steps = 20;
        let mut last_beta_1 = 0;
        let mut active_cycles: Vec<PersistentCycle> = Vec::new();

        for s in 0..=steps {
            let threshold = 1.0 - (s as f64 / steps as f64);
            let b = self.compute_betti_numbers(cov, n);
            let current_beta_1 = b[1];

            if current_beta_1 > last_beta_1 {
                // Birth of cycles
                for _ in 0..(current_beta_1 - last_beta_1) {
                    active_cycles.push(PersistentCycle {
                        birth: threshold,
                        death: 0.0,
                        lifespan: 0.0,
                        participants: Vec::new(),
                    });
                }
            } else if current_beta_1 < last_beta_1 {
                // Death of cycles
                for _ in 0..(last_beta_1 - current_beta_1) {
                    if let Some(mut cycle) = active_cycles.pop() {
                        cycle.death = threshold;
                        cycle.lifespan = cycle.birth - cycle.death;
                        // Only keep significant cycles (Persistence Gate)
                        if cycle.lifespan > 0.15 {
                            results.push(cycle);
                        }
                    }
                }
            }
            last_beta_1 = current_beta_1;
        }

        // Close any remaining active cycles
        for mut cycle in active_cycles {
            cycle.death = 0.0;
            cycle.lifespan = cycle.birth;
            if cycle.lifespan > 0.15 {
                results.push(cycle);
            }
        }

        results
    }

    /// Reset all internal state for a new measurement window.
    pub fn reset(&mut self) {
        self.mip_finder.reset();
        self.obs_window.clear();
        if let Some(prober) = &mut self.temporal_prober {
            prober.reset(self.encoder.hv_dimension(), self.config.seed);
        }
        self.num_observations = 0;
        self.covariance_mode = None;
    }

    /// Number of observations accumulated since last reset.
    pub fn num_observations(&self) -> usize {
        self.num_observations
    }

    /// Whether the oracle has enough data to produce a result.
    pub fn ready(&self) -> bool {
        if self.covariance_mode.is_some() {
            return true;
        }
        self.obs_window.len() >= 3 && self.obs_window.num_vars() >= 2
    }

    /// Compute hierarchical (multi-scale) integration by subsampling the
    /// covariance matrix at progressively coarser scales.
    ///
    /// Returns `None` if insufficient data or fewer than 4 variables (need
    /// at least 2 scales for a meaningful hierarchy).
    pub fn measure_hierarchical(&self) -> Option<HierarchicalReport> {
        let finest = self.measure()?;

        let (cov, n) = if let Some(cov_state) = &self.covariance_mode {
            (cov_state.cov.clone(), cov_state.n)
        } else {
            (
                self.obs_window.build_covariance(),
                self.obs_window.num_vars(),
            )
        };

        let window_used = self
            .covariance_mode
            .as_ref()
            .map_or(self.obs_window.len(), |cs| cs.window_used);

        let mut scales = vec![n];
        let mut phi_by_scale = vec![finest.integration_index];

        // Subsample: take every 2nd, 4th, ... variable
        let mut stride = 2;
        while n / stride >= 2 {
            let sub_n = n / stride;
            let indices: Vec<usize> = (0..sub_n).map(|i| i * stride).collect();

            // Build subsampled covariance
            let mut sub_cov = vec![0.0; sub_n * sub_n];
            for (si, &i) in indices.iter().enumerate() {
                for (sj, &j) in indices.iter().enumerate() {
                    sub_cov[si * sub_n + sj] = cov[i * n + j];
                }
            }

            if let Some(result) =
                self.mip_finder
                    .compute_from_covariance(&sub_cov, sub_n, window_used)
            {
                scales.push(sub_n);
                phi_by_scale.push(result.phi);
            }
            stride *= 2;
        }

        Some(HierarchicalReport {
            finest,
            scales,
            phi_by_scale,
        })
    }

    /// Reference to the system encoder.
    pub fn encoder(&self) -> &dyn SystemEncoder {
        &*self.encoder
    }

    /// Compute leave-one-out contributions: for each variable i,
    /// contribution[i] = total_mi - MI(system without variable i).
    fn compute_variable_contributions(cov: &[f64], n: usize, total_mi: f64) -> Vec<f64> {
        if n < 3 {
            // With 2 variables, removing one leaves a single variable (MI = 0)
            return vec![total_mi; n];
        }

        (0..n)
            .map(|exclude| {
                let sub_mi = Self::mi_of_submatrix(cov, n, exclude);
                total_mi - sub_mi
            })
            .collect()
    }

    /// Total pairwise MI of an (n-1)×(n-1) submatrix excluding one variable.
    /// MI(i,j) = -0.5 * ln(1 - r²) where r = cov(i,j)/sqrt(var_i * var_j).
    fn mi_of_submatrix(cov: &[f64], n: usize, exclude: usize) -> f64 {
        let indices: Vec<usize> = (0..n).filter(|&i| i != exclude).collect();
        let m = indices.len();
        let mut mi = 0.0;
        for ai in 0..m {
            for bi in (ai + 1)..m {
                let i = indices[ai];
                let j = indices[bi];
                let var_i = cov[i * n + i];
                let var_j = cov[j * n + j];
                if var_i > 1e-15 && var_j > 1e-15 {
                    let r = cov[i * n + j] / (var_i * var_j).sqrt();
                    let r2 = (r * r).min(1.0 - 1e-15);
                    mi += -0.5 * (1.0 - r2).ln();
                }
            }
        }
        mi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::TimeSeriesEncoder;

    #[test]
    fn test_oracle_creation() {
        let enc = TimeSeriesEncoder::new(8, 256, 42);
        let oracle = IntegrationOracle::with_defaults(Box::new(enc));
        assert!(oracle.is_ok());
    }

    #[test]
    fn test_oracle_dimension_mismatch() {
        let enc = TimeSeriesEncoder::new(5, 64, 1);
        let mut oracle = IntegrationOracle::with_defaults(Box::new(enc)).unwrap();
        let result = oracle.observe(&[1.0, 2.0, 3.0]); // 3 != 5
        assert!(result.is_err());
    }

    #[test]
    fn test_oracle_not_ready_initially() {
        let enc = TimeSeriesEncoder::new(4, 64, 1);
        let oracle = IntegrationOracle::with_defaults(Box::new(enc)).unwrap();
        assert!(!oracle.ready());
        assert!(oracle.measure().is_none());
    }

    #[test]
    fn test_invalid_config() {
        let enc = TimeSeriesEncoder::new(4, 64, 1);
        let config = OracleConfig {
            window_size: 1, // too small
            ..Default::default()
        };
        let result = IntegrationOracle::new(Box::new(enc), config);
        assert!(result.is_err());
    }

    #[test]
    fn test_covariance_invalid_size() {
        let enc = TimeSeriesEncoder::new(4, 64, 1);
        let mut oracle = IntegrationOracle::with_defaults(Box::new(enc)).unwrap();
        let result = oracle.observe_covariance(&[1.0; 9], 4, 100); // 9 != 4*4=16
        assert!(result.is_err());
    }

    #[test]
    fn test_reset_clears_state() {
        let enc = TimeSeriesEncoder::new(4, 64, 1);
        let mut oracle = IntegrationOracle::with_defaults(Box::new(enc)).unwrap();
        for i in 0..10 {
            let obs = vec![i as f64 * 0.1; 4];
            oracle.observe(&obs).unwrap();
        }
        assert_eq!(oracle.num_observations(), 10);
        oracle.reset();
        assert_eq!(oracle.num_observations(), 0);
        assert!(!oracle.ready());
    }
}
