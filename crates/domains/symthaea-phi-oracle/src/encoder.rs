// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use symthaea_core::hdc::ContinuousHV;

/// Encodes observations from an external system into hyperdimensional vectors
/// for integration measurement.
///
/// Implementors map domain-specific data (sensor readings, node states, etc.)
/// into the HDC representation used by the spectral MIP pipeline.
pub trait SystemEncoder: Send + Sync {
    /// Encode one observation (e.g., one timestep of sensor readings) into a
    /// [`ContinuousHV`]. The observation slice length must equal
    /// [`num_variables`](Self::num_variables).
    fn encode(&self, observation: &[f64]) -> ContinuousHV;

    /// Number of variables in the system being measured.
    fn num_variables(&self) -> usize;

    /// Dimension of the output hypervectors produced by [`encode`](Self::encode).
    fn hv_dimension(&self) -> usize;

    /// Human-readable name of the system.
    fn system_name(&self) -> &str;
}

/// Maps N-dimensional time series into ContinuousHV via sparse random
/// projection (Johnson-Lindenstrauss style).
///
/// Each variable gets a random basis vector; an observation is encoded as the
/// weighted sum of these basis vectors.
pub struct TimeSeriesEncoder {
    name: String,
    basis_vectors: Vec<ContinuousHV>,
    num_variables: usize,
}

impl TimeSeriesEncoder {
    /// Create an encoder for a system with `num_variables` dimensions.
    ///
    /// - `hv_dim`: dimension of the output hypervectors (typically 256 for
    ///   lightweight oracle use).
    /// - `seed`: deterministic seed for reproducible random projections.
    pub fn new(num_variables: usize, hv_dim: usize, seed: u64) -> Self {
        let basis_vectors: Vec<ContinuousHV> = (0..num_variables)
            .map(|i| ContinuousHV::random(hv_dim, seed + i as u64))
            .collect();

        Self {
            name: format!("timeseries-{num_variables}var"),
            basis_vectors,
            num_variables,
        }
    }

    /// Create an encoder with a custom system name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl SystemEncoder for TimeSeriesEncoder {
    fn encode(&self, observation: &[f64]) -> ContinuousHV {
        debug_assert_eq!(observation.len(), self.num_variables);

        let weights: Vec<f32> = observation.iter().map(|&v| v as f32).collect();
        let refs: Vec<&ContinuousHV> = self.basis_vectors.iter().collect();
        ContinuousHV::weighted_bundle(&refs, &weights)
    }

    fn num_variables(&self) -> usize {
        self.num_variables
    }

    fn hv_dimension(&self) -> usize {
        self.basis_vectors.first().map_or(256, |hv| hv.dim())
    }

    fn system_name(&self) -> &str {
        &self.name
    }
}

/// Bypass encoder for users who already have a covariance/correlation matrix.
///
/// This encoder doesn't do real HV encoding — it exists so that
/// [`IntegrationOracle`](crate::IntegrationOracle) can accept pre-computed
/// covariance data via [`observe_covariance`](crate::IntegrationOracle::observe_covariance).
pub struct CovarianceEncoder {
    name: String,
    num_variables: usize,
}

impl CovarianceEncoder {
    /// Create a covariance-bypass encoder for a system with `num_variables`
    /// dimensions.
    pub fn new(num_variables: usize) -> Self {
        Self {
            name: format!("covariance-{num_variables}var"),
            num_variables,
        }
    }

    /// Create an encoder with a custom system name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl SystemEncoder for CovarianceEncoder {
    fn encode(&self, observation: &[f64]) -> ContinuousHV {
        // CovarianceEncoder is not meant for per-observation encoding;
        // users should call observe_covariance() instead.
        let values: Vec<f32> = observation.iter().map(|&v| v as f32).collect();
        ContinuousHV::from_vec(values)
    }

    fn num_variables(&self) -> usize {
        self.num_variables
    }

    fn hv_dimension(&self) -> usize {
        self.num_variables // covariance bypass operates on raw variable count
    }

    fn system_name(&self) -> &str {
        &self.name
    }
}

/// Minimal no-op encoder for [`IntegrationOracle::new_simple`](crate::IntegrationOracle::new_simple).
///
/// Produces identity-like HVs (raw values cast to f32). Not suitable for
/// temporal probing — use [`TimeSeriesEncoder`] for that.
pub(crate) struct SimpleEncoder {
    num_vars: usize,
}

impl SimpleEncoder {
    pub(crate) fn new(num_vars: usize) -> Self {
        Self { num_vars }
    }
}

impl SystemEncoder for SimpleEncoder {
    fn encode(&self, observation: &[f64]) -> ContinuousHV {
        let values: Vec<f32> = observation.iter().map(|&v| v as f32).collect();
        ContinuousHV::from_vec(values)
    }

    fn num_variables(&self) -> usize {
        self.num_vars
    }

    fn hv_dimension(&self) -> usize {
        self.num_vars
    }

    fn system_name(&self) -> &str {
        "simple"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_encoder_dimension() {
        let enc = TimeSeriesEncoder::new(10, 256, 42);
        assert_eq!(enc.num_variables(), 10);
        let obs = vec![1.0; 10];
        let hv = enc.encode(&obs);
        assert_eq!(hv.dim(), 256);
    }

    #[test]
    fn test_time_series_deterministic() {
        let enc = TimeSeriesEncoder::new(5, 128, 99);
        let obs = vec![0.5, -0.3, 1.0, 0.0, 0.7];
        let hv1 = enc.encode(&obs);
        let hv2 = enc.encode(&obs);
        assert_eq!(hv1.values, hv2.values);
    }

    #[test]
    fn test_covariance_encoder_basics() {
        let enc = CovarianceEncoder::new(8);
        assert_eq!(enc.num_variables(), 8);
        assert_eq!(enc.system_name(), "covariance-8var");
    }

    #[test]
    fn test_custom_name() {
        let enc = TimeSeriesEncoder::new(3, 64, 1).with_name("power-grid");
        assert_eq!(enc.system_name(), "power-grid");
    }
}
