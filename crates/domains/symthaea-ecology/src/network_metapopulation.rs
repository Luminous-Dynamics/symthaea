// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Directed patch-network occupancy dynamics and persistence threshold.
//!
//! Colonization is represented by a non-negative target-by-source matrix. The
//! extinction equilibrium loses stability when the spectral radius of the
//! next-generation matrix `diag(1 / extinction) * colonization` exceeds one.
//! This is a deterministic occupancy baseline, not a finite-patch stochastic
//! colonization process.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::integration::{MAX_TRAJECTORY_STEPS, validate_trajectory_request};

pub const MAX_NETWORK_PATCHES: usize = 64;
pub const MAX_NETWORK_TRAJECTORY_VALUES: usize = 2_000_000;
const SPECTRAL_TOLERANCE: f64 = 1.0e-12;
const MAX_SPECTRAL_ITERATIONS: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkPersistenceRegime {
    ExtinctionStable,
    Threshold,
    PersistencePossible,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkPersistenceDiagnostic {
    pub next_generation_spectral_radius: f64,
    pub regime: NetworkPersistenceRegime,
    pub iterations: usize,
    pub eigen_residual: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkOccupancySample {
    pub time: f64,
    pub occupancy: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatchNetworkMetapopulation {
    /// `colonization[target][source]`, inverse model-time.
    pub colonization: Vec<Vec<f64>>,
    /// Patch-specific local extinction rates, inverse model-time.
    pub extinction: Vec<f64>,
}

impl PatchNetworkMetapopulation {
    pub fn try_new(colonization: Vec<Vec<f64>>, extinction: Vec<f64>) -> Result<Self, ModelError> {
        let model = Self {
            colonization,
            extinction,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn patches(&self) -> usize {
        self.extinction.len()
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        let patches = self.extinction.len();
        if patches == 0 {
            return Err(ModelError::EmptySeries {
                series: "patch_extinction_rates",
            });
        }
        if patches > MAX_NETWORK_PATCHES {
            return Err(ModelError::DimensionMismatch {
                context: "patch_network_size",
                expected: MAX_NETWORK_PATCHES,
                found: patches,
            });
        }
        if self.colonization.len() != patches {
            return Err(ModelError::DimensionMismatch {
                context: "colonization_rows",
                expected: patches,
                found: self.colonization.len(),
            });
        }
        for extinction in &self.extinction {
            require_positive("patch_extinction_rate", *extinction)?;
        }
        for row in &self.colonization {
            if row.len() != patches {
                return Err(ModelError::DimensionMismatch {
                    context: "colonization_columns",
                    expected: patches,
                    found: row.len(),
                });
            }
            for rate in row {
                require_non_negative("colonization_rate", *rate)?;
            }
        }
        Ok(())
    }

    /// Spectral persistence diagnostic for the extinction equilibrium.
    ///
    /// Power iteration is applied to `I + K`, where
    /// `K = diag(1/e) C`. The identity shift removes period-two oscillation in
    /// bipartite non-negative networks while preserving the Perron eigenvector;
    /// the reported radius subtracts the shift exactly.
    pub fn persistence_diagnostic(&self) -> Result<NetworkPersistenceDiagnostic, ModelError> {
        self.validate()?;
        let n = self.patches();
        let mut vector = vec![1.0 / n as f64; n];
        let mut eigenvalue_shifted = 1.0;
        let mut iterations = 0;
        let mut converged = false;
        for iteration in 1..=MAX_SPECTRAL_ITERATIONS {
            let next = self.apply_shifted_next_generation(&vector);
            let norm = next.iter().sum::<f64>();
            require_positive("spectral_iteration_norm", norm)?;
            let normalized: Vec<f64> = next.iter().map(|value| value / norm).collect();
            let delta = normalized
                .iter()
                .zip(&vector)
                .map(|(next, previous)| (next - previous).abs())
                .fold(0.0_f64, f64::max);
            vector = normalized;
            eigenvalue_shifted = norm;
            iterations = iteration;
            if delta <= SPECTRAL_TOLERANCE {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(ModelError::NoConvergence {
                context: "patch_network_spectral_radius",
                iterations,
            });
        }
        let applied = self.apply_shifted_next_generation(&vector);
        eigenvalue_shifted = applied.iter().sum::<f64>();
        require_positive("shifted_next_generation_eigenvalue", eigenvalue_shifted)?;
        let residual = applied
            .iter()
            .zip(&vector)
            .map(|(value, component)| (value - eigenvalue_shifted * component).abs())
            .fold(0.0_f64, f64::max);
        let radius = (eigenvalue_shifted - 1.0).max(0.0);
        let threshold_tolerance = 1.0e-10 * radius.max(1.0);
        let regime = if (radius - 1.0).abs() <= threshold_tolerance {
            NetworkPersistenceRegime::Threshold
        } else if radius < 1.0 {
            NetworkPersistenceRegime::ExtinctionStable
        } else {
            NetworkPersistenceRegime::PersistencePossible
        };
        Ok(NetworkPersistenceDiagnostic {
            next_generation_spectral_radius: radius,
            regime,
            iterations,
            eigen_residual: residual,
        })
    }

    pub fn derivatives(&self, occupancy: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.validate_occupancy(occupancy, 0, "occupancy")?;
        let mut derivatives = Vec::with_capacity(self.patches());
        for target in 0..self.patches() {
            let colonization_pressure = self.colonization[target]
                .iter()
                .zip(occupancy)
                .map(|(rate, occupied)| rate * occupied)
                .sum::<f64>();
            derivatives.push(
                (1.0 - occupancy[target]) * colonization_pressure
                    - self.extinction[target] * occupancy[target],
            );
        }
        Ok(derivatives)
    }

    pub fn step_rk4(
        &self,
        occupancy: &[f64],
        dt: f64,
        step: usize,
    ) -> Result<Vec<f64>, ModelError> {
        require_positive("dt", dt)?;
        self.validate_occupancy(occupancy, step, "occupancy")?;
        let k1 = self.derivatives(occupancy)?;
        let stage2 = add_scaled(occupancy, &k1, 0.5 * dt);
        self.validate_occupancy(&stage2, step, "occupancy_stage_2")?;
        let k2 = self.derivatives(&stage2)?;
        let stage3 = add_scaled(occupancy, &k2, 0.5 * dt);
        self.validate_occupancy(&stage3, step, "occupancy_stage_3")?;
        let k3 = self.derivatives(&stage3)?;
        let stage4 = add_scaled(occupancy, &k3, dt);
        self.validate_occupancy(&stage4, step, "occupancy_stage_4")?;
        let k4 = self.derivatives(&stage4)?;
        let next: Vec<f64> = occupancy
            .iter()
            .zip(&k1)
            .zip(&k2)
            .zip(&k3)
            .zip(&k4)
            .map(|((((occupied, k1), k2), k3), k4)| {
                occupied + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            })
            .collect();
        self.validate_occupancy(&next, step, "occupancy")?;
        Ok(next)
    }

    pub fn try_simulate(
        &self,
        initial_occupancy: &[f64],
        dt: f64,
        steps: usize,
    ) -> Result<Vec<NetworkOccupancySample>, ModelError> {
        self.validate()?;
        validate_trajectory_request(dt, steps)?;
        self.validate_occupancy(initial_occupancy, 0, "initial_occupancy")?;
        let sample_count = steps.checked_add(1).ok_or(ModelError::TrajectoryTooLarge {
            requested: usize::MAX,
            maximum: MAX_TRAJECTORY_STEPS,
        })?;
        let values =
            sample_count
                .checked_mul(self.patches())
                .ok_or(ModelError::TrajectoryTooLarge {
                    requested: usize::MAX,
                    maximum: MAX_NETWORK_TRAJECTORY_VALUES,
                })?;
        if values > MAX_NETWORK_TRAJECTORY_VALUES {
            return Err(ModelError::TrajectoryTooLarge {
                requested: values,
                maximum: MAX_NETWORK_TRAJECTORY_VALUES,
            });
        }
        let mut samples = Vec::with_capacity(sample_count);
        let mut occupancy = initial_occupancy.to_vec();
        samples.push(NetworkOccupancySample {
            time: 0.0,
            occupancy: occupancy.clone(),
        });
        for step in 1..=steps {
            occupancy = self.step_rk4(&occupancy, dt, step)?;
            samples.push(NetworkOccupancySample {
                time: step as f64 * dt,
                occupancy: occupancy.clone(),
            });
        }
        Ok(samples)
    }

    fn apply_shifted_next_generation(&self, vector: &[f64]) -> Vec<f64> {
        (0..self.patches())
            .map(|target| {
                vector[target]
                    + self.colonization[target]
                        .iter()
                        .zip(vector)
                        .map(|(rate, source)| rate * source / self.extinction[target])
                        .sum::<f64>()
            })
            .collect()
    }

    fn validate_occupancy(
        &self,
        occupancy: &[f64],
        step: usize,
        component: &'static str,
    ) -> Result<(), ModelError> {
        if occupancy.len() != self.patches() {
            return Err(ModelError::DimensionMismatch {
                context: "patch_occupancy",
                expected: self.patches(),
                found: occupancy.len(),
            });
        }
        for value in occupancy {
            if !value.is_finite() || !(0.0..=1.0).contains(value) {
                return Err(ModelError::IntegrationDomainViolation {
                    step,
                    component,
                    value: *value,
                });
            }
        }
        Ok(())
    }
}

fn add_scaled(values: &[f64], derivatives: &[f64], scale: f64) -> Vec<f64> {
    values
        .iter()
        .zip(derivatives)
        .map(|(value, derivative)| value + scale * derivative)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_two_patch(colonization: f64, extinction: f64) -> PatchNetworkMetapopulation {
        PatchNetworkMetapopulation::try_new(
            vec![vec![0.0, colonization], vec![colonization, 0.0]],
            vec![extinction, extinction],
        )
        .unwrap()
    }

    #[test]
    fn spectral_threshold_recovers_symmetric_two_patch_result() {
        let below = symmetric_two_patch(0.1, 0.2)
            .persistence_diagnostic()
            .unwrap();
        assert!((below.next_generation_spectral_radius - 0.5).abs() < 1e-12);
        assert_eq!(below.regime, NetworkPersistenceRegime::ExtinctionStable);

        let threshold = symmetric_two_patch(0.2, 0.2)
            .persistence_diagnostic()
            .unwrap();
        assert!((threshold.next_generation_spectral_radius - 1.0).abs() < 1e-12);
        assert_eq!(threshold.regime, NetworkPersistenceRegime::Threshold);

        let above = symmetric_two_patch(0.5, 0.2)
            .persistence_diagnostic()
            .unwrap();
        assert!((above.next_generation_spectral_radius - 2.5).abs() < 1e-12);
        assert_eq!(above.regime, NetworkPersistenceRegime::PersistencePossible);
        assert!(above.eigen_residual < 1e-10);
    }

    #[test]
    fn persistent_symmetric_network_converges_to_expected_occupancy() {
        let model = symmetric_two_patch(0.5, 0.2);
        let samples = model.try_simulate(&[0.1, 0.1], 0.02, 10_000).unwrap();
        let final_state = &samples.last().unwrap().occupancy;
        let expected = 1.0 - 0.2 / 0.5;
        assert!((final_state[0] - expected).abs() < 1e-8);
        assert!((final_state[1] - expected).abs() < 1e-8);
    }

    #[test]
    fn disconnected_network_has_zero_next_generation_radius() {
        let model = PatchNetworkMetapopulation::try_new(
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![0.2, 0.3],
        )
        .unwrap();
        let diagnostic = model.persistence_diagnostic().unwrap();
        assert_eq!(diagnostic.next_generation_spectral_radius, 0.0);
        assert_eq!(
            diagnostic.regime,
            NetworkPersistenceRegime::ExtinctionStable
        );
    }

    #[test]
    fn asymmetric_bipartite_network_converges_after_identity_shift() {
        let model = PatchNetworkMetapopulation::try_new(
            vec![vec![0.0, 0.8], vec![0.2, 0.0]],
            vec![0.4, 0.1],
        )
        .unwrap();
        let diagnostic = model.persistence_diagnostic().unwrap();
        let expected: f64 = ((0.8_f64 / 0.4) * (0.2 / 0.1)).sqrt();
        assert!((diagnostic.next_generation_spectral_radius - expected).abs() < 1e-10);
        assert!(diagnostic.eigen_residual < 1e-10);
    }

    #[test]
    fn malformed_networks_and_excessive_trajectory_storage_are_rejected() {
        assert!(
            PatchNetworkMetapopulation::try_new(vec![vec![0.0, 1.0]], vec![0.2, 0.3],).is_err()
        );
        let patches = MAX_NETWORK_PATCHES;
        let model = PatchNetworkMetapopulation::try_new(
            vec![vec![0.0; patches]; patches],
            vec![0.2; patches],
        )
        .unwrap();
        assert!(matches!(
            model.try_simulate(&vec![0.1; patches], 0.1, 40_000),
            Err(ModelError::TrajectoryTooLarge { .. })
        ));
    }
}
