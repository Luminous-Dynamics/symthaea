// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Streaming Φ Computation and Real-Time Monitoring
//!
//! Provides incremental Φ computation for real-time applications,
//! avoiding full O(n²) recomputation when only a few components change.

use crate::hdc::unified_hv::ContinuousHV;

use super::{ApproximateMIPFinder, ContinuousEntropyEstimator, PhiAlert, PhiAlertType};

/// Streaming Φ calculator for incremental updates
///
/// Maintains a cached MI matrix and entropy values, supporting O(n)
/// updates when a single component changes instead of O(n²) full recomputation.
#[derive(Debug, Clone)]
pub struct StreamingPhiCalculator {
    /// Current component entropies
    entropies: Vec<f64>,
    /// Current MI matrix (upper triangle stored)
    mi_matrix: Vec<Vec<f64>>,
    /// Current system EI
    system_ei: f64,
    /// Estimator for entropy computation
    estimator: ContinuousEntropyEstimator,
    /// Number of components
    n: usize,
}

impl StreamingPhiCalculator {
    /// Initialize from a set of components
    pub fn new(components: &[ContinuousHV]) -> Self {
        let estimator = ContinuousEntropyEstimator::fast();
        let n = components.len();

        // Compute initial entropies
        let entropies: Vec<f64> = components.iter().map(|c| estimator.entropy(c)).collect();

        // Compute initial MI matrix
        let mut mi_matrix = vec![vec![0.0; n]; n];
        let mut system_ei = 0.0;

        for i in 0..n {
            mi_matrix[i][i] = entropies[i];
            for j in (i + 1)..n {
                let mi = estimator.mutual_information_fast(&components[i], &components[j]);
                mi_matrix[i][j] = mi;
                mi_matrix[j][i] = mi;
                system_ei += mi;
            }
        }

        Self {
            entropies,
            mi_matrix,
            system_ei,
            estimator,
            n,
        }
    }

    /// Update when a single component changes
    ///
    /// Complexity: O(n) instead of O(n²)
    pub fn update_component(
        &mut self,
        index: usize,
        new_component: &ContinuousHV,
        all_components: &[ContinuousHV],
    ) {
        if index >= self.n {
            return;
        }

        // Update entropy for changed component
        let _old_entropy = self.entropies[index];
        let new_entropy = self.estimator.entropy(new_component);
        self.entropies[index] = new_entropy;
        self.mi_matrix[index][index] = new_entropy;

        // Update MI with all other components
        for j in 0..self.n {
            if j != index {
                // Subtract old MI from system EI
                let old_mi = self.mi_matrix[index][j];
                self.system_ei -= old_mi;

                // Compute new MI
                let new_mi = self
                    .estimator
                    .mutual_information_fast(new_component, &all_components[j]);
                self.mi_matrix[index][j] = new_mi;
                self.mi_matrix[j][index] = new_mi;

                // Add new MI to system EI
                self.system_ei += new_mi;
            }
        }
    }

    /// Update when multiple components change
    pub fn update_components(
        &mut self,
        indices: &[usize],
        new_components: &[&ContinuousHV],
        all_components: &[ContinuousHV],
    ) {
        for (idx, &i) in indices.iter().enumerate() {
            if i < self.n && idx < new_components.len() {
                self.update_component(i, new_components[idx], all_components);
            }
        }
    }

    /// Get current system EI
    pub fn system_ei(&self) -> f64 {
        self.system_ei
    }

    /// Get current entropies
    pub fn entropies(&self) -> &[f64] {
        &self.entropies
    }

    /// Get current MI matrix
    pub fn mi_matrix(&self) -> &[Vec<f64>] {
        &self.mi_matrix
    }

    /// Compute approximate Φ using cached values
    ///
    /// Uses the cached MI matrix to avoid recomputation.
    pub fn compute_phi_fast(&self, _components: &[ContinuousHV]) -> f64 {
        if self.n < 2 {
            return 0.0;
        }

        // Use approximate MIP finder with cached MI matrix
        let finder = ApproximateMIPFinder::new();
        let mip = finder.find_mip_graph_cut(&self.mi_matrix);

        // Compute MIP EI
        let mut mip_ei = 0.0;

        // EI within part A
        for i in 0..mip.part_a.len() {
            for j in (i + 1)..mip.part_a.len() {
                mip_ei += self.mi_matrix[mip.part_a[i]][mip.part_a[j]];
            }
        }

        // EI within part B
        for i in 0..mip.part_b.len() {
            for j in (i + 1)..mip.part_b.len() {
                mip_ei += self.mi_matrix[mip.part_b[i]][mip.part_b[j]];
            }
        }

        (self.system_ei - mip_ei).max(0.0)
    }

    /// Get Φ change since last update (for threshold detection)
    pub fn phi_delta(&self, previous_phi: f64, components: &[ContinuousHV]) -> f64 {
        let current_phi = self.compute_phi_fast(components);
        current_phi - previous_phi
    }
}

/// Real-time Φ monitor with threshold alerting
#[derive(Debug, Clone)]
pub struct PhiMonitor {
    /// Streaming calculator
    calculator: StreamingPhiCalculator,
    /// History of Φ values
    history: Vec<f64>,
    /// Maximum history length
    max_history: usize,
    /// Alert threshold (significant Φ change)
    alert_threshold: f64,
}

impl PhiMonitor {
    /// Create a new monitor
    pub fn new(components: &[ContinuousHV], alert_threshold: f64) -> Self {
        let calculator = StreamingPhiCalculator::new(components);
        let initial_phi = calculator.compute_phi_fast(components);

        Self {
            calculator,
            history: vec![initial_phi],
            max_history: 1000,
            alert_threshold,
        }
    }

    /// Update and check for alerts
    pub fn update(
        &mut self,
        index: usize,
        new_component: &ContinuousHV,
        all_components: &[ContinuousHV],
    ) -> Option<PhiAlert> {
        let previous_phi = *self.history.last().unwrap_or(&0.0);

        self.calculator
            .update_component(index, new_component, all_components);
        let current_phi = self.calculator.compute_phi_fast(all_components);

        // Maintain history
        self.history.push(current_phi);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        // Check for alert
        let delta = current_phi - previous_phi;
        if delta.abs() > self.alert_threshold {
            Some(PhiAlert {
                previous_phi,
                current_phi,
                delta,
                alert_type: if delta > 0.0 {
                    PhiAlertType::Integration
                } else {
                    PhiAlertType::Disintegration
                },
            })
        } else {
            None
        }
    }

    /// Get current Φ
    pub fn current_phi(&self) -> f64 {
        *self.history.last().unwrap_or(&0.0)
    }

    /// Get Φ history
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Get trend (positive = increasing integration)
    pub fn trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let recent = &self.history[self.history.len().saturating_sub(10)..];
        if recent.len() < 2 {
            return 0.0;
        }
        (recent.last().unwrap() - recent.first().unwrap()) / recent.len() as f64
    }
}
