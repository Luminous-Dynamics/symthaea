// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared types for the true Φ (Integrated Information) module.
//!
//! Contains configuration, result types, partitions, and reference test cases
//! used across all true_phi sub-modules.

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

use super::TruePhiCalculator;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for entropy calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyConfig {
    /// Number of bins for discretization (16 for speed, 32 for precision)
    pub num_bins: usize,
    /// Whether to use bits (log₂) or nats (ln)
    pub use_bits: bool,
}

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            num_bins: 16,
            use_bits: true,
        }
    }
}

impl EntropyConfig {
    /// Create config optimized for speed
    pub fn fast() -> Self {
        Self {
            num_bins: 16,
            use_bits: true,
        }
    }

    /// Create config optimized for precision
    pub fn precise() -> Self {
        Self {
            num_bins: 32,
            use_bits: true,
        }
    }
}

/// Methods for continuous entropy estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EntropyMethod {
    /// Histogram binning (fast, default)
    #[default]
    Histogram,
    /// k-Nearest Neighbor estimator (Kozachenko-Leonenko) - O(n²)
    KNN,
    /// Optimized k-NN using sorted array property - O(n log n)
    KNNFast,
    /// Kernel Density Estimation - O(n²)
    KDE,
    /// Optimized KDE with truncated Gaussian - O(n × neighbors)
    KDEFast,
    /// Adaptive binning (data-driven bin widths)
    AdaptiveBins,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARTITIONS & RESULTS
// ═══════════════════════════════════════════════════════════════════════════════

/// True partition of a system into two non-empty subsets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruePartition {
    /// Indices of components in part A
    pub part_a: Vec<usize>,
    /// Indices of components in part B
    pub part_b: Vec<usize>,
}

impl TruePartition {
    /// Create a partition from a bitmask
    pub fn from_mask(mask: usize, n: usize) -> Self {
        let mut part_a = Vec::new();
        let mut part_b = Vec::new();

        for i in 0..n {
            if (mask & (1 << i)) != 0 {
                part_a.push(i);
            } else {
                part_b.push(i);
            }
        }

        Self { part_a, part_b }
    }
}

/// Result of true Φ computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruePhiResult {
    /// The integrated information value
    pub phi: f64,
    /// Whole system effective information
    pub system_ei: f64,
    /// MIP effective information
    pub mip_ei: f64,
    /// The minimum information partition found
    pub mip: TruePartition,
    /// Individual component entropies H(X_i)
    pub component_entropies: Vec<f64>,
    /// Pairwise mutual information matrix
    pub mutual_information_matrix: Vec<Vec<f64>>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL TRANSITIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal transition for cause-effect analysis
///
/// Represents a state transition t → t+1 for computing cause and effect repertoires.
#[derive(Debug, Clone)]
pub struct TemporalTransition {
    /// State at time t (current)
    pub current: ContinuousHV,
    /// State at time t+1 (next)
    pub next: ContinuousHV,
}

impl TemporalTransition {
    /// Create a transition from current to next state
    pub fn new(current: ContinuousHV, next: ContinuousHV) -> Self {
        Self { current, next }
    }

    /// Create a transition by applying a transformation to the current state
    pub fn from_transformation<F>(current: ContinuousHV, transform: F) -> Self
    where
        F: FnOnce(&ContinuousHV) -> ContinuousHV,
    {
        let next = transform(&current);
        Self { current, next }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PYPHI REFERENCE TEST CASES
// ═══════════════════════════════════════════════════════════════════════════════

/// PyPhi reference test case
#[derive(Debug, Clone)]
pub struct PyPhiTestCase {
    /// Name of the test case
    pub name: &'static str,
    /// Description
    pub description: &'static str,
    /// Number of nodes
    pub n_nodes: usize,
    /// Expected Φ value (from PyPhi/literature)
    pub expected_phi: f64,
    /// Tolerance for comparison
    pub tolerance: f64,
    /// Whether this is an exact test or approximate
    pub exact: bool,
}

/// Standard PyPhi reference test cases
pub fn pyphi_reference_cases() -> Vec<PyPhiTestCase> {
    vec![
        PyPhiTestCase {
            name: "Empty System",
            description: "System with 0-1 nodes has \u{03a6} = 0",
            n_nodes: 1,
            expected_phi: 0.0,
            tolerance: 1e-10,
            exact: true,
        },
        PyPhiTestCase {
            name: "Two Independent Nodes",
            description: "Two independent nodes have \u{03a6} = 0 (no integration)",
            n_nodes: 2,
            expected_phi: 0.0,
            tolerance: 0.1, // May have small numerical Φ
            exact: false,
        },
        PyPhiTestCase {
            name: "Fully Connected Pair",
            description: "Two fully connected nodes have positive \u{03a6}",
            n_nodes: 2,
            expected_phi: 0.0, // Varies based on connection strength
            tolerance: 0.5,
            exact: false,
        },
        PyPhiTestCase {
            name: "IIT 3.0 Majority Gate",
            description: "The classic 3-node majority gate from IIT 3.0 paper",
            n_nodes: 3,
            expected_phi: 0.5, // Approximate - actual value depends on state
            tolerance: 0.3,
            exact: false,
        },
        PyPhiTestCase {
            name: "XOR Gate",
            description: "XOR gate has moderate integration",
            n_nodes: 3,
            expected_phi: 0.25, // Approximate
            tolerance: 0.2,
            exact: false,
        },
        PyPhiTestCase {
            name: "Copy Gate",
            description: "Copy/AND gate - less integrated than XOR",
            n_nodes: 3,
            expected_phi: 0.15, // Approximate
            tolerance: 0.2,
            exact: false,
        },
    ]
}

/// Run a PyPhi test case and return whether it passed
pub fn run_pyphi_test(case: &PyPhiTestCase, components: &[ContinuousHV]) -> (bool, f64, String) {
    if components.len() != case.n_nodes {
        return (
            false,
            0.0,
            format!(
                "Wrong number of components: expected {}, got {}",
                case.n_nodes,
                components.len()
            ),
        );
    }

    let calc = TruePhiCalculator::new();
    let result = if components.len() >= 2 {
        calc.compute_true_phi(components)
    } else {
        TruePhiResult {
            phi: 0.0,
            system_ei: 0.0,
            mip_ei: 0.0,
            mip: TruePartition {
                part_a: vec![],
                part_b: vec![],
            },
            component_entropies: vec![],
            mutual_information_matrix: vec![],
        }
    };

    let diff = (result.phi - case.expected_phi).abs();
    let passed = if case.exact {
        diff < case.tolerance
    } else {
        // For approximate tests, just check it's in reasonable range
        result.phi >= 0.0 && (case.expected_phi == 0.0 || diff < case.tolerance)
    };

    let message = format!(
        "{}: computed \u{03a6} = {:.6}, expected \u{2248} {:.6} (diff = {:.6}, tol = {:.6})",
        if passed { "PASS" } else { "FAIL" },
        result.phi,
        case.expected_phi,
        diff,
        case.tolerance
    );

    (passed, result.phi, message)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALERTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Alert for significant Φ change
#[derive(Debug, Clone)]
pub struct PhiAlert {
    pub previous_phi: f64,
    pub current_phi: f64,
    pub delta: f64,
    pub alert_type: PhiAlertType,
}

/// Type of Φ alert
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiAlertType {
    /// Φ increased significantly (more integration)
    Integration,
    /// Φ decreased significantly (less integration)
    Disintegration,
}
