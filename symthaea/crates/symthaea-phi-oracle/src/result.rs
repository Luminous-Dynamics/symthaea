// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::fmt;

/// Full report from an integration measurement.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntegrationReport {
    /// Integration index — the irreducible information of the system above its
    /// minimum information partition. Higher values indicate stronger structural
    /// coherence that cannot be decomposed into independent subsystems.
    pub integration_index: f64,

    /// Total mutual information across all subsystem pairs.
    pub total_mutual_information: f64,

    /// The minimum information partition: (part_a_indices, part_b_indices).
    /// This is the cut that *least* reduces mutual information — the system's
    /// weakest structural link.
    pub minimum_information_partition: (Vec<usize>, Vec<usize>),

    /// Fiedler ordering of subsystem indices — spectral decomposition of the
    /// mutual information graph. Adjacent indices in this ordering are most
    /// tightly coupled.
    pub spectral_order: Vec<usize>,

    /// Temporal coherence profile, if CfC temporal probing was used.
    pub temporal_coherence: Option<TemporalCoherence>,

    /// Normalized integration index: `integration_index / total_mutual_information`,
    /// clamped to [0, 1]. Enables cross-system comparison regardless of variable count.
    /// 0.0 when total MI is negligible.
    pub normalized_index: f64,

    /// Per-variable contribution via leave-one-out decomposition.
    /// `variable_contributions[i]` = total_mi - MI(system without variable i).
    /// Higher values indicate variables that contribute more to the system's
    /// overall integration.
    pub variable_contributions: Vec<f64>,

    /// Topological Betti numbers [beta_0, beta_1, beta_2].
    /// beta_0: number of connected components.
    /// beta_1: number of one-dimensional holes (cycles).
    pub betti_numbers: [usize; 3],

    /// Deeply persistent topological cycles discovered via filtration.
    pub persistent_cycles: Vec<PersistentCycle>,

    /// Number of observations used to compute this report.
    pub num_observations: usize,
}

/// A topological cycle discovered across a correlation filtration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PersistentCycle {
    /// Threshold at which this cycle was born.
    pub birth: f64,
    /// Threshold at which this cycle died (collapsed).
    pub death: f64,
    /// Normalized lifespan (birth - death). Higher = stronger signal.
    pub lifespan: f64,
    /// Indices of nodes involved in this cycle (heuristic approximation).
    pub participants: Vec<usize>,
}

impl fmt::Display for IntegrationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- Integration Report ---")?;
        writeln!(f, "Integration index:        {:.6}", self.integration_index)?;
        writeln!(f, "Normalized index:         {:.6}", self.normalized_index)?;
        writeln!(
            f,
            "Total mutual information:  {:.6}",
            self.total_mutual_information
        )?;
        writeln!(f, "Observations used:         {}", self.num_observations)?;
        writeln!(f)?;

        let (part_a, part_b) = &self.minimum_information_partition;
        writeln!(f, "Minimum information partition:")?;
        writeln!(f, "  Part A: {:?}", part_a)?;
        writeln!(f, "  Part B: {:?}", part_b)?;
        writeln!(f)?;

        writeln!(f, "Spectral (Fiedler) ordering: {:?}", self.spectral_order)?;

        if !self.variable_contributions.is_empty() {
            writeln!(f)?;
            writeln!(f, "Variable contributions (leave-one-out):")?;
            let mut indexed: Vec<(usize, f64)> = self
                .variable_contributions
                .iter()
                .copied()
                .enumerate()
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(i, c) in indexed.iter().take(5) {
                writeln!(f, "  var {i}: {c:+.6}")?;
            }
        }

        if let Some(tc) = &self.temporal_coherence {
            writeln!(f)?;
            write!(f, "{tc}")?;
        }

        Ok(())
    }
}

/// Temporal coherence profile from CfC multi-timescale probing.
///
/// The CfC network acts as a temporal probe: the same encoded observations are
/// evolved with different `dt` values, and the coefficient of variation (CV) of
/// the output across those timescales reveals the system's temporal structure.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TemporalCoherence {
    /// (tau, coefficient_of_variation) pairs. Low CV at a given tau means the
    /// system is coherent at that timescale.
    pub cv_by_tau: Vec<(f64, f64)>,

    /// The timescale (tau) with the lowest CV — the system's dominant timescale
    /// of integration.
    pub dominant_timescale: f64,

    /// Range of timescales over which CV remains below a coherence threshold.
    /// Wider temporal depth means the system integrates across more timescales.
    pub temporal_depth: f64,
}

/// Multi-scale hierarchical integration report.
///
/// Measures integration at progressively coarser scales by subsampling the
/// covariance matrix (every 2nd variable, every 4th, etc.).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HierarchicalReport {
    /// Finest-scale result (same as [`IntegrationOracle::measure()`](crate::IntegrationOracle::measure)).
    pub finest: IntegrationReport,
    /// Number of variables at each scale.
    pub scales: Vec<usize>,
    /// Integration index (phi) at each scale, corresponding to [`scales`].
    pub phi_by_scale: Vec<f64>,
}

impl fmt::Display for HierarchicalReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- Hierarchical Integration ---")?;
        for (i, (&scale, &phi)) in self.scales.iter().zip(&self.phi_by_scale).enumerate() {
            let marker = if i == 0 { " (finest)" } else { "" };
            writeln!(f, "  {scale:>3} vars: phi={phi:.6}{marker}")?;
        }
        writeln!(f)?;
        write!(f, "{}", self.finest)?;
        Ok(())
    }
}

impl fmt::Display for TemporalCoherence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- Temporal Coherence ---")?;
        for &(tau, cv) in &self.cv_by_tau {
            let bar_len = (cv * 40.0).min(40.0) as usize;
            let bar: String = "#".repeat(bar_len);
            writeln!(f, "  tau={tau:>8.4}  CV={cv:.4}  {bar}")?;
        }
        writeln!(f, "  Dominant timescale: {:.4}s", self.dominant_timescale)?;
        writeln!(
            f,
            "  Temporal depth:     {:.4} (log-ratio)",
            self.temporal_depth
        )?;
        Ok(())
    }
}
