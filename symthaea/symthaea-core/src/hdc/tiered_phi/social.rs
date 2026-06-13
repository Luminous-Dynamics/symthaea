// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Social Phi — Integrated Information of Agent Collectives
//!
//! Measures the level of cognitive integration across multiple Symthaea agents.
//! Uses the Locality Ratio from PhiPyramid to determine if a group acts
//! as a unified 'Social Spacetime Crystal' or just a collection of individuals.

use super::analysis::{PhiPyramid, PhiPyramidConfig};
use crate::hdc::binary_hv::BinaryHV;

/// Result of a Social Phi analysis
#[derive(Debug, Clone)]
pub struct SocialPhiResult {
    /// Integrated information of the entire collective
    pub collective_phi: f64,
    /// Average integrated information of individual agents
    pub individual_avg_phi: f64,
    /// Integration Ratio: Collective / Individual ( > 1.0 means emergent collective intelligence )
    pub integration_ratio: f64,
    /// Whether the collective has 'crystallized' into a unified social agent
    pub crystallized: bool,
}

/// Calculator for Social Phi metrics
pub struct SocialPhiCalculator {
    pyramid: PhiPyramid,
}

impl SocialPhiCalculator {
    pub fn new() -> Self {
        Self {
            pyramid: PhiPyramid::with_config(PhiPyramidConfig {
                // Each agent is a cluster; we want to compare individual vs global
                min_components_per_scale: 4,
                max_scales: 4,
                ..Default::default()
            }),
        }
    }

    /// Compute Social Phi for a group of agents, each represented by their component set.
    pub fn compute_social_phi(&mut self, agents_components: &[Vec<BinaryHV>]) -> SocialPhiResult {
        // 1. Flatten all components into a single global set
        let all_components: Vec<BinaryHV> = agents_components.iter().flatten().cloned().collect();

        // 2. Compute Phi Pyramid on the flattened set
        let pyramid_res = self.pyramid.compute(&all_components);

        // 3. Extract Collective vs Individual metrics
        // Global scale (last) is the collective
        let collective_phi = pyramid_res.phi_by_scale.last().copied().unwrap_or(0.0);

        // Meso scale (middle-ish) often corresponds to individual agent boundaries in component space
        let individual_avg_phi = if pyramid_res.phi_by_scale.len() > 1 {
            pyramid_res.phi_by_scale[pyramid_res.phi_by_scale.len() / 2]
        } else {
            collective_phi
        };

        let integration_ratio = if individual_avg_phi > 1e-10 {
            collective_phi / individual_avg_phi
        } else {
            1.0
        };

        SocialPhiResult {
            collective_phi,
            individual_avg_phi,
            integration_ratio,
            // Crystallization occurs when collective integration significantly exceeds individual parts
            crystallized: integration_ratio > 1.5,
        }
    }
}

impl Default for SocialPhiCalculator {
    fn default() -> Self {
        Self::new()
    }
}
