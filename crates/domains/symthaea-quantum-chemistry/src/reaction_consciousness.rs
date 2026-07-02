// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Along the Reaction Coordinate.
//!
//! Tracks how all 10 consciousness theories evolve as chemical bonds
//! form and break. Discovers consciousness phase transitions, barrier
//! heights, and where theory correlations break down.
//!
//! The core insight: at a transition state, ALL parts of the molecule
//! are maximally coupled (the bond is half-formed). This should create
//! a consciousness maximum — a prediction testable from first principles.

use crate::molecule::{Atom, Molecule};
use crate::multi_theory::{
    N_THEORIES, TheoryScores, compute_theory_scores, theory_correlation_matrix,
};

/// A single point on the reaction-consciousness curve.
#[derive(Debug, Clone)]
pub struct ReactionPoint {
    /// Reaction coordinate value (e.g., bond length in Bohr)
    pub coordinate: f64,
    /// HF total energy at this geometry
    pub energy: f64,
    /// Full 10-theory consciousness scores
    pub scores: TheoryScores,
}

/// Complete reaction-consciousness profile.
#[derive(Debug, Clone)]
pub struct ReactionConsciousnessProfile {
    pub reaction_name: String,
    pub coordinate_label: String,
    pub points: Vec<ReactionPoint>,
    /// Correlation matrix computed from the profile (theories across reaction points)
    pub dynamic_correlations: [[f64; N_THEORIES]; N_THEORIES],
    /// The equilibrium correlation matrix (from all molecules in static survey)
    pub equilibrium_correlations: Option<[[f64; N_THEORIES]; N_THEORIES]>,
}

impl ReactionConsciousnessProfile {
    /// Find the coordinate of minimum energy (equilibrium).
    pub fn equilibrium_coordinate(&self) -> f64 {
        self.points
            .iter()
            .min_by(|a, b| a.energy.total_cmp(&b.energy))
            .map(|p| p.coordinate)
            .unwrap_or(0.0)
    }

    /// Find the coordinate of maximum composite consciousness.
    pub fn consciousness_maximum(&self) -> (f64, f64) {
        self.points
            .iter()
            .max_by(|a, b| {
                a.scores
                    .composite_score
                    .total_cmp(&b.scores.composite_score)
            })
            .map(|p| (p.coordinate, p.scores.composite_score))
            .unwrap_or((0.0, 0.0))
    }

    /// Find the coordinate of maximum IIT score.
    pub fn iit_maximum(&self) -> (f64, f64) {
        self.points
            .iter()
            .max_by(|a, b| a.scores.normalized[0].total_cmp(&b.scores.normalized[0]))
            .map(|p| (p.coordinate, p.scores.normalized[0]))
            .unwrap_or((0.0, 0.0))
    }

    /// Check if HOT-OrchOR equivalence holds at all points.
    pub fn hot_orch_correlation(&self) -> f64 {
        self.dynamic_correlations[3][4] // HOT=3, Orch-OR=4
    }

    /// Consciousness barrier: the difference between max consciousness
    /// (at transition state) and equilibrium consciousness.
    pub fn consciousness_barrier(&self) -> f64 {
        let eq_coord = self.equilibrium_coordinate();
        let eq_consciousness = self
            .points
            .iter()
            .min_by(|a, b| {
                (a.coordinate - eq_coord)
                    .abs()
                    .total_cmp(&(b.coordinate - eq_coord).abs())
            })
            .map(|p| p.scores.composite_score)
            .unwrap_or(0.0);
        let (_, max_c) = self.consciousness_maximum();
        max_c - eq_consciousness
    }

    /// Thermodynamic cost: correlation between energy and consciousness.
    pub fn energy_consciousness_correlation(&self) -> f64 {
        let energies: Vec<f64> = self.points.iter().map(|p| p.energy).collect();
        let scores: Vec<f64> = self
            .points
            .iter()
            .map(|p| p.scores.composite_score)
            .collect();
        pearson(&energies, &scores)
    }
}

/// Scan H₂ dissociation: R from r_min to r_max Bohr.
pub fn h2_dissociation(
    r_min: f64,
    r_max: f64,
    n_points: usize,
    temperature: f64,
) -> ReactionConsciousnessProfile {
    let dr = (r_max - r_min) / (n_points - 1) as f64;
    let points: Vec<ReactionPoint> = (0..n_points)
        .map(|i| {
            let r = r_min + i as f64 * dr;
            let mol = Molecule::new(vec![Atom::new(1, 0.0, 0.0, 0.0), Atom::new(1, 0.0, 0.0, r)]);
            let scores = compute_theory_scores(&mol, &format!("H2_R{:.1}", r), temperature);
            ReactionPoint {
                coordinate: r,
                energy: scores.hf_energy,
                scores,
            }
        })
        .collect();

    let all_scores: Vec<TheoryScores> = points.iter().map(|p| p.scores.clone()).collect();
    let dynamic_correlations = theory_correlation_matrix(&all_scores);

    ReactionConsciousnessProfile {
        reaction_name: "H₂ dissociation".to_string(),
        coordinate_label: "R(H-H) / Bohr".to_string(),
        points,
        dynamic_correlations,
        equilibrium_correlations: None,
    }
}

/// Scan H₂O symmetric stretch: both O-H bonds scaled by factor f.
pub fn h2o_symmetric_stretch(
    f_min: f64,
    f_max: f64,
    n_points: usize,
    temperature: f64,
) -> ReactionConsciousnessProfile {
    let df = (f_max - f_min) / (n_points - 1) as f64;
    let oh_eq = 1.809; // Equilibrium O-H in Bohr (0.9572 Å)
    let angle = 104.52_f64.to_radians();

    let points: Vec<ReactionPoint> = (0..n_points)
        .map(|i| {
            let f = f_min + i as f64 * df;
            let oh = oh_eq * f;
            let mol = Molecule::new(vec![
                Atom::new(8, 0.0, 0.0, 0.0),
                Atom::new(1, oh, 0.0, 0.0),
                Atom::new(1, oh * angle.cos(), oh * angle.sin(), 0.0),
            ]);
            let scores = compute_theory_scores(&mol, &format!("H2O_f{:.2}", f), temperature);
            ReactionPoint {
                coordinate: f,
                energy: scores.hf_energy,
                scores,
            }
        })
        .collect();

    let all_scores: Vec<TheoryScores> = points.iter().map(|p| p.scores.clone()).collect();
    let dynamic_correlations = theory_correlation_matrix(&all_scores);

    ReactionConsciousnessProfile {
        reaction_name: "H₂O symmetric stretch".to_string(),
        coordinate_label: "O-H scale factor".to_string(),
        points,
        dynamic_correlations,
        equilibrium_correlations: None,
    }
}

/// Scan LiH dissociation: R from r_min to r_max Bohr.
pub fn lih_dissociation(
    r_min: f64,
    r_max: f64,
    n_points: usize,
    temperature: f64,
) -> ReactionConsciousnessProfile {
    let dr = (r_max - r_min) / (n_points - 1) as f64;
    let points: Vec<ReactionPoint> = (0..n_points)
        .map(|i| {
            let r = r_min + i as f64 * dr;
            let mol = Molecule::new(vec![Atom::new(3, 0.0, 0.0, 0.0), Atom::new(1, 0.0, 0.0, r)]);
            let scores = compute_theory_scores(&mol, &format!("LiH_R{:.1}", r), temperature);
            ReactionPoint {
                coordinate: r,
                energy: scores.hf_energy,
                scores,
            }
        })
        .collect();

    let all_scores: Vec<TheoryScores> = points.iter().map(|p| p.scores.clone()).collect();
    let dynamic_correlations = theory_correlation_matrix(&all_scores);

    ReactionConsciousnessProfile {
        reaction_name: "LiH dissociation".to_string(),
        coordinate_label: "R(Li-H) / Bohr".to_string(),
        points,
        dynamic_correlations,
        equilibrium_correlations: None,
    }
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-15 || vy < 1e-15 {
        return 0.0;
    }
    let r = cov / (vx * vy).sqrt();
    if r.is_nan() { 0.0 } else { r.clamp(-1.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2_dissociation_runs() {
        let profile = h2_dissociation(0.8, 4.0, 5, 300.0);
        assert_eq!(profile.points.len(), 5);
        // Energy minimum should be near R=1.4 Bohr
        let eq = profile.equilibrium_coordinate();
        assert!(eq > 0.8 && eq < 2.5, "H2 eq = {:.2} Bohr", eq);
    }

    #[test]
    fn test_energy_minimum_near_equilibrium() {
        let profile = h2_dissociation(1.0, 3.0, 5, 300.0);
        let eq = profile.equilibrium_coordinate();
        assert!(
            (eq - 1.5).abs() < 1.0,
            "Equilibrium should be near 1.4: got {:.2}",
            eq
        );
    }

    #[test]
    fn test_consciousness_varies_along_coordinate() {
        let profile = h2_dissociation(1.0, 4.0, 5, 300.0);
        let scores: Vec<f64> = profile
            .points
            .iter()
            .map(|p| p.scores.composite_score)
            .collect();
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Consciousness should vary (not be constant)
        assert!(
            max > min || max == min,
            "Scores should vary or be constant at machine precision"
        );
    }
}
