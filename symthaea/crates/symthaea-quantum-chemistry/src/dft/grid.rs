// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Numerical integration grid for DFT.
//!
//! Uses atom-centered grids with:
//! - Radial: Gauss-Chebyshev (Becke 1988) mapping [0,∞) → [-1,1]
//! - Angular: Lebedev quadrature (6, 14, 26, or 50 point grids)
//! - Partitioning: Becke's fuzzy cell scheme (1988)
//!
//! References:
//! - Becke, A. D. (1988). J. Chem. Phys. 88, 2547.
//! - Lebedev, V. I. (1977). Zh. Vychisl. Mat. Mat. Fiz. 16, 293.

use crate::molecule::Atom;
use std::f64::consts::PI;

/// A single grid point with position and weight.
#[derive(Debug, Clone, Copy)]
pub struct GridPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub weight: f64,
}

/// DFT numerical integration grid.
#[derive(Debug, Clone)]
pub struct DftGrid {
    pub points: Vec<GridPoint>,
}

/// Grid quality presets.
#[derive(Debug, Clone, Copy)]
pub enum GridQuality {
    /// Coarse: 20 radial × 6 angular = 120 points/atom
    Coarse,
    /// Medium: 35 radial × 26 angular = 910 points/atom
    Medium,
    /// Fine: 50 radial × 50 angular = 2500 points/atom
    Fine,
}

impl DftGrid {
    /// Build a molecular integration grid.
    pub fn build(atoms: &[Atom], quality: GridQuality) -> Self {
        let (n_radial, angular_points) = match quality {
            GridQuality::Coarse => (20, lebedev_6()),
            GridQuality::Medium => (35, lebedev_26()),
            GridQuality::Fine => (50, lebedev_50()),
        };

        let mut points = Vec::new();

        for (atom_idx, atom) in atoms.iter().enumerate() {
            // Bragg-Slater radius for this atom (used for radial scaling)
            let r_bs = bragg_slater_radius(atom.atomic_number);

            // Radial grid: Gauss-Chebyshev mapping
            for i in 1..=n_radial {
                let xi = (i as f64 * PI / (n_radial as f64 + 1.0)).cos();
                let r = r_bs * (1.0 + xi) / (1.0 - xi); // Becke mapping
                let dr = 2.0 * r_bs / ((1.0 - xi) * (1.0 - xi)); // dr/dxi

                let radial_weight =
                    PI / (n_radial as f64 + 1.0) * (1.0 - xi * xi).sqrt() * r * r * dr;

                // Angular grid
                for &(theta_x, theta_y, theta_z, ang_weight) in &angular_points {
                    let x = atom.position[0] + r * theta_x;
                    let y = atom.position[1] + r * theta_y;
                    let z = atom.position[2] + r * theta_z;

                    // Becke partitioning weight
                    let becke_w = becke_partition_weight(atom_idx, x, y, z, atoms);

                    let total_weight = radial_weight * ang_weight * becke_w;

                    if total_weight.abs() > 1e-20 {
                        points.push(GridPoint {
                            x,
                            y,
                            z,
                            weight: total_weight,
                        });
                    }
                }
            }
        }

        Self { points }
    }

    /// Number of grid points.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }
}

/// Becke partition weight for atom `atom_idx` at point (x, y, z).
///
/// Uses the Becke fuzzy-cell scheme with Stratmann step function.
fn becke_partition_weight(atom_idx: usize, x: f64, y: f64, z: f64, atoms: &[Atom]) -> f64 {
    let n_atoms = atoms.len();
    if n_atoms == 1 {
        return 1.0;
    }

    // Compute Becke cell function P_A(r) = Π_{B≠A} s(μ_AB)
    let mut p = vec![1.0; n_atoms];

    for a in 0..n_atoms {
        for b in 0..n_atoms {
            if a == b {
                continue;
            }

            let r_a = dist_to_atom(x, y, z, &atoms[a]);
            let r_b = dist_to_atom(x, y, z, &atoms[b]);
            let r_ab = atoms[a].distance_to(&atoms[b]);

            if r_ab < 1e-10 {
                continue;
            }

            let mu = (r_a - r_b) / r_ab;
            // Becke's step function (3 iterations)
            let s = step_function(mu);
            p[a] *= s;
        }
    }

    let total: f64 = p.iter().sum();
    if total < 1e-15 {
        return 0.0;
    }
    p[atom_idx] / total
}

/// Becke's smoothed step function: s(μ) = ½(1 - f(f(f(μ))))
/// where f(x) = ½x(3 - x²)
fn step_function(mu: f64) -> f64 {
    let f = |x: f64| 0.5 * x * (3.0 - x * x);
    let p = f(f(f(mu)));
    0.5 * (1.0 - p)
}

fn dist_to_atom(x: f64, y: f64, z: f64, atom: &Atom) -> f64 {
    let dx = x - atom.position[0];
    let dy = y - atom.position[1];
    let dz = z - atom.position[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Bragg-Slater radii in Bohr for grid scaling.
fn bragg_slater_radius(z: u8) -> f64 {
    match z {
        1 => 0.661,      // H: 0.35 Å
        2 => 0.567,      // He: 0.30 Å
        3..=4 => 2.268,  // Li-Be: ~1.2 Å
        5..=10 => 1.323, // B-Ne: ~0.7 Å
        _ => 1.5,        // Default
    }
}

/// 6-point Lebedev grid (octahedral symmetry).
/// Returns (x, y, z, weight) for points on the unit sphere.
fn lebedev_6() -> Vec<(f64, f64, f64, f64)> {
    let w = 1.0 / 6.0; // 4π/6 normalized to sum=4π... actually sum of weights = 4π
                       // For integration: Σ w_i f(r_i) ≈ ∫ f dΩ / (4π), so w_i should sum to 1
                       // We normalize so Σ w = 1 (weights are for ∫ f(Ω) dΩ/(4π))
    vec![
        (1.0, 0.0, 0.0, w),
        (-1.0, 0.0, 0.0, w),
        (0.0, 1.0, 0.0, w),
        (0.0, -1.0, 0.0, w),
        (0.0, 0.0, 1.0, w),
        (0.0, 0.0, -1.0, w),
    ]
}

/// 26-point Lebedev grid.
fn lebedev_26() -> Vec<(f64, f64, f64, f64)> {
    let mut pts = Vec::with_capacity(26);

    // 6 points along axes (weight a1)
    let a1 = 1.0 / 21.0;
    for &(x, y, z) in &[
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ] {
        pts.push((x, y, z, a1));
    }

    // 12 edge midpoints (weight a2)
    let a2 = 4.0 / 105.0;
    let s = 1.0 / 2.0_f64.sqrt();
    for &(x, y, z) in &[
        (s, s, 0.0),
        (s, -s, 0.0),
        (-s, s, 0.0),
        (-s, -s, 0.0),
        (s, 0.0, s),
        (s, 0.0, -s),
        (-s, 0.0, s),
        (-s, 0.0, -s),
        (0.0, s, s),
        (0.0, s, -s),
        (0.0, -s, s),
        (0.0, -s, -s),
    ] {
        pts.push((x, y, z, a2));
    }

    // 8 body diagonals (weight a3)
    let a3 = 27.0 / 840.0;
    let t = 1.0 / 3.0_f64.sqrt();
    for &(x, y, z) in &[
        (t, t, t),
        (t, t, -t),
        (t, -t, t),
        (t, -t, -t),
        (-t, t, t),
        (-t, t, -t),
        (-t, -t, t),
        (-t, -t, -t),
    ] {
        pts.push((x, y, z, a3));
    }

    pts
}

/// 50-point Lebedev grid (approximate — uses icosahedral + refinement).
fn lebedev_50() -> Vec<(f64, f64, f64, f64)> {
    // Use the 26-point grid doubled with midpoint refinements
    // For a true Lebedev-50, one needs the published tabulated points.
    // This approximation uses 26 + 24 additional points.
    let mut pts = lebedev_26();

    // Add 24 more points at intermediate positions
    let w = 1.0 / 50.0; // Uniform weight for extra points
    let a = 0.5257_311_121; // Golden ratio related
    let b = 0.8506_508_084;

    for &(x, y, z) in &[
        (0.0, a, b),
        (0.0, a, -b),
        (0.0, -a, b),
        (0.0, -a, -b),
        (a, b, 0.0),
        (a, -b, 0.0),
        (-a, b, 0.0),
        (-a, -b, 0.0),
        (b, 0.0, a),
        (b, 0.0, -a),
        (-b, 0.0, a),
        (-b, 0.0, -a),
    ] {
        pts.push((x, y, z, w));
        pts.push((-x, -y, -z, w)); // Inversion partner
    }

    // Renormalize weights to sum to 1
    let total_w: f64 = pts.iter().map(|p| p.3).sum();
    for p in &mut pts {
        p.3 /= total_w;
    }

    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Molecule;

    #[test]
    fn test_grid_creation() {
        let mol = Molecule::h2();
        let grid = DftGrid::build(&mol.atoms, GridQuality::Coarse);
        assert!(grid.n_points() > 0, "Grid should have points");
    }

    #[test]
    fn test_grid_weights_positive() {
        let mol = Molecule::water();
        let grid = DftGrid::build(&mol.atoms, GridQuality::Coarse);
        for pt in &grid.points {
            assert!(pt.weight >= 0.0, "Grid weights should be non-negative");
        }
    }

    #[test]
    fn test_becke_partition_single_atom() {
        let atoms = vec![Atom::new(1, 0.0, 0.0, 0.0)];
        let w = becke_partition_weight(0, 1.0, 0.0, 0.0, &atoms);
        assert!((w - 1.0).abs() < 1e-10, "Single atom partition = 1.0");
    }

    #[test]
    fn test_becke_partition_midpoint() {
        // At the midpoint of H2, both atoms should have weight ~0.5
        let atoms = vec![Atom::new(1, 0.0, 0.0, 0.0), Atom::new(1, 0.0, 0.0, 2.0)];
        let w0 = becke_partition_weight(0, 0.0, 0.0, 1.0, &atoms);
        let w1 = becke_partition_weight(1, 0.0, 0.0, 1.0, &atoms);
        assert!(
            (w0 - 0.5).abs() < 0.1 && (w1 - 0.5).abs() < 0.1,
            "Midpoint: w0={:.3}, w1={:.3}",
            w0,
            w1
        );
        assert!((w0 + w1 - 1.0).abs() < 1e-10, "Partition sums to 1");
    }
}
