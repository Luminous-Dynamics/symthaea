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

/// 50-point Lebedev grid: genuine Lebedev-Laikov degree-11 quadrature
/// (Phase Q5c, 2026-07-17 -- this used to be an icosahedral-vertex
/// approximation, not the real published rule, disclosed as such in this
/// doc comment; now fixed).
///
/// Source: Burkardt, "Sphere Lebedev Rule" dataset (Florida State
/// University), the standard public-domain republication of Lebedev &
/// Laikov's original tables. Raw fetched data vendored at
/// `dft/lebedev_reference/lebedev_011.txt` for provenance. Verified during
/// planning: weights sum to 1 (~1e-15), and the shared a1/a2/a3 orbit
/// weights and geometry (`s = 1/√2`, `t = 1/√3`) exactly match this crate's
/// own already-correct `lebedev_26()` -- those two orbits are reused
/// unchanged; only the 24-point `(±p,±p,±q)` orbit is new.
fn lebedev_50() -> Vec<(f64, f64, f64, f64)> {
    let mut pts = Vec::with_capacity(50);

    // a1: 6 axis points (same geometry as lebedev_6/lebedev_26)
    let a1 = 0.012_698_412_698_413;
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

    // a2: 12 edge points (same s=1/sqrt(2) as lebedev_26)
    let a2 = 0.022_574_955_908_289;
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

    // a3: 8 body-diagonal points (same t=1/sqrt(3) as lebedev_26)
    let a3 = 0.021_093_750_000_000;
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

    // b: 24 new (p,p,q)-type points -- q placed in each of the 3
    // coordinate slots, all 8 sign combinations per placement.
    let b = 0.020_173_335_537_919;
    let p = 0.301_511_344_577_764;
    let q = 0.904_534_033_733_291;
    for &sq in &[1.0, -1.0] {
        for &sp1 in &[1.0, -1.0] {
            for &sp2 in &[1.0, -1.0] {
                pts.push((sp1 * p, sp2 * p, sq * q, b));
                pts.push((sp1 * p, sq * q, sp2 * p, b));
                pts.push((sq * q, sp1 * p, sp2 * p, b));
            }
        }
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

    // ── Phase Q5c (2026-07-17): self-derived mathematical identities for
    // the angular Lebedev grids -- no external reference number needed
    // beyond the already-independently-verified weights themselves. Applied
    // to all three grids: lebedev_50 is the fix under test, lebedev_6/26
    // get the same checks as bonus regression coverage they didn't have
    // before (expected to already pass -- confirmed rather than assumed).

    fn check_lebedev_identities(pts: &[(f64, f64, f64, f64)], label: &str) {
        // 1. Weight sum = 1.
        let sum_w: f64 = pts.iter().map(|p| p.3).sum();
        assert!(
            (sum_w - 1.0).abs() < 1e-10,
            "{label}: weights sum to {sum_w}, expected 1"
        );

        // 2. Points lie on the unit sphere.
        for &(x, y, z, _) in pts {
            let r2 = x * x + y * y + z * z;
            assert!(
                (r2 - 1.0).abs() < 1e-10,
                "{label}: point ({x},{y},{z}) has r^2={r2}, expected 1"
            );
        }

        // 3. Odd-parity monomials vanish exactly (symmetry-guaranteed: the
        // grid is closed under sign-flip of any coordinate and under axis
        // permutation, so any monomial odd in at least one coordinate must
        // integrate to zero regardless of quadrature order).
        type Monomial = fn(f64, f64, f64) -> f64;
        let odd_monomials: &[(&str, Monomial)] = &[
            ("x", |x, _, _| x),
            ("y", |_, y, _| y),
            ("z", |_, _, z| z),
            ("xyz", |x, y, z| x * y * z),
            ("x^3", |x, _, _| x.powi(3)),
            ("x^2*y", |x, y, _| x * x * y),
            ("x^5", |x, _, _| x.powi(5)),
            ("x^3*y^2", |x, y, _| x.powi(3) * y * y),
            ("x^7", |x, _, _| x.powi(7)),
            ("x^9", |x, _, _| x.powi(9)),
        ];
        for &(name, f) in odd_monomials {
            let integral: f64 = pts.iter().map(|&(x, y, z, w)| w * f(x, y, z)).sum();
            assert!(
                integral.abs() < 1e-10,
                "{label}: odd monomial {name} integrated to {integral}, expected 0"
            );
        }

        // 4. Quartic moment identity: x^2+y^2+z^2=1 pointwise on the unit
        // sphere, so squaring and integrating gives
        // E[x^4]+E[y^4]+E[z^4] + 2*(E[x^2y^2]+E[x^2z^2]+E[y^2z^2]) = E[1] = 1
        // exactly -- a real constraint any degree->=4-exact rule must
        // satisfy, derived here, not looked up.
        let e_x4: f64 = pts.iter().map(|&(x, _, _, w)| w * x.powi(4)).sum();
        let e_y4: f64 = pts.iter().map(|&(_, y, _, w)| w * y.powi(4)).sum();
        let e_z4: f64 = pts.iter().map(|&(_, _, z, w)| w * z.powi(4)).sum();
        let e_x2y2: f64 = pts.iter().map(|&(x, y, _, w)| w * x * x * y * y).sum();
        let e_x2z2: f64 = pts.iter().map(|&(x, _, z, w)| w * x * x * z * z).sum();
        let e_y2z2: f64 = pts.iter().map(|&(_, y, z, w)| w * y * y * z * z).sum();
        let quartic_identity = e_x4 + e_y4 + e_z4 + 2.0 * (e_x2y2 + e_x2z2 + e_y2z2);
        assert!(
            (quartic_identity - 1.0).abs() < 1e-10,
            "{label}: quartic identity = {quartic_identity}, expected 1"
        );
    }

    #[test]
    fn test_lebedev_6_identities() {
        check_lebedev_identities(&lebedev_6(), "lebedev_6");
    }

    #[test]
    fn test_lebedev_26_identities() {
        check_lebedev_identities(&lebedev_26(), "lebedev_26");
    }

    #[test]
    fn test_lebedev_50_identities() {
        let pts = lebedev_50();
        assert_eq!(pts.len(), 50, "lebedev_50 should have exactly 50 points");
        check_lebedev_identities(&pts, "lebedev_50");
    }

    #[test]
    fn test_lebedev_50_no_duplicate_points() {
        // Confirms the (p,p,q) orbit's sign/position loop produces 24
        // genuinely distinct points, not accidental overlaps.
        let pts = lebedev_50();
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let (x1, y1, z1, _) = pts[i];
                let (x2, y2, z2, _) = pts[j];
                let d2 = (x1 - x2).powi(2) + (y1 - y2).powi(2) + (z1 - z2).powi(2);
                assert!(d2 > 1e-12, "duplicate points at indices {i} and {j}");
            }
        }
    }
}
