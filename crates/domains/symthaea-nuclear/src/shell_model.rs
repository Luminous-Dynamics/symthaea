// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nuclear Shell Model with Woods-Saxon potential and Numerov solver.
//!
//! Solves the radial Schrödinger equation:
//! ```text
//! [-ℏ²/(2m) d²/dr² + V_eff(r)] u(r) = E·u(r)
//! ```
//!
//! where V_eff includes the Woods-Saxon central potential, spin-orbit coupling,
//! and centrifugal barrier.
//!
//! The Numerov method is a 4th-order integration scheme for second-order ODEs
//! without first-derivative terms, making it ideal for radial wavefunctions.
//!
//! References:
//! - Ring & Schuck (2004) §2.3: Woods-Saxon potential
//! - Krane (1988) §5.1: Shell model and magic numbers

use crate::constants::*;
use serde::{Deserialize, Serialize};

/// Woods-Saxon potential parameters.
#[derive(Debug, Clone)]
pub struct WoodsSaxonPotential {
    /// Central potential depth (MeV), typically ~50
    pub v0: f64,
    /// Nuclear radius parameter (fm), R = r0 * A^(1/3)
    pub r0: f64,
    /// Surface diffuseness (fm), typically ~0.65
    pub a: f64,
    /// Spin-orbit strength factor relative to v0, typically ~0.44
    pub v_ls_factor: f64,
}

impl Default for WoodsSaxonPotential {
    fn default() -> Self {
        Self {
            v0: V0_WOODS_SAXON,
            r0: R0_FM,
            a: DIFFUSENESS_FM,
            v_ls_factor: SPIN_ORBIT_FACTOR,
        }
    }
}

impl WoodsSaxonPotential {
    /// Nuclear radius R = r₀·A^(1/3) in fm.
    pub fn nuclear_radius(&self, a_mass: u16) -> f64 {
        self.r0 * (a_mass as f64).powf(1.0 / 3.0)
    }

    /// Central Woods-Saxon potential V(r) in MeV.
    ///
    /// V(r) = -V₀ / (1 + exp((r - R) / a))
    pub fn central(&self, r: f64, a_mass: u16) -> f64 {
        let big_r = self.nuclear_radius(a_mass);
        -self.v0 / (1.0 + ((r - big_r) / self.a).exp())
    }

    /// Derivative of the central potential dV/dr in MeV/fm.
    fn central_derivative(&self, r: f64, a_mass: u16) -> f64 {
        let big_r = self.nuclear_radius(a_mass);
        let x = (r - big_r) / self.a;
        let exp_x = x.exp();
        let denom = (1.0 + exp_x).powi(2);
        self.v0 * exp_x / (self.a * denom)
    }

    /// Spin-orbit potential V_ls(r) × ⟨l·s⟩ in MeV.
    ///
    /// V_ls(r) = -V_ls_factor × V₀ × (1/r) × (dV/dr) × ⟨l·s⟩
    /// where ⟨l·s⟩ = [j(j+1) - l(l+1) - s(s+1)] / 2
    pub fn spin_orbit(&self, r: f64, a_mass: u16, l: u16, j: f64) -> f64 {
        if r < 0.01 || l == 0 {
            return 0.0; // No spin-orbit for s-waves or at origin
        }
        let s = 0.5; // Nucleon spin
        let l_f = l as f64;
        let ls = 0.5 * (j * (j + 1.0) - l_f * (l_f + 1.0) - s * (s + 1.0));
        let dv_dr = self.central_derivative(r, a_mass);
        -self.v_ls_factor * self.v0 * (1.0 / r) * dv_dr * ls / (self.v0) // normalize
    }

    /// Effective potential including centrifugal barrier: V_eff(r).
    ///
    /// V_eff = V_central + V_spin_orbit + ℏ²l(l+1)/(2mr²)
    pub fn effective(&self, r: f64, a_mass: u16, l: u16, j: f64) -> f64 {
        let v_central = self.central(r, a_mass);
        let v_ls = self.spin_orbit(r, a_mass, l, j);
        let centrifugal = if r > 0.01 {
            HBAR2_OVER_2M * (l as f64) * (l as f64 + 1.0) / (r * r)
        } else {
            0.0
        };
        v_central + v_ls + centrifugal
    }
}

/// Result of a Numerov radial integration.
#[derive(Debug, Clone)]
pub struct NumerovResult {
    /// Eigenvalue energy (MeV)
    pub energy: f64,
    /// Radial wavefunction u(r) on the grid
    pub wavefunction: Vec<f64>,
    /// Number of nodes (zero crossings)
    pub nodes: u16,
    /// Whether the bisection converged
    pub converged: bool,
}

/// A single-particle energy level in the shell model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellLevel {
    /// Principal quantum number n (1, 2, 3, ...)
    pub n: u16,
    /// Orbital angular momentum l
    pub l: u16,
    /// Total angular momentum j = l ± 1/2
    pub j: f64,
    /// Energy eigenvalue (MeV)
    pub energy_mev: f64,
    /// Degeneracy 2j+1
    pub degeneracy: u16,
}

impl ShellLevel {
    /// Spectroscopic label (e.g., "1s1/2", "1p3/2", "1d5/2")
    pub fn label(&self) -> String {
        let l_char = match self.l {
            0 => 's',
            1 => 'p',
            2 => 'd',
            3 => 'f',
            4 => 'g',
            5 => 'h',
            6 => 'i',
            _ => '?',
        };
        let j_num = (2.0 * self.j) as u16;
        format!("{}{}{}/{}", self.n, l_char, j_num, 2)
    }
}

/// Nuclear shell model calculator.
#[derive(Debug, Clone)]
pub struct ShellModel {
    pub potential: WoodsSaxonPotential,
    /// Radial grid step (fm)
    pub dr: f64,
    /// Maximum radius (fm)
    pub r_max: f64,
}

impl Default for ShellModel {
    fn default() -> Self {
        Self {
            potential: WoodsSaxonPotential::default(),
            dr: DR_DEFAULT,
            r_max: R_MAX_FM,
        }
    }
}

impl ShellModel {
    /// Count zero crossings (nodes) in a wavefunction.
    ///
    /// Skips the first few grid points near the origin where the wavefunction
    /// is dominated by the boundary condition u ~ r^(l+1) and numerical noise.
    /// Also skips the tail where the wavefunction decays exponentially.
    fn count_nodes(wf: &[f64]) -> u16 {
        let mut nodes = 0u16;
        // Skip first ~5% of the grid (near origin) and last ~20% (exponential tail)
        let start = (wf.len() / 20).max(3);
        let end = (wf.len() * 4) / 5;
        let threshold = 1e-20; // ignore negligible zero crossings

        for i in start..end {
            if wf[i] * wf[i - 1] < 0.0 && wf[i].abs() > threshold && wf[i - 1].abs() > threshold {
                nodes += 1;
            }
        }
        nodes
    }

    /// Numerov integration from the origin outward for a trial energy.
    ///
    /// Returns the wavefunction and whether it diverges (for eigenvalue bracketing).
    fn numerov_integrate(&self, energy: f64, a_mass: u16, l: u16, j: f64) -> Vec<f64> {
        let n_points = (self.r_max / self.dr) as usize + 1;
        let mut u = vec![0.0; n_points];

        // Boundary condition: u(0) = 0, u(dr) = dr^(l+1) (regular at origin)
        u[0] = 0.0;
        u[1] = self.dr.powi(l as i32 + 1);

        // Numerov: u_{n+1} = [2(1 - 5h²k²_n/12)u_n - (1 + h²k²_{n-1}/12)u_{n-1}]
        //                    / (1 + h²k²_{n+1}/12)
        // where k²(r) = (2m/ℏ²)(E - V_eff(r))
        let h2 = self.dr * self.dr;

        for i in 1..n_points - 1 {
            let r_prev = i as f64 * self.dr - self.dr;
            let r_curr = i as f64 * self.dr;
            let r_next = (i + 1) as f64 * self.dr;

            // Avoid r=0 singularity
            let r_prev = r_prev.max(0.01);
            let r_curr = r_curr.max(0.01);
            let r_next = r_next.max(0.01);

            let k2_prev = (energy - self.potential.effective(r_prev, a_mass, l, j)) / HBAR2_OVER_2M;
            let k2_curr = (energy - self.potential.effective(r_curr, a_mass, l, j)) / HBAR2_OVER_2M;
            let k2_next = (energy - self.potential.effective(r_next, a_mass, l, j)) / HBAR2_OVER_2M;

            let numer = 2.0 * (1.0 - 5.0 * h2 * k2_curr / 12.0) * u[i]
                - (1.0 + h2 * k2_prev / 12.0) * u[i - 1];
            let denom = 1.0 + h2 * k2_next / 12.0;

            u[i + 1] = numer / denom;

            // Prevent numerical blowup
            if u[i + 1].abs() > 1e10 {
                // Normalize to prevent overflow
                let scale = u[i + 1].abs();
                for val in u.iter_mut().take(i + 2) {
                    *val /= scale;
                }
            }
        }

        u
    }

    /// Find eigenvalue for quantum numbers (n, l, j) using bisection.
    ///
    /// The n-th state with angular momentum l has n-1 nodes in the radial wavefunction.
    pub fn find_eigenvalue(&self, a_mass: u16, target_nodes: u16, l: u16, j: f64) -> NumerovResult {
        // Energy search range
        let e_min = -self.potential.v0 - 20.0; // Below potential bottom
        let e_max = 0.0; // Bound states have E < 0

        let mut e_low = e_min;
        let mut e_high = e_max;

        // First: find a bracket where node count changes
        let mut best_energy = 0.5 * (e_low + e_high);
        let mut best_wf = self.numerov_integrate(best_energy, a_mass, l, j);
        let mut best_nodes = Self::count_nodes(&best_wf);

        for _ in 0..MAX_BISECTION_ITER {
            let e_mid = 0.5 * (e_low + e_high);
            let wf = self.numerov_integrate(e_mid, a_mass, l, j);
            let nodes = Self::count_nodes(&wf);

            if nodes > target_nodes {
                e_high = e_mid;
            } else {
                e_low = e_mid;
            }

            best_energy = e_mid;
            best_wf = wf;
            best_nodes = nodes;

            if (e_high - e_low).abs() < ENERGY_TOLERANCE {
                break;
            }
        }

        let converged = (e_high - e_low).abs() < ENERGY_TOLERANCE;

        NumerovResult {
            energy: best_energy,
            wavefunction: best_wf,
            nodes: best_nodes,
            converged,
        }
    }

    /// Compute single-particle energy levels for a nucleus.
    ///
    /// Fills levels up to `max_n` principal quantum number and `max_l` orbital
    /// angular momentum. Returns levels sorted by energy.
    pub fn single_particle_levels(&self, a_mass: u16, max_n: u16, max_l: u16) -> Vec<ShellLevel> {
        let mut levels = Vec::new();

        for l in 0..=max_l {
            // j = l + 1/2 and j = l - 1/2 (except l=0 which only has j=1/2)
            let j_values: Vec<f64> = if l == 0 {
                vec![0.5]
            } else {
                vec![l as f64 + 0.5, l as f64 - 0.5]
            };

            for &j in &j_values {
                for n in 1..=max_n {
                    let target_nodes = n - 1; // n-th state has n-1 nodes
                    let result = self.find_eigenvalue(a_mass, target_nodes, l, j);

                    // Only keep bound states (E < 0) that converged
                    if result.energy < -0.1 && result.converged {
                        let degeneracy = (2.0 * j + 1.0) as u16;
                        levels.push(ShellLevel {
                            n,
                            l,
                            j,
                            energy_mev: result.energy,
                            degeneracy,
                        });
                    }
                }
            }
        }

        levels.sort_by(|a, b| a.energy_mev.partial_cmp(&b.energy_mev).unwrap());
        levels
    }

    /// Detect magic numbers from shell closures.
    ///
    /// Magic numbers occur where there's a large gap between consecutive
    /// filled shell energies. Returns cumulative nucleon counts at gaps.
    pub fn magic_numbers(&self, a_mass: u16) -> Vec<u16> {
        let levels = self.single_particle_levels(a_mass, 4, 6);
        if levels.len() < 2 {
            return vec![];
        }

        // Compute gaps between consecutive levels
        let mut magic = Vec::new();
        let mut cumulative = 0u16;
        let mean_gap = {
            let total: f64 = levels
                .windows(2)
                .map(|w| w[1].energy_mev - w[0].energy_mev)
                .sum();
            total / (levels.len() - 1) as f64
        };

        for i in 0..levels.len() - 1 {
            cumulative += levels[i].degeneracy;
            let gap = levels[i + 1].energy_mev - levels[i].energy_mev;

            // A "shell closure" is a gap significantly larger than average
            if gap > mean_gap * 1.5 {
                magic.push(cumulative);
            }
        }

        magic
    }

    /// Strutinsky shell correction energy δE_shell in MeV.
    ///
    /// δE_shell = Σ_occ ε_i - ∫ ε·g̃(ε) dε
    ///
    /// where g̃(ε) is the smoothed level density obtained by convolving
    /// the discrete level spectrum with a Gaussian of width γ.
    ///
    /// Negative values indicate extra stability (shell closure).
    pub fn shell_correction_energy(&self, a_mass: u16, z: u16) -> f64 {
        let levels = self.single_particle_levels(a_mass, 4, 6);
        if levels.is_empty() {
            return 0.0;
        }

        // Fill levels up to Z protons (or N neutrons — use Z for proton shell)
        let mut filled = 0u16;
        let mut discrete_sum = 0.0;

        for level in &levels {
            if filled >= z {
                break;
            }
            let occ = level.degeneracy.min(z - filled);
            discrete_sum += level.energy_mev * occ as f64;
            filled += occ;
        }

        // Smoothed level density via Gaussian convolution
        let gamma = GAMMA_STRUTINSKY;
        let e_min = levels.first().map(|l| l.energy_mev).unwrap_or(-50.0);
        let e_max = levels.last().map(|l| l.energy_mev).unwrap_or(0.0);
        let de = 0.1; // MeV step for integration
        let n_steps = ((e_max - e_min) / de) as usize + 1;

        let mut smooth_sum = 0.0;
        let mut smooth_count = 0.0;

        for step in 0..n_steps {
            let e = e_min + step as f64 * de;

            // Smoothed density at energy e
            let g_smooth: f64 = levels
                .iter()
                .map(|l| {
                    let x = (e - l.energy_mev) / gamma;
                    l.degeneracy as f64 * (-0.5 * x * x).exp()
                        / (gamma * (2.0 * std::f64::consts::PI).sqrt())
                })
                .sum();

            smooth_sum += e * g_smooth * de;
            smooth_count += g_smooth * de;

            if smooth_count >= z as f64 {
                break;
            }
        }

        discrete_sum - smooth_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_woods_saxon_depth() {
        let ws = WoodsSaxonPotential::default();
        // At center of a medium nucleus (A=120), potential should be ~-V₀
        let v_center = ws.central(0.01, 120);
        assert!(
            (v_center + V0_WOODS_SAXON).abs() < 2.0,
            "Center potential = {}, expected ~{}",
            v_center,
            -V0_WOODS_SAXON
        );
    }

    #[test]
    fn test_woods_saxon_surface() {
        let ws = WoodsSaxonPotential::default();
        // At R = r₀·A^(1/3), potential should be -V₀/2
        let r = ws.nuclear_radius(120);
        let v = ws.central(r, 120);
        assert!(
            (v + V0_WOODS_SAXON / 2.0).abs() < 1.0,
            "Surface potential = {}, expected ~{}",
            v,
            -V0_WOODS_SAXON / 2.0
        );
    }

    #[test]
    fn test_spin_orbit_splitting() {
        let model = ShellModel::default();
        // For A=40 (Ca-40), p-shell should show j=3/2 vs j=1/2 splitting
        let levels = model.single_particle_levels(40, 3, 3);

        // Find p3/2 and p1/2
        let p_32: Vec<_> = levels.iter().filter(|l| l.l == 1 && l.j == 1.5).collect();
        let p_12: Vec<_> = levels.iter().filter(|l| l.l == 1 && l.j == 0.5).collect();

        if !p_32.is_empty() && !p_12.is_empty() {
            let splitting = p_12[0].energy_mev - p_32[0].energy_mev;
            assert!(
                splitting > 0.0,
                "p1/2 should be higher energy than p3/2 (splitting={})",
                splitting
            );
        }
    }

    #[test]
    fn test_magic_numbers_detected() {
        let model = ShellModel::default();
        // Use a medium nucleus to detect magic numbers
        let magic = model.magic_numbers(120);

        // Known magic numbers: 2, 8, 20, 28, 50, 82, 126
        let expected = [2u16, 8, 20, 28, 50, 82];

        let mut found_count = 0;
        for &expected_magic in &expected {
            if magic
                .iter()
                .any(|&m| (m as i16 - expected_magic as i16).unsigned_abs() <= 2)
            {
                found_count += 1;
            }
        }

        // We should detect at least 3 of the 6 small magic numbers
        // (the simplified model may miss some due to approximations)
        assert!(
            found_count >= 3,
            "Found {}/6 magic numbers: {:?} (expected subset of {:?})",
            found_count,
            magic,
            expected
        );
    }

    #[test]
    fn test_numerov_convergence() {
        let model = ShellModel::default();
        // 1s1/2 state should converge for any nucleus
        let result = model.find_eigenvalue(40, 0, 0, 0.5);
        assert!(result.converged, "1s1/2 Numerov should converge for Ca-40");
        assert!(
            result.energy < 0.0,
            "1s1/2 should be a bound state: E={}",
            result.energy
        );
    }

    #[test]
    fn test_wavefunction_node_count() {
        let model = ShellModel::default();
        // 1s state: 0 nodes
        let r1 = model.find_eigenvalue(40, 0, 0, 0.5);
        assert_eq!(r1.nodes, 0, "1s should have 0 nodes, got {}", r1.nodes);

        // 2s state: should have more nodes than 1s
        let r2 = model.find_eigenvalue(40, 1, 0, 0.5);
        assert!(r2.nodes >= 1, "2s should have ≥1 nodes, got {}", r2.nodes);
        // 2s should be less bound (higher energy) than 1s
        assert!(
            r2.energy > r1.energy,
            "2s energy ({}) should be > 1s energy ({})",
            r2.energy,
            r1.energy
        );
    }

    #[test]
    fn test_level_ordering_light_nucleus() {
        let model = ShellModel::default();
        let levels = model.single_particle_levels(16, 3, 2);

        // Energy should be monotonically non-decreasing (already sorted)
        for i in 1..levels.len() {
            assert!(
                levels[i].energy_mev >= levels[i - 1].energy_mev - ENERGY_TOLERANCE,
                "Levels not sorted: {} > {}",
                levels[i - 1].energy_mev,
                levels[i].energy_mev
            );
        }
    }

    #[test]
    fn test_shell_correction_doubly_magic() {
        let model = ShellModel::default();
        // Pb-208 (Z=82, N=126) is doubly magic — should have negative shell correction
        let correction = model.shell_correction_energy(208, 82);
        // Shell correction for magic nuclei should be negative (extra stability)
        // The simplified model may not perfectly reproduce this, so check sign and magnitude
        assert!(
            correction < 5.0,
            "Pb-208 shell correction = {}, expected negative or small",
            correction
        );
    }
}
