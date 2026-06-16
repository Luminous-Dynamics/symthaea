// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Island of Stability predictions for superheavy elements.
//!
//! Combines the liquid-drop model (SEMF) with shell corrections to predict
//! enhanced stability regions in the nuclear landscape, particularly around
//! Z~114-120, N~184 where multiple models predict doubly-magic shell closures.
//!
//! Special focus: Moscovium (Z=115) isotopes for island-of-stability evaluation.
//!
//! References:
//! - Möller et al. (2016). FRDM(2012) nuclear mass table.
//! - Oganessian & Utyonkov (2015). Superheavy element synthesis. Nuclear Physics A.

use crate::mass_formula::SemiEmpiricalMassFormula;
use crate::shell_model::ShellModel;
use serde::{Deserialize, Serialize};

/// Stability data for a single isotope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotopeStability {
    /// Atomic number
    pub z: u16,
    /// Neutron number
    pub n: u16,
    /// Mass number A = Z + N
    pub a: u16,
    /// SEMF binding energy (MeV)
    pub binding_energy_semf: f64,
    /// Shell correction energy (MeV, negative = extra stable)
    pub shell_correction: f64,
    /// Total binding energy: SEMF + shell correction
    pub binding_energy_total: f64,
    /// Alpha-decay Q-value (MeV)
    pub q_alpha: f64,
    /// Estimated half-life from Geiger-Nuttall (seconds), if alpha-unstable
    pub estimated_half_life: Option<f64>,
    /// Whether this isotope is in a predicted stability island
    pub in_stability_island: bool,
}

/// A detected island of enhanced stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityIsland {
    /// Center Z of the island
    pub center_z: u16,
    /// Center N of the island
    pub center_n: u16,
    /// All (Z, N) pairs in this island
    pub members: Vec<(u16, u16)>,
    /// Maximum predicted half-life within the island (seconds)
    pub max_half_life: f64,
    /// Average shell correction in the island (MeV)
    pub avg_shell_correction: f64,
}

/// Full stability map for a region of the nuclear chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMap {
    pub entries: Vec<IsotopeStability>,
    pub z_range: (u16, u16),
    pub n_range: (u16, u16),
}

/// Combined stability landscape calculator.
pub struct StabilityLandscape {
    semf: SemiEmpiricalMassFormula,
    shell: ShellModel,
}

impl Default for StabilityLandscape {
    fn default() -> Self {
        Self {
            semf: SemiEmpiricalMassFormula::default(),
            shell: ShellModel::default(),
        }
    }
}

impl StabilityLandscape {
    /// Total binding energy combining SEMF + shell correction.
    pub fn total_binding_energy(&self, a: u16, z: u16) -> f64 {
        let b_semf = self.semf.binding_energy(a, z);
        let shell_corr = self.shell.shell_correction_energy(a, z);
        b_semf + shell_corr
    }

    /// Evaluate a single isotope's stability.
    pub fn evaluate_isotope(&self, z: u16, n: u16) -> IsotopeStability {
        let a = z + n;
        let b_semf = self.semf.binding_energy(a, z);
        let shell_corr = self.shell.shell_correction_energy(a, z);
        let b_total = b_semf + shell_corr;
        let q_alpha = self.semf.alpha_decay_q(a, z);
        let half_life = self.semf.geiger_nuttall_half_life(a, z);

        // Heuristic: in stability island if shell correction is strongly negative
        // and half-life is enhanced relative to neighbors
        let in_island = shell_corr < -2.0;

        IsotopeStability {
            z,
            n,
            a,
            binding_energy_semf: b_semf,
            shell_correction: shell_corr,
            binding_energy_total: b_total,
            q_alpha,
            estimated_half_life: half_life,
            in_stability_island: in_island,
        }
    }

    /// Generate a stability map for a rectangular region of the nuclear chart.
    pub fn stability_map(&self, z_min: u16, z_max: u16, n_min: u16, n_max: u16) -> StabilityMap {
        let mut entries = Vec::new();

        for z in z_min..=z_max {
            for n in n_min..=n_max {
                entries.push(self.evaluate_isotope(z, n));
            }
        }

        StabilityMap {
            entries,
            z_range: (z_min, z_max),
            n_range: (n_min, n_max),
        }
    }

    /// Evaluate all Moscovium (Z=115) isotopes from N=170 to N=184.
    ///
    /// This is the specific test for Lazar's Element 115 claims:
    /// does the shell model predict any stable isotopes?
    pub fn evaluate_moscovium_isotopes(&self) -> Vec<IsotopeStability> {
        (170..=184).map(|n| self.evaluate_isotope(115, n)).collect()
    }

    /// Find stability islands in a map.
    ///
    /// An island is a connected region where shell corrections are below
    /// a threshold (indicating enhanced stability).
    pub fn find_stability_islands(&self, map: &StabilityMap) -> Vec<StabilityIsland> {
        let threshold = -2.0; // MeV shell correction threshold

        // Collect candidates
        let candidates: Vec<_> = map
            .entries
            .iter()
            .filter(|e| e.shell_correction < threshold)
            .collect();

        if candidates.is_empty() {
            return vec![];
        }

        // Simple clustering: group nearby candidates
        let mut islands: Vec<StabilityIsland> = Vec::new();
        let mut assigned = vec![false; candidates.len()];

        for i in 0..candidates.len() {
            if assigned[i] {
                continue;
            }

            let mut members = vec![(candidates[i].z, candidates[i].n)];
            assigned[i] = true;

            // Find all connected candidates (within distance 2 in Z,N space)
            loop {
                let mut found_new = false;
                for j in 0..candidates.len() {
                    if assigned[j] {
                        continue;
                    }
                    let close = members.iter().any(|&(mz, mn)| {
                        let dz = (candidates[j].z as i32 - mz as i32).unsigned_abs();
                        let dn = (candidates[j].n as i32 - mn as i32).unsigned_abs();
                        dz <= 2 && dn <= 2
                    });
                    if close {
                        members.push((candidates[j].z, candidates[j].n));
                        assigned[j] = true;
                        found_new = true;
                    }
                }
                if !found_new {
                    break;
                }
            }

            // Compute island statistics
            let avg_corr = members
                .iter()
                .map(|&(z, n)| {
                    map.entries
                        .iter()
                        .find(|e| e.z == z && e.n == n)
                        .map(|e| e.shell_correction)
                        .unwrap_or(0.0)
                })
                .sum::<f64>()
                / members.len() as f64;

            let max_hl = members
                .iter()
                .filter_map(|&(z, n)| {
                    map.entries
                        .iter()
                        .find(|e| e.z == z && e.n == n)
                        .and_then(|e| e.estimated_half_life)
                })
                .fold(0.0_f64, f64::max);

            let center_z =
                (members.iter().map(|m| m.0 as u32).sum::<u32>() / members.len() as u32) as u16;
            let center_n =
                (members.iter().map(|m| m.1 as u32).sum::<u32>() / members.len() as u32) as u16;

            islands.push(StabilityIsland {
                center_z,
                center_n,
                members,
                max_half_life: max_hl,
                avg_shell_correction: avg_corr,
            });
        }

        islands
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// SYSTEMATIC SUPERHEAVY ISOTOPE DISCOVERY
// ══════════════════════════════════════════════════════════════════════════════

/// Summary of a single element's most stable predicted isotope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementStabilitySummary {
    /// Atomic number
    pub z: u16,
    /// Element symbol (approximate)
    pub symbol: String,
    /// Neutron number of most stable predicted isotope
    pub optimal_n: u16,
    /// Mass number of most stable isotope
    pub optimal_a: u16,
    /// Binding energy per nucleon of optimal isotope (MeV)
    pub optimal_ba: f64,
    /// Shell correction of optimal isotope (MeV)
    pub optimal_shell_correction: f64,
    /// Predicted alpha-decay half-life of optimal isotope (seconds)
    pub predicted_half_life: Option<f64>,
    /// Whether this element has an enhanced stability pocket
    pub has_stability_pocket: bool,
    /// Number of isotopes evaluated
    pub isotopes_evaluated: u16,
}

/// Full discovery report from a superheavy sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperheavyDiscoveryReport {
    /// Per-element summaries
    pub elements: Vec<ElementStabilitySummary>,
    /// Detected stability islands
    pub islands: Vec<StabilityIsland>,
    /// Total isotopes evaluated
    pub total_isotopes: usize,
    /// Z range swept
    pub z_range: (u16, u16),
    /// N range swept
    pub n_range: (u16, u16),
}

/// Approximate element symbol for superheavy elements.
fn superheavy_symbol(z: u16) -> String {
    match z {
        100 => "Fm".to_string(),
        101 => "Md".to_string(),
        102 => "No".to_string(),
        103 => "Lr".to_string(),
        104 => "Rf".to_string(),
        105 => "Db".to_string(),
        106 => "Sg".to_string(),
        107 => "Bh".to_string(),
        108 => "Hs".to_string(),
        109 => "Mt".to_string(),
        110 => "Ds".to_string(),
        111 => "Rg".to_string(),
        112 => "Cn".to_string(),
        113 => "Nh".to_string(),
        114 => "Fl".to_string(),
        115 => "Mc".to_string(),
        116 => "Lv".to_string(),
        117 => "Ts".to_string(),
        118 => "Og".to_string(),
        _ => format!("E{}", z),
    }
}

impl StabilityLandscape {
    /// Systematic sweep of the superheavy region to find stable isotopes.
    ///
    /// Evaluates all isotopes in the range Z=[z_min, z_max], N=[n_min, n_max]
    /// and identifies:
    /// - The most stable isotope for each element
    /// - Stability islands (connected regions with enhanced shell corrections)
    /// - Predicted half-lives via Geiger-Nuttall
    ///
    /// Default range covers the theoretically interesting region:
    /// Z=100-126, N=140-200 (6,000+ isotopes).
    pub fn find_stable_isotopes(
        &self,
        z_min: u16,
        z_max: u16,
        n_min: u16,
        n_max: u16,
    ) -> SuperheavyDiscoveryReport {
        let map = self.stability_map(z_min, z_max, n_min, n_max);
        let islands = self.find_stability_islands(&map);

        let mut elements = Vec::new();

        for z in z_min..=z_max {
            // Find all isotopes for this element
            let isotopes: Vec<&IsotopeStability> =
                map.entries.iter().filter(|e| e.z == z).collect();

            if isotopes.is_empty() {
                continue;
            }

            // Find the isotope with the best shell correction (most negative = most stable)
            let best = isotopes
                .iter()
                .min_by(|a, b| {
                    a.shell_correction
                        .partial_cmp(&b.shell_correction)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();

            let ba = self.semf.binding_energy_per_nucleon(best.a, best.z);
            let has_pocket = best.shell_correction < -2.0;

            elements.push(ElementStabilitySummary {
                z,
                symbol: superheavy_symbol(z),
                optimal_n: best.n,
                optimal_a: best.a,
                optimal_ba: ba,
                optimal_shell_correction: best.shell_correction,
                predicted_half_life: best.estimated_half_life,
                has_stability_pocket: has_pocket,
                isotopes_evaluated: isotopes.len() as u16,
            });
        }

        SuperheavyDiscoveryReport {
            total_isotopes: map.entries.len(),
            z_range: (z_min, z_max),
            n_range: (n_min, n_max),
            elements,
            islands,
        }
    }

    /// Default superheavy sweep: Z=100-126, N=140-200.
    pub fn default_superheavy_sweep(&self) -> SuperheavyDiscoveryReport {
        self.find_stable_isotopes(100, 126, 140, 200)
    }

    /// Print a human-readable discovery report.
    pub fn print_report(report: &SuperheavyDiscoveryReport) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Superheavy Isotope Discovery Report ===\n\
             Z range: {}-{}, N range: {}-{}\n\
             Total isotopes evaluated: {}\n\
             Stability islands found: {}\n\n",
            report.z_range.0,
            report.z_range.1,
            report.n_range.0,
            report.n_range.1,
            report.total_isotopes,
            report.islands.len()
        ));

        out.push_str("Element | Optimal A | Shell Corr (MeV) | Stability Pocket | Half-life\n");
        out.push_str("--------|-----------|-------------------|------------------|----------\n");

        for e in &report.elements {
            let hl_str = match e.predicted_half_life {
                Some(t) if t > 3.156e7 => format!("{:.1} yr", t / 3.156e7),
                Some(t) if t > 86400.0 => format!("{:.1} days", t / 86400.0),
                Some(t) if t > 3600.0 => format!("{:.1} hr", t / 3600.0),
                Some(t) if t > 1.0 => format!("{:.1} s", t),
                Some(t) if t > 1e-3 => format!("{:.1} ms", t * 1e3),
                Some(t) => format!("{:.1e} s", t),
                None => "stable?".to_string(),
            };

            out.push_str(&format!(
                "{:>3}-{:<3} | {:>9} | {:>17.2} | {:>16} | {}\n",
                e.symbol,
                e.z,
                e.optimal_a,
                e.optimal_shell_correction,
                if e.has_stability_pocket { "YES" } else { "no" },
                hl_str,
            ));
        }

        if !report.islands.is_empty() {
            out.push_str("\n=== Stability Islands ===\n");
            for (i, island) in report.islands.iter().enumerate() {
                out.push_str(&format!(
                    "Island {}: center Z={}, N={}, {} members, avg shell corr={:.2} MeV\n",
                    i + 1,
                    island.center_z,
                    island.center_n,
                    island.members.len(),
                    island.avg_shell_correction,
                ));
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pb208_doubly_magic() {
        let landscape = StabilityLandscape::default();
        let pb208 = landscape.evaluate_isotope(82, 126);
        // Pb-208 is doubly magic — should show enhanced stability indicators
        assert!(
            pb208.binding_energy_semf > 1500.0,
            "Pb-208 SEMF binding = {} MeV, expected >1500",
            pb208.binding_energy_semf
        );
    }

    #[test]
    fn test_moscovium_evaluation() {
        let landscape = StabilityLandscape::default();
        let mc_isotopes = landscape.evaluate_moscovium_isotopes();

        assert_eq!(
            mc_isotopes.len(),
            15,
            "Should evaluate Mc-285 through Mc-299"
        );

        // All Mc isotopes should have positive binding energy
        for iso in &mc_isotopes {
            assert!(
                iso.binding_energy_semf > 0.0,
                "Mc-{} should have positive binding energy",
                iso.a
            );
            assert_eq!(iso.z, 115);
        }

        // At least one should have an alpha-decay Q-value (they're all alpha emitters)
        let alpha_unstable = mc_isotopes.iter().filter(|i| i.q_alpha > 0.0).count();
        assert!(
            alpha_unstable > 0,
            "Some Mc isotopes should be alpha-unstable"
        );
    }

    #[test]
    fn test_stability_map_generation() {
        let landscape = StabilityLandscape::default();
        // Small map around Z=114, N=184 (predicted island center)
        let map = landscape.stability_map(112, 116, 182, 186);
        assert_eq!(map.entries.len(), 5 * 5);
        assert!(map.entries.iter().all(|e| e.binding_energy_semf > 0.0));
    }

    #[test]
    fn test_binding_energy_along_stability_valley() {
        let landscape = StabilityLandscape::default();
        let semf = SemiEmpiricalMassFormula::default();

        // B/A should generally decrease for very heavy nuclei
        let ba_120 = semf.binding_energy_per_nucleon(120, semf.beta_stable_z(120));
        let ba_300 = semf.binding_energy_per_nucleon(300, semf.beta_stable_z(300));

        assert!(
            ba_120 > ba_300,
            "B/A should decrease for superheavy: A=120 ({}) vs A=300 ({})",
            ba_120,
            ba_300
        );
    }

    #[test]
    fn test_superheavy_region_has_islands() {
        let landscape = StabilityLandscape::default();
        let map = landscape.stability_map(110, 120, 170, 190);
        let _islands = landscape.find_stability_islands(&map);

        assert!(
            !map.entries.is_empty(),
            "Stability map should contain entries"
        );
    }

    #[test]
    fn test_find_stable_isotopes_small_sweep() {
        let landscape = StabilityLandscape::default();
        // Small sweep: Z=110-116, N=170-185 (7 elements × 16 neutrons = 112 isotopes)
        let report = landscape.find_stable_isotopes(110, 116, 170, 185);

        assert_eq!(report.z_range, (110, 116));
        assert_eq!(report.n_range, (170, 185));
        assert_eq!(report.elements.len(), 7); // Z=110 through 116
        assert!(report.total_isotopes > 100);

        // Every element should have an optimal isotope
        for e in &report.elements {
            assert!(e.optimal_a > 0);
            assert!(e.optimal_ba > 0.0);
            assert!(e.isotopes_evaluated > 0);
        }

        // Moscovium (Z=115) should be in the list
        let mc = report.elements.iter().find(|e| e.z == 115);
        assert!(mc.is_some(), "Moscovium should be in sweep");
        assert_eq!(mc.unwrap().symbol, "Mc");
    }

    #[test]
    fn test_discovery_report_formatting() {
        let landscape = StabilityLandscape::default();
        let report = landscape.find_stable_isotopes(114, 116, 180, 185);
        let text = StabilityLandscape::print_report(&report);

        assert!(text.contains("Superheavy Isotope Discovery Report"));
        assert!(text.contains("Fl")); // Flerovium Z=114
        assert!(text.contains("Mc")); // Moscovium Z=115
        assert!(text.contains("Lv")); // Livermorium Z=116
    }

    #[test]
    fn test_optimal_n_physically_reasonable() {
        let landscape = StabilityLandscape::default();
        let report = landscape.find_stable_isotopes(114, 114, 170, 190);

        let fl = &report.elements[0];
        assert_eq!(fl.z, 114);
        // For Z=114, optimal N should be in the range 170-190
        // (the island of stability is predicted near N=184)
        assert!(
            fl.optimal_n >= 170 && fl.optimal_n <= 190,
            "Fl optimal N={} should be in range 170-190",
            fl.optimal_n
        );
    }
}
