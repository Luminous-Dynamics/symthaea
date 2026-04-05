// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AME2020 / NUBASE2020 Reference Nuclear Data
//!
//! Curated subset of ~70 nuclei from the Atomic Mass Evaluation 2020
//! (Wang et al., Chinese Physics C, 2021) and NUBASE2020 (Kondev et al., 2021).
//!
//! Binding energies are total binding energy in MeV, sourced from AME2020 Table I.
//! Superheavy element values are extrapolated estimates where noted.
//!
//! Zero external dependencies — all data is hardcoded for WASM compatibility.

use crate::discovery::MeasuredNucleus;

/// Return ~70 curated nuclei spanning H-2 to Og-294.
///
/// Coverage:
/// - All doubly-magic nuclei (shell closures)
/// - Iron peak (maximum B/A)
/// - Beta-stable nuclei at A≈20 intervals
/// - Light nuclei (H through B)
/// - Benchmark heavy nuclei (U, Th, Pu)
/// - All synthesized superheavy elements (Z=104-118)
pub fn ame2020_reference_nuclei() -> Vec<MeasuredNucleus> {
    vec![
        // ── Light nuclei ──
        mn(1, 1, 2.225, true),    // H-2 (deuteron)
        mn(1, 2, 8.482, true),    // H-3 (tritium)
        mn(2, 1, 7.718, true),    // He-3
        mn(2, 2, 28.296, true),   // He-4 (alpha) — doubly magic
        mn(3, 3, 31.995, true),   // Li-6
        mn(3, 4, 39.245, true),   // Li-7
        mn(4, 5, 58.165, true),   // Be-9
        mn(5, 5, 64.751, true),   // B-10
        mn(5, 6, 76.205, true),   // B-11
        mn(6, 6, 92.162, true),   // C-12 (defines AMU)
        mn(7, 7, 104.659, true),  // N-14
        mn(8, 8, 127.619, true),  // O-16 — doubly magic

        // ── Beta-stable at A≈20 intervals ──
        mn(10, 10, 160.645, true), // Ne-20
        mn(12, 12, 198.257, true), // Mg-24
        mn(14, 14, 236.537, true), // Si-28
        mn(16, 16, 271.780, true), // S-32
        mn(18, 22, 343.810, true), // Ar-40
        mn(20, 20, 342.052, true), // Ca-40 — doubly magic
        mn(20, 28, 415.991, true), // Ca-48 — doubly magic
        mn(22, 26, 418.699, true), // Ti-48
        mn(24, 28, 456.349, true), // Cr-52

        // ── Iron peak (maximum B/A) ──
        mn(26, 28, 471.763, true), // Fe-54
        mn(26, 30, 492.254, true), // Fe-56 (highest B/A)
        mn(27, 32, 498.286, true), // Co-59
        mn(28, 28, 484.003, true), // Ni-56 — doubly magic
        mn(28, 30, 506.454, true), // Ni-58
        mn(28, 32, 526.842, true), // Ni-60
        mn(28, 34, 545.259, true), // Ni-62 (most tightly bound per nucleon)

        // ── Medium-heavy ──
        mn(30, 34, 559.094, true), // Zn-64
        mn(32, 40, 614.275, true), // Ge-72
        mn(36, 48, 714.273, true), // Kr-84
        mn(38, 50, 748.927, true), // Sr-88
        mn(40, 50, 783.893, true), // Zr-90
        mn(42, 56, 846.243, true), // Mo-98
        mn(46, 60, 909.476, true), // Pd-106
        mn(48, 66, 972.600, true), // Cd-114
        mn(50, 70, 1009.870, true), // Sn-120
        mn(50, 82, 1102.850, true), // Sn-132 — doubly magic

        // ── Heavy stable ──
        mn(54, 78, 1083.677, true), // Xe-132
        mn(56, 82, 1158.296, true), // Ba-138
        mn(58, 82, 1172.687, true), // Ce-140
        mn(60, 84, 1191.246, true), // Nd-144
        mn(64, 94, 1299.150, true), // Gd-158
        mn(68, 100, 1361.553, true), // Er-168
        mn(72, 108, 1452.180, true), // Hf-180
        mn(76, 116, 1526.107, true), // Os-192
        mn(78, 118, 1546.617, true), // Pt-196
        mn(80, 122, 1581.197, true), // Hg-202

        // ── Lead region (doubly magic) ──
        mn(82, 126, 1636.430, true), // Pb-208 — doubly magic (most stable heavy)
        mn(83, 126, 1640.244, true), // Bi-209 (heaviest "stable")

        // ── Actinides ──
        mn(88, 138, 1731.610, true), // Ra-226
        mn(90, 142, 1766.690, true), // Th-232
        mn(92, 143, 1783.870, true), // U-235
        mn(92, 146, 1801.695, true), // U-238
        mn(94, 145, 1806.920, true), // Pu-239

        // ── Superheavy (synthesized, many extrapolated) ──
        mn(104, 163, 1876.0, false), // Rf-267
        mn(105, 163, 1883.0, false), // Db-268
        mn(106, 165, 1900.0, false), // Sg-271
        mn(107, 163, 1895.0, false), // Bh-270
        mn(108, 169, 1930.0, false), // Hs-277
        mn(109, 169, 1935.0, false), // Mt-278
        mn(110, 171, 1950.0, false), // Ds-281
        mn(111, 171, 1953.0, false), // Rg-282
        mn(112, 173, 1970.0, false), // Cn-285
        mn(113, 173, 1972.0, false), // Nh-286
        mn(114, 175, 1990.0, false), // Fl-289
        mn(115, 175, 1992.0, false), // Mc-290
        mn(116, 177, 2005.0, false), // Lv-293
        mn(117, 177, 2007.0, false), // Ts-294
        mn(118, 176, 2010.0, false), // Og-294
    ]
}

/// Shorthand constructor for MeasuredNucleus.
fn mn(z: u16, n: u16, binding_energy_mev: f64, is_measured: bool) -> MeasuredNucleus {
    MeasuredNucleus {
        z,
        n,
        binding_energy_mev,
        is_measured,
    }
}

/// Number of measured (experimental) nuclei in the AME2020 set.
pub fn measured_count() -> usize {
    ame2020_reference_nuclei()
        .iter()
        .filter(|n| n.is_measured)
        .count()
}

/// Number of extrapolated nuclei in the AME2020 set.
pub fn extrapolated_count() -> usize {
    ame2020_reference_nuclei()
        .iter()
        .filter(|n| !n.is_measured)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ame2020_count() {
        let nuclei = ame2020_reference_nuclei();
        assert!(
            nuclei.len() >= 65,
            "Expected >= 65 nuclei, got {}",
            nuclei.len()
        );
    }

    #[test]
    fn test_all_binding_energies_positive() {
        for n in ame2020_reference_nuclei() {
            assert!(
                n.binding_energy_mev > 0.0,
                "Z={} N={}: BE={} should be positive",
                n.z, n.n, n.binding_energy_mev
            );
        }
    }

    #[test]
    fn test_binding_energy_ordering() {
        let nuclei = ame2020_reference_nuclei();
        // He-4 should have less total BE than Pb-208
        let he4 = nuclei.iter().find(|n| n.z == 2 && n.n == 2).unwrap();
        let pb208 = nuclei.iter().find(|n| n.z == 82 && n.n == 126).unwrap();
        assert!(pb208.binding_energy_mev > he4.binding_energy_mev);
    }

    #[test]
    fn test_measured_vs_extrapolated() {
        assert!(measured_count() >= 50, "Should have >= 50 measured nuclei");
        assert!(extrapolated_count() >= 10, "Should have >= 10 superheavy extrapolated");
    }

    #[test]
    fn test_fe56_ba_peak() {
        let nuclei = ame2020_reference_nuclei();
        let fe56 = nuclei.iter().find(|n| n.z == 26 && n.n == 30).unwrap();
        let ba = fe56.binding_energy_mev / (fe56.z + fe56.n) as f64;
        // Fe-56 B/A should be ~8.79 MeV
        assert!(
            (ba - 8.79).abs() < 0.1,
            "Fe-56 B/A = {}, expected ~8.79",
            ba
        );
    }
}
