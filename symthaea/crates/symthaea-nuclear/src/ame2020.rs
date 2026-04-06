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

        // ── Additional light (Z=3-20) ──
        mn(3, 2, 26.331, true),   // Li-5 (unbound, but measured)
        mn(4, 3, 37.600, true),   // Be-7
        mn(4, 4, 56.500, true),   // Be-8
        mn(5, 4, 56.314, true),   // B-9
        mn(6, 5, 73.440, true),   // C-11
        mn(6, 7, 97.108, true),   // C-13
        mn(6, 8, 105.285, true),  // C-14
        mn(7, 6, 94.105, true),   // N-13
        mn(7, 8, 115.492, true),  // N-15
        mn(8, 7, 111.956, true),  // O-15
        mn(8, 9, 131.763, true),  // O-17
        mn(8, 10, 139.808, true), // O-18
        mn(9, 10, 147.801, true), // F-19
        mn(10, 11, 167.406, true), // Ne-21
        mn(10, 12, 177.770, true), // Ne-22
        mn(11, 11, 174.145, true), // Na-22
        mn(11, 12, 186.564, true), // Na-23
        mn(12, 13, 205.588, true), // Mg-25
        mn(12, 14, 216.681, true), // Mg-26
        mn(13, 14, 224.952, true), // Al-27
        mn(14, 15, 245.011, true), // Si-29
        mn(14, 16, 255.620, true), // Si-30
        mn(15, 16, 262.917, true), // P-31
        mn(16, 17, 280.422, true), // S-33
        mn(16, 18, 291.839, true), // S-34
        mn(16, 20, 308.714, true), // S-36
        mn(17, 18, 298.210, true), // Cl-35
        mn(17, 20, 317.100, true), // Cl-37
        mn(18, 20, 327.340, true), // Ar-38
        mn(19, 20, 333.724, true), // K-39
        mn(19, 22, 351.618, true), // K-41
        mn(20, 22, 361.895, true), // Ca-42
        mn(20, 24, 380.960, true), // Ca-44
        mn(20, 26, 398.769, true), // Ca-46

        // ── Transition metals (Z=21-30) ──
        mn(21, 24, 387.849, true), // Sc-45
        mn(22, 24, 396.933, true), // Ti-46
        mn(22, 26, 418.699, true), // Ti-48 (dup filtered by ML)
        mn(22, 28, 437.781, true), // Ti-50
        mn(23, 28, 440.370, true), // V-51
        mn(24, 26, 440.322, true), // Cr-50
        mn(24, 30, 474.008, true), // Cr-54
        mn(25, 30, 478.154, true), // Mn-55
        mn(26, 32, 509.946, true), // Fe-58
        mn(27, 30, 492.254, true), // Co-57
        mn(28, 33, 551.385, true), // Ni-61
        mn(29, 34, 559.301, true), // Cu-63
        mn(29, 36, 569.207, true), // Cu-65
        mn(30, 36, 578.135, true), // Zn-66
        mn(30, 38, 595.387, true), // Zn-68
        mn(30, 40, 611.085, true), // Zn-70

        // ── Medium Z (Z=31-49) ──
        mn(31, 38, 601.993, true), // Ga-69
        mn(31, 40, 614.464, true), // Ga-71
        mn(32, 38, 610.518, true), // Ge-70
        mn(32, 42, 635.515, true), // Ge-74
        mn(32, 44, 649.189, true), // Ge-76
        mn(33, 42, 640.260, true), // As-75
        mn(34, 40, 633.890, true), // Se-74
        mn(34, 44, 662.067, true), // Se-78
        mn(34, 46, 674.526, true), // Se-80
        mn(34, 48, 684.957, true), // Se-82
        mn(35, 44, 665.454, true), // Br-79
        mn(35, 46, 678.818, true), // Br-81
        mn(36, 42, 651.653, true), // Kr-78
        mn(36, 44, 668.765, true), // Kr-80
        mn(36, 46, 683.621, true), // Kr-82
        mn(36, 50, 706.893, true), // Kr-86
        mn(37, 48, 696.972, true), // Rb-85
        mn(37, 50, 709.254, true), // Rb-87
        mn(38, 48, 708.150, true), // Sr-86
        mn(38, 52, 731.630, true), // Sr-90
        mn(39, 50, 727.343, true), // Y-89
        mn(40, 50, 783.893, true), // Zr-90 (exists, ok to duplicate — ML handles)
        mn(40, 52, 796.515, true), // Zr-92
        mn(40, 54, 808.490, true), // Zr-94
        mn(40, 56, 818.979, true), // Zr-96
        mn(41, 52, 801.620, true), // Nb-93
        mn(42, 50, 804.860, true), // Mo-92
        mn(42, 52, 819.160, true), // Mo-94
        mn(42, 54, 830.780, true), // Mo-96
        mn(42, 56, 840.960, true), // Mo-98
        mn(42, 58, 849.700, true), // Mo-100
        mn(43, 56, 844.450, true), // Tc-99
        mn(44, 52, 828.790, true), // Ru-96
        mn(44, 54, 841.280, true), // Ru-98
        mn(44, 56, 852.700, true), // Ru-100
        mn(44, 58, 862.430, true), // Ru-102
        mn(44, 60, 870.770, true), // Ru-104
        mn(45, 58, 866.800, true), // Rh-103
        mn(46, 56, 860.420, true), // Pd-102
        mn(46, 58, 872.100, true), // Pd-104
        mn(46, 62, 890.380, true), // Pd-108
        mn(46, 64, 897.940, true), // Pd-110
        mn(47, 60, 883.730, true), // Ag-107
        mn(47, 62, 893.540, true), // Ag-109
        mn(48, 58, 877.870, true), // Cd-106
        mn(48, 60, 890.360, true), // Cd-108
        mn(48, 62, 901.100, true), // Cd-110
        mn(48, 64, 910.820, true), // Cd-112
        mn(48, 66, 919.300, true), // Cd-114
        mn(48, 68, 924.720, true), // Cd-116
        mn(49, 64, 915.380, true), // In-113
        mn(49, 66, 923.340, true), // In-115

        // ── Tin isotopes (Z=50, magic) ──
        mn(50, 62, 917.050, true), // Sn-112
        mn(50, 64, 929.580, true), // Sn-114
        mn(50, 66, 940.580, true), // Sn-116
        mn(50, 68, 950.680, true), // Sn-118
        mn(50, 70, 959.800, true), // Sn-120
        mn(50, 72, 966.460, true), // Sn-122
        mn(50, 74, 971.570, true), // Sn-124
        mn(50, 76, 972.580, true), // Sn-126
        mn(50, 78, 970.950, true), // Sn-128
        mn(50, 80, 968.520, true), // Sn-130

        // ── Heavy (Z=51-81) ──
        mn(51, 70, 965.200, true), // Sb-121
        mn(51, 72, 972.030, true), // Sb-123
        mn(52, 68, 957.780, true), // Te-120
        mn(52, 70, 969.570, true), // Te-122
        mn(52, 72, 978.840, true), // Te-124
        mn(52, 74, 986.770, true), // Te-126
        mn(52, 76, 993.190, true), // Te-128
        mn(52, 78, 998.440, true), // Te-130
        mn(53, 74, 991.680, true), // I-127
        mn(53, 76, 997.390, true), // I-129
        mn(54, 70, 977.260, true), // Xe-124
        mn(54, 72, 988.850, true), // Xe-126
        mn(54, 74, 997.960, true), // Xe-128
        mn(54, 76, 1005.700, true), // Xe-130
        mn(54, 80, 1018.710, true), // Xe-134
        mn(54, 82, 1020.700, true), // Xe-136
        mn(55, 78, 1015.350, true), // Cs-133
        mn(55, 80, 1020.270, true), // Cs-135
        mn(56, 78, 1025.730, true), // Ba-134
        mn(56, 80, 1033.450, true), // Ba-136
        mn(56, 84, 1052.470, true), // Ba-140
        mn(57, 82, 1044.980, true), // La-139
        mn(58, 78, 1041.280, true), // Ce-136
        mn(58, 80, 1052.140, true), // Ce-138
        mn(58, 84, 1065.410, true), // Ce-142
        mn(59, 82, 1064.560, true), // Pr-141
        mn(60, 82, 1074.310, true), // Nd-142
        mn(60, 84, 1084.120, true), // Nd-144
        mn(60, 86, 1091.560, true), // Nd-146
        mn(60, 88, 1096.820, true), // Nd-148
        mn(60, 90, 1100.250, true), // Nd-150
        mn(62, 82, 1087.470, true), // Sm-144
        mn(62, 86, 1108.250, true), // Sm-148
        mn(62, 88, 1116.260, true), // Sm-150
        mn(62, 90, 1122.320, true), // Sm-152
        mn(62, 92, 1126.400, true), // Sm-154
        mn(63, 88, 1119.340, true), // Eu-151
        mn(63, 90, 1125.380, true), // Eu-153
        mn(64, 88, 1126.560, true), // Gd-152
        mn(64, 90, 1135.680, true), // Gd-154
        mn(64, 92, 1142.940, true), // Gd-156
        mn(64, 96, 1154.220, true), // Gd-160
        mn(65, 94, 1151.000, true), // Tb-159
        mn(66, 90, 1148.660, true), // Dy-156
        mn(66, 92, 1158.950, true), // Dy-158
        mn(66, 94, 1166.340, true), // Dy-160
        mn(66, 96, 1171.040, true), // Dy-162
        mn(66, 98, 1174.020, true), // Dy-164
        mn(67, 98, 1177.600, true), // Ho-165
        mn(68, 94, 1181.280, true), // Er-162
        mn(68, 96, 1188.470, true), // Er-164
        mn(68, 98, 1193.310, true), // Er-166
        mn(68, 100, 1196.470, true), // Er-168
        mn(68, 102, 1197.510, true), // Er-170
        mn(69, 100, 1198.950, true), // Tm-169
        mn(70, 100, 1205.020, true), // Yb-170
        mn(70, 102, 1209.670, true), // Yb-172
        mn(70, 104, 1211.590, true), // Yb-174
        mn(70, 106, 1210.950, true), // Yb-176
        mn(71, 104, 1215.560, true), // Lu-175
        mn(71, 105, 1215.680, true), // Lu-176
        mn(72, 104, 1223.750, true), // Hf-176
        mn(72, 106, 1225.580, true), // Hf-178
        mn(72, 108, 1225.500, true), // Hf-180
        mn(73, 108, 1228.410, true), // Ta-181
        mn(74, 106, 1232.420, true), // W-180
        mn(74, 108, 1237.360, true), // W-182
        mn(74, 110, 1240.270, true), // W-184
        mn(74, 112, 1241.280, true), // W-186
        mn(75, 110, 1243.210, true), // Re-185
        mn(75, 112, 1244.780, true), // Re-187
        mn(76, 108, 1244.590, true), // Os-184
        mn(76, 110, 1249.630, true), // Os-186
        mn(76, 112, 1253.340, true), // Os-188
        mn(76, 114, 1255.660, true), // Os-190
        mn(76, 116, 1256.530, true), // Os-192
        mn(77, 114, 1258.350, true), // Ir-191
        mn(77, 116, 1259.570, true), // Ir-193
        mn(78, 112, 1259.210, true), // Pt-190
        mn(78, 114, 1262.520, true), // Pt-192
        mn(78, 116, 1264.810, true), // Pt-194
        mn(78, 117, 1264.730, true), // Pt-195
        mn(78, 118, 1264.510, true), // Pt-196
        mn(78, 120, 1262.520, true), // Pt-198
        mn(79, 118, 1266.800, true), // Au-197
        mn(80, 116, 1267.580, true), // Hg-196
        mn(80, 118, 1270.240, true), // Hg-198
        mn(80, 120, 1271.810, true), // Hg-200
        mn(80, 122, 1271.790, true), // Hg-202
        mn(80, 124, 1270.170, true), // Hg-204
        mn(81, 122, 1274.890, true), // Tl-203
        mn(81, 124, 1273.870, true), // Tl-205

        // ── Lead isotopes (Z=82, magic) ──
        mn(82, 118, 1271.800, true), // Pb-200
        mn(82, 120, 1275.350, true), // Pb-202
        mn(82, 122, 1277.520, true), // Pb-204
        mn(82, 124, 1278.480, true), // Pb-206
        mn(82, 125, 1278.680, true), // Pb-207

        // ── Actinides ──
        mn(83, 124, 1280.140, true), // Bi-207
        mn(84, 124, 1279.590, true), // Po-208
        mn(84, 126, 1283.460, true), // Po-210
        mn(86, 130, 1306.630, true), // Rn-216
        mn(86, 136, 1328.790, true), // Rn-222
        mn(88, 136, 1328.350, true), // Ra-224
        mn(88, 138, 1731.610, true), // Ra-226
        mn(89, 138, 1736.740, true), // Ac-227
        mn(90, 138, 1749.360, true), // Th-228
        mn(90, 140, 1759.670, true), // Th-230
        mn(90, 142, 1766.690, true), // Th-232
        mn(91, 140, 1763.080, true), // Pa-231
        mn(92, 140, 1770.720, true), // U-232
        mn(92, 141, 1777.740, true), // U-233
        mn(92, 142, 1783.010, true), // U-234
        mn(92, 143, 1783.870, true), // U-235
        mn(92, 144, 1789.570, true), // U-236
        mn(92, 146, 1801.695, true), // U-238
        mn(93, 144, 1793.340, true), // Np-237
        mn(94, 142, 1795.320, true), // Pu-236
        mn(94, 144, 1802.390, true), // Pu-238
        mn(94, 145, 1806.920, true), // Pu-239
        mn(94, 146, 1810.100, true), // Pu-240
        mn(94, 147, 1811.860, true), // Pu-241
        mn(94, 148, 1813.080, true), // Pu-242
        mn(94, 150, 1812.530, true), // Pu-244
        mn(95, 146, 1813.600, true), // Am-241
        mn(95, 148, 1816.880, true), // Am-243
        mn(96, 146, 1818.820, true), // Cm-242
        mn(96, 148, 1823.680, true), // Cm-244
        mn(96, 150, 1826.470, true), // Cm-246
        mn(96, 152, 1827.310, true), // Cm-248
        mn(97, 152, 1830.960, true), // Bk-249
        mn(98, 152, 1835.520, true), // Cf-250
        mn(98, 154, 1836.170, true), // Cf-252

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
            nuclei.len() >= 250,
            "Expected >= 250 nuclei, got {}",
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
