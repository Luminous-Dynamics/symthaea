// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! STO-3G minimal basis set.
//!
//! Each Slater-type orbital is expanded in 3 primitive Gaussians.
//! Data from Hehre, Stewart & Pople (1969), J. Chem. Phys. 51, 2657 (H-Ar)
//! and Pietro et al. (1980/1981/1983) for K through Xe. Fetched directly
//! from the Basis Set Exchange REST API
//! (basissetexchange.org/api/basis/sto-3g/format/json) as raw JSON,
//! deliberately not via a summarizing fetch (an earlier attempt at that
//! garbled several elements' exponents, Phase A.7, 2026-07-16) --
//! vendored verbatim at `basis/reference/sto3g_bse_h_xe.json`. Generated into
//! Rust via `scripts/generate_sto3g.py`, a mechanical, per-shell generator
//! (not hand-picked per-element/per-block match arms): shell shapes vary
//! meaningfully across the periodic table (H-He is S-only; Li-Ar is
//! S,SP,SP; K-Ca adds a 4th SP shell; Sc-Zn adds a trailing standalone D
//! shell; Ga-Kr's outer shell is a combined SPD block; Rb-Sr repeats K-Ca's
//! shape plus an SPD block; Y-Cd adds a trailing standalone D on top of
//! that; In-Xe has two SPD blocks) -- at least 8 distinct shapes,
//! confirmed against the raw JSON before writing the generator (Phase A.8,
//! 2026-07-16).
//!
//! **Complete real STO-3G coverage: Z=1-54 (H through Xe).** This is not
//! an arbitrary cutoff -- BSE's canonical STO-3G family genuinely stops at
//! Xe (verified by fetching the basis with no element filter: exactly 54
//! contiguous elements, no gaps, every element Z=21-54 carries real,
//! distinct literature references). Lanthanides, actinides, and anything
//! past Xe are deliberately not implemented: honest STO-3G coverage there
//! needs relativistic corrections and/or effective-core-potential
//! treatments this minimal basis can't provide -- claiming otherwise would
//! misrepresent what this basis set actually is.
//!
//! D-containing shells (Sc-Zn's standalone D; Ga-Kr/Rb-Xe's combined SPD
//! blocks) use BSE's `gto_spherical` function type (5 spherical d
//! functions); this crate's `ShellType::D` produces 6 Cartesian d
//! functions instead (see `ShellType::cartesian_components`) -- a real,
//! common alternative convention, not a data error, but it means
//! basis-function *counts* for any D-block element won't exactly match a
//! strict spherical-harmonic reference even though the underlying
//! primitive exponents/coefficients are correct.
//!
//! Na/Mg/Al's (and separately Si vs Na, Mg vs Al) valence-shell exponents
//! being pairwise identical is a genuine, verified feature of STO-3G's
//! original "standard scale factor" convention (Hehre/Stewart/Pople
//! assigned valence scale factors from a small discrete similarity table,
//! not continuous per-element optimization) -- confirmed via the raw BSE
//! JSON, not an artifact of how this data was fetched.
//!
//! **Deliberately not widened elsewhere from this extension alone**:
//! `symthaea-process-discovery`'s `ALLOWED_ELEMENTS` stays at
//! H/C/N/O/F/Si/P/S/Cl/Br -- the frozen USPTO evaluation corpus contains
//! zero transition-metal chemistry, and that crate has no catalysis-aware
//! template logic to make use of Sc-Cd/etc. even if allowed. This basis
//! extension is `symthaea-quantum-chemistry`-internal.

use super::{BasisSet, BasisSetProvider, ContractedGaussian, PrimitiveGaussian, ShellType};
use crate::molecule::Molecule;

/// STO-3G basis set provider.
pub struct Sto3g;

/// STO-3G shell data: (exponents, coefficients) for each contraction.
struct ShellData {
    shell_type: ShellType,
    exponents: [f64; 3],
    coefficients: [f64; 3],
}

/// Get STO-3G shell data for an element by atomic number.
/// Returns a list of shells (1s for H/He, 1s+2s+2p for Li-Ne, etc.)
fn shells_for_element(z: u8) -> Vec<ShellData> {
    match z {
        // ── Hydrogen (Z=1) ───────────────────────────────────
        1 => vec![ShellData {
            shell_type: ShellType::S,
            exponents: [3.425250914, 0.6239137298, 0.168855404],
            coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
        }],

        // ── Helium (Z=2) ─────────────────────────────────────
        2 => vec![ShellData {
            shell_type: ShellType::S,
            exponents: [6.362421394, 1.158922999, 0.3136497915],
            coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
        }],

        // ── Lithium (Z=3) ────────────────────────────────────
        3 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [16.11957475, 2.936200663, 0.794650487],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.6362897469, 0.1478600533, 0.0480886784],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.6362897469, 0.1478600533, 0.0480886784],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
        ],

        // ── Beryllium (Z=4) ──────────────────────────────────
        4 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [30.16787069, 5.495115306, 1.487192653],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.31483311, 0.3055389383, 0.0993707456],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.31483311, 0.3055389383, 0.0993707456],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
        ],

        // ── Boron (Z=5) ──────────────────────────────────────
        5 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [48.79111318, 8.887362172, 2.40526704],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.236956142, 0.5198204999, 0.16906176],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.236956142, 0.5198204999, 0.16906176],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
        ],

        // ── Carbon (Z=6) ─────────────────────────────────────
        6 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [71.61683735, 13.04509632, 3.53051216],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.941249355, 0.6834830964, 0.2222899159],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.941249355, 0.6834830964, 0.2222899159],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
        ],

        // ── Nitrogen (Z=7) ───────────────────────────────────
        7 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [99.10616896, 18.05231239, 4.885660238],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [3.780455879, 0.8784966449, 0.2857143744],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [3.780455879, 0.8784966449, 0.2857143744],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
        ],

        // ── Oxygen (Z=8) ─────────────────────────────────────
        8 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [130.7093214, 23.80886605, 6.443608313],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [5.033151319, 1.169596125, 0.38038896],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [5.033151319, 1.169596125, 0.38038896],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
        ],

        // ── Fluorine (Z=9) ───────────────────────────────────
        9 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [166.679134, 30.36081233, 8.216820672],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [6.464803249, 1.502281245, 0.4885884864],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [6.464803249, 1.502281245, 0.4885884864],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
        ],

        // ── Neon (Z=10) ──────────────────────────────────────
        10 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [207.015607, 37.70815124, 10.20529731],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [8.24631512, 1.916266291, 0.6232292721],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [8.24631512, 1.916266291, 0.6232292721],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
        ],

        // ── Sodium (Z=11) ────────────────────────────────────
        11 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [250.77243, 45.67851117, 12.36238776],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [12.04019274, 2.797881859, 0.909958017],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [12.04019274, 2.797881859, 0.909958017],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.478740622, 0.4125648801, 0.1614750979],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.478740622, 0.4125648801, 0.1614750979],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
        ],

        // ── Magnesium (Z=12) ─────────────────────────────────
        12 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [299.2374137, 54.50646845, 14.75157752],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [15.12182352, 3.513986579, 1.142857498],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [15.12182352, 3.513986579, 1.142857498],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.395448293, 0.3893265318, 0.1523797659],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.395448293, 0.3893265318, 0.1523797659],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
        ],

        // ── Aluminum (Z=13) ──────────────────────────────────
        13 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [351.4214767, 64.01186067, 17.32410761],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [18.89939621, 4.391813233, 1.42835397],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [18.89939621, 4.391813233, 1.42835397],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.395448293, 0.3893265318, 0.1523797659],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.395448293, 0.3893265318, 0.1523797659],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
        ],

        // ── Silicon (Z=14) ───────────────────────────────────
        14 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [407.7975514, 74.28083305, 20.10329229],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [23.19365606, 5.389706871, 1.752899952],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [23.19365606, 5.389706871, 1.752899952],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.478740622, 0.4125648801, 0.1614750979],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.478740622, 0.4125648801, 0.1614750979],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
        ],

        // ── Phosphorus (Z=15) ────────────────────────────────
        15 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [468.3656378, 85.31338559, 23.08913156],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [28.03263958, 6.514182577, 2.118614352],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [28.03263958, 6.514182577, 2.118614352],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.743103231, 0.4863213771, 0.1903428909],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.743103231, 0.4863213771, 0.1903428909],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
        ],

        // ── Sulfur (Z=16) ────────────────────────────────────
        16 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [533.1257359, 97.1095183, 26.28162542],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [33.32975173, 7.745117521, 2.518952599],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [33.32975173, 7.745117521, 2.518952599],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.029194274, 0.5661400518, 0.2215833792],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.029194274, 0.5661400518, 0.2215833792],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
        ],

        // ── Chlorine (Z=17) ──────────────────────────────────
        17 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [601.3456136, 109.5358542, 29.64467686],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [38.96041889, 9.053563477, 2.944499834],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [38.96041889, 9.053563477, 2.944499834],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.129386495, 0.5940934274, 0.232524141],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.129386495, 0.5940934274, 0.232524141],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
        ],

        // ── Argon (Z=18) ─────────────────────────────────────
        18 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [674.4465184, 122.8512753, 33.24834945],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [45.16424392, 10.495199, 3.413364448],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [45.16424392, 10.495199, 3.413364448],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.621366518, 0.731354605, 0.2862472356],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.621366518, 0.731354605, 0.2862472356],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
        ],

        // ── Potassium (Z=19) ─────────────────────────────────
        19 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [771.5103681, 140.5315766, 38.03332899],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [52.40203979, 12.1771071, 3.960373165],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [52.40203979, 12.1771071, 3.960373165],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [3.651583985, 1.018782663, 0.3987446295],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [3.651583985, 1.018782663, 0.3987446295],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.5039822505, 0.1860011465, 0.08214006743],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.5039822505, 0.1860011465, 0.08214006743],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
        ],

        // ── Calcium (Z=20) ───────────────────────────────────
        20 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [854.0324951, 155.5630851, 42.10144179],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [59.56029944, 13.8405327, 4.501370797],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [59.56029944, 13.8405327, 4.501370797],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [4.374706256, 1.220531941, 0.4777079296],
                coefficients: [-0.219620369, 0.2255954336, 0.900398426],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [4.374706256, 1.220531941, 0.4777079296],
                coefficients: [0.01058760429, 0.5951670053, 0.462001012],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4558489757, 0.168236941, 0.07429520696],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4558489757, 0.168236941, 0.07429520696],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
        ],

        // ── Scandium (Z=21) ──────────────────────────────────
        21 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [941.662425, 171.5249862, 46.42135516],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [67.17668771, 15.61041754, 5.076992278],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [67.17668771, 15.61041754, 5.076992278],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [4.698159231, 1.433088313, 0.5529300235],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [4.698159231, 1.433088313, 0.5529300235],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.6309328384, 0.2328538976, 0.1028307363],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.6309328384, 0.2328538976, 0.1028307363],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [0.5517000679, 0.1682861055, 0.0649300112],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Titanium (Z=22) ──────────────────────────────────
        22 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1033.571245, 188.2662926, 50.95220601],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [75.2512046, 17.48676162, 5.687237606],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [75.2512046, 17.48676162, 5.687237606],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [5.395535474, 1.645810296, 0.6350047773],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [5.395535474, 1.645810296, 0.6350047773],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.7122640246, 0.2628702203, 0.1160862609],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.7122640246, 0.2628702203, 0.1160862609],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [1.645981194, 0.5020767279, 0.1937168103],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Vanadium (Z=23) ──────────────────────────────────
        23 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1130.762517, 205.9698041, 55.74346711],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [83.78385011, 19.46956493, 6.332106784],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [83.78385011, 19.46956493, 6.332106784],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [6.141151276, 1.873246881, 0.7227568825],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [6.141151276, 1.873246881, 0.7227568825],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.7122640246, 0.2628702203, 0.1160862609],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.7122640246, 0.2628702203, 0.1160862609],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [2.964817927, 0.9043639676, 0.3489317337],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Chromium (Z=24) ──────────────────────────────────
        24 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1232.32045, 224.4687082, 60.74999251],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [92.77462423, 21.55882749, 7.01159981],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [92.77462423, 21.55882749, 7.01159981],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [6.899488096, 2.104563782, 0.8120061343],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [6.899488096, 2.104563782, 0.8120061343],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.7547780537, 0.2785605708, 0.1230152851],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.7547780537, 0.2785605708, 0.1230152851],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [4.241479241, 1.29378636, 0.4991829993],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Manganese (Z=25) ─────────────────────────────────
        25 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1337.153266, 243.5641365, 65.91796062],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [102.0220021, 23.70771923, 7.710486098],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [102.0220021, 23.70771923, 7.710486098],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [7.701960922, 2.349343572, 0.9064497869],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [7.701960922, 2.349343572, 0.9064497869],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.6709822861, 0.2476346626, 0.1093580779],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.6709822861, 0.2476346626, 0.1093580779],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [5.426950461, 1.655392868, 0.6387020316],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Iron (Z=26) ──────────────────────────────────────
        26 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1447.400411, 263.6457916, 71.35284019],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [111.9194891, 26.00768236, 8.45850549],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [111.9194891, 26.00768236, 8.45850549],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [8.548569754, 2.60758625, 1.00608784],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [8.548569754, 2.60758625, 1.00608784],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.5921156814, 0.2185279254, 0.0965042359],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.5921156814, 0.2185279254, 0.0965042359],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [6.411803475, 1.955804428, 0.7546101508],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Cobalt (Z=27) ────────────────────────────────────
        27 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1560.83467, 284.3079835, 76.94483567],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [122.2751047, 28.41410473, 9.241148731],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [122.2751047, 28.41410473, 9.241148731],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [9.43931459, 2.879291816, 1.110920295],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [9.43931459, 2.879291816, 1.110920295],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.5921156814, 0.2185279254, 0.0965042359],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.5921156814, 0.2185279254, 0.0965042359],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [7.664527389, 2.337925151, 0.9020442052],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Nickel (Z=28) ────────────────────────────────────
        28 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1679.771028, 305.9723896, 82.80806943],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [132.8588899, 30.87354878, 10.04103627],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [132.8588899, 30.87354878, 10.04103627],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [10.33074335, 3.151206003, 1.215833241],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [10.33074335, 3.151206003, 1.215833241],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.6309328384, 0.2328538976, 0.1028307363],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.6309328384, 0.2328538976, 0.1028307363],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [8.627722755, 2.631730438, 1.015403419],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Copper (Z=29) ────────────────────────────────────
        29 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1801.80673, 328.201345, 88.82409228],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [144.1212184, 33.49067173, 10.89220588],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [144.1212184, 33.49067173, 10.89220588],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [11.30775402, 3.449225397, 1.330818388],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [11.30775402, 3.449225397, 1.330818388],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.6309328384, 0.2328538976, 0.1028307363],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.6309328384, 0.2328538976, 0.1028307363],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [9.64791193, 2.942920654, 1.135470278],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Zinc (Z=30) ──────────────────────────────────────
        30 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [1929.432301, 351.4485021, 95.11568021],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [155.8416755, 36.21425391, 11.77799934],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [155.8416755, 36.21425391, 11.77799934],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [12.28152744, 3.746257327, 1.445422541],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [12.28152744, 3.746257327, 1.445422541],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.8897138854, 0.328360379, 0.1450074055],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.8897138854, 0.328360379, 0.1450074055],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [10.94737077, 3.339297018, 1.288404602],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Gallium (Z=31) ───────────────────────────────────
        31 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [2061.424532, 375.4910517, 101.6225324],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [167.761868, 38.98425028, 12.67888813],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [167.761868, 38.98425028, 12.67888813],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.7985243736, 0.2947057141, 0.1301451506],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.7985243736, 0.2947057141, 0.1301451506],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [12.6150552, 3.847993927, 1.484675684],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [12.6150552, 3.847993927, 1.484675684],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [12.6150552, 3.847993927, 1.484675684],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Germanium (Z=32) ─────────────────────────────────
        32 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [2196.384229, 400.0741292, 108.2756726],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [180.389038, 41.91853304, 13.63320795],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [180.389038, 41.91853304, 13.63320795],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.98583256, 0.363834215, 0.1606730254],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.98583256, 0.363834215, 0.1606730254],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [14.19665619, 4.33043264, 1.670815538],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [14.19665619, 4.33043264, 1.670815538],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [14.19665619, 4.33043264, 1.670815538],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Arsenic (Z=33) ───────────────────────────────────
        33 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [2337.065673, 425.6994298, 115.210879],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [193.1970535, 44.8948404, 14.60119548],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [193.1970535, 44.8948404, 14.60119548],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.107681464, 0.4088041239, 0.1805322114],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.107681464, 0.4088041239, 0.1805322114],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [15.87163584, 4.841354819, 1.867945198],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [15.87163584, 4.841354819, 1.867945198],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [15.87163584, 4.841354819, 1.867945198],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Selenium (Z=34) ──────────────────────────────────
        34 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [2480.626814, 451.8492708, 122.2880464],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [206.157878, 47.90665727, 15.5807318],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [206.157878, 47.90665727, 15.5807318],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.214644297, 0.4482801363, 0.1979652346],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.214644297, 0.4482801363, 0.1979652346],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [17.63999414, 5.380760465, 2.076064666],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [17.63999414, 5.380760465, 2.076064666],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [17.63999414, 5.380760465, 2.076064666],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Bromine (Z=35) ───────────────────────────────────
        35 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [2629.997471, 479.0573224, 129.651607],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [219.8350255, 51.08493222, 16.61440546],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [219.8350255, 51.08493222, 16.61440546],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.396037488, 0.5152256318, 0.2275290713],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.396037488, 0.5152256318, 0.2275290713],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [19.50173109, 5.948649577, 2.29517394],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [19.50173109, 5.948649577, 2.29517394],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [19.50173109, 5.948649577, 2.29517394],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Krypton (Z=36) ───────────────────────────────────
        36 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [2782.160055, 506.773927, 137.1528019],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [233.9514118, 54.36527681, 17.68127533],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [233.9514118, 54.36527681, 17.68127533],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [1.590049336, 0.5868282053, 0.2591495227],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [1.590049336, 0.5868282053, 0.2591495227],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [21.45684671, 6.545022156, 2.525273021],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [21.45684671, 6.545022156, 2.525273021],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [21.45684671, 6.545022156, 2.525273021],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Rubidium (Z=37) ──────────────────────────────────
        37 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [2938.601529, 535.2699368, 144.8649342],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [248.5070369, 57.74769105, 18.78134142],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [248.5070369, 57.74769105, 18.78134142],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.24779682, 0.8295783935, 0.3663505653],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.24779682, 0.8295783935, 0.3663505653],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4869939919, 0.2622161565, 0.1158254875],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4869939919, 0.2622161565, 0.1158254875],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [23.50534097, 7.169878201, 2.766361909],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [23.50534097, 7.169878201, 2.766361909],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [23.50534097, 7.169878201, 2.766361909],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Strontium (Z=38) ─────────────────────────────────
        38 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [3100.983951, 564.8480978, 152.8699389],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [263.5019007, 61.23217493, 19.91460372],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [263.5019007, 61.23217493, 19.91460372],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.461032403, 0.9082757342, 0.4011041407],
                coefficients: [-0.3088441214, 0.01960641165, 1.131034442],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.461032403, 0.9082757342, 0.4011041407],
                coefficients: [-0.12154686, 0.5715227604, 0.5498949471],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4370804803, 0.2353408164, 0.1039541771],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4370804803, 0.2353408164, 0.1039541771],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [25.57886692, 7.802369707, 3.010396794],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [25.57886692, 7.802369707, 3.010396794],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [25.57886692, 7.802369707, 3.010396794],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
        ],

        // ── Yttrium (Z=39) ───────────────────────────────────
        39 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [3266.026869, 594.9108712, 161.0060986],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [277.9377244, 64.58674989, 21.00561561],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [277.9377244, 64.58674989, 21.00561561],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.527274884, 0.9841077401, 0.4332066499],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.527274884, 0.9841077401, 0.4332066499],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4370804803, 0.2353408164, 0.1039541771],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4370804803, 0.2353408164, 0.1039541771],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [28.96238417, 8.834450311, 3.408605577],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [28.96238417, 8.834450311, 3.408605577],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [28.96238417, 8.834450311, 3.408605577],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [0.4576323918, 0.1781996813, 0.07844393842],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Zirconium (Z=40) ─────────────────────────────────
        40 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [3435.348677, 625.7530498, 169.3531958],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [293.7830292, 68.26885797, 22.20315144],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [293.7830292, 68.26885797, 22.20315144],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [2.827607815, 1.101055827, 0.4846874856],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [2.827607815, 1.101055827, 0.4846874856],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4869939919, 0.2622161565, 0.1158254875],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4869939919, 0.2622161565, 0.1158254875],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [30.73293103, 9.374523538, 3.616982618],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [30.73293103, 9.374523538, 3.616982618],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [30.73293103, 9.374523538, 3.616982618],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [0.8878301887, 0.3457164736, 0.1521852428],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Niobium (Z=41) ───────────────────────────────────
        41 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [3610.742864, 657.7013201, 177.9996445],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [310.0675728, 72.05303569, 23.43388348],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [310.0675728, 72.05303569, 23.43388348],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [3.14479843, 1.224568208, 0.5390579399],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [3.14479843, 1.224568208, 0.5390579399],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4869939919, 0.2622161565, 0.1158254875],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4869939919, 0.2622161565, 0.1158254875],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [33.01997858, 10.07214594, 3.886147028],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [33.01997858, 10.07214594, 3.886147028],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [33.01997858, 10.07214594, 3.886147028],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [1.344878866, 0.5236888594, 0.2305291251],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Molybdenum (Z=42) ────────────────────────────────
        42 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [3788.666115, 690.1102623, 186.7707691],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [326.4309567, 75.8555342, 24.67057401],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [326.4309567, 75.8555342, 24.67057401],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [3.496895188, 1.361672861, 0.5994117456],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [3.496895188, 1.361672861, 0.5994117456],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.5129625081, 0.276198597, 0.1220017773],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.5129625081, 0.276198597, 0.1220017773],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [35.46948129, 10.81932234, 4.174430912],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [35.46948129, 10.81932234, 4.174430912],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [35.46948129, 10.81932234, 4.174430912],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [1.702112315, 0.6627937127, 0.291763424],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Technetium (Z=43) ────────────────────────────────
        43 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [3970.868257, 723.2986098, 195.7528311],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [343.5846323, 79.84167952, 25.96699219],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [343.5846323, 79.84167952, 25.96699219],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [3.829752708, 1.491285854, 0.656467704],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [3.829752708, 1.491285854, 0.656467704],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4616999826, 0.2485968963, 0.1098096207],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4616999826, 0.2485968963, 0.1098096207],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [38.08991983, 11.61863962, 4.482832367],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [38.08991983, 11.61863962, 4.482832367],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [38.08991983, 11.61863962, 4.482832367],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [2.101373228, 0.8182638428, 0.360201758],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Ruthenium (Z=44) ─────────────────────────────────
        44 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [4159.27421, 757.6169894, 205.0407239],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [360.7986561, 83.84184843, 27.26797127],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [360.7986561, 83.84184843, 27.26797127],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [4.197516371, 1.634491118, 0.7195070139],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [4.197516371, 1.634491118, 0.7195070139],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4131354848, 0.2224479167, 0.09825915662],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4131354848, 0.2224479167, 0.09825915662],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [40.71751678, 12.42014044, 4.792076302],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [40.71751678, 12.42014044, 4.792076302],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [40.71751678, 12.42014044, 4.792076302],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [2.390895761, 0.9310024167, 0.4098295558],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Rhodium (Z=45) ───────────────────────────────────
        45 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [4350.077794, 792.3721005, 214.4468133],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [378.4334264, 87.93978981, 28.60074899],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [378.4334264, 87.93978981, 28.60074899],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [4.540857408, 1.768186338, 0.7783599789],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [4.540857408, 1.768186338, 0.7783599789],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4131354848, 0.2224479167, 0.09825915662],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4131354848, 0.2224479167, 0.09825915662],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [43.52179455, 13.27553454, 5.122113939],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [43.52179455, 13.27553454, 5.122113939],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [43.52179455, 13.27553454, 5.122113939],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [2.779066094, 1.082153932, 0.476366825],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Palladium (Z=46) ─────────────────────────────────
        46 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [4545.160269, 827.9066168, 224.0638402],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [396.4889433, 92.13550365, 29.96532535],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [396.4889433, 92.13550365, 29.96532535],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [4.919104589, 1.91547383, 0.8431962954],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [4.919104589, 1.91547383, 0.8431962954],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4370804803, 0.2353408164, 0.1039541771],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4370804803, 0.2353408164, 0.1039541771],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [46.41945097, 14.15941211, 5.463141383],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [46.41945097, 14.15941211, 5.463141383],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [46.41945097, 14.15941211, 5.463141383],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [3.025977448, 1.178299934, 0.5186905316],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Silver (Z=47) ────────────────────────────────────
        47 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [4744.521634, 864.2205383, 233.8918045],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [414.9652069, 96.42898995, 31.36170035],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [414.9652069, 96.42898995, 31.36170035],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [5.29023045, 2.059988316, 0.9068119281],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [5.29023045, 2.059988316, 0.9068119281],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.4370804803, 0.2353408164, 0.1039541771],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.4370804803, 0.2353408164, 0.1039541771],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [49.41048605, 15.07177314, 5.815158634],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [49.41048605, 15.07177314, 5.815158634],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [49.41048605, 15.07177314, 5.815158634],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [3.283395668, 1.278537254, 0.5628152469],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Cadmium (Z=48) ───────────────────────────────────
        48 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [4950.261905, 901.6963856, 244.0342313],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [433.4469385, 100.7237469, 32.75848861],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [433.4469385, 100.7237469, 32.75848861],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [5.674851796, 2.209757875, 0.9727408566],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [5.674851796, 2.209757875, 0.9727408566],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.5949150981, 0.320325, 0.1414931855],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.5949150981, 0.320325, 0.1414931855],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [52.59279235, 16.042478, 6.189686744],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [52.59279235, 16.042478, 6.189686744],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [52.59279235, 16.042478, 6.189686744],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [3.642963976, 1.41855129, 0.62444977],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Indium (Z=49) ────────────────────────────────────
        49 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [5158.224714, 939.5770707, 254.2862231],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [452.3313223, 105.1120716, 34.18570799],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [452.3313223, 105.1120716, 34.18570799],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.5669230612, 0.3052530187, 0.1348356264],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.5669230612, 0.3052530187, 0.1348356264],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [55.97539769, 17.07428044, 6.587788204],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [55.97539769, 17.07428044, 6.587788204],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [55.97539769, 17.07428044, 6.587788204],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [5.04854918, 1.965878882, 0.8653847237],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [5.04854918, 1.965878882, 0.8653847237],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [5.04854918, 1.965878882, 0.8653847237],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Tin (Z=50) ───────────────────────────────────────
        50 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [5370.466413, 978.2371611, 264.7491522],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [472.0515322, 109.6946243, 35.67609636],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [472.0515322, 109.6946243, 35.67609636],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.623581642, 0.3357601616, 0.1483111678],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.623581642, 0.3357601616, 0.1483111678],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [59.15141188, 18.043066, 6.96157579],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [59.15141188, 18.043066, 6.96157579],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [59.15141188, 18.043066, 6.96157579],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [5.583138529, 2.174045204, 0.9570200509],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [5.583138529, 2.174045204, 0.9570200509],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [5.583138529, 2.174045204, 0.9570200509],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Antimony (Z=51) ──────────────────────────────────
        51 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [5586.987002, 1017.676657, 275.4230189],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [492.1924888, 114.3749494, 37.19828336],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [492.1924888, 114.3749494, 37.19828336],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.6529226928, 0.3515585034, 0.1552895732],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.6529226928, 0.3515585034, 0.1552895732],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [62.52179775, 19.07114112, 7.358239131],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [62.52179775, 19.07114112, 7.358239131],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [62.52179775, 19.07114112, 7.358239131],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [6.120693149, 2.383366187, 1.049163663],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [6.120693149, 2.383366187, 1.049163663],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [6.120693149, 2.383366187, 1.049163663],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Tellurium (Z=52) ─────────────────────────────────
        52 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [5810.061591, 1058.309972, 286.4199797],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [512.754192, 119.1530471, 38.752269],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [512.754192, 119.1530471, 38.752269],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.7012713483, 0.3775912653, 0.166788702],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.7012713483, 0.3775912653, 0.166788702],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [65.98556227, 20.1276997, 7.765892279],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [65.98556227, 20.1276997, 7.765892279],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [65.98556227, 20.1276997, 7.765892279],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [6.707956921, 2.612043655, 1.149828048],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [6.707956921, 2.612043655, 1.149828048],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [6.707956921, 2.612043655, 1.149828048],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Iodine (Z=53) ────────────────────────────────────
        53 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [6035.183623, 1099.316231, 297.5178737],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [533.7366418, 124.0289171, 40.33805328],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [533.7366418, 124.0289171, 40.33805328],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.7900364582, 0.4253857892, 0.1879003836],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.7900364582, 0.4253857892, 0.1879003836],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [69.54270545, 21.21274175, 8.184535234],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [69.54270545, 21.21274175, 8.184535234],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [69.54270545, 21.21274175, 8.184535234],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [7.295991196, 2.841021154, 1.250624506],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [7.295991196, 2.841021154, 1.250624506],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [7.295991196, 2.841021154, 1.250624506],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // ── Xenon (Z=54) ─────────────────────────────────────
        54 => vec![
            ShellData {
                shell_type: ShellType::S,
                exponents: [6264.584546, 1141.101895, 308.8267052],
                coefficients: [0.1543289673, 0.5353281423, 0.4446345422],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [555.1398381, 129.0025597, 41.9556362],
                coefficients: [-0.09996722919, 0.3995128261, 0.7001154689],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [555.1398381, 129.0025597, 41.9556362],
                coefficients: [0.155916275, 0.6076837186, 0.3919573931],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [0.8910101433, 0.4797538759, 0.2119157236],
                coefficients: [-0.3842642608, -0.1972567438, 1.375495512],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [0.8910101433, 0.4797538759, 0.2119157236],
                coefficients: [-0.3481691526, 0.629032369, 0.6662832743],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [73.07773504, 22.29103845, 8.600575622],
                coefficients: [-0.2277635023, 0.2175436044, 0.9166769611],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [73.07773504, 22.29103845, 8.600575622],
                coefficients: [0.004951511155, 0.5777664691, 0.4846460366],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [73.07773504, 22.29103845, 8.600575622],
                coefficients: [0.2197679508, 0.6555473627, 0.286573259],
            },
            ShellData {
                shell_type: ShellType::S,
                exponents: [7.90872828, 3.079617799, 1.355655337],
                coefficients: [-0.3306100626, 0.05761095338, 1.115578745],
            },
            ShellData {
                shell_type: ShellType::P,
                exponents: [7.90872828, 3.079617799, 1.355655337],
                coefficients: [-0.1283927634, 0.5852047641, 0.543944204],
            },
            ShellData {
                shell_type: ShellType::D,
                exponents: [7.90872828, 3.079617799, 1.355655337],
                coefficients: [0.1250662138, 0.6686785577, 0.3052468245],
            },
        ],

        // BSE's canonical STO-3G family genuinely stops at Xe (Z=54) --
        // verified 2026-07-16 by fetching the basis with no element filter:
        // exactly 54 contiguous elements, no gaps. Lanthanides, actinides,
        // and anything past Xe are deliberately NOT implemented here: honest
        // STO-3G coverage there needs relativistic corrections and/or
        // effective-core-potential treatments this minimal basis alone
        // can't provide (Phase A.8, 2026-07-16).
        _ => panic!("STO-3G not implemented for Z={}", z),
    }
}

impl Sto3g {
    /// Non-panicking capability query: does this provider have real STO-3G
    /// data for element `z`? (Phase Q1, 2026-07-16.)
    ///
    /// A pure addition alongside the existing panic-on-unsupported-Z
    /// behavior in `shells_for_element`/`build` -- deliberately NOT a
    /// `Result`-wrapping change to `build()`'s signature, which is called
    /// across dozens of sites in this crate and would repeat the
    /// call-site-cascade problem Phase Q0 found and avoided for
    /// `restricted_hartree_fock`. Callers who want to check before calling
    /// `build()` can; callers who already know their molecule is
    /// STO-3G-supported (the common case) are unaffected.
    pub fn supports_element(z: u8) -> bool {
        (1..=54).contains(&z)
    }
}

impl BasisSetProvider for Sto3g {
    fn name() -> &'static str {
        "STO-3G"
    }

    fn build(molecule: &Molecule) -> BasisSet {
        let mut functions = Vec::new();

        for atom in &molecule.atoms {
            let shells = shells_for_element(atom.atomic_number);

            for shell in shells {
                // For each Cartesian component of this shell
                for (l, m, n) in shell.shell_type.cartesian_components() {
                    let primitives: Vec<PrimitiveGaussian> = shell
                        .exponents
                        .iter()
                        .zip(shell.coefficients.iter())
                        .map(|(&alpha, &coeff)| PrimitiveGaussian {
                            alpha,
                            coeff,
                            center: atom.position,
                            l,
                            m,
                            n,
                        })
                        .collect();

                    functions.push(ContractedGaussian {
                        primitives,
                        shell_type: shell.shell_type,
                    });
                }
            }
        }

        BasisSet {
            name: "STO-3G".to_string(),
            functions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Molecule;

    #[test]
    fn test_h2_basis_count() {
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        // H2: each H has 1 s-function → 2 total
        assert_eq!(basis.n_basis(), 2);
    }

    #[test]
    fn test_water_basis_count() {
        let water = Molecule::water();
        let basis = Sto3g::build(&water);
        // O: 1s + 2s + 2px + 2py + 2pz = 5
        // H: 1s each = 2
        // Total = 7
        assert_eq!(basis.n_basis(), 7);
    }

    #[test]
    fn test_heh_plus_basis_count() {
        let heh = Molecule::heh_plus();
        let basis = Sto3g::build(&heh);
        // He: 1s = 1, H: 1s = 1 → 2
        assert_eq!(basis.n_basis(), 2);
    }

    #[test]
    fn test_primitives_per_contraction() {
        let h2 = Molecule::h2();
        let basis = Sto3g::build(&h2);
        // STO-3G = 3 primitives per contracted function
        for func in &basis.functions {
            assert_eq!(func.primitives.len(), 3);
        }
    }

    #[test]
    fn test_h_exponents_match_bse() {
        // BSE reference for H STO-3G:
        // α = 3.42525091, 0.62391373, 0.16885540
        let h = Molecule::new(vec![crate::molecule::Atom::new(1, 0.0, 0.0, 0.0)]);
        let basis = Sto3g::build(&h);
        let alphas: Vec<f64> = basis.functions[0]
            .primitives
            .iter()
            .map(|p| p.alpha)
            .collect();
        assert!((alphas[0] - 3.425_250_914).abs() < 1e-6);
        assert!((alphas[1] - 0.623_913_730).abs() < 1e-6);
        assert!((alphas[2] - 0.168_855_404).abs() < 1e-6);
    }

    // ── Phase A.7 (2026-07-16): Z=11-20, 31-36 element-scope extension ──

    #[test]
    fn test_chlorine_1s_exponents_match_bse() {
        // BSE reference for Cl (Z=17) 1s shell, raw JSON fetched directly
        // via curl (not WebFetch, see module doc): 601.3456136, 109.5358542,
        // 29.64467686.
        let shells = shells_for_element(17);
        assert_eq!(shells[0].shell_type, ShellType::S);
        assert!((shells[0].exponents[0] - 601.345_613_6).abs() < 1e-6);
        assert!((shells[0].exponents[1] - 109.535_854_2).abs() < 1e-6);
        assert!((shells[0].exponents[2] - 29.644_676_86).abs() < 1e-6);
    }

    #[test]
    fn test_bromine_valence_shell_exponents_match_bse() {
        // BSE reference for Br (Z=35) valence 4s4p shell (index 4 in the
        // 8-entry shell list: 1s, 2s, 2p, 3s, 3p, 4s, 4p, 3d):
        // 1.396037488, 0.5152256318, 0.2275290713.
        let shells = shells_for_element(35);
        let s4 = &shells[3]; // 4s
        assert_eq!(s4.shell_type, ShellType::S);
        assert!((s4.exponents[0] - 1.396_037_488).abs() < 1e-6);
        assert!((s4.exponents[1] - 0.515_225_631_8).abs() < 1e-6);
        assert!((s4.exponents[2] - 0.227_529_071_3).abs() < 1e-6);
    }

    #[test]
    fn test_period4_p_block_includes_spd_shell_with_matching_exponents() {
        // Ga-Kr's outermost shell must be S+P+D, all three sharing the same
        // exponent triple (the combined 3d+4s+4p block STO-3G uses for
        // these elements) -- structural check, not just numeric.
        for z in [31u8, 32, 33, 34, 35, 36] {
            let shells = shells_for_element(z);
            assert_eq!(
                shells.len(),
                8,
                "Z={z} should have 8 shells (1s,2s,2p,3s,3p,4s,4p,3d)"
            );
            let last_three = &shells[5..8];
            assert_eq!(last_three[0].shell_type, ShellType::S);
            assert_eq!(last_three[1].shell_type, ShellType::P);
            assert_eq!(last_three[2].shell_type, ShellType::D);
            assert_eq!(last_three[0].exponents, last_three[1].exponents);
            assert_eq!(last_three[1].exponents, last_three[2].exponents);
        }
    }

    // ── Phase A.8 (2026-07-16): complete H-Xe (Z=1-54) coverage ─────────

    #[test]
    fn test_all_54_elements_generate_without_panicking() {
        for z in 1u8..=54 {
            let shells = shells_for_element(z);
            assert!(!shells.is_empty(), "Z={z} produced no shells");
        }
    }

    #[test]
    #[should_panic(expected = "STO-3G not implemented for Z=55")]
    fn test_cesium_still_panics() {
        // Z=55 (Cs) is the real, verified boundary: BSE's canonical STO-3G
        // family genuinely stops at Xe (Z=54), confirmed by fetching the
        // basis with no element filter -- not an arbitrary cutoff.
        shells_for_element(55);
    }

    #[test]
    fn test_scandium_zinc_block_has_standalone_d_shell() {
        // Sc-Zn's last shell is a LONE D shell (BSE angular_momentum=[2]),
        // distinct from Ga-Kr's combined SPD block -- structural check that
        // the generic per-shell generator handles this shape correctly.
        for z in 21u8..=30 {
            let shells = shells_for_element(z);
            assert_eq!(shells.len(), 8, "Z={z} should have 8 shells");
            assert_eq!(shells[7].shell_type, ShellType::D);
            // The D shell's own exponents must differ from the preceding
            // 4s4p shell's -- it is NOT part of a combined SPD block.
            assert_ne!(shells[7].exponents, shells[6].exponents);
        }
    }

    #[test]
    fn test_scandium_d_shell_exponent_matches_bse() {
        // BSE reference for Sc (Z=21) standalone 3d shell: 0.5517000679.
        let shells = shells_for_element(21);
        assert_eq!(shells[7].shell_type, ShellType::D);
        assert!((shells[7].exponents[0] - 0.551_700_067_9).abs() < 1e-6);
    }

    #[test]
    fn test_iron_d_shell_exponent_matches_bse() {
        // BSE reference for Fe (Z=26) standalone 3d shell: 6.411803475.
        let shells = shells_for_element(26);
        assert_eq!(shells.len(), 8);
        assert_eq!(shells[7].shell_type, ShellType::D);
        assert!((shells[7].exponents[0] - 6.411_803_475).abs() < 1e-6);
    }

    #[test]
    fn test_palladium_has_both_combined_and_standalone_d_shells() {
        // BSE reference for Pd (Z=46): 11 flattened ShellData entries
        // (1s,2s,2p,3s,3p,4s,4p,4d/5s/5p-combined-SPD,4d-standalone), with
        // the combined-block D at exponents[0]=46.41945097 and the trailing
        // standalone D at exponents[0]=3.025977448 -- two DIFFERENT D
        // shells, not a duplicate.
        let shells = shells_for_element(46);
        assert_eq!(shells.len(), 11, "Pd should have 11 flattened entries");
        assert_eq!(shells[9].shell_type, ShellType::D);
        assert!((shells[9].exponents[0] - 46.419_450_97).abs() < 1e-5);
        assert_eq!(shells[10].shell_type, ShellType::D);
        assert!((shells[10].exponents[0] - 3.025_977_448).abs() < 1e-6);
        assert_ne!(shells[9].exponents, shells[10].exponents);
    }

    #[test]
    fn test_xenon_has_two_distinct_spd_blocks() {
        // BSE reference for Xe (Z=54): two combined SPD blocks with
        // different exponents (73.07773504 and 7.90872828) -- the In-Xe
        // shape, distinct from every other block's shell layout.
        let shells = shells_for_element(54);
        assert_eq!(shells.len(), 11, "Xe should have 11 flattened entries");
        assert_eq!(shells[7].shell_type, ShellType::D);
        assert!((shells[7].exponents[0] - 73.077_735_04).abs() < 1e-5);
        assert_eq!(shells[10].shell_type, ShellType::D);
        assert!((shells[10].exponents[0] - 7.908_728_28).abs() < 1e-5);
        assert_ne!(shells[7].exponents, shells[10].exponents);
    }

    #[test]
    fn test_period3_main_group_has_no_d_shell() {
        // Na-Ar (Z=11-18) are 5-shell s/p-only elements; K/Ca (Z=19-20) add
        // a 4s4p shell (7 total) but still no d-block.
        for z in 11u8..=18 {
            let shells = shells_for_element(z);
            assert_eq!(shells.len(), 5, "Z={z} should have 5 shells");
            assert!(shells.iter().all(|s| s.shell_type != ShellType::D));
        }
        for z in [19u8, 20] {
            let shells = shells_for_element(z);
            assert_eq!(shells.len(), 7, "Z={z} should have 7 shells");
            assert!(shells.iter().all(|s| s.shell_type != ShellType::D));
        }
    }

    #[test]
    fn test_sulfur_basis_builds_without_panicking() {
        // End-to-end: build a real basis set for a sulfur atom through the
        // public API, not just the private shell table.
        let s_atom = Molecule::new(vec![crate::molecule::Atom::new(16, 0.0, 0.0, 0.0)]);
        let basis = Sto3g::build(&s_atom);
        // 1s + 2s + 2p(x3) + 3s + 3p(x3) = 9 contracted functions.
        assert_eq!(basis.n_basis(), 9);
    }

    #[test]
    fn test_bromine_basis_includes_cartesian_d_functions() {
        let br_atom = Molecule::new(vec![crate::molecule::Atom::new(35, 0.0, 0.0, 0.0)]);
        let basis = Sto3g::build(&br_atom);
        let d_count = basis
            .functions
            .iter()
            .filter(|f| f.shell_type == ShellType::D)
            .count();
        // ShellType::D expands to 6 Cartesian components (see module doc).
        assert_eq!(d_count, 6);
    }

    #[test]
    fn test_supports_element_agrees_with_shells_for_element_boundary() {
        // Phase Q1 (2026-07-16): the query must agree with what
        // shells_for_element actually does -- sampled across the real
        // boundary (Xe=54 supported, Cs=55 not) plus interior points.
        for z in [1u8, 26, 54] {
            assert!(Sto3g::supports_element(z), "Z={z} should be supported");
            // Must not panic.
            let _ = shells_for_element(z);
        }
        for z in [0u8, 55, 118] {
            assert!(!Sto3g::supports_element(z), "Z={z} should not be supported");
        }
    }
}
