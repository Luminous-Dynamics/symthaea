// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mental rotation with RENDERED stimuli — the real-perception twin of
//! psych-bench's synthetic `spatial/mental_rotation.rs`.
//!
//! P3 experiment #2 of VISION_PROJECTION_REVIEW_2026-07-15.md. The synthetic
//! benchmark simulates rotation as HDC cyclic permutation (`permute(k)`) of
//! random hypervectors — it validates the algebra, not perception. This
//! experiment renders actual asymmetric polyominoes at real angles through
//! the same rasterize→encode path the live loop uses (art-eye resvg →
//! vision-manifold PatchHdcEncoder) and asks two questions:
//!
//! 1. **Similarity–disparity curve** (Shepard & Metzler 1971 signature):
//!    does encoded similarity to the upright shape decline monotonically
//!    with angular disparity? (Human RT rises ~linearly with angle; a
//!    perceptual representation should show graded, orderly disparity.)
//! 2. **2-AFC mirror discrimination**: at each angle, is the rotated TRUE
//!    shape encoded closer to the upright original than its MIRROR image
//!    at the same angle? (The classic same/different judgment.)
//!
//! Plus a scrambled-pixel control: same pixel histogram, destroyed
//! structure — similarity should collapse, proving the encoding carries
//! spatial structure rather than global statistics.
//!
//! Run:
//! ```bash
//! cargo run --release --example benchmark_mental_rotation_rendered \
//!   --features art-eye,vision-manifold
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;
use symthaea_vision_manifold::{PatchHdcEncoder, VisionConfig};

const CANVAS: u32 = 128;
const GRID: i32 = 5; // polyomino cell grid
const CELLS: usize = 6; // hexominoes: enough structure to be chiral
const N_SHAPES: usize = 8;
const ANGLES: [f32; 7] = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0];

type Cells = BTreeSet<(i32, i32)>;

fn normalize(cells: &Cells) -> Cells {
    let min_r = cells.iter().map(|c| c.0).min().unwrap_or(0);
    let min_c = cells.iter().map(|c| c.1).min().unwrap_or(0);
    cells.iter().map(|&(r, c)| (r - min_r, c - min_c)).collect()
}

fn rotate90(cells: &Cells) -> Cells {
    normalize(&cells.iter().map(|&(r, c)| (c, -r)).collect())
}

fn reflect(cells: &Cells) -> Cells {
    normalize(&cells.iter().map(|&(r, c)| (r, -c)).collect())
}

/// Random connected polyomino, rejected unless fully asymmetric: no
/// rotational self-symmetry (flat curves by construction) and no mirror
/// self-symmetry (2-AFC undefined).
fn gen_asymmetric_polyomino(rng: &mut StdRng) -> Cells {
    loop {
        let mut cells: Cells = BTreeSet::new();
        cells.insert((GRID / 2, GRID / 2));
        while cells.len() < CELLS {
            let &(r, c) = cells
                .iter()
                .nth(rng.gen_range(0..cells.len()))
                .expect("nonempty");
            let (dr, dc) = [(0, 1), (0, -1), (1, 0), (-1, 0)][rng.gen_range(0..4)];
            let cand = (r + dr, c + dc);
            if cand.0 >= 0 && cand.0 < GRID && cand.1 >= 0 && cand.1 < GRID {
                cells.insert(cand);
            }
        }
        let norm = normalize(&cells);
        let r90 = rotate90(&norm);
        let r180 = rotate90(&r90);
        let r270 = rotate90(&r180);
        let refl = reflect(&norm);
        if norm != r90 && norm != r180 && norm != r270 && norm != refl {
            return norm;
        }
    }
}

/// Render the polyomino as SVG, rotated `angle` degrees about the canvas
/// center, optionally mirrored. White shape on black — maximal contrast for
/// the encoder's luminance features.
fn polyomino_svg(cells: &Cells, angle: f32, mirrored: bool) -> String {
    let cell_px = 16.0_f32;
    let max_r = cells.iter().map(|c| c.0).max().unwrap_or(0) as f32 + 1.0;
    let max_c = cells.iter().map(|c| c.1).max().unwrap_or(0) as f32 + 1.0;
    let off_x = (CANVAS as f32 - max_c * cell_px) / 2.0;
    let off_y = (CANVAS as f32 - max_r * cell_px) / 2.0;
    let center = CANVAS as f32 / 2.0;
    let mirror = if mirrored {
        format!("translate({} 0) scale(-1 1) ", CANVAS)
    } else {
        String::new()
    };
    let mut rects = String::new();
    for &(r, c) in cells {
        rects.push_str(&format!(
            r#"<rect x="{:.1}" y="{:.1}" width="{cell_px}" height="{cell_px}" fill="white"/>"#,
            off_x + c as f32 * cell_px,
            off_y + r as f32 * cell_px,
        ));
    }
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS}" height="{CANVAS}">
<rect width="{CANVAS}" height="{CANVAS}" fill="black"/>
<g transform="{mirror}rotate({angle} {center} {center})">{rects}</g>
</svg>"#
    )
}

/// Rasterize an SVG and encode it with a FRESH PatchHdcEncoder (bases are
/// deterministic from config.seed, and a fresh encoder zeroes the
/// motion-feature state so static stimuli compare cleanly).
fn encode_svg(svg: &str, config: &VisionConfig) -> symthaea_core::hdc::ContinuousHV {
    let raster =
        symthaea_art_eye::rasterize_svg_exact(svg, CANVAS, CANVAS).expect("rasterize failed");
    let pixels = symthaea_art_eye::to_channels(&raster, 3).expect("channel conversion failed");
    let mut encoder = PatchHdcEncoder::new(config, CANVAS, CANVAS);
    let (hv, _) = encoder.encode_frame(&pixels, CANVAS, CANVAS, 3);
    hv
}

fn scramble(pixels: &mut [u8], rng: &mut StdRng) {
    // Fisher-Yates over 3-byte pixels: identical histogram, destroyed layout.
    let n = pixels.len() / 3;
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        for k in 0..3 {
            pixels.swap(i * 3 + k, j * 3 + k);
        }
    }
}

fn main() {
    let config = VisionConfig::default();
    let mut rng = StdRng::seed_from_u64(0x5EED_2026_0716);

    println!("Mental rotation with rendered stimuli");
    println!(
        "  {N_SHAPES} asymmetric {CELLS}-cell polyominoes x {} angles, {CANVAS}x{CANVAS} raster,",
        ANGLES.len()
    );
    println!(
        "  art-eye resvg -> vision-manifold PatchHdcEncoder (dim {})\n",
        config.hdc_dim
    );

    let mut sim_by_angle: Vec<Vec<f32>> = vec![Vec::new(); ANGLES.len()];
    let mut afc_correct: Vec<u32> = vec![0; ANGLES.len()];
    let mut afc_total: Vec<u32> = vec![0; ANGLES.len()];
    let mut scramble_sims: Vec<f32> = Vec::new();

    for shape_idx in 0..N_SHAPES {
        let cells = gen_asymmetric_polyomino(&mut rng);
        let base_svg = polyomino_svg(&cells, 0.0, false);
        let base_hv = encode_svg(&base_svg, &config);

        // Scrambled-pixel control on the upright rendering.
        let raster = symthaea_art_eye::rasterize_svg_exact(&base_svg, CANVAS, CANVAS).unwrap();
        let mut scrambled = symthaea_art_eye::to_channels(&raster, 3).unwrap();
        scramble(&mut scrambled, &mut rng);
        let mut enc = PatchHdcEncoder::new(&config, CANVAS, CANVAS);
        let (scram_hv, _) = enc.encode_frame(&scrambled, CANVAS, CANVAS, 3);
        scramble_sims.push(base_hv.similarity(&scram_hv));

        for (ai, &angle) in ANGLES.iter().enumerate() {
            let rot_hv = encode_svg(&polyomino_svg(&cells, angle, false), &config);
            let ref_hv = encode_svg(&polyomino_svg(&cells, angle, true), &config);
            let s_rot = base_hv.similarity(&rot_hv);
            let s_ref = base_hv.similarity(&ref_hv);
            sim_by_angle[ai].push(s_rot);
            afc_total[ai] += 1;
            if s_rot > s_ref {
                afc_correct[ai] += 1;
            }
        }
        println!("  shape {} encoded ({} cells)", shape_idx + 1, cells.len());
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;

    println!("\nangle   mean sim(base, rotated)   2-AFC vs mirror");
    let mut means = Vec::new();
    for (ai, &angle) in ANGLES.iter().enumerate() {
        let m = mean(&sim_by_angle[ai]);
        means.push(m);
        println!(
            "{angle:5.0}°           {m:+.4}              {}/{}",
            afc_correct[ai], afc_total[ai]
        );
    }
    let scram_mean = mean(&scramble_sims);
    println!("\nscrambled-pixel control: mean sim {scram_mean:+.4} (same histogram, no structure)");

    // Monotonicity: count of adjacent angle pairs where similarity declines.
    let declines = means.windows(2).filter(|w| w[1] < w[0]).count();
    let total_afc: u32 = afc_correct.iter().skip(1).sum(); // exclude 0° (trivial)
    let total_afc_n: u32 = afc_total.iter().skip(1).sum();

    println!("\n── Findings ──");
    println!(
        "monotonic declines: {declines}/{} adjacent angle steps \
         (graded disparity curve = perceptual-representation signature)",
        means.len() - 1
    );
    println!(
        "2-AFC mirror discrimination (30°-180°): {total_afc}/{total_afc_n} \
         ({:.0}% vs 50% chance)",
        100.0 * total_afc as f32 / total_afc_n.max(1) as f32
    );
    println!(
        "structure sensitivity: sim at 180° ({:+.4}) vs scrambled control ({scram_mean:+.4}) — \
         a perceptual code should hold MORE similarity for a rigid rotation than for \
         histogram-matched noise",
        means[ANGLES.len() - 1]
    );
    println!(
        "\nInterpretation guide: the SYNTHETIC psych-bench rotation benchmark \
         (spatial/mental_rotation.rs) models rotation as permute(k) on random HVs, where \
         these properties hold by construction. Here they are measured through the real \
         rasterize->encode path; disagreement between the two is the finding."
    );
}
