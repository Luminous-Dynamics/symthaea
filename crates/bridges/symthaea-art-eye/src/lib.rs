// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-art-eye
//!
//! The artist's eye: lets Symthaea *see* her own artwork.
//!
//! Until 2026-07-10 the art stack (atelier/canvas/aesthetic) and the vision
//! stack never exchanged a pixel — no SVG rasterizer existed anywhere in the
//! monorepo, so every critique of generated art was computed from the scene
//! graph or (worse) from the generating cognitive snapshot. This crate is the
//! missing link (VISUAL_ART_IMPROVEMENT_PLAN_2026-07-10.md Phase 1):
//!
//! ```text
//! SVG string ──rasterize──▶ RGBA pixels ──percept──▶ RenderPercept
//!                                          │
//! SceneNode (intent) ──────────────────────┴──▶ critic::PerceptualInput
//! ```
//!
//! `PerceptualInput` is `symthaea_atelier::critic`'s input type, documented
//! since its creation as "can be constructed from any perception source" —
//! this crate is the first source that actually measures rendered pixels.
//!
//! Honesty notes:
//! - Rasterization is static: CSS `@keyframes` animation in canvas SVG is
//!   ignored (we perceive one frame, the way a screenshot would).
//! - `creative_surprise` is NOT computed here — without a perceptual memory
//!   (vision-manifold corpus integration, a planned follow-up) any "surprise"
//!   would be fabricated. Callers get a neutral 0.5 and should replace it
//!   when they have a real memory-backed signal.
//! - `perceptual_coherence` IS real: cosine similarity between the hue
//!   distribution the scene graph *intends* and the hue distribution the
//!   rendered pixels actually *show* (intent → percept fidelity).

#![deny(unsafe_code)]

use resvg::{tiny_skia, usvg};
use symthaea_atelier::critic::PerceptualInput;
use symthaea_canvas::SceneNode;
use symthaea_canvas::scene_features::{HUE_BINS, extract_scene_features};

/// Errors from the rasterization pipeline.
#[derive(Debug)]
pub enum ArtEyeError {
    /// SVG failed to parse.
    SvgParse(String),
    /// Pixmap allocation failed (zero/huge dimensions).
    PixmapAlloc { width: u32, height: u32 },
}

impl std::fmt::Display for ArtEyeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SvgParse(e) => write!(f, "SVG parse failed: {e}"),
            Self::PixmapAlloc { width, height } => {
                write!(f, "pixmap allocation failed ({width}x{height})")
            }
        }
    }
}

impl std::error::Error for ArtEyeError {}

/// A rasterized artwork: straight (unpremultiplied) RGBA8 pixels.
#[derive(Debug, Clone)]
pub struct Raster {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA8, `width * height * 4` bytes, alpha unpremultiplied.
    pub rgba: Vec<u8>,
}

/// Rasterize an SVG string, scaling the longest side to `max_dim` pixels
/// (aspect preserved). 256 is plenty for perceptual feature extraction and
/// keeps per-candidate cost in the low milliseconds.
pub fn rasterize_svg(svg: &str, max_dim: u32) -> Result<Raster, ArtEyeError> {
    let options = usvg::Options::default();
    let tree =
        usvg::Tree::from_str(svg, &options).map_err(|e| ArtEyeError::SvgParse(e.to_string()))?;

    let size = tree.size();
    let (src_w, src_h) = (size.width().max(1.0), size.height().max(1.0));
    let scale = (max_dim.max(1) as f32) / src_w.max(src_h);
    let width = ((src_w * scale).round() as u32).max(1);
    let height = ((src_h * scale).round() as u32).max(1);

    let mut pixmap =
        tiny_skia::Pixmap::new(width, height).ok_or(ArtEyeError::PixmapAlloc { width, height })?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );

    // Pixmap stores premultiplied alpha; demultiply so downstream feature
    // math sees true colors.
    let rgba = pixmap
        .pixels()
        .iter()
        .flat_map(|p| {
            let d = p.demultiply();
            [d.red(), d.green(), d.blue(), d.alpha()]
        })
        .collect();

    Ok(Raster {
        width,
        height,
        rgba,
    })
}

/// Rasterize an SVG into an EXACT `width`×`height` frame (scaled to fit,
/// centered, black letterbox) — the shape a fixed-dimension vision pipeline
/// (e.g. vision-manifold's `PatchHdcEncoder`) expects.
pub fn rasterize_svg_exact(svg: &str, width: u32, height: u32) -> Result<Raster, ArtEyeError> {
    let options = usvg::Options::default();
    let tree =
        usvg::Tree::from_str(svg, &options).map_err(|e| ArtEyeError::SvgParse(e.to_string()))?;

    let (width, height) = (width.max(1), height.max(1));
    let size = tree.size();
    let (src_w, src_h) = (size.width().max(1.0), size.height().max(1.0));
    let scale = (width as f32 / src_w).min(height as f32 / src_h);
    let offset_x = (width as f32 - src_w * scale) / 2.0;
    let offset_y = (height as f32 - src_h * scale) / 2.0;

    let mut pixmap =
        tiny_skia::Pixmap::new(width, height).ok_or(ArtEyeError::PixmapAlloc { width, height })?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale, scale).post_translate(offset_x, offset_y),
        &mut pixmap.as_mut(),
    );

    let rgba = pixmap
        .pixels()
        .iter()
        .flat_map(|p| {
            let d = p.demultiply();
            [d.red(), d.green(), d.blue(), d.alpha()]
        })
        .collect();

    Ok(Raster {
        width,
        height,
        rgba,
    })
}

/// Convert a raster to a packed pixel buffer with the given channel count:
/// `1` = luma (Rec. 709), `3` = RGB. Any other count returns `None`.
pub fn to_channels(raster: &Raster, channels: u8) -> Option<Vec<u8>> {
    let n = (raster.width * raster.height) as usize;
    match channels {
        1 => Some(
            (0..n)
                .map(|i| {
                    let r = raster.rgba[i * 4] as f32;
                    let g = raster.rgba[i * 4 + 1] as f32;
                    let b = raster.rgba[i * 4 + 2] as f32;
                    (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as u8
                })
                .collect(),
        ),
        3 => Some(
            (0..n)
                .flat_map(|i| {
                    [
                        raster.rgba[i * 4],
                        raster.rgba[i * 4 + 1],
                        raster.rgba[i * 4 + 2],
                    ]
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Scramble a packed pixel buffer in place: Fisher–Yates over pixel
/// positions (channel groups stay intact), seeded deterministically.
///
/// This is the CONTROL condition for the observer-ΔΨ A/B: the scrambled
/// frame has exactly the artwork's color/luminance histogram but zero
/// composition, so a Δψ difference between the two arms isolates response
/// to *structure*. Uses an inline splitmix64 PRNG (no dependency, fully
/// deterministic for a given seed).
///
/// `channels` must be 1 or 3 and `frame.len()` a multiple of it; violations
/// leave the frame untouched (a wrong-shape control would be a silent lie).
pub fn scramble_pixels(frame: &mut [u8], channels: u8, seed: u64) {
    let ch = channels as usize;
    if !(ch == 1 || ch == 3) || frame.is_empty() || frame.len() % ch != 0 {
        return;
    }
    let n = frame.len() / ch;

    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };

    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        if i != j {
            for c in 0..ch {
                frame.swap(i * ch + c, j * ch + c);
            }
        }
    }
}

/// Perceptual features measured from rendered pixels.
#[derive(Debug, Clone)]
pub struct RenderPercept {
    /// Hue histogram over [`HUE_BINS`] bins (chromatic pixels only,
    /// saturation ≥ 0.1), normalized to sum to 1 (all-zero if achromatic).
    pub hue_histogram: [f32; HUE_BINS],
    /// Fraction of pixels that are chromatic.
    pub chromatic_fraction: f32,
    /// Mean luma in [0, 1].
    pub mean_brightness: f32,
    /// Luma standard deviation in [0, ~0.5] (contrast).
    pub brightness_contrast: f32,
    /// Fraction of pixels whose Sobel gradient magnitude exceeds a fixed
    /// threshold — a cheap busy-ness / detail measure.
    pub edge_density: f32,
    /// Mean chroma (max(r,g,b) − min(r,g,b)) in [0, 1].
    pub colorfulness: f32,
    /// 4×4 grid of mean luma per cell, row-major — coarse spatial layout.
    pub luma_grid: [f32; 16],
}

impl RenderPercept {
    /// Flatten into the feature vector shape `critic::PerceptualInput`
    /// expects: 4 global stats + 12 hue bins + 16 grid cells = 32 features.
    pub fn to_visual_features(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(4 + HUE_BINS + 16);
        features.push(self.mean_brightness);
        features.push(self.brightness_contrast);
        features.push(self.edge_density);
        features.push(self.colorfulness);
        features.extend_from_slice(&self.hue_histogram);
        features.extend_from_slice(&self.luma_grid);
        features
    }
}

/// Extract perceptual features from straight-alpha RGBA8 pixels.
pub fn percept(raster: &Raster) -> RenderPercept {
    let (w, h) = (raster.width as usize, raster.height as usize);
    let n = (w * h).max(1);

    let mut hue_histogram = [0.0f32; HUE_BINS];
    let mut chromatic = 0.0f32;
    let mut luma_sum = 0.0f32;
    let mut luma_sq_sum = 0.0f32;
    let mut chroma_sum = 0.0f32;
    let mut luma = vec![0.0f32; n];
    let mut grid_sum = [0.0f32; 16];
    let mut grid_count = [0.0f32; 16];

    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let r = raster.rgba[i] as f32 / 255.0;
            let g = raster.rgba[i + 1] as f32 / 255.0;
            let b = raster.rgba[i + 2] as f32 / 255.0;

            let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            luma[y * w + x] = l;
            luma_sum += l;
            luma_sq_sum += l * l;

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let chroma = max - min;
            chroma_sum += chroma;

            // HSL-style saturation
            let lightness = (max + min) / 2.0;
            let saturation = if chroma < 1e-6 {
                0.0
            } else {
                chroma / (1.0 - (2.0 * lightness - 1.0).abs()).max(1e-6)
            };
            if saturation >= 0.1 {
                let hue = hue_of(r, g, b, max, chroma);
                let bin = ((hue / 30.0) as usize).min(HUE_BINS - 1);
                hue_histogram[bin] += 1.0;
                chromatic += 1.0;
            }

            let gx = (x * 4) / w.max(1);
            let gy = (y * 4) / h.max(1);
            let gi = gy.min(3) * 4 + gx.min(3);
            grid_sum[gi] += l;
            grid_count[gi] += 1.0;
        }
    }

    // Normalize hue histogram to a distribution
    if chromatic > 0.0 {
        for bin in &mut hue_histogram {
            *bin /= chromatic;
        }
    }

    // Sobel edge density on the luma plane
    let mut edges = 0usize;
    let mut interior = 0usize;
    const EDGE_THRESHOLD: f32 = 0.25;
    if w >= 3 && h >= 3 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let p = |dx: isize, dy: isize| -> f32 {
                    luma[((y as isize + dy) as usize) * w + ((x as isize + dx) as usize)]
                };
                let gx =
                    -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);
                let gy =
                    -p(-1, -1) - 2.0 * p(0, -1) - p(1, -1) + p(-1, 1) + 2.0 * p(0, 1) + p(1, 1);
                if (gx * gx + gy * gy).sqrt() > EDGE_THRESHOLD {
                    edges += 1;
                }
                interior += 1;
            }
        }
    }

    let mean = luma_sum / n as f32;
    let variance = (luma_sq_sum / n as f32 - mean * mean).max(0.0);
    let mut luma_grid = [0.0f32; 16];
    for i in 0..16 {
        luma_grid[i] = if grid_count[i] > 0.0 {
            grid_sum[i] / grid_count[i]
        } else {
            0.0
        };
    }

    RenderPercept {
        hue_histogram,
        chromatic_fraction: chromatic / n as f32,
        mean_brightness: mean,
        brightness_contrast: variance.sqrt(),
        edge_density: if interior > 0 {
            edges as f32 / interior as f32
        } else {
            0.0
        },
        colorfulness: chroma_sum / n as f32,
        luma_grid,
    }
}

fn hue_of(r: f32, g: f32, b: f32, max: f32, chroma: f32) -> f32 {
    if chroma < 1e-6 {
        return 0.0;
    }
    let h = if (max - r).abs() < 1e-6 {
        ((g - b) / chroma).rem_euclid(6.0)
    } else if (max - g).abs() < 1e-6 {
        (b - r) / chroma + 2.0
    } else {
        (r - g) / chroma + 4.0
    };
    (h * 60.0).rem_euclid(360.0)
}

/// Build a [`PerceptualInput`] for `symthaea_atelier::critic::SelfCritic`
/// from the intended scene graph and its rendered percept.
///
/// - `perceptual_coherence` = cosine similarity between the scene graph's
///   intended hue distribution and the rendered pixels' hue distribution —
///   real intent→percept fidelity, not a placeholder.
/// - `creative_surprise` = neutral 0.5: this crate has no perceptual memory
///   to be surprised against. Replace via
///   [`PerceptualInput::creative_surprise`] when a memory-backed signal
///   (vision-manifold corpus) is available.
pub fn perceptual_input(scene: &SceneNode, render: &RenderPercept) -> PerceptualInput {
    let intended = extract_scene_features(scene);
    let coherence = hue_cosine(&intended.hue_histogram, &render.hue_histogram);

    PerceptualInput {
        creative_surprise: 0.5,
        perceptual_coherence: coherence,
        visual_features: render.to_visual_features(),
    }
}

/// See her art: rasterize an SVG and build the critic input in one call.
pub fn see(scene: &SceneNode, svg: &str, max_dim: u32) -> Result<PerceptualInput, ArtEyeError> {
    let raster = rasterize_svg(svg, max_dim)?;
    let render = percept(&raster);
    Ok(perceptual_input(scene, &render))
}

/// Cosine similarity between two hue histograms; both all-zero (two
/// achromatic pieces) counts as coherent (1.0), one-sided zero as 0.0.
fn hue_cosine(a: &[f32; HUE_BINS], b: &[f32; HUE_BINS]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    match (na > 1e-6, nb > 1e-6) {
        (true, true) => (dot / (na * nb)).clamp(0.0, 1.0),
        (false, false) => 1.0,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_atelier::{AtelierConfig, AtelierStyle, create_artwork};
    use symthaea_canvas::CognitiveSnapshot;

    fn rich_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5, 0.7, 0.3, 0.8, 0.4, 0.6, 0.9, 0.2],
            dopamine: 0.6,
            serotonin: 0.5,
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn rasterizes_minimal_svg() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
            <rect x="0" y="0" width="100" height="50" fill="#ff0000"/>
        </svg>"##;
        let raster = rasterize_svg(svg, 64).expect("rasterize");
        assert_eq!(raster.width, 64);
        assert_eq!(raster.height, 32);
        // Red rect → strongly chromatic percept in the red bin
        let p = percept(&raster);
        assert!(p.chromatic_fraction > 0.9, "red fill should be chromatic");
        assert!(
            p.hue_histogram[0] > 0.9,
            "hue mass should be in bin 0 (red): {:?}",
            p.hue_histogram
        );
    }

    #[test]
    fn invalid_svg_errors() {
        assert!(rasterize_svg("not svg at all", 64).is_err());
    }

    #[test]
    fn scramble_preserves_histogram_destroys_order() {
        // RGB frame with a distinctive gradient: scrambling must keep the
        // exact multiset of pixels while changing their arrangement.
        let mut frame: Vec<u8> = (0..64u32 * 3).map(|i| (i % 251) as u8).collect();
        let original = frame.clone();
        scramble_pixels(&mut frame, 3, 42);
        assert_ne!(frame, original, "scramble must change pixel order");

        let mut sorted_pixels = |buf: &[u8]| {
            let mut px: Vec<[u8; 3]> = buf.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
            px.sort_unstable();
            px
        };
        assert_eq!(
            sorted_pixels(&frame),
            sorted_pixels(&original),
            "scramble must preserve the exact pixel multiset"
        );

        // Deterministic per seed; different seeds differ.
        let mut again = original.clone();
        scramble_pixels(&mut again, 3, 42);
        assert_eq!(again, frame, "same seed must reproduce the same shuffle");
        let mut other = original.clone();
        scramble_pixels(&mut other, 3, 43);
        assert_ne!(other, frame, "different seeds should differ");

        // Wrong shape is left untouched, never mangled.
        let mut odd = vec![1u8, 2, 3, 4];
        scramble_pixels(&mut odd, 3, 7);
        assert_eq!(odd, vec![1, 2, 3, 4]);
    }

    #[test]
    fn exact_rasterization_letterboxes() {
        // 2:1 SVG into a square frame: exact dims, content centered, black
        // letterbox bands above/below.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">
            <rect x="0" y="0" width="200" height="100" fill="#ffffff"/>
        </svg>"##;
        let raster = rasterize_svg_exact(svg, 64, 64).expect("rasterize exact");
        assert_eq!((raster.width, raster.height), (64, 64));
        let luma = to_channels(&raster, 1).expect("luma");
        assert_eq!(luma.len(), 64 * 64);
        // Top row = letterbox (empty → 0), center row = white content.
        assert!(
            luma[..64].iter().all(|&v| v < 10),
            "top band should be dark"
        );
        let mid = &luma[32 * 64..33 * 64];
        assert!(mid.iter().all(|&v| v > 200), "center should be white");
        // RGB conversion has 3x the bytes.
        assert_eq!(to_channels(&raster, 3).expect("rgb").len(), 64 * 64 * 3);
        assert!(to_channels(&raster, 4).is_none());
    }

    #[test]
    fn sees_real_atelier_artwork() {
        let config = AtelierConfig {
            style: AtelierStyle::Composite,
            iteration_budget: 1,
            ..Default::default()
        };
        let artwork = create_artwork(&config, &rich_snapshot(), 42);
        let input = see(&artwork.scene, &artwork.svg, 192).expect("see artwork");
        assert_eq!(input.visual_features.len(), 4 + HUE_BINS + 16);
        assert!((0.0..=1.0).contains(&input.perceptual_coherence));
        // A real composite render must not be a blank frame.
        let p = rasterize_svg(&artwork.svg, 192)
            .map(|r| percept(&r))
            .unwrap();
        assert!(
            p.brightness_contrast > 0.0 || p.chromatic_fraction > 0.0,
            "render should contain visible content"
        );
    }

    #[test]
    fn different_palettes_produce_different_percepts() {
        // Two artworks identical except color arrangement — the pixel percept
        // must distinguish them (the whole point of seeing).
        let make_svg = |color: &str| {
            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
                <rect width="64" height="64" fill="#202020"/>
                <circle cx="20" cy="32" r="10" fill="{color}"/>
                <circle cx="44" cy="32" r="10" fill="{color}"/>
                </svg>"##
            )
        };
        let red = percept(&rasterize_svg(&make_svg("#ff2020"), 64).unwrap());
        let blue = percept(&rasterize_svg(&make_svg("#2020ff"), 64).unwrap());
        assert!(
            hue_cosine(&red.hue_histogram, &blue.hue_histogram) < 0.5,
            "red vs blue percepts should diverge"
        );
    }

    #[test]
    fn coherence_high_when_render_matches_scene() {
        // A scene whose SVG faithfully renders it should score high
        // intent→percept coherence.
        let config = AtelierConfig {
            style: AtelierStyle::ColorField,
            iteration_budget: 1,
            ..Default::default()
        };
        let artwork = create_artwork(&config, &rich_snapshot(), 7);
        let input = see(&artwork.scene, &artwork.svg, 192).expect("see");
        assert!(
            input.perceptual_coherence > 0.2,
            "faithful render should not read as incoherent: {}",
            input.perceptual_coherence
        );
    }

    #[test]
    fn critic_accepts_real_percepts() {
        use symthaea_atelier::critic::SelfCritic;
        let config = AtelierConfig::default();
        let snapshot = rich_snapshot();
        let artwork = create_artwork(&config, &snapshot, 42);
        let input = see(&artwork.scene, &artwork.svg, 192).expect("see");
        let mut critic = SelfCritic::new();
        let verdict = critic.evaluate(&input, &snapshot);
        assert!((0.0..=1.0).contains(&verdict.composite));
        assert!((0.0..=1.0).contains(&verdict.intention_coherence));
    }
}
