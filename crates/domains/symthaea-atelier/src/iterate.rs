// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explore-then-mutate hill-climbing creative loop.
//!
//! The iteration budget is split into two phases:
//!
//! 1. **Explore** (first ~half of the budget): generate INDEPENDENT artwork
//!    variants (fresh seed per iteration) and keep the best — divergent
//!    generation, best-of-N random restart.
//! 2. **Exploit** (remaining budget): MUTATE the current best scene
//!    ([`mutate_scene`]) and keep the mutant only if it scores higher —
//!    hill-climbing on a parent scene, with mutation strength annealed
//!    down as the budget runs out.
//!
//! Together this is a microcosm of the Wallas cycle: divergent generation →
//! convergent selection → incremental refinement. The whole loop is
//! deterministic for a given `base_seed`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symthaea_canvas::color::Color;
use symthaea_canvas::scene_graph::{NodeKind, Style};
use symthaea_canvas::{CognitiveSnapshot, SceneNode};

use crate::{Artwork, AtelierConfig, generate, score_scene};

/// External per-candidate scorer, e.g. a rasterize-then-critique pipeline
/// (`symthaea-art-eye` + `critic::SelfCritic`). Receives the candidate scene
/// and the generating snapshot; returns a composite in [0, 1], or `None` to
/// skip external scoring for that candidate (e.g. rasterization failed —
/// the internal score alone then decides).
pub type ExternalScorer<'a> = dyn FnMut(&SceneNode, &CognitiveSnapshot) -> Option<f32> + 'a;

/// Blend weight of the external (perceptual) score against the internal
/// scene-graph score during the exploit phase.
pub const EXTERNAL_SCORE_WEIGHT: f32 = 0.4;

/// Create artwork with iterative refinement.
///
/// Spends roughly the first half of `config.iteration_budget` generating
/// independent variants with different random seeds (explore), then
/// hill-climbs the best one via [`mutate_scene`] for the remaining
/// iterations (exploit), keeping a mutant only if it scores strictly
/// higher. The best-scoring scene across all iterations is returned, so
/// the final score never decreases as iterations proceed.
pub fn create_iterative(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    base_seed: u64,
) -> Artwork {
    create_iterative_scored(config, snapshot, base_seed, None)
}

/// [`create_iterative`] with an optional external perceptual scorer.
///
/// The external scorer runs in the **exploit phase only** (the explore phase
/// selects on the internal scene-graph score alone): rasterizing every
/// explore candidate would multiply the budgeted cost on candidates that
/// mostly get discarded, and the exploit phase is where a perceptual
/// gradient actually steers mutation acceptance. When the scorer returns a
/// value it is blended at [`EXTERNAL_SCORE_WEIGHT`] into the selection
/// composite; the returned `Artwork.aesthetic_score` remains the internal
/// score (its dimensions stay individually meaningful — the blend is a
/// selection signal, not a replacement score).
pub fn create_iterative_scored(
    config: &AtelierConfig,
    snapshot: &CognitiveSnapshot,
    base_seed: u64,
    mut external: Option<&mut ExternalScorer<'_>>,
) -> Artwork {
    let budget = config.iteration_budget.max(1);
    // At least one explore iteration so exploit always has a parent.
    let explore = budget.div_ceil(2);

    let mut best_score = f32::NEG_INFINITY;
    let mut best_scene = None;
    let mut best_aesthetic = symthaea_aesthetic::AestheticScore::zero();
    let mut cycles = 0;

    // Phase 1: explore — independent generations (best-of-N restart).
    for i in 0..explore {
        let seed = base_seed.wrapping_add(i as u64 * 7919); // prime step
        let mut rng = StdRng::seed_from_u64(seed);
        let scene = generate(config, snapshot, &mut rng);
        let score = score_scene(&scene, snapshot);

        cycles += 1;

        if score.composite > best_score {
            best_score = score.composite;
            best_scene = Some(scene);
            best_aesthetic = score;
        }
    }

    // Phase 2: exploit — hill-climb by mutating the current best scene.
    // Separate RNG stream (splitmix-style scramble of the base seed) so the
    // mutation sequence is deterministic but decorrelated from generation.
    let mut mut_rng = StdRng::seed_from_u64(
        base_seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x2545_F491_4F6C_DD1D),
    );
    let exploit_iters = budget - explore;

    // Selection score for hill-climbing. With an external scorer this is the
    // blended internal+perceptual composite; the internal-only `best_score`
    // keeps tracking the internal composite for `Artwork.aesthetic_score`.
    let blend = |internal: f32, ext: Option<f32>| match ext {
        Some(e) => (1.0 - EXTERNAL_SCORE_WEIGHT) * internal + EXTERNAL_SCORE_WEIGHT * e,
        None => internal,
    };
    // Re-baseline the explore winner on the blended scale so exploit
    // comparisons are like-for-like (one extra external call).
    let mut selection_best = match (external.as_mut(), best_scene.as_ref()) {
        (Some(scorer), Some(scene)) if exploit_iters > 0 => {
            blend(best_score, scorer(scene, snapshot))
        }
        _ => best_score,
    };

    for i in 0..exploit_iters {
        let parent = best_scene
            .as_ref()
            .expect("explore phase runs at least once");
        // Anneal mutation strength: bold early, gentle late.
        let progress = i as f32 / exploit_iters.max(1) as f32;
        let strength = 0.6 * (1.0 - 0.5 * progress);
        let mutant = mutate_scene(
            parent,
            &mut mut_rng,
            strength,
            (config.width, config.height),
        );
        let score = score_scene(&mutant, snapshot);
        let selection = blend(
            score.composite,
            external
                .as_mut()
                .and_then(|scorer| scorer(&mutant, snapshot)),
        );

        cycles += 1;

        if selection > selection_best {
            selection_best = selection;
            best_score = score.composite;
            best_scene = Some(mutant);
            best_aesthetic = score;
        }
    }

    let scene = best_scene.unwrap_or_else(|| {
        let mut rng = StdRng::seed_from_u64(base_seed);
        generate(config, snapshot, &mut rng)
    });

    let svg = symthaea_canvas::render_svg(&scene, snapshot.consciousness_level);

    Artwork {
        scene,
        svg,
        aesthetic_score: best_aesthetic,
        style: config.style,
        generation_cycles: cycles,
    }
}

/// Mutate a scene by perturbing a random subset of its nodes.
///
/// Operators (each applied per-node with probability scaled by `strength`):
/// - **Geometry jitter**: nudge positions/sizes of circles, ellipses, lines,
///   rects, polygons, and group/path transforms by a small fraction of the
///   viewport.
/// - **Color jitter**: shift fill/stroke hue and lightness in HSL space.
/// - **Structural**: occasionally drop a leaf child (never the last one) or
///   duplicate a leaf child.
///
/// Gradient/filter definition nodes are left untouched so `url(#id)`
/// references stay valid. Deterministic given the same RNG state; `strength`
/// is clamped to `[0, 1]` and `viewport` is `(width, height)`.
pub fn mutate_scene(
    scene: &SceneNode,
    rng: &mut StdRng,
    strength: f32,
    viewport: (f32, f32),
) -> SceneNode {
    let strength = strength.clamp(0.0, 1.0);
    let mut mutant = scene.clone();
    mutate_node(&mut mutant, rng, strength, viewport);
    mutant
}

fn mutate_node(node: &mut SceneNode, rng: &mut StdRng, strength: f32, viewport: (f32, f32)) {
    // Per-node probability of perturbing each aspect.
    let p = 0.15 + 0.35 * strength;
    // Positional jitter magnitude: a small fraction of the viewport.
    let dpos = viewport.0.max(viewport.1) * 0.03 * strength;

    // Never touch defs-style nodes: renaming/perturbing them would break
    // url(#id) references elsewhere in the scene.
    let is_defs = matches!(
        node.kind,
        NodeKind::RadialGradient { .. } | NodeKind::Filter { .. }
    );

    if !is_defs {
        if rng.r#gen::<f32>() < p {
            jitter_geometry(node, rng, dpos);
        }
        if rng.r#gen::<f32>() < p {
            jitter_style(&mut node.style, rng, strength);
        }
    }

    // Structural mutation: occasionally drop or duplicate a leaf child.
    if !node.children.is_empty() && rng.r#gen::<f32>() < 0.08 * strength {
        let idx = rng.gen_range(0..node.children.len());
        let child_is_plain_leaf = node.children[idx].children.is_empty()
            && !matches!(
                node.children[idx].kind,
                NodeKind::RadialGradient { .. }
                    | NodeKind::Filter { .. }
                    | NodeKind::UseFilter { .. }
            );
        if child_is_plain_leaf {
            if node.children.len() > 1 && rng.r#gen::<bool>() {
                node.children.remove(idx);
            } else {
                let dup = node.children[idx].clone();
                node.children.push(dup);
            }
        }
    }

    for child in &mut node.children {
        mutate_node(child, rng, strength, viewport);
    }
}

/// Jitter the numeric geometry of a node (or its transform, for kinds
/// without inline coordinates such as groups and paths).
fn jitter_geometry(node: &mut SceneNode, rng: &mut StdRng, dpos: f32) {
    fn j(rng: &mut StdRng, dpos: f32) -> f32 {
        (rng.r#gen::<f32>() * 2.0 - 1.0) * dpos
    }

    match &mut node.kind {
        NodeKind::Circle { cx, cy, r } => {
            *cx += j(rng, dpos);
            *cy += j(rng, dpos);
            *r = (*r + j(rng, dpos) * 0.5).max(0.5);
        }
        NodeKind::Ellipse { cx, cy, rx, ry } => {
            *cx += j(rng, dpos);
            *cy += j(rng, dpos);
            *rx = (*rx + j(rng, dpos) * 0.5).max(0.5);
            *ry = (*ry + j(rng, dpos) * 0.5).max(0.5);
        }
        NodeKind::Line { x1, y1, x2, y2 } => {
            *x1 += j(rng, dpos);
            *y1 += j(rng, dpos);
            *x2 += j(rng, dpos);
            *y2 += j(rng, dpos);
        }
        NodeKind::Rect { x, y, w, h, .. } => {
            *x += j(rng, dpos);
            *y += j(rng, dpos);
            *w = (*w + j(rng, dpos) * 0.5).max(1.0);
            *h = (*h + j(rng, dpos) * 0.5).max(1.0);
        }
        NodeKind::Polygon { points, .. } => {
            // Translate the whole polygon so the shape stays coherent.
            let dx = j(rng, dpos);
            let dy = j(rng, dpos);
            for (x, y) in points.iter_mut() {
                *x += dx;
                *y += dy;
            }
        }
        // Groups and raw paths carry no inline coordinates: nudge the
        // node's affine transform instead.
        NodeKind::Group { .. } | NodeKind::Path { .. } => {
            node.transform.translate_x += j(rng, dpos) * 0.5;
            node.transform.translate_y += j(rng, dpos) * 0.5;
        }
        // Defs and references: leave untouched (ids must stay stable).
        NodeKind::RadialGradient { .. } | NodeKind::Filter { .. } | NodeKind::UseFilter { .. } => {}
    }
}

/// Jitter fill/stroke colors in HSL space (hue and lightness).
fn jitter_style(style: &mut Style, rng: &mut StdRng, strength: f32) {
    if let Some(c) = style.fill {
        style.fill = Some(jitter_color(c, rng, strength));
    }
    if let Some(c) = style.stroke {
        style.stroke = Some(jitter_color(c, rng, strength));
    }
}

fn jitter_color(c: Color, rng: &mut StdRng, strength: f32) -> Color {
    let (h, s, l) = c.to_hsl();
    let dh = (rng.r#gen::<f32>() * 2.0 - 1.0) * 24.0 * strength;
    let dl = (rng.r#gen::<f32>() * 2.0 - 1.0) * 0.08 * strength;
    Color::from_hsla(h + dh, s, (l + dl).clamp(0.05, 0.95), c.a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rich_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5; 8],
            dopamine: 0.6,
            serotonin: 0.5,
            thought_vector: vec![0.3, -0.2, 0.5, 0.1],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn iterative_improves_or_maintains() {
        let config = AtelierConfig {
            iteration_budget: 5,
            ..AtelierConfig::default()
        };
        let snapshot = rich_snapshot();

        let artwork = create_iterative(&config, &snapshot, 42);
        assert!(artwork.generation_cycles == 5);
        assert!(artwork.aesthetic_score.composite >= 0.0);
    }

    #[test]
    fn external_scorer_runs_in_exploit_only() {
        let config = AtelierConfig {
            iteration_budget: 8, // 4 explore + 4 exploit
            ..AtelierConfig::default()
        };
        let snapshot = rich_snapshot();
        let mut calls = 0usize;
        let mut scorer = |_scene: &SceneNode, _snap: &CognitiveSnapshot| -> Option<f32> {
            calls += 1;
            Some(0.5)
        };
        let artwork = create_iterative_scored(&config, &snapshot, 42, Some(&mut scorer));
        assert!(artwork.svg.contains("<svg"));
        // 1 re-baseline call + 4 exploit candidates = 5; never the 4 explore.
        assert_eq!(
            calls, 5,
            "scorer must run once per exploit candidate + baseline"
        );
    }

    /// A constant external score preserves the internal ordering, so the
    /// scored variant must select the exact same artwork as the unscored one
    /// — proof the blend is a selection signal, not noise.
    #[test]
    fn constant_external_scorer_preserves_selection() {
        let config = AtelierConfig {
            iteration_budget: 8,
            ..AtelierConfig::default()
        };
        let snapshot = rich_snapshot();
        let unscored = create_iterative(&config, &snapshot, 42);
        let mut scorer = |_: &SceneNode, _: &CognitiveSnapshot| -> Option<f32> { Some(0.5) };
        let scored = create_iterative_scored(&config, &snapshot, 42, Some(&mut scorer));
        assert_eq!(unscored.svg, scored.svg);
    }

    /// A maximally-disapproving external scorer must be able to veto every
    /// exploit mutation — deterministic proof the external signal steers
    /// acceptance rather than being ignored.
    #[test]
    fn external_scorer_vetoes_exploit_mutations() {
        let config = AtelierConfig {
            iteration_budget: 12,
            ..AtelierConfig::default()
        };
        let snapshot = rich_snapshot();

        // Reproduce the explore-phase winner exactly as create_iterative does.
        let explore = config.iteration_budget.div_ceil(2);
        let mut explore_best = f32::NEG_INFINITY;
        let mut explore_scene = None;
        for i in 0..explore {
            let seed = 42u64.wrapping_add(i as u64 * 7919);
            let mut rng = StdRng::seed_from_u64(seed);
            let scene = generate(&config, &snapshot, &mut rng);
            let s = score_scene(&scene, &snapshot).composite;
            if s > explore_best {
                explore_best = s;
                explore_scene = Some(scene);
            }
        }
        let explore_svg = symthaea_canvas::render_svg(
            &explore_scene.expect("explore ran"),
            snapshot.consciousness_level,
        );
        // Precondition for the veto arithmetic below: acceptance under a
        // baseline=1.0 / mutant=0.0 scorer needs internal > baseline + 2/3,
        // impossible once the baseline internal composite is ≥ 1/3.
        assert!(
            explore_best >= 0.34,
            "test precondition: explore best {explore_best} too low for a guaranteed veto"
        );

        let mut first_call = true;
        let mut scorer = |_: &SceneNode, _: &CognitiveSnapshot| -> Option<f32> {
            // Baseline (first call) gets full approval; every mutant gets
            // zero — an absolute veto on change.
            let v = if first_call { 1.0 } else { 0.0 };
            first_call = false;
            Some(v)
        };
        let vetoed = create_iterative_scored(&config, &snapshot, 42, Some(&mut scorer));
        assert_eq!(
            vetoed.svg, explore_svg,
            "a vetoing external scorer must freeze the explore winner"
        );

        // And whenever the unscored run does accept an exploit mutation,
        // the veto demonstrably changed the outcome.
        let unscored = create_iterative(&config, &snapshot, 42);
        if unscored.svg != explore_svg {
            assert_ne!(vetoed.svg, unscored.svg, "veto should have steered");
        }
    }

    #[test]
    fn zero_budget_still_produces() {
        let config = AtelierConfig {
            iteration_budget: 0,
            ..AtelierConfig::default()
        };
        let snapshot = CognitiveSnapshot::dormant();
        let artwork = create_iterative(&config, &snapshot, 42);
        assert!(artwork.svg.contains("<svg"));
        assert!(artwork.generation_cycles >= 1);
    }

    #[test]
    fn mutation_produces_different_valid_scene() {
        let config = AtelierConfig::default();
        let snapshot = rich_snapshot();
        let mut gen_rng = StdRng::seed_from_u64(7);
        let scene = generate(&config, &snapshot, &mut gen_rng);

        let mut mut_rng = StdRng::seed_from_u64(99);
        let mutant = mutate_scene(&scene, &mut mut_rng, 0.8, (config.width, config.height));

        // Valid: renders to well-formed SVG and keeps a non-trivial tree.
        let mutant_svg = symthaea_canvas::render_svg(&mutant, snapshot.consciousness_level);
        assert!(mutant_svg.contains("<svg"));
        assert!(mutant_svg.contains("</svg>"));
        assert!(mutant.node_count() > 1);

        // Structurally different from the parent.
        let parent_svg = symthaea_canvas::render_svg(&scene, snapshot.consciousness_level);
        assert_ne!(parent_svg, mutant_svg);
    }

    #[test]
    fn iterative_is_deterministic() {
        let config = AtelierConfig {
            iteration_budget: 8,
            ..AtelierConfig::default()
        };
        let snapshot = rich_snapshot();
        let a1 = create_iterative(&config, &snapshot, 42);
        let a2 = create_iterative(&config, &snapshot, 42);
        assert_eq!(a1.svg, a2.svg);
        assert_eq!(
            a1.aesthetic_score.composite.to_bits(),
            a2.aesthetic_score.composite.to_bits()
        );
    }

    #[test]
    fn exploit_never_scores_below_explore_best() {
        let config = AtelierConfig {
            iteration_budget: 10,
            ..AtelierConfig::default()
        };
        let snapshot = rich_snapshot();
        let base_seed = 42u64;

        // Recompute the explore-phase best exactly as create_iterative does.
        let explore = config.iteration_budget.div_ceil(2);
        let mut explore_best = f32::NEG_INFINITY;
        for i in 0..explore {
            let seed = base_seed.wrapping_add(i as u64 * 7919);
            let mut rng = StdRng::seed_from_u64(seed);
            let scene = generate(&config, &snapshot, &mut rng);
            explore_best = explore_best.max(score_scene(&scene, &snapshot).composite);
        }

        let artwork = create_iterative(&config, &snapshot, base_seed);
        assert!(
            artwork.aesthetic_score.composite >= explore_best,
            "hill-climbing must never lose ground: final {} < explore best {}",
            artwork.aesthetic_score.composite,
            explore_best
        );
    }
}
