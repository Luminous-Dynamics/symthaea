// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The Premise layer: seeds choose MUSICAL PREMISES, not just notes.
//!
//! The listening analysis that demanded it (four same-style candidates,
//! measured): "event-level: different — note overlap under 10%.
//! Perceptual/style-level: still very similar — same tempo, same meter,
//! same track count, similar palette, similar density arcs... the seeds
//! are changing the surface, not the identity. I would not start by
//! randomizing more notes. I would randomize higher-level musical
//! identity... a seeded identity layer above note generation. Not 'more
//! randomness,' but 'different musical premises.'"
//!
//! [`premise_for`] deterministically derives a per-candidate premise from
//! the seed — a MODIFIED clone of the style's spec plus a phrase-length
//! multiplier — varying exactly the uniforms the analysis measured:
//!
//! - **Tempo**: the style's tempo band is split into thirds; each
//!   candidate lives in one (so a batch spans slow/mid/quick readings of
//!   the same style instead of clustering at one BPM).
//! - **Texture budget**: Sparse / Standard / Full tiers, flipped through
//!   the REAL TextureSpec switches (counter-melody, return color, intro
//!   length, coda length) — some pieces 2-3 active layers, others the
//!   full seven.
//! - **Length**: a bars multiplier of 1 or 2 — short statements and long
//!   arcs in the same batch.
//! - **Ensemble persona**: the pick is decorrelated from the other seed
//!   slices (seed/61) so nearby candidates stop sharing a sound-world by
//!   accident.
//! - **Mode/gravity**: styles may opt into a `mode_pool`; candidates then
//!   inhabit different modal centers (Classical/Folk opt in; Tango does
//!   NOT — harmonic minor IS that style's identity).
//!
//! Deliberately NOT varied here: **meter**. Motif banks are meter-locked
//! by invariant (every template totals exactly `meter` beats), so meter
//! families (3/4, 5/4, 6/8, additive) require per-meter banks — a real
//! future layer, not a switch to flip.
//!
//! Every variation stays inside the style's own legal bounds: the premise
//! narrows or selects, it never invents values the spec didn't offer.

use crate::spec::{CompositionSpec, SpecNote};

/// Adapt one motif template to a new meter, EXACTLY. Longer templates are
/// truncated with a rational cut (the crossing note shortened to fill the
/// bar precisely, in 480ths — the crate's tick-like exact denominator);
/// shorter ones breathe out: the final tone is extended by the deficit.
/// Either way the result totals exactly `target` beats, so an adapted bank
/// still satisfies the invariant every consumer relies on. This is what
/// makes meter families honest: the banks are authored once, in the
/// style's home meter, and each premise carries a self-consistent copy.
pub(crate) fn adapt_template_to_meter(template: &[SpecNote], target: u8) -> Vec<SpecNote> {
    let target_480: i64 = target as i64 * 480;
    let mut out: Vec<SpecNote> = Vec::new();
    let mut acc: i64 = 0; // in 480ths
    for &(deg, num, den) in template {
        let dur = num * 480 / den;
        if acc >= target_480 {
            break;
        }
        if acc + dur <= target_480 {
            out.push((deg, num, den));
            acc += dur;
        } else {
            let remaining = target_480 - acc;
            out.push((deg, remaining, 480));
            acc = target_480;
        }
    }
    if acc < target_480 {
        let deficit = target_480 - acc;
        if let Some(last) = out.last_mut() {
            // Extend the final tone: re-express it in 480ths + deficit.
            let last_480 = last.1 * 480 / last.2 + deficit;
            *last = (last.0, last_480, 480);
        } else {
            out.push((1, deficit, 480));
        }
    }
    out
}

/// The texture tiers a premise may choose. Tiers only ever FLIP existing
/// TextureSpec switches — the engine's invariants are untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureTier {
    /// 2-3 active layers: no counter-melody, no return color, cold open,
    /// short coda — the piece as a drawing, not a painting.
    Sparse,
    /// The style's own texture, exactly as authored.
    Standard,
    /// Everything the style allows, plus the patient opening: full
    /// counter/color, longer intro, full coda.
    Full,
}

/// A candidate's premise: the modified spec it composes under, plus the
/// phrase-length multiplier applied to the intent's bars.
pub struct Premise {
    pub spec: CompositionSpec,
    pub bars_multiplier: usize,
    pub tempo_third: usize,
    pub texture_tier: TextureTier,
    /// The meter this premise chose (== the base meter unless the style
    /// opted into a meter_pool).
    pub meter: u8,
}

/// Derive the premise a seed implies under a spec. Deterministic; every
/// slice of the seed is decorrelated from the slices the composer already
/// uses (form: seed%pool, accompaniment: seed/2, motif: seed-based,
/// ensemble default: seed%pool) so premises don't shadow existing choices.
pub fn premise_for(base: &CompositionSpec, seed: u64) -> Premise {
    let mut spec = base.clone();

    // ── Tempo: live in one third of the style's band ─────────────────────
    let tempo_third = ((seed / 7) % 3) as usize;
    let (lo, hi) = spec.tempo_range;
    let step = (hi - lo) / 3.0;
    spec.tempo_range = (
        lo + step * tempo_third as f32,
        lo + step * (tempo_third + 1) as f32,
    );

    // ── Texture budget ───────────────────────────────────────────────────
    let texture_tier = match (seed / 11) % 3 {
        0 => TextureTier::Sparse,
        1 => TextureTier::Standard,
        _ => TextureTier::Full,
    };
    match texture_tier {
        TextureTier::Sparse => {
            spec.texture.counter_melody = false;
            spec.texture.return_color = false;
            spec.texture.intro_bars = 0;
            spec.texture.coda_bars = spec.texture.coda_bars.min(1);
        }
        TextureTier::Standard => {}
        TextureTier::Full => {
            spec.texture.counter_melody = true;
            spec.texture.return_color = true;
            spec.texture.intro_bars = spec.texture.intro_bars.max(3);
        }
    }

    // ── Length: statements and arcs ──────────────────────────────────────
    let bars_multiplier = if (seed / 5) % 3 == 2 { 2 } else { 1 };

    // ── Ensemble persona: decorrelated pick ──────────────────────────────
    if !spec.ensemble_pool.is_empty() {
        let n = spec.ensemble_pool.len();
        spec.ensemble_pool.rotate_left(((seed / 61) as usize) % n);
    }

    // ── Mode/gravity: only where the style opted in ──────────────────────
    if !spec.mode_pool.is_empty() {
        let pick = ((seed / 17) as usize) % spec.mode_pool.len();
        spec.mode = spec.mode_pool[pick];
    }

    // ── Meter family: only where the style opted in ──────────────────────
    // The pick rewrites spec.meter AND adapts every bank exactly, so the
    // premise spec remains self-consistent (bank invariant included).
    if !spec.meter_pool.is_empty() {
        let pick = spec.meter_pool[((seed / 23) as usize) % spec.meter_pool.len()];
        if pick != spec.meter && pick >= 2 {
            spec.meter = pick;
            for bank in [
                &mut spec.motifs_calm,
                &mut spec.motifs_medium,
                &mut spec.motifs_busy,
            ] {
                for template in bank.iter_mut() {
                    *template = adapt_template_to_meter(template, pick);
                }
            }
            if pick == 5 {
                // Quintuple meter gets its own gait — a 4/4 pattern
                // looping arithmetically is metrically legal but
                // gaitless; the FiveGait cell spells the 3+2 grouping.
                spec.accompaniment_pool = vec![crate::accompaniment::Accompaniment::FiveGait];
            }
        }
    }
    let meter = spec.meter;

    Premise {
        spec,
        bars_multiplier,
        tempo_third,
        texture_tier,
        meter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Style;

    #[test]
    fn premises_stay_inside_the_style() {
        // The premise narrows or selects; it never leaves the style's own
        // legal bounds.
        for style in [Style::Classical, Style::Tango, Style::Nocturne] {
            let base = style.spec();
            for seed in 0..48u64 {
                let p = premise_for(&base, seed);
                let (blo, bhi) = base.tempo_range;
                let (lo, hi) = p.spec.tempo_range;
                assert!(
                    lo >= blo - 1e-6 && hi <= bhi + 1e-6,
                    "{style:?} tempo escaped"
                );
                assert!(lo < hi);
                // Ensembles are a rotation of the style's own pool.
                let mut a = base.ensemble_pool.clone();
                let mut b = p.spec.ensemble_pool.clone();
                a.sort();
                b.sort();
                assert_eq!(a, b, "{style:?} ensemble pool mutated");
                // The premise spec still validates.
                p.spec
                    .validate()
                    .unwrap_or_else(|e| panic!("{style:?} seed {seed}: {e:?}"));
                assert!(p.bars_multiplier == 1 || p.bars_multiplier == 2);
            }
        }
    }

    #[test]
    fn a_batch_of_premises_genuinely_varies() {
        // The uniforms the listening analysis measured must actually vary
        // across a small window: tempo thirds, texture tiers, and lengths
        // all take at least two values in 16 seeds.
        let base = Style::Classical.spec();
        let premises: Vec<Premise> = (0..16).map(|s| premise_for(&base, s)).collect();
        let thirds: std::collections::HashSet<usize> =
            premises.iter().map(|p| p.tempo_third).collect();
        let tiers: std::collections::HashSet<u8> =
            premises.iter().map(|p| p.texture_tier as u8).collect();
        let lengths: std::collections::HashSet<usize> =
            premises.iter().map(|p| p.bars_multiplier).collect();
        assert!(thirds.len() >= 2, "tempo never varied");
        assert!(tiers.len() >= 2, "texture never varied");
        assert!(lengths.len() >= 2, "length never varied");
    }

    #[test]
    fn tango_keeps_its_mode_but_classical_may_wander() {
        // Mode variation is OPT-IN: harmonic minor IS tango's identity.
        let tango = Style::Tango.spec();
        for seed in 0..24u64 {
            assert_eq!(premise_for(&tango, seed).spec.mode, tango.mode);
        }
        let classical = Style::Classical.spec();
        if !classical.mode_pool.is_empty() {
            let modes: std::collections::HashSet<_> = (0..24u64)
                .map(|s| premise_for(&classical, s).spec.mode)
                .collect();
            assert!(modes.len() >= 2, "classical opted in but never wandered");
        }
    }

    #[test]
    fn a_meter_five_premise_walks_the_gait() {
        // End to end: a Cinematic premise that picks 5 must swap its
        // accompaniment to the FiveGait, and the composed bass must spell
        // the 3+2 split — a three-beat anchor and a two-beat answer per
        // body bar. (Seed 23: (23/23) % 2 = 1 -> the pool's meter 5.)
        let base = Style::Cinematic.spec();
        let p = premise_for(&base, 23);
        assert_eq!(p.meter, 5);
        assert_eq!(
            p.spec.accompaniment_pool,
            vec![crate::accompaniment::Accompaniment::FiveGait]
        );
        let intent = crate::composer::MusicalIntent {
            seed: 23,
            ..Default::default()
        };
        let s = crate::composer::compose_with_spec(&intent, &p.spec);
        assert_eq!(s.meter, 5);
        // Check a clean CONSEQUENT bar — staged entrances keep the bass
        // out of the opening phrase by design, so probe past it.
        let intro = p.spec.texture.intro_bars as f64;
        let bars = intent.bars.max(1) as f64;
        let lo = (intro + bars + 1.0) * 5.0;
        let bar: Vec<_> = s
            .notes
            .iter()
            .filter(|n| {
                n.role == crate::score::VoiceRole::Bass
                    && n.onset.beats() >= lo - 1e-9
                    && n.onset.beats() < lo + 5.0 - 1e-9
            })
            .collect();
        assert_eq!(bar.len(), 2, "the five-gait bass is two anchors");
        let mut durs: Vec<f64> = bar.iter().map(|n| n.duration.beats()).collect();
        durs.sort_by(f64::total_cmp);
        assert_eq!(durs, vec![2.0, 3.0], "the 3+2 split, spelled");
    }

    #[test]
    fn premises_are_deterministic() {
        let base = Style::Nocturne.spec();
        for seed in [0u64, 9, 77] {
            let (a, b) = (premise_for(&base, seed), premise_for(&base, seed));
            assert_eq!(a.spec, b.spec);
            assert_eq!(a.bars_multiplier, b.bars_multiplier);
        }
    }

    #[test]
    fn template_adaptation_is_rationally_exact() {
        // Truncation and extension both land EXACTLY on the target meter.
        let four_beats: Vec<crate::spec::SpecNote> =
            vec![(1, 1, 1), (3, 1, 2), (2, 1, 2), (5, 2, 1)];
        for target in [3u8, 4, 5] {
            let adapted = adapt_template_to_meter(&four_beats, target);
            let total: i64 = adapted.iter().map(|&(_, n, d)| n * 480 / d).sum();
            assert_eq!(total, target as i64 * 480, "target {target}");
            assert!(!adapted.is_empty());
        }
        // Truncation cuts the crossing note, never drops the head.
        let t3 = adapt_template_to_meter(&four_beats, 3);
        assert_eq!(t3[0], (1, 1, 1));
        // Extension breathes out the final tone (same degree, longer).
        let t5 = adapt_template_to_meter(&four_beats, 5);
        assert_eq!(t5.last().unwrap().0, 5);
    }

    #[test]
    fn meter_premises_stay_in_the_pool_and_stay_valid() {
        let base = Style::Classical.spec();
        let mut seen = std::collections::HashSet::new();
        for seed in 0..64u64 {
            let p = premise_for(&base, seed);
            assert!(
                base.meter_pool.contains(&p.meter) || p.meter == base.meter,
                "meter {} escaped the pool",
                p.meter
            );
            seen.insert(p.meter);
            p.spec
                .validate()
                .unwrap_or_else(|e| panic!("seed {seed}: adapted spec invalid: {e:?}"));
        }
        assert!(seen.len() >= 2, "classical never varied meter: {seen:?}");
        // Fixed-meter styles never move: the waltz IS 3/4.
        let waltz = Style::Waltz.spec();
        for seed in 0..24u64 {
            assert_eq!(premise_for(&waltz, seed).meter, waltz.meter);
        }
    }
}
