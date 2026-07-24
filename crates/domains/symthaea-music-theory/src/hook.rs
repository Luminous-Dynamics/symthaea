// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The hook cell: a tiny memorable phrase generated BEFORE the melody, so
//! the piece has a name to say. The listening-review arc that led here:
//! "mood without motion" → "motion without consequence" → (damage pass) →
//! "the structure has wounds... but the melody still feels more like a
//! generated line than a remembered theme... generate a tiny memorable
//! cell before generating the full melody. Something like a 3-5 note
//! 'name' that can survive augmentation, inversion, silence, and
//! reharmonization."
//!
//! What makes a cell memorable is IDENTITY, and identity is enforced here
//! by construction and checked by predicate:
//! - **rhythmic identity**: at least two distinct durations, the longest
//!   at least twice the shortest (short-short-LONG, LONG-short-short…) —
//!   a uniform rhythm is a texture, not a name;
//! - **contour identity**: a leap of at least a third followed by motion
//!   the other way (reach and recoil), or an immediate repeated note
//!   (insistence) — a scale fragment is a direction, not a name.
//!
//! The cell becomes the HEAD of the bar-motif ([`graft_hook`] keeps the
//! spec template's tail as connective tissue), which means the existing
//! machinery gives it survival for free: Caplin sentence structure
//! restates the head, the development fragments it, the transformed coda
//! already quotes the piece's opening notes — which are now the hook.

use crate::motif::{Motif, MotifNote};
use crate::rhythm::Duration;

/// A tiny memorable cell: 3-5 (degree, duration) notes with enforced
/// rhythmic and contour identity. Degrees are 1-based scale degrees, the
/// same language as spec motif templates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookCell {
    pub notes: Vec<(i32, Duration)>,
}

/// Rhythm skeletons, in half-beat units (so 1 = an eighth at 4/4's beat=1
/// granularity… concretely: values are (num, den) beats). Every skeleton
/// carries rhythmic identity by construction; the predicate re-checks.
const RHYTHMS: &[&[(i64, i64)]] = &[
    &[(1, 2), (1, 2), (2, 1)],         // short-short-LONG
    &[(2, 1), (1, 2), (1, 2)],         // LONG-short-short
    &[(3, 2), (1, 2), (1, 1)],         // dotted call
    &[(1, 2), (3, 2), (1, 1)],         // snap: short-LONG-med
    &[(1, 2), (1, 2), (1, 2), (3, 2)], // three steps and a landing
    &[(1, 1), (1, 2), (1, 2), (2, 1)], // walk, hurry, arrive
];

/// Contour skeletons (scale degrees). Each carries contour identity:
/// a leap ≥ a third with recoil, or an immediate repetition.
const CONTOURS: &[&[i32]] = &[
    &[1, 1, 5],    // insistence, then the reach
    &[1, 5, 4],    // reach and recoil
    &[3, 3, 1],    // insistence, then the fall
    &[5, 1, 2],    // the fall and the turn
    &[1, 2, 1, 5], // turn, then the reach
    &[5, 5, 6, 3], // insistence, rise, the fall
    &[1, 4, 3, 1], // reach, recoil, home
    &[2, 1, 1, 5], // lean, insist, reach
];

/// Splitmix-style scramble: variant selection must not correlate with the
/// pool pick (seed % pool_len) or the premise slices (seed / small-divisor).
///
/// `pub(crate)` so [`crate::composer::CompositionSeeds`] can reuse this same
/// mixing function to derive its domain-separated substreams — one
/// implementation, several call sites, rather than a second copy of the
/// same SplitMix64-style algorithm.
pub(crate) fn scramble(seed: u64, salt: u64) -> u64 {
    let mut z = seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl HookCell {
    /// Diatonic transposition — intervals (and so both identities)
    /// untouched. `pub(crate)` so [`crate::motif_foundry`] can reuse it for
    /// transform-invariant canonical fingerprinting.
    pub(crate) fn transposed(&self, steps: i32) -> HookCell {
        HookCell {
            notes: self
                .notes
                .iter()
                .map(|&(d, dur)| (d + steps, dur))
                .collect(),
        }
    }

    /// Event order reversed (pitch AND rhythm run backwards). `pub(crate)`
    /// -- see [`Self::transposed`].
    pub(crate) fn retrograded(&self) -> HookCell {
        let mut notes = self.notes.clone();
        notes.reverse();
        HookCell { notes }
    }

    /// Does the LONGEST note fall on the signature moment (the target of
    /// the largest step)? The pool-level reach-alignment preference, as a
    /// predicate — variant shapes must keep it too ("a reach that lands
    /// on the long note is a hook"). `pub(crate)` -- see
    /// [`Self::transposed`]; used by [`crate::motif_foundry`] to give
    /// procedural variants the same reach-alignment filter the classic
    /// pool's combinatorial-variety step applies.
    pub(crate) fn is_reach_aligned(&self) -> bool {
        let longest = self
            .notes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.1.beats().total_cmp(&b.1.1.beats()))
            .map(|(i, _)| i)
            .unwrap_or(0);
        longest == self.signature_index()
    }

    /// Contour mirrored about the first degree. `pub(crate)` -- see
    /// [`Self::transposed`].
    pub(crate) fn inverted(&self) -> HookCell {
        let pivot = self.notes.first().map(|&(d, _)| d).unwrap_or(1);
        HookCell {
            notes: self
                .notes
                .iter()
                .map(|&(d, dur)| (2 * pivot - d, dur))
                .collect(),
        }
    }

    /// Generate the piece's hook deterministically from its seed, using
    /// the classic shared pools. Only candidates that FIT the meter (cell
    /// ≤ meter, leaving room for a tail) are considered, so every meter
    /// gets a valid name.
    pub fn generate(seed: u64, meter_beats: f64) -> HookCell {
        Self::generate_with(&crate::spec::MelodicDna::default(), seed, meter_beats)
    }

    /// Generate the hook from a style's MELODIC DNA — the layer the tango
    /// listening review demanded ("the melody still sounds like Muse...
    /// the style should own the hook"). Non-empty DNA pools replace the
    /// classic shared pools for their axis; every DNA candidate is still
    /// filtered through the SAME identity predicates (rhythmic identity,
    /// contour identity) — DNA changes which identities a style prefers,
    /// never whether the hook has identity. Invalid or meter-unfittable
    /// DNA degrades gracefully to the classic pools.
    ///
    /// `dna.use_procedural_foundry` is a separate opt-in (default false,
    /// unset by every style preset in this crate today): when set,
    /// delegates entirely to [`crate::motif_foundry::generate_with_foundry`]
    /// instead of the hand-authored pools below — the Motif Foundry's
    /// procedural generator, biased toward this DNA's own measured
    /// contour/rhythm character. Existing callers that never touch this
    /// field see byte-identical behavior.
    pub fn generate_with(dna: &crate::spec::MelodicDna, seed: u64, meter_beats: f64) -> HookCell {
        if dna.use_procedural_foundry {
            return crate::motif_foundry::generate_with_foundry(dna, seed, meter_beats);
        }
        let dna_rhythms: Vec<&[(i64, i64)]> =
            dna.hook_rhythms.iter().map(|r| r.as_slice()).collect();
        let dna_contours: Vec<&[i32]> = dna.hook_contours.iter().map(|c| c.as_slice()).collect();
        let rhythms: &[&[(i64, i64)]] = if dna_rhythms.is_empty() {
            RHYTHMS
        } else {
            &dna_rhythms
        };
        let contours: &[&[i32]] = if dna_contours.is_empty() {
            CONTOURS
        } else {
            &dna_contours
        };
        let build = |rhythms: &[&[(i64, i64)]], contours: &[&[i32]]| -> Vec<HookCell> {
            let mut out = Vec::new();
            for r in rhythms {
                for c in contours {
                    if r.len() != c.len() {
                        continue;
                    }
                    let total: f64 = r.iter().map(|&(n, d)| n as f64 / d as f64).sum();
                    if total > meter_beats + 1e-9 {
                        continue;
                    }
                    let cell = HookCell {
                        notes: c
                            .iter()
                            .zip(r.iter())
                            .map(|(&deg, &(n, d))| (deg, Duration::new(n, d)))
                            .collect(),
                    };
                    // DNA pools are user/style data: filter, don't assert.
                    if cell.has_rhythmic_identity() && cell.has_contour_identity() {
                        out.push(cell);
                    }
                }
            }
            out
        };
        let mut candidates = build(rhythms, contours);
        if candidates.is_empty() {
            candidates = build(RHYTHMS, CONTOURS);
        }
        debug_assert!(!candidates.is_empty(), "no hook fits meter {meter_beats}");
        // REACH ALIGNMENT (hook surgery round 2): a name sticks when its
        // signature moment is dwelt on — prefer cells whose LONGEST note
        // is the contour's signature note (the target of the largest
        // step). A reach that lands on a short note is a gesture; a reach
        // that lands on the long note is a hook.
        let aligned: Vec<HookCell> = candidates
            .iter()
            .filter(|c| {
                let longest = c
                    .notes
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.1.beats().total_cmp(&b.1.1.beats()))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                longest == c.signature_index()
            })
            .cloned()
            .collect();
        let pool = if aligned.is_empty() {
            &candidates
        } else {
            &aligned
        };
        let base = pool[(seed % pool.len().max(1) as u64) as usize].clone();
        // COMBINATORIAL VARIETY (a Listen-tab report: "some of the same
        // hooks are too common and recognizable"): the authored pools are
        // small — a dozen-odd valid cells — so `seed % pool_len` mapped
        // thousands of seeds onto the same few names. Every classical
        // motivic transformation of a valid cell is another valid name:
        // diatonic transposition (identities untouched by construction),
        // retrograde, and inversion — the latter two kept only when they
        // still pass BOTH identity predicates. Up to ~12 variants per
        // cell, selected by a scrambled slice so the choice correlates
        // with neither the pool pick nor the premise arithmetic.
        let mut shapes: Vec<HookCell> = vec![base.clone()];
        let retro = base.retrograded();
        if retro.has_rhythmic_identity() && retro.has_contour_identity() && retro.is_reach_aligned()
        {
            shapes.push(retro);
        }
        let inv = base.inverted();
        if inv.has_rhythmic_identity() && inv.has_contour_identity() && inv.is_reach_aligned() {
            shapes.push(inv);
        }
        let mut variants: Vec<HookCell> = Vec::new();
        for shape in &shapes {
            for t in 0..4 {
                variants.push(shape.transposed(t));
            }
        }
        variants[(scramble(seed, 0x400C) as usize) % variants.len()].clone()
    }

    /// The index of the cell's SIGNATURE note: the target of its largest
    /// step — the moment the ear names first.
    pub fn signature_index(&self) -> usize {
        let degs: Vec<i32> = self.notes.iter().map(|(d, _)| *d).collect();
        let mut best = (degs.len().saturating_sub(1), 0i32);
        for i in 1..degs.len() {
            let d = degs[i] - degs[i - 1];
            if d.abs() > best.1.abs() {
                best = (i, d);
            }
        }
        best.0
    }

    /// Total length in beats.
    pub fn beats(&self) -> f64 {
        self.notes.iter().map(|(_, d)| d.beats()).sum()
    }

    /// At least two distinct durations, longest ≥ 2× shortest.
    pub fn has_rhythmic_identity(&self) -> bool {
        let mut beats: Vec<f64> = self.notes.iter().map(|(_, d)| d.beats()).collect();
        beats.sort_by(|a, b| a.total_cmp(b));
        match (beats.first(), beats.last()) {
            (Some(&lo), Some(&hi)) => hi >= lo * 2.0 - 1e-9 && hi - lo > 1e-9,
            _ => false,
        }
    }

    /// A leap ≥ a third answered by opposite-direction motion, or an
    /// immediate repeated note.
    pub fn has_contour_identity(&self) -> bool {
        let degs: Vec<i32> = self.notes.iter().map(|(d, _)| *d).collect();
        let repeats = degs.windows(2).any(|w| w[0] == w[1]);
        let leap_recoil = degs.windows(3).any(|w| {
            let a = w[1] - w[0];
            let b = w[2] - w[1];
            a.abs() >= 2 && b != 0 && a.signum() != b.signum()
        });
        // A closing leap (reach as the last gesture) also counts — the
        // recoil is the next bar's restatement.
        let closing_leap = degs
            .windows(2)
            .last()
            .map(|w| (w[1] - w[0]).abs() >= 3)
            .unwrap_or(false);
        repeats || leap_recoil || closing_leap
    }
}

/// Shift `degree` by whole octaves (steps of 7, this crate's diatonic
/// octave) to land as close as possible to `reference`. Used at the
/// hook/style-tail splice in [`graft_hook`]: without it, a raw style-tail
/// degree (typically 1, the template's tonic-anchored opening) could sit
/// many octaves away from wherever the hook's own material (which can
/// itself be transposed by the combinatorial-variety step in
/// [`HookCell::generate_with`]) actually left off, producing a huge,
/// unconstrained leap at the splice point regardless of which hook was
/// picked. Found 2026-07-24 via a real listening-test-driven trace (a
/// March clip misheard as Tango turned out to have this exact splice as
/// its dominant source of octave-plus leaps, not the hook content itself).
fn nearest_octave_degree(degree: i32, reference: i32) -> i32 {
    (-3..=3)
        .map(|k| degree + 7 * k)
        .min_by_key(|&d| (d - reference).abs())
        .unwrap_or(degree)
}

/// Graft the hook onto a spec-template motif: the hook becomes the HEAD
/// (the phrase's name), and the remaining beats become a varied ECHO of
/// the head — its first tones restated a step lower, compressed. The bar
/// says the name TWICE (statement + echo), which is the actual earworm
/// mechanism; the first version filled the tail from the style template,
/// and that anonymous connective tissue diluted the very identity the
/// hook existed to create ("the memory arc is now stronger than the hook
/// itself"). Total stays exactly `meter_beats`. The template parameter is
/// kept as the fallback shape when a cell is too exotic to echo.
pub fn graft_hook(template: &Motif, hook: &HookCell, meter_beats: f64) -> Motif {
    let mut notes: Vec<MotifNote> = hook
        .notes
        .iter()
        .map(|&(deg, dur)| MotifNote::new(deg, dur))
        .collect();
    let remaining = meter_beats - hook.beats();
    // Quarter-beat grid: remaining is always a multiple of 0.5 (integer
    // meters, half-beat rhythm skeletons).
    if remaining >= 0.999 {
        // Half ECHO, half STYLE: the first tail tone restates the hook's
        // opening a step lower (the name said twice), the rest comes from
        // the style template (the bar wears its style's colors). The
        // pure-echo first cut erased style identity from the melody
        // entirely — every 4/4 style composed the same line per seed,
        // caught when March collided with Classical the moment their
        // progressions coincided.
        let echo_degree = hook.notes[0].0 - 1;
        notes.push(MotifNote::new(echo_degree, Duration::new(1, 2)));
        let style_deg = template
            .notes
            .first()
            .and_then(|n| n.degree)
            .unwrap_or(hook.notes[0].0);
        notes.push(MotifNote::new(
            nearest_octave_degree(style_deg, echo_degree),
            Duration::new(((remaining - 0.5) * 2.0).round() as i64, 2),
        ));
    } else if remaining >= 0.499 {
        // One breath of echo: the opening tone, a step lower.
        notes.push(MotifNote::new(hook.notes[0].0 - 1, Duration::new(1, 2)));
    } else if remaining > 1e-9 {
        // Sub-half-beat remainder (unusual meters): template tissue.
        let mut rem = remaining;
        for n in template.notes.iter().rev() {
            if rem <= 1e-9 {
                break;
            }
            let take = n.duration.beats().min(rem);
            let dur = Duration::new((take * 4.0).round() as i64, 4);
            if dur.beats() > 1e-9 {
                notes.push(MotifNote {
                    degree: n.degree,
                    duration: dur,
                });
                rem -= dur.beats();
            }
        }
    }
    Motif::new(notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_generated_hook_has_identity_and_fits() {
        for meter in [3.0, 4.0] {
            for seed in 0..32 {
                let h = HookCell::generate(seed, meter);
                assert!(h.notes.len() >= 3, "seed {seed}: too short to be a name");
                assert!(h.has_rhythmic_identity(), "seed {seed}: uniform rhythm");
                assert!(h.has_contour_identity(), "seed {seed}: no gesture");
                assert!(h.beats() <= meter + 1e-9, "seed {seed}: overflows {meter}");
            }
        }
    }

    #[test]
    fn nearest_octave_degree_picks_the_closest_octave_equivalent() {
        assert_eq!(nearest_octave_degree(1, 10), 8); // 1, 8, 15... -> 8 is closest to 10
        assert_eq!(nearest_octave_degree(1, 11), 8); // |8-11|=3 < |15-11|=4
        assert_eq!(nearest_octave_degree(1, 12), 15); // |15-12|=3 < |8-12|=4
        assert_eq!(nearest_octave_degree(5, 5), 5); // already exact, no shift
        assert_eq!(nearest_octave_degree(3, -10), -11); // negative references work too
    }

    #[test]
    fn the_style_tail_splice_never_produces_an_unconstrained_leap() {
        // The real bug this fixes: a raw, un-normalized style-tail degree
        // could land many octaves from wherever the hook (itself possibly
        // transposed by HookCell::generate_with's combinatorial-variety
        // step) actually left off. Every splice must now stay within one
        // octave-ish of the echo note, regardless of seed or template.
        let templates = [
            Motif::from_degrees(&[(1, Duration::new(1, 1)); 4]),
            Motif::from_degrees(&[(8, Duration::new(1, 1)); 4]), // an octave-up template head
            Motif::from_degrees(&[(-6, Duration::new(1, 1)); 4]), // an octave-down template head
        ];
        for template in &templates {
            for seed in 0..32 {
                let hook = HookCell::generate(seed, 4.0);
                let grafted = graft_hook(template, &hook, 4.0);
                let n = hook.notes.len();
                if grafted.notes.len() <= n + 1 {
                    continue; // no room for both echo and style tissue
                }
                let echo = grafted.notes[n].degree.unwrap();
                let style_tissue = grafted.notes[n + 1].degree.unwrap();
                assert!(
                    (style_tissue - echo).abs() <= 3,
                    "seed {seed}: splice leap too large (echo={echo}, style_tissue={style_tissue})"
                );
            }
        }
    }

    #[test]
    fn hooks_vary_across_seeds_and_stay_deterministic() {
        let distinct: std::collections::BTreeSet<String> = (0..16)
            .map(|s| format!("{:?}", HookCell::generate(s, 4.0)))
            .collect();
        assert!(
            distinct.len() >= 6,
            "16 seeds gave only {} hooks — every piece would share a name",
            distinct.len()
        );
        assert_eq!(HookCell::generate(7, 4.0), HookCell::generate(7, 4.0));
    }

    #[test]
    fn every_hook_dwells_on_its_signature_moment() {
        // Reach alignment: the generated cell's longest note IS its
        // signature note (the target of the largest step) — "a reach that
        // lands on a short note is a gesture; a reach that lands on the
        // long note is a hook."
        for seed in 0..24 {
            let h = HookCell::generate(seed, 4.0);
            let longest = h
                .notes
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.1.beats().total_cmp(&b.1.1.beats()))
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(
                longest,
                h.signature_index(),
                "seed {seed}: the reach must be dwelt on"
            );
        }
    }

    #[test]
    fn the_bar_says_the_name_twice() {
        // The graft's tail is an ECHO of the head — its opening tones a
        // step lower — not anonymous template tissue.
        let template = Motif::from_degrees(&[
            (1, Duration::new(1, 1)),
            (2, Duration::new(1, 1)),
            (3, Duration::new(1, 1)),
            (2, Duration::new(1, 1)),
        ]);
        for seed in 0..8 {
            let hook = HookCell::generate(seed, 4.0);
            let grafted = graft_hook(&template, &hook, 4.0);
            let n = hook.notes.len();
            if grafted.notes.len() <= n {
                continue; // cell filled the whole bar — no room to echo
            }
            let tail: Vec<i32> = grafted.notes[n..].iter().filter_map(|m| m.degree).collect();
            // First tail tone: the echo (head a step lower). Any further
            // tail tone: style tissue from the template's head.
            assert_eq!(
                tail[0],
                hook.notes[0].0 - 1,
                "seed {seed}: the echo restates the head a step lower"
            );
            if tail.len() > 1 {
                // The style tissue quotes the template head (degree 1) --
                // but now octave-normalized to land near the echo (see
                // `nearest_octave_degree`), so it's degree 1 modulo a
                // diatonic octave (7 degrees), not necessarily literal 1.
                assert_eq!(
                    (tail[1] - 1).rem_euclid(7),
                    0,
                    "seed {seed}: the style tissue quotes the template head (octave-normalized): got {}",
                    tail[1]
                );
                assert!(
                    (tail[1] - tail[0]).abs() <= 3,
                    "seed {seed}: the octave-normalized splice should stay close to the echo \
                     (echo={}, style tissue={})",
                    tail[0],
                    tail[1]
                );
            }
        }
    }

    #[test]
    fn grafting_preserves_the_meter_and_leads_with_the_hook() {
        let template = Motif::from_degrees(&[
            (1, Duration::new(1, 1)),
            (2, Duration::new(1, 1)),
            (3, Duration::new(1, 1)),
            (2, Duration::new(1, 1)),
        ]);
        for seed in 0..8 {
            let hook = HookCell::generate(seed, 4.0);
            let grafted = graft_hook(&template, &hook, 4.0);
            let total: f64 = grafted.notes.iter().map(|n| n.duration.beats()).sum();
            assert!((total - 4.0).abs() < 1e-9, "seed {seed}: total {total}");
            // The head IS the hook.
            for (i, (deg, dur)) in hook.notes.iter().enumerate() {
                assert_eq!(grafted.notes[i].degree, Some(*deg));
                assert_eq!(grafted.notes[i].duration, *dur);
            }
        }
    }

    #[test]
    fn melodic_dna_separates_styles_at_the_hook() {
        // The tango review's experiment, pinned at the layer this wave
        // changed: "strip away the accompaniment... if I can't reliably
        // tell which style each melody came from, that's the bottleneck."
        // Tango hooks must reach WIDER than nocturne sighs, and the same
        // seed must usually birth different hooks under different DNA.
        let tango = crate::style::Style::Tango.spec().melody;
        let nocturne = crate::style::Style::Nocturne.spec().melody;
        let mean_interval = |dna: &crate::spec::MelodicDna| -> f64 {
            let (mut sum, mut n) = (0.0f64, 0u32);
            for seed in 0..16u64 {
                let h = HookCell::generate_with(dna, seed, 4.0);
                for w in h.notes.windows(2) {
                    sum += (w[1].0 - w[0].0).abs() as f64;
                    n += 1;
                }
            }
            sum / n as f64
        };
        let (t, noc) = (mean_interval(&tango), mean_interval(&nocturne));
        assert!(
            t > noc + 0.5,
            "tango hooks must reach wider than nocturne sighs: {t:.2} vs {noc:.2}"
        );
        let differ = (0..16u64)
            .filter(|&seed| {
                HookCell::generate_with(&tango, seed, 4.0)
                    != HookCell::generate_with(&nocturne, seed, 4.0)
            })
            .count();
        assert!(
            differ >= 12,
            "same seed must usually birth different hooks under different DNA ({differ}/16)"
        );
    }

    #[test]
    fn dna_hooks_carry_identity_and_bad_dna_degrades_gracefully() {
        use crate::style::Style;
        // DNA changes WHICH identities a style prefers, never whether the
        // hook has identity.
        for style in [Style::Tango, Style::Nocturne, Style::March] {
            let dna = style.spec().melody;
            for seed in 0..12u64 {
                let h = HookCell::generate_with(&dna, seed, 4.0);
                assert!(h.has_rhythmic_identity(), "{style:?} seed {seed}");
                assert!(h.has_contour_identity(), "{style:?} seed {seed}");
            }
        }
        // Garbage DNA (no rhythmic contrast, no contour signature) falls
        // back to the classic pools instead of shipping an identity-less
        // hook.
        let garbage = crate::spec::MelodicDna {
            hook_rhythms: vec![vec![(1, 1), (1, 1), (1, 1)]],
            hook_contours: vec![vec![1, 2, 3]],
            appoggiatura_rate: 0.0,
            ornament_rate: 0.0,
            blue_note_rate: 0.0,
            use_procedural_foundry: false,
        };
        let h = HookCell::generate_with(&garbage, 3, 4.0);
        assert!(h.has_rhythmic_identity() && h.has_contour_identity());
    }

    #[test]
    fn new_style_dna_pools_actually_get_used_not_silently_defaulted() {
        // The Diversity Truth pass gave 6 more styles (Classical/Waltz
        // share one spec, plus Cinematic/Playful/Folk/ModalFolk) their own
        // hook_rhythms/hook_contours, replacing the shared default pool
        // 12/29 styles used to draw from identically. The stronger check
        // than "has identity" (already covered above): every generated
        // hook's DEGREE SEQUENCE must match one of the style's own
        // authored hook_contours exactly — proving the DNA pool was
        // actually consulted, not silently degraded back to the classic
        // shared pool by a mistake in the authored rhythm/contour data
        // (e.g. a pairing that never fits the meter or never passes the
        // identity predicates).
        use crate::style::Style;
        for style in [
            Style::Classical,
            Style::Waltz,
            Style::Cinematic,
            Style::Playful,
            Style::Folk,
            Style::ModalFolk,
        ] {
            let dna = style.spec().melody;
            let meter = if style == Style::Waltz { 3.0 } else { 4.0 };
            let mut hit_own_pool = false;
            for seed in 0..40u64 {
                let h = HookCell::generate_with(&dna, seed, meter);
                let degrees: Vec<i32> = h.notes.iter().map(|&(d, _)| d).collect();
                if dna.hook_contours.contains(&degrees) {
                    hit_own_pool = true;
                }
            }
            assert!(
                hit_own_pool,
                "{style:?}: no seed in 0..40 drew from its own authored \
                 hook_contours — the DNA pool is silently inert"
            );
        }
    }

    #[test]
    fn empty_dna_is_byte_identical_to_the_classic_pools() {
        // The compatibility contract: styles/specs without DNA are
        // unchanged — generate() IS generate_with(default).
        for seed in 0..24u64 {
            for meter in [3.0, 4.0] {
                assert_eq!(
                    HookCell::generate(seed, meter),
                    HookCell::generate_with(&crate::spec::MelodicDna::default(), seed, meter)
                );
            }
        }
    }

    #[test]
    fn hook_variety_scales_beyond_the_authored_pool() {
        // The Listen-tab pigeonhole, pinned: the classic pools hold only
        // a dozen-odd valid cells, so before combinatorial variants,
        // distinct hooks over any seed range plateaued there. Variants
        // must lift 96 seeds well past the raw pool — and every variant
        // must still carry both identities.
        let mut distinct = std::collections::HashSet::new();
        for seed in 0..96u64 {
            let h = HookCell::generate(seed, 4.0);
            assert!(h.has_rhythmic_identity(), "seed {seed}");
            assert!(h.has_contour_identity(), "seed {seed}");
            distinct.insert(
                h.notes
                    .iter()
                    .map(|(d, dur)| (*d, dur.beats().to_bits()))
                    .collect::<Vec<_>>(),
            );
        }
        assert!(
            distinct.len() >= 30,
            "hook space too small: {} distinct over 96 seeds",
            distinct.len()
        );
        // Deterministic: the same seed is the same name, forever.
        assert_eq!(HookCell::generate(41, 4.0), HookCell::generate(41, 4.0));
    }
}
