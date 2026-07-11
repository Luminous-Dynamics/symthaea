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

impl HookCell {
    /// Generate the piece's hook deterministically from its seed. Only
    /// candidates that FIT the meter (cell ≤ meter, leaving room for a
    /// tail) are considered, so every meter gets a valid name.
    pub fn generate(seed: u64, meter_beats: f64) -> HookCell {
        // All (rhythm, contour) pairs of equal length that fit the meter.
        let mut candidates: Vec<HookCell> = Vec::new();
        for r in RHYTHMS {
            for c in CONTOURS {
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
                debug_assert!(cell.has_rhythmic_identity());
                debug_assert!(cell.has_contour_identity());
                candidates.push(cell);
            }
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
        pool[(seed % pool.len().max(1) as u64) as usize].clone()
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
        notes.push(MotifNote::new(hook.notes[0].0 - 1, Duration::new(1, 2)));
        let style_deg = template
            .notes
            .first()
            .and_then(|n| n.degree)
            .unwrap_or(hook.notes[0].0);
        notes.push(MotifNote::new(
            style_deg,
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
                assert_eq!(
                    tail[1], 1,
                    "seed {seed}: the style tissue quotes the template head"
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
}
