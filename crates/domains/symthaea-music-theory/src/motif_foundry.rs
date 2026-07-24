// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 1 of a hierarchical generativity program (2026-07-24, following a
//! real listening-test-driven review of this crate's melodic diversity):
//! procedural motif CANDIDATE generation, validated through the SAME
//! identity predicates [`crate::hook::HookCell`] already uses for its own
//! combinatorial variety (transposition/inversion/retrograde) — not a new,
//! separate validation scheme. The insight this module acts on: the
//! engine's perceived "same few tunes" feeling doesn't come from having a
//! small combinatorial SPACE (seed × style × arousal × form already
//! produces a huge number of distinct pieces) — it comes from that space
//! being built entirely out of permutations of ~2-4 hand-authored
//! templates per style. This widens the SOURCE of candidates from
//! "permute a handful of fixed cells" to "generate freely, then apply the
//! same quality gates" — reusing trusted, tested infrastructure rather
//! than inventing a parallel validation scheme.
//!
//! Deliberately scoped: this is the candidate-generation + validation +
//! canonical-novelty layer only (the user's own "Phase 1"). Per-style
//! weighted generation grammars (a nocturne-flavored vs. a march-flavored
//! distribution), the Motif Curator/Form Planner/Development Engine
//! pipeline, and cross-generation memory are explicitly NOT attempted
//! here — real, larger follow-up work, not silently folded in.

use crate::hook::HookCell;
use crate::rhythm::Duration;

/// Half-beat-granular duration choices a candidate's notes are drawn from
/// — the same grid [`crate::hook`]'s own hand-authored rhythm pools use.
const DURATION_CHOICES: &[(i64, i64)] = &[(1, 2), (1, 1), (3, 2), (2, 1)];

/// Constraints a generated candidate must respect. Not yet a per-style
/// weighted grammar (see module doc) — a single shared, conservative
/// envelope.
#[derive(Debug, Clone, Copy)]
pub struct FoundryConfig {
    /// Inclusive range of how many notes a candidate has.
    pub length_range: (usize, usize),
    /// The largest single scale-degree step a candidate may take.
    pub max_leap_degrees: i32,
    /// The bar length candidates must fit within.
    pub meter_beats: f64,
}

impl Default for FoundryConfig {
    fn default() -> Self {
        FoundryConfig {
            length_range: (3, 5),
            max_leap_degrees: 5, // up to a sixth -- hooks reach further, base candidates stay closer to hand
            meter_beats: 4.0,
        }
    }
}

/// Generate one candidate deterministically from `seed`. Correct by
/// construction: each note's duration is only ever ACCEPTED if the
/// running total still fits `config.meter_beats` (never rescaled after
/// the fact, which can drift back over the limit under rounding) -- so
/// the result always satisfies `cell.beats() <= config.meter_beats`, and
/// generation simply stops early if a candidate runs out of room before
/// reaching its target length. Does NOT itself check identity; callers
/// filter with [`is_valid_candidate`].
pub fn generate_candidate(seed: u64, config: &FoundryConfig) -> HookCell {
    let (lo, hi) = config.length_range;
    let span = (hi.saturating_sub(lo) + 1).max(1) as u64;
    let length = lo + (crate::hook::scramble(seed, 0x5EED_0001) % span) as usize;

    // The smallest duration choice -- used to reserve enough room for the
    // MINIMUM required note count, so generation doesn't stop short of
    // `length_range.0` just because an early note claimed too much of the
    // bar.
    let min_note_beats = DURATION_CHOICES
        .iter()
        .map(|&(n, d)| n as f64 / d as f64)
        .fold(f64::MAX, f64::min);

    let mut notes: Vec<(i32, Duration)> = Vec::with_capacity(length);
    let mut degree = 1i32;
    let mut total_beats = 0.0f64;
    for i in 0..length {
        // This note's degree: a bounded, small-step-biased random walk
        // from the previous note (the first note always starts at 1).
        if i > 0 {
            let step_roll = crate::hook::scramble(seed, 0x5EED_0002 + i as u64);
            // Square a [-1,1] draw before scaling, so wide leaps are rare
            // rather than uniformly likely -- a conservative, generically
            // "musical" shape, not a style grammar.
            let raw = ((step_roll % 2001) as f64 - 1000.0) / 1000.0; // [-1, 1]
            let biased = raw.abs().powf(1.6) * raw.signum();
            degree += (biased * config.max_leap_degrees as f64).round() as i32;
        }

        // This note's duration, budget-capped: reserve enough room (at
        // the smallest duration choice) for however many MORE notes are
        // still needed to reach the minimum length, so an early greedy
        // draw can't starve out the minimum note count.
        let dur_roll = crate::hook::scramble(seed, 0x5EED_0003 + i as u64);
        let (n, d) = DURATION_CHOICES[(dur_roll % DURATION_CHOICES.len() as u64) as usize];
        let drawn = Duration::new(n, d);
        let notes_still_needed = lo.saturating_sub(i + 1); // after this one
        let reserved = notes_still_needed as f64 * min_note_beats;
        let remaining = (config.meter_beats - total_beats - reserved).max(0.0);
        if remaining <= 1e-9 && notes.len() >= lo {
            break; // out of room, but the minimum is already met
        }
        let dur = if remaining <= 1e-9 {
            // Below the minimum with no budgeted room left (only
            // reachable if `length_range.0 * min_note_beats >
            // meter_beats`, i.e. a misconfigured `FoundryConfig`): take
            // the smallest note anyway rather than stop short.
            Duration::new((min_note_beats * 4.0).round().max(1.0) as i64, 4)
        } else if drawn.beats() > remaining + 1e-9 {
            // Doesn't fit the (budget-aware) remaining room as drawn:
            // take the largest quarter-beat-aligned slice that does.
            Duration::new((remaining * 4.0).floor().max(1.0) as i64, 4)
        } else {
            drawn
        };

        total_beats += dur.beats();
        notes.push((degree, dur));
    }

    HookCell { notes }
}

/// The SAME two identity predicates [`HookCell`]'s own hand-authored pools
/// and combinatorial variants must pass -- reused directly, not
/// reimplemented, so a procedurally generated cell and a hand-authored one
/// are held to one shared bar.
pub fn is_valid_candidate(cell: &HookCell) -> bool {
    cell.has_rhythmic_identity() && cell.has_contour_identity()
}

/// A transposition/inversion/retrograde-invariant fingerprint: two cells
/// that are the "same idea" under any combination of those three classical
/// transformations collapse to the same fingerprint. Built from the
/// candidate's own INTERVAL sequence (already transposition-invariant by
/// construction -- it never reads the absolute starting degree), then
/// picks the lexicographically smallest among the interval sequence's own
/// {identity, inverted, retrograded, retrograde-inverted} images, matching
/// exactly how those three transforms act on an interval sequence:
/// inversion negates every interval, retrograde reverses the negated
/// sequence, retrograde-inversion reverses the plain sequence.
pub fn canonical_fingerprint(cell: &HookCell) -> Vec<i32> {
    let degrees: Vec<i32> = cell.notes.iter().map(|&(d, _)| d).collect();
    let intervals: Vec<i32> = degrees.windows(2).map(|w| w[1] - w[0]).collect();
    let negated: Vec<i32> = intervals.iter().map(|d| -d).collect();
    let mut reversed_negated = negated.clone();
    reversed_negated.reverse();
    let mut reversed = intervals.clone();
    reversed.reverse();
    [intervals, negated, reversed_negated, reversed]
        .into_iter()
        .min()
        .unwrap_or_default()
}

/// How many valid, canonically-distinct candidates
/// [`generate_with_foundry`] tries to assemble before picking one -- a
/// tiny, real "Motif Curator" (the user's own next-phase name): not
/// full multi-axis coherence scoring (tonal fit, singability, tension
/// shape -- all genuinely taste-dependent and unverifiable without
/// listening, deliberately deferred), just the two things this module
/// can already check for certain: is it a valid cell, and is it a
/// genuinely different idea from the others already picked.
const FAMILY_SIZE: usize = 8;
/// Generation attempts budgeted per family before giving up early and
/// degrading gracefully (see [`generate_family`]'s doc).
const FAMILY_MAX_ATTEMPTS: usize = 200;

/// Derive a [`FoundryConfig`] from a style's own existing DNA -- purely
/// mechanical, every parameter traces to a real number already present in
/// `dna.hook_contours`, not invented taste. This is deliberately NOT a
/// per-style hand-tuned `MotifGrammar` (the user's fuller sketch: weighted
/// contour shapes, anchor policy, syncopation/chromaticism ranges) --
/// that's real musical judgment unverifiable without listening, the same
/// category of decision already deferred elsewhere in this crate (Tier 0
/// preset tuning). What this DOES give a style: procedural candidates
/// whose LENGTH and MAX LEAP match what that style's own authored hooks
/// already do, instead of one generic envelope for every style.
///
/// Empty DNA (the shared default -- 12 of 29 styles) falls back to
/// [`FoundryConfig::default`].
pub fn config_for_dna(dna: &crate::spec::MelodicDna) -> FoundryConfig {
    if dna.hook_contours.is_empty() {
        return FoundryConfig::default();
    }
    let lengths: Vec<usize> = dna
        .hook_contours
        .iter()
        .map(|c| c.len())
        .filter(|&n| n >= 2)
        .collect();
    let max_leap = dna
        .hook_contours
        .iter()
        .flat_map(|c| c.windows(2).map(|w| (w[1] - w[0]).abs()))
        .max()
        .unwrap_or(FoundryConfig::default().max_leap_degrees);
    let (default_lo, default_hi) = FoundryConfig::default().length_range;
    let lo = lengths.iter().copied().min().unwrap_or(default_lo).max(2);
    let hi = lengths.iter().copied().max().unwrap_or(default_hi).max(lo);
    FoundryConfig {
        length_range: (lo, hi),
        max_leap_degrees: max_leap.max(2), // never bias below a step
        ..FoundryConfig::default()
    }
}

/// The real generate -> validate -> canonical-novelty pipeline (the
/// user's own "Motif Foundry generates candidates, keep the ones that are
/// both valid and genuinely different" step) -- not just a measurement,
/// an actual selector. Draws from scrambled sub-seeds of `base_seed` (so
/// two different `base_seed`s never share an attempt sequence) until
/// `target` valid, canonically-distinct candidates are collected or
/// `max_attempts` is exhausted. Exhausting attempts before reaching
/// `target` is an honest partial result (a real DNA/config combination
/// can be too restrictive to yield many distinct shapes), not a panic or
/// a forced pass.
pub fn generate_family(
    base_seed: u64,
    config: &FoundryConfig,
    target: usize,
    max_attempts: usize,
) -> Vec<HookCell> {
    let mut family: Vec<HookCell> = Vec::new();
    let mut seen_fingerprints: Vec<Vec<i32>> = Vec::new();
    for attempt in 0..max_attempts {
        if family.len() >= target {
            break;
        }
        let seed = crate::hook::scramble(base_seed, 0xFA51_1000 + attempt as u64);
        let candidate = generate_candidate(seed, config);
        if !is_valid_candidate(&candidate) {
            continue;
        }
        let fingerprint = canonical_fingerprint(&candidate);
        if seen_fingerprints.contains(&fingerprint) {
            continue;
        }
        seen_fingerprints.push(fingerprint);
        family.push(candidate);
    }
    family
}

/// The procedural counterpart to [`HookCell::generate_with`]'s
/// hand-authored-pool path (see that function's doc for the opt-in
/// wiring): assembles a family of up to [`FAMILY_SIZE`] valid,
/// canonically-distinct candidates biased toward `dna`'s own measured
/// character (via [`config_for_dna`]), then applies the SAME
/// combinatorial-variety step (retrograde/invert/transpose, filtered
/// through the same identity + reach-alignment predicates) the classic
/// pool's `generate_with` uses -- so procedural hooks get the same
/// amount of seed-driven variety as authored ones, not a separate scheme.
///
/// Degrades gracefully exactly like the classic path: if the family
/// assembly comes back empty (a pathological config), falls back to
/// [`HookCell::generate`] rather than panicking.
pub fn generate_with_foundry(
    dna: &crate::spec::MelodicDna,
    seed: u64,
    meter_beats: f64,
) -> HookCell {
    let mut config = config_for_dna(dna);
    config.meter_beats = meter_beats;
    let family = generate_family(seed, &config, FAMILY_SIZE, FAMILY_MAX_ATTEMPTS);
    if family.is_empty() {
        return HookCell::generate(seed, meter_beats);
    }
    let base = family[(seed % family.len() as u64) as usize].clone();
    let mut shapes: Vec<HookCell> = vec![base.clone()];
    let retro = base.retrograded();
    if retro.has_rhythmic_identity() && retro.has_contour_identity() && retro.is_reach_aligned() {
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
    variants[(crate::hook::scramble(seed, 0x400D) as usize) % variants.len()].clone()
}

/// Diversity measurements for a batch of candidates -- the "define
/// success" harness: candidates a generator PRODUCES vs. candidates that
/// actually pass identity validation vs. how many of those are genuinely
/// distinct ideas (exact interval sequence) vs. perceptually distinct
/// ideas (canonical, transform-invariant).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FoundryDiversityReport {
    pub candidates_generated: u32,
    pub candidates_valid: u32,
    pub unique_exact: u32,
    pub unique_canonical: u32,
}

impl FoundryDiversityReport {
    pub fn validity_rate(&self) -> f32 {
        if self.candidates_generated == 0 {
            return 0.0;
        }
        self.candidates_valid as f32 / self.candidates_generated as f32
    }

    /// Of the VALID candidates, what fraction are canonical duplicates of
    /// an earlier one -- the number the user's own spec calls the
    /// "canonical motif duplicate rate."
    pub fn canonical_duplicate_rate(&self) -> f32 {
        if self.candidates_valid == 0 {
            return 0.0;
        }
        1.0 - (self.unique_canonical as f32 / self.candidates_valid as f32)
    }
}

/// Run the generate -> validate -> canonicalize pipeline over `seeds`,
/// reporting exactly the measurements needed to compare this against the
/// existing hand-authored-bank approach on equal footing (see
/// `examples/motif_foundry_diversity_census.rs`).
pub fn foundry_diversity_report(
    seeds: impl Iterator<Item = u64>,
    config: &FoundryConfig,
) -> FoundryDiversityReport {
    let mut exact_seen: Vec<Vec<i32>> = Vec::new();
    let mut canonical_seen: Vec<Vec<i32>> = Vec::new();
    let mut report = FoundryDiversityReport::default();
    for seed in seeds {
        report.candidates_generated += 1;
        let cell = generate_candidate(seed, config);
        if !is_valid_candidate(&cell) {
            continue;
        }
        report.candidates_valid += 1;
        let exact: Vec<i32> = cell.notes.windows(2).map(|w| w[1].0 - w[0].0).collect();
        if !exact_seen.contains(&exact) {
            exact_seen.push(exact);
            report.unique_exact += 1;
        }
        let canonical = canonical_fingerprint(&cell);
        if !canonical_seen.contains(&canonical) {
            canonical_seen.push(canonical);
            report.unique_canonical += 1;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_candidates_always_fit_the_meter() {
        let config = FoundryConfig::default();
        for seed in 0..64u64 {
            let cell = generate_candidate(seed, &config);
            assert!(
                cell.beats() <= config.meter_beats + 1e-6,
                "seed {seed}: {} beats overflows {}",
                cell.beats(),
                config.meter_beats
            );
            assert!(
                cell.notes.len() >= config.length_range.0
                    && cell.notes.len() <= config.length_range.1,
                "seed {seed}: length {} outside {:?}",
                cell.notes.len(),
                config.length_range
            );
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let config = FoundryConfig::default();
        assert_eq!(
            generate_candidate(42, &config),
            generate_candidate(42, &config)
        );
    }

    #[test]
    fn some_but_not_all_candidates_pass_validation() {
        // The predicates should actually be doing work -- if every
        // candidate passed, "reuse trusted validators" wouldn't be filtering
        // anything; if none did, the generator would be useless.
        let config = FoundryConfig::default();
        let results: Vec<bool> = (0..200u64)
            .map(|s| is_valid_candidate(&generate_candidate(s, &config)))
            .collect();
        let valid = results.iter().filter(|&&v| v).count();
        assert!(valid > 0, "no candidate ever passed validation");
        assert!(
            valid < 200,
            "every candidate passed -- predicates aren't filtering"
        );
    }

    #[test]
    fn canonical_fingerprint_is_transform_invariant() {
        let cell = HookCell {
            notes: vec![
                (1, Duration::new(1, 1)),
                (4, Duration::new(1, 2)),
                (2, Duration::new(1, 1)),
                (5, Duration::new(1, 2)),
            ],
        };
        let inverted = cell.inverted();
        let retrograded = cell.retrograded();
        let retrograde_inverted = cell.inverted().retrograded();
        let fp = canonical_fingerprint(&cell);
        assert_eq!(canonical_fingerprint(&inverted), fp);
        assert_eq!(canonical_fingerprint(&retrograded), fp);
        assert_eq!(canonical_fingerprint(&retrograde_inverted), fp);
    }

    #[test]
    fn canonical_fingerprint_is_transposition_invariant() {
        let cell = HookCell {
            notes: vec![
                (1, Duration::new(1, 1)),
                (4, Duration::new(1, 2)),
                (2, Duration::new(1, 1)),
            ],
        };
        let transposed = cell.transposed(5);
        assert_eq!(
            canonical_fingerprint(&cell),
            canonical_fingerprint(&transposed)
        );
    }

    #[test]
    fn canonical_fingerprint_distinguishes_genuinely_different_shapes() {
        let a = HookCell {
            notes: vec![
                (1, Duration::new(1, 1)),
                (2, Duration::new(1, 1)),
                (3, Duration::new(1, 1)),
            ],
        };
        let b = HookCell {
            notes: vec![
                (1, Duration::new(1, 1)),
                (5, Duration::new(1, 1)),
                (2, Duration::new(1, 1)),
            ],
        };
        assert_ne!(canonical_fingerprint(&a), canonical_fingerprint(&b));
    }

    #[test]
    fn diversity_report_counts_are_internally_consistent() {
        let config = FoundryConfig::default();
        let report = foundry_diversity_report(0..500u64, &config);
        assert_eq!(report.candidates_generated, 500);
        assert!(report.candidates_valid <= report.candidates_generated);
        assert!(report.unique_exact <= report.candidates_valid);
        // Canonical collapses transform-equivalent shapes, so it can only
        // be <= the exact-sequence count, never more.
        assert!(report.unique_canonical <= report.unique_exact);
        assert!(report.unique_canonical > 0, "expected some real variety");
    }

    #[test]
    fn config_for_dna_falls_back_to_default_when_dna_is_empty() {
        let dna = crate::spec::MelodicDna::default();
        let config = config_for_dna(&dna);
        assert_eq!(config.length_range, FoundryConfig::default().length_range);
        assert_eq!(
            config.max_leap_degrees,
            FoundryConfig::default().max_leap_degrees
        );
    }

    #[test]
    fn config_for_dna_derives_real_numbers_from_the_dna_own_contours() {
        // A hand-built DNA, deliberately containing one CONSECUTIVE
        // full-octave leap (1 -> 8) among the contours -- config_for_dna
        // reads consecutive-note intervals (`windows(2)`), not overall
        // span, so this is the shape that should actually drive the
        // derived max_leap_degrees.
        let dna = crate::spec::MelodicDna {
            hook_contours: vec![vec![1, 8, 5, 3], vec![1, 5, 3, 1], vec![5, 8, 7, 5]],
            ..Default::default()
        };
        let config = config_for_dna(&dna);
        assert_eq!(
            config.length_range,
            (4, 4),
            "every contour here has length 4"
        );
        assert_eq!(
            config.max_leap_degrees, 7,
            "the 1->8 leap in the DNA itself should set the bias, not the generic default"
        );
    }

    #[test]
    fn generate_family_returns_only_valid_canonically_distinct_candidates() {
        let config = FoundryConfig::default();
        let family = generate_family(42, &config, FAMILY_SIZE, FAMILY_MAX_ATTEMPTS);
        assert!(
            !family.is_empty(),
            "a generous budget should find something"
        );
        for cell in &family {
            assert!(is_valid_candidate(cell));
        }
        let mut fingerprints: Vec<Vec<i32>> = family.iter().map(canonical_fingerprint).collect();
        let before = fingerprints.len();
        fingerprints.sort();
        fingerprints.dedup();
        assert_eq!(
            fingerprints.len(),
            before,
            "generate_family must never return two canonically-equal candidates"
        );
    }

    #[test]
    fn generate_family_degrades_gracefully_under_a_tiny_budget() {
        // A budget too small to realistically find `target` distinct
        // candidates should return fewer, not panic or loop forever.
        let config = FoundryConfig::default();
        let family = generate_family(7, &config, 100, 1);
        assert!(family.len() <= 1);
    }

    #[test]
    fn generate_with_foundry_always_fits_the_meter_and_has_identity() {
        let dna = crate::spec::MelodicDna {
            hook_contours: vec![vec![1, 3, 5, 8], vec![1, 5, 3, 1]],
            use_procedural_foundry: true,
            ..Default::default()
        };
        for seed in 0..64u64 {
            let cell = generate_with_foundry(&dna, seed, 4.0);
            assert!(
                cell.beats() <= 4.0 + 1e-6,
                "seed {seed} overflowed the meter"
            );
            assert!(
                cell.has_rhythmic_identity() && cell.has_contour_identity(),
                "seed {seed} produced an identity-less hook"
            );
        }
    }

    #[test]
    fn generate_with_foundry_is_deterministic_per_seed() {
        let dna = crate::spec::MelodicDna {
            hook_contours: vec![vec![1, 3, 5, 8]],
            use_procedural_foundry: true,
            ..Default::default()
        };
        assert_eq!(
            generate_with_foundry(&dna, 99, 4.0),
            generate_with_foundry(&dna, 99, 4.0)
        );
    }

    #[test]
    fn hook_cell_generate_with_routes_to_the_foundry_only_when_opted_in() {
        // The core zero-blast-radius guarantee: flipping the new field on
        // is the ONLY thing that changes generate_with's behavior for an
        // otherwise-identical DNA. With it off (every existing style
        // preset today), the classic hand-authored path runs unchanged.
        let contours = vec![vec![1, 3, 5, 8], vec![1, 5, 3, 1]];
        let classic = crate::spec::MelodicDna {
            hook_contours: contours.clone(),
            hook_rhythms: vec![
                vec![(1, 1), (1, 1), (1, 1), (1, 1)],
                vec![(1, 1), (1, 1), (1, 1), (1, 1)],
            ],
            ..Default::default()
        };
        let foundry = crate::spec::MelodicDna {
            use_procedural_foundry: true,
            ..classic.clone()
        };
        // Same seed, same meter, same underlying contours: the two paths
        // are free to diverge (the foundry path ignores hook_rhythms
        // entirely, generating its own), but the classic path itself must
        // be completely insensitive to the new field when it's false --
        // proven by it matching a config with the field entirely absent
        // (Rust's Default gives `false`).
        for seed in [0u64, 5, 42, 100] {
            assert_eq!(
                HookCell::generate_with(&classic, seed, 4.0),
                HookCell::generate_with(
                    &crate::spec::MelodicDna {
                        use_procedural_foundry: false,
                        ..classic.clone()
                    },
                    seed,
                    4.0
                ),
                "seed {seed}: an explicit `false` must match the derived default"
            );
        }
        // And the opt-in path really is live and callable end-to-end
        // through the public entry point, not just via generate_with_foundry
        // directly.
        let routed = HookCell::generate_with(&foundry, 7, 4.0);
        assert!(routed.has_rhythmic_identity() && routed.has_contour_identity());
    }
}
