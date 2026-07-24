// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The Identity Explorer: candidates as maximally-different musical
//! identities instead of seed-neighborhood variants.
//!
//! The listening review that demanded it (after a four-candidate batch):
//! "the four outputs aren't really four candidates. They're four
//! renderings of one compositional idea... your seeds are exploring a
//! neighborhood instead of different continents." Every Muse subsystem
//! asks "how do I preserve identity?"; this is the one that asks "how do
//! I offer a DIFFERENT one?"
//!
//! It was deliberately built AFTER melodic DNA and phrase rhetoric — the
//! same review, one wave later: "clustering before this wave would have
//! mostly found variations of the same melodic language. Now I think it
//! can discover genuinely different musical identities."
//!
//! Mechanism (all deterministic, no scoring model): every seed in a
//! window implies an IDENTITY — the hook it births, the form it picks,
//! the accompaniment, the ensemble, the motif head. [`explore_identities`]
//! featurizes the window, then greedily selects the requested number of
//! seeds by MAX-MIN distance (each pick maximizes its distance to the
//! nearest already-picked identity). The user's own base seed is always
//! the first pick — exploration widens the offer, it never overrides the
//! artist's chosen starting point.
//!
//! Novelty here is bounded diversity, not randomness: every candidate
//! still comes from the same spec (same style DNA, same rhetoric, same
//! grammars) — the explorer chooses among the identities the style can
//! conceive, as far apart as the window allows.

use crate::composer::MusicalIntent;
use crate::hook::HookCell;
use crate::spec::CompositionSpec;

/// How many seeds the explorer examines per requested candidate (window =
/// `n * SEEDS_PER_SLOT`, min 64). Wide enough that hook pools, form pools
/// and accompaniment pools all cycle several times.
const SEEDS_PER_SLOT: usize = 16;

/// The featurized identity a single seed implies under a spec — including
/// the PREMISE the seed selects (tempo third, texture tier, length, mode):
/// the listening analysis showed those uniforms, not the notes, are what
/// makes a batch feel samey, so they carry heavy weight in the distance.
struct Identity {
    seed: u64,
    hook: HookCell,
    form: usize,
    accompaniment: crate::accompaniment::Accompaniment,
    ensemble: usize,
    motif_head: Vec<i32>,
    tempo_third: usize,
    texture_tier: crate::premise::TextureTier,
    bars_multiplier: usize,
    mode: Option<crate::scale::Mode>,
    meter: u8,
    /// The actual chord-degree sequence this seed's progression source
    /// produces — real harmonic content, not just the mode label. The
    /// original review named "harmonic distance" as its own channel
    /// alongside melodic/rhythmic/orchestration; this was the one still
    /// missing.
    progression_degrees: Vec<i32>,
}

fn identity_of(spec: &CompositionSpec, intent: &MusicalIntent, seed: u64) -> Identity {
    let meter = spec.meter as f64;
    let premise = crate::premise::premise_for(spec, seed);
    Identity {
        seed,
        hook: HookCell::generate_with(&spec.melody, seed, meter),
        form: (seed % spec.form_pool.len().max(1) as u64) as usize,
        accompaniment: spec.accompaniment(seed),
        ensemble: (seed % spec.ensemble_pool.len().max(1) as u64) as usize,
        motif_head: spec
            .motif(intent.arousal, seed)
            .notes
            .iter()
            .filter_map(|n| n.degree)
            .take(4)
            .collect(),
        tempo_third: premise.tempo_third,
        texture_tier: premise.texture_tier,
        bars_multiplier: premise.bars_multiplier,
        mode: premise.spec.mode,
        meter: premise.meter,
        progression_degrees: spec.progression(intent.bars.max(1), seed).degrees,
    }
}

/// The named channels of identity distance — the review's request made
/// concrete: "imagine every candidate reports identity distance, rhythmic
/// distance, harmonic distance, melodic distance, orchestration distance."
/// Channels here match what the engine can honestly measure symbolically.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct NoveltyBreakdown {
    /// Hook contour + first developed material (the melodic thought).
    pub melodic: f64,
    /// Hook rhythm profile.
    pub rhythmic: f64,
    /// The premise: tempo third, texture tier, length, mode, meter, form.
    pub premise: f64,
    /// Ensemble + accompaniment (the sound-world).
    pub orchestration: f64,
    /// The actual chord-degree sequence the two seeds' progressions
    /// produce — real root motion, not just the mode label (which
    /// `premise` already covers). Honestly near-zero for Archetype-
    /// sourced styles across most seed pairs (the archetype barely
    /// depends on seed beyond cycling) and genuinely large for Grammar-
    /// sourced ones — that asymmetry is itself informative, not a bug.
    pub harmonic: f64,
    /// The weighted whole (what the Explorer optimizes).
    pub overall: f64,
}

fn channels(a: &Identity, b: &Identity) -> NoveltyBreakdown {
    let (ar, br) = (&a.hook.notes, &b.hook.notes);
    let aligned = ar.len().min(br.len());
    let mut rhythm = 0.0;
    for i in 0..aligned {
        rhythm += (ar[i].1.beats() - br[i].1.beats()).abs();
    }
    rhythm = rhythm / aligned.max(1) as f64 + (ar.len() as f64 - br.len() as f64).abs() * 0.5;
    let mut contour = 0.0;
    for i in 0..aligned {
        contour += (ar[i].0 - br[i].0).abs() as f64;
    }
    contour /= aligned.max(1) as f64;
    let mh = a
        .motif_head
        .iter()
        .zip(b.motif_head.iter())
        .filter(|(x, y)| x != y)
        .count() as f64;
    let melodic = contour * 0.6 + mh * 0.15;
    let rhythmic = rhythm;
    let premise = if a.form != b.form { 0.8 } else { 0.0 }
        + if a.tempo_third != b.tempo_third {
            0.7
        } else {
            0.0
        }
        + if a.texture_tier != b.texture_tier {
            0.7
        } else {
            0.0
        }
        + if a.bars_multiplier != b.bars_multiplier {
            0.9
        } else {
            0.0
        }
        + if a.mode != b.mode { 0.8 } else { 0.0 }
        + if a.meter != b.meter { 1.0 } else { 0.0 };
    let orchestration = if a.accompaniment != b.accompaniment {
        0.6
    } else {
        0.0
    } + if a.ensemble != b.ensemble { 0.4 } else { 0.0 };
    let (pa, pb) = (&a.progression_degrees, &b.progression_degrees);
    let p_aligned = pa.len().min(pb.len());
    let mut differing = 0.0;
    for i in 0..p_aligned {
        if pa[i] != pb[i] {
            differing += 1.0;
        }
    }
    let harmonic =
        differing / p_aligned.max(1) as f64 * 2.0 + (pa.len() as f64 - pb.len() as f64).abs() * 0.3;
    NoveltyBreakdown {
        melodic,
        rhythmic,
        premise,
        orchestration,
        harmonic,
        overall: melodic + rhythmic + premise + orchestration + harmonic,
    }
}

/// Distance between two identities: the weighted whole of [`channels`].
fn distance(a: &Identity, b: &Identity) -> f64 {
    channels(a, b).overall
}

/// Identity distance between two (possibly unrelated) compositions — each
/// resolved from its OWN spec/intent/seed, so this is the cross-history
/// counterpart to [`novelty_within`]'s within-batch comparison: e.g.
/// checking a fresh candidate against a keeper from a past session (a
/// different style, premise, or seed entirely). `Identity`/`identity_of`
/// stay private; this is the narrow public surface a caller needs.
pub fn identity_distance(
    spec_a: &CompositionSpec,
    intent_a: &MusicalIntent,
    seed_a: u64,
    spec_b: &CompositionSpec,
    intent_b: &MusicalIntent,
    seed_b: u64,
) -> NoveltyBreakdown {
    channels(
        &identity_of(spec_a, intent_a, seed_a),
        &identity_of(spec_b, intent_b, seed_b),
    )
}

/// Per-candidate novelty within a batch: each seed's channel distances to
/// its NEAREST batch neighbor (nearest by overall distance — the same
/// worst-pair semantics the Explorer optimizes). An observable for cards
/// and taste logs, NEVER a fitness function — the standing policy.
pub fn novelty_within(
    spec: &CompositionSpec,
    intent: &MusicalIntent,
    seeds: &[u64],
) -> Vec<NoveltyBreakdown> {
    let ids: Vec<Identity> = seeds
        .iter()
        .map(|&s| identity_of(spec, intent, s))
        .collect();
    ids.iter()
        .enumerate()
        .map(|(i, a)| {
            ids.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, b)| channels(a, b))
                .min_by(|x, y| x.overall.total_cmp(&y.overall))
                .unwrap_or(NoveltyBreakdown {
                    melodic: 0.0,
                    rhythmic: 0.0,
                    premise: 0.0,
                    orchestration: 0.0,
                    harmonic: 0.0,
                    overall: 0.0,
                })
        })
        .collect()
}

/// Select `n` seeds whose identities are maximally different, from a
/// deterministic window starting at the intent's own seed. The base seed
/// is ALWAYS the first pick. Greedy max-min: each subsequent pick
/// maximizes its distance to the nearest already-picked identity (ties
/// broken toward the earlier seed, keeping results stable).
pub fn explore_identities(spec: &CompositionSpec, intent: &MusicalIntent, n: usize) -> Vec<u64> {
    let n = n.max(1);
    let window = (n * SEEDS_PER_SLOT).max(64);
    let pool: Vec<Identity> = (0..window as u64)
        .map(|k| identity_of(spec, intent, intent.seed.wrapping_add(k)))
        .collect();
    let mut picked: Vec<usize> = vec![0]; // the artist's own seed leads
    while picked.len() < n {
        let mut best: Option<(usize, f64)> = None;
        for (i, cand) in pool.iter().enumerate() {
            if picked.contains(&i) {
                continue;
            }
            let nearest = picked
                .iter()
                .map(|&p| distance(cand, &pool[p]))
                .fold(f64::MAX, f64::min);
            if best.map(|(_, d)| nearest > d + 1e-12).unwrap_or(true) {
                best = Some((i, nearest));
            }
        }
        match best {
            Some((i, _)) => picked.push(i),
            None => break,
        }
    }
    // Local improvement: greedy max-min is only a 1/2-approximation — its
    // sequential choices can strand the SET's minimum below what a swap
    // would achieve (found by the acceptance test itself: one Classical
    // window where plain consecutive seeds beat the greedy pick). Try
    // swapping each non-base pick for any unpicked candidate whenever it
    // strictly raises the set's minimum pairwise distance; iterate to a
    // fixpoint (bounded). Deterministic scan order keeps results stable.
    let set_min = |picked: &[usize], pool: &[Identity]| -> f64 {
        let mut min = f64::MAX;
        for i in 0..picked.len() {
            for j in (i + 1)..picked.len() {
                min = min.min(distance(&pool[picked[i]], &pool[picked[j]]));
            }
        }
        min
    };
    // The status quo is itself a candidate: if the plain consecutive
    // window (the Studio's old behavior, base seed leading) has a better
    // worst-pair than the greedy pick, start from IT — the explorer must
    // never offer less diversity than the neighborhood it replaces. (The
    // acceptance test found real windows where hill-climbing from the
    // greedy start alone got stuck below the consecutive baseline.)
    let consecutive: Vec<usize> = (0..picked.len().min(pool.len())).collect();
    if set_min(&consecutive, &pool) > set_min(&picked, &pool) + 1e-12 {
        picked = consecutive;
    }
    for _ in 0..32 {
        let mut improved = false;
        for slot in 1..picked.len() {
            let current = set_min(&picked, &pool);
            let mut best_swap: Option<(usize, f64)> = None;
            for cand in 0..pool.len() {
                if picked.contains(&cand) {
                    continue;
                }
                let saved = picked[slot];
                picked[slot] = cand;
                let m = set_min(&picked, &pool);
                picked[slot] = saved;
                if m > current + 1e-9 && best_swap.map(|(_, b)| m > b + 1e-12).unwrap_or(true) {
                    best_swap = Some((cand, m));
                }
            }
            if let Some((cand, _)) = best_swap {
                picked[slot] = cand;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    picked.into_iter().map(|i| pool[i].seed).collect()
}

/// The MINIMUM pairwise identity distance of a seed set — the diversity
/// guarantee the explorer optimizes and its acceptance test pins. The
/// review's complaint ("four renderings of one compositional idea") is a
/// NEAR-DUPLICATE PAIR problem, so the worst pair is the honest measure;
/// mean pairwise distance can be lucked into by any selection that cycles
/// the modular pools (consecutive seeds do) while still containing twins.
pub fn seed_set_min_diversity(
    spec: &CompositionSpec,
    intent: &MusicalIntent,
    seeds: &[u64],
) -> f64 {
    if seeds.len() < 2 {
        return 0.0;
    }
    let ids: Vec<Identity> = seeds
        .iter()
        .map(|&s| identity_of(spec, intent, s))
        .collect();
    let mut min = f64::MAX;
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            min = min.min(distance(&ids[i], &ids[j]));
        }
    }
    min
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Style;

    fn intent(seed: u64) -> MusicalIntent {
        MusicalIntent {
            seed,
            ..Default::default()
        }
    }

    #[test]
    fn explorer_leads_with_the_artists_seed_and_returns_n_distinct() {
        let spec = Style::Classical.spec();
        for base in [0u64, 7, 69] {
            let seeds = explore_identities(&spec, &intent(base), 4);
            assert_eq!(seeds.len(), 4);
            assert_eq!(seeds[0], base, "the artist's seed always leads");
            let mut uniq = seeds.clone();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), 4, "picks must be distinct");
        }
    }

    #[test]
    fn explorer_out_diversifies_consecutive_seeds() {
        // THE acceptance test, from the review: "your seeds are exploring
        // a neighborhood instead of different continents." The claim the
        // greedy max-min selection actually makes is about the WORST pair
        // (no near-duplicates), so that is what's pinned: the explorer's
        // closest pair is never closer than the consecutive window's
        // closest pair, and strictly farther in aggregate. (The first
        // draft asserted MEAN pairwise distance and failed on a near-tie
        // — consecutive seeds cycle every modular pool by construction,
        // so their mean can luck high while still containing twins.)
        let (mut e_sum, mut c_sum) = (0.0f64, 0.0f64);
        for style in [Style::Classical, Style::Tango, Style::Nocturne] {
            let spec = style.spec();
            for base in [0u64, 5, 42] {
                let it = intent(base);
                let explored = explore_identities(&spec, &it, 4);
                let consecutive: Vec<u64> = (0..4).map(|k| base + k).collect();
                let de = seed_set_min_diversity(&spec, &it, &explored);
                let dc = seed_set_min_diversity(&spec, &it, &consecutive);
                assert!(
                    de >= dc - 1e-9,
                    "{style:?} base {base}: explorer's closest pair {de:.3} must not be \
                     closer than consecutive's {dc:.3}"
                );
                e_sum += de;
                c_sum += dc;
            }
        }
        assert!(
            e_sum > c_sum,
            "aggregate min-diversity must strictly improve: {e_sum:.3} vs {c_sum:.3}"
        );
    }

    #[test]
    fn explorer_is_deterministic() {
        let spec = Style::Tango.spec();
        assert_eq!(
            explore_identities(&spec, &intent(9), 6),
            explore_identities(&spec, &intent(9), 6)
        );
    }

    #[test]
    fn explored_hooks_are_pairwise_distinct_when_the_pool_allows() {
        // Four picks from a window of 64 should birth four different
        // HOOKS (the name is the identity's core) whenever the style's
        // hook pool has at least four cells.
        let spec = Style::Classical.spec();
        let seeds = explore_identities(&spec, &intent(3), 4);
        let hooks: Vec<_> = seeds
            .iter()
            .map(|&s| crate::hook::HookCell::generate_with(&spec.melody, s, spec.meter as f64))
            .collect();
        for i in 0..hooks.len() {
            for j in (i + 1)..hooks.len() {
                assert_ne!(hooks[i], hooks[j], "picks {i} and {j} share a hook");
            }
        }
    }

    #[test]
    fn novelty_channels_sum_to_the_overall_and_batch_is_positive() {
        let spec = Style::Classical.spec();
        let it = intent(10);
        let seeds = explore_identities(&spec, &it, 4);
        let nov = novelty_within(&spec, &it, &seeds);
        assert_eq!(nov.len(), 4);
        for n in &nov {
            assert!(
                (n.melodic + n.rhythmic + n.premise + n.orchestration + n.harmonic - n.overall)
                    .abs()
                    < 1e-9,
                "channels must decompose the overall"
            );
            assert!(n.overall > 0.0, "an explored batch has no twins");
        }
        // Deterministic.
        assert_eq!(nov, novelty_within(&spec, &it, &seeds));
    }

    #[test]
    fn harmonic_channel_reflects_the_real_progression_source() {
        // The review's own original request named "harmonic distance" as
        // a channel alongside melodic/rhythmic/orchestration — this is
        // the one that was still missing. Grammar-sourced progressions
        // (Classical) genuinely vary their chord-degree sequence by seed,
        // so real seed pairs must show nonzero harmonic distance; an
        // Archetype-sourced style's FIXED list barely depends on seed at
        // all, so most pairs are honestly near-zero — asserting that
        // asymmetry directly, not papering over it.
        let classical = Style::Classical.spec();
        assert_eq!(classical.progression, crate::spec::ProgressionSpec::Grammar);
        let (a, b) = (
            identity_of(&classical, &intent(0), 1),
            identity_of(&classical, &intent(0), 2),
        );
        assert_ne!(
            a.progression_degrees, b.progression_degrees,
            "grammar-sourced progressions must genuinely vary by seed"
        );
        assert!(
            channels(&a, &b).harmonic > 0.0,
            "a real degree-sequence difference must show up as harmonic distance"
        );

        let tango = Style::Tango.spec();
        assert!(matches!(
            tango.progression,
            crate::spec::ProgressionSpec::Archetype(_)
        ));
        let (ta, tb) = (
            identity_of(&tango, &intent(0), 1),
            identity_of(&tango, &intent(0), 2),
        );
        assert_eq!(
            ta.progression_degrees, tb.progression_degrees,
            "a fixed archetype cycled to the same length is identical across seeds"
        );
        assert_eq!(
            channels(&ta, &tb).harmonic,
            0.0,
            "identical progressions must report zero harmonic distance, honestly"
        );
    }

    #[test]
    fn identity_distance_matches_within_batch_channels_and_is_symmetric_on_identity() {
        let spec = Style::Classical.spec();
        let a = identity_of(&spec, &intent(0), 0);
        let b = identity_of(&spec, &intent(0), 1);
        assert_eq!(
            identity_distance(&spec, &intent(0), 0, &spec, &intent(0), 1),
            channels(&a, &b),
            "cross-history distance must agree with the within-batch calculation \
             it's built from, for the same spec/intent/seed inputs"
        );
        assert_eq!(
            identity_distance(&spec, &intent(0), 5, &spec, &intent(0), 5).overall,
            0.0,
            "a composition compared against itself has zero identity distance"
        );
    }
}
