// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Passacaglia (style-roadmap item 6, promoted by an experimental result):
//! variations over a GROUND — an invariant bass line the whole piece stands
//! on. "Everything changes, the foundation remains."
//!
//! **Why this style, now**: the species-counterpoint falsification showed
//! that musical Φ's bottleneck in a fugue is TEMPORAL — the weakest cut in
//! the integration graph runs between the piece's thirds, spanned only by
//! motif-continuity edges. A ground bass is the most direct attack on that
//! exact dimension: the same line, restated across every segment, threads
//! the piece's whole length. This module carries a pre-registered,
//! deterministic experiment (`invariant_ground_out_integrates_fresh_
//! grounds`) — if an invariant ground does NOT out-integrate a control
//! whose bass changes every cycle, the hypothesis dies in CI, exactly as
//! the last one did.
//!
//! **The ground remembers** (the listening review's design, verbatim: "I
//! would make the ground itself capable of remembering. Not just
//! repeating."): seven cycles form a memory arc —
//!
//! 1. **Ground alone** — the foundation stated bare.
//! 2. **+ melody** — the subject walks over it.
//! 3. **+ counter** — the texture fills.
//! 4. **Ground ALTERED** — one remembered alteration: its deepest note
//!    lifted a step (the memory softens the lowest point). The uppers
//!    turn inward (retrograde, calmer).
//! 5. **Ground RESTORED** — verbatim again, under the piece's PEAK
//!    (octave-up figuration): recognition as climax.
//! 6. **Ground FRAGMENTED** — only the first tone of each bar survives;
//!    the uppers echo in fragments. The withdrawal before the end.
//! 7. **Ground COMPLETE** — verbatim, tail bent to the tonic, a cadence
//!    above, and a small velocity lift: finally whole. (The same
//!    "judgment" grammar as the rondo's final return.)
//!
//! The ground itself is the piece's NAME: the (hook-grafted) subject
//! augmented to double length, closed by a lamento descent (6̂-5̂) and a
//! leading-tone pickup — so every cycle re-states the identity the rest
//! of the engine already develops.
//!
//! Upper voices are fitted against the ground with the species fitter
//! ([`crate::counterpoint::fit_against`]) — the ground is a literal cantus
//! firmus, the textbook case the fitter was built for. Thematic register:
//! the ground is NEVER fitted (it is the theme); cycle 4's alteration is a
//! composed memory device, not a fitting artifact.

use crate::counterpoint::{CantusEvent, fit_against};
use crate::form::{contrasting_transform, figuration_variation};
use crate::fugue::{emit, hold};
use crate::harmony::Key;
use crate::motif::{Motif, MotifNote};
use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, VoiceRole};

/// Cycles in the memory arc (see module docs).
pub(crate) const CYCLES: usize = 7;

/// Build the 4-bar ground from the subject: the subject AUGMENTED ×2
/// (bars 1-2 — the piece's name, slowed into a foundation), then the
/// lamento close (bar 3: 6̂ held; bar 4: 5̂ then a leading-tone 7̂ pickup
/// that resolves to the tonic at every cycle restart — in the harmonic-
/// minor scale this preset uses, a real leading tone).
pub(crate) fn ground_from(subject: &Motif, meter: u8) -> Motif {
    let mut notes = subject.scale_rhythm(2, 1).notes;
    let m = meter as i64;
    notes.push(MotifNote::new(6, Duration::new(m, 1)));
    notes.push(MotifNote::new(5, Duration::new(m - 1, 1)));
    notes.push(MotifNote::new(7, Duration::new(1, 1)));
    Motif { notes }
}

/// Cycle 4's remembered alteration: the ground's DEEPEST tone lifted one
/// step. One note changes; everything else is intact — an alteration the
/// ear reads as "the same line, remembered slightly differently."
pub(crate) fn altered_ground(ground: &Motif) -> Motif {
    let mut g = ground.clone();
    if let Some(idx) = (0..g.notes.len())
        .filter(|&i| g.notes[i].degree.is_some())
        .min_by_key(|&i| g.notes[i].degree.unwrap())
    {
        g.notes[idx].degree = Some(g.notes[idx].degree.unwrap() + 1);
    }
    g
}

/// Cycle 6's fragmentation: only the first pitched tone of each bar
/// survives (held for its bar's remainder as written — the rest of the
/// bar falls silent). The foundation reduced to its skeleton.
pub(crate) fn fragmented_ground(ground: &Motif, meter: u8) -> Motif {
    let bar_beats = meter as f64;
    let mut out = Vec::new();
    let mut t = 0.0;
    let mut bar_of_last_kept = usize::MAX;
    for n in &ground.notes {
        let bar = (t / bar_beats + 1e-9) as usize;
        if n.degree.is_some() && bar != bar_of_last_kept {
            out.push(*n);
            bar_of_last_kept = bar;
        } else {
            out.push(MotifNote::rest(n.duration));
        }
        t += n.duration.beats();
    }
    Motif { notes: out }
}

/// Compose the passacaglia: seven statements of the ground (bass, octave
/// 2), each cycle's upper material fitted against it. `fresh_grounds` is
/// the EXPERIMENT CONTROL: when true, every cycle's bass keeps the
/// ground's rhythm and register but changes its line (re-oriented and
/// re-transposed per cycle) — same density, no invariance. Production
/// always passes `false`; the control exists so the Φ hypothesis stays
/// falsifiable in-tree.
pub(crate) fn realize_passacaglia(
    key: Key,
    tempo_bpm: f32,
    meter: u8,
    subject: &Motif,
    seed: u64,
    fresh_grounds: bool,
) -> Score {
    let mut score = Score::new(key, tempo_bpm, meter);
    let scale = key.scale();
    let ground = ground_from(subject, meter);
    let ground_beats = ground.total_duration();
    let pivot = subject.notes.iter().find_map(|x| x.degree).unwrap_or(1);

    use Emphasis::{Cadential, Climax, Normal, PhraseStart};
    use VoiceRole::{Bass, CounterMelody, Melody};
    // The memory arc's long-range dynamics: establish, fill, alter (a
    // shade more present), RESTORE as the peak, withdraw, complete.
    const INTENSITY: [f32; CYCLES] = [0.8, 0.85, 0.9, 0.95, 1.15, 0.9, 0.95];

    let cantus_of = |m: &Motif, octave: i32, start: Duration| -> Vec<CantusEvent> {
        let mut t = start.beats();
        let mut out = Vec::new();
        for n in &m.notes {
            if let Some(d) = n.degree {
                out.push(CantusEvent {
                    onset: t,
                    duration: n.duration.beats(),
                    pitch: scale.degree_pitch(d, 2),
                });
            }
            t += n.duration.beats();
        }
        out
    };

    for cycle in 0..CYCLES {
        let start = Duration::new(ground_beats.num() * cycle as i64, ground_beats.den());
        let intensity = INTENSITY[cycle];

        // ── The ground for this cycle ────────────────────────────────────
        let mut g = if fresh_grounds {
            // Control: same rhythm, register, role, and density — but an
            // ANONYMOUS stepwise walk, different every cycle and unrelated
            // to the subject.
            //
            // CONTROL-DESIGN LESSON (the first draft of this control was
            // wrong in an instructive way): it built "fresh" grounds from
            // oriented()/transpose() transforms of the real ground — but
            // the ground is subject-derived, so that control's bass
            // explored the same transformation family the melody uses,
            // and it measured WITHIN NOISE OF THE INVARIANT ARM (seed 5:
            // Φ 0.0386 vs 0.0378, with HIGHER motif-channel overlap).
            // That is itself a finding — a bass that imitates the
            // melody's transformation family integrates about as well as
            // a strict ostinato — but it is a rival integration strategy,
            // not a null control. The hypothesis "an invariant,
            // identity-bearing ground threads the piece together" needs a
            // control with NEITHER invariance NOR subject identity.
            let mut state = seed ^ (cycle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut deg = 1i32;
            let notes = ground
                .notes
                .iter()
                .map(|n| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let step = ((state >> 33) % 5) as i32 - 2; // -2..=2
                    deg = (deg + step).clamp(-3, 8);
                    MotifNote {
                        degree: n.degree.map(|_| deg),
                        duration: n.duration,
                    }
                })
                .collect();
            Motif { notes }
        } else {
            match cycle {
                3 => altered_ground(&ground),
                5 => fragmented_ground(&ground, meter),
                _ => ground.clone(),
            }
        };
        if cycle == CYCLES - 1 {
            // The final statement closes on the tonic — the same tail-bend
            // every final entry gets so the piece can cadence.
            if let Some(last) = g.notes.iter_mut().rev().find(|n| n.degree.is_some()) {
                last.degree = Some(1);
            }
        }
        let emphasis = match cycle {
            0 => PhraseStart,
            4 => Climax, // the restoration IS the peak
            c if c == CYCLES - 1 => Cadential,
            _ => Normal,
        };
        emit(&mut score, &g, start, Bass, 2, intensity, emphasis);
        let cf = cantus_of(&g, 2, start);
        // The fitter must know each statement's TRUE onset (a bar within the
        // cycle), not the cycle start — fitting against the wrong moment of
        // the ground was a real bug the on-beat contract test caught.
        let fit = |line: &Motif, octave: i32, at: Duration| {
            fit_against(&cf, line, scale, octave, at.beats())
        };

        // ── Upper material per cycle (all fitted against the ground) ────
        let bar = |n: i64| start + Duration::new(meter as i64 * n, 1);
        match cycle {
            0 => {} // the foundation, bare
            1 => {
                // The subject walks over its own slowed self.
                for (i, step) in [0i32, 1, 2, 1].iter().enumerate() {
                    emit(
                        &mut score,
                        &fit(&subject.transpose(*step), 5, bar(i as i64)),
                        bar(i as i64),
                        Melody,
                        5,
                        intensity,
                        if i == 0 { PhraseStart } else { Normal },
                    );
                }
            }
            2 => {
                for (i, step) in [2i32, 1, 0, 1].iter().enumerate() {
                    emit(
                        &mut score,
                        &fit(
                            &figuration_variation(subject, seed).transpose(*step),
                            5,
                            bar(i as i64),
                        ),
                        bar(i as i64),
                        Melody,
                        5,
                        intensity,
                        Normal,
                    );
                }
                let counter = contrasting_transform(subject, pivot, seed);
                for (i, step) in [0i32, 0, -1, -1].iter().enumerate() {
                    emit(
                        &mut score,
                        &fit(&counter.transpose(*step), 4, bar(i as i64)),
                        bar(i as i64),
                        CounterMelody,
                        4,
                        intensity,
                        Normal,
                    );
                }
            }
            3 => {
                // Over the altered ground, the melody looks BACKWARD —
                // retrograde, inward.
                let looking_back = subject.retrograde();
                for (i, step) in [0i32, -1, 0, -1].iter().enumerate() {
                    emit(
                        &mut score,
                        &fit(&looking_back.transpose(*step), 5, bar(i as i64)),
                        bar(i as i64),
                        Melody,
                        5,
                        intensity,
                        Normal,
                    );
                }
            }
            4 => {
                // Restoration = recognition = climax: octave-up
                // figuration blazing over the returned foundation.
                for (i, step) in [7i32, 8, 9, 8].iter().enumerate() {
                    emit(
                        &mut score,
                        &fit(
                            &figuration_variation(subject, seed).transpose(*step),
                            5,
                            bar(i as i64),
                        ),
                        bar(i as i64),
                        Melody,
                        5,
                        intensity,
                        if i == 0 { Climax } else { Normal },
                    );
                }
                let counter = figuration_variation(&subject.invert(pivot), seed);
                for (i, step) in [2i32, 2, 1, 1].iter().enumerate() {
                    emit(
                        &mut score,
                        &fit(&counter.transpose(*step), 4, bar(i as i64)),
                        bar(i as i64),
                        CounterMelody,
                        4,
                        intensity,
                        Normal,
                    );
                }
            }
            5 => {
                // Over the fragmented ground: sparse echoes — the head of
                // the subject, once per alternate bar.
                let head = crate::fugue::head_fragment(subject, Duration::new(meter as i64, 2));
                for (i, step) in [(0usize, 0i32), (2, -1)] {
                    emit(
                        &mut score,
                        &fit(&head.transpose(step), 5, bar(i as i64)),
                        bar(i as i64),
                        Melody,
                        5,
                        intensity,
                        Normal,
                    );
                }
            }
            _ => {
                // Complete: the subject once more, then a plain 2̂-1̂
                // cadence in the final bar, landing with the ground's
                // bent tonic tail.
                for (i, step) in [0i32, 1, 0].iter().enumerate() {
                    emit(
                        &mut score,
                        &fit(&subject.transpose(*step), 5, bar(i as i64)),
                        bar(i as i64),
                        Melody,
                        5,
                        intensity,
                        Normal,
                    );
                }
                let m = meter as i64;
                hold(
                    &mut score,
                    2,
                    bar(3),
                    Duration::new(m - 1, 1),
                    Melody,
                    5,
                    intensity,
                    Normal,
                );
                hold(
                    &mut score,
                    1,
                    bar(3) + Duration::new(m - 1, 1),
                    Duration::new(1, 1),
                    Melody,
                    5,
                    intensity + 0.05, // finally complete
                    Cadential,
                );
            }
        }
    }

    score
}

// ═══════════════════════════════════════════════════════════════════════
// Ground-worthiness: which subjects deserve to become grounds
// ═══════════════════════════════════════════════════════════════════════

/// The judgment scores for a candidate ground subject. The listening
/// review that demanded this (after seed 8's forgettable ground): "ground
/// suitability is a distinct compositional property — a good melodic hook
/// is not automatically a good ostinato," and separately: separate
/// FITNESS from CHARACTER — "a perfectly behaved ground that nobody
/// remembers isn't doing much work."
///
/// One character score, four fitness scores, all in [0, 1]:
/// - `distinctiveness` (character): if you heard only the ground, would
///   you recognize it? Duration contrast + a contour signature (the same
///   identity predicates the hook machinery enforces).
/// - `harmonic_affordance`: how GENEROUS the ground is to write over —
///   measured with the species fitter itself: fit a probe line over the
///   candidate's ground and count how few notes needed bending. "A ground
///   that constantly forces repairs isn't impossible to write over. It
///   just isn't generous."
/// - `rhythmic_durability`: does the rhythm survive being heard seven
///   times? Monotone rhythms erode; fussy ones blur.
/// - `fragmentation_survival`: does the one-tone-per-bar skeleton retain
///   the line's contour range (can cycle 6 still evoke it)?
/// - `transformation_legibility`: is the memory alteration (deepest tone
///   lifted) unambiguous — a unique lowest point to remember wrongly?
///
/// `composite` is the equal-weight mean — DELIBERATELY naive v0. The
/// Studio logs all five into keeper entries, so the ♥ data can eventually
/// learn which weighting corresponds to the listener's ear instead of
/// this guess. Judgment is advisory, not authoritative.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct GroundWorthiness {
    pub distinctiveness: f32,
    pub harmonic_affordance: f32,
    pub rhythmic_durability: f32,
    pub fragmentation_survival: f32,
    pub transformation_legibility: f32,
    pub composite: f32,
}

/// Score one candidate subject for ground duty. Deterministic, purely
/// symbolic — every score is computable before a single note is rendered.
pub fn ground_worthiness(subject: &Motif, key: Key, meter: u8) -> GroundWorthiness {
    let scale = key.scale();
    let ground = ground_from(subject, meter);

    // ── Character: distinctiveness ───────────────────────────────────────
    let durs: Vec<f64> = subject
        .notes
        .iter()
        .filter(|n| n.degree.is_some())
        .map(|n| n.duration.beats())
        .collect();
    let dur_contrast = if durs.is_empty() {
        0.0
    } else {
        let (lo, hi) = durs
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), d| (l.min(*d), h.max(*d)));
        if hi >= 2.0 * lo { 0.5 } else { 0.15 }
    };
    let degs: Vec<i32> = subject.notes.iter().filter_map(|n| n.degree).collect();
    let has_motion = degs.windows(2).any(|w| w[1] != w[0]);
    let mut contour_signature = 0.0f32;
    for w in degs.windows(3) {
        let (a, b) = (w[1] - w[0], w[2] - w[1]);
        // Leap answered by recoil, or an insistence (repetition) — but
        // repetition is only a SIGNATURE alongside motion; a drone is all
        // repetition and no identity (the first drone test caught the
        // naive predicate scoring it 0.5).
        if (a.abs() >= 2 && a.signum() != b.signum() && b != 0)
            || ((a == 0 || b == 0) && has_motion)
        {
            contour_signature = 0.5;
            break;
        }
    }
    let distinctiveness = dur_contrast + contour_signature;

    // ── Fitness: harmonic affordance (the fitter as measuring device) ───
    let cf: Vec<CantusEvent> = {
        let mut t = 0.0;
        let mut out = Vec::new();
        for n in &ground.notes {
            if let Some(d) = n.degree {
                out.push(CantusEvent {
                    onset: t,
                    duration: n.duration.beats(),
                    pitch: scale.degree_pitch(d, 2),
                });
            }
            t += n.duration.beats();
        }
        out
    };
    let mut probed = 0u32;
    let mut bent = 0u32;
    for (bar_idx, step) in [0i32, 1, 2, 1].iter().enumerate() {
        let probe = subject.transpose(*step);
        let start = (meter as f64) * bar_idx as f64;
        let fitted = fit_against(&cf, &probe, scale, 5, start);
        for (a, b) in probe.notes.iter().zip(fitted.notes.iter()) {
            if a.degree.is_some() {
                probed += 1;
                if a.degree != b.degree {
                    bent += 1;
                }
            }
        }
    }
    let harmonic_affordance = if probed == 0 {
        0.0
    } else {
        1.0 - bent as f32 / probed as f32
    };

    // ── Fitness: rhythmic durability ─────────────────────────────────────
    let mut uniq: Vec<i64> = subject
        .notes
        .iter()
        .filter(|n| n.degree.is_some())
        .map(|n| (n.duration.beats() * 480.0).round() as i64)
        .collect();
    uniq.sort_unstable();
    uniq.dedup();
    let mut rhythmic_durability = match uniq.len() {
        0 => 0.0,
        1 => 0.4, // monotone: erodes under seven repetitions
        2 | 3 => 1.0,
        _ => 0.7, // fussy: blurs under repetition
    };
    if durs.iter().any(|d| *d < 0.5) {
        rhythmic_durability *= 0.8; // sub-eighth chatter in a FOUNDATION
    }

    // ── Fitness: fragmentation survival ──────────────────────────────────
    let frag = fragmented_ground(&ground, meter);
    let range = |m: &Motif| -> i32 {
        let ds: Vec<i32> = m.notes.iter().filter_map(|n| n.degree).collect();
        match (ds.iter().min(), ds.iter().max()) {
            (Some(lo), Some(hi)) => hi - lo,
            _ => 0,
        }
    };
    let (gr, fr) = (range(&ground), range(&frag));
    let fragmentation_survival = if gr == 0 {
        0.2 // a flat drone: fragmentation "survives" but nothing identifies it
    } else {
        (fr as f32 / gr as f32).clamp(0.0, 1.0)
    };

    // ── Fitness: transformation legibility ───────────────────────────────
    let ground_degs: Vec<i32> = ground.notes.iter().filter_map(|n| n.degree).collect();
    let transformation_legibility = match ground_degs.iter().min() {
        None => 0.0,
        Some(min) => {
            if ground_degs.iter().filter(|d| *d == min).count() == 1 {
                1.0 // a unique deepest point: the altered memory is unambiguous
            } else {
                0.4 // tied minima: which note "was remembered wrong"?
            }
        }
    };

    let composite = (distinctiveness
        + harmonic_affordance
        + rhythmic_durability
        + fragmentation_survival
        + transformation_legibility)
        / 5.0;
    GroundWorthiness {
        distinctiveness,
        harmonic_affordance,
        rhythmic_durability,
        fragmentation_survival,
        transformation_legibility,
        composite,
    }
}

/// Audition `n` candidate subjects (deterministically derived from `base`
/// by re-rolling the bank pick) and grant the ground to the most worthy.
/// `derive` maps an audition index to a candidate subject — the composer
/// passes its own bank+hook pipeline so candidates are real subjects, not
/// mutations. Returns the winner and its scores. Selection is judgment,
/// not generation: "the ability to reject ideas is as important as the
/// ability to produce them."
///
/// `seed` breaks NEAR-ties only: `ground_worthiness` scoring is itself
/// seed-independent, so across many different global seeds the audition
/// kept crowning the same one or two structurally-strongest bank entries —
/// the census (`c11cfa43b7`) found this collapsed Passacaglia to ~0.0
/// median within-style nearest-neighbor distance, the real bottleneck
/// (selection averaging away randomness), not an undersized subject bank.
/// A small deterministic jitter (bounded to 0.08 — measured against real
/// bank candidates: comparably-worthy pairs differ by ~0.06 composite,
/// while a real subject beats a drone by ~0.19, over a 2x margin) lets the
/// seed decide among candidates that are genuinely comparably worthy,
/// without ever letting a clearly worse candidate win.
pub fn audition_ground(
    key: Key,
    meter: u8,
    n: usize,
    seed: u64,
    derive: impl Fn(usize) -> Motif,
) -> (Motif, GroundWorthiness) {
    let mut best: Option<(Motif, GroundWorthiness, f32)> = None;
    for k in 0..n.max(1) {
        let cand = derive(k);
        let w = ground_worthiness(&cand, key, meter);
        let mut state = seed ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let jitter = ((state >> 40) as f32 / (1u64 << 24) as f32) * 0.08;
        let scored = w.composite + jitter;
        if best.as_ref().map(|(_, _, bs)| scored > *bs).unwrap_or(true) {
            best = Some((cand, w, scored));
        }
    }
    let (m, w, _) = best.expect("n >= 1");
    (m, w)
}

// ═══════════════════════════════════════════════════════════════════════
// Erosion: persistence failing
// ═══════════════════════════════════════════════════════════════════════

/// How an erosion piece is allowed to end. Memory asks "what returns?";
/// persistence asks "what refuses to disappear?"; erosion asks what it
/// COSTS — and not every piece should earn reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErosionEnding {
    /// Everything returns: the final cycle restores the full ground.
    Recovery,
    /// Some things return, others don't: a partial restoration.
    Acceptance,
    /// The final return tries, but cannot remember itself completely —
    /// still eroded, and it does NOT resolve to the tonic.
    Elegy,
}

impl ErosionEnding {
    /// Stable lowercase name for telemetry/taste logs.
    pub fn name(self) -> &'static str {
        match self {
            ErosionEnding::Recovery => "recovery",
            ErosionEnding::Acceptance => "acceptance",
            ErosionEnding::Elegy => "elegy",
        }
    }
}

/// Erode `level` non-skeletal tones out of the ground (deterministically,
/// by seeded order). Detail dies first: bar-first tones (the skeleton the
/// fragmented cycle proved identifiable) are only consumed after every
/// interior tone is gone. Level 0 is the ground itself.
pub(crate) fn eroded_ground(ground: &Motif, meter: u8, level: usize, seed: u64) -> Motif {
    if level == 0 {
        return ground.clone();
    }
    let bar_beats = meter as f64;
    // Partition pitched indices into interior (erode first) and skeleton.
    let mut interior = Vec::new();
    let mut skeleton = Vec::new();
    let mut t = 0.0;
    let mut bar_of_last_first = usize::MAX;
    for (i, n) in ground.notes.iter().enumerate() {
        let bar = (t / bar_beats + 1e-9) as usize;
        if n.degree.is_some() {
            if bar != bar_of_last_first {
                skeleton.push(i);
                bar_of_last_first = bar;
            } else {
                interior.push(i);
            }
        }
        t += n.duration.beats();
    }
    // Seeded deterministic order within each class.
    let shuffle = |v: &mut Vec<usize>, salt: u64| {
        let mut state = seed ^ salt;
        v.sort_by_key(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state.wrapping_add(*i as u64 * 0x9E37_79B9)
        });
    };
    shuffle(&mut interior, 0x0E0D_E001);
    shuffle(&mut skeleton, 0x0E0D_E002);
    let mut g = ground.clone();
    for idx in interior.into_iter().chain(skeleton).take(level) {
        g.notes[idx] = MotifNote::rest(g.notes[idx].duration);
    }
    g
}

/// Compose the erosion piece: seven cycles in which the GROUND loses
/// confidence — one more tone gone each cycle — while the melody above
/// keeps stating the subject steadily. The figure/ground reversal is the
/// point: the world continues while the foundation fades. The ending
/// decides what persistence cost (see [`ErosionEnding`]).
pub(crate) fn realize_erosion(
    key: Key,
    tempo_bpm: f32,
    meter: u8,
    subject: &Motif,
    seed: u64,
    ending: ErosionEnding,
) -> Score {
    let mut score = Score::new(key, tempo_bpm, meter);
    let scale = key.scale();
    let ground = ground_from(subject, meter);
    let ground_beats = ground.total_duration();

    use Emphasis::{Cadential, Normal, PhraseStart};
    use VoiceRole::{Bass, Melody};
    // Confidence drains cycle by cycle; the ending sets the last word.
    const FADE: [f32; 6] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75];

    for cycle in 0..CYCLES {
        let start = Duration::new(ground_beats.num() * cycle as i64, ground_beats.den());
        let last = cycle == CYCLES - 1;
        // The FALSE RECOVERY (elegy only), from the listening review that
        // judged the first elegy "convincing" and prescribed how to make
        // it devastating: "let the listener believe recovery is still
        // possible. Then deny it." The penultimate cycle nearly restores
        // the ground and the confidence lifts — hope — so the final
        // collapse is a denial of something the ear was just promised.
        let false_recovery = ending == ErosionEnding::Elegy && cycle == CYCLES - 2;
        let (mut g, intensity, bass_emphasis) = if false_recovery {
            (eroded_ground(&ground, meter, 1, seed), 0.95, PhraseStart)
        } else if last {
            match ending {
                ErosionEnding::Recovery => (ground.clone(), 1.05, PhraseStart),
                ErosionEnding::Acceptance => (eroded_ground(&ground, meter, 2, seed), 0.9, Normal),
                ErosionEnding::Elegy => (eroded_ground(&ground, meter, 3, seed), 0.6, Normal),
            }
        } else {
            (
                eroded_ground(&ground, meter, cycle, seed),
                FADE[cycle],
                if cycle == 0 { PhraseStart } else { Normal },
            )
        };
        if last && ending != ErosionEnding::Elegy {
            // Recovery and acceptance still cadence; the elegy is DENIED
            // its tonic.
            if let Some(tail) = g.notes.iter_mut().rev().find(|n| n.degree.is_some()) {
                tail.degree = Some(1);
            }
        }
        if last && ending == ErosionEnding::Elegy {
            // "...or the ground almost completes and then simply lacks one
            // interval": the leading-tone pickup — the ONE tone that would
            // carry the line home — is removed. The ground stops on the
            // dominant, mid-sentence.
            if let Some(tail) = g.notes.iter_mut().rev().find(|n| n.degree.is_some()) {
                *tail = MotifNote::rest(tail.duration);
            }
        }
        emit(
            &mut score,
            &g,
            start,
            Bass,
            2,
            intensity,
            if last { Cadential } else { bass_emphasis },
        );

        // The melody persists, steady, fitted against whatever ground
        // remains (where the ground is silent, the line stands alone).
        let cf: Vec<CantusEvent> = {
            let mut t = start.beats();
            let mut out = Vec::new();
            for n in &g.notes {
                if let Some(d) = n.degree {
                    out.push(CantusEvent {
                        onset: t,
                        duration: n.duration.beats(),
                        pitch: scale.degree_pitch(d, 2),
                    });
                }
                t += n.duration.beats();
            }
            out
        };
        for (i, step) in [0i32, 1, 0, -1].iter().enumerate() {
            let at = start + Duration::new(meter as i64 * i as i64, 1);
            let mut line = fit_against(&cf, &subject.transpose(*step), scale, 5, at.beats());
            if last && ending == ErosionEnding::Elegy && i == 3 {
                // "The melody reaches for the remembered note and falls
                // short": its very last tone lands on degree 2 — one step
                // from home — and stays there.
                if let Some(tail) = line.notes.iter_mut().rev().find(|n| n.degree.is_some()) {
                    tail.degree = Some(2);
                }
            }
            emit(
                &mut score,
                &line,
                at,
                Melody,
                5,
                intensity,
                if last && ending == ErosionEnding::Elegy && i == 3 {
                    Cadential
                } else if cycle == 0 && i == 0 {
                    PhraseStart
                } else {
                    Normal
                },
            );
        }
    }
    score
}

// ═══════════════════════════════════════════════════════════════════════
// Lineage: identity evolving through kinship
// ═══════════════════════════════════════════════════════════════════════

/// The lineage chain: each cycle's ground is ONE legible transformation
/// step away from its parent, chosen deterministically per generation.
/// Formalizes the contaminated-control finding as a first-class mode:
/// persistence says "I remain"; lineage says "I become my descendants."
///
/// THE EXPERIMENT'S FULL ARC (three falsified designs, then a confound
/// isolation that CONFIRMED the hypothesis with an ordering):
///
/// 1. invert/retrograde chain lost to anonymity — trigrams are
///    transform-blind to contour flips, and musically a bass that flips
///    contour every four bars reads as different lines, not family.
/// 2. Interval-preserving DRIFT chain lost — cumulative transposition is
///    slow anonymity; the control's tonic-restarting walks were
///    accidentally more stationary.
/// 3. ROOTED mixed chain (every generation descends from the ancestor)
///    was seed-dependent: won when the seed drew figurations, lost when
///    it drew the retrograde mutation.
/// 4. Fixed-mix chains isolated the variable and CONFIRMED kinship, with
///    an ordering: figuration-kinship Φ=0.150 — THE MOST INTEGRATED FORM
///    MEASURED IN THE PROJECT, 3× strict invariance (0.048) — then
///    transposition-kinship 0.093, then anonymity 0.044-0.083.
///    "I become my ornamented descendants" out-integrates "I remain."
///
/// The lesson, now load-bearing in this design: kinship integrates in
/// proportion to how much INTERVAL IDENTITY the transformation preserves
/// — figuration (skeleton-exact) > neighbor transposition (contour-exact) >
/// contour flips (invisible, and inaudible AS KINSHIP). So the
/// production chain is figuration-dominant with neighbor-transposition
/// excursions and NO contour flips: every generation a recognizable
/// descendant of the ancestor.
pub(crate) fn lineage_chain(ground: &Motif, generations: usize, seed: u64) -> Vec<Motif> {
    let mut chain = vec![ground.clone()];
    let mut state = seed;
    for _ in 1..generations {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let child = match (state >> 33) % 4 {
            0 => ground.transpose(1),
            1 => ground.transpose(-1),
            _ => figuration_variation(ground, state),
        };
        chain.push(child);
    }
    chain
}

/// Compose the lineage piece: seven generations of the ground, each one
/// transformation step from its parent, the subject stated above each and
/// fitted to it. The final generation's tail is bent to the tonic so the
/// evolution can cadence — identity evolves away and still comes home.
pub(crate) fn realize_lineage(
    key: Key,
    tempo_bpm: f32,
    meter: u8,
    subject: &Motif,
    seed: u64,
) -> Score {
    realize_lineage_inner(key, tempo_bpm, meter, subject, seed, false)
}

/// Body with the experiment control switchable: `anonymous` replaces the
/// kinship chain with per-cycle anonymous walks (identical rhythm,
/// register, density — no kinship), holding the upper voices identical.
/// The pre-registered claim: kinship should out-integrate anonymity just
/// as strict invariance did.
fn realize_lineage_inner(
    key: Key,
    tempo_bpm: f32,
    meter: u8,
    subject: &Motif,
    seed: u64,
    anonymous: bool,
) -> Score {
    let ground = ground_from(subject, meter);
    let chain: Vec<Motif> = if anonymous {
        (0..CYCLES)
            .map(|cycle| {
                let mut state = seed ^ (cycle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut deg = 1i32;
                let notes = ground
                    .notes
                    .iter()
                    .map(|n| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let step = ((state >> 33) % 5) as i32 - 2;
                        deg = (deg + step).clamp(-3, 8);
                        MotifNote {
                            degree: n.degree.map(|_| deg),
                            duration: n.duration,
                        }
                    })
                    .collect();
                Motif { notes }
            })
            .collect()
    } else {
        lineage_chain(&ground, CYCLES, seed)
    };
    realize_over_chain(key, tempo_bpm, meter, subject, &chain)
}

/// Realize seven ground statements from an explicit chain — the shared
/// engine behind lineage and its experiment arms (tests drive custom
/// chains through it to isolate which TRANSFORMATION MIXES integrate).
fn realize_over_chain(
    key: Key,
    tempo_bpm: f32,
    meter: u8,
    subject: &Motif,
    chain: &[Motif],
) -> Score {
    let mut score = Score::new(key, tempo_bpm, meter);
    let scale = key.scale();
    let ground_beats = chain[0].total_duration();

    use Emphasis::{Cadential, Climax, Normal, PhraseStart};
    use VoiceRole::{Bass, Melody};
    const INTENSITY: [f32; CYCLES] = [0.8, 0.85, 0.9, 0.95, 1.1, 0.95, 0.9];

    for (cycle, g0) in chain.iter().enumerate() {
        let start = Duration::new(ground_beats.num() * cycle as i64, ground_beats.den());
        let intensity = INTENSITY[cycle];
        let last = cycle == CYCLES - 1;
        let mut g = g0.clone();
        let tail_opt = if last {
            g.notes.iter_mut().rev().find(|n| n.degree.is_some())
        } else {
            None
        };
        if let Some(tail) = tail_opt {
            tail.degree = Some(1);
        }
        let emphasis = match cycle {
            0 => PhraseStart,
            4 => Climax,
            c if c == CYCLES - 1 => Cadential,
            _ => Normal,
        };
        emit(&mut score, &g, start, Bass, 2, intensity, emphasis);

        let cf: Vec<CantusEvent> = {
            let mut t = start.beats();
            let mut out = Vec::new();
            for n in &g.notes {
                if let Some(d) = n.degree {
                    out.push(CantusEvent {
                        onset: t,
                        duration: n.duration.beats(),
                        pitch: scale.degree_pitch(d, 2),
                    });
                }
                t += n.duration.beats();
            }
            out
        };
        if cycle > 0 {
            for (i, step) in [0i32, 1, 2, 1].iter().enumerate() {
                let at = start + Duration::new(meter as i64 * i as i64, 1);
                let line = fit_against(&cf, &subject.transpose(*step), scale, 5, at.beats());
                emit(&mut score, &line, at, Melody, 5, intensity, Normal);
            }
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn subject() -> Motif {
        // A 3-beat (3/4) subject, as the Passacaglia preset's banks supply.
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (3, Duration::quarter()),
            (2, Duration::quarter()),
        ])
    }

    fn passacaglia(seed: u64, fresh: bool) -> Score {
        realize_passacaglia(Key::minor(PitchClass::A), 70.0, 3, &subject(), seed, fresh)
    }

    #[test]
    fn ground_is_four_bars_ending_on_the_leading_tone() {
        let g = ground_from(&subject(), 3);
        assert_eq!(g.total_duration(), Duration::new(12, 1));
        assert_eq!(
            g.notes.iter().rev().find_map(|n| n.degree),
            Some(7),
            "the pickup into every cycle restart is the leading tone"
        );
    }

    #[test]
    fn altered_ground_changes_exactly_one_note_upward() {
        let g = ground_from(&subject(), 3);
        let a = altered_ground(&g);
        let diffs: Vec<(Option<i32>, Option<i32>)> = g
            .notes
            .iter()
            .zip(a.notes.iter())
            .filter(|(x, y)| x.degree != y.degree)
            .map(|(x, y)| (x.degree, y.degree))
            .collect();
        assert_eq!(diffs.len(), 1, "exactly one remembered alteration");
        assert_eq!(diffs[0].1.unwrap(), diffs[0].0.unwrap() + 1);
        assert_eq!(a.total_duration(), g.total_duration());
    }

    #[test]
    fn fragmented_ground_keeps_one_tone_per_bar() {
        let g = ground_from(&subject(), 3);
        let f = fragmented_ground(&g, 3);
        assert_eq!(f.total_duration(), g.total_duration());
        let pitched = f.notes.iter().filter(|n| n.degree.is_some()).count();
        assert_eq!(pitched, 4, "one surviving tone per bar of the 4-bar ground");
        assert!(pitched < g.notes.iter().filter(|n| n.degree.is_some()).count());
    }

    #[test]
    fn seven_cycles_and_the_ground_memory_arc() {
        let s = passacaglia(1, false);
        assert_eq!(s.total_beats, Duration::new(7 * 12, 1));
        let bass = s.voice(VoiceRole::Bass);
        let cycle_degrees = |c: usize| -> Vec<u8> {
            let (lo, hi) = (c as f64 * 12.0, (c + 1) as f64 * 12.0);
            bass.iter()
                .filter(|n| n.onset.beats() >= lo && n.onset.beats() < hi)
                .map(|n| n.pitch.midi())
                .collect()
        };
        let c0 = cycle_degrees(0);
        assert_eq!(cycle_degrees(4), c0, "cycle 5 restores the ground verbatim");
        assert_ne!(cycle_degrees(3), c0, "cycle 4 is the altered memory");
        assert!(
            cycle_degrees(5).len() < c0.len(),
            "cycle 6 is the fragmented ground"
        );
        // The final statement ends on the tonic.
        assert_eq!(
            bass.last().unwrap().pitch.pitch_class(),
            PitchClass::A,
            "the completed ground closes on the tonic"
        );
    }

    #[test]
    fn restoration_is_the_peak() {
        let s = passacaglia(1, false);
        let max = s
            .notes
            .iter()
            .map(|n| n.section_intensity)
            .fold(f32::MIN, f32::max);
        assert_eq!(max, 1.15);
        // And every note at peak intensity lives in cycle 5's span.
        for n in s.notes.iter().filter(|n| n.section_intensity == 1.15) {
            assert!(
                (48.0..60.0).contains(&n.onset.beats()),
                "peak material outside the restoration cycle at beat {}",
                n.onset.beats()
            );
        }
    }

    #[test]
    fn upper_lines_land_on_beat_consonant_with_the_ground() {
        // The fitter contract over the whole piece: every upper note that
        // LANDS on an integer beat while the ground sounds must be
        // consonant with it (the ground is a literal cantus firmus).
        let s = passacaglia(1, false);
        let bass = s.voice(VoiceRole::Bass);
        for role in [VoiceRole::Melody, VoiceRole::CounterMelody] {
            for n in s.voice(role) {
                let t = n.onset.beats();
                if (t - t.round()).abs() > 1e-6 {
                    continue;
                }
                let Some(g) = bass
                    .iter()
                    .find(|b| b.onset.beats() - 1e-9 <= t && t < (b.onset + b.duration).beats())
                else {
                    continue;
                };
                assert!(
                    crate::counterpoint::is_consonant(g.pitch, n.pitch),
                    "beat {t}: {role:?} lands {} against ground {}",
                    n.pitch.midi(),
                    g.pitch.midi()
                );
            }
        }
    }

    #[test]
    fn invariant_ground_out_integrates_fresh_grounds() {
        // THE PRE-REGISTERED EXPERIMENT (from the species-counterpoint
        // falsification): musical Φ's bottleneck is temporal — cross-
        // segment motif continuity. An invariant ground threads every
        // segment with the same line; the control keeps the ground's
        // rhythm, register, and role but changes its LINE every cycle.
        // If invariance does not out-integrate the control, the
        // hypothesis dies here, in CI, with diagnostics — exactly as the
        // previous hypothesis did.
        //
        // RESULT (first honest run, 2026-07-11): CONFIRMED on all seeds —
        //   seed 1: invariant Φ=0.0481 vs control 0.0401
        //   seed 5: invariant Φ=0.0378 vs control 0.0288
        //   seed 9: invariant Φ=0.0394 vs control 0.0193 (2×)
        // Seed 1 is the instructive one: the control had HIGHER mean
        // edges on BOTH channels (cons 0.802 vs 0.710, motif 0.108 vs
        // 0.105) yet LOWER Φ — mean connectivity is not what λ₂
        // measures. The invariant ground raises the piece's WEAKEST
        // cross-segment coupling, which is exactly the temporal
        // bottleneck the species-counterpoint falsification identified.
        // These passacaglia Φ values (0.038-0.048) are also the highest
        // of any contrapuntal form measured so far (fugue: 0.006-0.02).
        for seed in [1u64, 5, 9] {
            let invariant = passacaglia(seed, false);
            let control = passacaglia(seed, true);
            let pi = crate::integration::musical_phi(&invariant);
            let pc = crate::integration::musical_phi(&control);
            assert!(
                pi.phi > pc.phi,
                "seed {seed}: invariant ground Φ={:.4} (cons {:.3}, motif {:.3}) must beat \
                 fresh-grounds control Φ={:.4} (cons {:.3}, motif {:.3})",
                pi.phi,
                pi.mean_consonance_edge,
                pi.mean_trigram_edge,
                pc.phi,
                pc.mean_consonance_edge,
                pc.mean_trigram_edge
            );
        }
    }

    #[test]
    fn passacaglia_is_deterministic() {
        let a = passacaglia(7, false);
        let b = passacaglia(7, false);
        assert_eq!(a, b);
    }

    // ── ground-worthiness ────────────────────────────────────────────────

    fn drone() -> Motif {
        // Monotone, flat: behaves perfectly, says nothing.
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (1, Duration::quarter()),
            (1, Duration::quarter()),
        ])
    }

    #[test]
    fn worthiness_prefers_a_real_subject_over_a_drone() {
        let key = Key::minor(PitchClass::A);
        let good = ground_worthiness(&subject(), key, 3);
        let flat = ground_worthiness(&drone(), key, 3);
        assert!(
            good.composite > flat.composite,
            "subject {:?} vs drone {:?}",
            good,
            flat
        );
        // The drone specifically fails CHARACTER. (It does NOT fail
        // fragmentation: ground_from appends the lamento close, which
        // gives even a drone-subject's ground a skeleton range — the
        // close is part of the ground identity by design.)
        assert!(flat.distinctiveness < 0.5);
    }

    #[test]
    fn worthiness_scores_are_bounded_and_deterministic() {
        let key = Key::minor(PitchClass::A);
        let a = ground_worthiness(&subject(), key, 3);
        let b = ground_worthiness(&subject(), key, 3);
        for v in [
            a.distinctiveness,
            a.harmonic_affordance,
            a.rhythmic_durability,
            a.fragmentation_survival,
            a.transformation_legibility,
            a.composite,
        ] {
            assert!((0.0..=1.0).contains(&v), "{v} out of range");
        }
        assert_eq!(a.composite, b.composite);
    }

    #[test]
    fn audition_grants_the_ground_to_the_most_worthy() {
        let key = Key::minor(PitchClass::A);
        let (winner, w) =
            audition_ground(key, 3, 2, 1, |k| if k == 0 { drone() } else { subject() });
        assert_eq!(winner, subject(), "the drone must lose the audition");
        assert!(w.composite > 0.5);
    }

    #[test]
    fn seed_breaks_ties_among_comparably_worthy_candidates_without_ever_picking_the_drone() {
        // The bug this fixes: ground_worthiness scoring is itself seed-
        // independent, so the OLD audition (strict `>`, first-seen wins
        // ties) crowned the same structurally-strongest candidate across
        // almost every global seed — the census found this collapsed
        // Passacaglia to ~0.0 median within-style nearest-neighbor
        // distance. Two real subjects of comparable worth (both beat a
        // drone) must be winnable by EITHER index depending on seed...
        let subject_b = Motif::from_degrees(&[
            (2, Duration::quarter()),
            (5, Duration::quarter()),
            (3, Duration::quarter()),
        ]);
        // These two candidates' real composite scores differ by only
        // ~0.023 (measured directly, not assumed) — comfortably inside
        // the 0.08 jitter ceiling, so flips should be common across many
        // seeds; 300 seeds is a generous margin, not a coin-flip pass.
        let key = Key::minor(PitchClass::A);
        let mut winners = std::collections::HashSet::new();
        for seed in 0u64..300 {
            let (winner, _) = audition_ground(key, 3, 2, seed, |k| {
                if k == 0 { subject() } else { subject_b.clone() }
            });
            winners.insert(format!("{winner:?}"));
        }
        assert!(
            winners.len() > 1,
            "seed must be able to break a near-tie between two comparably worthy \
             candidates, got only {} distinct winner(s)",
            winners.len()
        );
        // ...but a clearly worse candidate (the drone) must NEVER win,
        // regardless of seed — the jitter is bounded well below the real
        // composite gap this relies on.
        for seed in 0u64..300 {
            let (winner, _) =
                audition_ground(
                    key,
                    3,
                    2,
                    seed,
                    |k| if k == 0 { drone() } else { subject() },
                );
            assert_eq!(winner, subject(), "seed {seed}: the drone must never win");
        }
    }

    // ── erosion ──────────────────────────────────────────────────────────

    #[test]
    fn erosion_eats_interior_tones_before_the_skeleton() {
        let g = ground_from(&subject(), 3);
        let pitched = |m: &Motif| m.notes.iter().filter(|n| n.degree.is_some()).count();
        let full = pitched(&g);
        for level in 0..full {
            assert_eq!(
                pitched(&eroded_ground(&g, 3, level, 9)),
                full - level,
                "level {level} must remove exactly {level} tones"
            );
        }
        // With every interior tone gone, the bar-first skeleton survives:
        // erode exactly the interior count and compare to the fragmented
        // ground's skeleton size.
        let skeleton = pitched(&fragmented_ground(&g, 3));
        let interior = full - skeleton;
        assert_eq!(pitched(&eroded_ground(&g, 3, interior, 9)), skeleton);
    }

    #[test]
    fn erosion_endings_are_genuinely_different() {
        let key = Key::minor(PitchClass::A);
        let last_cycle_bass = |ending: ErosionEnding| -> Vec<crate::score::ScoreNote> {
            let s = realize_erosion(key, 70.0, 3, &subject(), 4, ending);
            s.voice(VoiceRole::Bass)
                .into_iter()
                .filter(|n| n.onset.beats() >= 6.0 * 12.0)
                .collect()
        };
        let recovery = last_cycle_bass(ErosionEnding::Recovery);
        let acceptance = last_cycle_bass(ErosionEnding::Acceptance);
        let elegy = last_cycle_bass(ErosionEnding::Elegy);
        // Recovery restores everything; acceptance keeps some losses;
        // the elegy keeps more losses still.
        assert!(recovery.len() > acceptance.len());
        assert!(acceptance.len() > elegy.len());
        // Recovery and acceptance cadence on the tonic; the elegy is
        // DENIED the tonic — it cannot remember itself completely.
        assert_eq!(recovery.last().unwrap().pitch.pitch_class(), PitchClass::A);
        assert_eq!(
            acceptance.last().unwrap().pitch.pitch_class(),
            PitchClass::A
        );
        assert_ne!(elegy.last().unwrap().pitch.pitch_class(), PitchClass::A);
    }

    #[test]
    fn elegy_offers_false_recovery_then_denies_it() {
        // The review's prescription, pinned: "let the listener believe
        // recovery is still possible. Then deny it." The penultimate
        // cycle must SURGE (nearly whole ground, lifted confidence); the
        // final cycle must collapse, lack the one tone that would carry
        // it home, and leave the melody one step short of the tonic.
        let key = Key::minor(PitchClass::A);
        let s = realize_erosion(key, 70.0, 3, &subject(), 4, ErosionEnding::Elegy);
        let bass = s.voice(VoiceRole::Bass);
        let cycle_notes = |c: usize| -> Vec<crate::score::ScoreNote> {
            bass.iter()
                .copied()
                .filter(|n| {
                    n.onset.beats() >= c as f64 * 12.0 && n.onset.beats() < (c + 1) as f64 * 12.0
                })
                .collect()
        };
        let (c4, c5, c6) = (cycle_notes(4), cycle_notes(5), cycle_notes(6));
        // Hope: the false recovery is fuller than what preceded it...
        assert!(
            c5.len() > c4.len(),
            "false recovery must nearly restore the ground"
        );
        // ...and louder (confidence lifts).
        assert!(c5[0].section_intensity > c4[0].section_intensity);
        // Denial: the final cycle collapses again, quieter than the hope.
        assert!(
            c6.len() < c5.len(),
            "the denial must collapse after the hope"
        );
        assert!(c6[0].section_intensity < c5[0].section_intensity);
        // The ground lacks the one tone that would carry it home: it does
        // not end on the tonic OR the leading tone (the pickup is gone).
        let last_bass = c6.last().unwrap().pitch.pitch_class();
        assert_ne!(last_bass, PitchClass::A, "denied the tonic");
        assert_ne!(
            last_bass,
            PitchClass::GSHARP,
            "the pickup itself is removed"
        );
        // The melody reaches for the remembered note and falls short:
        // its final tone is degree 2 (B in A minor), one step from home.
        assert_eq!(
            s.voice(VoiceRole::Melody)
                .last()
                .unwrap()
                .pitch
                .pitch_class(),
            PitchClass::B,
            "the melody's last tone must land one step short of the tonic"
        );
    }

    #[test]
    fn erosion_confidence_drains_until_the_ending_speaks() {
        let key = Key::minor(PitchClass::A);
        let s = realize_erosion(key, 70.0, 3, &subject(), 4, ErosionEnding::Recovery);
        // Intensity is non-increasing through the six eroding cycles...
        let cycle_intensity = |c: usize| -> f32 {
            s.notes
                .iter()
                .find(|n| n.onset.beats() >= c as f64 * 12.0)
                .unwrap()
                .section_intensity
        };
        for c in 0..5 {
            assert!(cycle_intensity(c) >= cycle_intensity(c + 1));
        }
        // ...and recovery's final cycle rises above the fade.
        assert!(cycle_intensity(6) > cycle_intensity(5));
    }

    // ── lineage ──────────────────────────────────────────────────────────

    #[test]
    fn lineage_chain_is_kinship_all_the_way_down() {
        let g = ground_from(&subject(), 3);
        let chain = lineage_chain(&g, CYCLES, 7);
        assert_eq!(chain.len(), CYCLES);
        assert_eq!(chain[0], g);
        let degs = |m: &Motif| -> Vec<i32> { m.notes.iter().filter_map(|n| n.degree).collect() };
        let is_subsequence = |needle: &[i32], hay: &[i32]| -> bool {
            let mut it = hay.iter();
            needle.iter().all(|d| it.any(|h| h == d))
        };
        // Every generation must be a legible descendant of the ANCESTOR
        // (the rooted design — see lineage_chain's two design lessons).
        for child in &chain[1..] {
            let exact = [g.transpose(1), g.transpose(-1)];
            // Figuration inserts connecting tones but keeps the root's
            // degree sequence as an in-order subsequence, duration-exact.
            let figured_kin = child.total_duration() == g.total_duration()
                && child.notes.len() >= g.notes.len()
                && is_subsequence(&degs(&g), &degs(child));
            assert!(
                exact.contains(child) || figured_kin,
                "a generation is not a legible descendant of the ancestor"
            );
        }
    }

    #[test]
    fn kinship_out_integrates_anonymity_in_proportion_to_interval_identity() {
        // THE SECOND PRE-REGISTERED EXPERIMENT — falsified three times,
        // then CONFIRMED once the confound was isolated. History:
        // contour-flip chains and drift chains lost to anonymity; the
        // rooted mixed chain was seed-dependent (won on figuration draws,
        // lost on retrograde draws). Fixed-mix chains settled it with an
        // ORDERING pinned below: kinship integrates in proportion to how
        // much interval identity the transformation preserves.
        //
        //   figuration-kinship  Φ≈0.150  (highest form measured; 3× the
        //                                 invariant passacaglia's 0.048)
        //   transposition-kin   Φ≈0.093
        //   anonymous walks     Φ≈0.044-0.083 (seed-dependent)
        //
        // "I become my ornamented descendants" out-integrates "I remain"
        // — lineage earned its place as a primitive, with a condition:
        // the descent must be interval-legible.
        let key = Key::minor(PitchClass::A);
        let g = ground_from(&subject(), 3);
        let fig_chain: Vec<Motif> = std::iter::once(g.clone())
            .chain((1..CYCLES).map(|k| figuration_variation(&g, k as u64 * 77)))
            .collect();
        let trans_chain: Vec<Motif> = (0..CYCLES)
            .map(|k| match k % 3 {
                0 => g.clone(),
                1 => g.transpose(1),
                _ => g.transpose(-1),
            })
            .collect();
        let phi_of = |chain: &[Motif]| {
            crate::integration::musical_phi(&realize_over_chain(key, 70.0, 3, &subject(), chain))
                .phi
        };
        let (fig, trans) = (phi_of(&fig_chain), phi_of(&trans_chain));
        assert!(
            fig > trans,
            "figuration (skeleton-exact) must out-integrate transposition: {fig:.4} vs {trans:.4}"
        );
        for seed in [1u64, 5, 9] {
            let anon = crate::integration::musical_phi(&realize_lineage_inner(
                key,
                70.0,
                3,
                &subject(),
                seed,
                true,
            ))
            .phi;
            assert!(
                fig > anon,
                "seed {seed}: figuration-kin {fig:.4} vs anon {anon:.4}"
            );
            assert!(
                trans > anon,
                "seed {seed}: transposition-kin {trans:.4} vs anon {anon:.4}"
            );
        }
        // The headline: legible lineage out-integrates strict persistence.
        let invariant = crate::integration::musical_phi(&realize_passacaglia(
            key,
            70.0,
            3,
            &subject(),
            1,
            false,
        ))
        .phi;
        assert!(
            fig > invariant,
            "figuration-kinship {fig:.4} must beat the invariant ground {invariant:.4}"
        );
    }

    #[test]
    fn erosion_and_lineage_are_deterministic() {
        let key = Key::minor(PitchClass::A);
        assert_eq!(
            realize_erosion(key, 70.0, 3, &subject(), 3, ErosionEnding::Elegy),
            realize_erosion(key, 70.0, 3, &subject(), 3, ErosionEnding::Elegy)
        );
        assert_eq!(
            realize_lineage(key, 70.0, 3, &subject(), 3),
            realize_lineage(key, 70.0, 3, &subject(), 3)
        );
    }
}
