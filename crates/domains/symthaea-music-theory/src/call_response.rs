// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real call-and-response phrase grammar for `GrammarFamily::
//! BluesCallResponse` (2026-07-24, closing a gap Blues's own doc comment
//! previously admitted: "Call-and-response isn't new machinery here — it
//! falls out of the engine's existing antecedent/consequent period
//! grammar." An audit of every dedicated phrase engine in this crate
//! confirmed that claim doesn't hold up: `Period::parallel_in` develops
//! the SAME motif in both halves (differing only in cadence), which is
//! not a genuine call-vs-response distinction — the only style in this
//! crate literally named after call-and-response had zero real mechanism
//! for it.
//!
//! Adapted from [`crate::opera`]'s Dialogue mechanism (two independent,
//! genuinely different melodic ideas in structured alternation) rather
//! than invented from scratch — but Blues has no `CounterMelody` voice
//! (`counter_instrument: None`), so this stays single-voice: `call` is
//! the style's own established motif ([`crate::spec::CompositionSpec::
//! motif`], hook-grafted exactly like every other production path);
//! `response` is that SAME motif's INVERSION ([`crate::motif::Motif::
//! invert`], an existing, well-defined, already-used-everywhere-else
//! transformation) — mechanically "the response answers the call by
//! turning it upside down," not a guessed-at new melodic idea. This
//! keeps the whole mechanism free of invented taste: every parameter
//! traces to something the crate already does elsewhere.
//!
//! Alternates call (bars 0-1) / response (bars 2-3) across the style's
//! own declared 12-bar chorus progression (3 four-bar lines), repeated
//! for [`chorus_count`] choruses — the traditional blues call/response
//! pattern, 3 times per chorus. Harmony and bass are realized normally
//! over a real multi-section [`Form`] exactly as [`crate::opera`] does;
//! the novelty is entirely in how the melody is placed.
//!
//! **Duration contract** (fixed 2026-07-26, found via a real listening
//! session: 3 different seeds at `bars: 4` all realized as identical
//! 36-bar/97.5s pieces, because chorus count used to be a hardcoded `3`
//! that never read `intent.bars` at all — the only one of this crate's 4
//! dedicated grammar engines with that gap; `process_grammar.rs` and
//! `modal_arc.rs` both already scale from `intent.bars`). Blues's 12-bar
//! chorus is a structurally meaningful unit -- a request for `bars: 4`
//! can't literally mean "realize 4 bars" without cutting a chorus mid-
//! progression, so [`chorus_count`] rounds UP to the nearest whole
//! chorus (`div_ceil`) rather than either truncating a chorus or
//! silently ignoring the request. The realized length is reported
//! honestly on [`CallResponsePlan`] (`requested_bars` vs `realized_bars`)
//! rather than left for a caller to infer.
//!
//! **Chorus development** (added 2026-07-26, the user's own follow-up
//! design once duration was fixed: "Once chorus count varies, each
//! chorus should have a different role... do not generate identical
//! chorus templates back-to-back"). Each chorus gets a [`ChorusRole`]
//! from a seed-chosen [`trajectory_for`] the piece's chorus count, and
//! each role maps the RESPONSE to a different, already-existing
//! [`Motif`] transformation rather than always `invert` — so a 3-chorus
//! piece is a real developing conversation (e.g. Statement -> Breakdown
//! -> Intensification), not three copies of the same exchange. Every
//! role's transformation is one this crate already implements and uses
//! elsewhere (`invert`/`retrograde`/`diminish`/`fragment`, and
//! `invert`+`retrograde` composed is literally the 4th classical
//! transformation this crate's own `canonical_fingerprint` already
//! treats as a first-class transform) -- no invented melodic material,
//! same discipline as the original single-chorus mechanism.

use crate::cadence::Cadence;
use crate::composer::MusicalIntent;
use crate::form::{Form, Section, SectionRole};
use crate::harmony::Key;
use crate::motif::Motif;
use crate::phrase::{Period, Phrase};
use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};
use crate::spec::CompositionSpec;

const BARS_PER_CHORUS: usize = 12;

/// How many whole 12-bar choruses realize `requested_bars` -- rounds UP
/// (`div_ceil`) so a short request (e.g. the default `bars: 4`) still
/// gets one complete, harmonically coherent chorus rather than 0 or a
/// truncated one. `requested_bars: 0` is treated as 1 (a piece can't be
/// zero bars long).
fn chorus_count(requested_bars: usize) -> usize {
    requested_bars.max(1).div_ceil(BARS_PER_CHORUS)
}

/// What a chorus is doing in the piece's overall conversation. The FIRST
/// chorus of every trajectory is always [`ChorusRole::Statement`] (the
/// call/response idiom has to be established before it can be varied,
/// broken down, or resolved) -- see [`trajectory_for`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChorusRole {
    /// The call/response idiom stated plainly: response = call inverted.
    Statement,
    /// The response looks backward instead of upside-down: response =
    /// call retrograded.
    Variation,
    /// Denser, faster exchange: response = call diminished (halved
    /// durations).
    Intensification,
    /// The material fractures: response = the first half of the call's
    /// own notes (a genuine partial quotation, not a full statement).
    Breakdown,
    /// Both prior idioms combined: response = call inverted THEN
    /// retrograded -- the classical retrograde-inversion, the 4th
    /// transform `canonical_fingerprint` already treats as equivalent
    /// family to plain inversion/retrograde.
    Reconciliation,
    /// The conversation converges: response = the call itself, verbatim
    /// -- call and response now say the same thing.
    Coda,
}

impl ChorusRole {
    /// Which existing [`SectionRole`] (and so which established
    /// intensity curve -- A=establish 0.85, B=depart/tension 1.0,
    /// ReturnA=settle 0.95, C=genuine peak 1.15) this role's harmonic/
    /// dynamic character maps onto. Reuses the crate's existing intensity
    /// semantics rather than inventing new per-role values.
    ///
    /// `Intensification` maps to `SectionRole::C` specifically (fixed
    /// 2026-07-26, was `ReturnA`): a real listening session traced
    /// audible crackle to `theory_realize.rs`'s climax-doubling pass,
    /// which pulls EVERY melody note sharing the piece's single highest
    /// `section_intensity` into an extra octave-doubled voice. With
    /// `Intensification` and `Reconciliation` both at `ReturnA` (0.95),
    /// the `Statement -> Intensification -> Reconciliation` trajectory
    /// tied both choruses for that maximum, so BOTH got doubled together
    /// -- stacking on top of Intensification's already-denser (halved-
    /// duration) response. `C` (1.15) is strictly greater than every
    /// other role's mapping, so `Intensification` is now the UNIQUE max
    /// in every trajectory that contains it (verified directly against
    /// all 6 predefined trajectories, see `intensification_is_always_the_
    /// unique_intensity_peak_across_every_trajectory` below) -- which is
    /// also the musically correct fix on its own terms: a role literally
    /// named "Intensification" should outrank `Variation`/`Breakdown`
    /// (1.0), not sit below them.
    pub(crate) fn section_role(self) -> SectionRole {
        match self {
            ChorusRole::Statement => SectionRole::A,
            ChorusRole::Variation | ChorusRole::Breakdown => SectionRole::B,
            ChorusRole::Reconciliation | ChorusRole::Coda => SectionRole::ReturnA,
            ChorusRole::Intensification => SectionRole::C,
        }
    }

    /// The response motif for this role, derived from `call` alone --
    /// every arm is an existing, already-tested [`Motif`] method, not
    /// new melodic material.
    pub(crate) fn response_for(self, call: &Motif, pivot: i32) -> Motif {
        match self {
            ChorusRole::Statement => call.invert(pivot),
            ChorusRole::Variation => call.retrograde(),
            ChorusRole::Intensification => call.diminish(),
            ChorusRole::Breakdown => {
                let half = (call.len() / 2).max(1);
                call.fragment(0, half)
            }
            ChorusRole::Reconciliation => call.invert(pivot).retrograde(),
            ChorusRole::Coda => call.clone(),
        }
    }
}

/// The pre-defined, valid role sequences for each chorus count this
/// engine can realize through the live API (1-3, since `bars` is capped
/// at 36 = 3 choruses) -- the user's own specified trajectory set,
/// chosen deterministically per seed rather than always picking the
/// first option (so two different seeds at the same chorus count can
/// still develop differently). Every sequence starts with `Statement`.
const TRAJECTORIES_1: &[[ChorusRole; 1]] = &[[ChorusRole::Statement]];
const TRAJECTORIES_2: &[[ChorusRole; 2]] = &[
    [ChorusRole::Statement, ChorusRole::Variation],
    [ChorusRole::Statement, ChorusRole::Intensification],
    [ChorusRole::Statement, ChorusRole::Coda],
];
const TRAJECTORIES_3: &[[ChorusRole; 3]] = &[
    [
        ChorusRole::Statement,
        ChorusRole::Variation,
        ChorusRole::Coda,
    ],
    [
        ChorusRole::Statement,
        ChorusRole::Breakdown,
        ChorusRole::Intensification,
    ],
    [
        ChorusRole::Statement,
        ChorusRole::Intensification,
        ChorusRole::Reconciliation,
    ],
];

/// The seed-chosen role sequence for a piece with `choruses` choruses.
/// Always exactly `choruses` roles, always opens on [`ChorusRole::
/// Statement`]. `choruses` values above 3 are only reachable via direct
/// library use with `intent.bars > 36` (the live `/api/compose` caps at
/// 36 bars = 3 choruses) -- handled gracefully by extending the
/// Statement-opens/Coda-closes shape rather than indexing a table that
/// doesn't have an entry for it.
pub(crate) fn trajectory_for(choruses: usize, seed: u64) -> Vec<ChorusRole> {
    match choruses {
        0 => Vec::new(), // defensive; chorus_count's own `.max(1)` never actually produces this
        1 => TRAJECTORIES_1[0].to_vec(),
        2 => {
            let pick = (crate::hook::scramble(seed, 0x7A01_0002) as usize) % TRAJECTORIES_2.len();
            TRAJECTORIES_2[pick].to_vec()
        }
        3 => {
            let pick = (crate::hook::scramble(seed, 0x7A01_0003) as usize) % TRAJECTORIES_3.len();
            TRAJECTORIES_3[pick].to_vec()
        }
        n => {
            let mut roles = vec![ChorusRole::Statement];
            roles.extend(std::iter::repeat_n(ChorusRole::Variation, n - 2));
            roles.push(ChorusRole::Coda);
            roles
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CallResponsePlan {
    /// The `MusicalIntent::bars` the caller actually asked for.
    pub requested_bars: usize,
    /// `choruses * bars_per_chorus` -- what was actually realized, always
    /// >= `requested_bars` (rounded up to a whole chorus).
    pub realized_bars: usize,
    pub choruses: usize,
    pub bars_per_chorus: usize,
    /// The role each chorus plays, in order -- always `choruses` long,
    /// always opens on `ChorusRole::Statement`. See [`trajectory_for`].
    pub trajectory: Vec<ChorusRole>,
    /// Bar index (0-based, within a chorus) where each call phrase starts
    /// — always `[0, 4, 8]` for a 12-bar chorus's 3 four-bar lines.
    pub call_starts: Vec<usize>,
    /// Bar index (0-based, within a chorus) where each response phrase
    /// starts — always `[2, 6, 10]`.
    pub response_starts: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallResponseRealization {
    pub score: Score,
    pub plan: CallResponsePlan,
}

/// Real gap (milliseconds) held back from each note's own sounding
/// duration before the next note begins. Found necessary 2026-07-26:
/// `push_phrase` used to emit consecutive notes with literally zero gap
/// (next onset = previous onset + its full nominal duration) -- fine for
/// the score's own timing, but a real listening session traced audible
/// crackle to exactly this on a sampled wind/flute/clarinet melody line:
/// zero gap can leave a note's own release phase abruptly cut by the
/// next note-on. The artifact concentrated specifically in
/// `ChorusRole::Intensification`'s halved-duration response (~2x the
/// note-onset rate of every other role) is consistent with this -- if
/// abutting notes even occasionally clip, doubling the onset rate
/// roughly doubles the opportunities for it.
const ARTICULATION_GAP_MS: f64 = 10.0;
/// Never trim a note's SOUNDING duration below this, regardless of the
/// gap above -- protects already-brief notes from being trimmed into a
/// near-inaudible sliver. `nominal` is also never exceeded (a note is
/// never lengthened): a note already at or below this floor is left
/// completely untouched.
const MIN_NOTE_MS: f64 = 40.0;

/// The sounding duration to actually emit for a note of `nominal`
/// duration at `tempo_bpm`, after holding back [`ARTICULATION_GAP_MS`]
/// (never below [`MIN_NOTE_MS`], never above `nominal` itself). Snapped
/// to a 1/64-beat grid rather than an arbitrary float-derived value,
/// keeping this crate's exact-rational-`Duration` invariant intact
/// (see `rhythm.rs`'s own module doc on why durations are rational, not
/// float).
fn trimmed_for_articulation(nominal: Duration, tempo_bpm: f64) -> Duration {
    let nominal_ms = nominal.seconds(tempo_bpm) * 1000.0;
    let target_ms = (nominal_ms - ARTICULATION_GAP_MS)
        .max(MIN_NOTE_MS)
        .min(nominal_ms);
    if target_ms >= nominal_ms - 1e-9 {
        return nominal;
    }
    let target_beats = target_ms / 1000.0 * tempo_bpm / 60.0;
    let sixty_fourths = (target_beats * 64.0).round().max(1.0) as i64;
    Duration::new(sixty_fourths, 64)
}

/// Place `motif` starting at `start`, degrees resolved against `key`'s
/// scale at `octave`. Mirrors [`crate::opera::push_theme`]'s note-pushing
/// logic (kept as a separate, smaller copy here rather than widening that
/// function's visibility, since this version doesn't need a cutoff —
/// every call/response motif here is already capped at one bar by
/// construction, see the module doc).
///
/// `t` (the phrase's internal onset clock) always advances by each
/// note's FULL nominal duration -- only the EMITTED, sounding duration
/// is trimmed via [`trimmed_for_articulation`] -- so the phrase's own
/// overall rhythm/onset structure (and every existing bar-boundary test)
/// is completely unaffected; only the tail end of each note's audible
/// sustain shortens slightly.
fn push_phrase(
    score: &mut Score,
    motif: &Motif,
    key: Key,
    start: Duration,
    octave: i32,
    intensity: f32,
    tempo_bpm: f64,
) {
    let scale = key.scale();
    let mut t = Duration::zero();
    for n in &motif.notes {
        if let Some(d) = n.degree {
            score.push(ScoreNote {
                pitch: scale.degree_pitch(d, octave),
                onset: start + t,
                duration: trimmed_for_articulation(n.duration, tempo_bpm),
                velocity: (0.75 * intensity).clamp(0.1, 1.0),
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: intensity,
            });
        }
        t = t + n.duration;
    }
}

pub fn realize_call_response(
    intent: &MusicalIntent,
    spec: &CompositionSpec,
) -> CallResponseRealization {
    let key = spec
        .mode
        .and_then(|mode| Key::modal(intent.tonic, mode))
        .unwrap_or_else(|| {
            if intent.valence >= 0.0 {
                Key::major(intent.tonic)
            } else {
                Key::minor(intent.tonic)
            }
        });
    let tempo = spec.tempo(intent.arousal);
    let meter = spec.meter as f64;
    let bar = Duration::new(meter as i64, 1);

    // Call: the style's own established motif, hook-grafted exactly like
    // `compose_with_grammar_plan`'s own hook-cell grafting, so this stays
    // consistent with every other production path rather than inventing a
    // parallel melody-generation scheme.
    let base_motif = spec.motif(intent.arousal, intent.seed);
    let call = if spec.texture.hook_cell {
        crate::hook::graft_hook(
            &base_motif,
            &crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter),
            meter,
        )
    } else {
        base_motif
    };
    // Pivot for whichever roles' response transformation needs it
    // (Statement, Reconciliation); other roles derive their response
    // without it -- see `ChorusRole::response_for`.
    let pivot = call.notes.first().and_then(|n| n.degree).unwrap_or(1);

    let choruses = chorus_count(intent.bars);
    let trajectory = trajectory_for(choruses, intent.seed);

    let sections: Vec<Section> = (0..choruses)
        .map(|c| {
            let seed_variant = intent.seed ^ (0xCA11_u64.wrapping_mul(c as u64 + 1));
            let progression = spec.progression(BARS_PER_CHORUS, seed_variant);
            Section {
                role: trajectory[c].section_role(),
                key,
                period: Period {
                    antecedent: Phrase {
                        line: call.clone(),
                        progression: progression.degrees,
                        cadence: Cadence::Authentic,
                    },
                    consequent: Phrase {
                        line: Motif::new(Vec::new()),
                        progression: Vec::new(),
                        cadence: Cadence::Authentic,
                    },
                },
            }
        })
        .collect();
    let form = Form { sections };

    let mut score = Score::new(key, tempo, meter as u8);
    let mut prev_upper: Vec<crate::pitch::Pitch> = Vec::new();
    let mut prev_bass: Option<crate::pitch::Pitch> = None;
    let pattern = spec.accompaniment(intent.seed);
    crate::composer::realize_harmony(
        &mut score,
        &form,
        meter,
        intent,
        &mut prev_upper,
        pattern,
        false,
        false,
        false,
    );
    crate::composer::realize_bass(
        &mut score,
        &form,
        meter,
        intent,
        &mut prev_bass,
        pattern,
        false,
    );

    let octave = 5;
    let line_starts = [0usize, 4, 8]; // the 3 four-bar lines within a chorus
    let mut call_starts = Vec::new();
    let mut response_starts = Vec::new();
    for c in 0..choruses {
        let role = trajectory[c];
        let chorus_start = bar.scale((c * BARS_PER_CHORUS) as i64, 1);
        let intensity = role.section_role().intensity();
        let response = role.response_for(&call, pivot);
        for &line_bar in &line_starts {
            let call_bar = line_bar;
            let response_bar = line_bar + 2;
            push_phrase(
                &mut score,
                &call,
                key,
                chorus_start + bar.scale(call_bar as i64, 1),
                octave,
                intensity,
                tempo as f64,
            );
            push_phrase(
                &mut score,
                &response,
                key,
                chorus_start + bar.scale(response_bar as i64, 1),
                octave,
                intensity,
                tempo as f64,
            );
            if c == 0 {
                call_starts.push(call_bar);
                response_starts.push(response_bar);
            }
        }
    }

    CallResponseRealization {
        score,
        plan: CallResponsePlan {
            requested_bars: intent.bars,
            realized_bars: choruses * BARS_PER_CHORUS,
            choruses,
            bars_per_chorus: BARS_PER_CHORUS,
            trajectory,
            call_starts,
            response_starts,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn intent() -> MusicalIntent {
        MusicalIntent {
            valence: 0.0,
            arousal: 0.5,
            energy: 0.6,
            bars: 4,
            seed: 7,
            tonic: PitchClass::C,
        }
    }

    fn blues_spec() -> CompositionSpec {
        crate::style::Style::Blues.spec()
    }

    #[test]
    fn call_and_response_are_genuinely_different_material() {
        let realized = realize_call_response(&intent(), &blues_spec());
        // Reconstruct the two motifs the same way realize_call_response
        // does, and confirm they're not accidentally identical.
        let spec = blues_spec();
        let meter = spec.meter as f64;
        let base = spec.motif(intent().arousal, intent().seed);
        let call = crate::hook::graft_hook(
            &base,
            &crate::hook::HookCell::generate_with(&spec.melody, intent().seed, meter),
            meter,
        );
        let pivot = call.notes.first().and_then(|n| n.degree).unwrap_or(1);
        let response = call.invert(pivot);
        assert_ne!(
            call.degrees(),
            response.degrees(),
            "response must be genuinely different material, not a copy"
        );
        assert!(!realized.score.notes.is_empty());
    }

    #[test]
    fn response_is_the_calls_own_inversion() {
        // The response's degree sequence should be the mathematical
        // inversion of the call's -- not an unrelated random motif.
        let spec = blues_spec();
        let meter = spec.meter as f64;
        let base = spec.motif(intent().arousal, intent().seed);
        let call = crate::hook::graft_hook(
            &base,
            &crate::hook::HookCell::generate_with(&spec.melody, intent().seed, meter),
            meter,
        );
        let pivot = call.notes.first().and_then(|n| n.degree).unwrap_or(1);
        let response = call.invert(pivot);
        let expected = call.invert(pivot);
        assert_eq!(response.degrees(), expected.degrees());
    }

    #[test]
    fn call_and_response_alternate_at_the_expected_bar_offsets() {
        let realized = realize_call_response(&intent(), &blues_spec());
        assert_eq!(realized.plan.call_starts, vec![0, 4, 8]);
        assert_eq!(realized.plan.response_starts, vec![2, 6, 10]);
    }

    #[test]
    fn chorus_count_rounds_up_to_a_whole_chorus() {
        // The quantization matrix from the duration-contract fix: a
        // request never realizes FEWER bars than asked (never truncates a
        // chorus), and never realizes needlessly MORE than the next whole
        // chorus (never over-rounds).
        assert_eq!(chorus_count(4), 1, "4 bars -> 1 chorus (12 bars)");
        assert_eq!(chorus_count(12), 1, "an exact chorus stays 1 chorus");
        assert_eq!(chorus_count(13), 2, "13 bars -> rounds up to 2 choruses");
        assert_eq!(chorus_count(24), 2, "24 bars -> exactly 2 choruses");
        assert_eq!(chorus_count(36), 3, "36 bars -> exactly 3 choruses");
        assert_eq!(chorus_count(0), 1, "0 bars still realizes 1 whole chorus");
    }

    #[test]
    fn realized_bars_matches_the_quantization_policy_across_requests() {
        // End-to-end version of the unit-level chorus_count check: for
        // several requested bar counts, the REAL realize_call_response
        // output reports the same quantization the standalone function
        // predicts, and the score's own total length never exceeds it.
        for &(requested, expected_choruses) in
            &[(4usize, 1usize), (12, 1), (13, 2), (24, 2), (36, 3)]
        {
            let realized = realize_call_response(
                &MusicalIntent {
                    bars: requested,
                    ..intent()
                },
                &blues_spec(),
            );
            assert_eq!(
                realized.plan.choruses, expected_choruses,
                "requested {requested} bars"
            );
            assert_eq!(realized.plan.requested_bars, requested);
            assert_eq!(
                realized.plan.realized_bars,
                expected_choruses * BARS_PER_CHORUS
            );
            let bar = Duration::new(realized.score.meter as i64, 1);
            assert!(
                realized.score.total_beats.beats()
                    <= bar.scale(realized.plan.realized_bars as i64, 1).beats() + 1e-6,
                "requested {requested} bars: score must not run past the \
                 realized {} bars",
                realized.plan.realized_bars
            );
        }
    }

    #[test]
    fn harmony_genuinely_varies_chorus_to_chorus_not_just_the_melody() {
        // Regression test for the ArchetypePool fix: `realize_call_response`
        // has computed a distinct `seed_variant` per chorus since the
        // ChorusRole trajectory system landed, but Blues's spec used
        // `ProgressionSpec::Archetype`, which ignores its seed argument --
        // every chorus got byte-identical harmony despite genuinely varied
        // melody. Across several seeds, at least one pair of choruses in a
        // 3-chorus piece must have a different harmony pitch sequence.
        let bar_beats = 4.0;
        let chorus_beats = BARS_PER_CHORUS as f64 * bar_beats;
        let varied = (0..20u64).any(|seed| {
            let realized = realize_call_response(
                &MusicalIntent {
                    bars: 36,
                    seed,
                    ..intent()
                },
                &blues_spec(),
            );
            assert_eq!(realized.plan.choruses, 3);
            let chorus_harmony = |c: usize| -> Vec<u8> {
                let (lo, hi) = (c as f64 * chorus_beats, (c + 1) as f64 * chorus_beats);
                realized
                    .score
                    .voice(crate::score::VoiceRole::Harmony)
                    .iter()
                    .filter(|n| n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi)
                    .map(|n| n.pitch.midi())
                    .collect()
            };
            let (c0, c1, c2) = (chorus_harmony(0), chorus_harmony(1), chorus_harmony(2));
            !(c0 == c1 && c1 == c2)
        });
        assert!(
            varied,
            "expected at least one seed among 0..20 to produce genuinely \
             different harmony across the 3 choruses"
        );
    }

    #[test]
    fn melody_notes_only_ever_land_within_a_calls_or_responses_own_bar_pair() {
        // No melody note should land in the "silent" bars between a
        // call/response pair and the next line's call -- confirms
        // push_phrase's motifs stay within their own 1-bar allotment
        // (graft_hook's own meter-fitting contract) rather than spilling
        // over.
        let realized = realize_call_response(&intent(), &blues_spec());
        let meter = realized.score.meter as f64;
        let bar_beats = meter;
        for n in realized
            .score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
        {
            let bar_in_chorus = ((n.onset.beats() / bar_beats) as i64) % BARS_PER_CHORUS as i64;
            let offset_in_line = bar_in_chorus % 4;
            assert!(
                offset_in_line == 0 || offset_in_line == 2,
                "melody note at bar-in-chorus {bar_in_chorus} (offset {offset_in_line}) \
                 is not on a call or response downbeat"
            );
        }
    }

    #[test]
    fn deterministic_per_seed() {
        let a = realize_call_response(&intent(), &blues_spec());
        let b = realize_call_response(&intent(), &blues_spec());
        assert_eq!(a.score.notes.len(), b.score.notes.len());
        assert_eq!(a.plan, b.plan);
    }

    #[test]
    fn every_trajectory_has_the_right_length_and_opens_on_statement() {
        for choruses in 1..=3 {
            for seed in 0..20u64 {
                let trajectory = trajectory_for(choruses, seed);
                assert_eq!(
                    trajectory.len(),
                    choruses,
                    "choruses={choruses} seed={seed}"
                );
                assert_eq!(
                    trajectory[0],
                    ChorusRole::Statement,
                    "every trajectory must open on Statement (choruses={choruses} seed={seed})"
                );
            }
        }
    }

    #[test]
    fn trajectory_beyond_the_predefined_table_still_opens_and_closes_correctly() {
        // choruses > 3 is only reachable via direct library use (the live
        // API caps bars at 36 = 3 choruses) -- confirm the graceful
        // extension path doesn't panic and keeps the same shape.
        let trajectory = trajectory_for(5, 3);
        assert_eq!(trajectory.len(), 5);
        assert_eq!(trajectory[0], ChorusRole::Statement);
        assert_eq!(trajectory[4], ChorusRole::Coda);
    }

    #[test]
    fn each_role_produces_the_response_its_own_documented_transformation_predicts() {
        // Ground-truth test: every ChorusRole::response_for arm reuses an
        // EXISTING Motif method -- confirm each one really does, by
        // reconstructing the expected value independently rather than
        // just trusting the match arm compiles.
        //
        // Compares full `Motif` equality (degrees AND durations), not
        // just `.degrees()`. Found necessary 2026-07-26 (an independent
        // code-review workflow caught this): `diminish()`/`scale_rhythm`
        // only ever changes `duration`, never `degree` -- `m.diminish().
        // degrees() == m.degrees()` holds for ANY motif by construction,
        // so the old `.degrees()`-only assertion for Intensification
        // would have passed identically even if `response_for` returned
        // a no-op, or `call.clone()`, instead of a real diminution. Full
        // `Motif` equality (it derives `PartialEq`) actually exercises
        // the duration change each transform claims to make.
        let call = Motif::from_degrees(&[
            (1, Duration::new(1, 2)),
            (3, Duration::new(1, 2)),
            (5, Duration::new(1, 1)),
        ]);
        let pivot = 1;
        assert_eq!(
            ChorusRole::Statement.response_for(&call, pivot),
            call.invert(pivot)
        );
        assert_eq!(
            ChorusRole::Variation.response_for(&call, pivot),
            call.retrograde()
        );
        let intensification = ChorusRole::Intensification.response_for(&call, pivot);
        assert_eq!(intensification, call.diminish());
        assert_ne!(
            intensification.notes[0].duration, call.notes[0].duration,
            "Intensification must genuinely change durations, not just pass \
             degrees through unchanged"
        );
        assert_eq!(
            ChorusRole::Breakdown.response_for(&call, pivot),
            call.fragment(0, 1)
        );
        assert_eq!(
            ChorusRole::Reconciliation.response_for(&call, pivot),
            call.invert(pivot).retrograde()
        );
        assert_eq!(ChorusRole::Coda.response_for(&call, pivot), call);
    }

    #[test]
    fn intensification_is_always_the_unique_intensity_peak_across_every_trajectory() {
        // The real fix for the compounding-doubling bug: `theory_realize.
        // rs`'s climax-doubling pass pulls every melody note sharing the
        // piece's SINGLE highest `section_intensity` into an extra
        // octave-doubled voice. If two different chorus roles ever tie
        // for that maximum, BOTH choruses get doubled together --
        // confirmed as the real mechanism behind rougher audio on the
        // `Statement -> Intensification -> Reconciliation` trajectory
        // (both used to map to `SectionRole::ReturnA`). Assert directly,
        // for every predefined trajectory, that whenever `Intensification`
        // is present its intensity is the STRICT, UNIQUE maximum.
        let all_trajectories: Vec<Vec<ChorusRole>> = TRAJECTORIES_1
            .iter()
            .map(|t| t.to_vec())
            .chain(TRAJECTORIES_2.iter().map(|t| t.to_vec()))
            .chain(TRAJECTORIES_3.iter().map(|t| t.to_vec()))
            .collect();
        for trajectory in &all_trajectories {
            if !trajectory.contains(&ChorusRole::Intensification) {
                continue;
            }
            let intensities: Vec<f32> = trajectory
                .iter()
                .map(|r| r.section_role().intensity())
                .collect();
            let max = intensities.iter().cloned().fold(f32::MIN, f32::max);
            let peak_count = intensities
                .iter()
                .filter(|&&i| (i - max).abs() < 1e-6)
                .count();
            assert_eq!(
                peak_count, 1,
                "trajectory {trajectory:?} has {peak_count} roles tied for the \
                 max intensity -- Intensification must be the UNIQUE peak"
            );
            let intensification_intensity = ChorusRole::Intensification.section_role().intensity();
            assert!(
                (intensification_intensity - max).abs() < 1e-6,
                "Intensification's own intensity ({intensification_intensity}) must \
                 BE the trajectory's max ({max}) in {trajectory:?}"
            );
        }
    }

    #[test]
    fn a_three_chorus_piece_genuinely_varies_its_response_across_choruses() {
        // The real point of this whole patch: a multi-chorus piece must
        // not repeat the identical call/response exchange chorus after
        // chorus. Find a seed whose 3-chorus trajectory isn't all-
        // Statement-shaped (impossible by construction -- every
        // TRAJECTORIES_3 entry has 2 non-Statement roles) and confirm the
        // realized melody notes actually differ between at least two
        // choruses' response slots.
        let realized = realize_call_response(
            &MusicalIntent {
                bars: 36,
                ..intent()
            },
            &blues_spec(),
        );
        assert_eq!(realized.plan.choruses, 3);
        assert_eq!(realized.plan.trajectory.len(), 3);
        assert_eq!(realized.plan.trajectory[0], ChorusRole::Statement);
        // At least one of the other two choruses must use a DIFFERENT
        // role from chorus 0 -- every TRAJECTORIES_3 entry guarantees
        // this, but assert it directly against the real output rather
        // than just trusting the table.
        assert!(
            realized.plan.trajectory[1] != ChorusRole::Statement
                || realized.plan.trajectory[2] != ChorusRole::Statement,
            "a 3-chorus piece must develop beyond repeating Statement: {:?}",
            realized.plan.trajectory
        );

        // Pull the actual response notes played in chorus 0's first line
        // vs chorus 1's first line, and confirm they're not identical --
        // real, audible melodic development, not just a differently-
        // labeled repeat.
        let bar = Duration::new(realized.score.meter as i64, 1);
        let response_notes_in = |chorus: i64, response_bar: i64| -> Vec<u8> {
            let start = bar.scale(chorus * BARS_PER_CHORUS as i64 + response_bar, 1);
            let end = start + bar.scale(2, 1);
            realized
                .score
                .notes
                .iter()
                .filter(|n| {
                    n.role == VoiceRole::Melody
                        && n.onset.beats() >= start.beats() - 1e-9
                        && n.onset.beats() < end.beats() - 1e-9
                })
                .map(|n| n.pitch.midi())
                .collect()
        };
        let chorus0_response = response_notes_in(0, 2);
        let chorus1_response = response_notes_in(1, 2);
        assert_ne!(
            chorus0_response, chorus1_response,
            "chorus 0 and chorus 1's response material must genuinely differ \
             (trajectory: {:?})",
            realized.plan.trajectory
        );
    }

    #[test]
    fn articulation_trim_never_lengthens_and_respects_the_floor() {
        let tempo = 90.0;
        // A long note: trimmed by exactly the gap (within 1/64-beat
        // rounding), never lengthened.
        let long = Duration::new(2, 1); // a half note, 2 beats
        let trimmed_long = trimmed_for_articulation(long, tempo);
        assert!(trimmed_long.beats() < long.beats());
        assert!(
            trimmed_long.seconds(tempo) > long.seconds(tempo) - ARTICULATION_GAP_MS / 1000.0 - 0.02,
            "trim should be close to the intended gap, not excessive"
        );
        // A note already at/under the floor: left completely untouched --
        // trimming it further would risk an inaudible sliver.
        let brief = Duration::new(1, 32); // a 128th note at 90 BPM: ~20.8ms, under the 40ms floor
        let trimmed_brief = trimmed_for_articulation(brief, tempo);
        assert_eq!(
            trimmed_brief, brief,
            "a note already near the floor must not be trimmed shorter"
        );
        // Never exceeds nominal, for a range of realistic note lengths.
        for &(num, den) in &[(1i64, 4i64), (1, 2), (1, 1), (3, 2), (2, 1)] {
            let nominal = Duration::new(num, den);
            let trimmed = trimmed_for_articulation(nominal, tempo);
            assert!(
                trimmed.beats() <= nominal.beats() + 1e-9,
                "trimmed duration must never exceed nominal ({num}/{den})"
            );
        }
    }

    #[test]
    fn realized_melody_notes_have_a_genuine_gap_before_the_next_note() {
        // End-to-end: within each call/response's own note sequence, a
        // note's sounding end must land strictly before the next note's
        // onset -- not exactly touching it, which was the pre-fix
        // behavior a real listening session traced to audible crackle.
        let realized = realize_call_response(
            &MusicalIntent {
                bars: 36,
                ..intent()
            },
            &blues_spec(),
        );
        let mut melody: Vec<_> = realized
            .score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .collect();
        melody.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        let mut checked_adjacent_pairs = 0;
        for pair in melody.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let a_end = a.onset.beats() + a.duration.beats();
            let gap = b.onset.beats() - a_end;
            // Only meaningful for genuinely adjacent notes (back-to-back
            // within the same call or response) -- a pair separated by
            // the deliberate silence between a response and the next
            // line's call has a large, unrelated gap already.
            if gap < 0.5 {
                assert!(
                    gap > 1e-6,
                    "adjacent melody notes must have a real gap, not touch exactly: \
                     a_end={a_end:.4} b_onset={:.4}",
                    b.onset.beats()
                );
                checked_adjacent_pairs += 1;
            }
        }
        assert!(
            checked_adjacent_pairs > 0,
            "expected at least one genuinely adjacent melody note pair to check"
        );
    }
}
