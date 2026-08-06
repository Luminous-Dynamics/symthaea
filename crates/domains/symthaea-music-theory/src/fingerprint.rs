// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composition-level fingerprints shared by the diversity census
//! (`examples/diversity_census.rs`) and the Muse Atlas endpoint
//! (`symthaea-muse`'s `/api/atlas`). Extracted here so both consumers call
//! the same code rather than diverging copies.
//!
//! Two tiers:
//! - **Exact**: a hash of the canonicalized note-event stream (role, pitch,
//!   onset, duration only) — catches byte-for-byte-equivalent pieces.
//! - **Structural**: a fixed feature vector, in named layers (see [`LAYERS`]):
//!   form shape, harmonic trajectory (tonic/tonality/section deltas), a
//!   per-voice-role register/count profile as an orchestration proxy, a
//!   rhythmic duration histogram, melodic contour (direction only),
//!   tempo/meter, a **pitch-class profile** (12 bins, relative to the
//!   piece's own tonic — see below), a **melodic interval histogram** (12
//!   bins by magnitude), a **chord-quality histogram** (from the Harmony
//!   voice), a **cadence-type histogram**, and a **motif layer**
//!   (transformation-mix — see below). Layer widths are computed, not
//!   hand-counted (`LAYERS`/[`STRUCT_DIMS`] derive from named per-layer
//!   `_LEN` constants below) so adding a layer can't silently miscount an
//!   offset the way five separate hand-copied sums could.
//!
//! **Transposition invariance**: the pitch-class layer bins by
//! `(pitch_class - tonic) mod 12` — semitone distance FROM the tonic, not
//! absolute pitch class — so transposing an entire piece by N semitones
//! (which shifts both every note and the tonic by N) leaves every bin
//! unchanged. This is a genuine, information-preserving 12-bin profile
//! (which degree relative to tonic each pitch prefers), not a lossy
//! folded-distance scalar.
//!
//! **Motif layer**: built on [`crate::motif_return::compare_melodic_sequences`]
//! — the crate's existing, ground-truth-tested transformation-return
//! classifier — rather than a new detector. The melody is segmented into
//! occurrence regions at each `Emphasis::PhraseStart`; every region after
//! the first is compared against the first (the "head") and classified
//! into one of [`crate::obligation::ReturnTransformation`]'s 7 kinds when
//! it clears a similarity threshold.
//!
//! This is composition-level only — it cannot see
//! `symthaea-muse::theory_realize`'s downstream performance-realization layer
//! (swing/rubato/articulation), since that crate depends on this one, not the
//! reverse.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::chord::ChordQuality;
use crate::form::{Form, SectionRole};
use crate::harmony::Tonality;
use crate::obligation::ReturnTransformation;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};

const FORM_SLOTS: usize = 6;
const VOICE_ROLES: [VoiceRole; 4] = [
    VoiceRole::Melody,
    VoiceRole::Harmony,
    VoiceRole::Bass,
    VoiceRole::CounterMelody,
];

/// All 12 [`ChordQuality`] variants, for building the chord-quality
/// histogram — kept local to this module rather than added to `chord.rs`
/// since the ordering here is specifically the fingerprint's bin order.
const CHORD_QUALITIES: [ChordQuality; 12] = [
    ChordQuality::Major,
    ChordQuality::Minor,
    ChordQuality::Diminished,
    ChordQuality::Augmented,
    ChordQuality::Major7,
    ChordQuality::Minor7,
    ChordQuality::Dominant7,
    ChordQuality::MinorMajor7,
    ChordQuality::HalfDiminished7,
    ChordQuality::Diminished7,
    ChordQuality::Sus2,
    ChordQuality::Sus4,
];

/// All 4 [`crate::cadence::Cadence`] variants, in the cadence histogram's
/// bin order.
const CADENCE_KINDS: [crate::cadence::Cadence; 4] = [
    crate::cadence::Cadence::Authentic,
    crate::cadence::Cadence::Half,
    crate::cadence::Cadence::Plagal,
    crate::cadence::Cadence::Deceptive,
];

/// All 7 [`ReturnTransformation`] variants, in the motif-transformation
/// histogram's bin order.
const RETURN_TRANSFORMATIONS: [ReturnTransformation; 7] = [
    ReturnTransformation::Literal,
    ReturnTransformation::Transposed,
    ReturnTransformation::Inverted,
    ReturnTransformation::Augmented,
    ReturnTransformation::Diminished,
    ReturnTransformation::Fragmented,
    ReturnTransformation::Restored,
];

const FORM_LEN: usize = FORM_SLOTS * 2;
const HARMONY_LEN: usize = 2 + FORM_SLOTS;
const ORCHESTRATION_LEN: usize = VOICE_ROLES.len() * 3;
const RHYTHM_LEN: usize = 5;
const CONTOUR_LEN: usize = 3;
/// Tempo (normalized BPM) + meter (normalized beats-per-bar).
const TEMPO_METER_LEN: usize = 2;
const PITCH_CLASS_LEN: usize = 12;
const INTERVAL_LEN: usize = 12;
const CHORD_QUALITY_LEN: usize = CHORD_QUALITIES.len();
const CADENCE_LEN: usize = CADENCE_KINDS.len();
/// [`RETURN_TRANSFORMATIONS`]'s histogram (7) + occurrence count + density.
const MOTIF_LEN: usize = RETURN_TRANSFORMATIONS.len() + 2;

const FORM_START: usize = 0;
const HARMONY_START: usize = FORM_START + FORM_LEN;
const ORCHESTRATION_START: usize = HARMONY_START + HARMONY_LEN;
const RHYTHM_START: usize = ORCHESTRATION_START + ORCHESTRATION_LEN;
const CONTOUR_START: usize = RHYTHM_START + RHYTHM_LEN;
const TEMPO_METER_START: usize = CONTOUR_START + CONTOUR_LEN;
const PITCH_CLASS_START: usize = TEMPO_METER_START + TEMPO_METER_LEN;
const INTERVAL_START: usize = PITCH_CLASS_START + PITCH_CLASS_LEN;
const CHORD_QUALITY_START: usize = INTERVAL_START + INTERVAL_LEN;
const CADENCE_START: usize = CHORD_QUALITY_START + CHORD_QUALITY_LEN;
const MOTIF_START: usize = CADENCE_START + CADENCE_LEN;

/// Dimensionality of [`structural_fingerprint`]'s output vector.
pub const STRUCT_DIMS: usize = MOTIF_START + MOTIF_LEN;

/// Named layer boundaries within the structural vector — `(name, start, len)`.
pub const LAYERS: [(&str, usize, usize); 11] = [
    ("form", FORM_START, FORM_LEN),
    ("harmony", HARMONY_START, HARMONY_LEN),
    ("orchestration", ORCHESTRATION_START, ORCHESTRATION_LEN),
    ("rhythm", RHYTHM_START, RHYTHM_LEN),
    ("contour", CONTOUR_START, CONTOUR_LEN),
    ("tempo_meter", TEMPO_METER_START, TEMPO_METER_LEN),
    ("pitch_class", PITCH_CLASS_START, PITCH_CLASS_LEN),
    ("interval", INTERVAL_START, INTERVAL_LEN),
    ("chord_quality", CHORD_QUALITY_START, CHORD_QUALITY_LEN),
    ("cadence", CADENCE_START, CADENCE_LEN),
    ("motif", MOTIF_START, MOTIF_LEN),
];

/// Canonicalized-note-event hash — identical scores hash identically
/// regardless of internal event ordering.
pub fn exact_fingerprint(score: &Score) -> u64 {
    let mut events: Vec<(u8, u8, i64, i64, i64, i64)> = score
        .events()
        .iter()
        .map(|n| {
            (
                match n.role {
                    VoiceRole::Melody => 0,
                    VoiceRole::Harmony => 1,
                    VoiceRole::Bass => 2,
                    VoiceRole::CounterMelody => 3,
                },
                n.pitch.midi(),
                n.onset.num(),
                n.onset.den(),
                n.duration.num(),
                n.duration.den(),
            )
        })
        .collect();
    events.sort();
    let mut hasher = DefaultHasher::new();
    events.hash(&mut hasher);
    hasher.finish()
}

fn tonality_code(t: Tonality) -> f64 {
    match t {
        Tonality::Major => 0.0,
        Tonality::Minor => 0.5,
        Tonality::Modal(_) => 1.0,
    }
}

/// Onsets (in beats, ascending) of every `Emphasis::PhraseStart` melody
/// note — the shared, engine-agnostic segmentation cue for both the
/// bypass-form fallback and the motif layer.
fn phrase_start_onsets(score: &Score) -> Vec<f64> {
    let mut onsets: Vec<f64> = score
        .voice(VoiceRole::Melody)
        .iter()
        .filter(|n| n.emphasis == Emphasis::PhraseStart)
        .map(|n| n.onset.beats())
        .collect();
    onsets.sort_by(|a, b| a.total_cmp(b));
    onsets
}

/// Duration-weighted pitch-class profile, binned by semitone distance FROM
/// THE TONIC (`(pc - tonic) mod 12`) rather than absolute pitch class — see
/// the module doc note on transposition invariance.
fn pitch_class_profile(score: &Score) -> [f64; PITCH_CLASS_LEN] {
    let mut hist = [0.0f64; PITCH_CLASS_LEN];
    let tonic_value = score.key.tonic.value() as i32;
    for n in &score.notes {
        let pc_value = n.pitch.pitch_class().value() as i32;
        let bin = (pc_value - tonic_value).rem_euclid(12) as usize;
        hist[bin] += n.duration.beats().max(0.0);
    }
    let total: f64 = hist.iter().sum::<f64>().max(1e-9);
    for h in hist.iter_mut() {
        *h /= total;
    }
    hist
}

/// Melodic interval-magnitude histogram (semitones between consecutive
/// melody notes, absolute value, clamped into the last bin at
/// `INTERVAL_LEN - 1` semitones and up) — a real-magnitude companion to the
/// `contour` layer's direction-only up/down/same.
fn interval_histogram(score: &Score) -> [f64; INTERVAL_LEN] {
    let mut hist = [0.0f64; INTERVAL_LEN];
    let melody = score.voice(VoiceRole::Melody);
    for w in melody.windows(2) {
        let semitones =
            (w[1].pitch.midi() as i32 - w[0].pitch.midi() as i32).unsigned_abs() as usize;
        hist[semitones.min(INTERVAL_LEN - 1)] += 1.0;
    }
    let total: f64 = hist.iter().sum::<f64>().max(1.0);
    for h in hist.iter_mut() {
        *h /= total;
    }
    hist
}

/// Best-fit chord quality for an observed simultaneous pitch-class set:
/// smallest symmetric-difference tone mismatch against every (root,
/// quality) combination's theoretical tone set, ties broken toward
/// [`CHORD_QUALITIES`]'s declared order (triads before sevenths/sus
/// chords). `None` for an empty observation.
fn classify_chord_quality(observed: &[crate::pitch::PitchClass]) -> Option<ChordQuality> {
    if observed.is_empty() {
        return None;
    }
    let mut best_quality = CHORD_QUALITIES[0];
    let mut best_diff = usize::MAX;
    for root_value in 0..12 {
        let root_pc = crate::pitch::PitchClass::new(root_value);
        for &quality in &CHORD_QUALITIES {
            let theoretical = crate::chord::Chord::new(root_pc, quality).pitch_classes();
            let a_only = observed
                .iter()
                .copied()
                .filter(|pc| !theoretical.contains(pc))
                .count();
            let b_only = theoretical
                .iter()
                .copied()
                .filter(|pc| !observed.contains(pc))
                .count();
            let diff = a_only + b_only;
            if diff < best_diff {
                best_diff = diff;
                best_quality = quality;
            }
        }
    }
    Some(best_quality)
}

/// Chord-quality histogram: classifies every distinct-onset simultaneous
/// pitch-class set in the Harmony voice via [`classify_chord_quality`].
fn chord_quality_histogram(score: &Score) -> [f64; CHORD_QUALITY_LEN] {
    let mut counts = [0.0f64; CHORD_QUALITY_LEN];
    let harmony = score.voice(VoiceRole::Harmony);
    let mut i = 0;
    while i < harmony.len() {
        let onset = harmony[i].onset;
        let mut j = i;
        let mut pcs = Vec::new();
        while j < harmony.len() && harmony[j].onset == onset {
            pcs.push(harmony[j].pitch.pitch_class());
            j += 1;
        }
        pcs.sort_by_key(|pc| pc.value());
        pcs.dedup();
        if let Some(quality) = classify_chord_quality(&pcs) {
            let bin = CHORD_QUALITIES.iter().position(|&q| q == quality).unwrap();
            counts[bin] += 1.0;
        }
        i = j;
    }
    let total: f64 = counts.iter().sum::<f64>().max(1.0);
    for c in counts.iter_mut() {
        *c /= total;
    }
    counts
}

/// Reverse-lookup: which scale degree (1-based) of `key`'s scale `pc`
/// belongs to, or `None` for a chromatic tone outside the scale.
fn pitch_class_to_degree(key: &crate::harmony::Key, pc: crate::pitch::PitchClass) -> Option<i32> {
    let scale = key.scale();
    (1..=scale.mode.degree_count() as i32).find(|&d| scale.degree_pitch_class(d) == pc)
}

/// Cadence-type histogram: every `Emphasis::Cadential` bass note paired
/// with its immediately preceding bass note, converted to scale degrees
/// against the piece's home key and classified via the crate's own
/// ground-truth [`crate::cadence::Cadence::detect`]. Degree conversion
/// against the home key (not a per-section key) is a real, disclosed
/// simplification — the bypass engines this layer also serves have no
/// reliable per-section key data to use instead.
fn cadence_histogram(score: &Score) -> [f64; CADENCE_LEN] {
    let mut counts = [0.0f64; CADENCE_LEN];
    let bass = score.voice(VoiceRole::Bass);
    for idx in 1..bass.len() {
        if bass[idx].emphasis != Emphasis::Cadential {
            continue;
        }
        let (Some(penultimate), Some(final_degree)) = (
            pitch_class_to_degree(&score.key, bass[idx - 1].pitch.pitch_class()),
            pitch_class_to_degree(&score.key, bass[idx].pitch.pitch_class()),
        ) else {
            continue;
        };
        if let Some(cadence) = crate::cadence::Cadence::detect(penultimate, final_degree) {
            let bin = CADENCE_KINDS.iter().position(|&k| k == cadence).unwrap();
            counts[bin] += 1.0;
        }
    }
    let total: f64 = counts.iter().sum::<f64>().max(1.0);
    for c in counts.iter_mut() {
        *c /= total;
    }
    counts
}

/// Region bounds `(lo, hi)` in beats for each phrase-start-delimited
/// occurrence, the last one running to `total_beats`.
fn region_bounds(starts: &[f64], total_beats: f64) -> Vec<(f64, f64)> {
    starts
        .iter()
        .enumerate()
        .map(|(idx, &lo)| (lo, starts.get(idx + 1).copied().unwrap_or(total_beats)))
        .collect()
}

/// Motif transformation-mix layer: segments the melody at
/// `Emphasis::PhraseStart` markers, treats the FIRST region as the "head"
/// (opening statement), and classifies every later region against it via
/// [`crate::motif_return::compare_melodic_sequences`] — the crate's
/// existing, ground-truth-tested transformation-return classifier, not a
/// new detector. Two-call pattern per occurrence: the first call's
/// `detected_transformation` (computed independent of what `expected` was
/// passed) discovers the best-fitting kind; a second call with `expected`
/// set to that kind reads its own confidence via `overall_similarity`,
/// gated by the existing `meets_threshold(0.5)` — below threshold, the
/// region doesn't count as a motif return at all (unrelated material).
/// Output: a 7-bin histogram (fraction of counted occurrences per
/// [`ReturnTransformation`] kind) + occurrence count (normalized against a
/// ceiling of 8) + occurrence density (count / number of regions).
fn motif_transformation_layer(score: &Score) -> [f64; MOTIF_LEN] {
    let mut out = [0.0f64; MOTIF_LEN];
    let starts = phrase_start_onsets(score);
    let total_beats = score.total_beats.beats();
    if starts.len() < 2 || total_beats <= 1e-9 {
        return out;
    }
    let bounds = region_bounds(&starts, total_beats);
    let melody = score.voice(VoiceRole::Melody);
    let notes_in = |lo: f64, hi: f64| -> Vec<ScoreNote> {
        melody
            .iter()
            .copied()
            .filter(|n| n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi - 1e-9)
            .collect()
    };
    let (head_lo, head_hi) = bounds[0];
    let head = notes_in(head_lo, head_hi);
    if head.is_empty() {
        return out;
    }

    let mut histogram = [0.0f64; 7];
    let mut occurrences = 0u32;
    for &(lo, hi) in &bounds[1..] {
        let region = notes_in(lo, hi);
        if region.is_empty() {
            continue;
        }
        let discovery = crate::motif_return::compare_melodic_sequences(
            &head,
            &region,
            ReturnTransformation::Literal,
        );
        let confidence = crate::motif_return::compare_melodic_sequences(
            &head,
            &region,
            discovery.detected_transformation,
        );
        if confidence.meets_threshold(0.5) {
            let bin = RETURN_TRANSFORMATIONS
                .iter()
                .position(|&k| k == discovery.detected_transformation)
                .unwrap();
            histogram[bin] += 1.0;
            occurrences += 1;
        }
    }

    let hist_total: f64 = histogram.iter().sum::<f64>().max(1.0);
    for h in histogram.iter_mut() {
        *h /= hist_total;
    }
    out[..7].copy_from_slice(&histogram);
    out[7] = (occurrences as f64 / 8.0).clamp(0.0, 1.0);
    out[8] = (occurrences as f64 / bounds.len().max(1) as f64).clamp(0.0, 1.0);
    out
}

/// The structural fingerprint — see module docs for the layer breakdown.
pub fn structural_fingerprint(score: &Score, form: &Option<Form>) -> [f64; STRUCT_DIMS] {
    let mut v = [0.0f64; STRUCT_DIMS];
    let mut i = 0;

    if let Some(form) = form {
        let bars = crate::composer::section_bar_map(form);
        let total_bars: i64 = bars
            .iter()
            .map(|b| b.antecedent_bars + b.consequent_bars)
            .sum();
        for (slot, sb) in bars.iter().take(FORM_SLOTS).enumerate() {
            let role_code = match sb.role {
                SectionRole::A => 1.0,
                SectionRole::B => 2.0,
                SectionRole::ReturnA => 3.0,
                SectionRole::C => 4.0,
            };
            v[i + slot] = role_code / 4.0;
            let len = (sb.antecedent_bars + sb.consequent_bars) as f64;
            v[i + FORM_SLOTS + slot] = if total_bars > 0 {
                len / total_bars as f64
            } else {
                0.0
            };
        }
    } else {
        // Generic, score-derived fallback for the bypass engines (Fugue/
        // Sonata/Renaissance/Opera/ProgSuite/Passacaglia never build a
        // `Form`), so these dims degrade to real, non-zero values instead
        // of silently zero for exactly the pieces with the most
        // distinctive architecture. Section boundaries come from the
        // melody's own `Emphasis::PhraseStart` markers (present regardless
        // of which engine built the score); role identity is genuinely
        // unknown without a real `Form`, so every derived section reports
        // the same neutral code rather than guessing A/B/C — coarser than
        // the true form, not a substitute for it.
        let starts = phrase_start_onsets(score);
        let total = score.total_beats.beats();
        if !starts.is_empty() && total > 1e-9 {
            for (slot, &start) in starts.iter().take(FORM_SLOTS).enumerate() {
                let end = starts.get(slot + 1).copied().unwrap_or(total);
                v[i + slot] = 0.5;
                v[i + FORM_SLOTS + slot] = ((end - start) / total).clamp(0.0, 1.0);
            }
        }
    }
    i += FORM_SLOTS * 2;

    v[i] = score.key.tonic.value() as f64 / 12.0;
    v[i + 1] = tonality_code(score.key.tonality);
    i += 2;
    if let Some(form) = form {
        for (slot, section) in form.sections.iter().take(FORM_SLOTS).enumerate() {
            let delta = score.key.tonic.interval_to(section.key.tonic) as f64;
            v[i + slot] = delta / 12.0;
        }
    }
    i += FORM_SLOTS;

    let total_notes = score.notes.len().max(1) as f64;
    for (r, role) in VOICE_ROLES.iter().enumerate() {
        let notes = score.voice(*role);
        let count_frac = notes.len() as f64 / total_notes;
        let (center, span) = if notes.is_empty() {
            (0.0, 0.0)
        } else {
            let midis: Vec<f64> = notes.iter().map(|n| n.pitch.midi() as f64).collect();
            let lo = midis.iter().cloned().fold(f64::MAX, f64::min);
            let hi = midis.iter().cloned().fold(f64::MIN, f64::max);
            let mean = midis.iter().sum::<f64>() / midis.len() as f64;
            (mean / 127.0, (hi - lo) / 127.0)
        };
        v[i + r * 3] = count_frac;
        v[i + r * 3 + 1] = center;
        v[i + r * 3 + 2] = span;
    }
    i += VOICE_ROLES.len() * 3;

    let mut hist = [0.0f64; 5];
    for n in &score.notes {
        let beats = n.duration.beats();
        let bucket = if beats < 0.26 {
            0
        } else if beats < 0.51 {
            1
        } else if beats < 1.01 {
            2
        } else if beats < 2.01 {
            3
        } else {
            4
        };
        hist[bucket] += 1.0;
    }
    let hist_total: f64 = hist.iter().sum::<f64>().max(1.0);
    for h in hist.iter_mut() {
        *h /= hist_total;
    }
    v[i..i + 5].copy_from_slice(&hist);
    i += 5;

    let melody = score.voice(VoiceRole::Melody);
    let mut up: f64 = 0.0;
    let mut down: f64 = 0.0;
    let mut same: f64 = 0.0;
    for w in melody.windows(2) {
        match w[1].pitch.midi().cmp(&w[0].pitch.midi()) {
            std::cmp::Ordering::Greater => up += 1.0,
            std::cmp::Ordering::Less => down += 1.0,
            std::cmp::Ordering::Equal => same += 1.0,
        }
    }
    let steps = (up + down + same).max(1.0);
    v[i] = up / steps;
    v[i + 1] = down / steps;
    v[i + 2] = same / steps;
    i += 3;

    // Tempo/meter: absent from every earlier layer. Tempo normalized
    // against a fixed [30, 190] BPM span (covers the full roster, from
    // Ambient's 32 floor to Waltz's 180 ceiling, with headroom on both
    // ends for a style not yet in the roster); meter (beats per bar)
    // against [1, 16] — generous above the largest known value (Flamenco's
    // 12-beat compás) rather than tight to today's roster.
    v[i] = ((score.tempo_bpm as f64 - 30.0) / 160.0).clamp(0.0, 1.0);
    v[i + 1] = ((score.meter as f64 - 1.0) / 15.0).clamp(0.0, 1.0);
    i += TEMPO_METER_LEN;

    v[i..i + PITCH_CLASS_LEN].copy_from_slice(&pitch_class_profile(score));
    i += PITCH_CLASS_LEN;

    v[i..i + INTERVAL_LEN].copy_from_slice(&interval_histogram(score));
    i += INTERVAL_LEN;

    v[i..i + CHORD_QUALITY_LEN].copy_from_slice(&chord_quality_histogram(score));
    i += CHORD_QUALITY_LEN;

    v[i..i + CADENCE_LEN].copy_from_slice(&cadence_histogram(score));
    i += CADENCE_LEN;

    v[i..i + MOTIF_LEN].copy_from_slice(&motif_transformation_layer(score));
    i += MOTIF_LEN;

    debug_assert_eq!(i, STRUCT_DIMS);
    v
}

/// Euclidean distance between two structural fingerprints.
pub fn dist(a: &[f64; STRUCT_DIMS], b: &[f64; STRUCT_DIMS]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Per-layer L2 distances, in [`LAYERS`] order.
pub fn layer_dists(a: &[f64; STRUCT_DIMS], b: &[f64; STRUCT_DIMS]) -> [f64; LAYERS.len()] {
    let mut out = [0.0f64; LAYERS.len()];
    for (li, &(_, start, len)) in LAYERS.iter().enumerate() {
        let mut s = 0.0f64;
        for k in start..start + len {
            s += (a[k] - b[k]).powi(2);
        }
        out[li] = s.sqrt();
    }
    out
}

/// Scale a fingerprint's dimensions by `sqrt(layer_weights[layer])`, in
/// [`LAYERS`] order — equivalent to weighting each layer's contribution to
/// squared Euclidean distance by `layer_weights[layer]`. Used by Atlas
/// "lenses" (Combined/Motif&Form/Harmony/Rhythm/Orchestration) to reweight
/// the SAME fingerprints before [`project_2d`], rather than computing a new
/// embedding per lens.
pub fn weighted(v: &[f64; STRUCT_DIMS], layer_weights: &[f64; LAYERS.len()]) -> [f64; STRUCT_DIMS] {
    let mut out = *v;
    for (li, &(_, start, len)) in LAYERS.iter().enumerate() {
        let w = layer_weights[li].max(0.0).sqrt();
        for k in start..start + len {
            out[k] *= w;
        }
    }
    out
}

/// Top-2-principal-components projection via power iteration on the
/// covariance matrix — deterministic, dependency-light (no new crate),
/// good enough for a diagnostic 2D scatter of a few hundred pieces.
pub fn project_2d(vectors: &[[f64; STRUCT_DIMS]]) -> Vec<(f64, f64)> {
    let n = vectors.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![(0.0, 0.0)];
    }

    let mut mean = [0.0f64; STRUCT_DIMS];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v.iter()) {
            *m += x / n as f64;
        }
    }
    let centered: Vec<[f64; STRUCT_DIMS]> = vectors
        .iter()
        .map(|v| {
            let mut c = *v;
            for (ci, m) in c.iter_mut().zip(mean.iter()) {
                *ci -= m;
            }
            c
        })
        .collect();

    // Power iteration to find the top eigenvector of X^T X (equivalent to
    // the top principal component of the centered data), then deflate and
    // repeat for the second.
    fn power_iterate(centered: &[[f64; STRUCT_DIMS]], seed: u64) -> [f64; STRUCT_DIMS] {
        let mut v = [0.0f64; STRUCT_DIMS];
        // Deterministic pseudo-random init via a simple LCG on the seed —
        // avoids depending on the `rand` crate for a one-shot projection.
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for x in v.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = ((state >> 33) as f64 / u32::MAX as f64) - 0.5;
        }
        for _ in 0..100 {
            let mut next = [0.0f64; STRUCT_DIMS];
            for row in centered {
                let dot: f64 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                for (n, r) in next.iter_mut().zip(row.iter()) {
                    *n += dot * r;
                }
            }
            let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            for x in next.iter_mut() {
                *x /= norm;
            }
            v = next;
        }
        v
    }

    let pc1 = power_iterate(&centered, 42);
    // Deflate: remove the pc1 component from each row before finding pc2.
    let deflated: Vec<[f64; STRUCT_DIMS]> = centered
        .iter()
        .map(|row| {
            let dot: f64 = row.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
            let mut d = *row;
            for (di, p) in d.iter_mut().zip(pc1.iter()) {
                *di -= dot * p;
            }
            d
        })
        .collect();
    let pc2 = power_iterate(&deflated, 1337);

    centered
        .iter()
        .map(|row| {
            let x: f64 = row.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
            let y: f64 = row.iter().zip(pc2.iter()).map(|(a, b)| a * b).sum();
            (x, y)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composer::compose_with_spec_and_form;
    use crate::score::PartId;
    use crate::{Duration, MusicalIntent, Pitch, PitchClass, Style};

    #[test]
    fn exact_fingerprint_is_deterministic() {
        let intent = MusicalIntent {
            valence: 0.2,
            arousal: 0.5,
            energy: 0.5,
            bars: 8,
            seed: 7,
            tonic: PitchClass::new(0),
        };
        let (score1, _) = compose_with_spec_and_form(&intent, &Style::Folk.spec());
        let (score2, _) = compose_with_spec_and_form(&intent, &Style::Folk.spec());
        assert_eq!(exact_fingerprint(&score1), exact_fingerprint(&score2));
    }

    #[test]
    fn different_seeds_usually_differ() {
        let base = MusicalIntent {
            valence: 0.2,
            arousal: 0.5,
            energy: 0.5,
            bars: 8,
            seed: 1,
            tonic: PitchClass::new(0),
        };
        let other = MusicalIntent { seed: 2, ..base };
        let (s1, f1) = compose_with_spec_and_form(&base, &Style::Folk.spec());
        let (s2, f2) = compose_with_spec_and_form(&other, &Style::Folk.spec());
        let fp1 = structural_fingerprint(&s1, &f1);
        let fp2 = structural_fingerprint(&s2, &f2);
        assert!(dist(&fp1, &fp2) > 0.0);
    }

    #[test]
    fn project_2d_handles_small_inputs() {
        assert_eq!(project_2d(&[]).len(), 0);
        assert_eq!(project_2d(&[[0.0; STRUCT_DIMS]]).len(), 1);
    }

    #[test]
    fn weighted_zeroes_out_a_muted_layer() {
        let v = [1.0f64; STRUCT_DIMS];
        // Zero out layer index 1 ("harmony"), keep everything else at 1.0.
        let mut weights = [1.0; LAYERS.len()];
        weights[1] = 0.0;
        let w = weighted(&v, &weights);
        let (_, start, len) = LAYERS[1];
        for k in start..start + len {
            assert_eq!(w[k], 0.0);
        }
        assert_eq!(w[0], 1.0); // form layer untouched
    }

    #[test]
    fn weighted_scales_distance_contribution() {
        // Two vectors differing only in the "rhythm" layer (index 3).
        let a = [0.0f64; STRUCT_DIMS];
        let mut b = [0.0f64; STRUCT_DIMS];
        let (_, start, _len) = LAYERS[3];
        b[start] = 1.0;
        let base_dist = dist(&a, &b);
        assert!(base_dist > 0.0);

        // Weighting rhythm to 0 should collapse that distance to ~0.
        let mut muted = [1.0; LAYERS.len()];
        muted[3] = 0.0;
        let wa = weighted(&a, &muted);
        let wb = weighted(&b, &muted);
        assert!(dist(&wa, &wb) < 1e-9);

        // Weighting rhythm up to 4.0 should scale distance by sqrt(4)=2.
        let mut boosted = [1.0; LAYERS.len()];
        boosted[3] = 4.0;
        let wa2 = weighted(&a, &boosted);
        let wb2 = weighted(&b, &boosted);
        assert!((dist(&wa2, &wb2) - base_dist * 2.0).abs() < 1e-9);
    }

    #[test]
    fn project_2d_separates_distinct_vectors() {
        let mut a = [0.0f64; STRUCT_DIMS];
        let mut b = [0.0f64; STRUCT_DIMS];
        a[0] = 1.0;
        b[0] = -1.0;
        b[1] = 1.0;
        let proj = project_2d(&[a, b, [0.0; STRUCT_DIMS]]);
        assert_eq!(proj.len(), 3);
        let d = ((proj[0].0 - proj[1].0).powi(2) + (proj[0].1 - proj[1].1).powi(2)).sqrt();
        assert!(
            d > 0.5,
            "expected distinct vectors to project apart, got dist {d}"
        );
    }

    #[test]
    fn tempo_and_meter_are_ground_truth_present_in_their_own_layer() {
        let a = Score::new(crate::harmony::Key::major(PitchClass::new(0)), 90.0, 4);
        let mut b = a.clone();
        b.tempo_bpm = 150.0;
        b.meter = 3;
        let fp_a = structural_fingerprint(&a, &None);
        let fp_b = structural_fingerprint(&b, &None);
        let (_, start, len) = LAYERS[5];
        assert_eq!(LAYERS[5].0, "tempo_meter");
        assert_eq!(len, 2);
        // Every OTHER layer must be identical (empty scores, same key) —
        // tempo/meter differences must land ONLY in their own layer.
        for k in 0..start {
            assert_eq!(
                fp_a[k], fp_b[k],
                "layer before tempo_meter must be untouched"
            );
        }
        assert_ne!(fp_a[start], fp_b[start], "tempo dimension must differ");
        assert_ne!(
            fp_a[start + 1],
            fp_b[start + 1],
            "meter dimension must differ"
        );

        // Ground truth on the normalization formula itself.
        assert!((fp_a[start] - (90.0 - 30.0) / 160.0).abs() < 1e-9);
        assert!((fp_a[start + 1] - (4.0 - 1.0) / 15.0).abs() < 1e-9);
        assert!((fp_b[start] - (150.0 - 30.0) / 160.0).abs() < 1e-9);
        assert!((fp_b[start + 1] - (3.0 - 1.0) / 15.0).abs() < 1e-9);

        // Two otherwise-identical pieces used to fingerprint IDENTICALLY
        // when they differed only in tempo/meter (the bug this layer
        // fixes) — now they don't.
        assert!(dist(&fp_a, &fp_b) > 0.0);
    }

    fn note(pitch: Pitch, onset_beats: i64, dur_beats: i64, role: VoiceRole) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch,
            onset: Duration::new(onset_beats, 1),
            duration: Duration::new(dur_beats, 1),
            velocity: 0.7,
            role,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    #[test]
    fn pitch_class_profile_is_transposition_invariant() {
        let c_major = crate::harmony::Key::major(PitchClass::new(0));
        let mut a = Score::new(c_major, 100.0, 4);
        a.push(note(
            Pitch::new(PitchClass::new(0), 4),
            0,
            1,
            VoiceRole::Melody,
        )); // C
        a.push(note(
            Pitch::new(PitchClass::new(4), 4),
            1,
            1,
            VoiceRole::Melody,
        )); // E
        a.push(note(
            Pitch::new(PitchClass::new(7), 4),
            2,
            1,
            VoiceRole::Melody,
        )); // G

        // The identical melodic shape, transposed up 5 semitones, in a key
        // whose tonic is ALSO transposed up 5 semitones (F major).
        let f_major = crate::harmony::Key::major(PitchClass::new(5));
        let mut b = Score::new(f_major, 100.0, 4);
        b.push(note(
            Pitch::new(PitchClass::new(5), 4),
            0,
            1,
            VoiceRole::Melody,
        )); // F
        b.push(note(
            Pitch::new(PitchClass::new(9), 4),
            1,
            1,
            VoiceRole::Melody,
        )); // A
        b.push(note(
            Pitch::new(PitchClass::new(0), 5),
            2,
            1,
            VoiceRole::Melody,
        )); // C (up an octave)

        let fp_a = structural_fingerprint(&a, &None);
        let fp_b = structural_fingerprint(&b, &None);
        let (_, start, len) = LAYERS[6];
        assert_eq!(LAYERS[6].0, "pitch_class");
        assert_eq!(len, 12);
        for k in start..start + len {
            assert!(
                (fp_a[k] - fp_b[k]).abs() < 1e-9,
                "bin {k}: pitch-class profile must be transposition-invariant, got {} vs {}",
                fp_a[k],
                fp_b[k]
            );
        }
        // And it isn't just trivially all-zero: the profile is non-uniform.
        assert!(fp_a[start..start + len].iter().any(|&x| x > 0.0));
    }

    #[test]
    fn interval_histogram_ground_truth() {
        let key = crate::harmony::Key::major(PitchClass::new(0));
        let mut s = Score::new(key, 100.0, 4);
        s.push(note(
            Pitch::new(PitchClass::new(0), 4),
            0,
            1,
            VoiceRole::Melody,
        ));
        s.push(note(
            Pitch::new(PitchClass::new(4), 4),
            1,
            1,
            VoiceRole::Melody,
        )); // +4
        s.push(note(
            Pitch::new(PitchClass::new(7), 4),
            2,
            1,
            VoiceRole::Melody,
        )); // +3
        let fp = structural_fingerprint(&s, &None);
        let (_, start, len) = LAYERS[7];
        assert_eq!(LAYERS[7].0, "interval");
        assert_eq!(len, 12);
        assert!(
            (fp[start + 4] - 0.5).abs() < 1e-9,
            "one of two intervals is +4"
        );
        assert!(
            (fp[start + 3] - 0.5).abs() < 1e-9,
            "one of two intervals is +3"
        );
    }

    #[test]
    fn chord_quality_histogram_recognizes_a_major_triad() {
        let key = crate::harmony::Key::major(PitchClass::new(0));
        let mut s = Score::new(key, 100.0, 4);
        for pc in [0, 4, 7] {
            s.push(note(
                Pitch::new(PitchClass::new(pc), 3),
                0,
                2,
                VoiceRole::Harmony,
            ));
        }
        let fp = structural_fingerprint(&s, &None);
        let (_, start, len) = LAYERS[8];
        assert_eq!(LAYERS[8].0, "chord_quality");
        assert_eq!(len, 12);
        let major_bin = CHORD_QUALITIES
            .iter()
            .position(|&q| q == ChordQuality::Major)
            .unwrap();
        assert!(
            (fp[start + major_bin] - 1.0).abs() < 1e-9,
            "the only chord is a major triad"
        );
    }

    #[test]
    fn cadence_histogram_recognizes_an_authentic_cadence() {
        let key = crate::harmony::Key::major(PitchClass::new(0));
        let mut s = Score::new(key, 100.0, 4);
        // Bass: G (degree 5, dominant) -> C (degree 1, tonic, Cadential).
        s.push(note(
            Pitch::new(PitchClass::new(7), 2),
            0,
            1,
            VoiceRole::Bass,
        ));
        let mut final_note = note(Pitch::new(PitchClass::new(0), 2), 1, 1, VoiceRole::Bass);
        final_note.emphasis = Emphasis::Cadential;
        s.push(final_note);
        let fp = structural_fingerprint(&s, &None);
        let (_, start, len) = LAYERS[9];
        assert_eq!(LAYERS[9].0, "cadence");
        assert_eq!(len, 4);
        let authentic_bin = CADENCE_KINDS
            .iter()
            .position(|&c| c == crate::cadence::Cadence::Authentic)
            .unwrap();
        assert!((fp[start + authentic_bin] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn motif_layer_finds_a_literal_return() {
        let key = crate::harmony::Key::major(PitchClass::new(0));
        let mut s = Score::new(key, 100.0, 4);
        // Same 4-note shape (degrees C-E-G-C), twice, back to back: the
        // second occurrence is an exact literal repeat of the first.
        let shape = [0i32, 4, 7, 0];
        for region_offset in [0i64, 4] {
            for (step, &pc) in shape.iter().enumerate() {
                let mut n = note(
                    Pitch::new(PitchClass::new(pc), 4),
                    region_offset + step as i64,
                    1,
                    VoiceRole::Melody,
                );
                if step == 0 {
                    n.emphasis = Emphasis::PhraseStart;
                }
                s.push(n);
            }
        }
        s.total_beats = Duration::new(8, 1);
        let fp = structural_fingerprint(&s, &None);
        let (_, start, len) = LAYERS[10];
        assert_eq!(LAYERS[10].0, "motif");
        assert_eq!(len, 9);
        let literal_bin = RETURN_TRANSFORMATIONS
            .iter()
            .position(|&t| t == ReturnTransformation::Literal)
            .unwrap();
        assert!(
            (fp[start + literal_bin] - 1.0).abs() < 1e-9,
            "the exact repeat must classify as Literal, got layer {:?}",
            &fp[start..start + len]
        );
        assert!(fp[start + 7] > 0.0, "occurrence count must be non-zero");
    }

    #[test]
    fn motif_layer_is_zero_with_fewer_than_two_phrase_starts() {
        let key = crate::harmony::Key::major(PitchClass::new(0));
        let s = Score::new(key, 100.0, 4);
        let fp = structural_fingerprint(&s, &None);
        let (_, start, len) = LAYERS[10];
        assert!(fp[start..start + len].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn bypass_form_fallback_derives_real_proportions_from_phrase_starts() {
        let key = crate::harmony::Key::major(PitchClass::new(0));
        let mut s = Score::new(key, 100.0, 4);
        let mut first = note(Pitch::new(PitchClass::new(0), 4), 0, 1, VoiceRole::Melody);
        first.emphasis = Emphasis::PhraseStart;
        s.push(first);
        let mut second = note(Pitch::new(PitchClass::new(0), 4), 6, 1, VoiceRole::Melody);
        second.emphasis = Emphasis::PhraseStart;
        s.push(second);
        s.total_beats = Duration::new(8, 1);

        // form = None, exactly the bypass-engine case (Fugue/Sonata/...).
        let fp = structural_fingerprint(&s, &None);
        let (_, start, len) = LAYERS[0];
        assert_eq!(LAYERS[0].0, "form");
        assert_eq!(len, FORM_SLOTS * 2);
        // Role codes: unknown without a real Form -> neutral 0.5, not 0.
        assert_eq!(fp[start], 0.5);
        assert_eq!(fp[start + 1], 0.5);
        // Proportions: section 0 spans [0,6) of 8 total = 0.75; section 1
        // spans [6,8) = 0.25 -- real, non-zero values instead of the old
        // hard-zero degradation for every bypass-engine piece.
        assert!((fp[start + FORM_SLOTS] - 0.75).abs() < 1e-9);
        assert!((fp[start + FORM_SLOTS + 1] - 0.25).abs() < 1e-9);
    }
}
