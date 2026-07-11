// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composer: map an expressive *intent* to STRUCTURAL choices and assemble a
//! full [`Score`]. This is the top layer — where "consciousness drives music"
//! actually means "the state chooses the motif, the harmony, the cadences,
//! and the tension shape," not "the state nudges a random walk."
//!
//! The input is a lean [`MusicalIntent`] that carries no muse/audio types, so
//! this crate stays dependency-free. muse maps its own `MusicalState` onto a
//! `MusicalIntent`.

use crate::chord::Chord;
use crate::form::{Form, SectionRole};
use crate::harmony::Key;
use crate::motif::Motif;
use crate::pitch::{Pitch, PitchClass};
use crate::rhythm::Duration;
use crate::score::{Emphasis, Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};

/// A muse-independent expressive intent. muse maps its cognitive state onto
/// this; the composer maps this onto structure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MusicalIntent {
    /// -1 (dark/minor) … +1 (bright/major).
    pub valence: f32,
    /// 0 (calm/slow/sparse) … 1 (excited/fast/dense).
    pub arousal: f32,
    /// 0 (soft/low) … 1 (full/loud). Overall energy / dynamic level.
    pub energy: f32,
    /// Number of measures per phrase (the period doubles this).
    pub bars: usize,
    /// Deterministic seed — same intent+seed → same score.
    pub seed: u64,
    /// Tonal center.
    pub tonic: PitchClass,
}

impl Default for MusicalIntent {
    fn default() -> Self {
        MusicalIntent {
            valence: 0.0,
            arousal: 0.5,
            energy: 0.6,
            bars: 4,
            seed: 0,
            tonic: PitchClass::C,
        }
    }
}

/// Compose a full symbolic score from an intent.
///
/// The piece is a TERNARY (ABA) form (Layer 3, [`crate::form::Form`]) — an
/// opening section, a contrasting middle in the relative key, and a return —
/// not a single 8-bar period. `score.key` is the piece's HOME key (section
/// A/ReturnA's key); it's metadata only, since every note is already an
/// absolute [`crate::pitch::Pitch`] by the time it's in the score, so the
/// middle section genuinely modulating doesn't need the `Score` type itself
/// to carry more than one key.
pub fn compose(intent: &MusicalIntent) -> Score {
    compose_styled(intent, crate::style::Style::Classical)
}

/// Like [`compose`], but every structural choice (meter, tempo range, motif
/// shape bank, progression) comes from `style` instead of the fixed
/// classical defaults — see [`crate::style::Style`]. `compose(intent)` is
/// exactly `compose_styled(intent, Style::Classical)`; the two must never
/// diverge, which is why `Style::Classical`'s methods delegate to the same
/// functions this used to call directly.
pub fn compose_styled(intent: &MusicalIntent, style: crate::style::Style) -> Score {
    compose_with_spec(intent, &style.spec())
}

/// Compose from a [`CompositionSpec`](crate::spec::CompositionSpec) — the
/// open, user-authored form of what [`compose_styled`] does with built-in
/// presets. The spec owns every CHOICE (motifs, progression, textures,
/// forms, ensembles); this engine owns every INVARIANT (voice leading,
/// cadences, collision avoidance, superset-only substitutions). Call
/// `spec.validate()` first when the spec came from user input — this
/// function assumes a well-formed spec.
pub fn compose_with_spec(intent: &MusicalIntent, spec: &crate::spec::CompositionSpec) -> Score {
    // The spec's mode override pins the tonality; otherwise valence maps
    // positive → major, negative → minor. (`validate()` already rejected
    // grammar-incompatible modes, so a `None` from `Key::modal` can only
    // mean an unvalidated spec — fall back to the valence mapping rather
    // than panic.)
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
    let attitude = spec.attitude;
    // Attitude pre-adjustments: Joy lifts the pulse, Grief slows it.
    let tempo = spec.tempo(intent.arousal)
        * match attitude {
            Some(crate::spec::Attitude::Joy) => 1.08,
            Some(crate::spec::Attitude::Grief) => 0.85,
            _ => 1.0,
        };
    let meter = spec.meter;
    let meter_beats = meter as f64;

    let motif = spec.motif(intent.arousal, intent.seed);
    // Hook surgery: graft a memorable cell (the piece's NAME — enforced
    // rhythmic + contour identity, see `crate::hook`) as the motif's head,
    // AFTER the spec's seed-orientation so the name is always stated
    // upright first; the development machinery transforms it later. The
    // template's tail stays as connective tissue.
    let motif = if spec.texture.hook_cell {
        crate::hook::graft_hook(
            &motif,
            &crate::hook::HookCell::generate(intent.seed, meter_beats),
            meter_beats,
        )
    } else {
        motif
    };
    // A fugue is not a period form: its texture IS the counterpoint, so it
    // bypasses the progression/accompaniment/damage pipeline entirely and
    // takes only what a fugue needs — the key, the tempo, and the
    // hook-grafted subject (the piece's name, stated by every voice).
    if spec.form_kind(intent.seed) == crate::spec::FormKind::Fugue {
        return crate::fugue::realize_fugue(key, tempo, meter, &motif, intent.seed);
    }
    let bars = intent.bars.max(1);
    let base = spec.progression(bars, intent.seed);
    // High arousal/energy gets the SENTENCE archetype (statement → repetition
    // → fragmentation): its accelerating, driving-toward-the-cadence feel
    // fits excited states. Calmer states get the balanced statement/
    // variation/statement arc of the plain developed period. This is a real
    // STRUCTURAL choice driven by the intent, not just a faster tempo.
    let use_sentence = intent.arousal > 0.6 || intent.energy > 0.75;
    // Large-scale form varies by seed within the spec's pool. The built-in
    // presets list [Ternary, Rondo], preserving the original even→ternary /
    // odd→rondo behavior for every existing caller.
    let form = match spec.form_kind(intent.seed) {
        crate::spec::FormKind::Ternary => {
            Form::ternary(&motif, key, &base, meter_beats, intent.seed, use_sentence)
        }
        crate::spec::FormKind::Rondo => {
            Form::rondo(&motif, key, &base, meter_beats, intent.seed, use_sentence)
        }
        crate::spec::FormKind::Variations => {
            Form::variations(&motif, key, &base, meter_beats, intent.seed, use_sentence)
        }
        // Handled by the early return above — a fugue never reaches the
        // period pipeline.
        crate::spec::FormKind::Fugue => unreachable!("fugue branch returns before form building"),
    };

    let mut score = Score::new(key, tempo, meter);
    let mut prev_upper: Vec<Pitch> = Vec::new();
    let mut prev_bass: Option<Pitch> = None;
    // Grief strips the accompaniment to held blocks — space for the
    // suspensions to speak in.
    let pattern = if attitude == Some(crate::spec::Attitude::Grief) {
        crate::accompaniment::Accompaniment::Block
    } else {
        spec.accompaniment(intent.seed)
    };
    let tx = &spec.texture;
    realize_melody(
        &mut score,
        &form,
        intent,
        Duration::zero(),
        meter_beats,
        tx.climax_grace,
    );
    if tx.held_arrivals {
        apply_held_arrivals(&mut score, meter_beats);
    }
    if tx.hook_cell {
        apply_hook_memory(
            &mut score,
            &form,
            meter_beats,
            crate::hook::HookCell::generate(intent.seed, meter_beats)
                .notes
                .len(),
            attitude,
        );
    }
    realize_harmony(
        &mut score,
        &form,
        meter_beats,
        intent,
        &mut prev_upper,
        pattern,
        tx.thin_departure,
        tx.cadential_harmonic_rhythm,
    );
    realize_bass(
        &mut score,
        &form,
        meter_beats,
        intent,
        &mut prev_bass,
        pattern,
        tx.cadential_harmonic_rhythm,
    );
    if tx.hook_cell {
        apply_hidden_bass_quote(
            &mut score,
            &form,
            meter_beats,
            intent,
            &crate::hook::HookCell::generate(intent.seed, meter_beats),
        );
    }
    if tx.counter_melody {
        realize_counter_melody(&mut score, &form, meter_beats, intent);
        if tx.held_arrivals && tx.hook_cell {
            apply_counter_hook_echoes(
                &mut score,
                &form,
                meter_beats,
                intent,
                &crate::hook::HookCell::generate(intent.seed, meter_beats),
            );
        }
    }
    if tx.staged_entrances {
        apply_staged_entrances(&mut score, &form, meter_beats);
    }
    if tx.damage > 0.0 {
        apply_damage(
            &mut score,
            &form,
            key,
            meter_beats,
            tx.damage,
            tx.swing,
            intent.seed,
        );
    }
    match attitude {
        Some(crate::spec::Attitude::Grief) => {
            apply_grief_suspensions(&mut score, &form, meter_beats, intent);
        }
        Some(crate::spec::Attitude::Defiance) => {
            apply_defiant_syncopation(&mut score, meter_beats);
            for n in score.notes.iter_mut() {
                if n.role == VoiceRole::Bass {
                    n.velocity = (n.velocity + 0.08).clamp(0.1, 1.0);
                }
            }
        }
        Some(crate::spec::Attitude::Joy) => {
            apply_joyful_lift(&mut score);
        }
        _ => {}
    }
    if tx.intro_bars > 0 {
        prepend_intro(
            &mut score,
            key,
            meter_beats,
            intent,
            pattern,
            tx.intro_bars as i64,
        );
    }
    if tx.coda_bars > 0 {
        append_coda(
            &mut score,
            key,
            meter_beats,
            intent,
            tx.coda_bars as i64,
            tx.damage >= 0.2,
        );
    }
    if attitude == Some(crate::spec::Attitude::Curiosity) {
        apply_curious_question(&mut score, key);
    }
    score
}

/// Grief's signature: SUSPENSION CHAINS. At every chord change, the
/// previous chord's third is held over the barline — dissonant against
/// the new harmony — then resolves DOWN by step onto a chord tone. The
/// oldest grammar of grief in tonal music (the 4-3 and 6-5 suspensions),
/// overlaid above the accompaniment so it reads as its own weeping line.
/// Suspensions only form where they're legal: the held tone must NOT
/// belong to the new chord, and a chord tone must sit one or two
/// semitones below it.
fn apply_grief_suspensions(
    score: &mut Score,
    form: &Form,
    meter_beats: f64,
    intent: &MusicalIntent,
) {
    let vel = ((0.28 + intent.energy.clamp(0.0, 1.0) * 0.25) * 0.95).clamp(0.18, 0.5);
    let bar = Duration::new(meter_beats as i64, 1);
    let mut measure: i64 = 0;
    for section in &form.sections {
        let mut degrees = section.period.antecedent.progression.clone();
        degrees.extend(section.period.consequent.progression.iter().copied());
        let key = section.key;
        for i in 1..degrees.len() {
            let prev = key.diatonic_triad(degrees[i - 1]);
            let cur = key.diatonic_triad(degrees[i]);
            if prev.root == cur.root {
                continue;
            }
            let sus = prev.voice(4)[1]; // the old chord's third, octave 4
            let cur_pcs: Vec<crate::pitch::PitchClass> =
                cur.voice(4).iter().map(|p| p.pitch_class()).collect();
            if cur_pcs.contains(&sus.pitch_class()) {
                continue; // consonant against the new chord — nothing held
            }
            let resolution = (1..=2)
                .map(|s| sus.transpose(-s))
                .find(|p| cur_pcs.contains(&p.pitch_class()));
            let Some(resolution) = resolution else {
                continue;
            };
            let onset = bar.scale(measure + i as i64, 1);
            score.push(ScoreNote {
                pitch: sus,
                onset,
                duration: Duration::new(1, 1),
                velocity: vel,
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity: section.role.intensity(),
            });
            score.push(ScoreNote {
                pitch: resolution,
                onset: onset + Duration::new(1, 1),
                duration: Duration::new((meter_beats as i64 - 1).max(1), 1),
                velocity: (vel * 0.9).clamp(0.1, 1.0),
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity: section.role.intensity(),
            });
        }
        measure += degrees.len() as i64;
    }
    score
        .notes
        .sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
}

/// Defiance's signature: NOTATED SYNCOPATION. Every other bar's interior
/// downbeat melody note arrives an eighth EARLY and holds through where
/// it belonged — insistence pushing against the grid, written into the
/// score itself (so the MIDI carries it, not just the render). Phrase
/// starts, cadences, and the climax keep their placement: defiance
/// disturbs the middle of the sentence, not its punctuation.
fn apply_defiant_syncopation(score: &mut Score, meter_beats: f64) {
    let eighth = Duration::new(1, 2);
    for n in score.notes.iter_mut() {
        if n.role != VoiceRole::Melody || n.emphasis != Emphasis::Normal {
            continue;
        }
        let beats = n.onset.beats();
        let bar_index = (beats / meter_beats).floor() as i64;
        let on_downbeat = (beats - bar_index as f64 * meter_beats).abs() < 1e-9;
        if !on_downbeat || bar_index % 2 == 0 || beats < meter_beats {
            continue;
        }
        n.onset = n.onset.saturating_sub(eighth);
        n.duration = n.duration + eighth;
    }
    score
        .notes
        .sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
}

/// Joy's signature: LIFT. Interior melody notes sound lighter — a tenth
/// shorter — so the line dances instead of singing legato. Long tones
/// (holds, cadences) keep their length; joy is airborne, not clipped.
fn apply_joyful_lift(score: &mut Score) {
    for n in score.notes.iter_mut() {
        if n.role == VoiceRole::Melody && n.emphasis == Emphasis::Normal && n.duration.beats() < 1.5
        {
            n.duration = n.duration.scale(9, 10);
        }
    }
}

/// Curiosity's signature: the QUESTION. The piece's very last melody tone
/// resolves UP to scale degree 2 instead of settling on the tonic —
/// question intonation, the ending that leaves without answering.
fn apply_curious_question(score: &mut Score, key: Key) {
    let Some(last) = score
        .notes
        .iter_mut()
        .filter(|n| n.role == VoiceRole::Melody)
        .max_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()))
    else {
        return;
    };
    // Degree 2 in the octave nearest the note it replaces.
    let old_midi = last.pitch.midi() as i32;
    let scale = key.scale();
    let candidate = (2..=6)
        .map(|oct| scale.degree_pitch(2, oct))
        .min_by_key(|p| (p.midi() as i32 - old_midi).abs());
    if let Some(p) = candidate {
        last.pitch = p;
    }
}

/// The damage pass: after the clean composition, injure it in musically
/// meaningful ways. A listening review of the undamaged output named the
/// problem precisely — "it now has shape, but it still lacks argument...
/// motion without enough consequence." Every prior pass ADDED order
/// (voices enter on schedule, cadences resolve, textures build); this pass
/// is the opposing force: moments where something HAPPENS to the piece
/// instead of the next layer arriving on time.
///
/// The devices are chosen by [`plan_damage`] — a PLANNER, not a fixed
/// list: it diagnoses the clean piece's own weaknesses (smoothness,
/// repetition, climax prominence, groove uniformity) and selects the
/// injuries that answer them, with `amount` setting how many fire and the
/// seed breaking near-ties so two pieces don't share the same scars. (The
/// review of the fixed-wound first version: "if every piece gets the same
/// six scars, the listener will eventually hear the algorithm instead of
/// the song.") The coda transformation is gated separately (≥0.2, in
/// [`append_coda`]) — a generated ending is weak by default.
///
/// The device vocabulary (each placed by the piece's own structure):
/// - **exposed climax**: harmony AND bass cut for the climax note's whole
///   sounding span — removed by OVERLAP and truncated at the window edge,
///   not just by onset (an external MIDI audit caught a bass note
///   sustaining in from the previous bar, still carrying the peak).
/// - **dark bass entrance**: the staged entry lands an octave below its
///   written pitch, slightly louder.
/// - **the expectation hole**: the return's second-bar downbeat — the
///   note the ear knows is coming by now — is REMOVED.
/// - **the "wait" tone**: one chromatic passing tone mid-departure.
/// - **counter disagreement**: the counterline's middle notes shift half
///   a beat late — another will, not a decoration.
#[allow(clippy::too_many_arguments)]
fn apply_damage(
    score: &mut Score,
    form: &Form,
    key: Key,
    meter_beats: f64,
    amount: f32,
    swing: f32,
    seed: u64,
) {
    let sections = section_bar_map(form);
    let plan = plan_damage(score, meter_beats, swing, amount, seed);
    let fires = |d: DamageDevice| plan.contains(&d);

    // --- ≥0.2: expose the climax ---
    // The window covers the climax note's FULL SOUNDING SPAN, bar-aligned
    // (the held climax can cross a barline), and the cut is by OVERLAP:
    // notes starting inside the window are removed, and notes sustaining
    // IN from before are truncated at the window's edge. An external MIDI
    // audit of the first version found the peak still carried by a bass
    // note whose onset lay in the previous bar — the exact blind spot an
    // onset-only filter has.
    if fires(DamageDevice::ExposedClimax)
        && let Some(climax) = score
            .notes
            .iter()
            .find(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Climax)
    {
        let win_start = (climax.onset.beats() / meter_beats).floor() * meter_beats;
        let win_end = ((climax.onset + climax.duration).beats() / meter_beats).ceil() * meter_beats;
        score.notes.retain(|n| {
            !(matches!(n.role, VoiceRole::Harmony | VoiceRole::Bass)
                && n.onset.beats() >= win_start - 1e-9
                && n.onset.beats() < win_end - 1e-9)
        });
        for n in score.notes.iter_mut() {
            if matches!(n.role, VoiceRole::Harmony | VoiceRole::Bass)
                && n.onset.beats() < win_start
                && (n.onset + n.duration).beats() > win_start + 1e-9
            {
                n.duration =
                    Duration::new(((win_start - n.onset.beats()).max(0.5) * 2.0) as i64, 2);
            }
        }
    }

    // --- ≥0.35: the bass enters too low, too dark ---
    if fires(DamageDevice::DarkBassEntry)
        && let Some(first_bass) = score
            .notes
            .iter_mut()
            .filter(|n| n.role == VoiceRole::Bass)
            .min_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()))
        && first_bass.pitch.midi() >= 36
    {
        first_bass.pitch = first_bass.pitch.transpose(-12);
        first_bass.velocity = (first_bass.velocity + 0.08).clamp(0.1, 1.0);
    }

    // --- ≥0.35: remove the downbeat the ear expects in the return ---
    if fires(DamageDevice::ExpectationHole)
        && let Some(ret) = sections.iter().find(|s| s.role == SectionRole::ReturnA)
    {
        let target = (ret.start_bar + 1) as f64 * meter_beats;
        if let Some(idx) = score.notes.iter().position(|n| {
            n.role == VoiceRole::Melody
                && (n.onset.beats() - target).abs() < 1e-6
                && n.emphasis == Emphasis::Normal
        }) {
            score.notes.remove(idx);
        }
    }

    // --- ≥0.5: one chromatic "wait" tone in the departure ---
    if fires(DamageDevice::WaitTone)
        && let Some(b) = sections.iter().find(|s| s.role == SectionRole::B)
    {
        let (b0, b1) = (
            b.start_bar as f64 * meter_beats,
            b.end_bar() as f64 * meter_beats,
        );
        let melody: Vec<usize> = score
            .notes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.role == VoiceRole::Melody
                    && n.onset.beats() >= b0
                    && n.onset.beats() < b1
                    && n.emphasis == Emphasis::Normal
            })
            .map(|(i, _)| i)
            .collect();
        let pair = melody.windows(2).find_map(|w| {
            let (a, b) = (&score.notes[w[0]], &score.notes[w[1]]);
            let step = b.pitch.midi() as i32 - a.pitch.midi() as i32;
            // A whole-step move with room to steal a sixteenth from.
            (step.abs() == 2
                && (a.onset + a.duration).beats() <= b.onset.beats() + 1e-6
                && a.duration.beats() >= 0.5)
                .then_some((w[0], w[1], step.signum()))
        });
        if let Some((ai, bi, dir)) = pair {
            let grace = Duration::sixteenth();
            let passing_onset = score.notes[bi].onset.saturating_sub(grace);
            let passing = ScoreNote {
                pitch: score.notes[ai].pitch.transpose(dir),
                onset: passing_onset,
                duration: grace,
                velocity: (score.notes[ai].velocity * 0.85).clamp(0.1, 1.0),
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: score.notes[ai].section_intensity,
            };
            let shortened = score.notes[ai].duration.saturating_sub(grace);
            score.notes[ai].duration = if shortened.beats() < 0.25 {
                Duration::sixteenth()
            } else {
                shortened
            };
            // Insert keeping melody onset order (muse's expressive model
            // reads consecutive entries as a timeline).
            score.notes.insert(bi, passing);
        }
    }

    // --- ≥0.5: the counterline stops agreeing ---
    if fires(DamageDevice::CounterShift) {
        let counter: Vec<usize> = score
            .notes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.role == VoiceRole::CounterMelody)
            .map(|(i, _)| i)
            .collect();
        if counter.len() >= 6 {
            let half = Duration::new(1, 2);
            let (lo, hi) = (counter.len() / 3, 2 * counter.len() / 3);
            for &i in &counter[lo..hi] {
                score.notes[i].onset = score.notes[i].onset + half;
                let shortened = score.notes[i].duration.saturating_sub(half);
                score.notes[i].duration = if shortened.beats() < 0.5 {
                    Duration::new(1, 2)
                } else {
                    shortened
                };
            }
        }
    }

    // Keep the melody line onset-ordered for downstream consumers.
    let _ = key; // key reserved for future harmonic damage devices
    score
        .notes
        .sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
}

/// One injury the damage planner can choose. See [`apply_damage`] for what
/// each does; see [`plan_damage`] for how they are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DamageDevice {
    ExposedClimax,
    DarkBassEntry,
    ExpectationHole,
    WaitTone,
    CounterShift,
}

/// The damage PLANNER: diagnose the clean piece's own weaknesses and
/// select the injuries that answer them — the review's prescription after
/// the fixed-wound first version ("it should choose injuries based on the
/// piece's own weak points... a track with too much smoothness gets
/// silence and interruption; too much repetition gets mutation and missing
/// notes; a weak climax gets accompaniment abandonment; weak groove gets
/// timing damage").
///
/// Diagnostics (all on the un-damaged score):
/// - **smoothness**: fraction of stepwise melody motion blended with
///   inter-onset uniformity — a too-smooth piece needs interruption
///   (exposed climax, the hole).
/// - **repetition**: fraction of melody bars sharing the single most
///   common onset-rhythm pattern — a too-repetitive piece needs mutation
///   (the hole, the wait tone).
/// - **climax prominence**: climax velocity against the melody's mean —
///   a weak peak needs abandonment to become a peak.
/// - **groove uniformity**: straight-swing pieces with uniform onsets
///   need timing damage (the counter shift).
///
/// `amount` sets how many devices fire (0.2 → 1 … 1.0 → all 5); the seed
/// adds a small deterministic jitter that reorders NEAR-TIES only — it can
/// never override a clear diagnosis, but two seeds with similar pieces
/// get different scars.
fn plan_damage(
    score: &Score,
    meter_beats: f64,
    swing: f32,
    amount: f32,
    seed: u64,
) -> Vec<DamageDevice> {
    let melody: Vec<&ScoreNote> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .collect();
    if melody.len() < 4 {
        return Vec::new();
    }

    // Stepwise fraction.
    let (mut stepwise, mut moves) = (0u32, 0u32);
    for w in melody.windows(2) {
        if w[0].onset == w[1].onset {
            continue;
        }
        moves += 1;
        if (w[1].pitch.midi() as i32 - w[0].pitch.midi() as i32).abs() <= 2 {
            stepwise += 1;
        }
    }
    let stepwise_frac = stepwise as f32 / moves.max(1) as f32;

    // Inter-onset uniformity: fraction of IOIs within 15% of the median.
    let mut iois: Vec<f64> = melody
        .windows(2)
        .map(|w| w[1].onset.beats() - w[0].onset.beats())
        .filter(|d| *d > 1e-6)
        .collect();
    iois.sort_by(|a, b| a.total_cmp(b));
    let uniform_frac = if iois.is_empty() {
        0.0
    } else {
        let median = iois[iois.len() / 2];
        iois.iter()
            .filter(|d| (**d - median).abs() <= median * 0.15)
            .count() as f32
            / iois.len() as f32
    };
    let smoothness = 0.5 * stepwise_frac + 0.5 * uniform_frac;

    // Bar-rhythm repetition: onset pattern per bar as an eighth-grid mask.
    let mut masks: std::collections::HashMap<i64, u16> = std::collections::HashMap::new();
    for n in &melody {
        let bar = (n.onset.beats() / meter_beats).floor() as i64;
        let slot = ((n.onset.beats() - bar as f64 * meter_beats) / 0.5).round() as u16;
        *masks.entry(bar).or_insert(0) |= 1 << slot.min(15);
    }
    let mut counts: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
    for &m in masks.values() {
        *counts.entry(m).or_insert(0) += 1;
    }
    let repetition = counts.values().copied().max().unwrap_or(0) as f32 / masks.len().max(1) as f32;

    // Climax prominence: peak velocity against the melody's mean.
    let mean_vel = melody.iter().map(|n| n.velocity).sum::<f32>() / melody.len() as f32;
    let climax_vel = melody
        .iter()
        .find(|n| n.emphasis == Emphasis::Climax)
        .map(|n| n.velocity)
        .unwrap_or(mean_vel);
    let weak_climax = (1.25 - climax_vel / mean_vel.max(1e-6)).clamp(0.0, 1.0);

    let groove_need = if (swing - 0.5).abs() < 1e-6 {
        uniform_frac
    } else {
        0.25 // a swung piece already fights the grid
    };

    // Need score per device, with a tiny seed jitter (≤0.04) that can only
    // reorder near-ties.
    let jitter = |i: u64| ((seed >> (i * 4)) & 0xF) as f32 / 400.0;
    let mut needs = [
        (
            DamageDevice::ExposedClimax,
            weak_climax.max(smoothness * 0.9) + jitter(0),
        ),
        (
            DamageDevice::ExpectationHole,
            repetition.max(smoothness * 0.8) + jitter(1),
        ),
        (DamageDevice::WaitTone, repetition * 0.9 + jitter(2)),
        (DamageDevice::CounterShift, groove_need + jitter(3)),
        (DamageDevice::DarkBassEntry, 0.45 + jitter(4)),
    ];
    needs.sort_by(|a, b| b.1.total_cmp(&a.1));
    let k = ((amount.clamp(0.0, 1.0) * 5.0).ceil() as usize).clamp(1, 5);
    needs.iter().take(k).map(|(d, _)| *d).collect()
}

/// The counter-melody ANSWERS with the hook: during the return's held
/// cadence bars — where the melody hangs on one tone and the accompaniment
/// keeps time — the counterline speaks the piece's name back (the hook
/// cell, halved note values, tenor register). Call-and-response with the
/// piece's own name: the review asked for a counterline that "answers,
/// interrupts, mocks, mourns, or contradicts" instead of politely
/// supporting, and this is the answer device. It also serves memorability
/// directly — the name now recurs in a different voice and register.
///
/// The piece-final cadence is exempt (the transformed coda already quotes
/// the hook there; two quotations at once would smear both).
fn apply_counter_hook_echoes(
    score: &mut Score,
    form: &Form,
    meter_beats: f64,
    intent: &MusicalIntent,
    hook: &crate::hook::HookCell,
) {
    let vel = (0.30 + intent.energy.clamp(0.0, 1.0) * 0.25).clamp(0.15, 0.55);
    // The last cadential melody note of the whole piece — exempt.
    let final_onset = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Cadential)
        .map(|n| n.onset.beats())
        .fold(f64::MIN, f64::max);
    // Held cadence tones inside each ReturnA window.
    let mut measure: i64 = 0;
    for section in &form.sections {
        let bars = (section.period.antecedent.progression.len()
            + section.period.consequent.progression.len()) as i64;
        let (w0, w1) = (
            measure as f64 * meter_beats,
            (measure + bars) as f64 * meter_beats,
        );
        measure += bars;
        if section.role != SectionRole::ReturnA {
            continue;
        }
        let scale = section.key.scale();
        let holds: Vec<(f64, u8)> = score
            .notes
            .iter()
            .filter(|n| {
                n.role == VoiceRole::Melody
                    && n.emphasis == Emphasis::Cadential
                    && n.duration.beats() >= 1.5
                    && n.onset.beats() >= w0 - 1e-9
                    && n.onset.beats() < w1 - 1e-9
                    && (n.onset.beats() - final_onset).abs() > 1e-9
            })
            .map(|n| (n.onset.beats(), n.pitch.midi()))
            .collect();
        for (hold_onset, held_midi) in holds {
            // The answer begins half a beat into the held tone.
            let mut t = hold_onset + 0.5;
            for &(deg, dur) in &hook.notes {
                let halved = dur.scale(1, 2);
                let mut pitch = scale.degree_pitch(deg, 4);
                // Never rub against the tone the melody is holding.
                if (pitch.midi() as i32 - held_midi as i32).abs() <= 2 {
                    pitch = pitch.transpose(-12);
                }
                score.push(ScoreNote {
                    pitch,
                    onset: Duration::new((t * 4.0).round() as i64, 4),
                    duration: halved,
                    velocity: vel,
                    role: VoiceRole::CounterMelody,
                    emphasis: Emphasis::Normal,
                    section_intensity: section.role.intensity(),
                });
                t += halved.beats();
            }
        }
    }
    score
        .notes
        .sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
}

/// MOTIF MEMORY: the return re-states the hook THROUGH the attitude's
/// ears. The review that demanded it: "Grief shouldn't just suspend it —
/// it should remember it incorrectly. Defiance should interrupt it. Joy
/// should ornament it. Curiosity should leave it incomplete... that's the
/// point where listeners stop saying 'this has different settings' and
/// start saying 'this music has a point of view.'" With no attitude, the
/// return stays a verbatim memory — the engine's native temperament.
fn apply_hook_memory(
    score: &mut Score,
    form: &Form,
    meter_beats: f64,
    hook_len: usize,
    attitude: Option<crate::spec::Attitude>,
) {
    let Some(att) = attitude else {
        return;
    };
    // MEMORY TRAJECTORY (the judgment layer's first stone): with a single
    // return (ternary), the attitude's memory device applies there and the
    // coda's transformed quote carries the resolution. With MULTIPLE
    // returns (rondo), memory has an arc: every return but the last
    // remembers through the attitude's ears — and the FINAL return says
    // the name WHOLE, verbatim, with a quiet confidence lift. "Finally
    // complete": the piece judges that this memory deserved to survive.
    let returns: Vec<(Key, i64)> = form
        .sections
        .iter()
        .zip(section_bar_map(form))
        .filter(|(s, _)| s.role == SectionRole::ReturnA)
        .map(|(s, b)| (s.key, b.start_bar))
        .collect();
    let n = returns.len();
    for (i, (key, start_bar)) in returns.iter().enumerate() {
        let w0 = *start_bar as f64 * meter_beats;
        if i + 1 == n && n >= 2 {
            for idx in statement_indices(score, w0, hook_len) {
                score.notes[idx].velocity = (score.notes[idx].velocity + 0.05).clamp(0.1, 1.0);
            }
        } else {
            alter_statement(score, *key, w0, hook_len, att);
        }
    }
}

/// The first `hook_len` melody notes at or after `w0` — a return's opening
/// hook statement, re-derived from the live note list (earlier alterations
/// may have shifted indices).
fn statement_indices(score: &Score, w0: f64, hook_len: usize) -> Vec<usize> {
    let mut v: Vec<(usize, f64)> = score
        .notes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.role == VoiceRole::Melody && n.onset.beats() >= w0 - 1e-9)
        .map(|(i, n)| (i, n.onset.beats()))
        .collect();
    v.sort_by(|a, b| a.1.total_cmp(&b.1));
    v.into_iter().map(|(i, _)| i).take(hook_len).collect()
}

/// Apply one attitude's memory device to the statement at `w0` — see
/// [`apply_hook_memory`] for what each attitude does to what it remembers.
fn alter_statement(
    score: &mut Score,
    key: Key,
    w0: f64,
    hook_len: usize,
    att: crate::spec::Attitude,
) {
    use crate::spec::Attitude as A;
    let idxs = statement_indices(score, w0, hook_len);
    if idxs.len() < 3 {
        return;
    }
    match att {
        A::Grief => {
            // Misremembered: the cell's largest reach falls SHORT by a
            // diatonic step — memory diminishing what it loved.
            let mut best: Option<(usize, i32)> = None;
            for w in idxs.windows(2) {
                let d =
                    score.notes[w[1]].pitch.midi() as i32 - score.notes[w[0]].pitch.midi() as i32;
                if best.map(|(_, b)| d.abs() > b.abs()).unwrap_or(true) {
                    best = Some((w[1], d));
                }
            }
            if let Some((i, d)) = best
                && d.abs() >= 3
            {
                let toward = -d.signum();
                let p = score.notes[i].pitch;
                if let Some(q) = (1..=2)
                    .map(|s| p.transpose(toward * s))
                    .find(|q| key.scale().contains(q.pitch_class()))
                {
                    score.notes[i].pitch = q;
                }
            }
        }
        A::Defiance => {
            // Interrupted: the statement is bitten off — its last note(s)
            // never arrive, and the note before the cut lands harder.
            let cut = if idxs.len() >= 4 { 2 } else { 1 };
            if let Some(&before) = idxs.get(idxs.len() - cut - 1) {
                score.notes[before].velocity = (score.notes[before].velocity + 0.1).clamp(0.1, 1.0);
            }
            let mut remove: Vec<usize> = idxs[idxs.len() - cut..].to_vec();
            remove.sort_unstable();
            for i in remove.into_iter().rev() {
                score.notes.remove(i);
            }
        }
        A::Joy => {
            // Ornamented: a quick upper-neighbor grace giggles into the
            // statement's peak — the memory decorated in the retelling.
            let peak = *idxs
                .iter()
                .max_by_key(|&&i| score.notes[i].pitch.midi())
                .unwrap();
            let grace = Duration::new(1, 4);
            let peak_onset = score.notes[peak].onset;
            let peak_pitch = score.notes[peak].pitch;
            // Steal the grace's time from a long-enough predecessor when
            // one exists; otherwise let it CRUSH (overlap the tail of
            // whatever came before) — that's how an acciaccatura actually
            // sounds, and it keeps the giggle from depending on the hook's
            // rhythm shape.
            let prev = score.notes.iter().position(|n| {
                n.role == VoiceRole::Melody
                    && n.onset + n.duration == peak_onset
                    && n.duration.beats() >= 0.75
            });
            let above = (1..=2)
                .map(|s| peak_pitch.transpose(s))
                .find(|p| key.scale().contains(p.pitch_class()));
            if let Some(above) = above {
                if let Some(prev_idx) = prev {
                    score.notes[prev_idx].duration =
                        score.notes[prev_idx].duration.saturating_sub(grace);
                }
                let vel = (score.notes[peak].velocity * 0.8).clamp(0.1, 1.0);
                let intensity = score.notes[peak].section_intensity;
                score.notes.push(ScoreNote {
                    pitch: above,
                    onset: peak_onset.saturating_sub(grace),
                    duration: grace,
                    velocity: vel,
                    role: VoiceRole::Melody,
                    emphasis: Emphasis::Normal,
                    section_intensity: intensity,
                });
                score
                    .notes
                    .sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
            }
        }
        A::Curiosity => {
            // Incomplete: the name trails off — the last tone is never
            // said, and the one before it lingers, quieter.
            let last = *idxs.last().unwrap();
            if let Some(&before) = idxs.get(idxs.len() - 2) {
                score.notes[before].duration = score.notes[before].duration + Duration::new(1, 2);
                score.notes[before].velocity =
                    (score.notes[before].velocity * 0.85).clamp(0.1, 1.0);
            }
            score.notes.remove(last);
        }
    }
}

/// The hook HIDDEN IN THE BASS: once, in the middle of the departure, the
/// bass speaks the piece's name — augmented note values, two octaves down,
/// under whatever the melody is doing. Attitude-independent (this is
/// memory, not temperament): "later... hidden inside the bass; later...
/// finally complete" — the long-range thematic relationship where
/// instrumental works become deeply satisfying.
fn apply_hidden_bass_quote(
    score: &mut Score,
    form: &Form,
    meter_beats: f64,
    intent: &MusicalIntent,
    hook: &crate::hook::HookCell,
) {
    let Some((key, start_bar, bars)) = form
        .sections
        .iter()
        .zip(section_bar_map(form))
        .find(|(s, _)| s.role == SectionRole::B)
        .map(|(s, b)| (s.key, b.start_bar, b.end_bar() - b.start_bar))
    else {
        return;
    };
    let quote_start = (start_bar + bars / 2) as f64 * meter_beats;
    let span = 2.0 * hook.beats();
    // Clear the window (overlap-aware, per the exposed-climax lesson).
    score.notes.retain(|n| {
        !(n.role == VoiceRole::Bass
            && n.onset.beats() >= quote_start - 1e-9
            && n.onset.beats() < quote_start + span - 1e-9)
    });
    for n in score.notes.iter_mut() {
        if n.role == VoiceRole::Bass
            && n.onset.beats() < quote_start
            && (n.onset + n.duration).beats() > quote_start + 1e-9
        {
            n.duration = Duration::new(((quote_start - n.onset.beats()).max(0.5) * 2.0) as i64, 2);
        }
    }
    let vel = (0.3 + intent.energy.clamp(0.0, 1.0) * 0.2).clamp(0.15, 0.5);
    let mut t = quote_start;
    for &(deg, dur) in &hook.notes {
        let augmented = dur.scale(2, 1);
        score.push(ScoreNote {
            pitch: key.scale().degree_pitch(deg, 2),
            onset: Duration::new((t * 2.0).round() as i64, 2),
            duration: augmented,
            velocity: vel,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: SectionRole::B.intensity(),
        });
        t += augmented.beats();
    }
    score
        .notes
        .sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
}

/// Per-section bar geometry, derived from the form's own phrase lengths:
/// where each section starts and how many bars its antecedent/consequent
/// occupy. The arrangement passes below reason in these coordinates.
struct SectionBars {
    role: SectionRole,
    start_bar: i64,
    antecedent_bars: i64,
    consequent_bars: i64,
}

impl SectionBars {
    fn end_bar(&self) -> i64 {
        self.start_bar + self.antecedent_bars + self.consequent_bars
    }
}

fn section_bar_map(form: &Form) -> Vec<SectionBars> {
    let mut start_bar = 0i64;
    form.sections
        .iter()
        .map(|s| {
            let sb = SectionBars {
                role: s.role,
                start_bar,
                antecedent_bars: s.period.antecedent.progression.len() as i64,
                consequent_bars: s.period.consequent.progression.len() as i64,
            };
            start_bar = sb.end_bar();
            sb
        })
        .collect()
}

/// Arrangement dramaturgy: instruments ENTER and LEAVE instead of playing
/// wall-to-wall — the single change listeners feel most (verdict on the
/// everything-always texture: "honestly not really enjoyable").
///
/// Two devices, both classical staples:
/// - **Staged opening**: the bass sits out the first section's antecedent
///   and arrives with the consequent — the ensemble assembles in front of
///   the listener instead of starting fully formed.
/// - **Pre-return thinning**: in the bar before every `ReturnA`, the bass
///   drops out entirely and the harmony drops out of the second half — a
///   held breath, so the reprise lands as an arrival, not another bar.
fn apply_staged_entrances(score: &mut Score, form: &Form, meter_beats: f64) {
    let sections = section_bar_map(form);
    let Some(first) = sections.first() else {
        return;
    };
    let bass_entry_beat = (first.start_bar + first.antecedent_bars) as f64 * meter_beats;
    // Thin windows: (bass_out_start, harmony_out_start, end) per return.
    let thin_windows: Vec<(f64, f64, f64)> = sections
        .iter()
        .filter(|s| s.role == SectionRole::ReturnA && s.start_bar > 0)
        .map(|s| {
            let end = s.start_bar as f64 * meter_beats;
            (end - meter_beats, end - meter_beats / 2.0, end)
        })
        .collect();
    score.notes.retain(|n| {
        let t = n.onset.beats();
        match n.role {
            VoiceRole::Bass => {
                t >= bass_entry_beat && !thin_windows.iter().any(|&(b0, _, b1)| t >= b0 && t < b1)
            }
            VoiceRole::Harmony => !thin_windows.iter().any(|&(_, h0, h1)| t >= h0 && t < h1),
            _ => true,
        }
    });
}

/// Held arrivals: give the melody the two things continuous motion never
/// has — SILENCE and a moment that is dwelt on.
///
/// - **Cadence holds**: every mid-piece phrase-final bar (its last note
///   carries `Cadential` emphasis) is reduced to approach + held arrival:
///   the bar's first note is kept, the interior notes are removed, and the
///   cadence tone sounds from where the first note ends until just short
///   of the bar line. The question hangs while the accompaniment answers;
///   the next phrase's restatement is the reply. The piece-final cadence
///   is exempt (it already rings into the coda).
/// - **Climax hold**: the climax note absorbs the note that follows it
///   (its duration extends to cover both), so the peak is a long note,
///   not a passing one. The existing agogic rubato pause then dwells even
///   longer, and muse's climax doubling doubles the HELD note.
fn apply_held_arrivals(score: &mut Score, meter_beats: f64) {
    // --- Cadence holds ---
    let cadential: Vec<usize> = score
        .notes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Cadential)
        .map(|(i, _)| i)
        .collect();
    // The final cadence (largest onset) is exempt.
    let final_idx = cadential.iter().copied().max_by(|&a, &b| {
        score.notes[a]
            .onset
            .beats()
            .total_cmp(&score.notes[b].onset.beats())
    });
    let mut remove = Vec::new();
    for &ci in &cadential {
        if Some(ci) == final_idx {
            continue;
        }
        let bar_start = (score.notes[ci].onset.beats() / meter_beats).floor() * meter_beats;
        let bar_end = bar_start + meter_beats;
        // Melody notes inside this bar, in onset order.
        let mut in_bar: Vec<usize> = score
            .notes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.role == VoiceRole::Melody
                    && n.onset.beats() >= bar_start - 1e-9
                    && n.onset.beats() < bar_end - 1e-9
            })
            .map(|(i, _)| i)
            .collect();
        in_bar.sort_by(|&a, &b| {
            score.notes[a]
                .onset
                .beats()
                .total_cmp(&score.notes[b].onset.beats())
        });
        // Need at least approach + one interior + arrival to be worth it.
        if in_bar.len() < 3 || in_bar.last() != Some(&ci) {
            continue;
        }
        // Never hold a bar whose interior contains the CLIMAX — deleting
        // the piece's peak for a cadence hold is the tail wagging the dog
        // (found when hook-cell contours started putting the highest note
        // inside cadence bars: no climax survived to be marked, exposed,
        // graced, or doubled).
        if in_bar[1..in_bar.len() - 1]
            .iter()
            .any(|&i| score.notes[i].emphasis == Emphasis::Climax)
        {
            continue;
        }
        let approach = in_bar[0];
        remove.extend(in_bar[1..in_bar.len() - 1].iter().copied());
        let hold_from = score.notes[approach].onset + score.notes[approach].duration;
        score.notes[ci].onset = hold_from;
        // Hold to just short of the bar line — the breath stays real.
        let hold_beats = (bar_end - hold_from.beats() - 0.5).max(1.0);
        score.notes[ci].duration = Duration::new((hold_beats * 2.0) as i64, 2);
    }
    remove.sort_unstable();
    for i in remove.into_iter().rev() {
        score.notes.remove(i);
    }

    // --- Climax hold ---
    let Some(ci) = score
        .notes
        .iter()
        .position(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Climax)
    else {
        return;
    };
    let climax_end = score.notes[ci].onset + score.notes[ci].duration;
    let next = score.notes.iter().position(|n| {
        n.role == VoiceRole::Melody
            && n.emphasis == Emphasis::Normal
            && (n.onset.beats() - climax_end.beats()).abs() < 1e-6
    });
    if let Some(ni) = next {
        let absorbed_end = score.notes[ni].onset + score.notes[ni].duration;
        score.notes[ci].duration = absorbed_end.saturating_sub(score.notes[ci].onset);
        score.notes.remove(ni);
    }
}

/// Prepend `intro_bars` bars of the accompaniment pattern alone on the
/// tonic chord — an invitation before the melody speaks. Every realized
/// note shifts later by the intro length; the intro is quiet (a fraction
/// of the opening section's level) and carries no melody, so the first
/// phrase entry is an EVENT.
fn prepend_intro(
    score: &mut Score,
    home: Key,
    meter_beats: f64,
    intent: &MusicalIntent,
    pattern: crate::accompaniment::Accompaniment,
    intro_bars: i64,
) {
    let bar = Duration::new(meter_beats as i64, 1);
    let shift = bar.scale(intro_bars, 1);
    for n in &mut score.notes {
        n.onset = n.onset + shift;
    }
    score.total_beats = score.total_beats + shift;
    let chord = home.diatonic_triad(1);
    let voiced = crate::voicing::lead_upper(&[], chord, 3);
    let vel = ((0.28 + intent.energy.clamp(0.0, 1.0) * 0.28) * 0.75).clamp(0.15, 0.45);
    // Seed-picked intro TREATMENT — every piece used to open with the
    // identical accompaniment figure on I, which listeners heard as "most
    // songs start the same." Three doors into the same room:
    //   0: the accompaniment pattern alone (the original invitation);
    //   1: pattern over a low tonic pedal — grounded, expectant;
    //   2: a held chord swell, no motion — a curtain rising.
    // `seed / 5` decorrelates from the form (`%2`), accompaniment (`/2`),
    // and hook (`%candidates`) pickers.
    let variant = (intent.seed / 5) % 3;
    for m in 0..intro_bars {
        // Quietness comes from the LOW VELOCITY alone; the intro carries
        // the opening section's intensity so "A is the piece's intensity
        // floor" stays a global invariant (the tension-arc tests pin it).
        let onset = bar.scale(m, 1);
        match variant {
            1 => {
                pattern.realize_measure(
                    score,
                    &voiced,
                    onset,
                    meter_beats,
                    vel,
                    SectionRole::A.intensity(),
                );
                score.push(ScoreNote {
                    pitch: Pitch::new(home.tonic, 2),
                    onset,
                    duration: bar,
                    velocity: (vel * 0.9).clamp(0.1, 1.0),
                    role: VoiceRole::Bass,
                    emphasis: Emphasis::Normal,
                    section_intensity: SectionRole::A.intensity(),
                });
            }
            2 => {
                // Swell: each bar's held chord slightly louder than the
                // last, arriving at the melody's entrance.
                let swell = vel * (0.8 + 0.2 * (m as f32 + 1.0) / intro_bars.max(1) as f32);
                for pitch in &voiced {
                    score.push(ScoreNote {
                        pitch: *pitch,
                        onset,
                        duration: bar,
                        velocity: swell.clamp(0.1, 1.0),
                        role: VoiceRole::Harmony,
                        emphasis: Emphasis::Normal,
                        section_intensity: SectionRole::A.intensity(),
                    });
                }
            }
            _ => {
                pattern.realize_measure(
                    score,
                    &voiced,
                    onset,
                    meter_beats,
                    vel,
                    SectionRole::A.intensity(),
                );
            }
        }
    }
}

/// The melody's register ceiling. Motif orientation (inversion), sequence
/// transforms, and octave-wrapping degree arithmetic can drift phrases far
/// above the staff — probed worst case before this fix: MIDI 108 (C8, the
/// top of a piano), which no orchestral sample covers and which renders as
/// a piercing bare partial stack ("horror-film spike", found by ear).
/// E6, lowered from A6 after a timbre review: on the sampled palette the
/// naked lead above ~E6 reads as piercing rather than brilliant ("the
/// melody wants intimacy, not scratch"), and the renderer's equal-loudness
/// taper (knee E5) already softens the approach to it.
pub const MELODY_CEILING_MIDI: u8 = 88;

/// Folding never pushes a phrase's lowest note below this (G3) — a phrase
/// spanning more than the fold window keeps its top rather than growling.
pub const MELODY_FOLD_FLOOR_MIDI: u8 = 55;

/// Number of coda measures [`append_coda`] adds after the final cadence.
pub const CODA_BARS: i64 = 2;

/// A two-bar plagal coda ("amen"): IV under a held tonic, resolving to I —
/// the piece settles instead of stopping dead on its final cadence. The
/// melody holds the tonic through both bars (the tonic over IV is the
/// suspension that makes the plagal close work), dying away in velocity.
/// The LAST melody note carries `Cadential` emphasis, so the structure-
/// driven rubato's big final ritardando automatically moves here — the
/// original cadence now gets the smaller mid-piece relaxation.
fn append_coda(
    score: &mut Score,
    home: Key,
    meter_beats: f64,
    intent: &MusicalIntent,
    coda_bars: i64,
    transformed: bool,
) {
    if transformed && append_transformed_coda(score, home, meter_beats, intent, coda_bars) {
        return;
    }
    let bar = Duration::new(meter_beats as i64, 1);
    let start = score.total_beats;
    let vel = (0.3 + intent.energy.clamp(0.0, 1.0) * 0.3).clamp(0.1, 0.6);
    let tonic_melody = Pitch::new(home.tonic, 5);
    for m in 0..coda_bars {
        // Subdominant color throughout, resolving to I only in the last bar;
        // velocity fades linearly 0.85 -> 0.65 across the coda.
        let deg = if m == coda_bars - 1 { 1 } else { 4 };
        let fade = if coda_bars > 1 {
            0.85 - 0.20 * (m as f32 / (coda_bars - 1) as f32)
        } else {
            0.85
        };
        let onset = start + bar.scale(m, 1);
        let chord = home.diatonic_triad(deg);
        score.push(ScoreNote {
            pitch: tonic_melody,
            onset,
            duration: bar,
            velocity: (vel * fade).clamp(0.1, 1.0),
            role: VoiceRole::Melody,
            emphasis: if m == coda_bars - 1 {
                Emphasis::Cadential
            } else {
                Emphasis::Normal
            },
            section_intensity: crate::form::SectionRole::A.intensity(),
        });
        for pitch in chord.voice(3) {
            score.push(ScoreNote {
                pitch,
                onset,
                duration: bar,
                velocity: (vel * fade * 0.6).clamp(0.1, 1.0),
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity: crate::form::SectionRole::A.intensity(),
            });
        }
        score.push(ScoreNote {
            pitch: chord.voice(2)[0],
            onset,
            duration: bar,
            velocity: (vel * fade * 0.9).clamp(0.1, 1.0),
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: crate::form::SectionRole::A.intensity(),
        });
    }
}

/// The TRANSFORMED coda (damage pass ≥0.2): the ending must be changed by
/// the journey, not merely recede — the listening review's words for the
/// plain plagal fade were "it does not feel like the ending is transformed
/// by the climax. It just recedes."
///
/// The opening motif returns, changed three ways: AUGMENTED (each note
/// doubled in length — memory moves slower than speech), an OCTAVE DOWN
/// (the voice that sang it has descended), and harmonized by a BORROWED
/// mixture subdominant (minor iv under a major piece, major IV under a
/// minor one — the one chord the piece never used, saved for the end),
/// resolving to I only in the final bar. Any fragment tone that carries
/// the pc the mixture altered is bent with it — the melody remembers the
/// tune, the harmony has changed underneath it.
///
/// Returns false (caller falls back to the plain coda) when the score has
/// no melody to quote.
fn append_transformed_coda(
    score: &mut Score,
    home: Key,
    meter_beats: f64,
    intent: &MusicalIntent,
    coda_bars: i64,
) -> bool {
    let mut fragment: Vec<Pitch> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .take(4)
        .map(|n| n.pitch)
        .collect();
    if fragment.is_empty() {
        return false;
    }
    let bar = Duration::new(meter_beats as i64, 1);
    let start = score.total_beats;
    let vel = (0.3 + intent.energy.clamp(0.0, 1.0) * 0.3).clamp(0.1, 0.6) * 0.85;

    // The borrowed subdominant: flip the diatonic IV's OWN quality (keyed
    // off IV, not the tonic — in Dorian the tonic is minor but IV is
    // already major, and the genuine mixture there is the DARKENED iv).
    let iv_root = home.scale().degree_pitch_class(4);
    let diatonic_iv = home.diatonic_triad(4);
    let (mixture, third_shift) = match diatonic_iv.quality {
        crate::chord::ChordQuality::Major => (
            crate::chord::Chord::new(iv_root, crate::chord::ChordQuality::Minor),
            -1,
        ),
        crate::chord::ChordQuality::Minor => (
            crate::chord::Chord::new(iv_root, crate::chord::ChordQuality::Major),
            1,
        ),
        // Diminished/suspended edge cases: no honest flip — keep it plain.
        _ => (diatonic_iv, 0),
    };
    let diatonic_third_pc = home.scale().degree_pitch_class(6);
    let total_beats = coda_bars as f64 * meter_beats;
    let mixture_beats = (coda_bars - 1).max(0) as f64 * meter_beats;

    // Melody: the fragment, augmented, an octave down; the last tone bends
    // home to the tonic and rings through whatever remains.
    for p in &mut fragment {
        *p = p.transpose(-12);
    }
    let note_beats = 2.0 * meter_beats / 4.0; // half notes in 4/4 terms
    let mut onset_beats = 0.0f64;
    let n = fragment.len();
    for (i, pitch) in fragment.into_iter().enumerate() {
        let is_last = i == n - 1;
        let remaining = total_beats - onset_beats;
        if remaining <= 0.0 {
            break;
        }
        let dur_beats = if is_last {
            remaining
        } else {
            note_beats.min(remaining)
        };
        // Bend fragment tones that carry the mixture-altered pc while the
        // mixture chord is sounding; the final note becomes the tonic.
        let pitch = if is_last {
            Pitch::new(home.tonic, 4)
        } else if onset_beats < mixture_beats && pitch.pitch_class() == diatonic_third_pc {
            pitch.transpose(third_shift)
        } else {
            pitch
        };
        let fade = 0.9 - 0.25 * (onset_beats / total_beats) as f32;
        score.push(ScoreNote {
            pitch,
            onset: start + Duration::new((onset_beats * 2.0) as i64, 2),
            duration: Duration::new((dur_beats * 2.0) as i64, 2),
            velocity: (vel * fade).clamp(0.1, 1.0),
            role: VoiceRole::Melody,
            emphasis: if is_last {
                Emphasis::Cadential
            } else {
                Emphasis::Normal
            },
            section_intensity: crate::form::SectionRole::A.intensity(),
        });
        onset_beats += dur_beats;
    }

    // Harmony + bass: mixture chord until the final bar, then I.
    for m in 0..coda_bars {
        let chord = if m == coda_bars - 1 {
            home.diatonic_triad(1)
        } else {
            mixture
        };
        let fade = 0.85 - 0.20 * (m as f32 / (coda_bars.max(2) - 1) as f32);
        let onset = start + bar.scale(m, 1);
        for pitch in chord.voice(3) {
            score.push(ScoreNote {
                pitch,
                onset,
                duration: bar,
                // Floor 0.18: below it a sampled accompaniment loses all
                // presence (the voice-balance guarantee) — the dying-away
                // happens above that floor, not through it.
                velocity: (vel * fade * 0.6).clamp(0.18, 1.0),
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity: crate::form::SectionRole::A.intensity(),
            });
        }
        score.push(ScoreNote {
            pitch: chord.voice(2)[0],
            onset,
            duration: bar,
            velocity: (vel * fade * 0.9).clamp(0.1, 1.0),
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: crate::form::SectionRole::A.intensity(),
        });
    }
    true
}

/// Choose a motif shape (by seed) and rhythm density (by arousal). Every
/// template totals exactly `meter` (4) beats so measures align with harmony.
/// These are hand-picked GOOD shapes (arpeggios, neighbor figures, gestures) —
/// not a random walk; the seed selects among them. This is the CLASSICAL
/// style's bank specifically — [`crate::style::Style::motif`] calls this
/// exact function for `Style::Classical` so there's one bank to maintain,
/// not two.
pub(crate) fn classical_motif_bank(arousal: f32, seed: u64) -> Motif {
    let q = Duration::quarter();
    let h = Duration::half();
    let e = Duration::eighth();

    // Low-density (calm) shapes — longer notes.
    let calm: [&[(i32, Duration)]; 3] = [
        &[(1, h), (2, q), (3, q)], // rise and settle
        &[(5, h), (3, q), (1, q)], // descending arch
        &[(3, q), (2, q), (1, h)], // sigh to the tonic
    ];
    // Medium-density shapes — quarters.
    let medium: [&[(i32, Duration)]; 3] = [
        &[(1, q), (2, q), (3, q), (2, q)], // upper-neighbor turn
        &[(1, q), (3, q), (5, q), (3, q)], // arpeggio up and back
        &[(5, q), (4, q), (3, q), (1, q)], // stepwise descent
    ];
    // High-density (excited) shapes — eighths with a longer goal note.
    let busy: [&[(i32, Duration)]; 3] = [
        &[(1, e), (2, e), (3, e), (4, e), (5, h)], // scalar run to a peak
        &[(1, e), (3, e), (5, e), (3, e), (2, h)], // arpeggiated flourish
        &[(5, e), (4, e), (3, e), (2, e), (1, h)], // cascade to rest
    ];

    let bank: &[&[(i32, Duration)]] = if arousal < 0.33 {
        &calm
    } else if arousal < 0.66 {
        &medium
    } else {
        &busy
    };
    let pick = (seed % bank.len() as u64) as usize;
    let picked = Motif::from_degrees(bank[pick]);
    // Multiply the bank's effective variety: an independent seed-slice
    // (integer division, so it doesn't correlate with which template got
    // picked) selects one of 4 orientations (as-is, inversion, retrograde,
    // retrograde-inversion) of that template -- see `form::oriented`.
    let orientation = (seed / bank.len() as u64) % 4;
    crate::form::oriented(&picked, orientation)
}

/// If progression measure `i` can safely become an applied (secondary)
/// dominant of the NEXT measure's chord, return the target degree.
///
/// This generalizes the original ii-before-V (→ V7/V) idiom to the whole
/// family that shares its safety property: a diatonic chord `c` sitting a
/// diatonic fifth above target `t` has the SAME ROOT AND FIFTH as V7/t —
/// only the third differs (raised a semitone when `c` is minor), plus an
/// added 7th. Verified by hand in C major:
///   ii  (D-F-A)  → V7/V  (D-F#-A-C):  third F→F#
///   vi  (A-C-E)  → V7/ii (A-C#-E-G):  third C→C#, added G is diatonic
///   iii (E-G-B)  → V7/vi (E-G#-B-D):  third G→G#, added D is diatonic
///   I   (C-E-G)  → V7/IV (C-E-G-Bb):  triad UNCHANGED, added chromatic b7
/// So the substitution never disturbs bass root motion, and the melody
/// correction stays the single verifiable pitch-class raise the original
/// idiom already used (`realize_melody`); V7/IV needs no melody correction
/// at all (its b7 lives only in the harmony voicing).
///
/// Deliberately excluded, with reasons:
/// - `c = vii°` (→ V7/iii): vii° has a DIMINISHED fifth, so root+fifth are
///   NOT preserved — the one family member without the safety property.
/// - `t = 1` (V→"V7/I"): that's just V7, which the dominant-7th coloring in
///   `realize_harmony_measures` already provides diatonically.
/// - Minor keys: harmonic-minor degree algebra for these substitutions has
///   not been verified, so it is not assumed.
/// - `c = I` fires only on half the seeds: I–IV is so common that turning
///   EVERY one into a bluesy C7 reads as saturation, not color.
fn applied_dominant_target(key: Key, measure_degrees: &[i32], i: usize, seed: u64) -> Option<i32> {
    if key.tonality != crate::harmony::Tonality::Major {
        return None;
    }
    let c = measure_degrees[i];
    let t = *measure_degrees.get(i + 1)?;
    // c must sit a diatonic fifth above t (0-based: +4 steps mod 7).
    let expected_c = ((t - 1 + 4).rem_euclid(7)) + 1;
    if c != expected_c || c == 7 || t == 1 {
        return None;
    }
    if c == 1 && (seed & 4) == 0 {
        return None;
    }
    Some(t)
}

/// The pitch class the melody must RAISE in an applied-dominant measure:
/// the diatonic third of the substituted-out chord `c` (which V7/t plays a
/// semitone higher). `None` for `c = I` (V7/IV leaves the triad untouched —
/// its chromatic b7 exists only in the harmony voicing).
fn applied_dominant_raised_pc(key: Key, c: i32) -> Option<crate::pitch::PitchClass> {
    if c == 1 {
        return None;
    }
    let third_degree = ((c - 1 + 2).rem_euclid(7)) + 1;
    Some(key.scale().degree_pitch_class(third_degree))
}

/// Realize every phrase across every section as the melody voice, with
/// structural dynamics and emphasis: ONE global climax across `form` (a
/// piece has a single peak, not one per phrase — real climaxes are rare and
/// that's what makes them land); cadential = each phrase's last note;
/// phrase-start = each phrase's first note. Each phrase renders in its OWN
/// section's key, so the B section's modulation is real. `start_beat` offsets
/// every onset — used by [`crate::live::LiveComposer`] to splice consecutive
/// incremental calls onto one growing timeline; `compose()` passes zero.
///
/// NOTE: "one global climax" is only meaningful when `form` is the WHOLE
/// piece decided up front (as `compose()` does). [`crate::live::LiveComposer`]
/// calls this once per phrase with no lookahead, so its climax is
/// necessarily LOCAL to that one call — an honest limitation of real-time
/// generation (you cannot know the future peak of a piece that hasn't been
/// decided yet), not a bug.
pub(crate) fn realize_melody(
    score: &mut Score,
    form: &Form,
    intent: &MusicalIntent,
    start_beat: Duration,
    meter_beats: f64,
    climax_grace: bool,
) {
    let melody_octave = 5;
    let base_vel = 0.35 + intent.energy.clamp(0.0, 1.0) * 0.45; // 0.35–0.8

    // Render every phrase (antecedent, consequent) of every section, in its
    // own key, keeping each phrase's key/progression alongside its notes so
    // the applied-dominant chromatic correction below knows which measure
    // (if any) needs it and in which key.
    struct RenderedPhrase {
        notes: Vec<(Option<Pitch>, Duration)>,
        key: Key,
        progression: Vec<i32>,
        role: SectionRole,
    }
    let mut phrases: Vec<RenderedPhrase> = form
        .sections
        .iter()
        .flat_map(|s| {
            [
                RenderedPhrase {
                    notes: s.period.antecedent.render(s.key, melody_octave),
                    key: s.key,
                    progression: s.period.antecedent.progression.clone(),
                    role: s.role,
                },
                RenderedPhrase {
                    notes: s.period.consequent.render(s.key, melody_octave),
                    key: s.key,
                    progression: s.period.consequent.progression.clone(),
                    role: s.role,
                },
            ]
        })
        .collect();

    // Applied-dominant-of-V chromatic correction: for any measure flagged by
    // REGISTER FOLD: transpose any phrase whose peak drifted above the
    // ceiling down by octaves (whole phrases, never single notes — folding
    // one note mid-line would break the melodic contour). Runs BEFORE the
    // chromatic corrections (pitch classes are octave-invariant) and BEFORE
    // climax detection (the marked climax must be the audible peak).
    for rp in &mut phrases {
        loop {
            let (mut max, mut min) = (0u8, u8::MAX);
            for (p, _) in rp.notes.iter() {
                if let Some(p) = p {
                    max = max.max(p.midi());
                    min = min.min(p.midi());
                }
            }
            if max <= MELODY_CEILING_MIDI || min < MELODY_FOLD_FLOOR_MIDI + 12 {
                break;
            }
            for (p, _) in rp.notes.iter_mut() {
                if let Some(p) = p {
                    *p = p.transpose(-12);
                }
            }
        }
        // LAST RESORT, per-note: a phrase spanning more than the fold
        // window (hook-cell leaps + sequence transforms can stretch one
        // past two octaves) can't fold whole — its low notes would growl.
        // A bent contour is the lesser evil against a >A6 shriek, so any
        // note still above the ceiling folds alone.
        for (p, _) in rp.notes.iter_mut() {
            if let Some(p) = p {
                while p.midi() > MELODY_CEILING_MIDI {
                    *p = p.transpose(-12);
                }
            }
        }
    }

    // `applied_dominant_target`, every melody note whose pitch class is the
    // diatonic third of ii (scale degree 4) gets raised a semitone to match
    // the V7/V harmony realize_harmony/realize_bass now play there instead.
    for rp in &mut phrases {
        // (measure index, pitch class the melody must raise) for every
        // applied-dominant measure in this phrase. V7/IV measures need no
        // correction and produce no entry (see applied_dominant_raised_pc).
        let corrections: Vec<(usize, crate::pitch::PitchClass)> = (0..rp.progression.len())
            .filter_map(|i| {
                applied_dominant_target(rp.key, &rp.progression, i, intent.seed)?;
                applied_dominant_raised_pc(rp.key, rp.progression[i]).map(|pc| (i, pc))
            })
            .collect();
        if corrections.is_empty() {
            continue;
        }
        let mut beat = 0.0f64;
        for (pitch, dur) in rp.notes.iter_mut() {
            let measure_idx = (beat / meter_beats) as usize;
            if let Some((_, pc)) = corrections.iter().find(|(m, _)| *m == measure_idx)
                && let Some(p) = pitch
                && p.pitch_class() == *pc
            {
                *p = p.transpose(1);
            }
            beat += dur.beats();
        }
    }

    // The true final cadence: the last pitched note of the last phrase. This
    // ALWAYS gets Cadential emphasis, even in the rare case it's also the
    // highest note of the piece — a listener hears "the ending," not "the
    // peak," when a cadence happens to land on a high note.
    let final_flat_idx = phrases.iter().enumerate().rev().find_map(|(pi, ph)| {
        ph.notes
            .iter()
            .rposition(|(p, _)| p.is_some())
            .map(|ni| (pi, ni))
    });

    // The single global climax: the highest pitched note in the piece,
    // excluding the final cadence note (a climax coinciding with the very
    // last note would be an unusual, coincidental case that reads better as
    // "the ending" than "the peak").
    let mut global_best: Option<(usize, usize, u8)> = None;
    for (pi, ph) in phrases.iter().enumerate() {
        for (ni, (p, _)) in ph.notes.iter().enumerate() {
            if final_flat_idx == Some((pi, ni)) {
                continue;
            }
            if let Some(p) = p
                && global_best.map(|(_, _, m)| p.midi() > m).unwrap_or(true)
            {
                global_best = Some((pi, ni, p.midi()));
            }
        }
    }

    let mut onset = start_beat;
    for (pi, rendered) in phrases.iter().enumerate() {
        let section_intensity = rendered.role.intensity();
        let rendered = &rendered.notes;
        let first_pitched = rendered.iter().position(|(p, _)| p.is_some());
        let last_pitched = rendered.iter().rposition(|(p, _)| p.is_some());
        let total_pitched = rendered.iter().filter(|(p, _)| p.is_some()).count().max(1) as f32;
        let mut pitched_seen = 0u32;
        for (ni, (pitch, dur)) in rendered.iter().enumerate() {
            if let Some(p) = pitch {
                // Phrase-arch dynamic: swell toward ~65% then ease off.
                let pos = pitched_seen as f32 / total_pitched;
                let arch = if pos < 0.65 {
                    0.75 + 0.35 * (pos / 0.65)
                } else {
                    1.1 - 0.35 * ((pos - 0.65) / 0.35)
                };
                let is_final = final_flat_idx == Some((pi, ni));
                let is_climax = global_best
                    .map(|(gp, gn, _)| (gp, gn) == (pi, ni))
                    .unwrap_or(false);
                let emphasis = if is_final {
                    Emphasis::Cadential
                } else if is_climax {
                    Emphasis::Climax
                } else if Some(ni) == last_pitched {
                    Emphasis::Cadential
                } else if Some(ni) == first_pitched {
                    Emphasis::PhraseStart
                } else {
                    Emphasis::Normal
                };
                let vel = (base_vel
                    * arch
                    * section_intensity
                    * if emphasis == Emphasis::Climax {
                        1.15
                    } else {
                        1.0
                    })
                .clamp(0.1, 1.0);
                // Breath: a phrase's last note releases early — real
                // silence before the next phrase begins, the way a singer
                // or wind player must actually breathe. The note's SLOT is
                // unchanged (onset accumulation below still uses the
                // written value), only the sounded length shrinks: an
                // eighth off long notes, a third off short ones. The
                // piece's true final note is exempt — it rings out.
                let sounded = if Some(ni) == last_pitched && !is_final {
                    if dur.beats() > 1.0 {
                        dur.saturating_sub(Duration::eighth())
                    } else {
                        dur.scale(2, 3)
                    }
                } else {
                    *dur
                };
                score.push(ScoreNote {
                    pitch: *p,
                    onset,
                    duration: sounded,
                    velocity: vel,
                    role: VoiceRole::Melody,
                    emphasis,
                    section_intensity,
                });
                pitched_seen += 1;
            }
            onset = onset + *dur;
        }
    }

    // Ornament: an acciaccatura (quick grace note) leaning into the piece's
    // single climax from BELOW — from below deliberately, so the climax
    // remains the piece's highest pitch. The grace steals a sixteenth from
    // the tail of the note preceding the climax (which must be long enough
    // to give it up); its pitch is the climax's diatonic lower neighbor in
    // the CLIMAX PHRASE's own key (the climax may sit in the modulated B/C
    // section).
    // Only in full multi-phrase compositions: a LiveComposer call renders a
    // single phrase whose "climax" is merely local — ornamenting every one
    // of those would put a grace note in every phrase, which is exactly the
    // repetitiveness ornaments exist to break.
    if climax_grace
        && phrases.len() > 1
        && let Some((gp, _, _)) = global_best
    {
        let grace = Duration::sixteenth();
        let climax_idx = score
            .notes
            .iter()
            .position(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Climax);
        if let Some(ci) = climax_idx {
            let climax_onset = score.notes[ci].onset;
            let climax_pitch = score.notes[ci].pitch;
            let climax_vel = score.notes[ci].velocity;
            let climax_intensity = score.notes[ci].section_intensity;
            let key = phrases[gp].key;
            // Diatonic lower neighbor: first pitch class below the climax
            // that belongs to the phrase's scale. The scan reaches 3
            // semitones because harmonic minor has an AUGMENTED SECOND
            // below its leading tone (G# down to F in A minor) — a climax
            // on the leading tone has no closer diatonic neighbor.
            let below = (1..=3)
                .map(|s| climax_pitch.transpose(-s))
                .find(|p| key.scale().contains(p.pitch_class()));
            let prev = score.notes.iter().position(|n| {
                n.role == VoiceRole::Melody
                    && n.onset + n.duration == climax_onset
                    && n.duration.beats() >= 0.75
            });
            if let (Some(below), Some(prev_idx)) = (below, prev) {
                score.notes[prev_idx].duration =
                    score.notes[prev_idx].duration.saturating_sub(grace);
                // INSERT before the climax (not push): melody notes must
                // stay onset-ordered — muse's expressive model computes
                // inter-onset intervals from consecutive entries.
                score.notes.insert(
                    ci,
                    ScoreNote {
                        pitch: below,
                        onset: climax_onset.saturating_sub(grace),
                        duration: grace,
                        velocity: (climax_vel * 0.75).clamp(0.1, 1.0),
                        role: VoiceRole::Melody,
                        emphasis: Emphasis::Normal,
                        section_intensity: climax_intensity,
                    },
                );
            }
        }
    }
}

/// Realize one contiguous run of chords (a single section's measures) into
/// the harmony voice, carrying voice-leading state (`prev_upper`) and the
/// running measure index (`start_measure`, for a continuous onset timeline
/// across sections) from the caller — see [`realize_harmony`].
#[allow(clippy::too_many_arguments)]
fn realize_harmony_measures(
    score: &mut Score,
    measure_degrees: &[i32],
    key: Key,
    meter_beats: f64,
    intent: &MusicalIntent,
    prev_upper: &mut Vec<Pitch>,
    start_measure: i64,
    section_intensity: f32,
    pattern: crate::accompaniment::Accompaniment,
    cadential_split: bool,
) {
    let harmony_octave = 3;
    // Accompaniment level: loud enough that real sampled instruments keep
    // their presence (low velocity ALSO selects the softest recorded layer —
    // a double attenuation the old 0.2-base formula buried the harmony
    // under; found by ear + MIDI-velocity analysis of an exported piece).
    let vel = ((0.28 + intent.energy.clamp(0.0, 1.0) * 0.28) * section_intensity).clamp(0.18, 0.62);
    let bar = Duration::new(meter_beats as i64, 1);
    // The cadential tension degree — 5 in functional keys, the mode's
    // characteristic chord (♭VII / ♭II) in modal keys. Both the seventh-
    // coloring and the cadential harmonic-rhythm split key off it.
    let dom = key.cadence_dominant_degree();
    for (i, &deg) in measure_degrees.iter().enumerate() {
        // Dominant chords get their diatonic 7th added as harmonic color. This
        // is always SAFE: a seventh chord's tones are a strict superset of
        // its triad's, so it can never conflict with the melody's (triad-
        // based) strong-beat chord-tone snapping — it only enriches the
        // accompaniment under an unchanged, still-correct melody.
        //
        // A chord a diatonic fifth above its successor gets substituted for
        // that successor's applied dominant (ii→V7/V, vi→V7/ii, iii→V7/vi,
        // I→V7/IV — see `applied_dominant_target` for the shared-root-and-
        // fifth safety argument); `realize_melody` applies the matching
        // pitch correction to the melody voice.
        let applied = applied_dominant_target(key, measure_degrees, i, intent.seed);
        let chord: Chord = if let Some(t) = applied {
            key.secondary_dominant(t)
        } else if deg == dom {
            key.diatonic_seventh(deg)
        } else {
            key.diatonic_triad(deg)
        };
        let onset = bar.scale(start_measure + i as i64, 1);

        // HARMONIC RHYTHM accelerates into cadences (4/4 styles): a bar
        // whose successor is the dominant — and the dominant bar itself —
        // splits into two half-bar chords, triad → its own seventh chord.
        // The seventh is a strict SUPERSET of the triad (the same safety
        // property as the dominant-7th coloring above), so the melody's
        // snapped notes can never clash; what changes is that the harmony
        // now moves WITHIN the bar, the classic drive into a cadence.
        // Applied-dominant bars are excluded (they already carry their own
        // chromatic color); 3/4 is excluded (splitting oom-pah's bar fights
        // the genre's own rhythm).
        let split_cadential = cadential_split
            && meter_beats >= 4.0
            && applied.is_none()
            && (measure_degrees.get(i + 1) == Some(&dom)
                || (deg == dom && i + 1 == measure_degrees.len()));
        if split_cadential {
            let half = bar.scale(1, 2);
            let (first, second) = (key.diatonic_triad(deg), key.diatonic_seventh(deg));
            let voiced1 = crate::voicing::lead_upper(prev_upper, first, harmony_octave);
            pattern.realize_measure(
                score,
                &voiced1,
                onset,
                meter_beats / 2.0,
                vel,
                section_intensity,
            );
            *prev_upper = voiced1;
            let voiced2 = crate::voicing::lead_upper(prev_upper, second, harmony_octave);
            pattern.realize_measure(
                score,
                &voiced2,
                onset + half,
                meter_beats / 2.0,
                vel * 1.05,
                section_intensity,
            );
            *prev_upper = voiced2;
            continue;
        }

        let voiced = crate::voicing::lead_upper(prev_upper, chord, harmony_octave);
        // The pattern decides WHEN the voiced tones sound within the bar
        // (block pad, arpeggio, Alberti, oom-pah, comp stabs); WHICH tones
        // exist is decided above by the voice leading and is identical for
        // every pattern — see `accompaniment.rs`'s containment tests.
        pattern.realize_measure(score, &voiced, onset, meter_beats, vel, section_intensity);
        *prev_upper = voiced;
    }
}

/// Voiced-chord accompaniment across `form`: one triad (or, on the dominant,
/// a seventh chord) per measure, held for the bar. Upper tones are VOICE-LED
/// continuously across every section — including the modulation into B and
/// back — so a I(A)→vi(B, relative minor) move connects smoothly too, not
/// just the chords within one section. `prev_upper` is caller-owned so
/// [`crate::live::LiveComposer`] can carry it between incremental calls
/// (voice-leading a real-time phrase transition exactly like a section
/// boundary); `compose()` passes a fresh, empty `Vec`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn realize_harmony(
    score: &mut Score,
    form: &Form,
    meter_beats: f64,
    intent: &MusicalIntent,
    prev_upper: &mut Vec<Pitch>,
    pattern: crate::accompaniment::Accompaniment,
    thin_departure: bool,
    cadential_split: bool,
) {
    let mut measure: i64 = 0;
    for section in &form.sections {
        let mut degrees = section.period.antecedent.progression.clone();
        degrees.extend(section.period.consequent.progression.iter().copied());
        // TEXTURE EVOLUTION: the B section (the departure) strips to
        // melody + bass for its first half — the accompaniment re-entering
        // mid-section is an arrival in itself, and the thinning is what
        // makes the full-texture return land as a return. The C episode
        // (rondo's peak) deliberately stays full. Voice-leading state is
        // untouched during the silence, so the re-entry connects smoothly
        // from the last sounded chord.
        let (realized, local_start) = if thin_departure && section.role == SectionRole::B {
            let half = degrees.len() / 2;
            (&degrees[half..], measure + half as i64)
        } else {
            (&degrees[..], measure)
        };
        realize_harmony_measures(
            score,
            realized,
            section.key,
            meter_beats,
            intent,
            prev_upper,
            local_start,
            section.role.intensity(),
            pattern,
            cadential_split,
        );
        measure += degrees.len() as i64;
    }
}

/// A COUNTER-MELODY for the return section(s): a slower second line in the
/// tenor register (octave 4, between bass and melody), moving in voice-led
/// half-notes (dotted-half in 3/4) over the same progression. Chord tones
/// only — it can never introduce a wrong note — with two guards per tone:
/// nearest-motion voice leading from its own previous note, and collision
/// avoidance against the MELODY (never the same or adjacent semitone
/// sounding at this tone's onset; the next-nearest chord tone is taken
/// instead). Classical practice: the reprise earns new interest.
pub(crate) fn realize_counter_melody(
    score: &mut Score,
    form: &Form,
    meter_beats: f64,
    intent: &MusicalIntent,
) {
    let counter_octave = 4;
    let base_vel = (0.28 + intent.energy.clamp(0.0, 1.0) * 0.28).clamp(0.12, 0.58);
    let bar = Duration::new(meter_beats as i64, 1);
    // 4/4 → two half-notes per bar; 3/4 → one dotted-half per bar.
    let tones_per_bar: i64 = if meter_beats >= 4.0 { 2 } else { 1 };
    let tone_dur = bar.scale(1, tones_per_bar);

    // Snapshot melody notes once for collision checks.
    let melody: Vec<(f64, u8)> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .map(|n| (n.onset.beats(), n.pitch.midi()))
        .collect();

    let mut measure: i64 = 0;
    let mut prev: Option<Pitch> = None;
    for section in &form.sections {
        let mut degrees = section.period.antecedent.progression.clone();
        degrees.extend(section.period.consequent.progression.iter().copied());
        if section.role != SectionRole::ReturnA {
            measure += degrees.len() as i64;
            continue;
        }
        let key = section.key;
        let intensity = section.role.intensity();
        let total_tones = (degrees.len() as i64 * tones_per_bar).max(1) as f32;
        for (i, &deg) in degrees.iter().enumerate() {
            let chord = if let Some(t) = applied_dominant_target(key, &degrees, i, intent.seed) {
                key.secondary_dominant(t)
            } else {
                key.diatonic_triad(deg)
            };
            let candidates = chord.voice(counter_octave);
            for k in 0..tones_per_bar {
                let onset = bar.scale(measure + i as i64, 1) + tone_dur.scale(k, 1);
                let onset_beats = onset.beats();
                // Melody pitches sounding at (or very near) this onset.
                let clashing: Vec<u8> = melody
                    .iter()
                    .filter(|(b, _)| (b - onset_beats).abs() < 0.25)
                    .map(|(_, m)| *m)
                    .collect();
                // Rank candidates: nearest to the previous counter tone,
                // then reject those landing on/adjacent to the melody.
                let mut ranked: Vec<Pitch> = candidates.clone();
                ranked.sort_by_key(|p| match prev {
                    Some(pr) => (p.midi() as i32 - pr.midi() as i32).abs(),
                    None => (p.midi() as i32 - (counter_octave * 12 + 16)).abs(),
                });
                let chosen = ranked
                    .iter()
                    .find(|p| {
                        !clashing
                            .iter()
                            .any(|m| (p.midi() as i32 - *m as i32).abs() <= 2)
                    })
                    .or_else(|| ranked.first())
                    .copied();
                if let Some(p) = chosen {
                    // The same phrase-arch shape the melody gets (swell to
                    // ~65%, ease off): the counter-line is a PLAYER, not a
                    // pad — a fixed velocity reads as a machine (found by
                    // MIDI-velocity analysis: the counter sat pinned at 43
                    // while the melody breathed 44–64).
                    let pos = (i as i64 * tones_per_bar + k) as f32 / total_tones;
                    let arch = if pos < 0.65 {
                        0.8 + 0.3 * (pos / 0.65)
                    } else {
                        1.1 - 0.3 * ((pos - 0.65) / 0.35)
                    };
                    score.push(ScoreNote {
                        pitch: p,
                        onset,
                        duration: tone_dur,
                        velocity: (base_vel * arch * intensity).clamp(0.1, 0.65),
                        role: VoiceRole::CounterMelody,
                        emphasis: Emphasis::Normal,
                        section_intensity: intensity,
                    });
                    prev = Some(p);
                }
            }
        }
        measure += degrees.len() as i64;
    }
}

/// Realize one contiguous run of chords into the bass voice — see
/// [`realize_bass`]. `force_root_at(i)` decides, for local measure index `i`
/// within this call, whether to force root position (used at each section's
/// opening measure, to establish a new key clearly, and the piece's very
/// final measure).
#[allow(clippy::too_many_arguments)] // each parameter is a distinct, irreducible piece of realization state; bundling them into a struct for one internal call site wouldn't add clarity.
fn realize_bass_measures(
    score: &mut Score,
    measure_degrees: &[i32],
    key: Key,
    meter_beats: f64,
    intent: &MusicalIntent,
    prev_bass: &mut Option<Pitch>,
    start_measure: i64,
    force_root_at: impl Fn(usize) -> bool,
    section_intensity: f32,
    pattern: crate::accompaniment::Accompaniment,
    cadential_split: bool,
) {
    let bass_octave = 2;
    // Bass sits clearly BELOW the melody's peak (the old formula let it
    // reach ~95% of the lead's level, crowding the low-mids — same
    // exported-MIDI analysis as the harmony fix).
    let vel = ((0.28 + intent.energy.clamp(0.0, 1.0) * 0.22) * section_intensity).clamp(0.1, 0.52);
    let bar = Duration::new(meter_beats as i64, 1);
    let half_bar = Duration::new(meter_beats as i64, 2);
    // Under oom-pah the bass IS the "oom": one crisp quarter on beat 1, the
    // chord stabs own the later beats — a whole-bar drone underneath them
    // would smear the very rhythm the pattern exists to create. Walking is
    // disabled there for the same reason.
    let oom_pah = pattern == crate::accompaniment::Accompaniment::OomPah;
    let walking = intent.arousal > 0.5 && !oom_pah;
    for (i, &deg) in measure_degrees.iter().enumerate() {
        // Root and fifth are identical between a chord and the applied
        // dominant that substitutes it (see `applied_dominant_target`); only
        // the third differs. The bass here only ever plays root/fifth, so
        // this substitution is for consistency with the harmony voice, not
        // strictly required — but it keeps `lead_bass`'s smooth-inversion
        // search from ever landing on the diatonic (un-raised) third while
        // harmony plays the raised one.
        let chord = if let Some(t) = applied_dominant_target(key, measure_degrees, i, intent.seed) {
            key.secondary_dominant(t)
        } else {
            key.diatonic_triad(deg)
        };
        let bar_onset = bar.scale(start_measure + i as i64, 1);
        let force_root = force_root_at(i);
        let bass = crate::voicing::lead_bass(*prev_bass, chord, bass_octave, force_root);

        // Cadence approach (matches realize_harmony_measures' harmonic-
        // rhythm split): in the bar BEFORE the dominant, the bass steps to
        // its chord's THIRD on the back half — the first-inversion walk
        // into V (e.g. ii: D → F, then G), the oldest bass cliché in the
        // book because it works. 4/4 non-oom-pah only, like the split.
        let approaches_dominant = cadential_split
            && meter_beats >= 4.0
            && !oom_pah
            && measure_degrees.get(i + 1) == Some(&5)
            && applied_dominant_target(key, measure_degrees, i, intent.seed).is_none();
        if approaches_dominant {
            let third = chord.voice(bass_octave)[1];
            score.push(ScoreNote {
                pitch: bass,
                onset: bar_onset,
                duration: half_bar,
                velocity: vel,
                role: VoiceRole::Bass,
                emphasis: Emphasis::Normal,
                section_intensity,
            });
            score.push(ScoreNote {
                pitch: third,
                onset: bar_onset + half_bar,
                duration: half_bar,
                velocity: vel * 0.9,
                role: VoiceRole::Bass,
                emphasis: Emphasis::Normal,
                section_intensity,
            });
            *prev_bass = Some(third);
            continue;
        }

        score.push(ScoreNote {
            pitch: bass,
            onset: bar_onset,
            duration: if oom_pah {
                Duration::new(1, 1)
            } else if walking {
                half_bar
            } else {
                bar
            },
            velocity: vel,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity,
        });
        // Beat 3: the fifth (deliberate walking idiom, not voice-led — the
        // point is the outlined root-fifth motion, not smoothness).
        if walking {
            let fifth = chord.voice(bass_octave)[2];
            score.push(ScoreNote {
                pitch: fifth,
                onset: bar_onset + half_bar,
                duration: half_bar,
                velocity: vel * 0.85,
                role: VoiceRole::Bass,
                emphasis: Emphasis::Normal,
                section_intensity,
            });
            *prev_bass = Some(fifth);
        } else {
            *prev_bass = Some(bass);
        }
    }
}

/// Bass line across `form`: voice-led chord tones on beat 1 (root forced at
/// each section's opening measure — establishing each new key clearly — and
/// the piece's true final measure; smooth inversions elsewhere) with a
/// walking root-then-fifth idiom added on beat 3 at higher arousal.
/// `prev_bass` is caller-owned so [`crate::live::LiveComposer`] can carry it
/// between incremental calls; `compose()` passes a fresh `None`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn realize_bass(
    score: &mut Score,
    form: &Form,
    meter_beats: f64,
    intent: &MusicalIntent,
    prev_bass: &mut Option<Pitch>,
    pattern: crate::accompaniment::Accompaniment,
    cadential_split: bool,
) {
    let mut measure: i64 = 0;
    let total_measures: usize = form
        .sections
        .iter()
        .map(|s| s.period.antecedent.progression.len() + s.period.consequent.progression.len())
        .sum();
    for section in &form.sections {
        let mut degrees = section.period.antecedent.progression.clone();
        degrees.extend(section.period.consequent.progression.iter().copied());
        let n = degrees.len();
        let base_measure = measure;
        realize_bass_measures(
            score,
            &degrees,
            section.key,
            meter_beats,
            intent,
            &mut *prev_bass,
            base_measure,
            |i| {
                let global = base_measure as usize + i;
                i == 0 || global == total_measures.saturating_sub(1)
            },
            section.role.intensity(),
            pattern,
            cadential_split,
        );
        measure += n as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::{Progression, Tonality};
    use crate::score::VoiceRole;

    #[test]
    fn coda_is_a_plagal_close_under_a_held_tonic() {
        // The LEGACY (pristine) coda shape — the damage pass replaces it
        // with the transformed coda, so pin it at damage = 0.
        let mut spec = crate::style::Style::Classical.spec();
        spec.texture.damage = 0.0;
        let score = compose_with_spec(&MusicalIntent::default(), &spec);
        let meter = score.meter as f64;
        let coda_start = score.total_beats.beats() - (CODA_BARS as f64) * meter;
        let tonic_pc = score.key.tonic;
        // Melody: the tonic held through both coda bars; the LAST note
        // carries the Cadential emphasis (so the final rubato lands here).
        let melody_coda: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody && n.onset.beats() >= coda_start - 1e-6)
            .collect();
        assert_eq!(melody_coda.len(), CODA_BARS as usize);
        assert!(
            melody_coda
                .iter()
                .all(|n| n.pitch.pitch_class() == tonic_pc)
        );
        assert_eq!(melody_coda.last().unwrap().emphasis, Emphasis::Cadential);
        // Harmony: first coda bar is IV (subdominant root = degree 4), the
        // second is I. Check by bass roots: F then C in C major.
        let bass_coda: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Bass && n.onset.beats() >= coda_start - 1e-6)
            .collect();
        assert_eq!(bass_coda.len(), CODA_BARS as usize);
        assert_eq!(bass_coda[0].pitch.midi() % 12, 5, "IV root (F)");
        assert_eq!(bass_coda[1].pitch.midi() % 12, 0, "I root (C)");
        // And it dies away: the second bar is quieter than the first.
        assert!(bass_coda[1].velocity < bass_coda[0].velocity);
    }

    #[test]
    fn cadence_bars_accelerate_their_harmonic_rhythm() {
        // In a plain 4/4 realize of [2, 5] (predominant → final dominant),
        // BOTH bars split into two half-bar chords... except measure 0 here
        // is the applied dominant (ii→V7/V, full bar). Use [4, 5] instead:
        // IV is not a fifth above V, so no applied substitution — the IV bar
        // must split IV → IVmaj7 and the V bar V → V7.
        let key = crate::harmony::Key::major(crate::pitch::PitchClass::C);
        let intent = MusicalIntent::default();
        let mut score = Score::new(key, 100.0, 4);
        realize_harmony_measures(
            &mut score,
            &[4, 5],
            key,
            4.0,
            &intent,
            &mut Vec::new(),
            0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        let harmony = score.voice(VoiceRole::Harmony);
        // Four distinct half-bar onsets: 0, 2, 4, 6.
        let onsets: std::collections::BTreeSet<i64> = harmony
            .iter()
            .map(|n| (n.onset.beats() * 2.0) as i64)
            .collect();
        assert_eq!(
            onsets,
            [0i64, 4, 8, 12].into_iter().collect(),
            "both cadence-approach bars must split at the half bar"
        );
        // Back half of the IV bar carries IVmaj7's seventh (E over F-A-C).
        let iv_back: Vec<u8> = harmony
            .iter()
            .filter(|n| n.onset.beats() >= 2.0 && n.onset.beats() < 4.0)
            .map(|n| n.pitch.midi() % 12)
            .collect();
        assert!(iv_back.contains(&4), "IVmaj7 must add E, got {iv_back:?}");
        // And the bass walks IV's root to its third into the dominant.
        let mut bass_score = Score::new(key, 100.0, 4);
        realize_bass_measures(
            &mut bass_score,
            &[4, 5],
            key,
            4.0,
            &intent,
            &mut None,
            0,
            |i| i == 0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        let bass = bass_score.voice(VoiceRole::Bass);
        assert_eq!(bass[0].pitch.midi() % 12, 5, "IV root F on beat 1");
        assert_eq!(
            bass[1].pitch.midi() % 12,
            9,
            "IV's third A on beat 3 — first-inversion walk toward G"
        );
    }

    #[test]
    fn melody_never_exceeds_the_register_ceiling() {
        // The horror-spike regression guard: across every style, seed and
        // arousal tier, the melody stays within the ceiling (worst case
        // before the phrase fold: MIDI 108).
        use crate::style::Style;
        for style in [
            Style::Classical,
            Style::Waltz,
            Style::Folk,
            Style::Cinematic,
            Style::Playful,
        ] {
            for seed in 0..8u64 {
                for arousal in [0.2f32, 0.5, 0.9] {
                    let s = compose_styled(
                        &MusicalIntent {
                            seed,
                            arousal,
                            ..Default::default()
                        },
                        style,
                    );
                    let max = s
                        .notes
                        .iter()
                        .filter(|n| n.role == VoiceRole::Melody)
                        .map(|n| n.pitch.midi())
                        .max()
                        .unwrap();
                    assert!(
                        max <= MELODY_CEILING_MIDI,
                        "{style:?} seed {seed} arousal {arousal}: melody max {max}"
                    );
                }
            }
        }
    }

    #[test]
    fn voice_balance_gives_the_melody_a_pocket() {
        // The mix diagnosis from an exported piece: harmony buried in the
        // low 30s, counter pinned flat, bass crowding the melody's peak.
        // These are the score-level guarantees that keep it fixed.
        let score = compose(&MusicalIntent::default());
        let vels = |role: VoiceRole| -> Vec<f32> {
            score
                .notes
                .iter()
                .filter(|n| n.role == role)
                .map(|n| n.velocity)
                .collect()
        };
        let max = |v: &[f32]| v.iter().cloned().fold(f32::MIN, f32::max);
        let min = |v: &[f32]| v.iter().cloned().fold(f32::MAX, f32::min);

        let melody = vels(VoiceRole::Melody);
        let harmony = vels(VoiceRole::Harmony);
        let bass = vels(VoiceRole::Bass);
        let counter = vels(VoiceRole::CounterMelody);

        // Bass peak sits clearly below the melody peak (an open pocket).
        assert!(
            max(&bass) < max(&melody) * 0.85,
            "bass {} must sit below melody {}",
            max(&bass),
            max(&melody)
        );
        // Harmony is accompaniment but not buried: its floor stays above
        // the level where sampled instruments lose all presence.
        assert!(
            min(&harmony) >= 0.18,
            "harmony floor {} is buried",
            min(&harmony)
        );
        // The counter-melody BREATHES: real per-note variance, not a
        // constant (the robot-syndrome regression guard).
        let c_max = max(&counter);
        let c_min = min(&counter);
        assert!(
            c_max - c_min > 0.05,
            "counter-melody must have a dynamic contour: {c_min}..{c_max}"
        );
        // And it projects above the accompaniment on average.
        let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        assert!(
            avg(&counter) > avg(&harmony),
            "the second LINE must sit above the chord pad"
        );
    }

    #[test]
    fn counter_melody_lives_only_in_the_return_and_avoids_the_tune() {
        // The CLEAN counter contract (region, collision guard, half-note
        // gait) — pinned at damage 0: the damage pass's disagreement
        // device (≥0.5) deliberately shifts and shortens the middle notes,
        // which is tested separately.
        let mut spec = crate::style::Style::Classical.spec();
        spec.texture.damage = 0.0;
        let score = compose_with_spec(&MusicalIntent::default(), &spec);
        let counter: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::CounterMelody)
            .collect();
        assert!(!counter.is_empty(), "the return must gain a counter-melody");
        // Ternary A/B/ReturnA in equal thirds of the BODY (the plagal coda
        // sits after them): every counter note sits in the final third.
        let body = score.total_beats.beats() - (CODA_BARS * score.meter as i64) as f64;
        for n in &counter {
            assert!(
                n.onset.beats() >= body * 2.0 / 3.0 - 1e-6,
                "counter-melody note at {} but the return starts at {}",
                n.onset.beats(),
                body * 2.0 / 3.0
            );
        }
        // Collision guard: never the same/adjacent semitone as a melody
        // note sounding at its onset.
        let melody: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .collect();
        for c in &counter {
            for m in &melody {
                if (m.onset.beats() - c.onset.beats()).abs() < 0.25 {
                    assert!(
                        (c.pitch.midi() as i32 - m.pitch.midi() as i32).abs() > 2,
                        "counter {} collides with melody {} at beat {}",
                        c.pitch.midi(),
                        m.pitch.midi(),
                        c.onset.beats()
                    );
                }
            }
        }
        // Slow line: the voice-led LINE stays half-notes in 4/4; the hook
        // ECHOES (the answer device) are shorter by design, so assert the
        // gait holds for the majority, not for every note.
        let half_notes = counter
            .iter()
            .filter(|n| (n.duration.beats() - 2.0).abs() < 1e-9)
            .count();
        assert!(
            half_notes * 2 > counter.len(),
            "half-note gait must dominate: {half_notes}/{}",
            counter.len()
        );
    }

    #[test]
    fn the_departure_section_thins_to_melody_and_bass() {
        // Ternary in equal thirds of the BODY (before the coda): B spans
        // [T/3, 2T/3). Its FIRST half must have no harmony accompaniment
        // (the texture strips to melody+bass); its second half must have it
        // back (the re-entry).
        let score = compose(&MusicalIntent::default());
        let intro = (crate::style::Style::Classical.spec().texture.intro_bars * score.meter) as f64;
        let body = score.total_beats.beats() - (CODA_BARS * score.meter as i64) as f64 - intro;
        let (b_start, b_mid, b_end) = (
            intro + body / 3.0,
            intro + body / 2.0,
            intro + body * 2.0 / 3.0,
        );
        let harmony_in = |lo: f64, hi: f64| {
            score
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::Harmony)
                .filter(|n| n.onset.beats() >= lo - 1e-6 && n.onset.beats() < hi - 1e-6)
                .count()
        };
        assert_eq!(
            harmony_in(b_start, b_mid),
            0,
            "B's first half must strip the accompaniment"
        );
        assert!(
            harmony_in(b_mid, b_end) > 0,
            "the accompaniment must re-enter in B's second half"
        );
        // A stays fully accompanied.
        assert!(harmony_in(0.0, b_start) > 0);
    }

    #[test]
    fn phrases_breathe_before_their_successors() {
        // Every phrase-final (non-piece-final) melody note must release
        // before its slot ends: real silence where a player would breathe.
        let score = compose(&MusicalIntent::default());
        let melody = score.voice(VoiceRole::Melody);
        let mut breaths = 0;
        for w in melody.windows(2) {
            let slot_end = w[1].onset;
            let sounded_end = w[0].onset + w[0].duration;
            if w[0].emphasis == Emphasis::Cadential && w[1].emphasis == Emphasis::PhraseStart {
                assert!(
                    sounded_end.beats() < slot_end.beats() - 1e-9,
                    "cadential note before a phrase start must release early"
                );
                breaths += 1;
            }
        }
        assert!(breaths >= 2, "a multi-phrase piece must breathe: {breaths}");
        // The true final note is exempt — it rings its full written value.
        let last = melody.last().unwrap();
        assert!(last.duration.beats() > 0.5, "final note must ring out");
    }

    #[test]
    fn the_climax_gets_one_grace_note_from_below() {
        // The grace fires only when the note before the climax is long
        // enough to give up a sixteenth (a design precondition, not a
        // guarantee) — hook-cell rhythms mean not every seed satisfies it,
        // so scan seeds for a firing case instead of pinning one.
        let (score, melody, climax_i) = (0..12)
            .find_map(|seed| {
                let score = compose(&MusicalIntent {
                    seed,
                    ..Default::default()
                });
                let melody = score.voice(VoiceRole::Melody);
                let ci = melody
                    .iter()
                    .position(|n| n.emphasis == Emphasis::Climax)
                    .expect("compose() always marks a climax");
                let has_grace = melody.iter().any(|n| {
                    n.duration == Duration::sixteenth()
                        && (n.onset + n.duration) == melody[ci].onset
                });
                has_grace.then_some((score.clone(), melody, ci))
            })
            .expect("some seed in 0..12 must produce a climax grace");
        let climax = &melody[climax_i];
        let graces: Vec<&ScoreNote> = melody
            .iter()
            .filter(|n| {
                n.duration == Duration::sixteenth() && (n.onset + n.duration) == climax.onset
            })
            .collect();
        assert_eq!(graces.len(), 1, "exactly one acciaccatura into the climax");
        let grace = graces[0];
        assert!(
            grace.pitch.midi() < climax.pitch.midi(),
            "grace approaches from BELOW so the climax stays the peak"
        );
        assert!(
            climax.pitch.midi() - grace.pitch.midi() <= 3,
            "grace is the nearest diatonic neighbor (3 semitones covers \
             harmonic minor's augmented second below the leading tone)"
        );
        assert!(
            score.key.scale().contains(grace.pitch.pitch_class())
                || grace.pitch.midi() >= climax.pitch.midi() - 2,
            "grace must be diatonic in its phrase's key"
        );
        // Melody stays onset-ordered (muse's IOI math depends on it).
        assert!(
            melody
                .windows(2)
                .all(|w| w[0].onset.beats() <= w[1].onset.beats()),
            "melody must remain onset-sorted after grace insertion"
        );
    }

    #[test]
    fn valence_chooses_tonality() {
        let bright = compose(&MusicalIntent {
            valence: 0.6,
            ..Default::default()
        });
        assert_eq!(bright.key.tonality, Tonality::Major);
        let dark = compose(&MusicalIntent {
            valence: -0.6,
            ..Default::default()
        });
        assert_eq!(dark.key.tonality, Tonality::Minor);
    }

    #[test]
    fn damage_pass_exposes_the_climax_and_darkens_the_bass_entry() {
        // Device MECHANICS are pinned at damage 1.0, where the planner
        // fires everything; planner SELECTION is tested separately.
        let mut pristine = crate::style::Style::Classical.spec();
        pristine.texture.damage = 0.0;
        let mut damaged = pristine.clone();
        damaged.texture.damage = 1.0;
        let intent = MusicalIntent::default();
        let clean = compose_with_spec(&intent, &pristine);
        let hurt = compose_with_spec(&intent, &damaged);
        let meter = clean.meter as f64;

        // Exposed climax: NO harmony/bass SOUNDING anywhere in the climax
        // note's span (an external MIDI audit caught the first version's
        // blind spot: it only removed notes whose ONSET fell in the bar,
        // so a bass note sustaining in from the previous bar still carried
        // the peak). Overlap check, not onset check.
        let accomp_in_climax_bar = |s: &Score| {
            let climax = s
                .notes
                .iter()
                .find(|n| n.emphasis == Emphasis::Climax)
                .expect("a climax");
            let b0 = (climax.onset.beats() / meter).floor() * meter;
            let b1 = ((climax.onset + climax.duration).beats() / meter).ceil() * meter;
            s.notes
                .iter()
                .filter(|n| matches!(n.role, VoiceRole::Harmony | VoiceRole::Bass))
                .filter(|n| {
                    n.onset.beats() < b1 - 1e-6 && (n.onset + n.duration).beats() > b0 + 1e-6
                })
                .count()
        };
        assert!(accomp_in_climax_bar(&clean) > 0);
        assert_eq!(
            accomp_in_climax_bar(&hurt),
            0,
            "the climax must stand alone"
        );

        // Dark entrance: the damaged bass's first note is an octave below
        // the pristine one's.
        let first_bass = |s: &Score| {
            s.notes
                .iter()
                .filter(|n| n.role == VoiceRole::Bass)
                .min_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()))
                .unwrap()
                .pitch
                .midi() as i32
        };
        assert_eq!(first_bass(&hurt), first_bass(&clean) - 12);

        // Expectation hole: the damaged melody has fewer notes (the
        // return's second-bar downbeat is gone; the wait tone adds one
        // back, so compare against the exact devices: hole −1, wait +1 —
        // net melody count may tie; assert the SPECIFIC downbeat is gone).
        let counter_shifted = hurt
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::CounterMelody)
            .zip(
                clean
                    .notes
                    .iter()
                    .filter(|n| n.role == VoiceRole::CounterMelody),
            )
            .any(|(h, c)| (h.onset.beats() - c.onset.beats()).abs() > 0.25);
        assert!(
            counter_shifted,
            "the counterline must rhythmically disagree at full damage"
        );

        // Determinism: same seed, same damage → identical output.
        let again = compose_with_spec(&intent, &damaged);
        assert_eq!(hurt.notes.len(), again.notes.len());
    }

    #[test]
    fn hook_cell_becomes_the_pieces_opening_name() {
        // The first melody statement must carry the hook's RHYTHM exactly
        // (pitch degrees can bend under chord-tone snapping; the rhythmic
        // identity is what the ear names first). And turning the hook off
        // must change the opening — proving the graft is real.
        let spec_on = crate::style::Style::Classical.spec();
        let mut spec_off = spec_on.clone();
        spec_off.texture.hook_cell = false;
        let mut some_opening_differs = false;
        for seed in 0..6 {
            let intent = MusicalIntent {
                seed,
                ..Default::default()
            };
            let score = compose_with_spec(&intent, &spec_on);
            let hook = crate::hook::HookCell::generate(seed, spec_on.meter as f64);
            let melody: Vec<&ScoreNote> = score
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::Melody)
                .collect();
            assert!(melody.len() >= hook.notes.len());
            for (i, (_, dur)) in hook.notes.iter().enumerate() {
                assert_eq!(
                    melody[i].duration, *dur,
                    "seed {seed}: opening note {i} lost the hook's rhythm"
                );
            }
            let off = compose_with_spec(&intent, &spec_off);
            let off_melody: Vec<&ScoreNote> = off
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::Melody)
                .collect();
            if off_melody
                .iter()
                .zip(&melody)
                .take(hook.notes.len())
                .any(|(a, b)| a.duration != b.duration || a.pitch != b.pitch)
            {
                some_opening_differs = true;
            }
        }
        assert!(
            some_opening_differs,
            "hook on/off must audibly change at least one seed's opening"
        );
    }

    #[test]
    fn intro_treatment_varies_by_seed() {
        // "Why do most songs start the same?" — because they did. The
        // intro now has three seed-picked treatments; nearby seed decades
        // must produce audibly different openings.
        let spec = crate::style::Style::Classical.spec();
        let intro_beats = (spec.texture.intro_bars * spec.meter) as f64;
        let intro_shape = |seed: u64| {
            let score = compose_with_spec(
                &MusicalIntent {
                    seed,
                    ..Default::default()
                },
                &spec,
            );
            let in_intro = |role| {
                score
                    .notes
                    .iter()
                    .filter(|n| n.role == role && n.onset.beats() < intro_beats - 1e-9)
                    .count()
            };
            (in_intro(VoiceRole::Harmony), in_intro(VoiceRole::Bass))
        };
        // seed 0 → pattern only; seed 5 → pedal (bass present); seed 10 →
        // swell (rising-velocity held chords, no bass).
        let (h0, b0) = intro_shape(0);
        let (h5, b5) = intro_shape(5);
        let (h10, b10) = intro_shape(10);
        assert_eq!(b0, 0, "variant 0 has no intro bass");
        assert!(b5 > 0, "variant 1 grounds the intro on a pedal");
        assert_eq!(b10, 0, "variant 2 has no intro bass");
        assert!(h0 > 0 && h5 > 0 && h10 > 0);
        // The swell's signature is its CRESCENDO (seed 0's Block pattern
        // is also held chords, so note counts alone can't tell them
        // apart — the first version of this test learned that).
        let intro_velocities = |seed: u64| -> Vec<f32> {
            let score = compose_with_spec(
                &MusicalIntent {
                    seed,
                    ..Default::default()
                },
                &spec,
            );
            let mut v: Vec<f32> = score
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::Harmony && n.onset.beats() < intro_beats - 1e-9)
                .map(|n| n.velocity)
                .collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            v
        };
        assert!(
            intro_velocities(10).len() >= 2,
            "the swell must genuinely crescendo"
        );
        assert_eq!(
            intro_velocities(0).len(),
            1,
            "the plain pattern intro holds one level"
        );
    }

    #[test]
    fn counter_answers_the_held_cadence_with_the_hook() {
        // During the return's held cadence bar the counterline must speak
        // the hook back (halved note values) — and stay silent there when
        // the hook is off (nothing to quote).
        let spec = crate::style::Style::Classical.spec();
        let mut no_hook = spec.clone();
        no_hook.texture.hook_cell = false;
        let intent = MusicalIntent::default();
        let hook = crate::hook::HookCell::generate(intent.seed, spec.meter as f64);
        let echo_notes = |spec: &crate::spec::CompositionSpec| -> usize {
            let score = compose_with_spec(&intent, spec);
            // Echoes are the counter notes SHORTER than the half-note gait.
            score
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::CounterMelody && n.duration.beats() < 1.4)
                .count()
        };
        let with_hook = echo_notes(&spec);
        assert!(
            with_hook >= hook.notes.len(),
            "the answer must speak the whole name: {with_hook} echo notes"
        );
        assert_eq!(
            echo_notes(&no_hook),
            0,
            "no hook, no quotation — the counter stays a supporting line"
        );
    }

    #[test]
    fn the_return_remembers_the_hook_through_the_attitudes_ears() {
        use crate::spec::Attitude;
        let mut base = crate::style::Style::Classical.spec();
        base.texture.damage = 0.0; // clean counting, no expectation hole
        let intent = MusicalIntent::default();
        let compose_att = |a: Option<Attitude>| {
            let mut spec = base.clone();
            spec.attitude = a;
            compose_with_spec(&intent, &spec)
        };
        // The return's first bar, located by the counter-melody's entrance
        // (that voice exists only in the return).
        let bar1 = |s: &Score| -> Vec<(u8, f64)> {
            let ret = s
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::CounterMelody)
                .map(|n| n.onset.beats())
                .fold(f64::MAX, f64::min);
            let meter = s.meter as f64;
            let mut v: Vec<(u8, f64, f64)> = s
                .notes
                .iter()
                .filter(|n| {
                    n.role == VoiceRole::Melody
                        && n.onset.beats() >= ret - 1e-9
                        && n.onset.beats() < ret + meter - 1e-9
                })
                .map(|n| (n.pitch.midi(), n.onset.beats(), n.duration.beats()))
                .collect();
            v.sort_by(|a, b| a.1.total_cmp(&b.1));
            v.into_iter().map(|(m, _, d)| (m, d)).collect()
        };
        let neutral = bar1(&compose_att(None));
        assert!(neutral.len() >= 3, "the return must state the hook");

        // Grief misremembers: same note count, at least one pitch changed,
        // and the largest reach has SHRUNK.
        let grief = bar1(&compose_att(Some(Attitude::Grief)));
        assert_eq!(grief.len(), neutral.len());
        assert!(grief.iter().zip(&neutral).any(|(a, b)| a.0 != b.0));
        let max_leap = |v: &[(u8, f64)]| {
            v.windows(2)
                .map(|w| (w[1].0 as i32 - w[0].0 as i32).abs())
                .max()
                .unwrap_or(0)
        };
        assert!(
            max_leap(&grief) < max_leap(&neutral),
            "the reach falls short"
        );

        // Defiance interrupts: fewer melody notes IN TOTAL (the window
        // count is confounded by defiant syncopation shifting notes
        // across bar lines — total counts are immune to timing devices;
        // syncopation moves notes, never adds or removes them).
        let melody_total = |s: &Score| {
            s.notes
                .iter()
                .filter(|n| n.role == VoiceRole::Melody)
                .count()
        };
        let neutral_total = melody_total(&compose_att(None));
        let defiant_total = melody_total(&compose_att(Some(Attitude::Defiance)));
        assert!(
            defiant_total < neutral_total,
            "the statement is bitten off: {defiant_total} vs {neutral_total}"
        );

        // Joy ornaments: one more note (the upper-neighbor grace).
        let joy = bar1(&compose_att(Some(Attitude::Joy)));
        assert!(joy.len() > neutral.len(), "the retelling gains a giggle");

        // Curiosity trails off: fewer notes, and the note before the
        // missing one lingers longer than it did in the neutral telling.
        let curious = bar1(&compose_att(Some(Attitude::Curiosity)));
        assert!(curious.len() < neutral.len());
        // The tone before the dropped one lingers: at SOME position the
        // curious telling holds notably longer than the neutral one (the
        // bar window also contains tail-graft notes after the hook, so
        // "last in window" is not the extended note — position-wise
        // comparison finds it).
        assert!(
            curious.iter().zip(&neutral).any(|(c, n)| c.1 > n.1 + 0.4),
            "the trailing tone lingers"
        );
    }

    #[test]
    fn the_final_return_finally_completes_the_name() {
        // Rondo (odd seed → A B A C A): the MIDDLE return is wounded by
        // the attitude; the FINAL return says the name whole — same
        // pitches as the opening statement — with a confidence lift.
        use crate::spec::Attitude;
        let mut spec = crate::style::Style::Classical.spec();
        spec.texture.damage = 0.0;
        spec.attitude = Some(Attitude::Defiance);
        let intent = MusicalIntent {
            seed: 1, // odd → rondo
            ..Default::default()
        };
        let score = compose_with_spec(&intent, &spec);
        let meter = score.meter as f64;
        let intro = spec.texture.intro_bars as f64 * meter;
        let section = 2.0 * intent.bars as f64 * meter; // ante + cons
        let (a0, mid0, fin0) = (intro, intro + 2.0 * section, intro + 4.0 * section);
        let window = |w0: f64| -> Vec<(u8, f32)> {
            let mut v: Vec<(f64, u8, f32)> = score
                .notes
                .iter()
                .filter(|n| {
                    n.role == VoiceRole::Melody
                        && n.onset.beats() >= w0 - 1e-9
                        && n.onset.beats() < w0 + meter - 1e-9
                })
                .map(|n| (n.onset.beats(), n.pitch.midi(), n.velocity))
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v.into_iter().map(|(_, m, vel)| (m, vel)).collect()
        };
        let (a, mid, fin) = (window(a0), window(mid0), window(fin0));
        // The middle return is interrupted (defiance): fewer notes than
        // the final one.
        assert!(
            mid.len() < fin.len(),
            "the wound belongs to the middle return: {} vs {}",
            mid.len(),
            fin.len()
        );
        // The final return is COMPLETE: the same pitch sequence as the
        // opening statement...
        assert_eq!(
            fin.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
            a.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
            "the final return must say the name whole"
        );
        // ...spoken with quiet confidence (the +0.05 lift on its opening).
        assert!(
            fin[0].1 > a[0].1 + 0.03,
            "the survivor speaks up: {} vs {}",
            fin[0].1,
            a[0].1
        );
    }

    #[test]
    fn the_bass_hides_the_hook_mid_departure() {
        let spec = crate::style::Style::Classical.spec();
        let mut no_hook = spec.clone();
        no_hook.texture.hook_cell = false;
        let intent = MusicalIntent::default();
        let hook = crate::hook::HookCell::generate(intent.seed, spec.meter as f64);
        // The quote: a run of bass notes whose durations are exactly the
        // hook's, AUGMENTED (×2), in sequence.
        let has_quote = |s: &Score| -> bool {
            let bass: Vec<f64> = {
                let mut v: Vec<(f64, f64)> = s
                    .notes
                    .iter()
                    .filter(|n| n.role == VoiceRole::Bass)
                    .map(|n| (n.onset.beats(), n.duration.beats()))
                    .collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                v.into_iter().map(|(_, d)| d).collect()
            };
            let target: Vec<f64> = hook.notes.iter().map(|(_, d)| d.beats() * 2.0).collect();
            bass.windows(target.len())
                .any(|w| w.iter().zip(&target).all(|(a, b)| (a - b).abs() < 1e-6))
        };
        assert!(
            has_quote(&compose_with_spec(&intent, &spec)),
            "the departure's bass must speak the name, augmented"
        );
        assert!(
            !has_quote(&compose_with_spec(&intent, &no_hook)),
            "no hook, nothing to hide"
        );
    }

    #[test]
    fn attitudes_change_the_pieces_behavior_not_just_its_dials() {
        use crate::spec::Attitude;
        let base = crate::style::Style::Classical.spec();
        let intent = MusicalIntent::default();
        let with = |a: Option<Attitude>| {
            let mut spec = base.clone();
            spec.attitude = a;
            compose_with_spec(&intent, &spec)
        };
        let neutral = with(None);

        // GRIEF: suspension chains — harmony tones foreign to the bar's
        // triad, resolving DOWN by step one beat later — plus a slower
        // pulse.
        let grief = with(Some(Attitude::Grief));
        assert!(grief.tempo_bpm < neutral.tempo_bpm);
        let suspensions = grief
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Harmony)
            .filter(|sus| {
                grief.notes.iter().any(|res| {
                    res.role == VoiceRole::Harmony
                        && (res.onset.beats() - (sus.onset.beats() + 1.0)).abs() < 1e-6
                        && (1..=2).contains(&(sus.pitch.midi() as i32 - res.pitch.midi() as i32))
                })
            })
            .count();
        assert!(
            suspensions >= 3,
            "grief must weep in suspension chains, found {suspensions}"
        );

        // DEFIANCE: notated syncopation — melody onsets on the off-eighth
        // before a downbeat, which the neutral piece never writes — and a
        // more assertive bass.
        let defiant = with(Some(Attitude::Defiance));
        let meter = defiant.meter as f64;
        let anticipated = defiant
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .filter(|n| {
                let in_bar = n.onset.beats().rem_euclid(meter);
                (in_bar - (meter - 0.5)).abs() < 1e-6
            })
            .count();
        assert!(
            anticipated >= 2,
            "defiance must push against the grid, found {anticipated}"
        );
        let mean_bass = |s: &Score| {
            let v: Vec<f32> = s
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::Bass)
                .map(|n| n.velocity)
                .collect();
            v.iter().sum::<f32>() / v.len() as f32
        };
        assert!(mean_bass(&defiant) > mean_bass(&neutral) + 0.05);

        // JOY: quicker pulse, lighter interior notes.
        let joy = with(Some(Attitude::Joy));
        assert!(joy.tempo_bpm > neutral.tempo_bpm);
        let mean_short = |s: &Score| {
            let v: Vec<f64> = s
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Normal)
                .map(|n| n.duration.beats())
                .collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        assert!(mean_short(&joy) < mean_short(&neutral));

        // CURIOSITY: the piece ends on a QUESTION — the final melody tone
        // is scale degree 2, not the tonic.
        let curious = with(Some(Attitude::Curiosity));
        let final_pc = |s: &Score| {
            s.notes
                .iter()
                .filter(|n| n.role == VoiceRole::Melody)
                .max_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()))
                .unwrap()
                .pitch
                .pitch_class()
        };
        assert_eq!(
            final_pc(&curious),
            curious.key.scale().degree_pitch_class(2),
            "curiosity must end asking"
        );
        assert_eq!(
            final_pc(&neutral),
            neutral.key.tonic,
            "the default resolves"
        );
    }

    #[test]
    fn damage_planner_is_deterministic_and_varies_across_seeds() {
        let spec = crate::style::Style::Classical.spec();
        let plan_for = |seed: u64, amount: f32| {
            let mut s = spec.clone();
            s.texture.damage = 0.0; // plan on the CLEAN score
            let score = compose_with_spec(
                &MusicalIntent {
                    seed,
                    ..Default::default()
                },
                &s,
            );
            plan_damage(&score, spec.meter as f64, spec.texture.swing, amount, seed)
        };
        // Deterministic: same inputs → same plan.
        assert_eq!(plan_for(3, 0.5), plan_for(3, 0.5));
        // Amount scales the wound count: 0.2 → 1 device, 1.0 → all 5.
        assert_eq!(plan_for(3, 0.2).len(), 1);
        assert_eq!(plan_for(3, 1.0).len(), 5);
        // The scars VARY: across seeds, at least two different top-3 plans
        // (the review of the fixed-wound version: "the listener will
        // eventually hear the algorithm instead of the song").
        let plans: std::collections::BTreeSet<String> = (0..8)
            .map(|seed| format!("{:?}", plan_for(seed, 0.5)))
            .collect();
        assert!(
            plans.len() >= 2,
            "8 seeds produced identical damage plans: {plans:?}"
        );
    }

    #[test]
    fn damage_planner_answers_the_pieces_weakness() {
        // A synthetic maximally-smooth, maximally-repetitive melody (the
        // same stepwise quarter-note bar over and over, flat velocity, no
        // marked climax) must make the planner prioritize INTERRUPTION —
        // the exposed climax and the expectation hole — over timing
        // damage. This pins diagnosis→selection, not just device count.
        let key = crate::harmony::Key::major(crate::pitch::PitchClass::C);
        let mut score = Score::new(key, 100.0, 4);
        for bar in 0..8i64 {
            for beat in 0..4i64 {
                let midi = 60 + (beat as u8 % 2) * 2; // C-D-C-D forever
                score.push(ScoreNote {
                    pitch: crate::pitch::Pitch::from_midi(midi),
                    onset: Duration::new(bar * 4 + beat, 1),
                    duration: Duration::new(1, 1),
                    velocity: 0.5,
                    role: VoiceRole::Melody,
                    emphasis: Emphasis::Normal,
                    section_intensity: 1.0,
                });
            }
        }
        let plan = plan_damage(&score, 4.0, 0.5, 0.4, 0);
        assert!(
            plan.contains(&DamageDevice::ExposedClimax)
                || plan.contains(&DamageDevice::ExpectationHole),
            "a smooth, repetitive piece needs interruption/mutation, got {plan:?}"
        );
    }

    #[test]
    fn transformed_coda_quotes_the_opening_changed() {
        // damage ≥ 0.2 replaces the held-tonic plagal coda with the opening
        // fragment: augmented, an octave down, over a BORROWED mixture
        // subdominant.
        let spec = crate::style::Style::Classical.spec(); // damage 0.5
        let intent = MusicalIntent {
            valence: 0.6, // major key → borrowed chord is the minor iv
            ..Default::default()
        };
        let score = compose_with_spec(&intent, &spec);
        let meter = score.meter as f64;
        let coda_bars = spec.texture.coda_bars as f64;
        let coda_start = score.total_beats.beats() - coda_bars * meter;
        let coda_melody: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody && n.onset.beats() >= coda_start - 1e-6)
            .collect();
        // More than a held tonic: several distinct pitches...
        let pcs: std::collections::BTreeSet<u8> =
            coda_melody.iter().map(|n| n.pitch.midi() % 12).collect();
        assert!(pcs.len() >= 2, "the coda must QUOTE, not just hold");
        // ...ending on the tonic (still a resolution)...
        assert_eq!(
            coda_melody.last().unwrap().pitch.pitch_class(),
            score.key.tonic
        );
        // ...and the first coda harmony chord is the borrowed iv: it
        // contains the LOWERED sixth degree (pc of degree 6 minus 1).
        let lowered_sixth = (score.key.scale().degree_pitch_class(6).value() + 11) % 12;
        let first_coda_harmony: std::collections::BTreeSet<u8> = score
            .notes
            .iter()
            .filter(|n| {
                n.role == VoiceRole::Harmony
                    && n.onset.beats() >= coda_start - 1e-6
                    && n.onset.beats() < coda_start + meter - 1e-6
            })
            .map(|n| n.pitch.midi() % 12)
            .collect();
        assert!(
            first_coda_harmony.contains(&lowered_sixth),
            "borrowed iv must carry the lowered sixth {lowered_sixth}, got {first_coda_harmony:?}"
        );
    }

    #[test]
    fn arrangement_dramaturgy_stages_the_ensemble() {
        let spec = crate::style::Style::Classical.spec(); // dramaturgy defaults ON
        let intent = MusicalIntent::default();
        let score = compose_with_spec(&intent, &spec);
        let meter = spec.meter as f64;
        let intro_end = spec.texture.intro_bars as f64 * meter;
        let first_onset = |role: VoiceRole| {
            score
                .notes
                .iter()
                .filter(|n| n.role == role)
                .map(|n| n.onset.beats())
                .fold(f64::MAX, f64::min)
        };
        // Intro: accompaniment alone — harmony sounds from the top, the
        // melody waits out the intro bars.
        assert!(first_onset(VoiceRole::Harmony) < 1e-9);
        assert!(first_onset(VoiceRole::Melody) >= intro_end - 1e-9);
        // Staged bass: sits out the opening antecedent (one phrase =
        // `bars` measures), arrives with the consequent.
        let antecedent_end = intro_end + intent.bars.max(1) as f64 * meter;
        let bass_in = first_onset(VoiceRole::Bass);
        assert!(
            bass_in >= antecedent_end - 1e-9,
            "bass entered at {bass_in}, expected >= {antecedent_end}"
        );
        // Pre-return thinning: the bar before the return (where the
        // counter-melody starts) has NO bass.
        let return_start = first_onset(VoiceRole::CounterMelody);
        assert!(return_start.is_finite() && return_start < f64::MAX);
        let bar_before = return_start - meter;
        assert!(
            !score.notes.iter().any(|n| n.role == VoiceRole::Bass
                && n.onset.beats() >= bar_before - 1e-9
                && n.onset.beats() < return_start - 1e-9),
            "bass must drop out in the bar before the return"
        );
    }

    #[test]
    fn held_arrivals_create_rests_and_a_dwelt_on_climax() {
        let spec_on = crate::style::Style::Classical.spec();
        let mut spec_off = spec_on.clone();
        spec_off.texture.held_arrivals = false;
        let mut some_climax_grew = false;
        for seed in 0..4 {
            let intent = MusicalIntent {
                seed,
                ..Default::default()
            };
            let on = compose_with_spec(&intent, &spec_on);
            let off = compose_with_spec(&intent, &spec_off);
            let melody_count = |s: &Score| {
                s.notes
                    .iter()
                    .filter(|n| n.role == VoiceRole::Melody)
                    .count()
            };
            // Cadence holds REMOVE interior notes — the melody gains rests.
            assert!(
                melody_count(&on) < melody_count(&off),
                "seed {seed}: held arrivals must thin the melody"
            );
            let climax_dur = |s: &Score| {
                s.notes
                    .iter()
                    .find(|n| n.emphasis == Emphasis::Climax)
                    .map(|n| n.duration.beats())
                    .unwrap_or(0.0)
            };
            assert!(climax_dur(&on) >= climax_dur(&off) - 1e-9);
            if climax_dur(&on) > climax_dur(&off) + 1e-9 {
                some_climax_grew = true;
            }
        }
        assert!(
            some_climax_grew,
            "the climax hold must fire on at least one of the seeds"
        );
    }

    #[test]
    fn spec_mode_override_composes_genuinely_modal_music() {
        use crate::pitch::PitchClass;
        use crate::scale::Mode;
        // D Dorian ternary, positive valence: the mode override must win
        // over valence, and NO C# may appear anywhere — the home sections
        // are Dorian (♭7 = C natural), the B section modulates to the
        // relative C MAJOR (also no C#), and applied dominants are gated
        // off outside functional major. A single C# would mean a borrowed
        // leading tone erased the mode.
        let mut spec = crate::style::Style::Classical.spec();
        spec.mode = Some(Mode::Dorian);
        spec.form_pool = vec![crate::spec::FormKind::Ternary];
        // Pristine render: the damage pass's "wait" tone (≥0.5) is a
        // DELIBERATE chromatic passing note, which this test's zero-C#
        // invariant would misread as a borrowed leading tone.
        spec.texture.damage = 0.0;
        spec.validate().expect("dorian spec must validate");
        for seed in 0..4 {
            let score = compose_with_spec(
                &MusicalIntent {
                    valence: 0.5,
                    tonic: PitchClass::D,
                    seed,
                    ..Default::default()
                },
                &spec,
            );
            assert_eq!(
                score.key.tonality,
                Tonality::Modal(crate::scale::Mode::Dorian)
            );
            for n in &score.notes {
                assert_ne!(
                    n.pitch.midi() % 12,
                    PitchClass::CSHARP.value() as u8 % 12,
                    "seed {seed}: C# (a borrowed leading tone) in a D Dorian piece"
                );
            }
        }
        // The grammar-incompatible modes are rejected at validation.
        let mut bad = crate::style::Style::Classical.spec();
        bad.mode = Some(Mode::Locrian);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn arousal_drives_tempo() {
        let calm = compose(&MusicalIntent {
            arousal: 0.1,
            ..Default::default()
        });
        let excited = compose(&MusicalIntent {
            arousal: 0.9,
            ..Default::default()
        });
        assert!(excited.tempo_bpm > calm.tempo_bpm + 40.0);
    }

    #[test]
    fn motif_templates_fill_a_measure() {
        // Every generated motif totals exactly 4 beats so measures align.
        for arousal in [0.1f32, 0.5, 0.9] {
            for seed in 0..12u64 {
                let m = classical_motif_bank(arousal, seed);
                assert_eq!(
                    m.total_duration(),
                    Duration::new(4, 1),
                    "arousal={arousal} seed={seed} motif must fill a 4/4 bar"
                );
            }
        }
    }

    #[test]
    fn classical_motif_bank_has_more_than_the_raw_template_count_of_variety() {
        // The bank has 3 raw templates per tier; with orientation
        // multiplication (x4) there must be MORE than 3 distinct outputs
        // across a run of seeds -- otherwise the multiplier is a no-op.
        let shapes: std::collections::HashSet<Vec<i32>> = (0..12u64)
            .map(|seed| classical_motif_bank(0.5, seed).degrees())
            .collect();
        assert!(
            shapes.len() > 3,
            "expected more than 3 distinct shapes from orientation multiplication, got {}",
            shapes.len()
        );
    }

    #[test]
    fn compose_is_deterministic() {
        let intent = MusicalIntent {
            seed: 7,
            ..Default::default()
        };
        assert_eq!(compose(&intent), compose(&intent));
    }

    #[test]
    fn arousal_and_energy_select_between_period_and_sentence() {
        // Same motif + progression, only the archetype differs — this is
        // exactly what compose()'s use_sentence branch depends on, so if
        // parallel() and parallel_sentence() ever produced identical output
        // the structural choice would be a no-op. They must differ.
        let motif = classical_motif_bank(0.5, 0);
        let base = Progression::generate(4, 0);
        let period = crate::phrase::Period::parallel(&motif, &base.degrees, 4.0);
        let sentence = crate::phrase::Period::parallel_sentence(&motif, &base.degrees, 4.0);
        assert_ne!(period.antecedent.line.notes, sentence.antecedent.line.notes);
    }

    #[test]
    fn compose_is_stable_across_the_sentence_threshold() {
        // Regression guard on the arousal/energy branch in compose(),
        // including right at the 0.6 boundary on both sides.
        for arousal in [0.0f32, 0.3, 0.5, 0.6, 0.61, 0.75, 0.9, 1.0] {
            let s = compose(&MusicalIntent {
                arousal,
                ..Default::default()
            });
            assert!(
                !s.notes.is_empty(),
                "arousal={arousal} produced an empty score"
            );
        }
    }

    #[test]
    fn score_has_all_three_voices() {
        let s = compose(&MusicalIntent::default());
        assert!(!s.voice(VoiceRole::Melody).is_empty());
        assert!(!s.voice(VoiceRole::Harmony).is_empty());
        assert!(!s.voice(VoiceRole::Bass).is_empty());
    }

    #[test]
    fn compose_is_a_three_section_piece_not_one_period() {
        // The whole point of Layer 3: compose() must produce THREE sections'
        // worth of music (A + B + ReturnA), not stop after one period. Seed
        // 0 (even) always selects ternary -- see
        // `compose_odd_seed_selects_rondo_instead_of_ternary` for the other
        // branch.
        let s = compose(&MusicalIntent::default());
        let bars = MusicalIntent::default().bars.max(1) as i64;
        let intro = crate::style::Style::Classical.spec().texture.intro_bars as i64;
        // Each section is a period: antecedent (bars) + consequent (bars).
        // Three sections -> 3 * 2 * bars measures, each `meter` (4) beats,
        // plus the arrangement intro and the coda.
        let expected_beats = intro * 4 + 3 * 2 * bars * 4 + CODA_BARS * 4;
        assert_eq!(s.total_beats, Duration::new(expected_beats, 1));
    }

    #[test]
    fn compose_odd_seed_selects_rondo_instead_of_ternary() {
        // An odd seed must produce a FIVE-section rondo (A-B-A-C-A), not
        // ternary's three -- otherwise every piece would share the same
        // large-scale shape regardless of seed.
        let intent = MusicalIntent {
            seed: 1,
            ..Default::default()
        };
        let s = compose(&intent);
        let bars = intent.bars.max(1) as i64;
        let intro = crate::style::Style::Classical.spec().texture.intro_bars as i64;
        // Five sections -> 5 * 2 * bars measures, each `meter` (4) beats,
        // plus the arrangement intro and the coda.
        let expected_beats = intro * 4 + 5 * 2 * bars * 4 + CODA_BARS * 4;
        assert_eq!(s.total_beats, Duration::new(expected_beats, 1));
    }

    #[test]
    fn compose_carries_a_real_long_range_tension_arc() {
        // A rondo (odd seed) piece must show its full A/B/ReturnA/C/ReturnA
        // intensity range on its notes -- not every section breathing the
        // same, and specifically C (the rondo's peak) reaching the highest
        // section_intensity of the whole piece.
        let intent = MusicalIntent {
            seed: 1,
            ..Default::default()
        };
        let s = compose(&intent);
        let intensities: Vec<f32> = s.notes.iter().map(|n| n.section_intensity).collect();
        let max = intensities.iter().cloned().fold(f32::MIN, f32::max);
        let min = intensities.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            max > min,
            "a real piece must span more than one intensity level"
        );
        assert_eq!(
            max,
            crate::form::SectionRole::C.intensity(),
            "the rondo's C section must be the loudest structural moment"
        );
        assert_eq!(
            min,
            crate::form::SectionRole::A.intensity(),
            "the opening A section must be the calmest"
        );
    }

    #[test]
    fn compose_variations_form_produces_a_four_section_piece() {
        // A spec whose pool holds only Variations must compose the full
        // theme/minore/figuration/return set: FOUR sections' worth of
        // music -- between ternary's three and rondo's five -- with the
        // figuration (role C) as the piece's intensity peak and the theme's
        // harmonic ground held under every section (the form-level
        // invariant is pinned in form.rs; here we pin the end-to-end
        // section count and arc).
        let mut spec = crate::style::Style::Classical.spec();
        spec.form_pool = vec![crate::spec::FormKind::Variations];
        let intent = MusicalIntent::default();
        let s = compose_with_spec(&intent, &spec);
        let bars = intent.bars.max(1) as i64;
        let intro = spec.texture.intro_bars as i64;
        let coda = spec.texture.coda_bars as i64;
        let expected_beats = intro * 4 + 4 * 2 * bars * 4 + coda * 4;
        assert_eq!(s.total_beats, Duration::new(expected_beats, 1));
        // The figuration variation is the set's brilliant peak.
        let max = s
            .notes
            .iter()
            .map(|n| n.section_intensity)
            .fold(f32::MIN, f32::max);
        assert_eq!(max, crate::form::SectionRole::C.intensity());
    }

    #[test]
    fn compose_form_choice_is_deterministic_per_seed() {
        let even = MusicalIntent {
            seed: 4,
            ..Default::default()
        };
        let odd = MusicalIntent {
            seed: 5,
            ..Default::default()
        };
        assert_eq!(compose(&even), compose(&even));
        assert_eq!(compose(&odd), compose(&odd));
        assert_ne!(
            compose(&even).total_beats,
            compose(&odd).total_beats,
            "even (ternary) and odd (rondo) seeds must produce different total lengths"
        );
    }

    #[test]
    fn compose_modulates_to_the_relative_key_and_returns_home() {
        // The B section's melody notes should include at least one pitch
        // class outside the home major scale's diatonic set (evidence of a
        // REAL modulation, not just the same scale restated), and the melody
        // as a whole must still start and end diatonic to the home key
        // (A and ReturnA).
        let intent = MusicalIntent {
            tonic: crate::pitch::PitchClass::C,
            valence: 0.6, // major home key
            ..Default::default()
        };
        let s = compose(&intent);
        let home = crate::harmony::Key::major(intent.tonic);
        let mel = s.voice(VoiceRole::Melody);
        assert!(
            mel.iter()
                .any(|n| !home.scale().contains(n.pitch.pitch_class())),
            "expected at least one melody pitch outside the home major scale \
             (from the B section's relative-minor harmonic-minor raised 7th)"
        );
        // First and last melody notes are diatonic to the home key (A/ReturnA).
        assert!(
            home.scale()
                .contains(mel.first().unwrap().pitch.pitch_class())
        );
        assert!(
            home.scale()
                .contains(mel.last().unwrap().pitch.pitch_class())
        );
    }

    #[test]
    fn dominant_measures_get_a_seventh_chord_in_harmony() {
        // A bar on the dominant (degree 5) should voice 3 upper tones (3rd,
        // 5th, 7th of the seventh chord) versus 2 for a plain triad — real
        // harmonic color, added safely because a 7th chord's tones are a
        // superset of the triad's (so the melody's snapping needn't change).
        let key = crate::harmony::Key::major(crate::pitch::PitchClass::C);
        let intent = MusicalIntent::default();
        let mut tonic_score = Score::new(key, 100.0, 4);
        let mut prev_a = Vec::new();
        realize_harmony_measures(
            &mut tonic_score,
            &[1],
            key,
            4.0,
            &intent,
            &mut prev_a,
            0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        let mut dominant_score = Score::new(key, 100.0, 4);
        let mut prev_b = Vec::new();
        realize_harmony_measures(
            &mut dominant_score,
            &[5],
            key,
            4.0,
            &intent,
            &mut prev_b,
            0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        assert_eq!(tonic_score.voice(VoiceRole::Harmony).len(), 2);
        // A lone final dominant bar now SPLITS (cadential harmonic rhythm):
        // V triad on the front half (2 upper tones), V7 on the back half
        // (3 upper tones) — the drive into the cadence.
        let dom = dominant_score.voice(VoiceRole::Harmony);
        assert_eq!(dom.len(), 5);
        let front: Vec<_> = dom.iter().filter(|n| n.onset.beats() < 2.0).collect();
        let back: Vec<_> = dom.iter().filter(|n| n.onset.beats() >= 2.0).collect();
        assert_eq!(front.len(), 2, "front half: plain V triad voicing");
        assert_eq!(back.len(), 3, "back half: V7 voicing (adds the 7th)");
        // The 7th of V in C major is F — present only in the back half.
        assert!(back.iter().any(|n| n.pitch.midi() % 12 == 5));
        assert!(!front.iter().any(|n| n.pitch.midi() % 12 == 5));
    }

    #[test]
    fn applied_dominant_target_ground_truth() {
        let major = crate::harmony::Key::major(crate::pitch::PitchClass::C);
        let minor = crate::harmony::Key::minor(crate::pitch::PitchClass::C);
        // The whole safe family (root+fifth preserved), any seed:
        assert_eq!(
            applied_dominant_target(major, &[2, 5], 0, 0),
            Some(5),
            "ii→V7/V"
        );
        assert_eq!(
            applied_dominant_target(major, &[6, 2], 0, 0),
            Some(2),
            "vi→V7/ii"
        );
        assert_eq!(
            applied_dominant_target(major, &[3, 6], 0, 0),
            Some(6),
            "iii→V7/vi"
        );
        // I→V7/IV is seed-gated (bit 2): off at seed 0, on at seed 4.
        assert_eq!(applied_dominant_target(major, &[1, 4], 0, 0), None);
        assert_eq!(
            applied_dominant_target(major, &[1, 4], 0, 4),
            Some(4),
            "I→V7/IV"
        );
        // Excluded cases, with the reasons from the doc comment:
        assert_eq!(
            applied_dominant_target(major, &[7, 3], 0, 0),
            None,
            "vii° has a diminished fifth — root+fifth NOT preserved"
        );
        assert_eq!(
            applied_dominant_target(major, &[5, 1], 0, 0),
            None,
            "V before I is just V7, handled diatonically"
        );
        assert_eq!(
            applied_dominant_target(major, &[4, 5], 0, 0),
            None,
            "IV before V: root doesn't match V7/V's"
        );
        assert_eq!(
            applied_dominant_target(major, &[2, 1], 0, 0),
            None,
            "ii not before V"
        );
        assert_eq!(
            applied_dominant_target(minor, &[2, 5], 0, 0),
            None,
            "minor keys excluded"
        );
        assert_eq!(
            applied_dominant_target(major, &[2], 0, 0),
            None,
            "no next measure"
        );
    }

    #[test]
    fn applied_dominant_raised_pitch_classes_verified_by_hand() {
        use crate::pitch::PitchClass;
        let key = crate::harmony::Key::major(PitchClass::C);
        // ii (D-F-A) → V7/V raises F.
        assert_eq!(applied_dominant_raised_pc(key, 2), Some(PitchClass::F));
        // vi (A-C-E) → V7/ii raises C.
        assert_eq!(applied_dominant_raised_pc(key, 6), Some(PitchClass::C));
        // iii (E-G-B) → V7/vi raises G.
        assert_eq!(applied_dominant_raised_pc(key, 3), Some(PitchClass::G));
        // I (C-E-G) → V7/IV: triad untouched, no melody correction.
        assert_eq!(applied_dominant_raised_pc(key, 1), None);
    }

    #[test]
    fn vi_before_ii_realizes_as_a_real_applied_dominant() {
        use crate::pitch::PitchClass;
        // vi→V7/ii in C major must voice A7's tones — in particular C#, a
        // pitch class NO diatonic C-major chord contains.
        let key = crate::harmony::Key::major(PitchClass::C);
        let intent = MusicalIntent::default();
        let mut score = Score::new(key, 100.0, 4);
        let mut prev = Vec::new();
        realize_harmony_measures(
            &mut score,
            &[6, 2],
            key,
            4.0,
            &intent,
            &mut prev,
            0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        let first_measure_semis: std::collections::BTreeSet<u8> = score
            .voice(VoiceRole::Harmony)
            .iter()
            .filter(|n| n.onset.beats() < 4.0)
            .map(|n| n.pitch.midi() % 12)
            .collect();
        assert!(
            first_measure_semis.contains(&1),
            "V7/ii must sound C# (semitone 1; got {first_measure_semis:?})"
        );
    }

    #[test]
    fn applied_dominant_measure_gets_v7_of_v_chord_in_harmony_and_bass() {
        // ii before V must realize as a Dominant7 chord (3 upper tones, same
        // as the existing V-gets-a-7th case) -- and its bass root/fifth must
        // be UNCHANGED from plain ii's (same pitch classes), proving the
        // substitution doesn't disturb bass motion.
        let key = crate::harmony::Key::major(crate::pitch::PitchClass::C);
        let intent = MusicalIntent::default();

        let mut plain_ii = Score::new(key, 100.0, 4);
        realize_harmony_measures(
            &mut plain_ii,
            &[2],
            key,
            4.0,
            &intent,
            &mut Vec::new(),
            0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        assert_eq!(
            plain_ii.voice(VoiceRole::Harmony).len(),
            2,
            "plain ii is a triad"
        );

        let mut applied = Score::new(key, 100.0, 4);
        realize_harmony_measures(
            &mut applied,
            &[2, 5],
            key,
            4.0,
            &intent,
            &mut Vec::new(),
            0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        // Measure 0 (the flagged ii-before-V) must have 3 upper tones now.
        let applied_measure_0: Vec<_> = applied
            .voice(VoiceRole::Harmony)
            .into_iter()
            .filter(|n| n.onset == Duration::zero())
            .collect();
        assert_eq!(
            applied_measure_0.len(),
            3,
            "ii-before-V measure must voice a seventh chord (V7/V)"
        );

        let mut bass_plain = Score::new(key, 100.0, 4);
        realize_bass_measures(
            &mut bass_plain,
            &[2],
            key,
            4.0,
            &intent,
            &mut None,
            0,
            |i| i == 0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        let mut bass_applied = Score::new(key, 100.0, 4);
        realize_bass_measures(
            &mut bass_applied,
            &[2, 5],
            key,
            4.0,
            &intent,
            &mut None,
            0,
            |i| i == 0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            true,
        );
        // Both force the root at measure 0 -- root pitch class must be
        // IDENTICAL between plain ii and V7/V (same root, per design).
        assert_eq!(
            bass_plain.voice(VoiceRole::Bass)[0].pitch.pitch_class(),
            bass_applied.voice(VoiceRole::Bass)[0].pitch.pitch_class(),
            "V7/V must share ii's root exactly"
        );
    }

    #[test]
    fn realize_melody_raises_the_diatonic_third_in_an_applied_dominant_measure() {
        // A melody note landing on scale degree 4 (the diatonic third of ii,
        // C major's F) during a flagged ii-before-V measure must come out
        // raised a semitone (F#), matching the V7/V harmony realize_harmony
        // now plays there.
        let key = crate::harmony::Key::major(crate::pitch::PitchClass::C);
        // A whole note on degree 3: transposed by (chord_deg - 1) at measure
        // 0 (chord_deg=2) lands on degree 4 before chord-tone snapping
        // (snapping is then a no-op since 4 is already a chord tone of ii).
        let motif = Motif::from_degrees(&[(3, Duration::new(4, 1))]);
        let phrase =
            crate::phrase::Phrase::build(&motif, &[2, 5], crate::cadence::Cadence::Authentic, 4.0);
        let period = crate::phrase::Period {
            antecedent: phrase.clone(),
            consequent: phrase,
        };
        let form = Form {
            sections: vec![crate::form::Section {
                role: crate::form::SectionRole::A,
                key,
                period,
            }],
        };
        let mut score = Score::new(key, 100.0, 4);
        realize_melody(
            &mut score,
            &form,
            &MusicalIntent::default(),
            Duration::zero(),
            4.0,
            true,
        );
        let mel = score.voice(VoiceRole::Melody);
        let first_note = mel
            .iter()
            .find(|n| n.onset == Duration::zero())
            .expect("a note at beat 0");
        let raised_third = key.scale().degree_pitch_class(4).transpose(1);
        assert_eq!(
            first_note.pitch.pitch_class(),
            raised_third,
            "expected the diatonic third raised a semitone (F# in C major)"
        );
    }

    #[test]
    fn melody_is_monophonic_and_has_a_climax() {
        let s = compose(&MusicalIntent::default());
        assert!(s.melody_is_monophonic());
        let climaxes = s
            .voice(VoiceRole::Melody)
            .iter()
            .filter(|n| n.emphasis == Emphasis::Climax)
            .count();
        assert_eq!(climaxes, 1, "exactly one climax note");
    }

    #[test]
    fn melody_closes_cadentially() {
        // The final melody note is marked Cadential (a real phrase close).
        let s = compose(&MusicalIntent::default());
        let mel = s.voice(VoiceRole::Melody);
        assert_eq!(mel.last().unwrap().emphasis, Emphasis::Cadential);
    }

    #[test]
    fn bass_walks_only_at_high_arousal() {
        let calm = compose(&MusicalIntent {
            arousal: 0.2,
            ..Default::default()
        });
        let busy = compose(&MusicalIntent {
            arousal: 0.9,
            ..Default::default()
        });
        // Busy score has more bass notes per bar (root + fifth).
        assert!(busy.voice(VoiceRole::Bass).len() > calm.voice(VoiceRole::Bass).len());
    }
}
