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
///
/// Routes through [`crate::grammar::compose_with_grammar_plan`] via
/// `style.grammar_profile()` rather than calling [`compose_with_spec`]
/// directly. For the 25 styles whose grammar family has no dedicated
/// engine, `compose_with_grammar_plan`'s own fallback arm calls
/// `compose_with_spec_and_form` (the same function `compose_with_spec`
/// wraps), so this is behavior-preserving for them. For the three
/// "flagship" styles (`AfroCuban`/`Minimalism`/`HindustaniInspired`), this
/// is what finally lets them reach their dedicated
/// `groove_cycle`/`process_grammar`/`modal_arc` engines instead of the
/// shared period pipeline — see `Style::grammar_family`'s doc comment.
pub fn compose_styled(intent: &MusicalIntent, style: crate::style::Style) -> Score {
    crate::grammar::compose_with_grammar_plan(style.grammar_profile(), intent, &style.spec()).score
}

/// Compose from a [`CompositionSpec`](crate::spec::CompositionSpec) — the
/// open, user-authored form of what [`compose_styled`] does with built-in
/// presets. The spec owns every CHOICE (motifs, progression, textures,
/// forms, ensembles); this engine owns every INVARIANT (voice leading,
/// cadences, collision avoidance, superset-only substitutions). Call
/// `spec.validate()` first when the spec came from user input — this
/// function assumes a well-formed spec.
/// Which erosion ending this intent+spec selects. The attitude decides
/// what persistence costs: grief is denied reconstruction, curiosity
/// accepts partial return, everything else earns recovery (defiance
/// REFUSES to fade; joy restores). No attitude: the seed decides —
/// drawn from seed/3 because 3-entry form pools select Erosion at
/// seed % 3 == 1, and drawing the ending from seed % 3 too would lock
/// every seed-picked erosion into Acceptance.
pub fn erosion_ending_for(
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
) -> crate::passacaglia::ErosionEnding {
    match spec.attitude {
        Some(crate::spec::Attitude::Grief) => crate::passacaglia::ErosionEnding::Elegy,
        Some(crate::spec::Attitude::Curiosity) => crate::passacaglia::ErosionEnding::Acceptance,
        Some(_) => crate::passacaglia::ErosionEnding::Recovery,
        None => match (intent.seed / 3) % 3 {
            0 => crate::passacaglia::ErosionEnding::Recovery,
            1 => crate::passacaglia::ErosionEnding::Acceptance,
            _ => crate::passacaglia::ErosionEnding::Elegy,
        },
    }
}

/// The IDENTITY GRAMMAR a composition from this intent+spec will use —
/// the vocabulary the listening reviews built for how musical ideas live
/// over time: **memory** (ideas return — the period forms' hook-memory /
/// judgment arc), **subject** (the fugue's democratic development),
/// **persistence** (the ground remains), **erosion** (the ground loses
/// itself; the second element names what the ending cost), **lineage**
/// (the ground becomes its descendants). Surfaced per candidate so the
/// taste log records WHICH identity mechanism every ♥ endorsed — the
/// listener stays the final arbiter of whether a grammar is audible.
pub fn identity_grammar_for(
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
) -> (&'static str, Option<&'static str>) {
    match spec.form_kind(intent.seed) {
        crate::spec::FormKind::Ternary
        | crate::spec::FormKind::Rondo
        | crate::spec::FormKind::Variations => ("memory", None),
        crate::spec::FormKind::Fugue => ("subject", None),
        crate::spec::FormKind::Passacaglia => ("persistence", None),
        crate::spec::FormKind::Erosion => {
            ("erosion", Some(erosion_ending_for(intent, spec).name()))
        }
        crate::spec::FormKind::Lineage => ("lineage", None),
        crate::spec::FormKind::ProgSuite => ("long form", None),
        crate::spec::FormKind::Sonata => ("resolution", None),
        crate::spec::FormKind::Renaissance => ("equal voices", None),
        crate::spec::FormKind::Opera => ("dialogue", None),
    }
}

/// The shared ground audition: candidate 0 is the pipeline's own motif
/// (an artist-authored subject competes on equal terms and usually wins
/// its own audition); candidates 1..5 re-roll the spec's bank + hook
/// pipeline. Used by both `compose_with_spec` and
/// [`ground_audition_for`] so the Studio's logged scores are exactly the
/// scores the composition acted on.
fn ground_audition_impl(
    spec: &crate::spec::CompositionSpec,
    intent: &MusicalIntent,
    key: Key,
    meter: u8,
    motif: &Motif,
) -> (Motif, crate::passacaglia::GroundWorthiness) {
    let meter_beats = meter as f64;
    let derive = |k: usize| {
        if k == 0 {
            return motif.clone();
        }
        let candidate = spec.motif(intent.arousal, intent.seed.wrapping_add(k as u64 * 7919));
        if spec.texture.hook_cell {
            crate::hook::graft_hook(
                &candidate,
                &crate::hook::HookCell::generate_with(
                    &spec.melody,
                    intent.seed.wrapping_add(k as u64 * 7919),
                    meter_beats,
                ),
                meter_beats,
            )
        } else {
            candidate
        }
    };
    crate::passacaglia::audition_ground(key, meter, 5, intent.seed, derive)
}

/// The audition a composition from this intent+spec WOULD run — `None`
/// when the seed doesn't select a ground form. Exposed so the Studio can
/// log the winning subject's worthiness scores into keeper entries (the
/// ♥ data learns which weighting matches the listener's ear) without
/// duplicating audition logic; determinism guarantees these are the very
/// scores the composition acted on.
pub fn ground_audition_for(
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
) -> Option<crate::passacaglia::GroundWorthiness> {
    if !matches!(
        spec.form_kind(intent.seed),
        crate::spec::FormKind::Passacaglia
            | crate::spec::FormKind::Erosion
            | crate::spec::FormKind::Lineage
    ) {
        return None;
    }
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
    let meter = spec.meter;
    let motif = spec.motif(intent.arousal, intent.seed);
    let motif = if spec.texture.hook_cell {
        crate::hook::graft_hook(
            &motif,
            &crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter as f64),
            meter as f64,
        )
    } else {
        motif
    };
    Some(ground_audition_impl(spec, intent, key, meter, &motif).1)
}

/// Compose a Sonata-form score while retaining its exact plan and obligation
/// evidence. Returns `None` when the resolved form for this seed is not Sonata.
///
/// This is the plan-preserving counterpart of [`compose_with_spec`] used by
/// Studio's narrow cognitive-intervention path. It intentionally resolves the
/// same key, tempo, motif orientation, and hook graft as the canonical composer.
pub fn compose_sonata_with_plan(
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
) -> Option<crate::sonata::SonataRealization> {
    if spec.form_kind(intent.seed) != crate::spec::FormKind::Sonata {
        return None;
    }
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
    let tempo = spec.tempo(intent.arousal)
        * match spec.attitude {
            Some(crate::spec::Attitude::Joy) => 1.08,
            Some(crate::spec::Attitude::Grief) => 0.85,
            _ => 1.0,
        };
    let meter = spec.meter;
    let meter_beats = meter as f64;
    let motif = spec.motif(intent.arousal, intent.seed);
    let motif = if spec.texture.hook_cell {
        crate::hook::graft_hook(
            &motif,
            &crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter_beats),
            meter_beats,
        )
    } else {
        motif
    };
    Some(crate::sonata::realize_sonata_with_plan(
        key,
        tempo,
        meter_beats,
        &motif,
        intent.seed,
        intent,
    ))
}

pub fn compose_with_spec(intent: &MusicalIntent, spec: &crate::spec::CompositionSpec) -> Score {
    compose_with_spec_and_form(intent, spec).0
}

/// Same composition as [`compose_with_spec`], but ALSO returns the internal
/// [`Form`] when the chosen form kind builds one — `None` for the 6+ kinds
/// (Fugue, ProgSuite, Sonata, Renaissance, Opera, the 3 ground forms) whose
/// early-return branches never construct a `Form` at all; only Ternary/
/// Rondo/Variations do (see the `match spec.form_kind(...)` below). This is
/// the same function body as `compose_with_spec` — every early `return`
/// below is now written `return (value, None)` so the two functions can
/// never drift apart, and `compose_with_spec` is a thin wrapper so its
/// 60+ existing callers are byte-for-byte unaffected.
pub fn compose_with_spec_and_form(
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
) -> (Score, Option<Form>) {
    compose_with_spec_and_form_and_grammar(intent, spec, None)
}

/// [`compose_with_spec_and_form`], but the phrase-type choice (period vs.
/// sentence) and the harmonic-cadence steering for Ternary/Rondo/Variations
/// forms can be driven by a family's declared
/// [`GrammarProfile`](crate::grammar::GrammarProfile) instead of always
/// falling back to the plain arousal/energy heuristic and forced V–I
/// cadences. `grammar: None` makes this function IDENTICAL to
/// `compose_with_spec_and_form` (which forwards `None` here) — the extra
/// parameter only changes behavior when a caller (currently only
/// [`crate::grammar::compose_with_grammar_plan`]) opts in with `Some`.
/// Decide period-vs-sentence phrase structure for the shared Ternary/Rondo/
/// Variations pipeline. High arousal/energy gets the SENTENCE archetype
/// (statement → repetition → fragmentation): its accelerating,
/// driving-toward-the-cadence feel fits excited states. Calmer states get
/// the balanced statement/variation/statement arc of the plain developed
/// period. That's a real STRUCTURAL choice driven by the intent, not just a
/// faster tempo — EXCEPT for grammar families other than `PeriodSentence`,
/// which have their own declared `PhraseGrammar` identity (fragmentation-
/// driven families always use sentence structure, symmetric repeat-and-vary
/// families always use period structure) that the arousal heuristic would
/// otherwise silently overwrite. `PeriodSentence`'s own name says it
/// legitimately uses BOTH, chosen by energy — exactly today's behavior — so
/// it (and the ungraded `None` case) keeps the heuristic.
fn use_sentence_for(
    grammar: Option<crate::grammar::GrammarProfile>,
    intent: &MusicalIntent,
) -> bool {
    match grammar.map(|g| g.family) {
        Some(crate::grammar::GrammarFamily::PeriodSentence) | None => {
            intent.arousal > 0.6 || intent.energy > 0.75
        }
        // Fragmentation-driven, tension-building families: sentence's
        // statement -> repetition -> fragmentation arc fits their own
        // nature better than a manufactured arousal cutoff.
        Some(
            crate::grammar::GrammarFamily::Developmental
            | crate::grammar::GrammarFamily::BluesCallResponse
            | crate::grammar::GrammarFamily::JazzChorus
            | crate::grammar::GrammarFamily::DramaticAdaptive,
        ) => true,
        // Symmetric repeat-and-vary families: period's balanced antecedent/
        // consequent fits better than manufactured fragmentation.
        Some(
            crate::grammar::GrammarFamily::GroundVariation
            | crate::grammar::GrammarFamily::StrophicSong
            | crate::grammar::GrammarFamily::AmbientTextural,
        ) => false,
        // Contrapuntal/GrooveCycle/RagaModalArc/ProcessAdditive never reach
        // this function (dedicated bypass engines) — kept only so the
        // match stays exhaustive if that ever changes.
        Some(_) => false,
    }
}

pub fn compose_with_spec_and_form_and_grammar(
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
    grammar: Option<crate::grammar::GrammarProfile>,
) -> (Score, Option<Form>) {
    let (mut score, form) = compose_with_spec_and_form_and_grammar_impl(intent, spec, grammar);
    deoverlap_all_voices(&mut score);
    sync_total_beats(&mut score);
    crate::score_validation::debug_assert_no_structural_defects(
        &score,
        "compose_with_spec_and_form_and_grammar",
    );
    (score, form)
}

fn compose_with_spec_and_form_and_grammar_impl(
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
    grammar: Option<crate::grammar::GrammarProfile>,
) -> (Score, Option<Form>) {
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
            &crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter_beats),
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
        return (
            crate::fugue::realize_fugue(key, tempo, meter, &motif, intent.seed),
            None,
        );
    }
    // Progressive Folk/Rock's habit: a genuine mid-piece METER CHANGE. No
    // single `meter_beats` scalar can represent more than one meter, so
    // this bypasses the period pipeline exactly like the fugue/
    // ground-music families do — see `crate::prog_suite`.
    if spec.form_kind(intent.seed) == crate::spec::FormKind::ProgSuite {
        return (
            crate::prog_suite::realize_prog_suite(key, tempo, &motif, intent.seed, intent, spec),
            None,
        );
    }
    // Sonata's habit: tonal CONFLICT AND RESOLUTION — a second subject
    // introduced in a foreign key that the recapitulation brings home.
    // The key relationships across exposition/development/recap can't be
    // expressed by the plain ternary/rondo B-key machinery, so this
    // bypasses the period pipeline too — see `crate::sonata`.
    if spec.form_kind(intent.seed) == crate::spec::FormKind::Sonata {
        return (
            compose_sonata_with_plan(intent, spec)
                .expect("the resolved Sonata form must produce a plan")
                .score,
            None,
        );
    }
    // Renaissance polyphony's habit: three EQUAL voices, no subject-entry
    // hierarchy — the whole texture is species-fitted independent lines,
    // not a melody-plus-accompaniment realization, so this bypasses the
    // period pipeline too — see `crate::renaissance`.
    if spec.form_kind(intent.seed) == crate::spec::FormKind::Renaissance {
        return (
            crate::renaissance::realize_renaissance(key, tempo, meter, &motif, intent.seed),
            None,
        );
    }
    // Opera/Art Song's habit: two INDEPENDENT melodic identities (Theme A
    // in Melody, Theme B in CounterMelody) in structured conversation —
    // no single continuous melody voice can carry a dialogue between two
    // characters, so this bypasses the period pipeline too. Builds its own
    // themes internally (unlike the other bypass forms), so it doesn't
    // consume the spec-derived `motif` at all — see `crate::opera`.
    if spec.form_kind(intent.seed) == crate::spec::FormKind::Opera {
        return (
            crate::opera::realize_opera(key, tempo, meter_beats, intent.seed, intent),
            None,
        );
    }
    // Same reasoning for the ground-music family: the ground is the form.
    // All three grammars first AUDITION candidate subjects (the same bank
    // + hook pipeline, re-rolled) and grant the ground to the most worthy
    // — selection over generation. Judgment is advisory at this layer too:
    // candidate 0 is the spec's own pick, so an artist-authored motif
    // competes on equal terms and usually wins its own audition.
    let ground_form = matches!(
        spec.form_kind(intent.seed),
        crate::spec::FormKind::Passacaglia
            | crate::spec::FormKind::Erosion
            | crate::spec::FormKind::Lineage
    );
    if ground_form {
        let (subject, _worthiness) = ground_audition_impl(spec, intent, key, meter, &motif);
        let score = match spec.form_kind(intent.seed) {
            crate::spec::FormKind::Erosion => {
                let ending = erosion_ending_for(intent, spec);
                crate::passacaglia::realize_erosion(
                    key,
                    tempo,
                    meter,
                    &subject,
                    intent.seed,
                    ending,
                )
            }
            crate::spec::FormKind::Lineage => {
                crate::passacaglia::realize_lineage(key, tempo, meter, &subject, intent.seed)
            }
            _ => crate::passacaglia::realize_passacaglia(
                key,
                tempo,
                meter,
                &subject,
                intent.seed,
                false,
            ),
        };
        return (score, None);
    }
    let bars = intent.bars.max(1);
    let base = spec.progression(bars, intent.seed);
    let use_sentence = use_sentence_for(grammar, intent);
    let form_grammar = grammar.map(|g| crate::form::FormGrammarContext {
        harmony: g.harmony,
        spec,
    });
    // Large-scale form varies by seed within the spec's pool. The built-in
    // presets list [Ternary, Rondo], preserving the original even→ternary /
    // odd→rondo behavior for every existing caller.
    let mut form = match spec.form_kind(intent.seed) {
        crate::spec::FormKind::Ternary => Form::ternary(
            &motif,
            key,
            &base,
            meter_beats,
            intent.seed,
            use_sentence,
            form_grammar,
        ),
        crate::spec::FormKind::Rondo => Form::rondo(
            &motif,
            key,
            &base,
            meter_beats,
            intent.seed,
            use_sentence,
            form_grammar,
        ),
        crate::spec::FormKind::Variations => Form::variations(
            &motif,
            key,
            &base,
            meter_beats,
            intent.seed,
            use_sentence,
            form_grammar,
        ),
        // Handled by the early returns above — none of these reach the
        // period pipeline.
        crate::spec::FormKind::Fugue => unreachable!("fugue branch returns before form building"),
        crate::spec::FormKind::ProgSuite => {
            unreachable!("prog-suite branch returns before form building")
        }
        crate::spec::FormKind::Sonata => {
            unreachable!("sonata branch returns before form building")
        }
        crate::spec::FormKind::Renaissance => {
            unreachable!("renaissance branch returns before form building")
        }
        crate::spec::FormKind::Opera => {
            unreachable!("opera branch returns before form building")
        }
        crate::spec::FormKind::Passacaglia
        | crate::spec::FormKind::Erosion
        | crate::spec::FormKind::Lineage => {
            unreachable!("ground-form branch returns before form building")
        }
    };
    if spec.texture.harmonic_sequence {
        apply_harmonic_sequence(&mut form);
    }
    let form = form;

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
            crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter_beats)
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
        tx.seventh_chords,
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
            &crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter_beats),
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
                &crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter_beats),
            );
        }
    }
    apply_development_style(
        &mut score,
        &form,
        spec.development,
        &motif,
        intent.bars.max(1) as i64,
        meter_beats,
        intent.seed,
    );
    if tx.additive_process {
        apply_additive_process(
            &mut score,
            &form,
            intent.bars.max(1) as i64,
            meter_beats,
            intent.seed,
        );
    }
    if tx.deceptive_close {
        apply_deceptive_first_close(&mut score, key, intent.bars.max(1) as i64, meter_beats);
    }
    apply_pivot_modulations(&mut score, &form, intent.bars.max(1) as i64, meter_beats);
    if tx.deceptive_close {
        // The evasion travels with the deception: styles that deny their
        // first close also let their LAST mid-piece close slip past, so
        // the coda becomes the true arrival.
        apply_evaded_return_close(
            &mut score,
            key,
            &form,
            intent.bars.max(1) as i64,
            meter_beats,
        );
    }
    apply_appoggiaturas(&mut score, key, &spec.melody, intent.seed);
    if tx.roll_ornaments {
        apply_roll_ornaments(&mut score, key, &spec.melody, intent.seed);
    } else {
        apply_grace_ornaments(&mut score, key, &spec.melody, intent.seed);
    }
    apply_blue_notes(&mut score, key, spec.melody.blue_note_rate, intent.seed);
    apply_suspensions(&mut score, key, tx.suspension_rate, intent.seed);
    if tx.planing {
        // After every melody-altering pass, so the chord shape rides the
        // FINAL melodic contour, not a pre-ornament draft of it.
        apply_parallel_planing(&mut score, &form, bars as i64, meter_beats);
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
    if tx.harmonic_stasis {
        // After intro/coda, same reasoning as the drone pedal below — a
        // drone/stasis texture that dropped its habit for the intro or
        // coda would stop being what it is.
        apply_harmonic_stasis(&mut score);
    }
    if tx.full_drone {
        // Supersedes the plain drone below (it calls `apply_drone_pedal`
        // itself internally, then replaces Harmony too) — after intro/coda
        // so the pad covers the FULL span they extend.
        apply_full_drone(&mut score, key);
    } else if tx.drone {
        // After intro/coda so the pedal covers the FULL span they extend
        // — a drone that dropped out for the intro or coda would stop
        // being a drone.
        apply_drone_pedal(&mut score, key);
    }
    // LAST, so intro shifts and the coda's own cadence are included: the
    // style's phrase rhetoric — the conversational layer the second
    // melodic-DNA review localized ("the opening sentence differs; the
    // conversation afterward still sounds recognizably Muse").
    apply_phrase_rhetoric(&mut score, spec.rhetoric, meter_beats);
    (score, Some(form))
}

/// PARALLEL PLANING — the Impressionist style's habit: in the contrast
/// (B/C) section, the harmony stops following functional root motion.
/// Instead it captures the chord SHAPE sounding at the section's first
/// melody note (its exact stack of intervals), then re-strikes that same
/// shape at every subsequent melody note, shifted by the precise interval
/// the melody just traveled from its starting note — the chord literally
/// rides the tune's contour. "Color over function": nothing here
/// resolves; the harmony is entirely defined by where the melody happens
/// to be standing right now.
fn apply_parallel_planing(score: &mut Score, form: &Form, bars: i64, meter_beats: f64) {
    for (si, section) in form.sections.iter().enumerate() {
        if !matches!(
            section.role,
            crate::form::SectionRole::B | crate::form::SectionRole::C
        ) {
            continue;
        }
        // A full section is a PERIOD — antecedent + consequent, each
        // `bars` bars — so its span is `2*bars`, not `bars`. (This is the
        // one place this pass DELIBERATELY differs from
        // `apply_development_style`'s identical-looking `si*2*bars`
        // start: development only ever touches the antecedent, half a
        // section; planing needs the whole thing, including the
        // consequent half `thin_departure` — on by default — would
        // otherwise strip of harmony entirely, leaving nothing to plane.)
        let start_bar = si as i64 * 2 * bars;
        let (lo, hi) = (
            start_bar as f64 * meter_beats,
            (start_bar + 2 * bars) as f64 * meter_beats,
        );
        // Snapshot the melody's own data BEFORE any mutation — indices
        // into `score.notes` go stale the moment `retain` runs.
        let mut melody: Vec<(Duration, Duration, i32, f32)> = score
            .notes
            .iter()
            .filter(|n| {
                n.role == VoiceRole::Melody && n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi
            })
            .map(|n| {
                (
                    n.onset,
                    n.duration,
                    n.pitch.midi() as i32,
                    n.section_intensity,
                )
            })
            .collect();
        melody.sort_by(|a, b| a.0.beats().total_cmp(&b.0.beats()));
        if melody.len() < 2 {
            continue;
        }
        let (base_onset, _, base_melody_midi, _) = melody[0];
        // The chord shape sounding CLOSEST to the section's first melody
        // note — not necessarily an exact onset match, since the
        // accompaniment's own grid (e.g. Arpeggio's eighths) needn't
        // land precisely where the motif's first tone does, and
        // `thin_departure` may leave the antecedent half with no harmony
        // at all (the nearest search still finds the consequent's).
        let base_harmony_onset = score
            .notes
            .iter()
            .filter(|n| {
                n.role == VoiceRole::Harmony && n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi
            })
            .map(|n| n.onset)
            .min_by(|a, b| {
                (a.beats() - base_onset.beats())
                    .abs()
                    .total_cmp(&(b.beats() - base_onset.beats()).abs())
            });
        let Some(base_harmony_onset) = base_harmony_onset else {
            continue;
        };
        let base_shape: Vec<(i32, f32)> = score
            .notes
            .iter()
            .filter(|n| {
                n.role == VoiceRole::Harmony
                    && (n.onset.beats() - base_harmony_onset.beats()).abs() < 1e-6
            })
            .map(|n| (n.pitch.midi() as i32, n.velocity))
            .collect();
        if base_shape.is_empty() {
            continue;
        }
        score.notes.retain(|n| {
            !(n.role == VoiceRole::Harmony && n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi)
        });
        for (onset, duration, midi, si_intensity) in melody {
            let shift = midi - base_melody_midi;
            for &(base_midi, vel) in &base_shape {
                score.push(ScoreNote {
                    pitch: Pitch::from_midi((base_midi + shift).clamp(0, 127) as u8),
                    onset,
                    duration,
                    velocity: vel,
                    role: VoiceRole::Harmony,
                    emphasis: Emphasis::Normal,
                    section_intensity: si_intensity,
                });
            }
        }
    }
}

/// DEVELOPMENT DNA — the last shared language the listening reviews
/// localized ("the opening sentence differs; the conversation afterward
/// still sounds recognizably Muse... the hooks are different; the
/// development grammar is still shared"). Each departure section's
/// ANTECEDENT melody is rebuilt from the HOME theme through the style's
/// own way of developing — then fitted against the section's real bass
/// with the species fitter, in the section's own key:
///
/// - **Sequential**: the theme restated one diatonic step lower (or
///   higher, by seed) each bar — the classical sequence schema.
/// - **Figural**: progressive figuration — each bar's statement carries
///   more connecting tones than the last.
/// - **Fragmenting**: each bar keeps a shorter head than the last,
///   silence growing behind it.
/// - **Intensifying**: register climbs a diatonic step every bar
///   (monotonically), figuration accumulates like Figural, AND velocity
///   itself crescendos bar over bar — a real dramatic arc, not a
///   decorative wash.
/// - **Wandering**: a genuine random walk — each bar steps ±1 or ±2
///   diatonic degrees from wherever the last bar ended, so the line can
///   reverse direction mid-passage, unlike Sequential's single commitment.
/// - **Classic**: untouched (pinned no-op).
///
/// Only DEPARTURES (B/C) develop — the theme's own sections state, the
/// returns remember; development is what departures are FOR. The
/// consequent is left intact (it owns the section's cadence).
fn apply_development_style(
    score: &mut Score,
    form: &Form,
    dna: crate::spec::DevelopmentDna,
    motif: &crate::motif::Motif,
    bars: i64,
    meter_beats: f64,
    seed: u64,
) {
    use crate::spec::DevelopmentDna;
    if dna == DevelopmentDna::Classic {
        return;
    }
    let meter_i = meter_beats as i64;
    // The climax note (and whatever `realize_melody`'s grace-note pass or
    // `apply_held_arrivals` already attached to it — a dwelt-on duration,
    // an adjacent grace note) must survive a rebuild intact: both of
    // those ran BEFORE this pass and specifically decorated that one
    // note's exact position, so re-marking a DIFFERENT note as climax
    // afterward (the fallback below) would strand their work. Protect the
    // climax's own bar from being cleared/rebuilt at all.
    let climax_span = score
        .notes
        .iter()
        .find(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Climax)
        .map(|n| (n.onset.beats(), (n.onset + n.duration).beats()));
    for (si, section) in form.sections.iter().enumerate() {
        if !matches!(
            section.role,
            crate::form::SectionRole::B | crate::form::SectionRole::C
        ) {
            continue;
        }
        let start_bar = si as i64 * 2 * bars;
        let (lo, hi) = (
            start_bar as f64 * meter_beats,
            (start_bar + bars) as f64 * meter_beats,
        );
        // The section's real bass is the cantus the new line must obey.
        let cantus: Vec<crate::counterpoint::CantusEvent> = score
            .notes
            .iter()
            .filter(|n| {
                n.role == VoiceRole::Bass && n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi
            })
            .map(|n| crate::counterpoint::CantusEvent {
                onset: n.onset.beats(),
                duration: n.duration.beats(),
                pitch: n.pitch,
            })
            .collect();
        // Which bars (if any) the climax NOTE'S FULL SPAN touches — not just
        // its onset bar. `apply_held_arrivals`'s climax hold can absorb the
        // bar's own following note and extend the climax past its onset
        // bar's boundary; protecting onset-bar alone left the NEXT bar free
        // to be cleared and rebuilt from its nominal start, producing a new
        // note that began before the extended climax actually ended (a real
        // overlap, not just a stale-name bug). Protect every bar the span
        // `[onset, onset+duration)` touches.
        let climax_bars: Option<(i64, i64)> = climax_span
            .filter(|&(co, _)| co >= lo - 1e-9 && co < hi)
            .map(|(co, ce)| {
                let start_bar = ((co - lo) / meter_beats) as i64;
                let end_bar = (((ce - 1e-9).max(co) - lo) / meter_beats) as i64;
                (start_bar, end_bar.max(start_bar))
            });
        // Clear the antecedent melody EXCEPT the climax's own bar(s); the
        // device line replaces the rest.
        score.notes.retain(|n| {
            if n.role != VoiceRole::Melody || n.onset.beats() < lo - 1e-9 || n.onset.beats() >= hi {
                return true;
            }
            let this_bar = ((n.onset.beats() - lo) / meter_beats) as i64;
            climax_bars.is_some_and(|(s, e)| this_bar >= s && this_bar <= e)
        });
        let scale = section.key.scale();
        let intensity = section.role.intensity();
        let descending = seed.is_multiple_of(2);
        let mut wander_pos: i32 = 0; // Wandering's cumulative random-walk position
        for b in 0..bars {
            if climax_bars.is_some_and(|(s, e)| b >= s && b <= e) {
                // This bar's original notes (the climax itself, plus
                // whatever grace note or held-arrival dwelling already
                // decorates it) were preserved above rather than rebuilt.
                // Intensifying's arc still needs to stay coherent, though
                // — scale the ALREADY-realized velocity by the same
                // crescendo curve the rebuilt bars get, rather than
                // rebuilding pitch/rhythm content the climax note owns.
                if dna == DevelopmentDna::Intensifying {
                    let bar_crescendo = 1.0 + 0.35 * (b as f32 / (bars - 1).max(1) as f32);
                    let (bar_lo, bar_hi) = (
                        lo + b as f64 * meter_beats,
                        lo + (b + 1) as f64 * meter_beats,
                    );
                    for n in score.notes.iter_mut() {
                        if n.role == VoiceRole::Melody
                            && n.onset.beats() >= bar_lo - 1e-9
                            && n.onset.beats() < bar_hi
                        {
                            n.velocity = (n.velocity * bar_crescendo).clamp(0.0, 1.0);
                        }
                    }
                }
                continue;
            }
            let line = match dna {
                DevelopmentDna::Sequential => {
                    let step = if descending { -(b as i32) } else { b as i32 };
                    motif.transpose(step)
                }
                DevelopmentDna::Figural => {
                    // Progressive: bar 0 plain, later bars figured (twice
                    // by the midpoint) — decoration accumulating.
                    let mut m = motif.clone();
                    for _ in 0..(b.min(2)) {
                        m = crate::form::figuration_variation(&m, seed.wrapping_add(b as u64));
                    }
                    m
                }
                DevelopmentDna::Fragmenting => {
                    let keep = meter_beats * (bars - b) as f64 / bars as f64;
                    crate::fugue::head_fragment(
                        motif,
                        Duration::new((keep.max(1.0) * 480.0) as i64, 480),
                    )
                }
                DevelopmentDna::Intensifying => {
                    // Register climbs MONOTONICALLY (never the down-by-seed
                    // option Sequential allows) while figuration accumulates
                    // like Figural — both axes point toward the same peak
                    // on the last bar.
                    let mut m = motif.transpose(b as i32);
                    for _ in 0..(b.min(2)) {
                        m = crate::form::figuration_variation(&m, seed.wrapping_add(b as u64));
                    }
                    m
                }
                DevelopmentDna::Wandering => {
                    // A genuine random walk: each bar takes a small step
                    // (±1 or ±2 diatonic degrees) from wherever the LAST
                    // bar ended, so the line can reverse direction
                    // mid-passage — unlike Sequential, which commits to
                    // one direction for the whole span.
                    let h = {
                        let mut z = seed ^ (b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                        z ^ (z >> 31)
                    };
                    let step = match h % 4 {
                        0 => -2,
                        1 => -1,
                        2 => 1,
                        _ => 2,
                    };
                    wander_pos += step;
                    motif.transpose(wander_pos)
                }
                DevelopmentDna::Classic => unreachable!(),
            };
            // Intensifying's third axis: velocity itself crescendos bar
            // over bar, climbing to +35% by the section's last bar — no
            // other DNA touches loudness across the span, only pitch/
            // rhythm content, so this is what makes a "build" audibly
            // different from a "wash" of the same note count.
            let bar_crescendo = if dna == DevelopmentDna::Intensifying {
                1.0 + 0.35 * (b as f32 / (bars - 1).max(1) as f32)
            } else {
                1.0
            };
            let at = Duration::new((start_bar + b) * meter_i, 1);
            let fitted = crate::counterpoint::fit_against(&cantus, &line, scale, 5, at.beats());
            let mut t = at;
            let mut first = true;
            for note in &fitted.notes {
                if let Some(d) = note.degree {
                    // Respect the melody register ceiling the realizer
                    // enforces elsewhere: fold down by octaves as needed.
                    let mut pitch = scale.degree_pitch(d, 5);
                    while pitch.midi() > MELODY_CEILING_MIDI {
                        pitch = pitch.transpose(-12);
                    }
                    score.push(ScoreNote {
                        pitch,
                        onset: t,
                        duration: note.duration,
                        velocity: (0.72 * intensity * bar_crescendo).clamp(0.0, 1.0),
                        role: VoiceRole::Melody,
                        emphasis: if first && b == 0 {
                            Emphasis::PhraseStart
                        } else {
                            Emphasis::Normal
                        },
                        section_intensity: intensity,
                    });
                    first = false;
                }
                t = t + note.duration;
            }
        }
    }
    // Rebuilding a departure's melody can silently ERASE the note the
    // initial realize_melody pass marked Climax, if it happened to fall
    // inside the window just cleared and rebuilt — a real, pre-existing
    // gap this pass never guarded (every style that opted into a
    // non-Classic DevelopmentDna before this fix could silently lose its
    // climax on some fraction of seeds; it just never had a test check
    // compose()'s own climax invariant through a non-Classic style until
    // Classical/Waltz/Folk/Playful/Lullaby/ModalFolk picked one up).
    // Restore the invariant: if development wiped the climax, the new
    // global melodic peak inherits it — "climax" IS "the highest note,"
    // so this is the correct general recovery, not a special case.
    if !score
        .notes
        .iter()
        .any(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Climax)
        && let Some(peak_idx) = score
            .notes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.role == VoiceRole::Melody)
            .max_by_key(|(_, n)| n.pitch.midi())
            .map(|(i, _)| i)
    {
        score.notes[peak_idx].emphasis = Emphasis::Climax;
        score.notes[peak_idx].velocity = (score.notes[peak_idx].velocity * 1.15).clamp(0.0, 1.0);
    }
}

/// THE EVADED CADENCE — the third expectation device (after the deceptive
/// close and the false recovery). Where the deception SUBSTITUTES an
/// arrival (V→vi), the evasion lets the cadence begin and then SLIP PAST:
/// the return section's own close lands on a first-inversion tonic (bass
/// on 3̂ instead of the root) with the melody kept off the tonic — closure
/// gestured at, weight withheld — so the coda that follows becomes the
/// piece's one true arrival. Gated with `deceptive_close` (the styles
/// whose drama wants expectation games); the FINAL sounding cadence (the
/// coda's) is never touched.
fn apply_evaded_return_close(
    score: &mut Score,
    key: Key,
    form: &Form,
    bars: i64,
    meter_beats: f64,
) {
    let sections = form.sections.len() as i64;
    if sections < 2 {
        return; // nothing mid-piece to evade
    }
    let target_bar = sections * 2 * bars - 1; // the last section's close
    let (lo, hi) = (
        target_bar as f64 * meter_beats,
        (target_bar + 1) as f64 * meter_beats,
    );
    let scale = key.scale();
    let mediant_bass = scale.degree_pitch(3, 2);
    let tonic_pc = key.tonic;
    let in_bar = |on: f64| on >= lo - 1e-9 && on < hi;
    for i in 0..score.notes.len() {
        let n = &score.notes[i];
        if !in_bar(n.onset.beats()) {
            continue;
        }
        match n.role {
            VoiceRole::Bass => {
                // First inversion: the ground under the cadence is the
                // mediant, not the root — the weight never lands.
                score.notes[i].pitch = mediant_bass;
            }
            VoiceRole::Melody => {
                if n.pitch.pitch_class() == tonic_pc {
                    // The tune gestures at home and steps to 3̂ instead.
                    score.notes[i].pitch = scale.degree_pitch(3, 5);
                }
            }
            _ => {}
        }
    }
}

/// APPOGGIATURAS — the leaning tone, by style DNA. A cadential melody
/// tone long enough to bear it is approached by an ACCENTED upper
/// neighbor ON the beat: the dissonance takes the accent and the beat,
/// the resolution takes the release — the exact ornament the tango
/// listening review requested ("dramatic appoggiaturas"). Rate comes from
/// [`crate::spec::MelodicDna::appoggiatura_rate`] (0 = the pass is
/// inert); which cadences lean is a scrambled per-note seed choice, so
/// two pieces at the same rate lean in different places.
fn apply_appoggiaturas(score: &mut Score, key: Key, dna: &crate::spec::MelodicDna, seed: u64) {
    if dna.appoggiatura_rate <= 0.0 {
        return;
    }
    let scale = key.scale();
    let mut additions: Vec<ScoreNote> = Vec::new();
    let mut counter = 0u64;
    for i in 0..score.notes.len() {
        let n = &score.notes[i];
        if n.role != VoiceRole::Melody
            || n.emphasis != Emphasis::Cadential
            || n.duration.beats() < 1.0
        {
            continue;
        }
        counter += 1;
        let roll = {
            let mut z = seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (z >> 33) as f64 / (1u64 << 31) as f64
        };
        if roll >= dna.appoggiatura_rate as f64 {
            continue;
        }
        // Find the resolution tone's degree so the neighbor is diatonic:
        // nearest degree whose pitch matches, then +1.
        let target_midi = n.pitch.midi() as i32;
        let degree = (1..=15)
            .min_by_key(|&d| (scale.degree_pitch(d, 4).midi() as i32 - target_midi).abs())
            .unwrap();
        let octave_hint = n.pitch;
        let lean_pitch = {
            let cands = [
                scale.degree_pitch(degree + 1, 4),
                scale.degree_pitch(degree + 1, 5),
                scale.degree_pitch(degree + 1, 3),
            ];
            cands
                .into_iter()
                .min_by_key(|p| (p.midi() as i32 - octave_hint.midi() as i32).abs())
                .unwrap()
        };
        // The lean takes the beat (half the tone, capped at one beat and
        // expressed in exact 480ths); the resolution takes the release.
        let total = n.duration.beats();
        let lean_beats = (total / 2.0).min(1.0);
        let lean_480 = (lean_beats * 480.0).round() as i64;
        let (onset, velocity, si) = (n.onset, n.velocity, n.section_intensity);
        let note = &mut score.notes[i];
        note.onset = onset + Duration::new(lean_480, 480);
        note.duration = note.duration.saturating_sub(Duration::new(lean_480, 480));
        note.velocity *= 0.92; // the resolution releases
        additions.push(ScoreNote {
            pitch: lean_pitch,
            onset,
            duration: Duration::new(lean_480, 480),
            velocity: (velocity * 1.1).clamp(0.0, 1.0), // the lean takes the accent
            role: VoiceRole::Melody,
            emphasis: Emphasis::Cadential,
            section_intensity: si,
        });
    }
    for a in additions {
        score.push(a);
    }
}

/// ORNAMENTS ("cuts") — the Celtic style's habit: a quick, UNACCENTED
/// grace note a diatonic step above ANY melody tone long enough to spare
/// it (not just cadential ones, unlike appoggiaturas). The grace borrows a
/// sliver of time from the note it decorates rather than taking the beat,
/// and lands quieter, never louder — the mechanism that keeps it from
/// being just another appoggiatura wearing a different name. Rate comes
/// from [`crate::spec::MelodicDna::ornament_rate`] (0 = the pass is
/// inert).
fn apply_grace_ornaments(score: &mut Score, key: Key, dna: &crate::spec::MelodicDna, seed: u64) {
    if dna.ornament_rate <= 0.0 {
        return;
    }
    let scale = key.scale();
    let mut additions: Vec<ScoreNote> = Vec::new();
    let mut counter = 0u64;
    for i in 0..score.notes.len() {
        let n = &score.notes[i];
        if n.role != VoiceRole::Melody || n.duration.beats() < 0.4 {
            continue;
        }
        counter += 1;
        let roll = {
            let mut z = seed ^ counter.wrapping_mul(0xD1B5_4A32_D192_ED03);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (z >> 33) as f64 / (1u64 << 31) as f64
        };
        if roll >= dna.ornament_rate as f64 {
            continue;
        }
        let target_midi = n.pitch.midi() as i32;
        let degree = (1..=15)
            .min_by_key(|&d| (scale.degree_pitch(d, 4).midi() as i32 - target_midi).abs())
            .unwrap();
        let octave_hint = n.pitch;
        let cut_pitch = {
            let cands = [
                scale.degree_pitch(degree + 1, 4),
                scale.degree_pitch(degree + 1, 5),
                scale.degree_pitch(degree + 1, 3),
            ];
            cands
                .into_iter()
                .min_by_key(|p| (p.midi() as i32 - octave_hint.midi() as i32).abs())
                .unwrap()
        };
        // The cut borrows a sixteenth, capped so it never eats more than a
        // fifth of the main note — a flick, not a lean.
        let total = n.duration.beats();
        let cut_beats: f64 = 0.25_f64.min(total * 0.2);
        let cut_480 = ((cut_beats * 480.0).round() as i64).max(1);
        let (onset, velocity, si) = (n.onset, n.velocity, n.section_intensity);
        let note = &mut score.notes[i];
        note.onset = onset + Duration::new(cut_480, 480);
        note.duration = note.duration.saturating_sub(Duration::new(cut_480, 480));
        additions.push(ScoreNote {
            pitch: cut_pitch,
            onset,
            duration: Duration::new(cut_480, 480),
            velocity: (velocity * 0.75).clamp(0.0, 1.0), // unaccented — a flick, not a lean
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: si,
        });
    }
    for a in additions {
        score.push(a);
    }
}

/// ROLLS — the Irish Traditional style's habit, and the engine's first
/// ORNAMENT CHAIN: not one grace note (Celtic's "cut", above) but a full
/// five-note figure — main, upper cut, main, lower cut, main — filling
/// EXACTLY the space of the original note. Every sub-note's duration
/// still sums to the original total: a pure surface elaboration, the same
/// safety property `Accompaniment` patterns already guarantee ("never a
/// wrong note, only a new placement of the right ones") applied to melody
/// instead of accompaniment. Only long-enough notes qualify — a roll
/// needs room to breathe — and a climax note is never touched (its
/// identity as THE arrival matters more than decoration). Mutually
/// exclusive with `apply_grace_ornaments`; reuses `MelodicDna::
/// ornament_rate` as its gate/probability, same as the cut.
fn apply_roll_ornaments(score: &mut Score, key: Key, dna: &crate::spec::MelodicDna, seed: u64) {
    if dna.ornament_rate <= 0.0 {
        return;
    }
    let scale = key.scale();
    let mut replace_at: Vec<usize> = Vec::new();
    let mut chains: Vec<Vec<ScoreNote>> = Vec::new();
    let mut counter = 0u64;
    for i in 0..score.notes.len() {
        let n = &score.notes[i];
        if n.role != VoiceRole::Melody || n.duration.beats() < 0.9 || n.emphasis == Emphasis::Climax
        {
            continue;
        }
        counter += 1;
        let roll_chance = {
            let mut z = seed ^ counter.wrapping_mul(0xE07C_2C1A_9B5D_71F3);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (z >> 33) as f64 / (1u64 << 31) as f64
        };
        if roll_chance >= dna.ornament_rate as f64 {
            continue;
        }
        let target_midi = n.pitch.midi() as i32;
        let degree = (1..=15)
            .min_by_key(|&d| (scale.degree_pitch(d, 4).midi() as i32 - target_midi).abs())
            .unwrap();
        let octave_hint = n.pitch;
        let neighbor = |offset: i32| -> crate::pitch::Pitch {
            let cands = [
                scale.degree_pitch(degree + offset, 4),
                scale.degree_pitch(degree + offset, 5),
                scale.degree_pitch(degree + offset, 3),
            ];
            cands
                .into_iter()
                .min_by_key(|p| (p.midi() as i32 - octave_hint.midi() as i32).abs())
                .unwrap()
        };
        let upper = neighbor(1);
        let lower = neighbor(-1);
        let total = n.duration.beats();
        // Two brief cuts (capped so the pattern never inverts into a
        // held-neighbor-tone feel), three mains sharing the rest.
        let cut_beats = (total * 0.1).min(0.12);
        let cut_dur = Duration::new(((cut_beats * 480.0).round() as i64).max(1), 480);
        let main_beats = ((total - 2.0 * cut_dur.beats()) / 3.0).max(0.01);
        let main_dur = Duration::new(((main_beats * 480.0).round() as i64).max(1), 480);
        let plan = [
            (n.pitch, main_dur, false),
            (upper, cut_dur, true),
            (n.pitch, main_dur, false),
            (lower, cut_dur, true),
            (n.pitch, main_dur, false),
        ];
        let mut t = n.onset;
        let mut chain = Vec::with_capacity(5);
        for (idx, &(pitch, dur, is_cut)) in plan.iter().enumerate() {
            let is_last = idx == plan.len() - 1;
            let used: f64 = chain.iter().map(|c: &ScoreNote| c.duration.beats()).sum();
            let this_dur = if is_last {
                // Consume the exact remainder — the chain's total duration
                // must equal the original note's, no rounding drift.
                Duration::new(
                    (((total - used).max(1e-6) * 480.0).round() as i64).max(1),
                    480,
                )
            } else {
                dur
            };
            chain.push(ScoreNote {
                pitch,
                onset: t,
                duration: this_dur,
                velocity: if is_cut {
                    n.velocity * 0.7 // unaccented — a flick, not a lean
                } else {
                    n.velocity
                },
                role: VoiceRole::Melody,
                emphasis: if idx == 0 {
                    n.emphasis
                } else {
                    Emphasis::Normal
                },
                section_intensity: n.section_intensity,
            });
            t = t + this_dur;
        }
        replace_at.push(i);
        chains.push(chain);
    }
    // Remove the originals from the highest index down (so earlier
    // indices stay valid), then push every chain's notes.
    for &i in replace_at.iter().rev() {
        score.notes.remove(i);
    }
    for chain in chains {
        for n in chain {
            score.push(n);
        }
    }
}

/// BLUE NOTES — the Blues style's habit: a melody tone landing on the
/// major third or the leading tone (scale degrees 3 and 7 — the two tones
/// blues coloring targets) is flattened by a semitone at random, WITHOUT
/// touching the harmony underneath, which stays major/dominant the whole
/// time. The mechanism that keeps this distinct from every prior
/// ornament: appoggiaturas and cuts both ADD a note; a blue note ALTERS
/// one that's already there — a deliberate melody/harmony scale mismatch,
/// not a new pitch event. Rate comes from
/// [`crate::spec::MelodicDna::blue_note_rate`] (0 = the pass is inert).
fn apply_blue_notes(score: &mut Score, key: Key, rate: f32, seed: u64) {
    if rate <= 0.0 {
        return;
    }
    let scale = key.scale();
    let third_pc = scale.degree_pitch_class(3);
    let seventh_pc = scale.degree_pitch_class(7);
    let mut counter = 0u64;
    for n in score.notes.iter_mut() {
        if n.role != VoiceRole::Melody {
            continue;
        }
        let pc = n.pitch.pitch_class();
        if pc != third_pc && pc != seventh_pc {
            continue;
        }
        counter += 1;
        let roll = {
            let mut z = seed ^ counter.wrapping_mul(0xA5A5_1234_5678_9ABC);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (z >> 33) as f64 / (1u64 << 31) as f64
        };
        if roll >= rate as f64 {
            continue;
        }
        n.pitch = n.pitch.transpose(-1);
    }
}

/// SUSPENSIONS — the Sacred Choral style's habit: the first ornament to
/// live in the HARMONY voice rather than the melody. At a bar-to-bar
/// chord change, find a real voice-leading candidate — a tone in the
/// OUTGOING chord sitting exactly one diatonic step above a tone in the
/// INCOMING chord — and, instead of striking the incoming chord cleanly,
/// tie the outgoing tone over (a prepared dissonance against the new
/// chord, accented — it leans in) then resolve it DOWN BY STEP a beat or
/// two later (releasing). This is the classical 4-3/7-6 suspension
/// mechanism: real chorale harmony, not appoggiatura's melodic upper
/// neighbor wearing a different name. Rate comes from
/// [`crate::spec::TextureSpec::suspension_rate`] (0 = the pass is inert).
fn apply_suspensions(score: &mut Score, key: Key, rate: f32, seed: u64) {
    if rate <= 0.0 {
        return;
    }
    let scale = key.scale();
    let mut onsets: Vec<f64> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Harmony)
        .map(|n| n.onset.beats())
        .collect();
    onsets.sort_by(f64::total_cmp);
    onsets.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let chord_at = |t: f64, score: &Score| -> Vec<(usize, Pitch, Duration, f32, f32)> {
        score
            .notes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.role == VoiceRole::Harmony && (n.onset.beats() - t).abs() < 1e-9)
            .map(|(idx, n)| (idx, n.pitch, n.duration, n.velocity, n.section_intensity))
            .collect()
    };

    let mut counter = 0u64;
    let mut resolved_indices: Vec<usize> = Vec::new();
    let mut additions: Vec<ScoreNote> = Vec::new();

    for w in onsets.windows(2) {
        let (t1, t2) = (w[0], w[1]);
        let chord1 = chord_at(t1, score);
        let chord2 = chord_at(t2, score);
        if chord1.is_empty() || chord2.is_empty() {
            continue;
        }
        let mut found: Option<(Pitch, usize, Pitch, Duration, f32, f32)> = None;
        for &(j_idx, p2, dur2, vel2, si2) in &chord2 {
            let degree = (1..=15)
                .min_by_key(|&d| (scale.degree_pitch(d, 4).midi() as i32 - p2.midi() as i32).abs())
                .unwrap();
            let target_pc = scale.degree_pitch_class(degree + 1);
            if let Some(&(_, p1, ..)) = chord1.iter().find(|&&(_, p1, ..)| {
                p1.pitch_class() == target_pc && (p1.midi() as i32 - p2.midi() as i32).abs() <= 3
            }) {
                found = Some((p1, j_idx, p2, dur2, vel2, si2));
                break;
            }
        }
        let Some((suspended_pitch, j_idx, resolved_pitch, dur2, vel2, si2)) = found else {
            continue;
        };
        counter += 1;
        let roll = {
            let mut z = seed ^ counter.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            (z >> 33) as f64 / (1u64 << 31) as f64
        };
        if roll >= rate as f64 {
            continue;
        }
        let onset2 = score.notes[j_idx].onset;
        let hold_beats = (dur2.beats() / 2.0).max(0.25);
        let hold = Duration::new((hold_beats * 480.0).round() as i64, 480);
        resolved_indices.push(j_idx);
        additions.push(ScoreNote {
            pitch: suspended_pitch,
            onset: onset2,
            duration: hold,
            velocity: (vel2 * 1.12).clamp(0.0, 1.0), // the dissonance leans in
            role: VoiceRole::Harmony,
            emphasis: Emphasis::Normal,
            section_intensity: si2,
        });
        additions.push(ScoreNote {
            pitch: resolved_pitch,
            onset: onset2 + hold,
            duration: dur2.saturating_sub(hold),
            velocity: (vel2 * 0.92).clamp(0.0, 1.0), // the resolution releases
            role: VoiceRole::Harmony,
            emphasis: Emphasis::Normal,
            section_intensity: si2,
        });
    }
    if !resolved_indices.is_empty() {
        let remove_set: std::collections::HashSet<usize> = resolved_indices.into_iter().collect();
        let mut idx = 0usize;
        score.notes.retain(|_| {
            let keep = !remove_set.contains(&idx);
            idx += 1;
            keep
        });
    }
    for a in additions {
        score.push(a);
    }
}

/// HARMONIC SEQUENCE — the Baroque Dance Suite style's habit: rewrite the
/// B section's harmonic PLAN itself with a real descending-fifths circle
/// (each chord a diatonic fifth above its successor: I-IV-vii°-iii-vi-ii-
/// V-I), the quintessential Baroque development device (the "Pachelbel"/
/// circle-of-fifths sequence). Every prior pass in this file works on
/// realized Score notes or realized chords; this one is upstream of all
/// of that — it mutates the stored scale-degree progression on the
/// [`Form`] BEFORE melody/harmony/bass realization, so every downstream
/// voice (chromatic corrections, chord choice, bass root motion) sees a
/// single consistent sequence rather than three independently-patched
/// voices. Each phrase's CADENTIAL TAIL is left untouched — the half
/// cadence's dominant (antecedent) and the authentic cadence's
/// dominant+tonic (consequent) are exactly what [`crate::phrase::Period::
/// parallel_in`] already steered the melody's final pitches toward, so
/// overwriting them would strand the melody's already-baked cadential
/// motion against a harmony that no longer resolves under it.
///
/// The circle-of-fifths relationship this generates (each degree a fifth
/// above its successor) is EXACTLY the condition [`applied_dominant_target`]
/// already tests for — so in a major-mode section the sequence naturally
/// chains into secondary dominants (a real, idiomatic Baroque variant);
/// in a minor-mode section (where that function is gated off) it stays
/// the plain diatonic circle. No new chromatic logic needed either way.
fn apply_harmonic_sequence(form: &mut Form) {
    // A fifth above `d` (1-indexed, wraps 1..=7): `((d - 1 + 4) % 7) + 1`.
    // Walking the circle DOWN by fifths is the inverse step, `+3 mod 7`
    // (equivalently `-4 mod 7`) — see `applied_dominant_target`'s own
    // `expected_c` formula for the same arithmetic in the other direction.
    let degree_at =
        |start: i32, i: i64| -> i32 { (((start - 1) as i64 + 3 * i).rem_euclid(7)) as i32 + 1 };
    for section in &mut form.sections {
        if section.role != SectionRole::B {
            continue;
        }
        let start = 1; // the section's own tonic — home_key.relative() in ternary/rondo
        let ante = &mut section.period.antecedent.progression;
        let ante_len = ante.len();
        for (i, d) in ante.iter_mut().enumerate() {
            if i + 1 == ante_len {
                continue; // preserve the half-cadence dominant
            }
            *d = degree_at(start, i as i64);
        }
        let cons = &mut section.period.consequent.progression;
        let cons_len = cons.len();
        for (i, d) in cons.iter_mut().enumerate() {
            if i + 2 >= cons_len {
                continue; // preserve the authentic cadence's [dominant, tonic]
            }
            // Continue the SAME circle where the antecedent left off — a
            // real sequence walks continuously across the phrase
            // boundary, it doesn't reset at the question/answer seam.
            // `ante_len - 1`, not `ante_len`: the antecedent's LAST slot
            // was pulled aside for its half-cadence dominant rather than
            // spending a step in the arithmetic walk, so the circle picks
            // back up from the same point it would have reached had that
            // slot not been diverted.
            *d = degree_at(start, (ante_len - 1 + i) as i64);
        }
    }
}

/// HARMONIC STASIS — the Ambient style's habit: when a voice (Harmony or
/// Bass) repeats the EXACT SAME pitch across two consecutive chord
/// onsets — the underlying harmony hasn't actually moved — tie the two
/// notes into one longer sustained note instead of re-striking. This is
/// the mechanism that turns "the same chord repeated every bar" (what
/// `realize_harmony_measures` produces on its own for a static
/// progression) into a genuine drone: repetition becomes duration, not
/// re-attack. A sequential sweep across each voice's onsets chains
/// arbitrarily long runs (a tone repeated across 8 consecutive bars ties
/// into ONE 8-bar note, not seven separate 2-bar ties) by tracking, per
/// pitch, the index of the note currently being extended; a pitch that
/// drops out of one onset's chord and later returns is treated as a NEW
/// attack, not a stale reconnection. Reusable by any future style
/// wanting genuine harmonic stillness (drone, soundscape, film-score
/// pad) rather than block-chord re-attack on every bar.
fn apply_harmonic_stasis(score: &mut Score) {
    for role in [VoiceRole::Harmony, VoiceRole::Bass] {
        let mut onsets: Vec<f64> = score
            .notes
            .iter()
            .filter(|n| n.role == role)
            .map(|n| n.onset.beats())
            .collect();
        onsets.sort_by(f64::total_cmp);
        onsets.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        let mut active: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
        let mut removed: Vec<usize> = Vec::new();

        for &t in &onsets {
            let chord: Vec<usize> = score
                .notes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.role == role && (n.onset.beats() - t).abs() < 1e-9)
                .map(|(idx, _)| idx)
                .collect();
            let mut still_active: std::collections::HashMap<u8, usize> =
                std::collections::HashMap::new();
            for &idx in &chord {
                let midi = score.notes[idx].pitch.midi();
                if let Some(&source) = active.get(&midi) {
                    let new_end = score.notes[idx].onset + score.notes[idx].duration;
                    let src_onset = score.notes[source].onset;
                    score.notes[source].duration = new_end.saturating_sub(src_onset);
                    removed.push(idx);
                    still_active.insert(midi, source);
                } else {
                    still_active.insert(midi, idx);
                }
            }
            active = still_active;
        }

        removed.sort_unstable();
        removed.dedup();
        for &i in removed.iter().rev() {
            score.notes.remove(i);
        }
    }
}

/// ADDITIVE PROCESS — the Minimalism style's habit: in the theme (A/
/// ReturnA) sections, the standard developed melody is replaced entirely
/// by a Glass-style additive-then-subtractive process built from the
/// piece's own hook cell — 1 note, then 1-2, then 1-2-3, growing by one
/// note each repetition until the WHOLE cell repeats, then shrinking back
/// down to 1 the same way, bouncing for as long as the section lasts. The
/// process substitutes for melodic argument entirely — nothing here
/// "develops" in the Classical sense; the process itself IS the content.
/// Every other pass in this file adds to or alters an existing voice;
/// this is the first that wholesale REPLACES one within its window
/// (mirroring how [`apply_drone_pedal`] replaces the whole bass, but
/// scoped to specific sections rather than the whole piece).
fn apply_additive_process(score: &mut Score, form: &Form, bars: i64, meter_beats: f64, seed: u64) {
    let cell = crate::hook::HookCell::generate_with(
        &crate::spec::MelodicDna::default(),
        seed,
        meter_beats,
    );
    if cell.notes.is_empty() {
        return;
    }
    for (si, section) in form.sections.iter().enumerate() {
        if !matches!(
            section.role,
            crate::form::SectionRole::A | crate::form::SectionRole::ReturnA
        ) {
            continue;
        }
        let start_bar = si as i64 * 2 * bars;
        let (lo, hi) = (
            start_bar as f64 * meter_beats,
            (start_bar + 2 * bars) as f64 * meter_beats,
        );
        let scale = section.key.scale();
        let intensity = section.role.intensity();
        score.notes.retain(|n| {
            !(n.role == VoiceRole::Melody && n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi)
        });
        let mut t = lo;
        let mut k: usize = 1;
        let mut growing = true;
        // Bound the loop by beats remaining, not iteration count — a
        // pathologically tiny cell duration could otherwise spin forever.
        while t < hi - 1e-9 {
            let prefix_len = k.min(cell.notes.len());
            for &(degree, dur) in &cell.notes[..prefix_len] {
                if t >= hi - 1e-9 {
                    break;
                }
                let remaining = hi - t;
                let this_dur = dur.beats().min(remaining);
                if this_dur <= 1e-9 {
                    break;
                }
                let mut pitch = scale.degree_pitch(degree, 5);
                while pitch.midi() > MELODY_CEILING_MIDI {
                    pitch = pitch.transpose(-12);
                }
                score.push(ScoreNote {
                    pitch,
                    onset: Duration::new((t * 480.0).round() as i64, 480),
                    duration: Duration::new((this_dur * 480.0).round().max(1.0) as i64, 480),
                    velocity: 0.5,
                    role: VoiceRole::Melody,
                    emphasis: Emphasis::Normal,
                    section_intensity: intensity,
                });
                t += this_dur;
            }
            if growing {
                k += 1;
                if k > cell.notes.len() {
                    k = cell.notes.len();
                    growing = false;
                }
            } else {
                k = k.saturating_sub(1);
                if k == 0 {
                    k = 1;
                    growing = true;
                }
            }
        }
    }
}

/// THE DRONE — the Celtic style's other habit: a sustained tonic-fifth
/// pedal replaces the walking bass for the whole piece. Deliberately
/// independent of the harmony above it (unlike every other bass note the
/// engine writes, which follows the chord roots) — that's the entire
/// point of a drone: it's the fixed ground a modal melody floats over, the
/// harmony moving freely against it. Runs LAST, after every other bass
/// note is in place, and simply replaces them all.
fn apply_drone_pedal(score: &mut Score, key: Key) {
    let total_bars = (score.total_beats.beats() / score.meter as f64).ceil() as i64;
    if total_bars <= 0 {
        return;
    }
    let bar = Duration::new(score.meter as i64, 1);
    let half = Duration::new(score.meter as i64, 2);
    let tonic = Pitch::new(key.tonic, 2);
    let fifth = key.scale().degree_pitch(5, 2);
    // Snapshot melody onsets/intensities BEFORE mutating the score, so the
    // pedal can still track the narrative arc even though it ignores the
    // harmony above it.
    let melody_track: Vec<(f64, f32)> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .map(|n| (n.onset.beats(), n.section_intensity))
        .collect();
    let intensity_near = |t: f64| -> f32 {
        melody_track
            .iter()
            .min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs()))
            .map(|&(_, si)| si)
            .unwrap_or(1.0)
    };
    score.notes.retain(|n| n.role != VoiceRole::Bass);
    for m in 0..total_bars {
        let onset = bar.scale(m, 1);
        let si = intensity_near(onset.beats());
        score.push(ScoreNote {
            pitch: tonic,
            onset,
            duration: half,
            velocity: 0.32,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: si,
        });
        score.push(ScoreNote {
            pitch: fifth,
            onset: onset + half,
            duration: half,
            velocity: 0.28,
            role: VoiceRole::Bass,
            emphasis: Emphasis::Normal,
            section_intensity: si,
        });
    }
}

/// THE FULL DRONE — the Hindustani-inspired style's habit, and an
/// escalation beyond Celtic's `apply_drone_pedal` above: that one
/// replaces only the walking BASS with a tonic-fifth pedal while Harmony
/// still moves through a real chord progression above it — "the harmony
/// moving freely against it," in its own doc's words. This one replaces
/// Harmony TOO, with a static tonic-fifth-octave pad, identical every
/// bar. There is no chord progression left anywhere in the piece, so
/// "tension without modulation" is a structural guarantee, not a mood —
/// there is nothing left TO modulate. The identical repeated pad is then
/// tied by `apply_harmonic_stasis` into one continuous sustain spanning
/// the whole piece — reusing the Ambient style's own device (repetition
/// becoming duration) rather than re-deriving tying logic, since the
/// effect wanted is identical.
fn apply_full_drone(score: &mut Score, key: Key) {
    apply_drone_pedal(score, key);
    let total_bars = (score.total_beats.beats() / score.meter as f64).ceil() as i64;
    if total_bars <= 0 {
        return;
    }
    let bar = Duration::new(score.meter as i64, 1);
    let scale = key.scale();
    let tonic = scale.degree_pitch(1, 3);
    let fifth = scale.degree_pitch(5, 3);
    let octave = scale.degree_pitch(8, 3);
    let melody_track: Vec<(f64, f32)> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .map(|n| (n.onset.beats(), n.section_intensity))
        .collect();
    let intensity_near = |t: f64| -> f32 {
        melody_track
            .iter()
            .min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs()))
            .map(|&(_, si)| si)
            .unwrap_or(1.0)
    };
    score.notes.retain(|n| n.role != VoiceRole::Harmony);
    for m in 0..total_bars {
        let onset = bar.scale(m, 1);
        let si = intensity_near(onset.beats());
        for pitch in [tonic, fifth, octave] {
            score.push(ScoreNote {
                pitch,
                onset,
                duration: bar,
                velocity: 0.22,
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity: si,
            });
        }
    }
    apply_harmonic_stasis(score);
}

/// PIVOT-CHORD MODULATION — the first REAL key travel in the engine.
/// Until now a departure section simply started in its new key: the
/// "relative" B was same-collection degree math and even the parallel-key
/// C teleported. Real modulation ANNOUNCES the new key: the outgoing
/// section's final bar keeps its close in its first half (which, when the
/// deceptive first close ran, is vi — and vi of the home key IS the
/// relative key's tonic: the deception becomes the pivot), then its
/// second half becomes the NEW key's dominant — with the raised leading
/// tone where the new key is minor, the chromatic tone that tells the
/// ear something true is changing — so the departure's opening tonic
/// arrives as a RESOLUTION.
///
/// Deliberate asymmetry: only DEPARTURES (B/C) are modulated into.
/// Returns keep the staged-entrance treatment (the thinned bar, the
/// arrival) — you leave home by modulation, you come home by arrival.
/// The fugue keeps its documented no-modulation limit (its own pipeline).
fn apply_pivot_modulations(score: &mut Score, form: &Form, bars: i64, meter_beats: f64) {
    let section_bars = 2 * bars; // antecedent + consequent
    let meter_i = meter_beats as i64;
    for i in 1..form.sections.len() {
        let (prev, next) = (&form.sections[i - 1], &form.sections[i]);
        if next.key == prev.key {
            continue;
        }
        if !matches!(
            next.role,
            crate::form::SectionRole::B | crate::form::SectionRole::C
        ) {
            continue;
        }
        let target_bar = i as i64 * section_bars - 1;
        let bar_lo = target_bar as f64 * meter_beats;
        let win_lo = bar_lo + meter_beats / 2.0;
        let win_hi = bar_lo + meter_beats;
        let dominant = next.key.diatonic_triad(5);
        let dom_voiced = dominant.voice(3);
        let dom_pitches_melody = dominant.voice(5);
        let dom_root_bass = next.key.scale().degree_pitch(5, 2);
        let win_onset = Duration::new(meter_i * target_bar, 1) + Duration::new(meter_i, 2);
        let half = Duration::new(meter_i, 2);

        // THE ANNOUNCEMENT, THROUGH THE TEXTURE (v2 — v1 cleared the
        // window and pushed a block chord, interrupting the style's own
        // accompaniment for the half-bar; two texture tests needed
        // documented exemptions. Now the pattern KEEPS SPEAKING): every
        // harmony/bass event in the window is re-pitched in place —
        // rhythm, velocity, accent untouched — with boundary-crossing
        // notes split so their second halves join the announcement. The
        // FIRST harmony event in the window is forced to the new key's
        // leading tone (nearest octave), guaranteeing the chromatic
        // announcement regardless of how nearest-tone mapping falls out.
        let nearest_dom = |pitch: crate::pitch::Pitch, pool: &[crate::pitch::Pitch]| {
            pool.iter()
                .copied()
                .min_by_key(|p| (p.midi() as i32 - pitch.midi() as i32).abs())
                .unwrap()
        };
        let leading_pc = next.key.scale().degree_pitch_class(7);
        let dom_pool_low = dominant.voice(3);
        let mut splits: Vec<ScoreNote> = Vec::new();
        let mut window_touched = false;
        let mut leading_assigned = false;
        for j in 0..score.notes.len() {
            let n = &score.notes[j];
            let (on, off) = (n.onset.beats(), (n.onset + n.duration).beats());
            if !(on < win_hi - 1e-9 && off > win_lo + 1e-9) {
                continue;
            }
            match n.role {
                VoiceRole::Harmony | VoiceRole::Bass => {
                    window_touched = true;
                    let role = n.role;
                    if on < win_lo - 1e-9 {
                        // Crossing: keep the first half as-is, the second
                        // half joins the announcement.
                        let mut tail = *n;
                        let kept = ((win_lo - on) * 480.0).round() as i64;
                        let tail_len = ((off - win_lo) * 480.0).round() as i64;
                        score.notes[j].duration = Duration::new(kept, 480);
                        tail.onset = win_onset;
                        tail.duration = Duration::new(tail_len, 480);
                        tail.pitch = if role == VoiceRole::Bass {
                            dom_root_bass
                        } else if !leading_assigned {
                            leading_assigned = true;
                            nearest_leading(tail.pitch, leading_pc)
                        } else {
                            nearest_dom(tail.pitch, &dom_pool_low)
                        };
                        splits.push(tail);
                    } else {
                        let old_pitch = score.notes[j].pitch;
                        score.notes[j].pitch = if role == VoiceRole::Bass {
                            dom_root_bass
                        } else if !leading_assigned {
                            leading_assigned = true;
                            nearest_leading(old_pitch, leading_pc)
                        } else {
                            nearest_dom(old_pitch, &dom_pool_low)
                        };
                    }
                }
                VoiceRole::Melody => {
                    if on < win_lo - 1e-9 {
                        let mut tail = *n;
                        let kept = ((win_lo - on) * 480.0).round() as i64;
                        let tail_len = ((off - win_lo) * 480.0).round() as i64;
                        score.notes[j].duration = Duration::new(kept, 480);
                        tail.onset = win_onset;
                        tail.duration = Duration::new(tail_len, 480);
                        tail.pitch = nearest_dom(tail.pitch, &dom_pitches_melody);
                        tail.velocity *= 0.9;
                        tail.emphasis = Emphasis::Normal;
                        splits.push(tail);
                    } else {
                        let old_pitch = score.notes[j].pitch;
                        score.notes[j].pitch = nearest_dom(old_pitch, &dom_pitches_melody);
                    }
                }
                _ => {}
            }
        }
        for tail in splits {
            score.push(tail);
        }
        // Sparse-texture fallback: the announcement must sound even when
        // nothing was scheduled in the window.
        if !window_touched {
            for &pitch in &dom_voiced {
                score.push(ScoreNote {
                    pitch,
                    onset: win_onset,
                    duration: half,
                    velocity: 0.5,
                    role: VoiceRole::Harmony,
                    emphasis: Emphasis::Normal,
                    section_intensity: prev.role.intensity(),
                });
            }
            score.push(ScoreNote {
                pitch: dom_root_bass,
                onset: win_onset,
                duration: half,
                velocity: 0.55,
                role: VoiceRole::Bass,
                emphasis: Emphasis::Normal,
                section_intensity: prev.role.intensity(),
            });
        }
    }
}

/// The new key's leading tone in the octave nearest `near`.
fn nearest_leading(
    near: crate::pitch::Pitch,
    leading_pc: crate::pitch::PitchClass,
) -> crate::pitch::Pitch {
    (1..=7)
        .map(|oct| crate::pitch::Pitch::new(leading_pc, oct))
        .min_by_key(|p| (p.midi() as i32 - near.midi() as i32).abs())
        .unwrap()
}

/// The DECEPTIVE FIRST CLOSE — [`crate::cadence::Cadence::Deceptive`],
/// which existed in the vocabulary from the start but was never once
/// composed. The piece's first full cadence (the opening period's
/// consequent close) resolves V→vi instead of V→I: the bass steps up to
/// the submediant, the harmony re-voices to the vi triad, and the melody
/// — usually already on 1̂ or 3̂, both shared with vi, the classical trick
/// that makes the deception land smoothly — is snapped to a vi tone only
/// if it must be. Closure is promised, then denied; every later authentic
/// cadence, and above all the final one, now arrives as something EARNED.
/// The harmonic member of the expectation family: the erosion elegy's
/// false recovery is the same grammar acting on form.
///
/// Scope guards: only the FIRST period's final bar (the piece must still
/// close honestly — the deception works because the promise is
/// eventually kept), and the pass runs before intro-shift so bar
/// arithmetic is section-relative.
fn apply_deceptive_first_close(score: &mut Score, key: Key, bars: i64, meter_beats: f64) {
    // The opening period spans bars 0..2*bars; its consequent's final bar:
    let target_bar = 2 * bars - 1;
    let (lo, hi) = (
        target_bar as f64 * meter_beats,
        (target_bar + 1) as f64 * meter_beats,
    );
    let vi = key.diatonic_triad(6);
    let vi_pcs: Vec<crate::pitch::PitchClass> =
        vi.voice(3).iter().map(|p| p.pitch_class()).collect();
    let vi_voiced = vi.voice(3);
    let scale = key.scale();

    // Harmony: re-voice every chord tone in the bar to the vi triad,
    // preserving rhythm, velocity, and relative register order.
    let mut harmony_idx: Vec<usize> = (0..score.notes.len())
        .filter(|&i| {
            let n = &score.notes[i];
            n.role == VoiceRole::Harmony && n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi
        })
        .collect();
    harmony_idx.sort_by(|&a, &b| {
        score.notes[a]
            .pitch
            .midi()
            .cmp(&score.notes[b].pitch.midi())
    });
    let mut sorted_vi = vi_voiced.clone();
    sorted_vi.sort_by_key(|p| p.midi());
    for (k, &i) in harmony_idx.iter().enumerate() {
        score.notes[i].pitch = sorted_vi[k % sorted_vi.len()];
    }

    // Bass: the deceptive step — up to the submediant root.
    for i in 0..score.notes.len() {
        let n = &score.notes[i];
        if n.role == VoiceRole::Bass && n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi {
            score.notes[i].pitch = scale.degree_pitch(6, 2);
        }
    }

    // Melody: only bend tones that are NOT already vi chord tones (1̂ and
    // 3̂ are — the deception is smoothest when the tune doesn't move).
    for i in 0..score.notes.len() {
        let n = &score.notes[i];
        if n.role == VoiceRole::Melody
            && n.onset.beats() >= lo - 1e-9
            && n.onset.beats() < hi
            && !vi_pcs.contains(&n.pitch.pitch_class())
        {
            // Snap to the nearest vi chord tone in register.
            let target = vi
                .voice(5)
                .into_iter()
                .min_by_key(|p| (p.midi() as i32 - n.pitch.midi() as i32).abs())
                .unwrap();
            score.notes[i].pitch = target;
        }
    }
}

/// Apply the style's phrase rhetoric to every melodic cadence (and its
/// approach). Anchored on `Emphasis` markers, so it is position-independent
/// (intro shifts, codas, damage holes are all already in place).
///
/// - **Classic**: untouched — a strict no-op, pinned byte-identical by
///   test.
/// - **Declamatory** (statement — interruption — answer — sharp stop):
///   ~1.5 bars before each cadence the melody STOPS DEAD — the note
///   sounding there is truncated to compose a real silence (≥ half a
///   beat) before the answering approach; the cadence itself is cut short
///   (≤ 1 beat) and accented instead of held. Climax notes are exempt —
///   the interruption serves the peak, never swallows it.
/// - **Singing** (question — expansion — suspension — quiet arrival): the
///   tone before each cadence OVERSTAYS by half a beat (written-out
///   suspension timing — the resolution is delayed, not decorated) and
///   the arrival lands softly (velocity ×0.85) with its tail intact.
/// - **Martial** (statement — statement — strike): NO interruption, NO
///   silence — a march never stops. Each cadence is clipped short (≤ 1
///   beat) and hit harder than Declamatory's (×1.2), and the note landing
///   immediately before it gets its own accent (×1.1) — the pickup kick a
///   drum-major would give even with no kit playing.
fn apply_phrase_rhetoric(
    score: &mut Score,
    rhetoric: crate::spec::PhraseRhetoric,
    meter_beats: f64,
) {
    use crate::spec::PhraseRhetoric;
    if rhetoric == PhraseRhetoric::Classic {
        return;
    }
    let cadentials: Vec<usize> = {
        let mut idx: Vec<usize> = (0..score.notes.len())
            .filter(|&i| {
                score.notes[i].role == VoiceRole::Melody
                    && score.notes[i].emphasis == Emphasis::Cadential
            })
            .collect();
        idx.sort_by(|&a, &b| {
            score.notes[a]
                .onset
                .beats()
                .total_cmp(&score.notes[b].onset.beats())
        });
        idx
    };
    for &ci in &cadentials {
        let cad_onset = score.notes[ci].onset;
        match rhetoric {
            PhraseRhetoric::Classic => unreachable!(),
            PhraseRhetoric::Declamatory => {
                // 1. The interruption: find the melody note sounding at
                //    the stop point and truncate it so at least half a
                //    beat of genuine silence follows.
                let stop = cad_onset.beats() - 1.5 * meter_beats;
                if stop > 0.0 {
                    let target = (0..score.notes.len())
                        .filter(|&j| {
                            let n = &score.notes[j];
                            n.role == VoiceRole::Melody
                                && n.emphasis != Emphasis::Climax
                                && n.onset.beats() < stop - 0.25
                                && (n.onset + n.duration).beats() > stop
                        })
                        .max_by(|&a, &b| {
                            score.notes[a]
                                .onset
                                .beats()
                                .total_cmp(&score.notes[b].onset.beats())
                        });
                    if let Some(j) = target {
                        let n = &mut score.notes[j];
                        let kept = stop - 0.5 - n.onset.beats();
                        if kept >= 0.5 {
                            let num = (kept * 480.0).round() as i64;
                            n.duration = Duration::new(num, 480);
                        }
                    }
                }
                // 2. The sharp stop: short and accented, never held.
                let n = &mut score.notes[ci];
                if n.duration.beats() > 1.0 {
                    n.duration = Duration::new(1, 1);
                }
                n.velocity = (n.velocity * 1.12).clamp(0.0, 1.0);
            }
            PhraseRhetoric::Singing => {
                // 1. Written-out suspension timing: the previous melody
                //    tone overstays half a beat; the arrival waits.
                let prev = (0..score.notes.len())
                    .filter(|&j| {
                        let n = &score.notes[j];
                        n.role == VoiceRole::Melody
                            && ((n.onset + n.duration).beats() - cad_onset.beats()).abs() < 1e-6
                    })
                    .max_by(|&a, &b| {
                        score.notes[a]
                            .onset
                            .beats()
                            .total_cmp(&score.notes[b].onset.beats())
                    });
                if let Some(j) = prev
                    && score.notes[ci].duration.beats() > 1.0
                {
                    let half = Duration::new(1, 2);
                    score.notes[j].duration = score.notes[j].duration + half;
                    let n = &mut score.notes[ci];
                    n.onset = n.onset + half;
                    n.duration = n.duration.saturating_sub(half);
                }
                // 2. The quiet arrival.
                let n = &mut score.notes[ci];
                n.velocity = (n.velocity * 0.85).clamp(0.0, 1.0);
            }
            PhraseRhetoric::Martial => {
                // 1. The pickup accent: the note landing exactly where the
                //    cadence note begins gets its own strike — found by
                //    matching the note whose END touches the cadence onset.
                if let Some(j) = (0..score.notes.len())
                    .filter(|&j| {
                        let n = &score.notes[j];
                        n.role == VoiceRole::Melody
                            && ((n.onset + n.duration).beats() - cad_onset.beats()).abs() < 1e-6
                    })
                    .max_by(|&a, &b| {
                        score.notes[a]
                            .onset
                            .beats()
                            .total_cmp(&score.notes[b].onset.beats())
                    })
                {
                    score.notes[j].velocity = (score.notes[j].velocity * 1.1).clamp(0.0, 1.0);
                }
                // 2. The strike: clipped short and hit harder than a
                //    Declamatory stop — but never preceded by silence.
                let n = &mut score.notes[ci];
                if n.duration.beats() > 1.0 {
                    n.duration = Duration::new(1, 1);
                }
                n.velocity = (n.velocity * 1.2).clamp(0.0, 1.0);
            }
            PhraseRhetoric::Chorale => {
                // The fermata: hold the cadential note at least a full
                // extra beat past its written value -- never shortens it,
                // unlike every other non-Classic rhetoric. No timing
                // shift, no suspension: arrives exactly when written.
                let n = &mut score.notes[ci];
                let held = n.duration.beats() + 1.0;
                n.duration = Duration::new((held * 480.0).round() as i64, 480);
                // A steady, slightly firm dynamic -- not softened
                // (Singing) and not sharply struck (Declamatory/Martial).
                n.velocity = (n.velocity * 1.05).clamp(0.0, 1.0);
            }
        }
    }
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
            let gap_start = (a.onset + a.duration).beats();
            let gap_end = b.onset.beats();
            // `melody` only lists Normal-emphasis notes in the window, so a
            // Cadential/Climax-emphasis melody note can sit chronologically
            // between `a` and `b` without ever appearing as a pair here —
            // but the passing tone below is placed purely from `a`/`b`'s
            // own onsets, with no awareness of anything else. Require the
            // gap to be genuinely empty of ANY other melody note before
            // treating it as "room to steal a sixteenth from".
            let gap_is_clear = !score.notes.iter().enumerate().any(|(idx, n)| {
                idx != w[0]
                    && idx != w[1]
                    && n.role == VoiceRole::Melody
                    && n.onset.beats() < gap_end - 1e-9
                    && (n.onset + n.duration).beats() > gap_start + 1e-9
            });
            // A whole-step move with room to steal a sixteenth from.
            (step.abs() == 2
                && gap_end - gap_start >= -1e-6
                && a.duration.beats() >= 0.5
                && gap_is_clear)
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
    // CounterShift moves notes later without checking what that collides
    // with: the forced 0.5-beat floor (a few lines up) can LENGTHEN a short
    // note past where the next counter-melody note (shifted or not) begins.
    // De-overlap defensively rather than trying to reason about every
    // shift/floor interaction inline.
    deoverlap_voice(score, VoiceRole::CounterMelody);

    // Keep the melody line onset-ordered for downstream consumers.
    let _ = key; // key reserved for future harmonic damage devices
    score
        .notes
        .sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
}

/// De-overlap one voice's own notes: walking them in onset order, truncate
/// any note whose end now runs past the very next same-voice note's onset
/// to end EXACTLY there (exact rational arithmetic — no rounding, so this
/// can never overshoot back into an overlap). A note with no usable room at
/// all (an exact-duplicate onset, or a negative gap) is dropped entirely
/// rather than forced to a degenerate/overlapping duration. A defensive
/// pass for devices that shift/shorten/insert notes in ways too entangled
/// to reason about neighbor-by-neighbor inline — see
/// [`deoverlap_all_voices`] for why this is applied universally rather than
/// chasing each device's own root cause.
pub(crate) fn deoverlap_voice(score: &mut Score, role: VoiceRole) {
    let mut idxs: Vec<usize> = score
        .notes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.role == role)
        .map(|(i, _)| i)
        .collect();
    idxs.sort_by(|&a, &b| {
        score.notes[a]
            .onset
            .beats()
            .total_cmp(&score.notes[b].onset.beats())
    });
    let mut to_remove = Vec::new();
    for w in idxs.windows(2) {
        let (a, b) = (w[0], w[1]);
        let a_end = (score.notes[a].onset + score.notes[a].duration).beats();
        let b_onset_beats = score.notes[b].onset.beats();
        if a_end > b_onset_beats + 1e-9 {
            let gap = score.notes[b].onset.saturating_sub(score.notes[a].onset);
            if gap.beats() <= 1e-9 {
                to_remove.push(a);
            } else {
                score.notes[a].duration = gap;
            }
        }
    }
    to_remove.sort_unstable();
    to_remove.dedup();
    for i in to_remove.into_iter().rev() {
        score.notes.remove(i);
    }
}

/// Universal safety net: de-overlap EVERY voice, applied once at the very
/// end of composition regardless of which grammar family/engine produced
/// the score. Multiple independent devices across this crate (ornaments,
/// attitude devices, damage devices, and each dedicated grammar engine)
/// mutate or insert notes without cross-checking every other device that
/// might have touched the same window — chasing each one's root cause
/// individually is open-ended, while guaranteeing the invariant holds on
/// the way out is not. See `debug_assert_no_structural_defects`'s call
/// sites for where this runs.
pub(crate) fn deoverlap_all_voices(score: &mut Score) {
    // Matches `score_validation::validate_voice_monophony`'s own role set
    // exactly: `Harmony` is deliberately excluded there (and here) because
    // it's legitimately POLYPHONIC — a chord is multiple simultaneous
    // Harmony-role notes by design, not an overlap bug.
    for role in [VoiceRole::Melody, VoiceRole::Bass, VoiceRole::CounterMelody] {
        deoverlap_voice(score, role);
    }
}

/// Keep `Score::total_beats` honest after any pass that lengthens a note by
/// mutating its `duration` field directly (bypassing `Score::push`, which
/// is the only place that normally extends `total_beats`). Several
/// attitude/damage devices do exactly this (e.g. `alter_statement`'s
/// Curiosity case extends the second-to-last note), which silently let a
/// note's end run past the score's own declared length — caught by
/// `NoteBounds`' `end <= total_beats` check once the debug validation gate
/// was wired in.
pub(crate) fn sync_total_beats(score: &mut Score) {
    if let Some(max_end) = score
        .notes
        .iter()
        .map(|n| n.onset + n.duration)
        .max_by(|a, b| a.beats().total_cmp(&b.beats()))
        && max_end.beats() > score.total_beats.beats()
    {
        score.total_beats = max_end;
    }
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
    // The echo's total span (constant across every hold — it's the same
    // hook cell every time), needed to clear its own window in the
    // counter-melody voice before answering into it (see the loop below).
    let echo_span: Duration = hook
        .notes
        .iter()
        .fold(Duration::zero(), |acc, &(_, dur)| acc + dur.scale(1, 2));
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
            let echo_start = hold_onset + 0.5;
            let echo_end = echo_start + echo_span.beats();
            // Clear the counter-melody's own window before answering into
            // it (mirrors `apply_hidden_bass_quote`'s overlap-aware
            // clearing) — without this, whatever the base composition
            // already placed here (e.g. a hidden-bass-quote-style
            // counterline, or a prior echo) collides with the new hook
            // notes: exact-duplicate onsets/pitches or straight overlap.
            score.notes.retain(|n| {
                !(n.role == VoiceRole::CounterMelody
                    && n.onset.beats() >= echo_start - 1e-9
                    && n.onset.beats() < echo_end - 1e-9)
            });
            for n in score.notes.iter_mut() {
                if n.role == VoiceRole::CounterMelody
                    && n.onset.beats() < echo_start - 1e-9
                    && (n.onset + n.duration).beats() > echo_start + 1e-9
                {
                    let raw = ((echo_start - n.onset.beats()) * 4.0).round() as i64;
                    n.duration = Duration::new(raw.max(1), 4);
                }
            }
            let mut t = echo_start;
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
///
/// `pub` (not `pub(crate)`): `muse_studio.rs`'s Motifs endpoint
/// (`symthaea-muse`, a different crate) reuses this exact geometry rather
/// than recomputing it — see that endpoint's doc comment.
pub struct SectionBars {
    pub role: SectionRole,
    pub start_bar: i64,
    pub antecedent_bars: i64,
    pub consequent_bars: i64,
}

impl SectionBars {
    pub fn end_bar(&self) -> i64 {
        self.start_bar + self.antecedent_bars + self.consequent_bars
    }
}

pub fn section_bar_map(form: &Form) -> Vec<SectionBars> {
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
    seventh_chords: bool,
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
        } else if deg == dom || seventh_chords {
            // JAZZ BALLAD's habit: `seventh_chords` extends the existing
            // dominant-only 7th coloring (above) to EVERY chord, not just
            // the cadential one — the lush maj7/min7/dom7 vocabulary
            // that's jazz harmony's actual identity, not an occasional
            // color tone. Still the same safety property: a seventh is a
            // strict superset of its triad, so the melody's triad-based
            // chord-tone snapping never clashes.
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
        // the genre's own rhythm). The montuno is excluded too: its onset
        // table is computed relative to the FULL bar via `clave_side`, and
        // halving `meter_beats` for a split would silently recompute a
        // different (and wrong) clave side for the second half — breaking
        // the tumbao bass's non-collision guarantee. The compás-gait is
        // excluded for the same family of reason: its 5 fixed onsets
        // (2,5,7,9,11) are computed against the FULL 12-beat bar, and a
        // halved local meter would drop or mis-locate every onset past 6.
        // Bossa comp is excluded too: its durations are chained to end
        // EXACTLY where the next one begins (0..1.5..3.0..4.0); a split
        // would truncate the 1.5-beat stab's tail into the second half's
        // own onset 0, overlapping two different (and differently
        // colored, triad vs 7th) chords at once.
        let split_cadential = cadential_split
            && meter_beats >= 4.0
            && applied.is_none()
            && pattern != crate::accompaniment::Accompaniment::Montuno
            && pattern != crate::accompaniment::Accompaniment::CompasGait
            && pattern != crate::accompaniment::Accompaniment::BossaComp
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
    seventh_chords: bool,
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
            seventh_chords,
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
    // Under the habanera the bass is the cell's anchor: one dotted-quarter
    // gravity note per bar; walking or half-bar splits would fight the
    // accent hierarchy the cell exists to create.
    let habanera = pattern == crate::accompaniment::Accompaniment::Habanera;
    // The five-gait's bass IS the 3+2 split: a three-beat anchor and a
    // two-beat answer; walking would blur the grouping the cell exists
    // to spell.
    let five_gait = pattern == crate::accompaniment::Accompaniment::FiveGait;
    // The montuno's bass is a TUMBAO: onsets deliberately chosen to fall
    // between the montuno's own onsets on either clave side, never on top
    // of them — the "rhythmic conversation" made into interlocking timing
    // rather than a shared downbeat. See the onset tables below.
    let montuno = pattern == crate::accompaniment::Accompaniment::Montuno;
    // The compás-gait's bass is a single grounding anchor at the top of
    // the 12-beat cycle — every OTHER voice-leading device in this
    // function assumes a much shorter bar and would just add noise to a
    // pattern whose whole identity is "mostly silence, five hits."
    let compas_gait = pattern == crate::accompaniment::Accompaniment::CompasGait;
    // Bossa bass: root through most of the bar, then a soft re-attack
    // right at the bar's edge — the "anticipation" push real bossa bass
    // is known for, kept inside the bar (no cross-bar sustain) to avoid
    // the overlap class of bug the montuno/compás waves already found.
    let bossa_comp = pattern == crate::accompaniment::Accompaniment::BossaComp;
    let walking = intent.arousal > 0.5
        && !oom_pah
        && !habanera
        && !five_gait
        && !montuno
        && !compas_gait
        && !bossa_comp;
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
            && !habanera
            && !five_gait
            && !montuno
            && !compas_gait
            && !bossa_comp
            && measure_degrees.get(i + 1) == Some(&5)
            && applied_dominant_target(key, measure_degrees, i, intent.seed).is_none();
        if bossa_comp {
            let quarter = Duration::quarter();
            let hold = bar.saturating_sub(quarter);
            score.push(ScoreNote {
                pitch: bass,
                onset: bar_onset,
                duration: if meter_beats > 1.0 { hold } else { bar },
                velocity: vel * 0.8, // understated, matching the harmony's soft ceiling
                role: VoiceRole::Bass,
                emphasis: Emphasis::Normal,
                section_intensity,
            });
            if meter_beats > 1.0 {
                // The anticipation push, right at the bar's edge — never
                // crossing into the next bar (the montuno/compás waves'
                // lesson: cross-bar sustain risks colliding with the next
                // bar's own freshly-voiced chord).
                score.push(ScoreNote {
                    pitch: bass,
                    onset: bar_onset + hold,
                    duration: quarter,
                    velocity: vel * 0.7,
                    role: VoiceRole::Bass,
                    emphasis: Emphasis::Normal,
                    section_intensity,
                });
            }
            *prev_bass = Some(bass);
            continue;
        }
        if compas_gait {
            // A single anchor at the cycle's opening, dotted to span up to
            // the harmony's first stab (beat 2) — the same "gravity note"
            // shape Habanera's bass established, at a new cycle length.
            let anchor_dur = if meter_beats >= 2.0 {
                Duration::new(2, 1)
            } else {
                bar
            };
            score.push(ScoreNote {
                pitch: bass,
                onset: bar_onset,
                duration: anchor_dur,
                velocity: vel,
                role: VoiceRole::Bass,
                emphasis: Emphasis::Normal,
                section_intensity,
            });
            *prev_bass = Some(bass);
            continue;
        }
        if montuno {
            // Tumbao: root at the "call" onsets, the fifth at the final
            // "anticipation" onset — echoing Shuffle's root/fifth habit.
            // Onsets are chosen to clear every onset `Accompaniment::
            // Montuno` uses on the matching clave side (0/1.5/3.0 on the
            // three-side; 1.0/2.0 on the two-side) — verified in
            // `montuno_bass_never_shares_an_onset_with_the_montuno_stabs`.
            let fifth = chord.voice(bass_octave)[2];
            let eighth = Duration::eighth();
            let onsets: &[(f64, bool)] =
                match crate::accompaniment::clave_side(bar_onset, meter_beats) {
                    crate::accompaniment::ClaveSide::Three => {
                        &[(0.75, false), (2.25, false), (3.5, true)]
                    }
                    crate::accompaniment::ClaveSide::Two => {
                        &[(0.0, false), (2.5, false), (3.5, true)]
                    }
                };
            for &(beat, is_fifth) in onsets {
                if beat >= meter_beats - 1e-9 {
                    continue;
                }
                score.push(ScoreNote {
                    pitch: if is_fifth { fifth } else { bass },
                    onset: bar_onset + Duration::new((beat * 2.0).round() as i64, 2),
                    duration: eighth,
                    velocity: if is_fifth { vel * 0.9 } else { vel },
                    role: VoiceRole::Bass,
                    emphasis: Emphasis::Normal,
                    section_intensity,
                });
            }
            *prev_bass = Some(bass);
            continue;
        }
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
            duration: if five_gait {
                Duration::new(3, 1) // the THREE group, whole
            } else if habanera {
                Duration::new(3, 2) // the cell's dotted anchor
            } else if oom_pah {
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
        } else if five_gait {
            // The TWO group's anchor: the root again at beat 3, two beats.
            score.push(ScoreNote {
                pitch: bass,
                onset: bar_onset + Duration::new(3, 1),
                duration: Duration::new(2, 1),
                velocity: vel * 0.9,
                role: VoiceRole::Bass,
                emphasis: Emphasis::Normal,
                section_intensity,
            });
            *prev_bass = Some(bass);
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

    /// Regression test for the counter-melody overlap bugs fixed in
    /// `apply_counter_hook_echoes` and `DamageDevice::CounterShift`: a
    /// 300-seed sweep of `Style::Classical`'s default composition — the
    /// exact reproduction that originally found 750 Fatal `VoiceMonophony`
    /// violations — found in the muse improvement-roadmap audit. This test
    /// doesn't need to call `validate_score` itself: `compose_styled` routes
    /// through `compose_with_grammar_plan`, which now calls
    /// `debug_assert_no_structural_defects` on every composed score, so any
    /// regression of this bug class panics right here under `cargo test`
    /// (debug_assertions are on). Also sweeps every other style at a handful
    /// of seeds so the gate isn't only exercised for Classical.
    #[test]
    fn no_style_produces_a_structurally_defective_score_across_many_seeds() {
        for seed in 0..300u64 {
            let intent = MusicalIntent {
                seed,
                ..MusicalIntent::default()
            };
            let _ = compose_styled(&intent, crate::style::Style::Classical);
        }
        for style in crate::style::Style::ALL {
            for seed in 0..8u64 {
                let intent = MusicalIntent {
                    seed,
                    ..MusicalIntent::default()
                };
                let _ = compose_styled(&intent, style);
            }
        }
    }

    #[test]
    fn use_sentence_for_period_sentence_keeps_the_arousal_heuristic() {
        let profile = crate::grammar::GrammarFamily::PeriodSentence.profile();
        let calm = MusicalIntent {
            arousal: 0.1,
            energy: 0.1,
            ..MusicalIntent::default()
        };
        let excited = MusicalIntent {
            arousal: 0.9,
            energy: 0.9,
            ..MusicalIntent::default()
        };
        assert!(!use_sentence_for(Some(profile), &calm));
        assert!(use_sentence_for(Some(profile), &excited));
        // Ungraded (`None`) matches the exact same heuristic.
        assert!(!use_sentence_for(None, &calm));
        assert!(use_sentence_for(None, &excited));
    }

    #[test]
    fn use_sentence_for_other_families_overrides_the_arousal_heuristic() {
        // A sentence-bucketed family stays sentence even at LOW arousal.
        let jazz = crate::grammar::GrammarFamily::JazzChorus.profile();
        let calm = MusicalIntent {
            arousal: 0.1,
            energy: 0.1,
            ..MusicalIntent::default()
        };
        assert!(use_sentence_for(Some(jazz), &calm));

        // A period-bucketed family stays period even at HIGH arousal.
        let strophic = crate::grammar::GrammarFamily::StrophicSong.profile();
        let excited = MusicalIntent {
            arousal: 0.9,
            energy: 0.9,
            ..MusicalIntent::default()
        };
        assert!(!use_sentence_for(Some(strophic), &excited));
    }

    #[test]
    fn grammar_progression_styles_are_unaffected_by_the_grammar_aware_pipeline() {
        // No-regression contract, narrowed to what's ACTUALLY invariant:
        // a style produces byte-identical output through both entry points
        // ONLY when its own `progression: ProgressionSpec::Grammar` (i.e.
        // `spec.progression(...)` reduces to the exact same
        // `Progression::generate(...)` call the plain 2-arg path already
        // used for Ternary/Rondo's B/C sections). Styles with their own
        // `Archetype`/`ArchetypePool` progression are EXPECTED to diverge
        // now (see `use_sentence_for_other_families_overrides_...` and the
        // `Form::ternary`/`rondo` grammar-context tests in `form.rs`) --
        // their B/C sections correctly start honoring the style's own
        // harmonic vocabulary instead of borrowing Classical's. Classical
        // is currently the ONLY style using `ProgressionSpec::Grammar`.
        // Same enumeration `style.rs`'s own
        // `grammar_family_covers_every_style_exactly_once` test uses — this
        // crate has no shared `Style::ALL` const.
        let all_styles = [
            crate::style::Style::Classical,
            crate::style::Style::Waltz,
            crate::style::Style::Folk,
            crate::style::Style::Cinematic,
            crate::style::Style::Playful,
            crate::style::Style::Nocturne,
            crate::style::Style::March,
            crate::style::Style::Lullaby,
            crate::style::Style::ModalFolk,
            crate::style::Style::Fugue,
            crate::style::Style::Passacaglia,
            crate::style::Style::Tango,
            crate::style::Style::Celtic,
            crate::style::Style::Blues,
            crate::style::Style::Impressionism,
            crate::style::Style::SacredChoral,
            crate::style::Style::Minimalism,
            crate::style::Style::JazzBallad,
            crate::style::Style::BaroqueSuite,
            crate::style::Style::ProgFolk,
            crate::style::Style::Ambient,
            crate::style::Style::Sonata,
            crate::style::Style::RenaissancePolyphony,
            crate::style::Style::AfroCuban,
            crate::style::Style::Flamenco,
            crate::style::Style::BossaNova,
            crate::style::Style::Opera,
            crate::style::Style::IrishTraditional,
            crate::style::Style::HindustaniInspired,
        ];
        let grammar_progression_styles: Vec<crate::style::Style> = all_styles
            .into_iter()
            .filter(|s| s.grammar_family() == crate::grammar::GrammarFamily::PeriodSentence)
            .filter(|s| s.spec().progression == crate::spec::ProgressionSpec::Grammar)
            .collect();
        assert!(
            !grammar_progression_styles.is_empty(),
            "expected at least one PeriodSentence-mapped style using ProgressionSpec::Grammar"
        );
        let profile = crate::grammar::GrammarFamily::PeriodSentence.profile();
        for style in grammar_progression_styles {
            let spec = style.spec();
            for (arousal, energy, seed) in [(0.2, 0.2, 1u64), (0.8, 0.8, 2u64), (0.5, 0.9, 3u64)] {
                let intent = MusicalIntent {
                    arousal,
                    energy,
                    seed,
                    ..MusicalIntent::default()
                };
                let plain = compose_with_spec_and_form(&intent, &spec);
                let graded = compose_with_spec_and_form_and_grammar(&intent, &spec, Some(profile));
                assert_eq!(
                    plain, graded,
                    "style {style:?} diverged for arousal={arousal} energy={energy} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn archetype_progression_period_sentence_styles_do_diverge_through_the_grammar_pipeline() {
        // The flip side of the above: Waltz is PeriodSentence-mapped but
        // uses `ProgressionSpec::Archetype`, not `Grammar` -- its B section
        // in Ternary/Rondo form should now honor Waltz's own progression
        // instead of the generic classical grammar, end-to-end through the
        // real `compose_with_spec_and_form_and_grammar` production path
        // (not just the `Form::ternary`/`rondo` unit tests in `form.rs`).
        let spec = crate::style::Style::Waltz.spec();
        assert_eq!(
            spec.progression,
            crate::spec::ProgressionSpec::Archetype(vec![1, 5, 6, 3, 4, 1, 4, 5])
        );
        let profile = crate::grammar::GrammarFamily::PeriodSentence.profile();
        let diverged = (0..20u64).any(|seed| {
            let intent = MusicalIntent {
                seed,
                ..MusicalIntent::default()
            };
            let plain = compose_with_spec_and_form(&intent, &spec);
            let graded = compose_with_spec_and_form_and_grammar(&intent, &spec, Some(profile));
            plain != graded
        });
        assert!(
            diverged,
            "expected Waltz's grammar-aware output to differ from the plain path \
             for at least one seed among 0..20 (Ternary/Rondo B-section progression fix)"
        );
    }

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
            false,
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
            let hook =
                crate::hook::HookCell::generate_with(&spec_on.melody, seed, spec_on.meter as f64);
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
        let intent = MusicalIntent::default();
        let hook =
            crate::hook::HookCell::generate_with(&spec.melody, intent.seed, spec.meter as f64);
        let meter_beats = spec.meter as f64;
        // The quote: a run of bass notes whose durations are exactly the
        // hook's, AUGMENTED (×2), in sequence, specifically at the position
        // `apply_hidden_bass_quote` targets (mid-departure, same formula as
        // that function's own `quote_start`/`span`) — anchoring to the real
        // insertion point, not just "somewhere in the whole bass line",
        // since Classical's default walking-bass rhythm can otherwise
        // coincidentally repeat a short/generic hook's duration pattern
        // (e.g. half-half-whole) elsewhere in the piece.
        let quote_window = |form: &Form| -> Option<(f64, f64)> {
            let (start_bar, bars) = form
                .sections
                .iter()
                .zip(section_bar_map(form))
                .find(|(sec, _)| sec.role == SectionRole::B)
                .map(|(_, b)| (b.start_bar, b.end_bar() - b.start_bar))?;
            let quote_start = (start_bar + bars / 2) as f64 * meter_beats;
            Some((quote_start, 2.0 * hook.beats()))
        };
        let has_quote_in_window = |s: &Score, form: &Form| -> bool {
            let Some((quote_start, span)) = quote_window(form) else {
                return false;
            };
            let target: Vec<f64> = hook.notes.iter().map(|(_, d)| d.beats() * 2.0).collect();
            let bass: Vec<f64> = s
                .notes
                .iter()
                .filter(|n| {
                    n.role == VoiceRole::Bass
                        && n.onset.beats() >= quote_start - 1e-9
                        && n.onset.beats() < quote_start + span - 1e-9
                })
                .map(|n| n.duration.beats())
                .collect();
            bass.len() == target.len()
                && bass.iter().zip(&target).all(|(a, b)| (a - b).abs() < 1e-6)
        };
        let (score, form) = compose_with_spec_and_form(&intent, &spec);
        let form = form.expect("Ternary/Rondo forms always produce a Form");
        assert!(
            has_quote_in_window(&score, &form),
            "the departure's bass must speak the name, augmented, at the mid-departure position"
        );
        // No negative ("hook_cell off => no quote there") assertion here:
        // verified (2026-07-23) that for Style::Classical/seed 0 specifically,
        // the style's ordinary walking-bass rhythm in that exact window is
        // ALSO shaped [2.0, 2.0, 4.0] even with hook_cell off — the ×2
        // augmentation of this hook's short [half, half, whole] cell isn't
        // distinctive from Classical's default bass rhythm for this seed, so
        // a duration-only check can't discriminate "deliberate quote" from
        // "coincidence" here, and the composed pitches diverge from the
        // deliberately-inserted ones past the first note in a way not yet
        // root-caused (worth its own investigation — possibly a later
        // voice-leading pass altering notes it doesn't recognize as the
        // quote). Flagging rather than asserting an unverified claim either
        // way.
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
    fn departures_are_announced_by_the_new_keys_dominant() {
        // The engine's first REAL modulation: the half-bar before a
        // departure carries the NEW key's dominant — including, for a
        // major->relative-minor move, the raised leading tone that exists
        // in NEITHER home collection nor plain A-minor-as-relative degree
        // math. C major -> A minor: the pivot dominant is E major, whose
        // G# (pc 8) is chromatic to C major — the audible proof the key
        // actually travels.
        let spec = crate::style::Style::Classical.spec();
        let intent = MusicalIntent {
            valence: 0.5, // major home key
            ..Default::default()
        };
        let s = compose_with_spec(&intent, &spec);
        let key = s.key;
        let b_key = key.relative();
        let dom = b_key.diatonic_triad(5);
        let dom_pcs: Vec<_> = dom.voice(3).iter().map(|p| p.pitch_class()).collect();
        let bars = intent.bars.max(1) as i64;
        let intro = spec.texture.intro_bars as i64;
        let target_bar = intro + 2 * bars - 1;
        let (wlo, whi) = (target_bar as f64 * 4.0 + 2.0, (target_bar + 1) as f64 * 4.0);
        let in_win = |n: &ScoreNote| n.onset.beats() >= wlo - 1e-6 && n.onset.beats() < whi;
        let win_notes: Vec<_> = s
            .notes
            .iter()
            .filter(|n| {
                in_win(n)
                    && matches!(
                        n.role,
                        VoiceRole::Harmony | VoiceRole::Bass | VoiceRole::Melody
                    )
            })
            .collect();
        assert!(!win_notes.is_empty(), "the announcement window is empty");
        for n in &win_notes {
            assert!(
                dom_pcs.contains(&n.pitch.pitch_class()),
                "{:?} tone {:?} outside the pivot dominant",
                n.role,
                n.pitch
            );
        }
        // The chromatic proof: the raised leading tone of the new key is
        // present and does NOT belong to the home collection.
        let leading = b_key.scale().degree_pitch_class(7);
        assert!(
            win_notes.iter().any(|n| n.pitch.pitch_class() == leading),
            "no raised leading tone — not a real modulation"
        );
        assert!(
            !key.scale().contains(leading),
            "test premise: the leading tone must be chromatic to home"
        );
        // And the bass announces the new dominant root.
        assert!(win_notes.iter().any(|n| n.role == VoiceRole::Bass
            && n.pitch.pitch_class() == b_key.scale().degree_pitch_class(5)));
    }

    #[test]
    fn development_dna_gives_each_style_its_own_departure() {
        // The last shared language, split three ways. Measure the B
        // section's ANTECEDENT melody (the developed span) per style.
        let intent = MusicalIntent::default();
        let bars = intent.bars.max(1) as i64;
        let span = |spec: &crate::spec::CompositionSpec| -> Vec<ScoreNote> {
            let s = compose_with_spec(&intent, spec);
            let intro = spec.texture.intro_bars as i64;
            let m = spec.meter as f64;
            let (lo, hi) = ((intro + 2 * bars) as f64 * m, (intro + 3 * bars) as f64 * m);
            let mut v: Vec<ScoreNote> = s
                .notes
                .iter()
                .copied()
                .filter(|n| {
                    n.role == VoiceRole::Melody
                        && n.onset.beats() >= lo - 1e-9
                        && n.onset.beats() < hi - 1e-9
                })
                .collect();
            v.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
            v
        };
        // SEQUENTIAL (Tango, seed 0 = descending): each bar repeats the
        // theme's rhythm, and the mean pitch walks monotonically in one
        // direction across the bars — the audible schema.
        let tango = crate::style::Style::Tango.spec();
        assert_eq!(tango.development, crate::spec::DevelopmentDna::Sequential);
        let seq = span(&tango);
        let m = tango.meter as f64;
        let bar_of = |n: &ScoreNote, lo_bar: f64| ((n.onset.beats() - lo_bar) / m) as i64;
        let lo_bar = seq
            .first()
            .map(|n| (n.onset.beats() / m).floor() * m)
            .unwrap();
        let mut bar_means: Vec<f64> = Vec::new();
        for b in 0..bars {
            let notes: Vec<&ScoreNote> = seq.iter().filter(|n| bar_of(n, lo_bar) == b).collect();
            assert!(!notes.is_empty(), "sequential bar {b} empty");
            bar_means.push(
                notes.iter().map(|n| n.pitch.midi() as f64).sum::<f64>() / notes.len() as f64,
            );
        }
        // The species fitter bends individual tones against the moving
        // bass, so a FITTED sequence wiggles — the honest schema claims
        // are direction of travel and a majority of steps agreeing.
        let travel = bar_means[bars as usize - 1] - bar_means[0];
        assert!(
            travel.abs() > 1.0,
            "the sequence must actually travel: {bar_means:?}"
        );
        let agreeing = bar_means
            .windows(2)
            .filter(|w| (w[1] - w[0]).signum() == travel.signum())
            .count();
        assert!(
            agreeing * 2 > (bars as usize - 1),
            "most steps must walk the travel direction: {bar_means:?}"
        );
        // FIGURAL (Nocturne): later bars carry MORE tones than the first
        // (figuration accumulates).
        let nocturne = crate::style::Style::Nocturne.spec();
        assert_eq!(nocturne.development, crate::spec::DevelopmentDna::Figural);
        let fig = span(&nocturne);
        let nm = nocturne.meter as f64;
        let flo = fig
            .first()
            .map(|n| (n.onset.beats() / nm).floor() * nm)
            .unwrap();
        let count = |b: i64| {
            fig.iter()
                .filter(|n| ((n.onset.beats() - flo) / nm) as i64 == b)
                .count()
        };
        assert!(
            count(2) > count(0),
            "figuration must accumulate: bar0={} bar2={}",
            count(0),
            count(2)
        );
        // FRAGMENTING (March): later bars carry FEWER tones (the head
        // shrinks; silence grows).
        let march = crate::style::Style::March.spec();
        assert_eq!(march.development, crate::spec::DevelopmentDna::Fragmenting);
        let fr = span(&march);
        let mm = march.meter as f64;
        let rlo = fr
            .first()
            .map(|n| (n.onset.beats() / mm).floor() * mm)
            .unwrap();
        let rcount = |b: i64| {
            fr.iter()
                .filter(|n| ((n.onset.beats() - rlo) / mm) as i64 == b)
                .count()
        };
        assert!(
            rcount(bars - 1) < rcount(0),
            "fragmentation must shrink: bar0={} last={}",
            rcount(0),
            rcount(bars - 1)
        );
        // INTENSIFYING (Cinematic): a real dramatic arc — register climbs
        // monotonically (unlike Sequential's could-go-either-way-by-seed)
        // AND velocity itself crescendos, unlike every other DNA which
        // never touches loudness across the span.
        let cinematic = crate::style::Style::Cinematic.spec();
        assert_eq!(
            cinematic.development,
            crate::spec::DevelopmentDna::Intensifying
        );
        let int_span = span(&cinematic);
        let im = cinematic.meter as f64;
        let ilo = int_span
            .first()
            .map(|n| (n.onset.beats() / im).floor() * im)
            .unwrap();
        let bar_of_int = |n: &ScoreNote| ((n.onset.beats() - ilo) / im) as i64;
        let mean_pitch = |b: i64| {
            let notes: Vec<&ScoreNote> = int_span.iter().filter(|n| bar_of_int(n) == b).collect();
            assert!(!notes.is_empty(), "intensifying bar {b} empty");
            notes.iter().map(|n| n.pitch.midi() as f64).sum::<f64>() / notes.len() as f64
        };
        let mean_vel = |b: i64| {
            let notes: Vec<&ScoreNote> = int_span.iter().filter(|n| bar_of_int(n) == b).collect();
            notes.iter().map(|n| n.velocity as f64).sum::<f64>() / notes.len() as f64
        };
        assert!(
            mean_pitch(bars - 1) > mean_pitch(0),
            "register must climb toward the last bar: bar0={} last={}",
            mean_pitch(0),
            mean_pitch(bars - 1)
        );
        assert!(
            mean_vel(bars - 1) > mean_vel(0),
            "velocity must crescendo toward the last bar: bar0={} last={}",
            mean_vel(0),
            mean_vel(bars - 1)
        );
        // WANDERING (Folk): a genuine random WALK, not a directed one —
        // the schema must be able to reverse direction mid-passage,
        // unlike Sequential which commits to one direction for the
        // whole span.
        let folk = crate::style::Style::Folk.spec();
        assert_eq!(folk.development, crate::spec::DevelopmentDna::Wandering);
        let wan = span(&folk);
        let wm = folk.meter as f64;
        let wlo = wan
            .first()
            .map(|n| (n.onset.beats() / wm).floor() * wm)
            .unwrap();
        let bar_of_wan = |n: &ScoreNote| ((n.onset.beats() - wlo) / wm) as i64;
        let wan_mean_pitch = |b: i64| {
            let notes: Vec<&ScoreNote> = wan.iter().filter(|n| bar_of_wan(n) == b).collect();
            assert!(!notes.is_empty(), "wandering bar {b} empty");
            notes.iter().map(|n| n.pitch.midi() as f64).sum::<f64>() / notes.len() as f64
        };
        let wan_means: Vec<f64> = (0..bars).map(wan_mean_pitch).collect();
        let deltas: Vec<f64> = wan_means.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(
            deltas.iter().any(|d| *d > 0.0) && deltas.iter().any(|d| *d < 0.0),
            "a real walk must move in both directions across the span: {wan_means:?}"
        );
    }

    #[test]
    fn the_return_close_evades_and_the_coda_arrives() {
        // The third expectation device: the return section's own close
        // lands on first-inversion tonic (bass on the mediant) with the
        // melody kept off the tonic; the coda's final tones still land
        // home — the one true arrival.
        let spec = crate::style::Style::Classical.spec();
        let intent = MusicalIntent::default();
        let s = compose_with_spec(&intent, &spec);
        let key = s.key;
        let bars = intent.bars.max(1) as i64;
        let intro = spec.texture.intro_bars as i64;
        // Default seed 0 -> ternary: 3 sections.
        let target_bar = intro + 3 * 2 * bars - 1;
        let (lo, hi) = (target_bar as f64 * 4.0, (target_bar + 1) as f64 * 4.0);
        let mediant = key.scale().degree_pitch_class(3);
        let mut saw_bass = false;
        for n in &s.notes {
            let on = n.onset.beats();
            if on < lo - 1e-9 || on >= hi {
                continue;
            }
            match n.role {
                VoiceRole::Bass => {
                    saw_bass = true;
                    assert_eq!(
                        n.pitch.pitch_class(),
                        mediant,
                        "the evaded close stands on the mediant"
                    );
                }
                VoiceRole::Melody => {
                    assert_ne!(
                        n.pitch.pitch_class(),
                        key.tonic,
                        "the evaded melody must not land home"
                    );
                }
                _ => {}
            }
        }
        assert!(saw_bass);
        // The coda still arrives.
        let mut bass: Vec<_> = s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Bass)
            .collect();
        bass.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        assert_eq!(bass.last().unwrap().pitch.pitch_class(), key.tonic);
    }

    #[test]
    fn tango_cadences_lean_and_marches_do_not() {
        // The appoggiatura: an accented upper neighbor ON the beat,
        // resolving down onto the true cadential tone. Rate is style DNA:
        // the tango leans (0.6, 'dramatic' per the review), the march
        // leans on nothing (0.0).
        let spec = crate::style::Style::Tango.spec();
        assert!(spec.melody.appoggiatura_rate > 0.5);
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        let mut melody = s.voice(VoiceRole::Melody);
        melody.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        let mut leans = 0;
        for w in melody.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if a.emphasis == Emphasis::Cadential
                && b.emphasis == Emphasis::Cadential
                && ((a.onset + a.duration).beats() - b.onset.beats()).abs() < 1e-6
                && a.pitch.midi() > b.pitch.midi()
                && a.velocity > b.velocity
            {
                leans += 1;
            }
        }
        assert!(leans >= 1, "a 0.6-rate tango must lean at least once");
        let march = compose_with_spec(
            &MusicalIntent::default(),
            &crate::style::Style::March.spec(),
        );
        assert_eq!(
            crate::style::Style::March.spec().melody.appoggiatura_rate,
            0.0
        );
        // No structural check needed for the march: rate 0 makes the pass
        // inert by its first branch (pinned here as the contract).
        assert!(!march.notes.is_empty());
    }

    #[test]
    fn the_first_close_is_deceptive_and_the_last_is_earned() {
        // Cadence::Deceptive existed in the vocabulary from the start but
        // was never composed — this pins its first real use. The opening
        // period's final bar must carry vi (bass on the submediant,
        // harmony within the vi triad); the piece's LAST close must stay
        // authentic — the deception only works because the promise is
        // eventually kept.
        let spec = crate::style::Style::Classical.spec();
        assert!(spec.texture.deceptive_close);
        let intent = MusicalIntent::default();
        let s = compose_with_spec(&intent, &spec);
        let key = s.key;
        let bars = intent.bars.max(1) as i64;
        let intro = spec.texture.intro_bars as i64;
        let target_bar = intro + 2 * bars - 1; // intro shifts everything
        // Only the FIRST half of the bar belongs to the deception; the
        // second half is the pivot dominant into the departure (see
        // departures_are_announced_by_the_new_keys_dominant) — vi resolves
        // deceptively, then becomes the pivot into the relative key.
        let (lo, hi) = (target_bar as f64 * 4.0, target_bar as f64 * 4.0 + 2.0);
        let vi_pcs: Vec<_> = key
            .diatonic_triad(6)
            .voice(3)
            .iter()
            .map(|p| p.pitch_class())
            .collect();
        let in_bar = |n: &ScoreNote| n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi;
        let bass_in_bar: Vec<_> = s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Bass && in_bar(n))
            .collect();
        assert!(!bass_in_bar.is_empty());
        for b in &bass_in_bar {
            assert_eq!(
                b.pitch.pitch_class(),
                key.scale().degree_pitch_class(6),
                "the deceptive bass must land on the submediant"
            );
        }
        for h in s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Harmony && in_bar(n))
        {
            assert!(
                vi_pcs.contains(&h.pitch.pitch_class()),
                "harmony must re-voice inside vi"
            );
        }
        // Structural melody notes only — a brief passing/neighbor tone
        // (< 0.9 beats, the same "ornamental, not structural" threshold
        // used elsewhere, e.g. the climax-hold search above) is legitimate
        // decoration connecting two vi-triad tones, not a claim that the
        // whole bar's harmonic identity left vi. Classical's melodic DNA
        // (added after this test was first written) can now place such a
        // tone here; requiring EVERY note including ornaments to be a bare
        // chord tone was stricter than the actual "consonant with the
        // deception" claim this test intends.
        for m in s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody && in_bar(n) && n.duration.beats() >= 0.9)
        {
            assert!(
                vi_pcs.contains(&m.pitch.pitch_class()),
                "melody must be consonant with the deception"
            );
        }
        // And the final bass tone of the piece is the tonic: earned.
        let mut bass: Vec<_> = s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Bass)
            .collect();
        bass.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        assert_eq!(bass.last().unwrap().pitch.pitch_class(), key.tonic);
    }

    #[test]
    fn styles_that_never_deny_closure_are_untouched() {
        // A lullaby's first close stays authentic — deceptive_close is
        // opt-in drama, not a default.
        let spec = crate::style::Style::Lullaby.spec();
        assert!(!spec.texture.deceptive_close);
        let intent = MusicalIntent::default();
        let s = compose_with_spec(&intent, &spec);
        let key = s.key;
        let bars = intent.bars.max(1) as i64;
        let intro = spec.texture.intro_bars as i64;
        let target_bar = intro + 2 * bars - 1;
        let (lo, hi) = (target_bar as f64 * 3.0, (target_bar + 1) as f64 * 3.0);
        let sub = key.scale().degree_pitch_class(6);
        for b in s.notes.iter().filter(|n| {
            n.role == VoiceRole::Bass && n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi
        }) {
            assert_ne!(b.pitch.pitch_class(), sub, "lullaby close must not deceive");
        }
    }

    #[test]
    fn classic_rhetoric_is_a_strict_noop() {
        // The compatibility contract for the rhetoric layer.
        let mut spec = crate::style::Style::Classical.spec();
        spec.rhetoric = crate::spec::PhraseRhetoric::Classic;
        let score = compose_with_spec(&MusicalIntent::default(), &spec);
        let mut twin = score.clone();
        apply_phrase_rhetoric(&mut twin, crate::spec::PhraseRhetoric::Classic, 4.0);
        assert_eq!(score, twin);
    }

    #[test]
    fn declamatory_rhetoric_interrupts_and_stops_sharp() {
        // Tango's conversation: statement — INTERRUPTION — answer — sharp
        // stop. Every cadence is short (never held), and the approach to
        // most cadences contains a composed silence.
        let spec = crate::style::Style::Tango.spec();
        assert_eq!(spec.rhetoric, crate::spec::PhraseRhetoric::Declamatory);
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        let mut melody = s.voice(VoiceRole::Melody);
        melody.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        let cadences: Vec<&ScoreNote> = melody
            .iter()
            .filter(|n| n.emphasis == Emphasis::Cadential)
            .collect();
        assert!(!cadences.is_empty());
        for c in &cadences {
            assert!(
                c.duration.beats() <= 1.0 + 1e-9,
                "a declamatory cadence is a stop, not a hold ({} beats)",
                c.duration.beats()
            );
        }
        // Composed silence before most cadences: a melody gap >= 0.4
        // beats somewhere in the two bars before the cadence.
        let mut with_gap = 0;
        for c in &cadences {
            let lo = c.onset.beats() - 8.0;
            let mut covered_until = lo;
            let mut gap = 0.0f64;
            for n in &melody {
                let (on, off) = (n.onset.beats(), (n.onset + n.duration).beats());
                if off <= lo || on >= c.onset.beats() {
                    continue;
                }
                if on > covered_until {
                    gap = gap.max(on - covered_until);
                }
                covered_until = covered_until.max(off);
            }
            if gap >= 0.4 {
                with_gap += 1;
            }
        }
        assert!(
            with_gap * 2 >= cadences.len(),
            "the interruption must precede most cadences ({with_gap}/{})",
            cadences.len()
        );
    }

    #[test]
    fn singing_rhetoric_suspends_and_arrives_quietly() {
        // Nocturne's conversation: the tone before the cadence OVERSTAYS
        // (delaying the arrival off the barline) and the arrival lands
        // more softly than its approach.
        let spec = crate::style::Style::Nocturne.spec();
        assert_eq!(spec.rhetoric, crate::spec::PhraseRhetoric::Singing);
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        let mut melody = s.voice(VoiceRole::Melody);
        melody.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        let mut delayed_and_soft = 0;
        for (i, n) in melody.iter().enumerate() {
            if n.emphasis != Emphasis::Cadential || i == 0 {
                continue;
            }
            let frac = n.onset.beats() - n.onset.beats().floor();
            if (frac - 0.5).abs() < 1e-6 && n.velocity < melody[i - 1].velocity {
                delayed_and_soft += 1;
            }
        }
        assert!(
            delayed_and_soft >= 1,
            "at least one cadence must show the written-out suspension \
             (delayed onset) and land softly"
        );
    }

    #[test]
    fn seventh_chords_extend_every_chord_not_just_the_cadential_one() {
        // Isolated (the pattern established over this session's waves):
        // realize the SAME non-dominant, non-applied degree with the flag
        // off and on, and confirm the ONLY difference is the tone count.
        let key = crate::harmony::Key::major(crate::pitch::PitchClass::C);
        let intent = MusicalIntent::default();

        let mut triad_score = Score::new(key, 100.0, 4);
        realize_harmony_measures(
            &mut triad_score,
            &[1], // tonic — neither the cadential dominant nor an applied one
            key,
            4.0,
            &intent,
            &mut Vec::new(),
            0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            false,
            false, // seventh_chords OFF
        );
        // The Harmony voice carries the UPPER structure only (`lead_upper`
        // deliberately skips the root — that's the Bass voice's job), so
        // a plain triad shows up as 2 tones here, not 3.
        assert_eq!(
            triad_score.voice(VoiceRole::Harmony).len(),
            2,
            "off, a non-dominant degree stays a plain triad's upper voices"
        );

        let mut seventh_score = Score::new(key, 100.0, 4);
        realize_harmony_measures(
            &mut seventh_score,
            &[1],
            key,
            4.0,
            &intent,
            &mut Vec::new(),
            0,
            1.0,
            crate::accompaniment::Accompaniment::Block,
            false,
            true, // seventh_chords ON
        );
        let seventh_notes = seventh_score.voice(VoiceRole::Harmony);
        assert_eq!(
            seventh_notes.len(),
            3, // upper structure only: third + fifth + seventh
            "on, EVERY degree gets its diatonic seventh, not just the dominant"
        );
        // The seventh chord must be a strict superset of the triad — the
        // safety property the doc comment claims.
        let triad_pcs: std::collections::HashSet<_> = triad_score
            .voice(VoiceRole::Harmony)
            .iter()
            .map(|n| n.pitch.pitch_class())
            .collect();
        let seventh_pcs: std::collections::HashSet<_> = seventh_notes
            .iter()
            .map(|n| n.pitch.pitch_class())
            .collect();
        assert!(
            triad_pcs.is_subset(&seventh_pcs),
            "the seventh chord must contain every triad tone, plus one more"
        );
    }

    #[test]
    fn jazz_ballad_is_aeolian_with_sevenths_and_a_ii_v_i_turnaround() {
        let spec = crate::style::Style::JazzBallad.spec();
        assert_eq!(spec.mode, Some(crate::scale::Mode::Aeolian));
        assert!(spec.texture.seventh_chords);
        assert_eq!(
            spec.progression,
            crate::spec::ProgressionSpec::Archetype(vec![2, 5, 1, 6, 2, 5, 1, 5])
        );
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert_eq!(
            s.key.tonality,
            crate::harmony::Tonality::Modal(crate::scale::Mode::Aeolian)
        );
        // The Harmony voice carries the upper structure only (the root
        // lives in Bass — see the isolated test above), so a seventh
        // chord shows up as 3 tones there (3rd+5th+7th), not 4. At least
        // one onset in the real piece must show 3 — proof sevenths
        // actually reached the real pipeline, not just the isolated unit.
        let mut by_onset: std::collections::HashMap<i64, std::collections::BTreeSet<u8>> =
            std::collections::HashMap::new();
        for n in s.notes.iter().filter(|n| n.role == VoiceRole::Harmony) {
            by_onset
                .entry((n.onset.beats() * 4.0).round() as i64)
                .or_default()
                .insert(n.pitch.midi());
        }
        assert!(
            by_onset.values().any(|tones| tones.len() >= 3),
            "at least one struck chord in a real JazzBallad piece must be a real seventh chord"
        );
    }

    #[test]
    fn additive_process_grows_then_shrinks_the_hook_cell() {
        // Isolated, using the same real, cheap Form constructor form.rs's
        // own tests use — sidesteps the section-boundary pitfalls that
        // cost a whole debugging cycle on parallel planing (this pass
        // reuses that FIXED `2*bars`-per-section formula, but verify it
        // directly rather than assume).
        let key = Key::major(PitchClass::C);
        let motif = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let prog = crate::harmony::Progression::authentic();
        let seed = 11u64;
        let form = Form::ternary(&motif, key, &prog, 4.0, seed, false, None);
        let mut score = Score::new(key, 100.0, 4);
        // A real cadence-free canvas: no existing melody to interfere with
        // (the pass removes whatever's there in its own window anyway).
        score.total_beats = Duration::new(24, 1); // 3 sections * 2*4 bars * 4 beats
        apply_additive_process(&mut score, &form, 4, 4.0, seed);

        let cell =
            crate::hook::HookCell::generate_with(&crate::spec::MelodicDna::default(), seed, 4.0);
        assert!(!cell.notes.is_empty());

        let mut melody: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .collect();
        melody.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        assert!(!melody.is_empty(), "the A section must get a real melody");

        // Reconstruct the expected degree sequence: 1 note of the cell,
        // then 2, ... up to the full cell, then back down to 1, repeating
        // until the section's beats run out — and confirm the ACTUAL
        // melody's degree-index pattern (which cell note each pitch
        // matches) follows the same growth/shrink shape.
        let scale = key.scale();
        let cell_pitch_at = |i: usize| -> Pitch {
            let mut p = scale.degree_pitch(cell.notes[i].0, 5);
            while p.midi() > MELODY_CEILING_MIDI {
                p = p.transpose(-12);
            }
            p
        };
        // The very first note must be the cell's own first note (k starts
        // at 1) and the SECOND repetition must be longer (k=2), proving
        // real growth, not a fixed-length loop.
        assert_eq!(melody[0].pitch, cell_pitch_at(0));
        // Find where the pattern restarts at cell[0] again (the second
        // repetition's start) and confirm it's followed by cell[1] before
        // restarting a third time — i.e. the second group has >= 2 notes.
        let restart = melody
            .iter()
            .skip(1)
            .position(|n| n.pitch == cell_pitch_at(0))
            .map(|i| i + 1)
            .expect("the cell must repeat at least once");
        assert!(
            restart + 1 < melody.len() && melody[restart + 1].pitch == cell_pitch_at(1),
            "the second repetition must be longer than the first — real growth"
        );
    }

    #[test]
    fn harmonic_sequence_rewrites_the_b_section_as_a_continuous_circle_of_fifths() {
        // Isolated, using the same real, cheap Form constructor the
        // additive-process test above established as the pattern: build a
        // genuine Form::ternary and inspect what the pass did to its B
        // section's stored progression directly, before any note gets
        // realized.
        let key = Key::major(PitchClass::C);
        let motif = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let prog = crate::harmony::Progression::authentic(); // [1,4,5,1], 4 bars
        let mut form = Form::ternary(&motif, key, &prog, 4.0, 11, false, None);
        apply_harmonic_sequence(&mut form);

        let b = form
            .sections
            .iter()
            .find(|s| s.role == SectionRole::B)
            .expect("ternary always has a B section");
        let dom = b.key.cadence_dominant_degree();
        let ante = &b.period.antecedent.progression;
        let cons = &b.period.consequent.progression;

        // Each phrase's cadential tail must survive untouched — exactly
        // what Period::parallel_in already steered the melody toward.
        assert_eq!(*ante.last().unwrap(), dom, "half cadence preserved");
        assert_eq!(
            cons[cons.len() - 2],
            dom,
            "authentic cadence dominant preserved"
        );
        assert_eq!(
            *cons.last().unwrap(),
            1,
            "authentic cadence tonic preserved"
        );

        // The sequenced portion (everything before each tail) must walk a
        // REAL, CONTINUOUS circle of fifths: every consecutive pair,
        // including across the antecedent/consequent seam, sits a
        // diatonic fifth apart, and it must actually differ from the
        // original archetype (not a no-op).
        let mut sequenced: Vec<i32> = ante[..ante.len() - 1].to_vec();
        sequenced.extend_from_slice(&cons[..cons.len() - 2]);
        assert!(
            sequenced.len() >= 3,
            "need enough sequenced bars to prove continuity: {sequenced:?}"
        );
        for w in sequenced.windows(2) {
            let expected_next = (((w[0] - 1) + 3).rem_euclid(7)) + 1;
            assert_eq!(
                w[1], expected_next,
                "each step must be a real fifth from its predecessor: {sequenced:?}"
            );
        }
        // The exact textbook opening of the descending-fifths circle
        // (I-IV-vii°-iii-vi-...), starting from the section's own tonic —
        // not the [1,4,5,1] archetype it replaced.
        assert_eq!(sequenced, vec![1, 4, 7, 3, 6]);
    }

    #[test]
    fn suspensions_tie_a_real_voice_leading_candidate_and_resolve_down_by_step() {
        // Isolated, mirroring the blue-notes pattern: build a score by
        // hand from two consecutive triads with a genuine voice-leading
        // relationship (iv's root, F, sits one diatonic step below i's
        // fifth, G) and confirm the pass finds it and only it.
        let key = Key::modal(crate::pitch::PitchClass::C, crate::scale::Mode::Phrygian).unwrap();
        let mut score = Score::new(key, 100.0, 4);
        let bar = Duration::new(4, 1);
        for pitch in key.diatonic_triad(1).voice(4) {
            score.push(ScoreNote {
                pitch,
                onset: Duration::zero(),
                duration: bar,
                velocity: 0.5,
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        for pitch in key.diatonic_triad(4).voice(4) {
            score.push(ScoreNote {
                pitch,
                onset: bar,
                duration: bar,
                velocity: 0.5,
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        apply_suspensions(&mut score, key, 1.0, 3);
        let mut harmony: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Harmony)
            .collect();
        harmony.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        // 3 untouched chord-1 notes + 2 untouched chord-2 notes + a
        // suspend/resolve pair replacing the one candidate = 7.
        assert_eq!(
            harmony.len(),
            7,
            "exactly one voice must be replaced by a suspend+resolve pair"
        );
        let at_bar_onset: Vec<&&ScoreNote> = harmony
            .iter()
            .filter(|n| (n.onset.beats() - bar.beats()).abs() < 1e-9)
            .collect();
        // The suspension strikes alongside chord-2's two UNTOUCHED voices
        // (Ab and C both still land cleanly on the bar) — three notes
        // total, only one of which is the tied-over suspension.
        assert_eq!(
            at_bar_onset.len(),
            3,
            "chord-2's other two voices strike normally"
        );
        let fifth_of_i = key.diatonic_triad(1).voice(4)[2]; // G, the outgoing tone
        let suspended = **at_bar_onset
            .iter()
            .find(|n| n.pitch == fifth_of_i)
            .expect("the outgoing chord's fifth must appear, tied over");
        assert!(
            suspended.velocity > 0.5,
            "the suspension leans in — accented, not just tied"
        );
        let resolution = harmony
            .iter()
            .find(|n| n.onset.beats() > bar.beats() + 1e-9 && n.role == VoiceRole::Harmony)
            .expect("a resolution note must follow");
        assert_eq!(
            resolution.pitch.midi(),
            key.diatonic_triad(4).voice(4)[0].midi(), // F, iv's root — the true chord-2 tone
            "the resolution must land on the real chord-2 tone"
        );
        assert!(
            resolution.pitch.midi() < suspended.pitch.midi(),
            "the resolution must be BELOW the suspension — it resolves down"
        );
        assert!(
            resolution.velocity < suspended.velocity,
            "the resolution releases — quieter than the suspension"
        );
    }

    #[test]
    fn harmonic_stasis_ties_a_repeated_chord_into_one_long_sustain() {
        // Isolated: three consecutive bars, the SAME triad struck fresh
        // each bar (what `realize_harmony_measures` produces on its own
        // for a static progression) — the pass must merge all three
        // bars' worth of each voice into ONE note spanning all three,
        // not just tie pairs.
        let key = Key::major(crate::pitch::PitchClass::C);
        let mut score = Score::new(key, 100.0, 4);
        let bar = Duration::new(4, 1);
        for bar_idx in 0..3 {
            for pitch in key.diatonic_triad(1).voice(4) {
                score.push(ScoreNote {
                    pitch,
                    onset: bar.scale(bar_idx, 1),
                    duration: bar,
                    velocity: 0.5,
                    role: VoiceRole::Harmony,
                    emphasis: Emphasis::Normal,
                    section_intensity: 1.0,
                });
            }
        }
        // A fourth bar changes chord — this boundary must NOT merge.
        for pitch in key.diatonic_triad(4).voice(4) {
            score.push(ScoreNote {
                pitch,
                onset: bar.scale(3, 1),
                duration: bar,
                velocity: 0.5,
                role: VoiceRole::Harmony,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        apply_harmonic_stasis(&mut score);
        let mut harmony: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Harmony)
            .collect();
        harmony.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        // 3 tones tied across bars 0-2 into one long note each, plus 3
        // fresh tones for the changed chord in bar 3 = 6 notes total,
        // down from the original 12.
        assert_eq!(
            harmony.len(),
            6,
            "three tied 3-bar sustains plus three fresh notes for the new chord"
        );
        let tied: Vec<&&ScoreNote> = harmony
            .iter()
            .filter(|n| (n.onset.beats()).abs() < 1e-9)
            .collect();
        assert_eq!(tied.len(), 3, "the three voices of the static chord");
        for n in &tied {
            assert!(
                (n.duration.beats() - 12.0).abs() < 1e-9,
                "each tied voice must span all three bars (12 beats), got {}",
                n.duration.beats()
            );
        }
        let after: Vec<&&ScoreNote> = harmony
            .iter()
            .filter(|n| (n.onset.beats() - 12.0).abs() < 1e-9)
            .collect();
        assert_eq!(after.len(), 3, "the changed chord's three fresh voices");
        for n in &after {
            assert!(
                (n.duration.beats() - 4.0).abs() < 1e-9,
                "the changed chord is NOT tied to anything before it"
            );
        }
    }

    #[test]
    fn ambient_is_the_slowest_style_and_composes_with_real_stasis() {
        let spec = crate::style::Style::Ambient.spec();
        assert!(spec.texture.harmonic_stasis);
        assert_eq!(spec.texture.coda_bars, 0);
        let (lo, _) = spec.tempo_range;
        // Slower than every other style's own floor.
        for other in [
            crate::style::Style::Nocturne,
            crate::style::Style::Lullaby,
            crate::style::Style::SacredChoral,
        ] {
            assert!(
                lo < other.spec().tempo_range.0,
                "Ambient must be slower than {other:?}"
            );
        }
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert!(!s.notes.is_empty(), "a real piece must come out");
        // Real stasis reaching the full pipeline: at least one Harmony
        // note must span MORE than one bar (4 beats) — proof repeated
        // chords actually tied together in a real composed piece, not
        // just the isolated unit above.
        let longest_harmony = s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Harmony)
            .map(|n| n.duration.beats())
            .fold(0.0_f64, f64::max);
        assert!(
            longest_harmony > 4.0 + 1e-6,
            "at least one real sustained tone must span more than one bar, got {longest_harmony}"
        );
    }

    #[test]
    fn sacred_choral_is_phrygian_with_a_plagal_amen() {
        let spec = crate::style::Style::SacredChoral.spec();
        assert_eq!(spec.mode, Some(crate::scale::Mode::Phrygian));
        assert!(spec.texture.suspension_rate > 0.0);
        assert!(spec.texture.coda_bars > 0, "the plagal Amen needs a coda");
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert_eq!(
            s.key.tonality,
            crate::harmony::Tonality::Modal(crate::scale::Mode::Phrygian)
        );
        // The coda's penultimate bar is subdominant (IV/iv) — the plagal
        // half of the "Amen" — landing on the tonic in the final bar.
        let sub_pc = s.key.scale().degree_pitch_class(4);
        let tonic_pc = s.key.tonic;
        let mut bass: Vec<&ScoreNote> = s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Bass)
            .collect();
        bass.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        let last_two: Vec<_> = bass.iter().rev().take(2).collect();
        assert_eq!(last_two.len(), 2);
        assert_eq!(
            last_two[0].pitch.pitch_class(),
            tonic_pc,
            "the piece must end on the tonic"
        );
        assert_eq!(
            last_two[1].pitch.pitch_class(),
            sub_pc,
            "the tonic must be approached plagally, from the subdominant"
        );
    }

    #[test]
    fn parallel_planing_rides_the_melodys_contour_in_the_contrast_section() {
        // Position-independent by construction: rather than reconstruct
        // exactly where the B section lands in FINAL beat-space (which
        // depends on section layout, intro shift, and staged-entrance
        // thinning all lining up), scan the whole piece for the
        // CONTRACT's signature wherever it appears — two moments, each
        // with a concurrent melody note and a same-size struck chord,
        // where the chord shifted by EXACTLY the interval the melody did.
        let spec = crate::style::Style::Impressionism.spec();
        assert!(spec.texture.planing);
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert_eq!(
            s.key.tonality,
            crate::harmony::Tonality::Modal(crate::scale::Mode::Lydian)
        );

        let mut melody_onsets: Vec<f64> = s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .map(|n| n.onset.beats())
            .collect();
        melody_onsets.sort_by(f64::total_cmp);
        melody_onsets.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        let melody_at = |t: f64| -> Option<i32> {
            s.notes
                .iter()
                .find(|n| n.role == VoiceRole::Melody && (n.onset.beats() - t).abs() < 1e-9)
                .map(|n| n.pitch.midi() as i32)
        };
        let chord_at = |t: f64| -> Vec<i32> {
            let mut v: Vec<i32> = s
                .notes
                .iter()
                .filter(|n| n.role == VoiceRole::Harmony && (n.onset.beats() - t).abs() < 1e-9)
                .map(|n| n.pitch.midi() as i32)
                .collect();
            v.sort_unstable();
            v
        };

        let mut confirmed = 0usize;
        for w in melody_onsets.windows(2) {
            let (Some(m0), Some(m1)) = (melody_at(w[0]), melody_at(w[1])) else {
                continue;
            };
            let (c0, c1) = (chord_at(w[0]), chord_at(w[1]));
            if c0.is_empty() || c0.len() != c1.len() {
                continue;
            }
            let shift = m1 - m0;
            let shifted: Vec<i32> = c0.iter().map(|p| (p + shift).clamp(0, 127)).collect();
            if shifted == c1 {
                confirmed += 1;
            }
        }
        assert!(
            confirmed >= 1,
            "at least one pair of consecutive melody notes must show the chord \
             shape shifted by their exact interval"
        );
    }

    #[test]
    fn blue_notes_flatten_the_third_and_seventh_without_touching_harmony() {
        // Isolated, not through the full pipeline: downstream passes
        // (damage, in particular) react to melody CONTENT, so diffing two
        // full compositions that differ upstream in pitch is not a stable
        // signal — the note count itself can drift. Build a score by hand
        // instead, mirroring accompaniment.rs's local `realize()` pattern.
        let key = Key::major(PitchClass::C);
        let mut score = Score::new(key, 100.0, 4);
        let scale = key.scale();
        let third = scale.degree_pitch(3, 4); // E4 — eligible
        let seventh = scale.degree_pitch(7, 4); // B4 — eligible
        let fifth = scale.degree_pitch(5, 4); // G4 — NOT eligible
        for (i, pitch) in [third, seventh, fifth].into_iter().enumerate() {
            score.push(ScoreNote {
                pitch,
                onset: Duration::new(i as i64, 1),
                duration: Duration::new(1, 1),
                velocity: 0.5,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        // A harmony note at the SAME pitch class as the third — must
        // never be touched; the pass is melody-only by construction.
        score.push(ScoreNote {
            pitch: third,
            onset: Duration::zero(),
            duration: Duration::new(1, 1),
            velocity: 0.5,
            role: VoiceRole::Harmony,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        apply_blue_notes(&mut score, key, 1.0, 7);
        let melody: Vec<&ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .collect();
        assert_eq!(
            melody[0].pitch.midi(),
            third.midi() - 1,
            "the third must flatten"
        );
        assert_eq!(
            melody[1].pitch.midi(),
            seventh.midi() - 1,
            "the seventh must flatten"
        );
        assert_eq!(
            melody[2].pitch.midi(),
            fifth.midi(),
            "the fifth must NEVER flatten"
        );
        let harmony = score
            .notes
            .iter()
            .find(|n| n.role == VoiceRole::Harmony)
            .unwrap();
        assert_eq!(
            harmony.pitch.midi(),
            third.midi(),
            "harmony must never be touched"
        );
    }

    #[test]
    fn blues_really_is_the_twelve_bar_i_iv_v_turnaround() {
        let spec = crate::style::Style::Blues.spec();
        assert_eq!(
            spec.mode, None,
            "blues stays major — the color is melodic, not modal"
        );
        // A pool of real 12-bar-blues variants (not a single fixed
        // Archetype) since `realize_call_response` gives each chorus in a
        // multi-chorus piece its own seed_variant -- see
        // call_response::harmony_genuinely_varies_chorus_to_chorus_not_just_the_melody.
        // Index 0 (seed 0 % pool.len()) is still the exact standard
        // turnaround this test always asserted.
        let crate::spec::ProgressionSpec::ArchetypePool(pool) = &spec.progression else {
            panic!("expected an ArchetypePool: {:?}", spec.progression);
        };
        assert_eq!(pool[0], vec![1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 1]);
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert!(!s.notes.is_empty());
        assert_eq!(s.key.tonality, crate::harmony::Tonality::Major);
    }

    #[test]
    fn baroque_suite_compatibility_baseline_still_composes_and_genuinely_differs_from_the_new_route()
     {
        // The 2026-07-26/27 harmonic-syntax pilot moved BaroqueSuite from a
        // fixed I-IV-V-I loop to the real functional-harmony generator
        // (ProgressionSpec::Grammar) -- A/B evidence (note-data + rendered-
        // audio analysis across 2 seeds) favored the new route, but per
        // that review's own recommendation the old route is preserved as
        // a named constant (`style::BAROQUE_SUITE_COMPATIBILITY_PROGRESSION`),
        // not deleted: a compatibility baseline, an Analyst A/B control,
        // and this regression fixture proving it still composes cleanly.
        let new_spec = crate::style::Style::BaroqueSuite.spec();
        assert_eq!(
            new_spec.progression,
            crate::spec::ProgressionSpec::Grammar,
            "the pilot's landed route"
        );
        let mut old_spec = new_spec.clone();
        old_spec.progression = crate::spec::ProgressionSpec::Archetype(
            crate::style::BAROQUE_SUITE_COMPATIBILITY_PROGRESSION.to_vec(),
        );

        // The compatibility baseline must still compose a valid, non-empty
        // piece -- it's a real fallback, not a fossil.
        let old_score = compose_with_spec(&MusicalIntent::default(), &old_spec);
        assert!(!old_score.notes.is_empty());

        // And the two routes must genuinely differ in at least one real
        // seed's harmony-voice pitch sequence -- the actual A/B evidence
        // this pilot was judged on, now codified rather than left as a
        // one-off example script.
        let differs = (0..10u64).any(|seed| {
            let intent = MusicalIntent {
                seed,
                ..MusicalIntent::default()
            };
            let old = compose_with_spec(&intent, &old_spec);
            let new = compose_with_spec(&intent, &new_spec);
            let harmony_pitches = |s: &crate::score::Score| -> Vec<u8> {
                s.voice(VoiceRole::Harmony)
                    .iter()
                    .map(|n| n.pitch.midi())
                    .collect()
            };
            harmony_pitches(&old) != harmony_pitches(&new)
        });
        assert!(
            differs,
            "expected at least one seed among 0..10 where the compatibility \
             baseline and the new functional route produce different harmony"
        );
    }

    #[test]
    fn celtic_really_is_mixolydian_with_a_double_tonic_close() {
        // Not Dorian (ModalFolk already owns that) — degree 7 of
        // Mixolydian IS the flat seventh, so the bVII-I progression in the
        // spec needs no special-casing to sound modal.
        let mut spec = crate::style::Style::Celtic.spec();
        assert_eq!(spec.mode, Some(crate::scale::Mode::Mixolydian));
        spec.texture.damage = 0.0; // pristine — no chromatic wait-tone
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert_eq!(
            s.key.tonality,
            crate::harmony::Tonality::Modal(crate::scale::Mode::Mixolydian)
        );
        // No raised leading tone (Ionian's #7) anywhere in the melody —
        // that would betray a functional dominant sneaking into a modal
        // piece.
        let leading_tone = s.key.scale().degree_pitch_class(7);
        let major_leading = crate::harmony::Key::major(s.key.tonic)
            .scale()
            .degree_pitch_class(7);
        assert_ne!(
            leading_tone, major_leading,
            "Mixolydian's flat seventh must differ from the major leading tone"
        );
    }

    #[test]
    fn the_drone_is_a_fixed_pedal_that_ignores_the_harmony() {
        // The Celtic habit: the bass becomes a sustained tonic-fifth
        // pedal, independent of whatever the harmony above it is doing —
        // covering the FULL span (intro and coda included), never
        // following the progression's roots the way every other style's
        // bass does.
        let spec = crate::style::Style::Celtic.spec();
        assert!(spec.texture.drone);
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        let bass: Vec<&ScoreNote> = s
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Bass)
            .collect();
        assert!(!bass.is_empty());
        let tonic_pc = s.key.tonic;
        let fifth_pc = s.key.scale().degree_pitch_class(5);
        for b in &bass {
            let pc = b.pitch.pitch_class();
            assert!(
                pc == tonic_pc || pc == fifth_pc,
                "every drone tone must be the tonic or the fifth, got {pc:?}"
            );
        }
        // Full coverage: the pedal must reach all the way to the piece's
        // final bar, not just the opening.
        let last_bass_end = bass
            .iter()
            .map(|n| (n.onset + n.duration).beats())
            .fold(0.0_f64, f64::max);
        assert!(
            last_bass_end >= s.total_beats.beats() - 1e-6,
            "the drone must cover the piece's full span ({last_bass_end} < {})",
            s.total_beats.beats()
        );
    }

    #[test]
    fn ornaments_are_unaccented_cuts_not_leans() {
        // The Celtic habit's other half: distinguish a "cut" from an
        // appoggiatura by construction — quieter than the note it
        // decorates (never louder), and brief (never taking the beat).
        let spec = crate::style::Style::Celtic.spec();
        assert!(spec.melody.ornament_rate > 0.0);
        assert_eq!(
            spec.melody.appoggiatura_rate, 0.0,
            "a jig cuts, it doesn't lean"
        );
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        let mut melody = s.voice(VoiceRole::Melody);
        melody.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        // Find at least one very short, quiet note immediately followed
        // by a louder one at the SAME onset-adjacency the ornament pass
        // creates (grace note, then the shortened main note right after).
        let mut found_a_cut = false;
        for w in melody.windows(2) {
            let (grace, main) = (&w[0], &w[1]);
            let touches =
                ((grace.onset + grace.duration).beats() - main.onset.beats()).abs() < 1e-6;
            if touches && grace.duration.beats() <= 0.26 && grace.velocity < main.velocity {
                found_a_cut = true;
                break;
            }
        }
        assert!(found_a_cut, "at least one quick unaccented cut must appear");
    }

    #[test]
    fn martial_rhetoric_strikes_hard_and_never_falls_silent() {
        // March's conversation: statement — statement — STRIKE. No
        // interruption, no silence — a march never stops. Diff against a
        // Classic-rhetoric twin composed from the identical seed (the same
        // technique `classic_rhetoric_is_a_strict_noop` uses to isolate the
        // rhetoric layer): every cadence must come out clipped and hit
        // harder, and the note landing on the cadence's own downbeat must
        // gain its own accent — with nothing about the note ONSETS moved
        // (a march never stops to make room for the effect; it only leans
        // harder on where it already was).
        let mut spec = crate::style::Style::March.spec();
        assert_eq!(spec.rhetoric, crate::spec::PhraseRhetoric::Martial);
        let intent = MusicalIntent::default();
        let martial = compose_with_spec(&intent, &spec);
        spec.rhetoric = crate::spec::PhraseRhetoric::Classic;
        let classic = compose_with_spec(&intent, &spec);

        let cad_classic: Vec<&ScoreNote> = classic
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Cadential)
            .collect();
        let cad_martial: Vec<&ScoreNote> = martial
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Cadential)
            .collect();
        assert!(!cad_classic.is_empty());
        assert_eq!(cad_classic.len(), cad_martial.len());

        let mut struck_harder = 0;
        for (c, m) in cad_classic.iter().zip(cad_martial.iter()) {
            assert!(
                (c.onset.beats() - m.onset.beats()).abs() < 1e-9,
                "a march never stops — the cadence onset must not move"
            );
            assert!(
                m.duration.beats() <= 1.0 + 1e-9,
                "a martial cadence is a strike, not a hold ({} beats)",
                m.duration.beats()
            );
            if m.velocity > c.velocity + 1e-6 {
                struck_harder += 1;
            }
        }
        assert!(
            struck_harder >= 1,
            "at least one cadence must be struck harder than its classic twin"
        );

        // The pickup: the note whose end touches a cadence onset must gain
        // its own accent too.
        let mut pickup_accented = 0;
        for m_cad in &cad_martial {
            let onset = m_cad.onset.beats();
            let pickup_m = martial.notes.iter().find(|n| {
                n.role == VoiceRole::Melody && ((n.onset + n.duration).beats() - onset).abs() < 1e-6
            });
            let pickup_c = classic.notes.iter().find(|n| {
                n.role == VoiceRole::Melody && ((n.onset + n.duration).beats() - onset).abs() < 1e-6
            });
            if let (Some(pm), Some(pc)) = (pickup_m, pickup_c)
                && pm.velocity > pc.velocity + 1e-6
            {
                pickup_accented += 1;
            }
        }
        assert!(
            pickup_accented >= 1,
            "at least one pickup note must gain its own accent"
        );
    }

    #[test]
    fn chorale_rhetoric_holds_cadences_without_shifting_onset_or_softening() {
        // SacredChoral's conversation: statement — statement — a SUSTAINED
        // close. Added after a melody-only listening test found this style
        // collapsing into Nocturne's `Singing` identity. Diff against a
        // Classic-rhetoric twin (same technique the Martial test uses):
        // every cadence's ONSET must be unchanged (no suspension-lean,
        // unlike Singing) but its DURATION must grow (the fermata hold),
        // and its velocity must not drop (unlike Singing's quiet arrival).
        let mut spec = crate::style::Style::SacredChoral.spec();
        assert_eq!(spec.rhetoric, crate::spec::PhraseRhetoric::Chorale);
        let intent = MusicalIntent::default();
        let chorale = compose_with_spec(&intent, &spec);
        spec.rhetoric = crate::spec::PhraseRhetoric::Classic;
        let classic = compose_with_spec(&intent, &spec);

        let cad_classic: Vec<&ScoreNote> = classic
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Cadential)
            .collect();
        let cad_chorale: Vec<&ScoreNote> = chorale
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody && n.emphasis == Emphasis::Cadential)
            .collect();
        assert!(!cad_classic.is_empty());
        assert_eq!(cad_classic.len(), cad_chorale.len());

        let mut held_longer = 0;
        for (c, h) in cad_classic.iter().zip(cad_chorale.iter()) {
            assert!(
                (c.onset.beats() - h.onset.beats()).abs() < 1e-9,
                "a chorale cadence arrives exactly on time -- the onset must not move"
            );
            assert!(
                h.velocity + 1e-6 >= c.velocity,
                "a chorale cadence must not soften below its classic twin ({} < {})",
                h.velocity,
                c.velocity
            );
            if h.duration.beats() > c.duration.beats() + 1e-6 {
                held_longer += 1;
            }
        }
        assert!(
            held_longer >= 1,
            "at least one cadence must be held longer than its classic twin"
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
            false,
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
            false,
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
            false,
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
            false,
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
            false,
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

    #[test]
    fn montuno_tumbao_bass_never_lands_on_the_montuno_stabs() {
        // Isolated: one bar of each clave side, bass in isolation against
        // the exact onsets `Accompaniment::Montuno` is known to use.
        let key = Key::major(crate::pitch::PitchClass::C);
        let intent = MusicalIntent::default();
        let bass_onsets_for = |start_measure: i64| -> Vec<f64> {
            let mut score = Score::new(key, 100.0, 4);
            let mut prev_bass = None;
            realize_bass_measures(
                &mut score,
                &[1],
                key,
                4.0,
                &intent,
                &mut prev_bass,
                start_measure,
                |_| true,
                1.0,
                crate::accompaniment::Accompaniment::Montuno,
                false,
            );
            let bar_start = start_measure as f64 * 4.0;
            score
                .voice(VoiceRole::Bass)
                .iter()
                .map(|n| n.onset.beats() - bar_start)
                .collect()
        };
        let three_side = bass_onsets_for(0);
        assert_eq!(three_side.len(), 3);
        for &b in &three_side {
            for &h in &[0.0, 1.5, 3.0] {
                assert!(
                    (b - h).abs() > 1e-6,
                    "tumbao onset {b} collides with the three-side montuno onset {h}"
                );
            }
        }
        let two_side = bass_onsets_for(1);
        assert_eq!(two_side.len(), 3);
        for &b in &two_side {
            for &h in &[1.0, 2.0] {
                assert!(
                    (b - h).abs() > 1e-6,
                    "tumbao onset {b} collides with the two-side montuno onset {h}"
                );
            }
        }
    }

    #[test]
    fn afro_cuban_bass_and_harmony_never_share_an_onset_in_the_same_bar() {
        // Full pipeline: a real composed piece must keep the interlock the
        // isolated unit test above proves in miniature.
        let spec = crate::style::Style::AfroCuban.spec();
        assert_eq!(
            spec.accompaniment_pool,
            vec![crate::accompaniment::Accompaniment::Montuno]
        );
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert!(!s.notes.is_empty(), "a real piece must come out");
        let meter = s.meter as f64;
        // The interlock is a property of the montuno/tumbao MECHANISM
        // (`realize_harmony`/`realize_bass` under `Accompaniment::Montuno`),
        // not of the whole rendered piece: the generic intro pedal
        // (`prepend_intro`'s variant 1) and the generic plagal coda
        // (`append_coda`) both deliberately put a bass note under a
        // harmony chord's own onset for EVERY style — that's a normal,
        // desired doubling (a pedal tone, a cadential root), not a
        // violation of anything. Scope the check to the main body, where
        // the pattern is actually what's driving both voices.
        let intro_bars = spec.texture.intro_bars as i64;
        let total_bars = (s.total_beats.beats() / meter).round() as i64;
        let body_end = total_bars - CODA_BARS;
        // Grouped by ABSOLUTE bar index, not folded across the whole piece:
        // the clave alternates side by bar, so a bass onset that's fine on
        // a two-side bar can legitimately equal a harmony onset that only
        // ever occurs on a three-side bar — those two never actually sound
        // together. What must never happen is a collision WITHIN one bar.
        let by_bar = |notes: &[ScoreNote]| -> std::collections::HashMap<i64, Vec<i64>> {
            let mut m: std::collections::HashMap<i64, Vec<i64>> = std::collections::HashMap::new();
            for n in notes {
                let b = n.onset.beats();
                let bar_idx = (b / meter).floor() as i64;
                if bar_idx < intro_bars || bar_idx >= body_end {
                    continue; // outside the montuno/tumbao's own territory
                }
                let frac = ((b - bar_idx as f64 * meter) * 1000.0).round() as i64;
                m.entry(bar_idx).or_default().push(frac);
            }
            m
        };
        let harmony: Vec<ScoreNote> = s.voice(VoiceRole::Harmony);
        let bass: Vec<ScoreNote> = s.voice(VoiceRole::Bass);
        let harmony_by_bar = by_bar(&harmony);
        let bass_by_bar = by_bar(&bass);
        assert!(
            !bass_by_bar.is_empty(),
            "the tumbao must produce bass notes"
        );
        for (bar_idx, bass_fracs) in &bass_by_bar {
            let Some(harmony_fracs) = harmony_by_bar.get(bar_idx) else {
                continue;
            };
            for &bf in bass_fracs {
                assert!(
                    !harmony_fracs.contains(&bf),
                    "bar {bar_idx}: bass and harmony share onset {bf} — \
                     the montuno/tumbao interlock must prevent this"
                );
            }
        }
    }

    #[test]
    fn compas_gait_bass_anchor_never_lands_on_a_compas_stab() {
        // Isolated: one 12-beat bar. The bass anchor (onset 0, duration 2)
        // must clear every one of the compás's own onsets (2,5,7,9,11).
        let key = Key::major(crate::pitch::PitchClass::C);
        let intent = MusicalIntent::default();
        let mut score = Score::new(key, 100.0, 12);
        let mut prev_bass = None;
        realize_bass_measures(
            &mut score,
            &[1],
            key,
            12.0,
            &intent,
            &mut prev_bass,
            0,
            |_| true,
            1.0,
            crate::accompaniment::Accompaniment::CompasGait,
            false,
        );
        let bass = score.voice(VoiceRole::Bass);
        assert_eq!(bass.len(), 1, "one grounding anchor per cycle");
        let n = &bass[0];
        assert_eq!(n.onset.beats(), 0.0);
        for &stab in &[2.0, 5.0, 7.0, 9.0, 11.0] {
            assert!(
                !(n.onset.beats() <= stab && stab < (n.onset + n.duration).beats()),
                "bass anchor (0..{}) must not sustain through the compás stab at {stab}",
                n.duration.beats()
            );
        }
    }

    #[test]
    fn flamenco_composes_a_real_twelve_beat_compas_piece() {
        // Full pipeline: the compás must reach a real composed piece, not
        // just the isolated pattern.
        let spec = crate::style::Style::Flamenco.spec();
        assert_eq!(
            spec.accompaniment_pool,
            vec![crate::accompaniment::Accompaniment::CompasGait]
        );
        assert_eq!(spec.meter, 12);
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert!(!s.notes.is_empty(), "a real piece must come out");
        assert_eq!(s.meter, 12);
        // The harmony's onsets, folded into the 12-beat cycle, must be
        // exactly the compás's five counted beats — no more, no less —
        // proving the cell reached the real pipeline unmodified.
        let meter = 12.0;
        let fracs: std::collections::BTreeSet<i64> = s
            .voice(VoiceRole::Harmony)
            .iter()
            .map(|n| ((n.onset.beats().rem_euclid(meter)) * 1000.0).round() as i64)
            .collect();
        for expected in [2000, 5000, 7000, 9000, 11000] {
            assert!(
                fracs.contains(&expected),
                "expected a compás stab at beat {} within the bar, got {:?}",
                expected as f64 / 1000.0,
                fracs
            );
        }
    }

    #[test]
    fn bossa_comp_bass_holds_then_anticipates_the_next_bar() {
        // Isolated: one bar. Root through beat 3, a soft push right at the
        // bar's edge — never crossing into the next bar.
        let key = Key::major(crate::pitch::PitchClass::C);
        let intent = MusicalIntent::default();
        let mut score = Score::new(key, 100.0, 4);
        let mut prev_bass = None;
        realize_bass_measures(
            &mut score,
            &[1],
            key,
            4.0,
            &intent,
            &mut prev_bass,
            0,
            |_| true,
            1.0,
            crate::accompaniment::Accompaniment::BossaComp,
            false,
        );
        let bass = score.voice(VoiceRole::Bass);
        assert_eq!(bass.len(), 2, "one hold, one anticipation push");
        let hold = bass.iter().find(|n| n.onset.beats() == 0.0).unwrap();
        assert_eq!(hold.duration.beats(), 3.0);
        let push = bass
            .iter()
            .find(|n| (n.onset.beats() - 3.0).abs() < 1e-9)
            .unwrap();
        assert_eq!(push.duration.beats(), 1.0);
        assert!(
            (push.onset + push.duration).beats() <= 4.0 + 1e-9,
            "the push must never cross into the next bar"
        );
        assert!(
            push.velocity < hold.velocity,
            "the push must be softer, not louder"
        );
    }

    #[test]
    fn bossa_nova_composes_with_zero_gap_floating_harmony() {
        let spec = crate::style::Style::BossaNova.spec();
        assert_eq!(
            spec.accompaniment_pool,
            vec![crate::accompaniment::Accompaniment::BossaComp]
        );
        assert!(
            spec.texture.seventh_chords,
            "bossa's floating harmony is jazz-extended"
        );
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert!(!s.notes.is_empty(), "a real piece must come out");
        // Within any single bar, the harmony's spans must chain with zero
        // gaps — the "floating" identity reaching the real pipeline, not
        // just the isolated pattern.
        let meter = s.meter as f64;
        let mut by_bar: std::collections::HashMap<i64, Vec<(f64, f64)>> =
            std::collections::HashMap::new();
        for n in s.voice(VoiceRole::Harmony) {
            let b = n.onset.beats();
            let bar_idx = (b / meter).floor() as i64;
            let local = b - bar_idx as f64 * meter;
            by_bar
                .entry(bar_idx)
                .or_default()
                .push((local, local + n.duration.beats()));
        }
        let mut checked_a_full_bar = false;
        for spans in by_bar.values() {
            let mut sorted = spans.clone();
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            sorted.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
            if sorted.len() < 3 {
                continue; // a thinned/partial bar (e.g. intro, departure texture)
            }
            checked_a_full_bar = true;
            for pair in sorted.windows(2) {
                assert!(
                    (pair[0].1 - pair[1].0).abs() < 1e-9,
                    "gap or overlap between {:?} and {:?} in a full bossa bar",
                    pair[0],
                    pair[1]
                );
            }
        }
        assert!(
            checked_a_full_bar,
            "must find at least one full 3-stab bar to verify the chain"
        );
    }

    #[test]
    fn roll_ornament_expands_a_long_note_into_five_notes_preserving_total_duration() {
        let key = Key::major(crate::pitch::PitchClass::C);
        let mut score = Score::new(key, 100.0, 4);
        let onset = Duration::zero();
        let dur = Duration::new(2, 1); // a half note — well above the 0.9-beat threshold
        score.push(ScoreNote {
            pitch: key.scale().degree_pitch(3, 4),
            onset,
            duration: dur,
            velocity: 0.6,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        let dna = crate::spec::MelodicDna {
            ornament_rate: 1.0,
            ..Default::default()
        };
        apply_roll_ornaments(&mut score, key, &dna, 1);
        let mut mel: Vec<ScoreNote> = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .copied()
            .collect();
        mel.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
        assert_eq!(mel.len(), 5, "a roll must expand into exactly 5 notes");
        // Pattern: main, upper, main, lower, main.
        assert_eq!(mel[0].pitch, mel[2].pitch);
        assert_eq!(mel[2].pitch, mel[4].pitch);
        assert!(
            mel[1].pitch.midi() > mel[0].pitch.midi(),
            "the second note must be an upper neighbor"
        );
        assert!(
            mel[3].pitch.midi() < mel[0].pitch.midi(),
            "the fourth note must be a lower neighbor"
        );
        // Total duration preserved exactly — pure surface elaboration.
        let total: f64 = mel.iter().map(|n| n.duration.beats()).sum();
        assert!((total - dur.beats()).abs() < 1e-6);
        // The chain tiles with zero gaps, starting exactly at the original onset.
        assert_eq!(mel[0].onset, onset);
        for pair in mel.windows(2) {
            assert!((pair[0].onset + pair[0].duration).beats() - pair[1].onset.beats() < 1e-9);
        }
        // Cuts (indices 1 and 3) are unaccented relative to the mains.
        assert!(mel[1].velocity < mel[0].velocity);
        assert!(mel[3].velocity < mel[0].velocity);
    }

    #[test]
    fn roll_ornament_leaves_short_notes_and_the_climax_untouched() {
        let key = Key::major(crate::pitch::PitchClass::C);
        let mut score = Score::new(key, 100.0, 4);
        // Short note (below threshold).
        score.push(ScoreNote {
            pitch: key.scale().degree_pitch(3, 4),
            onset: Duration::zero(),
            duration: Duration::eighth(),
            velocity: 0.6,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
        // Long note but marked Climax.
        score.push(ScoreNote {
            pitch: key.scale().degree_pitch(5, 4),
            onset: Duration::new(1, 1),
            duration: Duration::new(2, 1),
            velocity: 0.6,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Climax,
            section_intensity: 1.0,
        });
        let dna = crate::spec::MelodicDna {
            ornament_rate: 1.0,
            ..Default::default()
        };
        apply_roll_ornaments(&mut score, key, &dna, 1);
        let mel_count = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .count();
        assert_eq!(
            mel_count, 2,
            "neither the short note nor the climax should be expanded"
        );
    }

    #[test]
    fn irish_traditional_composes_with_a_real_roll_chain() {
        let spec = crate::style::Style::IrishTraditional.spec();
        assert!(spec.texture.roll_ornaments);
        assert_eq!(spec.meter, 4);
        // Try a handful of seeds — the roll is probabilistic
        // (`ornament_rate: 0.4`), so at least one should land.
        let mut found_roll = false;
        for seed in 0..8u64 {
            let intent = MusicalIntent {
                seed,
                ..MusicalIntent::default()
            };
            let s = compose_with_spec(&intent, &spec);
            assert!(!s.notes.is_empty(), "a real piece must come out");
            let mut mel: Vec<ScoreNote> = s.voice(VoiceRole::Melody);
            mel.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
            for w in mel.windows(5) {
                let contiguous = w.windows(2).all(|p| {
                    ((p[0].onset + p[0].duration).beats() - p[1].onset.beats()).abs() < 1e-6
                });
                if contiguous && w[0].pitch == w[2].pitch && w[2].pitch == w[4].pitch {
                    let upper = w[1].pitch.midi() > w[0].pitch.midi();
                    let lower = w[3].pitch.midi() < w[0].pitch.midi();
                    if upper && lower {
                        found_roll = true;
                        break;
                    }
                }
            }
            if found_roll {
                break;
            }
        }
        assert!(
            found_roll,
            "at least one real 5-note roll chain must reach a full composed piece"
        );
    }

    #[test]
    fn full_drone_replaces_harmony_with_only_tonic_and_fifth_pitch_classes() {
        let key = Key::major(crate::pitch::PitchClass::C);
        let mut score = Score::new(key, 100.0, 4);
        let bar = Duration::new(4, 1);
        // Seed a real, varied chord progression across 6 bars — exactly
        // what the pass must wipe out.
        for (m, &deg) in [1, 4, 5, 6, 2, 1].iter().enumerate() {
            for pitch in key.diatonic_triad(deg).voice(3) {
                score.push(ScoreNote {
                    pitch,
                    onset: bar.scale(m as i64, 1),
                    duration: bar,
                    velocity: 0.4,
                    role: VoiceRole::Harmony,
                    emphasis: Emphasis::Normal,
                    section_intensity: 1.0,
                });
            }
            score.push(ScoreNote {
                pitch: key.diatonic_triad(deg).voice(2)[0],
                onset: bar.scale(m as i64, 1),
                duration: bar,
                velocity: 0.4,
                role: VoiceRole::Bass,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        apply_full_drone(&mut score, key);
        let tonic_pc = key.tonic;
        let fifth_pc = key.scale().degree_pitch_class(5);
        let harmony = score.voice(VoiceRole::Harmony);
        assert!(!harmony.is_empty(), "a drone pad must exist");
        for n in &harmony {
            let pc = n.pitch.pitch_class();
            assert!(
                pc == tonic_pc || pc == fifth_pc,
                "drone harmony must only ever sound the tonic or the fifth, found {pc:?}"
            );
        }
        // Stasis-tying must have collapsed the 6 identical bars into far
        // fewer, longer notes — not 6 re-struck chords.
        assert!(
            harmony.len() < 6 * 3,
            "identical repeated bars must be tied into long sustains, not re-struck every bar"
        );
        let bass = score.voice(VoiceRole::Bass);
        for n in &bass {
            let pc = n.pitch.pitch_class();
            assert!(pc == tonic_pc || pc == fifth_pc);
        }
    }

    #[test]
    fn hindustani_composes_with_zero_harmonic_movement() {
        let spec = crate::style::Style::HindustaniInspired.spec();
        assert!(spec.texture.full_drone);
        let s = compose_with_spec(&MusicalIntent::default(), &spec);
        assert!(!s.notes.is_empty(), "a real piece must come out");
        let tonic_pc = s.key.tonic;
        let fifth_pc = s.key.scale().degree_pitch_class(5);
        let harmony_pcs: std::collections::HashSet<crate::pitch::PitchClass> = s
            .voice(VoiceRole::Harmony)
            .iter()
            .map(|n| n.pitch.pitch_class())
            .collect();
        assert!(
            harmony_pcs
                .iter()
                .all(|pc| *pc == tonic_pc || *pc == fifth_pc),
            "a real composed piece must never modulate — no pitch class outside \
             tonic/fifth may ever appear in Harmony, got {harmony_pcs:?}"
        );
    }

    #[test]
    fn plan_preserving_sonata_entry_matches_canonical_composer() {
        let intent = MusicalIntent {
            seed: 31,
            bars: 4,
            ..MusicalIntent::default()
        };
        let spec = crate::style::Style::Sonata.spec();
        let planned = compose_sonata_with_plan(&intent, &spec).unwrap();
        let canonical = compose_with_spec(&intent, &spec);
        assert_eq!(planned.score, canonical);
        assert!(planned.resolution.all_fulfilled());
    }

    #[test]
    fn plan_preserving_entry_rejects_non_sonata_forms() {
        let intent = MusicalIntent::default();
        let spec = crate::style::Style::Classical.spec();
        assert!(compose_sonata_with_plan(&intent, &spec).is_none());
    }
}
