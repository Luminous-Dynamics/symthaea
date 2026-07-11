// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CompositionSpec: the composer's aesthetic space as USER-OWNED data.
//!
//! Everything that used to be a hard-coded architectural decision — motif
//! banks, progression source, accompaniment textures, large-scale forms,
//! ensembles, meter, tempo, texture/ornament policies — lives in one
//! declarative, serializable structure. The five built-in [`Style`]s are
//! nothing more than five preset values of this type ([`Style::spec`]);
//! a user can author their own spec (JSON in Muse Studio, or Rust) and get
//! the same engine with THEIR aesthetics.
//!
//! The division of power is deliberate:
//! - the SPEC owns every *choice* (which motifs, which textures, how the
//!   piece is allowed to breathe);
//! - the ENGINE owns every *invariant* (voice leading, cadence grammar,
//!   collision avoidance, the superset-only substitution rules). A spec can
//!   reshape the aesthetic space completely, but it cannot make the engine
//!   write a wrong note.
//! [`CompositionSpec::validate`] enforces the boundary: malformed specs are
//! rejected with human-readable reasons, never silently "fixed".
//!
//! [`Style`]: crate::style::Style

use crate::accompaniment::Accompaniment;
use crate::harmony::Progression;
use crate::motif::Motif;
use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};

/// One motif-template note: (scale degree, duration numerator, denominator).
/// Rational beats, exactly as [`Duration::new`] takes them.
pub type SpecNote = (i32, i64, i64);

/// Where the chord progression comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressionSpec {
    /// The functional-harmony grammar generator ([`Progression::generate`]).
    Grammar,
    /// A fixed archetype of scale degrees, cycled to the requested length.
    Archetype(Vec<i32>),
}

/// Large-scale forms the seed may choose between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormKind {
    /// A–B–A′: one departure, one return.
    Ternary,
    /// A–B–A–C–A: two contrasting episodes, each framed by the theme.
    Rondo,
    /// Theme–minore–figuration–theme: every section keeps the theme's
    /// harmonic skeleton (unlike ternary/rondo's fresh episode
    /// progressions); only the surface varies. Variation is structured
    /// remembering — see [`crate::form::Form::variations`].
    Variations,
    /// A three-voice fughetta: exposition (answer at the fifth, derived
    /// countersubject), sequential episodes, inverted middle entry,
    /// stretto, augmented final entry. Bypasses the period/accompaniment
    /// pipeline entirely — a fugue's texture IS its counterpoint. See
    /// [`crate::fugue`].
    Fugue,
}

/// Drum policy, interpreted by the renderer (the symbolic layer only names
/// the intent; kick/snare/hat synthesis is muse's business).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrumPolicy {
    /// No kit (chamber music).
    None,
    /// Kick on the barline + soft hats on beats.
    LightPulse,
    /// A sparse kick on each barline only.
    BarPulse,
    /// Full backbeat: kick 1 & 3, snare 2 & 4, eighth-note hats.
    Backbeat,
}

/// A compositional ATTITUDE — an emotional posture deeper than a style
/// preset. The listening review that demanded it: "if I listened to ten
/// Muse pieces in a row... they all inhabit a similar emotional world:
/// thoughtful, bittersweet, careful, restrained... teach Muse to inhabit
/// different emotional behaviors. Those are deeper than style presets.
/// They're compositional attitudes." Each attitude bundles tempo/texture
/// dials with a SIGNATURE DEVICE the default temperament never uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Attitude {
    /// Little rhythmic questions, an unresolved ending: the final melody
    /// tone resolves UP to degree 2 — question intonation — and the
    /// piece leaves without answering.
    Curiosity,
    /// Notated syncopation (interior downbeats arrive an eighth EARLY,
    /// held through where they belonged), an assertive bass. Insistence,
    /// not reflection.
    Defiance,
    /// Lighter, detached articulation and a quicker pulse — lift.
    Joy,
    /// Suspension chains at chord changes (the classical grammar of
    /// grief: a tone held over from the old harmony, dissonant against
    /// the new, resolving down by step), sparse block accompaniment, a
    /// slower pulse.
    Grief,
}

/// Texture and ornament policies — the "how is this piece allowed to
/// breathe" switches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureSpec {
    /// Strip the departure (B) section to melody+bass for its first half.
    pub thin_departure: bool,
    /// Add the voice-led, collision-avoided second line in return sections.
    pub counter_melody: bool,
    /// One acciaccatura into the piece's climax.
    pub climax_grace: bool,
    /// Split cadence-approach bars into triad → seventh (4/4 only).
    pub cadential_harmonic_rhythm: bool,
    /// Plagal coda length in bars (0 = end on the final cadence).
    pub coda_bars: u8,
    /// Renderer drum policy.
    pub drums: DrumPolicy,
    /// Swing: the position of the off-beat eighth within its beat.
    /// 0.5 = straight (the default), 0.6 = light shuffle, 2/3 ≈ 0.667 =
    /// triplet swing. Applied at the PERFORMANCE layer (notation stays
    /// straight, the player swings — exactly how real jazz charts work),
    /// to melody, harmony, bass, counter-melody and drums together.
    #[serde(default = "default_swing")]
    pub swing: f32,
    /// Bars of accompaniment alone before the melody enters (0 = the
    /// melody starts cold on beat one). Part of the arrangement
    /// dramaturgy wave: a listener's first impression should be an
    /// invitation, not three voices already mid-sentence.
    #[serde(default = "default_intro_bars")]
    pub intro_bars: u8,
    /// Staged entrances and exits: the bass sits out the opening phrase
    /// and arrives with the consequent; the bar before every return is
    /// thinned (bass out, harmony out for its second half) so the
    /// reprise lands as an ARRIVAL. The single biggest "sounds like a
    /// piece, not wallpaper" lever found by listening.
    #[serde(default = "default_true")]
    pub staged_entrances: bool,
    /// Held arrivals: mid-piece phrase cadences hold their final tone
    /// through the bar (the question hangs while the accompaniment
    /// answers), and the climax note absorbs the note after it so the
    /// peak is DWELT ON, not passed through. Gives the melody the
    /// silence and the one big moment continuous motion never has.
    #[serde(default = "default_true")]
    pub held_arrivals: bool,
    /// The damage pass (0.0 = pristine, 1.0 = full): after the clean
    /// composition, deliberately injure it in musically meaningful ways —
    /// the listener-review verdict on the undamaged output was "it has
    /// shape, but it lacks argument; motion without consequence." Rising
    /// thresholds add devices: ≥0.2 exposes the climax bar (accompaniment
    /// cut, the peak stands alone) and transforms the coda (the opening
    /// motif returns changed — augmented, an octave down, over a borrowed
    /// mixture chord — instead of merely receding); ≥0.35 darkens the
    /// bass entrance an octave and removes the downbeat the ear expects
    /// in the return's second bar; ≥0.5 inserts one chromatic "wait"
    /// passing tone and makes the counterline rhythmically disagree.
    #[serde(default = "default_damage")]
    pub damage: f32,
    /// Hook surgery: generate a tiny memorable cell (3-5 notes with
    /// enforced rhythmic + contour identity — see [`crate::hook`]) BEFORE
    /// the melody, and graft it as the head of the bar-motif. The phrase
    /// gets a NAME the development machinery then restates, fragments,
    /// and (in the transformed coda) remembers. The review that demanded
    /// it: "the melody still feels more like a generated line than a
    /// remembered theme... can it make a theme worth injuring?"
    #[serde(default = "default_true")]
    pub hook_cell: bool,
    /// A subtle new instrumental color at the final return — sparse
    /// chord-top sparkles an octave up on a contrasting timbre. The
    /// listening review: "the palette stays fairly constant... imagine if
    /// the final return introduced a subtle new color — the listener
    /// subconsciously feels 'we're somewhere different now.'"
    #[serde(default = "default_true")]
    pub return_color: bool,
}

fn default_swing() -> f32 {
    0.5
}

fn default_intro_bars() -> u8 {
    2
}

fn default_damage() -> f32 {
    0.5
}

fn default_true() -> bool {
    true
}

/// The full declarative spec. See the module docs for the philosophy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositionSpec {
    pub name: String,
    /// The piece's emotional posture. `None` (the default) keeps the
    /// engine's native temperament — reflective, restrained. Setting an
    /// [`Attitude`] changes how the piece BEHAVES, not just its dials.
    #[serde(default)]
    pub attitude: Option<Attitude>,
    /// Mode override. `None` (the default) keeps the valence mapping:
    /// positive intent → major, negative → (harmonic) minor. Setting a mode
    /// pins the key's tonality REGARDLESS of valence — `Dorian`,
    /// `Phrygian`, `Lydian`, `Mixolydian`, and `Aeolian` give genuinely
    /// modal harmony (mode-native cadences, no borrowed leading tone);
    /// `Ionian`/`HarmonicMinor` pin plain major/minor. The grammar-
    /// incompatible modes (Locrian, pentatonics, whole-tone, melodic
    /// minor) are rejected by `validate()`.
    #[serde(default)]
    pub mode: Option<crate::scale::Mode>,
    /// Beats per bar (2–12 validated; 3 and 4 are the well-trodden paths).
    pub meter: u8,
    /// BPM at arousal 0 and arousal 1.
    pub tempo_range: (f32, f32),
    /// Motif templates by arousal tier (calm < 0.33 ≤ medium < 0.66 ≤ busy).
    /// Every template must total exactly `meter` beats.
    pub motifs_calm: Vec<Vec<SpecNote>>,
    pub motifs_medium: Vec<Vec<SpecNote>>,
    pub motifs_busy: Vec<Vec<SpecNote>>,
    pub progression: ProgressionSpec,
    /// Accompaniment textures the seed picks from (non-empty).
    pub accompaniment_pool: Vec<Accompaniment>,
    /// Forms the seed picks from (non-empty).
    pub form_pool: Vec<FormKind>,
    /// Instrument-name triples (melody, harmony, bass) the seed picks from.
    /// Names are resolved by the renderer; unrecognized names fall back to
    /// its defaults LOUDLY, never silently.
    pub ensemble_pool: Vec<[String; 3]>,
    /// The counter-melody's own voice. `None` (the default) lets the
    /// renderer pick a timbral CONTRAST to the melody and bass — a
    /// counterline played by the bass instrument doesn't separate as its
    /// own emotional line (listening-review finding). Set a name here to
    /// choose explicitly; unrecognized names fall back loudly.
    #[serde(default)]
    pub counter_instrument: Option<String>,
    pub texture: TextureSpec,
}

impl CompositionSpec {
    /// Reject malformed specs with every problem listed (not just the
    /// first). The engine's invariants (voice leading, cadences, collision
    /// rules) are enforced downstream regardless — validation covers the
    /// spec-shaped mistakes a hand-edited JSON can make.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if !(2..=12).contains(&self.meter) {
            errors.push(format!("meter {} outside 2..=12", self.meter));
        }
        let (lo, hi) = self.tempo_range;
        if !(20.0..=400.0).contains(&lo) || !(20.0..=400.0).contains(&hi) || lo > hi {
            errors.push(format!("tempo_range ({lo}, {hi}) not a sane BPM range"));
        }
        for (tier, bank) in [
            ("calm", &self.motifs_calm),
            ("medium", &self.motifs_medium),
            ("busy", &self.motifs_busy),
        ] {
            if bank.is_empty() {
                errors.push(format!("motifs_{tier} is empty"));
            }
            for (i, template) in bank.iter().enumerate() {
                if template.is_empty() {
                    errors.push(format!("motifs_{tier}[{i}] has no notes"));
                    continue;
                }
                let mut total = Duration::zero();
                let mut bad = false;
                for &(_deg, num, den) in template {
                    if den == 0 || num <= 0 {
                        errors.push(format!(
                            "motifs_{tier}[{i}] has a non-positive duration {num}/{den}"
                        ));
                        bad = true;
                        break;
                    }
                    total = total + Duration::new(num, den);
                }
                if !bad && (total.beats() - self.meter as f64).abs() > 1e-9 {
                    errors.push(format!(
                        "motifs_{tier}[{i}] totals {} beats, meter is {}",
                        total.beats(),
                        self.meter
                    ));
                }
            }
        }
        if let ProgressionSpec::Archetype(degs) = &self.progression {
            if degs.is_empty() {
                errors.push("progression archetype is empty".into());
            }
            for &d in degs {
                if !(1..=7).contains(&d) {
                    errors.push(format!("progression degree {d} outside 1..=7"));
                }
            }
        }
        if self.accompaniment_pool.is_empty() {
            errors.push("accompaniment_pool is empty".into());
        }
        if self.form_pool.is_empty() {
            errors.push("form_pool is empty".into());
        }
        if self.ensemble_pool.is_empty() {
            errors.push("ensemble_pool is empty".into());
        }
        if let Some(name) = &self.counter_instrument
            && name.trim().is_empty()
        {
            errors.push("counter_instrument is an empty string (use null for auto)".into());
        }
        if let Some(mode) = self.mode
            && crate::harmony::Key::modal(crate::pitch::PitchClass::new(0), mode).is_none()
        {
            errors.push(format!(
                "mode {mode:?} is not usable for composition (needs 7 degrees and a stable tonic \
                 triad — use Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, or \
                 HarmonicMinor)"
            ));
        }
        if self.texture.coda_bars > 8 {
            errors.push(format!("coda_bars {} > 8", self.texture.coda_bars));
        }
        if !(0.5..=0.75).contains(&self.texture.swing) {
            errors.push(format!(
                "swing {} outside 0.5 (straight) ..= 0.75 (hard shuffle)",
                self.texture.swing
            ));
        }
        if !(0.0..=1.0).contains(&self.texture.damage) {
            errors.push(format!(
                "damage {} outside 0.0 (pristine) ..= 1.0 (full damage pass)",
                self.texture.damage
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Motif for an arousal tier and seed, with the same 4-orientation
    /// variety multiplication the built-in banks always had.
    pub fn motif(&self, arousal: f32, seed: u64) -> Motif {
        let bank = if arousal < 0.33 {
            &self.motifs_calm
        } else if arousal < 0.66 {
            &self.motifs_medium
        } else {
            &self.motifs_busy
        };
        let idx = (seed % bank.len() as u64) as usize;
        let notes: Vec<(i32, Duration)> = bank[idx]
            .iter()
            .map(|&(deg, num, den)| (deg, Duration::new(num, den)))
            .collect();
        let picked = Motif::from_degrees(&notes);
        let orientation = (seed / bank.len() as u64) % 4;
        crate::form::oriented(&picked, orientation)
    }

    /// A progression `bars` measures long, from the grammar or the cycled
    /// archetype (same length contract as [`crate::style::Style`] had).
    pub fn progression(&self, bars: usize, seed: u64) -> Progression {
        let bars = bars.max(1);
        match &self.progression {
            ProgressionSpec::Grammar => Progression::generate(bars, seed),
            ProgressionSpec::Archetype(degrees) => {
                let cycled: Vec<i32> = (0..bars).map(|i| degrees[i % degrees.len()]).collect();
                Progression::new(cycled)
            }
        }
    }

    /// Accompaniment texture by seed (`seed / 2` decorrelates from the form
    /// pick, as the style system established).
    pub fn accompaniment(&self, seed: u64) -> Accompaniment {
        self.accompaniment_pool[(seed as usize / 2) % self.accompaniment_pool.len()]
    }

    /// Large-scale form by seed.
    pub fn form_kind(&self, seed: u64) -> FormKind {
        self.form_pool[(seed % self.form_pool.len() as u64) as usize]
    }

    /// Ensemble instrument names by seed.
    pub fn ensemble(&self, seed: u64) -> &[String; 3] {
        &self.ensemble_pool[(seed % self.ensemble_pool.len() as u64) as usize]
    }

    /// Tempo for an arousal, interpolated across `tempo_range`.
    pub fn tempo(&self, arousal: f32) -> f32 {
        let (lo, hi) = self.tempo_range;
        lo + arousal.clamp(0.0, 1.0) * (hi - lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Style;

    #[test]
    fn every_builtin_style_spec_validates() {
        for style in [
            Style::Classical,
            Style::Waltz,
            Style::Folk,
            Style::Cinematic,
            Style::Playful,
        ] {
            let spec = style.spec();
            assert_eq!(
                spec.validate(),
                Ok(()),
                "{style:?}'s preset spec must validate"
            );
        }
    }

    #[test]
    fn validation_catches_the_hand_edit_mistakes() {
        let mut spec = Style::Classical.spec();
        spec.meter = 0;
        spec.motifs_calm = vec![vec![(1, 1, 1)]]; // 1 beat, meter says 0
        spec.accompaniment_pool.clear();
        spec.progression = ProgressionSpec::Archetype(vec![9]);
        let errors = spec.validate().unwrap_err();
        let text = errors.join("\n");
        assert!(text.contains("meter"), "{text}");
        assert!(text.contains("accompaniment_pool"), "{text}");
        assert!(text.contains("degree 9"), "{text}");
    }

    #[test]
    fn specs_round_trip_through_json() {
        let spec = Style::Playful.spec();
        let json = serde_json::to_string_pretty(&spec).unwrap();
        let back: CompositionSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn saved_specs_from_before_the_new_fields_still_load() {
        // Users have saved specs on disk (Studio's data/specs/) from before
        // `mode` and `counter_instrument` existed — those JSONs must keep
        // deserializing, with both defaulting to None.
        let mut v = serde_json::to_value(Style::Classical.spec()).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("mode");
        obj.remove("counter_instrument");
        let spec: CompositionSpec = serde_json::from_value(v).unwrap();
        assert_eq!(spec.mode, None);
        assert_eq!(spec.counter_instrument, None);
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn spec_motifs_match_the_style_banks_they_transcribe() {
        // The preset spec's motif picker must agree with the original
        // style-bank picker for every tier and a spread of seeds — this is
        // the byte-identical-preset guarantee at the motif level.
        for style in [
            Style::Classical,
            Style::Waltz,
            Style::Folk,
            Style::Cinematic,
            Style::Playful,
        ] {
            let spec = style.spec();
            for arousal in [0.1f32, 0.5, 0.9] {
                for seed in 0..12u64 {
                    assert_eq!(
                        spec.motif(arousal, seed),
                        style.motif(arousal, seed),
                        "{style:?} arousal {arousal} seed {seed}"
                    );
                }
            }
        }
    }
}
