// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Identity cards: a candidate as a NAME and a handful of honest traits,
//! instead of a seed number.
//!
//! From the candidate-browser review: "the UI should communicate
//! 'discover music,' not 'configure a generator'... those are identities,
//! not seeds" — and, one review later, the premise layer gave every
//! candidate genuinely different premises worth describing. This module
//! turns a (spec, seed) pair into:
//!
//! - a **title**: an evocative two-or-three-word name, deterministically
//!   drawn from word banks whose SELECTION is biased by the identity's
//!   real features (mode color, pacing, texture) — a label to remember a
//!   piece by, not a claim about it;
//! - **traits**: short honest words derived from measurable features —
//!   pacing from the premise's tempo third, texture from its tier, length
//!   from its bars multiplier, mode color, and the hook's own character
//!   (wide-reaching / sighing / insistent, from its interval profile).
//!
//! Everything is deterministic: the same seed under the same spec is the
//! same identity, forever — so a title is a stable name, not a caption
//! that changes on refresh.

use crate::composer::MusicalIntent;
use crate::hook::HookCell;
use crate::premise::{TextureTier, premise_for};
use crate::scale::Mode;
use crate::spec::CompositionSpec;

/// A candidate's card: its name and its honest traits.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct IdentityCard {
    pub title: String,
    pub traits: Vec<String>,
}

/// The broad naming voice used for one generated title.
///
/// Families are presentation choices, not musical claims. The engine only
/// selects families whose wording can be grounded in the resolved form,
/// identity grammar, pacing, mode color, or ending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TitleFamily {
    Image,
    Place,
    Threshold,
    Narrative,
    Structural,
    Formal,
    Minimal,
}

/// Reproducible evidence for a generated title.
///
/// A title remains a poetic label rather than analysis. `source_traits`
/// records which real composition traits constrained the vocabulary and
/// template, while `alternatives` provides deterministic choices without
/// recomposing or changing the piece's identity.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TitleRecipe {
    pub family: TitleFamily,
    pub template_id: String,
    pub source_traits: Vec<String>,
    pub generated_title: String,
    pub alternatives: Vec<String>,
}

const BRIGHT_ADJ: &[&str] = &[
    "Amber", "Morning", "Gilded", "Clear", "Meadow", "Copper", "Daylight", "Open", "Apricot",
    "Lucent", "Sunward", "Honeyed",
];
const SHADOW_ADJ: &[&str] = &[
    "Ember",
    "Winter",
    "Violet",
    "Ashen",
    "Midnight",
    "Hollow",
    "Iron",
    "Quiet",
    "Sable",
    "Duskbound",
    "Smoke",
    "Moonless",
];
const STILL_ADJ: &[&str] = &[
    "Sleeping",
    "Patient",
    "Slow",
    "Deep",
    "Distant",
    "Low",
    "Still",
    "Long",
    "Held",
    "Unhurried",
    "Faint",
    "Tender",
];
const QUICK_ADJ: &[&str] = &[
    "Running",
    "Bright",
    "Restless",
    "Turning",
    "Rising",
    "Sudden",
    "Wild",
    "Sparking",
    "Kinetic",
    "Winged",
    "Forward",
    "Quickening",
];
const OBJECTS: &[&str] = &[
    "Lantern",
    "Compass",
    "Letter",
    "Window",
    "Belltower",
    "Thread",
    "Paper Bird",
    "Clockwork",
    "Mirror",
    "Bell",
    "Map",
    "Key",
    "Vessel",
    "Book",
    "Flame",
    "Name",
];
const PLACES: &[&str] = &[
    "River",
    "Orchard",
    "Harbor",
    "Garden",
    "Stairwell",
    "Meridian",
    "Archway",
    "Field",
    "Shoreline",
    "Attic",
    "Courtyard",
    "Causeway",
    "Low Water",
    "North Room",
    "Threshold",
    "Distant Hill",
];
const WEATHER_AND_LIGHT: &[&str] = &[
    "Rain",
    "Dawn",
    "Dusk",
    "Afterlight",
    "Snow",
    "Thunder",
    "Blue Hour",
    "First Light",
    "Low Sun",
    "Night Air",
    "Mist",
    "Warm Wind",
];
const MEMORY_WORDS: &[&str] = &[
    "Return",
    "Afterimage",
    "Remembrance",
    "Refrain",
    "Trace",
    "Echo",
    "Second Arrival",
    "Familiar Distance",
];
const MOTION_WORDS: &[&str] = &[
    "Crossing",
    "Turning",
    "Ascent",
    "Drift",
    "Procession",
    "Flight",
    "Current",
    "Passage",
];

/// Splitmix-style scramble so nearby seeds don't pick nearby words — the
/// premise arithmetic already uses small divisors of the raw seed.
fn scramble(seed: u64, salt: u64) -> u64 {
    let mut z = seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn pick<'a>(bank: &'a [&'a str], seed: u64, salt: u64) -> &'a str {
    bank[(scramble(seed, salt) as usize) % bank.len()]
}

fn form_label(form: crate::spec::FormKind) -> &'static str {
    use crate::spec::FormKind as F;
    match form {
        F::Ternary => "Ternary",
        F::Rondo => "Rondo",
        F::Variations => "Variations",
        F::Fugue => "Fughetta",
        F::Passacaglia => "Passacaglia",
        F::Erosion => "Erosion Study",
        F::Lineage => "Lineage Study",
        F::ProgSuite => "Suite",
        F::Sonata => "Sonata",
        F::Renaissance => "Ricercar",
        F::Opera => "Scene",
    }
}

fn title_traits(
    spec: &CompositionSpec,
    intent: &MusicalIntent,
    grammar: &str,
    ending: Option<&str>,
) -> Vec<String> {
    let mut traits = Vec::with_capacity(6);
    traits.push(format!(
        "{} form",
        form_label(spec.form_kind(intent.seed)).to_lowercase()
    ));
    traits.push(format!("{grammar} identity"));
    traits.push(if intent.arousal < 0.34 {
        "unhurried motion".to_string()
    } else if intent.arousal < 0.67 {
        "flowing motion".to_string()
    } else {
        "quick motion".to_string()
    });
    traits.push(if intent.valence < -0.15 {
        "shadowed color".to_string()
    } else if intent.valence > 0.15 {
        "bright color".to_string()
    } else {
        "balanced color".to_string()
    });
    if let Some(ending) = ending {
        traits.push(format!("{ending} ending"));
    }
    if spec.meter != 4 {
        traits.push(format!("{}-beat meter", spec.meter));
    }
    traits
}

fn adjective_bank(minorish: bool, tempo_third: usize) -> &'static [&'static str] {
    match (minorish, tempo_third) {
        (_, 0) => STILL_ADJ,
        (true, _) => SHADOW_ADJ,
        (false, 2) => QUICK_ADJ,
        (false, _) => BRIGHT_ADJ,
    }
}

fn structural_title(grammar: &str, ending: Option<&str>, seed: u64) -> String {
    let object = pick(OBJECTS, seed, 31);
    let place = pick(PLACES, seed, 32);
    match (grammar, ending) {
        ("memory", _) => match scramble(seed, 33) % 3 {
            0 => format!("{object}, Remembered"),
            1 => format!("Return to {place}"),
            _ => format!("The Second {object}"),
        },
        ("subject", _) | ("equal voices", _) => match scramble(seed, 33) % 3 {
            0 => format!("Three Voices at {place}"),
            1 => format!("Counterlines for {object}"),
            _ => format!("Voices Across {place}"),
        },
        ("persistence", _) => match scramble(seed, 33) % 3 {
            0 => format!("Beneath the {object}"),
            1 => format!("Ground at {place}"),
            _ => format!("What Remains Below"),
        },
        ("erosion", Some("recovery")) => format!("The {object} Returned"),
        ("erosion", Some("acceptance")) => format!("What {place} Kept"),
        ("erosion", Some("elegy")) => format!("Almost Home at {place}"),
        ("erosion", _) => format!("What the {object} Lost"),
        ("lineage", _) => match scramble(seed, 33) % 3 {
            0 => format!("Descendants of the {object}"),
            1 => format!("The {object} Becomes"),
            _ => format!("From {place}, Another {object}"),
        },
        ("dialogue", _) => match scramble(seed, 33) % 3 {
            0 => format!("Two Letters at {place}"),
            1 => format!("The Interrupted {object}"),
            _ => format!("One Voice Before Another"),
        },
        ("resolution", _) => match scramble(seed, 33) % 3 {
            0 => "The Long Way Home".to_string(),
            1 => format!("Home Through {place}"),
            _ => format!("The {object} in Two Keys"),
        },
        _ => format!("{} {}", pick(MEMORY_WORDS, seed, 34), object),
    }
}

fn title_from_family(
    family: TitleFamily,
    spec: &CompositionSpec,
    intent: &MusicalIntent,
    grammar: &str,
    ending: Option<&str>,
    seed: u64,
) -> (String, &'static str) {
    let minorish = match spec.mode {
        Some(Mode::Aeolian) | Some(Mode::Phrygian) | Some(Mode::HarmonicMinor) => true,
        Some(_) => false,
        None => intent.valence < 0.0,
    };
    let tempo_third = if intent.arousal < 0.34 {
        0
    } else if intent.arousal < 0.67 {
        1
    } else {
        2
    };
    let adjective = pick(adjective_bank(minorish, tempo_third), seed, 11);
    let object = pick(OBJECTS, seed, 12);
    let place = pick(PLACES, seed, 13);
    let light = pick(WEATHER_AND_LIGHT, seed, 14);
    let motion = pick(MOTION_WORDS, seed, 15);
    let memory = pick(MEMORY_WORDS, seed, 16);

    match family {
        TitleFamily::Image => (format!("{adjective} {object}"), "adjective-object-v2"),
        TitleFamily::Place => match scramble(seed, 17) % 3 {
            0 => (format!("{object} at {place}"), "object-at-place-v2"),
            1 => (format!("{light} over {place}"), "light-over-place-v2"),
            _ => (format!("{motion} toward {place}"), "motion-toward-place-v2"),
        },
        TitleFamily::Threshold => match scramble(seed, 18) % 3 {
            0 => (format!("At the {place}"), "at-place-v2"),
            1 => (format!("Beyond the {object}"), "beyond-object-v2"),
            _ => (
                format!("Between {light} and {place}"),
                "between-image-place-v2",
            ),
        },
        TitleFamily::Narrative => match scramble(seed, 19) % 4 {
            0 => (format!("Before the {object}"), "before-object-v2"),
            1 => (format!("After {light}"), "after-light-v2"),
            2 => (format!("When the {object} Opened"), "when-object-v2"),
            _ => (format!("Where {memory} Begins"), "where-memory-begins-v2"),
        },
        TitleFamily::Structural => (
            structural_title(grammar, ending, seed),
            "identity-grammar-v2",
        ),
        TitleFamily::Formal => {
            let subtitle = match scramble(seed, 20) % 3 {
                0 => format!("{adjective} {memory}"),
                1 => format!("{object} at {place}"),
                _ => format!("{light} {motion}"),
            };
            (
                format!("{} — {subtitle}", form_label(spec.form_kind(intent.seed))),
                "grounded-form-subtitle-v2",
            )
        }
        TitleFamily::Minimal => match scramble(seed, 21) % 3 {
            0 => (object.to_string(), "single-image-v2"),
            1 => (
                format!("{memory} {}", 1 + scramble(seed, 22) % 9),
                "named-study-number-v2",
            ),
            _ => (format!("{motion} / {light}"), "paired-minimal-v2"),
        },
    }
}

fn allowed_families(grammar: &str, form: crate::spec::FormKind) -> &'static [TitleFamily] {
    use crate::spec::FormKind as F;
    const GENERAL: &[TitleFamily] = &[
        TitleFamily::Image,
        TitleFamily::Place,
        TitleFamily::Threshold,
        TitleFamily::Narrative,
        TitleFamily::Structural,
        TitleFamily::Minimal,
    ];
    const FORMAL: &[TitleFamily] = &[
        TitleFamily::Image,
        TitleFamily::Place,
        TitleFamily::Structural,
        TitleFamily::Formal,
        TitleFamily::Minimal,
    ];
    if matches!(
        form,
        F::Fugue
            | F::Passacaglia
            | F::Erosion
            | F::Lineage
            | F::ProgSuite
            | F::Sonata
            | F::Renaissance
            | F::Opera
    ) || matches!(
        grammar,
        "subject"
            | "persistence"
            | "erosion"
            | "lineage"
            | "resolution"
            | "equal voices"
            | "dialogue"
    ) {
        FORMAL
    } else {
        GENERAL
    }
}

/// Build a deterministic, composition-aware title recipe.
///
/// The same resolved `(spec, intent, seed, grammar, ending)` always yields
/// the same primary title and alternatives. Alternatives intentionally use
/// different family picks and salted seeds; duplicate strings are removed.
pub fn title_recipe(
    spec: &CompositionSpec,
    intent: &MusicalIntent,
    seed: u64,
    grammar: &str,
    ending: Option<&str>,
) -> TitleRecipe {
    let families = allowed_families(grammar, spec.form_kind(seed));
    let primary_index = (scramble(seed, 70) as usize) % families.len();
    let family = families[primary_index];
    let (generated_title, template_id) =
        title_from_family(family, spec, intent, grammar, ending, seed);

    let mut alternatives = Vec::with_capacity(3);
    for offset in 1..=8u64 {
        let alternative_family = families[(primary_index + offset as usize) % families.len()];
        let (candidate, _) = title_from_family(
            alternative_family,
            spec,
            intent,
            grammar,
            ending,
            seed ^ offset.wrapping_mul(0xA24B_AED4_963E_E407),
        );
        if candidate != generated_title && !alternatives.contains(&candidate) {
            alternatives.push(candidate);
        }
        if alternatives.len() == 3 {
            break;
        }
    }

    TitleRecipe {
        family,
        template_id: template_id.to_string(),
        source_traits: title_traits(spec, intent, grammar, ending),
        generated_title,
        alternatives,
    }
}

/// Build the identity card a seed implies under a spec.
///
/// CONTRACT: `spec` is the BASE spec (style preset or authored), never an
/// already-premised one — the card derives the premise itself, exactly as
/// the exploring composer does, so its traits describe the composition
/// that actually happened. (Passing a premised spec would re-premise it:
/// the first meter-family gate caught cards whose "in three" trait
/// vanished because the comparison baseline had already moved.)
pub fn identity_card(spec: &CompositionSpec, intent: &MusicalIntent, seed: u64) -> IdentityCard {
    let premise = premise_for(spec, seed);
    let resolved_intent = MusicalIntent { seed, ..*intent };
    let (grammar, ending) = crate::composer::identity_grammar_for(&resolved_intent, &premise.spec);
    let title =
        title_recipe(&premise.spec, &resolved_intent, seed, grammar, ending).generated_title;

    // ── The traits (each one earned by a measurable feature) ─────────────
    let mut traits: Vec<String> = Vec::new();
    traits.push(
        match premise.tempo_third {
            0 => "unhurried",
            1 => "flowing",
            _ => "quick",
        }
        .into(),
    );
    traits.push(
        match premise.texture_tier {
            TextureTier::Sparse => "spare",
            TextureTier::Standard => "chamber",
            TextureTier::Full => "full-voiced",
        }
        .into(),
    );
    if premise.bars_multiplier > 1 {
        traits.push("a long arc".into());
    } else {
        traits.push("a statement".into());
    }
    if let Some(mode) = premise.spec.mode {
        traits.push(
            match mode {
                Mode::Dorian => "dorian, wandering",
                Mode::Aeolian => "minor, shadowed",
                Mode::HarmonicMinor => "minor, taut",
                Mode::Lydian => "bright-edged",
                Mode::Mixolydian => "sunlit, modal",
                Mode::Phrygian => "dark, leaning",
                _ => "modal",
            }
            .into(),
        );
    }
    if premise.meter != spec.meter {
        traits.push(match premise.meter {
            3 => "in three".into(),
            5 => "in five".into(),
            m => format!("in {m}"),
        });
    }
    // The hook's own character, from its interval profile.
    let hook = HookCell::generate_with(&premise.spec.melody, seed, premise.spec.meter as f64);
    traits.push(format!("a {} hook", hook_character(&hook)));

    IdentityCard { title, traits }
}

/// A title for ANY piece, premised or not — unlike [`identity_card`],
/// which needs a premise to describe its features FAIRLY (a trait must
/// match what actually got composed), a bare name carries no such claim,
/// so it can read its two inputs straight off the resolved spec/intent
/// instead: `mode`/`valence` for color (identical fallback logic to
/// `identity_card`'s `minorish`), `arousal` bucketed into thirds for
/// motion (the same three-way split [`crate::spec::CompositionSpec::
/// tempo`] itself uses to turn arousal into a real BPM, so "quick" here
/// always agrees with the tempo that actually played). Closes the gap
/// `why_lines` didn't: a Listen-tab piece (never premised — see
/// `muse_studio.rs`'s `exploring` gate) still gets a real name instead of
/// "seed N", not just an explanation of why it sounds the way it does.
pub fn title_for(spec: &CompositionSpec, intent: &MusicalIntent, seed: u64) -> String {
    let resolved_intent = MusicalIntent { seed, ..*intent };
    let (grammar, ending) = crate::composer::identity_grammar_for(&resolved_intent, spec);
    title_recipe(spec, &resolved_intent, seed, grammar, ending).generated_title
}

fn hook_character(hook: &HookCell) -> &'static str {
    let degs: Vec<i32> = hook.notes.iter().map(|(d, _)| *d).collect();
    let max_leap = degs
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .max()
        .unwrap_or(0);
    let repeats = degs.windows(2).any(|w| w[0] == w[1]);
    if max_leap >= 3 {
        "wide-reaching"
    } else if repeats {
        "insistent"
    } else {
        "sighing"
    }
}

/// "Why this piece": 2-6 short, honest sentences translating the REAL
/// mechanisms that composed this specific piece into prose — grammar
/// (form-level identity), development (how the middle section behaves),
/// the named rhythm cell (if any), any notable texture device, and the
/// hook's own melodic character (shared classification with
/// [`identity_card`]'s trait — see [`hook_character`]). Unlike
/// [`identity_card`], this needs no premise and works for EVERY compose
/// (Discovery, a plain Listen-tab piece, or a hand-authored spec) — the
/// facts it reads (`spec.development`, `spec.texture`, `spec.
/// accompaniment_pool`, the hook itself) are always resolved by compose
/// time, premised or not. `grammar`/`ending` are the same values
/// [`crate::composer::identity_grammar_for`] already computes; passed in
/// rather than re-derived so there is exactly one source of truth.
pub fn why_lines(
    spec: &CompositionSpec,
    grammar: &str,
    ending: Option<&str>,
    seed: u64,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(grammar_sentence(grammar, ending));
    if let Some(s) = development_sentence(spec.development) {
        lines.push(s);
    }
    if let Some(pattern) = spec.accompaniment_pool.first() {
        if let Some(s) = accompaniment_sentence(*pattern) {
            lines.push(s);
        }
    }
    lines.push(hook_sentence(spec, seed));
    lines.extend(texture_sentences(&spec.texture));
    lines
}

fn hook_sentence(spec: &CompositionSpec, seed: u64) -> String {
    let hook = HookCell::generate_with(&spec.melody, seed, spec.meter as f64);
    let desc = match hook_character(&hook) {
        "wide-reaching" => "built from bold leaps",
        "insistent" => "leans on a repeated note",
        _ => "moves by smooth, stepwise motion",
    };
    format!("The piece's hook is {} — {desc}.", hook_character(&hook))
}

fn grammar_sentence(grammar: &str, ending: Option<&str>) -> String {
    let base = match grammar {
        "memory" => "Ideas return — and the final return is judged: earned, altered, or complete.",
        "subject" => {
            "One subject is spoken by every voice in turn — development as democracy, not \
             solo development."
        }
        "persistence" => {
            "A repeating ground holds steady while everything above it must keep \
             reinterpreting it."
        }
        "lineage" => {
            "Each return is a recognizable descendant of what came before, not a repetition."
        }
        "erosion" => "The central idea gradually loses itself over the course of the piece.",
        "long form" => {
            "A genuine mid-piece meter change carries the piece through several distinct \
             sections."
        }
        "resolution" => {
            "A second idea is introduced in a foreign key, then earns its way home — real \
             tonal conflict and resolution, not just contrast."
        }
        "equal voices" => {
            "Three independent voices imitate each other as equals — no voice is a \
             privileged \"answer.\""
        }
        "dialogue" => {
            "Two unrelated musical ideas trade turns, then one interrupts the other to close."
        }
        _ => "The piece follows its style's own form.",
    };
    match (grammar, ending) {
        ("erosion", Some("recovery")) => {
            format!("{base} The final return recovers everything that was lost.")
        }
        ("erosion", Some("acceptance")) => {
            format!("{base} Some of what was lost returns; the rest is let go.")
        }
        ("erosion", Some("elegy")) => format!(
            "{base} Recovery seems within reach, then the final return can't quite get back."
        ),
        _ => base.to_string(),
    }
}

fn development_sentence(dev: crate::spec::DevelopmentDna) -> Option<String> {
    use crate::spec::DevelopmentDna as D;
    let s = match dev {
        D::Classic => return None, // the shared default — not worth naming
        D::Sequential => {
            "The middle section transposes the same idea step by step, building through \
             repetition."
        }
        D::Figural => "The middle section elaborates its own figuration progressively as it goes.",
        D::Fragmenting => {
            "The middle section works by breaking its idea into shorter and shorter fragments."
        }
        D::Intensifying => {
            "The music builds steadily toward its climax — register climbs, the texture \
             thickens, and it grows louder."
        }
        D::Wandering => {
            "The middle section wanders — a real, reversible drift away from its starting idea."
        }
    };
    Some(s.into())
}

fn accompaniment_sentence(pattern: crate::accompaniment::Accompaniment) -> Option<String> {
    use crate::accompaniment::Accompaniment as A;
    let s = match pattern {
        A::Block | A::Arpeggio | A::Alberti | A::OomPah | A::Comp => return None, // no strong identity story
        A::Habanera => {
            "The accompaniment moves in a habanera rhythm cell — a dotted anchor, a pickup, \
             two answering beats."
        }
        A::FiveGait => {
            "The accompaniment spells out quintuple meter's 3+2 gait explicitly, instead of \
             just looping arithmetically."
        }
        A::JigGait => {
            "The accompaniment spells out the jig's 3+3 lilt — two mirrored dotted-anchor \
             groups."
        }
        A::Shuffle => "The bass strictly alternates root and fifth in a blues shuffle.",
        A::Montuno => {
            "The accompaniment locks to a real two-bar son clave, alternating a three-stab \
             side with a two-stab side, while the bass interlocks around it without ever \
             landing on the same beat."
        }
        A::CompasGait => {
            "The accompaniment follows an asymmetric 12-beat compás cycle (3+3+2+2+2), \
             hitting only five of the twelve counts."
        }
        A::BossaComp => {
            "The accompaniment's syncopated chords chain together with zero gaps — floating, \
             legato harmony instead of punctuated stabs."
        }
    };
    Some(s.into())
}

fn texture_sentences(t: &crate::spec::TextureSpec) -> Vec<String> {
    let mut out = Vec::new();
    if t.full_drone {
        out.push(
            "The harmony never changes — a single sustained chord underlies the entire piece."
                .into(),
        );
    } else if t.drone {
        out.push(
            "A steady pedal tone grounds the piece while the harmony above it still moves.".into(),
        );
    }
    if t.roll_ornaments {
        out.push(
            "Melodic decorations expand into full five-note flourishes, not just a single \
             grace note."
                .into(),
        );
    }
    if t.harmonic_sequence {
        out.push(
            "The harmony's middle section walks a real sequence — the same shape repeating \
             at a new pitch each time."
                .into(),
        );
    }
    if t.harmonic_stasis {
        out.push(
            "Repeated identical chords are tied into one long sustained note instead of \
             being re-struck."
                .into(),
        );
    }
    if t.seventh_chords {
        out.push(
            "Every chord carries an extended (7th) color, not just the cadential ones.".into(),
        );
    }
    if t.additive_process {
        out.push(
            "The main theme is replaced by a Glass-style additive process — one note, then \
             two, then three, growing and shrinking."
                .into(),
        );
    }
    if t.planing {
        out.push(
            "Chords slide in parallel motion under the melody rather than following \
             independent voice leading."
                .into(),
        );
    }
    if t.deceptive_close {
        out.push(
            "The first arrival avoids resolving — a deceptive close that keeps the tension \
             alive."
                .into(),
        );
    }
    out
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
    fn cards_are_deterministic_stable_names() {
        let spec = Style::Nocturne.spec();
        for seed in [0u64, 7, 69] {
            assert_eq!(
                identity_card(&spec, &intent(seed), seed),
                identity_card(&spec, &intent(seed), seed)
            );
        }
    }

    #[test]
    fn traits_reflect_the_premise_honestly() {
        let spec = Style::Classical.spec();
        for seed in 0..48u64 {
            let p = crate::premise::premise_for(&spec, seed);
            let card = identity_card(&spec, &intent(seed), seed);
            let expect_pace = match p.tempo_third {
                0 => "unhurried",
                1 => "flowing",
                _ => "quick",
            };
            assert!(card.traits.iter().any(|t| t == expect_pace), "seed {seed}");
            let expect_tex = match p.texture_tier {
                TextureTier::Sparse => "spare",
                TextureTier::Standard => "chamber",
                TextureTier::Full => "full-voiced",
            };
            assert!(card.traits.iter().any(|t| t == expect_tex), "seed {seed}");
        }
    }

    #[test]
    fn an_explored_batch_gets_distinct_names() {
        // The Explorer's picks should read as different pieces at a
        // glance: titles pairwise distinct for the standard batch sizes.
        let spec = Style::Classical.spec();
        let it = intent(0);
        let seeds = crate::explorer::explore_identities(&spec, &it, 4);
        let titles: Vec<String> = seeds
            .iter()
            .map(|&s| identity_card(&spec, &it, s).title)
            .collect();
        let mut uniq = titles.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), titles.len(), "duplicate titles: {titles:?}");
    }

    #[test]
    fn title_for_is_deterministic_and_needs_no_premise() {
        // The Listen tab's actual case: a direct, unpremised compose
        // (never routed through `premise_for`) still gets a real name.
        let spec = Style::Classical.spec();
        let it = intent(42);
        let a = title_for(&spec, &it, 42);
        let b = title_for(&spec, &it, 42);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn title_recipe_exposes_distinct_deterministic_alternatives() {
        let spec = Style::Sonata.spec();
        let it = intent(27);
        let (grammar, ending) = crate::composer::identity_grammar_for(&it, &spec);
        let first = title_recipe(&spec, &it, 27, grammar, ending);
        let second = title_recipe(&spec, &it, 27, grammar, ending);
        assert_eq!(first, second);
        assert_eq!(first.alternatives.len(), 3);
        assert!(!first.source_traits.is_empty());
        assert!(
            first
                .source_traits
                .iter()
                .any(|trait_name| trait_name.contains("form"))
        );
        assert!(
            first
                .alternatives
                .iter()
                .all(|title| title != &first.generated_title)
        );
        let mut all = first.alternatives.clone();
        all.push(first.generated_title);
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn title_grammar_has_real_family_and_lexical_range() {
        let spec = Style::Classical.spec();
        let it = intent(0);
        let mut titles = Vec::new();
        let mut families = Vec::new();
        let mut article_titles = 0usize;
        for seed in 0..128u64 {
            let seeded_intent = MusicalIntent { seed, ..it };
            let (grammar, ending) = crate::composer::identity_grammar_for(&seeded_intent, &spec);
            let recipe = title_recipe(&spec, &seeded_intent, seed, grammar, ending);
            if recipe.generated_title.starts_with("The ") {
                article_titles += 1;
            }
            titles.push(recipe.generated_title);
            families.push(recipe.family);
        }
        titles.sort();
        titles.dedup();
        families.sort_by_key(|family| *family as u8);
        families.dedup();
        assert!(
            titles.len() >= 80,
            "title range collapsed: {} unique",
            titles.len()
        );
        assert!(families.len() >= 5, "too few title families: {families:?}");
        assert!(
            article_titles < 43,
            "too many titles begin with The: {article_titles}"
        );
    }

    #[test]
    fn title_for_varies_across_seeds_and_uses_mode_for_color() {
        let spec = Style::Flamenco.spec(); // Phrygian — always minorish
        let it = intent(0);
        let titles: Vec<String> = (0..6u64).map(|s| title_for(&spec, &it, s)).collect();
        let mut uniq = titles.clone();
        uniq.sort();
        uniq.dedup();
        assert!(
            uniq.len() > 1,
            "titles should vary across seeds: {titles:?}"
        );
        // Phrygian is always minorish, so every title must draw from the
        // shadow/still adjective banks, never bright/quick.
        for adj in BRIGHT_ADJ.iter().chain(QUICK_ADJ.iter()) {
            for t in &titles {
                assert!(
                    !t.starts_with(adj),
                    "minorish spec must never use a bright/quick adjective: {t:?}"
                );
            }
        }
    }

    fn why_for(style: Style, seed: u64) -> Vec<String> {
        let spec = style.spec();
        let it = intent(seed);
        let (grammar, ending) = crate::composer::identity_grammar_for(&it, &spec);
        why_lines(&spec, grammar, ending, seed)
    }

    #[test]
    fn why_lines_never_empty_for_any_built_in_style() {
        for style in [
            Style::Classical,
            Style::Sonata,
            Style::RenaissancePolyphony,
            Style::AfroCuban,
            Style::Flamenco,
            Style::BossaNova,
            Style::Opera,
            Style::IrishTraditional,
            Style::HindustaniInspired,
            Style::Ambient,
            Style::BaroqueSuite,
        ] {
            let lines = why_for(style, 3);
            assert!(!lines.is_empty(), "{style:?} produced no explanation");
        }
    }

    #[test]
    fn why_lines_hook_sentence_matches_the_cards_own_classification() {
        // One classification, two renderings: the word `identity_card`
        // puts in a trait must be the same word `why_lines` puts in a
        // sentence, for the identical (spec, seed).
        let spec = Style::Classical.spec();
        let it = intent(11);
        let card = identity_card(&spec, &it, 11);
        let card_word = card
            .traits
            .iter()
            .find(|t| t.ends_with("hook"))
            .expect("a hook trait must exist")
            .split(' ')
            .nth(1)
            .unwrap()
            .to_string();
        let (grammar, ending) = crate::composer::identity_grammar_for(&it, &spec);
        let lines = why_lines(&spec, grammar, ending, 11);
        let hook_line = lines
            .iter()
            .find(|l| l.starts_with("The piece's hook"))
            .expect("a hook sentence must exist");
        assert!(
            hook_line.contains(&card_word),
            "hook_line {hook_line:?} must use the same word as the card trait {card_word:?}"
        );
    }

    #[test]
    fn why_lines_name_each_new_style_wave_real_mechanism() {
        // Every distinct mechanism this session built must actually show
        // up in prose, not just exist internally.
        assert!(
            why_for(Style::Sonata, 3)[0].contains("foreign key"),
            "Sonata must explain the tonal conflict/resolution"
        );
        assert!(
            why_for(Style::RenaissancePolyphony, 3)[0].contains("equal"),
            "Renaissance must explain the equal-voices identity"
        );
        assert!(
            why_for(Style::Opera, 3)[0].contains("interrupts"),
            "Opera must explain the interruption"
        );
        assert!(
            why_for(Style::AfroCuban, 3)
                .iter()
                .any(|l| l.contains("clave")),
            "AfroCuban must explain the montuno/clave mechanism"
        );
        assert!(
            why_for(Style::Flamenco, 3)
                .iter()
                .any(|l| l.contains("compás")),
            "Flamenco must explain the compás cycle"
        );
        assert!(
            why_for(Style::BossaNova, 3)
                .iter()
                .any(|l| l.contains("floating")),
            "BossaNova must explain the floating legato harmony"
        );
        assert!(
            why_for(Style::IrishTraditional, 3)
                .iter()
                .any(|l| l.contains("five-note")),
            "IrishTraditional must explain the roll ornament"
        );
        assert!(
            why_for(Style::HindustaniInspired, 3)
                .iter()
                .any(|l| l.contains("never changes")),
            "Hindustani must explain the full drone"
        );
    }

    #[test]
    fn why_lines_erosion_ending_appends_to_the_grammar_sentence() {
        let spec = crate::style::Style::Nocturne.spec();
        let lines = why_lines(&spec, "erosion", Some("elegy"), 1);
        assert!(lines[0].contains("gradually loses itself"));
        assert!(lines[0].contains("can't quite get back"));
    }
}
