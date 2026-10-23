// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The public Muse 152 taxonomy.
//!
//! Catalog entries and callable composer presets are deliberately separate.
//! `composer_style` is present only when the current engine can route the entry;
//! a research entry can therefore be visible without pretending to be implemented.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Constellation {
    ClassicalLyricCharacter,
    BaroqueEarlyDance,
    DevelopmentalLargeForms,
    ContrapuntalPolyphonic,
    GroundOstinatoTransformation,
    SongNarrative,
    BluesGospelSoul,
    JazzImprovisatory,
    LatinCaribbeanCycles,
    EuropeanNorthAtlanticFolk,
    MediterraneanMiddleEasternNorthAfrican,
    SouthAsianRagaTala,
    EastSoutheastAsian,
    AfricanDiasporicGroove,
    MinimalProcessExperimental,
    AmbientElectronicTexture,
    ClubBeatMusic,
    PopRock,
    DramaticScreenStage,
}

impl Constellation {
    pub const ALL: [Self; 19] = [
        Self::ClassicalLyricCharacter,
        Self::BaroqueEarlyDance,
        Self::DevelopmentalLargeForms,
        Self::ContrapuntalPolyphonic,
        Self::GroundOstinatoTransformation,
        Self::SongNarrative,
        Self::BluesGospelSoul,
        Self::JazzImprovisatory,
        Self::LatinCaribbeanCycles,
        Self::EuropeanNorthAtlanticFolk,
        Self::MediterraneanMiddleEasternNorthAfrican,
        Self::SouthAsianRagaTala,
        Self::EastSoutheastAsian,
        Self::AfricanDiasporicGroove,
        Self::MinimalProcessExperimental,
        Self::AmbientElectronicTexture,
        Self::ClubBeatMusic,
        Self::PopRock,
        Self::DramaticScreenStage,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::ClassicalLyricCharacter => "Classical Lyric & Character",
            Self::BaroqueEarlyDance => "Baroque and Early Dance Forms",
            Self::DevelopmentalLargeForms => "Developmental and Large Forms",
            Self::ContrapuntalPolyphonic => "Contrapuntal and Polyphonic",
            Self::GroundOstinatoTransformation => "Ground, Ostinato and Transformation",
            Self::SongNarrative => "Song and Narrative",
            Self::BluesGospelSoul => "Blues, Gospel and Soul",
            Self::JazzImprovisatory => "Jazz and Improvisatory Forms",
            Self::LatinCaribbeanCycles => "Latin and Caribbean Cycles",
            Self::EuropeanNorthAtlanticFolk => "European and North Atlantic Folk",
            Self::MediterraneanMiddleEasternNorthAfrican => {
                "Mediterranean, Middle Eastern and North African Modal"
            }
            Self::SouthAsianRagaTala => "South Asian Raga and Tala",
            Self::EastSoutheastAsian => "East and Southeast Asian Traditions",
            Self::AfricanDiasporicGroove => "African and Diasporic Groove",
            Self::MinimalProcessExperimental => "Minimal, Process and Experimental",
            Self::AmbientElectronicTexture => "Ambient and Electronic Texture",
            Self::ClubBeatMusic => "Club and Beat Music",
            Self::PopRock => "Pop and Rock",
            Self::DramaticScreenStage => "Dramatic, Screen and Stage",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleStatus {
    Foundation,
    Developing,
    Research,
    ExpertReviewRequired,
}

impl StyleStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Foundation => "Foundation",
            Self::Developing => "Developing",
            Self::Research => "Research",
            Self::ExpertReviewRequired => "Expert review required",
        }
    }

    pub const fn is_composable(self) -> bool {
        matches!(
            self,
            Self::Foundation | Self::Developing | Self::ExpertReviewRequired
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleAnatomy {
    pub grammar: &'static str,
    pub phrase_behavior: &'static str,
    pub harmonic_system: &'static str,
    pub rhythm: &'static str,
    pub melodic_language: &'static str,
    pub ensemble: &'static str,
    pub performance_dialect: &'static str,
    pub production_environment: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalStyle {
    pub id: u16,
    pub name: &'static str,
    pub display_name: &'static str,
    pub constellation: Constellation,
    pub status: StyleStatus,
    /// Existing `symthaea_music_theory::Style` serde name, if callable.
    pub composer_style: Option<&'static str>,
    pub requires_expert_review: bool,
}

/// Evidence required to promote a catalog entry to `Foundation`.
///
/// This is deliberately independent of implementation tests: passing unit
/// tests is necessary for a callable style, but it is not evidence that people
/// can distinguish its grammar by listening.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StylePromotionEvidence {
    pub grammar_family_blind_gate_passed: bool,
    pub within_style_identity_passed: bool,
    pub expert_review_completed: bool,
}

/// One immutable human-evidence submission. Unit tests may generate artifacts,
/// but these booleans are only set by the listening/review workflow itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleEvidenceRecord {
    pub catalog_id: u16,
    pub recorded_at_unix_ms: u64,
    pub engine_version: String,
    pub reviewer: String,
    pub evidence: StylePromotionEvidence,
    #[serde(default)]
    pub artifacts: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedStyleReadiness {
    pub catalog_id: u16,
    pub catalog_status: StyleStatus,
    pub effective_status: StyleStatus,
    pub blockers: Vec<PromotionBlocker>,
    pub latest_evidence: Option<StyleEvidenceRecord>,
}

/// A hybrid has one structural owner; every other source may contribute only
/// its named layer. It is never serialized as a historical genre claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridStyleSpec {
    pub structural_style_id: u16,
    pub rhythmic_style_id: Option<u16>,
    pub harmonic_style_id: Option<u16>,
    pub ensemble_style_id: Option<u16>,
    pub performance_style_id: Option<u16>,
    pub production_style_id: Option<u16>,
    #[serde(default)]
    pub culturally_qualified_sources_acknowledged: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridValidationIssue {
    pub code: String,
    pub message: String,
}

impl HybridStyleSpec {
    pub fn honest_label(&self) -> String {
        let name = |id| {
            catalog_entry(id)
                .map(|entry| entry.display_name)
                .unwrap_or("Unknown")
        };
        let mut layers = vec![format!("{} structure", name(self.structural_style_id))];
        for (id, layer) in [
            (self.rhythmic_style_id, "rhythm"),
            (self.harmonic_style_id, "harmony"),
            (self.ensemble_style_id, "ensemble"),
            (self.performance_style_id, "performance"),
            (self.production_style_id, "production"),
        ] {
            if let Some(id) = id {
                layers.push(format!("{} {layer}", name(id)));
            }
        }
        format!("Hybrid — {}", layers.join(" + "))
    }

    pub fn validate(&self) -> Vec<HybridValidationIssue> {
        let mut issues = Vec::new();
        let ids = [
            Some(self.structural_style_id),
            self.rhythmic_style_id,
            self.harmonic_style_id,
            self.ensemble_style_id,
            self.performance_style_id,
            self.production_style_id,
        ];
        for id in ids.into_iter().flatten() {
            if catalog_entry(id).is_none() {
                issues.push(HybridValidationIssue {
                    code: "unknown_style".into(),
                    message: format!("catalog style {id} does not exist"),
                });
            }
        }
        let Some(structural) = catalog_entry(self.structural_style_id) else {
            return issues;
        };
        if structural.composer_style.is_none() {
            issues.push(HybridValidationIssue {
                code: "structural_owner_not_implemented".into(),
                message: "the structural owner must have an executable grammar".into(),
            });
        }
        let selected: Vec<_> = ids
            .into_iter()
            .flatten()
            .filter_map(catalog_entry)
            .collect();
        if selected.iter().any(|entry| entry.requires_expert_review)
            && !self.culturally_qualified_sources_acknowledged
        {
            issues.push(HybridValidationIssue {
                code: "cultural_qualification_required".into(),
                message: "culturally specific sources require explicit qualification and may not be labelled authentic".into(),
            });
        }
        if let Some(rhythm) = self.rhythmic_style_id.and_then(catalog_entry) {
            if structural.constellation == Constellation::SouthAsianRagaTala
                && rhythm.constellation == Constellation::LatinCaribbeanCycles
            {
                issues.push(HybridValidationIssue {
                    code: "free_time_cycle_transition_required".into(),
                    message: "a free-time modal exposition cannot be cycle-locked without an explicit stage transition".into(),
                });
            }
        }
        if let Some(harmony) = self.harmonic_style_id.and_then(catalog_entry) {
            if structural.constellation == Constellation::SouthAsianRagaTala
                && matches!(
                    harmony.constellation,
                    Constellation::ClassicalLyricCharacter | Constellation::DevelopmentalLargeForms
                )
            {
                issues.push(HybridValidationIssue {
                    code: "pitch_hierarchy_conflict".into(),
                    message: "functional cadence syntax conflicts with invariant modal pitch hierarchy; declare a modal adaptation instead".into(),
                });
            }
        }
        issues
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionBlocker {
    NotImplemented,
    GrammarFamilyBlindGate,
    WithinStyleIdentityGate,
    ExpertReview,
}

impl CanonicalStyle {
    pub const fn is_composable(self) -> bool {
        self.composer_style.is_some() && self.status.is_composable()
    }

    pub const fn anatomy(self) -> StyleAnatomy {
        let family = anatomy_for(self.constellation);
        StyleAnatomy {
            grammar: CANONICAL_GRAMMAR_FOCI[(self.id as usize - 1) / 8][(self.id as usize - 1) % 8],
            phrase_behavior: family.phrase_behavior,
            harmonic_system: family.harmonic_system,
            rhythm: family.rhythm,
            melodic_language: family.melodic_language,
            ensemble: family.ensemble,
            performance_dialect: family.performance_dialect,
            production_environment: family.production_environment,
        }
    }

    pub const fn research_gaps(self) -> &'static [&'static str] {
        if self.requires_expert_review {
            &[
                "tradition-specific source model and terminology",
                "review by knowledgeable musicians",
                "blind listening and within-style identity evidence",
            ]
        } else if matches!(self.status, StyleStatus::Research) {
            &[
                "executable canonical grammar",
                "style-specific anatomy beyond the family hypothesis",
                "blind listening and within-style identity evidence",
            ]
        } else {
            &[
                "grammar-family blind-listening gate",
                "within-style identity gate",
            ]
        }
    }
}

/// Canonical structural hypotheses, in catalog order. The remaining anatomy
/// axes inherit a constellation research model until style-specific evidence
/// replaces them; [`CanonicalStyle::research_gaps`] makes that incompleteness
/// visible instead of presenting generic fields as finished scholarship.
const CANONICAL_GRAMMAR_FOCI: [[&str; 8]; 19] = [
    [
        "balanced period",
        "night-song arc",
        "triple-meter rotation",
        "cradle repetition",
        "processional strain",
        "lament arc",
        "comic interruption",
        "coloristic prelude",
    ],
    [
        "suite succession",
        "minuet–trio return",
        "gavotte upbeat dance",
        "slow sarabande",
        "compound-meter gigue",
        "allemande flow",
        "motoric toccata",
        "dotted overture–fugue",
    ],
    [
        "sonata obligations",
        "refrain rotation",
        "theme transformation",
        "scherzo–trio return",
        "programmatic transformation",
        "solo–tutti argument",
        "multi-movement suite arc",
        "free chamber fantasia",
    ],
    [
        "fugal subject entries",
        "compact fugal exposition",
        "two-voice invention",
        "strict canon",
        "ricercar investigation",
        "equal-voice sacred weave",
        "text-shaped motet",
        "chorale–counterpoint overlay",
    ],
    [
        "passacaglia ground",
        "chaconne harmonic ground",
        "tetrachord lament",
        "folia variation chain",
        "romanesca schema",
        "ostinato accumulation",
        "progressive loss",
        "genealogical transformation",
    ],
    [
        "communal strophic song",
        "modal strophic song",
        "narrative ballad",
        "confessional verse song",
        "poem-led art song",
        "verse–chorus contrast",
        "continuous text setting",
        "work-song call and response",
    ],
    [
        "Delta verse chorus",
        "urban amplified chorus",
        "acoustic country chorus",
        "gospel escalation",
        "spiritual call and response",
        "soul slow build",
        "R&B groove song",
        "neo-soul pocket",
    ],
    [
        "ballad choruses",
        "swing head–solos–head",
        "bebop chorus navigation",
        "cool-jazz restraint",
        "hard-bop chorus drive",
        "modal vamp exploration",
        "triple-meter jazz choruses",
        "open-form interaction",
    ],
    [
        "clave-owned cycle",
        "son–montuno cycle",
        "salsa energy cycle",
        "mambo section cycle",
        "cha-cha dance cycle",
        "bossa continuous syncopation",
        "samba layered cycle",
        "habanera dramatic cycle",
    ],
    [
        "Irish tune sets",
        "pan-Celtic tune sequence",
        "reel–strathspey contrast",
        "English narrative tune",
        "Nordic asymmetric dance",
        "odd-meter sectional dance",
        "klezmer dance suite",
        "developmental folk suite",
    ],
    [
        "flamenco compás forms",
        "fado narrative strophe",
        "maqam pathway",
        "makam seyir",
        "dastgah modal journey",
        "nuba suite succession",
        "taqsim modal improvisation",
        "rebetiko song form",
    ],
    [
        "khayal-informed modal arc",
        "dhrupad-informed austere arc",
        "alap–jor–jhala density arc",
        "kriti sectional form",
        "ragam–tanam–pallavi expansion",
        "devotional bhajan strophe",
        "ghazal couplet setting",
        "qawwali cumulative response",
    ],
    [
        "gagaku temporal sequence",
        "honkyoku breath form",
        "min’yō song cycle",
        "guqin programmatic miniature",
        "sizhu ensemble conversation",
        "gugak suite logic",
        "gamelan colotomic cycle",
        "piphat dramatic suite",
    ],
    [
        "Afrobeat interlocking vamp",
        "highlife guitar cycle",
        "juju cumulative groove",
        "soukous acceleration",
        "amapiano log-drum arc",
        "gqom sparse pressure cycle",
        "reggae riddim song",
        "dub version transformation",
    ],
    [
        "additive repetition",
        "phase divergence",
        "explicit additive rule",
        "post-minimal process arc",
        "spectral evolution",
        "bounded chance form",
        "rule-owned counterpoint",
        "cellular emergence",
    ],
    [
        "ambient density horizon",
        "dark-ambient pressure arc",
        "continuous drone evolution",
        "sequencer-led Berlin arc",
        "IDM metric mutation",
        "glitch discontinuity process",
        "memory-loop vaporwave",
        "acoustic–electronic transformation",
    ],
    [
        "house phrase-energy grid",
        "deep-house gradual layering",
        "techno machine process",
        "trance tension–release",
        "drum-and-bass break architecture",
        "garage shuffle cycle",
        "breakbeat recombination",
        "hip-hop loop and sectional beat",
    ],
    [
        "pop verse–chorus",
        "indie-pop hook song",
        "dream-pop textural song",
        "synthpop production song",
        "riff-owned rock song",
        "developmental rock suite",
        "post-rock cumulative arc",
        "folk-rock narrative song",
    ],
    [
        "operatic scene sequence",
        "book-song dramatic sequence",
        "leitmotif scene arc",
        "noir cue grammar",
        "epic escalation cue",
        "horror threat-state cue",
        "science-fiction world-state cue",
        "interactive state-transition score",
    ],
];

const fn anatomy_for(constellation: Constellation) -> StyleAnatomy {
    use Constellation::*;
    match constellation {
        ClassicalLyricCharacter => anatomy(
            "period/sentence",
            "balanced antecedent–consequent rhetoric",
            "functional or coloristic tonal syntax",
            "metered character pulse",
            "motivic singing line",
            "solo or chamber",
            "classical rubato",
            "natural chamber room",
        ),
        BaroqueEarlyDance => anatomy(
            "dance/fortspinnung",
            "sequence-led dance phrases",
            "functional sequence and figured bass",
            "dance-specific gait",
            "ornamented motivic line",
            "continuo ensemble",
            "articulated dance",
            "intimate early-music room",
        ),
        DevelopmentalLargeForms => anatomy(
            "developmental",
            "long-range obligation and return",
            "tonal-region plan",
            "form-dependent pulse",
            "transformational thematic DNA",
            "chamber to orchestral",
            "structural rubato",
            "concert hall",
        ),
        ContrapuntalPolyphonic => anatomy(
            "contrapuntal",
            "imitative independent voices",
            "controlled contrapuntal dissonance",
            "voice-led pulse",
            "subject and countersubject",
            "polyphonic ensemble",
            "independent articulation",
            "coherent acoustic space",
        ),
        GroundOstinatoTransformation => anatomy(
            "ground/variation",
            "cumulative variation over a remembered ground",
            "cyclic ground syntax",
            "ostinato cycle",
            "transformational upper voices",
            "chamber bass foundation",
            "ground-aware arc",
            "focused chamber room",
        ),
        SongNarrative => anatomy(
            "strophic/song",
            "verse, refrain, or through-composed story",
            "song-specific harmonic syntax",
            "text-shaped pulse",
            "vocal narrative contour",
            "voice-led ensemble",
            "lyric phrasing",
            "song production",
        ),
        BluesGospelSoul => anatomy(
            "chorus/call-response",
            "chorus memory and response",
            "blues/gospel harmonic syntax",
            "shuffle or pocket groove",
            "blue-note vocal rhetoric",
            "rhythm section and voices",
            "laid-back vocal dialect",
            "close, warm room",
        ),
        JazzImprovisatory => anatomy(
            "jazz chorus",
            "head, choruses, exchanges, return",
            "turnaround and chord-scale syntax",
            "swing or jazz meter",
            "improvisational phrase memory",
            "jazz combo",
            "jazz laid-back",
            "club room",
        ),
        LatinCaribbeanCycles => anatomy(
            "groove-cycle",
            "cycle-spanning calls and montuno energy",
            "cyclic vamp and cadence",
            "clave/tumbao/dance cycle",
            "syncopated call-response",
            "interlocking dance ensemble",
            "dance-locked",
            "live dance room",
        ),
        EuropeanNorthAtlanticFolk => anatomy(
            "strophic/dance",
            "tradition-specific tune strains",
            "modal cadence pathways",
            "dance meter and asymmetry",
            "ornamented traditional tune",
            "regional folk ensemble",
            "folk lift",
            "acoustic session",
        ),
        MediterraneanMiddleEasternNorthAfrican => anatomy(
            "modal arc",
            "modal pathway and improvisatory development",
            "maqam/makam/dastgah-informed pathway",
            "cycle or free pulse",
            "microtonal ornament language",
            "tradition-specific ensemble",
            "ornament-led",
            "acoustic performance space",
        ),
        SouthAsianRagaTala => anatomy(
            "raga/modal arc",
            "temporal unfolding from exposition to intensification",
            "pitch hierarchy over drone",
            "free time into tala cycle",
            "raga-informed ornament and hierarchy",
            "drone and melodic soloist",
            "elastic-to-cycle",
            "acoustic recital space",
        ),
        EastSoutheastAsian => anatomy(
            "ensemble/modal process",
            "role-based cycles and silence",
            "tuning- and mode-specific organization",
            "ensemble cycle",
            "heterophonic or sparse identity",
            "tradition-specific ensemble",
            "silence-aware",
            "natural ensemble space",
        ),
        AfricanDiasporicGroove => anatomy(
            "groove-cycle",
            "cyclic accumulation and conversation",
            "vamp-centered syntax",
            "layered interlocking pulse",
            "riff and call-response",
            "interlocking groove ensemble",
            "dance-locked",
            "rhythm-forward production",
        ),
        MinimalProcessExperimental => anatomy(
            "process/additive",
            "audible rule-governed transformation",
            "static, spectral, or algorithmic field",
            "process pulse or controlled freedom",
            "cellular process identity",
            "process ensemble",
            "process-exact",
            "transparent process space",
        ),
        AmbientElectronicTexture => anatomy(
            "ambient/textural",
            "long-horizon density and timbre arc",
            "spectral field or harmonic stasis",
            "sparse or sequenced pulse",
            "textural motif",
            "electronic layers",
            "drone-elastic",
            "immersive production space",
        ),
        ClubBeatMusic => anatomy(
            "drop/energy",
            "build, break, drop, recovery",
            "loop and tension-release syntax",
            "grid-locked club groove",
            "hook and riff cells",
            "electronic rhythm stack",
            "grid-locked",
            "club master",
        ),
        PopRock => anatomy(
            "verse/chorus/riff",
            "verse, pre-chorus, chorus, bridge",
            "song and riff harmony",
            "backbeat or rock meter",
            "hook-led melodic DNA",
            "band or pop production",
            "studio band",
            "production-forward mix",
        ),
        DramaticScreenStage => anatomy(
            "dramatic/adaptive",
            "scene and character obligations",
            "leitmotivic narrative syntax",
            "state-dependent pulse",
            "character and scene motifs",
            "stage or scoring ensemble",
            "narrative timing",
            "cinematic or theatrical space",
        ),
    }
}

const fn anatomy(
    grammar: &'static str,
    phrase_behavior: &'static str,
    harmonic_system: &'static str,
    rhythm: &'static str,
    melodic_language: &'static str,
    ensemble: &'static str,
    performance_dialect: &'static str,
    production_environment: &'static str,
) -> StyleAnatomy {
    StyleAnatomy {
        grammar,
        phrase_behavior,
        harmonic_system,
        rhythm,
        melodic_language,
        ensemble,
        performance_dialect,
        production_environment,
    }
}

macro_rules! style {
    ($id:literal, $name:literal, $constellation:ident) => {
        CanonicalStyle {
            id: $id,
            name: $name,
            display_name: $name,
            constellation: Constellation::$constellation,
            status: StyleStatus::Research,
            composer_style: None,
            requires_expert_review: false,
        }
    };
    ($id:literal, $name:literal, $constellation:ident, $composer:literal, $status:ident) => {
        CanonicalStyle {
            id: $id,
            name: $name,
            display_name: $name,
            constellation: Constellation::$constellation,
            status: StyleStatus::$status,
            composer_style: Some($composer),
            requires_expert_review: matches!(
                StyleStatus::$status,
                StyleStatus::ExpertReviewRequired
            ),
        }
    };
    ($id:literal, $name:literal, $display:literal, $constellation:ident, $composer:literal, $status:ident) => {
        CanonicalStyle {
            id: $id,
            name: $name,
            display_name: $display,
            constellation: Constellation::$constellation,
            status: StyleStatus::$status,
            composer_style: Some($composer),
            requires_expert_review: matches!(
                StyleStatus::$status,
                StyleStatus::ExpertReviewRequired
            ),
        }
    };
}

/// The complete, stable Muse 152 registry. IDs are public catalog identities.
pub const CATALOG: [CanonicalStyle; 152] = [
    style!(
        1,
        "Classical",
        ClassicalLyricCharacter,
        "Classical",
        Developing
    ),
    style!(
        2,
        "Romantic Nocturne",
        ClassicalLyricCharacter,
        "Nocturne",
        Developing
    ),
    style!(3, "Waltz", ClassicalLyricCharacter, "Waltz", Developing),
    style!(4, "Lullaby", ClassicalLyricCharacter, "Lullaby", Developing),
    style!(5, "March", ClassicalLyricCharacter, "March", Developing),
    style!(6, "Elegy", ClassicalLyricCharacter),
    style!(
        7,
        "Playful Character Piece",
        ClassicalLyricCharacter,
        "Playful",
        Developing
    ),
    style!(
        8,
        "Impressionist Prelude",
        ClassicalLyricCharacter,
        "Impressionism",
        Developing
    ),
    style!(
        9,
        "Baroque Suite",
        BaroqueEarlyDance,
        "BaroqueSuite",
        Developing
    ),
    style!(10, "Minuet and Trio", BaroqueEarlyDance),
    style!(11, "Gavotte", BaroqueEarlyDance),
    style!(12, "Sarabande", BaroqueEarlyDance),
    style!(13, "Gigue", BaroqueEarlyDance),
    style!(14, "Allemande", BaroqueEarlyDance),
    style!(15, "Toccata", BaroqueEarlyDance),
    style!(16, "French Overture", BaroqueEarlyDance),
    style!(17, "Sonata", DevelopmentalLargeForms, "Sonata", Developing),
    style!(18, "Rondo", DevelopmentalLargeForms),
    style!(19, "Theme and Variations", DevelopmentalLargeForms),
    style!(20, "Scherzo", DevelopmentalLargeForms),
    style!(21, "Symphonic Poem", DevelopmentalLargeForms),
    style!(22, "Concerto Movement", DevelopmentalLargeForms),
    style!(23, "Progressive Suite", DevelopmentalLargeForms),
    style!(24, "Chamber Fantasia", DevelopmentalLargeForms),
    style!(25, "Fugue", ContrapuntalPolyphonic, "Fugue", Developing),
    style!(26, "Fughetta", ContrapuntalPolyphonic),
    style!(27, "Two-Part Invention", ContrapuntalPolyphonic),
    style!(28, "Canon", ContrapuntalPolyphonic),
    style!(29, "Ricercar", ContrapuntalPolyphonic),
    style!(
        30,
        "Renaissance Polyphony",
        ContrapuntalPolyphonic,
        "RenaissancePolyphony",
        Developing
    ),
    style!(
        31,
        "Motet",
        ContrapuntalPolyphonic,
        "SacredChoral",
        Developing
    ),
    style!(32, "Chorale Prelude", ContrapuntalPolyphonic),
    style!(
        33,
        "Passacaglia",
        GroundOstinatoTransformation,
        "Passacaglia",
        Developing
    ),
    style!(34, "Chaconne", GroundOstinatoTransformation),
    style!(
        35,
        "Descending-Tetrachord Lament",
        GroundOstinatoTransformation
    ),
    style!(36, "Folia Variations", GroundOstinatoTransformation),
    style!(37, "Romanesca", GroundOstinatoTransformation),
    style!(38, "Ostinato Variations", GroundOstinatoTransformation),
    style!(39, "Erosion", GroundOstinatoTransformation),
    style!(40, "Lineage", GroundOstinatoTransformation),
    style!(41, "Folk", SongNarrative, "Folk", Developing),
    style!(42, "Modal Folk", SongNarrative, "ModalFolk", Developing),
    style!(43, "Folk Ballad", SongNarrative),
    style!(44, "Singer-Songwriter", SongNarrative),
    style!(45, "Art Song / Lied", SongNarrative),
    style!(46, "Verse–Chorus Song", SongNarrative),
    style!(47, "Through-Composed Song", SongNarrative),
    style!(48, "Sea Shanty", SongNarrative),
    style!(49, "Delta Blues", BluesGospelSoul),
    style!(50, "Chicago Blues", BluesGospelSoul, "Blues", Developing),
    style!(51, "Country Blues", BluesGospelSoul),
    style!(52, "Gospel", BluesGospelSoul),
    style!(53, "Spiritual", BluesGospelSoul),
    style!(54, "Soul Ballad", BluesGospelSoul),
    style!(55, "Rhythm and Blues", BluesGospelSoul),
    style!(56, "Neo-Soul", BluesGospelSoul),
    style!(
        57,
        "Jazz Ballad",
        JazzImprovisatory,
        "JazzBallad",
        Developing
    ),
    style!(58, "Swing", JazzImprovisatory),
    style!(59, "Bebop", JazzImprovisatory),
    style!(60, "Cool Jazz", JazzImprovisatory),
    style!(61, "Hard Bop", JazzImprovisatory),
    style!(62, "Modal Jazz", JazzImprovisatory),
    style!(63, "Jazz Waltz", JazzImprovisatory),
    style!(64, "Free Jazz", JazzImprovisatory),
    style!(
        65,
        "Afro-Cuban",
        LatinCaribbeanCycles,
        "AfroCuban",
        ExpertReviewRequired
    ),
    style!(66, "Son Cubano", LatinCaribbeanCycles),
    style!(67, "Salsa", LatinCaribbeanCycles),
    style!(68, "Mambo", LatinCaribbeanCycles),
    style!(69, "Cha-Cha-Chá", LatinCaribbeanCycles),
    style!(
        70,
        "Bossa Nova",
        LatinCaribbeanCycles,
        "BossaNova",
        ExpertReviewRequired
    ),
    style!(71, "Samba", LatinCaribbeanCycles),
    style!(72, "Tango", LatinCaribbeanCycles, "Tango", Developing),
    style!(
        73,
        "Irish Traditional",
        EuropeanNorthAtlanticFolk,
        "IrishTraditional",
        ExpertReviewRequired
    ),
    style!(
        74,
        "Celtic",
        EuropeanNorthAtlanticFolk,
        "Celtic",
        Developing
    ),
    style!(
        75,
        "Scottish Reel and Strathspey",
        EuropeanNorthAtlanticFolk
    ),
    style!(76, "English Folk", EuropeanNorthAtlanticFolk),
    style!(77, "Nordic Folk", EuropeanNorthAtlanticFolk),
    style!(78, "Balkan Odd-Meter", EuropeanNorthAtlanticFolk),
    style!(79, "Klezmer", EuropeanNorthAtlanticFolk),
    style!(
        80,
        "Progressive Folk",
        EuropeanNorthAtlanticFolk,
        "ProgFolk",
        Developing
    ),
    style!(
        81,
        "Flamenco",
        MediterraneanMiddleEasternNorthAfrican,
        "Flamenco",
        ExpertReviewRequired
    ),
    style!(82, "Fado", MediterraneanMiddleEasternNorthAfrican),
    style!(83, "Arabic Maqam", MediterraneanMiddleEasternNorthAfrican),
    style!(84, "Turkish Makam", MediterraneanMiddleEasternNorthAfrican),
    style!(
        85,
        "Persian Dastgah",
        MediterraneanMiddleEasternNorthAfrican
    ),
    style!(
        86,
        "Andalusian Nuba",
        MediterraneanMiddleEasternNorthAfrican
    ),
    style!(87, "Oud Taqsim", MediterraneanMiddleEasternNorthAfrican),
    style!(88, "Greek Rebetiko", MediterraneanMiddleEasternNorthAfrican),
    style!(
        89,
        "Hindustani Khayal",
        "Hindustani-informed",
        SouthAsianRagaTala,
        "HindustaniInspired",
        ExpertReviewRequired
    ),
    style!(90, "Dhrupad", SouthAsianRagaTala),
    style!(91, "Alap–Jor–Jhala", SouthAsianRagaTala),
    style!(92, "Carnatic Kriti", SouthAsianRagaTala),
    style!(93, "Ragam–Tanam–Pallavi", SouthAsianRagaTala),
    style!(94, "Bhajan", SouthAsianRagaTala),
    style!(95, "Ghazal", SouthAsianRagaTala),
    style!(96, "Qawwali", SouthAsianRagaTala),
    style!(97, "Japanese Gagaku", EastSoutheastAsian),
    style!(98, "Shakuhachi Honkyoku", EastSoutheastAsian),
    style!(99, "Japanese Min’yō", EastSoutheastAsian),
    style!(100, "Chinese Guqin", EastSoutheastAsian),
    style!(101, "Jiangnan Sizhu", EastSoutheastAsian),
    style!(102, "Korean Gugak", EastSoutheastAsian),
    style!(103, "Indonesian Gamelan", EastSoutheastAsian),
    style!(104, "Thai Piphat", EastSoutheastAsian),
    style!(105, "Afrobeat", AfricanDiasporicGroove),
    style!(106, "Highlife", AfricanDiasporicGroove),
    style!(107, "Juju", AfricanDiasporicGroove),
    style!(108, "Soukous", AfricanDiasporicGroove),
    style!(109, "Amapiano", AfricanDiasporicGroove),
    style!(110, "Gqom", AfricanDiasporicGroove),
    style!(111, "Reggae", AfricanDiasporicGroove),
    style!(112, "Dub", AfricanDiasporicGroove),
    style!(
        113,
        "Minimalism",
        MinimalProcessExperimental,
        "Minimalism",
        Developing
    ),
    style!(114, "Phase Music", MinimalProcessExperimental),
    style!(115, "Additive Process", MinimalProcessExperimental),
    style!(116, "Post-Minimalism", MinimalProcessExperimental),
    style!(117, "Spectralism", MinimalProcessExperimental),
    style!(118, "Aleatoric Chamber", MinimalProcessExperimental),
    style!(119, "Algorithmic Counterpoint", MinimalProcessExperimental),
    style!(120, "Generative Cellular Music", MinimalProcessExperimental),
    style!(
        121,
        "Ambient",
        AmbientElectronicTexture,
        "Ambient",
        Developing
    ),
    style!(122, "Dark Ambient", AmbientElectronicTexture),
    style!(123, "Drone", AmbientElectronicTexture),
    style!(124, "Berlin School", AmbientElectronicTexture),
    style!(125, "IDM", AmbientElectronicTexture),
    style!(126, "Glitch", AmbientElectronicTexture),
    style!(127, "Vaporwave", AmbientElectronicTexture),
    style!(128, "Electroacoustic", AmbientElectronicTexture),
    style!(129, "House", ClubBeatMusic),
    style!(130, "Deep House", ClubBeatMusic),
    style!(131, "Techno", ClubBeatMusic),
    style!(132, "Trance", ClubBeatMusic),
    style!(133, "Drum and Bass", ClubBeatMusic),
    style!(134, "UK Garage", ClubBeatMusic),
    style!(135, "Breakbeat", ClubBeatMusic),
    style!(136, "Hip-Hop Instrumental", ClubBeatMusic),
    style!(137, "Pop", PopRock),
    style!(138, "Indie Pop", PopRock),
    style!(139, "Dream Pop", PopRock),
    style!(140, "Synthpop", PopRock),
    style!(141, "Rock", PopRock),
    style!(142, "Progressive Rock", PopRock),
    style!(143, "Post-Rock", PopRock),
    style!(144, "Folk Rock", PopRock),
    style!(145, "Opera", DramaticScreenStage, "Opera", Developing),
    style!(146, "Musical Theatre", DramaticScreenStage),
    style!(
        147,
        "Cinematic",
        DramaticScreenStage,
        "Cinematic",
        Developing
    ),
    style!(148, "Film Noir Score", DramaticScreenStage),
    style!(149, "Epic Orchestral", DramaticScreenStage),
    style!(150, "Horror Score", DramaticScreenStage),
    style!(151, "Science-Fiction Score", DramaticScreenStage),
    style!(152, "Adaptive Game Score", DramaticScreenStage),
];

pub fn catalog_entry(id: u16) -> Option<&'static CanonicalStyle> {
    id.checked_sub(1)
        .and_then(|index| CATALOG.get(index as usize))
}

pub fn catalog_for_constellation(
    constellation: Constellation,
) -> impl Iterator<Item = &'static CanonicalStyle> {
    CATALOG
        .iter()
        .filter(move |entry| entry.constellation == constellation)
}

pub fn implemented_catalog() -> impl Iterator<Item = &'static CanonicalStyle> {
    CATALOG.iter().filter(|entry| entry.is_composable())
}

pub fn foundation_promotion_blockers(
    entry: &CanonicalStyle,
    evidence: StylePromotionEvidence,
) -> Vec<PromotionBlocker> {
    let mut blockers = Vec::new();
    if entry.composer_style.is_none() {
        blockers.push(PromotionBlocker::NotImplemented);
    }
    if !evidence.grammar_family_blind_gate_passed {
        blockers.push(PromotionBlocker::GrammarFamilyBlindGate);
    }
    if !evidence.within_style_identity_passed {
        blockers.push(PromotionBlocker::WithinStyleIdentityGate);
    }
    if entry.requires_expert_review && !evidence.expert_review_completed {
        blockers.push(PromotionBlocker::ExpertReview);
    }
    blockers
}

pub fn derive_style_readiness(
    entry: &CanonicalStyle,
    latest_evidence: Option<StyleEvidenceRecord>,
) -> DerivedStyleReadiness {
    let evidence = latest_evidence
        .as_ref()
        .map(|record| record.evidence)
        .unwrap_or_default();
    let blockers = foundation_promotion_blockers(entry, evidence);
    DerivedStyleReadiness {
        catalog_id: entry.id,
        catalog_status: entry.status,
        effective_status: if blockers.is_empty() {
            StyleStatus::Foundation
        } else {
            entry.status
        },
        blockers,
        latest_evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn catalog_is_exactly_nineteen_by_eight() {
        assert_eq!(CATALOG.len(), 152);
        for constellation in Constellation::ALL {
            assert_eq!(
                catalog_for_constellation(constellation).count(),
                8,
                "{}",
                constellation.name()
            );
        }
    }

    #[test]
    fn ids_and_names_are_unique_and_stable() {
        let mut names = HashSet::new();
        for (index, entry) in CATALOG.iter().enumerate() {
            assert_eq!(entry.id as usize, index + 1);
            assert!(names.insert(entry.name));
            assert!(!entry.name.is_empty());
        }
    }

    #[test]
    fn all_current_composer_styles_are_mapped_once() {
        let implemented: Vec<_> = implemented_catalog().collect();
        assert_eq!(implemented.len(), 29);
        let names: HashSet<_> = implemented
            .iter()
            .filter_map(|entry| entry.composer_style)
            .collect();
        assert_eq!(names.len(), 29);
        assert!(
            implemented
                .iter()
                .all(|entry| entry.status != StyleStatus::Foundation)
        );
    }

    #[test]
    fn foundation_status_is_derived_only_from_complete_evidence() {
        let entry = catalog_entry(1).unwrap();
        let readiness = derive_style_readiness(
            entry,
            Some(StyleEvidenceRecord {
                catalog_id: 1,
                recorded_at_unix_ms: 1,
                engine_version: "test".into(),
                reviewer: "listener-panel".into(),
                evidence: StylePromotionEvidence {
                    grammar_family_blind_gate_passed: true,
                    within_style_identity_passed: true,
                    expert_review_completed: false,
                },
                artifacts: vec!["manifest.json".into()],
                notes: String::new(),
            }),
        );
        assert_eq!(readiness.effective_status, StyleStatus::Foundation);
        assert!(readiness.blockers.is_empty());
    }

    #[test]
    fn hybrid_labels_are_honest_and_modal_conflicts_are_rejected() {
        let hybrid = HybridStyleSpec {
            structural_style_id: 89,
            rhythmic_style_id: Some(65),
            harmonic_style_id: Some(1),
            ensemble_style_id: None,
            performance_style_id: None,
            production_style_id: None,
            culturally_qualified_sources_acknowledged: false,
        };
        assert!(hybrid.honest_label().starts_with("Hybrid — "));
        let codes: Vec<_> = hybrid
            .validate()
            .into_iter()
            .map(|issue| issue.code)
            .collect();
        assert!(codes.contains(&"cultural_qualification_required".into()));
        assert!(codes.contains(&"free_time_cycle_transition_required".into()));
        assert!(codes.contains(&"pitch_hierarchy_conflict".into()));
    }

    #[test]
    fn research_entries_cannot_be_composed() {
        assert!(
            CATALOG
                .iter()
                .filter(|entry| entry.status == StyleStatus::Research)
                .all(|entry| !entry.is_composable())
        );
    }

    #[test]
    fn foundation_gate_requires_listening_and_cultural_review() {
        let afro_cuban = catalog_entry(65).unwrap();
        let evidence = StylePromotionEvidence {
            grammar_family_blind_gate_passed: true,
            within_style_identity_passed: true,
            expert_review_completed: false,
        };
        assert_eq!(
            foundation_promotion_blockers(afro_cuban, evidence),
            vec![PromotionBlocker::ExpertReview]
        );
        let classical = catalog_entry(1).unwrap();
        assert!(
            foundation_promotion_blockers(
                classical,
                StylePromotionEvidence {
                    expert_review_completed: false,
                    ..evidence
                }
            )
            .is_empty()
        );
    }
}
