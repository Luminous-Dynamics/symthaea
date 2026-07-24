// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Motif Foundry, Phase 1: the `MotifFamily` schema and identity contract.
//!
//! A motif is not just a melodic fragment — it is an identity that can
//! survive transformation. This module gives that identity three layers
//! ([`Motif`] itself for the literal notes, [`MotifSignature`] for the
//! transposition/tempo-invariant structural shape, [`PhraseRole`] for what
//! the motif is FOR) plus an explicit contract for how far it may drift
//! before it stops being recognizable as itself ([`MotifFamily::recognizes`]).
//!
//! Deliberately reuses rather than reinvents: transformations are named
//! with [`ReturnTransformation`] (already the vocabulary `ObligationLedger`
//! uses for "this motif must return, transformed thus"), and the rhythmic/
//! contour identity predicates are the ones [`crate::hook::HookCell`]
//! already enforces on generated hooks — a motif family's canonical form is
//! held to the same bar a single hook cell is.

use crate::motif::{Contour, Motif};
use crate::obligation::ReturnTransformation;
use crate::rhythm::Duration;
use crate::style::Style;
use serde::{Deserialize, Serialize};

/// What a motif is FOR — its semantic identity, independent of the exact
/// notes. Restricted per grammar family (a foundry generator only ever
/// requests roles its grammar actually has a shape for).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhraseRole {
    OpeningStatement,
    Answer,
    Transition,
    GrooveSeed,
    Lament,
    ClimaxSeed,
    Return,
    Countermelody,
    CadentialTag,
}

/// A property the family's identity depends on: violate it and a candidate
/// is a *different* motif, not a development of this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotifInvariant {
    /// At least two distinct durations, longest >= 2x shortest
    /// ([`crate::hook::HookCell::has_rhythmic_identity`]'s rule).
    RhythmicIdentity,
    /// A leap of a third or more with recoil, or an immediate repeated
    /// note ([`crate::hook::HookCell::has_contour_identity`]'s rule).
    ContourIdentity,
    /// The first pitched event must sit on this scale degree.
    FirstDegree(i32),
    /// The last pitched event must sit on this scale degree (e.g. a
    /// cadential tag that must always resolve to the tonic).
    LastDegree(i32),
    /// The event count (rests included) must not change.
    EventCount(usize),
}

/// A property the family's identity does NOT depend on — explicitly free
/// to vary. Purely descriptive (nothing here is enforced; its role is to
/// make "this may safely change" a first-class, inspectable fact rather
/// than an implicit absence of a check).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotifDimension {
    PitchLevel,
    FinalInterval,
    MeterPlacement,
    Mode,
    Instrumentation,
    HarmonicSupport,
    Register,
}

/// How closely a candidate motif relates to a family — the four-way
/// distinction the design brief calls out as essential for long-range
/// musical memory: without it, exact repetition, recognizable development,
/// distant relation, and unrelated new material are indistinguishable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityVerdict {
    ExactMatch,
    RecognizableDevelopment,
    DistantRelation,
    Unrelated,
}

/// Transposition/tempo-invariant structural identity: contour (direction
/// only, so pitch level is free to vary) and each note's duration as a
/// fraction of the phrase's total length (so augmentation/diminution,
/// which scale every duration uniformly, leave this unchanged).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MotifSignature {
    pub contour: Vec<Contour>,
    pub duration_fractions: Vec<f32>,
}

/// Derive a motif's structural signature.
pub fn structural_signature(motif: &Motif) -> MotifSignature {
    let total = motif.total_duration().beats().max(1e-9);
    MotifSignature {
        contour: motif.contour(),
        duration_fractions: motif
            .notes
            .iter()
            .map(|n| (n.duration.beats() / total) as f32)
            .collect(),
    }
}

impl MotifSignature {
    /// Distance in `[0, 1]`: 0 = identical shape, 1 = maximally different.
    /// Signatures of different length are treated as maximally distant —
    /// this signature space does not (yet) model fragmentation/expansion.
    pub fn distance(&self, other: &MotifSignature) -> f32 {
        if self.duration_fractions.len() != other.duration_fractions.len() {
            return 1.0;
        }
        if self.duration_fractions.is_empty() {
            return 0.0;
        }
        let contour_mismatch = if self.contour.is_empty() {
            0.0
        } else {
            let wrong = self
                .contour
                .iter()
                .zip(other.contour.iter())
                .filter(|(a, b)| a != b)
                .count();
            wrong as f32 / self.contour.len() as f32
        };
        let rhythm_mismatch: f32 = self
            .duration_fractions
            .iter()
            .zip(other.duration_fractions.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / self.duration_fractions.len() as f32;
        (0.6 * contour_mismatch + 0.4 * rhythm_mismatch.min(1.0)).clamp(0.0, 1.0)
    }
}

/// Where a motif family came from — the acquisition channel plus enough to
/// audit it later. Distinct from `symthaea-muse`'s `PieceRecipe`
/// provenance: this is motif-scoped, not piece-scoped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProvenanceSource {
    HandAuthoredArchetype { author_note: String },
    ProceduralGeneration { grammar_family: String, seed: u64 },
    CorpusExtraction { corpus: String, license: String },
    UserImport { owner: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReviewStatus {
    Unreviewed,
    PassedGates,
    RejectedDuplicate,
    RejectedLowQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifProvenance {
    pub source: ProvenanceSource,
    pub transformation_history: Vec<String>,
    pub review_status: ReviewStatus,
}

/// Independent quality axes, each `[0, 1]`. Deliberately NOT collapsed into
/// one score — a motif can be highly memorable but too rigid to develop,
/// or the reverse, and a planner picking by role needs to see that.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotifEvaluation {
    pub memorability: f32,
    pub distinctiveness: f32,
    pub developability: f32,
    pub style_fit: f32,
    pub rhythmic_identity_strength: f32,
}

impl MotifEvaluation {
    /// The release gate (design brief's closing checklist, narrowed to what
    /// is mechanically checkable without listening evidence): technically
    /// valid, distinguishable from its neighborhood, and capable of
    /// meaningful development. Thresholds are placeholders pending the
    /// listening-evidence program (recognition/recall/fatigue testing),
    /// not a claim of a musically validated cutoff. The distinctiveness
    /// floor deliberately matches `MotifFoundry`'s near-duplicate rejection
    /// cutoff: anything scored at all already cleared that bar, so this
    /// just makes the floor explicit on the evaluation itself rather than
    /// leaving it implicit in a caller's pre-check.
    pub fn passes_release_gate(&self) -> bool {
        self.rhythmic_identity_strength > 0.0
            && self.distinctiveness >= 0.15
            && self.developability >= 0.4
            && self.memorability >= 0.3
    }
}

/// A motif identity that can survive transformation: the literal notes,
/// the structural shape those notes reduce to, what it's for, what must
/// hold for a variant to still count as "it," and what's allowed to move.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifFamily {
    pub id: String,
    pub version: u32,
    pub canonical: Motif,
    pub signature: MotifSignature,
    pub role: PhraseRole,
    pub invariants: Vec<MotifInvariant>,
    pub mutable_features: Vec<MotifDimension>,
    pub permitted_transforms: Vec<ReturnTransformation>,
    pub prohibited_transforms: Vec<ReturnTransformation>,
    pub compatible_styles: Vec<Style>,
    pub provenance: MotifProvenance,
    pub evaluation: MotifEvaluation,
}

/// The full transform vocabulary a family draws `permitted`/`prohibited`
/// from — kept here so a caller can compute the complement of one to get
/// the other, rather than enumerating both by hand.
pub const ALL_TRANSFORMS: &[ReturnTransformation] = &[
    ReturnTransformation::Literal,
    ReturnTransformation::Transposed,
    ReturnTransformation::Inverted,
    ReturnTransformation::Augmented,
    ReturnTransformation::Diminished,
    ReturnTransformation::Fragmented,
    ReturnTransformation::Restored,
];

impl MotifFamily {
    pub fn new(
        id: impl Into<String>,
        canonical: Motif,
        role: PhraseRole,
        invariants: Vec<MotifInvariant>,
        permitted_transforms: Vec<ReturnTransformation>,
        compatible_styles: Vec<Style>,
        provenance: MotifProvenance,
    ) -> Self {
        let signature = structural_signature(&canonical);
        let prohibited_transforms = ALL_TRANSFORMS
            .iter()
            .filter(|t| !permitted_transforms.contains(t))
            .copied()
            .collect();
        let mutable_features = vec![
            MotifDimension::PitchLevel,
            MotifDimension::MeterPlacement,
            MotifDimension::Mode,
            MotifDimension::Instrumentation,
            MotifDimension::HarmonicSupport,
            MotifDimension::Register,
        ];
        let family = MotifFamily {
            id: id.into(),
            version: 1,
            canonical: canonical.clone(),
            signature,
            role,
            invariants,
            mutable_features,
            permitted_transforms,
            prohibited_transforms,
            compatible_styles,
            provenance,
            evaluation: MotifEvaluation {
                memorability: 0.0,
                distinctiveness: 0.0,
                developability: 0.0,
                style_fit: 0.0,
                rhythmic_identity_strength: 0.0,
            },
        };
        assert!(
            family.satisfies_invariants(&canonical),
            "a motif family's own canonical form must satisfy its declared invariants"
        );
        family
    }

    /// Does `candidate` hold every invariant this family declares? Ignores
    /// [`MotifDimension`] entirely — those are explicitly free to vary.
    pub fn satisfies_invariants(&self, candidate: &Motif) -> bool {
        self.invariants.iter().all(|inv| match inv {
            MotifInvariant::RhythmicIdentity => has_rhythmic_identity(candidate),
            MotifInvariant::ContourIdentity => has_contour_identity(candidate),
            MotifInvariant::FirstDegree(d) => {
                candidate.notes.first().and_then(|n| n.degree) == Some(*d)
            }
            MotifInvariant::LastDegree(d) => {
                candidate.notes.last().and_then(|n| n.degree) == Some(*d)
            }
            MotifInvariant::EventCount(n) => candidate.len() == *n,
        })
    }

    /// The identity contract in one call: how much has `candidate` changed
    /// relative to this family's canonical form?
    pub fn recognizes(&self, candidate: &Motif) -> IdentityVerdict {
        if candidate == &self.canonical {
            return IdentityVerdict::ExactMatch;
        }
        if !self.satisfies_invariants(candidate) {
            return IdentityVerdict::Unrelated;
        }
        let distance = self.signature.distance(&structural_signature(candidate));
        if distance <= 0.25 {
            IdentityVerdict::RecognizableDevelopment
        } else {
            IdentityVerdict::DistantRelation
        }
    }

    /// True when `other`'s canonical form is close enough to this family's
    /// that promoting both would just duplicate the library. Distance-based
    /// (not invariant-based): two families can hold different invariants
    /// yet still sound like the same idea.
    pub fn is_near_duplicate_of(&self, other: &MotifFamily) -> bool {
        self.canonical == other.canonical || self.signature.distance(&other.signature) < 0.15
    }
}

/// Mirrors [`crate::hook::HookCell::has_rhythmic_identity`] for a
/// [`Motif`]'s pitched (non-rest) events — kept as a free function so both
/// [`MotifFamily`] and the foundry generator can filter candidates before
/// a `MotifFamily` even exists to ask.
pub fn has_rhythmic_identity(motif: &Motif) -> bool {
    let mut durs: Vec<Duration> = motif
        .notes
        .iter()
        .filter(|n| n.degree.is_some())
        .map(|n| n.duration)
        .collect();
    durs.sort_by(|a, b| a.beats().total_cmp(&b.beats()));
    match (durs.first(), durs.last()) {
        (Some(&lo), Some(&hi)) => {
            hi.beats() >= lo.beats() * 2.0 - 1e-9 && hi.beats() - lo.beats() > 1e-9
        }
        _ => false,
    }
}

/// Mirrors [`crate::hook::HookCell::has_contour_identity`] exactly: an
/// immediate repeated note, a leap of a third or more answered by
/// opposite-direction motion, or a closing leap (a third or more into the
/// final event — the recoil left for the next bar to supply).
pub fn has_contour_identity(motif: &Motif) -> bool {
    let degrees = motif.degrees();
    let repeats = degrees.windows(2).any(|w| w[0] == w[1]);
    let leap_recoil = degrees.windows(3).any(|w| {
        let a = w[1] - w[0];
        let b = w[2] - w[1];
        a.abs() >= 2 && b != 0 && a.signum() != b.signum()
    });
    let closing_leap = degrees
        .windows(2)
        .last()
        .map(|w| (w[1] - w[0]).abs() >= 3)
        .unwrap_or(false);
    repeats || leap_recoil || closing_leap
}

/// Whether the family's longest note lands on the target of its biggest
/// leap — the "does the reach land on the moment you dwell on" check
/// `hook::HookCell` keeps private; reimplemented here at motif scope since
/// a `MotifFamily` candidate isn't a `HookCell`. Grammar-general (not
/// lyrical/period-specific), so any foundry generator can share it rather
/// than reimplementing per grammar.
pub(crate) fn is_reach_aligned(motif: &Motif) -> bool {
    let degrees = motif.degrees();
    if degrees.len() < 2 {
        return false;
    }
    let signature_index = (1..degrees.len())
        .max_by_key(|&i| (degrees[i] - degrees[i - 1]).abs())
        .unwrap_or(0);
    let longest_index = motif
        .notes
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.duration.beats().total_cmp(&b.1.duration.beats()))
        .map(|(i, _)| i)
        .unwrap_or(0);
    longest_index == signature_index
}

/// Grammar-general rhythmic-identity strength score, `[0, 1]`: how far the
/// longest note's duration exceeds the shortest's.
pub(crate) fn rhythmic_identity_strength(motif: &Motif) -> f32 {
    let mut beats: Vec<f64> = motif.notes.iter().map(|n| n.duration.beats()).collect();
    beats.sort_by(f64::total_cmp);
    match (beats.first(), beats.last()) {
        (Some(&lo), Some(&hi)) if lo > 0.0 => (((hi / lo) - 1.0) as f32).clamp(0.0, 1.0),
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motif::MotifNote;

    fn cell(pairs: &[(i32, Duration)]) -> Motif {
        Motif::new(
            pairs
                .iter()
                .map(|&(d, dur)| MotifNote::new(d, dur))
                .collect(),
        )
    }

    fn lament() -> Motif {
        // Short-short-LONG rhythm, reach-and-recoil contour: 1 3 1 5.
        cell(&[
            (1, Duration::eighth()),
            (3, Duration::eighth()),
            (1, Duration::eighth()),
            (5, Duration::half()),
        ])
    }

    fn family() -> MotifFamily {
        MotifFamily::new(
            "test.lament",
            lament(),
            PhraseRole::Lament,
            vec![
                MotifInvariant::RhythmicIdentity,
                MotifInvariant::ContourIdentity,
            ],
            vec![
                ReturnTransformation::Transposed,
                ReturnTransformation::Augmented,
                ReturnTransformation::Diminished,
            ],
            vec![Style::Classical],
            MotifProvenance {
                source: ProvenanceSource::HandAuthoredArchetype {
                    author_note: "test fixture".into(),
                },
                transformation_history: vec![],
                review_status: ReviewStatus::Unreviewed,
            },
        )
    }

    #[test]
    fn exact_copy_is_exact_match() {
        let f = family();
        assert_eq!(
            f.recognizes(&f.canonical.clone()),
            IdentityVerdict::ExactMatch
        );
    }

    #[test]
    fn transposed_and_augmented_is_recognizable_development() {
        let f = family();
        let varied = f.canonical.transpose(2).augment();
        assert_eq!(
            f.recognizes(&varied),
            IdentityVerdict::RecognizableDevelopment
        );
    }

    #[test]
    fn destroying_contour_makes_it_unrelated() {
        let f = family();
        // Flatten to a single repeated pitch: still holds a repeat (so a
        // naive contour check might pass) but wipe rhythmic identity too
        // by making every duration equal — invariant violated either way.
        let flat = cell(&[
            (1, Duration::quarter()),
            (1, Duration::quarter()),
            (1, Duration::quarter()),
            (1, Duration::quarter()),
        ]);
        assert_eq!(f.recognizes(&flat), IdentityVerdict::Unrelated);
    }

    #[test]
    fn distant_relation_when_shape_drifts_a_lot() {
        let f = family();
        // Same rhythm as the canonical (so RhythmicIdentity holds trivially)
        // but the contour direction is reversed at every step (Up-Down-Up
        // -> Down-Up-Down) — still passes the generic ContourIdentity
        // predicate, but shares almost nothing with THIS family's shape.
        let drifted = cell(&[
            (5, Duration::eighth()),
            (1, Duration::eighth()),
            (5, Duration::eighth()),
            (3, Duration::half()),
        ]);
        assert!(f.satisfies_invariants(&drifted));
        assert_eq!(f.recognizes(&drifted), IdentityVerdict::DistantRelation);
    }

    #[test]
    fn near_duplicate_detection_catches_transposed_copies() {
        let f = family();
        let shifted = MotifFamily::new(
            "test.lament.shifted",
            f.canonical.transpose(4),
            PhraseRole::Lament,
            vec![],
            vec![],
            vec![],
            MotifProvenance {
                source: ProvenanceSource::ProceduralGeneration {
                    grammar_family: "test".into(),
                    seed: 1,
                },
                transformation_history: vec![],
                review_status: ReviewStatus::Unreviewed,
            },
        );
        assert!(f.is_near_duplicate_of(&shifted));
    }

    #[test]
    fn prohibited_is_the_complement_of_permitted() {
        let f = family();
        for t in ALL_TRANSFORMS {
            assert_ne!(
                f.permitted_transforms.contains(t),
                f.prohibited_transforms.contains(t),
                "{t:?} must be in exactly one of permitted/prohibited"
            );
        }
    }

    #[test]
    fn release_gate_requires_all_axes() {
        let mut eval = MotifEvaluation {
            memorability: 0.9,
            distinctiveness: 0.9,
            developability: 0.9,
            style_fit: 0.9,
            rhythmic_identity_strength: 0.9,
        };
        assert!(eval.passes_release_gate());
        eval.developability = 0.1;
        assert!(!eval.passes_release_gate());
    }
}
