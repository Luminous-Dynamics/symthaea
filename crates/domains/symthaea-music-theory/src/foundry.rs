// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Motif Foundry, Phase 2: a working generator for ONE grammar family —
//! **lyrical/period** (the Classical-style antecedent/consequent phrase
//! world: `Style::Classical`, `Style::Waltz`, `Style::Folk`). Scoped
//! deliberately narrow (one family, five phrase roles) as the first proof
//! slice before generalizing to the other ~120 [`Style`] variants.
//!
//! Two of the design brief's acquisition channels are combined here, on
//! purpose: the rhythm/contour skeleton pools below are **hand-authored
//! archetypes** (channel A), and the foundry **procedurally varies** them
//! by seed-selecting skeleton pairs and (when it survives the identity
//! predicates) their inversion (channel C). Every candidate — hand-drawn
//! shape or not — is held to the same [`has_rhythmic_identity`] /
//! [`has_contour_identity`] bar [`crate::hook::HookCell`] already enforces,
//! so "hand-authored" is a source of raw material, never an exemption from
//! the gates.

use crate::motif::{Motif, MotifNote};
use crate::motif_family::{
    self, MotifDimension, MotifEvaluation, MotifFamily, MotifInvariant, MotifProvenance,
    PhraseRole, ProvenanceSource, ReviewStatus, structural_signature,
};
use crate::obligation::ReturnTransformation;
use crate::rhythm::Duration;
use crate::style::Style;

pub const LYRICAL_PERIOD_GRAMMAR: &str = "lyrical_period";

/// The styles this grammar family's motifs are built to sit under.
const HOME_STYLES: &[Style] = &[Style::Classical, Style::Waltz, Style::Folk];

/// Rhythm skeletons, `(num, den)` beat pairs, hand-authored for the
/// lyrical/period grammar (a moderate walking pace resolving to a held
/// arrival — the family's rhythmic accent, distinct from `hook.rs`'s
/// general-purpose pool).
const RHYTHMS: &[&[(i64, i64)]] = &[
    &[(1, 1), (1, 1), (2, 1)],
    &[(2, 1), (1, 1), (1, 1)],
    &[(1, 1), (2, 1), (1, 1)],
    &[(1, 2), (1, 2), (1, 1), (2, 1)],
    &[(2, 1), (1, 2), (1, 2), (1, 1)],
    &[(1, 2), (1, 2), (1, 2), (1, 2), (2, 1)],
];

// Every role below draws from its OWN 4-note contour pool. This matters
// for more than variety: `MotifSignature` is transposition/tempo-invariant
// by design (contour DIRECTION + duration ratios only — see
// `motif_family::structural_signature`), so two roles drawing from pools
// that reduce to the SAME direction pattern (Up/Down/Same per step) would
// always collide as near-duplicates regardless of which absolute degrees
// or rhythm skeleton were used. Each pool below therefore claims a
// disjoint set of 3-step direction patterns (out of 27 possible for a
// 4-note motif) — allocated by hand once here so no two roles can ever
// promote the "same idea" under different names. Every entry is valid
// under `has_contour_identity` by construction (re-checked by the filter
// in `build_candidates`, matching `hook.rs`'s discipline of filtering
// rather than trusting hand-authored data).

/// OpeningStatement: patterns Up-Down-Up, Same-Up-Down, Down-Same-Up.
/// Open endings (never the tonic) — a statement has somewhere left to go.
const OPENING_CONTOURS: &[&[i32]] = &[&[1, 2, 1, 5], &[5, 5, 6, 3], &[2, 1, 1, 5]];

/// ClimaxSeed: patterns Up-Up-Down, Same-Down-Up, Down-Up-Down. A bigger
/// reach than OpeningStatement's pool, still open-ended.
const CLIMAX_CONTOURS: &[&[i32]] = &[&[1, 2, 4, 3], &[3, 3, 1, 2], &[5, 1, 4, 3]];

/// Return: patterns Up-Down-Down, Same-Up-Up, Down-Down-Up. Open-ended,
/// deliberately disjoint from OpeningStatement's own patterns even though
/// both roles are "outward-facing" — a literal reuse of Opening's shape
/// would just BE an Opening family under a different label, not a
/// separate promotable one.
const RETURN_CONTOURS: &[&[i32]] = &[&[1, 4, 3, 2], &[1, 1, 3, 6], &[5, 3, 1, 3]];

/// Answer: patterns Down-Down-Down, Same-Same-Down, Up-Same-Same. Every
/// entry resolves to scale degree 1 (checked by `LastDegree(1)` below).
const ANSWER_CONTOURS: &[&[i32]] = &[&[8, 6, 4, 1], &[4, 4, 4, 1], &[0, 1, 1, 1]];

/// CadentialTag: patterns Down-Same-Down, Same-Down-Down, Up-Same-Down.
/// Also resolves to the tonic, but via a distinct shape from Answer's —
/// a cadential tag is a closing gesture in its own right, not just
/// another answer.
const CADENTIAL_CONTOURS: &[&[i32]] = &[&[5, 3, 3, 1], &[6, 6, 3, 1], &[0, 3, 3, 1]];

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn role_salt(role: PhraseRole) -> u64 {
    match role {
        PhraseRole::OpeningStatement => 1,
        PhraseRole::Answer => 2,
        PhraseRole::Transition => 3,
        PhraseRole::GrooveSeed => 4,
        PhraseRole::Lament => 5,
        PhraseRole::ClimaxSeed => 6,
        PhraseRole::Return => 7,
        PhraseRole::Countermelody => 8,
        PhraseRole::CadentialTag => 9,
    }
}

fn build_candidates(contours: &[&[i32]]) -> Vec<Motif> {
    let mut out = Vec::new();
    for r in RHYTHMS {
        for c in contours {
            if r.len() != c.len() {
                continue;
            }
            let notes: Vec<MotifNote> = c
                .iter()
                .zip(r.iter())
                .map(|(&deg, &(n, d))| MotifNote::new(deg, Duration::new(n, d)))
                .collect();
            let motif = Motif::new(notes);
            if motif_family::has_rhythmic_identity(&motif)
                && motif_family::has_contour_identity(&motif)
            {
                out.push(motif);
            }
        }
    }
    out
}

/// A generator plus a growing library of promoted families — the "living
/// ecology," scoped for now to a single grammar family and an in-memory
/// registry (persistence/HDC binding/Atlas UI are later phases).
#[derive(Debug, Default)]
pub struct MotifFoundry {
    registry: Vec<MotifFamily>,
}

impl MotifFoundry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn registry(&self) -> &[MotifFamily] {
        &self.registry
    }

    /// Push an already-gated family into the shared registry — the one
    /// mutation point other grammar generators (in their own files, e.g.
    /// [`crate::contrapuntal_foundry`]) need, since `registry` itself stays
    /// private to keep gate-checking a MotifFoundry-owned invariant rather
    /// than something any caller could bypass by pushing directly.
    pub(crate) fn promote(&mut self, family: MotifFamily) {
        self.registry.push(family);
    }

    /// Generate one candidate for `role` from the lyrical/period grammar,
    /// score it, and promote it into the registry if it clears the release
    /// gate and isn't a near-duplicate of anything already promoted.
    /// Returns `None` for a role this grammar doesn't have shapes for, or
    /// when the candidate is rejected (in which case nothing is added).
    pub fn generate_lyrical_period(&mut self, seed: u64, role: PhraseRole) -> Option<MotifFamily> {
        let (contours, closes_on_tonic) = match role {
            PhraseRole::OpeningStatement => (OPENING_CONTOURS, false),
            PhraseRole::ClimaxSeed => (CLIMAX_CONTOURS, false),
            PhraseRole::Return => (RETURN_CONTOURS, false),
            PhraseRole::Answer => (ANSWER_CONTOURS, true),
            PhraseRole::CadentialTag => (CADENTIAL_CONTOURS, true),
            _ => return None, // outside this grammar's covered roles
        };

        let candidates = build_candidates(contours);
        debug_assert!(
            !candidates.is_empty(),
            "lyrical/period grammar produced no valid candidates for {role:?}"
        );
        if candidates.is_empty() {
            return None;
        }
        let pick = (splitmix64(seed ^ role_salt(role)) as usize) % candidates.len();
        let canonical = candidates[pick].clone();

        // Distinctiveness against everything already promoted, computed
        // BEFORE construction so a near-duplicate can be rejected outright.
        let candidate_signature = structural_signature(&canonical);
        let min_distance = self
            .registry
            .iter()
            .map(|f| candidate_signature.distance(&f.signature))
            .fold(1.0_f32, f32::min);
        if !self.registry.is_empty() && min_distance < 0.15 {
            return None; // near-duplicate of an already-promoted family
        }

        let mut invariants = vec![
            MotifInvariant::RhythmicIdentity,
            MotifInvariant::ContourIdentity,
            MotifInvariant::FirstDegree(canonical.notes[0].degree.unwrap_or(1)),
        ];
        if closes_on_tonic {
            invariants.push(MotifInvariant::LastDegree(1));
        }

        let mut permitted = vec![
            ReturnTransformation::Transposed,
            ReturnTransformation::Augmented,
            ReturnTransformation::Diminished,
            ReturnTransformation::Fragmented,
        ];
        let inverted = canonical.invert(canonical.notes[0].degree.unwrap_or(1));
        if motif_family::has_rhythmic_identity(&inverted)
            && motif_family::has_contour_identity(&inverted)
        {
            permitted.push(ReturnTransformation::Inverted);
        }
        if role == PhraseRole::Return {
            permitted.push(ReturnTransformation::Restored);
        }

        let fitting_styles: Vec<Style> = HOME_STYLES
            .iter()
            .filter(|s| canonical.total_duration().beats() <= s.spec().meter as f64)
            .copied()
            .collect();
        let style_fit = fitting_styles.len() as f32 / HOME_STYLES.len() as f32;

        let strength = motif_family::rhythmic_identity_strength(&canonical);
        let memorability = (strength
            + if motif_family::is_reach_aligned(&canonical) {
                1.0
            } else {
                0.0
            })
            / 2.0;
        let developability = permitted.len() as f32 / motif_family::ALL_TRANSFORMS.len() as f32;
        let distinctiveness = if self.registry.is_empty() {
            1.0
        } else {
            min_distance
        };

        let mut family = MotifFamily::new(
            format!("{LYRICAL_PERIOD_GRAMMAR}.{role:?}.{seed:016x}"),
            canonical,
            role,
            invariants,
            permitted,
            fitting_styles,
            MotifProvenance {
                source: ProvenanceSource::ProceduralGeneration {
                    grammar_family: LYRICAL_PERIOD_GRAMMAR.into(),
                    seed,
                },
                transformation_history: Vec::new(),
                review_status: ReviewStatus::Unreviewed,
            },
        );
        family
            .mutable_features
            .retain(|d| *d != MotifDimension::MeterPlacement);
        family.evaluation = MotifEvaluation {
            memorability,
            distinctiveness,
            developability,
            style_fit,
            rhythmic_identity_strength: strength,
        };

        if !family.evaluation.passes_release_gate() {
            return None;
        }
        family.provenance.review_status = ReviewStatus::PassedGates;
        self.registry.push(family.clone());
        Some(family)
    }

    /// Sweep a range of seeds for `role`, promoting every candidate that
    /// clears the gate. Returns how many were actually promoted (fewer
    /// than `attempts` is expected — that is the gate doing its job, not a
    /// bug).
    pub fn generate_many(&mut self, attempts: u64, role: PhraseRole, seed_base: u64) -> usize {
        let mut promoted = 0;
        for i in 0..attempts {
            if self
                .generate_lyrical_period(seed_base.wrapping_add(i), role)
                .is_some()
            {
                promoted += 1;
            }
        }
        promoted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motif_family::IdentityVerdict;

    #[test]
    fn generates_and_promotes_families_for_covered_roles() {
        let mut foundry = MotifFoundry::new();
        for role in [
            PhraseRole::OpeningStatement,
            PhraseRole::Answer,
            PhraseRole::ClimaxSeed,
            PhraseRole::Return,
            PhraseRole::CadentialTag,
        ] {
            let promoted = foundry.generate_many(24, role, role_salt(role) * 1000);
            assert!(promoted > 0, "{role:?} promoted nothing across 24 seeds");
        }
        assert!(foundry.registry().len() >= 5);
    }

    #[test]
    fn unsupported_role_returns_none() {
        let mut foundry = MotifFoundry::new();
        assert!(
            foundry
                .generate_lyrical_period(1, PhraseRole::GrooveSeed)
                .is_none()
        );
    }

    #[test]
    fn same_seed_and_role_is_deterministic() {
        let mut a = MotifFoundry::new();
        let mut b = MotifFoundry::new();
        let fam_a = a.generate_lyrical_period(42, PhraseRole::OpeningStatement);
        let fam_b = b.generate_lyrical_period(42, PhraseRole::OpeningStatement);
        assert_eq!(fam_a.map(|f| f.canonical), fam_b.map(|f| f.canonical));
    }

    #[test]
    fn promoted_families_pass_their_own_release_gate() {
        let mut foundry = MotifFoundry::new();
        foundry.generate_many(30, PhraseRole::OpeningStatement, 7);
        assert!(!foundry.registry().is_empty());
        for family in foundry.registry() {
            assert!(family.evaluation.passes_release_gate());
            assert_eq!(family.provenance.review_status, ReviewStatus::PassedGates);
            assert!(family.satisfies_invariants(&family.canonical));
        }
    }

    #[test]
    fn registry_stays_pairwise_distinct() {
        let mut foundry = MotifFoundry::new();
        foundry.generate_many(40, PhraseRole::Answer, 99);
        let registry = foundry.registry();
        for i in 0..registry.len() {
            for j in (i + 1)..registry.len() {
                assert_ne!(
                    registry[i].recognizes(&registry[j].canonical),
                    IdentityVerdict::ExactMatch,
                    "registry entries {i} and {j} are literal duplicates"
                );
                assert!(!registry[i].is_near_duplicate_of(&registry[j]));
            }
        }
    }

    #[test]
    fn opening_statement_stays_open_and_answer_closes() {
        let mut foundry = MotifFoundry::new();
        let opening = foundry
            .generate_lyrical_period(5, PhraseRole::OpeningStatement)
            .expect("opening candidate should be generatable");
        assert_ne!(opening.canonical.notes.last().unwrap().degree, Some(1));

        let answer = foundry
            .generate_lyrical_period(5, PhraseRole::Answer)
            .expect("answer candidate should be generatable");
        assert_eq!(answer.canonical.notes.last().unwrap().degree, Some(1));
        assert!(answer.invariants.contains(&MotifInvariant::LastDegree(1)));
    }

    /// `HOME_STYLES` must stay a subset of the canonical `Style::grammar_family()`
    /// mapping's `PeriodSentence` family — otherwise this generator and the
    /// crate's own grammar-family taxonomy silently disagree about which
    /// styles the lyrical/period grammar belongs to.
    #[test]
    fn home_styles_agree_with_the_canonical_grammar_family() {
        use crate::grammar::GrammarFamily;
        for style in HOME_STYLES {
            assert_eq!(
                style.grammar_family(),
                GrammarFamily::PeriodSentence,
                "{style:?} is in foundry.rs's HOME_STYLES but Style::grammar_family() \
                 no longer maps it to PeriodSentence"
            );
        }
    }
}
