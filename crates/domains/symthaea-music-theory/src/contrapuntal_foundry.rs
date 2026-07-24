// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Motif Foundry, second grammar family: **contrapuntal** subjects and
//! countersubjects (`Style::Fugue`) — per the Motif Foundry design brief's
//! "contrapuntal motifs — subjects, countersubjects, imitation cells" class.
//! Deliberately scoped to the one style [`crate::fugue`] actually realizes
//! (a fughetta); `RenaissancePolyphony`/`SacredChoral` use a different,
//! equal-voice imitation texture ([`crate::renaissance`]) with its own
//! devices and are left for a future generator rather than force-fit here.
//!
//! Reuses rather than reinvents: [`crate::fugue::answer`] and
//! [`crate::fugue::countersubject`] ARE the transform vocabulary — a
//! subject's `permitted_transforms` map onto the exposition's real devices
//! (answer = [`crate::obligation::ReturnTransformation::Transposed`],
//! middle entry = `Inverted`, final entry = `Augmented`, episode sequencing
//! = `Fragmented`), and the identity predicates
//! ([`crate::motif_family::has_rhythmic_identity`]/`has_contour_identity`)
//! are the same bar [`foundry::MotifFoundry`](crate::foundry)'s lyrical/
//! period generator holds candidates to.
//!
//! A subject and its countersubject are promoted as TWO separate
//! [`MotifFamily`] entries (roles [`PhraseRole::OpeningStatement`] and
//! [`PhraseRole::Countermelody`] respectively) into the SAME shared
//! registry [`crate::foundry::MotifFoundry`] already uses for lyrical/
//! period — a fugue subject and a period opening statement legitimately
//! share a phrase role; what differs is `compatible_styles` and the
//! transform vocabulary, exactly the axes the schema separates.

use crate::foundry::MotifFoundry;
use crate::motif::{Motif, MotifNote};
use crate::motif_family::{
    self, MotifEvaluation, MotifFamily, MotifInvariant, MotifProvenance, PhraseRole,
    ProvenanceSource, ReviewStatus, structural_signature,
};
use crate::obligation::ReturnTransformation;
use crate::rhythm::Duration;
use crate::style::Style;

pub const CONTRAPUNTAL_GRAMMAR: &str = "contrapuntal";

/// Rhythm skeletons for a fughetta subject — a more running, motoric
/// character than lyrical/period's walking-pace pool (a subject drives an
/// exposition rather than resolving a phrase), and deliberately its own
/// pool rather than reusing [`crate::foundry`]'s: each grammar owns its own
/// rhythmic vocabulary per the design brief (section 4C).
const SUBJECT_RHYTHMS: &[&[(i64, i64)]] = &[
    &[(1, 1), (1, 2), (1, 2), (2, 1)],
    &[(1, 2), (1, 2), (1, 1), (2, 1)],
    &[(2, 1), (1, 1), (1, 2), (1, 2)],
    &[(1, 2), (1, 1), (1, 2), (2, 1)],
];

/// Subject contours: triadic outlines with a recoil, the classic fugue-
/// subject shape (leap that immediately answers itself), distinct from
/// lyrical/period's stepwise phrase contours.
const SUBJECT_CONTOURS: &[&[i32]] = &[&[1, 3, 5, 4], &[5, 3, 1, 2], &[1, 5, 4, 3], &[3, 1, 2, 5]];

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const SUBJECT_SALT: u64 = 0xC0FF_EE00_1234_5678;
const COUNTERSUBJECT_SALT: u64 = 0xC0FF_EE00_8765_4321;

fn build_subject_candidates() -> Vec<Motif> {
    let mut out = Vec::new();
    for r in SUBJECT_RHYTHMS {
        for c in SUBJECT_CONTOURS {
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

fn score(
    canonical: &Motif,
    permitted: &[ReturnTransformation],
    distinctiveness: f32,
) -> MotifEvaluation {
    let strength = motif_family::rhythmic_identity_strength(canonical);
    let memorability = (strength
        + if motif_family::is_reach_aligned(canonical) {
            1.0
        } else {
            0.0
        })
        / 2.0;
    let developability = permitted.len() as f32 / motif_family::ALL_TRANSFORMS.len() as f32;
    MotifEvaluation {
        memorability,
        distinctiveness,
        developability,
        style_fit: if canonical.total_duration().beats() <= Style::Fugue.spec().meter as f64 {
            1.0
        } else {
            0.0
        },
        rhythmic_identity_strength: strength,
    }
}

impl MotifFoundry {
    /// Generate one fughetta subject candidate, score it, and promote it
    /// into the shared registry if it clears the release gate, isn't a
    /// near-duplicate of anything already promoted (any grammar), AND its
    /// derived countersubject ([`crate::fugue::countersubject`]) itself
    /// carries real rhythmic/contour identity — a subject whose
    /// countersubject collapses into undifferentiated motion is not usable
    /// in a real exposition, so this is a hard gate, not an optional
    /// permitted-transform check the way subject-inversion is below.
    pub fn generate_contrapuntal_subject(&mut self, seed: u64) -> Option<MotifFamily> {
        let candidates = build_subject_candidates();
        debug_assert!(
            !candidates.is_empty(),
            "contrapuntal grammar produced no valid subject candidates"
        );
        if candidates.is_empty() {
            return None;
        }
        let pick = (splitmix64(seed ^ SUBJECT_SALT) as usize) % candidates.len();
        let canonical = candidates[pick].clone();

        let candidate_signature = structural_signature(&canonical);
        let min_distance = self
            .registry()
            .iter()
            .map(|f| candidate_signature.distance(&f.signature))
            .fold(1.0_f32, f32::min);
        if !self.registry().is_empty() && min_distance < 0.15 {
            return None; // near-duplicate of an already-promoted family
        }

        let countersubject = crate::fugue::countersubject(&canonical);
        if !(motif_family::has_rhythmic_identity(&countersubject)
            && motif_family::has_contour_identity(&countersubject))
        {
            return None;
        }

        let pivot = canonical.notes[0].degree.unwrap_or(1);
        let mut permitted = vec![
            ReturnTransformation::Transposed, // the answer, up a diatonic fifth
            ReturnTransformation::Augmented,  // the final entry, doubled note values
            ReturnTransformation::Fragmented, // episode head-fragment sequencing
        ];
        let inverted = canonical.invert(pivot);
        if motif_family::has_rhythmic_identity(&inverted)
            && motif_family::has_contour_identity(&inverted)
        {
            permitted.push(ReturnTransformation::Inverted); // the middle entry
        }

        let distinctiveness = if self.registry().is_empty() {
            1.0
        } else {
            min_distance
        };
        let evaluation = score(&canonical, &permitted, distinctiveness);
        if !evaluation.passes_release_gate() {
            return None;
        }

        let family = MotifFamily::new(
            format!("{CONTRAPUNTAL_GRAMMAR}.subject.{seed:016x}"),
            canonical,
            PhraseRole::OpeningStatement,
            vec![
                MotifInvariant::RhythmicIdentity,
                MotifInvariant::ContourIdentity,
                MotifInvariant::FirstDegree(pivot),
            ],
            permitted,
            vec![Style::Fugue],
            MotifProvenance {
                source: ProvenanceSource::ProceduralGeneration {
                    grammar_family: CONTRAPUNTAL_GRAMMAR.into(),
                    seed,
                },
                transformation_history: Vec::new(),
                review_status: ReviewStatus::Unreviewed,
            },
        );
        let mut family = family;
        family.evaluation = evaluation;
        family.provenance.review_status = ReviewStatus::PassedGates;
        self.promote(family.clone());
        Some(family)
    }

    /// Derive and promote `subject`'s countersubject
    /// ([`crate::fugue::countersubject`]: retrograde inversion transposed
    /// up a third) as its own [`MotifFamily`] — [`PhraseRole::Countermelody`],
    /// provenance pointing back at the subject. A countersubject's entry
    /// pitch is context-dependent on which voice states it, so unlike the
    /// subject this carries no [`MotifInvariant::FirstDegree`]; and in this
    /// fughetta's device set a countersubject appears transposed alongside
    /// subject/answer entries but is never independently inverted or
    /// augmented, so `permitted_transforms` is narrower than the subject's.
    pub fn generate_contrapuntal_countersubject(
        &mut self,
        subject: &MotifFamily,
        seed: u64,
    ) -> Option<MotifFamily> {
        let canonical = crate::fugue::countersubject(&subject.canonical);
        if !(motif_family::has_rhythmic_identity(&canonical)
            && motif_family::has_contour_identity(&canonical))
        {
            return None;
        }

        let candidate_signature = structural_signature(&canonical);
        let min_distance = self
            .registry()
            .iter()
            .map(|f| candidate_signature.distance(&f.signature))
            .fold(1.0_f32, f32::min);
        if !self.registry().is_empty() && min_distance < 0.15 {
            return None;
        }

        let permitted = vec![
            ReturnTransformation::Transposed,
            ReturnTransformation::Fragmented,
        ];
        let distinctiveness = if self.registry().is_empty() {
            1.0
        } else {
            min_distance
        };
        let evaluation = score(&canonical, &permitted, distinctiveness);
        // NOT `evaluation.passes_release_gate()`: that gate's developability
        // (>= 0.4) and memorability (>= 0.3) floors are tuned for primary
        // thematic material meant to carry a whole piece. A countersubject's
        // job is to accompany, not to be independently developable — this
        // fughetta's device set only ever transposes/fragments one
        // (`permitted` above), which caps developability at 2/7 ≈ 0.29 by
        // construction, not as a quality defect. What DOES matter for
        // accompanying material: real rhythmic identity (already required to
        // reach this point via `has_rhythmic_identity` above) and genuine
        // distinctiveness from whatever's already promoted.
        if evaluation.rhythmic_identity_strength <= 0.0 || evaluation.distinctiveness < 0.15 {
            return None;
        }

        let mut family = MotifFamily::new(
            format!("{CONTRAPUNTAL_GRAMMAR}.countersubject.{seed:016x}"),
            canonical,
            PhraseRole::Countermelody,
            vec![
                MotifInvariant::RhythmicIdentity,
                MotifInvariant::ContourIdentity,
            ],
            permitted,
            vec![Style::Fugue],
            MotifProvenance {
                source: ProvenanceSource::ProceduralGeneration {
                    grammar_family: CONTRAPUNTAL_GRAMMAR.into(),
                    seed,
                },
                transformation_history: vec![format!("countersubject of {}", subject.id)],
                review_status: ReviewStatus::Unreviewed,
            },
        );
        family.evaluation = evaluation;
        family.provenance.review_status = ReviewStatus::PassedGates;
        self.promote(family.clone());
        Some(family)
    }

    /// Sweep a range of seeds, promoting every subject that clears the
    /// gate AND has a promotable countersubject, returning `(subject,
    /// countersubject)` pairs. Fewer than `attempts` pairs is expected —
    /// the gates doing their job, not a bug.
    pub fn generate_contrapuntal_pairs(
        &mut self,
        attempts: u64,
        seed_base: u64,
    ) -> Vec<(MotifFamily, MotifFamily)> {
        let mut pairs = Vec::new();
        for i in 0..attempts {
            let seed = seed_base.wrapping_add(i);
            if let Some(subject) = self.generate_contrapuntal_subject(seed)
                && let Some(countersubject) =
                    self.generate_contrapuntal_countersubject(&subject, seed ^ COUNTERSUBJECT_SALT)
            {
                pairs.push((subject, countersubject));
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motif_family::IdentityVerdict;

    #[test]
    fn generates_and_promotes_a_subject() {
        let mut foundry = MotifFoundry::new();
        let promoted = (0..30)
            .filter(|&i| foundry.generate_contrapuntal_subject(i * 1013).is_some())
            .count();
        assert!(promoted > 0, "no subject promoted across 30 seeds");
    }

    #[test]
    fn same_seed_is_deterministic() {
        let mut a = MotifFoundry::new();
        let mut b = MotifFoundry::new();
        let fam_a = a.generate_contrapuntal_subject(4242);
        let fam_b = b.generate_contrapuntal_subject(4242);
        assert_eq!(fam_a.map(|f| f.canonical), fam_b.map(|f| f.canonical));
    }

    #[test]
    fn subject_carries_fugue_transform_vocabulary() {
        let mut foundry = MotifFoundry::new();
        let subject = (0..30)
            .find_map(|i| foundry.generate_contrapuntal_subject(i * 1013))
            .expect("at least one subject should be generatable");
        assert!(
            subject
                .permitted_transforms
                .contains(&ReturnTransformation::Transposed)
        );
        assert!(
            subject
                .permitted_transforms
                .contains(&ReturnTransformation::Augmented)
        );
        assert_eq!(subject.compatible_styles, vec![Style::Fugue]);
        assert_eq!(subject.role, PhraseRole::OpeningStatement);
    }

    #[test]
    fn countersubject_is_derived_and_recognizably_distinct_from_subject() {
        let mut foundry = MotifFoundry::new();
        let subject = (0..30)
            .find_map(|i| foundry.generate_contrapuntal_subject(i * 1013))
            .expect("at least one subject should be generatable");
        let countersubject = foundry
            .generate_contrapuntal_countersubject(&subject, 99)
            .expect("this subject's countersubject should be promotable");
        assert_eq!(countersubject.role, PhraseRole::Countermelody);
        assert_ne!(
            subject.recognizes(&countersubject.canonical),
            IdentityVerdict::ExactMatch,
            "a countersubject must not literally be its own subject"
        );
        assert!(
            countersubject
                .provenance
                .transformation_history
                .iter()
                .any(|h| h.contains(&subject.id))
        );
    }

    #[test]
    fn pairs_share_one_registry_with_lyrical_period() {
        let mut foundry = MotifFoundry::new();
        foundry.generate_many(20, PhraseRole::OpeningStatement, 7);
        let before = foundry.registry().len();
        let pairs = foundry.generate_contrapuntal_pairs(30, 555);
        assert!(
            !pairs.is_empty(),
            "no subject/countersubject pairs promoted"
        );
        assert_eq!(foundry.registry().len(), before + pairs.len() * 2);
    }

    /// This generator's hardcoded `vec![Style::Fugue]` compatible-styles
    /// list must stay in agreement with the canonical
    /// `Style::grammar_family()` mapping — `Style::Fugue` is the ONLY style
    /// mapped to `GrammarFamily::Contrapuntal` precisely because that
    /// family's `compose_with_grammar_plan` dispatch is hardcoded to the
    /// fugue engine (see the doc comment on `Style::grammar_family`).
    #[test]
    fn compatible_styles_agree_with_the_canonical_grammar_family() {
        use crate::grammar::GrammarFamily;
        assert_eq!(Style::Fugue.grammar_family(), GrammarFamily::Contrapuntal);
    }
}
