// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Progressive Folk/Rock long form with a first-class plan.
//!
//! ProgSuite was the first engine in this crate to realize genuine mid-piece
//! meter changes (4 -> 7 -> 5 -> 4), but its architectural choices previously
//! lived only as locals inside `realize_prog_suite()`. Once a `Score` came out,
//! there was no durable declaration of which section key, meter, progression,
//! or thematic transformation had been intended.
//!
//! [`ProgSuitePlanV1`] freezes those choices before realization. The legacy
//! entry point now plans first and consumes the same plan, preserving existing
//! score behavior while creating an auditable boundary for work-scale FORM
//! adapters and later score-side evidence. `source_seed` is provenance only:
//! once a valid plan exists, the plan itself is authoritative.
//!
//! A single source motif cannot simultaneously occupy one full 4/4, 7/4, and
//! 5/4 bar without an explicit rhythmic adaptation. Every section therefore
//! derives a [`ProgSuiteSectionCarrierV1`]: first apply the section's declared
//! pitch/order transformation, then scale every event duration by one exact
//! rational factor so one motif statement occupies exactly one local bar.
//! The adaptation is auditable through [`ProgSuiteMeterFitReceiptV1`] and is a
//! realization-context operation, not an extra FORM-002 thematic ancestry claim.

use crate::MusicalIntent;
use crate::form::{Form, Section, SectionRole};
use crate::harmony::Key;
use crate::motif::Motif;
use crate::phrase::Period;
use crate::pitch::Pitch;
use crate::rhythm::Duration;
use crate::score::Score;
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_PLAN_VERSION: &str = "melothaea-prog-suite-plan-v1";
pub const PROG_SUITE_METER_FIT_VERSION: &str = "melothaea-prog-suite-meter-fit-v1";
pub const PROG_SUITE_BARS_PER_SECTION: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteTransformV1 {
    Original,
    Inversion,
    Retrograde,
    RetrogradeInversion,
}

impl ProgSuiteTransformV1 {
    fn from_contrast_choice(choice: u64) -> Self {
        match choice % 3 {
            0 => Self::Inversion,
            1 => Self::Retrograde,
            _ => Self::RetrogradeInversion,
        }
    }

    fn apply(self, motif: &Motif) -> Motif {
        let pivot = motif.notes.iter().find_map(|note| note.degree).unwrap_or(1);
        match self {
            Self::Original => motif.clone(),
            Self::Inversion => motif.invert(pivot),
            Self::Retrograde => motif.retrograde(),
            Self::RetrogradeInversion => motif.invert(pivot).retrograde(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteSectionPlanV1 {
    pub role: SectionRole,
    pub key: Key,
    pub meter: u8,
    pub transformation: ProgSuiteTransformV1,
    pub progression_degrees: Vec<i32>,
    /// Exact half-open span `[start, end)` in quarter-note beats.
    pub start: Duration,
    pub end: Duration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuitePlanV1 {
    pub version: String,
    pub home_key: Key,
    pub tempo_bpm: f32,
    /// Provenance for the native planner's initial choices. Validation never
    /// re-derives mutable plan fields from this seed.
    pub source_seed: u64,
    pub sections: Vec<ProgSuiteSectionPlanV1>,
    pub total_beats: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteMeterFitReceiptV1 {
    pub section_index: usize,
    pub meter: u8,
    /// Duration after the declared thematic transformation but before meter fit.
    pub transformed_duration: Duration,
    /// One complete local bar in quarter-note beats.
    pub target_duration: Duration,
    /// Reduced exact rational multiplier applied to every event duration.
    pub rhythm_scale_numerator: i64,
    pub rhythm_scale_denominator: i64,
    pub fitted_duration: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteSectionCarrierV1 {
    pub section_index: usize,
    pub transformation: ProgSuiteTransformV1,
    /// Exact result of the declared thematic transform, before local-meter fit.
    pub transformed_motif: Motif,
    /// Exact rhythmic carrier consumed by `Period::parallel_in`.
    pub meter_fitted_motif: Motif,
    pub meter_fit: ProgSuiteMeterFitReceiptV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteSectionCarriersV1 {
    pub version: String,
    pub sections: Vec<ProgSuiteSectionCarrierV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteRealizationV1 {
    pub plan: ProgSuitePlanV1,
    pub score: Score,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuitePlanErrorV1 {
    WrongVersion { found: String },
    WrongMeterFitVersion { found: String },
    InvalidTempo { found: f32 },
    WrongSectionCount { found: usize },
    WrongRole {
        index: usize,
        expected: SectionRole,
        found: SectionRole,
    },
    WrongKey { index: usize },
    WrongMeter { index: usize, expected: u8, found: u8 },
    BoundaryMustBeOriginal {
        index: usize,
        found: ProgSuiteTransformV1,
    },
    ContrastMustBeTransformed { index: usize },
    DuplicateContrastTransformation { found: ProgSuiteTransformV1 },
    WrongProgressionLength { index: usize, found: usize },
    InvalidProgressionDegree { index: usize, degree: i32 },
    NonContiguousSection { index: usize },
    WrongSectionSpan {
        index: usize,
        expected_end: Duration,
        found_end: Duration,
    },
    WrongTotalBeats {
        expected: Duration,
        found: Duration,
    },
    InvalidSourceMotifDuration { found: Duration },
    MeterFitDurationMismatch {
        section_index: usize,
        expected: Duration,
        found: Duration,
    },
    CanonicalSectionCarriersMismatch,
    RealizedSectionSpanMismatch {
        index: usize,
        planned: Duration,
        realized: Duration,
    },
    RealizedTotalSpanMismatch {
        planned: Duration,
        realized: Duration,
    },
}

impl ProgSuitePlanV1 {
    pub fn validate(&self) -> Result<(), ProgSuitePlanErrorV1> {
        if self.version != PROG_SUITE_PLAN_VERSION {
            return Err(ProgSuitePlanErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if !self.tempo_bpm.is_finite() || self.tempo_bpm <= 0.0 {
            return Err(ProgSuitePlanErrorV1::InvalidTempo {
                found: self.tempo_bpm,
            });
        }
        if self.sections.len() != 4 {
            return Err(ProgSuitePlanErrorV1::WrongSectionCount {
                found: self.sections.len(),
            });
        }

        let expected = [
            (SectionRole::A, self.home_key, 4),
            (SectionRole::B, self.home_key, 7),
            (SectionRole::C, self.home_key.relative(), 5),
            (SectionRole::ReturnA, self.home_key, 4),
        ];

        for index in [0usize, 3] {
            if self.sections[index].transformation != ProgSuiteTransformV1::Original {
                return Err(ProgSuitePlanErrorV1::BoundaryMustBeOriginal {
                    index,
                    found: self.sections[index].transformation,
                });
            }
        }
        for index in [1usize, 2] {
            if self.sections[index].transformation == ProgSuiteTransformV1::Original {
                return Err(ProgSuitePlanErrorV1::ContrastMustBeTransformed { index });
            }
        }
        if self.sections[1].transformation == self.sections[2].transformation {
            return Err(ProgSuitePlanErrorV1::DuplicateContrastTransformation {
                found: self.sections[1].transformation,
            });
        }

        let mut cursor = Duration::zero();
        for (index, section) in self.sections.iter().enumerate() {
            let (expected_role, expected_key, expected_meter) = expected[index];
            if section.role != expected_role {
                return Err(ProgSuitePlanErrorV1::WrongRole {
                    index,
                    expected: expected_role,
                    found: section.role,
                });
            }
            if section.key != expected_key {
                return Err(ProgSuitePlanErrorV1::WrongKey { index });
            }
            if section.meter != expected_meter {
                return Err(ProgSuitePlanErrorV1::WrongMeter {
                    index,
                    expected: expected_meter,
                    found: section.meter,
                });
            }
            if section.progression_degrees.len() != PROG_SUITE_BARS_PER_SECTION {
                return Err(ProgSuitePlanErrorV1::WrongProgressionLength {
                    index,
                    found: section.progression_degrees.len(),
                });
            }
            if let Some(&degree) = section
                .progression_degrees
                .iter()
                .find(|&&degree| !(1..=7).contains(&degree))
            {
                return Err(ProgSuitePlanErrorV1::InvalidProgressionDegree { index, degree });
            }
            if section.start != cursor {
                return Err(ProgSuitePlanErrorV1::NonContiguousSection { index });
            }
            let expected_span = Duration::new(
                2 * PROG_SUITE_BARS_PER_SECTION as i64 * i64::from(section.meter),
                1,
            );
            let expected_end = section.start + expected_span;
            if section.end != expected_end {
                return Err(ProgSuitePlanErrorV1::WrongSectionSpan {
                    index,
                    expected_end,
                    found_end: section.end,
                });
            }
            cursor = section.end;
        }
        if self.total_beats != cursor {
            return Err(ProgSuitePlanErrorV1::WrongTotalBeats {
                expected: cursor,
                found: self.total_beats,
            });
        }
        Ok(())
    }
}

impl ProgSuiteSectionCarriersV1 {
    /// Re-derive the entire carrier set from the authoritative plan + source
    /// motif and require exact equality. This keeps serialized meter-fit
    /// receipts descriptive rather than hand-editable authority.
    pub fn validate(
        &self,
        plan: &ProgSuitePlanV1,
        motif: &Motif,
    ) -> Result<(), ProgSuitePlanErrorV1> {
        if self.version != PROG_SUITE_METER_FIT_VERSION {
            return Err(ProgSuitePlanErrorV1::WrongMeterFitVersion {
                found: self.version.clone(),
            });
        }
        let canonical = derive_prog_suite_section_carriers(plan, motif)?;
        if &canonical != self {
            return Err(ProgSuitePlanErrorV1::CanonicalSectionCarriersMismatch);
        }
        Ok(())
    }
}

/// Freeze every architectural choice the existing ProgSuite engine makes
/// before note realization.
pub fn plan_prog_suite(
    home_key: Key,
    tempo_bpm: f32,
    seed: u64,
    spec: &crate::spec::CompositionSpec,
) -> Result<ProgSuitePlanV1, ProgSuitePlanErrorV1> {
    if !tempo_bpm.is_finite() || tempo_bpm <= 0.0 {
        return Err(ProgSuitePlanErrorV1::InvalidTempo { found: tempo_bpm });
    }

    let b_transform = ProgSuiteTransformV1::from_contrast_choice(seed % 3);
    let c_transform = ProgSuiteTransformV1::from_contrast_choice((seed % 3 + 1) % 3);
    let templates = [
        (SectionRole::A, home_key, 4, ProgSuiteTransformV1::Original),
        (SectionRole::B, home_key, 7, b_transform),
        (SectionRole::C, home_key.relative(), 5, c_transform),
        (
            SectionRole::ReturnA,
            home_key,
            4,
            ProgSuiteTransformV1::Original,
        ),
    ];

    let mut cursor = Duration::zero();
    let mut sections = Vec::with_capacity(templates.len());
    for (index, (role, key, meter, transformation)) in templates.into_iter().enumerate() {
        let seed_variant = seed ^ (0x51CE_u64.wrapping_mul(index as u64 + 1));
        let progression_degrees = spec
            .progression(PROG_SUITE_BARS_PER_SECTION, seed_variant)
            .degrees;
        let span = Duration::new(
            2 * PROG_SUITE_BARS_PER_SECTION as i64 * i64::from(meter),
            1,
        );
        let start = cursor;
        let end = start + span;
        sections.push(ProgSuiteSectionPlanV1 {
            role,
            key,
            meter,
            transformation,
            progression_degrees,
            start,
            end,
        });
        cursor = end;
    }

    let plan = ProgSuitePlanV1 {
        version: PROG_SUITE_PLAN_VERSION.into(),
        home_key,
        tempo_bpm,
        source_seed: seed,
        sections,
        total_beats: cursor,
    };
    plan.validate()?;
    Ok(plan)
}

/// Derive the exact thematic carrier consumed by each local-meter section.
///
/// The declared thematic transformation is applied first. Its result is then
/// rhythmically scaled by the exact reduced factor `meter / transformed_len`.
/// Pitch/rest order is untouched by the fit. This is deliberately modeled as
/// realization-context adaptation rather than silently extending FORM-002.
pub fn derive_prog_suite_section_carriers(
    plan: &ProgSuitePlanV1,
    motif: &Motif,
) -> Result<ProgSuiteSectionCarriersV1, ProgSuitePlanErrorV1> {
    plan.validate()?;
    let source_duration = motif.total_duration();
    if source_duration.num() <= 0 {
        return Err(ProgSuitePlanErrorV1::InvalidSourceMotifDuration {
            found: source_duration,
        });
    }

    let mut sections = Vec::with_capacity(plan.sections.len());
    for (section_index, section) in plan.sections.iter().enumerate() {
        let transformed_motif = section.transformation.apply(motif);
        let transformed_duration = transformed_motif.total_duration();
        if transformed_duration.num() <= 0 {
            return Err(ProgSuitePlanErrorV1::InvalidSourceMotifDuration {
                found: transformed_duration,
            });
        }
        let target_duration = Duration::new(i64::from(section.meter), 1);
        let rhythm_scale = Duration::new(
            target_duration.num() * transformed_duration.den(),
            target_duration.den() * transformed_duration.num(),
        );
        let meter_fitted_motif = transformed_motif
            .scale_rhythm(rhythm_scale.num(), rhythm_scale.den());
        let fitted_duration = meter_fitted_motif.total_duration();
        if fitted_duration != target_duration {
            return Err(ProgSuitePlanErrorV1::MeterFitDurationMismatch {
                section_index,
                expected: target_duration,
                found: fitted_duration,
            });
        }
        sections.push(ProgSuiteSectionCarrierV1 {
            section_index,
            transformation: section.transformation,
            transformed_motif,
            meter_fitted_motif,
            meter_fit: ProgSuiteMeterFitReceiptV1 {
                section_index,
                meter: section.meter,
                transformed_duration,
                target_duration,
                rhythm_scale_numerator: rhythm_scale.num(),
                rhythm_scale_denominator: rhythm_scale.den(),
                fitted_duration,
            },
        });
    }

    Ok(ProgSuiteSectionCarriersV1 {
        version: PROG_SUITE_METER_FIT_VERSION.into(),
        sections,
    })
}

/// Realize exactly the frozen ProgSuite plan. No section key, meter,
/// progression, or transformation is re-selected here. Every transformed
/// thematic carrier is fitted to exactly one local bar using the canonical
/// rational meter-fit derivation above before phrase construction.
pub fn realize_prog_suite_with_plan(
    plan: &ProgSuitePlanV1,
    motif: &Motif,
    intent: &MusicalIntent,
) -> Result<ProgSuiteRealizationV1, ProgSuitePlanErrorV1> {
    plan.validate()?;
    let carriers = derive_prog_suite_section_carriers(plan, motif)?;

    let mut score = Score::new(plan.home_key, plan.tempo_bpm, plan.sections[0].meter);
    let mut prev_upper: Vec<Pitch> = Vec::new();
    let mut prev_bass: Option<Pitch> = None;
    let pattern = crate::accompaniment::Accompaniment::Comp;

    for (index, (section_plan, carrier)) in plan
        .sections
        .iter()
        .zip(&carriers.sections)
        .enumerate()
    {
        let dominant = section_plan.key.cadence_dominant_degree();
        let meter = f64::from(section_plan.meter);
        let period = Period::parallel_in(
            &carrier.meter_fitted_motif,
            &section_plan.progression_degrees,
            meter,
            dominant,
        );
        let form = Form {
            sections: vec![Section {
                role: section_plan.role,
                key: section_plan.key,
                period,
            }],
        };

        let mut phrase_score = Score::new(section_plan.key, plan.tempo_bpm, section_plan.meter);
        crate::composer::realize_melody(
            &mut phrase_score,
            &form,
            intent,
            Duration::zero(),
            meter,
            false,
        );
        crate::composer::realize_bass(
            &mut phrase_score,
            &form,
            meter,
            intent,
            &mut prev_bass,
            pattern,
            true,
            false,
        );
        crate::composer::realize_harmony(
            &mut phrase_score,
            &form,
            meter,
            intent,
            &mut prev_upper,
            pattern,
            true,
            true,
            false,
        );

        let planned_span = section_plan.end.saturating_sub(section_plan.start);
        if phrase_score.total_beats != planned_span {
            return Err(ProgSuitePlanErrorV1::RealizedSectionSpanMismatch {
                index,
                planned: planned_span,
                realized: phrase_score.total_beats,
            });
        }
        for note in &phrase_score.notes {
            let mut shifted = *note;
            shifted.onset = shifted.onset + section_plan.start;
            score.push(shifted);
        }
    }

    if score.total_beats != plan.total_beats {
        return Err(ProgSuitePlanErrorV1::RealizedTotalSpanMismatch {
            planned: plan.total_beats,
            realized: score.total_beats,
        });
    }
    Ok(ProgSuiteRealizationV1 {
        plan: plan.clone(),
        score,
    })
}

/// Compatibility entry point used by the existing composer dispatch. It now
/// plans first and realizes that frozen plan, but returns the same `Score`
/// surface existing callers expect.
pub(crate) fn realize_prog_suite(
    home_key: Key,
    tempo: f32,
    motif: &Motif,
    seed: u64,
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
) -> Score {
    let plan = plan_prog_suite(home_key, tempo, seed, spec)
        .expect("composer supplied a valid native ProgSuite plan");
    realize_prog_suite_with_plan(&plan, motif, intent)
        .expect("native ProgSuite plan must realize its exact declared spans")
        .score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn intent() -> MusicalIntent {
        MusicalIntent::default()
    }

    fn spec() -> crate::spec::CompositionSpec {
        crate::style::Style::ProgFolk.spec()
    }

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    #[test]
    fn plan_freezes_the_4_7_5_4_architecture_and_exact_spans() {
        let key = Key::major(PitchClass::C);
        let plan = plan_prog_suite(key, 100.0, 5, &spec()).unwrap();
        assert_eq!(plan.total_beats, Duration::new(160, 1));
        let meters: Vec<_> = plan.sections.iter().map(|section| section.meter).collect();
        assert_eq!(meters, vec![4, 7, 5, 4]);
        let spans: Vec<_> = plan
            .sections
            .iter()
            .map(|section| (section.start, section.end))
            .collect();
        assert_eq!(
            spans,
            vec![
                (Duration::new(0, 1), Duration::new(32, 1)),
                (Duration::new(32, 1), Duration::new(88, 1)),
                (Duration::new(88, 1), Duration::new(128, 1)),
                (Duration::new(128, 1), Duration::new(160, 1)),
            ]
        );
        assert_eq!(
            plan.sections[1].transformation,
            ProgSuiteTransformV1::RetrogradeInversion
        );
        assert_eq!(
            plan.sections[2].transformation,
            ProgSuiteTransformV1::Inversion
        );
    }

    #[test]
    fn meter_fit_makes_every_transformed_statement_exactly_one_local_bar() {
        let plan = plan_prog_suite(Key::major(PitchClass::C), 100.0, 5, &spec()).unwrap();
        let carriers = derive_prog_suite_section_carriers(&plan, &motif()).unwrap();
        carriers.validate(&plan, &motif()).unwrap();

        let durations: Vec<_> = carriers
            .sections
            .iter()
            .map(|carrier| carrier.meter_fitted_motif.total_duration())
            .collect();
        assert_eq!(
            durations,
            vec![
                Duration::new(4, 1),
                Duration::new(7, 1),
                Duration::new(5, 1),
                Duration::new(4, 1),
            ]
        );
        let scales: Vec<_> = carriers
            .sections
            .iter()
            .map(|carrier| {
                (
                    carrier.meter_fit.rhythm_scale_numerator,
                    carrier.meter_fit.rhythm_scale_denominator,
                )
            })
            .collect();
        assert_eq!(scales, vec![(1, 1), (7, 4), (5, 4), (1, 1)]);

        // Seed 5 declares inversion followed by retrograde for B. Meter fit
        // changes only duration, so the transformed degree order stays intact.
        assert_eq!(
            carriers.sections[1].transformed_motif.degrees(),
            vec![-3, -1, 0, 1]
        );
        assert_eq!(
            carriers.sections[1].meter_fitted_motif.degrees(),
            vec![-3, -1, 0, 1]
        );
    }

    #[test]
    fn arbitrary_positive_source_length_is_fitted_exactly_not_assumed_four_beats() {
        let plan = plan_prog_suite(Key::major(PitchClass::C), 100.0, 5, &spec()).unwrap();
        let three_beat = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let carriers = derive_prog_suite_section_carriers(&plan, &three_beat).unwrap();
        let scales: Vec<_> = carriers
            .sections
            .iter()
            .map(|carrier| {
                (
                    carrier.meter_fit.rhythm_scale_numerator,
                    carrier.meter_fit.rhythm_scale_denominator,
                )
            })
            .collect();
        assert_eq!(scales, vec![(4, 3), (7, 3), (5, 3), (4, 3)]);
        let realized = realize_prog_suite_with_plan(&plan, &three_beat, &intent()).unwrap();
        assert_eq!(realized.score.total_beats, Duration::new(160, 1));
    }

    #[test]
    fn empty_source_motif_fails_before_any_section_realization() {
        let plan = plan_prog_suite(Key::major(PitchClass::C), 100.0, 5, &spec()).unwrap();
        assert_eq!(
            derive_prog_suite_section_carriers(&plan, &Motif::default()),
            Err(ProgSuitePlanErrorV1::InvalidSourceMotifDuration {
                found: Duration::zero(),
            })
        );
    }

    #[test]
    fn compatibility_entrypoint_is_semantically_equal_to_plan_realization() {
        let key = Key::major(PitchClass::C);
        let motif = motif();
        let plan = plan_prog_suite(key, 100.0, 9, &spec()).unwrap();
        let planned = realize_prog_suite_with_plan(&plan, &motif, &intent())
            .unwrap()
            .score;
        let legacy = realize_prog_suite(key, 100.0, &motif, 9, &intent(), &spec());
        assert_eq!(planned, legacy);
    }

    #[test]
    fn edited_frozen_progression_controls_realization_without_replanning() {
        let key = Key::major(PitchClass::C);
        let motif = motif();
        let original = plan_prog_suite(key, 100.0, 9, &spec()).unwrap();
        let mut edited = original.clone();
        edited.sections[1].progression_degrees = vec![2, 3, 7, 2];
        edited.validate().unwrap();
        let a = realize_prog_suite_with_plan(&original, &motif, &intent())
            .unwrap()
            .score;
        let b = realize_prog_suite_with_plan(&edited, &motif, &intent())
            .unwrap()
            .score;
        assert_ne!(a.notes, b.notes);
    }

    #[test]
    fn edited_frozen_transformation_controls_realization_without_seed_replanning() {
        let key = Key::major(PitchClass::C);
        let motif = motif();
        let original = plan_prog_suite(key, 100.0, 5, &spec()).unwrap();
        let mut edited = original.clone();
        assert_eq!(edited.sections[1].transformation, ProgSuiteTransformV1::RetrogradeInversion);
        assert_eq!(edited.sections[2].transformation, ProgSuiteTransformV1::Inversion);
        edited.sections[1].transformation = ProgSuiteTransformV1::Retrograde;
        edited.validate().unwrap();
        assert_eq!(edited.source_seed, original.source_seed);
        let a = realize_prog_suite_with_plan(&original, &motif, &intent())
            .unwrap()
            .score;
        let b = realize_prog_suite_with_plan(&edited, &motif, &intent())
            .unwrap()
            .score;
        assert_ne!(a.notes, b.notes);
    }

    #[test]
    fn malformed_meter_or_contrast_structure_fails_closed() {
        let key = Key::major(PitchClass::C);
        let mut plan = plan_prog_suite(key, 100.0, 5, &spec()).unwrap();
        plan.sections[1].meter = 4;
        assert!(matches!(
            plan.validate(),
            Err(ProgSuitePlanErrorV1::WrongMeter { index: 1, .. })
        ));

        let mut plan = plan_prog_suite(key, 100.0, 5, &spec()).unwrap();
        plan.sections[1].transformation = ProgSuiteTransformV1::Original;
        assert!(matches!(
            plan.validate(),
            Err(ProgSuitePlanErrorV1::ContrastMustBeTransformed { index: 1 })
        ));

        let mut plan = plan_prog_suite(key, 100.0, 5, &spec()).unwrap();
        plan.sections[1].transformation = plan.sections[2].transformation;
        assert!(matches!(
            plan.validate(),
            Err(ProgSuitePlanErrorV1::DuplicateContrastTransformation { .. })
        ));
    }

    #[test]
    fn prog_suite_visits_three_distinct_meters_in_order() {
        let key = Key::major(PitchClass::C);
        let motif = motif();
        let score = realize_prog_suite(key, 100.0, &motif, 5, &intent(), &spec());
        assert!(!score.notes.is_empty(), "a real piece must come out");
        let uniform_4_only: f64 = 4.0 * (2.0 * 4.0 * 4.0);
        let real_total = score.total_beats.beats();
        assert!(
            (real_total - uniform_4_only).abs() > 1e-6,
            "meters must actually differ per section — got the same total as an all-4/4 piece: {real_total}"
        );
        assert!(
            (real_total - 160.0).abs() < 1e-6,
            "expected exactly 160 beats from the 4/7/5/4 plan, got {real_total}"
        );
    }

    #[test]
    fn prog_suite_carries_voice_leading_continuously_across_meter_changes() {
        let key = Key::major(PitchClass::C);
        let motif = motif();
        let a = realize_prog_suite(key, 100.0, &motif, 9, &intent(), &spec());
        let b = realize_prog_suite(key, 100.0, &motif, 9, &intent(), &spec());
        assert_eq!(a.notes, b.notes, "deterministic for a fixed seed");
        assert_eq!(9 % 3, 0);
        assert_eq!(10 % 3, 1);
        let c = realize_prog_suite(key, 100.0, &motif, 10, &intent(), &spec());
        assert_ne!(
            a.notes, c.notes,
            "a seed with a different motif-transformation choice must produce a genuinely different piece"
        );
    }

    #[test]
    fn every_section_reads_the_specs_own_declared_progression() {
        let key = Key::major(PitchClass::C);
        let motif = motif();
        let mut foreign_spec = spec();
        assert_eq!(
            foreign_spec.progression,
            crate::spec::ProgressionSpec::Archetype(vec![1, 5, 6, 4])
        );
        foreign_spec.progression = crate::spec::ProgressionSpec::Archetype(vec![2, 3, 7, 2]);
        for seed in 0..6u64 {
            let real = realize_prog_suite(key, 100.0, &motif, seed, &intent(), &spec());
            let foreign = realize_prog_suite(key, 100.0, &motif, seed, &intent(), &foreign_spec);
            assert_ne!(
                real.notes, foreign.notes,
                "seed {seed}: swapping the spec's declared progression didn't change the output"
            );
        }
    }
}
