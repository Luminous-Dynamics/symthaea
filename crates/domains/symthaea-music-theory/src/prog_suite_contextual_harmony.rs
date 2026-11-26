// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optional context-aware harmonic arc for native ProgSuite.
//!
//! Legacy [`crate::prog_suite::plan_prog_suite`] intentionally remains
//! unchanged. This adapter consumes the already-canonical long-range ProgSuite
//! context and produces an explicit alternative plan that changes only the four
//! frozen progression-degree vectors. Keys, meters, spans, thematic
//! transformations, tempo, and source provenance remain untouched.
//!
//! The profile is a compositional policy to evaluate, not evidence that the
//! resulting music is historically correct, preferred, or artistically better.

use crate::form::SectionRole;
use crate::harmony::Key;
use crate::prog_suite::{ProgSuitePlanErrorV1, ProgSuitePlanV1};
use crate::prog_suite_development_context::{
    ProgSuiteDevelopmentContextErrorV1, ProgSuiteDevelopmentContextV1,
};
use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_CONTEXTUAL_HARMONY_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteContextualHarmonyProfileV1 {
    /// A deterministic four-section arc with stable bookends, a home-key B
    /// departure, and a locally centered relative-key C development.
    DirectedDepartureReturn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteHarmonicNarrativeRoleV1 {
    EstablishHome,
    DestabilizeHome,
    DevelopRelative,
    ResolveHome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteContextualHarmonyReceiptV1 {
    pub section_index: usize,
    pub section_role: SectionRole,
    pub key: Key,
    pub meter: u8,
    pub start: Duration,
    pub end: Duration,
    pub tonal_region_id: String,
    pub metric_region_id: String,
    pub narrative_role: ProgSuiteHarmonicNarrativeRoleV1,
    pub source_progression_degrees: Vec<i32>,
    pub contextual_progression_degrees: Vec<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteContextualHarmonyNonClaimV1 {
    PolicyDoesNotEstablishHistoricalStyle,
    PolicyDoesNotEstablishListenerPreference,
    PolicyDoesNotEstablishArtisticQuality,
    PolicyDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteContextualHarmonyPlanV1 {
    pub version: String,
    pub profile: ProgSuiteContextualHarmonyProfileV1,
    /// Exact native plan before the optional harmonic rewrite.
    pub source_plan: ProgSuitePlanV1,
    /// Alternative plan with only progression-degree vectors changed.
    pub contextual_plan: ProgSuitePlanV1,
    pub receipts: Vec<ProgSuiteContextualHarmonyReceiptV1>,
    pub nonclaims: Vec<ProgSuiteContextualHarmonyNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteContextualHarmonyErrorV1 {
    SourceContext(ProgSuiteDevelopmentContextErrorV1),
    SourcePlan(ProgSuitePlanErrorV1),
    ContextualPlan(ProgSuitePlanErrorV1),
    WrongVersion { found: String },
    WrongReceiptCount { found: usize },
    RegionOrderCountMismatch,
    ReceiptBindingMismatch { section_index: usize },
    UnexpectedSourceMutation { section_index: usize },
    NonCanonicalNonClaims,
    CanonicalPlanMismatch,
}

pub fn derive_prog_suite_contextual_harmony(
    context: &ProgSuiteDevelopmentContextV1,
    profile: ProgSuiteContextualHarmonyProfileV1,
) -> Result<ProgSuiteContextualHarmonyPlanV1, ProgSuiteContextualHarmonyErrorV1> {
    context
        .validate()
        .map_err(ProgSuiteContextualHarmonyErrorV1::SourceContext)?;
    let source_plan = context.work_architecture.binding.native_plan.clone();
    source_plan
        .validate()
        .map_err(ProgSuiteContextualHarmonyErrorV1::SourcePlan)?;

    let tonal_order = &context.work_architecture.tonal_trajectory.region_order;
    let metric_order = &context.work_architecture.metric_architecture.region_order;
    if tonal_order.len() != source_plan.sections.len()
        || metric_order.len() != source_plan.sections.len()
    {
        return Err(ProgSuiteContextualHarmonyErrorV1::RegionOrderCountMismatch);
    }

    let mut contextual_plan = source_plan.clone();
    let mut receipts = Vec::with_capacity(source_plan.sections.len());
    for (section_index, (source, contextual)) in source_plan
        .sections
        .iter()
        .zip(contextual_plan.sections.iter_mut())
        .enumerate()
    {
        let (narrative_role, degrees) = profile_policy(profile, section_index);
        contextual.progression_degrees = degrees.clone();
        receipts.push(ProgSuiteContextualHarmonyReceiptV1 {
            section_index,
            section_role: source.role,
            key: source.key,
            meter: source.meter,
            start: source.start,
            end: source.end,
            tonal_region_id: tonal_order[section_index].clone(),
            metric_region_id: metric_order[section_index].clone(),
            narrative_role,
            source_progression_degrees: source.progression_degrees.clone(),
            contextual_progression_degrees: degrees,
        });
    }
    contextual_plan
        .validate()
        .map_err(ProgSuiteContextualHarmonyErrorV1::ContextualPlan)?;
    assert_only_progressions_changed(&source_plan, &contextual_plan)?;

    Ok(ProgSuiteContextualHarmonyPlanV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_VERSION.into(),
        profile,
        source_plan,
        contextual_plan,
        receipts,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteContextualHarmonyPlanV1 {
    pub fn validate(
        &self,
        context: &ProgSuiteDevelopmentContextV1,
    ) -> Result<(), ProgSuiteContextualHarmonyErrorV1> {
        if self.version != PROG_SUITE_CONTEXTUAL_HARMONY_VERSION {
            return Err(ProgSuiteContextualHarmonyErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteContextualHarmonyErrorV1::NonCanonicalNonClaims);
        }
        if self.receipts.len() != self.source_plan.sections.len() {
            return Err(ProgSuiteContextualHarmonyErrorV1::WrongReceiptCount {
                found: self.receipts.len(),
            });
        }
        context
            .validate()
            .map_err(ProgSuiteContextualHarmonyErrorV1::SourceContext)?;
        if self.source_plan != context.work_architecture.binding.native_plan {
            return Err(ProgSuiteContextualHarmonyErrorV1::CanonicalPlanMismatch);
        }
        assert_only_progressions_changed(&self.source_plan, &self.contextual_plan)?;
        for (index, receipt) in self.receipts.iter().enumerate() {
            let source = &self.source_plan.sections[index];
            let contextual = &self.contextual_plan.sections[index];
            if receipt.section_index != index
                || receipt.section_role != source.role
                || receipt.key != source.key
                || receipt.meter != source.meter
                || receipt.start != source.start
                || receipt.end != source.end
                || receipt.source_progression_degrees != source.progression_degrees
                || receipt.contextual_progression_degrees != contextual.progression_degrees
                || receipt.tonal_region_id
                    != context.work_architecture.tonal_trajectory.region_order[index]
                || receipt.metric_region_id
                    != context.work_architecture.metric_architecture.region_order[index]
            {
                return Err(ProgSuiteContextualHarmonyErrorV1::ReceiptBindingMismatch {
                    section_index: index,
                });
            }
        }

        let canonical = derive_prog_suite_contextual_harmony(context, self.profile)?;
        if &canonical != self {
            return Err(ProgSuiteContextualHarmonyErrorV1::CanonicalPlanMismatch);
        }
        Ok(())
    }
}

fn profile_policy(
    profile: ProgSuiteContextualHarmonyProfileV1,
    section_index: usize,
) -> (ProgSuiteHarmonicNarrativeRoleV1, Vec<i32>) {
    match profile {
        ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn => match section_index {
            // Stable home statement and explicit home closure share bookends.
            0 => (
                ProgSuiteHarmonicNarrativeRoleV1::EstablishHome,
                vec![1, 4, 5, 1],
            ),
            // Stay in the home key but avoid tonic arrival until the next
            // section boundary; the final dominant leaves forward pressure.
            1 => (
                ProgSuiteHarmonicNarrativeRoleV1::DestabilizeHome,
                vec![6, 4, 2, 5],
            ),
            // Establish and develop the already-declared relative-key region.
            2 => (
                ProgSuiteHarmonicNarrativeRoleV1::DevelopRelative,
                vec![1, 4, 2, 5],
            ),
            3 => (
                ProgSuiteHarmonicNarrativeRoleV1::ResolveHome,
                vec![1, 4, 5, 1],
            ),
            _ => unreachable!("validated ProgSuite has exactly four sections"),
        },
    }
}

fn assert_only_progressions_changed(
    source: &ProgSuitePlanV1,
    contextual: &ProgSuitePlanV1,
) -> Result<(), ProgSuiteContextualHarmonyErrorV1> {
    if source.version != contextual.version
        || source.home_key != contextual.home_key
        || source.tempo_bpm.to_bits() != contextual.tempo_bpm.to_bits()
        || source.source_seed != contextual.source_seed
        || source.total_beats != contextual.total_beats
        || source.sections.len() != contextual.sections.len()
    {
        return Err(ProgSuiteContextualHarmonyErrorV1::UnexpectedSourceMutation {
            section_index: usize::MAX,
        });
    }
    for (index, (left, right)) in source
        .sections
        .iter()
        .zip(&contextual.sections)
        .enumerate()
    {
        if left.role != right.role
            || left.key != right.key
            || left.meter != right.meter
            || left.transformation != right.transformation
            || left.start != right.start
            || left.end != right.end
        {
            return Err(ProgSuiteContextualHarmonyErrorV1::UnexpectedSourceMutation {
                section_index: index,
            });
        }
    }
    Ok(())
}

fn required_nonclaims() -> Vec<ProgSuiteContextualHarmonyNonClaimV1> {
    vec![
        ProgSuiteContextualHarmonyNonClaimV1::PolicyDoesNotEstablishHistoricalStyle,
        ProgSuiteContextualHarmonyNonClaimV1::PolicyDoesNotEstablishListenerPreference,
        ProgSuiteContextualHarmonyNonClaimV1::PolicyDoesNotEstablishArtisticQuality,
        ProgSuiteContextualHarmonyNonClaimV1::PolicyDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, PitchClass, Style,
        derive_prog_suite_development_context, plan_prog_suite,
        realize_prog_suite_with_plan,
    };

    fn context(seed: u64) -> ProgSuiteDevelopmentContextV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            seed,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        derive_prog_suite_development_context(&plan).unwrap()
    }

    #[test]
    fn contextual_arc_changes_only_progressions() {
        let context = context(5);
        let artifact = derive_prog_suite_contextual_harmony(
            &context,
            ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn,
        )
        .unwrap();
        artifact.validate(&context).unwrap();
        assert_only_progressions_changed(&artifact.source_plan, &artifact.contextual_plan).unwrap();
        assert_eq!(artifact.contextual_plan.sections[0].progression_degrees, vec![1, 4, 5, 1]);
        assert_eq!(artifact.contextual_plan.sections[1].progression_degrees, vec![6, 4, 2, 5]);
        assert_eq!(artifact.contextual_plan.sections[2].progression_degrees, vec![1, 4, 2, 5]);
        assert_eq!(artifact.contextual_plan.sections[3].progression_degrees, vec![1, 4, 5, 1]);
    }

    #[test]
    fn receipts_bind_policy_to_exact_tonal_and_metric_regions() {
        let context = context(5);
        let artifact = derive_prog_suite_contextual_harmony(
            &context,
            ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn,
        )
        .unwrap();
        assert_eq!(artifact.receipts[1].tonal_region_id, "prog-suite:B");
        assert_eq!(artifact.receipts[1].metric_region_id, "prog-suite:B");
        assert_eq!(artifact.receipts[2].tonal_region_id, "prog-suite:C");
        assert_eq!(artifact.receipts[2].metric_region_id, "prog-suite:C");
    }

    #[test]
    fn contextual_plan_remains_realizable_with_exact_total_span() {
        let context = context(5);
        let artifact = derive_prog_suite_contextual_harmony(
            &context,
            ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn,
        )
        .unwrap();
        let motif = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let intent = MusicalIntent::default();
        let legacy = realize_prog_suite_with_plan(&artifact.source_plan, &motif, &intent)
            .unwrap();
        let contextual = realize_prog_suite_with_plan(&artifact.contextual_plan, &motif, &intent)
            .unwrap();
        assert_eq!(contextual.score.total_beats, artifact.contextual_plan.total_beats);
        assert_ne!(legacy.score.notes, contextual.score.notes);
    }

    #[test]
    fn serialized_progression_tampering_fails_canonical_validation() {
        let context = context(5);
        let mut artifact = derive_prog_suite_contextual_harmony(
            &context,
            ProgSuiteContextualHarmonyProfileV1::DirectedDepartureReturn,
        )
        .unwrap();
        artifact.contextual_plan.sections[1].progression_degrees = vec![1, 1, 1, 1];
        assert!(artifact.validate(&context).is_err());
    }
}
