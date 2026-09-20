// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subject-bound native ProgSuite declarations and structural realizations.
//!
//! A `ProgSuitePlanV1` freezes architecture, `ProgSuiteWorkBindingV1` freezes
//! its FORM translation, and `ProgSuiteDevelopmentProgramV1` freezes ordered
//! thematic operations. Until now the actual P motif still arrived as a loose
//! function argument. This module closes that declaration-time provenance gap:
//! the exact independent thematic source material travels with the same frozen
//! work/program declaration.
//!
//! The realization wrapper intentionally grants only structural authority. It
//! proves that the stored score is structurally compatible with the same native
//! plan/FORM timeline while retaining the exact `MusicalIntent` that drove the
//! deterministic realization. It does **not** prove that the bound source motif
//! and intent caused those score bytes; exact causal replay belongs to a later
//! evidence contract.

use crate::development_program::DevelopmentProgramErrorV1;
use crate::motif::Motif;
use crate::prog_suite::{
    ProgSuitePlanErrorV1, ProgSuitePlanV1, ProgSuiteRealizationV1,
    realize_prog_suite_with_plan,
};
use crate::prog_suite_development_program::{
    ProgSuiteDevelopmentProgramErrorV1, ProgSuiteDevelopmentProgramV1,
    derive_prog_suite_development_program,
};
use crate::prog_suite_work_bridge::{
    ProgSuiteWorkBindingV1, ProgSuiteWorkBridgeErrorV1, bind_prog_suite_realization,
    bridge_prog_suite_plan,
};
use crate::thematic_source_material::{
    ThematicSourceMaterialErrorV1, ThematicSourceMaterialPlanV1,
};
use crate::MusicalIntent;
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_SUBJECT_BOUND_PLAN_VERSION: &str =
    "melothaea-prog-suite-subject-bound-plan-v1";
pub const PROG_SUITE_SUBJECT_BOUND_REALIZATION_VERSION: &str =
    "melothaea-prog-suite-subject-bound-realization-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSubjectBoundPlanV1 {
    pub version: String,
    pub work_binding: ProgSuiteWorkBindingV1,
    pub development_program: ProgSuiteDevelopmentProgramV1,
    pub source_materials: ThematicSourceMaterialPlanV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSubjectBoundRealizationV1 {
    pub version: String,
    pub declaration: ProgSuiteSubjectBoundPlanV1,
    /// Exact deterministic realization input retained for later causal replay.
    /// Structural validation does not interpret this field as evidence.
    pub intent: MusicalIntent,
    pub realization: ProgSuiteRealizationV1,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteSubjectBoundErrorV1 {
    WrongPlanVersion { found: String },
    WrongRealizationVersion { found: String },
    WorkBridge(ProgSuiteWorkBridgeErrorV1),
    DevelopmentProgram(ProgSuiteDevelopmentProgramErrorV1),
    DevelopmentProgramValidation(DevelopmentProgramErrorV1),
    SourceMaterial(ThematicSourceMaterialErrorV1),
    NativeRealization(ProgSuitePlanErrorV1),
    CanonicalWorkBindingMismatch,
    CanonicalDevelopmentProgramMismatch,
    MissingProgramSourceMaterial { identity_id: String },
    RealizationPlanMismatch,
    RealizationStructuralBindingMismatch,
}

impl ProgSuiteSubjectBoundPlanV1 {
    /// Fail closed after serialization/mutation by reconstructing every derived
    /// declaration layer from the retained native plan and comparing exactly.
    pub fn validate(&self) -> Result<(), ProgSuiteSubjectBoundErrorV1> {
        if self.version != PROG_SUITE_SUBJECT_BOUND_PLAN_VERSION {
            return Err(ProgSuiteSubjectBoundErrorV1::WrongPlanVersion {
                found: self.version.clone(),
            });
        }

        let canonical_work = bridge_prog_suite_plan(&self.work_binding.native_plan)
            .map_err(ProgSuiteSubjectBoundErrorV1::WorkBridge)?;
        if canonical_work != self.work_binding {
            return Err(ProgSuiteSubjectBoundErrorV1::CanonicalWorkBindingMismatch);
        }

        // Preserve the generic DevelopmentProgram invariant failure before
        // checking ProgSuite's exact canonical projection identity.
        self.development_program
            .program
            .validate(
                &self.work_binding.work_plan,
                &self.work_binding.thematic_graph,
                &self.work_binding.obligation_plan,
            )
            .map_err(ProgSuiteSubjectBoundErrorV1::DevelopmentProgramValidation)?;

        let canonical_program = derive_prog_suite_development_program(&self.work_binding)
            .map_err(ProgSuiteSubjectBoundErrorV1::DevelopmentProgram)?;
        if canonical_program != self.development_program {
            return Err(
                ProgSuiteSubjectBoundErrorV1::CanonicalDevelopmentProgramMismatch,
            );
        }

        self.source_materials
            .validate(
                &self.work_binding.work_plan,
                &self.work_binding.thematic_graph,
            )
            .map_err(ProgSuiteSubjectBoundErrorV1::SourceMaterial)?;

        let source_identity_id = &self.development_program.program.source_identity_id;
        if self.source_materials.source(source_identity_id).is_none() {
            return Err(ProgSuiteSubjectBoundErrorV1::MissingProgramSourceMaterial {
                identity_id: source_identity_id.clone(),
            });
        }
        Ok(())
    }

    pub fn source_motif(&self) -> Result<&Motif, ProgSuiteSubjectBoundErrorV1> {
        let source_identity_id = &self.development_program.program.source_identity_id;
        self.source_materials
            .source(source_identity_id)
            .map(|source| &source.motif)
            .ok_or_else(|| ProgSuiteSubjectBoundErrorV1::MissingProgramSourceMaterial {
                identity_id: source_identity_id.clone(),
            })
    }
}

impl ProgSuiteSubjectBoundRealizationV1 {
    /// Validate declaration identity plus score/work structural compatibility.
    ///
    /// This deliberately does not re-run the composer and therefore does not
    /// upgrade structural compatibility or retained intent into a causal
    /// motif+intent->score claim.
    pub fn validate_structure(&self) -> Result<(), ProgSuiteSubjectBoundErrorV1> {
        if self.version != PROG_SUITE_SUBJECT_BOUND_REALIZATION_VERSION {
            return Err(ProgSuiteSubjectBoundErrorV1::WrongRealizationVersion {
                found: self.version.clone(),
            });
        }
        self.declaration.validate()?;
        if self.realization.plan != self.declaration.work_binding.native_plan {
            return Err(ProgSuiteSubjectBoundErrorV1::RealizationPlanMismatch);
        }
        let structural = bind_prog_suite_realization(&self.realization)
            .map_err(ProgSuiteSubjectBoundErrorV1::WorkBridge)?;
        if structural.declaration != self.declaration.work_binding {
            return Err(
                ProgSuiteSubjectBoundErrorV1::RealizationStructuralBindingMismatch,
            );
        }
        Ok(())
    }
}

/// Bind one already-frozen native ProgSuite plan to exact independent source
/// material. No score is consumed and `source_seed` is never reconsulted.
pub fn bind_prog_suite_subject(
    plan: &ProgSuitePlanV1,
    source_motif: &Motif,
) -> Result<ProgSuiteSubjectBoundPlanV1, ProgSuiteSubjectBoundErrorV1> {
    let work_binding = bridge_prog_suite_plan(plan)
        .map_err(ProgSuiteSubjectBoundErrorV1::WorkBridge)?;
    let development_program = derive_prog_suite_development_program(&work_binding)
        .map_err(ProgSuiteSubjectBoundErrorV1::DevelopmentProgram)?;

    let source_identity_id = development_program.program.source_identity_id.clone();
    let mut source_materials = ThematicSourceMaterialPlanV1::default();
    source_materials
        .insert(source_identity_id, source_motif.clone())
        .map_err(ProgSuiteSubjectBoundErrorV1::SourceMaterial)?;
    source_materials
        .validate(&work_binding.work_plan, &work_binding.thematic_graph)
        .map_err(ProgSuiteSubjectBoundErrorV1::SourceMaterial)?;

    let bound = ProgSuiteSubjectBoundPlanV1 {
        version: PROG_SUITE_SUBJECT_BOUND_PLAN_VERSION.into(),
        work_binding,
        development_program,
        source_materials,
    };
    bound.validate()?;
    Ok(bound)
}

/// Realize exactly the subject-bound declaration.
///
/// This constructor removes the loose motif argument from the realization
/// boundary and retains the exact `MusicalIntent` used. The returned wrapper is
/// still declaration+structure, not evidence that those inputs causally
/// produced the stored score after arbitrary serialization.
pub fn realize_prog_suite_subject_bound(
    declaration: &ProgSuiteSubjectBoundPlanV1,
    intent: &MusicalIntent,
) -> Result<ProgSuiteSubjectBoundRealizationV1, ProgSuiteSubjectBoundErrorV1> {
    declaration.validate()?;
    let source_motif = declaration.source_motif()?;
    let realization = realize_prog_suite_with_plan(
        &declaration.work_binding.native_plan,
        source_motif,
        intent,
    )
    .map_err(ProgSuiteSubjectBoundErrorV1::NativeRealization)?;

    let bound = ProgSuiteSubjectBoundRealizationV1 {
        version: PROG_SUITE_SUBJECT_BOUND_REALIZATION_VERSION.into(),
        declaration: declaration.clone(),
        intent: *intent,
        realization,
    };
    bound.validate_structure()?;
    Ok(bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, MotifNote, PitchClass, ProgSuiteTransformV1, Style,
        plan_prog_suite,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn plan() -> ProgSuitePlanV1 {
        plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap()
    }

    #[test]
    fn one_capsule_binds_work_program_and_exact_primary_material() {
        let bound = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        bound.validate().unwrap();
        assert_eq!(bound.source_motif().unwrap(), &motif());
        assert_eq!(
            bound.development_program.program.source_identity_id,
            "prog-suite:P"
        );
        assert_eq!(bound.development_program.program.stages.len(), 3);
    }

    #[test]
    fn exact_source_material_is_part_of_declaration_identity() {
        let left = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        let alternate = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (4, Duration::quarter()),
            (2, Duration::quarter()),
            (6, Duration::quarter()),
        ]);
        let right = bind_prog_suite_subject(&plan(), &alternate).unwrap();
        assert_ne!(left, right);
        assert_eq!(left.work_binding, right.work_binding);
        assert_eq!(left.development_program, right.development_program);
    }

    #[test]
    fn forged_work_binding_fails_canonical_declaration_validation() {
        let mut bound = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        bound
            .work_binding
            .thematic_graph
            .derivations
            .get_mut("prog-suite:derive-B")
            .unwrap()
            .transformations = vec![crate::ThematicTransformationClassV1::Retrograde];
        assert_eq!(
            bound.validate(),
            Err(ProgSuiteSubjectBoundErrorV1::CanonicalWorkBindingMismatch)
        );
    }

    #[test]
    fn malformed_program_preserves_generic_validation_error() {
        let mut bound = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        bound.development_program.program.stages[0].operations.clear();
        assert!(matches!(
            bound.validate(),
            Err(ProgSuiteSubjectBoundErrorV1::DevelopmentProgramValidation(
                DevelopmentProgramErrorV1::EmptyOperations { .. }
            ))
        ));
    }

    #[test]
    fn valid_but_noncanonical_program_fails_exact_projection_identity() {
        let mut bound = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        bound.development_program.program.program_id = "alternate-valid-id".into();
        assert_eq!(
            bound.validate(),
            Err(
                ProgSuiteSubjectBoundErrorV1::CanonicalDevelopmentProgramMismatch
            )
        );
    }

    #[test]
    fn structural_realization_retains_exact_intent_and_bound_motif() {
        let declaration = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        let intent = MusicalIntent {
            energy: 0.73,
            seed: 9182,
            ..MusicalIntent::default()
        };
        let bound = realize_prog_suite_subject_bound(&declaration, &intent).unwrap();
        bound.validate_structure().unwrap();
        assert_eq!(bound.intent, intent);

        let direct = realize_prog_suite_with_plan(
            &declaration.work_binding.native_plan,
            declaration.source_motif().unwrap(),
            &intent,
        )
        .unwrap();
        assert_eq!(bound.realization, direct);
    }

    #[test]
    fn stale_realization_plan_cannot_reuse_declaration_authority() {
        let declaration = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        let mut bound = realize_prog_suite_subject_bound(
            &declaration,
            &MusicalIntent::default(),
        )
        .unwrap();
        bound.realization.plan.sections[1].transformation = ProgSuiteTransformV1::Retrograde;
        assert_eq!(
            bound.validate_structure(),
            Err(ProgSuiteSubjectBoundErrorV1::RealizationPlanMismatch)
        );
    }

    #[test]
    fn stale_score_span_fails_structural_binding_without_claiming_causality() {
        let declaration = bind_prog_suite_subject(&plan(), &motif()).unwrap();
        let mut bound = realize_prog_suite_subject_bound(
            &declaration,
            &MusicalIntent::default(),
        )
        .unwrap();
        bound.realization.score.total_beats = Duration::new(159, 1);
        assert!(matches!(
            bound.validate_structure(),
            Err(ProgSuiteSubjectBoundErrorV1::WorkBridge(_))
        ));
    }

    #[test]
    fn source_material_validation_still_rejects_rest_only_subject() {
        let rest_only = Motif::new(vec![MotifNote::rest(Duration::quarter())]);
        assert!(matches!(
            bind_prog_suite_subject(&plan(), &rest_only),
            Err(ProgSuiteSubjectBoundErrorV1::SourceMaterial(
                ThematicSourceMaterialErrorV1::RestOnlyMotif { .. }
            ))
        ));
    }
}
