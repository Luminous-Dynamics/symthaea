// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-axis work context for ordered thematic development programs.
//!
//! [`crate::development_program::DevelopmentProgramV1`] owns ordered thematic
//! operations and their exact work spans. [`crate::work_tonal_trajectory`] and
//! [`crate::work_metric_architecture`] independently own long-range tonal and
//! metric declarations. This module binds those three planning axes without
//! merging their authority or mutating any source contract.
//!
//! A stage context means only that the listed tonal/metric regions form the
//! canonical contiguous coverage of that stage span. It is not evidence that a
//! completed score realizes the declared key, meter, pulse, or transformation.

use crate::development_program::{DevelopmentProgramV1, DEVELOPMENT_PROGRAM_VERSION};
use crate::rhythm::Duration;
use crate::work_metric_architecture::{
    WorkMetricArchitectureErrorV1, WorkMetricArchitectureV1,
};
use crate::work_plan::{HierarchicalWorkPlanV1, WorkPlanErrorV1};
use crate::work_tonal_trajectory::{WorkTonalTrajectoryErrorV1, WorkTonalTrajectoryV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const DEVELOPMENT_CONTEXT_VERSION: &str = "melothaea-development-context-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevelopmentStageContextV1 {
    pub stage_id: String,
    /// Canonical chronological tonal regions whose union covers the complete
    /// development-stage span. Boundary regions may begin before or end after
    /// the stage; no gap inside the stage is allowed.
    pub tonal_region_ids: Vec<String>,
    /// Canonical chronological metric regions whose union covers the complete
    /// development-stage span. Boundary regions may extend beyond the stage.
    pub metric_region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkDevelopmentContextV1 {
    pub version: String,
    pub development_program_id: String,
    pub tonal_trajectory_id: String,
    pub metric_architecture_id: String,
    /// Exactly one canonical context record per development stage.
    pub stages: BTreeMap<String, DevelopmentStageContextV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkDevelopmentContextErrorV1 {
    WrongVersion { found: String },
    InvalidWorkPlan(WorkPlanErrorV1),
    WrongDevelopmentProgramVersion { found: String },
    EmptyDevelopmentProgramId,
    EmptyDevelopmentStages,
    EmptyStageId { stage_index: usize },
    DuplicateStageId { stage_id: String },
    MissingStageWorkNode { stage_id: String, work_node_id: String },
    InvalidStageSpan { stage_id: String },
    StageOutsideWorkNode { stage_id: String, work_node_id: String },
    InvalidTonalTrajectory(WorkTonalTrajectoryErrorV1),
    InvalidMetricArchitecture(WorkMetricArchitectureErrorV1),
    EmptyTonalTrajectoryId,
    EmptyMetricArchitectureId,
    DevelopmentProgramIdentityMismatch { expected: String, found: String },
    TonalTrajectoryIdentityMismatch { expected: String, found: String },
    MetricArchitectureIdentityMismatch { expected: String, found: String },
    StageCountMismatch { expected: usize, found: usize },
    MissingStageContext { stage_id: String },
    StageKeyMismatch { key: String, stage_id: String },
    EmptyTonalCoverage { stage_id: String },
    EmptyMetricCoverage { stage_id: String },
    TonalCoverageGap { stage_id: String },
    MetricCoverageGap { stage_id: String },
    TonalCoverageOrderMismatch { stage_id: String },
    MetricCoverageOrderMismatch { stage_id: String },
    CanonicalContextMismatch,
}

pub fn derive_work_development_context(
    program: &DevelopmentProgramV1,
    work_plan: &HierarchicalWorkPlanV1,
    tonal: &WorkTonalTrajectoryV1,
    metric: &WorkMetricArchitectureV1,
) -> Result<WorkDevelopmentContextV1, WorkDevelopmentContextErrorV1> {
    work_plan
        .validate()
        .map_err(WorkDevelopmentContextErrorV1::InvalidWorkPlan)?;
    validate_program_shape(program, work_plan)?;
    tonal
        .validate(work_plan)
        .map_err(WorkDevelopmentContextErrorV1::InvalidTonalTrajectory)?;
    metric
        .validate(work_plan)
        .map_err(WorkDevelopmentContextErrorV1::InvalidMetricArchitecture)?;
    if tonal.trajectory_id.trim().is_empty() {
        return Err(WorkDevelopmentContextErrorV1::EmptyTonalTrajectoryId);
    }
    if metric.architecture_id.trim().is_empty() {
        return Err(WorkDevelopmentContextErrorV1::EmptyMetricArchitectureId);
    }

    let mut stages = BTreeMap::new();
    for stage in &program.stages {
        let tonal_region_ids = covering_regions(
            stage.start,
            stage.end,
            &tonal.region_order,
            |id| {
                let region = &tonal.regions[id];
                (region.start, region.end)
            },
        )
        .ok_or_else(|| WorkDevelopmentContextErrorV1::TonalCoverageGap {
            stage_id: stage.stage_id.clone(),
        })?;
        let metric_region_ids = covering_regions(
            stage.start,
            stage.end,
            &metric.region_order,
            |id| {
                let region = &metric.regions[id];
                (region.start, region.end)
            },
        )
        .ok_or_else(|| WorkDevelopmentContextErrorV1::MetricCoverageGap {
            stage_id: stage.stage_id.clone(),
        })?;

        stages.insert(
            stage.stage_id.clone(),
            DevelopmentStageContextV1 {
                stage_id: stage.stage_id.clone(),
                tonal_region_ids,
                metric_region_ids,
            },
        );
    }

    Ok(WorkDevelopmentContextV1 {
        version: DEVELOPMENT_CONTEXT_VERSION.into(),
        development_program_id: program.program_id.clone(),
        tonal_trajectory_id: tonal.trajectory_id.clone(),
        metric_architecture_id: metric.architecture_id.clone(),
        stages,
    })
}

impl WorkDevelopmentContextV1 {
    /// Validate against the retained planning axes by canonical rederivation.
    /// The full thematic/obligation theorem of `DevelopmentProgramV1` remains
    /// the responsibility of the source adapter that owns those authorities;
    /// this layer independently rechecks every structural fact it consumes.
    pub fn validate(
        &self,
        program: &DevelopmentProgramV1,
        work_plan: &HierarchicalWorkPlanV1,
        tonal: &WorkTonalTrajectoryV1,
        metric: &WorkMetricArchitectureV1,
    ) -> Result<(), WorkDevelopmentContextErrorV1> {
        if self.version != DEVELOPMENT_CONTEXT_VERSION {
            return Err(WorkDevelopmentContextErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        validate_program_shape(program, work_plan)?;
        if self.development_program_id != program.program_id {
            return Err(
                WorkDevelopmentContextErrorV1::DevelopmentProgramIdentityMismatch {
                    expected: program.program_id.clone(),
                    found: self.development_program_id.clone(),
                },
            );
        }
        if self.tonal_trajectory_id != tonal.trajectory_id {
            return Err(WorkDevelopmentContextErrorV1::TonalTrajectoryIdentityMismatch {
                expected: tonal.trajectory_id.clone(),
                found: self.tonal_trajectory_id.clone(),
            });
        }
        if self.metric_architecture_id != metric.architecture_id {
            return Err(
                WorkDevelopmentContextErrorV1::MetricArchitectureIdentityMismatch {
                    expected: metric.architecture_id.clone(),
                    found: self.metric_architecture_id.clone(),
                },
            );
        }
        if self.stages.len() != program.stages.len() {
            return Err(WorkDevelopmentContextErrorV1::StageCountMismatch {
                expected: program.stages.len(),
                found: self.stages.len(),
            });
        }
        for stage in &program.stages {
            let context = self.stages.get(&stage.stage_id).ok_or_else(|| {
                WorkDevelopmentContextErrorV1::MissingStageContext {
                    stage_id: stage.stage_id.clone(),
                }
            })?;
            if context.stage_id != stage.stage_id {
                return Err(WorkDevelopmentContextErrorV1::StageKeyMismatch {
                    key: stage.stage_id.clone(),
                    stage_id: context.stage_id.clone(),
                });
            }
            if context.tonal_region_ids.is_empty() {
                return Err(WorkDevelopmentContextErrorV1::EmptyTonalCoverage {
                    stage_id: stage.stage_id.clone(),
                });
            }
            if context.metric_region_ids.is_empty() {
                return Err(WorkDevelopmentContextErrorV1::EmptyMetricCoverage {
                    stage_id: stage.stage_id.clone(),
                });
            }
            if !is_contiguous_subsequence(&context.tonal_region_ids, &tonal.region_order) {
                return Err(WorkDevelopmentContextErrorV1::TonalCoverageOrderMismatch {
                    stage_id: stage.stage_id.clone(),
                });
            }
            if !is_contiguous_subsequence(&context.metric_region_ids, &metric.region_order) {
                return Err(WorkDevelopmentContextErrorV1::MetricCoverageOrderMismatch {
                    stage_id: stage.stage_id.clone(),
                });
            }
        }

        let canonical = derive_work_development_context(program, work_plan, tonal, metric)?;
        if &canonical != self {
            return Err(WorkDevelopmentContextErrorV1::CanonicalContextMismatch);
        }
        Ok(())
    }
}

fn validate_program_shape(
    program: &DevelopmentProgramV1,
    work_plan: &HierarchicalWorkPlanV1,
) -> Result<(), WorkDevelopmentContextErrorV1> {
    if program.version != DEVELOPMENT_PROGRAM_VERSION {
        return Err(WorkDevelopmentContextErrorV1::WrongDevelopmentProgramVersion {
            found: program.version.clone(),
        });
    }
    if program.program_id.trim().is_empty() {
        return Err(WorkDevelopmentContextErrorV1::EmptyDevelopmentProgramId);
    }
    if program.stages.is_empty() {
        return Err(WorkDevelopmentContextErrorV1::EmptyDevelopmentStages);
    }
    let mut stage_ids = BTreeSet::new();
    for (stage_index, stage) in program.stages.iter().enumerate() {
        if stage.stage_id.trim().is_empty() {
            return Err(WorkDevelopmentContextErrorV1::EmptyStageId { stage_index });
        }
        if !stage_ids.insert(stage.stage_id.clone()) {
            return Err(WorkDevelopmentContextErrorV1::DuplicateStageId {
                stage_id: stage.stage_id.clone(),
            });
        }
        if compare_duration(stage.start, stage.end) != Ordering::Less {
            return Err(WorkDevelopmentContextErrorV1::InvalidStageSpan {
                stage_id: stage.stage_id.clone(),
            });
        }
        let node = work_plan.nodes.get(&stage.work_node_id).ok_or_else(|| {
            WorkDevelopmentContextErrorV1::MissingStageWorkNode {
                stage_id: stage.stage_id.clone(),
                work_node_id: stage.work_node_id.clone(),
            }
        })?;
        if compare_duration(stage.start, node.start) == Ordering::Less
            || compare_duration(stage.end, node.end) == Ordering::Greater
        {
            return Err(WorkDevelopmentContextErrorV1::StageOutsideWorkNode {
                stage_id: stage.stage_id.clone(),
                work_node_id: stage.work_node_id.clone(),
            });
        }
    }
    Ok(())
}

fn covering_regions<F>(
    stage_start: Duration,
    stage_end: Duration,
    region_order: &[String],
    mut span: F,
) -> Option<Vec<String>>
where
    F: FnMut(&str) -> (Duration, Duration),
{
    if compare_duration(stage_start, stage_end) != Ordering::Less {
        return None;
    }
    let mut selected = Vec::new();
    let mut covered_until = stage_start;
    let mut started = false;

    for region_id in region_order {
        let (region_start, region_end) = span(region_id);
        if compare_duration(region_end, stage_start) != Ordering::Greater {
            continue;
        }
        if compare_duration(region_start, stage_end) != Ordering::Less {
            break;
        }
        if !started {
            if compare_duration(region_start, stage_start) == Ordering::Greater {
                return None;
            }
            started = true;
        } else if compare_duration(region_start, covered_until) == Ordering::Greater {
            return None;
        }
        selected.push(region_id.clone());
        if compare_duration(region_end, covered_until) == Ordering::Greater {
            covered_until = region_end;
        }
        if compare_duration(covered_until, stage_end) != Ordering::Less {
            break;
        }
    }

    (started && compare_duration(covered_until, stage_end) != Ordering::Less).then_some(selected)
}

fn is_contiguous_subsequence(selected: &[String], canonical: &[String]) -> bool {
    if selected.is_empty() || selected.len() > canonical.len() {
        return false;
    }
    canonical
        .windows(selected.len())
        .any(|window| window == selected)
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::development_program::{
        DevelopmentEvidenceRequirementV1, DevelopmentGraphProjectionV1,
        DevelopmentOperationV1, DevelopmentStageV1,
    };
    use crate::harmony::Key;
    use crate::meter::TimeSignature;
    use crate::pitch::PitchClass;
    use crate::temporal_map::TempoV1;
    use crate::thematic_identity::ThematicTransformationClassV1;
    use crate::work_metric_architecture::{
        MetricArchitectureCoverageV1, MetricRegionRoleV1, MetricRegionV1,
        MetricTransitionRelationV1, MetricTransitionV1, WORK_METRIC_ARCHITECTURE_VERSION,
    };
    use crate::work_tonal_trajectory::{
        TonalClosurePolicyV1, TonalRegionRoleV1, TonalRegionV1, TonalRelationV1,
        TonalTrajectoryCoverageV1, TonalTransitionIntentV1, TonalTransitionV1,
        WORK_TONAL_TRAJECTORY_VERSION,
    };

    fn work() -> HierarchicalWorkPlanV1 {
        HierarchicalWorkPlanV1::new("work", Duration::new(24, 1)).unwrap()
    }

    fn program() -> DevelopmentProgramV1 {
        DevelopmentProgramV1 {
            version: DEVELOPMENT_PROGRAM_VERSION.into(),
            program_id: "program".into(),
            source_identity_id: "P".into(),
            stages: vec![DevelopmentStageV1 {
                stage_id: "dev".into(),
                work_node_id: "work".into(),
                start: Duration::new(6, 1),
                end: Duration::new(18, 1),
                input_identity_id: "P".into(),
                output_identity_id: "Q".into(),
                derivation_id: "derive-q".into(),
                obligation_id: "realize-q".into(),
                prerequisite_stage_ids: vec![],
                operations: vec![DevelopmentOperationV1 {
                    operation_id: "op".into(),
                    class: ThematicTransformationClassV1::Inversion,
                }],
                graph_projection: DevelopmentGraphProjectionV1::ExactSingleClass,
                evidence_requirement:
                    DevelopmentEvidenceRequirementV1::SingleOperationRelationMeasured,
            }],
        }
    }

    fn tonal() -> WorkTonalTrajectoryV1 {
        let c = Key::major(PitchClass::C);
        let g = c.dominant();
        WorkTonalTrajectoryV1 {
            version: WORK_TONAL_TRAJECTORY_VERSION.into(),
            trajectory_id: "tonal".into(),
            home_key: c,
            coverage: TonalTrajectoryCoverageV1::CompleteWork,
            closure: TonalClosurePolicyV1::Open,
            regions: [
                ("t1".into(), TonalRegionV1 { work_node_id: "work".into(), start: Duration::new(0,1), end: Duration::new(8,1), key: c, role: TonalRegionRoleV1::Home }),
                ("t2".into(), TonalRegionV1 { work_node_id: "work".into(), start: Duration::new(8,1), end: Duration::new(16,1), key: g, role: TonalRegionRoleV1::Departure }),
                ("t3".into(), TonalRegionV1 { work_node_id: "work".into(), start: Duration::new(16,1), end: Duration::new(24,1), key: c, role: TonalRegionRoleV1::Return }),
            ].into_iter().collect(),
            region_order: vec!["t1".into(), "t2".into(), "t3".into()],
            transitions: vec![
                TonalTransitionV1 { from_region_id: "t1".into(), to_region_id: "t2".into(), intent: TonalTransitionIntentV1::Depart, relation: TonalRelationV1::SourceToDominant },
                TonalTransitionV1 { from_region_id: "t2".into(), to_region_id: "t3".into(), intent: TonalTransitionIntentV1::ReturnHome, relation: TonalRelationV1::DominantToSource },
            ],
        }
    }

    fn metric() -> WorkMetricArchitectureV1 {
        let meter = TimeSignature::new(4,4).unwrap();
        let tempo = TempoV1::integer(120).unwrap();
        WorkMetricArchitectureV1 {
            version: WORK_METRIC_ARCHITECTURE_VERSION.into(),
            architecture_id: "metric".into(),
            coverage: MetricArchitectureCoverageV1::CompleteWork,
            regions: [
                ("m1".into(), MetricRegionV1 { work_node_id: "work".into(), start: Duration::new(0,1), end: Duration::new(12,1), meter: meter.clone(), tempo, role: MetricRegionRoleV1::Stable }),
                ("m2".into(), MetricRegionV1 { work_node_id: "work".into(), start: Duration::new(12,1), end: Duration::new(24,1), meter, tempo, role: MetricRegionRoleV1::Stable }),
            ].into_iter().collect(),
            region_order: vec!["m1".into(), "m2".into()],
            transitions: vec![MetricTransitionV1 { from_region_id: "m1".into(), to_region_id: "m2".into(), relation: MetricTransitionRelationV1::Continuation }],
        }
    }

    #[test]
    fn stage_collects_all_contiguous_tonal_and_metric_regions_it_crosses() {
        let program = program();
        let work = work();
        let tonal = tonal();
        let metric = metric();
        let context = derive_work_development_context(&program, &work, &tonal, &metric).unwrap();
        assert_eq!(context.stages["dev"].tonal_region_ids, vec!["t1", "t2", "t3"]);
        assert_eq!(context.stages["dev"].metric_region_ids, vec!["m1", "m2"]);
        context.validate(&program, &work, &tonal, &metric).unwrap();
    }

    #[test]
    fn duplicate_stage_identity_is_rejected_before_context_map_construction() {
        let mut program = program();
        program.stages.push(program.stages[0].clone());
        assert_eq!(
            derive_work_development_context(&program, &work(), &tonal(), &metric()),
            Err(WorkDevelopmentContextErrorV1::DuplicateStageId {
                stage_id: "dev".into(),
            })
        );
    }

    #[test]
    fn tampered_region_context_fails_canonical_rederivation() {
        let program = program();
        let work = work();
        let tonal = tonal();
        let metric = metric();
        let mut context = derive_work_development_context(&program, &work, &tonal, &metric).unwrap();
        context.stages.get_mut("dev").unwrap().tonal_region_ids = vec!["t2".into()];
        assert_eq!(
            context.validate(&program, &work, &tonal, &metric),
            Err(WorkDevelopmentContextErrorV1::CanonicalContextMismatch)
        );
    }

    #[test]
    fn uncovered_stage_span_fails_instead_of_guessing_context() {
        let mut tonal = tonal();
        tonal.coverage = TonalTrajectoryCoverageV1::DeclaredRegionsOnly;
        tonal.regions.get_mut("t2").unwrap().start = Duration::new(9,1);
        assert_eq!(
            derive_work_development_context(&program(), &work(), &tonal, &metric()),
            Err(WorkDevelopmentContextErrorV1::TonalCoverageGap { stage_id: "dev".into() })
        );
    }
}
