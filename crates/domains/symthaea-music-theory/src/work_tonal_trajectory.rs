// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Work-scale tonal trajectory declaration.
//!
//! Existing form engines already choose local keys, but a collection of local
//! section keys is not yet a work-scale tonal story. This contract freezes an
//! ordered sequence of exact tonal regions over a [`crate::work_plan::HierarchicalWorkPlanV1`],
//! including narrative role, directed key relation, adjacency, coverage policy,
//! and optional return-to-home closure.
//!
//! V1 is declaration authority only. A valid trajectory does not establish that
//! the completed score acoustically, symbolically, or perceptually realizes the
//! declared tonal centers. Evidence belongs in a separate completed-score layer.

use crate::harmony::Key;
use crate::rhythm::Duration;
use crate::work_plan::{HierarchicalWorkPlanV1, WorkPlanErrorV1};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

pub const WORK_TONAL_TRAJECTORY_VERSION: &str = "melothaea-work-tonal-trajectory-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TonalTrajectoryCoverageV1 {
    /// Regions must cover the complete work root with no gaps or overlap.
    CompleteWork,
    /// Only declared spans carry tonal authority; gaps are intentionally silent.
    DeclaredRegionsOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TonalClosurePolicyV1 {
    /// No claim that the final tonal region returns to the declared home key.
    Open,
    /// The final region must return to the exact declared home key.
    ReturnToHome,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TonalRegionRoleV1 {
    Home,
    Departure,
    Contrast,
    Development,
    Preparation,
    Return,
    Resolution,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TonalRegionV1 {
    /// Exact formal context containing the complete tonal span.
    pub work_node_id: String,
    /// Exact half-open span `[start, end)` in quarter-note beats.
    pub start: Duration,
    pub end: Duration,
    pub key: Key,
    pub role: TonalRegionRoleV1,
}

/// Directed relation from one exact region key to the next.
///
/// These labels are mechanically rederived from the two keys. They describe
/// planning identity only; they do not prove a modulation mechanism or pivot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TonalRelationV1 {
    SameKey,
    SourceToDominant,
    DominantToSource,
    SourceToRelative,
    SourceToParallel,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TonalTransitionIntentV1 {
    Hold,
    Depart,
    Contrast,
    Develop,
    PrepareReturn,
    ReturnHome,
    ResolveHome,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TonalTransitionV1 {
    pub from_region_id: String,
    pub to_region_id: String,
    /// Narrative planning intent. This is not evidence of listener perception.
    pub intent: TonalTransitionIntentV1,
    /// Exact directed relation rederived from the two region keys.
    pub relation: TonalRelationV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkTonalTrajectoryV1 {
    pub version: String,
    pub trajectory_id: String,
    pub home_key: Key,
    pub coverage: TonalTrajectoryCoverageV1,
    pub closure: TonalClosurePolicyV1,
    /// Stable region identities. Chronology is carried by `region_order`.
    pub regions: BTreeMap<String, TonalRegionV1>,
    /// Every region ID exactly once, in strict chronological order.
    pub region_order: Vec<String>,
    /// Exactly one transition for each adjacent pair in `region_order`.
    pub transitions: Vec<TonalTransitionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkTonalTrajectoryErrorV1 {
    WrongVersion { found: String },
    InvalidWorkPlan(WorkPlanErrorV1),
    EmptyTrajectoryId,
    EmptyRegions,
    RegionOrderCountMismatch { regions: usize, ordered: usize },
    EmptyRegionId { order_index: usize },
    UnknownOrderedRegion { region_id: String },
    DuplicateOrderedRegion { region_id: String },
    UnorderedRegion { region_id: String },
    EmptyWorkNode { region_id: String },
    MissingWorkNode { region_id: String, work_node_id: String },
    InvalidRegionSpan { region_id: String },
    RegionOutsideWorkNode { region_id: String, work_node_id: String },
    EmptyOtherRole { region_id: String },
    RegionOverlap { left_region_id: String, right_region_id: String },
    CompleteCoverageMustStartAtRoot,
    CompleteCoverageGap { left_region_id: String, right_region_id: String },
    CompleteCoverageMustEndAtRoot,
    FirstRegionMustBeHome { region_id: String },
    ReturnClosureMustEndAtHome { region_id: String },
    TransitionCountMismatch { expected: usize, found: usize },
    TransitionAdjacencyMismatch { transition_index: usize },
    TonalRelationMismatch {
        transition_index: usize,
        expected: TonalRelationV1,
        found: TonalRelationV1,
    },
    HoldRequiresSameKey { transition_index: usize },
    ChangeIntentRequiresDifferentKey { transition_index: usize },
    HomeIntentRequiresHomeTarget { transition_index: usize },
    EmptyOtherIntent { transition_index: usize },
    QueryBeforeWork,
}

impl WorkTonalTrajectoryV1 {
    pub fn validate(
        &self,
        work_plan: &HierarchicalWorkPlanV1,
    ) -> Result<(), WorkTonalTrajectoryErrorV1> {
        if self.version != WORK_TONAL_TRAJECTORY_VERSION {
            return Err(WorkTonalTrajectoryErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        work_plan
            .validate()
            .map_err(WorkTonalTrajectoryErrorV1::InvalidWorkPlan)?;
        if self.trajectory_id.trim().is_empty() {
            return Err(WorkTonalTrajectoryErrorV1::EmptyTrajectoryId);
        }
        if self.regions.is_empty() {
            return Err(WorkTonalTrajectoryErrorV1::EmptyRegions);
        }
        if self.region_order.len() != self.regions.len() {
            return Err(WorkTonalTrajectoryErrorV1::RegionOrderCountMismatch {
                regions: self.regions.len(),
                ordered: self.region_order.len(),
            });
        }

        let root = work_plan
            .nodes
            .get(&work_plan.root_id)
            .expect("validated work plan retains root");
        let mut seen = BTreeSet::new();
        let mut previous: Option<(&str, &TonalRegionV1)> = None;

        for (order_index, region_id) in self.region_order.iter().enumerate() {
            if region_id.trim().is_empty() {
                return Err(WorkTonalTrajectoryErrorV1::EmptyRegionId { order_index });
            }
            let region = self.regions.get(region_id).ok_or_else(|| {
                WorkTonalTrajectoryErrorV1::UnknownOrderedRegion {
                    region_id: region_id.clone(),
                }
            })?;
            if !seen.insert(region_id.clone()) {
                return Err(WorkTonalTrajectoryErrorV1::DuplicateOrderedRegion {
                    region_id: region_id.clone(),
                });
            }
            if region.work_node_id.trim().is_empty() {
                return Err(WorkTonalTrajectoryErrorV1::EmptyWorkNode {
                    region_id: region_id.clone(),
                });
            }
            let node = work_plan.nodes.get(&region.work_node_id).ok_or_else(|| {
                WorkTonalTrajectoryErrorV1::MissingWorkNode {
                    region_id: region_id.clone(),
                    work_node_id: region.work_node_id.clone(),
                }
            })?;
            if compare_duration(region.start, region.end) != Ordering::Less {
                return Err(WorkTonalTrajectoryErrorV1::InvalidRegionSpan {
                    region_id: region_id.clone(),
                });
            }
            if compare_duration(region.start, node.start) == Ordering::Less
                || compare_duration(region.end, node.end) == Ordering::Greater
            {
                return Err(WorkTonalTrajectoryErrorV1::RegionOutsideWorkNode {
                    region_id: region_id.clone(),
                    work_node_id: region.work_node_id.clone(),
                });
            }
            if matches!(&region.role, TonalRegionRoleV1::Other(label) if label.trim().is_empty()) {
                return Err(WorkTonalTrajectoryErrorV1::EmptyOtherRole {
                    region_id: region_id.clone(),
                });
            }

            if let Some((previous_id, previous_region)) = previous {
                match compare_duration(previous_region.end, region.start) {
                    Ordering::Greater => {
                        return Err(WorkTonalTrajectoryErrorV1::RegionOverlap {
                            left_region_id: previous_id.into(),
                            right_region_id: region_id.clone(),
                        });
                    }
                    Ordering::Less
                        if self.coverage == TonalTrajectoryCoverageV1::CompleteWork =>
                    {
                        return Err(WorkTonalTrajectoryErrorV1::CompleteCoverageGap {
                            left_region_id: previous_id.into(),
                            right_region_id: region_id.clone(),
                        });
                    }
                    _ => {}
                }
            }
            previous = Some((region_id.as_str(), region));
        }

        for region_id in self.regions.keys() {
            if !seen.contains(region_id) {
                return Err(WorkTonalTrajectoryErrorV1::UnorderedRegion {
                    region_id: region_id.clone(),
                });
            }
        }

        let first_id = &self.region_order[0];
        let first = &self.regions[first_id];
        let last_id = self.region_order.last().expect("nonempty order");
        let last = &self.regions[last_id];
        if first.key != self.home_key {
            return Err(WorkTonalTrajectoryErrorV1::FirstRegionMustBeHome {
                region_id: first_id.clone(),
            });
        }
        if self.coverage == TonalTrajectoryCoverageV1::CompleteWork {
            if first.start != root.start {
                return Err(WorkTonalTrajectoryErrorV1::CompleteCoverageMustStartAtRoot);
            }
            if last.end != root.end {
                return Err(WorkTonalTrajectoryErrorV1::CompleteCoverageMustEndAtRoot);
            }
        }
        if self.closure == TonalClosurePolicyV1::ReturnToHome && last.key != self.home_key {
            return Err(WorkTonalTrajectoryErrorV1::ReturnClosureMustEndAtHome {
                region_id: last_id.clone(),
            });
        }

        let expected_transitions = self.region_order.len().saturating_sub(1);
        if self.transitions.len() != expected_transitions {
            return Err(WorkTonalTrajectoryErrorV1::TransitionCountMismatch {
                expected: expected_transitions,
                found: self.transitions.len(),
            });
        }
        for (index, transition) in self.transitions.iter().enumerate() {
            let expected_from = &self.region_order[index];
            let expected_to = &self.region_order[index + 1];
            if &transition.from_region_id != expected_from || &transition.to_region_id != expected_to
            {
                return Err(WorkTonalTrajectoryErrorV1::TransitionAdjacencyMismatch {
                    transition_index: index,
                });
            }
            let from = &self.regions[expected_from];
            let to = &self.regions[expected_to];
            let expected_relation = classify_tonal_relation(from.key, to.key);
            if transition.relation != expected_relation {
                return Err(WorkTonalTrajectoryErrorV1::TonalRelationMismatch {
                    transition_index: index,
                    expected: expected_relation,
                    found: transition.relation,
                });
            }
            validate_intent(index, &transition.intent, from.key, to.key, self.home_key)?;
        }
        Ok(())
    }

    /// Return the declared tonal region containing `at`, if V1 assigns one.
    pub fn region_at<'a>(
        &'a self,
        work_plan: &HierarchicalWorkPlanV1,
        at: Duration,
    ) -> Result<Option<(&'a str, &'a TonalRegionV1)>, WorkTonalTrajectoryErrorV1> {
        self.validate(work_plan)?;
        if at.num() < 0 {
            return Err(WorkTonalTrajectoryErrorV1::QueryBeforeWork);
        }
        for region_id in &self.region_order {
            let region = &self.regions[region_id];
            if compare_duration(region.start, at) != Ordering::Greater
                && compare_duration(at, region.end) == Ordering::Less
            {
                return Ok(Some((region_id.as_str(), region)));
            }
        }
        Ok(None)
    }
}

pub fn classify_tonal_relation(from: Key, to: Key) -> TonalRelationV1 {
    if from == to {
        TonalRelationV1::SameKey
    } else if from.dominant() == to {
        TonalRelationV1::SourceToDominant
    } else if to.dominant() == from {
        TonalRelationV1::DominantToSource
    } else if from.relative() == to {
        TonalRelationV1::SourceToRelative
    } else if from.parallel() == to {
        TonalRelationV1::SourceToParallel
    } else {
        TonalRelationV1::Other
    }
}

fn validate_intent(
    transition_index: usize,
    intent: &TonalTransitionIntentV1,
    from: Key,
    to: Key,
    home: Key,
) -> Result<(), WorkTonalTrajectoryErrorV1> {
    match intent {
        TonalTransitionIntentV1::Hold if from != to => {
            Err(WorkTonalTrajectoryErrorV1::HoldRequiresSameKey { transition_index })
        }
        TonalTransitionIntentV1::Depart | TonalTransitionIntentV1::Contrast if from == to => {
            Err(WorkTonalTrajectoryErrorV1::ChangeIntentRequiresDifferentKey {
                transition_index,
            })
        }
        TonalTransitionIntentV1::ReturnHome | TonalTransitionIntentV1::ResolveHome
            if to != home =>
        {
            Err(WorkTonalTrajectoryErrorV1::HomeIntentRequiresHomeTarget {
                transition_index,
            })
        }
        TonalTransitionIntentV1::Other(label) if label.trim().is_empty() => {
            Err(WorkTonalTrajectoryErrorV1::EmptyOtherIntent { transition_index })
        }
        _ => Ok(()),
    }
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn work() -> HierarchicalWorkPlanV1 {
        HierarchicalWorkPlanV1::new("work", Duration::new(32, 1)).unwrap()
    }

    fn region(start: i64, end: i64, key: Key, role: TonalRegionRoleV1) -> TonalRegionV1 {
        TonalRegionV1 {
            work_node_id: "work".into(),
            start: Duration::new(start, 1),
            end: Duration::new(end, 1),
            key,
            role,
        }
    }

    fn transition(
        from: &str,
        to: &str,
        intent: TonalTransitionIntentV1,
        from_key: Key,
        to_key: Key,
    ) -> TonalTransitionV1 {
        TonalTransitionV1 {
            from_region_id: from.into(),
            to_region_id: to.into(),
            intent,
            relation: classify_tonal_relation(from_key, to_key),
        }
    }

    fn trajectory() -> WorkTonalTrajectoryV1 {
        let home = Key::major(PitchClass::C);
        let dominant = home.dominant();
        let relative = home.relative();
        let regions = [
            (
                "A".into(),
                region(0, 8, home, TonalRegionRoleV1::Home),
            ),
            (
                "B".into(),
                region(8, 16, dominant, TonalRegionRoleV1::Departure),
            ),
            (
                "C".into(),
                region(16, 24, relative, TonalRegionRoleV1::Development),
            ),
            (
                "A-return".into(),
                region(24, 32, home, TonalRegionRoleV1::Return),
            ),
        ]
        .into_iter()
        .collect();
        WorkTonalTrajectoryV1 {
            version: WORK_TONAL_TRAJECTORY_VERSION.into(),
            trajectory_id: "test-trajectory".into(),
            home_key: home,
            coverage: TonalTrajectoryCoverageV1::CompleteWork,
            closure: TonalClosurePolicyV1::ReturnToHome,
            regions,
            region_order: vec!["A".into(), "B".into(), "C".into(), "A-return".into()],
            transitions: vec![
                transition("A", "B", TonalTransitionIntentV1::Depart, home, dominant),
                transition(
                    "B",
                    "C",
                    TonalTransitionIntentV1::Develop,
                    dominant,
                    relative,
                ),
                transition(
                    "C",
                    "A-return",
                    TonalTransitionIntentV1::ReturnHome,
                    relative,
                    home,
                ),
            ],
        }
    }

    #[test]
    fn complete_home_departure_development_return_is_valid() {
        let trajectory = trajectory();
        trajectory.validate(&work()).unwrap();
        assert_eq!(
            trajectory.transitions[0].relation,
            TonalRelationV1::SourceToDominant
        );
        assert_eq!(
            trajectory.transitions[2].relation,
            TonalRelationV1::SourceToRelative
        );
    }

    #[test]
    fn complete_coverage_rejects_gaps() {
        let mut trajectory = trajectory();
        trajectory.regions.get_mut("B").unwrap().start = Duration::new(9, 1);
        assert_eq!(
            trajectory.validate(&work()),
            Err(WorkTonalTrajectoryErrorV1::CompleteCoverageGap {
                left_region_id: "A".into(),
                right_region_id: "B".into(),
            })
        );
    }

    #[test]
    fn declared_only_policy_may_leave_silence_between_tonal_claims() {
        let mut trajectory = trajectory();
        trajectory.coverage = TonalTrajectoryCoverageV1::DeclaredRegionsOnly;
        trajectory.regions.get_mut("B").unwrap().start = Duration::new(9, 1);
        trajectory.validate(&work()).unwrap();
        assert!(
            trajectory
                .region_at(&work(), Duration::new(17, 2))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn return_to_home_closure_is_exact_key_identity() {
        let mut trajectory = trajectory();
        trajectory.regions.get_mut("A-return").unwrap().key = Key::minor(PitchClass::C);
        trajectory.transitions[2].relation = classify_tonal_relation(
            trajectory.regions["C"].key,
            trajectory.regions["A-return"].key,
        );
        assert_eq!(
            trajectory.validate(&work()),
            Err(WorkTonalTrajectoryErrorV1::ReturnClosureMustEndAtHome {
                region_id: "A-return".into(),
            })
        );
    }

    #[test]
    fn stored_relation_must_be_mechanically_rederivable() {
        let mut trajectory = trajectory();
        trajectory.transitions[0].relation = TonalRelationV1::Other;
        assert_eq!(
            trajectory.validate(&work()),
            Err(WorkTonalTrajectoryErrorV1::TonalRelationMismatch {
                transition_index: 0,
                expected: TonalRelationV1::SourceToDominant,
                found: TonalRelationV1::Other,
            })
        );
    }

    #[test]
    fn home_return_intent_cannot_point_elsewhere() {
        let mut trajectory = trajectory();
        trajectory.transitions[1].intent = TonalTransitionIntentV1::ReturnHome;
        assert_eq!(
            trajectory.validate(&work()),
            Err(WorkTonalTrajectoryErrorV1::HomeIntentRequiresHomeTarget {
                transition_index: 1,
            })
        );
    }

    #[test]
    fn region_query_uses_half_open_spans() {
        let trajectory = trajectory();
        let (region_id, _) = trajectory
            .region_at(&work(), Duration::new(8, 1))
            .unwrap()
            .unwrap();
        assert_eq!(region_id, "B");
        assert!(
            trajectory
                .region_at(&work(), Duration::new(32, 1))
                .unwrap()
                .is_none()
        );
    }
}
