// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, bounded fault-tree analysis for release evidence.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_CUT_SETS: usize = 64;
pub const MAX_EVENTS_PER_CUT_SET: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BasicFault {
    CriticalSensorQuorumLost,
    HazardSupervisorUnavailable,
    RuntimeInvariantUnavailable,
    ProductiveActuatorStuckOn,
    CoolingUnavailable,
    ThermalLoadHigh,
    ReturnRouteInfeasible,
    BatteryReserveInsufficient,
    MobilityUnavailable,
    OperatorReplayAccepted,
    RecoveryQuorumBypassed,
    CheckpointSafetyStateLost,
}

impl BasicFault {
    pub const fn code(self) -> &'static str {
        match self {
            Self::CriticalSensorQuorumLost => "FLT-SEN-001",
            Self::HazardSupervisorUnavailable => "FLT-SAF-001",
            Self::RuntimeInvariantUnavailable => "FLT-MON-001",
            Self::ProductiveActuatorStuckOn => "FLT-ACT-001",
            Self::CoolingUnavailable => "FLT-THM-001",
            Self::ThermalLoadHigh => "FLT-THM-002",
            Self::ReturnRouteInfeasible => "FLT-RET-001",
            Self::BatteryReserveInsufficient => "FLT-PWR-001",
            Self::MobilityUnavailable => "FLT-MOB-001",
            Self::OperatorReplayAccepted => "FLT-AUT-001",
            Self::RecoveryQuorumBypassed => "FLT-AUT-002",
            Self::CheckpointSafetyStateLost => "FLT-RST-001",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultTreeNode {
    Basic(BasicFault),
    Any(Vec<FaultTreeNode>),
    All(Vec<FaultTreeNode>),
}

impl FaultTreeNode {
    pub fn evaluate(&self, active: &BTreeSet<BasicFault>) -> bool {
        match self {
            Self::Basic(fault) => active.contains(fault),
            Self::Any(children) => children.iter().any(|child| child.evaluate(active)),
            Self::All(children) => children.iter().all(|child| child.evaluate(active)),
        }
    }

    fn cut_sets(&self) -> Vec<BTreeSet<BasicFault>> {
        match self {
            Self::Basic(fault) => [BTreeSet::from([*fault])].into(),
            Self::Any(children) => {
                let mut result = Vec::new();
                for child in children {
                    result.extend(child.cut_sets());
                    if result.len() >= MAX_CUT_SETS {
                        break;
                    }
                }
                minimize(result)
            }
            Self::All(children) => {
                let mut combined = vec![BTreeSet::new()];
                for child in children {
                    let child_sets = child.cut_sets();
                    let mut next = Vec::new();
                    for left in &combined {
                        for right in &child_sets {
                            let mut merged = left.clone();
                            merged.extend(right.iter().copied());
                            if merged.len() <= MAX_EVENTS_PER_CUT_SET {
                                next.push(merged);
                            }
                            if next.len() >= MAX_CUT_SETS {
                                break;
                            }
                        }
                        if next.len() >= MAX_CUT_SETS {
                            break;
                        }
                    }
                    combined = minimize(next);
                }
                combined
            }
        }
    }
}

fn minimize(mut sets: Vec<BTreeSet<BasicFault>>) -> Vec<BTreeSet<BasicFault>> {
    sets.sort_by_key(|set| set.len());
    let mut minimal: Vec<BTreeSet<BasicFault>> = Vec::new();
    for candidate in sets {
        if minimal
            .iter()
            .any(|existing| existing.is_subset(&candidate))
        {
            continue;
        }
        minimal.retain(|existing| !candidate.is_subset(existing));
        minimal.push(candidate);
        if minimal.len() >= MAX_CUT_SETS {
            break;
        }
    }
    minimal
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TopEvent {
    UncontrolledProductiveMotion,
    ThermalRunaway,
    Entrapment,
    UnauthorizedMotionRecovery,
    UnsafeRestart,
}

impl TopEvent {
    pub const ALL: [Self; 5] = [
        Self::UncontrolledProductiveMotion,
        Self::ThermalRunaway,
        Self::Entrapment,
        Self::UnauthorizedMotionRecovery,
        Self::UnsafeRestart,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultTreeModel {
    trees: BTreeMap<TopEvent, FaultTreeNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultTreeEvaluation {
    pub active_top_events: Vec<TopEvent>,
    pub active_basic_faults: Vec<BasicFault>,
}

impl FaultTreeModel {
    pub fn canonical() -> Self {
        use BasicFault::*;
        use FaultTreeNode::{All, Any, Basic};
        let trees = BTreeMap::from([
            (
                TopEvent::UncontrolledProductiveMotion,
                All(vec![
                    Basic(ProductiveActuatorStuckOn),
                    Any(vec![
                        Basic(HazardSupervisorUnavailable),
                        Basic(RuntimeInvariantUnavailable),
                    ]),
                ]),
            ),
            (
                TopEvent::ThermalRunaway,
                All(vec![Basic(ThermalLoadHigh), Basic(CoolingUnavailable)]),
            ),
            (
                TopEvent::Entrapment,
                Any(vec![
                    All(vec![
                        Basic(ReturnRouteInfeasible),
                        Basic(MobilityUnavailable),
                    ]),
                    All(vec![
                        Basic(ReturnRouteInfeasible),
                        Basic(BatteryReserveInsufficient),
                    ]),
                ]),
            ),
            (
                TopEvent::UnauthorizedMotionRecovery,
                Any(vec![
                    Basic(OperatorReplayAccepted),
                    Basic(RecoveryQuorumBypassed),
                ]),
            ),
            (
                TopEvent::UnsafeRestart,
                All(vec![
                    Basic(CheckpointSafetyStateLost),
                    Any(vec![
                        Basic(CriticalSensorQuorumLost),
                        Basic(RuntimeInvariantUnavailable),
                    ]),
                ]),
            ),
        ]);
        Self { trees }
    }

    pub fn evaluate(&self, active: &BTreeSet<BasicFault>) -> FaultTreeEvaluation {
        FaultTreeEvaluation {
            active_top_events: self
                .trees
                .iter()
                .filter_map(|(event, tree)| tree.evaluate(active).then_some(*event))
                .collect(),
            active_basic_faults: active.iter().copied().collect(),
        }
    }

    pub fn minimal_cut_sets(&self, event: TopEvent) -> Vec<Vec<BasicFault>> {
        self.trees
            .get(&event)
            .map(FaultTreeNode::cut_sets)
            .unwrap_or_default()
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect()
    }

    pub fn validate(&self) -> bool {
        TopEvent::ALL.into_iter().all(|event| {
            self.trees.contains_key(&event) && !self.minimal_cut_sets(event).is_empty()
        })
    }
}

impl Default for FaultTreeModel {
    fn default() -> Self {
        Self::canonical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_fault_tree_has_bounded_minimal_cut_sets() {
        let model = FaultTreeModel::canonical();
        assert!(model.validate());
        for event in TopEvent::ALL {
            let cuts = model.minimal_cut_sets(event);
            assert!(!cuts.is_empty());
            assert!(cuts.len() <= MAX_CUT_SETS);
            assert!(cuts.iter().all(|cut| cut.len() <= MAX_EVENTS_PER_CUT_SET));
        }
    }

    #[test]
    fn thermal_top_event_requires_load_and_lost_cooling() {
        let model = FaultTreeModel::canonical();
        let active = BTreeSet::from([BasicFault::ThermalLoadHigh]);
        assert!(
            !model
                .evaluate(&active)
                .active_top_events
                .contains(&TopEvent::ThermalRunaway)
        );
        let active = BTreeSet::from([BasicFault::ThermalLoadHigh, BasicFault::CoolingUnavailable]);
        assert!(
            model
                .evaluate(&active)
                .active_top_events
                .contains(&TopEvent::ThermalRunaway)
        );
    }
}
