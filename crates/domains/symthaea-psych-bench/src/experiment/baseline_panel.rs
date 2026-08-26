// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Atomic matched-exposure execution for SYM-ARCH-002B1 baselines.
//!
//! A static fairness audit can prove that baseline configurations are comparable,
//! but it cannot prove that a later runner actually showed every model the same
//! examples in the same order. `MatchedBaselinePanel` is the preferred claim-
//! bearing execution path: one call feeds all baselines, updates are fail-atomic,
//! and training/evaluation streams receive deterministic cryptographic digests.

use crate::experiment_baselines::{
    BaselineResourceFootprint, MatchedBaselineFamilySpec, SimpleBaselineAgent, SimpleBaselineKind,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MATCHED_BASELINE_PANEL_SCHEMA_V1: &str = "symthaea.matched-baseline-panel/v1";
const TRAIN_STREAM_DOMAIN: &[u8] = b"symthaea.matched-baseline-panel.train/v1";
const EVAL_STREAM_DOMAIN: &[u8] = b"symthaea.matched-baseline-panel.eval/v1";
const PANEL_SNAPSHOT_DOMAIN: &[u8] = b"symthaea.matched-baseline-panel.snapshot/v1";

fn initialized_hasher(domain: &[u8]) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher
}

fn update_stream_digest(
    hasher: &mut blake3::Hasher,
    index: usize,
    assignment: &BTreeMap<String, i64>,
    label: bool,
) -> Result<(), String> {
    let index = u64::try_from(index).map_err(|_| "stream index exceeds u64".to_string())?;
    let bytes = serde_json::to_vec(&(index, assignment, label)).map_err(|error| error.to_string())?;
    let len = u64::try_from(bytes.len()).map_err(|_| "serialized stream item too large".to_string())?;
    hasher.update(&len.to_le_bytes());
    hasher.update(&bytes);
    Ok(())
}

fn digest_hex(hasher: &blake3::Hasher) -> String {
    hasher.clone().finalize().to_hex().to_string()
}

fn looks_like_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn kind_rank(kind: SimpleBaselineKind) -> u8 {
    match kind {
        SimpleBaselineKind::OneHotRls => 0,
        SimpleBaselineKind::FixedRandomTanhRls => 1,
        SimpleBaselineKind::VanillaHdcRls => 2,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineUpdateReceipt {
    pub kind: SimpleBaselineKind,
    /// RLS target-minus-prediction residual before the update.
    pub residual: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineEvaluationReceipt {
    pub kind: SimpleBaselineKind,
    pub score: f64,
    pub prediction: bool,
    pub expected_label: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselinePanelMemberSnapshot {
    pub kind: SimpleBaselineKind,
    pub spec_digest: String,
    pub updates: usize,
    pub resources: BaselineResourceFootprint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchedBaselinePanelSnapshot {
    pub schema: String,
    pub training_observations: usize,
    pub evaluation_observations: usize,
    pub training_stream_digest: String,
    pub evaluation_stream_digest: String,
    pub members: Vec<BaselinePanelMemberSnapshot>,
}

impl MatchedBaselinePanelSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MATCHED_BASELINE_PANEL_SCHEMA_V1 {
            return Err(format!("unsupported matched-baseline panel schema: {}", self.schema));
        }
        if !looks_like_digest(&self.training_stream_digest)
            || !looks_like_digest(&self.evaluation_stream_digest)
        {
            return Err("panel stream digests must be 32-byte hex digests".into());
        }
        if self.members.len() != 3 {
            return Err("B1 matched panel must contain exactly three baseline members".into());
        }
        let mut kinds = Vec::with_capacity(self.members.len());
        for member in &self.members {
            if kinds.contains(&member.kind) {
                return Err("B1 matched panel contains a duplicate baseline kind".into());
            }
            kinds.push(member.kind);
            if !looks_like_digest(&member.spec_digest) {
                return Err("baseline member spec digest must be a 32-byte hex digest".into());
            }
            if member.updates != self.training_observations {
                return Err(format!(
                    "baseline {:?} update count {} does not match panel training count {}",
                    member.kind, member.updates, self.training_observations
                ));
            }
        }
        for required in [
            SimpleBaselineKind::OneHotRls,
            SimpleBaselineKind::FixedRandomTanhRls,
            SimpleBaselineKind::VanillaHdcRls,
        ] {
            if !kinds.contains(&required) {
                return Err(format!("B1 matched panel is missing baseline kind {required:?}"));
            }
        }
        Ok(())
    }

    /// Canonical digest over the exact baseline specs/resources, matched exposure
    /// streams, and update counts. Member order is normalized by baseline kind so
    /// serialization order cannot create a fake scientific variant.
    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        let mut canonical_members = self.members.clone();
        canonical_members.sort_by_key(|member| kind_rank(member.kind));
        let bytes = serde_json::to_vec(&(
            self.schema.as_str(),
            self.training_observations,
            self.evaluation_observations,
            self.training_stream_digest.as_str(),
            self.evaluation_stream_digest.as_str(),
            canonical_members,
        ))
        .map_err(|error| error.to_string())?;
        let mut hasher = initialized_hasher(PANEL_SNAPSHOT_DOMAIN);
        hasher.update(&bytes);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Debug, Clone)]
pub struct MatchedBaselinePanel {
    agents: Vec<SimpleBaselineAgent>,
    training_observations: usize,
    evaluation_observations: usize,
    training_hasher: blake3::Hasher,
    evaluation_hasher: blake3::Hasher,
}

impl MatchedBaselinePanel {
    pub fn new(family: &MatchedBaselineFamilySpec) -> Result<Self, String> {
        let agents = family.agents()?;
        if agents.len() != 3 {
            return Err("B1 matched family must emit exactly three agents".into());
        }
        Ok(Self {
            agents,
            training_observations: 0,
            evaluation_observations: 0,
            training_hasher: initialized_hasher(TRAIN_STREAM_DOMAIN),
            evaluation_hasher: initialized_hasher(EVAL_STREAM_DOMAIN),
        })
    }

    pub fn training_observations(&self) -> usize {
        self.training_observations
    }

    pub fn evaluation_observations(&self) -> usize {
        self.evaluation_observations
    }

    /// Feed one labeled training item to every baseline atomically.
    ///
    /// All agents are updated on clones first. If any encoder/readout rejects the
    /// item, no agent state, update counter, or stream digest changes.
    pub fn observe_all(
        &mut self,
        assignment: &BTreeMap<String, i64>,
        label: bool,
    ) -> Result<Vec<BaselineUpdateReceipt>, String> {
        let next_count = self
            .training_observations
            .checked_add(1)
            .ok_or_else(|| "training observation counter overflow".to_string())?;
        let mut next_agents = self.agents.clone();
        let mut receipts = Vec::with_capacity(next_agents.len());
        for agent in &mut next_agents {
            let residual = agent.observe(assignment, label).map_err(|error| {
                format!("baseline {:?} rejected matched training item: {error}", agent.kind())
            })?;
            receipts.push(BaselineUpdateReceipt {
                kind: agent.kind(),
                residual,
            });
        }

        let mut next_hasher = self.training_hasher.clone();
        update_stream_digest(
            &mut next_hasher,
            self.training_observations,
            assignment,
            label,
        )?;

        self.agents = next_agents;
        self.training_hasher = next_hasher;
        self.training_observations = next_count;
        Ok(receipts)
    }

    /// Evaluate every baseline on the same labeled item without changing model
    /// state. The evaluation exposure is recorded only after all models score it.
    pub fn evaluate_all(
        &mut self,
        assignment: &BTreeMap<String, i64>,
        expected_label: bool,
    ) -> Result<Vec<BaselineEvaluationReceipt>, String> {
        let next_count = self
            .evaluation_observations
            .checked_add(1)
            .ok_or_else(|| "evaluation observation counter overflow".to_string())?;
        let mut receipts = Vec::with_capacity(self.agents.len());
        for agent in &self.agents {
            let score = agent.score(assignment).map_err(|error| {
                format!("baseline {:?} rejected matched evaluation item: {error}", agent.kind())
            })?;
            receipts.push(BaselineEvaluationReceipt {
                kind: agent.kind(),
                score,
                prediction: score > 0.0,
                expected_label,
            });
        }

        let mut next_hasher = self.evaluation_hasher.clone();
        update_stream_digest(
            &mut next_hasher,
            self.evaluation_observations,
            assignment,
            expected_label,
        )?;
        self.evaluation_hasher = next_hasher;
        self.evaluation_observations = next_count;
        Ok(receipts)
    }

    pub fn snapshot(&self) -> Result<MatchedBaselinePanelSnapshot, String> {
        let mut members = Vec::with_capacity(self.agents.len());
        for agent in &self.agents {
            members.push(BaselinePanelMemberSnapshot {
                kind: agent.kind(),
                spec_digest: agent.spec_digest().to_string(),
                updates: agent.updates(),
                resources: agent.resources()?,
            });
        }
        let snapshot = MatchedBaselinePanelSnapshot {
            schema: MATCHED_BASELINE_PANEL_SCHEMA_V1.into(),
            training_observations: self.training_observations,
            evaluation_observations: self.evaluation_observations,
            training_stream_digest: digest_hex(&self.training_hasher),
            evaluation_stream_digest: digest_hex(&self.evaluation_hasher),
            members,
        };
        snapshot.validate()?;
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment_baselines::{
        CategoricalFeatureSchema, CategoricalFeatureSpec, RlsConfig,
    };

    fn family() -> MatchedBaselineFamilySpec {
        MatchedBaselineFamilySpec {
            feature_schema: CategoricalFeatureSchema {
                features: vec![
                    CategoricalFeatureSpec {
                        name: "a".into(),
                        values: vec![0, 1],
                    },
                    CategoricalFeatureSpec {
                        name: "b".into(),
                        values: vec![0, 1],
                    },
                ],
            },
            encoded_dimension: 16,
            representation_seed: 77,
            rls: RlsConfig {
                ridge: 1.0,
                forgetting_factor: 1.0,
                include_bias: true,
            },
        }
    }

    fn assignment(a: i64, b: i64) -> BTreeMap<String, i64> {
        BTreeMap::from([("a".into(), a), ("b".into(), b)])
    }

    #[test]
    fn panel_keeps_training_and_evaluation_exposure_matched() {
        let mut panel = MatchedBaselinePanel::new(&family()).unwrap();
        panel.observe_all(&assignment(0, 0), false).unwrap();
        panel.observe_all(&assignment(1, 0), true).unwrap();
        panel.observe_all(&assignment(0, 1), true).unwrap();
        let evaluations = panel.evaluate_all(&assignment(1, 1), false).unwrap();
        assert_eq!(evaluations.len(), 3);

        let snapshot = panel.snapshot().unwrap();
        assert_eq!(snapshot.training_observations, 3);
        assert_eq!(snapshot.evaluation_observations, 1);
        assert_eq!(snapshot.members.len(), 3);
        assert!(snapshot.members.iter().all(|member| member.updates == 3));
        assert!(looks_like_digest(&snapshot.training_stream_digest));
        assert!(looks_like_digest(&snapshot.evaluation_stream_digest));
        assert!(looks_like_digest(&snapshot.digest().unwrap()));
    }

    #[test]
    fn panel_stream_digests_are_deterministic_and_order_sensitive() {
        let mut first = MatchedBaselinePanel::new(&family()).unwrap();
        let mut second = MatchedBaselinePanel::new(&family()).unwrap();
        for panel in [&mut first, &mut second] {
            panel.observe_all(&assignment(0, 0), false).unwrap();
            panel.observe_all(&assignment(1, 0), true).unwrap();
            panel.evaluate_all(&assignment(0, 1), true).unwrap();
        }
        let a = first.snapshot().unwrap();
        let b = second.snapshot().unwrap();
        assert_eq!(a.training_stream_digest, b.training_stream_digest);
        assert_eq!(a.evaluation_stream_digest, b.evaluation_stream_digest);
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());

        let mut reversed = MatchedBaselinePanel::new(&family()).unwrap();
        reversed.observe_all(&assignment(1, 0), true).unwrap();
        reversed.observe_all(&assignment(0, 0), false).unwrap();
        reversed.evaluate_all(&assignment(0, 1), true).unwrap();
        let c = reversed.snapshot().unwrap();
        assert_ne!(a.training_stream_digest, c.training_stream_digest);
        assert_ne!(a.digest().unwrap(), c.digest().unwrap());
        assert_eq!(a.evaluation_stream_digest, c.evaluation_stream_digest);
    }

    #[test]
    fn snapshot_digest_is_independent_of_member_serialization_order() {
        let mut panel = MatchedBaselinePanel::new(&family()).unwrap();
        panel.observe_all(&assignment(0, 0), false).unwrap();
        let snapshot = panel.snapshot().unwrap();
        let expected = snapshot.digest().unwrap();
        let mut reordered = snapshot.clone();
        reordered.members.reverse();
        reordered.validate().unwrap();
        assert_eq!(expected, reordered.digest().unwrap());
    }

    #[test]
    fn failed_training_item_is_atomic() {
        let mut panel = MatchedBaselinePanel::new(&family()).unwrap();
        panel.observe_all(&assignment(0, 0), false).unwrap();
        let before = panel.snapshot().unwrap();

        let invalid = assignment(9, 0);
        assert!(panel.observe_all(&invalid, true).is_err());

        let after = panel.snapshot().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn failed_evaluation_item_does_not_change_exposure_ledger() {
        let mut panel = MatchedBaselinePanel::new(&family()).unwrap();
        panel.evaluate_all(&assignment(0, 0), false).unwrap();
        let before = panel.snapshot().unwrap();

        let invalid = assignment(0, 9);
        assert!(panel.evaluate_all(&invalid, true).is_err());

        let after = panel.snapshot().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn evaluation_label_is_bound_into_the_stream_digest() {
        let mut first = MatchedBaselinePanel::new(&family()).unwrap();
        let mut second = MatchedBaselinePanel::new(&family()).unwrap();
        first.evaluate_all(&assignment(1, 1), false).unwrap();
        second.evaluate_all(&assignment(1, 1), true).unwrap();
        assert_ne!(
            first.snapshot().unwrap().evaluation_stream_digest,
            second.snapshot().unwrap().evaluation_stream_digest
        );
    }
}
