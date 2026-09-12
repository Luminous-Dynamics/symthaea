// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Frozen candidate/corpus lineage for EUREKA-002E.
//!
//! This module binds the target subject and execution-environment commitments,
//! allocates disjoint deterministic synthetic-world schedules, freezes evaluator
//! action choice, and prevents held-out evidence from starting before candidate
//! and comparator identities are fixed.
//!
//! Parent issue: <https://github.com/Luminous-Dynamics/symthaea/issues/2103>

use std::collections::HashSet;

use super::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
use super::baselines::{HeldOutTransitionCorpus, TransitionRecorder};
use super::hidden_world::{
    CorpusPartition, EvaluatorWorld, FixtureFamily, PublicAction, WorldBuildProfile,
};
use super::selection::{
    PrimaryComparatorSelection, PrimaryComparatorSelectionStatus,
};

pub(super) const CAMPAIGN_SCHEDULE_REVISION: &str = "EUREKA.002E.CORPUS_SCHEDULE.v1";
pub(super) const CAMPAIGN_LINEAGE_REVISION: &str = "EUREKA.002E.CAMPAIGN_LINEAGE.v1";

pub(super) const DEVELOPMENT_WORLDS_PER_FAMILY: u16 = 256;
pub(super) const CALIBRATION_WORLDS_PER_FAMILY: u16 = 128;
pub(super) const HELD_OUT_WORLDS_PER_FAMILY: u16 = 64;
pub(super) const EXTERNAL_WORLDS_PER_FAMILY: u16 = 32;

pub(super) const EUREKA_FIXTURE_FAMILIES: [FixtureFamily; 2] = [
    FixtureFamily::CausalBits,
    FixtureFamily::ResourceFlow,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct ScheduledWorldProfile {
    pub schedule_revision: &'static str,
    pub partition: CorpusPartition,
    pub family: FixtureFamily,
    pub index: u16,
    pub world: WorldBuildProfile,
}

impl ScheduledWorldProfile {
    pub(super) fn replay_digest(self) -> u64 {
        let mut bytes = Vec::new();
        encode_str(&mut bytes, self.schedule_revision);
        bytes.push(partition_tag(self.partition));
        bytes.push(family_tag(self.family));
        bytes.extend_from_slice(&self.index.to_le_bytes());
        bytes.extend_from_slice(&self.world.seed.to_le_bytes());
        bytes.push(self.world.mechanism_variant);
        fnv1a64(&bytes)
    }
}

pub(super) fn scheduled_world_count(partition: CorpusPartition) -> u16 {
    match partition {
        CorpusPartition::Development => DEVELOPMENT_WORLDS_PER_FAMILY,
        CorpusPartition::Calibration => CALIBRATION_WORLDS_PER_FAMILY,
        CorpusPartition::HeldOutEvaluation => HELD_OUT_WORLDS_PER_FAMILY,
        CorpusPartition::ExternalReplication => EXTERNAL_WORLDS_PER_FAMILY,
    }
}

pub(super) fn scheduled_profiles(
    partition: CorpusPartition,
    family: FixtureFamily,
) -> Vec<ScheduledWorldProfile> {
    let count = scheduled_world_count(partition);
    let base = seed_namespace_base(partition, family);
    (0..count)
        .map(|index| ScheduledWorldProfile {
            schedule_revision: CAMPAIGN_SCHEDULE_REVISION,
            partition,
            family,
            index,
            world: WorldBuildProfile {
                family,
                seed: base + u64::from(index),
                mechanism_variant: (index % 2) as u8,
                partition,
            },
        })
        .collect()
}

/// Deterministic evaluator action choice from the exact public legal-action
/// list. The target prediction/confidence is not an input.
pub(super) fn scheduled_action(
    profile: ScheduledWorldProfile,
    legal_actions: &[PublicAction],
) -> Result<PublicAction, CampaignScheduleError> {
    if legal_actions.is_empty() {
        return Err(CampaignScheduleError::NoLegalActions);
    }
    Ok(legal_actions[usize::from(profile.index) % legal_actions.len()])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CampaignScheduleError {
    NoLegalActions,
}

fn seed_namespace_base(partition: CorpusPartition, family: FixtureFamily) -> u64 {
    let partition_prefix = match partition {
        CorpusPartition::Development => 0x1000_0000_0000_0000,
        CorpusPartition::Calibration => 0x2000_0000_0000_0000,
        CorpusPartition::HeldOutEvaluation => 0x3000_0000_0000_0000,
        CorpusPartition::ExternalReplication => 0x4000_0000_0000_0000,
    };
    let family_offset = match family {
        FixtureFamily::CausalBits => 0,
        FixtureFamily::ResourceFlow => 0x0100_0000_0000_0000,
    };
    partition_prefix + family_offset
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CandidateSubjectBindingV1 {
    pub source_commit_hex: String,
    pub source_tree_sha256: String,
    pub candidate_config_sha256: String,
    pub target_adapter_revision: String,
    pub flake_lock_sha256: String,
    pub toolchain_evidence_sha256: String,
    pub execution_environment_sha256: String,
    pub command_runner_sha256: String,
    pub analysis_plan_digest: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CandidateBindingError {
    InvalidSourceCommit,
    InvalidSourceTreeDigest,
    InvalidCandidateConfigDigest,
    EmptyTargetAdapterRevision,
    InvalidFlakeLockDigest,
    InvalidToolchainDigest,
    InvalidExecutionEnvironmentDigest,
    InvalidCommandRunnerDigest,
    AnalysisPlanMismatch,
}

impl CandidateSubjectBindingV1 {
    pub(super) fn validate(&self) -> Result<(), CandidateBindingError> {
        if !is_git_commit_hex(&self.source_commit_hex) {
            return Err(CandidateBindingError::InvalidSourceCommit);
        }
        for (value, error) in [
            (
                self.source_tree_sha256.as_str(),
                CandidateBindingError::InvalidSourceTreeDigest,
            ),
            (
                self.candidate_config_sha256.as_str(),
                CandidateBindingError::InvalidCandidateConfigDigest,
            ),
            (
                self.flake_lock_sha256.as_str(),
                CandidateBindingError::InvalidFlakeLockDigest,
            ),
            (
                self.toolchain_evidence_sha256.as_str(),
                CandidateBindingError::InvalidToolchainDigest,
            ),
            (
                self.execution_environment_sha256.as_str(),
                CandidateBindingError::InvalidExecutionEnvironmentDigest,
            ),
            (
                self.command_runner_sha256.as_str(),
                CandidateBindingError::InvalidCommandRunnerDigest,
            ),
        ] {
            if !is_sha256(value) {
                return Err(error);
            }
        }
        if self.target_adapter_revision.trim().is_empty() {
            return Err(CandidateBindingError::EmptyTargetAdapterRevision);
        }
        if self.analysis_plan_digest != EUREKA_002_ANALYSIS_PLAN_V1.replay_digest() {
            return Err(CandidateBindingError::AnalysisPlanMismatch);
        }
        Ok(())
    }

    pub(super) fn replay_digest(&self) -> Result<u64, CandidateBindingError> {
        self.validate()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"eureka.candidate-subject-binding.v1\0");
        encode_str(&mut bytes, &self.source_commit_hex);
        encode_str(&mut bytes, &self.source_tree_sha256);
        encode_str(&mut bytes, &self.candidate_config_sha256);
        encode_str(&mut bytes, &self.target_adapter_revision);
        encode_str(&mut bytes, &self.flake_lock_sha256);
        encode_str(&mut bytes, &self.toolchain_evidence_sha256);
        encode_str(&mut bytes, &self.execution_environment_sha256);
        encode_str(&mut bytes, &self.command_runner_sha256);
        bytes.extend_from_slice(&self.analysis_plan_digest.to_le_bytes());
        Ok(fnv1a64(&bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum CampaignPhase {
    DesignOpen,
    CandidateFrozen,
    ComparatorFrozen,
    HeldOutRunning,
    HeldOutComplete,
    ExternalReplicationRunning,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CampaignLineageError {
    WrongPhase,
    InvalidCandidate(CandidateBindingError),
    ComparatorNotSelected,
    ComparatorAnalysisPlanMismatch,
    WrongCorpusPartition,
    CorpusScheduleMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Eureka002CampaignLineage {
    phase: CampaignPhase,
    candidate: Option<CandidateSubjectBindingV1>,
    candidate_digest: Option<u64>,
    comparator_selection_digest: Option<u64>,
    held_out_corpus_digest: Option<u64>,
    external_corpus_digest: Option<u64>,
}

impl Default for Eureka002CampaignLineage {
    fn default() -> Self {
        Self {
            phase: CampaignPhase::DesignOpen,
            candidate: None,
            candidate_digest: None,
            comparator_selection_digest: None,
            held_out_corpus_digest: None,
            external_corpus_digest: None,
        }
    }
}

impl Eureka002CampaignLineage {
    pub(super) fn phase(&self) -> CampaignPhase {
        self.phase
    }

    pub(super) fn freeze_candidate(
        &mut self,
        candidate: CandidateSubjectBindingV1,
    ) -> Result<(), CampaignLineageError> {
        if self.phase != CampaignPhase::DesignOpen {
            return Err(CampaignLineageError::WrongPhase);
        }
        let digest = candidate
            .replay_digest()
            .map_err(CampaignLineageError::InvalidCandidate)?;
        self.candidate = Some(candidate);
        self.candidate_digest = Some(digest);
        self.phase = CampaignPhase::CandidateFrozen;
        Ok(())
    }

    pub(super) fn freeze_comparator(
        &mut self,
        selection: &PrimaryComparatorSelection,
    ) -> Result<(), CampaignLineageError> {
        if self.phase != CampaignPhase::CandidateFrozen {
            return Err(CampaignLineageError::WrongPhase);
        }
        if selection.status != PrimaryComparatorSelectionStatus::Selected
            || selection.selected.is_none()
        {
            return Err(CampaignLineageError::ComparatorNotSelected);
        }
        if selection.analysis_plan_digest != EUREKA_002_ANALYSIS_PLAN_V1.replay_digest() {
            return Err(CampaignLineageError::ComparatorAnalysisPlanMismatch);
        }
        self.comparator_selection_digest = Some(selection.replay_digest());
        self.phase = CampaignPhase::ComparatorFrozen;
        Ok(())
    }

    pub(super) fn begin_held_out(
        &mut self,
        corpus: &HeldOutTransitionCorpus,
    ) -> Result<(), CampaignLineageError> {
        if self.phase != CampaignPhase::ComparatorFrozen {
            return Err(CampaignLineageError::WrongPhase);
        }
        validate_corpus_schedule(CorpusPartition::HeldOutEvaluation, corpus)?;
        self.held_out_corpus_digest = Some(corpus.digest());
        self.phase = CampaignPhase::HeldOutRunning;
        Ok(())
    }

    pub(super) fn complete_held_out(&mut self) -> Result<(), CampaignLineageError> {
        if self.phase != CampaignPhase::HeldOutRunning {
            return Err(CampaignLineageError::WrongPhase);
        }
        self.phase = CampaignPhase::HeldOutComplete;
        Ok(())
    }

    pub(super) fn begin_external_replication(
        &mut self,
        corpus: &HeldOutTransitionCorpus,
    ) -> Result<(), CampaignLineageError> {
        if self.phase != CampaignPhase::HeldOutComplete {
            return Err(CampaignLineageError::WrongPhase);
        }
        validate_corpus_schedule(CorpusPartition::ExternalReplication, corpus)?;
        self.external_corpus_digest = Some(corpus.digest());
        self.phase = CampaignPhase::ExternalReplicationRunning;
        Ok(())
    }

    pub(super) fn complete_external_replication(&mut self) -> Result<(), CampaignLineageError> {
        if self.phase != CampaignPhase::ExternalReplicationRunning {
            return Err(CampaignLineageError::WrongPhase);
        }
        self.phase = CampaignPhase::Complete;
        Ok(())
    }

    /// Stable campaign-lineage root once candidate and comparator are frozen.
    /// Evidence outcomes are not inputs to this root.
    pub(super) fn lineage_root_digest(&self) -> Option<u64> {
        let candidate = self.candidate_digest?;
        let comparator = self.comparator_selection_digest?;
        let mut bytes = Vec::new();
        encode_str(&mut bytes, CAMPAIGN_LINEAGE_REVISION);
        encode_str(&mut bytes, CAMPAIGN_SCHEDULE_REVISION);
        bytes.extend_from_slice(&EUREKA_002_ANALYSIS_PLAN_V1.replay_digest().to_le_bytes());
        bytes.extend_from_slice(&candidate.to_le_bytes());
        bytes.extend_from_slice(&comparator.to_le_bytes());
        Some(fnv1a64(&bytes))
    }
}

fn validate_corpus_schedule(
    partition: CorpusPartition,
    corpus: &HeldOutTransitionCorpus,
) -> Result<(), CampaignLineageError> {
    if corpus
        .records()
        .iter()
        .any(|record| record.partition() != partition)
    {
        return Err(CampaignLineageError::WrongCorpusPartition);
    }

    let actual: HashSet<u64> = corpus
        .records()
        .iter()
        .map(|record| record.world_digest())
        .collect();
    let expected: HashSet<u64> = EUREKA_FIXTURE_FAMILIES
        .iter()
        .flat_map(|family| scheduled_profiles(partition, *family))
        .map(|profile| EvaluatorWorld::build(profile.world).world_digest())
        .collect();

    if actual.len() != corpus.records().len()
        || actual.len() != expected.len()
        || actual != expected
    {
        return Err(CampaignLineageError::CorpusScheduleMismatch);
    }
    Ok(())
}

fn is_git_commit_hex(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn encode_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

const fn partition_tag(value: CorpusPartition) -> u8 {
    match value {
        CorpusPartition::Development => 1,
        CorpusPartition::Calibration => 2,
        CorpusPartition::HeldOutEvaluation => 3,
        CorpusPartition::ExternalReplication => 4,
    }
}

const fn family_tag(value: FixtureFamily) -> u8 {
    match value {
        FixtureFamily::CausalBits => 1,
        FixtureFamily::ResourceFlow => 2,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::benchmarks::eureka::baselines::HeldOutTransitionCorpus;
    use crate::benchmarks::eureka::selection::{
        PrimaryComparatorSelection, PrimaryComparatorSelectionStatus,
    };

    fn sha(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn candidate() -> CandidateSubjectBindingV1 {
        CandidateSubjectBindingV1 {
            source_commit_hex: std::iter::repeat_n('a', 40).collect(),
            source_tree_sha256: sha('b'),
            candidate_config_sha256: sha('c'),
            target_adapter_revision: "EUREKA.FULL_SYMTHAEA_ADAPTER.v1".into(),
            flake_lock_sha256: sha('d'),
            toolchain_evidence_sha256: sha('e'),
            execution_environment_sha256: sha('f'),
            command_runner_sha256: sha('1'),
            analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
        }
    }

    fn comparator() -> PrimaryComparatorSelection {
        PrimaryComparatorSelection {
            analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
            fit_corpus_digest: 11,
            selection_corpus_digest: 22,
            selection_revision: super::super::selection::COMPARATOR_SELECTION_REVISION,
            status: PrimaryComparatorSelectionStatus::Selected,
            selected: Some(super::super::baselines::ShortcutBaselineKind::ActionMarginalDelta),
            receipts: Vec::new(),
        }
    }

    fn scheduled_corpus(partition: CorpusPartition) -> HeldOutTransitionCorpus {
        let mut records = Vec::new();
        for family in EUREKA_FIXTURE_FAMILIES {
            for profile in scheduled_profiles(partition, family) {
                let mut recorder = TransitionRecorder::build(profile.world);
                let legal = recorder.runtime().legal_actions();
                let action = scheduled_action(profile, &legal).unwrap();
                records.push(recorder.execute_and_record(action).unwrap().1);
            }
        }
        HeldOutTransitionCorpus::freeze(records).unwrap()
    }

    #[test]
    fn seed_namespaces_are_pairwise_disjoint_across_partition_and_family() {
        let mut seen = HashSet::new();
        for partition in [
            CorpusPartition::Development,
            CorpusPartition::Calibration,
            CorpusPartition::HeldOutEvaluation,
            CorpusPartition::ExternalReplication,
        ] {
            for family in EUREKA_FIXTURE_FAMILIES {
                for profile in scheduled_profiles(partition, family) {
                    assert!(seen.insert(profile.world.seed));
                }
            }
        }
    }

    #[test]
    fn schedule_counts_and_mechanism_variants_are_frozen_and_balanced() {
        for partition in [
            CorpusPartition::Development,
            CorpusPartition::Calibration,
            CorpusPartition::HeldOutEvaluation,
            CorpusPartition::ExternalReplication,
        ] {
            for family in EUREKA_FIXTURE_FAMILIES {
                let profiles = scheduled_profiles(partition, family);
                assert_eq!(profiles.len(), usize::from(scheduled_world_count(partition)));
                let zeros = profiles
                    .iter()
                    .filter(|profile| profile.world.mechanism_variant == 0)
                    .count();
                let ones = profiles
                    .iter()
                    .filter(|profile| profile.world.mechanism_variant == 1)
                    .count();
                assert_eq!(zeros, ones);
            }
        }
    }

    #[test]
    fn scheduled_action_is_legal_deterministic_and_balanced() {
        for family in EUREKA_FIXTURE_FAMILIES {
            let profiles = scheduled_profiles(CorpusPartition::HeldOutEvaluation, family);
            let mut counts: HashMap<PublicAction, usize> = HashMap::new();
            for profile in profiles {
                let mut evaluator = EvaluatorWorld::build(profile.world);
                let legal = evaluator.runtime().legal_actions();
                let a = scheduled_action(profile, &legal).unwrap();
                let b = scheduled_action(profile, &legal).unwrap();
                assert_eq!(a, b);
                assert!(legal.contains(&a));
                *counts.entry(a).or_insert(0) += 1;
            }
            let min = counts.values().copied().min().unwrap();
            let max = counts.values().copied().max().unwrap();
            assert!(max - min <= 1);
        }
    }

    #[test]
    fn candidate_binding_rejects_malformed_identity_and_is_content_sensitive() {
        let valid = candidate();
        let digest = valid.replay_digest().unwrap();
        let mut malformed = valid.clone();
        malformed.flake_lock_sha256 = "not-a-sha".into();
        assert_eq!(
            malformed.validate(),
            Err(CandidateBindingError::InvalidFlakeLockDigest)
        );
        let mut changed = valid;
        changed.candidate_config_sha256 = sha('2');
        assert_ne!(digest, changed.replay_digest().unwrap());
    }

    #[test]
    fn held_out_cannot_begin_before_candidate_and_comparator_freeze() {
        let held_out = scheduled_corpus(CorpusPartition::HeldOutEvaluation);
        let mut lineage = Eureka002CampaignLineage::default();
        assert_eq!(
            lineage.begin_held_out(&held_out),
            Err(CampaignLineageError::WrongPhase)
        );
        lineage.freeze_candidate(candidate()).unwrap();
        assert_eq!(
            lineage.begin_held_out(&held_out),
            Err(CampaignLineageError::WrongPhase)
        );
    }

    #[test]
    fn phase_machine_is_one_way_and_external_replication_stays_separate() {
        let held_out = scheduled_corpus(CorpusPartition::HeldOutEvaluation);
        let external = scheduled_corpus(CorpusPartition::ExternalReplication);
        let mut lineage = Eureka002CampaignLineage::default();
        lineage.freeze_candidate(candidate()).unwrap();
        lineage.freeze_comparator(&comparator()).unwrap();
        let root = lineage.lineage_root_digest().unwrap();
        lineage.begin_held_out(&held_out).unwrap();
        assert_eq!(lineage.phase(), CampaignPhase::HeldOutRunning);
        assert_eq!(lineage.lineage_root_digest(), Some(root));
        lineage.complete_held_out().unwrap();
        lineage.begin_external_replication(&external).unwrap();
        assert_eq!(lineage.phase(), CampaignPhase::ExternalReplicationRunning);
        lineage.complete_external_replication().unwrap();
        assert_eq!(lineage.phase(), CampaignPhase::Complete);
        assert_eq!(lineage.lineage_root_digest(), Some(root));
    }

    #[test]
    fn frozen_candidate_cannot_be_replaced_in_same_lineage() {
        let mut lineage = Eureka002CampaignLineage::default();
        lineage.freeze_candidate(candidate()).unwrap();
        assert_eq!(
            lineage.freeze_candidate(candidate()),
            Err(CampaignLineageError::WrongPhase)
        );
    }

    #[test]
    fn corpus_missing_any_scheduled_world_fails_closed() {
        let mut records = Vec::new();
        let partition = CorpusPartition::HeldOutEvaluation;
        for family in EUREKA_FIXTURE_FAMILIES {
            for profile in scheduled_profiles(partition, family) {
                let mut recorder = TransitionRecorder::build(profile.world);
                let legal = recorder.runtime().legal_actions();
                let action = scheduled_action(profile, &legal).unwrap();
                records.push(recorder.execute_and_record(action).unwrap().1);
            }
        }
        records.pop();
        let corpus = HeldOutTransitionCorpus::freeze(records).unwrap();
        let mut lineage = Eureka002CampaignLineage::default();
        lineage.freeze_candidate(candidate()).unwrap();
        lineage.freeze_comparator(&comparator()).unwrap();
        assert_eq!(
            lineage.begin_held_out(&corpus),
            Err(CampaignLineageError::CorpusScheduleMismatch)
        );
    }

    #[test]
    fn wrong_partition_cannot_substitute_for_held_out() {
        let external = scheduled_corpus(CorpusPartition::ExternalReplication);
        let mut lineage = Eureka002CampaignLineage::default();
        lineage.freeze_candidate(candidate()).unwrap();
        lineage.freeze_comparator(&comparator()).unwrap();
        assert_eq!(
            lineage.begin_held_out(&external),
            Err(CampaignLineageError::WrongCorpusPartition)
        );
    }

    #[test]
    fn profile_identity_is_deterministic_and_schedule_bound() {
        let profile = scheduled_profiles(
            CorpusPartition::HeldOutEvaluation,
            FixtureFamily::CausalBits,
        )[0];
        assert_eq!(profile.replay_digest(), profile.replay_digest());
        let changed = ScheduledWorldProfile {
            schedule_revision: "EUREKA.002E.CORPUS_SCHEDULE.v2",
            ..profile
        };
        assert_ne!(profile.replay_digest(), changed.replay_digest());
    }
}
