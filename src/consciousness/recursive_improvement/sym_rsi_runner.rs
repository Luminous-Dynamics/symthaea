// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic execution and evidence capture for SYM-RSI-001 fixtures.
//!
//! The runner executes one policy against one frozen fixture/seed/split and records
//! every observed transition into an [`ExperienceTree`]. It never manufactures a
//! missing transition. Replay and policy evolution are separate consumers of the
//! resulting evidence.

use super::experience_tree::{
    ExperienceNode, ExperienceNodeId, ExperienceProvenance, ExperienceTree, ExperienceTreeError,
};
use super::sym_rsi_experiment::{
    ArmRunMetrics, EvaluationSplit, ExperimentArm, ExperimentDomainSpec, ExperimentHarnessError,
    SymRsiExperimentManifest, SymRsiRunReceipt, SYM_RSI_001_RECEIPT_SCHEMA,
};
use super::sym_rsi_fixtures::{
    FixtureDomainError, FixtureDomainKind, FixtureState, FixtureTransition,
    SYM_RSI_001_FIXTURE_ADAPTER_VERSION,
};
use serde::{Deserialize, Serialize};

pub trait FixturePolicy {
    fn policy_id(&self) -> &str;

    fn choose_action(
        &mut self,
        domain: FixtureDomainKind,
        seed: u64,
        split: EvaluationSplit,
        state: &FixtureState,
        legal_actions: &[u8],
    ) -> Option<u8>;
}

/// Fixed deterministic baseline. Changing `salt` creates another frozen policy
/// without introducing hidden RNG state.
#[derive(Debug, Clone)]
pub struct FixedHashPolicy {
    id: String,
    salt: u64,
}

impl FixedHashPolicy {
    pub fn new(id: impl Into<String>, salt: u64) -> Self {
        Self {
            id: id.into(),
            salt,
        }
    }
}

impl FixturePolicy for FixedHashPolicy {
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn choose_action(
        &mut self,
        domain: FixtureDomainKind,
        seed: u64,
        split: EvaluationSplit,
        state: &FixtureState,
        legal_actions: &[u8],
    ) -> Option<u8> {
        if legal_actions.is_empty() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.sym-rsi-001.fixed-hash-policy.v1\0");
        hasher.update(domain.id().as_bytes());
        hasher.update(&seed.to_le_bytes());
        hasher.update(&split_tag(split).to_le_bytes());
        hasher.update(&state.step.to_le_bytes());
        hasher.update(&state.a.to_le_bytes());
        hasher.update(&state.b.to_le_bytes());
        hasher.update(&state.aux.to_le_bytes());
        hasher.update(&self.salt.to_le_bytes());
        let digest = hasher.finalize();
        let bytes = digest.as_bytes();
        let raw = u64::from_le_bytes(bytes[0..8].try_into().expect("BLAKE3 has 32 bytes"));
        Some(legal_actions[raw as usize % legal_actions.len()])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureObservedStep {
    pub node_id: ExperienceNodeId,
    pub parent_node_id: ExperienceNodeId,
    pub action: u8,
    pub before: FixtureState,
    pub transition: FixtureTransition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureRunTrace {
    pub domain: FixtureDomainKind,
    pub split: EvaluationSplit,
    pub seed: u64,
    pub policy_id: String,
    pub root_node_id: ExperienceNodeId,
    pub final_node_id: ExperienceNodeId,
    pub initial_state: FixtureState,
    pub final_state: FixtureState,
    pub observed_steps: Vec<FixtureObservedStep>,
    pub best_solution_quality: f64,
    pub evaluator_calls: u64,
    pub reached_terminal: bool,
    pub evidence_digest: String,
    pub experience_tree: ExperienceTree,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiptDiagnostics {
    pub brier_score: Option<f64>,
    pub regression_rate: f64,
    pub policy_churn_rate: f64,
    pub replay_pool_coverage: f64,
    pub unsupported_action_rate: f64,
    pub safety_constraint_violations: u64,
    pub authority_boundary_violations: u64,
    pub generated_evidence_promoted: bool,
}

impl Default for ReceiptDiagnostics {
    fn default() -> Self {
        Self {
            brier_score: None,
            regression_rate: 0.0,
            policy_churn_rate: 0.0,
            replay_pool_coverage: 0.0,
            unsupported_action_rate: 0.0,
            safety_constraint_violations: 0,
            authority_boundary_violations: 0,
            generated_evidence_promoted: false,
        }
    }
}

pub fn run_fixture_policy<P: FixturePolicy>(
    manifest: &SymRsiExperimentManifest,
    domain: FixtureDomainKind,
    split: EvaluationSplit,
    seed: u64,
    policy: &mut P,
) -> Result<FixtureRunTrace, FixtureRunnerError> {
    manifest
        .validate()
        .map_err(FixtureRunnerError::ManifestInvalid)?;
    let spec = registered_domain(manifest, domain)?;
    ensure_registered_seed(spec, split, seed)?;

    let initial_state = domain.reset(seed, split);
    let mut state = initial_state.clone();
    let mut tree = ExperienceTree::new();
    let root_id = 1;
    let root_state_digest = fixture_state_digest(domain, seed, split, &state);
    let root_evidence = chained_evidence_digest(
        "ROOT",
        &root_state_digest,
        &root_state_digest,
        "ROOT",
    );
    tree.append(ExperienceNode {
        id: root_id,
        parent: None,
        world_state_digest: root_state_digest.clone(),
        action_digest: fixture_root_action_digest(domain),
        observation_digest: fixture_observation_digest(&state),
        realized_outcome_digest: root_state_digest,
        prediction_error: None,
        utility: vec![state.quality],
        compute_cost: 0.0,
        uncertainty: None,
        model_version: spec.adapter_version.clone(),
        evidence_digest: root_evidence.clone(),
        provenance: ExperienceProvenance::Observed,
    })
    .map_err(FixtureRunnerError::ExperienceTree)?;

    let mut observed_steps = Vec::new();
    let mut parent_id = root_id;
    let mut next_node_id = root_id + 1;
    let mut calls = 0_u64;
    let mut best_quality = state.quality;
    let mut evidence_digest = root_evidence;

    while !state.terminal && calls < spec.max_evaluator_calls {
        let legal = domain.legal_actions(&state, split);
        let action = policy
            .choose_action(domain, seed, split, &state, &legal)
            .ok_or_else(|| FixtureRunnerError::PolicyDeclinedAction {
                policy_id: policy.policy_id().to_owned(),
                step: state.step,
            })?;
        if !legal.contains(&action) {
            return Err(FixtureRunnerError::IllegalPolicyAction {
                policy_id: policy.policy_id().to_owned(),
                step: state.step,
                action,
            });
        }

        let before = state.clone();
        let transition = domain
            .step(seed, split, &before, action)
            .map_err(FixtureRunnerError::Domain)?;
        calls += 1;
        best_quality = best_quality.max(transition.observed_quality);

        let state_digest = fixture_state_digest(domain, seed, split, &transition.next_state);
        let action_digest = fixture_action_digest(domain, action);
        let observation_digest = fixture_observation_digest(&transition.next_state);
        let next_evidence = chained_evidence_digest(
            &evidence_digest,
            &action_digest,
            &state_digest,
            &observation_digest,
        );

        tree.append(ExperienceNode {
            id: next_node_id,
            parent: Some(parent_id),
            world_state_digest: state_digest.clone(),
            action_digest,
            observation_digest,
            realized_outcome_digest: state_digest,
            prediction_error: None,
            utility: vec![transition.observed_quality],
            compute_cost: 1.0,
            uncertainty: None,
            model_version: spec.adapter_version.clone(),
            evidence_digest: next_evidence.clone(),
            provenance: ExperienceProvenance::Observed,
        })
        .map_err(FixtureRunnerError::ExperienceTree)?;

        observed_steps.push(FixtureObservedStep {
            node_id: next_node_id,
            parent_node_id: parent_id,
            action,
            before,
            transition: transition.clone(),
        });

        state = transition.next_state;
        parent_id = next_node_id;
        next_node_id += 1;
        evidence_digest = next_evidence;
    }

    Ok(FixtureRunTrace {
        domain,
        split,
        seed,
        policy_id: policy.policy_id().to_owned(),
        root_node_id: root_id,
        final_node_id: parent_id,
        initial_state,
        final_state: state.clone(),
        observed_steps,
        best_solution_quality: best_quality,
        evaluator_calls: calls,
        reached_terminal: state.terminal,
        evidence_digest,
        experience_tree: tree,
    })
}

pub fn build_fixture_receipt(
    manifest: &SymRsiExperimentManifest,
    arm: ExperimentArm,
    trace: &FixtureRunTrace,
    diagnostics: ReceiptDiagnostics,
) -> Result<SymRsiRunReceipt, FixtureRunnerError> {
    manifest
        .validate()
        .map_err(FixtureRunnerError::ManifestInvalid)?;
    let spec = registered_domain(manifest, trace.domain)?;
    ensure_registered_seed(spec, trace.split, trace.seed)?;

    let receipt = SymRsiRunReceipt {
        schema: SYM_RSI_001_RECEIPT_SCHEMA.into(),
        experiment_id: manifest.experiment_id.clone(),
        preregistration_digest: manifest.preregistration_digest.clone(),
        subject_digest: manifest.subject_digest.clone(),
        environment_digest: manifest.environment_digest.clone(),
        domain_id: trace.domain.id().into(),
        adapter_version: spec.adapter_version.clone(),
        arm,
        split: trace.split,
        seed: trace.seed,
        policy_id: trace.policy_id.clone(),
        metrics: ArmRunMetrics {
            best_solution_quality: trace.best_solution_quality,
            evaluator_calls: trace.evaluator_calls,
            normalized_compute_cost: trace.evaluator_calls as f64,
            brier_score: diagnostics.brier_score,
            regression_rate: diagnostics.regression_rate,
            policy_churn_rate: diagnostics.policy_churn_rate,
            replay_pool_coverage: diagnostics.replay_pool_coverage,
            unsupported_action_rate: diagnostics.unsupported_action_rate,
            safety_constraint_violations: diagnostics.safety_constraint_violations,
            authority_boundary_violations: diagnostics.authority_boundary_violations,
        },
        evidence_digest: trace.evidence_digest.clone(),
        generated_evidence_promoted: diagnostics.generated_evidence_promoted,
    };
    receipt
        .validate()
        .map_err(FixtureRunnerError::ReceiptInvalid)?;
    Ok(receipt)
}

pub fn fixture_action_digest(domain: FixtureDomainKind, action: u8) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.fixture-action.v1\0");
    hasher.update(domain.id().as_bytes());
    hasher.update(&[action]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

pub fn fixture_state_digest(
    domain: FixtureDomainKind,
    seed: u64,
    split: EvaluationSplit,
    state: &FixtureState,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.fixture-state.v1\0");
    hasher.update(domain.id().as_bytes());
    hasher.update(&seed.to_le_bytes());
    hasher.update(&split_tag(split).to_le_bytes());
    hasher.update(&state.step.to_le_bytes());
    hasher.update(&state.a.to_le_bytes());
    hasher.update(&state.b.to_le_bytes());
    hasher.update(&state.aux.to_le_bytes());
    hasher.update(&state.quality.to_bits().to_le_bytes());
    hasher.update(&[u8::from(state.terminal)]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn fixture_root_action_digest(domain: FixtureDomainKind) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.fixture-root.v1\0");
    hasher.update(domain.id().as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn fixture_observation_digest(state: &FixtureState) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.fixture-observation.v1\0");
    hasher.update(&state.quality.to_bits().to_le_bytes());
    hasher.update(&[u8::from(state.terminal)]);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn chained_evidence_digest(
    previous: &str,
    action: &str,
    outcome: &str,
    observation: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea.sym-rsi-001.fixture-evidence-chain.v1\0");
    for value in [previous, action, outcome, observation] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn registered_domain<'a>(
    manifest: &'a SymRsiExperimentManifest,
    domain: FixtureDomainKind,
) -> Result<&'a ExperimentDomainSpec, FixtureRunnerError> {
    manifest
        .domains
        .iter()
        .find(|spec| spec.domain_id == domain.id())
        .ok_or(FixtureRunnerError::DomainNotRegistered(domain))
        .and_then(|spec| {
            if spec.adapter_version == SYM_RSI_001_FIXTURE_ADAPTER_VERSION {
                Ok(spec)
            } else {
                Err(FixtureRunnerError::AdapterVersionMismatch {
                    domain,
                    observed: spec.adapter_version.clone(),
                })
            }
        })
}

fn ensure_registered_seed(
    spec: &ExperimentDomainSpec,
    split: EvaluationSplit,
    seed: u64,
) -> Result<(), FixtureRunnerError> {
    let seeds = match split {
        EvaluationSplit::TrainingReplay => &spec.seeds.training_replay,
        EvaluationSplit::HeldOutReplay => &spec.seeds.held_out_replay,
        EvaluationSplit::FreshExecution => &spec.seeds.fresh_execution,
        EvaluationSplit::OutOfDistribution => &spec.seeds.out_of_distribution,
    };
    if seeds.contains(&seed) {
        Ok(())
    } else {
        Err(FixtureRunnerError::SeedNotRegistered {
            domain_id: spec.domain_id.clone(),
            split,
            seed,
        })
    }
}

fn split_tag(split: EvaluationSplit) -> u8 {
    match split {
        EvaluationSplit::TrainingReplay => 0,
        EvaluationSplit::HeldOutReplay => 1,
        EvaluationSplit::FreshExecution => 2,
        EvaluationSplit::OutOfDistribution => 3,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixtureRunnerError {
    ManifestInvalid(ExperimentHarnessError),
    DomainNotRegistered(FixtureDomainKind),
    AdapterVersionMismatch {
        domain: FixtureDomainKind,
        observed: String,
    },
    SeedNotRegistered {
        domain_id: String,
        split: EvaluationSplit,
        seed: u64,
    },
    PolicyDeclinedAction {
        policy_id: String,
        step: u32,
    },
    IllegalPolicyAction {
        policy_id: String,
        step: u32,
        action: u8,
    },
    Domain(FixtureDomainError),
    ExperienceTree(ExperienceTreeError),
    ReceiptInvalid(ExperimentHarnessError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        ExactReplayWorld, canonical_sym_rsi_001_fixture_manifest,
    };

    struct IllegalPolicy;

    impl FixturePolicy for IllegalPolicy {
        fn policy_id(&self) -> &str {
            "illegal"
        }

        fn choose_action(
            &mut self,
            _domain: FixtureDomainKind,
            _seed: u64,
            _split: EvaluationSplit,
            _state: &FixtureState,
            _legal_actions: &[u8],
        ) -> Option<u8> {
            Some(255)
        }
    }

    #[test]
    fn identical_execution_produces_identical_trace_identity() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut p1 = FixedHashPolicy::new("baseline", 7);
        let mut p2 = FixedHashPolicy::new("baseline", 7);
        let a = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::TrainingReplay,
            1,
            &mut p1,
        )
        .unwrap();
        let b = run_fixture_policy(
            &manifest,
            FixtureDomainKind::BranchingSearch,
            EvaluationSplit::TrainingReplay,
            1,
            &mut p2,
        )
        .unwrap();
        assert_eq!(a.observed_steps, b.observed_steps);
        assert_eq!(a.evidence_digest, b.evidence_digest);
        assert_eq!(a.best_solution_quality, b.best_solution_quality);
    }

    #[test]
    fn exact_replay_can_follow_recorded_first_edge() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut policy = FixedHashPolicy::new("baseline", 7);
        let trace = run_fixture_policy(
            &manifest,
            FixtureDomainKind::RuggedOptimization,
            EvaluationSplit::TrainingReplay,
            1,
            &mut policy,
        )
        .unwrap();
        let first = &trace.observed_steps[0];
        let replay = ExactReplayWorld::new(&trace.experience_tree);
        let replayed = replay
            .step(
                trace.root_node_id,
                &fixture_action_digest(trace.domain, first.action),
            )
            .unwrap();
        assert_eq!(replayed.id, first.node_id);
    }

    #[test]
    fn unregistered_seed_fails_before_execution() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut policy = FixedHashPolicy::new("baseline", 7);
        assert!(matches!(
            run_fixture_policy(
                &manifest,
                FixtureDomainKind::BranchingSearch,
                EvaluationSplit::TrainingReplay,
                999,
                &mut policy,
            ),
            Err(FixtureRunnerError::SeedNotRegistered { seed: 999, .. })
        ));
    }

    #[test]
    fn illegal_policy_action_fails_closed() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut policy = IllegalPolicy;
        assert!(matches!(
            run_fixture_policy(
                &manifest,
                FixtureDomainKind::DelayedNavigation,
                EvaluationSplit::TrainingReplay,
                1,
                &mut policy,
            ),
            Err(FixtureRunnerError::IllegalPolicyAction { action: 255, .. })
        ));
    }

    #[test]
    fn baseline_trace_builds_a_valid_receipt() {
        let manifest = canonical_sym_rsi_001_fixture_manifest("pre", "subject", "env");
        let mut policy = FixedHashPolicy::new("baseline", 7);
        let trace = run_fixture_policy(
            &manifest,
            FixtureDomainKind::DelayedNavigation,
            EvaluationSplit::HeldOutReplay,
            101,
            &mut policy,
        )
        .unwrap();
        let receipt = build_fixture_receipt(
            &manifest,
            ExperimentArm::AFixedExploration,
            &trace,
            ReceiptDiagnostics::default(),
        )
        .unwrap();
        assert_eq!(receipt.domain_id, FixtureDomainKind::DelayedNavigation.id());
        assert_eq!(receipt.seed, 101);
        assert!(!receipt.generated_evidence_promoted);
        assert!(receipt.evidence_digest.starts_with("blake3:"));
    }
}
