// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measurement-only harness for the shadow entity↔HDC bridge.
//!
//! This module deliberately reports retrieval metrics without defining a PASS
//! threshold. A later preregistered qualification protocol can choose acceptance
//! criteria without changing the measurement implementation.

use super::claim_evidence::EpistemicLedger;
use super::entity_event::{EntityEventStore, EntityId};
use super::entity_hdc_bridge::{EntityHdcBridgeError, ShadowEntityHdcBridge};
use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use symthaea_core::hdc::unified_hv::BinaryHV;

/// One labeled, immutable shadow-retrieval case.
#[derive(Debug, Clone)]
pub struct EntityHdcEvaluationCase {
    pub case_id: String,
    pub observation: BinaryHV,
    pub expected_entity: EntityId,
    /// Explicit candidate universe for this case, including hard negatives.
    pub candidate_ids: Vec<EntityId>,
}

/// Per-case retrieval measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityHdcCaseResult {
    pub case_id: String,
    pub expected_entity: EntityId,
    /// One-based rank of the expected entity.
    pub expected_rank: usize,
    pub reciprocal_rank: f64,
    pub expected_similarity: f32,
    pub best_wrong_similarity: Option<f32>,
    /// expected_similarity - best_wrong_similarity. Positive means expected won.
    pub expected_margin: Option<f32>,
    pub top1_correct: bool,
    pub candidate_count: usize,
}

/// Aggregate descriptive measurements over a frozen case set.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityHdcEvaluationReport {
    pub case_count: usize,
    pub top1_correct_count: usize,
    pub top1_accuracy: f64,
    pub mean_reciprocal_rank: f64,
    pub mean_expected_similarity: f64,
    /// Mean margin for cases containing at least one wrong candidate.
    pub mean_expected_margin: Option<f64>,
    pub cases: Vec<EntityHdcCaseResult>,
}

impl Default for EntityHdcEvaluationReport {
    fn default() -> Self {
        Self {
            case_count: 0,
            top1_correct_count: 0,
            top1_accuracy: 0.0,
            mean_reciprocal_rank: 0.0,
            mean_expected_similarity: 0.0,
            mean_expected_margin: None,
            cases: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityHdcEvaluationError {
    EmptyCaseId,
    EmptyCandidateSet { case_id: String },
    ExpectedEntityMissing {
        case_id: String,
        expected_entity: EntityId,
    },
    DuplicateCandidate {
        case_id: String,
        entity_id: EntityId,
    },
    Bridge(EntityHdcBridgeError),
}

impl fmt::Display for EntityHdcEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCaseId => write!(f, "evaluation case id cannot be empty"),
            Self::EmptyCandidateSet { case_id } => {
                write!(f, "evaluation case '{case_id}' has no candidates")
            }
            Self::ExpectedEntityMissing {
                case_id,
                expected_entity,
            } => write!(
                f,
                "evaluation case '{case_id}' does not include expected entity {}",
                expected_entity.0
            ),
            Self::DuplicateCandidate { case_id, entity_id } => write!(
                f,
                "evaluation case '{case_id}' includes duplicate entity {}",
                entity_id.0
            ),
            Self::Bridge(error) => write!(f, "entity HDC bridge error: {error}"),
        }
    }
}

impl Error for EntityHdcEvaluationError {}

impl From<EntityHdcBridgeError> for EntityHdcEvaluationError {
    fn from(value: EntityHdcBridgeError) -> Self {
        Self::Bridge(value)
    }
}

/// Executes frozen shadow cases without mutating persistent knowledge state.
pub struct ShadowEntityHdcEvaluator;

impl ShadowEntityHdcEvaluator {
    pub fn evaluate(
        bridge: &mut ShadowEntityHdcBridge,
        store: &EntityEventStore,
        ledger: &EpistemicLedger,
        cases: &[EntityHdcEvaluationCase],
    ) -> Result<EntityHdcEvaluationReport, EntityHdcEvaluationError> {
        if cases.is_empty() {
            return Ok(EntityHdcEvaluationReport::default());
        }

        let mut results = Vec::with_capacity(cases.len());
        for case in cases {
            validate_case(case)?;
            let ranked = bridge.rank_candidates(
                &case.observation,
                store,
                ledger,
                &case.candidate_ids,
                case.candidate_ids.len(),
            )?;

            let expected_index = ranked
                .iter()
                .position(|candidate| candidate.entity_id == case.expected_entity)
                .expect("validated expected candidate must be returned by full ranking");
            let expected = &ranked[expected_index];
            let best_wrong_similarity = ranked
                .iter()
                .filter(|candidate| candidate.entity_id != case.expected_entity)
                .map(|candidate| candidate.similarity)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let expected_margin =
                best_wrong_similarity.map(|wrong| expected.similarity - wrong);
            let expected_rank = expected_index + 1;

            results.push(EntityHdcCaseResult {
                case_id: case.case_id.clone(),
                expected_entity: case.expected_entity,
                expected_rank,
                reciprocal_rank: 1.0 / expected_rank as f64,
                expected_similarity: expected.similarity,
                best_wrong_similarity,
                expected_margin,
                top1_correct: expected_rank == 1,
                candidate_count: ranked.len(),
            });
        }

        let case_count = results.len();
        let top1_correct_count = results.iter().filter(|case| case.top1_correct).count();
        let top1_accuracy = top1_correct_count as f64 / case_count as f64;
        let mean_reciprocal_rank =
            results.iter().map(|case| case.reciprocal_rank).sum::<f64>() / case_count as f64;
        let mean_expected_similarity = results
            .iter()
            .map(|case| case.expected_similarity as f64)
            .sum::<f64>()
            / case_count as f64;
        let margins: Vec<f64> = results
            .iter()
            .filter_map(|case| case.expected_margin.map(|margin| margin as f64))
            .collect();
        let mean_expected_margin = (!margins.is_empty())
            .then(|| margins.iter().sum::<f64>() / margins.len() as f64);

        Ok(EntityHdcEvaluationReport {
            case_count,
            top1_correct_count,
            top1_accuracy,
            mean_reciprocal_rank,
            mean_expected_similarity,
            mean_expected_margin,
            cases: results,
        })
    }
}

fn validate_case(case: &EntityHdcEvaluationCase) -> Result<(), EntityHdcEvaluationError> {
    if case.case_id.trim().is_empty() {
        return Err(EntityHdcEvaluationError::EmptyCaseId);
    }
    if case.candidate_ids.is_empty() {
        return Err(EntityHdcEvaluationError::EmptyCandidateSet {
            case_id: case.case_id.clone(),
        });
    }
    if !case.candidate_ids.contains(&case.expected_entity) {
        return Err(EntityHdcEvaluationError::ExpectedEntityMissing {
            case_id: case.case_id.clone(),
            expected_entity: case.expected_entity,
        });
    }

    let mut seen = HashSet::with_capacity(case.candidate_ids.len());
    for entity_id in case.candidate_ids.iter().copied() {
        if !seen.insert(entity_id) {
            return Err(EntityHdcEvaluationError::DuplicateCandidate {
                case_id: case.case_id.clone(),
                entity_id,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{ClaimKind, EntityType};

    #[test]
    fn reports_retrieval_metrics_without_defining_pass_fail() {
        let mut store = EntityEventStore::new();
        let mut ledger = EpistemicLedger::new();
        let alpha = store
            .create_entity("Alpha Observatory", EntityType::Organization, 1)
            .unwrap();
        let beta = store
            .create_entity("Beta Observatory", EntityType::Organization, 1)
            .unwrap();
        let gamma = store
            .create_entity("Gamma Observatory", EntityType::Organization, 1)
            .unwrap();
        let claim = ledger.add_claim(
            "Alpha Observatory measures stellar spectra",
            ClaimKind::Descriptive,
            Some("astronomy".into()),
            None,
            2,
        );
        store.attach_entity_claim(alpha, claim).unwrap();

        let mut bridge = ShadowEntityHdcBridge::with_seed(41);
        let alpha_signature = bridge
            .encode_entity_signature(&store, &ledger, alpha)
            .unwrap();
        let case = EntityHdcEvaluationCase {
            case_id: "alpha-exact".into(),
            observation: alpha_signature.vector,
            expected_entity: alpha,
            candidate_ids: vec![beta, gamma, alpha],
        };

        let report = ShadowEntityHdcEvaluator::evaluate(
            &mut bridge,
            &store,
            &ledger,
            &[case],
        )
        .unwrap();

        assert_eq!(report.case_count, 1);
        assert_eq!(report.top1_correct_count, 1);
        assert_eq!(report.top1_accuracy, 1.0);
        assert_eq!(report.mean_reciprocal_rank, 1.0);
        assert!(report.cases[0].expected_margin.unwrap() > 0.0);
    }

    #[test]
    fn evaluation_does_not_mutate_identity_or_epistemic_state() {
        let mut store = EntityEventStore::new();
        let ledger = EpistemicLedger::new();
        let alpha = store
            .create_entity("Alpha", EntityType::Concept, 1)
            .unwrap();
        let beta = store
            .create_entity("Beta", EntityType::Concept, 1)
            .unwrap();
        let entity_count = store.entity_count();
        let claim_count = ledger.claim_count();

        let mut bridge = ShadowEntityHdcBridge::with_seed(43);
        let observation = bridge
            .encode_entity_signature(&store, &ledger, alpha)
            .unwrap()
            .vector;
        let case = EntityHdcEvaluationCase {
            case_id: "read-only".into(),
            observation,
            expected_entity: alpha,
            candidate_ids: vec![alpha, beta],
        };

        ShadowEntityHdcEvaluator::evaluate(&mut bridge, &store, &ledger, &[case]).unwrap();
        assert_eq!(store.entity_count(), entity_count);
        assert_eq!(ledger.claim_count(), claim_count);
        assert_eq!(store.resolve_name("Alpha"), Some(alpha));
        assert_eq!(store.resolve_name("Beta"), Some(beta));
    }

    #[test]
    fn missing_expected_candidate_fails_closed() {
        let mut store = EntityEventStore::new();
        let ledger = EpistemicLedger::new();
        let alpha = store
            .create_entity("Alpha", EntityType::Concept, 1)
            .unwrap();
        let beta = store
            .create_entity("Beta", EntityType::Concept, 1)
            .unwrap();
        let mut bridge = ShadowEntityHdcBridge::with_seed(47);
        let observation = bridge
            .encode_entity_signature(&store, &ledger, alpha)
            .unwrap()
            .vector;
        let case = EntityHdcEvaluationCase {
            case_id: "missing-target".into(),
            observation,
            expected_entity: alpha,
            candidate_ids: vec![beta],
        };

        assert_eq!(
            ShadowEntityHdcEvaluator::evaluate(&mut bridge, &store, &ledger, &[case])
                .unwrap_err(),
            EntityHdcEvaluationError::ExpectedEntityMissing {
                case_id: "missing-target".into(),
                expected_entity: alpha,
            }
        );
    }

    #[test]
    fn duplicate_candidates_are_rejected_before_measurement() {
        let mut store = EntityEventStore::new();
        let ledger = EpistemicLedger::new();
        let alpha = store
            .create_entity("Alpha", EntityType::Concept, 1)
            .unwrap();
        let mut bridge = ShadowEntityHdcBridge::with_seed(53);
        let observation = bridge
            .encode_entity_signature(&store, &ledger, alpha)
            .unwrap()
            .vector;
        let case = EntityHdcEvaluationCase {
            case_id: "duplicates".into(),
            observation,
            expected_entity: alpha,
            candidate_ids: vec![alpha, alpha],
        };

        assert_eq!(
            ShadowEntityHdcEvaluator::evaluate(&mut bridge, &store, &ledger, &[case])
                .unwrap_err(),
            EntityHdcEvaluationError::DuplicateCandidate {
                case_id: "duplicates".into(),
                entity_id: alpha,
            }
        );
    }

    #[test]
    fn empty_suite_has_explicit_zero_measurements() {
        let store = EntityEventStore::new();
        let ledger = EpistemicLedger::new();
        let mut bridge = ShadowEntityHdcBridge::with_seed(59);

        assert_eq!(
            ShadowEntityHdcEvaluator::evaluate(&mut bridge, &store, &ledger, &[]).unwrap(),
            EntityHdcEvaluationReport::default()
        );
    }
}