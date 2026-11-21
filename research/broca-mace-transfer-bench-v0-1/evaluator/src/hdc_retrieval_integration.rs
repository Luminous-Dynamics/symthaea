use std::collections::{BTreeMap, BTreeSet};

use broca_mace_hdc_retrieval::{
    CandidateHit, CandidateIndex, IndexConfig, RepresentationProfile,
};
use broca_mace_policy::{decide, Action, Policy};

use super::{cases, Experience};

fn resolve_hits(
    hits: &[CandidateHit],
    canonical: &BTreeMap<u32, Experience>,
) -> Result<Vec<Experience>, u32> {
    let mut resolved = Vec::with_capacity(hits.len());
    let mut seen = BTreeSet::new();
    for hit in hits {
        if !seen.insert(hit.event_id) {
            return Err(hit.event_id);
        }
        let Some(experience) = canonical.get(&hit.event_id) else {
            return Err(hit.event_id);
        };
        resolved.push(*experience);
    }
    Ok(resolved)
}

#[test]
fn hdc_retrieval_ids_are_resolved_before_the_existing_mace_policy_runs() {
    let case = cases()
        .into_iter()
        .find(|item| item.name == "observational_high_reward_decoy")
        .expect("observational decoy exists");
    let index = CandidateIndex::build(
        &case.session_a,
        IndexConfig {
            dimension: 4096,
            codebook_seed: 0x4D41_4345,
            profile: RepresentationProfile::StaticContinuousHv,
        },
    )
    .expect("candidate index builds");
    let a = index.retrieve(case.session_b.scope, Action::A, 16);
    let b = index.retrieve(case.session_b.scope, Action::B, 16);
    let canonical: BTreeMap<u32, Experience> = case
        .session_a
        .iter()
        .map(|item| (item.event_id, *item))
        .collect();
    let mut resolved = resolve_hits(&a.candidates, &canonical).unwrap();
    resolved.extend(resolve_hits(&b.candidates, &canonical).unwrap());

    // Retrieval may surface a high-reward observational decoy. It cannot
    // promote its causal-identification method or bypass the policy gate.
    let decision = decide(Policy::CausalQualified, &resolved, case.session_b);
    assert_eq!(decision.action, None);
    assert!(decision.evidence_ids.is_empty());
    assert!(!a.candidates.is_empty());
    assert!(!b.candidates.is_empty());
}

#[test]
fn hdc_ltc_temporal_profile_changes_index_identity_but_not_event_authority() {
    let case = cases()
        .into_iter()
        .find(|item| item.name == "identified_effect_transfer")
        .expect("identified scenario exists");
    let index = CandidateIndex::build(
        &case.session_a,
        IndexConfig {
            dimension: 4096,
            codebook_seed: 99,
            profile: RepresentationProfile::HdcLtcZeroInput {
                elapsed_seconds: 0.1,
                tau_base_seconds: 10.0,
                tau_min_seconds: 0.01,
                tau_max_seconds: 20.0,
                neuron_seed: 777,
            },
        },
    )
    .expect("HDC-LTC index builds");
    let result = index.retrieve(case.session_b.scope, Action::A, 16);
    let canonical: BTreeMap<u32, Experience> = case
        .session_a
        .iter()
        .map(|item| (item.event_id, *item))
        .collect();
    let resolved = resolve_hits(&result.candidates, &canonical).unwrap();
    assert!(resolved.iter().all(|item| {
        item.scope == case.session_b.scope && item.action == Action::A
    }));
    assert_eq!(result.candidate_count, case.session_a.len());
    assert!(result.index_payload_bytes > 0);
    assert!(!result.abstained);
}

#[test]
fn unresolved_hdc_candidate_identifier_fails_closed() {
    let canonical = BTreeMap::new();
    let forged = CandidateHit {
        event_id: u32::MAX,
        similarity: 1.0,
        rank: 1,
    };
    assert!(matches!(
        resolve_hits(&[forged], &canonical),
        Err(event_id) if event_id == u32::MAX
    ));
}

#[test]
fn retracted_high_similarity_candidates_are_still_rejected_by_mace() {
    let case = cases()
        .into_iter()
        .find(|item| item.name == "source_retraction")
        .expect("retraction scenario exists");
    let index = CandidateIndex::build(
        &case.session_a,
        IndexConfig {
            dimension: 4096,
            ..IndexConfig::default()
        },
    )
    .unwrap();
    let mut resolved = Vec::new();
    let canonical: BTreeMap<u32, Experience> = case
        .session_a
        .iter()
        .map(|item| (item.event_id, *item))
        .collect();
    for action in [Action::A, Action::B] {
        let result = index.retrieve(case.session_b.scope, action, 16);
        resolved.extend(resolve_hits(&result.candidates, &canonical).unwrap());
    }
    let decision = decide(Policy::CausalQualified, &resolved, case.session_b);
    assert_eq!(decision.action, None);
    assert!(decision.evidence_ids.is_empty());
}
