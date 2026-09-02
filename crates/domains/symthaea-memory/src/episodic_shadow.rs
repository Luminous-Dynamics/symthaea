use crate::{episode_subject_sha256, Episode, EpisodicMemory, EpisodicProvenanceIndex};
use std::collections::HashSet;
use symthaea_epistemic_types::{ProvenanceRetrievalMode, RealityDomain};

/// Behavior-neutral audit for one provenance retrieval view.
///
/// This reports what a provenance-aware policy *would* have admitted from the
/// same similarity-ranked candidate set. It does not mutate memory, change
/// ranking, update confidence, or authorize any downstream behavior.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShadowRecallModeAudit {
    pub similarity_eligible: usize,
    pub provenance_admitted: usize,
    pub would_return: usize,
    pub excluded_unknown: usize,
    pub excluded_taint: usize,
    pub excluded_domain: usize,
    pub truncated_by_top_k: usize,
    pub overlap_with_raw_top_k: usize,
    pub would_change_selection: bool,
}

/// Comparison between the production raw top-k recall and epistemic shadow views.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EpisodicShadowRecallAudit {
    pub raw_returned: usize,
    pub grounded_history: ShadowRecallModeAudit,
    pub grounded_or_imported: ShadowRecallModeAudit,
    pub counterfactual_only: ShadowRecallModeAudit,
}

/// Audit input-similarity recall without changing production behavior.
pub fn shadow_audit_input_similarity(
    memory: &EpisodicMemory,
    provenance: &EpisodicProvenanceIndex,
    query: &[f32],
    top_k: usize,
) -> EpisodicShadowRecallAudit {
    let candidates = memory.retrieve_by_input_similarity(query, memory.len());
    shadow_audit_candidates(provenance, candidates, top_k)
}

/// Audit semantic-embedding recall without changing production behavior.
pub fn shadow_audit_embedding_similarity(
    memory: &EpisodicMemory,
    provenance: &EpisodicProvenanceIndex,
    query: &[f32],
    top_k: usize,
) -> EpisodicShadowRecallAudit {
    let candidates = memory.retrieve_by_embedding_similarity(query, memory.len());
    shadow_audit_candidates(provenance, candidates, top_k)
}

fn shadow_audit_candidates(
    provenance: &EpisodicProvenanceIndex,
    candidates: Vec<(Episode, f32)>,
    top_k: usize,
) -> EpisodicShadowRecallAudit {
    let raw_subjects: Vec<String> = candidates
        .iter()
        .take(top_k)
        .map(|(episode, _)| episode_subject_sha256(episode))
        .collect();
    let raw_set: HashSet<&str> = raw_subjects.iter().map(String::as_str).collect();

    EpisodicShadowRecallAudit {
        raw_returned: raw_subjects.len(),
        grounded_history: audit_mode(
            provenance,
            &candidates,
            top_k,
            ProvenanceRetrievalMode::GroundedHistory,
            &raw_subjects,
            &raw_set,
        ),
        grounded_or_imported: audit_mode(
            provenance,
            &candidates,
            top_k,
            ProvenanceRetrievalMode::GroundedOrImported,
            &raw_subjects,
            &raw_set,
        ),
        counterfactual_only: audit_mode(
            provenance,
            &candidates,
            top_k,
            ProvenanceRetrievalMode::CounterfactualOnly,
            &raw_subjects,
            &raw_set,
        ),
    }
}

fn audit_mode(
    provenance: &EpisodicProvenanceIndex,
    candidates: &[(Episode, f32)],
    top_k: usize,
    mode: ProvenanceRetrievalMode,
    raw_subjects: &[String],
    raw_set: &HashSet<&str>,
) -> ShadowRecallModeAudit {
    let mut admitted_subjects = Vec::new();
    let mut audit = ShadowRecallModeAudit {
        similarity_eligible: candidates.len(),
        ..ShadowRecallModeAudit::default()
    };

    for (episode, _) in candidates {
        let envelope = provenance.effective_provenance(episode);
        if mode.allows(&envelope) {
            audit.provenance_admitted += 1;
            admitted_subjects.push(episode_subject_sha256(episode));
            continue;
        }

        if envelope.domain == RealityDomain::Unknown {
            audit.excluded_unknown += 1;
        } else if envelope.counterfactual_taint {
            audit.excluded_taint += 1;
        } else {
            audit.excluded_domain += 1;
        }
    }

    audit.truncated_by_top_k = admitted_subjects.len().saturating_sub(top_k);
    admitted_subjects.truncate(top_k);
    audit.would_return = admitted_subjects.len();
    audit.overlap_with_raw_top_k = admitted_subjects
        .iter()
        .filter(|subject| raw_set.contains(subject.as_str()))
        .count();
    audit.would_change_selection = admitted_subjects != raw_subjects;
    audit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EpisodicReplayConfig;
    use symthaea_core::hdc::unified_hv::ContinuousHV;
    use symthaea_epistemic_types::{GroundingEvidence, ProvenanceEnvelope};

    fn episode_with_input(input: ContinuousHV, seed: u64) -> Episode {
        Episode::new(
            input,
            ContinuousHV::random(64, seed + 100),
            0.8,
            seed,
        )
    }

    #[test]
    fn legacy_unknown_memory_is_visible_raw_but_not_grounded_history() {
        let input = ContinuousHV::random(64, 10);
        let query = input.values.clone();
        let episode = episode_with_input(input, 10);
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        assert!(memory.store_if_significant(episode));
        let provenance = EpisodicProvenanceIndex::default();

        let audit = shadow_audit_input_similarity(&memory, &provenance, &query, 1);
        assert_eq!(audit.raw_returned, 1);
        assert_eq!(audit.grounded_history.would_return, 0);
        assert_eq!(audit.grounded_history.excluded_unknown, 1);
        assert!(audit.grounded_history.would_change_selection);
    }

    #[test]
    fn grounded_and_counterfactual_memories_are_separated_without_changing_raw_recall() {
        let shared_input = ContinuousHV::random(64, 20);
        let query = shared_input.values.clone();
        let grounded_episode = episode_with_input(shared_input.clone(), 20);
        let imagined_episode = episode_with_input(shared_input, 21);

        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        assert!(memory.store_if_significant(grounded_episode.clone()));
        assert!(memory.store_if_significant(imagined_episode.clone()));

        let grounded_subject = episode_subject_sha256(&grounded_episode);
        let imagined_subject = episode_subject_sha256(&imagined_episode);
        let grounded = ProvenanceEnvelope::from_grounding(
            GroundingEvidence::direct_observation(
                grounded_subject,
                "obs-20",
                "sensor",
                Some(20),
                0.95,
            )
            .unwrap(),
        );
        let imagined = ProvenanceEnvelope::new(
            imagined_subject,
            RealityDomain::Counterfactual,
            vec!["planner".into()],
            Some(21),
            0.7,
        )
        .unwrap();

        let mut provenance = EpisodicProvenanceIndex::default();
        provenance.attach(&grounded_episode, grounded).unwrap();
        provenance.attach(&imagined_episode, imagined).unwrap();

        let audit = shadow_audit_input_similarity(&memory, &provenance, &query, 2);
        assert_eq!(audit.raw_returned, 2);
        assert_eq!(audit.grounded_history.provenance_admitted, 1);
        assert_eq!(audit.grounded_history.would_return, 1);
        assert_eq!(audit.grounded_history.excluded_taint, 1);
        assert_eq!(audit.counterfactual_only.provenance_admitted, 1);
        assert_eq!(audit.counterfactual_only.would_return, 1);
        assert!(audit.grounded_history.would_change_selection);
        assert!(audit.counterfactual_only.would_change_selection);
    }

    #[test]
    fn top_k_zero_is_auditable_and_never_returns_shadow_matches() {
        let input = ContinuousHV::random(64, 30);
        let query = input.values.clone();
        let episode = episode_with_input(input, 30);
        let mut memory = EpisodicMemory::new(EpisodicReplayConfig::broad_capture());
        assert!(memory.store_if_significant(episode));

        let audit = shadow_audit_input_similarity(
            &memory,
            &EpisodicProvenanceIndex::default(),
            &query,
            0,
        );
        assert_eq!(audit.raw_returned, 0);
        assert_eq!(audit.grounded_history.would_return, 0);
        assert_eq!(audit.grounded_history.similarity_eligible, 1);
    }
}
