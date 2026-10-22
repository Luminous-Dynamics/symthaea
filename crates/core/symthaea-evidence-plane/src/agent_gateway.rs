//! Ingress boundary for external scientific-agent proposals.
//!
//! Proposals are normalized and integrity-checked here. Ingress never promotes
//! a proposal to an observation, replication, or official criterion evidence.

use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentProposal {
    pub proposal_id: String,
    pub agent_id: String,
    pub source: String,
    pub actor_id: String,
    pub payload: String,
    pub input_digest: String,
    pub artifact_digest: String,
    pub model_lineage: Option<String>,
    pub knowledge_cutoff: Option<String>,
    pub exposure_cutoff: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedProposal {
    pub proposal_id: String,
    pub agent_id: String,
    pub source: String,
    pub actor_id: String,
    pub payload_digest: String,
    pub input_digest: String,
    pub artifact_digest: String,
    pub model_lineage: Option<String>,
    pub knowledge_cutoff: Option<String>,
    pub exposure_cutoff: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalDisposition {
    AcceptedForPlanning,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayError {
    MissingField(&'static str),
    EmptyPayload,
    ExposureBeforeKnowledgeCutoff,
    PayloadDigestMismatch,
}

impl AgentProposal {
    pub fn normalize(&self) -> Result<NormalizedProposal, GatewayError> {
        for (name, value) in [
            ("proposal_id", self.proposal_id.as_str()),
            ("agent_id", self.agent_id.as_str()),
            ("source", self.source.as_str()),
            ("actor_id", self.actor_id.as_str()),
            ("input_digest", self.input_digest.as_str()),
            ("artifact_digest", self.artifact_digest.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(GatewayError::MissingField(name));
            }
        }
        if self.payload.trim().is_empty() {
            return Err(GatewayError::EmptyPayload);
        }
        if let (Some(knowledge), Some(exposure)) =
            (&self.knowledge_cutoff, &self.exposure_cutoff)
        {
            if exposure < knowledge {
                return Err(GatewayError::ExposureBeforeKnowledgeCutoff);
            }
        }
        let mut hasher = Sha256::new();
        hasher.update(self.payload.as_bytes());
        let payload_digest = format!("sha256:{:x}", hasher.finalize());

        Ok(NormalizedProposal {
            proposal_id: self.proposal_id.clone(),
            agent_id: self.agent_id.clone(),
            source: self.source.clone(),
            actor_id: self.actor_id.clone(),
            payload_digest,
            input_digest: self.input_digest.clone(),
            artifact_digest: self.artifact_digest.clone(),
            model_lineage: self.model_lineage.clone(),
            knowledge_cutoff: self.knowledge_cutoff.clone(),
            exposure_cutoff: self.exposure_cutoff.clone(),
        })
    }
}

/// The gateway deliberately has no API that emits Observation, Replication, or
/// OfficialCriterionEvidence. A normalized proposal is eligible only for
/// downstream planning/model reasoning.
pub fn disposition(proposal: &AgentProposal) -> Result<ProposalDisposition, GatewayError> {
    proposal.normalize().map(|_| ProposalDisposition::AcceptedForPlanning)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal() -> AgentProposal {
        AgentProposal {
            proposal_id: "p1".into(),
            agent_id: "agent:future".into(),
            source: "external-agent".into(),
            actor_id: "actor:a".into(),
            payload: "hypothesis".into(),
            input_digest: "sha256:input".into(),
            artifact_digest: "sha256:artifact".into(),
            model_lineage: Some("lineage:model".into()),
            knowledge_cutoff: Some("2026-09-28T10:00:00Z".into()),
            exposure_cutoff: Some("2026-09-28T10:01:00Z".into()),
        }
    }

    #[test]
    fn valid_proposal_is_planning_only() {
        assert_eq!(disposition(&proposal()).unwrap(), ProposalDisposition::AcceptedForPlanning);
    }

    #[test]
    fn payload_digest_is_deterministic() {
        let normalized = proposal().normalize().unwrap();
        assert_eq!(normalized.payload_digest, proposal().normalize().unwrap().payload_digest);
    }

    #[test]
    fn missing_provenance_is_rejected() {
        let mut p = proposal();
        p.input_digest.clear();
        assert_eq!(p.normalize(), Err(GatewayError::MissingField("input_digest")));
    }

    #[test]
    fn exposure_before_knowledge_is_rejected() {
        let mut p = proposal();
        p.exposure_cutoff = Some("2026-09-28T09:00:00Z".into());
        assert_eq!(p.normalize(), Err(GatewayError::ExposureBeforeKnowledgeCutoff));
    }

    #[test]
    fn empty_payload_is_rejected() {
        let mut p = proposal();
        p.payload.clear();
        assert_eq!(p.normalize(), Err(GatewayError::EmptyPayload));
    }

    #[test]
    fn payload_changes_digest() {
        let a = proposal().normalize().unwrap();
        let mut p = proposal();
        p.payload = "different hypothesis".into();
        let b = p.normalize().unwrap();
        assert_ne!(a.payload_digest, b.payload_digest);
    }
}
