//! Multi-agent model council for scientific discovery.
//!
//! The council organizes competing proposals and diagnoses independence. It
//! never votes on scientific truth and never promotes proposals to evidence.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CouncilProposal {
    pub proposal_id: String,
    pub agent_id: String,
    pub model_lineage: String,
    pub mechanism_id: String,
    pub payload_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageCluster {
    pub lineage: String,
    pub proposal_ids: Vec<String>,
    pub agent_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CouncilAssessment {
    pub proposals: Vec<String>,
    pub lineage_clusters: Vec<LineageCluster>,
    pub unique_lineages: usize,
    pub unique_mechanisms: usize,
    pub independent_agent_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CouncilError {
    EmptyField(&'static str),
    DuplicateProposal(String),
}

pub fn assess(proposals: &[CouncilProposal]) -> Result<CouncilAssessment, CouncilError> {
    let mut ids = BTreeSet::new();
    let mut by_lineage: BTreeMap<String, LineageCluster> = BTreeMap::new();
    let mut mechanisms = BTreeSet::new();
    let mut agents = BTreeSet::new();

    for p in proposals {
        for (name, value) in [
            ("proposal_id", p.proposal_id.as_str()),
            ("agent_id", p.agent_id.as_str()),
            ("model_lineage", p.model_lineage.as_str()),
            ("mechanism_id", p.mechanism_id.as_str()),
            ("payload_digest", p.payload_digest.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CouncilError::EmptyField(name));
            }
        }
        if !ids.insert(p.proposal_id.clone()) {
            return Err(CouncilError::DuplicateProposal(p.proposal_id.clone()));
        }

        let cluster = by_lineage.entry(p.model_lineage.clone()).or_insert_with(|| LineageCluster {
            lineage: p.model_lineage.clone(),
            proposal_ids: Vec::new(),
            agent_ids: Vec::new(),
        });
        cluster.proposal_ids.push(p.proposal_id.clone());
        if !cluster.agent_ids.contains(&p.agent_id) {
            cluster.agent_ids.push(p.agent_id.clone());
        }
        mechanisms.insert(p.mechanism_id.clone());
        agents.insert(p.agent_id.clone());
    }

    let clusters = by_lineage.into_values().collect::<Vec<_>>();
    Ok(CouncilAssessment {
        proposals: ids.into_iter().collect(),
        unique_lineages: clusters.len(),
        unique_mechanisms: mechanisms.len(),
        independent_agent_count: agents.len(),
        lineage_clusters: clusters,
    })
}

/// Returns whether two proposals have distinct model ancestry. Distinct agents
/// alone are insufficient to establish independence.
pub fn distinct_model_ancestry(a: &CouncilProposal, b: &CouncilProposal) -> bool {
    a.model_lineage != b.model_lineage
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(id: &str, agent: &str, lineage: &str, mechanism: &str) -> CouncilProposal {
        CouncilProposal {
            proposal_id: id.into(),
            agent_id: agent.into(),
            model_lineage: lineage.into(),
            mechanism_id: mechanism.into(),
            payload_digest: format!("sha256:{id}"),
        }
    }

    #[test]
    fn groups_shared_ancestry() {
        let a = p("a", "agent-a", "lineage-x", "mechanism-1");
        let b = p("b", "agent-b", "lineage-x", "mechanism-1");
        let c = p("c", "agent-c", "lineage-y", "mechanism-2");
        let assessment = assess(&[a, b, c]).unwrap();
        assert_eq!(assessment.unique_lineages, 2);
        assert_eq!(assessment.unique_mechanisms, 2);
        assert_eq!(assessment.independent_agent_count, 3);
        assert_eq!(assessment.lineage_clusters[0].proposal_ids.len(), 2);
    }

    #[test]
    fn distinct_agents_do_not_imply_distinct_ancestry() {
        let a = p("a", "agent-a", "lineage-x", "mechanism-1");
        let b = p("b", "agent-b", "lineage-x", "mechanism-2");
        assert!(!distinct_model_ancestry(&a, &b));
    }

    #[test]
    fn distinct_lineages_are_explicit() {
        let a = p("a", "agent-a", "lineage-x", "mechanism-1");
        let b = p("b", "agent-b", "lineage-y", "mechanism-1");
        assert!(distinct_model_ancestry(&a, &b));
    }

    #[test]
    fn duplicate_identity_rejected() {
        let a = p("same", "agent-a", "lineage-x", "mechanism-1");
        let b = p("same", "agent-b", "lineage-y", "mechanism-2");
        assert_eq!(assess(&[a, b]), Err(CouncilError::DuplicateProposal("same".into())));
    }
}
