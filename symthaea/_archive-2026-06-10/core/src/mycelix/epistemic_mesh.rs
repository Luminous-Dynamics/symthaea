// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Epistemic Mesh — Collective Knowledge Gap Detection
//!
//! Network-level epistemic awareness: know what the community doesn't know.
//!
//! Each agent publishes an [`EpistemicSummary`] derived from their KosmicSong's
//! ignorance detection. The [`EpistemicMesh`] aggregates these to find:
//!
//! - **Collective blind spots**: domains where >50% of agents have high uncertainty
//! - **Proposal escalation**: blind-spot domains require higher consciousness tier
//! - **Expertise boost**: domain experts get boosted weight in relevant proposals
//!
//! ## Integration
//!
//! The GovernanceManager optionally holds an EpistemicMesh snapshot and uses it
//! to flag exploration when blind spots exist and to escalate proposals that
//! touch unknown territory.

use super::gis::IgnoranceType;

/// Escalation tier for proposals touching collective blind spots.
///
/// Maps to `CivicTier` in mycelix-bridge-common when crossing
/// the bridge boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EscalationTier {
    /// Voting rights with extra scrutiny (severity >= 0.5).
    Citizen,
    /// Constitutional actions required (severity >= 0.6).
    Steward,
    /// Emergency powers required (severity >= 0.8).
    Guardian,
}

/// Summary of a single agent's epistemic state.
#[derive(Debug, Clone)]
pub struct EpistemicSummary {
    /// Agent identifier.
    pub agent_id: String,
    /// Dominant type of ignorance for this agent.
    pub dominant_ignorance: IgnoranceType,
    /// Domain expertise: (domain_name, confidence 0.0–1.0).
    pub domain_expertise: Vec<(String, f64)>,
    /// Domains where the agent has high uncertainty (confidence < 0.3).
    pub blind_spots: Vec<String>,
}

/// A domain where most agents are uncertain.
#[derive(Debug, Clone)]
pub struct CollectiveBlindSpot {
    /// The domain name.
    pub domain: String,
    /// Number of agents who are uncertain in this domain.
    pub agents_uncertain: usize,
    /// Total agents contributing summaries.
    pub total_agents: usize,
    /// Fraction of agents uncertain (agents_uncertain / total_agents).
    pub severity: f64,
}

/// Aggregated epistemic state of the network.
pub struct EpistemicMesh {
    summaries: Vec<EpistemicSummary>,
    blind_spots: Vec<CollectiveBlindSpot>,
}

/// Minimum fraction of agents uncertain for a blind spot to be flagged.
const BLIND_SPOT_THRESHOLD: f64 = 0.5;

/// Minimum confidence threshold — below this, an agent counts as uncertain.
const UNCERTAINTY_THRESHOLD: f64 = 0.3;

impl EpistemicMesh {
    /// Create a new mesh from agent summaries. Immediately computes blind spots.
    pub fn new(summaries: Vec<EpistemicSummary>) -> Self {
        let mut mesh = Self {
            summaries,
            blind_spots: Vec::new(),
        };
        mesh.aggregate();
        mesh
    }

    /// Aggregate summaries to identify collective blind spots.
    ///
    /// A blind spot is a domain where >[`BLIND_SPOT_THRESHOLD`] fraction of
    /// agents report it as a blind spot.
    pub fn aggregate(&mut self) {
        use std::collections::HashMap;

        if self.summaries.is_empty() {
            self.blind_spots.clear();
            return;
        }

        // Count uncertain agents per domain
        let mut domain_uncertain: HashMap<String, usize> = HashMap::new();
        let total = self.summaries.len();

        for summary in &self.summaries {
            for spot in &summary.blind_spots {
                *domain_uncertain.entry(spot.clone()).or_insert(0) += 1;
            }
        }

        // Flag domains above threshold
        self.blind_spots = domain_uncertain
            .into_iter()
            .filter_map(|(domain, count)| {
                let severity = count as f64 / total as f64;
                if severity >= BLIND_SPOT_THRESHOLD {
                    Some(CollectiveBlindSpot {
                        domain,
                        agents_uncertain: count,
                        total_agents: total,
                        severity,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by severity descending for deterministic ordering
        self.blind_spots.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get all detected blind spots.
    pub fn blind_spots(&self) -> &[CollectiveBlindSpot] {
        &self.blind_spots
    }

    /// Check if any of the given domains touch a collective blind spot.
    /// If so, return the minimum consciousness tier required for the proposal.
    ///
    /// Escalation rules:
    /// - severity >= 0.8 → Guardian (emergency-level caution)
    /// - severity >= 0.6 → Steward (constitutional actions)
    /// - severity >= 0.5 → Citizen (voting, with extra scrutiny)
    pub fn proposal_escalation_required(
        &self,
        proposal_domains: &[String],
    ) -> Option<EscalationTier> {
        let mut max_severity = 0.0f64;
        for domain in proposal_domains {
            for spot in &self.blind_spots {
                if spot.domain == *domain {
                    max_severity = max_severity.max(spot.severity);
                }
            }
        }

        if max_severity >= 0.8 {
            Some(EscalationTier::Guardian)
        } else if max_severity >= 0.6 {
            Some(EscalationTier::Steward)
        } else if max_severity >= BLIND_SPOT_THRESHOLD {
            Some(EscalationTier::Citizen)
        } else {
            None
        }
    }

    /// Compute an expertise boost for an agent in the given domains.
    ///
    /// Returns a multiplier (1.0 = no boost, up to 2.0 for high expertise).
    /// Agents with domain expertise on blind-spot domains get boosted.
    pub fn expertise_boost(&self, summary: &EpistemicSummary, domains: &[String]) -> f64 {
        if self.blind_spots.is_empty() || domains.is_empty() {
            return 1.0;
        }

        let mut boost = 1.0;
        for domain in domains {
            // Is this a blind spot domain?
            let is_blind_spot = self.blind_spots.iter().any(|bs| bs.domain == *domain);
            if !is_blind_spot {
                continue;
            }
            // Does this agent have expertise here?
            if let Some((_, confidence)) =
                summary.domain_expertise.iter().find(|(d, _)| d == domain)
            {
                if *confidence > UNCERTAINTY_THRESHOLD {
                    // Expert in a blind-spot domain → boost proportional to confidence
                    boost += confidence * 0.5; // max +0.5 per domain
                }
            }
        }

        boost.min(2.0) // cap at 2x
    }

    /// Number of summaries in the mesh.
    pub fn agent_count(&self) -> usize {
        self.summaries.len()
    }

    /// Check if any blind spots exist.
    pub fn has_blind_spots(&self) -> bool {
        !self.blind_spots.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn summary_with_spots(id: &str, spots: &[&str]) -> EpistemicSummary {
        EpistemicSummary {
            agent_id: id.to_string(),
            dominant_ignorance: IgnoranceType::KnownUnknown,
            domain_expertise: vec![],
            blind_spots: spots.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn expert_summary(id: &str, spots: &[&str], expertise: &[(&str, f64)]) -> EpistemicSummary {
        EpistemicSummary {
            agent_id: id.to_string(),
            dominant_ignorance: IgnoranceType::Known,
            domain_expertise: expertise.iter().map(|(d, c)| (d.to_string(), *c)).collect(),
            blind_spots: spots.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_blind_spot_detection() {
        // 3 out of 5 agents uncertain about "climate" → blind spot
        let summaries = vec![
            summary_with_spots("a1", &["climate", "energy"]),
            summary_with_spots("a2", &["climate"]),
            summary_with_spots("a3", &["climate", "finance"]),
            summary_with_spots("a4", &["finance"]),
            summary_with_spots("a5", &["energy"]),
        ];
        let mesh = EpistemicMesh::new(summaries);

        assert!(mesh.has_blind_spots());
        let climate_spot = mesh.blind_spots().iter().find(|bs| bs.domain == "climate");
        assert!(climate_spot.is_some(), "Climate should be a blind spot");
        assert_eq!(climate_spot.unwrap().agents_uncertain, 3);
        assert!((climate_spot.unwrap().severity - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_no_blind_spots_below_threshold() {
        // Only 1 out of 5 uncertain → no blind spot
        let summaries = vec![
            summary_with_spots("a1", &["climate"]),
            summary_with_spots("a2", &[]),
            summary_with_spots("a3", &[]),
            summary_with_spots("a4", &[]),
            summary_with_spots("a5", &[]),
        ];
        let mesh = EpistemicMesh::new(summaries);
        assert!(!mesh.has_blind_spots());
    }

    #[test]
    fn test_proposal_escalation() {
        // 4 out of 5 uncertain about "biotech" → severity 0.8 → Guardian
        let summaries = vec![
            summary_with_spots("a1", &["biotech"]),
            summary_with_spots("a2", &["biotech"]),
            summary_with_spots("a3", &["biotech"]),
            summary_with_spots("a4", &["biotech"]),
            summary_with_spots("a5", &[]),
        ];
        let mesh = EpistemicMesh::new(summaries);

        let tier = mesh.proposal_escalation_required(&["biotech".to_string()]);
        assert_eq!(tier, Some(EscalationTier::Guardian));
    }

    #[test]
    fn test_proposal_no_escalation() {
        let summaries = vec![summary_with_spots("a1", &[]), summary_with_spots("a2", &[])];
        let mesh = EpistemicMesh::new(summaries);

        let tier = mesh.proposal_escalation_required(&["anything".to_string()]);
        assert_eq!(tier, None);
    }

    #[test]
    fn test_expertise_boost() {
        // "climate" is a blind spot; agent a1 has expertise
        let summaries = vec![
            summary_with_spots("a1", &[]), // expert, no spots
            summary_with_spots("a2", &["climate"]),
            summary_with_spots("a3", &["climate"]),
            summary_with_spots("a4", &["climate"]),
        ];
        let mesh = EpistemicMesh::new(summaries);

        let expert = expert_summary("a1", &[], &[("climate", 0.9)]);
        let boost = mesh.expertise_boost(&expert, &["climate".to_string()]);
        assert!(boost > 1.0, "Domain expert should get boost: {}", boost);
        assert!(boost <= 2.0, "Boost should be capped at 2.0");
    }

    #[test]
    fn test_expertise_no_boost_without_blind_spot() {
        let summaries = vec![summary_with_spots("a1", &[]), summary_with_spots("a2", &[])];
        let mesh = EpistemicMesh::new(summaries);

        let expert = expert_summary("a1", &[], &[("climate", 0.9)]);
        let boost = mesh.expertise_boost(&expert, &["climate".to_string()]);
        assert!((boost - 1.0).abs() < 1e-6, "No blind spot → no boost");
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = EpistemicMesh::new(vec![]);
        assert_eq!(mesh.agent_count(), 0);
        assert!(!mesh.has_blind_spots());
    }
}
