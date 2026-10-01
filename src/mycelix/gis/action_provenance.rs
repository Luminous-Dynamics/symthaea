//! Provenance-aware action gating for GIS.
//!
//! This module separates historical authorization from current authorization.
//! A frame revision can require fresh support for future high-risk execution
//! without rewriting the historical record of an action that already happened.

use super::ignorance_types::{EpistemicFrameImpact, EpistemicFrameRevision};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionRisk {
    Informational,
    Low,
    High,
    Critical,
}

impl ActionRisk {
    pub const fn requires_current_support(self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionDependencyKind {
    ConclusionSupport,
    CausalBasis,
    OntologyBasis,
    EvidenceBasis,
    AssumptionBasis,
}

impl ActionDependencyKind {
    pub const fn affects(self, impact: EpistemicFrameImpact) -> bool {
        match self {
            Self::ConclusionSupport => impact.evidence_boundary || impact.ontology || impact.causal_model,
            Self::CausalBasis => impact.causal_model,
            Self::OntologyBasis => impact.ontology,
            Self::EvidenceBasis => impact.evidence_boundary,
            Self::AssumptionBasis => impact.evidence_boundary
                || impact.ontology
                || impact.causal_model
                || impact.exclusions
                || impact.blind_spots,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    Ready,
    RequiresReevaluation,
    Deferred,
    Executed,
    Superseded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDependency {
    pub conclusion_id: String,
    pub kind: ActionDependencyKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionDecisionWitness {
    pub frame: String,
    pub conclusions: Vec<String>,
    pub evidence: Vec<String>,
    pub policy: String,
    pub decision: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionReevaluationWitness {
    pub prior_frame: String,
    pub revised_frame: String,
    pub affected_conclusions: Vec<String>,
    pub reasons: Vec<ActionDependencyKind>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicAction {
    pub id: String,
    pub description: String,
    pub risk: ActionRisk,
    pub dependencies: Vec<ActionDependency>,
    pub status: ActionStatus,
    pub historical_decisions: Vec<ActionDecisionWitness>,
    pub reevaluation: Option<ActionReevaluationWitness>,
}

impl EpistemicAction {
    pub fn new(id: impl Into<String>, description: impl Into<String>, risk: ActionRisk) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            risk,
            dependencies: Vec::new(),
            status: ActionStatus::Ready,
            historical_decisions: Vec::new(),
            reevaluation: None,
        }
    }

    pub fn require_reevaluation(
        &mut self,
        revision: &EpistemicFrameRevision,
        affected_conclusions: Vec<String>,
        reasons: Vec<ActionDependencyKind>,
    ) {
        if matches!(self.status, ActionStatus::Executed | ActionStatus::Superseded) {
            return;
        }

        self.status = ActionStatus::RequiresReevaluation;
        self.reevaluation = Some(ActionReevaluationWitness {
            prior_frame: revision.prior_frame.clone(),
            revised_frame: revision.revised_frame.clone(),
            affected_conclusions,
            reasons,
            reason: "epistemic prerequisite changed; current support required by policy".into(),
        });
    }

    pub fn record_decision(&mut self, witness: ActionDecisionWitness) {
        self.historical_decisions.push(witness);
        self.status = ActionStatus::Executed;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ActionDependencyGraph {
    pub actions: Vec<EpistemicAction>,
}

impl ActionDependencyGraph {
    pub fn add(&mut self, action: EpistemicAction) {
        self.actions.push(action);
    }

    pub fn reevaluate_from_frame_revision(
        &mut self,
        revision: &EpistemicFrameRevision,
        affected_conclusions: &[String],
    ) -> Vec<String> {
        let affected: std::collections::HashSet<&str> =
            affected_conclusions.iter().map(String::as_str).collect();
        let mut gated = Vec::new();

        for action in &mut self.actions {
            if !action.risk.requires_current_support() {
                continue;
            }

            let impacted: Vec<_> = action
                .dependencies
                .iter()
                .filter(|dependency| affected.contains(dependency.conclusion_id.as_str()) && dependency.kind.affects(revision.impact))
                .collect();

            if impacted.is_empty() {
                continue;
            }

            action.require_reevaluation(
                revision,
                impacted.iter().map(|d| d.conclusion_id.clone()).collect(),
                impacted.iter().map(|d| d.kind).collect(),
            );

            if action.status == ActionStatus::RequiresReevaluation {
                gated.push(action.id.clone());
            }
        }

        gated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_risk_action_is_gated_but_executed_history_is_preserved() {
        let revision = EpistemicFrameRevision {
            prior_frame: "frame@1".into(),
            revised_frame: "frame@2".into(),
            trigger: "causal model changed".into(),
            newly_represented: None,
            scope_change: "causal model".into(),
            affected_conclusions: vec!["c1".into()],
            impact: EpistemicFrameImpact {
                evidence_boundary: false,
                ontology: false,
                causal_model: true,
                exclusions: false,
                blind_spots: false,
            },
        };

        let mut action = EpistemicAction::new("a1", "intervention", ActionRisk::High);
        action.dependencies.push(ActionDependency {
            conclusion_id: "c1".into(),
            kind: ActionDependencyKind::CausalBasis,
        });

        let mut graph = ActionDependencyGraph::default();
        graph.add(action);

        assert_eq!(
            graph.reevaluate_from_frame_revision(&revision, &["c1".into()]),
            vec!["a1"]
        );
        assert_eq!(graph.actions[0].status, ActionStatus::RequiresReevaluation);
        assert_eq!(
            graph.actions[0].reevaluation.as_ref().unwrap().revised_frame,
            "frame@2"
        );
    }
}
