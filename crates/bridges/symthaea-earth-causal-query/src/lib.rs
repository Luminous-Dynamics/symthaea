//! Counterfactual identification adapter for evidence-backed Earth causal workspaces.
//!
//! This crate converts the reviewed structural graph in `symthaea-earth-causal`
//! into the identification DAG used by Symthaea's counterfactual reasoner.
//!
//! A critical boundary is preserved: the current counterfactual reasoner uses
//! `CausalEstimand.effect = 0.0` as a placeholder when it has identified a
//! symbolic estimand but has not estimated an effect from data. This adapter
//! therefore deliberately does **not** expose that field as an estimated Earth
//! effect. It exposes identification status, method, assumptions, adjustment
//! evidence, and estimand description only.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_causal_reasoning::counterfactual::{
    CausalAssumption, CausalDAG as IdentificationDAG, CausalQuery, CausalQueryOutcome,
    CounterfactualReasoner, IdentificationMethod, UnidentifiedReason,
};
use symthaea_earth_causal::EarthCausalWorkspace;

pub type Result<T> = std::result::Result<T, QueryBridgeError>;

#[derive(Debug, Clone, PartialEq)]
pub enum QueryBridgeError {
    EmptyEvidenceId(&'static str),
    MissingEvidenceId(String),
    UnmappedNode(usize),
    NonFiniteScore { field: &'static str, value: f64 },
}

impl Display for QueryBridgeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyEvidenceId(field) => write!(f, "{field} must not be empty"),
            Self::MissingEvidenceId(id) => write!(f, "evidence id {id} is not present in query view"),
            Self::UnmappedNode(node) => write!(f, "causal node {node} has no evidence binding"),
            Self::NonFiniteScore { field, value } => {
                write!(f, "{field} must be finite, got {value}")
            }
        }
    }
}

impl Error for QueryBridgeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalEffectRequest {
    pub treatment_evidence_id: String,
    pub outcome_evidence_id: String,
    pub conditioning_evidence_ids: Vec<String>,
}

impl CausalEffectRequest {
    pub fn new(
        treatment_evidence_id: impl Into<String>,
        outcome_evidence_id: impl Into<String>,
        conditioning_evidence_ids: Vec<String>,
    ) -> Result<Self> {
        let treatment_evidence_id = treatment_evidence_id.into();
        let outcome_evidence_id = outcome_evidence_id.into();
        if treatment_evidence_id.trim().is_empty() {
            return Err(QueryBridgeError::EmptyEvidenceId("treatment evidence id"));
        }
        if outcome_evidence_id.trim().is_empty() {
            return Err(QueryBridgeError::EmptyEvidenceId("outcome evidence id"));
        }
        if conditioning_evidence_ids
            .iter()
            .any(|id| id.trim().is_empty())
        {
            return Err(QueryBridgeError::EmptyEvidenceId(
                "conditioning evidence id",
            ));
        }
        Ok(Self {
            treatment_evidence_id,
            outcome_evidence_id,
            conditioning_evidence_ids,
        })
    }
}

/// Earth-facing query outcome that mirrors the causal reasoner's epistemic
/// states without exposing its placeholder effect magnitude as an estimate.
#[derive(Debug, Clone)]
pub enum EarthCausalQueryOutcome {
    Identified {
        estimand_description: String,
        method: IdentificationMethod,
        identification_confidence: f64,
        adjustment_evidence_ids: Vec<String>,
    },
    Unidentified {
        reason: UnidentifiedReason,
        missing: Vec<String>,
        suggestions: Vec<String>,
    },
    AssumptionRequired {
        assumption: CausalAssumption,
        estimand_description: String,
        adjustment_evidence_ids: Vec<String>,
        plausibility: f64,
    },
}

impl EarthCausalQueryOutcome {
    pub const fn is_identified(&self) -> bool {
        matches!(self, Self::Identified { .. })
    }
}

/// Read-only identification view of one reviewed Earth causal workspace.
#[derive(Debug, Clone)]
pub struct EarthCausalQueryView {
    dag: IdentificationDAG,
    evidence_id_by_node: Vec<String>,
    node_by_evidence_id: HashMap<String, usize>,
}

impl EarthCausalQueryView {
    pub fn from_workspace(workspace: &EarthCausalWorkspace) -> Result<Self> {
        let dag = workspace.dag();
        let mut evidence_id_by_node = vec![None::<String>; dag.nodes.len()];
        let mut node_by_evidence_id = HashMap::new();

        for binding in workspace.bindings() {
            let Some(slot) = evidence_id_by_node.get_mut(binding.node_id) else {
                return Err(QueryBridgeError::UnmappedNode(binding.node_id));
            };
            *slot = Some(binding.evidence_id.clone());
            node_by_evidence_id.insert(binding.evidence_id.clone(), binding.node_id);
        }

        let evidence_id_by_node = evidence_id_by_node
            .into_iter()
            .enumerate()
            .map(|(node, id)| id.ok_or(QueryBridgeError::UnmappedNode(node)))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            dag: IdentificationDAG::new(
                dag.nodes.iter().map(|node| node.name.clone()).collect(),
                dag.edges.clone(),
            ),
            evidence_id_by_node,
            node_by_evidence_id,
        })
    }

    pub fn dag(&self) -> &IdentificationDAG {
        &self.dag
    }

    pub fn evidence_id_for_node(&self, node: usize) -> Option<&str> {
        self.evidence_id_by_node.get(node).map(String::as_str)
    }

    pub fn query(&self, request: &CausalEffectRequest) -> Result<EarthCausalQueryOutcome> {
        let treatment = self.require_node(&request.treatment_evidence_id)?;
        let outcome = self.require_node(&request.outcome_evidence_id)?;
        let conditioning = request
            .conditioning_evidence_ids
            .iter()
            .map(|id| self.require_node(id))
            .collect::<Result<Vec<_>>>()?;

        let raw = CounterfactualReasoner::new().query(
            &self.dag,
            &CausalQuery {
                treatment,
                outcome,
                conditioning,
            },
        );
        self.map_outcome(raw)
    }

    fn require_node(&self, evidence_id: &str) -> Result<usize> {
        self.node_by_evidence_id
            .get(evidence_id)
            .copied()
            .ok_or_else(|| QueryBridgeError::MissingEvidenceId(evidence_id.to_string()))
    }

    fn adjustment_ids(&self, nodes: &[usize]) -> Result<Vec<String>> {
        nodes
            .iter()
            .map(|&node| {
                self.evidence_id_by_node
                    .get(node)
                    .cloned()
                    .ok_or(QueryBridgeError::UnmappedNode(node))
            })
            .collect()
    }

    fn map_outcome(&self, outcome: CausalQueryOutcome) -> Result<EarthCausalQueryOutcome> {
        match outcome {
            CausalQueryOutcome::Identified {
                estimand,
                method,
                confidence,
            } => {
                if !confidence.is_finite() {
                    return Err(QueryBridgeError::NonFiniteScore {
                        field: "identification confidence",
                        value: confidence,
                    });
                }
                Ok(EarthCausalQueryOutcome::Identified {
                    estimand_description: estimand.description,
                    method,
                    identification_confidence: confidence,
                    adjustment_evidence_ids: self.adjustment_ids(&estimand.adjustment_set)?,
                })
            }
            CausalQueryOutcome::Unidentified {
                reason,
                missing,
                suggestions,
            } => Ok(EarthCausalQueryOutcome::Unidentified {
                reason,
                missing,
                suggestions,
            }),
            CausalQueryOutcome::AssumptionRequired {
                assumption,
                estimand_if_assumed,
                plausibility,
            } => {
                if !plausibility.is_finite() {
                    return Err(QueryBridgeError::NonFiniteScore {
                        field: "assumption plausibility",
                        value: plausibility,
                    });
                }
                Ok(EarthCausalQueryOutcome::AssumptionRequired {
                    assumption,
                    estimand_description: estimand_if_assumed.description,
                    adjustment_evidence_ids: self
                        .adjustment_ids(&estimand_if_assumed.adjustment_set)?,
                    plausibility,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_earth_causal::{StructuralEdgeBasis, StructuralEdgeClaim};
    use symthaea_earth_observation::{EvidenceRef, EvidenceStage};

    fn evidence(id: &str) -> EvidenceRef {
        EvidenceRef::new(id, EvidenceStage::Measurement).unwrap()
    }

    fn two_node_workspace(with_edge: bool) -> EarthCausalWorkspace {
        let mut workspace = EarthCausalWorkspace::new();
        workspace.register_evidence(&evidence("a"), "A", true).unwrap();
        workspace.register_evidence(&evidence("b"), "B", true).unwrap();

        if with_edge {
            workspace
                .assert_structural_edge(StructuralEdgeClaim {
                    parent_evidence_id: "a".into(),
                    child_evidence_id: "b".into(),
                    basis: StructuralEdgeBasis::DomainAssumption {
                        assumption: "fixture A directly influences B".into(),
                    },
                    supporting_evidence: vec![],
                    assumptions: vec![],
                })
                .unwrap();
        }
        workspace
    }

    #[test]
    fn structural_graph_converts_without_reindexing_evidence() {
        let workspace = two_node_workspace(true);
        let view = EarthCausalQueryView::from_workspace(&workspace).unwrap();

        assert_eq!(view.evidence_id_for_node(0), Some("a"));
        assert_eq!(view.evidence_id_for_node(1), Some("b"));
        assert_eq!(view.dag().edges, vec![(0, 1)]);
    }

    #[test]
    fn connected_structural_query_is_identifiable_without_exposing_effect_placeholder() {
        let workspace = two_node_workspace(true);
        let view = EarthCausalQueryView::from_workspace(&workspace).unwrap();
        let request = CausalEffectRequest::new("a", "b", vec![]).unwrap();
        let outcome = view.query(&request).unwrap();

        assert!(outcome.is_identified());
        match outcome {
            EarthCausalQueryOutcome::Identified {
                estimand_description,
                identification_confidence,
                ..
            } => {
                assert!(!estimand_description.is_empty());
                assert!(identification_confidence.is_finite());
            }
            other => panic!("expected identified outcome, got {other:?}"),
        }
    }

    #[test]
    fn disconnected_query_remains_unidentified() {
        let workspace = two_node_workspace(false);
        let view = EarthCausalQueryView::from_workspace(&workspace).unwrap();
        let request = CausalEffectRequest::new("a", "b", vec![]).unwrap();

        assert!(matches!(
            view.query(&request).unwrap(),
            EarthCausalQueryOutcome::Unidentified {
                reason: UnidentifiedReason::NotConnected,
                ..
            }
        ));
    }

    #[test]
    fn unknown_evidence_id_is_rejected_before_reasoning() {
        let workspace = two_node_workspace(true);
        let view = EarthCausalQueryView::from_workspace(&workspace).unwrap();
        let request = CausalEffectRequest::new("missing", "b", vec![]).unwrap();

        assert_eq!(
            view.query(&request).unwrap_err(),
            QueryBridgeError::MissingEvidenceId("missing".into())
        );
    }
}
