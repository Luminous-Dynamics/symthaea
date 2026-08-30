//! Evidence-preserving bridge from Earth observation into Symthaea causal reasoning.
//!
//! The load-bearing rule is that observational association is not a causal edge.
//! Earth evidence may become variables in a causal workspace, and associations
//! may be retained for later analysis, but a directed edge requires an explicit
//! structural claim with a declared basis. This crate never learns or invents
//! direct causal structure merely because two observations co-vary.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use symthaea_causal_reasoning::causal_calculus::CausalDAG;
use symthaea_earth_observation::{
    DerivedFeature, EvidenceRef, EvidenceStage, Hypothesis, ObservationEvidence,
};

pub type Result<T> = std::result::Result<T, BridgeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    EmptyField(&'static str),
    DuplicateEvidenceId(String),
    MissingEvidenceId(String),
    SelfEdge(String),
    MissingStructuralSupport,
    MissingDomainAssumption,
    StructuralCycle { parent: String, child: String },
}

impl Display for BridgeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "{field} must not be empty"),
            Self::DuplicateEvidenceId(id) => write!(f, "evidence id {id} is already registered"),
            Self::MissingEvidenceId(id) => write!(f, "evidence id {id} is not registered"),
            Self::SelfEdge(id) => write!(f, "causal self-edge is not allowed for {id}"),
            Self::MissingStructuralSupport => write!(
                f,
                "this structural-edge basis requires explicit supporting evidence"
            ),
            Self::MissingDomainAssumption => {
                write!(f, "domain-assumption edges require an explicit assumption")
            }
            Self::StructuralCycle { parent, child } => write!(
                f,
                "adding structural edge {parent} -> {child} would create a causal cycle"
            ),
        }
    }
}

impl Error for BridgeError {}

fn non_empty(value: &str, field: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(BridgeError::EmptyField(field));
    }
    Ok(())
}

/// How an Earth-evidence object is represented in the causal workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceNodeBinding {
    pub evidence_id: String,
    pub evidence_stage: EvidenceStage,
    pub node_id: usize,
    pub observed: bool,
}

/// An association is deliberately not a causal assertion.
///
/// This can carry a correlation, similarity, lag score, HDC similarity, or
/// another descriptive relation while leaving the causal DAG untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceAssociation {
    pub left_evidence_id: String,
    pub right_evidence_id: String,
    pub relation: String,
    pub score: Option<f64>,
    pub support: Vec<EvidenceRef>,
}

impl EvidenceAssociation {
    pub fn new(
        left_evidence_id: impl Into<String>,
        right_evidence_id: impl Into<String>,
        relation: impl Into<String>,
        score: Option<f64>,
        support: Vec<EvidenceRef>,
    ) -> Result<Self> {
        let left_evidence_id = left_evidence_id.into();
        let right_evidence_id = right_evidence_id.into();
        let relation = relation.into();
        non_empty(&left_evidence_id, "left evidence id")?;
        non_empty(&right_evidence_id, "right evidence id")?;
        non_empty(&relation, "association relation")?;
        if let Some(score) = score {
            if !score.is_finite() {
                return Err(BridgeError::EmptyField("finite association score"));
            }
        }
        Ok(Self {
            left_evidence_id,
            right_evidence_id,
            relation,
            score,
            support,
        })
    }
}

/// Explicit provenance for a direct structural edge in a causal model.
///
/// None of these variants means "we observed a correlation". In particular,
/// an identified total causal effect is not automatically evidence that a
/// *direct* edge exists; direct structure still needs a declared structural
/// basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralEdgeBasis {
    /// A structural assumption supplied by a named domain model or expert model.
    DomainAssumption { assumption: String },
    /// A controlled intervention provides evidence for the proposed direct link.
    ControlledIntervention { intervention_id: String },
    /// A separately validated structural model supplies the direct relation.
    ExternalValidatedModel { model_id: String, version: String },
}

/// A reviewed request to add one directed edge to the causal DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralEdgeClaim {
    pub parent_evidence_id: String,
    pub child_evidence_id: String,
    pub basis: StructuralEdgeBasis,
    pub supporting_evidence: Vec<EvidenceRef>,
    pub assumptions: Vec<String>,
}

impl StructuralEdgeClaim {
    pub fn validate(&self) -> Result<()> {
        non_empty(&self.parent_evidence_id, "parent evidence id")?;
        non_empty(&self.child_evidence_id, "child evidence id")?;
        if self.parent_evidence_id == self.child_evidence_id {
            return Err(BridgeError::SelfEdge(self.parent_evidence_id.clone()));
        }

        match &self.basis {
            StructuralEdgeBasis::DomainAssumption { assumption } => {
                non_empty(assumption, "domain assumption")?;
                if self.assumptions.is_empty() {
                    return Err(BridgeError::MissingDomainAssumption);
                }
            }
            StructuralEdgeBasis::ControlledIntervention { intervention_id } => {
                non_empty(intervention_id, "intervention id")?;
                if self.supporting_evidence.is_empty() {
                    return Err(BridgeError::MissingStructuralSupport);
                }
            }
            StructuralEdgeBasis::ExternalValidatedModel { model_id, version } => {
                non_empty(model_id, "validated model id")?;
                non_empty(version, "validated model version")?;
                if self.supporting_evidence.is_empty() {
                    return Err(BridgeError::MissingStructuralSupport);
                }
            }
        }
        Ok(())
    }
}

/// A causal workspace with an explicit evidence-to-node registry.
///
/// Registering evidence never creates an edge. Recording an association never
/// creates an edge. Only `assert_structural_edge` mutates graph structure.
#[derive(Debug, Default)]
pub struct EarthCausalWorkspace {
    dag: CausalDAG,
    bindings: Vec<EvidenceNodeBinding>,
    node_by_evidence_id: HashMap<String, usize>,
    associations: Vec<EvidenceAssociation>,
    structural_claims: Vec<StructuralEdgeClaim>,
}

impl EarthCausalWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dag(&self) -> &CausalDAG {
        &self.dag
    }

    pub fn bindings(&self) -> &[EvidenceNodeBinding] {
        &self.bindings
    }

    pub fn associations(&self) -> &[EvidenceAssociation] {
        &self.associations
    }

    pub fn structural_claims(&self) -> &[StructuralEdgeClaim] {
        &self.structural_claims
    }

    pub fn node_for(&self, evidence_id: &str) -> Option<usize> {
        self.node_by_evidence_id.get(evidence_id).copied()
    }

    pub fn register_evidence(
        &mut self,
        evidence: &EvidenceRef,
        label: &str,
        observed: bool,
    ) -> Result<usize> {
        non_empty(&evidence.id, "evidence id")?;
        non_empty(label, "causal node label")?;
        if self.node_by_evidence_id.contains_key(&evidence.id) {
            return Err(BridgeError::DuplicateEvidenceId(evidence.id.clone()));
        }

        // CausalDAG currently represents discrete variable values. The bridge
        // intentionally does not invent bins here; downstream model construction
        // owns discretization or a future continuous-variable adapter.
        let node_id = self.dag.add_node(label, Vec::new(), observed);
        self.node_by_evidence_id
            .insert(evidence.id.clone(), node_id);
        self.bindings.push(EvidenceNodeBinding {
            evidence_id: evidence.id.clone(),
            evidence_stage: evidence.stage,
            node_id,
            observed,
        });
        Ok(node_id)
    }

    pub fn register_observation(
        &mut self,
        observation: &ObservationEvidence,
        label: &str,
    ) -> Result<usize> {
        let reference = EvidenceRef {
            id: observation.id.0.clone(),
            stage: EvidenceStage::Observation,
        };
        self.register_evidence(&reference, label, true)
    }

    pub fn register_feature(&mut self, feature: &DerivedFeature) -> Result<usize> {
        let reference = EvidenceRef {
            id: feature.id.clone(),
            stage: EvidenceStage::DerivedFeature,
        };
        self.register_evidence(&reference, &feature.name, true)
    }

    pub fn register_hypothesis(&mut self, hypothesis: &Hypothesis) -> Result<usize> {
        let reference = EvidenceRef {
            id: hypothesis.id.clone(),
            stage: EvidenceStage::Hypothesis,
        };
        self.register_evidence(&reference, &hypothesis.statement, false)
    }

    pub fn record_association(&mut self, association: EvidenceAssociation) -> Result<()> {
        self.require_registered(&association.left_evidence_id)?;
        self.require_registered(&association.right_evidence_id)?;
        self.associations.push(association);
        Ok(())
    }

    pub fn assert_structural_edge(&mut self, claim: StructuralEdgeClaim) -> Result<()> {
        claim.validate()?;
        let parent = self.require_registered(&claim.parent_evidence_id)?;
        let child = self.require_registered(&claim.child_evidence_id)?;

        // If child already reaches parent, parent -> child would close a cycle.
        if self.dag.descendants(child).contains(&parent) {
            return Err(BridgeError::StructuralCycle {
                parent: claim.parent_evidence_id.clone(),
                child: claim.child_evidence_id.clone(),
            });
        }

        self.dag.add_edge(parent, child);
        self.structural_claims.push(claim);
        Ok(())
    }

    fn require_registered(&self, evidence_id: &str) -> Result<usize> {
        self.node_by_evidence_id
            .get(evidence_id)
            .copied()
            .ok_or_else(|| BridgeError::MissingEvidenceId(evidence_id.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_earth_observation::{
        ClaimMode, Confidence, HypothesisDomain,
    };

    fn evidence(id: &str, stage: EvidenceStage) -> EvidenceRef {
        EvidenceRef::new(id, stage).unwrap()
    }

    #[test]
    fn observational_association_never_creates_a_causal_edge() {
        let mut workspace = EarthCausalWorkspace::new();
        workspace
            .register_evidence(
                &evidence("rainfall", EvidenceStage::Measurement),
                "rainfall",
                true,
            )
            .unwrap();
        workspace
            .register_evidence(
                &evidence("vegetation", EvidenceStage::DerivedFeature),
                "vegetation health",
                true,
            )
            .unwrap();

        workspace
            .record_association(
                EvidenceAssociation::new(
                    "rainfall",
                    "vegetation",
                    "positive correlation",
                    Some(0.8),
                    vec![evidence("paired-series", EvidenceStage::Measurement)],
                )
                .unwrap(),
            )
            .unwrap();

        assert_eq!(workspace.associations().len(), 1);
        assert!(workspace.dag().edges.is_empty());
    }

    #[test]
    fn explicit_structural_assumption_can_create_an_edge() {
        let mut workspace = EarthCausalWorkspace::new();
        workspace
            .register_evidence(
                &evidence("rainfall", EvidenceStage::Measurement),
                "rainfall",
                true,
            )
            .unwrap();
        workspace
            .register_evidence(
                &evidence("soil-moisture", EvidenceStage::Measurement),
                "soil moisture",
                true,
            )
            .unwrap();

        workspace
            .assert_structural_edge(StructuralEdgeClaim {
                parent_evidence_id: "rainfall".into(),
                child_evidence_id: "soil-moisture".into(),
                basis: StructuralEdgeBasis::DomainAssumption {
                    assumption: "rainfall contributes water to the modeled soil control volume".into(),
                },
                supporting_evidence: vec![],
                assumptions: vec!["no unmodeled irrigation during the interval".into()],
            })
            .unwrap();

        assert_eq!(workspace.dag().edges.len(), 1);
        assert_eq!(workspace.structural_claims().len(), 1);
    }

    #[test]
    fn structural_cycle_is_rejected() {
        let mut workspace = EarthCausalWorkspace::new();
        for id in ["a", "b"] {
            workspace
                .register_evidence(&evidence(id, EvidenceStage::Measurement), id, true)
                .unwrap();
        }

        let claim = |parent: &str, child: &str| StructuralEdgeClaim {
            parent_evidence_id: parent.into(),
            child_evidence_id: child.into(),
            basis: StructuralEdgeBasis::DomainAssumption {
                assumption: format!("fixture structural assumption {parent}->{child}"),
            },
            supporting_evidence: vec![],
            assumptions: vec!["fixture assumption".into()],
        };

        workspace.assert_structural_edge(claim("a", "b")).unwrap();
        assert!(matches!(
            workspace.assert_structural_edge(claim("b", "a")),
            Err(BridgeError::StructuralCycle { .. })
        ));
    }

    #[test]
    fn hypothesis_nodes_are_latent_not_observed_measurements() {
        let mut workspace = EarthCausalWorkspace::new();
        let hypothesis = Hypothesis::new(
            "candidate-cavity",
            "candidate shallow cavity",
            HypothesisDomain::Subsurface {
                estimated_depth_m: Some(2.0),
            },
            ClaimMode::IndirectInference,
            Confidence::new(0.55).unwrap(),
            vec![symthaea_earth_observation::ObservationId::new("obs-1").unwrap()],
        )
        .unwrap();

        let node_id = workspace.register_hypothesis(&hypothesis).unwrap();
        assert!(!workspace.dag().nodes[node_id].is_observed);
        assert_eq!(
            workspace.bindings()[0].evidence_stage,
            EvidenceStage::Hypothesis
        );
    }

    #[test]
    fn intervention_and_external_model_edges_require_support() {
        let intervention = StructuralEdgeClaim {
            parent_evidence_id: "a".into(),
            child_evidence_id: "b".into(),
            basis: StructuralEdgeBasis::ControlledIntervention {
                intervention_id: "trial-1".into(),
            },
            supporting_evidence: vec![],
            assumptions: vec![],
        };
        assert_eq!(
            intervention.validate(),
            Err(BridgeError::MissingStructuralSupport)
        );

        let external = StructuralEdgeClaim {
            parent_evidence_id: "a".into(),
            child_evidence_id: "b".into(),
            basis: StructuralEdgeBasis::ExternalValidatedModel {
                model_id: "hydrology-model".into(),
                version: "1.0".into(),
            },
            supporting_evidence: vec![],
            assumptions: vec![],
        };
        assert_eq!(
            external.validate(),
            Err(BridgeError::MissingStructuralSupport)
        );
    }
}
