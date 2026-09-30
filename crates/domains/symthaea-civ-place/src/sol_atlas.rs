//! Narrow, explicit adapter contract for Sol Atlas PlacePlanV1.
//!
//! The adapter deliberately mirrors only the fields needed to project a
//! PlacePlan into CIV-PLACE. Currentness and qualification inputs are supplied
//! separately; they are never invented from geometry, containment, or source
//! presence alone.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    DependencyEdgeV1, IndependenceWitnessV1, NodeV1, PlaceEvaluationV1, ProjectionV1,
    ServiceQuestionV1, evaluate_service, sha256_hex, canonical_json,
};

pub const SOL_ATLAS_PROFILE: &str = "SOL-PLACE-001D";
pub const SOL_ATLAS_SCHEMA_VERSION: &str = "place-plan-v1";
pub const SOL_ATLAS_CLAIM_CEILING: &str = "plan-declared";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasPlacePlanV1 {
    pub schema_version: String,
    pub plan_id: String,
    pub version: u64,
    pub site: SolAtlasSiteRefV1,
    pub geometry_revision: String,
    pub coordinate_reference_system: String,
    pub source_snapshots: Vec<SolAtlasSourceSnapshotRefV1>,
    pub intent: SolAtlasPlanIntentV1,
    pub elements: Vec<SolAtlasPlanElementV1>,
    pub dependencies: Vec<SolAtlasPlanDependencyV1>,
    pub assumptions: Vec<SolAtlasPlanAssumptionV1>,
    pub review_state: SolAtlasReviewStateV1,
    pub claim_ceiling: SolAtlasClaimCeilingV1,
}

impl SolAtlasPlacePlanV1 {
    pub fn canonicalized(&self) -> Self {
        let mut plan = self.clone();
        plan.site.external_refs.sort_by(|a,b| a.namespace.cmp(&b.namespace).then_with(|| a.external_id.cmp(&b.external_id)));
        plan.source_snapshots.sort_by(|a,b| a.id.cmp(&b.id).then_with(|| a.provider.cmp(&b.provider)).then_with(|| a.profile.cmp(&b.profile)).then_with(|| a.release.cmp(&b.release)));
        plan.intent.non_goals.sort();
        plan.elements.sort_by(|a,b| a.id.cmp(&b.id).then_with(|| a.kind.cmp(&b.kind)).then_with(|| a.parent_id.cmp(&b.parent_id)).then_with(|| a.geometry_ref.cmp(&b.geometry_ref)));
        for element in &mut plan.elements {
            element.external_refs.sort_by(|a,b| a.namespace.cmp(&b.namespace).then_with(|| a.external_id.cmp(&b.external_id)));
        }
        plan.dependencies.sort_by(|a,b| a.id.cmp(&b.id).then_with(|| a.from_id.cmp(&b.from_id)).then_with(|| a.to_id.cmp(&b.to_id)).then_with(|| a.service.cmp(&b.service)).then_with(|| a.required.cmp(&b.required)).then_with(|| a.common_mode_group.cmp(&b.common_mode_group)));
        plan.assumptions.sort_by(|a,b| a.id.cmp(&b.id).then_with(|| a.statement.cmp(&b.statement)).then_with(|| a.status.cmp(&b.status)).then_with(|| a.source_ref.cmp(&b.source_ref)));
        plan
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, AdapterErrorV1> {
        canonical_json(&self.canonicalized()).map_err(AdapterErrorV1::Kernel)
    }

    pub fn validate(&self) -> Result<(), AdapterErrorV1> {
        if self.schema_version != SOL_ATLAS_SCHEMA_VERSION {
            return Err(AdapterErrorV1::UnsupportedSchema(self.schema_version.clone()));
        }
        if self.plan_id.is_empty() || self.site.canonical_id.is_empty() {
            return Err(AdapterErrorV1::InvalidPlan("plan/site identity must not be empty".into()));
        }
        if self.geometry_revision.is_empty() || self.coordinate_reference_system.is_empty() {
            return Err(AdapterErrorV1::InvalidPlan("geometry revision and CRS are required".into()));
        }

        let mut element_ids = HashSet::new();
        for element in &self.elements {
            if element.id.is_empty() || !element_ids.insert(element.id.clone()) {
                return Err(AdapterErrorV1::InvalidPlan(format!("duplicate or empty element id: {}", element.id)));
            }
        }
        for element in &self.elements {
            if let Some(parent) = &element.parent_id {
                if !element_ids.contains(parent) {
                    return Err(AdapterErrorV1::InvalidPlan(format!("missing parent {parent} for {}", element.id)));
                }
            }
        }

        let mut dependency_ids = HashSet::new();
        for dep in &self.dependencies {
            if dep.id.is_empty() || !dependency_ids.insert(dep.id.clone()) {
                return Err(AdapterErrorV1::InvalidPlan(format!("duplicate or empty dependency id: {}", dep.id)));
            }
            if !element_ids.contains(&dep.from_id) || !element_ids.contains(&dep.to_id) {
                return Err(AdapterErrorV1::InvalidPlan(format!("dependency {} references an element outside the plan", dep.id)));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasSiteRefV1 {
    pub canonical_id: String,
    #[serde(default)]
    pub external_refs: Vec<SolAtlasExternalRefV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasSourceSnapshotRefV1 {
    pub id: String,
    pub provider: String,
    pub profile: String,
    pub release: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasExternalRefV1 {
    pub namespace: String,
    pub external_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasPlanIntentV1 {
    pub statement: String,
    #[serde(default)]
    pub non_goals: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum SolAtlasPlanElementKindV1 {
    Site, Parcel, Lot, Room, Home, Building, Block, Neighborhood, District,
    Street, OpenSpace, EnergySystem, WaterSystem, Utility, ServiceNode, Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasPlanElementV1 {
    pub id: String,
    pub kind: SolAtlasPlanElementKindV1,
    pub parent_id: Option<String>,
    pub geometry_ref: Option<String>,
    #[serde(default)]
    pub external_refs: Vec<SolAtlasExternalRefV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasPlanDependencyV1 {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub service: String,
    pub required: bool,
    pub common_mode_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum SolAtlasPlanAssumptionStatusV1 { Supported, Scenario, Unknown, Conflicted }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasPlanAssumptionV1 {
    pub id: String,
    pub statement: String,
    pub status: SolAtlasPlanAssumptionStatusV1,
    pub source_ref: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum SolAtlasReviewStateV1 { Draft, ScenarioReady, UnderReview, Reviewed, Published, Superseded, Withdrawn }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum SolAtlasClaimCeilingV1 { ScenarioOnly, QualifiedUnderAssumptions, NeedsProfessionalReview }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasNodeBindingV1 {
    pub element_id: String,
    pub currentness: String,
    pub projection: String,
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasProjectionBindingV1 {
    pub id: String,
    pub completeness: String,
    pub contradiction: String,
    pub convergence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasServiceBindingV1 {
    pub id: String,
    pub root_element_id: String,
    pub currentness_required: bool,
    pub dependency_discovery: String,
    pub included_dependency_classes: Vec<String>,
    pub required_dependency_classes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolAtlasQualificationInputV1 {
    pub profile: String,
    pub plan: SolAtlasPlacePlanV1,
    pub node_bindings: Vec<SolAtlasNodeBindingV1>,
    pub projections: Vec<SolAtlasProjectionBindingV1>,
    pub services: Vec<SolAtlasServiceBindingV1>,
    #[serde(default)]
    pub independence_witnesses: Vec<IndependenceWitnessV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolAtlasProjectionReceiptV1 {
    pub profile: String,
    pub plan_id: String,
    pub plan_version: u64,
    pub plan_sha256: String,
    pub projected_dependency_ids: Vec<String>,
    pub service_ids: Vec<String>,
    pub kernel_schema_version: String,
    pub claim_ceiling: SolAtlasClaimCeilingV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterErrorV1 {
    UnsupportedSchema(String),
    InvalidPlan(String),
    InvalidQualificationInput(String),
    MissingNodeBinding(String),
    MissingProjectionBinding(String),
    DuplicateBinding(String),
    UnsupportedProfile(String),
    Kernel(crate::KernelError),
}

impl std::fmt::Display for AdapterErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchema(v) => write!(f, "unsupported Sol Atlas schema: {v}"),
            Self::InvalidPlan(v) => write!(f, "invalid Sol Atlas plan: {v}"),
            Self::InvalidQualificationInput(v) => write!(f, "invalid qualification input: {v}"),
            Self::MissingNodeBinding(v) => write!(f, "missing node binding: {v}"),
            Self::MissingProjectionBinding(v) => write!(f, "missing projection binding: {v}"),
            Self::DuplicateBinding(v) => write!(f, "duplicate binding: {v}"),
            Self::UnsupportedProfile(v) => write!(f, "unsupported adapter profile: {v}"),
            Self::Kernel(e) => write!(f, "kernel error: {e}"),
        }
    }
}
impl std::error::Error for AdapterErrorV1 {}
impl From<crate::KernelError> for AdapterErrorV1 { fn from(e: crate::KernelError) -> Self { Self::Kernel(e) } }

pub fn project(input: &SolAtlasQualificationInputV1) -> Result<(Vec<NodeV1>, Vec<DependencyEdgeV1>, Vec<ProjectionV1>, Vec<ServiceQuestionV1>, SolAtlasProjectionReceiptV1), AdapterErrorV1> {
    if input.profile != SOL_ATLAS_PROFILE {
        return Err(AdapterErrorV1::UnsupportedProfile(input.profile.clone()));
    }
    input.plan.validate()?;

    let mut nodes = BTreeMap::<String, NodeV1>::new();
    for b in &input.node_bindings {
        if nodes.insert(b.element_id.clone(), NodeV1 {
            id: b.element_id.clone(),
            currentness: b.currentness.clone(),
            projection: b.projection.clone(),
            state: b.state.clone(),
        }).is_some() {
            return Err(AdapterErrorV1::DuplicateBinding(b.element_id.clone()));
        }
    }
    let plan_ids = input.plan.elements.iter().map(|e| e.id.as_str()).collect::<BTreeSet<_>>();
    for id in &plan_ids {
        if !nodes.contains_key(*id) {
            return Err(AdapterErrorV1::MissingNodeBinding((*id).into()));
        }
    }

    let mut projections = BTreeMap::<String, ProjectionV1>::new();
    for p in &input.projections {
        if projections.insert(p.id.clone(), ProjectionV1 {
            id: p.id.clone(),
            completeness: p.completeness.clone(),
            contradiction: p.contradiction.clone(),
            convergence: p.convergence.clone(),
        }).is_some() {
            return Err(AdapterErrorV1::DuplicateBinding(p.id.clone()));
        }
    }
    for node in nodes.values() {
        if !projections.contains_key(&node.projection) {
            return Err(AdapterErrorV1::MissingProjectionBinding(node.projection.clone()));
        }
    }

    let mut deps = Vec::new();
    for d in &input.plan.dependencies {
        // PlacePlan: from = dependent, to = dependency/provider.
        // CIV-PLACE: source = dependency/provider, target = dependent.
        deps.push(DependencyEdgeV1 {
            id: d.id.clone(),
            source: d.to_id.clone(),
            target: d.from_id.clone(),
            dependency_class: d.service.clone(),
            required: d.required,
            common_mode: d.common_mode_group.clone(),
        });
    }

    deps.sort_by(|a,b| a.id.cmp(&b.id));
    
    let mut services = Vec::new();
    for s in &input.services {
        if !plan_ids.contains(s.root_element_id.as_str()) {
            return Err(AdapterErrorV1::InvalidQualificationInput(format!("service {} root is not in plan", s.id)));
        }
        let mut included = s.included_dependency_classes.clone();
        included.sort();
        included.dedup();
        let mut required = s.required_dependency_classes.clone();
        required.sort();
        required.dedup();
        services.push(ServiceQuestionV1 {
            id: s.id.clone(),
            currentness_required: s.currentness_required,
            dependency_discovery: s.dependency_discovery.clone(),
            included_dependency_classes: included,
            required_dependency_classes: required,
            root: s.root_element_id.clone(),
        });
    }
    services.sort_by(|a,b| a.id.cmp(&b.id));

    let plan_sha256 = sha256_hex(&input.plan.canonical_json()?);
    let receipt = SolAtlasProjectionReceiptV1 {
        profile: SOL_ATLAS_PROFILE.into(),
        plan_id: input.plan.plan_id.clone(),
        plan_version: input.plan.version,
        plan_sha256,
        projected_dependency_ids: deps.iter().map(|d| d.id.clone()).collect(),
        service_ids: services.iter().map(|s| s.id.clone()).collect(),
        kernel_schema_version: crate::SCHEMA_VERSION.into(),
        claim_ceiling: input.plan.claim_ceiling,
    };

    Ok((nodes.into_values().collect(), deps, projections.into_values().collect(), services, receipt))
}

pub fn evaluate(input: &SolAtlasQualificationInputV1) -> Result<BTreeMap<String, PlaceEvaluationV1>, AdapterErrorV1> {
    let (nodes, deps, projections, services, _) = project(input)?;
    let nodes = nodes.into_iter().collect::<BTreeMap<_,_>>();
    let projections = projections.into_iter().map(|p|(p.id.clone(),p)).collect::<BTreeMap<_,_>>();
    services.iter().map(|service| evaluate_service(service, &nodes, &deps, &projections).map(|e|(service.id.clone(),e)).map_err(AdapterErrorV1::Kernel)).collect()
}

pub fn projection_receipt(input: &SolAtlasQualificationInputV1) -> Result<SolAtlasProjectionReceiptV1, AdapterErrorV1> {
    Ok(project(input)?.4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> SolAtlasQualificationInputV1 {
        serde_json::from_str(include_str!("../../../../docs/engineering/fixtures/sol-atlas-place-001d.json")).unwrap()
    }

    #[test]
    fn plan_canonicalization_is_order_invariant() {
        let input = fixture();
        let mut reversed = input.clone();
        reversed.plan.elements.reverse();
        reversed.plan.dependencies.reverse();
        assert_eq!(input.plan.canonical_json().unwrap(), reversed.plan.canonical_json().unwrap());
    }

    #[test]
    fn dependency_direction_is_inverted_exactly_once() {
        let input = fixture();
        let (_, deps, _, _, _) = project(&input).unwrap();
        let transformer = deps.iter().find(|d| d.id == "dep:home-01:transformer").unwrap();
        assert_eq!(transformer.source, "service:transformer-t4");
        assert_eq!(transformer.target, "home-01");
    }

    #[test]
    fn qualification_requires_explicit_node_bindings() {
        let mut input = fixture();
        input.node_bindings.pop();
        assert!(matches!(project(&input), Err(AdapterErrorV1::MissingNodeBinding(_))));
    }

    #[test]
    fn qualification_requires_explicit_projection_bindings() {
        let mut input = fixture();
        input.projections.clear();
        assert!(matches!(project(&input), Err(AdapterErrorV1::MissingProjectionBinding(_))));
    }

    #[test]
    fn plan_change_changes_identity() {
        let mut input = fixture();
        let before = projection_receipt(&input).unwrap().plan_sha256;
        input.plan.geometry_revision = "geom-r4".into();
        let after = projection_receipt(&input).unwrap().plan_sha256;
        assert_ne!(before, after);
    }

    #[test]
    fn nominal_home_evaluates_full_service_under_declared_inputs() {
        let input = fixture();
        let evals = evaluate(&input).unwrap();
        assert_eq!(evals["svc:home-01:electric"].status, "FullService");
    }
}
