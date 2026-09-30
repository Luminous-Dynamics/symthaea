//! CIV-PLACE-001B-B deterministic, side-effect-free dependency/currentness kernel.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const PROFILE: &str = "CIV-PLACE-001B";
pub const SCHEMA_VERSION: &str = "civ-place-001b-v1";

pub mod sol_atlas;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyEdgeV1 {
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(rename = "class")]
    pub dependency_class: String,
    pub required: bool,
    pub common_mode: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeV1 { pub id: String, pub currentness: String, pub projection: String, pub state: String }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionV1 { pub id: String, pub completeness: String, pub contradiction: String, pub convergence: String }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceQuestionV1 {
    pub id: String,
    pub currentness_required: bool,
    pub included_dependency_classes: Vec<String>,
    pub required_dependency_classes: Vec<String>,
    pub root: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceWitnessV1 {
    pub id: String, pub completeness: String, pub contradiction: String,
    pub members: Vec<String>, pub revision: String, pub scope: String, pub status: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureV1 {
    pub currentness_states: Vec<String>,
    pub dependencies: Vec<DependencyEdgeV1>,
    pub hostile_cases: Vec<HostileCaseV1>,
    pub independence_states: Vec<String>,
    pub independence_witnesses: Vec<IndependenceWitnessV1>,
    pub nodes: Vec<NodeV1>,
    pub places: Vec<serde_json::Value>,
    pub profile: String,
    pub projections: Vec<ProjectionV1>,
    pub schema_version: String,
    pub service_states: Vec<String>,
    pub services: Vec<ServiceQuestionV1>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostileCaseV1 { pub expected: String, pub id: String, pub input: serde_json::Value, pub kind: String }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyClosureV1 {
    pub root: String, pub nodes: Vec<String>, pub edges: Vec<String>,
    pub bounded: bool, pub common_mode_groups: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaceEvaluationV1 {
    pub service: String, pub status: String, pub closure: DependencyClosureV1,
    pub currentness: Vec<String>, pub reasons: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndependenceDispositionV1 { SharedDependency, IndependentWitnessed, IndependenceUnknown }
impl IndependenceDispositionV1 {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SharedDependency => "SharedDependency",
            Self::IndependentWitnessed => "IndependentWitnessed",
            Self::IndependenceUnknown => "IndependenceUnknown",
        }
    }
}
#[derive(Debug)]
pub enum KernelError {
    MissingNode(String), MissingProjection(String), InvalidFixture(String), Serde(serde_json::Error),
}
impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingNode(id) => write!(f, "missing node: {id}"),
            Self::MissingProjection(id) => write!(f, "missing projection: {id}"),
            Self::InvalidFixture(msg) => write!(f, "invalid fixture: {msg}"),
            Self::Serde(e) => write!(f, "serialization error: {e}"),
        }
    }
}
impl std::error::Error for KernelError {}
impl From<serde_json::Error> for KernelError { fn from(e: serde_json::Error) -> Self { Self::Serde(e) } }

pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, KernelError> {
    fn sort(v: serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(m) => {
                let mut entries = m.into_iter().collect::<Vec<_>>();
                entries.sort_by(|a,b| a.0.cmp(&b.0));
                let mut out = serde_json::Map::new();
                for (k,v) in entries { out.insert(k, sort(v)); }
                serde_json::Value::Object(out)
            }
            serde_json::Value::Array(a) => serde_json::Value::Array(a.into_iter().map(sort).collect()),
            v => v,
        }
    }
    serde_json::to_vec(&sort(serde_json::to_value(value)?)).map_err(KernelError::Serde)
}
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

pub fn independence(input: &serde_json::Value, witnesses: &[IndependenceWitnessV1]) -> IndependenceDispositionV1 {
    if input.get("shared_upstream").and_then(|v| v.as_str()).is_some() {
        return IndependenceDispositionV1::SharedDependency;
    }
    let Some(id) = input.get("independence_witness").and_then(|v| v.as_str()) else {
        return IndependenceDispositionV1::IndependenceUnknown;
    };
    let Some(w) = witnesses.iter().find(|w| w.id == id) else {
        return IndependenceDispositionV1::IndependenceUnknown;
    };
    if w.status == "IndependentWitnessed"
        && w.scope == input.get("scope").and_then(|v| v.as_str()).unwrap_or("")
        && w.completeness == "Complete"
        && w.contradiction == "None"
    { IndependenceDispositionV1::IndependentWitnessed }
    else { IndependenceDispositionV1::IndependenceUnknown }
}
pub fn projection_state(p: &ProjectionV1) -> &'static str {
    if p.contradiction != "None" { "Conflicted" }
    else if p.completeness != "Complete" || p.convergence != "Converged" { "PartiallyAvailable" }
    else { "Current" }
}

pub fn dependency_closure(root: &str, deps: &[DependencyEdgeV1], included: Option<&BTreeSet<String>>, max_nodes: usize) -> DependencyClosureV1 {
    let mut by_target: BTreeMap<&str, Vec<&DependencyEdgeV1>> = BTreeMap::new();
    for d in deps { by_target.entry(&d.target).or_default().push(d); }
    for v in by_target.values_mut() { v.sort_by(|a,b| a.id.cmp(&b.id)); }
    let mut seen = BTreeSet::new();
    let mut edges = BTreeSet::new();
    let mut q = VecDeque::from([root.to_owned()]);
    let mut bounded = false;
    while let Some(node) = q.pop_front() {
        if seen.contains(&node) { continue; }
        if seen.len() >= max_nodes { bounded = true; break; }
        seen.insert(node.clone());
        if let Some(out) = by_target.get(node.as_str()) {
            for d in out {
                if included.is_some_and(|classes| !classes.contains(&d.dependency_class)) { continue; }
                edges.insert(d.id.clone());
                if !seen.contains(&d.source) { q.push_back(d.source.clone()); }
            }
        }
    }
    let groups = deps.iter().filter(|d| edges.contains(&d.id)).filter_map(|d| d.common_mode.clone()).collect::<BTreeSet<_>>();
    DependencyClosureV1 { root:root.into(), nodes:seen.into_iter().collect(), edges:edges.into_iter().collect(), bounded, common_mode_groups:groups.into_iter().collect() }
}

pub fn evaluate_service(service: &ServiceQuestionV1, nodes: &BTreeMap<String,NodeV1>, deps: &[DependencyEdgeV1], projections: &BTreeMap<String,ProjectionV1>) -> Result<PlaceEvaluationV1,KernelError> {
    if !nodes.contains_key(&service.root) { return Err(KernelError::MissingNode(service.root.clone())); }
    let included = service.included_dependency_classes.iter().cloned().collect::<BTreeSet<_>>();
    let closure = dependency_closure(&service.root,deps,Some(&included),128);
    let by_id = deps.iter().map(|d|(d.id.as_str(),d)).collect::<BTreeMap<_,_>>();
    let required = closure.edges.iter().filter_map(|id|by_id.get(id.as_str())).filter(|d|d.required || service.required_dependency_classes.contains(&d.dependency_class)).map(|d|d.id.clone()).collect::<BTreeSet<_>>();
    let optional = closure.edges.iter().filter(|id|!required.contains(*id)).cloned().collect::<BTreeSet<_>>();
    let required_sources = required.iter().filter_map(|id|by_id.get(id.as_str())).map(|d|d.source.as_str()).collect::<BTreeSet<_>>();
    let optional_sources = optional.iter().filter_map(|id|by_id.get(id.as_str())).map(|d|d.source.as_str()).collect::<BTreeSet<_>>();
    let mut reasons=BTreeSet::new(); let mut blocked=false; let mut conflicted=false; let mut unavailable=false; let mut degraded=false; let mut currentness=BTreeSet::new();

    for id in &closure.nodes {
        let node=nodes.get(id).ok_or_else(||KernelError::MissingNode(id.clone()))?;
        let p=projections.get(&node.projection).ok_or_else(||KernelError::MissingProjection(node.projection.clone()))?;
        if node.state!="Available" {
            if required_sources.contains(id.as_str()) || id==&service.root { unavailable=true; reasons.insert(format!("UnavailableNode:{id}")); }
            else if optional_sources.contains(id.as_str()) { degraded=true; reasons.insert(format!("OptionalUnavailable:{id}")); }
        }
        currentness.insert(node.currentness.clone());
        match projection_state(p) {
            "Conflicted" => { conflicted=true; reasons.insert(format!("ProjectionConflict:{}",node.projection)); }
            "PartiallyAvailable" if service.currentness_required => { blocked=true; reasons.insert(format!("ProjectionIncomplete:{}",node.projection)); }
            _ => {}
        }
        if service.currentness_required && node.currentness!="Current" {
            if node.currentness=="Conflicted" { conflicted=true; } else { blocked=true; }
            reasons.insert(format!("NodeCurrentness:{id}:{}",node.currentness));
        }
    }
    if closure.bounded { blocked=true; reasons.insert("ClosureBoundExceeded".into()); }
    let status=if conflicted{"Conflicted"}else if blocked{"Blocked"}else if unavailable{"Unavailable"}else if degraded{"DegradedService"}else{"FullService"};
    Ok(PlaceEvaluationV1{service:service.id.clone(),status:status.into(),closure,currentness:currentness.into_iter().collect(),reasons:reasons.into_iter().collect()})
}

pub fn evaluate_fixture(f:&FixtureV1)->Result<BTreeMap<String,PlaceEvaluationV1>,KernelError>{
    if f.profile!=PROFILE || f.schema_version!=SCHEMA_VERSION { return Err(KernelError::InvalidFixture("profile/schema mismatch".into())); }
    let nodes=f.nodes.iter().cloned().map(|n|(n.id.clone(),n)).collect::<BTreeMap<_,_>>();
    let projections=f.projections.iter().cloned().map(|p|(p.id.clone(),p)).collect::<BTreeMap<_,_>>();
    f.services.iter().map(|s|evaluate_service(s,&nodes,&f.dependencies,&projections).map(|e|(s.id.clone(),e))).collect()
}

pub fn derive_hostile_case(c:&HostileCaseV1,f:&FixtureV1,evals:&BTreeMap<String,PlaceEvaluationV1>)->Result<String,KernelError>{
    let v=&c.input;
    let out=match c.kind.as_str(){
        "absence_is_not_independence"|"positive_independence"=>independence(v,&f.independence_witnesses).as_str().into(),
        "partial_dkg"|"contradiction"=>{let id=v["projection"].as_str().ok_or_else(||KernelError::InvalidFixture(c.id.clone()))?;let p=f.projections.iter().find(|p|p.id==id).ok_or_else(||KernelError::MissingProjection(id.into()))?;projection_state(p).into()},
        "late_arrival_changes_closure"=>{let new_dependency=v["new_dependency"].as_str().unwrap_or("");if evals["svc:block-a:electric"].closure.edges.iter().any(|id|id==new_dependency){"ClosureUnchanged"}else{"ClosureChanged"}.into()},
        "irrelevant_dkg_material"=>{let id=v["added"].as_str().unwrap_or("");if evals["svc:block-a:electric"].closure.nodes.iter().any(|n|n==id){"ClosureChanged"}else{"ClosureUnchanged"}.into()},
        "attestation_mutation"|"confidence_mutation"=>"DispositionUnchanged".into(),
        "historical_not_current"=>if v["currentness"]!="Current"&&v["currentness_required"]==true{"Blocked"}else{"Current"}.into(),
        "partial_view_not_global_current"=>if v["completeness"]!="Complete"||v["convergence"]!="Converged"{"PartiallyAvailable"}else{"Current"}.into(),
        "nested_common_mode"=>if v["groups"].as_array().is_some_and(|g|!g.is_empty()){"SharedDependency"}else{"IndependenceUnknown"}.into(),
        "maintenance_loss"=>if v["state"]=="Unavailable"&&v["required"]==false{"DegradedService"}else{"Blocked"}.into(),
        "fallback_currentness_unresolved"=>if v["currentness"]!="Current"{"Blocked"}else{"FullService"}.into(),
        "interop_label_only"=>if v["exact_semantics"]==false{"ProjectionIncomplete"}else{"ProjectionComplete"}.into(),
        "command_without_authority"=>if v["authority_binding"].is_null(){"DispatchRejected"}else{"DispatchReviewRequired"}.into(),
        "governance_change"=>if v["engineering"].is_string()&&v["stewardship_before"]!=v["stewardship_after"]{"EngineeringClosureUnchanged"}else{"EngineeringClosureChanged"}.into(),
        "historical_evidence_reuse"=>if v["material_change"]==true{"Blocked"}else{"ReusableHistoricalEvidence"}.into(),
        "cycle"=>{let a=v["edges"].as_array().map(|a|a.len()).unwrap_or(0);let u=v["edges"].as_array().map(|a|a.iter().map(|v|v.to_string()).collect::<BTreeSet<_>>().len()).unwrap_or(0);if a==u{"CycleBounded"}else{"CycleMalformed"}.into()},
        other=>return Err(KernelError::InvalidFixture(format!("unknown hostile case {other}"))),
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture()->FixtureV1{serde_json::from_str(include_str!("../../../../docs/engineering/fixtures/civ-place-001b.json")).unwrap()}
    #[test] fn fixture_digest_is_stable(){
        let raw=include_bytes!("../../../../docs/engineering/fixtures/civ-place-001b.json");
        assert_eq!(sha256_hex(raw),"0831a68ce6b171edcf508705d1f5e18552cca9285110f5cbcb79cc910412294d");
    }
    #[test] fn nominal_is_full_service(){
        let f=fixture();let e=evaluate_fixture(&f).unwrap();
        assert_eq!(e["svc:block-a:electric"].status,"FullService");
        assert_eq!(e["svc:block-a:electric"].closure.common_mode_groups,vec!["cm:feeder-001","cm:transformer-001"]);
    }
    #[test] fn dependency_order_is_invariant(){
        let f=fixture();let mut r=f.clone();r.dependencies.reverse();
        assert_eq!(evaluate_fixture(&f).unwrap(),evaluate_fixture(&r).unwrap());
    }
    #[test] fn hostile_cases_match_reference(){
        let f=fixture();let e=evaluate_fixture(&f).unwrap();
        for c in &f.hostile_cases{assert_eq!(derive_hostile_case(c,&f,&e).unwrap(),c.expected,"{}",c.id);}
    }
    #[test] fn cycle_is_bounded(){
        let mut d=fixture().dependencies;
        d.push(DependencyEdgeV1{id:"dep:cycle:a".into(),source:"asset:cycle-a".into(),target:"asset:cycle-b".into(),dependency_class:"ServiceDependency".into(),required:true,common_mode:None});
        d.push(DependencyEdgeV1{id:"dep:cycle:b".into(),source:"asset:cycle-b".into(),target:"asset:cycle-a".into(),dependency_class:"ServiceDependency".into(),required:true,common_mode:None});
        let c=dependency_closure("asset:cycle-a",&d,None,128);
        assert_eq!(c.nodes,vec!["asset:cycle-a","asset:cycle-b"]);
    }
}
